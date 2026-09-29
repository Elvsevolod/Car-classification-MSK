"""Frozen, query-independent local OSNet pilot; no learned head or label inputs.

The only experiment-selected quantities are a predeclared scorer, K and mixing
weight, selected on primary and frozen before alternate. This is not an OOF
trained verifier: adding a learned head would require separate OOF training.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

import evaluate as official
from training.retrieval_policy import frames


SCORERS = ("mean_best", "partial")


def extract_local_tokens(encoder, batch, grid=(4, 4)):
    """Capture conv5 before global pooling without modifying the frozen model.

    Caller owns preprocessing, checkpoint identity, device, batching and cache
    fingerprints. Each image's pooled spatial tokens are normalized separately.
    All-zero tokens are valid no-evidence tokens, never fabricated embeddings.
    """
    if any(module.training for module in encoder.modules()):
        raise ValueError("Local extraction requires an entirely eval-mode encoder")
    if not hasattr(encoder, "conv5") or not hasattr(encoder, "global_pool"):
        raise ValueError("Expected OSNet conv5 and global_pool modules")
    if (not isinstance(batch, torch.Tensor) or batch.ndim != 4 or len(batch) < 1
            or not torch.is_floating_point(batch) or not torch.isfinite(batch).all()):
        raise ValueError("Expected a finite nonempty NCHW floating-point batch")
    if len(grid) != 2 or any(type(n) is not int or n < 1 for n in grid):
        raise ValueError("grid must contain two positive integers")
    captured = []

    def capture(_module, _inputs, output):
        if output.ndim != 4 or output.shape[0] != len(batch):
            raise ValueError("Unexpected OSNet spatial feature shape")
        if any(size > available for size, available in zip(grid, output.shape[2:])):
            raise ValueError("Local grid must not exceed the feature-map size")
        captured.append(F.adaptive_avg_pool2d(output.float(), grid).flatten(2).transpose(1, 2))

    hook = encoder.conv5.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            encoder(batch)
    finally:
        hook.remove()
    if len(captured) != 1 or not torch.isfinite(captured[0]).all():
        raise ValueError("Need exactly one finite conv5 feature map")
    return _normalize_tokens(captured[0].cpu().numpy())


def _normalize_tokens(tokens):
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim not in (2, 3) or any(n < 1 for n in values.shape) or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty token matrices")
    # Float64 norm avoids float32 overflow for externally supplied finite caches.
    norm = np.linalg.norm(values.astype(np.float64), axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 0)


def local_pair_scores(query_tokens, gallery_tokens, scorer="partial"):
    """One query versus candidate images; labels and other queries are absent.

    Symmetric mean-best supports all visible tokens; partial averages the best
    half in each direction to tolerate non-overlapping views. Neither assumes a
    planar geometry. A low score gives no bonus, not an explicit mismatch penalty.
    These are similarities, not calibrated probabilities.
    """
    if scorer not in SCORERS:
        raise ValueError("Unknown frozen local scorer")
    query, gallery = _normalize_tokens(query_tokens), _normalize_tokens(gallery_tokens)
    if query.ndim != 2 or gallery.ndim != 3 or query.shape[-1] != gallery.shape[-1]:
        raise ValueError("Expected one TxD query and KxSxD candidates")
    similarities = np.clip(np.einsum("td,ksd->kts", query, gallery), 0, 1)
    query_valid = np.linalg.norm(query, axis=-1) > 0
    gallery_valid = np.linalg.norm(gallery, axis=-1) > 0

    def support(values):
        if not len(values):
            return 0.0
        if scorer == "partial":
            values = np.sort(values)[-math.ceil(len(values) / 2):]
        return float(np.mean(values))

    scores = []
    for index, pairs in enumerate(similarities):
        if not query_valid.any() or not gallery_valid[index].any():
            scores.append(0.0)
            continue
        left = pairs.max(axis=1)[query_valid]
        right = pairs.max(axis=0)[gallery_valid[index]]
        scores.append((support(left) + support(right)) / 2)
    return np.asarray(scores, dtype=np.float32)


def _orders(order, queries, gallery):
    order = np.asarray(order)
    if (order.shape != (queries, gallery) or not np.issubdtype(order.dtype, np.integer)
            or not np.all(np.sort(order, axis=1) == np.arange(gallery))):
        raise ValueError("Each order must be a complete permutation of gallery indices")
    return order


def rerank_topk(base_scores, base_order, query_tokens, gallery_tokens, *, top_k, weight,
                scorer="partial"):
    """Bounded positive bonus inside the original top-K; tail is untouched.

    base_scores are real higher-is-better cosine or negative graph distances,
    indexed in gallery order, not rank positions. The returned mixed scores are
    diagnostic only: the returned order is authoritative because its candidate
    pool is deliberately fixed. Confidence/refusal stays a separate policy.
    """
    scores = np.asarray(base_scores, dtype=np.float64)
    queries, gallery = np.asarray(query_tokens), np.asarray(gallery_tokens)
    if (queries.ndim != 3 or gallery.ndim != 3 or not len(queries) or not len(gallery)
            or scores.shape != (len(queries), len(gallery)) or not np.isfinite(scores).all()):
        raise ValueError("Mismatched/non-finite query, gallery or score matrices")
    if type(top_k) is not int or top_k < 1 or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Use positive integer K and a finite mixing weight in [0, 1]")
    if scorer not in SCORERS:
        raise ValueError("Unknown frozen local scorer")
    order = _orders(base_order, len(queries), len(gallery))
    if np.any(np.diff(np.take_along_axis(scores, order, axis=1), axis=1) > 1e-6):
        raise ValueError("base_order must rank the supplied base_scores descending")
    # Validate the entire cache, including features outside the selected pool.
    queries, gallery = _normalize_tokens(queries), _normalize_tokens(gallery)
    if queries.shape[-1] != gallery.shape[-1]:
        raise ValueError("Query/gallery token dimensions differ")
    count = min(top_k, len(gallery))
    local = np.empty((len(queries), count), dtype=np.float32)
    for index, query in enumerate(queries):
        pool = order[index, :count]
        local[index] = local_pair_scores(query, gallery[pool], scorer)
    return {**mix_topk_scores(scores, order, local, top_k=top_k, weight=weight), "scorer": scorer}


def mix_topk_scores(base_scores, base_order, local_scores, *, top_k, weight):
    """Reuse local scores for the first Kmax positions across a frozen K/weight grid.

    local_scores[:, j] belongs to base_order[:, j], NOT gallery index j. Kmax
    may exceed top_k; provenance and cache hashes remain the caller's obligation.
    """
    scores = np.asarray(base_scores, dtype=np.float64)
    local = np.asarray(local_scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite nonempty base score matrices")
    if type(top_k) is not int or top_k < 1 or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Use positive integer K and a finite mixing weight in [0, 1]")
    order = _orders(base_order, *scores.shape)
    count = min(top_k, scores.shape[1])
    if (local.ndim != 2 or local.shape[0] != len(scores) or not count <= local.shape[1] <= scores.shape[1]
            or not np.isfinite(local).all() or np.any((local < 0) | (local > 1))):
        raise ValueError("Need finite [0, 1] local scores aligned to the base top-Kmax")
    if np.any(np.diff(np.take_along_axis(scores, order, axis=1), axis=1) > 1e-6):
        raise ValueError("base_order must rank the supplied base_scores descending")
    result, mixed = order.copy(), scores.copy()
    for index in range(len(scores)):
        pool = order[index, :count]
        mixed[index, pool] += float(weight) * local[index, :count]
        if weight != 0:
            result[index, :count] = pool[np.argsort(-mixed[index, pool], kind="stable")]
    return {"order": result, "scores": mixed, "local_scores": local[:, :count].copy(),
            "pool_indices": order[:, :count].copy(), "top_k": count, "weight": float(weight)}


def topk_oracle_diagnostics(queries, gallery, base_order, *, ks=(20, 50), comparison_order=None):
    """Label-only diagnostics, never called by the inference scorer.

    Real AP uses the physical exported first ten, then official junk filtering.
    Oracle can rearrange the full fixed K pool ideally; its denominator counts
    ALL valid gallery positives, not only hits within that pool. Unknown queries
    are excluded from both averages, matching the official ranking metric.
    """
    q, g = frames(queries, gallery)
    if not len(q) or not len(g) or not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("Need nonempty protocols and positive integer K values")
    if len(set(ks)) != len(ks):
        raise ValueError("Duplicate K values")
    base = _orders(base_order, len(q), len(g))
    comparison = None if comparison_order is None else _orders(comparison_order, len(q), len(g))
    gids = g.index.tolist()
    vids, cams = g.vehicle_id.to_dict(), g.camera_id.to_dict()
    entries = {}
    transitions = {"ranking_wrong_to_correct": 0, "ranking_correct_to_wrong": 0,
                   "candidate_wrong_to_correct": 0, "candidate_correct_to_wrong": 0}
    for index, (qid, row) in enumerate(q.iterrows()):
        positive = ((g.vehicle_id == row.vehicle_id) & (g.camera_id != row.camera_id)).to_numpy()
        n_pos = int(positive.sum())
        entry = {"known": bool(n_pos), "valid_gallery_positives": n_pos, "pools": {}}
        for k in ks:
            r = int(positive[base[index, :k]].sum())
            entry["pools"][str(k)] = {"effective_k": min(k, len(g)), "valid_positives": r,
                                      "any_positive": bool(r),
                                      "AP10_oracle": min(r, 10) / min(n_pos, 10) if n_pos else None}
        for name, order in (("baseline", base), ("comparison", comparison)):
            if order is None:
                continue
            selected = [gids[int(j)] for j in order[index, :10]]
            clean = official.strip_junk(selected, row, vids, cams)
            metric = official.ranking_metrics(q.loc[[qid]], g, {qid: selected})
            entry[name] = {"AP10": metric["mAP@10"] if n_pos else None,
                           "ranking_top1_correct": bool(n_pos and clean and vids[clean[0]] == row.vehicle_id),
                           "candidate_top1_correct": bool(n_pos and vids[selected[0]] == row.vehicle_id),
                           "physical_top1": selected[0]}
        if comparison is not None and n_pos:
            for kind in ("ranking", "candidate"):
                before = entry["baseline"][f"{kind}_top1_correct"]
                after = entry["comparison"][f"{kind}_top1_correct"]
                transitions[f"{kind}_wrong_to_correct"] += int(not before and after)
                transitions[f"{kind}_correct_to_wrong"] += int(before and not after)
        entries[qid] = entry
    known = [entry for entry in entries.values() if entry["known"]]
    summary = {"known_queries": len(known), "unknown_queries": len(entries) - len(known), "pools": {}}
    for k in ks:
        summary["pools"][str(k)] = {
            "any_positive_hit_at_K": float(np.mean([x["pools"][str(k)]["any_positive"] for x in known])) if known else None,
            "mean_AP10_oracle": float(np.mean([x["pools"][str(k)]["AP10_oracle"] for x in known])) if known else None}
    for name in ("baseline", "comparison"):
        if name == "comparison" and comparison is None:
            continue
        summary[f"{name}_mAP10"] = float(np.mean([x[name]["AP10"] for x in known])) if known else None
    return {"summary": summary, "per_query": entries, "top1_transitions": transitions,
            "label_use": "diagnostics only; never scorer inputs",
            "ap_semantics": "export ten first, official same-ID-and-camera junk filter, all-gallery denominator"}
