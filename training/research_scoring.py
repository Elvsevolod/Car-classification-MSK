"""Image-only G1/G2, unchanged organizer top-10 metric and fold-matched scoring."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import numpy as np
import evaluate as official
from backend.core import normalize
from backend.rerank import KReciprocalReranker
from training import retrieval_policy as policy


def mix(blocks, weights):
    if len(blocks) != len(weights) or any(w < 0 for w in weights) or not np.isclose(sum(weights), 1):
        raise ValueError("Invalid cosine mixture")
    return normalize(np.concatenate([normalize(b)*np.float32(w**.5) for b, w in zip(blocks, weights) if w], axis=1))


def control_vectors(bank):
    if bank.ndim != 2 or bank.shape[1] != 2048:
        raise ValueError("Expected B0/MVP 512 + R1 equal3 1536")
    return mix([bank[:, :512], bank[:, 512:]], [.5, .5])


class FixedGraph:
    def __init__(self, gallery_bank):
        values = control_vectors(gallery_bank)
        self.graph = KReciprocalReranker(values, min(20, len(values)-1), min(3, len(values)))

    def components(self, query_bank):
        pairs = [self.graph.components(q) for q in normalize(control_vectors(query_bank))]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def fuse_distances(raw, jaccard, expert_query, expert_gallery, mode, weight):
    if mode not in {"G1", "G2"} or not 0 <= weight <= 1:
        raise ValueError("Unknown fixed-graph policy")
    baseline = (np.float32(.5)*jaccard + np.float32(.5)*raw).astype(np.float32)
    if weight == 0:
        return baseline  # Bit-exact control; avoid needless rounding.
    expert = (1 - np.clip(normalize(expert_query) @ normalize(expert_gallery).T, -1, 1)) / np.float32(2)
    if expert.shape != raw.shape or jaccard.shape != raw.shape:
        raise ValueError("Mismatched query/gallery order")
    a = np.float32(weight)
    return ((1-a)*baseline + a*expert if mode == "G1" else
            np.float32(.5)*jaccard + np.float32(.5)*((1-a)*raw+a*expert)).astype(np.float32)


def ranking_report(query, gallery, order):
    q, g = policy.frames(query, gallery)
    if order.shape != (len(query), len(gallery)) or len(gallery) < 10:
        raise ValueError("Need full gallery order and at least ten images")
    ids = [r["image_id"] for r in gallery]
    predictions = {r["image_id"]: [ids[int(j)] for j in indices[:10]] for r, indices in zip(query, order)}
    metrics = official.ranking_metrics(q, g, predictions)
    per_query = {}
    for qid, row in q.iterrows():
        result = official.ranking_metrics(q.loc[[qid]], g, {qid: predictions[qid]})
        per_query[qid] = {"vehicle_id": int(row.vehicle_id), "known": bool(result["n_scored"]),
                          "ap": result["mAP@10"] if result["n_scored"] else None, "top10": predictions[qid]}
    return {"metrics": metrics, "per_query": per_query}


def score_features(features, rows, draws, control=None):
    features = normalize(features)
    if len(features) != len(rows):
        raise ValueError("Feature/ID order mismatch")
    index = {r["image_id"]: i for i, r in enumerate(rows)}
    reports = {}
    for name, p in draws.items():
        qi, gi = ([index[i] for i in p[key]] for key in ("query_ids", "gallery_ids"))
        query, gallery = ([rows[i] for i in indices] for indices in (qi, gi))
        q, g = features[qi], features[gi]
        ranks = policy.rank_vectors(q, g, "less_graph")
        current = {"raw": ranking_report(query, gallery, ranks["raw_order"]),
                   "graph": ranking_report(query, gallery, ranks["order"])}
        if control is not None:
            graph = FixedGraph(control[gi])
            raw, jac = graph.components(control[qi])
            current["control"] = ranking_report(query, gallery, np.argsort(.5*jac+.5*raw, axis=1, kind="stable"))
            combined = mix([control_vectors(control), features], [.9, .1])
            pre = policy.rank_vectors(combined[qi], combined[gi], "legacy")
            current["pre_graph_10"] = ranking_report(query, gallery, pre["order"])
            for mode in ("G1", "G2"):
                distances = fuse_distances(raw, jac, q, g, mode, .1)
                current[mode] = ranking_report(query, gallery, np.argsort(distances, axis=1, kind="stable"))
        reports[name] = current
    means = {key: float(np.mean([v[key]["metrics"]["mAP@10"] for v in reports.values()]))
             for key in next(iter(reports.values()))}
    # Fixed, predeclared system policy for HPO selection. Others are diagnostics only.
    return {"means": means, "selection_score": means.get("G2", means["graph"]), "draws": reports,
            "selection_policy": "G2 expert10 against C_primary; no outer evaluation",
            "delta_system": means.get("G2", 0)-means.get("control", 0)}
