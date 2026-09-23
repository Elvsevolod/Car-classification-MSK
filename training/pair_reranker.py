"""Small pair heads on frozen embeddings, with query-balanced training and no metadata inputs."""
import copy
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

import evaluate as official
from backend.core import DATASET, normalize, preprocess, sha256
from backend.scoring import ranked_queries
from training.audit import digest, load_crop
from training.mask_calibration import _load, _save
from training.rerank_score_control import ScoreControl, ranking_scores
from training.stage6 import _save_checkpoint

TOP_K = 50
SCALARS = ("cosine", "one_minus_raw", "one_minus_jaccard", "kreciprocal_similarity",
           "support", "mutual", "raw_rank_fraction", "gap_to_best_cosine")
HEADS = (
    dict(name="scalar_linear", features="scalar", hidden=0, lr=.01, weight_decay=.001, seed=20260915),
    dict(name="pair_mlp64", features="full", hidden=64, lr=.001, weight_decay=.001, seed=20260915),
)
BUDGET = dict(max_epochs=25, min_epochs=5, patience=6, query_batch=32)


class PairFeatures:
    """The complete model input: current query + static gallery vectors only."""
    def __init__(self, gallery, baseline):
        self.gallery = normalize(gallery)
        if len(gallery) < TOP_K:
            raise ValueError("Pair reranker requires at least 50 gallery entries")
        self.graph = ScoreControl(self.gallery, baseline["k1"], baseline["k2"])
        self.baseline = baseline

    def one(self, vector):
        query = normalize(vector)
        p = self.graph.components(query)
        indices = np.argsort(-p["cosine"], kind="stable")[:TOP_K]
        cosine = p["cosine"][indices]
        scalar = np.stack([cosine, 1-p["raw"][indices], 1-p["jaccard"][indices],
            1-((1-self.baseline["lambda_value"])*p["jaccard"][indices] + self.baseline["lambda_value"]*p["raw"][indices]),
            p["support"][indices], p["mutual"][indices], np.arange(TOP_K)/(TOP_K-1), cosine[0]-cosine], axis=1)
        neighbors, d = self.gallery[indices], self.gallery.shape[1]
        features = np.concatenate([scalar, np.abs(query-neighbors)*np.sqrt(d), query*neighbors*d], axis=1).astype(np.float32)
        base = ranking_scores(p, {**self.baseline, "pool": 0})[indices] + 1
        return indices, features, base.astype(np.float32), float(cosine[0])


class PairSet:
    """Offline collection of independent one-query results; labels are separate."""
    def __init__(self, query, gallery, embeddings, baseline):
        self.query, self.gallery, self.embeddings = query, gallery, embeddings
        engine = PairFeatures(np.stack([embeddings[r["image_id"]] for r in gallery]), baseline)
        extracted = [engine.one(embeddings[r["image_id"]]) for r in query]
        self.indices = np.stack([e[0] for e in extracted])
        self.features = np.stack([e[1] for e in extracted])
        self.base = np.stack([e[2] for e in extracted])
        self.raw = ranked_queries(query, gallery, embeddings)
        self.confidence = self.raw.confidence.astype(float)

    def labels(self, frame_hashes):
        targets = np.zeros(self.indices.shape, dtype=np.float32)
        valid = np.ones(self.indices.shape, dtype=bool)
        for i, q in enumerate(self.query):
            for j, index in enumerate(self.indices[i]):
                g = self.gallery[int(index)]
                same_identity = q["vehicle_id"] == g["vehicle_id"]
                targets[i, j] = same_identity and q["camera_id"] != g["camera_id"]
                # Never train a junk/duplicate-frame pair as either positive or negative.
                valid[i, j] = not (same_identity and q["camera_id"] == g["camera_id"] or
                                  frame_hashes[q["image_id"]] == frame_hashes[g["image_id"]])
        positive, negative = valid & (targets == 1), valid & (targets == 0)
        groups = positive.any(1).astype(int) + negative.any(1)
        if np.any(groups == 0):
            raise ValueError("A training query has no usable pairs")
        weights = (positive/np.maximum(positive.sum(1, keepdims=True), 1) +
                   negative/np.maximum(negative.sum(1, keepdims=True), 1)) / groups[:, None]
        return targets, weights.astype(np.float32), valid

    def ranked(self, scores):
        scores = np.asarray(scores)
        if scores.shape != self.indices.shape or not np.isfinite(scores).all():
            raise ValueError("Need finite scores for exactly the raw top50")
        full = np.broadcast_to(scores.min(1, keepdims=True)-1, (len(self.query), len(self.gallery))).copy()
        np.put_along_axis(full, self.indices, scores, axis=1)
        order = np.argsort(-full, axis=1, kind="stable")[:, :10]
        g_ids = [r["image_id"] for r in self.gallery]
        predictions = {r["image_id"]: [g_ids[j] for j in indices] for r, indices in zip(self.query, order)}
        return replace(self.raw, predictions=predictions,
                       ranking=official.ranking_metrics(self.raw.query, self.raw.gallery, predictions))


class PairHead(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("std", torch.ones(dimension))
        self.net = (nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, 1))
                    if hidden else nn.Linear(dimension, 1))

    def forward(self, features):
        return self.net((features-self.mean)/self.std).squeeze(-1)


def selected_features(pair_set, config):
    return pair_set.features[..., :len(SCALARS)] if config["features"] == "scalar" else pair_set.features


@torch.no_grad()
def torch_logits(model, features):
    model.eval()
    matrix = features.reshape(-1, features.shape[-1])
    values = [model(torch.from_numpy(np.ascontiguousarray(part))).numpy() for part in np.array_split(matrix, max(1, len(matrix)//2048))]
    return np.concatenate(values).reshape(features.shape[:-1])


def fit_head(training, targets, weights, valid, config, directory, signature, validation=None,
             fixed_epochs=None, budget=None):
    """Choose epochs only on inner validation, OR refit a fixed budget without validation."""
    budget = BUDGET if budget is None else budget
    if (validation is None) != (fixed_epochs is not None):
        raise ValueError("Use either inner selection or a fixed final epoch count")
    if fixed_epochs is not None and not 1 <= fixed_epochs <= budget["max_epochs"]:
        raise ValueError("Final epochs must fit the predeclared budget")
    if len({r["vehicle_id"] for r in training.query}) != len(training.query):
        raise ValueError("One query per identity is required for identity-balanced loss")
    if validation and ({r["vehicle_id"] for r in training.query+training.gallery} &
                       {r["vehicle_id"] for r in validation.query+validation.gallery}):
        raise ValueError("Pair-head inner identity leakage")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = {"experiment": signature, "config": config, "budget": budget, "fixed_epochs": fixed_epochs,
                 "train_queries": [r["image_id"] for r in training.query],
                 "inner_queries": [r["image_id"] for r in validation.query] if validation else None}
    report_path, last_path = directory / "summary.json", directory / "last.pt"
    if report_path.exists():
        report = _load(report_path, signature)
        if sha256(directory / "selected.pt") != report["weights_sha256"]:
            raise ValueError("Selected pair-head weights changed")
        return report
    torch.manual_seed(config["seed"])
    features = selected_features(training, config)
    model = PairHead(features.shape[-1], config["hidden"])
    observed = features[valid]
    model.mean.copy_(torch.from_numpy(observed.mean(0)))
    model.std.copy_(torch.from_numpy(np.maximum(observed.std(0), 1e-4)))
    del observed
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    history, best_model, best_epoch, best_score = [], None, 0, (-1., -1.)
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Interrupted head belongs to another experiment")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_model, best_epoch, best_score = (state[k] for k in ("history", "best_model", "best_epoch", "best_score"))
    x, y, w = (torch.from_numpy(np.ascontiguousarray(a)) for a in (features, targets, weights))
    horizon = fixed_epochs if fixed_epochs is not None else budget["max_epochs"]
    for epoch in range(len(history)+1, horizon+1):
        if validation and len(history) >= budget["min_epochs"] and len(history)-best_epoch >= budget["patience"]:
            break
        started = time.perf_counter()
        torch.manual_seed(config["seed"] + epoch)
        model.train()
        total = 0.
        for index in torch.randperm(len(x)).split(budget["query_batch"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[index])
            loss = (F.binary_cross_entropy_with_logits(logits, y[index], reduction="none") * w[index]).sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pair-head loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(index)
        record = {"epoch": epoch, "loss": total/len(x)}
        if validation:
            ranked = validation.ranked(torch_logits(model, selected_features(validation, config)))
            score = (ranked.ranking["mAP@10"], ranked.ranking["Rank-1"])
            record["inner"] = ranked.ranking
            if score > tuple(best_score):
                best_score, best_epoch, best_model = score, epoch, copy.deepcopy(model.state_dict())
        else:
            best_epoch, best_model = epoch, copy.deepcopy(model.state_dict())
        record["epoch_seconds"] = time.perf_counter()-started
        history.append(record)
        _save_checkpoint(last_path, {"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "best_model": best_model, "best_epoch": best_epoch, "best_score": best_score})
        elapsed = sum(r["epoch_seconds"] for r in history)
        print(f"{config['name']} | epoch {epoch}/{horizon}, remaining {horizon-epoch} | "
              f"epoch {record['epoch_seconds']:.1f}s | ETA <= {elapsed/epoch*(horizon-epoch):.1f}s" +
              (f" | inner mAP {score[0]:.6f}" if validation else " | fixed-budget refit"), flush=True)
    _save_checkpoint(directory / "selected.pt", {"config": config, "model": best_model, "dimension": features.shape[-1],
                     "epoch": best_epoch, "signature": signature})
    report = {"signature": signature, "selected_epoch": best_epoch, "history": history,
              "best_inner": list(best_score) if validation else None,
              "weights_sha256": sha256(directory / "selected.pt"), "outer_used_for_training": False}
    _save(report_path, report)
    return report


class HeadEncoder:
    def __init__(self, path, expected_hash, config):
        if sha256(path) != expected_hash:
            raise ValueError("Pair-head ONNX checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.config = config

    def probabilities(self, features):
        features = features[..., :len(SCALARS)] if self.config["features"] == "scalar" else features
        shape = features.shape[:-1]
        matrix = np.ascontiguousarray(features.reshape(-1, features.shape[-1]), dtype=np.float32)
        logits = self.session.run(["logits"], {"pairs": matrix})[0]
        return (1/(1+np.exp(-np.clip(logits, -80, 80)))).reshape(shape)


def export_head(checkpoint, output, features, signature):
    output = Path(output)
    manifest, path = output / "export.json", output / "pair_head.onnx"
    if manifest.exists():
        report = _load(manifest, signature)
        return HeadEncoder(path, report["sha256"], report["config"]), report
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = PairHead(state["dimension"], state["config"]["hidden"]).eval()
    model.load_state_dict(state["model"])
    samples = features[..., :state["dimension"]].reshape(-1, state["dimension"])[:100]
    temporary = output / "pair_head.partial.onnx"
    with torch.no_grad():
        torch.onnx.export(model, torch.from_numpy(samples[:1]), str(temporary), dynamo=False, opset_version=17,
            input_names=["pairs"], output_names=["logits"], dynamic_axes={"pairs": {0: "batch"}, "logits": {0: "batch"}})
    path_hash = sha256(temporary)
    encoder = HeadEncoder(temporary, path_hash, state["config"])
    errors = []
    for count in (1, 50, 100):
        actual = encoder.session.run(["logits"], {"pairs": np.ascontiguousarray(samples[:count])})[0]
        expected = torch_logits(model, samples[:count])
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        errors.append(float(np.max(np.abs(actual-expected))))
    temporary.replace(path)
    report = {"signature": signature, "sha256": path_hash, "bytes": path.stat().st_size,
        "config": state["config"], "dimension": state["dimension"], "max_absolute_error": max(errors),
        "batches_checked": [1, 50, 100], "normalization_included": True}
    _save(manifest, report)
    return encoder, report


def encode_cached(encoder, rows, path, signature, dataset=DATASET, batch_size=32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = [r["image_id"] for r in rows]
    fingerprint = digest({"signature": signature, "rows": rows, "encoder": encoder.model_sha256})
    ids, vectors = [], np.empty((0, 512), dtype=np.float32)
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            ids, vectors = cache["ids"].tolist(), cache["vectors"].copy()
            valid = str(cache["fingerprint"]) == fingerprint and ids == expected[:len(ids)]
        if (not valid or vectors.shape != (len(ids), 512) or vectors.dtype != np.float32
                or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError("Stale or invalid OSNet cache")
    started, initial = time.perf_counter(), len(ids)
    for start in range(initial, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        crops = [load_crop(r, dataset) for r in batch]
        found = encoder.encode_batch([preprocess(c, (0, 0, *c.size)) for c in crops])
        vectors = np.concatenate([vectors, found])
        ids.extend(r["image_id"] for r in batch)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, ids=np.asarray(ids), vectors=vectors, fingerprint=fingerprint)
        temporary.replace(path)
        if len(ids) % 512 == 0 or len(ids) == len(rows):
            elapsed = time.perf_counter()-started
            print(f"{path.name}: {len(ids)}/{len(rows)} | ETA {elapsed/(len(ids)-initial)*(len(rows)-len(ids)):.1f}s", flush=True)
    return dict(zip(ids, vectors))
