"""Создаёт и проверяет конкурсные submission.csv, embeddings.npy и candidates.csv.

Формулы метрик организаторов остаются в неизменённом evaluate.py в корне проекта.
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

from .bootstrap import gallery_repository_from_environment
from .core import (ARTIFACTS, DATASET, MODEL, MODEL_FINE_TUNED, MODEL_NAME, ROOT,
                   MODEL_TRAINING_EPOCH, PREPROCESS, Encoder, Gallery, bbox,
                   encode_rows, read_rows, sha256)
from .rerank import (ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA,
                     rerank_protocol)
from .scoring import calibrate, metrics, ranked_queries, threshold_curve

SEED = 20260915
CANDIDATES_HEADER = ["query_id", "gallery_id", "confidence"]


def make_splits(rows, frame_hashes, seed=SEED):
    # Connected identities sharing any exact frame must stay in one partition.
    parent = {r["vehicle_id"]: r["vehicle_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    frames = {}
    for row in rows:
        identity = row["vehicle_id"]
        digest = frame_hashes[row["image_id"]]
        if digest in frames:
            parent[find(identity)] = find(frames[digest])
        frames[digest] = identity
    groups = defaultdict(list)
    for identity in sorted(parent):
        groups[find(identity)].append(identity)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    # 60% training, 20% calibration, 20% validation.
    a, b = int(len(groups) * .6), int(len(groups) * .8)
    return {name: sorted(i for group in part for i in group) for name, part in
            zip(("train", "calibration", "validation"), (groups[:a], groups[a:b], groups[b:]))}


def make_protocol(rows, identities, seed=SEED):
    by_id = defaultdict(list)
    for row in rows:
        if row["vehicle_id"] in set(identities):
            by_id[row["vehicle_id"]].append(row)
    order = sorted(by_id)
    rng = random.Random(seed)
    rng.shuffle(order)
    unknown = set(order[:max(1, round(len(order) * .2))])
    queries, gallery = [], []
    for identity in order:
        items = sorted(by_id[identity], key=lambda r: r["image_id"])
        cameras = sorted({r["camera_id"] for r in items})
        if len(cameras) < 2:
            raise ValueError(f"Identity {identity} has fewer than two cameras")
        camera = rng.choice(cameras)
        queries.append(rng.choice([r for r in items if r["camera_id"] == camera]))
        if identity not in unknown:
            gallery.extend(r for r in items if r["camera_id"] != camera)
    return queries, gallery


def _read_strict_csv(path, expected_header):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != expected_header:
            raise ValueError(f"Invalid header in {path.name}: {header}; expected {expected_header}")
        rows = list(reader)
    if any(len(row) != len(expected_header) for row in rows):
        raise ValueError(f"Invalid column count in {path.name}")
    return rows


def validate_artifacts(dataset=DATASET, output=ARTIFACTS):
    """Validate the three submission artifacts against the published contract."""
    query_ids = [row["image_id"] for row in read_rows(dataset / "test_query.csv")]
    gallery_ids = [row["image_id"] for row in read_rows(dataset / "test_gallery.csv")]
    query_set = set(query_ids)
    gallery_set = set(gallery_ids)

    embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(query_ids) + len(gallery_ids):
        raise ValueError("embeddings.npy must be a 2D array with query rows followed by gallery rows")
    if embeddings.dtype != np.float32 or embeddings.shape[1] < 1:
        raise ValueError("embeddings.npy must have dtype float32 and a non-empty embedding dimension")
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise ValueError("embeddings.npy contains non-finite or zero vectors")

    with open(output / "submission.csv", newline="", encoding="utf-8-sig") as stream:
        submission = list(csv.reader(stream))
    top_k = min(10, len(gallery_ids))  # Organizer example has only eight gallery objects.
    if any(len(row) != top_k + 1 for row in submission):
        raise ValueError("submission.csv must have no header and query_id plus Top-K gallery IDs")
    if [row[0] for row in submission] != query_ids:
        raise ValueError("submission.csv must contain every query exactly once in test_query.csv order")
    for row in submission:
        candidates = row[1:]
        if len(set(candidates)) != top_k or not set(candidates).issubset(gallery_set):
            raise ValueError(f"submission.csv query {row[0]} must contain {top_k} distinct gallery IDs")

    candidates = _read_strict_csv(output / "candidates.csv", CANDIDATES_HEADER)
    seen_pairs = set()
    accepted_queries = set()
    for query_id, gallery_id, confidence in candidates:
        if query_id not in query_set or gallery_id not in gallery_set:
            raise ValueError("candidates.csv contains an unknown or empty query/gallery ID")
        try:
            score = float(confidence)
        except ValueError as error:
            raise ValueError("candidates.csv confidence must be numeric") from error
        if not np.isfinite(score):
            raise ValueError("candidates.csv confidence must be finite")
        pair = query_id, gallery_id
        if pair in seen_pairs:
            raise ValueError(f"Duplicate candidate pair: {query_id}, {gallery_id}")
        seen_pairs.add(pair)
        accepted_queries.add(query_id)
    return {"queries": len(query_ids), "gallery": len(gallery_ids),
            "embedding_shape": list(embeddings.shape), "embedding_dtype": str(embeddings.dtype),
            "submission_rows": len(submission), "candidate_rows": len(candidates),
            "accepted_queries": len(accepted_queries), "refused_queries": len(query_ids) - len(accepted_queries)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(encoder, dataset=DATASET, output=ARTIFACTS):
    rows = read_rows(dataset / "train.csv")
    print("Hashing train frames and fixing identity-disjoint partitions…", flush=True)
    hashes = {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg") for r in rows}
    splits = make_splits(rows, hashes)
    protocols = {name: make_protocol(rows, splits[name]) for name in ("calibration", "validation")}
    selected = {r["image_id"]: r for q, g in protocols.values() for r in q + g}
    selected = list(selected.values())
    features = encode_rows(encoder, selected, dataset)
    embeddings = dict(zip((r["image_id"] for r in selected), features))
    raw_cal = ranked_queries(*protocols["calibration"], embeddings)
    raw_val = ranked_queries(*protocols["validation"], embeddings)
    cal, cal_confidence = rerank_protocol(*protocols["calibration"], embeddings)
    val, val_confidence = rerank_protocol(*protocols["validation"], embeddings)
    threshold = calibrate(cal, cal_confidence)
    manifest = {"seed": SEED, "identities": splits, "train_csv_sha256": sha256(dataset / "train.csv"),
                "frame_sha256": hashes,
                "protocols": {name: {"query_ids": [r["image_id"] for r in q],
                                      "gallery_ids": [r["image_id"] for r in g]} for name, (q, g) in protocols.items()}}
    write_json(output / "splits.json", manifest)
    # Inference batch=1, including image decode/crop/resize; warm up first.
    example = selected[0]
    def single():
        with Image.open(dataset / "images" / f"{example['image_id']}.jpg") as image:
            encoder.encode(image, bbox(example))
    for _ in range(3):
        single()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        single()
        times.append((time.perf_counter() - start) * 1000)
    report = {"model": MODEL_NAME, "fine_tuned": MODEL_FINE_TUNED,
              "evaluator": {"script": "evaluate.py", "sha256": sha256(ROOT / "evaluate.py")},
              "training_stage": "development", "training_epoch": MODEL_TRAINING_EPOCH,
              "model_sha256": encoder.model_sha256, "encoder_fingerprint": encoder.fingerprint,
              "preprocessing": PREPROCESS, "threshold": threshold,
              "threshold_metric": "query-level 0.7*F1 + 0.3*TNR on calibration raw cosine only",
              "confidence_definition": "maximum raw cosine over gallery, not a probability", "seed": SEED,
              "search": {"ranking": "streaming k-reciprocal", "k1": ACTIVE_K1,
                         "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA,
                         "refusal": "maximum raw cosine"},
              "calibration": metrics(cal, threshold, cal_confidence),
              "validation": metrics(val, threshold, val_confidence),
              "raw_baseline": {"calibration": metrics(raw_cal, threshold),
                               "validation": metrics(raw_val, threshold)},
              "partition_identity_counts": {k: len(v) for k, v in splits.items()},
              "benchmark": {"platform": platform.platform(), "device": "CPU", "threads": 2,
                            "batch": 1, "includes": "JPEG decode + preprocessing + inference + L2",
                            "samples": 30, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))},
              "weights_bytes": MODEL.stat().st_size,
              "limitations": ["Local train holdout, not the organizer test score",
                              "Development checkpoint selected on local validation; not the final train+validation model",
                              "camera_id used only for validation splitting/filtering", "TNR uses 20% synthetic no-match identities",
                              "mAP/CMC/mINP exclude queries without a cross-camera match",
                              "Exact frame hashes detect exact copies only, not near-duplicates"]}
    write_json(output / "baseline_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def export(encoder, report, dataset=DATASET, output=ARTIFACTS, gallery_repository=None):
    queries = read_rows(dataset / "test_query.csv")
    repository = gallery_repository or gallery_repository_from_environment()
    gallery = Gallery(encoder, dataset, repository=repository)
    query_vectors = encode_rows(encoder, queries, dataset)
    np.save(output / "embeddings.npy", np.concatenate([query_vectors, gallery.vectors]).astype(np.float32))
    threshold = report["threshold"]
    with open(output / "submission.csv", "w", newline="") as submission, open(output / "candidates.csv", "w", newline="") as candidates:
        writer, accepted = csv.writer(submission), csv.writer(candidates)
        accepted.writerow(CANDIDATES_HEADER)
        for query, vector in zip(queries, query_vectors):
            results = gallery.search(vector, 10)
            writer.writerow([query["image_id"]] + [r["image_id"] for r in results])
            confidence = gallery.confidence(vector)
            if confidence >= threshold:
                result = results[0]
                # Monotone [0,1] score, explicitly NOT calibrated probability.
                accepted.writerow([query["image_id"], result["image_id"], (confidence + 1) / 2])
    validation = validate_artifacts(dataset, output)
    write_json(output / "export_manifest.json", {
        "evaluator_sha256": sha256(ROOT / "evaluate.py"),
        "model_sha256": encoder.model_sha256, "encoder_fingerprint": encoder.fingerprint,
        "gallery_fingerprint": gallery.fingerprint,
        "query_csv_sha256": sha256(dataset / "test_query.csv"),
        "gallery_csv_sha256": sha256(dataset / "test_gallery.csv"),
        "query_image_sha256": {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg") for r in queries},
        "embedding_ids": [r["image_id"] for r in queries + gallery.rows],
        "shape": [len(queries) + len(gallery.rows), query_vectors.shape[1]], "dtype": "float32", "l2_normalized": True,
        "cosine_threshold": threshold, "confidence_threshold": (threshold + 1) / 2,
        "reranking": {"method": "streaming k-reciprocal", "k1": ACTIVE_K1,
                      "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA},
        "confidence": "(maximum raw gallery cosine + 1) / 2; not a probability",
        "refusal_encoding": "no candidates.csv rows for the refused query",
        "validation": validation})
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Also generate official test artifacts")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing submission artifacts")
    parser.add_argument("--output", type=Path, help="Directory for generated or validated artifacts")
    args = parser.parse_args()
    output = args.output or ARTIFACTS
    if args.validate_only:
        print(json.dumps(validate_artifacts(output=output), ensure_ascii=False, indent=2))
        raise SystemExit
    encoder = Encoder()
    report = evaluate(encoder, output=output)
    if args.export:
        export(encoder, report, output=output)
