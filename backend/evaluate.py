"""Reproducible baseline, threshold calibration and official test exports."""
import argparse
import csv
import json
import platform
import random
import time
from collections import defaultdict

import numpy as np
from PIL import Image

from .core import (ARTIFACTS, DATASET, MODEL, PREPROCESS, Encoder, Gallery,
                   bbox, encode_rows, rank, read_rows, sha256)

SEED = 20260915


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
    # 60% reserved for future fine-tuning, 20% calibration, 20% validation.
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


def ranked_queries(queries, gallery, embeddings):
    vectors = np.stack([embeddings[r["image_id"]] for r in gallery])
    output = []
    for query in queries:
        scores = np.clip(vectors @ embeddings[query["image_id"]], -1, 1)
        # TЗ: all comparisons from the same camera are excluded.
        eligible = np.array([i for i, r in enumerate(gallery) if r["camera_id"] != query["camera_id"]])
        order = eligible[rank(scores[eligible], len(eligible))]
        matches = np.array([gallery[int(i)]["vehicle_id"] == query["vehicle_id"] for i in order])
        output.append((scores[order], matches))
    return output


def metrics(ranked, threshold):
    aps, inps, r1, r5 = [], [], [], []
    tp = fp = total_relevant = unknown = true_negative = 0
    for scores, matches in ranked:
        positions = np.flatnonzero(matches)
        total_relevant += len(positions)
        accepted = scores[:10] >= threshold
        tp += int(np.sum(matches[:10] & accepted))
        fp += int(np.sum(~matches[:10] & accepted))
        if len(positions):
            aps.append(float(np.mean(np.arange(1, len(positions) + 1) / (positions + 1))))
            inps.append(float(len(positions) / (positions[-1] + 1)))
            r1.append(bool(matches[0]))
            r5.append(bool(np.any(matches[:5])))
        else:
            unknown += 1
            true_negative += int(not accepted.any())
    fn = total_relevant - tp  # Includes positives beyond the returned top-10.
    return {"mAP": float(np.mean(aps)) if aps else None,
            "Rank_1": float(np.mean(r1)) if r1 else None,
            "Rank_5": float(np.mean(r5)) if r5 else None,
            "mINP": float(np.mean(inps)) if inps else None,
            "candidate_precision": tp / (tp + fp) if tp + fp else 0,
            "candidate_recall": tp / total_relevant if total_relevant else 0,
            "candidate_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "TNR": true_negative / unknown if unknown else None,
            "known_queries": len(aps), "unknown_queries": unknown,
            "TP": tp, "FP": fp, "FN": fn, "true_refusals": true_negative}


def calibrate(ranked):
    # Search every distinct top-10 score, plus the all-refuse boundary.
    # Equal F1 prefers the higher (more conservative) threshold.
    values = np.unique(np.concatenate([s[:10] for s, _ in ranked])).astype(float)
    choices = np.append(values, np.nextafter(1.0, 2.0))
    return max(choices, key=lambda t: (metrics(ranked, float(t))["candidate_F1"], t)).item()


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
    cal = ranked_queries(*protocols["calibration"], embeddings)
    val = ranked_queries(*protocols["validation"], embeddings)
    threshold = calibrate(cal)
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
    report = {"model": "vehicle-reid-0001 / OSNet-AIN x1.0", "fine_tuned": False,
              "model_sha256": encoder.model_sha256, "encoder_fingerprint": encoder.fingerprint,
              "preprocessing": PREPROCESS, "threshold": threshold, "threshold_metric": "top10 pairwise F1 on calibration only",
              "confidence_definition": "cosine similarity, not a probability", "seed": SEED,
              "calibration": metrics(cal, threshold), "validation": metrics(val, threshold),
              "partition_identity_counts": {k: len(v) for k, v in splits.items()},
              "benchmark": {"platform": platform.platform(), "device": "CPU", "threads": 2,
                            "batch": 1, "includes": "JPEG decode + preprocessing + inference + L2",
                            "samples": 30, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))},
              "weights_bytes": MODEL.stat().st_size,
              "limitations": ["Local train holdout, not the organizer test score", "No model fine-tuning performed",
                              "camera_id used only for validation splitting/filtering", "TNR uses 20% synthetic no-match identities",
                              "mAP/CMC/mINP exclude queries without a cross-camera match",
                              "Exact frame hashes detect exact copies only, not near-duplicates"]}
    write_json(output / "baseline_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def export(encoder, report, dataset=DATASET, output=ARTIFACTS):
    queries = read_rows(dataset / "test_query.csv")
    gallery = Gallery(encoder, dataset, output / "gallery.sqlite3")
    if len(gallery.rows) < 10:
        raise ValueError("Official export requires at least 10 gallery objects")
    query_vectors = encode_rows(encoder, queries, dataset)
    np.save(output / "embeddings.npy", np.concatenate([query_vectors, gallery.vectors]).astype(np.float32))
    threshold = report["threshold"]
    with open(output / "submission.csv", "w", newline="") as submission, open(output / "candidates.csv", "w", newline="") as candidates:
        writer, accepted = csv.writer(submission), csv.writer(candidates)
        writer.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, 11)])
        accepted.writerow(["query_id", "gallery_id", "confidence"])
        for query, vector in zip(queries, query_vectors):
            results = gallery.search(vector, 10)
            writer.writerow([query["image_id"]] + [r["image_id"] for r in results])
            matches = [r for r in results if r["similarity"] >= threshold]
            if not matches:
                accepted.writerow([query["image_id"], "", ""])
            for result in matches:
                # Monotone [0,1] score, explicitly NOT calibrated probability.
                accepted.writerow([query["image_id"], result["image_id"], (result["similarity"] + 1) / 2])
    write_json(output / "export_manifest.json", {
        "model_sha256": encoder.model_sha256, "encoder_fingerprint": encoder.fingerprint,
        "gallery_fingerprint": gallery.fingerprint,
        "query_csv_sha256": sha256(dataset / "test_query.csv"),
        "gallery_csv_sha256": sha256(dataset / "test_gallery.csv"),
        "query_image_sha256": {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg") for r in queries},
        "embedding_ids": [r["image_id"] for r in queries + gallery.rows],
        "shape": [len(queries) + len(gallery.rows), 512], "dtype": "float32", "l2_normalized": True,
        "cosine_threshold": threshold, "confidence_threshold": (threshold + 1) / 2,
        "confidence": "(cosine_similarity + 1) / 2; not a probability",
        "refusal_encoding": "query_id with empty gallery_id and confidence; confirm with organizers"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Also generate official test artifacts")
    args = parser.parse_args()
    encoder = Encoder()
    report = evaluate(encoder)
    if args.export:
        export(encoder, report)
