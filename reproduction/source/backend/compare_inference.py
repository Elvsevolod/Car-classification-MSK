"""Stage 2: compare raw, Flip TTA and streaming k-reciprocal inference."""
import json
import platform
import time
from itertools import product

import numpy as np
from PIL import Image

from .core import ARTIFACTS, DATASET, MODEL_NAME, ROOT, Encoder, bbox, encode_rows, read_rows, sha256
from .evaluate import (SEED, calibrate, make_protocol, make_splits, metrics,
                       ranked_queries, write_json)
from .rerank import KReciprocalReranker

OUTPUT = ROOT / "OSNet-AIN-x1.0" / "stage_02_inference" / "results.json"
K1_VALUES = (10, 20, 30)
K2_VALUES = (1, 3, 6)
LAMBDA_VALUES = (.2, .3, .5)


def load_protocols(dataset=DATASET):
    rows = read_rows(dataset / "train.csv")
    split_path = ARTIFACTS / "splits.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        identities = split.get("identities", {})
        valid = split.get("train_csv_sha256") == sha256(dataset / "train.csv")
        valid &= set(identities) == {"train", "calibration", "validation"}
    else:
        valid = False
    if not valid:
        hashes = {row["image_id"]: sha256(dataset / "images" / f"{row['image_id']}.jpg") for row in rows}
        identities = make_splits(rows, hashes)
    protocols = {name: make_protocol(rows, identities[name]) for name in ("calibration", "validation")}
    return protocols


def encode_protocols(encoder, protocols, dataset=DATASET):
    selected = {row["image_id"]: row for query, gallery in protocols.values() for row in query + gallery}
    rows = list(selected.values())
    cache = ARTIFACTS / f"stage2_embeddings_{encoder.model_sha256[:12]}.npz"
    identifiers = np.array([row["image_id"] for row in rows])
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if np.array_equal(saved["ids"], identifiers):
            return {"baseline": dict(zip(identifiers, saved["baseline"])),
                    "flip_tta": dict(zip(identifiers, saved["flip_tta"]))}
    print("Encoding stage 2 baseline embeddings…", flush=True)
    baseline = encode_rows(encoder, rows, dataset)
    print("Encoding stage 2 Flip TTA embeddings…", flush=True)
    flip_tta = encode_rows(encoder, rows, dataset, flip_tta=True)
    np.savez(cache, ids=identifiers, baseline=baseline, flip_tta=flip_tta)
    return {"baseline": dict(zip(identifiers, baseline)),
            "flip_tta": dict(zip(identifiers, flip_tta))}


def benchmark_extract(encoder, row, dataset=DATASET, flip_tta=False):
    def single():
        with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
            encoder.encode(image, bbox(row), flip_tta=flip_tta)
    for _ in range(3):
        single()
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        single()
        timings.append((time.perf_counter() - start) * 1000)
    includes = "JPEG decode + crop/resize + ONNX"
    if flip_tta:
        includes += " + horizontal-flip TTA"
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "samples": len(timings), "includes": includes + " + L2"}


def rerank_components(queries, gallery, embeddings, k1, k2):
    gallery_vectors = np.stack([embeddings[row["image_id"]] for row in gallery])
    start = time.perf_counter()
    reranker = KReciprocalReranker(gallery_vectors, k1, k2)
    build_seconds = time.perf_counter() - start
    raw, jaccard, raw_confidence = [], [], []
    start = time.perf_counter()
    for query in queries:
        vector = embeddings[query["image_id"]]
        raw_part, jaccard_part = reranker.components(vector)
        raw.append(raw_part)
        jaccard.append(jaccard_part)
        cosine = gallery_vectors @ vector
        raw_confidence.append(float(np.max(cosine)))
    query_seconds = time.perf_counter() - start
    return np.stack(raw), np.stack(jaccard), raw_confidence, {
        "gallery_graph_seconds": build_seconds,
        "mean_query_ms": query_seconds * 1000 / len(queries),
    }


def evaluate_direct(protocols, embeddings):
    calibration = ranked_queries(*protocols["calibration"], embeddings)
    validation = ranked_queries(*protocols["validation"], embeddings)
    threshold = calibrate(calibration)
    return {"threshold": threshold, "calibration": metrics(calibration, threshold),
            "validation": metrics(validation, threshold)}


def tune_reranking(protocols, embeddings):
    queries, gallery = protocols["calibration"]
    trials = []
    for k1, k2 in product(K1_VALUES, K2_VALUES):
        print(f"Calibration reranking: k1={k1}, k2={k2}", flush=True)
        raw, jaccard, raw_confidence, timing = rerank_components(queries, gallery, embeddings, k1, k2)
        for lambda_value in LAMBDA_VALUES:
            distances = (1 - lambda_value) * jaccard + lambda_value * raw
            ranked = ranked_queries(queries, gallery, embeddings, -distances)
            confidence_modes = {
                "raw_cosine": raw_confidence,
                "rerank_score": -distances.min(axis=1),
            }
            for confidence_mode, confidence in confidence_modes.items():
                threshold = calibrate(ranked, confidence)
                result = metrics(ranked, threshold, confidence)
                trials.append({"k1": k1, "k2": k2, "lambda": lambda_value,
                               "confidence_mode": confidence_mode, "threshold": threshold,
                               "quality_score": .45 * result["mAP_at_10"] + .10 * result["candidate_score"],
                               "metrics": result, "timing": timing})
    best = max(trials, key=lambda item: (item["quality_score"], item["metrics"]["mAP_at_10"],
                                         item["metrics"]["candidate_score"]))

    queries, gallery = protocols["validation"]
    raw, jaccard, raw_confidence, timing = rerank_components(
        queries, gallery, embeddings, best["k1"], best["k2"])
    distances = (1 - best["lambda"]) * jaccard + best["lambda"] * raw
    ranked = ranked_queries(queries, gallery, embeddings, -distances)
    confidence = raw_confidence if best["confidence_mode"] == "raw_cosine" else -distances.min(axis=1)
    validation = metrics(ranked, best["threshold"], confidence)
    return {"selected_on_calibration": best,
            "validation": validation,
            "validation_quality_score": .45 * validation["mAP_at_10"] + .10 * validation["candidate_score"],
            "validation_timing": timing,
            "trials": trials}


def main():
    encoder = Encoder()
    protocols = load_protocols()
    embeddings = encode_protocols(encoder, protocols)
    first_row = protocols["validation"][0][0]
    report = {
        "model": MODEL_NAME,
        "model_sha256": encoder.model_sha256,
        "seed": SEED,
        "selection_rule": "maximize 0.45*mAP@10 + 0.10*(0.7*F1 + 0.3*TNR) on calibration",
        "streaming_constraint": "one current query plus a precomputed static-gallery graph; no other queries",
        "grid": {"k1": K1_VALUES, "k2": K2_VALUES, "lambda": LAMBDA_VALUES,
                 "confidence_modes": ["raw_cosine", "rerank_score"]},
        "benchmark": {"platform": platform.platform(), "device": "CPU", "batch": 1,
                      "baseline": benchmark_extract(encoder, first_row),
                      "flip_tta": benchmark_extract(encoder, first_row, flip_tta=True)},
        "modes": {},
    }
    for name in ("baseline", "flip_tta"):
        print(f"Evaluating {name} without reranking…", flush=True)
        report["modes"][name] = evaluate_direct(protocols, embeddings[name])
        print(f"Tuning {name} reranking on calibration…", flush=True)
        report["modes"][f"{name}_rerank"] = tune_reranking(protocols, embeddings[name])
        write_json(OUTPUT, report)
    write_json(OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
