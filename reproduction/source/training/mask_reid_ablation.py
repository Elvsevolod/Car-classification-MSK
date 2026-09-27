"""Paired full-validation audit of frozen YOLO masks with the active MVP encoder."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import evaluate as official
from backend.core import (ARTIFACTS, DATASET, MODEL, MODEL_NAME, PREPROCESS, ROOT,
                          Encoder, bbox, preprocess, read_rows, sha256)
from backend.evaluate import SEED, make_protocol
from backend.rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA, rerank_protocol
from backend.scoring import metrics, ranked_queries
from training.audit import digest, load_crop, opaque_mask
from training.mask_calibration import _context, _load, _save
from training.mask_detection import restore_boxes
from training.mask_finetune import EXPERIMENT as DETECTOR_EXPERIMENT, load_json
from training.stage6 import audit_partitions
from training.yolo_masks import device_name, runtime


EXPERIMENT = ROOT / "YOLO11/variant_02_mvp_reid_ablation"
CALIBRATION = DETECTOR_EXPERIMENT / "runs/pilot_01/calibration_v1"
WEIGHTS = DETECTOR_EXPERIMENT / "runs/pilot_01/weights/best.pt"


def check_detector_separation(selected, frame_hashes, detector_plan):
    ids = {r["image_id"] for r in selected}
    identities = {r["vehicle_id"] for r in selected}
    frames = {frame_hashes[i] for i in ids}
    for image_id, item in detector_plan["images"].items():
        if image_id in ids or item["vehicle_id"] in identities or item["frame_sha256"] in frames:
            raise ValueError("Detector data overlaps ReID validation by image, identity or exact frame")


def prepare(dataset=DATASET):
    rows = read_rows(dataset / "train.csv")
    split = load_json(ARTIFACTS / "splits.json")
    if sha256(dataset / "train.csv") != split["train_csv_sha256"]:
        raise ValueError("Changed train.csv")
    frames = {}
    for n, row in enumerate(rows, 1):
        frames[row["image_id"]] = sha256(dataset / "images" / f"{row['image_id']}.jpg")
        if n % 2000 == 0:
            print(f"Source integrity: {n}/{len(rows)}", flush=True)
    if frames != split["frame_sha256"]:
        raise ValueError("Source frames changed since the frozen ReID split")
    audit_partitions(rows, frames, split["identities"])
    queries, gallery = make_protocol(rows, split["identities"]["validation"], SEED)
    protocol = {"query_ids": [r["image_id"] for r in queries], "gallery_ids": [r["image_id"] for r in gallery]}
    if protocol != split["protocols"]["validation"]:
        raise ValueError("Protocol differs from MVP validation")
    selected = queries + gallery
    if len({r["image_id"] for r in selected}) != len(selected):
        raise ValueError("Query/gallery IDs must be disjoint and unique")

    detector_plan, _, detector_signature = _context(
        WEIGHTS, DETECTOR_EXPERIMENT / "annotation/reviewed_masks.json", DETECTOR_EXPERIMENT)
    frozen = _load(CALIBRATION / "calibration.json", detector_signature)
    if frozen["raw_sha256"] != sha256(CALIBRATION / "raw_val.json"):
        raise ValueError("Changed mask calibration cache")
    if detector_plan["outer_split_sha256"] != sha256(ARTIFACTS / "splits.json"):
        raise ValueError("Detector was prepared from a different outer split")
    check_detector_separation(selected, frames, detector_plan)

    encoder = Encoder()
    reference = load_json(ARTIFACTS / "baseline_metrics.json")
    rerank = {"k1": ACTIVE_K1, "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA}
    if (reference["model_sha256"] != encoder.model_sha256 or reference["encoder_fingerprint"] != encoder.fingerprint
            or reference["evaluator"]["sha256"] != sha256(ROOT / "evaluate.py")
            or reference["preprocessing"] != PREPROCESS
            or any(reference["search"][k] != v for k, v in rerank.items())):
        raise ValueError("MVP model, preprocessing, evaluator or reranking differs from its baseline report")
    sources = ["training/mask_reid_ablation.py", "training/audit.py", "training/mask_detection.py",
               "backend/core.py", "backend/rerank.py", "backend/scoring.py", "backend/evaluate.py", "evaluate.py"]
    signature = {"model": MODEL_NAME, "model_sha256": encoder.model_sha256, "preprocess": PREPROCESS,
        "detector_sha256": sha256(WEIGHTS), "calibration_sha256": sha256(CALIBRATION / "calibration.json"),
        "mask_policy": frozen["selected"], "inference": frozen["signature"]["inference"],
        "threshold_fixed": float(reference["threshold"]), "reranking_fixed": rerank, "flip_tta": False,
        "split_sha256": sha256(ARTIFACTS / "splits.json"),
        "reference_sha256": sha256(ARTIFACTS / "baseline_metrics.json"),
        "train_csv_sha256": split["train_csv_sha256"], "protocol": protocol,
        "rows": selected, "frames": {r["image_id"]: frames[r["image_id"]] for r in selected},
        "code": {p: sha256(ROOT / p) for p in sources}}
    return queries, gallery, encoder, reference, signature


def detect_mask(model, crop, policy, inference, device):
    result = model.predict(crop, device=device, **{**inference, "conf": policy["confidence"]})[0]
    boxes = result.boxes.xyxy.cpu().numpy().astype(float)
    scores = result.boxes.conf.cpu().numpy().astype(float)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite mask detections")
    grow = (boxes[:, 2:] - boxes[:, :2]) * policy["margin"]
    boxes[:, :2] -= grow
    boxes[:, 2:] += grow
    rects, scores = restore_boxes(boxes, scores, 1., (0, 0), crop.size)
    return {"rectangles": rects, "confidences": scores, "reviewed": False}


def load_cache(path, fingerprint, expected_ids):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        ids = saved["ids"].tolist()
        vectors = {k: saved[k].copy() for k in ("original", "masked")}
    if metadata["fingerprint"] != fingerprint or ids != expected_ids[:len(ids)]:
        raise ValueError("Paired cache belongs to a different protocol or configuration")
    if set(metadata["predictions"]) != set(ids):
        raise ValueError("Paired cache masks/embeddings disagree")
    for array in vectors.values():
        if (array.shape != (len(ids), 512) or array.dtype != np.float32 or not np.isfinite(array).all()
                or not np.allclose(np.linalg.norm(array, axis=1), 1., atol=1e-5)):
            raise ValueError("Invalid cached embeddings")
    return ids, vectors, metadata


def save_cache(path, ids, vectors, metadata):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), metadata=json.dumps(metadata), **vectors)
    temporary.replace(path)


def paired_embeddings(encoder, rows, signature, output, device, dataset=DATASET, batch_size=16):
    """Decode once; mask BOTH query and gallery before the unchanged MVP resize."""
    ids = [r["image_id"] for r in rows]
    cache = output / "paired_embeddings.npz"
    fingerprint = digest(signature)
    if cache.exists():
        done, vectors, metadata = load_cache(cache, fingerprint, ids)
        if metadata["device"] != device and len(done) != len(ids):
            raise ValueError("Resume incomplete detector inference on the original device")
    else:
        done, vectors = [], {k: np.empty((0, 512), np.float32) for k in ("original", "masked")}
        metadata = {"fingerprint": fingerprint, "device": device, "predictions": {}, "elapsed_seconds": 0.}
    if len(done) == len(ids):
        print("Reuse complete paired embedding cache", flush=True)
        return vectors, metadata
    model = runtime()(str(WEIGHTS))
    started, previous_seconds, initial_count = time.perf_counter(), metadata["elapsed_seconds"], len(done)
    for start in range(len(done), len(rows), batch_size):
        batch = rows[start:start+batch_size]
        before, after = [], []
        for row in batch:
            crop = load_crop(row, dataset)
            predicted = detect_mask(model, crop, signature["mask_policy"], signature["inference"], device)
            metadata["predictions"][row["image_id"]] = predicted
            before.append(preprocess(crop, (0, 0, *crop.size)))
            masked = opaque_mask(crop, predicted["rectangles"])
            after.append(preprocess(masked, (0, 0, *masked.size)))
        for name, tensors in (("original", before), ("masked", after)):
            vectors[name] = np.concatenate([vectors[name], encoder.encode_batch(tensors)])
        done.extend(r["image_id"] for r in batch)
        elapsed = time.perf_counter() - started
        metadata["elapsed_seconds"] = previous_seconds + elapsed
        if len(done) % 64 == 0 or len(done) == len(ids):
            save_cache(cache, done, vectors, metadata)
            eta = elapsed / (len(done)-initial_count) * (len(ids)-len(done))
            print(f"Paired crops: {len(done)}/{len(ids)} · elapsed {elapsed:.1f}s · ETA {eta:.1f}s", flush=True)
    return vectors, metadata


def evaluate_embeddings(queries, gallery, vectors, threshold):
    ids = [r["image_id"] for r in queries + gallery]
    embeddings = dict(zip(ids, vectors))
    raw = ranked_queries(queries, gallery, embeddings)
    reranked, _ = rerank_protocol(queries, gallery, embeddings)
    scores, details = {}, {}
    for mode, ranked in (("raw", raw), ("reranked", reranked)):
        scores[mode] = metrics(ranked, threshold)
        details[mode] = {}
        for n, qid in enumerate(ranked.query.index):
            one = official.ranking_metrics(ranked.query.loc[[qid]], ranked.gallery, ranked.predictions)
            known = one["n_scored"] == 1
            details[mode][qid] = {"known": known, "AP_at_10": one["mAP@10"] if known else None,
                "top1_correct": bool(one["Rank-1"]) if known else False,
                "accepted": bool(ranked.confidence[n] >= threshold),
                "confidence": float(ranked.confidence[n]), "top10": ranked.predictions[qid]}
    return scores, details


def paired_deltas(before, after):
    if set(before) != set(after) or any(before[i]["known"] != after[i]["known"] for i in before):
        raise ValueError("Paired comparison needs the same queries and known/unknown labels")
    known = [i for i in before if before[i]["known"]]
    delta = np.array([after[i]["AP_at_10"] - before[i]["AP_at_10"] for i in known])
    rng = np.random.default_rng(20260921)
    boot = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
    changes = {"top1_improved": [], "top1_worsened": [], "acceptance_changed": [], "top1_changed": []}
    for i in before:
        a, b = before[i], after[i]
        if b["top1_correct"] and not a["top1_correct"]: changes["top1_improved"].append(i)
        if a["top1_correct"] and not b["top1_correct"]: changes["top1_worsened"].append(i)
        if a["accepted"] != b["accepted"]: changes["acceptance_changed"].append(i)
        if a["top10"][0] != b["top10"][0]: changes["top1_changed"].append(i)
    return {"mAP_at_10_delta": float(delta.mean()),
        "paired_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
        "bootstrap_note": "2000 paired query resamples, seed 20260921, fixed gallery; local diagnostic only",
        "AP_improved": int((delta > 1e-12).sum()), "AP_worsened": int((delta < -1e-12).sum()),
        "AP_unchanged": int((np.abs(delta) <= 1e-12).sum()), **changes}


def run(output=EXPERIMENT / "runs/paired_v1", device=None):
    device = device or device_name()
    protected_paths = [MODEL, WEIGHTS, CALIBRATION / "calibration.json", CALIBRATION / "holdout_final.json",
        *[ARTIFACTS / name for name in ("gallery.sqlite3", "embeddings.npy", "submission.csv", "candidates.csv",
                                      "baseline_metrics.json", "splits.json", "export_manifest.json")]]
    guard = {str(p): sha256(p) for p in protected_paths if p.exists()}
    queries, gallery, encoder, reference, signature = prepare()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path, cache = output / "comparison.json", output / "paired_embeddings.npz"
    manifest = output / "experiment.json"
    if manifest.exists():
        _load(manifest, signature)
    else:
        if any(output.iterdir()):
            raise ValueError("Choose a new experiment output directory")
        _save(manifest, {"signature": signature})
    if report_path.exists():
        report = _load(report_path, signature)
        if sha256(cache) != report["cache_sha256"]:
            raise ValueError("Completed paired cache changed")
        print("Reuse completed comparison; no inference", flush=True)
        return report
    print(f"MVP: {MODEL_NAME}; {len(queries)} query / {len(gallery)} gallery; masks={signature['mask_policy']}", flush=True)
    vectors, metadata = paired_embeddings(encoder, queries + gallery, signature, output, device)
    scores, details = {}, {}
    for variant in ("original", "masked"):
        print(f"Official evaluation: {variant}", flush=True)
        scores[variant], details[variant] = evaluate_embeddings(queries, gallery, vectors[variant], signature["threshold_fixed"])
    for mode, expected in (("raw", reference["raw_baseline"]["validation"]), ("reranked", reference["validation"])):
        for key in ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "TP", "FP", "FN", "TN"):
            if not np.isclose(scores["original"][mode][key], expected[key], atol=1e-8, rtol=0):
                raise ValueError(f"Baseline reproduction failed: {mode}/{key}; do not interpret the mask delta")
    if any(sha256(Path(p)) != h for p, h in guard.items()):
        raise RuntimeError("MVP or frozen detector artifacts changed during the experiment")
    if any(sha256(DATASET / "images" / f"{i}.jpg") != h for i, h in signature["frames"].items()):
        raise RuntimeError("Validation source frames changed during the experiment")
    cosines = np.clip((vectors["original"] * vectors["masked"]).sum(axis=1), -1, 1)
    report = {"signature": signature, "purpose": "paired diagnostic, not training or validation hyperparameter search",
        "queries": len(queries), "gallery": len(gallery), "device": {"reid": "ONNX CPU", "detector": metadata["device"]},
        "baseline_reproduced": True, "protected_unchanged": True, "protected_sha256": guard,
        "cache_sha256": sha256(cache), "scores": scores, "per_query": details,
        "paired": {mode: paired_deltas(details["original"][mode], details["masked"][mode]) for mode in ("raw", "reranked")},
        "mask_statistics": {"crops": len(metadata["predictions"]),
            "crops_with_masks": sum(bool(p["rectangles"]) for p in metadata["predictions"].values()),
            "rectangles": sum(len(p["rectangles"]) for p in metadata["predictions"].values())},
        "embedding_cosine": {"mean": float(cosines.mean()), "min": float(cosines.min()),
                             "median": float(np.median(cosines))},
        "paired_extraction_seconds": metadata["elapsed_seconds"],
        "limitations": ["OSNet checkpoint was previously selected on this outer validation; not a new independent jury test",
            "Detector training/calibration/holdout identities and exact frames are disjoint from this ReID validation",
            "Masks applied to both query and gallery, before original 208x208 preprocessing; no original files modified",
            "Threshold and streaming reranking fixed; scores are not optimized for the masked distribution",
            "No proof of complete plate independence; detector can miss regions or overmask useful vehicle details"]}
    _save(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "runs/paired_v1")
    parser.add_argument("--device", default=None, help="Detector device; ReID always uses the MVP ONNX CPU encoder")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    report = run(args.output, args.device)
    print(json.dumps({"scores": report["scores"], "paired": report["paired"]}, ensure_ascii=False, indent=2))
