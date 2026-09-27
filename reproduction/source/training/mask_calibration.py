"""Validation-only mask calibration; frozen holdout evaluation, never training."""
from pathlib import Path
import time

import numpy as np

from backend.core import sha256
from training.audit import digest, validate_annotations
from training.mask_detection import compare_regions, restore_boxes, summarize
from training.mask_finetune import EXPERIMENT, export_reviewed, load_json
from training.stage6 import write_json
from training.yolo_masks import VERSION, device_name, runtime


CONFIDENCES = (.05, .10, .15, .20, .25, .35, .50)
MARGINS = (0., .03, .05, .075, .10, .15, .20)
LIMITS = {"outside_fraction_of_all_crops": .03, "outside_fraction_of_any_crop": .10}
INFERENCE = dict(imgsz=640, conf=min(CONFIDENCES), iou=.45, max_det=300,
                 rect=True, half=False, augment=False, save=False, verbose=False)
BASELINE = {"confidence": .25, "margin": 0.}


def _save(path, value):
    write_json(path, {**value, "fingerprint": digest(value)})


def _load(path, signature=None):
    value = load_json(path)
    fingerprint = value.pop("fingerprint", None)
    if digest(value) != fingerprint:
        raise ValueError(f"Changed result/cache: {path}")
    if signature is not None and value["signature"] != signature:
        raise ValueError(f"Different weights, data or settings; use a new directory: {path}")
    return value


def _context(weights, annotations, experiment):
    experiment, weights = Path(experiment), Path(weights)
    if not (experiment / "data/manifest.json").is_file():
        raise ValueError("Export reviewed annotations before calibration")
    manifest = export_reviewed(experiment / "annotation/mask_plan.json", annotations, experiment / "data")
    state = load_json(weights.parent.parent / "experiment.json")
    weight_hash = sha256(weights)
    if state["status"] != "complete" or state["best_sha256"] != weight_hash:
        raise ValueError("Use best.pt from a completed training run")
    for key in ("plan_fingerprint", "annotations_sha256"):
        if state["signature"][key] != manifest[key]:
            raise ValueError("Training run and reviewed data differ")
    plan = load_json(experiment / "annotation/mask_plan.json")
    manual = validate_annotations(plan, load_json(annotations))
    signature = {"weights_sha256": weight_hash, "annotations_sha256": manifest["annotations_sha256"],
        "plan_fingerprint": plan["fingerprint"], "ultralytics": VERSION,
        "calibration_code_sha256": sha256(Path(__file__)),
        "coverage_code_sha256": sha256(Path(__file__).with_name("mask_detection.py")),
        "inference": INFERENCE, "confidences": list(CONFIDENCES), "margins": list(MARGINS),
        "limits": LIMITS, "baseline": BASELINE,
        "selection": "max regions90, max covered pixels, min outside pixels, max confidence, min margin"}
    return plan, manual, signature


def _predict_split(weights, plan, split, path, signature, experiment, device=None):
    """Cache floating xyxy AFTER NMS, BEFORE rounding/dilation; only the requested split."""
    expected = {i for i, item in plan["images"].items() if item["split"] == split}
    if split not in ("val", "holdout") or not expected:
        raise ValueError("Need a non-empty val or holdout split")
    path = Path(path)
    cache_signature = {**signature, "split": split}
    cache = _load(path, cache_signature) if path.exists() else {
        "signature": cache_signature, "device": device or device_name(), "images": {}}
    if not set(cache["images"]).issubset(expected):
        raise ValueError("Prediction cache contains a different split")
    if set(cache["images"]) == expected:
        print(f"Reuse {split} predictions: {len(expected)} images; device={cache['device']}")
        return cache
    selected_device = device or device_name()
    if cache["device"] != selected_device:
        raise ValueError("Resume an incomplete prediction cache on its original device")
    model = runtime()(str(weights))
    folder = "val" if split == "val" else "test"
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for image_id in sorted(expected - set(cache["images"])):
        image_path = Path(experiment) / "data/images" / folder / f"{image_id}.png"
        result = model.predict(str(image_path), device=selected_device, **INFERENCE)[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(float)
        scores = result.boxes.conf.cpu().numpy().astype(float)
        if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                or len(boxes) != len(scores) or np.any((scores < 0) | (scores > 1))):
            raise ValueError("Invalid detector output")
        if len(scores) >= INFERENCE["max_det"]:
            raise ValueError("max_det reached; cached threshold filtering would be truncated")
        cache["images"][image_id] = {"xyxy": boxes.tolist(), "confidences": scores.tolist()}
        _save(path, cache)
        if len(cache["images"]) % 10 == 0 or len(cache["images"]) == len(expected):
            print(f"{split}: {len(cache['images'])}/{len(expected)} · {time.perf_counter()-started:.1f} s", flush=True)
    return cache


def score_predictions(plan, manual, raw, confidence, margin, split="val"):
    """The same fixed policy is used for both validation and holdout."""
    if not np.isfinite([confidence, margin]).all() or not 0 < confidence < 1 or not 0 <= margin <= .25:
        raise ValueError("Invalid confidence or margin")
    expected = {i for i, item in plan["images"].items() if item["split"] == split}
    if not expected or set(raw) != expected:
        raise ValueError("Predictions must cover exactly the requested split")
    per_image, predictions = {}, {}
    for image_id in sorted(expected):
        item, result = plan["images"][image_id], raw[image_id]
        scores = np.asarray(result["confidences"], dtype=float)
        boxes = np.asarray(result["xyxy"], dtype=float).reshape(-1, 4)
        keep = scores >= confidence
        boxes, scores = boxes[keep].copy(), scores[keep]
        grow = (boxes[:, 2:] - boxes[:, :2]) * margin
        boxes[:, :2] -= grow
        boxes[:, 2:] += grow
        rects, scores = restore_boxes(boxes, scores, 1., (0, 0), (item["width"], item["height"]))
        predictions[image_id] = {"reviewed": False, "rectangles": rects, "confidences": scores}
        per_image[image_id] = compare_regions((item["width"], item["height"]),
                                              manual[image_id]["rectangles"], rects, scores)
    stats = summarize(per_image)
    negatives = [i for i in expected if not manual[i]["rectangles"]]
    false_masks = sum(bool(predictions[i]["rectangles"]) for i in negatives)
    stats.update({"negative_images": len(negatives), "negative_images_with_false_masks": false_masks,
        "negative_image_false_mask_rate": false_masks / len(negatives) if negatives else None,
        "regions_covered_90pct_fraction": stats["regions_covered_90pct"] / stats["reference_regions"],
        "regions_zero_coverage": sum(c == 0 for v in per_image.values() for c in v["region_coverages"]),
        "crops_all_regions_covered_90pct": sum(bool(v["region_coverages"]) and min(v["region_coverages"]) >= .9
                                              for v in per_image.values()),
        "max_outside_reference_fraction_of_crop": max(v["outside_reference_pixels"] / v["crop_pixels"]
                                                      for v in per_image.values())})
    return {"coverage": stats, "per_image": per_image, "predictions": predictions}


def select_candidate(leaderboard, limits=LIMITS):
    feasible = [r for r in leaderboard if
        r["coverage"]["outside_reference_fraction_of_crop"] <= limits["outside_fraction_of_all_crops"] and
        r["coverage"]["max_outside_reference_fraction_of_crop"] <= limits["outside_fraction_of_any_crop"]]
    if not feasible:
        raise ValueError("No candidate satisfies the area limits; holdout remains locked")
    return max(feasible, key=lambda r: (r["coverage"]["regions_covered_90pct"],
        r["coverage"]["reference_pixel_coverage"], -r["coverage"]["outside_reference_fraction_of_crop"],
        r["confidence"], -r["margin"]))


def calibrate_masks(weights, annotations, output, device=None, experiment=EXPERIMENT):
    """49 validation policies; completed results are immutable and reusable."""
    plan, manual, signature = _context(weights, annotations, experiment)
    output = Path(output)
    result_path, raw_path = output / "calibration.json", output / "raw_val.json"
    if result_path.exists():
        report = _load(result_path, signature)
        if report["raw_sha256"] != sha256(raw_path):
            raise ValueError("Changed validation prediction cache")
        print("Reuse frozen calibration; no inference or training")
        return report
    if (output / "raw_holdout.json").exists() or (output / "holdout_final.json").exists():
        raise ValueError("Cannot recalibrate after holdout was opened")
    raw = _predict_split(weights, plan, "val", raw_path, signature, experiment, device)
    leaderboard = []
    for confidence in CONFIDENCES:
        for margin in MARGINS:
            result = score_predictions(plan, manual, raw["images"], confidence, margin)
            leaderboard.append({"confidence": confidence, "margin": margin, "coverage": result["coverage"]})
        print(f"Calibration: {len(leaderboard)}/{len(CONFIDENCES)*len(MARGINS)} policies", flush=True)
    selected = select_candidate(leaderboard)
    policy = {key: selected[key] for key in ("confidence", "margin")}
    report = {"signature": signature, "split": "val", "device": raw["device"],
        "raw_sha256": sha256(raw_path), "selected": policy, "leaderboard": leaderboard,
        "baseline": score_predictions(plan, manual, raw["images"], **BASELINE),
        "calibrated": score_predictions(plan, manual, raw["images"], **policy),
        "note": "Coverage, not detector mAP or ReID accuracy. Limits are pilot heuristics, not organizer requirements."}
    _save(result_path, report)
    return report


def evaluate_frozen_holdout(weights, annotations, calibration, confirm=False, device=None, experiment=EXPERIMENT):
    """No search on holdout. Compare frozen policy to the predeclared baseline once."""
    if confirm is not True:
        raise ValueError("Holdout requires explicit confirm=True after validation calibration")
    plan, manual, signature = _context(weights, annotations, experiment)
    calibration = Path(calibration)
    report = _load(calibration / "calibration.json", signature)
    if report["raw_sha256"] != sha256(calibration / "raw_val.json"):
        raise ValueError("Changed validation prediction cache")
    locked = {**signature, "calibration_sha256": sha256(calibration / "calibration.json")}
    output, raw_path = calibration / "holdout_final.json", calibration / "raw_holdout.json"
    if output.exists():
        result = _load(output, locked)
        if result["raw_sha256"] != sha256(raw_path):
            raise ValueError("Changed holdout prediction cache")
        print("Reuse the existing holdout result; no repeat evaluation")
        return result
    raw = _predict_split(weights, plan, "holdout", raw_path, locked, experiment, device)
    result = {"signature": locked, "split": "holdout", "device": raw["device"],
        "raw_sha256": sha256(raw_path), "selected": report["selected"],
        "baseline": score_predictions(plan, manual, raw["images"], split="holdout", **BASELINE),
        "calibrated": score_predictions(plan, manual, raw["images"], split="holdout", **report["selected"]),
        "note": "Do not tune settings using these results. ReID usefulness is not evaluated here."}
    _save(output, result)
    return result
