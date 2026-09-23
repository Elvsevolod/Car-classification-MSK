"""Inference-only DeepMosaics/EgoBlur audit. No restoration, training, or MVP writes."""
import argparse
import gc
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # Registers the torchvision operators used by EgoBlur TorchScript.

from backend.core import DATASET, MODEL, ROOT, Encoder, bbox, read_rows, sha256
from training.audit import digest, encode, evaluate_pair_protocol, fixed_reference, load_crop, validate_annotations
from training.mask_detection import REFERENCE, compare_regions, restore_boxes, summarize, write_comparison
from training.stage6 import write_json

EXPERIMENT = ROOT / "Mask-Detectors/benchmark_02_mosaic_egoblur"
SOURCES = ROOT / "artifacts/mask_detector_sources"


def mosaic_input(crop):
    """Publisher's BGR / 255, shortest side 360; no ImageNet normalization."""
    width, height = crop.size
    size = (int(360 * width / height), 360) if width >= height else (360, int(360 * height / width))
    bgr = np.asarray(crop.convert("RGB"))[:, :, ::-1].copy()
    pixels = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR) if min(width, height) != 360 else bgr
    return torch.from_numpy((pixels / 255.).transpose(2, 0, 1).copy()).float()[None]


def mosaic_rectangles(probability, size, expanded=False):
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Invalid DeepMosaics probability map")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("DeepMosaics must return sigmoid probabilities")
    # Preserve the author's uint8 quantization before thresholding.
    gray = (probability * 255).clip(0, 255).astype(np.uint8)
    mask = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)[1]
    if expanded:
        kernel = max(1, int(min(size) / 20))
        mask = cv2.blur(mask, (kernel, kernel))
        mask = cv2.threshold(mask, 64 / 5, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    probability = cv2.resize(probability, size, interpolation=cv2.INTER_LINEAR)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rectangles, scores = [], []
    for label in range(1, count):
        x, y, width, height, _ = stats[label].tolist()
        rectangles.append([x, y, x + width, y + height])
        scores.append(float(probability[labels == label].mean()))
    order = np.argsort(-np.asarray(scores), kind="stable")
    return {"reviewed": False, "rectangles": [rectangles[i] for i in order],
            "confidences": [scores[i] for i in order]}


def egoblur_input(crop):
    return torch.from_numpy(np.asarray(crop.convert("RGB"))[:, :, ::-1].transpose(2, 0, 1).copy())


def egoblur_rectangles(output, size):
    boxes, labels, scores, dims = output
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),):
        raise ValueError("Unexpected EgoBlur output shape")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("Non-finite EgoBlur prediction")
    indices = torchvision.ops.nms(boxes, scores, .3)
    indices = indices[scores[indices] > .25]
    rectangles, confidences = restore_boxes(boxes[indices].cpu().numpy(), scores[indices].cpu().numpy(),
                                            1., (0, 0), size)
    return {"reviewed": False, "rectangles": rectangles, "confidences": confidences}


def combine_predictions(left, right):
    if set(left) != set(right):
        raise ValueError("Different crop sets in face/plate predictions")
    combined = {}
    for image_id in left:
        pairs = list(zip(left[image_id]["rectangles"], left[image_id]["confidences"]))
        pairs += list(zip(right[image_id]["rectangles"], right[image_id]["confidences"]))
        pairs.sort(key=lambda pair: -pair[1])
        combined[image_id] = {"reviewed": False, "rectangles": [p[0] for p in pairs],
                              "confidences": [p[1] for p in pairs]}
    return combined


def load_model(entry):
    path = EXPERIMENT / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Weight checksum mismatch: {entry['id']}")
    if entry["id"] == "deepmosaics":
        source = SOURCES / "DeepMosaics"
        for name, expected in entry["source_files"].items():
            if sha256(source / name) != expected:
                raise ValueError(f"DeepMosaics source checksum mismatch: {name}")
        # Isolate the upstream package name; do not shadow the project's models directory.
        package_name = "_audit_deepmosaics_models"
        spec = importlib.util.spec_from_file_location(package_name, source / "models/__init__.py",
                       submodule_search_locations=[str(source / "models")])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        from importlib import import_module
        module = import_module(package_name + ".BiSeNet_model")
        model = module.BiSeNet(num_classes=1, context_path="resnet18", train_flag=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        model = torch.jit.load(str(path), map_location="cpu")
    return model.eval()


def predict(entry, crops):
    model = load_model(entry)
    mosaic = entry["id"] == "deepmosaics"
    prepare = mosaic_input if mosaic else egoblur_input
    predictions = {"deepmosaics_raw": {}, "deepmosaics_published": {}} if mosaic else {entry["id"]: {}}
    durations = []
    with torch.inference_mode():
        model(prepare(next(iter(crops.values()))))
        for index, (image_id, crop) in enumerate(crops.items(), 1):
            start = time.perf_counter()
            output = model(prepare(crop))
            if mosaic:
                probability = output[0, 0].cpu().numpy()
                predictions["deepmosaics_raw"][image_id] = mosaic_rectangles(probability, crop.size)
                predictions["deepmosaics_published"][image_id] = mosaic_rectangles(probability, crop.size, True)
            else:
                predictions[entry["id"]][image_id] = egoblur_rectangles(output, crop.size)
            durations.append(time.perf_counter() - start)
            if index % 10 == 0 or index == len(crops):
                elapsed = sum(durations)
                print(f"{entry['id']}: {index}/{len(crops)} | elapsed {elapsed:.1f}s | "
                      f"ETA {(len(crops)-index)*elapsed/index:.1f}s", flush=True)
    del model
    gc.collect()
    timing = {"mean_ms": float(np.mean(durations) * 1000), "total_seconds": sum(durations),
              "scope": "CPU 2 threads; excludes load/warmup/IO; DeepMosaics includes both postprocess variants"}
    return predictions, timing


def run(output, model_names):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; previous results are preserved")
    torch.set_num_threads(2)
    manifest = json.loads((EXPERIMENT / "models.json").read_text())
    if not set(model_names).issubset({entry["id"] for entry in manifest["models"]}):
        raise ValueError("Requested model is not yet present in the verified manifest")
    plan = json.loads((REFERENCE / "mask_plan.json").read_text())
    annotations = json.loads((REFERENCE / "masks.json").read_text())
    manual = validate_annotations(plan, annotations)
    if digest({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("Plan fingerprint mismatch")
    previous_path = ROOT / "Mask-Detectors/benchmark_01_pretrained/results/benchmark.json"
    previous = json.loads(previous_path.read_text())
    if previous["sample_fingerprint"] != plan["fingerprint"] or previous["manual_annotations_sha256"] != digest(annotations):
        raise ValueError("Previous benchmark belongs to different annotations/sample")
    threshold, encoder = fixed_reference(), Encoder()
    if previous["reid"]["model_sha256"] != encoder.model_sha256 or previous["reid"]["threshold"] != threshold:
        raise ValueError("ReID model/calibration changed since the control experiment")
    guard_paths = [REFERENCE / name for name in ("masks.json", "mask_plan.json", "mask_results.json")]
    guard_paths += [MODEL, ROOT / "artifacts/baseline_metrics.json", previous_path]
    guard = {str(p): sha256(p) for p in guard_paths}
    by_id = {r["image_id"]: r for r in read_rows(DATASET / "train.csv")}
    selected = [by_id[i] for i in plan["images"]]
    crops = {}
    for row in selected:
        image_id = row["image_id"]
        expected = plan["images"][image_id]
        if (sha256(DATASET / "images" / f"{image_id}.jpg") != expected["frame_sha256"]
                or list(bbox(row)) != expected["bbox"]):
            raise ValueError(f"Frame/bbox changed since annotation: {image_id}")
        crops[image_id] = load_crop(row)
        if crops[image_id].size != (expected["width"], expected["height"]):
            raise ValueError(f"Crop dimensions changed: {image_id}")
    queries, gallery = ([by_id[i] for i in plan[key]] for key in ("query_ids", "gallery_ids"))
    original = encode(encoder, selected)
    before, _ = evaluate_pair_protocol(queries, gallery, original, threshold)
    if before != previous["reid"]["original"]:
        raise ValueError("Recomputed ReID baseline differs from previous control")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnostic_audit_not_training_or_threshold_selection", "status": "running",
        "sample_fingerprint": plan["fingerprint"], "annotations_sha256": digest(annotations),
        "manifest": manifest, "requested_models": model_names, "code_sha256": sha256(Path(__file__)),
        "comparison_helpers_sha256": sha256(ROOT / "training/mask_detection.py"),
        "prior_benchmark_sha256": guard[str(previous_path)], "reid_control": previous["reid"],
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "opencv": cv2.__version__, "device": "cpu"},
        "detectors": {}}
    output.mkdir(parents=True, exist_ok=True)
    all_predictions = {}

    def evaluate(name, predictions, timing):
        all_predictions[name] = predictions
        write_json(output / f"{name}_predictions.json", {"version": 1,
            "source": "automatic_prediction_not_reviewed", "fingerprint": plan["fingerprint"],
            "coordinates": plan["coordinates"], "model": name, "images": predictions})
        per_image = {i: compare_regions(crops[i].size, manual[i]["rectangles"], p["rectangles"], p["confidences"])
                     for i, p in predictions.items()}
        masked = encode(encoder, selected, annotations=predictions)
        metrics, _ = evaluate_pair_protocol(queries, gallery, masked, threshold)
        report["detectors"][name] = {"summary": summarize(per_image), "per_image": per_image,
                                    "latency": timing, "reid": metrics}
        print(json.dumps({"model": name, **summarize(per_image)}, ensure_ascii=False), flush=True)
        write_json(output / "benchmark.json", report)

    for entry in manifest["models"]:
        if entry["id"] in model_names:
            variants, timing = predict(entry, crops)
            for name, predictions in variants.items():
                evaluate(name, predictions, timing)
    if {"egoblur_lp", "egoblur_face"}.issubset(all_predictions):
        combined = combine_predictions(all_predictions["egoblur_lp"], all_predictions["egoblur_face"])
        evaluate("egoblur_combined", combined, {"scope": "union of face/plate predictions; requires both models"})
    if any(sha256(Path(p)) != checksum for p, checksum in guard.items()):
        raise RuntimeError("A protected reference/model file changed during the benchmark")
    report["reference_unchanged"], report["status"] = True, "complete"
    write_json(output / "benchmark.json", report)
    page = output / "comparison.html"
    write_comparison(page, crops, manual, all_predictions)
    page.write_text(page.read_text().replace("Порог 0.25, без расширения рамок.",
        "DeepMosaics: порог 64/255, raw и published (расширение автора); EgoBlur: порог 0.25 без расширения."))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results")
    parser.add_argument("--models", nargs="+", choices=["deepmosaics", "egoblur_lp", "egoblur_face"],
                        default=["deepmosaics", "egoblur_lp", "egoblur_face"])
    args = parser.parse_args()
    run(args.output, args.models)
