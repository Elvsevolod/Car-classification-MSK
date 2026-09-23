"""Frozen pretrained-detector comparison; never writes to the manual reference or MVP."""
import argparse
import html
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.core import DATASET, ROOT, Encoder, bbox, read_rows, sha256
from backend.rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA
from training.audit import (_picture, digest, encode, evaluate_pair_protocol,
                            fixed_reference, load_crop, validate_annotations)
from training.stage6 import write_json

EXPERIMENT = ROOT / "Mask-Detectors/benchmark_01_pretrained"
REFERENCE = ROOT / "OSNet-AIN-x1.0/audit_07_errors_masks/results"


def letterbox(image, size=640):
    import cv2

    width, height = image.size
    gain = min(size / width, size / height)
    resized = (round(width * gain), round(height * gain))
    dw, dh = (size - resized[0]) / 2, (size - resized[1]) / 2
    left, top = round(dw - .1), round(dh - .1)
    pixels = cv2.resize(np.asarray(image.convert("RGB")), resized, interpolation=cv2.INTER_LINEAR)
    pixels = cv2.copyMakeBorder(pixels, top, round(dh + .1), left, round(dw + .1),
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, gain, (dw, dh), (left, top)


def box_iou(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    box = np.asarray(box, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:])
                               - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-12)


def nms(boxes, scores, threshold):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        order = order[1:][box_iou(boxes[current], boxes[order[1:]]) <= threshold]
    return keep


def decode(output, model_format, confidence=.25, nms_iou=.45):
    output = np.asarray(output)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite detector output")
    if model_format == "yolov9_end2end":
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2 or output.shape[1] != 7:
            raise ValueError("YOLOv9 expected N x 7: batch, xyxy, class, confidence")
        if np.any(output[:, 0] != 0) or np.any(output[:, 5] != 0):
            raise ValueError("Unexpected batch/class in single-image single-class detector")
        selected = output[output[:, 6] >= confidence]
        boxes, scores = selected[:, 1:5], selected[:, 6]
    elif model_format == "yolo11_raw":
        if output.ndim != 3 or output.shape[:2] != (1, 5):
            raise ValueError("YOLO11 expected 1 x 5 x N: xywh, single-class confidence")
        selected = output[0].T[output[0, 4] >= confidence]
        centers, sizes = selected[:, :2], selected[:, 2:4]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
        scores = selected[:, 4]
        keep = nms(boxes, scores, nms_iou)
        boxes, scores = boxes[keep], scores[keep]
    else:
        raise ValueError(f"Unsupported detector format: {model_format}")
    order = np.argsort(-scores, kind="stable")
    return boxes[order], scores[order]


def restore_boxes(boxes, scores, gain, padding, size):
    boxes = (np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
             - np.tile(padding, 2)) / gain
    # Cover fractional boundary pixels, without tunable dilation/margins.
    boxes[:, :2] = np.floor(boxes[:, :2])
    boxes[:, 2:] = np.ceil(boxes[:, 2:])
    boxes = np.clip(boxes, 0, np.tile(size, 2)).astype(int)
    valid = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    return boxes[valid].tolist(), np.asarray(scores)[valid].astype(float).tolist()


def compare_regions(size, manual, predicted, scores, match_iou=.5, coverage_threshold=.9):
    """Confidence-ordered one-to-one matching AND pixel-union coverage."""
    used, matches = set(), []
    for index in np.argsort(-np.asarray(scores), kind="stable"):
        overlaps = box_iou(predicted[index], manual)
        available = [i for i in range(len(manual)) if i not in used]
        if available:
            best = max(available, key=lambda i: overlaps[i])
            if overlaps[best] >= match_iou:
                used.add(best)
                matches.append({"prediction": int(index), "reference": best,
                                "iou": float(overlaps[best])})
    width, height = size
    reference_mask, prediction_mask = (np.zeros((height, width), dtype=bool) for _ in range(2))
    for mask, rectangles in ((reference_mask, manual), (prediction_mask, predicted)):
        for x1, y1, x2, y2 in rectangles:
            mask[y1:y2, x1:x2] = True
    covered = [float(prediction_mask[y1:y2, x1:x2].mean()) for x1, y1, x2, y2 in manual]
    return {"reference_regions": len(manual), "predicted_regions": len(predicted),
            "matched_regions": len(matches), "matches": matches,
            "reference_pixels": int(reference_mask.sum()),
            "predicted_pixels": int(prediction_mask.sum()),
            "intersection_pixels": int((reference_mask & prediction_mask).sum()),
            "outside_reference_pixels": int((prediction_mask & ~reference_mask).sum()),
            "crop_pixels": width * height, "region_coverages": covered,
            "regions_covered_90pct": sum(c >= coverage_threshold for c in covered)}


def summarize(per_image):
    fields = ("reference_regions", "predicted_regions", "matched_regions", "reference_pixels",
              "predicted_pixels", "intersection_pixels", "outside_reference_pixels", "crop_pixels",
              "regions_covered_90pct")
    totals = {key: sum(item[key] for item in per_image.values()) for key in fields}
    matched, predicted, reference = (totals[k] for k in
                                     ("matched_regions", "predicted_regions", "reference_regions"))
    totals.update({"crops": len(per_image),
        "crops_with_predictions": sum(x["predicted_regions"] > 0 for x in per_image.values()),
        "unmatched_predictions": predicted - matched, "missed_reference_regions": reference - matched,
        "agreement_precision_at_05": matched / predicted if predicted else 0.,
        "agreement_recall_at_05": matched / reference if reference else None,
        "agreement_f1_at_05": 2 * matched / (predicted + reference) if predicted + reference else 0.,
        "reference_pixel_coverage": totals["intersection_pixels"] / totals["reference_pixels"],
        "outside_reference_fraction_of_crop": totals["outside_reference_pixels"] / totals["crop_pixels"],
        "outside_reference_fraction_of_prediction": (totals["outside_reference_pixels"] / totals["predicted_pixels"]
                                                     if totals["predicted_pixels"] else 0.)})
    return totals


def predict_crops(model, crops, config):
    model_path = EXPERIMENT / model["path"]
    if sha256(model_path) != model["sha256"]:
        raise ValueError(f"Detector checksum mismatch: {model['id']}")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # A single unmeasured warmup; no detector optimization/tuning.
    session.run(None, {input_name: letterbox(next(iter(crops.values())), config["input_size"])[0]})
    predictions, durations = {}, []
    for index, (image_id, crop) in enumerate(crops.items(), 1):
        start = time.perf_counter()
        tensor, gain, fractional_pad, integer_pad = letterbox(crop, config["input_size"])
        output, = session.run(None, {input_name: tensor})
        boxes, scores = decode(output, model["format"], config["confidence"], model["nms"]["iou"])
        padding = fractional_pad if model["format"] == "yolov9_end2end" else integer_pad
        rectangles, confidences = restore_boxes(boxes, scores, gain, padding, crop.size)
        durations.append(time.perf_counter() - start)
        predictions[image_id] = {"reviewed": False, "rectangles": rectangles, "confidences": confidences}
        if index % 25 == 0 or index == len(crops):
            print(f"{model['id']}: {index}/{len(crops)}", flush=True)
    latency = {"mean_ms": float(np.mean(durations) * 1000), "median_ms": float(np.median(durations) * 1000),
               "p95_ms": float(np.percentile(durations, 95) * 1000), "total_seconds": sum(durations),
               "scope": "CPU 2 threads, preprocess + inference + postprocess; excludes disk IO, session init, warmup"}
    return predictions, latency


def write_comparison(path, crops, manual, detectors):
    cards = []
    for image_id, crop in crops.items():
        panes = []
        for name, predictions in detectors.items():
            item = predictions[image_id]
            shapes = []
            for color, rects in (("#00bb44", manual[image_id]["rectangles"]), ("#ff3344", item["rectangles"])):
                for x1, y1, x2, y2 in rects:
                    shapes.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
                                  f'fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/>')
            panes.append(f'<div><h3>{html.escape(name)} · {len(item["rectangles"])} рамок</h3>'
                f'<svg viewBox="0 0 {crop.width} {crop.height}" width="440">'
                f'<image href="{_picture(crop)}" width="{crop.width}" height="{crop.height}"/>'
                + "".join(shapes) + '</svg></div>')
        cards.append(f'<section><h2>{html.escape(image_id)}</h2><div class="row">' + "".join(panes) + '</div></section>')
    path.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение масок</title>'
        '<style>body{font:16px sans-serif;margin:24px}.row{display:flex;flex-wrap:wrap;gap:24px}'
        'section{border-top:1px solid #ccc}svg{max-width:100%;height:auto}</style>'
        '<h1>126 кропов: ручные и автоматические маски</h1>'
        '<p>Зелёный — ручной эталон; красный — непроверенное предсказание. Только целевая машина из bbox. '
        'Порог 0.25, без расширения рамок. Разметку эта страница не меняет.</p>'
        + "".join(cards) + '</html>', encoding="utf-8")


def run(output):
    import cv2

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier benchmark results")
    config = json.loads((EXPERIMENT / "models.json").read_text())
    plan = json.loads((REFERENCE / "mask_plan.json").read_text())
    annotations = json.loads((REFERENCE / "masks.json").read_text())
    manual = validate_annotations(plan, annotations)
    if digest({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("Annotation plan fingerprint mismatch")
    protected = [REFERENCE / "masks.json", REFERENCE / "mask_plan.json", REFERENCE / "mask_results.json"]
    guard = {str(p): sha256(p) for p in protected}
    by_id = {r["image_id"]: r for r in read_rows(DATASET / "train.csv")}
    selected = [by_id[i] for i in plan["images"]]
    crops = {}
    for row in selected:
        image_id = row["image_id"]
        expected = plan["images"][image_id]
        if (sha256(DATASET / "images" / f"{image_id}.jpg") != expected["frame_sha256"]
                or list(bbox(row)) != expected["bbox"]):
            raise ValueError(f"Frame/bbox changed since manual annotation: {image_id}")
        crops[image_id] = load_crop(row)
        if crops[image_id].size != (expected["width"], expected["height"]):
            raise ValueError(f"Crop dimensions changed: {image_id}")
    queries = [by_id[i] for i in plan["query_ids"]]
    gallery = [by_id[i] for i in plan["gallery_ids"]]
    threshold, encoder = fixed_reference(), Encoder()
    original = encode(encoder, selected)
    before, _ = evaluate_pair_protocol(queries, gallery, original, threshold)
    manual_embeddings = encode(encoder, selected, annotations=manual)
    manual_metrics, _ = evaluate_pair_protocol(queries, gallery, manual_embeddings, threshold)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnostic_audit_not_training_or_threshold_selection",
        "sample_fingerprint": plan["fingerprint"], "manual_annotations_sha256": digest(annotations),
        "reference_file_sha256": sha256(REFERENCE / "masks.json"),
        "code_sha256": sha256(Path(__file__)), "detector_config": config,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "onnxruntime": ort.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
        "reid": {"model_sha256": encoder.model_sha256, "threshold": threshold,
                 "evaluator_sha256": sha256(ROOT / "evaluate.py"),
                 "reranking": {"k1": ACTIVE_K1, "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA},
                 "query_count": len(queries), "gallery_count": len(gallery),
                 "original": before, "manual": manual_metrics},
        "limitations": ["Manual regions have no classes; agreement is not plate-specific precision/recall",
                        "All 126 crops have manual masks; negative-image specificity is not measurable",
                        "Small audit gallery; ReID scores are not full validation results",
                        "Pretrained plate detectors are not trained to detect arbitrary anonymization blur"],
        "detectors": {}}
    output.mkdir(parents=True, exist_ok=True)
    all_predictions = {}
    for model in config["models"]:
        predictions, latency = predict_crops(model, crops, config)
        all_predictions[model["id"]] = predictions
        per_image = {i: compare_regions(crops[i].size, manual[i]["rectangles"], p["rectangles"],
            p["confidences"], config["match_iou"], config["region_coverage_threshold"])
            for i, p in predictions.items()}
        write_json(output / f"{model['id']}_predictions.json", {"version": 1,
            "source": "automatic_prediction_not_reviewed", "coordinates": plan["coordinates"],
            "fingerprint": plan["fingerprint"], "detector": model,
            "confidence_threshold": config["confidence"], "images": predictions})
        masked = encode(encoder, selected, annotations=predictions)
        after, _ = evaluate_pair_protocol(queries, gallery, masked, threshold)
        cosines = [float(np.clip(original[i] @ masked[i], -1, 1)) for i in predictions]
        report["detectors"][model["id"]] = {"summary": summarize(per_image), "latency": latency,
            "per_image": per_image, "reid": after,
            "embedding_cosine_mean": float(np.mean(cosines)), "embedding_cosine_min": min(cosines)}
        print(json.dumps({"detector": model["id"], **summarize(per_image)}, ensure_ascii=False), flush=True)
    if any(sha256(Path(p)) != checksum for p, checksum in guard.items()):
        raise RuntimeError("Manual reference files changed during benchmark")
    report["reference_unchanged"] = True
    write_json(output / "benchmark.json", report)
    write_comparison(output / "comparison.html", crops, manual, all_predictions)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results")
    run(parser.parse_args().output)
