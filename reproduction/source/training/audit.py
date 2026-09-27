"""Read-only OSNet diagnostics and paired, manually reviewed anonymization audit."""
import base64
import hashlib
import html
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from backend.core import (DATASET, MODEL, PREPROCESS, ROOT, Encoder, bbox, crop_image,
                          normalize, preprocess, sha256)
from backend.evaluate import SEED, make_protocol
from backend.rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA, rerank_protocol
from backend.scoring import metrics, ranked_queries
from training.stage6 import audit_partitions, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_crop(row, dataset=DATASET):
    with Image.open(Path(dataset) / "images" / f"{row['image_id']}.jpg") as image:
        return crop_image(image, bbox(row))


def data_signature(rows, dataset=DATASET):
    return {"rows": rows, "frames": {
        row["image_id"]: sha256(Path(dataset) / "images" / f"{row['image_id']}.jpg")
        for row in rows}}


def _picture(crop):
    image = crop.copy()
    image.thumbnail((640, 480))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def fixed_reference(path=ROOT / "artifacts/baseline_metrics.json"):
    report = json.loads(Path(path).read_text())
    if report["model_sha256"] != sha256(MODEL):
        raise ValueError("Calibration report belongs to another model")
    split = json.loads((ROOT / "artifacts/splits.json").read_text())
    if split["train_csv_sha256"] != sha256(DATASET / "train.csv"):
        raise ValueError("Calibration report belongs to another train.csv")
    if report["evaluator"]["sha256"] != sha256(ROOT / "evaluate.py"):
        raise ValueError("Calibration report uses another evaluator")
    return float(report["threshold"])


def encode(encoder, rows, dataset=DATASET, annotations=None, batch_size=16):
    result = {}
    for start in tqdm(range(0, len(rows), batch_size), desc="OSNet audit embeddings"):
        batch = rows[start:start + batch_size]
        inputs = []
        for row in batch:
            crop = load_crop(row, dataset)
            if annotations is not None:
                crop = opaque_mask(crop, annotations[row["image_id"]]["rectangles"])
            inputs.append(preprocess(crop, (0, 0, *crop.size)))
        result.update(zip((r["image_id"] for r in batch), encoder.encode_batch(inputs)))
    return result


def evaluate_pair_protocol(queries, gallery, embeddings, threshold):
    raw = ranked_queries(queries, gallery, embeddings)
    reranked, _ = rerank_protocol(queries, gallery, embeddings)
    # MVP acceptance uses the maximum raw cosine, even after reranking.
    return {"raw": metrics(raw, threshold), "reranked": metrics(reranked, threshold)}, reranked


def error_audit(rows, split, output, dataset=DATASET, threshold=None):
    """Full outer validation for diagnosis, never training/checkpoint selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = data_signature(rows, dataset)
    if split.get("frame_sha256") != signature["frames"]:
        raise ValueError("Images changed since the saved split/calibration; rebuild that reference first")
    audit_partitions(rows, signature["frames"], split["identities"])
    queries, gallery = make_protocol(rows, split["identities"]["validation"], SEED)
    selected = queries + gallery
    encoder = Encoder()
    threshold = fixed_reference() if threshold is None else threshold
    fingerprint = digest({"data": signature, "query": queries, "gallery": gallery,
                          "model": encoder.model_sha256, "preprocess": PREPROCESS})
    cache = output / "validation_embeddings.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) != fingerprint:
                raise ValueError("Audit data/model changed; choose a new results directory")
            embeddings = dict(zip(saved["ids"].tolist(), normalize(saved["vectors"])))
    else:
        embeddings = encode(encoder, selected, dataset)
        with cache.with_suffix(".npz.tmp").open("wb") as stream:
            np.savez_compressed(stream, fingerprint=fingerprint,
                                ids=np.array(list(embeddings)), vectors=np.stack(list(embeddings.values())))
        cache.with_suffix(".npz.tmp").replace(cache)
    scores, ranked = evaluate_pair_protocol(queries, gallery, embeddings, threshold)
    by_id = {r["image_id"]: r for r in selected}
    errors, cards = [], []
    for position, query in enumerate(queries):
        qid = query["image_id"]
        predictions = ranked.predictions[qid]
        positives = [r for r in gallery if r["vehicle_id"] == query["vehicle_id"]
                     and r["camera_id"] != query["camera_id"]]
        correct = bool(positives) and predictions[0] in {r["image_id"] for r in positives}
        accepted = float(ranked.confidence[position]) >= threshold
        if (positives and (not correct or not accepted)) or (not positives and accepted):
            kind = ("unknown_accepted" if not positives else
                    "known_refused" if not accepted else "wrong_identity_accepted")
            entry = {"query_id": qid, "kind": kind, "top1_correct": correct,
                     "confidence": float(ranked.confidence[position]),
                     "crop_width": query["w"], "crop_height": query["h"],
                     "top5": predictions[:5], "positive_ids": [r["image_id"] for r in positives]}
            errors.append(entry)
            if len(cards) < 60:
                show = [qid] + predictions[:5] + entry["positive_ids"][:1]
                pictures = "".join(f'<figure><img width="160" src="{_picture(load_crop(by_id[i], dataset))}">'
                                   f'<figcaption>{html.escape(i)}</figcaption></figure>' for i in show)
                cards.append(f'<section><h3>{html.escape(qid)} · {kind} · {entry["confidence"]:.3f}</h3>'
                             f'<p>Query → Top-5 → первый cross-camera positive (если есть)</p>'
                             f'<div style="display:flex;flex-wrap:wrap">{pictures}</div></section>')
    report = {"purpose": "diagnostic_only_not_model_selection", "fingerprint": fingerprint,
              "model_sha256": encoder.model_sha256, "evaluator_sha256": sha256(ROOT / "evaluate.py"),
              "threshold": threshold, "metrics": scores, "queries": len(queries),
              "gallery": len(gallery), "error_count": len(errors), "errors": errors}
    write_json(output / "errors.json", report)
    (output / "errors.html").write_text('<!doctype html><meta charset="utf-8"><title>OSNet errors</title>'
        '<h1>OSNet: диагностика outer validation</h1><p>Первые 60 ошибок; все записи — errors.json.</p>'
        + "".join(cards), encoding="utf-8")
    return report


def prepare_mask_audit(rows, split, output, dataset=DATASET, identities=32, seed=SEED):
    """Small RANDOM identity subset, not the worst errors; annotate every crop in it."""
    available = sorted(split["identities"]["validation"])
    if identities is not None and not 5 <= identities <= len(available):
        raise ValueError("Mask audit requires 5..N validation identities, or None for all")
    selected = available if identities is None else sorted(random.Random(seed).sample(available, identities))
    queries, gallery = make_protocol(rows, selected, seed)
    signature = data_signature(queries + gallery, dataset)
    frames = {}
    for row in queries + gallery:
        crop = load_crop(row, dataset)
        frames[row["image_id"]] = {"frame_sha256": signature["frames"][row["image_id"]],
                                   "bbox": list(bbox(row)), "width": crop.width, "height": crop.height}
    plan = {"version": 1, "coordinates": "crop_xyxy_pixels_exclusive", "seed": seed,
            "purpose": "paired_robustness_not_full_validation_score", "identities": selected,
            "query_ids": [r["image_id"] for r in queries],
            "gallery_ids": [r["image_id"] for r in gallery], "images": frames}
    plan["fingerprint"] = digest(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "mask_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Mask sample/data changed; choose a new results directory")
    write_json(path, plan)
    write_mask_annotator(plan, output, dataset)
    return plan


def write_mask_annotator(plan, output, dataset=DATASET):
    """Refresh the offline UI without changing the plan or crop-relative masks."""
    pictures, full_frames = {}, {}
    for image_id, expected in plan["images"].items():
        path = Path(dataset) / "images" / f"{image_id}.jpg"
        if sha256(path) != expected["frame_sha256"]:
            raise ValueError("Image changed since annotation plan")
        with Image.open(path) as source:
            frame = ImageOps.exif_transpose(source).convert("RGB")
            crop = crop_image(frame, expected["bbox"])
        if crop.size != (expected["width"], expected["height"]):
            raise ValueError("Crop changed since annotation plan")
        pictures[image_id] = _picture(crop)
        full_frames[image_id] = {"picture": _picture(frame),
                                 "width": frame.width, "height": frame.height}
    template = (Path(__file__).parent / "mask_annotator.html").read_text()
    # IDs and JSON are escaped; generated HTML works offline without an API/server.
    payload = json.dumps({"plan": plan, "pictures": pictures, "frames": full_frames}).replace("<", "\\u003c")
    (Path(output) / "annotate_masks.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")


def validate_annotations(plan, annotations):
    if annotations.get("fingerprint") != plan["fingerprint"]:
        raise ValueError("Annotation fingerprint does not match this sample")
    images = annotations.get("images", {})
    if set(images) != set(plan["images"]):
        raise ValueError("Review every selected query AND gallery crop before the paired audit")
    for image_id, expected in plan["images"].items():
        item = images[image_id]
        if item.get("reviewed") is not True or not isinstance(item.get("rectangles"), list):
            raise ValueError(f"Unreviewed crop: {image_id}")
        for rect in item["rectangles"]:
            if (not isinstance(rect, list) or len(rect) != 4
                    or any(type(v) is not int for v in rect)):
                raise ValueError(f"Rectangles must be integer xyxy: {image_id}")
            x1, y1, x2, y2 = rect
            if not (0 <= x1 < x2 <= expected["width"] and 0 <= y1 < y2 <= expected["height"]):
                raise ValueError(f"Rectangle outside crop: {image_id}")
    return images


def save_annotations(content, plan, path):
    annotations = json.loads(bytes(content).decode("utf-8"))
    validate_annotations(plan, annotations)
    write_json(path, annotations)
    return annotations


def opaque_mask(crop, rectangles):
    """Never modify source image; no inpainting, OCR, proxy or guessed plate bbox."""
    masked = crop.copy()
    for x1, y1, x2, y2 in rectangles:
        masked.paste((0, 0, 0), (x1, y1, x2, y2))
    return masked


def paired_mask_audit(rows, plan, annotations, output, dataset=DATASET, threshold=None):
    images = validate_annotations(plan, annotations)
    by_id = {row["image_id"]: row for row in rows}
    selected = [by_id[i] for i in plan["images"]]
    current = data_signature(selected, dataset)
    for row in selected:
        expected = plan["images"][row["image_id"]]
        if current["frames"][row["image_id"]] != expected["frame_sha256"] or list(bbox(row)) != expected["bbox"]:
            raise ValueError("Image or crop changed since annotation")
    query = [by_id[i] for i in plan["query_ids"]]
    gallery = [by_id[i] for i in plan["gallery_ids"]]
    encoder = Encoder()
    threshold = fixed_reference() if threshold is None else threshold
    original = encode(encoder, selected, dataset)
    masked = encode(encoder, selected, dataset, images)
    before, ranks_before = evaluate_pair_protocol(query, gallery, original, threshold)
    after, ranks_after = evaluate_pair_protocol(query, gallery, masked, threshold)
    changed = [i for i in plan["images"] if images[i]["rectangles"]]
    similarities = {i: float(np.clip(original[i] @ masked[i], -1, 1)) for i in changed}
    report = {"purpose": plan["purpose"], "sample_fingerprint": plan["fingerprint"],
              "model_sha256": encoder.model_sha256, "evaluator_sha256": sha256(ROOT / "evaluate.py"),
              "annotations_sha256": digest(annotations), "threshold_fixed": threshold,
              "reranking_fixed": {"k1": ACTIVE_K1, "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA},
              "reviewed_crops": len(images), "masked_crops": len(changed),
              "query_count": len(query), "gallery_count": len(gallery),
              "before": before, "after": after,
              "delta_mAP_at_10": {mode: after[mode]["mAP_at_10"] - before[mode]["mAP_at_10"]
                                  for mode in ("raw", "reranked")},
              "embedding_cosines": similarities,
              "top1_changed": [i for i in plan["query_ids"]
                               if ranks_before.predictions[i][0] != ranks_after.predictions[i][0]],
              "limitation": "Small locally annotated sample; not proof of plate independence or jury-test performance"}
    if not changed:
        raise ValueError("No masked regions: this is not a masking robustness experiment")
    write_json(Path(output) / "mask_results.json", report)
    return report
