"""Versioned references and reviewed-only, frame/identity-disjoint YOLO data."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from backend.core import DATASET, ROOT, bbox, crop_image, read_rows, sha256
from backend.evaluate import make_splits
from training.audit import _picture, digest, validate_annotations
from training.mask_detection import REFERENCE, compare_regions, summarize
from training.stage6 import audit_partitions, write_json

EXPERIMENT = ROOT / "YOLO11/variant_01_anonymized_regions"
SEED = 20260920


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_plan(plan):
    if digest({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("Plan fingerprint mismatch")


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output is not empty; preserve it and choose a new version: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def revise_reference(source, output):
    """Rescore saved predictions only; do not rerun models or replace v1."""
    plan = load_json(REFERENCE / "mask_plan.json")
    check_plan(plan)
    old, new = load_json(REFERENCE / "masks.json"), load_json(source)
    validate_annotations(plan, new)
    protected = {p: sha256(p) for p in (REFERENCE / "masks.json", REFERENCE / "mask_plan.json")}
    reports = {}
    for directory in (ROOT / "Mask-Detectors/benchmark_01_pretrained/results",
                      ROOT / "Mask-Detectors/benchmark_02_mosaic_egoblur/results"):
        for path in sorted(directory.glob("*_predictions.json")):
            prediction = load_json(path)
            if prediction["fingerprint"] != plan["fingerprint"] or set(prediction["images"]) != set(new["images"]):
                raise ValueError(f"Different prediction sample: {path}")
            per_image = {i: compare_regions((p["width"], p["height"]), new["images"][i]["rectangles"],
                        prediction["images"][i]["rectangles"], prediction["images"][i]["confidences"])
                         for i, p in plan["images"].items()}
            reports[path.stem] = {"source": str(path.relative_to(ROOT)), "sha256": sha256(path),
                                  "summary": summarize(per_image), "per_image": per_image}
    changes = {i: {"before": old["images"][i]["rectangles"], "after": item["rectangles"]}
               for i, item in new["images"].items() if item != old["images"][i]}
    report = {"version": 2, "source_name": Path(source).name, "source_sha256": sha256(source),
              "previous_sha256": sha256(REFERENCE / "masks.json"), "fingerprint": plan["fingerprint"],
              "images": len(new["images"]), "regions": sum(len(v["rectangles"]) for v in new["images"].values()),
              "changed_images": changes, "detectors": reports,
              "scope": "rectangle agreement only; old ReID scores are not recomputed for revised masks"}
    output = new_directory(output)
    (output / "masks.json").write_bytes(Path(source).read_bytes())
    write_json(output / "mask_plan.json", plan)
    write_json(output / "comparison.json", report)
    if any(sha256(p) != h for p, h in protected.items()):
        raise RuntimeError("Original reference changed")
    return report


def select_rows(rows, hashes, counts, seed=SEED):
    """Partition connected identities first, then balance cameras/crop sizes without model scores."""
    partitions = make_splits(rows, hashes, seed)
    partitions = dict(zip(("train", "val", "holdout"), partitions.values()))
    audit_partitions(rows, hashes, partitions)
    selected = {}
    rng = random.Random(seed)
    for name, count in counts.items():
        candidates = [r for r in rows if r["vehicle_id"] in set(partitions[name])]
        rng.shuffle(candidates)
        used_frames, used_identities, strata = set(), Counter(), Counter()
        chosen = []
        def bucket(row):
            return (row["camera_id"], 0 if min(row["w"], row["h"]) < 160 else
                    1 if min(row["w"], row["h"]) < 320 else 2)
        while len(chosen) < count:
            available = [r for r in candidates if hashes[r["image_id"]] not in used_frames]
            if not available:
                raise ValueError(f"Not enough distinct frames for {name}: {len(chosen)} < {count}")
            row = min(available, key=lambda r: (used_identities[r["vehicle_id"]], strata[bucket(r)]))
            chosen.append(row)
            used_frames.add(hashes[row["image_id"]])
            used_identities[row["vehicle_id"]] += 1
            strata[bucket(row)] += 1
        selected[name] = chosen
    return selected, partitions


def verify_sample(plan, dataset=DATASET):
    check_plan(plan)
    if sha256(Path(dataset) / "train.csv") != plan["train_csv_sha256"]:
        raise ValueError("train.csv changed")
    by_id = {r["image_id"]: r for r in read_rows(Path(dataset) / "train.csv")}
    frames, identities = {}, {}
    for image_id, item in plan["images"].items():
        row = by_id[image_id]
        if list(bbox(row)) != item["bbox"] or row["vehicle_id"] != item["vehicle_id"]:
            raise ValueError(f"Annotation source changed: {image_id}")
        if sha256(Path(dataset) / "images" / f"{image_id}.jpg") != item["frame_sha256"]:
            raise ValueError(f"Frame changed: {image_id}")
        for key, owners in ((item["frame_sha256"], frames), (item["vehicle_id"], identities)):
            if key in owners and owners[key] != item["split"]:
                raise ValueError("Frame/identity leakage in detector split")
            owners[key] = item["split"]


def write_annotator(plan, suggestions, output, dataset=DATASET):
    verify_sample(plan, dataset)
    pictures, frames = {}, {}
    for image_id, item in plan["images"].items():
        with Image.open(Path(dataset) / "images" / f"{image_id}.jpg") as source:
            frame = ImageOps.exif_transpose(source).convert("RGB")
            crop = crop_image(frame, item["bbox"])
        if crop.size != (item["width"], item["height"]):
            raise ValueError("Crop dimensions changed")
        # Full crop resolution is retained: tiny border masks must remain visible.
        import base64
        import io
        stream = io.BytesIO()
        crop.save(stream, format="PNG")
        pictures[image_id] = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        frames[image_id] = {"picture": _picture(frame), "width": frame.width, "height": frame.height}
    payload = json.dumps({"plan": plan, "suggestions": suggestions, "pictures": pictures,
                          "frames": frames}, ensure_ascii=False).replace("<", "\\u003c")
    template = (ROOT / "training/mask_training_annotator.html").read_text()
    (Path(output) / "annotate_masks.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")


def prepare(output=EXPERIMENT / "annotation", counts=None, dataset=DATASET):
    counts = counts or {"train": 160, "val": 40, "holdout": 40}
    if set(counts) != {"train", "val", "holdout"} or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("Need positive train/val/holdout counts")
    if Path(output).exists() and any(Path(output).iterdir()):
        raise ValueError("Choose a fresh annotation directory")
    rows = read_rows(Path(dataset) / "train.csv")
    split_path = ROOT / "artifacts/splits.json"
    outer = load_json(split_path)
    if sha256(Path(dataset) / "train.csv") != outer["train_csv_sha256"]:
        raise ValueError("Stale outer ReID split")
    hashes = {}
    for index, row in enumerate(rows, 1):
        hashes[row["image_id"]] = sha256(Path(dataset) / "images" / f"{row['image_id']}.jpg")
        if index % 1000 == 0:
            print(f"Verify source frames: {index}/{len(rows)}", flush=True)
    if hashes != outer["frame_sha256"]:
        raise ValueError("Frames changed since outer split")
    audit_partitions(rows, hashes, outer["identities"])
    audit = load_json(REFERENCE / "mask_plan.json")
    forbidden_frames = {v["frame_sha256"] for v in audit["images"].values()}
    forbidden_ids = set(audit["images"])
    for filename in ("test_query.csv", "test_gallery.csv"):
        for row in read_rows(Path(dataset) / filename):
            forbidden_ids.add(row["image_id"])
            forbidden_frames.add(sha256(Path(dataset) / "images" / f"{row['image_id']}.jpg"))
    eligible = [r for r in rows if r["vehicle_id"] in set(outer["identities"]["train"])]
    # Drop whole identities touching an excluded frame, not just the overlapping row.
    excluded_identities = {r["vehicle_id"] for r in eligible if r["image_id"] in forbidden_ids
                           or hashes[r["image_id"]] in forbidden_frames}
    eligible = [r for r in eligible if r["vehicle_id"] not in excluded_identities]
    selected, partitions = select_rows(eligible, hashes, counts)
    images, crops = {}, {}
    for name, items in selected.items():
        for row in items:
            image_id = row["image_id"]
            with Image.open(Path(dataset) / "images" / f"{image_id}.jpg") as frame:
                crop = crop_image(frame, bbox(row))
            images[image_id] = {"bbox": list(bbox(row)), "width": crop.width, "height": crop.height,
                "frame_sha256": hashes[image_id], "vehicle_id": row["vehicle_id"],
                "camera_id": row["camera_id"], "split": name}
            if name != "holdout":
                crops[image_id] = crop
    plan = {"version": 1, "purpose": "YOLO11 anonymized_region pilot; only outer ReID train",
        "coordinates": "crop_xyxy_pixels_exclusive", "seed": SEED, "counts": counts,
        "train_csv_sha256": outer["train_csv_sha256"], "outer_split_sha256": sha256(split_path),
        "partitions": partitions, "images": images, "excluded_identities": sorted(excluded_identities),
        "selection": "camera/size-balanced, distinct frames, diverse identities; no detector score selection"}
    plan["fingerprint"] = digest(plan)
    from training.mask_detection import EXPERIMENT as plate_dir, predict_crops
    from training.mask_specialists import EXPERIMENT as mosaic_dir, predict
    import torch
    torch.set_num_threads(2)
    plate_manifest, mosaic_manifest = load_json(plate_dir / "models.json"), load_json(mosaic_dir / "models.json")
    plate_entry = next(x for x in plate_manifest["models"] if x["id"] == "yolo11n")
    mosaic_entry = next(x for x in mosaic_manifest["models"] if x["id"] == "deepmosaics")
    plate, _ = predict_crops(plate_entry, crops, plate_manifest)
    mosaic, _ = predict(mosaic_entry, crops)
    suggestions = {i: {"deepmosaics": mosaic["deepmosaics_raw"][i]["rectangles"],
                        "yolo11n": plate[i]["rectangles"]} for i in crops}
    output = new_directory(output)
    write_json(output / "mask_plan.json", plan)
    write_json(output / "proposals.json", {"fingerprint": plan["fingerprint"], "reviewed": False,
        "models": [plate_entry, mosaic_entry], "images": suggestions,
        "holdout": "No proposals generated or displayed; independent manual annotation"})
    write_annotator(plan, suggestions, output, dataset)
    report = {name: {"images": len(items), "identities": len({r["vehicle_id"] for r in items}),
                        "cameras": dict(Counter(r["camera_id"] for r in items))}
              for name, items in selected.items()}
    report["proposal_stats"] = {"without_both": sum(not v["yolo11n"] and not v["deepmosaics"] for v in suggestions.values()),
                                "note": "Absence of prediction is NOT a negative label"}
    write_json(output / "summary.json", report)
    return report


def yolo_line(rect, width, height):
    x1, y1, x2, y2 = rect
    return "0 " + " ".join(f"{x:.9f}" for x in ((x1+x2)/(2*width), (y1+y2)/(2*height),
                                                (x2-x1)/width, (y2-y1)/height))


def export_reviewed(plan_path, annotations_path, output, dataset=DATASET):
    """Unreviewed or missing frames never silently become negative examples."""
    plan, annotations = load_json(plan_path), load_json(annotations_path)
    verify_sample(plan, dataset)
    reviewed = validate_annotations(plan, annotations)
    split_stats = defaultdict(lambda: {"images": 0, "regions": 0, "negative_images": 0})
    for i, item in plan["images"].items():
        stat = split_stats[item["split"]]
        rects = reviewed[i]["rectangles"]
        if len(set(map(tuple, rects))) != len(rects):
            raise ValueError(f"Duplicate rectangles: {i}")
        stat["images"] += 1
        stat["regions"] += len(rects)
        stat["negative_images"] += not rects
    if any(split_stats[s]["regions"] == 0 for s in ("train", "val", "holdout")):
        raise ValueError("Each split needs positive reviewed examples")
    signature = {"plan_fingerprint": plan["fingerprint"], "annotations_sha256": sha256(annotations_path)}
    output = Path(output)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if any(manifest[k] != v for k, v in signature.items()):
            raise ValueError("Annotations changed; export to a new directory")
        if any(sha256(output / p) != h for p, h in manifest["files"].items()):
            raise ValueError("Exported YOLO data changed")
        return manifest
    new_directory(output)
    for i, item in plan["images"].items():
        split = "test" if item["split"] == "holdout" else item["split"]
        for folder in ("images", "labels"):
            (output / folder / split).mkdir(parents=True, exist_ok=True)
        with Image.open(Path(dataset) / "images" / f"{i}.jpg") as frame:
            crop = crop_image(frame, item["bbox"])
        crop.save(output / "images" / split / f"{i}.png")
        lines = [yolo_line(r, crop.width, crop.height) for r in reviewed[i]["rectangles"]]
        (output / "labels" / split / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    # JSON is valid YAML; no auto-download stanza, no holdout in training validation.
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val",
              "names": {0: "anonymized_region"}}
    write_json(output / "data.yaml", config)
    write_json(output / "holdout.yaml", {**config, "val": "images/test"})
    manifest = {**signature, "splits": dict(split_stats), "files": {
        str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("source", type=Path)
    reference.add_argument("--output", type=Path, default=ROOT / "Mask-Detectors/reference_v02/results")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=EXPERIMENT / "annotation")
    prep.add_argument("--counts", nargs=3, type=int, default=[160, 40, 40])
    export = sub.add_parser("export")
    export.add_argument("annotations", type=Path)
    export.add_argument("--plan", type=Path, default=EXPERIMENT / "annotation/mask_plan.json")
    export.add_argument("--output", type=Path, default=EXPERIMENT / "data")
    args = parser.parse_args()
    if args.command == "reference":
        result = revise_reference(args.source, args.output)
        print({k: result[k] for k in ("images", "regions")})
    elif args.command == "prepare":
        print(prepare(args.output, dict(zip(("train", "val", "holdout"), args.counts))))
    else:
        print(export_reviewed(args.plan, args.annotations, args.output)["splits"])
