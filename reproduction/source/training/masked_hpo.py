"""Isolated variant-2 HPO on frozen, automatic opaque masks; no MVP writes."""
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from backend.core import ARTIFACTS, DATASET, ROOT, STOCK_MODEL, read_rows, sha256
from backend.evaluate import SEED, make_protocol
from backend.rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA, rerank_protocol
from backend.scoring import calibrate, metrics, ranked_queries
from training.audit import digest, load_crop
from training.hpo import (ExperimentConfig, encode_experiment, fit_inner_candidate,
                          initialize_experiment, prepare_experiment)
from training.mask_calibration import _context, _load, _save
from training.mask_finetune import EXPERIMENT as DETECTOR, load_json
from training.mask_reid_ablation import (CALIBRATION, WEIGHTS, check_detector_separation,
                                        detect_mask)
from training.pipeline import format_duration, set_seed
from training.stage6 import audit_partitions
from training.yolo_masks import runtime


VARIANT = ROOT / "OSNet-AIN-x1.0/variant_08_masked_hpo"


def inner_split(rows, train_ids, hashes, detector_plan, seed=SEED):
    """Keep connected frames together and all detector data on the training side."""
    parent = {i: i for i in sorted(train_ids)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    frames = {}
    selected = [r for r in rows if r["vehicle_id"] in parent]
    for row in selected:
        identity, frame = row["vehicle_id"], hashes[row["image_id"]]
        if frame in frames:
            parent[find(identity)] = find(frames[frame])
        frames[frame] = identity
    groups = defaultdict(list)
    for identity in parent:
        groups[find(identity)].append(identity)
    detector_ids = {v["vehicle_id"] for v in detector_plan["images"].values()}
    detector_frames = {v["frame_sha256"] for v in detector_plan["images"].values()}
    if not detector_ids.issubset(parent):
        raise ValueError("Detector identities must belong to outer train")
    forced = {find(i) for i in detector_ids}
    forced.update(find(r["vehicle_id"]) for r in selected if hashes[r["image_id"]] in detector_frames)
    eligible = [group for root, group in groups.items() if root not in forced]
    random.Random(seed).shuffle(eligible)
    target = max(1, round(len(parent) * .2))
    validation = []
    for group in eligible:
        if len(validation) >= target:
            break
        validation.extend(group)
    if len(validation) < target or len(validation) == len(parent):
        raise ValueError("Not enough detector-independent identities for HPO validation")
    result = {"train": sorted(set(parent) - set(validation)), "validation": sorted(validation)}
    audit_partitions(selected, hashes, result)
    check_detector_separation([r for r in selected if r["vehicle_id"] in set(validation)],
                              hashes, detector_plan)
    return result


def prepare_protocol(variant=VARIANT, dataset=DATASET):
    """Verify inputs without rewriting the active splits, weights, or calibration."""
    rows = read_rows(dataset / "train.csv")
    split = load_json(ARTIFACTS / "splits.json")
    if sha256(dataset / "train.csv") != split["train_csv_sha256"]:
        raise ValueError("train.csv changed; do not silently regenerate the outer split")
    hashes = {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg")
              for r in tqdm(rows, desc="Проверка исходных кадров")}
    if hashes != split["frame_sha256"]:
        raise ValueError("Source frames differ from the frozen outer split")
    audit_partitions(rows, hashes, split["identities"])
    plan, _, detector_signature = _context(WEIGHTS, DETECTOR / "annotation/reviewed_masks.json", DETECTOR)
    frozen = _load(CALIBRATION / "calibration.json", detector_signature)
    if (frozen["raw_sha256"] != sha256(CALIBRATION / "raw_val.json")
            or plan["outer_split_sha256"] != sha256(ARTIFACTS / "splits.json")):
        raise ValueError("Changed mask calibration or detector outer split")
    outer_train = set(split["identities"]["train"])
    check_detector_separation([r for r in rows if r["vehicle_id"] not in outer_train], hashes, plan)
    split = {**split, "hpo_identities": inner_split(rows, outer_train, hashes, plan)}
    sources = ("training/masked_hpo.py", "training/hpo.py", "training/pipeline.py",
               "training/osnet.py", "training/preprocessing.py", "training/mask_reid_ablation.py",
               "training/audit.py", "training/mask_detection.py", "backend/core.py",
               "backend/scoring.py", "backend/rerank.py", "backend/evaluate.py", "evaluate.py")
    signature = {"version": 1, "rows": rows, "frames": hashes,
        "outer_split_sha256": sha256(ARTIFACTS / "splits.json"), "inner": split["hpo_identities"],
        "stock_sha256": sha256(STOCK_MODEL), "detector_sha256": sha256(WEIGHTS),
        "calibration_sha256": sha256(CALIBRATION / "calibration.json"),
        "policy": frozen["selected"], "inference": frozen["signature"]["inference"],
        "masking": "all branches; crop-local opaque black xyxy before resize and augmentation",
        "code": {p: sha256(ROOT / p) for p in sources}}
    path = Path(variant) / "results/protocol.json"
    if path.exists():
        _load(path, signature)
    else:
        _save(path, {"signature": signature})
    return rows, split, signature


def cache_masks(rows, signature, path, device="cpu", dataset=DATASET, save_every=100):
    """Cache coordinates only; interrupted detection resumes without new image copies."""
    path = Path(path)
    expected = {r["image_id"] for r in rows}
    cache_signature = {"protocol": digest(signature), "device": str(device)}
    cache = _load(path, cache_signature) if path.exists() else {
        "signature": cache_signature, "images": {}, "elapsed_seconds": 0.}
    if not set(cache["images"]).issubset(expected):
        raise ValueError("Mask cache contains unexpected images")
    pending = [r for r in rows if r["image_id"] not in cache["images"]]
    if not pending:
        print(f"Готовые маски: {len(expected)}/{len(expected)}; YOLO повторно не запускается")
        return cache
    model = runtime()(str(WEIGHTS))
    started, previous = time.perf_counter(), cache["elapsed_seconds"]
    for n, row in enumerate(pending, 1):
        crop = load_crop(row, dataset)
        prediction = detect_mask(model, crop, signature["policy"], signature["inference"], device)
        cache["images"][row["image_id"]] = {**prediction, "width": crop.width, "height": crop.height}
        if n % save_every == 0 or n == len(pending):
            elapsed = time.perf_counter() - started
            cache["elapsed_seconds"] = previous + elapsed
            _save(path, cache)
            print(f"Маски: {len(cache['images'])}/{len(rows)} | прошло {format_duration(elapsed)} | "
                  f"ETA {format_duration(elapsed / n * (len(pending) - n))}", flush=True)
    return cache


def attach_masks(rows, cache):
    """A missing prediction is an error, not an implicit unmasked training image."""
    if set(cache["images"]) != {r["image_id"] for r in rows}:
        raise ValueError("Mask cache must cover every training/calibration/validation row")
    result = []
    for row in rows:
        item = cache["images"][row["image_id"]]
        if (item["width"], item["height"]) != (row["w"], row["h"]):
            raise ValueError("Cached mask crop dimensions changed")
        rectangles = item["rectangles"]
        for rect in rectangles:
            if (len(rect) != 4 or any(type(v) is not int for v in rect)
                    or not 0 <= rect[0] < rect[2] <= row["w"]
                    or not 0 <= rect[1] < rect[3] <= row["h"]):
                raise ValueError("Invalid cached mask rectangle")
        result.append({**row, "mask_rectangles": rectangles})
    return result


def freeze_run(results, signature, cache_path, budget):
    """Reject a resumed study with changed masks, source code, or training budget."""
    path = Path(results) / "run_manifest.json"
    value = {"signature": {"protocol": digest(signature), "mask_cache_sha256": sha256(cache_path),
                           "budget": budget, "mask_policy": signature["policy"],
                           "initializer": "stock ONNX; not current MVP weights",
                           "selection": "HPO inner; selected checkpoint on outer calibration only"}}
    if path.exists():
        _load(path, value["signature"])
    else:
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError("Run directory is not empty and has no manifest; choose a new RUN_NAME")
        _save(path, value)
    return value["signature"]


def fit_masked_selected(rows, split, device, config, weights_dir, results_dir):
    """Reuse HPO training/resume; select on calibration, never on outer validation."""
    set_seed(config.seed)
    train_ids = split["identities"]["train"]
    _, labels, sampler, loader = prepare_experiment(rows, train_ids, config)
    model, _ = initialize_experiment(len(labels), config, device)
    try:
        return fit_inner_candidate(model, rows, split["identities"]["calibration"],
                                   loader, sampler, device, config, weights_dir, results_dir)
    finally:
        del model, loader, sampler
        from training.hpo import _release_device
        _release_device(device)


def evaluate_selected(model, rows, split, device):
    """Evaluate masked query/gallery once after checkpoint selection, with fixed reranking."""
    protocols = {name: make_protocol(rows, split["identities"][name], SEED)
                 for name in ("calibration", "validation")}
    selected = list({r["image_id"]: r for q, g in protocols.values() for r in q + g}.values())
    vectors = encode_experiment(model, selected, device)
    embeddings = dict(zip((r["image_id"] for r in selected), vectors))
    ranked = {}
    for name, (query, gallery) in protocols.items():
        ranked[name] = {"raw": ranked_queries(query, gallery, embeddings),
                        "reranked": rerank_protocol(query, gallery, embeddings)[0]}
    report = {}
    for mode in ("raw", "reranked"):
        threshold = calibrate(ranked["calibration"][mode])
        report[mode] = {"threshold": threshold,
            **{name: metrics(ranked[name][mode], threshold) for name in protocols}}
    return {"scores": report, "reranking": {"k1": ACTIVE_K1, "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA},
            "selection_split": "calibration", "outer_validation_used_for_selection": False,
            "warning": "Outer validation has been used in past project experiments; not a new independent test"}


def export_selected(rows, split, weights, results, run_signature):
    """Keep exported weights local to this variant; masks are NOT embedded in ONNX."""
    import onnxruntime as ort
    from training.osnet import export_encoder_onnx
    from training.pipeline import VehicleDataset

    weights, results = Path(weights), Path(results)
    checkpoint_path = weights / "best_map.pt"
    signature = {"run": run_signature, "checkpoint_sha256": sha256(checkpoint_path)}
    report_path = results / "evaluation.json"
    onnx_path = weights / "osnet_masked_best_map.onnx"
    if report_path.exists():
        report = _load(report_path, signature)
        if report["onnx_sha256"] != sha256(onnx_path):
            raise ValueError("Exported ONNX changed")
        return report
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    model, _ = initialize_experiment(len(split["identities"]["train"]), config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    export_encoder_onnx(model.inference_module(), onnx_path, "cpu")
    sample = next(r for r in rows if r["vehicle_id"] in set(split["identities"]["calibration"]))
    array = VehicleDataset([{**sample, "label": 0}])[0][0].unsqueeze(0).numpy()
    with torch.inference_mode():
        expected = model.embedding(torch.from_numpy(array)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: array})[0]
    difference = float(np.max(np.abs(actual - expected)))
    cosine = float(torch.nn.functional.cosine_similarity(torch.from_numpy(actual), torch.from_numpy(expected)).item())
    if difference >= 1e-3 or cosine <= .99999:
        raise ValueError(f"ONNX parity failed: difference={difference}, cosine={cosine}")
    report = {"signature": signature, "checkpoint_epoch": checkpoint["epoch"], "config": asdict(config),
              "onnx": str(onnx_path.relative_to(ROOT)), "onnx_sha256": sha256(onnx_path),
              "parity": {"max_absolute_difference": difference, "cosine_similarity": cosine},
              "mask_policy_required": run_signature["mask_policy"],
              **evaluate_selected(model, rows, split, torch.device("cpu"))}
    _save(report_path, report)
    return report
