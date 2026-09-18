"""Stage-3 preprocessing ablation for the selected OSNet checkpoint."""
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from backend.core import DATASET, Encoder, bbox, normalize, sha256
from backend.evaluate import SEED, make_protocol, write_json
from backend.scoring import calibrate, metrics, ranked_queries
from backend.rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA, rerank_protocol
from training.hpo import (ExperimentConfig, ReIDExperimentModel, encode_experiment,
                          evaluate_experiment, make_optimizer, prepare_experiment,
                          set_epoch_learning_rates, train_epoch)
from training.osnet import export_encoder_onnx
from training.pipeline import ensure_splits, format_duration, set_seed
from training.preprocessing import preprocess_mode


def encode_rows_mode(encoder, rows, dataset=DATASET, mode="square", batch_size=16):
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
                batch.append(preprocess_mode(image, bbox(row), mode))
        vectors.append(encoder.encode_batch(batch))
    return normalize(np.concatenate(vectors))


def evaluate_preprocessing(encoder, rows, split, mode, dataset=DATASET):
    protocols = {
        name: make_protocol(rows, split["identities"][name], SEED)
        for name in ("calibration", "validation")
    }
    selected = list({row["image_id"]: row
                     for protocol in protocols.values() for part in protocol for row in part}.values())
    vectors = encode_rows_mode(encoder, selected, dataset, mode)
    embeddings = dict(zip((row["image_id"] for row in selected), vectors))

    raw_calibration = ranked_queries(*protocols["calibration"], embeddings)
    raw_validation = ranked_queries(*protocols["validation"], embeddings)
    raw_threshold = calibrate(raw_calibration)

    reranked_calibration, calibration_confidence = rerank_protocol(
        *protocols["calibration"], embeddings, ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA)
    reranked_validation, validation_confidence = rerank_protocol(
        *protocols["validation"], embeddings, ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA)
    rerank_threshold = calibrate(reranked_calibration, calibration_confidence)
    return {
        "mode": mode,
        "raw": {
            "threshold": raw_threshold,
            "calibration": metrics(raw_calibration, raw_threshold),
            "validation": metrics(raw_validation, raw_threshold),
        },
        "reranked": {
            "k1": ACTIVE_K1,
            "k2": ACTIVE_K2,
            "lambda": ACTIVE_LAMBDA,
            "threshold": rerank_threshold,
            "calibration": metrics(reranked_calibration, rerank_threshold, calibration_confidence),
            "validation": metrics(reranked_validation, rerank_threshold, validation_confidence),
        },
    }


def compare_preprocessing(output_path, encoder=None, dataset=DATASET):
    encoder = encoder or Encoder()
    rows, split = ensure_splits(dataset)
    modes = {mode: evaluate_preprocessing(encoder, rows, split, mode, dataset)
             for mode in ("square", "letterbox")}
    for result in modes.values():
        validation = result["reranked"]["validation"]
        result["selection_score"] = (
            .45 * validation["mAP_at_10"] + .10 * validation["candidate_score"])
    winner = max(modes, key=lambda mode: (modes[mode]["selection_score"],
                                          modes[mode]["reranked"]["validation"]["mAP_at_10"]))
    report = {
        "experiment": "square resize versus ImageNet-mean letterbox",
        "model_sha256": encoder.model_sha256,
        "selection_rule": "0.45*mAP@10 + 0.10*(0.7*F1 + 0.3*TNR) on validation; no training",
        "winner": winner,
        "modes": modes,
    }
    write_json(Path(output_path), report)
    return report


def evaluate_active_retrieval(model, rows, split, device, dataset=DATASET, num_workers=0):
    """Evaluate one trained model with the active streaming reranker and raw-cosine refusal."""
    protocols = {
        name: make_protocol(rows, split["identities"][name], SEED)
        for name in ("calibration", "validation")
    }
    selected = list({row["image_id"]: row
                     for protocol in protocols.values() for part in protocol for row in part}.values())
    vectors = encode_experiment(model, selected, device, dataset, num_workers=num_workers)
    embeddings = dict(zip((row["image_id"] for row in selected), vectors))
    calibration, calibration_confidence = rerank_protocol(
        *protocols["calibration"], embeddings, ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA)
    validation, validation_confidence = rerank_protocol(
        *protocols["validation"], embeddings, ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA)
    threshold = calibrate(calibration, calibration_confidence)
    return {
        "k1": ACTIVE_K1,
        "k2": ACTIVE_K2,
        "lambda": ACTIVE_LAMBDA,
        "threshold": threshold,
        "calibration": metrics(calibration, threshold, calibration_confidence),
        "validation": metrics(validation, threshold, validation_confidence),
    }


def config_from_checkpoint(checkpoint_path, pooling, resize_mode, seed, epochs):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = ExperimentConfig(**checkpoint["config"])
    return replace(base, pooling=pooling, resize_mode=resize_mode, seed=seed, epochs=epochs)


def initialize_from_checkpoint(checkpoint_path, config, device):
    """Load all selected variant-2 weights and initialize only GeM's exponent."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classifier = checkpoint["model"]["classifier.weight"]
    model = ReIDExperimentModel(
        classifier.shape[0], config.use_bnneck, config.pooling, config.resize_mode)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = ["backbone.global_pool.p"] if config.pooling == "gem" else []
    if sorted(missing) != expected_missing or unexpected:
        raise ValueError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    return model.to(device), classifier.shape[0]


def _best_from_history(history):
    mapping = {"map": "mAP", "f1": "candidate_F1", "tnr": "TNR"}
    return {
        tag: max(
            ({"value": item["validation"][metric], "epoch": item["epoch"]} for item in history),
            key=lambda item: item["value"], default={"value": -1., "epoch": None})
        for tag, metric in mapping.items()
    }


def fit_stage3_seed(model, rows, split, loader, sampler, device, config,
                    weights_dir, results_dir, patience=4, minimum_epochs=6, dataset=DATASET):
    """Fine-tune one control/GeM seed with resumable early stopping."""
    weights_dir, results_dir = Path(weights_dir), Path(results_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "history.json"
    last_path = weights_dir / "last.pt"
    optimizer = make_optimizer(model, config)

    if history_path.exists() != last_path.exists():
        raise RuntimeError("Resume requires both history.json and last.pt")
    if history_path.exists():
        history = json.loads(history_path.read_text())
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    else:
        history, start_epoch = [], 0

    best = _best_from_history(history)
    started = time.perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        lrs = set_epoch_learning_rates(optimizer, config, epoch, config.epochs)
        train_result = train_epoch(model, loader, sampler, optimizer, device, config, epoch)
        train_seconds = time.perf_counter() - epoch_started
        calibration, threshold = evaluate_experiment(
            model, rows, split["identities"]["calibration"], device,
            seed=config.seed, num_workers=config.num_workers)
        validation, _ = evaluate_experiment(
            model, rows, split["identities"]["validation"], device,
            seed=config.seed, threshold=threshold, num_workers=config.num_workers)
        epoch_seconds = time.perf_counter() - epoch_started
        completed = epoch + 1
        elapsed = time.perf_counter() - started
        timing = {
            "train_seconds": train_seconds,
            "evaluation_seconds": epoch_seconds - train_seconds,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": elapsed / (completed - start_epoch) * (config.epochs - completed),
        }
        record = {"epoch": completed, "lr": lrs, "train": train_result,
                  "threshold": threshold, "calibration": calibration,
                  "validation": validation, "timing": timing}
        history.append(record)
        write_json(history_path, history)
        payload = {"epoch": completed, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "config": asdict(config),
                   "threshold": threshold, "calibration": calibration,
                   "validation": validation}
        torch.save(payload, last_path)
        improved = []
        for tag, metric in (("map", "mAP"), ("f1", "candidate_F1"), ("tnr", "TNR")):
            if validation[metric] > best[tag]["value"]:
                best[tag] = {"value": validation[metric], "epoch": completed}
                torch.save(payload, weights_dir / f"best_{tag}.pt")
                improved.append(tag)
        if improved:
            torch.save(payload, weights_dir / f"epoch_{completed:02d}.pt")
        write_json(results_dir / "checkpoint_index.json", {
            "best": best, "last_epoch": completed,
            "important_epoch_files": sorted(path.name for path in weights_dir.glob("epoch_*.pt")),
        })
        print(" | ".join([
            f"Epoch {completed}/{config.epochs}", f"осталось эпох: {config.epochs - completed}",
            f"эпоха: {format_duration(epoch_seconds)}", f"прошло: {format_duration(elapsed)}",
            f"ETA: {format_duration(timing['estimated_remaining_seconds'])}",
            f"best mAP: {best['map']['value']:.4f}",
        ]), flush=True)
        if (completed >= minimum_epochs and completed - best["map"]["epoch"] >= patience):
            stopped_early = True
            break

    best_record = next(item for item in history if item["epoch"] == best["map"]["epoch"])
    best_checkpoint = torch.load(weights_dir / "best_map.pt", map_location="cpu", weights_only=False)
    gem_p = best_checkpoint["model"].get("backbone.global_pool.p")
    model.load_state_dict(best_checkpoint["model"])
    active_reranking = evaluate_active_retrieval(
        model, rows, split, device, dataset, config.num_workers)
    summary = {
        "pooling": config.pooling,
        "resize_mode": config.resize_mode,
        "seed": config.seed,
        "configured_epochs": config.epochs,
        "completed_epochs": history[-1]["epoch"],
        "stopped_early": stopped_early,
        "patience": patience,
        "minimum_epochs": minimum_epochs,
        "best_epoch": best_record["epoch"],
        "best_validation": best_record["validation"],
        "best_threshold": best_record["threshold"],
        "active_reranking": active_reranking,
        "gem_p": float(gem_p) if gem_p is not None else None,
    }
    write_json(results_dir / "summary.json", summary)
    return summary


def run_pooling_ablation(rows, split, base_checkpoint, device, results_dir, weights_dir,
                         resize_mode="square", seeds=(SEED, SEED + 1, SEED + 2),
                         epochs=15, patience=4, minimum_epochs=6, dataset=DATASET):
    """Compare matched AvgPool and GeM continuations over the same three seeds."""
    results_dir, weights_dir = Path(results_dir), Path(weights_dir)
    runs = {"avg": [], "gem": []}
    for pooling in runs:
        for seed in seeds:
            run_name = f"{pooling}_seed_{seed}"
            run_results = results_dir / run_name
            summary_path = run_results / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                expected = {"pooling": pooling, "resize_mode": resize_mode,
                            "seed": seed, "configured_epochs": epochs}
                actual = {key: summary.get(key) for key in expected}
                if actual != expected:
                    raise RuntimeError(
                        f"Existing {summary_path} has {actual}, expected {expected}; "
                        "use a new results directory for a different experiment")
            else:
                config = config_from_checkpoint(
                    base_checkpoint, pooling, resize_mode, seed, epochs)
                set_seed(seed)
                selected, labels, sampler, loader = prepare_experiment(
                    rows, split["identities"]["train"], config, dataset)
                model, num_classes = initialize_from_checkpoint(base_checkpoint, config, device)
                if num_classes != len(labels):
                    raise ValueError("Base checkpoint classifier does not match train identities")
                try:
                    summary = fit_stage3_seed(
                        model, rows, split, loader, sampler, device, config,
                        weights_dir / run_name, run_results, patience, minimum_epochs, dataset)
                finally:
                    del model, loader, sampler
                    gc.collect()
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
            runs[pooling].append(summary)

    aggregates = {}
    for pooling, pooling_runs in runs.items():
        maps = [run["active_reranking"]["validation"]["mAP"] for run in pooling_runs]
        candidate_scores = [run["active_reranking"]["validation"]["candidate_score"]
                            for run in pooling_runs]
        aggregates[pooling] = {
            "mean_best_mAP": float(np.mean(maps)),
            "std_best_mAP": float(np.std(maps)),
            "mean_candidate_score_at_best_mAP": float(np.mean(candidate_scores)),
            "runs": pooling_runs,
        }
    control_score = aggregates["avg"]["mean_candidate_score_at_best_mAP"]
    eligible = [name for name in aggregates
                if aggregates[name]["mean_candidate_score_at_best_mAP"] >= control_score - .005]
    winner = max(eligible, key=lambda name: aggregates[name]["mean_best_mAP"])
    mean_map = aggregates[winner]["mean_best_mAP"]
    selected = min(aggregates[winner]["runs"],
                   key=lambda run: abs(run["active_reranking"]["validation"]["mAP"] - mean_map))
    report = {
        "base_checkpoint": str(Path(base_checkpoint)),
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "resize_mode": resize_mode,
        "seeds": list(seeds),
        "max_epochs": epochs,
        "selection_rule": "highest mean active-reranked mAP at each seed's raw-best checkpoint, "
                          "with mean candidate score no more than 0.005 below AvgPool",
        "winner": winner,
        "selected_representative": {
            "pooling": winner,
            "seed": selected["seed"],
            "best_epoch": selected["best_epoch"],
        },
        "aggregates": aggregates,
    }
    write_json(results_dir / "pooling_ablation.json", report)
    return report


def export_pooling_winner(report, rows, split, weights_dir, output_path, metadata_path,
                          dataset=DATASET):
    selected = report["selected_representative"]
    run_name = f"{selected['pooling']}_seed_{selected['seed']}"
    checkpoint_path = Path(weights_dir) / run_name / "best_map.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    model = ReIDExperimentModel(
        len(split["identities"]["train"]), config.use_bnneck,
        config.pooling, config.resize_mode)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    export_encoder_onnx(model.inference_module(), output_path, "cpu")

    sample = next(row for row in rows if row["vehicle_id"] in set(split["identities"]["validation"]))
    with Image.open(dataset / "images" / f"{sample['image_id']}.jpg") as image:
        array = preprocess_mode(image, bbox(sample), config.resize_mode)[None]
    with torch.inference_mode():
        expected = model.embedding(torch.from_numpy(array)).numpy()
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(["output"], {session.get_inputs()[0].name: array})[0]
    parity = {
        "max_absolute_difference": float(np.max(np.abs(expected - actual))),
        "cosine_similarity": float(F.cosine_similarity(
            torch.from_numpy(expected), torch.from_numpy(actual)).item()),
    }
    if parity["cosine_similarity"] <= .99999 or parity["max_absolute_difference"] >= 1e-3:
        raise ValueError(f"ONNX parity failed: {parity}")
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
        "threshold": checkpoint["threshold"],
        "calibration": checkpoint["calibration"],
        "validation": checkpoint["validation"],
        "onnx": str(output_path),
        "onnx_sha256": sha256(output_path),
        "parity": parity,
    }
    write_json(metadata_path, metadata)
    return metadata


if __name__ == "__main__":
    output = Path("OSNet-AIN-x1.0/variant_03_gem/results/preprocessing_ablation.json")
    print(json.dumps(compare_preprocessing(output), ensure_ascii=False, indent=2))
