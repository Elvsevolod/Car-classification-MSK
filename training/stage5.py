"""Stage-5 comparison of ResNet50-IBN-a + GeM + BNNeck with active OSNet."""
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from backend.core import ARTIFACTS, DATASET, bbox, sha256
from backend.evaluate import SEED, write_json
from training.hpo import (ExperimentConfig, fit_inner_candidate,
                          prepare_experiment, split_hpo_identities)
from training.osnet import export_encoder_onnx
from training.pipeline import set_seed
from training.preprocessing import preprocess_mode
from training.resnet_ibn import (EMBEDDING_DIMENSION, PRETRAINED_URL,
                                 PRETRAINED_VERSION, ResNetIBNReIDModel,
                                 load_official_imagenet_weights)
from training.stage4 import ACTIVE_REFERENCE, fit_fixed_epochs, tuned_retrieval


DEFAULT_SCREENING_LRS = (3e-5, 6e-5, 1e-4)


def stage5_config(base_config_path, epochs, seed, encoder_lr):
    """Reuse the selected loss recipe while changing only the backbone and LR."""
    saved = json.loads(Path(base_config_path).read_text())
    base = ExperimentConfig(**saved)
    return replace(
        base,
        epochs=epochs,
        seed=seed,
        encoder_lr=encoder_lr,
        pooling="gem",
        resize_mode="square",
        metric_loss="supcon",
        use_bnneck=True,
        use_mixstyle=False,
        hard_negative_sampling=False,
        loss_weight_schedule="constant",
    )


def initialize_resnet_experiment(num_classes, config, device, progress=True):
    model = ResNetIBNReIDModel(
        num_classes,
        pooling=config.pooling,
        resize_mode=config.resize_mode,
        use_bnneck=config.use_bnneck,
    )
    initializer = load_official_imagenet_weights(
        model.backbone, progress=progress, model_dir=ARTIFACTS / "pretrained")
    return model.to(device), initializer


def _release(model, loader, sampler, device):
    del model, loader, sampler
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _run_screening_candidate(name, config, rows, train_ids, validation_ids,
                             device, results_dir, weights_dir, dataset):
    result_dir = Path(results_dir) / "screening" / name
    weight_dir = Path(weights_dir) / "screening" / name
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if "config" in summary and summary["config"] != asdict(config):
            raise RuntimeError(f"Existing screening result has another config: {summary_path}")
        if "config" in summary:
            return summary

    set_seed(config.seed)
    selected, labels, sampler, loader = prepare_experiment(
        rows, train_ids, config, dataset)
    model, initializer = initialize_resnet_experiment(len(labels), config, device)
    try:
        result = fit_inner_candidate(
            model, rows, validation_ids, loader, sampler, device, config,
            weight_dir, result_dir)
    finally:
        _release(model, loader, sampler, device)
    summary = {
        "name": name,
        "config": asdict(config),
        "initializer": initializer,
        "train_images": len(selected),
        **result,
    }
    write_json(summary_path, summary)
    return summary


def run_backbone_screening(rows, split, device, base_config_path,
                           results_dir, weights_dir,
                           learning_rates=DEFAULT_SCREENING_LRS,
                           epochs=6, dataset=DATASET):
    """Select only the encoder LR on an inner identity-disjoint split."""
    train_ids, validation_ids = split_hpo_identities(
        split["identities"]["train"], SEED)
    candidates = []
    for learning_rate in learning_rates:
        config = stage5_config(
            base_config_path, epochs, SEED, learning_rate)
        name = f"lr_{learning_rate:.0e}"
        candidates.append(_run_screening_candidate(
            name, config, rows, train_ids, validation_ids, device,
            results_dir, weights_dir, dataset))

    candidates.sort(key=lambda item: item["best_mAP"], reverse=True)
    report = {
        "architecture": "ResNet50-IBN-a + GeM + BNNeck",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "pretrained_source": PRETRAINED_VERSION,
        "pretrained_url": PRETRAINED_URL,
        "protocol": "inner 80/20 identity-disjoint split of outer-train identities",
        "outer_calibration_or_validation_used": False,
        "selection_metric": "best inner-validation raw mAP@10",
        "screening_epochs": epochs,
        "learning_rates": list(learning_rates),
        "winner": candidates[0],
        "candidates": candidates,
    }
    write_json(Path(results_dir) / "screening_summary.json", report)
    return report


def _run_final_seed(config, rows, split, device, results_dir, weights_dir,
                    dataset=DATASET):
    name = f"seed_{config.seed}"
    result_dir = Path(results_dir) / "final_seeds" / name
    weight_dir = Path(weights_dir) / "final_seeds" / name
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if "config" in summary and summary["config"] != asdict(config):
            raise RuntimeError(f"Existing final result has another config: {summary_path}")
        if "tuned_reranking" in summary:
            return summary

    set_seed(config.seed)
    selected, labels, sampler, loader = prepare_experiment(
        rows, split["identities"]["train"], config, dataset)
    model, initializer = initialize_resnet_experiment(len(labels), config, device)
    try:
        summary = fit_fixed_epochs(
            model, loader, sampler, device, config, weight_dir, result_dir)
        tuned = tuned_retrieval(
            model, rows, split, device, dataset, config.num_workers)
    finally:
        _release(model, loader, sampler, device)
    summary.update({
        "name": name,
        "config": asdict(config),
        "initializer": initializer,
        "train_images": len(selected),
        "tuned_reranking": tuned,
    })
    write_json(summary_path, summary)
    return summary


def run_backbone_seeds(rows, split, screening, device, results_dir, weights_dir,
                       seeds=(SEED, SEED + 1, SEED + 2), dataset=DATASET):
    """Train the inner-selected recipe for fixed epochs and evaluate once per seed."""
    winner = screening["winner"]
    fixed_epochs = winner["best_epoch"]
    template = replace(ExperimentConfig(**winner["config"]), epochs=fixed_epochs)
    runs = []
    for seed in seeds:
        runs.append(_run_final_seed(
            replace(template, seed=seed), rows, split, device,
            results_dir, weights_dir, dataset))

    maps = [run["tuned_reranking"]["validation"]["mAP_at_10"] for run in runs]
    candidate_scores = [
        run["tuned_reranking"]["validation"]["candidate_score"] for run in runs]
    quality_scores = [
        run["tuned_reranking"]["validation_quality_score"] for run in runs]
    aggregate = {
        "mean_mAP_at_10": float(np.mean(maps)),
        "std_mAP_at_10": float(np.std(maps)),
        "mean_candidate_score": float(np.mean(candidate_scores)),
        "std_candidate_score": float(np.std(candidate_scores)),
        "mean_quality_score": float(np.mean(quality_scores)),
        "std_quality_score": float(np.std(quality_scores)),
    }
    beats_active = (
        aggregate["mean_mAP_at_10"] > ACTIVE_REFERENCE["mAP_at_10"] and
        aggregate["mean_quality_score"] > ACTIVE_REFERENCE["quality_score"] and
        aggregate["mean_candidate_score"] >= ACTIVE_REFERENCE["candidate_score"] - .005)
    representative = min(
        runs,
        key=lambda run: abs(
            run["tuned_reranking"]["validation"]["mAP_at_10"] -
            aggregate["mean_mAP_at_10"]),
    )
    report = {
        "architecture": "ResNet50-IBN-a + GeM + BNNeck",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "active_osnet_embedding_dimension": 512,
        "vector_storage_multiplier": EMBEDDING_DIMENSION / 512,
        "pretrained_source": PRETRAINED_VERSION,
        "pretrained_sha256": representative["initializer"]["sha256"],
        "seeds": list(seeds),
        "fixed_training_epochs": fixed_epochs,
        "epoch_selection": "best epoch on inner screening; outer validation evaluated once",
        "active_reference": ACTIVE_REFERENCE,
        "aggregate": aggregate,
        "beats_active_reference": beats_active,
        "selected_representative": {
            "name": representative["name"],
            "seed": representative["config"]["seed"],
        },
        "selection_rule": (
            "mean mAP@10 and quality must exceed active OSNet, while mean candidate "
            "score may not fall by more than 0.005"
        ),
        "runs": runs,
    }
    write_json(Path(results_dir) / "final_comparison.json", report)
    return report


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _benchmark_pytorch(module, array, device, repeats=30):
    module = module.eval().to(device)
    values = torch.from_numpy(array).to(device)
    with torch.inference_mode():
        for _ in range(3):
            module(values)
        _synchronize(device)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            module(values)
            _synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
    return {
        "device": str(device),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "samples": repeats,
        "includes": "model forward only, batch=1",
    }


def _benchmark_onnx(session, array, repeats=30):
    input_name = session.get_inputs()[0].name
    for _ in range(3):
        session.run(["output"], {input_name: array})
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run(["output"], {input_name: array})
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "device": "ONNX Runtime CPU",
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "samples": repeats,
        "includes": "model forward only, batch=1",
    }


def export_stage5_winner(report, split, rows, weights_dir, output_path,
                         metadata_path, benchmark_device=torch.device("cpu"),
                         dataset=DATASET):
    name = report["selected_representative"]["name"]
    checkpoint_path = Path(weights_dir) / "final_seeds" / name / "final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    model = ResNetIBNReIDModel(
        len(split["identities"]["train"]),
        pooling=config.pooling,
        resize_mode=config.resize_mode,
        use_bnneck=config.use_bnneck,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    inference = model.inference_module()
    export_encoder_onnx(inference, output_path, "cpu")

    validation_ids = set(split["identities"]["validation"])
    sample = next(row for row in rows if row["vehicle_id"] in validation_ids)
    with Image.open(dataset / "images" / f"{sample['image_id']}.jpg") as image:
        array = preprocess_mode(image, bbox(sample), config.resize_mode)[None]
    with torch.inference_mode():
        expected = inference(torch.from_numpy(array)).numpy()

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

    selected_run = next(run for run in report["runs"] if run["name"] == name)
    metadata = {
        "architecture": report["architecture"],
        "embedding_dimension": EMBEDDING_DIMENSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
        "pretrained_source": PRETRAINED_VERSION,
        "pretrained_url": PRETRAINED_URL,
        "pretrained_sha256": selected_run["initializer"]["sha256"],
        "onnx": str(output_path),
        "onnx_sha256": sha256(output_path),
        "onnx_size_mb": Path(output_path).stat().st_size / (1024 ** 2),
        "parameter_count": sum(parameter.numel() for parameter in inference.parameters()),
        "parity": parity,
        "benchmark": {
            "pytorch": _benchmark_pytorch(inference, array, benchmark_device),
            "onnx_cpu": _benchmark_onnx(session, array),
        },
        "tuned_reranking": selected_run["tuned_reranking"],
    }
    write_json(metadata_path, metadata)
    return metadata
