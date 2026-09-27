"""Stage-4 training-strategy ablations for OSNet vehicle ReID."""
import gc
import hashlib
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from backend.compare_inference import tune_reranking
from backend.core import ARTIFACTS, DATASET, bbox, sha256
from backend.evaluate import SEED, make_protocol, write_json
from training.hpo import (CameraAwarePKBatchSampler, ExperimentConfig,
                          ReIDExperimentModel, encode_experiment,
                          fit_inner_candidate, make_optimizer,
                          set_epoch_learning_rates, train_epoch)
from training.osnet import export_encoder_onnx
from training.pipeline import VehicleDataset, format_duration, set_seed
from training.preprocessing import preprocess_mode


ACTIVE_REFERENCE = {
    "mAP_at_10": .8146886982413298,
    "candidate_score": .747121224071299,
    "quality_score": .44132203661572833,
}


class SimilarityPKBatchSampler(CameraAwarePKBatchSampler):
    """Group visually similar identities while retaining cross-camera positives."""

    def __init__(self, rows, identities_per_batch, images_per_identity, neighbors,
                 prefer_cross_camera=True, seed=SEED):
        super().__init__(rows, identities_per_batch, images_per_identity,
                         prefer_cross_camera, seed)
        if set(neighbors) != set(self.identities):
            raise ValueError("Hard-negative neighbors must cover every training identity")
        self.neighbors = neighbors

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        remaining = set(self.identities)
        anchors = rng.permutation(self.identities).tolist()
        while len(remaining) >= self.p:
            anchor = next(identity for identity in anchors if identity in remaining)
            group = [anchor]
            group.extend(identity for identity in self.neighbors[anchor]
                         if identity in remaining and identity != anchor)
            group = group[:self.p]
            if len(group) < self.p:
                choices = sorted(remaining - set(group))
                group.extend(rng.choice(choices, self.p - len(group), replace=False).tolist())
            remaining.difference_update(group)
            python_rng = random.Random(self.seed + self.epoch + anchor)
            yield [index for identity in group
                   for index in self._sample_identity(identity, python_rng)]


def prepare_strategy(rows, identities, config, hard_neighbors=None, dataset=DATASET):
    config.validate()
    identities = sorted(identities)
    labels = {identity: index for index, identity in enumerate(identities)}
    selected = [{**row, "label": labels[row["vehicle_id"]]}
                for row in rows if row["vehicle_id"] in labels]
    sampler_args = (selected, config.identities_per_batch, config.images_per_identity)
    if config.hard_negative_sampling:
        if hard_neighbors is None:
            raise ValueError("hard_neighbors are required for hard-negative sampling")
        sampler = SimilarityPKBatchSampler(
            *sampler_args, hard_neighbors, config.prefer_cross_camera, config.seed)
    else:
        sampler = CameraAwarePKBatchSampler(
            *sampler_args, config.prefer_cross_camera, config.seed)
    loader = DataLoader(
        VehicleDataset(selected, dataset, augment=True, resize_mode=config.resize_mode),
        batch_sampler=sampler, num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=config.num_workers > 0)
    return selected, labels, sampler, loader


@torch.inference_mode()
def build_hard_neighbors(model, selected, device, dataset=DATASET, maximum_neighbors=64):
    vectors = encode_experiment(model, selected, device, dataset)
    labels = np.asarray([row["label"] for row in selected])
    identities = np.unique(labels)
    prototypes = []
    for identity in identities:
        prototype = vectors[labels == identity].mean(axis=0)
        prototypes.append(prototype / max(np.linalg.norm(prototype), 1e-12))
    similarities = np.stack(prototypes) @ np.stack(prototypes).T
    order = np.argsort(-similarities, axis=1, kind="stable")
    return {int(identity): [int(value) for value in order[index]
                            if value != identity][:maximum_neighbors]
            for index, identity in enumerate(identities)}


def cached_hard_neighbors(model, selected, device, cache_path, model_path, dataset=DATASET):
    cache_path = Path(cache_path)
    signature = hashlib.sha256()
    signature.update(sha256(model_path).encode())
    signature.update(sha256(Path(dataset) / "train.csv").encode())
    signature.update("\n".join(
        f'{row["image_id"]}:{row["label"]}:{bbox(row)}' for row in selected).encode())
    digest = signature.hexdigest()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["signature"]) == digest:
                labels, neighbors = saved["labels"], saved["neighbors"]
                return {int(label): [int(value) for value in row]
                        for label, row in zip(labels, neighbors)}
    result = build_hard_neighbors(model, selected, device, dataset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(sorted(result), dtype=np.int64)
    neighbors = np.asarray([result[int(label)] for label in labels], dtype=np.int64)
    np.savez_compressed(cache_path, signature=np.asarray(digest),
                        labels=labels, neighbors=neighbors)
    return result


def base_stage4_config(base_checkpoint, epochs=8, seed=SEED):
    saved = torch.load(base_checkpoint, map_location="cpu", weights_only=False)["config"]
    return replace(
        ExperimentConfig(**saved), epochs=epochs, seed=seed,
        pooling="avg", resize_mode="square", metric_loss="supcon",
        use_mixstyle=False, hard_negative_sampling=False,
        loss_weight_schedule="constant")


def initialize_from_base_checkpoint(base_checkpoint, num_classes, config, device,
                                    load_classifier):
    """Reuse the selected encoder/BNNeck; reset only an incompatible inner head."""
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model = ReIDExperimentModel(
        num_classes, config.use_bnneck, config.pooling, config.resize_mode,
        config.use_mixstyle, config.mixstyle_probability, config.mixstyle_alpha)
    state = checkpoint["model"]
    if load_classifier:
        classifier = state.get("classifier.weight")
        if classifier is None or classifier.shape[0] != num_classes:
            raise ValueError("Base checkpoint classifier does not match final train identities")
        transferred = state
        expected_missing = []
    else:
        transferred = {key: value for key, value in state.items()
                       if not key.startswith("classifier.")}
        expected_missing = sorted(
            key for key in model.state_dict() if key.startswith("classifier."))
    missing, unexpected = model.load_state_dict(transferred, strict=False)
    if sorted(missing) != expected_missing or unexpected:
        raise ValueError(
            f"Base checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    return model.to(device), len(transferred)


def _release(model, loader, sampler, device):
    del model, loader, sampler
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def screen_candidate(name, config, rows, train_ids, validation_ids, device,
                     results_dir, weights_dir, base_checkpoint,
                     hard_neighbors=None, dataset=DATASET):
    result_dir = Path(results_dir) / "screening" / name
    weight_dir = Path(weights_dir) / "screening" / name
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if "config" not in summary:
            train_id_set = set(train_ids)
            summary = {
                "name": name,
                "config": asdict(config),
                "loaded_base_tensors": None,
                "train_images": sum(row["vehicle_id"] in train_id_set for row in rows),
                **summary,
            }
            write_json(summary_path, summary)
        if summary.get("config") != asdict(config):
            raise RuntimeError(f"Existing screening result has another config: {summary_path}")
        return summary

    set_seed(config.seed)
    selected, labels, sampler, loader = prepare_strategy(
        rows, train_ids, config, hard_neighbors, dataset)
    model, loaded = initialize_from_base_checkpoint(
        base_checkpoint, len(labels), config, device, load_classifier=False)
    try:
        result = fit_inner_candidate(
            model, rows, validation_ids, loader, sampler, device, config,
            weight_dir, result_dir)
    finally:
        _release(model, loader, sampler, device)
    summary = {"name": name, "config": asdict(config),
               "loaded_base_tensors": loaded, "train_images": len(selected), **result}
    write_json(summary_path, summary)
    return summary


def run_strategy_screening(rows, split, device, base_checkpoint,
                           results_dir, weights_dir, epochs=8, dataset=DATASET):
    """Sequential inner-split screening; outer calibration/validation stay hidden."""
    from training.hpo import split_hpo_identities

    base = base_stage4_config(base_checkpoint, epochs, SEED)
    train_ids, validation_ids = split_hpo_identities(split["identities"]["train"], SEED)
    baseline_rows, baseline_labels, _, _ = prepare_strategy(rows, train_ids, base, dataset=dataset)
    prototype_model, _ = initialize_from_base_checkpoint(
        base_checkpoint, len(baseline_labels), base, device, load_classifier=False)
    try:
        neighbors = cached_hard_neighbors(
            prototype_model, baseline_rows, device,
            ARTIFACTS / "stage4_hard_neighbors_inner.npz", base_checkpoint, dataset)
    finally:
        del prototype_model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    candidates = {}

    def run(name, config):
        candidates[name] = screen_candidate(
            name, config, rows, train_ids, validation_ids, device,
            results_dir, weights_dir, base_checkpoint,
            neighbors if config.hard_negative_sampling else None, dataset)
        return candidates[name]

    baseline = run("baseline", base)
    mixstyle = run("mixstyle", replace(base, use_mixstyle=True))
    hard = run("hard_negatives", replace(base, hard_negative_sampling=True))

    baseline_map = baseline["best_mAP"]
    feature_candidates = [baseline, mixstyle, hard]
    if mixstyle["best_mAP"] > baseline_map and hard["best_mAP"] > baseline_map:
        combined = run("mixstyle_hard_negatives", replace(
            base, use_mixstyle=True, hard_negative_sampling=True))
        feature_candidates.append(combined)
    feature_winner = max(feature_candidates, key=lambda item: item["best_mAP"])
    feature_config = ExperimentConfig(**feature_winner["config"])

    dynamic = run("dynamic_weights", replace(
        feature_config, loss_weight_schedule="metric_warmup"))
    schedule_winner = max((feature_winner, dynamic), key=lambda item: item["best_mAP"])
    schedule_config = ExperimentConfig(**schedule_winner["config"])

    circle = run("circle_loss", replace(
        schedule_config, metric_loss="circle", metric_weight=1.0))
    winner = max((schedule_winner, circle), key=lambda item: item["best_mAP"])
    report = {
        "protocol": "inner 80/20 split of the 925 outer-train identities",
        "base_checkpoint": str(Path(base_checkpoint)),
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "outer_calibration_or_validation_used": False,
        "selection_metric": "best inner-validation raw mAP@10",
        "winner": winner["name"],
        "winner_config": winner["config"],
        "candidates": candidates,
    }
    write_json(Path(results_dir) / "screening_summary.json", report)
    return report


def tuned_retrieval(model, rows, split, device, dataset=DATASET, num_workers=0):
    protocols = {name: make_protocol(rows, split["identities"][name], SEED)
                 for name in ("calibration", "validation")}
    selected = list({row["image_id"]: row
                     for protocol in protocols.values() for part in protocol for row in part}.values())
    vectors = encode_experiment(model, selected, device, dataset, num_workers=num_workers)
    embeddings = dict(zip((row["image_id"] for row in selected), vectors))
    result = tune_reranking(protocols, embeddings)
    chosen = result["selected_on_calibration"]
    return {
        "selected_on_calibration": {
            key: chosen[key] for key in ("k1", "k2", "lambda", "confidence_mode", "threshold")},
        "calibration": chosen["metrics"],
        "validation": result["validation"],
        "validation_quality_score": result["validation_quality_score"],
    }


def fit_fixed_epochs(model, loader, sampler, device, config, weights_dir, results_dir):
    """Train a fixed inner-selected epoch count without consulting outer validation."""
    weights_dir, results_dir = Path(weights_dir), Path(results_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "history.json"
    summary_path = results_dir / "summary.json"
    last_path = weights_dir / "last.pt"
    final_path = weights_dir / "final.pt"
    optimizer = make_optimizer(model, config)

    if history_path.exists() != last_path.exists():
        raise RuntimeError("Resume requires both history.json and last.pt")
    if history_path.exists():
        history = json.loads(history_path.read_text())
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != asdict(config):
            raise RuntimeError(f"Existing fixed-epoch run has another config: {last_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    else:
        history, checkpoint, start_epoch = [], None, 0
    if start_epoch > config.epochs:
        raise ValueError(f"Checkpoint epoch {start_epoch} exceeds target {config.epochs}")

    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        lrs = set_epoch_learning_rates(optimizer, config, epoch, config.epochs)
        train_result = train_epoch(model, loader, sampler, optimizer, device, config, epoch)
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed = time.perf_counter() - started
        completed = epoch + 1
        timing = {
            "train_seconds": epoch_seconds,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": (
                elapsed / (completed - start_epoch) * (config.epochs - completed)),
        }
        history.append({"epoch": completed, "lr": lrs,
                        "train": train_result, "timing": timing})
        write_json(history_path, history)
        checkpoint = {
            "epoch": completed, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "config": asdict(config),
        }
        torch.save(checkpoint, last_path)
        print(" | ".join([
            f"Epoch {completed}/{config.epochs}",
            f"осталось эпох: {config.epochs - completed}",
            f"эпоха: {format_duration(epoch_seconds)}",
            f"прошло: {format_duration(elapsed)}",
            f"ETA: {format_duration(timing['estimated_remaining_seconds'])}",
        ]), flush=True)

    if checkpoint is None:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    torch.save(checkpoint, final_path)
    summary = {
        "seed": config.seed,
        "configured_epochs": config.epochs,
        "completed_epochs": history[-1]["epoch"],
        "epoch_selection": "fixed on inner screening; outer validation not used",
    }
    write_json(summary_path, summary)
    return summary


def run_final_seed(name, config, rows, split, device, results_dir, weights_dir,
                   base_checkpoint, hard_neighbors=None, dataset=DATASET):
    result_dir = Path(results_dir) / "final_seeds" / name
    weight_dir = Path(weights_dir) / "final_seeds" / name
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if "tuned_reranking" in summary:
            if summary.get("config") != asdict(config):
                raise RuntimeError(f"Existing final result has another config: {summary_path}")
            return summary
        checkpoint = torch.load(weight_dir / "final.pt", map_location=device,
                                weights_only=False)
        if checkpoint.get("config") != asdict(config):
            raise RuntimeError(f"Existing final checkpoint has another config: {summary_path}")
        model, loaded = initialize_from_base_checkpoint(
            base_checkpoint, len(split["identities"]["train"]), config, device,
            load_classifier=True)
        model.load_state_dict(checkpoint["model"])
        tuned = tuned_retrieval(model, rows, split, device, dataset, config.num_workers)
        train_id_set = set(split["identities"]["train"])
        summary.update({"name": name, "config": asdict(config),
                        "loaded_base_tensors": loaded,
                        "train_images": sum(row["vehicle_id"] in train_id_set for row in rows),
                        "tuned_reranking": tuned})
        write_json(summary_path, summary)
        del model
        return summary

    set_seed(config.seed)
    selected, labels, sampler, loader = prepare_strategy(
        rows, split["identities"]["train"], config, hard_neighbors, dataset)
    model, loaded = initialize_from_base_checkpoint(
        base_checkpoint, len(labels), config, device, load_classifier=True)
    try:
        summary = fit_fixed_epochs(
            model, loader, sampler, device, config, weight_dir, result_dir)
        tuned = tuned_retrieval(model, rows, split, device, dataset, config.num_workers)
    finally:
        _release(model, loader, sampler, device)
    summary.update({"name": name, "config": asdict(config),
                    "loaded_base_tensors": loaded, "train_images": len(selected),
                    "tuned_reranking": tuned})
    write_json(summary_path, summary)
    return summary


def run_final_comparison(rows, split, screening, device, base_checkpoint,
                         results_dir, weights_dir, seeds=(SEED, SEED + 1, SEED + 2),
                         epochs=None, dataset=DATASET):
    if epochs is None:
        epochs = screening["candidates"][screening["winner"]]["best_epoch"]
    control = base_stage4_config(base_checkpoint, epochs, SEED)
    strategy = replace(ExperimentConfig(**screening["winner_config"]), epochs=epochs)

    neighbors = None
    if strategy.hard_negative_sampling:
        full_rows, full_labels, _, _ = prepare_strategy(
            rows, split["identities"]["train"], control, dataset=dataset)
        prototype_model, _ = initialize_from_base_checkpoint(
            base_checkpoint, len(full_labels), control, device, load_classifier=True)
        try:
            neighbors = cached_hard_neighbors(
                prototype_model, full_rows, device,
                ARTIFACTS / "stage4_hard_neighbors_full.npz", base_checkpoint, dataset)
        finally:
            del prototype_model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    groups = {"control": [], "strategy": []}
    for group, template in (("control", control), ("strategy", strategy)):
        for seed in seeds:
            config = replace(template, seed=seed)
            name = f"{group}_seed_{seed}"
            groups[group].append(run_final_seed(
                name, config, rows, split, device, results_dir, weights_dir,
                base_checkpoint,
                neighbors if config.hard_negative_sampling else None, dataset))

    aggregates = {}
    for group, runs in groups.items():
        maps = [run["tuned_reranking"]["validation"]["mAP_at_10"] for run in runs]
        candidate_scores = [run["tuned_reranking"]["validation"]["candidate_score"] for run in runs]
        quality = [run["tuned_reranking"]["validation_quality_score"] for run in runs]
        aggregates[group] = {
            "mean_mAP_at_10": float(np.mean(maps)),
            "std_mAP_at_10": float(np.std(maps)),
            "mean_candidate_score": float(np.mean(candidate_scores)),
            "mean_quality_score": float(np.mean(quality)),
            "runs": runs,
        }
    control_result, strategy_result = aggregates["control"], aggregates["strategy"]
    strategy_wins = (
        strategy_result["mean_mAP_at_10"] > control_result["mean_mAP_at_10"] and
        strategy_result["mean_quality_score"] > control_result["mean_quality_score"] and
        strategy_result["mean_candidate_score"] >= control_result["mean_candidate_score"] - .005)
    winner = "strategy" if strategy_wins else "control"
    target = aggregates[winner]["mean_mAP_at_10"]
    representative = min(
        aggregates[winner]["runs"],
        key=lambda run: abs(run["tuned_reranking"]["validation"]["mAP_at_10"] - target))
    report = {
        "seeds": list(seeds),
        "fixed_training_epochs": epochs,
        "base_checkpoint": str(Path(base_checkpoint)),
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "active_reference": ACTIVE_REFERENCE,
        "winner": winner,
        "beats_active_reference": (
            aggregates[winner]["mean_mAP_at_10"] > ACTIVE_REFERENCE["mAP_at_10"] and
            aggregates[winner]["mean_quality_score"] > ACTIVE_REFERENCE["quality_score"] and
            aggregates[winner]["mean_candidate_score"] >=
            ACTIVE_REFERENCE["candidate_score"] - .005),
        "strategy_name": screening["winner"],
        "selection_rule": "strategy must improve mean tuned-rerank mAP and quality, "
                          "without losing more than 0.005 candidate score",
        "selected_representative": {
            "name": representative["name"], "seed": representative["seed"],
            "final_epoch": representative["completed_epochs"]},
        "aggregates": aggregates,
    }
    write_json(Path(results_dir) / "final_comparison.json", report)
    return report


def export_stage4_winner(report, split, rows, weights_dir, output_path, metadata_path,
                         dataset=DATASET):
    name = report["selected_representative"]["name"]
    checkpoint_path = Path(weights_dir) / "final_seeds" / name / "final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    model = ReIDExperimentModel(
        len(split["identities"]["train"]), config.use_bnneck,
        config.pooling, config.resize_mode, config.use_mixstyle,
        config.mixstyle_probability, config.mixstyle_alpha)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    export_encoder_onnx(model.inference_module(), output_path, "cpu")

    sample = next(row for row in rows if row["vehicle_id"] in set(split["identities"]["validation"]))
    from PIL import Image
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
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
        "config": checkpoint["config"], "onnx": str(output_path),
        "onnx_sha256": sha256(output_path), "parity": parity,
        "tuned_reranking": next(
            run["tuned_reranking"] for group in report["aggregates"].values()
            for run in group["runs"] if run["name"] == name),
    }
    write_json(metadata_path, metadata)
    return metadata
