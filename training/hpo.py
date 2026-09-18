"""Variant-2 OSNet experiments: HPO, BNNeck, SupCon and checkpoint selection."""
import gc
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from backend.core import DATASET, STOCK_MODEL
from backend.evaluate import SEED, make_protocol, write_json
from backend.scoring import calibrate, metrics, ranked_queries
from training.osnet import VehicleOSNet, load_encoder_from_onnx
from training.pipeline import VehicleDataset, format_duration, set_seed


@dataclass
class ExperimentConfig:
    epochs: int = 8
    identities_per_batch: int = 16
    images_per_identity: int = 2
    encoder_lr: float = 1e-4
    head_lr_multiplier: float = 10.0
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    min_lr_ratio: float = .02
    metric_loss: str = "supcon"
    metric_weight: float = 1.0
    triplet_margin: float = .5
    supcon_temperature: float = .1
    consistency_weight: float = .1
    label_smoothing: float = .1
    use_bnneck: bool = True
    prefer_cross_camera: bool = True
    num_workers: int = 0
    seed: int = SEED
    pooling: str = "avg"
    resize_mode: str = "square"
    use_mixstyle: bool = False
    mixstyle_probability: float = .5
    mixstyle_alpha: float = .1
    hard_negative_sampling: bool = False
    loss_weight_schedule: str = "constant"
    metric_warmup_epochs: int = 4
    circle_margin: float = .25
    circle_gamma: float = 64.

    @property
    def batch_size(self):
        return self.identities_per_batch * self.images_per_identity

    def validate(self):
        if self.epochs < 1 or self.identities_per_batch < 2 or self.images_per_identity < 2:
            raise ValueError("epochs, P and K must be at least 1, 2 and 2")
        if self.encoder_lr <= 0 or self.head_lr_multiplier <= 0 or self.weight_decay < 0:
            raise ValueError("learning rates must be positive and weight decay non-negative")
        if self.metric_loss not in {"triplet", "supcon", "circle"}:
            raise ValueError("metric_loss must be 'triplet', 'supcon' or 'circle'")
        if self.pooling not in {"avg", "gem"}:
            raise ValueError("pooling must be 'avg' or 'gem'")
        if self.resize_mode not in {"square", "letterbox"}:
            raise ValueError("resize_mode must be 'square' or 'letterbox'")
        if not 0 <= self.mixstyle_probability <= 1 or self.mixstyle_alpha <= 0:
            raise ValueError("MixStyle probability must be in [0, 1] and alpha positive")
        if self.loss_weight_schedule not in {"constant", "metric_warmup"}:
            raise ValueError("loss_weight_schedule must be 'constant' or 'metric_warmup'")
        if self.metric_warmup_epochs < 1 or not 0 < self.circle_margin < 1 or self.circle_gamma <= 0:
            raise ValueError("Invalid metric warmup or Circle Loss parameters")
        if not 0 < self.supcon_temperature or not 0 <= self.consistency_weight:
            raise ValueError("temperature must be positive and consistency weight non-negative")


class ReIDExperimentModel(nn.Module):
    """OSNet with an optional BNNeck used by variant 2 only."""

    def __init__(self, num_classes, use_bnneck, pooling="avg", resize_mode="square",
                 use_mixstyle=False, mixstyle_probability=.5, mixstyle_alpha=.1):
        super().__init__()
        self.backbone = VehicleOSNet(
            pooling, use_mixstyle, mixstyle_probability, mixstyle_alpha)
        self.resize_mode = resize_mode
        self.bnneck = nn.BatchNorm1d(512) if use_bnneck else nn.Identity()
        self.classifier = nn.Linear(512, num_classes, bias=not use_bnneck)
        if use_bnneck:
            self.bnneck.bias.requires_grad_(False)
        nn.init.normal_(self.classifier.weight, std=.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def embedding(self, images):
        return self.bnneck(self.backbone(images))

    def forward(self, images):
        raw = self.backbone(images)
        embedding = self.bnneck(raw)
        return self.classifier(embedding), raw, embedding

    def inference_module(self):
        return nn.Sequential(self.backbone, self.bnneck)


class CameraAwarePKBatchSampler(Sampler):
    """P×K sampler that prefers different cameras for positives when available."""

    def __init__(self, rows, identities_per_batch, images_per_identity,
                 prefer_cross_camera=True, seed=SEED):
        self.groups = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            self.groups[row["label"]][row.get("camera_id", 0)].append(index)
        self.identities = sorted(self.groups)
        self.p = identities_per_batch
        self.k = images_per_identity
        self.prefer_cross_camera = prefer_cross_camera
        self.seed = seed
        self.epoch = 0
        if len(self.identities) < self.p:
            raise ValueError("Not enough identities for one P×K batch")

    def __len__(self):
        return len(self.identities) // self.p

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _sample_identity(self, identity, rng):
        by_camera = self.groups[identity]
        cameras = list(by_camera)
        selected = []
        if self.prefer_cross_camera and len(cameras) > 1:
            rng.shuffle(cameras)
            for camera in cameras[:self.k]:
                selected.append(rng.choice(by_camera[camera]))
        choices = [index for values in by_camera.values() for index in values]
        while len(selected) < self.k:
            candidate = rng.choice(choices)
            if candidate not in selected or len(choices) < self.k:
                selected.append(candidate)
        return selected

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        identities = self.identities.copy()
        rng.shuffle(identities)
        usable = len(identities) - len(identities) % self.p
        for start in range(0, usable, self.p):
            batch = []
            for identity in identities[start:start + self.p]:
                batch.extend(self._sample_identity(identity, rng))
            yield batch


def split_hpo_identities(train_identities, seed=SEED, validation_fraction=.2):
    identities = list(train_identities)
    random.Random(seed).shuffle(identities)
    validation_size = max(1, round(len(identities) * validation_fraction))
    return sorted(identities[validation_size:]), sorted(identities[:validation_size])


def prepare_experiment(rows, identities, config, dataset=DATASET):
    config.validate()
    identities = sorted(identities)
    labels = {identity: index for index, identity in enumerate(identities)}
    selected = [{**row, "label": labels[row["vehicle_id"]]}
                for row in rows if row["vehicle_id"] in labels]
    sampler = CameraAwarePKBatchSampler(
        selected, config.identities_per_batch, config.images_per_identity,
        config.prefer_cross_camera, config.seed,
    )
    loader = DataLoader(
        VehicleDataset(selected, dataset, augment=True, resize_mode=config.resize_mode), batch_sampler=sampler,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    return selected, labels, sampler, loader


def initialize_experiment(num_classes, config, device, onnx_path=STOCK_MODEL):
    model = ReIDExperimentModel(
        num_classes, config.use_bnneck, config.pooling, config.resize_mode,
        config.use_mixstyle, config.mixstyle_probability, config.mixstyle_alpha)
    allowed_missing = ("global_pool.p",) if config.pooling == "gem" else ()
    loaded = load_encoder_from_onnx(model.backbone, onnx_path, allowed_missing)
    return model.to(device), loaded


def batch_hard_triplet_loss(embeddings, labels, margin):
    embeddings = F.normalize(embeddings, dim=1)
    distances = torch.cdist(embeddings, embeddings)
    positive = labels[:, None].eq(labels[None, :])
    positive.fill_diagonal_(False)
    negative = ~labels[:, None].eq(labels[None, :])
    hardest_positive = distances.masked_fill(~positive, -torch.inf).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative, torch.inf).min(dim=1).values
    if not torch.isfinite(hardest_positive).all() or not torch.isfinite(hardest_negative).all():
        raise ValueError("Every batch must contain positive and negative pairs")
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def supervised_contrastive_loss(embeddings, labels, temperature=.1):
    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError("Supervised contrastive loss requires a positive for every anchor")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return -(log_prob * positive).sum(dim=1).div(positive.sum(dim=1)).mean()


def circle_loss(embeddings, labels, margin=.25, gamma=64.):
    """Pair-wise Circle Loss over all positive and negative pairs in a P×K batch."""
    embeddings = F.normalize(embeddings, dim=1)
    similarities = embeddings @ embeddings.T
    same = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = similarities[same & ~diagonal]
    negative = similarities[~same]
    if not len(positive) or not len(negative):
        raise ValueError("Circle Loss requires positive and negative pairs")
    positive_weight = torch.clamp_min(1 + margin - positive.detach(), 0.)
    negative_weight = torch.clamp_min(negative.detach() + margin, 0.)
    positive_logits = -gamma * positive_weight * (positive - (1 - margin))
    negative_logits = gamma * negative_weight * (negative - margin)
    return F.softplus(torch.logsumexp(positive_logits, dim=0) +
                      torch.logsumexp(negative_logits, dim=0))


def metric_weight_for_epoch(config, epoch):
    if config.loss_weight_schedule == "constant" or config.metric_warmup_epochs == 1:
        return config.metric_weight
    progress = min(1., epoch / (config.metric_warmup_epochs - 1))
    return config.metric_weight * (.25 + .75 * progress)


def experiment_losses(model, clean_images, robust_images, labels, config, metric_weight=None):
    model.eval()
    with torch.no_grad():
        clean_embedding = model.embedding(clean_images)
    model.train()
    logits, raw_embedding, robust_embedding = model(robust_images)
    classification = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
    if config.metric_loss == "triplet":
        metric = batch_hard_triplet_loss(raw_embedding, labels, config.triplet_margin)
    elif config.metric_loss == "supcon":
        metric = supervised_contrastive_loss(raw_embedding, labels, config.supcon_temperature)
    else:
        metric = circle_loss(raw_embedding, labels, config.circle_margin, config.circle_gamma)
    consistency = (1 - F.cosine_similarity(robust_embedding, clean_embedding.detach(), dim=1)).mean()
    metric_weight = config.metric_weight if metric_weight is None else metric_weight
    loss = classification + metric_weight * metric + config.consistency_weight * consistency
    return {"loss": loss, "classification": classification, "metric": metric,
            "consistency": consistency, "accuracy": (logits.argmax(1) == labels).float().mean()}


def make_optimizer(model, config):
    encoder = list(model.backbone.parameters())
    head = list(model.bnneck.parameters()) + list(model.classifier.parameters())
    return torch.optim.AdamW([
        {"params": encoder, "lr": config.encoder_lr, "base_lr": config.encoder_lr, "name": "encoder"},
        {"params": head, "lr": config.encoder_lr * config.head_lr_multiplier,
         "base_lr": config.encoder_lr * config.head_lr_multiplier, "name": "head"},
    ], weight_decay=config.weight_decay)


def set_epoch_learning_rates(optimizer, config, epoch, total_epochs):
    if epoch < config.warmup_epochs:
        factor = (epoch + 1) / max(1, config.warmup_epochs)
    else:
        decay_epochs = max(1, total_epochs - config.warmup_epochs - 1)
        progress = min(1., (epoch - config.warmup_epochs) / decay_epochs)
        cosine = .5 * (1 + math.cos(math.pi * progress))
        factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def train_epoch(model, loader, sampler, optimizer, device, config, epoch):
    model.train()
    sampler.set_epoch(epoch)
    totals = defaultdict(float)
    metric_weight = metric_weight_for_epoch(config, epoch)
    for clean, robust, labels, _ in tqdm(loader, desc=f"epoch {epoch + 1}", leave=False):
        clean, robust, labels = clean.to(device), robust.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = experiment_losses(model, clean, robust, labels, config, metric_weight)
        losses["loss"].backward()
        optimizer.step()
        for name, value in losses.items():
            totals[name] += value.item()
        totals["batches"] += 1
    result = {key: value / totals["batches"] for key, value in totals.items() if key != "batches"}
    result["metric_weight"] = metric_weight
    return result


@torch.inference_mode()
def encode_experiment(model, rows, device, dataset=DATASET, batch_size=64, num_workers=0):
    prepared = [{**row, "label": 0} for row in rows]
    loader = DataLoader(VehicleDataset(prepared, dataset, augment=False,
                                       resize_mode=getattr(model, "resize_mode", "square")), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    model.eval()
    vectors = []
    for images, _, _ in loader:
        vectors.append(F.normalize(model.embedding(images.to(device)), dim=1).cpu().numpy())
    return np.concatenate(vectors).astype(np.float32)


def rank_experiment(model, rows, identities, device, dataset=DATASET, seed=SEED, num_workers=0):
    query, gallery = make_protocol(rows, identities, seed)
    selected = list({row["image_id"]: row for row in query + gallery}.values())
    vectors = encode_experiment(model, selected, device, dataset, num_workers=num_workers)
    embeddings = dict(zip((row["image_id"] for row in selected), vectors))
    return ranked_queries(query, gallery, embeddings)


def evaluate_experiment(model, rows, identities, device, dataset=DATASET, seed=SEED,
                        threshold=None, num_workers=0):
    ranked = rank_experiment(model, rows, identities, device, dataset, seed, num_workers)
    threshold = calibrate(ranked) if threshold is None else threshold
    return metrics(ranked, threshold), threshold


def _timing(started, epoch_started, train_seconds, completed, total_epochs):
    epoch_seconds = time.perf_counter() - epoch_started
    elapsed = time.perf_counter() - started
    return {"train_seconds": train_seconds, "evaluation_seconds": epoch_seconds - train_seconds,
            "epoch_seconds": epoch_seconds, "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": elapsed / completed * (total_epochs - completed)}


def _print_progress(epoch, total, timing, best_map):
    print(" | ".join([
        f"Epoch {epoch}/{total}", f"осталось эпох: {total - epoch}",
        f"эпоха: {format_duration(timing['epoch_seconds'])}",
        f"прошло: {format_duration(timing['elapsed_seconds'])}",
        f"ETA: {format_duration(timing['estimated_remaining_seconds'])}", f"best mAP: {best_map:.4f}",
    ]), flush=True)


def _inner_checkpoint_payload(epoch, model, optimizer, config, threshold, validation):
    return {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": asdict(config), "threshold": threshold, "validation": validation}


def fit_trial(model, rows, validation_ids, loader, sampler, device, config, trial,
              output_path, weights_dir):
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(model, config)
    history = []
    started = time.perf_counter()
    best_map = -1.
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        lrs = set_epoch_learning_rates(optimizer, config, epoch, config.epochs)
        train_result = train_epoch(model, loader, sampler, optimizer, device, config, epoch)
        train_seconds = time.perf_counter() - epoch_started
        validation, threshold = evaluate_experiment(
            model, rows, validation_ids, device, seed=config.seed, num_workers=config.num_workers,
        )
        timing = _timing(started, epoch_started, train_seconds, epoch + 1, config.epochs)
        record = {"epoch": epoch + 1, "lr": lrs, "train": train_result,
                  "threshold": threshold, "validation": validation, "timing": timing}
        history.append(record)
        write_json(output_path, history)
        payload = _inner_checkpoint_payload(
            epoch + 1, model, optimizer, config, threshold, validation,
        )
        torch.save(payload, weights_dir / "last.pt")
        if validation["mAP"] > best_map:
            best_map = validation["mAP"]
            torch.save(payload, weights_dir / "best_map.pt")
        _print_progress(epoch + 1, config.epochs, timing, best_map)
        trial.report(validation["mAP"], epoch)
        if trial.should_prune():
            best = max(history, key=lambda item: item["validation"]["mAP"])
            trial.set_user_attr("best_epoch", best["epoch"])
            trial.set_user_attr("best_validation", best["validation"])
            raise optuna.TrialPruned()
    best = max(history, key=lambda item: item["validation"]["mAP"])
    trial.set_user_attr("best_epoch", best["epoch"])
    trial.set_user_attr("best_validation", best["validation"])
    return best["validation"]["mAP"]


def suggest_config(trial, epochs):
    metric_loss = trial.suggest_categorical("metric_loss", ["triplet", "supcon"])
    return ExperimentConfig(
        epochs=epochs,
        identities_per_batch=16,
        images_per_identity=2,
        encoder_lr=trial.suggest_float("encoder_lr", 5e-5, 3e-4, log=True),
        head_lr_multiplier=trial.suggest_categorical("head_lr_multiplier", [5., 10.]),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        metric_loss=metric_loss,
        metric_weight=trial.suggest_float("metric_weight", .5, 1.5),
        triplet_margin=trial.suggest_float("triplet_margin", .3, .6) if metric_loss == "triplet" else .5,
        consistency_weight=trial.suggest_float("consistency_weight", .05, .25),
        use_bnneck=trial.suggest_categorical("use_bnneck", [False, True]),
        # Keep the stochastic data order identical across trials so the HPO
        # comparison measures parameters rather than a different lucky seed.
        seed=SEED,
    )


def config_from_trial(trial, epochs, seed=SEED):
    params = trial.params
    return ExperimentConfig(
        epochs=epochs,
        identities_per_batch=16,
        images_per_identity=2,
        encoder_lr=params["encoder_lr"],
        head_lr_multiplier=params["head_lr_multiplier"],
        weight_decay=params["weight_decay"],
        metric_loss=params["metric_loss"],
        metric_weight=params["metric_weight"],
        triplet_margin=params.get("triplet_margin", .5),
        consistency_weight=params["consistency_weight"],
        use_bnneck=params["use_bnneck"],
        seed=seed,
    )


def _release_device(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def run_hpo(rows, split, device, results_dir, weights_dir, target_trials=16,
            trial_epochs=8, dataset=DATASET):
    results_dir = Path(results_dir)
    weights_dir = Path(weights_dir)
    trials_dir = results_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    hpo_train, hpo_validation = split_hpo_identities(split["identities"]["train"], SEED)
    write_json(results_dir / "hpo_split.json", {
        "seed": SEED, "train_identities": hpo_train, "validation_identities": hpo_validation,
        "outer_calibration_and_validation_used_by_hpo": False,
    })
    storage = f"sqlite:///{(results_dir / 'optuna.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name="osnet_ain_x1_0_variant_02", storage=storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
        # step=3 is the fourth completed epoch, so weak trials are never
        # stopped before the requested 3-4 epoch warm-up.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4, n_warmup_steps=3, interval_steps=1, n_min_trials=2,
        ),
    )

    def objective(trial):
        config = suggest_config(trial, trial_epochs)
        trial.set_user_attr("config", asdict(config))
        set_seed(config.seed)
        selected, labels, sampler, loader = prepare_experiment(rows, hpo_train, config, dataset)
        model, loaded = initialize_experiment(len(labels), config, device)
        trial.set_user_attr("loaded_stock_tensors", loaded)
        trial.set_user_attr("train_images", len(selected))
        try:
            return fit_trial(model, rows, hpo_validation, loader, sampler, device, config, trial,
                             trials_dir / f"trial_{trial.number:03d}.json",
                             weights_dir / "stage1" / f"trial_{trial.number:03d}")
        finally:
            del model, loader, sampler
            _release_device(device)

    finished = sum(t.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
                   for t in study.trials)
    if finished < target_trials:
        study.optimize(objective, n_trials=target_trials - finished, gc_after_trial=True)
    write_json(results_dir / "study_summary.json", {
        "target_trials": target_trials,
        "trial_epochs": trial_epochs,
        "pruning_can_start_after_epoch": 4,
        "completed_or_pruned": sum(t.state in {optuna.trial.TrialState.COMPLETE,
                                                 optuna.trial.TrialState.PRUNED} for t in study.trials),
        "best_trial": study.best_trial.number, "best_mAP": study.best_value,
        "best_params": study.best_trial.params,
        "trials": [{"number": t.number, "state": t.state.name, "value": t.value,
                    "params": t.params, "user_attrs": t.user_attrs} for t in study.trials],
    })
    return study


def _rank_complete_trials(study, limit):
    complete = [trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    complete.sort(key=lambda trial: trial.value, reverse=True)
    if len(complete) < limit:
        raise RuntimeError(f"Нужно минимум {limit} завершённых trials, получено {len(complete)}")
    return complete[:limit]


def fit_inner_candidate(model, rows, validation_ids, loader, sampler, device, config,
                        weights_dir, results_dir, initial_checkpoint=None,
                        initial_best_checkpoint=None, initial_history_path=None):
    """Train or resume one promoted candidate on the inner HPO split."""
    weights_dir, results_dir = Path(weights_dir), Path(results_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "history.json"
    last_path = weights_dir / "last.pt"
    best_path = weights_dir / "best_map.pt"
    optimizer = make_optimizer(model, config)

    if history_path.exists() and last_path.exists():
        history = json.loads(history_path.read_text())
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    elif initial_checkpoint is not None:
        history = json.loads(Path(initial_history_path).read_text())
        checkpoint = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        if initial_best_checkpoint is not None and not best_path.exists():
            shutil.copy2(initial_best_checkpoint, best_path)
    else:
        history = []
        checkpoint = None

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    if start_epoch > config.epochs:
        raise ValueError(f"Checkpoint epoch {start_epoch} exceeds target {config.epochs}")

    best_record = max(history, key=lambda item: item["validation"]["mAP"], default=None)
    best_map = best_record["validation"]["mAP"] if best_record else -1.
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        lrs = set_epoch_learning_rates(optimizer, config, epoch, config.epochs)
        train_result = train_epoch(model, loader, sampler, optimizer, device, config, epoch)
        train_seconds = time.perf_counter() - epoch_started
        validation, threshold = evaluate_experiment(
            model, rows, validation_ids, device, seed=config.seed,
            num_workers=config.num_workers,
        )
        timing = _timing(started, epoch_started, train_seconds,
                         epoch - start_epoch + 1, config.epochs - start_epoch)
        record = {"epoch": epoch + 1, "lr": lrs, "train": train_result,
                  "threshold": threshold, "validation": validation, "timing": timing}
        history.append(record)
        write_json(history_path, history)
        payload = _inner_checkpoint_payload(
            epoch + 1, model, optimizer, config, threshold, validation,
        )
        torch.save(payload, last_path)
        if validation["mAP"] > best_map:
            best_map = validation["mAP"]
            best_record = record
            torch.save(payload, best_path)
        _print_progress(epoch + 1, config.epochs, timing, best_map)

    best_record = max(history, key=lambda item: item["validation"]["mAP"])
    result = {"best_epoch": best_record["epoch"],
              "best_mAP": best_record["validation"]["mAP"],
              "best_validation": best_record["validation"],
              "completed_epochs": history[-1]["epoch"]}
    write_json(results_dir / "summary.json", result)
    return result


def continue_top_candidates(rows, split, study, device, results_dir, weights_dir,
                            top_k=4, target_epochs=20, dataset=DATASET):
    """Continue the best completed HPO trials from epoch 8 to epoch 20."""
    results_dir, weights_dir = Path(results_dir), Path(weights_dir)
    hpo_train, hpo_validation = split_hpo_identities(split["identities"]["train"], SEED)
    candidates = []
    for trial in _rank_complete_trials(study, top_k):
        config = config_from_trial(trial, target_epochs, seed=SEED)
        set_seed(config.seed)
        selected, labels, sampler, loader = prepare_experiment(rows, hpo_train, config, dataset)
        model, loaded = initialize_experiment(len(labels), config, device)
        trial_name = f"trial_{trial.number:03d}"
        try:
            result = fit_inner_candidate(
                model, rows, hpo_validation, loader, sampler, device, config,
                weights_dir / "stage2_top4" / trial_name,
                results_dir / "stage2_top4" / trial_name,
                initial_checkpoint=weights_dir / "stage1" / trial_name / "last.pt",
                initial_best_checkpoint=weights_dir / "stage1" / trial_name / "best_map.pt",
                initial_history_path=results_dir / "trials" / f"{trial_name}.json",
            )
        finally:
            del model, loader, sampler
            _release_device(device)
        candidates.append({"trial_number": trial.number, "stage1_mAP": trial.value,
                           "config": asdict(config), "loaded_stock_tensors": loaded,
                           "train_images": len(selected), **result})
    candidates.sort(key=lambda item: item["best_mAP"], reverse=True)
    summary = {"top_k": top_k, "target_epochs": target_epochs, "candidates": candidates}
    write_json(results_dir / "stage2_top4_summary.json", summary)
    return summary


def run_finalist_seeds(rows, split, study, stage2_summary, device, results_dir, weights_dir,
                       top_k=2, target_epochs=30,
                       seeds=(SEED, SEED + 1, SEED + 2), dataset=DATASET):
    """Retrain the two finalists independently and rank them by mean seed mAP."""
    results_dir, weights_dir = Path(results_dir), Path(weights_dir)
    hpo_train, hpo_validation = split_hpo_identities(split["identities"]["train"], SEED)
    trial_lookup = {trial.number: trial for trial in study.trials}
    finalists = []
    for candidate in stage2_summary["candidates"][:top_k]:
        trial = trial_lookup[candidate["trial_number"]]
        runs = []
        for seed in seeds:
            config = config_from_trial(trial, target_epochs, seed=seed)
            set_seed(seed)
            selected, labels, sampler, loader = prepare_experiment(rows, hpo_train, config, dataset)
            model, loaded = initialize_experiment(len(labels), config, device)
            run_name = f"trial_{trial.number:03d}_seed_{seed}"
            try:
                result = fit_inner_candidate(
                    model, rows, hpo_validation, loader, sampler, device, config,
                    weights_dir / "stage3_top2_seeds" / run_name,
                    results_dir / "stage3_top2_seeds" / run_name,
                )
            finally:
                del model, loader, sampler
                _release_device(device)
            runs.append({"seed": seed, "config": asdict(config),
                         "loaded_stock_tensors": loaded, "train_images": len(selected), **result})
        scores = [run["best_mAP"] for run in runs]
        finalists.append({"trial_number": trial.number, "stage2_mAP": candidate["best_mAP"],
                          "mean_best_mAP": float(np.mean(scores)),
                          "std_best_mAP": float(np.std(scores)), "runs": runs})
    finalists.sort(key=lambda item: item["mean_best_mAP"], reverse=True)
    summary = {"top_k": top_k, "target_epochs": target_epochs, "seeds": list(seeds),
               "selection_metric": "mean of per-seed best inner-validation mAP",
               "best_trial_number": finalists[0]["trial_number"], "finalists": finalists}
    write_json(results_dir / "stage3_top2_seeds_summary.json", summary)
    return summary


def load_hpo_study(results_dir):
    results_dir = Path(results_dir)
    return optuna.load_study(study_name="osnet_ain_x1_0_variant_02",
                             storage=f"sqlite:///{(results_dir / 'optuna.sqlite3').resolve()}")


def _checkpoint_payload(epoch, model, optimizer, config, threshold, calibration, validation):
    return {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": asdict(config), "threshold": threshold,
            "calibration": calibration, "validation": validation}


def fit_selected(model, rows, split, loader, sampler, device, config, weights_dir, results_dir):
    """Train the chosen config and preserve every epoch that improves a key metric."""
    weights_dir, results_dir = Path(weights_dir), Path(results_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(model, config)
    history = []
    best = {"map": {"value": -1., "epoch": None},
            "f1": {"value": -1., "epoch": None},
            "tnr": {"value": -1., "epoch": None}}
    started = time.perf_counter()
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        lrs = set_epoch_learning_rates(optimizer, config, epoch, config.epochs)
        train_result = train_epoch(model, loader, sampler, optimizer, device, config, epoch)
        train_seconds = time.perf_counter() - epoch_started
        calibration, threshold = evaluate_experiment(
            model, rows, split["identities"]["calibration"], device,
            seed=config.seed, num_workers=config.num_workers,
        )
        validation, _ = evaluate_experiment(
            model, rows, split["identities"]["validation"], device,
            seed=config.seed, threshold=threshold, num_workers=config.num_workers,
        )
        timing = _timing(started, epoch_started, train_seconds, epoch + 1, config.epochs)
        record = {"epoch": epoch + 1, "lr": lrs, "train": train_result,
                  "threshold": threshold, "calibration": calibration,
                  "validation": validation, "timing": timing}
        history.append(record)
        write_json(results_dir / "history.json", history)
        payload = _checkpoint_payload(epoch + 1, model, optimizer, config,
                                      threshold, calibration, validation)
        torch.save(payload, weights_dir / "last.pt")
        improved = []
        for tag, metric_name in (("map", "mAP"), ("f1", "candidate_F1"), ("tnr", "TNR")):
            value = validation[metric_name]
            if value is not None and value > best[tag]["value"]:
                best[tag] = {"value": value, "epoch": epoch + 1}
                torch.save(payload, weights_dir / f"best_{tag}.pt")
                improved.append(tag)
        if improved:
            torch.save(payload, weights_dir / f"epoch_{epoch + 1:02d}.pt")
        write_json(results_dir / "checkpoint_index.json", {
            "best": best, "last_epoch": epoch + 1,
            "important_epoch_files": sorted(path.name for path in weights_dir.glob("epoch_*.pt")),
        })
        _print_progress(epoch + 1, config.epochs, timing, best["map"]["value"])
        print(json.dumps({**record, "improved_checkpoints": improved}, ensure_ascii=False, indent=2))
    return history, best
