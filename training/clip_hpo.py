"""Isolated, resumable CLIP-ReID search. Never trains on outer/test or updates MVP.

Variant 1 modules are deliberately left unchanged: their hashes protect existing
checkpoints. Every rung uses the same 60-epoch schedule and source initialization.
"""
import fcntl
import json
import math
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from backend.core import DATASET, ROOT, STOCK_MODEL, sha256
from backend.evaluate import SEED
from training.audit import digest
from training.clip_experiment import (ClipConfig, class_text_features, evaluate_inner,
                                      image_lr_factor, prepare_protocol, run_prompt_stage,
                                      run_signature, training_rows)
from training.clip_reid import ClipDataset, load_pretrained, release_device, synchronize, triplet_loss
from training.clip_source import CHECKPOINT_SHA256
from training.pipeline import format_duration, set_seed
from training.stage6 import StepPKBatchSampler, _save_checkpoint, write_json


MODEL_DIR = ROOT / "CLIP-ReID-ViT-B-16"
PREVIOUS = MODEL_DIR / "variant_01_vehicle_transfer"
VARIANT = MODEL_DIR / "variant_02_hpo"
SOURCE = PREVIOUS / "weights/VeRi_clipreid_ViT-B-16_60.pth"
SPACE = {"image_lr": [2e-6, 2e-5], "head_lr_multiplier": [1., 5., 10., 20.],
         "weight_decay": [1e-6, 1e-3], "batch_layout": ["8x4", "16x2"],
         "id_loss_weight": [.25, .5, 1.]}
BASE_PARAMS = {"image_lr": 5e-6, "head_lr_multiplier": 1., "weight_decay": 1e-4,
               "batch_layout": "8x4", "id_loss_weight": .25}
ANCHORS = [BASE_PARAMS, {**BASE_PARAMS, "head_lr_multiplier": 10.},
           {**BASE_PARAMS, "batch_layout": "16x2"},
           {**BASE_PARAMS, "id_loss_weight": 1.}]


@dataclass(frozen=True)
class TrialConfig(ClipConfig):
    head_lr_multiplier: float = 1.
    id_loss_weight: float = .25

    def validate(self):
        super().validate()
        if self.image_epochs != 60:
            raise ValueError("The CLIP HPO scheduler horizon must remain 60 epochs")
        values = (self.image_lr, self.weight_decay, self.head_lr_multiplier, self.id_loss_weight)
        if not all(math.isfinite(x) for x in values) or min(values) <= 0:
            raise ValueError("HPO rates and loss weights must be finite and positive")


@dataclass(frozen=True)
class SearchPlan:
    trials: int = 12
    screen_epochs: int = 15
    top_k: int = 4
    promotion_epochs: int = 35
    finalists: int = 2
    final_epochs: int = 60
    prune_after_epoch: int = 12
    extra_seeds: tuple = (SEED + 1, SEED + 2)
    # A budget gate, not statistical significance. Extra seeds confirm this decision.
    confirmation_min_gain: float = .005

    def validate(self):
        if not 1 <= self.finalists <= self.top_k <= self.trials or self.trials < len(ANCHORS):
            raise ValueError("Require 1 <= finalists <= top_k <= trials, at least four anchors")
        if not 10 < self.prune_after_epoch <= self.screen_epochs < self.promotion_epochs < self.final_epochs == 60:
            raise ValueError("Pruning must follow warmup; require screen < promotion < final=60")
        if len(set(self.extra_seeds)) != len(self.extra_seeds) or SEED in self.extra_seeds:
            raise ValueError("Confirmation seeds must be unique and different from the search seed")
        if not math.isfinite(self.confirmation_min_gain) or self.confirmation_min_gain < 0:
            raise ValueError("Invalid confirmation gain")


@contextmanager
def experiment_lock(directory):
    """Kernel/process death releases the OS lock; no stale-PID guessing/deletion."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This HPO directory is already running in another kernel") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def require_disk(directory, gib=3):
    free = shutil.disk_usage(directory).free / 2**30
    if free < gib:
        raise OSError(f"Only {free:.1f} GiB free; need {gib} GiB for safe checkpoint writes. "
                      "Free space and Run All again; saved epochs will resume.")
    return free


def manifest(protocol, plan):
    sources = ["training/clip_hpo.py", "training/clip_experiment.py", "training/clip_reid.py",
               "training/clip_source.py", "training/vendor/clip_reid/model.py", "training/stage6.py",
               "training/hpo.py", "training/pipeline.py", "backend/core.py", "backend/evaluate.py",
               "backend/scoring.py", "evaluate.py"]
    return {"version": 1, "protocol": digest(protocol), "plan": asdict(plan), "search_space": SPACE,
            "anchors": ANCHORS, "prompt_config": asdict(ClipConfig()),
            "code": {name: sha256(ROOT / name) for name in sources},
            "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "optuna", "Pillow")}}


def checked_json(path, expected):
    """Freeze decisions before continuation; JSON tuple/list roundtrip is canonical."""
    path = Path(path)
    if path.exists() and digest(json.loads(path.read_text())) != digest(expected):
        raise RuntimeError(f"Configuration/protocol/code changed: {path}. Use a new variant directory.")
    write_json(path, expected)


def load_references(protocol, previous=PREVIOUS):
    """Only same-protocol references; never take an old outer mAP as an inner target."""
    previous = Path(previous)
    old_protocol = json.loads((previous / "results/protocol.json").read_text())
    if old_protocol != protocol:
        raise ValueError("Variant 1 reference has a different protocol/data/evaluator")
    clip = json.loads((previous / "results/training_summary.json").read_text())
    expected = run_signature(protocol, ClipConfig())
    if {k: v for k, v in clip["signature"].items() if k != "prompt_sha256"} != expected:
        raise ValueError("Variant 1 code/config differs; refresh references in a separate experiment")
    osnet = json.loads((previous / "results/osnet_control/training_summary.json").read_text())
    control_protocol = {**protocol, "initializer_sha256": sha256(STOCK_MODEL),
                        "control_recipe": "stock OSNet + avg/BNNeck/SupCon; inner-only; 4000 steps max"}
    if osnet["protocol_sha256"] != digest(control_protocol):
        raise ValueError("OSNet control is not from the same inner split")
    return {"clip_variant_1": clip["best"]["validation"]["mAP_at_10"],
            "clean_osnet": osnet["best_mAP_at_10"], "protocol": digest(protocol),
            "selection_metric": "organizer raw inner mAP@10", "outer_test_used": False}


def prompt_assets(rows, protocol, device, source, weights, results, seed=SEED, previous=PREVIOUS, dataset=DATASET):
    """Same fixed prompt recipe per seed; first seed may reuse verified variant 1.

    Image LR/PK/head/loss HPO does not change the prompt stage. Different seeds
    train their own prompts from scratch. Only local copies are ever written.
    """
    config = ClipConfig(seed=seed)
    signature = run_signature(protocol, config)
    folder, output = Path(weights) / "prompts" / str(seed), Path(results) / "prompts" / str(seed)
    checkpoint = folder / "prompt_last.pt"
    if not checkpoint.exists() and seed == SEED:
        original = Path(previous) / "weights/prompt_last.pt"
        if original.exists():
            saved = torch.load(original, map_location="cpu", weights_only=True)
            if saved["signature"] != signature or len(saved["history"]) != config.prompt_epochs:
                raise ValueError("Variant 1 prompt checkpoint is incomplete or incompatible")
            _save_checkpoint(checkpoint, saved)
            write_json(output / "provenance.json", {"reused_from": str(original), "sha256": sha256(original)})
            del saved
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
    if saved and saved["signature"] != signature:
        raise RuntimeError("Prompt signature changed; use a new variant directory")
    complete = saved is not None and len(saved["history"]) == config.prompt_epochs
    if complete:
        write_json(output / "prompt_history.json", saved["history"])
        cache = folder / "text_features.pt"
        fingerprint = sha256(checkpoint)
        if cache.exists():
            asset = torch.load(cache, map_location="cpu", weights_only=True)
            if asset["prompt_sha256"] != fingerprint or asset["signature"] != signature:
                raise RuntimeError("Text feature cache does not match its prompts")
            return asset
    set_seed(seed)
    model = load_pretrained(source, classes=len(protocol["inner"]["train"]), device=device)
    try:
        if complete:
            model.prompt_learner.load_state_dict(saved["prompt"])
        else:
            run_prompt_stage(model, training_rows(rows, protocol), device, config, protocol,
                             folder, output, dataset)
        model.eval()
        features = class_text_features(model, device).detach().cpu()
        if features.shape != (len(protocol["inner"]["train"]), 512) or not torch.isfinite(features).all():
            raise FloatingPointError("Invalid prompt text features")
        asset = {"features": features, "prompt_sha256": sha256(checkpoint), "signature": signature}
        _save_checkpoint(folder / "text_features.pt", asset)
        return asset
    finally:
        del model
        release_device(device)


def suggest_config(trial, seed=SEED):
    layout = trial.suggest_categorical("batch_layout", SPACE["batch_layout"])
    p, k = map(int, layout.split("x"))
    return TrialConfig(seed=seed, identities_per_batch=p, images_per_identity=k,
                       image_lr=trial.suggest_float("image_lr", *SPACE["image_lr"], log=True),
                       head_lr_multiplier=trial.suggest_categorical("head_lr_multiplier", SPACE["head_lr_multiplier"]),
                       weight_decay=trial.suggest_float("weight_decay", *SPACE["weight_decay"], log=True),
                       id_loss_weight=trial.suggest_categorical("id_loss_weight", SPACE["id_loss_weight"]))


def make_optimizer(model, config):
    model.set_stage(2)
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Only newly initialized classification weights get the head multiplier.
        # Loaded BNNeck parameters keep the encoder rate and original bias rule.
        multiplier = config.head_lr_multiplier if name.startswith(("classifier.", "classifier_proj.")) else 1.
        rate = config.image_lr * multiplier * (2 if "bias" in name else 1)
        groups.append({"params": [parameter], "lr": rate, "base_lr": rate, "name": name})
    return torch.optim.Adam(groups, weight_decay=config.weight_decay)


def losses_and_accuracy(model, images, labels, texts, config):
    scores, features, projected = model(images)
    identity = sum(F.cross_entropy(score, labels, label_smoothing=.1) for score in scores)
    triplet = sum(triplet_loss(feature, labels) for feature in features)
    logits = projected @ texts.T
    image_text = F.cross_entropy(logits, labels, label_smoothing=.1)
    return {"loss": config.id_loss_weight * identity + triplet + image_text,
            "id_loss": identity, "triplet": triplet, "image_text": image_text,
            "accuracy_head_768": (scores[0].argmax(1) == labels).float().mean(),
            "accuracy_head_512": (scores[1].argmax(1) == labels).float().mean(),
            "accuracy_image_text": (logits.argmax(1) == labels).float().mean()}


def technical_smoke(rows, protocol, device, source=SOURCE, dataset=DATASET):
    """One real P16/K2 Adam step with 10x head LR; its model is then discarded."""
    config = TrialConfig(identities_per_batch=16, images_per_identity=2,
                         head_lr_multiplier=10., id_loss_weight=1.)
    set_seed(config.seed)
    model = load_pretrained(source, classes=len(protocol["inner"]["train"]), device=device)
    try:
        model.eval()
        texts = class_text_features(model, device).detach()
        selected = training_rows(rows, protocol)
        sampler = StepPKBatchSampler(selected, config, steps=1)
        loader = DataLoader(ClipDataset(selected, dataset, train=True), batch_sampler=sampler, num_workers=0)
        images, labels, _ = next(iter(loader))
        optimizer = make_optimizer(model, config)
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        assert math.isclose(rates["classifier.weight"], config.image_lr * 10)
        assert math.isclose(rates["classifier_proj.weight"], config.image_lr * 10)
        synchronize(device)
        started = time.perf_counter()
        values = losses_and_accuracy(model, images.to(device), labels.to(device), texts, config)
        values["loss"].backward()
        if not all(torch.isfinite(value) for value in values.values()) or any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("CLIP HPO smoke: non-finite loss/gradient")
        optimizer.step()
        synchronize(device)
        return {"device": str(device), "batch": 32, "layout": "16x2", "head_lr_multiplier": 10,
                "backward_and_adam_checked": True, "seconds": time.perf_counter()-started,
                "diagnostics": {name: float(value.detach().cpu()) for name, value in values.items()},
                "not_a_quality_result": True, "saved_training_weights": False}
    finally:
        del model
        release_device(device)


def train_epoch(model, loader, sampler, texts, optimizer, device, config, epoch):
    set_seed(config.seed + 10000 + epoch)
    sampler.set_epoch(epoch)
    model.set_stage(2)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * image_lr_factor(epoch)
    totals, seen = defaultdict(float), set()
    for images, labels, ids in tqdm(loader, desc=f"CLIP image {epoch}/60", leave=False):
        optimizer.zero_grad(set_to_none=True)
        values = losses_and_accuracy(model, images.to(device), labels.to(device), texts, config)
        if not all(torch.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite CLIP loss/accuracy")
        values["loss"].backward()
        optimizer.step()
        for name, value in values.items():
            totals[name] += float(value.detach().cpu())
        seen.update(ids)
    optimizer.zero_grad(set_to_none=True)
    return {name: value / len(loader) for name, value in totals.items()}, seen


def _result(state, config, status):
    return {"status": status, "config": asdict(config), "completed_epochs": len(state["history"]),
            "best": state["best"], "signature": state["signature"],
            "seconds": sum(r["epoch_seconds"] for r in state["history"]),
            "outer_test_used": False, "mvp_changed": False}


def _save_reports(state, config, status, weights, output):
    result = _result(state, config, status)
    _save_checkpoint(weights / "image_best.pt", {"signature": state["signature"], **state["best"],
                     "model": state["best_model"], "config": asdict(config)})
    write_json(output / "image_history.json", state["history"])
    write_json(output / "summary.json", result)
    return result


def _report_trial(trial, state, prune_after):
    if trial is None or not state["history"]:
        return False
    # Replay any reports missed between the atomic checkpoint and SQLite commit.
    reported = trial.study.get_trials()[trial.number].intermediate_values
    for record in state["history"]:
        step = record["epoch"] - 1
        if step not in reported:
            trial.report(record["validation"]["mAP_at_10"], step)
    return len(state["history"]) >= prune_after and trial.should_prune()


def fit_candidate(model, rows, protocol, asset, device, config, target_epochs, weights, output,
                  experiment_signature, trial=None, prune_after=12, dataset=DATASET, progress=None):
    """Atomic authoritative last.pt owns optimizer, history AND best; resumes a rung.

    An interrupted epoch is replayed, never skipped. Targets are NOT signatures:
    raising 15 -> 35 -> 60 must not reset optimizer or alter earlier LR values.
    """
    config.validate()
    if not 1 <= target_epochs <= config.image_epochs:
        raise ValueError("Target must be within the fixed 60-epoch horizon")
    if asset["signature"]["config"]["seed"] != config.seed:
        raise ValueError("Each confirmation seed requires its own prompt stage")
    if not torch.isfinite(asset["features"]).all():
        raise FloatingPointError("Invalid cached text features")
    weights, output = Path(weights), Path(output)
    weights.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    signature = {"experiment": experiment_signature, "config": asdict(config),
                 "prompt": asset["prompt_sha256"], "protocol": digest(protocol)}
    optimizer = make_optimizer(model, config)
    last = weights / "image_last.pt"
    state = torch.load(last, map_location="cpu", weights_only=True) if last.exists() else None
    if state:
        if state["signature"] != signature:
            raise RuntimeError("Candidate resume signature changed; use a new variant directory")
        model.load_image_state(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Do not hold duplicate CPU Adam tensors throughout an MPS epoch.
        del state["optimizer"], state["model"]
    else:
        if (output / "summary.json").exists() or (weights / "image_best.pt").exists():
            raise RuntimeError("Missing authoritative image_last.pt; restore it, do not restart silently")
        validation = evaluate_inner(model, rows, protocol, device, dataset, config.eval_batch)
        state = {"signature": signature, "history": [], "best": {"epoch": 0, "validation": validation},
                 "best_model": model.image_state(), "seen": []}
    if len(state["history"]) > target_epochs:
        raise ValueError("Cannot rewind a candidate; use its saved rung summary")
    if _report_trial(trial, state, prune_after):
        return _save_reports(state, config, "pruned", weights, output)
    selected = training_rows(rows, protocol)
    sampler = StepPKBatchSampler(selected, config, steps=math.ceil(len(selected) / config.batch_size))
    loader = DataLoader(ClipDataset(selected, dataset, train=True), batch_sampler=sampler, num_workers=0)
    texts, seen = asset["features"].to(device), set(state["seen"])
    for epoch in range(len(state["history"]) + 1, target_epochs + 1):
        require_disk(weights)
        synchronize(device)
        started = time.perf_counter()
        losses, epoch_seen = train_epoch(model, loader, sampler, texts, optimizer, device, config, epoch)
        synchronize(device)
        train_seconds = time.perf_counter() - started
        validation = evaluate_inner(model, rows, protocol, device, dataset, config.eval_batch)
        if not all(math.isfinite(value) for value in validation.values()):
            raise FloatingPointError("Non-finite inner validation")
        seen.update(epoch_seen)
        record = {"epoch": epoch, "train": losses, "validation": validation,
                  "encoder_lr": config.image_lr * image_lr_factor(epoch),
                  "head_lr": config.image_lr * config.head_lr_multiplier * image_lr_factor(epoch),
                  "train_seconds": train_seconds, "steps": epoch * len(loader),
                  "image_presentations": epoch * len(loader) * config.batch_size,
                  "unique_images_seen": len(seen), "train_images": len(selected)}
        current = model.image_state()
        if validation["mAP_at_10"] > state["best"]["validation"]["mAP_at_10"]:
            state["best"], state["best_model"] = {"epoch": epoch, "validation": validation}, current
        record["epoch_seconds"] = time.perf_counter() - started
        state["history"].append(record)
        state["seen"] = sorted(seen)
        _save_checkpoint(last, {**state, "model": current, "optimizer": optimizer.state_dict()})
        del current
        write_json(output / "image_history.json", state["history"])
        elapsed = sum(r["epoch_seconds"] for r in state["history"])
        eta = elapsed / epoch * (target_epochs - epoch)
        print(f"{output.name}: эпоха {epoch}/{target_epochs} (предел 60), осталось {target_epochs-epoch} | "
              f"эпоха {format_duration(record['epoch_seconds'])} | ETA этапа {format_duration(eta)} | "
              f"mAP@10 {validation['mAP_at_10']:.4%}, best {state['best']['validation']['mAP_at_10']:.4%}", flush=True)
        if progress:
            progress(record, _result(state, config, "running"))
        if _report_trial(trial, state, prune_after):
            return _save_reports(state, config, "pruned", weights, output)
    return _save_reports(state, config, "complete", weights, output)


def open_study(results, signature, plan):
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name="clip_reid_variant_02", direction="maximize",
        storage=f"sqlite:///{(results / 'optuna.sqlite3').resolve()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4,
            n_warmup_steps=plan.prune_after_epoch - 1, n_min_trials=3))
    if study.user_attrs.get("signature", signature) != signature:
        raise RuntimeError("Optuna study signature changed")
    if "signature" not in study.user_attrs and study.get_trials():
        raise RuntimeError("Existing Optuna study lacks its signature")
    study.set_user_attr("signature", signature)
    for index, params in enumerate(ANCHORS):
        if not any(t.user_attrs.get("anchor") == index for t in study.get_trials()):
            study.enqueue_trial(params, user_attrs={"anchor": index})
    return study


def next_trial(study):
    running = study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
    if len(running) > 1:
        raise RuntimeError("Multiple running trials in a sequential study; inspect before resuming")
    if running:
        # Optuna 4.x exposes the storage ID only as _trial_id on FrozenTrial.
        # Keep this compatibility access isolated and regression-tested.
        return optuna.trial.Trial(study, running[0]._trial_id)
    # Reseed per trial so restarting Python does not repeat the sampler RNG stream.
    number = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,
                  optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)))
    study.sampler = optuna.samplers.TPESampler(seed=SEED + number, n_startup_trials=4)
    return study.ask()


def run_search(study, plan, run, output):
    while sum(t.state.is_finished() for t in study.get_trials()) < plan.trials:
        trial = next_trial(study)
        config = suggest_config(trial)
        trial.set_user_attr("config", asdict(config))
        try:
            result = run(f"trial_{trial.number:03d}", config, plan.screen_epochs, trial)
        except FloatingPointError as error:
            # An unstable numeric configuration is a failed trial. Resource failures
            # and user interruptions instead leave RUNNING intact for exact resume.
            trial.set_user_attr("failure", str(error))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            write_json(Path(output) / "study_summary.json", study_summary(study))
            print(f"Trial {trial.number} остановлен: {error}", flush=True)
            continue
        trial.set_user_attr("screen_result", result)
        state = optuna.trial.TrialState.PRUNED if result["status"] == "pruned" else optuna.trial.TrialState.COMPLETE
        study.tell(trial, result["best"]["validation"]["mAP_at_10"] if state.name == "COMPLETE" else None, state=state)
        write_json(Path(output) / "study_summary.json", study_summary(study))
    report = study_summary(study)
    write_json(Path(output) / "study_summary.json", report)
    return report


def study_summary(study):
    return {"trials": [{"number": t.number, "state": t.state.name, "value": t.value,
                        "params": t.params, "config": t.user_attrs.get("config"),
                        "screen_result": t.user_attrs.get("screen_result"),
                        "failure": t.user_attrs.get("failure")} for t in study.get_trials()]}


def rank_results(results):
    return sorted(results, key=lambda r: (-r["best"]["validation"]["mAP_at_10"], r["name"]))


def continue_rung(candidates, count, target, run, output, stage):
    selected = rank_results(candidates)[:count]
    if len(selected) != count:
        raise RuntimeError(f"Need {count} complete candidates for {stage}")
    decision = {"target_epochs": target, "names": [r["name"] for r in selected]}
    checked_json(Path(output) / f"{stage}_selection.json", decision)
    results = []
    for candidate in selected:
        path = Path(output) / "rungs" / stage / f"{candidate['name']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["config"] != candidate["config"] or result["completed_epochs"] != target:
                raise RuntimeError("Saved rung does not match its selected candidate")
        else:
            result = {**run(candidate["name"], TrialConfig(**candidate["config"]), target, None),
                      "name": candidate["name"]}
            write_json(path, result)
        results.append(result)
    ranked = rank_results(results)
    write_json(Path(output) / f"{stage}_summary.json", {"candidates": ranked})
    return ranked


def confirmation_decision(winner, references, plan):
    gain = winner["best"]["validation"]["mAP_at_10"] - references["clip_variant_1"]
    return {"run": gain >= plan.confirmation_min_gain, "gain_over_variant_1": gain,
            "minimum_gain": plan.confirmation_min_gain,
            "reason": "Budget gate on inner validation, not statistical significance"}


def progress_estimate(output, plan):
    """Measured epoch-time estimate, with separate optional confirmation budget."""
    histories = {path.parent.name: json.loads(path.read_text())
                 for path in Path(output).glob("runs/*/image_history.json")}
    records = [record for history in histories.values() for record in history]
    average = np.mean([r["epoch_seconds"] for r in records]) if records else 251.
    search_done = sum(len(history) for name, history in histories.items() if "_seed_" not in name)
    confirmation_done = sum(len(history) for name, history in histories.items() if "_seed_" in name)
    skipped = 0
    study_path = Path(output) / "study_summary.json"
    if study_path.exists():
        study = json.loads(study_path.read_text())
        skipped = sum(plan.screen_epochs - t["screen_result"]["completed_epochs"]
                      for t in study["trials"] if t["state"] == "PRUNED")
    budget = (plan.trials * plan.screen_epochs + plan.top_k * (plan.promotion_epochs-plan.screen_epochs)
              + plan.finalists * (plan.final_epochs-plan.promotion_epochs))
    search_remaining = max(0, budget - search_done - skipped)
    return {"mean_epoch_seconds": float(average), "search_epochs_remaining_upper": search_remaining,
            "eta_search_seconds_upper": float(search_remaining * average),
            "optional_confirmation_seconds_upper": float(max(0, len(plan.extra_seeds)*60-confirmation_done)*average
                                                          + len(plan.extra_seeds)*20*60)}


def write_report(output, report):
    write_json(Path(output) / "experiment_summary.json", report)
    winner, refs = report["winner"], report["references"]
    scores = report["seed_scores"]
    lines = ["# CLIP-ReID — HPO results", "", "Только raw inner mAP@10. Outer/test не использованы.", "",
             f"- CLIP variant 1: {refs['clip_variant_1']:.4%}.",
             f"- Чистый OSNet-контроль: {refs['clean_osnet']:.4%}.",
             f"- Победитель поиска: {winner['name']}, {winner['best']['validation']['mAP_at_10']:.4%}, "
             f"эпоха {winner['best']['epoch']}.",
             f"- Seed mAP@10: {', '.join(f'{x:.4%}' for x in scores)}.",
             f"- Среднее: {np.mean(scores):.4%}; std (population): {np.std(scores):.4%}.",
             f"- Дополнительные seed выполнены: {len(scores)-1}.", "",
             "Checkpoint каждого seed выбран по той же validation; это не независимый тест.",
             "Повторы seed не проверяют устойчивость к выбору разбиения.",
             "Порог отказа, реранкинг, номерные маски и MVP не изменены.",
             "Следующий шаг — ручная разметка и парный аудит анонимизации, а не автоматическая замена MVP."]
    (Path(output) / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(rows, split, device, variant=VARIANT, plan=SearchPlan(), source=SOURCE,
                   previous=PREVIOUS, dataset=DATASET, allow_cpu=False):
    """Notebook Run All entry point. Interrupt -> rerun the same cells/config."""
    plan.validate()
    variant, previous, source = Path(variant).resolve(), Path(previous).resolve(), Path(source).resolve()
    if variant == previous or variant.is_relative_to(previous) or previous.is_relative_to(variant):
        raise ValueError("HPO must have its own directory, separate from variant 1")
    if torch.device(device).type == "cpu" and not allow_cpu:
        raise RuntimeError("Full CLIP HPO on CPU is disabled; select MPS/CUDA")
    if not source.exists() or sha256(source) != CHECKPOINT_SHA256:
        raise ValueError("Missing or incompatible source checkpoint; no random-weight fallback")
    with experiment_lock(variant):
        weights, output = variant / "weights", variant / "results"
        weights.mkdir(exist_ok=True)
        output.mkdir(exist_ok=True)
        # Conservative space estimate includes last+best for 12 trials, 2 extra seeds,
        # selected rung snapshots and one atomic replacement. No old artifacts deleted.
        fresh = not (output / "manifest.json").exists()
        free = require_disk(variant, max(26, 1.7 * (plan.trials + len(plan.extra_seeds)) + 3) if fresh else 3)
        print(f"Свободно {free:.1f} GiB. Аудит исходных данных и разделения…", flush=True)
        protocol = prepare_protocol(rows, split, output, dataset)
        specification = {**manifest(protocol, plan), "device": str(device)}
        checked_json(output / "manifest.json", specification)
        signature = digest(specification)
        references = load_references(protocol, previous)
        checked_json(output / "references.json", references)
        finished = output / "experiment_summary.json"
        if finished.exists():
            report = json.loads(finished.read_text())
            if report["signature"] != signature:
                raise RuntimeError("Completed report belongs to a different experiment")
            print("Эксперимент уже завершён: возвращаю сохранённый отчёт, обучение не повторяется.")
            return report
        assets = {}

        def run(name, config, target, trial):
            if config.seed not in assets:
                assets[config.seed] = prompt_assets(rows, protocol, device, source, weights, output,
                                                    config.seed, previous, dataset)
            print(f"\n{name}: обучение/продолжение до {target} эпох, seed={config.seed}", flush=True)
            set_seed(config.seed)
            model = load_pretrained(source, classes=len(protocol["inner"]["train"]), device=device)
            folder, result_dir = weights / name, output / "runs" / name
            # Only promoted candidates keep extra rung best snapshots (six small files).
            if target in (plan.promotion_epochs, plan.final_epochs) and (result_dir / "summary.json").exists():
                prior = json.loads((result_dir / "summary.json").read_text())
                epoch = prior["completed_epochs"]
                snapshot = folder / f"best_after_{epoch:03d}.pt"
                if epoch < target and not snapshot.exists():
                    best = torch.load(folder / "image_best.pt", map_location="cpu", weights_only=True)
                    _save_checkpoint(snapshot, best)
                    del best

            def progress(record, result):
                estimate = progress_estimate(output, plan)
                write_json(output / "progress.json", {"status": "running", "candidate": name,
                    "target_epochs": target, "epoch": record["epoch"], "remaining_epochs": target-record["epoch"],
                    "last_epoch_seconds": record["epoch_seconds"],
                    "eta_current_run_seconds": result["seconds"] / record["epoch"] * (target-record["epoch"]),
                    "best": result["best"], "time_unix": time.time(), **estimate})
                print(f"ETA оставшегося поиска ≤ {format_duration(estimate['eta_search_seconds_upper'])}; "
                      f"дополнительные seed, если пройдёт порог: ещё ≈ "
                      f"{format_duration(estimate['optional_confirmation_seconds_upper'])}", flush=True)
            try:
                return fit_candidate(model, rows, protocol, assets[config.seed], device, config,
                    target, folder, result_dir, signature, trial, plan.prune_after_epoch, dataset, progress)
            finally:
                del model
                release_device(device)

        try:
            study = open_study(output, signature, plan)
            screen = run_search(study, plan, run, output)
            candidates = [{**t["screen_result"], "name": f"trial_{t['number']:03d}"}
                          for t in screen["trials"] if t["state"] == "COMPLETE"]
            promoted = continue_rung(candidates, plan.top_k, plan.promotion_epochs, run, output, "stage2_top4")
            finalists = continue_rung(promoted, plan.finalists, plan.final_epochs, run, output, "stage3_top2")
            winner = finalists[0]
            decision = confirmation_decision(winner, references, plan)
            checked_json(output / "confirmation_decision.json", {"winner": winner["name"], **decision})
            seeds = [winner]
            if decision["run"]:
                for seed in plan.extra_seeds:
                    config = replace(TrialConfig(**winner["config"]), seed=seed)
                    name = f"{winner['name']}_seed_{seed}"
                    result = {**run(name, config, plan.final_epochs, None), "name": name}
                    seeds.append(result)
            report = {"signature": signature, "references": references, "winner": winner,
                      "confirmation": decision, "seeds": seeds,
                      "seed_scores": [s["best"]["validation"]["mAP_at_10"] for s in seeds],
                      "outer_test_used": False, "mvp_changed": False}
            write_report(output, report)
            write_json(output / "progress.json", {"status": "complete", "time_unix": time.time(),
                       "winner": winner["name"], "confirmation": decision})
            return report
        except BaseException as error:
            write_json(output / "interruption.json", {"type": type(error).__name__, "message": str(error),
                       "time_unix": time.time(), "action": "Fix cause, restart kernel and Run All; do not delete results"})
            raise
