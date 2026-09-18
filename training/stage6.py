"""Stage 6: frame-disjoint, step-budgeted ResNet experiments; no MVP mutations."""
import gc
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from backend.core import DATASET, sha256
from backend.evaluate import SEED, write_json as write_json_file
from training.hpo import (CameraAwarePKBatchSampler, ExperimentConfig,
                          evaluate_experiment, experiment_losses, make_optimizer)
from training.pipeline import VehicleDataset, format_duration, set_seed
from training.stage4 import ACTIVE_REFERENCE, tuned_retrieval
from training.stage5 import initialize_resnet_experiment, stage5_config


PRETRAINED_SHA256 = "d9d0bb7b2ba34d7e6e12c4616c9d233914762e49fbb9fcb49d7f4d65da6f759c"


def write_json(path, value):
    """Never expose a partially written protocol or cached outer evaluation."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json_file(temporary, value)
    temporary.replace(path)


@dataclass(frozen=True)
class StepBudget:
    max_steps: int = 4000
    evaluation_interval: int = 200
    warmup_steps: int = 400
    min_steps: int = 2000
    patience: int = 5
    promotion_min_map: float = .50

    def validate(self):
        if not 0 < self.evaluation_interval <= self.min_steps <= self.max_steps:
            raise ValueError("Require 0 < evaluation_interval <= min_steps <= max_steps")
        if not 0 <= self.warmup_steps < self.max_steps or self.patience < 1:
            raise ValueError("Invalid warmup or patience")
        if not 0 <= self.promotion_min_map <= 1:
            raise ValueError("promotion_min_map must be in [0, 1]")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frame_disjoint_inner_split(rows, train_ids, frame_hashes, seed=SEED):
    """Keep connected identities sharing any exact full frame together."""
    parent = {identity: identity for identity in sorted(train_ids)}

    def find(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    frames = {}
    for row in rows:
        identity = row["vehicle_id"]
        if identity not in parent:
            continue
        digest = frame_hashes[row["image_id"]]
        if digest in frames:
            parent[find(identity)] = find(frames[digest])
        frames[digest] = identity
    groups = defaultdict(list)
    for identity in parent:
        groups[find(identity)].append(identity)
    groups = list(groups.values())
    if len(groups) < 2:
        raise ValueError("Need at least two independent frame/identity groups")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * .2)))
    return {
        "train": sorted(i for group in groups[count:] for i in group),
        "validation": sorted(i for group in groups[:count] for i in group),
    }


def audit_partitions(rows, frame_hashes, partitions):
    owners, frames = {}, {}
    for name, identities in partitions.items():
        for identity in identities:
            if identity in owners:
                raise ValueError(f"Identity leakage: {identity}")
            owners[identity] = name
    if set(owners) != {row["vehicle_id"] for row in rows}:
        raise ValueError("Split does not cover exactly the supplied identities")
    for row in rows:
        owner = owners[row["vehicle_id"]]
        digest = frame_hashes[row["image_id"]]
        if digest in frames and frames[digest] != owner:
            raise ValueError(f"Exact-frame leakage: {row['image_id']}")
        frames[digest] = owner


def prepare_protocol(rows, split, results_dir, dataset=DATASET):
    """Rehash actual images, audit both splits and reject stale experiment reuse."""
    hashes = {row["image_id"]: sha256(Path(dataset) / "images" / f"{row['image_id']}.jpg")
              for row in tqdm(rows, desc="Audit train frame hashes")}
    audit_partitions(rows, hashes, split["identities"])
    inner = frame_disjoint_inner_split(rows, split["identities"]["train"], hashes)
    outer_train = set(split["identities"]["train"])
    selected = [row for row in rows if row["vehicle_id"] in outer_train]
    audit_partitions(selected, hashes, inner)
    protocol = {
        "version": 1, "seed": SEED,
        "data_sha256": _digest({"rows": rows, "frames": hashes}),
        "outer": split["identities"], "inner": inner,
        "identity_and_exact_frame_disjoint": True,
        "initializer_sha256": PRETRAINED_SHA256,
    }
    path = Path(results_dir) / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Protocol/data changed; use a new experiment directory")
    write_json(path, protocol)
    return protocol


class StepPKBatchSampler(CameraAwarePKBatchSampler):
    """An explicit number of P×K draws, not one short pass over identities."""

    def __init__(self, rows, config, steps):
        super().__init__(rows, config.identities_per_batch, config.images_per_identity,
                         config.prefer_cross_camera, config.seed)
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.steps):
            yield [index for identity in rng.sample(self.identities, self.p)
                   for index in self._sample_identity(identity, rng)]


def set_step_learning_rates(optimizer, config, budget, step):
    """The horizon never depends on early stopping or the final run's stop step."""
    if step < budget.warmup_steps:
        factor = (step + 1) / budget.warmup_steps
    else:
        progress = min(1., (step - budget.warmup_steps) /
                       max(1, budget.max_steps - budget.warmup_steps - 1))
        factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * .5 * (
            1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def train_step_block(model, loader, sampler, optimizer, device, config, budget, start):
    """Seeds are reset per block, so a restart replays only an unfinished block."""
    block = start // budget.evaluation_interval
    set_seed(config.seed + block)
    sampler.set_epoch(block)
    totals, seen = defaultdict(float), set()
    first_lr = last_lr = None
    for offset, (clean, robust, labels, image_ids) in enumerate(
            tqdm(loader, desc=f"steps {start + 1}–{start + len(loader)}", leave=False)):
        last_lr = set_step_learning_rates(optimizer, config, budget, start + offset)
        if first_lr is None:
            first_lr = last_lr
        optimizer.zero_grad(set_to_none=True)
        losses = experiment_losses(
            model, clean.to(device), robust.to(device), labels.to(device), config)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Non-finite loss at step {start + offset + 1}")
        losses["loss"].backward()
        optimizer.step()
        for name, value in losses.items():
            totals[name] += value.item()
        seen.update(image_ids)
    return ({name: value / len(loader) for name, value in totals.items()},
            seen, {"first": first_lr, "last": last_lr})


def _save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def should_stop(history, budget):
    if not history or history[-1]["step"] < budget.min_steps:
        return False
    best_index = max(range(len(history)), key=lambda i: history[i]["validation"]["mAP_at_10"])
    return len(history) - best_index - 1 >= budget.patience


def fit_steps(model, rows, train_ids, validation_ids, device, config, budget,
              weights_dir, results_dir, protocol, stop_steps=None, dataset=DATASET):
    """Atomic boundary resume; inner early stopping OR fixed final steps, never both."""
    config.validate()
    budget.validate()
    if config.num_workers != 0 or config.loss_weight_schedule != "constant":
        raise ValueError("Stage 6 uses num_workers=0 and a constant loss recipe")
    if (validation_ids is None) != (stop_steps is not None):
        raise ValueError("Use inner validation or a fixed final stop_steps")
    target = budget.max_steps if stop_steps is None else stop_steps
    if not 1 <= target <= budget.max_steps:
        raise ValueError("stop_steps must fit inside the unchanged schedule horizon")
    weights_dir, results_dir = Path(weights_dir), Path(results_dir)
    signature = {"protocol": _digest(protocol), "config": asdict(config),
                 "budget": asdict(budget), "train_ids": sorted(train_ids),
                 "validation_ids": validation_ids, "stop_steps": stop_steps}
    labels = {identity: label for label, identity in enumerate(sorted(train_ids))}
    selected = [{**row, "label": labels[row["vehicle_id"]]}
                for row in rows if row["vehicle_id"] in labels]
    sampler = StepPKBatchSampler(selected, config, budget.evaluation_interval)
    loader = DataLoader(VehicleDataset(selected, dataset, augment=True,
                                        resize_mode=config.resize_mode),
                        batch_sampler=sampler, num_workers=0)
    optimizer = make_optimizer(model, config)
    last_path = weights_dir / "last.pt"
    if not last_path.exists() and any(path.exists() for path in (
            results_dir / "history.json", weights_dir / "best_map.pt", weights_dir / "final.pt")):
        raise RuntimeError("Missing last.pt; restore it or use a new experiment directory")
    history, seen, best_model, initial_validation = [], set(), None, None
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise RuntimeError("Resume config/protocol changed; use a new experiment directory")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, seen = checkpoint["history"], set(checkpoint["seen"])
        best_model, initial_validation = checkpoint["best_model"], checkpoint["initial_validation"]
        del checkpoint
    elif validation_ids is not None:
        initial_validation, _ = evaluate_experiment(
            model, rows, validation_ids, device, dataset=dataset, seed=SEED)

    step = history[-1]["step"] if history else 0
    while step < target and not (validation_ids is not None and should_stop(history, budget)):
        started = time.perf_counter()
        sampler.steps = min(budget.evaluation_interval, target - step)
        train, block_seen, lr = train_step_block(
            model, loader, sampler, optimizer, device, config, budget, step)
        train_seconds = time.perf_counter() - started
        step += sampler.steps
        seen.update(block_seen)
        record = {"epoch": len(history) + 1, "step": step, "lr": lr, "train": train,
                  "image_presentations": step * config.batch_size,
                  "equivalent_passes": step * config.batch_size / len(selected),
                  "unique_images_seen": len(seen), "train_images": len(selected)}
        if validation_ids is not None:
            validation, _ = evaluate_experiment(
                model, rows, validation_ids, device, dataset=dataset, seed=SEED)
            if not math.isfinite(validation["mAP_at_10"]):
                raise FloatingPointError("Non-finite inner mAP@10")
            previous = max((r["validation"]["mAP_at_10"] for r in history), default=-1.)
            if validation["mAP_at_10"] > previous:
                best_model = _cpu_state(model)
            record["validation"] = validation
        record["timing"] = {"train_seconds": train_seconds,
                            "epoch_seconds": time.perf_counter() - started}
        history.append(record)
        elapsed = sum(r["timing"]["epoch_seconds"] for r in history)
        record["timing"].update(elapsed_seconds=elapsed,
                                estimated_remaining_seconds=elapsed / step * (target - step))
        # History and best weights live in the same authoritative atomic checkpoint.
        _save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "signature": signature, "history": history, "seen": sorted(seen),
            "best_model": best_model, "initial_validation": initial_validation,
        })
        if step % 2000 == 0:
            _save_checkpoint(weights_dir / f"step_{step:06d}.pt", {
                "model": _cpu_state(model), "config": asdict(config), "step": step,
                "epoch": len(history), "signature": signature})
        write_json(results_dir / "history.json", history)
        score = f" | inner mAP@10: {record['validation']['mAP_at_10']:.4f}" if validation_ids else ""
        print(f"Эпоха {len(history)}/{math.ceil(target / budget.evaluation_interval)} | "
              f"осталось эпох: {math.ceil((target - step) / budget.evaluation_interval)} | "
              f"шаги {step}/{target}, осталось {target - step} | "
              f"эпоха: {format_duration(record['timing']['epoch_seconds'])} | "
              f"прошло: {format_duration(elapsed)} | "
              f"ETA ≤ {format_duration(record['timing']['estimated_remaining_seconds'])} | "
              f"предъявлений: {record['image_presentations']}, "
              f"условных проходов: {record['equivalent_passes']:.2f}" + score, flush=True)

    best = max(history, key=lambda r: r["validation"]["mAP_at_10"]) if validation_ids else history[-1]
    payload = {"model": best_model if validation_ids else _cpu_state(model),
               "config": asdict(config), "budget": asdict(budget), "step": best["step"],
               "epoch": best["epoch"], "signature": signature}
    selected_path = weights_dir / ("best_map.pt" if validation_ids else "final.pt")
    existing = torch.load(selected_path, map_location="cpu", weights_only=False) if selected_path.exists() else None
    if existing is None or existing["signature"] != signature or existing["step"] != best["step"]:
        _save_checkpoint(selected_path, payload)
    elif any(not torch.equal(existing["model"][key], value) for key, value in payload["model"].items()):
        raise RuntimeError("Selected weights differ from the authoritative last.pt")
    del existing
    write_json(results_dir / "history.json", history)  # Repair a stale/missing JSON after interruption.
    summary = {"config": asdict(config), "budget": asdict(budget),
               "completed_steps": step, "selected_steps": best["step"],
               "best_mAP_at_10": best.get("validation", {}).get("mAP_at_10"),
               "initial_validation": initial_validation,
               "stopped_early": step < target, "train_images": len(selected),
               "image_presentations": step * config.batch_size,
               "unique_images_seen": len(seen), "protocol_sha256": _digest(protocol)}
    write_json(results_dir / "training_summary.json", summary)
    return summary


def _run(name, rows, train_ids, validation_ids, config, budget, protocol, device,
         results_dir, weights_dir, dataset=DATASET, stop_steps=None, split=None):
    print(f"\nЗапуск: {name} | LR={config.encoder_lr:g} | "
          f"{config.pooling}/{config.metric_loss} | seed={config.seed}", flush=True)
    set_seed(config.seed)
    model, initializer = initialize_resnet_experiment(len(train_ids), config, device)
    if initializer["sha256"] != PRETRAINED_SHA256:
        raise ValueError("Unexpected ImageNet initializer checksum")
    try:
        summary = fit_steps(model, rows, train_ids, validation_ids, device, config, budget,
                            Path(weights_dir) / name, Path(results_dir) / name, protocol,
                            stop_steps=stop_steps, dataset=dataset)
        summary.update(name=name, initializer=initializer)
        if split is not None:
            evaluation_path = Path(results_dir) / name / "outer_evaluation.json"
            checkpoint_path = Path(weights_dir) / name / "final.pt"
            key = {"checkpoint_sha256": sha256(checkpoint_path),
                   "protocol_sha256": _digest(protocol)}
            if evaluation_path.exists():
                saved = json.loads(evaluation_path.read_text())
                if saved["key"] != key:
                    raise RuntimeError("Outer evaluation does not match frozen checkpoint")
                summary["tuned_reranking"] = saved["tuned_reranking"]
            else:
                summary["tuned_reranking"] = tuned_retrieval(model, rows, split, device, dataset)
                write_json(evaluation_path, {"key": key, "tuned_reranking": summary["tuned_reranking"]})
        write_json(Path(results_dir) / name / "summary.json", summary)
        return summary
    finally:
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()


def run_controlled_screening(rows, protocol, device, base_config_path, results_dir,
                             weights_dir, budget=StepBudget(), dataset=DATASET):
    """Four fits: two LRs, then pooling, then metric loss; all from ImageNet."""
    budget.validate()
    runs = []

    def run(name, config):
        result = _run(f"screening/{name}", rows, protocol["inner"]["train"],
                      protocol["inner"]["validation"], config, budget, protocol,
                      device, results_dir, weights_dir, dataset)
        runs.append(result)
        return result

    configs = [replace(stage5_config(base_config_path, math.ceil(
        budget.max_steps / budget.evaluation_interval), SEED, lr), num_workers=0)
        for lr in (1e-4, 3e-4)]
    lr_runs = [run(f"lr_{config.encoder_lr:.0e}_gem_supcon", config) for config in configs]
    key = lambda result: result["best_mAP_at_10"]
    lr_winner = max(lr_runs, key=key)
    pooling = run("avg_pool", replace(ExperimentConfig(**lr_winner["config"]), pooling="avg"))
    pool_winner = max([lr_winner, pooling], key=key)
    triplet = run("triplet", replace(ExperimentConfig(**pool_winner["config"]), metric_loss="triplet"))
    winner = max([pool_winner, triplet], key=key)
    report = {"protocol_sha256": _digest(protocol), "budget": asdict(budget),
              "selection_metric": "inner raw mAP@10; fixed query/gallery seed",
              "outer_calibration_or_validation_used": False,
              "promoted": winner["best_mAP_at_10"] >= budget.promotion_min_map,
              "promotion_rule": "inner raw mAP@10 >= predeclared promotion_min_map",
              "lr_winner": lr_winner["name"], "pooling_winner": pool_winner["name"],
              "winner": winner, "runs": runs}
    write_json(Path(results_dir) / "screening_summary.json", report)
    return report


def run_controlled_seeds(rows, split, protocol, screening, device, results_dir,
                         weights_dir, seeds=(SEED, SEED + 1, SEED + 2), dataset=DATASET):
    if screening["protocol_sha256"] != _digest(protocol):
        raise ValueError("Screening protocol mismatch")
    if split["identities"] != protocol["outer"]:
        raise ValueError("Outer split changed after screening")
    budget = StepBudget(**screening["budget"])
    winner = screening["winner"]
    if not screening["promoted"]:
        report = {"status": "not_promoted", "inner_mAP_at_10": winner["best_mAP_at_10"],
                  "promotion_min_map": budget.promotion_min_map,
                  "reason": "Inner-only budget gate; no outer evaluation or final training",
                  "beats_active_reference": False}
    else:
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("Final comparison requires exactly three distinct seeds")
        template = ExperimentConfig(**winner["config"])
        runs = [_run(f"final_seeds/seed_{seed}", rows, split["identities"]["train"],
                     None, replace(template, seed=seed), budget, protocol, device,
                     results_dir, weights_dir, dataset,
                     stop_steps=winner["selected_steps"], split=split) for seed in seeds]
        metrics = ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "candidate_score")
        aggregate = {}
        for metric in metrics:
            values = [r["tuned_reranking"]["validation"][metric] for r in runs]
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values))
        quality = [r["tuned_reranking"]["validation_quality_score"] for r in runs]
        aggregate.update(mean_quality_score=float(np.mean(quality)), std_quality_score=float(np.std(quality)))
        report = {
            "status": "completed", "architecture": f"ResNet50-IBN-a + {template.pooling} + BNNeck",
            "protocol_sha256": _digest(protocol), "budget": asdict(budget),
            "fixed_training_steps": winner["selected_steps"], "seeds": list(seeds),
            "active_reference": ACTIVE_REFERENCE, "aggregate": aggregate, "runs": runs,
            # Export a predeclared seed, never choose a seed using outer validation.
            "selected_representative": {"name": f"seed_{seeds[0]}", "seed": seeds[0]},
            "representative_rule": "first predeclared seed, independent of validation",
            "beats_active_reference": (
                aggregate["mean_mAP_at_10"] > ACTIVE_REFERENCE["mAP_at_10"] and
                aggregate["mean_quality_score"] > ACTIVE_REFERENCE["quality_score"] and
                aggregate["mean_candidate_score"] >= ACTIVE_REFERENCE["candidate_score"] - .005),
        }
        # Keep compatibility with the existing audited ONNX exporter.
        report["runs"] = [{**run, "name": run["name"].split("/")[-1]} for run in runs]
    write_json(Path(results_dir) / "final_comparison.json", report)
    return report


def write_results_markdown(screening, comparison, results_dir):
    winner = screening["winner"]
    lines = ["# Результаты варианта 6", "",
             "Данные взяты из сохранённых результатов; активная модель MVP не изменена.", "",
             "| Inner-запуск | mAP@10 | Лучший шаг | Выполнено шагов |",
             "|---|---:|---:|---:|"]
    for run in screening["runs"]:
        lines.append(f"| {run['name']} | {run['best_mAP_at_10']:.4f} | "
                     f"{run['selected_steps']} | {run['completed_steps']} |")
    lines += ["", f"Выбрано только по inner: `{winner['name']}`.",
              f"Горизонт LR: {screening['budget']['max_steps']}; "
              f"финальная остановка: {winner['selected_steps']} обновлений.", ""]
    if comparison["status"] == "not_promoted":
        lines += [f"Порог продвижения {comparison['promotion_min_map']:.2f} не достигнут.",
                  "Три финальных seed, outer evaluation и ONNX export не запускались."]
    else:
        lines += ["| Outer validation, 3 seed | Среднее | Std |", "|---|---:|---:|"]
        for metric in ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "candidate_score"):
            values = comparison["aggregate"]
            lines.append(f"| {metric} | {values['mean_' + metric]:.4f} | {values['std_' + metric]:.4f} |")
        lines += ["", f"Превзойдён активный ориентир: `{comparison['beats_active_reference']}`.",
                  "Это локальная development-оценка, не результат закрытого теста организаторов."]
    path = Path(results_dir) / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
