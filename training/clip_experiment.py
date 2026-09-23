"""One local, two-stage CLIP-ReID transfer experiment on a clean inner split."""
import json
import hashlib
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from backend.core import DATASET, ROOT, STOCK_MODEL, Encoder, encode_rows as encode_osnet, sha256
from backend.evaluate import SEED, make_protocol
from backend.scoring import ranked_queries
from training.audit import data_signature, digest
from training.clip_reid import (PREPROCESS, ClipDataset, encode_rows, image_losses,
                                prompt_loss, release_device, synchronize)
from training.clip_source import CHECKPOINT_SHA256, SOURCE_COMMIT
from training.hpo import ExperimentConfig, initialize_experiment
from training.pipeline import format_duration, set_seed
from training.stage6 import (StepBudget, StepPKBatchSampler, _save_checkpoint, audit_partitions,
                             fit_steps, frame_disjoint_inner_split, write_json)


@dataclass(frozen=True)
class ClipConfig:
    seed: int = SEED
    prompt_epochs: int = 60
    image_epochs: int = 60
    prompt_batch: int = 64
    identities_per_batch: int = 8
    images_per_identity: int = 4
    prefer_cross_camera: bool = True
    prompt_lr: float = 3.5e-4
    image_lr: float = 5e-6
    weight_decay: float = 1e-4
    eval_batch: int = 16

    @property
    def batch_size(self):
        return self.identities_per_batch * self.images_per_identity

    def validate(self):
        if min(self.prompt_epochs, self.image_epochs, self.prompt_batch, self.eval_batch) < 1:
            raise ValueError("Epochs and batch sizes must be positive")
        if self.identities_per_batch < 2 or self.images_per_identity < 2:
            raise ValueError("Require P>=2, K>=2")
        if min(self.prompt_lr, self.image_lr) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning rate or weight decay")


def prepare_protocol(rows, split, output, dataset=DATASET):
    data = data_signature(rows, dataset)
    audit_partitions(rows, data["frames"], split["identities"])
    inner = frame_disjoint_inner_split(rows, split["identities"]["train"], data["frames"], SEED)
    development_rows = [r for r in rows if r["vehicle_id"] in set(split["identities"]["train"])]
    audit_partitions(development_rows, data["frames"], inner)
    protocol = {"version": 1, "seed": SEED, "data_sha256": digest(data),
                "outer": split["identities"], "inner": inner,
                "identity_and_exact_frame_disjoint": True,
                "initializer_sha256": CHECKPOINT_SHA256, "source_commit": SOURCE_COMMIT,
                "evaluator_sha256": sha256(ROOT / "evaluate.py"), "preprocess": PREPROCESS}
    path = Path(output) / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Protocol/data changed; use a new variant directory")
    write_json(path, protocol)
    return protocol


def run_signature(protocol, config):
    config.validate()
    sources = ["training/clip_reid.py", "training/clip_experiment.py",
               "training/vendor/clip_reid/model.py", "backend/scoring.py"]
    return {"protocol": digest(protocol), "config": asdict(config),
            "code": {path: sha256(ROOT / path) for path in sources}, "torch": str(torch.__version__)}


def training_rows(rows, protocol):
    labels = {identity: i for i, identity in enumerate(protocol["inner"]["train"])}
    return [{**row, "label": labels[row["vehicle_id"]]} for row in rows if row["vehicle_id"] in labels]


def _ranking_metrics(query, gallery, vectors):
    result = ranked_queries(query, gallery, vectors)
    return {"mAP_at_10": result.ranking["mAP@10"], "Rank_1": result.ranking["Rank-1"],
            "Rank_5": result.ranking["Rank-5"], "full_mAP": result.full_ranking["mAP_full"],
            "mINP": result.full_ranking["mINP"], "known_queries": result.ranking["n_scored"],
            "unknown_queries": result.ranking["n_openset_excluded"]}


def evaluate_inner(model, rows, protocol, device, dataset=DATASET, batch_size=16):
    query, gallery = make_protocol(rows, protocol["inner"]["validation"], SEED)
    vectors = encode_rows(model, query + gallery, device, dataset, batch_size)
    return _ranking_metrics(query, gallery, vectors)


def public_baselines(model, rows, protocol, device, output, dataset=DATASET, batch_size=16):
    """Active fine-tuned OSNet is deliberately excluded: it saw inner validation IDs."""
    path = Path(output) / "baselines.json"
    signature = {"protocol": digest(protocol), "clip_sha256": CHECKPOINT_SHA256,
                 "osnet_sha256": sha256(STOCK_MODEL), "preprocess": PREPROCESS,
                 "adapter_sha256": sha256(ROOT / "training/clip_reid.py")}
    if path.exists():
        report = json.loads(path.read_text())
        if report["signature"] != signature:
            raise ValueError("Baseline source/data changed; use a new results directory")
        return report
    query, gallery = make_protocol(rows, protocol["inner"]["validation"], SEED)
    clip_metrics = evaluate_inner(model, rows, protocol, device, dataset, batch_size)
    osnet_vectors = encode_osnet(Encoder(STOCK_MODEL), query + gallery, dataset)
    report = {"signature": signature, "clip_vehicle_pretrained": clip_metrics,
              "osnet_stock": _ranking_metrics(query, gallery, dict(zip(
                  [r["image_id"] for r in query + gallery], osnet_vectors))),
              "active_osnet_excluded": "Already trained on all 925 development IDs; inner leakage",
              "no_refusal_threshold_or_outer_tuning": True}
    write_json(path, report)
    return report


def _resume(path, signature):
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["signature"] != signature:
        raise RuntimeError("Resume configuration/code/protocol changed; use a new variant directory")
    return checkpoint


def prompt_learning_rate(config, epoch):
    warmup = min(5, config.prompt_epochs)
    if epoch <= warmup:
        return 1e-5 + (config.prompt_lr - 1e-5) * epoch / warmup
    progress = (epoch - warmup) / max(1, config.prompt_epochs - warmup)
    return 1e-6 + .5 * (config.prompt_lr - 1e-6) * (1 + math.cos(math.pi * progress))


def image_lr_factor(epoch):
    # Fixed 60-epoch upstream horizon: 10-epoch warmup, drops at 30 and 50.
    warmup = .1 + .9 * min(epoch / 10, 1)
    return warmup * (.1 ** sum(epoch >= boundary for boundary in (30, 50)))


def _log_epoch(stage, record, history, target):
    elapsed = sum(r["epoch_seconds"] for r in history)
    eta = elapsed / len(history) * (target - record["epoch"])
    print(f"{stage} | эпоха {record['epoch']}/{target}, осталось {target-record['epoch']} | "
          f"эпоха: {format_duration(record['epoch_seconds'])} | "
          f"прошло: {format_duration(elapsed)} | ETA этапа: {format_duration(eta)}", flush=True)


def run_prompt_stage(model, selected, device, config, protocol, weights, output, dataset=DATASET):
    signature = run_signature(protocol, config)
    weights, output = Path(weights), Path(output)
    cache = output / "cache/image_features.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    ids = [r["image_id"] for r in selected]
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["signature"]) != digest(signature) or saved["ids"].tolist() != ids:
                raise RuntimeError("Stale prompt feature cache; use a new variant")
            features = torch.from_numpy(saved["features"].copy())
    else:
        model.eval()
        vectors = encode_rows(model, selected, device, dataset, config.eval_batch, projected=True)
        features = torch.from_numpy(np.stack([vectors[i] for i in ids]))
        temporary = cache.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, ids=np.array(ids), signature=digest(signature), features=features.numpy())
        temporary.replace(cache)
    labels = torch.tensor([r["label"] for r in selected], dtype=torch.long)
    model.set_stage(1)
    optimizer = torch.optim.Adam([model.prompt_learner.cls_ctx], lr=config.prompt_lr,
                                 weight_decay=config.weight_decay)
    checkpoint_path = weights / "prompt_last.pt"
    checkpoint = _resume(checkpoint_path, signature)
    history = []
    if checkpoint:
        model.prompt_learner.load_state_dict(checkpoint["prompt"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        del checkpoint
    elif (output / "prompt_history.json").exists():
        raise RuntimeError("Missing prompt_last.pt; restore it or use a new variant")
    for epoch in range(len(history) + 1, config.prompt_epochs + 1):
        set_seed(config.seed + epoch)
        synchronize(device)
        started, total, count = time.perf_counter(), 0., 0
        lr = prompt_learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = torch.randperm(len(selected))
        for batch in tqdm(order.split(config.prompt_batch), desc=f"Prompt {epoch}/{config.prompt_epochs}", leave=False):
            target, image = labels[batch].to(device), features[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = prompt_loss(image, model.text(target), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite prompt loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        synchronize(device)
        record = {"epoch": epoch, "loss": total / count, "lr": lr,
                  "epoch_seconds": time.perf_counter() - started}
        history.append(record)
        _save_checkpoint(checkpoint_path, {"signature": signature, "history": history,
                         "prompt": {k: v.detach().cpu() for k, v in model.prompt_learner.state_dict().items()},
                         "optimizer": optimizer.state_dict()})
        write_json(output / "prompt_history.json", history)
        _log_epoch("1/2: prompts", record, history, config.prompt_epochs)
    write_json(output / "prompt_history.json", history)  # Repair JSON after interruption.
    optimizer.zero_grad(set_to_none=True)
    return history


@torch.no_grad()
def class_text_features(model, device, batch_size=32):
    labels = torch.arange(model.classifier.out_features, device=device)
    return torch.cat([model.text(batch) for batch in labels.split(batch_size)])


def image_optimizer(model, config):
    groups = [{"params": [p], "lr": config.image_lr * (2 if "bias" in name else 1),
               "base_lr": config.image_lr * (2 if "bias" in name else 1)}
              for name, p in model.named_parameters() if p.requires_grad]
    return torch.optim.Adam(groups, weight_decay=config.weight_decay)


def train_image_epoch(model, loader, sampler, texts, optimizer, device, config, epoch):
    set_seed(config.seed + 10000 + epoch)
    sampler.set_epoch(epoch)
    model.set_stage(2)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * image_lr_factor(epoch)
    totals = defaultdict(float)
    seen = set()
    for images, labels, ids in tqdm(loader, desc=f"Image {epoch}/{config.image_epochs}", leave=False):
        optimizer.zero_grad(set_to_none=True)
        losses = image_losses(model, images.to(device), labels.to(device), texts)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Non-finite CLIP image loss")
        losses["loss"].backward()
        optimizer.step()
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        seen.update(ids)
    optimizer.zero_grad(set_to_none=True)
    return {key: value / len(loader) for key, value in totals.items()}, seen


def run_image_stage(model, rows, selected, device, config, protocol, weights, output, dataset=DATASET):
    signature = run_signature(protocol, config)
    weights, output = Path(weights), Path(output)
    prompt_checkpoint = _resume(weights / "prompt_last.pt", signature)
    if not prompt_checkpoint or len(prompt_checkpoint["history"]) != config.prompt_epochs:
        raise RuntimeError("Complete prompt training before starting image training")
    prompt_state = model.prompt_learner.state_dict()
    if any(not torch.equal(v.detach().cpu(), prompt_checkpoint["prompt"][k]) for k, v in prompt_state.items()):
        raise RuntimeError("Model prompts differ from the completed prompt checkpoint")
    signature["prompt_sha256"] = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes() for _, value in sorted(prompt_state.items()))).hexdigest()
    del prompt_checkpoint
    # Freeze learned texts once. No other queries and no test/gallery labels are involved.
    model.eval()
    texts = class_text_features(model, device).detach()
    model.set_stage(2)
    optimizer = image_optimizer(model, config)
    sampler = StepPKBatchSampler(selected, config, steps=math.ceil(len(selected) / config.batch_size))
    loader = DataLoader(ClipDataset(selected, dataset, train=True), batch_sampler=sampler, num_workers=0)
    last_path = weights / "image_last.pt"
    checkpoint = _resume(last_path, signature)
    if checkpoint:
        model.load_image_state(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best, best_model = checkpoint["history"], checkpoint["best"], checkpoint["best_model"]
        seen = set(checkpoint["seen"])
        del checkpoint
    else:
        if any(path.exists() for path in (output / "image_history.json", weights / "image_best.pt")):
            raise RuntimeError("Missing image_last.pt; restore it or use a new variant")
        baseline = evaluate_inner(model, rows, protocol, device, dataset, config.eval_batch)
        history, seen = [], set()
        best, best_model = {"epoch": 0, "validation": baseline}, model.image_state()
    for epoch in range(len(history) + 1, config.image_epochs + 1):
        synchronize(device)
        started = time.perf_counter()
        losses, epoch_seen = train_image_epoch(model, loader, sampler, texts, optimizer, device, config, epoch)
        synchronize(device)
        train_seconds = time.perf_counter() - started
        seen.update(epoch_seen)
        validation = evaluate_inner(model, rows, protocol, device, dataset, config.eval_batch)
        if not math.isfinite(validation["mAP_at_10"]):
            raise FloatingPointError("Non-finite inner validation mAP")
        record = {"epoch": epoch, "train": losses, "validation": validation,
                  "encoder_lr": config.image_lr * image_lr_factor(epoch),
                  "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - started,
                  "steps": epoch * len(loader), "image_presentations": epoch * len(loader) * config.batch_size,
                  "unique_images_seen": len(seen), "train_images": len(selected)}
        history.append(record)
        if validation["mAP_at_10"] > best["validation"]["mAP_at_10"]:
            best, best_model = {"epoch": epoch, "validation": validation}, model.image_state()
        state = model.image_state()
        # One authoritative atomic file owns history + best. Resume replays an unfinished epoch.
        _save_checkpoint(last_path, {"signature": signature, "history": history, "model": state,
                         "optimizer": optimizer.state_dict(), "best": best, "best_model": best_model,
                         "seen": sorted(seen)})
        if epoch in (10, 30, 50, config.image_epochs):
            _save_checkpoint(weights / f"image_epoch_{epoch:03d}.pt",
                             {"signature": signature, "epoch": epoch, "model": state})
        _save_checkpoint(weights / "image_best.pt", {"signature": signature, **best, "model": best_model})
        write_json(output / "image_history.json", history)
        _log_epoch("2/2: image encoder", record, history, config.image_epochs)
        print(f"inner raw mAP@10: {validation['mAP_at_10']:.4f}; "
              f"best {best['validation']['mAP_at_10']:.4f}, epoch {best['epoch']}", flush=True)
    _save_checkpoint(weights / "image_best.pt", {"signature": signature, **best, "model": best_model})
    write_json(output / "image_history.json", history)
    result = {"signature": signature, "completed_epochs": len(history), "best": best,
              "beats_own_initialization": best["epoch"] > 0,
              "outer_evaluation_run": False, "mvp_changed": False}
    write_json(output / "training_summary.json", result)
    return result


def run_clean_osnet_control(rows, protocol, device, weights, output, dataset=DATASET):
    """Same clean inner split, own fixed OSNet recipe; not an equal-compute claim."""
    config = ExperimentConfig(epochs=20, identities_per_batch=16, images_per_identity=2,
                              encoder_lr=1e-4, metric_loss="supcon", use_bnneck=True)
    set_seed(config.seed)
    model, _ = initialize_experiment(len(protocol["inner"]["train"]), config, device)
    control_protocol = {**protocol, "initializer_sha256": sha256(STOCK_MODEL),
                        "control_recipe": "stock OSNet + avg/BNNeck/SupCon; inner-only; 4000 steps max"}
    try:
        return fit_steps(model, rows, protocol["inner"]["train"], protocol["inner"]["validation"],
                         device, config, StepBudget(), Path(weights) / "osnet_control",
                         Path(output) / "osnet_control", control_protocol, dataset=dataset)
    finally:
        del model
        release_device(device)


def write_results(output):
    """Human-readable report based only on actual saved values, never promised gains."""
    output = Path(output)
    baseline = json.loads((output / "baselines.json").read_text())
    summary = json.loads((output / "training_summary.json").read_text())
    best = summary["best"]
    control_path = output / "osnet_control/training_summary.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    lines = ["# CLIP-ReID: результаты development", "",
             "Все значения ниже — raw mAP@10 на одном inner split. Outer/test не оценивались.", "",
             f"- Исходный CLIP-ReID VeRi: {baseline['clip_vehicle_pretrained']['mAP_at_10']:.4%}.",
             f"- Stock OSNet: {baseline['osnet_stock']['mAP_at_10']:.4%}.",
             f"- CLIP-ReID после отбора: {best['validation']['mAP_at_10']:.4%}, эпоха {best['epoch']}.",
             f"- Выполнено image-эпох: {summary['completed_epochs']}."]
    if control:
        lines.append(f"- Чистый inner-trained OSNet: {control['best_mAP_at_10']:.4%}; "
                     f"инициализация {control['initial_validation']['mAP_at_10']:.4%}.")
    else:
        lines.append("- Чистый inner-trained OSNet пока не запущен; честное сравнение дообучения не завершено.")
    lines += ["", "Эпоха 0 означает, что дообучение не улучшило исходные vehicle-веса.",
              "Это выбор на validation, а не независимое подтверждение прироста. Следующее решение —",
              "после анализа ошибок, затем дополнительные seed/калибровка/outer при перспективном результате.",
              "Порог отказа, reranking и MVP этим экспериментом не изменены."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
