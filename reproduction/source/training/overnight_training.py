"""Isolated fixed-step CE-scale / relational-distillation training pilots.

No historical trainer is patched. Only inner train identities enter optimization;
the caller evaluates the fixed final state with its frozen retrieval policy.
"""
import gc
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from backend.core import sha256
from training import osnet_review_protocol as review
from training.audit import digest
from training.hpo import supervised_contrastive_loss
from training.osnet_ablations import AblationDataset, optimizer_for
from training.pipeline import format_duration, set_seed
from training.stage6 import StepPKBatchSampler, set_step_learning_rates, write_json


class TrainingPaused(RuntimeError):
    """A requested boundary pause, with model/optimizer/RNG already persisted."""


@dataclass(frozen=True)
class TrainingJob:
    name: str
    base_variant: str
    ce_scale: float = 1.
    relation_weight: float = 0.
    stop_step: int = 800
    lr_horizon: int = 1700

    def validate(self):
        if (not self.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                 for c in self.name)
                or self.base_variant not in {"R1_resolution256", "K2_color32_fixed"}
                or not math.isfinite(self.ce_scale) or self.ce_scale <= 0
                or not math.isfinite(self.relation_weight) or self.relation_weight < 0
                or not 0 < self.stop_step <= self.lr_horizon):
            raise ValueError("Invalid overnight training job")
        if self.relation_weight and (self.base_variant != "R1_resolution256" or self.ce_scale != 1.):
            raise ValueError("Distillation changes only the R1 relational loss")


def training_jobs():
    return {job.name: job for job in (
        TrainingJob("R1_control", "R1_resolution256"),
        TrainingJob("K2_ce1", "K2_color32_fixed"),
        TrainingJob("K2_ce_sqrt544", "K2_color32_fixed", ce_scale=math.sqrt(544)),
        TrainingJob("R1_kd01", "R1_resolution256", relation_weight=.1),
        TrainingJob("R1_kd1", "R1_resolution256", relation_weight=1.),
    )}


def fold_training_ids(context, fold):
    """Explicitly disallow full-train weights and identity leakage in this pilot."""
    if fold not in {"primary", "alternate"}:
        raise ValueError("Overnight training only permits primary/alternate inner folds")
    partition = context["manifest"]["inner"][fold]
    allowed = set(partition["train"])
    held = {identity for name, values in partition.items() if name != "train" for identity in values}
    outer = context["split"]["identities"]
    outer_held = {identity for name, values in outer.items() if name != "train" for identity in values}
    if not allowed or allowed & (held | outer_held) or not allowed <= set(outer["train"]):
        raise ValueError("Training identities overlap a protected holdout")
    return allowed


def teacher_specs(context, teacher_context, summaries, fold):
    if teacher_context is None or len(summaries) != 3:
        raise ValueError("Distillation requires exactly three frozen same-fold teachers")
    allowed = fold_training_ids(context, fold)
    if allowed != fold_training_ids(teacher_context, fold):
        raise ValueError("Teacher/student training identities differ")
    if {s["seed"] for s in summaries} != set(teacher_context["seeds"]):
        raise ValueError("Use the three predetermined teacher seeds without member selection")
    specs = []
    for summary in summaries:
        if (summary["fold"] != fold or summary["variant"] != "R1_resolution256"
                or summary["context_signature"] != teacher_context["signature"]
                or summary["train_identities"] != len(allowed)):
            raise ValueError("Teacher provenance is not the matching inner-fold R1")
        entry = summary["checkpoints"].get("800")
        if entry is None:
            raise ValueError("Teacher lacks the frozen step800 checkpoint")
        path = (teacher_context["output"] / entry["path"]).resolve()
        if not path.is_relative_to(teacher_context["output"].resolve()) or sha256(path) != entry["sha256"]:
            raise ValueError("Teacher checkpoint path/hash mismatch")
        specs.append({"seed": summary["seed"], "fold": fold, "step": 800,
                      "path": str(path), "sha256": entry["sha256"], "signature": summary["signature"]})
    return sorted(specs, key=lambda value: value["seed"])


def load_teachers(teacher_context, summaries):
    result = []
    for summary in sorted(summaries, key=lambda value: value["seed"]):
        model, _ = review.load_model(teacher_context, {**summary, "stop_step": 800})
        model.eval().requires_grad_(False)
        result.append(model)
    return result


def relational_gram(embeddings):
    """Average relations, not independently trained vector coordinates."""
    if not embeddings:
        raise ValueError("Empty teacher ensemble")
    grams = []
    batch = embeddings[0].shape[0]
    for features in embeddings:
        if features.ndim != 2 or features.shape[0] != batch or batch < 2:
            raise ValueError("Relational features need aligned batches of at least two")
        features = F.normalize(features, dim=1)
        grams.append(features @ features.T)
    return torch.stack(grams).mean(0)


def relational_loss(student, target_gram):
    student_gram = relational_gram([student])
    if student_gram.shape != target_gram.shape:
        raise ValueError("Student and teacher pair counts differ")
    off_diagonal = ~torch.eye(student_gram.shape[0], dtype=torch.bool, device=student.device)
    return F.mse_loss(student_gram[off_diagonal], target_gram.detach()[off_diagonal])


def training_losses(model, clean, robust, labels, config, job, teachers=(), diagnostics=False):
    if config.metric_loss != "supcon":
        raise ValueError("These controlled pilots retain the original SupCon metric")
    if bool(teachers) != bool(job.relation_weight):
        raise ValueError("Teacher ensemble and distillation weight must agree")
    target = None
    if config.consistency_weight:
        model.eval()
        with torch.no_grad():
            target = model.embedding(clean)
    relation_target = None
    if teachers:
        with torch.no_grad():
            for teacher in teachers:
                teacher.eval()
                if any(p.requires_grad for p in teacher.parameters()):
                    raise ValueError("Teacher parameters must be frozen")
            relation_target = relational_gram([teacher.embedding(robust) for teacher in teachers])
    model.train()
    logits, raw, embedding = model(robust)
    logits = logits * job.ce_scale  # Classification only: metric/retrieval tensors are unchanged.
    classification = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
    metric = supervised_contrastive_loss(raw, labels, config.supcon_temperature)
    consistency = raw.new_zeros(()) if target is None else (1 - F.cosine_similarity(embedding, target, dim=1)).mean()
    relation = raw.new_zeros(()) if relation_target is None else relational_loss(embedding, relation_target)
    total = classification + config.metric_weight * metric + config.consistency_weight * consistency + job.relation_weight * relation
    values = {"loss": total, "classification": classification, "metric": metric, "consistency": consistency,
              "relation": relation, "embedding_norm": raw.norm(dim=1).mean(),
              "embedding_std": embedding.std(dim=0).mean(), "logits_std": logits.std(unbiased=False),
              "classifier_norm": model.classifier.weight.norm(),
              "train_accuracy": (logits.argmax(1) == labels).float().mean()}
    if diagnostics:
        # These are gradients at the shared raw feature tensor, NOT parameter-gradient norms.
        ce_grad, = torch.autograd.grad(classification, raw, retain_graph=True)
        metric_grad, = torch.autograd.grad(config.metric_weight * metric, raw, retain_graph=True)
        values.update(ce_feature_grad_norm=ce_grad.norm(), metric_feature_grad_norm=metric_grad.norm())
    return values


def capture_rng():
    numpy_state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
              "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def initialize(classes, config, variant, device):
    return review.initialize(classes, config, variant, device)


def fit_job(context, job, seed, fold="primary", teacher_context=None, teacher_summaries=(), should_stop=None):
    """Resume only this job; save boundaries, never select a checkpoint or read outer."""
    job.validate()
    if seed not in context["seeds"] or context["budget"].max_steps != job.lr_horizon:
        raise ValueError("Use frozen training seeds and the full original LR horizon")
    allowed = fold_training_ids(context, fold)
    labels = {identity: index for index, identity in enumerate(sorted(allowed))}
    rows = [{**row, "label": labels[row["vehicle_id"]]} for row in context["rows"] if row["vehicle_id"] in allowed]
    teachers = teacher_specs(context, teacher_context, teacher_summaries, fold) if job.relation_weight else []
    if not job.relation_weight and teacher_summaries:
        raise ValueError("Control/CE jobs must not receive teachers")
    signature = digest({"context": context["signature"], "job": asdict(job), "fold": fold, "seed": seed,
                        "rows": rows, "teachers": teachers, "source_sha256": sha256(Path(__file__))})
    directory = context["output"] / "training" / fold / job.name / f"seed_{seed}"
    with review.old.run_lock(directory):
        return _fit_job(context, job, seed, fold, rows, labels, signature, directory, teachers,
                        teacher_context, teacher_summaries, should_stop)


def _fit_job(context, job, seed, fold, rows, labels, signature, directory, teacher_metadata,
             teacher_context, teacher_summaries, should_stop):
    summary_path, last_path = directory / "summary.json", directory / "last.pt"
    if summary_path.exists():
        summary = review.old.load_json(summary_path)
        if (summary["signature"] != signature or sha256(last_path) != summary["last_sha256"]
                or digest({k: v for k, v in summary.items() if k != "digest"}) != summary["digest"]):
            raise ValueError("Completed overnight job changed")
        for entry in summary["checkpoints"].values():
            if sha256(context["output"] / entry["path"]) != entry["sha256"]:
                raise ValueError("Completed overnight checkpoint changed")
        return summary
    variant = context["variants"][job.base_variant]
    config = variant.recipe(context["base"], seed)
    teachers = load_teachers(teacher_context, teacher_summaries) if teacher_metadata else []
    set_seed(seed)  # Teacher construction must not change student initialization/sampling.
    model = initialize(len(labels), config, variant, context["device"])
    optimizer = optimizer_for(model, config)
    history, checkpoints, start, elapsed = [], {}, 0, 0.
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Resume job configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, checkpoints, start, elapsed = saved["history"], saved["checkpoints"], saved["step"], saved["elapsed"]
        for entry in checkpoints.values():
            if sha256(context["output"] / entry["path"]) != entry["sha256"]:
                raise ValueError("Resumed overnight checkpoint changed")
        del saved
    dataset = AblationDataset(rows, variant, context["dataset"], augment=True)
    for end in context["budget"].boundaries(job.stop_step):
        if end <= start:
            continue
        set_seed(seed + start)  # Exactly the v16 block schedule, including step285.
        sampler = StepPKBatchSampler(rows, config, end - start)
        sampler.set_epoch(start)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        totals, counts, began = {}, {}, time.perf_counter()
        for offset, (clean, robust, target, _image_ids) in enumerate(loader):
            step = start + offset
            set_step_learning_rates(optimizer, config, context["budget"], step)
            values = training_losses(model, clean.to(context["device"]), robust.to(context["device"]),
                                     target.to(context["device"]), config, job, teachers,
                                     diagnostics=(step % 25 == 0))
            if not all(bool(torch.isfinite(value).all()) for value in values.values()):
                raise FloatingPointError("Non-finite overnight training loss/diagnostic")
            optimizer.zero_grad(set_to_none=True)
            values["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            for key, value in {**values, "gradient_norm": norm}.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())
                counts[key] = counts.get(key, 0) + 1
            if (step + 1) % 25 == 0 or step + 1 == end:
                spent = elapsed + time.perf_counter() - began
                print(f"{fold}/{job.name}/{seed}: {step + 1}/{job.stop_step} | loss={float(values['loss'].detach()):.4f} | "
                      f"elapsed {format_duration(spent)} | ETA {format_duration(spent / (step + 1) * (job.stop_step-step-1))}", flush=True)
        elapsed += time.perf_counter() - began
        history.append({"step": end, "train": {key: value / counts[key] for key, value in totals.items()},
                        "diagnostic_counts": counts})
        path = directory / f"step_{end:05d}.pt"
        review.old.save_checkpoint(path, {"signature": signature, "model": model.state_dict(), "step": end})
        checkpoints[str(end)] = {"path": str(path.relative_to(context["output"])), "sha256": sha256(path)}
        review.old.save_checkpoint(last_path, {"signature": signature, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": capture_rng(), "step": end, "history": history,
            "checkpoints": checkpoints, "elapsed": elapsed})
        write_json(directory / "history.json", history)
        start = end
        if end < job.stop_step and should_stop is not None and should_stop():
            # Release bulky owners before the exception propagates to the queue.
            del model, optimizer, teachers, values, loader
            gc.collect()
            raise TrainingPaused(f"Paused {fold}/{job.name}/{seed} after saved step {end}/{job.stop_step}")
    summary = {"signature": signature, "context_signature": context["signature"], "job": asdict(job),
        "variant": job.base_variant, "seed": seed, "fold": fold, "stop_step": job.stop_step,
        "lr_horizon": job.lr_horizon, "updates": job.stop_step, "train_identities": len(labels),
        "train_images": len(rows), "train_identity_digest": digest(sorted(labels)), "teachers": teacher_metadata,
        "elapsed_seconds": elapsed, "history": history, "checkpoints": checkpoints, "last_sha256": sha256(last_path),
        "gradient_diagnostic": "CE and weighted metric gradients at shared raw feature tensor, every 25 steps",
        "outer_evaluated": False, "checkpoint_selection": "fixed step; no best-checkpoint search"}
    summary["digest"] = digest(summary)
    write_json(summary_path, summary)
    del model, optimizer, teachers
    gc.collect()
    return summary


def load_job_model(context, summary):
    if (summary["context_signature"] != context["signature"]
            or digest({key: value for key, value in summary.items() if key != "digest"}) != summary["digest"]):
        raise ValueError("Overnight summary signature/digest mismatch")
    job = TrainingJob(**summary["job"])
    job.validate()
    entry = summary["checkpoints"][str(job.stop_step)]
    path = (context["output"] / entry["path"]).resolve()
    if not path.is_relative_to(context["output"].resolve()) or sha256(path) != entry["sha256"]:
        raise ValueError("Overnight checkpoint path/hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != summary["signature"] or payload["step"] != job.stop_step:
        raise ValueError("Overnight checkpoint metadata mismatch")
    variant = context["variants"][job.base_variant]
    model = initialize(summary["train_identities"], variant.recipe(context["base"], summary["seed"]),
                       variant, context["device"])
    model.load_state_dict(payload["model"])
    return model.eval(), variant
