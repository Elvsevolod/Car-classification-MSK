"""Shared deterministic training loop for v41. No outer evaluation or release writes."""
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training import research_io as io, research_models as models
from training.hpo import CameraAwarePKBatchSampler, supervised_contrastive_loss
from training.transreid_model import soft_triplet


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def labeled(rows):
    identities = sorted({r["vehicle_id"] for r in rows})
    labels = {v:i for i,v in enumerate(identities)}
    return [{**r, "label": labels[r["vehicle_id"]]} for r in rows]


def batch_indices(rows, p, k, seed, step):
    sampler = CameraAwarePKBatchSampler(rows, p, k, prefer_cross_camera=True, seed=seed+step)
    # Same first draw for every arm at a given step; resume needs no sampler history.
    rng = random.Random(seed+step)
    if len(sampler.identities) < p:
        raise ValueError("Fewer training identities than P")
    return [i for identity in rng.sample(sampler.identities, p) for i in sampler._sample_identity(identity, rng)]


def fetch(dataset, indices, device, seed):
    seed_all(seed)
    items = [dataset[i] for i in indices]
    return [torch.stack([v[j] for v in items]).to(device) if torch.is_tensor(items[0][j]) else
            torch.tensor([v[j] for v in items], device=device) for j in range(len(items[0]))]


def auxiliary_indices(rows, main_ids, p, k, seed, step):
    # Target-extra control must not repeat an image from the current main batch.
    available = [i for i,r in enumerate(rows) if r["image_id"] not in main_ids]
    subset = [rows[i] for i in available]
    local = batch_indices(subset, p, k, seed, step)
    return [available[i] for i in local]


def domain_loss(model, batch, config, domain, mode):
    clean, robust, labels = batch
    models.set_domain(model, domain, mode, training=False)
    with torch.no_grad(): target = model.embedding(clean)
    models.set_domain(model, domain, mode, training=True)
    raw = model.backbone(robust)
    embedding = model.bnneck(raw)
    head = model.classifier if domain == "main" else model.aux_head
    logits = head(embedding)
    ce = F.cross_entropy(logits, labels, label_smoothing=config["label_smoothing"])
    metric = supervised_contrastive_loss(raw, labels, config["supcon_temperature"])
    consistency = (1-F.cosine_similarity(embedding, target, dim=1)).mean()
    return {"loss": ce+config["metric_weight"]*metric+config["consistency_weight"]*consistency,
            "ce": ce, "metric": metric, "accuracy": (logits.argmax(1)==labels).float().mean()}


def metric_loss(model, images, labels, spec):
    model.train()
    # Frozen prefix stays eval (BN/dropout); only selected blocks are trained.
    if spec.get("family") == "external":
        model.base.eval()
        for module in getattr(model, "trainable_blocks", []): module.train()
    logits, features = model(images)
    ce = torch.stack([F.cross_entropy(v, labels, label_smoothing=.1) for v in logits]).mean()
    distance = torch.stack([soft_triplet(v, labels) if spec["loss"] == "soft_triplet" else
                            supervised_contrastive_loss(v, labels, .1) for v in features]).mean()
    return {"loss": ce+distance, "ce": ce, "metric": distance,
            "accuracy": (logits[0].argmax(1)==labels).float().mean(),
            "feature_norm": features[0].norm(dim=1).mean()}


def gradient_diagnostics(main, auxiliary):
    main_norm = torch.stack([v.float().square().sum() for v in main]).sum().sqrt()
    aux_norm = torch.stack([v.float().square().sum() for v in auxiliary]).sum().sqrt()
    dot = torch.stack([(a.float()*b.float()).sum() for a,b in zip(main, auxiliary)]).sum()
    return {"main_grad_norm": float(main_norm), "weighted_aux_grad_norm": float(aux_norm),
            "weighted_aux_ratio": float(aux_norm/main_norm.clamp_min(1e-12)),
            "gradient_cosine": float(dot/(main_norm*aux_norm).clamp_min(1e-12))}


def optimizer_for(model, spec):
    backbone = model.backbone if spec["family"] == "nive" else model.base
    params = [p for p in backbone.parameters() if p.requires_grad]
    ids = {id(p) for p in params}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return torch.optim.AdamW([{"params": params, "base_lr": spec["lr"]},
                              {"params": heads, "base_lr": spec["lr"]*10}],
                             lr=spec["lr"], weight_decay=spec["weight_decay"], foreach=False)


def lr_step(optimizer, step, horizon, warmup):
    fraction = ((step+1)/warmup if step < warmup else .02+.98*.5*(1+math.cos(
        math.pi*(step-warmup)/max(1,horizon-warmup-1))))
    for group in optimizer.param_groups: group["lr"] = group["base_lr"]*fraction


def save_torch(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def train(model, c, spec, target_steps, horizon, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    signature = io.digest({"run": c["signature"], "spec": spec, "horizon": horizon})
    io.freeze(directory / "trial.json", {"signature": signature, "spec": spec, "horizon": horizon})
    optimizer = optimizer_for(model, spec)
    start, history, elapsed = 0, [], 0.
    pointer = directory / "resume.json"
    if pointer.exists():
        entry = io.read(pointer)
        io.verify(directory, {entry["path"]:entry["sha256"]})
        # Deserialize on CPU: avoid a second full GPU model/optimizer and keep Adam step scalars on CPU.
        saved = torch.load(io.child(directory, entry["path"]), map_location="cpu", weights_only=True)
        if saved["signature"] != signature or saved["step"] > horizon:
            raise ValueError("Resume belongs to another trial/runtime")
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        start, history, elapsed = saved["step"], saved["history"], saved["elapsed"]
        del saved
    if start > target_steps:
        raise ValueError("Cannot evaluate an earlier rung from a later resume; use its saved checkpoint")
    paired = spec["family"] == "nive"
    main_data = models.Images(c["inputs"], c["train"], spec["size"], train=True, paired=paired,
                              interpolation=spec.get("interpolation", "bilinear"))
    aux_rows = c["external"] if spec.get("aux") == "nive" else c["train"]
    aux_data = models.Images(c["inputs"], aux_rows, spec["size"], train=True, paired=True) if paired else None
    parameters = [p for p in model.backbone.parameters() if p.requires_grad] if paired else []
    began = time.perf_counter()
    if c["device"].type == "cuda": torch.cuda.reset_peak_memory_stats()
    for step in range(start, target_steps):
        seed_all(spec["seed"]+step)
        lr_step(optimizer, step, horizon, min(100, horizon//10))
        indices = batch_indices(c["train"], spec["p"], spec["k"], spec["seed"], step)
        batch = fetch(main_data, indices, c["device"], spec["seed"]+100001+step*3)
        optimizer.zero_grad(set_to_none=True)
        diagnostics = {}
        if paired:
            aux_grad = None
            if spec["aux"] != "none" and step < spec["joint_steps"]:
                blocked = {c["train"][i]["image_id"] for i in indices} if spec["aux"] == "target" else set()
                ai = auxiliary_indices(aux_rows, blocked, spec["p"], spec["k"], spec["seed"]+101, step)
                aux_batch = fetch(aux_data, ai, c["device"], spec["seed"]+100000+step*3)
                auxiliary = domain_loss(model, aux_batch, c["manifest"]["config"], "aux", spec["bn_mode"])
                if not torch.isfinite(auxiliary["loss"]): raise FloatingPointError("Auxiliary loss is not finite")
                (spec["alpha"]*auxiliary["loss"]).backward()
                diagnostics["aux_loss"] = float(auxiliary["loss"].detach())
                if step % 25 == 0:
                    aux_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters]
            values = domain_loss(model, batch, c["manifest"]["config"], "main", spec["bn_mode"])
            values["loss"].backward()
            if aux_grad is not None:
                main = [(p.grad if p.grad is not None else torch.zeros_like(p))-a for p,a in zip(parameters, aux_grad)]
                diagnostics.update(gradient_diagnostics(main, aux_grad))
        else:
            values = metric_loss(model, *batch, spec)
            values["loss"].backward()
        if not all(torch.isfinite(v) for v in values.values()): raise FloatingPointError("Non-finite training values")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.get("clip", float("inf")), error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step+1, **{k:float(v.detach()) for k,v in values.items()}, **diagnostics,
                        "grad_norm": float(norm), "lr": [g["lr"] for g in optimizer.param_groups],
                        "main_presentations": (step+1)*spec["p"]*spec["k"],
                        "equivalent_passes": (step+1)*spec["p"]*spec["k"]/len(c["train"])})
        if (step+1) % 25 == 0 or step+1 == target_steps:
            if c["device"].type == "mps":
                from training.research_mac import memory_sample
                history[-1].update(memory_sample())
            seconds = elapsed+time.perf_counter()-began
            print(f"{spec['id']} | step {step+1}/{target_steps} | passes {history[-1]['equivalent_passes']:.2f} | "
                  f"loss {history[-1]['loss']:.4f} | acc {history[-1]['accuracy']:.3f} | {seconds/60:.1f} min", flush=True)
        if (step+1) % 100 == 0 or step+1 == target_steps:
            slot = directory / f"resume_{((step+1)//100)%2}.pt"
            save_torch(slot, {"signature": signature, "step": step+1, "model": model.state_dict(),
                             "optimizer": optimizer.state_dict(), "history": history,
                             "elapsed": elapsed+time.perf_counter()-began})
            io.write(pointer, {"path":slot.name, "sha256":io.sha(slot)})
            io.write(directory / "history.json", history)
    path = directory / f"step_{target_steps:06d}.pt"
    save_torch(path, {"signature":signature, "step":target_steps, "spec":spec, "model":model.state_dict()})
    return {"checkpoint":path.name, "sha256":io.sha(path), "step":target_steps,
            "elapsed_seconds":elapsed+time.perf_counter()-began,
            "peak_torch_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if c["device"].type=="cuda" else None,
            "max_observed_mps_driver_bytes":max((h.get("mps_driver_bytes",0) for h in history),default=0) or None,
            "memory_note":("MPS samples at step ends every 25 steps; NOT a guaranteed peak or total system RAM"
                           if c["device"].type=="mps" else "PyTorch allocator peak, not total board VRAM")}
