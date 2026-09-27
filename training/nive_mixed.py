"""v32: one matched N0/N1 continuation, inner only, never promotes/recalibrates v25."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import copy
import gc
from dataclasses import asdict, replace
from pathlib import Path
import platform
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate

import evaluate as official
from backend.core import ROOT, DATASET, ARTIFACTS, read_rows, sha256, normalize
from training import nive_mixed_data as data, osnet_review_protocol as review, retrieval_policy as policy
from training.audit import digest
from training.hpo import ExperimentConfig, supervised_contrastive_loss
from training.osnet_ablations import Ablation, AblationDataset, optimizer_for
from training.overnight_training import capture_rng, restore_rng
from training.pipeline import set_seed, format_duration
from training.stage6 import StepPKBatchSampler, set_step_learning_rates, write_json

VARIANT = ROOT / "OSNet-AIN-x1.0/variant_32_nive_mixed"
APP = ROOT.parent / "Car-classification-MSK-release-integration"
ARMS = ("N0_target_extra", "N1_nive_mixed")
old = review.old


def verify_files(files):
    for name, expected in files.items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise ValueError(f"Protected file changed: {name}")


def parent_provenance(plan, rows, split):
    directory = ROOT / plan["parent_directory"]
    summary_path = directory / "primary/R1_resolution256/seed_20260915/summary.json"
    verify_files({str(directory / "manifest.json"): plan["parent_manifest_sha256"],
                  str(summary_path): plan["parent_summary_sha256"]})
    manifest, summary = old.load_json(directory / "manifest.json"), old.load_json(summary_path)
    allowed = set(manifest["inner"]["primary"]["train"])
    held = set(manifest["inner"]["primary"]["validation"])
    outer_held = set(split["identities"]["calibration"]) | set(split["identities"]["validation"])
    labels = {identity: i for i, identity in enumerate(sorted(allowed))}
    training_rows = [{**r, "label": labels[r["vehicle_id"]]} for r in rows if r["vehicle_id"] in allowed]
    expected = digest({"context": digest(manifest), "variant": "R1_resolution256", "seed": plan["seed"],
                       "fold": "primary", "stop": summary["stop_step"], "rows": training_rows})
    if (not allowed or allowed & (held | outer_held) or allowed | held != set(split["identities"]["train"])
            or summary["context_signature"] != digest(manifest) or summary["signature"] != expected
            or summary["fold"] != "primary" or summary["seed"] != plan["seed"]
            or summary["variant"] != "R1_resolution256" or summary["train_identities"] != len(allowed)
            or summary["train_images"] != len(training_rows)):
        raise ValueError("Parent is not the exact fold-matched R1; full-train/holdout leakage forbidden")
    entry = summary["checkpoints"][str(plan["parent_step"])]
    path = (directory / entry["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or entry["sha256"] != plan["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint path/hash differs from the fixed pilot")
    verify_files({str(path): entry["sha256"]})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != expected or payload["step"] != plan["parent_step"]:
        raise ValueError("Parent checkpoint metadata mismatch")
    return manifest, summary, allowed, path


def prepare(run_name="pilot_v1", device="mps", dataset=DATASET, nive_root=ROOT / "NiVe1303"):
    """Audits only; writes solely into the new v32 run, never calls old prepare()."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple new RUN_NAME")
    if device not in {"cpu", "mps", "cuda"} or (device == "mps" and not torch.backends.mps.is_available()) or (
            device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested device unavailable (no fallback): {device}")
    plan = old.load_json(VARIANT / "configs/pilot_v1.json")
    if (plan["joint_updates"], plan["tail_updates"], plan["checkpoints"], plan["aux_weight"], plan["seed"]) != (
            1600, 200, [0, 400, 800, 1200, 1600, 1800], .25, 20260915):
        raise ValueError("Use the frozen first pilot, not an unrecorded sweep")
    if (plan["main_batch"] != [16, 2] or plan["aux_batch"] != [16, 2]
            or plan["graph"] != policy.POLICIES["less_graph"] or plan["fusion_weights"] != [.5, .5]
            or plan["save_interval"] != 100 or plan["warmup_updates"] != 100
            or plan["parent_step"] != 800 or plan["automatic_promotion"] or plan["wall_time_limit"] is not None):
        raise ValueError("Pilot schedule/graph/export policy changed")
    output = VARIANT / "runs" / run_name
    if not output.resolve().is_relative_to((VARIANT / "runs").resolve()):
        raise ValueError("Output escapes v32")
    dataset, nive_root = Path(dataset).resolve(), Path(nive_root).resolve()
    rows, split = read_rows(dataset / "train.csv"), old.load_json(ARTIFACTS / "splits.json")
    if sha256(dataset / "train.csv") != split["train_csv_sha256"]:
        raise ValueError("Organizer annotations changed")
    frames = {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg") for r in rows}
    if frames != split["frame_sha256"]:
        raise ValueError("Organizer images changed")
    parent, summary, allowed, parent_path = parent_provenance(plan, rows, split)
    variant = Ablation(**parent["variants"]["R1_resolution256"])
    base = variant.recipe(ExperimentConfig(**parent["base_recipe"]), plan["seed"])
    config = replace(base, encoder_lr=base.encoder_lr * .1)
    if config.encoder_lr != plan["encoder_peak_lr"] or variant.size != 256 or variant.freeze_bn != "none":
        raise ValueError("Parent LR/BN/architecture differs from the fixed recipe")
    if (config.identities_per_batch, config.images_per_identity, config.metric_loss, config.consistency_weight > 0) != (16, 2, "supcon", True):
        raise ValueError("Parent batch/loss does not match the pilot")
    old_nive = ROOT / "OSNet-AIN-x1.0/variant_15_nive_transfer/runs/nive_pilot_v1/manifest.json"
    prior = old.load_json(old_nive)
    if not prior["nive"]["source"]["local_copy_source_confirmed_by_user"]:
        raise ValueError("The NiVe download source has not been confirmed")
    external, inventory = data.audit_nive(nive_root, frames.values())
    if inventory["files_fingerprint"] != prior["nive"]["files_fingerprint"]:
        raise ValueError("NiVe differs from the previously confirmed local copy")
    last_manifest = ROOT / "OSNet-AIN-x1.0/variant_31_multiscale/runs/multiscale_v1/manifest.json"
    protected = dict(old.load_json(last_manifest)["protected"])
    protected.update({str(dataset / "images" / f"{i}.jpg"): h for i, h in frames.items()})
    protected.update({str(nive_root / p): h for p, h in inventory["files"].items()})
    for path in [parent_path, old_nive, last_manifest, ARTIFACTS / "splits.json", dataset / "train.csv",
                 ROOT / "evaluate.py", ROOT / "ORGANIZER_QA.md", APP / "release_decision.json",
                 APP / "models/profiles.json", ROOT / plan["parent_directory"] / "manifest.json",
                 ROOT / plan["parent_directory"] / "primary/R1_resolution256/seed_20260915/summary.json"]:
        protected[str(path)] = sha256(path)
    for path in (APP / "models").rglob("*"):
        if path.is_file():
            protected[str(path)] = sha256(path)
    decision = old.load_json(APP / "release_decision.json")
    if decision["active_profile"] != "MVP_fusion_v25":
        raise ValueError("Active baseline is no longer v25; review before starting this pilot")
    verify_files(protected)
    with old.run_lock(output):
        fingerprint = digest({"frames": frames, "nive": inventory["files_fingerprint"],
                              "allowed": sorted(allowed), "code": sha256(Path(data.__file__))})
        audit = data.audit_domains(output, rows, allowed, dataset, external, nive_root, fingerprint)
        review_path = output / "near_duplicate_review.json"
        external, excluded = data.reviewed_external(external, audit,
                                                    old.load_json(review_path) if review_path.exists() else None)
        target, external = data.label_domains(rows, external, allowed)
        total = plan["joint_updates"] + plan["tail_updates"]
        main = list(StepPKBatchSampler(target, config, total))
        main_ids = [[target[i]["image_id"] for i in b] for b in main]
        auxiliary = {arm: data.cyclic_schedule(target if arm == ARMS[0] else external, plan["joint_updates"],
                     config.identities_per_batch, config.images_per_identity, plan["seed"] + 101,
                     forbidden=main_ids if arm == ARMS[0] else None) for arm in ARMS}
        sources = {str(p): sha256(p) for folder in (ROOT / "training", ROOT / "backend")
                   for p in folder.glob("*.py")}
        sources.update({str(p): sha256(p) for p in (VARIANT / "configs/pilot_v1.json", VARIANT / "SOURCE_NIVE.json")})
        protected[str(output / "domain_audit.json")] = sha256(output / "domain_audit.json")
        if review_path.exists():
            protected[str(review_path)] = sha256(review_path)
        manifest = {"version": 32, "plan": plan, "parent": {"path": str(parent_path),
                    "sha256": sha256(parent_path), "signature": summary["signature"],
                    "train_ids": sorted(allowed), "train_identity_fingerprint": digest(sorted(allowed))},
                    "variant": asdict(variant), "config": asdict(config), "inner": parent["inner"]["primary"],
                    "draws": {k: v for k, v in parent["draws"]["primary"].items() if v["selection_eligible"]},
                    "nive": {**inventory, "source": old.load_json(VARIANT / "SOURCE_NIVE.json"),
                             "excluded": excluded, "used_images": len(external),
                             "used_identities": len({r["vehicle_id"] for r in external}),
                             "used_train_fingerprint": digest(external)},
                    "sampling": {"main_sha256": digest(main), "aux_sha256": {a: digest(b) for a, b in auxiliary.items()},
                        "main": "historical camera-aware PK; identical IDs and augmentation seeds in both arms",
                        "auxiliary": "cyclic images, prefer different view; N0 excludes current main images",
                        "planned_coverage": {a: data.coverage(target if a == ARMS[0] else external, b)
                                             for a, b in auxiliary.items()}},
                    "policy": {"bn": "same train BN; aux first then main; clean views eval/no_grad",
                        "gradients": "L_main + .25 L_aux; two backwards, exactly one optimizer step",
                        "main_head": "retain fold parent classifier", "aux_head": "fresh independent Gaussian .01",
                        "selection": "mean fixed less_graph mAP over three existing primary draws; ties earlier step",
                        "fusion": "one 50/50 cosine mix N_ref+N0 and N_ref+N1 after single-encoder selection",
                        "threshold_fit": False, "original_validation_evaluated": False, "promoted": False,
                        "time_limit": None, "source_test_used_in_loss": False},
                    "runtime": {"device": device, "torch": str(torch.__version__), "numpy": np.__version__,
                                "python": platform.python_version(), "platform": platform.platform(),
                                "torch_threads": torch.get_num_threads(), "cuda": torch.version.cuda,
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                    "protected": protected, "source_sha256": sources,
                    "baseline": {"release_decision": decision, "profiles": old.load_json(APP / "models/profiles.json")}}
        old.freeze_json(output / "manifest.json", manifest)
    print(f"PREFLIGHT OK: parent primary/740 IDs/step800; NiVe {len(external)} train photos; v25 protected", flush=True)
    return {"output": output, "manifest": manifest, "signature": digest(manifest), "plan": plan,
            "rows": rows, "target": target, "external": external, "config": config, "variant": variant,
            "dataset": dataset, "nive_root": nive_root, "device": torch.device(device),
            "main_schedule": main, "aux_schedules": auxiliary, "masks": {}}


def check_inputs(context):
    if digest(context["manifest"]) != context["signature"]:
        raise ValueError("Context changed")
    verify_files(context["manifest"]["protected"])
    verify_files(context["manifest"]["source_sha256"])


class MixedModel(nn.Module):
    def __init__(self, parent, auxiliary_classes):
        super().__init__()
        self.backbone, self.bnneck, self.main_head = parent.backbone, parent.bnneck, parent.classifier
        self.aux_head = nn.Linear(parent.classifier.in_features, auxiliary_classes, bias=False)
        nn.init.normal_(self.aux_head.weight, std=.01)

    def embedding(self, images):
        return self.bnneck(self.backbone(images))


class RetrievalEncoder(nn.Module):
    """Inference-only model; neither classifier nor training IDs are owned by this module."""
    def __init__(self, model):
        super().__init__()
        self.backbone, self.bnneck = copy.deepcopy(model.backbone), copy.deepcopy(model.bnneck)

    def forward(self, images):
        return F.normalize(self.bnneck(self.backbone(images)), dim=1)


def new_model(context, arm):
    if arm not in ARMS:
        raise ValueError("Only the two registered arms are trainable")
    set_seed(context["plan"]["seed"])
    parent = review.initialize(len({r["label"] for r in context["target"]}), context["config"],
                               context["variant"], context["device"])
    entry = context["manifest"]["parent"]
    verify_files({entry["path"]: entry["sha256"]})
    saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
    parent.load_state_dict(saved["model"])
    auxiliary = context["target"] if arm == ARMS[0] else context["external"]
    set_seed(context["plan"]["seed"] + 77)
    return MixedModel(parent, len({r["label"] for r in auxiliary})).to(context["device"])


def domain_loss(model, batch, config, domain):
    if domain not in {"main", "aux"} or config.metric_loss != "supcon":
        raise ValueError("Use the frozen within-domain SupCon/CE recipe")
    clean, robust, labels = batch[:3]
    target = None
    if config.consistency_weight:
        model.eval()
        with torch.no_grad():
            target = model.embedding(clean)
    model.train()
    raw = model.backbone(robust)
    embedding = model.bnneck(raw)
    head = model.main_head if domain == "main" else model.aux_head
    logits = head(embedding)
    ce = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
    metric = supervised_contrastive_loss(raw, labels, config.supcon_temperature)
    consistency = raw.new_zeros(()) if target is None else (1 - F.cosine_similarity(embedding, target, dim=1)).mean()
    total = ce + config.metric_weight * metric + config.consistency_weight * consistency
    return {"loss": total, "ce": ce, "metric": metric, "consistency": consistency,
            "accuracy": (logits.argmax(1) == labels).float().mean()}


def load_batch(dataset, indices, device, seed):
    # Main augmentations are independent of auxiliary class count, IO, evaluation and resume.
    set_seed(seed)
    batch = default_collate([dataset[i] for i in indices])
    return [v.to(device) if torch.is_tensor(v) else v for v in batch]


def update(model, optimizer, main_batch, aux_batch, config, aux_weight, diagnostics=False):
    optimizer.zero_grad(set_to_none=True)
    parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    values, aux_grad = {}, None
    if aux_batch is not None:
        aux = domain_loss(model, aux_batch, config, "aux")
        if not all(torch.isfinite(v).all() for v in aux.values()):
            raise FloatingPointError("Non-finite auxiliary loss")
        (aux_weight * aux["loss"]).backward()
        values.update({"aux_" + k: float(v.detach()) for k, v in aux.items()})
        if diagnostics:
            aux_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters]
    main = domain_loss(model, main_batch, config, "main")
    if not all(torch.isfinite(v).all() for v in main.values()):
        raise FloatingPointError("Non-finite main loss")
    main["loss"].backward()
    if diagnostics:
        current = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
        norm = lambda tensors: float(torch.stack([x.square().sum() for x in tensors]).sum().sqrt())
        values["weighted_aux_backbone_grad_norm"] = norm(aux_grad) if aux_grad is not None else 0.
        values["main_backbone_grad_norm"] = norm([g-a for g, a in zip(current, aux_grad)]) if aux_grad is not None else norm(current)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    values.update({"main_" + k: float(v.detach()) for k, v in main.items()})
    values["total_loss"] = values["main_loss"] + aux_weight * values.get("aux_loss", 0.)
    values["gradient_norm"] = float(norm)
    return values


def bn_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def bn_drift(model, reference):
    current = bn_state(model)
    return {k: float((current[k].double() - value.double()).norm()) for k, value in reference.items()}


def save_boundary(context, arm, model, optimizer, step, history, checkpoints, elapsed, initial_bn):
    directory = context["output"] / "training" / arm
    if step in context["plan"]["checkpoints"]:
        path = directory / f"step_{step:05d}.pt"
        payload = {"signature": context["signature"], "arm": arm, "step": step, "model": model.state_dict()}
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["signature"] != context["signature"] or saved["arm"] != arm or saved["step"] != step or any(
                    not torch.equal(v.cpu(), saved["model"][k]) for k, v in model.state_dict().items()):
                raise ValueError("Existing selectable checkpoint changed")
        else:
            old.save_checkpoint(path, payload)
        checkpoints[str(step)] = {"path": str(path), "sha256": sha256(path)}
    payload = {"signature": context["signature"], "arm": arm, "step": step, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "rng": capture_rng(), "history": history,
               "checkpoints": checkpoints, "elapsed": elapsed, "initial_bn": initial_bn}
    # Write the inactive slot, then atomically commit one pointer. An interruption
    # between the two writes leaves the previous optimizer/RNG checkpoint usable.
    slot = directory / f"resume_{(step // context['plan']['save_interval']) % 2}.pt"
    old.save_checkpoint(slot, payload)
    write_json(directory / "resume.json", {"path": slot.name, "sha256": sha256(slot)})
    write_json(directory / "history.json", history)


def resume_path(directory):
    entry = old.load_json(directory / "resume.json")
    if entry["path"] not in {"resume_0.pt", "resume_1.pt"}:
        raise ValueError("Resume pointer escapes its two local slots")
    path = directory / entry["path"]
    verify_files({str(path): entry["sha256"]})
    return path


def train_arm(context, arm, stop_after=None):
    """Fixed schedule; resume at 100-update boundaries. No selection/evaluation in this loop."""
    directory = context["output"] / "training" / arm
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    directory.mkdir(parents=True, exist_ok=True)
    model = new_model(context, arm)
    optimizer = optimizer_for(model, context["config"])
    initial_bn = bn_state(model)
    start, history, checkpoints, elapsed = 0, [], {}, 0.
    if (directory / "resume.json").exists():
        last = resume_path(directory)
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["signature"] != context["signature"] or saved["arm"] != arm:
            raise ValueError("Resume manifest/arm changed")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, history, checkpoints, elapsed, initial_bn = (saved[k] for k in
            ("step", "history", "checkpoints", "elapsed", "initial_bn"))
        if start < 0 or start > context["plan"]["joint_updates"] + context["plan"]["tail_updates"] or start % context["plan"]["save_interval"]:
            raise ValueError("Invalid resume boundary")
        verify_files({v["path"]: v["sha256"] for v in checkpoints.values()})
        del saved
    else:
        save_boundary(context, arm, model, optimizer, 0, history, checkpoints, elapsed, initial_bn)
    plan = context["plan"]
    total = plan["joint_updates"] + plan["tail_updates"]
    main_dataset = AblationDataset(context["target"], context["variant"], context["dataset"], augment=True)
    auxiliary = (AblationDataset(context["target"], context["variant"], context["dataset"], augment=True)
                 if arm == ARMS[0] else data.NiVeDataset(context["external"], context["variant"], context["nive_root"]))
    budget = old.Budget(total, plan["save_interval"], plan["warmup_updates"])
    for begin in range(start, total, plan["save_interval"]):
        end = min(begin + plan["save_interval"], total)
        began, logs = time.perf_counter(), []
        for step in range(begin, end):
            lrs = set_step_learning_rates(optimizer, context["config"], budget, step)
            aux = (load_batch(auxiliary, context["aux_schedules"][arm][step], context["device"],
                              plan["seed"] + 100000 + step*3) if step < plan["joint_updates"] else None)
            main = load_batch(main_dataset, context["main_schedule"][step], context["device"],
                              plan["seed"] + 100001 + step*3)
            values = update(model, optimizer, main, aux, context["config"], plan["aux_weight"], step % 25 == 0)
            logs.append({"step": step+1, **values, "lr": lrs})
            if (step + 1) % 25 == 0 or step + 1 == total:
                spent = elapsed + time.perf_counter() - began
                print(f"TRAIN {arm}: {step+1}/{total} | {'joint' if aux is not None else 'target-only'} | "
                      f"loss={values['total_loss']:.4f} | elapsed {format_duration(spent)}", flush=True)
        elapsed += time.perf_counter() - began
        history.append({"step": end, "updates": logs, "bn_drift_from_parent": bn_drift(model, initial_bn)})
        save_boundary(context, arm, model, optimizer, end, history, checkpoints, elapsed, initial_bn)
        if stop_after is not None and end >= stop_after and end < total:
            return {"status": "paused", "step": end}
    main_used = data.coverage(context["target"], context["main_schedule"])
    aux_rows = context["target"] if arm == ARMS[0] else context["external"]
    aux_used = data.coverage(aux_rows, context["aux_schedules"][arm])
    report = {"status": "complete", "signature": context["signature"], "arm": arm, "updates": total,
              "main_coverage": main_used, "aux_coverage": aux_used, "logical_images": main_used["logical_images"] + aux_used["logical_images"],
              "encoder_image_forwards": 2*(main_used["logical_images"] + aux_used["logical_images"]),
              "views": "one clean eval + one robust train per logical image", "checkpoints": checkpoints,
              "elapsed_seconds": elapsed, "optimizer": "fresh AdamW; no parent state", "last_sha256": sha256(resume_path(directory))}
    old.freeze_json(directory / "summary.json", report)
    del model, optimizer
    gc.collect()
    return report


def development_rows(context):
    draws = context["manifest"]["draws"]
    ids = sorted({i for d in draws.values() for key in ("query_ids", "gallery_ids") for i in d[key]})
    by_id = {r["image_id"]: r for r in context["rows"]}
    allowed = set(context["manifest"]["inner"]["validation"])
    if not ids or any(by_id[i]["vehicle_id"] not in allowed for i in ids):
        raise ValueError("Evaluation requested images outside the frozen inner holdout")
    return [by_id[i] for i in ids]


def score_features(context, vectors, rows):
    if vectors.shape[0] != len(rows) or not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1., atol=2e-5, rtol=0):
        raise ValueError("Invalid retrieval vectors")
    positions = {r["image_id"]: i for i, r in enumerate(rows)}
    reports = {}
    for name, protocol in context["manifest"]["draws"].items():
        qi, gi = ([positions[i] for i in protocol[key]] for key in ("query_ids", "gallery_ids"))
        q, g = [rows[i] for i in qi], [rows[i] for i in gi]
        ranking = policy.rank_vectors(vectors[qi], vectors[gi], "less_graph")
        qf, gf = policy.frames(q, g)
        diagnostics = policy.query_diagnostics(q, g, ranking)["per_query"]
        measures = {}
        for kind, key in (("raw", "raw_order"), ("ranking", "order")):
            predictions = {r["image_id"]: [g[j]["image_id"] for j in order[:10]]
                           for r, order in zip(q, ranking[key])}
            measures[kind] = official.ranking_metrics(qf, gf, predictions)
            hits = {10: [], 50: []}
            for row, order in zip(q, ranking[key]):
                item = diagnostics[row["image_id"]]
                item[kind]["top10"] = predictions[row["image_id"]]
                if item["known"]:
                    for k in hits:
                        hit = any(g[j]["vehicle_id"] == row["vehicle_id"] and
                                  g[j]["camera_id"] != row["camera_id"] for j in order[:k])
                        item[kind][f"Hit@{k}"] = hit
                        hits[k].append(hit)
            measures[kind].update({f"Hit@{k}": float(np.mean(v)) for k, v in hits.items()})
        reports[name] = {**measures, "per_query": diagnostics}
    return {"mean_map": float(np.mean([v["ranking"]["mAP@10"] for v in reports.values()])),
            "mean_raw_map": float(np.mean([v["raw"]["mAP@10"] for v in reports.values()])), "draws": reports}


def feature_task(context, name, checkpoint, arm=ARMS[0]):
    directory = context["output"] / "evaluation" / name
    receipt = directory / "complete.json"
    identity = {"signature": context["signature"], "checkpoint": checkpoint}
    if receipt.exists():
        saved = old.load_json(receipt)
        if saved["identity"] != identity:
            raise ValueError("Feature task identity changed")
        verify_files({str(directory / k): v for k, v in saved["files"].items()})
        return old.load_json(directory / "metrics.json"), np.load(directory / "features.npy", allow_pickle=False)
    directory.mkdir(parents=True, exist_ok=True)
    rows = development_rows(context)
    model = new_model(context, arm)
    verify_files({checkpoint["path"]: checkpoint["sha256"]})
    if name != "N_ref":
        saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
        if saved["signature"] != context["signature"] or saved["arm"] != arm:
            raise ValueError("Evaluation checkpoint provenance mismatch")
        model.load_state_dict(saved["model"])
    encoded = old.encode(model, rows, context, context["variant"])
    vectors = np.stack([encoded[r["image_id"]] for r in rows]).astype(np.float32)
    result = score_features(context, vectors, rows)
    np.save(directory / "features.npy", vectors)
    write_json(directory / "metrics.json", result)
    write_json(directory / "order.json", [r["image_id"] for r in rows])
    write_json(receipt, {"identity": identity, "files": {p: sha256(directory / p)
                          for p in ("features.npy", "metrics.json", "order.json")}})
    print(f"EVAL {name}: fixed graph={result['mean_map']:.6f}, raw={result['mean_raw_map']:.6f}", flush=True)
    del model
    gc.collect()
    return result, vectors


def select_steps(evaluations):
    return {arm: min(values, key=lambda step: (-values[step]["mean_map"], int(step)))
            for arm, values in evaluations.items()}


def paired_ap(before, after):
    by_draw = {}
    for name, left in before["draws"].items():
        right = after["draws"][name]
        if set(left["per_query"]) != set(right["per_query"]):
            raise ValueError("Unpaired development queries")
        delta = np.array([right["per_query"][q]["ranking"]["ap"] - p["ranking"]["ap"]
                          for q, p in left["per_query"].items() if p["known"]])
        by_draw[name] = {"delta_map": float(delta.mean()), "better": int((delta > 1e-12).sum()),
                         "worse": int((delta < -1e-12).sum()), "equal": int((abs(delta) <= 1e-12).sum())}
    return {"delta_map": after["mean_map"] - before["mean_map"], "draws": by_draw,
            "limitation": "Draws share identities, not independent datasets"}


def export_encoder(context, arm, checkpoint):
    """Portable head-free state + ONNX; not a release profile or threshold calibration."""
    directory = context["output"] / "exports" / arm
    receipt = directory / "complete.json"
    identity = {"signature": context["signature"], "checkpoint": checkpoint}
    if receipt.exists():
        saved = old.load_json(receipt)
        if saved["identity"] != identity:
            raise ValueError("Export selection changed")
        verify_files({str(directory / k): v for k, v in saved["files"].items()})
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    model = new_model(context, arm)
    verify_files({checkpoint["path"]: checkpoint["sha256"]})
    saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if saved["signature"] != context["signature"] or saved["arm"] != arm:
        raise ValueError("Export checkpoint provenance mismatch")
    model.load_state_dict(saved["model"])
    encoder = RetrievalEncoder(model).cpu().eval()
    if any(not k.startswith(("backbone.", "bnneck.")) for k in encoder.state_dict()):
        raise ValueError("Training heads entered encoder export")
    old.save_checkpoint(directory / "encoder.pt", encoder.state_dict())
    rows = development_rows(context)[:8]
    dataset = AblationDataset(rows, context["variant"], context["dataset"])
    images = torch.stack([dataset[i][0] for i in range(len(rows))])
    path = directory / "encoder.onnx"
    torch.onnx.export(encoder, images[:2], path, input_names=["images"], output_names=["embedding"],
                      dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx
    import onnxruntime as ort
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or any("head" in p.name or "classifier" in p.name for p in graph.graph.initializer):
        raise ValueError("Non-image input or ID-head in ONNX")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = encoder(images).numpy()
    actual = session.run(None, {"images": images.numpy()})[0]
    single = np.concatenate([session.run(None, {"images": x[None].numpy()})[0] for x in images])
    if actual.shape != (len(rows), 512) or not np.allclose(actual, expected, rtol=0, atol=2e-5) or not np.allclose(
            single, actual, rtol=0, atol=2e-5):
        raise ValueError("Head-free CPU ONNX parity/batch independence failed; tolerances are fixed")
    result = {"identity": identity, "dimension": 512, "input_size": 256, "heads_exported": False,
              "preprocessing": "same organizer bbox, RGB, bilinear256, ImageNet normalization, L2",
              "max_onnx_error": float(abs(actual-expected).max()), "max_batch_error": float(abs(single-actual).max()),
              "threshold": None, "promoted": False, "GPU_verified": False,
              "files": {p: sha256(directory / p) for p in ("encoder.pt", "encoder.onnx")}}
    write_json(receipt, result)
    del model, encoder, session
    gc.collect()
    return result


def run_experiment(context):
    check_inputs(context)
    output, started = context["output"], time.perf_counter()
    with old.run_lock(output):
        reference, ref_vectors = feature_task(context, "N_ref", {
            k: context["manifest"]["parent"][k] for k in ("path", "sha256")})
        summaries, evaluations, vectors = {}, {}, {}
        for arm in ARMS:
            print(f"STAGE {arm}: paired continuation; no original validation", flush=True)
            summaries[arm] = train_arm(context, arm)
            evaluations[arm], vectors[arm] = {}, {}
            for step in context["plan"]["checkpoints"]:
                report, features = feature_task(context, f"{arm}_{step:05d}", summaries[arm]["checkpoints"][str(step)], arm)
                evaluations[arm][str(step)], vectors[arm][str(step)] = report, features
                if step == 0 and not np.array_equal(features, ref_vectors):
                    raise ValueError("The two arms no longer start from identical parent embeddings")
            check_inputs(context)
        selected = select_steps(evaluations)
        old.freeze_json(output / "frozen_selection.json", {"signature": context["signature"], "steps": selected,
            "criterion": "mean fixed graph mAP on three primary draws; ties prefer fewer updates"})
        mixtures, exports = {}, {}
        for arm in ARMS:
            combined = normalize(np.concatenate([ref_vectors, vectors[arm][selected[arm]]], axis=1))
            mixtures[arm] = score_features(context, combined, development_rows(context))
            exports[arm] = export_encoder(context, arm, summaries[arm]["checkpoints"][selected[arm]])
        a, b = (evaluations[arm][selected[arm]] for arm in ARMS)
        single_gain = b["mean_map"] > max(reference["mean_map"], a["mean_map"]) + 1e-12
        mixture_gain = mixtures[ARMS[1]]["mean_map"] > max(reference["mean_map"], a["mean_map"],
                                                               mixtures[ARMS[0]]["mean_map"]) + 1e-12
        decision = ("continue" if single_gain or mixture_gain else "diagnose_once"
                    if summaries[ARMS[1]]["aux_coverage"]["image_fraction"] < .9 else "stop")
        check_inputs(context)
        results = {"status": "complete", "signature": context["signature"], "reference": reference,
                   "training": summaries, "evaluations": evaluations, "selection": selected,
                   "fixed_half_reference_mixtures": mixtures, "exports": exports,
                   "N1_vs_N0": paired_ap(a, b), "mixture_N1_vs_N0": paired_ap(mixtures[ARMS[0]], mixtures[ARMS[1]]),
                   "decision": decision, "decision_basis": {"single_gain": single_gain, "mixture_gain": mixture_gain,
                       "rule": "continue = positive screening only; diagnose_once = auxiliary coverage below .90; otherwise stop"},
                   "protected_unchanged": True, "threshold_fit": False, "original_validation_evaluated": False,
                   "promoted": False, "elapsed_this_call_seconds": time.perf_counter() - started}
        if (output / "results.json").exists():
            prior = old.load_json(output / "results.json")
            if {k: v for k, v in prior.items() if k != "elapsed_this_call_seconds"} != {
                    k: v for k, v in results.items() if k != "elapsed_this_call_seconds"}:
                raise ValueError("Completed result changed")
            results = prior
        lines = ["# v32 — N0/N1: mixed NiVe pilot", "", "Статус: complete. Исходная validation не оценивалась; v25 не изменён.",
                 "", "| Модель | Выбранный update | Mean raw mAP | Mean fixed-graph mAP |", "|---|---:|---:|---:|",
                 f"| N_ref | 0 | {reference['mean_raw_map']:.6f} | {reference['mean_map']:.6f} |"]
        for arm in ARMS:
            chosen = evaluations[arm][selected[arm]]
            lines.append(f"| {arm} | {selected[arm]} | {chosen['mean_raw_map']:.6f} | {chosen['mean_map']:.6f} |")
            mix = mixtures[arm]
            lines.append(f"| 50% N_ref + 50% {arm} | {selected[arm]} | {mix['mean_raw_map']:.6f} | {mix['mean_map']:.6f} |")
        lines += ["", f"Рекомендация screening: **{decision}**, не решение о релизе.",
                  "", "Все назначенные checkpoint:", "", "| Arm | Update | Raw | Graph |", "|---|---:|---:|---:|"]
        for arm in ARMS:
            for step, item in evaluations[arm].items():
                lines.append(f"| {arm} | {step} | {item['mean_raw_map']:.6f} | {item['mean_map']:.6f} |")
        lines += ["", "## Бюджет и ограничения", "", "1800 updates на ветвь: 1600 joint + 200 target-only.",
                  "108800 логических предъявлений на ветвь; N1: 57600 organizer + 51200 NiVe; N0: 108800 organizer.",
                  "Clean/robust дают 217600 image-forwards, а не удвоенное число уникальных фото.",
                  "Aux-loss .25 не означает 25% градиента; реальные нормы backbone-градиентов записаны каждые 25 updates.",
                  "Основные organizer-пачки/аугментации одинаковы; общее target-exposure различается намеренно.",
                  "BN обновляется одинаково: aux robust, затем main robust; clean eval; tail только target в обеих ветвях.",
                  "Primary draws используют одни identity и не являются независимыми датасетами. Это адаптивный development-screening.",
                  "Frozen step/50:50 mixture не подбирают граф/порог. Active v25 не является inner-контролем (его R1 видел holdout).",
                  "Negative pilot не доказывает бесполезность всех внешних данных; positive pilot не означает прирост над v25.",
                  "NiVe test/маски не обучались; bbox/evaluator не менялись. Номерная устойчивость и GPU не подтверждены.",
                  "", "## Покрытие NiVe", "",
                  f"{summaries[ARMS[1]]['aux_coverage']['unique_images']}/{len(context['external'])} уникальных train-фото использовано.",
                  "Подробности: manifest.json, training/*/history.json, evaluation/*/metrics.json, frozen_selection.json, results.json."]
        report_text = "\n".join(lines) + "\n"
        report_path = output / "REPORT.md"
        if report_path.exists() and report_path.read_text(encoding="utf-8") != report_text:
            raise ValueError("Completed report changed")
        if not report_path.exists():
            temporary = output / "REPORT.md.tmp"
            temporary.write_text(report_text, encoding="utf-8")
            temporary.replace(report_path)
        # The completion marker is last, after every human-readable artifact.
        if not (output / "results.json").exists():
            write_json(output / "results.json", results)
    return results


def technical_smoke(context):
    with old.run_lock(context["output"]):
        return _technical_smoke(context)


def _technical_smoke(context):
    """Three disposable updates per arm; real pilot always starts from the untouched parent."""
    check_inputs(context)
    directory = context["output"] / "technical_smoke"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        saved = old.load_json(report_path)
        if saved["signature"] != context["signature"]:
            raise ValueError("Smoke belongs to different inputs")
        for arm, item in saved["arms"].items():
            verify_files({str(directory / "exports" / arm / p): h for p, h in item["export"]["files"].items()})
        return saved
    reports = {}
    main_dataset = AblationDataset(context["target"], context["variant"], context["dataset"], augment=True)
    for arm in ARMS:
        model = new_model(context, arm)
        optimizer = optimizer_for(model, context["config"])
        aux_dataset = (main_dataset if arm == ARMS[0] else
                       data.NiVeDataset(context["external"], context["variant"], context["nive_root"]))
        logs = []
        for step in range(3):
            aux = (load_batch(aux_dataset, context["aux_schedules"][arm][step], context["device"], 101+step)
                   if step < 2 else None)
            main = load_batch(main_dataset, context["main_schedule"][step], context["device"], 201+step)
            logs.append(update(model, optimizer, main, aux, context["config"], .25, diagnostics=True))
        path = directory / f"{arm}.pt"
        old.save_checkpoint(path, {"model": model.state_dict(), "signature": context["signature"], "arm": arm, "step": 3})
        exported = export_encoder({**context, "output": directory}, arm, {"path": str(path), "sha256": sha256(path)})
        reports[arm] = {"updates": logs, "export": exported}
        del model, optimizer
        gc.collect()
    check_inputs(context)
    report = {"signature": context["signature"], "status": "passed", "arms": reports,
              "note": "Technical only: 3 disposable updates/arm, not quality evidence"}
    write_json(report_path, report)
    return report
