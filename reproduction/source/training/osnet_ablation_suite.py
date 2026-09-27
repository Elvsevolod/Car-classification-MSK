"""Resumable, isolated OSNet ablations; the notebook is the user-run entry point."""
import copy
import fcntl
import gc
import json
import platform
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from backend.core import ARTIFACTS, DATASET, MODEL, ROOT, STOCK_MODEL, normalize, read_rows, sha256
from backend.evaluate import SEED, make_protocol
from backend.rerank import rerank_protocol
from backend.scoring import calibrate, metrics, ranked_queries
from training.audit import digest
from training.hpo import ExperimentConfig
from training.osnet_ablations import (AblationDataset, BalancedPositiveSampler,
                                     InferenceEncoder, MemoryBank, ema_teacher,
                                     experiment_grid, initialize, losses, optimizer_for,
                                     update_teacher)
from training.pipeline import format_duration, set_seed
from training.stage6 import (StepPKBatchSampler, audit_partitions, set_step_learning_rates,
                             write_json)


VARIANT = ROOT / "OSNet-AIN-x1.0/variant_14_controlled_ablations"
RECIPE = ROOT / "OSNet-AIN-x1.0/variant_02_hpo_bnneck_supcon/results/selected_run_02/summary.json"
MASK_ROOT = ROOT / "OSNet-AIN-x1.0/variant_08_masked_hpo/results"
DETECTOR = ROOT / "YOLO11/variant_01_anonymized_regions"
STOCK_SHA256 = "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f"
MVP_SHA256 = "01466f503232467224774b6e3bafe6c4393b1de30b47b1bd714908a62f2006a2"
# Audited, completed suite_v1. Only its unchanged B0 may cross the constructor repair.
CONTROL_REFERENCE_SHA256 = "620b1e5c13c05ec9fd3b1c7fe760e5f6ee0d05eaa127b2b177f08858243af1d1"


@dataclass(frozen=True)
class Budget:
    max_steps: int = 1700
    evaluation_interval: int = 200
    warmup_steps: int = 100

    def validate(self):
        if not 0 <= self.warmup_steps < self.max_steps or not 0 < self.evaluation_interval <= self.max_steps:
            raise ValueError("Invalid step budget")

    def boundaries(self, stop):
        # 285 is only a diagnostic point, NOT a reproduction of the old epoch LR schedule.
        # 850 permits an exposure-matched P*K=64 vs control-1700 comparison.
        return sorted({min(step, stop) for step in
                       [*range(self.evaluation_interval, stop + 1, self.evaluation_interval), 285, 850, stop]
                       if min(step, stop) > 0})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_json(path):
    value = load_json(path)
    fingerprint = value.pop("fingerprint")
    if digest(value) != fingerprint:
        raise ValueError(f"Corrupted fingerprint: {path}")
    return value


def freeze_json(path, value):
    if Path(path).exists():
        if load_json(path) != value:
            raise ValueError(f"Changed protocol/configuration; choose a new RUN_NAME: {path}")
    else:
        write_json(Path(path), value)


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This RUN_NAME is already running in another kernel") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_masks(rows, split, frames):
    """Read existing frozen predictions, without exporting labels or invoking YOLO."""
    from training.masked_hpo import attach_masks, inner_split

    signature = checked_json(MASK_ROOT / "protocol.json")["signature"]
    cache = checked_json(MASK_ROOT / "cache/automatic_masks.json")
    plan = checked_json(DETECTOR / "annotation/mask_plan.json")
    calibration = DETECTOR / "runs/pilot_01/calibration_v1/calibration.json"
    if (signature["rows"] != rows or signature["frames"] != frames
            or signature["outer_split_sha256"] != sha256(ARTIFACTS / "splits.json")
            or signature["stock_sha256"] != sha256(STOCK_MODEL)
            or signature["detector_sha256"] != sha256(DETECTOR / "runs/pilot_01/weights/best.pt")
            or signature["calibration_sha256"] != sha256(calibration)
            or cache["signature"]["protocol"] != digest(signature)
            or plan["outer_split_sha256"] != signature["outer_split_sha256"]):
        raise ValueError("Mask provenance changed; do not silently regenerate cache")
    for image_id, item in plan["images"].items():
        if item["frame_sha256"] != frames[image_id]:
            raise ValueError("Detector plan frame mismatch")
    attach_masks(rows, cache)  # Validate full coverage and crop-local coordinates; discard copies.
    inner = {name: inner_split(rows, split["identities"]["train"], frames, plan, seed)
             for name, seed in (("primary", SEED), ("alternate", SEED + 7919))}
    return cache["images"], inner


def control_reference(run_name, manifest):
    """Allow read-only B0 reuse across this specific, tested constructor repair only."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Reference run must be a simple directory name")
    directory = VARIANT / "runs" / run_name
    if not directory.resolve().is_relative_to((VARIANT / "runs").resolve()):
        raise ValueError("Reference run must stay inside variant_14/runs")
    if sha256(directory / "manifest.json") != CONTROL_REFERENCE_SHA256:
        raise ValueError("Only the audited suite_v1 manifest can supply a reused control")
    original = load_json(directory / "manifest.json")
    for key in set(original) | set(manifest):
        if key not in {"variants", "source_sha256"} and original.get(key) != manifest.get(key):
            raise ValueError(f"Reused control is incompatible: {key}")
    if original["variants"]["B0_control"] != manifest["variants"]["B0_control"]:
        raise ValueError("Reused B0 ablation changed")
    old_sources, new_sources = original["source_sha256"], manifest["source_sha256"]
    repair_sources = {"training/osnet_ablations.py", "training/osnet_ablation_suite.py"}
    if (old_sources.keys() != new_sources.keys()
            or any(old_sources[p] != new_sources[p] for p in old_sources if p not in repair_sources)):
        raise ValueError("Reused control source mismatch outside the audited repair")
    completed = load_json(directory / "summary.json")
    if not all(completed.get(k) is True for k in
               ("inner_training_complete", "final_training_complete", "final_evaluation_complete")):
        raise ValueError("Reference suite is incomplete")
    paths = [directory / "manifest.json", directory / "summary.json"]
    for phase in ("primary", "alternate", "final"):
        seeds = original["seeds"][:1] if phase == "final" else original["seeds"]
        for seed in seeds:
            folder = directory / phase / "B0_control" / f"seed_{seed}"
            summary = load_json(folder / "summary.json")
            checkpoint = folder / "best.pt"
            if (summary["variant"] != "B0_control" or summary["phase"] != phase
                    or summary["seed"] != seed or summary["checkpoint"] != str(checkpoint.relative_to(directory))
                    or summary["checkpoint_sha256"] != sha256(checkpoint)):
                raise ValueError("Reused control artifact mismatch")
            paths.extend([folder / "summary.json", folder / "history.json", checkpoint])
    return {"run_name": run_name, "policy": "B0 only; original signatures retained; never resume old training",
            "files": {str(p.relative_to(directory)): sha256(p) for p in paths}}


def verify_control_reference(reference):
    directory = VARIANT / "runs" / reference["run_name"]
    for path, checksum in reference["files"].items():
        if sha256(directory / path) != checksum:
            raise ValueError(f"Reused control artifact changed: {path}")


def reuse_control(context, variant, seed, phase, stop, rows, config):
    reference = context["manifest"].get("control_reference")
    if reference is None or variant.name != "B0_control":
        return None
    verify_control_reference(reference)
    directory = VARIANT / "runs" / reference["run_name"]
    relative = Path(phase) / variant.name / f"seed_{seed}" / "summary.json"
    if str(relative) not in reference["files"]:
        raise ValueError("No audited control for this phase/seed")
    summary = load_json(directory / relative)
    signature = digest({"context": digest(load_json(directory / "manifest.json")),
                        "variant": asdict(variant), "seed": seed, "fold": phase, "stop": stop, "rows": rows})
    if (summary["signature"] != signature or summary["updates"] != stop
            or summary["config"] != asdict(config) or summary["ablation"] != asdict(variant)):
        raise ValueError("Reused control training signature/configuration mismatch")
    # Reference the immutable original best.pt, never copy/re-sign its optimizer or last.pt.
    summary["checkpoint"] = str(Path("..") / reference["run_name"] / summary["checkpoint"])
    summary["reused_from"] = {"run_name": reference["run_name"], "summary_sha256": reference["files"][str(relative)]}
    freeze_json(context["output"] / relative, summary)
    print(f"{phase}/B0_control/{seed}: reused from {reference['run_name']} (no training)", flush=True)
    return summary


def prepare(run_name="suite_v1", budget=Budget(), seeds=(SEED, SEED + 1, SEED + 2),
            variants=None, device="cuda", dataset=DATASET, reuse_control_from=None):
    """Read-only data audit; only variant_14/runs receives experiment metadata."""
    import onnxruntime
    import torchvision

    budget.validate()
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Exactly three distinct training seeds are required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("RUN_NAME must be a simple directory name")
    output = VARIANT / "runs" / run_name
    if not output.resolve().is_relative_to(VARIANT.resolve() / "runs"):
        raise ValueError("Output must stay inside variant_14/runs")
    grid = experiment_grid()
    names = list(grid) if variants is None else list(variants)
    if len(set(names)) != len(names) or "B0_control" not in names or any(n not in grid for n in names):
        raise ValueError("Use unique registered variants, including B0_control")
    if len(names) < 2:
        raise ValueError("Need control and at least one hypothesis")
    selected = {name: grid[name] for name in names}
    dataset = Path(dataset).resolve()
    rows = read_rows(dataset / "train.csv")
    split = load_json(ARTIFACTS / "splits.json")
    if sha256(dataset / "train.csv") != split["train_csv_sha256"]:
        raise ValueError("Organizer train.csv changed")
    frames = {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg")
              for r in tqdm(rows, desc="Source integrity (no edits)")}
    if frames != split["frame_sha256"]:
        raise ValueError("Original image bytes changed")
    audit_partitions(rows, frames, split["identities"])
    if sha256(STOCK_MODEL) != STOCK_SHA256 or sha256(MODEL) != MVP_SHA256:
        raise ValueError("Unexpected stock or MVP weights")
    reference = load_json(ARTIFACTS / "baseline_metrics.json")
    if (reference["model_sha256"] != MVP_SHA256
            or reference["evaluator"]["sha256"] != sha256(ROOT / "evaluate.py")
            or {k: reference["search"][k] for k in ("k1", "k2", "lambda")} !=
            {"k1": 20, "k2": 3, "lambda": .5}):
        raise ValueError("Frozen MVP reference/evaluator mismatch")
    masks, inner = load_masks(rows, split, frames)
    protocols = {}
    for name, identities in {**{n: p["validation"] for n, p in inner.items()},
                             **{n: split["identities"][n] for n in ("calibration", "validation")}}.items():
        query, gallery = make_protocol(rows, identities, SEED)
        protocols[name] = {"query_ids": [r["image_id"] for r in query],
                           "gallery_ids": [r["image_id"] for r in gallery]}
        if name in split["protocols"] and protocols[name] != split["protocols"][name]:
            raise ValueError("Outer protocol differs from the original MVP")
    base = ExperimentConfig(**load_json(RECIPE)["config"])
    protected = [STOCK_MODEL, MODEL, ARTIFACTS / "splits.json", ARTIFACTS / "baseline_metrics.json",
                 dataset / "train.csv", ROOT / "evaluate.py", ROOT / "ORGANIZER_QA.md"]
    sources = [ROOT / "training" / f"{name}.py" for name in
               ("osnet_ablations", "osnet_ablation_suite", "osnet", "hpo", "pipeline", "preprocessing",
                "stage6", "masked_hpo", "mask_reid_ablation", "audit")]
    sources += [ROOT / "backend" / f"{name}.py" for name in ("core", "evaluate", "scoring", "rerank")]
    manifest = {"version": 1, "budget": asdict(budget), "seeds": list(seeds),
                "variants": {n: asdict(v) for n, v in selected.items()}, "base_recipe": asdict(base),
                "train_csv_sha256": split["train_csv_sha256"], "frames_sha256": digest(frames),
                "outer": split["identities"], "inner": inner, "protocols": protocols,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "stock_sha256": STOCK_SHA256, "mvp_sha256": MVP_SHA256,
                "baseline_sha256": sha256(ARTIFACTS / "baseline_metrics.json"),
                "evaluator_sha256": sha256(ROOT / "evaluate.py"), "recipe_sha256": sha256(RECIPE),
                "organizer_qa_sha256": sha256(ROOT / "ORGANIZER_QA.md"),
                "masks_sha256": sha256(MASK_ROOT / "cache/automatic_masks.json"),
                "detector_plan_sha256": sha256(DETECTOR / "annotation/mask_plan.json"),
                "runtime": {"torch": str(torch.__version__), "torchvision": torchvision.__version__,
                            "numpy": np.__version__, "onnxruntime": onnxruntime.__version__,
                            "python": platform.python_version(), "device": str(device),
                            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                            "deterministic": torch.are_deterministic_algorithms_enabled()},
                "policy": {"annotation_edits": 0, "removed_rows": 0, "outer_selection": False,
                           "initializer": "stock only", "evaluation_seed": SEED,
                           "confirmation": "top 2 non-control + control, three paired seeds",
                           "alternate": "control + winner, three seeds, no second selection",
                           "final_seed": seeds[0], "final_steps": "median primary inner best step",
                           "rerank": {"k1": 20, "k2": 3, "lambda_value": .5}}}
    if reuse_control_from is not None:
        if run_name == reuse_control_from:
            raise ValueError("Repair output must differ from the reference run")
        manifest["control_reference"] = control_reference(reuse_control_from, manifest)
    with run_lock(output):
        freeze_json(output / "manifest.json", manifest)
    return {"output": output, "manifest": manifest, "signature": digest(manifest),
            "rows": rows, "split": split, "masks": masks, "base": base, "budget": budget,
            "seeds": seeds, "variants": selected, "device": torch.device(device), "dataset": dataset,
            "protected": {str(p): sha256(p) for p in protected}}


def verify_protected(context, rehash_frames=False):
    if "control_reference" in context["manifest"]:
        verify_control_reference(context["manifest"]["control_reference"])
    for path, checksum in context["protected"].items():
        if sha256(path) != checksum:
            raise ValueError(f"Protected input changed during this run: {path}")
    for path, checksum in context["manifest"]["source_sha256"].items():
        if sha256(ROOT / path) != checksum:
            raise ValueError(f"Source code changed during this run; restart kernel/new RUN_NAME: {path}")
    if rehash_frames:
        frames = {r["image_id"]: sha256(context["dataset"] / "images" / f"{r['image_id']}.jpg")
                  for r in context["rows"]}
        if digest(frames) != context["manifest"]["frames_sha256"]:
            raise ValueError("Original images changed during this run")


def protocol_rows(context, name):
    by_id = {r["image_id"]: r for r in context["rows"]}
    p = context["manifest"]["protocols"][name]
    return [by_id[i] for i in p["query_ids"]], [by_id[i] for i in p["gallery_ids"]]


@torch.no_grad()
def encode(model, rows, context, variant, masked=False, batch_size=32):
    loader = DataLoader(AblationDataset(rows, variant, context["dataset"], masks=context["masks"],
                                      masked=masked), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    result = {}
    for images, _, ids in loader:
        values = model.embedding(images.to(context["device"])).cpu().numpy()
        vectors = normalize(values)
        result.update(zip(ids, vectors))
    return result


def evaluate_inner(model, context, variant, fold):
    query, gallery = protocol_rows(context, fold)
    embeddings = encode(model, query + gallery, context, variant)
    ranked = ranked_queries(query, gallery, embeddings)
    return metrics(ranked, 0.)  # Refusal at 0 is diagnostic only; no inner threshold search.


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(cpu_copy(payload), temporary)
    temporary.replace(path)


def fit(context, variant, seed, fold="primary", final_steps=None):
    """Checkpoints at evaluation-block boundaries; at most one block is replayed."""
    budget, device = context["budget"], context["device"]
    final = final_steps is not None
    stop = int(final_steps) if final else budget.max_steps
    if not 0 <= stop <= budget.max_steps:
        raise ValueError("Invalid final stop step")
    phase = "final" if final else fold
    directory = context["output"] / phase / variant.name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config = variant.recipe(context["base"], seed)
    identity_ids = (context["split"]["identities"]["train"] if final else
                    context["manifest"]["inner"][fold]["train"])
    labels = {identity: i for i, identity in enumerate(sorted(identity_ids))}
    rows = [{**r, "label": labels[r["vehicle_id"]]} for r in context["rows"] if r["vehicle_id"] in labels]
    reused = reuse_control(context, variant, seed, phase, stop, rows, config)
    if reused is not None:
        return reused
    signature = digest({"context": context["signature"], "variant": asdict(variant),
                        "seed": seed, "fold": phase, "stop": stop, "rows": rows})
    last_path = directory / "last.pt"
    if not last_path.exists() and (directory / "summary.json").exists():
        raise ValueError("Missing authoritative last.pt; restore it or use a new RUN_NAME")
    set_seed(seed)
    model = initialize(len(labels), config, variant, device)
    optimizer = optimizer_for(model, config)
    teacher = ema_teacher(model) if variant.ema else None
    bank = MemoryBank(variant.memory_size)
    previous = torch.load(last_path, map_location="cpu", weights_only=True) if last_path.exists() else None
    if previous is not None and previous["signature"] != signature:
        raise ValueError("Checkpoint/configuration mismatch")
    start = previous["step"] if previous else 0
    positive_features = None
    if variant.positive_sampling == "balanced" and start < stop:
        # Always computed before loading locally trained state: no holdout, no MVP initializer.
        positive_features = encode(model, rows, context, variant)
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        bank.load_state_dict(previous["bank"], device)
        if teacher is not None:
            teacher.load_state_dict(previous["teacher"])
        history, best, seen, elapsed = previous["history"], previous["best"], set(previous["seen"]), previous["elapsed"]
    else:
        initial = None if final else evaluate_inner(model, context, variant, fold)
        history, seen, elapsed = [], set(), 0.
        best = {"step": 0, "map": initial["mAP_at_10"] if initial else None,
                "metrics": initial, "state": cpu_copy(model.state_dict())}
    dataset = AblationDataset(rows, variant, context["dataset"], augment=True, masks=context["masks"])
    row_indices = {row["image_id"]: i for i, row in enumerate(rows)}
    for end in budget.boundaries(stop):
        if end <= start:
            continue
        set_seed(seed + start)  # Replays augmentation + sampler + stochastic model at block boundary.
        count = end - start
        sampler = (BalancedPositiveSampler(rows, config, count, positive_features)
                   if positive_features is not None else StepPKBatchSampler(rows, config, count))
        sampler.set_epoch(start)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        totals, began = {}, time.perf_counter()
        for offset, (clean, robust, target, image_ids) in enumerate(loader):
            step = start + offset
            set_step_learning_rates(optimizer, config, budget, step)
            ids = torch.tensor([row_indices[i] for i in image_ids], device=device)
            target = target.to(device)
            values, raw = losses(model, clean.to(device), robust.to(device), target, ids,
                                 config, variant, bank, step, teacher)
            if not torch.isfinite(values["loss"]):
                raise FloatingPointError(f"Non-finite loss: {variant.name}, step={step}")
            optimizer.zero_grad(set_to_none=True)
            values["loss"].backward()
            # Measure, but do not silently introduce gradient clipping to the control.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            bank.push(raw, target, ids)
            if teacher is not None:
                update_teacher(teacher, model)
            seen.update(image_ids)
            for name, value in {**values, "gradient_norm": norm}.items():
                totals[name] = totals.get(name, 0.) + float(value.detach())
        validation = None if final else evaluate_inner(model, context, variant, fold)
        seconds = time.perf_counter() - began
        elapsed += seconds
        if final or validation["mAP_at_10"] > best["map"]:
            best = {"step": end, "map": validation["mAP_at_10"] if validation else None,
                    "metrics": validation, "state": cpu_copy(model.state_dict())}
        history.append({"step": end, "samples_seen": end * variant.p * variant.k,
                        "unique_train_images": len(seen), "train": {k: v / count for k, v in totals.items()},
                        "validation": validation, "block_seconds": seconds, "elapsed_seconds": elapsed})
        payload = {"signature": signature, "step": end, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "teacher": teacher.state_dict() if teacher else None,
                   "bank": bank.state_dict(), "best": best, "history": history,
                   "seen": sorted(seen), "elapsed": elapsed}
        save_checkpoint(last_path, payload)  # Authoritative commit, before derived JSON/best.pt.
        write_json(directory / "history.json", history)
        score = "final, no evaluation" if final else f"inner mAP={validation['mAP_at_10']:.4f}, best={best['map']:.4f}"
        print(f"{phase}/{variant.name}/{seed}: {end}/{stop} | {score} | "
              f"elapsed {format_duration(elapsed)} | ETA {format_duration(elapsed / end * (stop - end))}", flush=True)
        start = end
    if not last_path.exists():  # Valid when stock (step zero) is the selected checkpoint.
        save_checkpoint(last_path, {"signature": signature, "step": 0, "model": model.state_dict(),
                                   "optimizer": optimizer.state_dict(), "teacher": teacher.state_dict() if teacher else None,
                                   "bank": bank.state_dict(), "best": best, "history": history,
                                   "seen": [], "elapsed": 0.})
    model.load_state_dict(best["state"])
    best_path = directory / "best.pt"
    save_checkpoint(best_path, {"signature": signature, "model": best["state"]})
    summary = {"variant": variant.name, "seed": seed, "phase": phase, "signature": signature,
               "best_step": best["step"], "inner_map": best["map"], "inner_metrics": best["metrics"],
               "updates": stop, "samples_seen": stop * variant.p * variant.k, "unique_images": len(seen),
               "train_images": len(rows), "train_identities": len(labels), "elapsed_seconds": elapsed,
               "checkpoint": str(best_path.relative_to(context["output"])), "checkpoint_sha256": sha256(best_path),
               "config": asdict(config), "ablation": asdict(variant)}
    write_json(directory / "summary.json", summary)
    write_json(directory / "history.json", history)
    del model, optimizer, teacher, bank, previous
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def select_finalists(screening):
    alternatives = sorted((s for s in screening if s["variant"] != "B0_control"),
                          key=lambda s: (-s["inner_map"], s["variant"]))
    return ["B0_control"] + [s["variant"] for s in alternatives[:2]]


def select_winner(confirmed, seeds):
    names = sorted({s["variant"] for s in confirmed})
    aggregate = {}
    for name in names:
        runs = [s for s in confirmed if s["variant"] == name]
        if sorted(s["seed"] for s in runs) != sorted(seeds):
            raise ValueError("Missing or duplicate confirmation seeds")
        aggregate[name] = {"mean": float(np.mean([s["inner_map"] for s in runs])),
                           "std": float(np.std([s["inner_map"] for s in runs], ddof=1)),
                           "final_steps": int(np.median([s["best_step"] for s in runs]))}
    # Ties favor the control, not a more complicated representation.
    winner = max(names, key=lambda n: (aggregate[n]["mean"], n == "B0_control", n))
    return winner, aggregate


def load_final_model(context, summary):
    variant = context["variants"][summary["variant"]]
    path = context["output"] / summary["checkpoint"]
    if sha256(path) != summary["checkpoint_sha256"]:
        raise ValueError("Final checkpoint checksum mismatch")
    model = initialize(summary["train_identities"], variant.recipe(context["base"], summary["seed"]),
                       variant, context["device"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != summary["signature"]:
        raise ValueError("Final checkpoint signature mismatch")
    model.load_state_dict(payload["model"])
    return model.eval(), variant


def ranking_pair(query, gallery, embeddings):
    raw = ranked_queries(query, gallery, embeddings)
    reranked, confidence = rerank_protocol(query, gallery, embeddings, k1=20, k2=3, lambda_value=.5)
    return {"raw": (raw, None), "reranked": (reranked, confidence)}


def evaluate_final(context, summary):
    """Calibrate once, freeze thresholds, then report original + masked 2x2 diagnostics."""
    model, variant = load_final_model(context, summary)
    directory = context["output"] / "final" / variant.name / f"seed_{summary['seed']}"
    calibration_query, calibration_gallery = protocol_rows(context, "calibration")
    original = encode(model, calibration_query + calibration_gallery, context, variant)
    thresholds = {name: calibrate(ranked, confidence) for name, (ranked, confidence) in
                  ranking_pair(calibration_query, calibration_gallery, original).items()}
    freeze_json(directory / "thresholds.json", {"checkpoint_sha256": summary["checkpoint_sha256"],
                                               "thresholds": thresholds,
                                               "selection": "original calibration only; no mask tuning"})
    query, gallery = protocol_rows(context, "validation")
    clean = encode(model, query + gallery, context, variant)
    masked = encode(model, query + gallery, context, variant, masked=True)
    report = {"checkpoint_sha256": summary["checkpoint_sha256"], "thresholds": thresholds, "conditions": {}}
    for name, q_mask, g_mask in (("original", False, False), ("masked_query", True, False),
                                  ("masked_gallery", False, True), ("masked_both", True, True)):
        embeddings = {r["image_id"]: (masked if q_mask else clean)[r["image_id"]] for r in query}
        embeddings.update({r["image_id"]: (masked if g_mask else clean)[r["image_id"]] for r in gallery})
        condition = {}
        for method, (ranked, confidence) in ranking_pair(query, gallery, embeddings).items():
            condition[method] = metrics(ranked, thresholds[method], confidence)
            if name == "original":
                import evaluate as official
                per_query = {}
                for row in query:
                    qid = row["image_id"]
                    single = official.ranking_metrics(ranked.query.loc[[qid]], ranked.gallery, ranked.predictions)
                    if single["n_scored"]:
                        per_query[qid] = {"vehicle_id": row["vehicle_id"], "ap": single["mAP@10"]}
                write_json(directory / f"per_query_{method}.json", per_query)
        report["conditions"][name] = condition
    write_json(directory / "evaluation.json", report)
    del model
    return report


def export_final(context, summary):
    """Experimental ONNX only. Do not install it into backend/models or change the API."""
    import onnxruntime as ort

    model, variant = load_final_model(context, summary)
    encoder = InferenceEncoder(model).eval().cpu()
    directory = context["output"] / "final" / variant.name / f"seed_{summary['seed']}"
    path = directory / "encoder.onnx"
    temporary = directory / "encoder.tmp.onnx"
    torch.onnx.export(encoder, torch.zeros(1, 3, variant.size, variant.size), temporary,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(temporary), options, providers=["CPUExecutionProvider"])
    query, gallery = protocol_rows(context, "validation")
    dataset = AblationDataset(query + gallery, variant, context["dataset"], masks=context["masks"])
    differences = {}
    for batch in (1, 3, 8):
        images = torch.stack([dataset[i][0] for i in range(batch)])
        with torch.no_grad():
            expected = encoder(images).numpy()
        actual = session.run(["output"], {"input": images.numpy()})[0]
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError("Invalid exported embeddings")
        error = float(np.abs(expected - actual).max())
        if error > 2e-4 or not np.allclose(np.linalg.norm(actual, axis=1), 1., atol=1e-5):
            raise ValueError(f"ONNX parity failed for batch {batch}: {error}")
        differences[str(batch)] = error
    sample = torch.stack([dataset[0][0]]).numpy()
    for _ in range(3):
        session.run(None, {"input": sample})
    latencies = []
    for _ in range(20):
        began = time.perf_counter()
        session.run(None, {"input": sample})
        latencies.append((time.perf_counter() - began) * 1000)
    temporary.replace(path)
    report = {"onnx_sha256": sha256(path), "bytes": path.stat().st_size,
              "dimension": model.dimension, "size": variant.size, "resize": variant.resize,
              "normalization": "RGB ImageNet then L2", "batch_parity_max_abs": differences,
              "cpu_forward_ms": {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))},
              "performance_note": "CPU forward only; NOT official A5000 full extract benchmark",
              "promoted": False}
    write_json(directory / "export.json", report)
    return report


def paired_bootstrap(before, after, seed=SEED, repeats=2000):
    if set(before) != set(after):
        raise ValueError("Unpaired query protocols")
    grouped = {}
    for key, item in before.items():
        if item["vehicle_id"] != after[key]["vehicle_id"]:
            raise ValueError("Identity mismatch")
        grouped.setdefault(item["vehicle_id"], []).append(after[key]["ap"] - item["ap"])
    groups = list(grouped.values())
    rng = np.random.default_rng(seed)
    draws = [np.mean([value for i in rng.integers(0, len(groups), len(groups)) for value in groups[i]])
             for _ in range(repeats)]
    return {"mean_delta": float(np.mean([x for g in groups for x in g])),
            "identity_bootstrap_95pct": np.quantile(draws, [.025, .975]).tolist(),
            "limitation": "Conditional on fixed gallery; not an independent test or multiple-search correction"}


def write_report(context, screening, confirmed, alternate, selection, outer):
    lines = ["# OSNet variant 14 — результаты", "", "Исходные строки/bbox/validation не изменены. MVP не заменён.",
             "Selection: primary inner raw mAP@10; outer используется только после freeze выбора."]
    if reference := context.get("manifest", {}).get("control_reference"):
        lines += ["", f"B0 повторно использован из {reference['run_name']} без переобучения; provenance в manifest.json.",
                  "Это отдельное сравнение исправленных вариантов с B0, не новый выбор среди всех 22 вариантов."]
    lines += ["", "## Screening (один seed, не доказательство)", "",
             "| Вариант | inner mAP@10 | best step | samples seen |", "|---|---:|---:|---:|"]
    lines += [f"| {s['variant']} | {s['inner_map']:.5f} | {s['best_step']} | {s['samples_seen']} |" for s in screening]
    lines += ["", "## Confirmation: три seed", "", "| Вариант | mean | sample std | final steps |",
              "|---|---:|---:|---:|"]
    lines += [f"| {n} | {a['mean']:.5f} | {a['std']:.5f} | {a['final_steps']} |"
              for n, a in selection["aggregate"].items()]
    lines += ["", f"Замороженный выбор: **{selection['winner']}**. Альтернативный split не меняет выбор.",
              "", "## Alternate frame-grouped split", "", "| Вариант | seed | inner mAP@10 |",
              "|---|---:|---:|"]
    lines += [f"| {s['variant']} | {s['seed']} | {s['inner_map']:.5f} |" for s in alternate]
    lines += ["", "## Outer validation — только отчёт", "",
              "| Вариант / input | raw mAP@10 | reranked mAP@10 | F1 | TNR | 0.7F1+0.3TNR |",
              "|---|---:|---:|---:|---:|---:|"]
    if outer:
        reference = load_json(ARTIFACTS / "baseline_metrics.json")
        raw, rerank = reference["raw_baseline"]["validation"], reference["validation"]
        lines.append(f"| MVP / frozen reference | {raw['mAP_at_10']:.5f} | {rerank['mAP_at_10']:.5f} | "
                     f"{rerank['candidate_F1']:.5f} | {rerank['TNR']:.5f} | {rerank['candidate_score']:.5f} |")
    for name, report in outer.items():
        for condition, results in report["conditions"].items():
            raw, rerank = results["raw"], results["reranked"]
            lines.append(f"| {name} / {condition} | {raw['mAP_at_10']:.5f} | {rerank['mAP_at_10']:.5f} | "
                         f"{rerank['candidate_F1']:.5f} | {rerank['TNR']:.5f} | {rerank['candidate_score']:.5f} |")
    lines += ["", "Validation уже использовалась в исследованиях: это development, не независимый финальный тест.",
              "MVP reference взят из проверенного baseline_metrics.json, не из нового обучения; hashes зафиксированы.",
              "Mask-условия — дополнительные diagnostics; основная validation остаётся original.",
              "CPU forward timing не заменяет официальный A5000 extract benchmark. Автопродвижения нет."]
    # Report generation is program output, not a source-file edit.
    (context["output"] / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(context, final_evaluation=True):
    """Notebook Run All: screening -> paired seeds -> alternate -> frozen final report."""
    device = context["device"]
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Choose MPS/CPU explicitly; CPU full suite is slow")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    with run_lock(context["output"]):
        verify_protected(context)
        screening = [fit(context, v, context["seeds"][0]) for v in context["variants"].values()]
        write_json(context["output"] / "screening.json", screening)
        finalists = select_finalists(screening)
        confirmed = [fit(context, context["variants"][name], seed)
                     for name in finalists for seed in context["seeds"]]
        winner, aggregate = select_winner(confirmed, context["seeds"])
        selection = {"signature": context["signature"], "winner": winner, "aggregate": aggregate,
                     "basis": "primary inner raw mAP only", "final_seed": context["seeds"][0]}
        freeze_json(context["output"] / "selection.json", selection)
        write_json(context["output"] / "confirmation.json", confirmed)
        names = list(dict.fromkeys(["B0_control", winner]))
        alternate = [fit(context, context["variants"][name], seed, fold="alternate")
                     for name in names for seed in context["seeds"]]
        write_json(context["output"] / "alternate.json", alternate)
        outer = {}
        if final_evaluation:
            # Both models are refit on the entire ORIGINAL outer train. No inner data removal.
            final_runs = [fit(context, context["variants"][name], context["seeds"][0],
                              final_steps=aggregate[name]["final_steps"]) for name in names]
            freeze_json(context["output"] / "final_selection.json", final_runs)
            for summary in final_runs:
                outer[summary["variant"]] = evaluate_final(context, summary)
                export_final(context, summary)
            if winner != "B0_control":
                paired = {}
                for method in ("raw", "reranked"):
                    reports = [load_json(context["output"] / "final" / name /
                                         f"seed_{context['seeds'][0]}" / f"per_query_{method}.json") for name in names]
                    paired[method] = paired_bootstrap(*reports)
                write_json(context["output"] / "paired_bootstrap.json", paired)
        verify_protected(context, rehash_frames=True)
        write_report(context, screening, confirmed, alternate, selection, outer)
        result = {"selection": selection, "outer": outer, "promoted": False,
                  "inner_training_complete": True, "final_training_complete": final_evaluation,
                  "final_evaluation_complete": final_evaluation}
        write_json(context["output"] / "summary.json", result)
        return result
