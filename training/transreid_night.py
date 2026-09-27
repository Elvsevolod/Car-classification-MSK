"""v36 staged night search: 24 fixed trials, four continuations, inner-only."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import gc
import importlib.metadata
from itertools import product
import math
from pathlib import Path
import platform
import re
import shutil
import time

import numpy as np
import torch

from training import transreid_model as vision, nive_mixed as base, nive_system as previous
from training.hpo import ExperimentConfig
from training.overnight_training import capture_rng, restore_rng
from training.stage6 import StepPKBatchSampler

VARIANT = vision.VARIANT


def trial_grid(settings):
    g = settings["grid"]
    result = []
    for i, (architecture, lr, wd, metric) in enumerate(product(g["architectures"], g["encoder_lr"], g["weight_decay"], g["metric_loss"])):
        if architecture not in {"global", "jpm"} or metric not in {"soft_triplet", "supcon"} or lr <= 0 or wd < 0:
            raise ValueError("Invalid trial specification")
        result.append({"id": f"T{i+1:02d}_{architecture}_{metric}", "architecture": architecture,
                       "encoder_lr": lr, "weight_decay": wd, "metric_loss": metric})
    return result


def check_inputs(c):
    m = c["manifest"]
    if base.digest(m) != c["signature"]:
        raise ValueError("Frozen search manifest changed")
    base.verify_files(m["protected"]); base.verify_files(m["source_sha256"])
    vision.verify_weights(m["weight_path"])


def disk_guard(output, settings):
    used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) if output.exists() else 0
    needed = max(2*1024**3, settings["estimated_run_disk_gib"]*1024**3-used)
    free = shutil.disk_usage(output if output.exists() else output.parent.parent).free
    if free < needed:
        raise RuntimeError(f"Need about {needed/1024**3:.1f} GiB free for remaining checkpoints; have {free/1024**3:.1f}. "
                           "Free space yourself; historical artifacts are never deleted.")


def prepare(run_name="night_v1", device="mps"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple RUN_NAME")
    weight_path = vision.verify_weights()
    device = vision.device_for(device)
    config_path = VARIANT / "configs/night_v1.json"
    settings = base.old.load_json(config_path)
    if (settings["SIE_CAMERA"] or settings["SIE_VIEW"] or settings["external_train"] or settings["threshold_fit"]
            or settings["original_outer_evaluation"] or settings["promoted"] or settings["wall_time_limit"] is not None
            or settings["image_size"] != 256 or settings["graph"] != base.policy.POLICIES["less_graph"]
            or settings["new_encoder_weight"] != .1 or settings["finalists_per_architecture"] != 2):
        raise ValueError("Search may not change protected protocol, graph, fusion or metadata inputs")
    if (settings["screen_steps"] >= settings["final_steps"] or
        settings["screen_steps"] not in settings["checkpoints"] or
        settings["final_steps"] != max(settings["checkpoints"]) or
        any(step % settings["save_interval"] for step in settings["checkpoints"])):
        raise ValueError("Invalid fixed search horizons/checkpoint boundaries")
    trials = trial_grid(settings)
    source = base.ROOT / settings["source_run"]
    base.verify_files({str(source / "manifest.json"): settings["source_manifest_sha256"],
                       str(source / "results.json"): settings["source_results_sha256"]})
    m34, r34 = (base.old.load_json(source / name) for name in ("manifest.json", "results.json"))
    if r34["status"] != "complete" or r34["signature"] != base.digest(m34):
        raise ValueError("Need completed v34 reference")
    inner_path = Path(m34["source_directory"]) / "manifest.json"
    inner = base.old.load_json(inner_path)
    if base.digest(inner) != m34["source_signature"]:
        raise ValueError("Inner provenance changed")
    rows = base.read_rows(base.DATASET / "train.csv")
    split = base.old.load_json(base.ARTIFACTS / "splits.json")
    _, _, allowed, _ = base.parent_provenance(inner["plan"], rows, split)
    held = set(inner["inner"]["validation"])
    if allowed & held or allowed | held != set(split["identities"]["train"]):
        raise ValueError("Training/inner/outer identity leakage")
    labels = {identity: i for i, identity in enumerate(sorted(allowed))}
    target = [{**r, "label": labels[r["vehicle_id"]]} for r in rows if r["vehicle_id"] in allowed]
    score_context = {"manifest": inner, "rows": rows}
    development = base.development_rows(score_context)
    paths = vision.image_paths(base.DATASET, target+development)
    reference = inner_path.parent / "evaluation/N_ref"
    references = {str(reference / name): settings[key] for name, key in (
        ("features.npy", "reference_features_sha256"), ("order.json", "reference_order_sha256"),
        ("metrics.json", "reference_metrics_sha256"))}
    base.verify_files(references)
    if base.old.load_json(reference / "order.json") != [r["image_id"] for r in development]:
        raise ValueError("R1 cached row order changed")
    protected = {p: h for p, h in m34["protected"].items() if not Path(p).is_relative_to(base.ROOT / "NiVe1303")}
    protected.update(references)
    protected.update({str(p): base.sha256(p) for p in source.rglob("*") if p.is_file()})
    if any(str(path) not in protected for path in paths.values()):
        raise ValueError("An input image is outside the inherited byte protection")
    sources = {**m34["source_sha256"], **{str(p): base.sha256(p) for p in (
        Path(__file__).resolve(), Path(vision.__file__).resolve(), config_path,
        base.ROOT / "training/transreid_vendor/vit_pytorch.py", base.ROOT / "training/transreid_vendor/LICENSE",
        base.ROOT / "training/transreid_vendor/DEIT_LICENSE")}}
    runtime = {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "numpy", "pillow")}
    runtime.update(device=str(device), python=platform.python_version(), platform=platform.platform(),
                   torch_threads=torch.get_num_threads(), dtype="float32", compile=False, amp=False)
    sampler_config = ExperimentConfig(identities_per_batch=settings["identities_per_batch"],
                                     images_per_identity=settings["images_per_identity"], seed=settings["seed"])
    schedule = list(StepPKBatchSampler(target, sampler_config, settings["final_steps"]))
    manifest = {"version": 36, "settings": settings, "trials": trials, "runtime": runtime,
        "weight_path": str(weight_path), "weight_sha256": vision.WEIGHT_SHA256, "weight_url": vision.WEIGHT_URL,
        "upstream_revision": vision.REVISION, "preprocessing": vision.PREPROCESS,
        "reference_directory": str(reference), "protected": protected, "source_sha256": sources,
        "train_ids": sorted(allowed), "holdout_ids": sorted(held), "draws": inner["draws"],
        "image_order": [r["image_id"] for r in development], "schedule_sha256": base.digest(schedule),
        "train_rows_sha256": base.digest(target), "train_images": len(target),
        "selection_scope": "adaptive inner development, not independent validation or hidden test",
        "promoted": False, "original_outer_evaluation": False, "threshold_fit": False}
    output = VARIANT / "runs" / run_name
    c = {"manifest": manifest, "settings": settings, "trials": trials, "signature": base.digest(manifest),
         "output": output, "device": device, "target": target, "rows": development, "paths": paths,
         "score_context": score_context, "schedule": schedule}
    check_inputs(c); disk_guard(output, settings)
    with base.old.run_lock(output):
        base.old.freeze_json(output / "manifest.json", manifest)
    print(f"PREFLIGHT: {len(trials)} trials, {len(allowed)} train-ID / {len(held)} holdout-ID, "
          f"{len(development)} evaluation images; outer disabled", flush=True)
    return c


def new_model(c, trial):
    base.set_seed(c["settings"]["seed"])
    return vision.ReIDModel(len({r["label"] for r in c["target"]}), trial["architecture"],
        weight_path=c["manifest"]["weight_path"], drop_path=c["settings"]["drop_path"]).to(c["device"])


def optimizer_for(model, trial, settings):
    head = [p for name, p in model.named_parameters() if name.startswith(("necks.", "heads.")) and p.requires_grad]
    ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return torch.optim.AdamW([{"params": backbone, "base_lr": trial["encoder_lr"]},
        {"params": head, "base_lr": trial["encoder_lr"]*settings["head_lr_multiplier"]}],
        lr=trial["encoder_lr"], weight_decay=trial["weight_decay"], foreach=False)


def set_lr(optimizer, settings, step):
    warm, total = settings["warmup_steps"], settings["final_steps"]
    fraction = (step+1)/warm if step < warm else (
        settings["min_lr_ratio"]+(1-settings["min_lr_ratio"])*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1))))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"]*fraction
    return [group["lr"] for group in optimizer.param_groups]


def runtime_smoke(c):
    """Four disposable updates on the requested device, before the long queue."""
    def action(directory):
        reports = []
        dataset = vision.Images(c["target"], c["paths"], train=True)
        for architecture in dict.fromkeys(t["architecture"] for t in c["trials"]):
            trial = next(t for t in c["trials"] if t["architecture"] == architecture)
            model = new_model(c, trial)
            optimizer = optimizer_for(model, trial, c["settings"])
            batch = [dataset[i] for i in c["schedule"][0]]
            images = torch.stack([v[0] for v in batch]).to(c["device"])
            labels = torch.tensor([v[1] for v in batch], device=c["device"])
            for metric in c["settings"]["grid"]["metric_loss"]:
                began = time.perf_counter()
                model.train(); optimizer.zero_grad(set_to_none=True)
                values = vision.losses(model, images, labels, {**trial, "metric_loss": metric}, c["settings"])
                if not all(torch.isfinite(v) for v in values.values()):
                    raise FloatingPointError("Requested-device smoke has nonfinite loss")
                values["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["settings"]["gradient_clip"])
                if not torch.isfinite(norm): raise FloatingPointError("Requested-device smoke has nonfinite gradients")
                optimizer.step()
                # A CPU transfer synchronizes the just-completed device work.
                loss = float(values["loss"].detach().cpu())
                seconds = time.perf_counter()-began
                reports.append({"architecture": architecture, "metric": metric, "loss": loss, "seconds": seconds})
                print(f"DEVICE SMOKE {architecture}/{metric}: batch {len(batch)}, loss={loss:.4f}, {seconds:.2f}s", flush=True)
            model.eval()
            with torch.inference_mode():
                together = model(images[:2]).cpu()
                separate = torch.cat([model(x[None]).cpu() for x in images[:2]])
            torch.testing.assert_close(together, separate, atol=2e-5, rtol=0)
            del model, optimizer, images, labels, values, norm
            gc.collect()
            if c["device"].type == "mps": torch.mps.empty_cache()
            if c["device"].type == "cuda": torch.cuda.empty_cache()
        return {"status": "passed", "device": str(c["device"]), "updates": reports,
                "discarded_smoke_models": True, "timing_is_not_a_benchmark_or_eta": True}
    return previous.task(c, "runtime_smoke", action)


def save_boundary(c, trial, model, optimizer, step, history, checkpoints, elapsed):
    directory = c["output"] / "training" / trial["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if step in c["settings"]["checkpoints"]:
        path = directory / f"step_{step:05d}.pt"
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if path.exists():
            prior = torch.load(path, map_location="cpu", weights_only=True)
            if prior["signature"] != c["signature"] or prior["trial"] != trial or prior["step"] != step or any(
                    not torch.equal(v, prior["model"][k]) for k, v in state.items()):
                raise ValueError("Saved selectable checkpoint changed")
        else:
            base.old.save_checkpoint(path, {"signature": c["signature"], "trial": trial, "step": step, "model": state})
        checkpoints[str(step)] = {"path": str(path), "sha256": base.sha256(path)}
    payload = {"signature": c["signature"], "trial": trial, "step": step, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "rng": capture_rng(), "history": history,
               "checkpoints": checkpoints, "elapsed_seconds": elapsed}
    pointer = directory / "resume.json"
    prior_name = base.old.load_json(pointer)["path"] if pointer.exists() else None
    if prior_name is not None and prior_name not in {"resume_0.pt", "resume_1.pt"}:
        raise ValueError("Invalid local resume slot")
    next_name = "resume_1.pt" if prior_name == "resume_0.pt" else "resume_0.pt"
    destination = directory / next_name
    base.old.save_checkpoint(destination, payload)
    base.write_json(pointer, {"path": next_name, "sha256": base.sha256(destination)})
    # Only our superseded optimizer scratch slot is removed, AFTER pointer commit.
    # All selectable weights and all historical experiments are preserved.
    if prior_name is not None:
        (directory / prior_name).unlink(missing_ok=True)
    base.write_json(directory / "history.json", history)


def train_until(c, trial, stop_step):
    settings = c["settings"]
    if stop_step not in (settings["screen_steps"], settings["final_steps"]):
        raise ValueError("Only the two registered training horizons are allowed")
    directory = c["output"] / "training" / trial["id"]
    model = new_model(c, trial)
    optimizer = optimizer_for(model, trial, settings)
    start, history, checkpoints, elapsed = 0, [], {}, 0.
    if (directory / "resume.json").exists():
        saved = torch.load(base.resume_path(directory), map_location="cpu", weights_only=True)
        if saved["signature"] != c["signature"] or saved["trial"] != trial:
            raise ValueError("Resume belongs to another trial/protocol")
        base.verify_files({v["path"]: v["sha256"] for v in saved["checkpoints"].values()})
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, history, checkpoints, elapsed = saved["step"], saved["history"], saved["checkpoints"], saved["elapsed_seconds"]
        del saved
    if start >= stop_step:
        print(f"{trial['id']}: verified saved {start} updates (not new training)", flush=True)
        return {"step": start, "checkpoints": checkpoints, "elapsed_seconds": elapsed}
    dataset = vision.Images(c["target"], c["paths"], train=True)
    for begin in range(start, stop_step, settings["save_interval"]):
        end = min(begin+settings["save_interval"], stop_step)
        started, logs = time.perf_counter(), []
        model.train()
        for step in range(begin, end):
            # Same sampled IDs and augmentation seed across all trials. No DataLoader workers.
            base.set_seed(settings["seed"]+100001+step)
            batch = [dataset[i] for i in c["schedule"][step]]
            images = torch.stack([v[0] for v in batch]).to(c["device"])
            labels = torch.tensor([v[1] for v in batch], device=c["device"])
            lr = set_lr(optimizer, settings, step)
            optimizer.zero_grad(set_to_none=True)
            values = vision.losses(model, images, labels, trial, settings)
            if not all(torch.isfinite(v) for v in values.values()):
                raise FloatingPointError(f"Nonfinite loss in {trial['id']} at update {step+1}")
            values["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=False)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient in {trial['id']} at update {step+1}")
            optimizer.step()
            logs.append({"step": step+1, **{k: float(v.detach()) for k, v in values.items()}, "grad_norm": float(norm), "lr": lr})
            if (step+1) % settings["log_interval"] == 0:
                spent = elapsed+time.perf_counter()-started
                print(f"TRAIN {trial['id']} | {step+1}/{stop_step} (horizon {settings['final_steps']}) | "
                      f"loss={logs[-1]['loss']:.4f} | {base.format_duration(spent)}", flush=True)
        elapsed += time.perf_counter()-started
        history.extend(logs)
        save_boundary(c, trial, model, optimizer, end, history, checkpoints, elapsed)
    result = {"step": stop_step, "checkpoints": checkpoints, "elapsed_seconds": elapsed}
    del optimizer, model
    gc.collect()
    if c["device"].type == "mps": torch.mps.empty_cache()
    if c["device"].type == "cuda": torch.cuda.empty_cache()
    return result


def mix(r1, values):
    vision.validate_block(r1, 512); vision.validate_block(values, values.shape[1])
    if len(r1) != len(values): raise ValueError("Mixture order/count mismatch")
    return base.normalize(np.concatenate([r1*np.float32(np.sqrt(.9)), values*np.float32(np.sqrt(.1))], axis=1))


def evaluate(c, trial, entry=None, step=0):
    name = f"evaluate_{trial['id']}_{step:05d}" if step else f"stock_{trial['architecture']}"
    def action(directory):
        model = new_model(c, trial)
        if entry:
            base.verify_files({entry["path"]: entry["sha256"]})
            state = torch.load(entry["path"], map_location="cpu", weights_only=True)
            if state["signature"] != c["signature"] or state["trial"] != trial or state["step"] != step:
                raise ValueError("Evaluation checkpoint provenance mismatch")
            model.load_state_dict(state["model"], strict=True)
            del state
        began = time.perf_counter()
        values = vision.encode(model, c["rows"], c["paths"], c["device"], c["settings"]["eval_batch_size"])
        elapsed = time.perf_counter()-began
        np.save(directory / "features.npy", values)
        base.old.freeze_json(directory / "order.json", [r["image_id"] for r in c["rows"]])
        r1 = np.load(Path(c["manifest"]["reference_directory"]) / "features.npy", allow_pickle=False)
        single = base.score_features(c["score_context"], values, c["rows"])
        fused = base.score_features(c["score_context"], mix(r1, values), c["rows"])
        print(f"SCORE {name}: single={single['mean_map']:.6f} | R1 90/10={fused['mean_map']:.6f}", flush=True)
        del model; gc.collect()
        return {"trial_id": trial["id"] if step else f"stock_{trial['architecture']}", "step": step,
                "architecture": trial["architecture"], "single": single, "mixture": fused,
                "features": str(directory / "features.npy"), "extract_seconds": elapsed,
                "checkpoint": entry, "original_outer_evaluation": False}
    return previous.task(c, name, action)


def choose_finalists(trials, reports, per_architecture=2):
    chosen = []
    for architecture in dict.fromkeys(t["architecture"] for t in trials):
        entries = [t for t in trials if t["architecture"] == architecture and t["id"] in reports]
        selected = []
        # Preserve a strong single model as well as a potentially useful complement.
        for mode in ("single", "mixture"):
            if entries:
                winner = max(entries, key=lambda t: reports[t["id"]][mode]["mean_map"])
                if winner not in selected: selected.append(winner)
        for trial in sorted(entries, key=lambda t: -reports[t["id"]]["mixture"]["mean_map"]):
            if len(selected) >= per_architecture: break
            if trial not in selected: selected.append(trial)
        chosen.extend(t["id"] for t in selected[:per_architecture])
    return chosen


def cached_trial_scores(c, trial, horizon):
    """Keep completed checkpoints even after a later evaluation failure/restart."""
    def disappeared(directory):
        raise ValueError("A completed evaluation disappeared during resume")
    results = []
    for step in c["settings"]["checkpoints"]:
        name = f"evaluate_{trial['id']}_{step:05d}"
        if step <= horizon and (c["output"] / "tasks" / name / "complete.json").exists():
            results.append(previous.task(c, name, disappeared))
    return results


def final_probe(c, record):
    """Fresh image batching/permutation, then exact top10 against the full static gallery."""
    def action(directory):
        trial = next(t for t in c["trials"] if (t["id"] == record["trial_id"] if record["step"] else
                                                t["architecture"] == record["architecture"]))
        model = new_model(c, trial)
        if record["checkpoint"]:
            entry = record["checkpoint"]
            base.verify_files({entry["path"]: entry["sha256"]})
            saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
            if saved["signature"] != c["signature"] or saved["trial"] != trial or saved["step"] != record["step"]:
                raise ValueError("Probe checkpoint changed")
            model.load_state_dict(saved["model"], strict=True)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        values = np.load(record["features"], allow_pickle=False)
        r1 = np.load(Path(c["manifest"]["reference_directory"]) / "features.npy", allow_pickle=False)
        positions = {row["image_id"]: i for i, row in enumerate(c["rows"])}
        draw = next(iter(c["manifest"]["draws"].values()))
        qi, gi = ([positions[i] for i in draw[k]] for k in ("query_ids", "gallery_ids"))
        qi = qi[:32]
        all_vectors = {"single": values, "mixture": mix(r1, values)}
        expected = {k: base.policy.rank_vectors(v[qi], v[gi], "less_graph") for k, v in all_vectors.items()}
        cases = [(np.arange(len(qi)), size) for size in (1, 8, 16, 32)]
        cases += [(np.arange(len(qi))[::-1], 16), (np.array([0]), 1)]
        maximum = 0.
        for indices, size in cases:
            selected = np.asarray(qi)[indices]
            actual = vision.encode(model, [c["rows"][i] for i in selected], c["paths"], c["device"], size, False)
            error = float(abs(actual-values[selected]).max()); maximum = max(maximum, error)
            if not np.allclose(actual, values[selected], atol=2e-5, rtol=0):
                raise ValueError("Image batch/permutation drift exceeds fixed 2e-5 tolerance")
            for name, query in {"single": actual, "mixture": mix(r1[selected], actual)}.items():
                ranking = base.policy.rank_vectors(query, all_vectors[name][gi], "less_graph")
                for key in ("order", "raw_order"):
                    if not np.array_equal(ranking[key][:, :10], expected[name][key][indices, :10]):
                        raise ValueError(f"Image batch/permutation changed {name} {key} top10")
            print(f"STREAM: batch {size}, queries {len(selected)}, vector error {error:.3g}", flush=True)
        if any(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.state_dict().items()):
            raise ValueError("Evaluation updated model/BN state")
        return {"status": "passed", "trial_id": record["trial_id"], "step": record["step"],
                "max_vector_error": maximum, "batch_sizes": [1,8,16,32], "exact_raw_graph_top10": True,
                "permutation_and_removal": True, "model_state_unchanged": True}
    return previous.task(c, "final_stream_probe", action)


def record_summary(c, records, failures, *, complete=False, selected=None):
    reference = base.old.load_json(Path(c["manifest"]["reference_directory"]) / "metrics.json")
    entries = [{"trial_id": r["trial_id"], "step": r["step"], "architecture": r["architecture"],
                "single_raw": r["single"]["mean_raw_map"], "single_graph": r["single"]["mean_map"],
                "mixture_raw": r["mixture"]["mean_raw_map"], "mixture_graph": r["mixture"]["mean_map"],
                "checkpoint": r["checkpoint"]} for r in records]
    result = {"status": "complete" if complete else "running", "signature": c["signature"],
              "leaderboard": sorted(entries, key=lambda r: -r["mixture_graph"]), "failures": failures,
              "finalists": selected, "reference": reference, "promoted": False,
              "original_outer_evaluation": False, "threshold_fit": False}
    if entries:
        best = max(entries, key=lambda r: r["mixture_graph"])
        single = max(entries, key=lambda r: r["single_graph"])
        result.update(best_mixture=best, best_single=single,
                      mixture_delta_to_r1=best["mixture_graph"]-reference["mean_map"])
    base.write_json(c["output"] / ("results.json" if complete else "progress.json"), result)
    lines = ["# v36 — ночной поиск TransReID/DeiT-Small", "",
             f"Статус: {result['status']}. Исходная validation, порог и v25 не менялись.",
             f"R1: raw {reference['mean_raw_map']:.6f}; graph {reference['mean_map']:.6f}.", "",
             "| Trial | Step | Single raw | Single graph | R190/T10 raw | R190/T10 graph |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in result["leaderboard"]:
        lines.append(f"| {r['trial_id']} | {r['step']} | {r['single_raw']:.6f} | {r['single_graph']:.6f} | "
                     f"{r['mixture_raw']:.6f} | {r['mixture_graph']:.6f} |")
    lines += ["", f"Финалисты: {selected}. Ошибки отдельных конфигураций: {len(failures)}.",
              "24 коротких обучения — не 24 независимых теста. Все checkpoint/параметры отобраны на одном inner holdout.",
              "Ранний отбор на 400 шагах может отсеять медленно сходящиеся варианты; сетка не является исчерпывающей.",
              "Победа над этим R1 не равна победе над full-train v25: для последней нужен отдельный системный этап.",
              "Смесь фиксирована 90/10, граф 20/3/.75, без подбора порога. Метаданные не подаются модели.",
              "Полные per-query метрики, AP/top10/Hit@50 находятся в tasks/*/result.json.",
              "Старые данные, bbox и evaluator сохранены. Номерная устойчивость и GPU-релиз не подтверждены."]
    temporary = c["output"] / "REPORT.md.tmp"
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8"); temporary.replace(c["output"] / "REPORT.md")
    return result


def run(c):
    check_inputs(c)
    with base.old.run_lock(c["output"]):
        if (c["output"] / "complete.json").exists():
            saved = base.old.load_json(c["output"] / "complete.json")
            if saved["signature"] != c["signature"]: raise ValueError("Completed run fingerprint changed")
            base.verify_files(saved["files"])
            print("Verified completed run; no new training or measurements", flush=True)
            return base.old.load_json(c["output"] / "results.json")
        reference_dir = Path(c["manifest"]["reference_directory"])
        ref = base.score_features(c["score_context"], np.load(reference_dir / "features.npy", allow_pickle=False), c["rows"])
        if ref != base.old.load_json(reference_dir / "metrics.json"):
            raise ValueError("R1 control no longer matches exact historical metrics/top10")
        records, screen, failures = [], {}, {}
        by_id = {t["id"]: t for t in c["trials"]}
        for architecture in dict.fromkeys(t["architecture"] for t in c["trials"]):
            trial = next(t for t in c["trials"] if t["architecture"] == architecture)
            records.append(evaluate(c, trial))
        def attempt(trial, horizon):
            failed_path = c["output"] / "training" / trial["id"] / "failed.json"
            if failed_path.exists():
                saved = base.old.load_json(failed_path)
                if saved["signature"] != c["signature"]: raise ValueError("Failure record fingerprint changed")
                failures[trial["id"]] = saved
                if horizon >= saved["horizon"]:
                    return cached_trial_scores(c, trial, horizon)
            try:
                trained = train_until(c, trial, horizon)
                results = []
                for step in c["settings"]["checkpoints"]:
                    if step <= horizon:
                        results.append(evaluate(c, trial, trained["checkpoints"][str(step)], step))
                return results
            except (FloatingPointError, RuntimeError) as exc:
                if not (isinstance(exc, (FloatingPointError, torch.OutOfMemoryError)) or
                        "mps backend out of memory" in str(exc).lower() or "cuda out of memory" in str(exc).lower()):
                    raise
                saved = {"signature": c["signature"], "horizon": horizon, "error": str(exc), "type": type(exc).__name__}
                base.write_json(failed_path, saved); failures[trial["id"]] = saved
                gc.collect()
                if c["device"].type == "mps": torch.mps.empty_cache()
                if c["device"].type == "cuda": torch.cuda.empty_cache()
                print(f"FAILED {trial['id']}: {exc}. No fallback/hyperparameter changes.", flush=True)
                # Completed evaluations are retained even if a later checkpoint fails.
                return cached_trial_scores(c, trial, horizon)
        for index, trial in enumerate(c["trials"], 1):
            print(f"SCREEN {index}/{len(c['trials'])}: {trial}", flush=True)
            found = attempt(trial, c["settings"]["screen_steps"])
            if found:
                records.extend(found); screen[trial["id"]] = found[-1]
            record_summary(c, records, failures)
        selected = choose_finalists(c["trials"], screen, c["settings"]["finalists_per_architecture"])
        if not selected: raise RuntimeError("No successful screening trials; see progress.json/failures")
        base.old.freeze_json(c["output"] / "finalists.json", {"signature": c["signature"], "selected": selected})
        for index, name in enumerate(selected, 1):
            print(f"FINALIST {index}/{len(selected)}: {name}", flush=True)
            found = attempt(by_id[name], c["settings"]["final_steps"])
            if found: records.extend(r for r in found if r["step"] > c["settings"]["screen_steps"])
            record_summary(c, records, failures, selected=selected)
        best = max(records, key=lambda r: r["mixture"]["mean_map"])
        final_probe(c, best)
        check_inputs(c)
        result = record_summary(c, records, failures, complete=True, selected=selected)
        files = {str(p): base.sha256(p) for p in c["output"].rglob("*")
                 if p.is_file() and p.name not in {"complete.json", ".lock"} and not p.name.endswith(".tmp")}
        base.old.freeze_json(c["output"] / "complete.json", {"signature": c["signature"], "files": files})
        print("NIGHT SEARCH COMPLETE. v25 unchanged. Report:", c["output"] / "REPORT.md", flush=True)
        return result
