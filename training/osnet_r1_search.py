"""v42: matched-exposure R1 continuation; no release, outer or v41 mutations."""
import os
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import argparse
from itertools import product
from pathlib import Path
import platform
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from training import research_io as io, research_models as models, research_runtime as runtime
from training import research_scoring as scoring, research_training as common
from training import retrieval_policy as policy
from training.hpo import supervised_contrastive_loss
from training.transreid_model import soft_triplet

VARIANT = io.ROOT / "OSNet-AIN-x1.0/variant_42_osnet_r1_search"
PARENT = "R1_20260915"
SYSTEM_POLICIES = ("add_G2_10", "replace_R1_20260915")


def trial_grid(recipe):
    # The anchor is the ORIGINAL v16 R1 LR, not the tenfold-reduced NiVe LR.
    return [{"id": f"R{i:02d}_p{p}k{k}_{loss}_lr{factor:g}", "p": p, "k": k,
             "loss": loss, "lr_factor": factor, "lr": recipe["encoder_lr"] * factor,
             "seed": 20260915, "size": 256}
            for i, ((p, k), loss, factor) in enumerate(product(
                ((16, 2), (16, 4), (32, 2)), ("supcon", "soft_triplet"), (.1, .3, 1.)), 1)]


def check_settings(settings, trials):
    boundaries = settings["presentations"]
    if (not boundaries or boundaries != sorted(set(boundaries))
            or any(type(n) is not int or n <= 0 for n in boundaries)
            or any(n % (s["p"] * s["k"]) for n in boundaries for s in trials)):
        raise ValueError("Every boundary must give exactly equal image exposure for all P/K")
    if not 0 < settings["warmup_presentations"] < boundaries[-1]:
        raise ValueError("Warmup must be inside the fixed full horizon")
    if not 0 < settings["min_lr_ratio"] <= 1:
        raise ValueError("Invalid minimum LR")
    if len(trials) != 18 or len({s["id"] for s in trials}) != 18:
        raise ValueError("Expected the frozen 18-trial matrix")


def lr_fraction(presentations, settings):
    warmup, horizon = settings["warmup_presentations"], settings["presentations"][-1]
    if not 0 < presentations <= horizon:
        raise ValueError("LR exposure is outside the fixed horizon")
    if presentations <= warmup:
        return presentations / warmup
    phase = (presentations - warmup) / (horizon - warmup)
    floor = settings["min_lr_ratio"]
    return float(floor + (1 - floor) * .5 * (1 + np.cos(np.pi * phase)))


def losses(model, clean, robust, labels, recipe, spec):
    model.eval()
    with torch.no_grad():
        target = model.embedding(clean)
    model.train()
    logits, raw, embedding = model(robust)
    ce = F.cross_entropy(logits, labels, label_smoothing=recipe["label_smoothing"])
    metric = (supervised_contrastive_loss(raw, labels, recipe["supcon_temperature"])
              if spec["loss"] == "supcon" else soft_triplet(raw, labels))
    consistency = (1 - F.cosine_similarity(embedding, target, dim=1)).mean()
    total = ce + recipe["metric_weight"] * metric + recipe["consistency_weight"] * consistency
    return {"loss": total, "ce": ce, "metric": metric, "consistency": consistency,
            "accuracy": (logits.argmax(1) == labels).float().mean(),
            "feature_norm": raw.norm(dim=1).mean()}


def optimizer_for(model, recipe, spec):
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return torch.optim.AdamW([
        {"params": backbone, "base_lr": spec["lr"]},
        {"params": heads, "base_lr": spec["lr"] * recipe["head_lr_multiplier"]}],
        lr=spec["lr"], weight_decay=recipe["weight_decay"], foreach=False)


def train_to(model, c, spec, exposure, directory):
    """Exact step-addressed resume; all arms share the same exposure-based schedule."""
    directory = Path(directory)
    signature = io.digest({"run": c["signature"], "spec": spec})
    io.freeze(directory / "trial.json", {"signature": signature, "spec": spec})
    batch_size = spec["p"] * spec["k"]
    if exposure % batch_size or exposure > c["settings"]["presentations"][-1]:
        raise ValueError("Invalid exposure boundary")
    stop = exposure // batch_size
    optimizer = optimizer_for(model, c["recipe"], spec)
    start, history, elapsed = 0, [], 0.
    pointer = directory / "resume.json"
    if pointer.exists():
        entry = io.read(pointer)
        io.verify(directory, {entry["path"]: entry["sha256"]})
        saved = torch.load(io.child(directory, entry["path"]), map_location="cpu", weights_only=True)
        if saved["signature"] != signature or saved["step"] > stop:
            raise ValueError("Resume changed or is newer than the missing evaluation stage")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, history, elapsed = saved["step"], saved["history"], saved["elapsed"]
        del saved
    data = models.Images(c["inputs"], c["train"], spec["size"], train=True, paired=True)
    began = time.perf_counter()
    for step in range(start, stop):
        seen = (step + 1) * batch_size
        fraction = lr_fraction(seen, c["settings"])
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * fraction
        indices = common.batch_indices(c["train"], spec["p"], spec["k"], spec["seed"], step)
        batch = common.fetch(data, indices, c["device"], spec["seed"] + 100001 + step)
        optimizer.zero_grad(set_to_none=True)
        values = losses(model, *batch, c["recipe"], spec)
        if not all(torch.isfinite(v) for v in values.values()):
            raise FloatingPointError("Non-finite R1 loss")
        values["loss"].backward()
        # Preserve R1: measure the norm, without introducing a new clipping hypothesis.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step + 1, "presentations": seen,
                        "equivalent_passes": seen / len(c["train"]),
                        **{k: float(v.detach()) for k, v in values.items()},
                        "grad_norm": float(norm), "lr": [g["lr"] for g in optimizer.param_groups]})
        seconds = elapsed + time.perf_counter() - began
        if (step + 1) % 25 == 0 or step + 1 == stop:
            if c["device"].type == "mps":
                from training.research_mac import memory_sample
                history[-1].update(memory_sample())
            remaining = (time.perf_counter() - began) / (step + 1 - start) * (stop - step - 1)
            print(f"{spec['id']} | step {step+1}/{stop} | exposure {seen}/{exposure} | "
                  f"passes {seen/len(c['train']):.2f} | loss {history[-1]['loss']:.4f} | "
                  f"time {seconds/60:.1f} min | stage ETA {remaining/60:.1f} min", flush=True)
        if (step + 1) % 100 == 0 or step + 1 == stop:
            slot = directory / f"resume_{((step+1)//100)%2}.pt"
            common.save_torch(slot, {"signature": signature, "step": step + 1,
                                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "history": history, "elapsed": seconds})
            io.write(pointer, {"path": slot.name, "sha256": io.sha(slot)})
            io.write(directory / "history.json", history)
    stage = directory / f"exposure_{exposure:06d}"
    path = stage / "checkpoint.pt"
    common.save_torch(path, {"signature": signature, "model": model.state_dict(), "spec": spec,
                            "step": stop, "presentations": exposure,
                            "parent": c["manifest"]["models"][PARENT]})
    return {"path": path.relative_to(c["output"]).as_posix(), "sha256": io.sha(path),
            "step": stop, "presentations": exposure,
            "elapsed_seconds": elapsed + time.perf_counter() - began}


def replacement_bank(parts, features):
    if features.shape != parts[1].shape:
        raise ValueError("Replacement must have the same R1 dimension and row order")
    return np.concatenate([parts[0], scoring.mix([features, parts[2], parts[3]], [1/3]*3)], axis=1)


def evaluation_context(c):
    names = ("B0", PARENT, "R1_20260916", "R1_20260917")
    parts = [np.load(c["output"] / "baseline" / f"{n}.npy", allow_pickle=False) for n in names]
    if not np.array_equal(replacement_bank(parts, parts[1]), c["control"]):
        raise ValueError("Replacing R1 with itself must exactly reproduce the control bank")
    index = {r["image_id"]: i for i, r in enumerate(c["rows"])}
    draws = {}
    for name, draw in c["manifest"]["draws"].items():
        qi, gi = ([index[i] for i in draw[key]] for key in ("query_ids", "gallery_ids"))
        query, gallery = ([c["rows"][i] for i in indices] for indices in (qi, gi))
        raw, jac = scoring.FixedGraph(c["control"][gi]).components(c["control"][qi])
        order = np.argsort(.5 * jac + .5 * raw, axis=1, kind="stable")
        draws[name] = {"qi": qi, "gi": gi, "query": query, "gallery": gallery,
                       "raw": raw, "jac": jac,
                       "control": scoring.ranking_report(query, gallery, order)}
    return {"parts": parts, "draws": draws}


def score_checkpoint(features, c):
    e = c["evaluation"]
    if (features.shape != e["parts"][1].shape or not np.isfinite(features).all()
            or np.any(np.linalg.norm(features, axis=1) == 0)):
        raise ValueError("Invalid real feature matrix")
    replacement = scoring.control_vectors(replacement_bank(e["parts"], features))
    reports = {}
    for name, d in e["draws"].items():
        qi, gi, query, gallery = (d[k] for k in ("qi", "gi", "query", "gallery"))
        single = policy.rank_vectors(features[qi], features[gi], "less_graph")
        added = scoring.fuse_distances(d["raw"], d["jac"], features[qi], features[gi], "G2", .1)
        replaced = policy.rank_vectors(replacement[qi], replacement[gi], "legacy")
        orders = {"raw": single["raw_order"], "graph": single["order"],
                  "add_G2_10": np.argsort(added, axis=1, kind="stable"),
                  "replace_R1_20260915": replaced["order"]}
        current = {"control": d["control"], **{key: scoring.ranking_report(query, gallery, order)
                                               for key, order in orders.items()}}
        for key in SYSTEM_POLICIES:
            for qid, item in current[key]["per_query"].items():
                item["delta_ap"] = (item["ap"] - current["control"]["per_query"][qid]["ap"]
                                    if item["known"] else None)
        reports[name] = current
    means = {key: float(np.mean([r[key]["metrics"]["mAP@10"] for r in reports.values()]))
             for key in next(iter(reports.values()))}
    if not all(np.isfinite(v) for v in means.values()):
        raise FloatingPointError("Non-finite ranking metric")
    return {"means": means, "draws": reports}


def evaluate_parent(c):
    directory = c["output"] / "parent"
    if io.completed(directory, c["signature"]):
        return io.read(directory / "result.json")
    report = {"trial": "unchanged_R1", "presentations": 0, "checkpoint": None,
              "metrics": score_checkpoint(c["evaluation"]["parts"][1], c)}
    # A real no-op control checks top-10, not merely an approximately equal scalar score.
    for draw in report["metrics"]["draws"].values():
        for qid, item in draw["control"]["per_query"].items():
            if item["top10"] != draw["replace_R1_20260915"]["per_query"][qid]["top10"]:
                raise ValueError("Unchanged R1 replacement changed control decisions")
    io.write(directory / "result.json", report)
    io.finish(directory, c["signature"])
    return report


def run_trial(c, spec):
    model = None
    directory = c["output"] / "trials" / spec["id"]
    try:
        for exposure in c["settings"]["presentations"]:
            stage = directory / f"exposure_{exposure:06d}"
            signature = io.digest({"run": c["signature"], "spec": spec, "exposure": exposure})
            if io.completed(stage, signature):
                print(f"RESUME verified: {spec['id']} / {exposure}", flush=True)
                continue
            if model is None:
                common.seed_all(spec["seed"])
                model = models.load_osnet(c["inputs"], c["manifest"], PARENT).to(c["device"])
            check_disk(c["output"], 1024**3)
            checkpoint = train_to(model, c, spec, exposure, directory)
            features = models.encode(model, c["inputs"], c["rows"], c["device"], 256, 16)
            np.save(stage / "features.npy", features)
            io.write(stage / "order.json", [r["image_id"] for r in c["rows"]])
            report = {"trial": spec["id"], "spec": spec, "presentations": exposure,
                      "checkpoint": checkpoint, "metrics": score_checkpoint(features, c)}
            io.write(stage / "result.json", report)
            io.finish(stage, signature)
            print(f"SCORED {spec['id']} / {exposure}: {report['metrics']['means']}", flush=True)
            summarize(c, "running")
    finally:
        del model
        runtime.cleanup()


def summarize(c, status):
    parent = io.read(c["output"] / "parent/result.json")
    reports = [parent]
    for path in sorted((c["output"] / "trials").glob("*/exposure_*/result.json")):
        if (path.parent / "complete.json").exists():
            reports.append(io.read(path))
    baseline = parent["metrics"]["means"]["control"]
    best = {"policy": "control", "score": baseline, "delta": 0., "trial": None,
            "presentations": 0, "checkpoint": None}
    leaderboard = []
    for r in reports:
        means = r["metrics"]["means"]
        chosen = max(SYSTEM_POLICIES, key=lambda key: means[key])
        entry = {"trial": r["trial"], "presentations": r["presentations"], "means": means,
                 "policy": chosen, "score": means[chosen], "delta": means[chosen] - baseline,
                 "checkpoint": r["checkpoint"]}
        leaderboard.append(entry)
        if entry["score"] > best["score"]:
            best = entry
    leaderboard.sort(key=lambda r: r["score"], reverse=True)
    statuses = [io.read(p) for p in sorted((c["output"] / "status").glob("*.json"))]
    result = {"status": status, "signature": c["signature"], "baseline": c["manifest"]["baseline"],
              "control_map": baseline, "completed_trials": sum(s["status"] == "complete" for s in statuses),
              "planned_trials": len(c["trials"]), "statuses": statuses, "leaderboard": leaderboard,
              "best_system": best, "best_single_raw": max(leaderboard, key=lambda r: r["means"]["raw"]),
              "best_single_graph": max(leaderboard, key=lambda r: r["means"]["graph"]),
              "selection_is_provisional": status != "complete", "promoted": False,
              "outer_evaluated": False, "threshold_fit": False, "external_training": False}
    io.write(c["output"] / "results.json", result)
    if status == "complete":
        io.freeze(c["output"] / "selected_candidate.json", {
            "signature": c["signature"], "best_system": best, "promoted": False,
            "scope": "primary development selection, NOT final release or unseen-test quality"})
    lines = ["# v42 — целевой поиск OSNet R1", "", f"Статус: {status}.", "",
             f"Завершено {result['completed_trials']}/{len(c['trials'])} конфигураций.",
             c["manifest"]["baseline"], "",
             "Продолжение одного R1 step800 с новым AdamW. Не обучение с нуля и не новый независимый тест.",
             "Best concrete checkpoint: без усреднения качества по seed. v25 и v41 не изменены.",
             "Порог/F1/TNR не подбираются; это исследование ranking, не готовая замена MVP.", "",
             "| Trial | Доп. предъявления | Single raw | Single graph | Add G2 | Replace R1 | Δ system |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in leaderboard:
        m = r["means"]
        lines.append(f"| {r['trial']} | {r['presentations']} | {m['raw']:.6f} | {m['graph']:.6f} | "
                     f"{m['add_G2_10']:.6f} | {m['replace_R1_20260915']:.6f} | {r['delta']:+.6f} |")
    lines += ["", f"Контроль: {baseline:.6f}. Лучший system: {best['score']:.6f} ({best['policy']}).",
              "", "## Ошибки", ""]
    lines += [f"- {s['trial']}: {s['error']}" for s in statuses if s["status"] == "failed"]
    lines += ["", "В компактном ZIP нет весов. Сохраните runs целиком: checkpoints/features и per-query top10.",
              "Три draws используют одни holdout-ID. Это не три независимых fold.",
              "Одинаковые предъявления не означают одинаковое число optimizer updates.",
              "Presentations считают robust-примеры; clean-consistency forward выполняется дополнительно."]
    temporary = c["output"] / "REPORT.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(c["output"] / "REPORT.md")
    return result


def check_disk(path, required):
    free = shutil.disk_usage(path).free
    if free < required:
        raise OSError(f"Недостаточно места: {free/2**30:.1f} GiB; нужно {required/2**30:.1f} GiB. "
                      "Освободите место самостоятельно; данные/история не удаляются.")


def mps_preflight():
    if platform.system() != "Darwin" or platform.machine() != "arm64" or not torch.backends.mps.is_available():
        raise ValueError("v42 requires native Apple Silicon Python with working MPS; no CPU fallback")
    if any(os.environ.get(k) != "0" for k in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")):
        raise ValueError("Disable MPS fallback/fast math BEFORE importing torch; restart the kernel")
    torch.mps.set_per_process_memory_fraction(.85)
    parameter = torch.nn.Parameter(torch.ones(4, 4, device="mps"))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, foreach=False)
    parameter.square().mean().backward()
    optimizer.step()
    torch.mps.synchronize()
    if not torch.isfinite(parameter).all():
        raise FloatingPointError("MPS numerical preflight failed")
    del parameter, optimizer
    torch.mps.empty_cache()
    recommended = torch.mps.recommended_max_memory()
    print(f"MPS FP32 | allocator limit {recommended*.85/2**30:.1f} GiB | "
          "no fallback | disk minimum 3 GiB, recommended 5–6 GiB", flush=True)
    return {"machine": platform.machine(), "recommended_mps_bytes": recommended, "allocator_fraction": .85}


def run(inputs, run_name="r1_search_v1", device="mps"):
    if device not in {"mps", "cuda"}:
        raise ValueError("Use explicit MPS/CUDA; no silent CPU fallback")
    if not run_name or not run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Simple RUN_NAME required")
    settings = io.read(VARIANT / "config.json")
    if device == "mps":
        settings["hardware"] = mps_preflight()
    output = VARIANT / "runs" / run_name
    with io.lock(output):
        check_disk(output, 3 * 2**30)
        c = runtime.prepare(inputs, output, device, settings)
        recipe = io.read(c["inputs"] / "provenance/v16_manifest.json")["base_recipe"]
        c.update(settings=settings, recipe=recipe, trials=trial_grid(recipe))
        # Portable inputs also contain NiVe for v41. It is NEVER sampled in this experiment.
        c["external"] = []
        check_settings(settings, c["trials"])
        io.freeze(output / "plan.json", {"trials": c["trials"], "recipe": recipe,
                                         "settings": settings, "initial_checkpoint": c["manifest"]["models"][PARENT]})
        c["control"] = runtime.baselines(c)
        c["evaluation"] = evaluation_context(c)
        evaluate_parent(c)
        summarize(c, "running")
        try:
            for index, spec in enumerate(c["trials"], 1):
                print(f"\nTRIAL {index}/{len(c['trials'])}: {spec['id']} | {device} | lr={spec['lr']:.8g}", flush=True)
                try:
                    run_trial(c, spec)
                    entry = {"trial": spec["id"], "status": "complete"}
                except (RuntimeError, FloatingPointError) as error:
                    # OOM/numerical failures are visible; never alter batch, LR or precision silently.
                    entry = {"trial": spec["id"], "status": "failed", "error": str(error),
                             "traceback": traceback.format_exc()}
                    print(f"FAILED {spec['id']}: {error}. Continuing other trials.", flush=True)
                io.write(output / "status" / f"{spec['id']}.json", entry)
                summarize(c, "running")
        except BaseException:
            summarize(c, "interrupted")
            raise
        # Verify immutable inputs again, including the original annotation/provenance bytes.
        io.verify(c["inputs"], c["manifest"]["files"])
        statuses = [io.read(p) for p in (output / "status").glob("*.json")]
        result = summarize(c, "complete" if all(s["status"] == "complete" for s in statuses)
                           else "completed_with_failures")
        io.archive(output, VARIANT / f"{run_name}_analysis.zip", light=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=io.ROOT / "research_transfer/v41_inputs")
    parser.add_argument("--run-name", default="r1_search_v1")
    parser.add_argument("--device", choices=["mps", "cuda"], default="mps")
    args = parser.parse_args()
    run(args.inputs, args.run_name, args.device)
