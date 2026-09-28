"""v38: one fixed T12 full-train refit; never select a checkpoint/mixture on outer data."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import copy
import importlib.metadata
from pathlib import Path
import platform
import re

import numpy as np
import torch

from training import transreid_night as night, transreid_system as previous
from training import transreid_model as vision, transreid_system_inference as inference

base = night.base
VARIANT = base.ROOT / "OSNet-AIN-x1.0/variant_38_transreid_full_train"
TRIAL = {"id": "T12_global_supcon", "architecture": "global", "encoder_lr": 1e-4,
         "weight_decay": .01, "metric_loss": "supcon"}
CASES = {"V25_control": "V25_control", "Full_T12_w10": "T12_w10"}
task = previous.task
protocol_rows = previous.protocol_rows
baseline_features = previous.baseline_features


def validate_plan(plan, source_training):
    settings = plan["training"]
    if (plan["trial"] != TRIAL or plan["train_identities"] != 925 or plan["selected_step"] != 1800
            or settings["final_steps"] != 1800 or settings["checkpoints"] != [1800]
            or plan["initializer"] != "official_DeiT_pretrained_not_old_T12"
            or plan["system"] != "Full_T12_w10" or plan["new_encoder_weight"] != .1
            or plan["graph"] != inference.GRAPH or plan["vector_atol"] != 2e-5
            or any(plan[k] for k in ("threshold_fit", "inner_evaluation", "external_train", "promoted"))
            or plan["wall_time_limit"] is not None):
        raise ValueError("Use the fixed full-train T12 recipe, final step and 90/10 mixture")
    # These only control which checkpoints/smoke cases are retained, not the recipe.
    administrative = {"checkpoints", "estimated_run_disk_gib", "grid"}
    if any(source_training[k] != v for k, v in settings.items() if k not in administrative):
        raise ValueError("Full refit must preserve the selected T12 training hyperparameters")
    if settings["grid"] != {"metric_loss": ["supcon"]}:
        raise ValueError("Exactly one disposable SupCon device smoke is allowed")


def full_train_rows(rows, splits, protocols, previous_train_ids, expected_count=925):
    allowed = set(splits["identities"]["train"])
    previous.previous.validate_splits(rows, protocols, splits["identities"], sorted(allowed))
    if len(allowed) != expected_count or not set(previous_train_ids) < allowed:
        raise ValueError("Full train must include the old inner holdout and only original train IDs")
    labels = {identity: i for i, identity in enumerate(sorted(allowed))}
    target = [{**r, "label": labels[r["vehicle_id"]]} for r in rows if r["vehicle_id"] in allowed]
    if {r["vehicle_id"] for r in target} != allowed:
        raise ValueError("Some full-train identities have no images")
    return target, sorted(allowed)


def check_inputs(c):
    m = c["manifest"]
    if (base.digest(m) != c["signature"] or c["settings"] != m["plan"]["training"]
            or c["trials"] != [TRIAL] or str(c["device"]) != m["runtime"]["device"]
            or base.digest(c["target"]) != m["train_rows_sha256"]
            or base.digest(c["schedule"]) != m["schedule_sha256"]):
        raise ValueError("Frozen full-train context changed")
    print(f"CHECK: {len(m['protected'])} protected files and source; no historical edits", flush=True)
    base.verify_files(m["protected"])
    base.verify_files(m["source_sha256"])
    vision.verify_weights(m["weight_path"])


def prepare(run_name="refit_v1", device="mps"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple RUN_NAME")
    device = vision.device_for(device)
    weight = vision.verify_weights()  # Already downloaded for v36; never download implicitly.
    config = VARIANT / "configs/refit_v1.json"
    plan = base.old.load_json(config)
    source = base.ROOT / plan["source_run"]
    pins = {str(source / f"{n}.json"): plan[f"source_{n}_sha256"] for n in ("manifest", "results", "complete")}
    base.verify_files(pins)
    sm, result, complete = (base.old.load_json(source / f"{n}.json") for n in ("manifest", "results", "complete"))
    if (result["status"] != "complete" or result["signature"] != base.digest(sm)
            or complete["signature"] != result["signature"] or result["promoted"]
            or sm["models"]["T12"]["trial"] != TRIAL or result["selection"]["selected"] != "T12_w10"):
        raise ValueError("Expected the completed unchanged v37/T12 comparison")
    source36 = base.old.load_json(Path(sm["source_directory"]) / "manifest.json")
    if base.digest(source36) != sm["source_signature"]:
        raise ValueError("Original recipe provenance changed")
    validate_plan(plan, source36["settings"])
    protected = {**sm["protected"], **pins, **{str(source / p): h for p, h in complete["files"].items()}}
    print("PREFLIGHT: verify saved v36/v37, v25, data and original protocols", flush=True)
    base.verify_files(protected)
    base.verify_files(sm["source_sha256"])
    rows = base.read_rows(base.DATASET / "train.csv")
    splits = base.old.load_json(base.ARTIFACTS / "splits.json")
    target, identities = full_train_rows(rows, splits, sm["protocols"], sm["train_ids"], plan["train_identities"])
    paths = vision.image_paths(base.DATASET, target)
    if any(str(p) not in protected for p in paths.values()):
        raise ValueError("An input image is outside original byte protection")
    if base.old.load_json(base.ROOT.parent / "Car-classification-MSK-main/release_decision.json")["active_profile"] != "MVP_fusion_v25":
        raise ValueError("Active MVP changed; review the control")
    settings = plan["training"]
    sampler = night.ExperimentConfig(identities_per_batch=settings["identities_per_batch"],
                                    images_per_identity=settings["images_per_identity"], seed=settings["seed"])
    schedule = list(night.StepPKBatchSampler(target, sampler, settings["final_steps"]))
    if {target[i]["vehicle_id"] for batch in schedule for i in batch} != set(identities):
        raise ValueError("Fixed full-train schedule omitted an identity")
    runtime = {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "numpy", "pillow", "onnxruntime")}
    runtime.update(device=str(device), python=platform.python_version(), platform=platform.platform(),
                   torch_threads=torch.get_num_threads(), dtype="float32", amp=False, compile=False)
    sources = {**sm["source_sha256"], **{str(p): base.sha256(p) for p in (Path(__file__).resolve(), config)}}
    manifest = {"version": 38, "plan": plan, "systems": CASES, "train_ids": identities,
        "previous_train_ids": sm["train_ids"], "train_images": len(target), "train_rows_sha256": base.digest(target),
        "schedule_sha256": base.digest(schedule), "all_train_ids_sampled": True,
        "weight_path": str(weight), "weight_sha256": vision.WEIGHT_SHA256, "preprocessing": vision.PREPROCESS,
        "protocols": sm["protocols"], "threshold": sm["threshold"], "v24_directory": sm["v24_directory"],
        "v25_directory": sm["v25_directory"], "source_directory": str(source), "source_signature": result["signature"],
        "runtime": runtime, "protected": protected, "source_sha256": sources,
        "selection": "no search: final 1800 checkpoint and 10% mixture fixed before training",
        "scope": "development comparison, not a new independent test; original inner holdout is now training",
        "promoted": False}
    c = {"output": VARIANT / "runs" / run_name, "manifest": manifest, "signature": base.digest(manifest),
         "settings": copy.deepcopy(settings), "trials": [copy.deepcopy(TRIAL)], "device": device,
         "rows": rows, "target": target, "paths": paths, "schedule": schedule, "dataset": base.DATASET}
    night.disk_guard(c["output"], settings)
    with base.old.run_lock(c["output"]):
        base.old.freeze_json(c["output"] / "manifest.json", manifest)
    print(f"PREFLIGHT OK: {len(identities)} train-ID, {len(target)} images, 1800 updates, "
          "batch=8x2=16; original calibration/validation excluded from training", flush=True)
    return c


def freeze_candidate(c, trained):
    step = c["manifest"]["plan"]["selected_step"]
    if trained["step"] != step or set(trained["checkpoints"]) != {str(step)}:
        raise ValueError("Only the predeclared final checkpoint may be evaluated")
    cp = trained["checkpoints"][str(step)]
    expected = c["output"] / "training" / c["trials"][0]["id"] / f"step_{step:05d}.pt"
    if Path(cp["path"]).resolve() != expected.resolve():
        raise ValueError("Checkpoint must belong to this full-train run")
    base.verify_files({cp["path"]: cp["sha256"]})
    payload = torch.load(cp["path"], map_location="cpu", weights_only=True)
    if (payload["signature"] != c["signature"] or payload["trial"] != c["trials"][0]
            or payload["step"] != step or payload["model"]["heads.0.weight"].shape[0] != len(c["manifest"]["train_ids"])):
        raise ValueError("Checkpoint is not the full-train T12 with the declared classifier")
    value = {"signature": c["signature"], "checkpoint": cp, "step": step, "trial": c["trials"][0],
        "train_ids_sha256": base.digest(c["manifest"]["train_ids"]), "systems": CASES,
        "graph": inference.GRAPH, "new_encoder_weight": .1, "threshold": c["manifest"]["threshold"],
        "selection_split": None, "selection": "fixed before training, no quality-based selection", "promoted": False}
    base.old.freeze_json(c["output"] / "frozen_candidate.json", value)
    return value


def require_candidate(c):
    path = c["output"] / "frozen_candidate.json"
    if not path.is_file():
        raise ValueError("Outer evaluation is closed until full training and checkpoint freeze finish")
    saved = base.old.load_json(path)
    expected = freeze_candidate(c, {"step": c["manifest"]["plan"]["selected_step"],
        "checkpoints": {str(c["manifest"]["plan"]["selected_step"]): saved["checkpoint"]}})
    if saved != expected: raise ValueError("Frozen candidate changed")
    return saved


def load_model(c):
    chosen = require_candidate(c)
    model_context = {"device": c["device"], "manifest": {"source_signature": c["signature"],
        "train_ids": c["manifest"]["train_ids"], "drop_path": c["settings"]["drop_path"],
        "models": {"T12": {**chosen["checkpoint"], "step": chosen["step"], "trial": chosen["trial"], "architecture": "global"}}}}
    return previous.load_model(model_context, "T12")


def features(c, split):
    chosen = require_candidate(c)
    q, g = protocol_rows(c, split)
    rows = q+g
    name = f"features_{split}_Full_T12"
    def action(directory):
        model = load_model(c)
        values = vision.encode(model, rows, vision.image_paths(c["dataset"], rows), c["device"], c["settings"]["eval_batch_size"])
        np.save(directory / "features.npy", values)
        del model
        previous.release_device(c)
        return {"ids": [r["image_id"] for r in rows], "shape": list(values.shape), "checkpoint": chosen["checkpoint"]}
    result = task(c, name, action)
    values = np.load(c["output"] / "tasks" / name / "features.npy", allow_pickle=False)
    inference.dual.validate_block(values, 384)
    if (result["ids"] != [r["image_id"] for r in rows] or values.shape != (len(rows), 384)
            or result["checkpoint"] != chosen["checkpoint"]):
        raise ValueError("Feature cache identity/order changed")
    return values


def stream_probe(c, original, extra):
    """Fresh real images; exact graph decisions with a fixed calibration gallery."""
    require_candidate(c)
    def action(directory):
        q, g = protocol_rows(c, "calibration")
        model = load_model(c)
        initial = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        bank = np.concatenate([original, extra], axis=1)
        expected = inference.rank(bank, len(q), "T12_w10")
        paths = vision.image_paths(c["dataset"], q[:32])
        query_indices = np.arange(min(32, len(q)))
        cases = [(query_indices, n) for n in (1, 8, 16, 32)] + [(query_indices[::-1], 16), (query_indices[:1], 1)]
        maximum = 0.
        for indices, batch in cases:
            actual = vision.encode(model, [q[i] for i in indices], paths, c["device"], batch, progress=False)
            error = float(abs(actual-extra[indices]).max()); maximum = max(maximum, error)
            if not np.allclose(actual, extra[indices], atol=2e-5, rtol=0):
                raise ValueError("Batch/order extraction drift exceeds fixed tolerance")
            query = np.concatenate([original[indices], actual], axis=1)
            ranked = inference.rank(np.concatenate([query, bank[len(q):]]), len(indices), "T12_w10")
            if (not np.array_equal(ranked["order"][:, :10], expected["order"][indices, :10])
                    or not np.array_equal(ranked["raw_order"][:, 0], expected["raw_order"][indices, 0])
                    or not np.array_equal(ranked["confidence"] >= c["manifest"]["threshold"],
                                          expected["confidence"][indices] >= c["manifest"]["threshold"])):
                raise ValueError("Batch/order changed top-10, candidate or refusal")
            print(f"STREAM: batch={batch}, queries={len(indices)}, max error={error:.3g}", flush=True)
        if any(not torch.equal(v.cpu(), initial[k]) for k, v in model.named_buffers()):
            raise ValueError("Inference changed BN/model buffers")
        del model
        encoder = inference.dual.DualRoleEncoder(Path(c["manifest"]["v24_directory"]) / "dual_role_profile.json")
        actual = encoder.encode_rows(q[:8], c["dataset"], 8)
        reference = original[:min(8, len(q))]
        if actual.shape != reference.shape or not np.allclose(actual, reference, atol=2e-5, rtol=0):
            raise ValueError("v25 cache no longer reproduces from images")
        old_error = float(abs(actual-reference).max())
        del encoder
        previous.release_device(c)
        return {"status": "passed", "max_vector_error": maximum, "v25_cache_error": old_error,
                "batch_sizes": [1, 8, 16, 32], "permutation_and_removal": True, "exact_top10_candidates_refusals": True,
                "inference_bn_updates": 0, "scope": "fresh correctness probe, not a speed benchmark"}
    return task(c, "stream_probe", action)


def evaluate(c, split, system, values, directory):
    chosen = require_candidate(c)
    if system not in CASES: raise ValueError("Only v25 and fixed Full_T12_w10 are allowed")
    q, g = protocol_rows(c, split)
    threshold = c["manifest"]["threshold"]
    ranked = inference.rank(values, len(q), CASES[system], progress=True)
    report = {"split": split, "system": system, "threshold": threshold,
        **base.policy.evaluate(q, g, ranked, threshold, "raw_top1"), **base.policy.query_diagnostics(q, g, ranked)}
    ordered, accepted = base.policy.predictions(q, g, ranked, threshold, "raw_top1")
    old_run = Path(c["manifest"]["source_directory"])
    expected = base.old.load_json(old_run / "tasks" / f"{split}_V25_control" / "result.json")
    original, _ = baseline_features(c, split, q, g)
    _, r1 = inference.dual.unpack(original)
    raw = base.policy.rank_vectors(r1[:len(q)], r1[len(q):], "raw")
    if accepted != base.policy.predictions(q, g, raw, threshold, "raw_top1")[1] or report["candidates"] != expected["candidates"]:
        raise ValueError("v25 candidates/confidences/refusals changed")
    for qid, top10 in ordered.items(): report["per_query"][qid]["ranking"]["top10"] = top10
    if system == "V25_control" and (report["ranking"] != expected["ranking"] or report["per_query"] != expected["per_query"]):
        raise ValueError("v25 control no longer reproduces exact metrics and decisions")
    # Diagnostic only: same frozen mixture without graph, not an alternative selector/candidate policy.
    mixed = inference.ranking_features(values, CASES[system])
    raw_mixed = base.policy.rank_vectors(mixed[:len(q)], mixed[len(q):], "raw")
    report["raw_ranking"] = base.policy.evaluate(q, g, raw_mixed, threshold, "raw_top1")["ranking"]
    output = directory / "export"
    metrics = inference.export_arrays(output, q, g, values, CASES[system], threshold)
    if any(metrics[k] != report[k] for k in ("ranking", "candidates")):
        raise ValueError("Export/replay differs from reported decisions")
    before = old_run / "tasks" / f"export_{split}_V25_control" / "export"
    base.verify_files({str(output / "candidates.csv"): base.sha256(before / "candidates.csv")})
    if system == "V25_control":
        base.verify_files({str(output / n): base.sha256(before / n) for n in ("submission.csv", "embeddings.npy")})
    base.write_json(output / "model_provenance.json", {"system": system, "signature": c["signature"],
        "new_encoder": chosen if system != "V25_control" else None,
        "v25_profile": str(Path(c["manifest"]["v24_directory"]) / "dual_role_profile.json"), "promoted": False})
    print(f"EVAL {split}/{system}: raw={report['raw_ranking']['mAP@10']:.6f}, "
          f"graph={report['ranking']['mAP@10']:.6f}; candidates unchanged", flush=True)
    return report


def write_report(c, result):
    old = base.old.load_json(Path(c["manifest"]["source_directory"]) / "results.json")
    lines = ["# v38 — один T12 на полном train", "",
        f"Обучение: {len(c['manifest']['train_ids'])} ID, {c['manifest']['train_images']} изображений, "
        f"{result['training']['step']} optimizer updates. Инициализация — исходный официальный DeiT, не старый T12.",
        "Смесь 90% v25 + 10% Full T12 и final checkpoint зафиксированы до обучения; подбора по внешним метрикам нет.", "",
        "| Split | Система | Raw mAP@10 | Graph mAP@10 | Δ graph к v25, п.п. | Rank-1 |",
        "|---|---|---:|---:|---:|---:|"]
    for split, reports in result["evaluations"].items():
        baseline = reports["V25_control"]["ranking"]["mAP@10"]
        for name, r in reports.items():
            lines.append(f"| {split} | {name} | {r['raw_ranking']['mAP@10']:.6f} | {r['ranking']['mAP@10']:.6f} | "
                         f"{100*(r['ranking']['mAP@10']-baseline):+.4f} | {r['ranking']['Rank-1']:.6f} |")
        historical = old["evaluations"][split]["T12_w10"]["ranking"]
        lines.append(f"| {split} | T12 740 ID — сохранённый v37, не новый замер | — | {historical['mAP@10']:.6f} | "
                     f"{100*(historical['mAP@10']-baseline):+.4f} | {historical['Rank-1']:.6f} |")
    pair = result["paired_validation"]
    lines += ["", f"Validation AP: улучшилось {pair['improved']}, ухудшилось {pair['worsened']}, без изменения {pair['unchanged']}.",
        f"Top-1 исправлен у {pair['top1_fixed']}, испорчен у {pair['top1_broken']}; top-10 изменился у {pair['top10_changed']} запросов.",
        "", "## Ограничения и воспроизведение", "",
        "Рецепт T12 сохранён: global384, CE + SupCon, lr=1e-4, wd=.01, seed=20260915, P=8/K=2, 1800 шагов.",
        "Сохранено число шагов, не число проходов по каждому ID: при 925 вместо 740 средняя экспозиция на ID ниже.",
        "Это полный refit с новой 925-классовой головой; 740-ID checkpoint не перезаписан и не используется как initializer.",
        "Один дополнительный disposable smoke-update проверяет устройство и отбрасывается; в рабочие 1800 обновлений не входит.",
        "Прежний inner holdout теперь входит в train: метрики на нём как на holdout не вычисляются.",
        "Original calibration/validation исключены из train, не редактировались; обе уже наблюдались раньше, это development, не независимый тест.",
        "Доля .10 и граф legacy 20/3/.50 фиксированы. Raw mAP — диагностика этой же смеси, не новая кандидатская политика.",
        "Кандидат/confidence/порог/отказ — прежний R1 equal3: candidates.csv совпадает с v25 побайтно.",
        "Три файла экспорта: tasks/<split>_<system>/export/. Контроль 2048D, новая система 2432D; реальные unit-блоки с явными slices.",
        "model_provenance.json содержит происхождение новых весов; NPY replay проверяет точные решения. Первая половина системы v25 не переобучается.",
        "Resume сохраняет optimizer/RNG каждые 50 шагов; логи — каждые 25. На прерывании повторяется только несохранённый остаток.",
        "Отработанные этапы проверяются по хешам, кэш не выдаётся за новое обучение или замер скорости. Ограничения времени нет.",
        "Проверены batch 1/8/16/32, порядок/удаление query и неизменность inference-буферов; допуски не увеличивались.",
        "Автоматического promotion, ONNX-переноса в продукт, GPU/Linux benchmark и нового порога нет. Рабочий MVP_fusion_v25 остаётся активным.",
        "NiVe, OCR, изменение bbox и дополнительные metadata-входы не используются; junk/top-10 и остаточный номерной сигнал остаются открытыми ограничениями."]
    text = "\n".join(lines) + "\n"
    path = c["output"] / "REPORT.md"
    if path.exists() and path.read_text(encoding="utf-8") != text: raise ValueError("Completed report changed")
    if not path.exists():
        pending = path.with_suffix(".md.tmp")
        pending.write_text(text, encoding="utf-8"); pending.replace(path)


def run(c, *, allow_outer=False):
    if not allow_outer: raise ValueError("Explicit allow_outer=True is required")
    with base.old.run_lock(c["output"]):
        check_inputs(c)
        completion = c["output"] / "complete.json"
        if completion.exists():
            saved = base.old.load_json(completion)
            if saved["signature"] != c["signature"]: raise ValueError("Completed refit fingerprint changed")
            base.verify_files({str(c["output"] / p): h for p, h in saved["files"].items()})
            print("Verified completed refit; no new training or measurements", flush=True)
            return base.old.load_json(c["output"] / "results.json")
        night.disk_guard(c["output"], c["settings"])
        smoke = night.runtime_smoke(c)
        trained = night.train_until(c, c["trials"][0], c["settings"]["final_steps"])
        chosen = freeze_candidate(c, trained)
        print("FROZEN: Full T12 final checkpoint, weight 10%; now evaluating original outer splits", flush=True)
        reports = {}
        probe = None
        for split in ("calibration", "validation"):
            q, g = protocol_rows(c, split)
            original, _ = baseline_features(c, split, q, g)
            extra = features(c, split)
            if split == "calibration": probe = stream_probe(c, original, extra)
            reports[split] = {}
            for name in CASES:
                values = original if name == "V25_control" else np.concatenate([original, extra], axis=1)
                reports[split][name] = task(c, f"{split}_{name}",
                    lambda d, s=split, n=name, v=values: evaluate(c, s, n, v, d))
        check_inputs(c)
        result = {"status": "complete", "signature": c["signature"], "training": trained, "frozen_candidate": chosen,
            "device_smoke": smoke, "stream_probe": probe, "evaluations": reports,
            "paired_validation": previous.paired_changes(reports["validation"]["V25_control"], reports["validation"]["Full_T12_w10"]),
            "optimizer_updates": trained["step"], "disposable_smoke_updates": len(smoke["updates"]),
            "inference_bn_updates": 0, "threshold_fit": False, "candidate_unchanged": True,
            "promoted": False, "protected_unchanged": True, "inner_evaluation": False,
            "scope": "one predeclared full-train checkpoint/mixture, development comparison only"}
        write_report(c, result)
        base.old.freeze_json(c["output"] / "results.json", result)
        base.old.freeze_json(completion, {"signature": c["signature"], "files": {
            str(p.relative_to(c["output"])): base.sha256(p) for p in c["output"].rglob("*")
            if p.is_file() and p != completion and p.name not in {".lock", ".run.lock"}
            and not p.name.endswith(".tmp") and not any(part.startswith(".pending_") for part in p.parts)}})
    return result
