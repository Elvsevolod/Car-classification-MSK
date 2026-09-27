"""v33: one paired .025 auxiliary-weight correction; immutable v32 trainer and results."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import copy
from dataclasses import asdict
from pathlib import Path
import platform
import re
import time

import numpy as np
import torch

from training import nive_mixed as base

VARIANT = base.ROOT / "OSNet-AIN-x1.0/variant_33_nive_low_aux"
ARMS = base.ARMS


def assert_matched(previous, current):
    """Only loss weight and experiment label may differ in the scientific setup."""
    expected = {**previous["plan"], "experiment": "v33_nive_low_aux_primary_v1", "aux_weight": .025}
    if previous["plan"]["aux_weight"] != .25 or current["plan"] != expected:
        raise ValueError("Only the registered .25 -> .025 paired correction is allowed")
    for key in ("parent", "variant", "config", "inner", "draws", "nive", "sampling", "runtime", "baseline"):
        if previous[key] != current[key]:
            raise ValueError(f"Matched experiment changed: {key}")
    expected_policy = {**previous["policy"],
                       "gradients": "L_main + .025 L_aux; two backwards, exactly one optimizer step"}
    if current["policy"] != expected_policy:
        raise ValueError("BN, selection, fusion or inference policy changed")


def prepare(run_name="low_aux_v1", device="mps"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple new RUN_NAME")
    if device not in {"cpu", "mps", "cuda"} or (device == "mps" and not torch.backends.mps.is_available()) or (
            device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested device unavailable (no fallback): {device}")
    settings_path = VARIANT / "configs/low_aux_v1.json"
    settings = base.old.load_json(settings_path)
    if settings["aux_weight"] != .025 or settings["corrections_remaining_after_this_run"] != 0:
        raise ValueError("This is one fixed correction, not another sweep")
    previous_dir = base.ROOT / settings["previous_run"]
    base.verify_files({str(previous_dir / "manifest.json"): settings["previous_manifest_sha256"],
                       str(previous_dir / "results.json"): settings["previous_results_sha256"]})
    previous = base.old.load_json(previous_dir / "manifest.json")
    prior_result = base.old.load_json(previous_dir / "results.json")
    if prior_result["status"] != "complete" or prior_result["signature"] != base.digest(previous):
        raise ValueError("The .25 baseline is incomplete or has different provenance")
    print("PREFLIGHT: verifying frozen v32, local images and v25 (read-only)", flush=True)
    base.verify_files(previous["protected"])
    base.verify_files(previous["source_sha256"])
    for arm in ARMS:
        base.verify_files({c["path"]: c["sha256"] for c in prior_result["training"][arm]["checkpoints"].values()})
        base.verify_files({str(previous_dir / "exports" / arm / p): h
                           for p, h in prior_result["exports"][arm]["files"].items()})
    plan = {**previous["plan"], "experiment": settings["experiment"], "aux_weight": settings["aux_weight"]}
    runtime = {"device": device, "torch": str(torch.__version__), "numpy": np.__version__,
               "python": platform.python_version(), "platform": platform.platform(),
               "torch_threads": torch.get_num_threads(), "cuda": torch.version.cuda,
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if runtime != previous["runtime"]:
        raise ValueError("Use the same device/runtime as v32; do not confound the loss-weight comparison")
    dataset, nive_root = base.DATASET, base.ROOT / "NiVe1303"
    rows, split = base.read_rows(dataset / "train.csv"), base.old.load_json(base.ARTIFACTS / "splits.json")
    _, _, allowed, parent_path = base.parent_provenance(plan, rows, split)
    if str(parent_path) != previous["parent"]["path"] or sorted(allowed) != previous["parent"]["train_ids"]:
        raise ValueError("Parent no longer matches the completed paired pilot")
    external, inventory = base.data.audit_nive(nive_root, split["frame_sha256"].values())
    if inventory["files_fingerprint"] != previous["nive"]["files_fingerprint"]:
        raise ValueError("NiVe differs from the .25 pilot")
    audit_path = previous_dir / "domain_audit.json"
    audit = base.old.load_json(audit_path)
    fingerprint = base.digest({"frames": split["frame_sha256"], "nive": inventory["files_fingerprint"],
                               "allowed": sorted(allowed), "code": base.sha256(Path(base.data.__file__))})
    if audit["input_fingerprint"] != fingerprint:
        raise ValueError("Saved near-duplicate audit no longer matches the images/split/auditor")
    review = previous_dir / "near_duplicate_review.json"
    external, excluded = base.data.reviewed_external(external, audit, base.old.load_json(review) if review.exists() else None)
    target, external = base.data.label_domains(rows, external, allowed)
    if excluded != previous["nive"]["excluded"] or base.digest(external) != previous["nive"]["used_train_fingerprint"]:
        raise ValueError("External train rows changed")
    config, variant = base.ExperimentConfig(**previous["config"]), base.Ablation(**previous["variant"])
    main = list(base.StepPKBatchSampler(target, config, plan["joint_updates"] + plan["tail_updates"]))
    main_ids = [[target[i]["image_id"] for i in batch] for batch in main]
    auxiliary = {arm: base.data.cyclic_schedule(target if arm == ARMS[0] else external, plan["joint_updates"],
                 config.identities_per_batch, config.images_per_identity, plan["seed"] + 101,
                 forbidden=main_ids if arm == ARMS[0] else None) for arm in ARMS}
    sampling = {**previous["sampling"], "main_sha256": base.digest(main),
                "aux_sha256": {a: base.digest(b) for a, b in auxiliary.items()},
                "planned_coverage": {a: base.data.coverage(target if a == ARMS[0] else external, b)
                                     for a, b in auxiliary.items()}}
    output = VARIANT / "runs" / run_name
    if not output.resolve().is_relative_to((VARIANT / "runs").resolve()):
        raise ValueError("Output escapes v33")
    protected = dict(previous["protected"])
    protected.update({str(p): base.sha256(p) for p in previous_dir.rglob("*") if p.is_file()})
    sources = {**previous["source_sha256"], str(Path(__file__).resolve()): base.sha256(__file__),
               str(settings_path): base.sha256(settings_path)}
    manifest = {**copy.deepcopy(previous), "version": 33, "plan": plan, "config": asdict(config),
                "variant": asdict(variant), "runtime": runtime, "sampling": sampling,
                "policy": {**previous["policy"],
                           "gradients": "L_main + .025 L_aux; two backwards, exactly one optimizer step"},
                "protected": protected, "source_sha256": sources,
                "previous_run": {"path": str(previous_dir), "signature": prior_result["signature"], **settings},
                "audit_reuse": {"path": str(audit_path), "sha256": base.sha256(audit_path),
                                "note": "same byte-verified data; reused audit, not a new visual examination"}}
    assert_matched(previous, manifest)
    with base.old.run_lock(output):
        base.old.freeze_json(output / "manifest.json", manifest)
    print("PREFLIGHT OK: same parent, data, batches, LR, BN, draws; auxiliary .025 in BOTH arms", flush=True)
    return {"output": output, "manifest": manifest, "signature": base.digest(manifest), "plan": plan,
            "rows": rows, "target": target, "external": external, "config": config, "variant": variant,
            "dataset": dataset, "nive_root": nive_root, "device": torch.device(device),
            "main_schedule": main, "aux_schedules": auxiliary, "masks": {}}


def previous_results(context):
    return base.old.load_json(Path(context["manifest"]["previous_run"]["path"]) / "results.json")


def assert_reference(context, report, vectors):
    directory = Path(context["manifest"]["previous_run"]["path"]) / "evaluation/N_ref"
    if (base.old.load_json(directory / "order.json") != [r["image_id"] for r in base.development_rows(context)]
            or not np.array_equal(vectors, np.load(directory / "features.npy", allow_pickle=False))
            or report != previous_results(context)["reference"]):
        raise ValueError("Fresh unchanged R1 does not reproduce v32 reference; do not relax tolerances")


def compare_previous(previous, current):
    report = {}
    for arm in ARMS:
        old_step, new_step = previous["selection"][arm], current["selection"][arm]
        report[arm] = {"previous_selected_step": old_step, "current_selected_step": new_step,
                      "selected_vs_previous": base.paired_ap(previous["evaluations"][arm][old_step],
                                                             current["evaluations"][arm][new_step]),
                      "mixture_vs_previous": base.paired_ap(previous["fixed_half_reference_mixtures"][arm],
                                                            current["fixed_half_reference_mixtures"][arm]),
                      "same_step_delta_map": {s: v["mean_map"] - previous["evaluations"][arm][s]["mean_map"]
                                              for s, v in current["evaluations"][arm].items()}}
    return report


def report_text(context, result, previous):
    plan = context["plan"]
    lines = ["# v33 — один N0/N1 повтор: auxiliary 0.025", "",
             "Статус: complete. Исходная validation и рабочий v25 не изменены.",
             "Parent, seed, данные, порядок пачек/аугментаций, LR, BN, updates и граф сохранены из v32.",
             "", "| Вариант | Update | Raw mAP | Fixed graph mAP |", "|---|---:|---:|---:|",
             f"| N_ref | 0 | {result['reference']['mean_raw_map']:.6f} | {result['reference']['mean_map']:.6f} |"]
    for arm in ARMS:
        step = result["selection"][arm]
        for label, item in ((arm, result["evaluations"][arm][step]),
                            (f"50% N_ref + 50% {arm}", result["fixed_half_reference_mixtures"][arm])):
            lines.append(f"| {label} | {step} | {item['mean_raw_map']:.6f} | {item['mean_map']:.6f} |")
    lines += ["", "## Сравнение с первым опытом (0.25)", "",
              "| Ветка | Старый лучший step | Новый лучший step | Δ single, п.п. | Δ смеси, п.п. |",
              "|---|---:|---:|---:|---:|"]
    for arm, item in result["comparison_to_v32"].items():
        lines.append(f"| {arm} | {item['previous_selected_step']} | {item['current_selected_step']} | "
                     f"{100*item['selected_vs_previous']['delta_map']:+.4f} | {100*item['mixture_vs_previous']['delta_map']:+.4f} |")
    lines += ["", "## Все заранее назначенные точки", "",
              "| Ветка | Update | Старый graph (.25) | Новый raw (.025) | Новый graph (.025) |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for step, item in result["evaluations"][arm].items():
            prior = previous["evaluations"][arm][step]
            lines.append(f"| {arm} | {step} | {prior['mean_map']:.6f} | {item['mean_raw_map']:.6f} | {item['mean_map']:.6f} |")
    lines += ["", f"Рекомендация screening: **{result['decision']}**, не разрешение релиза.",
              "Если выбран update 0, этот encoder не обучался на NiVe; его смесь с N_ref не доказывает пользу внешних данных.",
              "Повтор слабее старого N0/его смеси — не новый лучший результат, даже если стал лучше прежнего N1.",
              "", f"Бюджет на ветвь: {plan['joint_updates']} joint + {plan['tail_updates']} target-only updates.",
              f"Loss в обеих ветвях: L_main + {plan['aux_weight']} L_aux. Это не доля итогового градиента.",
              "BN-политика не менялась; уменьшение веса loss не уменьшает число обновлений BN на NiVe.",
              "Нормы градиентов — training/*/history.json; per-query AP/top-10 — evaluation/*/metrics.json.",
              "Primary draws разделяют identity. Отбор конкретного checkpoint — адаптивный development, не hidden-test оценка.",
              "Граф/порог/веса смеси не подбирались. Head-free export — исследовательский, без новой candidate-policy.",
              "Это единственная адресная коррекция; нового автоматического поиска и promotion нет."]
    return "\n".join(lines) + "\n"


def run_experiment(context):
    base.check_inputs(context)
    output, started = context["output"], time.perf_counter()
    with base.old.run_lock(output):
        reference, ref_vectors = base.feature_task(context, "N_ref", {
            k: context["manifest"]["parent"][k] for k in ("path", "sha256")})
        assert_reference(context, reference, ref_vectors)
        previous = previous_results(context)
        summaries, evaluations, vectors = {}, {}, {}
        for arm in ARMS:
            print(f"STAGE {arm}: auxiliary={context['plan']['aux_weight']}; new optimizer from original R1", flush=True)
            summaries[arm] = base.train_arm(context, arm)
            evaluations[arm], vectors[arm] = {}, {}
            for step in context["plan"]["checkpoints"]:
                metric, feature = base.feature_task(context, f"{arm}_{step:05d}", summaries[arm]["checkpoints"][str(step)], arm)
                evaluations[arm][str(step)], vectors[arm][str(step)] = metric, feature
                if step == 0 and not np.array_equal(feature, ref_vectors):
                    raise ValueError("Both arms must start from identical untouched R1 embeddings")
            base.check_inputs(context)
        selected = base.select_steps(evaluations)
        base.old.freeze_json(output / "frozen_selection.json", {"signature": context["signature"], "steps": selected,
                             "criterion": "mean fixed graph mAP on three primary draws; ties prefer fewer updates"})
        mixtures, exports = {}, {}
        for arm in ARMS:
            combined = base.normalize(np.concatenate([ref_vectors, vectors[arm][selected[arm]]], axis=1))
            mixtures[arm] = base.score_features(context, combined, base.development_rows(context))
            exports[arm] = base.export_encoder(context, arm, summaries[arm]["checkpoints"][selected[arm]])
        a, b = (evaluations[arm][selected[arm]] for arm in ARMS)
        single_gain = b["mean_map"] > max(reference["mean_map"], a["mean_map"]) + 1e-12
        mixture_gain = mixtures[ARMS[1]]["mean_map"] > max(reference["mean_map"], a["mean_map"], mixtures[ARMS[0]]["mean_map"]) + 1e-12
        results = {"status": "complete", "signature": context["signature"], "reference": reference,
                   "training": summaries, "evaluations": evaluations, "selection": selected,
                   "fixed_half_reference_mixtures": mixtures, "exports": exports,
                   "N1_vs_N0": base.paired_ap(a, b), "mixture_N1_vs_N0": base.paired_ap(mixtures[ARMS[0]], mixtures[ARMS[1]]),
                   "decision": "continue" if single_gain or mixture_gain else "stop",
                   "decision_basis": {"single_gain": single_gain, "mixture_gain": mixture_gain,
                                      "corrections_remaining": 0, "note": "screening only; inspect comparison_to_v32"},
                   "protected_unchanged": True, "threshold_fit": False, "original_validation_evaluated": False,
                   "promoted": False, "elapsed_this_call_seconds": time.perf_counter() - started}
        results["comparison_to_v32"] = compare_previous(previous, results)
        base.check_inputs(context)
        if (output / "results.json").exists():
            saved = base.old.load_json(output / "results.json")
            if {k: v for k, v in saved.items() if k != "elapsed_this_call_seconds"} != {
                    k: v for k, v in results.items() if k != "elapsed_this_call_seconds"}:
                raise ValueError("Completed result changed")
            results = saved
        text = report_text(context, results, previous)
        path = output / "REPORT.md"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError("Completed report changed")
        if not path.exists():
            temp = output / "REPORT.md.tmp"
            temp.write_text(text, encoding="utf-8"); temp.replace(path)
        if not (output / "results.json").exists():
            base.write_json(output / "results.json", results)
    return results


def technical_smoke(context):
    """Disposable tiny schedule uses the real trainer, including .025, tail and resume slots."""
    base.check_inputs(context)
    with base.old.run_lock(context["output"]):
        plan = {**context["plan"], "joint_updates": 2, "tail_updates": 1, "save_interval": 1,
                "warmup_updates": 1, "checkpoints": [0, 2, 3]}
        smoke = {**context, "output": context["output"] / "technical_smoke", "plan": plan,
                 "signature": base.digest({"full_signature": context["signature"], "technical_only": plan}),
                 "main_schedule": context["main_schedule"][:3],
                 "aux_schedules": {a: s[:2] for a, s in context["aux_schedules"].items()}}
        reports = {}
        for arm in ARMS:
            trained = base.train_arm(smoke, arm)
            exported = base.export_encoder(smoke, arm, trained["checkpoints"]["3"])
            reports[arm] = {"training": trained, "export": exported}
        base.check_inputs(context)
        result = {"signature": context["signature"], "status": "passed", "aux_weight": plan["aux_weight"],
                  "arms": reports, "note": "Three disposable updates per arm; not full training or quality evidence"}
        base.old.freeze_json(smoke["output"] / "report.json", result)
    return result
