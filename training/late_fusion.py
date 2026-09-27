"""v27: calibration-only late fusion search against the retained concrete v25."""
import time
from pathlib import Path

import numpy as np

from backend.core import ROOT, DATASET, read_rows, sha256
from training import fusion_extension as v26, map_search as search
from training import dual_role_experiment as source, final_model_comparison as comparison
from training.audit import digest
from training.stage6 import write_json
from training.late_fusion_inference import BASELINE, rank, systems

old, dual = search.old, search.dual
VARIANT = ROOT/"OSNet-AIN-x1.0/variant_27_late_fusion"


def prepare(run_name="late_fusion_v1", *, source_run="fusion_extension_v1", dataset=DATASET):
    directory = old.seeds.run_directory(v26.VARIANT, source_run)
    sm, result = (old.read(directory/n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    source_context = {"output": directory, "signature": signature, "manifest": sm}
    if (sm["version"] != 26 or result["status"] != "complete" or result["signature"] != signature
            or not result["protected_unchanged"] or result["optimizer_updates"] or result["encoder_forwards"]
            or result["threshold_fit"] or result["selection"] != search.read_selection(source_context)
            or sm["map_search_plan"]["systems"][0] != v26.BASELINE):
        raise old.IntegrityError("Need completed unchanged v26 with the original v25 control")
    protected = dict(sm["protected"])
    for path in directory.glob("tasks/*/complete.json"):
        protected.update(source.search.verify_task_source(directory, signature, path))
    for split, key in (("calibration", "calibration"), ("validation", "evaluations")):
        for name, report in result[key].items():
            if report != old.read(directory/"tasks"/f"{split}_{name}"/"result.json"):
                raise old.IntegrityError("Source aggregate differs from protected task")
    for name in ("manifest.json", "results.json", "frozen_selection.json"):
        protected[str(directory/name)] = sha256(directory/name)
    plan = {"source_directory": str(directory), "source_signature": signature, "baseline": v26.BASELINE,
            "graphs": "independent MVP and R1 static galleries; k1=20, k2=3; MVP lambda=.5, R1=.5/.75",
            "fusion": "distance mean or weighted reciprocal rank with k=10/60; full gallery before top10",
            "candidate": "unchanged raw R1 cosine, candidate and frozen threshold",
            "selection": "maximum calibration mAP; ties retain v25 first",
            "validation": "v25 control and one calibration winner only; no automatic promotion",
            "scope": "adaptive search on observed development splits, not an independent hidden test",
            "rrf_source": "https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf",
            "threshold_fit": False, "encoder_forwards": 0, "optimizer_updates": 0, "wall_time_limit": None}
    modules = ("training/late_fusion.py", "training/late_fusion_inference.py", "training/map_inference.py")
    manifest = {**sm, "version": 27, "late_fusion_plan": plan, "protected": protected,
                "map_search_plan": {**sm["map_search_plan"], "systems": systems(), "selection": plan["selection"]},
                "source_sha256": {**sm["source_sha256"], **{p: sha256(ROOT/p) for p in modules}}}
    source24 = Path(sm["map_search_plan"]["source_directory"])
    source22 = Path(sm["dual_role_plan"]["source_directory"])
    ctx = {"output": old.seeds.run_directory(VARIANT, run_name), "manifest": manifest, "signature": digest(manifest),
           "protected": protected, "source_output": source22, "source_manifest": old.read(source22/"manifest.json"),
           "reference_output": directory, "profile_path": source24/"dual_role_profile.json",
           "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv")}
    comparison.validate_protocols(ctx)
    source.frozen_profile(ctx)
    old.review.check_inputs(ctx, rehash=True)
    old.review.check_other_runs(ctx)
    with old.review.old.run_lock(ctx["output"]):
        old.review.old.freeze_json(ctx["output"]/"manifest.json", manifest)
    return ctx


def reference(context, split):
    directory = context["reference_output"]
    task = directory/"tasks"/f"{split}_{v26.BASELINE['name']}"
    source.search.verify_task_source(directory, context["manifest"]["late_fusion_plan"]["source_signature"], task/"complete.json")
    return old.read(task/"result.json"), task/"export"


def evaluate(context, split, spec, values, directory):
    if split not in ("calibration", "validation"):
        raise ValueError("Use original calibration/validation only")
    if split == "validation" and spec not in search.read_selection(context)["evaluations"]:
        raise old.IntegrityError("Validation accepts only the frozen winner and v25 control")
    q, g = old.review.old.protocol_rows(context, split)
    threshold = source.frozen_profile(context)["threshold"]
    expected, source_export = reference(context, split)
    if threshold != expected["threshold"]:
        raise old.IntegrityError("Frozen R1 threshold differs from v25")
    ranked = rank(values, len(q), spec)
    metrics = dual.policy.evaluate(q, g, ranked, threshold, "raw_top1")
    decisions = dual.policy.predictions(q, g, ranked, threshold, "raw_top1")
    _, r1 = dual.unpack(values)
    raw = dual.policy.rank_vectors(r1[:len(q)], r1[len(q):], "raw")
    if (decisions[1] != dual.policy.predictions(q, g, raw, threshold, "raw_top1")[1]
            or metrics["candidates"] != expected["candidates"]):
        raise old.IntegrityError("Fixed R1 candidates/confidence/refusals changed")
    if spec == BASELINE and metrics["ranking"] != expected["ranking"]:
        raise old.IntegrityError("Current v25 ranking no longer reproduces")
    report = {"system": spec, "split": split, "threshold": threshold, **metrics,
              "protocol_sha256": digest(context["manifest"]["protocols"][split]), "candidate_unchanged": True}
    if split == "validation":
        export = directory/"export"
        saved = old.previous.verify_or_export_csv(export, q, g, ranked, threshold, "raw_top1")
        if saved != metrics or sha256(export/"candidates.csv") != sha256(source_export/"candidates.csv"):
            raise old.IntegrityError("CSV metrics or candidate bytes differ")
        npy = export/"embeddings.npy"
        if npy.exists():
            if not np.array_equal(np.load(npy, allow_pickle=False), values):
                raise old.IntegrityError("Existing embedding export changed")
        else:
            np.save(npy, values)
        if sha256(npy) != sha256(source_export/"embeddings.npy"):
            raise old.IntegrityError("Embedding bytes differ from v25")
        if spec == BASELINE and sha256(export/"submission.csv") != sha256(source_export/"submission.csv"):
            raise old.IntegrityError("v25 control ranking bytes changed")
        old.review.old.freeze_json(export/"embedding_order.json", {
            "ids": [r["image_id"] for r in q+g], "query_count": len(q), "gallery_count": len(g),
            "layout": dual.LAYOUT, "ranking": spec, "candidate": dual.ROLES["candidate"],
            "threshold": threshold, "sha256": sha256(npy),
            "ranking_source_sha256": context["manifest"]["source_sha256"]["training/late_fusion_inference.py"]})
        replay = np.load(npy, allow_pickle=False)
        if dual.policy.predictions(q, g, rank(replay, len(q), spec), threshold, "raw_top1") != decisions:
            raise old.IntegrityError("NPY replay changed decisions")
        report.update(export=str(export), **dual.policy.query_diagnostics(q, g, ranked))
    print(f"{split} {spec['name']}: mAP={metrics['ranking']['mAP@10']:.6f}; R1 candidates unchanged", flush=True)
    return report


def write_report(context, result):
    lines = ["# v27 — объединение после независимого реранкинга", "", f"Статус: {result['status']}",
             "43 фиксированных варианта: контроль v25 и 42 смеси distance/RRF10/RRF60.",
             "Доля R1: 10/25/40/50/60/75/90%; λ R1: 0.50/0.75; оба графа 20/3, λ MVP=0.50.",
             "Выбор только на calibration; при равенстве сохраняется v25.", "",
             "| Validation: v25 и выбор | mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
    for name, r in result["evaluations"].items():
        if r:
            lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['ranking']['Rank-1']:.6f} | "
                         f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |")
    if result.get("selected_vs_v25"):
        d = result["selected_vs_v25"]
        lines += ["", f"ΔmAP против v25: {d['mAP_delta']:+.6f}. AP лучше/хуже/равно: {d['improved']}/{d['worsened']}/{d['unchanged']}."]
    lines += ["", "## Calibration", "", "| Вариант | mAP@10 |", "|---|---:|"]
    for name, r in sorted(result["calibration"].items(), key=lambda x: -(x[1]["ranking"]["mAP@10"] if x[1] else -1)):
        lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} |" if r else f"| {name} | FAILED |")
    lines += ["", "## Ограничения", "",
              "Это адаптивный поиск на уже наблюдавшихся development-данных, не независимая оценка hidden test.",
              "Validation — только контроль v25 и один calibration-победитель. Автоматического продвижения нет.",
              "Порог, кандидаты, confidence, эмбеддинги, bbox и evaluator неизменны; другие query недоступны scorer.",
              "RRF использует позиции во всей gallery, с единицы; это взвешенная адаптация, не обещание выигрыша.",
              "[Источник RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).",
              f"Источники неизменны: {result.get('protected_unchanged', False)}. Время текущего вызова: {result.get('elapsed_seconds', 0)/60:.1f} мин."]
    (context["output"]/"REPORT.md").write_text("\n".join(lines)+"\n")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True required for development evaluation")
    if source.runtime() != context["manifest"]["analysis_runtime"]:
        raise old.IntegrityError("Runtime changed; use the original research environment")
    queue = old.Queue(context, wall_hours=None)
    result = {"signature": context["signature"], "status": "running", "calibration": {}, "evaluations": {},
              "optimizer_updates": 0, "encoder_forwards": 0, "threshold_fit": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            values = source.load_vectors(context, "calibration")
            specs = context["manifest"]["map_search_plan"]["systems"]
            for index, spec in enumerate(specs, 1):
                print(f"\nCALIBRATION {index}/{len(specs)} | elapsed {time.monotonic()-queue.started:.1f}s", flush=True)
                result["calibration"][spec["name"]] = queue.task(f"calibration_{spec['name']}",
                    lambda d: evaluate(context, "calibration", spec, values, d))
            frozen = search.selection(context, result["calibration"])
            old.review.old.freeze_json(context["output"]/"frozen_selection.json", frozen)
            result["selection"] = search.read_selection(context)
            print(f"\nFROZEN: {frozen['selected']['name']}; validation v25 + winner only", flush=True)
            values = source.load_vectors(context, "validation")
            for spec in frozen["evaluations"]:
                result["evaluations"][spec["name"]] = queue.task(f"validation_{spec['name']}",
                    lambda d: evaluate(context, "validation", spec, values, d))
            if any(r is None for r in result["evaluations"].values()):
                raise ValueError("Incomplete validation; inspect task error and resume")
            result["selected_vs_v25"] = comparison.paired_delta(
                result["evaluations"][BASELINE["name"]], result["evaluations"][frozen["selected"]["name"]])
            result["status"] = "complete"
        except BaseException:
            result["status"] = "incomplete"
            raise
        finally:
            result.update(events=queue.events, elapsed_seconds=time.monotonic()-queue.started)
            try:
                old.review.check_inputs(context, rehash=True)
                result["protected_unchanged"] = True
            except BaseException:
                result.update(status="integrity_check_failed", protected_unchanged=False)
                raise
            finally:
                write_json(context["output"]/"results.json", result)
                write_report(context, result)
    return result
