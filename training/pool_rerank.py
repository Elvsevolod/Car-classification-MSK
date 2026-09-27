"""v30 Run All: four frozen raw-pool comparisons, calibration selection, no MVP edits."""
import time
from pathlib import Path

import numpy as np

from backend.core import ROOT, DATASET, read_rows, sha256
from training import family_finalists as previous, map_search as search
from training import final_model_comparison as comparison
from training.audit import digest
from training.stage6 import write_json
from training.pool_rerank_inference import BASELINE, rank, systems

old, source, dual = previous.old, previous.source, previous.dual
VARIANT = ROOT/"OSNet-AIN-x1.0/variant_30_pool_rerank"


def completed_source(directory):
    sm, result = (old.read(directory/n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    ctx = {"output": directory, "manifest": sm, "signature": signature}
    if (sm["version"] != 29 or result["status"] != "complete" or result["signature"] != signature
            or not result["protected_unchanged"] or result["optimizer_updates"] or result["threshold_fit"]
            or result["promoted"] or set(result["evaluations"]) != {s["name"] for s in previous.systems()}
            or result["frozen_comparison"] != previous.read_frozen(ctx)):
        raise old.IntegrityError("Need completed unchanged v29 with the original v25 control")
    protected = dict(sm["protected"])
    tasks = {"verify_calibration": result["calibration_replay"],
             "flip_validation": result["features"]["validation"],
             **{f"validation_{name}": r for name, r in result["evaluations"].items()}}
    for name, report in tasks.items():
        task = directory/"tasks"/name
        protected.update(source.search.verify_task_source(directory, signature, task/"complete.json"))
        if not report or report != old.read(task/"result.json"):
            raise old.IntegrityError("Source aggregate differs from its protected task")
    if not result["calibration_replay"]["passed"]:
        raise old.IntegrityError("Source calibration replay did not pass")
    for name in ("manifest.json", "results.json", "frozen_comparison.json"):
        protected[str(directory/name)] = sha256(directory/name)
    return sm, signature, protected


def check_inputs(context):
    manifest = context["manifest"]
    if (context["signature"] != digest(manifest) or manifest["version"] != 30
            or manifest["pool_rerank_plan"]["systems"] != systems()
            or manifest["map_search_plan"]["systems"] != systems()):
        raise old.IntegrityError("Only the four frozen v30 systems are allowed; configuration changed")
    previous.previous.check_inputs(context)


def prepare(run_name="pool_rerank_v1", *, source_run="family_finalists_v1", dataset=DATASET):
    directory = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, signature, protected = completed_source(directory)
    plan = {"source_directory": str(directory), "source_signature": signature, "systems": systems(),
            "pool": "top K of original normalized MVP/R1 50:50 cosine fusion; K=10/20/50",
            "reranking": "restrict full static-gallery v25 k20/k3/lambda.50 order to the raw pool",
            "tail": "unchanged raw-fusion order outside the pool; no outsiders enter the pool",
            "candidate": "original raw R1 candidate/cosine/refusal and frozen threshold",
            "selection": "maximum calibration mAP over all four systems; ties retain v25 first",
            "validation": "v25 and one frozen calibration winner only; no automatic promotion",
            "scope": "post-hoc search on observed development data, not independent hidden-test evidence",
            "optimizer_updates": 0, "encoder_forwards": 0, "threshold_fit": False, "wall_time_limit": None}
    modules = ("training/pool_rerank.py", "training/pool_rerank_inference.py", "training/map_inference.py")
    manifest = {**sm, "version": 30, "pool_rerank_plan": plan, "protected": protected,
                "map_search_plan": {**sm["map_search_plan"], "systems": systems(), "selection": plan["selection"]},
                "source_sha256": {**sm["source_sha256"], **{p: sha256(ROOT/p) for p in modules}}}
    source22 = Path(sm["dual_role_plan"]["source_directory"])
    context = {"output": old.seeds.run_directory(VARIANT, run_name), "manifest": manifest, "signature": digest(manifest),
               "protected": protected, "source_output": source22, "source_manifest": old.read(source22/"manifest.json"),
               "profile_path": Path(sm["map_search_plan"]["source_directory"])/"dual_role_profile.json",
               "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv")}
    comparison.validate_protocols(context)
    source.frozen_profile(context)
    check_inputs(context)
    old.review.check_other_runs(context)
    with old.review.old.run_lock(context["output"]):
        old.review.old.freeze_json(context["output"]/"manifest.json", manifest)
    print("Preflight OK: v25 + raw top10/20/50; cached embeddings; no training or automatic promotion", flush=True)
    return context


def reference(context, split):
    if split == "calibration":
        return previous.reference(context, split)  # Protected v28 control replayed by v29.
    plan = context["manifest"]["pool_rerank_plan"]
    directory = Path(plan["source_directory"])
    task = directory/"tasks"/"validation_V25_control"
    source.search.verify_task_source(directory, plan["source_signature"], task/"complete.json")
    return old.read(task/"result.json"), task/"export"


def evaluate(context, split, spec, values, directory):
    if split not in ("calibration", "validation") or spec not in systems():
        raise ValueError("Use only the original development splits and four frozen systems")
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
            "ranking_source_sha256": context["manifest"]["source_sha256"]["training/pool_rerank_inference.py"]})
        replay = np.load(npy, allow_pickle=False)
        if dual.policy.predictions(q, g, rank(replay, len(q), spec), threshold, "raw_top1") != decisions:
            raise old.IntegrityError("NPY replay changed decisions")
        report.update(export=str(export), **dual.policy.query_diagnostics(q, g, ranked))
    print(f"{split} {spec['name']}: mAP={metrics['ranking']['mAP@10']:.6f}; R1 candidates unchanged", flush=True)
    return report


def write_report(context, result):
    lines = ["# v30 — ограничение реранкинга исходным пулом", "", f"Статус: {result['status']}",
             "Четыре варианта: v25, raw top-10/20/50; fusion 50:50, граф 20/3/λ0.50 неизменны.",
             "Пул выбирается по raw cosine fusion, не по отдельному raw R1-кандидату.",
             "Внутри пула — порядок полного графа v25; за его пределами — исходный raw-порядок.",
             "Все четыре сравнения на calibration; на validation — контроль и один замороженный выбор.", "",
             "## Calibration", "", "| Вариант | mAP@10 |", "|---|---:|"]
    for name, r in result["calibration"].items():
        lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} |" if r else f"| {name} | FAILED |")
    lines += ["", f"Выбор: {result.get('selection', {}).get('selected', {}).get('name', 'не завершён')}", "",
              "## Validation", "", "| Вариант | mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
    for name, r in result["evaluations"].items():
        lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['ranking']['Rank-1']:.6f} | "
                     f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |" if r else f"| {name} | FAILED | — | — | — |")
    if result.get("selected_vs_v25"):
        d = result["selected_vs_v25"]
        lines += ["", f"ΔmAP против v25: {d['mAP_delta']:+.6f}. AP лучше/хуже/равно: {d['improved']}/{d['worsened']}/{d['unchanged']}."]
    lines += ["", "## Ограничения", "",
              "Адаптивный поиск после изучения ошибок на development-данных, не независимый тест и не обещание hidden-test прироста.",
              "Веса, эмбеддинги, кандидаты, confidence, порог, bbox, evaluator и рабочий MVP не изменяются.",
              "Обучения, нового извлечения признаков и автоматического продвижения нет. Каждый query независим.",
              "Экспорт: десять уникальных gallery-ID даже при отказе, отказ — отсутствие строки кандидата, реальные float32-векторы.",
              "Ограничение junk/top-10 остаётся: evaluator фильтрует junk, но 11-е место из файла top-10 не восстановить.",
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
            check_inputs(context)
            values = source.load_vectors(context, "calibration")
            for index, spec in enumerate(systems(), 1):
                print(f"\nCALIBRATION {index}/4 | elapsed {time.monotonic()-queue.started:.1f}s", flush=True)
                result["calibration"][spec["name"]] = queue.task(f"calibration_{spec['name']}",
                    lambda d: evaluate(context, "calibration", spec, values, d))
            frozen = search.selection(context, result["calibration"])
            old.review.old.freeze_json(context["output"]/"frozen_selection.json", frozen)
            result["selection"] = search.read_selection(context)
            print(f"\nFROZEN: {frozen['selected']['name']}; validation v25 + winner only", flush=True)
            values = source.load_vectors(context, "validation")
            for index, spec in enumerate(frozen["evaluations"], 1):
                print(f"\nVALIDATION {index}/{len(frozen['evaluations'])} | elapsed {time.monotonic()-queue.started:.1f}s", flush=True)
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
                check_inputs(context)
                result["protected_unchanged"] = True
            except BaseException:
                result.update(status="integrity_check_failed", protected_unchanged=False)
                raise
            finally:
                write_json(context["output"]/"results.json", result)
                write_report(context, result)
    return result
