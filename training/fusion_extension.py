"""v26: extend ranking fusion beyond v25's boundary; candidates remain frozen."""
import time
from pathlib import Path

from backend.core import ROOT, DATASET, read_rows, sha256
from training import map_search as previous
from training.audit import digest
from training.stage6 import write_json

old = previous.old
VARIANT = ROOT/"OSNet-AIN-x1.0/variant_26_fusion_extension"
BASELINE = {"name": "r1w50_k20_q3_l50", "k1": 20, "k2": 3, "lambda": .5, "r1_weight": .5}


def systems():
    grid = [{"name": f"r1w{round(w*100):02d}_k{k1}_q3_l{round(lam*100):02d}",
             "k1": k1, "k2": 3, "lambda": lam, "r1_weight": w}
            for w in (.4, .5, .6, .7, .8, .9, 1.) for k1 in (10, 20, 30) for lam in (.4, .5, .65, .75)]
    return [dict(BASELINE)]+[s for s in grid if s != BASELINE]


def prepare(run_name="fusion_extension_v1", *, source_run="map_search_v1", dataset=DATASET):
    source = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, result = (old.read(source/n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    source_context = {"output": source, "signature": signature, "manifest": sm}
    if (sm["version"] != 25 or result["status"] != "complete" or result["signature"] != signature
            or not result["protected_unchanged"] or result["optimizer_updates"] or result["encoder_forwards"]
            or result["threshold_fit"] or result["selection"] != previous.read_selection(source_context)
            or result["selection"]["selected"] != BASELINE):
        raise old.IntegrityError("Need completed unchanged v25 and its frozen 50/50 winner")
    protected = dict(sm["protected"])
    for path in source.glob("tasks/*/complete.json"):
        protected.update(previous.previous.search.verify_task_source(source, signature, path))
    for split, key in (("calibration", "calibration"), ("validation", "evaluations")):
        for name, report in result[key].items():
            if report != old.read(source/"tasks"/f"{split}_{name}"/"result.json"):
                raise old.IntegrityError("v25 aggregate differs from its protected task")
    for name in ("manifest.json", "results.json", "frozen_selection.json"):
        protected[str(source/name)] = sha256(source/name)
    plan = {"source_directory": str(source), "source_signature": signature, "baseline": BASELINE,
            "reason": "v25 winner reached the highest tested R1 weight; extend calibration search to 100%",
            "validation": "current v25 baseline plus one calibration winner; no automatic promotion"}
    manifest = {**sm, "version": 26, "fusion_extension_plan": plan, "protected": protected,
                "map_search_plan": {**sm["map_search_plan"], "systems": systems(),
                                    "selection": "maximum calibration mAP@10; ties keep v25 first"},
                "source_sha256": {**sm["source_sha256"], "training/fusion_extension.py": sha256(Path(__file__))}}
    source24 = Path(sm["map_search_plan"]["source_directory"])
    source22 = Path(sm["dual_role_plan"]["source_directory"])
    ctx = {"output": old.seeds.run_directory(VARIANT, run_name), "manifest": manifest, "signature": digest(manifest),
           "protected": protected, "source_output": source22, "source_manifest": old.read(source22/"manifest.json"),
           "v24_output": source24, "v25_output": source, "profile_path": source24/"dual_role_profile.json",
           "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv")}
    previous.previous.previous.validate_protocols(ctx)
    previous.previous.frozen_profile(ctx)
    old.review.check_inputs(ctx, rehash=True)
    old.review.check_other_runs(ctx)
    with old.review.old.run_lock(ctx["output"]):
        old.review.old.freeze_json(ctx["output"]/"manifest.json", manifest)
    return ctx


def evaluate(context, split, spec, values, directory):
    report = previous.evaluate(context, split, spec, values, directory)
    if spec == BASELINE:
        source = context["v25_output"]
        task = source/"tasks"/f"{split}_{BASELINE['name']}"
        previous.previous.search.verify_task_source(source,
            context["manifest"]["fusion_extension_plan"]["source_signature"], task/"complete.json")
        expected = old.read(task/"result.json")
        for key in ("ranking", "candidates", "threshold", "protocol_sha256"):
            if report[key] != expected[key]:
                raise old.IntegrityError(f"Current v25 baseline changed: {key}")
        if split == "validation":
            for name in ("submission.csv", "candidates.csv", "embeddings.npy"):
                if sha256(directory/"export"/name) != sha256(task/"export"/name):
                    raise old.IntegrityError(f"Current v25 baseline export changed: {name}")
    return report


def write_report(context, result):
    lines = ["# v26 — расширение смеси после v25", "", f"Статус: {result['status']}",
             "84 фиксированных варианта: доля R1 40/50/60/70/80/90/100%, k1 10/20/30, k2=3, λ 0.40/0.50/0.65/0.75.",
             "Выбор только по calibration mAP; при равенстве сохраняется текущий v25 (50/50, 20/3/0.50).", "",
             "| Validation: текущий v25 и выбор | mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
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
              "Это следующий адаптивный development-поиск на уже наблюдавшихся данных, не независимый hidden test.",
              "Сетка зафиксирована до нового запуска; validation только для v25 и одного calibration-победителя.",
              "Никакого обучения, encoder inference, подбора порога, правок bbox или обмена между query.",
              "Кандидаты/отказы/confidence неизменны; CSV кандидатов побайтно совпадает с v24/v25.",
              "Реальные float32-эмбеддинги 2048, query затем gallery; отдельная конфигурация ranking, NPY replay проверен.",
              "MVP не переключается автоматически. При проигрыше сохраняем конкретный v25, а не средний рецепт.",
              f"Источники неизменны: {result.get('protected_unchanged', False)}. Время текущего вызова: {result.get('elapsed_seconds', 0)/60:.1f} мин."]
    (context["output"]/"REPORT.md").write_text("\n".join(lines)+"\n")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True required for development evaluation")
    if previous.previous.runtime() != context["manifest"]["analysis_runtime"]:
        raise old.IntegrityError("Runtime changed; use the original research environment")
    queue = old.Queue(context, wall_hours=None)
    result = {"signature": context["signature"], "status": "running", "calibration": {}, "evaluations": {},
              "optimizer_updates": 0, "encoder_forwards": 0, "threshold_fit": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            values = previous.previous.load_vectors(context, "calibration")
            specs = context["manifest"]["map_search_plan"]["systems"]
            for index, spec in enumerate(specs, 1):
                print(f"\nCALIBRATION {index}/{len(specs)} | elapsed {time.monotonic()-queue.started:.1f}s", flush=True)
                result["calibration"][spec["name"]] = queue.task(f"calibration_{spec['name']}",
                    lambda d: evaluate(context, "calibration", spec, values, d))
            frozen = previous.selection(context, result["calibration"])
            old.review.old.freeze_json(context["output"]/"frozen_selection.json", frozen)
            result["selection"] = previous.read_selection(context)
            print(f"\nFROZEN: {frozen['selected']['name']}; validation v25 + winner only", flush=True)
            values = previous.previous.load_vectors(context, "validation")
            for spec in frozen["evaluations"]:
                result["evaluations"][spec["name"]] = queue.task(f"validation_{spec['name']}",
                    lambda d: evaluate(context, "validation", spec, values, d))
            if any(r is None for r in result["evaluations"].values()):
                raise ValueError("Incomplete validation; inspect task error and resume")
            result["selected_vs_v25"] = previous.previous.previous.paired_delta(
                result["evaluations"][specs[0]["name"]], result["evaluations"][frozen["selected"]["name"]])
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
