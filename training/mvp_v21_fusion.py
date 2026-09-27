"""v23: a bounded calibration-only fusion search over immutable v22 embeddings."""
import platform
import time
from pathlib import Path

import numpy as np

from backend.core import DATASET, ROOT, normalize, read_rows, sha256
from training import final_model_comparison as previous
from training.audit import digest
from training.stage6 import write_json

old, search = previous.old, previous.search
VARIANT = ROOT / "OSNet-AIN-x1.0/variant_23_mvp_v21_fusion"
CONTROLS = {"MVP_control": ("MVP_recalibrated", "MVP_active"),
            "V21_control": ("V21_selected_primary", "V21_selected_primary")}


def systems():
    def spec(name, weight, lam, candidate="raw_top1"):
        return {"name": name, "mvp_weight": weight, "lambda": lam, "candidate_policy": candidate,
                "dimension": 512 if weight == 1 else 1536 if weight == 0 else 2048}
    # Stable ties retain a control, then the larger MVP fraction and lambda .50.
    return [spec("MVP_control", 1., .50, "ranking_top1"), spec("V21_control", 0., .65)] + [
        spec(f"mix_mvp{int(w*100):02d}_lambda{int(lam*100):02d}", w, lam)
        for w in (.75, .50, .25) for lam in (.50, .65)]


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "device": "CPU cached arrays only"}


def source_task(context, name):
    directory = context["source_output"] / "tasks" / name
    search.verify_task_source(context["source_output"], context["manifest"]["fusion_plan"]["source_signature"],
                              directory / "complete.json")
    return old.read(directory / "result.json")


def prepare(run_name="fusion_v1", *, source_run="comparison_v1", dataset=DATASET):
    output = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, sr, frozen = (old.read(output / n) for n in ("manifest.json", "results.json", "frozen_comparison.json"))
    signature = digest(sm)
    if (sm["version"] != 22 or sr["status"] != "complete" or not sr["protected_unchanged"]
            or sr["signature"] != signature or frozen["signature"] != signature or sr["frozen"] != frozen
            or sr["optimizer_updates"] != 0 or sr["bn_updates"] != 0 or sr["promoted"]):
        raise old.IntegrityError("Need the complete, unchanged v22 comparison")
    source = {"output": output, "manifest": sm, "signature": signature}
    previous.read_frozen(source)
    protected = dict(sm["protected"])
    for path in output.glob("tasks/*/complete.json"):
        protected.update(search.verify_task_source(output, signature, path))
    for name in ("manifest.json", "results.json", "frozen_comparison.json"):
        protected[str(output / name)] = sha256(output / name)
    source_systems = {s["name"]: s for s in sm["comparison_plan"]["systems"]}
    selected = source_systems["V21_selected_primary"]
    if (source_systems["MVP_recalibrated"]["members"] != ["mvp"] or selected["dimension"] != 1536
            or selected["weights"] != [1/3]*3 or selected["head_weight"] != 0
            or selected["lambda"] != .65 or selected["candidate_policy"] != "raw_top1"):
        raise old.IntegrityError("Unexpected source MVP/v21 composition")
    components = ["mvp", *selected["members"]]
    plan = {"source_run": source_run, "source_directory": str(output), "source_signature": signature,
            "components": components, "v21_system": selected, "systems": systems(),
            "fusion": "L2(concat(sqrt(w)*MVP_512, sqrt(1-w)*V21_1536)); endpoints unchanged",
            "k1": 20, "k2": 3, "confidence": "maximum raw cosine of fused embedding",
            "selection": "calibration mAP@10, ties candidate C then frozen grid order; controls may win",
            "threshold": "calibration only; maximum C=.7F1+.3TNR, ties F1 then higher threshold",
            "validation": "selected configuration and two controls only; no validation selection",
            "scope": "post-hoc development experiment; calibration/validation were already observed",
            "optimizer_updates": 0, "encoder_forwards": 0, "wall_time_limit": None, "promoted": False}
    manifest = {**sm, "version": 23, "fusion_plan": plan, "analysis_runtime": runtime(), "protected": protected,
                "source_sha256": {**sm["source_sha256"], "training/mvp_v21_fusion.py": sha256(Path(__file__))}}
    context = {"output": old.seeds.run_directory(VARIANT, run_name), "source_output": output,
               "source_manifest": sm, "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv"),
               "manifest": manifest, "signature": digest(manifest), "protected": protected}
    previous.validate_protocols(context)
    old.review.check_inputs(context, rehash=True)
    # Verify completeness of the specific caches we will consume, never regenerate them.
    for split in ("calibration", "validation"):
        for name in components:
            source_task(context, f"features_{split}_{name}")
    old.review.check_other_runs(context)
    with old.review.old.run_lock(context["output"]):
        old.review.old.freeze_json(context["output"] / "manifest.json", manifest)
    return context


def load_vectors(context, split):
    if split not in ("calibration", "validation"):
        raise ValueError("Only original calibration/validation caches are supported")
    if split == "validation":
        read_selection(context)
    query, gallery = old.review.old.protocol_rows(context, split)
    ids = [r["image_id"] for r in query + gallery]
    features = {}
    for name in context["manifest"]["fusion_plan"]["components"]:
        item = source_task(context, f"features_{split}_{name}")
        path = context["source_output"] / "tasks" / f"features_{split}_{name}" / "features.npz"
        if (Path(item["path"]).resolve() != path.resolve() or sha256(path) != item["sha256"]
                or item["split"] != split
                or item["model"] != context["source_manifest"]["comparison_plan"]["components"][name]):
            raise old.IntegrityError("Source cache/model provenance changed")
        with np.load(path, allow_pickle=False) as data:
            if data["ids"].tolist() != ids:
                raise old.IntegrityError("Source embedding IDs/order differ from the original protocol")
            features[name] = data["vectors"]
        previous.validate_vectors(features[name], len(ids), 512)
    v21 = previous.system_vectors(context["manifest"]["fusion_plan"]["v21_system"], features)
    previous.validate_vectors(v21, len(ids), 1536)
    print(f"Loaded verified {split} embeddings: {len(query)} query + {len(gallery)} gallery; no inference", flush=True)
    return features["mvp"], v21


def fuse(mvp, v21, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1 or len(mvp) != len(v21) or not len(mvp):
        raise ValueError("Fusion requires aligned rows and weight in [0, 1]")
    previous.validate_vectors(mvp, len(mvp), 512)
    previous.validate_vectors(v21, len(mvp), 1536)
    if weight == 1:
        return mvp
    if weight == 0:
        return v21
    # Different feature spaces/dimensions: concatenate, never add coordinates.
    return normalize(np.concatenate([normalize(mvp)*np.float32(np.sqrt(weight)),
                                     normalize(v21)*np.float32(np.sqrt(1-weight))], axis=1))


def check_reference(context, split, spec, report):
    if spec["name"] not in CONTROLS:
        return
    cal_name, val_name = CONTROLS[spec["name"]]
    if split == "calibration":
        expected = source_task(context, f"calibrate_{cal_name}")
        equal = expected["selected"] == report["calibration"]["selected"]
    else:
        expected = source_task(context, f"evaluate_{val_name}")
        equal = all(expected[k] == report[k] for k in ("threshold", "ranking", "candidates", "per_query"))
        # Equal AP/top-1 can hide a different ordering of negative gallery IDs.
        for name in ("submission.csv", "candidates.csv"):
            before = context["source_output"] / "tasks" / f"evaluate_{val_name}" / "export" / name
            equal = equal and sha256(before) == sha256(Path(report["export"]) / name)
    if not equal:
        raise old.IntegrityError(f"{spec['name']} no longer reproduces v22 {split}; do not widen tolerances")


def calibrate(context, spec, features, *, split):
    if split != "calibration":
        raise ValueError("Mixture/threshold selection is calibration-only")
    query, gallery = old.review.old.protocol_rows(context, split)
    values = fuse(*features, spec["mvp_weight"])
    ranking, _ = search.rank(values[:len(query)], values[len(query):], spec["lambda"])
    cal = old.policy.calibrate_policy(query, gallery, ranking, spec["candidate_policy"], split=split)
    metrics = old.policy.evaluate(query, gallery, ranking, cal["selected"]["threshold"], spec["candidate_policy"])
    report = {"system": spec, "split": split, "protocol_sha256": digest(context["manifest"]["protocols"][split]),
              "calibration": cal, **metrics}
    check_reference(context, split, spec, report)
    print(f"  {spec['name']}: calibration mAP={metrics['ranking']['mAP@10']:.6f}, C={metrics['candidates']['C']:.6f}", flush=True)
    return report


def choose(reports, specs):
    names = [s["name"] for s in specs]
    if set(reports) != set(names) or any(reports[n] is None for n in names):
        raise ValueError("Finish every planned calibration before selection/validation")
    for spec in specs:
        report = reports[spec["name"]]
        if (report["split"] != "calibration" or report["system"] != spec
                or not np.isfinite(report["ranking"]["mAP@10"])
                or not np.isfinite(report["candidates"]["C"])):
            raise old.IntegrityError("Invalid calibration selection input")
    return max(names, key=lambda n: (reports[n]["ranking"]["mAP@10"], reports[n]["candidates"]["C"]))


def selection(context, reports):
    plan = context["manifest"]["fusion_plan"]
    winner = choose(reports, plan["systems"])
    names = list(dict.fromkeys([*CONTROLS, winner]))
    return {"signature": context["signature"], "selected": winner,
            "selection_split": "calibration", "protocol_sha256": digest(context["manifest"]["protocols"]["calibration"]),
            "evaluations": [{"system": reports[n]["system"],
                             "threshold": reports[n]["calibration"]["selected"]["threshold"]} for n in names],
            "source_signature": plan["source_signature"], "promoted": False}


def read_selection(context):
    saved = old.read(context["output"] / "frozen_selection.json")
    reports = {}
    for spec in context["manifest"]["fusion_plan"]["systems"]:
        directory = context["output"] / "tasks" / f"calibrate_{spec['name']}"
        search.verify_task_source(context["output"], context["signature"], directory / "complete.json")
        reports[spec["name"]] = old.read(directory / "result.json")
        if reports[spec["name"]]["protocol_sha256"] != digest(context["manifest"]["protocols"]["calibration"]):
            raise old.IntegrityError("Calibration protocol changed")
    if saved != selection(context, reports):
        raise old.IntegrityError("Frozen selection/threshold differs from completed calibration")
    return saved


def evaluate(context, case, features, directory):
    if case not in read_selection(context)["evaluations"]:
        raise old.IntegrityError("Only the selected mixture and controls may evaluate validation")
    spec, threshold = case["system"], case["threshold"]
    query, gallery = old.review.old.protocol_rows(context, "validation")
    values = fuse(*features, spec["mvp_weight"])
    previous.validate_vectors(values, len(query)+len(gallery), spec["dimension"])
    ranking, _ = search.rank(values[:len(query)], values[len(query):], spec["lambda"])
    output = directory / "export"
    metrics = old.previous.verify_or_export_csv(output, query, gallery, ranking, threshold, spec["candidate_policy"])
    if metrics != old.policy.evaluate(query, gallery, ranking, threshold, spec["candidate_policy"]):
        raise old.IntegrityError("CSV/in-memory metrics differ")
    path = output / "embeddings.npy"
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), values):
            raise old.IntegrityError("Export embeddings changed")
    else:
        pending = output / "embeddings.pending.npy"
        np.save(pending, values)
        pending.replace(path)
    old.review.old.freeze_json(output / "embedding_order.json", {
        "ids": [r["image_id"] for r in query+gallery], "query_count": len(query), "gallery_count": len(gallery),
        "dimension": spec["dimension"], "sha256": sha256(path), "fusion": context["manifest"]["fusion_plan"]["fusion"]})
    _, accepted = old.policy.predictions(query, gallery, ranking, threshold, spec["candidate_policy"])
    report = {**case, **metrics, **old.policy.query_diagnostics(query, gallery, ranking),
              "accepted": len(accepted), "refused": len(query)-len(accepted), "export": str(output),
              "quality_points_out_of_55": 45*metrics["ranking"]["mAP@10"]+10*metrics["candidates"]["C"]}
    check_reference(context, "validation", spec, report)
    return report


def write_report(context, result):
    lines = ["# v23 — MVP + v21 на сохранённых эмбеддингах", "", f"Статус: **{result['status']}**.",
             "Нового обучения и encoder inference нет. Calibration/validation уже наблюдались: это development-поиск, не независимый тест.",
             "", "## Calibration: вся заранее фиксированная сетка", "",
             "Выбор по mAP@10; при равенстве C, затем исходный порядок сетки. Каждый порог — по calibration C.",
             "", "| Система | mAP@10 | C | Порог |", "|---|---:|---:|---:|"]
    for name, r in result["calibration"].items():
        lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['candidates']['C']:.6f} | "
                     f"{r['calibration']['selected']['threshold']:.8f} |" if r else f"| {name} | FAILED | — | — |")
    if result.get("selection"):
        lines += ["", f"Зафиксированный выбор: **{result['selection']['selected']}**. После validation выбор не меняется."]
    lines += ["", "## Validation: только выбор и контроли", "",
              "| Система | mAP@10 | Rank-1 | F1 | TNR | C | 45×mAP+10×C | Принято / отказ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in result["evaluations"].items():
        if r:
            a, b = r["ranking"], r["candidates"]
            lines.append(f"| {name} | {a['mAP@10']:.6f} | {a['Rank-1']:.6f} | {b['F1']:.6f} | "
                         f"{b['TNR']:.6f} | {b['C']:.6f} | {r['quality_points_out_of_55']:.4f} | {r['accepted']} / {r['refused']} |")
        else:
            lines.append(f"| {name} | FAILED | — | — | — | — | — | — |")
    for name, delta in result.get("selected_vs", {}).items():
        lines += ["", f"Выбор против {name}: ΔmAP {100*delta['mAP_delta']:+.3f} п.п.; ΔC {100*delta['C_delta']:+.3f} п.п.; "
                  f"AP лучше/хуже/равно: {delta['improved']}/{delta['worsened']}/{delta['unchanged']}."]
    lines += ["", "## Ограничения", "",
              "- Смеси имеют 2048 измерений; доля v21 делится поровну между его тремя encoder. Контроли сохраняют исходные 512/1536 измерений и политики кандидатов.",
              "- Выбор может остаться чистым MVP или v21. Нет обязательного выигрыша смеси и нет gate по среднему seed.",
              "- Контроли обязаны точно воспроизвести v22: исходные threshold, метрики и per-query решения. Допуски не расширяются.",
              "- Поиск проводится на calibration, потому что full-train MVP видел внутренние primary ID. Старые внутренние draws не подходят для честного сравнения этой смеси.",
              "- 45×mAP+10×C — только два компонента из 55 возможных баллов; это не полный конкурсный балл и не критерий выбора вместо mAP.",
              "- submission.csv содержит десять ID при любом отказе; кандидаты отсутствуют при отказе. Junk/top-10 считаются исходным evaluator без изменения меток.",
              "- embeddings.npy содержит реальные float32-векторы в порядке query, затем gallery. Это validation-экспорты, не test submission и не готовый deployment bundle.",
              "- Общего ограничения времени нет. Cached-задачи не являются новым измерением скорости. Ни GPU, ни сеть не нужны.",
              f"- Исходные файлы сохранены: {result.get('protected_unchanged', False)}. MVP не меняется; promoted=False.",
              f"- Время текущего запуска: {result.get('elapsed_seconds', 0)/60:.1f} мин."]
    (context["output"] / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True is required for the development comparison")
    if runtime() != context["manifest"]["analysis_runtime"]:
        raise old.IntegrityError("Cached-array analysis runtime changed")
    queue = old.Queue(context, wall_hours=None)
    result = {"status": "running", "calibration": {}, "evaluations": {}, "optimizer_updates": 0,
              "encoder_forwards": 0, "promoted": False, "signature": context["signature"]}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            print("\nSTAGE 1/3: eight calibration configurations, saved features only", flush=True)
            features = load_vectors(context, "calibration")
            for spec in context["manifest"]["fusion_plan"]["systems"]:
                result["calibration"][spec["name"]] = queue.task(f"calibrate_{spec['name']}", lambda d: calibrate(
                    context, spec, features, split="calibration"))
            frozen = selection(context, result["calibration"])
            old.review.old.freeze_json(context["output"] / "frozen_selection.json", frozen)
            result["selection"] = read_selection(context)
            print(f"\nSTAGE 2/3: frozen {frozen['selected']}; validation selection disabled", flush=True)
            features = load_vectors(context, "validation")
            for case in frozen["evaluations"]:
                name = case["system"]["name"]
                result["evaluations"][name] = queue.task(f"evaluate_{name}", lambda d: evaluate(context, case, features, d))
            if any(v is None for v in result["evaluations"].values()):
                raise ValueError("Incomplete export; resume the same RUN_NAME")
            selected = result["evaluations"][frozen["selected"]]
            result["selected_vs"] = {n: previous.paired_delta(result["evaluations"][n], selected) for n in CONTROLS}
            result["status"] = "complete"
            print("\nSTAGE 3/3: report and source checks; MVP unchanged", flush=True)
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
                write_json(context["output"] / "results.json", result)
                write_report(context, result)
    return result
