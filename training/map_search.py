"""v25: improve ranking only; v24 candidate, confidence and threshold never change."""
import time
from pathlib import Path

import numpy as np

from backend.core import DATASET, ROOT, normalize, read_rows, sha256
from backend.rerank import KReciprocalReranker
from training import dual_role_experiment as previous
from training.audit import digest
from training.stage6 import write_json

old, dual = previous.old, previous.inference
VARIANT = ROOT / "OSNet-AIN-x1.0/variant_25_map_search"


def systems():
    def spec(k1, k2, lam, weight):
        return {"name": f"r1w{round(weight*100):02d}_k{k1}_q{k2}_l{round(lam*100):02d}",
                "k1": k1, "k2": k2, "lambda": lam, "r1_weight": weight}
    baseline = spec(20, 3, .5, 0.)
    grid = [spec(k1, k2, lam, 0.) for k1 in (10, 20, 30, 40) for k2 in (1, 3, 6)
            for lam in (.25, .4, .5, .65, .75, .9)]
    grid += [spec(20, 3, lam, weight) for weight in (.1, .25, .5) for lam in (.4, .5, .65, .75)]
    return [baseline]+[s for s in grid if s != baseline]  # Keep current MVP on exact calibration ties.


def prepare(run_name="map_search_v1", *, source_run="dual_role_v1", dataset=DATASET):
    source = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, result = (old.read(source/n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    if (sm["version"] != 24 or result["status"] != "complete" or result["signature"] != signature
            or not result["protected_unchanged"] or result["optimizer_updates"] or result["bn_updates"]
            or result["threshold_fit"] or not result["evaluations"]["fresh_validation"]["parity"]["bit_exact_vectors"]):
        raise old.IntegrityError("Need completed, unchanged v24")
    protected = dict(sm["protected"])
    for path in source.glob("tasks/*/complete.json"):
        protected.update(previous.search.verify_task_source(source, signature, path))
    for name in ("manifest.json", "results.json", "dual_role_profile.json"):
        protected[str(source/name)] = sha256(source/name)
    plan = {"source_signature": signature, "source_directory": str(source), "systems": systems(),
            "selection": "maximum calibration mAP@10; ties frozen order, baseline first",
            "candidate": "unchanged v24 R1 raw_top1, confidence and frozen threshold",
            "validation": "baseline and one calibration winner only",
            "scope": "observed development splits, not an independent test; no automatic promotion",
            "threshold_fit": False, "encoder_forwards": 0, "optimizer_updates": 0, "wall_time_limit": None}
    manifest = {**sm, "version": 25, "map_search_plan": plan, "protected": protected,
                "source_sha256": {**sm["source_sha256"], "training/map_search.py": sha256(Path(__file__))}}
    source22 = Path(sm["dual_role_plan"]["source_directory"])
    ctx = {"output": old.seeds.run_directory(VARIANT, run_name), "manifest": manifest, "signature": digest(manifest),
           "protected": protected, "source_output": source22, "source_manifest": old.read(source22/"manifest.json"),
           "v24_output": source, "profile_path": source/"dual_role_profile.json", "dataset": Path(dataset),
           "rows": read_rows(Path(dataset)/"train.csv")}
    previous.previous.validate_protocols(ctx)
    previous.frozen_profile(ctx)
    old.review.check_inputs(ctx, rehash=True)
    old.review.check_other_runs(ctx)
    with old.review.old.run_lock(ctx["output"]):
        old.review.old.freeze_json(ctx["output"]/"manifest.json", manifest)
    return ctx


def rank(values, n_query, spec):
    mvp, r1 = dual.unpack(values)
    weight = spec["r1_weight"]
    if not 0 <= weight <= 1 or spec["k1"] < 1 or not 1 <= spec["k2"] <= spec["k1"]+1:
        raise ValueError("Invalid frozen ranking parameters")
    features = mvp if weight == 0 else normalize(np.concatenate([
        normalize(mvp)*np.float32(np.sqrt(1-weight)), normalize(r1)*np.float32(np.sqrt(weight))], axis=1))
    q, g = normalize(features[:n_query]), normalize(features[n_query:])
    k1 = min(spec["k1"], len(g)-1)
    graph = KReciprocalReranker(g, k1, min(spec["k2"], k1+1))
    distances = np.stack([graph.distances(v, spec["lambda"]) for v in q])
    # Candidate always comes from the original R1 block, never the mixed ranking vector.
    candidate = dual.policy.rank_vectors(r1[:n_query], r1[n_query:], "raw")
    return {"order": np.argsort(distances, axis=1, kind="stable"),
            "raw_order": candidate["raw_order"], "confidence": candidate["confidence"]}


def reference(context, split):
    name = f"cached_{split}"
    directory = context["v24_output"]/"tasks"/name
    previous.search.verify_task_source(context["v24_output"], context["manifest"]["map_search_plan"]["source_signature"],
                                       directory/"complete.json")
    return old.read(directory/"result.json"), directory/"export"


def evaluate(context, split, spec, values, directory):
    if split == "validation" and spec not in read_selection(context)["evaluations"]:
        raise old.IntegrityError("Validation accepts only the frozen winner and baseline")
    q, g = old.review.old.protocol_rows(context, split)
    threshold = previous.frozen_profile(context)["threshold"]
    ranked = rank(values, len(q), spec)
    source, source_export = reference(context, split)
    metrics = dual.policy.evaluate(q, g, ranked, threshold, "raw_top1")
    decisions = dual.policy.predictions(q, g, ranked, threshold, "raw_top1")
    accepted = decisions[1]
    expected = dual.policy.predictions(q, g, dual.rank(values[:len(q)], values[len(q):]), threshold, "raw_top1")
    if accepted != expected[1] or metrics["candidates"] != source["candidates"]:
        raise old.IntegrityError("Fixed R1 candidate/confidence/refusal changed")
    if spec == systems()[0]:
        actual = dual.policy.predictions(q, g, ranked, threshold, "raw_top1")
        if actual != expected or metrics["ranking"] != source["ranking"]:
            raise old.IntegrityError("Baseline fails to reproduce v24")
    report = {"system": spec, "split": split, "threshold": threshold, **metrics,
              "protocol_sha256": digest(context["manifest"]["protocols"][split]), "candidate_unchanged": True}
    if split == "validation":
        export = directory/"export"
        saved = old.previous.verify_or_export_csv(export, q, g, ranked, threshold, "raw_top1")
        if saved != metrics or sha256(export/"candidates.csv") != sha256(source_export/"candidates.csv"):
            raise old.IntegrityError("CSV metrics/candidate bytes differ from frozen decisions")
        npy = export/"embeddings.npy"
        if npy.exists():
            if not np.array_equal(np.load(npy, allow_pickle=False), values):
                raise old.IntegrityError("Existing embedding export changed")
        else:
            np.save(npy, values)
        old.review.old.freeze_json(export/"embedding_order.json", {
            "ids": [r["image_id"] for r in q+g], "layout": dual.LAYOUT, "ranking": spec,
            "candidate": dual.ROLES["candidate"], "threshold": threshold, "sha256": sha256(npy)})
        # Reproduce the actual top10 and candidates from the exported block layout + frozen ranking spec.
        replay = np.load(npy, allow_pickle=False)
        if dual.policy.predictions(q, g, rank(replay, len(q), spec), threshold, "raw_top1") != decisions:
            raise old.IntegrityError("NPY replay changed decisions")
        report.update(export=str(export), **dual.policy.query_diagnostics(q, g, ranked))
    print(f"{split} {spec['name']}: mAP={metrics['ranking']['mAP@10']:.6f}; R1 candidates unchanged", flush=True)
    return report


def selection(context, reports):
    specs = context["manifest"]["map_search_plan"]["systems"]
    if set(reports) != {s["name"] for s in specs} or any(r is None for r in reports.values()):
        raise ValueError("Complete every calibration comparison before selection")
    for spec in specs:
        r = reports[spec["name"]]
        if (r["split"] != "calibration" or r["system"] != spec or not r["candidate_unchanged"]
                or not np.isfinite(r["ranking"]["mAP@10"])
                or r["protocol_sha256"] != digest(context["manifest"]["protocols"]["calibration"])):
            raise old.IntegrityError("Invalid calibration comparison")
    winner = max(specs, key=lambda s: reports[s["name"]]["ranking"]["mAP@10"])
    return {"signature": context["signature"], "selection_split": "calibration", "selected": winner,
            "evaluations": [specs[0]]+([winner] if winner != specs[0] else []), "promoted": False}


def read_selection(context):
    reports = {}
    for spec in context["manifest"]["map_search_plan"]["systems"]:
        directory = context["output"]/"tasks"/f"calibration_{spec['name']}"
        previous.search.verify_task_source(context["output"], context["signature"], directory/"complete.json")
        reports[spec["name"]] = old.read(directory/"result.json")
    saved = old.read(context["output"]/"frozen_selection.json")
    if saved != selection(context, reports):
        raise old.IntegrityError("Ranking selection differs from completed calibration")
    return saved


def write_report(context, result):
    lines = ["# v25 — поиск mAP при фиксированных кандидатах v24", "", f"Статус: {result['status']}",
             "84 заранее фиксированных варианта: 72 настройки MVP + 12 смесей MVP/full-train R1.",
             "Выбор только по calibration mAP; при равенстве сохраняется исходный порядок (MVP первый).",
             "Никакого обучения, нового извлечения признаков, подбора порога или изменения кандидатов.", "",
             "| Validation: контроль и выбор | mAP@10 | F1 | TNR | C |", "|---|---:|---:|---:|---:|"]
    for name, r in result["evaluations"].items():
        if r:
            lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['candidates']['F1']:.6f} | "
                         f"{r['candidates']['TNR']:.6f} | {r['candidates']['C']:.6f} |")
    if result.get("selected_vs_v24"):
        d = result["selected_vs_v24"]
        lines += ["", f"ΔmAP выбора против v24: {d['mAP_delta']:+.6f}. Улучшено / ухудшено / равно AP: "
                  f"{d['improved']} / {d['worsened']} / {d['unchanged']}."]
    lines += ["", "## Calibration: вся фиксированная сетка", "", "| Вариант | mAP@10 |", "|---|---:|"]
    for name, r in sorted(result["calibration"].items(), key=lambda x: -(x[1]["ranking"]["mAP@10"] if x[1] else -1)):
        lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} |" if r else f"| {name} | FAILED |")
    lines += ["", "## Ограничения", "",
              "Данные уже наблюдались: это development-поиск, не независимая оценка скрытого теста.",
              "Сетка не расширяется после просмотра validation; на validation проверяются лишь baseline и один заранее выбранный вариант.",
              "Результат не переключает активный MVP. Порог и кандидаты должны побайтно совпасть с v24.",
              "Для смешанного ranking в export сохраняются исходные 2048 признаков и точная конфигурация смешивания; NPY replay проверен.",
              "Junk/top-10 и evaluator не меняются. Не используются другие query, камера, время, OCR или правки bbox.",
              f"Исходники неизменны: {result.get('protected_unchanged', False)}. Время текущего вызова: {result.get('elapsed_seconds', 0)/60:.1f} мин."]
    (context["output"]/"REPORT.md").write_text("\n".join(lines)+"\n")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True required for the development comparison")
    if previous.runtime() != context["manifest"]["analysis_runtime"]:
        raise old.IntegrityError("Runtime changed; use the original research environment")
    queue = old.Queue(context, wall_hours=None)
    result = {"signature": context["signature"], "status": "running", "calibration": {}, "evaluations": {},
              "optimizer_updates": 0, "encoder_forwards": 0, "threshold_fit": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            values = previous.load_vectors(context, "calibration")
            specs = context["manifest"]["map_search_plan"]["systems"]
            for index, spec in enumerate(specs, 1):
                print(f"\nCALIBRATION {index}/{len(specs)}", flush=True)
                result["calibration"][spec["name"]] = queue.task(f"calibration_{spec['name']}",
                    lambda d: evaluate(context, "calibration", spec, values, d))
            frozen = selection(context, result["calibration"])
            old.review.old.freeze_json(context["output"]/"frozen_selection.json", frozen)
            result["selection"] = read_selection(context)
            print(f"\nFROZEN: {frozen['selected']['name']}; validation baseline + winner only", flush=True)
            values = previous.load_vectors(context, "validation")
            for spec in frozen["evaluations"]:
                result["evaluations"][spec["name"]] = queue.task(f"validation_{spec['name']}",
                    lambda d: evaluate(context, "validation", spec, values, d))
            if any(r is None for r in result["evaluations"].values()):
                raise ValueError("Incomplete validation; inspect task error and resume")
            result["selected_vs_v24"] = previous.previous.paired_delta(
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
