"""v29: user-authorized fixed power/flip finalists, not another parameter search."""
import gc
import time
from pathlib import Path

import numpy as np

from backend.core import ROOT, DATASET, read_rows, sha256
from training import multi_hypothesis as previous
from training.audit import digest
from training.stage6 import write_json

old, source, inference, dual = previous.old, previous.source, previous.inference, previous.dual
BASELINE = inference.BASELINE
VARIANT = ROOT/"OSNet-AIN-x1.0/variant_29_family_finalists"


def systems():
    names = ("V25_control", "power_50_l50", "flip_mvp_w75_l50")
    grid = {s["name"]: s for s in inference.systems()}
    return [grid[name] for name in names]


def family_shortlist(report):
    calibration = report["calibration"]
    grid = inference.systems()
    if set(calibration) != {s["name"] for s in grid} or any(v is None for v in calibration.values()):
        raise old.IntegrityError("Need all original v28 calibration comparisons")
    for spec in grid:
        r = calibration[spec["name"]]
        if (r["system"] != spec or r["split"] != "calibration" or not r["candidate_unchanged"]
                or not np.isfinite(r["ranking"]["mAP@10"])):
            raise old.IntegrityError("Invalid source calibration report")
    chosen = [dict(BASELINE)]
    for family in ("power", "flip"):
        chosen.append(max((s for s in grid if s["family"] == family),
                          key=lambda s: calibration[s["name"]]["ranking"]["mAP@10"]))
    if chosen != systems():
        raise old.IntegrityError("Family winners differ from the two explicitly authorized finalists")
    return chosen


def source_report(context):
    plan = context["manifest"]["finalists_plan"]
    path = Path(plan["source_directory"])/"results.json"
    if sha256(path) != plan["source_result_sha256"]:
        raise old.IntegrityError("Completed v28 report changed")
    report = old.read(path)
    if report["signature"] != plan["source_signature"] or report["status"] != "complete":
        raise old.IntegrityError("Completed v28 signature/status changed")
    family_shortlist(report)
    return report


def frozen_value(context):
    report = source_report(context)
    return {"signature": context["signature"], "evaluations": systems(), "selection_split": "calibration",
            "calibration_reports": {s["name"]: report["calibration"][s["name"]] for s in systems()},
            "scope": "post-hoc development shortlist authorized after observing v28; not an independent test",
            "new_parameter_search": False, "threshold_fit": False, "promoted": False}


def read_frozen(context):
    if context["manifest"]["finalists_plan"]["systems"] != systems():
        raise old.IntegrityError("Only the three frozen finalists are allowed")
    value = old.read(context["output"]/"frozen_comparison.json")
    if value != frozen_value(context):
        raise old.IntegrityError("Frozen shortlist changed")
    return value


def prepare(run_name="family_finalists_v1", *, source_run="multi_hypothesis_v1", dataset=DATASET):
    directory = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, result = (old.read(directory/n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    if (sm["version"] != 28 or result["status"] != "complete" or result["signature"] != signature
            or not result["protected_unchanged"] or result["optimizer_updates"] or result["threshold_fit"]
            or result["promoted"] or sm["map_search_plan"]["systems"] != inference.systems()
            or result["selection"] != previous.search.read_selection({"output": directory, "manifest": sm, "signature": signature})):
        raise old.IntegrityError("Need complete unchanged v28")
    family_shortlist(result)
    protected = dict(sm["protected"])
    for path in directory.glob("tasks/*/complete.json"):
        protected.update(source.search.verify_task_source(directory, signature, path))
    for split, key in (("calibration", "calibration"), ("validation", "evaluations")):
        for name, report in result[key].items():
            if report != old.read(directory/"tasks"/f"{split}_{name}"/"result.json"):
                raise old.IntegrityError("Source task and aggregate disagree")
    for name in ("manifest.json", "results.json", "frozen_selection.json"):
        protected[str(directory/name)] = sha256(directory/name)
    plan = {"source_directory": str(directory), "source_signature": signature,
            "source_result_sha256": sha256(directory/"results.json"), "systems": systems(),
            "calibration_reports": {s["name"]: result["calibration"][s["name"]] for s in systems()},
            "calibration": "replay exactly three existing results, using the saved v28 flip cache",
            "validation": "compare exactly v25, power_50_l50 and flip_mvp_w75_l50; no further tuning",
            "scope": "post-hoc development comparison, not an independent hidden-test estimate",
            "candidate": "unchanged original full R1 raw_top1/cosine/threshold",
            "optimizer_updates": 0, "threshold_fit": False, "wall_time_limit": None, "promoted": False}
    module = "training/family_finalists.py"
    manifest = {**sm, "version": 29, "finalists_plan": plan, "protected": protected,
                "source_sha256": {**sm["source_sha256"], module: sha256(ROOT/module)}}
    source22 = Path(sm["dual_role_plan"]["source_directory"])
    context = {"output": old.seeds.run_directory(VARIANT, run_name), "manifest": manifest, "signature": digest(manifest),
               "protected": protected, "source_output": source22, "source_manifest": old.read(source22/"manifest.json"),
               "profile_path": Path(sm["map_search_plan"]["source_directory"])/"dual_role_profile.json",
               "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv")}
    previous.previous.comparison.validate_protocols(context)
    source.frozen_profile(context)
    previous.check_inputs(context)
    old.review.check_other_runs(context)
    with old.review.old.run_lock(context["output"]):
        old.review.old.freeze_json(context["output"]/"manifest.json", manifest)
        old.review.old.freeze_json(context["output"]/"frozen_comparison.json", frozen_value(context))
    print("Preflight OK: three fixed finalists, unchanged candidates, no training or automatic promotion", flush=True)
    return context


def reference(context, split):
    plan = context["manifest"]["finalists_plan"]
    directory = Path(plan["source_directory"])
    task = directory/"tasks"/f"{split}_V25_control"
    source.search.verify_task_source(directory, plan["source_signature"], task/"complete.json")
    return old.read(task/"result.json"), task/"export"


def verify_calibration(context, directory):
    frozen = read_frozen(context)
    original = source.load_vectors(context, "calibration")
    plan = context["manifest"]["finalists_plan"]
    source_context = {**context, "output": Path(plan["source_directory"]), "signature": plan["source_signature"]}
    flipped = previous.load_flip(source_context, "calibration", source_report(context)["features"]["calibration"])
    q, g = old.review.old.protocol_rows(context, "calibration")
    threshold = source.frozen_profile(context)["threshold"]
    reports = {}
    for spec in frozen["evaluations"]:
        values = np.concatenate([original, flipped], axis=1) if spec["family"] == "flip" else original
        ranked = inference.rank(values, len(q), spec)
        metrics = dual.policy.evaluate(q, g, ranked, threshold, "raw_top1")
        expected = frozen["calibration_reports"][spec["name"]]
        if (threshold != expected["threshold"] or expected["protocol_sha256"] != digest(context["manifest"]["protocols"]["calibration"])
                or any(metrics[k] != expected[k] for k in ("ranking", "candidates"))):
            raise old.IntegrityError("Calibration replay differs from completed v28; do not retune")
        reports[spec["name"]] = metrics
        print(f"CALIBRATION REPLAY {spec['name']}: {metrics['ranking']['mAP@10']:.6f} exact", flush=True)
    return {"passed": True, "systems": systems(), "metrics": reports, "encoder_forwards": 0}


def validation_ready(context):
    frozen = read_frozen(context)
    directory = context["output"]/"tasks"/"verify_calibration"
    source.search.verify_task_source(context["output"], context["signature"], directory/"complete.json")
    report = old.read(directory/"result.json")
    if not report["passed"] or report["systems"] != systems():
        raise old.IntegrityError("Calibration replay must finish before validation")
    return frozen


def flip_validation(context, directory):
    validation_ready(context)
    q, g = old.review.old.protocol_rows(context, "validation")
    rows, original = q+g, source.load_vectors(context, "validation")
    plan = context["manifest"]["multi_hypothesis_plan"]
    batch, chunk = plan["flip_batch_size"], plan["flip_chunk_size"]
    encoder, blocks, fresh, cached = None, [], 0, 0
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for start in range(0, len(rows), chunk):
            entries = rows[start:start+chunk]
            path = directory/f"block_{start:05d}.npy"
            receipt = path.with_suffix(".json")
            identity = {"signature": context["signature"], "split": "validation",
                        "ids": [r["image_id"] for r in entries], "transform": "horizontal_flip"}
            if receipt.exists():
                saved = old.read(receipt)
                if saved.get("identity") != identity or not path.is_file() or sha256(path) != saved["sha256"]:
                    raise old.IntegrityError("Changed flip cache block; stop without regeneration")
                values = np.load(path, allow_pickle=False)
                cached += len(entries)
            else:
                if encoder is None:
                    encoder = dual.DualRoleEncoder(context["profile_path"])
                    probe = inference.encode_rows(encoder, rows[:batch], context["dataset"], flip=False, batch_size=batch)
                    if not np.allclose(probe, original[:len(probe)], rtol=0, atol=plan["flip_parity_atol"]):
                        raise old.IntegrityError("Original inference differs from saved vectors")
                values = inference.encode_rows(encoder, entries, context["dataset"], flip=True, batch_size=batch)
                dual.unpack(values)
                pending = path.with_suffix(".pending.npy")
                np.save(pending, values)
                pending.replace(path)
                write_json(receipt, {"identity": identity, "sha256": sha256(path)})
                fresh += len(entries)
            dual.unpack(values)
            if values.shape != (len(entries), 2048):
                raise old.IntegrityError("Flip block shape changed")
            blocks.append(values)
            print(f"FLIP validation: {min(start+chunk,len(rows))}/{len(rows)} images", flush=True)
        values, path = np.concatenate(blocks), directory/"features.npy"
        if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), values):
            raise old.IntegrityError("Existing flip aggregate changed")
        if not path.exists():
            np.save(path, values)
        return {"path": str(path), "sha256": sha256(path), "split": "validation", "ids": [r["image_id"] for r in rows],
                "fresh_flip_images_this_attempt": fresh, "cached_flip_images_this_attempt": cached, "encoder_count": 4,
                "original_probe_images_this_attempt": min(batch, len(rows)) if encoder is not None else 0}
    finally:
        del encoder
        gc.collect()


def evaluate(context, spec, values, directory):
    if spec not in validation_ready(context)["evaluations"]:
        raise old.IntegrityError("Only the three frozen finalists may be evaluated")
    q, g = old.review.old.protocol_rows(context, "validation")
    threshold = source.frozen_profile(context)["threshold"]
    expected, source_export = reference(context, "validation")
    if threshold != expected["threshold"]:
        raise old.IntegrityError("Original R1 threshold changed")
    original = np.load(source_export/"embeddings.npy", allow_pickle=False)
    if not np.array_equal(values[:, :2048], original):
        raise old.IntegrityError("Original embedding blocks changed")
    ranked = inference.rank(values, len(q), spec)
    metrics = dual.policy.evaluate(q, g, ranked, threshold, "raw_top1")
    decisions = dual.policy.predictions(q, g, ranked, threshold, "raw_top1")
    _, r1 = dual.unpack(original)
    raw = dual.policy.rank_vectors(r1[:len(q)], r1[len(q):], "raw")
    if (metrics["candidates"] != expected["candidates"]
            or decisions[1] != dual.policy.predictions(q, g, raw, threshold, "raw_top1")[1]):
        raise old.IntegrityError("Original R1 candidates/confidence/refusals changed")
    if spec == BASELINE and metrics["ranking"] != expected["ranking"]:
        raise old.IntegrityError("v25 control ranking changed")
    export = directory/"export"
    saved = old.previous.verify_or_export_csv(export, q, g, ranked, threshold, "raw_top1")
    if saved != metrics or sha256(export/"candidates.csv") != sha256(source_export/"candidates.csv"):
        raise old.IntegrityError("Candidate CSV bytes or exported metrics changed")
    npy = export/"embeddings.npy"
    if npy.exists() and not np.array_equal(np.load(npy, allow_pickle=False), values):
        raise old.IntegrityError("Existing exported vectors changed")
    if not npy.exists():
        np.save(npy, values)
    if spec == BASELINE and any(sha256(export/n) != sha256(source_export/n) for n in ("submission.csv", "embeddings.npy")):
        raise old.IntegrityError("v25 control artifact bytes changed")
    old.review.old.freeze_json(export/"embedding_order.json", {
        "ids": [r["image_id"] for r in q+g], "query_count": len(q), "gallery_count": len(g), "dimension": values.shape[1],
        "original_slice": [0, 2048], "original_layout": dual.LAYOUT,
        "flip_slice": [2048, 4096] if spec["family"] == "flip" else None,
        "ranking": spec, "candidate": dual.ROLES["candidate"], "threshold": threshold, "sha256": sha256(npy),
        "ranking_source_sha256": context["manifest"]["source_sha256"]["training/multi_hypothesis_inference.py"]})
    replay = inference.rank(np.load(npy, allow_pickle=False), len(q), spec)
    if dual.policy.predictions(q, g, replay, threshold, "raw_top1") != decisions:
        raise old.IntegrityError("NPY replay changed decisions")
    print(f"VALIDATION {spec['name']}: mAP={metrics['ranking']['mAP@10']:.6f}; candidates unchanged", flush=True)
    return {"system": spec, "split": "validation", "threshold": threshold, **metrics,
            "protocol_sha256": digest(context["manifest"]["protocols"]["validation"]), "candidate_unchanged": True,
            "export": str(export), **dual.policy.query_diagnostics(q, g, ranked)}


def write_report(context, result):
    lines = ["# v29 — два фиксированных финалиста против v25", "", f"Статус: {result['status']}",
             "Это post-hoc сравнение на уже наблюдавшейся development validation, не независимый тест.",
             "Нового перебора параметров, обучения, изменения порога и автоматического продвижения нет.", "",
             "| Вариант | Calibration v28 | Validation mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|---:|"]
    calibration = context["manifest"]["finalists_plan"]["calibration_reports"]
    for spec in systems():
        name = spec["name"]
        cal = calibration[name]["ranking"]["mAP@10"]
        r = result["evaluations"].get(name)
        lines.append(f"| {name} | {cal:.6f} | {r['ranking']['mAP@10']:.6f} | {r['ranking']['Rank-1']:.6f} | "
                     f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |" if r else
                     f"| {name} | {cal:.6f} | NOT COMPLETE | — | — | — |")
    for name, d in result.get("vs_v25", {}).items():
        lines += ["", f"{name}: ΔmAP {d['mAP_delta']:+.6f}; AP лучше/хуже/равно {d['improved']}/{d['worsened']}/{d['unchanged']}."]
    if result.get("best_observed_validation"):
        lines += ["", f"Лучший наблюдавшийся результат из этих трёх: {result['best_observed_validation']['name']}.",
                  "Это описание development-результата, не подтверждение превосходства на скрытом тесте и не смена MVP."]
    lines += ["", "Порог, raw R1 candidate/confidence, bbox, evaluator и исходная validation неизменны.",
              "Контроль и power: реальные 2048 признаков; flip: original2048 + flipped2048. NPY replay проверяется.",
              "Кандидаты всех систем обязаны побайтно совпасть с v25. Каждый query независим от остальных.",
              "v28 calibration flip-кэш переиспользуется; заново извлекаются только validation flip-признаки.",
              "Готовые задачи/блоки возобновляются. Неполное сравнение не объявляется завершённым.",
              f"Защищённые источники неизменны: {result.get('protected_unchanged', False)}.",
              f"Время текущего вызова: {result.get('elapsed_seconds', 0)/60:.1f} мин; не замер конкурсной скорости."]
    (context["output"]/"REPORT.md").write_text("\n".join(lines)+"\n")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True required for the post-hoc comparison")
    if source.runtime() != context["manifest"]["analysis_runtime"]:
        raise old.IntegrityError("Runtime changed; use the original research environment")
    queue = old.Queue(context, wall_hours=None)
    result = {"signature": context["signature"], "status": "running", "evaluations": {}, "features": {},
              "optimizer_updates": 0, "threshold_fit": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            previous.check_inputs(context)
            result["frozen_comparison"] = read_frozen(context)
            result["calibration_replay"] = queue.task("verify_calibration", lambda d: verify_calibration(context, d))
            if result["calibration_replay"] is None:
                raise ValueError("Calibration replay incomplete; validation is blocked")
            original = source.load_vectors(context, "validation")
            for index, spec in enumerate(systems(), 1):
                print(f"\nFINALIST {index}/3: {spec['name']} | elapsed {time.monotonic()-queue.started:.1f}s", flush=True)
                values = original
                if spec["family"] == "flip":
                    feature = queue.task("flip_validation", lambda d: flip_validation(context, d))
                    result["features"]["validation"] = feature
                    values = (None if feature is None else np.concatenate([
                        original, previous.load_flip(context, "validation", feature)], axis=1))
                result["evaluations"][spec["name"]] = (None if values is None else queue.task(
                    f"validation_{spec['name']}", lambda d: evaluate(context, spec, values, d)))
            if any(v is None for v in result["evaluations"].values()):
                raise ValueError("Incomplete finalist comparison; inspect errors and resume")
            base = result["evaluations"][BASELINE["name"]]
            result["vs_v25"] = {s["name"]: previous.previous.comparison.paired_delta(base, result["evaluations"][s["name"]])
                                for s in systems()[1:]}
            result["best_observed_validation"] = max(systems(), key=lambda s: result["evaluations"][s["name"]]["ranking"]["mAP@10"])
            result["status"] = "complete"
        except BaseException:
            result["status"] = "incomplete"
            raise
        finally:
            result.update(events=queue.events, elapsed_seconds=time.monotonic()-queue.started)
            try:
                previous.check_inputs(context)
                result["protected_unchanged"] = True
            except BaseException:
                result.update(status="integrity_check_failed", protected_unchanged=False)
                raise
            finally:
                write_json(context["output"]/"results.json", result)
                write_report(context, result)
    return result
