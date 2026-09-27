"""v28 Run All: fixed multi-family calibration queue, then v25 and one winner."""
import gc
import time
from pathlib import Path

import numpy as np

from backend.core import ROOT, DATASET, read_rows, sha256
from training import late_fusion as previous
from training import multi_hypothesis_inference as inference
from training.audit import digest
from training.stage6 import write_json

old, source, search, dual = previous.old, previous.source, previous.search, previous.dual
BASELINE, systems = inference.BASELINE, inference.systems
VARIANT = ROOT/"OSNet-AIN-x1.0/variant_28_multi_hypothesis"


def check_inputs(context):
    try:
        old.review.check_inputs(context, rehash=True)
    except (ValueError, OSError) as error:
        raise old.IntegrityError(f"Protected source changed: {error}") from error


def prepare(run_name="multi_hypothesis_v1", *, source_run="late_fusion_v1", dataset=DATASET):
    directory = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, result = (old.read(directory/n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    if (sm["version"] != 27 or result["status"] != "complete" or result["signature"] != signature
            or not result["protected_unchanged"] or result["optimizer_updates"] or result["encoder_forwards"]
            or result["threshold_fit"] or result["promoted"]
            or sm["map_search_plan"]["systems"] != previous.systems()
            or result["selection"] != search.read_selection({"output": directory, "manifest": sm, "signature": signature})):
        raise old.IntegrityError("Need complete unchanged v27 with original v25 control")
    protected = dict(sm["protected"])
    for path in directory.glob("tasks/*/complete.json"):
        protected.update(source.search.verify_task_source(directory, signature, path))
    for split, key in (("calibration", "calibration"), ("validation", "evaluations")):
        for name, report in result[key].items():
            if report != old.read(directory/"tasks"/f"{split}_{name}"/"result.json"):
                raise old.IntegrityError("Source aggregate differs from protected task")
    for name in ("manifest.json", "results.json", "frozen_selection.json"):
        protected[str(directory/name)] = sha256(directory/name)
    plan = {"source_directory": str(directory), "source_signature": signature,
            "families": list(inference.FAMILIES), "systems": systems(), "k1": 20, "k2": 3,
            "selection": "maximum calibration mAP across the complete fixed grid; ties v25 first",
            "validation": "v25 control and one overall winner only, no family winner sweep",
            "candidate": "original full R1 raw_top1/cosine, unchanged threshold; never TTA",
            "gallery_center": "per-block mean of static gallery only; no query statistics, labels or updates",
            "power": "signed abs(x)**gamma then per-block L2; experimental adaptation, not RootSIFT",
            "flip": "horizontal mirror of each organizer crop after fixed preprocessing; same transform for q/g",
            "flip_batch_size": 16, "flip_chunk_size": 128, "flip_parity_atol": sm["dual_role_plan"]["vector_atol"],
            "query_expansion": False, "optimizer_updates": 0, "threshold_fit": False,
            "wall_time_limit": None, "promoted": False,
            "scope": "adaptive search on observed development splits, not independent hidden-test evidence",
            "power_source": "https://europe.naverlabs.com/wp-content/uploads/2010/09/PSM10_0766.pdf"}
    modules = ("training/multi_hypothesis.py", "training/multi_hypothesis_inference.py")
    manifest = {**sm, "version": 28, "multi_hypothesis_plan": plan, "protected": protected,
                "map_search_plan": {**sm["map_search_plan"], "systems": systems(), "selection": plan["selection"]},
                "source_sha256": {**sm["source_sha256"], **{p: sha256(ROOT/p) for p in modules}}}
    source22 = Path(sm["dual_role_plan"]["source_directory"])
    ctx = {"output": old.seeds.run_directory(VARIANT, run_name), "manifest": manifest, "signature": digest(manifest),
           "protected": protected, "source_output": source22, "source_manifest": old.read(source22/"manifest.json"),
           "reference_output": directory, "profile_path": Path(sm["map_search_plan"]["source_directory"])/"dual_role_profile.json",
           "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv")}
    previous.comparison.validate_protocols(ctx)
    source.frozen_profile(ctx)
    check_inputs(ctx)
    old.review.check_other_runs(ctx)
    with old.review.old.run_lock(ctx["output"]):
        old.review.old.freeze_json(ctx["output"]/"manifest.json", manifest)
    print(f"Preflight OK: {len(systems())} configurations, four families, no training or promotion", flush=True)
    return ctx


def reference(context, split):
    directory = context["reference_output"]
    task = directory/"tasks"/f"{split}_V25_control"
    source.search.verify_task_source(directory, context["manifest"]["multi_hypothesis_plan"]["source_signature"], task/"complete.json")
    return old.read(task/"result.json"), task/"export"


def flip_features(context, split, directory):
    if split not in ("calibration", "validation"):
        raise ValueError("Only original development splits")
    if split == "validation" and search.read_selection(context)["selected"]["family"] != "flip":
        raise old.IntegrityError("Validation flip inference only for a frozen flip winner")
    q, g = old.review.old.protocol_rows(context, split)
    rows = q+g
    original = source.load_vectors(context, split)
    plan = context["manifest"]["multi_hypothesis_plan"]
    batch, chunk = plan["flip_batch_size"], plan["flip_chunk_size"]
    encoder, blocks, fresh_images, cached_images = None, [], 0, 0
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for start in range(0, len(rows), chunk):
            entries = rows[start:start+chunk]
            path = directory/f"block_{start:05d}.npy"
            receipt = path.with_suffix(".json")
            identity = {"signature": context["signature"], "split": split,
                        "ids": [r["image_id"] for r in entries], "transform": "horizontal_flip"}
            if receipt.exists():
                saved = old.read(receipt)
                if saved.get("identity") != identity or not path.is_file() or sha256(path) != saved["sha256"]:
                    raise old.IntegrityError("Changed flip cache block; do not silently regenerate")
                values = np.load(path, allow_pickle=False)
                cached_images += len(entries)
            else:
                if encoder is None:
                    encoder = dual.DualRoleEncoder(context["profile_path"])
                    probe = inference.encode_rows(encoder, rows[:batch], context["dataset"], flip=False, batch_size=batch)
                    if not np.allclose(probe, original[:len(probe)], rtol=0, atol=plan["flip_parity_atol"]):
                        raise old.IntegrityError("Original image inference no longer matches source cache")
                values = inference.encode_rows(encoder, entries, context["dataset"], flip=True, batch_size=batch)
                inference.dual.unpack(values)
                pending = path.with_suffix(".pending.npy")
                np.save(pending, values)
                pending.replace(path)
                write_json(receipt, {"identity": identity, "sha256": sha256(path)})
                fresh_images += len(entries)
            dual.unpack(values)
            if values.shape != (len(entries), 2048):
                raise old.IntegrityError("Flip block shape/order differs from original protocol")
            blocks.append(values)
            print(f"FLIP {split}: {min(start+chunk,len(rows))}/{len(rows)} images; saved every {chunk}", flush=True)
        path = directory/"features.npy"
        values = np.concatenate(blocks)
        if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), values):
            raise old.IntegrityError("Existing flip feature aggregate changed")
        if not path.exists():
            np.save(path, values)
        return {"path": str(path), "sha256": sha256(path), "split": split,
                "ids": [r["image_id"] for r in rows], "fresh_flip_images_this_attempt": fresh_images,
                "cached_flip_images_this_attempt": cached_images, "encoder_count": 4,
                "original_probe_images_this_attempt": min(batch, len(rows)) if encoder is not None else 0}
    finally:
        del encoder
        gc.collect()


def load_flip(context, split, result):
    directory = context["output"]/"tasks"/f"flip_{split}"
    source.search.verify_task_source(context["output"], context["signature"], directory/"complete.json")
    path = directory/"features.npy"
    q, g = old.review.old.protocol_rows(context, split)
    if (result != old.read(directory/"result.json") or result["path"] != str(path)
            or result["sha256"] != sha256(path) or result["split"] != split
            or result["ids"] != [r["image_id"] for r in q+g]):
        raise old.IntegrityError("Flip cache provenance/order changed")
    values = np.load(path, allow_pickle=False)
    dual.unpack(values)
    if len(values) != len(q)+len(g):
        raise old.IntegrityError("Flip cache row count changed")
    return values


def evaluate(context, split, spec, values, directory):
    if split not in ("calibration", "validation"):
        raise ValueError("Only original calibration/validation")
    if split == "validation" and spec not in search.read_selection(context)["evaluations"]:
        raise old.IntegrityError("Validation permits only the frozen overall winner and v25")
    q, g = old.review.old.protocol_rows(context, split)
    threshold = source.frozen_profile(context)["threshold"]
    expected, source_export = reference(context, split)
    if threshold != expected["threshold"]:
        raise old.IntegrityError("Frozen threshold differs from v25")
    ranked = inference.rank(values, len(q), spec)
    metrics = dual.policy.evaluate(q, g, ranked, threshold, "raw_top1")
    decisions = dual.policy.predictions(q, g, ranked, threshold, "raw_top1")
    _, raw_r1 = dual.unpack(values[:, :2048])
    raw = dual.policy.rank_vectors(raw_r1[:len(q)], raw_r1[len(q):], "raw")
    if (decisions[1] != dual.policy.predictions(q, g, raw, threshold, "raw_top1")[1]
            or metrics["candidates"] != expected["candidates"]):
        raise old.IntegrityError("Original R1 candidates/confidence/refusals changed")
    if spec == BASELINE and metrics["ranking"] != expected["ranking"]:
        raise old.IntegrityError("v25 control ranking changed")
    report = {"system": spec, "split": split, "threshold": threshold, **metrics,
              "protocol_sha256": digest(context["manifest"]["protocols"][split]), "candidate_unchanged": True}
    if split == "validation":
        export = directory/"export"
        saved = old.previous.verify_or_export_csv(export, q, g, ranked, threshold, "raw_top1")
        if saved != metrics or sha256(export/"candidates.csv") != sha256(source_export/"candidates.csv"):
            raise old.IntegrityError("Candidate CSV bytes or exported metrics changed")
        if not np.array_equal(values[:, :2048], np.load(source_export/"embeddings.npy", allow_pickle=False)):
            raise old.IntegrityError("Original embedding blocks changed")
        npy = export/"embeddings.npy"
        if npy.exists() and not np.array_equal(np.load(npy, allow_pickle=False), values):
            raise old.IntegrityError("Existing exported embeddings changed")
        if not npy.exists():
            np.save(npy, values)
        if spec == BASELINE and any(sha256(export/n) != sha256(source_export/n) for n in ("submission.csv", "embeddings.npy")):
            raise old.IntegrityError("v25 control artifact bytes changed")
        old.review.old.freeze_json(export/"embedding_order.json", {
            "ids": [r["image_id"] for r in q+g], "query_count": len(q), "gallery_count": len(g),
            "dimension": values.shape[1], "original_slice": [0, 2048], "original_layout": dual.LAYOUT,
            "flip_slice": [2048, 4096] if spec["family"] == "flip" else None,
            "ranking": spec, "candidate": dual.ROLES["candidate"], "threshold": threshold, "sha256": sha256(npy),
            "ranking_source_sha256": context["manifest"]["source_sha256"]["training/multi_hypothesis_inference.py"]})
        replay = inference.rank(np.load(npy, allow_pickle=False), len(q), spec)
        if dual.policy.predictions(q, g, replay, threshold, "raw_top1") != decisions:
            raise old.IntegrityError("NPY replay changed decisions")
        report.update(export=str(export), **dual.policy.query_diagnostics(q, g, ranked))
    print(f"{split} {spec['name']}: mAP={metrics['ranking']['mAP@10']:.6f}; candidates unchanged", flush=True)
    return report


def write_report(context, result):
    lines = ["# v28 — четыре направления в одном Run All", "", f"Статус: {result['status']}",
             "57 конфигураций: v25 + 24 состава ансамбля + 6 power + 8 center + 18 flip-TTA.",
             "Это адаптивный development-поиск, не независимый hidden test. Никакого автоматического продвижения.",
             "", "## Calibration по направлениям", "", "| Семейство | Готово / всего | Лучший вариант | mAP@10 |",
             "|---|---:|---|---:|"]
    specs = context["manifest"]["map_search_plan"]["systems"]
    for family in ("control", *inference.FAMILIES):
        group = [s for s in specs if s["family"] == family]
        ready = [result["calibration"].get(s["name"]) for s in group]
        ready = [r for r in ready if r]
        best = max(ready, key=lambda r: r["ranking"]["mAP@10"]) if ready else None
        lines.append(f"| {family} | {len(ready)}/{len(group)} | {best['system']['name'] if best else '—'} | "
                     f"{best['ranking']['mAP@10'] if best else '—'} |")
    lines += ["", "## Validation: контроль и один общий победитель", "",
              "| Вариант | mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
    for name, r in result["evaluations"].items():
        if r:
            lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['ranking']['Rank-1']:.6f} | "
                         f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |")
    if result.get("selected_vs_v25"):
        d = result["selected_vs_v25"]
        lines += ["", f"ΔmAP: {d['mAP_delta']:+.6f}; AP лучше/хуже/равно: {d['improved']}/{d['worsened']}/{d['unchanged']}."]
    lines += ["", "## Все calibration-варианты", "", "| Вариант | mAP@10 |", "|---|---:|"]
    for spec in specs:
        r = result["calibration"].get(spec["name"])
        lines.append(f"| {spec['name']} | {r['ranking']['mAP@10'] if r else 'NOT COMPLETE'} |")
    lines += ["", "## Проверки и ограничения", "",
              "Порог, raw R1 кандидат и confidence, исходная validation, bbox и evaluator неизменны.",
              "Center использует только среднее статической gallery. Другие query недоступны всем scorer.",
              "Query expansion не реализован (организаторы, вопрос 10). Flip-TTA — того же изображения, без правки bbox.",
              "Экспорт: реальные 2048 признаков; для flip-победителя 4096 = original2048 + flipped2048. NPY replay проверяется.",
              "Стоимость TTA выше: нужны дополнительные encoder passes; официальный performance здесь не измеряется.",
              "Изолированная ошибка не прекращает независимые опыты, но неполный набор блокирует отбор/validation.",
              "После исправления причины повторный Run All продолжает тот же RUN_NAME; код во время серии менять нельзя.",
              "[Источник идеи power normalization](https://europe.naverlabs.com/wp-content/uploads/2010/09/PSM10_0766.pdf): "
              "перенос на OSNet и gamma=1.25 — наши гипотезы, не результат статьи.",
              f"Источники неизменны: {result.get('protected_unchanged', False)}. "
              f"Время текущего вызова: {result.get('elapsed_seconds', 0)/60:.1f} мин; не скорость инференса."]
    (context["output"]/"REPORT.md").write_text("\n".join(lines)+"\n")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True required for development evaluation")
    if source.runtime() != context["manifest"]["analysis_runtime"]:
        raise old.IntegrityError("Runtime changed; use the original research environment")
    queue = old.Queue(context, wall_hours=None)
    result = {"signature": context["signature"], "status": "running", "calibration": {}, "evaluations": {},
              "features": {}, "optimizer_updates": 0, "threshold_fit": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            check_inputs(context)
            original = source.load_vectors(context, "calibration")
            specs = context["manifest"]["map_search_plan"]["systems"]
            flipped = None
            for index, spec in enumerate(specs, 1):
                if spec["family"] == "flip" and "calibration" not in result["features"]:
                    feature = queue.task("flip_calibration", lambda d: flip_features(context, "calibration", d))
                    result["features"]["calibration"] = feature
                    if feature is not None:
                        flipped = np.concatenate([original, load_flip(context, "calibration", feature)], axis=1)
                print(f"\nCALIBRATION {index}/{len(specs)} | {spec['family']} | elapsed {time.monotonic()-queue.started:.1f}s", flush=True)
                values = flipped if spec["family"] == "flip" else original
                result["calibration"][spec["name"]] = (None if values is None else queue.task(
                    f"calibration_{spec['name']}", lambda d: evaluate(context, "calibration", spec, values, d)))
            frozen = search.selection(context, result["calibration"])
            old.review.old.freeze_json(context["output"]/"frozen_selection.json", frozen)
            result["selection"] = search.read_selection(context)
            print(f"\nFROZEN overall winner: {frozen['selected']['name']}. Validation v25 + winner only.", flush=True)
            original = source.load_vectors(context, "validation")
            flipped = None
            if frozen["selected"]["family"] == "flip":
                feature = queue.task("flip_validation", lambda d: flip_features(context, "validation", d))
                result["features"]["validation"] = feature
                if feature is not None:
                    flipped = np.concatenate([original, load_flip(context, "validation", feature)], axis=1)
            for spec in frozen["evaluations"]:
                values = flipped if spec["family"] == "flip" else original
                result["evaluations"][spec["name"]] = (None if values is None else queue.task(
                    f"validation_{spec['name']}", lambda d: evaluate(context, "validation", spec, values, d)))
            if any(r is None for r in result["evaluations"].values()):
                raise ValueError("Incomplete validation; inspect errors and resume")
            result["selected_vs_v25"] = previous.comparison.paired_delta(
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
