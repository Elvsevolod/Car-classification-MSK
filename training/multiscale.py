"""v31: five fixed multi-scale comparisons; frozen weights, candidates and validation gate."""
import gc
import time
from pathlib import Path

import numpy as np

from backend.core import ROOT, DATASET, read_rows, sha256
from training import pool_rerank as previous, multiscale_inference as inference
from training.audit import digest
from training.stage6 import write_json

old, source, search, dual = previous.old, previous.source, previous.search, previous.dual
BASELINE, systems = inference.BASELINE, inference.systems
VARIANT = ROOT/"OSNet-AIN-x1.0/variant_31_multiscale"


def runtime():
    return {**source.runtime(), "onnx": inference.onnx.__version__}


def completed_source(directory):
    sm, result = (old.read(directory/n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    ctx = {"output": directory, "manifest": sm, "signature": signature}
    if (sm["version"] != 30 or result["status"] != "complete" or result["signature"] != signature
            or not result["protected_unchanged"] or result["optimizer_updates"] or result["encoder_forwards"]
            or result["threshold_fit"] or result["promoted"] or sm["map_search_plan"]["systems"] != previous.systems()
            or result["selection"] != search.read_selection(ctx)
            or set(result["evaluations"]) != {s["name"] for s in result["selection"]["evaluations"]}):
        raise old.IntegrityError("Need completed unchanged v30 with the original v25 control")
    protected = dict(sm["protected"])
    for split, key in (("calibration", "calibration"), ("validation", "evaluations")):
        for name, report in result[key].items():
            task = directory/"tasks"/f"{split}_{name}"
            protected.update(source.search.verify_task_source(directory, signature, task/"complete.json"))
            if not report or report != old.read(task/"result.json"):
                raise old.IntegrityError("Source task and aggregate disagree")
    for name in ("manifest.json", "results.json", "frozen_selection.json"):
        protected[str(directory/name)] = sha256(directory/name)
    return sm, signature, protected


def check_inputs(context):
    m = context["manifest"]
    if (context["signature"] != digest(m) or m["version"] != 31
            or m["multiscale_plan"]["systems"] != systems() or m["map_search_plan"]["systems"] != systems()):
        raise old.IntegrityError("Only the five frozen v31 configurations are allowed")
    try:
        old.review.check_inputs(context, rehash=True)
    except (ValueError, OSError) as error:
        raise old.IntegrityError(f"Protected source changed: {error}") from error


def prepare(run_name="multiscale_v1", *, source_run="pool_rerank_v1", dataset=DATASET):
    directory = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, signature, protected = completed_source(directory)
    profile = Path(sm["map_search_plan"]["source_directory"])/"dual_role_profile.json"
    plan = {"source_directory": str(directory), "source_signature": signature, "systems": systems(),
            "source_models": inference.model_sources(profile), "batch_size": 16, "chunk_size": 128,
            "vector_atol": sm["dual_role_plan"]["vector_atol"],
            "input_adaptation": "ONNX input H/W only; graph operators, constants and weights unchanged; in-memory copies",
            "preprocessing": "same organizer bbox, RGB, bilinear square resize and ImageNet normalization",
            "fusion": "MVP .50; R1_256 .50*(1-w); R1_high .50*w; w=.25/.50; sizes 320/384",
            "graph": "full static gallery, k1=20, k2=3, lambda=.50; no pool restriction",
            "candidate": "unchanged original R1_256 raw candidate/cosine/frozen threshold",
            "selection": "maximum calibration mAP over all five systems; exact ties retain v25 first",
            "validation": "v25 and one calibration winner; high-resolution extraction for selected size only",
            "scope": "post-hoc development search; not independent hidden-test evidence",
            "optimizer_updates": 0, "bn_updates": 0, "threshold_fit": False, "wall_time_limit": None}
    modules = ("training/multiscale.py", "training/multiscale_inference.py")
    manifest = {**sm, "version": 31, "multiscale_plan": plan, "scale_runtime": runtime(), "protected": protected,
                "map_search_plan": {**sm["map_search_plan"], "systems": systems(), "selection": plan["selection"]},
                "source_sha256": {**sm["source_sha256"], **{p: sha256(ROOT/p) for p in modules}}}
    src = Path(sm["dual_role_plan"]["source_directory"])
    context = {"output": old.seeds.run_directory(VARIANT, run_name), "manifest": manifest, "signature": digest(manifest),
               "protected": protected, "source_output": src, "source_manifest": old.read(src/"manifest.json"),
               "profile_path": profile, "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv")}
    previous.comparison.validate_protocols(context)
    source.frozen_profile(context)
    check_inputs(context)
    old.review.check_other_runs(context)
    with old.review.old.run_lock(context["output"]):
        old.review.old.freeze_json(context["output"]/"manifest.json", manifest)
    print("Preflight OK: v25 + R1 320/384 at two fixed weights; no training or MVP promotion", flush=True)
    return context


def features(context, split, size, directory):
    if split not in ("calibration", "validation") or size not in inference.SIZES:
        raise ValueError("Use original splits and one of the two fixed larger sizes")
    if split == "validation" and search.read_selection(context)["selected"]["size"] != size:
        raise old.IntegrityError("Validation extraction only for the frozen winning size")
    q, g = old.review.old.protocol_rows(context, split)
    rows = q+g
    plan = context["manifest"]["multiscale_plan"]
    batch, chunk, atol = plan["batch_size"], plan["chunk_size"], plan["vector_atol"]
    original = source.load_vectors(context, split)[:, 512:2048]
    blocks, encoder, fresh, cached, probe = [], None, 0, 0, None
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for start in range(0, len(rows), chunk):
            entries = rows[start:start+chunk]
            path = directory/f"block_{start:05d}.npy"
            identity = {"signature": context["signature"], "split": split, "size": size,
                        "ids": [r["image_id"] for r in entries], "source_models": plan["source_models"]}
            receipt = path.with_suffix(".json")
            if receipt.exists():
                saved = old.read(receipt)
                if saved["identity"] != identity or not path.is_file() or sha256(path) != saved["sha256"]:
                    raise old.IntegrityError("Changed scale cache block; do not silently regenerate")
                values = np.load(path, allow_pickle=False)
                cached += len(entries)
            else:
                if encoder is None:
                    probe_rows = rows[:min(batch, len(rows))]
                    original_encoder = inference.R1ScaleEncoder(context["profile_path"], 256)
                    if original_encoder.sources != plan["source_models"]:
                        raise old.IntegrityError("Original R1 source models changed")
                    reconstructed = original_encoder.encode_rows(probe_rows, context["dataset"], batch)
                    if not np.allclose(reconstructed, original[:len(probe_rows)], rtol=0, atol=atol):
                        raise old.IntegrityError("Original 256 image inference no longer matches saved R1")
                    del original_encoder
                    encoder = inference.R1ScaleEncoder(context["profile_path"], size)
                    if encoder.sources != plan["source_models"]:
                        raise old.IntegrityError("High-resolution R1 sources changed")
                    together = encoder.encode_rows(probe_rows, context["dataset"], batch)
                    alone = encoder.encode_rows(probe_rows[:3], context["dataset"], 1)
                    if not np.allclose(together[:len(alone)], alone, rtol=0, atol=atol):
                        raise old.IntegrityError("High-resolution image inference depends on batching")
                    probe = {"images": len(probe_rows), "original_max_error": float(abs(reconstructed-original[:len(probe_rows)]).max()),
                             "batch_max_error": float(abs(together[:len(alone)]-alone).max()),
                             "adapted_sha256": encoder.adapted_sha256}
                values = encoder.encode_rows(entries, context["dataset"], batch)
                dual.validate_block(values, 1536)
                if len(values) != len(entries):
                    raise old.IntegrityError("Wrong high-resolution output row count")
                pending = path.with_suffix(".pending.npy")
                np.save(pending, values)
                pending.replace(path)
                write_json(receipt, {"identity": identity, "sha256": sha256(path)})
                fresh += len(entries)
            dual.validate_block(values, 1536)
            if len(values) != len(entries):
                raise old.IntegrityError("Wrong cached feature row count")
            blocks.append(values)
            print(f"EXTRACT {split} {size}px: {min(start+chunk,len(rows))}/{len(rows)} | fresh {fresh}, cached {cached}", flush=True)
        path = directory/"features.npy"
        values = np.concatenate(blocks)
        if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), values):
            raise old.IntegrityError("Existing scale feature aggregate changed")
        if not path.exists():
            np.save(path, values)
        return {"path": str(path), "sha256": sha256(path), "split": split, "size": size,
                "ids": [r["image_id"] for r in rows], "source_models": plan["source_models"],
                "fresh_images_this_attempt": fresh, "cached_images_this_attempt": cached,
                "encoder_count": 3, "probe_this_attempt": probe}
    finally:
        del encoder
        gc.collect()


def load_features(context, split, size, report):
    directory = context["output"]/"tasks"/f"features_{split}_{size}"
    source.search.verify_task_source(context["output"], context["signature"], directory/"complete.json")
    path = directory/"features.npy"
    q, g = old.review.old.protocol_rows(context, split)
    if (report != old.read(directory/"result.json") or report["path"] != str(path) or report["sha256"] != sha256(path)
            or report["split"] != split or report["size"] != size or report["ids"] != [r["image_id"] for r in q+g]
            or report["source_models"] != context["manifest"]["multiscale_plan"]["source_models"]):
        raise old.IntegrityError("High-resolution feature provenance/order changed")
    values = np.load(path, allow_pickle=False)
    dual.validate_block(values, 1536)
    if len(values) != len(q)+len(g):
        raise old.IntegrityError("Scale feature count differs from protocol")
    return values


def reference(context, split):
    plan = context["manifest"]["multiscale_plan"]
    directory = Path(plan["source_directory"])
    task = directory/"tasks"/f"{split}_V25_control"
    source.search.verify_task_source(directory, plan["source_signature"], task/"complete.json")
    return old.read(task/"result.json"), task/"export"


def evaluate(context, split, spec, values, directory):
    if split not in ("calibration", "validation") or spec not in systems():
        raise ValueError("Use only original development splits and five frozen systems")
    if split == "validation" and spec not in search.read_selection(context)["evaluations"]:
        raise old.IntegrityError("Validation accepts only the frozen winner and v25")
    q, g = old.review.old.protocol_rows(context, split)
    threshold = source.frozen_profile(context)["threshold"]
    expected, source_export = reference(context, split)
    if threshold != expected["threshold"]:
        raise old.IntegrityError("Frozen R1 threshold differs from v25")
    ranked = inference.rank(values, len(q), spec)
    metrics = dual.policy.evaluate(q, g, ranked, threshold, "raw_top1")
    decisions = dual.policy.predictions(q, g, ranked, threshold, "raw_top1")
    _, r1 = dual.unpack(values[:, :2048])
    raw = dual.policy.rank_vectors(r1[:len(q)], r1[len(q):], "raw")
    if (decisions[1] != dual.policy.predictions(q, g, raw, threshold, "raw_top1")[1]
            or metrics["candidates"] != expected["candidates"]):
        raise old.IntegrityError("Original R1 candidate/confidence/refusals changed")
    if spec == BASELINE and metrics["ranking"] != expected["ranking"]:
        raise old.IntegrityError("Current v25 ranking no longer reproduces")
    report = {"system": spec, "split": split, "threshold": threshold, **metrics,
              "protocol_sha256": digest(context["manifest"]["protocols"][split]), "candidate_unchanged": True}
    if split == "validation":
        export = directory/"export"
        saved = old.previous.verify_or_export_csv(export, q, g, ranked, threshold, "raw_top1")
        if saved != metrics or sha256(export/"candidates.csv") != sha256(source_export/"candidates.csv"):
            raise old.IntegrityError("CSV metrics or candidates differ")
        if not np.array_equal(values[:, :2048], np.load(source_export/"embeddings.npy", allow_pickle=False)):
            raise old.IntegrityError("Original embedding blocks changed")
        path = export/"embeddings.npy"
        if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), values):
            raise old.IntegrityError("Existing export vectors changed")
        if not path.exists():
            np.save(path, values)
        if spec == BASELINE and any(sha256(export/n) != sha256(source_export/n) for n in ("submission.csv", "embeddings.npy")):
            raise old.IntegrityError("v25 control export bytes changed")
        old.review.old.freeze_json(export/"embedding_order.json", {
            "ids": [r["image_id"] for r in q+g], "query_count": len(q), "gallery_count": len(g),
            "dimension": values.shape[1], "original_slice": [0, 2048], "original_layout": dual.LAYOUT,
            "high_resolution_r1_slice": [2048, 3584] if spec != BASELINE else None,
            "ranking": spec, "candidate": dual.ROLES["candidate"], "threshold": threshold, "sha256": sha256(path),
            "ranking_source_sha256": context["manifest"]["source_sha256"]["training/multiscale_inference.py"]})
        replay = inference.rank(np.load(path, allow_pickle=False), len(q), spec)
        if dual.policy.predictions(q, g, replay, threshold, "raw_top1") != decisions:
            raise old.IntegrityError("NPY replay changed decisions")
        report.update(export=str(export), **dual.policy.query_diagnostics(q, g, ranked))
    print(f"{split} {spec['name']}: mAP={metrics['ranking']['mAP@10']:.6f}; original R1 candidates unchanged", flush=True)
    return report


def write_report(context, result):
    lines = ["# v31 — multi-scale OSNet", "", f"Статус: {result['status']}",
             "Те же веса R1: 256 + 320/384 px, доля высокого разрешения внутри R1 25/50%.",
             "MVP/R1 остаётся 50/50, граф 20/3/λ0.50. Входной bbox не меняется."]
    for key, title in (("calibration", "Calibration: пять вариантов"), ("evaluations", "Validation: контроль и один выбор")):
        lines += ["", f"## {title}", "", "| Вариант | mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
        for name, r in result[key].items():
            lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['ranking']['Rank-1']:.6f} | "
                         f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |" if r else f"| {name} | FAILED | — | — | — |")
    lines += ["", f"Выбор: {result.get('selection', {}).get('selected', {}).get('name', 'не завершён')}"]
    if result.get("selected_vs_v25"):
        d = result["selected_vs_v25"]
        lines += [f"ΔmAP против v25: {d['mAP_delta']:+.6f}; AP лучше/хуже/равно: {d['improved']}/{d['worsened']}/{d['unchanged']}."]
    lines += ["", "Это адаптивный development-поиск, не независимый тест; прирост не гарантирован.",
              "Веса, bbox, candidate/confidence/threshold, evaluator и рабочий MVP неизменны. Обучения и автоматического продвижения нет.",
              "Изменяется только размер подачи того же изображения; деталей сверх разрешения исходного crop метод не создаёт.",
              "Номерная зона не обрабатывается; отсутствие остаточного сигнала номера не доказано.",
              "Экспорт: реальные float32-векторы, 2048 для контроля / 3584 для multi-scale; top-10 сохраняется при отказах.",
              "Ограничение junk/top-10 сохраняется. Дополнительные encoder passes увеличивают стоимость; это не официальный benchmark.",
              f"Источники неизменны: {result.get('protected_unchanged', False)}. Время текущего вызова: {result.get('elapsed_seconds', 0)/60:.1f} мин."]
    (context["output"]/"REPORT.md").write_text("\n".join(lines)+"\n")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True required for development evaluation")
    if runtime() != context["manifest"]["scale_runtime"]:
        raise old.IntegrityError("Runtime changed; use the original research environment")
    queue = old.Queue(context, wall_hours=None)
    result = {"signature": context["signature"], "status": "running", "calibration": {}, "evaluations": {}, "features": {},
              "optimizer_updates": 0, "bn_updates": 0, "threshold_fit": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            check_inputs(context)
            for split in ("calibration", "validation"):
                if split == "validation":
                    frozen = search.selection(context, result["calibration"])
                    old.review.old.freeze_json(context["output"]/"frozen_selection.json", frozen)
                    result["selection"] = search.read_selection(context)
                    print(f"\nFROZEN: {frozen['selected']['name']}; validation control + one winner only", flush=True)
                specs = systems() if split == "calibration" else frozen["evaluations"]
                original = source.load_vectors(context, split)
                feature_cache = {}
                for index, spec in enumerate(specs, 1):
                    values = original
                    if spec != BASELINE:
                        size = spec["size"]
                        if size not in feature_cache:
                            task = f"features_{split}_{size}"
                            feature = queue.task(task, lambda d: features(context, split, size, d))
                            result["features"][task] = feature
                            feature_cache[size] = (None if feature is None else
                                np.concatenate([original, load_features(context, split, size, feature)], axis=1))
                        values = feature_cache[size]
                    print(f"\n{split.upper()} {index}/{len(specs)} {spec['name']} | elapsed {time.monotonic()-queue.started:.1f}s", flush=True)
                    result["calibration" if split == "calibration" else "evaluations"][spec["name"]] = (
                        None if values is None else queue.task(f"{split}_{spec['name']}",
                            lambda d: evaluate(context, split, spec, values, d)))
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
