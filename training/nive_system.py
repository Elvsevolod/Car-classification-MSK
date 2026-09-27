"""v34: preserve concrete v33 weights, export them, compare five fixed outer systems."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import copy
import gc
from pathlib import Path
import re
import time

import numpy as np
import torch

from training import nive_mixed as base, nive_system_inference as inference

VARIANT = base.ROOT / "OSNet-AIN-x1.0/variant_34_nive_system"


def protocol_rows(context, split):
    if split not in ("calibration", "validation"):
        raise ValueError("Only original development protocols are supported")
    protocol = context["manifest"]["protocols"][split]
    by_id = {r["image_id"]: r for r in context["rows"]}
    return tuple([by_id[i] for i in protocol[key]] for key in ("query_ids", "gallery_ids"))


def validate_splits(rows, protocols, outer, trained):
    groups = {k: set(v) for k, v in outer.items()}
    if not trained or not set(trained) <= groups["train"] or any(
            groups[a] & groups[b] for a, b in (("train", "calibration"), ("train", "validation"), ("calibration", "validation"))):
        raise ValueError("Encoder training/outer identity leakage")
    lookup = {r["image_id"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate organizer IDs")
    used = set()
    for split in ("calibration", "validation"):
        p = protocols[split]; ids = p["query_ids"] + p["gallery_ids"]
        if (not p["query_ids"] or len(p["gallery_ids"]) < 10 or len(set(ids)) != len(ids)
                or used & set(ids) or not {lookup[i]["vehicle_id"] for i in ids} <= groups[split]):
            raise ValueError("Changed original query/gallery split")
        used.update(ids)


def check_inputs(context):
    m = context["manifest"]
    if base.digest(m) != context["signature"] or m["systems"] != inference.SYSTEMS:
        raise ValueError("Frozen v34 context changed")
    base.verify_files(m["protected"]); base.verify_files(m["source_sha256"])


def prepare(run_name="system_v1"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple new RUN_NAME")
    settings_path = VARIANT / "configs/system_v1.json"
    settings = base.old.load_json(settings_path)
    if (settings["provider"] != "CPUExecutionProvider" or settings["vector_atol"] != 2e-5
            or settings["optimizer_updates"] or settings["bn_updates"] or settings["threshold_fit"]
            or settings["promoted"] or settings["wall_time_limit"] is not None
            or {k: v["step"] for k, v in settings["models"].items()} != {"parent": 0, "N0": 1600, "N1": 1800}):
        raise ValueError("Use the concrete fixed v33 candidate, without new training or retuning")
    source = base.ROOT / settings["source_run"]
    base.verify_files({str(source / "manifest.json"): settings["source_manifest_sha256"],
                       str(source / "results.json"): settings["source_results_sha256"]})
    sm, result = (base.old.load_json(source / f) for f in ("manifest.json", "results.json"))
    if result["status"] != "complete" or result["signature"] != base.digest(sm) or sm["plan"]["aux_weight"] != .025:
        raise ValueError("Need completed unchanged v33")
    print("PREFLIGHT: verify v33 checkpoints, v25 and original splits (read-only)", flush=True)
    base.verify_files(sm["protected"]); base.verify_files(sm["source_sha256"])
    v25, v24 = (base.ROOT / settings[k] for k in ("v25_run", "v24_run"))
    for path in (v25 / "manifest.json", v25 / "results.json", v24 / "dual_role_profile.json"):
        if str(path) not in sm["protected"]:
            raise ValueError("Unprotected baseline provenance")
    baseline = base.old.load_json(v25 / "manifest.json")
    profile, _, _ = inference.dual.load_profile(v24 / "dual_role_profile.json")
    active = base.old.load_json(base.APP / "release_decision.json")
    if active["active_profile"] != "MVP_fusion_v25":
        raise ValueError("Active MVP changed; review the control before running")
    rows = base.read_rows(base.DATASET / "train.csv")
    split = base.old.load_json(base.ARTIFACTS / "splits.json")
    base.parent_provenance(sm["plan"], rows, split)
    validate_splits(rows, baseline["protocols"], split["identities"], sm["parent"]["train_ids"])
    external, inventory = base.data.audit_nive(base.ROOT / "NiVe1303", split["frame_sha256"].values())
    if inventory["files_fingerprint"] != sm["nive"]["files_fingerprint"]:
        raise ValueError("NiVe provenance changed")
    excluded = {r["path"] for r in sm["nive"]["excluded"]}
    target, external = base.data.label_domains(rows, [r for r in external if r["path"] not in excluded], set(sm["parent"]["train_ids"]))
    if base.digest(external) != sm["nive"]["used_train_fingerprint"]:
        raise ValueError("Source identity labels changed")
    selected = {}
    for name, spec in settings["models"].items():
        cp = result["training"][spec["arm"]]["checkpoints"][str(spec["step"])]
        if cp["sha256"] != spec["sha256"]:
            raise ValueError("Wrong concrete checkpoint (do not use the step0 N1 export)")
        base.verify_files({cp["path"]: cp["sha256"]})
        payload = torch.load(cp["path"], map_location="cpu", weights_only=True)
        if (payload["step"], payload["arm"], payload["signature"]) != (spec["step"], spec["arm"], result["signature"]):
            raise ValueError("Checkpoint provenance differs")
        selected[name] = {**cp, **spec}
    protected = dict(sm["protected"])
    protected.update({str(p): base.sha256(p) for p in source.rglob("*") if p.is_file()})
    sources = {**sm["source_sha256"], **{str(p): base.sha256(p) for p in
               (Path(__file__).resolve(), Path(inference.__file__).resolve(), settings_path)}}
    manifest = {"version": 34, "settings": settings, "systems": copy.deepcopy(inference.SYSTEMS), "models": selected,
                "protocols": baseline["protocols"], "outer": split["identities"], "trained_ids": sm["parent"]["train_ids"],
                "source_signature": result["signature"], "source_directory": str(source),
                "v25_directory": str(v25), "v24_directory": str(v24), "threshold": profile["threshold"],
                "selection": "None: all five cases fixed before original calibration/validation evaluation",
                "scope": "post-hoc development comparison of existing weights; no full-train refit or hidden-test claim",
                "runtime": {"numpy": np.__version__, "torch_export": str(torch.__version__),
                            "onnxruntime": inference.frozen.ort.__version__, "provider": settings["provider"]},
                "protected": protected, "source_sha256": sources, "promoted": False}
    output = VARIANT / "runs" / run_name
    if not output.resolve().is_relative_to((VARIANT / "runs").resolve()):
        raise ValueError("Output escapes v34")
    context = {"output": output, "manifest": manifest, "signature": base.digest(manifest), "rows": rows,
               "dataset": base.DATASET, "source_context": {"manifest": sm, "signature": result["signature"],
               "rows": rows, "target": target, "external": external, "config": base.ExperimentConfig(**sm["config"]),
               "variant": base.Ablation(**sm["variant"]), "dataset": base.DATASET, "nive_root": base.ROOT / "NiVe1303",
               "plan": sm["plan"], "device": torch.device("cpu"), "masks": {}}}
    with base.old.run_lock(output):
        base.old.freeze_json(output / "manifest.json", manifest)
        base.old.freeze_json(output / "frozen_candidate.json", {"source": selected, "systems": inference.SYSTEMS,
                             "threshold": profile["threshold"], "promoted": False, "source_was_post_hoc_inner": True})
    return context


def task(context, name, action):
    directory = context["output"] / "tasks" / name
    receipt = directory / "complete.json"
    if receipt.exists():
        saved = base.old.load_json(receipt)
        if saved["signature"] != context["signature"]:
            raise ValueError("Task belongs to another experiment")
        base.verify_files({str(directory / p): h for p, h in saved["files"].items()})
        print(f"STAGE {name}: verified saved result (not a new measurement)", flush=True)
        return base.old.load_json(directory / "result.json")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"STAGE {name}: start", flush=True)
    result = action(directory)
    base.old.freeze_json(directory / "result.json", result)
    base.write_json(receipt, {"signature": context["signature"], "files": {
        str(p.relative_to(directory)): base.sha256(p) for p in directory.rglob("*")
        if p.is_file() and p != receipt and not p.name.endswith((".tmp", ".pending.npy"))}})
    print(f"STAGE {name}: complete in {time.perf_counter()-started:.1f}s", flush=True)
    return result


def export_models(context):
    entries = {}
    for name, spec in context["manifest"]["models"].items():
        c = {**context["source_context"], "output": context["output"] / "encoders" / name}
        receipt = base.export_encoder(c, spec["arm"], {k: spec[k] for k in ("path", "sha256")})
        path = c["output"] / "exports" / spec["arm"] / "encoder.onnx"
        entries[name] = {"path": os.path.relpath(path, context["output"]), "sha256": receipt["files"]["encoder.onnx"],
                         "source": spec, "parity": {k: receipt[k] for k in ("max_onnx_error", "max_batch_error")}}
        print(f"EXPORT {name}: step {spec['step']}, head-free 512D", flush=True)
    profile = Path(context["manifest"]["v24_directory"]) / "dual_role_profile.json"
    bundle = {"schema": "nive-system-v34", "encoders": entries, "systems": inference.SYSTEMS, "layout": inference.LAYOUT,
              "preprocessing": inference.frozen._preprocessing(256, "square"),
              "v25_profile": {"path": os.path.relpath(profile, context["output"]), "sha256": base.sha256(profile)},
              "threshold": context["manifest"]["threshold"], "promoted": False}
    base.old.freeze_json(context["output"] / "bundle.json", bundle)
    inference.load_bundle(context["output"] / "bundle.json")
    return bundle


def features(context, bundle, rows, split, name):
    def extract(directory):
        entry = bundle["encoders"][name]; model = context["output"] / entry["path"]
        blocks, encoder = [], None
        chunk, batch = (context["manifest"]["settings"][k] for k in ("chunk_size", "batch_size"))
        for start in range(0, len(rows), chunk):
            block = rows[start:start+chunk]; path = directory / f"block_{start:05d}.npy"
            receipt = path.with_suffix(".json")
            identity = {"signature": context["signature"], "model": entry["sha256"], "ids": [r["image_id"] for r in block]}
            cached = receipt.exists()
            if cached:
                saved = base.old.load_json(receipt)
                if saved["identity"] != identity:
                    raise ValueError("Feature block identity changed")
                base.verify_files({str(path): saved["sha256"]})
                values = np.load(path, allow_pickle=False)
            else:
                if encoder is None:
                    encoder = inference.ImageEncoder(model, entry["sha256"])
                values = encoder.encode_rows(block, context["dataset"], batch)
                pending = path.with_suffix(".pending.npy"); np.save(pending, values); pending.replace(path)
                base.write_json(receipt, {"identity": identity, "sha256": base.sha256(path)})
            inference.dual.validate_block(values, 512)
            if len(values) != len(block):
                raise ValueError("Wrong cached image count")
            blocks.append(values)
            print(f"EXTRACT {split}/{name}: {min(start+chunk,len(rows))}/{len(rows)} | {'cache' if cached else 'fresh'}", flush=True)
        combined = np.concatenate(blocks); path = directory / "features.npy"
        if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), combined):
            raise ValueError("Existing feature aggregate changed")
        if not path.exists(): np.save(path, combined)
        del encoder; gc.collect()
        return {"path": str(path), "sha256": base.sha256(path), "ids": [r["image_id"] for r in rows]}
    saved = task(context, f"features_{split}_{name}", extract)
    if saved["ids"] != [r["image_id"] for r in rows]:
        raise ValueError("Feature order differs from protocol")
    return np.load(saved["path"], allow_pickle=False)


def compare_inner_decisions(before, after):
    """Exact deployed graph ranking; pair-raw is diagnostic, never the v25 candidate."""
    if set(before["draws"]) != set(after["draws"]):
        raise ValueError("Inner draws changed")
    changes = []
    for draw, left in before["draws"].items():
        right = after["draws"][draw]
        if set(left["per_query"]) != set(right["per_query"]):
            raise ValueError("Inner query IDs changed")
        for qid, q in left["per_query"].items():
            other = right["per_query"][qid]
            if q["ranking"]["top10"] != other["ranking"]["top10"]:
                raise ValueError("ONNX changed deployed inner top10; do not loosen the gate")
            if q["raw"]["top10"] != other["raw"]["top10"]:
                changes.append({"draw": draw, "query_id": qid, "before": q["raw"], "after": other["raw"]})
    if before["mean_map"] != after["mean_map"]:
        raise ValueError("Deployed inner ranking metric changed")
    return changes


def inner_parity(context, bundle):
    def check(directory):
        source = Path(context["manifest"]["source_directory"])
        c = context["source_context"]; rows = base.development_rows(c)
        values, saved_values = {}, {}
        for name, spec in context["manifest"]["models"].items():
            values[name] = features(context, bundle, rows, "inner", name)
            label = "N_ref" if name == "parent" else f"{spec['arm']}_{spec['step']:05d}"
            cache = source / "evaluation" / label
            if base.old.load_json(cache / "order.json") != [r["image_id"] for r in rows]:
                raise ValueError("Inner row order changed")
            saved_values[name] = np.load(cache / "features.npy", allow_pickle=False)
            if not np.allclose(values[name], saved_values[name], rtol=0, atol=2e-5):
                raise ValueError("ONNX vs original inner embedding parity failed")
        reports, diagnostic_changes = {}, {}
        for name in ("N0", "N1"):
            before = base.score_features(c, base.normalize(np.concatenate([saved_values["parent"], saved_values[name]], axis=1)), rows)
            after = base.score_features(c, base.normalize(np.concatenate([values["parent"], values[name]], axis=1)), rows)
            diagnostic_changes[name] = compare_inner_decisions(before, after)
            if abs(before["mean_map"] - context["manifest"]["settings"]["inner_expected_map"][name]) > 1e-12:
                raise ValueError("Fixed post-hoc inner candidate was not reproduced")
            reports[name] = after
            print(f"INNER {name}: {after['mean_map']:.6f}, deployed graph top10 reproduced; "
                  f"unused pair-raw changes={len(diagnostic_changes[name])}", flush=True)
        return {"status": "passed", "pairs": reports, "source_post_hoc": True,
                "deployed_ranking_top10_equal": True, "raw_diagnostic_changes": diagnostic_changes,
                "candidate_policy": "pair raw is NOT used; actual v25 candidates checked byte-for-byte on original splits",
                "max_vector_errors": {k: float(abs(values[k]-saved_values[k]).max()) for k in values}}
    return task(context, "inner_parity", check)


def baseline_features(context, split, query, gallery):
    directory = Path(context["manifest"]["v24_directory"]) / "tasks" / f"cached_{split}" / "export"
    order = base.old.load_json(directory / "embedding_order.json")
    values = np.load(directory / "embeddings.npy", allow_pickle=False)
    if order["ids"] != [r["image_id"] for r in query+gallery]:
        raise ValueError("Cached v25 row order differs")
    inference.dual.unpack(values)
    return values, directory


def runtime_probe(context, bundle):
    """Small fresh image check of the old cache and CPU batch-independent extraction."""
    def check(directory):
        query, gallery = protocol_rows(context, "calibration")
        expected, _ = baseline_features(context, "calibration", query, gallery)
        rows = query[:8]
        encoder = inference.dual.DualRoleEncoder(Path(context["manifest"]["v24_directory"]) / "dual_role_profile.json")
        actual = encoder.encode_rows(rows, context["dataset"], 8)
        if not np.allclose(actual, expected[:len(rows)], rtol=0, atol=2e-5):
            raise ValueError("Fresh v25 images do not reproduce protected feature cache")
        errors = {"v25_cache": float(abs(actual-expected[:len(rows)]).max())}
        single = encoder.encode_rows(rows, context["dataset"], 1)
        errors["v25_batch"] = float(abs(single-actual).max())
        if not np.allclose(single, actual, rtol=0, atol=2e-5):
            raise ValueError("v25 batch feature parity failed")
        del encoder
        for name, entry in bundle["encoders"].items():
            encoder = inference.ImageEncoder(context["output"] / entry["path"], entry["sha256"])
            single = encoder.encode_rows(rows, context["dataset"], 1)
            actual = encoder.encode_rows(rows, context["dataset"], 8)
            if not np.allclose(single, actual, rtol=0, atol=2e-5):
                raise ValueError(f"New encoder {name} batch parity failed")
            errors[name] = float(abs(single-actual).max())
            del encoder
        return {"status": "passed", "images": len(rows), "max_errors": errors,
                "scope": "fresh CPU feature/cache probe; not a full device benchmark"}
    return task(context, "runtime_probe", check)


def evaluate(context, split, system, values, directory):
    q, g = protocol_rows(context, split)
    threshold = context["manifest"]["threshold"]
    ranked = inference.rank(values, len(q), system)
    report = {"system": system, "split": split, "threshold": threshold,
              **base.policy.evaluate(q, g, ranked, threshold, "raw_top1"),
              **base.policy.query_diagnostics(q, g, ranked)}
    expected = base.old.load_json(Path(context["manifest"]["v25_directory"]) / "tasks" /
                                 f"{split}_r1w50_k20_q3_l50" / "result.json")
    if report["candidates"] != expected["candidates"] or threshold != expected["threshold"]:
        raise ValueError("Frozen v25 candidates/refusals changed")
    if system == "V25_control" and report["ranking"] != expected["ranking"]:
        raise ValueError("v25 ranking no longer reproduces")
    export = directory / "export"
    if not export.exists(): inference.export_arrays(export, q, g, values, system, threshold)
    ordered, accepted = base.policy.predictions(q, g, ranked, threshold, "raw_top1")
    loaded_candidates = base.official.load_candidates(export / "candidates.csv")
    if (base.official.load_submission(export / "submission.csv", {r['image_id'] for r in g}) != ordered
            or {k: v[0][0] for k, v in loaded_candidates.items()} != {k: v[0][0] for k, v in accepted.items()}
            or not np.array_equal(np.load(export / "embeddings.npy", allow_pickle=False), values)):
        raise ValueError("Existing export differs from exact decisions/vectors")
    _, baseline_dir = baseline_features(context, split, q, g)
    if base.sha256(export / "candidates.csv") != base.sha256(baseline_dir / "candidates.csv"):
        raise ValueError("Candidate CSV bytes differ from v25")
    if split == "validation" and system == "V25_control":
        original = Path(context["manifest"]["v25_directory"]) / "tasks/validation_r1w50_k20_q3_l50/export"
        base.verify_files({str(export / f): base.sha256(original / f) for f in ("submission.csv", "candidates.csv", "embeddings.npy")})
    print(f"EVAL {split}/{system}: mAP={report['ranking']['mAP@10']:.6f}; v25 candidates unchanged", flush=True)
    return report


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True required for original development comparison")
    check_inputs(context)
    with base.old.run_lock(context["output"]):
        bundle = export_models(context)
        inner = inner_parity(context, bundle)
        probe = runtime_probe(context, bundle)
        reports = {}
        for split in ("calibration", "validation"):
            q, g = protocol_rows(context, split); original, _ = baseline_features(context, split, q, g)
            components = {n: features(context, bundle, q+g, split, n) for n in ("parent", "N0", "N1")}
            reports[split] = {}
            for system, spec in inference.SYSTEMS.items():
                values = original if system == "V25_control" else np.concatenate([
                    original, components["parent"], components[spec["member"]]], axis=1)
                reports[split][system] = task(context, f"{split}_{system}",
                    lambda d, s=system, v=values: evaluate(context, split, s, v, d))
            check_inputs(context)
        result = {"status": "complete", "signature": context["signature"], "inner": inner, "runtime_probe": probe, "evaluations": reports,
                  "optimizer_updates": 0, "bn_updates": 0, "threshold_fit": False, "promoted": False,
                  "protected_unchanged": True, "candidate_unchanged": True,
                  "scope": "fixed post-hoc development evaluation of concrete inner-trained checkpoints, not full refit"}
        lines = ["# v34 — сохранённая NiVe-смесь против v25", "", "Обучения, подбора порога и переключения MVP не было.",
                 "Сохранённые parent/N0/N1 обучались на 740 train-ID; эти ID не пересекаются с original calibration/validation.",
                 "v25 использует собственный full-train ансамбль. Сравниваются готовые системы, не одинаковый объём обучения.",
                 "", "| Split | Система | mAP@10 | Δ к v25, п.п. | Rank-1 | F1 | TNR |", "|---|---|---:|---:|---:|---:|---:|"]
        for split, systems in reports.items():
            control = systems["V25_control"]["ranking"]["mAP@10"]
            for name, r in systems.items():
                lines.append(f"| {split} | {name} | {r['ranking']['mAP@10']:.6f} | {100*(r['ranking']['mAP@10']-control):+.4f} | "
                             f"{r['ranking']['Rank-1']:.6f} | {r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |")
        lines += ["", "Пять конфигураций зафиксированы до этого запуска; нет выбора по calibration перед просмотром validation.",
                  f"Inner graph top-10 совпали точно; изменения диагностического raw пары: { {k: len(v) for k, v in inner['raw_diagnostic_changes'].items()} }.",
                  "Эти raw-списки не используются для candidate/отказа. Полные расхождения сохранены в results.json; это не заявление о побитовой переносимости всех внутренних оценок.",
                  "Исходная validation уже многократно использовалась: это development, не независимый тест.",
                  "N0/1600 и N1/1800 выбраны после inner-анализа: это контроль конкретных кандидатов, не причинная оценка NiVe при равном числе обновлений.",
                  "R1_N* — равная пара с графом 20/3/.75; MVP_R1_N* — .50 MVP + .25 parent + .25 N* с графом 20/3/.50.",
                  "Кандидат/уверенность/отказ всегда от прежнего full-train R1 equal3 и его порога. CSV кандидатов совпадает побайтно.",
                  "3072D export — реальные блоки MVP512 + candidate1536 + parent512 + member512; это не один cosine ranking-vector.",
                  "NPY replay использует явные slices/веса. Не утверждается, что вспомогательная mINP raw-bank совпадает с graph mAP.",
                  "У полной новой системы 6 encoder forwards вместо 4 у v25: сохранён старый candidate-ансамбль. GPU/скорость не подтверждены.",
                  "Номерная устойчивость не доказана; OCR/обработка номеров не добавлялись. Junk/top10 и evaluator не менялись.",
                  "Для нового профиля/замены MVP требуется отдельное решение. Старые v32/v33 отчёты не переписаны."]
        path = context["output"] / "REPORT.md"; text = "\n".join(lines) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text: raise ValueError("Report changed")
        if not path.exists():
            pending = path.with_suffix(".md.tmp"); pending.write_text(text, encoding="utf-8"); pending.replace(path)
        base.old.freeze_json(context["output"] / "results.json", result)
    return result
