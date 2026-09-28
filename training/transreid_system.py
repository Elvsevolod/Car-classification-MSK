"""v37 Run All: frozen T12/T22, calibration-only mixture search, one outer winner."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import copy
import gc
import importlib.metadata
from pathlib import Path
import platform
import re
import shutil
import tempfile
import time

import numpy as np
import torch

from training import nive_mixed as base, nive_system as previous
from training import transreid_model as vision, transreid_system_inference as inference

VARIANT = base.ROOT / "OSNet-AIN-x1.0/variant_37_transreid_system"
protocol_rows = previous.protocol_rows
baseline_features = previous.baseline_features


def check_inputs(c):
    m = c["manifest"]
    if (base.digest(m) != c["signature"] or m["systems"] != inference.SYSTEMS
            or m["graph"] != inference.GRAPH or str(c["device"]) != m["runtime"]["device"]):
        raise ValueError("Frozen v37 context changed")
    print(f"CHECK: {len(m['protected'])} protected files and frozen source", flush=True)
    base.verify_files(m["protected"])
    base.verify_files(m["source_sha256"])


def prepare(run_name="system_v1", device="mps"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple RUN_NAME")
    device = vision.device_for(device)
    path = VARIANT / "configs/system_v1.json"
    settings = base.old.load_json(path)
    if (settings["optimizer_updates"] or settings["bn_updates"] or settings["threshold_fit"] or settings["promoted"]
            or settings["wall_time_limit"] is not None or settings["vector_atol"] != 2e-5
            or settings["batch_size"] != 16 or settings["selection_split"] != "calibration"
            or set(settings["models"]) != set(inference.DIMENSIONS)):
        raise ValueError("Use frozen v37 settings; no training/threshold fitting")
    source = base.ROOT / settings["source_run"]
    pins = {str(source / f"{name}.json"): settings[f"source_{name}_sha256"]
            for name in ("manifest", "results", "complete")}
    base.verify_files(pins)
    sm, result, complete = (base.old.load_json(source / f"{n}.json") for n in ("manifest", "results", "complete"))
    if (result["status"] != "complete" or result["signature"] != base.digest(sm)
            or complete["signature"] != result["signature"] or result["failures"]
            or result["promoted"] or result["original_outer_evaluation"] or result["threshold_fit"]):
        raise ValueError("Need the completed unchanged inner-only v36")
    models = {}
    for name, spec in settings["models"].items():
        matching = [r for r in result["leaderboard"] if r["trial_id"] == spec["trial_id"] and r["step"] == spec["step"]]
        if len(matching) != 1 or matching[0]["checkpoint"]["sha256"] != spec["sha256"]:
            raise ValueError("Wrong concrete v36 checkpoint")
        trial = next(t for t in sm["trials"] if t["id"] == spec["trial_id"])
        if trial["architecture"] != spec["architecture"] or spec["step"] != 1800:
            raise ValueError("Changed shortlist architecture/step")
        models[name] = {**spec, "path": matching[0]["checkpoint"]["path"], "trial": trial}
    protected = {**sm["protected"], **pins, **{str(source / p): h for p, h in complete["files"].items()}}
    base.verify_files(protected)
    base.verify_files(sm["source_sha256"])
    v25, v24 = (base.ROOT / settings[k] for k in ("v25_run", "v24_run"))
    for p in (v25 / "manifest.json", v25 / "frozen_selection.json", v24 / "dual_role_profile.json"):
        if str(p) not in protected:
            raise ValueError("Unprotected v25 control")
    selected = base.old.load_json(v25 / "frozen_selection.json")
    if selected["selected"] != inference.map_inference.SPEC or selected["selection_split"] != "calibration":
        raise ValueError("v25 control selection changed")
    baseline = base.old.load_json(v25 / "manifest.json")
    profile, _, _ = inference.dual.load_profile(v24 / "dual_role_profile.json")
    product = base.ROOT.parent / "Car-classification-MSK-main"
    if base.old.load_json(product / "release_decision.json")["active_profile"] != "MVP_fusion_v25":
        raise ValueError("Active main MVP changed; review the baseline")
    protected[str(product / "release_decision.json")] = base.sha256(product / "release_decision.json")
    for p in (product / "models").rglob("*"):
        if p.is_file(): protected[str(p)] = base.sha256(p)
    rows = base.read_rows(base.DATASET / "train.csv")
    splits = base.old.load_json(base.ARTIFACTS / "splits.json")
    previous.validate_splits(rows, baseline["protocols"], splits["identities"], sm["train_ids"])
    held = set(sm["holdout_ids"])
    if held & set(sm["train_ids"]) or held | set(sm["train_ids"]) != set(splits["identities"]["train"]):
        raise ValueError("v36 inner train/holdout provenance changed")
    outer_ids = {i for p in baseline["protocols"].values() for k in ("query_ids", "gallery_ids") for i in p[k]}
    outer_paths = vision.image_paths(base.DATASET, [r for r in rows if r["image_id"] in outer_ids])
    if any(str(p) not in protected for p in outer_paths.values()):
        raise ValueError("An outer input is outside the original image byte protection")
    sources = {**sm["source_sha256"], **{str(p): base.sha256(p) for p in
        (Path(__file__).resolve(), Path(inference.__file__).resolve(), path)}}
    runtime = {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "numpy", "pillow", "onnxruntime")}
    runtime.update(device=str(device), python=platform.python_version(), platform=platform.platform(),
                   torch_threads=torch.get_num_threads(), dtype="float32", amp=False, compile=False)
    manifest = {"version": 37, "settings": settings, "systems": copy.deepcopy(inference.SYSTEMS), "graph": inference.GRAPH,
        "models": models, "train_ids": sm["train_ids"], "holdout_ids": sm["holdout_ids"], "preprocessing": vision.PREPROCESS,
        "protocols": baseline["protocols"], "threshold": profile["threshold"], "source_signature": result["signature"],
        "source_directory": str(source), "v25_directory": str(v25), "v24_directory": str(v24),
        "inner_image_order": sm["image_order"], "drop_path": sm["settings"]["drop_path"],
        "runtime": runtime, "protected": protected, "source_sha256": sources,
        "selection": "maximum calibration mAP@10; exact ties in fixed system order, control first",
        "validation": "control and one frozen calibration winner only", "promoted": False}
    output = VARIANT / "runs" / run_name
    if shutil.disk_usage(VARIANT).free < 1024**3:
        raise RuntimeError("Need at least 1 GiB free; no historical files will be removed")
    c = {"output": output, "manifest": manifest, "signature": base.digest(manifest),
         "rows": rows, "dataset": base.DATASET, "device": device}
    with base.old.run_lock(output):
        base.old.freeze_json(output / "manifest.json", manifest)
    print(f"PREFLIGHT OK: {len(inference.SYSTEMS)} systems, {len(sm['train_ids'])} trained IDs, "
          f"device={device}; original validation stays closed until selection", flush=True)
    return c


def task(c, name, action):
    """Publish a whole stage atomically; completed stages are verified before reuse."""
    target = c["output"] / "tasks" / name
    if target.exists():
        saved = base.old.load_json(target / "complete.json")
        if saved["signature"] != c["signature"] or "result.json" not in saved["files"]:
            raise ValueError("Stage belongs to another experiment")
        base.verify_files({str(target / p): h for p, h in saved["files"].items()})
        print(f"STAGE {name}: verified saved result (not a new measurement)", flush=True)
        return base.old.load_json(target / "result.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"STAGE {name}: start", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".pending_{name}_", dir=target.parent) as temp:
        directory = Path(temp) / "stage"
        directory.mkdir()
        result = action(directory)
        base.write_json(directory / "result.json", result)
        base.write_json(directory / "complete.json", {"signature": c["signature"], "files": {
            str(p.relative_to(directory)): base.sha256(p) for p in directory.rglob("*") if p.is_file()}})
        directory.rename(target)
    print(f"STAGE {name}: complete in {time.perf_counter()-started:.1f}s", flush=True)
    return result


def load_model(c, member):
    spec = c["manifest"]["models"][member]
    base.verify_files({spec["path"]: spec["sha256"]})
    payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
    if (payload["signature"] != c["manifest"]["source_signature"] or payload["trial"] != spec["trial"]
            or payload["step"] != spec["step"]):
        raise ValueError("Checkpoint provenance differs")
    # No initialization download and no optimizer: restore every tensor strictly.
    model = vision.ReIDModel(len(c["manifest"]["train_ids"]), spec["architecture"],
                             pretrained=False, drop_path=c["manifest"]["drop_path"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(c["device"]).eval().requires_grad_(False)


def release_device(c):
    gc.collect()
    if c["device"].type == "mps": torch.mps.empty_cache()
    if c["device"].type == "cuda": torch.cuda.empty_cache()


def runtime_probe(c):
    def action(directory):
        query, gallery = protocol_rows(c, "calibration")
        expected, _ = baseline_features(c, "calibration", query, gallery)
        encoder = inference.dual.DualRoleEncoder(Path(c["manifest"]["v24_directory"]) / "dual_role_profile.json")
        actual = encoder.encode_rows(query[:8], c["dataset"], 8)
        if not np.allclose(actual, expected[:8], rtol=0, atol=2e-5):
            raise ValueError("Fresh v25 images do not reproduce the protected cache")
        errors = {"v25_cache": float(abs(actual-expected[:8]).max())}
        del encoder
        lookup = {r["image_id"]: r for r in c["rows"]}
        rows = [lookup[i] for i in c["manifest"]["inner_image_order"][:32]]
        paths = vision.image_paths(c["dataset"], rows)
        for member, spec in c["manifest"]["models"].items():
            model = load_model(c, member)
            before = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
            source = Path(c["manifest"]["source_directory"]) / "tasks" / f"evaluate_{spec['trial_id']}_{spec['step']:05d}" / "features.npy"
            saved = np.load(source, allow_pickle=False)[:32]
            errors[member] = {}
            for batch in (1, 8, 16, 32):
                actual = vision.encode(model, rows, paths, c["device"], batch, progress=False)
                error = float(abs(actual-saved).max())
                if not np.allclose(actual, saved, rtol=0, atol=2e-5):
                    raise ValueError(f"{member} saved features/batch {batch} differ")
                errors[member][str(batch)] = error
                print(f"PROBE {member}, batch={batch}: max error={error:.3g}", flush=True)
            reverse = vision.encode(model, rows[::-1], paths, c["device"], 16, progress=False)[::-1]
            alone = vision.encode(model, rows[:1], paths, c["device"], 1, progress=False)
            if not np.allclose(reverse, saved, rtol=0, atol=2e-5) or not np.allclose(alone, saved[:1], rtol=0, atol=2e-5):
                raise ValueError("Query extraction depends on order/neighbors")
            if any(not torch.equal(v.cpu(), before[k]) for k, v in model.named_buffers()):
                raise ValueError("Inference changed a model buffer")
            del model
            release_device(c)
        return {"status": "passed", "max_errors": errors, "bn_updates": 0,
                "scope": "fresh feature/batch probe, not a latency or GPU benchmark"}
    return task(c, "runtime_probe", action)


def select_calibration(reports):
    if set(reports) != set(inference.SYSTEMS):
        raise ValueError("All nine calibration systems must finish before selection")
    for name, r in reports.items():
        if r["split"] != "calibration" or r["system"] != name or not np.isfinite(r["ranking"]["mAP@10"]):
            raise ValueError("Selection accepts finite calibration-only results")
    return max(inference.SYSTEMS, key=lambda n: reports[n]["ranking"]["mAP@10"])


def freeze_selection(c, reports):
    selected = select_calibration(reports)
    value = {"signature": c["signature"], "selection_split": "calibration", "selected": selected,
             "spec": inference.SYSTEMS[selected], "graph": inference.GRAPH, "threshold": c["manifest"]["threshold"],
             "calibration": {n: r["ranking"]["mAP@10"] for n, r in reports.items()},
             "calibration_files": {str(c["output"] / "tasks" / f"calibration_{n}" / "result.json"):
                 base.sha256(c["output"] / "tasks" / f"calibration_{n}" / "result.json") for n in reports},
             "models": c["manifest"]["models"], "promoted": False}
    base.old.freeze_json(c["output"] / "frozen_selection.json", value)
    return value


def require_selection(c, system=None, member=None):
    path = c["output"] / "frozen_selection.json"
    if not path.is_file():
        raise ValueError("Validation is closed until calibration selection is frozen")
    value = base.old.load_json(path)
    base.verify_files(value["calibration_files"])
    reports = {n: base.old.load_json(c["output"] / "tasks" / f"calibration_{n}" / "result.json") for n in inference.SYSTEMS}
    expected = freeze_selection(c, reports)
    if value != expected or (system is not None and system not in {"V25_control", value["selected"]}):
        raise ValueError("Only control and the frozen calibration winner may access validation")
    if member is not None and member != value["spec"]["member"]:
        raise ValueError("Only the selected member may extract validation")
    return value


def features(c, rows, split, member):
    if split == "validation": require_selection(c, member=member)
    if split not in {"calibration", "validation"} or member not in inference.DIMENSIONS:
        raise ValueError("Unknown extraction split/member")
    query, gallery = protocol_rows(c, split)
    if rows != query+gallery:
        raise ValueError("Feature extraction must follow the frozen query then gallery order")
    name = f"features_{split}_{member}"
    def action(directory):
        model = load_model(c, member)
        paths = vision.image_paths(c["dataset"], rows)
        values = vision.encode(model, rows, paths, c["device"], c["manifest"]["settings"]["batch_size"])
        np.save(directory / "features.npy", values)
        del model
        release_device(c)
        return {"ids": [r["image_id"] for r in rows], "shape": list(values.shape),
                "checkpoint": c["manifest"]["models"][member]["sha256"], "split": split, "member": member}
    result = task(c, name, action)
    values = np.load(c["output"] / "tasks" / name / "features.npy", allow_pickle=False)
    inference.dual.validate_block(values, inference.DIMENSIONS[member])
    if result["ids"] != [r["image_id"] for r in rows] or len(values) != len(rows):
        raise ValueError("Cached feature order changed")
    return values


def evaluate(c, split, system, values, directory):
    if split == "validation": require_selection(c, system=system)
    q, g = protocol_rows(c, split)
    threshold = c["manifest"]["threshold"]
    ranked = inference.rank(values, len(q), system, progress=True)
    report = {"system": system, "split": split, "threshold": threshold,
              **base.policy.evaluate(q, g, ranked, threshold, "raw_top1"),
              **base.policy.query_diagnostics(q, g, ranked)}
    ordered, accepted = base.policy.predictions(q, g, ranked, threshold, "raw_top1")
    original, _ = baseline_features(c, split, q, g)
    _, r1 = inference.dual.unpack(original)
    reference_raw = base.policy.rank_vectors(r1[:len(q)], r1[len(q):], "raw")
    _, expected_accepted = base.policy.predictions(q, g, reference_raw, threshold, "raw_top1")
    if accepted != expected_accepted:
        raise ValueError("Frozen candidate IDs/confidences/refusals changed")
    previous_dir = Path(c["manifest"]["v25_directory"]) / "tasks" / f"{split}_r1w50_k20_q3_l50"
    expected = base.old.load_json(previous_dir / "result.json")
    if threshold != expected["threshold"] or report["candidates"] != expected["candidates"]:
        raise ValueError("Frozen v25 candidate metrics/threshold changed")
    if system == "V25_control":
        if report["ranking"] != expected["ranking"]:
            raise ValueError("v25 control metrics no longer reproduce")
        if split == "validation" and ordered != base.official.load_submission(previous_dir / "export/submission.csv", {r['image_id'] for r in g}):
            raise ValueError("v25 control top-10 no longer reproduces")
    for qid, top10 in ordered.items(): report["per_query"][qid]["ranking"]["top10"] = top10
    print(f"EVAL {split}/{system}: mAP={report['ranking']['mAP@10']:.6f}; candidates unchanged", flush=True)
    return report


def export_result(c, split, system, values, report):
    if split == "validation": require_selection(c, system=system)
    def action(directory):
        q, g = protocol_rows(c, split)
        output = directory / "export"
        metrics = inference.export_arrays(output, q, g, values, system, c["manifest"]["threshold"])
        if any(metrics[k] != report[k] for k in ("ranking", "candidates")):
            raise ValueError("Export/replay differs from evaluated result")
        _, original = baseline_features(c, split, q, g)
        if base.sha256(output / "candidates.csv") != base.sha256(original / "candidates.csv"):
            raise ValueError("Candidate CSV bytes differ from v25")
        if split == "validation" and system == "V25_control":
            before = Path(c["manifest"]["v25_directory"]) / "tasks/validation_r1w50_k20_q3_l50/export"
            base.verify_files({str(output / n): base.sha256(before / n) for n in ("submission.csv", "candidates.csv", "embeddings.npy")})
        return {"status": "passed", "system": system, "split": split, "candidate_bytes_unchanged": True,
                "npy_replay": "exact decisions", "dimension": values.shape[1]}
    return task(c, f"export_{split}_{system}", action)


def paired_changes(control, winner):
    left, right = control["per_query"], winner["per_query"]
    if set(left) != set(right): raise ValueError("Unpaired query reports")
    changes = []
    for qid, before in left.items():
        after = right[qid]
        if before["ranking"]["ap"] is not None:
            changes.append({"query_id": qid, "vehicle_id": before["vehicle_id"],
                "delta_ap": after["ranking"]["ap"]-before["ranking"]["ap"],
                "before_top1": before["ranking"]["top1"], "after_top1": after["ranking"]["top1"]})
    return {"improved": sum(x["delta_ap"] > 0 for x in changes), "worsened": sum(x["delta_ap"] < 0 for x in changes),
            "unchanged": sum(x["delta_ap"] == 0 for x in changes),
            "top10_changed": sum(left[q]["ranking"]["top10"] != right[q]["ranking"]["top10"] for q in left),
            "top1_fixed": sum(not left[q]["ranking"]["correct"] and right[q]["ranking"]["correct"] for q in left),
            "top1_broken": sum(left[q]["ranking"]["correct"] and not right[q]["ranking"]["correct"] for q in left),
            "per_query": changes}


def write_report(c, result):
    selected = result["selection"]["selected"]
    lines = ["# v37 — v25 + TransReID без нового обучения", "", f"Выбор на calibration: **{selected}**.",
        "MVP_fusion_v25 не изменён. Порог, кандидат, confidence и отказы остаются от v25.", "",
        "| Split | Система | mAP@10 | Δ к v25, п.п. | Rank-1 | F1 | TNR |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for split, reports in result["evaluations"].items():
        control = reports["V25_control"]["ranking"]["mAP@10"]
        for name, r in reports.items():
            lines.append(f"| {split} | {name} | {r['ranking']['mAP@10']:.6f} | "
                f"{100*(r['ranking']['mAP@10']-control):+.4f} | {r['ranking']['Rank-1']:.6f} | "
                f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |")
    d = result["paired_validation"]
    lines += ["", f"Validation AP: улучшилось {d['improved']}, ухудшилось {d['worsened']}, без изменения {d['unchanged']}.",
        f"Top-10 изменились у {d['top10_changed']} запросов; top-1 исправлен у {d['top1_fixed']}, испорчен у {d['top1_broken']}.",
        "", "## Как читать результат", "",
        "Подбиралась только доля новой модели: 5/10/15/20%, отдельно T12 и T22, плюс контроль с нулевой долей.",
        "Косинус до графа: (1-w) × v25 + w × TransReID; внутри v25 равные доли MVP и R1 equal3.",
        "Реранкер legacy 20/3/0.50 зафиксирован, не подбирался. Точные равенства решаются в пользу контроля, меньшей доли, затем T12.",
        "До чтения validation-признаков/оценок записан frozen_selection.json; validation считает только контроль и выбранную систему.",
        "Исходная validation уже использовалась раньше: это development-сравнение, не независимый тест и не обещание hidden-test результата.",
        "T12/T22 — конкретные сохранённые checkpoints на 740 train-ID; новый full-train refit не выполнялся. v25 обучался на своём полном train.",
        "Всего 0 optimizer updates, 0 BN updates; исходные bbox, разбиения и данные не менялись, NiVe не используется.",
        "CSV каждого экспорта содержит top-10 даже при отказе; candidates.csv совпадает с v25 побайтно.",
        "embeddings.npy содержит реальные unit-блоки: MVP512 + R1_1536 [+ T12_384 или T22_1920]. Это не единый ranking cosine-vector.",
        "NPY replay использует явные slices/веса. Экспорты лежат в tasks/export_<split>_<system>/export/.",
        "Новые признаки — PyTorch float32 на явно выбранном устройстве; v25 — проверенный прежний CPU ONNX-кэш.",
        "Кэш не выдаётся за новый замер скорости. Это исследовательский runner, не готовый новый ONNX-профиль продукта.",
        "Независимость изображений: свежие batch 1/8/16/32, перестановка/одиночное изображение; буферы модели неизменны.",
        "CPU/GPU скорость новой полной системы, Linux и экспорт в продукт проверяются отдельно только при полезном качестве.",
        "Один новый encoder добавляется к четырём encoder v25. Ограничения junk/top-10 и остаточного сигнала номеров не устранены.",
        "Ни результаты, ни победа на validation не переключают MVP автоматически. Новые пороги/веса по validation не подбираются."]
    text = "\n".join(lines) + "\n"
    path = c["output"] / "REPORT.md"
    if path.exists() and path.read_text(encoding="utf-8") != text: raise ValueError("Completed report changed")
    if not path.exists():
        pending = path.with_suffix(".md.tmp")
        pending.write_text(text, encoding="utf-8")
        pending.replace(path)


def run(c, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True is required")
    with base.old.run_lock(c["output"]):
        check_inputs(c)
        probe = runtime_probe(c)
        q, g = protocol_rows(c, "calibration")
        original, _ = baseline_features(c, "calibration", q, g)
        components = {member: features(c, q+g, "calibration", member) for member in inference.DIMENSIONS}
        calibration = {}
        for i, (name, spec) in enumerate(inference.SYSTEMS.items(), 1):
            print(f"CALIBRATION {i}/{len(inference.SYSTEMS)}: {name}", flush=True)
            values = original if spec["member"] is None else np.concatenate([original, components[spec["member"]]], axis=1)
            calibration[name] = task(c, f"calibration_{name}",
                lambda d, n=name, v=values: evaluate(c, "calibration", n, v, d))
        selection = freeze_selection(c, calibration)
        print(f"FROZEN: {selection['selected']}; only now opening original validation", flush=True)
        cases = list(dict.fromkeys(["V25_control", selection["selected"]]))
        for name in cases:
            member = inference.SYSTEMS[name]["member"]
            values = original if member is None else np.concatenate([original, components[member]], axis=1)
            export_result(c, "calibration", name, values, calibration[name])
        require_selection(c)
        q, g = protocol_rows(c, "validation")
        original, _ = baseline_features(c, "validation", q, g)
        member = selection["spec"]["member"]
        extra = features(c, q+g, "validation", member) if member else None
        validation = {}
        for name in cases:
            values = original if name == "V25_control" else np.concatenate([original, extra], axis=1)
            validation[name] = task(c, f"validation_{name}",
                lambda d, n=name, v=values: evaluate(c, "validation", n, v, d))
            export_result(c, "validation", name, values, validation[name])
        check_inputs(c)
        result = {"status": "complete", "signature": c["signature"], "selection": selection,
            "runtime_probe": probe, "evaluations": {"calibration": calibration, "validation": validation},
            "paired_validation": paired_changes(validation["V25_control"], validation[selection["selected"]]),
            "optimizer_updates": 0, "bn_updates": 0, "threshold_fit": False, "candidate_unchanged": True,
            "promoted": False, "protected_unchanged": True,
            "scope": "development quality comparison of saved concrete weights; no new training or automatic deployment"}
        write_report(c, result)
        base.old.freeze_json(c["output"] / "results.json", result)
        base.old.freeze_json(c["output"] / "complete.json", {"signature": c["signature"], "files": {
            str(p.relative_to(c["output"])): base.sha256(p) for p in c["output"].rglob("*")
            if p.is_file() and p.name not in {".lock", ".run.lock"} and p != c["output"] / "complete.json"
            and not any(part.startswith(".pending_") for part in p.relative_to(c["output"]).parts)}})
    return result
