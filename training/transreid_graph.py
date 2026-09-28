"""v39: cache-only graph comparison, calibration selection, one frozen validation winner."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from pathlib import Path
import platform
import re

import numpy as np

from training import transreid_system as previous, transreid_graph_inference as inference

base = previous.base
VARIANT = base.ROOT / "OSNet-AIN-x1.0/variant_39_transreid_graph"
task = previous.task
protocol_rows = previous.protocol_rows
SYSTEMS = inference.SYSTEMS
README_PATH = base.ROOT.parent / "Car-classification-MSK-main/models/README.md"
README_OLD = "2b34a64165009d38d963a82f847227b51905f7011a9d46a221ce88cd08b273c9"
README_NEW = "b58775744653ead7f5f5f53faa5d56aeb22cf88ce559fbfc2754b94150544500"
RELEASE_PATH = base.ROOT.parent / "Car-classification-MSK-main/release_decision.json"
RELEASE_OLD = "c95695b3eb3613c1d436e4622647e32f925c7cad814b723105f2a13a632767be"
RELEASE_NEW = "f306cef11ad5c7951dde3b1a59e9a55e2e03ff7bd16f996ad59c95a067bdd5c9"


def reviewed_protection(inherited):
    """Two exact reviewed metadata transitions in a NEW manifest; no generic exclusions."""
    transitions = [
        {"path": str(README_PATH), "old_sha256": README_OLD, "new_sha256": README_NEW,
         "reason": "Documentation of existing MVP_fusion_v25; no weight/code/data change"},
        {"path": str(RELEASE_PATH), "old_sha256": RELEASE_OLD, "new_sha256": RELEASE_NEW,
         "reason": "Only removes obsolete extractor open-check and links organizer clarification; active model unchanged"},
    ]
    protected = dict(inherited)
    for item in transitions:
        if inherited.get(item["path"]) != item["old_sha256"]:
            raise ValueError(f"Expected the original v38 protection: {item['path']}")
        base.verify_files({item["path"]: item["new_sha256"]})
        protected[item["path"]] = item["new_sha256"]
    base.verify_files(protected)
    return protected, {"files": transitions, "old_manifests_modified": False}


def check_inputs(c):
    m = c["manifest"]
    if (base.digest(m) != c["signature"] or m["systems"] != SYSTEMS
            or base.digest(c["rows"]) != m["rows_sha256"]):
        raise ValueError("Frozen v39 context changed")
    print(f"CHECK: {len(m['protected'])} protected files; old manifests remain unchanged", flush=True)
    base.verify_files(m["protected"])
    base.verify_files(m["source_sha256"])
    active = base.ROOT.parent / "Car-classification-MSK-main/release_decision.json"
    if base.old.load_json(active)["active_profile"] != "MVP_fusion_v25":
        raise ValueError("Active MVP changed; review the control")


def prepare(run_name="graph_v1"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple RUN_NAME")
    config = VARIANT / "configs/graph_v1.json"
    plan = base.old.load_json(config)
    if (plan["new_encoder_weight"] != .1 or plan["policies"] != ["legacy", "less_graph", "raw"]
            or plan["selection_split"] != "calibration" or plan["selection_metric"] != "mAP@10"
            or any(plan[k] for k in ("optimizer_updates", "encoder_forwards", "threshold_fit", "promoted"))
            or plan["wall_time_limit"] is not None):
        raise ValueError("Only the fixed cache-only graph comparison is allowed")
    source = base.ROOT / plan["source_run"]
    names = ("manifest", "results", "complete", "frozen_candidate")
    pins = {str(source / f"{n}.json"): plan[f"source_{n}_sha256"] for n in names}
    base.verify_files(pins)
    sm, result, complete, candidate = (base.old.load_json(source / f"{n}.json") for n in names)
    signature = plan["source_signature"]
    if (base.digest(sm) != signature or any(x["signature"] != signature for x in (result, complete, candidate))
            or result["status"] != "complete" or result["promoted"] or not result["candidate_unchanged"]
            or len(sm["train_ids"]) != 925 or candidate["step"] != 1800
            or candidate["checkpoint"]["sha256"] != plan["checkpoint_sha256"]
            or candidate["new_encoder_weight"] != .1 or candidate["threshold"] != sm["threshold"]):
        raise ValueError("Expected the unchanged completed full-train T12 run")
    print("PREFLIGHT: verify v38 caches, weights, original data and two reviewed metadata transitions", flush=True)
    inherited = {**sm["protected"], **pins, **{str(source / p): h for p, h in complete["files"].items()}}
    protected, transition = reviewed_protection(inherited)
    base.verify_files(sm["source_sha256"])
    rows = base.read_rows(base.DATASET / "train.csv")
    splits = base.old.load_json(base.ARTIFACTS / "splits.json")
    previous.previous.validate_splits(rows, sm["protocols"], splits["identities"], sm["train_ids"])
    sources = {**sm["source_sha256"], **{str(p): base.sha256(p) for p in (
        Path(__file__).resolve(), Path(inference.__file__).resolve(), config)}}
    manifest = {"version": 39, "plan": plan, "systems": SYSTEMS, "protocols": sm["protocols"],
        "rows_sha256": base.digest(rows), "threshold": sm["threshold"], "checkpoint": candidate["checkpoint"],
        "source_directory": str(source), "source_signature": signature, "source_candidate": candidate,
        "v25_directory": sm["v25_directory"], "v24_directory": sm["v24_directory"],
        "protected": protected, "source_sha256": sources, "reviewed_document_transition": transition,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "device": "cpu", "features": "reused v38 arrays, not a new encoder measurement"},
        "selection": "maximum calibration mAP@10; exact ties use SYSTEMS order, control first",
        "scope": "previously observed development splits, not an independent test", "promoted": False}
    c = {"output": VARIANT / "runs" / run_name, "manifest": manifest, "signature": base.digest(manifest), "rows": rows}
    check_inputs(c)
    with base.old.run_lock(c["output"]):
        base.old.freeze_json(c["output"] / "manifest.json", manifest)
    print("PREFLIGHT OK: 4 calibration cases; CPU caches only; validation closed until selection", flush=True)
    return c


def selection(c):
    reports, hashes = {}, {}
    for name in SYSTEMS:
        directory = c["output"] / "tasks" / f"calibration_{name}"
        if not (directory / "complete.json").is_file():
            raise ValueError("Selection requires all four completed calibration stages")
        saved = base.old.load_json(directory / "complete.json")
        if saved["signature"] != c["signature"] or "result.json" not in saved["files"]:
            raise ValueError("Calibration stage belongs to another experiment")
        base.verify_files({str(directory / p): h for p, h in saved["files"].items()})
        r = base.old.load_json(directory / "result.json")
        if (r["split"] != "calibration" or r["system"] != name or r["spec"] != SYSTEMS[name]
                or r["threshold"] != c["manifest"]["threshold"] or not r["candidate_unchanged"]
                or r["protocol_sha256"] != base.digest(c["manifest"]["protocols"]["calibration"])
                or not np.isfinite(r["ranking"]["mAP@10"])):
            raise ValueError("Invalid calibration selection input")
        reports[name] = r
        hashes[name] = base.sha256(directory / "result.json")
    winner = max(SYSTEMS, key=lambda n: reports[n]["ranking"]["mAP@10"])
    return {"signature": c["signature"], "split": "calibration", "metric": "mAP@10", "selected": winner,
        "spec": SYSTEMS[winner], "tie_order": list(SYSTEMS), "threshold": c["manifest"]["threshold"],
        "checkpoint": c["manifest"]["checkpoint"], "calibration_sha256": hashes,
        "scores": {n: r["ranking"]["mAP@10"] for n, r in reports.items()},
        "evaluations": list(dict.fromkeys(["V25_control", winner])), "promoted": False}


def freeze_selection(c):
    chosen = selection(c)
    base.old.freeze_json(c["output"] / "frozen_selection.json", chosen)
    return chosen


def require_selection(c):
    path = c["output"] / "frozen_selection.json"
    if not path.is_file():
        raise ValueError("Validation is closed until calibration selection is frozen")
    saved = base.old.load_json(path)
    if saved != selection(c):
        raise ValueError("Frozen calibration selection changed")
    return saved


def authorize(c, split, system):
    if system not in SYSTEMS or split not in ("calibration", "validation"):
        raise ValueError("Unknown split/system")
    if split == "validation" and system not in require_selection(c)["evaluations"]:
        raise ValueError("Validation accepts only the frozen winner and v25 control")


def source_stage(c, split, system):
    return Path(c["manifest"]["source_directory"]) / "tasks" / f"{split}_{SYSTEMS[system]['source']}"


def features(c, split, system):
    authorize(c, split, system)  # Before even opening a validation array.
    q, g = protocol_rows(c, split)
    directory = source_stage(c, split, system) / "export"
    paths = [directory / n for n in ("embeddings.npy", "embedding_order.json")]
    base.verify_files({str(p): c["manifest"]["protected"][str(p)] for p in paths})
    order = base.old.load_json(paths[1])
    values = np.load(paths[0], allow_pickle=False)
    dimension = 2048 if system == "V25_control" else 2432
    if (order["ids"] != [r["image_id"] for r in q+g] or values.shape != (len(q)+len(g), dimension)
            or order["query_count"] != len(q) or order["gallery_count"] != len(g)
            or order["dimension"] != dimension or order["sha256"] != base.sha256(paths[0])):
        raise ValueError("Saved feature cache order/layout changed")
    inference.previous.ranking_features(values, SYSTEMS[system]["mixture"])
    return values


def evaluate(c, split, system, directory):
    authorize(c, split, system)
    q, g = protocol_rows(c, split)
    values = features(c, split, system)
    threshold = c["manifest"]["threshold"]
    ranked = inference.rank(values, len(q), system, progress=True)
    report = {"split": split, "system": system, "spec": SYSTEMS[system], "threshold": threshold,
        "protocol_sha256": base.digest(c["manifest"]["protocols"][split]),
        **base.policy.evaluate(q, g, ranked, threshold, "raw_top1"), **base.policy.query_diagnostics(q, g, ranked)}
    ordered, _ = base.policy.predictions(q, g, ranked, threshold, "raw_top1")
    for qid, top10 in ordered.items(): report["per_query"][qid]["ranking"]["top10"] = top10
    expected = base.old.load_json(source_stage(c, split, "V25_control") / "result.json")
    if report["candidates"] != expected["candidates"]:
        raise ValueError("Frozen v25 candidate metrics changed")
    before = source_stage(c, split, system)
    if system in ("V25_control", "Full_T12_l50"):
        reference = base.old.load_json(before / "result.json")
        if report["ranking"] != reference["ranking"] or report["per_query"] != reference["per_query"]:
            raise ValueError("Original v38 graph metrics/decisions no longer reproduce")
    if system == "Full_T12_raw":
        if report["ranking"] != base.old.load_json(before / "result.json")["raw_ranking"]:
            raise ValueError("Original raw mixture metrics no longer reproduce")
    output = directory / "export"
    metrics = inference.export_arrays(output, q, g, values, system, threshold)
    if any(metrics[k] != report[k] for k in ("ranking", "candidates")):
        raise ValueError("CSV/NPY replay differs from reported decisions")
    base.verify_files({str(output / "candidates.csv"): base.sha256(
        source_stage(c, split, "V25_control") / "export/candidates.csv")})
    if system in ("V25_control", "Full_T12_l50"):
        base.verify_files({str(output / n): base.sha256(before / "export" / n)
                           for n in ("submission.csv", "embeddings.npy")})
    base.write_json(output / "model_provenance.json", {"signature": c["signature"], "system": system,
        "spec": SYSTEMS[system], "source_signature": c["manifest"]["source_signature"],
        "checkpoint": c["manifest"]["checkpoint"] if system != "V25_control" else None,
        "v25_profile": str(Path(c["manifest"]["v24_directory"]) / "dual_role_profile.json"),
        "encoder_forwards": 0, "promoted": False})
    report["candidate_unchanged"] = True
    print(f"EVAL {split}/{system}: mAP@10={report['ranking']['mAP@10']:.6f}; candidates unchanged", flush=True)
    return report


def write_report(c, result):
    lines = ["# v39 — граф для сохранённого Full T12", "",
        f"Победитель calibration: **{result['selection']['selected']}**. MVP_fusion_v25 не изменён.", "",
        "| Split | Система | mAP@10 | Δ к v25, п.п. | Rank-1 | F1 | TNR |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for split, reports in result["evaluations"].items():
        baseline = reports["V25_control"]["ranking"]["mAP@10"]
        for name, r in reports.items():
            lines.append(f"| {split} | {name} | {r['ranking']['mAP@10']:.6f} | "
                f"{100*(r['ranking']['mAP@10']-baseline):+.4f} | {r['ranking']['Rank-1']:.6f} | "
                f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |")
    d = result["paired_validation"]
    lines += ["", f"Validation AP: улучшилось {d['improved']}, ухудшилось {d['worsened']}, без изменения {d['unchanged']}.",
        f"Top-1 исправлен у {d['top1_fixed']}, испорчен у {d['top1_broken']}; top-10 изменился у {d['top10_changed']} запросов.",
        "", "## Что было зафиксировано", "",
        "Один сохранённый Full T12 (925 train-ID, step 1800), доля 10%; 90% — прежняя смесь v25.",
        "Только четыре calibration-варианта: контроль v25, Full T12 λ=.50, λ=.75, raw без графа. k1=20/k2=3.",
        "Выбор по calibration mAP@10; точные равенства — в указанном порядке, контроль первый.",
        "Validation рассчитана только для v25 и одного уже зафиксированного победителя; при победе контроля — один расчёт.",
        "Кандидат/confidence/отказ — прежний R1 raw_top1; порог не меняется. candidates.csv совпадает с v25 побайтно.",
        "Результаты старых λ=.50 и raw воспроизведены точно; NPY replay проверяет top-10, кандидатов, confidence и отказы.",
        "Все эмбеддинги взяты из v38: 2048D для контроля, 2432D для Full T12. Новых encoder-forward, обучения, BN-update и threshold-fit нет.",
        "Экспорты и полный порядок векторов: tasks/<split>_<system>/export/. Три файла, десять разных ID на query, включая отказы.",
        "Статическая gallery, независимые query; ни имена файлов, ни ID/camera/time не передаются ранжировщику.",
        "Это очередной development-эксперимент: calibration/validation уже наблюдались, выбор гипотезы учитывает прошлые результаты.",
        "Ни независимый тест, ни новая оценка скорости модели, ни основание для автоматического promotion здесь не заявляются.",
        "Проверка неизменности всех унаследованных весов/данных/кода сохранена. Два точных metadata-перехода учтены только в новом manifest.",
        f"README: {README_OLD} → {README_NEW} (описание уже действующего v25).",
        f"release_decision: {RELEASE_OLD} → {RELEASE_NEW} (отмена extractor-check и ссылка на уточнение; модель та же).",
        "Старые manifests/guards/runs не изменены. Другой или последующий metadata-переход не разрешён автоматически.",
        "Изменения приложения/GPU в main вне эксперимента не включаются в его runtime. Активный MVP не переключается.",
        "Новых данных, OCR, изменений bbox и номерной зоны нет. Ограничения junk/top-10 и остаточного номерного сигнала остаются открытыми.",
        "Этапы публикуются атомарно; повторный Run All проверяет и переиспользует завершённые результаты. Ограничения времени нет."]
    text = "\n".join(lines)+"\n"
    path = c["output"] / "REPORT.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Completed report changed")
    if not path.exists():
        pending = path.with_suffix(".md.tmp")
        pending.write_text(text, encoding="utf-8"); pending.replace(path)


def run(c, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True is required for the development comparison")
    with base.old.run_lock(c["output"]):
        check_inputs(c)
        completion = c["output"] / "complete.json"
        if completion.exists():
            saved = base.old.load_json(completion)
            if saved["signature"] != c["signature"]:
                raise ValueError("Completed graph experiment fingerprint changed")
            base.verify_files({str(c["output"] / p): h for p, h in saved["files"].items()})
            require_selection(c)
            print("Verified completed comparison; no new calculations or measurements", flush=True)
            return base.old.load_json(c["output"] / "results.json")
        reports = {"calibration": {}, "validation": {}}
        for i, name in enumerate(SYSTEMS, 1):
            print(f"CALIBRATION {i}/4: {name}", flush=True)
            reports["calibration"][name] = task(c, f"calibration_{name}",
                lambda d, n=name: evaluate(c, "calibration", n, d))
        chosen = freeze_selection(c)
        print(f"FROZEN: {chosen['selected']}; validation now open for {chosen['evaluations']}", flush=True)
        for name in chosen["evaluations"]:
            reports["validation"][name] = task(c, f"validation_{name}",
                lambda d, n=name: evaluate(c, "validation", n, d))
        check_inputs(c)
        result = {"status": "complete", "signature": c["signature"], "selection": chosen, "evaluations": reports,
            "paired_validation": previous.paired_changes(reports["validation"]["V25_control"],
                                                          reports["validation"][chosen["selected"]]),
            "optimizer_updates": 0, "encoder_forwards": 0, "inference_bn_updates": 0, "threshold_fit": False,
            "candidate_unchanged": True, "promoted": False, "protected_unchanged_since_v39_freeze": True,
            "reviewed_document_transition": c["manifest"]["reviewed_document_transition"],
            "scope": c["manifest"]["scope"]}
        write_report(c, result)
        base.old.freeze_json(c["output"] / "results.json", result)
        base.old.freeze_json(completion, {"signature": c["signature"], "files": {
            str(p.relative_to(c["output"])): base.sha256(p) for p in c["output"].rglob("*")
            if p.is_file() and p != completion and p.name not in {".lock", ".run.lock"}
            and not p.name.endswith(".tmp") and not any(part.startswith(".pending_") for part in p.parts)}})
    return result
