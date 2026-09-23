"""Stage 3: clean-inner pair-head selection, fixed full-train refit, frozen outer evaluation."""
import argparse
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch

from backend.core import ARTIFACTS, DATASET, MODEL, ROOT, STOCK_MODEL, Encoder, normalize, preprocess, read_rows, sha256
from backend.evaluate import SEED, make_protocol
from backend.scoring import metrics
from training import rerank_score_control as control
from training.audit import digest, load_crop, validate_annotations
from training.hpo import ReIDExperimentModel
from training.mask_calibration import _load, _save
from training.mask_finetune import load_json
from training.mask_model_comparison import AuditEncoder, check_sample
from training.mask_reid_ablation import paired_deltas
from training.pair_reranker import (BUDGET, HEADS, SCALARS, PairFeatures, PairSet, encode_cached,
                                    export_head, fit_head)
from training.stage6 import audit_partitions

EXPERIMENT = ROOT / "OSNet-AIN-x1.0/variant_12_pair_reranker"
INNER_SOURCE = ROOT / "CLIP-ReID-ViT-B-16/variant_01_vehicle_transfer"
INNER_WEIGHTS = INNER_SOURCE / "weights/osnet_control/best_map.pt"
REFERENCE = ROOT / "Mask-Detectors/reference_v02/results"
BETAS = (0., .1, .25, .5, 1.)
CONFIDENCES = ("max_cosine", "max_pair", "selected_pair")


def verify_control(checkpoint, protocol, summary, rows, hashes, split):
    """Reject using the MVP as an apparently held-out inner encoder."""
    expected = {**protocol, "initializer_sha256": sha256(STOCK_MODEL),
                "control_recipe": "stock OSNet + avg/BNNeck/SupCon; inner-only; 4000 steps max"}
    signature, config = checkpoint["signature"], checkpoint["config"]
    if (protocol["outer"] != split["identities"] or protocol["seed"] != SEED
            or protocol["data_sha256"] != digest({"rows": rows, "frames": hashes})
            or protocol["evaluator_sha256"] != sha256(ROOT / "evaluate.py")
            or signature["protocol"] != digest(expected) or summary["protocol_sha256"] != digest(expected)
            or signature["train_ids"] != protocol["inner"]["train"]
            or signature["validation_ids"] != protocol["inner"]["validation"] or signature["stop_steps"] is not None
            or config != summary["config"] or config != signature["config"]
            or checkpoint["step"] != summary["selected_steps"] or checkpoint["budget"] != summary["budget"]
            or config["pooling"] != "avg" or config["resize_mode"] != "square"
            or not config["use_bnneck"] or config["use_mixstyle"]
            or checkpoint["model"]["classifier.weight"].shape[0] != len(protocol["inner"]["train"])):
        raise ValueError("Clean control provenance or encoder-training split mismatch")
    audit_partitions(rows, hashes, protocol["outer"])
    development = [r for r in rows if r["vehicle_id"] in set(protocol["outer"]["train"])]
    audit_partitions(development, hashes, protocol["inner"])


def prepare():
    protocols, embeddings, baseline, old_signature, hashes, difference = control.prepare()
    rows, split = read_rows(DATASET / "train.csv"), load_json(ARTIFACTS / "splits.json")
    paths = {"inner_weights": INNER_WEIGHTS, "inner_protocol": INNER_SOURCE / "results/protocol.json",
        "inner_summary": INNER_SOURCE / "results/osnet_control/training_summary.json",
        "stage1": control.EXPERIMENT / "results/run_01/frozen_selection.json",
        "manual": REFERENCE / "masks.json", "manual_plan": REFERENCE / "mask_plan.json",
        "manual_provenance": REFERENCE / "comparison.json", "stock_initializer": STOCK_MODEL}
    protocol, summary = load_json(paths["inner_protocol"]), load_json(paths["inner_summary"])
    checkpoint = torch.load(INNER_WEIGHTS, map_location="cpu", weights_only=True)
    verify_control(checkpoint, protocol, summary, rows, hashes, split)
    stage1 = _load(paths["stage1"], old_signature)
    if stage1["ranking"]["pool"] != 50:
        raise ValueError("Expected a frozen top50 neighborhood control")
    for name, ids in (("inner_train", protocol["inner"]["train"]), ("inner_validation", protocol["inner"]["validation"]),
                      ("development", protocol["outer"]["train"])):
        protocols[name] = make_protocol(rows, ids, SEED)
    plan = load_json(paths["manual_plan"])
    validate_annotations(plan, load_json(paths["manual"]))
    mq, mg = check_sample(rows, plan, split)
    audit = _load(control.MASK_AUDIT / "comparison.json")
    if (sha256(paths["manual"]) != load_json(paths["manual_provenance"])["source_sha256"]
            or audit["signature"]["source_sha256"]["manual"] != sha256(paths["manual"])
            or audit["signature"]["source_sha256"]["mvp_weights"] != sha256(MODEL)
            or audit["signature"]["query"] != mq or audit["signature"]["gallery"] != mg
            or audit["output_sha256"]["embeddings.npz"] != sha256(control.MASK_AUDIT / "embeddings.npz")):
        raise ValueError("Corrected manual reference or cached features changed")
    protected = {**old_signature["protected_sha256"], **{str(p): sha256(p) for p in paths.values()}}
    for source in ("training/pair_reranker.py", "training/pair_reranker_experiment.py", "training/osnet.py",
                   "training/hpo.py", "training/pipeline.py", "training/preprocessing.py", "training/stage6.py",
                   "training/audit.py", "training/mask_model_comparison.py", "training/mask_calibration.py"):
        protected[str(ROOT / source)] = sha256(ROOT / source)
    signature = {"version": 1, "protected_sha256": protected, "data_sha256": protocol["data_sha256"],
        "identities": {"outer": protocol["outer"], "inner": protocol["inner"]},
        "protocols": {name: {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g]}
                      for name, (q, g) in protocols.items()}, "heads": list(HEADS), "budget": BUDGET,
        "scalar_features": list(SCALARS), "betas": list(BETAS), "confidence_modes": list(CONFIDENCES),
        "baseline": stage1["ranking"], "main_masks": False, "flip_tta": False,
        "preprocessing": old_signature["preprocess"], "device": "CPU, torch/ONNX 2 threads",
        "runtime": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "onnxruntime", "Pillow")}}
    return protocols, embeddings, baseline, signature, hashes, summary, (mq, mg), difference


def export_control(output, signature, rows):
    output.mkdir(parents=True, exist_ok=True)
    path, report_path = output / "inner_control.onnx", output / "export.json"
    if report_path.exists():
        report = _load(report_path, signature)
        return AuditEncoder(path, report["sha256"]), report
    checkpoint = torch.load(INNER_WEIGHTS, map_location="cpu", weights_only=True)
    model = ReIDExperimentModel(len(checkpoint["signature"]["train_ids"]), True).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    inference = model.inference_module().eval()
    crops = [load_crop(r) for r in rows[:2]]
    samples = np.stack([preprocess(c, (0, 0, *c.size)) for c in crops])
    temporary = output / "inner_control.partial.onnx"
    with torch.no_grad():
        torch.onnx.export(inference, torch.from_numpy(samples[:1]), str(temporary), dynamo=False, opset_version=17,
            input_names=["images"], output_names=["output"], dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}})
    checksum = sha256(temporary)
    encoder = AuditEncoder(temporary, checksum)
    errors = []
    for size in (1, 2):
        with torch.no_grad():
            expected = normalize(inference(torch.from_numpy(samples[:size])).numpy())
        actual = encoder.encode_batch(samples[:size])
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        errors.append(float(np.abs(actual-expected).max()))
    temporary.replace(path)
    report = {"signature": signature, "sha256": checksum, "bytes": path.stat().st_size,
        "max_absolute_error": max(errors), "batches_checked": [1, 2], "training_only_encoder": True}
    _save(report_path, report)
    return encoder, report


def select_inner(train, validation, hashes, output, signature):
    path = output / "inner_selection.json"
    if path.exists():
        report = _load(path, signature)
        for trial in report["runs"]:
            if sha256(output / "weights/inner" / trial["config"]["name"] / "selected.pt") != trial["weights_sha256"]:
                raise ValueError("Selected inner head changed")
        return report
    targets, weights, valid = train.labels(hashes)
    runs = []
    for config in HEADS:
        report = fit_head(train, targets, weights, valid, config, output / "weights/inner" / config["name"], signature,
                          validation=validation)
        runs.append({"config": config, "selected_epoch": report["selected_epoch"], "best_inner": report["best_inner"],
                     "weights_sha256": report["weights_sha256"]})
    winner = max(runs, key=lambda r: tuple(r["best_inner"]))
    report = {"signature": signature, "runs": runs, "winner": winner,
        "training_pairs": int(valid.sum()), "training_positive_pairs": int(((targets == 1) & valid).sum()),
        "query_count": len(train.query), "queries_with_positive_in_top50": int(((targets == 1) & valid).any(1).sum()),
        "selection": "inner official top50-head mAP@10, Rank-1, earlier epoch, grid order",
        "outer_calibration_or_validation_used": False}
    _save(path, report)
    return report


def score_pair_set(pairs, probabilities, beta):
    scores = (1-beta)*pairs.base + beta*probabilities
    ranked = pairs.ranked(scores)
    by_id = {r["image_id"]: j for j, r in enumerate(pairs.gallery)}
    winners = [int(np.flatnonzero(pairs.indices[i] == by_id[ranked.predictions[q["image_id"]][0]])[0])
               for i, q in enumerate(pairs.query)]
    return ranked, {"max_cosine": pairs.confidence, "max_pair": probabilities.max(1).astype(float),
        "selected_pair": probabilities[np.arange(len(winners)), winners].astype(float)}


def tune_calibration(pairs, encoder, baseline_threshold):
    probabilities, trials = encoder.probabilities(pairs.features), []
    for beta in BETAS:
        ranked, _ = score_pair_set(pairs, probabilities, beta)
        score = metrics(ranked, baseline_threshold)
        trials.append({"beta": beta, "metrics_at_old_threshold": score})
        print(f"Calibration pair weight={beta:g}: mAP={score['mAP_at_10']:.6f}", flush=True)
    beta = max(trials, key=lambda r: (r["metrics_at_old_threshold"]["mAP_at_10"],
        r["metrics_at_old_threshold"]["Rank_1"], -r["beta"]))["beta"]
    choices, curves = {}, {}
    for name, weight in (("combined", beta), ("pair_only", 1.)):
        ranked, values = score_pair_set(pairs, probabilities, weight)
        curves[name] = [{"mode": mode, **control.choose_threshold(ranked, values[mode])} for mode in CONFIDENCES]
        choice = max(curves[name], key=lambda r: (r["metrics"]["candidate_score"], r["metrics"]["candidate_F1"]))
        choices[name] = {"beta": weight, "mode": choice["mode"], "threshold": choice["threshold"]}
    return {"beta": beta, "choices": choices, "ranking_trials": trials, "refusal_trials": curves,
            "validation_used": False, "head_weights_updated": False}


def evaluated(ranked, confidence, threshold):
    confidence = np.asarray(confidence, dtype=float)
    return {"metrics": metrics(ranked, threshold, confidence),
            "per_query": control.query_details(ranked, confidence, threshold)}


def evaluate_frozen(query, gallery, embeddings, encoder, frozen):
    pairs = PairSet(query, gallery, embeddings, frozen["baseline"])
    probabilities = encoder.probabilities(pairs.features)
    results = {}
    for name, config in (("mvp", control.ACTIVE), ("osnet_neighbors", frozen["baseline"])):
        ranked, _ = control.ProtocolScores(query, gallery, embeddings).evaluate(config)
        results[name] = evaluated(ranked, ranked.confidence, frozen["baseline_threshold"])
    ranked, confidence = score_pair_set(pairs, probabilities, frozen["beta"])
    results["ranking_only"] = evaluated(ranked, confidence["max_cosine"], frozen["baseline_threshold"])
    for name, choice in frozen["choices"].items():
        ranked, confidence = score_pair_set(pairs, probabilities, choice["beta"])
        results[name] = evaluated(ranked, confidence[choice["mode"]], choice["threshold"])
    for name, value in results.items():
        value["paired_vs_mvp"] = paired_deltas(results["mvp"]["per_query"], value["per_query"])
        value["paired_vs_neighbors"] = paired_deltas(results["osnet_neighbors"]["per_query"], value["per_query"])
    return results


def mask_check(sample, encoder, frozen):
    q, g = sample
    scores = {}
    with np.load(control.MASK_AUDIT / "embeddings.npz", allow_pickle=False) as cache:
        ids = cache["ids"].tolist()
        if ids != [r["image_id"] for r in q + g]:
            raise ValueError("Manual cache order mismatch")
        for condition in ("original", "manual"):
            scores[condition] = evaluate_frozen(q, g, dict(zip(ids, cache[f"mvp__{condition}"])), encoder, frozen)
    return {"scores": scores, "queries": len(q), "gallery": len(g), "threshold_retuned": False,
            "selection_used": False, "warning": "Small reused corrected masks-7 audit; not proof of plate independence"}


def benchmark(query, gallery, embeddings, encoder, frozen, repeats=3):
    started = time.perf_counter()
    engine = PairFeatures(np.stack([embeddings[r["image_id"]] for r in gallery]), frozen["baseline"])
    build = time.perf_counter()-started
    times = []
    for _ in range(repeats):
        for q in query:
            started = time.perf_counter()
            indices, features, base, cosine = engine.one(embeddings[q["image_id"]])
            probability = encoder.probabilities(features)
            scores = (1-frozen["beta"])*base + frozen["beta"]*probability
            order = np.lexsort((indices, -scores))[:10]
            choice = frozen["choices"]["combined"]
            confidence = {"max_cosine": cosine, "max_pair": float(probability.max()),
                          "selected_pair": float(probability[order[0]])}[choice["mode"]]
            bool(confidence >= choice["threshold"])
            times.append(1000*(time.perf_counter()-started))
    return {"graph_seconds": build, "median_query_ms": float(np.median(times)), "p95_query_ms": float(np.percentile(times, 95)),
            "samples": len(times), "scope": "post-OSNet top50 + features + ONNX head + blend + top10 + refusal",
            "device": "CPU, 2 threads", "official_A5000_benchmark": False, "extract_changed": False}


def report_markdown(report):
    winner, frozen = report["inner"]["winner"], report["frozen"]
    lines = ["# Этап 3: компактный обучаемый реранкер", "", "MVP и его эмбеддинги не изменены.", "",
        f"Внутренний отбор: {winner['config']['name']}, {winner['selected_epoch']} эпох. "
        "Отбор на 184 identity, не участвовавших в градиентном обучении контрольного OSNet; затем новая голова на MVP / 925 identity.", "",
        f"Calibration: beta={frozen['beta']}; отказ `{frozen['choices']['combined']}`. "
        "Голова и число эпох заморожены до calibration; итоговые параметры — до validation.", "",
        "## Validation", "", "| Вариант | mAP@10, % | Rank-1, % | Rank-5, % | F1, % | TNR, % | 0.7F1+0.3TNR, % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["validation"].items():
        lines.append("| " + name + " | " + " | ".join(f"{value['metrics'][k]*100:.4f}" for k in
            ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "candidate_score")) + " |")
    lines += ["", "ranking_only — выбранный порядок со старым cosine-порогом; combined — с новым отказом; "
              "pair_only — диагностическая выдача только по голове, без смешивания.", ""]
    for reference in ("mvp", "neighbors"):
        delta = report["validation"]["combined"][f"paired_vs_{reference}"]
        lo, hi = delta["paired_bootstrap_95pct"]
        lines.append(f"- ΔmAP к {reference}: {100*delta['mAP_at_10_delta']:+.4f} п.п.; "
                     f"парный 95% интервал [{100*lo:+.4f}; {100*hi:+.4f}] п.п.")
    b = report["benchmark"]
    lines += ["", "## Стоимость и проверки", "",
        f"- Голова ONNX: {report['head_export']['bytes']/1024:.2f} KiB; все deployment-веса: {report['deployment_bytes']/1024**2:.2f} MiB.",
        f"- Голова ONNX/PyTorch max error: {report['head_export']['max_absolute_error']:.3g} (batch 1/50/100).",
        f"- Поиск после OSNet: median/p95 {b['median_query_ms']:.3f}/{b['p95_query_ms']:.3f} мс; граф {b['graph_seconds']:.3f} с.",
        "- CPU-замер, не RTX A5000. Extract остаётся прежним OSNet, новый визуальный encoder не добавлен.",
        f"- Inner baseline raw mAP воспроизведён: {report['inner_baseline_reproduced']}; MVP calibration/validation воспроизведены.",
        f"- Защищённые файлы неизменны: {report['protected_unchanged']}; время текущего запуска {report['elapsed_seconds']:.1f} с.", "",
        "## Ручные маски без изменения порога", "", "| Вариант | Исходные mAP@10, % | С масками mAP@10, % |", "|---|---:|---:|"]
    for name in ("mvp", "osnet_neighbors", "combined"):
        values = [report["mask_audit"]["scores"][c][name]["metrics"]["mAP_at_10"] for c in ("original", "manual")]
        lines.append(f"| {name} | {100*values[0]:.4f} | {100*values[1]:.4f} |")
    lines += ["", "## Ограничения", "",
        "- Это отдельный inner-only OSNet-контроль, не OOF-признаки. Его checkpoint ранее выбирался по этой же inner validation; "
        "она служит отбору, а не независимой финальной оценке. Outer validation также уже использовалась в истории проекта.",
        "- Нормализация признаков обучается только на train; 1 query на identity, равный вклад каждого query, "
        "баланс позитивов/негативов внутри query. В голове нет identity/camera/time/filename признаков.",
        "- Пары same-identity/same-camera и одинаковые исходные кадры игнорируются в loss. На инференсе GT-фильтров нет.",
        "- Переносится рецепт, не веса головы: пространство контрольного OSNet и MVP различается, финальная голова обучается с нуля.",
        "- Позитивов вне raw top50 реранкер восстановить не может. Обучение на эмбеддингах не убирает унаследованную зависимость encoder от фона/номера.",
        "- Sigmoid головы — оценка, не откалиброванная вероятность. Порог подбирается только на calibration.",
        "- Bootstrap: 2000 парных query-выборок с фиксированной gallery. Ручные маски: 126 кропов, только диагностика.",
        "- full_mAP/mINP берутся из исходных OSNet-эмбеддингов; mAP@10/Rank-1/5 — по отправленному порядку.",
        "- Ни один результат не внедряется в MVP автоматически. Для принятия нужен прирост порядка +1 п.п. без существенного ухудшения отказа.", "",
        "Файлы: inner_selection.json, weights/inner/, weights/final/, calibration.json, frozen_selection.json, "
        "report.json и validation/ (только локальные CSV, не организаторский test)."]
    return "\n".join(lines)+"\n"


def run(output=EXPERIMENT / "results/run_01"):
    torch.set_num_threads(2)
    protocols, outer_vectors, baseline, signature, hashes, source_summary, sample, difference = prepare()
    output, started = Path(output), time.perf_counter()
    if (output / "experiment.json").exists():
        _load(output / "experiment.json", signature)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new empty results directory")
        output.mkdir(parents=True, exist_ok=True)
        _save(output / "experiment.json", {"signature": signature})
    if (output / "report.json").exists():
        report = _load(output / "report.json", signature)
        if any(sha256(output / p) != h for p, h in report["output_sha256"].items()):
            raise ValueError("Completed experiment output changed")
        print("Reuse completed pair-reranker experiment; no training or retuning", flush=True)
        return report
    q, g = protocols["inner_train"]
    encoder, inner_export = export_control(output / "weights/control", signature, q)
    inner_rows = [r for name in ("inner_train", "inner_validation") for part in protocols[name] for r in part]
    vectors = encode_cached(encoder, inner_rows, output / "cache/inner.npz", signature)
    train = PairSet(q, g, vectors, signature["baseline"])
    qi, gi = protocols["inner_validation"]
    validation = PairSet(qi, gi, vectors, signature["baseline"])
    if not np.isclose(validation.raw.ranking["mAP@10"], source_summary["best_mAP_at_10"], rtol=0, atol=1e-8):
        raise ValueError("Clean OSNet control does not reproduce its recorded inner baseline")
    inner = select_inner(train, validation, hashes, output, signature)
    del train, validation, vectors, encoder
    q, g = protocols["development"]
    vectors = encode_cached(Encoder(), q+g, output / "cache/development.npz", signature)
    train = PairSet(q, g, vectors, signature["baseline"])
    targets, weights, valid = train.labels(hashes)
    final = fit_head(train, targets, weights, valid, inner["winner"]["config"], output / "weights/final", signature,
                     fixed_epochs=inner["winner"]["selected_epoch"])
    head_signature = {"experiment": digest(signature), "inner_selection": sha256(output / "inner_selection.json"),
                      "weights": final["weights_sha256"], "OSNet": sha256(MODEL)}
    head, exported = export_head(output / "weights/final/selected.pt", output / "weights/final", train.features, head_signature)
    del train, vectors, targets, weights, valid
    deployment_bytes = MODEL.stat().st_size + exported["bytes"]
    if deployment_bytes > 2_000_000_000:
        raise ValueError("Inference weights exceed the 2 GB limit")
    q, g = protocols["calibration"]
    cal = PairSet(q, g, outer_vectors, signature["baseline"])
    ranked, _ = control.ProtocolScores(q, g, outer_vectors).evaluate(control.ACTIVE)
    control.assert_baseline(metrics(ranked, baseline["threshold"]), baseline["calibration"])
    cal_path, frozen_path = output / "calibration.json", output / "frozen_selection.json"
    if frozen_path.exists():
        frozen = _load(frozen_path, signature)
        if frozen["calibration_sha256"] != sha256(cal_path) or frozen["head_sha256"] != exported["sha256"]:
            raise ValueError("Frozen calibration or head changed")
    else:
        tuned = tune_calibration(cal, head, baseline["threshold"])
        _save(cal_path, {"signature": signature, **tuned})
        frozen = {"signature": signature, "calibration_sha256": sha256(cal_path), "head_sha256": exported["sha256"],
            **{k: tuned[k] for k in ("beta", "choices", "validation_used", "head_weights_updated")},
            "baseline": signature["baseline"], "baseline_threshold": baseline["threshold"]}
        _save(frozen_path, frozen)
        print(f"FROZEN before outer validation: {frozen['choices']}", flush=True)
    del cal
    # No training, epoch selection or calibration below this boundary.
    q, g = protocols["validation"]
    results = evaluate_frozen(q, g, outer_vectors, head, frozen)
    control.assert_baseline(results["mvp"]["metrics"], baseline["validation"])
    control.export_validation(output / "validation", q, g, outer_vectors, results)
    timings, masks = benchmark(q, g, outer_vectors, head, frozen), mask_check(sample, head, frozen)
    if any(sha256(Path(p)) != h for p, h in signature["protected_sha256"].items()):
        raise RuntimeError("Protected source, model or MVP artifacts changed")
    if any(sha256(DATASET / "images" / f"{i}.jpg") != h for i, h in hashes.items()):
        raise RuntimeError("Dataset frames changed")
    report = {"signature": signature, "inner": inner, "final": final, "inner_export": inner_export,
        "head_export": exported, "frozen": frozen, "validation": results, "mask_audit": masks, "benchmark": timings,
        "deployment_bytes": deployment_bytes, "protected_unchanged": True, "inner_baseline_reproduced": True,
        "mvp_spot_check_max_abs_difference": difference, "elapsed_seconds": time.perf_counter()-started,
        "counts": {name: [len(q), len(g)] for name, (q, g) in protocols.items()}}
    (output / "RESULTS.md").write_text(report_markdown(report), encoding="utf-8")
    report["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    _save(output / "report.json", report)
    print(f"Saved {output / 'RESULTS.md'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results/run_01")
    run(parser.parse_args().output)
