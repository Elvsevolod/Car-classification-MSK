"""Frozen 2-model x 3-input audit on the corrected manual reference; no training."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.core import ARTIFACTS, DATASET, MODEL, PREPROCESS, ROOT, Encoder, bbox, read_rows, sha256
from backend.rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA
from training.audit import digest, encode, load_crop, validate_annotations
from training.mask_calibration import _load, _save
from training.mask_detection import compare_regions, summarize
from training.mask_finetune import check_plan, load_json
from training.mask_reid_ablation import (CALIBRATION, DETECTOR_EXPERIMENT, WEIGHTS,
                                         check_detector_separation, evaluate_embeddings, paired_deltas)
from training.stage6 import audit_partitions

EXPERIMENT = ROOT / "OSNet-AIN-x1.0/audit_09_model_mask_comparison"
REFERENCE = ROOT / "Mask-Detectors/reference_v02/results"
MASKED = ROOT / "OSNet-AIN-x1.0/variant_08_masked_hpo"
CONDITIONS = ("original", "manual", "yolo")
MODES = ("raw", "reranked")


class AuditEncoder(Encoder):
    """Use the existing encode_batch, but authorize only this hash-locked audit export."""
    def __init__(self, path, expected_sha256):
        if sha256(path) != expected_sha256:
            raise ValueError("Audit ONNX checksum mismatch")
        self.model_sha256 = expected_sha256
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name


def check_sample(rows, plan, split, dataset=DATASET):
    """Keep the original ordered protocol; reject leakage and altered source crops."""
    check_plan(plan)
    if plan["coordinates"] != "crop_xyxy_pixels_exclusive":
        raise ValueError("Unsupported mask coordinates")
    ids = plan["query_ids"] + plan["gallery_ids"]
    if len(set(ids)) != len(ids) or set(ids) != set(plan["images"]):
        raise ValueError("Query/gallery must be disjoint and cover exactly the annotated crops")
    by_id = {r["image_id"]: r for r in rows}
    selected = [by_id[i] for i in ids]
    if not {r["vehicle_id"] for r in selected} <= set(split["identities"]["validation"]):
        raise ValueError("Manual reference is not confined to outer validation")
    for row in selected:
        i = row["image_id"]
        expected = plan["images"][i]
        if (list(bbox(row)) != expected["bbox"]
                or sha256(dataset / "images" / f"{i}.jpg") != expected["frame_sha256"]
                or split["frame_sha256"][i] != expected["frame_sha256"]
                or load_crop(row, dataset).size != (expected["width"], expected["height"])):
            raise ValueError(f"Source frame or bbox changed: {i}")
    return [by_id[i] for i in plan["query_ids"]], [by_id[i] for i in plan["gallery_ids"]]


def cached_predictions(plan, cache):
    """Validate detector rectangles without relabeling them as human-reviewed."""
    images = {}
    for i, expected in plan["images"].items():
        item = cache["images"][i]
        if (item["width"], item["height"]) != (expected["width"], expected["height"]):
            raise ValueError("Automatic mask dimensions differ from the manual reference")
        scores = item["confidences"]
        if len(scores) != len(item["rectangles"]) or not np.isfinite(scores).all():
            raise ValueError("Invalid automatic mask confidences")
        images[i] = item
    # The validator checks exact integer crop-local geometry; this copy is not saved.
    validate_annotations(plan, {"fingerprint": plan["fingerprint"], "images": {
        i: {**item, "reviewed": True} for i, item in images.items()}})
    return images


def prepare(dataset=DATASET):
    dataset = Path(dataset)
    paths = {
        "manual_plan": REFERENCE / "mask_plan.json", "manual": REFERENCE / "masks.json",
        "manual_provenance": REFERENCE / "comparison.json",
        "split": ARTIFACTS / "splits.json", "baseline": ARTIFACTS / "baseline_metrics.json",
        "training_protocol": MASKED / "results/protocol.json",
        "training_run": MASKED / "results/masked_run_01/run_manifest.json",
        "automatic_masks": MASKED / "results/cache/automatic_masks.json",
        "new_evaluation": MASKED / "results/masked_run_01/selected/evaluation.json",
        "detector_plan": DETECTOR_EXPERIMENT / "annotation/mask_plan.json",
        "detector_calibration": CALIBRATION / "calibration.json", "detector_weights": WEIGHTS,
        "train_csv": dataset / "train.csv", "mvp_weights": MODEL,
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    plan, split = load_json(paths["manual_plan"]), load_json(paths["split"])
    manual = validate_annotations(plan, load_json(paths["manual"]))
    if hashes["manual"] != load_json(paths["manual_provenance"])["source_sha256"]:
        raise ValueError("Corrected manual reference differs from masks-7 provenance")
    if hashes["train_csv"] != split["train_csv_sha256"]:
        raise ValueError("train.csv changed since calibration")
    rows = read_rows(paths["train_csv"])
    audit_partitions(rows, split["frame_sha256"], split["identities"])
    query, gallery = check_sample(rows, plan, split, dataset)
    detector_plan = load_json(paths["detector_plan"])
    check_plan(detector_plan)
    if detector_plan["outer_split_sha256"] != hashes["split"]:
        raise ValueError("Detector outer split changed")
    check_detector_separation(query + gallery, split["frame_sha256"], detector_plan)

    protocol = _load(paths["training_protocol"])["signature"]
    manifest = _load(paths["training_run"])["signature"]
    cache = _load(paths["automatic_masks"])
    evaluation = _load(paths["new_evaluation"])
    frozen = _load(paths["detector_calibration"])
    if (manifest["protocol"] != digest(protocol)
            or manifest["mask_cache_sha256"] != hashes["automatic_masks"]
            or cache["signature"]["protocol"] != digest(protocol)
            or evaluation["signature"]["run"] != manifest
            or protocol["outer_split_sha256"] != hashes["split"]
            or protocol["rows"] != rows or protocol["frames"] != split["frame_sha256"]
            or set(cache["images"]) != {r["image_id"] for r in rows}
            or protocol["detector_sha256"] != hashes["detector_weights"]
            or frozen["signature"]["weights_sha256"] != hashes["detector_weights"]
            or protocol["calibration_sha256"] != hashes["detector_calibration"]
            or protocol["policy"] != frozen["selected"]
            or evaluation["mask_policy_required"] != frozen["selected"]):
        raise ValueError("Frozen training export, detector and automatic mask cache disagree")
    if (evaluation["config"]["resize_mode"] != "square"
            or protocol["code"]["evaluate.py"] != sha256(ROOT / "evaluate.py")):
        raise ValueError("New model preprocessing or evaluator is incompatible with this audit")
    automatic = cached_predictions(plan, cache)

    mvp = Encoder()
    baseline = load_json(paths["baseline"])
    reranking = {"k1": ACTIVE_K1, "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA}
    if (baseline["model_sha256"] != mvp.model_sha256
            or baseline["encoder_fingerprint"] != mvp.fingerprint
            or baseline["preprocessing"] != PREPROCESS
            or baseline["evaluator"]["sha256"] != sha256(ROOT / "evaluate.py")
            or any(baseline["search"][k] != v for k, v in reranking.items())
            or evaluation["reranking"] != reranking):
        raise ValueError("Frozen MVP calibration or common reranking differs")
    thresholds = {"mvp": float(baseline["threshold"]),
                  "masked_trained": float(evaluation["scores"]["reranked"]["threshold"])}
    if evaluation["scores"]["raw"]["threshold"] != thresholds["masked_trained"]:
        raise ValueError("New model must have one fixed raw/reranked refusal threshold")
    paths["masked_weights"] = ROOT / evaluation["onnx"]
    hashes["masked_weights"] = sha256(paths["masked_weights"])
    encoders = {"mvp": mvp, "masked_trained": AuditEncoder(paths["masked_weights"], evaluation["onnx_sha256"])}
    sources = ("training/mask_model_comparison.py", "training/audit.py", "training/mask_reid_ablation.py",
               "training/mask_detection.py", "backend/core.py", "backend/rerank.py", "backend/scoring.py",
               "evaluate.py")
    signature = {"version": 1, "query": query, "gallery": gallery,
        "frames": {i: plan["images"][i]["frame_sha256"] for i in plan["images"]},
        "source_sha256": hashes, "thresholds": thresholds, "reranking": reranking,
        "preprocess": PREPROCESS, "flip_tta": False, "fill_rgb": [0, 0, 0],
        "mask_policy": frozen["selected"], "manual_source": "masks-7.json",
        "manual_sample_fingerprint": plan["fingerprint"], "detector_device": cache["signature"]["device"],
        "code_sha256": {p: sha256(ROOT / p) for p in sources},
        "runtime": {p: importlib.metadata.version(p) for p in ("numpy", "onnxruntime", "Pillow")}}
    protected = {str(p): sha256(p) for p in [*paths.values(), *ARTIFACTS.glob("*.npy"),
                 *ARTIFACTS.glob("*.sqlite3"), ARTIFACTS / "export_manifest.json"] if p.is_file()}
    return query, gallery, encoders, {"original": None, "manual": manual, "yolo": automatic}, signature, protected


def extract_matrix(encoders, rows, annotations, dataset=DATASET):
    vectors = {}
    for name, encoder in encoders.items():
        vectors[name] = {}
        for condition in CONDITIONS:
            print(f"Encoding {name} / {condition}: {len(rows)} crops", flush=True)
            found = encode(encoder, rows, dataset, annotations[condition])
            vectors[name][condition] = np.stack([found[r["image_id"]] for r in rows])
    return vectors


def evaluate_matrix(query, gallery, vectors, thresholds):
    scores, details, paired, cosines = {}, {}, {}, {}
    for name, conditions in vectors.items():
        scores[name], details[name], paired[name], cosines[name] = {}, {}, {}, {}
        for condition in CONDITIONS:
            scores[name][condition], details[name][condition] = evaluate_embeddings(
                query, gallery, conditions[condition], thresholds[name])
        for before, after in (("original", "manual"), ("original", "yolo"), ("manual", "yolo")):
            key = f"{after}_minus_{before}"
            paired[name][key] = {mode: paired_deltas(details[name][before][mode], details[name][after][mode])
                                 for mode in MODES}
            values = np.clip((conditions[before] * conditions[after]).sum(axis=1), -1, 1)
            cosines[name][key] = {"mean": float(values.mean()), "min": float(values.min()),
                "per_image": dict(zip((r["image_id"] for r in query + gallery), values.tolist()))}
    between = {condition: {mode: paired_deltas(details["mvp"][condition][mode],
                                              details["masked_trained"][condition][mode]) for mode in MODES}
               for condition in CONDITIONS}
    return {"scores": scores, "per_query": details, "within_model": paired,
            "masked_trained_minus_mvp": between, "embedding_cosine": cosines}


def markdown_report(report):
    sig, scores = report["signature"], report["scores"]
    first = scores["mvp"]["original"]["reranked"]
    lines = ["# Две модели × три варианта изображения", "",
        f"Общий протокол: {report['queries']} query / {report['gallery']} gallery; "
        f"{first['known_queries']} known / {first['unknown_queries']} unknown. Ручная разметка: masks-7.json.", "",
        "Маски применяются к query **и** gallery после BBox-кропа и до resize 208×208. "
        "ONNX CPU, без TTA. Реранкинг k1=20, k2=3, lambda=0.5; confidence = max raw cosine.", "",
        f"Фиксированные пороги: MVP **{sig['thresholds']['mvp']:.10f}**, "
        f"masked-trained **{sig['thresholds']['masked_trained']:.10f}**. "
        "Пороги взяты из прежней калибровки; на этой выборке ничего не подбиралось.", "",
        "| Модель | Изображения | mAP@10 raw, % | mAP@10 rerank, % | Rank-1 rerank, % | F1, % | TNR, % | TP/FP/FN/TN |",
        "|---|---|---:|---:|---:|---:|---:|---|" ]
    for model in scores:
        for condition in CONDITIONS:
            raw, rr = (scores[model][condition][mode] for mode in MODES)
            lines.append(f"| {model} | {condition} | {raw['mAP_at_10']*100:.4f} | "
                f"{rr['mAP_at_10']*100:.4f} | {rr['Rank_1']*100:.2f} | {rr['candidate_F1']*100:.2f} | "
                f"{rr['TNR']*100:.2f} | {rr['TP']}/{rr['FP']}/{rr['FN']}/{rr['TN']} |")
    lines += ["", "## Изменения внутри каждой модели", "",
        "Разности mAP@10 после реранкинга; интервалы — 2000 парных bootstrap-перевыборок query "
        "при фиксированной gallery. Это локальная диагностика, а не независимый статистический тест.", "",
        "| Модель | Сравнение | Δ mAP, п.п. | 95% интервал, п.п. | Top-1 лучше / хуже |",
        "|---|---|---:|---|---|" ]
    for model, changes in report["within_model"].items():
        for name, modes in changes.items():
            d = modes["reranked"]
            lo, hi = d["paired_bootstrap_95pct"]
            lines.append(f"| {model} | {name} | {d['mAP_at_10_delta']*100:+.4f} | "
                f"[{lo*100:+.4f}; {hi*100:+.4f}] | {len(d['top1_improved'])} / {len(d['top1_worsened'])} |")
    c = report["mask_agreement"]["summary"]
    lines += ["", "## Покрытие ручных областей автоматическими масками", "",
        f"YOLO confidence={sig['mask_policy']['confidence']}, margin={sig['mask_policy']['margin']} "
        "с каждой стороны; используются те же сохранённые маски, что при masked HPO.", "",
        f"- Ручных областей: {c['reference_regions']}; автоматических: {c['predicted_regions']}.",
        f"- Покрыто {c['reference_pixel_coverage']*100:.2f}% пикселей ручной разметки; "
        f"не менее 90% площади у {c['regions_covered_90pct']}/{c['reference_regions']} областей.",
        f"- За пределами ручных масок: {c['outside_reference_fraction_of_crop']*100:.2f}% площади всех кропов; "
        f"{c['outside_reference_fraction_of_prediction']*100:.2f}% площади автоматических масок.", "",
        "## Ограничения и интерпретация", "",
        *[f"- {item}" for item in report["limitations"]], "",
        "Подробности: comparison.json (метрики, top-10, confidence, парные изменения и покрытие каждого кропа), "
        "embeddings.npz (6 наборов векторов с единым порядком image_id), experiment.json (зафиксированные входы).", "",
        f"Проверка неизменности защищённых файлов: {report['protected_unchanged']}. "
        f"Время вычислений: {report['elapsed_seconds']:.1f} с. Обучение и переиндексация MVP не выполнялись."]
    return "\n".join(lines) + "\n"


def run(output=EXPERIMENT / "results/comparison_v1", dataset=DATASET):
    query, gallery, encoders, annotations, signature, protected = prepare(dataset)
    output = Path(output)
    manifest, result_path = output / "experiment.json", output / "comparison.json"
    if manifest.exists():
        _load(manifest, signature)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choose a new empty audit output directory")
        output.mkdir(parents=True, exist_ok=True)
        _save(manifest, {"signature": signature})
    if result_path.exists():
        report = _load(result_path, signature)
        if any(sha256(output / name) != value for name, value in report["output_sha256"].items()):
            raise ValueError("Completed audit artifacts changed")
        print("Reuse complete 2 x 3 comparison; no inference", flush=True)
        return report
    started = time.perf_counter()
    rows = query + gallery
    vectors = extract_matrix(encoders, rows, annotations, dataset)
    report = evaluate_matrix(query, gallery, vectors, signature["thresholds"])
    per_image = {r["image_id"]: compare_regions((r["w"], r["h"]),
        annotations["manual"][r["image_id"]]["rectangles"], annotations["yolo"][r["image_id"]]["rectangles"],
        annotations["yolo"][r["image_id"]]["confidences"]) for r in rows}
    if any(sha256(Path(p)) != value for p, value in protected.items()):
        raise RuntimeError("Protected source annotations, model or MVP artifacts changed")
    if any(sha256(Path(dataset) / "images" / f"{i}.jpg") != value for i, value in signature["frames"].items()):
        raise RuntimeError("Source images changed during the audit")
    temporary = output / "embeddings.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray([r["image_id"] for r in rows]),
            fingerprint=digest(signature), **{f"{model}__{condition}": array
                for model, conditions in vectors.items() for condition, array in conditions.items()})
    temporary.replace(output / "embeddings.npz")
    report.update({"signature": signature, "queries": len(query), "gallery": len(gallery),
        "mask_agreement": {"summary": summarize(per_image), "per_image": per_image},
        "protected_unchanged": True, "protected_sha256": protected,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Маленькая, относительно простая выборка: не полная validation и не закрытый тест организаторов. "
            "Историческая validation уже использовалась в проекте; это не новый независимый holdout.",
            "Ручные маски исправлены (v2); прошлый аудит использовал v1, поэтому его числа не подставляются в таблицу.",
            "Размечены области анонимизации, не отдельный класс только номеров. Это не точные маски контрольного теста организаторов.",
            "Отсутствие падения качества не доказывает независимость от номеров: остаются форма маски, контекст и возможные пропуски.",
            "YOLO vs manual меняет и пропуски, и лишнюю закрытую площадь одновременно; нельзя причинно разделить их по одной разности mAP.",
            "Две модели различаются не только масками, но и результатом HPO/выбором checkpoint. Это сравнение готовых моделей, не чистая причинная оценка обучения с масками.",
            "Пороги моделей разные и ранее калибровались на разных вариантах изображений; F1 нельзя трактовать как чистое качество эмбеддингов. "
            "Порог каждой модели одинаков во всех трёх условиях; mAP и Rank-1 от него не зависят.",
            "Исходные изображения, ручная разметка, веса и индекс MVP не изменялись. Новые веса автоматически не внедряются."]})
    (output / "RESULTS.md").write_text(markdown_report(report), encoding="utf-8")
    report["output_sha256"] = {name: sha256(output / name) for name in ("embeddings.npz", "RESULTS.md")}
    _save(result_path, report)
    print(f"Saved {output / 'RESULTS.md'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results/comparison_v1")
    args = parser.parse_args()
    run(args.output)
