"""Stage 2: frozen OSNet + CLIP, calibration-only fusion; no training or MVP changes."""
import argparse
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch

from backend.core import ARTIFACTS, DATASET, MODEL, ROOT, normalize, read_rows, sha256
from backend.scoring import metrics, ranked_queries
from training import rerank_score_control as control
from training.audit import digest, validate_annotations
from training.ensemble_inference import benchmark_extract, encode_cached, export_checkpoint, fuse_arrays
from training.mask_calibration import _load, _save
from training.mask_finetune import load_json
from training.mask_model_comparison import check_sample
from training.mask_reid_ablation import paired_deltas
from training.stage6 import audit_partitions

EXPERIMENT = ROOT / "OSNet-AIN-x1.0/audit_11_osnet_clip_ensemble"
CLIP = ROOT / "CLIP-ReID-ViT-B-16/variant_02_hpo"
CHECKPOINT = CLIP / "weights/trial_011/image_best.pt"
REFERENCE = ROOT / "Mask-Detectors/reference_v02/results"
ALPHAS = (1., .9, .75, .5, .25, 0.)
CLIP_PREPROCESS = "exif-rgb-exact-bbox-bilinear256-mean0.5-std0.5-beforeBN-concat1280-l2-v1"


def configuration(alpha, kind="kreciprocal", pool="full", **overrides):
    return {**control.ACTIVE, "alpha": alpha, "kind": kind, "pool": pool, **overrides}


def initial_grid():
    return [configuration(a, kind) for a in ALPHAS for kind in ("cosine", "kreciprocal")] + [
        configuration(a, pool="union50") for a in ALPHAS if 0 < a < 1]


def refinement_grid(base):
    return [base.copy()] + [{**base, key: value} for key, values in (
        ("k1", (10, 30)), ("k2", (1, 6)), ("lambda_value", (.35, .65))) for value in values]


def union_pool(osnet_scores, clip_scores, k=50):
    if osnet_scores.shape != clip_scores.shape or osnet_scores.ndim != 2:
        raise ValueError("Candidate union needs aligned query/gallery scores")
    mask = np.zeros(osnet_scores.shape, dtype=bool)
    for scores in (osnet_scores, clip_scores):
        indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        np.put_along_axis(mask, indices, True, axis=1)
    return mask


class FusionScores:
    """Graphs contain only gallery vectors; labels are used only by evaluate()."""
    def __init__(self, query, gallery, osnet, clip):
        self.query, self.gallery = query, gallery
        self.ids = [r["image_id"] for r in query + gallery]
        self.o = normalize(np.stack([osnet[i] for i in self.ids]))
        self.c = normalize(np.stack([clip[i] for i in self.ids]))
        n = len(query)
        self.union = union_pool(self.o[:n] @ self.o[n:].T, self.c[:n] @ self.c[n:].T)
        self.features, self.parts = {}, {}

    def embeddings(self, alpha):
        if alpha not in self.features:
            self.features[alpha] = dict(zip(self.ids, fuse_arrays(self.o, self.c, alpha)))
        return self.features[alpha]

    def scores(self, config):
        alpha, n = config["alpha"], len(self.query)
        embeddings = self.embeddings(alpha)
        vectors = np.stack([embeddings[i] for i in self.ids])
        cosine = np.clip(vectors[:n] @ vectors[n:].T, -1, 1)
        if config["kind"] == "cosine":
            scores = cosine.copy()
        else:
            key = (alpha, config["k1"], config["k2"])
            if key not in self.parts:
                engine = control.ScoreControl(vectors[n:], config["k1"], config["k2"])
                self.parts[key] = [engine.components(v) for v in vectors[:n]]
            scores = np.stack([control.ranking_scores(p, {**config, "pool": 0}) for p in self.parts[key]])
        if config["pool"] == "union50":
            scores = np.where(self.union, scores, scores.min(axis=1, keepdims=True) - 1)
        elif config["pool"] != "full":
            raise ValueError("Unknown candidate pool")
        return scores

    def evaluate(self, config):
        return ranked_queries(self.query, self.gallery, self.embeddings(config["alpha"]), self.scores(config))


def complementarity(engine):
    """Label-based diagnostics only; never supplies inference features or masks."""
    n = len(engine.query)
    orders = {name: np.argsort(-(v[:n] @ v[n:].T), axis=1, kind="stable")
              for name, v in (("osnet", engine.o), ("clip", engine.c))}
    raw = {name: control.query_details(engine.evaluate(configuration(a, "cosine")), np.zeros(n), 1.)
           for name, a in (("osnet", 1.), ("clip", 0.))}
    counts = dict(both=0, only_osnet=0, only_clip=0, neither=0)
    pools = {name: [] for name in ("osnet10", "clip10", "osnet50", "clip50", "union50")}
    details = {}
    for j, row in enumerate(engine.query):
        positives = {i for i, g in enumerate(engine.gallery)
                     if g["vehicle_id"] == row["vehicle_id"] and g["camera_id"] != row["camera_id"]}
        if not positives:
            continue
        qid = row["image_id"]
        o, c = raw["osnet"][qid]["top1_correct"], raw["clip"][qid]["top1_correct"]
        counts["both" if o and c else "only_osnet" if o else "only_clip" if c else "neither"] += 1
        sets = {f"{name}{k}": set(order[j, :k]) for name, order in orders.items() for k in (10, 50)}
        sets["union50"] = sets["osnet50"] | sets["clip50"]
        captured = {name: len(pool & positives) for name, pool in sets.items()}
        for name, found in captured.items():
            pools[name].append((found, len(positives)))
        details[qid] = {"valid_positives": len(positives), "captured": captured,
                       "shared_positive_top50": len(positives & sets["osnet50"] & sets["clip50"])}
    summary = {}
    for name, rows in pools.items():
        found, total = np.asarray(rows).T
        summary[name] = {"query_hit_rate": float(np.mean(found > 0)), "queries_with_positive": int((found > 0).sum()),
            "micro_positive_recall": float(found.sum() / total.sum()),
            "mean_positive_recall": float(np.mean(found / total)),
            "oracle_mAP_at_10_upper_bound": float(np.mean(np.minimum(found, 10) / np.minimum(total, 10)))}
    return {"known_queries": sum(counts.values()), "raw_top1_overlap": counts, "pools": summary,
            "union_size_mean": float(engine.union.sum(1).mean()), "per_query": details,
            "oracle_note": "Unattainable label oracle: all available valid positives first, junk last. Diagnostic only."}


def tune_calibration(query, gallery, osnet, clip):
    engine = FusionScores(query, gallery, osnet, clip)

    def trial(config):
        ranked = engine.evaluate(config)
        chosen = control.choose_threshold(ranked, ranked.confidence)
        print(f"Calibration alpha={config['alpha']:g} {config['kind']} {config['pool']} "
              f"k={config['k1']}/{config['k2']} lambda={config['lambda_value']} "
              f"mAP={chosen['metrics']['mAP_at_10']:.6f}", flush=True)
        return {"config": config, **chosen}

    initial = [trial(c) for c in initial_grid()]
    # Refine a reranked candidate only; raw remains an explicit diagnostic control.
    key = lambda r: (r["metrics"]["mAP_at_10"], r["metrics"]["Rank_1"],
                     r["config"]["alpha"], r["config"]["pool"] == "full")
    fixed = max((r for r in initial if r["config"]["kind"] == "kreciprocal"), key=key)
    refined = [fixed] + [trial(c) for c in refinement_grid(fixed["config"])[1:]]
    best = max(refined, key=key)
    choices = {"fusion_fixed": fixed, "fusion_selected": best,
        "clip_raw": next(r for r in initial if r["config"]["alpha"] == 0 and r["config"]["kind"] == "cosine"),
        "clip_reranked": next(r for r in initial if r["config"]["alpha"] == 0 and r["config"]["kind"] == "kreciprocal")}
    return {"initial": initial, "refinement": refined, "complementarity": complementarity(engine),
        "choices": {k: {"config": r["config"], "threshold": r["threshold"], "confidence": "max_fused_cosine"}
                    for k, r in choices.items()}, "validation_used": False,
        "selection": "calibration mAP@10, Rank-1, higher OSNet alpha, full pool, grid order; refusal .7F1+.3TNR, F1, threshold"}


def result(ranked, threshold):
    confidence = ranked.confidence.astype(float)
    return {"threshold": threshold, "metrics": metrics(ranked, threshold, confidence),
            "per_query": control.query_details(ranked, confidence, threshold)}


def evaluate_frozen(query, gallery, osnet, clip, frozen):
    engine = FusionScores(query, gallery, osnet, clip)
    results = {name: {"config": choice["config"], **result(engine.evaluate(choice["config"]), choice["threshold"])}
               for name, choice in frozen["choices"].items()}
    for name, config in (("mvp", control.ACTIVE), ("osnet_neighbors", frozen["osnet_neighbors"])):
        ranked, _ = control.ProtocolScores(query, gallery, osnet).evaluate(config)
        results[name] = {"config": config, **result(ranked, frozen["baseline_threshold"])}
    results = {name: results[name] for name in ("mvp", "osnet_neighbors", "clip_raw", "clip_reranked", "fusion_fixed", "fusion_selected")}
    for name, item in results.items():
        if name != "mvp":
            item["paired_vs_mvp"] = paired_deltas(results["mvp"]["per_query"], item["per_query"])
    return results, engine


def check_clip_protocol(protocol, manifest, summary, winner, rows, hashes, split):
    if (protocol["outer"] != split["identities"] or not protocol["identity_and_exact_frame_disjoint"]
            or protocol["data_sha256"] != digest({"rows": rows, "frames": hashes})
            or protocol["preprocess"] != CLIP_PREPROCESS
            or protocol["evaluator_sha256"] != sha256(ROOT / "evaluate.py")
            or manifest["protocol"] != digest(protocol) or digest(manifest) != winner["signature"]
            or summary["signature"]["protocol"] != digest(protocol)
            or summary["signature"]["experiment"] != winner["signature"]
            or winner["winner"] != {**summary, "name": "trial_011"} or summary["status"] != "complete"):
        raise ValueError("CLIP checkpoint provenance or outer protocol differs")
    audit_partitions(rows, hashes, protocol["outer"])
    development = [r for r in rows if r["vehicle_id"] in set(protocol["outer"]["train"])]
    audit_partitions(development, hashes, protocol["inner"])
    for source in ("training/clip_reid.py", "training/vendor/clip_reid/model.py", "backend/core.py", "evaluate.py"):
        if sha256(ROOT / source) != manifest["code"][source]:
            raise ValueError(f"CLIP inference source differs from training: {source}")


def prepare():
    protocols, osnet, baseline, old_signature, hashes, difference = control.prepare()
    rows, split = read_rows(DATASET / "train.csv"), load_json(ARTIFACTS / "splits.json")
    paths = {"checkpoint": CHECKPOINT, "protocol": CLIP / "results/protocol.json", "manifest": CLIP / "results/manifest.json",
        "summary": CLIP / "results/runs/trial_011/summary.json", "winner": CLIP / "results/experiment_summary.json",
        "stage1": control.EXPERIMENT / "results/run_01/frozen_selection.json",
        "manual": REFERENCE / "masks.json", "manual_plan": REFERENCE / "mask_plan.json",
        "manual_provenance": REFERENCE / "comparison.json"}
    protocol, manifest, summary, winner = [load_json(paths[k]) for k in ("protocol", "manifest", "summary", "winner")]
    check_clip_protocol(protocol, manifest, summary, winner, rows, hashes, split)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if (checkpoint["signature"] != summary["signature"] or checkpoint["epoch"] != summary["best"]["epoch"]
            or checkpoint["validation"] != summary["best"]["validation"] or checkpoint["config"] != summary["config"]):
        raise ValueError("CLIP checkpoint is not the recorded HPO winner")
    del checkpoint
    stage1 = _load(paths["stage1"], old_signature)
    plan = load_json(paths["manual_plan"])
    manual = validate_annotations(plan, load_json(paths["manual"]))
    mq, mg = check_sample(rows, plan, split)
    audit = _load(control.MASK_AUDIT / "comparison.json")
    if (sha256(paths["manual"]) != load_json(paths["manual_provenance"])["source_sha256"]
            or audit["signature"]["query"] != mq or audit["signature"]["gallery"] != mg
            or audit["signature"]["source_sha256"]["manual"] != sha256(paths["manual"])
            or audit["signature"]["source_sha256"]["mvp_weights"] != sha256(MODEL)
            or audit["output_sha256"]["embeddings.npz"] != sha256(control.MASK_AUDIT / "embeddings.npz")):
        raise ValueError("Manual cache, source crops or corrected masks changed")
    protected = {**old_signature["protected_sha256"], **{str(p): sha256(p) for p in paths.values()}}
    for name in ("training/ensemble_experiment.py", "training/ensemble_inference.py", "training/clip_export.py",
                 "training/clip_reid.py", "training/vendor/clip_reid/model.py", "training/audit.py",
                 "training/mask_model_comparison.py", "training/stage6.py", "training/mask_calibration.py"):
        protected[str(ROOT / name)] = sha256(ROOT / name)
    signature = {"version": 1, "protected_sha256": protected, "protocols": old_signature["protocols"],
        "grid": initial_grid(), "refinement_axes": {"k1": [10, 30], "k2": [1, 6], "lambda_value": [.35, .65]},
        "clip_epoch": summary["best"]["epoch"], "clip_train_identities": len(protocol["inner"]["train"]),
        "osnet_train_identities": len(protocol["outer"]["train"]), "clip_preprocess": CLIP_PREPROCESS,
        "osnet_preprocess": old_signature["preprocess"], "flip_tta": False, "main_masks": False,
        "runtime": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "onnxruntime", "Pillow")}}
    return protocols, osnet, baseline, signature, hashes, stage1, (mq, mg, manual), difference


def mask_check(encoder, output, signature, sample, frozen):
    q, g, manual = sample
    with np.load(control.MASK_AUDIT / "embeddings.npz", allow_pickle=False) as cache:
        ids = cache["ids"].tolist()
        if ids != [r["image_id"] for r in q + g]:
            raise ValueError("Manual audit cache order differs")
        osnet = {condition: dict(zip(ids, cache[f"mvp__{condition}"].copy())) for condition in ("original", "manual")}
    results = {}
    for condition, annotations in (("original", None), ("manual", manual)):
        clip = encode_cached(encoder, q + g, output / f"cache/manual_{condition}.npz", signature, annotations)
        scores, _ = evaluate_frozen(q, g, osnet[condition], clip, frozen)
        results[condition] = scores
    return {"query_count": len(q), "gallery_count": len(g), "scores": results,
            "paired_manual_vs_original": {name: paired_deltas(results["original"][name]["per_query"],
                results["manual"][name]["per_query"]) for name in results["original"]},
            "selection_used": False, "thresholds_retuned": False,
            "warning": "Small reused 126-crop reference, masks-7; not evidence of complete plate independence"}


def benchmark_search(engine, config, repeats=3):
    """One query plus static gallery, including both cosine lookups for union50."""
    n = len(engine.query)
    vectors = np.stack([engine.embeddings(config["alpha"])[i] for i in engine.ids])
    started = time.perf_counter()
    graph = control.ScoreControl(vectors[n:], config["k1"], config["k2"])
    build = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        for i in range(n):
            started = time.perf_counter()
            parts = graph.components(vectors[i])
            scores = control.ranking_scores(parts, {**config, "pool": 0})
            if config["pool"] == "union50":
                pool = union_pool((engine.o[n:] @ engine.o[i])[None], (engine.c[n:] @ engine.c[i])[None])[0]
                scores = np.where(pool, scores, scores.min() - 1)
            np.argsort(-scores, kind="stable")[:10]
            float(parts["cosine"].max())
            samples.append(1000 * (time.perf_counter() - started))
    return {"gallery_graph_seconds": build, "samples": len(samples),
        "median_query_ms": float(np.median(samples)), "p95_query_ms": float(np.percentile(samples, 95)),
        "scope": "post-embedding graph scoring + optional two-model union50 + top10 + confidence; no labels",
        "not_organizer_A5000_benchmark": True}


def report_markdown(report):
    lines = ["# Этап 2: OSNet + CLIP", "", "Эксперимент с готовыми весами; MVP не изменён, обучения нет.", "",
        f"Выбор только на calibration: `{report['frozen']['choices']['fusion_selected']}`.", "",
        "## Общая validation", "",
        "| Вариант | mAP@10, % | Rank-1, % | Rank-5, % | F1, % | TNR, % | 0.7F1+0.3TNR, % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in report["validation"].items():
        lines.append("| " + name + " | " + " | ".join(f"{item['metrics'][k]*100:.4f}" for k in
            ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "candidate_score")) + " |")
    lines += ["", "mvp — неизменённый baseline; osnet_neighbors — фиксированный контроль первого этапа; "
        "fusion_fixed — выбор доли и пула при 20/3/0.5; fusion_selected — после ограниченного уточнения реранкинга.", ""]
    for name in ("clip_reranked", "fusion_fixed", "fusion_selected"):
        d = report["validation"][name]["paired_vs_mvp"]
        lo, hi = d["paired_bootstrap_95pct"]
        lines.append(f"- {name}: ΔmAP {100*d['mAP_at_10_delta']:+.4f} п.п., парный 95% интервал "
                     f"[{lo*100:+.4f}; {hi*100:+.4f}] п.п.")
    lines += ["", "## Дополняемость (calibration, raw cosine)", "",
        f"Top-1: `{report['complementarity']['raw_top1_overlap']}`.", "",
        "| Пул | Query с позитивом, % | Доля всех найденных позитивов, % | Недостижимый oracle mAP@10, % |",
        "|---|---:|---:|---:|"]
    for name, s in report["complementarity"]["pools"].items():
        lines.append(f"| {name} | {100*s['query_hit_rate']:.3f} | {100*s['micro_positive_recall']:.3f} | "
                     f"{100*s['oracle_mAP_at_10_upper_bound']:.3f} |")
    lines += ["", "## Стоимость", "",
        f"- OSNet + CLIP ONNX: {report['export']['combined_osnet_clip_bytes']/1024**2:.2f} MiB (< 2 GB).",
        f"- CLIP ONNX/PyTorch max absolute difference: {report['export']['max_absolute_error']:.3g}."]
    for name, b in report["benchmark"]["extract"].items():
        lines.append(f"- {name}: extract batch1 median/p95 {b['median_ms']:.2f}/{b['p95_ms']:.2f} мс, "
                     f"batch16 {b['throughput_fps']:.2f} crop/s (CPU, 2 потока).")
    search = report["benchmark"]["search"]
    lines.append(f"- Выбранный поиск: граф {search['gallery_graph_seconds']:.3f} с; "
                 f"query median/p95 {search['median_query_ms']:.3f}/{search['p95_query_ms']:.3f} мс.")
    lines += ["- Extract включает чтение JPEG, EXIF, bbox, индивидуальные transform, энкодеры и fusion; поиск измерен отдельно.",
        "- Это не конкурсный замер на RTX A5000; соответствие latency/throughput/VRAM на ней не проверено.", "",
        "## Исправленные ручные маски: одинаковый порог до и после", "",
        "| Модель | Исходные mAP@10, % | С масками mAP@10, % |", "|---|---:|---:|"]
    for name in ("mvp", "clip_reranked", "fusion_selected"):
        original, manual = [report["mask_audit"]["scores"][c][name]["metrics"] for c in ("original", "manual")]
        lines.append(f"| {name} | {100*original['mAP_at_10']:.4f} | {100*manual['mAP_at_10']:.4f} |")
    lines += ["", "## Ограничения", "",
        "- CLIP: trial_011, epoch 33, обучение на 741 inner-train identity; OSNet: 925. Это сравнение готовых систем, не равное обучение архитектур.",
        "- Веса fusion: 1/.9/.75/.5/.25/0; 16 первичных конфигураций, затем 6 дополнительных изменений одного параметра. "
        "Нет нового Optuna, обучения, TTA, YOLO или cross-query операций.",
        "- Validation уже использовалась ранее: локальный диагностический контроль, не независимый финальный тест.",
        "- Все решения текущего этапа зафиксированы до validation; confidence=max fused cosine, не вероятность.",
        "- full_mAP и mINP в JSON по контракту считаются по cosine эмбеддингов, а mAP@10/Rank-1/5 — по отправленному ранжированию.",
        "- Bootstrap: 2000 парных query-перевыборок при фиксированной gallery; не учитывает историю подбора.",
        "- Ручная выборка 126 кропов — только диагностика; не доказывает независимость от номеров.",
        "- Победитель не внедряется автоматически; отдельно оцениваем retrieval, отказ, устойчивость и стоимость.", "",
        "calibration.json: вся сетка, кривые отказа и дополняемость; frozen_selection.json: зафиксированный выбор; "
        "report.json: метрики, top10, парные изменения, масочный аудит, замеры; validation/: локальные CSV + embeddings.npy по каждой системе.", "",
        f"Защищённые источники неизменны: {report['protected_unchanged']}. Время этого запуска: {report['elapsed_seconds']:.1f} с."]
    return "\n".join(lines) + "\n"


def run(output=EXPERIMENT / "results/run_01"):
    torch.set_num_threads(2)
    protocols, osnet, baseline, signature, hashes, stage1, sample, difference = prepare()
    output, started = Path(output), time.perf_counter()
    if (output / "experiment.json").exists():
        _load(output / "experiment.json", signature)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new empty directory; prior experiments are immutable")
        output.mkdir(parents=True, exist_ok=True)
        _save(output / "experiment.json", {"signature": signature})
    if (output / "report.json").exists():
        report = _load(output / "report.json", signature)
        if any(sha256(output / p) != h for p, h in report["output_sha256"].items()):
            raise ValueError("Completed experiment output changed")
        print("Reuse completed stage 2; no inference, tuning or validation rerun", flush=True)
        return report
    q, g = protocols["calibration"]
    encoder, exported = export_checkpoint(CHECKPOINT, output / "weights", signature, q, signature["clip_train_identities"])
    clip = encode_cached(encoder, q + g, output / "cache/calibration.npz", signature)
    baseline_ranked, _ = control.ProtocolScores(q, g, osnet).evaluate(control.ACTIVE)
    control.assert_baseline(metrics(baseline_ranked, baseline["threshold"]), baseline["calibration"])
    calibration_path, frozen_path = output / "calibration.json", output / "frozen_selection.json"
    if frozen_path.exists():
        frozen = _load(frozen_path, signature)
        if frozen["calibration_sha256"] != sha256(calibration_path):
            raise ValueError("Frozen calibration changed")
    else:
        tuned = tune_calibration(q, g, osnet, clip)
        _save(calibration_path, {"signature": signature, **tuned})
        frozen = {"signature": signature, "calibration_sha256": sha256(calibration_path),
            **{k: tuned[k] for k in ("choices", "selection", "validation_used")},
            "baseline_threshold": baseline["threshold"], "osnet_neighbors": stage1["ranking"]}
        _save(frozen_path, frozen)
        print(f"FROZEN before validation: {frozen['choices']['fusion_selected']}", flush=True)
    # No hyperparameter or threshold fitting below this boundary.
    q, g = protocols["validation"]
    clip = encode_cached(encoder, q + g, output / "cache/validation.npz", signature)
    validation, engine = evaluate_frozen(q, g, osnet, clip, frozen)
    control.assert_baseline(validation["mvp"]["metrics"], baseline["validation"])
    for name, item in validation.items():
        alpha = item["config"].get("alpha", 1.)
        control.export_validation(output / "validation" / name, q, g, engine.embeddings(alpha), {"predictions": item})
    selected = frozen["choices"]["fusion_selected"]["config"]
    timings = {"extract": benchmark_extract(encoder, q, selected["alpha"])}
    timings["search"] = benchmark_search(engine, selected)
    masks = mask_check(encoder, output, signature, sample, frozen)
    if any(sha256(Path(p)) != h for p, h in signature["protected_sha256"].items()):
        raise RuntimeError("Protected weights, code, annotations or MVP artifacts changed")
    if any(sha256(DATASET / "images" / f"{i}.jpg") != h for i, h in hashes.items()):
        raise RuntimeError("Source images changed")
    report = {"signature": signature, "frozen": frozen, "validation": validation, "export": exported,
        "complementarity": _load(calibration_path, signature)["complementarity"], "benchmark": timings,
        "mask_audit": masks, "baseline_reproduced": True, "spot_check_max_abs_difference": difference,
        "protected_unchanged": True, "elapsed_seconds": time.perf_counter() - started,
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
