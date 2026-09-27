"""Stage 1: calibration-only score controls on frozen OSNet; never modifies the MVP."""
import argparse
import csv
import importlib.metadata
import platform
import time
from pathlib import Path

import numpy as np

import evaluate as official
from backend.core import ARTIFACTS, DATASET, MODEL, PREPROCESS, ROOT, Encoder, encode_rows, normalize, read_rows, sha256
from backend.evaluate import SEED, make_protocol
from backend.rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA, KReciprocalReranker
from backend.scoring import metrics, ranked_queries
from training.mask_calibration import _load, _save
from training.mask_finetune import load_json
from training.mask_reid_ablation import paired_deltas
from training.stage6 import audit_partitions

EXPERIMENT = ROOT / "OSNet-AIN-x1.0/audit_10_rerank_score_control"
MASK_AUDIT = ROOT / "OSNet-AIN-x1.0/audit_09_model_mask_comparison/results/comparison_v1"
CONFIDENCE_MODES = ("max_cosine", "selected_cosine", "ranking_score",
                    "selected_plus_margin", "selected_plus_support", "selected_plus_mutual")
ACTIVE = dict(name="active", kind="kreciprocal", pool=0, k1=20, k2=3,
              lambda_value=.5, support_weight=0., mutual_weight=0.)


def configurations():
    """Small predeclared grid, not a Cartesian hyperparameter search."""
    restricted = {**ACTIVE, "name": "top50", "pool": 50}
    variants = [ACTIVE.copy(), {**ACTIVE, "name": "cosine", "kind": "cosine"}, restricted]
    for key, values in (("k1", (10, 30)), ("k2", (1, 6)), ("lambda_value", (.35, .65)),
                        ("support_weight", (.1, .2)), ("mutual_weight", (.025, .05))):
        variants.extend({**restricted, "name": f"{key}_{value}", key: value} for value in values)
    return variants


class ScoreControl:
    """Only gallery vectors are retained; a query never sees any other query."""
    def __init__(self, gallery_vectors, k1=20, k2=3):
        self.graph = KReciprocalReranker(gallery_vectors, k1, k2)
        self.gallery = self.graph.gallery
        self.neighbors = np.stack([order[order != i][:3] for i, order in enumerate(self.graph.initial_rank)])

    def components(self, vector):
        cosine = np.clip(self.gallery @ normalize(vector), -1, 1)
        raw, jaccard = self.graph.components(vector)
        order = np.argsort(-cosine, kind="stable")
        mutual = np.zeros(len(cosine), dtype=np.float32)
        near = order[:self.graph.k1]
        mutual[near] = (1 - cosine[near] <= self.graph.reciprocal_cutoff[near]).astype(np.float32)
        return {"cosine": cosine, "raw": raw, "jaccard": jaccard, "mutual": mutual,
                "support": cosine[self.neighbors].mean(axis=1)}


def ranking_scores(parts, config):
    if config["kind"] == "cosine":
        scores = parts["cosine"].copy()
    else:
        scores = -((1 - config["lambda_value"]) * parts["jaccard"] + config["lambda_value"] * parts["raw"])
        scores += config["support_weight"] * (parts["support"] - 1) / 2
        scores += config["mutual_weight"] * parts["mutual"]
    if config["pool"]:
        pool = np.argsort(-parts["cosine"], kind="stable")[:config["pool"]]
        restricted = np.full_like(scores, float(scores.min()) - 1)
        restricted[pool] = scores[pool]
        scores = restricted
    return scores


def confidence_scores(parts, scores, ranked):
    winner = np.argsort(-scores, axis=1, kind="stable")[:, 0]
    row = np.arange(len(winner))
    chosen = parts["cosine"][row, winner]
    # The two closest objects can be the SAME vehicle; margin is only an experimental feature.
    ordered = np.sort(parts["cosine"], axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return {"max_cosine": ranked.confidence.copy(), "selected_cosine": chosen,
        "ranking_score": scores[row, winner], "selected_plus_margin": chosen + .1 * margin,
        "selected_plus_support": .9 * chosen + .1 * parts["support"][row, winner],
        "selected_plus_mutual": chosen + .05 * parts["mutual"][row, winner]}


class ProtocolScores:
    """Offline cache of independently processed queries, never a cross-query graph."""
    def __init__(self, query, gallery, embeddings):
        self.query, self.gallery, self.embeddings = query, gallery, embeddings
        self.parts = {}
        self.timings = {}

    def evaluate(self, config):
        key = (config["k1"], config["k2"])
        if key not in self.parts:
            start = time.perf_counter()
            engine = ScoreControl(np.stack([self.embeddings[r["image_id"]] for r in self.gallery]), *key)
            build = time.perf_counter() - start
            values, times = [], []
            for row in self.query:
                start = time.perf_counter()
                values.append(engine.components(self.embeddings[row["image_id"]]))
                times.append((time.perf_counter() - start) * 1000)
            self.parts[key] = {name: np.stack([p[name] for p in values]) for name in values[0]}
            self.timings[key] = {"graph_seconds": build, "components_mean_ms": float(np.mean(times)),
                                 "components_p95_ms": float(np.percentile(times, 95))}
        parts = self.parts[key]
        scores = np.stack([ranking_scores({k: v[i] for k, v in parts.items()}, config) for i in range(len(self.query))])
        ranked = ranked_queries(self.query, self.gallery, self.embeddings, scores)
        return ranked, confidence_scores(parts, scores, ranked)


def acceptance_labels(ranked):
    known = np.asarray([official.valid_positives(row, ranked.gallery) > 0 for _, row in ranked.query.iterrows()])
    identity = ranked.gallery.vehicle_id.to_dict()
    # Match the organizer: candidate correctness checks identity, not the ranking junk rule.
    correct = known & np.asarray([identity[ranked.predictions[qid][0]] == row.vehicle_id
                                 for qid, row in ranked.query.iterrows()])
    return known, correct


def choose_threshold(ranked, confidence):
    """Vectorized official query-level counts; verify the selected point with official metrics."""
    confidence = np.asarray(confidence, dtype=float)
    if confidence.shape != (len(ranked.query),) or not np.isfinite(confidence).all():
        raise ValueError("Need one finite confidence for every calibration query")
    known, correct = acceptance_labels(ranked)
    if known.all() or not known.any():
        raise ValueError("Refusal calibration needs both known and unknown queries")
    values = np.unique(confidence)
    thresholds = np.append(values, np.nextafter(values[-1], np.inf))
    accepted = confidence[:, None] >= thresholds
    tp = (accepted & correct[:, None]).sum(axis=0)
    fp = (accepted & ~correct[:, None]).sum(axis=0)
    fn = (~accepted & known[:, None]).sum(axis=0)
    tn = (~accepted & ~known[:, None]).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=tp + fn > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)
    tnr = tn / (~known).sum()
    quality = .7 * f1 + .3 * tnr
    best = int(np.lexsort((thresholds, f1, quality))[-1])
    selected = metrics(ranked, float(thresholds[best]), confidence)
    if any(selected[key] != int(value[best]) for key, value in (("TP", tp), ("FP", fp), ("FN", fn), ("TN", tn))):
        raise AssertionError("Vectorized threshold selection differs from the organizer evaluator")
    return {"threshold": float(thresholds[best]), "metrics": selected,
            "curve": {"threshold": thresholds.tolist(), "candidate_F1": f1.tolist(), "TNR": tnr.tolist(),
                      "candidate_score": quality.tolist()}}


def select_confidence(ranked, values):
    trials = [{"mode": mode, **choose_threshold(ranked, values[mode])} for mode in CONFIDENCE_MODES]
    # First mode wins exact ties: prefer the original, simplest score.
    best = max(trials, key=lambda t: (t["metrics"]["candidate_score"], t["metrics"]["candidate_F1"]))
    return {"mode": best["mode"], "threshold": best["threshold"]}, trials


def tune_calibration(query, gallery, embeddings, baseline_threshold):
    """The selector has no validation argument and cannot inspect validation results."""
    engine = ProtocolScores(query, gallery, embeddings)
    leaderboard = []
    for config in configurations():
        ranked, _ = engine.evaluate(config)
        score = metrics(ranked, baseline_threshold)
        leaderboard.append({"config": config, "metrics_at_baseline_threshold": score})
        print(f"Calibration {config['name']}: mAP@10={score['mAP_at_10']:.6f}", flush=True)
    best = max(leaderboard, key=lambda row: (row["metrics_at_baseline_threshold"]["mAP_at_10"],
                                            row["metrics_at_baseline_threshold"]["Rank_1"]))["config"]
    choices, confidence_trials = {}, {}
    for label, config in (("active", ACTIVE), ("selected", best)):
        ranked, values = engine.evaluate(config)
        choices[label], confidence_trials[label] = select_confidence(ranked, values)
    return {"ranking": best, "confidence": choices, "leaderboard": leaderboard,
            "confidence_trials": confidence_trials,
            "selection": "ranking: max calibration mAP@10, Rank-1, grid order; refusal: max .7*F1+.3*TNR, F1, mode order",
            "baseline_threshold": baseline_threshold, "validation_used": False}


def assert_baseline(actual, expected):
    for key in ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "TP", "FP", "FN", "TN"):
        if not np.isclose(actual[key], expected[key], rtol=0, atol=1e-10):
            raise ValueError(f"Active baseline reproduction failed: {key}")


def query_details(ranked, confidence, threshold):
    known, correct = acceptance_labels(ranked)
    details = {}
    for n, qid in enumerate(ranked.query.index):
        one = official.ranking_metrics(ranked.query.loc[[qid]], ranked.gallery, ranked.predictions)
        accepted = bool(confidence[n] >= threshold)
        outcome = ("TP" if correct[n] else "FP") if accepted else ("FN" if known[n] else "TN")
        details[qid] = {"known": bool(known[n]), "AP_at_10": one["mAP@10"] if known[n] else None,
            "top1_correct": bool(one["Rank-1"]) if known[n] else False, "accepted": accepted,
            "confidence": float(confidence[n]), "candidate_outcome": outcome, "top10": ranked.predictions[qid]}
    return details


def evaluate_frozen(query, gallery, embeddings, frozen):
    engine = ProtocolScores(query, gallery, embeddings)
    choices = {
        "baseline": (ACTIVE, {"mode": "max_cosine", "threshold": frozen["baseline_threshold"]}),
        "ranking_only": (frozen["ranking"], {"mode": "max_cosine", "threshold": frozen["baseline_threshold"]}),
        "refusal_only": (ACTIVE, frozen["confidence"]["active"]),
        "combined": (frozen["ranking"], frozen["confidence"]["selected"])}
    results = {}
    for name, (config, choice) in choices.items():
        ranked, values = engine.evaluate(config)
        confidence = values[choice["mode"]]
        results[name] = {"ranking": config, "refusal": choice,
            "metrics": metrics(ranked, choice["threshold"], confidence),
            "per_query": query_details(ranked, confidence, choice["threshold"])}
    base = results["baseline"]
    for name, item in results.items():
        if name == "baseline":
            continue
        item["paired_vs_baseline"] = paired_deltas(base["per_query"], item["per_query"])
        item["decision_changes"] = [{"query_id": i, "before": base["per_query"][i]["candidate_outcome"],
            "after": p["candidate_outcome"]} for i, p in item["per_query"].items()
            if base["per_query"][i]["candidate_outcome"] != p["candidate_outcome"]]
    return results


def benchmark(query, gallery, embeddings, config, mode, repeats=3):
    start = time.perf_counter()
    engine = ScoreControl(np.stack([embeddings[r["image_id"]] for r in gallery]), config["k1"], config["k2"])
    build = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        for row in query:
            start = time.perf_counter()
            parts = engine.components(embeddings[row["image_id"]])
            scores = ranking_scores(parts, config)
            order = np.argsort(-scores, kind="stable")[:10]
            # Include the selected confidence formula, with no labels or other queries.
            class Raw:
                confidence = np.asarray([parts["cosine"].max()])
            confidence_scores({k: v[None] for k, v in parts.items()}, scores[None], Raw())[mode]
            assert len(order) == 10
            samples.append((time.perf_counter() - start) * 1000)
    return {"gallery_graph_seconds": build, "samples": len(samples),
            "median_query_ms": float(np.median(samples)), "p95_query_ms": float(np.percentile(samples, 95)),
            "scope": "CPU post-embedding score construction + top10 + confidence; excludes extract and metric calculation",
            "platform": platform.platform(), "not_organizer_A5000_benchmark": True}


def prepare():
    encoder = Encoder()
    split_path, baseline_path = ARTIFACTS / "splits.json", ARTIFACTS / "baseline_metrics.json"
    split, baseline = load_json(split_path), load_json(baseline_path)
    if (split["train_csv_sha256"] != sha256(DATASET / "train.csv")
            or baseline["model_sha256"] != encoder.model_sha256
            or baseline["encoder_fingerprint"] != encoder.fingerprint
            or baseline["preprocessing"] != PREPROCESS
            or baseline["evaluator"]["sha256"] != sha256(ROOT / "evaluate.py")
            or (ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA) != (20, 3, .5)
            or any(baseline["search"][k] != v for k, v in (("k1", 20), ("k2", 3), ("lambda", .5)))):
        raise ValueError("Frozen model, dataset, preprocessing, evaluator or MVP configuration changed")
    rows = read_rows(DATASET / "train.csv")
    print("Verify frozen frame hashes and identity/frame separation", flush=True)
    hashes = {r["image_id"]: sha256(DATASET / "images" / f"{r['image_id']}.jpg") for r in rows}
    if hashes != split["frame_sha256"]:
        raise ValueError("Source frames changed since the frozen split")
    audit_partitions(rows, hashes, split["identities"])
    protocols = {name: make_protocol(rows, split["identities"][name], SEED) for name in ("calibration", "validation")}
    for name, (q, g) in protocols.items():
        if split["protocols"][name] != {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g]}:
            raise ValueError("Ordered protocol differs from the frozen MVP protocol")
    cache_path = ARTIFACTS / f"stage2_embeddings_{encoder.model_sha256[:12]}.npz"
    selected = [r for q, g in protocols.values() for r in q + g]
    with np.load(cache_path, allow_pickle=False) as cache:
        ids, vectors = cache["ids"].tolist(), cache["baseline"].copy()
    if (ids != [r["image_id"] for r in selected] or len(set(ids)) != len(ids)
            or vectors.shape != (len(ids), 512) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError("Existing OSNet cache is incomplete, reordered or invalid")
    embeddings = dict(zip(ids, vectors))
    # The historical cache predates hash manifests: check fixed samples by fresh ONNX inference too.
    sample = [group[int(i)] for qg in protocols.values() for group in qg
              for i in np.linspace(0, len(group) - 1, 4, dtype=int)]
    fresh = encode_rows(encoder, sample)
    difference = float(np.max(np.abs(fresh - np.stack([embeddings[r["image_id"]] for r in sample]))))
    if difference > 1e-4:
        raise ValueError("Historical feature cache failed fresh ONNX spot-check")
    sources = [Path(__file__), ROOT / "backend/core.py", ROOT / "backend/rerank.py", ROOT / "backend/scoring.py",
               ROOT / "backend/evaluate.py", ROOT / "evaluate.py", ROOT / "training/mask_reid_ablation.py"]
    protected_paths = [MODEL, split_path, baseline_path, cache_path, DATASET / "train.csv", *sources,
                       *ARTIFACTS.glob("*.npy"), *ARTIFACTS.glob("*.sqlite3"), *ARTIFACTS.glob("*.csv"),
                       ARTIFACTS / "export_manifest.json", MASK_AUDIT / "comparison.json", MASK_AUDIT / "embeddings.npz"]
    protected = {str(p): sha256(p) for p in protected_paths if p.is_file()}
    signature = {"version": 1, "protected_sha256": protected, "protocols": split["protocols"],
        "preprocess": PREPROCESS, "model_sha256": encoder.model_sha256, "weights_bytes": MODEL.stat().st_size,
        "grid": configurations(), "confidence_modes": list(CONFIDENCE_MODES),
        "confidence_constants": {"margin_weight": .1, "support_weight": .1, "mutual_weight": .05, "neighbors": 3},
        "selection": "calibration only; mAP then Rank-1 for ranking; .7*F1+.3*TNR then F1 for refusal",
        "spot_check_count": len(sample), "runtime": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "onnxruntime")}}
    return protocols, embeddings, baseline, signature, hashes, difference


def export_validation(output, query, gallery, embeddings, results):
    """Separate local-validation artifacts, never organizer test submissions or the MVP index."""
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("query", query), ("gallery", gallery)):
        with (output / f"test_{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "x", "y", "w", "h"])
            writer.writeheader()
            writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    with (output / "ground_truth.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "vehicle_id", "camera_id", "split"])
        writer.writerows([r["image_id"], r["vehicle_id"], r["camera_id"], name]
                         for name, rows in (("query", query), ("gallery", gallery)) for r in rows)
    np.save(output / "embeddings.npy", np.stack([embeddings[r["image_id"]] for r in query + gallery]))
    for name, result in results.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        with (folder / "submission.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([r["image_id"], *result["per_query"][r["image_id"]]["top10"]] for r in query)
        with (folder / "candidates.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_id", "gallery_id", "confidence"])
            writer.writerows([i, d["top10"][0], d["confidence"]] for i, d in result["per_query"].items() if d["accepted"])
        # Read files back through the actual organizer parsers, not just in-memory adapters.
        import pandas as pd
        q, g = pd.DataFrame(query).set_index("image_id"), pd.DataFrame(gallery).set_index("image_id")
        ranking = official.ranking_metrics(q, g, official.load_submission(folder / "submission.csv", set(g.index)))
        candidates = official.candidate_metrics(q, g, official.load_candidates(folder / "candidates.csv"))
        if not np.isclose(ranking["mAP@10"], result["metrics"]["mAP_at_10"], atol=1e-12):
            raise AssertionError("Exported ranking differs from evaluation")
        for key in ("TP", "FP", "FN", "TN"):
            if candidates[key] != result["metrics"][key]:
                raise AssertionError("Exported candidates differ from evaluation")


def mask_check(frozen, model_hash):
    audit = _load(MASK_AUDIT / "comparison.json")
    signature = audit["signature"]
    if (signature["source_sha256"]["mvp_weights"] != model_hash
            or sha256(MASK_AUDIT / "embeddings.npz") != audit["output_sha256"]["embeddings.npz"]):
        raise ValueError("Manual audit embeddings belong to different weights or changed")
    q, g = signature["query"], signature["gallery"]
    results = {}
    with np.load(MASK_AUDIT / "embeddings.npz", allow_pickle=False) as cache:
        if cache["ids"].tolist() != [r["image_id"] for r in q + g]:
            raise ValueError("Manual audit ordering changed")
        for condition in ("original", "manual"):
            embeddings = dict(zip(cache["ids"].tolist(), cache[f"mvp__{condition}"]))
            values = evaluate_frozen(q, g, embeddings, frozen)
            results[condition] = {name: value["metrics"] for name, value in values.items()}
    return {"queries": len(q), "gallery": len(g), "manual_source": "masks-7.json", "scores": results,
            "selection_used": False, "warning": "Small reused diagnostic audit; not proof of plate independence"}


def report_markdown(report):
    frozen = report["frozen"]
    lines = ["# Этап 1: OSNet, реранкинг и отказ", "",
        "Без нового обучения, CLIP, TTA или автоматических масок. MVP не изменён.", "",
        f"Calibration: {report['counts']['calibration']}; validation: {report['counts']['validation']} (query/gallery).",
        "13 заранее заданных вариантов ранжирования; 6 функций уверенности. "
        "Все параметры выбраны только на calibration и сохранены до оценки validation.", "",
        f"Выбранный порядок: **{frozen['ranking']['name']}**. Отказ для него: "
        f"**{frozen['confidence']['selected']['mode']}**, порог **{frozen['confidence']['selected']['threshold']:.10f}**.", "",
        "## Validation", "",
        "| Вариант | mAP@10, % | Rank-1, % | Rank-5, % | F1, % | TNR, % | 0.7F1+0.3TNR, % | TP/FP/FN/TN |",
        "|---|---:|---:|---:|---:|---:|---:|---|" ]
    for name, item in report["validation"].items():
        m = item["metrics"]
        cells = [f"{m[k]*100:.4f}" for k in ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "candidate_score")]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {m['TP']}/{m['FP']}/{m['FN']}/{m['TN']} |")
    lines += ["", "baseline — действующий MVP; ranking_only — новый порядок со старым отказом; "
              "refusal_only — старый порядок с новой оценкой отказа; combined — оба изменения.", ""]
    for name in ("ranking_only", "refusal_only", "combined"):
        d = report["validation"][name]["paired_vs_baseline"]
        lo, hi = d["paired_bootstrap_95pct"]
        lines.append(f"- {name}: Δ mAP {d['mAP_at_10_delta']*100:+.4f} п.п.; "
                     f"парный 95% интервал [{lo*100:+.4f}; {hi*100:+.4f}] п.п.")
    lines += ["", "## Стоимость", ""]
    for name, b in report["benchmark"].items():
        lines.append(f"- {name}: граф {b['gallery_graph_seconds']:.3f} с; один query после эмбеддинга "
                     f"median {b['median_query_ms']:.3f} мс, p95 {b['p95_query_ms']:.3f} мс.")
    lines += [f"- Единственные веса инференса: исходный OSNet, {report['signature']['weights_bytes']/1024**2:.2f} MiB.",
        "- Это локальный CPU-замер поиска, не официальный extract-бенчмарк RTX A5000.", "",
        "## Ручной масочный аудит (без подбора параметров)", "",
        "| Вход | Вариант | mAP@10, % | F1, % | TNR, % |", "|---|---|---:|---:|---:|"]
    for condition, scores in report["mask_audit"]["scores"].items():
        for name in ("baseline", "combined"):
            m = scores[name]
            lines.append(f"| {condition} | {name} | {m['mAP_at_10']*100:.4f} | {m['candidate_F1']*100:.2f} | {m['TNR']*100:.2f} |")
    lines += ["", "## Ограничения", "",
        "- Validation уже использовалась в прошлых экспериментах и выборе OSNet; это не независимый финальный тест.",
        "- Нельзя выбирать новый вариант или менять сетку после просмотра validation этого запуска.",
        "- Bootstrap: 2000 парных перевыборок query при фиксированной gallery, локальная диагностика.",
        "- Обучения нейросети и дополнительной обучаемой головы нет; подбор коэффициентов и порогов на calibration есть.",
        "- Оценки уверенности не являются вероятностями. PR-AUC сохранён по неизменённому evaluator с его цензурированием отказов.",
        "- Для решений используются только векторы текущего query и статичной gallery. vehicle_id/camera_id — только для оценки.",
        "- Малая ручная выборка не доказывает независимости от номера. Разрыв top1/top2 не означает разных автомобилей.",
        "- Никакого автоматического внедрения победителя: отдельно оцениваются mAP, F1, TNR и стоимость.", "",
        "Файлы: calibration.json — вся сетка; frozen_selection.json — зафиксированный выбор; "
        "report.json — метрики, парные изменения и top10/confidence каждого query; "
        "validation/ — отдельные локальные CSV и embeddings.npy (не официальный test и не файлы MVP).", "",
        f"Неизменность защищённых файлов проверена: {report['protected_unchanged']}. "
        f"Время основного эксперимента: {report['elapsed_seconds']:.1f} с."]
    return "\n".join(lines) + "\n"


def run(output=EXPERIMENT / "results/run_01"):
    protocols, embeddings, baseline, signature, hashes, difference = prepare()
    output = Path(output)
    if (output / "experiment.json").exists():
        _load(output / "experiment.json", signature)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new empty results directory; existing experiments are immutable")
        output.mkdir(parents=True, exist_ok=True)
        _save(output / "experiment.json", {"signature": signature})
    if (output / "report.json").exists():
        report = _load(output / "report.json", signature)
        if any(sha256(output / p) != h for p, h in report["output_sha256"].items()):
            raise ValueError("Completed experiment output changed")
        print("Reuse completed stage 1; no tuning or validation rerun", flush=True)
        return report
    start = time.perf_counter()
    calibration_path, frozen_path = output / "calibration.json", output / "frozen_selection.json"
    if frozen_path.exists():
        frozen = _load(frozen_path, signature)
        if frozen["calibration_sha256"] != sha256(calibration_path):
            raise ValueError("Frozen calibration report changed")
    else:
        q, g = protocols["calibration"]
        tuned = tune_calibration(q, g, {r["image_id"]: embeddings[r["image_id"]] for r in q + g}, baseline["threshold"])
        assert_baseline(tuned["leaderboard"][0]["metrics_at_baseline_threshold"], baseline["calibration"])
        _save(calibration_path, {"signature": signature, **tuned})
        frozen = {"signature": signature, "calibration_sha256": sha256(calibration_path),
                  **{k: tuned[k] for k in ("ranking", "confidence", "baseline_threshold", "selection", "validation_used")}}
        _save(frozen_path, frozen)
        print(f"FROZEN on calibration: {frozen['ranking']['name']}; {frozen['confidence']}", flush=True)
    # No grid or threshold fitting is reachable below this boundary.
    q, g = protocols["validation"]
    validation = evaluate_frozen(q, g, embeddings, frozen)
    assert_baseline(validation["baseline"]["metrics"], baseline["validation"])
    timings = {name: benchmark(q, g, embeddings, config, mode) for name, config, mode in (
        ("baseline", ACTIVE, "max_cosine"),
        ("combined", frozen["ranking"], frozen["confidence"]["selected"]["mode"]))}
    masks = mask_check(frozen, signature["model_sha256"])
    export_validation(output / "validation", q, g, embeddings, validation)
    if any(sha256(Path(p)) != h for p, h in signature["protected_sha256"].items()):
        raise RuntimeError("Protected model, source, cache or MVP artifacts changed")
    if any(sha256(DATASET / "images" / f"{i}.jpg") != h for i, h in hashes.items()):
        raise RuntimeError("Dataset frames changed during the experiment")
    report = {"signature": signature, "frozen": frozen, "validation": validation, "benchmark": timings,
        "mask_audit": masks, "spot_check_max_abs_difference": difference, "protected_unchanged": True,
        "counts": {name: [len(q), len(g)] for name, (q, g) in protocols.items()},
        "baseline_reproduced": {"calibration": True, "validation": True},
        "elapsed_seconds": time.perf_counter() - start}
    (output / "RESULTS.md").write_text(report_markdown(report), encoding="utf-8")
    report["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    _save(output / "report.json", report)
    print(f"Saved {output / 'RESULTS.md'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results/run_01")
    run(parser.parse_args().output)
