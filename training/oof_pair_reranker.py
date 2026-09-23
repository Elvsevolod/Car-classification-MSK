"""Three identity/frame-disjoint OSNet folds; scalar OOF head, unchanged MVP/refusal."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from backend.core import ARTIFACTS, DATASET, MODEL, ROOT, STOCK_MODEL, Encoder, normalize, preprocess, read_rows, sha256
from backend.evaluate import SEED, make_protocol
from backend.scoring import metrics
from training import pair_reranker_experiment as previous
from training import rerank_score_control as control
from training.audit import digest, load_crop
from training.hpo import (ExperimentConfig, ReIDExperimentModel, initialize_experiment, make_optimizer,
                          prepare_experiment, set_epoch_learning_rates, train_epoch)
from training.mask_calibration import _load, _save
from training.mask_finetune import load_json
from training.mask_model_comparison import AuditEncoder
from training.mask_reid_ablation import paired_deltas
from training.pair_reranker import (HEADS, SCALARS, HeadEncoder, PairSet, encode_cached, export_head, fit_head)
from training.pipeline import format_duration, select_device, set_seed
from training.stage6 import _cpu_state, _save_checkpoint, audit_partitions

EXPERIMENT = ROOT / "OSNet-AIN-x1.0/variant_13_oof_pair_reranker"
RECIPE = ROOT / "OSNet-AIN-x1.0/variant_02_hpo_bnneck_supcon/results/selected_run_02/summary.json"
PREVIOUS = previous.EXPERIMENT / "results/run_01"
FOLDS, ENCODER_EPOCHS, HEAD_EPOCHS = 3, 5, 7
HEAD_CONFIG = dict(HEADS[0])
BETAS = (0., .1, .25, .5, 1.)


def make_folds(rows, identities, hashes, count=FOLDS, seed=SEED):
    """Assign whole connected components, including transitive shared full frames."""
    parent = {i: i for i in sorted(identities)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    frames = {}
    selected = [r for r in rows if r["vehicle_id"] in parent]
    for r in selected:
        i, frame = r["vehicle_id"], hashes[r["image_id"]]
        if frame in frames:
            parent[find(i)] = find(frames[frame])
        frames[frame] = i
    groups = defaultdict(list)
    for i in parent:
        groups[find(i)].append(i)
    groups = list(groups.values())
    if count < 2 or len(groups) < count:
        raise ValueError("Not enough independent identity/frame groups for OOF")
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)  # Stable, seeded ties; balance identities, not frame count.
    buckets = [[] for _ in range(count)]
    for group in groups:
        buckets[min(range(count), key=lambda n: (len(buckets[n]), n))].extend(group)
    parts = {f"fold_{n+1:02d}": sorted(ids) for n, ids in enumerate(buckets)}
    audit_partitions(selected, hashes, parts)
    return {name: {"train": sorted(set(identities)-set(ids)), "held_out": ids} for name, ids in parts.items()}


def prepare(device=None):
    """Read-only preflight: no gradient updates, result writes or downloads."""
    protocols, vectors, baseline, old_signature, hashes, _, sample, difference = previous.prepare()
    old = _load(PREVIOUS / "report.json", old_signature)
    old_frozen = _load(PREVIOUS / "frozen_selection.json", old_signature)
    if (old["inner"]["winner"]["config"] != HEAD_CONFIG
            or old["inner"]["winner"]["selected_epoch"] != HEAD_EPOCHS
            or old["frozen"] != old_frozen):
        raise ValueError("Expected the frozen scalar/7-epoch stage-3 recipe")
    for name in ("weights/final/pair_head.onnx", "weights/final/export.json", "frozen_selection.json"):
        if sha256(PREVIOUS / name) != old["output_sha256"][name]:
            raise ValueError(f"Previous experiment artifact changed: {name}")
    source = load_json(RECIPE)
    config = ExperimentConfig(**source["config"])
    config.validate()
    if (source["onnx_sha256"] != sha256(MODEL) or source["checkpoint_epoch"] != ENCODER_EPOCHS
            or config.epochs != 30 or config.num_workers != 0 or config.hard_negative_sampling
            or config.resize_mode != "square" or config.pooling != "avg" or not config.use_bnneck
            or config.use_mixstyle or config.loss_weight_schedule != "constant"):
        raise ValueError("OSNet source recipe no longer matches this fixed OOF experiment")
    rows = read_rows(DATASET / "train.csv")
    split = load_json(ARTIFACTS / "splits.json")["identities"]
    folds = make_folds(rows, split["train"], hashes)
    fold_protocols = {name: make_protocol(rows, ids["held_out"], SEED) for name, ids in folds.items()}
    for name, (q, g) in fold_protocols.items():
        if len(q) != len(folds[name]["held_out"]) or len(g) < 50 or len(folds[name]["train"]) < config.identities_per_batch:
            raise ValueError(f"Insufficient queries/gallery/training identities in {name}")
    device = select_device() if device is None else torch.device(device)
    if device.type not in ("cpu", "mps", "cuda"):
        raise ValueError("Use cpu, mps or cuda")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available to this Python kernel")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available to this Python kernel")
    protected = dict(old_signature["protected_sha256"])
    for p in (RECIPE, PREVIOUS / "report.json", PREVIOUS / "frozen_selection.json",
              PREVIOUS / "weights/final/pair_head.onnx", PREVIOUS / "weights/final/export.json", Path(__file__)):
        protected[str(p)] = sha256(p)
    signature = {"version": 1, "protected_sha256": protected, "data_sha256": old_signature["data_sha256"],
        "outer": split, "folds": folds, "seed": SEED, "config": asdict(config),
        "encoder_epochs": ENCODER_EPOCHS, "encoder_schedule_horizon": config.epochs,
        "head": HEAD_CONFIG, "head_epochs": HEAD_EPOCHS, "features": list(SCALARS), "betas": list(BETAS),
        "baseline": old_frozen["baseline"], "baseline_threshold": baseline["threshold"],
        "previous_beta": old_frozen["beta"], "previous_head_sha256": old["head_export"]["sha256"],
        "protocols": {name: {"query": q, "gallery": g} for name, (q, g) in fold_protocols.items()},
        "refusal_retuned": False, "device": str(device), "torch_threads": 2,
        "runtime": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "torchvision", "Pillow", "onnxruntime")}}
    plan = {"device": str(device), "outer_identities": {k: len(v) for k, v in split.items()},
        "folds": {name: {"train_ids": len(ids["train"]), "held_out_ids": len(ids["held_out"]),
            "queries": len(fold_protocols[name][0]), "gallery": len(fold_protocols[name][1]),
            "batches_per_epoch": len(ids["train"])//config.identities_per_batch} for name, ids in folds.items()},
        "encoder_epochs_per_fold": ENCODER_EPOCHS, "lr_horizon_epochs": config.epochs,
        "head_epochs": HEAD_EPOCHS, "free_disk_gib": round(shutil.disk_usage(ROOT).free/1024**3, 2),
        "backbone_initializer": str(STOCK_MODEL), "mvp_used_as_initializer": False,
        "training_started": False, "mvp_spot_check_max_abs_difference": difference}
    return dict(rows=rows, hashes=hashes, protocols=protocols, fold_protocols=fold_protocols,
                vectors=vectors, baseline=baseline, sample=sample, signature=signature, plan=plan)


def fold_signature(context, name):
    s = context["signature"]
    return {"experiment": digest(s), "name": name, **s["folds"][name], "device": s["device"],
            "config": s["config"], "epochs": s["encoder_epochs"], "initializer_sha256": sha256(STOCK_MODEL)}


def fit_fold(rows, train_ids, config, epochs, device, output, signature, remaining_folds=0):
    """Fixed epochs only. Held-out identities/metrics are never passed to the trainer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last_path, final_path, summary_path = output / "last.pt", output / "final.pt", output / "training.json"
    if summary_path.exists():
        summary = _load(summary_path, signature)
        if any(sha256(output / p) != h for p, h in summary["output_sha256"].items()):
            raise ValueError("Completed fold weights/history changed")
        print(f"Reuse completed {output.name}: {epochs} epochs", flush=True)
        return summary
    if not 1 <= epochs <= config.epochs or config.num_workers != 0:
        raise ValueError("Fixed epochs must fit the unchanged LR horizon; use num_workers=0")
    if not last_path.exists() and (final_path.exists() or (output / "history.json").exists()):
        raise ValueError("Missing last.pt; use a new run directory, do not delete history")
    # Restrict rows before constructing the dataset; no held-out row reaches the trainer.
    ids = set(train_ids)
    selected = [r for r in rows if r["vehicle_id"] in ids]
    if {r["vehicle_id"] for r in selected} != ids:
        raise ValueError("Missing fold training identities")
    set_seed(config.seed)
    model, _ = initialize_experiment(len(ids), config, device, onnx_path=STOCK_MODEL)
    optimizer = make_optimizer(model, config)
    _, _, sampler, loader = prepare_experiment(selected, train_ids, config)
    history = []
    try:
        if last_path.exists():
            state = torch.load(last_path, map_location="cpu", weights_only=True)
            if state["signature"] != signature:
                raise ValueError("Interrupted fold settings/data/device changed; use a new run directory")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
            if [r["epoch"] for r in history] != list(range(1, len(history)+1)) or len(history) > epochs:
                raise ValueError("Invalid checkpoint epoch history")
            del state
        for epoch in range(len(history), epochs):
            set_seed(config.seed + epoch)  # Replay only the interrupted epoch, including augmentations.
            started = time.perf_counter()
            lr = set_epoch_learning_rates(optimizer, config, epoch, config.epochs)
            train = train_epoch(model, loader, sampler, optimizer, device, config, epoch)
            if not all(np.isfinite(v) for v in train.values()):
                raise FloatingPointError("Non-finite OSNet training loss; checkpoint not advanced")
            seconds = time.perf_counter()-started
            history.append(dict(epoch=epoch+1, train=train, lr=lr, epoch_seconds=seconds))
            _save_checkpoint(last_path, {"signature": signature, "model": _cpu_state(model),
                "optimizer": optimizer.state_dict(), "history": history})
            _save(output / "history.json", {"signature": signature, "history": history})
            mean = sum(r["epoch_seconds"] for r in history)/len(history)
            print(f"{output.name} | epoch {epoch+1}/{epochs}, remaining {epochs-epoch-1} | "
                  f"epoch {format_duration(seconds)} | fold ETA {format_duration(mean*(epochs-epoch-1))} | "
                  f"all encoders ETA ~{format_duration(mean*(epochs-epoch-1+remaining_folds*epochs))} | "
                  f"loss {train['loss']:.5f} | no held-out evaluation", flush=True)
        _save_checkpoint(final_path, {"signature": signature, "model": _cpu_state(model),
                                     "config": asdict(config), "epoch": epochs})
        _save(output / "history.json", {"signature": signature, "history": history})
        summary = {"signature": signature, "epochs": epochs, "train_ids": sorted(ids), "train_images": len(selected),
            "batches_per_epoch": len(loader), "gradient_steps": len(loader)*epochs, "history": history,
            "held_out_used_for_selection": False,
            "output_sha256": {p.name: sha256(p) for p in (last_path, final_path, output / "history.json")}}
        _save(summary_path, summary)
        return summary
    finally:
        del model, optimizer, loader
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()


def export_fold(output, signature, rows):
    output = Path(output)
    path, manifest = output / "encoder.onnx", output / "export.json"
    checkpoint = output / "final.pt"
    key = {"fold": signature, "checkpoint_sha256": sha256(checkpoint)}
    if manifest.exists():
        report = _load(manifest, key)
        return AuditEncoder(path, report["sha256"]), report
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["signature"] != signature or state["epoch"] != signature["epochs"]:
        raise ValueError("Fold checkpoint does not match its frozen training protocol")
    config = ExperimentConfig(**state["config"])
    model = ReIDExperimentModel(len(signature["train"]), config.use_bnneck, config.pooling, config.resize_mode).eval()
    model.load_state_dict(state["model"], strict=True)
    inference = model.inference_module().eval()
    crops = [load_crop(r) for r in rows[:2]]
    samples = np.stack([preprocess(c, (0, 0, *c.size)) for c in crops])
    temporary = output / "encoder.partial.onnx"
    with torch.no_grad():
        torch.onnx.export(inference, torch.from_numpy(samples[:1]), str(temporary), dynamo=False, opset_version=17,
            input_names=["images"], output_names=["output"], dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}})
    checksum = sha256(temporary)
    encoder, errors = AuditEncoder(temporary, checksum), []
    for size in (1, 2):
        with torch.no_grad():
            expected = normalize(inference(torch.from_numpy(samples[:size])).numpy())
        actual = encoder.encode_batch(samples[:size])
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        errors.append(float(np.abs(actual-expected).max()))
    temporary.replace(path)
    report = {"signature": key, "sha256": checksum, "bytes": path.stat().st_size,
              "max_absolute_error": max(errors), "batches_checked": [1, 2], "training_only_encoder": True}
    _save(manifest, report)
    return encoder, report


@dataclass
class ScalarTraining:
    query: list
    features: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    valid: np.ndarray


def scalar_training(pairs, hashes):
    targets, weights, valid = pairs.labels(hashes)
    # Never pool per-coordinate features from independently fitted encoders.
    return ScalarTraining(pairs.query, pairs.features[..., :len(SCALARS)].copy(), targets, weights, valid)


def pool_training(parts, expected_ids):
    ids = [r["vehicle_id"] for part in parts for r in part.query]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("OOF must contain exactly one held-out query per development identity")
    if any(p.features.shape != (len(p.query), 50, len(SCALARS)) for p in parts):
        raise ValueError("Only eight scalar top50 features may be pooled across folds")
    return ScalarTraining([r for p in parts for r in p.query],
                          *(np.concatenate([getattr(p, key) for p in parts])
                            for key in ("features", "targets", "weights", "valid")))


def pair_diagnostics(pairs, training):
    pos = (training.targets == 1) & training.valid
    known, _ = control.acceptance_labels(pairs.raw)
    return {"queries": len(pairs.query), "gallery": len(pairs.gallery), "known": int(known.sum()),
        "positive_pairs": int(pos.sum()), "valid_pairs": int(training.valid.sum()),
        "known_queries_with_positive_in_top50": int((pos.any(1) & known).sum()),
        "known_hit_at_50": float(pos.any(1)[known].mean()), "raw_mAP_at_10": pairs.raw.ranking["mAP@10"],
        "scalar_mean": training.features[training.valid].mean(0).astype(float).tolist(),
        "scalar_std": training.features[training.valid].std(0).astype(float).tolist(), "used_for_selection": False}


def collect_training(context, output):
    s, rows, hashes = context["signature"], context["rows"], context["hashes"]
    device, config = torch.device(s["device"]), ExperimentConfig(**s["config"])
    parts, matched, reports = [], [], {}
    mvp = Encoder()
    for n, (name, ids) in enumerate(s["folds"].items()):
        print(f"\nFold {n+1}/{len(s['folds'])}: {len(ids['train'])} train / {len(ids['held_out'])} held-out identities", flush=True)
        directory, key = output / "folds" / name, fold_signature(context, name)
        train_rows = [r for r in rows if r["vehicle_id"] in set(ids["train"])]
        summary = fit_fold(train_rows, ids["train"], config, s["encoder_epochs"], device, directory, key,
                           remaining_folds=len(s["folds"])-n-1)
        q, g = context["fold_protocols"][name]
        if {r["vehicle_id"] for r in q+g} & set(summary["train_ids"]):
            raise ValueError("OOF encoder trained on one of its feature identities")
        encoder, exported = export_fold(directory, key, q)
        vectors = encode_cached(encoder, q+g, output / "cache" / f"{name}_oof.npz", key)
        pairs = PairSet(q, g, vectors, s["baseline"])
        train = scalar_training(pairs, hashes)
        parts.append(train)
        reports[name] = {"training": summary, "export": exported, "oof": pair_diagnostics(pairs, train)}
        del encoder, vectors, pairs
        # Same query/gallery and recipe, but the MVP has seen these identities.
        vectors = encode_cached(mvp, q+g, output / "cache" / f"{name}_matched.npz", key)
        pairs = PairSet(q, g, vectors, s["baseline"])
        train = scalar_training(pairs, hashes)
        matched.append(train)
        reports[name]["matched_in_sample"] = pair_diagnostics(pairs, train)
        _save(directory / "diagnostics.json", {"signature": key, **reports[name]})
        del vectors, pairs
    _save(output / "fold_diagnostics.json", {"signature": s, "folds": reports})
    return (pool_training(parts, s["outer"]["train"]), pool_training(matched, s["outer"]["train"]), reports)


def train_heads(oof, matched, output, signature):
    heads, reports = {}, {}
    for name, data in (("oof", oof), ("matched_in_sample", matched)):
        key = {"experiment": digest(signature), "condition": name,
               "training_sha256": digest({"query": data.query, "features": sha_array(data.features),
                   "targets": sha_array(data.targets), "weights": sha_array(data.weights), "valid": sha_array(data.valid)})}
        directory = output / "heads" / name
        trained = fit_head(data, data.targets, data.weights, data.valid, HEAD_CONFIG, directory, key,
                           fixed_epochs=signature["head_epochs"])
        heads[name], exported = export_head(directory / "selected.pt", directory, data.features,
                                            {**key, "weights_sha256": trained["weights_sha256"]})
        reports[name] = {"training": trained, "export": exported}
    return heads, reports


def sha_array(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def calibrate_blends(pairs, heads, threshold):
    """Only five preregistered blend weights; no refusal/architecture/epoch search."""
    selections, trials = {}, {}
    for name, head in heads.items():
        probabilities = head.probabilities(pairs.features)
        trials[name] = []
        for beta in BETAS:
            ranked, confidence = previous.score_pair_set(pairs, probabilities, beta)
            score = metrics(ranked, threshold, confidence["max_cosine"])
            trials[name].append({"beta": beta, "metrics": score})
        selections[name] = max(trials[name], key=lambda t: (t["metrics"]["mAP_at_10"], t["metrics"]["Rank_1"], -t["beta"]))["beta"]
        print(f"Calibration {name}: beta={selections[name]}; cosine refusal unchanged", flush=True)
    return {"betas": selections, "trials": trials, "threshold": threshold, "refusal_retuned": False,
            "validation_used": False, "head_weights_updated": False}


def evaluate(query, gallery, vectors, heads, frozen):
    pairs = PairSet(query, gallery, vectors, frozen["baseline"])
    results = {}
    for name, config in (("mvp", control.ACTIVE), ("osnet_neighbors", frozen["baseline"])):
        ranked, _ = control.ProtocolScores(query, gallery, vectors).evaluate(config)
        results[name] = previous.evaluated(ranked, ranked.confidence, frozen["threshold"])
    for name, head in heads.items():
        ranked, confidence = previous.score_pair_set(pairs, head.probabilities(pairs.features), frozen["betas"][name])
        results[name] = previous.evaluated(ranked, confidence["max_cosine"], frozen["threshold"])
    for value in results.values():
        for ref in ("osnet_neighbors", "previous_head", "matched_in_sample"):
            value[f"paired_vs_{ref}"] = paired_deltas(results[ref]["per_query"], value["per_query"])
    return results


def mask_check(context, heads, frozen):
    q, g = context["sample"]
    scores = {}
    with np.load(control.MASK_AUDIT / "embeddings.npz", allow_pickle=False) as cache:
        ids = cache["ids"].tolist()
        if ids != [r["image_id"] for r in q+g]:
            raise ValueError("Manual cache order mismatch")
        for condition in ("original", "manual"):
            scores[condition] = evaluate(q, g, dict(zip(ids, cache[f"mvp__{condition}"])), heads, frozen)
    return {"scores": scores, "selection_used": False, "threshold_retuned": False,
            "warning": "Reused small masks-7 audit, not proof of plate independence"}


def report_markdown(report):
    lines = ["# OOF-реранкер: результаты", "", "MVP и механизм отказа не изменены. Продвижения весов нет.", "",
        "## Validation", "", "| Система | mAP@10, % | Rank-1, % | F1, % | TNR, % |", "|---|---:|---:|---:|---:|"]
    for name, result in report["validation"].items():
        lines.append("| " + name + " | " + " | ".join(f"{result['metrics'][k]*100:.4f}" for k in
                                                      ("mAP_at_10", "Rank_1", "candidate_F1", "TNR")) + " |")
    lines += ["", "## Вклад OOF", ""]
    for ref in ("osnet_neighbors", "previous_head", "matched_in_sample"):
        d = report["validation"]["oof"][f"paired_vs_{ref}"]
        low, high = d["paired_bootstrap_95pct"]
        lines.append(f"- К {ref}: ΔmAP {100*d['mAP_at_10_delta']:+.4f} п.п.; "
                     f"95% парный интервал [{100*low:+.4f}; {100*high:+.4f}] п.п.")
    lines += ["", f"Зафиксированные на calibration веса: `{report['frozen']['betas']}`.", "",
        "## Трудность обучающих пар", "", "| Fold | OOF hit@50 | MVP hit@50 (те же q/g) | OOF raw mAP@10 |",
        "|---|---:|---:|---:|"]
    for name, value in report["folds"].items():
        lines.append(f"| {name} | {value['oof']['known_hit_at_50']:.4f} | "
                     f"{value['matched_in_sample']['known_hit_at_50']:.4f} | {value['oof']['raw_mAP_at_10']:.4f} |")
    lines += ["", "## Ограничения и воспроизводимость", "",
        "- Три encoder обучены с публичной stock-инициализации на двух фолдах, ровно 5 эпох с LR-горизонтом 30. "
        "Отложенный фолд не выбирает эпоху и не обновляет BatchNorm.",
        "- OOF-голова: 8 скалярных признаков, 7 фиксированных эпох. Она НЕ переобучается затем на seen-признаках MVP.",
        "- matched_in_sample: та же архитектура, seed, число эпох и q/g, но признаки MVP. "
        "Отличаются также сами encoder и число обучающих identity; это не идеально изолированный причинный эксперимент.",
        "- Векторы разных encoder не объединяются; объединяются только скалярные признаки независимо построенных top50.",
        "- Порог и confidence прежние. TNR/решения принять-отказать должны совпадать; F1 может меняться из-за нового top1.",
        "- Outer validation уже использовалась в истории проекта. Bootstrap по query с фиксированной gallery не учитывает историю подбора.",
        "- Ни camera/identity, ни другие query не являются входом реранкера. Позитивы вне raw top50 не восстанавливаются.",
        "- Основной эксперимент без масок; исправленные ручные маски используются только для диагностики с прежним порогом.",
        f"- Веса deployment (MVP + одна OOF-голова): {report['deployment_bytes']/1024**2:.2f} MiB. Три fold-encoder в deployment не нужны.",
        f"- CPU median/p95 после эмбеддинга: {report['benchmark']['median_query_ms']:.3f}/{report['benchmark']['p95_query_ms']:.3f} ms. "
        "Это не официальный RTX A5000 benchmark; extract не меняется.",
        "- Полный отчёт, mask audit, timing и SHA: report.json. Проверенные организаторским evaluator CSV: validation/.",
        "- Сначала оцениваем прирост порядка +1 п.п. без ухудшения отказа; автоматически никто в MVP не внедряется."]
    return "\n".join(lines)+"\n"


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This run is already active; do not launch a second notebook/CLI") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(output=EXPERIMENT / "results/run_01", device=None):
    torch.set_num_threads(2)
    context = prepare(device)
    s, output = context["signature"], Path(output)
    print(json.dumps(context["plan"], ensure_ascii=False, indent=2), flush=True)
    with run_lock(output):
        manifest = output / "experiment.json"
        if manifest.exists():
            _load(manifest, s)
        else:
            if any(p.name != ".run.lock" for p in output.iterdir()):
                raise ValueError("Use a new empty run directory; do not overwrite old experiments")
            _save(manifest, {"signature": s, "plan": context["plan"]})
        if (output / "report.json").exists():
            report = _load(output / "report.json", s)
            if any(sha256(output / p) != h for p, h in report["output_sha256"].items()):
                raise ValueError("Completed OOF experiment output changed")
            print("Completed run verified; no training, tuning or evaluation rerun", flush=True)
            return report
        if shutil.disk_usage(output).free < 1024**3:
            raise RuntimeError("Keep at least 1 GiB free for atomic checkpoints and temporary exports")
        started = time.perf_counter()
        oof, matched, folds = collect_training(context, output)
        heads, exports = train_heads(oof, matched, output, s)
        del oof, matched
        old_head = HeadEncoder(PREVIOUS / "weights/final/pair_head.onnx", s["previous_head_sha256"], HEAD_CONFIG)
        cal_path, frozen_path = output / "calibration.json", output / "frozen_selection.json"
        checksums = {name: info["export"]["sha256"] for name, info in exports.items()}
        if frozen_path.exists():
            frozen = _load(frozen_path, s)
            if frozen["calibration_sha256"] != sha256(cal_path) or frozen["heads_sha256"] != checksums:
                raise ValueError("Frozen calibration or heads changed")
        else:
            q, g = context["protocols"]["calibration"]
            ranked, _ = control.ProtocolScores(q, g, context["vectors"]).evaluate(control.ACTIVE)
            control.assert_baseline(metrics(ranked, s["baseline_threshold"]), context["baseline"]["calibration"])
            tuned = calibrate_blends(PairSet(q, g, context["vectors"], s["baseline"]), heads, s["baseline_threshold"])
            _save(cal_path, {"signature": s, **tuned})
            frozen = {"signature": s, "calibration_sha256": sha256(cal_path), "heads_sha256": checksums,
                "betas": {**tuned["betas"], "previous_head": s["previous_beta"]},
                "threshold": s["baseline_threshold"], "baseline": s["baseline"], "refusal_retuned": False}
            _save(frozen_path, frozen)
        print(f"FROZEN before validation: {frozen['betas']}", flush=True)
        # No training or calibration below this boundary. The previous head also keeps its old beta.
        heads["previous_head"] = old_head
        q, g = context["protocols"]["validation"]
        results = evaluate(q, g, context["vectors"], heads, frozen)
        control.assert_baseline(results["mvp"]["metrics"], context["baseline"]["validation"])
        old = _load(PREVIOUS / "report.json")
        control.assert_baseline(results["previous_head"]["metrics"], old["validation"]["ranking_only"]["metrics"])
        control.export_validation(output / "validation", q, g, context["vectors"], results)
        deployment_bytes = MODEL.stat().st_size + exports["oof"]["export"]["bytes"]
        if deployment_bytes > 2_000_000_000:
            raise ValueError("Deployment exceeds the 2 GB weight limit")
        timing_config = {"baseline": s["baseline"], "beta": frozen["betas"]["oof"],
                         "choices": {"combined": {"mode": "max_cosine", "threshold": frozen["threshold"]}}}
        timings = previous.benchmark(q, g, context["vectors"], heads["oof"], timing_config)
        masks = mask_check(context, heads, frozen)
        if any(sha256(Path(p)) != h for p, h in s["protected_sha256"].items()):
            raise RuntimeError("Protected code, annotations, historical results or MVP changed")
        if any(sha256(DATASET / "images" / f"{i}.jpg") != h for i, h in context["hashes"].items()):
            raise RuntimeError("Dataset frames changed during training")
        report = {"signature": s, "folds": folds, "heads": exports, "frozen": frozen, "validation": results,
            "mask_audit": masks, "benchmark": timings, "deployment_bytes": deployment_bytes,
            "protected_unchanged": True, "elapsed_this_invocation_seconds": time.perf_counter()-started,
            "encoder_training_seconds": sum(r["epoch_seconds"] for f in folds.values() for r in f["training"]["history"])}
        (output / "RESULTS.md").write_text(report_markdown(report), encoding="utf-8")
        report["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                                  if p.is_file() and p.name not in (".run.lock", "report.json")}
        _save(output / "report.json", report)
        print(f"Saved {output / 'RESULTS.md'}", flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results/run_01")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--check-only", action="store_true", help="Read-only preflight, no training or result writes")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(prepare(args.device)["plan"], ensure_ascii=False, indent=2))
    else:
        run(args.output, args.device)
