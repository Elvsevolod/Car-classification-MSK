"""Review v16: common-step selection, held-rule inner checks, explicit outer phase.

Historical trainers and their manifests are intentionally not edited or resumed here.
"""
import fcntl
import gc
import platform
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.core import ROOT, DATASET, ARTIFACTS, MODEL, STOCK_MODEL, read_rows, sha256
from backend.scoring import calibrate, metrics, ranked_queries
from training import osnet_ablation_suite as old
from training.audit import digest
from training.hpo import ExperimentConfig
from training.osnet_ablations import Ablation, AblationDataset, InferenceEncoder, MemoryBank, losses, optimizer_for
from training.osnet_fixed_fusion import initialize_fixed_fusion
from training.pipeline import format_duration, set_seed
from training.review_selection import make_draws, select_shared_steps, summarize_fixed
from training.stage6 import StepPKBatchSampler, audit_partitions, set_step_learning_rates, write_json


VARIANT = ROOT / "OSNet-AIN-x1.0/variant_16_review_protocol"


def variants():
    return {v.name: v for v in (
        Ablation(),
        Ablation("K1_color32_legacy", "Legacy color concat then shared BN", branch="color"),
        Ablation("K2_color32_fixed", "Separate BN/L2 then fixed cosine 80/20", branch="color"),
        Ablation("R1_resolution256", "256 train/inference", size=256),
        Ablation("M3_triplet", "Batch-hard triplet", metric="triplet"),
    )}


def initialize(classes, config, variant, device):
    factory = initialize_fixed_fusion if variant.name == "K2_color32_fixed" else old.initialize
    return factory(classes, config, variant, device)


def prepare(run_name="review_v1", device="mps", budget=old.Budget(),
            seeds=(20260915, 20260916, 20260917), draw_seeds=(20260915, 20261016, 20261117), dataset=DATASET):
    import onnxruntime
    import torchvision

    budget.validate()
    if len(seeds) != 3 or len(set(seeds)) != 3 or len(draw_seeds) != 3 or len(set(draw_seeds)) != 3:
        raise ValueError("Freeze three distinct training seeds and three distinct draw seeds")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("RUN_NAME must be a simple directory name")
    output = VARIANT / "runs" / run_name
    if not output.resolve().is_relative_to((VARIANT / "runs").resolve()):
        raise ValueError("Output escapes variant16")
    dataset = Path(dataset).resolve()
    rows, split = read_rows(dataset / "train.csv"), old.load_json(ARTIFACTS / "splits.json")
    frames = {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg") for r in rows}
    if sha256(dataset / "train.csv") != split["train_csv_sha256"] or frames != split["frame_sha256"]:
        raise ValueError("Organizer annotations/images changed")
    audit_partitions(rows, frames, split["identities"])
    if sha256(STOCK_MODEL) != old.STOCK_SHA256 or sha256(MODEL) != old.MVP_SHA256:
        raise ValueError("Stock/MVP changed")
    masks, inner = old.load_masks(rows, split, frames)
    draws = {fold: make_draws(rows, part["validation"], draw_seeds) for fold, part in inner.items()}
    for part in inner.values():
        audit_partitions([r for r in rows if r["vehicle_id"] in split["identities"]["train"]], frames, part)
    protected = [STOCK_MODEL, MODEL, dataset / "train.csv", ARTIFACTS / "splits.json",
                 ARTIFACTS / "baseline_metrics.json", ROOT / "evaluate.py", ROOT / "ORGANIZER_QA.md",
                 old.RECIPE, old.MASK_ROOT / "cache/automatic_masks.json", old.DETECTOR / "annotation/mask_plan.json"]
    sources = [ROOT / "training" / f"{n}.py" for n in (
        "osnet_review_protocol", "review_selection", "osnet_fixed_fusion", "frozen_inference",
        "osnet_ablation_suite", "osnet_ablations", "osnet", "hpo", "pipeline", "preprocessing",
        "stage6", "masked_hpo", "mask_reid_ablation", "audit")]
    sources += [ROOT / "backend" / f"{n}.py" for n in ("core", "evaluate", "scoring", "rerank")]
    base = ExperimentConfig(**old.load_json(old.RECIPE)["config"])
    manifest = {"version": 1, "budget": asdict(budget), "seeds": list(seeds), "draw_seeds": list(draw_seeds),
                "variants": {n: asdict(v) for n, v in variants().items()}, "base_recipe": asdict(base),
                "outer": split["identities"], "inner": inner, "draws": draws, "protocols": split["protocols"],
                "frames_sha256": digest(frames), "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "protected": {str(p): sha256(p) for p in protected},
                "runtime": {"device": str(device), "python": platform.python_version(), "torch": str(torch.__version__),
                            "torchvision": torchvision.__version__, "numpy": np.__version__,
                            "onnxruntime": onnxruntime.__version__, "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(), "torch_threads": torch.get_num_threads(),
                            "deterministic": torch.are_deterministic_algorithms_enabled()},
                "policy": {"selection": "one shared step per recipe; mean(draws) per seed then mean(seeds)",
                           "alternate": "evaluate exact frozen step; no alternate checkpoint/recipe selection",
                           "duration_transfer": "same optimizer updates AND full LR horizon on every fold/final",
                           "mvp": "historical outer-selected reference; NOT unbiased independent test",
                           "phases": "pilot -> confirm_inner -> separately authorized final",
                           "masks": "automatic anonymized-region diagnostic, NOT verified plate-only",
                           "junk": "unchanged official adapter; extra same-camera diagnostic only",
                           "annotation_edits": 0, "external_data": False, "promoted": False}}
    with old.run_lock(output):
        old.freeze_json(output / "manifest.json", manifest)
    return {"output": output, "manifest": manifest, "signature": digest(manifest), "rows": rows, "split": split,
            "masks": masks, "dataset": dataset, "device": torch.device(device), "budget": budget, "base": base,
            "seeds": tuple(seeds), "variants": variants(), "protected": manifest["protected"]}


def check_inputs(context, rehash=False):
    old.verify_protected(context, rehash_frames=rehash)


def check_other_runs(context):
    """Read-only best-effort guard against competing historical/new training kernels."""
    directories = [old.VARIANT, ROOT / "OSNet-AIN-x1.0/variant_15_nive_transfer", VARIANT]
    for directory in directories:
        for path in (directory / "runs").glob("*/.lock"):
            if path.parent.resolve() == context["output"].resolve():
                continue
            with path.open("r") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f"Another training run is active: {path.parent}") from error
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)


def confidence_summary(ranked):
    known = ranked.query["vehicle_id"].isin(ranked.gallery["vehicle_id"]).to_numpy()
    result = {}
    for name, mask in (("known", known), ("unknown", ~known)):
        values = ranked.confidence[mask]
        result[name] = ({"count": len(values), "mean": float(values.mean()),
                         "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}
                        if len(values) else {"count": 0})
    return result


def evaluate_draws(model, context, variant, fold, diagnostics=False):
    protocols = context["manifest"]["draws"][fold]
    selected = {n: p for n, p in protocols.items() if diagnostics or p["selection_eligible"]}
    ids = {i for p in selected.values() for i in p["query_ids"] + p["gallery_ids"]}
    rows = [r for r in context["rows"] if r["image_id"] in ids]
    by_id = {r["image_id"]: r for r in rows}
    encoded = old.encode(model, rows, context, variant)
    masked = old.encode(model, rows, context, variant, masked=True) if diagnostics else None
    scores = {}
    for name, protocol in selected.items():
        query, gallery = ([by_id[i] for i in protocol[k]] for k in ("query_ids", "gallery_ids"))
        ranked = ranked_queries(query, gallery, encoded)
        value = metrics(ranked, 0.)
        scores[name] = {"map": value["mAP_at_10"], "selection_eligible": protocol["selection_eligible"],
                        "known_queries": value["known_queries"], "unknown_queries": value["unknown_queries"]}
        if diagnostics:
            scores[name]["confidence"] = confidence_summary(ranked)
            if protocol["selection_eligible"]:
                conditions = {}
                for condition, qm, gm in (("masked_query", True, False), ("masked_gallery", False, True),
                                          ("masked_both", True, True)):
                    embeddings = {r["image_id"]: (masked if qm else encoded)[r["image_id"]] for r in query}
                    embeddings.update({r["image_id"]: (masked if gm else encoded)[r["image_id"]] for r in gallery})
                    changed = ranked_queries(query, gallery, embeddings)
                    conditions[condition] = {"map": metrics(changed, 0.)["mAP_at_10"],
                                             "confidence": confidence_summary(changed)}
                scores[name]["automatic_mask_diagnostics"] = conditions
    mean = float(np.mean([s["map"] for s in scores.values() if s["selection_eligible"]]))
    if not np.isfinite(mean):
        raise FloatingPointError("Non-finite inner metric")
    return {"mean_map": mean, "draws": scores}


def fit(context, name, seed, fold="primary", stop_step=None):
    """Primary retains every selectable state; alternate/final have no best-checkpoint search."""
    if fold not in {"primary", "alternate", "final"} or name not in context["variants"] or seed not in context["seeds"]:
        raise ValueError("Unknown training stage")
    budget = context["budget"]
    if fold == "primary":
        if stop_step is not None:
            raise ValueError("Primary must complete the full shared selection grid")
        stop = budget.max_steps
    else:
        if stop_step not in budget.boundaries(budget.max_steps):
            raise ValueError("Alternate/final require a frozen primary boundary")
        stop = int(stop_step)
    variant = context["variants"][name]
    config = variant.recipe(context["base"], seed)
    allowed = set(context["split"]["identities"]["train"] if fold == "final" else
                  context["manifest"]["inner"][fold]["train"])
    labels = {i: k for k, i in enumerate(sorted(allowed))}
    rows = [{**r, "label": labels[r["vehicle_id"]]} for r in context["rows"] if r["vehicle_id"] in allowed]
    signature = digest({"context": context["signature"], "variant": name, "seed": seed,
                        "fold": fold, "stop": stop, "rows": rows})
    directory = context["output"] / fold / name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    summary_path, last_path = directory / "summary.json", directory / "last.pt"
    if summary_path.exists():
        summary = old.load_json(summary_path)
        if summary["signature"] != signature:
            raise ValueError("Completed stage configuration changed")
        for item in summary["checkpoints"].values():
            if sha256(context["output"] / item["path"]) != item["sha256"]:
                raise ValueError("Completed stage checkpoint changed")
        if not last_path.exists() or sha256(last_path) != summary["last_sha256"]:
            raise ValueError("Completed authoritative checkpoint changed")
        return summary
    set_seed(seed)
    model = initialize(len(labels), config, variant, context["device"])
    optimizer, bank = optimizer_for(model, config), MemoryBank(0)
    history, checkpoints, start, elapsed = [], {}, 0, 0.
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Resume configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, checkpoints, start, elapsed = saved["history"], saved["checkpoints"], saved["step"], saved["elapsed"]
        del saved
        for item in checkpoints.values():
            if sha256(context["output"] / item["path"]) != item["sha256"]:
                raise ValueError("Resumed selectable checkpoint changed")
    dataset = AblationDataset(rows, variant, context["dataset"], augment=True)
    row_indices = {r["image_id"]: i for i, r in enumerate(rows)}
    for end in budget.boundaries(stop):
        if end <= start:
            continue
        set_seed(seed + start)
        sampler = StepPKBatchSampler(rows, config, end - start)
        sampler.set_epoch(start)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        totals, began = {}, time.perf_counter()
        for offset, (clean, robust, target, image_ids) in enumerate(loader):
            step = start + offset
            set_step_learning_rates(optimizer, config, budget, step)
            values, _ = losses(model, clean.to(context["device"]), robust.to(context["device"]),
                               target.to(context["device"]), torch.tensor([row_indices[i] for i in image_ids],
                               device=context["device"]), config, variant, bank, step)
            if not torch.isfinite(values["loss"]):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            values["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            for key, value in {**values, "gradient_norm": norm}.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())
        validation = evaluate_draws(model, context, variant, fold) if fold == "primary" else None
        elapsed += time.perf_counter() - began
        history.append({"step": end, "validation": validation, "train": {k: v/(end-start) for k,v in totals.items()}})
        if fold == "primary" or end == stop:
            path = directory / f"step_{end:05d}.pt"
            old.save_checkpoint(path, {"signature": signature, "model": model.state_dict(), "step": end})
            checkpoints[str(end)] = {"path": str(path.relative_to(context["output"])), "sha256": sha256(path)}
        old.save_checkpoint(last_path, {"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                       "step": end, "history": history, "checkpoints": checkpoints, "elapsed": elapsed})
        write_json(directory / "history.json", history)
        score = f"mean draw mAP={validation['mean_map']:.4f}" if validation else "fixed rule; no selection"
        print(f"{fold}/{name}/{seed}: {end}/{stop} | {score} | elapsed {format_duration(elapsed)} | "
              f"ETA {format_duration(elapsed/end*(stop-end))}", flush=True)
        start = end
    validation = (history[-1]["validation"] if fold == "primary" else
                  evaluate_draws(model, context, variant, fold, diagnostics=True) if fold == "alternate" else None)
    summary = {"signature": signature, "context_signature": context["signature"], "variant": name, "seed": seed,
               "fold": fold, "stop_step": stop, "lr_horizon": budget.max_steps, "updates": stop,
               "train_identities": len(labels), "train_images": len(rows), "elapsed_seconds": elapsed,
               "history": history, "validation": validation, "checkpoints": checkpoints,
               "last_sha256": sha256(last_path)}
    write_json(summary_path, summary)
    del model, optimizer
    gc.collect()
    return summary


def load_model(context, summary):
    entry = summary["checkpoints"][str(summary["stop_step"])]
    path = context["output"] / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise ValueError("Selected checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != summary["signature"] or payload["step"] != summary["stop_step"]:
        raise ValueError("Selected checkpoint step/signature mismatch")
    variant = context["variants"][summary["variant"]]
    model = initialize(summary["train_identities"], variant.recipe(context["base"], summary["seed"]), variant, context["device"])
    model.load_state_dict(payload["model"])
    return model.eval(), variant


def evaluate_final(context, summary):
    model, variant = load_model(context, summary)
    directory = context["output"] / "final" / variant.name / f"seed_{summary['seed']}"
    query, gallery = old.protocol_rows(context, "calibration")
    encoded = old.encode(model, query + gallery, context, variant)
    thresholds = {n: calibrate(ranked, confidence) for n, (ranked, confidence) in old.ranking_pair(query, gallery, encoded).items()}
    old.freeze_json(directory / "thresholds.json", {"thresholds": thresholds, "selection": "original calibration only",
                                                   "checkpoint": summary["checkpoints"][str(summary["stop_step"])]})
    query, gallery = old.protocol_rows(context, "validation")
    clean = old.encode(model, query + gallery, context, variant)
    masked = old.encode(model, query + gallery, context, variant, masked=True)
    result = {"thresholds": thresholds, "conditions": {}, "per_query": {},
              "reference_status": "outer development, not independent test"}
    for name, qm, gm in (("original", False, False), ("masked_query", True, False),
                         ("masked_gallery", False, True), ("masked_both", True, True)):
        embeddings = {r["image_id"]: (masked if qm else clean)[r["image_id"]] for r in query}
        embeddings.update({r["image_id"]: (masked if gm else clean)[r["image_id"]] for r in gallery})
        result["conditions"][name] = {}
        for method, (ranked, confidence) in old.ranking_pair(query, gallery, embeddings).items():
            result["conditions"][name][method] = metrics(ranked, thresholds[method], confidence)
            if name == "original":
                import evaluate as official
                per_query = {}
                for row in query:
                    qid = row["image_id"]
                    single = official.ranking_metrics(ranked.query.loc[[qid]], ranked.gallery, ranked.predictions)
                    if single["n_scored"]:
                        per_query[qid] = {"vehicle_id": row["vehicle_id"], "ap": single["mAP@10"]}
                result["per_query"][method] = per_query
    write_json(directory / "evaluation.json", result)
    export_model(context, summary, model, variant, thresholds["reranked"])
    return result


def phase_artifacts(context, result, phase):
    paths = [context["output"] / f"{phase.upper()}_RESULTS.md"]
    for summary in result.get("primary", []) + result.get("alternate", []) + result.get("final", []):
        directory = context["output"] / summary["fold"] / summary["variant"] / f"seed_{summary['seed']}"
        paths.extend(directory / name for name in ("summary.json", "last.pt", "history.json"))
        paths.extend(context["output"] / item["path"] for item in summary["checkpoints"].values())
        if summary["fold"] == "final":
            paths.extend(directory / name for name in ("evaluation.json", "thresholds.json", "encoder.onnx", "bundle.json", "export.json"))
    if phase == "confirm_inner":
        paths.append(context["output"] / "selection.json")
    if phase == "final":
        paths.append(context["output"] / "final_selection.json")
    return {str(path.relative_to(context["output"])): sha256(path) for path in paths}


def verify_phase(context, result, phase):
    if (result.get("signature") != context["signature"] or result.get("phase") != phase
            or result.get("complete") is not True or not result.get("artifacts")):
        raise ValueError("Completed phase signature/status changed")
    for relative, expected in result["artifacts"].items():
        path = context["output"] / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Completed phase artifact changed: {relative}")
    for summary in result.get("primary", []) + result.get("alternate", []) + result.get("final", []):
        path = context["output"] / summary["fold"] / summary["variant"] / f"seed_{summary['seed']}" / "summary.json"
        if old.load_json(path) != summary:
            raise ValueError("Completed phase summary changed")
    if phase in {"confirm_inner", "final"} and old.load_json(context["output"] / "selection.json") != result["selection"]:
        raise ValueError("Completed phase selection changed")


def export_model(context, summary, model, variant, threshold):
    import onnxruntime as ort
    from training.frozen_inference import write_bundle

    directory = context["output"] / "final" / variant.name / f"seed_{summary['seed']}"
    encoder = InferenceEncoder(model).eval().cpu()
    temporary, path = directory / "encoder.tmp.onnx", directory / "encoder.onnx"
    torch.onnx.export(encoder, torch.zeros(1, 3, variant.size, variant.size), temporary,
                      input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
    query, gallery = old.protocol_rows(context, "calibration")
    dataset = AblationDataset(query + gallery, variant, context["dataset"])
    parity = {}
    for batch in (1, 3, 8):
        images = torch.stack([dataset[i % len(dataset)][0] for i in range(batch)])
        with torch.no_grad():
            expected = encoder(images).numpy()
        actual = session.run(None, {"input": images.numpy()})[0]
        if actual.shape != expected.shape:
            raise ValueError("ONNX parity shape failed")
        error = float(np.abs(actual-expected).max())
        if not np.isfinite(actual).all() or error > 2e-4 or not np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5):
            raise ValueError("ONNX parity/shape/norm failed")
        parity[str(batch)] = error
    if path.exists() and sha256(path) != sha256(temporary):
        raise ValueError("Refusing to overwrite a different exported encoder")
    if not path.exists():
        temporary.replace(path)
    write_bundle(directory / "bundle.json", path, image_size=variant.size, resize_mode=variant.resize,
                 threshold=threshold, calibration={"split": "calibration", "protocol_sha256": digest(context["manifest"]["protocols"]["calibration"]),
                 "method": "frozen maximum raw cosine; select 0.7F1+0.3TNR on original calibration"})
    write_json(directory / "export.json", {"onnx_sha256": sha256(path), "cpu_parity": parity,
                                          "gpu_parity": "not tested", "promoted": False})


def write_report(context, result, phase):
    lines = ["# Review protocol: " + phase, "", "MVP unchanged; outer is historical development, not independent test.", ""]
    if phase == "pilot":
        lines += ["One seed, common last step only. No recipe/checkpoint selection yet.", "",
                  "| Recipe | Last-step mean draw mAP |", "|---|---:|"]
        lines += [f"| {s['variant']} | {s['validation']['mean_map']:.6f} |" for s in result["primary"]]
    elif phase == "confirm_inner":
        lines += ["| Recipe | Shared step | Primary mean | Fixed-rule alternate mean |", "|---|---:|---:|---:|"]
        for name, values in result["selection"]["aggregate"].items():
            alternate = result["alternate_summary"]["aggregate"][name]
            lines.append(f"| {name} | {values['step']} | {values['mean']:.6f} | {alternate['mean']:.6f} |")
        lines += ["", "Primary is tuned; alternate is another inner split, NOT an independent outer test.",
                  "Inspect fixed-minus-legacy fusion and seed deltas; do not count draws as independent seeds."]
    else:
        lines += ["| Recipe | Original reranked mAP@10 | F1 | TNR |", "|---|---:|---:|---:|"]
        for name, report in result["outer"].items():
            m = report["conditions"]["original"]["reranked"]
            lines.append(f"| {name} | {m['mAP_at_10']:.6f} | {m['candidate_F1']:.6f} | {m['TNR']:.6f} |")
    lines += ["", "NiVe/GeM/MixStyle are not added. Automatic masks are not verified plate-only masks.",
              "No automatic promotion, upload, or next phase. See README for the next explicit step."]
    (context["output"] / f"{phase.upper()}_RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(context, phase="pilot", allow_outer=False):
    if phase not in {"pilot", "confirm_inner", "final"}:
        raise ValueError("Unknown phase")
    if phase == "final" and not allow_outer:
        raise ValueError("Outer phase requires explicit ALLOW_OUTER_EVALUATION=True")
    device = context["device"]
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    with old.run_lock(context["output"]):
        check_inputs(context, rehash=True)
        output = context["output"] / f"{phase}.json"
        if output.exists():
            result = old.load_json(output)
            verify_phase(context, result, phase)
            return result
        if phase != "pilot" and not (context["output"] / "pilot.json").exists():
            raise ValueError("Complete and review the pilot first")
        if phase != "pilot":
            verify_phase(context, old.load_json(context["output"] / "pilot.json"), "pilot")
        check_other_runs(context)
        if phase == "final":
            if not (context["output"] / "confirm_inner.json").exists():
                raise ValueError("Complete and review confirm_inner before final")
            inner = old.load_json(context["output"] / "confirm_inner.json")
            verify_phase(context, inner, "confirm_inner")
            selection = inner["selection"]
            names = list(dict.fromkeys(["B0_control", selection["winner"]] +
                        (["K1_color32_legacy"] if selection["winner"] == "K2_color32_fixed" else [])))
            final = [fit(context, name, context["seeds"][0], "final", selection["aggregate"][name]["step"]) for name in names]
            old.freeze_json(context["output"] / "final_selection.json", final)
            outer = {s["variant"]: evaluate_final(context, s) for s in final}
            paired = {name: {method: old.paired_bootstrap(outer["B0_control"]["per_query"][method], report["per_query"][method])
                             for method in ("raw", "reranked")} for name, report in outer.items() if name != "B0_control"}
            result = {"final": final, "outer": outer, "selection": selection, "paired_vs_control": paired}
        else:
            seeds = context["seeds"][:1] if phase == "pilot" else context["seeds"]
            primary = [fit(context, name, seed) for seed in seeds for name in context["variants"]]
            result = {"primary": primary}
            if phase == "confirm_inner":
                selection = select_shared_steps(primary, list(context["variants"]), context["seeds"],
                                                context["budget"].boundaries(context["budget"].max_steps))
                old.freeze_json(context["output"] / "selection.json", selection)
                alternate = [fit(context, name, seed, "alternate", selection["aggregate"][name]["step"])
                             for seed in context["seeds"] for name in context["variants"]]
                fixed = summarize_fixed(alternate, selection, context["seeds"])
                scores = fixed["aggregate"]
                deltas = [scores["K2_color32_fixed"]["seed_scores"][str(seed)] -
                          scores["K1_color32_legacy"]["seed_scores"][str(seed)] for seed in context["seeds"]]
                fixed["fixed_minus_legacy_fusion"] = {"seed_deltas": deltas, "mean": float(np.mean(deltas))}
                result.update(selection=selection, alternate=alternate, alternate_summary=fixed)
        result.update(signature=context["signature"], phase=phase, complete=True, promoted=False, outer_evaluated=phase == "final")
        check_inputs(context, rehash=True)
        write_report(context, result, phase)
        result["artifacts"] = phase_artifacts(context, result, phase)
        write_json(output, result)
        return result
