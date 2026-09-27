"""Two additional final seeds, without changing the frozen variant16 experiment."""
import platform
import re
from pathlib import Path

import torch

from backend.core import ARTIFACTS, DATASET, ROOT, read_rows, sha256
from training import osnet_review_protocol as review
from training.audit import digest
from training.final_seed_report import aggregate_results, format_report
from training.hpo import ExperimentConfig
from training.osnet_ablations import Ablation
from training.stage6 import write_json


VARIANT = ROOT / "OSNet-AIN-x1.0/variant_17_final_seed_confirmation"
NAMES = ("B0_control", "R1_resolution256")
STEPS = 800
EVALUATION_FILES = ("evaluation.json", "thresholds.json", "encoder.onnx", "bundle.json", "export.json")


def runtime(device):
    import numpy
    import onnxruntime
    import torchvision
    return {"device": str(device), "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": torchvision.__version__, "numpy": numpy.__version__,
            "onnxruntime": onnxruntime.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "torch_threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def run_directory(variant, name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Run name must be a simple directory name")
    path = variant / "runs" / name
    if not path.resolve().is_relative_to((variant / "runs").resolve()):
        raise ValueError("Run directory escapes experiment")
    return path


def load_source(source_run="review_v1", dataset=DATASET):
    """Read-only reconstruction: never call the source prepare/run or touch its lock."""
    output = run_directory(review.VARIANT, source_run)
    manifest = review.old.load_json(output / "manifest.json")
    context = {"output": output, "manifest": manifest, "signature": digest(manifest),
               "protected": manifest["protected"], "dataset": Path(dataset),
               "rows": read_rows(Path(dataset) / "train.csv")}
    protected = {str(output / "manifest.json"): sha256(output / "manifest.json")}
    for phase in ("pilot", "confirm_inner", "final"):
        path = output / f"{phase}.json"
        result = review.old.load_json(path)
        review.verify_phase(context, result, phase)
        protected[str(path)] = sha256(path)
        protected.update({str(output / name): checksum for name, checksum in result["artifacts"].items()})
    seeds = manifest["seeds"]
    if (len(seeds) != 3 or len(set(seeds)) != 3 or manifest["budget"]["max_steps"] != 1700
            or result["selection"]["winner"] != NAMES[1]
            or result["final"] != review.old.load_json(output / "final_selection.json")
            or {(s["variant"], s["seed"]) for s in result["final"]} != {(n, seeds[0]) for n in NAMES}
            or len(result["final"]) != 2 or set(result["outer"]) != set(NAMES)):
        raise ValueError("Expected the completed B0/R1 first-seed final, not a new selection")
    for summary in result["final"]:
        name = summary["variant"]
        directory = output / "final" / name / f"seed_{seeds[0]}"
        if (summary["stop_step"] != STEPS or summary["lr_horizon"] != 1700
                or result["selection"]["aggregate"][name]["step"] != STEPS
                or summary["validation"] is not None
                or any(h["validation"] is not None for h in summary["history"])
                or result["outer"][name] != review.old.load_json(directory / "evaluation.json")):
            raise ValueError("Source final recipe/report differs from its frozen artifacts")
    review.check_inputs(context)
    frames = {r["image_id"]: sha256(context["dataset"] / "images" / f"{r['image_id']}.jpg")
              for r in context["rows"]}
    if digest(frames) != manifest["frames_sha256"]:
        raise ValueError("Original images changed")
    split = review.old.load_json(ARTIFACTS / "splits.json")
    if split["identities"] != manifest["outer"] or split["protocols"] != manifest["protocols"]:
        raise ValueError("Outer partitions/protocols changed")
    masks, inner = review.old.load_masks(context["rows"], split, frames)
    if inner != manifest["inner"]:
        raise ValueError("Inner partitions changed")
    context.update(split=split, masks=masks, device=torch.device(manifest["runtime"]["device"]),
                   budget=review.old.Budget(**manifest["budget"]), seeds=tuple(seeds),
                   base=ExperimentConfig(**manifest["base_recipe"]),
                   variants={n: Ablation(**manifest["variants"][n]) for n in NAMES})
    return context, result, protected


def prepare(run_name="final_seeds_v1", source_run="review_v1", dataset=DATASET):
    source, first, source_files = load_source(source_run, dataset)
    if runtime(source["device"]) != source["manifest"]["runtime"]:
        raise ValueError("Runtime differs from the first final: preserve device, versions, threads and determinism")
    output = run_directory(VARIANT, run_name)
    manifest = {**source["manifest"], "version": 1,
                "protected": {**source["protected"], **source_files},
                "source_sha256": {**source["manifest"]["source_sha256"],
                    **{f"training/{name}.py": sha256(ROOT / "training" / f"{name}.py")
                       for name in ("final_seed_confirmation", "final_seed_report")}},
                "extension": {"source_run": source_run, "source_directory": str(source["output"]),
                              "source_signature": source["signature"], "names": list(NAMES),
                              "reused_seed": source["seeds"][0], "new_seeds": list(source["seeds"][1:]),
                              "stop_steps": {n: STEPS for n in NAMES},
                              "decision": "additional seeds requested after seeing first outer; not a fresh independent test",
                              "aggregation": "all three seeds equally; sample SD and paired deltas; no best-seed choice or ensemble",
                              "thresholds": "each model calibrated separately on original calibration only"}}
    context = {**source, "output": output, "manifest": manifest, "signature": digest(manifest),
               "protected": manifest["protected"], "first_result": first}
    review.check_other_runs(context)
    with review.old.run_lock(output):
        review.old.freeze_json(output / "manifest.json", manifest)
    return context


def verify_receipt(context, path):
    receipt = review.old.load_json(path)
    if receipt["signature"] != context["signature"] or not receipt["artifacts"]:
        raise ValueError("Completion receipt configuration changed")
    for name, checksum in receipt["artifacts"].items():
        artifact = context["output"] / name
        if not artifact.is_file() or sha256(artifact) != checksum:
            raise ValueError(f"Completed artifact changed: {name}")
    return receipt


def evaluation(context, summary):
    directory = context["output"] / "final" / summary["variant"] / f"seed_{summary['seed']}"
    receipt_path = directory / "evaluation_complete.json"
    if receipt_path.exists():
        receipt = verify_receipt(context, receipt_path)
        if receipt["stage_signature"] != summary["signature"]:
            raise ValueError("Evaluation checkpoint changed")
        return review.old.load_json(directory / "evaluation.json")
    result = review.evaluate_final(context, summary)
    if result != review.old.load_json(directory / "evaluation.json"):
        raise ValueError("Evaluation result differs from saved report")
    artifacts = {str((directory / name).relative_to(context["output"])): sha256(directory / name)
                 for name in EVALUATION_FILES}
    write_json(receipt_path, {"signature": context["signature"], "stage_signature": summary["signature"],
                              "artifacts": artifacts})
    return result


def fit_final(context, name, seed):
    summary = review.fit(context, name, seed, "final", STEPS)
    directory = context["output"] / "final" / name / f"seed_{seed}"
    # The frozen trainer may commit last.pt and be interrupted before history.json.
    # Recover only that derived file; never change a completed checkpoint or summary.
    committed = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    history = committed["history"]
    if committed["signature"] != summary["signature"] or history != summary["history"]:
        raise ValueError("Summary differs from authoritative training history")
    path = directory / "history.json"
    if path.exists():
        saved = review.old.load_json(path)
        if saved == history:
            return summary
        if not isinstance(saved, list) or len(saved) >= len(history) or saved != history[:len(saved)]:
            raise ValueError("Derived training history changed, not a recoverable interrupted write")
    write_json(path, history)
    print(f"Restored derived training history from last.pt: {name}/{seed}", flush=True)
    return summary


def run(context, allow_outer=False, allow_cpu=False):
    if not allow_outer:
        raise ValueError("Explicit ALLOW_OUTER_EVALUATION=True is required for additional final evaluation")
    device = context["device"]
    if runtime(device) != context["manifest"]["runtime"]:
        raise ValueError("Runtime changed since preparation")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("CPU training requires explicit ALLOW_CPU_TRAINING=True")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    # The parent lock serializes variant17 runs; old trainers are checked read-only.
    with review.old.run_lock(VARIANT / "runs"), review.old.run_lock(context["output"]):
        review.check_other_runs(context)
        review.check_inputs(context, rehash=True)
        completed = context["output"] / "complete.json"
        if completed.exists():
            verify_receipt(context, completed)
            return review.old.load_json(context["output"] / "results.json")
        summaries = [fit_final(context, name, seed)
                     for seed in context["seeds"][1:] for name in NAMES]
        review.old.freeze_json(context["output"] / "final_selection.json", summaries)
        records = [{"variant": name, "seed": context["seeds"][0],
                    "evaluation": context["first_result"]["outer"][name]} for name in NAMES]
        # All four new checkpoints are frozen before any new outer evaluation.
        for summary in summaries:
            records.append({"variant": summary["variant"], "seed": summary["seed"],
                            "evaluation": evaluation(context, summary)})
        aggregate = aggregate_results(records, NAMES, context["seeds"])
        baseline = review.old.load_json(ARTIFACTS / "baseline_metrics.json")
        result = {"signature": context["signature"], "complete": True, "promoted": False,
                  "outer_evaluated": True, "new_optimizer_updates": sum(s["updates"] for s in summaries),
                  "reused_seed": context["seeds"][0], "source": context["manifest"]["extension"],
                  "final": summaries, "evaluations": records, **aggregate,
                  "baseline_reference": baseline}
        review.check_inputs(context, rehash=True)
        write_json(context["output"] / "results.json", result)
        (context["output"] / "FINAL_RESULTS.md").write_text(format_report(aggregate, baseline), encoding="utf-8")
        artifacts = review.phase_artifacts(context, {"final": summaries}, "final")
        for path in [context["output"] / "results.json", *context["output"].glob("final/*/seed_*/evaluation_complete.json")]:
            artifacts[str(path.relative_to(context["output"]))] = sha256(path)
        write_json(completed, {"signature": context["signature"], "artifacts": artifacts})
        print("Final seed confirmation complete: all three seeds averaged; MVP unchanged.", flush=True)
        return result
