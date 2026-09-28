"""Fold-matched runtime shared by quick diagnostics and the Windows queue."""
import gc
from pathlib import Path
import numpy as np
import torch

from training import research_io as io, research_inputs as inputs, research_models as models
from training import research_scoring as scoring, research_training as training
from training.transreid_model import device_for


def prepare(input_directory, output, device, settings):
    device = device_for(device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    directory, manifest = inputs.load(input_directory)
    runtime = io.runtime(device)
    meta = {"schema":"parallel-research-v1", "input_sha256":io.sha(directory / "inputs.json"),
            "settings":settings, "runtime":runtime, "source_sha256":io.source_hashes(),
            "baseline":manifest["baseline"], "outer_evaluation":False, "promoted":False}
    if settings.get("input_sha256") and settings["input_sha256"] != meta["input_sha256"]:
        raise ValueError("Transferred inputs differ from the pinned experiment")
    output = Path(output).resolve()
    if output.is_relative_to(directory) or output == io.ROOT or not output.is_relative_to(io.ROOT / "OSNet-AIN-x1.0"):
        raise ValueError("Outputs must be in a new research variant, not inputs or product")
    io.freeze(output / "manifest.json", meta)
    train = set(manifest["train_ids"])
    hold = {i for p in manifest["draws"].values() for i in p["query_ids"]+p["gallery_ids"]}
    c = {"inputs":directory, "output":output, "manifest":manifest, "meta":meta, "signature":io.digest(meta),
         "device":device, "train":training.labeled([r for r in manifest["rows"] if r["vehicle_id"] in train]),
         "external":training.labeled(manifest["external"]),
         "rows":[r for r in manifest["rows"] if r["image_id"] in hold]}
    print(f"PREFLIGHT: {len(train)} train-ID, {len(manifest['holdout_ids'])} holdout-ID | {runtime['device']} "
          f"{runtime['gpu'] or ''} | frozen primary only", flush=True)
    return c


def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()


def baselines(c):
    directory = c["output"] / "baseline"
    if io.completed(directory, c["signature"]):
        return np.load(directory / "control.npy", allow_pickle=False)
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in ("B0", "R1_20260915", "R1_20260916", "R1_20260917"):
        print(f"CONTROL: {name}", flush=True)
        model = models.load_osnet(c["inputs"], c["manifest"], name).to(c["device"])
        values = models.encode(model, c["inputs"], c["rows"], c["device"], c["manifest"]["models"][name]["size"])
        np.save(directory / f"{name}.npy", values); parts.append(values)
        del model; cleanup()
    control = np.concatenate([parts[0], scoring.mix(parts[1:], [1/3]*3)], axis=1)
    np.save(directory / "control.npy", control)
    io.write(directory / "order.json", [r["image_id"] for r in c["rows"]])
    report = scoring.score_features(parts[1], c["rows"], c["manifest"]["draws"], control)
    io.write(directory / "metrics.json", report)
    io.finish(directory, c["signature"])
    return control
