"""Offline contest inference: a test-only dataset in, three submission files out."""

import argparse
import json
import os
import time
from pathlib import Path

from .core import ARTIFACTS, DATASET, read_rows
from .frozen_encoder import PROVIDERS
from .runtime import DEFAULT_PROFILE, PROFILE_NAMES, Runtime, export, write_json


def run_inference(dataset=DATASET, output=ARTIFACTS, profile=DEFAULT_PROFILE,
                  provider="CPUExecutionProvider", batch_size=16, progress=None):
    """Use the frozen calibration and existing ranking/export path; never train or calibrate."""
    started = time.perf_counter()
    dataset, output = Path(dataset), Path(output)
    read_rows(dataset / "test_query.csv")
    if len(read_rows(dataset / "test_gallery.csv")) < 10:
        raise ValueError("Contest inference requires at least 10 gallery images for Top-10")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory; existing results are preserved")
    runtime = Runtime(profile, provider)
    loaded = time.perf_counter()
    result = export(runtime, dataset, output, batch_size=batch_size, progress=progress)
    write_json(output / "runtime_timing.json", {
        "profile": profile, "provider": provider, "model_load_seconds": loaded - started,
        "full_runtime_seconds": time.perf_counter() - started, "reused_measurement": False,
        "includes": "model validation/load, hashes, decode, preprocessing, embeddings, graph, search, files, validation",
        "official_gpu_verified": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET,
                        help="Directory containing images, test_query.csv and test_gallery.csv")
    parser.add_argument("--output", type=Path, default=ARTIFACTS,
                        help="Directory for submission.csv, embeddings.npy and candidates.csv")
    parser.add_argument("--profile", choices=PROFILE_NAMES, default=os.getenv("REID_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--provider", choices=PROVIDERS, default=os.getenv("REID_PROVIDER", PROVIDERS[0]))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        result = run_inference(args.dataset, args.output, args.profile, args.provider, args.batch_size,
                               lambda done, total: print(f"{args.profile}: {done}/{total}", flush=True))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
