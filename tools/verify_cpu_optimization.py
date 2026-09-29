"""Compare a preserved source snapshot to this checkout on CPU, without relaxing parity.

Run as python -m tools.verify_cpu_optimization --help. This diagnostic is NOT
the official latency/throughput benchmark (backend.benchmark).
"""
import argparse
import importlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, __version__ as pillow_version

from backend.artifacts import validate_artifacts
from backend.core import bbox, read_rows, sha256
from backend.images import ImageIndex
from backend.runtime import Runtime, encode_rows, write_json


def compare_exports(dataset, before, after):
    manifests = []
    for directory in (before, after):
        validate_artifacts(dataset, directory)
        manifests.append(json.loads((directory / "export_manifest.json").read_text()))
    # Implementation fingerprints SHOULD differ: old caches must not be reused.
    for key in ("profile", "provider", "encoder_fingerprint", "embedding_ids", "dimension",
                "preprocessing", "cosine_threshold", "ranking_policy", "candidate_policy",
                "ranking_r1_weight", "query_csv_sha256", "gallery_csv_sha256", "image_sha256"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Changed contract/input: {key}")
    files = {}
    for name in ("embeddings.npy", "submission.csv", "candidates.csv"):
        files[name] = sha256(before / name)
        if files[name] != sha256(after / name):
            raise ValueError(f"Not byte-identical: {name}")
    return {"byte_identical_sha256": files,
            "profile_fingerprints": [m["profile_fingerprint"] for m in manifests],
            "implementation_sha256": [m["implementation_sha256"] for m in manifests],
            "full_run_timings": [json.loads((p / "runtime_timing.json").read_text()) for p in (before, after)],
            "validation": manifests[1]["validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Unchanged snapshot with backend/, models/, evaluate.py")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--before-export", type=Path, required=True)
    parser.add_argument("--after-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file; previous evidence is preserved")
    evidence = compare_exports(args.dataset, args.before_export, args.after_export)
    spec = importlib.util.spec_from_file_location("baseline_backend", args.baseline / "backend/__init__.py",
                                                 submodule_search_locations=[str(args.baseline / "backend")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    original = importlib.import_module("baseline_backend.runtime")
    runtimes = {"before": original.Runtime(), "after": Runtime()}
    extractors = {"before": original.encode_rows, "after": encode_rows}
    # Ensure the timed sources are the same sources that produced the full exports.
    for index, name in enumerate(("before", "after")):
        if runtimes[name].fingerprint != evidence["profile_fingerprints"][index]:
            raise ValueError(f"Source changed after full export: {name}")
    rows = read_rows(args.dataset / "test_query.csv") + read_rows(args.dataset / "test_gallery.csv")
    images = ImageIndex(args.dataset / "images")
    for row in rows:
        with Image.open(images.resolve(row["image_id"])) as image:
            old = runtimes["before"].encoder.preprocess(image, bbox(row))
            new = runtimes["after"].encoder.preprocess(image, bbox(row))
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a, b)
    print(f"Pixel parity: {len(rows)} images, both input sizes, exact", flush=True)
    # Uniformly spaced real images; alternate A/B order to reduce order/thermal bias.
    selected = [rows[i] for i in np.linspace(0, len(rows) - 1, min(64, len(rows)), dtype=int)]
    timings = {"before": [], "after": []}
    for name in timings:
        extractors[name](runtimes[name], selected[:16], args.dataset, 16)
    reference = None
    for repeat in range(6):
        for name in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
            start = time.perf_counter()
            vectors = extractors[name](runtimes[name], selected, args.dataset, 16)
            elapsed = time.perf_counter() - start
            timings[name].append(elapsed)
            if reference is None:
                reference = vectors
            np.testing.assert_array_equal(vectors, reference)
            print(f"Extraction {repeat + 1}/6 {name}: {elapsed:.3f}s", flush=True)
    # Isolate the optimized preprocessing stage; this is not end-to-end latency.
    preprocessing = {"before": [], "after": []}
    decoded = []
    for row in selected:
        with Image.open(images.resolve(row["image_id"])) as image:
            image.load()
            decoded.append((image.copy(), bbox(row)))
    for repeat in range(10):
        for name in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
            start = time.perf_counter()
            for image, box in decoded:
                runtimes[name].encoder.preprocess(image, box)
            preprocessing[name].append(time.perf_counter() - start)
    medians = {k: statistics.median(v) for k, v in timings.items()}
    prep_medians = {k: statistics.median(v) for k, v in preprocessing.items()}
    result = {**evidence, "passed": True, "official_benchmark": False, "gpu_tested": False,
              "environment": {"platform": platform.platform(), "python": platform.python_version(),
                              "numpy": np.__version__, "pillow": pillow_version, "onnxruntime": ort.__version__,
                              "thread_environment": {k: os.getenv(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}},
              "pixel_parity_images": len(rows), "pixel_parity_sizes": [208, 256],
              "sample_ids": [r["image_id"] for r in selected], "batch_size": 16,
              "extraction_seconds": timings, "extraction_median_seconds": medians,
              "extraction_time_reduction_percent": 100 * (1 - medians["after"] / medians["before"]),
              "preprocessing_seconds": preprocessing, "preprocessing_median_seconds": prep_medians,
              "preprocessing_time_reduction_percent": 100 * (1 - prep_medians["after"] / prep_medians["before"]),
              "method": "Same process, CPU sessions intra=2/inter=1, 16-image warmup each; 6 alternating 64-image extracts. "
                        "File caches warm, image index/decode/preprocess/4 forwards/L2 included; no reranking. "
                        "Separate preprocessing-only measurements use decoded copies, 10 alternating repeats. "
                        "Full exports are separate fresh processes; single timings are descriptive, not a stable speed claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("passed", "extraction_time_reduction_percent", "preprocessing_time_reduction_percent")}, indent=2))


if __name__ == "__main__":
    main()
