"""Fresh end-to-end extractor timing, shared runtime; CPU or strict CUDA."""
import argparse
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .core import ROOT, bbox, read_rows, sha256
from .frozen_encoder import PROVIDERS
from .images import ImageIndex
from .runtime import DEFAULT_PROFILE, PROFILE_NAMES, Runtime, encode_rows, write_json


def weight_inventory(directory):
    directory = Path(directory).resolve()
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(directory):
                raise ValueError(f"Delivery weight link escapes directory: {path}")
            files.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    # Deliberately conservative: ALL bytes, including JSON/licenses and inactive stock weights.
    total = sum(x["bytes"] for x in files)
    return {"directory": str(directory), "files": files, "total_bytes": total,
            "limit_bytes": 2_000_000_000, "passed": bool(files) and total <= 2_000_000_000,
            "scope": "entire delivery models directory, not just the active ensemble"}


class MemorySampler:
    """Sample process RSS and NVML's per-process allocation through nvidia-smi."""
    def __init__(self, cuda, interval=.25):
        import psutil
        self.process = psutil.Process()
        self.cuda, self.interval = cuda, interval
        self.stop = threading.Event()
        self.rss = []
        self.vram = []
        self.errors = set()

    def sample(self):
        self.rss.append(self.process.memory_info().rss)
        if self.cuda:
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL, timeout=2)
                allocations = [float(line.split(",")[1].strip()) * 1024 ** 2
                               for line in output.splitlines()
                               if line.split(",")[0].strip() == str(os.getpid())]
                if allocations:
                    self.vram.append(sum(allocations))
                else:
                    self.errors.add("Process not visible to NVML (PID namespace/driver); VRAM unverified")
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.add(type(exc).__name__ + ": NVML sample unavailable")

    def __enter__(self):
        self.sample()
        def poll():
            while not self.stop.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)
        self.sample()

    def report(self):
        return {"process_rss_peak_sampled_bytes": max(self.rss) if self.rss else None,
                "process_vram_peak_sampled_bytes": max(self.vram) if self.vram else None,
                "sample_interval_seconds": self.interval, "rss_samples": len(self.rss),
                "vram_samples": len(self.vram), "vram_source": "nvidia-smi / NVML process allocation",
                "limitation": "Periodic samples, NOT a guaranteed exact peak; includes ORT allocations, not torch allocator",
                "warnings": sorted(self.errors)}


def benchmark(profile, dataset, provider=PROVIDERS[0], warmup=50, samples=300,
              throughput_seconds=10, progress=print):
    import psutil
    if warmup < 0 or samples < 1 or throughput_seconds <= 0:
        raise ValueError("Invalid benchmark counts")
    dataset = Path(dataset)
    queries = read_rows(dataset / "test_query.csv")
    gallery = read_rows(dataset / "test_gallery.csv")
    images = ImageIndex(dataset / "images")
    with MemorySampler(provider == "CUDAExecutionProvider") as memory:
        started = time.perf_counter()
        runtime = Runtime(profile, provider)
        model_load_seconds = time.perf_counter() - started
        sessions = [e.session for e in getattr(runtime.encoder, "members", [runtime.encoder])]
        if any(s.get_providers() != [provider] for s in sessions):
            raise RuntimeError("Provider fallback forbidden")

        def extract(rows):
            batch = []
            for row in rows:
                with Image.open(images.resolve(row["image_id"])) as image:
                    batch.append(runtime.encoder.preprocess(image, bbox(row)))
            # session.run returns host arrays; transfers and GPU execution finish before this returns.
            result = runtime.encoder.encode_batch(batch)
            if not np.isfinite(result).all():
                raise ValueError("Invalid benchmark vectors")
            return result

        latency = []
        for i in range(warmup + samples):
            started = time.perf_counter()
            extract([queries[i % len(queries)]])
            elapsed = time.perf_counter() - started
            if i >= warmup:
                latency.append(elapsed * 1000)
            if i % 25 == 0:
                progress(f"{profile}: latency {i + 1}/{warmup + samples} (warmup={warmup})")
        throughput = {}
        for count in (1, 8, 16, 32):
            rows = [queries[i % len(queries)] for i in range(count)]
            extract(rows)
            started, processed, iterations = time.perf_counter(), 0, 0
            while time.perf_counter() - started < throughput_seconds:
                extract(rows)
                processed += count
                iterations += 1
            seconds = time.perf_counter() - started
            throughput[str(count)] = {"images_per_second": processed / seconds, "seconds": seconds,
                                      "iterations": iterations, "images": processed}
            progress(f"{profile}: throughput batch={count}, {processed / seconds:.2f} images/s")
        started = time.perf_counter()
        full_vectors = encode_rows(runtime, queries + gallery, dataset, 16,
                                   lambda n, total: progress(f"{profile}: full extract {n}/{total}"))
        full_extract_seconds = time.perf_counter() - started
    return {**runtime.metadata(), "reused_measurement": False, "official_gpu_verified": False,
            "official_score": None, "interpretation": "Own-machine estimate only; organizers assign official scores",
            "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total,
            "onnxruntime": ort.__version__, "actual_providers": [s.get_providers() for s in sessions],
            "warmup": warmup, "samples": samples, "batch1_latency_ms": latency,
            "median_ms": float(np.median(latency)), "p95_ms": float(np.percentile(latency, 95)),
            "throughput": throughput, "model_load_seconds": model_load_seconds,
            "full_extract_seconds": full_extract_seconds, "full_extract_shape": list(full_vectors.shape),
            "memory": memory.report(), "images": len(queries) + len(gallery),
            "organizer_runtime_guideline_seconds": float(np.median(latency)) / 1000 * (len(queries) + len(gallery)) * 3,
            "includes": "decode, EXIF, bbox, resize, normalization, transfers, all member forwards, L2",
            "excludes": "ranking/export; measured separately by backend.infer"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, default=DEFAULT_PROFILE)
    parser.add_argument("--provider", choices=PROVIDERS, default=PROVIDERS[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new report path; speed measurements are never reused")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = benchmark(args.profile, args.dataset, args.provider)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
