"""Isolated schema-1/2 runtime measurements; never promote a model or calibrate it.

CPU measurements and CUDA measurements on an unconfirmed stand are diagnostics.
An official score requires both the full protocol and an explicitly confirmed,
observed organizer-class stand. No GPU package/driver compatibility is assumed.
"""
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.core import bbox, read_rows, sha256
from backend.evaluate import CANDIDATES_HEADER, validate_artifacts
from training.audit import digest
from training.stage6 import write_json
from training import frozen_inference as frozen
from training import policy_inference as deployment
from training import retrieval_policy as policy

WEIGHT_LIMIT_BYTES = 2_000_000_000  # Conservative decimal GB; report exact bytes too.
OFFICIAL_BATCHES = (1, 8, 16, 32)
WEIGHT_EXTENSIONS = {".pt", ".pth", ".bin", ".onnx", ".engine", ".plan",
                     ".safetensors", ".ckpt", ".trt", ".pb", ".tflite", ".npz"}


def load_encoder(bundle_path, provider="CPUExecutionProvider"):
    schema = json.loads(Path(bundle_path).read_text())["schema"]
    if schema not in (1, 2):
        raise ValueError("Unsupported runtime bundle schema")
    cls = frozen.FrozenEncoder if schema == 1 else deployment.PolicyEncoder
    return cls(bundle_path, provider)


def weight_inventory(bundle_path, extra_weight_paths=(), release_root=None):
    """Count every referenced weight, regardless of extension; aliases count once.

    Equal-content copies at distinct paths count separately, as in a release
    directory scan. Extra heads/local models must be declared explicitly. An
    optional release root adds every organizer-scanned weight extension.
    """
    files, visited = {}, set()

    def add(path, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Not a weight file: {path}")
        checksum = sha256(path)
        if expected is not None and checksum != expected:
            raise ValueError(f"Weight checksum mismatch: {path}")
        files[str(path)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": checksum}

    def visit(path, expected=None):
        path = Path(path).resolve(strict=True)
        if expected is not None and sha256(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {path}")
        if path in visited:
            raise ValueError("Cyclic or duplicate bundle reference")
        visited.add(path)
        bundle = json.loads(path.read_text())
        if bundle.get("schema") == 1:
            frozen._validate_policy(bundle)
            model = bundle["model"]
            add(path.parent / model["path"], model["sha256"])
        elif bundle.get("schema") == 2:
            deployment.validate_bundle(bundle)
            for member in bundle["members"]:
                visit(path.parent / member["path"], member["sha256"])
        else:
            raise ValueError("Unsupported runtime bundle schema")

    visit(bundle_path)
    for path in extra_weight_paths:
        add(path)
    if release_root is not None:
        root = Path(release_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("release_root must be a directory")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in WEIGHT_EXTENSIONS:
                add(path)
    total = sum(item["bytes"] for item in files.values())
    return {"files": sorted(files.values(), key=lambda item: item["path"]), "total_bytes": total,
            "limit_bytes": WEIGHT_LIMIT_BYTES, "within_limit": total <= WEIGHT_LIMIT_BYTES,
            "scope": "bundle weights + declared auxiliary weights + optional release scan",
            "release_directory_scanned": release_root is not None}


def gallery_cache_key(encoder_fingerprint, rows, dataset):
    """Includes space/preprocessing identity, gallery order, original bbox and bytes."""
    if not encoder_fingerprint or not rows:
        raise ValueError("Cache requires encoder fingerprint and nonempty gallery")
    records = [{"image_id": row["image_id"], "bbox": list(bbox(row)),
                "sha256": sha256(Path(dataset) / "images" / f"{row['image_id']}.jpg")}
               for row in rows]
    return digest({"schema": 1, "encoder_fingerprint": encoder_fingerprint, "gallery": records})


def validate_gallery_cache(manifest, encoder_fingerprint, rows, dataset):
    expected = gallery_cache_key(encoder_fingerprint, rows, dataset)
    if manifest.get("gallery_cache_key") != expected:
        raise ValueError("Stale gallery cache: encoder, preprocessing, order, bbox or image changed")
    return expected


def _extract(encoder, rows, dataset):
    # Image.open is deliberately INSIDE every timed invocation: no decoded cache.
    batch = []
    for row in rows:
        with Image.open(Path(dataset) / "images" / f"{row['image_id']}.jpg") as image:
            batch.append(encoder.preprocess(image, bbox(row)))
    vectors = encoder.encode_batch(batch)
    if (vectors.dtype != np.float32 or vectors.shape != (len(rows), encoder.dimension)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)):
        raise ValueError("Extractor must return finite float32 L2-normalized embeddings")
    return vectors


def _device(provider):
    if provider == "CPUExecutionProvider":
        return (lambda: None), {"kind": "CPU", "synchronization": "synchronous CPU call"}
    if provider != "CUDAExecutionProvider":
        raise ValueError("Only the existing frozen CPU/CUDA adapters are supported; no silent MPS fallback")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires a working CUDA synchronization backend (torch.cuda)")
    props = torch.cuda.get_device_properties(0)
    return (lambda: torch.cuda.synchronize(0)), {
        "kind": "CUDA", "name": props.name, "total_memory_bytes": props.total_memory,
        "device_index": 0, "synchronization": "torch.cuda.synchronize(0), all device streams",
        "torch_cuda_build": torch.version.cuda}


def _environment():
    versions = {}
    for package in ("numpy", "Pillow", "onnx", "onnxruntime", "onnxruntime-gpu", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                check=True, timeout=5).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        driver = None
    return {"platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "packages": versions, "nvidia_smi": driver,
            "gpu_lock_verified": False, "offline_container_verified": False}


def _protocol_valid(samples, warmup, seconds_per_batch, batches):
    return samples == 300 and warmup == 50 and seconds_per_batch >= 10 and tuple(batches) == OFFICIAL_BATCHES


def benchmark_bundle(bundle_path, dataset, provider="CPUExecutionProvider", *, rows=None,
                     samples=300, warmup=50, seconds_per_batch=10., batches=OFFICIAL_BATCHES,
                     official_hardware=False, extra_weight_paths=(), release_root=None):
    """Measure a frozen extractor; short settings are explicitly non-official.

    `official_hardware` is the operator's attestation of the organizer stand,
    not an automatic claim based on a GPU model. Observed hardware is checked
    as well. OS page cache is not purged; files are reopened and decoded.
    """
    batches = tuple(batches)
    if (type(samples) is not int or samples < 1 or type(warmup) is not int or warmup < 0
            or not np.isfinite(seconds_per_batch) or seconds_per_batch <= 0 or not batches
            or any(type(n) is not int or n < 1 for n in batches) or len(set(batches)) != len(batches)):
        raise ValueError("Invalid benchmark protocol")
    if provider not in frozen.PROVIDERS:
        raise ValueError("Unsupported provider; no implicit fallback")
    rows = read_rows(Path(dataset) / "test_query.csv") if rows is None else list(rows)
    if not rows:
        raise ValueError("Benchmark needs at least one image")
    synchronize, device = _device(provider)
    environment = _environment()
    weights = weight_inventory(bundle_path, extra_weight_paths, release_root)
    synchronize()
    started = time.perf_counter()
    encoder = load_encoder(bundle_path, provider)
    synchronize()
    loading_ms = 1000 * (time.perf_counter() - started)
    members = getattr(encoder, "members", [encoder])
    actual_providers = [member.session.get_providers() for member in members]
    if any(actual != [provider] for actual in actual_providers):
        raise RuntimeError(f"Provider fallback forbidden: {actual_providers}")

    timings = []
    for index in range(warmup + samples):
        synchronize()
        started = time.perf_counter()
        _extract(encoder, [rows[index % len(rows)]], dataset)
        synchronize()
        elapsed = time.perf_counter() - started
        if index >= warmup:
            timings.append(1000 * elapsed)
    throughput = []
    for batch_size in batches:
        batch = [rows[index % len(rows)] for index in range(batch_size)]
        _extract(encoder, batch, dataset)  # Untimed shape-specific warmup.
        synchronize()
        started, count = time.perf_counter(), 0
        elapsed = 0.
        while elapsed < seconds_per_batch:
            batch = [rows[(count + index) % len(rows)] for index in range(batch_size)]
            _extract(encoder, batch, dataset)
            synchronize()
            count += batch_size
            elapsed = time.perf_counter() - started
        throughput.append({"batch": batch_size, "images": count, "seconds": elapsed, "fps": count / elapsed})
    synchronize()
    first = _extract(encoder, rows[:min(8, len(rows))], dataset)
    second = _extract(encoder, rows[:min(8, len(rows))], dataset)
    synchronize()
    deterministic = bool(np.array_equal(first, second))
    protocol_ok = _protocol_valid(samples, warmup, seconds_per_batch, batches)
    observed_target = (device["kind"] == "CUDA" and device.get("name") == "NVIDIA RTX A5000"
                       and 23 * 2**30 <= device.get("total_memory_bytes", 0) <= 25 * 2**30
                       and "Xeon" in environment["cpu"] and "6338" in environment["cpu"]
                       and environment["logical_cpus"] == 128 and environment["nvidia_smi"] is not None
                       and platform.system() == "Linux")
    official = bool(official_hardware and observed_target and protocol_ok and deterministic and weights["within_limit"])
    result = {"schema": 1, "provider": provider, "actual_providers": actual_providers,
              "device": device, "environment": environment, "weights": weights,
              "bundle_sha256": sha256(bundle_path), "fingerprint": encoder.fingerprint,
              "dimension": encoder.dimension, "encoder_forward_count": len(members),
              "model_load_ms": loading_ms, "warmup": warmup, "samples": samples,
              "latency_b1": {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))},
              "throughput": throughput, "best_fps": max(item["fps"] for item in throughput),
              "determinism": {"bitwise_equal": deterministic, "max_abs_error": float(np.max(np.abs(first-second)))},
              "protocol_valid": protocol_ok, "official_hardware_attested": bool(official_hardware),
              "observed_target_hardware": observed_target, "official_gpu_verified": official,
              "measurement_kind": "organizer protocol on attested stand" if official else "diagnostic only",
              "peak_vram_bytes": None,
              "peak_vram_status": "not measured: PyTorch allocator statistics do not cover ONNX Runtime allocations",
              "includes": ["file read", "decode", "EXIF", "original bbox crop", "resize", "preprocessing",
                           "host/device transfers", "all encoder forwards", "postprocessing", "L2"],
              "excludes": ["model load (reported separately)", "gallery search", "reranking"],
              "disk_cache_note": "Files reopened each iteration; OS page cache not purged; no decoded image cache",
              "image_ids": [row["image_id"] for row in rows], "training_updates": 0, "promoted": False}
    result["score"] = performance_score(result)
    return result


def performance_score(report, map_at_10=None, f1=None, tnr=None):
    """No estimated competition score from CPU, short loops or overweight models."""
    if not (report.get("official_gpu_verified") and report.get("protocol_valid")
            and report.get("weights", {}).get("within_limit")):
        return {"performance_points": None, "auto_points": None,
                "reason": "Official GPU stand/protocol/weight admissibility not verified"}
    latency, fps = report["latency_b1"]["median_ms"], report["best_fps"]
    if not np.isfinite([latency, fps]).all() or latency < 0 or fps <= 0:
        raise ValueError("Invalid measured performance")
    points = 10 * float(np.clip((80-latency)/40, 0, 1)) + 10 * float(np.clip((fps-50)/50, 0, 1))
    metrics = (map_at_10, f1, tnr)
    automatic = None
    if any(value is not None for value in metrics):
        if any(value is None for value in metrics) or not np.isfinite(metrics).all() or any(not 0 <= v <= 1 for v in metrics):
            raise ValueError("Provide all three quality metrics in [0, 1]")
        automatic = 45 * map_at_10 + 10 * (.7 * f1 + .3 * tnr) + points
    return {"performance_points": points, "auto_points": automatic, "auto_maximum": 75,
            "engineering_and_presentation_not_estimated": True}


def audit_streaming(encoder, queries, gallery, dataset, *, batch_size=8, atol=2e-4):
    """Real file reads: singleton/batched/reversed queries against one fixed gallery.

    This is a runtime parity check, not a proof of network-isolated deployment.
    No labels are consumed by the extractor, ranking or candidate decisions.
    """
    if not queries or not gallery or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Streaming audit needs nonempty data and positive batch size")
    base = frozen._encode_rows(encoder, queries, dataset, 1)
    batched = frozen._encode_rows(encoder, queries, dataset, batch_size)
    reversed_vectors = frozen._encode_rows(encoder, list(reversed(queries)), dataset, batch_size)[::-1]
    gv = frozen._encode_rows(encoder, gallery, dataset, batch_size)
    bundle = encoder.bundle
    ranking = bundle["ranking"] if bundle["schema"] == 2 else "legacy"
    candidate = bundle["candidate_policy"] if bundle["schema"] == 2 else "ranking_top1"
    checks = {name: deployment.compare_decisions(queries, gallery, (base, gv), (vectors, gv),
                                                ranking, bundle["threshold"], candidate, atol)
              for name, vectors in (("batch_size", batched), ("query_permutation", reversed_vectors))}
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks,
            "query_count": len(queries), "gallery_count": len(gallery),
            "gallery_cache_key": gallery_cache_key(encoder.fingerprint, gallery, dataset),
            "network_isolation_verified": False, "training_updates": 0}


def timed_export(bundle_path, dataset, output, provider="CPUExecutionProvider", *, batch_size=16):
    """Full frozen inference on test CSVs, with stage timings and three real files.

    No quality evaluation, training, calibration or threshold selection occurs.
    Labels, even if present in supplied CSVs, are discarded. The gallery graph
    is static and each query is ranked independently by the existing policy.
    An existing nonempty output directory is never overwritten.
    """
    dataset, output = Path(dataset), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new/empty timed-export directory")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if provider not in frozen.PROVIDERS:
        raise ValueError("Unsupported provider; no implicit fallback")
    synchronize, device = _device(provider)
    timings = {}

    def measured(name, call):
        synchronize()
        started = time.perf_counter()
        value = call()
        synchronize()
        timings[name] = time.perf_counter() - started
        return value

    synchronize()
    started = time.perf_counter()
    encoder = measured("model_load_seconds", lambda: load_encoder(bundle_path, provider))
    members = getattr(encoder, "members", [encoder])
    actual_providers = [member.session.get_providers() for member in members]
    if any(actual != [provider] for actual in actual_providers):
        raise RuntimeError(f"Provider fallback forbidden: {actual_providers}")

    def inputs():
        keys = ("image_id", "x", "y", "w", "h")
        return tuple([{key: row[key] for key in keys} for row in read_rows(dataset / name)]
                     for name in ("test_query.csv", "test_gallery.csv"))

    queries, gallery = measured("read_csv_seconds", inputs)
    gv = measured("gallery_extraction_seconds", lambda: frozen._encode_rows(encoder, gallery, dataset, batch_size))
    qv = measured("query_extraction_seconds", lambda: frozen._encode_rows(encoder, queries, dataset, batch_size))
    bundle = encoder.bundle
    ranking_name = bundle["ranking"] if bundle["schema"] == 2 else "legacy"
    candidate = bundle["candidate_policy"] if bundle["schema"] == 2 else "ranking_top1"
    ranking = measured("gallery_graph_search_rerank_seconds", lambda: policy.rank_vectors(qv, gv, ranking_name))

    def export():
        if len(gallery) < 10:
            raise ValueError("Official submission requires at least ten gallery images")
        ordered, accepted = policy.predictions(queries, gallery, ranking, bundle["threshold"], candidate)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "submission.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([qid, *ids] for qid, ids in ordered.items())
        with (output / "candidates.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CANDIDATES_HEADER)
            writer.writerows([qid, pairs[0][0], (pairs[0][1] + 1) / 2 if bundle["schema"] == 1 else pairs[0][1]]
                             for qid, pairs in accepted.items())
        np.save(output / "embeddings.npy", np.concatenate([qv, gv]).astype(np.float32))
        return validate_artifacts(dataset, output)

    validation = measured("write_and_validate_artifacts_seconds", export)
    synchronize()
    timings["full_runtime_seconds"] = time.perf_counter() - started
    report = {"schema": 1, "purpose": "full frozen runtime timing, NOT a quality evaluation",
              "provider": provider, "actual_providers": actual_providers, "device": device,
              "bundle_sha256": sha256(bundle_path), "fingerprint": encoder.fingerprint,
              "dimension": encoder.dimension, "encoder_forward_count": len(members), "batch_size": batch_size,
              "query_count": len(queries), "gallery_count": len(gallery), "timings": timings,
              "timing_scope": "model load + CSV read + gallery/query extraction + graph/search/rerank + three-file write/validation",
              "timing_excludes": "device initialization and subsequent provenance hashing/report serialization",
              "gallery_cache_used": False,
              "gallery_cache_key": gallery_cache_key(encoder.fingerprint, gallery, dataset),
              "query_csv_sha256": sha256(dataset / "test_query.csv"),
              "gallery_csv_sha256": sha256(dataset / "test_gallery.csv"),
              "query_images": {row["image_id"]: sha256(dataset / "images" / f"{row['image_id']}.jpg") for row in queries},
              "embedding_ids": [row["image_id"] for row in queries + gallery],
              "ranking_policy": ranking_name, "candidate_policy": candidate, "threshold": bundle["threshold"],
              "validation": validation, "weights": weight_inventory(bundle_path),
              "artifacts": {name: sha256(output / name) for name in ("submission.csv", "candidates.csv", "embeddings.npy")},
              "official_gpu_verified": False, "network_isolation_verified": False,
              "quality_evaluated": False, "training_updates": 0, "promoted": False}
    write_json(output / "timed_export_manifest.json", report)
    return report
