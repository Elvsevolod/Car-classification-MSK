"""Opt-in frozen ONNX inference; never calibrates or reads train at export time."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from backend.core import bbox, crop_image, normalize, read_rows, sha256
from backend.evaluate import CANDIDATES_HEADER, validate_artifacts, write_json
from backend.rerank import KReciprocalReranker
from training.preprocessing import IMAGENET_MEAN, IMAGENET_STD, resize_crop

PROVIDERS = ("CPUExecutionProvider", "CUDAExecutionProvider")
RERANKING = {"method": "streaming k-reciprocal", "k1": 20, "k2": 3, "lambda": .5}
CONFIDENCE = "maximum raw gallery cosine; not a probability"


def _external_tensors(message):
    if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
        return True
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            if any(_external_tensors(item) for item in (value if field.is_repeated else [value])):
                return True
    return False


def _session(model_path, provider):
    if provider not in PROVIDERS or provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested provider is unavailable: {provider}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if provider == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    model_bytes = Path(model_path).read_bytes()
    if _external_tensors(onnx.load_model_from_string(model_bytes)):
        raise ValueError("Frozen bundles require embedded ONNX weights, not external tensor files")
    session = ort.InferenceSession(model_bytes, sess_options=options,
                                   providers=[provider], enable_fallback=False)
    session.disable_fallback()
    if session.get_providers() != [provider]:
        raise RuntimeError(f"Provider fallback is forbidden: {session.get_providers()}")
    return session


def _model_spec(session, size):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one image input and one embedding output")
    source, target = inputs[0], outputs[0]
    if source.type != "tensor(float)" or target.type != "tensor(float)":
        raise ValueError("Expected float32 ONNX input and output")
    if len(source.shape) != 4 or len(target.shape) != 2:
        raise ValueError("Expected NCHW images and 2D embeddings")
    if any(isinstance(actual, int) and actual != expected
           for actual, expected in zip(source.shape[1:], (3, size, size))):
        raise ValueError("ONNX input shape does not match preprocessing")
    dimension = None
    for count in (1, 3):
        probe = np.random.default_rng(0).normal(size=(count, 3, size, size)).astype(np.float32)
        result = session.run([target.name], {source.name: probe})[0]
        if result.ndim != 2 or result.shape[0] != count or result.shape[1] < 1:
            raise ValueError(f"Unexpected model output shape: {result.shape}")
        normalize(result)
        if dimension is not None and result.shape[1] != dimension:
            raise ValueError("Embedding dimension changes with batch size")
        dimension = result.shape[1]
    return {"input": source.name, "output": target.name, "dimension": dimension}


def _preprocessing(image_size, resize_mode):
    if type(image_size) is not int or image_size <= 0 or resize_mode not in {"square", "letterbox"}:
        raise ValueError("Invalid frozen preprocessing")
    return {"image_size": image_size, "resize_mode": resize_mode, "interpolation": "bilinear",
            "decode": "EXIF-oriented RGB", "crop": "organizer xywh before resize",
            "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
            "scale": 255, "layout": "NCHW", "l2_normalized": True, "flip_tta": False}


def _validate_policy(bundle):
    prep = bundle["preprocessing"]
    if bundle["schema"] != 1 or prep != _preprocessing(prep["image_size"], prep["resize_mode"]):
        raise ValueError("Unsupported frozen preprocessing/schema")
    if bundle["reranking"] != RERANKING or bundle["confidence"] != CONFIDENCE:
        raise ValueError("Unsupported ranking or confidence policy")
    if not np.isfinite(bundle["threshold"]):
        raise ValueError("Frozen threshold must be finite")
    provenance = bundle["calibration"]
    if (provenance.get("split") != "calibration" or not provenance.get("method")
            or not re.fullmatch(r"[0-9a-f]{64}", provenance.get("protocol_sha256", ""))):
        raise ValueError("Threshold needs calibration-only method and protocol SHA256")


def write_bundle(path, model_path, *, image_size, resize_mode, threshold, calibration):
    """Freeze an already calibrated model; provenance is supplied by the training runner."""
    path, model_path = Path(path).resolve(), Path(model_path).resolve()
    bundle = {"schema": 1, "model": {"path": os.path.relpath(model_path, path.parent),
              "sha256": sha256(model_path)}, "preprocessing": _preprocessing(image_size, resize_mode),
              "threshold": float(threshold), "confidence": CONFIDENCE,
              "reranking": dict(RERANKING), "calibration": calibration}
    _validate_policy(bundle)
    bundle["model"].update(_model_spec(_session(model_path, "CPUExecutionProvider"), image_size))
    if path.exists():
        if json.loads(path.read_text()) != bundle:
            raise ValueError("Refusing to replace an existing frozen bundle")
    else:
        write_json(path, bundle)
    return bundle


class FrozenEncoder:
    def __init__(self, bundle_path, provider="CPUExecutionProvider"):
        self.bundle_path = Path(bundle_path).resolve()
        self.bundle = json.loads(self.bundle_path.read_text())
        _validate_policy(self.bundle)
        model = self.bundle["model"]
        model_path = (self.bundle_path.parent / model["path"]).resolve()
        self.model_sha256 = sha256(model_path)
        if self.model_sha256 != model["sha256"]:
            raise ValueError("Frozen ONNX checksum mismatch")
        self.provider = provider
        self.session = _session(model_path, provider)
        self.size = self.bundle["preprocessing"]["image_size"]
        spec = _model_spec(self.session, self.size)
        if any(spec[key] != model[key] for key in spec):
            raise ValueError("Frozen ONNX input/output/dimension mismatch")
        self.dimension, self.input_name, self.output_name = spec["dimension"], spec["input"], spec["output"]
        self.fingerprint = hashlib.sha256(json.dumps(self.bundle, sort_keys=True).encode()).hexdigest()

    def preprocess(self, image, box):
        crop = resize_crop(crop_image(image, box), self.bundle["preprocessing"]["resize_mode"], self.size)
        pixels = (np.asarray(crop, dtype=np.float32) / np.float32(255) - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(pixels.transpose(2, 0, 1))

    def encode_batch(self, batch):
        inputs = np.asarray(batch, dtype=np.float32)
        if inputs.ndim != 4 or inputs.shape[1:] != (3, self.size, self.size) or not len(inputs):
            raise ValueError("Invalid image batch shape")
        if not np.isfinite(inputs).all():
            raise ValueError("Non-finite image batch")
        output = self.session.run([self.output_name], {self.input_name: inputs})[0]
        if output.shape != (len(inputs), self.dimension):
            raise ValueError(f"Unexpected embedding shape: {output.shape}")
        return normalize(output)

    def encode(self, image, box):
        return self.encode_batch([self.preprocess(image, box)])[0]


def _encode_rows(encoder, rows, dataset, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with Image.open(Path(dataset) / "images" / f"{row['image_id']}.jpg") as image:
                batch.append(encoder.preprocess(image, bbox(row)))
        vectors.append(encoder.encode_batch(batch))
    return np.concatenate(vectors)


def export_frozen(bundle_path, dataset, output, provider="CPUExecutionProvider", batch_size=16):
    """Official CSV semantics; no identity/camera use, train read, or calibration call."""
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty export directory; existing artifacts are preserved")
    encoder = FrozenEncoder(bundle_path, provider)
    queries = read_rows(dataset / "test_query.csv")
    gallery = read_rows(dataset / "test_gallery.csv")
    query_vectors = _encode_rows(encoder, queries, dataset, batch_size)
    gallery_vectors = _encode_rows(encoder, gallery, dataset, batch_size)
    reranker = (KReciprocalReranker(gallery_vectors, min(20, len(gallery) - 1), min(3, len(gallery)))
                if len(gallery) > 1 else None)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", np.concatenate([query_vectors, gallery_vectors]).astype(np.float32))
    with (output / "submission.csv").open("w", newline="") as submission, (output / "candidates.csv").open("w", newline="") as candidates:
        writer, accepted = csv.writer(submission), csv.writer(candidates)
        accepted.writerow(CANDIDATES_HEADER)
        for row, vector in zip(queries, query_vectors):
            scores = np.clip(gallery_vectors @ vector, -1, 1)
            distances = reranker.distances(vector, .5) if reranker else -scores
            order = np.argsort(distances, kind="stable")[:10]
            identifiers = [gallery[int(index)]["image_id"] for index in order]
            writer.writerow([row["image_id"], *identifiers])
            confidence = float(scores.max())
            if confidence >= encoder.bundle["threshold"]:
                accepted.writerow([row["image_id"], identifiers[0], (confidence + 1) / 2])
    validation = validate_artifacts(dataset, output)
    write_json(output / "export_manifest.json", {
        "bundle": encoder.bundle, "bundle_sha256": sha256(bundle_path), "provider": provider,
        "onnxruntime": ort.__version__, "fingerprint": encoder.fingerprint,
        "query_csv_sha256": sha256(dataset / "test_query.csv"),
        "gallery_csv_sha256": sha256(dataset / "test_gallery.csv"),
        "image_sha256": {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg") for r in queries + gallery},
        "embedding_ids": [r["image_id"] for r in queries + gallery], "validation": validation,
        "refusal_encoding": "no candidates.csv rows; submission still contains every query",
        "candidate_confidence": "(maximum raw gallery cosine + 1) / 2; not a probability"})
    return validation


def benchmark(bundle_path, dataset, provider="CPUExecutionProvider", samples=30, warmup=3):
    """Synchronous batch=1 including decode/crop/resize/transfers/forward/L2; excludes search."""
    if samples < 1 or warmup < 0:
        raise ValueError("Invalid benchmark sample counts")
    dataset = Path(dataset)
    encoder = FrozenEncoder(bundle_path, provider)
    rows = read_rows(dataset / "test_query.csv")
    times = []
    for index in range(warmup + samples):
        row = rows[index % len(rows)]
        started = time.perf_counter()
        with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
            encoder.encode(image, bbox(row))
        if index >= warmup:
            times.append((time.perf_counter() - started) * 1000)
    return {"provider": provider, "device": "CUDA" if provider == PROVIDERS[1] else "CPU",
            "platform": platform.platform(), "onnxruntime": ort.__version__, "batch": 1,
            "samples": samples, "warmup": warmup, "bundle_sha256": sha256(bundle_path),
            "includes": "JPEG decode + EXIF + bbox + resize + normalization + transfers + ONNX + L2",
            "excludes": "model loading, gallery search and reranking", "official_gpu_verified": False,
            "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def compare_providers(bundle_path, dataset, samples=8, atol=2e-4):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return {"status": "not-tested", "reason": "CUDAExecutionProvider unavailable"}
    if samples < 1:
        raise ValueError("samples must be positive")
    rows = read_rows(Path(dataset) / "test_query.csv")[:samples]
    cpu = _encode_rows(FrozenEncoder(bundle_path, PROVIDERS[0]), rows, dataset, samples)
    gpu = _encode_rows(FrozenEncoder(bundle_path, PROVIDERS[1]), rows, dataset, samples)
    error = float(np.max(np.abs(cpu - gpu)))
    return {"status": "passed" if np.allclose(cpu, gpu, atol=atol, rtol=0) else "failed",
            "max_abs_error": error, "atol": atol, "samples": len(rows), "official_gpu_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "benchmark", "parity"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provider", choices=PROVIDERS, default=PROVIDERS[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        if args.output is None:
            parser.error("export requires --output")
        result = export_frozen(args.bundle, args.dataset, args.output, args.provider)
    elif args.command == "benchmark":
        result = benchmark(args.bundle, args.dataset, args.provider)
    else:
        result = compare_providers(args.bundle, args.dataset)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
