"""Frozen image-only CLIP export and resumable inference for the OSNet ensemble audit."""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

from backend.core import DATASET, MODEL, Encoder, normalize, preprocess, sha256
from training.audit import digest, load_crop, opaque_mask
from training.clip_export import ImageOnly, export_smoke
from training.clip_reid import image_transform
from training.mask_calibration import _load, _save
from training.vendor.clip_reid.model import VisionTransformer


def fuse_arrays(osnet, clip, alpha):
    if not 0 <= alpha <= 1 or len(osnet) != len(clip):
        raise ValueError("Fusion requires aligned rows and alpha in [0, 1]")
    if alpha == 1:
        return normalize(osnet)
    if alpha == 0:
        return normalize(clip)
    o, c = normalize(osnet), normalize(clip)
    return normalize(np.concatenate([np.sqrt(alpha) * o, np.sqrt(1 - alpha) * c], axis=1))


class ClipEncoder:
    def __init__(self, path, expected_hash):
        if sha256(path) != expected_hash:
            raise ValueError("CLIP ONNX checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def encode_batch(self, tensors):
        vectors = self.session.run(["embeddings"], {"images": np.stack(tensors).astype(np.float32)})[0]
        if vectors.shape != (len(tensors), 1280):
            raise ValueError("Expected 1280-D image-only CLIP embeddings")
        return normalize(vectors)


def export_checkpoint(checkpoint_path, output, checkpoint_signature, sample_rows, classes, dataset=DATASET):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path, model_path = output / "export.json", output / "clip_trial_011.onnx"
    if report_path.exists():
        report = _load(report_path, checkpoint_signature)
        if sha256(model_path) != report["sha256"]:
            raise ValueError("Exported CLIP weights changed")
        return ClipEncoder(model_path, report["sha256"]), report
    print("Export frozen CLIP trial_011 epoch 33; check PyTorch/ONNX batches 1 and 2", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model"]["classifier.weight"].shape[0] != classes:
        raise ValueError("CLIP training identity count mismatch")
    if any("cv_embed" in k for k in checkpoint["model"]):
        raise ValueError("Camera/view embeddings are not allowed")
    visual = VisionTransformer(16, 16, 16, 16, 768, 12, 12, 512).float().eval()
    visual.load_state_dict({k.removeprefix("image_encoder."): v for k, v in checkpoint["model"].items()
                            if k.startswith("image_encoder.")}, strict=True)
    transform = image_transform(False)
    samples = torch.stack([transform(load_crop(r, dataset)) for r in sample_rows[:2]])
    temporary = output / "clip_trial_011.partial.onnx"
    report = export_smoke(SimpleNamespace(image_encoder=visual), samples, temporary)
    # Explicitly compare with the training embedding formula (before BN; concatenation then L2).
    with torch.no_grad():
        _, feature, projected = visual(samples)
        expected = torch.nn.functional.normalize(torch.cat([feature[:, 0], projected[:, 0]], dim=1), dim=1)
        np.testing.assert_allclose(ImageOnly(visual)(samples).numpy(), expected.numpy(), atol=1e-7)
    if report["bytes"] + MODEL.stat().st_size > 2_000_000_000:
        raise ValueError("Combined inference weights exceed the 2 GB limit")
    temporary.replace(model_path)
    report = {"signature": checkpoint_signature, **report, "model": model_path.name,
              "combined_osnet_clip_bytes": report["bytes"] + MODEL.stat().st_size}
    _save(report_path, report)
    del visual, checkpoint, samples
    return ClipEncoder(model_path, report["sha256"]), report


def encode_cached(encoder, rows, path, signature, annotations=None, dataset=DATASET, batch_size=16):
    """Masks, if supplied, affect the crop before the original CLIP 256x256 transform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = [r["image_id"] for r in rows]
    fingerprint = digest({"signature": signature, "rows": rows, "annotations": annotations})
    if len(set(expected_ids)) != len(rows) or (annotations is not None and set(annotations) != set(expected_ids)):
        raise ValueError("Need unique rows and exact manual mask coverage")
    ids, vectors = [], np.empty((0, 1280), dtype=np.float32)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            ids, vectors = saved["ids"].tolist(), saved["vectors"].copy()
            valid = str(saved["fingerprint"]) == fingerprint and ids == expected_ids[:len(ids)]
        if (not valid or vectors.shape != (len(ids), 1280) or vectors.dtype != np.float32
                or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError("CLIP feature cache belongs to different inputs or is invalid")
    if len(ids) == len(rows):
        print(f"Reuse CLIP {path.name}: {len(ids)} crops", flush=True)
        return dict(zip(ids, vectors))
    transform = image_transform(False)
    started, initial = time.perf_counter(), len(ids)
    for start in range(initial, len(rows), batch_size):
        batch, tensors = rows[start:start + batch_size], []
        for row in batch:
            crop = load_crop(row, dataset)
            if annotations is not None:
                crop = opaque_mask(crop, annotations[row["image_id"]]["rectangles"])
            tensors.append(transform(crop).numpy())
        vectors = np.concatenate([vectors, encoder.encode_batch(tensors)])
        ids.extend(r["image_id"] for r in batch)
        # Checkpoint every batch: interrupt loses no more than one batch, no full rerun.
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, ids=np.asarray(ids), vectors=vectors, fingerprint=fingerprint)
        temporary.replace(path)
        if len(ids) % 128 == 0 or len(ids) == len(rows):
            elapsed = time.perf_counter() - started
            eta = elapsed / (len(ids) - initial) * (len(rows) - len(ids))
            print(f"CLIP {path.name}: {len(ids)}/{len(rows)} | {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)
    return dict(zip(ids, vectors))


def benchmark_extract(clip_encoder, rows, alpha, dataset=DATASET):
    """End-to-end local CPU comparison, not the organizer RTX A5000 measurement."""
    osnet = Encoder()
    transform = image_transform(False)

    def extract(batch, weight):
        crops = [load_crop(r, dataset) for r in batch]
        o = osnet.encode_batch([preprocess(c, (0, 0, *c.size)) for c in crops]) if weight > 0 else None
        c = clip_encoder.encode_batch([transform(c).numpy() for c in crops]) if weight < 1 else None
        if weight == 1:
            return o
        if weight == 0:
            return c
        return fuse_arrays(o, c, weight)

    results = {}
    for name, weight in (("osnet", 1.), ("clip", 0.), ("selected", alpha)):
        for _ in range(2):
            extract(rows[:1], weight)
        elapsed = []
        for row in rows[:10]:
            start = time.perf_counter()
            extract([row], weight)
            elapsed.append((time.perf_counter() - start) * 1000)
        batch = rows[:16]
        start = time.perf_counter()
        for _ in range(3):
            extract(batch, weight)
        seconds = time.perf_counter() - start
        results[name] = {"batch1_samples": len(elapsed), "median_ms": float(np.median(elapsed)),
            "p95_ms": float(np.percentile(elapsed, 95)), "throughput_batch": len(batch),
            "throughput_fps": 3 * len(batch) / seconds, "device": "ONNX CPU, 2 threads",
            "includes": "JPEG decode + EXIF/BBox + per-model resize/normalize + encoder(s) + fusion/L2",
            "official_A5000_measurement": False, "VRAM_bytes": None}
    return results
