import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from .database import DatabaseSettings
from .gallery_repository import GalleryRepository, SQLiteGalleryRepository
from .postgres_gallery_repository import PostgresGalleryRepository

ort.disable_telemetry_events()

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("DATASET_DIR", ROOT / "dataset"))
ARTIFACTS = ROOT / "artifacts"
STOCK_MODEL = ROOT / "models/osnet_ain_x1_0_vehicle_reid.onnx"
STOCK_MODEL_SHA384 = "0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7"
MODEL = ROOT / "models/osnet_ain_x1_0_vehicle_reid_hpo_best_map.onnx"
MODEL_SHA384 = "4832ca8134b31f84ec52b9a6a72aa90f9e55e0b8ab7d52d02820040536677d712df39b9c9604d6da21ee7bc823db6818"
MODEL_NAME = "OSNet-AIN x1.0 / HPO best-mAP epoch 5"
MODEL_FINE_TUNED = True
MODEL_TRAINING_EPOCH = 5
PREPROCESS = "exif-rgb-bbox-bilinear208-imagenet-l2-v1"


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image_id", "x", "y", "w", "h"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing annotation columns: {path}")
        rows = list(reader)
    seen = set()
    for row in rows:
        identifier = row["image_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen:
            raise ValueError(f"Invalid or duplicate image_id: {identifier}")
        seen.add(identifier)
        for key in ("x", "y", "w", "h", "vehicle_id", "camera_id"):
            if key in row:
                row[key] = int(row[key])
    if not rows:
        raise ValueError(f"Empty annotations: {path}")
    return rows


def bbox(row):
    return tuple(row[key] for key in ("x", "y", "w", "h"))


def crop_image(image, box):
    # BBox uses the decoded, EXIF-oriented full image, before resizing.
    image = ImageOps.exif_transpose(image).convert("RGB")
    x, y, w, h = box
    if any(not isinstance(v, (int, np.integer)) for v in box):
        raise ValueError("BBox must contain integer x, y, w, h")
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.width or y + h > image.height:
        raise ValueError(f"BBox {box} is outside image {image.width}×{image.height}")
    return image.crop((x, y, x + w, y + h))


def preprocess(image, box):
    crop = crop_image(image, box).resize((208, 208), Image.Resampling.BILINEAR)
    pixels = np.asarray(crop, dtype=np.float32) / np.float32(255)
    pixels = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
        raise ValueError("Model produced invalid or zero embeddings")
    return vectors / norms


def gallery_repository_from_environment(db_path):
    """Select the configured persistence backend without changing search semantics."""
    storage = os.environ.get("GALLERY_STORAGE", "sqlite").strip().lower()
    if storage == "sqlite":
        return SQLiteGalleryRepository(db_path)
    if storage == "postgres":
        return PostgresGalleryRepository(DatabaseSettings.from_environment())
    raise ValueError(f"Unsupported GALLERY_STORAGE: {storage}")


def rank(scores, top_k):
    """Exact descending search; CSV order breaks equal-score ties."""
    return np.argsort(-scores, kind="stable")[:top_k]


class Encoder:
    def __init__(self, model_path=MODEL):
        model_path = Path(model_path)
        expected_checksum = {
            MODEL.resolve(): MODEL_SHA384,
            STOCK_MODEL.resolve(): STOCK_MODEL_SHA384,
        }.get(model_path.resolve())
        if expected_checksum is None:
            raise ValueError(f"Unrecognized OSNet checkpoint: {model_path}")
        with open(model_path, "rb") as stream:
            checksum = hashlib.file_digest(stream, "sha384").hexdigest()
        if checksum != expected_checksum:
            raise ValueError("OSNet checksum mismatch for bundled checkpoint")
        self.model_sha256 = sha256(model_path)
        self.fingerprint = hashlib.sha256((self.model_sha256 + PREPROCESS).encode()).hexdigest()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def encode_batch(self, batch, flip_tta=False):
        inputs = np.stack(batch)
        if flip_tta:
            inputs = np.concatenate([inputs, np.ascontiguousarray(inputs[..., ::-1])])
        output = self.session.run(["output"], {self.input_name: inputs})[0]
        expected = len(batch) * (2 if flip_tta else 1)
        if output.shape != (expected, 512):
            raise ValueError(f"Unexpected model output: {output.shape}")
        if flip_tta:
            original, mirrored = np.split(normalize(output), 2)
            output = original + mirrored
        return normalize(output)

    def encode(self, image, box, flip_tta=False):
        return self.encode_batch([preprocess(image, box)], flip_tta)[0]


def encode_rows(encoder, rows, dataset=DATASET, batch_size=16, flip_tta=False):
    result = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
                batch.append(preprocess(image, bbox(row)))
        result.append(encoder.encode_batch(batch, flip_tta))
        if start % (batch_size * 10) == 0:
            print(f"OSNet: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(result)




class Gallery:
    """Gallery vectors plus the unchanged in-memory streaming reranker."""

    def __init__(self, encoder, dataset=DATASET, db_path=ARTIFACTS / "gallery.sqlite3",
                 repository: GalleryRepository | None = None):
        self.rows = read_rows(dataset / "test_gallery.csv")
        self.dataset = dataset
        signature = hashlib.sha256(encoder.fingerprint.encode())
        signature.update(json.dumps(self.rows, sort_keys=True).encode())
        for row in self.rows:
            signature.update(sha256(dataset / "images" / f"{row['image_id']}.jpg").encode())
        self.fingerprint = signature.hexdigest()
        self.repository = repository or gallery_repository_from_environment(db_path)
        self.vectors = self.repository.load(
            self.fingerprint, [row["image_id"] for row in self.rows], 512
        )
        if self.vectors is None:
            self.vectors = encode_rows(encoder, self.rows, dataset)
            self.repository.replace(self.fingerprint, self.rows, self.vectors)
        if len(self.rows) > 1:
            from .rerank import ACTIVE_K1, ACTIVE_K2, KReciprocalReranker
            self.reranker = KReciprocalReranker(
                self.vectors, min(ACTIVE_K1, len(self.rows) - 1), min(ACTIVE_K2, len(self.rows))
            )
        else:
            self.reranker = None

    def confidence(self, vector):
        """Maximum raw cosine used only for the calibrated refusal decision."""
        return float(np.max(self.vectors @ normalize(vector)))

    def search(self, vector, top_k=10, threshold=None):
        vector = normalize(vector)
        scores = np.clip(self.vectors @ vector, -1, 1)
        if threshold is not None and float(np.max(scores)) < threshold:
            return []
        if self.reranker is None:
            selected = rank(scores, top_k)
            rerank_scores = scores
        else:
            from .rerank import ACTIVE_LAMBDA
            distances = self.reranker.distances(vector, ACTIVE_LAMBDA)
            selected = np.argsort(distances, kind="stable")[:top_k]
            rerank_scores = -distances
        return [{"rank": index + 1, **self.rows[int(row_index)], "similarity": float(scores[row_index]),
                 "rerank_score": float(rerank_scores[row_index]),
                 "crop_url": f"/api/images/gallery/{self.rows[int(row_index)]['image_id']}?crop=true"}
                for index, row_index in enumerate(selected)]
