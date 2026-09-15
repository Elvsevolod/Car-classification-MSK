import csv
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ort.disable_telemetry_events()

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("DATASET_DIR", ROOT / "dataset"))
ARTIFACTS = ROOT / "artifacts"
MODEL = ROOT / "models/osnet_ain_x1_0_vehicle_reid.onnx"
MODEL_SHA384 = "0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7"
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
        if not re.fullmatch(r"[0-9a-f]{32}", identifier) or identifier in seen:
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


def rank(scores, top_k):
    """Exact descending search; CSV order breaks equal-score ties."""
    return np.argsort(-scores, kind="stable")[:top_k]


class Encoder:
    def __init__(self, model_path=MODEL):
        with open(model_path, "rb") as stream:
            checksum = hashlib.file_digest(stream, "sha384").hexdigest()
        if checksum != MODEL_SHA384:
            raise ValueError("OSNet checksum mismatch; expected the official unchanged checkpoint")
        self.model_sha256 = sha256(model_path)
        self.fingerprint = hashlib.sha256((self.model_sha256 + PREPROCESS).encode()).hexdigest()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def encode_batch(self, batch):
        output = self.session.run(["output"], {self.input_name: np.stack(batch)})[0]
        if output.shape != (len(batch), 512):
            raise ValueError(f"Unexpected model output: {output.shape}")
        return normalize(output)

    def encode(self, image, box):
        return self.encode_batch([preprocess(image, box)])[0]


def encode_rows(encoder, rows, dataset=DATASET, batch_size=16):
    result = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
                batch.append(preprocess(image, bbox(row)))
        result.append(encoder.encode_batch(batch))
        if start % (batch_size * 10) == 0:
            print(f"OSNet: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(result)


class Gallery:
    """Persistent SQLite metadata/vectors, exact cosine search in memory."""

    def __init__(self, encoder, dataset=DATASET, db_path=ARTIFACTS / "gallery.sqlite3"):
        self.rows = read_rows(dataset / "test_gallery.csv")
        self.dataset = dataset
        signature = hashlib.sha256(encoder.fingerprint.encode())
        signature.update(json.dumps(self.rows, sort_keys=True).encode())
        for row in self.rows:
            signature.update(sha256(dataset / "images" / f"{row['image_id']}.jpg").encode())
        self.fingerprint = signature.hexdigest()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS gallery (position INTEGER PRIMARY KEY, image_id TEXT UNIQUE, metadata TEXT, embedding BLOB)")
            saved = db.execute("SELECT value FROM info WHERE key='fingerprint'").fetchone()
            stored = db.execute("SELECT image_id, embedding FROM gallery ORDER BY position").fetchall()
            if saved == (self.fingerprint,) and [r[0] for r in stored] == [r["image_id"] for r in self.rows]:
                self.vectors = np.stack([np.frombuffer(r[1], dtype="<f4") for r in stored])
                if self.vectors.shape != (len(self.rows), 512) or not np.isfinite(self.vectors).all():
                    raise ValueError("Invalid gallery cache; remove artifacts/gallery.sqlite3 and restart")
                if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
                    raise ValueError("Gallery cache contains non-normalized vectors")
            else:
                self.vectors = encode_rows(encoder, self.rows, dataset)
                db.execute("DELETE FROM gallery")
                db.executemany("INSERT INTO gallery VALUES (?, ?, ?, ?)", [
                    (i, row["image_id"], json.dumps(row), vector.astype("<f4").tobytes())
                    for i, (row, vector) in enumerate(zip(self.rows, self.vectors))
                ])
                db.execute("INSERT OR REPLACE INTO info VALUES ('fingerprint', ?)", (self.fingerprint,))

    def search(self, vector, top_k=10, threshold=None):
        scores = np.clip(self.vectors @ normalize(vector), -1, 1)
        selected = rank(scores, top_k)
        if threshold is not None:
            selected = selected[scores[selected] >= threshold]
        return [{"rank": i + 1, **self.rows[int(index)], "similarity": float(scores[index]),
                 "crop_url": f"/api/images/gallery/{self.rows[int(index)]['image_id']}?crop=true"}
                for i, index in enumerate(selected)]
