"""Gallery storage interfaces and the SQLite reference implementation."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np


class GalleryRepository(Protocol):
    """Persistent storage for gallery vectors and cache metadata."""

    def load(self, fingerprint: str, image_ids: list[str], dimension: int) -> np.ndarray | None:
        """Return vectors in image_ids order, or None when the cache is stale."""

    def replace(self, fingerprint: str, rows: list[dict], vectors: np.ndarray) -> None:
        """Atomically replace the persisted gallery with vectors in CSV order."""


class SQLiteGalleryRepository:
    """SQLite reference storage retained for baseline/parity testing."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path)
        db.execute("CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS gallery (position INTEGER PRIMARY KEY, image_id TEXT UNIQUE, metadata TEXT, embedding BLOB)")
        return db

    def load(self, fingerprint: str, image_ids: list[str], dimension: int) -> np.ndarray | None:
        with self._connect() as db:
            saved = db.execute("SELECT value FROM info WHERE key='fingerprint'").fetchone()
            stored = db.execute("SELECT image_id, embedding FROM gallery ORDER BY position").fetchall()
        if saved != (fingerprint,) or [item[0] for item in stored] != image_ids:
            return None
        vectors = np.stack([np.frombuffer(item[1], dtype="<f4") for item in stored])
        if vectors.shape != (len(image_ids), dimension) or not np.isfinite(vectors).all():
            raise ValueError("Invalid gallery cache; remove artifacts/gallery.sqlite3 and restart")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Gallery cache contains non-normalized vectors")
        return vectors

    def replace(self, fingerprint: str, rows: list[dict], vectors: np.ndarray) -> None:
        if len(rows) != len(vectors):
            raise ValueError("Gallery rows and vectors must have the same length")
        with self._connect() as db:
            db.execute("DELETE FROM gallery")
            db.executemany(
                "INSERT INTO gallery VALUES (?, ?, ?, ?)",
                [
                    (position, row["image_id"], json.dumps(row), vector.astype("<f4").tobytes())
                    for position, (row, vector) in enumerate(zip(rows, vectors))
                ],
            )
            db.execute("INSERT OR REPLACE INTO info VALUES ('fingerprint', ?)", (fingerprint,))
