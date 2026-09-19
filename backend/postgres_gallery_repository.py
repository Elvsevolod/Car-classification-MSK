"""PostgreSQL/pgvector gallery repository.

The repository deliberately returns the complete gallery in CSV position order.
The existing full-gallery k-reciprocal reranker therefore keeps exactly the same
behaviour as the SQLite baseline. Its tables are created by Alembic in TASK 3.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import psycopg
from psycopg.errors import UndefinedTable

if TYPE_CHECKING:
    from .database import DatabaseSettings


class PostgresGalleryRepository:
    """Persistent gallery storage backed by PostgreSQL and pgvector."""

    def __init__(self, settings: "DatabaseSettings"):
        self.database_url = settings.url

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=5)

    def ping(self) -> dict[str, str]:
        """Verify connectivity and that the pgvector extension is installed."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("pgvector extension is not installed; run database migrations")
        return {"pgvector_version": str(row[0])}

    @staticmethod
    def _vector_from_text(value: str, dimension: int) -> np.ndarray:
        vector = np.fromstring(value.strip()[1:-1], dtype=np.float32, sep=",")
        if vector.shape != (dimension,):
            raise ValueError(f"Invalid pgvector dimension: expected {dimension}, got {vector.shape}")
        return vector

    def load(self, fingerprint: str, image_ids: list[str], dimension: int) -> np.ndarray | None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT fingerprint FROM gallery_state WHERE id = 1")
                state = cursor.fetchone()
                cursor.execute("SELECT image_id, embedding::text FROM gallery_items ORDER BY position")
                stored = cursor.fetchall()
        except UndefinedTable as exc:
            raise RuntimeError("PostgreSQL schema is missing; run 'alembic upgrade head'") from exc

        if state != (fingerprint,) or [item[0] for item in stored] != image_ids:
            return None
        vectors = np.stack([self._vector_from_text(item[1], dimension) for item in stored])
        if vectors.shape != (len(image_ids), dimension) or not np.isfinite(vectors).all():
            raise ValueError("Invalid PostgreSQL gallery cache")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("PostgreSQL gallery contains non-normalized vectors")
        return vectors

    def replace(self, fingerprint: str, rows: list[dict], vectors: np.ndarray) -> None:
        if len(rows) != len(vectors):
            raise ValueError("Gallery rows and vectors must have the same length")
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("DELETE FROM gallery_items")
                cursor.executemany(
                    """
                    INSERT INTO gallery_items (position, image_id, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s::vector)
                    """,
                    [
                        (position, row["image_id"], json.dumps(row), self._vector_literal(vector))
                        for position, (row, vector) in enumerate(zip(rows, vectors))
                    ],
                )
                cursor.execute(
                    """
                    INSERT INTO gallery_state (id, fingerprint)
                    VALUES (1, %s)
                    ON CONFLICT (id) DO UPDATE SET fingerprint = EXCLUDED.fingerprint
                    """,
                    (fingerprint,),
                )
        except UndefinedTable as exc:
            raise RuntimeError("PostgreSQL schema is missing; run 'alembic upgrade head'") from exc

    @staticmethod
    def _vector_literal(vector: np.ndarray) -> str:
        normalized = np.asarray(vector, dtype=np.float32)
        return "[" + ",".join(str(float(value)) for value in normalized) + "]"
