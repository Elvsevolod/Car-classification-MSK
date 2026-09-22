"""Постоянное PostgreSQL + pgvector хранилище эмбеддингов gallery и cosine-поиска."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import psycopg
from psycopg.errors import UndefinedTable

from .gallery_repository import GalleryBuildState

if TYPE_CHECKING:
    from .database import DatabaseSettings


class PostgresGalleryRepository:
    """Хранит gallery в PostgreSQL и возвращает полный exact cosine-рейтинг для неизменного reranking."""

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

    def load(
        self, fingerprint: str, image_ids: list[str], dimension: int, state: GalleryBuildState
    ) -> np.ndarray | None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT fingerprint, encoder_fingerprint, preprocessing_fingerprint, csv_sha256, gallery_size
                    FROM gallery_state WHERE id = 1
                    """
                )
                saved_state = cursor.fetchone()
                cursor.execute("SELECT image_id, embedding::text FROM gallery_items ORDER BY position")
                stored = cursor.fetchall()
        except UndefinedTable as exc:
            raise RuntimeError("PostgreSQL schema is missing; run 'alembic upgrade head'") from exc

        expected_state = (
            fingerprint,
            state.encoder_fingerprint,
            state.preprocessing_fingerprint,
            state.csv_sha256,
            len(image_ids),
        )
        if saved_state != expected_state or [item[0] for item in stored] != image_ids:
            return None
        vectors = np.stack([self._vector_from_text(item[1], dimension) for item in stored])
        if vectors.shape != (len(image_ids), dimension) or not np.isfinite(vectors).all():
            raise ValueError("Invalid PostgreSQL gallery cache")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("PostgreSQL gallery contains non-normalized vectors")
        return vectors

    def search_cosine(self, query_vector: np.ndarray, limit: int) -> list[tuple[int, float]]:
        """Exact pgvector cosine search with CSV position as a stable tie-breaker."""
        if limit < 1:
            raise ValueError("limit must be positive")
        vector_literal = self._vector_literal(query_vector)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT position, 1 - (embedding <=> %s::vector) AS similarity
                    FROM gallery_items
                    ORDER BY embedding <=> %s::vector, position ASC
                    LIMIT %s
                    """,
                    (vector_literal, vector_literal, limit),
                )
                return [(int(position), float(similarity)) for position, similarity in cursor.fetchall()]
        except UndefinedTable as exc:
            raise RuntimeError("PostgreSQL schema is missing; run 'alembic upgrade head'") from exc

    def replace(
        self, fingerprint: str, rows: list[dict], vectors: np.ndarray, state: GalleryBuildState
    ) -> None:
        if len(rows) != len(vectors):
            raise ValueError("Gallery rows and vectors must have the same length")
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("LOCK TABLE gallery_items, gallery_state IN EXCLUSIVE MODE")
                cursor.execute("DELETE FROM gallery_items")
                cursor.executemany(
                    """
                    INSERT INTO gallery_items
                        (position, image_id, x, y, w, h, embedding, image_sha256,
                         encoder_fingerprint, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s::jsonb)
                    """,
                    [
                        (
                            position,
                            row["image_id"],
                            row["x"], row["y"], row["w"], row["h"],
                            self._vector_literal(vector),
                            state.image_sha256[row["image_id"]],
                            state.encoder_fingerprint,
                            json.dumps(row),
                        )
                        for position, (row, vector) in enumerate(zip(rows, vectors))
                    ],
                )
                cursor.execute(
                    """
                    INSERT INTO gallery_state
                        (id, fingerprint, encoder_fingerprint, preprocessing_fingerprint,
                         csv_sha256, gallery_size, schema_version)
                    VALUES (1, %s, %s, %s, %s, %s, 1)
                    ON CONFLICT (id) DO UPDATE SET
                        fingerprint = EXCLUDED.fingerprint,
                        encoder_fingerprint = EXCLUDED.encoder_fingerprint,
                        preprocessing_fingerprint = EXCLUDED.preprocessing_fingerprint,
                        csv_sha256 = EXCLUDED.csv_sha256,
                        gallery_size = EXCLUDED.gallery_size,
                        built_at = CURRENT_TIMESTAMP,
                        schema_version = EXCLUDED.schema_version
                    """,
                    (
                        fingerprint,
                        state.encoder_fingerprint,
                        state.preprocessing_fingerprint,
                        state.csv_sha256,
                        len(rows),
                    ),
                )
        except UndefinedTable as exc:
            raise RuntimeError("PostgreSQL schema is missing; run 'alembic upgrade head'") from exc

    @staticmethod
    def _vector_literal(vector: np.ndarray) -> str:
        normalized = np.asarray(vector, dtype=np.float32)
        return "[" + ",".join(str(float(value)) for value in normalized) + "]"
