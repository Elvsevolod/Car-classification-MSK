"""Immutable, fingerprint-addressed cache spaces. Publication is atomic."""
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np


def metadata(fingerprint, rows, vectors, state):
    from .runtime import validate_vectors
    validate_vectors(vectors, len(rows), vectors.shape[1])
    return {"fingerprint": fingerprint, "ids": [r["image_id"] for r in rows],
            "dimension": vectors.shape[1], "state": asdict(state),
            "vectors_sha256": hashlib.sha256(vectors.astype("<f4").tobytes()).hexdigest()}


def checked(saved, vectors, fingerprint, ids, dimension, state):
    from .runtime import validate_vectors
    if (saved["fingerprint"] != fingerprint or saved["ids"] != ids
            or saved["dimension"] != dimension or saved["state"] != asdict(state)):
        raise ValueError("Incompatible cache space metadata")
    validate_vectors(vectors, len(ids), dimension)
    if hashlib.sha256(vectors.astype("<f4").tobytes()).hexdigest() != saved["vectors_sha256"]:
        raise ValueError("Cache vectors checksum mismatch")
    return vectors


class FileGallerySpaces:
    """Local acceptance cache; production uses the same contract in PostgreSQL."""
    def __init__(self, directory):
        self.directory = Path(directory)

    def load(self, fingerprint, image_ids, dimension, state):
        path = self.directory / f"{fingerprint}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return checked(json.loads(str(data["metadata"])), data["vectors"].copy(),
                           fingerprint, image_ids, dimension, state)

    def replace(self, fingerprint, rows, vectors, state):
        saved = metadata(fingerprint, rows, vectors, state)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{fingerprint}.npz"
        existing = self.load(fingerprint, saved["ids"], vectors.shape[1], state)
        if existing is not None:
            if not np.array_equal(existing, vectors):
                raise ValueError("Refusing to overwrite a different immutable cache space")
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, vectors=vectors, metadata=json.dumps(saved, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic link refuses competing publishers, keeping a complete immutable first build.
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.load(fingerprint, saved["ids"], vectors.shape[1], state)
                if not np.array_equal(existing, vectors):
                    raise ValueError("Concurrent cache build produced different embeddings")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class PostgresGallerySpaces:
    """New dimension-aware tables only; never write legacy gallery_items/gallery_state."""
    def __init__(self, settings):
        self.database_url = settings.url

    def _connect(self):
        import psycopg
        return psycopg.connect(self.database_url, connect_timeout=5)

    def load(self, fingerprint, image_ids, dimension, state):
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT metadata FROM reid_gallery_spaces WHERE fingerprint=%s", (fingerprint,))
            saved = cursor.fetchone()
            if saved is None:
                return None
            cursor.execute("SELECT image_id, embedding::text FROM reid_gallery_vectors "
                           "WHERE fingerprint=%s ORDER BY position", (fingerprint,))
            rows = cursor.fetchall()
        if [r[0] for r in rows] != image_ids:
            raise ValueError("Corrupt gallery space order")
        vectors = np.stack([np.fromstring(r[1][1:-1], sep=",", dtype=np.float32) for r in rows])
        return checked(saved[0], vectors, fingerprint, image_ids, dimension, state)

    def replace(self, fingerprint, rows, vectors, state):
        saved = metadata(fingerprint, rows, vectors, state)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (fingerprint,))
            cursor.execute("SELECT metadata FROM reid_gallery_spaces WHERE fingerprint=%s", (fingerprint,))
            existing = cursor.fetchone()
            if existing:
                if existing[0] != saved:
                    raise ValueError("Conflicting immutable gallery space")
                return
            cursor.execute("INSERT INTO reid_gallery_spaces (fingerprint, dimension, metadata) "
                           "VALUES (%s, %s, %s::jsonb)", (fingerprint, vectors.shape[1], json.dumps(saved)))
            cursor.executemany(
                "INSERT INTO reid_gallery_vectors (fingerprint, dimension, position, image_id, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector)",
                [(fingerprint, vectors.shape[1], i, row["image_id"],
                  "[" + ",".join(str(float(x)) for x in vector) + "]")
                 for i, (row, vector) in enumerate(zip(rows, vectors))])
            # Commit publishes metadata + all vectors together. Failed builds roll back in full.
