import os

import numpy as np

from backend.database import DatabaseSettings
from backend.gallery_repository import GalleryBuildState
from backend.postgres_gallery_repository import PostgresGalleryRepository


def test_postgres_gallery_round_trip_exact_search_and_cache_invalidation():
    repository = PostgresGalleryRepository(DatabaseSettings(os.environ["DATABASE_URL"]))
    assert repository.ping()["pgvector_version"]

    rows = [
        {"image_id": "gallery-a", "x": 0, "y": 0, "w": 10, "h": 10},
        {"image_id": "gallery-b", "x": 1, "y": 1, "w": 10, "h": 10},
    ]
    vectors = np.zeros((2, 512), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    state = GalleryBuildState(
        encoder_fingerprint="encoder-test",
        preprocessing_fingerprint="preprocess-test",
        csv_sha256="csv-test",
        image_sha256={"gallery-a": "hash-a", "gallery-b": "hash-b"},
    )

    repository.replace("gallery-test", rows, vectors, state)
    loaded = repository.load("gallery-test", [row["image_id"] for row in rows], 512, state)
    np.testing.assert_array_equal(loaded, vectors)

    matches = repository.search_cosine(vectors[0], 2)
    assert [position for position, _ in matches] == [0, 1]
    assert matches[0][1] == 1.0
    assert matches[1][1] == 0.0

    changed = GalleryBuildState(
        encoder_fingerprint=state.encoder_fingerprint,
        preprocessing_fingerprint=state.preprocessing_fingerprint,
        csv_sha256="changed-csv",
        image_sha256=state.image_sha256,
    )
    assert repository.load("gallery-test", [row["image_id"] for row in rows], 512, changed) is None
