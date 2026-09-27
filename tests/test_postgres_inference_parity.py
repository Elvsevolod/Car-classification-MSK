"""Run only against the dedicated PostgreSQL test database (gallery tables are replaced)."""

import csv
import os

import numpy as np
from PIL import Image
import pytest

from backend.calibration import load_calibration
from backend.core import Encoder, Gallery, encode_rows, read_rows
from backend.database import DatabaseSettings
from backend.gallery_repository import InMemoryGalleryRepository
from backend.postgres_gallery_repository import PostgresGalleryRepository


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="Dedicated PostgreSQL test database required")
def test_actual_encoder_in_memory_and_postgres_have_same_top10_and_refusal(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    rows = []
    for index in range(12):
        image_id = f"{index:032x}"
        pixels = np.random.default_rng(index + 300).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(dataset / "images" / f"{image_id}.jpg")
        rows.append({"image_id": image_id, "x": 3, "y": 4, "w": 55, "h": 40})
    for filename, selected in (("test_gallery", rows[:10]), ("test_query", rows[10:])):
        with (dataset / f"{filename}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)

    encoder = Encoder()
    threshold = load_calibration(encoder)["threshold"]
    memory = Gallery(encoder, dataset, InMemoryGalleryRepository())
    repository = PostgresGalleryRepository(DatabaseSettings(os.environ["DATABASE_URL"]))
    persisted = Gallery(encoder, dataset, repository)
    # Exercise the serialized pgvector cache, not only the pre-insert vectors.
    persisted = Gallery(encoder, dataset, repository)
    np.testing.assert_array_equal(persisted.vectors, memory.vectors)
    queries = encode_rows(encoder, read_rows(dataset / "test_query.csv"), dataset)
    decisions = set()
    # The negative of an actual normalized embedding adds a deterministic refusal case.
    for query in [*queries, -queries[0]]:
        np.testing.assert_allclose(memory._raw_scores(query), persisted._raw_scores(query), atol=2e-6, rtol=0)
        expected, expected_confidence = memory.search_with_confidence(query, 10)
        actual, actual_confidence = persisted.search_with_confidence(query, 10)
        assert [item["image_id"] for item in actual] == [item["image_id"] for item in expected]
        assert len(actual) == 10
        assert abs(actual_confidence - expected_confidence) <= 2e-6
        expected_candidates, _ = memory.search_with_confidence(query, 10, threshold)
        actual_candidates, _ = persisted.search_with_confidence(query, 10, threshold)
        assert [item["image_id"] for item in actual_candidates] == [item["image_id"] for item in expected_candidates]
        decisions.add(bool(actual_candidates))
    assert decisions == {False, True}
