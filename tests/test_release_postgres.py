"""Only the explicitly named disposable integration-test database may be used."""
import os
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import pytest

from backend.cache_spaces import PostgresGallerySpaces
from backend.database import DatabaseSettings
from backend.gallery_repository import GalleryBuildState
from backend.runtime import Runtime, RuntimeGallery, encode_rows
from backend.core import read_rows
from tests.test_release_runtime import dataset, decision_ids


@pytest.fixture
def repository():
    url = os.environ.get("REID_TEST_DATABASE_URL")
    if not url:
        pytest.skip("REID_TEST_DATABASE_URL must name a dedicated disposable database")
    if urlparse(url).path != "/reid_integration_test":
        raise ValueError("Refusing any database except reid_integration_test")
    return PostgresGallerySpaces(DatabaseSettings(url))


def test_dimension_spaces_pg_parity_and_legacy_rollback(repository, dataset):
    with repository._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT row_to_json(t)::text FROM gallery_state t ORDER BY id")
        old_state = cursor.fetchall()
        cursor.execute("SELECT row_to_json(t)::text FROM gallery_items t ORDER BY position")
        old_items = cursor.fetchall()
    previous = {}
    for profile in ("MVP_legacy", "RC_R1_equal3_v18", "RC_R1_single_v18", "MVP_dual_role_v24", "MVP_fusion_v25", "MVP_dual_role_v24", "MVP_legacy"):
        runtime = Runtime(profile)
        memory = RuntimeGallery(runtime, dataset)
        persisted = RuntimeGallery(runtime, dataset, repository)
        reloaded = RuntimeGallery(runtime, dataset, repository)
        assert reloaded.cache_hit
        np.testing.assert_array_equal(memory.vectors, reloaded.vectors)
        queries = encode_rows(runtime, read_rows(dataset / "test_query.csv"), dataset)
        decisions = []
        for vector in [*queries, -queries[0]]:
            a, b = memory.decide(vector), reloaded.decide(vector)
            assert decision_ids(a) == decision_ids(b)
            assert a["confidence"] == b["confidence"]
            decisions.append(decision_ids(b))
        if profile in previous:
            assert decisions == previous[profile]
        previous[profile] = decisions
    with repository._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT row_to_json(t)::text FROM gallery_state t ORDER BY id")
        assert cursor.fetchall() == old_state
        cursor.execute("SELECT row_to_json(t)::text FROM gallery_items t ORDER BY position")
        assert cursor.fetchall() == old_items


def test_failed_build_not_published(repository):
    import psycopg
    fingerprint = uuid4().hex
    state = GalleryBuildState("encoder", "preprocess", "csv", {"duplicate": "image"})
    rows = [{"image_id": "duplicate"}, {"image_id": "duplicate"}]
    with pytest.raises(psycopg.errors.UniqueViolation):
        repository.replace(fingerprint, rows, np.eye(2, dtype=np.float32), state)
    assert repository.load(fingerprint, ["duplicate"], 2, state) is None
