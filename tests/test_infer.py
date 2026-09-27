import csv
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from backend import core, evaluate, infer
from backend import runtime as shared
from backend.calibration import DEFAULT_CALIBRATION
from backend.gallery_repository import InMemoryGalleryRepository, SQLiteGalleryRepository


def make_dataset(root, gallery_count=12):
    (root / "images").mkdir(parents=True)
    queries = ["query-known", "query-unknown"]
    gallery = [f"gallery-{index}" for index in range(gallery_count)]
    for name, identifiers in (("test_query", queries), ("test_gallery", gallery)):
        with (root / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows([identifier, 0, 0, 24, 24] for identifier in identifiers)
    for index, identifier in enumerate(queries + gallery):
        pixels = np.random.default_rng(index).integers(0, 256, (24, 24, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"{identifier}.jpg")
    return queries, gallery


def test_batch_without_train_or_database_preserves_ranking_refusal_and_embedding_order(tmp_path, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    queries, gallery_ids = make_dataset(dataset)
    report = json.loads(DEFAULT_CALIBRATION.read_text())
    encoder = SimpleNamespace(model_sha256=report["model_sha256"], fingerprint=report["encoder_fingerprint"])
    vectors = {}
    for index, identifier in enumerate(gallery_ids):
        cosine = 1 - index * .05
        vector = np.zeros(512, dtype=np.float32)
        vector[:2] = cosine, np.sqrt(1 - cosine ** 2)
        vectors[identifier] = vector
    vectors[queries[0]] = vectors[gallery_ids[0]].copy()
    vectors[queries[1]] = -vectors[gallery_ids[0]]

    def encoded_rows(encoder, rows, dataset):
        return np.stack([vectors[row["image_id"]] for row in rows])

    def forbidden(*args, **kwargs):
        pytest.fail("Batch inference must not calibrate, evaluate train data, or create a DB repository")

    runtime = shared.Runtime("MVP_legacy")
    monkeypatch.setattr(infer, "Runtime", lambda *args: runtime)
    monkeypatch.setattr(shared, "encode_rows", lambda runtime, rows, dataset, *args: encoded_rows(encoder, rows, dataset))
    monkeypatch.setattr(core, "encode_rows", encoded_rows)
    monkeypatch.setattr(evaluate, "encode_rows", encoded_rows)
    monkeypatch.setattr(evaluate, "evaluate", forbidden)
    monkeypatch.setattr(evaluate, "calibrate", forbidden)
    monkeypatch.setattr(evaluate, "gallery_repository_from_environment", forbidden)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("GALLERY_STORAGE", "not-postgres")
    validation = infer.run_inference(dataset, output)

    assert not (dataset / "train.csv").exists()
    assert validation["accepted_queries"] == validation["refused_queries"] == 1
    actual_vectors = np.load(output / "embeddings.npy", allow_pickle=False)
    assert actual_vectors.dtype == np.float32
    np.testing.assert_array_equal(actual_vectors, np.stack([vectors[key] for key in queries + gallery_ids]))
    with (output / "submission.csv").open(newline="") as stream:
        submission = list(csv.reader(stream))
    assert [row[0] for row in submission] == queries
    assert all(len(row) == 11 and len(set(row[1:])) == 10 for row in submission)
    reference = core.Gallery(encoder, dataset, SQLiteGalleryRepository(tmp_path / "reference.db"))
    for query, row in zip(queries, submission):
        assert row[1:] == [item["image_id"] for item in reference.search(vectors[query], 10)]
    with (output / "candidates.csv").open(newline="") as stream:
        candidates = list(csv.DictReader(stream))
    assert [row["query_id"] for row in candidates] == [queries[0]]
    assert candidates[0]["gallery_id"] == submission[0][1]
    assert float(candidates[0]["confidence"]) == pytest.approx((reference.confidence(vectors[queries[0]]) + 1) / 2)
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["embedding_ids"] == queries + gallery_ids
    assert manifest["cosine_threshold"] == report["threshold"]


def test_cli_runs_real_bundled_encoder_without_train_and_database(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    make_dataset(dataset, gallery_count=10)
    environment = dict(os.environ, GALLERY_STORAGE="not-postgres", DATABASE_URL="postgresql://unavailable:1/missing")
    result = subprocess.run(
        [sys.executable, "-m", "backend.infer", "--dataset", str(dataset), "--output", str(output)],
        cwd=core.ROOT, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert evaluate.validate_artifacts(dataset, output)["embedding_shape"] == [12, 2048]
    assert not (dataset / "train.csv").exists()


def test_batch_rejects_gallery_too_small_for_top_ten(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    make_dataset(dataset, gallery_count=9)
    with pytest.raises(ValueError, match="at least 10"):
        infer.run_inference(dataset, output)
    assert not output.exists()


def test_in_memory_repository_is_process_local_and_checks_identity_order():
    repository = InMemoryGalleryRepository()
    rows = [{"image_id": "a"}, {"image_id": "b"}]
    vectors = np.eye(2, dtype=np.float32)
    assert repository.load("fingerprint", ["a", "b"], 2, None) is None
    repository.replace("fingerprint", rows, vectors, None)
    vectors[0] = 0
    np.testing.assert_array_equal(repository.load("fingerprint", ["a", "b"], 2, None), np.eye(2))
    assert repository.load("changed", ["a", "b"], 2, None) is None
    assert repository.load("fingerprint", ["b", "a"], 2, None) is None
    assert repository.load("fingerprint", ["a", "b"], 3, None) is None
