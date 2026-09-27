"""Frozen integration contract, including negative cases; no metric tuning."""
import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.cache_spaces import FileGallerySpaces
from backend.core import ROOT, bbox, normalize, read_rows, sha256
from backend.frozen_encoder import FrozenEncoder, _session, combine_members
from backend.images import ImageIndex
from backend.infer import run_inference
from backend.runtime import ExactScorer, PROFILE_NAMES, Runtime, RuntimeGallery, encode_rows


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    rows = []
    for index in range(16):
        identifier = f"{index:032x}"
        extension = (".jpg", ".JPEG", ".png")[index % 3]
        pixels = np.random.default_rng(index).integers(0, 256, (36, 48, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / (identifier + extension))
        rows.append(dict(image_id=identifier, x=0, y=0, w=48, h=36))
    for split, chosen in (("test_query", rows[:4]), ("test_gallery", rows[4:])):
        with (root / f"{split}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(chosen)
    return root


def decision_ids(value):
    candidate = value["accepted_candidate"]
    return ([x["image_id"] for x in value["results"]], candidate["image_id"] if candidate else None,
            value["refused"])


@pytest.mark.parametrize("profile,dimension,threshold", [
    ("MVP_legacy", 512, .5948754549026489),
    ("RC_R1_equal3_v18", 1536, .534365177154541),
    ("MVP_dual_role_v24", 2048, .534365177154541),
    ("MVP_fusion_v25", 2048, .534365177154541),
    ("RC_R1_single_v18", 512, .5522230267524719)])
def test_real_profiles_export_mixed_images_and_api_match(dataset, tmp_path, profile, dimension, threshold):
    output = tmp_path / profile
    result = run_inference(dataset, output, profile, batch_size=8)
    assert result["embedding_shape"] == [16, dimension]
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["cosine_threshold"] == threshold
    vectors = np.load(output / "embeddings.npy")
    app = create_app(dataset=dataset, gallery_repository=FileGallerySpaces(tmp_path / "cache"), profile=profile)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["profile"] == profile and health["embedding_dim"] == dimension
        assert client.get("/api/metrics").json()["threshold"] == threshold
        ranking = list(csv.reader((output / "submission.csv").open()))
        candidates = {r["query_id"]: r["gallery_id"] for r in csv.DictReader((output / "candidates.csv").open())}
        for row in ranking:
            response = client.post("/api/search/query", json={"query_id": row[0], "mode": "candidates"}).json()
            assert [x["image_id"] for x in response["results"]] == row[1:]
            item = response["accepted_candidate"]
            assert (item["image_id"] if item else None) == candidates.get(row[0])
            assert response["refused"] == (item is None)
        for row in read_rows(dataset / "test_query.csv"):
            assert client.get(f"/api/images/query/{row['image_id']}").status_code == 200
        assert vectors.dtype == np.float32
        before = client.get("/api/metrics").json()
        response = client.post("/api/search/query", json={"query_id": ranking[0][0], "mode": "candidates", "threshold": 1}).json()
        assert response["refused"] and len(response["results"]) == 10 and response["accepted_candidate"] is None
        assert client.get("/api/metrics").json() == before
    with pytest.raises(ValueError, match="new or empty"):
        run_inference(dataset, output, profile)


def test_equal_concat_and_candidate_independent_of_ranking():
    a, b, c = [normalize(np.random.default_rng(i).normal(size=(6, 512))) for i in range(3)]
    combined = combine_members([a, b, c])
    np.testing.assert_allclose(combined @ combined.T, (a @ a.T + b @ b.T + c @ c.T) / 3, atol=1e-6)
    fake = SimpleNamespace(dimension=2, threshold=.6,
                           spec={"kind": "policy", "ranking": "less_graph", "candidate_policy": "raw_top1"})
    rows = [dict(image_id=str(i), x=0, y=0, w=1, h=1) for i in range(12)]
    vectors = normalize(np.array([[1, .01 * i] for i in range(12)], dtype=np.float32))
    scorer = ExactScorer(fake, rows, vectors)
    scorer.graph = SimpleNamespace(distances=lambda vector, lam: np.arange(12, 0, -1))
    answer = scorer.decide(np.array([1, 0], np.float32))
    assert answer["results"][0]["image_id"] == "11"
    assert answer["accepted_candidate"]["image_id"] == "0"  # outside displayed top-10
    assert answer["accepted_candidate"]["rank"] == 12
    refusal = scorer.decide(np.array([-1, 0], np.float32))
    assert len(refusal["results"]) == 10 and refusal["accepted_candidate"] is None


def test_batch_permutation_subset_and_rollback(dataset, tmp_path):
    repository = FileGallerySpaces(tmp_path / "spaces")
    queries = read_rows(dataset / "test_query.csv")
    original = None
    for profile in ("MVP_legacy", "RC_R1_equal3_v18", "MVP_dual_role_v24", "MVP_fusion_v25", "MVP_dual_role_v24", "MVP_legacy"):
        runtime = Runtime(profile)
        gallery = RuntimeGallery(runtime, dataset, repository)
        base = encode_rows(runtime, queries, dataset, 1)
        decisions = [decision_ids(gallery.decide(v)) for v in base]
        for batch in (8, 16, 32):
            actual = encode_rows(runtime, queries, dataset, batch)
            np.testing.assert_allclose(actual, base, atol=2e-4, rtol=0)
            assert [decision_ids(gallery.decide(v)) for v in actual] == decisions
        reversed_vectors = encode_rows(runtime, queries[::-1], dataset, 8)[::-1]
        assert [decision_ids(gallery.decide(v)) for v in reversed_vectors] == decisions
        subset = encode_rows(runtime, queries[::2], dataset, 8)
        assert [decision_ids(gallery.decide(v)) for v in subset] == decisions[::2]
        if profile == "MVP_legacy":
            if original is None:
                original = decisions
            else:
                assert gallery.cache_hit and decisions == original
        reloaded = RuntimeGallery(runtime, dataset, repository)
        assert reloaded.cache_hit
        np.testing.assert_array_equal(reloaded.vectors, gallery.vectors)
    assert len(list((tmp_path / "spaces").glob("*.npz"))) == 3


def test_resolver_missing_and_ambiguous(dataset):
    images = ImageIndex(dataset / "images")
    with pytest.raises(FileNotFoundError, match="No JPEG/PNG"):
        images.resolve("missing")
    first = read_rows(dataset / "test_query.csv")[0]["image_id"]
    Image.new("RGB", (48, 36)).save(dataset / "images" / (first + ".png"))
    with pytest.raises(ValueError, match="Ambiguous"):
        ImageIndex(dataset / "images").resolve(first)
    with pytest.raises(ValueError, match="Invalid image ID"):
        images.resolve("../secret")


def test_cache_content_order_bbox_and_pixels_invalidate(dataset, tmp_path):
    runtime, repository = Runtime("MVP_legacy"), FileGallerySpaces(tmp_path / "spaces")
    original = RuntimeGallery(runtime, dataset, repository)
    with pytest.raises(ValueError, match="Incompatible"):
        repository.load(original.fingerprint, [r["image_id"] for r in original.rows], 1536, original.build_state)
    with pytest.raises(ValueError, match="Incompatible"):
        repository.load(original.fingerprint, [r["image_id"] for r in original.rows][::-1], 512, original.build_state)
    rows = [dict(row) for row in original.rows]
    rows[0]["w"] -= 1
    def save():
        with (dataset / "test_gallery.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save()
    crop_changed = RuntimeGallery(runtime, dataset, repository)
    assert original.fingerprint != crop_changed.fingerprint
    rows.reverse(); save()
    order_changed = RuntimeGallery(runtime, dataset, repository)
    assert crop_changed.fingerprint != order_changed.fingerprint
    Image.new("RGB", (48, 36)).save(ImageIndex(dataset / "images").resolve(rows[0]["image_id"]))
    pixels_changed = RuntimeGallery(runtime, dataset, repository)
    assert pixels_changed.fingerprint != order_changed.fingerprint
    bad = original.vectors.copy(); bad[0] = bad[1]
    with pytest.raises(ValueError, match="different immutable"):
        repository.replace(original.fingerprint, original.rows, bad, original.build_state)


def test_bad_weights_and_unavailable_provider(tmp_path, monkeypatch):
    runtime = Runtime("RC_R1_single_v18")
    bundle = runtime.encoder.members[0].bundle
    copied = json.loads(json.dumps(bundle))
    copied["model"]["path"] = "damaged.onnx"
    (tmp_path / "damaged.onnx").write_bytes(b"damaged")
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(copied))
    with pytest.raises(ValueError, match="checksum mismatch"):
        FrozenEncoder(path)
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="unavailable"):
        Runtime(provider="CUDAExecutionProvider")


def test_runtime_does_not_import_training_calibration_or_postgres(dataset, tmp_path):
    # Block network, evaluation/calibration modules, torch, and DB in a clean process.
    code = """
import importlib.abc, socket, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'training', 'torch', 'psycopg'} or fullname == 'backend.evaluate':
            raise AssertionError('Forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
def denied(*args, **kwargs): raise AssertionError('Network attempted')
socket.socket.connect = denied
from backend.infer import run_inference
run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
"""
    for profile in ("MVP_legacy", "RC_R1_equal3_v18", "MVP_dual_role_v24", "MVP_fusion_v25"):
        result = subprocess.run([sys.executable, "-c", code, str(dataset), str(tmp_path / profile), profile],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    assert not (dataset / "train.csv").exists()
