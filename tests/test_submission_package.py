import csv
import hashlib
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from backend.runtime import ExactScorer
from tools.package_submission import package, replay_decisions, verify_source_kit


@pytest.fixture
def exported(tmp_path):
    runtime = SimpleNamespace(dimension=2048, threshold=.5,
                              spec={"kind": "dual_role", "ranking": "legacy",
                                    "r1_weight": .5, "candidate_policy": "raw_top1"})
    gallery = [{"image_id": f"g{i}", "x": 0, "y": 0, "w": 10, "h": 10} for i in range(12)]
    queries = [{"image_id": "accepted"}, {"image_id": "refused"}]
    rng = np.random.default_rng(7)
    blocks = []
    for size in (512, 1536):
        values = rng.normal(size=(12, size)).astype(np.float32)
        blocks.append(values / np.linalg.norm(values, axis=1, keepdims=True))
    gallery_vectors = np.concatenate(blocks, axis=1)
    vectors = np.concatenate([gallery_vectors[:1], -gallery_vectors[:1], gallery_vectors])
    scorer = ExactScorer(runtime, gallery, gallery_vectors)
    with (tmp_path / "submission.csv").open("w", newline="") as ranking, \
            (tmp_path / "candidates.csv").open("w", newline="") as candidates:
        ranked, accepted = csv.writer(ranking), csv.writer(candidates)
        accepted.writerow(["query_id", "gallery_id", "confidence"])
        for query, vector in zip(queries, vectors):
            decision = scorer.decide(vector)
            ranked.writerow([query["image_id"], *[r["image_id"] for r in decision["results"]]])
            if decision["accepted_candidate"] is not None:
                accepted.writerow([query["image_id"], decision["accepted_candidate"]["image_id"], decision["confidence"]])
    return runtime, queries, gallery, vectors, tmp_path


def test_replay_checks_accepted_and_refused_query(exported):
    result = replay_decisions(*exported)
    assert result["queries_checked"] == 2
    assert result["top10_exact_match"]
    assert result["confidence_max_abs_error"] == 0


def test_replay_rejects_reordered_valid_top_ten(exported):
    path = exported[-1] / "submission.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    rows[0][1], rows[0][2] = rows[0][2], rows[0][1]
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="Ranking does not match"):
        replay_decisions(*exported)


def test_replay_rejects_valid_id_with_wrong_confidence(exported):
    path = exported[-1] / "candidates.csv"
    path.write_text("query_id,gallery_id,confidence\naccepted,g0,0.8\n")
    with pytest.raises(ValueError, match="Candidate/confidence differs"):
        replay_decisions(*exported)


def test_replay_rejects_false_acceptance(exported):
    with (exported[-1] / "candidates.csv").open("a", newline="") as stream:
        csv.writer(stream).writerow(["refused", "g1", .8])
    with pytest.raises(ValueError, match="Refusal differs"):
        replay_decisions(*exported)


def test_existing_package_is_preserved(tmp_path):
    output = tmp_path / "package"
    output.mkdir()
    existing = output / "submission.csv"
    existing.write_bytes(b"keep this file")
    with pytest.raises(ValueError, match="existing packages are preserved"):
        package(tmp_path / "dataset", tmp_path / "source", output, tmp_path / "source.zip")
    assert existing.read_bytes() == b"keep this file"


@pytest.fixture
def source_kit(tmp_path):
    recipe = {"runtime_commit": "historical-training-runtime", "profile": "MVP_fusion_v25",
              "runtime_onnx_sha256": {"models/encoder.onnx": "frozen-model-hash"}}
    files = {"metadata/recipe.json": json.dumps(recipe).encode(),
             "runtime/models/profiles.json": b"frozen policy"}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    path = tmp_path / "kit.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in {**files, "KIT_SHA256.json": json.dumps(manifest).encode()}.items():
            archive.writestr("vehicle-reid-v25-reproduction/" + name, data)
    assets = {"models/encoder.onnx": {"sha256": "frozen-model-hash"},
              "models/profiles.json": {"sha256": manifest["runtime/models/profiles.json"]}}
    return path, SimpleNamespace(profile="MVP_fusion_v25"), assets


def test_historical_kit_requires_exact_current_assets(source_kit):
    result = verify_source_kit(*source_kit)
    assert result["historical_runtime_commit"] == "historical-training-runtime"


@pytest.mark.parametrize("name", ["models/encoder.onnx", "models/profiles.json"])
def test_source_kit_rejects_changed_model_or_policy(source_kit, name):
    source_kit[2][name]["sha256"] = "changed"
    with pytest.raises(ValueError, match="another frozen asset"):
        verify_source_kit(*source_kit)


def test_source_kit_rejects_changed_profile(source_kit):
    source_kit[1].profile = "other"
    with pytest.raises(ValueError, match="another profile"):
        verify_source_kit(*source_kit)


def test_source_kit_rejects_corrupt_member(source_kit, tmp_path):
    path, runtime, assets = source_kit
    corrupt = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(corrupt, "w") as archive:
        for name in original.namelist():
            data = b"changed policy" if name.endswith("profiles.json") else original.read(name)
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_source_kit(corrupt, runtime, assets)
