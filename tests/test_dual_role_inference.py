"""Frozen role separation, true ONNX image path and export, without real training."""
import csv
import hashlib
import json
import socket
import subprocess
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from backend import core
from training import dual_role_inference as dual
from training import frozen_inference as frozen


def features(count=49):
    rng = np.random.default_rng(76)
    return [core.normalize(rng.normal(size=(count, d)).astype(np.float32)) for d in (512, 1536)]


def rows(count, prefix):
    return [{"image_id": f"{prefix}_{i}", "x": 0, "y": 0, "w": 12, "h": 9} for i in range(count)]


def test_lossless_blocks_not_a_score_or_fake_vector():
    a, b = features()
    packed = dual.pack(a, b)
    assert packed.dtype == np.float32 and packed.shape == (49, 2048)
    for x, y in zip(dual.unpack(packed), (a, b)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(np.linalg.norm(packed, axis=1), np.sqrt(2), rtol=0, atol=5e-7)


@pytest.mark.parametrize("bad", ["dtype", "nan", "zero", "dimension", "count"])
def test_invalid_embeddings_fail_without_replacement(bad):
    a, b = features()
    a = {"dtype": lambda: a.astype(np.float64), "nan": lambda: np.full_like(a, np.nan),
         "zero": lambda: np.zeros_like(a), "dimension": lambda: a[:, :511], "count": lambda: a[:1]}[bad]()
    with pytest.raises(ValueError):
        dual.pack(a, b)


def test_roles_and_refusal_are_independent_even_outside_top10(tmp_path):
    a, b = features(15)
    a[0] = a[1]  # MVP prefers first gallery image.
    mvp = dual.policy.rank_vectors(a[:1], a[1:], "legacy")
    outside = int(mvp["order"][0, -1])
    b[0] = b[1+outside]  # R1 prefers a candidate outside MVP's top10.
    values = dual.pack(a, b)
    q, g = rows(1, "q"), rows(14, "g")
    ranking = dual.rank(values[:1], values[1:])
    ordered, accepted = dual.policy.predictions(q, g, ranking, .5, "raw_top1")
    assert accepted["q_0"][0][0] == g[outside]["image_id"]
    assert accepted["q_0"][0][0] not in ordered["q_0"]
    assert accepted["q_0"][0][1] == pytest.approx(1, abs=5e-7)
    expected = dual.policy.predictions(q, g, mvp, 2, "ranking_top1")[0]
    assert ordered == expected
    for threshold, count in ((.5, 1), (2., 0)):
        output = tmp_path / str(count)
        dual.export_arrays({"threshold": threshold}, q, g, values, output)
        saved = np.load(output / "embeddings.npy")
        np.testing.assert_array_equal(saved, values)
        submission = list(csv.reader((output / "submission.csv").open()))
        assert len(submission) == 1 and len(submission[0]) == 11
        assert len(set(submission[0][1:])) == 10
        assert len(list(csv.DictReader((output / "candidates.csv").open()))) == count
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        dual.export_arrays({"threshold": threshold}, q, g, values, output)
        assert before == {p.name: p.read_bytes() for p in output.iterdir()}
        with pytest.raises(ValueError, match="Existing export differs"):
            dual.export_arrays({"threshold": 2. if count else .5}, q, g, values, output)
        assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_query_permutation_removal_and_batches_keep_decisions():
    values = dual.pack(*features(61))
    q, g = values[:33], values[33:]
    baseline = dual.rank(q, g)
    cases = [(np.arange(len(q))[::-1], None), (np.array([5]), None)]
    cases += [(np.arange(len(q)), size) for size in (1, 8, 16, 32)]
    for indices, size in cases:
        selected = q[indices]
        size = size or len(selected)
        batches = [dual.rank(selected[start:start+size], g) for start in range(0, len(selected), size)]
        for key in ("order", "raw_order"):
            np.testing.assert_array_equal(np.concatenate([r[key] for r in batches]), baseline[key][indices])
        scores = np.concatenate([r["confidence"] for r in batches])
        np.testing.assert_allclose(scores, baseline["confidence"][indices], rtol=0, atol=2e-7)
        np.testing.assert_array_equal(scores >= .05, baseline["confidence"][indices] >= .05)


def tiny_model(path, size, seed):
    weights = np.random.default_rng(seed).normal(size=(3, 512)).astype(np.float32)
    graph = helper.make_graph([
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["mean", "weights"], ["output"]),
    ], "tiny", [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 512])],
        [numpy_helper.from_array(weights, "weights")])
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), path)
    return path


@pytest.fixture
def profile(tmp_path, monkeypatch):
    mvp = tiny_model(tmp_path / "mvp.onnx", 208, 1)
    monkeypatch.setattr(core, "MODEL", mvp)
    monkeypatch.setattr(core, "MODEL_SHA384", hashlib.sha384(mvp.read_bytes()).hexdigest())
    calibration = {"split": "calibration", "protocol_sha256": "a"*64, "method": "fixed synthetic threshold",
                   "candidate_policy": "raw_top1"}
    members = []
    for seed in (2, 3, 4):
        model = tiny_model(tmp_path / f"r{seed}.onnx", 256, seed)
        bundle = tmp_path / f"r{seed}.json"
        frozen.write_bundle(bundle, model, image_size=256, resize_mode="square", threshold=.5, calibration=calibration)
        members.append({"path": bundle.name, "sha256": core.sha256(bundle)})
    r1 = tmp_path / "r1.json"
    dual.write_json(r1, {"schema": 2, "members": members, "fusion": "equal normalized concatenation",
                        "ranking": "less_graph", "ranking_parameters": dual.policy.POLICIES["less_graph"],
                        "candidate_policy": "raw_top1", "threshold": .5, "calibration": calibration,
                        "confidence": "maximum raw cosine of combined features"})
    path = tmp_path / "profile.json"
    dual.write_json(path, dual.profile_value(path, mvp, r1))
    return path


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "dataset"
    (directory / "images").mkdir(parents=True)
    for split, count in (("query", 3), ("gallery", 12)):
        entries = rows(count, split)
        with (directory / f"test_{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
        for i, row in enumerate(entries):
            suffix = ".PNG" if i % 2 else ".jpg"
            pixels = np.random.default_rng(i).integers(0, 256, (9, 12, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / "images" / f"{row['image_id']}{suffix}")
    return directory


def test_image_export_offline_no_train_database_or_metadata(profile, dataset, tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Network access forbidden")
    monkeypatch.setattr(socket, "socket", deny)
    assert not (dataset / "train.csv").exists()
    result = dual.export_frozen(profile, dataset, tmp_path / "export", batch_size=8)
    assert result["embedding_shape"] == [15, 2048]
    assert result["submission_rows"] == 3
    encoder = dual.DualRoleEncoder(profile)
    entries = core.read_rows(dataset / "test_query.csv")
    a = encoder.encode_rows(entries, dataset, 1)
    b = encoder.encode_rows([{**r, "vehicle_id": 9, "camera_id": 4, "time": "ignored"} for r in entries], dataset, 1)
    np.testing.assert_array_equal(a, b)
    entries[0]["w"] = 13
    with pytest.raises(ValueError, match="outside"):
        encoder.encode_rows(entries, dataset)


def test_missing_and_ambiguous_images_fail(dataset):
    with pytest.raises(ValueError, match="found 0"):
        dual.image_path(dataset, "missing")
    original = dual.image_path(dataset, "query_0")
    (original.parent / "query_0.png").write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="found 2"):
        dual.image_path(dataset, "query_0")


def test_image_batches_keep_features_and_query_decisions(profile, dataset):
    encoder = dual.DualRoleEncoder(profile)
    q, g = (core.read_rows(dataset / f"test_{part}.csv") for part in ("query", "gallery"))
    expected = encoder.encode_rows(q+g, dataset, 16)
    ranking = dual.rank(expected[:len(q)], expected[len(q):])
    ordered, accepted = dual.policy.predictions(q, g, ranking, .5, "raw_top1")
    for size in (1, 8, 32):
        values = encoder.encode_rows(q+g, dataset, size)
        np.testing.assert_allclose(values, expected, rtol=0, atol=2e-5)
        current = dual.rank(values[:len(q)], values[len(q):])
        top10, candidates = dual.policy.predictions(q, g, current, .5, "raw_top1")
        assert top10 == ordered
        assert {k: v[0][0] for k, v in candidates.items()} == {k: v[0][0] for k, v in accepted.items()}
        np.testing.assert_allclose(current["confidence"], ranking["confidence"], rtol=0, atol=2e-5)


@pytest.mark.parametrize("target", ["mvp.onnx", "r1.json", "r2.json", "r2.onnx"])
def test_corrupt_weights_and_nested_bundles_are_rejected(profile, target):
    path = profile.parent / target
    path.write_bytes(path.read_bytes()+b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        dual.DualRoleEncoder(profile)


def test_provider_and_frozen_threshold_guards(profile):
    with pytest.raises(ValueError, match="no provider fallback"):
        dual.DualRoleEncoder(profile, "CUDAExecutionProvider")
    value = dual.read(profile)
    value["threshold"] += .01
    dual.write_json(profile, value)
    with pytest.raises(ValueError, match="threshold"):
        dual.load_profile(profile)


def test_runtime_imports_no_training_stack():
    subprocess.run([sys.executable, "-c", "import sys; import training.dual_role_inference; "
                    "assert 'torch' not in sys.modules; assert 'training.stage6' not in sys.modules; "
                    "assert 'training.audit' not in sys.modules"], check=True)
