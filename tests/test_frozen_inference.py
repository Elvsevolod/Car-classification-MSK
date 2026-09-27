"""Frozen deployment contracts, using tiny synthetic ONNX graphs and images."""
import builtins
import csv
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from backend.core import STOCK_MODEL, Encoder, preprocess
from training import frozen_inference as frozen


def tiny_model(path, size=8):
    # Six-dimensional output intentionally differs from historical hard-coded 512.
    graph = helper.make_graph([
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[2, 3], keepdims=0),
        helper.make_node("Concat", ["mean", "mean"], ["embedding"], axis=1),
    ], "tiny", [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 6])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save_model(model, path)
    return path


@pytest.fixture
def bundle(tmp_path):
    model = tiny_model(tmp_path / "tiny.onnx")
    path = tmp_path / "bundle.json"
    frozen.write_bundle(path, model, image_size=8, resize_mode="square", threshold=.5,
                        calibration={"split": "calibration", "protocol_sha256": "a" * 64,
                                     "method": "fixed synthetic test threshold"})
    return path


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "data"
    (path / "images").mkdir(parents=True)
    for split, count in (("query", 3), ("gallery", 12)):
        with (path / f"test_{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            for index in range(count):
                identifier = f"{split}_{index}"
                writer.writerow([identifier, 1, 1, 10, 7])
                pixels = np.random.default_rng(index).integers(0, 256, (10, 15, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(path / "images" / f"{identifier}.jpg")
    return path


def change_bundle(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_dimension_derived_and_dynamic_batch_checked(bundle):
    encoder = frozen.FrozenEncoder(bundle)
    assert encoder.dimension == 6
    assert encoder.session.get_providers() == ["CPUExecutionProvider"]
    batch = np.random.default_rng(23).normal(size=(8, 3, 8, 8)).astype(np.float32)
    result = encoder.encode_batch(batch)
    assert result.shape == (8, 6)
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)


def test_bundle_is_frozen_and_keeps_existing_file(bundle):
    old = bundle.read_bytes()
    value = json.loads(old)
    args = {"image_size": 8, "resize_mode": "square", "threshold": .5, "calibration": value["calibration"]}
    frozen.write_bundle(bundle, bundle.parent / "tiny.onnx", **args)
    assert bundle.read_bytes() == old
    with pytest.raises(ValueError, match="replace"):
        frozen.write_bundle(bundle, bundle.parent / "tiny.onnx", **{**args, "threshold": .6})
    assert bundle.read_bytes() == old


@pytest.mark.parametrize("key,value", [("split", "validation"), ("method", ""), ("protocol_sha256", "missing")])
def test_calibration_provenance_required(bundle, key, value):
    change_bundle(bundle, lambda b: b["calibration"].update({key: value}))
    with pytest.raises(ValueError, match="calibration-only"):
        frozen.FrozenEncoder(bundle)


@pytest.mark.parametrize("change,match", [
    (lambda b: b["model"].update(dimension=512), "dimension mismatch"),
    (lambda b: b["model"].update(sha256="0" * 64), "checksum mismatch"),
    (lambda b: b["preprocessing"].update(image_size=9), "input shape"),
    (lambda b: b["preprocessing"].update(flip_tta=True), "preprocessing"),
    (lambda b: b["preprocessing"].update(scale=1), "preprocessing"),
    (lambda b: b["reranking"].update(k2=6), "ranking"),
    (lambda b: b.update(confidence="reranked distance"), "confidence"),
    (lambda b: b.update(threshold=float("nan")), "finite"),
])
def test_bundle_shape_checksum_and_policy_guards(bundle, change, match):
    change_bundle(bundle, change)
    with pytest.raises(ValueError, match=match):
        frozen.FrozenEncoder(bundle)


def test_provider_unavailable_cannot_fallback(bundle, monkeypatch):
    monkeypatch.setattr(frozen.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="unavailable"):
        frozen.FrozenEncoder(bundle, "CUDAExecutionProvider")
    assert frozen.compare_providers(bundle, Path("does-not-exist")) == {
        "status": "not-tested", "reason": "CUDAExecutionProvider unavailable"}


def test_provider_creation_disables_fallback_and_rejects_wrong_actual_provider(bundle, monkeypatch):
    monkeypatch.setattr(frozen.ort, "get_available_providers", lambda: list(frozen.PROVIDERS))
    calls = []

    class WrongSession:
        def __init__(self, model, **kwargs):
            calls.append(kwargs)

        def disable_fallback(self):
            calls.append("disabled")

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(frozen.ort, "InferenceSession", WrongSession)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        frozen.FrozenEncoder(bundle, "CUDAExecutionProvider")
    assert calls[0]["providers"] == ["CUDAExecutionProvider"]
    assert calls[0]["enable_fallback"] is False
    assert calls[0]["sess_options"].get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
    assert calls[1] == "disabled"


def test_external_unchecksummed_weights_rejected(tmp_path):
    model = onnx.load(tiny_model(tmp_path / "tiny.onnx"))
    tensor = model.graph.initializer.add()
    tensor.name, tensor.data_type, tensor.data_location = "external", TensorProto.FLOAT, TensorProto.EXTERNAL
    tensor.dims.append(1)
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", "weights.bin"
    path = tmp_path / "external.onnx"
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="embedded ONNX weights"):
        frozen._session(path, "CPUExecutionProvider")


@pytest.mark.parametrize("inputs", [np.ones((1, 3, 9, 9)), np.zeros((0, 3, 8, 8)),
                                     np.full((1, 3, 8, 8), np.nan), np.zeros((1, 3, 8, 8))])
def test_invalid_input_or_zero_output(bundle, inputs):
    with pytest.raises(ValueError):
        frozen.FrozenEncoder(bundle).encode_batch(inputs)


@pytest.mark.parametrize("threshold,accepted", [(-2, 3), (2, 0)])
def test_export_uses_only_frozen_model_test_csv_images(bundle, dataset, tmp_path, monkeypatch, threshold, accepted):
    change_bundle(bundle, lambda b: b.update(threshold=threshold))
    output = tmp_path / "export"
    real_open = builtins.open
    seen = []

    def guard(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            seen.append(str(path))
            assert Path(path).name not in {"train.csv", "splits.json", "baseline_metrics.json"}
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard)
    result = frozen.export_frozen(bundle, dataset, output, batch_size=2)
    assert result["embedding_shape"] == [15, 6]
    assert result["accepted_queries"] == accepted
    assert result["submission_rows"] == 3
    assert not (dataset / "train.csv").exists()
    vectors = np.load(output / "embeddings.npy")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    submission = list(csv.reader((output / "submission.csv").open()))
    assert all(len(row) == 11 for row in submission)
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["embedding_ids"][:3] == [f"query_{i}" for i in range(3)]
    assert manifest["embedding_ids"][3:] == [f"gallery_{i}" for i in range(12)]
    assert manifest["bundle"]["threshold"] == threshold
    assert manifest["provider"] == "CPUExecutionProvider"
    assert len(manifest["image_sha256"]) == 15
    with pytest.raises(ValueError, match="existing artifacts"):
        frozen.export_frozen(bundle, dataset, output)


def test_benchmark_decodes_every_sample_and_never_claims_official_gpu(bundle, dataset, monkeypatch):
    seen = []
    real_open = Image.open

    def tracked(path):
        seen.append(path)
        return real_open(path)

    monkeypatch.setattr(Image, "open", tracked)
    report = frozen.benchmark(bundle, dataset, samples=5, warmup=2)
    assert len(seen) == 7
    assert report["device"] == "CPU"
    assert report["samples"] == 5
    assert report["official_gpu_verified"] is False
    assert 0 < report["median_ms"] <= report["p95_ms"]


def test_identity_camera_and_other_query_metadata_do_not_change_export(bundle, dataset, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    frozen.export_frozen(bundle, dataset, first)
    for split in ("query", "gallery"):
        path = dataset / f"test_{split}.csv"
        rows = list(csv.reader(path.open()))
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([*rows[0], "vehicle_id", "camera_id"])
            for index, row in enumerate(rows[1:]):
                writer.writerow([*row, 12 if split == "query" else index, index % 2])
    frozen.export_frozen(bundle, dataset, second)
    for name in ("submission.csv", "candidates.csv", "embeddings.npy"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_real_stock_model_cpu_matches_existing_preprocessing_and_vectors(tmp_path):
    if not STOCK_MODEL.exists():
        pytest.skip("Local stock weights unavailable; no download")
    path = tmp_path / "stock_bundle.json"
    frozen.write_bundle(path, STOCK_MODEL, image_size=208, resize_mode="square", threshold=.8,
                        calibration={"split": "calibration", "protocol_sha256": "b" * 64,
                                     "method": "synthetic smoke, no quality claim"})
    encoder = frozen.FrozenEncoder(path)
    image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (101, 155, 3), dtype=np.uint8))
    box = (3, 5, 120, 85)
    assert encoder.dimension == 512
    np.testing.assert_array_equal(encoder.preprocess(image, box), preprocess(image, box))
    np.testing.assert_allclose(encoder.encode(image, box), Encoder(STOCK_MODEL).encode(image, box), atol=1e-6)
