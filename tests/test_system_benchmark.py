"""Small synthetic ONNX/image fixtures; never run the long or official benchmark."""
import csv
import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from backend.core import read_rows
from training import frozen_inference as frozen
from training import policy_inference as deployment
from training import system_benchmark as benchmark


@pytest.fixture
def bundle(tmp_path):
    graph = helper.make_graph([
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[2, 3], keepdims=0),
        helper.make_node("Concat", ["mean", "mean"], ["embedding"], axis=1)], "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 8, 8])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 6])])
    path = tmp_path / "weights.custom_extension"
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), path)
    result = tmp_path / "bundle.json"
    frozen.write_bundle(result, path, image_size=8, resize_mode="square", threshold=.5,
                        calibration={"split": "calibration", "protocol_sha256": "a" * 64,
                                     "method": "fixed synthetic threshold"})
    return result


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dataset"
    (path / "images").mkdir(parents=True)
    for split, count in (("query", 3), ("gallery", 12)):
        with (path / f"test_{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            for index in range(count):
                identifier = f"{split}_{index}"
                writer.writerow([identifier, 1, 1, 8, 8])
                values = np.random.default_rng(index + 200 * (split == "gallery")).integers(
                    0, 255, (10, 12, 3), dtype=np.uint8)
                Image.fromarray(values).save(path / "images" / f"{identifier}.jpg")
    return path


def policy_bundle(path):
    output = path.parent / "policy.json"
    deployment.write_bundle(output, [path], ranking="less_graph", candidate_policy="raw_top1", threshold=.5,
                            calibration={"split": "calibration", "candidate_policy": "raw_top1",
                                         "protocol_sha256": "b" * 64, "method": "fixed synthetic threshold"})
    return output


@pytest.mark.parametrize("schema", [1, 2])
def test_load_schema_adapter_and_dimensions(bundle, schema):
    path = policy_bundle(bundle) if schema == 2 else bundle
    encoder = benchmark.load_encoder(path)
    assert encoder.dimension == 6
    assert encoder.bundle["schema"] == schema


def test_inventory_counts_extensionless_declared_weights_and_deduplicates_alias(bundle, tmp_path):
    model = tmp_path / "weights.custom_extension"
    alias = tmp_path / "alias"
    alias.symlink_to(model)
    extra = tmp_path / "auxiliary.odd"
    extra.write_bytes(b"1234567")
    inventory = benchmark.weight_inventory(policy_bundle(bundle), [model, alias, extra])
    assert len(inventory["files"]) == 2
    assert inventory["total_bytes"] == model.stat().st_size + 7
    assert inventory["within_limit"]


def test_inventory_counts_real_copies_and_directory_extensions(bundle, tmp_path):
    model = tmp_path / "weights.custom_extension"
    copy = tmp_path / "same_bytes.onnx"
    copy.write_bytes(model.read_bytes())
    additional = tmp_path / "aux.PTH"
    additional.write_bytes(b"1234")
    inventory = benchmark.weight_inventory(bundle, release_root=tmp_path)
    assert inventory["total_bytes"] == model.stat().st_size * 2 + 4
    assert inventory["release_directory_scanned"]


def test_weight_limit_inclusive_and_checksum_validation(bundle, monkeypatch):
    size = benchmark.weight_inventory(bundle)["total_bytes"]
    monkeypatch.setattr(benchmark, "WEIGHT_LIMIT_BYTES", size)
    assert benchmark.weight_inventory(bundle)["within_limit"]
    monkeypatch.setattr(benchmark, "WEIGHT_LIMIT_BYTES", size - 1)
    assert not benchmark.weight_inventory(bundle)["within_limit"]
    value = json.loads(bundle.read_text())
    value["model"]["sha256"] = "0" * 64
    bundle.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        benchmark.weight_inventory(bundle)


def test_cache_key_tracks_space_order_bbox_bytes_but_not_unavailable_labels(dataset):
    rows = read_rows(dataset / "test_gallery.csv")
    original = benchmark.gallery_cache_key("first-space", rows, dataset)
    same = [{**row, "vehicle_id": 1, "camera_id": 4} for row in rows]
    assert benchmark.gallery_cache_key("first-space", same, dataset) == original
    assert benchmark.gallery_cache_key("second-space", rows, dataset) != original
    assert benchmark.gallery_cache_key("first-space", rows[::-1], dataset) != original
    changed = [dict(row) for row in rows]
    changed[0]["w"] -= 1
    assert benchmark.gallery_cache_key("first-space", changed, dataset) != original
    manifest = {"gallery_cache_key": original}
    assert benchmark.validate_gallery_cache(manifest, "first-space", rows, dataset) == original
    Image.new("RGB", (12, 10), "red").save(dataset / "images" / f"{rows[0]['image_id']}.jpg")
    with pytest.raises(ValueError, match="Stale"):
        benchmark.validate_gallery_cache(manifest, "first-space", rows, dataset)


def test_cpu_short_benchmark_reopens_images_and_cannot_report_official_score(bundle, dataset, monkeypatch):
    openings, synchronizations = [], []
    original_open = benchmark.Image.open

    def opened(path, *args, **kwargs):
        openings.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(benchmark.Image, "open", opened)
    monkeypatch.setattr(benchmark, "_device", lambda provider: (
        lambda: synchronizations.append(1), {"kind": "CPU", "synchronization": "synthetic counter"}))
    result = benchmark.benchmark_bundle(bundle, dataset, samples=3, warmup=2,
                                        seconds_per_batch=.001, batches=(1, 2), official_hardware=True)
    assert result["latency_b1"]["median_ms"] > 0
    assert result["best_fps"] > 0
    assert result["actual_providers"] == [["CPUExecutionProvider"]]
    assert not result["official_gpu_verified"]
    assert not result["protocol_valid"]
    assert result["measurement_kind"] == "diagnostic only"
    assert result["score"]["performance_points"] is None
    assert result["score"]["auto_points"] is None
    expected_opens = 2 + 3 + sum(item["images"] + item["batch"] for item in result["throughput"]) + 6
    assert len(openings) == expected_opens
    assert len(synchronizations) >= 2 * (2 + 3)
    assert result["peak_vram_bytes"] is None
    assert result["training_updates"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [dict(samples=0), dict(warmup=-1), dict(seconds_per_batch=0),
                                   dict(seconds_per_batch=float("nan")), dict(batches=()),
                                   dict(batches=(1, 1)), dict(batches=(0,))])
def test_invalid_benchmark_protocol_fails_before_loading(bundle, dataset, kwargs):
    with pytest.raises(ValueError, match="protocol"):
        benchmark.benchmark_bundle(bundle, dataset, **kwargs)


def test_explicit_unavailable_cuda_cannot_silently_use_cpu(bundle, dataset, monkeypatch):
    def unavailable(provider):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(benchmark, "_device", unavailable)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        benchmark.benchmark_bundle(bundle, dataset, provider="CUDAExecutionProvider")
    with pytest.raises(ValueError, match="provider"):
        benchmark.benchmark_bundle(bundle, dataset, provider="MPS")


def test_exact_official_protocol_required():
    assert benchmark._protocol_valid(300, 50, 10, (1, 8, 16, 32))
    assert not benchmark._protocol_valid(30, 3, 10, (1, 8, 16, 32))
    assert not benchmark._protocol_valid(300, 50, 9.99, (1, 8, 16, 32))
    assert not benchmark._protocol_valid(300, 50, 10, (1, 8, 16))


@pytest.mark.parametrize("latency,fps,points", [(60, 100, 15), (40, 100, 20), (20, 200, 20), (90, 40, 0)])
def test_score_arithmetic_only_after_verified_conditions(latency, fps, points):
    report = {"official_gpu_verified": True, "protocol_valid": True, "weights": {"within_limit": True},
              "latency_b1": {"median_ms": latency}, "best_fps": fps}
    result = benchmark.performance_score(report, .8, .7, .9)
    assert result["performance_points"] == points
    assert result["auto_points"] == pytest.approx(45 * .8 + 10 * (.7 * .7 + .3 * .9) + points)
    report["weights"]["within_limit"] = False
    assert benchmark.performance_score(report, .8, .7, .9)["auto_points"] is None


@pytest.mark.parametrize("schema", [1, 2])
def test_streaming_audit_same_decisions_across_batch_and_query_order(bundle, dataset, schema):
    path = policy_bundle(bundle) if schema == 2 else bundle
    result = benchmark.audit_streaming(benchmark.load_encoder(path), read_rows(dataset / "test_query.csv"),
                                       read_rows(dataset / "test_gallery.csv"), dataset, batch_size=2)
    assert result["passed"]
    assert result["query_count"] == 3
    assert result["gallery_count"] == 12
    assert not result["network_isolation_verified"]
    assert all(not check["changed_top10"] for check in result["checks"].values())


def test_extractor_rejects_zero_nonfinite_or_wrong_dimension_output(bundle, dataset, monkeypatch):
    encoder = benchmark.load_encoder(bundle)
    monkeypatch.setattr(encoder, "encode_batch", lambda batch: np.zeros((len(batch), 6), np.float32))
    with pytest.raises(ValueError, match="L2-normalized"):
        benchmark._extract(encoder, read_rows(dataset / "test_query.csv"), dataset)


@pytest.mark.parametrize("schema", [1, 2])
def test_timed_export_matches_frozen_decisions_and_preserves_embedding_order(bundle, dataset, tmp_path, schema):
    path = policy_bundle(bundle) if schema == 2 else bundle
    reference, output = tmp_path / "reference", tmp_path / "timed"
    exporter = frozen.export_frozen if schema == 1 else deployment.export_frozen
    exporter(path, dataset, reference, batch_size=2)
    report = benchmark.timed_export(path, dataset, output, batch_size=2)
    assert (output / "submission.csv").read_bytes() == (reference / "submission.csv").read_bytes()
    np.testing.assert_array_equal(np.load(output / "embeddings.npy"), np.load(reference / "embeddings.npy"))
    with (output / "candidates.csv").open() as left, (reference / "candidates.csv").open() as right:
        a, b = list(csv.DictReader(left)), list(csv.DictReader(right))
    assert [(r["query_id"], r["gallery_id"]) for r in a] == [(r["query_id"], r["gallery_id"]) for r in b]
    np.testing.assert_allclose([float(r["confidence"]) for r in a], [float(r["confidence"]) for r in b], atol=2e-7)
    assert report["validation"]["embedding_shape"] == [15, 6]
    assert report["embedding_ids"] == [row["image_id"] for name in ("test_query.csv", "test_gallery.csv")
                                        for row in read_rows(dataset / name)]
    assert report["timings"]["full_runtime_seconds"] >= sum(
        value for key, value in report["timings"].items() if key != "full_runtime_seconds")
    assert not report["quality_evaluated"]
    assert not report["official_gpu_verified"]
    assert report["training_updates"] == 0
    assert json.loads((output / "timed_export_manifest.json").read_text()) == report
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="new/empty"):
        benchmark.timed_export(path, dataset, output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_timed_export_does_not_read_train_or_evaluate_labels(bundle, dataset, tmp_path, monkeypatch):
    import builtins
    import io
    import evaluate as official

    def forbidden(*args, **kwargs):
        raise AssertionError("No quality evaluation or calibration allowed")

    monkeypatch.setattr(official, "ranking_metrics", forbidden)
    monkeypatch.setattr(official, "candidate_metrics", forbidden)
    monkeypatch.setattr(benchmark.policy, "calibrate_policy", forbidden)
    for csv_name in ("test_query.csv", "test_gallery.csv"):
        rows = read_rows(dataset / csv_name)
        with (dataset / csv_name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*rows[0], "vehicle_id", "camera_id"])
            writer.writeheader()
            writer.writerows({**row, "vehicle_id": 1, "camera_id": 2} for row in rows)
    real_open, real_io_open = builtins.open, io.open

    def guard(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, type(dataset))):
                assert "train.csv" not in str(path)
            return original(path, *args, **kwargs)
        return opened

    monkeypatch.setattr(builtins, "open", guard(real_open))
    monkeypatch.setattr(io, "open", guard(real_io_open))
    result = benchmark.timed_export(policy_bundle(bundle), dataset, tmp_path / "timed")
    assert result["validation"]["submission_rows"] == 3
    assert not result["quality_evaluated"]
