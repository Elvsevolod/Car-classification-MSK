"""Actual CPU ONNX export integration; tiny model, synthetic photos, no training."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from backend.core import bbox, sha256
from training import osnet_review_protocol as review
from training.audit import digest
from training.frozen_inference import FrozenEncoder, RERANKING, export_frozen
from training.osnet_ablations import Ablation, AblationDataset, InferenceEncoder


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 544)

    def embedding(self, images):
        return self.projection(images.mean(dim=(2, 3)))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    rows = []
    for index in range(4):
        identifier = f"synthetic_{index}"
        pixels = np.random.default_rng(index).integers(0, 256, (16, 24, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(dataset / "images" / f"{identifier}.jpg")
        rows.append({"image_id": identifier, "x": 2, "y": 3, "w": 19, "h": 12})
    for name, selected in (("test_query.csv", rows[:1]), ("test_gallery.csv", rows[1:])):
        with (dataset / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "x", "y", "w", "h"])
            writer.writeheader()
            writer.writerows(selected)
    variant = Ablation(name="synthetic_export", size=16, resize="letterbox")
    summary = {"seed": 23}
    protocol = {"query_ids": [rows[0]["image_id"]], "gallery_ids": [r["image_id"] for r in rows[1:]]}
    context = {"output": tmp_path / "run", "dataset": dataset, "rows": rows,
               "manifest": {"protocols": {"calibration": protocol}}}
    directory = context["output"] / "final" / variant.name / "seed_23"
    directory.mkdir(parents=True)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = TinyEmbedding().eval()
    calls = []
    original_protocol_rows = review.old.protocol_rows

    def calibration_only(ctx, name):
        assert name == "calibration", "Export must not use validation to choose or verify its model"
        calls.append(name)
        return original_protocol_rows(ctx, name)

    monkeypatch.setattr(review.old, "protocol_rows", calibration_only)
    return SimpleNamespace(context=context, summary=summary, model=model, variant=variant,
                           directory=directory, rows=rows, dataset=dataset, calls=calls)


def run_export(prepared, threshold=.731):
    review.export_model(prepared.context, prepared.summary, prepared.model, prepared.variant, threshold)


def preserved_files(prepared):
    return {name: (prepared.directory / name).read_bytes()
            for name in ("encoder.onnx", "bundle.json", "export.json")}


def test_review_export_links_onnx_parity_preprocessing_dimension_and_threshold(prepared):
    run_export(prepared)
    bundle_path = prepared.directory / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    report = json.loads((prepared.directory / "export.json").read_text())
    assert bundle["model"]["path"] == "encoder.onnx"
    assert bundle["model"]["sha256"] == sha256(prepared.directory / "encoder.onnx") == report["onnx_sha256"]
    assert bundle["model"]["dimension"] == 544
    assert bundle["model"]["input"] == "input" and bundle["model"]["output"] == "output"
    assert bundle["threshold"] == .731 and bundle["reranking"] == RERANKING
    assert bundle["preprocessing"]["image_size"] == 16
    assert bundle["preprocessing"]["resize_mode"] == "letterbox"
    assert bundle["calibration"]["split"] == "calibration"
    assert bundle["calibration"]["protocol_sha256"] == digest(prepared.context["manifest"]["protocols"]["calibration"])
    assert set(report["cpu_parity"]) == {"1", "3", "8"}
    assert all(error <= 2e-4 for error in report["cpu_parity"].values())
    assert report["gpu_parity"] == "not tested" and report["promoted"] is False
    assert prepared.calls == ["calibration"]

    encoder = FrozenEncoder(bundle_path)
    transformed = AblationDataset(prepared.rows, prepared.variant, prepared.dataset)
    for index, row in enumerate(prepared.rows):
        with Image.open(prepared.dataset / "images" / f"{row['image_id']}.jpg") as image:
            runtime_pixels = encoder.preprocess(image, bbox(row))
        np.testing.assert_array_equal(runtime_pixels, transformed[index][0].numpy())
    for count in (1, 3, 8):
        images = torch.stack([transformed[index % len(transformed)][0] for index in range(count)])
        with torch.no_grad():
            expected = InferenceEncoder(prepared.model)(images).numpy()
        actual = encoder.encode_batch(images.numpy())
        assert actual.shape == (count, 544)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)

    # The artifact validator accepts a 544D submission without any train.csv.
    result = export_frozen(bundle_path, prepared.dataset, prepared.directory / "submission")
    assert result["embedding_shape"] == [4, 544]
    assert result["embedding_dtype"] == "float32"
    assert result["submission_rows"] == 1
    assert not (prepared.dataset / "train.csv").exists()


def test_same_weights_repeat_preserves_frozen_bundle_and_encoder(prepared):
    run_export(prepared)
    original = preserved_files(prepared)
    run_export(prepared)
    assert preserved_files(prepared) == original
    assert prepared.calls == ["calibration", "calibration"]


def test_changed_weights_do_not_overwrite_existing_encoder_or_bundle(prepared):
    run_export(prepared)
    original = preserved_files(prepared)
    with torch.no_grad():
        prepared.model.projection.weight[0, 0].add_(.75)
    with pytest.raises(ValueError, match="overwrite a different exported encoder"):
        run_export(prepared)
    assert preserved_files(prepared) == original
    # A failed re-export leaves the original frozen runtime loadable.
    assert FrozenEncoder(prepared.directory / "bundle.json").dimension == 544


def test_changed_threshold_does_not_mutate_existing_bundle(prepared):
    run_export(prepared)
    original = preserved_files(prepared)
    with pytest.raises(ValueError, match="replace an existing frozen bundle"):
        run_export(prepared, threshold=.8)
    assert preserved_files(prepared) == original
