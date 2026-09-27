"""D0 contracts. Random local test weights are NEVER an experimental quality result."""
import hashlib
import socket
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision.transforms import v2

from training import dino_experiment as experiment, dino_frozen as dino


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def bank(count, dimension, seed=13):
    return experiment.base.normalize(np.random.default_rng(seed).normal(size=(count, dimension)).astype(np.float32))


def test_pool_excludes_registers_and_normalizes_each_part():
    torch.manual_seed(7)
    tokens = torch.randn(2, 261, 384)
    value = dino.pool_tokens(tokens)
    tokens[:, 1:5] = 100000
    torch.testing.assert_close(dino.pool_tokens(tokens), value, atol=0, rtol=0)
    expected = torch.cat([torch.nn.functional.normalize(tokens[:, 0], dim=1),
                          torch.nn.functional.normalize(tokens[:, 5:].mean(1), dim=1)], 1) / np.sqrt(2.)
    torch.testing.assert_close(value, expected, atol=0, rtol=0)
    np.testing.assert_allclose(value.norm(dim=1), 1., atol=2e-7, rtol=0)
    assert value.shape == (2, 768)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "zero_cls", "zero_patches"])
def test_invalid_tokens_fail_without_invented_vectors(kind):
    tokens = torch.ones(1, 261, 384)
    if kind == "shape": tokens = tokens[:, :-1]
    if kind == "dtype": tokens = tokens.double()
    if kind == "nan": tokens[0, 0, 0] = float("nan")
    if kind == "zero_cls": tokens[:, 0] = 0
    if kind == "zero_patches": tokens[:, 5:] = 0
    with pytest.raises(ValueError): dino.pool_tokens(tokens)


def test_preprocessing_is_official_tensor_resize_not_pil_resize():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (21, 35, 3), dtype=np.uint8))
    official = v2.Compose([v2.ToImage(), v2.Resize((256, 256), antialias=True),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225))])
    torch.testing.assert_close(dino.transform()(image), official(image), atol=0, rtol=0)


@pytest.fixture
def pinned_files(tmp_path, monkeypatch):
    payload = b"synthetic fixture only, NOT DINO weights"
    path = tmp_path / "model.safetensors"
    path.write_bytes(payload)
    monkeypatch.setattr(dino, "WEIGHT_SIZE", len(payload))
    monkeypatch.setattr(dino, "WEIGHT_SHA256", hashlib.sha256(payload).hexdigest())
    blobs = {}
    for name in dino.BLOBS:
        content = f"fixture {name}".encode()
        (tmp_path / name).write_bytes(content)
        blobs[name] = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
    monkeypatch.setattr(dino, "BLOBS", blobs)
    return tmp_path


def test_verified_files_need_no_network(pinned_files, monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("No network for local verification"))
    assert len(dino.model_files(pinned_files)) == 5
    monkeypatch.setattr(dino, "activate_dependencies", lambda: {})
    assert dino.download_model(pinned_files) == dino.model_files(pinned_files)


@pytest.mark.parametrize("name", ["model.safetensors", *dino.BLOBS])
def test_corrupted_official_files_fail(pinned_files, name):
    path = pinned_files / name
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="official"): dino.model_files(pinned_files)


def test_missing_model_requests_access_not_substitution(tmp_path):
    with pytest.raises(FileNotFoundError, match="accept its license yourself"):
        dino.model_files(tmp_path)


def test_download_requires_local_login_and_pins_revision(tmp_path, monkeypatch):
    dino.activate_dependencies()
    import huggingface_hub
    calls = []
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(dino.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="hf auth login"): dino.download_model(tmp_path)
    assert not calls
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "fake-test-token-not-a-credential")
    def run(command, **kwargs):
        calls.append(command)
        assert "fake-test-token-not-a-credential" not in str(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(dino.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="no mirror/fallback"): dino.download_model(tmp_path)
    assert calls[0][calls[0].index("--revision")+1] == dino.REVISION
    assert dino.MODEL_ID in calls[0]


def test_no_silent_device_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(dino.device_for("cpu")) == "cpu"
    for device in ("cuda", "mps", "auto"):
        with pytest.raises(ValueError, match="no automatic fallback"): dino.device_for(device)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="Disable MPS CPU fallback"): dino.device_for("mps")


def test_actual_transformers_architecture_frozen_offline_images(tmp_path, monkeypatch):
    """Exercise real ViT-S/16 code with RANDOM test weights, not pretrained quality."""
    dino.activate_dependencies()
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    torch.manual_seed(9)
    config = DINOv3ViTConfig(hidden_size=384, intermediate_size=1536, num_hidden_layers=12,
                           num_attention_heads=6, patch_size=16, num_register_tokens=4)
    model = DINOv3ViTModel(config)
    checkpoint = tmp_path / "test_only_weights"
    model.save_pretrained(checkpoint)
    del model
    # Only this isolated fixture bypasses the official hash check; production never does.
    monkeypatch.setattr(dino, "model_files", lambda directory: {"test_fixture": "random"})
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("Network forbidden during inference"))
    encoder = dino.FrozenDino(checkpoint)
    assert encoder.model.training is False and not any(p.requires_grad for p in encoder.model.parameters())
    before = {k: v.clone() for k, v in encoder.model.state_dict().items()}
    images = tmp_path / "images"; images.mkdir()
    entries = []
    for i, suffix in enumerate(("jpg", "PNG", "jpeg")):
        pixels = np.random.default_rng(i).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"i{i}.{suffix}")
        entries.append({"image_id": f"i{i}", "x": 1, "y": 1, "w": 31, "h": 23})
    expected = encoder.encode_rows(entries, tmp_path, 1)
    assert expected.shape == (3, 768) and expected.dtype == np.float32
    for batch in (8, 16, 32):
        np.testing.assert_allclose(encoder.encode_rows(entries, tmp_path, batch), expected, rtol=0, atol=2e-5)
    changed = [{**row, "vehicle_id": 999, "camera_id": 111, "time": "ignored"} for row in entries[::-1]]
    np.testing.assert_allclose(encoder.encode_rows(changed, tmp_path, 16), expected[::-1], rtol=0, atol=2e-5)
    np.testing.assert_allclose(encoder.encode_rows(entries[1:2], tmp_path), expected[1:2], rtol=0, atol=2e-5)
    assert all(torch.equal(before[k], v) for k, v in encoder.model.state_dict().items())
    with pytest.raises(ValueError, match="outside"):
        encoder.encode_rows([{**entries[0], "w": 32}], tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        encoder.encode_rows([{**entries[0], "image_id": "missing"}], tmp_path)
    Image.new("RGB", (32, 24)).save(images / "i0.png")
    with pytest.raises(ValueError, match="Expected one JPEG/PNG.*found 2"):
        encoder.encode_rows(entries[:1], tmp_path)
    encoder.model.train()
    with pytest.raises(ValueError, match="eval mode"): encoder.encode_rows(entries, tmp_path)


def test_mixture_is_weighted_cosine_not_coordinate_average():
    r1, d0 = bank(40, 512), bank(40, 768, 14)
    result = experiment.mixed_features(r1, d0)
    assert result.shape == (40, 1280) and result.dtype == np.float32
    # Check the algebra in float64: float32 BLAS diagonal reductions over 1280
    # terms otherwise obscure this formula test. Runtime vectors remain float32.
    a, b, c = (v.astype(np.float64) for v in (r1, d0, result))
    np.testing.assert_allclose(c @ c.T, .9*(a @ a.T)+.1*(b @ b.T), rtol=0, atol=6e-7)
    with pytest.raises(ValueError): experiment.mixed_features(r1[:1], d0)


@pytest.fixture
def context(tmp_path):
    queries = [{"image_id": f"q{i}", "vehicle_id": i, "camera_id": 0} for i in range(4)]
    gallery = [{"image_id": f"g{i}_{j}", "vehicle_id": i, "camera_id": 1} for i in range(3) for j in range(5)]
    protocol = {"query_ids": [r["image_id"] for r in queries], "gallery_ids": [r["image_id"] for r in gallery],
                "selection_eligible": True}
    manifest = {"inner": {"train": [20], "validation": [0, 1, 2, 3]}, "draws": {"regular_1": protocol}}
    score = {"rows": queries+gallery, "manifest": manifest}
    rows = experiment.base.development_rows(score)
    reference = tmp_path / "reference"; reference.mkdir()
    r1 = bank(len(rows), 512)
    np.save(reference / "features.npy", r1)
    experiment.base.write_json(reference / "metrics.json", experiment.base.score_features(score, r1, rows))
    return {"output": tmp_path / "run", "signature": "synthetic_only", "rows": rows, "dataset": tmp_path,
        "score_context": score, "manifest": {"draws": manifest["draws"], "reference_directory": str(reference),
        "model_directory": str(tmp_path / "fake"), "runtime": {"device": "cpu"},
        "settings": {"chunk_size": 7, "batch_size": 16}}}


class FakeEncoder:
    def __init__(self, context):
        self.model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        self.values = dict(zip([r["image_id"] for r in context["rows"]], bank(len(context["rows"]), 768, 17)))
        self.calls = []

    def encode_rows(self, rows, dataset, batch_size=16):
        self.calls.append([r["image_id"] for r in rows])
        return np.stack([self.values[r["image_id"]] for r in rows])


def test_extract_resume_and_corrupted_cache(context):
    encoder = FakeEncoder(context)
    method = encoder.encode_rows
    def interrupt(rows, dataset, batch_size):
        if len(encoder.calls) == 1: raise RuntimeError("simulated interruption")
        return method(rows, dataset, batch_size)
    encoder.encode_rows = interrupt
    with pytest.raises(RuntimeError, match="interruption"): experiment.extract(context, encoder)
    encoder.encode_rows = method
    first = experiment.extract(context, encoder)
    assert len(encoder.calls) == 3  # First completed block reused, not re-extracted.
    np.testing.assert_array_equal(experiment.extract(context, encoder), first)
    assert len(encoder.calls) == 3
    path = context["output"] / "tasks/features_D0/block_00000.npy"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError): experiment.extract(context, encoder)


def test_end_to_end_synthetic_reports_resume_and_no_training(context, monkeypatch):
    encoder = FakeEncoder(context)
    monkeypatch.setattr(experiment, "check_inputs", lambda c: None)
    monkeypatch.setattr(dino, "FrozenDino", lambda *a: encoder)
    def deny(*a, **k): pytest.fail("No training or calibration in D0")
    monkeypatch.setattr(experiment.base, "train_arm", deny)
    monkeypatch.setattr(experiment.base.policy, "calibrate_policy", deny)
    first = experiment.run(context)
    assert first["status"] == "complete" and tuple(first["reports"]) == experiment.SYSTEMS
    assert first["optimizer_updates"] == first["bn_updates"] == 0
    assert first["threshold_fit"] is first["original_outer_evaluation"] is first["promoted"] is False
    assert first["stream_probe"]["exact_raw_and_graph_top10"]
    count = len(encoder.calls)
    assert experiment.run(context) == first and len(encoder.calls) == count
    metrics = experiment.base.old.load_json(context["output"] / "tasks/evaluate_R1_control/result.json")
    assert experiment.comparison(metrics, metrics)["regular_1"]["raw"]["delta_map"] == 0


def test_stream_guard_rejects_batch_feature_drift(context):
    encoder = FakeEncoder(context)
    values = encoder.encode_rows(context["rows"], context["dataset"])
    r1 = bank(len(context["rows"]), 512)
    method = encoder.encode_rows
    encoder.encode_rows = lambda rows, dataset, batch_size: np.roll(method(rows, dataset, batch_size), 1, axis=1)
    with pytest.raises(ValueError, match="drift"): experiment.probe(context, encoder, r1, values)


def test_manifest_mutation_detected():
    manifest = {"protected": {}, "source_sha256": {}}
    context = {"manifest": manifest, "signature": experiment.base.digest(manifest)}
    context["manifest"]["promoted"] = True
    with pytest.raises(ValueError, match="context changed"): experiment.check_inputs(context)


def test_fixed_source_hashes_and_no_outer_selection():
    settings = experiment.base.old.load_json(dino.VARIANT / "configs/d0_v1.json")
    source = experiment.base.ROOT / settings["source_run"]
    experiment.base.verify_files({str(source / "manifest.json"): settings["source_manifest_sha256"],
                                  str(source / "results.json"): settings["source_results_sha256"]})
    assert settings["dino_weight"] == .1 and settings["graph"] == experiment.base.policy.POLICIES["less_graph"]
    assert settings["optimizer_updates"] == settings["bn_updates"] == 0
    assert settings["original_outer_evaluation"] is settings["threshold_fit"] is settings["promoted"] is False
    assert settings["wall_time_limit"] is None

