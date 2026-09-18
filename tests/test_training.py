import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from training.osnet import (GeM, MixStyle, ReIDTrainerModel, VehicleOSNet,
                            load_encoder_from_onnx)
from training.hpo import (CameraAwarePKBatchSampler, ExperimentConfig,
                          circle_loss, initialize_experiment, make_optimizer,
                          metric_weight_for_epoch,
                          set_epoch_learning_rates, split_hpo_identities,
                          supervised_contrastive_loss)
from training.pipeline import (PKBatchSampler, batch_hard_triplet_loss, consistency_loss,
                               format_duration, mask_lower_center, progress_line)
from training.preprocessing import LETTERBOX_FILL, preprocess_mode, resize_crop
from training.stage3 import fit_stage3_seed
from training.stage4 import SimilarityPKBatchSampler, fit_fixed_epochs
from backend.core import STOCK_MODEL


def test_pk_sampler_makes_p_by_k_batches():
    rows = [{"label": label} for label in range(10) for _ in range(4)]
    sampler = PKBatchSampler(rows, identities_per_batch=3, images_per_identity=2)
    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        labels = [rows[index]["label"] for index in batch]
        assert len(labels) == 6
        assert len(set(labels)) == 3
        assert all(labels.count(label) == 2 for label in set(labels))


def test_camera_aware_sampler_uses_cross_camera_positives():
    rows = [{"label": label, "camera_id": camera}
            for label in range(4) for camera in (1, 2) for _ in range(2)]
    sampler = CameraAwarePKBatchSampler(rows, identities_per_batch=2, images_per_identity=2, seed=7)
    for batch in sampler:
        for label in {rows[index]["label"] for index in batch}:
            selected = [rows[index] for index in batch if rows[index]["label"] == label]
            assert {row["camera_id"] for row in selected} == {1, 2}


def test_hpo_identity_split_is_deterministic_and_disjoint():
    train, validation = split_hpo_identities(range(100), seed=7)
    assert (train, validation) == split_hpo_identities(range(100), seed=7)
    assert len(train) == 80 and len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(100))


def test_batch_hard_triplet_prefers_separated_identities():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1., 0.], [.9, .1], [-1., 0.], [-.9, -.1]])
    collapsed = torch.tensor([[1., 0.], [-1., 0.], [.9, .1], [-.9, -.1]])
    assert batch_hard_triplet_loss(separated, labels) < batch_hard_triplet_loss(collapsed, labels)


def test_supcon_prefers_separated_identities():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1., 0.], [.9, .1], [-1., 0.], [-.9, -.1]])
    collapsed = torch.tensor([[1., 0.], [-1., 0.], [.9, .1], [-.9, -.1]])
    assert supervised_contrastive_loss(separated, labels) < supervised_contrastive_loss(collapsed, labels)


def test_circle_loss_prefers_separated_identities():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1., 0.], [.9, .1], [-1., 0.], [-.9, -.1]])
    collapsed = torch.tensor([[1., 0.], [-1., 0.], [.9, .1], [-.9, -.1]])
    assert circle_loss(separated, labels) < circle_loss(collapsed, labels)


def test_metric_weight_warmup_reaches_configured_weight():
    config = ExperimentConfig(
        metric_weight=2., loss_weight_schedule="metric_warmup", metric_warmup_epochs=4)
    assert metric_weight_for_epoch(config, 0) == pytest.approx(.5)
    assert metric_weight_for_epoch(config, 3) == pytest.approx(2.)
    assert metric_weight_for_epoch(config, 8) == pytest.approx(2.)


def test_consistency_loss_prefers_matching_embeddings():
    first = torch.tensor([[1., 0.], [0., 1.]])
    same = first.clone()
    different = torch.tensor([[0., 1.], [1., 0.]])
    assert consistency_loss(first, same) == pytest.approx(0)
    assert consistency_loss(first, different) > consistency_loss(first, same)


def test_lower_center_mask_is_copy_and_keeps_outer_pixels():
    images = torch.ones(2, 3, 20, 20)
    masked = mask_lower_center(images, width_fraction=.5, height_fraction=.2, center_y=.7)
    assert torch.all(images == 1)
    assert torch.any(masked == 0)
    assert torch.all(masked[:, :, :10] == 1)
    assert torch.all(masked[:, :, :, :4] == 1)


def test_epoch_progress_contains_durations_and_remaining_epochs():
    timing = {"epoch_seconds": 3661, "train_seconds": 3000, "evaluation_seconds": 661,
              "elapsed_seconds": 7200, "estimated_remaining_seconds": 10800}
    line = progress_line(2, 5, timing, best_map=.71234)
    assert format_duration(3661) == "01:01:01"
    assert "Epoch 2/5" in line
    assert "осталось эпох: 3" in line
    assert "эпоха: 01:01:01" in line
    assert "ETA: 03:00:00" in line
    assert "best mAP: 0.7123" in line


def test_pytorch_encoder_matches_bundled_onnx():
    import onnxruntime as ort

    model = ReIDTrainerModel(num_classes=3).eval()
    count = load_encoder_from_onnx(model.encoder, STOCK_MODEL)
    assert count > 500
    array = np.random.default_rng(7).normal(size=(1, 3, 208, 208)).astype(np.float32)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    expected = ort.InferenceSession(str(STOCK_MODEL), sess_options=options, providers=["CPUExecutionProvider"]).run(
        ["output"], {"input": array})[0]
    with torch.inference_mode():
        actual = model.encoder(torch.from_numpy(array)).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_bnneck_model_and_separate_learning_rates():
    config = ExperimentConfig(epochs=4, encoder_lr=1e-4, head_lr_multiplier=10, use_bnneck=True)
    model, count = initialize_experiment(3, config, torch.device("cpu"))
    assert count > 500
    optimizer = make_optimizer(model, config)
    first = set_epoch_learning_rates(optimizer, config, epoch=0, total_epochs=4)
    assert first["encoder"] == pytest.approx(5e-5)
    assert first["head"] == pytest.approx(5e-4)
    model.eval()
    with torch.inference_mode():
        logits, raw, embedding = model(torch.randn(4, 3, 208, 208))
    assert logits.shape == (4, 3)
    assert raw.shape == embedding.shape == (4, 512)


def test_gem_pooling_is_trainable_and_keeps_embedding_shape():
    pooling = GeM()
    values = torch.rand(2, 8, 5, 7, requires_grad=True)
    pooled = pooling(values)
    pooled.sum().backward()
    assert pooled.shape == (2, 8, 1, 1)
    assert pooling.p.grad is not None and torch.isfinite(pooling.p.grad)

    encoder = VehicleOSNet(pooling="gem").eval()
    with torch.inference_mode():
        assert encoder(torch.rand(2, 3, 208, 208)).shape == (2, 512)


def test_gem_experiment_loads_stock_weights_except_new_exponent():
    config = ExperimentConfig(epochs=1, pooling="gem")
    model, count = initialize_experiment(3, config, torch.device("cpu"))
    assert count > 500
    assert model.backbone.global_pool.p.item() == pytest.approx(3.)


def test_mixstyle_is_training_only_and_keeps_shape(monkeypatch):
    layer = MixStyle(probability=1., alpha=.1)
    values = torch.cat((torch.zeros(1, 2, 4, 4), torch.ones(1, 2, 4, 4) * 10))
    monkeypatch.setattr(torch, "randperm", lambda length, device=None: torch.arange(
        length - 1, -1, -1, device=device))
    layer.eval()
    assert torch.equal(layer(values), values)
    layer.train()
    mixed = layer(values)
    assert mixed.shape == values.shape
    assert not torch.equal(mixed, values)


def test_similarity_sampler_groups_neighbor_identities_and_keeps_pk_shape():
    rows = [{"label": label, "camera_id": camera}
            for label in range(4) for camera in (1, 2)]
    neighbors = {
        0: [1, 2, 3], 1: [0, 2, 3],
        2: [3, 0, 1], 3: [2, 0, 1],
    }
    sampler = SimilarityPKBatchSampler(
        rows, identities_per_batch=2, images_per_identity=2,
        neighbors=neighbors, seed=7)
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        labels = [rows[index]["label"] for index in batch]
        assert len(labels) == 4
        assert len(set(labels)) == 2
        assert all(labels.count(label) == 2 for label in set(labels))
        assert all({rows[index]["camera_id"] for index in batch
                    if rows[index]["label"] == label} == {1, 2}
                   for label in set(labels))


def test_checkpoint_policy_tracks_map_f1_tnr_separately(tmp_path, monkeypatch):
    import training.hpo as hpo

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.bnneck = torch.nn.Identity()
            self.classifier = torch.nn.Linear(2, 2)

    values = [
        {"mAP": .70, "candidate_F1": .40, "TNR": .50},
        {"mAP": .69, "candidate_F1": .60, "TNR": .40},
        {"mAP": .71, "candidate_F1": .55, "TNR": .70},
    ]
    state = {"epoch": 0}

    def fake_train(*_args, **_kwargs):
        state["epoch"] += 1
        return {"loss": 1.0}

    def fake_evaluate(_model, _rows, identities, *_args, threshold=None, **_kwargs):
        if identities == [1]:
            return {"mAP": .5, "candidate_F1": .5, "TNR": .5}, .5
        return values[state["epoch"] - 1], threshold

    monkeypatch.setattr(hpo, "train_epoch", fake_train)
    monkeypatch.setattr(hpo, "evaluate_experiment", fake_evaluate)
    config = ExperimentConfig(epochs=3, use_bnneck=False)
    weights, results = tmp_path / "weights", tmp_path / "results"
    _, best = hpo.fit_selected(
        TinyModel(), [], {"identities": {"calibration": [1], "validation": [2]}},
        None, None, torch.device("cpu"), config, weights, results,
    )
    assert best["map"]["epoch"] == 3
    assert best["f1"]["epoch"] == 2
    assert best["tnr"]["epoch"] == 3
    assert torch.load(weights / "best_map.pt", weights_only=False)["epoch"] == 3
    assert torch.load(weights / "best_f1.pt", weights_only=False)["epoch"] == 2
    assert torch.load(weights / "best_tnr.pt", weights_only=False)["epoch"] == 3
    assert torch.load(weights / "last.pt", weights_only=False)["epoch"] == 3
    assert sorted(path.name for path in weights.glob("epoch_*.pt")) == [
        "epoch_01.pt", "epoch_02.pt", "epoch_03.pt"
    ]


def test_inner_candidate_resumes_from_last_epoch(tmp_path, monkeypatch):
    import training.hpo as hpo

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.bnneck = torch.nn.Identity()
            self.classifier = torch.nn.Linear(2, 2)

    state = {"trained_epochs": 0}

    def fake_train(*_args, **_kwargs):
        state["trained_epochs"] += 1
        return {"loss": 1.0}

    def fake_evaluate(*_args, **_kwargs):
        score = .5 + state["trained_epochs"] / 100
        return {"mAP": score}, .4

    monkeypatch.setattr(hpo, "train_epoch", fake_train)
    monkeypatch.setattr(hpo, "evaluate_experiment", fake_evaluate)
    weights, results = tmp_path / "weights", tmp_path / "results"
    first = hpo.fit_inner_candidate(
        TinyModel(), [], [1], None, None, torch.device("cpu"),
        ExperimentConfig(epochs=2, use_bnneck=False), weights, results,
    )
    assert first["completed_epochs"] == 2
    assert state["trained_epochs"] == 2

    resumed = hpo.fit_inner_candidate(
        TinyModel(), [], [1], None, None, torch.device("cpu"),
        ExperimentConfig(epochs=3, use_bnneck=False), weights, results,
    )
    assert resumed["completed_epochs"] == 3
    assert state["trained_epochs"] == 3
    assert len(json.loads((results / "history.json").read_text())) == 3
    assert torch.load(weights / "last.pt", weights_only=False)["epoch"] == 3


def test_square_stage3_preprocessing_matches_active_backend():
    from backend.core import preprocess

    image = Image.new("RGB", (80, 40), (10, 90, 180))
    box = (5, 4, 60, 30)
    assert np.array_equal(preprocess_mode(image, box, "square"), preprocess(image, box))


def test_letterbox_preserves_aspect_ratio_and_uses_mean_padding():
    crop = Image.new("RGB", (100, 50), (255, 0, 0))
    output = np.asarray(resize_crop(crop, "letterbox"))
    assert output.shape == (208, 208, 3)
    assert tuple(output[0, 0]) == LETTERBOX_FILL
    assert tuple(output[104, 104]) == (255, 0, 0)


def test_stage3_seed_resumes_from_last_epoch(tmp_path, monkeypatch):
    import training.stage3 as stage3

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.bnneck = torch.nn.Identity()
            self.classifier = torch.nn.Linear(2, 2)

    state = {"epoch": 0}

    def fake_train(*_args, **_kwargs):
        state["epoch"] += 1
        return {"loss": 1.0}

    def fake_evaluate(_model, _rows, identities, *_args, threshold=None, **_kwargs):
        value = .5 + state["epoch"] / 100
        result = {"mAP": value, "candidate_F1": value, "TNR": value,
                  "candidate_score": value}
        return result, .4 if threshold is None else threshold

    monkeypatch.setattr(stage3, "train_epoch", fake_train)
    monkeypatch.setattr(stage3, "evaluate_experiment", fake_evaluate)
    monkeypatch.setattr(stage3, "evaluate_active_retrieval", lambda *_args, **_kwargs: {
        "validation": {"mAP": .5, "candidate_score": .5}})
    split = {"identities": {"calibration": [1], "validation": [2]}}
    weights, results = tmp_path / "weights", tmp_path / "results"
    first = fit_stage3_seed(
        TinyModel(), [], split, None, None, torch.device("cpu"),
        ExperimentConfig(epochs=2, use_bnneck=False), weights, results,
        patience=10, minimum_epochs=10)
    assert first["completed_epochs"] == 2

    resumed = fit_stage3_seed(
        TinyModel(), [], split, None, None, torch.device("cpu"),
        ExperimentConfig(epochs=3, use_bnneck=False), weights, results,
        patience=10, minimum_epochs=10)
    assert resumed["completed_epochs"] == 3
    assert len(json.loads((results / "history.json").read_text())) == 3


def test_stage4_fixed_epochs_resume_without_outer_evaluation(tmp_path, monkeypatch):
    import training.stage4 as stage4

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.bnneck = torch.nn.Identity()
            self.classifier = torch.nn.Linear(2, 2)

    state = {"fail": True, "epochs": []}

    def fake_train(_model, _loader, _sampler, _optimizer, _device, _config, epoch):
        if epoch == 1 and state["fail"]:
            raise RuntimeError("simulated interruption")
        state["epochs"].append(epoch)
        return {"loss": 1.0, "metric_weight": 1.0}

    monkeypatch.setattr(stage4, "train_epoch", fake_train)
    weights, results = tmp_path / "weights", tmp_path / "results"
    config = ExperimentConfig(epochs=3, use_bnneck=False)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fit_fixed_epochs(
            TinyModel(), None, None, torch.device("cpu"), config, weights, results)
    assert len(json.loads((results / "history.json").read_text())) == 1

    state["fail"] = False
    summary = fit_fixed_epochs(
        TinyModel(), None, None, torch.device("cpu"), config, weights, results)
    assert summary["completed_epochs"] == 3
    assert state["epochs"] == [0, 1, 2]
    assert (weights / "final.pt").exists()
    assert len(json.loads((results / "history.json").read_text())) == 3
