"""Synthetic CPU checks only; no project photographs or training weights loaded."""
import copy
import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from backend.core import sha256
from training import overnight_training as night
from training.hpo import ExperimentConfig
from training.osnet_ablations import MemoryBank, losses


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


class TinyModel(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.bnneck = torch.nn.BatchNorm1d(4)
        self.classifier = torch.nn.Linear(4, classes, bias=False)

    def embedding(self, images):
        return self.bnneck(self.backbone(images))

    def forward(self, images):
        raw = self.backbone(images)
        embedding = self.bnneck(raw)
        return self.classifier(embedding), raw, embedding


class TinyDataset:
    seen_ids = set()

    def __init__(self, rows, *_args, **_kwargs):
        self.rows = rows
        self.seen_ids.update(row["vehicle_id"] for row in rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clean = torch.tensor([row["label"], index, row["camera_id"], 1.], dtype=torch.float32)
        robust = clean + torch.rand(4) * .1 + random.random() * .1 + np.random.random() * .1
        return clean, robust, row["label"], row["image_id"]


@pytest.fixture
def context(monkeypatch, tmp_path):
    monkeypatch.setattr(night, "AblationDataset", TinyDataset)
    monkeypatch.setattr(night, "initialize", lambda classes, config, variant, device: TinyModel(classes).to(device))
    TinyDataset.seen_ids = set()
    return {"output": tmp_path / "new-run", "signature": "synthetic-overnight", "base": ExperimentConfig(),
            "rows": [{"vehicle_id": identity, "camera_id": camera, "image_id": f"i{identity}_{camera}"}
                     for identity in range(6) for camera in (1, 2)],
            "dataset": tmp_path, "device": torch.device("cpu"), "masks": {},
            "variants": {name: replace(variant, p=2) for name, variant in night.review.variants().items()},
            "seeds": (1, 2, 3), "budget": night.review.old.Budget(8, 2, 1),
            "manifest": {"inner": {"primary": {"train": [0, 1], "validation": [2, 3]},
                                   "alternate": {"train": [2, 3], "validation": [0, 1]}}},
            "split": {"identities": {"train": [0, 1, 2, 3], "calibration": [4], "validation": [5]}}}


def tiny_job(name="R1_control", **kwargs):
    return replace(night.training_jobs()[name], stop_step=6, lr_horizon=8, **kwargs)


def equal_state(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def test_frozen_five_job_grid_changes_one_hypothesis():
    jobs = night.training_jobs()
    assert len(jobs) == 5
    assert {job.stop_step for job in jobs.values()} == {800}
    assert {job.lr_horizon for job in jobs.values()} == {1700}
    assert jobs["K2_ce_sqrt544"].ce_scale == math.sqrt(544)
    assert jobs["R1_control"].relation_weight == 0
    assert jobs["R1_kd01"].relation_weight == .1
    assert jobs["R1_kd1"].relation_weight == 1
    for job in jobs.values():
        job.validate()


@pytest.mark.parametrize("change", [{"name": "../old"}, {"ce_scale": 0}, {"ce_scale": float("nan")},
                                   {"relation_weight": -1}, {"stop_step": 1800},
                                   {"base_variant": "K2_color32_fixed", "relation_weight": 1}])
def test_invalid_jobs_rejected(change):
    with pytest.raises(ValueError):
        replace(night.training_jobs()["R1_control"], **change).validate()


def test_relations_invariant_to_teacher_coordinate_rotation():
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(6, 4, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    original = night.relational_gram([features, features])
    changed = night.relational_gram([features @ rotation, features])
    torch.testing.assert_close(original, changed)
    assert night.relational_loss(features, original) < 1e-12


def test_relations_support_independent_dimensions_and_detach_teacher():
    student = torch.randn(4, 3, requires_grad=True)
    teacher_a = torch.randn(4, 7, requires_grad=True)
    teacher_b = torch.randn(4, 11, requires_grad=True)
    target = night.relational_gram([teacher_a, teacher_b])
    loss = night.relational_loss(student, target)
    loss.backward()
    assert student.grad is not None
    assert teacher_a.grad is None and teacher_b.grad is None


def test_scale_changes_ce_only_and_logs_shared_feature_gradients():
    torch.manual_seed(4)
    model = TinyModel(2)
    scaled = copy.deepcopy(model)
    images, labels = torch.randn(4, 4), torch.tensor([0, 0, 1, 1])
    config = replace(ExperimentConfig(), consistency_weight=0)
    base = night.training_losses(model, images, images, labels, config,
                                 night.training_jobs()["K2_ce1"], diagnostics=True)
    bigger = night.training_losses(scaled, images, images, labels, config,
                                   night.training_jobs()["K2_ce_sqrt544"], diagnostics=True)
    for name in ("metric", "embedding_norm", "embedding_std", "classifier_norm", "train_accuracy"):
        torch.testing.assert_close(base[name], bigger[name], atol=0, rtol=0)
    assert base["classification"] != bigger["classification"]
    torch.testing.assert_close(bigger["logits_std"], base["logits_std"] * math.sqrt(544))
    assert base["ce_feature_grad_norm"] > 0 and base["metric_feature_grad_norm"] > 0
    model.eval(), scaled.eval()
    torch.testing.assert_close(model.embedding(images), scaled.embedding(images), atol=0, rtol=0)


def test_unscaled_loss_matches_historical_control():
    torch.manual_seed(8)
    first = TinyModel(2)
    second = copy.deepcopy(first)
    clean, robust, labels = torch.randn(4, 4), torch.randn(4, 4), torch.tensor([0, 0, 1, 1])
    config = ExperimentConfig()
    historical, _ = losses(first, clean, robust, labels, torch.arange(4), config,
                           night.review.variants()["R1_resolution256"], MemoryBank(0), 0)
    current = night.training_losses(second, clean, robust, labels, config, night.training_jobs()["R1_control"])
    for key in historical:
        torch.testing.assert_close(historical[key], current[key], atol=0, rtol=0)
    historical["loss"].backward()
    current["loss"].backward()
    for left, right in zip(first.parameters(), second.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_frozen_teachers_do_not_update_bn_or_receive_gradients():
    teachers = [TinyModel(2).eval().requires_grad_(False) for _ in range(3)]
    states = [copy.deepcopy(teacher.state_dict()) for teacher in teachers]
    model = TinyModel(2)
    images, labels = torch.randn(4, 4), torch.tensor([0, 0, 1, 1])
    values = night.training_losses(model, images, images, labels, ExperimentConfig(),
                                   night.training_jobs()["R1_kd1"], teachers)
    values["loss"].backward()
    for teacher, state in zip(teachers, states):
        assert not teacher.training and all(p.grad is None for p in teacher.parameters())
        equal_state(state, teacher.state_dict())
    assert values["relation"] > 0


def test_trainable_teacher_rejected():
    images = torch.randn(4, 4)
    with pytest.raises(ValueError, match="frozen"):
        night.training_losses(TinyModel(2), images, images, torch.tensor([0, 0, 1, 1]),
                              ExperimentConfig(), night.training_jobs()["R1_kd1"], [TinyModel(2)])


@pytest.mark.parametrize("fold", ["final", "outer", "unknown"])
def test_outer_training_not_allowed(context, fold):
    with pytest.raises(ValueError, match="inner folds"):
        night.fit_job(context, tiny_job(), 1, fold)


def test_identity_overlap_rejected(context):
    context["manifest"]["inner"]["primary"]["train"].append(5)
    with pytest.raises(ValueError, match="protected holdout"):
        night.fit_job(context, tiny_job(), 1)
    assert not context["output"].exists()


def test_rng_round_trip():
    state = night.capture_rng()
    first = (random.random(), np.random.random(), torch.rand(4))
    night.restore_rng(state)
    second = (random.random(), np.random.random(), torch.rand(4))
    equal_state(first, second)


def test_fixed_step_no_selection_and_completed_resume(context):
    summary = night.fit_job(context, tiny_job(), 1)
    assert summary["updates"] == summary["stop_step"] == 6
    assert summary["lr_horizon"] == 8 and not summary["outer_evaluated"]
    assert set(summary["checkpoints"]) == {"2", "4", "6"}
    assert TinyDataset.seen_ids == {0, 1}
    assert summary["teachers"] == []
    assert "ce_feature_grad_norm" in summary["history"][0]["train"]
    model, variant = night.load_job_model(context, summary)
    assert not model.training and variant.name == "R1_resolution256"
    assert night.fit_job(context, tiny_job(), 1) == summary


def test_interrupted_job_replays_only_last_unsaved_block(context, monkeypatch):
    baseline_context = {**context, "output": context["output"].parent / "baseline"}
    baseline = night.fit_job(baseline_context, tiny_job(), 1)
    original = night.training_losses
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(night, "training_losses", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        night.fit_job(context, tiny_job(), 1)
    checkpoint = context["output"] / "training/primary/R1_control/seed_1/last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["step"] == 2 and "rng" in saved and "optimizer" in saved
    monkeypatch.setattr(night, "training_losses", original)
    resumed = night.fit_job(context, tiny_job(), 1)
    first, _ = night.load_job_model(baseline_context, baseline)
    second, _ = night.load_job_model(context, resumed)
    equal_state(first.state_dict(), second.state_dict())
    equal_state(baseline["history"], resumed["history"])


def test_boundary_pause_saves_resume_state_and_does_not_change_signature(context):
    baseline_context = {**context, "output": context["output"].parent / "baseline"}
    baseline = night.fit_job(baseline_context, tiny_job(), 1)
    with pytest.raises(night.TrainingPaused, match="saved step 2/6"):
        night.fit_job(context, tiny_job(), 1, should_stop=lambda: True)
    directory = context["output"] / "training/primary/R1_control/seed_1"
    assert not (directory / "summary.json").exists()
    saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 2 and saved["rng"] and saved["optimizer"]
    assert set(saved["checkpoints"]) == {"2"}
    resumed = night.fit_job(context, tiny_job(), 1, should_stop=lambda: False)
    assert resumed["signature"] == baseline["signature"]
    first, _ = night.load_job_model(baseline_context, baseline)
    second, _ = night.load_job_model(context, resumed)
    equal_state(first.state_dict(), second.state_dict())
    equal_state(baseline["history"], resumed["history"])


def test_final_boundary_finishes_even_if_time_budget_expired(context):
    job = replace(tiny_job(), stop_step=2)
    summary = night.fit_job(context, job, 1, should_stop=lambda: True)
    assert summary["updates"] == 2
    assert night.fit_job(context, job, 1, should_stop=lambda: True) == summary


@pytest.fixture
def teachers(context, tmp_path):
    source = {**context, "output": tmp_path / "source", "signature": "source-signature"}
    source["output"].mkdir()
    summaries = []
    for seed in source["seeds"]:
        path = source["output"] / f"teacher_{seed}.pt"
        torch.save({"seed": seed}, path)
        summaries.append({"seed": seed, "fold": "primary", "variant": "R1_resolution256",
                          "context_signature": source["signature"], "signature": f"signature_{seed}",
                          "train_identities": 2, "checkpoints": {"800": {"path": path.name, "sha256": sha256(path)}}})
    return source, summaries


def test_teachers_require_exact_same_fold_and_three_frozen_seeds(context, teachers):
    source, summaries = teachers
    assert len(night.teacher_specs(context, source, summaries, "primary")) == 3
    with pytest.raises(ValueError, match="exactly three"):
        night.teacher_specs(context, source, summaries[:2], "primary")
    with pytest.raises(ValueError, match="matching inner-fold"):
        night.teacher_specs(context, source, summaries, "alternate")
    summaries[0]["seed"] = 999
    with pytest.raises(ValueError, match="predetermined"):
        night.teacher_specs(context, source, summaries, "primary")


def test_full_train_or_other_partition_teacher_rejected(context, teachers):
    source, summaries = teachers
    summaries[0]["fold"] = "final"
    with pytest.raises(ValueError, match="matching inner-fold"):
        night.teacher_specs(context, source, summaries, "primary")
    summaries[0]["fold"] = "primary"
    source = copy.deepcopy(source)
    source["manifest"]["inner"]["primary"] = source["manifest"]["inner"]["alternate"]
    with pytest.raises(ValueError, match="identities differ"):
        night.teacher_specs(context, source, summaries, "primary")


def test_corrupted_completed_checkpoint_rejected(context):
    summary = night.fit_job(context, tiny_job(), 1)
    path = context["output"] / summary["checkpoints"]["6"]["path"]
    path.write_bytes(b"broken synthetic checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        night.fit_job(context, tiny_job(), 1)
    with pytest.raises(ValueError, match="path/hash"):
        night.load_job_model(context, summary)
