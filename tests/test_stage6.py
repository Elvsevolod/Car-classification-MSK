"""Protocol and restart tests use tiny CPU models, not full ResNet training."""
import copy
import json
import random
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training import stage6
from training.hpo import ExperimentConfig, make_optimizer


def test_connected_frames_stay_together():
    rows = [{"vehicle_id": i, "image_id": f"{i}-a"} for i in range(30)]
    rows += [{"vehicle_id": 1, "image_id": "1-b"}]
    hashes = {row["image_id"]: row["image_id"] for row in rows}
    hashes.update({"0-a": "frame-x", "1-a": "frame-x", "1-b": "frame-y", "2-a": "frame-y"})
    split = stage6.frame_disjoint_inner_split(rows, range(30), hashes)
    assert split == stage6.frame_disjoint_inner_split(rows, range(30), hashes)
    stage6.audit_partitions(rows, hashes, split)
    assert any({0, 1, 2}.issubset(part) for part in map(set, split.values()))
    with pytest.raises(ValueError, match="Exact-frame leakage"):
        stage6.audit_partitions(rows, hashes, {"train": [0], "validation": list(range(1, 30))})
    with pytest.raises(ValueError, match="Identity leakage"):
        stage6.audit_partitions(rows, hashes, {"train": list(range(30)), "validation": [0]})


def test_step_sampler_is_deterministic_cross_camera_and_full_budget():
    rows = [{"label": label, "camera_id": camera} for label in range(6) for camera in (1, 2)]
    config = ExperimentConfig(identities_per_batch=3, images_per_identity=2)
    sampler = stage6.StepPKBatchSampler(rows, config, steps=11)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 11  # Old identity pass would yield only 2.
    assert batches == list(sampler)
    for batch in batches:
        labels = [rows[i]["label"] for i in batch]
        assert len(labels) == 6 and len(set(labels)) == 3
        for label in set(labels):
            assert labels.count(label) == 2
            assert {rows[i]["camera_id"] for i in batch if rows[i]["label"] == label} == {1, 2}
    sampler.set_epoch(1)
    assert batches != list(sampler)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.bnneck = torch.nn.Identity()
        self.classifier = torch.nn.Linear(2, 2)


def test_lr_prefix_is_identical_for_short_final_run():
    config = ExperimentConfig()
    budget = stage6.StepBudget()
    full, short = make_optimizer(TinyModel(), config), make_optimizer(TinyModel(), config)
    values = [stage6.set_step_learning_rates(full, config, budget, i) for i in range(4000)]
    prefix = [stage6.set_step_learning_rates(short, replace(config, epochs=5), budget, i)
              for i in range(1000)]
    assert prefix == values[:1000]
    assert values[399]["encoder"] == pytest.approx(config.encoder_lr)
    assert values[-1]["encoder"] == pytest.approx(config.encoder_lr * config.min_lr_ratio)
    assert all(v["head"] == pytest.approx(v["encoder"] * config.head_lr_multiplier) for v in values)


def test_early_stopping_waits_for_minimum_budget_and_patience():
    budget = stage6.StepBudget(max_steps=12, evaluation_interval=2, warmup_steps=2, min_steps=8, patience=2)
    history = [{"step": step, "validation": {"mAP_at_10": score}}
               for step, score in [(2, .3), (4, .2), (6, .1), (8, .1)]]
    assert not stage6.should_stop(history[:3], budget)
    assert stage6.should_stop(history, budget)
    history[-1]["validation"]["mAP_at_10"] = .4
    assert not stage6.should_stop(history, budget)


def _tiny_setup(monkeypatch):
    config = ExperimentConfig(identities_per_batch=2, images_per_identity=2)
    budget = stage6.StepBudget(max_steps=6, evaluation_interval=2, warmup_steps=2,
                               min_steps=4, patience=2)
    rows = [{"vehicle_id": identity, "image_id": f"{identity}-{camera}", "camera_id": camera}
            for identity in range(3) for camera in (1, 2)]

    class TinyDataset:
        def __init__(self, rows, *_args, **_kwargs):
            self.rows = rows

        def __getitem__(self, index):
            row = self.rows[index]
            vector = torch.rand(2) + random.random() + np.random.random()
            return vector, vector * .9, row["label"], row["image_id"]

        def __len__(self):
            return len(self.rows)

    def tiny_losses(model, clean, robust, labels, _config):
        output = model.classifier(model.backbone(robust))
        loss = torch.nn.functional.cross_entropy(output, labels)
        return {"loss": loss, "accuracy": (output.argmax(1) == labels).float().mean()}

    monkeypatch.setattr(stage6, "VehicleDataset", TinyDataset)
    monkeypatch.setattr(stage6, "experiment_losses", tiny_losses)
    return rows, config, budget


def test_restart_replays_unfinished_block_and_keeps_final_file(tmp_path, monkeypatch):
    rows, config, budget = _tiny_setup(monkeypatch)
    torch.manual_seed(7)
    initial = TinyModel().state_dict()

    def fit(name):
        model = TinyModel()
        model.load_state_dict(copy.deepcopy(initial))
        return stage6.fit_steps(model, rows, [0, 1], None, torch.device("cpu"), config, budget,
                                tmp_path / name / "weights", tmp_path / name / "results",
                                {"version": 1}, stop_steps=6)

    fit("uninterrupted")
    real_block = stage6.train_step_block

    def interrupt(*args):
        result = real_block(*args)
        if args[-1] == 2:
            raise RuntimeError("interrupted after updates, before checkpoint")
        return result

    monkeypatch.setattr(stage6, "train_step_block", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        fit("resumed")
    history = tmp_path / "resumed/results/history.json"
    history.write_text("broken JSON")  # last.pt, not this file, is authoritative.
    monkeypatch.setattr(stage6, "train_step_block", real_block)
    summary = fit("resumed")
    assert summary["completed_steps"] == summary["selected_steps"] == 6
    assert summary["image_presentations"] == 24
    paths = [tmp_path / name / "weights/final.pt" for name in ("uninterrupted", "resumed")]
    states = [torch.load(path, weights_only=False)["model"] for path in paths]
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert [r["step"] for r in json.loads(history.read_text())] == [2, 4, 6]
    old_bytes = paths[1].read_bytes()
    fit("resumed")
    assert paths[1].read_bytes() == old_bytes  # Cached outer evaluation remains valid.
    config.encoder_lr *= 2
    with pytest.raises(RuntimeError, match="Resume config/protocol changed"):
        fit("resumed")
    (tmp_path / "resumed/weights/last.pt").unlink()
    with pytest.raises(RuntimeError, match="Missing last.pt"):
        fit("resumed")


def test_inner_evaluation_seed_and_best_checkpoint(tmp_path, monkeypatch):
    rows, config, budget = _tiny_setup(monkeypatch)
    calls = []
    scores = iter([.05, .2, .4, .3])

    def evaluate(_model, _rows, identities, _device, **kwargs):
        calls.append((identities, kwargs["seed"]))
        return {"mAP_at_10": next(scores)}, .5

    monkeypatch.setattr(stage6, "evaluate_experiment", evaluate)
    summary = stage6.fit_steps(TinyModel(), rows, [0, 1], [2], torch.device("cpu"),
                               replace(config, seed=123), budget, tmp_path / "weights",
                               tmp_path / "results", {"version": 1})
    assert calls == [([2], stage6.SEED)] * 4
    assert summary["best_mAP_at_10"] == .4 and summary["selected_steps"] == 4
    assert summary["initial_validation"]["mAP_at_10"] == .05
    saved = torch.load(tmp_path / "weights/best_map.pt", weights_only=False)
    assert saved["step"] == 4


def test_screening_changes_one_component_at_a_time(tmp_path, monkeypatch):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(asdict(ExperimentConfig())))
    seen = []
    scores = iter([.4, .6, .65, .62])

    def fake_run(name, rows, train, val, config, budget, protocol, *args, **kwargs):
        assert train == [0, 1] and val == [2]
        assert kwargs.get("split") is None
        seen.append(config)
        return {"name": name, "config": asdict(config), "best_mAP_at_10": next(scores), "selected_steps": 2000}

    monkeypatch.setattr(stage6, "_run", fake_run)
    result = stage6.run_controlled_screening([], {"inner": {"train": [0, 1], "validation": [2]}},
                                            torch.device("cpu"), source, tmp_path, tmp_path)
    assert len(seen) == 4 and result["promoted"]
    expected = ["encoder_lr", "pooling", "metric_loss"]
    for first, second, field in zip(seen, seen[1:], expected):
        assert [key for key in asdict(first) if getattr(first, key) != getattr(second, key)] == [field]
    assert result["winner"]["name"] == "screening/avg_pool"
    assert not result["outer_calibration_or_validation_used"]


def test_nonpromising_candidate_never_trains_final_or_reads_outer(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Final phase should not run")

    monkeypatch.setattr(stage6, "_run", forbidden)
    protocol = {"outer": {"train": [1], "calibration": [2], "validation": [3]}}
    screening = {"protocol_sha256": stage6._digest(protocol), "budget": asdict(stage6.StepBudget()),
                 "winner": {"best_mAP_at_10": .3}, "promoted": False}
    result = stage6.run_controlled_seeds([], {"identities": protocol["outer"]}, protocol, screening,
                                       torch.device("cpu"), tmp_path, tmp_path)
    assert result["status"] == "not_promoted"


def test_final_seeds_keep_selected_steps_horizon_and_predeclared_export_seed(tmp_path, monkeypatch):
    protocol = {"outer": {"train": [0, 1], "calibration": [2], "validation": [3]}}
    config, budget = ExperimentConfig(), stage6.StepBudget()
    calls = []

    def fake_run(name, rows, train, val, config, budget, protocol, *args, **kwargs):
        calls.append((config.seed, train, val, budget.max_steps, kwargs["stop_steps"]))
        metrics = {key: .7 for key in ("mAP_at_10", "Rank_1", "Rank_5", "candidate_F1", "TNR", "candidate_score")}
        return {"name": name, "tuned_reranking": {"validation": metrics, "validation_quality_score": .38}}

    monkeypatch.setattr(stage6, "_run", fake_run)
    screening = {"protocol_sha256": stage6._digest(protocol), "budget": asdict(budget),
                 "winner": {"config": asdict(config), "selected_steps": 2400}, "promoted": True}
    result = stage6.run_controlled_seeds([], {"identities": protocol["outer"]}, protocol, screening,
                                       torch.device("cpu"), tmp_path, tmp_path, seeds=(1, 2, 3))
    assert calls == [(seed, [0, 1], None, 4000, 2400) for seed in (1, 2, 3)]
    assert result["selected_representative"] == {"name": "seed_1", "seed": 1}
    assert result["aggregate"]["mean_mAP_at_10"] == pytest.approx(.7)


def test_outer_evaluation_is_cached_for_frozen_weights(tmp_path, monkeypatch):
    rows, config, budget = _tiny_setup(monkeypatch)
    monkeypatch.setattr(stage6, "initialize_resnet_experiment",
                        lambda *_args: (TinyModel(), {"sha256": stage6.PRETRAINED_SHA256}))
    evaluations = []

    def evaluate(*args):
        evaluations.append(1)
        return {"validation": {"mAP_at_10": .6}}

    monkeypatch.setattr(stage6, "tuned_retrieval", evaluate)
    args = ("final_seeds/seed_1", rows, [0, 1], None, config, budget, {"version": 1},
            torch.device("cpu"), tmp_path / "results", tmp_path / "weights")
    first = stage6._run(*args, stop_steps=4, split={"identities": {"validation": [2]}})
    second = stage6._run(*args, stop_steps=4, split={"identities": {"validation": [2]}})
    assert first["tuned_reranking"] == second["tuned_reranking"]
    assert evaluations == [1]


def test_protocol_rechecks_image_bytes_and_refuses_stale_results(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    rows = [{"image_id": str(i), "vehicle_id": i} for i in range(10)]
    for row in rows:
        (dataset / "images" / f"{row['image_id']}.jpg").write_bytes(row["image_id"].encode())
    split = {"identities": {"train": list(range(8)), "calibration": [8], "validation": [9]}}
    first = stage6.prepare_protocol(rows, split, tmp_path / "results", dataset)
    assert first == stage6.prepare_protocol(rows, split, tmp_path / "results", dataset)
    (dataset / "images/0.jpg").write_bytes(b"changed frame")
    with pytest.raises(RuntimeError, match="Protocol/data changed"):
        stage6.prepare_protocol(rows, split, tmp_path / "results", dataset)
    (dataset / "images/0.jpg").write_bytes(b"9")
    with pytest.raises(ValueError, match="Exact-frame leakage"):
        stage6.prepare_protocol(rows, split, tmp_path / "results", dataset)
