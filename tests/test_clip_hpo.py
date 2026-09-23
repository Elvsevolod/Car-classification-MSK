"""CPU-only HPO tests: optimizer continuity, recovery, pruning and fair selection."""
import copy
import json
import random
from dataclasses import asdict, replace

import numpy as np
import optuna
import pytest
import torch

from training import clip_hpo as hpo
from training.clip_experiment import image_lr_factor
from training.clip_reid import image_losses


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Linear(2, 3)
        self.classifier = torch.nn.Linear(3, 2, bias=False)
        self.classifier_proj = torch.nn.Linear(3, 2, bias=False)
        self.text_encoder = torch.nn.Linear(3, 3)
        self.prompt_learner = torch.nn.Linear(3, 3)

    def set_stage(self, stage):
        self.train()
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not name.startswith(("text_encoder.", "prompt_learner.")))

    def forward(self, images):
        features = self.image_encoder(images)
        return [self.classifier(features), self.classifier_proj(features)], [features] * 3, features

    def image_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                if not key.startswith(("text_encoder.", "prompt_learner."))}

    def load_image_state(self, state):
        self.load_state_dict(state, strict=False)


def setup_tiny(monkeypatch):
    rows = [{"vehicle_id": i, "label": i, "image_id": f"{i}-{j}", "camera_id": j}
            for i in range(2) for j in (1, 2)]
    protocol = {"inner": {"train": [0, 1], "validation": [2, 3]}}
    config = hpo.TrialConfig(identities_per_batch=2, images_per_identity=2, image_lr=.001)
    asset = {"features": torch.tensor([[1., .5, .2], [.1, .3, 1.]]), "prompt_sha256": "prompt",
             "signature": {"config": {"seed": config.seed}}}

    class Dataset:
        def __init__(self, rows, *_args, **_kwargs):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            return torch.rand(2) + random.random() + np.random.rand(), row["label"], row["image_id"]

    monkeypatch.setattr(hpo, "ClipDataset", Dataset)
    monkeypatch.setattr(hpo, "require_disk", lambda *_a, **_k: 100)
    monkeypatch.setattr(hpo, "evaluate_inner", lambda model, *_a, **_k: {
        "mAP_at_10": float(torch.sigmoid(model.image_encoder.weight.sum()).detach())})
    return rows, protocol, config, asset


def test_optimizer_head_groups_and_loss_match_variant_one_at_baseline():
    model, config = TinyModel(), hpo.TrialConfig()
    images, labels, texts = torch.rand(4, 2), torch.tensor([0, 0, 1, 1]), torch.rand(2, 3)
    baseline = image_losses(model, images, labels, texts)
    actual = hpo.losses_and_accuracy(model, images, labels, texts, config)
    for key in baseline:
        torch.testing.assert_close(baseline[key], actual[key])
    for key in ("accuracy_head_768", "accuracy_head_512", "accuracy_image_text"):
        assert 0 <= actual[key] <= 1
    config = replace(config, head_lr_multiplier=10)
    optimizer = hpo.make_optimizer(model, config)
    rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert rates["classifier.weight"] == pytest.approx(5e-5)
    assert rates["classifier_proj.weight"] == pytest.approx(5e-5)
    assert rates["image_encoder.weight"] == pytest.approx(5e-6)
    assert rates["image_encoder.bias"] == pytest.approx(1e-5)
    assert not any(name.startswith(("text_encoder", "prompt_learner")) for name in rates)


def test_rungs_and_mid_epoch_resume_equal_uninterrupted_training(tmp_path, monkeypatch):
    rows, protocol, config, asset = setup_tiny(monkeypatch)
    torch.manual_seed(7)
    initial = copy.deepcopy(TinyModel().state_dict())

    def run(name, target):
        model = TinyModel()
        model.load_state_dict(initial)
        return hpo.fit_candidate(model, rows, protocol, asset, "cpu", config, target,
                    tmp_path/name/"weights", tmp_path/name/"results", "experiment")

    run("full", 60)
    run("resumed", 15)
    run("resumed", 35)
    original = hpo.train_epoch

    def fail(*args):
        result = original(*args)
        if args[-1] == 40:
            raise RuntimeError("Interrupted after optimizer updates")
        return result

    monkeypatch.setattr(hpo, "train_epoch", fail)
    with pytest.raises(RuntimeError, match="Interrupted"):
        run("resumed", 60)
    monkeypatch.setattr(hpo, "train_epoch", original)
    result = run("resumed", 60)
    assert result["completed_epochs"] == 60
    a = torch.load(tmp_path/"full/weights/image_last.pt", weights_only=True)
    b = torch.load(tmp_path/"resumed/weights/image_last.pt", weights_only=True)
    assert a["best"] == b["best"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for key, value in a["optimizer"]["state"].items():
        for field in value:
            torch.testing.assert_close(value[field], b["optimizer"]["state"][key][field], rtol=0, atol=0)
    for record in b["history"]:
        assert record["encoder_lr"] == pytest.approx(config.image_lr * image_lr_factor(record["epoch"]))
    # Missing secondary JSON is repaired from the authoritative checkpoint.
    (tmp_path/"resumed/results/image_history.json").unlink()
    monkeypatch.setattr(hpo, "train_epoch", lambda *_a: pytest.fail("A completed candidate retrained"))
    run("resumed", 60)
    assert len(json.loads((tmp_path/"resumed/results/image_history.json").read_text())) == 60


def test_epoch_zero_and_signature_protection(tmp_path, monkeypatch):
    rows, protocol, config, asset = setup_tiny(monkeypatch)
    model = TinyModel()
    initial = model.image_state()
    scores = iter([.8, .5, .6])
    monkeypatch.setattr(hpo, "evaluate_inner", lambda *_a, **_k: {"mAP_at_10": next(scores)})
    result = hpo.fit_candidate(model, rows, protocol, asset, "cpu", config, 2,
                               tmp_path/"w", tmp_path/"r", "experiment")
    assert result["best"]["epoch"] == 0
    saved = torch.load(tmp_path/"w/image_best.pt", weights_only=True)
    for key in initial:
        torch.testing.assert_close(saved["model"][key], initial[key])
    with pytest.raises(RuntimeError, match="signature changed"):
        hpo.fit_candidate(model, rows, protocol, asset, "cpu", replace(config, head_lr_multiplier=10), 3,
                          tmp_path/"w", tmp_path/"r", "experiment")
    with pytest.raises(ValueError, match="own prompt"):
        hpo.fit_candidate(model, rows, protocol, asset, "cpu", replace(config, seed=1), 3,
                          tmp_path/"w2", tmp_path/"r2", "experiment")


def test_pruning_after_warmup_reports_replayed_epochs(tmp_path, monkeypatch):
    rows, protocol, config, asset = setup_tiny(monkeypatch)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(
        n_startup_trials=4, n_warmup_steps=11, n_min_trials=3))
    for _ in range(4):
        study.add_trial(optuna.trial.create_trial(value=.9, intermediate_values={i: .9 for i in range(15)}))
    trial = study.ask()
    monkeypatch.setattr(hpo, "evaluate_inner", lambda *_a, **_k: {"mAP_at_10": .1})
    result = hpo.fit_candidate(TinyModel(), rows, protocol, asset, "cpu", config, 15,
                               tmp_path/"w", tmp_path/"r", "experiment", trial=trial)
    assert result["status"] == "pruned" and result["completed_epochs"] == 12
    assert len(study.get_trials()[4].intermediate_values) == 12


def test_sqlite_resume_same_trial_and_do_not_repeat_completed_search(tmp_path):
    plan = hpo.SearchPlan(trials=4)
    study = hpo.open_study(tmp_path, "signature", plan)
    interrupted = True

    def run(name, config, target, trial):
        nonlocal interrupted
        if interrupted:
            interrupted = False
            trial.report(.5, 0)
            raise KeyboardInterrupt()
        return {"name": name, "status": "complete", "config": asdict(config), "completed_epochs": target,
                "best": {"epoch": target, "validation": {"mAP_at_10": .5 + trial.number/100}}}

    with pytest.raises(KeyboardInterrupt):
        hpo.run_search(study, plan, run, tmp_path)
    assert len(study.get_trials()) == 4  # Four anchors; first remains RUNNING, not duplicated.
    study = hpo.open_study(tmp_path, "signature", plan)
    assert hpo.next_trial(study).number == 0
    report = hpo.run_search(study, plan, run, tmp_path)
    assert len(report["trials"]) == 4
    assert all(t["state"] == "COMPLETE" for t in report["trials"])
    assert report["trials"][0]["params"] == hpo.BASE_PARAMS
    assert report["trials"][1]["params"]["head_lr_multiplier"] == 10
    hpo.run_search(study, plan, lambda *_a: pytest.fail("Search repeated"), tmp_path)
    with pytest.raises(RuntimeError, match="signature changed"):
        hpo.open_study(tmp_path, "other", plan)


def test_continuation_selection_is_frozen_and_resumable(tmp_path):
    candidates = [{"name": f"trial_{i:03d}", "config": asdict(hpo.TrialConfig()),
                    "best": {"epoch": 15, "validation": {"mAP_at_10": i/10}}} for i in range(5)]
    calls = []

    def run(name, config, target, _trial):
        calls.append(name)
        return {"config": asdict(config), "completed_epochs": target,
                "best": {"epoch": target, "validation": {"mAP_at_10": int(name[-1])/10}}}

    result = hpo.continue_rung(candidates, 4, 35, run, tmp_path, "stage2")
    assert calls == ["trial_004", "trial_003", "trial_002", "trial_001"]
    assert result[0]["name"] == "trial_004"
    hpo.continue_rung(candidates, 4, 35, lambda *_a: pytest.fail("Rung repeated"), tmp_path, "stage2")
    with pytest.raises(RuntimeError, match="changed"):
        hpo.continue_rung(candidates, 2, 35, run, tmp_path, "stage2")


def test_lock_manifest_plan_and_confirmation_gate(tmp_path):
    with hpo.experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another kernel"):
            with hpo.experiment_lock(tmp_path):
                pass
    with hpo.experiment_lock(tmp_path):
        pass
    hpo.checked_json(tmp_path/"manifest.json", {"seeds": (1, 2)})
    hpo.checked_json(tmp_path/"manifest.json", {"seeds": [1, 2]})
    with pytest.raises(RuntimeError, match="changed"):
        hpo.checked_json(tmp_path/"manifest.json", {"seeds": [2, 3]})
    plan = hpo.SearchPlan()
    plan.validate()
    with pytest.raises(ValueError):
        replace(plan, prune_after_epoch=4).validate()
    refs = {"clip_variant_1": .7182}
    winner = {"best": {"validation": {"mAP_at_10": .721}}}
    assert not hpo.confirmation_decision(winner, refs, plan)["run"]
    winner["best"]["validation"]["mAP_at_10"] = .73
    assert hpo.confirmation_decision(winner, refs, plan)["run"]


def test_new_notebook_valid_no_training_on_import():
    import ast
    import nbformat
    notebook = nbformat.read(hpo.VARIANT/"train_clip_hpo.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None


def test_prompt_reuse_is_read_only_and_extra_seeds_train_their_own(tmp_path, monkeypatch):
    rows, protocol, _, _ = setup_tiny(monkeypatch)
    previous, weights, results = tmp_path/"previous", tmp_path/"weights", tmp_path/"results"
    signature = hpo.run_signature(protocol, hpo.ClipConfig())
    saved = {"signature": signature, "history": [{"epoch": i} for i in range(1, 61)],
             "prompt": TinyModel().prompt_learner.state_dict()}
    original = previous/"weights/prompt_last.pt"
    hpo._save_checkpoint(original, saved)
    original_hash = hpo.sha256(original)
    trained = []
    monkeypatch.setattr(hpo, "load_pretrained", lambda *_a, **_k: TinyModel())
    monkeypatch.setattr(hpo, "class_text_features", lambda *_a: torch.ones(2, 512))

    def prompt_stage(model, selected, device, config, protocol, folder, output, dataset):
        trained.append(config.seed)
        hpo._save_checkpoint(folder/"prompt_last.pt", {"signature": hpo.run_signature(protocol, config),
            "history": saved["history"], "prompt": model.prompt_learner.state_dict()})

    monkeypatch.setattr(hpo, "run_prompt_stage", prompt_stage)
    first = hpo.prompt_assets(rows, protocol, "cpu", "unused", weights, results, previous=previous)
    assert trained == []
    assert first["features"].shape == (2, 512)
    assert hpo.sha256(original) == original_hash
    monkeypatch.setattr(hpo, "load_pretrained", lambda *_a, **_k: pytest.fail("Cached prompts reloaded model"))
    again = hpo.prompt_assets(rows, protocol, "cpu", "unused", weights, results, previous=previous)
    assert again["prompt_sha256"] == first["prompt_sha256"]
    monkeypatch.setattr(hpo, "load_pretrained", lambda *_a, **_k: TinyModel())
    second = hpo.prompt_assets(rows, protocol, "cpu", "unused", weights, results, seed=hpo.SEED+1, previous=previous)
    assert trained == [hpo.SEED+1]
    assert second["signature"]["config"]["seed"] == hpo.SEED+1


def test_full_orchestration_twelve_four_two_and_two_seeds(tmp_path, monkeypatch):
    rows, protocol, _, asset = setup_tiny(monkeypatch)
    source = tmp_path/"source.pt"
    source.touch()
    monkeypatch.setattr(hpo, "sha256", lambda _path: hpo.CHECKPOINT_SHA256)
    monkeypatch.setattr(hpo, "prepare_protocol", lambda *_a: protocol)
    monkeypatch.setattr(hpo, "manifest", lambda *_a: {"test_signature": True})
    monkeypatch.setattr(hpo, "load_references", lambda *_a: {"clip_variant_1": .7, "clean_osnet": .78})
    monkeypatch.setattr(hpo, "load_pretrained", lambda *_a, **_k: TinyModel())
    monkeypatch.setattr(hpo, "prompt_assets", lambda *_a, **_k: asset)
    calls = []

    def fit(model, rows, protocol, asset, device, config, target, weights, output, *_a):
        calls.append((weights.name, target, config.seed))
        weights.mkdir(parents=True, exist_ok=True)
        result = {"status": "complete", "config": asdict(config), "completed_epochs": target,
                  "best": {"epoch": target, "validation": {"mAP_at_10": .8 + config.head_lr_multiplier/1000}}}
        hpo._save_checkpoint(weights/"image_best.pt", {"epoch": target})
        hpo.write_json(output/"summary.json", result)
        return result

    monkeypatch.setattr(hpo, "fit_candidate", fit)
    report = hpo.run_experiment(rows, {}, "cpu", variant=tmp_path/"new", source=source,
                                previous=tmp_path/"old", allow_cpu=True)
    assert [target for _, target, _ in calls].count(15) == 12
    assert [target for _, target, _ in calls].count(35) == 4
    assert [target for _, target, _ in calls].count(60) == 4  # Two finalists + two extra seeds.
    assert len(report["seeds"]) == 3
    assert set(seed for _, _, seed in calls) == {hpo.SEED, hpo.SEED+1, hpo.SEED+2}
    assert len(list((tmp_path/"new/weights").glob("*/best_after_*.pt"))) == 6
    assert not report["outer_test_used"] and not report["mvp_changed"]
    count = len(calls)
    hpo.run_experiment(rows, {}, "cpu", variant=tmp_path/"new", source=source,
                       previous=tmp_path/"old", allow_cpu=True)
    assert len(calls) == count


def test_numerically_failed_trial_does_not_block_remaining_trials(tmp_path):
    plan = hpo.SearchPlan(trials=4)
    study = hpo.open_study(tmp_path, "signature", plan)

    def run(name, config, target, trial):
        if trial.number == 0:
            raise FloatingPointError("NaN loss")
        return {"status": "complete", "config": asdict(config), "completed_epochs": target,
                "best": {"epoch": target, "validation": {"mAP_at_10": .8}}}

    summary = hpo.run_search(study, plan, run, tmp_path)
    assert [t["state"] for t in summary["trials"]] == ["FAIL", "COMPLETE", "COMPLETE", "COMPLETE"]
    assert summary["trials"][0]["failure"] == "NaN loss"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available to this process")
def test_optimizer_checkpoint_roundtrip_on_mps(tmp_path, monkeypatch):
    rows, protocol, config, asset = setup_tiny(monkeypatch)
    torch.manual_seed(13)
    initial = copy.deepcopy(TinyModel().state_dict())

    def run(name, target):
        model = TinyModel()
        model.load_state_dict(initial)
        model = model.to("mps")
        try:
            return hpo.fit_candidate(model, rows, protocol, asset, "mps", config, target,
                                     tmp_path/name/"w", tmp_path/name/"r", "test")
        finally:
            del model
            hpo.release_device("mps")

    run("continuous", 3)
    run("resumed", 1)
    run("resumed", 3)
    a = torch.load(tmp_path/"continuous/w/image_last.pt", map_location="cpu", weights_only=True)
    b = torch.load(tmp_path/"resumed/w/image_last.pt", map_location="cpu", weights_only=True)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=1e-5, atol=1e-6)
