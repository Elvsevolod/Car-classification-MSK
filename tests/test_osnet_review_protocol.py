"""CPU-only synthetic review runner checks; never train on project photographs."""
import copy
import random
from dataclasses import replace

import nbformat
import pytest
import torch

from backend.core import sha256
from training import osnet_review_protocol as review
from training.hpo import ExperimentConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyModel(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.bnneck = torch.nn.BatchNorm1d(4)
        self.classifier = torch.nn.Linear(4, classes)

    def embedding(self, images):
        return self.bnneck(self.backbone(images))

    def forward(self, images):
        assert self.training, "Training must restore train mode after evaluation"
        raw = self.backbone(images)
        embedding = self.bnneck(raw)
        return self.classifier(embedding), raw, embedding


class TinyDataset:
    def __init__(self, rows, *_args, **_kwargs):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clean = torch.tensor([row["label"], index, row["camera_id"], 1.], dtype=torch.float)
        robust = clean + torch.rand(4) * .1 + random.random() * .1
        return clean, robust, row["label"], row["image_id"]


@pytest.fixture
def context(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "AblationDataset", TinyDataset)
    monkeypatch.setattr(review, "initialize", lambda classes, config, variant, device: TinyModel(classes).to(device))
    monkeypatch.setattr(review, "check_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(review, "check_other_runs", lambda *args: None)

    def evaluate(model, ctx, variant, fold, diagnostics=False):
        model.eval()
        offset = {"K1_color32_legacy": .05, "K2_color32_fixed": .1}.get(variant.name, 0.)
        value = .5 + offset + .001 * int(model.bnneck.num_batches_tracked)
        return {"mean_map": value, "draws": {"regular_1": {"map": value, "selection_eligible": True}}}

    monkeypatch.setattr(review, "evaluate_draws", evaluate)
    rows = [{"vehicle_id": identity, "camera_id": camera, "image_id": f"image_{identity}_{camera}"}
            for identity in range(4) for camera in (1, 2)]
    return {"output": tmp_path / "run", "signature": "synthetic-review", "base": ExperimentConfig(),
            "rows": rows, "dataset": tmp_path, "device": torch.device("cpu"), "masks": {},
            "variants": {name: replace(variant, p=2) for name, variant in review.variants().items()},
            "seeds": (1, 2, 3), "budget": review.old.Budget(4, 2, 1),
            "manifest": {"inner": {"primary": {"train": [0, 1], "validation": [2, 3]},
                                   "alternate": {"train": [2, 3], "validation": [0, 1]}}},
            "split": {"identities": {"train": [0, 1, 2, 3]}}}


def state_equal(first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            state_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            state_equal(left, right)
    else:
        assert first == second


def test_primary_keeps_every_selectable_step_and_holdouts_out(context):
    result = review.fit(context, "B0_control", 1)
    assert result["stop_step"] == result["updates"] == result["lr_horizon"] == 4
    assert result["train_identities"] == 2 and result["train_images"] == 4
    assert set(result["checkpoints"]) == {"2", "4"}
    assert [item["step"] for item in result["history"]] == [2, 4]
    for step, entry in result["checkpoints"].items():
        path = context["output"] / entry["path"]
        assert sha256(path) == entry["sha256"]
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["step"] == int(step)
        assert int(checkpoint["model"]["bnneck.num_batches_tracked"]) == int(step)


def test_alternate_only_evaluates_after_fixed_training_and_keeps_full_horizon(monkeypatch, context):
    evaluate = review.evaluate_draws
    schedule = review.set_step_learning_rates
    evaluations, horizons = [], []

    def evaluate_once(model, ctx, variant, fold, diagnostics=False):
        assert fold == "alternate" and diagnostics
        assert int(model.bnneck.num_batches_tracked) == 2
        evaluations.append(fold)
        return evaluate(model, ctx, variant, fold, diagnostics)

    def schedule_check(optimizer, config, budget, step):
        horizons.append((step, budget.max_steps))
        return schedule(optimizer, config, budget, step)

    monkeypatch.setattr(review, "evaluate_draws", evaluate_once)
    monkeypatch.setattr(review, "set_step_learning_rates", schedule_check)
    result = review.fit(context, "B0_control", 1, "alternate", 2)
    assert evaluations == ["alternate"] and horizons == [(0, 4), (1, 4)]
    assert result["updates"] == result["stop_step"] == 2 and result["lr_horizon"] == 4
    assert all(item["validation"] is None for item in result["history"])
    assert set(result["checkpoints"]) == {"2"}


@pytest.mark.parametrize("fold,step", [("alternate", None), ("alternate", 3), ("final", 0), ("primary", 2)])
def test_only_registered_primary_boundaries_can_be_frozen(context, fold, step):
    with pytest.raises(ValueError):
        review.fit(context, "B0_control", 1, fold, step)


def test_selected_model_loads_exact_step_not_last_or_individual_best(context):
    result = review.fit(context, "B0_control", 1)
    model, _ = review.load_model(context, {**result, "stop_step": 2})
    step2 = torch.load(context["output"] / result["checkpoints"]["2"]["path"], weights_only=True)
    step4 = torch.load(context["output"] / result["checkpoints"]["4"]["path"], weights_only=True)
    state_equal(model.state_dict(), step2["model"])
    assert not torch.equal(model.backbone.weight, step4["model"]["backbone.weight"])
    assert not model.training


@pytest.mark.parametrize("at_step", [2, 4])
@pytest.mark.parametrize("interruption", ["selectable_before_last", "after_last"])
def test_resume_exact_across_authoritative_commit_boundary(monkeypatch, context, at_step, interruption):
    full = review.fit(context, "B0_control", 1)
    resumed_context = {**context, "output": context["output"].with_name("resumed")}
    save = review.old.save_checkpoint
    interrupted = False

    def interrupt(path, payload):
        nonlocal interrupted
        save(path, payload)
        match = (path.name == "last.pt" if interruption == "after_last" else path.name.startswith("step_"))
        if not interrupted and match and payload["step"] == at_step:
            interrupted = True
            raise KeyboardInterrupt("synthetic interruption")

    monkeypatch.setattr(review.old, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        review.fit(resumed_context, "B0_control", 1)
    monkeypatch.setattr(review.old, "save_checkpoint", save)
    resumed = review.fit(resumed_context, "B0_control", 1)
    for step in ["2", "4"]:
        assert full["checkpoints"][step]["sha256"] == resumed["checkpoints"][step]["sha256"]
    first = torch.load(context["output"] / "primary/B0_control/seed_1/last.pt", weights_only=True)
    second = torch.load(resumed_context["output"] / "primary/B0_control/seed_1/last.pt", weights_only=True)
    for field in ["model", "optimizer", "history", "checkpoints"]:
        state_equal(first[field], second[field])


def test_completed_fit_noop_and_configuration_checksum_guards(monkeypatch, context):
    result = review.fit(context, "B0_control", 1)
    monkeypatch.setattr(review, "initialize", lambda *args: pytest.fail("Completed fit must not initialize/retrain"))
    monkeypatch.setattr(review, "evaluate_draws", lambda *args, **kwargs: pytest.fail("Completed fit must not reevaluate"))
    assert review.fit(context, "B0_control", 1) == result
    with pytest.raises(ValueError, match="configuration changed"):
        review.fit({**context, "signature": "changed"}, "B0_control", 1)
    checkpoint = context["output"] / result["checkpoints"]["2"]["path"]
    review.write_json(checkpoint, {"corrupted": True})
    with pytest.raises(ValueError, match="checkpoint changed"):
        review.fit(context, "B0_control", 1)


def test_completed_last_checkpoint_guard(context):
    review.fit(context, "B0_control", 1)
    review.write_json(context["output"] / "primary/B0_control/seed_1/last.pt", {"corrupted": True})
    with pytest.raises(ValueError, match="authoritative checkpoint changed"):
        review.fit(context, "B0_control", 1)


def test_load_model_rejects_checkpoint_with_wrong_step_even_if_hash_matches(context):
    result = review.fit(context, "B0_control", 1)
    changed = copy.deepcopy(result)
    changed["checkpoints"]["4"] = changed["checkpoints"]["2"]
    with pytest.raises(ValueError, match="step/signature mismatch"):
        review.load_model(context, changed)


def test_pilot_never_evaluates_outer_or_selects_recipe(monkeypatch, context):
    monkeypatch.setattr(review, "evaluate_final", lambda *args: pytest.fail("No outer in pilot"))
    monkeypatch.setattr(review, "export_model", lambda *args: pytest.fail("No export in pilot"))
    result = review.run(context)
    assert result["complete"] and not result["outer_evaluated"] and not result["promoted"]
    assert len(result["primary"]) == 5 and {r["seed"] for r in result["primary"]} == {1}
    assert not (context["output"] / "selection.json").exists()
    assert len(list(context["output"].rglob("last.pt"))) == 5
    monkeypatch.setattr(review, "fit", lambda *args: pytest.fail("Completed pilot must not repeat training"))
    assert review.run(context) == result


def test_confirm_freezes_primary_before_alternate_and_keeps_matched_fusion_delta(monkeypatch, context):
    review.run(context)
    fit, alternate_steps = review.fit, []

    def checked_fit(ctx, name, seed, fold="primary", stop_step=None):
        if fold == "alternate":
            frozen = review.old.load_json(ctx["output"] / "selection.json")
            assert stop_step == frozen["aggregate"][name]["step"]
            alternate_steps.append((name, seed, stop_step))
        return fit(ctx, name, seed, fold, stop_step)

    monkeypatch.setattr(review, "fit", checked_fit)
    monkeypatch.setattr(review, "evaluate_final", lambda *args: pytest.fail("No outer in confirmation"))
    result = review.run(context, "confirm_inner")
    assert len(result["primary"]) == len(result["alternate"]) == len(alternate_steps) == 15
    assert result["selection"]["winner"] == "K2_color32_fixed"
    assert not result["outer_evaluated"] and not result["promoted"]
    paired = result["alternate_summary"]["fixed_minus_legacy_fusion"]
    assert paired["seed_deltas"] == pytest.approx([.05, .05, .05])
    assert paired["mean"] == pytest.approx(.05)


def test_final_requires_explicit_guard_before_training(monkeypatch, context):
    monkeypatch.setattr(review, "fit", lambda *args: pytest.fail("No training without explicit outer guard"))
    with pytest.raises(ValueError, match="ALLOW_OUTER_EVALUATION"):
        review.run(context, "final")


def test_final_freezes_all_checkpoints_before_first_outer_and_uses_full_train(monkeypatch, context):
    review.run(context)
    review.run(context, "confirm_inner")
    visited = []

    def outer(ctx, summary):
        final = review.old.load_json(ctx["output"] / "final_selection.json")
        assert {r["variant"] for r in final} == {"B0_control", "K1_color32_legacy", "K2_color32_fixed"}
        assert summary in final
        for item in final:
            assert item["train_identities"] == 4 and item["train_images"] == 8
            assert item["lr_horizon"] == 4
            assert item["validation"] is None
            entry = item["checkpoints"][str(item["stop_step"])]
            assert sha256(ctx["output"] / entry["path"]) == entry["sha256"]
        visited.append(summary["variant"])
        result = {"conditions": {"original": {"reranked": {"mAP_at_10": .5, "candidate_F1": .5, "TNR": .5}}},
                  "per_query": {method: {"q": {"vehicle_id": 7, "ap": .5}} for method in ("raw", "reranked")}}
        directory = ctx["output"] / "final" / summary["variant"] / f"seed_{summary['seed']}"
        review.write_json(directory / "evaluation.json", result)
        for name in ("thresholds.json", "encoder.onnx", "bundle.json", "export.json"):
            review.write_json(directory / name, {"synthetic_placeholder": True})
        return result

    monkeypatch.setattr(review, "evaluate_final", outer)
    monkeypatch.setattr(review, "evaluate_draws", lambda *args, **kwargs: pytest.fail("No holdout in final fit"))
    result = review.run(context, "final", allow_outer=True)
    assert len(visited) == 3 and result["outer_evaluated"] and not result["promoted"]
    assert set(result["paired_vs_control"]) == {"K1_color32_legacy", "K2_color32_fixed"}
    assert all(values["raw"]["mean_delta"] == 0 for values in result["paired_vs_control"].values())
    assert review.run(context, "final", allow_outer=True) == result
    assert len(visited) == 3


def test_later_phases_require_pilot(context):
    with pytest.raises(ValueError, match="pilot first"):
        review.run(context, "confirm_inner")


def test_junk_diagnostics_never_contribute_to_selection_mean(monkeypatch):
    identities = {"q1": 1, "q2": 1, "unknown": 2, "positive": 1, "negative": 3, "junk": 1}
    rows = [{"image_id": image_id, "vehicle_id": identity,
             "camera_id": 2 if image_id in {"positive", "negative"} else 1} for image_id, identity in identities.items()]
    protocols = {
        "regular_1": {"query_ids": ["q1", "unknown"], "gallery_ids": ["positive", "negative"], "selection_eligible": True},
        "regular_2": {"query_ids": ["q2", "unknown"], "gallery_ids": ["positive", "negative"], "selection_eligible": True},
        "same_camera_junk_1": {"query_ids": ["q1", "unknown"], "gallery_ids": ["junk", "positive", "negative"],
                               "selection_eligible": False},
    }
    context = {"rows": rows, "manifest": {"draws": {"primary": protocols}}}
    encoded_ids = []

    def encode(model, selected_rows, ctx, variant, masked=False):
        encoded_ids.append(([r["image_id"] for r in selected_rows], masked))
        vectors = {"q1": [1., 0.], "q2": [0., 1.], "unknown": [0., 1.],
                   "positive": [1., 0.], "negative": [.6, .8], "junk": [1., 0.]}
        if masked:
            vectors["q1"] = [0., 1.]
        return {row["image_id"]: vectors[row["image_id"]] for row in selected_rows}

    monkeypatch.setattr(review.old, "encode", encode)
    regular = review.evaluate_draws(None, context, None, "primary")
    diagnostic = review.evaluate_draws(None, context, None, "primary", diagnostics=True)
    assert regular["mean_map"] == diagnostic["mean_map"] == pytest.approx(.75)
    assert len(regular["draws"]) == 2 and len(diagnostic["draws"]) == 3
    first = diagnostic["draws"]["regular_1"]
    assert first["confidence"]["known"]["count"] == first["confidence"]["unknown"]["count"] == 1
    assert first["confidence"]["known"]["mean"] == pytest.approx(1.)
    assert first["confidence"]["unknown"]["mean"] == pytest.approx(.8)
    masks = first["automatic_mask_diagnostics"]
    assert set(masks) == {"masked_query", "masked_gallery", "masked_both"}
    assert masks["masked_query"]["map"] == pytest.approx(.5)
    assert masks["masked_query"]["confidence"]["known"]["mean"] == pytest.approx(.8)
    assert "automatic_mask_diagnostics" not in diagnostic["draws"]["same_camera_junk_1"]
    assert len(encoded_ids) == 3 and [item[1] for item in encoded_ids] == [False, False, True]
    assert all(len(ids) == len(set(ids)) for ids, _ in encoded_ids)


@pytest.mark.parametrize("artifact", ["step_00002.pt", "last.pt", "summary.json", "history.json"])
def test_cached_phase_rejects_tampered_stage_artifacts(monkeypatch, context, artifact):
    review.run(context)
    monkeypatch.setattr(review, "fit", lambda *args: pytest.fail("Corruption must not trigger training"))
    review.write_json(context["output"] / "primary/B0_control/seed_1" / artifact, {"changed": True})
    with pytest.raises(ValueError, match="phase artifact changed"):
        review.run(context)


def test_confirmation_rejects_modified_pilot_before_training(monkeypatch, context):
    review.run(context)
    pilot = review.old.load_json(context["output"] / "pilot.json")
    pilot["primary"][0]["updates"] += 1
    review.write_json(context["output"] / "pilot.json", pilot)
    monkeypatch.setattr(review, "fit", lambda *args: pytest.fail("Changed pilot must not train confirmation"))
    with pytest.raises(ValueError, match="phase summary changed"):
        review.run(context, "confirm_inner")


def test_cached_confirmation_rejects_selection_changes(context):
    review.run(context)
    result = review.run(context, "confirm_inner")
    result["selection"]["winner"] = "B0_control"
    review.write_json(context["output"] / "confirm_inner.json", result)
    with pytest.raises(ValueError, match="phase selection changed"):
        review.run(context, "confirm_inner")


def test_review_notebook_compiles_and_preserves_explicit_phase_guards():
    notebook = nbformat.read(review.VARIANT / "train_osnet_review_protocol.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "review notebook", "exec")
    settings = next(cell.source for cell in notebook.cells if cell.id == "settings")
    assert "PHASE = " in settings
    assert "ALLOW_OUTER_EVALUATION = " in settings
    assert "ALLOW_CPU_TRAINING = " in settings
    training = next(cell.source for cell in notebook.cells if cell.id == "training")
    assert "run(context, phase=PHASE, allow_outer=ALLOW_OUTER_EVALUATION)" in training
