"""v38: fixed full-train recipe, leakage guards, resumable training and honest exports."""
import copy
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

from training import transreid_full_train as experiment
from tests.test_transreid_night import TinyImages, TinyModel, random_features
from tests.test_transreid_system import vectors, rows


@pytest.fixture(autouse=True)
def threads():
    count = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(count)


def plan():
    return experiment.base.old.load_json(experiment.VARIANT / "configs/refit_v1.json")


def test_exact_one_refit_recipe_and_no_outer_search():
    p = plan()
    source = experiment.base.old.load_json(experiment.vision.VARIANT / "configs/night_v1.json")
    experiment.validate_plan(p, source)
    assert p["training"]["checkpoints"] == [1800]
    assert p["trial"] == experiment.TRIAL
    assert p["train_identities"] == 925 and p["new_encoder_weight"] == .1
    assert p["training"]["identities_per_batch"]*p["training"]["images_per_identity"] == 16
    assert len(experiment.CASES) == 2 and not p["inner_evaluation"]
    assert p["initializer"] == "official_DeiT_pretrained_not_old_T12"
    assert p["wall_time_limit"] is None and not p["promoted"] and not p["threshold_fit"]


@pytest.mark.parametrize("change", ["lr", "weight", "steps", "threshold", "checkpoint", "recipe"])
def test_recipe_changes_are_not_silent(change):
    p = plan()
    source = experiment.base.old.load_json(experiment.vision.VARIANT / "configs/night_v1.json")
    if change == "lr": p["trial"]["encoder_lr"] = 1e-3
    if change == "weight": p["new_encoder_weight"] = .05
    if change == "steps": p["selected_step"] = 2250
    if change == "threshold": p["threshold_fit"] = True
    if change == "checkpoint": p["training"]["checkpoints"].append(1200)
    if change == "recipe": p["training"]["label_smoothing"] = .2
    with pytest.raises(ValueError): experiment.validate_plan(p, source)


def test_full_train_ids_include_old_holdout_but_never_outer_ids():
    train = [{**r, "vehicle_id": 10+i//2} for i, r in enumerate(rows(6, "t"))]
    calibration = [{**r, "vehicle_id": 30} for r in rows(11, "c")]
    validation = [{**r, "vehicle_id": 40} for r in rows(11, "v")]
    protocols = {s: {"query_ids": [items[0]["image_id"]], "gallery_ids": [r["image_id"] for r in items[1:]]}
                 for s, items in (("calibration", calibration), ("validation", validation))}
    splits = {"identities": {"train": [10, 11, 12], "calibration": [30], "validation": [40]}}
    target, identities = experiment.full_train_rows(train+calibration+validation, splits, protocols, [10, 11], 3)
    assert identities == [10, 11, 12] and {r["label"] for r in target} == {0, 1, 2}
    assert {r["image_id"] for r in target} == {r["image_id"] for r in train}
    with pytest.raises(ValueError): experiment.full_train_rows(train+calibration+validation, splits, protocols, [30], 3)
    broken = copy.deepcopy(splits); broken["identities"]["train"].append(40)
    with pytest.raises(ValueError, match="leakage"):
        experiment.full_train_rows(train+calibration+validation, broken, protocols, [10, 11], 4)


@pytest.fixture
def context(tmp_path, monkeypatch):
    p = plan(); p["selected_step"] = 6
    s = p["training"]
    s.update(screen_steps=2, final_steps=6, checkpoints=[6], save_interval=1, log_interval=1,
             warmup_steps=1, identities_per_batch=2, images_per_identity=2)
    target = [{"image_id": f"t{i}", "vehicle_id": 20+i//2, "label": i//2, "camera_id": i%2} for i in range(6)]
    q = [{"image_id": f"q{i}", "vehicle_id": i, "camera_id": 0} for i in range(4)]
    g = [{"image_id": f"g{i}_{j}", "vehicle_id": i, "camera_id": 1} for i in range(3) for j in range(5)]
    c = {"output": tmp_path / "run", "signature": "synthetic", "device": torch.device("cpu"),
        "settings": s, "trials": [copy.deepcopy(experiment.TRIAL)], "target": target, "paths": {},
        "schedule": [[0, 1, 2, 3], [0, 1, 4, 5], [2, 3, 4, 5]]*2,
        "rows": q+g, "dataset": tmp_path, "manifest": {"plan": p, "train_ids": [20, 21, 22],
            "train_images": 6, "threshold": .03, "source_directory": str(tmp_path / "v37"),
            "protocols": {split: {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g]}
                          for split in ("calibration", "validation")},
            "v24_directory": str(tmp_path / "v24")}}
    def factory(classes, architecture, **kw):
        return TinyModel(classes, architecture)
    # Reuse real train_until/new_model/optimizer/schedule/checkpoint/resume code on tiny tensors.
    monkeypatch.setattr(experiment.vision, "ReIDModel", factory)
    monkeypatch.setattr(experiment.vision, "Images", TinyImages)
    c["manifest"]["weight_path"] = "unused-by-tiny-model"
    monkeypatch.setattr(experiment, "check_inputs", lambda c: None)
    monkeypatch.setattr(experiment.night, "disk_guard", lambda *a: None)
    return c


def test_full_train_resume_is_exact_and_old_weights_untouched(context):
    c = context; trial = c["trials"][0]
    full = {**c, "output": c["output"] / "full"}
    resumed = {**c, "output": c["output"] / "resumed"}
    sentinel = c["output"].parent / "old_t12.pt"; sentinel.write_bytes(b"historical checkpoint")
    a = experiment.night.train_until(full, trial, 6)
    experiment.night.train_until(resumed, trial, 2)
    b = experiment.night.train_until(resumed, trial, 6)
    left, right = [torch.load(experiment.base.resume_path(x["output"] / "training" / trial["id"]), weights_only=True)
                   for x in (full, resumed)]
    assert left["history"] == right["history"]
    for k, v in left["model"].items(): torch.testing.assert_close(v, right["model"][k], rtol=0, atol=0)
    assert set(a["checkpoints"]) == set(b["checkpoints"]) == {"6"}
    frozen = experiment.freeze_candidate(resumed, b)
    assert experiment.require_candidate(resumed) == frozen
    loaded = experiment.load_model(resumed)
    assert len(loaded.heads[0].weight) == 3 and not loaded.training
    assert not any(v.requires_grad for v in loaded.parameters())
    assert sentinel.read_bytes() == b"historical checkpoint"


def test_outer_guard_requires_final_checkpoint_and_correct_provenance(context):
    c = context
    with pytest.raises(ValueError, match="closed"): experiment.features(c, "validation")
    early = experiment.night.train_until(c, c["trials"][0], 2)
    with pytest.raises(ValueError, match="final checkpoint"): experiment.freeze_candidate(c, early)
    trained = experiment.night.train_until(c, c["trials"][0], 6)
    cp = Path(trained["checkpoints"]["6"]["path"])
    saved = torch.load(cp, weights_only=True); saved["signature"] = "old-inner-740"
    torch.save(saved, cp); trained["checkpoints"]["6"]["sha256"] = experiment.base.sha256(cp)
    with pytest.raises(ValueError, match="full-train T12"): experiment.freeze_candidate(c, trained)


def test_changed_checkpoint_after_freeze_fails(context):
    c = context; trained = experiment.night.train_until(c, c["trials"][0], 6)
    experiment.freeze_candidate(c, trained)
    Path(trained["checkpoints"]["6"]["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Protected file"): experiment.require_candidate(c)


def install_reference(c):
    runtime = experiment.inference
    old = Path(c["manifest"]["source_directory"])
    original = vectors(len(c["rows"]))[:, :2048]
    evaluations = {}
    for split in ("calibration", "validation"):
        q, g = experiment.protocol_rows(c, split)
        ranked = runtime.rank(original, len(q), "V25_control")
        report = {"split": split, "system": "V25_control", "threshold": .03,
            **runtime.dual.policy.evaluate(q, g, ranked, .03, "raw_top1"),
            **runtime.dual.policy.query_diagnostics(q, g, ranked)}
        top10, _ = runtime.dual.policy.predictions(q, g, ranked, .03, "raw_top1")
        for qid, ids in top10.items(): report["per_query"][qid]["ranking"]["top10"] = ids
        experiment.base.write_json(old / "tasks" / f"{split}_V25_control" / "result.json", report)
        runtime.export_arrays(old / "tasks" / f"export_{split}_V25_control" / "export", q, g, original, "V25_control", .03)
        runtime.export_arrays(Path(c["manifest"]["v24_directory"]) / "tasks" / f"cached_{split}" / "export",
                              q, g, original, "V25_control", .03)
        evaluations[split] = {"T12_w10": report}
    experiment.base.write_json(old / "results.json", {"evaluations": evaluations})
    return original


@pytest.mark.parametrize("drift", [False, True])
def test_stream_probe_checks_actual_decisions_and_rejects_feature_drift(context, monkeypatch, drift):
    c = context; original = install_reference(c)
    trained = experiment.night.train_until(c, c["trials"][0], 6)
    experiment.freeze_candidate(c, trained)
    extra = random_features(c["rows"], 384)
    monkeypatch.setattr(experiment.vision, "image_paths", lambda *a: {})
    def encode(model, rows, *a, **kw):
        values = random_features(rows, model.dimension)
        return -values if drift else values
    monkeypatch.setattr(experiment.vision, "encode", encode)
    lookup = {r["image_id"]: i for i, r in enumerate(c["rows"])}
    class CachedOriginal:
        def __init__(self, *a): pass
        def encode_rows(self, rows, *a): return original[[lookup[r["image_id"]] for r in rows]]
    monkeypatch.setattr(experiment.inference.dual, "DualRoleEncoder", CachedOriginal)
    if drift:
        with pytest.raises(ValueError, match="drift exceeds"): experiment.stream_probe(c, original, extra)
    else:
        result = experiment.stream_probe(c, original, extra)
        assert result["exact_top10_candidates_refusals"] and result["permutation_and_removal"]
        assert result["batch_sizes"] == [1, 8, 16, 32]
        assert result["max_vector_error"] == result["inference_bn_updates"] == 0


def test_full_runner_one_training_final_checkpoint_only_and_repeat_is_read_only(context, monkeypatch):
    c = context; install_reference(c)
    calls = []
    original = experiment.night.train_until
    def train(c, trial, stop):
        assert not (c["output"] / "frozen_candidate.json").exists()
        calls.append((trial["id"], stop))
        return original(c, trial, stop)
    monkeypatch.setattr(experiment.night, "train_until", train)
    monkeypatch.setattr(experiment.vision, "image_paths", lambda *a: {})
    monkeypatch.setattr(experiment.vision, "encode", lambda model, rows, *a, **kw: random_features(rows, model.dimension))
    monkeypatch.setattr(experiment, "stream_probe", lambda *a: {"status": "passed"})
    def forbidden(*a, **kw): raise AssertionError("No inner eval, extra trials or threshold search in full refit")
    monkeypatch.setattr(experiment.night, "evaluate", forbidden)
    monkeypatch.setattr(experiment.base.policy, "calibrate_policy", forbidden)
    result = experiment.run(c, allow_outer=True)
    assert calls == [(experiment.TRIAL["id"], 6)]
    assert result["optimizer_updates"] == 6 and result["disposable_smoke_updates"] == 1
    assert result["candidate_unchanged"] and not result["promoted"] and not result["inner_evaluation"]
    assert set(result["evaluations"]["validation"]) == set(experiment.CASES)
    assert experiment.run(c, allow_outer=True) == result and len(calls) == 1
    for split in ("calibration", "validation"):
        prefix = c["output"] / "tasks"
        a = prefix / f"{split}_V25_control/export/candidates.csv"
        b = prefix / f"{split}_Full_T12_w10/export/candidates.csv"
        assert a.read_bytes() == b.read_bytes()
        assert np.load(prefix / f"{split}_Full_T12_w10/export/embeddings.npy").shape == (len(c["rows"]), 2432)
    (c["output"] / "tasks/features_validation_Full_T12/features.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Protected"): experiment.run(c, allow_outer=True)


def test_incomplete_evaluation_resumes_without_extra_optimizer_updates(context, monkeypatch):
    c = context; install_reference(c)
    monkeypatch.setattr(experiment.vision, "image_paths", lambda *a: {})
    monkeypatch.setattr(experiment.vision, "encode", lambda model, rows, *a, **kw: random_features(rows, model.dimension))
    monkeypatch.setattr(experiment, "stream_probe", lambda *a: {"status": "passed"})
    evaluate = experiment.evaluate
    def stop(c, split, *a):
        if split == "validation": raise InterruptedError("synthetic stop")
        return evaluate(c, split, *a)
    monkeypatch.setattr(experiment, "evaluate", stop)
    with pytest.raises(InterruptedError): experiment.run(c, allow_outer=True)
    cp = c["output"] / "training" / c["trials"][0]["id"] / "step_00006.pt"
    before = experiment.base.sha256(cp)
    monkeypatch.setattr(experiment, "evaluate", evaluate)
    result = experiment.run(c, allow_outer=True)
    assert result["status"] == "complete" and experiment.base.sha256(cp) == before
    assert len(experiment.base.old.load_json(cp.parent / "history.json")) == 6


def test_valid_run_all_notebook():
    path = experiment.VARIANT / "train_transreid_full_train.ipynb"
    notebook = nbformat.read(path, as_version=4); nbformat.validate(notebook)
    cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in cells: compile(source, str(path), "exec")
    code = "\n".join(cells)
    assert code.index("ORT_DISABLE_TELEMETRY") < code.index("from training import")
    assert "experiment.prepare(" in code and "experiment.run(context, allow_outer=True)" in code
    assert "timeout=" not in code and "DEVICE = 'mps'" in code
