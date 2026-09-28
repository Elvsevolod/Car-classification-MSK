"""v37 contracts: calibration selection, frozen refusals, true arrays, resumable Run All."""
import socket
import subprocess
import sys

import nbformat
import numpy as np
import pytest
import torch

from training import transreid_system as experiment, transreid_system_inference as runtime


def vectors(count=19, member="T12"):
    rng = np.random.default_rng(18)
    blocks = [runtime.normalize(rng.normal(size=(count, d)).astype(np.float32))
              for d in (512, 1536, runtime.DIMENSIONS[member])]
    return np.concatenate(blocks, axis=1)


def rows(count, prefix):
    return [{"image_id": f"{prefix}{i}", "vehicle_id": i % 3, "camera_id": int(prefix == "g")} for i in range(count)]


def reports(winner="T12_w10"):
    return {name: {"split": "calibration", "system": name,
        "ranking": {"mAP@10": .82 if name == winner else .8, "Rank-1": .9},
        "candidates": {"F1": .79, "TNR": .71}, "per_query": {}}
        for name in runtime.SYSTEMS}


def context(tmp_path):
    return {"output": tmp_path, "signature": "test", "device": torch.device("cpu"),
        "manifest": {"threshold": .534365177154541, "models": {}, "settings": {"batch_size": 16}}}


def freeze(c, winner="T12_w10"):
    result = reports(winner)
    for name, r in result.items():
        experiment.task(c, f"calibration_{name}", lambda d, value=r: value)
    return experiment.freeze_selection(c, result)


def test_exact_fixed_grid_and_cosine_mixture():
    assert len(runtime.SYSTEMS) == 9
    assert list(runtime.SYSTEMS)[:3] == ["V25_control", "T12_w05", "T22_w05"]
    bank = vectors()
    base = runtime.ranking_features(bank[:, :2048], "V25_control")
    extra = bank[:, 2048:]
    for name, spec in runtime.SYSTEMS.items():
        if spec["member"] != "T12": continue
        mixed = runtime.ranking_features(bank, name)
        w = spec["weight"]
        np.testing.assert_allclose(mixed @ mixed.T, (1-w)*(base @ base.T)+w*(extra @ extra.T), atol=7e-7, rtol=0)


@pytest.mark.parametrize("member", ["T12", "T22"])
def test_control_exact_and_candidate_unchanged_at_all_weights(member):
    bank = vectors(member=member)
    expected = runtime.map_inference.rank(bank[:3, :2048], bank[3:, :2048])
    control = runtime.rank(bank[:, :2048], 3, "V25_control")
    for key in expected: np.testing.assert_array_equal(control[key], expected[key])
    for name, spec in runtime.SYSTEMS.items():
        if spec["member"] != member: continue
        ranked = runtime.rank(bank, 3, name)
        np.testing.assert_array_equal(ranked["raw_order"], expected["raw_order"])
        np.testing.assert_array_equal(ranked["confidence"], expected["confidence"])


@pytest.mark.parametrize("name", list(runtime.SYSTEMS))
def test_static_gallery_query_order_and_neighbors_do_not_change_decisions(name):
    member = runtime.SYSTEMS[name]["member"]
    bank = vectors(member=member or "T12")
    if member is None: bank = bank[:, :2048]
    full = runtime.rank(bank, 3, name)
    for indices in ([2, 1, 0], [1], [0, 2]):
        partial = runtime.rank(np.concatenate([bank[indices], bank[3:]]), len(indices), name)
        np.testing.assert_array_equal(partial["order"][:, :10], full["order"][indices, :10])
        np.testing.assert_array_equal(partial["raw_order"][:, 0], full["raw_order"][indices, 0])
        np.testing.assert_allclose(partial["confidence"], full["confidence"][indices], atol=2e-5, rtol=0)
        np.testing.assert_array_equal(partial["confidence"] >= .03, full["confidence"][indices] >= .03)


@pytest.mark.parametrize("member", ["T12", "T22"])
def test_export_has_real_vectors_ten_unique_ids_and_identical_refusals(tmp_path, member):
    q, g = rows(3, "q"), rows(16, "g")
    bank = vectors(member=member)
    control = tmp_path / "control"
    runtime.export_arrays(control, q, g, bank[:, :2048], "V25_control", .03)
    for name in (f"{member}_w05", f"{member}_w20"):
        output = tmp_path / name
        runtime.export_arrays(output, q, g, bank, name, .03)
        np.testing.assert_array_equal(np.load(output / "embeddings.npy"), bank)
        assert (output / "candidates.csv").read_bytes() == (control / "candidates.csv").read_bytes()
        parsed = experiment.base.official.load_submission(output / "submission.csv", {r["image_id"] for r in g})
        assert len(parsed) == 3 and all(len(set(ids)) == 10 for ids in parsed.values())
    # Refusal still emits top-10 and a header-only candidate file.
    output = tmp_path / "reject"
    runtime.export_arrays(output, q, g, bank, f"{member}_w10", 2.)
    assert len((output / "candidates.csv").read_text().splitlines()) == 1
    assert len((output / "submission.csv").read_text().splitlines()) == 3


def test_invalid_banks_rejected():
    bank = vectors()
    with pytest.raises(ValueError): runtime.rank(bank, 3, "V25_control")
    with pytest.raises(ValueError): runtime.rank(bank, 3, "T22_w05")
    with pytest.raises(ValueError): runtime.rank(bank.astype(np.float64), 3, "T12_w05")
    bank[0, -1] = np.nan
    with pytest.raises(ValueError): runtime.rank(bank, 3, "T12_w05")


def test_selection_never_uses_validation_and_control_wins_exact_ties():
    r = reports()
    assert experiment.select_calibration(r) == "T12_w10"
    equal = reports(winner=None)
    assert experiment.select_calibration(equal) == "V25_control"
    equal["T22_w05"]["ranking"]["mAP@10"] = .9
    equal["T12_w10"]["ranking"]["mAP@10"] = .9
    assert experiment.select_calibration(equal) == "T22_w05"
    r["T12_w10"]["split"] = "validation"
    with pytest.raises(ValueError, match="calibration-only"): experiment.select_calibration(r)
    with pytest.raises(ValueError, match="nine"): experiment.select_calibration({})


def test_validation_guard_rejects_unfrozen_unselected_and_changed_calibration(tmp_path):
    c = context(tmp_path)
    with pytest.raises(ValueError, match="closed"): experiment.require_selection(c)
    chosen = freeze(c)
    assert experiment.require_selection(c, member="T12") == chosen
    with pytest.raises(ValueError, match="selected member"): experiment.require_selection(c, member="T22")
    with pytest.raises(ValueError, match="Only control"): experiment.require_selection(c, system="T12_w20")
    # Direct evaluation/extraction cannot bypass the orchestration order.
    with pytest.raises(ValueError, match="selected member"): experiment.features(c, [], "validation", "T22")
    with pytest.raises(ValueError, match="Only control"): experiment.evaluate(c, "validation", "T12_w20", None, tmp_path)
    (tmp_path / "tasks/calibration_T12_w10/result.json").write_text("{}")
    with pytest.raises(ValueError, match="Protected"): experiment.require_selection(c)


def test_atomic_stage_recovers_after_interruption_and_rejects_corruption(tmp_path):
    c = context(tmp_path)
    def interrupted(d):
        (d / "partial.txt").write_text("not committed")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError): experiment.task(c, "probe", interrupted)
    assert not (tmp_path / "tasks/probe").exists()
    calls = []
    def action(d):
        calls.append(1)
        (d / "evidence.txt").write_text("verified")
        return {"ok": True}
    assert experiment.task(c, "probe", action) == experiment.task(c, "probe", action)
    assert calls == [1]
    (tmp_path / "tasks/probe/evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="Protected"): experiment.task(c, "probe", action)


def test_evaluation_and_export_reproduce_historical_control_and_candidate_bytes(tmp_path):
    q, g = rows(3, "q"), rows(16, "g")
    bank = vectors()
    c = context(tmp_path / "new")
    c.update(rows=q+g)
    c["manifest"].update(threshold=.03, v24_directory=str(tmp_path / "v24"), v25_directory=str(tmp_path / "v25"),
        protocols={split: {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g]}
                   for split in ("calibration", "validation")})
    for split in ("calibration", "validation"):
        reference = tmp_path / "v25/tasks" / f"{split}_r1w50_k20_q3_l50"
        metrics = runtime.export_arrays(reference / "export", q, g, bank[:, :2048], "V25_control", .03)
        experiment.base.write_json(reference / "result.json", {"threshold": .03, **metrics})
        runtime.export_arrays(tmp_path / "v24/tasks" / f"cached_{split}" / "export", q, g,
                              bank[:, :2048], "V25_control", .03)
    freeze(c)
    for split in ("calibration", "validation"):
        for name in ("V25_control", "T12_w10"):
            values = bank[:, :2048] if name == "V25_control" else bank
            report = experiment.evaluate(c, split, name, values, tmp_path)
            result = experiment.export_result(c, split, name, values, report)
            assert result["candidate_bytes_unchanged"] and result["npy_replay"] == "exact decisions"
            assert len(report["per_query"]) == len(q)


@pytest.mark.parametrize("winner", ["T12_w10", "T22_w05", "V25_control"])
def test_full_runner_freezes_before_validation_and_resumes_without_training(tmp_path, monkeypatch, winner):
    c = context(tmp_path)
    calls, extracts = [], []
    monkeypatch.setattr(experiment, "check_inputs", lambda c: None)
    monkeypatch.setattr(experiment, "runtime_probe", lambda c: {"status": "passed"})
    def protocol(c, split):
        if split == "validation": experiment.require_selection(c)
        return rows(3, "q"), rows(16, "g")
    monkeypatch.setattr(experiment, "protocol_rows", protocol)
    monkeypatch.setattr(experiment, "baseline_features", lambda *a: (vectors()[:, :2048], tmp_path))
    def extract(c, rows, split, member):
        if split == "validation": experiment.require_selection(c, member=member)
        extracts.append((split, member))
        return vectors(member=member)[:, 2048:]
    monkeypatch.setattr(experiment, "features", extract)
    def evaluate(c, split, name, bank, directory):
        if split == "validation": experiment.require_selection(c, system=name)
        calls.append((split, name))
        return {**reports(winner)[name], "split": split}
    monkeypatch.setattr(experiment, "evaluate", evaluate)
    monkeypatch.setattr(experiment, "export_result", lambda *a: None)
    def deny(*a, **kw): raise AssertionError("No optimizer, threshold fitting, or network in this experiment")
    monkeypatch.setattr(torch.optim, "AdamW", deny)
    monkeypatch.setattr(runtime.dual.policy, "calibrate_policy", deny)
    monkeypatch.setattr(socket, "socket", deny)
    result = experiment.run(c, allow_outer=True)
    assert result["selection"]["selected"] == winner
    assert len(calls) == 9+(1 if winner == "V25_control" else 2)
    assert [n for split, n in calls if split == "validation"] == list(dict.fromkeys(["V25_control", winner]))
    assert [m for split, m in extracts if split == "validation"] == ([] if winner == "V25_control" else [runtime.SYSTEMS[winner]["member"]])
    assert result["optimizer_updates"] == result["bn_updates"] == 0
    assert result["candidate_unchanged"] and not result["threshold_fit"] and not result["promoted"]
    assert experiment.run(c, allow_outer=True) == result
    assert len(calls) == 9+(1 if winner == "V25_control" else 2)


def test_strict_saved_model_restore_no_download_or_training(tmp_path, monkeypatch):
    from tests.test_transreid_night import TinyModel
    model = TinyModel(4, "global")
    path = tmp_path / "checkpoint.pt"
    trial = {"id": "T12_global_supcon", "architecture": "global"}
    torch.save({"model": model.state_dict(), "signature": "v36", "trial": trial, "step": 1800}, path)
    c = context(tmp_path)
    c["manifest"].update(source_signature="v36", train_ids=[0, 1, 2, 3], drop_path=.1,
        models={"T12": {"path": str(path), "sha256": experiment.base.sha256(path), "trial": trial, "step": 1800, "architecture": "global"}})
    def factory(classes, architecture, **kw):
        assert kw["pretrained"] is False
        return TinyModel(classes, architecture)
    monkeypatch.setattr(experiment.vision, "ReIDModel", factory)
    loaded = experiment.load_model(c, "T12")
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Protected"): experiment.load_model(c, "T12")


def test_inference_module_does_not_import_training_or_torch():
    subprocess.run([sys.executable, "-c", "import sys; import training.transreid_system_inference; "
        "assert 'torch' not in sys.modules; assert 'training.transreid_system' not in sys.modules"], check=True)


def test_notebook_valid_run_all_without_new_training():
    path = experiment.VARIANT / "compare_transreid_system.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    for cell in cells: compile(cell.source, str(path), "exec")
    code = "\n".join(c.source for c in cells)
    assert code.index("ORT_DISABLE_TELEMETRY") < code.index("from training import")
    assert "experiment.prepare(" in code and "allow_outer=True" in code
    assert "download_weights(" not in code and "night.run(" not in code
