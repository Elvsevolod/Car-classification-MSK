"""v39: cache-only reranking, one calibration winner, exact frozen candidates."""
import copy
import json
import socket
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest

from training import transreid_graph as experiment, transreid_graph_inference as runtime
from tests.test_transreid_system import vectors, rows


def test_fixed_grid_and_weights():
    assert list(runtime.SYSTEMS) == ["V25_control", "Full_T12_l50", "Full_T12_l75", "Full_T12_raw"]
    assert [s["weight"] for s in runtime.SYSTEMS.values()] == [0., .1, .1, .1]
    assert [s["policy"] for s in runtime.SYSTEMS.values()] == ["legacy", "legacy", "less_graph", "raw"]
    assert runtime.policy.POLICIES["legacy"] == {"k1": 20, "k2": 3, "lambda": .5}
    assert runtime.policy.POLICIES["less_graph"] == {"k1": 20, "k2": 3, "lambda": .75}
    plan = experiment.base.old.load_json(experiment.VARIANT / "configs/graph_v1.json")
    assert not any(plan[k] for k in ("optimizer_updates", "encoder_forwards", "threshold_fit", "promoted"))
    assert plan["wall_time_limit"] is None and plan["new_encoder_weight"] == .1
    assert plan["selection_split"] == "calibration" and plan["selection_metric"] == "mAP@10"


@pytest.mark.parametrize("system", list(runtime.SYSTEMS))
def test_matches_declared_policy_and_exact_v25_candidate(system):
    bank = vectors()
    if system == "V25_control": bank = bank[:, :2048]
    actual = runtime.rank(bank, 3, system)
    spec = runtime.SYSTEMS[system]
    mixed = runtime.previous.ranking_features(bank, spec["mixture"])
    reference = runtime.policy.rank_vectors(mixed[:3], mixed[3:], spec["policy"])
    control = runtime.previous.rank(bank[:, :2048], 3, "V25_control")
    np.testing.assert_array_equal(actual["order"], reference["order"])
    for key in ("raw_order", "confidence"):
        np.testing.assert_array_equal(actual[key], control[key])
    if spec["policy"] == "legacy":
        old = runtime.previous.rank(bank, 3, spec["mixture"])
        for key in old: np.testing.assert_array_equal(actual[key], old[key])


@pytest.mark.parametrize("system", list(runtime.SYSTEMS))
def test_query_batch_permutation_and_removal_independence(system):
    bank = vectors(49)
    if system == "V25_control": bank = bank[:, :2048]
    full = runtime.rank(bank, 33, system)
    for size in (1, 8, 16, 32):
        for start in range(0, 33, size):
            idx = np.arange(start, min(start+size, 33))
            part = runtime.rank(np.concatenate([bank[idx], bank[33:]]), len(idx), system)
            for key in ("order", "raw_order"):
                np.testing.assert_array_equal(part[key], full[key][idx])
            np.testing.assert_allclose(part["confidence"], full["confidence"][idx], atol=2e-5, rtol=0)
            np.testing.assert_array_equal(part["confidence"] >= .03, full["confidence"][idx] >= .03)
    for idx in ([32, 0, 5], list(range(32, -1, -1)), [12]):
        part = runtime.rank(np.concatenate([bank[idx], bank[33:]]), len(idx), system)
        for key in ("order", "raw_order"):
            np.testing.assert_array_equal(part[key], full[key][idx])
        np.testing.assert_array_equal(part["confidence"] >= .03, full["confidence"][idx] >= .03)


@pytest.mark.parametrize("bad", ["dimension", "float64", "nan", "zero", "queries", "gallery", "system"])
def test_invalid_vectors_fail_instead_of_repair(bad):
    bank = vectors(); nq, name = 3, "Full_T12_l75"
    if bad == "dimension": bank = bank[:, :-1]
    if bad == "float64": bank = bank.astype(np.float64)
    if bad == "nan": bank[0, -1] = np.nan
    if bad == "zero": bank[0, 2048:] = 0
    if bad == "queries": nq = 0
    if bad == "gallery": nq = len(bank)-9
    if bad == "system": name = "new_hypothesis"
    with pytest.raises(ValueError): runtime.rank(bank, nq, name)


@pytest.mark.parametrize("threshold", [.03, 2.])
def test_export_replay_and_refusal_keeps_ten_ids(tmp_path, threshold):
    q, g = rows(3, "q"), rows(16, "g"); bank = vectors()
    reference = tmp_path / "old"
    runtime.previous.export_arrays(reference, q, g, bank[:, :2048], "V25_control", threshold)
    for name in runtime.SYSTEMS:
        values = bank[:, :2048] if name == "V25_control" else bank
        output = tmp_path / name
        runtime.export_arrays(output, q, g, values, name, threshold)
        assert (output / "candidates.csv").read_bytes() == (reference / "candidates.csv").read_bytes()
        assert len((output / "submission.csv").read_text().splitlines()) == 3
        for row in (output / "submission.csv").read_text().splitlines():
            fields = row.split(",")
            assert len(fields) == 11 and len(set(fields[1:])) == 10
        restored = np.load(output / "embeddings.npy", allow_pickle=False)
        assert restored.dtype == np.float32
        np.testing.assert_array_equal(restored, values)
        order = experiment.base.old.load_json(output / "embedding_order.json")
        assert order["ids"] == [r["image_id"] for r in q+g] and order["spec"] == runtime.SYSTEMS[name]
        if threshold == 2.:
            assert len((output / "candidates.csv").read_text().splitlines()) == 1


def test_readme_transition_is_narrow_and_never_mutates_old_manifest(tmp_path, monkeypatch):
    readme, weights, release = tmp_path / "README.md", tmp_path / "model.onnx", tmp_path / "release_decision.json"
    readme.write_text("reviewed documentation"); weights.write_bytes(b"unchanged weights")
    release.write_text("same model, reviewed open-check list")
    monkeypatch.setattr(experiment, "README_PATH", readme)
    monkeypatch.setattr(experiment, "README_NEW", experiment.base.sha256(readme))
    monkeypatch.setattr(experiment, "RELEASE_PATH", release)
    monkeypatch.setattr(experiment, "RELEASE_NEW", experiment.base.sha256(release))
    old = {str(readme): experiment.README_OLD, str(release): experiment.RELEASE_OLD,
           str(weights): experiment.base.sha256(weights)}
    snapshot = dict(old)
    updated, transition = experiment.reviewed_protection(old)
    assert old == snapshot and updated != old and transition["old_manifests_modified"] is False
    assert set(updated) == set(old) and updated[str(weights)] == old[str(weights)]
    assert len(transition["files"]) == 2
    with pytest.raises(ValueError, match="original v38 protection"):
        experiment.reviewed_protection({**old, str(readme): "unexpected old hash"})
    weights.write_bytes(b"different weight")
    with pytest.raises(ValueError, match="Protected file"):
        experiment.reviewed_protection(old)
    weights.write_bytes(b"unchanged weights"); release.write_text("changed active model")
    with pytest.raises(ValueError, match="Protected file"):
        experiment.reviewed_protection(old)
    release.write_text("same model, reviewed open-check list"); readme.write_text("one more edit")
    with pytest.raises(ValueError, match="Protected file"):
        experiment.reviewed_protection(old)


@pytest.fixture
def context(tmp_path, monkeypatch):
    q, g = rows(3, "q"), rows(16, "g"); bank = vectors()
    q[-1]["vehicle_id"] = 99  # Include an open-set query, as in both original protocols.
    source = tmp_path / "v38"
    protocols = {split: {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g]}
                 for split in ("calibration", "validation")}
    c = {"output": tmp_path / "v39", "signature": "synthetic", "rows": q+g,
        "manifest": {"source_directory": str(source), "protocols": protocols, "threshold": .03,
            "checkpoint": {"path": "full_t12.pt", "sha256": "synthetic"}, "source_signature": "v38",
            "v24_directory": str(tmp_path / "v24"), "protected": {}, "scope": "synthetic",
            "reviewed_document_transition": {"old_manifests_modified": False}}}
    for split in protocols:
        for name, old_name in (("V25_control", "V25_control"), ("Full_T12_w10", "T12_w10")):
            directory = source / "tasks" / f"{split}_{name}"
            values = bank[:, :2048] if name == "V25_control" else bank
            ranked = runtime.previous.rank(values, 3, old_name)
            report = {"split": split, "system": name, "threshold": .03,
                **runtime.policy.evaluate(q, g, ranked, .03, "raw_top1"),
                **runtime.policy.query_diagnostics(q, g, ranked)}
            ordered, _ = runtime.policy.predictions(q, g, ranked, .03, "raw_top1")
            for qid, top10 in ordered.items(): report["per_query"][qid]["ranking"]["top10"] = top10
            mixed = runtime.previous.ranking_features(values, old_name)
            raw = runtime.policy.rank_vectors(mixed[:3], mixed[3:], "raw")
            report["raw_ranking"] = runtime.policy.evaluate(q, g, raw, .03, "raw_top1")["ranking"]
            experiment.base.write_json(directory / "result.json", report)
            runtime.previous.export_arrays(directory / "export", q, g, values, old_name, .03)
    c["manifest"]["protected"] = {str(p): experiment.base.sha256(p) for p in source.rglob("*") if p.is_file()}
    # Production preflight does the real frozen source audit; this fixture is synthetic.
    monkeypatch.setattr(experiment, "check_inputs", lambda c: None)
    return c


def install_calibration(c, winner):
    for name in runtime.SYSTEMS:
        report = {"split": "calibration", "system": name, "spec": runtime.SYSTEMS[name],
            "threshold": c["manifest"]["threshold"], "candidate_unchanged": True,
            "protocol_sha256": experiment.base.digest(c["manifest"]["protocols"]["calibration"]),
            "ranking": {"mAP@10": .85 if name == winner else .8}}
        experiment.task(c, f"calibration_{name}", lambda d, r=report: r)
    return experiment.freeze_selection(c)


@pytest.mark.parametrize("winner", [None, *runtime.SYSTEMS])
def test_calibration_only_selection_ties_and_validation_gate(context, winner, monkeypatch):
    c = context
    with pytest.raises(ValueError, match="four completed"):
        experiment.freeze_selection(c)
    original = np.load
    def forbidden(*a, **kw): raise AssertionError("Validation array opened before authorization")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="closed"):
        experiment.features(c, "validation", "V25_control")
    with pytest.raises(ValueError, match="closed"):
        experiment.evaluate(c, "validation", "Full_T12_l50", c["output"])
    frozen = install_calibration(c, winner)
    assert frozen["selected"] == (winner or "V25_control")
    assert frozen["evaluations"] == list(dict.fromkeys(["V25_control", winner or "V25_control"]))
    for name in set(runtime.SYSTEMS)-set(frozen["evaluations"]):
        with pytest.raises(ValueError, match="only the frozen"):
            experiment.features(c, "validation", name)
        with pytest.raises(ValueError, match="only the frozen"):
            experiment.evaluate(c, "validation", name, c["output"])
    monkeypatch.setattr(np, "load", original)
    for name in frozen["evaluations"]:
        assert len(experiment.features(c, "validation", name)) == 19
    assert experiment.require_selection(c) == frozen


def test_tampered_calibration_or_selection_rejected(context):
    c = context; install_calibration(c, "Full_T12_l75")
    path = c["output"] / "frozen_selection.json"
    saved = experiment.base.old.load_json(path); changed = copy.deepcopy(saved)
    changed["selected"] = "Full_T12_raw"
    experiment.base.write_json(path, changed)
    with pytest.raises(ValueError, match="selection changed"):
        experiment.require_selection(c)
    experiment.base.write_json(path, saved)
    (c["output"] / "tasks/calibration_Full_T12_l75/result.json").write_text("{}")
    with pytest.raises(ValueError, match="Protected file"):
        experiment.require_selection(c)


@pytest.mark.parametrize("winner", ["Full_T12_l50", "Full_T12_l75", "Full_T12_raw"])
def test_each_frozen_candidate_can_be_evaluated_without_other_validation_cases(context, winner):
    c = context; install_calibration(c, winner)
    control = experiment.task(c, "validation_V25_control",
        lambda d: experiment.evaluate(c, "validation", "V25_control", d))
    report = experiment.task(c, f"validation_{winner}",
        lambda d: experiment.evaluate(c, "validation", winner, d))
    assert report["candidates"] == control["candidates"] and report["candidate_unchanged"]
    actual = {p.name for p in (c["output"] / "tasks").glob("validation_*")}
    assert actual == {"validation_V25_control", f"validation_{winner}"}


@pytest.mark.parametrize("bad", ["split", "spec", "threshold", "protocol", "candidate", "nan"])
def test_invalid_calibration_records_cannot_select(context, bad):
    c = context; install_calibration(c, "Full_T12_l75")
    directory = c["output"] / "tasks/calibration_Full_T12_l75"
    path = directory / "result.json"; value = experiment.base.old.load_json(path)
    if bad == "split": value["split"] = "validation"
    if bad == "spec": value["spec"] = runtime.SYSTEMS["Full_T12_raw"]
    if bad == "threshold": value["threshold"] += .1
    if bad == "protocol": value["protocol_sha256"] = "different"
    if bad == "candidate": value["candidate_unchanged"] = False
    if bad == "nan": value["ranking"]["mAP@10"] = float("nan")
    path.write_text(json.dumps(value), encoding="utf-8")  # Deliberately permit corrupt NaN JSON.
    receipt_path = directory / "complete.json"; receipt = experiment.base.old.load_json(receipt_path)
    receipt["files"]["result.json"] = experiment.base.sha256(path)
    experiment.base.write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="Invalid calibration"):
        experiment.selection(c)


@pytest.mark.parametrize("bad", ["bytes", "order", "dimensions"])
def test_source_cache_corruption_is_never_reextracted(context, bad):
    c = context; directory = experiment.source_stage(c, "calibration", "Full_T12_l75") / "export"
    if bad == "bytes":
        (directory / "embeddings.npy").write_bytes(b"broken")
    else:
        path = directory / "embedding_order.json"; value = experiment.base.old.load_json(path)
        if bad == "order": value["ids"].reverse()
        if bad == "dimensions": value["dimension"] = 512
        experiment.base.write_json(path, value)
        # Even a freshly checksummed wrong cache must fail the semantic layout guard.
        c["manifest"]["protected"][str(path)] = experiment.base.sha256(path)
    with pytest.raises(ValueError): experiment.features(c, "calibration", "Full_T12_l75")


def test_run_and_resume_never_train_extract_fit_or_download(context, monkeypatch):
    from training import transreid_night, transreid_model
    c = context
    def forbidden(*a, **kw): raise AssertionError("Forbidden in a cache-only experiment")
    monkeypatch.setattr(transreid_night, "train_until", forbidden)
    monkeypatch.setattr(transreid_model, "encode", forbidden)
    monkeypatch.setattr(runtime.previous.dual, "DualRoleEncoder", forbidden)
    monkeypatch.setattr(runtime.policy, "calibrate_policy", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    actual = experiment.evaluate; calls = []
    def interrupted(ctx, split, name, directory):
        calls.append((split, name))
        if name == "Full_T12_l75": raise RuntimeError("synthetic interruption")
        return actual(ctx, split, name, directory)
    monkeypatch.setattr(experiment, "evaluate", interrupted)
    with pytest.raises(ValueError, match="Explicit"):
        experiment.run(c)
    with pytest.raises(RuntimeError, match="interruption"):
        experiment.run(c, allow_outer=True)
    assert not (c["output"] / "frozen_selection.json").exists()
    assert (c["output"] / "tasks/calibration_Full_T12_l50/complete.json").is_file()
    calls.clear()
    def recorded(ctx, split, name, directory):
        calls.append((split, name))
        return actual(ctx, split, name, directory)
    monkeypatch.setattr(experiment, "evaluate", recorded)
    result = experiment.run(c, allow_outer=True)
    assert calls[:2] == [("calibration", "Full_T12_l75"), ("calibration", "Full_T12_raw")]
    assert [name for split, name in calls if split == "validation"] == result["selection"]["evaluations"]
    assert result["status"] == "complete" and result["candidate_unchanged"]
    assert not any(result[k] for k in ("optimizer_updates", "encoder_forwards", "inference_bn_updates", "threshold_fit", "promoted"))
    monkeypatch.setattr(experiment, "evaluate", forbidden)
    assert experiment.run(c, allow_outer=True) == result
    assert (c["output"] / "REPORT.md").is_file()


def test_notebook_valid_compilable_and_cache_only():
    notebook = nbformat.read(experiment.VARIANT / "compare_transreid_graph.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    compile(code, "v39-notebook", "exec")
    assert "allow_outer=True" in code and "experiment.prepare(run_name=RUN_NAME)" in code
    assert "DEVICE" not in code and "timeout=" not in code
    assert code.index("ORT_DISABLE_TELEMETRY") < code.index("from training")
    # A completed research notebook is also the user's result artifact; preserve its outputs.
    assert all(output.output_type != "error" for cell in notebook.cells for output in cell.get("outputs", []))


def test_inference_module_does_not_import_training_runtime():
    code = ("import sys; from training import transreid_graph_inference; "
            "assert 'torch' not in sys.modules; "
            "assert 'training.transreid_model' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=experiment.base.ROOT)
