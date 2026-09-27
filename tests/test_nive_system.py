"""v34 fixed concrete checkpoint selection, image runtime and lossless v25 refusals."""
import copy
import csv
import socket
import subprocess
import sys

import nbformat
import numpy as np
import pytest

from training import nive_system as experiment, nive_system_inference as runtime
from test_dual_role_inference import profile, dataset, tiny_model, rows


def vectors(count=61):
    rng = np.random.default_rng(15)
    return np.concatenate([runtime.normalize(rng.normal(size=(count, n)).astype(np.float32))
                           for n in (512, 1536, 512, 512)], axis=1)


@pytest.fixture
def bundle(profile):
    encoders = {}
    for seed, name in enumerate(("parent", "N0", "N1"), 10):
        path = tiny_model(profile.parent / f"{name}.onnx", 256, seed)
        encoders[name] = {"path": path.name, "sha256": runtime.sha256(path)}
    path = profile.parent / "bundle.json"
    runtime.write_json(path, {"schema": "nive-system-v34", "encoders": encoders,
                             "systems": runtime.SYSTEMS, "layout": runtime.LAYOUT, "promoted": False,
                             "preprocessing": runtime.frozen._preprocessing(256, "square"),
                             "v25_profile": {"path": profile.name, "sha256": runtime.sha256(profile)},
                             "threshold": .5})
    return path


def test_exact_fixed_checkpoints_and_no_training_or_promotion():
    settings = runtime.dual.read(experiment.VARIANT / "configs/system_v1.json")
    assert {k: v["step"] for k, v in settings["models"].items()} == {"parent": 0, "N0": 1600, "N1": 1800}
    source = experiment.base.ROOT / settings["source_run"]
    experiment.base.verify_files({str(source / "manifest.json"): settings["source_manifest_sha256"],
                                  str(source / "results.json"): settings["source_results_sha256"]})
    result = runtime.dual.read(source / "results.json")
    for name, spec in settings["models"].items():
        checkpoint = result["training"][spec["arm"]]["checkpoints"][str(spec["step"])]
        assert checkpoint["sha256"] == spec["sha256"]
        experiment.base.verify_files({checkpoint["path"]: spec["sha256"]})
    assert settings["optimizer_updates"] == settings["bn_updates"] == 0
    assert settings["promoted"] is settings["threshold_fit"] is False
    assert settings["wall_time_limit"] is None
    with pytest.raises(ValueError, match="allow_outer"):
        experiment.run({})


@pytest.mark.parametrize("system", runtime.SYSTEMS)
def test_fixed_ranking_formula_and_unchanged_candidates(system):
    values = vectors()
    actual = runtime.rank(values[:, :2048] if system == "V25_control" else values, 33, system)
    baseline = runtime.map_inference.rank(values[:33, :2048], values[33:, :2048])
    np.testing.assert_array_equal(actual["raw_order"], baseline["raw_order"])
    np.testing.assert_array_equal(actual["confidence"], baseline["confidence"])
    if system == "V25_control":
        expected = baseline
    else:
        blocks = [values[:, 2048:2560], values[:, 2560:]]
        if system.startswith("MVP_"): blocks.insert(0, values[:, :512])
        mixed = runtime.normalize(np.concatenate([runtime.normalize(x)*np.float32(np.sqrt(w))
            for x, w in zip(blocks, runtime.SYSTEMS[system]["weights"])], axis=1))
        expected = runtime.dual.policy.rank_vectors(mixed[:33], mixed[33:], runtime.SYSTEMS[system]["graph"])
    np.testing.assert_array_equal(actual["order"], expected["order"])


@pytest.mark.parametrize("system", runtime.SYSTEMS)
def test_no_query_interaction_for_permutation_removal_and_batches(system):
    values = vectors()
    if system == "V25_control": values = values[:, :2048]
    q, g = values[:33], values[33:]
    baseline = runtime.rank(values, 33, system)
    cases = [(np.arange(33)[::-1], 33), (np.array([7]), 1)]
    cases += [(np.arange(33), batch) for batch in (1, 8, 16, 32)]
    for indices, batch in cases:
        selected = q[indices]
        results = [runtime.rank(np.concatenate([selected[i:i+batch], g]), len(selected[i:i+batch]), system)
                   for i in range(0, len(selected), batch)]
        for key in ("order", "raw_order"):
            np.testing.assert_array_equal(np.concatenate([r[key] for r in results]), baseline[key][indices])
        confidence = np.concatenate([r["confidence"] for r in results])
        np.testing.assert_allclose(confidence, baseline["confidence"][indices], rtol=0, atol=2e-7)
        np.testing.assert_array_equal(confidence >= .05, baseline["confidence"][indices] >= .05)


@pytest.mark.parametrize("kind", ["dtype", "zero", "nan", "width", "system"])
def test_bad_feature_banks_are_not_replaced(kind):
    values = vectors()
    if kind == "dtype": values = values.astype(np.float64)
    if kind == "zero": values[:, 2560:] = 0
    if kind == "nan": values[0, 2560] = np.nan
    if kind == "width": values = values[:, :-1]
    with pytest.raises(ValueError):
        runtime.rank(values, 33, "unknown" if kind == "system" else "MVP_R1_N1")


def test_offline_unlabeled_image_export_and_metadata_independence(bundle, dataset, tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Network is forbidden")
    monkeypatch.setattr(socket, "socket", deny)
    assert not (dataset / "train.csv").exists()
    output = tmp_path / "export"
    report = runtime.export(bundle, dataset, output, "MVP_R1_N1")
    assert report["embedding_shape"] == [15, 3072]
    assert report["submission_rows"] == 3
    encoder = runtime.SystemEncoder(bundle, "MVP_R1_N1")
    q = runtime.read_rows(dataset / "test_query.csv")
    original = encoder.encode_rows(q, dataset, 1)
    changed = encoder.encode_rows([{**r, "vehicle_id": 777, "camera_id": 888, "time": "ignored"} for r in q], dataset, 1)
    np.testing.assert_array_equal(changed, original)
    for size in (8, 16, 32):
        np.testing.assert_allclose(encoder.encode_rows(q, dataset, size), original, rtol=0, atol=2e-5)
    q[0]["w"] += 1
    with pytest.raises(ValueError, match="outside"):
        encoder.encode_rows(q, dataset)
    with pytest.raises(ValueError, match="new export"):
        runtime.export(bundle, dataset, output, "MVP_R1_N1")


def test_no_extra_model_in_control_and_no_n0_in_n1(bundle):
    assert len(runtime.SystemEncoder(bundle, "V25_control").encoders) == 0
    assert len(runtime.SystemEncoder(bundle, "MVP_R1_N1").encoders) == 2


@pytest.mark.parametrize("system", runtime.SYSTEMS)
def test_atomic_exports_preserve_top10_on_refusals_and_real_vectors(tmp_path, system):
    values = vectors(15)
    if system == "V25_control": values = values[:, :2048]
    query, gallery = rows(3, "q"), rows(12, "g")
    output = tmp_path / system
    runtime.export_arrays(output, query, gallery, values, system, 2.)
    assert len(list(csv.DictReader((output / "candidates.csv").open()))) == 0
    submission = list(csv.reader((output / "submission.csv").open()))
    assert len(submission) == 3 and all(len(row) == 11 and len(set(row[1:])) == 10 for row in submission)
    np.testing.assert_array_equal(np.load(output / "embeddings.npy", allow_pickle=False), values)
    with pytest.raises(ValueError, match="new output"):
        runtime.export_arrays(output, query, gallery, values, system, 2.)
    assert not list(tmp_path.glob("nive_pending_*"))


@pytest.mark.parametrize("target", ["parent.onnx", "N0.onnx", "N1.onnx", "profile.json"])
def test_corruption_rejected(bundle, target):
    path = bundle.parent / target
    path.write_bytes(path.read_bytes()+b"corrupt")
    with pytest.raises(ValueError, match="Changed"):
        runtime.load_bundle(bundle)


def test_threshold_weights_and_provider_cannot_silently_change(bundle):
    value = runtime.dual.read(bundle)
    entry = value["encoders"]["N1"]
    with pytest.raises(ValueError, match="no fallback"):
        runtime.ImageEncoder(bundle.parent / entry["path"], entry["sha256"], "CUDAExecutionProvider")
    for field, changed in (("threshold", .6), ("systems", {}), ("promoted", True)):
        other = copy.deepcopy(value); other[field] = changed
        runtime.write_json(bundle, other)
        with pytest.raises(ValueError): runtime.load_bundle(bundle)


def test_task_resume_checks_bytes_and_does_not_repeat_action(tmp_path):
    context = {"output": tmp_path, "signature": "frozen"}
    calls = []
    def action(directory):
        calls.append(1)
        (directory / "evidence.txt").write_text("original")
        return {"ok": True}
    assert experiment.task(context, "probe", action) == experiment.task(context, "probe", action)
    assert calls == [1]
    (tmp_path / "tasks/probe/evidence.txt").write_text("changed")
    with pytest.raises(ValueError): experiment.task(context, "probe", action)


def test_original_splits_and_training_id_isolation():
    outer = {"train": [1, 2], "calibration": [3], "validation": [4]}
    protocols, entries = {}, []
    for split, identity in (("calibration", 3), ("validation", 4)):
        images = [{**r, "vehicle_id": identity} for r in rows(11, split)]
        entries += images
        protocols[split] = {"query_ids": [images[0]["image_id"]], "gallery_ids": [r["image_id"] for r in images[1:]]}
    experiment.validate_splits(entries, protocols, outer, [1])
    with pytest.raises(ValueError, match="leakage"):
        experiment.validate_splits(entries, protocols, outer, [3])
    broken = copy.deepcopy(protocols)
    broken["validation"]["query_ids"] = protocols["calibration"]["query_ids"]
    with pytest.raises(ValueError, match="split"):
        experiment.validate_splits(entries, broken, outer, [1])


def test_evaluate_control_and_new_system_keep_exact_candidate_bytes(tmp_path):
    values = vectors(15)
    q, g = rows(3, "q"), rows(12, "g")
    for i, row in enumerate(q+g): row.update(vehicle_id=i%3, camera_id=0 if i < 3 else 1)
    original = tmp_path / "v24/tasks/cached_validation/export"
    baseline = runtime.rank(values[:, :2048], 3, "V25_control")
    # Keep the original v24/v25 candidate bytes and v25 ranking as separate references.
    runtime.export_arrays(original, q, g, values[:, :2048], "V25_control", -.5)
    v25 = tmp_path / "v25"
    source = v25 / "tasks/validation_r1w50_k20_q3_l50"
    source.mkdir(parents=True)
    runtime.write_json(source / "result.json", {"threshold": -.5, **runtime.dual.policy.evaluate(q, g, baseline, -.5, "raw_top1")})
    runtime.export_arrays(source / "export", q, g, values[:, :2048], "V25_control", -.5)
    context = {"rows": q+g, "manifest": {"threshold": -.5, "v24_directory": str(tmp_path / "v24"),
        "v25_directory": str(v25), "protocols": {"validation": {"query_ids": [r["image_id"] for r in q],
                                                                 "gallery_ids": [r["image_id"] for r in g]}}}}
    for name in runtime.SYSTEMS:
        bank = values[:, :2048] if name == "V25_control" else values
        report = experiment.evaluate(context, "validation", name, bank, tmp_path / name)
        assert report["threshold"] == -.5
        assert (tmp_path / name / "export/candidates.csv").read_bytes() == (original / "candidates.csv").read_bytes()


def test_inference_imports_no_training_stack():
    subprocess.run([sys.executable, "-c", "import sys; import training.nive_system_inference; "
                    "assert 'torch' not in sys.modules; assert 'training.nive_mixed' not in sys.modules; "
                    "assert 'training.nive_system' not in sys.modules"], check=True)


def test_inner_guard_requires_exact_deployed_decisions_and_reports_unused_raw_drift():
    before = {"mean_map": .8, "draws": {"one": {"per_query": {"q": {
        "ranking": {"top10": ["a", "b"]}, "raw": {"top10": ["b", "a"], "ap": .5}}}}}}
    after = copy.deepcopy(before)
    assert experiment.compare_inner_decisions(before, after) == []
    after["draws"]["one"]["per_query"]["q"]["raw"]["top10"].reverse()
    changes = experiment.compare_inner_decisions(before, after)
    assert len(changes) == 1 and changes[0]["query_id"] == "q"
    assert changes[0]["before"] != changes[0]["after"]
    after["draws"]["one"]["per_query"]["q"]["ranking"]["top10"].reverse()
    with pytest.raises(ValueError, match="deployed inner top10"):
        experiment.compare_inner_decisions(before, after)


def test_full_runner_reports_all_fixed_cases_and_resumes_without_refitting(tmp_path, monkeypatch):
    context = {"output": tmp_path, "signature": "synthetic"}
    calls = []
    monkeypatch.setattr(experiment, "check_inputs", lambda c: None)
    monkeypatch.setattr(experiment, "export_models", lambda c: {})
    monkeypatch.setattr(experiment, "inner_parity", lambda c, b: {"raw_diagnostic_changes": {"N0": [], "N1": []}})
    monkeypatch.setattr(experiment, "runtime_probe", lambda c, b: {"status": "passed"})
    monkeypatch.setattr(experiment, "protocol_rows", lambda c, split: ([{}], [{}]))
    monkeypatch.setattr(experiment, "baseline_features", lambda *args: (vectors(2)[:, :2048], tmp_path))
    monkeypatch.setattr(experiment, "features", lambda *args: vectors(2)[:, :512])
    def evaluate(c, split, system, values, directory):
        calls.append((split, system))
        return {"ranking": {"mAP@10": .8, "Rank-1": .9}, "candidates": {"F1": .9, "TNR": .8}}
    monkeypatch.setattr(experiment, "evaluate", evaluate)
    def deny(*args, **kwargs):
        raise AssertionError("No training or threshold search in v34")
    monkeypatch.setattr(experiment.base, "train_arm", deny)
    monkeypatch.setattr(runtime.dual.policy, "calibrate_policy", deny)
    result = experiment.run(context, allow_outer=True)
    assert len(calls) == 10 and result["promoted"] is False
    assert result["optimizer_updates"] == result["bn_updates"] == 0
    assert result["threshold_fit"] is False and result["status"] == "complete"
    assert experiment.run(context, allow_outer=True) == result and len(calls) == 10
    report = (tmp_path / "REPORT.md").read_text()
    assert all(name in report for name in runtime.SYSTEMS)


def test_notebook_has_explicit_run_all():
    path = experiment.VARIANT / "evaluate_nive_system.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    for cell in cells: compile(cell.source, str(path), "exec")
    code = "\n".join(c.source for c in cells)
    assert code.index("ORT_DISABLE_TELEMETRY") < code.index("from training import")
    assert "allow_outer=True" in code and "experiment.prepare(" in code
    assert "train_arm(" not in code and "calibrate_policy(" not in code
