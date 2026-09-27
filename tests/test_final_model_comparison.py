"""v22 contract tests: synthetic protocols; no historical data or weights written."""
from copy import deepcopy
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

from backend.core import normalize, sha256
from training import final_model_comparison as comparison
from training.stage6 import write_json


@pytest.fixture
def context(tmp_path, monkeypatch):
    rows, protocols, outer = [], {}, {"train": [0, 1]}
    for split, offset in (("calibration", 10), ("validation", 30)):
        query = [{"image_id": f"{split}_q{i}", "vehicle_id": offset+i, "camera_id": 0}
                 for i in (0, 1, 2, 3, 99, 100)]
        gallery = [{"image_id": f"{split}_g{i}", "vehicle_id": offset+i//2, "camera_id": 1}
                   for i in range(12)]
        rows.extend(query + gallery)
        protocols[split] = {"query_ids": [r["image_id"] for r in query], "gallery_ids": [r["image_id"] for r in gallery]}
        outer[split] = sorted({r["vehicle_id"] for r in query + gallery})
    components = {n: {"kind": "torch", "dimension": 512} for n in ("a", "b", "c", "d")}
    specs = []
    for name, members, lam, candidate in (
            ("V21_selected_primary", ["d", "b", "c"], .65, "raw_top1"),
            ("R1_equal3_primary", ["a", "b", "c"], .75, "raw_top1"),
            ("R1_equal3_full_v18", ["d", "a", "b"], .75, "raw_top1"),
            ("MVP_recalibrated", ["a"], .5, "ranking_top1")):
        specs.append({**comparison.search.system(name, members), "lambda": lam, "candidate_policy": candidate,
                      "train_identities": 2, "dimension": 512*len(members)})
    ctx = {"output": tmp_path / "run", "signature": "v22", "protected": {}, "rows": rows,
           "device": torch.device("cpu"), "manifest": {"source_sha256": {}, "protocols": protocols,
           "outer": outer, "inner": {"primary": {"train": [0]}}, "comparison_plan": {
               "systems": specs, "components": components, "active_mvp_threshold": .9}}}
    monkeypatch.setattr(comparison, "check_runtime", lambda *_: None)
    monkeypatch.setattr(comparison.old.review, "check_inputs", lambda *a, **kw: None)
    calls = []

    def feature_task(ctx, split, name, spec, directory):
        if split == "validation":
            comparison.read_frozen(ctx)
            assert len(list(ctx["output"].glob("tasks/calibrate_*/complete.json"))) == 4
        calls.append((split, name))
        q, g = comparison.old.review.old.protocol_rows(ctx, split)
        rng = np.random.default_rng(100 + ord(name))
        gv = normalize(rng.normal(size=(len(g), 512)).astype(np.float32))
        qv = normalize(np.stack([gv[0]+.01, gv[2]+.01, gv[4]+.01, gv[6]+.01,
                                *rng.normal(size=(2, 512))]).astype(np.float32))
        path = directory / "features.npz"
        np.savez_compressed(path, ids=[r["image_id"] for r in q+g], vectors=np.concatenate([qv, gv]))
        return {"path": str(path), "sha256": sha256(path)}

    monkeypatch.setattr(comparison, "feature_task", feature_task)
    return ctx, calls


def test_full_run_freezes_all_thresholds_before_validation_and_resumes(context):
    ctx, calls = context
    result = comparison.run(ctx, allow_outer=True)
    assert result["status"] == "complete" and result["protected_unchanged"]
    assert result["optimizer_updates"] == result["bn_updates"] == 0
    assert result["outer_evaluated"] and not result["promoted"]
    assert len(result["evaluations"]) == 5 and len(result["selected_vs"]) == 4
    assert [s for s, _ in calls] == ["calibration"]*4 + ["validation"]*4
    assert result["evaluations"]["MVP_active"]["threshold"] == .9
    for name, report in result["evaluations"].items():
        folder = Path(report["export"])
        rows = ctx["manifest"]["protocols"]["validation"]
        order = comparison.old.read(folder/"embedding_order.json")
        assert order["ids"] == rows["query_ids"] + rows["gallery_ids"]
        embeddings = np.load(folder/"embeddings.npy", allow_pickle=False)
        comparison.validate_vectors(embeddings, 18, report["system"]["dimension"])
        predictions = comparison.old.policy.official.load_submission(folder/"submission.csv", set(rows["gallery_ids"]))
        assert set(predictions) == set(rows["query_ids"])
        assert all(len(v) == len(set(v)) == 10 for v in predictions.values())
        candidates = comparison.old.policy.official.load_candidates(folder/"candidates.csv")
        assert report["accepted"] == len(candidates) and report["accepted"]+report["refused"] == 6
        assert name == "MVP_active" or report["threshold"] == result["frozen"]["thresholds"][name]
    again = comparison.run(ctx, allow_outer=True)
    assert again["evaluations"] == result["evaluations"] and len(calls) == 8
    assert all(e["status"] == "cached" for e in again["events"])


def test_calibration_failure_never_reaches_validation_and_can_resume(context, monkeypatch):
    ctx, calls = context
    original = comparison.calibrate

    def fail(ctx, spec, features, *, split):
        if spec["name"] == "R1_equal3_primary":
            raise RuntimeError("temporary failure")
        return original(ctx, spec, features, split=split)

    monkeypatch.setattr(comparison, "calibrate", fail)
    with pytest.raises(ValueError, match="every calibration"):
        comparison.run(ctx, allow_outer=True)
    assert not (ctx["output"]/"frozen_comparison.json").exists()
    assert {split for split, _ in calls} == {"calibration"}
    monkeypatch.setattr(comparison, "calibrate", original)
    assert comparison.run(ctx, allow_outer=True)["status"] == "complete"
    assert len(calls) == 8


def test_explicit_outer_permission_and_calibration_only(context):
    ctx, calls = context
    with pytest.raises(ValueError, match="allow_outer"):
        comparison.run(ctx)
    with pytest.raises(ValueError, match="calibration-only"):
        comparison.calibrate(ctx, {}, {}, split="validation")
    assert calls == [] and not ctx["output"].exists()


def test_corrupt_features_or_frozen_thresholds_fail_closed(context):
    ctx, _ = context
    comparison.run(ctx, allow_outer=True)
    path = ctx["output"]/"frozen_comparison.json"
    original = comparison.old.read(path)
    changed = deepcopy(original)
    changed["thresholds"]["V21_selected_primary"] += .1
    write_json(path, changed)
    with pytest.raises(comparison.old.IntegrityError, match="calibration receipt"):
        comparison.read_frozen(ctx)
    write_json(path, original)
    artifact = next(ctx["output"].glob("tasks/features_calibration_*/features.npz"))
    artifact.write_bytes(b"corrupt")
    with pytest.raises(comparison.old.IntegrityError, match="Changed task artifact"):
        comparison.run(ctx, allow_outer=True)


def test_original_protocol_order_and_identity_boundaries(context):
    ctx, _ = context
    comparison.validate_protocols(ctx)
    bad = deepcopy(ctx)
    bad["manifest"]["outer"]["validation"].append(0)
    with pytest.raises(comparison.old.IntegrityError, match="identity overlap"):
        comparison.validate_protocols(bad)
    bad = deepcopy(ctx)
    bad["manifest"]["protocols"]["validation"]["gallery_ids"][0] = bad["manifest"]["protocols"]["validation"]["query_ids"][0]
    with pytest.raises(comparison.old.IntegrityError, match="Invalid original"):
        comparison.validate_protocols(bad)


def test_saved_average_is_loaded_exactly_without_bn_updates(tmp_path, monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    with torch.no_grad():
        model[0].weight.fill_(.73)
        model[1].running_mean.fill_(4.)
    path = tmp_path/"derived.pt"
    torch.save({"model": model.state_dict()}, path)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    checksum = sha256(path)
    monkeypatch.setattr(comparison.old, "source_summary", lambda *args: {})
    monkeypatch.setattr(comparison.old.review, "load_model", lambda *args: (
        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)), None))
    loaded, _ = comparison.load_primary({"source": {}}, {"path": str(path), "sha256": checksum,
                                                         "kind": "average", "seed": 15})
    assert not any(m.training for m in loaded.modules())
    assert not any(p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in expected.items())
    assert sha256(path) == checksum
    path.write_bytes(b"damaged")
    with pytest.raises(comparison.old.IntegrityError, match="weights changed"):
        comparison.load_primary({"source": {}}, {"path": str(path), "sha256": checksum})


@pytest.mark.parametrize("dimension", [512, 1536])
def test_no_fabricated_zero_nonfinite_or_wrong_precision_vectors(dimension):
    for vectors in (np.zeros((2, dimension), np.float32), np.ones((2, dimension), np.float64),
                    np.full((2, dimension), np.nan, np.float32)):
        with pytest.raises(comparison.old.IntegrityError, match="Invalid real"):
            comparison.validate_vectors(vectors, 2, dimension)


def test_single_component_keeps_exact_stored_vectors():
    values = normalize(np.random.default_rng(2).normal(size=(4, 512)).astype(np.float32))
    assert comparison.system_vectors({"members": ["mvp"]}, {"mvp": values}) is values


def test_runtime_unavailable_is_not_silently_changed(monkeypatch):
    ctx = {"device": torch.device("mps"), "manifest": {"runtime": {"device": "mps"}}}
    monkeypatch.setattr(comparison.old.seeds, "runtime", lambda *_: {"device": "mps"})
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(comparison.old.IntegrityError, match="no silent CPU fallback"):
        comparison.check_runtime(ctx)


def test_notebook_is_valid_run_all_without_training():
    notebook = nbformat.read(comparison.VARIANT/"compare_saved_models.ipynb", as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"v22_cell_{i}", "exec")
    text = "\n".join(c.source for c in notebook.cells)
    assert "RUN_NAME = 'comparison_v1'" in text and "ALLOW_OUTER_EVALUATION = True" in text
    assert "experiment.prepare" in text and "experiment.run" in text
    assert "development-сравнение" in text and "Общего ограничения времени нет" in text
    assert "fit_job(" not in text and "recalibrate_bn(" not in text
