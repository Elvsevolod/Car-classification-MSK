"""v24 controller: frozen roles, source receipts, fresh pass, fail-closed resume."""
import ast
from pathlib import Path

import nbformat
import numpy as np
import pytest

from backend.core import normalize, sha256
from training import dual_role_experiment as experiment


@pytest.fixture
def context(tmp_path, monkeypatch):
    old, dual = experiment.old, experiment.inference
    rows, protocols, stored = [], {}, {}
    components = {n: {"kind": "saved", "dimension": d} for n, d in (("mvp", 512), ("full_v18", 1536))}
    source = {"output": tmp_path/"v22", "signature": "source", "manifest": {"source_sha256": {}}}
    queue = old.Queue(source)
    for split, offset in (("calibration", 0), ("validation", 20)):
        q = [{"image_id": f"{split}_q{i}", "vehicle_id": offset+i, "camera_id": 0} for i in (0, 1, 2, 3, 99, 100)]
        g = [{"image_id": f"{split}_g{i}", "vehicle_id": offset+i//2, "camera_id": 1} for i in range(12)]
        rows.extend(q+g)
        protocols[split] = {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g]}
        vectors = []
        for name, spec in components.items():
            rng = np.random.default_rng(spec["dimension"])
            gv = normalize(rng.normal(size=(12, spec["dimension"])).astype(np.float32))
            qv = normalize(np.stack([gv[0]+.01, gv[2]+.01, gv[4]+.01, gv[6]+.01,
                                    *rng.normal(size=(2, spec["dimension"]))]).astype(np.float32))
            values = np.concatenate([qv, gv])
            vectors.append(values)

            def save(directory):
                path = directory/"features.npz"
                np.savez_compressed(path, ids=[r["image_id"] for r in q+g], vectors=values)
                return {"path": str(path), "sha256": sha256(path), "model": spec, "split": split}

            queue.task(f"features_{split}_{name}", save)
            setting = "legacy" if name == "mvp" else "less_graph"
            candidate = "ranking_top1" if name == "mvp" else "raw_top1"
            ranking = dual.policy.rank_vectors(qv, gv, setting)
            if split == "validation":
                title = "MVP_active" if name == "mvp" else "R1_equal3_full_v18"
                queue.task(f"evaluate_{title}", lambda d: dual.policy.export_csv(d/"export", q, g, ranking, .1, candidate))
            elif name == "full_v18":
                selected = {"threshold": .1, **dual.policy.evaluate(q, g, ranking, .1, candidate)["candidates"]}
                queue.task("calibrate_R1_equal3_full_v18", lambda d: {"selected": selected})
        stored[split] = dual.pack(*vectors)
    ctx = {"output": tmp_path/"v24", "signature": "new", "source_output": source["output"], "rows": rows,
           "source_manifest": {"comparison_plan": {"components": components}}, "protected": {}, "dataset": tmp_path,
           "profile_path": tmp_path/"unused_fixture_profile", "manifest": {"source_sha256": {}, "protocols": protocols,
           "dual_role_plan": {"source_signature": "source", "batch_size": 16, "vector_atol": 2e-5},
           "analysis_runtime": experiment.runtime()}}
    monkeypatch.setattr(old.review, "check_inputs", lambda *a, **kw: None)
    monkeypatch.setattr(experiment, "frozen_profile", lambda c: {"threshold": .1})

    class Encoder:
        def __init__(self, path):
            assert path == ctx["profile_path"]

        def encode_rows(self, entries, dataset, batch_size):
            assert [r["image_id"] for r in entries] == protocols["validation"]["query_ids"]+protocols["validation"]["gallery_ids"]
            assert dataset == tmp_path and batch_size == 16
            return stored["validation"].copy()

    monkeypatch.setattr(dual, "DualRoleEncoder", Encoder)
    return ctx


def test_full_controller_exports_and_resumes_without_recomputing(context, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No threshold fitting in v24")
    monkeypatch.setattr(experiment.inference.policy, "calibrate_policy", forbidden)
    with pytest.raises(ValueError, match="allow_outer"):
        experiment.run(context)
    result = experiment.run(context, allow_outer=True)
    assert result["status"] == "complete" and result["protected_unchanged"]
    assert not result["promoted"] and not result["threshold_fit"]
    assert result["optimizer_updates"] == result["bn_updates"] == 0
    assert result["comparison"]["mAP_delta"] == 0
    assert len(result["evaluations"]) == 3
    fresh = result["evaluations"]["fresh_validation"]
    assert fresh["fresh_image_inference"] and fresh["parity"]["bit_exact_vectors"]
    assert fresh["parity"]["v22_csv_byte_parity"]
    cached = result["evaluations"]["cached_validation"]
    for name in ("submission.csv", "candidates.csv", "embeddings.npy"):
        assert sha256(Path(cached["export"])/name) == sha256(Path(fresh["export"])/name)
    assert "не независимый тест" in (context["output"]/"REPORT.md").read_text()
    monkeypatch.setattr(experiment, "fresh_validation", forbidden)
    resumed = experiment.run(context, allow_outer=True)
    assert resumed["evaluations"] == result["evaluations"]
    assert all(event["status"] == "cached" for event in resumed["events"])


def test_corrupt_source_cache_rejected_before_loading(context):
    path = context["source_output"]/"tasks/features_validation_mvp/features.npz"
    path.write_bytes(path.read_bytes()+b"modified")
    with pytest.raises(experiment.old.IntegrityError):
        experiment.load_vectors(context, "validation")


def test_changed_runtime_and_completed_export_fail_closed(context):
    context["manifest"]["analysis_runtime"]["numpy"] = "other"
    with pytest.raises(experiment.old.IntegrityError, match="runtime"):
        experiment.run(context, allow_outer=True)
    context["manifest"]["analysis_runtime"] = experiment.runtime()
    experiment.run(context, allow_outer=True)
    path = context["output"]/"tasks/cached_validation/export/candidates.csv"
    path.write_bytes(path.read_bytes()+b"modified")
    with pytest.raises(experiment.old.IntegrityError):
        experiment.run(context, allow_outer=True)
    assert experiment.old.read(context["output"]/"results.json")["status"] == "incomplete"


def test_fresh_failure_cannot_be_reported_complete(context, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Synthetic image inference failure")
    monkeypatch.setattr(experiment, "fresh_validation", fail)
    with pytest.raises(ValueError, match="Incomplete fresh"):
        experiment.run(context, allow_outer=True)
    result = experiment.old.read(context["output"]/"results.json")
    assert result["status"] == "incomplete" and not result["promoted"]
    assert not (context["output"]/"tasks/fresh_validation/complete.json").exists()


def test_parity_rejects_different_csv_even_with_unchanged_metrics(context):
    dual = experiment.inference
    q, g = experiment.old.review.old.protocol_rows(context, "validation")
    values = experiment.load_vectors(context, "validation")
    export = context["output"]/"deliberately_changed"
    metrics = dual.export_arrays({"threshold": .1}, q, g, values, export)
    path = export/"submission.csv"
    lines = path.read_text().splitlines()
    first = lines[0].split(",")
    first[-1], first[-2] = first[-2], first[-1]
    path.write_text("\n".join([",".join(first), *lines[1:]])+"\n")
    with pytest.raises(experiment.old.IntegrityError, match="exactly reproduces"):
        experiment.verify_reference(context, "validation", q, g, values, metrics, export)


def test_notebook_is_run_all_without_search_training_or_time_limit():
    notebook = nbformat.read(experiment.VARIANT/"verify_dual_role.ipynb", as_version=4)
    nbformat.validate(notebook)
    sources = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    ast.parse(sources)
    assert "experiment.prepare(RUN_NAME" in sources and "experiment.run(context" in sources
    assert "ALLOW_OUTER_EVALUATION = True" in sources
    assert not any(word in sources for word in ("calibrate_policy", "optimizer.step", "WALL_HOURS"))
