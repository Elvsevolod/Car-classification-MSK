"""v23 synthetic tests; never train or rewrite original v22 evidence."""
from copy import deepcopy
from pathlib import Path

import nbformat
import numpy as np
import pytest

from backend.core import normalize, sha256
from training import mvp_v21_fusion as fusion
from training.stage6 import write_json


def features(count=18):
    rng = np.random.default_rng(67)
    return tuple(normalize(rng.normal(size=(count, d)).astype(np.float32)) for d in (512, 1536))


@pytest.mark.parametrize("weight", [0., .25, .5, .75, 1.])
def test_different_dimensions_preserve_weighted_cosine_geometry(weight):
    mvp, v21 = features()
    result = fusion.fuse(mvp, v21, weight)
    if weight == 1:
        assert result is mvp
    elif weight == 0:
        assert result is v21
    else:
        assert result.shape == (18, 2048) and result.dtype == np.float32
    value = result.astype(np.float64)
    a, b = mvp.astype(np.float64), v21.astype(np.float64)
    np.testing.assert_allclose(value @ value.T, weight*(a @ a.T)+(1-weight)*(b @ b.T), atol=4e-7, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=2e-5, rtol=0)


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_weights(weight):
    with pytest.raises(ValueError, match="weight"):
        fusion.fuse(*features(), weight)


def test_bad_vectors_and_unaligned_rows():
    a, b = features()
    for bad in (a.astype(np.float64), np.zeros_like(a), np.full_like(a, np.nan), a[:, :511]):
        with pytest.raises(fusion.old.IntegrityError):
            fusion.fuse(bad, b, .5)
    with pytest.raises(ValueError, match="aligned"):
        fusion.fuse(a[:1], b, .5)


def test_frozen_grid_is_six_mixtures_and_two_controls():
    grid = fusion.systems()
    assert len(grid) == len({s["name"] for s in grid}) == 8
    assert [s["name"] for s in grid[:2]] == list(fusion.CONTROLS)
    assert {(s["mvp_weight"], s["lambda"]) for s in grid[2:]} == {
        (w, lam) for w in (.75, .5, .25) for lam in (.5, .65)}
    assert all(s["candidate_policy"] == "raw_top1" and s["dimension"] == 2048 for s in grid[2:])
    assert grid[0]["candidate_policy"] == "ranking_top1"


def test_selection_is_map_first_calibration_only_complete_and_stable():
    specs = fusion.systems()
    reports = {s["name"]: {"system": s, "split": "calibration", "ranking": {"mAP@10": .8},
                            "candidates": {"C": .7}} for s in specs}
    assert fusion.choose(reports, specs) == "MVP_control"
    reports[specs[2]["name"]]["candidates"]["C"] = .75
    assert fusion.choose(reports, specs) == specs[2]["name"]
    reports[specs[3]["name"]]["ranking"]["mAP@10"] = .81
    assert fusion.choose(reports, specs) == specs[3]["name"]
    bad = deepcopy(reports); bad[specs[0]["name"]] = None
    with pytest.raises(ValueError, match="every planned"):
        fusion.choose(bad, specs)
    for field, value in (("split", "validation"), ("ranking", {"mAP@10": float("nan")})):
        bad = deepcopy(reports); bad[specs[0]["name"]][field] = value
        with pytest.raises(fusion.old.IntegrityError, match="calibration"):
            fusion.choose(bad, specs)


@pytest.mark.parametrize("lam", [.50, .65])
def test_query_independence_for_real_fusion_geometry(lam):
    values = fusion.fuse(*features(31), .75)
    q, g = values[:7], values[7:]
    expected, _ = fusion.search.rank(q, g, lam)
    reverse, _ = fusion.search.rank(q[::-1], g, lam)
    np.testing.assert_array_equal(expected["order"], reverse["order"][::-1])
    for size in (1, 8, 16, 32):
        ranks = [fusion.search.rank(q[i:i+size], g, lam)[0]["order"] for i in range(0, len(q), size)]
        np.testing.assert_array_equal(expected["order"], np.concatenate(ranks))
    single, _ = fusion.search.rank(q[2:3], g, lam)
    np.testing.assert_array_equal(expected["order"][2], single["order"][0])


@pytest.fixture
def context(tmp_path, monkeypatch):
    rows, protocols = [], {}
    parts = ["mvp", "avg15", "r16", "r17"]
    components = {n: {"kind": "saved", "dimension": 512} for n in parts}
    parent = {"output": tmp_path/"v22", "signature": "source", "manifest": {"source_sha256": {}}}
    queue = fusion.old.Queue(parent)
    for split, offset in (("calibration", 0), ("validation", 20)):
        q = [{"image_id": f"{split}_q{i}", "vehicle_id": offset+i, "camera_id": 0} for i in (0, 1, 2, 3, 99, 100)]
        g = [{"image_id": f"{split}_g{i}", "vehicle_id": offset+i//2, "camera_id": 1} for i in range(12)]
        rows.extend(q+g)
        protocols[split] = {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g]}
        for j, name in enumerate(parts):
            rng = np.random.default_rng(100+j)
            gv = normalize(rng.normal(size=(12, 512)).astype(np.float32))
            qv = normalize(np.stack([gv[0]+.01, gv[2]+.01, gv[4]+.01, gv[6]+.01,
                                    *rng.normal(size=(2, 512))]).astype(np.float32))

            def save(directory):
                path = directory/"features.npz"
                np.savez_compressed(path, ids=[r["image_id"] for r in q+g], vectors=np.concatenate([qv, gv]))
                return {"path": str(path), "sha256": sha256(path), "model": components[name], "split": split}

            queue.task(f"features_{split}_{name}", save)
    plan = {"systems": fusion.systems(), "source_signature": "source", "components": parts,
            "v21_system": fusion.search.system("v21", parts[1:]), "fusion": "weighted concat"}
    ctx = {"output": tmp_path/"v23", "signature": "new", "source_output": parent["output"], "rows": rows,
           "source_manifest": {"comparison_plan": {"components": components}}, "protected": {},
           "manifest": {"source_sha256": {}, "protocols": protocols, "fusion_plan": plan, "analysis_runtime": fusion.runtime()}}
    monkeypatch.setattr(fusion.old.review, "check_inputs", lambda *a, **kw: None)
    monkeypatch.setattr(fusion, "check_reference", lambda *a: None)
    return ctx


def test_no_validation_load_before_calibration_freeze(context):
    with pytest.raises(FileNotFoundError):
        fusion.load_vectors(context, "validation")
    with pytest.raises(ValueError, match="calibration-only"):
        fusion.calibrate(context, {}, (), split="validation")
    with pytest.raises(ValueError, match="allow_outer"):
        fusion.run(context)


def test_end_to_end_only_selected_and_controls_export_then_resume(context, monkeypatch):
    original = fusion.calibrate

    def preferred(ctx, spec, vectors, *, split):
        result = original(ctx, spec, vectors, split=split)
        result["ranking"]["mAP@10"] = .95 if spec["name"] == "mix_mvp50_lambda65" else .8
        return result

    monkeypatch.setattr(fusion, "calibrate", preferred)
    calls = []
    load = fusion.load_vectors

    def spy(ctx, split):
        if split == "validation":
            assert len(list(ctx["output"].glob("tasks/calibrate_*/complete.json"))) == 8
            assert (ctx["output"]/"frozen_selection.json").is_file()
        calls.append(split)
        return load(ctx, split)

    monkeypatch.setattr(fusion, "load_vectors", spy)
    result = fusion.run(context, allow_outer=True)
    assert calls == ["calibration", "validation"]
    assert result["selection"]["selected"] == "mix_mvp50_lambda65"
    assert set(result["evaluations"]) == {*fusion.CONTROLS, "mix_mvp50_lambda65"}
    assert len(list(context["output"].glob("tasks/evaluate_*"))) == 3
    assert result["status"] == "complete" and result["protected_unchanged"]
    assert result["optimizer_updates"] == result["encoder_forwards"] == 0 and not result["promoted"]
    for name, report in result["evaluations"].items():
        output = Path(report["export"])
        q, g = fusion.old.review.old.protocol_rows(context, "validation")
        qf, gf = fusion.old.policy.frames(q, g)
        ranked = fusion.old.policy.official.load_submission(output/"submission.csv", set(gf.index))
        candidates = fusion.old.policy.official.load_candidates(output/"candidates.csv")
        assert all(len(ids) == len(set(ids)) == 10 for ids in ranked.values())
        assert report["ranking"] == fusion.old.policy.official.ranking_metrics(qf, gf, ranked)
        assert report["candidates"] == fusion.old.policy.candidate_metrics(qf, gf, candidates)
        values = np.load(output/"embeddings.npy", allow_pickle=False)
        fusion.previous.validate_vectors(values, 18, report["system"]["dimension"])
        assert fusion.old.read(output/"embedding_order.json")["ids"] == [r["image_id"] for r in q+g]
    again = fusion.run(context, allow_outer=True)
    assert again["evaluations"] == result["evaluations"]
    assert all(e["status"] == "cached" for e in again["events"])
    case = {"system": fusion.systems()[2], "threshold": .5}
    with pytest.raises(fusion.old.IntegrityError, match="Only the selected"):
        fusion.evaluate(context, case, (), context["output"])


def test_failed_calibration_never_selects_or_loads_validation(context, monkeypatch):
    original, load = fusion.calibrate, fusion.load_vectors
    calls = []

    def fail(ctx, spec, vectors, *, split):
        if spec["name"] == "mix_mvp25_lambda65":
            raise RuntimeError("temporary calibration failure")
        return original(ctx, spec, vectors, split=split)

    def spy(ctx, split):
        calls.append(split)
        return load(ctx, split)

    monkeypatch.setattr(fusion, "calibrate", fail)
    monkeypatch.setattr(fusion, "load_vectors", spy)
    with pytest.raises(ValueError, match="every planned"):
        fusion.run(context, allow_outer=True)
    assert calls == ["calibration"] and not (context["output"]/"frozen_selection.json").exists()
    monkeypatch.setattr(fusion, "calibrate", original)
    assert fusion.run(context, allow_outer=True)["status"] == "complete"


def test_changed_source_cache_and_row_order_rejected(context):
    path = context["source_output"]/"tasks/features_calibration_mvp/features.npz"
    path.write_bytes(b"corrupt")
    with pytest.raises(fusion.old.IntegrityError, match="Source artifact changed"):
        fusion.load_vectors(context, "calibration")


def test_changed_protocol_order_rejected(context):
    context["manifest"]["protocols"]["calibration"]["query_ids"].reverse()
    with pytest.raises(fusion.old.IntegrityError, match="IDs/order"):
        fusion.load_vectors(context, "calibration")


def test_frozen_winner_or_threshold_cannot_change_after_validation(context):
    fusion.run(context, allow_outer=True)
    path = context["output"]/"frozen_selection.json"
    original = fusion.old.read(path)
    changed = deepcopy(original); changed["evaluations"][0]["threshold"] += .01
    write_json(path, changed)
    with pytest.raises(fusion.old.IntegrityError, match="Frozen selection"):
        fusion.read_selection(context)
    changed = deepcopy(original); changed["selected"] = "unplanned"
    write_json(path, changed)
    with pytest.raises(fusion.old.IntegrityError, match="Frozen selection"):
        fusion.read_selection(context)


def test_reference_disagreement_is_not_silently_accepted(monkeypatch):
    spec = fusion.systems()[0]
    monkeypatch.setattr(fusion, "source_task", lambda *a: {"selected": {"threshold": .3}})
    fusion.check_reference({}, "calibration", spec, {"calibration": {"selected": {"threshold": .3}}})
    with pytest.raises(fusion.old.IntegrityError, match="no longer reproduces"):
        fusion.check_reference({}, "calibration", spec, {"calibration": {"selected": {"threshold": .4}}})


def test_control_csv_must_match_even_when_metrics_are_equal(tmp_path, monkeypatch):
    before = tmp_path/"source/tasks/evaluate_MVP_active/export"
    after = tmp_path/"new"
    before.mkdir(parents=True); after.mkdir()
    for name in ("submission.csv", "candidates.csv"):
        (before/name).write_text("original\n")
        (after/name).write_text("original\n")
    metrics = {"threshold": .3, "ranking": {}, "candidates": {}, "per_query": {}}
    monkeypatch.setattr(fusion, "source_task", lambda *a: metrics)
    ctx, report = {"source_output": tmp_path/"source"}, {**metrics, "export": str(after)}
    fusion.check_reference(ctx, "validation", fusion.systems()[0], report)
    (after/"submission.csv").write_text("different negative ordering\n")
    with pytest.raises(fusion.old.IntegrityError, match="no longer reproduces"):
        fusion.check_reference(ctx, "validation", fusion.systems()[0], report)


def test_notebook_valid_with_no_model_inference_or_training():
    notebook = nbformat.read(fusion.VARIANT/"fuse_mvp_v21.ipynb", as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"v23_cell_{i}", "exec")
    text = "\n".join(c.source for c in notebook.cells)
    assert "RUN_NAME = 'fusion_v1'" in text and "SOURCE_RUN = 'comparison_v1'" in text
    assert "experiment.prepare" in text and "experiment.run" in text
    assert "Общего ограничения времени нет" in text and "encoder_forwards" in text
    assert "fit_job(" not in text and "encode_rows(" not in text
