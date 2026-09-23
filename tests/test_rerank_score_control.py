"""Calibration isolation, streaming behavior and exact organizer-compatible refusal."""
import copy
import json

import numpy as np
import pytest

from backend.core import normalize
from backend.rerank import rerank_protocol
from backend.scoring import calibrate, metrics, ranked_queries
from training import rerank_score_control as audit


def example(seed=0, prefix=""):
    rng = np.random.default_rng(seed)
    query = [{"image_id": f"{prefix}q{i}", "vehicle_id": i, "camera_id": 0,
              "x": 0, "y": 0, "w": 10, "h": 10} for i in range(8)]
    gallery = [{"image_id": f"{prefix}g{i}", "vehicle_id": i % 6, "camera_id": i % 3,
                "x": 0, "y": 0, "w": 10, "h": 10} for i in range(65)]
    vectors = normalize(rng.normal(size=(len(query) + len(gallery), 16)).astype(np.float32))
    embeddings = dict(zip((r["image_id"] for r in query + gallery), vectors))
    return query, gallery, embeddings


def test_grid_is_small_fixed_and_changes_one_setting_at_a_time():
    grid = audit.configurations()
    assert len(grid) == 13 and len({c["name"] for c in grid}) == 13
    assert grid[0] == audit.ACTIVE
    assert all(c["pool"] == 50 for c in grid[2:])
    assert len(audit.CONFIDENCE_MODES) == 6
    assert audit.configurations() == grid


@pytest.mark.parametrize("seed", range(5))
def test_fast_threshold_matches_official_slow_selection_with_ties_and_junk(seed):
    q, g, embeddings = example(seed)
    ranked = ranked_queries(q, g, embeddings)
    confidence = np.round(np.random.default_rng(seed).random(len(q)), 1)
    result = audit.choose_threshold(ranked, confidence)
    assert result["threshold"] == calibrate(ranked, confidence)
    assert result["metrics"] == metrics(ranked, result["threshold"], confidence)
    for t, f1, tnr, score in zip(*(result["curve"][k] for k in ("threshold", "candidate_F1", "TNR", "candidate_score"))):
        actual = metrics(ranked, t, confidence)
        assert f1 == pytest.approx(actual["candidate_F1"])
        assert tnr == pytest.approx(actual["TNR"])
        assert score == pytest.approx(actual["candidate_score"])


def test_candidate_junk_semantics_are_not_replaced_by_ranking_semantics():
    q = [{"image_id": "q", "vehicle_id": 1, "camera_id": 0},
         {"image_id": "u", "vehicle_id": 99, "camera_id": 0}]
    g = [{"image_id": "junk", "vehicle_id": 1, "camera_id": 0},
         {"image_id": "negative", "vehicle_id": 2, "camera_id": 0},
         {"image_id": "positive", "vehicle_id": 1, "camera_id": 1}]
    ranked = ranked_queries(q, g, {r["image_id"]: [1., 0.] for r in q + g})
    result = audit.choose_threshold(ranked, [.9, .1])
    assert result["metrics"]["TP"] == result["metrics"]["TN"] == 1
    assert result["metrics"]["Rank_1"] == 0


def test_active_ranking_and_confidence_reproduce_current_backend():
    q, g, embeddings = example()
    expected, _ = rerank_protocol(q, g, embeddings)
    actual, confidence = audit.ProtocolScores(q, g, embeddings).evaluate(audit.ACTIVE)
    assert actual.predictions == expected.predictions
    np.testing.assert_array_equal(confidence["max_cosine"], expected.confidence)
    assert metrics(actual, .3) == metrics(expected, .3)


def test_ranking_features_use_only_current_query_and_static_gallery():
    q, g, embeddings = example()
    gallery = np.stack([embeddings[r["image_id"]] for r in g])
    engine = audit.ScoreControl(gallery)
    saved_graph = engine.graph.encoding.copy()
    before = engine.components(embeddings[q[0]["image_id"]])
    engine.components(embeddings[q[-1]["image_id"]])
    after = engine.components(embeddings[q[0]["image_id"]])
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(engine.graph.encoding, saved_graph)
    assert all(i not in neighbors for i, neighbors in enumerate(engine.neighbors))
    # Labels and camera metadata are unavailable to ScoreControl; changing them cannot affect predictions.
    relabeled = [{**r, "vehicle_id": 777, "camera_id": 999} for r in q]
    for config in audit.configurations()[::3]:
        first, _ = audit.ProtocolScores(q, g, embeddings).evaluate(config)
        reordered, _ = audit.ProtocolScores(relabeled[::-1], g, embeddings).evaluate(config)
        assert first.predictions == reordered.predictions


def test_top50_restriction_is_before_any_labels_or_junk_filter():
    parts = {"cosine": np.linspace(1, 0, 65, dtype=np.float32), "raw": np.linspace(0, 1, 65),
             "jaccard": np.zeros(65), "support": np.zeros(65), "mutual": np.zeros(65)}
    # Outside candidate gets an overwhelming auxiliary signal, but still cannot enter top10.
    parts["mutual"][-1] = 1000
    config = {**audit.configurations()[2], "mutual_weight": 1.}
    scores = audit.ranking_scores(parts, config)
    assert set(np.argsort(-scores, kind="stable")[:10]) <= set(range(50))
    assert 64 not in np.argsort(-scores, kind="stable")[:50]


def test_all_confidence_variants_are_explicit_and_finite():
    q, g, embeddings = example()
    ranked, values = audit.ProtocolScores(q, g, embeddings).evaluate(audit.ACTIVE)
    assert tuple(values) == audit.CONFIDENCE_MODES
    for array in values.values():
        assert array.shape == (len(q),) and np.isfinite(array).all()
    choice, trials = audit.select_confidence(ranked, values)
    assert len(trials) == 6 and choice["mode"] in audit.CONFIDENCE_MODES
    assert choice["threshold"] in next(t["curve"]["threshold"] for t in trials if t["mode"] == choice["mode"])


def test_calibration_has_no_validation_access_and_freezes_four_factorial_controls():
    q, g, embeddings = example()
    frozen = audit.tune_calibration(q, g, embeddings, .3)
    assert frozen["validation_used"] is False
    assert len(frozen["leaderboard"]) == 13
    assert len(frozen["confidence_trials"]["selected"]) == 6
    selected = copy.deepcopy(frozen)
    qv, gv, ev = example(3, "unseen_")
    results = audit.evaluate_frozen(qv, gv, ev, frozen)
    assert frozen == selected
    assert list(results) == ["baseline", "ranking_only", "refusal_only", "combined"]
    assert results["ranking_only"]["refusal"] == results["baseline"]["refusal"]
    assert results["refusal_only"]["ranking"] == results["baseline"]["ranking"]
    assert results["combined"]["ranking"] == frozen["ranking"]
    assert results["combined"]["refusal"] == frozen["confidence"]["selected"]
    assert results["refusal_only"]["metrics"]["mAP_at_10"] == results["baseline"]["metrics"]["mAP_at_10"]


def test_export_is_local_and_roundtrips_organizer_parsers(tmp_path):
    q, g, embeddings = example()
    frozen = {"ranking": audit.configurations()[2], "baseline_threshold": .3,
              "confidence": {k: {"mode": "selected_cosine", "threshold": .5} for k in ("active", "selected")}}
    results = audit.evaluate_frozen(q, g, embeddings, frozen)
    audit.export_validation(tmp_path / "local", q, g, embeddings, results)
    for name in results:
        rows = (tmp_path / "local" / name / "submission.csv").read_text().splitlines()
        assert len(rows) == len(q) and all(len(r.split(",")) == 11 for r in rows)
    assert np.load(tmp_path / "local/embeddings.npy").shape == (73, 16)


def test_run_freezes_before_validation_and_repeat_never_retunes(monkeypatch, tmp_path):
    q, g, e = example()
    qv, gv, ev = example(2, "validation_")
    protocols = {"calibration": (q, g), "validation": (qv, gv)}
    ranked, _ = rerank_protocol(q, g, e)
    val, _ = rerank_protocol(qv, gv, ev)
    baseline = {"threshold": .3, "calibration": metrics(ranked, .3), "validation": metrics(val, .3)}
    signature = {"protected_sha256": {}, "weights_bytes": 1, "model_sha256": "test"}
    monkeypatch.setattr(audit, "prepare", lambda: (protocols, {**e, **ev}, baseline, signature, {}, 0.))
    monkeypatch.setattr(audit, "mask_check", lambda *_: {"scores": {}})
    monkeypatch.setattr(audit, "benchmark", lambda *_: {"gallery_graph_seconds": 0., "median_query_ms": 1., "p95_query_ms": 1.})
    real_evaluate = audit.evaluate_frozen
    def checked(*args):
        assert (tmp_path / "frozen_selection.json").is_file()
        assert args[0] == qv
        return real_evaluate(*args)
    monkeypatch.setattr(audit, "evaluate_frozen", checked)
    report = audit.run(tmp_path)
    assert report["baseline_reproduced"] == {"calibration": True, "validation": True}
    def forbidden(*_):
        raise AssertionError("Completed run must not retune or reevaluate")
    monkeypatch.setattr(audit, "tune_calibration", forbidden)
    monkeypatch.setattr(audit, "evaluate_frozen", forbidden)
    assert audit.run(tmp_path) == report
    frozen = json.loads((tmp_path / "frozen_selection.json").read_text())
    assert frozen["validation_used"] is False
    (tmp_path / "validation/combined/candidates.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="output changed"):
        audit.run(tmp_path)


def test_notebook_is_clean_valid_and_contains_no_training():
    import nbformat
    notebook = nbformat.read(audit.EXPERIMENT / "rerank_score_control.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "rerank_score_control.ipynb", "exec")
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "report = run(OUTPUT)" in source
    assert "pip install" not in source and "train(" not in source
