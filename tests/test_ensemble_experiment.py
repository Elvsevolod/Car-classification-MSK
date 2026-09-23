"""Fusion math, data separation, deterministic inference and frozen calibration."""
import copy

import numpy as np
import pytest
from PIL import Image

from backend.core import normalize
from backend.rerank import rerank_protocol
from backend.scoring import metrics
from training import ensemble_experiment as audit
from training import ensemble_inference as inference


def example(seed=0, prefix=""):
    rng = np.random.default_rng(seed)
    q = [{"image_id": f"{prefix}q{i}", "vehicle_id": i, "camera_id": 0,
          "x": 0, "y": 0, "w": 10, "h": 10} for i in range(8)]
    g = [{"image_id": f"{prefix}g{i}", "vehicle_id": i % 6, "camera_id": i % 3,
          "x": 0, "y": 0, "w": 10, "h": 10} for i in range(65)]
    embeddings = [dict(zip((r["image_id"] for r in q + g), normalize(rng.normal(size=(73, d)).astype(np.float32))))
                  for d in (16, 32)]
    return q, g, *embeddings


@pytest.mark.parametrize("alpha", audit.ALPHAS)
def test_fusion_equals_weighted_cosine_with_different_feature_spaces(alpha):
    rng = np.random.default_rng(41)
    o, c = [rng.normal(size=(12, d)).astype(np.float32) for d in (16, 32)]
    actual = inference.fuse_arrays(o, c, alpha)
    expected = alpha * (normalize(o) @ normalize(o).T) + (1-alpha) * (normalize(c) @ normalize(c).T)
    np.testing.assert_allclose(actual @ actual.T, expected, atol=3e-7)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
    assert actual.dtype == np.float32
    assert actual.shape[1] == (16 if alpha == 1 else 32 if alpha == 0 else 48)
    with pytest.raises(ValueError):
        inference.fuse_arrays(o[:2], c, alpha)


def test_grid_is_preregistered_and_refinement_changes_only_one_axis():
    grid = audit.initial_grid()
    assert len(grid) == 16
    assert tuple(dict.fromkeys(c["alpha"] for c in grid)) == audit.ALPHAS
    assert sum(c["pool"] == "union50" for c in grid) == 4
    base = audit.configuration(.75, pool="union50")
    refined = audit.refinement_grid(base)
    assert len(refined) == 7 and refined[0] == base
    assert all(sum(c[k] != base[k] for k in c) == 1 for c in refined[1:])


def test_union_is_deduplicated_stable_and_restricts_before_ground_truth():
    o = np.zeros((1, 120), np.float32)
    c = o.copy()
    o[0, :50], c[0, 25:75] = 1, 1
    pool = audit.union_pool(o, c)
    assert pool.sum() == 75 and pool[0, :75].all()
    np.testing.assert_array_equal(audit.union_pool(o, o), o.astype(bool))
    q, g, osnet, clip = example()
    engine = audit.FusionScores(q, g, osnet, clip)
    config = audit.configuration(.75, pool="union50")
    scores = engine.scores(config)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    assert np.take_along_axis(engine.union, top, axis=1).all()


def test_ranking_uses_no_other_queries_or_labels_and_endpoints_match():
    q, g, o, c = example()
    first = audit.FusionScores(q, g, o, c)
    only = audit.FusionScores(q[:1], g, o, c)
    relabel = lambda rows: [{**r, "vehicle_id": 99, "camera_id": 77} for r in rows]
    changed = audit.FusionScores(relabel(q), relabel(g), o, c)
    for config in (audit.configuration(.75), audit.configuration(.5, pool="union50")):
        scores = first.scores(config)
        np.testing.assert_allclose(scores[0], only.scores(config)[0], atol=1e-6)
        np.testing.assert_array_equal(scores, changed.scores(config))
    for alpha, vectors in ((1., o), (0., c)):
        expected, _ = rerank_protocol(q, g, vectors)
        actual = first.evaluate(audit.configuration(alpha))
        assert expected.predictions == actual.predictions
        np.testing.assert_allclose(expected.confidence, actual.confidence, atol=1e-6)


def test_complementarity_counts_valid_positives_and_union_upper_bound():
    q, g, o, c = example()
    report = audit.complementarity(audit.FusionScores(q, g, o, c))
    assert report["known_queries"] == 4  # IDs 0 and 3 occur only in the query's camera.
    assert sum(report["raw_top1_overlap"].values()) == 4
    pools = report["pools"]
    for name in ("osnet50", "clip50"):
        assert pools["union50"]["query_hit_rate"] >= pools[name]["query_hit_rate"]
        assert pools["union50"]["oracle_mAP_at_10_upper_bound"] >= pools[name]["oracle_mAP_at_10_upper_bound"]
    for detail in report["per_query"].values():
        assert detail["captured"]["union50"] <= detail["valid_positives"]


def test_tuning_accepts_only_calibration_and_validation_does_not_mutate_choices():
    q, g, o, c = example()
    tuned = audit.tune_calibration(q, g, o, c)
    frozen = {**tuned, "baseline_threshold": .3, "osnet_neighbors": audit.control.ACTIVE}
    before = copy.deepcopy(frozen)
    qv, gv, ov, cv = example(11, "unseen_")
    evaluated, engine = audit.evaluate_frozen(qv, gv, ov, cv, frozen)
    assert before == frozen and tuned["validation_used"] is False
    assert len(evaluated) == 6
    assert {r["config"]["alpha"] for r in tuned["refinement"]} == {tuned["choices"]["fusion_fixed"]["config"]["alpha"]}
    assert all(qid.startswith("unseen_") for qid in engine.ids)


def test_cache_resumes_and_rejects_other_masks_or_signatures(monkeypatch, tmp_path):
    q = [{"image_id": str(i)} for i in range(3)]
    monkeypatch.setattr(inference, "load_crop", lambda *_: Image.new("RGB", (10, 10), "white"))
    class Encoder:
        calls = 0
        def encode_batch(self, tensors):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interruption")
            return normalize(np.ones((len(tensors), 1280), dtype=np.float32))
    encoder, path = Encoder(), tmp_path / "cache.npz"
    with pytest.raises(RuntimeError, match="interruption"):
        inference.encode_cached(encoder, q, path, {"weights": "hash"}, batch_size=1)
    with np.load(path) as cache:
        assert cache["ids"].tolist() == ["0"]
    actual = inference.encode_cached(encoder, q, path, {"weights": "hash"}, batch_size=1)
    assert len(actual) == 3 and encoder.calls == 4
    inference.encode_cached(encoder, q, path, {"weights": "hash"})
    assert encoder.calls == 4
    with pytest.raises(ValueError, match="different inputs"):
        inference.encode_cached(encoder, q, path, {"weights": "changed"})
    with pytest.raises(ValueError, match="different inputs"):
        inference.encode_cached(encoder, q, path, {"weights": "hash"}, {str(i): {"rectangles": []} for i in range(3)})


def test_mask_is_opaque_before_clip_resize_and_source_is_unchanged(monkeypatch, tmp_path):
    crop = Image.new("RGB", (10, 10), "white")
    monkeypatch.setattr(inference, "load_crop", lambda *_: crop)
    class Encoder:
        def encode_batch(self, tensors):
            assert tensors[0].shape == (3, 256, 256)
            np.testing.assert_array_equal(tensors[0][:, :70, :70], -1.)
            np.testing.assert_array_equal(tensors[0][:, 220:, 220:], 1.)
            return normalize(np.ones((1, 1280), dtype=np.float32))
    inference.encode_cached(Encoder(), [{"image_id": "q"}], tmp_path / "mask.npz", {},
                            {"q": {"rectangles": [[0, 0, 5, 5]]}})
    assert crop.getpixel((0, 0)) == (255, 255, 255)


def test_run_freezes_before_validation_resumes_and_roundtrips_official(monkeypatch, tmp_path):
    q, g, o, c = example()
    qv, gv, ov, cv = example(5, "validation_")
    prototypes = {"calibration": (q, g), "validation": (qv, gv)}
    baseline = {"threshold": .3, **{name: metrics(rerank_protocol(qa, ga, e)[0], .3)
        for name, (qa, ga, e) in {"calibration": (q, g, o), "validation": (qv, gv, ov)}.items()}}
    signature = {"protected_sha256": {}, "clip_train_identities": 741}
    monkeypatch.setattr(audit, "prepare", lambda: (prototypes, {**o, **ov}, baseline, signature, {},
                                                 {"ranking": audit.control.ACTIVE}, None, 0.))
    monkeypatch.setattr(audit, "export_checkpoint", lambda *_: (None, {}))
    def encode(_encoder, rows, path, *_):
        if path.name == "validation.npz":
            assert (tmp_path / "frozen_selection.json").is_file()
        return {r["image_id"]: {**c, **cv}[r["image_id"]] for r in rows}
    monkeypatch.setattr(audit, "encode_cached", encode)
    monkeypatch.setattr(audit, "mask_check", lambda *_: {})
    monkeypatch.setattr(audit, "benchmark_extract", lambda *_: {})
    monkeypatch.setattr(audit, "benchmark_search", lambda *_: {})
    monkeypatch.setattr(audit, "report_markdown", lambda *_: "Verified synthetic report")
    report = audit.run(tmp_path)
    assert report["baseline_reproduced"] and report["protected_unchanged"]
    def forbidden(*_):
        raise AssertionError("Completed run must not retune or reevaluate")
    monkeypatch.setattr(audit, "tune_calibration", forbidden)
    monkeypatch.setattr(audit, "evaluate_frozen", forbidden)
    assert audit.run(tmp_path) == report
    assert len((tmp_path / "validation/fusion_selected/predictions/submission.csv").read_text().splitlines()) == len(qv)
    (tmp_path / "validation/clip_raw/predictions/candidates.csv").write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        audit.run(tmp_path)


def test_checkpoint_provenance_rejects_split_or_preprocess_changes():
    root = audit.CLIP / "results"
    if not (root / "protocol.json").exists():
        pytest.skip("Local training artifacts not present")
    protocol, manifest, summary, winner = [audit.load_json(root / p) for p in
        ("protocol.json", "manifest.json", "runs/trial_011/summary.json", "experiment_summary.json")]
    split = audit.load_json(audit.ARTIFACTS / "splits.json")
    rows = audit.read_rows(audit.DATASET / "train.csv")
    args = (manifest, summary, winner, rows, split["frame_sha256"], split)
    audit.check_clip_protocol(protocol, *args)
    for corrupt in ({**protocol, "preprocess": "other"}, {**protocol, "outer": {}},
                    {**protocol, "data_sha256": "different frames"}):
        with pytest.raises(ValueError, match="provenance"):
            audit.check_clip_protocol(corrupt, *args)


def test_notebook_has_no_training_or_dependency_changes():
    import nbformat
    notebook = nbformat.read(audit.EXPERIMENT / "osnet_clip_ensemble.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "osnet_clip_ensemble.ipynb", "exec")
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "report = run(OUTPUT)" in source and "pip install" not in source and "train(" not in source
