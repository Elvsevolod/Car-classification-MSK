import copy

import numpy as np
import pytest
import torch

from backend.core import normalize
from backend.rerank import rerank_protocol
from backend.scoring import metrics
from training import pair_reranker as pair
from training import pair_reranker_experiment as experiment


def example(seed=0, prefix="", identity_offset=0):
    rng = np.random.default_rng(seed)
    q = [{"image_id": f"{prefix}q{i}", "vehicle_id": identity_offset+i, "camera_id": 0,
          "x": 0, "y": 0, "w": 10, "h": 10} for i in range(8)]
    g = [{"image_id": f"{prefix}g{i}", "vehicle_id": identity_offset+i%6, "camera_id": i%3,
          "x": 0, "y": 0, "w": 10, "h": 10} for i in range(65)]
    vectors = normalize(rng.normal(size=(73, 16)).astype(np.float32))
    e = dict(zip((r["image_id"] for r in q+g), vectors))
    return q, g, e


def dataset(seed=0, prefix="", identity_offset=0):
    q, g, e = example(seed, prefix, identity_offset)
    return pair.PairSet(q, g, e, {**experiment.control.ACTIVE, "pool": 50, "support_weight": .2})


def hashes(data):
    return {r["image_id"]: r["image_id"] for r in data.query+data.gallery}


class DummyHead:
    def probabilities(self, features):
        return 1/(1+np.exp(-features[..., 0]))


def test_features_are_streaming_and_do_not_accept_labels():
    q, g, e = example()
    engine = pair.PairFeatures(np.stack([e[r["image_id"]] for r in g]), experiment.control.ACTIVE)
    before = engine.one(e[q[0]["image_id"]])
    engine.one(e[q[-1]["image_id"]])
    after = engine.one(e[q[0]["image_id"]])
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    ids, features, _, _ = before
    assert features.shape == (50, len(pair.SCALARS)+32)
    np.testing.assert_allclose(features[:, 0], np.stack([e[r["image_id"]] for r in g])[ids] @ e[q[0]["image_id"]], atol=1e-6)
    relabel = lambda rows: [{**r, "vehicle_id": 888, "camera_id": 999} for r in rows]
    first = pair.PairSet(q, g, e, experiment.control.ACTIVE)
    second = pair.PairSet(relabel(q[::-1]), relabel(g), e, experiment.control.ACTIVE)
    np.testing.assert_array_equal(first.features, second.features[::-1])
    assert first.ranked(first.base).predictions == second.ranked(second.base).predictions


def test_labels_ignore_junk_and_duplicate_frames_and_balance_queries():
    data = dataset()
    frames = hashes(data)
    frames[data.gallery[int(data.indices[0, 0])]["image_id"]] = frames[data.query[0]["image_id"]]
    labels, weights, valid = data.labels(frames)
    assert weights[0, 0] == 0 and not valid[0, 0]
    for i, q in enumerate(data.query):
        for j, index in enumerate(data.indices[i]):
            g = data.gallery[int(index)]
            if q["vehicle_id"] == g["vehicle_id"] and q["camera_id"] == g["camera_id"]:
                assert weights[i, j] == 0
            if labels[i, j]:
                assert q["vehicle_id"] == g["vehicle_id"] and q["camera_id"] != g["camera_id"]
        if (valid[i] & (labels[i] == 1)).any() and (valid[i] & (labels[i] == 0)).any():
            assert weights[i, labels[i] == 1].sum() == pytest.approx(.5)
            assert weights[i, labels[i] == 0].sum() == pytest.approx(.5)
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)


def test_pool_restriction_and_zero_blend_match_frozen_neighborhood_control():
    data = dataset()
    config = {**experiment.control.ACTIVE, "pool": 50, "support_weight": .2}
    expected, _ = experiment.control.ProtocolScores(data.query, data.gallery, data.embeddings).evaluate(config)
    actual, confidence = experiment.score_pair_set(data, DummyHead().probabilities(data.features), 0.)
    assert expected.predictions == actual.predictions
    np.testing.assert_array_equal(expected.confidence, confidence["max_cosine"])
    tied = data.ranked(np.ones_like(data.base))
    for i, row in enumerate(data.query):
        permitted = {data.gallery[j]["image_id"] for j in data.indices[i]}
        assert len(tied.predictions[row["image_id"]]) == 10
        assert set(tied.predictions[row["image_id"]]) <= permitted
    # Sigmoid scores never expand the candidate pool or filter same-camera junk at inference.
    ranked, _ = experiment.score_pair_set(data, DummyHead().probabilities(data.features), 1.)
    assert set(ranked.predictions) == {r["image_id"] for r in data.query}


def test_training_uses_train_only_scaling_and_rejects_identity_overlap(tmp_path):
    train, val = dataset(), dataset(1, "v_", 100)
    labels, weights, valid = train.labels(hashes(train))
    config, budget = {**pair.HEADS[0]}, dict(max_epochs=3, min_epochs=1, patience=2, query_batch=4)
    report = pair.fit_head(train, labels, weights, valid, config, tmp_path, {}, validation=val, budget=budget)
    state = torch.load(tmp_path / "selected.pt", weights_only=True)
    expected = pair.selected_features(train, config)[valid].mean(0)
    np.testing.assert_allclose(state["model"]["mean"].numpy(), expected, atol=1e-7)
    assert 1 <= report["selected_epoch"] <= 3 and not report["outer_used_for_training"]
    with pytest.raises(ValueError, match="identity leakage"):
        pair.fit_head(train, labels, weights, valid, config, tmp_path / "leak", {}, validation=train, budget=budget)


def test_interrupted_training_resumes_exactly(monkeypatch, tmp_path):
    train = dataset()
    args = train.labels(hashes(train))
    config = {**pair.HEADS[1], "hidden": 8}
    budget = dict(max_epochs=4, min_epochs=1, patience=2, query_batch=4)
    pair.fit_head(train, *args, config, tmp_path / "reference", {}, fixed_epochs=4, budget=budget)
    save = pair._save_checkpoint
    def interrupt(path, value):
        save(path, value)
        if path.name == "last.pt" and len(value["history"]) == 2:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(pair, "_save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        pair.fit_head(train, *args, config, tmp_path / "resumed", {}, fixed_epochs=4, budget=budget)
    monkeypatch.setattr(pair, "_save_checkpoint", save)
    pair.fit_head(train, *args, config, tmp_path / "resumed", {}, fixed_epochs=4, budget=budget)
    reference, resumed = [torch.load(tmp_path / folder / "selected.pt", weights_only=True) for folder in ("reference", "resumed")]
    for key in reference["model"]:
        assert torch.equal(reference["model"][key], resumed["model"][key])
    (tmp_path / "resumed/selected.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="weights changed"):
        pair.fit_head(train, *args, config, tmp_path / "resumed", {}, fixed_epochs=4, budget=budget)


def test_head_export_parity_and_offline_batching(tmp_path):
    train = dataset()
    config = {**pair.HEADS[1], "hidden": 8}
    args = train.labels(hashes(train))
    pair.fit_head(train, *args, config, tmp_path, {}, fixed_epochs=2)
    encoder, report = pair.export_head(tmp_path / "selected.pt", tmp_path, train.features, {})
    state = torch.load(tmp_path / "selected.pt", weights_only=True)
    model = pair.PairHead(state["dimension"], config["hidden"]).eval()
    model.load_state_dict(state["model"])
    expected = 1/(1+np.exp(-pair.torch_logits(model, train.features)))
    np.testing.assert_allclose(encoder.probabilities(train.features), expected, atol=1e-6)
    np.testing.assert_allclose(encoder.probabilities(train.features[:1]), expected[:1], atol=1e-6)
    assert report["max_absolute_error"] < 1e-4 and report["bytes"] < 1_000_000


def test_embedding_cache_resumes_and_binds_rows_and_encoder(monkeypatch, tmp_path):
    from PIL import Image
    monkeypatch.setattr(pair, "load_crop", lambda *_: Image.new("RGB", (10, 10)))
    class Encoder:
        model_sha256 = "control-model"
        calls = 0
        def encode_batch(self, tensors):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interruption")
            return normalize(np.ones((len(tensors), 512), np.float32))
    encoder, rows = Encoder(), [{"image_id": str(i)} for i in range(3)]
    path = tmp_path / "cache.npz"
    with pytest.raises(RuntimeError, match="interruption"):
        pair.encode_cached(encoder, rows, path, {}, batch_size=1)
    assert len(pair.encode_cached(encoder, rows, path, {}, batch_size=1)) == 3
    assert encoder.calls == 4
    pair.encode_cached(encoder, rows, path, {})
    assert encoder.calls == 4
    encoder.model_sha256 = "different-model"
    with pytest.raises(ValueError, match="Stale"):
        pair.encode_cached(encoder, rows, path, {})


def test_calibration_frozen_and_no_training_during_outer_evaluation(monkeypatch):
    data = dataset()
    tuned = experiment.tune_calibration(data, DummyHead(), .3)
    frozen = {**tuned, "baseline": {**experiment.control.ACTIVE, "pool": 50, "support_weight": .2}, "baseline_threshold": .3}
    before = copy.deepcopy(frozen)
    q, g, e = example(6, "outer_", 200)
    monkeypatch.setattr(pair, "fit_head", lambda *_: pytest.fail("No training during evaluation"))
    results = experiment.evaluate_frozen(q, g, e, DummyHead(), frozen)
    assert before == frozen and not tuned["validation_used"] and not tuned["head_weights_updated"]
    assert len(results) == 5
    assert results["combined"]["metrics"]["mAP_at_10"] == results["ranking_only"]["metrics"]["mAP_at_10"]


def test_control_provenance_rejects_mvp_train_ids():
    if not experiment.INNER_WEIGHTS.exists():
        pytest.skip("Local control checkpoint unavailable")
    checkpoint = torch.load(experiment.INNER_WEIGHTS, weights_only=True, map_location="cpu")
    protocol = experiment.load_json(experiment.INNER_SOURCE / "results/protocol.json")
    summary = experiment.load_json(experiment.INNER_SOURCE / "results/osnet_control/training_summary.json")
    split = experiment.load_json(experiment.ARTIFACTS / "splits.json")
    rows = experiment.read_rows(experiment.DATASET / "train.csv")
    experiment.verify_control(checkpoint, protocol, summary, rows, split["frame_sha256"], split)
    checkpoint["signature"]["train_ids"] = split["identities"]["train"]
    with pytest.raises(ValueError, match="provenance"):
        experiment.verify_control(checkpoint, protocol, summary, rows, split["frame_sha256"], split)


def test_full_run_freezes_before_validation_and_second_run_only_reads(monkeypatch, tmp_path):
    names = ("inner_train", "inner_validation", "development", "calibration", "validation")
    datasets = {name: example(i, name, i*100) for i, name in enumerate(names)}
    protocols = {k: v[:2] for k, v in datasets.items()}
    vectors = {key: value for _, _, e in datasets.values() for key, value in e.items()}
    frame_hashes = {i: i for i in vectors}
    baseline = {"threshold": .3, **{k: metrics(rerank_protocol(*datasets[k])[0], .3) for k in ("calibration", "validation")}}
    signature = {"protected_sha256": {}, "baseline": {**experiment.control.ACTIVE, "pool": 50, "support_weight": .2}}
    inner_raw = pair.PairSet(*datasets["inner_validation"], signature["baseline"]).raw.ranking["mAP@10"]
    monkeypatch.setattr(experiment, "prepare", lambda: (protocols, vectors, baseline, signature, frame_hashes,
                                                       {"best_mAP_at_10": inner_raw}, None, 0.))
    monkeypatch.setattr(experiment, "export_control", lambda *_: (None, {}))
    monkeypatch.setattr(experiment, "Encoder", lambda: None)
    monkeypatch.setattr(experiment, "encode_cached", lambda _e, rows, *_: {r["image_id"]: vectors[r["image_id"]] for r in rows})
    monkeypatch.setattr(pair, "BUDGET", dict(max_epochs=3, min_epochs=1, patience=2, query_batch=4))
    monkeypatch.setattr(experiment, "mask_check", lambda *_: {})
    monkeypatch.setattr(experiment, "benchmark", lambda *_: {})
    monkeypatch.setattr(experiment, "report_markdown", lambda *_: "Synthetic report")
    original_sha = experiment.sha256
    def checksum(path):
        return Path(path).stem if Path(path).parent.name == "images" else original_sha(path)
    from pathlib import Path
    monkeypatch.setattr(experiment, "sha256", checksum)
    actual_evaluate = experiment.evaluate_frozen
    def checked(q, *args):
        assert (tmp_path / "inner_selection.json").is_file()
        assert (tmp_path / "frozen_selection.json").is_file()
        assert q == datasets["validation"][0]
        return actual_evaluate(q, *args)
    monkeypatch.setattr(experiment, "evaluate_frozen", checked)
    report = experiment.run(tmp_path)
    assert report["inner_baseline_reproduced"] and report["protected_unchanged"]
    def forbidden(*_):
        pytest.fail("Completed experiment must not retrain or retune")
    monkeypatch.setattr(experiment, "fit_head", forbidden)
    monkeypatch.setattr(experiment, "tune_calibration", forbidden)
    monkeypatch.setattr(experiment, "evaluate_frozen", forbidden)
    assert experiment.run(tmp_path) == report
    (tmp_path / "validation/combined/candidates.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="output changed"):
        experiment.run(tmp_path)


def test_notebook_is_clean_and_does_not_install_dependencies():
    import nbformat
    notebook = nbformat.read(experiment.EXPERIMENT / "train_pair_reranker.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "train_pair_reranker.ipynb", "exec")
    assert "pip install" not in "\n".join(c.source for c in notebook.cells)
