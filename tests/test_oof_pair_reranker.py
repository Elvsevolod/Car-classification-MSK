"""OOF isolation/resume tests; no real OSNet optimization or external downloads."""
import copy
from dataclasses import asdict, replace
import random

import numpy as np
import pytest
import torch

from backend.core import normalize
from backend.scoring import metrics
from training import hpo
from training import oof_pair_reranker as oof
from training import pair_reranker as pair


def example(seed=0, prefix="", identity_offset=0):
    rng = np.random.default_rng(seed)
    q = [{"image_id": f"{prefix}q{i}", "vehicle_id": identity_offset+i, "camera_id": 0,
          "x": 0, "y": 0, "w": 10, "h": 10} for i in range(8)]
    g = [{"image_id": f"{prefix}g{i}", "vehicle_id": identity_offset+i%6, "camera_id": i%3,
          "x": 0, "y": 0, "w": 10, "h": 10} for i in range(65)]
    vectors = normalize(rng.normal(size=(len(q)+len(g), 16)).astype(np.float32))
    return q, g, dict(zip((r["image_id"] for r in q+g), vectors))


def baseline_config():
    return {**oof.control.ACTIVE, "pool": 50, "support_weight": .2}


def training(seed=0, prefix="", offset=0):
    q, g, e = example(seed, prefix, offset)
    pairs = pair.PairSet(q, g, e, baseline_config())
    return pairs, oof.scalar_training(pairs, {r["image_id"]: r["image_id"] for r in q+g})


def test_folds_cover_exactly_once_and_keep_transitive_frame_groups():
    rows = [{"vehicle_id": i, "image_id": f"{i}-a"} for i in range(35)]
    rows.append({"vehicle_id": 1, "image_id": "1-b"})
    hashes = {r["image_id"]: r["image_id"] for r in rows}
    hashes.update({"0-a": "same-x", "1-a": "same-x", "1-b": "same-y", "2-a": "same-y"})
    folds = oof.make_folds(rows, list(range(30)), hashes)
    assert folds == oof.make_folds(rows[::-1], list(reversed(range(30))), hashes)
    parts = {name: f["held_out"] for name, f in folds.items()}
    oof.audit_partitions(rows[:30]+[rows[-1]], hashes, parts)
    assert any({0, 1, 2} <= set(ids) for ids in parts.values())
    assert sorted(i for ids in parts.values() for i in ids) == list(range(30))
    assert max(map(len, parts.values()))-min(map(len, parts.values())) <= 2
    for fold in folds.values():
        assert not set(fold["train"]) & set(fold["held_out"])
        assert set(fold["train"]) | set(fold["held_out"]) == set(range(30))
    with pytest.raises(ValueError, match="Not enough"):
        oof.make_folds(rows[:2], [0, 1], hashes)


class TinyModel(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.bnneck = torch.nn.Identity()
        self.classifier = torch.nn.Linear(2, classes)


def tiny_training(monkeypatch):
    rows = [{"vehicle_id": i, "camera_id": c, "image_id": f"{i}-{c}"} for i in range(3) for c in (0, 1)]
    config = hpo.ExperimentConfig(epochs=30, identities_per_batch=2, images_per_identity=2)
    seen = []

    class TinyDataset:
        def __init__(self, rows, *_args, **_kwargs):
            seen.extend(r["vehicle_id"] for r in rows)
            assert {r["vehicle_id"] for r in rows} == {0, 1}
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            x = torch.rand(2) + random.random() + np.random.random()
            return x, x*.9, self.rows[index]["label"], self.rows[index]["image_id"]

    def losses(model, clean, robust, labels, config, metric_weight):
        y = model.classifier(model.backbone(clean))
        return {"loss": torch.nn.functional.cross_entropy(y, labels)}

    def initialize(classes, config, device, onnx_path):
        assert onnx_path == oof.STOCK_MODEL and onnx_path != oof.MODEL
        return TinyModel(classes).to(device), {}

    monkeypatch.setattr(hpo, "VehicleDataset", TinyDataset)
    monkeypatch.setattr(hpo, "experiment_losses", losses)
    monkeypatch.setattr(oof, "initialize_experiment", initialize)
    return rows, config, seen


def test_fixed_fold_resume_lr_horizon_and_no_held_out_access(monkeypatch, tmp_path):
    rows, config, seen = tiny_training(monkeypatch)
    key = {"train": [0, 1], "config": asdict(config), "epochs": 3}

    def fit(name, signature=key):
        return oof.fit_fold(rows, [0, 1], config, 3, torch.device("cpu"), tmp_path / name, signature)

    expected = fit("uninterrupted")
    real_save = oof._save_checkpoint

    def interrupted(path, state):
        real_save(path, state)
        if path.name == "last.pt" and len(state["history"]) == 1:
            raise RuntimeError("simulated interruption after atomic save")

    monkeypatch.setattr(oof, "_save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        fit("resumed")
    (tmp_path / "resumed/history.json").write_text("broken secondary history")
    monkeypatch.setattr(oof, "_save_checkpoint", real_save)
    actual = fit("resumed")
    states = [torch.load(tmp_path / name / "final.pt", map_location="cpu", weights_only=True)
              for name in ("uninterrupted", "resumed")]
    assert all(torch.equal(states[0]["model"][k], states[1]["model"][k]) for k in states[0]["model"])
    assert [r["train"] for r in actual["history"]] == [r["train"] for r in expected["history"]]
    assert actual["epochs"] == 3 and not actual["held_out_used_for_selection"]
    assert actual["gradient_steps"] == 3 and set(seen) == {0, 1}
    # Third epoch starts cosine at its maximum, not at the end of a compressed 3-epoch horizon.
    assert actual["history"][-1]["lr"]["encoder"] == config.encoder_lr
    with pytest.raises(ValueError, match="Different weights"):
        fit("resumed", {**key, "epochs": 4})
    monkeypatch.setattr(oof, "initialize_experiment", lambda *_a, **_k: pytest.fail("Completed fold must not train"))
    assert fit("resumed") == actual
    (tmp_path / "resumed/final.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="weights/history changed"):
        fit("resumed")


def test_incomplete_fold_rejects_changed_signature(monkeypatch, tmp_path):
    rows, config, _ = tiny_training(monkeypatch)
    real_save = oof._save_checkpoint

    def interrupted(path, state):
        real_save(path, state)
        raise RuntimeError("stop")

    monkeypatch.setattr(oof, "_save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="stop"):
        oof.fit_fold(rows, [0, 1], config, 3, torch.device("cpu"), tmp_path, {"recipe": 1})
    monkeypatch.setattr(oof, "_save_checkpoint", real_save)
    with pytest.raises(ValueError, match="settings/data/device changed"):
        oof.fit_fold(rows, [0, 1], config, 3, torch.device("cpu"), tmp_path, {"recipe": 2})


def test_pool_only_scalars_and_train_scaling_no_embedding_axis_mixing():
    parts = [training(i, str(i), i*100)[1] for i in range(3)]
    ids = [r["vehicle_id"] for p in parts for r in p.query]
    pooled = oof.pool_training(parts, ids)
    assert pooled.features.shape == (24, 50, 8)
    np.testing.assert_allclose(pooled.weights.sum(1), 1, atol=1e-6)
    with pytest.raises(ValueError, match="exactly one"):
        oof.pool_training(parts+[parts[0]], ids)
    with pytest.raises(ValueError, match="exactly one"):
        oof.pool_training(parts, ids+[999])
    full = replace(parts[0], features=np.zeros((8, 50, 1032), dtype=np.float32))
    with pytest.raises(ValueError, match="eight scalar"):
        oof.pool_training([full, *parts[1:]], ids)


def test_fold_collection_routes_only_unseen_query_and_gallery(monkeypatch, tmp_path):
    datasets = {f"fold_{n+1:02d}": example(n, str(n), n*100) for n in range(3)}
    all_ids = [r["vehicle_id"] for q, _, _ in datasets.values() for r in q]
    rows = [r for q, g, _ in datasets.values() for r in q+g]
    vectors = {i: v for _, _, e in datasets.values() for i, v in e.items()}
    folds = {name: {"held_out": [r["vehicle_id"] for r in q],
                    "train": sorted(set(all_ids)-{r["vehicle_id"] for r in q})}
             for name, (q, _, _) in datasets.items()}
    signature = {"device": "cpu", "config": asdict(hpo.ExperimentConfig()), "folds": folds,
                 "outer": {"train": all_ids}, "encoder_epochs": 5, "baseline": baseline_config()}
    context = {"signature": signature, "rows": rows, "hashes": {i: i for i in vectors},
               "fold_protocols": {name: (q, g) for name, (q, g, _) in datasets.items()}}
    trained, extracted = [], []
    leak = False

    def fit(selected, ids, config, epochs, device, directory, key, remaining_folds):
        assert {r["vehicle_id"] for r in selected} == set(ids)
        assert not set(ids) & set(key["held_out"])
        trained.append(key["name"])
        return {"train_ids": ids + (key["held_out"][:1] if leak else [])}

    def encode(encoder, selected, path, key):
        assert {r["vehicle_id"] for r in selected} == set(key["held_out"])
        assert encoder == "mvp" or encoder == key["name"]
        extracted.append((key["name"], encoder, [r["image_id"] for r in selected]))
        return {r["image_id"]: vectors[r["image_id"]] for r in selected}

    monkeypatch.setattr(oof, "fit_fold", fit)
    monkeypatch.setattr(oof, "export_fold", lambda directory, key, q: (key["name"], {}))
    monkeypatch.setattr(oof, "Encoder", lambda: "mvp")
    monkeypatch.setattr(oof, "encode_cached", encode)
    unseen, matched, reports = oof.collect_training(context, tmp_path)
    assert len(trained) == 3 and len(extracted) == 6 and len(reports) == 3
    assert unseen.features.shape == matched.features.shape == (24, 50, 8)
    for n in range(0, 6, 2):
        assert extracted[n][2] == extracted[n+1][2]  # Identical q/g for both conditions.
    leak = True
    with pytest.raises(ValueError, match="encoder trained on"):
        oof.collect_training(context, tmp_path / "leak")


def test_scalar_heads_fixed_seven_epochs_and_onnx_parity(tmp_path):
    _, data = training()
    heads, reports = oof.train_heads(data, data, tmp_path, {"head_epochs": 7})
    for name in heads:
        assert reports[name]["training"]["selected_epoch"] == 7
        assert reports[name]["training"]["best_inner"] is None
        assert reports[name]["export"]["dimension"] == 8
        assert reports[name]["export"]["max_absolute_error"] < 1e-4
        state = torch.load(tmp_path / "heads" / name / "selected.pt", weights_only=True)
        np.testing.assert_allclose(state["model"]["mean"], data.features[data.valid].mean(0), atol=1e-7)
    np.testing.assert_array_equal(heads["oof"].probabilities(data.features), heads["matched_in_sample"].probabilities(data.features))


class DummyHead:
    def probabilities(self, features):
        return 1/(1+np.exp(-features[..., 0]))


def test_calibration_does_not_tune_refusal_and_evaluation_is_frozen(monkeypatch):
    pairs, _ = training()
    heads = {name: DummyHead() for name in ("oof", "matched_in_sample")}
    monkeypatch.setattr(oof.control, "choose_threshold", lambda *_: pytest.fail("Do not tune refusal"))
    monkeypatch.setattr(oof, "fit_head", lambda *_: pytest.fail("Do not train during calibration/validation"))
    tuned = oof.calibrate_blends(pairs, heads, .15)
    assert not tuned["refusal_retuned"] and not tuned["validation_used"] and not tuned["head_weights_updated"]
    heads["previous_head"] = DummyHead()
    frozen = {"baseline": baseline_config(), "betas": {**tuned["betas"], "previous_head": .1}, "threshold": .15}
    before = copy.deepcopy(frozen)
    results = oof.evaluate(*example(3, "outer", 300), heads, frozen)
    assert frozen == before and len(results) == 5
    decisions = [{qid: d["accepted"] for qid, d in item["per_query"].items()} for item in results.values()]
    assert all(d == decisions[0] for d in decisions)
    assert len({r["metrics"]["TNR"] for r in results.values()}) == 1
    assert "paired_vs_matched_in_sample" in results["oof"]


def test_concurrent_runs_are_rejected_and_lock_recovers(tmp_path):
    with oof.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already active"):
            with oof.run_lock(tmp_path):
                pytest.fail("Second writer entered")
    with oof.run_lock(tmp_path):
        pass


def test_run_freezes_before_validation_exports_official_csv_and_reuses_completed(monkeypatch, tmp_path):
    cal, val = example(4, "cal", 400), example(5, "val", 500)
    vectors = {**cal[2], **val[2]}
    baseline = {}
    for name, (q, g, e) in (("calibration", cal), ("validation", val)):
        ranked, _ = oof.control.ProtocolScores(q, g, e).evaluate(oof.control.ACTIVE)
        baseline[name] = metrics(ranked, .15)
    signature = {"baseline": baseline_config(), "baseline_threshold": .15, "head_epochs": 7,
                 "previous_head_sha256": "test", "previous_beta": .1, "protected_sha256": {}}
    context = {"signature": signature, "plan": {}, "protocols": {"calibration": cal[:2], "validation": val[:2]},
               "vectors": vectors, "baseline": baseline, "hashes": {}}
    monkeypatch.setattr(oof, "prepare", lambda *_: context)
    _, data = training()
    monkeypatch.setattr(oof, "collect_training", lambda *_: (data, data, {}))
    monkeypatch.setattr(oof, "HeadEncoder", lambda *_: DummyHead())
    monkeypatch.setattr(oof, "mask_check", lambda *_: {})
    monkeypatch.setattr(oof.previous, "benchmark", lambda *_: {})
    monkeypatch.setattr(oof, "report_markdown", lambda *_: "Synthetic OOF report")
    old_pairs = pair.PairSet(*val, baseline_config())
    ranked, confidence = oof.previous.score_pair_set(old_pairs, DummyHead().probabilities(old_pairs.features), .1)
    old_report = {"validation": {"ranking_only": oof.previous.evaluated(ranked, confidence["max_cosine"], .15)}}
    load = oof._load
    monkeypatch.setattr(oof, "_load", lambda path, *args: old_report if path == oof.PREVIOUS / "report.json" else load(path, *args))
    evaluate = oof.evaluate

    def checked(query, *args):
        assert query == val[0]
        frozen = load(tmp_path / "frozen_selection.json")
        assert not frozen["refusal_retuned"]
        assert (tmp_path / "heads/oof/export.json").is_file()
        return evaluate(query, *args)

    monkeypatch.setattr(oof, "evaluate", checked)
    report = oof.run(tmp_path)
    assert report["protected_unchanged"]
    np.testing.assert_array_equal(np.load(tmp_path / "validation/embeddings.npy"), np.stack([val[2][r["image_id"]] for r in val[0]+val[1]]))

    def forbidden(*_a, **_k):
        pytest.fail("Completed experiment should only be read")

    monkeypatch.setattr(oof, "collect_training", forbidden)
    monkeypatch.setattr(oof, "train_heads", forbidden)
    monkeypatch.setattr(oof, "calibrate_blends", forbidden)
    monkeypatch.setattr(oof, "evaluate", forbidden)
    assert oof.run(tmp_path) == report
    (tmp_path / "validation/oof/candidates.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="output changed"):
        oof.run(tmp_path)


def test_fold_export_parity_with_real_stock_architecture_no_training(tmp_path, monkeypatch):
    from PIL import Image
    key = {"train": [0, 1], "held_out": [2], "epochs": 5}
    config = hpo.ExperimentConfig(epochs=30)
    model, _ = hpo.initialize_experiment(2, config, torch.device("cpu"))
    model.eval()
    oof._save_checkpoint(tmp_path / "final.pt", {"model": model.state_dict(), "signature": key,
        "config": asdict(config), "epoch": 5})
    # Realistic textured input: a constant field makes InstanceNorm dominated by cancellation noise.
    crop = Image.fromarray(np.random.default_rng(17).integers(0, 256, (96, 128, 3), dtype=np.uint8))
    monkeypatch.setattr(oof, "load_crop", lambda *_: crop)
    encoder, report = oof.export_fold(tmp_path, key, [{"image_id": "a"}, {"image_id": "b"}])
    assert report["max_absolute_error"] < 1e-4 and report["training_only_encoder"]
    assert encoder.model_sha256 == report["sha256"]
    _, repeated = oof.export_fold(tmp_path, key, [])
    assert repeated == report
    (tmp_path / "encoder.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        oof.export_fold(tmp_path, key, [])


def test_notebook_is_clean_compiles_and_has_no_installer():
    import nbformat
    notebook = nbformat.read(oof.EXPERIMENT / "train_oof_pair_reranker.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "train_oof_pair_reranker.ipynb", "exec")
    text = "\n".join(c.source for c in notebook.cells)
    assert "pip install" not in text and "report = run(OUTPUT, device=DEVICE)" in text
