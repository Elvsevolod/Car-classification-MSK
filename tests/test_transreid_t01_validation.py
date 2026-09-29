import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import copy
import json
import socket
import urllib.request

import numpy as np
import pytest
import torch

from backend.core import normalize
from training import transreid_t01_validation as experiment


def arrays(count=47):
    rng = np.random.default_rng(20260929)
    blocks = [normalize(rng.normal(size=(count, 512)).astype(np.float32)) for _ in range(4)]
    original = experiment.dual.pack(blocks[0], experiment.dual.policy.combine_members(blocks[1:]))
    member = normalize(rng.normal(size=(count, 384)).astype(np.float32))
    return original, member


def rows():
    query = [{"image_id": f"q{i}", "vehicle_id": i if i < 4 else 90+i, "camera_id": 1,
              "x": 0, "y": 0, "w": 10, "h": 10} for i in range(6)]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i % 8, "camera_id": 2,
                "x": 0, "y": 0, "w": 10, "h": 10} for i in range(17)]
    return query, gallery


def test_pack_and_control_are_exact():
    original, member = arrays()
    bank = experiment.pack(original, member)
    assert bank.dtype == np.float32 and bank.shape == (47, 2432)
    np.testing.assert_array_equal(bank[:, :2048], original)
    np.testing.assert_array_equal(bank[:, 2048:], member)
    baseline = experiment.v25.rank(original[:4], original[4:])
    actual = experiment.rank(original, 4, experiment.CONTROL)
    for key in ("order", "raw_order", "confidence"):
        np.testing.assert_array_equal(actual[key], baseline[key])


def test_exact_v41_pre_graph_arithmetic_and_unchanged_candidates():
    original, member = arrays()
    actual = experiment.rank(experiment.pack(original, member), 4, experiment.CANDIDATE)
    mixed = experiment.scoring.mix([experiment.scoring.control_vectors(original), member], [.9, .1])
    expected = experiment.dual.policy.rank_vectors(mixed[:4], mixed[4:], "legacy")
    baseline = experiment.rank(original, 4, experiment.CONTROL)
    np.testing.assert_array_equal(actual["order"], expected["order"])
    assert not np.array_equal(actual["order"], baseline["order"])
    for key in ("raw_order", "confidence"):
        np.testing.assert_array_equal(actual[key], baseline[key])


@pytest.mark.parametrize("bad", ["nan", "zero", "dtype", "dimension", "count"])
def test_invalid_real_features_are_rejected(bad):
    original, member = arrays()
    if bad == "nan": member[0, 0] = np.nan
    if bad == "zero": member[0] = 0
    if bad == "dtype": member = member.astype(np.float64)
    if bad == "dimension": member = member[:, :383]
    if bad == "count": member = member[:1]
    with pytest.raises(ValueError): experiment.pack(original, member)


def test_query_independence_permutation_removal_and_batches():
    bank = experiment.pack(*arrays(62))
    query, gallery = bank[:33], bank[33:]
    expected = experiment.rank(bank, len(query), experiment.CANDIDATE)
    cases = [(np.arange(33)[::-1], 33), (np.array([7]), 1)]
    cases += [(np.arange(33), size) for size in (1, 8, 16, 32)]
    for indices, batch in cases:
        reports = [experiment.rank(np.concatenate([query[indices[start:start+batch]], gallery]),
                    len(indices[start:start+batch]), experiment.CANDIDATE) for start in range(0, len(indices), batch)]
        for key in ("order", "raw_order"):
            np.testing.assert_array_equal(np.concatenate([r[key] for r in reports]), expected[key][indices])
        confidence = np.concatenate([r["confidence"] for r in reports])
        np.testing.assert_allclose(confidence, expected["confidence"][indices], rtol=0, atol=2e-7)
        np.testing.assert_array_equal(confidence >= experiment.THRESHOLD, expected["confidence"][indices] >= experiment.THRESHOLD)


def test_export_real_vectors_preserves_acceptance_and_refusal(tmp_path):
    query, gallery = rows(); original, member = arrays(len(query)+len(gallery))
    original[0, 512:] = original[len(query), 512:]
    left = experiment.export_arrays(tmp_path/"control", query, gallery, original, experiment.CONTROL)
    bank = experiment.pack(original, member)
    right = experiment.export_arrays(tmp_path/"candidate", query, gallery, bank, experiment.CANDIDATE)
    assert (tmp_path/"control/candidates.csv").read_bytes() == (tmp_path/"candidate/candidates.csv").read_bytes()
    accepted = experiment.dual.policy.official.load_candidates(tmp_path/"candidate/candidates.csv")
    assert accepted["q0"][0][0] == "g0" and "q1" not in accepted
    lines = [x.split(",") for x in (tmp_path/"candidate/submission.csv").read_text().splitlines()]
    assert len(lines) == len(query) and all(len(x) == 11 and len(set(x[1:])) == 10 for x in lines)
    np.testing.assert_array_equal(np.load(tmp_path/"candidate/embeddings.npy"), bank)
    order = experiment.io.read(tmp_path/"candidate/embedding_order.json")
    assert order["ids"] == [r["image_id"] for r in query+gallery]
    assert order["layout"]["dimension"] == 2432 and order["ranking"] == experiment.POLICY
    result = experiment.checks.paired_comparison(left, right)
    assert result["candidates_unchanged"] and sum(result[k] for k in ("better", "worse", "equal")) == 4
    changed = copy.deepcopy(right); changed["per_query"]["q0"]["confidence"] += .1
    with pytest.raises(ValueError, match="candidate decisions"):
        experiment.checks.paired_comparison(left, changed)
    with pytest.raises(ValueError, match="new output"):
        experiment.export_arrays(tmp_path/"candidate", query, gallery, bank, experiment.CANDIDATE)


def test_bad_weights_rejected_before_loading(tmp_path):
    path = tmp_path/"checkpoint.pt"; path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint changed"):
        experiment.load_model({"checkpoint": path})


def test_source_probe_does_not_relax_tolerance(tmp_path, monkeypatch):
    _, features = arrays(32)
    np.save(tmp_path/"features.npy", features)
    c = {"stage": tmp_path, "dataset": tmp_path, "probe": [], "device": "cpu"}
    monkeypatch.setattr(experiment.models, "encode", lambda *a, **kw: features.copy())
    assert experiment.probe(c, None)["max_error"] == 0
    monkeypatch.setattr(experiment.models, "encode", lambda *a, **kw: features+.001)
    with pytest.raises(ValueError, match="do not increase tolerance"):
        experiment.probe(c, None)


def test_run_all_offline_inference_only_and_resumable(tmp_path, monkeypatch):
    query, gallery = rows(); original, member = arrays(len(query)+len(gallery))
    reference = tmp_path/"reference"
    experiment.export_arrays(reference/"export", query, gallery, original, experiment.CONTROL)
    protected = {str(p): experiment.io.sha(p) for p in reference.rglob("*") if p.is_file()}
    c = {"output": tmp_path/"runs/test", "signature": "synthetic", "device": torch.device("cpu"),
         "query": query, "gallery": gallery, "original": original, "reference": reference,
         "dataset": tmp_path, "profile": None, "manifest": {"protected": protected, "source_sha256": {}}}
    monkeypatch.setattr(experiment, "VARIANT", tmp_path)
    monkeypatch.setattr(experiment, "prepare", lambda *_: c)
    monkeypatch.setattr(experiment, "load_model", lambda _: torch.nn.Identity().eval())
    monkeypatch.setattr(experiment, "probe", lambda *_: {"status": "synthetic"})
    calls = []
    def encode(*a, **kw): calls.append("T01"); return member.copy()
    class Encoder:
        def __init__(self, _): pass
        def encode_rows(self, *a): calls.append("v25"); return original.copy()
    monkeypatch.setattr(experiment.models, "encode", encode)
    monkeypatch.setattr(experiment.dual, "DualRoleEncoder", Encoder)
    def forbidden(*a, **kw): raise AssertionError("No training, calibration or network")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(experiment.dual.policy, "calibrate_policy", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    first = experiment.run("test", "cpu")
    assert first["status"] == "complete" and first["optimizer_updates"] == first["bn_updates"] == 0
    assert not first["promoted"] and not first["threshold_fit"] and calls == ["T01", "v25"]
    assert experiment.run("test", "cpu") == first and calls == ["T01", "v25"]
    experiment.checks.verify_files(protected)


def test_notebook_and_config_are_fixed_without_search():
    settings = experiment.io.read(experiment.VARIANT/"config.json")
    assert settings["checkpoint_sha256"] == experiment.CHECKPOINT_SHA256
    assert settings["policy"] == experiment.POLICY and settings["threshold"] == experiment.THRESHOLD
    notebook = experiment.io.read(experiment.VARIANT/"validate_t01_vs_v25.ipynb")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code": compile("".join(cell["source"]), "notebook", "exec")
    text = json.dumps(notebook)
    assert "DEVICE = 'mps'" in text and "timeout=" not in text and "pip install" not in text
