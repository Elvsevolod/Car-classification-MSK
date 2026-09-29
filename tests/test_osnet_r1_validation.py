import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import copy
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import torch

from training import osnet_r1_validation as experiment
from training import research_io as io


def arrays(count=47):
    rng = np.random.default_rng(20260929)
    blocks = [experiment.normalize(rng.normal(size=(count, 512)).astype(np.float32)) for _ in range(5)]
    original = experiment.dual.pack(blocks[0], experiment.dual.policy.combine_members(blocks[1:4]))
    return original, blocks[4]


def rows():
    query = [{"image_id": f"q{i}", "vehicle_id": i if i < 4 else 90+i, "camera_id": 1,
              "x": 0, "y": 0, "w": 10, "h": 10} for i in range(6)]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i % 8, "camera_id": 2,
                "x": 0, "y": 0, "w": 10, "h": 10} for i in range(17)]
    return query, gallery


def test_control_is_exact_v25_and_pack_is_lossless():
    original, member = arrays()
    bank = experiment.pack(original, member)
    assert bank.dtype == np.float32 and bank.shape == (47, 2560)
    np.testing.assert_array_equal(bank[:, :2048], original)
    np.testing.assert_array_equal(bank[:, 2048:], member)
    reference = experiment.v25.rank(original[:4], original[4:])
    actual = experiment.rank(original, 4, experiment.CONTROL)
    for k in ("order", "raw_order", "confidence"):
        np.testing.assert_array_equal(actual[k], reference[k])


def test_replacement_changes_only_first_r1_ranking_member_and_keeps_refusals():
    original, member = arrays()
    bank = experiment.pack(original, member)
    baseline = experiment.rank(original, 4, experiment.CONTROL)
    actual = experiment.rank(bank, 4, experiment.CANDIDATE)
    mvp, r1 = experiment.dual.unpack(original)
    replaced = experiment.dual.pack(mvp, experiment.dual.policy.combine_members(
        [member, r1[:, 512:1024], r1[:, 1024:]]))
    expected = experiment.v25.rank(replaced[:4], replaced[4:])
    np.testing.assert_array_equal(actual["order"], expected["order"])
    for k in ("raw_order", "confidence"):
        np.testing.assert_array_equal(actual[k], baseline[k])
    assert not np.array_equal(actual["order"], baseline["order"])
    np.testing.assert_array_equal(actual["confidence"] >= experiment.THRESHOLD,
                                  baseline["confidence"] >= experiment.THRESHOLD)


def test_noop_member_keeps_top10():
    original, _ = arrays()
    first = experiment.normalize(original[:, 512:1024])
    control = experiment.rank(original, 4, experiment.CONTROL)
    noop = experiment.rank(experiment.pack(original, first), 4, experiment.CANDIDATE)
    for k in ("order", "raw_order", "confidence"):
        np.testing.assert_array_equal(noop[k], control[k])


@pytest.mark.parametrize("bad", ["nan", "zero", "dtype", "dim", "count"])
def test_invalid_features_are_not_replaced_with_fake_vectors(bad):
    original, member = arrays()
    if bad == "nan": member[0, 0] = np.nan
    if bad == "zero": member[0] = 0
    if bad == "dtype": member = member.astype(np.float64)
    if bad == "dim": member = member[:, :511]
    if bad == "count": member = member[:1]
    with pytest.raises(ValueError): experiment.pack(original, member)


def test_query_permutation_removal_and_batch_sizes():
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


def test_export_preserves_candidates_and_top10_even_for_refused_queries(tmp_path):
    query, gallery = rows()
    original, member = arrays(len(query)+len(gallery))
    left = experiment.export_arrays(tmp_path/"control", query, gallery, original, experiment.CONTROL)
    right = experiment.export_arrays(tmp_path/"candidate", query, gallery, experiment.pack(original, member), experiment.CANDIDATE)
    assert (tmp_path/"control/candidates.csv").read_bytes() == (tmp_path/"candidate/candidates.csv").read_bytes()
    assert len((tmp_path/"candidate/candidates.csv").read_text().splitlines()) == 1
    lines = [x.split(",") for x in (tmp_path/"candidate/submission.csv").read_text().splitlines()]
    assert len(lines) == len(query) and all(len(x) == 11 and len(set(x[1:])) == 10 for x in lines)
    values = np.load(tmp_path/"candidate/embeddings.npy")
    np.testing.assert_array_equal(values, experiment.pack(original, member))
    order = io.read(tmp_path/"candidate/embedding_order.json")
    assert order["ids"] == [r["image_id"] for r in query+gallery]
    comparison = experiment.paired_comparison(left, right)
    assert comparison["candidates_unchanged"] and sum(comparison[k] for k in ("better", "worse", "equal")) == 4
    altered = copy.deepcopy(right)
    altered["per_query"][query[0]["image_id"]]["confidence"] += .1
    with pytest.raises(ValueError, match="candidate decisions"): experiment.paired_comparison(left, altered)
    with pytest.raises(ValueError, match="new output"):
        experiment.export_arrays(tmp_path/"candidate", query, gallery, values, experiment.CANDIDATE)


def test_identity_leakage_and_wrong_protocol_are_rejected():
    query, gallery = rows()
    identities = {"train": [100, 101], "calibration": [200], "validation": list(range(8))+[94, 95]}
    protocol = {"query_ids": [r["image_id"] for r in query], "gallery_ids": [r["image_id"] for r in gallery]}
    assert experiment.validate_split(query+gallery, protocol, identities, [100], [101]) == (query, gallery)
    with pytest.raises(ValueError, match="leakage"):
        experiment.validate_split(query+gallery, protocol, identities, [100, 0], [101])
    altered = copy.deepcopy(protocol); altered["gallery_ids"][0] = altered["query_ids"][0]
    with pytest.raises(ValueError, match="overlapping"):
        experiment.validate_split(query+gallery, altered, identities, [100], [101])


def test_accepted_candidate_keeps_its_own_confidence(tmp_path):
    query, gallery = rows()
    original, member = arrays(len(query)+len(gallery))
    original[0, 512:] = original[len(query), 512:]
    left = experiment.export_arrays(tmp_path/"control", query, gallery, original, experiment.CONTROL)
    right = experiment.export_arrays(tmp_path/"candidate", query, gallery, experiment.pack(original, member), experiment.CANDIDATE)
    accepted = experiment.dual.policy.official.load_candidates(tmp_path/"candidate/candidates.csv")
    assert accepted["q0"][0][0] == "g0" and accepted["q0"][0][1] >= experiment.THRESHOLD
    assert "q1" not in accepted
    assert (tmp_path/"candidate/candidates.csv").read_bytes() == (tmp_path/"control/candidates.csv").read_bytes()
    assert experiment.paired_comparison(left, right)["candidates_unchanged"]


def test_corrupted_weight_fails_before_loading(tmp_path):
    (tmp_path/"checkpoint.pt").write_bytes(b"corrupt checkpoint")
    c = {"stage": tmp_path, "manifest": {"settings": {"checkpoint_sha256": "0"*64}}}
    with pytest.raises(ValueError, match="checkpoint changed"):
        experiment.load_model(c)


def test_jpeg_png_resolution_rejects_missing_or_ambiguous_files(tmp_path):
    from PIL import Image
    directory = tmp_path/"images"; directory.mkdir()
    Image.new("RGB", (12, 9)).save(directory/"one.jpg")
    Image.new("RGB", (12, 9)).save(directory/"two.PNG")
    assert experiment.dual.image_path(tmp_path, "one").name == "one.jpg"
    assert experiment.dual.image_path(tmp_path, "two").name == "two.PNG"
    with pytest.raises(ValueError, match="found 0"):
        experiment.dual.image_path(tmp_path, "missing")
    Image.new("RGB", (12, 9)).save(directory/"one.png")
    with pytest.raises(ValueError, match="found 2"):
        experiment.dual.image_path(tmp_path, "one")


def test_explicit_device_without_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("mps", "cuda", "auto"):
        with pytest.raises(ValueError): experiment.device_for(name)
    assert experiment.device_for("cpu").type == "cpu"


def test_atomic_task_resume_and_corruption(tmp_path):
    c = {"output": tmp_path, "signature": "test"}
    def fail(directory):
        (directory/"half.txt").write_text("incomplete")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError): experiment.task(c, "check", fail)
    assert not (tmp_path/"check").exists()
    assert experiment.task(c, "check", lambda _: {"status": "ok"}) == {"status": "ok"}
    assert experiment.task(c, "check", fail) == {"status": "ok"}
    (tmp_path/"check/result.json").write_text("{}")
    with pytest.raises(ValueError, match="Missing/changed"): experiment.task(c, "check", fail)


def test_run_all_is_inference_only_offline_and_resumable(tmp_path, monkeypatch):
    query, gallery = rows(); original, member = arrays(len(query)+len(gallery))
    reference = tmp_path/"reference"
    experiment.export_arrays(reference/"export", query, gallery, original, experiment.CONTROL)
    protected = {str(p): io.sha(p) for p in reference.rglob("*") if p.is_file()}
    c = {"output": tmp_path/"runs/test", "signature": "synthetic", "device": torch.device("cpu"),
         "query": query, "gallery": gallery, "original": original, "reference": reference, "dataset": tmp_path,
         "manifest": {"winner": {"trial": "synthetic"}, "protected": protected, "source_sha256": {}}}
    monkeypatch.setattr(experiment, "VARIANT", tmp_path)
    monkeypatch.setattr(experiment, "prepare", lambda *_: c)
    monkeypatch.setattr(experiment, "load_model", lambda _: torch.nn.Identity().eval())
    monkeypatch.setattr(experiment, "probe", lambda *_: {"status": "synthetic pass"})
    calls = []
    def encode(*a, **kw): calls.append(1); return member.copy()
    monkeypatch.setattr(experiment.models, "encode", encode)
    def forbidden(*a, **kw): raise AssertionError("No optimizer, calibration or network allowed")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(experiment.dual.policy, "calibrate_policy", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    first = experiment.run("test", "cpu")
    assert first["status"] == "complete" and first["optimizer_updates"] == first["bn_updates"] == 0
    assert not first["promoted"] and not first["threshold_fit"]
    assert calls == [1]
    second = experiment.run("test", "cpu")
    assert first == second and calls == [1]
    experiment.verify_files(protected)


def test_pinned_notebook_and_config_do_not_search_validation():
    settings = io.read(experiment.VARIANT/"config.json")
    assert settings["checkpoint_sha256"] == "0b442bb7ce7f8f922671e15d5e3f2ed343f8c3dab53a93b9111564a78e6cd20a"
    assert settings["threshold"] == experiment.THRESHOLD
    notebook = io.read(experiment.VARIANT/"validate_r03_vs_v25.ipynb")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")
    text = json.dumps(notebook)
    assert "DEVICE = 'mps'" in text and "timeout=" not in text and "pip install" not in text
