"""Synthetic v32 contracts; pytest never trains on project photographs."""
import copy
from dataclasses import replace
import random

import nbformat
import numpy as np
from PIL import Image
import pytest
import torch

from training import nive_mixed as n, nive_mixed_data as d
from training.audit import digest
from training.hpo import ExperimentConfig
from training.osnet_ablations import Ablation


@pytest.fixture(autouse=True)
def threads(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


class TinyParent(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        self.bnneck = torch.nn.BatchNorm1d(4)
        self.classifier = torch.nn.Linear(4, classes, bias=False)


class TinyDataset:
    def __init__(self, rows, *args, **kwargs):
        self.rows = rows

    def __getitem__(self, index):
        row = self.rows[index]
        clean = torch.tensor([row["label"], index/10, int(row["camera_id"]), 1.], dtype=torch.float32)
        robust = clean + torch.rand(4)*.1 + random.random()*.1 + np.random.random()*.1
        return clean, robust, row["label"], row["image_id"]


def synthetic_rows(prefix, count):
    return [{"image_id": f"{prefix}/{i}/{j}", "vehicle_id": f"{prefix}:{i}", "camera_id": j%2,
             "label": i, "path": f"train/{i}/N_{j}.jpg"} for i in range(count) for j in range(4)]


@pytest.fixture
def context(monkeypatch, tmp_path):
    monkeypatch.setattr(n.review, "initialize", lambda classes, config, variant, device: TinyParent(classes).to(device))
    monkeypatch.setattr(n, "AblationDataset", TinyDataset)
    monkeypatch.setattr(d, "NiVeDataset", TinyDataset)
    n.set_seed(8)
    parent = tmp_path / "parent.pt"
    torch.save({"model": TinyParent(4).state_dict()}, parent)
    target, external = synthetic_rows("organizer", 4), synthetic_rows("nive", 3)
    config = replace(ExperimentConfig(), identities_per_batch=2, images_per_identity=2, num_workers=0, seed=8)
    main = list(n.StepPKBatchSampler(target, config, 6))
    ids = [[target[i]["image_id"] for i in b] for b in main]
    return {"output": tmp_path / "run", "signature": "synthetic", "target": target, "external": external,
            "device": torch.device("cpu"), "config": config, "variant": Ablation(size=256, p=2),
            "dataset": tmp_path, "nive_root": tmp_path, "main_schedule": main,
            "aux_schedules": {a: d.cyclic_schedule(target if a == n.ARMS[0] else external, 4, 2, 2, 91,
                               forbidden=ids if a == n.ARMS[0] else None) for a in n.ARMS},
            "manifest": {"parent": {"path": str(parent), "sha256": n.sha256(parent)}},
            "plan": {"seed": 8, "joint_updates": 4, "tail_updates": 2, "save_interval": 2,
                     "warmup_updates": 1, "aux_weight": .25, "checkpoints": [0, 2, 4, 6]}}


def equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): equal(a, b)
    else:
        assert left == right


def test_frozen_recipe_budget_and_lr():
    p = n.old.load_json(n.VARIANT / "configs/pilot_v1.json")
    assert (p["joint_updates"], p["tail_updates"], p["aux_weight"]) == (1600, 200, .25)
    assert p["checkpoints"] == [0, 400, 800, 1200, 1600, 1800]
    assert p["main_batch"] == p["aux_batch"] == [16, 2]
    assert 1600*64+200*32 == 108800
    assert 1800*32 == 57600 and 1600*32 == 51200
    assert p["graph"] == {"k1": 20, "k2": 3, "lambda": .75}
    assert p["automatic_promotion"] is False and p["wall_time_limit"] is None


def test_namespaces_and_only_authorized_target_ids():
    organizer = [{"vehicle_id": i, "image_id": str(i)} for i in (1, 2, 3)]
    external = [{"vehicle_id": "nive:1", "image_id": "nive/1/a", "path": "train/1/N_a.jpg"}]
    a, b = d.label_domains(organizer, external, {1, 2})
    assert {r["vehicle_id"] for r in a} == {"organizer:1", "organizer:2"}
    assert {r["vehicle_id"] for r in a}.isdisjoint(r["vehicle_id"] for r in b)
    assert a[0]["label"] == b[0]["label"] == 0


@pytest.mark.parametrize("path", ["test/query/1/N_a.jpg", "train_MK_PURE/1/N_a.jpg", "train/../1/N_a.jpg",
                                   "/train/1/N_a.jpg", "train/1/N_a.txt"])
def test_no_external_test_masks_or_traversal(path):
    with pytest.raises(ValueError):
        d.label_domains([{"vehicle_id": 1}], [{"path": path, "vehicle_id": "nive:1"}], {1})


def test_auxiliary_cycles_and_prefers_groups():
    rows = synthetic_rows("nive", 3)
    a = d.cyclic_schedule(rows, 12, 2, 2, 21)
    assert a == d.cyclic_schedule(rows, 12, 2, 2, 21)
    assert d.coverage(rows, a)["image_fraction"] == 1
    for batch in a:
        assert len(batch) == len(set(batch)) == 4
        assert len({rows[i]["vehicle_id"] for i in batch}) == 2
        for x, y in zip(batch[::2], batch[1::2]):
            assert rows[x]["camera_id"] != rows[y]["camera_id"]


def test_main_aux_disjoint_and_same_main_augmentations(context):
    rows = context["target"]
    for main, aux in zip(context["main_schedule"], context["aux_schedules"][n.ARMS[0]]):
        assert set(main).isdisjoint(aux)
    a = n.load_batch(TinyDataset(rows), context["main_schedule"][0], "cpu", 77)
    n.load_batch(TinyDataset(context["external"]), [0, 1, 2, 3], "cpu", 888)
    b = n.load_batch(TinyDataset(rows), context["main_schedule"][0], "cpu", 77)
    equal(a, b)


def test_dhash_suspects_are_not_identity_proof():
    image = Image.fromarray(np.random.default_rng(21).integers(0, 256, (48, 48, 3), dtype=np.uint8))
    original = d.describe_image(image)
    h = int(original["dhash"], 16)
    left = {"train/1/a.jpg": original}
    right = {"same:crop": {"dhash": f"{h ^ 7:016x}"}, "other": {"dhash": f"{h ^ 255:016x}"}}
    assert d.near_pairs(left, right) == [{"external": "train/1/a.jpg", "reference": "same:crop", "hamming": 3}]
    assert d.near_pairs(left, left, same_set=True) == []


def test_near_review_blocks_until_resolved_and_only_excludes_external():
    rows = [{"path": "train/1/a.jpg"}, {"path": "train/2/b.jpg"}]
    audit = {"cross_domain_suspects": [{"external": rows[0]["path"], "reference": "protected:crop", "hamming": 1}]}
    with pytest.raises(ValueError, match="manual review"):
        d.reviewed_external(rows, audit, None)
    review = {"suspects_sha256": digest(audit["cross_domain_suspects"]), "decisions": {rows[0]["path"]: "confirmed_duplicate"}}
    selected, excluded = d.reviewed_external(rows, audit, review)
    assert selected == [rows[1]] and excluded[0]["path"] == rows[0]["path"] and len(rows) == 2
    review["decisions"][rows[0]["path"]] = "not_duplicate"
    assert d.reviewed_external(rows, audit, review) == (rows, [])


def test_parent_provenance_and_holdout_rejection(monkeypatch, tmp_path):
    monkeypatch.setattr(n, "ROOT", tmp_path)
    rows = [{"image_id": str(i), "vehicle_id": i} for i in range(6)]
    manifest = {"inner": {"primary": {"train": [0, 1], "validation": [2, 3]}}}
    part = tmp_path / "parent/primary/R1_resolution256/seed_20260915"
    part.mkdir(parents=True)
    n.write_json(tmp_path / "parent/manifest.json", manifest)
    signature = digest({"context": digest(manifest), "variant": "R1_resolution256", "seed": 20260915,
                        "fold": "primary", "stop": 1700, "rows": [{**rows[i], "label": i} for i in (0, 1)]})
    torch.save({"signature": signature, "step": 800, "model": {}}, part / "step_00800.pt")
    summary = {"context_signature": digest(manifest), "signature": signature, "stop_step": 1700,
               "variant": "R1_resolution256", "seed": 20260915, "fold": "primary", "train_identities": 2,
               "train_images": 2, "checkpoints": {"800": {"path": "primary/R1_resolution256/seed_20260915/step_00800.pt",
                                                         "sha256": n.sha256(part / "step_00800.pt")}}}
    n.write_json(part / "summary.json", summary)
    plan = {"parent_directory": "parent", "seed": 20260915, "parent_step": 800,
            "parent_manifest_sha256": n.sha256(tmp_path / "parent/manifest.json"),
            "parent_summary_sha256": n.sha256(part / "summary.json"), "parent_checkpoint_sha256": n.sha256(part / "step_00800.pt")}
    split = {"identities": {"train": [0, 1, 2, 3], "calibration": [4], "validation": [5]}}
    assert n.parent_provenance(plan, rows, split)[2] == {0, 1}
    summary["fold"] = "final"; n.write_json(part / "summary.json", summary)
    plan["parent_summary_sha256"] = n.sha256(part / "summary.json")
    with pytest.raises(ValueError, match="fold-matched"):
        n.parent_provenance(plan, rows, split)


def test_two_heads_correct_gradients_and_loss_weight(context):
    model = n.new_model(context, n.ARMS[1])
    main = n.load_batch(TinyDataset(context["target"]), [0, 1, 4, 5], "cpu", 4)
    aux = n.load_batch(TinyDataset(context["external"]), [0, 1, 8, 9], "cpu", 5)
    before = copy.deepcopy(model)
    optimizer = n.optimizer_for(model, context["config"])
    values = n.update(model, optimizer, main, aux, context["config"], .25, True)
    expected_opt = n.optimizer_for(before, context["config"])
    expected_opt.zero_grad()
    a = n.domain_loss(before, aux, context["config"], "aux")
    (.25*a["loss"]).backward()
    m = n.domain_loss(before, main, context["config"], "main")
    m["loss"].backward(); expected_opt.step()
    equal(model.state_dict(), before.state_dict())
    assert values["total_loss"] == values["main_loss"]+.25*values["aux_loss"]
    assert values["weighted_aux_backbone_grad_norm"] > 0 and values["main_backbone_grad_norm"] > 0
    assert model.main_head.weight.grad is not None and model.aux_head.weight.grad is not None
    assert all(int(s["step"]) == 1 for s in optimizer.state.values())


def test_bn_aux_then_main_clean_eval_and_tail(context):
    model = n.new_model(context, n.ARMS[1])
    main = n.load_batch(TinyDataset(context["target"]), [0, 1, 4, 5], "cpu", 4)
    optimizer = n.optimizer_for(model, context["config"])
    initial = n.bn_state(model)
    n.update(model, optimizer, main, main, context["config"], .25)
    after_joint = n.bn_state(model)
    n.update(model, optimizer, main, None, context["config"], .25)
    after_tail = n.bn_state(model)
    for key in initial:
        if key.endswith("num_batches_tracked"):
            assert after_joint[key]-initial[key] == 2
            assert after_tail[key]-after_joint[key] == 1
    assert model.aux_head.weight.grad is None


@pytest.mark.parametrize("arm", n.ARMS)
def test_resume_optimizer_bn_and_checkpoint_exact(context, arm):
    before_parent = n.sha256(context["manifest"]["parent"]["path"])
    continuous = {**context, "output": context["output"] / "continuous"}
    resumed = {**context, "output": context["output"] / "resumed"}
    full = n.train_arm(continuous, arm)
    assert n.train_arm(resumed, arm, stop_after=2) == {"status": "paused", "step": 2}
    actual = n.train_arm(resumed, arm)
    a, b = (torch.load(n.resume_path(c["output"] / "training" / arm), weights_only=True) for c in (continuous, resumed))
    for key in ("model", "optimizer", "rng", "history", "initial_bn"):
        equal(a[key], b[key])
    assert actual["updates"] == 6 and actual["logical_images"] == 40 and actual["encoder_image_forwards"] == 80
    assert set(full["checkpoints"]) == set(actual["checkpoints"]) == {"0", "2", "4", "6"}
    assert before_parent == n.sha256(context["manifest"]["parent"]["path"])
    assert n.train_arm(resumed, arm) == actual


def test_resume_rejects_corrupted_checkpoint(context):
    n.train_arm(context, n.ARMS[0], stop_after=2)
    path = n.resume_path(context["output"] / "training" / n.ARMS[0])
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Protected file"):
        n.train_arm(context, n.ARMS[0])


def test_inference_export_owns_no_classifier_or_metadata(context):
    model = n.new_model(context, n.ARMS[1]).eval()
    encoder = n.RetrievalEncoder(model).eval()
    assert set(encoder.state_dict())
    assert all(k.startswith(("backbone.", "bnneck.")) for k in encoder.state_dict())
    images = torch.randn(5, 4)
    with torch.no_grad():
        expected = torch.nn.functional.normalize(model.embedding(images), dim=1)
        actual = encoder(images)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    torch.testing.assert_close(actual.norm(dim=1), torch.ones(5))


def test_selects_concrete_checkpoint_with_earlier_tie():
    values = {a: {"0": {"mean_map": .8}, "400": {"mean_map": .81}, "800": {"mean_map": .81}} for a in n.ARMS}
    assert n.select_steps(values) == {a: "400" for a in n.ARMS}


def test_protection_rejects_active_mvp_change(tmp_path):
    p = tmp_path / "release_decision.json"
    p.write_text('{"active_profile":"MVP_fusion_v25"}')
    files = {str(p): n.sha256(p)}
    n.verify_files(files)
    p.write_text('{"active_profile":"bad"}')
    with pytest.raises(ValueError, match="Protected file"):
        n.verify_files(files)


def test_notebook_run_all_order_and_no_automatic_release():
    path = n.VARIANT / "train_nive_mixed.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert text.index("ORT_DISABLE_TELEMETRY") < text.index("from training")
    assert text.index("pytest") < text.index("mixed.prepare") < text.index("technical_smoke") < text.index("run_experiment")
    assert "promote" not in text and "confirm_nive_transfer" not in text
    for cell in nb.cells:
        if cell.cell_type == "code": compile(cell.source, str(path), "exec")


def evaluation_context():
    q = [{"image_id": f"q{i}", "vehicle_id": i, "camera_id": 0} for i in range(4)]
    g = [{"image_id": f"g{i}_{j}", "vehicle_id": i, "camera_id": 1} for i in range(3) for j in range(5)]
    protocol = {"query_ids": [r["image_id"] for r in q], "gallery_ids": [r["image_id"] for r in g], "selection_eligible": True}
    return {"rows": q+g, "manifest": {"inner": {"train": [20], "validation": [0, 1, 2, 3]},
                                       "draws": {"regular_1": protocol}}}


def test_only_inner_rows_evaluated_and_official_metrics_match():
    c = evaluation_context()
    rows = n.development_rows(c)
    vectors = n.normalize(np.random.default_rng(4).normal(size=(len(rows), 8)).astype(np.float32))
    result = n.score_features(c, vectors, rows)
    protocol = c["manifest"]["draws"]["regular_1"]
    by_id = {r["image_id"]: r for r in rows}
    q, g = ([by_id[i] for i in protocol[k]] for k in ("query_ids", "gallery_ids"))
    qf, gf = n.policy.frames(q, g)
    entries = result["draws"]["regular_1"]["per_query"]
    for kind in ("raw", "ranking"):
        value = n.official.ranking_metrics(qf, gf, {i: v[kind]["top10"] for i, v in entries.items()})
        assert value["mAP@10"] == result["draws"]["regular_1"][kind]["mAP@10"]
    assert entries["q3"]["ranking"]["ap"] is None
    c["manifest"]["inner"]["validation"] = [1, 2, 3]
    with pytest.raises(ValueError, match="outside"):
        n.development_rows(c)


def test_two_model_concat_is_equal_cosine_mix():
    rng = np.random.default_rng(8)
    a = n.normalize(rng.normal(size=(12, 4)).astype(np.float32))
    b = n.normalize(rng.normal(size=(12, 4)).astype(np.float32))
    combined = n.normalize(np.concatenate([a, b], axis=1))
    def cosine(v):
        # Compare actual cosine, not float32 self-dot products that are only approximately one.
        v = v.astype(np.float64)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return v @ v.T
    np.testing.assert_allclose(cosine(combined), .5*cosine(a)+.5*cosine(b), atol=2e-7)


def test_full_synthetic_run_resume_and_corrupt_cache(context, monkeypatch):
    e = evaluation_context()
    context["rows"] = e["rows"]
    context["manifest"].update(e["manifest"])
    monkeypatch.setattr(n, "check_inputs", lambda c: None)
    def encode(model, rows, *args, **kwargs):
        model.eval()
        x = torch.tensor([[r["vehicle_id"], r["camera_id"], i/10, 1.] for i, r in enumerate(rows)])
        with torch.no_grad():
            features = n.normalize(model.embedding(x).numpy())
        return {r["image_id"]: v for r, v in zip(rows, features)}
    monkeypatch.setattr(n.old, "encode", encode)
    monkeypatch.setattr(n, "export_encoder", lambda c, arm, ckpt: {"arm": arm, "checkpoint": ckpt})
    result = n.run_experiment(context)
    assert result["status"] == "complete" and result["protected_unchanged"]
    assert not result["original_validation_evaluated"] and not result["threshold_fit"] and not result["promoted"]
    assert all(set(result["evaluations"][a]) == {"0", "2", "4", "6"} for a in n.ARMS)
    assert (context["output"] / "REPORT.md").is_file()
    assert n.run_experiment(context) == result
    (context["output"] / "evaluation/N_ref/metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Protected file"):
        n.run_experiment(context)


def test_schedule_coverage_requires_enough_photos():
    with pytest.raises(ValueError, match="two unique"):
        d.cyclic_schedule([{"image_id": "a", "vehicle_id": "n:1", "camera_id": 1}], 1, 1, 2, 7)


def test_interrupted_pointer_commit_keeps_previous_state(context, monkeypatch):
    arm = n.ARMS[0]
    directory = context["output"] / "training" / arm
    writer = n.write_json
    def interrupt(path, value):
        if path.name == "resume.json" and value["path"] == "resume_1.pt":
            raise OSError("simulated interruption after inactive slot was written")
        return writer(path, value)
    with monkeypatch.context() as m:
        m.setattr(n, "write_json", interrupt)
        with pytest.raises(OSError, match="simulated interruption"):
            n.train_arm(context, arm)
    previous = torch.load(n.resume_path(directory), weights_only=True)
    assert previous["step"] == 0
    resumed = n.train_arm(context, arm)
    assert resumed["updates"] == 6
    assert len(list(directory.glob("resume_*.pt"))) == 2


def test_bad_resume_pointer_is_rejected(context):
    arm = n.ARMS[0]
    n.train_arm(context, arm, stop_after=2)
    directory = context["output"] / "training" / arm
    n.write_json(directory / "resume.json", {"path": "../../parent.pt", "sha256": "anything"})
    with pytest.raises(ValueError, match="escapes"):
        n.train_arm(context, arm)
