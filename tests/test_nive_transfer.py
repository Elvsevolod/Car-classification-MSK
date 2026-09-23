"""Small synthetic tests only: never train on organizer/NiVe photos in pytest."""
import random

import nbformat
import numpy as np
import pytest
import torch
from PIL import Image

from backend.core import sha256
from training import nive_transfer as n
from training.hpo import ExperimentConfig
from training.osnet_ablations import Ablation
from training.pipeline import set_seed


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def photos(monkeypatch, tmp_path):
    rng = np.random.default_rng(5)
    paths = ["train/1/N_01.jpg", "train/1/S_02.jpg", "train/2/N_03.jpg", "train/2/S_04.jpg",
             "test/query/3/N_05.jpg", "test/gallery/3/S_06.jpg"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(path)
    monkeypatch.setattr(n, "EXPECTED", {"train": (4, 2), "test/query": (1, 1), "test/gallery": (1, 1)})
    # Deliberately misplaced external-test mask: it must never define train membership.
    n.write_json(tmp_path / "train_MK_PURE/3/N_05.txt", {"not": "training data"})
    n.write_json(tmp_path / "test/gallery/3/uploading.cfg", {"ignored": True})
    return tmp_path


def test_photo_split_not_mask_split_and_namespaced_ids(photos):
    before = {str(p): sha256(p) for p in photos.rglob("*") if p.is_file()}
    rows, inventory = n.audit_nive(photos, [])
    assert len(rows) == 4 and len(inventory["files"]) == 6
    assert {r["vehicle_id"] for r in rows} == {"nive:1", "nive:2"}
    assert all(r["path"].startswith("train/") and r["image_id"].startswith("nive/") for r in rows)
    assert {r["camera_id"] for r in rows} == {"N", "S"}
    assert before == {str(p): sha256(p) for p in photos.rglob("*") if p.is_file()}
    labeled = [{**r, "label": i // 2} for i, r in enumerate(rows)]
    clean, robust, label, image_id = n.NiVeDataset(labeled, Ablation(), photos)[0]
    assert clean.shape == robust.shape == (3, 208, 208)
    assert label == 0 and image_id.startswith("nive/")


def test_audit_rejects_organizer_overlap_and_incomplete_data(photos):
    with pytest.raises(ValueError, match="organizer byte overlap"):
        n.audit_nive(photos, [sha256(photos / "train/1/N_01.jpg")])
    (photos / "test/query/3/N_05.jpg").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        n.audit_nive(photos, [])


def test_audit_rejects_cross_split_identity_and_byte_leakage(photos):
    (photos / "train/2").rename(photos / "train/3")
    with pytest.raises(ValueError, match="identity leakage"):
        n.audit_nive(photos, [])
    (photos / "test/query/3/N_05.jpg").unlink()
    (photos / "test/query/3/N_05.jpg").symlink_to(photos / "train/1/N_01.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        n.audit_nive(photos, [])


@pytest.mark.parametrize("path", ["test/query/3/N_05.jpg", "train/../../outside.jpg"])
def test_external_loader_rejects_test_or_escaping_path(photos, path):
    row = {"path": path, "label": 0, "image_id": "nive:bad"}
    with pytest.raises(ValueError):
        n.NiVeDataset([row], Ablation(), photos)[0]


class TinyModel(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.bnneck = torch.nn.BatchNorm1d(4)
        self.classifier = torch.nn.Linear(4, classes)

    def embedding(self, images):
        return self.bnneck(self.backbone(images))

    def forward(self, images):
        raw = self.backbone(images)
        embedding = self.bnneck(raw)
        return self.classifier(embedding), raw, embedding


class TinyDataset:
    def __init__(self, rows, *_args, **_kwargs):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clean = torch.tensor([row["label"], index, len(str(row["camera_id"])), 1.], dtype=torch.float)
        return clean, clean + torch.rand(4) * .1 + random.random() * .1, row["label"], row["image_id"]


@pytest.fixture
def context(monkeypatch, tmp_path):
    monkeypatch.setattr(n, "AblationDataset", TinyDataset)
    monkeypatch.setattr(n, "NiVeDataset", TinyDataset)
    monkeypatch.setattr(n, "initialize", lambda classes, config, variant, device: TinyModel(classes).to(device))
    monkeypatch.setattr(n.suite, "evaluate_inner", lambda model, *a:
                        {"mAP_at_10": float(torch.sigmoid(model.backbone.weight.detach().sum()))})
    monkeypatch.setattr(n, "verify_inputs", lambda *a, **k: None)
    monkeypatch.setattr(n, "ensure_previous_suite_idle", lambda: None)
    rows = [{"vehicle_id": i, "camera_id": c, "image_id": f"organizer/{i}/{c}"}
            for i in range(4) for c in (1, 2)]
    nive = [{"vehicle_id": f"nive:{i}", "camera_id": c, "image_id": f"nive/{i}/{c}", "path": f"train/{i}/{c}.jpg"}
            for i in range(3) for c in ("N", "S")]
    return {"output": tmp_path / "run", "signature": "synthetic", "base": ExperimentConfig(), "rows": rows,
            "nive_rows": nive, "dataset": tmp_path, "nive_root": tmp_path, "device": torch.device("cpu"),
            "masks": {}, "variants": {name: Ablation(name=name, p=2) for name in n.ARMS}, "seeds": (1, 2, 3),
            "budget": n.suite.Budget(4, 2, 1), "pretrain_budget": n.suite.Budget(2, 1, 1),
            "manifest": {"inner": {"primary": {"train": [0, 1], "validation": [2, 3]},
                                   "alternate": {"train": [2, 3], "validation": [0, 1]}},
                         "nive": {"source": {"local_copy_source_confirmed_by_user": True}}},
            "split": {"identities": {"train": [0, 1, 2, 3]}}}


def state_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            state_equal(left, right)
    else:
        assert a == b


def test_rows_keep_target_holdouts_out_of_both_stages(context):
    for arm in n.ARMS:
        for fold, expected in (("primary", {0, 1}), ("alternate", {2, 3}), ("final", {0, 1, 2, 3})):
            for stage in ("pretrain", "target"):
                rows, source, classes = n.training_rows(context, arm, fold, stage)
                ids = {r["vehicle_id"] for r in rows}
                if arm == "E1_nive_transfer" and stage == "pretrain":
                    assert source == "nive" and ids == {"nive:0", "nive:1", "nive:2"}
                else:
                    assert source == "organizer" and ids == expected
                assert classes == len(ids)


def test_source_uses_last_step_no_validation_and_resets_target_head_optimizer(monkeypatch, context):
    monkeypatch.setattr(n.suite, "evaluate_inner", lambda *a: pytest.fail("No holdout during source/final training"))
    parent = n.fit_stage(context, "E1_nive_transfer", 1, stage="pretrain")
    assert parent["best_step"] == parent["updates"] == 2 and parent["inner_map"] is None
    result = n.fit_stage(context, "E1_nive_transfer", 1, fold="final", parent=parent, final_steps=0)
    trained = torch.load(context["output"] / parent["checkpoint"], weights_only=True)["model"]
    target = torch.load(context["output"] / "final/E1_nive_transfer/seed_1/last.pt", weights_only=True)
    set_seed(1)
    fresh = TinyModel(4).state_dict()
    for key, value in target["model"].items():
        state_equal(value, fresh[key] if key.startswith("classifier.") else trained[key])
    assert target["optimizer"]["state"] == {}
    assert result["parent_sha256"] == parent["checkpoint_sha256"]
    assert result["train_identities"] == 4 and parent["train_identities"] == 3


def test_external_pretrain_reused_across_folds_but_control_is_fold_specific(context):
    e1 = n.fit_stage(context, "E1_nive_transfer", 1, stage="pretrain")
    e2 = n.fit_stage(context, "E1_nive_transfer", 1, fold="alternate", stage="pretrain")
    assert e1 == e2
    c1 = n.fit_stage(context, "C1_matched_budget", 1, stage="pretrain")
    c2 = n.fit_stage(context, "C1_matched_budget", 1, fold="alternate", stage="pretrain")
    assert c1["checkpoint"] != c2["checkpoint"] and c1["signature"] != c2["signature"]
    with pytest.raises(ValueError, match="source/fold/seed"):
        n.fit_stage(context, "C1_matched_budget", 1, fold="alternate", parent=c1)


def test_wrong_initialization_and_corrupted_source_checkpoint_are_rejected(context):
    with pytest.raises(ValueError, match="requires its pretraining"):
        n.fit_stage(context, "E1_nive_transfer", 1)
    with pytest.raises(ValueError, match="only exists for C1/E1"):
        n.fit_stage(context, "B0_control", 1, stage="pretrain")
    parent = n.fit_stage(context, "E1_nive_transfer", 1, stage="pretrain")
    with pytest.raises(ValueError, match="source/fold/seed"):
        n.fit_stage(context, "E1_nive_transfer", 2, parent=parent)
    n.write_json(context["output"] / parent["checkpoint"], {"corrupted": True})
    with pytest.raises(ValueError, match="checksum"):
        n.fit_stage(context, "E1_nive_transfer", 1, parent=parent)


@pytest.mark.parametrize("path", ["train/1/N_01.jpg", "test/query/3/N_05.jpg"])
def test_external_inventory_drift_is_rejected(monkeypatch, photos, path):
    _, inventory = n.audit_nive(photos, [])
    context = {"nive_root": photos, "manifest": {"nive": {**inventory, "source": {}}}}
    monkeypatch.setattr(n.suite, "verify_protected", lambda *a, **k: None)
    monkeypatch.setattr(n.suite, "load_json", lambda *a: {"frame_sha256": {}})
    n.verify_inputs(context, rehash_frames=True)
    Image.new("RGB", (256, 256), (91, 82, 73)).save(photos / path)
    with pytest.raises(ValueError, match="inventory changed"):
        n.verify_inputs(context, rehash_frames=True)


@pytest.mark.parametrize("arm", ["C1_matched_budget", "E1_nive_transfer"])
@pytest.mark.parametrize("interrupted_stage", ["pretrain", "target"])
def test_resume_matches_uninterrupted_training_and_completed_run_is_noop(monkeypatch, context, arm, interrupted_stage):
    full = n.fit_arm(context, arm, 1)
    other = {**context, "output": context["output"].with_name("resumed")}
    save = n.suite.save_checkpoint
    interrupted = False

    def stop(path, payload):
        nonlocal interrupted
        save(path, payload)
        stage = "pretrain" if "pretrain" in path.parts else "target"
        if not interrupted and stage == interrupted_stage and path.name == "last.pt":
            interrupted = True
            raise KeyboardInterrupt("after authoritative checkpoint")

    monkeypatch.setattr(n.suite, "save_checkpoint", stop)
    with pytest.raises(KeyboardInterrupt):
        n.fit_arm(other, arm, 1)
    monkeypatch.setattr(n.suite, "save_checkpoint", save)
    resumed = n.fit_arm(other, arm, 1)
    assert full["checkpoint_sha256"] == resumed["checkpoint_sha256"]
    path = f"primary/{arm}/seed_1/last.pt"
    a = torch.load(context["output"] / path, weights_only=True)
    b = torch.load(other["output"] / path, weights_only=True)
    for key in ("model", "optimizer", "best", "seen"):
        state_equal(a[key], b[key])
    monkeypatch.setattr(n, "losses", lambda *a, **k: pytest.fail("Completed stages must not retrain"))
    assert n.fit_arm(other, arm, 1) == resumed
    with pytest.raises(ValueError, match="mismatch"):
        n.fit_arm({**other, "signature": "changed"}, arm, 1)


def test_ordinary_control_matches_existing_trainer(monkeypatch, context):
    monkeypatch.setattr(n.suite, "AblationDataset", TinyDataset)
    monkeypatch.setattr(n.suite, "initialize", n.initialize)
    old_context = {**context, "output": context["output"].with_name("old")}
    old = n.suite.fit(old_context, context["variants"]["B0_control"], 1)
    new = n.fit_arm(context, "B0_control", 1)
    assert old["inner_map"] == new["inner_map"] and old["best_step"] == new["best_step"]
    a = torch.load(old_context["output"] / old["checkpoint"], weights_only=True)["model"]
    b = torch.load(context["output"] / new["checkpoint"], weights_only=True)["model"]
    state_equal(a, b)


def test_pilot_never_evaluates_outer_or_exports(monkeypatch, context):
    monkeypatch.setattr(n.suite, "evaluate_final", lambda *a: pytest.fail("No outer in pilot"))
    monkeypatch.setattr(n.suite, "export_final", lambda *a: pytest.fail("No export in pilot"))
    result = n.run_experiment(context)
    assert result["pilot_complete"] and not result["outer_evaluated"] and not result["promoted"]
    assert len(result["primary"]) == 3 and {s["seed"] for s in result["primary"]} == {1}
    assert not (context["output"] / "selection.json").exists()
    assert len(list(context["output"].rglob("last.pt"))) == 5
    assert (context["output"] / "PILOT_RESULTS.md").is_file()


def test_confirmation_freezes_selection_before_alternate_and_all_final_weights_before_outer(monkeypatch, context):
    calls = []
    actual = n.fit_arm

    def fit(ctx, arm, seed, fold="primary", final_steps=None):
        if fold != "primary":
            assert (ctx["output"] / "selection.json").is_file()
        calls.append((arm, seed, fold))
        return actual(ctx, arm, seed, fold, final_steps)

    def evaluate(ctx, summary):
        final = n.suite.load_json(ctx["output"] / "final_selection.json")
        assert len(final) == 3 and summary in final
        directory = ctx["output"] / "final" / summary["variant"] / f"seed_{summary['seed']}"
        for method in ("raw", "reranked"):
            n.write_json(directory / f"per_query_{method}.json", {"q": {"vehicle_id": 1, "ap": .5}})
        metrics = {"mAP_at_10": .5, "candidate_F1": .5, "TNR": .5}
        return {"conditions": {"original": {"raw": metrics, "reranked": metrics}}}

    monkeypatch.setattr(n, "fit_arm", fit)
    monkeypatch.setattr(n.suite, "evaluate_final", evaluate)
    monkeypatch.setattr(n.suite, "export_final", lambda *a: {})
    result = n.run_experiment(context, phase="confirm")
    assert result["confirmation_complete"] and not result["promoted"]
    assert len(calls) == 21 and sum(c[2] == "alternate" for c in calls) == 9
    assert len(list(context["output"].rglob("last.pt"))) == 31
    assert (context["output"] / "paired_bootstrap.json").is_file()


def test_unconfirmed_source_blocks_training(monkeypatch, context):
    context["manifest"]["nive"]["source"]["local_copy_source_confirmed_by_user"] = False
    monkeypatch.setattr(n, "fit_arm", lambda *a: pytest.fail("Source unconfirmed"))
    with pytest.raises(ValueError, match="Confirm the public source"):
        n.run_experiment(context)


def test_existing_training_lock_is_checked_without_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(n.suite, "VARIANT", tmp_path)
    run = tmp_path / "runs/active"
    with n.suite.run_lock(run):
        before = (run / ".lock").stat().st_mtime_ns
        with pytest.raises(RuntimeError, match="active variant 14"):
            n.ensure_previous_suite_idle()
        assert (run / ".lock").stat().st_mtime_ns == before
    n.ensure_previous_suite_idle()


def test_real_representation_transfer_and_onnx_export(monkeypatch, tmp_path):
    variant = Ablation(name="E1_nive_transfer")
    config = variant.recipe(ExperimentConfig(), 7)
    source = n.initialize(3, config, variant, "cpu")
    with torch.no_grad():
        source.bnneck.running_mean.fill_(.1)
    target = n.initialize(2, config, variant, "cpu")
    classifier = target.classifier.weight.detach().clone()
    n.transfer_representation(target, source.state_dict())
    torch.testing.assert_close(target.classifier.weight, classifier, rtol=0, atol=0)
    torch.testing.assert_close(target.bnneck.running_mean, source.bnneck.running_mean, rtol=0, atol=0)
    context = {"output": tmp_path, "base": ExperimentConfig(), "variants": {variant.name: variant},
               "device": torch.device("cpu"), "dataset": tmp_path, "masks": {}}
    directory = tmp_path / "final" / variant.name / "seed_7"
    directory.mkdir(parents=True)
    n.suite.save_checkpoint(directory / "best.pt", {"signature": "transfer-test", "model": target.state_dict()})
    summary = {"variant": variant.name, "seed": 7, "train_identities": 2, "signature": "transfer-test",
               "checkpoint": str((directory / "best.pt").relative_to(tmp_path)),
               "checkpoint_sha256": sha256(directory / "best.pt")}
    loaded, _ = n.suite.load_final_model(context, summary)
    state_equal(loaded.state_dict(), target.state_dict())
    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(9)
    for i in range(8):
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(tmp_path / f"images/{i}.jpg")
    context["rows"] = [{"image_id": str(i), "x": 0, "y": 0, "w": 64, "h": 48} for i in range(8)]
    context["manifest"] = {"protocols": {"validation": {"query_ids": ["0"], "gallery_ids": [str(i) for i in range(1, 8)]}}}
    report = n.suite.export_final(context, summary)
    assert report["dimension"] == 512 and not report["promoted"]
    assert max(report["batch_parity_max_abs"].values()) < 2e-4


def test_nive_notebook_is_valid_and_compiles_with_source_guard():
    notebook = nbformat.read(n.VARIANT / "train_nive_transfer.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "if not SOURCE_CONFIRMED:" in code and "source_confirmed=SOURCE_CONFIRMED" in code
    assert "run_experiment(context, phase=PHASE)" in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "NiVe notebook", "exec")
