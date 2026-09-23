"""Variant 14: synthetic steps/parity only, never train on the organizer dataset."""
import copy
import random
from dataclasses import asdict, replace

import nbformat
import numpy as np
import pytest
import torch
from PIL import Image

from backend.core import sha256
from training import osnet_ablation_suite as suite
from training import osnet_ablations as blocks
from training.hpo import ExperimentConfig, experiment_losses, initialize_experiment
from training.osnet import GeM, MixStyle
from training.pipeline import set_seed


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_grid_is_explicit_and_does_not_clean_annotations():
    grid = blocks.experiment_grid()
    assert len(grid) == 22
    for name, variant in grid.items():
        assert name == variant.name
        variant.recipe(ExperimentConfig(), 3).validate()
        assert not {"exclude", "ignore", "bbox", "overlay"} & set(asdict(variant))
    assert grid["P1_p16k4"].p * grid["P1_p16k4"].k == 64
    assert grid["R1_resolution256"].size == 256
    with pytest.raises(ValueError):
        replace(grid["M2_xbm1024"], metric="triplet").validate()


def test_control_matches_existing_forward_loss_and_gradients():
    variant = blocks.Ablation()
    config = variant.recipe(ExperimentConfig(), 31)
    set_seed(31)
    old, _ = initialize_experiment(2, config, "cpu")
    set_seed(31)
    new = blocks.initialize(2, config, variant, "cpu")
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    clean, robust = torch.randn(4, 3, 208, 208), torch.randn(4, 3, 208, 208)
    labels, ids = torch.tensor([0, 0, 1, 1]), torch.arange(4)
    old_values = experiment_losses(old, clean, robust, labels, config)
    new_values, _ = blocks.losses(new, clean, robust, labels, ids, config, variant, blocks.MemoryBank(0), 0)
    for key in ("loss", "classification", "metric", "consistency"):
        torch.testing.assert_close(new_values[key], old_values[key], rtol=0, atol=0)
    old_values["loss"].backward()
    new_values["loss"].backward()
    torch.testing.assert_close(old.classifier.weight.grad, new.classifier.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["B0_control", "G1_gem", "S2_mixstyle", "L1_letterbox"])
def test_constructor_forwards_actual_architecture_settings(name):
    variant = blocks.experiment_grid()[name]
    config = variant.recipe(ExperimentConfig(use_bnneck=False, mixstyle_probability=1.,
                                              mixstyle_alpha=.3), 31)
    model = blocks.AblationModel(2, config, variant)
    assert isinstance(model.bnneck, torch.nn.Identity)
    assert model.resize_mode == config.resize_mode
    assert isinstance(model.backbone.global_pool, GeM) == (config.pooling == "gem")
    assert isinstance(model.backbone.mixstyle, MixStyle) == config.use_mixstyle
    if config.use_mixstyle:
        assert model.backbone.mixstyle.probability == 1.
        assert float(model.backbone.mixstyle.beta.concentration1) == pytest.approx(.3)


def test_gem_is_trainable_and_mixstyle_changes_training_features():
    gem = blocks.experiment_grid()["G1_gem"]
    model = blocks.initialize(2, gem.recipe(ExperimentConfig(), 31), gem, "cpu")
    assert isinstance(model.backbone.global_pool, GeM)
    optimizer = blocks.optimizer_for(model, gem.recipe(ExperimentConfig(), 31))
    before = model.backbone.global_pool.p.detach().clone()
    model.raw_embedding(torch.randn(4, 3, 208, 208)).square().mean().backward()
    assert model.backbone.global_pool.p.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.backbone.global_pool.p)
    assert "backbone.global_pool.p" in model.state_dict()

    variant = blocks.experiment_grid()["S2_mixstyle"]
    config = variant.recipe(ExperimentConfig(mixstyle_probability=1.), 31)
    model = blocks.initialize(2, config, variant, "cpu")
    assert isinstance(model.backbone.mixstyle, MixStyle)
    calls = []
    hook = model.backbone.mixstyle.register_forward_hook(
        lambda module, args, output: calls.append(not torch.equal(args[0], output)))
    set_seed(31)
    images = torch.randn(4, 3, 208, 208)
    model.train()
    model(images)
    assert calls == [True, True]
    calls.clear()
    model.eval()
    model(images)
    assert calls == [False, False]
    hook.remove()


@pytest.mark.parametrize("name", list(blocks.experiment_grid()))
def test_every_variant_finite_backward_and_optimizer_coverage(name):
    variant = blocks.experiment_grid()[name]
    config = variant.recipe(ExperimentConfig(), 8)
    set_seed(8)
    model = blocks.initialize(2, config, variant, "cpu")
    optimizer = blocks.optimizer_for(model, config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    images = torch.randn(4, 3, variant.size, variant.size)
    labels, ids = torch.tensor([0, 0, 1, 1]), torch.arange(4)
    bank = blocks.MemoryBank(variant.memory_size)
    if variant.memory_size:
        bank.push(torch.randn(4, model.dimension), labels, ids + 10)
    teacher = blocks.ema_teacher(model) if variant.ema else None
    values, raw = blocks.losses(model, images, images.flip(-1), labels, ids,
                                config, variant, bank, 100, teacher)
    assert all(torch.isfinite(v) for v in values.values())
    values["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if variant.branch != "none":
        assert model.aux[0].weight.grad.abs().sum() > 0
    optimizer.step()
    if teacher is not None:
        blocks.update_teacher(teacher, model)
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    model.eval()
    with torch.no_grad():
        embedding = model.embedding(images[:1])
    assert embedding.shape == (1, model.dimension)
    assert raw.shape[1] == model.dimension


@pytest.mark.parametrize("policy", ["none", "backbone", "all"])
def test_freeze_bn_survives_consistency_eval_train_switch(policy):
    variant = blocks.Ablation(freeze_bn=policy)
    config = variant.recipe(ExperimentConfig(), 7)
    model = blocks.initialize(2, config, variant, "cpu")
    bn = next(m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    backbone_before, neck_before = bn.running_mean.clone(), model.bnneck.running_mean.clone()
    images = torch.randn(4, 3, 208, 208)
    blocks.losses(model, images, images, torch.tensor([0, 0, 1, 1]), torch.arange(4), config,
                  variant, blocks.MemoryBank(0), 0)
    assert torch.equal(bn.running_mean, backbone_before) == (policy != "none")
    assert torch.equal(model.bnneck.running_mean, neck_before) == (policy == "all")
    assert bn.weight.requires_grad


def test_masking_and_resize_never_modify_rows_or_image(tmp_path):
    (tmp_path / "images").mkdir()
    source = tmp_path / "images/a.jpg"
    Image.new("RGB", (24, 16), (150, 100, 50)).save(source)
    row = {"image_id": "a", "vehicle_id": 7, "camera_id": 9, "label": 0,
           "x": 4, "y": 3, "w": 16, "h": 8}
    original, checksum = copy.deepcopy(row), sha256(source)
    masks = {"a": {"rectangles": [[0, 0, 4, 4]]}}
    dataset = blocks.AblationDataset([row], blocks.Ablation(mask_probability=1), tmp_path,
                                     augment=True, masks=masks)
    dataset.clean_transform = dataset.robust_transform = lambda image: np.asarray(image)
    clean, robust, _, _ = dataset[0]
    assert clean.shape == robust.shape == (8, 16, 3)
    assert clean[:4, :4].sum() > 0 and robust[:4, :4].sum() == 0
    assert row == original and sha256(source) == checksum
    resized = blocks.AblationDataset([row], blocks.Ablation(size=256), tmp_path)[0][0]
    assert resized.shape == (3, 256, 256)
    with pytest.raises(ValueError, match="mask cache"):
        blocks.AblationDataset([row], blocks.Ablation(mask_probability=.5), tmp_path)


def test_memory_is_detached_bounded_and_masks_same_image():
    bank = blocks.MemoryBank(5)
    source = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    bank.push(source, labels, torch.arange(4))
    assert bank.features.grad_fn is None and not bank.features.requires_grad
    anchors = source.detach().clone().requires_grad_()
    value = bank.loss(anchors, labels, torch.arange(4), .1)
    value.backward()
    assert torch.isfinite(value) and source.grad is None and anchors.grad is not None
    saved = copy.deepcopy(bank.state_dict())
    resumed = blocks.MemoryBank(5)
    resumed.load_state_dict(saved, "cpu")
    torch.testing.assert_close(resumed.loss(anchors, labels, torch.arange(4), .1), value)
    bank.push(source, labels, torch.arange(10, 14))
    assert bank.ids.tolist() == [3, 10, 11, 12, 13]
    single = blocks.MemoryBank(3)
    single.push(torch.ones(1, 3), torch.tensor([0]), torch.tensor([1]))
    empty_loss = single.loss(torch.ones(1, 3, requires_grad=True), torch.tensor([0]), torch.tensor([1]), .1)
    assert float(empty_loss.detach()) == 0


def test_balanced_sampler_cross_camera_seed_and_coverage():
    rows = [{"image_id": f"{label}-{camera}-{i}", "label": label, "camera_id": camera}
            for label in range(4) for camera in (1, 2) for i in range(2)]
    rng = np.random.default_rng(1)
    features = {r["image_id"]: rng.normal(size=4) for r in rows}
    config = ExperimentConfig(identities_per_batch=2, images_per_identity=2)
    sampler = blocks.BalancedPositiveSampler(rows, config, 100, features)
    batches = list(sampler)
    assert batches == list(sampler)
    assert set(i for batch in batches for i in batch) == set(range(len(rows)))
    for batch in batches:
        for a, b in (batch[:2], batch[2:]):
            assert rows[a]["label"] == rows[b]["label"]
            assert rows[a]["camera_id"] != rows[b]["camera_id"]


def test_budget_and_selection_only_use_inner():
    budget = suite.Budget()
    assert {285, 850, 1700}.issubset(budget.boundaries(1700))
    assert budget.boundaries(0) == []
    values = [{"variant": n, "inner_map": score, "outer_map": 1 - score}
              for n, score in (("B0_control", .7), ("N1", .8), ("C1", .6), ("P1", .9))]
    assert suite.select_finalists(values) == ["B0_control", "P1", "N1"]
    confirmed = [{"variant": n, "seed": s, "inner_map": score, "best_step": s * 10}
                 for n, score in (("B0_control", .7), ("P1", .7)) for s in (1, 2, 3)]
    winner, aggregate = suite.select_winner(confirmed, (1, 2, 3))
    assert winner == "B0_control" and aggregate[winner]["final_steps"] == 20
    with pytest.raises(ValueError, match="seeds"):
        suite.select_winner(confirmed[:-1], (1, 2, 3))


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


def tiny_context(monkeypatch, output):
    class TinyDataset:
        def __init__(self, rows, *_args, **_kwargs):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            clean = torch.tensor([row["vehicle_id"], row["camera_id"], index, 1.], dtype=torch.float)
            robust = clean + torch.rand(4) * .1 + random.random() * .1
            return clean, robust, row["label"], row["image_id"]

    monkeypatch.setattr(suite, "AblationDataset", TinyDataset)
    monkeypatch.setattr(suite, "initialize", lambda n, c, v, d: TinyModel(n).to(d))
    monkeypatch.setattr(suite, "evaluate_inner", lambda model, *args:
                        {"mAP_at_10": float(model.backbone.weight.detach().sum())})
    rows = [{"image_id": f"{i}-{c}", "vehicle_id": i, "camera_id": c}
            for i in range(3) for c in (1, 2)]
    return {"output": output, "signature": "synthetic-test", "base": ExperimentConfig(),
            "rows": rows, "dataset": output, "device": torch.device("cpu"), "masks": {},
            "budget": suite.Budget(max_steps=6, evaluation_interval=2, warmup_steps=1),
            "manifest": {"inner": {"primary": {"train": [0, 1], "validation": [2]}}},
            "split": {"identities": {"train": [0, 1, 2]}}}


def assert_state_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        for left, right in zip(a, b):
            assert_state_equal(left, right)
    else:
        assert a == b


def test_interruption_resumes_optimizer_ema_queue_and_repairs_derived_files(monkeypatch, tmp_path):
    variant = blocks.Ablation(p=2, k=2, ema=True, memory_size=5)
    context = tiny_context(monkeypatch, tmp_path / "complete")
    full = suite.fit(context, variant, 5)
    resumed_context = {**context, "output": tmp_path / "interrupted"}
    original_save = suite.save_checkpoint
    count = 0

    def interrupt(path, payload):
        nonlocal count
        original_save(path, payload)
        count += 1
        if count == 1:
            raise KeyboardInterrupt("after authoritative checkpoint; before history.json")

    monkeypatch.setattr(suite, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        suite.fit(resumed_context, variant, 5)
    monkeypatch.setattr(suite, "save_checkpoint", original_save)
    resumed = suite.fit(resumed_context, variant, 5)
    assert resumed["best_step"] == full["best_step"]
    assert resumed["inner_map"] == full["inner_map"]
    assert resumed["checkpoint_sha256"] == full["checkpoint_sha256"]
    def last(c):
        return torch.load(c["output"] / "primary/B0_control/seed_5/last.pt", weights_only=True)
    before, after = last(context), last(resumed_context)
    for key in ("model", "optimizer", "teacher", "bank", "seen", "best"):
        assert_state_equal(before[key], after[key])
    # A complete rerun is an actual no-op for training and restores a missing best.pt.
    best_path = resumed_context["output"] / resumed["checkpoint"]
    best_path.unlink()
    monkeypatch.setattr(suite, "losses", lambda *a, **k: pytest.fail("Should not retrain"))
    repeated = suite.fit(resumed_context, variant, 5)
    assert repeated == resumed
    with pytest.raises(ValueError, match="mismatch"):
        suite.fit(resumed_context, replace(variant, erase=.1), 5)


def test_final_training_cannot_evaluate_outer_or_select_checkpoint(monkeypatch, tmp_path):
    context = tiny_context(monkeypatch, tmp_path)
    monkeypatch.setattr(suite, "evaluate_inner", lambda *a, **k: pytest.fail("No evaluation during refit"))
    variant = blocks.Ablation(p=2)
    summary = suite.fit(context, variant, 3, final_steps=3)
    assert summary["best_step"] == 3 and summary["inner_map"] is None
    assert summary["train_images"] == len(context["rows"])
    assert summary["train_identities"] == 3
    assert suite.fit(context, variant, 4, final_steps=0)["best_step"] == 0


def test_manifest_drift_lock_and_paired_bootstrap(tmp_path):
    path = tmp_path / "manifest.json"
    suite.freeze_json(path, {"bbox": "original"})
    suite.freeze_json(path, {"bbox": "original"})
    with pytest.raises(ValueError, match="RUN_NAME"):
        suite.freeze_json(path, {"bbox": "changed"})
    with suite.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with suite.run_lock(tmp_path):
                pass
    a = {"q": {"vehicle_id": 1, "ap": .4}}
    b = {"q": {"vehicle_id": 1, "ap": .6}}
    assert suite.paired_bootstrap(a, b, repeats=10)["mean_delta"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="Unpaired"):
        suite.paired_bootstrap(a, {}, repeats=10)


def test_historical_notebook_is_valid_and_compiles_without_clearing_results():
    notebook = nbformat.read(suite.VARIANT / "train_osnet_ablation_suite.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook", "exec")


def test_repair_notebook_is_unexecuted_valid_and_compiles():
    notebook = nbformat.read(suite.VARIANT / "train_osnet_gem_mixstyle_repair.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "suite_v2_gem_mixstyle_fix" in code and "reuse_control_from=REUSE_CONTROL_FROM" in code
    assert "['B0_control', 'G1_gem', 'S2_mixstyle']" in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
            compile(cell.source, "repair notebook", "exec")


@pytest.fixture
def reference_context(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "VARIANT", tmp_path / "variant")
    directory = suite.VARIANT / "runs/suite_v1"
    old = tiny_context(monkeypatch, directory)
    variant = blocks.Ablation(p=2)
    manifest = {"version": 1, "budget": asdict(old["budget"]), "seeds": [1, 2, 3],
                "base_recipe": asdict(old["base"]), "variants": {variant.name: asdict(variant)},
                "inner": {fold: {"train": [0, 1], "validation": [2]} for fold in ("primary", "alternate")},
                "runtime": {"device": "cpu", "torch": "test"},
                "source_sha256": {"training/osnet_ablations.py": "old-model",
                                  "training/osnet_ablation_suite.py": "old-suite", "training/hpo.py": "same"}}
    old.update(manifest=manifest, signature=suite.digest(manifest), variants={variant.name: variant})
    suite.write_json(directory / "manifest.json", manifest)
    for fold in ("primary", "alternate"):
        for seed in manifest["seeds"]:
            suite.fit(old, variant, seed, fold)
    suite.fit(old, variant, 1, final_steps=4)
    suite.write_json(directory / "summary.json", {k: True for k in
                     ("inner_training_complete", "final_training_complete", "final_evaluation_complete")})
    monkeypatch.setattr(suite, "CONTROL_REFERENCE_SHA256", sha256(directory / "manifest.json"))
    new = copy.deepcopy(manifest)
    new["source_sha256"].update({"training/osnet_ablations.py": "repaired-model",
                               "training/osnet_ablation_suite.py": "repaired-suite"})
    new["variants"]["G1_gem"] = asdict(blocks.experiment_grid()["G1_gem"])
    new["control_reference"] = suite.control_reference("suite_v1", new)
    return {**old, "output": suite.VARIANT / "runs/repair", "manifest": new,
            "signature": suite.digest(new)}, variant


def test_reused_control_never_retrains_or_resigns_old_checkpoints(monkeypatch, reference_context):
    context, variant = reference_context
    reference = context["manifest"]["control_reference"]
    directory = suite.VARIANT / "runs" / reference["run_name"]
    before = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}
    initializer = suite.initialize
    monkeypatch.setattr(suite, "initialize", lambda *a: pytest.fail("B0 must not train or initialize"))
    for phase in ("primary", "alternate", "final"):
        for seed in ([1] if phase == "final" else [1, 2, 3]):
            kwargs = {"final_steps": 4} if phase == "final" else {"fold": phase}
            summary = suite.fit(context, variant, seed, **kwargs)
            original = suite.load_json(directory / phase / "B0_control" / f"seed_{seed}" / "summary.json")
            assert summary["signature"] == original["signature"]
            assert summary["checkpoint_sha256"] == original["checkpoint_sha256"]
            assert summary["reused_from"]["run_name"] == "suite_v1"
            assert suite.fit(context, variant, seed, **kwargs) == summary
    assert not list(context["output"].rglob("*.pt"))
    after = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}
    assert before == after
    monkeypatch.setattr(suite, "initialize", initializer)
    suite.load_final_model(context, summary)
    with pytest.raises(ValueError, match="signature/configuration"):
        suite.fit(context, variant, 1, final_steps=3)
    for name in ("G1_gem", "S2_mixstyle"):
        assert suite.reuse_control(context, blocks.experiment_grid()[name], 1, "primary", 6, [], None) is None


@pytest.mark.parametrize("change", ["runtime", "budget", "seeds", "inner", "source", "ablation", "unknown"])
def test_control_reuse_rejects_incompatible_experiments(reference_context, change):
    context, _ = reference_context
    manifest = copy.deepcopy(context["manifest"])
    manifest.pop("control_reference")
    if change == "source":
        manifest["source_sha256"]["training/hpo.py"] = "different"
    elif change == "ablation":
        manifest["variants"]["B0_control"]["erase"] = .1
    else:
        manifest[change] = "different"
    with pytest.raises(ValueError):
        suite.control_reference("suite_v1", manifest)


@pytest.mark.parametrize("filename", ["manifest.json", "summary.json", "history.json", "best.pt"])
def test_control_reuse_rejects_modified_reference_files(reference_context, filename):
    context, variant = reference_context
    directory = suite.VARIANT / "runs/suite_v1"
    path = (directory / filename if filename == "manifest.json" else
            directory / "primary/B0_control/seed_1" / filename)
    suite.write_json(path, {"corrupted": True})
    with pytest.raises(ValueError, match="artifact changed"):
        suite.fit(context, variant, 1)


@pytest.mark.parametrize("name", ["B0_control", "R1_resolution256", "R2_local128", "K1_color32", "G1_gem", "S2_mixstyle"])
def test_real_architecture_onnx_export_variable_batches_without_training(monkeypatch, tmp_path, name):
    variant = blocks.experiment_grid()[name]
    config = variant.recipe(ExperimentConfig(), 6)
    model = blocks.initialize(2, config, variant, "cpu").eval()
    monkeypatch.setattr(suite, "load_final_model", lambda *_: (model, variant))
    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(5)
    rows = []
    for i in range(8):
        Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(tmp_path / f"images/{i}.jpg")
        rows.append({"image_id": str(i), "x": 3, "y": 2, "w": 40, "h": 28})
    context = {"output": tmp_path, "rows": rows, "dataset": tmp_path, "masks": {},
               "manifest": {"protocols": {"validation": {"query_ids": ["0", "1"],
                                                            "gallery_ids": [str(i) for i in range(2, 8)]}}}}
    directory = tmp_path / "final" / name / "seed_6"
    directory.mkdir(parents=True)
    report = suite.export_final(context, {"variant": name, "seed": 6})
    assert report["dimension"] == model.dimension
    assert report["size"] == variant.size
    assert report["promoted"] is False
    assert max(report["batch_parity_max_abs"].values()) < 2e-4


def test_suite_freezes_selection_before_alternate_and_outer(monkeypatch, tmp_path):
    grid = blocks.experiment_grid()
    names = ["B0_control", "N1_backbone_bn", "C1_no_consistency"]
    context = {"output": tmp_path, "signature": "pipeline-test", "seeds": (1, 2, 3),
               "device": torch.device("cpu"), "variants": {n: grid[n] for n in names}}
    calls = []
    monkeypatch.setattr(suite, "verify_protected", lambda *a, **k: None)

    def fake_fit(context, variant, seed, fold="primary", final_steps=None):
        phase = "final" if final_steps is not None else fold
        calls.append((phase, variant.name, seed))
        if phase != "primary":
            assert (tmp_path / "selection.json").is_file()
        return {"variant": variant.name, "seed": seed, "phase": phase,
                "inner_map": {names[0]: .7, names[1]: .8, names[2]: .6}[variant.name],
                "best_step": 2, "samples_seen": 64}

    def fake_evaluation(context, summary):
        frozen = suite.load_json(tmp_path / "final_selection.json")
        assert summary in frozen  # No outer score can affect the model/step/seed choice.
        directory = tmp_path / "final" / summary["variant"] / f"seed_{summary['seed']}"
        for method in ("raw", "reranked"):
            suite.write_json(directory / f"per_query_{method}.json", {"q": {"vehicle_id": 1, "ap": .5}})
        metric = {"mAP_at_10": .5, "candidate_F1": .5, "TNR": .5, "candidate_score": .5}
        return {"conditions": {"original": {"raw": metric, "reranked": metric}}}

    monkeypatch.setattr(suite, "fit", fake_fit)
    monkeypatch.setattr(suite, "evaluate_final", fake_evaluation)
    monkeypatch.setattr(suite, "export_final", lambda *_: {})
    result = suite.run_suite(context)
    assert result["selection"]["winner"] == names[1]
    assert result["promoted"] is False
    assert len([c for c in calls if c[0] == "alternate"]) == 6
    assert len([c for c in calls if c[0] == "final"]) == 2
    assert (tmp_path / "RESULTS.md").is_file()


def test_original_calibration_threshold_is_frozen_for_all_mask_conditions(monkeypatch, tmp_path):
    variant = blocks.Ablation()
    directory = tmp_path / "final/B0_control/seed_1"
    directory.mkdir(parents=True)
    query = [{"image_id": "q", "vehicle_id": 1, "camera_id": 1}]
    gallery = [{"image_id": "g", "vehicle_id": 1, "camera_id": 2}]
    monkeypatch.setattr(suite, "load_final_model", lambda *_: (object(), variant))
    monkeypatch.setattr(suite, "protocol_rows", lambda *args: (query, gallery))
    monkeypatch.setattr(suite, "encode", lambda *a, **k: {"q": np.array([1., 0.]), "g": np.array([1., 0.])})
    monkeypatch.setattr(suite, "ranking_pair", lambda q, g, e:
                        {"raw": (suite.ranked_queries(q, g, e), None),
                         "reranked": (suite.ranked_queries(q, g, e), None)})
    calibration_calls, thresholds = [], []
    monkeypatch.setattr(suite, "calibrate", lambda *a: calibration_calls.append(1) or .625)
    actual_metrics = suite.metrics

    def inspect_metrics(ranked, threshold, confidence):
        assert (directory / "thresholds.json").is_file()
        thresholds.append(threshold)
        return actual_metrics(ranked, threshold, confidence)

    monkeypatch.setattr(suite, "metrics", inspect_metrics)
    result = suite.evaluate_final({"output": tmp_path},
                                 {"variant": variant.name, "seed": 1, "checkpoint_sha256": "test"})
    assert len(calibration_calls) == 2  # Only original calibration, raw and reranked.
    assert thresholds == [.625] * 8
    assert len(result["conditions"]) == 4
