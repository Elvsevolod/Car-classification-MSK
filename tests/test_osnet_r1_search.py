import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import copy

import numpy as np
import pytest
import torch
from torch import nn

from training import osnet_r1_search as search
from training import research_io as io, research_training as common, research_scoring as scoring

torch.set_num_threads(2)


def settings():
    return io.read(search.VARIANT / "config.json")


def recipe():
    return {"encoder_lr": 1.2e-4, "head_lr_multiplier": 10., "weight_decay": 6e-5,
            "label_smoothing": .1, "supcon_temperature": .1,
            "metric_weight": 1.49, "consistency_weight": .22}


def test_grid_is_18_conditions_with_exact_exposure_and_original_lr():
    grid = search.trial_grid(recipe())
    search.check_settings(settings(), grid)
    assert len(grid) == 18
    assert {s["loss"] for s in grid} == {"supcon", "soft_triplet"}
    assert {(s["p"], s["k"]) for s in grid} == {(16, 2), (16, 4), (32, 2)}
    assert {s["lr"] for s in grid} == {1.2e-5, 3.6e-5, 1.2e-4}
    for exposure in settings()["presentations"]:
        assert {exposure // (s["p"] * s["k"]) * (s["p"] * s["k"]) for s in grid} == {exposure}
    assert max(settings()["presentations"]) // 32 == 1600
    assert max(settings()["presentations"]) // 64 == 800


def test_exposure_schedule_does_not_restart_at_evaluation_boundaries():
    plan = settings()
    assert search.lr_fraction(3200, plan) == 1.
    assert search.lr_fraction(51200, plan) == .02
    for seen in (6400, 12800, 25600, 51200):
        assert search.lr_fraction((seen // 32) * 32, plan) == search.lr_fraction((seen // 64) * 64, plan)
    assert 1 > search.lr_fraction(6400, plan) > search.lr_fraction(12800, plan)
    with pytest.raises(ValueError): search.lr_fraction(51232, plan)
    with pytest.raises(ValueError):
        search.check_settings({**plan, "presentations": [6401]}, search.trial_grid(recipe()))


class TinyR1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3*16*16, 8), nn.Dropout(.1))
        self.bnneck = nn.BatchNorm1d(8)
        self.classifier = nn.Linear(8, 3, bias=False)

    def embedding(self, x):
        return self.bnneck(self.backbone(x))

    def forward(self, x):
        raw = self.backbone(x)
        value = self.bnneck(raw)
        return self.classifier(value), raw, value


def tiny_context(tmp_path):
    from PIL import Image
    rows = []
    for identity in range(3):
        for n in range(2):
            name = f"{identity}_{n}.png"
            Image.new("RGB", (20, 20), (30+identity*60, 30+n*60, 110)).save(tmp_path/name)
            rows.append({"image_id": name, "vehicle_id": identity, "camera_id": n,
                         "path": name, "label": identity, "x": 0, "y": 0, "w": 20, "h": 20})
    return {"signature": "tiny", "device": torch.device("cpu"), "inputs": tmp_path,
            "output": tmp_path, "train": rows, "recipe": recipe(),
            "settings": {"presentations": [8, 16], "warmup_presentations": 4, "min_lr_ratio": .02},
            "manifest": {"models": {search.PARENT: {"sha256": "test-parent", "step": 800}}}}


@pytest.mark.parametrize("loss", ["supcon", "soft_triplet"])
def test_resume_reproduces_parameters_buffers_and_scientific_history(tmp_path, loss):
    c = tiny_context(tmp_path)
    spec = {"id": "test", "p": 2, "k": 2, "size": 16, "lr": 1e-4, "loss": loss, "seed": 7}
    common.seed_all(5)
    model = TinyR1()
    initial = copy.deepcopy(model)
    search.train_to(model, c, spec, 16, tmp_path/"full")
    search.train_to(initial, c, spec, 8, tmp_path/"resumed")
    resumed = TinyR1()
    search.train_to(resumed, c, spec, 16, tmp_path/"resumed")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    assert io.read(tmp_path/"full/history.json") == io.read(tmp_path/"resumed/history.json")
    assert io.read(tmp_path/"full/history.json")[-1]["presentations"] == 16
    pointer = io.read(tmp_path/"resumed/resume.json")
    (tmp_path/"resumed"/pointer["path"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Missing/changed"):
        search.train_to(resumed, c, spec, 16, tmp_path/"resumed")


@pytest.mark.parametrize("loss", ["supcon", "soft_triplet"])
def test_loss_uses_correct_metric_and_retains_consistency(loss):
    common.seed_all(11)
    model = TinyR1()
    x = torch.rand(6, 3, 16, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    spec = {"loss": loss}
    values = search.losses(model, x, x.flip(-1), labels, recipe(), spec)
    assert all(torch.isfinite(v) for v in values.values())
    expected = values["ce"] + 1.49*values["metric"] + .22*values["consistency"]
    torch.testing.assert_close(values["loss"], expected, rtol=0, atol=0)
    values["loss"].backward()
    assert model.backbone[1].weight.grad is not None
    assert model.bnneck.num_batches_tracked.item() == 1  # No update on clean target forward.


def scoring_context(tmp_path):
    rng = np.random.default_rng(42)
    rows = [{"image_id": str(i), "vehicle_id": i % 5, "camera_id": 0 if i < 5 else 1,
             "x": 0, "y": 0, "w": 16, "h": 16} for i in range(25)]
    parts = [scoring.mix([rng.normal(size=(25, 512))], [1]) for _ in range(4)]
    directory = tmp_path/"baseline"
    directory.mkdir()
    for name, values in zip(("B0", search.PARENT, "R1_20260916", "R1_20260917"), parts):
        np.save(directory/f"{name}.npy", values)
    c = {"output": tmp_path, "rows": rows, "control": search.replacement_bank(parts, parts[1]),
         "signature": "fixture", "manifest": {"baseline": "test control",
         "draws": {"first": {"query_ids": [str(i) for i in range(5)],
                             "gallery_ids": [str(i) for i in range(5, 25)]}}}}
    c["evaluation"] = search.evaluation_context(c)
    return c, parts


def test_replacement_noop_top10_and_zero_weight_control(tmp_path):
    c, parts = scoring_context(tmp_path)
    parent = search.evaluate_parent(c)
    assert parent["metrics"]["means"]["control"] == parent["metrics"]["means"]["replace_R1_20260915"]
    d = c["evaluation"]["draws"]["first"]
    distances = scoring.fuse_distances(d["raw"], d["jac"], parts[1][d["qi"]], parts[1][d["gi"]], "G2", 0)
    reference = np.argsort(.5*d["jac"]+.5*d["raw"], axis=1, kind="stable")
    np.testing.assert_array_equal(np.argsort(distances, axis=1, kind="stable"), reference)


def test_query_permutation_and_subset_leave_all_top10_unchanged(tmp_path):
    c, parts = scoring_context(tmp_path)
    features = np.roll(parts[1], 1, axis=0)
    original = search.score_checkpoint(features, c)["draws"]["first"]
    for ids in (["4", "2", "0"], ["3"], ["4", "3", "2", "1", "0"]):
        c["manifest"]["draws"]["first"]["query_ids"] = ids
        c["evaluation"] = search.evaluation_context(c)
        changed = search.score_checkpoint(features, c)["draws"]["first"]
        for key in original:
            assert all(changed[key]["per_query"][qid]["top10"] == original[key]["per_query"][qid]["top10"] for qid in ids)
    with pytest.raises(ValueError, match="Invalid real feature"):
        search.score_checkpoint(np.zeros_like(features), c)
    with pytest.raises(ValueError, match="Invalid real feature"):
        search.score_checkpoint(np.full_like(features, np.nan), c)


def test_selection_retains_earlier_best_weights_not_final_step_or_seed_mean(tmp_path):
    c, parts = scoring_context(tmp_path)
    parent = search.evaluate_parent(c)
    c["trials"] = [{"id": "one"}]
    # Override only synthetic scores, not a real validation or experiment result.
    for exposure, score in ((8, .99), (16, .98)):
        directory = tmp_path/f"trials/one/exposure_{exposure:06d}"
        r = copy.deepcopy(parent)
        r.update(trial="one", presentations=exposure, checkpoint={"path": f"best_{exposure}.pt", "sha256": "fixture"})
        r["metrics"]["means"].update(add_G2_10=score, replace_R1_20260915=score-.01)
        io.write(directory/"result.json", r)
        io.finish(directory, "fixture")
    io.write(tmp_path/"status/one.json", {"trial": "one", "status": "complete"})
    result = search.summarize(c, "complete")
    assert result["best_system"]["presentations"] == 8
    assert result["best_system"]["checkpoint"]["path"] == "best_8.pt"
    assert not result["promoted"] and not result["selection_is_provisional"]
    assert (tmp_path/"selected_candidate.json").is_file()


def test_incomplete_experiment_has_no_final_selection(tmp_path):
    c, _ = scoring_context(tmp_path)
    c["trials"] = [{"id": "one"}]
    search.evaluate_parent(c)
    result = search.summarize(c, "completed_with_failures")
    assert result["selection_is_provisional"]
    assert not (tmp_path/"selected_candidate.json").exists()


@pytest.mark.parametrize("failure", [RuntimeError("simulated OOM"), ValueError("corrupt signature")])
def test_runner_continues_oom_but_stops_integrity_failures(tmp_path, monkeypatch, failure):
    variant = tmp_path/"variant"
    output = variant/"runs/synthetic"
    output.mkdir(parents=True)
    c, _ = scoring_context(output)
    c.update(inputs=tmp_path, device=torch.device("cpu"))
    c["manifest"].update(models={search.PARENT: {"sha256": "fixture", "step": 800}}, files={})
    io.write(tmp_path/"provenance/v16_manifest.json", {"base_recipe": recipe()})
    io.write(variant/"config.json", settings())
    monkeypatch.setattr(search, "VARIANT", variant)
    monkeypatch.setattr(search.runtime, "prepare", lambda *args: c)
    monkeypatch.setattr(search.runtime, "baselines", lambda context: context["control"])
    monkeypatch.setattr(search, "check_disk", lambda *args: None)
    monkeypatch.setattr(search.io, "archive", lambda *args, **kwargs: None)
    calls = []

    def trial(context, spec):
        calls.append(spec["id"])
        if len(calls) == 1:
            raise failure

    monkeypatch.setattr(search, "run_trial", trial)
    if isinstance(failure, RuntimeError):
        result = search.run(tmp_path, "synthetic", device="cuda")
        assert len(calls) == 18 and result["completed_trials"] == 17
        assert result["status"] == "completed_with_failures" and result["selection_is_provisional"]
    else:
        with pytest.raises(ValueError, match="corrupt signature"):
            search.run(tmp_path, "synthetic", device="cuda")
        assert len(calls) == 1
        assert io.read(output/"results.json")["status"] == "interrupted"
    assert not (output/"selected_candidate.json").exists()


def test_completed_stages_skip_model_loading_and_verify_bytes(tmp_path, monkeypatch):
    c = {"output": tmp_path, "signature": "test", "settings": {"presentations": [64, 128]}}
    spec = {"id": "fixture"}
    for exposure in (64, 128):
        directory = tmp_path/f"trials/fixture/exposure_{exposure:06d}"
        io.write(directory/"result.json", {"test": True})
        io.finish(directory, io.digest({"run": "test", "spec": spec, "exposure": exposure}))
    monkeypatch.setattr(search.models, "load_osnet", lambda *args: pytest.fail("Completed stages must not reload model"))
    monkeypatch.setattr(search.runtime, "cleanup", lambda: None)
    search.run_trial(c, spec)
    io.write(tmp_path/"trials/fixture/exposure_000064/result.json", {"test": False})
    with pytest.raises(ValueError, match="Missing/changed"):
        search.run_trial(c, spec)


def test_notebook_run_all_is_valid_and_does_not_install_or_launch_other_variants():
    import nbformat
    path = search.VARIANT/"search_osnet_r1.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    compile(code, path.name, "exec")
    assert code.index("ORT_DISABLE_TELEMETRY") < code.index("from training")
    assert "pip install" not in code and "timeout=" not in code
    assert "from training.osnet_r1_search import run" in code
    assert "research_gpu import run" not in code


def test_no_cpu_fallback_or_output_path_escape():
    with pytest.raises(ValueError, match="MPS/CUDA"):
        search.run("missing", device="cpu")
    with pytest.raises(ValueError, match="RUN_NAME"):
        search.run("missing", "../escape")


def test_mps_preflight_rejects_rosetta_and_fallback(monkeypatch):
    monkeypatch.setattr(search.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(search.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="native Apple Silicon"):
        search.mps_preflight()
    monkeypatch.setattr(search.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="restart the kernel"):
        search.mps_preflight()


def test_low_disk_is_a_clear_error(tmp_path, monkeypatch):
    class Usage:
        free = 1
    monkeypatch.setattr(search.shutil, "disk_usage", lambda p: Usage())
    with pytest.raises(OSError, match="Недостаточно места"):
        search.check_disk(tmp_path, 100)


@pytest.mark.skipif(os.environ.get("RUN_R1_MPS_SMOKE") != "1", reason="opt-in actual OSNet/MPS smoke")
@pytest.mark.parametrize("loss", ["supcon", "soft_triplet"])
@pytest.mark.parametrize("p,k", [(16, 2), (16, 4), (32, 2)])
def test_real_r1_mps_forward_backward(loss, p, k):
    search.mps_preflight()
    root = io.ROOT/"research_transfer/v41_inputs"
    m = io.read(root/"inputs.json")
    r = io.read(root/"provenance/v16_manifest.json")["base_recipe"]
    rows = common.labeled([row for row in m["rows"] if row["vehicle_id"] in set(m["train_ids"])])
    spec = {"id": "smoke", "p": p, "k": k, "size": 256, "loss": loss, "lr": 1e-5, "seed": 7}
    common.seed_all(7)
    model = search.models.load_osnet(root, m).to("mps")
    data = search.models.Images(root, rows, 256, train=True, paired=True)
    batch = common.fetch(data, common.batch_indices(rows, p, k, 7, 0), torch.device("mps"), 77)
    optimizer = search.optimizer_for(model, r, spec)
    result = search.losses(model, *batch, r, spec)
    result["loss"].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    with torch.no_grad(): features = model.embedding(batch[0])
    assert features.shape == (p*k, 512) and torch.isfinite(features).all() and torch.isfinite(norm)
    del model, optimizer, result, features, batch
    search.runtime.cleanup()
