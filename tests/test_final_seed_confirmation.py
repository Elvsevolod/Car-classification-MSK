"""Synthetic CPU-only continuation checks; never train on organizer photographs."""
import copy
import random
from dataclasses import asdict, replace

import nbformat
import pytest
import torch

from backend.core import sha256
from training import final_seed_confirmation as continuation
from training.final_seed_report import CONDITIONS, COUNTS, METHODS, METRICS
from training.hpo import ExperimentConfig


review = continuation.review


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


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
        clean = torch.tensor([row["label"], index, row["camera_id"], 1.], dtype=torch.float)
        robust = clean + torch.rand(4) * .1 + random.random() * .1
        return clean, robust, row["label"], row["image_id"]


def evaluation_report(name, seed):
    value = {1: .9, 2: .8, 3: .7}[seed] + (.02 if name == continuation.NAMES[1] else 0.)
    scores = {metric: value for metric in METRICS}
    scores.update({metric: 10 for metric in COUNTS})
    return {"conditions": {condition: {method: copy.deepcopy(scores) for method in METHODS}
                           for condition in CONDITIONS},
            "per_query": {method: {"q1": {"vehicle_id": 1, "ap": value}} for method in METHODS}}


@pytest.fixture
def context(monkeypatch, tmp_path):
    monkeypatch.setattr(continuation, "STEPS", 2)
    monkeypatch.setattr(continuation, "VARIANT", tmp_path / "variant17")
    monkeypatch.setattr(continuation, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(review, "check_other_runs", lambda *_args: None)
    monkeypatch.setattr(review, "AblationDataset", TinyDataset)
    monkeypatch.setattr(review, "initialize", lambda classes, config, variant, device: TinyModel(classes).to(device))
    monkeypatch.setattr(review, "evaluate_draws", lambda *_args, **_kwargs: pytest.fail("No inner selection"))
    rows = [{"vehicle_id": identity, "camera_id": camera, "image_id": f"image_{identity}_{camera}"}
            for identity in range(4) for camera in (1, 2)]
    dataset = tmp_path / "data"
    for row in rows:
        review.write_json(dataset / "images" / f"{row['image_id']}.jpg", row)
    protected_file = tmp_path / "unchanged_source.json"
    review.write_json(protected_file, {"historical": True})
    manifest = {"source_sha256": {}, "protected": {str(protected_file): sha256(protected_file)},
                "runtime": continuation.runtime(torch.device("cpu")),
                "frames_sha256": continuation.digest({r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg")
                                                      for r in rows}),
                "extension": {"reused_seed": 1, "new_seeds": [2, 3]},
                "inner": {}, "protocols": {}}
    baseline_scores = evaluation_report(continuation.NAMES[0], 1)["conditions"]["original"]["raw"]
    review.write_json(continuation.ARTIFACTS / "baseline_metrics.json",
                      {"validation": baseline_scores, "raw_baseline": {"validation": baseline_scores}})
    return {"output": continuation.VARIANT / "runs" / "synthetic", "manifest": manifest,
            "signature": continuation.digest(manifest), "protected": manifest["protected"],
            "base": ExperimentConfig(), "rows": rows, "dataset": dataset,
            "device": torch.device("cpu"), "masks": {},
            "variants": {name: replace(review.variants()[name], p=2) for name in continuation.NAMES},
            "seeds": (1, 2, 3), "budget": review.old.Budget(4, 2, 1),
            "split": {"identities": {"train": [0, 1, 2, 3]}},
            "first_result": {"outer": {name: evaluation_report(name, 1) for name in continuation.NAMES}}}


@pytest.fixture
def synthetic_evaluation(monkeypatch):
    calls = []

    def evaluate(ctx, summary):
        frozen = review.old.load_json(ctx["output"] / "final_selection.json")
        assert len(frozen) == 4
        assert {(s["variant"], s["seed"]) for s in frozen} == {(n, s) for n in continuation.NAMES for s in (2, 3)}
        for item in frozen:
            assert item["fold"] == "final" and item["stop_step"] == 2 and item["lr_horizon"] == 4
            assert item["validation"] is None and all(h["validation"] is None for h in item["history"])
            for checkpoint in item["checkpoints"].values():
                assert sha256(ctx["output"] / checkpoint["path"]) == checkpoint["sha256"]
        assert summary["seed"] != 1
        calls.append((summary["variant"], summary["seed"]))
        directory = ctx["output"] / "final" / summary["variant"] / f"seed_{summary['seed']}"
        report = evaluation_report(summary["variant"], summary["seed"])
        for name in continuation.EVALUATION_FILES:
            review.write_json(directory / name, report if name == "evaluation.json" else {"synthetic": name})
        return report

    monkeypatch.setattr(review, "evaluate_final", evaluate)
    return calls


def run(context):
    return continuation.run(context, allow_outer=True, allow_cpu=True)


def test_exact_four_new_fits_all_frozen_before_outer_and_mean_all_three(context, synthetic_evaluation, monkeypatch):
    fit, fits = review.fit, []

    def record(ctx, name, seed, fold, steps):
        fits.append((name, seed, fold, steps))
        return fit(ctx, name, seed, fold, steps)

    monkeypatch.setattr(review, "fit", record)
    first = copy.deepcopy(context["first_result"])
    result = run(context)
    expected = [(name, seed) for seed in (2, 3) for name in continuation.NAMES]
    assert fits == [(name, seed, "final", 2) for name, seed in expected]
    assert synthetic_evaluation == expected
    assert result["new_optimizer_updates"] == 8 and result["reused_seed"] == 1
    assert result["complete"] and result["outer_evaluated"] and not result["promoted"]
    assert len(result["evaluations"]) == 6 and context["first_result"] == first
    metric = result["aggregate"]["B0_control"]["original"]["reranked"]["mAP_at_10"]
    assert metric["mean"] == pytest.approx(.8) and metric["std"] == pytest.approx(.1)
    assert metric["seed_values"] == {"1": .9, "2": .8, "3": .7}
    assert result["paired_deltas"]["original"]["reranked"]["mAP_at_10"]["mean"] == pytest.approx(.02)


def test_completed_noop_checks_artifacts_without_training_or_evaluation(context, synthetic_evaluation, monkeypatch):
    expected = run(context)
    monkeypatch.setattr(review, "fit", lambda *_args: pytest.fail("Completed run retrained"))
    monkeypatch.setattr(review, "evaluate_final", lambda *_args: pytest.fail("Completed run reevaluated"))
    assert run(context) == expected
    artifact = context["output"] / "final/B0_control/seed_2/bundle.json"
    review.write_json(artifact, {"corrupted": True})
    with pytest.raises(ValueError, match="Completed artifact changed"):
        run(context)


@pytest.mark.parametrize("relative", ["results.json", "FINAL_RESULTS.md", "final_selection.json",
                                      "final/R1_resolution256/seed_3/evaluation_complete.json",
                                      "final/R1_resolution256/seed_3/last.pt"])
def test_completed_run_detects_material_artifact_corruption(context, synthetic_evaluation, relative):
    run(context)
    review.write_json(context["output"] / relative, {"changed": True})
    with pytest.raises(ValueError, match="Completed artifact changed"):
        run(context)


def test_partial_training_resumes_authoritative_checkpoint(context, synthetic_evaluation, monkeypatch):
    save = review.old.save_checkpoint
    interrupted = False

    def interrupt(path, payload):
        nonlocal interrupted
        save(path, payload)
        if path.name == "last.pt" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after authoritative training commit")

    monkeypatch.setattr(review.old, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(context)
    assert not synthetic_evaluation and not (context["output"] / "complete.json").exists()
    first_path = context["output"] / "final/B0_control/seed_2/step_00002.pt"
    frozen_hash = sha256(first_path)
    monkeypatch.setattr(review.old, "save_checkpoint", save)
    result = run(context)
    assert result["new_optimizer_updates"] == 8 and sha256(first_path) == frozen_hash
    assert len(synthetic_evaluation) == 4


def test_stale_history_prefix_recovered_from_authoritative_checkpoint(context, monkeypatch):
    monkeypatch.setattr(continuation, "STEPS", 4)
    summary = continuation.fit_final(context, "B0_control", 2)
    directory = context["output"] / "final/B0_control/seed_2"
    assert len(summary["history"]) == 2
    history = directory / "history.json"
    review.write_json(history, summary["history"][:1])
    before = {name: sha256(directory / name) for name in ("summary.json", "last.pt", "step_00004.pt")}
    monkeypatch.setattr(review, "initialize", lambda *_args: pytest.fail("Only recover history; no retraining"))
    assert continuation.fit_final(context, "B0_control", 2) == summary
    assert review.old.load_json(history) == summary["history"]
    assert before == {name: sha256(directory / name) for name in before}


def test_arbitrary_history_mismatch_not_silently_repaired(context):
    continuation.fit_final(context, "B0_control", 2)
    history = context["output"] / "final/B0_control/seed_2/history.json"
    changed = [{"step": 999, "train": {"loss": 0.}}]
    review.write_json(history, changed)
    with pytest.raises(ValueError, match="[Hh]istory"):
        continuation.fit_final(context, "B0_control", 2)
    assert review.old.load_json(history) == changed


def test_summary_history_must_match_authoritative_checkpoint(context):
    continuation.fit_final(context, "B0_control", 2)
    path = context["output"] / "final/B0_control/seed_2/summary.json"
    summary = review.old.load_json(path)
    summary["history"][0]["train"]["loss"] += 1
    review.write_json(path, summary)
    with pytest.raises(ValueError, match="[Hh]istory|[Aa]uthoritative"):
        continuation.fit_final(context, "B0_control", 2)


@pytest.mark.parametrize("partial", ["evaluation_only", "all_export_files"])
def test_uncommitted_evaluation_is_retried_not_treated_as_complete(context, synthetic_evaluation, monkeypatch, partial):
    evaluate = review.evaluate_final

    def interrupt(ctx, summary):
        if partial == "all_export_files":
            evaluate(ctx, summary)
        else:
            directory = ctx["output"] / "final" / summary["variant"] / f"seed_{summary['seed']}"
            review.write_json(directory / "evaluation.json", evaluation_report(summary["variant"], summary["seed"]))
        raise KeyboardInterrupt("export or receipt interrupted")

    monkeypatch.setattr(review, "evaluate_final", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(context)
    assert not list(context["output"].glob("final/*/seed_*/evaluation_complete.json"))
    monkeypatch.setattr(review, "evaluate_final", evaluate)
    before = len(synthetic_evaluation)
    run(context)
    assert len(synthetic_evaluation) - before == 4


def test_committed_evaluations_cached_when_later_evaluation_interrupted(context, synthetic_evaluation, monkeypatch):
    evaluate = review.evaluate_final

    def interrupt(ctx, summary):
        if summary["variant"] == "R1_resolution256" and summary["seed"] == 2:
            raise KeyboardInterrupt("second model evaluation")
        return evaluate(ctx, summary)

    monkeypatch.setattr(review, "evaluate_final", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(context)
    assert synthetic_evaluation == [("B0_control", 2)]
    monkeypatch.setattr(review, "evaluate_final", evaluate)
    run(context)
    assert synthetic_evaluation == [(name, seed) for seed in (2, 3) for name in continuation.NAMES]


def test_evaluation_rejects_returned_report_different_from_saved(context, synthetic_evaluation, monkeypatch):
    evaluate = review.evaluate_final

    def mismatched(ctx, summary):
        report = evaluate(ctx, summary)
        return {**report, "unexpected": True}

    monkeypatch.setattr(review, "evaluate_final", mismatched)
    with pytest.raises(ValueError, match="differs from saved"):
        run(context)
    assert not list(context["output"].glob("final/*/seed_*/evaluation_complete.json"))


def test_cached_evaluation_rejects_different_checkpoint_signature(context, synthetic_evaluation):
    result = run(context)
    summary = {**result["final"][0], "signature": "different-stage"}
    with pytest.raises(ValueError, match="Evaluation checkpoint changed"):
        continuation.evaluation(context, summary)


def test_explicit_outer_and_cpu_guards_before_training(context, monkeypatch):
    monkeypatch.setattr(review, "fit", lambda *_args: pytest.fail("Guard must run before training"))
    with pytest.raises(ValueError, match="ALLOW_OUTER_EVALUATION"):
        continuation.run(context, allow_cpu=True)
    with pytest.raises(RuntimeError, match="ALLOW_CPU_TRAINING"):
        continuation.run(context, allow_outer=True)


def test_changed_runtime_and_protected_source_fail_before_training(context, monkeypatch):
    monkeypatch.setattr(review, "fit", lambda *_args: pytest.fail("Guard must run before training"))
    modified = copy.deepcopy(context)
    modified["manifest"]["runtime"]["torch_threads"] += 1
    with pytest.raises(ValueError, match="Runtime changed"):
        run(modified)
    protected = next(iter(context["protected"]))
    review.write_json(protected, {"changed": True})
    with pytest.raises(ValueError, match="Protected input changed"):
        run(context)


def test_changed_frame_fails_before_training(context, monkeypatch):
    monkeypatch.setattr(review, "fit", lambda *_args: pytest.fail("Guard must run before training"))
    review.write_json(context["dataset"] / "images/image_0_1.jpg", {"changed": True})
    with pytest.raises(ValueError, match="Original images changed"):
        run(context)


def test_parent_lock_blocks_concurrent_variant17_runs(context, monkeypatch):
    monkeypatch.setattr(review, "fit", lambda *_args: pytest.fail("Concurrent run must not train"))
    with review.old.run_lock(continuation.VARIANT / "runs"):
        with pytest.raises(RuntimeError, match="already running"):
            run(context)


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_unavailable_accelerator_fails_before_training(context, monkeypatch, device):
    context["device"] = torch.device(device)
    context["manifest"]["runtime"] = continuation.runtime(context["device"])
    target = torch.backends.mps if device == "mps" else torch.cuda
    monkeypatch.setattr(target, "is_available", lambda: False)
    monkeypatch.setattr(review, "fit", lambda *_args: pytest.fail("Unavailable accelerator must stop"))
    with pytest.raises(RuntimeError, match="unavailable"):
        run(context)


@pytest.mark.parametrize("name", ["../outside", "a/b", "", "with space", "/absolute"])
def test_run_name_rejects_traversal_or_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError, match="simple directory"):
        continuation.run_directory(tmp_path, name)


def test_run_directory_rejects_symlink_escape(tmp_path):
    variant, outside = tmp_path / "variant", tmp_path / "outside"
    (variant / "runs").mkdir(parents=True)
    outside.mkdir()
    (variant / "runs" / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        continuation.run_directory(variant, "escape")


def test_prepare_freezes_extension_without_mutating_source(context, monkeypatch):
    source = {**context, "output": context["output"].with_name("source")}
    source_file = context["dataset"] / "source-final.json"
    review.write_json(source_file, {"source": True})
    source_files = {str(source_file): sha256(source_file)}
    before = copy.deepcopy(source)
    monkeypatch.setattr(continuation, "load_source", lambda *_args: (source, source["first_result"], source_files))
    prepared = continuation.prepare("extension")
    assert source == before
    assert prepared["output"] != source["output"] and prepared["signature"] != source["signature"]
    assert prepared["manifest"]["extension"]["new_seeds"] == [2, 3]
    assert prepared["manifest"]["extension"]["stop_steps"] == {name: 2 for name in continuation.NAMES}
    assert prepared["protected"][str(source_file)] == source_files[str(source_file)]
    assert set(prepared["manifest"]["source_sha256"]) == {"training/final_seed_confirmation.py", "training/final_seed_report.py"}
    assert review.old.load_json(prepared["output"] / "manifest.json") == prepared["manifest"]
    assert continuation.prepare("extension")["signature"] == prepared["signature"]
    monkeypatch.setattr(continuation, "runtime", lambda _device: {"different": True})
    with pytest.raises(ValueError, match="Runtime differs"):
        continuation.prepare("wrong_runtime")


@pytest.fixture
def source_files(monkeypatch, tmp_path):
    """Isolate extra source-consistency checks; old phase verification has its own tests."""
    monkeypatch.setattr(review, "VARIANT", tmp_path / "variant16")
    monkeypatch.setattr(continuation, "ARTIFACTS", tmp_path / "artifacts")
    directory = review.VARIANT / "runs/review_v1"
    dataset = tmp_path / "dataset"
    rows = [{"image_id": "image_1", "vehicle_id": 1}]
    review.write_json(dataset / "images/image_1.jpg", {"synthetic": True})
    frames = {"image_1": sha256(dataset / "images/image_1.jpg")}
    inner, split = {"primary": {"train": [1]}}, {"identities": {"train": [1]}, "protocols": {}}
    manifest = {"protected": {}, "source_sha256": {}, "seeds": [1, 2, 3],
                "budget": asdict(review.old.Budget()), "frames_sha256": continuation.digest(frames),
                "outer": split["identities"], "protocols": {}, "inner": inner,
                "runtime": {"device": "cpu"}, "base_recipe": asdict(ExperimentConfig()),
                "variants": {name: asdict(review.variants()[name]) for name in continuation.NAMES}}
    summaries = [{"variant": name, "seed": 1, "stop_step": 800, "lr_horizon": 1700,
                  "validation": None, "history": [{"validation": None}]} for name in continuation.NAMES]
    final = {"artifacts": {}, "selection": {"winner": "R1_resolution256",
             "aggregate": {name: {"step": 800} for name in continuation.NAMES}},
             "final": summaries, "outer": {name: evaluation_report(name, 1) for name in continuation.NAMES}}
    review.write_json(directory / "manifest.json", manifest)
    review.write_json(directory / "pilot.json", {"artifacts": {}})
    review.write_json(directory / "confirm_inner.json", {"artifacts": {}})
    review.write_json(directory / "final.json", final)
    review.write_json(directory / "final_selection.json", summaries)
    review.write_json(continuation.ARTIFACTS / "splits.json", split)
    for name in continuation.NAMES:
        review.write_json(directory / "final" / name / "seed_1/evaluation.json", final["outer"][name])
    phases = []
    monkeypatch.setattr(review, "verify_phase", lambda context, result, phase: phases.append(phase))
    monkeypatch.setattr(continuation, "read_rows", lambda _path: rows)
    monkeypatch.setattr(review.old, "load_masks", lambda *_args: ({}, inner))
    return directory, dataset, phases


def test_load_source_reconstructs_frozen_data_without_prepare_or_source_writes(source_files, monkeypatch):
    directory, dataset, phases = source_files
    before = {str(p): sha256(p) for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(review, "prepare", lambda *_args, **_kwargs: pytest.fail("Do not call source prepare"))
    context, final, protected = continuation.load_source(dataset=dataset)
    assert phases == ["pilot", "confirm_inner", "final"]
    assert context["seeds"] == (1, 2, 3) and context["budget"].max_steps == 1700
    assert context["variants"]["R1_resolution256"].size == 256
    assert final["selection"]["winner"] == "R1_resolution256"
    assert str(directory / "manifest.json") in protected
    assert before == {str(p): sha256(p) for p in directory.rglob("*") if p.is_file()}
    assert not (directory / ".lock").exists()


@pytest.mark.parametrize("mutation", ["outer_report", "final_selection", "winner", "stop", "horizon",
                                      "validation", "history_validation", "selected_step", "duplicate_seed"])
def test_load_source_rejects_inconsistent_source_recipe_or_report(source_files, mutation):
    directory, dataset, _ = source_files
    final = review.old.load_json(directory / "final.json")
    if mutation == "outer_report":
        final["outer"]["B0_control"]["conditions"]["original"]["raw"]["mAP_at_10"] = .1
    elif mutation == "final_selection":
        review.write_json(directory / "final_selection.json", [])
    elif mutation == "winner":
        final["selection"]["winner"] = "M3_triplet"
    elif mutation == "selected_step":
        final["selection"]["aggregate"]["B0_control"]["step"] = 1700
    elif mutation == "duplicate_seed":
        final["final"][1]["seed"] = 2
    else:
        summary = final["final"][0]
        if mutation == "stop":
            summary["stop_step"] = 1700
        elif mutation == "horizon":
            summary["lr_horizon"] = 800
        elif mutation == "validation":
            summary["validation"] = {"selected": True}
        else:
            summary["history"][0]["validation"] = {"selected": True}
        review.write_json(directory / "final_selection.json", final["final"])
    review.write_json(directory / "final.json", final)
    with pytest.raises(ValueError, match="Expected|recipe/report"):
        continuation.load_source(dataset=dataset)


@pytest.mark.parametrize("mutation", ["frame", "outer", "protocol", "inner", "seed_count", "budget"])
def test_load_source_rejects_changed_frozen_data_or_training_matrix(source_files, mutation):
    directory, dataset, _ = source_files
    if mutation == "frame":
        review.write_json(dataset / "images/image_1.jpg", {"changed": True})
    elif mutation in {"outer", "protocol"}:
        split = review.old.load_json(continuation.ARTIFACTS / "splits.json")
        split["identities" if mutation == "outer" else "protocols"] = {"changed": True}
        review.write_json(continuation.ARTIFACTS / "splits.json", split)
    else:
        manifest = review.old.load_json(directory / "manifest.json")
        if mutation == "inner":
            manifest["inner"] = {"changed": True}
        elif mutation == "seed_count":
            manifest["seeds"] = [1, 2]
        else:
            manifest["budget"]["max_steps"] = 800
        review.write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed|Expected"):
        continuation.load_source(dataset=dataset)


def test_notebook_compiles_keeps_guards_and_preserves_user_outputs():
    path = continuation.ROOT / "OSNet-AIN-x1.0/variant_17_final_seed_confirmation/confirm_final_seeds.ipynb"
    original = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:{index}", "exec")
    assert path.read_bytes() == original
    assert "ALLOW_CPU_TRAINING = False" in code
    assert "ALLOW_OUTER_EVALUATION" in code and "final_seed_confirmation" in code
    assert "confirm_inner" not in code
