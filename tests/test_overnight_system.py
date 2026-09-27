import json

import numpy as np
import pytest
import torch

from training import overnight_system as system
from training import overnight_evaluation as evaluation
from training import overnight_training as training
from training import retrieval_policy as policy


def context(tmp_path, **flags):
    plan = {"training": False, "confirmation": True, "masks": False, "benchmark": False, **flags}
    return {"output": tmp_path, "signature": "synthetic", "device": torch.device("cpu"),
            "seeds": (11, 12, 13), "source": {},
            "manifest": {"source_sha256": {}, "system_plan": plan}}


def test_queue_resume_preserves_result_and_skips_compute(tmp_path):
    queue = system.Queue(context(tmp_path))
    assert queue.task("one", lambda directory: {"value": 17}) == {"value": 17}
    resume = system.Queue(context(tmp_path))
    resume.deadline = -1  # Cached completions remain readable even at the boundary.
    assert resume.task("one", lambda directory: pytest.fail("must not recompute")) == {"value": 17}
    assert resume.events == [{"task": "one", "status": "cached"}]


def test_queue_corruption_stops_instead_of_silently_recomputing(tmp_path):
    queue = system.Queue(context(tmp_path))
    queue.task("one", lambda directory: {"value": 17})
    (tmp_path / "tasks/one/result.json").write_text('{"value":18}')
    with pytest.raises(system.IntegrityError, match="Changed task artifact"):
        queue.task("one", lambda directory: {"value": 19})


def test_queue_configuration_change_is_not_resumed(tmp_path):
    system.Queue(context(tmp_path)).task("one", lambda directory: {"value": 1})
    changed = {**context(tmp_path), "signature": "changed"}
    with pytest.raises(system.IntegrityError, match="configuration changed"):
        system.Queue(changed).task("one", lambda directory: {})


def test_queue_failure_does_not_block_independent_task_or_create_receipt(tmp_path):
    queue = system.Queue(context(tmp_path))

    def fails(directory):
        raise RuntimeError("synthetic failure")

    assert queue.task("failed", fails) is None
    assert not (tmp_path / "tasks/failed/complete.json").exists()
    assert queue.task("next", lambda directory: {"ok": True}) == {"ok": True}
    assert [e["status"] for e in queue.events] == ["failed", "complete"]
    assert queue.task("failed", lambda directory: {"retried": True}) == {"retried": True}


def test_queue_budget_stops_before_new_task(tmp_path):
    queue = system.Queue(context(tmp_path))
    queue.deadline = -1
    with pytest.raises(system.BudgetReached):
        queue.task("too_late", lambda directory: pytest.fail("must not compute"))
    assert not (tmp_path / "tasks/too_late").exists()


def test_queue_default_has_no_time_limit(tmp_path):
    queue = system.Queue(context(tmp_path))
    assert np.isinf(queue.deadline)
    assert queue.task("unlimited", lambda directory: {"complete": True}) == {"complete": True}


def test_queue_pause_records_resumable_status(tmp_path):
    queue = system.Queue(context(tmp_path))

    def paused(directory):
        raise system.BudgetReached("saved boundary")

    with pytest.raises(system.BudgetReached):
        queue.task("paused", paused)
    assert json.loads((tmp_path / "tasks/paused/status.json").read_text())["status"] == "paused"
    assert queue.events[-1]["status"] == "paused"
    assert not (tmp_path / "tasks/paused/complete.json").exists()


def test_queue_integrity_error_is_not_swallowed(tmp_path):
    queue = system.Queue(context(tmp_path))

    def corrupt(directory):
        raise system.IntegrityError("bad source")

    with pytest.raises(system.IntegrityError, match="bad source"):
        queue.task("corrupt", corrupt)


def tiny_local_protocol():
    query = [{"image_id": f"q{i}", "vehicle_id": i + 1, "camera_id": 0} for i in range(3)]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i % 2 + 1, "camera_id": 1} for i in range(12)]
    rows = query + gallery
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(len(rows), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    tokens = rng.normal(size=(len(rows), 4, 8)).astype(np.float32)
    return query, gallery, rows, vectors, tokens


def test_local_trial_real_ranker_and_local_scorer_integration(tmp_path, monkeypatch):
    query, gallery, rows, vectors, tokens = tiny_local_protocol()
    ctx = context(tmp_path)
    ctx["manifest"]["draws"] = {"primary": {"regular0": {"selection_eligible": True,
        "query_ids": [r["image_id"] for r in query], "gallery_ids": [r["image_id"] for r in gallery]}}}
    monkeypatch.setattr(system, "cached_features", lambda *args, **kwargs: (rows, vectors, tokens))
    result = system.local_trial(ctx, "primary", 11, evaluation.LOCAL_GRID, tmp_path / "trial")
    assert set(result) == {c["name"] for c in evaluation.LOCAL_GRID}
    baseline, _ = evaluation.rank_with_scores(vectors[:3], vectors[3:])
    expected = policy.evaluate(query, gallery, baseline, 2., "raw_top1")["ranking"]
    assert result["baseline"]["regular0"]["ranking"] == expected
    for reports in result.values():
        report = reports["regular0"]
        assert report["oracle"]["summary"]["baseline_mAP10"] == expected["mAP@10"]
        assert report["oracle"]["summary"]["comparison_mAP10"] == report["ranking"]["mAP@10"]
    assert (tmp_path / "trial/baseline/regular0.json").exists()


def test_alternate_confidence_requires_frozen_primary_choice(tmp_path):
    with pytest.raises(system.IntegrityError, match="Freeze confidence"):
        system.confidence_trial(context(tmp_path), "alternate", tmp_path)


def test_confidence_trial_local_support_and_frozen_alternate_csv(tmp_path, monkeypatch):
    query = [{"image_id": f"q{i}_{j}", "vehicle_id": i, "camera_id": 0}
             for i in (1, 2, 3, 4, 91, 92, 93, 94) for j in range(2)]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i % 4 + 1, "camera_id": 1} for i in range(12)]
    rows = query + gallery
    rng = np.random.default_rng(19)
    features = rng.normal(size=(len(rows), 8)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    tokens = rng.normal(size=(len(rows), 4, 8)).astype(np.float32)
    ctx = context(tmp_path)
    ctx["rows"] = rows
    draw = {"regular0": {"selection_eligible": True, "query_ids": [r["image_id"] for r in query],
                         "gallery_ids": [r["image_id"] for r in gallery]}}
    ctx["manifest"]["draws"] = {"primary": draw, "alternate": draw}
    monkeypatch.setattr(system, "source_vectors", lambda *args: {r["image_id"]: v for r, v in zip(rows, features)})
    monkeypatch.setattr(system, "cached_features", lambda *args, **kwargs: (rows, features, tokens))
    primary = system.confidence_trial(ctx, "primary", tmp_path / "primary")
    assert len(primary) == 12
    assert "R1_first_seed/cosine_plus_local/raw_top1" in primary
    assert "R1_equal3/cosine_minus_disagreement/raw_top1" in primary
    selected = system.choose_confidence(primary)["selected"]
    alternate = system.confidence_trial(ctx, "alternate", tmp_path / "alternate", selected)
    assert set(alternate) == set(selected)
    for name, report in alternate.items():
        assert not set(report["calibration_identities"]) & set(report["evaluation_identities"])
        assert (tmp_path / "alternate" / name / "csv/candidates.csv").exists()


def test_training_pilot_must_beat_family_control_and_r1():
    values = {"R1_control": .8, "K2_ce1": .7, "K2_ce_sqrt544": .79,
              "R1_kd01": .81, "R1_kd1": .805}
    result = system.choose_training_pilots({n: {"mean_map": v} for n, v in values.items()})
    assert result["selected"] == ["R1_kd01"]
    assert result["families"]["ce"]["passed"] is False
    assert result["families"]["distillation"]["passed"] is True


@pytest.mark.parametrize("gains, passed", [([.01, .01, -.001], True), ([.02, -.001, -.001], False),
                                         ([.001, .001, .001], False)])
def test_confirmation_requires_three_seed_mean_and_two_positive(gains, passed):
    matrix = {"control": {str(s): {"mean_map": .8} for s in range(3)},
              "candidate": {str(s): {"mean_map": .8 + gains[s]} for s in range(3)}}
    result = system.confirmation_gate(matrix, "candidate", ["control"])
    assert result["complete"] is True
    assert result["passed"] is passed
    matrix["candidate"]["2"] = None
    assert system.confirmation_gate(matrix, "candidate", ["control"])["complete"] is False


def test_failed_pilot_does_not_freeze_selection_and_resume_retries(tmp_path, monkeypatch):
    ctx = context(tmp_path, confirmation=False)
    monkeypatch.setattr(system, "source_summary", lambda ctx, fold, seed: {"fold": fold, "seed": seed})
    failed_once = {"value": True}

    def fit(ctx, job, seed, *, fold, **kwargs):
        if job.name == "K2_ce_sqrt544" and failed_once["value"]:
            failed_once["value"] = False
            raise RuntimeError("synthetic training failure")
        return {"name": job.name, "seed": seed, "fold": fold}

    monkeypatch.setattr(training, "fit_job", fit)
    monkeypatch.setattr(system, "evaluate_trained", lambda ctx, summary, directory: {"mean_map": .8})
    first = system.training_queue(system.Queue(ctx))
    assert first["selection"] is None
    assert not (tmp_path / "tasks/training_pilot_selection/complete.json").exists()
    second = system.training_queue(system.Queue(ctx))
    assert second["selection"] is not None
    assert (tmp_path / "tasks/training_pilot_selection/complete.json").exists()


@pytest.mark.parametrize("pass_primary", [False, True])
def test_training_alternate_waits_for_complete_three_seed_primary_gate(tmp_path, monkeypatch, pass_primary):
    ctx = context(tmp_path)
    monkeypatch.setattr(system, "source_summary", lambda ctx, fold, seed: {"fold": fold, "seed": seed})
    calls = []

    def fit(ctx, job, seed, *, fold, **kwargs):
        calls.append((job.name, seed, fold))
        if fold == "alternate":
            assert (tmp_path / "tasks/primary_confirmation_gate_R1_kd01/complete.json").exists()
        return {"name": job.name, "seed": seed, "fold": fold}

    def evaluate(ctx, summary, directory):
        value = {"R1_control": .8, "K2_ce1": .7, "K2_ce_sqrt544": .79,
                 "R1_kd01": .82, "R1_kd1": .81}[summary["name"]]
        if not pass_primary and summary["name"] == "R1_kd01" and summary["seed"] != ctx["seeds"][0]:
            value = .799
        return {"mean_map": value}

    monkeypatch.setattr(training, "fit_job", fit)
    monkeypatch.setattr(system, "evaluate_trained", evaluate)
    result = system.training_queue(system.Queue(ctx))
    assert result["selection"]["selected"] == ["R1_kd01"]
    chosen = result["confirmation"]["R1_kd01"]
    assert chosen["primary_gate"]["passed"] is pass_primary
    alternate = [x for x in calls if x[2] == "alternate"]
    assert len(alternate) == (6 if pass_primary else 0)
    assert bool(chosen["alternate"]) is pass_primary


def test_run_all_phase_order_and_cached_second_run(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    events = []
    chosen = evaluation.LOCAL_GRID[1]["name"]
    monkeypatch.setattr(system.review, "check_inputs", lambda *args, **kwargs: None)

    def local(ctx, fold, seed, configs, directory, **kwargs):
        events.append(("local", fold, seed))
        if fold == "alternate":
            assert (tmp_path / "tasks/local_selection/complete.json").exists()
            assert {c["name"] for c in configs} == {"baseline", chosen}
        return {c["name"]: {"regular0": {"ranking": {"mAP@10": .81 if c["name"] == chosen else .8},
                "diagnostics": {"per_query": {"q": {"vehicle_id": 1,
                    "ranking": {"ap": .81 if c["name"] == chosen else .8}}}}}}
                for c in configs}

    def confidence(ctx, fold, directory, frozen_choices=None):
        events.append(("confidence", fold))
        if fold == "alternate":
            assert frozen_choices
            assert (tmp_path / "tasks/confidence_selection/complete.json").exists()
        names = [f"{system_name}/{signal}/{candidate}" for system_name in ("R1_first_seed", "R1_equal3")
                 for signal in ("cosine", "cosine_plus_margin") for candidate in policy.CANDIDATES]
        return {name: {"evaluation": {"candidates": {"C": .8 if "margin" in name else .7, "F1": .8, "TNR": .7}},
                       "acceptance": {"coverage": .8, "accepted_error_rate": .1}}
                for name in names if frozen_choices is None or name in frozen_choices}

    monkeypatch.setattr(system, "local_trial", local)
    monkeypatch.setattr(system, "confidence_trial", confidence)
    first = system.run(ctx)
    assert first["status"] == "complete"
    assert first["outer_evaluated"] is False and first["promoted"] is False
    assert events[:3] == [("local", "primary", seed) for seed in ctx["seeds"]]
    assert events[3:6] == [("local", "alternate", seed) for seed in ctx["seeds"]]
    assert events[6:] == [("confidence", "primary"), ("confidence", "alternate")]
    assert (tmp_path / "REPORT.md").exists()
    before = len(events)
    second = system.run(ctx)
    assert second["status"] == "complete"
    assert len(events) == before
    assert all(e["status"] == "cached" for e in second["events"])
    assert json.loads((tmp_path / "results.json").read_text())["signature"] == "synthetic"
