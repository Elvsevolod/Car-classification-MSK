"""Small synthetic checks; no organizer training, no production weights mutated."""
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from training import quality_verifier as head
from training import quality_clock as clock
from training import quality_experiment as experiment
from training import overnight_training as training
from training.local_verification import mix_topk_scores
from tests.test_overnight_training import context, cpu_threads, equal_state


def tokens(count, seed=7):
    x = np.random.default_rng(seed).normal(size=(count, 16, 512)).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_training_rows_are_train_only_and_camera_diverse():
    rows = [{"image_id": f"{i}_{c}_{j}", "vehicle_id": i, "camera_id": c}
            for i in range(5) for c in (1, 2) for j in range(3)]
    selected = head.training_rows(rows, {1, 2}, seed=123, per_identity=4)
    assert len(selected) == 8
    assert {r["vehicle_id"] for r in selected} == {1, 2}
    for identity in (1, 2):
        assert {r["camera_id"] for r in selected if r["vehicle_id"] == identity} == {1, 2}
    assert selected == head.training_rows(list(reversed(rows)), {1, 2}, seed=123, per_identity=4)


def test_mining_ignores_junk_and_uses_real_hard_negative_pool():
    rows = [{"image_id": f"{i}_{c}_{j}", "vehicle_id": i, "camera_id": c}
            for i in range(3) for c in (1, 2) for j in range(2)]
    x = np.random.default_rng(2).normal(size=(len(rows), 5)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    config = replace(head.HeadConfig(), pool=5)
    pairs, labels = head.mine_pairs(rows, x, config)
    assert set(labels) == {0., 1.}
    for (i, j), target in zip(pairs, labels):
        assert i != j
        if target:
            assert rows[i]["vehicle_id"] == rows[j]["vehicle_id"]
            assert rows[i]["camera_id"] != rows[j]["camera_id"]
        else:
            assert rows[i]["vehicle_id"] != rows[j]["vehicle_id"]
            order = np.argsort(-(x @ x[i]), kind="stable")
            assert j in order[order != i][:config.pool]


def test_same_camera_same_id_is_not_a_negative():
    rows = [{"image_id": str(i), "vehicle_id": i // 2, "camera_id": 1} for i in range(6)]
    with pytest.raises(ValueError, match="cross-camera"):
        head.mine_pairs(rows, np.eye(6, dtype=np.float32), head.HeadConfig())


def test_pair_descriptor_is_symmetric_finite_and_query_independent():
    x = tokens(5)
    a = head.pair_features(x[0], x[1:], np.array([.1, .2, .3, .4], np.float32))
    b = head.pair_features(x[1], x[:1], np.array([.1], np.float32))
    np.testing.assert_allclose(a[0], b[0], atol=2e-7, rtol=0)
    assert a.shape == (4, 289)
    solo = head.pair_features(x[0], x[3:4], np.array([.3], np.float32))
    np.testing.assert_array_equal(a[2], solo[0])
    zero = head.pair_features(np.zeros((16, 512), np.float32), x[:1], np.array([0.], np.float32))
    assert np.isfinite(zero).all()
    with pytest.raises(ValueError, match="unit length"):
        head.pair_features(x[0] * 2, x[:1], np.array([0.], np.float32))


@pytest.mark.parametrize("change", [{"epochs": 0}, {"pool": 0}, {"images_per_identity": 1},
                                    {"learning_rate": float("nan")}, {"weight_decay": -1}])
def test_invalid_head_config(change):
    with pytest.raises(ValueError):
        replace(head.HeadConfig(), **change).validate()


def test_head_checkpoint_resume_training_normalizer_and_corruption(tmp_path, monkeypatch, cpu_threads):
    features = np.random.default_rng(4).normal(size=(32, 289)).astype(np.float32)
    labels = np.array([0., 1.] * 16, np.float32)
    config = replace(head.HeadConfig(), epochs=3, batch_size=8)
    uninterrupted = head.fit_head(features, labels, tmp_path / "complete", seed=3, signature="train", config=config)
    original = head.save_checkpoint
    def interrupt(path, value):
        original(path, value)
        if value["epoch"] == 1:
            raise KeyboardInterrupt()
    monkeypatch.setattr(head, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        head.fit_head(features, labels, tmp_path / "resume", seed=3, signature="train", config=config)
    monkeypatch.setattr(head, "save_checkpoint", original)
    resumed = head.fit_head(features, labels, tmp_path / "resume", seed=3, signature="train", config=config)
    a, b = head.load_head(uninterrupted), head.load_head(resumed)
    equal_state(a.state_dict(), b.state_dict())
    torch.testing.assert_close(a.center, torch.from_numpy(features).mean(0), atol=0, rtol=0)
    before = Path(resumed["path"]).read_bytes()
    again = head.fit_head(features, labels, tmp_path / "resume", seed=3, signature="train", config=config)
    assert before == Path(again["path"]).read_bytes()
    with pytest.raises(ValueError, match="changed"):
        head.fit_head(features + .01, labels, tmp_path / "resume", seed=3, signature="train", config=config)
    Path(resumed["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        head.load_head(resumed)


def test_head_inference_has_no_batch_dependence_or_candidate_mutation(cpu_threads):
    x = tokens(6)
    torch.manual_seed(3)
    model = head.PairHead().eval()
    cosine = np.array([.9, .8, .7, .6, .5], np.float32)
    scores = head.score_pairs(model, x[0], x[1:], cosine)
    solo = head.score_pairs(model, x[0], x[3:4], cosine[2:3])
    assert scores[2] == pytest.approx(solo[0], abs=1e-7)
    reverse = head.score_pairs(model, x[0], x[:0:-1], cosine[::-1])
    np.testing.assert_allclose(scores, reverse[::-1], atol=1e-7, rtol=0)
    raw_order = np.arange(5)[None]
    base = cosine[None].copy()
    mixed = mix_topk_scores(base, raw_order, scores[None], top_k=3, weight=.1)
    assert np.array_equal(raw_order, np.arange(5)[None])
    assert np.array_equal(base, cosine[None])
    assert np.array_equal(mixed["order"][:, 3:], raw_order[:, 3:])


def test_fixed_clock_jobs_and_no_external_data(context):
    jobs = clock.jobs()
    assert {(j.stop_step, j.lr_horizon) for j in jobs.values()} == {(800, 1700), (800, 800), (1700, 1700)}
    assert all(j.base_variant == "R1_resolution256" and j.relation_weight == 0 and j.ce_scale == 1 for j in jobs.values())
    a = clock.job_context(context, "R1_control")
    b = clock.job_context(context, "R1_cosine800")
    assert a["signature"] != b["signature"]
    assert context["budget"].max_steps == 8
    assert b["budget"].max_steps == 800


@pytest.mark.parametrize("count", [2, 31, 32, 33, 64, 65, 100])
def test_bn_batches_cover_all_train_images_without_singletons(count):
    batches = clock.bn_batches(count)
    assert [i for b in batches for i in b] == list(range(count))
    assert all(len(b) >= 2 for b in batches)


def test_average_parameters_does_not_average_bn_buffers_and_reestimates_train_only(cpu_threads):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Dropout(.9))
    states = []
    for value in (1., 3., 5.):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(value)
            model[1].running_mean.fill_(value * 10)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    clock.average_parameters(model, states)
    assert all(torch.allclose(p, torch.full_like(p, 3.)) for p in model.parameters())
    assert torch.all(model[1].running_mean == 50)
    modes = []
    hook = model.register_forward_pre_hook(lambda m, args: modes.append((m[1].training, m[2].training)))
    count = clock.recalibrate_bn(model, [torch.ones(4, 3), torch.ones(3, 3)], torch.device("cpu"))
    hook.remove()
    assert count == 7 and modes == [(True, False), (True, False)]
    assert not any(m.training for m in model.modules())
    assert torch.allclose(model[1].running_mean, torch.full((3,), 12.))
    assert all(torch.allclose(p, torch.full_like(p, 3.)) for p in model.parameters())


def report(score):
    return {"mean_map": score, "draws": {"draw": {"per_query": {"q": {"vehicle_id": 7, "ap": score}}}}}


def test_pair_gate_all_seeds_and_fixed_alternate_choice():
    base = {str(s): report(.8) for s in (1, 2, 3)}
    matrix = {"baseline": base, "head_w05": {str(s): report(.804) for s in (1, 2, 3)}, "head_w10": base}
    decision = experiment.select_head(matrix)
    assert decision["selected"] == "head_w05"
    assert decision["comparisons"]["head_w05"]["conditional_bootstrap"]["identities"] == 1
    matrix["head_w05"]["3"] = None
    assert experiment.select_head(matrix)["selected"] is None


def test_averaging_needs_gain_over_bn_control_not_only_r1():
    pilots = {n: report(.8) for n in clock.CASES}
    pilots["R1_full1700_bn"] = report(.805)
    pilots["R1_full1700_avg"] = report(.806)
    selection = experiment.select_clock(pilots)
    assert not selection["comparisons"]["R1_full1700_avg"]["passed"]
    assert selection["selected"] == "R1_full1700_bn"
    pilots["R1_cosine800"] = None
    assert experiment.select_clock(pilots)["selected"] is None


def test_no_candidate_can_win_without_positive_mean_and_two_seeds():
    matrix = {"baseline": {str(s): report(.8) for s in (1, 2, 3)},
              "new": {"1": report(.82), "2": report(.795), "3": report(.795)}}
    assert not experiment.paired_comparison(matrix, "new")["passed"]


def test_final_fold_forbidden_by_training_boundary(context):
    with pytest.raises(ValueError, match="inner"):
        training.fold_training_ids(context, "final")
    context["manifest"]["inner"]["primary"]["train"].append(5)
    with pytest.raises(ValueError, match="holdout"):
        training.fold_training_ids(context, "primary")


def test_clock_training_matched_prefix_and_derived_bn_control(context, monkeypatch, tmp_path):
    jobs = {n: replace(j, stop_step=8 if n == "R1_full1700" else 6,
                       lr_horizon=6 if n == "R1_cosine800" else 8) for n, j in clock.jobs().items()}
    monkeypatch.setattr(clock, "jobs", lambda: jobs)
    summaries = {n: training.fit_job(clock.job_context(context, n), j, 1) for n, j in jobs.items()}
    def state(name, step):
        path = context["output"] / summaries[name]["checkpoints"][str(step)]["path"]
        return torch.load(path, map_location="cpu", weights_only=True)["model"]
    equal_state(state("R1_control", 6), state("R1_full1700", 6))
    assert any(not torch.equal(state("R1_control", 6)[k], state("R1_cosine800", 6)[k])
               for k in state("R1_control", 6))
    seen = []
    class BNDataset:
        def __init__(self, rows, *_args, **kwargs):
            self.rows = rows
            seen.extend(r["vehicle_id"] for r in rows)
            assert kwargs["augment"] is False
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            return torch.tensor([r["vehicle_id"], i, r["camera_id"], 1.], dtype=torch.float32), 0, r["image_id"]
    monkeypatch.setattr(clock, "AblationDataset", BNDataset)
    for case in ("R1_full1700_bn", "R1_full1700_avg"):
        directory = context["output"] / case
        directory.mkdir()
        model, _, meta = clock.derived_model(context, summaries["R1_full1700"], case, directory)
        assert meta["bn_train_images"] == 4 and meta["bn_reestimated"]
        assert len(meta["sources"]) == (3 if case.endswith("avg") else 1)
        assert not any(m.training for m in model.modules())
    assert set(seen) == {0, 1}
    with pytest.raises(ValueError, match="new run"):
        clock.derived_model(context, summaries["R1_full1700"], "R1_full1700_avg", tmp_path / "outside")
    assert not (tmp_path / "outside").exists()


def test_full_head_queue_resume_and_independent_failures(context, monkeypatch):
    context["manifest"].update(source_sha256={}, quality_plan={
        "run_head": True, "run_clock": False, "auto_confirm": True,
        "head": asdict(replace(head.HeadConfig(), epochs=2))})
    context["protected"] = {}
    context["manifest"]["draws"] = {"primary": {"draw": {
        "selection_eligible": True, "query_ids": ["i2_1", "i3_1"], "gallery_ids": ["i2_2", "i3_2"]}}}
    monkeypatch.setattr(experiment.old.review, "check_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment.old.review, "check_other_runs", lambda *args: None)
    def features(ctx, fold, seed, role):
        identities = {0, 1} if role == "train" else {2, 3}
        rows = [r for r in ctx["rows"] if r["vehicle_id"] in identities]
        vectors = np.zeros((len(rows), 512), np.float32)
        for i, row in enumerate(rows):
            vectors[i, row["vehicle_id"]] = 1.
        return rows, vectors, np.repeat(vectors[:, None], 16, axis=1)
    monkeypatch.setattr(experiment, "cached_features", features)
    fit = experiment.train_head
    def fail_one(ctx, fold, seed, directory):
        if seed == 2:
            raise ValueError("synthetic recoverable failure")
        return fit(ctx, fold, seed, directory)
    monkeypatch.setattr(experiment, "train_head", fail_one)
    failed = experiment.run(context)
    assert failed["status"] == "complete_with_failures"
    assert failed["head"]["selection"]["selected"] is None
    assert not (context["output"] / "tasks/head_selection/complete.json").exists()
    # Other seed tasks completed, so a retry must not retrain them.
    retrained = []
    def retry(ctx, fold, seed, directory):
        retrained.append(seed)
        return fit(ctx, fold, seed, directory)
    monkeypatch.setattr(experiment, "train_head", retry)
    complete = experiment.run(context)
    assert retrained == [2]
    assert complete["status"] == "complete" and complete["protected_unchanged"]
    assert complete["head"]["selection"]["selected"] == "baseline"
    assert complete["head"]["alternate"] == {}
    again = experiment.run(context)
    assert all(e["status"] == "cached" for e in again["events"])
    assert retrained == [2]


def test_head_alternate_only_receives_frozen_primary_winner(context, monkeypatch, tmp_path):
    context["manifest"]["quality_plan"] = {"auto_confirm": True}
    calls = []
    class Queue:
        def __init__(self):
            self.context = context
        def task(self, name, compute):
            return compute(tmp_path / name)
    monkeypatch.setattr(experiment, "train_head", lambda ctx, fold, seed, d: {"seed": seed})
    def trial(ctx, fold, seed, summary, directory, names):
        calls.append((fold, seed, list(names)))
        return {n: report(.805 if n == "head_w05" else .8) for n in names}
    monkeypatch.setattr(experiment, "head_trial", trial)
    result = experiment.head_queue(Queue())
    assert result["selection"]["selected"] == "head_w05"
    assert result["alternate_gate"]["passed"]
    assert len(calls) == 6
    assert all(names == ["baseline", "head_w05"] for fold, _, names in calls if fold == "alternate")


def test_clock_confirmation_only_runs_selected_recipe_and_controls(context, monkeypatch, tmp_path):
    context["manifest"]["quality_plan"] = {"auto_confirm": True}
    trained, evaluated = [], []
    class Queue:
        def __init__(self):
            self.context = context
        def task(self, name, compute):
            return compute(tmp_path / name)
    def fit(ctx, job, seed, fold):
        trained.append((fold, seed, job.name))
        return {"seed": seed, "fold": fold}
    def evaluate(ctx, summary, case, directory):
        evaluated.append((summary["fold"], summary["seed"], case))
        return report(.81 if case == "R1_cosine800" else .8)
    monkeypatch.setattr(training, "fit_job", fit)
    monkeypatch.setattr(experiment, "evaluate_clock", evaluate)
    result = experiment.clock_queue(Queue())
    assert result["selection"]["selected"] == "R1_cosine800"
    assert result["alternate_gate"]["R1_control"]["passed"]
    assert trained.count(("primary", 1, "R1_full1700")) == 1
    assert {case for fold, _, case in evaluated if fold == "alternate"} == {"R1_control", "R1_cosine800"}
    assert {fold for fold, _, _ in trained} == {"primary", "alternate"}
