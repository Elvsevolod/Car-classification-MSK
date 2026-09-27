"""Pure protocol tests; no training or changes to original split/evaluator."""
import copy

import pytest

from backend.evaluate import make_protocol
from backend.scoring import metrics, ranked_queries
from training.review_selection import make_draws, select_shared_steps, summarize_fixed


NAMES = ["B0_control", "R1_resolution256"]
SEEDS = [11, 12, 13]
STEPS = [0, 200, 400]


def primary_runs():
    values = {"B0_control": [[.5, .8, .7]] * 3,
              "R1_resolution256": [[.9, .6, .7], [.5, .9, .7], [.5, .6, .9]]}
    return [{"variant": name, "seed": seed,
             "history": [{"step": step, "validation": {"mean_map": score, "draws": []}}
                         for step, score in zip(STEPS, curve)]}
            for name in NAMES for seed, curve in zip(SEEDS, values[name])]


def test_selects_common_step_not_average_of_seed_maxima():
    runs = primary_runs()
    original = copy.deepcopy(runs)
    result = select_shared_steps(runs, NAMES, SEEDS, STEPS)
    assert result["winner"] == "B0_control"
    assert result["aggregate"]["B0_control"]["step"] == 200
    candidate = result["aggregate"]["R1_resolution256"]
    assert candidate["step"] == 400
    assert candidate["mean"] == pytest.approx((.7 + .7 + .9) / 3)
    assert candidate["seed_scores"] == {"11": .7, "12": .7, "13": .9}
    assert candidate["std"] == pytest.approx(.2 / 3 ** .5)
    assert runs == original


def test_ties_choose_earliest_step_and_baseline():
    runs = primary_runs()
    for run in runs:
        for item in run["history"]:
            item["validation"]["mean_map"] = .8
    result = select_shared_steps(runs, list(reversed(NAMES)), SEEDS, list(reversed(STEPS)))
    assert result["winner"] == "B0_control"
    assert all(item["step"] == 0 for item in result["aggregate"].values())


@pytest.mark.parametrize("problem", ["missing_seed", "duplicate_seed", "unexpected_variant", "missing_step",
                                    "duplicate_step", "unexpected_step", "nan", "inf", "negative", "too_large"])
def test_rejects_incomplete_or_invalid_primary(problem):
    runs = primary_runs()
    if problem == "missing_seed":
        runs.pop()
    elif problem == "duplicate_seed":
        runs.append(copy.deepcopy(runs[0]))
    elif problem == "unexpected_variant":
        runs[0]["variant"] = "unknown"
    elif problem == "missing_step":
        runs[0]["history"].pop()
    elif problem == "duplicate_step":
        runs[0]["history"].append(copy.deepcopy(runs[0]["history"][0]))
    elif problem == "unexpected_step":
        runs[0]["history"][0]["step"] = 1
    else:
        runs[0]["history"][0]["validation"]["mean_map"] = {
            "nan": float("nan"), "inf": float("inf"), "negative": -.1, "too_large": 1.1}[problem]
    with pytest.raises(ValueError):
        select_shared_steps(runs, NAMES, SEEDS, STEPS)


@pytest.mark.parametrize("steps", [[], [0, 0], [-1, 200, 400], [0., 200, 400]])
def test_rejects_invalid_boundary_grid(steps):
    with pytest.raises(ValueError):
        select_shared_steps(primary_runs(), NAMES, SEEDS, steps)


def fixed_runs(selection):
    return [{"variant": name, "seed": seed, "stop_step": selection["aggregate"][name]["step"],
             "validation": {"mean_map": .7 + index * .02 + (name != "B0_control") * .1, "draws": []}}
            for name in NAMES for index, seed in enumerate(SEEDS)]


def test_fixed_summary_preserves_selection_and_pairs_seed_deltas():
    selection = select_shared_steps(primary_runs(), NAMES, SEEDS, STEPS)
    original = copy.deepcopy(selection)
    result = summarize_fixed(fixed_runs(selection), selection, SEEDS)
    assert selection == original
    assert selection["winner"] == "B0_control"  # Alternate candidate wins, but cannot reselect.
    delta = result["paired_deltas_vs_control"]["R1_resolution256"]
    assert delta["mean"] == pytest.approx(.1)
    assert delta["std"] == pytest.approx(0.)
    assert list(delta["seed_deltas"]) == [str(seed) for seed in SEEDS]


@pytest.mark.parametrize("problem", ["wrong_stop", "missing_seed", "duplicate_seed", "nan"])
def test_fixed_summary_requires_exact_frozen_stop_and_seeds(problem):
    selection = select_shared_steps(primary_runs(), NAMES, SEEDS, STEPS)
    runs = fixed_runs(selection)
    if problem == "wrong_stop":
        runs[0]["stop_step"] += 1
    elif problem == "missing_seed":
        runs.pop()
    elif problem == "duplicate_seed":
        runs.append(runs[0])
    else:
        runs[0]["validation"]["mean_map"] = float("nan")
    with pytest.raises(ValueError):
        summarize_fixed(runs, selection, SEEDS)


def protocol_rows():
    return [{"image_id": f"{identity}_{camera}_{frame}", "vehicle_id": identity, "camera_id": camera}
            for identity in range(6) for camera in range(2) for frame in range(3)]


def test_protocol_draws_preserve_normal_split_and_add_only_known_junk():
    rows = protocol_rows()
    original = copy.deepcopy(rows)
    by_id = {row["image_id"]: row for row in rows}
    draws = make_draws(rows, [0, 1, 2, 3, 4], [17, 29])
    for seed in [17, 29]:
        query, gallery = make_protocol(rows, [0, 1, 2, 3, 4], seed)
        regular, diagnostic = draws[f"regular_{seed}"], draws[f"same_camera_junk_{seed}"]
        assert regular["query_ids"] == diagnostic["query_ids"] == [q["image_id"] for q in query]
        assert regular["gallery_ids"] == [g["image_id"] for g in gallery]
        assert regular["selection_eligible"] is True
        assert diagnostic["selection_eligible"] is False
        assert not set(diagnostic["query_ids"]) & set(diagnostic["gallery_ids"])
        assert len(diagnostic["gallery_ids"]) == len(set(diagnostic["gallery_ids"]))
        known = {g["vehicle_id"] for g in gallery}
        assert {by_id[i]["vehicle_id"] for i in diagnostic["gallery_ids"]} == known
        additions = set(diagnostic["gallery_ids"]) - set(regular["gallery_ids"])
        assert len(additions) == len(known) * 2
        for image_id in additions:
            row = by_id[image_id]
            q = next(q for q in query if q["vehicle_id"] == row["vehicle_id"])
            assert row["camera_id"] == q["camera_id"]
    assert rows == original
    assert make_draws(rows, [0, 1, 2, 3, 4], [17, 29]) == draws


@pytest.mark.parametrize("problem", ["missing_identity", "duplicate_image", "no_known"])
def test_protocol_rejects_missing_or_unscorable_data(problem):
    rows = protocol_rows()
    identities = [0, 1, 2, 3, 4]
    if problem == "missing_identity":
        identities.append(99)
    elif problem == "duplicate_image":
        rows.append(rows[0])
    else:
        identities = [0]
    with pytest.raises(ValueError):
        make_draws(rows, identities, [17])


def test_official_junk_order_remains_top10_before_filtering():
    query = [{"image_id": "q", "vehicle_id": 1, "camera_id": 1}]
    gallery = [{"image_id": "junk", "vehicle_id": 1, "camera_id": 1}]
    gallery += [{"image_id": f"negative_{i}", "vehicle_id": 2, "camera_id": 2} for i in range(9)]
    gallery += [{"image_id": "positive", "vehicle_id": 1, "camera_id": 2}]
    embeddings = {row["image_id"]: [1., 0.] for row in query + gallery}
    result = metrics(ranked_queries(query, gallery, embeddings), .5)
    assert result["mAP_at_10"] == 0.
    assert result["full_mAP"] == pytest.approx(.1)
    assert result["TP"] == 1  # Official refusal logic accepts same-camera identity.
