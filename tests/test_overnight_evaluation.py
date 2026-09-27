import numpy as np
import pytest

from training import overnight_evaluation as evaluation
from training import retrieval_policy as policy


def protocol():
    gallery = [{"image_id": f"g{i}", "vehicle_id": i % 4 + 1, "camera_id": 1} for i in range(12)]
    query = [{"image_id": f"q{v}_{j}", "vehicle_id": v, "camera_id": 0}
             for v in (1, 2, 3, 4, 90, 91, 92, 93) for j in range(2)]
    rng = np.random.default_rng(19)
    qv, gv = rng.normal(size=(len(query), 8)), rng.normal(size=(len(gallery), 8))
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    gv /= np.linalg.norm(gv, axis=1, keepdims=True)
    return query, gallery, qv, gv


def test_identity_calibration_partition_is_disjoint_and_query_order_independent():
    query, gallery, _, _ = protocol()
    cal, held = evaluation.identity_calibration_partition(query, gallery)
    ci, hi = ({query[int(i)]["vehicle_id"] for i in indices} for indices in (cal, held))
    assert len(ci) == len(hi) == 4
    assert not ci & hi
    assert set(cal) | set(held) == set(range(len(query)))
    assert len(ci & {1, 2, 3, 4}) == 2
    assert len(hi & {90, 91, 92, 93}) == 2
    reverse = list(reversed(query))
    ca, he = evaluation.identity_calibration_partition(reverse, gallery)
    assert {reverse[int(i)]["vehicle_id"] for i in ca} == ci
    assert {reverse[int(i)]["vehicle_id"] for i in he} == hi


def test_identity_partition_rejects_insufficient_unknown_identities():
    query, gallery, _, _ = protocol()
    with pytest.raises(ValueError, match=">=2"):
        evaluation.identity_calibration_partition([q for q in query if q["vehicle_id"] < 90], gallery)


def test_threshold_is_not_selected_from_heldout_confidence():
    query, gallery, qv, gv = protocol()
    ranking, _ = evaluation.rank_with_scores(qv, gv)
    confidence = np.linspace(.1, .9, len(query))
    first = evaluation.calibrate_inner_confidence(query, gallery, ranking, confidence, "raw_top1")
    _, held = evaluation.identity_calibration_partition(query, gallery)
    modified = confidence.copy()
    modified[held] = np.linspace(-100, 100, len(held))
    second = evaluation.calibrate_inner_confidence(query, gallery, ranking, modified, "raw_top1")
    assert first["threshold"] == second["threshold"]
    assert first["curve"] == second["curve"]
    assert not set(first["calibration_identities"]) & set(first["evaluation_identities"])
    assert not set(first["calibration_ids"]) & set(first["evaluation_ids"])


@pytest.mark.parametrize("candidate", policy.CANDIDATES)
def test_heldout_candidate_export_matches_official_evaluator(tmp_path, candidate):
    query, gallery, qv, gv = protocol()
    ranking, _ = evaluation.rank_with_scores(qv, gv)
    result = evaluation.calibrate_inner_confidence(query, gallery, ranking, ranking["confidence"], candidate)
    indices = np.array([i for i, row in enumerate(query) if row["image_id"] in result["evaluation_ids"]])
    held = [query[int(i)] for i in indices]
    selected = evaluation.slice_ranking(ranking, indices, ranking["confidence"])
    actual = policy.export_csv(tmp_path / "export", held, gallery, selected, result["threshold"], candidate)
    assert actual == result["evaluation"]
    lines = (tmp_path / "export/submission.csv").read_text().splitlines()
    assert len(lines) == len(held)
    assert all(len(line.split(",")) == 11 for line in lines)
    acceptance = result["acceptance"]
    assert sum(v["count"] for v in acceptance["states"].values()) == len(held)
    assert acceptance["coverage"] == acceptance["accepted_count"] / len(held)


def test_confidence_signals_are_query_independent():
    _, _, qv, gv = protocol()
    ranking, _ = evaluation.rank_with_scores(qv, gv)
    members = np.stack([qv @ gv.T] * 3)
    support = np.linspace(0, 1, len(qv))
    signals = evaluation.confidence_signals(qv, gv, ranking, members, support)
    assert set(signals) == {"cosine", "cosine_plus_margin", "cosine_minus_disagreement", "cosine_plus_local"}
    perm = np.arange(len(qv))[::-1]
    shuffled = {key: value[perm] if isinstance(value, np.ndarray) else value for key, value in ranking.items()}
    other = evaluation.confidence_signals(qv[perm], gv, shuffled, members[:, perm], support[perm])
    for key in signals:
        np.testing.assert_array_equal(other[key], signals[key][perm])
    np.testing.assert_allclose(signals["cosine_minus_disagreement"], signals["cosine"])


def primary_results(gains):
    name = evaluation.LOCAL_GRID[1]["name"]
    return {str(seed): {c["name"]: {"regular0": {"ranking": {"mAP@10": .8 + (gains[seed] if c["name"] == name else 0)}}}
                       for c in evaluation.LOCAL_GRID} for seed in range(3)}


@pytest.mark.parametrize("gains, passed", [([.01, .01, -.001], True), ([.02, -.001, -.001], False),
                                         ([.001, .001, .001], False)])
def test_local_selection_requires_mean_gain_and_two_of_three(gains, passed):
    result = evaluation.select_local(primary_results(gains))
    assert result["passed"] is passed
    assert result["selected"] == (evaluation.LOCAL_GRID[1]["name"] if passed else "baseline")


def test_local_selection_missing_seed_fails_closed():
    with pytest.raises(ValueError, match="three"):
        evaluation.select_local({"0": primary_results([.1] * 3)["0"]})


def test_bootstrap_groups_repeated_queries_by_identity():
    baseline = {"a1": {"vehicle_id": 1, "ap": .5}, "a2": {"vehicle_id": 1, "ap": .6},
                "b": {"vehicle_id": 2, "ap": .7}, "unknown": {"vehicle_id": 3, "ap": None}}
    candidate = {key: {**value, "ap": None if value["ap"] is None else value["ap"] + .1}
                 for key, value in baseline.items()}
    result = evaluation.paired_identity_bootstrap(baseline, candidate, repeats=30)
    assert result["identities"] == 2
    assert result["mean_delta"] == pytest.approx(.1)
    np.testing.assert_allclose(result["interval"], [.1, .1])
