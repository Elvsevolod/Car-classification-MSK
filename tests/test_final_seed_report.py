from copy import deepcopy
from statistics import stdev

import pytest

from training.final_seed_report import CONDITIONS, COUNTS, METHODS, METRICS, aggregate_results, format_report


NAMES = ["B0_control", "R1_resolution256"]
SEEDS = [20260915, 20260916, 20260917]


def evaluations():
    entries = []
    for index, name in enumerate(NAMES):
        for offset, seed in enumerate(SEEDS):
            scores = {metric: .5 + offset * .1 + index * .02 for metric in METRICS}
            scores.update({metric: 10 for metric in COUNTS})
            entries.append({"variant": name, "seed": seed, "evaluation": {
                "conditions": {condition: {method: deepcopy(scores) for method in METHODS}
                               for condition in CONDITIONS},
                "per_query": {method: {"q1": {"vehicle_id": "v1", "ap": scores["mAP_at_10"]}}
                              for method in METHODS}}})
    return entries


def test_arithmetic_mean_sample_sd_and_paired_values_not_best_seed():
    result = aggregate_results(evaluations(), NAMES, SEEDS)
    score = result["aggregate"][NAMES[0]]["original"]["raw"]["mAP_at_10"]
    assert score["mean"] == pytest.approx(.6)
    assert score["std"] == pytest.approx(stdev([.5, .6, .7]))
    assert score["seed_values"] == dict(zip(map(str, SEEDS), [.5, .6, .7]))
    delta = result["paired_deltas"]["original"]["raw"]["mAP_at_10"]
    assert delta["mean"] == pytest.approx(.02)
    assert delta["std"] == pytest.approx(0)
    assert delta["valid_count"] == 3
    assert result["policy"]["selection"] == "none"
    assert result["policy"]["ensemble"] is False
    assert "winner" not in result


def test_null_not_dropped_or_replaced_with_zero():
    rows = evaluations()
    rows[1]["evaluation"]["conditions"]["original"]["raw"]["PR_AUC"] = None
    result = aggregate_results(rows, NAMES, SEEDS)
    stats = result["aggregate"][NAMES[0]]["original"]["raw"]["PR_AUC"]
    assert stats == {"mean": None, "std": None, "valid_count": 2,
                     "seed_values": {str(SEEDS[0]): .5, str(SEEDS[1]): None, str(SEEDS[2]): .7}}
    paired = result["paired_deltas"]["original"]["raw"]["PR_AUC"]
    assert paired["mean"] is paired["std"] is None
    assert paired["valid_count"] == 2


def test_all_null_and_missing_at_different_paired_seeds():
    rows = evaluations()
    for row in rows:
        row["evaluation"]["conditions"]["masked_query"]["raw"]["PR_AUC"] = None
    rows[0]["evaluation"]["conditions"]["original"]["raw"]["TNR"] = None
    rows[4]["evaluation"]["conditions"]["original"]["raw"]["TNR"] = None
    result = aggregate_results(rows, NAMES, SEEDS)
    assert result["paired_deltas"]["original"]["raw"]["TNR"]["valid_count"] == 1
    assert result["paired_deltas"]["masked_query"]["raw"]["PR_AUC"]["valid_count"] == 0


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows.__setitem__(1, deepcopy(rows[0])),
    lambda rows: rows[0].update(variant="other"),
    lambda rows: rows[0].update(seed=1),
    lambda rows: rows[0].update(seed=True),
    lambda rows: rows[0].pop("evaluation"),
    lambda rows: rows[0]["evaluation"]["conditions"].pop("masked_both"),
    lambda rows: rows[0]["evaluation"]["conditions"]["original"].pop("raw"),
    lambda rows: rows[0]["evaluation"]["conditions"]["original"]["raw"].pop("PR_AUC"),
    lambda rows: rows[0]["evaluation"]["conditions"]["original"]["raw"].update(unknown_queries=9),
    lambda rows: rows[0]["evaluation"]["conditions"]["original"]["raw"].update(known_queries=None),
    lambda rows: rows[0]["evaluation"].update(protocol_sha256="different-protocol"),
    lambda rows: rows[0]["evaluation"].pop("per_query"),
    lambda rows: rows[0]["evaluation"]["per_query"]["raw"]["q1"].update(vehicle_id="changed"),
    lambda rows: rows[0]["evaluation"]["per_query"]["raw"].update(q2={"vehicle_id": "v2"}),
])
def test_rejects_malformed_or_incomplete_matrix_and_protocol_drift(mutation):
    rows = evaluations()
    mutation(rows)
    with pytest.raises(ValueError):
        aggregate_results(rows, NAMES, SEEDS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "0.8", True, -0.1, 1.1])
def test_rejects_invalid_metrics(value):
    rows = evaluations()
    rows[0]["evaluation"]["conditions"]["original"]["raw"]["mAP_at_10"] = value
    with pytest.raises(ValueError):
        aggregate_results(rows, NAMES, SEEDS)


@pytest.mark.parametrize("names,seeds", [(NAMES, SEEDS[:2]), (NAMES, SEEDS + [4]),
                                        (NAMES, [1, 1, 2]), (NAMES, [1, 2, True]),
                                        ([NAMES[0], NAMES[0]], SEEDS), (["a", "b"], SEEDS)])
def test_rejects_wrong_requested_matrix(names, seeds):
    with pytest.raises(ValueError):
        aggregate_results(evaluations(), names, seeds)


def test_order_independent_and_no_mutation():
    rows = evaluations()
    original = deepcopy(rows)
    expected = aggregate_results(rows, NAMES, SEEDS)
    assert aggregate_results(list(reversed(rows)), NAMES, SEEDS) == expected
    assert rows == original


def test_report_has_means_seed_values_reference_and_caveats():
    result = aggregate_results(evaluations(), NAMES, SEEDS)
    baseline = {"raw_baseline": {"validation": {"mAP_at_10": .79, "candidate_F1": .73, "TNR": .79}},
                "validation": {"mAP_at_10": .81, "candidate_F1": .73, "TNR": .79}}
    report = format_report(result, baseline)
    assert "60.000 ± 10.000 (n=3/3)" in report
    assert "2.000 ± 0.000 (n=3/3)" in report
    assert all(str(seed) in report for seed in SEEDS)
    assert "MVP, историческая ссылка" in report
    assert "после просмотра первого outer-результата" in report
    assert "не новый независимый тест" in report
    assert "не ансамбль" in report
    assert "не является доверительным интервалом" in report
    assert "plate-only" in report and "GPU" in report
