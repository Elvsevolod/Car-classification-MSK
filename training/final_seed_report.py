"""Descriptive, paired reporting of the frozen two-recipe/three-seed matrix."""
from collections.abc import Mapping
from math import isfinite
from numbers import Real
from statistics import mean, stdev


CONDITIONS = ("original", "masked_query", "masked_gallery", "masked_both")
METHODS = ("raw", "reranked")
COUNTS = ("known_queries", "unknown_queries", "TP", "FP", "FN", "TN", "open_set_FP", "true_refusals")
METRICS = ("mAP_at_10", "full_mAP", "mINP", "Rank_1", "Rank_5", "candidate_precision",
           "candidate_recall", "candidate_F1", "TNR", "candidate_score", "PR_AUC") + COUNTS


def _stats(values, seeds):
    valid = sum(value is not None for value in values)
    complete = valid == len(seeds)
    return {"mean": mean(values) if complete else None,
            "std": stdev(values) if complete else None,
            "seed_values": {str(seed): value for seed, value in zip(seeds, values)},
            "valid_count": valid}


def _protocol_identity(report):
    identity = {key: report.get(key) for key in ("protocol_sha256", "protocol")}
    if "per_query" in report:
        per_query = report["per_query"]
        if not isinstance(per_query, Mapping) or set(per_query) != set(METHODS):
            raise ValueError("per_query must contain both original ranking methods")
        identity["per_query"] = {}
        for method in METHODS:
            entries = per_query[method]
            if not isinstance(entries, Mapping) or any(
                not isinstance(row, Mapping) or "vehicle_id" not in row for row in entries.values()
            ):
                raise ValueError("Malformed per_query identities")
            identity["per_query"][method] = {qid: row["vehicle_id"] for qid, row in entries.items()}
        if identity["per_query"]["raw"] != identity["per_query"]["reranked"]:
            raise ValueError("Ranking methods have different query identities")
    return identity


def aggregate_results(evaluations, names, seeds):
    """No pooling, seed selection, threshold averaging, or missing-value omission."""
    names, seeds = list(names), list(seeds)
    if len(names) != 2 or set(names) != {"B0_control", "R1_resolution256"}:
        raise ValueError("Expected B0_control and R1_resolution256 exactly once")
    if len(seeds) != 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 3:
        raise ValueError("Expected three distinct integer seeds")
    if len(evaluations) != 6:
        raise ValueError("Expected the complete two-recipe/three-seed matrix")
    matrix, query_counts, protocol = {}, None, None
    for entry in evaluations:
        if not isinstance(entry, Mapping):
            raise ValueError("Malformed evaluation entry")
        name, seed, report = entry.get("variant"), entry.get("seed"), entry.get("evaluation")
        if name not in names or type(seed) is not int or seed not in seeds or (name, seed) in matrix:
            raise ValueError("Unexpected or duplicate variant/seed")
        if not isinstance(report, Mapping) or not isinstance(report.get("conditions"), Mapping):
            raise ValueError("Missing evaluation conditions")
        if set(report["conditions"]) != set(CONDITIONS):
            raise ValueError("Expected original and all three masked conditions")
        current_protocol = _protocol_identity(report)
        if protocol is not None and current_protocol != protocol:
            raise ValueError("Inconsistent evaluation protocols or query identities")
        protocol = current_protocol
        for condition in CONDITIONS:
            methods = report["conditions"][condition]
            if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
                raise ValueError("Expected raw and reranked metrics")
            for scores in methods.values():
                if not isinstance(scores, Mapping) or not set(METRICS) <= set(scores):
                    raise ValueError("Incomplete evaluation metrics")
                for metric in METRICS:
                    value = scores[metric]
                    if value is None and metric not in ("known_queries", "unknown_queries"):
                        continue
                    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                        raise ValueError(f"Nonfinite or invalid metric: {metric}")
                    if metric in COUNTS:
                        if value < 0 or int(value) != value:
                            raise ValueError(f"Invalid count: {metric}")
                    elif not 0 <= value <= 1:
                        raise ValueError(f"Metric outside [0, 1]: {metric}")
                counts = tuple(scores[key] for key in ("known_queries", "unknown_queries"))
                if query_counts is not None and counts != query_counts:
                    raise ValueError("Inconsistent known/unknown query counts")
                query_counts = counts
        matrix[name, seed] = report["conditions"]
    aggregate, paired = {name: {} for name in names}, {}
    for condition in CONDITIONS:
        paired[condition] = {}
        for name in names:
            aggregate[name][condition] = {}
        for method in METHODS:
            paired[condition][method] = {}
            for name in names:
                aggregate[name][condition][method] = {}
            for metric in METRICS:
                values = {name: [matrix[name, seed][condition][method][metric] for seed in seeds]
                          for name in names}
                for name in names:
                    aggregate[name][condition][method][metric] = _stats(values[name], seeds)
                deltas = [None if base is None or candidate is None else candidate - base
                          for base, candidate in zip(values["B0_control"], values["R1_resolution256"])]
                paired[condition][method][metric] = _stats(deltas, seeds)
    return {"aggregate": aggregate, "paired_deltas": paired, "names": names, "seeds": seeds,
            "policy": {"aggregation": "arithmetic mean and sample SD across all three training seeds",
                       "missing_values": "mean and SD are null unless all three seed values are defined",
                       "paired_delta": "R1_resolution256 minus B0_control at matching seed",
                       "selection": "none", "ensemble": False, "outer_status": "development, not independent test"}}


def _number(value):
    return "н/д" if value is None else f"{100 * value:.3f}"


def _mean_sd(stats):
    return f"{_number(stats['mean'])} ± {_number(stats['std'])} (n={stats['valid_count']}/3)"


def format_report(result, baseline):
    """Render descriptive percentages; the historical MVP is a reference, not a seed."""
    metrics = ("mAP_at_10", "candidate_F1", "TNR")
    lines = ["# Финальная проверка двух рецептов на трёх seed", "",
             "Среднее ± выборочное стандартное отклонение по seed; значения в процентах. "
             "При неполном наборе значений среднее и SD не вычисляются.", "",
             "| Модель | Поиск | mAP@10 | F1 | TNR |", "|---|---|---:|---:|---:|"]
    for name in result["names"]:
        for method in METHODS:
            scores = result["aggregate"][name]["original"][method]
            lines.append(f"| {name} | {method} | " + " | ".join(_mean_sd(scores[m]) for m in metrics) + " |")
    for method, scores in (("raw", baseline["raw_baseline"]["validation"]),
                           ("reranked", baseline["validation"])):
        lines.append(f"| MVP, историческая ссылка | {method} | " + " | ".join(_number(scores[m]) for m in metrics) + " |")
    lines += ["", "## Парные изменения R1 − B0", "", "Разности в процентных пунктах; seed не выбираются по метрике.", "",
              "| Поиск | Показатель | Среднее ± SD | " + " | ".join(str(s) for s in result["seeds"]) + " |",
              "|---|---|---:|" + "---:|" * len(result["seeds"])]
    for method in METHODS:
        for metric in metrics:
            stats = result["paired_deltas"]["original"][method][metric]
            lines.append(f"| {method} | {metric} | {_mean_sd(stats)} | " +
                         " | ".join(_number(stats["seed_values"][str(s)]) for s in result["seeds"]) + " |")
    lines += ["", "## Каждый seed: original", "", "| Модель | seed | Поиск | mAP@10 | F1 | TNR |",
              "|---|---:|---|---:|---:|---:|"]
    for name in result["names"]:
        for seed in result["seeds"]:
            for method in METHODS:
                scores = result["aggregate"][name]["original"][method]
                lines.append(f"| {name} | {seed} | {method} | " +
                             " | ".join(_number(scores[m]["seed_values"][str(seed)]) for m in metrics) + " |")
    lines += ["", "## Автоматические маски: reranked mAP@10", "",
              "| Условие | B0 | R1 | Парное R1 − B0, п.п. |", "|---|---:|---:|---:|"]
    for condition in CONDITIONS[1:]:
        scores = [_mean_sd(result["aggregate"][n][condition]["reranked"]["mAP_at_10"])
                  for n in ("B0_control", "R1_resolution256")]
        scores.append(_mean_sd(result["paired_deltas"][condition]["reranked"]["mAP_at_10"]))
        lines.append(f"| {condition} | " + " | ".join(scores) + " |")
    lines += ["", "## Ограничения", "",
              "- Это три отдельных обучения, не ансамбль. Нет выбора лучшего seed, усреднения порогов или автозамены MVP.",
              "- Запросы повторяются между seed: их нельзя считать независимыми новыми наблюдениями. SD не является доверительным интервалом.",
              "- Решение добавить два seed принято после просмотра первого outer-результата. Это дополнительная проверка устойчивости на development, не новый независимый тест.",
              "- Исторический MVP выбирался по outer validation; его процедура отбора несимметрична новому протоколу. MVP в таблице не входит в средние или парные разности.",
              "- Маски автоматические и не подтверждены как plate-only. Original validation и исходные bbox не меняются.",
              "- GPU-инференс и официальный скоростной тест не проверены этим отчётом."]
    return "\n".join(lines) + "\n"
