"""Frozen shared-step selection and train-only protocol draws for the review experiment."""
import math
import statistics

from backend.evaluate import make_protocol


CONTROL = "B0_control"


def _unique(values, label):
    values = list(values)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Expected nonempty unique {label}")
    return values


def _index(runs, names, seeds):
    names, seeds = _unique(names, "variants"), _unique(seeds, "seeds")
    if CONTROL not in names:
        raise ValueError("Missing B0_control")
    indexed = {}
    for run in runs:
        key = run["variant"], run["seed"]
        if key in indexed:
            raise ValueError("Duplicate variant/seed run")
        indexed[key] = run
    if set(indexed) != {(name, seed) for name in names for seed in seeds}:
        raise ValueError("Incomplete or unexpected variant/seed runs")
    return indexed, names, seeds


def _score(validation):
    value = float(validation["mean_map"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected finite mean_map in [0, 1]")
    return value


def _stats(scores):
    values = list(scores.values())
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.,
            "seed_scores": scores}


def select_shared_steps(runs, names, seeds, boundaries):
    """Choose ONE step per recipe, never average separately selected seed maxima."""
    indexed, names, seeds = _index(runs, names, seeds)
    boundaries = sorted(_unique(boundaries, "boundaries"))
    if any(type(step) is not int or step < 0 for step in boundaries):
        raise ValueError("Boundaries must be nonnegative integer update counts")
    histories = {}
    for key, run in indexed.items():
        history = run["history"]
        steps = [item["step"] for item in history]
        if len(steps) != len(set(steps)) or set(steps) != set(boundaries):
            raise ValueError("History must contain every boundary exactly once")
        histories[key] = {item["step"]: _score(item["validation"]) for item in history}
    aggregate = {}
    for name in names:
        candidates = []
        for step in boundaries:
            scores = {str(seed): histories[name, seed][step] for seed in seeds}
            candidates.append({"step": step, **_stats(scores)})
        aggregate[name] = min(candidates, key=lambda item: (-item["mean"], item["step"]))
    winner = min(names, key=lambda name: (-aggregate[name]["mean"], name != CONTROL, name))
    return {"winner": winner, "aggregate": aggregate,
            "policy": "primary only; mean over fixed draws within seed, then paired seeds at one shared step; "
                      "step ties prefer fewer updates, recipe ties prefer B0; alternate never selects"}


def summarize_fixed(runs, selection, seeds):
    """Summarize frozen-step alternate scores without modifying the primary choice."""
    indexed, names, seeds = _index(runs, selection["aggregate"], seeds)
    aggregate = {}
    for name in names:
        step = selection["aggregate"][name]["step"]
        scores = {}
        for seed in seeds:
            run = indexed[name, seed]
            if run["stop_step"] != step:
                raise ValueError("Alternate stop differs from frozen primary step")
            scores[str(seed)] = _score(run["validation"])
        aggregate[name] = {"step": step, **_stats(scores)}
    paired = {}
    for name in names:
        if name == CONTROL:
            continue
        deltas = {str(seed): aggregate[name]["seed_scores"][str(seed)] -
                  aggregate[CONTROL]["seed_scores"][str(seed)] for seed in seeds}
        result = _stats(deltas)
        paired[name] = {"mean": result["mean"], "std": result["std"], "seed_deltas": deltas}
    return {"aggregate": aggregate, "paired_deltas_vs_control": paired}


def make_draws(rows, identities, seeds):
    """Keep standard draws unchanged; same-camera additions are diagnostic only."""
    identities, seeds = _unique(identities, "identities"), _unique(seeds, "draw seeds")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    chosen = set(identities)
    if {row["vehicle_id"] for row in rows if row["vehicle_id"] in chosen} != chosen:
        raise ValueError("Missing protocol identities")
    result = {}
    for seed in seeds:
        query, gallery = make_protocol(rows, identities, seed)
        query_ids = {row["image_id"] for row in query}
        known = {row["vehicle_id"] for row in gallery}
        if not known or query_ids & {row["image_id"] for row in gallery}:
            raise ValueError("Protocol needs positives and disjoint query/gallery images")
        for q in query:
            if q["vehicle_id"] in known and not any(
                    g["vehicle_id"] == q["vehicle_id"] and g["camera_id"] != q["camera_id"] for g in gallery):
                raise ValueError("Known query lacks a cross-camera positive")
        cameras = {q["vehicle_id"]: q["camera_id"] for q in query if q["vehicle_id"] in known}
        additions = sorted((r for r in rows if r["vehicle_id"] in known
                            and r["camera_id"] == cameras[r["vehicle_id"]]
                            and r["image_id"] not in query_ids), key=lambda r: r["image_id"])
        for condition, items in (("regular", gallery), ("same_camera_junk", gallery + additions)):
            result[f"{condition}_{seed}"] = {
                "query_ids": [r["image_id"] for r in query],
                "gallery_ids": [r["image_id"] for r in items],
                "selection_eligible": condition == "regular", "seed": seed, "condition": condition,
            }
    return result
