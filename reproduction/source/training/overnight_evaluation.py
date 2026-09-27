"""Train-only system diagnostics; inference scores never receive identity/camera."""
from statistics import mean

import numpy as np

from training import retrieval_policy as policy
from training.audit import digest


LOCAL_GRID = [{"name": "baseline", "top_k": 50, "weight": 0., "scorer": "partial"}] + [
    {"name": f"{mode}_k{k}_w{int(w * 100):02d}", "top_k": k, "weight": w, "scorer": mode}
    for mode in ("mean_best", "partial") for k in (20, 50) for w in (.05, .10, .20)]


def rank_with_scores(qv, gv):
    """The existing frozen less_graph policy, including its actual score scale."""
    ranking = policy.rank_vectors(qv, gv, "less_graph")
    scores = np.stack([-ranking["graph"].distances(v, .75) for v in qv])
    return ranking, scores


def summarize_draws(draws):
    return mean(d["ranking"]["mAP@10"] for d in draws.values())


def select_local(primary, min_gain=.002):
    """All three seeds equally; no seed/alternate maximization."""
    if len(primary) != 3:
        raise ValueError("Local selection needs all three training seeds")
    names = [c["name"] for c in LOCAL_GRID]
    scores = {name: {seed: summarize_draws(values[name]) for seed, values in primary.items()}
              for name in names}
    averages = {name: mean(values.values()) for name, values in scores.items()}
    best = max(names, key=lambda name: averages[name])
    gains = {s: scores[best][s] - scores["baseline"][s] for s in primary}
    passed = averages[best] - averages["baseline"] >= min_gain and sum(v > 0 for v in gains.values()) >= 2
    return {"selected": best if passed else "baseline", "best_observed": best,
            "mean_map": averages, "seed_values": scores, "paired_gain": gains,
            "min_gain": min_gain, "passed": passed,
            "rule": "primary mean gain >= .002 and positive in at least 2/3 seeds"}


def identity_calibration_partition(queries, gallery):
    """Deterministic identity-grouped halves, stratified by open-set status.

    Both subsets are encoder-held-out train IDs. This is an inner calibration
    diagnostic, never the original outer calibration or a new independent test.
    A repeated identity receives one assignment across all query episodes.
    """
    q, g = policy.frames(queries, gallery)
    groups = {}
    for row in queries:
        groups.setdefault(int(row["vehicle_id"]), []).append(row)
    strata = {False: [], True: []}
    for identity, rows in groups.items():
        known = {policy.official.valid_positives(q.loc[r["image_id"]], g) > 0 for r in rows}
        if len(known) != 1:
            raise ValueError("Identity has mixed open-set status in this episode")
        strata[known.pop()].append(identity)
    if min(map(len, strata.values())) < 2:
        raise ValueError("Need >=2 known and >=2 unknown identities for held-out calibration")
    calibration = set()
    for identities in strata.values():
        ordered = sorted(identities, key=lambda i: digest({"identity": i, "salt": "v19-calibration"}))
        calibration.update(ordered[::2])
    a = np.array([i for i, r in enumerate(queries) if r["vehicle_id"] in calibration])
    b = np.array([i for i, r in enumerate(queries) if r["vehicle_id"] not in calibration])
    return a, b


def confidence_signals(qv, gv, ranking, member_cosines=None, local_support=None):
    """Fixed functions of one query and static gallery, not fitted rejectors."""
    cosine = np.clip(qv @ gv.T, -1, 1)
    order = ranking["raw_order"]
    top = np.take_along_axis(cosine, order[:, :2], axis=1)
    signals = {"cosine": top[:, 0], "cosine_plus_margin": top[:, 0] + .25 * (top[:, 0] - top[:, 1])}
    if member_cosines is not None:
        values = np.asarray(member_cosines)
        if values.shape != (3, *cosine.shape) or not np.isfinite(values).all():
            raise ValueError("Expected three member score matrices")
        support = values[:, np.arange(len(qv)), order[:, 0]]
        signals["cosine_minus_disagreement"] = top[:, 0] - .25 * support.std(axis=0)
    if local_support is not None:
        values = np.asarray(local_support)
        if values.shape != (len(qv),) or not np.isfinite(values).all():
            raise ValueError("Invalid per-query local support")
        signals["cosine_plus_local"] = top[:, 0] + .10 * values
    return signals


def slice_ranking(ranking, indices, confidence):
    return {"order": ranking["order"][indices], "raw_order": ranking["raw_order"][indices],
            "confidence": np.asarray(confidence)[indices]}


def acceptance_diagnostics(queries, gallery, ranking, threshold, candidate):
    q, g = policy.frames(queries, gallery)
    _, accepted = policy.predictions(queries, gallery, ranking, threshold, candidate)
    states = {name: {"count": 0, "accepted": 0} for name in ("known_correct", "known_wrong", "unknown")}
    order = ranking["raw_order"] if candidate == "raw_top1" else ranking["order"]
    wrong_accepted = 0
    for index, row in enumerate(queries):
        known = policy.official.valid_positives(q.loc[row["image_id"]], g) > 0
        correct = known and gallery[int(order[index, 0])]["vehicle_id"] == row["vehicle_id"]
        state = "unknown" if not known else "known_correct" if correct else "known_wrong"
        take = row["image_id"] in accepted
        states[state]["count"] += 1
        states[state]["accepted"] += int(take)
        wrong_accepted += int(take and not correct)
    return {"states": states, "coverage": len(accepted) / len(queries),
            "accepted_error_rate": wrong_accepted / len(accepted) if accepted else None,
            "accepted_count": len(accepted)}


def calibrate_inner_confidence(queries, gallery, ranking, confidence, candidate):
    """Threshold on calibration half; official C + coverage on untouched half."""
    cal_indices, held_indices = identity_calibration_partition(queries, gallery)
    cal, held = ([queries[int(i)] for i in indices] for indices in (cal_indices, held_indices))
    cal_rank = slice_ranking(ranking, cal_indices, confidence)
    choices = np.unique(cal_rank["confidence"])
    choices = np.append(choices, np.nextafter(choices[-1], np.inf))
    curve = [{"threshold": float(t), **policy.evaluate(cal, gallery, cal_rank, float(t), candidate)["candidates"]}
             for t in choices]
    best = max(curve, key=lambda x: (x["C"], x["F1"], x["threshold"]))
    held_rank = slice_ranking(ranking, held_indices, confidence)
    return {"threshold": best["threshold"], "calibration": best,
            "evaluation": policy.evaluate(held, gallery, held_rank, best["threshold"], candidate),
            "acceptance": acceptance_diagnostics(held, gallery, held_rank, best["threshold"], candidate),
            "calibration_ids": [r["image_id"] for r in cal], "evaluation_ids": [r["image_id"] for r in held],
            "calibration_identities": sorted({r["vehicle_id"] for r in cal}),
            "evaluation_identities": sorted({r["vehicle_id"] for r in held}),
            "scope": "encoder-held-out train IDs; identity-disjoint threshold calibration/evaluation",
            "curve": curve}


def paired_identity_bootstrap(baseline, candidate, *, seed=20260915, repeats=1000):
    """Conditional paired interval; repeated query episodes do not add identities."""
    groups = {}
    if set(baseline) != set(candidate):
        raise ValueError("Unpaired reports")
    for key, value in baseline.items():
        other = candidate[key]
        if value["vehicle_id"] != other["vehicle_id"]:
            raise ValueError("Changed identity pairing")
        if value["ap"] is not None and other["ap"] is not None:
            groups.setdefault(value["vehicle_id"], []).append(other["ap"] - value["ap"])
    if not groups:
        return {"identities": 0, "interval": None}
    values = [np.asarray(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    samples = [float(np.concatenate([values[i] for i in rng.integers(len(values), size=len(values))]).mean())
               for _ in range(repeats)]
    return {"identities": len(values), "mean_delta": float(np.concatenate(values).mean()),
            "interval": np.quantile(samples, [.025, .975]).tolist(),
            "note": "conditional identity-cluster bootstrap, not independent training-seed uncertainty"}
