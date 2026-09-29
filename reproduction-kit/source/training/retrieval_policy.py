"""Separate streaming top-10 ranking from the single accepted candidate."""
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as official
from backend.core import normalize
from backend.rerank import KReciprocalReranker


# Order also freezes tie-breaking: keep the historical policy on exact ties.
POLICIES = {
    "legacy": {"k1": 20, "k2": 3, "lambda": .50},
    "no_expansion": {"k1": 20, "k2": 1, "lambda": .50},
    "less_graph": {"k1": 20, "k2": 3, "lambda": .75},
    "no_expansion_less_graph": {"k1": 20, "k2": 1, "lambda": .75},
    "raw": None,
}
CANDIDATES = ("raw_top1", "ranking_top1")


def combine_members(members):
    """Equal cosine mixture, NOT an average in incompatible feature coordinates."""
    if len(members) not in (1, 3):
        raise ValueError("Use one encoder or all three equally weighted seeds")
    values = [normalize(np.asarray(x, dtype=np.float32)) for x in members]
    if any(x.ndim != 2 or x.shape != values[0].shape for x in values):
        raise ValueError("Members need matching image order and embedding shape")
    return normalize(np.concatenate(values, axis=1) / np.sqrt(len(values)))


def rank_vectors(query, gallery, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown frozen ranking policy")
    query, gallery = normalize(query), normalize(gallery)
    if (query.ndim != 2 or gallery.ndim != 2 or not len(query) or not len(gallery)
            or query.shape[1] != gallery.shape[1]):
        raise ValueError("Expected matching query/gallery feature matrices")
    cosine = np.clip(query @ gallery.T, -1, 1)
    raw_order = np.argsort(-cosine, axis=1, kind="stable")
    setting = POLICIES[policy]
    graph = None
    if setting is None or len(gallery) == 1:
        order = raw_order.copy()
    else:
        k1 = min(setting["k1"], len(gallery) - 1)
        graph = KReciprocalReranker(gallery, k1, min(setting["k2"], k1 + 1))
        distances = np.stack([graph.distances(vector, setting["lambda"]) for vector in query])
        order = np.argsort(distances, axis=1, kind="stable")
    return {"order": order, "raw_order": raw_order, "confidence": cosine.max(axis=1), "graph": graph}


def frames(queries, gallery):
    q, g = (pd.DataFrame(rows).set_index("image_id") for rows in (queries, gallery))
    if not q.index.is_unique or not g.index.is_unique or set(q.index) & set(g.index):
        raise ValueError("Query/gallery IDs must be unique and disjoint")
    return q, g


def predictions(queries, gallery, ranking, threshold, candidate_policy):
    if candidate_policy not in CANDIDATES or not np.isfinite(threshold):
        raise ValueError("Invalid candidate policy/threshold")
    gids = [row["image_id"] for row in gallery]
    if (len({r["image_id"] for r in queries}) != len(queries) or len(set(gids)) != len(gids)
            or ranking["order"].shape != (len(queries), len(gallery))
            or ranking["raw_order"].shape != ranking["order"].shape
            or ranking["confidence"].shape != (len(queries),)):
        raise ValueError("Prediction IDs/shapes do not match the protocol")
    ordered = {row["image_id"]: [gids[int(j)] for j in indices[:10]]
               for row, indices in zip(queries, ranking["order"])}
    candidate_order = ranking["raw_order"] if candidate_policy == "raw_top1" else ranking["order"]
    accepted = {row["image_id"]: [(gids[int(indices[0])], float(score))]
                for row, indices, score in zip(queries, candidate_order, ranking["confidence"])
                if float(score) >= threshold}
    return ordered, accepted


def candidate_metrics(query, gallery, accepted):
    result = official.candidate_metrics(query, gallery, accepted)
    result = {key: value if np.isfinite(value) else None for key, value in result.items()}
    result["C"] = .7 * result["F1"] + .3 * result["TNR"] if result["TNR"] is not None else None
    result["open_set_FP"] = result["n_openset_queries"] - result["TN"]
    return result


def evaluate(queries, gallery, ranking, threshold, candidate_policy):
    q, g = frames(queries, gallery)
    ordered, accepted = predictions(queries, gallery, ranking, threshold, candidate_policy)
    return {"ranking": official.ranking_metrics(q, g, ordered),
            "candidates": candidate_metrics(q, g, accepted)}


def calibrate_policy(queries, gallery, ranking, candidate_policy, *, split):
    """Only the original calibration split may choose a final operating point."""
    if split != "calibration":
        raise ValueError("Threshold selection is calibration-only")
    q, g = frames(queries, gallery)
    scores = np.unique(ranking["confidence"]).astype(float)
    choices = np.append(scores, np.nextafter(scores[-1], np.inf))
    curve = []
    for threshold in choices:
        _, accepted = predictions(queries, gallery, ranking, float(threshold), candidate_policy)
        curve.append({"threshold": float(threshold), **candidate_metrics(q, g, accepted)})
    if curve[0]["C"] is None:
        raise ValueError("Calibration requires unknown queries")
    best = max(curve, key=lambda item: (item["C"], item["F1"], item["threshold"]))
    near = [x["threshold"] for x in curve if x["C"] >= best["C"] - .005]
    fixed_tnr = {}
    for target in (.7, .8, .9):
        eligible = [x for x in curve if x["TNR"] >= target]
        fixed_tnr[str(target)] = max(eligible, key=lambda x: (x["C"], x["F1"], x["threshold"]))
    return {"split": split, "candidate_policy": candidate_policy, "selected": best,
            "curve": curve, "near_optimal_C_tolerance": .005,
            "near_optimal_threshold_range": [min(near), max(near)],
            "fixed_TNR_calibration_only": fixed_tnr}


def query_diagnostics(queries, gallery, ranking):
    q, g = frames(queries, gallery)
    orders = {"raw": ranking["raw_order"], "ranking": ranking["order"]}
    result = {}
    for index, (qid, row) in enumerate(q.iterrows()):
        known = official.valid_positives(row, g) > 0
        entry = {"vehicle_id": int(row.vehicle_id), "known": bool(known),
                 "confidence": float(ranking["confidence"][index])}
        for name, order in orders.items():
            ids = [g.index[int(j)] for j in order[index, :10]]
            value = official.ranking_metrics(q.loc[[qid]], g, {qid: ids})
            entry[name] = {"top1": ids[0], "correct": bool(known and g.loc[ids[0]].vehicle_id == row.vehicle_id),
                           "ap": value["mAP@10"] if value["n_scored"] else None}
        entry["confidence_group"] = ("unknown" if not known else
                                     "known_correct_raw" if entry["raw"]["correct"] else "known_wrong_raw")
        result[qid] = entry
    transitions = {"correct_to_wrong": sum(x["raw"]["correct"] and not x["ranking"]["correct"] for x in result.values()),
                   "wrong_to_correct": sum(not x["raw"]["correct"] and x["ranking"]["correct"] for x in result.values())}
    groups = {}
    for group in ("unknown", "known_correct_raw", "known_wrong_raw"):
        values = [x["confidence"] for x in result.values() if x["confidence_group"] == group]
        groups[group] = {"count": len(values), "quantiles_10_50_90": np.quantile(values, [.1, .5, .9]).tolist() if values else None}
    auc = official.pr_auc(np.array([x["confidence"] for x in result.values()]),
                          np.array([int(x["known"]) for x in result.values()]))
    return {"per_query": result, "top1_transitions": transitions, "confidence_before_threshold": groups,
            "all_query_PR_AUC_before_threshold": float(auc) if np.isfinite(auc) else None}


def gallery_diagnostics(gallery, graph):
    if graph is None:
        return {"graph": "not used"}
    frequency = np.zeros(len(gallery), dtype=int)
    same_id = same_camera = junk = total = 0
    for i in range(len(gallery)):
        neighbors = graph.initial_rank[i]
        neighbors = neighbors[neighbors != i][:graph.k1]
        frequency[neighbors] += 1
        reciprocal = graph._reciprocal(i, graph.k1)
        for j in reciprocal[reciprocal != i]:
            identity = gallery[i]["vehicle_id"] == gallery[int(j)]["vehicle_id"]
            camera = gallery[i]["camera_id"] == gallery[int(j)]["camera_id"]
            same_id += int(identity); same_camera += int(camera); junk += int(identity and camera); total += 1
    return {"neighbor_frequency": {r["image_id"]: int(n) for r, n in zip(gallery, frequency)},
            "reciprocal_edges": total, "same_identity_edges": same_id, "same_camera_edges": same_camera,
            "same_identity_same_camera_junk_edges": junk,
            "note": "Labels are diagnostics only; no gallery edge is filtered using identity/camera"}


def error_overlap(reports):
    result = {}
    for a, b in combinations(reports, 2):
        left, right = reports[a]["per_query"], reports[b]["per_query"]
        if set(left) != set(right):
            raise ValueError("Unpaired error diagnostic")
        errors = [{q for q, x in rows.items() if x["known"] and x["raw"]["ap"] < 1 - 1e-12}
                  for rows in (left, right)]
        union = errors[0] | errors[1]
        result[f"{a}/{b}"] = {"errors_a": len(errors[0]), "errors_b": len(errors[1]),
                              "intersection": len(errors[0] & errors[1]), "union": len(union),
                              "jaccard": len(errors[0] & errors[1]) / len(union) if union else None}
    return result


def export_csv(output, queries, gallery, ranking, threshold, candidate_policy):
    output = Path(output)
    if len(gallery) < 10:
        raise ValueError("Official submission requires at least ten gallery images")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty CSV output directory")
    ordered, accepted = predictions(queries, gallery, ranking, threshold, candidate_policy)
    if len(ordered) != len(queries) or any(len(set(ids)) != 10 for ids in ordered.values()):
        raise ValueError("Incomplete/duplicate official top-10")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "submission.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([qid, *ids] for qid, ids in ordered.items())
    with (output / "candidates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        writer.writerows([qid, pairs[0][0], pairs[0][1]] for qid, pairs in accepted.items())
    loaded = official.load_submission(output / "submission.csv", {r["image_id"] for r in gallery})
    candidates = official.load_candidates(output / "candidates.csv")
    if loaded != ordered or set(candidates) != set(accepted) or any(
            candidates[q][0][0] != accepted[q][0][0] or
            not np.isclose(candidates[q][0][1], accepted[q][0][1], rtol=0, atol=1e-15) for q in accepted):
        raise ValueError("Official CSV roundtrip mismatch")
    if all("vehicle_id" in row for row in queries + gallery):
        q, g = frames(queries, gallery)
        return {"ranking": official.ranking_metrics(q, g, loaded), "candidates": candidate_metrics(q, g, candidates)}
    return {"queries": len(ordered), "accepted": len(accepted)}
