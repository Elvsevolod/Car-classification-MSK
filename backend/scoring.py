"""Adapt local predictions to the unmodified organizer evaluate.py."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import evaluate as official
from .core import normalize


@dataclass
class RankedQueries:
    query: pd.DataFrame
    gallery: pd.DataFrame
    predictions: dict
    confidence: np.ndarray
    ranking: dict
    full_ranking: dict


def ranked_queries(queries, gallery, embeddings, scores=None):
    """Build exactly the submitted Top-10; only the organizer removes junk."""
    query = pd.DataFrame(queries).set_index("image_id")
    gallery_frame = pd.DataFrame(gallery).set_index("image_id")
    q_ids, g_ids = query.index.tolist(), gallery_frame.index.tolist()
    q_emb = normalize(np.stack([embeddings[i] for i in q_ids]))
    g_emb = normalize(np.stack([embeddings[i] for i in g_ids]))
    cosine = np.clip(q_emb @ g_emb.T, -1, 1)
    scores = cosine if scores is None else np.asarray(scores)
    if scores.shape != cosine.shape or not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be a finite query-by-gallery matrix")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :official.TOP_K]
    predictions = {qid: [g_ids[j] for j in indices] for qid, indices in zip(q_ids, order)}
    return RankedQueries(
        query, gallery_frame, predictions, cosine.max(axis=1),
        official.ranking_metrics(query, gallery_frame, predictions),
        official.full_ranking_metrics(q_emb, g_emb, q_ids, g_ids, query, gallery_frame),
    )


def _confidence(ranked, acceptance_scores):
    scores = ranked.confidence if acceptance_scores is None else np.asarray(acceptance_scores)
    if scores.shape != (len(ranked.query),) or not np.isfinite(scores).all():
        raise ValueError("acceptance_scores must contain one finite value per query")
    return scores


def metrics(ranked, threshold, acceptance_scores=None):
    confidence = _confidence(ranked, acceptance_scores)
    candidates = {
        qid: [(ranked.predictions[qid][0], float(score))]
        for qid, score in zip(ranked.query.index, confidence) if float(score) >= threshold
    }
    candidate = official.candidate_metrics(ranked.query, ranked.gallery, candidates)
    ranking, full = ranked.ranking, ranked.full_ranking
    # JSON/API reports use null for official undefined (NaN) diagnostic values.
    tnr = candidate["TNR"] if np.isfinite(candidate["TNR"]) else None
    pr_auc = candidate["PR-AUC"] if np.isfinite(candidate["PR-AUC"]) else None
    return {
        "mAP": ranking["mAP@10"], "mAP_at_10": ranking["mAP@10"],
        "full_mAP": full["mAP_full"], "mINP": full["mINP"],
        "Rank_1": ranking["Rank-1"], "Rank_5": ranking["Rank-5"],
        "candidate_precision": candidate["Precision"], "candidate_recall": candidate["Recall"],
        "candidate_F1": candidate["F1"], "TNR": tnr, "PR_AUC": pr_auc,
        "candidate_score": .7 * candidate["F1"] + .3 * tnr if tnr is not None else None,
        "known_queries": ranking["n_scored"], "unknown_queries": candidate["n_openset_queries"],
        **{key: candidate[key] for key in ("TP", "FP", "FN", "TN")},
        "open_set_FP": candidate["n_openset_queries"] - candidate["TN"],
        "true_refusals": candidate["TN"],
    }


def threshold_curve(ranked, acceptance_scores=None):
    """Evaluate every acceptance boundary using the organizer's candidate metrics."""
    scores = _confidence(ranked, acceptance_scores)
    values = np.unique(scores).astype(float)
    if not len(values):
        raise ValueError("Cannot calibrate a refusal threshold without query scores")
    choices = np.append(values, np.nextafter(values[-1], np.inf))
    return [{"threshold": float(threshold), **metrics(ranked, threshold, scores)} for threshold in choices]


def calibrate(ranked, acceptance_scores=None):
    curve = threshold_curve(ranked, acceptance_scores)
    if curve[0]["candidate_score"] is None:
        raise ValueError("Refusal calibration requires open-set queries to define TNR")
    return max(curve, key=lambda item: (item["candidate_score"], item["candidate_F1"], item["threshold"]))["threshold"]
