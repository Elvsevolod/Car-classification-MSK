"""v27 ranking only: independent static-gallery graphs, then distance/rank fusion."""
import numpy as np

from backend.core import normalize
from backend.rerank import KReciprocalReranker
from training import dual_role_inference as dual
from training import map_inference as v25

BASELINE = {"name": "V25_control", "fusion": "features"}


def systems():
    return [dict(BASELINE)]+[
        {"name": f"{method}_r1w{round(w*100):02d}_l{round(lam*100):02d}",
         "fusion": method, "r1_weight": w, "r1_lambda": lam}
        for method in ("distance", "rrf10", "rrf60")
        for lam in (.5, .75) for w in (.1, .25, .4, .5, .6, .75, .9)]


def fusion_order(left, right, method, weight):
    """Both inputs are per-query gallery distances (smaller is better)."""
    if (left.ndim != 2 or left.shape != right.shape or not all(left.shape)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError("Expected matching finite distance matrices and weight in [0,1]")
    if method == "distance":
        distance = (1-weight)*left+weight*right
    elif method in ("rrf10", "rrf60"):
        # Weighted adaptation of Cormack et al. SIGIR 2009; ranks start at one.
        k = 10 if method == "rrf10" else 60
        ranks = [np.argsort(np.argsort(d, axis=1, kind="stable"), axis=1, kind="stable")+1
                 for d in (left, right)]
        distance = -((1-weight)/(k+ranks[0])+weight/(k+ranks[1]))
    else:
        raise ValueError("Unknown late fusion method")
    return np.argsort(distance, axis=1, kind="stable")


def branch_distances(block, n_query, lam):
    q, g = normalize(block[:n_query]), normalize(block[n_query:])
    graph = KReciprocalReranker(g, min(20, len(g)-1), min(3, len(g)))
    return np.stack([graph.distances(vector, lam) for vector in q])


def rank(values, n_query, spec):
    if spec not in systems():
        raise ValueError("Ranking specification is outside the frozen v27 grid")
    if type(n_query) is not int or not 0 < n_query < len(values)-1:
        raise ValueError("Need at least one query and two gallery images")
    if spec == BASELINE:
        return v25.rank(values[:n_query], values[n_query:])
    mvp, r1 = dual.unpack(values)
    # Each graph sees only its own encoder's static gallery and the current query.
    left = branch_distances(mvp, n_query, .5)
    right = branch_distances(r1, n_query, spec["r1_lambda"])
    order = fusion_order(left, right, spec["fusion"], spec["r1_weight"])
    candidate = dual.policy.rank_vectors(r1[:n_query], r1[n_query:], "raw")
    return {"order": order, "raw_order": candidate["raw_order"], "confidence": candidate["confidence"]}
