"""v30: restrict the frozen v25 graph order to a raw-fusion pool; no training imports."""
import numpy as np

from backend.core import normalize
from training import map_inference as v25

BASELINE = {"name": "V25_control", "pool_k": None}


def systems():
    return [dict(BASELINE)] + [{"name": f"raw_pool{k}", "pool_k": k} for k in (10, 20, 50)]


def restrict_order(raw_order, graph_order, pool_k):
    """Reorder only the raw prefix by full-gallery graph position; keep the raw tail."""
    if type(pool_k) is not int or pool_k <= 0 or raw_order.shape != graph_order.shape or raw_order.ndim != 2:
        raise ValueError("Expected matching ranking matrices and a positive integer pool size")
    k = min(pool_k, raw_order.shape[1])
    pool = raw_order[:, :k]
    positions = np.argsort(graph_order, axis=1)
    keys = np.take_along_axis(positions, pool, axis=1)
    prefix = np.take_along_axis(pool, np.argsort(keys, axis=1, kind="stable"), axis=1)
    return np.concatenate([prefix, raw_order[:, k:]], axis=1)


def rank(values, n_query, spec):
    if spec not in systems() or type(n_query) is not int or not 0 < n_query < len(values):
        raise ValueError("Use a frozen v30 system and nonempty query/gallery")
    mvp, r1 = v25.dual.unpack(values)
    control = v25.rank(values[:n_query], values[n_query:])
    if spec == BASELINE:
        return control
    # Identical feature construction to v25, but no graph in this pool-selection step.
    mixed = normalize(np.concatenate([normalize(mvp)*np.float32(np.sqrt(.5)),
                                      normalize(r1)*np.float32(np.sqrt(.5))], axis=1))
    q, g = normalize(mixed[:n_query]), normalize(mixed[n_query:])
    # Each query is independent of its neighbours and of the caller's batch size.
    similarities = np.stack([g @ vector for vector in q])
    raw_fusion_order = np.argsort(-similarities, axis=1, kind="stable")
    return {**control, "order": restrict_order(raw_fusion_order, control["order"], spec["pool_k"])}
