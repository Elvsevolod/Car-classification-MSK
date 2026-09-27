"""v28: four fixed ranking families, immutable original R1 candidate branch."""
from itertools import combinations

import numpy as np
from PIL import Image

from backend.core import bbox, normalize, preprocess
from backend.rerank import KReciprocalReranker
from training import dual_role_inference as dual, map_inference as v25

BASELINE = {"name": "V25_control", "family": "control", "lambda": .5}
FAMILIES = ("members", "power", "center", "flip")


def systems():
    grid = [dict(BASELINE)]
    for size in (1, 2):
        for members in combinations(range(3), size):
            for w in (.5, .75):
                for lam in (.5, .75):
                    grid.append({"name": f"members_{''.join(str(i) for i in members)}_w{int(w*100)}_l{int(lam*100)}",
                                 "family": "members", "members": list(members), "r1_weight": w, "lambda": lam})
    for family, key, values in (("power", "gamma", (.5, .75, 1.25)),
                                ("center", "strength", (.25, .5, .75, 1.))):
        for value in values:
            for lam in (.5, .75):
                grid.append({"name": f"{family}_{int(value*100)}_l{int(lam*100)}", "family": family,
                             key: value, "lambda": lam})
    for scope in ("mvp", "r1", "both"):
        for w in (.25, .5, .75):
            for lam in (.5, .75):
                grid.append({"name": f"flip_{scope}_w{int(w*100)}_l{int(lam*100)}", "family": "flip",
                             "scope": scope, "flip_weight": w, "lambda": lam})
    return grid


def members(r1):
    return [normalize(block) for block in np.split(r1, 3, axis=1)]


def mix(left, right, weight=.5):
    return normalize(np.concatenate([normalize(left)*np.float32(np.sqrt(1-weight)),
                                     normalize(right)*np.float32(np.sqrt(weight))], axis=1))


def ranking_features(values, n_query, spec):
    mvp, r1 = dual.unpack(values[:, :2048])
    family = spec["family"]
    if family == "members":
        parts = members(r1)
        r1 = normalize(np.concatenate([parts[i] for i in spec["members"]], axis=1))
    elif family == "power":
        mvp, r1 = [normalize(np.sign(v)*np.abs(v)**np.float32(spec["gamma"])) for v in (mvp, r1)]
    elif family == "center":
        # Fit solely to this static gallery. Neither query batch nor labels enter the mean.
        mvp, r1 = [normalize(v-np.float32(spec["strength"])*v[n_query:].mean(axis=0)) for v in (mvp, r1)]
    elif family == "flip":
        fm, fr = dual.unpack(values[:, 2048:])
        w = np.float32(spec["flip_weight"])
        if spec["scope"] in ("mvp", "both"):
            mvp = normalize((1-w)*mvp+w*fm)
        if spec["scope"] in ("r1", "both"):
            # Average only within the same encoder's coordinate system, not between different seeds.
            r1 = dual.policy.combine_members([normalize((1-w)*a+w*b) for a, b in zip(members(r1), members(fr))])
    return mix(mvp, r1, spec.get("r1_weight", .5))


def rank(values, n_query, spec):
    if spec not in systems():
        raise ValueError("Ranking specification is outside the fixed v28 grid")
    dimension = 4096 if spec["family"] == "flip" else 2048
    if (values.ndim != 2 or values.shape[1] != dimension
            or type(n_query) is not int or not 0 < n_query < len(values)-1):
        raise ValueError("Wrong original/flip layout or query/gallery counts")
    mvp, r1 = dual.unpack(values[:, :2048])
    if spec == BASELINE:
        return v25.rank(values[:n_query], values[n_query:])
    features = ranking_features(values, n_query, spec)
    q, g = normalize(features[:n_query]), normalize(features[n_query:])
    graph = KReciprocalReranker(g, min(20, len(g)-1), min(3, len(g)))
    distances = np.stack([graph.distances(v, spec["lambda"]) for v in q])
    candidate = dual.policy.rank_vectors(r1[:n_query], r1[n_query:], "raw")
    return {"order": np.argsort(distances, axis=1, kind="stable"),
            "raw_order": candidate["raw_order"], "confidence": candidate["confidence"]}


def encode_rows(encoder, rows, dataset, *, flip, batch_size=16):
    """Same organizer crop/preprocessing; optionally mirror each prepared image horizontally."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive batch size required")
    blocks = []
    for start in range(0, len(rows), batch_size):
        mi, ri = [], []
        for row in rows[start:start+batch_size]:
            with Image.open(dual.image_path(dataset, row["image_id"])) as image:
                mi.append(preprocess(image, bbox(row)))
                ri.append(encoder.members[0].preprocess(image, bbox(row)))
        if flip:
            mi, ri = [np.ascontiguousarray(np.stack(batch)[..., ::-1]) for batch in (mi, ri)]
        blocks.append(dual.pack(encoder.mvp.encode_batch(mi),
                               dual.policy.combine_members([e.encode_batch(ri) for e in encoder.members])))
    return np.concatenate(blocks)
