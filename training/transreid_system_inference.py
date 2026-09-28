"""v37: fixed image-feature mixtures; v25 candidate/confidence remain independent."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from pathlib import Path
import tempfile

import numpy as np

from backend.core import normalize, sha256
from backend.evaluate import write_json
from backend.rerank import KReciprocalReranker
from training import dual_role_inference as dual, map_inference

DIMENSIONS = {"T12": 384, "T22": 1920}
# Exact ties: control, smaller new-model weight, then global before JPM.
SYSTEMS = {"V25_control": {"member": None, "weight": 0.0}}
SYSTEMS.update({f"{member}_w{percent:02d}": {"member": member, "weight": percent/100}
                for percent in (5, 10, 15, 20) for member in DIMENSIONS})
GRAPH = {"k1": 20, "k2": 3, "lambda": .5}


def ranking_features(values, system):
    if system not in SYSTEMS:
        raise ValueError("Unknown frozen v37 system")
    spec = SYSTEMS[system]
    dimension = 2048 + DIMENSIONS.get(spec["member"], 0)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError("Wrong real feature-bank layout")
    mvp, r1 = dual.unpack(values[:, :2048])
    original = normalize(np.concatenate([normalize(mvp)*np.float32(np.sqrt(.5)),
                                         normalize(r1)*np.float32(np.sqrt(.5))], axis=1))
    if spec["member"] is None:
        return original
    extra = values[:, 2048:]
    dual.validate_block(extra, DIMENSIONS[spec["member"]])
    w = spec["weight"]
    return normalize(np.concatenate([original*np.float32(np.sqrt(1-w)),
                                     normalize(extra)*np.float32(np.sqrt(w))], axis=1))


def rank(values, n_query, system, *, progress=False):
    if not 0 < n_query < len(values):
        raise ValueError("Invalid query/gallery count")
    mixed = ranking_features(values, system)
    if system == "V25_control":
        return map_inference.rank(values[:n_query], values[n_query:])
    q, g = normalize(mixed[:n_query]), normalize(mixed[n_query:])
    if len(g) < 10:
        raise ValueError("Official top-10 requires at least ten gallery images")
    graph = KReciprocalReranker(g, min(GRAPH["k1"], len(g)-1), min(GRAPH["k2"], len(g)))
    orders = []
    for i, vector in enumerate(q):
        orders.append(np.argsort(graph.distances(vector, GRAPH["lambda"]), kind="stable"))
        if progress and ((i+1) % 32 == 0 or i+1 == len(q)):
            print(f"RANK {system}: {i+1}/{len(q)} queries", flush=True)
    _, candidate = dual.unpack(values[:, :2048])
    raw = dual.policy.rank_vectors(candidate[:n_query], candidate[n_query:], "raw")
    return {"order": np.stack(orders), "raw_order": raw["raw_order"], "confidence": raw["confidence"]}


def export_arrays(output, query, gallery, values, system, threshold):
    """Three real artifacts plus explicit layout; NPY replay must preserve decisions."""
    output = Path(output)
    if output.exists() or len(values) != len(query)+len(gallery) or len(gallery) < 10:
        raise ValueError("Need a new output directory and a complete protocol")
    ranked = rank(values, len(query), system)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="transreid_export_", dir=output.parent) as tmp:
        directory = Path(tmp) / "export"
        metrics = dual.policy.export_csv(directory, query, gallery, ranked, threshold, "raw_top1")
        np.save(directory / "embeddings.npy", values)
        write_json(directory / "embedding_order.json", {
            "ids": [r["image_id"] for r in query+gallery], "query_count": len(query), "gallery_count": len(gallery),
            "dimension": values.shape[1], "layout": {"mvp": [0, 512], "candidate_r1": [512, 2048],
                "extra": [2048, values.shape[1]], "normalization": "unit blocks, not one ranking cosine"},
            "system": system, "mixture": SYSTEMS[system], "graph": GRAPH, "threshold": threshold,
            "candidate": "unchanged v25 R1 equal3 raw_top1", "sha256": sha256(directory / "embeddings.npy")})
        replay = rank(np.load(directory / "embeddings.npy", allow_pickle=False), len(query), system)
        before = dual.policy.predictions(query, gallery, ranked, threshold, "raw_top1")
        after = dual.policy.predictions(query, gallery, replay, threshold, "raw_top1")
        if before != after:
            raise ValueError("NPY replay changed top-10/candidate/confidence/refusal")
        directory.rename(output)
    return metrics
