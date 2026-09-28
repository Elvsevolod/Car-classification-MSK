"""v39: rerank the saved Full T12 mixture; never change the v25 candidate branch."""
from pathlib import Path
import tempfile

import numpy as np

from training import transreid_system_inference as previous

policy = previous.dual.policy
# Exact calibration ties keep this order, with the deployed control first.
SYSTEMS = {
    "V25_control": {"source": "V25_control", "mixture": "V25_control", "policy": "legacy", "weight": 0.0},
    "Full_T12_l50": {"source": "Full_T12_w10", "mixture": "T12_w10", "policy": "legacy", "weight": .1},
    "Full_T12_l75": {"source": "Full_T12_w10", "mixture": "T12_w10", "policy": "less_graph", "weight": .1},
    "Full_T12_raw": {"source": "Full_T12_w10", "mixture": "T12_w10", "policy": "raw", "weight": .1},
}


def rank(values, n_query, system, *, progress=False):
    if system not in SYSTEMS or not 0 < n_query <= len(values)-10:
        raise ValueError("Need a declared system, nonempty query and at least ten gallery images")
    spec = SYSTEMS[system]
    if spec["policy"] == "legacy":
        return previous.rank(values, n_query, spec["mixture"], progress=progress)
    mixed = previous.ranking_features(values, spec["mixture"])
    ranked = policy.rank_vectors(mixed[:n_query], mixed[n_query:], spec["policy"])
    _, candidate = previous.dual.unpack(values[:, :2048])
    raw = policy.rank_vectors(candidate[:n_query], candidate[n_query:], "raw")
    if progress:
        print(f"RANK {system}: {n_query}/{n_query} queries", flush=True)
    return {"order": ranked["order"], "raw_order": raw["raw_order"], "confidence": raw["confidence"]}


def export_arrays(output, query, gallery, values, system, threshold):
    """Three real contest artifacts, explicit block layout and exact decision replay."""
    output = Path(output)
    if output.exists() or len(values) != len(query)+len(gallery):
        raise ValueError("Need a new output directory and a complete protocol")
    ranked = rank(values, len(query), system)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="graph_export_", dir=output.parent) as tmp:
        directory = Path(tmp) / "export"
        metrics = policy.export_csv(directory, query, gallery, ranked, threshold, "raw_top1")
        np.save(directory / "embeddings.npy", values)
        previous.write_json(directory / "embedding_order.json", {
            "ids": [r["image_id"] for r in query+gallery], "query_count": len(query), "gallery_count": len(gallery),
            "dimension": values.shape[1], "layout": {"mvp": [0, 512], "candidate_r1": [512, 2048],
                "extra": [2048, values.shape[1]], "normalization": "unit blocks, not one ranking cosine"},
            "system": system, "spec": SYSTEMS[system], "graph": policy.POLICIES[SYSTEMS[system]["policy"]],
            "threshold": threshold, "candidate": "unchanged v25 R1 equal3 raw_top1",
            "sha256": previous.sha256(directory / "embeddings.npy")})
        replay = rank(np.load(directory / "embeddings.npy", allow_pickle=False), len(query), system)
        if policy.predictions(query, gallery, ranked, threshold, "raw_top1") != policy.predictions(
                query, gallery, replay, threshold, "raw_top1"):
            raise ValueError("NPY replay changed top-10/candidate/confidence/refusal")
        directory.rename(output)
    return metrics
