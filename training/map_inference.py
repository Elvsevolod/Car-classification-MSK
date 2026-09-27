"""Frozen v25 reference exporter; no training controller or threshold fitting."""
import argparse
from pathlib import Path

import numpy as np

from backend.core import normalize, read_rows
from backend.evaluate import validate_artifacts, write_json
from backend.rerank import KReciprocalReranker
from training import dual_role_inference as dual

SPEC = {"name": "r1w50_k20_q3_l50", "k1": 20, "k2": 3, "lambda": .5, "r1_weight": .5}


def rank(query, gallery):
    mvp, r1 = dual.unpack(np.concatenate([query, gallery]))
    mixed = normalize(np.concatenate([normalize(mvp)*np.float32(np.sqrt(.5)),
                                      normalize(r1)*np.float32(np.sqrt(.5))], axis=1))
    q, g = normalize(mixed[:len(query)]), normalize(mixed[len(query):])
    graph = KReciprocalReranker(g, min(20, len(g)-1), min(3, len(g)))
    distances = np.stack([graph.distances(v, .5) for v in q])
    raw = dual.policy.rank_vectors(r1[:len(query)], r1[len(query):], "raw")
    return {"order": np.argsort(distances, axis=1, kind="stable"),
            "raw_order": raw["raw_order"], "confidence": raw["confidence"]}


def export(profile, selection, dataset, output):
    selection = dual.read(selection)
    if selection["selected"] != SPEC or selection["selection_split"] != "calibration":
        raise ValueError("Only the frozen v25 calibration winner is supported")
    q, g = (read_rows(dataset/name) for name in ("test_query.csv", "test_gallery.csv"))
    if len(g) < 10 or output.exists():
        raise ValueError("Need ten gallery images and a new output directory")
    encoder = dual.DualRoleEncoder(profile)
    values = encoder.encode_rows(q+g, dataset)
    dual.policy.export_csv(output, q, g, rank(values[:len(q)], values[len(q):]), encoder.profile["threshold"], "raw_top1")
    np.save(output/"embeddings.npy", values)
    write_json(output/"embedding_order.json", {"ids": [r["image_id"] for r in q+g],
        "layout": dual.LAYOUT, "ranking": SPEC, "threshold": encoder.profile["threshold"]})
    return validate_artifacts(dataset, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "selection", "dataset", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(export(args.profile, args.selection, args.dataset, args.output))
