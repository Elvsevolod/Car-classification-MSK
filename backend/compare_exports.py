"""Compare repeat runs (exact by default) or CPU/CUDA exports with an explicit tolerance."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import validate_artifacts
from .runtime import write_json


def compare_exports(dataset, first, second, atol=0.):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    for path in (first, second):
        validate_artifacts(dataset, path)
    manifests = [json.loads((p / "export_manifest.json").read_text()) for p in (first, second)]
    for key in ("profile", "profile_fingerprint", "encoder_fingerprint", "embedding_ids",
                "query_csv_sha256", "gallery_csv_sha256", "image_sha256"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Inputs/model differ: {key}")
    if (first / "submission.csv").read_bytes() != (second / "submission.csv").read_bytes():
        raise ValueError("Top-10 IDs/order differ")
    candidates = []
    for path in (first, second):
        with (path / "candidates.csv").open(newline="") as stream:
            candidates.append(list(csv.DictReader(stream)))
    identities = [[(r["query_id"], r["gallery_id"]) for r in rows] for rows in candidates]
    if identities[0] != identities[1]:
        raise ValueError("Accepted candidate IDs/refusals differ")
    scores = [np.array([float(r["confidence"]) for r in rows]) for rows in candidates]
    if not np.allclose(*scores, atol=atol, rtol=0):
        raise ValueError("Candidate confidence differs beyond tolerance")
    vectors = [np.load(p / "embeddings.npy", allow_pickle=False) for p in (first, second)]
    if vectors[0].shape != vectors[1].shape or not np.allclose(*vectors, atol=atol, rtol=0):
        raise ValueError("Embeddings differ beyond tolerance")
    return {"passed": True, "atol": atol, "exact_embeddings": bool(np.array_equal(*vectors)),
            "max_embedding_absolute_difference": float(np.max(np.abs(vectors[0] - vectors[1]))),
            "top10_and_candidate_ids_exact": True, "accepted_queries": len(candidates[0]),
            "providers": [m["provider"] for m in manifests]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new report path")
    result = compare_exports(args.dataset, args.first, args.second, args.atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
