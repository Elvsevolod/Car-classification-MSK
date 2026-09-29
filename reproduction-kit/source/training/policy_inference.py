"""Schema-2 policy bundle: independent ranking/candidate policy, 1 or 3 encoders."""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from backend.core import read_rows, sha256
from backend.evaluate import validate_artifacts
from training.audit import digest
from training.stage6 import write_json
from training import frozen_inference as frozen
from training import retrieval_policy as policy


def validate_bundle(bundle):
    if (bundle.get("schema") != 2 or bundle.get("ranking") not in policy.POLICIES
            or bundle.get("ranking_parameters") != policy.POLICIES[bundle["ranking"]]
            or bundle.get("candidate_policy") not in policy.CANDIDATES
            or bundle.get("fusion") != "equal normalized concatenation"
            or bundle.get("confidence") != "maximum raw cosine of combined features"
            or not np.isfinite(bundle["threshold"]) or len(bundle["members"]) not in (1, 3)):
        raise ValueError("Unsupported policy bundle/schema")
    if len({m["sha256"] for m in bundle["members"]}) != len(bundle["members"]):
        raise ValueError("Duplicate ensemble member")
    c = bundle["calibration"]
    if (c.get("split") != "calibration" or c.get("candidate_policy") != bundle["candidate_policy"]
            or not c.get("method") or not re.fullmatch(r"[0-9a-f]{64}", c.get("protocol_sha256", ""))):
        raise ValueError("Policy needs its own calibration-only provenance")


def write_bundle(path, members, *, ranking, candidate_policy, threshold, calibration):
    path = Path(path).resolve()
    bundle = {"schema": 2, "members": [{"path": os.path.relpath(Path(p).resolve(), path.parent),
               "sha256": sha256(p)} for p in members], "ranking": ranking,
              "ranking_parameters": policy.POLICIES[ranking], "candidate_policy": candidate_policy,
              "threshold": float(threshold), "calibration": calibration,
              "fusion": "equal normalized concatenation",
              "confidence": "maximum raw cosine of combined features"}
    validate_bundle(bundle)
    if path.exists():
        if json.loads(path.read_text()) != bundle:
            raise ValueError("Refusing to replace frozen policy bundle")
    else:
        write_json(path, bundle)
    return bundle


class PolicyEncoder:
    def __init__(self, bundle_path, provider="CPUExecutionProvider"):
        self.bundle_path = Path(bundle_path).resolve()
        self.bundle = json.loads(self.bundle_path.read_text())
        validate_bundle(self.bundle)
        paths = [(self.bundle_path.parent / m["path"]).resolve() for m in self.bundle["members"]]
        if any(sha256(p) != m["sha256"] for p, m in zip(paths, self.bundle["members"])):
            raise ValueError("Source encoder bundle checksum mismatch")
        self.members = [frozen.FrozenEncoder(p, provider) for p in paths]
        first = self.members[0]
        if any(e.bundle["preprocessing"] != first.bundle["preprocessing"] or e.dimension != first.dimension
               for e in self.members):
            raise ValueError("Ensemble requires matching preprocessing and dimensions")
        if len({e.model_sha256 for e in self.members}) != len(self.members):
            raise ValueError("Duplicate encoder weights")
        self.size, self.dimension = first.size, sum(e.dimension for e in self.members)
        self.provider = provider
        self.fingerprint = digest({"bundle": self.bundle, "members": [e.fingerprint for e in self.members]})

    def preprocess(self, image, box):
        return self.members[0].preprocess(image, box)

    def encode_batch(self, batch):
        return policy.combine_members([e.encode_batch(batch) for e in self.members])

    def encode(self, image, box):
        return self.encode_batch([self.preprocess(image, box)])[0]


def export_frozen(bundle_path, dataset, output, provider="CPUExecutionProvider", batch_size=16):
    """No labels, training CSV, calibration, or other query affects a prediction."""
    output, dataset = Path(output), Path(dataset)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty export directory")
    encoder = PolicyEncoder(bundle_path, provider)
    query, gallery = (read_rows(dataset / name) for name in ("test_query.csv", "test_gallery.csv"))
    qv, gv = (frozen._encode_rows(encoder, rows, dataset, batch_size) for rows in (query, gallery))
    ranking = policy.rank_vectors(qv, gv, encoder.bundle["ranking"])
    policy.export_csv(output, query, gallery, ranking, encoder.bundle["threshold"], encoder.bundle["candidate_policy"])
    np.save(output / "embeddings.npy", np.concatenate([qv, gv]).astype(np.float32))
    validation = validate_artifacts(dataset, output)
    write_json(output / "export_manifest.json", {"schema": 2, "bundle": encoder.bundle,
               "bundle_sha256": sha256(bundle_path), "fingerprint": encoder.fingerprint, "provider": provider,
               "query_csv_sha256": sha256(dataset / "test_query.csv"),
               "gallery_csv_sha256": sha256(dataset / "test_gallery.csv"),
               "image_sha256": {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg") for r in query + gallery},
               "embedding_ids": [r["image_id"] for r in query + gallery],
               "dimension": encoder.dimension, "encoder_forward_count": len(encoder.members),
               "validation": validation, "promoted": False, "official_gpu_verified": False})
    return validation


def compare_decisions(queries, gallery, before, after, ranking_policy, threshold, candidate_policy, atol=2e-4):
    """Decision parity, in addition to tensor proximity, on the SAME fixed protocol."""
    left = policy.rank_vectors(*before, ranking_policy)
    right = policy.rank_vectors(*after, ranking_policy)
    lp, lc = policy.predictions(queries, gallery, left, threshold, candidate_policy)
    rp, rc = policy.predictions(queries, gallery, right, threshold, candidate_policy)
    changed_top10 = [q for q in lp if lp[q] != rp[q]]
    changed_candidates = [q["image_id"] for q in queries if
                          [x[0] for x in lc.get(q["image_id"], [])] != [x[0] for x in rc.get(q["image_id"], [])]]
    error = max(float(np.max(np.abs(a - b))) for a, b in zip(before, after))
    return {"passed": error <= atol and not changed_top10 and not changed_candidates,
            "max_abs_error": error, "atol": atol, "changed_top10": changed_top10,
            "changed_candidates_or_acceptance": changed_candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=frozen.PROVIDERS, default=frozen.PROVIDERS[0])
    args = parser.parse_args()
    print(json.dumps(export_frozen(args.bundle, args.dataset, args.output, args.provider), indent=2))


if __name__ == "__main__":
    main()
