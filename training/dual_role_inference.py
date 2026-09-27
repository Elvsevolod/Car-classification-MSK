"""Frozen CPU inference: MVP ranks, full-train R1 accepts/refuses. No training imports."""
import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from backend.core import Encoder, PREPROCESS, bbox, preprocess, read_rows, sha256
from backend.evaluate import validate_artifacts, write_json
from training.frozen_inference import FrozenEncoder
from training import retrieval_policy as policy

LAYOUT = {"dimension": 2048, "ranking_slice": [0, 512], "candidate_slice": [512, 2048],
          "normalization": "unit blocks, no global renormalization"}
ROLES = {"ranking": "MVP legacy k20/k3/lambda0.50", "candidate": "R1 equal3 raw_top1",
         "confidence": "maximum raw cosine of R1 block only"}


def read(path):
    return json.loads(Path(path).read_text())


def load_profile(path):
    path = Path(path).resolve()
    value = read(path)
    if (value.get("schema") != "dual-role-v1" or value.get("layout") != LAYOUT
            or value.get("roles") != ROLES or value.get("mvp_preprocessing") != PREPROCESS
            or not np.isfinite(value["threshold"]) or value.get("promoted") is not False):
        raise ValueError("Unsupported frozen dual-role profile")
    paths = {}
    for key in ("mvp", "r1_bundle"):
        paths[key] = (path.parent / value[key]["path"]).resolve()
        if sha256(paths[key]) != value[key]["sha256"]:
            raise ValueError(f"Frozen {key} checksum mismatch")
    r1 = read(paths["r1_bundle"])
    if (r1.get("schema") != 2 or len(r1["members"]) != 3
            or r1["fusion"] != "equal normalized concatenation"
            or r1["ranking"] != "less_graph" or r1["ranking_parameters"] != policy.POLICIES["less_graph"]
            or r1["candidate_policy"] != "raw_top1" or r1["threshold"] != value["threshold"]
            or r1.get("confidence") != "maximum raw cosine of combined features"
            or r1["calibration"] != value["calibration"] or value["calibration"]["split"] != "calibration"
            or value["calibration"].get("candidate_policy") != "raw_top1"
            or not value["calibration"].get("method")
            or not re.fullmatch(r"[0-9a-f]{64}", value["calibration"].get("protocol_sha256", ""))):
        raise ValueError("R1 profile/threshold differs from the frozen source")
    return value, paths, r1


def profile_value(output, mvp, r1_bundle):
    """Describe existing weights/threshold, never fit or copy them."""
    output, mvp, r1_bundle = (Path(p).resolve() for p in (output, mvp, r1_bundle))
    r1 = read(r1_bundle)
    return {"schema": "dual-role-v1", "layout": LAYOUT, "roles": ROLES, "mvp_preprocessing": PREPROCESS,
            "mvp": {"path": os.path.relpath(mvp, output.parent), "sha256": sha256(mvp)},
            "r1_bundle": {"path": os.path.relpath(r1_bundle, output.parent), "sha256": sha256(r1_bundle)},
            "threshold": r1["threshold"], "calibration": r1["calibration"], "promoted": False}


def validate_block(values, dimension):
    if (values.ndim != 2 or values.shape[1] != dimension or not len(values) or values.dtype != np.float32
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0, atol=2e-5)):
        raise ValueError("Expected finite real float32 unit embedding block")


def pack(mvp, r1):
    validate_block(mvp, 512)
    validate_block(r1, 1536)
    if len(mvp) != len(r1):
        raise ValueError("Role blocks must have identical row order/count")
    return np.concatenate([mvp, r1], axis=1)  # Lossless float32 block storage, norm sqrt(2).


def unpack(values):
    if values.ndim != 2 or values.shape[1] != 2048:
        raise ValueError("Expected MVP512 + R1_1536 embedding layout")
    mvp, r1 = np.ascontiguousarray(values[:, :512]), np.ascontiguousarray(values[:, 512:])
    validate_block(mvp, 512)
    validate_block(r1, 1536)
    return mvp, r1


def rank(query_vectors, gallery_vectors):
    """One scorer for cached evaluation, image inference and NPY replay; no labels."""
    qm, qr = unpack(query_vectors)
    gm, gr = unpack(gallery_vectors)
    ranking = policy.rank_vectors(qm, gm, "legacy")
    candidate = policy.rank_vectors(qr, gr, "raw")
    return {"order": ranking["order"], "raw_order": candidate["raw_order"], "confidence": candidate["confidence"]}


def image_path(dataset, identifier):
    directory = Path(dataset) / "images"
    # Enumerate actual names: .jpg/.JPG may resolve to the same file on macOS.
    matches = [p for p in directory.glob(f"{identifier}.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one JPEG/PNG for {identifier}; found {len(matches)}")
    return matches[0]


class DualRoleEncoder:
    def __init__(self, profile_path, provider="CPUExecutionProvider"):
        if provider != "CPUExecutionProvider":
            raise ValueError("This quality prototype supports explicit CPU only; no provider fallback")
        self.profile, paths, r1 = load_profile(profile_path)
        self.mvp = Encoder(paths["mvp"])
        if self.mvp.session.get_providers() != [provider]:
            raise ValueError("MVP provider differs from requested CPU")
        self.members = []
        for member in r1["members"]:
            path = (paths["r1_bundle"].parent / member["path"]).resolve()
            if sha256(path) != member["sha256"]:
                raise ValueError("R1 member bundle checksum mismatch")
            self.members.append(FrozenEncoder(path, provider))
        if (len({e.model_sha256 for e in self.members}) != 3
                or any(e.dimension != 512 or e.size != 256 for e in self.members)
                or any(e.bundle["preprocessing"] != self.members[0].bundle["preprocessing"] for e in self.members)
                or self.members[0].bundle["preprocessing"]["resize_mode"] != "square"):
            raise ValueError("Expected three distinct R1/256/square/512 encoders")

    def encode_rows(self, rows, dataset, batch_size=16):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        values = []
        for start in range(0, len(rows), batch_size):
            mvp_inputs, r1_inputs = [], []
            for row in rows[start:start+batch_size]:
                with Image.open(image_path(dataset, row["image_id"])) as image:
                    mvp_inputs.append(preprocess(image, bbox(row)))
                    r1_inputs.append(self.members[0].preprocess(image, bbox(row)))
            mvp = self.mvp.encode_batch(mvp_inputs)
            r1 = policy.combine_members([e.encode_batch(r1_inputs) for e in self.members])
            values.append(pack(mvp, r1))
            if start % (batch_size*10) == 0 or start+batch_size >= len(rows):
                print(f"Dual-role CPU: {min(start+batch_size,len(rows))}/{len(rows)} images", flush=True)
        return np.concatenate(values)


def export_arrays(profile, query, gallery, values, output):
    """Publish all artifacts atomically; refuse to overwrite a different completed export."""
    output = Path(output)
    if len(values) != len(query)+len(gallery) or len(gallery) < 10:
        raise ValueError("Embedding count/protocol mismatch or fewer than ten gallery rows")
    ranking = rank(values[:len(query)], values[len(query):])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dual_pending_", dir=output.parent) as temporary:
        directory = Path(temporary) / "export"
        metrics = policy.export_csv(directory, query, gallery, ranking, profile["threshold"], "raw_top1")
        np.save(directory / "embeddings.npy", values)
        write_json(directory / "embedding_order.json", {"ids": [r["image_id"] for r in query+gallery],
                   "query_count": len(query), "gallery_count": len(gallery), "layout": LAYOUT,
                   "roles": ROLES, "threshold": profile["threshold"], "sha256": sha256(directory / "embeddings.npy")})
        replay = np.load(directory / "embeddings.npy", allow_pickle=False)
        before = policy.predictions(query, gallery, ranking, profile["threshold"], "raw_top1")
        after = policy.predictions(query, gallery, rank(replay[:len(query)], replay[len(query):]), profile["threshold"], "raw_top1")
        if before != after:
            raise ValueError("NPY replay changed ranking/candidate/confidence decisions")
        if output.exists():
            if not output.is_dir() or any(not (output/p.name).is_file() or sha256(output/p.name) != sha256(p)
                                         for p in directory.iterdir()):
                raise ValueError("Existing export differs; use a new output directory")
        else:
            directory.rename(output)
    return metrics


def export_frozen(profile_path, dataset, output, batch_size=16, provider="CPUExecutionProvider"):
    """Known one-command contract: only images + test CSVs, no train/calibration/DB/network."""
    dataset = Path(dataset)
    query, gallery = (read_rows(dataset / f"test_{split}.csv") for split in ("query", "gallery"))
    encoder = DualRoleEncoder(profile_path, provider)
    values = encoder.encode_rows(query+gallery, dataset, batch_size)
    export_arrays(encoder.profile, query, gallery, values, output)
    return validate_artifacts(dataset, Path(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(export_frozen(args.profile, args.dataset, args.output, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
