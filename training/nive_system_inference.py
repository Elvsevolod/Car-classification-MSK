"""v34 fixed inference: image-only ONNX, static-gallery ranking, frozen v25 refusals."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from backend.core import bbox, crop_image, normalize, read_rows, sha256
from backend.evaluate import validate_artifacts, write_json
from training import dual_role_inference as dual, frozen_inference as frozen, map_inference
from training.preprocessing import resize_crop, IMAGENET_MEAN, IMAGENET_STD

SYSTEMS = {
    "V25_control": {"member": None, "graph": "legacy", "weights": [.5, .5]},
    "R1_N0_pair": {"member": "N0", "graph": "less_graph", "weights": [.5, .5]},
    "R1_N1_pair": {"member": "N1", "graph": "less_graph", "weights": [.5, .5]},
    "MVP_R1_N0": {"member": "N0", "graph": "legacy", "weights": [.5, .25, .25]},
    "MVP_R1_N1": {"member": "N1", "graph": "legacy", "weights": [.5, .25, .25]},
}
LAYOUT = {"v25_mvp": [0, 512], "v25_candidates": [512, 2048], "parent": [2048, 2560],
          "member": [2560, 3072], "normalization": "unit blocks; real feature bank, not a single ranking cosine"}


def rank(values, n_query, system):
    if system not in SYSTEMS or not 0 < n_query < len(values):
        raise ValueError("Unknown fixed system or invalid query count")
    expected = 2048 if system == "V25_control" else 3072
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError("Wrong feature-bank layout")
    mvp, candidate = dual.unpack(values[:, :2048])
    if system == "V25_control":
        return map_inference.rank(values[:n_query], values[n_query:])
    parent, member = values[:, 2048:2560], values[:, 2560:3072]
    dual.validate_block(parent, 512); dual.validate_block(member, 512)
    blocks = [parent, member] if system.startswith("R1_") else [mvp, parent, member]
    mixed = normalize(np.concatenate([normalize(v)*np.float32(np.sqrt(w))
                                      for v, w in zip(blocks, SYSTEMS[system]["weights"])], axis=1))
    ranked = dual.policy.rank_vectors(mixed[:n_query], mixed[n_query:], SYSTEMS[system]["graph"])
    accepted = dual.policy.rank_vectors(candidate[:n_query], candidate[n_query:], "raw")
    return {"order": ranked["order"], "raw_order": accepted["raw_order"], "confidence": accepted["confidence"]}


class ImageEncoder:
    def __init__(self, path, checksum, provider="CPUExecutionProvider"):
        if provider != "CPUExecutionProvider":
            raise ValueError("v34 prototype is CPU-only; no fallback")
        if sha256(path) != checksum:
            raise ValueError("Research encoder checksum mismatch")
        self.session = frozen._session(path, provider)
        self.spec = frozen._model_spec(self.session, 256)
        if self.spec["dimension"] != 512:
            raise ValueError("Expected a head-free 512D encoder")

    def encode_rows(self, rows, dataset, batch_size=16):
        if type(batch_size) is not int or batch_size < 1 or not rows:
            raise ValueError("Need nonempty rows and positive batch size")
        blocks = []
        for start in range(0, len(rows), batch_size):
            inputs = []
            for row in rows[start:start+batch_size]:
                with Image.open(dual.image_path(dataset, row["image_id"])) as image:
                    crop = resize_crop(crop_image(image, bbox(row)), "square", 256)
                    pixels = (np.asarray(crop, np.float32)/np.float32(255)-IMAGENET_MEAN)/IMAGENET_STD
                    inputs.append(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
            values = self.session.run([self.spec["output"]], {self.spec["input"]: np.stack(inputs)})[0]
            if values.shape != (len(inputs), 512):
                raise ValueError("Unexpected image encoder output")
            values = normalize(values)
            dual.validate_block(values, 512)
            blocks.append(values)
        return np.concatenate(blocks)


def load_bundle(path):
    path = Path(path).resolve(); bundle = dual.read(path)
    if (bundle.get("schema") != "nive-system-v34" or bundle.get("systems") != SYSTEMS
            or bundle.get("layout") != LAYOUT or bundle.get("promoted") is not False
            or bundle.get("preprocessing") != frozen._preprocessing(256, "square")
            or set(bundle["encoders"]) != {"parent", "N0", "N1"}):
        raise ValueError("Changed frozen v34 bundle")
    paths = {name: (path.parent / entry["path"]).resolve() for name, entry in bundle["encoders"].items()}
    profile_path = (path.parent / bundle["v25_profile"]["path"]).resolve()
    for name, p in paths.items():
        if sha256(p) != bundle["encoders"][name]["sha256"]:
            raise ValueError("Changed encoder in bundle")
    if sha256(profile_path) != bundle["v25_profile"]["sha256"]:
        raise ValueError("Changed original candidate profile")
    profile, _, _ = dual.load_profile(profile_path)
    if profile["threshold"] != bundle["threshold"]:
        raise ValueError("Candidate threshold must remain v25's own frozen threshold")
    return bundle, paths, profile_path


class SystemEncoder:
    def __init__(self, bundle_path, system):
        if system not in SYSTEMS:
            raise ValueError("Unknown fixed system")
        self.bundle, paths, profile = load_bundle(bundle_path)
        self.original = dual.DualRoleEncoder(profile)
        member = SYSTEMS[system]["member"]
        names = [] if member is None else ["parent", member]
        self.encoders = [ImageEncoder(paths[n], self.bundle["encoders"][n]["sha256"]) for n in names]

    def encode_rows(self, rows, dataset, batch_size=16):
        blocks = [self.original.encode_rows(rows, dataset, batch_size)]
        blocks += [encoder.encode_rows(rows, dataset, batch_size) for encoder in self.encoders]
        return np.concatenate(blocks, axis=1)


def export_arrays(output, query, gallery, values, system, threshold):
    """Atomic three-file export, with explicit raw-bank layout and exact decision replay."""
    output = Path(output)
    if output.exists() or len(values) != len(query)+len(gallery) or len(gallery) < 10:
        raise ValueError("Use a new output with matching rows and at least ten gallery images")
    ranked = rank(values, len(query), system)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nive_pending_", dir=output.parent) as tmp:
        directory = Path(tmp) / "export"
        metrics = dual.policy.export_csv(directory, query, gallery, ranked, threshold, "raw_top1")
        np.save(directory / "embeddings.npy", values)
        write_json(directory / "embedding_order.json", {"ids": [r["image_id"] for r in query+gallery],
                   "query_count": len(query), "gallery_count": len(gallery), "dimension": values.shape[1],
                   "layout": dual.LAYOUT if system == "V25_control" else LAYOUT,
                   "system": system, "ranking": SYSTEMS[system], "threshold": threshold,
                   "candidate": "unchanged v25 R1 equal3 raw_top1", "sha256": sha256(directory / "embeddings.npy")})
        replay = rank(np.load(directory / "embeddings.npy", allow_pickle=False), len(query), system)
        if dual.policy.predictions(query, gallery, replay, threshold, "raw_top1") != dual.policy.predictions(
                query, gallery, ranked, threshold, "raw_top1"):
            raise ValueError("NPY replay changed decisions")
        directory.rename(output)
    return metrics


def export(bundle, dataset, output, system):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Use a new export directory")
    query, gallery = (read_rows(dataset/name) for name in ("test_query.csv", "test_gallery.csv"))
    encoder = SystemEncoder(bundle, system)
    values = encoder.encode_rows(query+gallery, dataset)
    export_arrays(output, query, gallery, values, system, encoder.bundle["threshold"])
    return validate_artifacts(dataset, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    args = parser.parse_args()
    print(export(args.bundle, args.dataset, args.output, args.system))
