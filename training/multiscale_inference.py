"""v31: same frozen R1 weights at larger input sizes; no training imports or labels."""
import hashlib
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

from backend.core import bbox, crop_image, normalize, sha256
from backend.rerank import KReciprocalReranker
from training import dual_role_inference as dual, frozen_inference as frozen, map_inference as v25
from training.preprocessing import IMAGENET_MEAN, IMAGENET_STD, resize_crop

SIZES = (320, 384)
BASELINE = {"name": "V25_control", "size": 256, "high_weight": 0.}


def systems():
    return [dict(BASELINE)] + [{"name": f"r1_s{size}_w{int(w*100)}", "size": size, "high_weight": w}
                              for size in SIZES for w in (.25, .5)]


def rank(values, n_query, spec):
    dimension = 2048 if spec == BASELINE else 3584
    if (spec not in systems() or values.ndim != 2 or values.shape[1] != dimension
            or type(n_query) is not int or not 0 < n_query < len(values)-1):
        raise ValueError("Use a frozen v31 system and its original2048 [+ high-resolution R1_1536] layout")
    mvp, r1 = dual.unpack(values[:, :2048])
    if spec == BASELINE:
        return v25.rank(values[:n_query], values[n_query:])
    high = values[:, 2048:]
    dual.validate_block(high, 1536)
    w = spec["high_weight"]
    # Keep MVP/R1 at 50/50. Only split the R1 contribution between two scales.
    mixed = normalize(np.concatenate([normalize(mvp)*np.float32(np.sqrt(.5)),
                                      normalize(r1)*np.float32(np.sqrt(.5*(1-w))),
                                      normalize(high)*np.float32(np.sqrt(.5*w))], axis=1))
    q, g = normalize(mixed[:n_query]), normalize(mixed[n_query:])
    graph = KReciprocalReranker(g, min(20, len(g)-1), min(3, len(g)))
    distances = np.stack([graph.distances(vector, .5) for vector in q])
    raw = dual.policy.rank_vectors(r1[:n_query], r1[n_query:], "raw")
    return {"order": np.argsort(distances, axis=1, kind="stable"),
            "raw_order": raw["raw_order"], "confidence": raw["confidence"]}


def model_sources(profile_path):
    _, paths, r1 = dual.load_profile(profile_path)
    sources = []
    for member in r1["members"]:
        bp = (paths["r1_bundle"].parent/member["path"]).resolve()
        if sha256(bp) != member["sha256"]:
            raise ValueError("R1 member bundle changed")
        bundle = dual.read(bp)
        frozen._validate_policy(bundle)
        mp = (bp.parent/bundle["model"]["path"]).resolve()
        if (bundle["preprocessing"] != frozen._preprocessing(256, "square")
                or bundle["model"]["dimension"] != 512 or sha256(mp) != bundle["model"]["sha256"]):
            raise ValueError("Expected unchanged R1/256/square/512 model")
        sources.append({"bundle": str(bp), "bundle_sha256": sha256(bp),
                        "model": str(mp), "model_sha256": sha256(mp)})
    if len(sources) != 3 or len({s["model_sha256"] for s in sources}) != 3:
        raise ValueError("Expected the three distinct original R1 encoders")
    return sources


def resized_model_bytes(source_bytes, size):
    """Only change the declared H/W; every operator, weight and constant is retained."""
    if size not in (256, *SIZES):
        raise ValueError("Unsupported image size")
    model = onnx.load_model_from_string(source_bytes)
    if (frozen._external_tensors(model) or len(model.graph.input) != 1 or len(model.graph.output) != 1
            or model.graph.value_info or not any(n.op_type == "GlobalAveragePool" for n in model.graph.node)):
        raise ValueError("Expected the original embedded fully convolutional R1 export")
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4 or [d.dim_value for d in dims[1:]] != [3, 256, 256]:
        raise ValueError("Expected original NCHW/256 input")
    original = model.SerializeToString()
    dims[2].dim_value = dims[3].dim_value = size
    onnx.checker.check_model(model, full_check=True)
    adapted = model.SerializeToString()
    dims[2].dim_value = dims[3].dim_value = 256
    if model.SerializeToString() != original:
        raise ValueError("Adaptation altered more than input spatial dimensions")
    return source_bytes if size == 256 else adapted


class R1ScaleEncoder:
    """CPU-only independent image TTA. Original ONNX files are never overwritten."""
    def __init__(self, profile_path, size, provider="CPUExecutionProvider"):
        if provider != "CPUExecutionProvider" or provider not in frozen.ort.get_available_providers():
            raise RuntimeError("This experiment requires explicit CPU; no provider fallback")
        self.sources, self.size = model_sources(profile_path), size
        self.members, self.adapted_sha256 = [], []
        for source in self.sources:
            payload = resized_model_bytes(Path(source["model"]).read_bytes(), size)
            options = frozen.ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
            options.log_severity_level = 3
            session = frozen.ort.InferenceSession(payload, sess_options=options, providers=[provider], enable_fallback=False)
            session.disable_fallback()
            if session.get_providers() != [provider]:
                raise RuntimeError("Provider fallback is forbidden")
            spec = frozen._model_spec(session, size)
            if spec["dimension"] != 512:
                raise ValueError("Larger input must retain 512-D member output")
            self.members.append((session, spec))
            self.adapted_sha256.append(hashlib.sha256(payload).hexdigest())

    def preprocess(self, image, box):
        crop = resize_crop(crop_image(image, box), "square", self.size)
        pixels = (np.asarray(crop, dtype=np.float32)/np.float32(255)-IMAGENET_MEAN)/IMAGENET_STD
        return np.ascontiguousarray(pixels.transpose(2, 0, 1))

    def encode_rows(self, rows, dataset, batch_size=16):
        if type(batch_size) is not int or batch_size < 1 or not rows:
            raise ValueError("Need nonempty rows and a positive batch size")
        blocks = []
        for start in range(0, len(rows), batch_size):
            inputs = []
            for row in rows[start:start+batch_size]:
                with Image.open(dual.image_path(dataset, row["image_id"])) as image:
                    inputs.append(self.preprocess(image, bbox(row)))
            inputs = np.ascontiguousarray(np.stack(inputs), dtype=np.float32)
            members = [normalize(s.run([spec["output"]], {spec["input"]: inputs})[0]) for s, spec in self.members]
            for member in members:
                dual.validate_block(member, 512)
                if len(member) != len(inputs):
                    raise ValueError("Wrong encoder output row count")
            blocks.append(dual.policy.combine_members(members))
        return np.concatenate(blocks)
