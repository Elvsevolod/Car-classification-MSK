"""Inference-only port of the frozen v16/v18 encoders (see ASSET_PROVENANCE)."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from .core import crop_image, normalize, sha256

IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.array([.229, .224, .225], dtype=np.float32)
LETTERBOX_FILL = tuple(int(round(value * 255)) for value in IMAGENET_MEAN)
PROVIDERS = ("CPUExecutionProvider", "CUDAExecutionProvider")
RERANKING = {"method": "streaming k-reciprocal", "k1": 20, "k2": 3, "lambda": .5}
CONFIDENCE = "maximum raw gallery cosine; not a probability"
POLICIES = {"legacy": {"k1": 20, "k2": 3, "lambda": .5},
            "less_graph": {"k1": 20, "k2": 3, "lambda": .75}}
CANDIDATES = ("raw_top1", "ranking_top1")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def combine_members(members):
    values = [normalize(np.asarray(x, dtype=np.float32)) for x in members]
    if len(values) not in (1, 3) or any(x.ndim != 2 or x.shape != values[0].shape for x in values):
        raise ValueError("Expected one or three matching member matrices")
    return normalize(np.concatenate(values, axis=1) / np.sqrt(len(values)))


def resize_crop(crop, mode, size=208):
    """Resize a PIL crop either by distortion or aspect-ratio preserving padding."""
    if mode == "square":
        return crop.resize((size, size), Image.Resampling.BILINEAR)
    if mode != "letterbox":
        raise ValueError("resize mode must be 'square' or 'letterbox'")
    scale = min(size / crop.width, size / crop.height)
    width = max(1, round(crop.width * scale))
    height = max(1, round(crop.height * scale))
    resized = crop.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), LETTERBOX_FILL)
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


def _external_tensors(message):
    if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
        return True
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            if any(_external_tensors(item) for item in (value if field.is_repeated else [value])):
                return True
    return False


def _session(model_path, provider):
    if provider not in PROVIDERS or provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested provider is unavailable: {provider}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if provider == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    model_bytes = Path(model_path).read_bytes()
    if _external_tensors(onnx.load_model_from_string(model_bytes)):
        raise ValueError("Frozen bundles require embedded ONNX weights, not external tensor files")
    session = ort.InferenceSession(model_bytes, sess_options=options,
                                   providers=[provider], enable_fallback=False)
    session.disable_fallback()
    if session.get_providers() != [provider]:
        raise RuntimeError(f"Provider fallback is forbidden: {session.get_providers()}")
    return session


def _model_spec(session, size):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one image input and one embedding output")
    source, target = inputs[0], outputs[0]
    if source.type != "tensor(float)" or target.type != "tensor(float)":
        raise ValueError("Expected float32 ONNX input and output")
    if len(source.shape) != 4 or len(target.shape) != 2:
        raise ValueError("Expected NCHW images and 2D embeddings")
    if any(isinstance(actual, int) and actual != expected
           for actual, expected in zip(source.shape[1:], (3, size, size))):
        raise ValueError("ONNX input shape does not match preprocessing")
    dimension = None
    for count in (1, 3):
        probe = np.random.default_rng(0).normal(size=(count, 3, size, size)).astype(np.float32)
        result = session.run([target.name], {source.name: probe})[0]
        if result.ndim != 2 or result.shape[0] != count or result.shape[1] < 1:
            raise ValueError(f"Unexpected model output shape: {result.shape}")
        normalize(result)
        if dimension is not None and result.shape[1] != dimension:
            raise ValueError("Embedding dimension changes with batch size")
        dimension = result.shape[1]
    return {"input": source.name, "output": target.name, "dimension": dimension}


def _preprocessing(image_size, resize_mode):
    if type(image_size) is not int or image_size <= 0 or resize_mode not in {"square", "letterbox"}:
        raise ValueError("Invalid frozen preprocessing")
    return {"image_size": image_size, "resize_mode": resize_mode, "interpolation": "bilinear",
            "decode": "EXIF-oriented RGB", "crop": "organizer xywh before resize",
            "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
            "scale": 255, "layout": "NCHW", "l2_normalized": True, "flip_tta": False}


def _validate_policy(bundle):
    prep = bundle["preprocessing"]
    if bundle["schema"] != 1 or prep != _preprocessing(prep["image_size"], prep["resize_mode"]):
        raise ValueError("Unsupported frozen preprocessing/schema")
    if bundle["reranking"] != RERANKING or bundle["confidence"] != CONFIDENCE:
        raise ValueError("Unsupported ranking or confidence policy")
    if not np.isfinite(bundle["threshold"]):
        raise ValueError("Frozen threshold must be finite")
    provenance = bundle["calibration"]
    if (provenance.get("split") != "calibration" or not provenance.get("method")
            or not re.fullmatch(r"[0-9a-f]{64}", provenance.get("protocol_sha256", ""))):
        raise ValueError("Threshold needs calibration-only method and protocol SHA256")


class FrozenEncoder:
    def __init__(self, bundle_path, provider="CPUExecutionProvider"):
        self.bundle_path = Path(bundle_path).resolve()
        self.bundle = json.loads(self.bundle_path.read_text())
        _validate_policy(self.bundle)
        model = self.bundle["model"]
        model_path = (self.bundle_path.parent / model["path"]).resolve()
        self.model_sha256 = sha256(model_path)
        if self.model_sha256 != model["sha256"]:
            raise ValueError("Frozen ONNX checksum mismatch")
        self.provider = provider
        self.session = _session(model_path, provider)
        self.size = self.bundle["preprocessing"]["image_size"]
        spec = _model_spec(self.session, self.size)
        if any(spec[key] != model[key] for key in spec):
            raise ValueError("Frozen ONNX input/output/dimension mismatch")
        self.dimension, self.input_name, self.output_name = spec["dimension"], spec["input"], spec["output"]
        self.fingerprint = hashlib.sha256(json.dumps(self.bundle, sort_keys=True).encode()).hexdigest()

    def preprocess(self, image, box):
        crop = resize_crop(crop_image(image, box), self.bundle["preprocessing"]["resize_mode"], self.size)
        pixels = (np.asarray(crop, dtype=np.float32) / np.float32(255) - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(pixels.transpose(2, 0, 1))

    def encode_batch(self, batch):
        inputs = np.asarray(batch, dtype=np.float32)
        if inputs.ndim != 4 or inputs.shape[1:] != (3, self.size, self.size) or not len(inputs):
            raise ValueError("Invalid image batch shape")
        if not np.isfinite(inputs).all():
            raise ValueError("Non-finite image batch")
        output = self.session.run([self.output_name], {self.input_name: inputs})[0]
        if output.shape != (len(inputs), self.dimension):
            raise ValueError(f"Unexpected embedding shape: {output.shape}")
        return normalize(output)

    def encode(self, image, box):
        return self.encode_batch([self.preprocess(image, box)])[0]


def validate_bundle(bundle):
    if (bundle.get("schema") != 2 or bundle.get("ranking") not in POLICIES
            or bundle.get("ranking_parameters") != POLICIES[bundle["ranking"]]
            or bundle.get("candidate_policy") not in CANDIDATES
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


class PolicyEncoder:
    def __init__(self, bundle_path, provider="CPUExecutionProvider"):
        self.bundle_path = Path(bundle_path).resolve()
        self.bundle = json.loads(self.bundle_path.read_text())
        validate_bundle(self.bundle)
        paths = [(self.bundle_path.parent / m["path"]).resolve() for m in self.bundle["members"]]
        if any(sha256(p) != m["sha256"] for p, m in zip(paths, self.bundle["members"])):
            raise ValueError("Source encoder bundle checksum mismatch")
        self.members = [FrozenEncoder(p, provider) for p in paths]
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
        return combine_members([e.encode_batch(batch) for e in self.members])

    def encode(self, image, box):
        return self.encode_batch([self.preprocess(image, box)])[0]
