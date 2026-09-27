"""Pinned DINOv3 D0: official local weights, frozen image-only CLS/patch features."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import shutil
import sys

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from backend.core import ROOT, bbox, crop_image, sha256
from training.dual_role_inference import image_path, validate_block

VARIANT = ROOT / "OSNet-AIN-x1.0/variant_35_dino_frozen"
DEPS = VARIANT / ".deps"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
WEIGHT_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
WEIGHT_SIZE = 86406384
# Git blob identities from the official Hub API at REVISION. Weight uses LFS SHA-256.
BLOBS = {"config.json": "fdff48569812f3cbadf253762565c4ce9dacddac",
         "preprocessor_config.json": "0126173c1ab7b75bbd9102e37de8a56ffe0a012b",
         "LICENSE.md": "f531b1e6b5ab2318957bbf8ad1bda9f800a23e17",
         "README.md": "e1b5c4aff2cb20b55b90fc6c496d462358d0f620"}
PREPROCESS = {"image_size": 256, "geometry": "original organizer bbox, RGB, EXIF transpose",
              "resize": "torchvision v2 ToImage(uint8) -> Resize((256,256), bilinear, antialias=True)",
              "rescale": "ToDtype(float32, scale=True)", "mean": [.485, .456, .406], "std": [.229, .224, .225],
              "pool": "L2(CLS) + L2(mean of 256 patch tokens), concat / sqrt(2); exclude 4 registers"}


def activate_dependencies(*, install=False):
    """Add isolated optional packages without upgrading the historical environment."""
    requirements = VARIANT / "requirements-d0.txt"
    pins = dict(line.split("==") for line in requirements.read_text().splitlines() if line and not line.startswith("#"))
    installed = {d.metadata["Name"].lower().replace("-", "_"): d.version
                 for d in importlib.metadata.distributions(path=[str(DEPS)])}
    if any(installed.get(k.replace("-", "_")) != version for k, version in pins.items()):
        if not install:
            raise RuntimeError("Run activate_dependencies(install=True) once; optional packages live only in v35/.deps")
        env = {**os.environ, "PIP_CACHE_DIR": str(VARIANT / "cache/pip"), "UV_CACHE_DIR": str(VARIANT / "cache/uv")}
        if importlib.util.find_spec("pip"):
            command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "--upgrade"]
        elif shutil.which("uv"):
            command = [shutil.which("uv"), "pip", "install", "--python", sys.executable, "--no-deps", "--no-python-downloads"]
        else:
            raise RuntimeError("Need pip or uv to install the isolated optional dependencies")
        subprocess.run([*command, "--target", str(DEPS), "-r", str(requirements)], check=True, env=env)
    if str(DEPS) not in sys.path:
        sys.path.insert(0, str(DEPS))
    for name, version in pins.items():
        loaded = sys.modules.get(name)
        if importlib.metadata.version(name) != version or (loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(DEPS)):
            raise RuntimeError(f"Wrong {name} version; restart kernel and use the pinned v35 dependencies")
    return pins


def model_files(directory):
    """No network, no arbitrary model substitution, no unverified converted checkpoint."""
    directory = Path(directory).resolve()
    required = ["model.safetensors", *BLOBS]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"DINOv3 official weights are not ready ({', '.join(missing)}). "
            f"Request access at https://huggingface.co/{MODEL_ID}, accept its license yourself, "
            "then use hf auth login locally and download the pinned revision. Do not send tokens in chat.")
    weights = directory / "model.safetensors"
    if weights.stat().st_size != WEIGHT_SIZE or sha256(weights) != WEIGHT_SHA256:
        raise ValueError("DINOv3 weight SHA-256/size differs from the official pinned revision")
    for name, expected in BLOBS.items():
        payload = (directory / name).read_bytes()
        actual = hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
        if actual != expected:
            raise ValueError(f"DINOv3 {name} differs from the official revision")
    return {str(directory / name): sha256(directory / name) for name in required}


def download_model(directory):
    """Explicit preparation only. The user must already have accepted the gated license."""
    activate_dependencies()
    directory = Path(directory).resolve()
    try:
        return model_files(directory)
    except FileNotFoundError:
        pass
    from huggingface_hub import get_token
    if not get_token():
        raise RuntimeError(f"DINOv3 is gated. First request access at https://huggingface.co/{MODEL_ID} "
                           "and run hf auth login locally. No token is stored in the notebook or reports.")
    # Do not overwrite changed existing files; partial downloads are a separate Hub cache.
    for name in ["model.safetensors", *BLOBS]:
        p = directory / name
        if p.exists():
            if name == "model.safetensors":
                valid = p.stat().st_size == WEIGHT_SIZE and sha256(p) == WEIGHT_SHA256
            else:
                content = p.read_bytes()
                valid = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() == BLOBS[name]
            if not valid:
                raise ValueError(f"Existing model file changed: {name}; do not overwrite it silently")
    env = {**os.environ, "PYTHONPATH": str(DEPS), "HF_HUB_CACHE": str(VARIANT / "cache/hub")}
    result = subprocess.run([sys.executable, str(DEPS / "bin/hf"), "download", MODEL_ID,
                             "model.safetensors", *BLOBS, "--revision", REVISION, "--local-dir", str(directory)], env=env)
    if result.returncode:
        raise RuntimeError("Official download failed. Check approved access, read token and network; no mirror/fallback is used.")
    return model_files(directory)


def transform():
    # Official LVD transform: resize the tensor BEFORE float conversion, not a PIL approximation.
    return v2.Compose([v2.ToImage(), v2.Resize((256, 256), antialias=True),
                       v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean=PREPROCESS["mean"], std=PREPROCESS["std"])])


def pool_tokens(tokens):
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (261, 384) or tokens.dtype != torch.float32 or not torch.isfinite(tokens).all():
        raise ValueError("Expected float32 [B, 1 CLS + 4 registers + 256 patches, 384] tokens")
    cls, patches = tokens[:, 0], tokens[:, 5:].mean(dim=1)
    if (cls.norm(dim=1) <= 1e-12).any() or (patches.norm(dim=1) <= 1e-12).any():
        raise ValueError("Invalid zero CLS/patch mean; no invented replacement vector")
    return torch.cat([F.normalize(cls, dim=1), F.normalize(patches, dim=1)], dim=1) / np.sqrt(2.)


def device_for(name):
    if name not in {"cpu", "mps", "cuda"} or (name == "mps" and not torch.backends.mps.is_available()) or (
            name == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested device unavailable: {name}; no automatic fallback")
    if name == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback for a reproducible run")
    return torch.device(name)


class FrozenDino:
    def __init__(self, directory, device="cpu"):
        self.files = model_files(directory)
        activate_dependencies()
        from transformers import DINOv3ViTModel
        self.device = device_for(device)
        self.model, info = DINOv3ViTModel.from_pretrained(str(Path(directory).resolve()), local_files_only=True,
            use_safetensors=True, dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"DINOv3 checkpoint loading is not exact: {info}")
        config = self.model.config
        if (config.model_type, config.hidden_size, config.patch_size, config.num_register_tokens,
                config.num_hidden_layers, config.num_attention_heads) != ("dinov3_vit", 384, 16, 4, 12, 6):
            raise ValueError("Expected the fixed ViT-S/16 LVD backbone")
        self.model.to(self.device).eval().requires_grad_(False)
        self.preprocess = transform()

    @torch.inference_mode()
    def encode_rows(self, rows, dataset, batch_size=16):
        if type(batch_size) is not int or batch_size < 1 or not rows or self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("D0 requires nonempty images, a positive batch, eval mode and frozen weights")
        result = []
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start+batch_size]:
                with Image.open(image_path(dataset, row["image_id"])) as im:
                    images.append(self.preprocess(crop_image(im, bbox(row))))
            tokens = self.model(pixel_values=torch.stack(images).to(self.device)).last_hidden_state
            values = pool_tokens(tokens).cpu().numpy().astype(np.float32)
            validate_block(values, 768)
            result.append(values)
        return np.concatenate(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=VARIANT / "weights" / REVISION)
    args = parser.parse_args()
    activate_dependencies(install=args.install_deps)
    if args.download:
        download_model(args.model_dir)
    print("Pinned optional dependencies ready. Model:", args.model_dir)
