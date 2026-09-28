"""Portable model adapters, explicit BN domains, official pinned initialization."""
import copy
import os
from pathlib import Path
import urllib.request
import zipfile

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["XFORMERS_DISABLED"] = "1"
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from training import research_io as io
from backend.core import bbox, crop_image, normalize
from training.hpo import ReIDExperimentModel
from training.pipeline import image_transforms
from training.preprocessing import ResizeCrop

DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
FASTREID_REVISION = "c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d"
ASSETS = {
    **{f"dinov2_{size}": {"url": f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{size}/dinov2_{size}_pretrain.pth",
                            "license": "Apache-2.0", "source_revision": DINO_REVISION}
       for size in ("vits14", "vitb14")},
    **{f"r50_{name}": {"url": f"https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/{file}.pth",
                        "license": "See FastReID Apache-2.0 and source dataset research terms",
                        "source_revision": FASTREID_REVISION}
       for name, file in (("veri", "veri_sbs_R50-ibn"), ("vehicleid", "vehicleid_bot_R50-ibn"),
                          ("veriwild", "veriwild_bot_R50-ibn"))},
    "r50_imagenet": {"url": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth",
                     "license": "MIT", "source_revision": "v1.0",
                     "sha256": "d9d0bb7b2ba34d7e6e12c4616c9d233914762e49fbb9fcb49d7f4d65da6f759c"},
}


def download(cache, name, source):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path, record = cache / name, cache / (name + ".json")
    if record.exists():
        saved = io.read(record)
        if saved["source"] != source or not path.is_file() or io.sha(path) != saved["sha256"]:
            raise ValueError(f"Changed pinned asset: {name}")
        return path
    # No trust of manually dropped, unrecorded files; download to a separate pending file.
    if path.exists():
        raise ValueError(f"Unrecorded asset: {path}; do not silently accept foreign weights")
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"DOWNLOAD official source: {source['url']}", flush=True)
    with urllib.request.urlopen(source["url"], timeout=40) as response, temporary.open("wb") as output:
        while chunk := response.read(1024*1024):
            output.write(chunk)
    checksum = io.sha(temporary)
    if source.get("sha256") and checksum != source["sha256"]:
        raise ValueError("Pinned weight hash mismatch")
    temporary.replace(path)
    io.write(record, {"source": source, "sha256": checksum, "bytes": path.stat().st_size,
                      "verification": "official HTTPS bytes pinned locally; strict state_dict required"})
    return path


class DomainBN(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.main, self.aux = copy.deepcopy(original), copy.deepcopy(original)
        self.domain = "main"

    def forward(self, values):
        return (self.aux if self.domain == "aux" else self.main)(values)


def split_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            setattr(module, name, DomainBN(child))
        else:
            split_bn(child)


def set_domain(model, domain, mode, training=True):
    model.train(training)
    for layer in model.modules():
        if isinstance(layer, DomainBN):
            layer.domain = domain
        elif mode == "target_updates_only" and domain == "aux" and isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def load_osnet(directory, manifest, name="R1_20260915"):
    item = manifest["models"][name]
    path = io.child(directory, item["path"])
    if io.sha(path) != item["sha256"]:
        raise ValueError("Changed OSNet checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != item["signature"] or payload["step"] != item["step"]:
        raise ValueError("Checkpoint identity/split signature mismatch")
    model = ReIDExperimentModel(len(manifest["train_ids"]), use_bnneck=True)
    state = payload["model"]
    if name == "N1":
        state = {key.replace("main_head.", "classifier."): value for key, value in state.items()
                 if not key.startswith("aux_head.")}
    model.load_state_dict(state, strict=True)
    return model


class Images:
    def __init__(self, root, rows, size=256, train=False, paired=False, interpolation="bilinear"):
        self.root, self.rows, self.paired = Path(root), rows, paired
        self.clean = transforms.Compose([transforms.Resize((size, size), interpolation=(
            transforms.InterpolationMode.BICUBIC if interpolation == "bicubic" else transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
        self.robust = self.clean
        if train and paired:
            self.clean, self.robust = image_transforms(), image_transforms(True)
            self.clean.transforms[0] = self.robust.transforms[0] = ResizeCrop("square", size)
        elif train:
            self.robust = transforms.Compose([self.clean.transforms[0], transforms.RandomHorizontalFlip(),
                transforms.Pad(10), transforms.RandomCrop(size), *self.clean.transforms[1:],
                transforms.RandomErasing(.25, value=0)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(io.child(self.root, row["path"])) as image:
            crop = crop_image(image, bbox(row)) if "x" in row else image.convert("RGB")
            robust = self.robust(crop) if self.paired else None
            value = self.clean(crop) if self.paired else self.robust(crop)
        return (value, robust, row.get("label", 0)) if self.paired else (value, row.get("label", 0))


@torch.inference_mode()
def encode(model, root, rows, device, size=256, batch_size=16, interpolation="bilinear"):
    model.eval()
    for layer in model.modules():
        if isinstance(layer, DomainBN): layer.domain = "main"
    data, parts = Images(root, rows, size, interpolation=interpolation), []
    for start in range(0, len(rows), batch_size):
        x = torch.stack([data[i][0] for i in range(start, min(start+batch_size, len(rows)))]).to(device)
        values = model.embedding(x) if hasattr(model, "embedding") else model(x)
        parts.append(normalize(values.float().cpu().numpy()))
        if start % 256 == 0 or start+batch_size >= len(rows):
            print(f"FEATURES {min(start+batch_size,len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def recalibrate_bn(model, root, train_rows, device, size=256, batch_size=32):
    """One deterministic TRAIN-only pass; dropout off, no optimizer or affine updates."""
    model.eval()
    layers = [v for v in model.modules() if isinstance(v, nn.modules.batchnorm._BatchNorm)]
    momenta = [v.momentum for v in layers]
    for layer in layers:
        layer.reset_running_stats(); layer.momentum = None; layer.train()
    data = Images(root, train_rows, size)
    try:
        for start in range(0, len(data), batch_size):
            indices = list(range(start, min(start+batch_size, len(data))))
            # A singleton final batch cannot update BNNeck; combine with the preceding image.
            if len(indices) == 1: indices.insert(0, start-1)
            x = torch.stack([data[i][0] for i in indices]).to(device)
            model.embedding(x)
            if start % 512 == 0: print(f"BN train-only {min(start+batch_size,len(data))}/{len(data)}", flush=True)
    finally:
        for layer, momentum in zip(layers, momenta): layer.momentum = momentum
        model.eval()


def checkpoint_blocks(blocks):
    """Actual full P×K batch, activation recomputation instead of microbatch SupCon."""
    from torch.utils.checkpoint import checkpoint
    for block in blocks:
        original = block.forward
        def forward(*args, _original=original, _block=block, **kwargs):
            if _block.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)
        block.forward = forward


class NonLocal(nn.Module):
    """Adapted FastReID non-local block; Copyright 2019 JD.com Inc. JD AI.

    Apache-2.0, see research_licenses/FASTREID_LICENSE. Upstream revision above,
    fastreid/layers/non_local.py. Changed matmul association to reduce memory.
    Upstream intentionally uses ONE intermediate channel.
    """
    def __init__(self, channels):
        super().__init__()
        self.g, self.theta, self.phi = (nn.Conv2d(channels, 1, 1) for _ in range(3))
        self.W = nn.Sequential(nn.Conv2d(1, channels, 1), nn.BatchNorm2d(channels))

    def forward(self, x):
        g = self.g(x).flatten(2).transpose(1, 2)
        theta, phi = self.theta(x).flatten(2).transpose(1, 2), self.phi(x).flatten(2)
        # Associative product avoids allocating spatial×spatial attention; mathematically identical.
        y = theta @ (phi @ g) / phi.shape[-1]
        return x + self.W(y.transpose(1, 2).reshape(len(x), 1, *x.shape[2:]))


class VehicleIBN(nn.Module):
    def __init__(self, state, sbs=False, imagenet=False):
        super().__init__()
        from training.resnet_ibn import ResNet50IBNBackbone
        learned_pool = not imagenet and "heads.pool_layer.p" in state
        self.backbone = ResNet50IBNBackbone(pooling="gem" if learned_pool else "avg")
        if not imagenet:
            self.backbone.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.sbs = sbs
        if sbs:
            self.backbone.NL_1, self.backbone.NL_4 = nn.ModuleList(), nn.ModuleList()
            self.backbone.NL_2 = nn.ModuleList([NonLocal(512) for _ in range(2)])
            self.backbone.NL_3 = nn.ModuleList([NonLocal(1024) for _ in range(3)])
        self.neck = nn.BatchNorm1d(2048) if not imagenet else nn.Identity()
        if imagenet:
            if set(k for k in state if k.startswith("fc.")) != {"fc.weight", "fc.bias"}:
                raise ValueError("Wrong ImageNet classifier")
            self.backbone.load_state_dict({k:v for k,v in state.items() if not k.startswith("fc.")}, strict=True)
        else:
            state = canonical_fastreid_state(state)
            allowed = {"heads.weight", "heads.bottleneck.0.weight", "heads.bottleneck.0.bias",
                       "heads.bottleneck.0.running_mean", "heads.bottleneck.0.running_var",
                       "heads.bottleneck.0.num_batches_tracked"}
            if sbs and not learned_pool: raise ValueError("SBS checkpoint is missing learned GeM")
            if learned_pool: allowed.add("heads.pool_layer.p")
            extra = {k for k in state if not k.startswith("backbone.")}
            if extra != allowed:
                raise ValueError(f"Unexpected FastReID head fields: {extra ^ allowed}")
            backbone = {k.removeprefix("backbone."):v for k,v in state.items() if k.startswith("backbone.")}
            if learned_pool: backbone["global_pool.p"] = state["heads.pool_layer.p"].reshape(())
            self.backbone.load_state_dict(backbone, strict=True)
            self.neck.load_state_dict({k.removeprefix("heads.bottleneck.0."):v for k,v in state.items()
                                      if k.startswith("heads.bottleneck.0.")}, strict=True)
        self.dimension = 2048

    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        for stage, count in ((1,0), (2,2), (3,3), (4,0)):
            blocks = getattr(b, f"layer{stage}")
            for i, block in enumerate(blocks):
                x = block(x)
                if self.sbs and count and i >= len(blocks)-count:
                    x = getattr(b, f"NL_{stage}")[i-(len(blocks)-count)](x)
        return self.neck(b.global_pool(x).flatten(1))


def canonical_fastreid_state(original):
    """Explicit v0.1.1 release migration, not strict=False or a missing-key fallback."""
    state = dict(original)
    if ("pixel_mean" in state) != ("pixel_std" in state):
        raise ValueError("Incomplete FastReID normalization buffers")
    # The release checkpoints include preprocessing buffers; validate before removing.
    for key, expected in (("pixel_mean",[123.675,116.28,103.53]),("pixel_std",[58.395,57.12,57.375])):
        if key in state:
            value = state.pop(key).float().flatten()
            reference = torch.tensor(expected,dtype=torch.float32)
            if value.shape != (3,) or not torch.allclose(value,reference,atol=1e-5,rtol=0):
                raise ValueError(f"FastReID {key} does not match our RGB normalization")
    aliases = {"heads.classifier.weight":"heads.weight",
               "heads.bnneck.num_batches_tracked":"heads.bottleneck.0.num_batches_tracked"}
    for old,new in aliases.items():
        if old in state:
            if new in state: raise ValueError(f"Ambiguous FastReID field: {old} and {new}")
            state[new] = state.pop(old)
    head = state.get("heads.weight")
    if head is None or head.ndim != 2 or head.shape[1] != 2048:
        raise ValueError("Invalid FastReID classifier shape")
    return state


class ExternalModel(nn.Module):
    def __init__(self, base, dimension, classes, kind, pooling="cls", adapt=False):
        super().__init__()
        self.base, self.kind, self.pooling = base, kind, pooling
        width = dimension * (2 if pooling == "concat" else 1)
        self.projection = nn.Linear(width, 512, bias=False) if adapt and kind == "dino" else nn.Identity()
        self.dimension = 512 if adapt and kind == "dino" else width
        self.neck = nn.BatchNorm1d(self.dimension) if adapt and kind == "dino" else nn.Identity()
        self.classifier = nn.Linear(self.dimension, classes, bias=False)
        self.adapt = adapt

    def raw(self, images):
        if self.kind == "dino":
            output = self.base.forward_features(images)
            cls, patch = output["x_norm_clstoken"], output["x_norm_patchtokens"].mean(1)
            value = cls if self.pooling == "cls" else patch if self.pooling == "patch" else torch.cat(
                [F.normalize(cls, dim=1), F.normalize(patch, dim=1)], dim=1) / 2**.5
        else:
            value = self.base(images)
        return self.projection(value)

    def forward(self, images):
        raw = self.raw(images)
        if self.training:
            return [self.classifier(self.neck(raw))], [raw]
        return F.normalize(self.neck(raw) if self.adapt else raw, dim=1)


def external_model(name, pooling, classes, cache, adapt=False):
    path = download(cache, name + ".pth", ASSETS[name])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if name.startswith("dinov2"):
        source = {"url": f"https://github.com/facebookresearch/dinov2/archive/{DINO_REVISION}.zip",
                  "source_revision": DINO_REVISION, "license": "Apache-2.0"}
        archive = download(cache, "dinov2_source.zip", source)
        folder = Path(cache) / f"dinov2-{DINO_REVISION}"
        if not folder.exists():
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist(): io.child(cache, member.rstrip("/"))
                z.extractall(cache)
        # Verify extracted executable source against the pinned archive before importing.
        with zipfile.ZipFile(archive) as z:
            import hashlib
            for member in z.infolist():
                if not member.is_dir() and io.sha(io.child(cache, member.filename)) != hashlib.sha256(z.read(member)).hexdigest():
                    raise ValueError("DINO source cache changed")
        base = torch.hub.load(str(folder), name, source="local", pretrained=False)
        base.load_state_dict(state, strict=True)
        return ExternalModel(base, base.embed_dim, classes, "dino", pooling, adapt)
    base = VehicleIBN(state.get("model", state), sbs=name == "r50_veri", imagenet=name == "r50_imagenet")
    return ExternalModel(base, 2048, classes, "ibn", "cls", adapt)
