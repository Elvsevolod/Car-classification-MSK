"""TransReID/DeiT-Small image-only implementation with strict official initialization."""
import copy
import os
from pathlib import Path
import urllib.request

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from backend.core import ROOT, bbox, crop_image, sha256
from training.dual_role_inference import validate_block
from training.transreid_vendor.vit_pytorch import deit_small_patch16_224_TransReID, resize_pos_embed

VARIANT = ROOT / "OSNet-AIN-x1.0/variant_36_transreid_night"
REVISION = "dec55046fcdfadee14e2c28e2df89305d8f7557a"
WEIGHT_NAME = "deit_small_distilled_patch16_224-649709d9.pth"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/deit/" + WEIGHT_NAME
WEIGHT_SHA256 = "649709d94f9fd790ea86c16f99d788e709b86a1f64315a19d887895f9948fb09"
WEIGHT_SIZE = 89795170
WEIGHT_PATH = VARIANT / "weights" / WEIGHT_NAME
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
PREPROCESS = {"size": 256, "bbox": "original integer xywh, EXIF-aware RGB",
              "resize": "PIL bilinear square, no center crop", "mean": MEAN, "std": STD,
              "train": "flip=.5, zero padding=10 then random crop 256, random erasing=.25 after normalize",
              "retrieval": "before BN; global384 or concat(global384, four local384 / 4), L2"}


def verify_weights(path=WEIGHT_PATH):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != WEIGHT_SIZE or sha256(path) != WEIGHT_SHA256:
        raise ValueError("Missing/changed official DeiT weights; download the pinned file, no fallback")
    return path


def download_weights():
    if WEIGHT_PATH.exists():
        return verify_weights()
    WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = WEIGHT_PATH.with_suffix(".pth.part")
    # Reuse a completed verified download; never replace a changed final checkpoint.
    if temporary.exists() and temporary.stat().st_size == WEIGHT_SIZE and sha256(temporary) == WEIGHT_SHA256:
        temporary.replace(WEIGHT_PATH)
        return verify_weights()
    print("Downloading official DeiT-Small (86 MiB), no account/token required", flush=True)
    with urllib.request.urlopen(WEIGHT_URL, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024*1024):
            output.write(chunk)
    verify_weights(temporary)
    temporary.replace(WEIGHT_PATH)
    return verify_weights()


def device_for(name):
    if name not in {"cpu", "mps", "cuda"} or (name == "mps" and not torch.backends.mps.is_available()) or (
            name == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested device unavailable: {name}; no silent CPU fallback")
    if name == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback before starting this notebook")
    return torch.device(name)


def image_paths(dataset, rows):
    index = {}
    for path in (Path(dataset) / "images").iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index.setdefault(path.stem, []).append(path)
    result = {}
    for row in rows:
        matches = index.get(row["image_id"], [])
        if len(matches) != 1:
            raise ValueError(f"Expected one JPEG/PNG for {row['image_id']}; found {len(matches)}")
        result[row["image_id"]] = matches[0]
    return result


def transform(train=False):
    operations = [transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)]
    if train:
        operations += [transforms.RandomHorizontalFlip(.5), transforms.Pad(10), transforms.RandomCrop(256)]
    operations += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if train:
        operations += [transforms.RandomErasing(p=.25, value=0)]
    return transforms.Compose(operations)


class Images:
    def __init__(self, rows, paths, train=False):
        self.rows, self.paths, self.transform = rows, paths, transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.paths[row["image_id"]]) as image:
            value = self.transform(crop_image(image, bbox(row)))
        return value, row.get("label", 0)


def load_pretrained(backbone, path=WEIGHT_PATH):
    state = torch.load(verify_weights(path), map_location="cpu", weights_only=True)["model"]
    excluded = {"dist_token", "head.weight", "head.bias", "head_dist.weight", "head_dist.bias"}
    if not excluded <= state.keys() or tuple(state["pos_embed"].shape) != (1, 198, 384):
        raise ValueError("Expected official distilled DeiT-Small with two special position tokens")
    state = {k: v for k, v in state.items() if k not in excluded}
    position = torch.cat([state["pos_embed"][:, :1], state["pos_embed"][:, 2:]], dim=1)
    state["pos_embed"] = resize_pos_embed(position, backbone.pos_embed, 16, 16)
    # No broad strict=False or printed-and-ignored mismatches from the old loader.
    backbone.load_state_dict(state, strict=True)
    return {"loaded_tensors": len(state), "excluded": sorted(excluded), "position_grid": "14x14 -> 16x16 bilinear"}


def shuffle_patches(tokens):
    # Official shuffle_unit: shift=8, group=2, begin=1, 256 patch tokens.
    x = torch.cat([tokens[:, 8:], tokens[:, 1:8]], dim=1)
    return x.reshape(len(x), 2, -1, x.shape[-1]).transpose(1, 2).contiguous().reshape(len(x), -1, x.shape[-1])


class ReIDModel(nn.Module):
    def __init__(self, classes, architecture="global", *, pretrained=True, weight_path=WEIGHT_PATH, drop_path=.1):
        super().__init__()
        if architecture not in {"global", "jpm"}:
            raise ValueError("Only global and JPM architectures are registered")
        self.architecture = architecture
        self.base = deit_small_patch16_224_TransReID(img_size=(256, 256), stride_size=16,
            camera=0, view=0, local_feature=architecture == "jpm", num_classes=0, drop_path_rate=drop_path)
        self.initialization = load_pretrained(self.base, weight_path) if pretrained else {"test_only_random": True}
        if architecture == "jpm":
            self.local_block = copy.deepcopy(self.base.blocks[-1])
            self.local_norm = copy.deepcopy(self.base.norm)
        count = 5 if architecture == "jpm" else 1
        self.necks = nn.ModuleList(nn.BatchNorm1d(384) for _ in range(count))
        self.heads = nn.ModuleList(nn.Linear(384, classes, bias=False) for _ in range(count))
        for neck, head in zip(self.necks, self.heads):
            neck.bias.requires_grad_(False)
            nn.init.normal_(head.weight, std=.001)
        self.dimension = count*384
        if any("sie_embed" in k for k in self.state_dict()) or self.base.cam_num or self.base.view_num:
            raise ValueError("Camera/view embeddings are forbidden")

    def features(self, images):
        # No labels, camera, filename or view argument is accepted by this image API.
        tokens = self.base(images)
        if self.architecture == "global":
            return [tokens]
        global_feature = self.base.norm(self.base.blocks[-1](tokens))[:, 0]
        patches = shuffle_patches(tokens)
        local = [self.local_norm(self.local_block(torch.cat([tokens[:, :1], part], dim=1)))[:, 0]
                 for part in patches.chunk(4, dim=1)]
        return [global_feature, *local]

    def forward(self, images):
        values = self.features(images)
        if self.training:
            return [head(neck(v)) for head, neck, v in zip(self.heads, self.necks, values)], values
        combined = values[0] if len(values) == 1 else torch.cat([values[0], *[v/4 for v in values[1:]]], dim=1)
        return F.normalize(combined, dim=1)


def soft_triplet(values, labels):
    # Batch-hard soft-margin triplet, as in the author's NO_MARGIN=True recipe.
    squared = values.square().sum(1)
    distances = (squared[:, None]+squared[None, :]-2*(values @ values.T)).clamp_min(1e-12).sqrt()
    positive = labels[:, None] == labels[None, :]
    positive.fill_diagonal_(False)
    negative = labels[:, None] != labels[None, :]
    if not positive.any(1).all() or not negative.any(1).all():
        raise ValueError("Metric learning requires P>=2, K>=2")
    hardest_pos = distances.masked_fill(~positive, -torch.inf).max(1).values
    hardest_neg = distances.masked_fill(~negative, torch.inf).min(1).values
    return F.softplus(hardest_pos-hardest_neg).mean()


def losses(model, images, labels, trial, settings):
    from training.hpo import supervised_contrastive_loss
    logits, features = model(images)
    ce = [F.cross_entropy(v, labels, label_smoothing=settings["label_smoothing"]) for v in logits]
    metric = [soft_triplet(v, labels) if trial["metric_loss"] == "soft_triplet" else
              supervised_contrastive_loss(v, labels, settings["supcon_temperature"]) for v in features]
    def combine(values):
        return values[0] if len(values) == 1 else .5*values[0]+.5*torch.stack(values[1:]).mean()
    classification, distance = combine(ce), combine(metric)
    return {"loss": classification+settings["metric_weight"]*distance, "ce": classification, "metric": distance}


@torch.inference_mode()
def encode(model, rows, paths, device, batch_size=16, progress=True):
    model.eval()
    dataset = Images(rows, paths)
    blocks = []
    for begin in range(0, len(rows), batch_size):
        images = torch.stack([dataset[i][0] for i in range(begin, min(begin+batch_size, len(rows)))])
        vectors = model(images.to(device)).cpu().numpy().astype(np.float32)
        validate_block(vectors, model.dimension)
        blocks.append(vectors)
        if progress and (begin % 128 == 0 or begin+batch_size >= len(rows)):
            print(f"FEATURES {min(begin+batch_size,len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(blocks)

