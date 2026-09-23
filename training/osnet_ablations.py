"""Variant 14 building blocks. No annotation edits, downloads or MVP writes."""
import copy
import random
from dataclasses import dataclass, replace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from backend.core import DATASET, STOCK_MODEL, bbox, crop_image
from training.hpo import (ReIDExperimentModel,
                          batch_hard_triplet_loss, circle_loss,
                          supervised_contrastive_loss)
from training.osnet import GeM, MixStyle, load_encoder_from_onnx
from training.pipeline import (RandomLowerCenterOcclusion, VehicleDataset,
                               image_transforms)
from training.preprocessing import ResizeCrop, mask_crop
from training.stage6 import StepPKBatchSampler


@dataclass(frozen=True)
class Ablation:
    name: str = "B0_control"
    description: str = "Stock-start, исходный рецепт, фиксированный step budget"
    freeze_bn: str = "none"
    consistency: float | None = None
    ema: bool = False
    p: int = 16
    k: int = 2
    lower_occlusion: float = .35
    erase: float = .3
    classifier: str = "ce"
    memory_size: int = 0
    metric: str = "supcon"
    size: int = 208
    branch: str = "none"
    positive_sampling: str = "random"
    mask_probability: float = 0.
    pooling: str = "avg"
    resize: str = "square"
    mixstyle: bool = False

    def validate(self):
        if self.freeze_bn not in {"none", "backbone", "all"}:
            raise ValueError("Invalid BN policy")
        if self.classifier not in {"ce", "am_softmax"} or self.branch not in {"none", "local", "color"}:
            raise ValueError("Invalid head")
        if self.size not in {208, 256} or self.p < 2 or self.k < 2:
            raise ValueError("Invalid input size or P/K")
        if self.positive_sampling not in {"random", "balanced"}:
            raise ValueError("Invalid positive sampler")
        if self.positive_sampling == "balanced" and self.k != 2:
            raise ValueError("Balanced positive pilot uses K=2")
        if any(not 0 <= p <= 1 for p in (self.lower_occlusion, self.erase, self.mask_probability)):
            raise ValueError("Augmentation probabilities must be in [0, 1]")
        if self.memory_size < 0 or (self.memory_size and self.metric != "supcon"):
            raise ValueError("Memory is supported for SupCon only")
        if self.consistency is not None and self.consistency < 0:
            raise ValueError("Negative consistency weight")

    def recipe(self, base, seed):
        self.validate()
        config = replace(base, seed=int(seed), num_workers=0, identities_per_batch=self.p,
                         images_per_identity=self.k, pooling=self.pooling, resize_mode=self.resize,
                         use_mixstyle=self.mixstyle, metric_loss=self.metric,
                         consistency_weight=(base.consistency_weight if self.consistency is None
                                             else self.consistency))
        config.validate()
        return config


def experiment_grid():
    """One conceptual change per row; no automatic combinatorial search."""
    variants = [
        Ablation(),
        Ablation("N1_backbone_bn", "Freeze backbone BN statistics; affine обучается", freeze_bn="backbone"),
        Ablation("N2_all_bn", "Freeze backbone и BNNeck statistics", freeze_bn="all"),
        Ablation("C1_no_consistency", "Без self-consistency", consistency=0.),
        Ablation("C2_ema_consistency", "EMA target, decay=0.99; не внешний teacher", ema=True),
        Ablation("P1_p16k4", "P16K4; matched updates, удвоенный exposure", k=4),
        Ablation("P2_p32k2", "P32K2; больше identity в batch", p=32),
        Ablation("A1_no_lower_occlusion", "Без lower-center occlusion", lower_occlusion=0.),
        Ablation("A2_weak_lower_occlusion", "Lower-center occlusion p=0.1", lower_occlusion=.1),
        Ablation("A3_weak_erasing", "RandomErasing p=0.1", erase=.1),
        Ablation("M1_am_softmax", "AM-Softmax s=30, m=0.2 с ramp 100 steps", classifier="am_softmax"),
        Ablation("M2_xbm1024", "Train-only detached SupCon memory 1024, warmup 100", memory_size=1024),
        Ablation("M3_triplet", "Batch-hard triplet вместо SupCon", metric="triplet"),
        Ablation("M4_circle", "Circle вместо SupCon; остальные веса прежние", metric="circle"),
        Ablation("R1_resolution256", "256 на train и inference", size=256),
        Ablation("R2_local128", "Shared trunk + learned spatial attention 128D", branch="local"),
        Ablation("K1_color32", "RGB mean/std bypass до input IN, 32D", branch="color"),
        Ablation("T1_balanced_positives", "Stock similarity: easy/hard half cross-camera; не RPTM", positive_sampling="balanced"),
        Ablation("S1_mixed_masks", "50% original/opaque mask в robust view; bbox неизменны", mask_probability=.5),
        Ablation("G1_gem", "GeM train/inference от stock", pooling="gem"),
        Ablation("L1_letterbox", "Letterbox train/inference от stock", resize="letterbox"),
        Ablation("S2_mixstyle", "MixStyle на честном holdout", mixstyle=True),
    ]
    return {v.name: v for v in variants}


class AblationModel(ReIDExperimentModel):
    def __init__(self, num_classes, config, variant):
        super().__init__(num_classes, use_bnneck=config.use_bnneck, pooling=config.pooling,
                         resize_mode=config.resize_mode, use_mixstyle=config.use_mixstyle,
                         mixstyle_probability=config.mixstyle_probability,
                         mixstyle_alpha=config.mixstyle_alpha)
        self.variant = variant
        self.aux = None
        self.attention = None
        self.dimension = 512
        if variant.branch != "none":
            width = 128 if variant.branch == "local" else 32
            if variant.branch == "local":
                self.attention = nn.Conv2d(512, 1, 1)
            self.aux = nn.Sequential(nn.Linear(512 if variant.branch == "local" else 6, width),
                                     nn.BatchNorm1d(width))
            self.dimension += width
            self.bnneck = nn.BatchNorm1d(self.dimension)
            self.bnneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(self.dimension, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=.01)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.variant.freeze_bn != "none":
            modules = self.backbone.modules() if self.variant.freeze_bn == "backbone" else self.modules()
            for module in modules:
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()  # Affine weights remain trainable; InstanceNorm is untouched.
        return self

    def raw_embedding(self, images):
        if self.variant.branch == "none":
            return self.backbone(images)
        if self.variant.branch == "local":
            b = self.backbone
            features = b.pool1(b.conv1(b.input_IN(images)))
            features = b.mixstyle(b.pool2(b.conv2(features)))
            features = b.mixstyle(b.pool3(b.conv3(features)))
            features = b.conv5(b.conv4(features))
            pooled = b.global_pool(features).flatten(1)
            main = torch.cat([head(pooled) for head in b.fc], dim=1)
            attention = self.attention(features).flatten(2).softmax(dim=2)
            extra = self.aux((features.flatten(2) * attention).sum(dim=2))
        else:
            main = self.backbone(images)
            mean = images.mean(dim=(2, 3))
            std = ((images - mean[:, :, None, None]).square().mean(dim=(2, 3)) + 1e-6).sqrt()
            extra = self.aux(torch.cat([mean, std], dim=1))
        return torch.cat([F.normalize(main, dim=1) * .8 ** .5,
                          F.normalize(extra, dim=1) * .2 ** .5], dim=1)

    def embedding(self, images):
        return self.bnneck(self.raw_embedding(images))

    def forward(self, images):
        raw = self.raw_embedding(images)
        embedding = self.bnneck(raw)
        logits = (F.linear(F.normalize(embedding, dim=1), F.normalize(self.classifier.weight, dim=1))
                  if self.variant.classifier == "am_softmax" else self.classifier(embedding))
        return logits, raw, embedding


def initialize(num_classes, config, variant, device):
    model = AblationModel(num_classes, config, variant)
    if (isinstance(model.backbone.global_pool, GeM) != (config.pooling == "gem")
            or isinstance(model.backbone.mixstyle, MixStyle) != config.use_mixstyle):
        raise ValueError("Requested GeM/MixStyle does not match the actual architecture")
    load_encoder_from_onnx(model.backbone, STOCK_MODEL,
                           allowed_missing=("global_pool.p",) if variant.pooling == "gem" else ())
    return model.to(device)


def optimizer_for(model, config):
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    groups = [("encoder", list(model.backbone.parameters()), config.encoder_lr),
              ("head", [p for p in model.parameters() if id(p) not in backbone_ids],
               config.encoder_lr * config.head_lr_multiplier)]
    return torch.optim.AdamW([{"name": name, "params": params, "lr": lr, "base_lr": lr}
                             for name, params, lr in groups], weight_decay=config.weight_decay)


def transform_for(variant, robust=False, augment=False):
    if not augment:
        return transforms.Compose([ResizeCrop(variant.resize, variant.size), transforms.ToTensor(),
                                   transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    operations = image_transforms(robust, variant.resize).transforms
    operations[0] = ResizeCrop(variant.resize, variant.size)
    for op in operations:
        if isinstance(op, RandomLowerCenterOcclusion):
            op.probability = variant.lower_occlusion
        if isinstance(op, transforms.RandomErasing):
            op.p = variant.erase
    return transforms.Compose(operations)


class AblationDataset(VehicleDataset):
    def __init__(self, rows, variant, dataset=DATASET, augment=False, masks=None, masked=False):
        self.rows, self.variant, self.dataset = rows, variant, dataset
        self.augment, self.masks, self.masked = augment, masks, masked
        self.clean_transform = transform_for(variant, augment=augment)
        self.robust_transform = transform_for(variant, robust=True, augment=augment)
        if (masked or variant.mask_probability > 0) and masks is None:
            raise ValueError("Missing frozen mask cache")

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.dataset / "images" / f"{row['image_id']}.jpg") as image:
            crop = crop_image(image, bbox(row))
        if self.masked:
            crop = mask_crop(crop, self.masks[row["image_id"]]["rectangles"])
        clean = self.clean_transform(crop)
        if not self.augment:
            return clean, row.get("label", 0), row["image_id"]
        if self.variant.mask_probability and random.random() < self.variant.mask_probability:
            crop = mask_crop(crop, self.masks[row["image_id"]]["rectangles"])
        return clean, self.robust_transform(crop), row["label"], row["image_id"]


class BalancedPositiveSampler(StepPKBatchSampler):
    """Never removes rows/IDs. Proxy difficulty from frozen stock train features only."""
    def __init__(self, rows, config, steps, features):
        super().__init__(rows, config, steps)
        self.rows = rows
        self.features = np.stack([features[row["image_id"]] for row in rows])

    def _sample_identity(self, identity, rng):
        choices = [i for values in self.groups[identity].values() for i in values]
        anchor = rng.choice(choices)
        cross = [i for i in choices if self.rows[i]["camera_id"] != self.rows[anchor]["camera_id"]]
        candidates = cross or [i for i in choices if i != anchor] or [anchor]
        candidates.sort(key=lambda i: float(self.features[i] @ self.features[anchor]))
        middle = max(1, len(candidates) // 2)
        half = candidates[:middle] if rng.random() < .5 else candidates[middle:]
        return [anchor, rng.choice(half or candidates)]


class MemoryBank:
    def __init__(self, capacity):
        self.capacity = capacity
        self.features = self.labels = self.ids = None

    def push(self, features, labels, ids):
        if not self.capacity:
            return
        values = [F.normalize(features.detach(), dim=1), labels.detach(), ids.detach()]
        for name, value in zip(("features", "labels", "ids"), values):
            previous = getattr(self, name)
            value = value if previous is None else torch.cat([previous, value])
            setattr(self, name, value[-self.capacity:].clone())

    def state_dict(self):
        return {key: getattr(self, key) for key in ("features", "labels", "ids")}

    def load_state_dict(self, state, device):
        for key, value in state.items():
            setattr(self, key, None if value is None else value.to(device))

    def loss(self, features, labels, ids, temperature):
        anchors = F.normalize(features, dim=1)
        candidates = torch.cat([anchors, self.features])
        candidate_labels = torch.cat([labels, self.labels])
        candidate_ids = torch.cat([ids, self.ids])
        allowed = ids[:, None] != candidate_ids[None, :]
        positives = (labels[:, None] == candidate_labels[None, :]) & allowed
        valid = positives.any(dim=1)
        if not valid.any():
            return features.sum() * 0.
        logits = (anchors @ candidates.T / temperature)[valid]
        allowed, positives = allowed[valid], positives[valid]
        log_prob = logits - logits.masked_fill(~allowed, -torch.inf).logsumexp(dim=1, keepdim=True)
        return -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()


def ema_teacher(model):
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_teacher(teacher, model, decay=.99):
    for target, source in zip(teacher.parameters(), model.parameters()):
        target.lerp_(source.detach(), 1 - decay)
    # Running statistics belong to the current data domain, not another EMA filter.
    for target, source in zip(teacher.buffers(), model.buffers()):
        target.copy_(source)


def losses(model, clean, robust, labels, ids, config, variant, bank, step, teacher=None):
    target = None
    if config.consistency_weight:
        target_model = teacher if teacher is not None else model
        target_model.eval()
        with torch.no_grad():
            target = target_model.embedding(clean)
    model.train()
    logits, raw, embedding = model(robust)
    if variant.classifier == "am_softmax":
        margin = .2 * min(1., (step + 1) / 100)
        logits = 30 * (logits - margin * F.one_hot(labels, logits.shape[1]))
    classification = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
    if config.metric_loss == "triplet":
        metric = batch_hard_triplet_loss(raw, labels, config.triplet_margin)
    elif config.metric_loss == "circle":
        metric = circle_loss(raw, labels, config.circle_margin, config.circle_gamma)
    elif bank.features is not None and step >= 100:
        metric = bank.loss(raw, labels, ids, config.supcon_temperature)
    else:
        metric = supervised_contrastive_loss(raw, labels, config.supcon_temperature)
    consistency = raw.new_zeros(()) if target is None else (1 - F.cosine_similarity(embedding, target, dim=1)).mean()
    total = classification + config.metric_weight * metric + config.consistency_weight * consistency
    return {"loss": total, "classification": classification, "metric": metric,
            "consistency": consistency, "embedding_norm": raw.norm(dim=1).mean(),
            "embedding_std": embedding.std(dim=0).mean()}, raw


class InferenceEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return F.normalize(self.model.embedding(images), dim=1)
