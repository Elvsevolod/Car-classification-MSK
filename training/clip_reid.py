"""Vehicle CLIP-ReID: official ViT, fresh ID prompts, image-only inference, no SIE.

Adapted from Syliz517/CLIP-ReID (MIT), commit in clip_source.py. The unmodified
attention/vision implementation and its license are under training/vendor/clip_reid.
"""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from backend.core import DATASET, bbox, crop_image, normalize
from training import clip_source
from training.vendor.clip_reid.model import LayerNorm, Transformer, VisionTransformer

PREPROCESS = "exif-rgb-exact-bbox-bilinear256-mean0.5-std0.5-beforeBN-concat1280-l2-v1"


class PromptLearner(nn.Module):
    """A photo of a [X][X][X][X] vehicle. Same frozen prefix/suffix as the source."""

    def __init__(self, classes):
        super().__init__()
        self.cls_ctx = nn.Parameter(torch.empty(classes, 4, 512))
        nn.init.normal_(self.cls_ctx, std=.02)
        self.register_buffer("token_prefix", torch.zeros(1, 5, 512))
        self.register_buffer("token_suffix", torch.zeros(1, 68, 512))

    def forward(self, labels):
        return torch.cat((self.token_prefix.expand(len(labels), -1, -1),
                          self.cls_ctx[labels], self.token_suffix.expand(len(labels), -1, -1)), dim=1)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        mask = torch.full((77, 77), float("-inf")).triu_(1)
        self.transformer = Transformer(512, 12, 8, mask)
        self.positional_embedding = nn.Parameter(torch.zeros(77, 512))
        self.ln_final = LayerNorm(512)
        self.text_projection = nn.Parameter(torch.zeros(512, 512))

    def forward(self, prompts):
        x = self.transformer((prompts + self.positional_embedding).permute(1, 0, 2))
        # EOT index 11 for the FIXED official vehicle prompt; no free-text tokenizer.
        return self.ln_final(x.permute(1, 0, 2))[:, 11] @ self.text_projection


class ClipReID(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.image_encoder = VisionTransformer(16, 16, 16, 16, 768, 12, 12, 512)
        self.text_encoder = TextEncoder()
        self.prompt_learner = PromptLearner(classes)
        self.classifier = nn.Linear(768, classes, bias=False)
        self.classifier_proj = nn.Linear(512, classes, bias=False)
        self.bottleneck = nn.BatchNorm1d(768)
        self.bottleneck_proj = nn.BatchNorm1d(512)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck_proj.bias.requires_grad_(False)

    def reset_identities(self, classes):
        # Source IDs (VeRi) and local IDs have no correspondence. Reset both heads/prompts.
        device = next(self.parameters()).device
        prefix, suffix = self.prompt_learner.token_prefix, self.prompt_learner.token_suffix
        self.prompt_learner = PromptLearner(classes).to(device)
        self.prompt_learner.token_prefix.copy_(prefix)
        self.prompt_learner.token_suffix.copy_(suffix)
        self.classifier = nn.Linear(768, classes, bias=False).to(device)
        self.classifier_proj = nn.Linear(512, classes, bias=False).to(device)
        nn.init.normal_(self.classifier.weight, std=.001)
        nn.init.normal_(self.classifier_proj.weight, std=.001)

    def text(self, labels):
        return self.text_encoder(self.prompt_learner(labels))

    def projected_image(self, images):
        return self.image_encoder(images)[2][:, 0]

    def embedding(self, images):
        _, feature, projected = self.image_encoder(images)
        # Official VeRi NECK_FEAT='before': normalize the CONCATENATION, not each part.
        return F.normalize(torch.cat((feature[:, 0], projected[:, 0]), dim=1), dim=1)

    def forward(self, images):
        previous, feature, projected = self.image_encoder(images)
        features = [previous[:, 0], feature[:, 0], projected[:, 0]]
        scores = [self.classifier(self.bottleneck(features[1])),
                  self.classifier_proj(self.bottleneck_proj(features[2]))]
        return scores, features, features[2]

    def set_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError("CLIP-ReID stages are 1 (prompts) and 2 (images)")
        for name, parameter in self.named_parameters():
            enabled = name.startswith("prompt_learner.cls_ctx") if stage == 1 else not name.startswith(
                ("text_encoder.", "prompt_learner."))
            parameter.requires_grad_(enabled and name not in ("bottleneck.bias", "bottleneck_proj.bias"))
        self.train()
        self.text_encoder.eval()
        if stage == 1:
            self.image_encoder.eval()

    def image_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                if not key.startswith(("text_encoder.", "prompt_learner."))}

    def load_image_state(self, state):
        expected = {key for key in self.state_dict() if not key.startswith(("text_encoder.", "prompt_learner."))}
        if set(state) != expected:
            raise ValueError("Incomplete or incompatible image checkpoint")
        self.load_state_dict(state, strict=False)


def load_pretrained(path, classes=None, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Download the official VeRi CLIP-ReID checkpoint first: {path}")
    if clip_source.file_sha256(path) != clip_source.CHECKPOINT_SHA256:
        raise ValueError("CLIP-ReID source checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if any("cv_embed" in key for key in state):
        raise ValueError("Camera/view SIE checkpoint is not permitted")
    source_classes = state["classifier.weight"].shape[0]
    model = ClipReID(source_classes)
    model.load_state_dict(state, strict=True)  # No silent missing encoder weights.
    del state
    if classes is not None:
        model.reset_identities(classes)
    return model.float().to(device).eval()


def image_transform(train=False):
    steps = [T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC if train
                      else T.InterpolationMode.BILINEAR)]
    if train:
        steps += [T.RandomHorizontalFlip(.5), T.Pad(10), T.RandomCrop((256, 256))]
    steps += [T.ToTensor(), T.Normalize([.5] * 3, [.5] * 3)]
    if train:
        # torchvision implementation, documented adaptation of upstream timm pixel erasing.
        steps += [T.RandomErasing(p=.5, scale=(.02, 1 / 3), ratio=(.3, 1 / .3), value="random")]
    return T.Compose(steps)


class ClipDataset(Dataset):
    def __init__(self, rows, dataset=DATASET, train=False):
        self.rows, self.dataset, self.transform = rows, Path(dataset), image_transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        with Image.open(self.dataset / "images" / f"{row['image_id']}.jpg") as image:
            tensor = self.transform(crop_image(image, bbox(row)))
        return tensor, row.get("label", -1), row["image_id"]


def prompt_loss(image, text, labels):
    # Upstream temperature 1, raw (not unit-normalized) dot products; multi-positive CE.
    positive = labels[:, None].eq(labels[None, :]).to(image.dtype)
    logits = image @ text.T
    return -((F.log_softmax(logits, dim=1) * positive).sum(1) / positive.sum(1)).mean() \
           - ((F.log_softmax(logits.T, dim=1) * positive).sum(1) / positive.sum(1)).mean()


def triplet_loss(features, labels, margin=.3):
    # Squared distance formula avoids an unsupported MPS cdist-backward path.
    squared = features.square().sum(1, keepdim=True)
    distances = (squared + squared.T - 2 * features @ features.T).clamp_min(1e-12).sqrt()
    same = labels[:, None].eq(labels[None, :])
    if not (~same).any(1).all() or not (same.sum(1) >= 2).all():
        raise ValueError("Triplet requires P>=2, K>=2")
    hard_positive = distances.masked_fill(~same, float("-inf")).max(1).values
    hard_negative = distances.masked_fill(same, float("inf")).min(1).values
    return F.relu(hard_positive - hard_negative + margin).mean()


def image_losses(model, images, labels, text_features):
    scores, features, projected = model(images)
    identification = sum(F.cross_entropy(score, labels, label_smoothing=.1) for score in scores)
    triplet = sum(triplet_loss(feature, labels) for feature in features)
    image_text = F.cross_entropy(projected @ text_features.T, labels, label_smoothing=.1)
    return {"loss": .25 * identification + triplet + image_text,
            "id_loss": identification, "triplet": triplet, "image_text": image_text}


def synchronize(device):
    if torch.device(device).type == "mps":
        torch.mps.synchronize()
    elif torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def release_device(device):
    gc.collect()
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()
    elif torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def technical_smoke(model, images, labels, device, prompt_batch=64):
    """Real backward + Adam steps; restore EVERY parameter and BN buffer afterwards."""
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    was_training = model.training
    images, labels = images.to(device), labels.to(device)
    try:
        synchronize(device)
        started = time.perf_counter()
        model.set_stage(1)
        with torch.no_grad():
            visual = model.projected_image(images)
        # Cached features may be repeated for this resource check, never for a quality metric.
        indices = torch.arange(prompt_batch, device=device) % len(labels)
        prompt_labels = labels[indices]
        optimizer = torch.optim.Adam([model.prompt_learner.cls_ctx], lr=3.5e-4)
        loss = prompt_loss(visual[indices], model.text(prompt_labels), prompt_labels)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(model.prompt_learner.cls_ctx.grad).all():
            raise FloatingPointError("Prompt smoke failed")
        optimizer.step()
        del optimizer
        model.zero_grad(set_to_none=True)
        model.set_stage(2)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-6)
        with torch.no_grad():
            texts = torch.cat([model.text(batch) for batch in
                               torch.arange(model.classifier.out_features, device=device).split(32)])
        loss2 = image_losses(model, images, labels, texts)["loss"]
        loss2.backward()
        if not torch.isfinite(loss2) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                           for p in model.parameters()):
            raise FloatingPointError("Image smoke failed")
        optimizer.step()
        synchronize(device)
        report = {"device": str(device), "dtype": "float32", "batch_size": len(images),
                  "prompt_batch": prompt_batch, "adam_steps_checked": True,
                  "stage1_loss": float(loss.detach().cpu()), "stage2_loss": float(loss2.detach().cpu()),
                  "forward_backward_seconds": time.perf_counter() - started,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "image_weights_bytes": sum(v.numel() * v.element_size() for v in model.image_state().values()),
                  "not_official_performance_benchmark": True}
        if torch.device(device).type == "mps":
            report.update(mps_current_bytes=torch.mps.current_allocated_memory(),
                          mps_driver_bytes=torch.mps.driver_allocated_memory(),
                          mps_recommended_bytes=torch.mps.recommended_max_memory())
        return report
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial)
        for name, p in model.named_parameters():
            p.requires_grad_(requires_grad[name])
        model.train(was_training)


@torch.no_grad()
def encode_rows(model, rows, device, dataset=DATASET, batch_size=16, projected=False):
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    model.eval()
    result = {}
    loader = DataLoader(ClipDataset(rows, dataset), batch_size=batch_size, num_workers=0)
    for images, _, ids in tqdm(loader, desc="CLIP embeddings", leave=False):
        vectors = (model.projected_image(images.to(device)) if projected
                   else model.embedding(images.to(device))).cpu().numpy()
        if not np.isfinite(vectors).all():
            raise FloatingPointError("Invalid CLIP embeddings")
        vectors = vectors if projected else normalize(vectors)
        result.update(zip(ids, vectors))
    return result
