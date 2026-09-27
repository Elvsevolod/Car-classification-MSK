"""Isolated fixed-cosine branch experiment; historical ablations stay unchanged.

Both the training metric embedding (``raw_embedding``) and inference embedding
use independent branch BN -> branch L2 -> sqrt(.8/.2) -> concatenation. There is
no BN after concatenation. For nondegenerate branches, the resulting cosine is
exactly .8 * main cosine + .2 * auxiliary cosine. BN uses batch statistics during
training and running statistics during inference; the weighting rule is the same.

A zero or numerically tiny branch has no defined cosine. ``unit_branch`` maps it
to the first unit basis vector, explicitly retaining finite outputs and branch
energy even on degenerate inputs. Similarities involving this fallback are NOT
cosines of the original zero vector; collapsed branches must not be interpreted
as learned evidence. This deterministic rule is also used by the ONNX export.
"""

import torch
from torch import nn
from torch.nn import functional as F

from backend.core import STOCK_MODEL
from training import osnet_ablations as blocks
from training.osnet import GeM, MixStyle, load_encoder_from_onnx


def unit_branch(features):
    """L2 normalization with an explicit unit-basis fallback at norm <= 1e-12."""
    norm = features.norm(p=2, dim=1, keepdim=True)
    fallback = torch.cat([torch.ones_like(features[:, :1]),
                          torch.zeros_like(features[:, 1:])], dim=1)
    return torch.where(norm > 1e-12, F.normalize(features, dim=1), fallback)


class FixedFusionModel(blocks.AblationModel):
    """Local128/color32 with fixed 80/20 squared-norm contributions.

    ``raw`` and ``embedding`` in the legacy forward/loss interface intentionally
    coincide here: metric, classification and consistency see the same branch
    geometry. The name ``raw`` is retained only for loss API compatibility.
    """

    def __init__(self, num_classes, config, variant):
        variant.validate()
        if variant.branch not in {"local", "color"}:
            raise ValueError("FixedFusionModel requires a local or color branch")
        if not config.use_bnneck:
            raise ValueError("Fixed branch fusion requires independent BNNecks")
        super().__init__(num_classes, config, variant)
        self.main_bn = nn.BatchNorm1d(512)
        self.main_bn.bias.requires_grad_(False)
        # The auxiliary branch already has its own BN in self.aux[1].
        self.bnneck = nn.Identity()

    def branch_embeddings(self, images):
        """Return independently BN-normalized branches, before branch L2."""
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
        return self.main_bn(main), extra

    def raw_embedding(self, images):
        main, extra = self.branch_embeddings(images)
        return torch.cat([unit_branch(main) * .8 ** .5,
                          unit_branch(extra) * .2 ** .5], dim=1)

    def inference_module(self):
        # ReIDExperimentModel's backbone-only implementation would lose the aux branch.
        return blocks.InferenceEncoder(self)


def initialize_fixed_fusion(num_classes, config, variant, device):
    """Load stock weights without changing legacy initialization for branch='none'."""
    if variant.branch == "none":
        return blocks.initialize(num_classes, config, variant, device)
    model = FixedFusionModel(num_classes, config, variant)
    if (isinstance(model.backbone.global_pool, GeM) != (config.pooling == "gem")
            or isinstance(model.backbone.mixstyle, MixStyle) != config.use_mixstyle):
        raise ValueError("Requested GeM/MixStyle does not match the actual architecture")
    load_encoder_from_onnx(model.backbone, STOCK_MODEL,
                           allowed_missing=("global_pool.p",) if variant.pooling == "gem" else ())
    return model.to(device)
