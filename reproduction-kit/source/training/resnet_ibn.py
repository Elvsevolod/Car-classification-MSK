"""ResNet50-IBN-a backbone for the stage-5 vehicle ReID experiment.

The architecture and public ImageNet weights follow the MIT-licensed IBN-Net
release v1.0: https://github.com/XingangPan/IBN-Net
"""
import hashlib
from pathlib import Path

import torch
from torch import nn

from training.osnet import GeM


PRETRAINED_URL = (
    "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/"
    "resnet50_ibn_a-d9d0bb7b.pth"
)
PRETRAINED_VERSION = "XingangPan/IBN-Net v1.0"
EMBEDDING_DIMENSION = 2048


class IBN(nn.Module):
    """Split channels between instance and batch normalization."""

    def __init__(self, channels, ratio=.5):
        super().__init__()
        self.half = int(channels * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(channels - self.half)

    def forward(self, values):
        first, second = torch.split(values, self.half, dim=1)
        return torch.cat((self.IN(first.contiguous()), self.BN(second.contiguous())), dim=1)


class BottleneckIBN(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, use_ibn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = IBN(channels) if use_ibn else nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, values):
        residual = values
        values = self.relu(self.bn1(self.conv1(values)))
        values = self.relu(self.bn2(self.conv2(values)))
        values = self.bn3(self.conv3(values))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(values + residual)


class ResNet50IBNBackbone(nn.Module):
    """ResNet50-IBN-a with ReID last-stride=1 and AvgPool or GeM output."""

    def __init__(self, pooling="gem", last_stride=1):
        super().__init__()
        if pooling not in {"avg", "gem"}:
            raise ValueError("pooling must be 'avg' or 'gem'")
        if last_stride not in {1, 2}:
            raise ValueError("last_stride must be 1 or 2")
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, use_ibn=True)
        self.layer2 = self._make_layer(128, 4, stride=2, use_ibn=True)
        self.layer3 = self._make_layer(256, 6, stride=2, use_ibn=True)
        self.layer4 = self._make_layer(
            512, 3, stride=last_stride, use_ibn=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else GeM()
        self._initialize_weights()

    def _make_layer(self, channels, blocks, stride=1, use_ibn=True):
        output_channels = channels * BottleneckIBN.expansion
        downsample = None
        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        layers = [BottleneckIBN(
            self.in_channels, channels, stride, downsample, use_ibn)]
        self.in_channels = output_channels
        layers.extend(BottleneckIBN(
            self.in_channels, channels, use_ibn=use_ibn) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, values):
        values = self.maxpool(self.relu(self.bn1(self.conv1(values))))
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        values = self.layer4(values)
        return self.global_pool(values).flatten(1)


class ResNetIBNReIDModel(nn.Module):
    """ResNet50-IBN-a + GeM + BNNeck and a train-only classifier."""

    def __init__(self, num_classes, pooling="gem", last_stride=1,
                 resize_mode="square", use_bnneck=True):
        super().__init__()
        self.backbone = ResNet50IBNBackbone(pooling, last_stride)
        self.resize_mode = resize_mode
        self.bnneck = (nn.BatchNorm1d(EMBEDDING_DIMENSION)
                       if use_bnneck else nn.Identity())
        self.classifier = nn.Linear(
            EMBEDDING_DIMENSION, num_classes, bias=not use_bnneck)
        if use_bnneck:
            self.bnneck.bias.requires_grad_(False)
        nn.init.normal_(self.classifier.weight, std=.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def embedding(self, images):
        return self.bnneck(self.backbone(images))

    def forward(self, images):
        raw = self.backbone(images)
        embedding = self.bnneck(raw)
        return self.classifier(embedding), raw, embedding

    def inference_module(self):
        return nn.Sequential(self.backbone, self.bnneck)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_imagenet_weights(backbone, progress=True, model_dir=None):
    """Load the official v1.0 checkpoint, excluding its ImageNet classifier."""
    state = torch.hub.load_state_dict_from_url(
        PRETRAINED_URL,
        model_dir=str(model_dir) if model_dir is not None else None,
        map_location="cpu",
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    transferred = {key: value for key, value in state.items()
                   if not key.startswith("fc.")}
    missing, unexpected = backbone.load_state_dict(transferred, strict=False)
    expected_missing = ["global_pool.p"] if isinstance(backbone.global_pool, GeM) else []
    if sorted(missing) != expected_missing or unexpected:
        raise ValueError(
            f"Official IBN checkpoint mismatch; missing={missing}, unexpected={unexpected}")

    checkpoint_dir = (Path(model_dir) if model_dir is not None
                      else Path(torch.hub.get_dir()) / "checkpoints")
    checkpoint_path = checkpoint_dir / Path(PRETRAINED_URL).name
    return {
        "source": PRETRAINED_VERSION,
        "url": PRETRAINED_URL,
        "checkpoint": str(checkpoint_path),
        "sha256": _file_sha256(checkpoint_path),
        "loaded_tensors": len(transferred),
    }
