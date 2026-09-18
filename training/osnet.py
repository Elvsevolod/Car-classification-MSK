"""Trainable OSNet-AIN compatible with the bundled vehicle-reid-0001 ONNX.

The architecture follows the MIT-licensed vehicle_reid branch of deep-person-reid:
https://github.com/sovrasov/deep-person-reid/blob/vehicle_reid/torchreid/models/osnet_ain.py
Only the blocks used by vehicle-reid-0001 are retained here.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, instance_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = (nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d)(out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.layers = nn.Sequential(*(LightConv3x3(channels) for _ in range(depth)))

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        gate = self.fc2(self.relu(self.fc1(self.global_avgpool(x))))
        return x * self.gate_activation(gate)


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, instance_inside=False):
        super().__init__()
        middle = out_channels // 4
        self.conv1 = Conv1x1(in_channels, middle)
        self.conv2 = nn.ModuleList(LightConvStream(middle, depth) for depth in range(1, 5))
        self.gate = ChannelGate(middle)
        self.conv3 = Conv1x1Linear(middle, out_channels, batch_norm=not instance_inside)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if instance_inside else None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        x = self.conv1(x)
        combined = sum(self.gate(stream(x)) for stream in self.conv2)
        combined = self.conv3(combined)
        if self.IN is not None:
            combined = self.IN(combined)
        return F.relu(combined + identity)


class GeM(nn.Module):
    """Generalized mean pooling with one trainable exponent."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp_min(self.eps)
        return F.adaptive_avg_pool2d(x.clamp_min(self.eps).pow(p), 1).pow(1 / p)


class MixStyle(nn.Module):
    """Training-only mixing of per-instance feature statistics."""

    def __init__(self, probability=.5, alpha=.1, eps=1e-6):
        super().__init__()
        if not 0 <= probability <= 1 or alpha <= 0:
            raise ValueError("MixStyle probability must be in [0, 1] and alpha positive")
        self.probability = probability
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.probability:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        mean, std = mean.detach(), std.detach()
        normalized = (x - mean) / std
        weights = self.beta.sample((len(x), 1, 1, 1)).to(x.device)
        permutation = torch.randperm(len(x), device=x.device)
        mixed_mean = mean * weights + mean[permutation] * (1 - weights)
        mixed_std = std * weights + std[permutation] * (1 - weights)
        return normalized * mixed_std + mixed_mean


class VehicleOSNet(nn.Module):
    """Exact 512-D encoder structure used by the bundled ONNX checkpoint."""

    def __init__(self, pooling="avg", use_mixstyle=False,
                 mixstyle_probability=.5, mixstyle_alpha=.1):
        super().__init__()
        if pooling not in {"avg", "gem"}:
            raise ValueError("pooling must be 'avg' or 'gem'")
        self.input_IN = nn.InstanceNorm2d(3, affine=True)
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3, instance_norm=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OSBlock(64, 256, True), OSBlock(256, 256, True))
        self.pool2 = nn.Sequential(Conv1x1(256, 256), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(OSBlock(256, 384), OSBlock(384, 384, True))
        self.pool3 = nn.Sequential(Conv1x1(384, 384), nn.AvgPool2d(2, stride=2))
        self.mixstyle = (MixStyle(mixstyle_probability, mixstyle_alpha)
                         if use_mixstyle else nn.Identity())
        self.conv4 = nn.Sequential(OSBlock(384, 512, True), OSBlock(512, 512))
        self.conv5 = Conv1x1(512, 512)
        self.global_pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else GeM()
        # The public vehicle model concatenates two 256-D source heads.
        self.fc = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256)),
            nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256)),
        ])

    def forward(self, x):
        x = self.pool1(self.conv1(self.input_IN(x)))
        x = self.mixstyle(self.pool2(self.conv2(x)))
        x = self.mixstyle(self.pool3(self.conv3(x)))
        x = self.conv5(self.conv4(x))
        x = self.global_pool(x).flatten(1)
        return torch.cat([head(x) for head in self.fc], dim=1)


class ReIDTrainerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = VehicleOSNet()
        self.classifier = nn.Linear(512, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.classifier(embedding), embedding


def load_encoder_from_onnx(encoder, onnx_path, allowed_missing=()):
    """Load the frozen ONNX initializers into the trainable PyTorch encoder."""
    import onnx
    from onnx import numpy_helper

    initializers = {item.name: torch.from_numpy(np.array(numpy_helper.to_array(item), copy=True))
                    for item in onnx.load(str(onnx_path)).graph.initializer}
    state = encoder.state_dict()
    loaded = {}
    mismatched = []
    for name, tensor in state.items():
        source = initializers.get(name)
        if source is None:
            if not name.endswith("num_batches_tracked") and name not in allowed_missing:
                mismatched.append(name)
            loaded[name] = tensor
        elif source.shape != tensor.shape:
            mismatched.append(name)
            loaded[name] = tensor
        else:
            loaded[name] = source.to(dtype=tensor.dtype)
    unexpected = sorted(set(initializers) - set(state))
    if mismatched or unexpected:
        raise ValueError(f"ONNX/PyTorch mismatch; missing={mismatched}, unexpected={unexpected}")
    encoder.load_state_dict(loaded)
    return len(initializers)


def export_encoder_onnx(encoder, output_path, device="cpu"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = encoder.eval().to(device)
    example = torch.zeros(1, 3, 208, 208, device=device)
    torch.onnx.export(
        encoder, example, output_path, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17, dynamo=False,
    )
    return output_path
