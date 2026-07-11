"""Model builder: pretrained ResNet backbone with a documented norm policy.

PLAN §5. Default ResNet-18 (light enough for many-client simulation, appropriate for
64px), ImageNet-pretrained for transfer learning. The normalization choice is explicit:
BatchNorm by default; GroupNorm as a BN-free control. FedBN (keep BN local) is NOT a
model change — it is handled by the FL strategy in P2/P4, so the same model is reused.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _expand_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Multispectral stem (PLAN §3.4): widen conv1 to `in_channels`, copying ImageNet
    RGB weights into the first 3 channels and the mean of RGB into the rest."""
    new = nn.Conv2d(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=(conv.bias is not None))
    with torch.no_grad():
        w = conv.weight.data.clone()          # [out, 3, k, k]
        new.weight[:, :3] = w[:, :3]
        if in_channels > 3:
            new.weight[:, 3:] = w.mean(dim=1, keepdim=True).expand(-1, in_channels - 3, -1, -1)
    return new


def _bn_to_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """Recursively replace BatchNorm2d with GroupNorm (BN-free control for the E4 ablation)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            g = min(num_groups, child.num_features)
            while child.num_features % g != 0:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, child.num_features))
        else:
            _bn_to_gn(child, num_groups)
    return module


def build_model(backbone: str = "resnet18", num_classes: int = 10,
                pretrained: bool = True, in_channels: int = 3, norm: str = "bn") -> nn.Module:
    from torchvision import models

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    if in_channels != 3:
        net.conv1 = _expand_first_conv(net.conv1, in_channels)

    net.fc = nn.Linear(net.fc.in_features, num_classes)

    if norm == "gn":
        net = _bn_to_gn(net)
    elif norm != "bn":
        raise ValueError(f"unknown norm policy: {norm}")

    return net
