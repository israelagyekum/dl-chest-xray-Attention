"""
backbone.py — CNN feature extractor wrappers.

Supported backbones:
    "resnet50"        → feature map F: (2048, 7, 7)
    "densenet121"     → feature map F: (1024, 7, 7)
    "efficientnet_b0" → feature map F: (1280, 7, 7)

All are loaded with ImageNet-pretrained weights and the final
classification layer is removed — we attach our own head.

Usage:
    backbone, feat_dim = get_backbone("resnet50", pretrained=True)
    features = backbone(x)   # (B, C, 7, 7)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Tuple


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

_BACKBONE_CFG = {
    "resnet50": {
        "feat_dim": 2048,
        "spatial":  7,
    },
    "densenet121": {
        "feat_dim": 1024,
        "spatial":  7,
    },
    "efficientnet_b0": {
        "feat_dim": 1280,
        "spatial":  7,
    },
}


# ---------------------------------------------------------------------------
# ResNet50 wrapper
# ---------------------------------------------------------------------------

class ResNet50Backbone(nn.Module):
    """ResNet-50 up to (and including) layer4, with AdaptiveAvgPool removed."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tv_models.resnet50(weights=weights)

        # Keep everything up to (and including) layer4
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.feat_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # (B, 2048, 7, 7) for 224×224 input


# ---------------------------------------------------------------------------
# DenseNet121 wrapper
# ---------------------------------------------------------------------------

class DenseNet121Backbone(nn.Module):
    """DenseNet-121 feature extractor (all conv blocks, no classifier)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights  = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base     = tv_models.densenet121(weights=weights)
        # features includes all conv/pool/dense blocks + final BN
        self.features = base.features
        self.relu     = nn.ReLU(inplace=True)
        self.feat_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)   # (B, 1024, 7, 7)
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# EfficientNet-B0 wrapper
# ---------------------------------------------------------------------------

class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 feature extractor (all conv features, no head)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights  = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base     = tv_models.efficientnet_b0(weights=weights)
        # features: MBConv blocks, stops before AdaptiveAvgPool
        self.features = base.features
        self.feat_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # (B, 1280, 7, 7) for 224×224 input


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_backbone(name: str,
                 pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Returns (backbone_module, feature_dimension).

    Example
    -------
    >>> backbone, feat_dim = get_backbone("resnet50", pretrained=True)
    >>> x = torch.randn(2, 3, 224, 224)
    >>> feat_dim, backbone(x).shape
    (2048, torch.Size([2, 2048, 7, 7]))
    """
    name = name.lower()
    if name == "resnet50":
        m = ResNet50Backbone(pretrained)
    elif name == "densenet121":
        m = DenseNet121Backbone(pretrained)
    elif name in ("efficientnet_b0", "efficientnet-b0"):
        m = EfficientNetB0Backbone(pretrained)
    else:
        raise ValueError(f"Unknown backbone '{name}'. "
                         f"Choose from: {list(_BACKBONE_CFG.keys())}")
    return m, m.feat_dim


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    for name in ["resnet50", "densenet121", "efficientnet_b0"]:
        bb, fdim = get_backbone(name, pretrained=False)
        out = bb(x)
        print(f"{name:25s}  feat_dim={fdim:4d}  output={tuple(out.shape)}")
    print("✓ Backbone smoke-test passed.")
