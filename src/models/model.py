"""
model.py — Full end-to-end model assembly.

Two model variants:

1.  BaselineModel
    --------------
    backbone → GAP → Linear(14)
    Standard multi-label classifier — used for Phase 2 baselines.
    Grad-CAM is applied to this model post-hoc.

2.  AttentionModel  (the contribution)
    ----------------------------------------
    backbone → [ChannelAttn →] SpatialAttn → GAP → Linear(14)
    The spatial attention map is returned as a first-class output
    and supervised against bounding-box masks via L_attn.

Both models share the same forward signature so the training loop
can use them interchangeably.

Forward returns:
    logits   : (B, 14)       — raw (pre-sigmoid) scores
    attn_map : (B, 1, G, G)  — spatial attention  (zeros for BaselineModel)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.models.backbone     import get_backbone
from src.models.attention    import AttentionModule
from src.models.correlation  import CorrelationRegularizer


# ---------------------------------------------------------------------------
# Baseline model (no attention supervision)
# ---------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """
    Standard backbone + GAP + linear head.
    Returned attn_map is always zeros (placeholder for compatibility).
    """

    def __init__(self,
                 backbone_name: str  = "resnet50",
                 pretrained:    bool = True,
                 num_classes:   int  = 14):
        super().__init__()
        self.backbone, self.feat_dim = get_backbone(backbone_name, pretrained)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Linear(self.feat_dim, num_classes)
        self._G     = 7   # spatial size (for placeholder attn_map)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        feat     = self.backbone(x)                      # (B, C, G, G)
        G        = feat.size(-1)
        pooled   = self.pool(feat).view(B, -1)           # (B, C)
        logits   = self.head(pooled)                     # (B, 14)
        attn_map = torch.zeros(B, 1, G, G, device=x.device)  # placeholder
        return logits, attn_map

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Expose feature map for Grad-CAM."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Attention model (our contribution)
# ---------------------------------------------------------------------------

class AttentionModel(nn.Module):
    """
    Backbone + supervised attention + GAP + linear head.

    The spatial attention map A is:
      - Supervised on boxed images via L_attn (Dice + MSE).
      - Regularised on unboxed images via L1 sparsity.

    Parameters
    ----------
    backbone_name      : "resnet50" | "densenet121" | "efficientnet_b0"
    pretrained         : load ImageNet weights
    num_classes        : 14
    use_channel_attn   : prepend CBAM channel attention
    dropout_rate       : dropout before linear head (0 = disabled)
    """

    def __init__(self,
                 backbone_name:    str   = "resnet50",
                 pretrained:       bool  = True,
                 num_classes:      int   = 14,
                 use_channel_attn: bool  = True,
                 dropout_rate:     float = 0.0):
        super().__init__()
        self.backbone, self.feat_dim = get_backbone(backbone_name, pretrained)
        self.attn_module = AttentionModule(
            in_channels=self.feat_dim,
            use_channel_attn=use_channel_attn,
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head    = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits   : (B, 14)       — pre-sigmoid classification scores
        attn_map : (B, 1, G, G)  — spatial attention map in [0, 1]
        """
        feat             = self.backbone(x)              # (B, C, G, G)
        attended, attn   = self.attn_module(feat)        # (B, C, G, G), (B, 1, G, G)
        pooled           = self.pool(attended).view(x.size(0), -1)  # (B, C)
        pooled           = self.dropout(pooled)
        logits           = self.head(pooled)             # (B, 14)
        return logits, attn

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Expose feature map for Grad-CAM comparison."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict,
                cooc_matrix=None,
                variant: bool = True) -> nn.Module:
    """
    Build and return the model specified in cfg.

    Parameters
    ----------
    cfg          : config dict (from config.yaml)
    cooc_matrix  : optional np.ndarray (14, 14) for CorrelationRegularizer
                   (attached as model.corr_reg if provided)
    variant      : True → AttentionModel,  False → BaselineModel
    """
    name       = cfg.get("backbone",          "resnet50")
    pretrained = cfg.get("pretrained",        True)
    n_classes  = cfg.get("num_classes",       14)
    ch_attn    = cfg.get("use_channel_attn",  True)

    if variant:
        model = AttentionModel(
            backbone_name=name,
            pretrained=pretrained,
            num_classes=n_classes,
            use_channel_attn=ch_attn,
        )
    else:
        model = BaselineModel(
            backbone_name=name,
            pretrained=pretrained,
            num_classes=n_classes,
        )

    # Attach correlation regularizer as a sub-module (moves to device with model)
    if cooc_matrix is not None:
        model.corr_reg = CorrelationRegularizer(
            cooc_matrix,
            threshold=0.2,
        )
    else:
        model.corr_reg = None

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    x = torch.randn(2, 3, 224, 224)
    cooc = np.random.rand(14, 14).astype(np.float32)
    np.fill_diagonal(cooc, 1.0)

    for backbone in ["resnet50", "densenet121", "efficientnet_b0"]:
        for variant in [False, True]:
            label = "Attention" if variant else "Baseline"
            cfg   = {
                "backbone": backbone, "pretrained": False,
                "num_classes": 14, "use_channel_attn": True,
            }
            model  = build_model(cfg, cooc_matrix=cooc, variant=variant)
            logits, attn = model(x)
            n_params = count_parameters(model)
            print(f"[{label:10s}] {backbone:22s}  "
                  f"logits={tuple(logits.shape)}  "
                  f"attn={tuple(attn.shape)}  "
                  f"params={n_params/1e6:.1f}M")

    print("✓ Model smoke-test passed.")
