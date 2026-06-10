"""
attention.py — Explanation-supervised spatial attention module.

Architecture (CBAM-inspired, but attention map is a first-class output):

    Channel attention  (optional):
        A_c = σ( MLP( AvgPool_spatial(F) ) + MLP( MaxPool_spatial(F) ) )
        F_c = F ⊙ A_c

    Spatial attention (the core contribution):
        A_s = σ( Conv( [AvgPool_channel(F_c);  MaxPool_channel(F_c)] ) )
        F'  = F_c ⊙ A_s          (broadcast over channels)

The spatial map A_s is returned as a separate output so the training loop
can compute L_attn against the bounding-box mask.

Shapes:
    Input  F  : (B, C, G, G)   — feature map from backbone (G = 7 or 14)
    Output F' : (B, C, G, G)   — attended feature map
    Output A_s: (B, 1, G, G)   — spatial attention map  ∈ (0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Channel Attention (optional; CBAM-style)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    ratio: bottleneck reduction ratio for the MLP.
    """

    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // ratio)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = F.adaptive_avg_pool2d(x, 1)   # (B, C, 1, 1)
        max_ = F.adaptive_max_pool2d(x, 1)  # (B, C, 1, 1)
        avg  = self.mlp(avg.view(B, C))      # (B, C)
        max_ = self.mlp(max_.view(B, C))     # (B, C)
        scale = torch.sigmoid(avg + max_).view(B, C, 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# Spatial Attention (the novel supervised module)
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """
    CBAM-style spatial attention but its output A is a first-class trainable
    map that is supervised against bounding-box masks during training.

    kernel_size: conv kernel to aggregate avg+max channel pools.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # 2 input channels (avg-pool + max-pool along channel dim)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        attended : (B, C, G, G)  — x ⊙ A  (feature-weighted)
        attn_map : (B, 1, G, G)  — spatial attention map in [0, 1]
        """
        avg_pool = x.mean(dim=1, keepdim=True)          # (B, 1, G, G)
        max_pool = x.max(dim=1, keepdim=True).values    # (B, 1, G, G)
        pooled   = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, G, G)
        attn_map = torch.sigmoid(self.conv(pooled))     # (B, 1, G, G)
        attended = x * attn_map                         # broadcast over C
        return attended, attn_map


# ---------------------------------------------------------------------------
# Full Attention Module (channel + spatial, or spatial only)
# ---------------------------------------------------------------------------

class AttentionModule(nn.Module):
    """
    Combines optional channel attention with the supervised spatial attention.

    Parameters
    ----------
    in_channels      : number of channels in the feature map (C)
    use_channel_attn : whether to prepend CBAM channel attention
    channel_ratio    : SE bottleneck ratio (ignored if use_channel_attn=False)
    kernel_size      : spatial conv kernel size
    """

    def __init__(self,
                 in_channels:      int,
                 use_channel_attn: bool = True,
                 channel_ratio:    int  = 16,
                 kernel_size:      int  = 7):
        super().__init__()
        self.channel_attn = (
            ChannelAttention(in_channels, ratio=channel_ratio)
            if use_channel_attn else nn.Identity()
        )
        self.spatial_attn = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out      : (B, C, G, G)  — attended feature map F'
        attn_map : (B, 1, G, G)  — spatial attention map A in [0, 1]
        """
        x_c = self.channel_attn(x)              # (B, C, G, G)
        out, attn_map = self.spatial_attn(x_c)  # (B, C, G, G), (B, 1, G, G)
        return out, attn_map


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, G = 4, 2048, 7
    x = torch.randn(B, C, G, G)

    module = AttentionModule(in_channels=C, use_channel_attn=True)
    out, attn = module(x)
    print(f"Input:    {tuple(x.shape)}")
    print(f"Output:   {tuple(out.shape)}")
    print(f"Attn map: {tuple(attn.shape)}  "
          f"min={attn.min():.3f}  max={attn.max():.3f}")
    assert attn.min() >= 0.0 and attn.max() <= 1.0, "Attention not in [0,1]"
    print("✓ Attention module smoke-test passed.")
