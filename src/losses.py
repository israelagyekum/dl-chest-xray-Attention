"""
losses.py — All three loss terms for the variant model.

    L = L_cls  +  λ1 · L_attn  +  λ2 · L_corr

L_cls  : multi-label classification loss
         — FocalLoss (default) or class-weighted BCE
L_attn : explanation-supervision loss
         — Dice + β·MSE on boxed images
         — L1 sparsity regulariser on unboxed images
L_corr : label-correlation regulariser (imported from models/correlation.py)

CombinedLoss wraps all three; pass lambda1=0, lambda2=0 to reproduce a
standard baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ============================================================================
# L_cls: Classification losses
# ============================================================================

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for multi-label classification.

    pos_weights : (14,) tensor — per-class (neg_count / pos_count).
    """

    def __init__(self, pos_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weights", pos_weights)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )


class FocalLoss(nn.Module):
    """
    Focal loss for multi-label classification (Lin et al., ICCV 2017).

    Reduces the relative loss for well-classified examples and focuses
    on hard, mis-classified ones — especially helpful for the heavy
    class imbalance in ChestX-ray14.

    gamma         : focusing parameter  (0 → standard BCE)
    pos_weights   : optional (14,) per-class balancing weights
    """

    def __init__(self,
                 gamma:       float = 2.0,
                 pos_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights)
        else:
            self.pos_weights = None

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) — raw scores (pre-sigmoid)
        targets : (B, C) — multi-hot float labels
        """
        # Per-element BCE (without reduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weights if hasattr(self, "pos_weights") else None,
            reduction="none",
        )   # (B, C)

        # p_t = probability of the correct class
        p    = torch.sigmoid(logits)
        p_t  = targets * p + (1 - targets) * (1 - p)

        # Focal weight
        focal_w = (1 - p_t) ** self.gamma

        loss = (focal_w * bce).mean()
        return loss


# ============================================================================
# L_attn: Attention supervision loss
# ============================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice loss between predicted attention map and binary box mask.
    Both tensors are expected to be in [0, 1].
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        pred   : (B, 1, G, G) or (B, G, G)
        target : same shape
        """
        pred   = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class AttentionLoss(nn.Module):
    """
    L_attn supervision for the spatial attention module.

    On boxed images (has_box == 1):
        L_attn = dice_weight · DiceLoss(A, M) + mse_weight · MSE(A, M)

    On unboxed images (has_box == 0):
        L_attn = sparsity_weight · mean(|A|)   (L1 sparsity)

    If all images in a batch are unboxed, only sparsity fires; if all
    are boxed, only alignment fires.  Mixed batches contribute both terms.

    Parameters
    ----------
    dice_weight     : weight for Dice term (on boxed images)
    mse_weight      : weight for MSE term (on boxed images)
    sparsity_weight : weight for L1 sparsity (on unboxed images)
    """

    def __init__(self,
                 dice_weight:     float = 1.0,
                 mse_weight:      float = 0.5,
                 sparsity_weight: float = 0.01):
        super().__init__()
        self.dice_w    = dice_weight
        self.mse_w     = mse_weight
        self.sparse_w  = sparsity_weight
        self.dice_loss = DiceLoss()

    def forward(self,
                attn_map: torch.Tensor,
                mask:     torch.Tensor,
                has_box:  torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        attn_map : (B, 1, G, G)  — sigmoid attention map from model
        mask     : (B, G, G)     — binary box mask (0/1)
        has_box  : (B,)          — int/bool flag per image

        Returns
        -------
        loss : scalar
        """
        has_box = has_box.bool()
        mask_4d = mask.unsqueeze(1)  # (B, 1, G, G)

        # Resize mask to match attn_map resolution (handles 7 vs 14)
        if attn_map.shape[-1] != mask_4d.shape[-1]:
            mask_4d = F.interpolate(
                mask_4d.float(), size=attn_map.shape[-2:], mode="nearest"
            )

        loss = torch.tensor(0.0, device=attn_map.device)

        # ---- Boxed images: alignment supervision -------------------------
        n_boxed = has_box.sum()
        if n_boxed > 0:
            a_box = attn_map[has_box]    # (n_boxed, 1, G, G)
            m_box = mask_4d[has_box]     # (n_boxed, 1, G, G)
            dice  = self.dice_loss(a_box, m_box)
            mse   = F.mse_loss(a_box, m_box)
            loss  = loss + self.dice_w * dice + self.mse_w * mse

        # ---- Unboxed images: sparsity regulariser -------------------------
        n_unboxed = (~has_box).sum()
        if n_unboxed > 0:
            a_unbox = attn_map[~has_box]   # (n_unboxed, 1, G, G)
            sparsity = a_unbox.abs().mean()
            loss     = loss + self.sparse_w * sparsity

        return loss


# ============================================================================
# Combined loss
# ============================================================================

class CombinedLoss(nn.Module):
    """
    L = L_cls  +  λ1 · L_attn  +  λ2 · L_corr

    Parameters
    ----------
    pos_weights  : (14,) class-balance weights for L_cls
    cooc_matrix  : (14, 14) empirical co-occurrence for L_corr
    focal_gamma  : FocalLoss gamma (0 = weighted BCE)
    lambda1      : weight for L_attn
    lambda2      : weight for L_corr
    dice_weight  : Dice term weight inside L_attn
    mse_weight   : MSE term weight inside L_attn
    sparsity_weight : L1 sparsity weight inside L_attn
    corr_threshold  : co-occurrence threshold for L_corr pairs
    """

    def __init__(self,
                 pos_weights:      torch.Tensor,
                 cooc_matrix:      Optional[np.ndarray] = None,
                 focal_gamma:      float = 2.0,
                 lambda1:          float = 1.0,
                 lambda2:          float = 0.5,
                 dice_weight:      float = 1.0,
                 mse_weight:       float = 0.5,
                 sparsity_weight:  float = 0.01,
                 corr_threshold:   float = 0.2):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # L_cls
        self.cls_loss = FocalLoss(gamma=focal_gamma, pos_weights=pos_weights)

        # L_attn
        self.attn_loss = AttentionLoss(
            dice_weight=dice_weight,
            mse_weight=mse_weight,
            sparsity_weight=sparsity_weight,
        )

        # L_corr (imported from correlation module)
        if cooc_matrix is not None and lambda2 > 0:
            from src.models.correlation import CorrelationRegularizer
            self.corr_reg = CorrelationRegularizer(cooc_matrix, threshold=corr_threshold)
        else:
            self.corr_reg = None

    def forward(self,
                logits:   torch.Tensor,
                targets:  torch.Tensor,
                attn_map: torch.Tensor,
                mask:     torch.Tensor,
                has_box:  torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        logits   : (B, 14) raw scores
        targets  : (B, 14) multi-hot labels
        attn_map : (B, 1, G, G) attention map from model
        mask     : (B, G, G) box mask
        has_box  : (B,) box flag

        Returns
        -------
        total_loss : scalar
        loss_dict  : {"cls": ..., "attn": ..., "corr": ..., "total": ...}
        """
        l_cls  = self.cls_loss(logits, targets)

        l_attn = self.attn_loss(attn_map, mask, has_box)

        if self.corr_reg is not None:
            probs  = torch.sigmoid(logits)
            l_corr = self.corr_reg(probs)
        else:
            l_corr = torch.tensor(0.0, device=logits.device)

        total = l_cls + self.lambda1 * l_attn + self.lambda2 * l_corr

        return total, {
            "cls":   l_cls.item(),
            "attn":  l_attn.item(),
            "corr":  l_corr.item(),
            "total": total.item(),
        }


# ============================================================================
# Factory
# ============================================================================

def build_loss(cfg: dict,
               pos_weights: np.ndarray,
               cooc_matrix: Optional[np.ndarray] = None) -> CombinedLoss:
    """Build CombinedLoss from config dict."""
    pw = torch.tensor(pos_weights, dtype=torch.float32)
    return CombinedLoss(
        pos_weights=pw,
        cooc_matrix=cooc_matrix,
        focal_gamma=cfg.get("focal_gamma",      2.0),
        lambda1=cfg.get("lambda1",              1.0),
        lambda2=cfg.get("lambda2",              0.5),
        dice_weight=cfg.get("dice_weight",      1.0),
        mse_weight=cfg.get("mse_weight",        0.5),
        sparsity_weight=cfg.get("sparsity_weight", 0.01),
    )


# ============================================================================
# Smoke-test
# ============================================================================
if __name__ == "__main__":
    import numpy as np

    B, C, G = 4, 14, 7
    logits   = torch.randn(B, C)
    targets  = (torch.rand(B, C) > 0.7).float()
    attn_map = torch.sigmoid(torch.randn(B, 1, G, G))
    mask     = (torch.rand(B, G, G) > 0.5).float()
    has_box  = torch.tensor([1, 0, 1, 0])

    pos_w    = np.ones(C, dtype=np.float32) * 10.0
    cooc     = np.random.rand(C, C).astype(np.float32)
    np.fill_diagonal(cooc, 1.0)

    loss_fn = build_loss(
        {"focal_gamma": 2.0, "lambda1": 1.0, "lambda2": 0.5,
         "dice_weight": 1.0, "mse_weight": 0.5, "sparsity_weight": 0.01},
        pos_w, cooc
    )

    total, ld = loss_fn(logits, targets, attn_map, mask, has_box)
    print(f"cls={ld['cls']:.4f}  attn={ld['attn']:.4f}  "
          f"corr={ld['corr']:.4f}  total={ld['total']:.4f}")
    assert total.item() >= 0, "Total loss must be non-negative"
    print("✓ Loss smoke-test passed.")
