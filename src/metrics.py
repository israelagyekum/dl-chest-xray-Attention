"""
metrics.py — All evaluation metrics.

Classification:
    per_class_auc(y_true, y_score)   → array (14,)
    macro_auc(y_true, y_score)       → float
    per_class_f1(y_true, y_pred)     → array (14,)
    delong_test(y_true, score_a, score_b) → (z_stat, p_value)

Localisation (against bounding-box annotations):
    compute_iou(attn_map, mask, threshold) → float
    compute_pointing_game(attn_map, mask)  → float (0 or 1 per image)
    batch_localization_metrics(...)        → dict

All functions work on NumPy arrays; call .cpu().numpy() on tensors first.
"""

import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import scipy.stats as stats


# ============================================================================
# Classification metrics
# ============================================================================

def per_class_auc(y_true: np.ndarray,
                  y_score: np.ndarray,
                  class_names: Optional[list] = None) -> dict:
    """
    Per-class ROC-AUC.

    Parameters
    ----------
    y_true  : (N, 14) multi-hot ground truth
    y_score : (N, 14) predicted probabilities

    Returns
    -------
    dict mapping class_name → AUC (skipped if class has no positive samples)
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    aucs = {}
    for i, cls in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            continue   # no positive samples → undefined AUC
        try:
            aucs[cls] = roc_auc_score(y_true[:, i], y_score[:, i])
        except Exception:
            pass
    return aucs


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-average ROC-AUC over classes that have positive samples."""
    valid = [i for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0]
    if len(valid) == 0:
        return float("nan")
    return roc_auc_score(y_true[:, valid], y_score[:, valid], average="macro")


def per_class_f1(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 threshold: float = 0.5,
                 class_names: Optional[list] = None) -> dict:
    """
    Per-class F1 after thresholding probabilities.

    y_pred can be probabilities (will be thresholded) or binary.
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    y_bin = (y_pred >= threshold).astype(int) if y_pred.dtype != bool else y_pred
    results = {}
    for i, cls in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            continue
        results[cls] = f1_score(y_true[:, i], y_bin[:, i], zero_division=0)
    return results


def per_class_precision_recall(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                threshold: float = 0.5,
                                class_names: Optional[list] = None) -> dict:
    """Per-class precision and recall."""
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    y_bin = (y_pred >= threshold).astype(int)
    results = {}
    for i, cls in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            continue
        results[cls] = {
            "precision": precision_score(y_true[:, i], y_bin[:, i], zero_division=0),
            "recall":    recall_score(y_true[:, i], y_bin[:, i], zero_division=0),
        }
    return results


# ============================================================================
# DeLong's test for AUC significance
# ============================================================================

def _structural_components(y_true: np.ndarray,
                            y_score: np.ndarray) -> tuple:
    """
    Compute structural components (V10, V01) for DeLong's variance estimator.
    Follows: DeLong, DeLong & Clarke-Pearson (1988).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.array([]), np.array([])

    # V10[i] = (1/n_neg) * sum_j I(neg[j] < pos[i]) + 0.5*I(neg[j]==pos[i])
    V10 = np.array([
        np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos
    ])
    # V01[j] = (1/n_pos) * sum_i I(pos[i] > neg[j]) + 0.5*I(pos[i]==neg[j])
    V01 = np.array([
        np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg
    ])
    auc = V10.mean()
    return auc, V10, V01


def delong_variance(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Variance of the AUC estimator via DeLong's method."""
    auc, V10, V01 = _structural_components(y_true, y_score)
    n_pos, n_neg = len(V10), len(V01)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    var = (np.var(V10, ddof=1) / n_pos +
           np.var(V01, ddof=1) / n_neg)
    return var


def delong_test(y_true: np.ndarray,
                score_a: np.ndarray,
                score_b: np.ndarray) -> tuple[float, float]:
    """
    DeLong's test for the equality of two AUC values.

    Returns
    -------
    z_stat : float   — test statistic
    p_value: float   — two-sided p-value (H0: AUC_a == AUC_b)
    """
    auc_a, V10_a, V01_a = _structural_components(y_true, score_a)
    auc_b, V10_b, V01_b = _structural_components(y_true, score_b)

    n_pos = len(V10_a)
    n_neg = len(V01_a)

    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # Covariance matrix of [AUC_a, AUC_b]
    s10 = np.cov(V10_a, V10_b, ddof=1) / n_pos  if n_pos > 1 else np.zeros((2,2))
    s01 = np.cov(V01_a, V01_b, ddof=1) / n_neg  if n_neg > 1 else np.zeros((2,2))
    S   = s10 + s01   # (2, 2) covariance matrix

    diff   = auc_a - auc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return float("nan"), float("nan")

    z    = diff / np.sqrt(var_diff)
    p    = 2 * stats.norm.sf(np.abs(z))
    return float(z), float(p)


# ============================================================================
# Localisation metrics
# ============================================================================

def attn_to_binarymask(attn_map: np.ndarray,
                       threshold: float = 0.5) -> np.ndarray:
    """
    Binarise a spatial attention map at a given threshold.

    attn_map : (G, G)  or (H, W) — values in [0, 1]
    Returns  : binary mask (G, G)
    """
    return (attn_map >= threshold).astype(np.uint8)


def compute_iou(pred_mask: np.ndarray,
                gt_mask:   np.ndarray) -> float:
    """
    Intersection-over-Union between predicted and ground-truth binary masks.

    pred_mask, gt_mask : (G, G) binary arrays
    """
    intersection = (pred_mask & gt_mask).sum()
    union        = (pred_mask | gt_mask).sum()
    if union == 0:
        return float("nan")
    return float(intersection) / float(union)


def compute_pointing_game(attn_map: np.ndarray,
                          gt_mask:  np.ndarray) -> int:
    """
    Pointing game: 1 if the pixel with the highest attention is inside
    the ground-truth bounding-box region, else 0.

    attn_map : (G, G) float map
    gt_mask  : (G, G) binary mask
    """
    peak = np.unravel_index(attn_map.argmax(), attn_map.shape)
    return int(gt_mask[peak] > 0)


def batch_localization_metrics(attn_maps:   np.ndarray,
                                gt_masks:    np.ndarray,
                                has_box_arr: np.ndarray,
                                iou_threshold: float = 0.5
                                ) -> dict:
    """
    Compute mean IoU and pointing-game accuracy over the test set,
    considering only images that have bounding boxes.

    Parameters
    ----------
    attn_maps   : (N, G, G) — attention maps (sigmoid, [0,1])
    gt_masks    : (N, G, G) — binary box masks at grid resolution
    has_box_arr : (N,)      — 0/1 flag

    Returns
    -------
    {
        "mean_iou":           float,
        "pointing_game_acc":  float,
        "n_boxed":            int,
    }
    """
    ious    = []
    pg_hits = []

    for i in range(len(attn_maps)):
        if not has_box_arr[i]:
            continue
        attn = attn_maps[i]   # (G, G)
        gt   = gt_masks[i]    # (G, G)

        pred_bin = attn_to_binarymask(attn, threshold=iou_threshold)
        iou_val  = compute_iou(pred_bin, gt.astype(np.uint8))
        pg_hit   = compute_pointing_game(attn, gt)

        if not np.isnan(iou_val):
            ious.append(iou_val)
        pg_hits.append(pg_hit)

    return {
        "mean_iou":          float(np.mean(ious))    if ious    else float("nan"),
        "pointing_game_acc": float(np.mean(pg_hits)) if pg_hits else float("nan"),
        "n_boxed":           len(pg_hits),
    }


# ============================================================================
# Smoke-test
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, C, G = 200, 14, 7

    y_true  = (rng.random((N, C)) > 0.85).astype(int)
    y_score = np.clip(y_true + rng.normal(0, 0.3, (N, C)), 0, 1)

    class_names = [
        "Atelectasis","Cardiomegaly","Effusion","Infiltration",
        "Mass","Nodule","Pneumonia","Pneumothorax",
        "Consolidation","Edema","Emphysema","Fibrosis",
        "Pleural_Thickening","Hernia"
    ]

    auc_dict = per_class_auc(y_true, y_score, class_names)
    print("Per-class AUC:")
    for cls, auc in auc_dict.items():
        print(f"  {cls:<22}: {auc:.3f}")
    print(f"Macro AUC: {macro_auc(y_true, y_score):.3f}")

    # Simulate two models
    score_b = np.clip(y_true + rng.normal(0, 0.5, (N, C)), 0, 1)
    z, p = delong_test(y_true[:, 0], y_score[:, 0], score_b[:, 0])
    print(f"\nDeLong test (class 0): z={z:.3f}, p={p:.4f}")

    # Localisation
    attn_maps  = rng.uniform(0, 1, (N, G, G)).astype(np.float32)
    gt_masks   = (rng.random((N, G, G)) > 0.7).astype(np.float32)
    has_box    = (rng.random(N) > 0.5).astype(int)
    loc = batch_localization_metrics(attn_maps, gt_masks, has_box)
    print(f"\nLocalization: {loc}")
    print("✓ Metrics smoke-test passed.")
