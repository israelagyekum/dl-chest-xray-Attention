"""
plots.py — All visualisations for the project.

Functions:
    plot_training_curves(log_path, save_dir)
    plot_roc_curves(y_true, y_score, class_names, save_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir)
    plot_heatmap_overlay(image, attn_map, gt_mask, gradcam_map, save_path)
    plot_localization_comparison(results_json, save_dir)
    plot_cooccurrence_matrix(cooc, class_names, save_dir)

All figures saved as PNG at 150 dpi to the specified directory.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns
from typing import Optional


# ============================================================================
# Training curves
# ============================================================================

def plot_training_curves(log_path: str, save_dir: str):
    """
    Plot loss curves (train_total, val_total, val_AUC) from a JSON log.
    """
    with open(log_path) as f:
        history = json.load(f)

    epochs      = [r["epoch"] for r in history]
    train_total = [r.get("train_total", np.nan) for r in history]
    val_total   = [r.get("val_total",   np.nan) for r in history]
    val_auc     = [r.get("val_macro_auc", np.nan) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs, train_total, label="Train total loss", color="steelblue")
    ax.plot(epochs, val_total,   label="Val total loss",   color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, val_auc, label="Val macro AUC", color="seagreen")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro AUC")
    ax.set_title("Validation Macro AUC")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(log_path))[0]
    out  = os.path.join(save_dir, f"{name}_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Training curves → {out}")


# ============================================================================
# ROC curves
# ============================================================================

def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray,
                    class_names: list, save_dir: str,
                    title: str = "ROC Curves"):
    """Per-label ROC curves on one figure."""
    n = len(class_names)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for i, cls in enumerate(class_names):
        ax = axes[i]
        if y_true[:, i].sum() == 0:
            ax.text(0.5, 0.5, "No positives", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(cls, fontsize=9)
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, lw=1.5, color="steelblue",
                label=f"AUC={auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
        ax.set_title(f"{cls}", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{title.replace(' ', '_')}_roc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] ROC curves → {out}")


# ============================================================================
# Confusion matrix
# ============================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list, save_dir: str,
                           threshold: float = 0.5, title: str = "Confusion"):
    """Multi-label confusion matrix (per-class TP/FP/TN/FN in a grid)."""
    y_bin = (y_pred >= threshold).astype(int)
    n     = len(class_names)
    tps, fps, tns, fns = [], [], [], []
    for i in range(n):
        cm  = confusion_matrix(y_true[:, i], y_bin[:, i], labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        tps.append(tp); fps.append(fp); tns.append(tn); fns.append(fn)

    x     = np.arange(n)
    width = 0.2
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - 1.5 * width, tps, width, label="TP", color="steelblue")
    ax.bar(x - 0.5 * width, fps, width, label="FP", color="tomato")
    ax.bar(x + 0.5 * width, fns, width, label="FN", color="orange")
    ax.bar(x + 1.5 * width, tns, width, label="TN", color="seagreen")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Count"); ax.set_title(f"{title} — Per-class TP/FP/FN/TN")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{title.replace(' ', '_')}_confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Confusion matrix → {out}")


# ============================================================================
# Heatmap overlay
# ============================================================================

_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "attn_cmap", ["#000080", "#00FF00", "#FFFF00", "#FF0000"]
)


def _overlay(ax, img_np, heatmap, alpha=0.45, cmap=_HEATMAP_CMAP):
    """Helper: draw image + colourised heatmap overlay."""
    h, w = img_np.shape[:2]
    # Upsample heatmap to image resolution
    from PIL import Image as PILImage
    hm_resized = np.array(
        PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), PILImage.BILINEAR
        )
    ) / 255.0

    ax.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
    ax.imshow(hm_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)


def plot_heatmap_overlay(image_np:    np.ndarray,
                          attn_map:   np.ndarray,
                          gt_mask:    np.ndarray,
                          gradcam_map: Optional[np.ndarray],
                          save_path:  str,
                          image_id:   str = ""):
    """
    Side-by-side visualisation:
        [Original] [Ground-truth box] [Ours (supervised attn)] [Grad-CAM]

    Parameters
    ----------
    image_np    : (H, W) or (H, W, 3) uint8 array
    attn_map    : (G, G) float32 in [0, 1]
    gt_mask     : (G, G) binary 0/1
    gradcam_map : (G, G) float32 or None
    """
    n_cols = 4 if gradcam_map is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.5, 3.5))

    # Original
    axes[0].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
    axes[0].set_title("Input X-ray")
    axes[0].axis("off")

    # Ground-truth box
    axes[1].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
    H, W = (image_np.shape[:2])
    G    = gt_mask.shape[0]
    cell = W // G
    for gy in range(G):
        for gx in range(G):
            if gt_mask[gy, gx] > 0:
                rect = mpatches.Rectangle(
                    (gx * cell, gy * cell), cell, cell,
                    linewidth=0, edgecolor=None, facecolor="lime", alpha=0.5,
                )
                axes[1].add_patch(rect)
    axes[1].set_title("Ground-truth box")
    axes[1].axis("off")

    # Supervised attention
    _overlay(axes[2], image_np, attn_map)
    axes[2].set_title("Ours (supervised attn)")
    axes[2].axis("off")

    # Grad-CAM
    if gradcam_map is not None:
        _overlay(axes[3], image_np, gradcam_map)
        axes[3].set_title("Grad-CAM (baseline)")
        axes[3].axis("off")

    plt.suptitle(image_id, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Localisation comparison bar chart
# ============================================================================

def plot_localization_comparison(results: dict, save_dir: str):
    """
    Bar chart: supervised attention vs Grad-CAM on IoU and pointing-game.
    `results` is the dict returned by evaluate.full_evaluation().
    """
    metrics = ["mean_iou", "pointing_game_acc"]
    labels  = ["Mean IoU", "Pointing-Game Acc"]

    var_vals  = [results["variant"]["localization"].get(m, 0)  for m in metrics]
    gcam_vals = [results["gradcam_loc"].get(m, 0)              for m in metrics]
    base_vals = [results["baseline"]["localization"].get(m, 0) for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, var_vals,  width, label="Ours (supervised attn)", color="steelblue")
    ax.bar(x,         gcam_vals, width, label="Grad-CAM (baseline)",     color="tomato")
    ax.bar(x + width, base_vals, width, label="Baseline attn map",        color="seagreen")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.set_title("Localisation: Supervised Attention vs Grad-CAM")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "localization_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Localisation comparison → {out}")


# ============================================================================
# Co-occurrence matrix heatmap
# ============================================================================

def plot_cooccurrence_matrix(cooc: np.ndarray, class_names: list, save_dir: str):
    """Heatmap of the empirical label co-occurrence matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cooc, annot=True, fmt=".2f", xticklabels=class_names,
                yticklabels=class_names, cmap="YlOrRd",
                linewidths=0.3, ax=ax)
    ax.set_title("Empirical Label Co-occurrence  P(col | row)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "cooccurrence_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Co-occurrence matrix → {out}")


# ============================================================================
# AUC comparison table as PNG
# ============================================================================

def plot_auc_comparison_table(results: dict, class_names: list, save_dir: str):
    """
    Horizontal bar chart comparing per-class AUC for variant vs baseline.
    """
    var_auc  = results["variant"]["per_class_auc"]
    base_auc = results["baseline"]["per_class_auc"]

    classes  = [c for c in class_names if c in var_auc and c in base_auc]
    var_vals  = [var_auc[c]  for c in classes]
    base_vals = [base_auc[c] for c in classes]

    y     = np.arange(len(classes))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(y + height / 2, var_vals,  height, label="Attention variant", color="steelblue")
    ax.barh(y - height / 2, base_vals, height, label="Baseline",          color="tomato", alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("ROC-AUC"); ax.set_xlim(0.4, 1.05)
    ax.axvline(0.5, color="gray", linestyle="--", lw=0.8)
    ax.set_title("Per-class AUC: Variant vs Baseline")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "auc_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] AUC comparison → {out}")
