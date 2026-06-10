"""
app.py — Streamlit demo for Explanation-Supervised Attention
          on NIH ChestX-ray14

Usage:
    pip install streamlit Pillow torch torchvision
    streamlit run app.py

Features:
    • Upload a chest X-ray (or pick a sample)
    • Select backbone + which checkpoint to load
    • See 14-class probability bars
    • Side-by-side: Input | GT box (if available) | Ours (supervised attn) | Grad-CAM
    • Metrics summary (loaded from eval_results.json if it exists)
"""

import os
import sys
import json
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.splits import CLASS_NAMES
from src.data.masks  import load_bbox_lookup, get_mask_for_image
from src.models.model import build_model
from src.gradcam     import GradCAM, get_gradcam_layer

# ── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
ATTN_CMAP     = LinearSegmentedColormap.from_list(
    "attn", ["#000080", "#00FF00", "#FFFF00", "#FF0000"]
)
CHECKPOINT_DIR = os.path.join(ROOT, "outputs", "checkpoints")
EVAL_RESULTS   = os.path.join(ROOT, "outputs", "logs", "eval_results.json")
FIGURES_DIR    = os.path.join(ROOT, "outputs", "figures")
SAMPLE_DIR     = os.path.join(ROOT, "data", "sample")

BACKBONE_OPTIONS = {
    "ResNet-50":        "resnet50",
    "DenseNet-121":     "densenet121",
    "EfficientNet-B0":  "efficientnet_b0",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explanation-Supervised Attention — ChestX-ray14",
    page_icon="🫁",
    layout="wide",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(backbone_name: str, ckpt_path: str, variant: bool):
    cfg = {
        "backbone": backbone_name, "pretrained": False,
        "num_classes": 14, "use_channel_attn": True,
    }
    model = build_model(cfg, cooc_matrix=None, variant=variant)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        loaded = True
    else:
        loaded = False
    model.eval()
    return model, loaded


@st.cache_data(show_spinner=False)
def load_bbox_lookup_cached(data_dir: str):
    csv = os.path.join(data_dir, "BBox_List_2017.csv")
    if os.path.exists(csv):
        return load_bbox_lookup(csv)
    return {}


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised tensor back to uint8 numpy image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.squeeze(0) * std + mean
    img  = img.permute(1, 2, 0).clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)


def upsample_map(heatmap: np.ndarray, size: int = 224) -> np.ndarray:
    t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def overlay_heatmap(ax, img_np, heatmap, alpha=0.45, title=""):
    ax.imshow(img_np)
    ax.imshow(heatmap, cmap=ATTN_CMAP, alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def render_gt_box(ax, img_np, gt_mask_7x7, title="Ground-truth box"):
    ax.imshow(img_np)
    G    = gt_mask_7x7.shape[0]
    cell = img_np.shape[1] // G
    for gy in range(G):
        for gx in range(G):
            if gt_mask_7x7[gy, gx] > 0:
                rect = mpatches.Rectangle(
                    (gx * cell, gy * cell), cell, cell,
                    linewidth=0, facecolor="lime", alpha=0.45,
                )
                ax.add_patch(rect)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/NIH_logo.svg/200px-NIH_logo.svg.png",
             width=80)
    st.title("Settings")

    backbone_label = st.selectbox("Backbone", list(BACKBONE_OPTIONS.keys()), index=0)
    backbone_name  = BACKBONE_OPTIONS[backbone_label]

    model_type = st.radio(
        "Model type",
        ["Attention Variant (ours)", "Baseline"],
        index=0,
    )
    variant = (model_type == "Attention Variant (ours)")
    tag     = f"{backbone_name}_{'attention' if variant else 'baseline'}_best.pt"
    ckpt    = os.path.join(CHECKPOINT_DIR, tag)

    ckpt_exists = os.path.exists(ckpt)
    if ckpt_exists:
        st.success(f"✓ Checkpoint loaded: `{tag}`")
    else:
        st.warning(f"⚠ No checkpoint found at `{tag}`. Using random weights (for demo only).")

    thresh = st.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.05)
    alpha  = st.slider("Heatmap opacity", 0.2, 0.8, 0.45, 0.05)

    st.markdown("---")
    st.caption("**Explanation-Supervised Attention**\nCSCD 618 / DSCD 604\nIsrael Agyekum")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🫁 Explanation-Supervised Attention — ChestX-ray14")
st.markdown(
    "Upload a chest X-ray to see **14-class predictions** and compare "
    "our **supervised attention map** against vanilla **Grad-CAM**."
)

# ── Load model ────────────────────────────────────────────────────────────────
model, ckpt_loaded = load_model_cached(backbone_name, ckpt, variant)
bbox_lookup = load_bbox_lookup_cached(ROOT)

# ── Image input ───────────────────────────────────────────────────────────────
col_upload, col_sample = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload a chest X-ray PNG/JPG", type=["png", "jpg", "jpeg"]
    )
with col_sample:
    sample_files = []
    if os.path.exists(SAMPLE_DIR):
        sample_files = [f for f in os.listdir(SAMPLE_DIR)
                        if f.lower().endswith((".png", ".jpg"))]
    sample_choice = st.selectbox(
        "…or pick a sample image",
        ["(none)"] + sample_files,
    )

pil_img = None
image_id = None

if uploaded:
    pil_img  = Image.open(uploaded).convert("RGB")
    image_id = uploaded.name
elif sample_choice != "(none)":
    pil_img  = Image.open(os.path.join(SAMPLE_DIR, sample_choice)).convert("RGB")
    image_id = sample_choice

# ── Inference ─────────────────────────────────────────────────────────────────
if pil_img is not None:
    x       = preprocess_image(pil_img)
    img_224 = np.array(pil_img.resize((224, 224)))

    with torch.no_grad():
        logits, attn_map = model(x)

    probs   = torch.sigmoid(logits).squeeze().numpy()          # (14,)
    attn_np = attn_map.squeeze().numpy()                       # (7, 7)
    attn_up = upsample_map(attn_np, 224)                       # (224, 224)

    # Grad-CAM (always from baseline for fair comparison)
    base_cfg = {"backbone": backbone_name, "pretrained": False,
                "num_classes": 14, "use_channel_attn": True}
    base_ckpt = os.path.join(CHECKPOINT_DIR,
                             f"{backbone_name}_baseline_best.pt")
    base_model, _ = load_model_cached(backbone_name, base_ckpt, False)
    layer  = get_gradcam_layer(backbone_name)
    gcam   = GradCAM(base_model, layer)
    top_cls = int(probs.argmax())
    gcam_np = gcam(x, class_idx=top_cls, output_size=(224, 224)).squeeze()
    gcam.remove_hooks()

    # GT mask (if image has a bounding box)
    gt_mask_7, has_box = get_mask_for_image(image_id or "", bbox_lookup, 7)

    # ── Layout: heatmaps ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Visualisation")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")

    # 1. Input
    axes[0].imshow(img_224)
    axes[0].set_title("Input X-ray", color="white", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. GT box (if available)
    if has_box:
        render_gt_box(axes[1], img_224, gt_mask_7, "Ground-truth box")
    else:
        axes[1].imshow(img_224)
        axes[1].text(112, 112, "No GT box\nfor this image",
                     ha="center", va="center", color="white", fontsize=10,
                     bbox=dict(boxstyle="round", fc="#333", ec="none"))
        axes[1].set_title("Ground-truth box", color="white", fontsize=11)
        axes[1].axis("off")

    # 3. Supervised attention (ours)
    overlay_heatmap(axes[2], img_224, attn_up, alpha=alpha,
                    title="Ours — supervised attention")

    # 4. Grad-CAM
    overlay_heatmap(axes[3], img_224, gcam_np, alpha=alpha,
                    title="Grad-CAM (baseline)")

    for ax in axes:
        ax.title.set_color("white")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Prediction bars ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("14-Class Predictions")

    cols = st.columns(2)
    predictions = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])
    for i, (cls, prob) in enumerate(predictions):
        col = cols[i % 2]
        detected = prob >= thresh
        color    = "#FF4B4B" if detected else "#4B9EFF"
        label    = f"{'🔴' if detected else '⚪'} {cls}"
        col.markdown(
            f"**{label}**",
            unsafe_allow_html=True,
        )
        col.progress(float(prob), text=f"{prob:.1%}")

    # Top prediction highlight
    top_name = CLASS_NAMES[top_cls]
    top_prob = float(probs[top_cls])
    st.info(
        f"**Top prediction:** {top_name} ({top_prob:.1%})"
        + (" ✓ Above threshold" if top_prob >= thresh else " — Below threshold")
    )

    # ── Attention stats ───────────────────────────────────────────────────────
    with st.expander("Attention map details"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Attn min", f"{attn_np.min():.3f}")
        col2.metric("Attn max", f"{attn_np.max():.3f}")
        col3.metric("Attn mean", f"{attn_np.mean():.3f}")

        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))
        im1 = axes2[0].imshow(attn_np, cmap=ATTN_CMAP, vmin=0, vmax=1)
        axes2[0].set_title("Attention map (7×7)"); axes2[0].axis("off")
        plt.colorbar(im1, ax=axes2[0], fraction=0.046)
        im2 = axes2[1].imshow(gcam_np.reshape(
            int(gcam_np.size**0.5), -1) if gcam_np.ndim == 1 else gcam_np.reshape(7, -1)[:7, :7]
            if gcam_np.shape[0] > 7 else gcam_np[:7, :7],
            cmap=ATTN_CMAP, vmin=0, vmax=1)
        axes2[1].set_title("Grad-CAM (7×7)"); axes2[1].axis("off")
        plt.colorbar(im2, ax=axes2[1], fraction=0.046)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ── Metrics summary (from eval_results.json) ─────────────────────────────────
st.markdown("---")
st.subheader("📊 Experiment Results Summary")

if os.path.exists(EVAL_RESULTS):
    with open(EVAL_RESULTS) as f:
        results = json.load(f)

    var_auc  = results.get("variant",  {}).get("macro_auc", None)
    base_auc = results.get("baseline", {}).get("macro_auc", None)
    var_iou  = results.get("variant",  {}).get("localization", {}).get("mean_iou", None)
    gcam_iou = results.get("gradcam_loc", {}).get("mean_iou", None)
    var_pg   = results.get("variant",  {}).get("localization", {}).get("pointing_game_acc", None)
    gcam_pg  = results.get("gradcam_loc", {}).get("pointing_game_acc", None)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variant macro AUC",  f"{var_auc:.4f}"  if var_auc  else "—",
              delta=f"{var_auc - base_auc:+.4f}" if (var_auc and base_auc) else None)
    m2.metric("Baseline macro AUC", f"{base_auc:.4f}" if base_auc else "—")
    m3.metric("Supervised attn IoU", f"{var_iou:.4f}"  if var_iou  else "—",
              delta=f"{var_iou - gcam_iou:+.4f}" if (var_iou and gcam_iou) else None)
    m4.metric("Grad-CAM IoU",        f"{gcam_iou:.4f}" if gcam_iou else "—")

    # Per-class AUC table
    with st.expander("Per-class AUC"):
        import pandas as pd
        var_cls  = results.get("variant",  {}).get("per_class_auc", {})
        base_cls = results.get("baseline", {}).get("per_class_auc", {})
        rows = []
        for cls in CLASS_NAMES:
            rows.append({
                "Class":    cls,
                "Variant":  round(var_cls.get(cls, float("nan")), 4),
                "Baseline": round(base_cls.get(cls, float("nan")), 4),
                "Delta":    round(var_cls.get(cls, 0) - base_cls.get(cls, 0), 4)
                            if cls in var_cls and cls in base_cls else float("nan"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df.style.background_gradient(subset=["Variant", "Baseline"],
                                                    cmap="RdYlGn", vmin=0.4, vmax=1.0),
                     use_container_width=True)

    # Saved figures
    fig_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")] \
                if os.path.exists(FIGURES_DIR) else []
    if fig_files:
        with st.expander(f"Generated figures ({len(fig_files)})"):
            cols = st.columns(2)
            for i, fname in enumerate(sorted(fig_files)):
                cols[i % 2].image(os.path.join(FIGURES_DIR, fname),
                                   caption=fname, use_column_width=True)
else:
    st.info(
        "No evaluation results yet. Run training on Kaggle first, then run:\n\n"
        "```\npython -m src.evaluate "
        "--variant_ckpt outputs/checkpoints/resnet50_attention_best.pt "
        "--baseline_ckpt outputs/checkpoints/resnet50_baseline_best.pt\n```"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Explanation-Supervised Attention for Multi-Label Thoracic Disease Classification** "
    "| CSCD 618 / DSCD 604 | Israel Agyekum"
)
