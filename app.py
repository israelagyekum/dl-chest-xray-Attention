"""
app.py — Streamlit demo for Explanation-Supervised Attention
          on NIH ChestX-ray14

Usage (local):
    pip install -r requirements.txt
    streamlit run app.py

Streamlit Cloud:
    Add DRIVE_FOLDER_ID to app secrets (Settings → Secrets).
    The app downloads checkpoints from your shared Google Drive folder
    on first launch automatically.
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
import src.gradcam as _gcam

# Fix DenseNet Grad-CAM layer (norm5 has in-place op conflicts with backward hooks)
_gcam.GRADCAM_LAYERS['densenet121'] = 'backbone.features.denseblock4'

# ── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)
ATTN_CMAP      = LinearSegmentedColormap.from_list(
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

CHECKPOINT_FILES = [
    f"{bb}_{var}_best.pt"
    for bb in ["resnet50", "densenet121", "efficientnet_b0"]
    for var in ["attention", "baseline"]
]


# ── Google Drive checkpoint download ─────────────────────────────────────────

def _get_drive_folder_id() -> str:
    """Read Drive folder ID from Streamlit secrets or env var."""
    try:
        return st.secrets.get("DRIVE_FOLDER_ID", "")
    except Exception:
        return os.environ.get("DRIVE_FOLDER_ID", "")


@st.cache_resource(show_spinner=False)
def ensure_checkpoints():
    """Download missing checkpoints from Google Drive (runs once per session)."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    missing = [f for f in CHECKPOINT_FILES
               if not os.path.exists(os.path.join(CHECKPOINT_DIR, f))]
    if not missing:
        return True

    folder_id = _get_drive_folder_id()
    if not folder_id:
        st.warning(
            "⚠ Model checkpoints not found locally and no `DRIVE_FOLDER_ID` secret set. "
            "The app will run with **random weights** (predictions are meaningless). "
            "Add your Google Drive folder ID to Streamlit secrets to load real models."
        )
        return False

    try:
        import gdown
        with st.spinner(
            f"⬇ Downloading {len(missing)} model checkpoint(s) from Google Drive "
            "(first launch only — takes ~2 min)…"
        ):
            gdown.download_folder(
                id=folder_id,
                output=CHECKPOINT_DIR,
                quiet=True,
                use_cookies=False,
            )
        st.success("✓ Checkpoints downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Download failed: {e}\nApp will run with random weights.")
        return False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explanation-Supervised Attention — ChestX-ray14",
    page_icon="🫁",
    layout="wide",
)

# ── Custom styling (card-style metrics/expanders, matching brand palette) ─────
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(46, 38, 32, 0.07);
        border: 1px solid #F5E6D3;
    }
    div[data-testid="stMetric"] label { color: #A08B76 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #F0693C; }
    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        border: 1px solid #F5E6D3;
    }
    div[data-testid="stFileUploader"], div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #F5E6D3;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        background-color: #F0693C;
        color: #FFFFFF;
        border: none;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #E8437B;
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; }
    div[data-testid="stSidebarNav"] { background-color: #FFFFFF; }
    .stTabs [aria-selected="true"] {
        background-color: #E8437B !important;
        color: #FFFFFF !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Download checkpoints on startup
ensure_checkpoints()


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(backbone_name: str, ckpt_path: str, variant: bool):
    cfg = {
        "backbone": backbone_name, "pretrained": False,
        "num_classes": 14, "use_channel_attn": True,
    }
    model = build_model(cfg, cooc_matrix=None, variant=variant)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
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
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/NIH_logo.svg/200px-NIH_logo.svg.png",
        width=80,
    )
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
        st.warning(f"⚠ No checkpoint: `{tag}` — random weights.")

    thresh = st.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.05)
    alpha  = st.slider("Heatmap opacity",       0.2, 0.8, 0.45, 0.05)

    st.markdown("---")
    st.caption(
        "**Explanation-Supervised Attention**\nCSCD 618 / DSCD 604\n"
        "Israel Agyekum · Joel Dadi-Klutse · Eric Okyere"
    )


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
        sample_files = [
            f for f in sorted(os.listdir(SAMPLE_DIR))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and os.path.getsize(os.path.join(SAMPLE_DIR, f)) > 0
        ]
    # Default to first sample so XAI section is visible on page load
    default_idx = 1 if sample_files else 0
    sample_choice = st.selectbox(
        "…or pick a sample image",
        ["(none)"] + sample_files,
        index=default_idx,
    )

pil_img  = None
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

    probs   = torch.sigmoid(logits).squeeze().numpy()
    attn_np = attn_map.squeeze().numpy()
    attn_up = upsample_map(attn_np, 224)

    # Grad-CAM (always from baseline for fair comparison)
    base_ckpt  = os.path.join(CHECKPOINT_DIR, f"{backbone_name}_baseline_best.pt")
    base_model, _ = load_model_cached(backbone_name, base_ckpt, False)
    layer      = get_gradcam_layer(backbone_name)
    gcam       = GradCAM(base_model, layer)
    top_cls    = int(probs.argmax())
    gcam_np    = gcam(x, class_idx=top_cls, output_size=(224, 224)).squeeze()
    gcam.remove_hooks()

    # GT mask
    gt_mask_7, has_box = get_mask_for_image(image_id or "", bbox_lookup, 7)

    # ── Visualisation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Visualisation")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")

    axes[0].imshow(img_224)
    axes[0].set_title("Input X-ray", color="#2E2620", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    if has_box:
        render_gt_box(axes[1], img_224, gt_mask_7, "Ground-truth box")
    else:
        axes[1].imshow(img_224)
        axes[1].text(112, 112, "No GT box\nfor this image",
                     ha="center", va="center", color="white", fontsize=10,
                     bbox=dict(boxstyle="round", fc="#E8437B", ec="none"))
        axes[1].set_title("Ground-truth box", color="#2E2620", fontsize=11)
        axes[1].axis("off")

    overlay_heatmap(axes[2], img_224, attn_up, alpha=alpha,
                    title="Ours — supervised attention")
    overlay_heatmap(axes[3], img_224, gcam_np, alpha=alpha,
                    title="Grad-CAM (baseline)")

    for ax in axes:
        ax.title.set_color("#2E2620")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Prediction bars ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("14-Class Predictions")

    cols = st.columns(2)
    predictions = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])
    for i, (cls, prob) in enumerate(predictions):
        col     = cols[i % 2]
        detected = prob >= thresh
        label   = f"{'🔴' if detected else '⚪'} {cls}"
        col.markdown(f"**{label}**", unsafe_allow_html=True)
        col.progress(float(prob), text=f"{prob:.1%}")

    top_name = CLASS_NAMES[top_cls]
    top_prob = float(probs[top_cls])
    st.info(
        f"**Top prediction:** {top_name} ({top_prob:.1%})"
        + (" ✓ Above threshold" if top_prob >= thresh else " — Below threshold")
    )

    with st.expander("Attention map details"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Attn min",  f"{attn_np.min():.3f}")
        col2.metric("Attn max",  f"{attn_np.max():.3f}")
        col3.metric("Attn mean", f"{attn_np.mean():.3f}")

        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))
        im1 = axes2[0].imshow(attn_np, cmap=ATTN_CMAP, vmin=0, vmax=1)
        axes2[0].set_title("Attention map (7×7)")
        axes2[0].axis("off")
        plt.colorbar(im1, ax=axes2[0], fraction=0.046)
        gcam_small = gcam_np[:7, :7] if gcam_np.shape[0] >= 7 else gcam_np
        im2 = axes2[1].imshow(gcam_small, cmap=ATTN_CMAP, vmin=0, vmax=1)
        axes2[1].set_title("Grad-CAM (7×7)")
        axes2[1].axis("off")
        plt.colorbar(im2, ax=axes2[1], fraction=0.046)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ── Metrics summary ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Experiment Results Summary")

if os.path.exists(EVAL_RESULTS):
    with open(EVAL_RESULTS) as f:
        results = json.load(f)

    var_auc  = results.get("variant",  {}).get("macro_auc", None)
    base_auc = results.get("baseline", {}).get("macro_auc", None)
    var_iou  = results.get("variant",  {}).get("localization", {}).get("mean_iou", None)
    gcam_iou = results.get("gradcam_loc", {}).get("mean_iou", None)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variant macro AUC",   f"{var_auc:.4f}"  if var_auc  else "—",
              delta=f"{var_auc - base_auc:+.4f}" if (var_auc and base_auc) else None)
    m2.metric("Baseline macro AUC",  f"{base_auc:.4f}" if base_auc else "—")
    m3.metric("Supervised attn IoU", f"{var_iou:.4f}"  if var_iou  else "—",
              delta=f"{var_iou - gcam_iou:+.4f}" if (var_iou and gcam_iou) else None)
    m4.metric("Grad-CAM IoU",        f"{gcam_iou:.4f}" if gcam_iou else "—")

    with st.expander("Per-class AUC"):
        import pandas as pd
        var_cls  = results.get("variant",  {}).get("per_class_auc", {})
        base_cls = results.get("baseline", {}).get("per_class_auc", {})
        rows = [
            {
                "Class":    cls,
                "Variant":  round(var_cls.get(cls, float("nan")), 4),
                "Baseline": round(base_cls.get(cls, float("nan")), 4),
                "Delta":    round(var_cls.get(cls, 0) - base_cls.get(cls, 0), 4)
                            if cls in var_cls and cls in base_cls else float("nan"),
            }
            for cls in CLASS_NAMES
        ]
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.background_gradient(subset=["Variant", "Baseline"],
                                         cmap="RdYlGn", vmin=0.4, vmax=1.0),
            use_container_width=True,
        )

    fig_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")] \
                if os.path.exists(FIGURES_DIR) else []
    if fig_files:
        with st.expander(f"Generated figures ({len(fig_files)})"):
            cols = st.columns(2)
            for i, fname in enumerate(sorted(fig_files)):
                cols[i % 2].image(os.path.join(FIGURES_DIR, fname),
                                   caption=fname, use_container_width=True)
else:
    st.info(
        "No evaluation results file found (`outputs/logs/eval_results.json`). "
        "Run evaluation after training to populate this section."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Explanation-Supervised Attention for Multi-Label Thoracic Disease Classification** "
    "| CSCD 618 / DSCD 604 | Israel Agyekum · Joel Dadi-Klutse · Eric Okyere"
)
