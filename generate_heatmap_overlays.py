"""
Run this as a NEW CELL in your Colab notebook after training is complete.
It loads the ResNet50-Attn checkpoint, picks 6 annotated test images,
and saves a side-by-side figure:
  Col 1: Original chest X-ray
  Col 2: Supervised attention map (ours)
  Col 3: Grad-CAM (baseline)
  Col 4: Ground-truth bounding box overlay

Output: /content/drive/MyDrive/DL_Project_Outputs/figures/heatmap_overlays.png
Download this and add it to report/latex/figures/ then recompile.
"""

import torch, json, cv2, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ── Config ────────────────────────────────────────────────────
DRIVE_OUT   = Path("/content/drive/MyDrive/DL_Project_Outputs")
CKPT_PATH   = DRIVE_OUT / "checkpoints/resnet50_attention_best.pt"
DATASET_PATH = Path("/content/drive/MyDrive/NIH_CXR")   # adjust if different
BBOX_CSV    = Path("/content/BBox_List_2017.csv")
DATA_CSV    = Path("/content/Data_Entry_2017.csv")
SAVE_PATH   = DRIVE_OUT / "figures/heatmap_overlays.png"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Load model ────────────────────────────────────────────────
import sys
sys.path.insert(0, "/content/dl-project-code")   # path to your repo
from src.models.model import build_model

cfg = {
    "backbone": "resnet50", "use_attention": True,
    "use_correlation": True, "num_classes": 14,
    "lambda1": 1.0, "lambda2": 0.5, "grid_size": 7
}
model = build_model(cfg).to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

# ── Grad-CAM helper ───────────────────────────────────────────
class GradCAM:
    def __init__(self, model, layer):
        self.grads, self.acts = None, None
        layer.register_forward_hook(lambda m,i,o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(self, "grads", go[0]))

    def get_map(self, logits, class_idx):
        logits[:, class_idx].sum().backward(retain_graph=True)
        w = self.grads.mean(dim=[2,3], keepdim=True)
        cam = torch.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().detach().cpu().numpy()

target_layer = model.backbone.layer4[-1].conv3 \
    if hasattr(model.backbone, "layer4") else list(model.backbone.children())[-2]
gcam = GradCAM(model, target_layer)

# ── Load bbox CSV ─────────────────────────────────────────────
import pandas as pd
bbox_df = pd.read_csv(BBOX_CSV)
bbox_df.columns = [c.strip() for c in bbox_df.columns]
# Pick one image per finding (up to 6 unique findings with boxes)
findings_seen, rows = set(), []
for _, row in bbox_df.iterrows():
    f = row["Finding Label"].strip()
    if f not in findings_seen and len(rows) < 6:
        findings_seen.add(f)
        rows.append(row)

# ── Build figure ──────────────────────────────────────────────
fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.5 * len(rows)))
fig.suptitle("Supervised Attention vs Grad-CAM vs Ground-Truth Box",
             fontsize=14, fontweight="bold", y=1.01)
col_titles = ["X-Ray", "Supervised Attn (ours)", "Grad-CAM (baseline)", "GT Bounding Box"]
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=11, fontweight="bold")

cmap_attn = plt.cm.Blues
cmap_gcam = plt.cm.Reds

for row_i, row_data in enumerate(rows):
    fname   = row_data["Image Index"].strip()
    finding = row_data["Finding Label"].strip()
    x, y    = float(row_data["Bbox [x"]), float(row_data["y"])
    w, h    = float(row_data["w"]),        float(row_data["h)"])

    # Load image
    img_path = DATASET_PATH / fname
    if not img_path.exists():
        # Try finding it recursively
        matches = list(DATASET_PATH.rglob(fname))
        if not matches: continue
        img_path = matches[0]

    pil_img  = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    inp      = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.enable_grad():
        inp.requires_grad_(True)
        out = model(inp)
        logits = out["logits"] if isinstance(out, dict) else out

    cls_idx = CLASS_NAMES.index(finding) if finding in CLASS_NAMES else 0

    # Supervised attention map
    attn_map = None
    if isinstance(out, dict) and "attn_map" in out:
        attn_map = out["attn_map"].squeeze().detach().cpu().numpy()
    elif hasattr(model, "attn_map"):
        attn_map = model.attn_map.squeeze().detach().cpu().numpy()

    # Grad-CAM map
    model.zero_grad()
    gcam_map = gcam.get_map(logits, cls_idx)

    # Resize maps to original size
    def to_heatmap(arr, size):
        arr = cv2.resize(arr, size)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    img_np = np.array(pil_img)  # H x W x 3

    # Normalise bbox to original coords
    bx, by = int(x), int(y)
    bw, bh = int(w), int(h)

    axs = axes[row_i]

    # Col 0: raw X-ray
    axs[0].imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
    axs[0].set_ylabel(finding, fontsize=9, rotation=90, labelpad=4)

    # Col 1: supervised attention
    if attn_map is not None:
        overlay = to_heatmap(attn_map, (orig_w, orig_h))
        axs[1].imshow(img_np, cmap="gray")
        axs[1].imshow(overlay, cmap=cmap_attn, alpha=0.55, vmin=0, vmax=1)
    else:
        axs[1].imshow(img_np, cmap="gray")
        axs[1].text(0.5, 0.5, "N/A", transform=axs[1].transAxes,
                    ha="center", va="center", fontsize=10)

    # Col 2: Grad-CAM
    gcam_resized = to_heatmap(gcam_map, (orig_w, orig_h))
    axs[2].imshow(img_np, cmap="gray")
    axs[2].imshow(gcam_resized, cmap=cmap_gcam, alpha=0.55, vmin=0, vmax=1)

    # Col 3: GT bounding box
    axs[3].imshow(img_np, cmap="gray")
    rect = patches.Rectangle((bx, by), bw, bh,
                               linewidth=2, edgecolor="lime", facecolor="none")
    axs[3].add_patch(rect)

    for ax in axs:
        ax.axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {SAVE_PATH}")
