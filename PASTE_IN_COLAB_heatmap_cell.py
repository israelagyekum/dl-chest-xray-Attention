# ============================================================
# PASTE THIS AS A NEW CELL — generates heatmap_overlays.png
# ============================================================
import os, sys, cv2, torch, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

IMAGE_DIR = "/content/nih-chest-xrays/images"
CODE_DIR  = "/content/dl-project/dl-project-code"
DRIVE_OUT = Path("/content/drive/MyDrive/DL_Project_Outputs")
CKPT_PATH = DRIVE_OUT / "checkpoints/resnet50_attention_best.pt"
BBOX_CSV  = Path("/content/nih-chest-xrays/BBox_List_2017.csv")
SAVE_PATH = DRIVE_OUT / "figures/heatmap_overlays.png"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]
TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Fix for PyTorch 2.6 ───────────────────────────────────────
try:
    import numpy as _np
    torch.serialization.add_safe_globals([_np.core.multiarray.scalar])
except Exception:
    pass

# ── Load model ────────────────────────────────────────────────
sys.path.insert(0, CODE_DIR)
from src.models.model import build_model

model = build_model({
    "backbone": "resnet50", "use_attention": True,
    "use_correlation": True, "num_classes": 14, "grid_size": 7
}).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()
print("Model loaded on", DEVICE)

# ── Grad-CAM hooks ────────────────────────────────────────────
# backbone.features is a Sequential — hook the last block
grads, acts = {}, {}
target_layer = model.backbone.features[-1]
print("Target layer:", type(target_layer).__name__)

target_layer.register_forward_hook(
    lambda m, i, o: acts.update({"v": o.detach()}))
target_layer.register_full_backward_hook(
    lambda m, gi, go: grads.update({"v": go[0].detach()}))

def get_gradcam(inp, cls_idx):
    model.zero_grad()
    out = model(inp)
    if isinstance(out, dict):
        logits = out["logits"]
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out
    logits[0, cls_idx].backward(retain_graph=True)
    g = grads["v"]
    a = acts["v"]
    if g.dim() == 4: g = g.squeeze(0)
    if a.dim() == 4: a = a.squeeze(0)
    w   = g.mean(dim=[1, 2], keepdim=True)
    cam = torch.relu((w * a).sum(0)).cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    mn, mx = cam.min(), cam.max()
    return (cam - mn) / (mx - mn + 1e-8)

def get_attn(out):
    if isinstance(out, dict) and "attn_map" in out:
        a = out["attn_map"].squeeze().detach().cpu().numpy()
    elif isinstance(out, (tuple, list)) and len(out) > 1:
        a = out[1].squeeze().detach().cpu().numpy()
    else:
        return None
    a = cv2.resize(a, (224, 224))
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8)

# ── Load bbox CSV ─────────────────────────────────────────────
bbox_df = pd.read_csv(BBOX_CSV)
bbox_df.columns = [c.strip() for c in bbox_df.columns]
img_col, finding_col = bbox_df.columns[0], bbox_df.columns[1]
x_col, y_col, w_col, h_col = (bbox_df.columns[2], bbox_df.columns[3],
                               bbox_df.columns[4], bbox_df.columns[5])

# ── Pick up to 6 images (one per unique finding) ──────────────
seen, rows = set(), []
for _, row in bbox_df.iterrows():
    finding  = str(row[finding_col]).strip()
    fname    = str(row[img_col]).strip()
    img_path = Path(IMAGE_DIR) / fname
    if not img_path.exists():
        continue
    if finding not in seen and len(rows) < 6:
        seen.add(finding)
        rows.append({
            "path": img_path, "finding": finding,
            "bx": float(row[x_col]), "by": float(row[y_col]),
            "bw": float(row[w_col]), "bh": float(row[h_col])
        })
print(f"Found {len(rows)} images: {[r['finding'] for r in rows]}")

# ── Build 6×4 figure ──────────────────────────────────────────
n = len(rows)
fig, axes = plt.subplots(n, 4, figsize=(15, 3.6 * n))
if n == 1:
    axes = axes[np.newaxis, :]

for ax, title in zip(axes[0], ["Chest X-Ray", "Supervised Attn (ours)",
                                 "Grad-CAM (baseline)", "Ground-Truth Box"]):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

for ri, r in enumerate(rows):
    pil   = Image.open(r["path"]).convert("RGB")
    ow, oh = pil.size
    inp   = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    imgr  = np.array(pil.resize((224, 224)))

    bx = r["bx"] * 224 / ow;  by = r["by"] * 224 / oh
    bw = r["bw"] * 224 / ow;  bh = r["bh"] * 224 / oh
    cls_idx = CLASS_NAMES.index(r["finding"]) if r["finding"] in CLASS_NAMES else 0

    with torch.enable_grad():
        inp2 = inp.clone().detach().requires_grad_(True)
        out  = model(inp2)

    attn = get_attn(out)
    gcam = get_gradcam(inp2, cls_idx)

    axs = axes[ri]
    axs[0].imshow(imgr, cmap="gray")
    axs[0].set_ylabel(r["finding"], fontsize=9, rotation=90, va="center", labelpad=4)

    axs[1].imshow(imgr, cmap="gray")
    if attn is not None:
        axs[1].imshow(attn, cmap="Blues", alpha=0.55)
    else:
        axs[1].text(0.5, 0.5, "Attn N/A", transform=axs[1].transAxes,
                    ha="center", va="center", color="red", fontsize=9)

    axs[2].imshow(imgr, cmap="gray")
    axs[2].imshow(gcam, cmap="Reds", alpha=0.55)

    axs[3].imshow(imgr, cmap="gray")
    axs[3].add_patch(patches.Rectangle(
        (bx, by), bw, bh, linewidth=2.5, edgecolor="lime", facecolor="none"))

    for ax in axs:
        ax.axis("off")

plt.suptitle("Supervised Attention vs Grad-CAM vs Ground-Truth Bounding Box",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved to: {SAVE_PATH}")
print("Download from Drive and upload here so I can add it to the report.")
