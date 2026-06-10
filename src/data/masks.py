"""
masks.py — Convert NIH ChestX-ray14 bounding boxes → binary attention masks.

Pipeline:
  1. Load BBox_List_2017.csv (boxes in original 1024×1024 pixel space).
  2. Scale box coordinates to 224×224 (image resolution).
  3. Rasterize to a binary mask at the attention grid resolution (7×7 or 14×14).
  4. For images with multiple boxes (same or different findings), take the UNION.

Exported items:
  • load_bbox_lookup(bbox_csv)         → dict[image_id → list[box_dicts]]
  • boxes_to_mask(boxes, grid_size)    → np.ndarray (grid_size × grid_size) float32
  • ORIG_SIZE, IMAGE_SIZE              → constants
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIG_SIZE  = 1024   # original NIH ChestX-ray14 image dimension (square)
IMAGE_SIZE = 224    # we resize to this before feeding the model
SCALE      = IMAGE_SIZE / ORIG_SIZE   # 224/1024 ≈ 0.21875


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_bbox_lookup(bbox_csv: str) -> dict:
    """
    Parse BBox_List_2017.csv and return a dict:
        { image_id: [ {"label": str, "x": f, "y": f, "w": f, "h": f}, ... ] }

    Coordinates are kept in the ORIGINAL (1024×1024) pixel space;
    scaling happens in boxes_to_mask().
    """
    df = pd.read_csv(bbox_csv)
    df.columns = df.columns.str.strip()

    # Normalise column names — the CSV has a quirky multi-part header
    # Expected: Image Index, Finding Label, Bbox [x,y,w,h], (4 more cols for y w h)
    # The actual CSV has: Image Index, Finding Label, Bbox [x,y,w,h], ,, ,
    # We keep only the first 6 useful columns and rename them.
    cols = list(df.columns)
    rename = {
        cols[0]: "image_id",
        cols[1]: "label",
        cols[2]: "x",
        cols[3]: "y",
        cols[4]: "w",
        cols[5]: "h",
    }
    df = df.rename(columns=rename)[list(rename.values())]
    df["image_id"] = df["image_id"].str.strip()

    lookup: dict = {}
    for _, row in df.iterrows():
        img_id = row["image_id"]
        box = {
            "label": row["label"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "w": float(row["w"]),
            "h": float(row["h"]),
        }
        lookup.setdefault(img_id, []).append(box)

    return lookup


def boxes_to_mask(boxes: list[dict],
                  grid_size: int = 7,
                  orig_size: int = ORIG_SIZE,
                  image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Convert a list of bounding boxes (in orig_size pixel space) to a binary
    mask at resolution grid_size × grid_size.

    Steps:
        1. Scale coords from orig_size → image_size.
        2. Create a float mask at image_size × image_size.
        3. Average-pool to grid_size × grid_size (via reshape trick).
        4. Binarize: cell = 1 if any pixel inside it was covered.

    Returns
    -------
    mask : np.ndarray, shape (grid_size, grid_size), dtype float32, values {0,1}
    """
    scale = image_size / orig_size
    # Full-resolution binary canvas
    canvas = np.zeros((image_size, image_size), dtype=np.float32)

    for box in boxes:
        x0 = int(np.clip(box["x"] * scale, 0, image_size - 1))
        y0 = int(np.clip(box["y"] * scale, 0, image_size - 1))
        x1 = int(np.clip((box["x"] + box["w"]) * scale, 0, image_size))
        y1 = int(np.clip((box["y"] + box["h"]) * scale, 0, image_size))
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = 1.0

    # Downsample to grid_size × grid_size by reshaping and taking max per cell
    cell = image_size // grid_size
    if image_size % grid_size != 0:
        # Pad to make it divisible
        pad = grid_size * cell - image_size
        canvas = np.pad(canvas, ((0, pad), (0, pad)), mode="constant")
        padded_size = grid_size * cell
    else:
        padded_size = image_size

    mask = canvas.reshape(grid_size, cell, grid_size, cell).max(axis=(1, 3))
    return mask.astype(np.float32)


def get_mask_for_image(image_id: str,
                       bbox_lookup: dict,
                       grid_size: int = 7) -> tuple[Optional[np.ndarray], bool]:
    """
    Convenience wrapper.

    Returns (mask, has_box):
        mask    — np.ndarray (grid_size, grid_size) if has_box else zeros
        has_box — True if at least one bounding box exists for this image
    """
    if image_id in bbox_lookup:
        boxes = bbox_lookup[image_id]
        mask  = boxes_to_mask(boxes, grid_size=grid_size)
        return mask, True
    else:
        return np.zeros((grid_size, grid_size), dtype=np.float32), False


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, yaml

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    bbox_csv  = os.path.join(cfg["data_dir"], cfg["csv_bbox"])
    grid_size = cfg["attention_resolution"]

    lookup = load_bbox_lookup(bbox_csv)
    print(f"Loaded bbox lookup: {len(lookup)} images with boxes")

    # Inspect the first 5 entries
    for img_id, boxes in list(lookup.items())[:5]:
        mask, has_box = get_mask_for_image(img_id, lookup, grid_size)
        coverage = mask.sum() / (grid_size * grid_size) * 100
        print(f"  {img_id}: {len(boxes)} box(es), "
              f"grid={grid_size}×{grid_size}, "
              f"coverage={coverage:.1f}%")
        print(f"    mask:\n{mask}")
