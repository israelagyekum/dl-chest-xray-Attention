"""
dataset.py — PyTorch Dataset for NIH ChestX-ray14.

Each __getitem__ returns:
    image     : FloatTensor (3, 224, 224)  — ImageNet-normalised
    label     : FloatTensor (14,)          — multi-hot
    mask      : FloatTensor (G, G)         — attention grid mask (G = 7 or 14)
    has_box   : int (0 or 1)               — whether a bounding box exists

Augmentation (train only):
    • Random horizontal flip
    • Random rotation ±10°
    • Random brightness / contrast jitter
    • Resize to 224 (always)

Smoke-test mode:
    Pass a root_dir pointing to a folder of ≤50 images for CPU testing.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.data.masks import load_bbox_lookup, get_mask_for_image
from src.data.splits import CLASS_NAMES


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(image_size: int = 224, augment: bool = True):
    """Return torchvision transform pipelines."""
    if augment:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ChestXrayDataset(Dataset):
    """
    Parameters
    ----------
    df           : DataFrame from splits.py (image_id, label cols, has_box)
    image_dir    : directory containing the PNG images
    bbox_lookup  : dict from masks.load_bbox_lookup()
    grid_size    : attention map resolution (7 or 14)
    augment      : apply training augmentation
    image_size   : resize target (224)
    """

    def __init__(self,
                 df:          pd.DataFrame,
                 image_dir:   str,
                 bbox_lookup: dict,
                 grid_size:   int  = 7,
                 augment:     bool = True,
                 image_size:  int  = 224):
        self.df          = df.reset_index(drop=True)
        self.image_dir   = image_dir
        self.bbox_lookup = bbox_lookup
        self.grid_size   = grid_size
        self.transform   = get_transforms(image_size, augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_id   = row["image_id"]
        img_path = os.path.join(self.image_dir, img_id)

        # ---- Load image ------------------------------------------------
        img = Image.open(img_path).convert("RGB")   # grayscale → 3-channel
        img = self.transform(img)                   # FloatTensor (3, H, W)

        # ---- Label vector ----------------------------------------------
        label = torch.tensor(
            [row[cls] for cls in CLASS_NAMES], dtype=torch.float32
        )

        # ---- Attention mask --------------------------------------------
        mask, has_box = get_mask_for_image(img_id, self.bbox_lookup,
                                           self.grid_size)
        mask    = torch.tensor(mask, dtype=torch.float32)    # (G, G)
        has_box = int(has_box)

        return img, label, mask, has_box


# ---------------------------------------------------------------------------
# Factory: build all three DataLoaders from config
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict,
                      train_df: pd.DataFrame,
                      val_df:   pd.DataFrame,
                      test_df:  pd.DataFrame
                      ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders using paths from cfg.

    The image directory is resolved as:
        sample_mode=True  → cfg["sample_dir"]
        sample_mode=False → cfg["data_dir"] / "images"  (Kaggle path)
    """
    if cfg.get("sample_mode", True):
        image_dir = cfg["sample_dir"]
    else:
        image_dir = os.path.join(cfg["data_dir"], "images")

    grid_size  = cfg.get("attention_resolution", 7)
    image_size = cfg.get("image_size", 224)
    batch_size = cfg.get("batch_size", 32)
    num_workers= cfg.get("num_workers", 2)

    bbox_csv   = os.path.join(cfg["data_dir"], cfg["csv_bbox"])
    bbox_lookup = load_bbox_lookup(bbox_csv)

    train_ds = ChestXrayDataset(train_df, image_dir, bbox_lookup,
                                grid_size=grid_size, augment=True,
                                image_size=image_size)
    val_ds   = ChestXrayDataset(val_df,   image_dir, bbox_lookup,
                                grid_size=grid_size, augment=False,
                                image_size=image_size)
    test_ds  = ChestXrayDataset(test_df,  image_dir, bbox_lookup,
                                grid_size=grid_size, augment=False,
                                image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[dataset] train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, sys
    from src.data.splits import load_dataframe, patient_level_split, build_balanced_subset

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    df = load_dataframe(cfg["data_dir"])
    if cfg.get("subset_size"):
        df = build_balanced_subset(df, cfg["subset_size"], cfg["random_seed"])

    train_df, val_df, test_df = patient_level_split(
        df, cfg["val_frac"], cfg["test_frac"], cfg["random_seed"]
    )

    if cfg.get("sample_mode"):
        # Use only images present in sample_dir for smoke test
        sample_ids = set(os.listdir(cfg["sample_dir"]))
        train_df = train_df[train_df["image_id"].isin(sample_ids)]
        val_df   = val_df[val_df["image_id"].isin(sample_ids)]
        test_df  = test_df[test_df["image_id"].isin(sample_ids)]
        print(f"[smoke] Restricted to {len(sample_ids)} sample images")

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_df, val_df, test_df
    )

    # Grab one batch and print shapes
    imgs, labels, masks, has_box = next(iter(train_loader))
    print(f"[smoke] image shape : {imgs.shape}")
    print(f"[smoke] label shape : {labels.shape}")
    print(f"[smoke] mask shape  : {masks.shape}")
    print(f"[smoke] has_box     : {has_box}")
    print("[smoke] ✓ Dataset smoke-test passed.")
