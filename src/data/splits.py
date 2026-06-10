"""
splits.py — Patient-level train/val/test split for NIH ChestX-ray14.

Design guarantees:
  • All images from the SAME patient land in the SAME split.
  • Hard assertion verifies zero patient-ID leakage across splits.
  • Reproducible via random_seed.
  • Optionally builds a class-balanced subset before splitting.
"""

import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_labels(label_str: str) -> list[int]:
    """Convert 'Atelectasis|Effusion' → 14-dim multi-hot list."""
    findings = [f.strip() for f in label_str.split("|")]
    vec = [0] * NUM_CLASSES
    for f in findings:
        if f in CLASS_NAMES:
            vec[CLASS_NAMES.index(f)] = 1
    return vec


def load_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Load Data_Entry_2017.csv, parse multi-hot labels, and attach
    bbox-availability flag from BBox_List_2017.csv.
    """
    entry_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    bbox_path  = os.path.join(data_dir, "BBox_List_2017.csv")

    df = pd.read_csv(entry_path)
    df.columns = df.columns.str.strip()

    # Rename columns for convenience
    df = df.rename(columns={
        "Image Index":    "image_id",
        "Finding Labels": "finding_labels",
        "Patient ID":     "patient_id",
    })

    # Multi-hot label matrix
    label_matrix = np.array([parse_labels(s) for s in df["finding_labels"]], dtype=np.uint8)
    for i, cls in enumerate(CLASS_NAMES):
        df[cls] = label_matrix[:, i]

    # Mark images that have at least one bounding box
    bbox_df  = pd.read_csv(bbox_path)
    bbox_df.columns = bbox_df.columns.str.strip()
    boxed_ids = set(bbox_df["Image Index"].str.strip().values)
    df["has_box"] = df["image_id"].isin(boxed_ids).astype(np.uint8)

    return df


def build_balanced_subset(df: pd.DataFrame, subset_size: int,
                           seed: int = 42) -> pd.DataFrame:
    """
    Draw a class-balanced subset of ~subset_size images.

    Strategy: for each of the 14 classes + 'No Finding', sample
    up to subset_size // 15 images, then de-duplicate.
    """
    rng = np.random.default_rng(seed)
    per_class = subset_size // 15
    selected = set()

    # Ensure all boxed images are kept
    boxed_ids = set(df[df["has_box"] == 1]["image_id"].tolist())
    selected.update(boxed_ids)

    for cls in CLASS_NAMES:
        pool = df[df[cls] == 1]["image_id"].tolist()
        rng.shuffle(pool)
        selected.update(pool[:per_class])

    # Add 'No Finding' images
    nf_pool = df[df["finding_labels"] == "No Finding"]["image_id"].tolist()
    rng.shuffle(nf_pool)
    selected.update(nf_pool[:per_class])

    subset = df[df["image_id"].isin(selected)].copy()
    return subset.reset_index(drop=True)


def patient_level_split(df: pd.DataFrame,
                        val_frac:  float = 0.10,
                        test_frac: float = 0.10,
                        seed:      int   = 42
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test at the PATIENT level.

    Returns (train_df, val_df, test_df).

    Raises AssertionError if any patient ID appears in more than one split.
    """
    patients = df["patient_id"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(np.ceil(n * test_frac))
    n_val  = int(np.ceil(n * val_frac))

    test_patients  = set(patients[:n_test])
    val_patients   = set(patients[n_test: n_test + n_val])
    train_patients = set(patients[n_test + n_val:])

    # ---- Leakage assertion (the critical safety check) ----
    assert test_patients.isdisjoint(val_patients),   "LEAKAGE: test ∩ val"
    assert test_patients.isdisjoint(train_patients), "LEAKAGE: test ∩ train"
    assert val_patients.isdisjoint(train_patients),  "LEAKAGE: val ∩ train"
    assert train_patients | val_patients | test_patients == set(patients), \
        "Split does not cover all patients"

    train_df = df[df["patient_id"].isin(train_patients)].copy().reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_patients)  ].copy().reset_index(drop=True)
    test_df  = df[df["patient_id"].isin(test_patients) ].copy().reset_index(drop=True)

    _verify_no_leakage(train_df, val_df, test_df)

    print(f"[splits] Patients → train:{len(train_patients):,}  "
          f"val:{len(val_patients):,}  test:{len(test_patients):,}")
    print(f"[splits] Images   → train:{len(train_df):,}  "
          f"val:{len(val_df):,}  test:{len(test_df):,}")
    print(f"[splits] Boxed images → "
          f"train:{train_df['has_box'].sum()}  "
          f"val:{val_df['has_box'].sum()}  "
          f"test:{test_df['has_box'].sum()}")

    return train_df, val_df, test_df


def _verify_no_leakage(train_df, val_df, test_df):
    """Second-pass assertion directly on DataFrames."""
    tr_pts = set(train_df["patient_id"])
    va_pts = set(val_df["patient_id"])
    te_pts = set(test_df["patient_id"])

    assert tr_pts.isdisjoint(va_pts), "IMAGE-LEVEL LEAKAGE: train ∩ val"
    assert tr_pts.isdisjoint(te_pts), "IMAGE-LEVEL LEAKAGE: train ∩ test"
    assert va_pts.isdisjoint(te_pts), "IMAGE-LEVEL LEAKAGE: val ∩ test"
    print("[splits] ✓ No patient-level leakage detected.")


def get_class_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-class positive weights for weighted BCE.
    weight_i = (N - n_pos_i) / n_pos_i  (clipped to [0.1, 100]).
    """
    N = len(train_df)
    weights = []
    for cls in CLASS_NAMES:
        n_pos = train_df[cls].sum()
        if n_pos == 0:
            weights.append(1.0)
        else:
            w = (N - n_pos) / n_pos
            weights.append(float(np.clip(w, 0.1, 100.0)))
    return np.array(weights, dtype=np.float32)


def compute_cooccurrence_matrix(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute empirical label co-occurrence matrix P(i,j) = P(label_j=1 | label_i=1).
    Shape: (14, 14).  Diagonal = 1.
    """
    label_matrix = train_df[CLASS_NAMES].values.astype(np.float32)
    n_pos = label_matrix.sum(axis=0) + 1e-8   # per-class positive count
    cooc  = (label_matrix.T @ label_matrix)    # (14, 14) raw co-occurrence
    # Conditional: P(j|i) = cooc[i,j] / n_pos[i]
    cond  = cooc / n_pos[:, None]
    return cond.astype(np.float32)


# ---------------------------------------------------------------------------
# Main: call directly to inspect split stats
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    df = load_dataframe(cfg["data_dir"])
    print(f"Full dataset: {len(df):,} images, {df['patient_id'].nunique():,} patients")

    if cfg.get("subset_size"):
        df = build_balanced_subset(df, cfg["subset_size"], seed=cfg["random_seed"])
        print(f"Balanced subset: {len(df):,} images")

    train_df, val_df, test_df = patient_level_split(
        df,
        val_frac=cfg["val_frac"],
        test_frac=cfg["test_frac"],
        seed=cfg["random_seed"],
    )

    weights = get_class_weights(train_df)
    print("\nClass weights:")
    for cls, w in zip(CLASS_NAMES, weights):
        print(f"  {cls:<22}: {w:.2f}")

    cooc = compute_cooccurrence_matrix(train_df)
    print(f"\nCo-occurrence matrix shape: {cooc.shape}")
