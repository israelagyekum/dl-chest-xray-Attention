"""
evaluate.py — Full evaluation on the test set.

Computes:
    1. Classification: per-class AUC, macro AUC, F1, precision, recall
    2. Localisation:   mean IoU, pointing-game accuracy (supervised attn vs Grad-CAM)
    3. DeLong's test:  statistical significance of AUC difference vs baseline

Usage:
    python -m src.evaluate \
        --config config.yaml \
        --variant_ckpt  outputs/checkpoints/resnet50_attention_best.pt \
        --baseline_ckpt outputs/checkpoints/resnet50_baseline_best.pt \
        --backbone resnet50

Outputs a JSON results file to outputs/logs/eval_results.json.
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast

from src.data.splits  import (load_dataframe, build_balanced_subset,
                               patient_level_split, get_class_weights,
                               compute_cooccurrence_matrix, CLASS_NAMES)
from src.data.dataset import build_dataloaders
from src.models.model import build_model
from src.losses       import build_loss
from src.gradcam      import compute_gradcam_batch
from src.metrics      import (per_class_auc, macro_auc, per_class_f1,
                               per_class_precision_recall,
                               batch_localization_metrics, delong_test)


# ============================================================================
# Inference pass
# ============================================================================

@torch.no_grad()
def run_inference(model, loader, device, cfg):
    """
    Run model over loader.

    Returns
    -------
    all_probs   : (N, 14) sigmoid probabilities
    all_labels  : (N, 14) multi-hot ground truth
    all_attns   : (N, G, G) spatial attention maps (float32)
    all_masks   : (N, G, G) bounding-box masks
    all_has_box : (N,) int
    """
    model.eval()
    use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"
    G       = cfg.get("attention_resolution", 7)

    all_probs, all_labels, all_attns = [], [], []
    all_masks, all_has_box = [], []

    for imgs, labels, masks, has_box in loader:
        imgs = imgs.to(device)
        with autocast(enabled=use_amp):
            logits, attn_map = model(imgs)

        probs = torch.sigmoid(logits).cpu().numpy()
        attn  = attn_map.squeeze(1).cpu().numpy()   # (B, G, G)

        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_attns.append(attn)
        all_masks.append(masks.numpy())
        all_has_box.append(has_box.numpy())

    return (
        np.vstack(all_probs),
        np.vstack(all_labels),
        np.vstack(all_attns),
        np.vstack(all_masks),
        np.concatenate(all_has_box),
    )


# ============================================================================
# Evaluate one model
# ============================================================================

def evaluate_model(model, loader, device, cfg, model_name="model"):
    """
    Full evaluation of a single model.
    Returns dict of classification + localisation metrics.
    """
    print(f"\n[evaluate] Running inference for '{model_name}'...")
    probs, labels, attns, masks, has_box = run_inference(model, loader, device, cfg)

    # --- Classification ---------------------------------------------------
    cls_auc  = per_class_auc(labels, probs, CLASS_NAMES)
    mac_auc  = macro_auc(labels, probs)
    cls_f1   = per_class_f1(labels, probs, threshold=0.5, class_names=CLASS_NAMES)
    cls_pr   = per_class_precision_recall(labels, probs, class_names=CLASS_NAMES)

    print(f"[{model_name}] Macro AUC = {mac_auc:.4f}")
    print(f"[{model_name}] Per-class AUC:")
    for cls, auc in cls_auc.items():
        print(f"    {cls:<22}: {auc:.3f}")

    # --- Localisation (supervised attention) -------------------------------
    loc = batch_localization_metrics(attns, masks, has_box, iou_threshold=0.5)
    print(f"[{model_name}] Localisation (supervised attn): {loc}")

    return {
        "model_name":   model_name,
        "macro_auc":    mac_auc,
        "per_class_auc": cls_auc,
        "per_class_f1":  cls_f1,
        "per_class_pr":  cls_pr,
        "localization":  loc,
        # raw arrays stored for DeLong comparison
        "_probs":  probs,
        "_labels": labels,
        "_attns":  attns,
        "_masks":  masks,
        "_has_box": has_box,
    }


# ============================================================================
# Full comparative evaluation
# ============================================================================

def full_evaluation(cfg:           dict,
                    variant_ckpt:  str,
                    baseline_ckpt: str,
                    backbone:      str = "resnet50"):
    """
    1. Load variant and baseline checkpoints.
    2. Run inference for both on the test set.
    3. Run Grad-CAM on the baseline.
    4. Compute DeLong's test: variant vs baseline AUC.
    5. Compare localisation: supervised attention vs Grad-CAM.
    6. Save all results to JSON.
    """
    device = torch.device(cfg.get("device", "cpu"))
    cfg["backbone"] = backbone

    # ---- Data ------------------------------------------------------------
    df = load_dataframe(cfg["data_dir"])
    if not cfg.get("sample_mode", True) and cfg.get("subset_size"):
        df = build_balanced_subset(df, cfg["subset_size"], cfg["random_seed"])

    train_df, val_df, test_df = patient_level_split(
        df, cfg["val_frac"], cfg["test_frac"], cfg["random_seed"]
    )

    if cfg.get("sample_mode", True):
        sample_ids = set(os.listdir(cfg["sample_dir"]))
        test_df  = test_df[test_df["image_id"].isin(sample_ids)].reset_index(drop=True)

    pos_weights = get_class_weights(train_df)
    cooc        = compute_cooccurrence_matrix(train_df)

    _, _, test_loader = build_dataloaders(cfg, train_df, val_df, test_df)

    loss_fn = build_loss(cfg, pos_weights, cooc)
    loss_fn.to(device)

    # ---- Load models -----------------------------------------------------
    var_model  = build_model(cfg, cooc, variant=True)
    base_model = build_model(cfg, cooc, variant=False)

    def load_ckpt(model, path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        print(f"  Loaded checkpoint: {path}")

    if os.path.exists(variant_ckpt):
        load_ckpt(var_model, variant_ckpt)
    else:
        print(f"  [warn] Variant checkpoint not found: {variant_ckpt} — using untrained weights")
        var_model.to(device).eval()

    if os.path.exists(baseline_ckpt):
        load_ckpt(base_model, baseline_ckpt)
    else:
        print(f"  [warn] Baseline checkpoint not found: {baseline_ckpt} — using untrained weights")
        base_model.to(device).eval()

    # ---- Evaluate both ---------------------------------------------------
    var_res  = evaluate_model(var_model,  test_loader, device, cfg, "attention_variant")
    base_res = evaluate_model(base_model, test_loader, device, cfg, "baseline")

    # ---- Grad-CAM on baseline --------------------------------------------
    print("\n[evaluate] Computing Grad-CAM on baseline model...")
    gcam_cams, gcam_masks, gcam_has_box = compute_gradcam_batch(
        base_model, test_loader, backbone_name=backbone,
        device=str(device), grid_size=cfg.get("attention_resolution", 7),
        max_batches=None,
    )
    gcam_loc = batch_localization_metrics(gcam_cams, gcam_masks, gcam_has_box)
    print(f"[evaluate] Localisation (Grad-CAM):            {gcam_loc}")

    # ---- DeLong's test: variant vs baseline ------------------------------
    delong_results = {}
    for i, cls in enumerate(CLASS_NAMES):
        y_true = base_res["_labels"][:, i]
        if y_true.sum() < 2:
            continue
        z, p = delong_test(y_true, var_res["_probs"][:, i], base_res["_probs"][:, i])
        delong_results[cls] = {"z": z, "p": p}

    # ---- Print headline table -------------------------------------------
    print("\n" + "=" * 70)
    print("HEADLINE: Variant vs Baseline")
    print("=" * 70)
    print(f"  Variant  macro AUC : {var_res['macro_auc']:.4f}")
    print(f"  Baseline macro AUC : {base_res['macro_auc']:.4f}")
    print(f"  Variant  mean IoU  : {var_res['localization']['mean_iou']:.4f}")
    print(f"  Grad-CAM mean IoU  : {gcam_loc['mean_iou']:.4f}")
    print(f"  Variant  PG acc    : {var_res['localization']['pointing_game_acc']:.4f}")
    print(f"  Grad-CAM PG acc    : {gcam_loc['pointing_game_acc']:.4f}")

    # ---- Save results ----------------------------------------------------
    # Strip internal _arrays before serialising
    def strip(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    results = {
        "variant":      strip(var_res),
        "baseline":     strip(base_res),
        "gradcam_loc":  gcam_loc,
        "delong":       delong_results,
    }
    out_path = os.path.join(cfg["log_dir"], "eval_results.json")
    os.makedirs(cfg["log_dir"], exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate] Results saved → {out_path}")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--variant_ckpt",  required=True)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--backbone",      default="resnet50")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    full_evaluation(cfg, args.variant_ckpt, args.baseline_ckpt, args.backbone)
