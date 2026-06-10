"""
train.py — Full training loop for both baseline and attention models.

Usage (from project root):
    python -m src.train --config config.yaml [--variant] [--backbone resnet50]

Features:
    • Mixed-precision (AMP) when device=cuda and mixed_precision=true
    • Cosine / step / none learning-rate scheduler with warm-up
    • Gradient clipping
    • Best-checkpoint saving by macro AUC
    • Early stopping
    • Loss-component logging (cls / attn / corr)
    • Ablation flags: --no_lattn, --no_lcorr
"""

import os
import sys
import time
import json
import yaml
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

# Project imports
from src.data.splits  import (load_dataframe, build_balanced_subset,
                               patient_level_split, get_class_weights,
                               compute_cooccurrence_matrix, CLASS_NAMES)
from src.data.dataset import build_dataloaders, ChestXrayDataset
from src.data.masks   import load_bbox_lookup
from src.models.model import build_model
from src.losses       import build_loss


# ============================================================================
# Utilities
# ============================================================================

def get_scheduler(optimizer, cfg, n_train_steps):
    sched_name = cfg.get("scheduler", "cosine")
    warmup     = cfg.get("warmup_epochs", 2)
    epochs     = cfg.get("epochs", 30)

    if sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup,
            eta_min=1e-7,
        )
    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return None


def save_checkpoint(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ============================================================================
# Train one epoch
# ============================================================================

def train_one_epoch(model, loader, loss_fn, optimizer, scaler,
                    device, cfg, epoch):
    model.train()
    running = {"cls": 0.0, "attn": 0.0, "corr": 0.0, "total": 0.0}
    n_batches     = 0
    log_interval  = cfg.get("log_interval", 50)
    use_amp       = cfg.get("mixed_precision", False) and device.type == "cuda"
    grad_clip     = cfg.get("grad_clip", 1.0)

    for step, (imgs, labels, masks, has_box) in enumerate(loader):
        imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
        has_box = has_box.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits, attn_map = model(imgs)
            total_loss, ld   = loss_fn(logits, labels, attn_map, masks, has_box)

        if use_amp:
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k in running:
            running[k] += ld[k]
        n_batches += 1

        if (step + 1) % log_interval == 0:
            avg = {k: v / n_batches for k, v in running.items()}
            print(f"  [E{epoch:03d} step {step+1:4d}] "
                  f"cls={avg['cls']:.4f}  attn={avg['attn']:.4f}  "
                  f"corr={avg['corr']:.4f}  total={avg['total']:.4f}")

    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg):
    model.eval()
    all_probs, all_labels = [], []
    running = {"cls": 0.0, "attn": 0.0, "corr": 0.0, "total": 0.0}
    n_batches = 0
    use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"

    for imgs, labels, masks, has_box in loader:
        imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
        has_box = has_box.to(device)

        with autocast(enabled=use_amp):
            logits, attn_map = model(imgs)
            _, ld            = loss_fn(logits, labels, attn_map, masks, has_box)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

        for k in running:
            running[k] += ld[k]
        n_batches += 1

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Macro AUC (skip classes with no positives)
    valid = [i for i in range(all_labels.shape[1]) if all_labels[:, i].sum() > 0]
    if valid:
        macro = roc_auc_score(all_labels[:, valid], all_probs[:, valid], average="macro")
    else:
        macro = float("nan")

    avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
    return macro, avg_losses


# ============================================================================
# Main training loop
# ============================================================================

def train(cfg: dict,
          variant:    bool = True,
          no_lattn:   bool = False,
          no_lcorr:   bool = False,
          resume:     str  = None):
    """
    Full training loop.

    Parameters
    ----------
    cfg       : loaded config dict
    variant   : True → AttentionModel, False → BaselineModel
    no_lattn  : ablation — disable L_attn (set lambda1=0)
    no_lcorr  : ablation — disable L_corr (set lambda2=0)
    resume    : path to checkpoint to resume from
    """
    device = torch.device(cfg.get("device", "cpu"))
    seed   = cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Data ---------------------------------------------------------------
    print("[train] Loading data...")
    df = load_dataframe(cfg["data_dir"])
    if not cfg.get("sample_mode", True) and cfg.get("subset_size"):
        df = build_balanced_subset(df, cfg["subset_size"], seed)

    train_df, val_df, test_df = patient_level_split(
        df, cfg["val_frac"], cfg["test_frac"], seed
    )

    # Filter to sample dir if in sample_mode
    if cfg.get("sample_mode", True):
        sample_ids = set(os.listdir(cfg["sample_dir"]))
        train_df = train_df[train_df["image_id"].isin(sample_ids)].reset_index(drop=True)
        val_df   = val_df[val_df["image_id"].isin(sample_ids)].reset_index(drop=True)
        test_df  = test_df[test_df["image_id"].isin(sample_ids)].reset_index(drop=True)
        print(f"[train] sample_mode: using {len(train_df)} train / "
              f"{len(val_df)} val / {len(test_df)} test images")

    train_loader, val_loader, _ = build_dataloaders(cfg, train_df, val_df, test_df)

    # ---- Class weights & co-occurrence ------------------------------------
    pos_weights = get_class_weights(train_df)
    cooc        = compute_cooccurrence_matrix(train_df)

    # ---- Model ------------------------------------------------------------
    print(f"[train] Building {'AttentionModel' if variant else 'BaselineModel'} "
          f"({cfg.get('backbone', 'resnet50')})")
    model = build_model(cfg, cooc_matrix=cooc, variant=variant)
    model.to(device)

    # ---- Loss -------------------------------------------------------------
    lambda1 = 0.0 if no_lattn else cfg.get("lambda1", 1.0)
    lambda2 = 0.0 if no_lcorr else cfg.get("lambda2", 0.5)
    loss_cfg = dict(cfg)
    loss_cfg["lambda1"] = lambda1
    loss_cfg["lambda2"] = lambda2

    loss_fn = build_loss(loss_cfg, pos_weights, cooc if not no_lcorr else None)
    loss_fn.to(device)

    # ---- Optimizer --------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scaler    = GradScaler(enabled=cfg.get("mixed_precision", False) and device.type == "cuda")
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))

    start_epoch    = 0
    best_auc       = -float("inf")
    patience_count = 0
    patience       = cfg.get("early_stop_patience", 7)
    history        = []

    # ---- Resume -----------------------------------------------------------
    if resume:
        start_epoch, prev_metrics = load_checkpoint(model, optimizer, resume, device)
        best_auc = prev_metrics.get("val_macro_auc", best_auc)
        print(f"[train] Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}")

    # ---- Warm-up ----------------------------------------------------------
    warmup_epochs = cfg.get("warmup_epochs", 2)
    base_lr       = cfg.get("lr", 1e-4)

    # ---- Training loop ----------------------------------------------------
    backbone_tag = cfg.get("backbone", "resnet50")
    variant_tag  = "attention" if variant else "baseline"
    ablation_tag = ("_no_lattn" if no_lattn else "") + ("_no_lcorr" if no_lcorr else "")
    run_name     = f"{backbone_tag}_{variant_tag}{ablation_tag}"

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    log_path = os.path.join(cfg["log_dir"], f"{run_name}_log.json")
    os.makedirs(cfg["log_dir"], exist_ok=True)

    print(f"[train] Starting run: {run_name}")
    print(f"[train] Epochs={cfg.get('epochs',30)}  batch={cfg.get('batch_size',32)}  "
          f"lr={base_lr}  device={device}")

    for epoch in range(start_epoch, cfg.get("epochs", 30)):
        t0 = time.time()

        # Warm-up: linear LR ramp
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, cfg, epoch
        )

        val_auc, val_losses = validate(model, val_loader, loss_fn, device, cfg)

        if epoch >= warmup_epochs and scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        row = {
            "epoch":         epoch,
            "val_macro_auc": val_auc,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_losses.items()},
            "time_s":        round(elapsed, 1),
        }
        history.append(row)

        print(f"[E{epoch:03d}] val_AUC={val_auc:.4f}  "
              f"train_total={train_metrics['total']:.4f}  "
              f"val_total={val_losses['total']:.4f}  "
              f"[{elapsed:.0f}s]")

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc       = val_auc
            patience_count = 0
            ckpt_path = os.path.join(cfg["checkpoint_dir"],
                                     f"{run_name}_best.pt")
            save_checkpoint(model, optimizer, epoch,
                            {"val_macro_auc": val_auc}, ckpt_path)
            print(f"  ↑ New best AUC={best_auc:.4f} — saved {ckpt_path}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[train] Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Save log
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Done. Best val AUC={best_auc:.4f}. Log → {log_path}")
    return history


# ============================================================================
# CLI entry-point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Explanation-Supervised Attention model"
    )
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--variant",  action="store_true",
                        help="Train AttentionModel (default: BaselineModel)")
    parser.add_argument("--backbone", default=None,
                        help="Override backbone in config")
    parser.add_argument("--no_lattn", action="store_true",
                        help="Ablation: disable L_attn")
    parser.add_argument("--no_lcorr", action="store_true",
                        help="Ablation: disable L_corr")
    parser.add_argument("--resume",   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.backbone:
        cfg["backbone"] = args.backbone

    train(cfg,
          variant=args.variant,
          no_lattn=args.no_lattn,
          no_lcorr=args.no_lcorr,
          resume=args.resume)
