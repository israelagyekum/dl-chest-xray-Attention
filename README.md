# Explanation-Supervised Attention for Multi-Label Thoracic Disease Classification

**Course:** CSCD 618 / DSCD 604 — Algorithmic Track  
**Team:** Israel Agyekum · Joel Dadi-Klutse · Eric Okyere  
**Dataset:** NIH ChestX-ray14 (112,120 images, 30,805 patients, 14 findings)

---

## One-sentence north star

A variant CNN whose spatial attention map is *trained to land on the right region* (supervised by bounding boxes) and whose loss respects clinical label co-occurrence — proving, in numbers, that this makes explanations **measurably more faithful** (IoU / pointing-game) than post-hoc Grad-CAM **while keeping AUC competitive at zero extra inference cost.**

---

## Repo structure

```
.
├── config.yaml              ← single source of truth for all hyperparameters
├── Data_Entry_2017.csv      ← NIH ChestX-ray14 metadata (112,120 images)
├── BBox_List_2017.csv       ← 984 bounding boxes for 8 findings
├── data/sample/             ← 30–50 images for CPU smoke-tests (add manually)
├── src/
│   ├── data/
│   │   ├── splits.py        ← patient-level split + class weights + co-occurrence
│   │   ├── masks.py         ← box → binary attention grid mask
│   │   └── dataset.py       ← PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── backbone.py      ← ResNet50 / DenseNet121 / EfficientNet-B0 wrappers
│   │   ├── attention.py     ← supervised CBAM-style spatial attention module
│   │   ├── correlation.py   ← label co-occurrence regulariser + GCN (stretch)
│   │   └── model.py         ← BaselineModel + AttentionModel + build_model()
│   ├── losses.py            ← FocalLoss, AttentionLoss (Dice+MSE), CombinedLoss
│   ├── train.py             ← full training loop (AMP, scheduler, checkpointing)
│   ├── evaluate.py          ← classification + localisation + DeLong's test
│   ├── gradcam.py           ← vanilla Grad-CAM for baselines
│   ├── metrics.py           ← AUC, F1, IoU, pointing-game, DeLong
│   └── plots.py             ← training curves, ROC, heatmaps, comparison charts
├── notebooks/
│   ├── eda.ipynb            ← exploratory data analysis
│   └── kaggle_train.ipynb   ← GPU training notebook for Kaggle
├── outputs/
│   ├── checkpoints/         ← saved model weights
│   ├── logs/                ← training JSON logs
│   └── figures/             ← all generated plots
└── report/                  ← final write-up (LaTeX / PDF)
```

---

## Quickstart

### 1. Environment

```bash
pip install torch torchvision scikit-learn pandas numpy pyyaml \
            seaborn matplotlib scipy
```

### 2. Add sample images (CPU smoke-test)

Copy 30–50 PNG images from the NIH dataset into `data/sample/`.  
Update `config.yaml`: `sample_mode: true`.

### 3. CPU smoke-test (verify everything works before Kaggle)

```bash
# Test splits + masks
python -m src.data.splits
python -m src.data.masks

# Test model forward + backward pass
python -m src.models.model

# Test loss computation
python -m src.losses

# Test Grad-CAM
python -m src.gradcam

# Test metrics
python -m src.metrics
```

### 4. Train on Kaggle (GPU)

1. Upload this repo as a Kaggle dataset.
2. Attach **NIH ChestX-ray14** (read-only).
3. Open `notebooks/kaggle_train.ipynb`.
4. Set `sample_mode: false` in the config cell.
5. Run all cells.

Training commands (from repo root, after adjusting config):

```bash
# Phase 2 — Baseline
python -m src.train --config config.yaml --backbone resnet50

# Phase 3 — Attention variant
python -m src.train --config config.yaml --backbone resnet50 --variant

# Phase 4 — Ablations
python -m src.train --config config.yaml --backbone resnet50 --variant --no_lattn
python -m src.train --config config.yaml --backbone resnet50 --variant --no_lcorr
```

### 5. Evaluate

```bash
python -m src.evaluate \
  --config config.yaml \
  --variant_ckpt  outputs/checkpoints/resnet50_attention_best.pt \
  --baseline_ckpt outputs/checkpoints/resnet50_baseline_best.pt \
  --backbone resnet50
```

---

## Architecture

```
Chest X-ray (224×224×3)
        │
   CNN Backbone (ResNet50 / DenseNet121 / EfficientNet-B0)
        │  F ∈ R^(C×7×7)
   [Channel Attention — CBAM style]
        │
   Spatial Attention Module  →  A ∈ R^(1×7×7)  ──► L_attn (supervised vs box mask)
        │  F' = F ⊙ A
   Global Average Pool
        │
   Linear → 14 logits → sigmoid
        │
   L_cls (Focal BCE)  +  λ1·L_attn  +  λ2·L_corr
```

**L_attn** (on boxed images): `Dice(A, M) + 0.5·MSE(A, M)`  
**L_attn** (on unboxed images): `0.01·‖A‖₁` (sparsity)  
**L_corr**: penalises prediction patterns that violate empirical label co-occurrence

---

## Expected headline table

| Model | Macro AUC | Mean IoU | Pointing-Game Acc |
|---|---|---|---|
| Attention variant (ours) | ≥ baseline | **↑ vs Grad-CAM** | **↑ vs Grad-CAM** |
| Baseline + Grad-CAM | (reference) | reference | reference |
| Baseline (no supervision) | reference | lower | lower |

*DeLong's test assesses whether the AUC difference is statistically significant.*

---

## Kaggle GPU quota

Each account gets ~30 GPU-hrs/week on free tier.  
With 3 accounts the team has ≈90 GPU-hrs/week — enough to run all 3 backbones × both variants × ablations in parallel.

---

## Definition of done

- [ ] Headline table: variant vs baseline (AUC + IoU + pointing-game)
- [ ] Ablation table isolating L_attn and L_corr
- [ ] Heatmap overlays: ours vs Grad-CAM vs ground-truth boxes
- [ ] Per-label ROC curves + training curves
- [ ] Report with method, experiments, results, ablations, limitations
- [ ] Repo reproduces headline table from one command
