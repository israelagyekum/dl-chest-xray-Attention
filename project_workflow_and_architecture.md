# Project Plan — Explanation-Supervised Attention for Multi-Label Thoracic Disease Classification

**Course:** CSCD 618 / DSCD 604 (Algorithmic Track)
**Team:** Israel Agyekum · Joel Dadi-Klutse · Eric Okyere
**Compute model:** Develop on laptops (CPU) → train on Kaggle (free P100/T4, ~30 GPU-hrs/week per account)
**Dataset:** NIH ChestX-ray14, attached read-only on Kaggle (no download)

---

## 0. What we are building (the one-sentence north star)

A **variant CNN** whose spatial attention map is *trained to land on the right region* (supervised by bounding boxes) and whose loss respects clinical label co-occurrence — and proof, in numbers, that this makes explanations **measurably more faithful** (IoU / pointing-game) than post-hoc Grad-CAM **while keeping AUC competitive at zero extra inference cost.**

Everything below serves that sentence. The deliverable is a *model + experiments + report*, not a dashboard.

---

## 1. The Architecture

### 1.1 End-to-end block diagram

```
   Chest X-ray (224 x 224 x 3)              [grayscale replicated to 3 channels
            │                                for ImageNet-pretrained weights]
            ▼
   ┌───────────────────────────┐
   │  CNN BACKBONE (pretrained) │   ResNet50 / DenseNet121 / EfficientNet-B0
   └───────────────────────────┘
            │  feature map  F  ∈  R^(C x 7 x 7)
            ▼
   ┌───────────────────────────┐
   │  (optional) channel attn   │   CBAM-style
   └───────────────────────────┘
            │
            ▼
   ┌───────────────────────────┐        ┌──────────────────────────────┐
   │  SPATIAL ATTENTION MODULE  │ ─────► │  attention map  A ∈ R^(1x7x7) │
   └───────────────────────────┘        └──────────────┬───────────────┘
            │  F' = F ⊙ A  (broadcast over channels)    │  supervised against
            ▼                                           │  box mask M  →  L_attn
   ┌───────────────────────────┐                        │
   │  Global Average Pooling    │                       │
   └───────────────────────────┘                        │
            │  feature vector                            │
            ▼                                            │
   ┌───────────────────────────┐                        │
   │  Linear classifier (→14)   │                        │
   └───────────────────────────┘                        │
            │  14 logits → sigmoid                       │
            ▼                                            │
        ŷ  (14 multi-label probabilities)                │
            │                                            │
            ├── L_cls (focal / weighted BCE)             │
            ├── L_corr (label-correlation term) ─────────┘
            └──────────────►  Total loss
```

### 1.2 Components

| Component | What it is | Notes / design choice |
|---|---|---|
| **Backbone** | ImageNet-pretrained ResNet50 (main), DenseNet121, EfficientNet-B0 | Last conv stage → `F`: 2048×7×7 (ResNet50), 1024×7×7 (DenseNet121), 1280×7×7 (EffNet-B0). For finer localization you can tap a 14×14 stage instead of 7×7 — keep as an ablation. |
| **Spatial attention module** | CBAM-style spatial branch, but the map `A` is promoted to a **trainable, supervised output** (not post-hoc). `A = σ(conv([avgpool_c(F); maxpool_c(F)]))` | This is the core novelty. `A` is what we align to boxes. Then `F' = F ⊙ A`. |
| **Classifier head** | Global average pool → linear → 14 logits → sigmoid | Multi-label (each of 14 findings independent at the output). |
| **Label-correlation component** | A label-relation term so the 14 outputs respect clinical co-occurrence | **Primary (low-risk):** a `L_corr` regularizer using empirical co-occurrence from the training set. **Stretch/ablation:** full ML-GCN label graph. Start simple. |

### 1.3 The losses

```
L = L_cls  +  λ1 · L_attn  +  λ2 · L_corr
```

- **`L_cls`** — multi-label classification. Start with **class-weighted BCE** (handles the heavy imbalance); upgrade to **focal loss** as an improvement/ablation.
- **`L_attn`** — explanation supervision:
  - *On boxed images:* align attention to the box mask, `Dice(A, M) + β·MSE(A, M)` (upsample `A` to mask resolution, or downsample `M` to the grid).
  - *On unboxed images:* a **sparsity/entropy regularizer** (e.g. L1 on `A`) so attention stays focused instead of spreading everywhere.
- **`L_corr`** — penalizes prediction patterns that violate the empirical label co-occurrence learned from training data.
- `λ1`, `λ2` are tuned on the validation set.

> **Honest modeling note:** `A` is a single, class-agnostic spatial map, but boxes are class-specific. For this project we supervise `A` against the **union** of boxes present in an image — a reasonable simplification. *Per-class attention* is the obvious stretch goal if time allows.

---

## 2. The Phased Workflow (every step, beginning to end)

Each step lists **where it runs** (💻 laptop / ☁️ Kaggle) and the **output** it must produce.

### Phase 0 — Setup & alignment
1. 💻 Agree on the north-star sentence (§0) so all three of us optimize the same thing.
2. 💻 Create the Git repo with the structure in §4. Commit a `config.yaml` from day one (no hard-coded paths).
3. 💻 Set up local Python env (`torch`, `timm`, `torchvision`, `scikit-learn`, `pandas`, `albumentations`, `matplotlib`).
4. ☁️ Each member creates a Kaggle account and a notebook; **attach the NIH ChestX-ray14 dataset** (read-only, no download).
5. 💻 Pull ~30–50 sample images locally for offline smoke-testing.
   - **Output:** working repo, runnable env locally + on Kaggle, dataset mounted.

### Phase 1 — Data pipeline (everything depends on this)
6. ☁️/💻 Parse `Data_Entry_2017.csv` → 14-dim multi-hot labels. Parse `BBox_List_2017.csv` → boxes for the 8 localizable findings.
7. 💻 **Patient-level split**: group by `Patient ID`, split *patients* (not images) into train/val/test. Add an assertion that no patient ID appears in two splits.
8. 💻 Build the **class-balanced subset** (your feasibility throttle — start small).
9. 💻 **Box → mask function**: scale box coords from the original 1024×1024 to 224×224, then to the attention grid (7×7 or 14×14); binary mask = 1 inside box (union across boxes), else 0.
10. 💻 `Dataset` returns `(image, label_vector, mask, has_box_flag)`. Resize to 224, ImageNet-normalize, augment.
11. 💻 Smoke-test the `Dataset` on the 30–50 local images, 1 batch, on CPU.
    - **Output:** verified data pipeline; a notebook of exploratory stats (class counts, co-occurrence matrix, box coverage).

### Phase 2 — Baselines (the bar we must beat)
12. ☁️ Train **ResNet50 / DenseNet121 / EfficientNet-B0** baselines (no attention supervision, no correlation term) with weighted BCE.
13. ☁️ Implement **vanilla Grad-CAM** on each baseline.
14. ☁️ Measure baseline **classification** (per-class & macro AUC, P/R/F1) and **localization** (Grad-CAM IoU + pointing-game vs boxes).
    - **Output:** a baseline results table + saved checkpoints. *These numbers are half the paper.*

### Phase 3 — Algorithmic core (our contribution)
15. 💻 Build the **explanation-supervised spatial attention module** (§1.2); smoke-test on CPU sample.
16. 💻 Implement the three loss terms (`L_cls`, `L_attn`, `L_corr`) and the combined `L`; unit-test shapes on the sample.
17. 💻 Implement the **label-correlation term** (simple co-occurrence regularizer first).
18. ☁️ Assemble the full variant model; first full training run on Kaggle.
19. ☁️ Tune `λ1`, `λ2` and learning rate on the validation set; checkpoint best model.
    - **Output:** a trained variant model + its intrinsic attention maps.

### Phase 4 — Evaluation & ablations (this *is* the result)
20. 💻/☁️ Build the **metrics harness**: AUC/F1, confusion matrices, **DeLong's test** for AUC significance vs baseline; IoU + pointing-game for localization.
21. ☁️ Head-to-head: **our supervised attention vs vanilla Grad-CAM** on IoU & pointing-game.
22. ☁️ **Ablations**: (a) remove `L_attn`, (b) remove `L_corr`, (c) neither (= baseline), (d) backbone swap, (e) λ sweep, (f) optional 7×7 vs 14×14 attention.
23. 💻 Generate all **figures**: heatmap overlays (ours vs Grad-CAM vs box), per-label ROC curves, training curves.
    - **Output:** the headline comparison table + ablation table + all figures.

### Phase 5 — Write-up & package
24. 💻 Write the **report/paper** (intro, related work, method, experiments, results, ablations, limitations, conclusion). Each member writes the section matching what they built.
25. 💻 Polish the **repo + README** so the headline table reproduces from one command.
26. 💻 Build **slides** for the presentation.
27. 💻 *(Optional, last)* a tiny **Gradio/Streamlit demo**: drop in an X-ray → see prediction + faithful attention overlay. Only if time remains.
    - **Output:** submitted report, reproducible repo, slides, (optional) demo.

---

## 3. Suggested 1-week timeline (aggressive but doable)

> **Quota tip:** each Kaggle account gets its own ~30 GPU-hrs/week. With three accounts you collectively have **up to ~90 GPU-hrs/week** — parallelize training across them.

| Day | Focus | Runs on |
|---|---|---|
| **1** | Phase 0 + Phase 1 (split, masks, Dataset, EDA, local smoke test) | 💻 + ☁️ setup |
| **2** | First baseline (ResNet50) trains; attention module + losses built & CPU-tested | ☁️ (B) + 💻 (C) |
| **3** | All 3 baselines + Grad-CAM localization done; variant assembled, first run | ☁️ |
| **4** | Variant training + λ tuning; first localization-vs-Grad-CAM comparison | ☁️ |
| **5** | Ablations (−L_attn, −L_corr, backbone swaps) across the three accounts | ☁️ |
| **6** | Aggregate results, all figures/tables, draft the report | 💻 |
| **7** | Polish report + slides + README reproducibility; buffer for reruns; optional demo | 💻 |

---

## 4. Repo structure

```
repo/
├── config.yaml              # device, sample_mode, subset size, λ1, λ2, lr, epochs
├── data/sample/             # 30–50 images for local CPU smoke tests
├── src/
│   ├── data/  splits.py  masks.py  dataset.py
│   ├── models/  backbone.py  attention.py  correlation.py  model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── metrics.py            # AUC, F1, DeLong, IoU, pointing-game
│   └── plots.py
├── notebooks/  kaggle_train.ipynb  eda.ipynb
├── outputs/   checkpoints/  logs/  figures/
├── report/
└── README.md
```

The `config.yaml` carries a `device` and `sample_mode` switch so the **same code** runs as a 50-image CPU smoke test locally and a full GPU run on Kaggle with no edits.

---

## 5. Team division

- **Person A — Data & infra:** Phase 0 scaffold + Phase 1 (split, masks, Dataset) + metrics harness. The patient-level split and box→mask are the highest-risk pieces; put a careful person here.
- **Person B — Baselines & evaluation:** Phase 2 + the eval/ablation runs + Grad-CAM, IoU, pointing-game, DeLong.
- **Person C — Algorithmic core:** Phase 3 (attention module, three losses, correlation term, variant training).

Everyone writes the report section for what they built.

---

## 6. Guardrails (what quietly kills projects like this)

1. **Patient-level leakage** — verify the split with a code assertion, not your eyes. Leakage = great AUC that means nothing.
2. **The box subset is tiny** (~880 images / ~984 boxes, 8 findings). `L_attn` only fires there. Use augmentation + the unboxed regularizer, and frame effect sizes honestly.
3. **Debug locally, train on Kaggle** — never burn GPU quota discovering a typo. Smoke-test on the 50-image sample first.
4. **No baseline = no story** — your contribution is *defined* as "better than vanilla Grad-CAM." Get clean baselines before chasing the variant.
5. **Scope creep** — the demo/dashboard is the last optional half-day, not week one.

---

## 7. Definition of done

- A headline table: **variant vs baseline** showing AUC stays competitive (DeLong) **and** localization (IoU / pointing-game) improves.
- An ablation table isolating `L_attn` and `L_corr`.
- Heatmap overlays showing ours vs Grad-CAM vs ground-truth boxes.
- A report that tells the §0 story, and a repo that reproduces the headline table.
