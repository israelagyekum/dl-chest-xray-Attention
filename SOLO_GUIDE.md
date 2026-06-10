# Solo Execution Guide — Step by Step
## Explanation-Supervised Attention | CSCD 618 / DSCD 604
### Israel Agyekum

This is your complete playbook from zero to submitted project — every step in order.

---

## PHASE 0 — Local Setup (Day 1, ~1 hour)

### Step 1 — Install Python dependencies locally

```bash
pip install torch torchvision scikit-learn pandas numpy pyyaml \
            seaborn matplotlib scipy streamlit Pillow
```

### Step 2 — Verify everything works (no images needed)

Run these from the project root (`DL PROJECT WORK/`):

```bash
python -m src.data.splits        # tests CSV parsing + patient split
python -m src.data.masks         # tests bounding-box → mask
python -m src.models.backbone    # tests ResNet50/DenseNet121/EfficientNet-B0
python -m src.models.attention   # tests CBAM spatial attention
python -m src.models.model       # tests full forward pass
python -m src.losses             # tests combined loss
python -m src.gradcam            # tests Grad-CAM hooks
python -m src.metrics            # tests AUC / IoU / pointing-game / DeLong
```

All should print ✓ at the end. If any fail, tell me the error.

### Step 3 — Add sample images for local smoke-test

1. From the NIH ChestX-ray14 dataset, copy **any 50 PNG images** into `data/sample/`
2. Open `config.yaml` and confirm:
   ```yaml
   sample_mode: true
   device: "cpu"
   batch_size: 8
   epochs: 2
   ```
3. Run the full training smoke-test:
   ```bash
   python -m src.train --config config.yaml --variant
   ```
   This verifies the complete pipeline (data → model → loss → checkpoint) without GPU.

---

## PHASE 1 — EDA (Day 1, ~30 min)

### Step 4 — Run the EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

Run all cells. This generates 5 figures in `outputs/figures/`:
- `class_distribution.png`
- `multilabel_distribution.png`
- `cooccurrence_matrix.png`
- `bbox_analysis.png`
- `split_distribution.png`
- `class_weights.png`

**These go directly into your report (Section 3 — Data).**

---

## PHASE 2 — Kaggle Setup (Day 1–2, ~1 hour)

### Step 5 — Create your Kaggle account

Go to https://www.kaggle.com and sign up (free).  
Enable **phone verification** to unlock GPU access.

### Step 6 — Attach the NIH ChestX-ray14 dataset

1. On Kaggle, click **+ Create → New Notebook**
2. On the right panel → **Add Data** → search `NIH Chest X-rays`
3. Add it as a read-only input (no download — it stays on Kaggle's servers)

### Step 7 — Upload this project as a Kaggle dataset

1. Zip your entire `DL PROJECT WORK` folder
2. Go to https://kaggle.com/datasets → **New Dataset**
3. Name it exactly: `dl-project-code`
4. Upload the zip → wait for processing

> **Tip:** Every time you update the code locally, re-upload the dataset with version +1.

### Step 8 — Open the training notebook

1. In your Kaggle notebook, click **+ Add Data** → search your `dl-project-code` dataset → add it
2. Open `notebooks/kaggle_train.ipynb` from Kaggle (or copy-paste its contents into a new notebook)
3. In **Cell 2**, update the paths:
   ```python
   PROJECT_CODE = '/kaggle/input/dl-project-code'      # your uploaded repo
   DATASET_PATH = '/kaggle/input/nih-chest-xrays'      # the NIH dataset
   ```
4. Go to **Settings → Accelerator → GPU T4 × 2** (or P100)
5. Click **Run All**

---

## PHASE 2 — Baselines Training (Day 2, ~10 GPU-hrs)

### Step 9 — Train all 3 baselines

The notebook (Cell 6–8) trains them sequentially. Expected times on T4:
- ResNet50 baseline: ~2–3 hrs for 30 epochs
- DenseNet121 baseline: ~2–3 hrs
- EfficientNet-B0 baseline: ~2–3 hrs

Each saves a checkpoint to `/kaggle/working/outputs/checkpoints/`.

**After each run completes — download the checkpoint immediately:**
- Go to the notebook's output files panel
- Download `outputs/checkpoints/resnet50_baseline_best.pt` etc.
- Save them to your local `outputs/checkpoints/`

### Step 10 — Baseline Grad-CAM localisation

This runs automatically inside `evaluate.py` (Cell 11 in the notebook). You'll see the IoU and pointing-game numbers for each baseline. **Write these numbers down — they are your baseline to beat.**

---

## PHASE 3 — Attention Variant Training (Day 3, ~3 GPU-hrs)

### Step 11 — Train the attention variant

Cell 9 in the notebook. One run:
```
resnet50_attention_best.pt
```

If your AUC drops a lot compared to baseline, check `lambda1`. Try reducing it to 0.5.

### Step 12 — λ tuning (if needed)

If localisation improved but AUC dropped significantly, edit the config cell in the notebook:
```python
cfg['lambda1'] = 0.5   # try 0.5 instead of 1.0
cfg['lambda2'] = 0.3   # try lower
```
Retrain. You only need the best 1–2 configs; this isn't an exhaustive sweep.

---

## PHASE 4 — Ablations + Full Evaluation (Day 4, ~6 GPU-hrs)

### Step 13 — Run ablations

Cell 10 trains 3 ablations:
- (a) No L_attn — removes explanation supervision entirely
- (b) No L_corr — removes label correlation
- (c) DenseNet121 attention variant — backbone swap

These give you the **ablation table** in your report.

### Step 14 — Full evaluation + headline table

Cell 11 runs `evaluate.py` which:
1. Computes classification AUC/F1 for both models
2. Runs Grad-CAM on baseline
3. Computes IoU + pointing-game for: ours vs Grad-CAM
4. Runs DeLong's test (statistical significance)
5. Saves `eval_results.json` to `outputs/logs/`

Download `eval_results.json` and save it to your local `outputs/logs/`.

### Step 15 — Generate figures

Cell 12 generates:
- `training_curves.png`
- `localization_comparison.png`
- `auc_comparison.png`

Then run locally (with your saved checkpoints + eval_results.json):
```bash
python -c "
import json, yaml
from src.plots import plot_roc_curves, plot_heatmap_overlay
# see src/plots.py for usage
"
```
Or just use the figures already saved by the notebook.

**All figures go into your report.**

---

## PHASE 5 — Report Writing (Day 5–6)

### Step 16 — Report structure

Write in this order (use the figures you already have):

```
1. Introduction (~1 page)
   - Why chest X-ray AI matters
   - Problem: Grad-CAM is post-hoc and unfaithful
   - Our contribution: train attention to attend to the right regions

2. Related Work (~0.5 page)
   - ResNet/DenseNet/EfficientNet backbones
   - CBAM attention
   - Grad-CAM
   - ChestX-ray14 dataset & CheXNet

3. Data (~0.5 page)
   - Dataset stats (use EDA figures)
   - Patient-level split (no leakage)
   - Box coverage stats

4. Method (~1.5 pages)
   - Architecture diagram (from project_workflow_and_architecture.md)
   - Attention module
   - Loss: L = L_cls + λ1·L_attn + λ2·L_corr
   - Each term explained

5. Experiments (~2 pages)
   - Baselines setup
   - Headline table (variant vs baselines — AUC + IoU + pointing-game)
   - Ablation table (−L_attn, −L_corr, backbone swap)
   - Heatmap overlays figure

6. Results & Discussion (~1 page)
   - DeLong test: is the AUC difference significant?
   - Localisation improvement: how much? For which diseases?
   - Failure cases (honest)

7. Conclusion (~0.5 page)
   - Restate the north-star result
   - Limitations (box subset is tiny ~880 images)
   - Future: per-class attention maps
```

**Target length: 6–8 pages (IEEE double-column format).**

---

## PHASE 5 — Demo Dashboard (Day 6, ~30 min)

### Step 17 — Run the Streamlit demo locally

```bash
pip install streamlit
streamlit run app.py
```

Opens at `http://localhost:8501`.

The dashboard:
- Shows prediction probabilities for all 14 classes
- Side-by-side: Input | GT box | Supervised attention | Grad-CAM
- Loads `eval_results.json` for the metrics summary panel
- Shows all generated figures

**For your presentation, run this app and upload a chest X-ray live.**

### Step 18 — Add a few sample images

Copy 5–10 PNG images from the NIH dataset into `data/sample/`.  
These appear in the "pick a sample image" dropdown in the app — useful for the demo.

---

## PHASE 5 — Presentation Slides (Day 7, ~2 hours)

### Step 19 — Slide structure (10–12 slides)

| Slide | Content |
|---|---|
| 1 | Title + team |
| 2 | Motivation: chest X-ray + black-box problem |
| 3 | Our contribution (1-sentence north star) |
| 4 | Architecture diagram |
| 5 | The three loss terms |
| 6 | Results — headline table |
| 7 | Heatmap overlays (ours vs Grad-CAM vs GT box) |
| 8 | Ablation table |
| 9 | Per-class AUC comparison chart |
| 10 | Demo (live or screenshot) |
| 11 | Limitations + future work |
| 12 | Conclusion |

---

## Output Files Reference

After all phases are done, your `outputs/` folder should contain:

```
outputs/
├── checkpoints/
│   ├── resnet50_baseline_best.pt
│   ├── resnet50_attention_best.pt
│   ├── densenet121_baseline_best.pt
│   ├── resnet50_attention_no_lattn_best.pt
│   └── resnet50_attention_no_lcorr_best.pt
│
├── logs/
│   ├── resnet50_baseline_log.json       ← training curves
│   ├── resnet50_attention_log.json
│   └── eval_results.json               ← all metrics for the dashboard
│
└── figures/
    ├── class_distribution.png           → report Section 3
    ├── cooccurrence_matrix.png          → report Section 3
    ├── bbox_analysis.png                → report Section 3
    ├── split_distribution.png           → report Section 3
    ├── resnet50_attention_curves.png    → report Section 5
    ├── localization_comparison.png      → report Section 5
    ├── auc_comparison.png               → report Section 5
    └── [heatmap overlays]               → report Section 5
```

---

## Quick Reference — Key Commands

| Task | Command |
|---|---|
| Local smoke-test | `python -m src.train --config config.yaml --variant` |
| Train baseline (Kaggle) | `python -m src.train --backbone resnet50` |
| Train variant (Kaggle) | `python -m src.train --backbone resnet50 --variant` |
| Ablation −L_attn | `python -m src.train --backbone resnet50 --variant --no_lattn` |
| Ablation −L_corr | `python -m src.train --backbone resnet50 --variant --no_lcorr` |
| Full evaluation | `python -m src.evaluate --variant_ckpt ... --baseline_ckpt ...` |
| Launch demo | `streamlit run app.py` |

---

## When to Ask for Help

Alert me at any of these points:
- Any `python -m src.*` smoke-test fails with an error
- Kaggle paths need adjusting after you see the dataset structure
- AUC or localisation results look unexpected after training
- You want help writing a specific section of the report
- Slides need to be created as a `.pptx` file
