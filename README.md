# 🫁 Explanation-Supervised Attention for Multi-Label Thoracic Disease Classification

**Course:** CSCD 618 / DSCD 604 — MPhil Data Science, Algorithmic Track  
**Team:** Israel Agyekum · Joel Dadi-Klutse · Eric Okyere  
**Dataset:** NIH ChestX-ray14 — 112,120 chest X-rays · 30,805 patients · 14 pathology classes

---

## 🔴 Live Demos

| | Link |
|---|---|
| 📊 **Interactive Dashboard** | [israelagyekum.github.io/dl-chest-xray-Attention/dashboard.html](https://israelagyekum.github.io/dl-chest-xray-Attention/dashboard.html) |
| 🤖 **Streamlit Inference App** | [dl-chest-xray-attention-1.streamlit.app](https://dl-chest-xray-attention-1.streamlit.app/) |
| 💾 **GitHub Repository** | [github.com/israelagyekum/dl-chest-xray-Attention](https://github.com/israelagyekum/dl-chest-xray-Attention) |

---

## 📌 Core Hypothesis

Training spatial attention maps to align with clinical bounding boxes (*explanation supervision*) produces explanations that are **measurably more faithful** than post-hoc Grad-CAM — while preserving competitive classification AUC at **zero extra inference cost**.

Validated across three backbones (ResNet-50, DenseNet-121, EfficientNet-B0) and 14 thoracic pathologies.

---

## 📊 Headline Results

| Model | Macro AUC ↑ | Mean IoU ↑ | Pointing Game ↑ | Note |
|---|---|---|---|---|
| ResNet50-Baseline | **0.8434** | 0.2228 | 0.3789 | Best classification |
| ResNet50-Attn (ours) | 0.7603 | 0.2991 | **0.5063** | Best pointing game |
| DenseNet121-Baseline | 0.8198 | 0.2342 | 0.3912 | — |
| DenseNet121-Attn (ours) | 0.7487 | **0.3178** | 0.5201 | Best IoU |
| EfficientNet-B0-Baseline | 0.7989 | 0.2156 | 0.3645 | — |
| EfficientNet-B0-Attn (ours) | 0.7234 | 0.2834 | 0.4812 | — |

**Key findings:**
- Supervised attention improves Mean IoU by **+34.7%** over post-hoc Grad-CAM (ResNet50)
- Pointing Game accuracy improves by **+33.6%** (ResNet50) and **+33.0%** (DenseNet121)
- AUC cost is ~9.9% for ResNet50 — expected: attention supervision shifts capacity toward spatial alignment
- DeLong's test confirms AUC differences are statistically significant (p < 3×10⁻³¹)

---

## 🏗️ Architecture

```
Chest X-ray (224×224×3)
        │
   CNN Backbone  ──  ResNet-50 / DenseNet-121 / EfficientNet-B0  (ImageNet pretrained)
        │  F ∈ ℝ^(C×7×7)
   Channel Attention  ──  CBAM-style squeeze-excitation  (optional)
        │
   Spatial Attention  ──  Conv 1×1 → sigmoid → A ∈ ℝ^(1×7×7)  ◄── ★ KEY COMPONENT
        │  F' = F ⊙ A  (broadcast)
   Global Average Pool
        │
   Linear → 14 logits → sigmoid → P(disease | image)
        │
   L = L_cls  +  λ_attn · L_attn  +  λ_corr · L_corr
```

| Loss | Formula | Purpose |
|---|---|---|
| **L_cls** | Focal Loss + weighted BCE | Multi-label classification |
| **L_attn** | BCE(A↑, M_box) on boxed images | Explanation supervision — aligns attention with GT boxes |
| **L_corr** | Co-occurrence penalty | Penalises clinically implausible label combinations |

---

## 📁 Repository Structure

```
.
├── app.py                        ← Streamlit inference app (deployed on Streamlit Cloud)
├── dashboard.html                ← Interactive results dashboard (deployed on GitHub Pages)
├── config.yaml                   ← All hyperparameters (single source of truth)
├── requirements.txt              ← Python dependencies for Streamlit Cloud
├── Data_Entry_2017.csv           ← NIH ChestX-ray14 metadata (112,120 images)
├── BBox_List_2017.csv            ← 880 bounding boxes for 8 pathologies
├── src/
│   ├── data/
│   │   ├── splits.py             ← Patient-level split + class weights + co-occurrence
│   │   ├── masks.py              ← Bounding box → binary 7×7 attention mask
│   │   └── dataset.py            ← PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── backbone.py           ← ResNet50 / DenseNet121 / EfficientNet-B0 wrappers
│   │   ├── attention.py          ← Supervised spatial + channel attention modules
│   │   ├── correlation.py        ← Label co-occurrence regulariser
│   │   └── model.py              ← BaselineModel + AttentionModel + build_model()
│   ├── losses.py                 ← FocalLoss, AttentionLoss, CombinedLoss
│   ├── train.py                  ← Training loop (AMP, scheduler, checkpointing)
│   ├── evaluate.py               ← Classification + localisation + DeLong's test
│   ├── gradcam.py                ← Grad-CAM implementation for baseline comparison
│   ├── metrics.py                ← AUC, F1, IoU, pointing-game, DeLong
│   └── plots.py                  ← Training curves, ROC, heatmaps, comparison charts
├── notebooks/
│   ├── colab_inference_server.ipynb  ← Flask + ngrok inference server for Live Demo
│   ├── colab_train.ipynb             ← Full training notebook (Google Colab + Drive)
│   ├── kaggle_train.ipynb            ← GPU training notebook (Kaggle)
│   └── eda.ipynb                     ← Exploratory data analysis
├── outputs/
│   ├── checkpoints/              ← Model weights (stored on Google Drive)
│   ├── logs/eval_results.json    ← Evaluation metrics (AUC, IoU, pointing-game)
│   └── figures/                  ← EDA and training visualisation plots
├── report/
│   ├── DL_Final_Report_Conference.pdf   ← Conference-format final report
│   ├── DL_Project_Report_FINAL.docx     ← Word version
│   └── latex/                           ← Full LaTeX source
└── data/
    ├── sample/                   ← Sample X-ray images for app demo
    ├── BBox_List_2017.csv
    └── Data_Entry_2017.csv
```

---

## 🚀 Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

> **Note:** Model checkpoints are stored on Google Drive (too large for GitHub).  
> Set `DRIVE_FOLDER_ID` in your environment or `.streamlit/secrets.toml` to auto-download them.

### 3. Live inference server (for dashboard Live Demo tab)
1. Open `notebooks/colab_inference_server.ipynb` in Google Colab
2. Run cells 1 → 2 → 3 → 4 in order
3. Copy the ngrok URL → paste into dashboard **Live Demo → Server URL**

### 4. Train from scratch
```bash
# Baseline
python -m src.train --config config.yaml --backbone resnet50

# Attention variant
python -m src.train --config config.yaml --backbone resnet50 --variant
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

## ✅ Project Status

- [x] Dataset analysis & patient-level splits
- [x] Model implementation (BaselineModel + AttentionModel × 3 backbones)
- [x] Custom loss functions (Focal + Attention + Co-occurrence)
- [x] Full training pipeline with AMP + checkpointing
- [x] Evaluation: AUC, IoU, Pointing Game, DeLong's test
- [x] 6 models trained + evaluated (3 backbones × baseline/attention)
- [x] Ablation study (L_attn only, L_corr only, combined)
- [x] Interactive results dashboard (GitHub Pages)
- [x] Streamlit inference app (Streamlit Cloud)
- [x] Colab inference server with real-time XAI (Grad-CAM + attention maps)
- [x] Full research report (PDF + DOCX + LaTeX)
- [x] GitHub repository with complete codebase

---

## 📄 Citation

If you use this work, please cite:

```
Agyekum, I., Dadi-Klutse, J., & Okyere, E. (2025).
Explanation-Supervised Attention for Multi-Label Thoracic Disease Classification.
CSCD 618 / DSCD 604, MPhil Data Science.
```
