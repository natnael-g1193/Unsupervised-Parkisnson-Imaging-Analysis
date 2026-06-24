# Unsupervised Deep Learning for Parkinson's Disease Imaging
## Thesis Project — Natnael Solomon Gebremichael

MSc Data Science, University of Milano-Bicocca  
Supervisor: Professor Presotto
---

## Project Overview

This project investigates the clinical and demographic correlates of latent
representations learned by a β-Variational Autoencoder (β-VAE) trained on
DaTSCAN SPECT imaging data from the PPMI database.

The model compresses each 3D brain scan into 256 latent dimensions. This
analysis identifies which of those dimensions encode clinically meaningful
information — motor severity, cognitive status, disease duration — and which
are confounded by scanner hardware effects.

---

## Data

Raw data is governed by the PPMI Data Use Agreement and is **not committed
to this repository**. Access requires registration at:
https://www.ppmi-info.org/

Data must be placed in `data/raw/ppmi_clinical/` before running any notebook.

---

## Folder Structure

```
data/
  raw/ppmi_clinical/       ← PPMI table downloads (not in git)
  raw/ppmi_imaging/        ← DaTSCAN scan files (not in git)
  processed/latent_vectors/← VAE output CSVs (not in git)
  processed/clinical_merged/← Merged analysis-ready files (not in git)
  existing_mahmoud/        ← Baseline analysis files from prior work (not in git)

notebooks/
  pipeline/                ← Main analysis notebooks (5.0 onwards)
  exploration/             ← Scratch notebooks prefixed EXP_YYYY-MM_

src/
  data_utils.py            ← Patient split, active dimension filtering
  stats_utils.py           ← Correlation tests, Bonferroni, ANCOVA

results/
  correlation/             ← Output CSVs (not in git)
  figures/                 ← Publication figures (tracked in git)
  models/                  ← Fitted scaler/PCA objects (not in git)

reference/                 ← PPMI codebooks, papers
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Pipeline

Run notebooks in order:

| Notebook | Description |
|---|---|
| `notebooks/pipeline/5_0_prepare_data.ipynb` | Merge latents + clinical data, patient-stratified split |
| `notebooks/pipeline/5_1_correlation_analysis.ipynb` | Bonferroni-corrected correlation, ANCOVA |

---

## Key Design Decisions

- **Patient-stratified split**: all visits of one patient stay in train or val — never split across both
- **PCA fitted once**: `results/models/scaler_sbr.pkl` and `pca_sbr.pkl` are fit on training data only and applied to validation
- **Active dimension filtering**: dimensions with std < 0.1 treated as collapsed and excluded from analysis
- **Multiple comparison correction**: Bonferroni (α / n_active_dims) applied to all correlation tests
