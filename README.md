# ForestFireGT 🌲🔥

A reproducible machine learning pipeline for forest fire risk prediction in Guatemala, developed alongside a scientific publication.

---

## Overview

This repository contains a structured ML pipeline that trains and evaluates binary classifiers to predict fire occurrence using environmental, climatic, and topographic features. The pipeline is designed for reproducibility and scientific transparency, with outputs compatible with publication-quality figures.

**Target variable:** `fire` (binary: 0 = no fire, 1 = fire)  
**Class imbalance:** ~0.05% positive rate (~5,360 fires vs. ~11,046,352 non-fire observations)  
**Primary metric:** Matthews Correlation Coefficient (MCC)

---

## Repository Structure

```
ForestFireGT/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                        # allvars_2017-2024.csv (input data)
│   └── processed/                  # split_index.csv
├── src/
│   ├── utils/                      # Reusable plotter modules
│   ├── 01_exploration/
│   ├── 02_model_training/          # Main pipeline scripts
│   ├── 03_model_validation/
│   ├── 04_fire_risk_validation/
│   └── 05_future_predictions/
├── results/
│   └── 01_model_testing/
│       ├── <model_name>/           # One directory per model
│       │   ├── metrics.json
│       │   ├── confusion_matrix.csv
│       │   ├── predictions.csv
│       │   ├── pr_curve.csv
│       │   ├── roc_curve.csv
│       │   └── feature_importance.csv
│       └── figures/                # All output plots
└── outputs/
```

---

## Input Data

**File:** `data/raw/allvars_2017-2024.csv`

| Category | Features |
|---|---|
| Temporal | `month` |
| Climate | `PET`, `climbal`, `pr`, `rsds`, `tas`, `tasmax`, `tasmin` |
| Topography | `aspect_sin`, `aspect_cos`, `elev`, `slope` |
| Proximity | `distRoads`, `distUrban`, `distWater` |

> `lat`, `lon`, `year`, and `fire` are excluded from the feature set.

---

## Pipeline

### Step 0 — Data Splitting (`split_data.py`)

Splits the dataset into **train (65%) / test (25%) / validation (10%)** using stratified sampling on the target column. Outputs `data/processed/split_index.csv` containing only row indices and split labels — no data duplication.

### Step 1 — Baseline Model Testing (`01_model_testing.py`)

Trains a set of classifiers as a baseline to characterize the class imbalance problem. One SLURM job is submitted per model via `submit_model_jobs.sh`. Each model run saves:

- `metrics.json` — MCC, F1, precision, recall, ROC-AUC, PR-AUC
- `confusion_matrix.csv`
- `predictions.csv`
- `pr_curve.csv` / `roc_curve.csv`
- `feature_importance.csv`

**Models tested:** Logistic Regression, Naive Bayes, Random Forest, Extra Trees, Gradient Boosting, XGBoost, KNN

**Key finding:** Most tree-based models predict all zeros due to extreme class imbalance. This is expected — the baseline exists to demonstrate the problem and motivate the next step.

> **Note on XGBoost:** `class_weight="balanced"` is not supported by XGBoost. `scale_pos_weight` must be used instead (not yet implemented in this step).

---

## Reusable Plotters (`src/utils/`)

All plotters are standalone functions called by separate caller scripts. Figures are output as **PNG + PDF + SVG at 300 DPI**, styled with `seaborn whitegrid` (vertical gridlines only).

| Module | Function | Description |
|---|---|---|
| `plot_metrics.py` | `plot_model_comparison()` | 2×2 subplot comparing all models across key metrics |
| `plot_confusion_matrix.py` | `plot_confusion_matrix()` | Single-model confusion matrix with proportions and raw counts |
| `plot_pr_curve.py` | `plot_pr_curve()` | PR curve with no-skill baseline and AP annotation |
| `plot_roc_curve.py` | `plot_roc_curve()` | ROC curve with diagonal baseline and AUC annotation |
| `plot_feature_importance.py` | `plot_feature_importance()` | Feature importances (tree-based) or coefficients (linear models) |

Caller script: `01_plot_model_testing.py`. To plot a subset of models, copy the caller and edit `MODELS_TO_PLOT`.

---

## Design Decisions

- **Split index only:** `split_index.csv` stores row indices and labels only. Downstream scripts load the original CSV and filter by split — no data is duplicated.
- **Plotter/caller separation:** Plotters in `utils/` are stable and never modified. Callers control what gets plotted and where, and can freely reuse plotters across pipeline steps.
- **SLURM:** Each model is an independent job submitted via `submit_model_jobs.sh`.
- **`class_weight="balanced"`** is applied across all `sklearn` models as a baseline imbalance handling strategy.
- **MCC** is the primary evaluation metric throughout, given the severe class imbalance.

---

## What's Next

```
src/02_model_training/
├── 01_model_testing.py          ✅ Done
├── 01_plot_model_testing.py     ✅ Done
├── 02_overunder_sampling.py     ← In progress
├── 03_hyperparameter_tuning.py
└── plot_performance.py
```

- **`02_overunder_sampling.py`** — Test resampling strategies (random under/oversampling, SMOTE, SMOTEENN, SMOTETomek) focused on RF and XGBoost.
- **`03_hyperparameter_tuning.py`** — RandomizedSearchCV or Optuna search after best resampling strategy is identified.

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn   # required for Step 2 (SMOTE etc.)
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

All scripts use a **fixed random seed** defined at the top of each file. The train/test/validation split is determined solely by `split_index.csv`, ensuring consistent data partitioning across all pipeline steps.
