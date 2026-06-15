# ForestFireGT

A reproducible Python-based machine learning pipeline for forest fire risk prediction and susceptibility mapping in Guatemala. Developed alongside a scientific paper, and structured for open-source publication.

---

## Project Structure

```
ForestFireGT/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── indata/                       # allvars_2017-2024.csv (training input);
│   │                                 #   data_fullfuture_processed/ (future climate)
│   ├── processed/                    # split_index.csv
│   ├── boundaries/                   # Natural Earth admin-1 (auto-downloaded)
│   └── classified_future/            # future predictions: <MODEL>/<scenario>/
├── src/
│   ├── utils/                        # Reusable plotter and utility modules
│   ├── 01_data_exploration/          # Exploratory analysis (maps, PCA, distributions…)
│   ├── 02_model_training/            # Main pipeline (steps 00–06)
│   └── 03_future_classification/     # Future projection (predict + plot, step 07)
├── results/
│   ├── 01_model_testing/
│   ├── 02_overundersampling/
│   ├── 03_hyperparameter_tuning/
│   ├── 04_final_model_training/      # incl. risk_thresholds.json per model
│   ├── 06_risk_maps/
│   └── 07_future_classification/     # future trend + change-map figures
├── models/                           # Saved .joblib model files
└── outputs/
```

---

## Input Data

**File:** `data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv`
**Target column:** `fire` (binary: 0/1)
**Study period:** 2017–2024
**Class imbalance:** ~5,360 fire events vs ~11,046,352 non-fire cells (~0.05% positive rate)
**Spatial resolution:** 1 km grid cells (lat/lon represent cell centers)

**Features used:**
```
PET, climbal, pr, rsds, tas, tasmax, tasmin,
aspect_sin, aspect_cos,
distRoads, distUrban, distWater,
elev, slope
```

**Excluded from features:** `lat`, `lon`, `year`, `month`, `fire`

---

## Data Exploration (`src/01_data_exploration/`)

Preliminary, non-pipeline analysis of the predictors (run independently of the modeling steps):

| Script | Output |
|---|---|
| `map_visualization.py` | Per-variable maps of the period-average (2017–2024) value across Guatemala |
| `plot_distributions.py` | Boxplot and histogram grids, one panel per predictor |
| `corrheatmap.py` | Pairwise correlation heatmap of the predictors |
| `PCA_analysis.py` | PCA scree plot, loadings heatmap, and biplot |
| `timevars_visualization.py` | Yearly spatial-mean trend per temporal predictor (historical baseline + 3 SSP scenarios) |

---

## Pipeline Overview

The pipeline is organized as sequential numbered steps. Each step produces outputs consumed by the next.

### Step 00 — Data Splitting
**Script:** `00_split_data.py`
Stratified train (65%) / test (25%) / validation (10%) split preserving the fire class ratio. Outputs `split_index.csv` — only row indices and split labels, no data duplication.

### Step 01 — Baseline Model Testing
**Script:** `01_model_testing.py <model_name>`
**Plot script:** `01_plot_model_testing.py`
**SLURM:** `sbatch_01_model_testing.sh`

Tests 8 baseline classifiers with class weighting as the only imbalance correction:
`logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `xgboost`, `extra_trees`, `knn`, `naive_bayes`

Evaluates across multiple thresholds: `[0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]`

### Step 02 — Over/Undersampling
**Script:** `02_overundersampling.py <model_name> <sampler_name>`
**Plot script:** `02_plot_overundersampling.py`
**SLURM:** `sbatch_02_overundersampling.sh`

Tests 8 resampling strategies on RF, XGB, and XT (24 combinations). No class weights used — resampling is the only imbalance correction, keeping experiments isolated.

Samplers: `ros`, `smote`, `svmsmote`, `adasyn`, `rus`, `nearmiss`, `tomek`, `ncl`

**Best samplers identified (by ROC AUC / average precision):**
- RF → `rus` (Random Undersampling)
- XGB → `ncl` (Neighbourhood Cleaning Rule)
- XT → `rus` (Random Undersampling)

Selection uses the threshold-independent metrics (ROC AUC, average precision): at this imbalance the peak MCC is nearly flat across samplers and does not discriminate them. Bagging ensembles (RF, XT) rank best with `rus`; boosting (XGB) ranks best with the gentle cleaning rules (`ncl`/`tomek`). `nearmiss` is degenerate (AUC < 0.5) and excluded.

### Step 03 — Hyperparameter Tuning
**Script:** `03_hyperparameter_tuning.py <model_name>`
**Plot script:** `03_plot_hyperparameter_tuning.py`
**SLURM:** `sbatch_03_hyperparameter_tuning.sh`

RandomizedSearchCV, 3-fold CV, 100 iterations. Scored by the **best achievable MCC over thresholds** within each fold (threshold-free), so the search optimizes MCC without locking to a fixed operating point. The best sampler from step 02 is wrapped in an imblearn `Pipeline` so resampling happens inside each CV fold (no leakage into the validation fold). One SLURM job per model (3 total).

Saves `best_params.json` and `cv_results.csv` alongside standard outputs.

**Finding:** tuning produced no improvement over the default hyperparameters for any model (RF, XGB, XT) — test scores were equal or slightly worse, while CV scores were inflated by selection optimism (winner's curse). Step 04 therefore uses the **base** hyperparameters, not the tuned ones. The plot script (`03_plot_hyperparameter_tuning.py`) renders a paired base-vs-tuned comparison per model to make this explicit.

### Step 04 — Final Model Training
**Script:** `04_final_model_training.py <model_name>`
**Plot script:** `04_plot_topk_finalmodel.py`
**SLURM:** `sbatch_04_final_model_training.sh`

Retrains each model from scratch using the **base (default) hyperparameters** — identical to step 02 — with the best sampler from step 02. The tuned hyperparameters from step 03 are deliberately not used (they gave no improvement), so the final models are exactly the step-02 base models, retrained on the full training set. Evaluates on train, test, and validation sets and saves the fitted model as `.joblib` under `models/`.

The decision threshold is committed once as the value maximizing MCC on the **test** set, saved in `training_info.json`, and applied unchanged to all subsets — never re-tuned on the validation set, so the blind validation result stays honest.

Predictions include `lat`, `lon`, `month`, `year` columns — required for spatial mapping and top-k analysis.

The plot script (`04_plot_topk_finalmodel.py`) also performs **top-k coverage analysis** at k = 1%, 2%, 5%, 10%: the proportion of historical fire events captured within the top-k% of cells ranked by predicted probability.

### Step 05 — Risk Thresholds
**Script:** `05_compute_risk_thresholds.py`
**SLURM:** `sbatch_05_compute_risk_thresholds.sh`

Computes **frozen** 5-class fire-risk thresholds per model from the **prediction-rate curve** (Chung & Fabbri, 2003) — replacing Jenks, which was distorted by the per-model probability scale that over/under-sampling induces. Cells are ranked by predicted susceptibility and class boundaries are placed where the ranked cells cumulatively capture target fractions of observed fires (50/75/90/99%). Derived from **held-out data (test + validation)**, never training (which is in-sample and biases the thresholds — markedly for XGBoost). Saves `risk_thresholds.json` (ascending probability `breaks` + per-class area/capture summary + prediction-rate AUC) per model under `results/04_final_model_training/<MODEL_SHORT>/`. Skips if the file exists.

Because these are fixed probability cutoffs, the same standard applies to every map — present and future — so risk-class areas can grow/shrink between scenarios.

### Step 06 — Risk Map Plotting
**Script:** `06_plot_risk_maps.py`

Applies the frozen risk thresholds from step 05 to predicted probabilities and produces fire risk classification maps for Guatemala. Each 1km grid cell is rendered as an exact pixel using `pcolormesh()` on a reconstructed 2D raster — no interpolation. Administrative boundaries are the Natural Earth admin-1 layer (auto-downloaded once to `data/boundaries/`).

**Data:** maps are built from **all subsets combined** (train + test + val) for complete spatial coverage and each cell's full month history. This is a visualization of the model's risk surface — quantitative performance is reported separately on held-out data (steps 02–04), and the risk classes come from the frozen thresholds fit on held-out data (step 05), so the classification is not biased by in-sample training probabilities.

**Aggregation:** for each cell the predicted probability is averaged over the relevant months (across all years), then the frozen thresholds are applied to that **mean probability**. Probabilities are averaged rather than class labels, since the classes are unequal-width bins. The same fixed thresholds are used for every map, so seasons/months are directly comparable.

Produces two figure types:
- **Fire-season comparison** (main text): one figure with a panel per model (RF, XGB, XT), each showing the dry-season (Nov–Apr — the official fire season) risk climatology. Panels are lettered A), B), C) for cross-referencing. → `results/06_risk_maps/fire_season_comparison.{png,pdf,svg}`
- **Monthly climatology** (appendix): one 3×4 panel figure per model, with the 12 calendar-month climatologies. → `results/06_risk_maps/<MODEL_SHORT>/<MODEL_SHORT>_monthly_climatology.{…}`

*(Observed fires per month — used to justify the fire-season window — are reported separately as a table.)*

### Step 07 — Future Classification
**Predict:** `01_future_classification.py <model_name>` (SLURM: `sbatch_01_future_classification.sh`)
**Plot:** `01_plot_future_classification.py` (SLURM: `sbatch_01_plot_futclassif.sh`)
**Directory:** `src/03_future_classification/`

Applies each saved final model to preprocessed future climate (`data/indata/data_fullfuture_processed/`) for the historical period (1961–2014) and three SSP scenarios (SSP1-2.6, SSP3-7.0, SSP5-8.5) through 2100, writing monthly per-cell probabilities to `data/classified_future/<MODEL>/<scenario>/`. No classification here — only `predict_proba` (kept fast and compact).

The plot script applies the **frozen risk thresholds** (step 05) and writes two figures to `results/07_future_classification/`:
- **Temporal trend** (results): `fig1_trend.*` — 3 rows (one per model, lettered A/B/C), % of national area in the High+Very-High classes vs year (1961–2100); historical + 3 SSP lines, 10-year rolling mean.
- **Spatial change** (results / appendix), per model in two versions: `fig2_<MODEL>_normal.*` (absolute fire-season classes) and `fig2_<MODEL>_change.*` (Δ risk class vs the 1991–2014 baseline). Layout: a centered baseline map + a 3×2 grid (rows = SSP scenarios with vertical labels; columns = Mid-century 2041–2060, End-century 2081–2100).

Caveat (stated in the paper): static predictors are held constant over time, and tree models do not extrapolate beyond the training feature range.

---

## Utility Modules (`src/utils/`)

| File | Function(s) | Description |
|---|---|---|
| `threshold_selection.py` | `select_threshold_on_validation()`, `mcc_best_threshold_scorer()` | Select the MCC-optimal decision threshold; threshold-free MCC scorer for hyperparameter search |
| `plot_metrics.py` | `plot_model_comparison()` | 2×2 bar subplot, one bar per model per metric |
| `plot_confmatrix.py` | `plot_confusion_matrix()` | Confusion matrix with proportions + raw counts |
| `plot_confmatrix_panel.py` | `plot_confusion_matrix_panel()` | Confusion matrices arranged in a multi-panel figure |
| `plot_prcurve.py` | `plot_pr_curve()` | PR curve with no-skill baseline + AP annotation |
| `plot_prcurve_panel.py` | `plot_pr_curve_panel()` | PR curves for multiple models in one multi-panel figure |
| `plot_roccurve.py` | `plot_roc_curve()` | ROC curve with diagonal baseline + AUC annotation |
| `plot_roccurve_panel.py` | `plot_roc_curve_panel()` | ROC curves for multiple models in one multi-panel figure |
| `plot_featimportance.py` | `plot_feature_importance()` | Tree importances or linear coefficients |
| `plot_featimportance_panel.py` | `plot_feature_importance_panel()` | Feature importances arranged in a multi-panel figure |
| `plot_mcc_heatmap.py` | `plot_mcc_heatmap()` | Heatmap of MCC across models × samplers |
| `plot_topk.py` | `plot_topk_coverage()` | Top-k coverage bar charts, one figure per k level |
| `risk_thresholds.py` | `compute_risk_thresholds()`, `apply_thresholds()`, `save_thresholds()`, `load_thresholds()` | Frozen prediction-rate fire-risk thresholds (replaces Jenks) |
| `plot_risk_map.py` | `plot_risk_map()`, `plot_risk_map_grid()`, `make_delta_cmap()` | Spatial fire risk maps for Guatemala (single map, multi-panel grid, Δ-class diverging colormap) |

All plots saved as PNG + PDF + SVG at 300 DPI. Style: seaborn `whitegrid`, publication-ready.

---

## Key Design Decisions

- **Split index only:** `split_index.csv` stores row indices and labels — no data duplication across subsets.
- **Plotter/caller separation:** Utility plotters in `utils/` are generic and never modified. Caller scripts control what gets plotted and where.
- **No double-correction:** Steps 02–04 use resampling as the only imbalance correction. No `class_weight` or `scale_pos_weight` is combined with resampling.
- **Primary metric:** Matthews Correlation Coefficient (MCC), reported at each model's MCC-optimal threshold. Model and sampler *selection* uses the threshold-independent ROC AUC and average precision, since peak MCC is too flat to discriminate options at this imbalance.
- **Threshold handling:** No fixed threshold. Each run is reported at the threshold that maximizes its MCC (the peak of its MCC-vs-threshold curve), selected via `utils/threshold_selection.py` and stored inside the metrics JSON under `"threshold"`. Threshold-sensitive outputs use fixed filenames (`metrics.json`, `confusion_matrix.csv`; with a `_<subset>` suffix in step 04). PR/ROC curves and feature importance are threshold-independent and saved once per run.
- **Validation set:** Held out entirely until step 04 — never used for model selection or threshold tuning.
- **Risk classification:** Frozen 5-class thresholds from the prediction-rate curve (step 05), derived from held-out data and applied unchanged to all subsets, the present-day maps, and the future scenarios — so risk classes stay comparable across time. (Replaced Jenks, which was distorted by the per-model probability scale.)
- **Random seed:** `42` throughout (except `00_split_data.py` which uses `107`).

---

## Decision Threshold

No fixed per-model threshold is used. The decision threshold is treated as a post-hoc operating point: each model is reported at the threshold that **maximizes its MCC** (the peak of its MCC-vs-threshold curve), selected on the evaluation predictions and stored inside the metrics JSON. In step 04 the threshold is committed once on the **test** set and applied unchanged to all subsets (train / test / validation).

This avoids comparing models at a single shared cut, which would conflate model quality with how well that cut aligns with each model's probability scale. The deployed risk maps use the frozen prediction-rate thresholds (step 05), not this binary operating threshold.

---

## Risk Classification

Five risk classes based on the frozen prediction-rate thresholds (step 05) applied to predicted probabilities:

| Class | Label |
|---|---|
| 1 | Very low |
| 2 | Low |
| 3 | Medium |
| 4 | High |
| 5 | Very high |

Colors: `#ffffb2` → `#fecc5c` → `#fd8d3c` → `#f03b20` → `#bd0026`

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
geopandas
joblib
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Reproducing the Pipeline

```bash
# Step 00 — Split data
python 00_split_data.py

# Step 01 — Baseline model testing (one job per model)
bash sbatch_01_model_testing.sh

# Step 02 — Over/undersampling (one job per model+sampler combination)
bash sbatch_02_overundersampling.sh

# Step 03 — Hyperparameter tuning (one job per model)
bash sbatch_03_hyperparameter_tuning.sh

# Step 04 — Final model training (one job per model)
bash sbatch_04_final_model_training.sh

# Step 05 — Compute risk thresholds
bash sbatch_05_compute_risk_thresholds.sh

# Step 06 — Plot risk maps
python 06_plot_risk_maps.py

# Step 07 — Future classification (one job per model), then plot
#   (scripts in src/03_future_classification/)
bash sbatch_01_future_classification.sh
python 01_plot_future_classification.py

# Plotting scripts (run after each step completes)
python 01_plot_model_testing.py
python 02_plot_overundersampling.py
python 03_plot_hyperparameter_tuning.py
python 04_plot_topk_finalmodel.py
```