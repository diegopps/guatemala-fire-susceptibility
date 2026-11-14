# ForestFireGT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24080892.svg)](https://doi.org/10.5281/zenodo.24080892)

Code of the paper

> **Mapping present and future forest fire susceptibility in Guatemala: machine learning projections to 2100**
>
> Diegopablo Pineda Schwarz¹, Fred F. Hattermann¹ and Dirk Walther²
>
> ¹ Potsdam Institute for Climate Impact Research (PIK), Telegrafenberg A 31, 14473 Potsdam, Germany
> ² Max Planck Institute of Molecular Plant Physiology, Am Mühlenberg 1, 14476 Potsdam, Germany
>
> Correspondence: Diegopablo Pineda Schwarz (diegopab@pik-potsdam.de)

ForestFireGT trains machine-learning classifiers on monthly forest-fire records of Guatemala
and the climate of nine global climate models (GCMs). It maps the fire susceptibility of every
30 arc-second (about 1 km) land cell of the country in five equal-area classes, and projects
it from 1961 to 2100 under three scenarios (SSP1-2.6, SSP3-7.0, SSP5-8.5). The repository holds
the whole pipeline, from the daily climate data to the tables and figures of the paper.

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Input data](#input-data)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Using the final models](#using-the-final-models)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

## Overview

The pipeline has four stages, one folder of `src/` each:

| Stage | Folder | What it does |
|---|---|---|
| 0 | `src/00_preprocessing` | Aggregates the daily climate of each GCM to monthly means; builds the training table (fire label and 14 predictors of every GCM, cell and month of the training years) |
| 1 | `src/01_data_exploration` | Maps of the fire records and the predictors, climate trends 1961–2100, correlation of the predictors |
| 2 | `src/02_model_training` | Split, choice of the algorithms, correction of the class imbalance, hyperparameter tuning, final models and their validation, susceptibility classes |
| 3 | `src/03_future_classification` | Fire probability of every cell-month 1961–2100 for each final model, GCM and scenario; classes and class change relative to 1991–2014 |

Every script starts with a docstring that documents its method, inputs, outputs and usage in
full. This README gives the overview.

### Data and predictors

- **Grid and cells.** The CHELSA grid (30 arc-seconds). A cell is used if every static layer has
  a value there: the land cells of Guatemala, lakes excluded (131,568 cells).
- **Target.** Whether a cell-month has at least one forest-fire record (`fire`, 0/1).
- **Training years.** 2018–2022 and 2024 (`training.years`; 2017 is excluded, 2023 has no fire
  records): 5,348 fire cell-months out of 9.47 million (0.06%).
- **Climate.** Daily CHELSA-downscaled, bias-adjusted ISIMIP3b data of nine GCMs (CanESM5,
  CNRM-CM6-1, CNRM-ESM2-1, EC-Earth3, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, MRI-ESM2-0,
  UKESM1-0-LL), aggregated to monthly means. The predictors are model climate, not
  observations. For the training years they come from the SSP3-7.0 simulations
  (`training.climate_scenario`).
- **Predictors (14).** Mean, maximum and minimum temperature, precipitation, solar radiation,
  potential evapotranspiration (Priestley–Taylor) and climatic water balance; elevation, slope,
  aspect (sine and cosine), and the distances to roads, urban areas and water bodies. The
  static predictors are held constant in the projections.

### Method

- **The unit is the cell-month.** The training table has one row per GCM for each cell-month:
  the nine GCMs are pooled (85.3 million rows), and all rows of a cell-month share its fire
  label. The split, the cross-validation folds and the resampling act on whole cell-months, so
  the rows of a cell-month are never separated. Scores use the mean probability of each
  cell-month over its GCM rows, so every fire counts once.
- **Subsets.** The cell-months are split 65/25/10% into train, test and validation subsets,
  stratified by the fire label. The validation subset is used for the first time in step 04.
- **Selection by peak MCC.** Algorithms, samplers and configurations are compared at the
  maximum of the Matthews correlation coefficient over all decision thresholds, alongside
  average precision and ROC-AUC.
- **Decision threshold.** Each final model has one threshold: the peak MCC on the test subset.
  It is never changed on the validation subset.
- **Equal-area classes.** The susceptibility of a cell is the mean probability of a final model
  over the GCMs and all months of the training years. Its quintiles (Verde and Zêzere, 2010)
  give four breaks per model, so each class holds 20% of the cells. The breaks are frozen in
  [`breaks.json`](breaks.json) and applied unchanged to the monthly maps and to every projected
  period, so class areas can change over time.
- **No calibration.** Probabilities are used as the models give them. Projected changes are
  therefore expressed in classes: the class change of a cell (future minus baseline class,
  with the frozen breaks) and the share of the country in the High and Very high classes.

### Steps of the model training (stage 2)

| Step | Scripts | What it does |
|---|---|---|
| 00 | `00_split_data.py` | Splits the cell-months into train, test and validation; writes the model matrices of each subset |
| 01 | `01_model_testing*.py` | Eight algorithms with default settings and balanced class weights (logistic regression, decision tree, random forest, gradient boosting, XGBoost, extra trees, k-nearest neighbours, naive Bayes), fitted on train and scored on test. Selected: extra trees, XGBoost, random forest (`model_training.models`) |
| 02 | `02_resampling*.py` | Six samplers on the training cell-months of each selected algorithm (random over- and undersampling, SMOTE, ADASYN, Tomek links, neighbourhood cleaning rule), compared with class weights. Selected per algorithm: `model_training.samplers` |
| 03 | `03_hyperparameter_tuning*.py` | Random search: the defaults and 100 configurations per algorithm, threefold cross-validation over the training cell-months, scored by peak MCC; the selected configuration is refitted |
| 04 | `04_final_models*.py` | Final models fitted on the whole train subset and saved; decision threshold; validation scores with 95% bootstrap intervals, top-k coverage and permutation importance of predictor groups |
| 04b | `04b_cv_test*.py` | Cross-validation test: the final configurations refitted on folds of all cell-months of the training years, by year (leave-one-year-out) and by 30 km spatial blocks; out-of-fold AP, ROC-AUC and top-k coverage, compared with a baseline from the fire records alone (kernel density of the training fires times their seasonal cycle); spatial autocorrelation of the burned cells |
| 05 | `05_risk_maps*.py` | Susceptibility of each cell, frozen class breaks (`breaks.json`), classes per model and month, agreement between the models, share of the validation fires per class, maps |

Each step is split across scripts: `<step>.py` fits and predicts, `<step>_evaluate.py` scores
the predictions into tables, and `<step>_plot.py` draws the figures from those tables.

## Repository layout

```
.
├── README.md
├── LICENSE                  MIT License
├── CITATION.cff             citation metadata of this code
├── config.toml              all paths and method settings
├── environment.yml          conda environment, with the versions used for the paper
├── data_manifest.csv        size and SHA-256 checksum of every input file
├── breaks.json              frozen susceptibility-class breaks of the final models
├── src/
│   ├── 00_preprocessing/            stage 0: monthly climate, training table
│   ├── 01_data_exploration/         stage 1: maps, trends, correlation
│   ├── 02_model_training/           stage 2: steps 00–05
│   ├── 03_future_classification/    stage 3: projections
│   ├── utils/                       modules shared by the stages
│   ├── write_data_manifest.py       writes or verifies data_manifest.csv
│   └── sbatch_write_data_manifest.sh
│
│   not in the repository (created when the pipeline runs):
├── data/                    the data directory (or a symbolic link to it), see Input data
├── results/                 tables and figures
└── logs/                    SLURM logs
```

Modules in `src/utils/`:

| Module | Content |
|---|---|
| `load_config.py` | Reads `config.toml` and resolves its paths (`from load_config import cfg`) |
| `chelsa_grid.py` | The analysis grid and the set of cells |
| `predictors.py` | The predictors of one GCM-year from the monthly climate and the static layers; shared by the training table and the projections |
| `climate_derived.py` | Potential evapotranspiration (Priestley–Taylor) and climatic water balance |
| `model_data.py` | Layout and loading of the split and the model matrices |
| `models.py` | The eight candidate algorithms and their default settings |
| `resampling.py` | The samplers, applied to cell-months (with a self-test) |
| `tuning.py` | Configurations of the random search, folds, cross-validation scores |
| `fitting.py` | Fitting and prediction, shared by steps 03–04b and stage 3 |
| `cv_folds.py` | Data and folds of the cross-validation test (step 04b): the three subsets combined, year and spatial-block folds, spatial autocorrelation |
| `kde_baseline.py` | The fire-record baseline of step 04b: kernel density of the training fires times their seasonal cycle |
| `classification_metrics.py` | MCC curve over all thresholds, average precision, ROC-AUC, top-k coverage |
| `risk_thresholds.py` | Quintile breaks, classification, `load_breaks()` |
| `resources.py` | CPUs of the SLURM allocation, peak memory |
| `figures.py`, `maps.py` | Figure style (Copernicus guidelines) and map helpers |

## Installation

```bash
conda env create -f environment.yml
conda activate forestFireGT
```

The environment pins Python 3.13 and the package versions used for the paper. The saved
models (joblib files) need the same scikit-learn and XGBoost versions to load.

The full pipeline needs a computing cluster:

- **Data.** The daily climate input alone is 3.4 TB; the generated data take about 150 GB more.
- **Memory.** The largest jobs need about 110 GB (extra trees with SMOTE, the fold fits of
  step 04b).
- **Model size.** The extra-trees final model is 19.7 GB on disk.
- **Scheduler.** The paper runs used SLURM; see [Running the pipeline](#running-the-pipeline).

## Configuration

All paths and method settings are in [`config.toml`](config.toml), read through
`src/utils/load_config.py`. No script takes a path as an argument.

- **Data directory.** `paths.data_dir = "data"`, relative to the repository root. Either create
  it as a symbolic link (`ln -s /path/to/data data`) or set the environment variable
  `FFGT_DATA_DIR`. Inputs, large intermediate files, predictions and fitted models live here.
  Paths recorded in output files (`breaks.json`, `data_manifest.csv`, the model and split
  metadata) are relative to it, so the data directory can be moved.
- **Results.** Tables and figures go to `results/`, which the scripts create.
- **Method settings.** These are in the same file: training years, GCMs, periods, split
  fractions and seeds, selected algorithms and samplers, search spaces, settings of the final
  models, class percentiles, figure style.

The SLURM wrappers keep their own list of the selected algorithms (`MODELS=(...)`). It must
match `model_training.models`.

## Input data

The input data are not part of this repository. The pipeline expects them under the data
directory, at the paths of `[paths.raw]` and `[paths.inputs]` in `config.toml`:

| Input | Path under the data directory | Content |
|---|---|---|
| Fire records | `rawdata/histfires/histfires_1kmGT_2017-2024.csv` | Cell-months with at least one forest-fire report: lat, lon, year, month, n_reports, on_grid |
| Daily climate | `rawdata/chelsaGT_raw/menz_output/` | Daily data of each GCM, historical and SSP scenarios: pr, rsds, tas, tasmax, tasmin (file pattern `files.climate_daily_gcm`) |
| Static layers | `rawdata/static_vars/` | CSV files with lat, lon and the value: elevation, slope, aspect, distances to roads, urban areas and water bodies (30''); aspect sine and cosine (1/112°) (`[preprocessing.static_files]`) |
| Boundaries | `boundaries_shp/ne_10m_admin_1_states_provinces/` | Natural Earth admin-1 boundaries, for the maps only |

[`data_manifest.csv`](data_manifest.csv) lists the size and SHA-256 checksum of every input
file used for the paper (28,096 files), including the monthly climate and the training table
built from them. The daily climate is recorded by size only. To compare a data directory with
the manifest:

```bash
bash src/sbatch_write_data_manifest.sh verify      # or: python src/write_data_manifest.py verify
```

## Running the pipeline

Each stage has SLURM wrappers (`sbatch_*.sh`) that submit its Python scripts with the memory,
CPUs and time limits measured on the paper runs (noted in comments). They were written for the
PIK cluster (SLURM account, partition, QOS, `ml miniforge`, `source activate forestFireGT`).
For another cluster, adapt the `#SBATCH` lines in the `submit` function of each wrapper.

A wrapper prints its usage when called without arguments. The Python scripts can also be run
directly (`python src/<stage>/<script>.py`); their usage is in the docstring at the top of each
script. All commands below run from the repository root.

Options shared by most scripts and wrappers:

- **`--force`** recomputes output that already exists. Without it, existing output is kept, so
  an interrupted run can be restarted.
- **`--after JOB_ID`** starts a job only after another has finished successfully.
- **`--trial`** (steps 01–03) runs on 1% of the data, with output in separate `*_trial`
  folders, to time a run before the full one.

### Stage 0: preprocessing

```bash
bash src/00_preprocessing/sbatch_00_aggregate_monthly_gcm.sh               # one task per GCM and scenario
bash src/00_preprocessing/sbatch_01_build_training_set.sh <aggregation_job_id>
```

### Stage 1: data exploration

```bash
bash src/01_data_exploration/sbatch_data_exploration.sh    # fire map, predictor maps, climate trends, correlation
```

### Stage 2: model training

```bash
S=src/02_model_training
bash $S/sbatch_split_data.sh
bash $S/sbatch_model_testing.sh all                # 8 algorithms, then the evaluation
python $S/01_model_testing_plot.py
bash $S/sbatch_resampling.sh all all               # 3 algorithms x 6 samplers, then the evaluation
python $S/02_resampling_plot.py
bash $S/sbatch_hyperparameter_tuning.sh all        # folds and resampling, then the CV fits
bash $S/sbatch_hyperparameter_tuning.sh refit all  # once every CV fit is scored; then the evaluation
python $S/03_hyperparameter_tuning_plot.py
bash $S/sbatch_final_models.sh all                 # fit, check, importance, export; then the evaluation
python $S/04_final_models_plot.py
bash $S/sbatch_cv_test.sh all                      # folds, fold fits and baselines, autocorrelation; then the evaluation
bash $S/sbatch_risk_maps.sh all                    # susceptibility, breaks.json, classes; then the maps
```

Two choices are made between steps and recorded in `config.toml`: the algorithms after step
01 (`model_training.models`) and the correction of the class imbalance after step 02
(`model_training.samplers`). The file holds those of the paper. The `_plot.py` scripts of
steps 01–04 read only the result tables and run in seconds.

Step 04b needs the final models' settings (`models/<model>.json`, step 04) and, for its
comparison table, the validation scores of `04_final_models_evaluate.py`. Its commands can
also be submitted one at a time (`bash $S/sbatch_cv_test.sh` prints them): `folds`;
`fit <model|all> <loyo|blocks|all>` (one task per fold); `baseline <loyo|blocks|validation|all>`;
`autocorrelation`; `evaluate`. The 11 fold fits of extra trees take about 25 min and 110 GB
each.

### Stage 3: projections

```bash
F=src/03_future_classification
bash $F/sbatch_future_classification.sh predict all                  # per model: 9 GCMs x 4 scenarios
bash $F/sbatch_future_classification.sh evaluate --after <predict_job_ids>
bash $F/sbatch_future_classification.sh plot --after <evaluate_job_id>
```

Stage 3 needs the saved final models (step 04), `breaks.json` (step 05), the monthly climate
(stage 0) and the static layers.

## Outputs

### Data directory (large files)

| Path | Content | Written by |
|---|---|---|
| `rawdata/processed/climate_monthly_gcm/` | Monthly climate per GCM, variable, scenario and year (NetCDF) | `00_aggregate_monthly_gcm.py` |
| `indata/training/allvars_2018-2024.parquet` | Training table | `01_build_training_set.py` |
| `indata/training/split_index/` | Split and model matrices of each subset (see `utils/model_data.py`) | `00_split_data.py` |
| `predictions/02_model_training/` | Predictions of steps 01–04 (`.npy`); validation predictions of the final models with cell, month, GCM and fire label (`<model>_validation.parquet`); permutation importance | steps 01–04 |
| `predictions/02_model_training/cv_test/` | Cross-validation test: the three subsets combined (`combined/`, 4.8 GB), the cell-month table, the folds, the out-of-fold predictions of each model and fold (`oof/`) and the baseline scores (`baselines/`); no fold model is saved | `04b_cv_test.py` |
| `models/` | Final models (`<model>.joblib`) and their settings, decision threshold and test scores (`<model>.json`) | `04_final_models.py fit` |
| `predictions/03_future_classification/` | Fire probability per model, GCM and scenario (`<model>/<gcm>_<scenario>.npy`: years x 12 months x cells, float32) and the cell list (`cells.npy`) | `01_future_classification.py` |

### `results/` (tables and figures)

| Folder | Content |
|---|---|
| `01_data_exploration/` | Fire cells, per-cell predictor means, national climate means and changes, correlation matrix |
| `02_model_training/01_model_testing/` … `05_risk_maps/` | Per step: scores, threshold curves, cross-validation results, selected hyperparameters, validation scores, top-k coverage, importance; susceptibility and classes of each cell and calendar month, validation fires per class |
| `02_model_training/04b_cv_test/` | Cross-validation test: the comparison of the models and the baseline on the validation subset, leave-one-year-out and spatial blocks (`comparison.csv`), the scores per fold or year, fold sizes, the deal of the blocks, the spatial autocorrelation of the burned cells |
| `03_future_classification/` | Share of the country in the High and Very high classes per year (`national_yearly.csv`) and per period (`high_risk_shares.csv`), class shares (`class_shares.csv`), shares of the class changes (`class_changes.csv`), baseline class, class and class change per cell (`cells.parquet`) and per cell and calendar month (`cell_months.parquet`, 2081–2100), period means per cell (`period_means.parquet`) |
| `mainfigs/`, `appendix/` | Figures of the paper and its appendix (PDF) |

The figures and the scripts that draw them:

| Figure | Script |
|---|---|
| `mainfigs/00_fire_records_map.pdf` | `01_data_exploration/01_fire_records_map.py` |
| `mainfigs/01_predictor_maps.pdf`, `appendix/01_predictor_maps_climate.pdf`, `appendix/01_predictor_maps_static.pdf` | `01_data_exploration/02_predictor_maps.py` |
| `mainfigs/01_climate_trends.pdf` | `01_data_exploration/03_climate_trends.py` |
| `appendix/01_correlation_spearman.pdf` | `01_data_exploration/04_correlation.py` |
| `mainfigs/02_model_testing.pdf` | `02_model_training/01_model_testing_plot.py` |
| `mainfigs/02_resampling.pdf` | `02_model_training/02_resampling_plot.py` |
| `mainfigs/02_hyperparameter_tuning.pdf` | `02_model_training/03_hyperparameter_tuning_plot.py` |
| `mainfigs/02_final_models_validation.pdf`, `appendix/02_final_models_importance.pdf` | `02_model_training/04_final_models_plot.py` |
| `mainfigs/02_risk_maps.pdf`, `appendix/02_risk_maps_monthly_<model>.pdf` | `02_model_training/05_risk_maps_plot.py` |
| `mainfigs/03_change_maps.pdf`, `mainfigs/03_high_risk_share.pdf`, `mainfigs/03_classes_monthly.pdf`, `appendix/03_change_monthly_<model>_<scenario>.pdf` | `03_future_classification/01_future_classification_plot.py` |

### `breaks.json`

[`breaks.json`](breaks.json) holds the frozen class breaks of the three final models, written
by `05_risk_maps.py` and read by stage 3 (`utils/risk_thresholds.load_breaks()`). It is kept in
the repository so the classes of the published models can be reproduced without rerunning stage
2. For each model it gives four ascending breaks of the susceptibility, the cells per class and
the prediction files the breaks were computed from (relative to the data directory). A value's
class is 1 plus the number of breaks below it, so a value equal to a break falls in the lower
class. Classes 1–5 are labelled Very low, Low, Medium, High and Very high.

## Using the final models

The final models are scikit-learn or XGBoost classifiers saved with joblib (`models/` in the
data directory). They take the 14 predictors as float32 columns in the order of
`data.features` in `config.toml`:

```python
import joblib
model = joblib.load("data/models/xgboost.joblib")
prob = model.predict_proba(X)[:, 1]      # fire probability of each row of X
```

`utils/predictors.predictor_matrix()` builds `X` for every cell and month of one GCM-year from
the monthly climate. The classes of `breaks.json` apply to the susceptibility (the mean
probability over the GCMs and the months of a period), not to the probability of a single
cell-month.

## Reproducibility

- **Seeds.** All are fixed:
  - split and folds: `model_training.seed` (107);
  - deal of the spatial blocks of step 04b: `model_training.cv_test.seed` (107);
  - algorithms and samplers: 42 (`utils/models.py`, `utils/resampling.py`);
  - draws of the random search: `model_training.tuning.seed` (42);
  - bootstrap and permutations: `model_training.final_models.seed` (42).
- **Built-in checks.**
  - The model matrices are checked against the training table and `config.toml` before every
    use.
  - `04_final_models.py fit --check` compares the test predictions of the final models with
    those of the same configuration in steps 01–03.
  - Saved models are reloaded and their predictions compared with those of the fitted models.
  - `python src/utils/resampling.py --test` checks that the cell-month SMOTE and ADASYN
    reproduce imbalanced-learn.
- **Logs.** Every job writes its log to `logs/`; the computing scripts end it with their run
  time and peak memory.

## Citation

If you use this code, please cite the paper:

> Pineda Schwarz, D., Hattermann, F. F., and Walther, D.: Mapping present and future forest fire
> susceptibility in Guatemala: machine learning projections to 2100, *journal, volume, pages,
> DOI, year (to be added)*.

and this code:

> Pineda Schwarz, D.: Mapping present and future forest fire susceptibility in Guatemala: machine
> learning projections to 2100, Zenodo [code], https://doi.org/10.5281/zenodo.24080892, 2026.

This DOI always resolves to the latest version. The version used for the paper, v1.0.0, has
its own DOI: https://doi.org/10.5281/zenodo.24080893. The citation metadata of the code are in
[`CITATION.cff`](CITATION.cff).

## License

The code is released under the MIT License, see [`LICENSE`](LICENSE).

## Reference

Verde, J. C. and Zêzere, J. L.: Assessment and validation of wildfire susceptibility and hazard
in Portugal, Nat. Hazards Earth Syst. Sci., 10, 485–497,
https://doi.org/10.5194/nhess-10-485-2010, 2010.
