"""
01_future_classification.py
---------------------------
Fire probability of every cell-month of one GCM and scenario, 1961-2100, from one final model
(04_final_models.py). The predictors are computed straight from the monthly climate archive
with the code of the training set (utils/predictors.py); nothing is classified here. The
frozen class breaks of step 05 are applied later, to the saved predictions.

The SSP3-7.0 rows of the training years (training.years, training.climate_scenario) are the
model's own training rows, so their predictions are in-sample for the rows of the train subset.

Input:  saved model of 04_final_models.py (paths.models_dir/<model>.joblib); monthly climate
        (paths.generated.climate_monthly_gcm); static layers (paths.raw.static_vars)
Output: paths.generated.future_predictions/
          <model>/<gcm>_<scenario>.npy
                     fire probability, float32, shape (years, 12, cells): years = periods.
                     historical for the historical scenario, periods.future for the SSPs;
                     months January to December; cells in the order of cells.npy
          cells.npy  the cell ids, sorted (the order of 05_risk_maps.py); shared by all files,
                     written if missing, checked if present
Existing output is kept unless --force is given.

Threads: at most MAX_THREADS (or --threads N).

Usage:
    python 01_future_classification.py <model> <gcm> <scenario> [--threads N] [--force]
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import reference_grid
from resources import n_cpus, report_usage
import fitting
import predictors

MODELS = cfg.model_training.models
SCENARIOS = [cfg.scenarios.historical] + cfg.scenarios.ssp
OUT = cfg.paths.generated.future_predictions
MAX_THREADS = 16


def save_npy(path, array):
    """Writes to a unique temporary name first: array tasks may write cells.npy at the same time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.rename(path)


def check_cells(ids):
    """Writes the shared cell list if it is missing; an error if it differs from these cells."""
    path = OUT / "cells.npy"
    if not path.exists():
        save_npy(path, ids)
    elif not np.array_equal(np.load(path), ids):
        raise ValueError(f"{path} lists other cells than the static layers.")


def predict(model_name, gcm, scenario, threads, force):
    out = OUT / model_name / f"{gcm}_{scenario}.npy"
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return

    grid = reference_grid()
    cells = predictors.load_static(grid).sort_values("cell_id", ignore_index=True)
    check_cells(cells.cell_id.to_numpy(np.int32))

    cpus = min(n_cpus(), threads)
    t0 = time.monotonic()
    model = joblib.load(cfg.paths.models_dir / f"{model_name}.joblib")
    model.set_params(n_jobs=cpus)
    print(f"Model: {model_name}; loaded in {(time.monotonic() - t0) / 60:.1f} min, {cpus} CPUs")

    years = predictors.scenario_years(scenario)
    prob = np.empty((len(years), predictors.MONTHS, len(cells)), dtype=np.float32)
    print(f"{gcm} {scenario}: {years.start}-{years.stop - 1}, {prob.size:,} rows")
    read_minutes = predict_minutes = 0.0
    for k, year in enumerate(years):
        t0 = time.monotonic()
        X = predictors.predictor_matrix(grid, cells, gcm, year, scenario)
        t1 = time.monotonic()
        rows = X.reshape(-1, X.shape[-1])
        prob[k] = fitting.predict_rows(model, rows, np.arange(len(rows))).reshape(prob.shape[1:])
        read_minutes += (t1 - t0) / 60
        predict_minutes += (time.monotonic() - t1) / 60
        if (k + 1) % 10 == 0 or k + 1 == len(years):
            print(f"  {year}: {k + 1}/{len(years)} years; read {read_minutes:.1f} min, "
                  f"predicted {predict_minutes:.1f} min", flush=True)

    save_npy(out, prob)
    print(f"  Saved: {out} ({out.stat().st_size / 1e6:,.0f} MB)")
    report_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("gcm", choices=cfg.gcms.models)
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--threads", type=int, default=MAX_THREADS,
                        help=f"maximum threads of the algorithm (default {MAX_THREADS})")
    parser.add_argument("--force", action="store_true", help="recompute existing output")
    a = parser.parse_args()
    predict(a.model, a.gcm, a.scenario, a.threads, a.force)
