"""
fitting.py
----------
Fitting and prediction of a configuration of the selected algorithms, shared by the steps of
the model training that fit tuned configurations (03_hyperparameter_tuning.py and
04_final_models.py in src/02_model_training).

A fit trains on a set of training cell-months, resampled with the sampler of the model
(model_training.samplers; utils/resampling.py) or, with "class_weights", all of them once
with balanced class weights (utils/tuning.py). Predictions are the fire probability of each
row (float32).

Usage:
    import fitting
    rows, res = fitting.training_data("extra_trees", train, cell_months, cpus)
    model, info = fitting.fit("extra_trees", config, train, rows, res, cpus)
    prob = fitting.predict_rows(model, test["X"], np.arange(len(test["y"])))
"""

import time

import numpy as np

import model_data
import resampling
import tuning

PREDICT_BLOCK = 1_000_000   # rows per predict_proba call


def training_data(model_name, train, cell_months, cpus):
    """(rows, res) of a fit on some training cell-months (indices into the cell-months of
    model_data.cell_month_rows): rows[g, c] the row of each of these cell-months in GCM g,
    and their resampling with the sampler of the model (all of them once with class
    weights)."""
    sampler = tuning.sampler(model_name)
    if sampler == tuning.CLASS_WEIGHTS:
        rows, _ = model_data.cell_month_rows(train)
        return rows[:, cell_months], resampling.Resampled(np.arange(cell_months.size))
    table = resampling.cell_month_table(train, cell_months)
    return table.rows, resampling.resample(sampler, table, cpus)


def fit(model_name, config, train, rows, res, cpus):
    """Fits a configuration on the resampled training cell-months (rows: their rows in each
    GCM); returns the model, the training size and the fit time."""
    X, y = resampling.training_rows(train, rows, res)
    model, fit_kwargs = tuning.estimator(model_name, config, y, cpus)
    t0 = time.monotonic()
    model.fit(X, y, **fit_kwargs)
    info = {"train_rows": len(y), "train_fire_rows": int(y.sum()), "cpus": cpus,
            "fit_minutes": round((time.monotonic() - t0) / 60, 3)}
    return model, info


def predict_rows(model, X_all, rows):
    """Fire probability of the given rows of a predictor matrix (float32)."""
    prob = np.empty(rows.size, dtype=np.float32)
    for start in range(0, rows.size, PREDICT_BLOCK):
        block = rows[start:start + PREDICT_BLOCK]
        prob[start:start + block.size] = model.predict_proba(np.asarray(X_all[block]))[:, 1]
    return prob


def tree_nodes(model):
    """Nodes of all trees of a forest (the size of the fitted model); None otherwise."""
    if hasattr(model, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    return None
