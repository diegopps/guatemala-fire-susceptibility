"""
01_model_testing.py
-------------------
Fits one candidate algorithm, with default settings, on the train subset and predicts
the fire probability of every test row (all GCMs). 01_model_testing_evaluate.py scores
the predictions.

Candidates: the 8 algorithms of models.py (logistic regression, decision tree, random
forest, gradient boosting, XGBoost, extra trees, k-nearest neighbours and Gaussian naive
Bayes), each with balanced class weights where the algorithm supports them; logistic
regression and k-nearest neighbours standardize the predictors first.

Input:  model data of 00_split_data.py (see utils/model_data.py)
Output: model_training.paths.generated.model_testing/<model>.npy — fire probability of each
        test row (float32, in the row order of the test subset)
        results/<model_training model_testing>/<model>/
          model_info.json         settings, train size, fit and prediction times
          feature_importance.csv  impurity-based importance (tree models) or coefficients
                                  of the standardized predictors (logistic regression)
Existing output is kept unless --force is given.

Prediction in parts (used for k-nearest neighbours): --part I --parts N fits the model
and predicts only the I-th of N blocks of test rows (<model>.part<I>of<N>.npy). The
evaluation joins the parts once all of them exist.

--trial fits on 1% of the train rows and predicts 1% of the test rows (drawn at random
with seed model_training.seed), to time a model before the full run. Its output goes to
folders named <...>_trial and is never evaluated.

Usage:
    python 01_model_testing.py <model> [--part I --parts N] [--trial] [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from resources import n_cpus, report_usage
import model_data
from models import MODELS, build

TRIAL_FRACTION = 0.01
PREDICT_BLOCK = 1_000_000   # test rows per predict_proba call


def prediction_file(folder, model, part=None, parts=None):
    name = model if part is None else f"{model}.part{part}of{parts}"
    return folder / f"{name}.npy"


def feature_importance(model):
    """Impurity-based importance, or coefficients of the standardized predictors; None otherwise."""
    final = model[-1] if hasattr(model, "steps") else model
    if hasattr(final, "feature_importances_"):
        return pd.DataFrame({"feature": cfg.data.features, "importance": final.feature_importances_})
    if hasattr(final, "coef_"):
        return pd.DataFrame({"feature": cfg.data.features, "coefficient": final.coef_[0]})
    return None


# =============================================================================
# MAIN
# =============================================================================

def save_npy(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.rename(path)


def main(model_name, part, parts, trial, force):
    pred_dir = cfg.model_training.paths.generated.model_testing
    res_dir = cfg.model_training.paths.results.model_testing / model_name
    if trial:
        pred_dir = pred_dir.with_name(pred_dir.name + "_trial")
        res_dir = res_dir.parent.with_name(res_dir.parent.name + "_trial") / model_name
    out = prediction_file(pred_dir, model_name, part, parts)
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    cpus = n_cpus()
    rng = np.random.default_rng(cfg.model_training.seed)

    print(f"Model: {model_name}" + (" (trial)" if trial else ""))
    train = model_data.load_subset("train", arrays=("X", "y"))
    rows = np.arange(len(train["y"]))
    if trial:
        rows = np.sort(rng.choice(rows, int(TRIAL_FRACTION * rows.size), replace=False))
    X, y = np.ascontiguousarray(train["X"][rows]), np.asarray(train["y"][rows])
    n_train, n_train_fire = len(y), int(y.sum())
    print(f"  train rows: {n_train:,} ({n_train_fire:,} fire), {cpus} CPUs")

    model, fit_kwargs = build(model_name, y, cpus)
    t0 = time.monotonic()
    model.fit(X, y, **fit_kwargs)
    fit_seconds = time.monotonic() - t0
    print(f"  fitted in {fit_seconds / 60:.1f} min")
    del X, y, fit_kwargs

    test_X = model_data.load_subset("test", arrays=("X",))["X"]
    rows = np.arange(len(test_X))
    if trial:
        rows = np.sort(rng.choice(rows, int(TRIAL_FRACTION * rows.size), replace=False))
    if part is not None:
        rows = np.array_split(rows, parts)[part - 1]
    prob = np.empty(rows.size, dtype=np.float32)
    t0 = time.monotonic()
    for start in range(0, rows.size, PREDICT_BLOCK):
        block = rows[start:start + PREDICT_BLOCK]
        prob[start:start + block.size] = model.predict_proba(np.asarray(test_X[block]))[:, 1]
    predict_seconds = time.monotonic() - t0
    print(f"  predicted {rows.size:,} test rows in {predict_seconds / 60:.1f} min")
    save_npy(out, prob)
    print(f"  Saved: {out}")

    if part in (None, 1):
        info = {"model": model_name, "estimator": repr(model), "trial": trial,
                "train_rows": n_train, "train_fire_rows": n_train_fire,
                "cpus": cpus, "fit_minutes": round(fit_seconds / 60, 2),
                "predict_minutes": round(predict_seconds / 60, 2),
                "predicted_rows": int(rows.size), "part": part, "parts": parts}
        (res_dir / "model_info.json").write_text(json.dumps(info, indent=2))
        importance = feature_importance(model)
        if importance is not None:
            importance.to_csv(res_dir / "feature_importance.csv", index=False)
        print(f"  Saved: {res_dir}/")
    report_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--part", type=int, help="block of test rows to predict (1..parts)")
    parser.add_argument("--parts", type=int, help="number of blocks of test rows")
    parser.add_argument("--trial", action="store_true", help="1%% of the rows, separate output")
    parser.add_argument("--force", action="store_true", help="recompute existing output")
    a = parser.parse_args()
    if (a.part is None) != (a.parts is None) or (a.part is not None and not 1 <= a.part <= a.parts):
        parser.error("--part I and --parts N go together, with 1 <= I <= N.")
    main(a.model, a.part, a.parts, a.trial, a.force)
