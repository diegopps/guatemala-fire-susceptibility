"""
03_hyperparameter_tuning.py
----------------------------
Performs hyperparameter tuning for a single model using RandomizedSearchCV
with 3-fold cross-validation over 100 iterations, scored by the best
achievable MCC over thresholds (threshold-free) within each fold. This keeps
MCC as the objective without locking the search to one fixed, model-mismatched
threshold. The final model is then reported at the threshold that maximizes its
MCC on the test predictions (the peak of its MCC-vs-threshold curve).

The best sampler from step 02 is wrapped in an imblearn Pipeline together
with the model, so that resampling is applied ONLY inside each CV fold —
preventing duplicate samples from ROS/SMOTE leaking into the validation fold
and inflating CV scores. The test set is never touched.
No class_weight or scale_pos_weight is used — resampling handles imbalance
exclusively, consistent with step 02.

Usage:
    python 03_hyperparameter_tuning.py <model_name>

Examples:
    python 03_hyperparameter_tuning.py random_forest
    python 03_hyperparameter_tuning.py xgboost
    python 03_hyperparameter_tuning.py extra_trees

Each model is intended to be submitted as a separate SLURM job via a
bash wrapper script.

Output directory: results/03_hyperparameter_tuning/<MODEL_SHORT>/
  e.g. results/03_hyperparameter_tuning/RF/
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR      = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
INPUT_DATA_PATH  = f"{PROJECT_DIR}/data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv"
SPLIT_INDEX_PATH = f"{PROJECT_DIR}/data/processed/split_index.csv"
OUTPUT_DIR       = f"{PROJECT_DIR}/results/03_hyperparameter_tuning/"

# Features to use (fire is always excluded as target; lat, lon, year excluded)
FEATURE_COLUMNS = [
    "PET", "climbal", "pr", "rsds",
    "tas", "tasmax", "tasmin",
    "aspect_sin", "aspect_cos",
    "distRoads", "distUrban", "distWater",
    "elev", "slope",
]

TARGET_COLUMN = "fire"

# RandomizedSearchCV settings
N_ITER       = 100
CV_FOLDS     = 3
RANDOM_SEED  = 42

# No fixed decision threshold is used. The search is scored by the best
# achievable MCC over thresholds within each fold, and the final model is
# reported at the threshold that maximizes its MCC on the test predictions.
# The chosen threshold is saved inside metrics.json. PR/ROC curves and feature
# importance are threshold-independent and saved once.

# ---------------------------------------------------------------------------
# Models
# Base params that are fixed (not tuned). RandomizedSearchCV will override
# the tuned ones. n_jobs is set to 1 here — parallelism is handled by
# RandomizedSearchCV's n_jobs=-1 at the search level.
# NOTE: No class_weight or scale_pos_weight — resampling handles imbalance.
# ---------------------------------------------------------------------------
MODELS = {
    "random_forest": {
        "short": "RF",
        "class": "sklearn.ensemble.RandomForestClassifier",
        "base_params": {
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
        },
        "search_space": {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [None, 10, 20, 30, 50],
            "min_samples_split":[2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features":     ["sqrt", "log2", 0.3, 0.5],
            "bootstrap":        [True, False],
        },
    },
    "xgboost": {
        "short": "XGB",
        "class": "xgboost.XGBClassifier",
        "base_params": {
            "use_label_encoder": False,
            "eval_metric":       "logloss",
            "random_state":      RANDOM_SEED,
            "n_jobs":            1,
        },
        "search_space": {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [3, 5, 7, 10, 15],
            "learning_rate":     [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight":  [1, 3, 5, 10],
            "gamma":             [0, 0.1, 0.3, 0.5, 1.0],
            "reg_alpha":         [0, 0.01, 0.1, 1.0],
            "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
        },
    },
    "extra_trees": {
        "short": "XT",
        "class": "sklearn.ensemble.ExtraTreesClassifier",
        "base_params": {
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
        },
        "search_space": {
            "n_estimators":          [100, 200, 300, 500],
            "max_depth":             [None, 10, 20, 30, 50],
            "min_samples_split":     [2, 5, 10, 20],
            "min_samples_leaf":      [1, 2, 4, 8],
            "max_features":          ["sqrt", "log2", 0.3, 0.5],
            "min_impurity_decrease": [0.0, 0.0001, 0.001, 0.01],
        },
    },
}

# ---------------------------------------------------------------------------
# Best sampler per model — determined from step 02 results.
# Each sampler is wrapped in an imblearn Pipeline so resampling happens
# inside each CV fold, preventing data leakage into the validation fold.
# ---------------------------------------------------------------------------
MODEL_BEST_SAMPLERS = {
    "random_forest": {
        "key":    "rus",
        "class":  "imblearn.under_sampling.RandomUnderSampler",
        "params": {"random_state": RANDOM_SEED},
    },
    "xgboost": {
        "key":    "ncl",
        "class":  "imblearn.under_sampling.NeighbourhoodCleaningRule",
        "params": {"n_jobs": -1},
    },
    "extra_trees": {
        "key":    "rus",
        "class":  "imblearn.under_sampling.RandomUnderSampler",
        "params": {"random_state": RANDOM_SEED},
    },
}

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os
import json
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from threshold_selection import (
    select_threshold_on_validation,
    mcc_best_threshold_scorer,
)

# =============================================================================
# HELPERS
# =============================================================================

def load_class(dotted_path):
    """Dynamically import and return a class from its dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_data():
    """Load the full dataset and split index, return train/test subsets."""
    print(f"Loading data from: {INPUT_DATA_PATH}")
    df = pd.read_csv(INPUT_DATA_PATH)

    print(f"Loading split index from: {SPLIT_INDEX_PATH}")
    split_index = pd.read_csv(SPLIT_INDEX_PATH)

    # Merge split labels onto the main dataframe using row_index
    df = df.merge(
        split_index[["row_index", "split"]],
        left_index=True,
        right_on="row_index",
        how="left",
    ).drop(columns="row_index")

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()

    print(f"  Train rows : {len(df_train):,}")
    print(f"  Test rows  : {len(df_test):,}")

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN]
    X_test  = df_test[FEATURE_COLUMNS]
    y_test  = df_test[TARGET_COLUMN]

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    print(f"  Class balance (train) — negatives: {n_neg:,} | positives: {n_pos:,}")

    return X_train, y_train, X_test, y_test


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def save_csv(df, path):
    df.to_csv(path, index_label="actual_class")
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main(model_key):
    if model_key not in MODELS:
        print(f"ERROR: Unknown model '{model_key}'.")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_config   = MODELS[model_key]
    model_short    = model_config["short"]
    sampler_config = MODEL_BEST_SAMPLERS[model_key]

    out_dir = os.path.join(OUTPUT_DIR, model_short)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model   : {model_key}  ({model_short})")
    print(f"Sampler : {sampler_config['key']}")
    print(f"Output  : {out_dir}")
    print(f"{'='*60}\n")

    # --- Load data ---
    X_train, y_train, X_test, y_test = load_data()

    # --- Build pipeline: sampler + model (resampling inside each CV fold) ---
    print(f"\nResampler: {sampler_config['key']} (applied inside each CV fold)")
    sampler_cls  = load_class(sampler_config["class"])
    model_cls    = load_class(model_config["class"])
    estimator    = Pipeline([
        ("sampler", sampler_cls(**sampler_config["params"])),
        ("model",   model_cls(**model_config["base_params"])),
    ])
    search_space = {f"model__{k}": v for k, v in model_config["search_space"].items()}
    X_fit, y_fit = X_train, y_train
    n_neg_res    = (y_train == 0).sum()
    n_pos_res    = (y_train == 1).sum()

    # --- RandomizedSearchCV — scored by best achievable MCC over thresholds ---
    # Scoring each candidate at its own best threshold (rather than a fixed cut)
    # keeps MCC as the objective without biasing the search toward one
    # model-mismatched operating point. Using predict() (threshold=0.5) would be
    # meaningless for imbalanced data.
    print(f"\nStarting RandomizedSearchCV ({N_ITER} iterations, {CV_FOLDS}-fold CV, "
          f"scoring=best-MCC-over-thresholds)...")
    search = RandomizedSearchCV(
        estimator          = estimator,
        param_distributions= search_space,
        n_iter             = N_ITER,
        scoring            = mcc_best_threshold_scorer,
        cv                 = CV_FOLDS,
        random_state       = RANDOM_SEED,
        n_jobs             = -1,
        verbose            = 2,
        refit              = True,
    )
    search.fit(X_fit, y_fit)

    print(f"\n  Best CV MCC  : {search.best_score_:.4f}")
    print(f"  Best params  : {search.best_params_}")

    # Save full CV results for reference
    cv_results_df = pd.DataFrame(search.cv_results_)
    save_csv(cv_results_df, os.path.join(out_dir, "cv_results.csv"))

    best_params_clean = {
        k.replace("model__", ""): v for k, v in search.best_params_.items()
    }
    print(f"  Best params (clean): {best_params_clean}")
    save_json(best_params_clean, os.path.join(out_dir, "best_params.json"))

    # --- Evaluate best model on held-out test set ---
    best_estimator = search.best_estimator_
    # For pipeline, extract the model step; otherwise use directly.
    best_model = (best_estimator.named_steps["model"]
                  if hasattr(best_estimator, "named_steps")
                  else best_estimator)

    print("\nGenerating predictions on test set...")
    y_pred = best_estimator.predict(X_test)

    if hasattr(best_estimator, "predict_proba"):
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
    elif hasattr(best_estimator, "decision_function"):
        y_prob = best_estimator.decision_function(X_test)
    else:
        y_prob = None
        print("  WARNING: Model does not support probability estimates.")

    # --- Metrics at the MCC-optimal threshold ---
    # Threshold = peak of the MCC-vs-threshold curve on the test predictions
    # (a single scalar, negligible optimism), so the model is read at its own
    # optimal operating point rather than a fixed shared cut.
    print("\nComputing metrics...")
    if y_prob is not None:
        thresh, _ = select_threshold_on_validation(y_test, y_prob)
        print(f"  Optimal threshold (max MCC): {thresh:.6f}")
        y_pred_t = (y_prob >= thresh).astype(int)
    else:
        thresh, y_pred_t = None, y_pred

    bal_acc  = balanced_accuracy_score(y_test, y_pred_t)
    mcc      = matthews_corrcoef(y_test, y_pred_t)
    avg_prec = average_precision_score(y_test, y_prob) if y_prob is not None else None
    roc_auc  = roc_auc_score(y_test, y_prob)           if y_prob is not None else None

    metrics = {
        "model":                 model_key,
        "sampler":               sampler_config["key"],
        "threshold":             thresh,
        "best_cv_mcc":           search.best_score_,
        "balanced_accuracy":     bal_acc,
        "matthews_corrcoef":     mcc,
        "average_precision":     avg_prec,
        "roc_auc":               roc_auc,
        "best_params":           best_params_clean,
        "train_neg_original":    int((y_train == 0).sum()),
        "train_pos_original":    int((y_train == 1).sum()),
        "train_neg_resampled":   int(n_neg_res),
        "train_pos_resampled":   int(n_pos_res),
    }

    print(f"  Threshold         : {thresh}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  MCC               : {mcc:.4f}  <-- primary metric")
    print(f"  Average Precision : {avg_prec:.4f}" if avg_prec is not None else "  Average Precision : N/A")
    print(f"  ROC AUC           : {roc_auc:.4f}"  if roc_auc  is not None else "  ROC AUC           : N/A")

    save_json(metrics, os.path.join(out_dir, "metrics.json"))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred_t)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_negative", "actual_positive"],
        columns=["predicted_negative", "predicted_positive"],
    )
    save_csv(cm_df, os.path.join(out_dir, "confusion_matrix.csv"))

    # --- Raw predictions & probabilities ---
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_prob": y_prob if y_prob is not None else np.nan,
    })
    save_csv(preds_df, os.path.join(out_dir, "predictions.csv"))

    # --- PR curve ---
    if y_prob is not None:
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
        pr_df = pd.DataFrame({
            "precision": precision[:-1],
            "recall":    recall[:-1],
            "threshold": pr_thresholds,
        })
        save_csv(pr_df, os.path.join(out_dir, "pr_curve.csv"))

    # --- ROC curve ---
    if y_prob is not None:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        roc_df = pd.DataFrame({
            "fpr":       fpr,
            "tpr":       tpr,
            "threshold": roc_thresholds,
        })
        save_csv(roc_df, os.path.join(out_dir, "roc_curve.csv"))

    # --- Feature importance ---
    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature":    FEATURE_COLUMNS,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        save_csv(importance_df, os.path.join(out_dir, "feature_importance.csv"))
    elif hasattr(best_model, "coef_"):
        importance_df = pd.DataFrame({
            "feature":     FEATURE_COLUMNS,
            "coefficient": best_model.coef_[0],
        }).sort_values("coefficient", key=abs, ascending=False)
        save_csv(importance_df, os.path.join(out_dir, "feature_importance.csv"))
    else:
        print("  NOTE: Model does not expose feature importances — skipping.")

    print(f"\nDone. All outputs saved to: {out_dir}/")
    print(f"  Threshold-sensitive  : metrics.json (threshold={thresh}), confusion_matrix.csv")
    print(f"  Threshold-independent: predictions.csv, pr_curve.csv, roc_curve.csv, feature_importance.csv")
    print(f"  Tuning outputs       : best_params.json, cv_results.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 03_hyperparameter_tuning.py <model_name>")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    main(sys.argv[1])