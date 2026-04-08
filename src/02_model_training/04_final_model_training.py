"""
04_final_model_training.py
---------------------------
Trains the final models using the best
hyperparameters found in step 03, with the best sampler from step 02
applied to the training set.

  1. Loads best_params.json from step 03 results.
  2. Resamples the training set using the best sampler (same as step 03).
  3. Retrains the model from scratch on the resampled training set using
     the best hyperparameters.
  4. Saves the fitted model as a .joblib file under models/.
  5. Generates predictions on both the test set and the validation set
     (the validation set is used here for the first time). 
  6. Saves all outputs (predictions, metrics, PR/ROC curves, confusion
     matrix, feature importance) separately for each subset.

Usage:
    python 04_final_model_training.py <model_name>

Examples:
    python 04_final_model_training.py random_forest
    python 04_final_model_training.py xgboost
    python 04_final_model_training.py extra_trees

Each model is intended to be submitted as a separate SLURM job via a
bash wrapper script.

Output directory: results/04_final_model_training/<MODEL_SHORT>/
  e.g. results/04_final_model_training/RF/
Trained models : models/<MODEL_SHORT>_final.joblib
  e.g. models/RF_final.joblib
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR      = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
INPUT_DATA_PATH  = f"{PROJECT_DIR}/data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv"
SPLIT_INDEX_PATH = f"{PROJECT_DIR}/data/processed/split_index.csv"
OUTPUT_DIR       = f"{PROJECT_DIR}/results/04_final_model_training/"
TUNING_DIR       = f"{PROJECT_DIR}/results/03_hyperparameter_tuning/"
MODELS_DIR       = f"{PROJECT_DIR}/models/"

# Features to use (fire is always excluded as target; lat, lon, year excluded)
FEATURE_COLUMNS = [
    "PET", "climbal", "pr", "rsds",
    "tas", "tasmax", "tasmin",
    "aspect_sin", "aspect_cos",
    "distRoads", "distUrban", "distWater",
    "elev", "slope",
]

TARGET_COLUMN = "fire"

RANDOM_SEED = 42

# Custom threshold per model — same as step 03.
# Used for threshold-sensitive outputs (metrics, confusion matrix).
# PR/ROC curves and feature importance are threshold-independent.
MODEL_THRESHOLDS = {
    "random_forest": 0.03,
    "xgboost":       0.01,
    "extra_trees":   0.03,
}

# ---------------------------------------------------------------------------
# Models
# base_params are fixed (not tuned). Best hyperparameters from step 03
# are loaded from best_params.json and merged in at runtime.
# NOTE: No class_weight or scale_pos_weight — resampling handles imbalance.
# ---------------------------------------------------------------------------
MODELS = {
    "random_forest": {
        "short":       "RF",
        "class":       "sklearn.ensemble.RandomForestClassifier",
        "base_params": {
            "random_state": RANDOM_SEED,
            "n_jobs":       -1,
        },
    },
    "xgboost": {
        "short":       "XGB",
        "class":       "xgboost.XGBClassifier",
        "base_params": {
            "use_label_encoder": False,
            "eval_metric":       "logloss",
            "random_state":      RANDOM_SEED,
            "n_jobs":            -1,
        },
    },
    "extra_trees": {
        "short":       "XT",
        "class":       "sklearn.ensemble.ExtraTreesClassifier",
        "base_params": {
            "random_state": RANDOM_SEED,
            "n_jobs":       -1,
        },
    },
}

# ---------------------------------------------------------------------------
# Best sampler per model — same as step 03.
# ---------------------------------------------------------------------------
MODEL_BEST_SAMPLERS = {
    "random_forest": {
        "key":    "ros",
        "class":  "imblearn.over_sampling.RandomOverSampler",
        "params": {"random_state": RANDOM_SEED},
    },
    "xgboost": {
        "key":    "ncl",
        "class":  "imblearn.under_sampling.NeighbourhoodCleaningRule",
        "params": {"n_jobs": -1},
    },
    "extra_trees": {
        "key":    "svmsmote",
        "class":  "imblearn.over_sampling.SVMSMOTE",
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
import joblib
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
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
    """Load the full dataset and split index, return train/test/validation subsets."""
    print(f"Loading data from: {INPUT_DATA_PATH}")
    df = pd.read_csv(INPUT_DATA_PATH)

    print(f"Loading split index from: {SPLIT_INDEX_PATH}")
    split_index = pd.read_csv(SPLIT_INDEX_PATH)

    df = df.merge(
        split_index[["row_index", "split"]],
        left_index=True,
        right_on="row_index",
        how="left",
    ).drop(columns="row_index")

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    df_val   = df[df["split"] == "validation"].copy()

    # Save mapping data: coords and time, for future results plotting.
    mapping_train = df_train[["lat", "lon", "month", "year"]].reset_index(drop=True)
    mapping_test  = df_test[["lat", "lon", "month", "year"]].reset_index(drop=True)
    mapping_val   = df_val[["lat", "lon", "month", "year"]].reset_index(drop=True)

    print(f"  Train rows      : {len(df_train):,}")
    print(f"  Test rows       : {len(df_test):,}")
    print(f"  Validation rows : {len(df_val):,}")

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN]
    X_test  = df_test[FEATURE_COLUMNS]
    y_test  = df_test[TARGET_COLUMN]
    X_val   = df_val[FEATURE_COLUMNS]
    y_val   = df_val[TARGET_COLUMN]

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    print(f"  Class balance (train) — negatives: {n_neg:,} | positives: {n_pos:,}")

    return X_train, y_train, X_test, y_test, X_val, y_val, mapping_train, mapping_test, mapping_val


def load_best_params(model_short):
    """Load best hyperparameters from step 03 output."""
    path = os.path.join(TUNING_DIR, model_short, "best_params.json")
    if not os.path.exists(path):
        print(f"ERROR: best_params.json not found at {path}")
        sys.exit(1)
    with open(path, "r") as f:
        params = json.load(f)
    print(f"  Loaded best params from: {path}")
    return params


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def save_csv(df, path):
    df.to_csv(path, index_label="actual_class")
    print(f"  Saved: {path}")


def evaluate_and_save(model, X, y, mapping, thresh, subset_label, out_dir):
    """
    Run predictions for a given subset, compute all metrics and save all
    outputs. subset_label is used as a filename suffix (e.g. 'test', 'val').
    """
    print(f"\n  --- Evaluating on {subset_label} set ---")

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = None
        print("  WARNING: Model does not support probability estimates.")

    # --- Threshold-sensitive metrics ---
    y_pred_t = (y_prob >= thresh).astype(int) if y_prob is not None else y_pred

    bal_acc  = balanced_accuracy_score(y, y_pred_t)
    mcc      = matthews_corrcoef(y, y_pred_t)
    avg_prec = average_precision_score(y, y_prob) if y_prob is not None else None
    roc_auc  = roc_auc_score(y, y_prob)           if y_prob is not None else None

    metrics = {
        "subset":            subset_label,
        "threshold":         thresh,
        "balanced_accuracy": bal_acc,
        "matthews_corrcoef": mcc,
        "average_precision": avg_prec,
        "roc_auc":           roc_auc,
    }

    print(f"    Threshold         : {thresh}")
    print(f"    Balanced Accuracy : {bal_acc:.4f}")
    print(f"    MCC               : {mcc:.4f}  <-- primary metric")
    print(f"    Average Precision : {avg_prec:.4f}" if avg_prec is not None else "    Average Precision : N/A")
    print(f"    ROC AUC           : {roc_auc:.4f}"  if roc_auc  is not None else "    ROC AUC           : N/A")

    save_json(metrics, os.path.join(out_dir, f"metrics_{thresh}_{subset_label}.json"))

    # --- Confusion matrix ---
    cm = confusion_matrix(y, y_pred_t)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_negative", "actual_positive"],
        columns=["predicted_negative", "predicted_positive"],
    )
    save_csv(cm_df, os.path.join(out_dir, f"confusion_matrix_{thresh}_{subset_label}.csv"))

    # --- Raw predictions & probabilities ---
    preds_df = pd.DataFrame({
        "y_true": y.values,
        "y_pred": y_pred,
        "y_prob": y_prob if y_prob is not None else np.nan,
        "lat":    mapping["lat"].values,
        "lon":    mapping["lon"].values,
        "month":    mapping["month"].values,
        "year":    mapping["year"].values,

    })
    save_csv(preds_df, os.path.join(out_dir, f"predictions_{subset_label}.csv"))

    # --- PR curve ---
    if y_prob is not None:
        precision, recall, pr_thresholds = precision_recall_curve(y, y_prob)
        pr_df = pd.DataFrame({
            "precision": precision[:-1],
            "recall":    recall[:-1],
            "threshold": pr_thresholds,
        })
        save_csv(pr_df, os.path.join(out_dir, f"pr_curve_{subset_label}.csv"))

    # --- ROC curve ---
    if y_prob is not None:
        fpr, tpr, roc_thresholds = roc_curve(y, y_prob)
        roc_df = pd.DataFrame({
            "fpr":       fpr,
            "tpr":       tpr,
            "threshold": roc_thresholds,
        })
        save_csv(roc_df, os.path.join(out_dir, f"roc_curve_{subset_label}.csv"))

    # --- Feature importance ---
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature":    FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        save_csv(importance_df, os.path.join(out_dir, "feature_importance.csv"))
    elif hasattr(model, "coef_"):
        importance_df = pd.DataFrame({
            "feature":     FEATURE_COLUMNS,
            "coefficient": model.coef_[0],
        }).sort_values("coefficient", key=abs, ascending=False)
        save_csv(importance_df, os.path.join(out_dir, "feature_importance.csv"))


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
    thresh         = MODEL_THRESHOLDS[model_key]

    out_dir = os.path.join(OUTPUT_DIR, model_short)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model   : {model_key}  ({model_short})")
    print(f"Sampler : {sampler_config['key']}")
    print(f"Output  : {out_dir}")
    print(f"{'='*60}\n")

    # --- Load data ---
    X_train, y_train, X_test, y_test, X_val, y_val, mapping_train, mapping_test, mapping_val = load_data()

    # --- Load best hyperparameters from step 03 ---
    print("\nLoading best hyperparameters from step 03...")
    best_params = load_best_params(model_short)
    print(f"  Best params: {best_params}")

    # --- Resample training set only ---
    print(f"\nApplying resampler: {sampler_config['key']}...")
    sampler_cls = load_class(sampler_config["class"])
    sampler     = sampler_cls(**sampler_config["params"])
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    n_neg_res = (y_train_res == 0).sum()
    n_pos_res = (y_train_res == 1).sum()
    print(f"  Post-resampling — negatives: {n_neg_res:,} | positives: {n_pos_res:,}")
    print(f"  Total training samples     : {len(y_train_res):,}")

    # --- Build final model with best params merged into base params ---
    print(f"\nTraining final {model_key} with best hyperparameters...")
    model_cls    = load_class(model_config["class"])
    final_params = {**model_config["base_params"], **best_params}
    model        = model_cls(**final_params)
    model.fit(X_train_res, y_train_res)
    print("  Training complete.")

    # --- Save fitted model ---
    model_path = os.path.join(MODELS_DIR, f"{model_short}_final.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    # --- Save training info ---
    training_info = {
        "model":                 model_key,
        "model_short":           model_short,
        "sampler":               sampler_config["key"],
        "threshold":             thresh,
        "final_params":          final_params,
        "train_neg_original":    int((y_train == 0).sum()),
        "train_pos_original":    int((y_train == 1).sum()),
        "train_neg_resampled":   int(n_neg_res),
        "train_pos_resampled":   int(n_pos_res),
        "model_path":            model_path,
    }
    save_json(training_info, os.path.join(out_dir, "training_info.json"))

    # --- Evaluate on train set  (For use in the Jenks classification step) ---
    evaluate_and_save(model, X_train, y_train, mapping_train, thresh, "train", out_dir)

    # --- Evaluate on test set ---
    evaluate_and_save(model, X_test, y_test, mapping_test, thresh, "test", out_dir)

    # --- Evaluate on validation set (first time this set is used) ---
    evaluate_and_save(model, X_val, y_val, mapping_val, thresh, "val", out_dir)

    print(f"\n{'='*60}")
    print(f"Done. All outputs saved to : {out_dir}/")
    print(f"Trained model saved to     : {model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 04_final_model_training.py <model_name>")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    main(sys.argv[1])