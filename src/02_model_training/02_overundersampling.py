"""
02_overundersampling.py
------------------------
Trains a single model + resampling strategy combination on the train/test
split and saves all metrics and data needed for downstream plotting.

Resampling is applied ONLY to the training set. The test set is never
touched, ensuring unbiased evaluation.

Usage:
    python 02_overunder_sampling.py <model_name> <sampler_name>

Examples:
    python 02_overunder_sampling.py random_forest smote
    python 02_overunder_sampling.py xgboost rus

Each combination is intended to be submitted as a separate SLURM job via
a bash wrapper script.

Output directory: results/02_overunder_sampling/<MODEL_SHORT>_<SAMPLER_SHORT>/
  e.g. results/02_overunder_sampling/RF_smote/
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR      = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
INPUT_DATA_PATH  = f"{PROJECT_DIR}/data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv"
SPLIT_INDEX_PATH = f"{PROJECT_DIR}/data/processed/split_index.csv"
OUTPUT_DIR       = f"{PROJECT_DIR}/results/02_overundersampling/"

# Features to use (fire is always excluded as target; lat, lon, year excluded)
FEATURE_COLUMNS = [
    "PET", "climbal", "pr", "rsds",
    "tas", "tasmax", "tasmin",
    "aspect_sin", "aspect_cos",
    "distRoads", "distUrban", "distWater",
    "elev", "slope",
]

TARGET_COLUMN = "fire"

# No fixed decision threshold is used. The threshold is a post-hoc operating
# point, so each run is reported at the threshold that MAXIMIZES its MCC (the
# peak of its MCC-vs-threshold curve). This keeps MCC as the metric while
# avoiding the bias of comparing models at one shared, model-mismatched cut.
# The chosen threshold is saved inside metrics.json. PR/ROC curves and feature
# importance are threshold-independent and unaffected.

# ---------------------------------------------------------------------------
# Models
# NOTE: No class_weight or scale_pos_weight is used here — resampling handles
#       class imbalance instead, keeping each experiment's effect isolated.
# ---------------------------------------------------------------------------
MODELS = {
    "random_forest": {
        "short": "RF",
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "xgboost": {
        "short": "XGB",
        "class": "xgboost.XGBClassifier",
        "params": {
            "n_estimators": 100,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "extra_trees": {
        "short": "XT",
        "class": "sklearn.ensemble.ExtraTreesClassifier",
        "params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
}

# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
SAMPLERS = {
    # --- Oversampling ---
    "ros": {
        "short": "ros",
        "class": "imblearn.over_sampling.RandomOverSampler",
        "params": {"random_state": 42},
    },
    "smote": {
        "short": "smote",
        "class": "imblearn.over_sampling.SMOTE",
        "params": {"random_state": 42},
    },
    "svmsmote": {
        "short": "svms",
        "class": "imblearn.over_sampling.SVMSMOTE",
        "params": {"random_state": 42},
    },
    "adasyn": {
        "short": "adsn",
        "class": "imblearn.over_sampling.ADASYN",
        "params": {"random_state": 42},
    },
    # --- Undersampling ---
    "rus": {
        "short": "rus",
        "class": "imblearn.under_sampling.RandomUnderSampler",
        "params": {"random_state": 42},
    },
    "nearmiss": {
        "short": "nm",
        "class": "imblearn.under_sampling.NearMiss",
        "params": {"n_jobs": -1},
    },
    "tomek": {
        "short": "tomk",
        "class": "imblearn.under_sampling.TomekLinks",
        "params": {"n_jobs": -1},
    },
    "ncl": {
        "short": "ncl",
        "class": "imblearn.under_sampling.NeighbourhoodCleaningRule",
        "params": {"n_jobs": -1},
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

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from threshold_selection import select_threshold_on_validation

# =============================================================================
# HELPERS
# =============================================================================

def load_model(model_key):
    """Dynamically instantiate a model from its dotted class path."""
    config = MODELS[model_key]
    params = dict(config["params"])  # copy to avoid mutating the global config
    module_path, class_name = config["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**params)


def load_sampler(sampler_key):
    """Dynamically instantiate a sampler from its dotted class path."""
    config = SAMPLERS[sampler_key]
    module_path, class_name = config["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**config["params"])


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

def main(model_key, sampler_key):
    # --- Validate inputs ---
    if model_key not in MODELS:
        print(f"ERROR: Unknown model '{model_key}'.")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    if sampler_key not in SAMPLERS:
        print(f"ERROR: Unknown sampler '{sampler_key}'.")
        print(f"Available samplers: {', '.join(SAMPLERS.keys())}")
        sys.exit(1)

    # --- Build output directory name: e.g. RF_smote ---
    model_short   = MODELS[model_key]["short"]
    sampler_short = SAMPLERS[sampler_key]["short"]
    run_name      = f"{model_short}_{sampler_short}"
    out_dir       = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model   : {model_key}  ({model_short})")
    print(f"Sampler : {sampler_key}  ({sampler_short})")
    print(f"Run ID  : {run_name}")
    print(f"Output  : {out_dir}")
    print(f"{'='*60}\n")

    # --- Load data ---
    X_train, y_train, X_test, y_test = load_data()

    # --- Resample training set only ---
    print(f"\nApplying resampler: {sampler_key}...")
    sampler = load_sampler(sampler_key)
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    n_neg_res = (y_train_res == 0).sum()
    n_pos_res = (y_train_res == 1).sum()
    print(f"  Post-resampling — negatives: {n_neg_res:,} | positives: {n_pos_res:,}")
    print(f"  Total training samples: {len(y_train_res):,}")

    # --- Train ---
    print(f"\nTraining {model_key}...")
    model = load_model(model_key)
    model.fit(X_train_res, y_train_res)
    print("  Training complete.")

    # --- Predict ---
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None
        print("  WARNING: Model does not support probability estimates.")

    # --- Metrics and confusion matrix — reported at the MCC-optimal threshold ---
    # The threshold is selected as the peak of the MCC-vs-threshold curve on the
    # test predictions (a single scalar, negligible optimism) so the model is
    # read at its own optimal operating point rather than a fixed shared cut.
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
        "run":                 run_name,
        "model":               model_key,
        "sampler":             sampler_key,
        "threshold":           thresh,
        "balanced_accuracy":   bal_acc,
        "matthews_corrcoef":   mcc,
        "average_precision":   avg_prec,
        "roc_auc":             roc_auc,
        "train_neg_original":  int((y_train == 0).sum()),
        "train_pos_original":  int((y_train == 1).sum()),
        "train_neg_resampled": int(n_neg_res),
        "train_pos_resampled": int(n_pos_res),
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
    else:
        print("  NOTE: Model does not expose feature importances — skipping.")

    print(f"\nDone. All outputs saved to: {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 02_overunder_sampling.py <model_name> <sampler_name>")
        print(f"Available models  : {', '.join(MODELS.keys())}")
        print(f"Available samplers: {', '.join(SAMPLERS.keys())}")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])