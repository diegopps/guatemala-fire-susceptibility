"""
model_testing.py
----------------
Runs a single baseline ML model on the train/test split and saves all
metrics and data needed for downstream plotting. Additionally, tests for multiple thresholds
to evaluate model metrics. This is meant for exploring using different thresholds for classifying 
the very unbalanced forest fire dataset

Usage:
    python model_testing.py <model_name>

Example:
    python model_testing.py random_forest

Each model is intended to be submitted as a separate SLURM job via a
bash wrapper script.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
INPUT_DATA_PATH  = f"{PROJECT_DIR}/data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv"
SPLIT_INDEX_PATH = f"{PROJECT_DIR}/data/processed/split_index.csv"
OUTPUT_DIR       = f"{PROJECT_DIR}/results/01_model_testing/"

# Features to use (fire is always excluded as target; lat, lon, year excluded)
FEATURE_COLUMNS = [
    "PET", "climbal", "pr", "rsds",
    "tas", "tasmax", "tasmin",
    "aspect_sin", "aspect_cos",
    "distRoads", "distUrban", "distWater",
    "elev", "slope",
]

TARGET_COLUMN = "fire"

# Thresholds to evaluate — metrics.json and confusion_matrix.csv are saved
# separately as a file for each threshold.
# PR/ROC curves and feature importance are threshold-independent and saved once.
THRESHOLDS = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]

# Models available to run — pass one of these keys as CLI argument
MODELS = {
    "logistic_regression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
    },
    "decision_tree": {
        "class": "sklearn.tree.DecisionTreeClassifier",
        "params": {
            "class_weight": "balanced",
            "random_state": 42,
        },
    },
    "random_forest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "gradient_boosting": {
        "class": "sklearn.ensemble.GradientBoostingClassifier",
        "params": {
            "n_estimators": 100,
            "random_state": 42,
        },
    },
    "xgboost": {
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
        "class": "sklearn.ensemble.ExtraTreesClassifier",
        "params": {
            "n_estimators": 100,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "knn": {
        "class": "sklearn.neighbors.KNeighborsClassifier",
        "params": {
            "n_neighbors": 5,
            "n_jobs": -1,
        },
    },
    "naive_bayes": {
        "class": "sklearn.naive_bayes.GaussianNB",
        "params": {},
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

# =============================================================================
# HELPERS
# =============================================================================

def load_model(model_key):
    """Dynamically instantiate a model from its dotted class path."""
    config = MODELS[model_key]
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

    # Merge split labels onto the main dataframe by position
    df["_split"] = split_index.set_index("row_index")["split"].values

    df_train = df[df["_split"] == "train"].copy()
    df_test  = df[df["_split"] == "test"].copy()

    print(f"  Train rows : {len(df_train):,}")
    print(f"  Test rows  : {len(df_test):,}")

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN]
    X_test  = df_test[FEATURE_COLUMNS]
    y_test  = df_test[TARGET_COLUMN]

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

    # --- Output directory for this model ---
    model_out_dir = os.path.join(OUTPUT_DIR, model_key)
    os.makedirs(model_out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"Output directory: {model_out_dir}")
    print(f"{'='*60}\n")

    # --- Load data ---
    X_train, y_train, X_test, y_test = load_data()

    # --- Train ---
    print(f"\nTraining {model_key}...")
    model = load_model(model_key)
    model.fit(X_train, y_train)
    print("  Training complete.")

    # --- Predict ---
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)

    # Predicted probabilities (positive class)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Fallback for models without predict_proba (e.g. SVM)
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None
        print("  WARNING: Model does not support probability estimates.")

    # --- Metrics and confusion matrix — one set per threshold ---
    # PR/ROC curves and feature importance are threshold-independent and saved once below.
    print("\nComputing metrics and confusion matrices per threshold...")

    # Threshold-independent metrics computed once
    avg_prec = average_precision_score(y_test, y_prob) if y_prob is not None else None
    roc_auc  = roc_auc_score(y_test, y_prob)           if y_prob is not None else None

    for thresh in THRESHOLDS:
        # Binarize probabilities at this threshold; fall back to default y_pred if no probs
        if y_prob is not None:
            y_pred_t = (y_prob >= thresh).astype(int)
        else:
            y_pred_t = y_pred  # only one threshold makes sense without probabilities

        bal_acc = balanced_accuracy_score(y_test, y_pred_t)
        mcc     = matthews_corrcoef(y_test, y_pred_t)

        metrics = {
            "model":             model_key,
            "threshold":         thresh,
            "balanced_accuracy": bal_acc,
            "matthews_corrcoef": mcc,
            "average_precision": avg_prec,
            "roc_auc":           roc_auc,
        }

        print(f"\n  Threshold: {thresh}")
        print(f"    Balanced Accuracy : {bal_acc:.4f}")
        print(f"    MCC               : {mcc:.4f}  <-- primary metric")
        print(f"    Average Precision : {avg_prec:.4f}" if avg_prec is not None else "    Average Precision : N/A")
        print(f"    ROC AUC           : {roc_auc:.4f}" if roc_auc is not None else "    ROC AUC           : N/A")

        save_json(metrics, os.path.join(model_out_dir, f"metrics_{thresh}.json"))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_t)
        cm_df = pd.DataFrame(
            cm,
            index=["actual_negative", "actual_positive"],
            columns=["predicted_negative", "predicted_positive"],
        )
        save_csv(cm_df, os.path.join(model_out_dir, f"confusion_matrix_{thresh}.csv"))

    # --- Raw predictions & probabilities ---
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_prob": y_prob if y_prob is not None else np.nan,
    })
    save_csv(preds_df, os.path.join(model_out_dir, "predictions.csv"))

    # --- PR curve ---
    if y_prob is not None:
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
        pr_df = pd.DataFrame({
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": pr_thresholds,
        })
        save_csv(pr_df, os.path.join(model_out_dir, "pr_curve.csv"))

    # --- ROC curve ---
    if y_prob is not None:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        roc_df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thresholds,
        })
        save_csv(roc_df, os.path.join(model_out_dir, "roc_curve.csv"))

    # --- Variable importance ---
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        save_csv(importance_df, os.path.join(model_out_dir, "feature_importance.csv"))
    elif hasattr(model, "coef_"):
        # Linear models (e.g. logistic regression)
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "coefficient": model.coef_[0],
        }).sort_values("coefficient", key=abs, ascending=False)
        save_csv(importance_df, os.path.join(model_out_dir, "feature_importance.csv"))
    else:
        print("  NOTE: Model does not expose feature importances — skipping.")

    print(f"\nDone. All outputs saved under: {model_out_dir}/")
    print(f"  Threshold-specific outputs: metrics.json, confusion_matrix.csv")
    print(f"  Threshold-independent outputs: predictions.csv, pr_curve.csv, roc_curve.csv, feature_importance.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_testing.py <model_name>")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    main(sys.argv[1])