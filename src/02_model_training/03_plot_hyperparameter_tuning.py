"""
03_plot_hyperparameter_tuning.py
---------------------------------
Caller script for comparing hyperparameter tuning results across all
three tuned models (RF, XGB, XT) from step 03.

This script:
  1. Loads metrics.json for each model → calls plot_model_comparison()
     for all three models side by side.
  2. Loads confusion_matrix.csv for each model → calls plot_confusion_matrix()
     once per model (3 figures total).
  3. Loads pr_curve.csv for each model → calls plot_pr_curve()
     once per model (3 figures total).
  4. Loads roc_curve.csv for each model → calls plot_roc_curve()
     once per model (3 figures total).
  5. Loads feature_importance.csv for each model → calls plot_feature_importance()
     once per model (3 figures total).

Output naming convention:
  - Metrics comparison : tuned_model_comparison.<ext>
  - All others         : <MODEL_SHORT>_tuned_<plot_type>.<ext>

NOTE: Do NOT modify the plotter modules in utils/.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/03_hyperparameter_tuning"
OUTPUT_DIR  = f"{PROJECT_DIR}/results/03_hyperparameter_tuning/figures/"

# Models and their short codes — must match 03_hyperparameter_tuning.py
MODELS = {
    "random_forest": "RF",
    "xgboost":       "XGB",
    "extra_trees":   "XT",
}

FILENAME_PREFIX = "tuned_model_comparison"
PLOT_TITLE      = "Tuned Model Comparison — RF, XGB, XT (MCC at optimal threshold)"

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from plot_metrics import plot_model_comparison
from plot_confmatrix import plot_confusion_matrix
from plot_prcurve import plot_pr_curve
from plot_roccurve import plot_roc_curve
from plot_featimportance import plot_feature_importance

# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def model_dir(model_short):
    """Full path to the results directory for a given model."""
    return os.path.join(RESULTS_DIR, model_short)


def load_metrics(model_short):
    """Load metrics.json (threshold is stored inside, under 'threshold')."""
    path = os.path.join(model_dir(model_short), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Metrics comparison — all three tuned models side by side
    # -------------------------------------------------------------------------
    section("Plotting metric comparison (all tuned models)...")

    metrics_list = []
    for model_key, model_short in MODELS.items():
        metrics = load_metrics(model_short)
        if metrics is None:
            print(f"  WARNING: metrics_*.json not found for '{model_short}' — skipping.")
            continue
        # Use short name as label so bars are clearly identified in the figure
        metrics["model"] = f"{model_short}_tuned"
        metrics_list.append(metrics)
        print(f"  Loaded metrics for: {model_short}")

    if not metrics_list:
        print("ERROR: No metrics loaded. Check RESULTS_DIR and model short names.")
        sys.exit(1)

    print(f"\n  Plotting comparison for {len(metrics_list)} model(s)...")
    plot_model_comparison(
        metrics_list=metrics_list,
        output_dir=OUTPUT_DIR,
        filename_prefix=FILENAME_PREFIX,
        title=PLOT_TITLE,
    )

    # -------------------------------------------------------------------------
    # 2. Confusion matrices — one figure per model
    # -------------------------------------------------------------------------
    section("Plotting confusion matrices...")

    for model_key, model_short in MODELS.items():
        directory = model_dir(model_short)
        cm_path   = os.path.join(directory, "confusion_matrix.csv")

        if not os.path.exists(cm_path):
            print(f"  WARNING: confusion_matrix.csv not found for '{model_short}' — skipping.")
            continue

        cm_df = pd.read_csv(cm_path)
        cm    = np.column_stack([
            cm_df["predicted_negative"].values,
            cm_df["predicted_positive"].values,
        ])

        label = f"{model_short}_tuned"
        # Show the reported (MCC-optimal) threshold in the title; keep the
        # filename clean so re-runs overwrite cleanly.
        metrics    = load_metrics(model_short)
        thresh     = metrics.get("threshold") if metrics else None
        title_name = f"{label} (thr={thresh:.4g})" if thresh is not None else label

        print(f"  Plotting confusion matrix for: {label}")
        plot_confusion_matrix(
            cm=cm,
            model_name=title_name,
            output_dir=OUTPUT_DIR,
            filename_prefix=f"confusion_matrix_{label}",
        )

    # -------------------------------------------------------------------------
    # 3. PR curves — one figure per model
    # -------------------------------------------------------------------------
    section("Plotting Precision-Recall curves...")

    for model_key, model_short in MODELS.items():
        label   = f"{model_short}_tuned"
        pr_path = os.path.join(model_dir(model_short), "pr_curve.csv")

        if not os.path.exists(pr_path):
            print(f"  WARNING: pr_curve.csv not found for '{model_short}' — skipping.")
            continue

        pr_df = pd.read_csv(pr_path)

        metrics = load_metrics(model_short)
        if metrics is None:
            print(f"  WARNING: metrics_*.json not found for '{model_short}' — skipping.")
            continue

        avg_precision = metrics.get("average_precision")
        if avg_precision is None:
            print(f"  WARNING: average_precision missing for '{model_short}' — skipping.")
            continue

        preds_path = os.path.join(model_dir(model_short), "predictions.csv")
        if os.path.exists(preds_path):
            preds_df   = pd.read_csv(preds_path, usecols=["y_true"])
            prevalence = preds_df["y_true"].mean()
        else:
            print(f"  WARNING: predictions.csv not found for '{model_short}', "
                  "prevalence will default to 0.")
            prevalence = 0.0

        print(f"  Plotting PR curve for: {label}")
        plot_pr_curve(
            pr_df=pr_df,
            model_name=label,
            avg_precision=avg_precision,
            prevalence=prevalence,
            output_dir=OUTPUT_DIR,
        )

    # -------------------------------------------------------------------------
    # 4. ROC curves — one figure per model
    # -------------------------------------------------------------------------
    section("Plotting ROC curves...")

    for model_key, model_short in MODELS.items():
        label    = f"{model_short}_tuned"
        roc_path = os.path.join(model_dir(model_short), "roc_curve.csv")

        if not os.path.exists(roc_path):
            print(f"  WARNING: roc_curve.csv not found for '{model_short}' — skipping.")
            continue

        roc_df = pd.read_csv(roc_path)

        metrics = load_metrics(model_short)
        if metrics is None:
            print(f"  WARNING: metrics_*.json not found for '{model_short}' — skipping.")
            continue

        roc_auc = metrics.get("roc_auc")
        if roc_auc is None:
            print(f"  WARNING: roc_auc missing for '{model_short}' — skipping.")
            continue

        print(f"  Plotting ROC curve for: {label}")
        plot_roc_curve(
            roc_df=roc_df,
            model_name=label,
            roc_auc=roc_auc,
            output_dir=OUTPUT_DIR,
        )

    # -------------------------------------------------------------------------
    # 5. Feature importance — one figure per model
    # -------------------------------------------------------------------------
    section("Plotting feature importances...")

    for model_key, model_short in MODELS.items():
        label   = f"{model_short}_tuned"
        fi_path = os.path.join(model_dir(model_short), "feature_importance.csv")

        if not os.path.exists(fi_path):
            print(f"  NOTE: No feature_importance.csv for '{model_short}' — skipping.")
            continue

        importance_df = pd.read_csv(fi_path)

        print(f"  Plotting feature importance for: {label}")
        plot_feature_importance(
            importance_df=importance_df,
            model_name=label,
            output_dir=OUTPUT_DIR,
        )

    print("\nAll plots complete.")


if __name__ == "__main__":
    main()