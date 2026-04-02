"""
01_plot_model_testing.py
--------------------------
Caller script for comparing all baseline models from the model testing step.

This script:
  1. Loads metrics.json for each model → calls plot_model_comparison().
  2. Loads confusion_matrix.csv for each model → calls plot_confusion_matrix().
  3. Loads pr_curve.csv for each model → calls plot_pr_curve().
  4. Loads roc_curve.csv for each model → calls plot_roc_curve().
  5. Loads feature_importance.csv for each model (if available) → calls plot_feature_importance().

To compare a different subset of models in the future (e.g. only RF and XGB),
create a new caller script — do NOT modify the plotter modules in utils/.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/01_model_testing"
OUTPUT_DIR  = f"{PROJECT_DIR}/results/01_model_testing/figures/"

MODELS_TO_PLOT = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "extra_trees",
    "knn",
    "naive_bayes",
]

FILENAME_PREFIX = "baseline_model_comparison"
PLOT_TITLE      = "Baseline Model Comparison — All Models"

# Thresholds to plot — must match those used in 01_model_testing.py.
# One set of threshold-sensitive figures (metrics, confusion matrix) is
# produced per threshold. PR/ROC curves and feature importance are
# threshold-independent and plotted once as before.
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

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


def load_metrics(model_name, threshold):
    path = os.path.join(RESULTS_DIR, model_name, f"metrics_{threshold}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # 1. Metrics bar plot — all models compared, one figure per threshold
    # -------------------------------------------------------------------------
    section("Plotting metric comparison bar plots...")

    for thresh in THRESHOLDS:
        metrics_list = []
        for model_name in MODELS_TO_PLOT:
            metrics = load_metrics(model_name, thresh)
            if metrics is None:
                print(f"  WARNING: metrics_{thresh}.json not found for '{model_name}' — skipping.")
                continue
            metrics.setdefault("model", model_name)
            metrics_list.append(metrics)
            print(f"  Loaded metrics for: {model_name} (threshold={thresh})")

        if not metrics_list:
            print(f"  WARNING: No metrics loaded for threshold {thresh} — skipping.")
            continue

        print(f"\n  Plotting comparison for {len(metrics_list)} model(s) at threshold {thresh}...")
        plot_model_comparison(
            metrics_list=metrics_list,
            output_dir=OUTPUT_DIR,
            filename_prefix=f"{FILENAME_PREFIX}_t{thresh}",
            title=f"{PLOT_TITLE} (threshold={thresh})",
        )

    # -------------------------------------------------------------------------
    # 2. Confusion matrices — one figure per model per threshold
    # -------------------------------------------------------------------------
    section("Plotting confusion matrices...")

    for thresh in THRESHOLDS:
        for model_name in MODELS_TO_PLOT:
            cm_path = os.path.join(RESULTS_DIR, model_name, f"confusion_matrix_{thresh}.csv")

            if not os.path.exists(cm_path):
                print(f"  WARNING: confusion_matrix_{thresh}.csv not found for '{model_name}' — skipping.")
                continue

            cm_df = pd.read_csv(cm_path)
            cm = np.column_stack([
                cm_df["predicted_negative"].values,
                cm_df["predicted_positive"].values,
            ])

            print(f"  Plotting confusion matrix for: {model_name} (threshold={thresh})")
            plot_confusion_matrix(
                cm=cm,
                model_name=f"{model_name}_t{thresh}",
                output_dir=OUTPUT_DIR,
            )

    # -------------------------------------------------------------------------
    # 3. PR curves — one figure per model
    # -------------------------------------------------------------------------
    section("Plotting Precision-Recall curves...")

    for model_name in MODELS_TO_PLOT:
        pr_path = os.path.join(RESULTS_DIR, model_name, "pr_curve.csv")

        if not os.path.exists(pr_path):
            print(f"  WARNING: pr_curve.csv not found for '{model_name}' — skipping.")
            continue

        pr_df = pd.read_csv(pr_path)

        metrics = load_metrics(model_name, THRESHOLDS[0])
        if metrics is None:
            print(f"  WARNING: metrics_{THRESHOLDS[0]}.json not found for '{model_name}' — skipping.")
            continue

        avg_precision = metrics.get("average_precision")
        if avg_precision is None:
            print(f"  WARNING: average_precision missing for '{model_name}' — skipping.")
            continue

        preds_path = os.path.join(RESULTS_DIR, model_name, "predictions.csv")
        if os.path.exists(preds_path):
            preds_df   = pd.read_csv(preds_path, usecols=["y_true"])
            prevalence = preds_df["y_true"].mean()
        else:
            print(f"  WARNING: predictions.csv not found for '{model_name}', "
                  "prevalence will default to 0.")
            prevalence = 0.0

        print(f"  Plotting PR curve for: {model_name}")
        plot_pr_curve(
            pr_df=pr_df,
            model_name=model_name,
            avg_precision=avg_precision,
            prevalence=prevalence,
            output_dir=OUTPUT_DIR,
        )

    # -------------------------------------------------------------------------
    # 4. ROC curves — one figure per model
    # -------------------------------------------------------------------------
    section("Plotting ROC curves...")

    for model_name in MODELS_TO_PLOT:
        roc_path = os.path.join(RESULTS_DIR, model_name, "roc_curve.csv")

        if not os.path.exists(roc_path):
            print(f"  WARNING: roc_curve.csv not found for '{model_name}' — skipping.")
            continue

        roc_df = pd.read_csv(roc_path)

        metrics = load_metrics(model_name, THRESHOLDS[0])
        if metrics is None:
            print(f"  WARNING: metrics_{THRESHOLDS[0]}.json not found for '{model_name}' — skipping.")
            continue

        roc_auc = metrics.get("roc_auc")
        if roc_auc is None:
            print(f"  WARNING: roc_auc missing for '{model_name}' — skipping.")
            continue

        print(f"  Plotting ROC curve for: {model_name}")
        plot_roc_curve(
            roc_df=roc_df,
            model_name=model_name,
            roc_auc=roc_auc,
            output_dir=OUTPUT_DIR,
        )

    # -------------------------------------------------------------------------
    # 5. Feature importance — one figure per model (skips models without it)
    # -------------------------------------------------------------------------
    section("Plotting feature importances...")

    for model_name in MODELS_TO_PLOT:
        fi_path = os.path.join(RESULTS_DIR, model_name, "feature_importance.csv")

        if not os.path.exists(fi_path):
            print(f"  NOTE: No feature_importance.csv for '{model_name}' — skipping.")
            continue

        importance_df = pd.read_csv(fi_path)

        print(f"  Plotting feature importance for: {model_name}")
        plot_feature_importance(
            importance_df=importance_df,
            model_name=model_name,
            output_dir=OUTPUT_DIR,
        )

    print("\nAll plots complete.")


if __name__ == "__main__":
    main()