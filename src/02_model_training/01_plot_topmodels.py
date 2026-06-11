"""
01_plot_topmodels.py
---------------------
Caller script for summarising the top-performing models identified
during model testing.

Selected models and their best thresholds:
    - Random Forest  (RF)  → threshold 0.03
    - Extra Trees    (XT)  → threshold 0.03
    - XGBoost        (XGB) → threshold 0.01

This script produces three figures:
    1. 4-panel metric barplot   — balanced accuracy, MCC, avg precision, ROC-AUC
    2. Confusion matrix panel   — one panel per model (vertical layout, 1 col × 3 rows)
    3. Feature importance panel — one panel per model (vertical layout, 1 col × 3 rows)

Do NOT modify the plotter utilities in src/utils/ — create a new caller
script if a different model selection is needed in the future.
"""

# =============================================================================
# SETTINGS — edit here to change model/threshold selection
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/01_model_testing"
OUTPUT_DIR  = f"{PROJECT_DIR}/results/01_model_testing/figures/top_models/"

# Each entry: (model_dir_name, display_name, best_threshold)
TOP_MODELS = [
    ("random_forest",     "Random Forest", 0.03),
    ("extra_trees",       "Extra Trees",   0.03),
    ("xgboost",           "XGBoost",       0.01),
]

FILENAME_PREFIX = "top_models"
PLOT_TITLE      = "RF, XT, XGB Models Comparison"

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from plot_metrics            import plot_model_comparison
from plot_confmatrix_panel   import plot_confusion_matrix_panel
from plot_featimportance_panel import plot_feature_importance_panel

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
        print(f"  WARNING: metrics_{threshold}.json not found for '{model_name}'.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_confusion_matrix(model_name, threshold):
    path = os.path.join(RESULTS_DIR, model_name, f"confusion_matrix_{threshold}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: confusion_matrix_{threshold}.csv not found for '{model_name}'.")
        return None
    cm_df = pd.read_csv(path)
    return np.column_stack([
        cm_df["predicted_negative"].values,
        cm_df["predicted_positive"].values,
    ])


def load_feature_importance(model_name):
    path = os.path.join(RESULTS_DIR, model_name, "feature_importance.csv")
    if not os.path.exists(path):
        print(f"  NOTE: feature_importance.csv not found for '{model_name}' — skipping.")
        return None
    return pd.read_csv(path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Metric barplot — all three models, each at its best threshold
    # -------------------------------------------------------------------------
    section("Plotting metric comparison barplot...")

    metrics_list = []
    for model_dir, display_name, thresh in TOP_MODELS:
        metrics = load_metrics(model_dir, thresh)
        if metrics is None:
            continue
        metrics["model"] = display_name   # use friendly display name
        metrics_list.append(metrics)
        print(f"  Loaded metrics: {display_name} (threshold={thresh})")

    if metrics_list:
        plot_model_comparison(
            metrics_list    = metrics_list,
            output_dir      = OUTPUT_DIR,
            filename_prefix = f"{FILENAME_PREFIX}_metrics",
            title           = f"{PLOT_TITLE} — Metrics",
        )
    else:
        print("  WARNING: No metrics loaded — skipping.")

    # -------------------------------------------------------------------------
    # 2. Confusion matrix panel — each model at its own best threshold
    # -------------------------------------------------------------------------
    section("Plotting confusion matrix panel...")

    cm_data = []
    for model_dir, display_name, thresh in TOP_MODELS:
        cm = load_confusion_matrix(model_dir, thresh)
        if cm is None:
            continue
        cm_data.append({
            "model_name": display_name,
            "cm":         cm,
            "threshold":  thresh,
        })
        print(f"  Loaded confusion matrix: {display_name} (threshold={thresh})")

    if cm_data:
        plot_confusion_matrix_panel(
            models_data = cm_data,
            output_dir  = OUTPUT_DIR,
            filename    = f"{FILENAME_PREFIX}_confusion_matrices",
            title       = f"{PLOT_TITLE} — Confusion Matrices",
        )
    else:
        print("  WARNING: No confusion matrices loaded — skipping.")

    # -------------------------------------------------------------------------
    # 3. Feature importance panel — threshold-independent
    # -------------------------------------------------------------------------
    section("Plotting feature importance panel...")

    fi_data = []
    for model_dir, display_name, _ in TOP_MODELS:
        fi_df = load_feature_importance(model_dir)
        if fi_df is None:
            continue
        fi_data.append({
            "model_name":    display_name,
            "importance_df": fi_df,
        })
        print(f"  Loaded feature importance: {display_name}")

    if fi_data:
        plot_feature_importance_panel(
            models_data = fi_data,
            output_dir  = OUTPUT_DIR,
            filename    = f"{FILENAME_PREFIX}_feature_importances",
            title       = f"{PLOT_TITLE} — Feature Importances",
        )
    else:
        print("  WARNING: No feature importances loaded — skipping.")

    print("\nAll top-model plots complete.")


if __name__ == "__main__":
    main()
