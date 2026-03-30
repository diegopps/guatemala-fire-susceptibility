"""
01_plot_model_testing.py
--------------------------
Caller script for comparing all baseline models from the model testing step.

This script:
  1. Defines which models to compare and where their results live.
  2. Loads the metrics.json for each model.
  3. Calls the generic plot_model_comparison() function.
  4. Loads the confusion_matrix.csv for each model.
  5. Calls the generic plot_confusion_matrix() function for each model.

To compare a different subset of models in the future (e.g. only RF and XGB),
create a new caller script — do NOT modify plot_metrics.py or
plot_confusion_matrix.py.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/01_model_testing"
OUTPUT_DIR  = f"{PROJECT_DIR}/results/01_model_testing/figures/"

# Models to include in the comparison plot.
# These must match the directory names under RESULTS_DIR.
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

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import sys
import pandas as pd
import numpy as np

# Import generic plotters from utils
sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from plot_metrics import plot_model_comparison
from plot_confmatrix import plot_confusion_matrix

# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # 1. Metrics bar plot — all models compared
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Plotting metric comparison bar plots...")
    print("=" * 60)

    metrics_list = []

    for model_name in MODELS_TO_PLOT:
        metrics_path = os.path.join(RESULTS_DIR, model_name, "metrics.json")

        if not os.path.exists(metrics_path):
            print(f"  WARNING: metrics.json not found for '{model_name}' — skipping.")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics.setdefault("model", model_name)
        metrics_list.append(metrics)
        print(f"  Loaded metrics for: {model_name}")

    if not metrics_list:
        print("ERROR: No metrics loaded. Check RESULTS_DIR and model names.")
        sys.exit(1)

    print(f"\nPlotting comparison for {len(metrics_list)} model(s)...")
    plot_model_comparison(
        metrics_list=metrics_list,
        output_dir=OUTPUT_DIR,
        filename_prefix=FILENAME_PREFIX,
        title=PLOT_TITLE,
    )

    # -------------------------------------------------------------------------
    # 2. Confusion matrices — one figure per model
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Plotting confusion matrices...")
    print("=" * 60)

    for model_name in MODELS_TO_PLOT:
        cm_path = os.path.join(RESULTS_DIR, model_name, "confusion_matrix.csv")

        if not os.path.exists(cm_path):
            print(f"  WARNING: confusion_matrix.csv not found for '{model_name}' — skipping.")
            continue

        cm_df = pd.read_csv(cm_path)
        cm = np.column_stack([
            cm_df["predicted_negative"].values,
            cm_df["predicted_positive"].values,
        ])
        print(f"\n  Plotting confusion matrix for: {model_name}")


        plot_confusion_matrix(
            cm=cm,
            model_name=model_name,
            output_dir=OUTPUT_DIR,
        )

    print("\nAll plots complete.")


if __name__ == "__main__":
    main()