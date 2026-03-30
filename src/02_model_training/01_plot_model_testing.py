"""
01_plot_model_testing.py
--------------------------
Caller script for comparing all baseline models from the model testing step.

This script:
  1. Defines which models to compare and where their results live.
  2. Loads the metrics.json for each model.
  3. Calls the generic plot_model_comparison() function.

To compare a different subset of models in the future (e.g. only RF and XGB),
create a new caller script — do NOT modify plot_model_comparison.py.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/01_model_testing"
OUTPUT_DIR  = f"{PROJECT_DIR}/resutls/01_model_testing/figures/"

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

# Import the generic plotter from utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from plot_model_comparison import plot_model_comparison

# =============================================================================
# MAIN
# =============================================================================

def main():
    metrics_list = []

    for model_name in MODELS_TO_PLOT:
        metrics_path = os.path.join(RESULTS_DIR, model_name, "metrics.json")

        if not os.path.exists(metrics_path):
            print(f"WARNING: metrics.json not found for '{model_name}' — skipping.")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Use the directory name as display label if "model" key is missing
        metrics.setdefault("model", model_name)
        metrics_list.append(metrics)
        print(f"  Loaded metrics for: {model_name}")

    if not metrics_list:
        print("ERROR: No metrics loaded. Check that RESULTS_DIR and model names are correct.")
        sys.exit(1)

    print(f"\nPlotting comparison for {len(metrics_list)} model(s)...")
    plot_model_comparison(
        metrics_list=metrics_list,
        output_dir=OUTPUT_DIR,
        filename_prefix=FILENAME_PREFIX,
        title=PLOT_TITLE,
    )


if __name__ == "__main__":
    main()