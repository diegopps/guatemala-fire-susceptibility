"""
02_plot_overunder_sampling.py
-----------------------------
Caller script for comparing over/undersampling results across all
model + sampler combinations from step 02.

This script:
  1. Loads metrics.json for each combination → calls plot_model_comparison()
     once per model (all 8 samplers side by side), producing 3 figures.
  2. Loads confusion_matrix.csv for each combination → calls plot_confusion_matrix()
     once per combination (24 figures total).
  3. Loads pr_curve.csv for each combination → calls plot_pr_curve()
     once per combination (24 figures total).
  4. Loads roc_curve.csv for each combination → calls plot_roc_curve()
     once per combination (24 figures total).
  5. Loads feature_importance.csv for each combination → calls plot_feature_importance()
     once per combination (24 figures total).

Output naming convention:
  - Metrics comparison : <MODEL_SHORT>_sampler_comparison.<ext>
  - All others         : <MODEL_SHORT>_<SAMPLER_SHORT>_<plot_type>.<ext>

NOTE: do NOT modify the plotter modules in utils/.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/02_overundersampling"
OUTPUT_DIR  = f"{PROJECT_DIR}/results/02_overundersampling/figures/"

# Models and their short codes — must match 02_overundersampling.py
MODELS = {
    "random_forest": "RF",
    "xgboost":       "XGB",
    "extra_trees":   "XT",
}

# Samplers and their short codes — must match 02_overundersampling.py
SAMPLERS = {
    "ros":      "ros",
    "smote":    "smote",
    "svmsmote": "svms",
    "adasyn":   "adsn",
    "rus":      "rus",
    "nearmiss": "nm",
    "tomek":    "tomk",
    "ncl":      "ncl",
}

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


def run_name(model_short, sampler_short):
    """Construct the result directory name, e.g. RF_smote."""
    return f"{model_short}_{sampler_short}"


def run_dir(model_short, sampler_short):
    """Full path to the results directory for a given combination."""
    return os.path.join(RESULTS_DIR, run_name(model_short, sampler_short))


def load_metrics(model_short, sampler_short):
    """Load metrics JSON, auto-detecting the threshold suffix from the filename."""
    directory = run_dir(model_short, sampler_short)
    matches = [f for f in os.listdir(directory) if f.startswith("metrics_") and f.endswith(".json")] \
              if os.path.isdir(directory) else []
    if not matches:
        return None
    path = os.path.join(directory, matches[0])
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Metrics comparison — one figure per model (all samplers compared)
    # -------------------------------------------------------------------------
    section("Plotting metric comparison plots (one per model: all samplers)...")

    for model_key, model_short in MODELS.items():
        metrics_list = []

        for sampler_key, sampler_short in SAMPLERS.items():
            metrics = load_metrics(model_short, sampler_short)
            if metrics is None:
                print(f"  WARNING: metrics_*.json not found for "
                      f"'{run_name(model_short, sampler_short)}' — skipping.")
                continue
            # Use the run name as the model label so bars are clearly identified
            metrics["model"] = run_name(model_short, sampler_short)
            metrics_list.append(metrics)
            print(f"  Loaded metrics for: {run_name(model_short, sampler_short)}")

        if not metrics_list:
            print(f"  WARNING: No metrics loaded for model '{model_key}' — skipping plot.")
            continue

        print(f"  Plotting all samplers for model: {model_key}")
        plot_model_comparison(
            metrics_list=metrics_list,
            output_dir=OUTPUT_DIR,
            filename_prefix=f"{model_short}_sampler_comparison",
            title=f"{model_short} — Sampler Comparison",
        )

    # -------------------------------------------------------------------------
    # 2. Confusion matrices — one figure per combination
    # -------------------------------------------------------------------------
    section("Plotting confusion matrices...")

    for model_key, model_short in MODELS.items():
        for sampler_key, sampler_short in SAMPLERS.items():
            directory = run_dir(model_short, sampler_short)
            cm_matches = [f for f in os.listdir(directory) if f.startswith("confusion_matrix_") and f.endswith(".csv")] \
                         if os.path.isdir(directory) else []

            if not cm_matches:
                print(f"  WARNING: confusion_matrix_*.csv not found for "
                      f"'{run_name(model_short, sampler_short)}' — skipping.")
                continue

            cm_path = os.path.join(directory, cm_matches[0])

            cm_df = pd.read_csv(cm_path)
            cm = np.column_stack([
                cm_df["predicted_negative"].values,
                cm_df["predicted_positive"].values,
            ])

            label = run_name(model_short, sampler_short)
            print(f"  Plotting confusion matrix for: {label}")
            plot_confusion_matrix(
                cm=cm,
                model_name=label,
                output_dir=OUTPUT_DIR,
            )

    # -------------------------------------------------------------------------
    # 3. PR curves — one figure per combination
    # -------------------------------------------------------------------------
    section("Plotting Precision-Recall curves...")

    for model_key, model_short in MODELS.items():
        for sampler_key, sampler_short in SAMPLERS.items():
            label   = run_name(model_short, sampler_short)
            pr_path = os.path.join(run_dir(model_short, sampler_short), "pr_curve.csv")

            if not os.path.exists(pr_path):
                print(f"  WARNING: pr_curve.csv not found for '{label}' — skipping.")
                continue

            pr_df = pd.read_csv(pr_path)

            metrics = load_metrics(model_short, sampler_short)
            if metrics is None:
                print(f"  WARNING: metrics_*.json not found for '{label}' — skipping.")
                continue

            avg_precision = metrics.get("average_precision")
            if avg_precision is None:
                print(f"  WARNING: average_precision missing for '{label}' — skipping.")
                continue

            preds_path = os.path.join(run_dir(model_short, sampler_short), "predictions.csv")
            if os.path.exists(preds_path):
                preds_df   = pd.read_csv(preds_path, usecols=["y_true"])
                prevalence = preds_df["y_true"].mean()
            else:
                print(f"  WARNING: predictions.csv not found for '{label}', "
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
    # 4. ROC curves — one figure per combination
    # -------------------------------------------------------------------------
    section("Plotting ROC curves...")

    for model_key, model_short in MODELS.items():
        for sampler_key, sampler_short in SAMPLERS.items():
            label    = run_name(model_short, sampler_short)
            roc_path = os.path.join(run_dir(model_short, sampler_short), "roc_curve.csv")

            if not os.path.exists(roc_path):
                print(f"  WARNING: roc_curve.csv not found for '{label}' — skipping.")
                continue

            roc_df = pd.read_csv(roc_path)

            metrics = load_metrics(model_short, sampler_short)
            if metrics is None:
                print(f"  WARNING: metrics_*.json not found for '{label}' — skipping.")
                continue

            roc_auc = metrics.get("roc_auc")
            if roc_auc is None:
                print(f"  WARNING: roc_auc missing for '{label}' — skipping.")
                continue

            print(f"  Plotting ROC curve for: {label}")
            plot_roc_curve(
                roc_df=roc_df,
                model_name=label,
                roc_auc=roc_auc,
                output_dir=OUTPUT_DIR,
            )

    # -------------------------------------------------------------------------
    # 5. Feature importance — one figure per combination
    # -------------------------------------------------------------------------
    section("Plotting feature importances...")

    for model_key, model_short in MODELS.items():
        for sampler_key, sampler_short in SAMPLERS.items():
            label   = run_name(model_short, sampler_short)
            fi_path = os.path.join(run_dir(model_short, sampler_short), "feature_importance.csv")

            if not os.path.exists(fi_path):
                print(f"  NOTE: No feature_importance.csv for '{label}' — skipping.")
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