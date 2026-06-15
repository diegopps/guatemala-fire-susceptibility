"""
04_plot_topK_finalmodel.py
---------------------------
Caller script for plotting all results from step 04 final model training,
including standard evaluation plots and top-k coverage analysis.

This script:
  1. Loads metrics.json for each model (test + val) → calls plot_model_comparison()
     twice: once for test set, once for val set.
  2. Loads confusion_matrix.csv for each model (test + val) → one unified panel
     figure per subset (3 rows × 1 col, one row per model).
  3. Loads pr_curve.csv for each model (test + val) → one unified panel figure
     per subset.
  4. Loads roc_curve.csv for each model (test + val) → one unified panel figure
     per subset.
  5. Loads feature_importance.csv for each model → one unified panel figure
     (threshold-independent).
  6. Loads predictions.csv for each model (test + val), computes top-k coverage
     at k = 1%, 2%, 5%, 10% → calls plot_topk_coverage() once per subset
     (2 figures total), each showing all k levels as grouped bars per model.

Each panel figure has 3 rows (one per model), with:
  - A vertical model label on the left side of each row.
  - A floating letter (A), B), C)) at the top-left of each row for figure
    referencing (e.g. "Figure 1-A").

Output naming convention:
  - Metrics comparison : final_model_comparison_<subset>.<ext>
  - Panel figures      : <plot_type>_panel_<subset>.<ext>
  - Feature importance : feature_importance_panel.<ext>
  - Top-k              : topk_coverage_top<k>pct_<subset>.<ext>

NOTE: Do NOT modify the plotter modules in utils/.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/04_final_model_training"
OUTPUT_DIR  = f"{PROJECT_DIR}/results/04_final_model_training/figures/"

# Models and their short codes — must match 04_final_model_training.py
MODELS = {
    "random_forest": "RF",
    "xgboost":       "XGB",
    "extra_trees":   "XT",
}

# Subsets to evaluate — must match subset labels used in step 04
SUBSETS = ["test", "val"]

# Top-k levels to evaluate
K_LEVELS = [0.01, 0.02, 0.05, 0.10]

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
from plot_confmatrix_panel import plot_confusion_matrix_panel
from plot_prcurve_panel import plot_pr_curve_panel
from plot_roccurve_panel import plot_roc_curve_panel
from plot_featimportance_panel import plot_feature_importance_panel
from plot_topk import plot_topk_coverage

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


def load_metrics(model_short, subset_label):
    """Load metrics_<subset>.json (threshold is stored inside, under 'threshold')."""
    path = os.path.join(model_dir(model_short), f"metrics_{subset_label}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_topk_coverage(y_true, y_prob, k_levels):
    """
    Compute top-k coverage for a list of k levels.

    For each k, ranks all grid cells by predicted probability (descending),
    takes the top k% of cells, and computes the fraction of true fires
    captured within that subset.

    Parameters
    ----------
    y_true   : array-like of int — true binary labels (0/1)
    y_prob   : array-like of float — predicted probabilities
    k_levels : list of float — e.g. [0.01, 0.02, 0.05, 0.10]

    Returns
    -------
    coverage : list of float — one coverage fraction per k level (0–1)
    n_fires  : int — total number of fires in the subset
    n_total  : int — total number of grid cells in the subset
    """
    y_true  = np.array(y_true)
    y_prob  = np.array(y_prob)
    n_total = len(y_true)
    n_fires = int(y_true.sum())

    # Sort by predicted probability descending
    sorted_idx = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_idx]

    coverage = []
    for k in k_levels:
        n_top     = max(1, int(np.ceil(k * n_total)))
        fires_in_top = y_true_sorted[:n_top].sum()
        cov = fires_in_top / n_fires if n_fires > 0 else 0.0
        coverage.append(float(cov))

    return coverage, n_fires, n_total


# =============================================================================
# MAIN
# =============================================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Metrics comparison — one figure per subset (test and val)
    # -------------------------------------------------------------------------
    section("Plotting metric comparison plots...")

    for subset in SUBSETS:
        metrics_list = []

        for model_key, model_short in MODELS.items():
            metrics = load_metrics(model_short, subset)
            if metrics is None:
                print(f"  WARNING: metrics_*_{subset}.json not found for "
                      f"'{model_short}' — skipping.")
                continue
            metrics["model"] = f"{model_short}_final"
            metrics_list.append(metrics)
            print(f"  Loaded metrics for: {model_short} ({subset})")

        if not metrics_list:
            print(f"  WARNING: No metrics loaded for subset '{subset}' — skipping plot.")
            continue

        print(f"  Plotting comparison for {len(metrics_list)} model(s) — {subset} set...")
        plot_model_comparison(
            metrics_list=metrics_list,
            output_dir=OUTPUT_DIR,
            filename_prefix=f"final_model_comparison_{subset}",
            title="Final Model Comparison (MCC at committed threshold)",
        )

    # -------------------------------------------------------------------------
    # 2. Confusion matrices — one unified panel figure per subset
    # -------------------------------------------------------------------------
    section("Plotting confusion matrices (panel)...")

    for subset in SUBSETS:
        cm_panel_data = []

        for model_key, model_short in MODELS.items():
            cm_path = os.path.join(model_dir(model_short), f"confusion_matrix_{subset}.csv")

            if not os.path.exists(cm_path):
                print(f"  WARNING: confusion_matrix_{subset}.csv not found for "
                      f"'{model_short}' — skipping.")
                continue

            cm_df = pd.read_csv(cm_path)
            cm    = np.column_stack([
                cm_df["predicted_negative"].values,
                cm_df["predicted_positive"].values,
            ])

            metrics = load_metrics(model_short, subset)
            thresh  = metrics.get("threshold") if metrics else None

            print(f"  Loaded confusion matrix for: {model_short} ({subset})")
            cm_panel_data.append({
                "model_name": f"{model_short}",
                "cm":         cm,
                "threshold":  f"{thresh:.4g}" if thresh is not None else "N/A",
            })

        if not cm_panel_data:
            print(f"  WARNING: No confusion matrix data for subset '{subset}' — skipping.")
            continue

        print(f"  Plotting confusion matrix panel for {subset} set...")
        plot_confusion_matrix_panel(
            models_data=cm_panel_data,
            output_dir=OUTPUT_DIR,
            filename=f"confusion_matrix_panel_{subset}",
            title="Confusion Matrices",
        )

    # -------------------------------------------------------------------------
    # 3. PR curves — one unified panel figure per subset
    # -------------------------------------------------------------------------
    section("Plotting Precision-Recall curves (panel)...")

    for subset in SUBSETS:
        pr_panel_data = []

        for model_key, model_short in MODELS.items():
            label   = f"{model_short}_final_{subset}"
            pr_path = os.path.join(model_dir(model_short), f"pr_curve_{subset}.csv")

            if not os.path.exists(pr_path):
                print(f"  WARNING: pr_curve_{subset}.csv not found for "
                      f"'{model_short}' — skipping.")
                continue

            pr_df = pd.read_csv(pr_path)

            metrics = load_metrics(model_short, subset)
            if metrics is None:
                print(f"  WARNING: metrics_{subset}.json not found for "
                      f"'{model_short}' — skipping.")
                continue

            avg_precision = metrics.get("average_precision")
            if avg_precision is None:
                print(f"  WARNING: average_precision missing for '{label}' — skipping.")
                continue

            preds_path = os.path.join(model_dir(model_short), f"predictions_{subset}.csv")
            if os.path.exists(preds_path):
                preds_df   = pd.read_csv(preds_path, usecols=["y_true"])
                prevalence = preds_df["y_true"].mean()
            else:
                print(f"  WARNING: predictions_{subset}.csv not found for "
                      f"'{model_short}', prevalence will default to 0.")
                prevalence = 0.0

            print(f"  Loaded PR curve for: {label}")
            pr_panel_data.append({
                "model_name":    model_short,
                "pr_df":         pr_df,
                "avg_precision": avg_precision,
                "prevalence":    prevalence,
            })

        if not pr_panel_data:
            print(f"  WARNING: No PR curve data for subset '{subset}' — skipping.")
            continue

        print(f"  Plotting PR curve panel for {subset} set...")
        plot_pr_curve_panel(
            models_data=pr_panel_data,
            output_dir=OUTPUT_DIR,
            filename=f"pr_curve_panel_{subset}",
            title="Precision-Recall Curves",
        )

    # -------------------------------------------------------------------------
    # 4. ROC curves — one unified panel figure per subset
    # -------------------------------------------------------------------------
    section("Plotting ROC curves (panel)...")

    for subset in SUBSETS:
        roc_panel_data = []

        for model_key, model_short in MODELS.items():
            label    = f"{model_short}_final_{subset}"
            roc_path = os.path.join(model_dir(model_short), f"roc_curve_{subset}.csv")

            if not os.path.exists(roc_path):
                print(f"  WARNING: roc_curve_{subset}.csv not found for "
                      f"'{model_short}' — skipping.")
                continue

            roc_df = pd.read_csv(roc_path)

            metrics = load_metrics(model_short, subset)
            if metrics is None:
                print(f"  WARNING: metrics_{subset}.json not found for "
                      f"'{model_short}' — skipping.")
                continue

            roc_auc = metrics.get("roc_auc")
            if roc_auc is None:
                print(f"  WARNING: roc_auc missing for '{label}' — skipping.")
                continue

            print(f"  Loaded ROC curve for: {label}")
            roc_panel_data.append({
                "model_name": model_short,
                "roc_df":     roc_df,
                "roc_auc":    roc_auc,
            })

        if not roc_panel_data:
            print(f"  WARNING: No ROC curve data for subset '{subset}' — skipping.")
            continue

        print(f"  Plotting ROC curve panel for {subset} set...")
        plot_roc_curve_panel(
            models_data=roc_panel_data,
            output_dir=OUTPUT_DIR,
            filename=f"roc_curve_panel_{subset}",
            title="ROC Curves",
        )

    # -------------------------------------------------------------------------
    # 5. Feature importance — one unified panel figure (threshold-independent)
    # -------------------------------------------------------------------------
    section("Plotting feature importances (panel)...")

    fi_panel_data = []
    for model_key, model_short in MODELS.items():
        fi_path = os.path.join(model_dir(model_short), "feature_importance.csv")

        if not os.path.exists(fi_path):
            print(f"  NOTE: No feature_importance.csv for '{model_short}' — skipping.")
            continue

        importance_df = pd.read_csv(fi_path)
        print(f"  Loaded feature importance for: {model_short}")
        fi_panel_data.append({
            "model_name":    model_short,
            "importance_df": importance_df,
        })

    if fi_panel_data:
        print("  Plotting feature importance panel...")
        plot_feature_importance_panel(
            models_data=fi_panel_data,
            output_dir=OUTPUT_DIR,
            filename="feature_importance_panel",
            title="Feature Importances — Final Models",
        )
    else:
        print("  WARNING: No feature importance data found — skipping panel.")

    # -------------------------------------------------------------------------
    # 6. Top-k coverage — one figure per k level per subset
    # -------------------------------------------------------------------------
    section("Computing and plotting top-k coverage...")

    for subset in SUBSETS:
        topk_results = []

        for model_key, model_short in MODELS.items():
            preds_path = os.path.join(model_dir(model_short), f"predictions_{subset}.csv")

            if not os.path.exists(preds_path):
                print(f"  WARNING: predictions_{subset}.csv not found for "
                      f"'{model_short}' — skipping.")
                continue

            preds_df = pd.read_csv(preds_path)

            if "y_prob" not in preds_df.columns or preds_df["y_prob"].isna().all():
                print(f"  WARNING: y_prob missing or all NaN for '{model_short}' "
                      f"({subset}) — skipping.")
                continue

            y_true = preds_df["y_true"].values
            y_prob = preds_df["y_prob"].values

            coverage, n_fires, n_total = compute_topk_coverage(y_true, y_prob, K_LEVELS)

            print(f"  Top-k coverage for {model_short} ({subset}):")
            for k, cov in zip(K_LEVELS, coverage):
                print(f"    Top {k*100:.0f}% : {cov*100:.2f}% of {n_fires:,} fires captured "
                      f"({int(np.ceil(k * n_total)):,} of {n_total:,} cells)")

            topk_results.append({
                "model":    f"{model_short}_final",
                "k_levels": K_LEVELS,
                "coverage": coverage,
                "n_fires":  n_fires,
                "n_total":  n_total,
            })

        if not topk_results:
            print(f"  WARNING: No top-k data available for subset '{subset}' — skipping plots.")
            continue

        print(f"\n  Plotting top-k figures for {subset} set...")
        plot_topk_coverage(
            topk_results=topk_results,
            k_levels=K_LEVELS,
            subset_label=subset,
            output_dir=OUTPUT_DIR,
            filename_prefix="topk_coverage",
        )

    print("\nAll plots complete.")


if __name__ == "__main__":
    main()