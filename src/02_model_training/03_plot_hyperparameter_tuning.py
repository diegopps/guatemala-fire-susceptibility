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

PROJECT_DIR      = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR      = f"{PROJECT_DIR}/results/03_hyperparameter_tuning"
BASE_RESULTS_DIR = f"{PROJECT_DIR}/results/02_overundersampling"
OUTPUT_DIR       = f"{PROJECT_DIR}/results/03_hyperparameter_tuning/figures/"

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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

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


def load_base_metrics(model_short):
    """Load step-02 metrics for the sampler variant used in the tuned run."""
    tuned = load_metrics(model_short)
    if tuned is None:
        return None
    sampler = tuned.get("sampler")
    if not sampler:
        return None
    path = os.path.join(BASE_RESULTS_DIR, f"{model_short}_{sampler}", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# Metrics shown in the comparison figure
_METRICS = [
    ("balanced_accuracy", "Balanced Accuracy",                   (0, 1)),
    ("matthews_corrcoef", "Matthews Correlation Coefficient (MCC)", (-1, 1)),
    ("average_precision", "Average Precision",                   (0, 1)),
    ("roc_auc",           "ROC AUC",                             (0, 1)),
]


def plot_paired_comparison(tuned_list, base_list, output_dir, filename_prefix, title=None):
    """
    2x2 figure with grouped horizontal bars: one grey (base) + one color (tuned)
    per model per metric.  Order within each group: base on top, tuned below.
    """
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")

    n_models   = len(tuned_list)
    colors     = sns.color_palette("colorblind", n_colors=n_models)
    BASE_COLOR = "#aaaaaa"

    # Sort by tuned MCC descending so best model is at the top
    order = sorted(
        range(n_models),
        key=lambda i: tuned_list[i].get("matthews_corrcoef") or -999,
        reverse=True,
    )
    tuned_sorted  = [tuned_list[i]  for i in order]
    base_sorted   = [base_list[i]   for i in order]
    colors_sorted = [colors[i]      for i in order]

    BAR_H   = 0.35   # height of each individual bar
    SPACING = 0.9    # centre-to-centre distance between model groups

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.flatten()

    for ax, (metric_key, metric_label, x_range) in zip(axes_flat, _METRICS):
        x_span = x_range[1] - x_range[0]

        for idx, (tm, bm, color) in enumerate(zip(tuned_sorted, base_sorted, colors_sorted)):
            group_centre = idx * SPACING
            # base bar sits above (higher y), tuned bar below
            y_base  = group_centre + BAR_H / 2
            y_tuned = group_centre - BAR_H / 2

            v_tuned = tm.get(metric_key) or 0.0
            v_base  = bm.get(metric_key) or 0.0 if bm else 0.0

            ax.barh(y_base,  v_base,  height=BAR_H, color=BASE_COLOR, edgecolor="white", linewidth=0.5)
            ax.barh(y_tuned, v_tuned, height=BAR_H, color=color,      edgecolor="white", linewidth=0.5)

            offset = x_span * 0.015
            for y_pos, val, missing in [
                (y_base,  v_base,  bm is None or bm.get(metric_key) is None),
                (y_tuned, v_tuned, tm.get(metric_key) is None),
            ]:
                lbl = "N/A" if missing else f"{val:.3f}"
                ax.text(
                    val + offset, y_pos, lbl,
                    va="center", ha="left", fontsize=11, color="#333333",
                )

        # Y-axis: one tick per model group, labelled with model short name
        group_centres = [i * SPACING for i in range(n_models)]
        model_labels  = [tm["model"] for tm in tuned_sorted]
        ax.set_yticks(group_centres)
        ax.set_yticklabels(model_labels, fontsize=11)
        ax.set_xlim(x_range[0], x_range[1] + x_span * 0.15)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold", color="#1a1a2e")
        ax.invert_yaxis()

        if metric_key == "matthews_corrcoef":
            ax.axvline(0,   color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        else:
            ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.4)

        sns.despine(ax=ax, left=True, bottom=False)

    # Legend
    legend_handles = [mpatches.Patch(color=BASE_COLOR, label="Base (step 02)")]
    for color, tm in zip(colors_sorted, tuned_sorted):
        legend_handles.append(mpatches.Patch(color=color, label=f"{tm['model']} tuned"))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(n_models + 1, 5),
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.subplots_adjust(hspace=0.4)

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        out_path = os.path.join(output_dir, f"{filename_prefix}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)
    print("Done.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Metrics comparison — all three tuned models side by side
    # -------------------------------------------------------------------------
    section("Plotting metric comparison (all tuned models)...")

    tuned_metrics_list = []
    base_metrics_list  = []
    for model_key, model_short in MODELS.items():
        tuned = load_metrics(model_short)
        if tuned is None:
            print(f"  WARNING: metrics.json not found for tuned '{model_short}' — skipping.")
            continue
        tuned["model"] = model_short
        tuned_metrics_list.append(tuned)

        base = load_base_metrics(model_short)
        if base is None:
            print(f"  WARNING: base metrics not found for '{model_short}' — base bar will be zero.")
        base_metrics_list.append(base)

        sampler = tuned.get("sampler", "?")
        print(f"  Loaded tuned + base ({model_short}_{sampler}) for: {model_short}")

    if not tuned_metrics_list:
        print("ERROR: No metrics loaded. Check RESULTS_DIR and model short names.")
        sys.exit(1)

    print(f"\n  Plotting paired comparison for {len(tuned_metrics_list)} model(s)...")
    plot_paired_comparison(
        tuned_list=tuned_metrics_list,
        base_list=base_metrics_list,
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