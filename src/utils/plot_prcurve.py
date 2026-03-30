"""
plot_prcurve.py
----------------
Generic plotting module for a single model's Precision-Recall curve.

This script is purely a plotter — it receives a PR curve dataframe,
a model name, average precision score, class prevalence, and an output
directory, and produces the figure.

Intended to be imported and called by a separate caller script.

Main function:
    plot_pr_curve(pr_df, model_name, avg_precision, prevalence, output_dir)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PLOT SETTINGS
# =============================================================================

SEABORN_STYLE   = "whitegrid"
SEABORN_CONTEXT = "paper"
SEABORN_FONT    = "DejaVu Sans"

FIGURE_SIZE = (5, 4)
DPI         = 300

OUTPUT_FORMATS = ["png", "pdf", "svg"]

# Curve color
CURVE_COLOR     = "#2166ac"
BASELINE_COLOR  = "#888888"

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_pr_curve(
    pr_df,
    model_name,
    avg_precision,
    prevalence,
    output_dir,
    filename_prefix=None,
):
    """
    Create a Precision-Recall curve figure for a single model.

    Parameters
    ----------
    pr_df : pd.DataFrame
        Must contain columns: 'precision', 'recall', 'threshold'.
    model_name : str
        Display name of the model (used in title and filename).
    avg_precision : float
        Average precision score (AP), annotated as a text box.
    prevalence : float
        Positive class prevalence in the test set (used as no-skill baseline).
    output_dir : str
        Directory where output files will be saved.
    filename_prefix : str or None
        Base filename (without extension). Defaults to
        'pr_curve_<model_name>'.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename_prefix is None:
        filename_prefix = f"pr_curve_{model_name}"

    # --- Apply style ---
    sns.set_theme(
        style=SEABORN_STYLE,
        context=SEABORN_CONTEXT,
        font=SEABORN_FONT,
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # --- No-skill baseline ---
    ax.axhline(
        y=prevalence,
        color=BASELINE_COLOR,
        linewidth=1.2,
        linestyle="--",
        label=f"No-skill baseline (prevalence = {prevalence:.4f})",
    )

    # --- PR curve ---
    ax.plot(
        pr_df["recall"],
        pr_df["precision"],
        color=CURVE_COLOR,
        linewidth=1.8,
        label="PR curve",
    )

    # --- Average Precision text box ---
    ax.text(
        0.97, 0.97,
        f"AP = {avg_precision:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    # --- Formatting ---
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(
        f"Precision-Recall Curve — {model_name}",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=7, loc="upper right", frameon=True,
              bbox_to_anchor=(0.97, 0.88))
    ax.tick_params(axis="both", labelsize=7)

    sns.despine(ax=ax)
    plt.tight_layout()

    # --- Save ---
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(output_dir, f"{filename_prefix}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)