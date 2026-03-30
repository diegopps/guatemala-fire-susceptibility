"""
plot_roccurve.py
-----------------
Generic plotting module for a single model's ROC curve.

This script is purely a plotter — it receives a ROC curve dataframe,
a model name, ROC AUC score, and an output directory, and produces
the figure.

Intended to be imported and called by a separate caller script.

Main function:
    plot_roc_curve(roc_df, model_name, roc_auc, output_dir)
"""

import os
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

CURVE_COLOR    = "#d6604d"
BASELINE_COLOR = "#888888"

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_roc_curve(
    roc_df,
    model_name,
    roc_auc,
    output_dir,
    filename_prefix=None,
):
    """
    Create a ROC curve figure for a single model.

    Parameters
    ----------
    roc_df : pd.DataFrame
        Must contain columns: 'fpr', 'tpr', 'threshold'.
    model_name : str
        Display name of the model (used in title and filename).
    roc_auc : float
        ROC AUC score, annotated as a text box.
    output_dir : str
        Directory where output files will be saved.
    filename_prefix : str or None
        Base filename (without extension). Defaults to
        'roc_curve_<model_name>'.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename_prefix is None:
        filename_prefix = f"roc_curve_{model_name}"

    # --- Apply style ---
    sns.set_theme(
        style=SEABORN_STYLE,
        context=SEABORN_CONTEXT,
        font=SEABORN_FONT,
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # --- No-skill baseline (diagonal) ---
    ax.plot(
        [0, 1], [0, 1],
        color=BASELINE_COLOR,
        linewidth=1.2,
        linestyle="--",
        label="No-skill baseline (AUC = 0.500)",
    )

    # --- ROC curve ---
    ax.plot(
        roc_df["fpr"],
        roc_df["tpr"],
        color=CURVE_COLOR,
        linewidth=1.8,
        label="ROC curve",
    )

    # --- ROC AUC text box ---
    ax.text(
        0.97, 0.15,
        f"AUC = {roc_auc:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
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
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(
        f"ROC Curve — {model_name}",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=7, loc="lower right", frameon=True)
    ax.tick_params(axis="both", labelsize=7)

    sns.despine(ax=ax)
    plt.tight_layout()

    # --- Save ---
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(output_dir, f"{filename_prefix}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)