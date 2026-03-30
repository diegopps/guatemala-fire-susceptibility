"""
plot_confmatrix.py
------------------------
Generic plotting module for a single model's confusion matrix.

This script is purely a plotter — it receives a confusion matrix dataframe
or array, a model name, and an output directory, and produces the figure.

Intended to be imported and called by a separate caller script.

Main function:
    plot_confusion_matrix(cm_df, model_name, output_dir, filename_prefix)
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Colormap for proportions (main)
CMAP_PROPORTIONS = "Blues"

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_confusion_matrix(
    cm,
    model_name,
    output_dir,
    filename_prefix=None,
):
    """
    Create a confusion matrix figure showing proportions (main) and
    raw counts (secondary, smaller and lighter).

    Parameters
    ----------
    cm : array-like of shape (2, 2) or pd.DataFrame
        Confusion matrix with layout:
            [[TN, FP],
             [FN, TP]]
        If a DataFrame, expected index/columns are
        ['actual_negative', 'actual_positive'] /
        ['predicted_negative', 'predicted_positive'].
    model_name : str
        Display name of the model (used in title and filename).
    output_dir : str
        Directory where output files will be saved.
    filename_prefix : str or None
        Base filename (without extension). Defaults to
        'confusion_matrix_<model_name>'.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename_prefix is None:
        filename_prefix = f"confusion_matrix_{model_name}"

    # --- Convert to numpy array ---
    cm_array = np.array(cm)  # shape (2, 2): [[TN, FP], [FN, TP]]

    # --- Compute proportions (row-normalised = recall perspective) ---
    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_prop  = cm_array / row_sums.astype(float)

    # --- Labels ---
    axis_labels = ["Negative", "Positive"]

    # --- Apply style ---
    sns.set_theme(
        style=SEABORN_STYLE,
        context=SEABORN_CONTEXT,
        font=SEABORN_FONT,
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # --- Heatmap of proportions ---
    sns.heatmap(
        cm_prop,
        annot=False,          # we add custom annotations below
        fmt="",
        cmap=CMAP_PROPORTIONS,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Proportion (row-normalised)"},
    )

    # --- Custom annotations: proportion (main) + raw count (secondary) ---
    for i in range(2):
        for j in range(2):
            prop_val  = cm_prop[i, j]
            count_val = cm_array[i, j]

            # Choose text color for readability against background
            text_color = "white" if prop_val > 0.55 else "#1a1a2e"
            count_color = "white" if prop_val > 0.55 else "#888888"

            # Main: proportion
            ax.text(
                j + 0.5, i + 0.38,
                f"{prop_val:.3f}",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=text_color,
            )

            # Secondary: raw count (smaller, lighter)
            ax.text(
                j + 0.5, i + 0.63,
                f"n = {count_val:,}",
                ha="center", va="center",
                fontsize=7.5,
                color=count_color,
            )

    # --- Axis labels ---
    ax.set_xticklabels(axis_labels, fontsize=9)
    ax.set_yticklabels(axis_labels, fontsize=9, rotation=0)
    ax.set_xlabel("Predicted", fontsize=9, labelpad=8)
    ax.set_ylabel("Actual", fontsize=9, labelpad=8)
    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=10, fontweight="bold", pad=10,
    )

    # --- Annotate quadrants subtly ---
    quadrant_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            prop_val = cm_prop[i, j]
            q_color = "white" if prop_val > 0.55 else "#cccccc"
            ax.text(
                j + 0.08, i + 0.13,
                quadrant_labels[i][j],
                ha="left", va="top",
                fontsize=7, color=q_color,
                fontstyle="italic",
            )

    plt.tight_layout()

    # --- Save ---
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(output_dir, f"{filename_prefix}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)