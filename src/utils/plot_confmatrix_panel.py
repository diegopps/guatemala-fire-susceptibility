"""
plot_confmatrix_panel.py
------------------------
Multi-panel wrapper around plot_confmatrix.py logic.
Plots confusion matrices for multiple models side by side in a single figure
(1 column, N rows layout).

Intended to be imported and called by a separate caller script.

Main function:
    plot_confusion_matrix_panel(models_data, output_dir, filename, title)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PLOT SETTINGS
# =============================================================================

DPI            = 300
OUTPUT_FORMATS = ["png", "pdf", "svg"]
CMAP           = "Blues"

# Text sizes — large for print/paper
TITLE_FONTSIZE      = 26
SUPTITLE_FONTSIZE   = 28
LABEL_FONTSIZE      = 22
TICK_FONTSIZE       = 20
ANNOT_MAIN_FONTSIZE = 20   # proportion value
ANNOT_SUB_FONTSIZE  = 14   # raw count
QUADRANT_FONTSIZE   = 12

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_confusion_matrix_panel(
    models_data: list[dict],
    output_dir:  str,
    filename:    str  = "confusion_matrix_panel",
    title:       str  = "Confusion Matrices — Top Models",
) -> None:
    """
    Parameters
    ----------
    models_data : list of dict, each with keys:
        - "model_name" : str   display name shown in subplot title
        - "cm"         : array-like (2,2)  [[TN, FP], [FN, TP]]
        - "threshold"  : float  shown in subplot title
    output_dir : str
    filename : str  base filename without extension
    title : str  figure suptitle
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(models_data)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(7, 7 * n),
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.5, top=0.93)
    if n == 1:
        axes = [axes]

    axis_labels = ["Negative", "Positive"]
    quadrant_labels = [["TN", "FP"], ["FN", "TP"]]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, (ax, entry) in enumerate(zip(axes, models_data)):
        cm_array = np.array(entry["cm"])
        row_sums = cm_array.sum(axis=1, keepdims=True)
        cm_prop  = cm_array / row_sums.astype(float)

        sns.heatmap(
            cm_prop,
            ax=ax,
            annot=False,
            cmap=CMAP,
            vmin=0.0, vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Proportion (row-normalised)", "shrink": 0.8},
        )

        # Fix colorbar label and tick font sizes
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE - 4)
        cbar.ax.yaxis.label.set_size(LABEL_FONTSIZE - 4)

        # Annotations: proportion + raw count
        for i in range(2):
            for j in range(2):
                prop_val  = cm_prop[i, j]
                count_val = cm_array[i, j]
                text_color  = "white" if prop_val > 0.55 else "#1a1a2e"
                count_color = "white" if prop_val > 0.55 else "#888888"
                q_color     = "white" if prop_val > 0.55 else "#cccccc"

                ax.text(j + 0.5, i + 0.38, f"{prop_val:.3f}",
                        ha="center", va="center",
                        fontsize=ANNOT_MAIN_FONTSIZE, fontweight="bold",
                        color=text_color)
                ax.text(j + 0.5, i + 0.63, f"n = {count_val:,}",
                        ha="center", va="center",
                        fontsize=ANNOT_SUB_FONTSIZE, color=count_color)
                ax.text(j + 0.08, i + 0.13, quadrant_labels[i][j],
                        ha="left", va="top",
                        fontsize=QUADRANT_FONTSIZE, color=q_color,
                        fontstyle="italic")

        ax.set_xticklabels(axis_labels, fontsize=TICK_FONTSIZE)
        ax.set_yticklabels(axis_labels, fontsize=TICK_FONTSIZE, rotation=0)
        ax.set_xlabel("Predicted", fontsize=LABEL_FONTSIZE, labelpad=8)
        ax.set_ylabel("Actual",    fontsize=LABEL_FONTSIZE, labelpad=8)
        ax.set_title(
            entry["model_name"],
            fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10,
        )
        # Row letter — floats to the left of the subplot title
        ax.text(
            -0.18, 1.04,
            f"{letters[idx]})",
            transform=ax.transAxes,
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")

    for fmt in OUTPUT_FORMATS:
        out = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(out, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)
