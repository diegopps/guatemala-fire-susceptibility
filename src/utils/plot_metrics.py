"""
plot_metrics.py
------------------------
Generic plotting module for comparing ML model performance metrics.

This script is purely a plotter. It only receives a list of metrics dicts and
an output directory, and produces the figures.

Intended to be imported and called by a separate caller script.

Main function:
    plot_model_comparison(metrics_list, output_dir, filename_prefix)
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

# Publication-style seaborn theme
SEABORN_STYLE   = "whitegrid"
SEABORN_CONTEXT = "paper"       # Options: paper, notebook, talk, poster
SEABORN_FONT    = "DejaVu Sans"

FIGURE_SIZE = (10, 8)   # inches (width, height)
DPI         = 300       # for PNG output

# Metrics to plot: (key in metrics dict, display label, x-axis range)
METRICS = [
    ("balanced_accuracy", "Balanced Accuracy", (0, 1)),
    ("matthews_corrcoef", "Matthews Correlation Coefficient (MCC)", (-1, 1)),
    ("average_precision", "Average Precision",  (0, 1)),
    ("roc_auc",           "ROC AUC",            (0, 1)),
]

# Font sizes — large for print/paper
TITLE_FONTSIZE  = 12
LABEL_FONTSIZE  = 15
TICK_FONTSIZE   = 12
ANNOT_FONTSIZE  = 13
LEGEND_FONTSIZE = 13

# Color palette — one color per model, publication-friendly
COLOR_PALETTE = "colorblind"   # seaborn colorblind-safe palette

OUTPUT_FORMATS = ["png", "pdf", "svg"]

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_model_comparison(
    metrics_list,
    output_dir,
    filename_prefix="model_comparison",
    title=None,
):
    """
    Create a 2x2 figure of horizontal bar plots comparing model metrics.

    Parameters
    ----------
    metrics_list : list of dict
        Each dict must contain:
            - "model"             : str, model display name
            - "balanced_accuracy" : float
            - "matthews_corrcoef" : float
            - "average_precision" : float or None
            - "roc_auc"           : float or None
    output_dir : str
        Directory where output files will be saved.
    filename_prefix : str
        Base filename (without extension) for saved figures.
    title : str or None
        Optional super-title for the figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Apply style ---
    sns.set_theme(
        style=SEABORN_STYLE,
        context=SEABORN_CONTEXT,
        font=SEABORN_FONT,
    )

    # --- Prepare data ---
    model_names = [m["model"] for m in metrics_list]
    n_models    = len(model_names)
    colors      = sns.color_palette(COLOR_PALETTE, n_colors=n_models)

    # Sort models by MCC descending for consistent ordering across subplots
    sorted_indices  = sorted(
        range(n_models),
        key=lambda i: metrics_list[i].get("matthews_corrcoef") or -999,
        reverse=True,
    )
    sorted_names    = [model_names[i] for i in sorted_indices]
    sorted_colors   = [colors[i] for i in sorted_indices]
    sorted_metrics  = [metrics_list[i] for i in sorted_indices]

    # --- Build figure ---
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    axes_flat = axes.flatten()

    for ax, (metric_key, metric_label, x_range) in zip(axes_flat, METRICS):
        values = [
            m.get(metric_key) if m.get(metric_key) is not None else 0.0
            for m in sorted_metrics
        ]
        missing = [m.get(metric_key) is None for m in sorted_metrics]

        y_pos = np.arange(len(sorted_names))

        bars = ax.barh(
            y_pos,
            values,
            color=sorted_colors,
            edgecolor="white",
            linewidth=0.6,
            height=0.5,
        )

        # Value labels at end of each bar
        for bar, val, is_missing in zip(bars, values, missing):
            label = "N/A" if is_missing else f"{val:.3f}"
            x_label = bar.get_width()
            # Place label just outside the bar
            offset = (x_range[1] - x_range[0]) * 0.015
            ax.text(
                x_label + offset,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=ANNOT_FONTSIZE,
                color="#333333",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=TICK_FONTSIZE)
        ax.set_xlim(x_range[0], x_range[1] + (x_range[1] - x_range[0]) * 0.12)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)

        # All subplot titles bold
        ax.set_title(metric_label, fontsize=TITLE_FONTSIZE, fontweight="bold",
                     color="#1a1a2e")

        # Reference line at 0 for MCC
        if metric_key == "matthews_corrcoef":
            ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

        # Reference line at 0.5 for accuracy-like metrics
        if metric_key in ("balanced_accuracy", "roc_auc", "average_precision"):
            ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.4)

        sns.despine(ax=ax, left=True, bottom=False)
        ax.invert_yaxis()   # best model at top

    # --- Legend ---
    legend_patches = [
        mpatches.Patch(color=c, label=n)
        for c, n in zip(sorted_colors, sorted_names)
    ]
    fig.legend(
        handles=legend_patches,
        title="Model",
        loc="lower center",
        ncol=min(n_models, 4),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )

    # --- Super title ---
    if title:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE + 4, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.subplots_adjust(hspace=0.4)

    # --- Save ---
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(output_dir, f"{filename_prefix}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)
    print("Done.")
