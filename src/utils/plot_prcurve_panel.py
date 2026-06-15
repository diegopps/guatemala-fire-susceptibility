"""
plot_prcurve_panel.py
----------------------
Multi-panel wrapper for Precision-Recall curves.
Plots PR curves for multiple models in a single figure (N rows, 1 column).

Main function:
    plot_pr_curve_panel(models_data, output_dir, filename, title)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PLOT SETTINGS
# =============================================================================

DPI            = 300
OUTPUT_FORMATS = ["png", "pdf", "svg"]

CURVE_COLOR    = "#2166ac"
BASELINE_COLOR = "#888888"

SUPTITLE_FONTSIZE = 28
TITLE_FONTSIZE    = 24
LABEL_FONTSIZE    = 20
TICK_FONTSIZE     = 18
ANNOT_FONTSIZE    = 14

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_pr_curve_panel(
    models_data: list[dict],
    output_dir:  str,
    filename:    str = "pr_curve_panel",
    title:       str = "Precision-Recall Curves — Top Models",
) -> None:
    """
    Parameters
    ----------
    models_data : list of dict, each with keys:
        - "model_name"    : str
        - "pr_df"         : pd.DataFrame with columns 'recall', 'precision'
        - "avg_precision" : float
        - "prevalence"    : float
    output_dir : str
    filename : str  base filename without extension
    title : str  figure suptitle
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(models_data)
    fig, axes = plt.subplots(n, 1, figsize=(6, 5 * n), constrained_layout=False)
    fig.subplots_adjust(hspace=0.45, top=0.93)
    if n == 1:
        axes = [axes]

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, (ax, entry) in enumerate(zip(axes, models_data)):
        prevalence = entry["prevalence"]
        ax.axhline(
            y=prevalence,
            color=BASELINE_COLOR,
            linewidth=1.5,
            linestyle="--",
            label=f"No-skill (prevalence = {prevalence:.4f})",
        )
        ax.plot(
            entry["pr_df"]["recall"],
            entry["pr_df"]["precision"],
            color=CURVE_COLOR,
            linewidth=2.0,
            label="PR curve",
        )
        ax.text(
            0.97, 0.97,
            f"AP = {entry['avg_precision']:.3f}",
            transform=ax.transAxes,
            fontsize=ANNOT_FONTSIZE,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
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
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Precision", fontsize=LABEL_FONTSIZE)
        ax.set_title(
            entry["model_name"],
            fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10,
        )
        ax.legend(fontsize=TICK_FONTSIZE - 2, loc="upper right", frameon=True,
                  bbox_to_anchor=(0.97, 0.88))
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        sns.despine(ax=ax)

    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")

    for fmt in OUTPUT_FORMATS:
        out = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(out, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)
