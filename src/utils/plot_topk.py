"""
plot_topk.py
-------------
Generic plotter for top-k coverage analysis.

Produces one figure per subset showing all k levels on the x-axis with one
grouped bar per model, so coverage across k levels can be compared at a glance.

Function
--------
plot_topk_coverage(
    topk_results   : list[dict],
    k_levels       : list[float],
    subset_label   : str,
    output_dir     : str,
    filename_prefix: str = "topk_coverage",
)

Parameters
----------
topk_results : list of dicts, one per model. Each dict must contain:
    {
        "model"       : str,          # display label, e.g. "RF_final"
        "k_levels"    : list[float],  # e.g. [0.01, 0.02, 0.05, 0.10]
        "coverage"    : list[float],  # coverage fraction per k level (0–1)
        "n_fires"     : int,          # total number of fires in the subset
        "n_total"     : int,          # total number of grid cells in the subset
    }
k_levels       : list of float — k levels to plot, e.g. [0.01, 0.02, 0.05, 0.10]
subset_label   : str — used in figure title and filename, e.g. "test" or "val"
output_dir     : str — directory where figures are saved
filename_prefix: str — prefix for output filenames (default: "topk_coverage")

Outputs
-------
One PNG + PDF + SVG per subset, saved to output_dir.
Filename format: <filename_prefix>_<subset_label>.<ext>
  e.g. topk_coverage_val.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =============================================================================
# STYLE
# =============================================================================

sns.set_theme(style="whitegrid")

FIGURE_SIZE        = (9, 5)
DPI                = 300
FONT_SIZE_TITLE    = 13
FONT_SIZE_LABEL    = 11
FONT_SIZE_TICK     = 10
FONT_SIZE_ANNOTATE = 8

OUTPUT_FORMATS = ["png", "pdf", "svg"]

PALETTE = sns.color_palette("muted")


# =============================================================================
# HELPERS
# =============================================================================

def _k_to_label(k: float) -> str:
    pct = k * 100
    if pct == int(pct):
        return f"Top {int(pct)}%"
    return f"Top {pct:.1f}%"


# =============================================================================
# MAIN PLOTTER
# =============================================================================

def plot_topk_coverage(
    topk_results,
    k_levels,
    subset_label,
    output_dir,
    filename_prefix="topk_coverage",
):
    """
    Produce one figure showing coverage (%) for all k levels, with one grouped
    bar per model at each k level.

    Parameters: see module docstring.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_models  = len(topk_results)
    n_k       = len(k_levels)
    x         = np.arange(n_k)
    bar_width = 0.8 / n_models  # divide group width evenly among models

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, result in enumerate(topk_results):
        coverages = []
        for k in k_levels:
            if k not in result["k_levels"]:
                coverages.append(0.0)
                print(f"  WARNING: k={k} not found for model '{result['model']}' — using 0.")
            else:
                k_idx = result["k_levels"].index(k)
                coverages.append(result["coverage"][k_idx] * 100)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars   = ax.bar(
            x + offset,
            coverages,
            width=bar_width,
            label=result["model"],
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white",
            linewidth=0.6,
        )

        for bar, cov in zip(bars, coverages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{cov:.1f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_ANNOTATE,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_k_to_label(k) for k in k_levels], fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Fire Coverage (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Top-k Threshold", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.legend(title="Model", fontsize=FONT_SIZE_TICK, title_fontsize=FONT_SIZE_TICK)

    ax.set_title(
        f"Top-k Coverage — {subset_label.capitalize()} Set\n"
        f"Proportion of fires captured within top-k% of cells ranked by predicted probability",
        fontsize=FONT_SIZE_TITLE,
    )

    sns.despine(left=False, bottom=False)
    fig.tight_layout()

    base_name = f"{filename_prefix}_{subset_label}"
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)
