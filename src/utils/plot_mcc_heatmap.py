"""
plot_mcc_heatmap.py
-------------------
Utility for visualising MCC performance across all model × threshold
combinations. Produces two figures saved to output_dir:

  1. Heatmap  — rows = models, columns = thresholds, colour = MCC.
               Top cell(s) are annotated with a star marker.
               Intended as the main paper figure.

  2. Line plot — one line per model, threshold on the x-axis, MCC on y.
                Shows how sensitive each model is to threshold choice.
                Useful as a supplementary diagnostic.

Usage:
    from plot_mcc_heatmap import plot_mcc_heatmap

    # mcc_matrix: {model_name: {threshold: mcc_value_or_None}}
    plot_mcc_heatmap(
        mcc_matrix = mcc_matrix,
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        output_dir = "./figures/",
        filename   = "baseline_mcc_summary",
        title      = "MCC by Model and Threshold",
    )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titlesize":  28,
    "axes.labelsize":  20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi":      150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUTPUT_FORMATS = ["png", "pdf", "svg"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_dataframe(mcc_matrix: dict, thresholds: list) -> pd.DataFrame:
    """
    Convert the nested dict {model: {threshold: mcc}} into a tidy DataFrame
    with shape (n_models, n_thresholds), rows sorted by peak MCC descending.
    """
    df = pd.DataFrame(mcc_matrix, index=thresholds).T   # models × thresholds
    df.columns = [str(c) for c in df.columns]
    df = df.astype(float)

    # Sort rows by the best MCC each model achieves across all thresholds
    df = df.loc[df.max(axis=1).sort_values(ascending=False).index]
    return df


def _friendly_name(raw: str) -> str:
    return raw.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Figure 1 — Heatmap
# ---------------------------------------------------------------------------

def _plot_heatmap(df: pd.DataFrame, output_dir: str, filename: str, title: str) -> None:
    n_models, n_thresh = df.shape

    fig, ax = plt.subplots(figsize=(max(10, n_thresh * 1.2), max(6, n_models * 0.9)))

    cmap = "YlOrRd"

    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=0.1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 11, "color": "black"},
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={"label": "MCC", "shrink": 0.8},
    )

    
    for flat_i in np.argwhere(df.values >= 0.045).tolist():
        row, col = flat_i
        ax.text(
            col + 0.5, row + 0.18,
            "*",
            ha="center", va="center",
            fontsize=14, color="#1a1a2e",
            fontweight="bold",
        )

    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(
        [_friendly_name(m) for m in df.index],
        rotation=0,
    )
    ax.set_xlabel("Decision Threshold", labelpad=10)
    ax.set_ylabel("Model", labelpad=10)
    ax.set_title(title, pad=14, fontweight="bold", fontsize=16)

    plt.tight_layout()

    for fmt in OUTPUT_FORMATS:
        out = os.path.join(output_dir, f"{filename}_heatmap.{fmt}")
        fig.savefig(out, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Line plot
# ---------------------------------------------------------------------------

def _plot_lineplot(df: pd.DataFrame, output_dir: str, filename: str, title: str) -> None:
    n_models   = len(df)
    thresholds = [float(c) for c in df.columns]
    colors     = sns.color_palette("colorblind", n_colors=n_models)

    fig, ax = plt.subplots(figsize=(10, 6))

    for (model_name, row), color in zip(df.iterrows(), colors):
        values = row.values.astype(float)
        ax.plot(
            thresholds, values,
            label=_friendly_name(model_name),
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            zorder=3,
        )

        # Mark each model's personal peak with a larger outlined dot
        valid_mask = ~np.isnan(values)
        if valid_mask.any():
            best_idx = np.nanargmax(values)
            ax.scatter(
                thresholds[best_idx], values[best_idx],
                color=color,
                s=120,
                zorder=5,
                edgecolors="#1a1a2e",
                linewidths=1.2,
            )

    # Reference line at MCC = 0
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("MCC")
    ax.set_title(title, pad=14, fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(ticker.FixedLocator(thresholds))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.6)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        framealpha=0.9,
        title="Model",
        title_fontsize=13,
    )

    plt.tight_layout()

    for fmt in OUTPUT_FORMATS:
        out = os.path.join(output_dir, f"{filename}_lineplot.{fmt}")
        fig.savefig(out, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_mcc_heatmap(
    mcc_matrix: dict,
    thresholds: list,
    output_dir: str,
    filename:   str = "mcc_threshold_summary",
    title:      str = "MCC by Model and Threshold",
) -> None:
    """
    Generate both the heatmap and line plot for MCC across all
    model x threshold combinations.

    Parameters
    ----------
    mcc_matrix : dict
        Nested dict of shape {model_name: {threshold: mcc_value_or_None}}.
        Missing values (None or NaN) are handled gracefully.
    thresholds : list
        Ordered list of threshold values — used to set column order.
    output_dir : str
        Directory where output files will be saved.
    filename : str
        Base filename (without extension). Suffixes _heatmap and _lineplot
        are appended automatically.
    title : str
        Title string shared by both figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = _build_dataframe(mcc_matrix, thresholds)

    print("  Building heatmap...")
    _plot_heatmap(df, output_dir, filename, title)

    print("  Building line plot...")
    _plot_lineplot(df, output_dir, filename, title)
