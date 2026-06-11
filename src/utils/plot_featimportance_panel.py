"""
plot_featimportance_panel.py
----------------------------
Multi-panel wrapper around plot_featimportance.py logic.
Plots feature importances for multiple models in a single figure
(1 column, N rows layout).

Intended to be imported and called by a separate caller script.

Main function:
    plot_feature_importance_panel(models_data, output_dir, filename, title)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# PLOT SETTINGS
# =============================================================================

DPI            = 300
OUTPUT_FORMATS = ["png", "pdf", "svg"]

BAR_COLOR      = "#4393c3"
NEGATIVE_COLOR = "#d6604d"

# Text sizes — large for print/paper
SUPTITLE_FONTSIZE = 28
TITLE_FONTSIZE    = 24
LABEL_FONTSIZE    = 20
TICK_FONTSIZE     = 18
ANNOT_FONTSIZE    = 14

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_feature_importance_panel(
    models_data: list[dict],
    output_dir:  str,
    filename:    str = "feature_importance_panel",
    title:       str = "Feature Importances — Top Models",
) -> None:
    """
    Parameters
    ----------
    models_data : list of dict, each with keys:
        - "model_name"    : str
        - "importance_df" : pd.DataFrame with 'feature' and either
                            'importance' or 'coefficient' column
    output_dir : str
    filename : str  base filename without extension
    title : str  figure suptitle
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(models_data)

    # Determine figure height: scale with the number of features in the
    # largest dataframe so bars don't get squished
    max_features = max(len(e["importance_df"]) for e in models_data)
    panel_height = max(6, max_features * 0.45 + 1.5)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(10, panel_height * n),
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.4, top=0.94)
    if n == 1:
        axes = [axes]

    for ax, entry in zip(axes, models_data):
        df       = entry["importance_df"].copy()
        m_name   = entry["model_name"]

        # Detect value column
        if "importance" in df.columns:
            value_col  = "importance"
            x_label    = "Feature Importance"
            use_signed = False
        elif "coefficient" in df.columns:
            value_col  = "coefficient"
            x_label    = "Coefficient"
            use_signed = True
        else:
            raise ValueError(f"No 'importance' or 'coefficient' column for {m_name}.")

        # Sort by absolute value ascending (so top bar = most important)
        df["_abs"] = df[value_col].abs()
        df = df.sort_values("_abs", ascending=True)

        features = df["feature"].tolist()
        values   = df[value_col].tolist()
        n_feat   = len(features)
        y_pos    = np.arange(n_feat)

        colors = (
            [NEGATIVE_COLOR if v < 0 else BAR_COLOR for v in values]
            if use_signed else [BAR_COLOR] * n_feat
        )

        bars = ax.barh(
            y_pos, values,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )

        # Value labels
        x_range = max(abs(v) for v in values) if values else 1
        offset  = x_range * 0.015
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + (offset if val >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=ANNOT_FONTSIZE,
                color="#333333",
            )

        if use_signed:
            ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
            legend_elements = [
                mpatches.Patch(facecolor=BAR_COLOR,      label="Positive"),
                mpatches.Patch(facecolor=NEGATIVE_COLOR, label="Negative"),
            ]
            ax.legend(handles=legend_elements, fontsize=TICK_FONTSIZE,
                      loc="lower right", frameon=True)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=TICK_FONTSIZE)
        ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)

        x_min = min(values) if use_signed else 0
        x_max = max(values) if values else 1
        x_pad = x_range * 0.18
        ax.set_xlim(
            x_min - x_pad if use_signed else -x_pad * 0.5,
            x_max + x_pad,
        )

        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
        sns.despine(ax=ax, left=True)

        ax.set_title(
            f"Feature Importance — {m_name}",
            fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10,
        )

    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")

    for fmt in OUTPUT_FORMATS:
        out = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(out, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)
