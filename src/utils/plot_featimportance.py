"""
plot_featimportance.py
--------------------------
Generic plotting module for a single model's feature/variable importance.

This script is purely a plotter — it receives a feature importance dataframe,
a model name, and an output directory, and produces the figure.

Intended to be imported and called by a separate caller script.

Handles both tree-based models (feature_importances_) and linear models
(coef_), as both are saved by model_testing.py with different column names.

Main function:
    plot_feature_importance(importance_df, model_name, output_dir)
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

# Figure height scales with number of features
FIGURE_WIDTH    = 6
BAR_HEIGHT      = 0.35   # inches per feature bar
FIGURE_HEIGHT_MIN = 3    # minimum figure height

DPI             = 300
OUTPUT_FORMATS  = ["png", "pdf", "svg"]

BAR_COLOR       = "#4393c3"
NEGATIVE_COLOR  = "#d6604d"   # for negative coefficients (linear models)

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_feature_importance(
    importance_df,
    model_name,
    output_dir,
    filename_prefix=None,
):
    """
    Create a horizontal bar plot of feature importances for a single model.
    Features are sorted by absolute importance value (descending).

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must contain a 'feature' column and either:
            - 'importance'  : for tree-based models
            - 'coefficient' : for linear models
    model_name : str
        Display name of the model (used in title and filename).
    output_dir : str
        Directory where output files will be saved.
    filename_prefix : str or None
        Base filename (without extension). Defaults to
        'feature_importance_<model_name>'.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename_prefix is None:
        filename_prefix = f"feature_importance_{model_name}"

    # --- Detect value column ---
    if "importance" in importance_df.columns:
        value_col  = "importance"
        x_label    = "Feature Importance"
        use_signed = False   # importances are always positive
    elif "coefficient" in importance_df.columns:
        value_col  = "coefficient"
        x_label    = "Coefficient (absolute value = importance)"
        use_signed = True    # coefficients can be negative
    else:
        raise ValueError(
            "importance_df must contain either 'importance' or 'coefficient' column."
        )

    # --- Sort by absolute value descending ---
    df = importance_df.copy()
    df["_abs"] = df[value_col].abs()
    df = df.sort_values("_abs", ascending=True)   # ascending=True → top bar = most important

    features = df["feature"].tolist()
    values   = df[value_col].tolist()
    n        = len(features)

    # --- Bar colors: negative coefficients get a different color ---
    if use_signed:
        colors = [NEGATIVE_COLOR if v < 0 else BAR_COLOR for v in values]
    else:
        colors = [BAR_COLOR] * n

    # --- Figure size scales with number of features ---
    fig_height = max(FIGURE_HEIGHT_MIN, n * BAR_HEIGHT + 1.2)

    # --- Apply style ---
    sns.set_theme(
        style=SEABORN_STYLE,
        context=SEABORN_CONTEXT,
        font=SEABORN_FONT,
    )


    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, fig_height))
    # Remove horizontal gridlines, keep vertical
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    
    y_pos = np.arange(n)
    bars  = ax.barh(
        y_pos,
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
    )

    # --- Value labels at end of each bar ---
    x_range  = max(abs(v) for v in values) if values else 1
    offset   = x_range * 0.015

    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        label = f"{val:.4f}"
        ax.text(
            x_pos + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=6.5,
            color="#333333",
        )

    # --- Reference line at 0 for signed coefficients ---
    if use_signed:
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

        # Legend for positive/negative
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=BAR_COLOR,      label="Positive coefficient"),
            Patch(facecolor=NEGATIVE_COLOR, label="Negative coefficient"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right", frameon=True)

    # --- Axis formatting ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel(x_label, fontsize=8)

    # Add padding on x-axis for labels
    x_min = min(values) if use_signed else 0
    x_max = max(values) if values else 1
    x_padding = x_range * 0.15
    ax.set_xlim(
        x_min - x_padding if use_signed else -x_padding * 0.5,
        x_max + x_padding,
    )

    ax.set_title(
        f"Feature Importance — {model_name}",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="x", labelsize=7)

    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    # --- Save ---
    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(output_dir, f"{filename_prefix}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)