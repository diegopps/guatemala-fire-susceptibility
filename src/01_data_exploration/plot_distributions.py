"""
plot_distributions.py
---------------------
Generates two  grid figures from the fire dataset:
  1. boxplots_grid.png  — one boxplot per predictor variable
  2. histograms_grid.png — one histogram per predictor variable

Dependencies:
    pip install pandas matplotlib seaborn
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

INPUT_FILE = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv"    
OUTPUT_DIR = "../../figures/01_data_exploration"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Variables to plot and their display labels — split by type
VARIABLES_TEMPORAL = {
    "tas":        "Mean Temperature (°C)",
    "tasmax":     "Max Temperature (°C)",
    "tasmin":     "Min Temperature (°C)",
    "pr":         "Precipitation (mm/day)",
    "rsds":       "Solar Radiation (W/m²)",
    "PET":        "Potential Evapotranspiration",
    "climbal":    "Climatic Water Balance",
}

VARIABLES_STATIC = {
    "elev":       "Elevation (m)",
    "slope":      "Slope (°)",
    "aspect_sin":     "Aspect Sine",
    "aspect_cos":     "Aspect Cosine",
    "distRoads":  "Distance to Roads (m)",
    "distUrban":  "Distance to Urban Areas (m)",
    "distWater":  "Distance to Water Bodies (m)",
}

# Combined — used only for loading/validation
VARIABLES = {**VARIABLES_TEMPORAL, **VARIABLES_STATIC}

NCOLS = 3
FIG_WIDTH = 15
COLOR = "#2E86AB"       # consistent fill colour for all plots
FLIER_COLOR = "#E84855" # outlier colour for boxplots

# Seaborn theme — whitegrid gives clean horizontal reference lines
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize":  11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [v for v in VARIABLES if v not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")
    return df


def make_grid(n_vars: int, ncols: int, row_height: float = 3.2):
    nrows = -(-n_vars // ncols)          # ceiling division
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIG_WIDTH, nrows * row_height),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    # Hide unused axes
    for ax in axes_flat[n_vars:]:
        ax.set_visible(False)
    return fig, axes_flat


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_boxplots(df: pd.DataFrame, output_path: str, variables: dict, title: str):
    vars_list = list(variables.keys())
    labels    = list(variables.values())
    # Apply unit conversions where needed
    converted = {}
    for var in vars_list:
        data = df[var].dropna()
        if var in ("tas", "tasmax", "tasmin"):
            data = data - 273.15          # K → °C
        elif var == "pr":
            data = data * 86400           # kg/m²/s → mm/day
        converted[var] = data

    fig, axes = make_grid(len(vars_list), NCOLS, row_height=4.2)

    for ax, var, label in zip(axes, vars_list, labels):
        data = converted[var]
        bp = ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            widths=0.5,
            flierprops=dict(
                marker="o", markersize=2,
                markerfacecolor=FLIER_COLOR,
                markeredgecolor=FLIER_COLOR,
                alpha=0.4,
            ),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#555555"),
            capprops=dict(color="#555555"),
            boxprops=dict(facecolor=COLOR, color=COLOR, alpha=0.85),
        )
        ax.set_title(label, pad=6)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.grid(axis="y", linewidth=0.5, alpha=0.7)

    fig.suptitle(title, fontsize=16, y=1.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_histograms(df: pd.DataFrame, output_path: str, variables: dict, title: str):
    vars_list = list(variables.keys())
    labels    = list(variables.values())
    # Apply unit conversions where needed
    converted = {}
    for var in vars_list:
        data = df[var].dropna()
        if var in ("tas", "tasmax", "tasmin"):
            data = data - 273.15          # K → °C
        elif var == "pr":
            data = data * 86400           # kg/m²/s → mm/day
        converted[var] = data

    fig, axes = make_grid(len(vars_list), NCOLS)

    for ax, var, label in zip(axes, vars_list, labels):
        data = converted[var]
        ax.hist(
            data,
            bins=50,
            color=COLOR,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_title(label, pad=6)
        ax.set_ylabel("Count", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.grid(axis="y", linewidth=0.5, alpha=0.7)

    fig.suptitle(title, fontsize=16, y=1.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data(INPUT_FILE)
    print(f"  {len(df):,} rows loaded.")

    print("Plotting boxplots — temporal variables...")
    plot_boxplots(df, os.path.join(OUTPUT_DIR, "boxplots_temporal.png"),
                  VARIABLES_TEMPORAL, "Boxplots — Temporal Variables")

    print("Plotting boxplots — static variables...")
    plot_boxplots(df, os.path.join(OUTPUT_DIR, "boxplots_static.png"),
                  VARIABLES_STATIC, "Boxplots — Static Variables")

    print("Plotting histograms — temporal variables...")
    plot_histograms(df, os.path.join(OUTPUT_DIR, "histograms_temporal.png"),
                    VARIABLES_TEMPORAL, "Histograms — Temporal Variables")

    print("Plotting histograms — static variables...")
    plot_histograms(df, os.path.join(OUTPUT_DIR, "histograms_static.png"),
                    VARIABLES_STATIC, "Histograms — Static Variables")

    print("Done.")


if __name__ == "__main__":
    main()