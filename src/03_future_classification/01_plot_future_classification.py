"""
01_plot_future_risk.py
-----------------------
Generates fire risk classification maps for all future climate scenarios
using predictions from step 07 and Jenks breaks from step 05.

For each model this script produces two sets of figures:

  1. DECADAL SEASONAL MAPS (results / appendix)
     For each decade (2020s–2090s) and each season (dry/wet):
       - One figure per season per decade
       - All 3 SSP scenarios shown as columns within each figure
       - Rows = models (RF, XGB, XT)
     Dry season : Nov–Apr | Wet season : May–Oct

  2. MONTHLY CLIMATOLOGY MAPS (appendix)
     For each decade and each calendar month (1–12):
       - One figure per month per decade
       - All 3 SSP scenarios shown as columns
       - Rows = models (RF, XGB, XT)
     Each cell value = mean risk class across all years within the decade.

Jenks breaks are loaded from each model's jenks_breaks.json (computed in
step 05 on training set predictions) and applied consistently here.

Historical data (1961–2014) is NOT plotted — future scenarios only.

Usage:
    python 01_plot_future_risk.py

No CLI arguments — runs all three models and produces all figures.

Output structure:
    results/01_future_risk_maps/
        seasonal/
            dry/
                decade_2020s_dry.{png,pdf,svg}
                decade_2030s_dry.{png,pdf,svg}
                ...
            wet/
                decade_2020s_wet.{png,pdf,svg}
                ...
        climatology/
            decade_2020s/
                month_01_january.{png,pdf,svg}
                ...
            decade_2030s/
                ...
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR      = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
PREDICTIONS_DIR  = f"{PROJECT_DIR}/results/01_future_classification"
JENKS_DIR        = f"{PROJECT_DIR}/results/04_final_model_training"
OUTPUT_DIR       = f"{PROJECT_DIR}/results/01_future_risk_maps"

# Models — order determines row order in multi-model figures
MODELS = {
    "random_forest": "RF",
    "xgboost":       "XGB",
    "extra_trees":   "XT",
}

# SSP scenarios to plot — historical excluded from figures
SSP_SCENARIOS = ["ssp126", "ssp370", "ssp585"]

# Human-readable scenario labels for figure titles and axis labels
SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6\nSustainability",
    "ssp370": "SSP3-7.0\nRegional Rivalry",
    "ssp585": "SSP5-8.5\nFossil-Fuelled Dev.",
}

# Decades to plot — each decade is defined as a 10-year range
DECADES = {
    "2020s": range(2020, 2030),
    "2030s": range(2030, 2040),
    "2040s": range(2040, 2050),
    "2050s": range(2050, 2060),
    "2060s": range(2060, 2070),
    "2070s": range(2070, 2080),
    "2080s": range(2080, 2090),
    "2090s": range(2090, 2100),
}

# Season definitions
SEASONS = {
    "dry": {
        "months":    [11, 12, 1, 2, 3, 4],
        "label":     "Dry Season (Nov–Apr)",
        "filename":  "dry",
    },
    "wet": {
        "months":    [5, 6, 7, 8, 9, 10],
        "label":     "Wet Season (May–Oct)",
        "filename":  "wet",
    },
}

# Month names for climatology figures
MONTH_NAMES = {
    1: "January",   2: "February",  3: "March",
    4: "April",     5: "May",       6: "June",
    7: "July",      8: "August",    9: "September",
    10: "October",  11: "November", 12: "December",
}

# Risk class colours — consistent with steps 05/06
RISK_COLORS = {
    1: "#ffffb2",   # Very low
    2: "#fecc5c",   # Low
    3: "#fd8d3c",   # Medium
    4: "#f03b20",   # High
    5: "#bd0026",   # Very high
}
RISK_LABELS = {
    1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Very high"}

N_RISK_CLASSES = 5

# File naming pattern for prediction CSVs
PRED_FILENAME_PATTERN = "{scenario}_{year}_predictions.csv"

# Output formats
SAVE_FORMATS = ["png", "pdf", "svg"]
DPI = 300

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from jenks_breaks import load_breaks, apply_jenks_breaks
from plot_risk_map import _build_raster

# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def subsection(title):
    print(f"\n  --- {title} ---")


def load_jenks_breaks(model_short):
    """Load pre-computed Jenks breaks for a model from step 05 output."""
    path = os.path.join(JENKS_DIR, model_short, "jenks_breaks.json")
    if not os.path.exists(path):
        print(f"  ERROR: jenks_breaks.json not found for {model_short} at {path}")
        return None
    return load_breaks(path)


def load_decade_predictions(model_short, scenario, decade_years):
    """
    Load and concatenate all yearly prediction CSVs for a given
    model / scenario / decade combination.
    Returns a DataFrame with lat, lon, year, month, y_prob columns,
    or None if no files were found.
    """
    frames = []
    pred_dir = os.path.join(PREDICTIONS_DIR, model_short, scenario)

    for year in decade_years:
        filename = PRED_FILENAME_PATTERN.format(scenario=scenario, year=year)
        filepath = os.path.join(pred_dir, filename)

        if not os.path.exists(filepath):
            print(f"    WARNING: Missing {filename} — skipping year {year}.")
            continue

        df = pd.read_csv(filepath, usecols=["lat", "lon", "year", "month", "y_prob"])
        frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def apply_breaks_to_df(df, breaks):
    """Apply Jenks breaks to y_prob column, adding a risk_class column."""
    df = df.copy()
    df["risk_class"] = apply_jenks_breaks(df["y_prob"].values, breaks)
    return df


def build_risk_colormap():
    """Build a discrete ListedColormap and BoundaryNorm for risk classes 1–5."""
    colors = [RISK_COLORS[i] for i in range(1, N_RISK_CLASSES + 1)]
    cmap   = mcolors.ListedColormap(colors)
    norm   = mcolors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                                   ncolors=N_RISK_CLASSES)
    return cmap, norm


def build_legend_patches():
    """Build legend patch handles for risk classes 1–5."""
    return [
        mpatches.Patch(facecolor=RISK_COLORS[i], label=RISK_LABELS[i])
        for i in range(1, N_RISK_CLASSES + 1)
    ]


def save_figure(fig, out_dir, filename_stem):
    """Save a figure in all configured formats."""
    os.makedirs(out_dir, exist_ok=True)
    for fmt in SAVE_FORMATS:
        out_path = os.path.join(out_dir, f"{filename_stem}.{fmt}")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"    Saved: {filename_stem}.{{png,pdf,svg}} → {out_dir}/")


def plot_risk_raster_ax(ax, df_subset, breaks, title):
    """
    Render a mean risk class raster on a given Axes object.
    df_subset must have lat, lon, risk_class columns.
    Mean risk class per cell is computed before rasterising.
    """
    # Average risk class across all months/years in subset
    mean_risk = (
        df_subset
        .groupby(["lat", "lon"])["risk_class"]
        .mean()
        .reset_index()
        .rename(columns={"risk_class": "risk_mean"})
    )
    # Round to nearest integer class for display
    mean_risk["risk_class"] = mean_risk["risk_mean"].round().astype(int).clip(1, 5)

    cmap, norm = build_risk_colormap()

    try:
        lats, lons, grid = _build_raster(
            mean_risk, lat_col="lat", lon_col="lon", value_col="risk_class"
        )
        ax.pcolormesh(lons, lats, grid, cmap=cmap, norm=norm,
                      shading="nearest", rasterized=True)
    except Exception as e:
        ax.text(0.5, 0.5, f"Raster error:\n{e}",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)

    ax.set_title(title, fontsize=8, pad=4)
    ax.set_aspect("equal")
    ax.axis("off")


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_decadal_seasonal_maps(all_breaks):
    """
    Produce decadal seasonal maps.
    Layout: rows = models, columns = SSP scenarios.
    One figure per decade × season combination.
    """
    section("Plotting decadal seasonal maps...")

    cmap, norm   = build_risk_colormap()
    legend_patches = build_legend_patches()

    n_rows = len(MODELS)
    n_cols = len(SSP_SCENARIOS)

    for decade_label, decade_years in DECADES.items():
        for season_key, season_info in SEASONS.items():

            fig_filename = f"decade_{decade_label}_{season_info['filename']}"
            out_dir      = os.path.join(OUTPUT_DIR, "seasonal", season_info["filename"])

            # Skip if already exists
            if os.path.exists(os.path.join(out_dir, f"{fig_filename}.png")):
                print(f"  SKIP: {fig_filename} (already exists)")
                continue

            subsection(f"{decade_label} — {season_info['label']}")

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 4.5 * n_rows),
            )
            # Ensure axes is always 2D
            if n_rows == 1:
                axes = axes[np.newaxis, :]
            if n_cols == 1:
                axes = axes[:, np.newaxis]

            fig.suptitle(
                f"Mean Fire Risk — {season_info['label']} | {decade_label}",
                fontsize=13, fontweight="bold", y=1.01,
            )

            for row_idx, (model_key, model_short) in enumerate(MODELS.items()):
                breaks = all_breaks.get(model_short)

                for col_idx, scenario in enumerate(SSP_SCENARIOS):
                    ax = axes[row_idx, col_idx]

                    # Column header (scenario) — top row only
                    if row_idx == 0:
                        ax.set_title(
                            SCENARIO_LABELS[scenario], fontsize=9, pad=6)

                    # Row label (model) — left column only
                    if col_idx == 0:
                        ax.set_ylabel(model_short, fontsize=9, labelpad=6)

                    if breaks is None:
                        ax.text(0.5, 0.5, "No Jenks breaks",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=8)
                        ax.axis("off")
                        continue

                    # Load decade predictions
                    df = load_decade_predictions(
                        model_short, scenario, decade_years)
                    if df is None:
                        ax.text(0.5, 0.5, "No data",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=8)
                        ax.axis("off")
                        continue

                    # Filter to season months
                    df_season = df[df["month"].isin(season_info["months"])]
                    if df_season.empty:
                        ax.text(0.5, 0.5, "No data (season filter)",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=8)
                        ax.axis("off")
                        continue

                    # Apply Jenks breaks
                    df_season = apply_breaks_to_df(df_season, breaks)

                    # Render raster — no title on individual panels (handled above)
                    plot_risk_raster_ax(ax, df_season, breaks, title="")

            # Shared legend below figure
            fig.legend(
                handles=legend_patches,
                loc="lower center",
                ncol=N_RISK_CLASSES,
                bbox_to_anchor=(0.5, -0.02),
                frameon=False,
                fontsize=9,
            )

            plt.tight_layout()
            save_figure(fig, out_dir, fig_filename)
            plt.close(fig)

    print("\n  Decadal seasonal maps complete.")


def plot_monthly_climatology_maps(all_breaks):
    """
    Produce monthly climatology maps for each decade.
    Layout: rows = models, columns = SSP scenarios.
    One figure per decade × calendar month (12 months × 8 decades = 96 figures).
    """
    section("Plotting monthly climatology maps...")

    legend_patches = build_legend_patches()

    n_rows = len(MODELS)
    n_cols = len(SSP_SCENARIOS)

    for decade_label, decade_years in DECADES.items():
        for month_num, month_name in MONTH_NAMES.items():

            month_str    = f"{month_num:02d}"
            fig_filename = f"month_{month_str}_{month_name.lower()}"
            out_dir      = os.path.join(OUTPUT_DIR, "climatology",
                                        f"decade_{decade_label}")

            # Skip if already exists
            if os.path.exists(os.path.join(out_dir, f"{fig_filename}.png")):
                print(f"  SKIP: decade_{decade_label}/{fig_filename} (already exists)")
                continue

            subsection(f"{decade_label} — {month_name}")

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 4.5 * n_rows),
            )
            if n_rows == 1:
                axes = axes[np.newaxis, :]
            if n_cols == 1:
                axes = axes[:, np.newaxis]

            fig.suptitle(
                f"Mean Fire Risk — {month_name} | {decade_label}",
                fontsize=13, fontweight="bold", y=1.01,
            )

            for row_idx, (model_key, model_short) in enumerate(MODELS.items()):
                breaks = all_breaks.get(model_short)

                for col_idx, scenario in enumerate(SSP_SCENARIOS):
                    ax = axes[row_idx, col_idx]

                    if row_idx == 0:
                        ax.set_title(
                            SCENARIO_LABELS[scenario], fontsize=9, pad=6)
                    if col_idx == 0:
                        ax.set_ylabel(model_short, fontsize=9, labelpad=6)

                    if breaks is None:
                        ax.text(0.5, 0.5, "No Jenks breaks",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=8)
                        ax.axis("off")
                        continue

                    # Load decade predictions
                    df = load_decade_predictions(
                        model_short, scenario, decade_years)
                    if df is None:
                        ax.text(0.5, 0.5, "No data",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=8)
                        ax.axis("off")
                        continue

                    # Filter to this calendar month only
                    df_month = df[df["month"] == month_num]
                    if df_month.empty:
                        ax.text(0.5, 0.5, "No data (month filter)",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=8)
                        ax.axis("off")
                        continue

                    # Apply Jenks breaks
                    df_month = apply_breaks_to_df(df_month, breaks)

                    plot_risk_raster_ax(ax, df_month, breaks, title="")

            fig.legend(
                handles=legend_patches,
                loc="lower center",
                ncol=N_RISK_CLASSES,
                bbox_to_anchor=(0.5, -0.02),
                frameon=False,
                fontsize=9,
            )

            plt.tight_layout()
            save_figure(fig, out_dir, fig_filename)
            plt.close(fig)

    print("\n  Monthly climatology maps complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    sns.set_style("whitegrid")

    # --- Load Jenks breaks for all models upfront ---
    section("Loading Jenks breaks...")
    all_breaks = {}
    for model_key, model_short in MODELS.items():
        breaks = load_jenks_breaks(model_short)
        if breaks is None:
            print(f"  WARNING: Could not load breaks for {model_short} — "
                  f"this model will show 'No Jenks breaks' in all panels.")
        else:
            print(f"  {model_short}: {breaks}")
        all_breaks[model_short] = breaks

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Decadal seasonal maps ---
    plot_decadal_seasonal_maps(all_breaks)

    # --- Monthly climatology maps ---
    plot_monthly_climatology_maps(all_breaks)

    print()
    print("=" * 60)
    print("ALL FIGURES COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"    seasonal/dry/     — {len(DECADES)} figures")
    print(f"    seasonal/wet/     — {len(DECADES)} figures")
    print(f"    climatology/      — {len(DECADES) * len(MONTH_NAMES)} figures "
          f"({len(DECADES)} decades × 12 months)")


if __name__ == "__main__":
    main()