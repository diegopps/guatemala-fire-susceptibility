"""
plot_temporal_trends.py
-----------------------
Generates one line plot per temporal predictor variable,
showing the spatially averaged yearly mean across the full time period.
The historical period is plotted as a shared grey baseline; the three future
scenarios are plotted as separate coloured lines.

Output: one PNG per variable saved to OUTPUT_DIR.

Directory structure expected:
    INPUT_DIR/
        historical/   → fullvarset_historical_<year>.csv  (e.g. 1960–2014)
        ssp126/       → fullvarset_ssp126_<year>.csv      (e.g. 2015–2100)
        ssp370/       → fullvarset_ssp370_<year>.csv
        ssp585/       → fullvarset_ssp585_<year>.csv

Dependencies:
    pip install pandas matplotlib seaborn
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

INPUT_DIR   = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/data/indata/data_fullfuture_processed/"
OUTPUT_DIR  = "../../figures/01_data_exploration/trends"
CACHE_FILE  = os.path.join(OUTPUT_DIR, "trends_cache.csv")

# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

SCENARIOS = {
    "historical": {
        "subdir":  "historical",
        "label":   "Historical",
        "color":   "#555555",
        "ls":      "-",
        "zorder":  5,
    },
    "ssp126": {
        "subdir":  "ssp126",
        "label":   "SSP1-2.6 (Sustainability)",
        "color":   "#2E86AB",
        "ls":      "-",
        "zorder":  4,
    },
    "ssp370": {
        "subdir":  "ssp370",
        "label":   "SSP3-7.0 (Regional Rivalry)",
        "color":   "#E8A838",
        "ls":      "-",
        "zorder":  3,
    },
    "ssp585": {
        "subdir":  "ssp585",
        "label":   "SSP5-8.5 (Fossil-Fueled Development)",
        "color":   "#E84855",
        "ls":      "-",
        "zorder":  2,
    },
}

# ---------------------------------------------------------------------------
# Temporal variables and unit conversions
# conversion: None = no conversion; callable = applied to the column
# ---------------------------------------------------------------------------

VARIABLES = {
    "tas":     {"label": "Mean Temperature (°C)",                   "convert": lambda x: x - 273.15},
    "tasmax":  {"label": "Max Temperature (°C)",                    "convert": lambda x: x - 273.15},
    "tasmin":  {"label": "Min Temperature (°C)",                    "convert": lambda x: x - 273.15},
    "pr":      {"label": "Precipitation (mm/month)",                "convert": lambda x: x * 2_592_000},
    "rsds":    {"label": "Solar Radiation (W/m²)",                  "convert": None},
    "PET":     {"label": "Potential Evapotranspiration (mm/month)", "convert": None},
    "climbal": {"label": "Climatic Water Balance (mm/month)",       "convert": None},
}

# ---------------------------------------------------------------------------
# Seaborn / matplotlib theme — all font sizes doubled from original
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titlesize":  13,
    "axes.labelsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scenario(scenario_dir: str, var: str, convert) -> pd.Series:
    """
    Load all CSVs in scenario_dir, compute the spatial average of var
    for each year, apply unit conversion, and return a Series indexed by year.
    """
    pattern = os.path.join(scenario_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {scenario_dir}")

    yearly = {}
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        year = int(basename.split("_")[-1])
        df = pd.read_csv(f, usecols=[var])
        annual_mean = df[var].mean()
        if convert is not None:
            annual_mean = convert(annual_mean)
        yearly[year] = annual_mean

    return pd.Series(yearly).sort_index()


def load_all_scenarios(var: str, convert) -> dict:
    """Return {scenario_key: pd.Series(year → value)} for all scenarios."""
    data = {}
    for key, cfg in SCENARIOS.items():
        scenario_dir = os.path.join(INPUT_DIR, cfg["subdir"])
        print(f"    Loading {key} / {var}...")
        data[key] = load_scenario(scenario_dir, var, convert)
    return data


def build_cache() -> pd.DataFrame:
    """Load raw CSVs for all variables/scenarios, return tidy DataFrame."""
    rows = []
    for var, vcfg in VARIABLES.items():
        data = load_all_scenarios(var, vcfg["convert"])
        for scenario, series in data.items():
            for year, value in series.items():
                rows.append({"scenario": scenario, "variable": var,
                             "year": year, "value": value})
    return pd.DataFrame(rows)


def cache_to_nested(df: pd.DataFrame) -> dict:
    """Convert tidy cache DataFrame to {var: {scenario: pd.Series}} structure."""
    result = {}
    for var in df["variable"].unique():
        result[var] = {}
        sub = df[df["variable"] == var]
        for scenario in sub["scenario"].unique():
            s = sub[sub["scenario"] == scenario].set_index("year")["value"]
            result[var][scenario] = s.sort_index()
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wrap_label(text: str, max_chars: int = 20) -> str:
    """Break text onto two lines at the last space before max_chars."""
    if len(text) <= max_chars:
        return text
    pivot = text.rfind(" ", 0, max_chars)
    if pivot == -1:
        pivot = text.find(" ", max_chars)
    if pivot == -1:
        return text
    return text[:pivot] + "\n" + text[pivot + 1:]


# ---------------------------------------------------------------------------
# Plot function — single subplot
# ---------------------------------------------------------------------------

def plot_ax(ax, var: str, cfg: dict, data: dict):
    """Draw one variable's trend lines onto an existing Axes object."""

    # Vertical dashed line at historical/future boundary
    hist_end = data["historical"].index.max()
    ax.axvline(hist_end + 0.5, color="#aaaaaa", linewidth=0.8,
               linestyle="--", zorder=1)

    for key, series in data.items():
        scen = SCENARIOS[key]
        ax.plot(
            series.index, series.values,
            label=scen["label"],
            color=scen["color"],
            linestyle=scen["ls"],
            linewidth=1.6,
            zorder=scen["zorder"],
        )

    ax.set_title(cfg["label"], pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel(wrap_label(cfg["label"]))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(axis="y", linewidth=0.5, alpha=0.6)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)


# ---------------------------------------------------------------------------
# Main — grid figure
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- cache logic ----------
    if os.path.exists(CACHE_FILE):
        print(f"Cache found — loading from {CACHE_FILE}")
        cache_df = pd.read_csv(CACHE_FILE)
    else:
        print("No cache found — computing averages from raw CSVs...")
        cache_df = build_cache()
        cache_df.to_csv(CACHE_FILE, index=False)
        print(f"Cache saved to {CACHE_FILE}")

    all_data = cache_to_nested(cache_df)   # {var: {scenario: Series}}
    # ---------------------------------

    vars_list = list(VARIABLES.items())   # [(var, cfg), ...]
    n_vars    = len(vars_list)            # 7
    ncols     = 2
    nrows     = -(-n_vars // ncols)       # ceiling division → 4

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(18, nrows * 4.5),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    # Plot all 7 variables
    for i, (var, cfg) in enumerate(vars_list):
        print(f"Plotting: {var}...")
        plot_ax(axes_flat[i], var, cfg, all_data[var])

    # Place shared legend in the empty 8th cell (after the plotting loop
    # so that axes_flat[0] already has its lines drawn)
    legend_ax = axes_flat[n_vars]
    legend_ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_ax.legend(
        handles, labels,
        loc="center",
        framealpha=0.9,
        fontsize=11,
        title="Scenario",
        title_fontsize=13,
        handlelength=2.0,
        handleheight=1.2,
    )

    out = os.path.join(OUTPUT_DIR, "trends_grid.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()