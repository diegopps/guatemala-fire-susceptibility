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

INPUT_DIR  = "/home/diegopab/projects/ForestFire_GT/paperwriting/data/data_fullfuture"   # Root directory containing one folder per scenario
OUTPUT_DIR = "./figures/trends"   # Directory to save output figures

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
    "tas":     {"label": "Mean Temperature (°C)",              "convert": lambda x: x - 273.15},
    "tasmax":  {"label": "Max Temperature (°C)",               "convert": lambda x: x - 273.15},
    "tasmin":  {"label": "Min Temperature (°C)",               "convert": lambda x: x - 273.15},
    "pr":      {"label": "Precipitation (mm/month)",           "convert": lambda x: x * 2_592_000},
    "rsds":    {"label": "Solar Radiation (W/m²)",             "convert": None},
    "PET":     {"label": "Potential Evapotranspiration (mm/month)", "convert": None},
    "climbal": {"label": "Climatic Water Balance (mm/month)",  "convert": None},
}
 
# ---------------------------------------------------------------------------
# Seaborn / matplotlib theme
# ---------------------------------------------------------------------------
 
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titlesize":  12,
    "axes.labelsize":  10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  9,
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
        # Extract year from filename — expects something like fullvarset_ssp126_2055.csv
        basename = os.path.splitext(os.path.basename(f))[0]
        year = int(basename.split("_")[-1])
 
        df = pd.read_csv(f, usecols=["lat", "lon", "month", var])
        # Spatial + monthly average → single yearly mean
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
 
 
# ---------------------------------------------------------------------------
# Plot function
# ---------------------------------------------------------------------------
 
def plot_variable(var: str, cfg: dict, output_path: str):
    data = load_all_scenarios(var, cfg["convert"])
 
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
 
    # Draw a vertical dashed line at the historical/future boundary
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
    ax.set_ylabel(cfg["label"])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(axis="y", linewidth=0.5, alpha=0.6)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9)
 
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    for var, cfg in VARIABLES.items():
        print(f"Plotting: {var}...")
        out = os.path.join(OUTPUT_DIR, f"trend_{var}.png")
        plot_variable(var, cfg, out)
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()
 
