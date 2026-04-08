"""
plot_maps.py
------------
Generates one map per predictor variable, showing the
average value across the full study period (2017–2024). Country borders
are downloaded automatically via the cartopy natural earth dataset.

Output: one PNG per variable saved to OUTPUT_DIR.

Dependencies:
    pip install pandas matplotlib seaborn cartopy numpy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

INPUT_FILE = "/home/diegopab/projects/ForestFire_GT/paperwriting/data/data_trainingset_2017-2024/allvars_2017-2024_v2.csv"
OUTPUT_DIR = "./figures/maps/"  

# ---------------------------------------------------------------------------
# Variable configuration
# colourmap choices:
#   - Sequential variables (low→high): viridis, plasma, YlOrRd, etc.
#   - Diverging variables (meaningful zero): RdBu, BrBG, RdYlGn
#   - Cyclic variable (aspect 0–360°): twilight
# ---------------------------------------------------------------------------

VARIABLES = {
    "tas":       {"label": "Mean Temperature (°C)",           "cmap": "plasma",    "diverging": False},
    "tasmax":    {"label": "Max Temperature (°C)",            "cmap": "YlOrRd",    "diverging": False},
    "tasmin":    {"label": "Min Temperature (°C)",            "cmap": "YlGnBu",    "diverging": False},
    "pr":        {"label": "Precipitation (mm/month)",        "cmap": "YlGnBu",    "diverging": False},
    "rsds":      {"label": "Solar Radiation (W/m²)",          "cmap": "plasma",    "diverging": False},
    "PET":       {"label": "Potential Evapotranspiration (mm/month)", "cmap": "YlOrRd", "diverging": False},
    "climbal":   {"label": "Climatic Water Balance (mm/month)", "cmap": "RdBu",    "diverging": True},
    "elev":      {"label": "Elevation (m)",                   "cmap": "terrain",   "diverging": False},
    "slope":     {"label": "Slope (°)",                       "cmap": "viridis",   "diverging": False},
    "aspect_sin":    {"label": "Aspect Sine",                      "cmap": "coolwarm",  "diverging": False},
    "aspect_cos":    {"label": "Aspect Cosine",                      "cmap": "coolwarm",  "diverging": False},
    "distRoads": {"label": "Distance to Roads (m)",           "cmap": "viridis",   "diverging": False},
    "distUrban": {"label": "Distance to Urban Areas (m)",     "cmap": "viridis",   "diverging": False},
    "distWater": {"label": "Distance to Water Bodies (m)",    "cmap": "viridis",   "diverging": False},
}

# Guatemala bounding box (with a small padding)
LON_MIN, LON_MAX = -92.5, -88.0
LAT_MIN, LAT_MAX =  13.5,  18.5

# ---------------------------------------------------------------------------
# Seaborn / matplotlib theme
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":        150,
})

# ---------------------------------------------------------------------------
# Helper: pivot average values to a 2-D grid
# ---------------------------------------------------------------------------

def compute_grid_average(df: pd.DataFrame, var: str):
    """Return lons, lats, and a 2-D array of per-cell mean values."""
    avg = df.groupby(["lat", "lon"])[var].mean().reset_index()
    lons = np.sort(avg["lon"].unique())
    lats = np.sort(avg["lat"].unique())
    grid = np.full((len(lats), len(lons)), np.nan)
    lon_idx = {v: i for i, v in enumerate(lons)}
    lat_idx = {v: i for i, v in enumerate(lats)}
    for _, row in avg.iterrows():
        grid[lat_idx[row["lat"]], lon_idx[row["lon"]]] = row[var]
    return lons, lats, grid

# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_variable_map(df: pd.DataFrame, var: str, cfg: dict, output_path: str):
    lons, lats, grid = compute_grid_average(df, var)

    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    # Colourmap + normalisation
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    if cfg["diverging"]:
        # Centre diverging colourmap at zero (or midpoint if zero is outside range)
        vcenter = 0.0 if vmin < 0 < vmax else (vmin + vmax) / 2
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot raster
    im = ax.pcolormesh(
        lons, lats, grid,
        cmap=cfg["cmap"],
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
    )

    # Country borders — Guatemala highlighted, neighbours in grey
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_0_countries", "10m",
            facecolor="none",
            edgecolor="#333333",
            linewidth=0.8,
        )
    )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4,
        color="grey", alpha=0.5, linestyle="--",
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = ticker.MultipleLocator(1)
    gl.ylocator = ticker.MultipleLocator(1)
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cbar.set_label(cfg["label"], fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"{cfg['label']}\nAverage 2017–2024", fontsize=12, pad=10)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    missing = [v for v in VARIABLES if v not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")
    print(f"  {len(df):,} rows loaded.")

    # Unit conversions
    for col in ["tas", "tasmax", "tasmin"]:
        df[col] = df[col] - 273.15          # K → °C
    df["pr"] = df["pr"] * 2_592_000         # kg/m²/s → mm/month (30-day month)

    for var, cfg in VARIABLES.items():
        print(f"Plotting map: {var}...")
        out = os.path.join(OUTPUT_DIR, f"map_{var}.png")
        plot_variable_map(df, var, cfg, out)

    print("Done.")


if __name__ == "__main__":
    main()
