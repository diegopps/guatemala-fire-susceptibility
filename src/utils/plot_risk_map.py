"""
plot_risk_map.py
-----------------
Generic utility for plotting fire risk classification maps for Guatemala.

Renders each 1km grid cell as an exact pixel using pcolormesh() on a
reconstructed 2D raster from center coordinates. No interpolation is
applied — class boundaries are preserved exactly.

Administrative boundaries (country outline + departments) are overlaid
using Natural Earth data via geopandas + geodatasets. No external
shapefiles required.

Functions
---------
plot_risk_map(
    df             : pd.DataFrame,
    title          : str,
    output_path    : str,
    risk_col       : str = "risk_class",
)

plot_seasonal_maps(
    df             : pd.DataFrame,
    model_label    : str,
    output_dir     : str,
    risk_col       : str = "risk_class",
)

plot_monthly_climatology(
    df             : pd.DataFrame,
    model_label    : str,
    output_dir     : str,
    risk_col       : str = "risk_class",
)

Parameters (plot_risk_map)
--------------------------
df           : pd.DataFrame with columns: lat, lon, <risk_col>
title        : str — figure title
output_path  : str — full output path without extension (PNG + PDF + SVG saved)
risk_col     : str — column name containing integer risk class (1–5)

Notes
-----
- Requires: geopandas, geodatasets, matplotlib, numpy, pandas
- Install geodatasets if needed: pip install geodatasets
- Risk classes: 1=very low, 2=low, 3=medium, 4=high, 5=very high
- Missing/unclassified cells rendered in white
- Dry season: Nov–Apr | Wet season: May–Oct
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd

try:
    import geodatasets
except ImportError:
    raise ImportError(
        "The 'geodatasets' package is required for boundary data. "
        "Install it with: pip install geodatasets"
    )

# =============================================================================
# STYLE & CONSTANTS
# =============================================================================

# Risk class colors — consistent with project palette
RISK_COLORS = {
    0: "#ffffff",  # missing / no data — white
    1: "#ffffb2",  # very low
    2: "#fecc5c",  # low
    3: "#fd8d3c",  # medium
    4: "#f03b20",  # high
    5: "#bd0026",  # very high
}

RISK_LABELS = {
    1: "Very low",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Very high",
}

DRY_MONTHS = [11, 12, 1, 2, 3, 4]
WET_MONTHS = [5, 6, 7, 8, 9, 10]

MONTH_NAMES = {
    1: "January",   2: "February",  3: "March",
    4: "April",     5: "May",       6: "June",
    7: "July",      8: "August",    9: "September",
    10: "October",  11: "November", 12: "December",
}

FIGURE_SIZE    = (10, 8)
DPI            = 300
OUTPUT_FORMATS = ["png", "pdf", "svg"]

# Department boundary style
DEPT_LINEWIDTH  = 0.4
DEPT_EDGECOLOR  = "#555555"
DEPT_ALPHA      = 0.4
COUNTRY_LINEWIDTH = 1.2
COUNTRY_EDGECOLOR = "#000000"

# Guatemala bounding box (degrees) — used to clip the raster extent
GT_LON_MIN, GT_LON_MAX = -92.3, -88.2
GT_LAT_MIN, GT_LAT_MAX =  13.7,  17.9


# =============================================================================
# HELPERS
# =============================================================================

def _load_boundaries():
    """
    Load Guatemala country and department boundaries from Natural Earth
    via geodatasets. Returns (country_gdf, departments_gdf).
    """
    try:
        ne_path = geodatasets.get_path("naturalearth admin_1_states_provinces")
        admin1  = gpd.read_file(ne_path)
    except Exception:
        # Fallback path name variation
        ne_path = geodatasets.get_path("naturalearth.admin_1_states_provinces")
        admin1  = gpd.read_file(ne_path)

    departments = admin1[admin1["admin"] == "Guatemala"].copy()

    try:
        ne_countries = geodatasets.get_path("naturalearth land")
        countries    = gpd.read_file(ne_countries)
    except Exception:
        ne_countries = geodatasets.get_path("naturalearth.land")
        countries    = gpd.read_file(ne_countries)

    # Clip country outline to Guatemala bounding box
    from shapely.geometry import box
    bbox    = box(GT_LON_MIN - 0.1, GT_LAT_MIN - 0.1,
                  GT_LON_MAX + 0.1, GT_LAT_MAX + 0.1)
    country = countries.clip(bbox)

    return country, departments


def _build_raster(df, risk_col):
    """
    Reconstruct a 2D raster array from a DataFrame of center coordinates
    and risk class values.

    Assumes a regular grid. Returns (raster_2d, lon_edges, lat_edges)
    suitable for pcolormesh().
    """
    lons = np.sort(df["lon"].unique())
    lats = np.sort(df["lat"].unique())[::-1]  # descending: top of map = north

    # Infer cell size from grid spacing
    dlon = np.median(np.diff(np.sort(df["lon"].unique())))
    dlat = np.median(np.diff(np.sort(df["lat"].unique())))

    # Build edges for pcolormesh (n+1 edges for n cells)
    lon_edges = np.append(lons - dlon / 2, lons[-1] + dlon / 2)
    lat_edges = np.append(lats + dlat / 2, lats[-1] - dlat / 2)

    # Map lat/lon to raster indices
    lon_idx = {v: i for i, v in enumerate(lons)}
    lat_idx = {v: i for i, v in enumerate(lats)}

    raster = np.zeros((len(lats), len(lons)), dtype=float)  # 0 = missing

    for _, row in df.iterrows():
        r = lat_idx.get(row["lat"])
        c = lon_idx.get(row["lon"])
        if r is not None and c is not None:
            raster[r, c] = row[risk_col]

    return raster, lon_edges, lat_edges


def _make_colormap():
    """Build a discrete matplotlib colormap and norm for the 5 risk classes."""
    colors = [RISK_COLORS[i] for i in range(6)]  # 0–5
    cmap   = mcolors.ListedColormap(colors)
    norm   = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                                   ncolors=6)
    return cmap, norm


def _add_boundaries(ax, country, departments):
    """Overlay country outline and department boundaries on an axes."""
    departments.boundary.plot(
        ax=ax,
        linewidth=DEPT_LINEWIDTH,
        edgecolor=DEPT_EDGECOLOR,
        alpha=DEPT_ALPHA,
    )
    country.boundary.plot(
        ax=ax,
        linewidth=COUNTRY_LINEWIDTH,
        edgecolor=COUNTRY_EDGECOLOR,
    )


def _add_legend(ax):
    """Add a risk class legend to the axes."""
    patches = [
        mpatches.Patch(facecolor=RISK_COLORS[i], edgecolor="#aaaaaa",
                       linewidth=0.5, label=RISK_LABELS[i])
        for i in range(1, 6)
    ]
    ax.legend(
        handles=patches,
        title="Fire Risk",
        loc="lower left",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
    )


def _save_figure(fig, output_path):
    """Save figure to PNG, PDF, and SVG."""
    for fmt in OUTPUT_FORMATS:
        path = f"{output_path}.{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_risk_map(df, title, output_path, risk_col="risk_class"):
    """
    Plot a single fire risk classification map for Guatemala.

    Parameters
    ----------
    df          : pd.DataFrame with columns: lat, lon, <risk_col>
    title       : str — figure title
    output_path : str — output path without extension
    risk_col    : str — column containing integer risk class (1–5)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Building raster...")
    raster, lon_edges, lat_edges = _build_raster(df, risk_col)

    print(f"  Loading boundaries...")
    country, departments = _load_boundaries()

    cmap, norm = _make_colormap()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Plot raster
    ax.pcolormesh(lon_edges, lat_edges, raster, cmap=cmap, norm=norm,
                  shading="flat", rasterized=True)

    # Overlay boundaries
    _add_boundaries(ax, country, departments)

    # Formatting
    ax.set_xlim(GT_LON_MIN, GT_LON_MAX)
    ax.set_ylim(GT_LAT_MIN, GT_LAT_MAX)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)
    ax.tick_params(labelsize=9)

    _add_legend(ax)

    fig.tight_layout()
    _save_figure(fig, output_path)


# =============================================================================
# SEASONAL MAPS (results section)
# =============================================================================

def plot_seasonal_maps(df, model_label, output_dir, risk_col="risk_class"):
    """
    Produce two maps for the results section:
      1. Dry season mean risk (Nov–Apr)
      2. Wet season mean risk (May–Oct)

    Risk class per cell is determined by rounding the mean risk class
    across all months in the season across all years.

    Parameters
    ----------
    df          : pd.DataFrame with columns: lat, lon, month, <risk_col>
    model_label : str — e.g. "RF_final"
    output_dir  : str — directory to save figures
    risk_col    : str — column containing integer risk class (1–5)
    """
    os.makedirs(output_dir, exist_ok=True)

    seasons = {
        "dry": {"months": DRY_MONTHS, "label": "Dry Season (Nov–Apr)"},
        "wet": {"months": WET_MONTHS, "label": "Wet Season (May–Oct)"},
    }

    for season_key, season_info in seasons.items():
        print(f"\n  Plotting {season_info['label']} map for {model_label}...")

        subset = df[df["month"].isin(season_info["months"])].copy()

        if subset.empty:
            print(f"  WARNING: No data for {season_info['label']} — skipping.")
            continue

        # Mean risk class per cell across all years in the season, rounded
        agg = (subset.groupby(["lat", "lon"])[risk_col]
               .mean()
               .round()
               .clip(1, 5)
               .astype(int)
               .reset_index())
        agg.columns = ["lat", "lon", risk_col]

        title       = f"{model_label} — Fire Risk: {season_info['label']}"
        output_path = os.path.join(output_dir,
                                   f"{model_label}_risk_map_{season_key}")

        plot_risk_map(agg, title=title, output_path=output_path,
                      risk_col=risk_col)


# =============================================================================
# MONTHLY CLIMATOLOGY MAPS (appendix)
# =============================================================================

def plot_monthly_climatology(df, model_label, output_dir, risk_col="risk_class"):
    """
    Produce 12 maps for the appendix — one per calendar month, averaged
    across all years (monthly climatology).

    Risk class per cell is determined by rounding the mean risk class
    for that calendar month across all years.

    Parameters
    ----------
    df          : pd.DataFrame with columns: lat, lon, month, <risk_col>
    model_label : str — e.g. "RF_final"
    output_dir  : str — directory to save figures
    risk_col    : str — column containing integer risk class (1–5)
    """
    os.makedirs(output_dir, exist_ok=True)

    for month_num, month_name in MONTH_NAMES.items():
        print(f"\n  Plotting monthly climatology: {month_name} ({model_label})...")

        subset = df[df["month"] == month_num].copy()

        if subset.empty:
            print(f"  WARNING: No data for {month_name} — skipping.")
            continue

        # Mean risk class per cell across all years for this month, rounded
        agg = (subset.groupby(["lat", "lon"])[risk_col]
               .mean()
               .round()
               .clip(1, 5)
               .astype(int)
               .reset_index())
        agg.columns = ["lat", "lon", risk_col]

        title       = f"{model_label} — Fire Risk: {month_name} (climatology)"
        output_path = os.path.join(output_dir,
                                   f"{model_label}_risk_map_month{month_num:02d}_{month_name.lower()}")

        plot_risk_map(agg, title=title, output_path=output_path,
                      risk_col=risk_col)