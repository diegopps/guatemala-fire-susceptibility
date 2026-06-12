"""
plot_risk_map.py
-----------------
Generic utility for plotting fire risk classification maps for Guatemala.

Renders each 1km grid cell as an exact pixel using pcolormesh() on a
reconstructed 2D raster from center coordinates. No interpolation is
applied — class boundaries are preserved exactly.

Administrative boundaries (departments + country outline) are overlaid using
Natural Earth's admin-1 layer. The shapefile is downloaded once and cached
under data/boundaries/; Guatemala's departments are filtered from it and the
country outline is the union of those departments.

Functions
---------
plot_risk_map(
    df             : pd.DataFrame,
    title          : str,
    output_path    : str,
    risk_col       : str = "risk_class",
)

plot_risk_map_grid(
    panels         : list of {"df": pd.DataFrame, "title": str},
    output_path    : str,
    nrows          : int,
    ncols          : int,
    suptitle       : str = None,
    panel_letters  : bool = False,
    risk_col       : str = "risk_class",
)

Aggregation of multiple month/year records to one risk class per cell is the
caller's responsibility (mean probability per cell, then apply Jenks breaks);
these functions only render pre-classified maps.

Parameters (plot_risk_map)
--------------------------
df           : pd.DataFrame with columns: lat, lon, <risk_col>
title        : str — figure title
output_path  : str — full output path without extension (PNG + PDF + SVG saved)
risk_col     : str — column name containing integer risk class (1–5)

Notes
-----
- Requires: geopandas, matplotlib, numpy, pandas
- Boundary shapefile auto-downloaded once to data/boundaries/ (needs internet
  on first run; on offline nodes, pre-place the .zip — see _ensure_admin1_shapefile)
- Risk classes: 1=very low, 2=low, 3=medium, 4=high, 5=very high
- Missing/unclassified cells rendered in white
- Dry season: Nov–Apr | Wet season: May–Oct
"""

import os
import warnings
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd

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

# Per-panel size (inches) for multi-panel grids
PANEL_W = 4.6
PANEL_H = 4.7

# Cache for the (country, departments) boundary GeoDataFrames — loaded once,
# reused across every map (avoids re-reading Natural Earth on each call).
_BOUNDARY_CACHE = None

# Natural Earth admin-1 (states/provinces) boundaries. geodatasets does not
# ship this layer, so we download it once and cache it under data/boundaries/.
# Guatemala's departments are filtered from it; the country outline is the union
# of those departments.
NE_ADMIN1_URL = ("https://naturalearth.s3.amazonaws.com/"
                 "10m_cultural/ne_10m_admin_1_states_provinces.zip")
NE_ADMIN1_NAME = "ne_10m_admin_1_states_provinces"
BOUNDARY_DIR  = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "boundaries")
)


# =============================================================================
# HELPERS
# =============================================================================

def _ensure_admin1_shapefile():
    """
    Return the path to the Natural Earth admin-1 shapefile, downloading and
    extracting it into data/boundaries/ on first use.

    Raises a clear error (with the URL and target path) if the download fails —
    e.g. on an offline compute node, in which case fetch it manually on a node
    with internet access and drop the .zip in place.
    """
    shp_dir  = os.path.join(BOUNDARY_DIR, NE_ADMIN1_NAME)
    shp_path = os.path.join(shp_dir, f"{NE_ADMIN1_NAME}.shp")
    if os.path.exists(shp_path):
        return shp_path

    os.makedirs(BOUNDARY_DIR, exist_ok=True)
    zip_path = os.path.join(BOUNDARY_DIR, f"{NE_ADMIN1_NAME}.zip")

    if not os.path.exists(zip_path):
        print(f"  Downloading Natural Earth admin-1 boundaries:\n    {NE_ADMIN1_URL}")
        try:
            urllib.request.urlretrieve(NE_ADMIN1_URL, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not download Natural Earth admin-1 boundaries ({e}).\n"
                f"If this node has no internet, download the file manually from:\n"
                f"    {NE_ADMIN1_URL}\n"
                f"and place it at:\n    {zip_path}"
            )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(shp_dir)

    if os.path.exists(shp_path):
        return shp_path
    # Fallback: archive may not use the expected filename — pick any .shp
    for f in os.listdir(shp_dir):
        if f.endswith(".shp"):
            return os.path.join(shp_dir, f)
    raise FileNotFoundError(f"No .shp found after extracting {zip_path}")


def _load_boundaries():
    """
    Load Guatemala department and country boundaries from Natural Earth's
    admin-1 layer. Departments are filtered by admin == "Guatemala"; the country
    outline is the union (dissolve) of those departments.

    Returns (country_gdf, departments_gdf). Cached after the first call.
    """
    global _BOUNDARY_CACHE
    if _BOUNDARY_CACHE is not None:
        return _BOUNDARY_CACHE

    admin1      = gpd.read_file(_ensure_admin1_shapefile())
    departments = admin1[admin1["admin"] == "Guatemala"].copy()
    if departments.empty:
        raise ValueError(
            "No Guatemala features found in the Natural Earth admin-1 layer "
            "(expected column 'admin' == 'Guatemala')."
        )

    country = departments.dissolve()  # union of departments -> country outline

    _BOUNDARY_CACHE = (country, departments)
    return _BOUNDARY_CACHE


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

    # Map lat/lon to raster indices (vectorized — no per-row Python loop)
    lon_idx = {v: i for i, v in enumerate(lons)}
    lat_idx = {v: i for i, v in enumerate(lats)}

    raster = np.zeros((len(lats), len(lons)), dtype=float)  # 0 = missing

    rows = df["lat"].map(lat_idx).to_numpy()
    cols = df["lon"].map(lon_idx).to_numpy()
    raster[rows, cols] = df[risk_col].to_numpy()

    return raster, lon_edges, lat_edges


def _make_colormap():
    """Build a discrete matplotlib colormap and norm for the 5 risk classes."""
    colors = [RISK_COLORS[i] for i in range(6)]  # 0–5
    cmap   = mcolors.ListedColormap(colors)
    norm   = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                                   ncolors=6)
    return cmap, norm


def make_delta_cmap(vmax=4):
    """
    Discrete diverging colormap + norm for integer risk-class CHANGES in
    [-vmax, +vmax]. Blue = risk decreased, white = no change, red = increased.
    Returns (cmap, norm).
    """
    n    = 2 * vmax + 1
    cmap = plt.get_cmap("RdBu_r", n)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(-vmax - 0.5, vmax + 1.5, 1.0), ncolors=n)
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

def _render_risk_map(ax, df, title, risk_col="risk_class",
                     country=None, departments=None, cmap=None, norm=None,
                     show_xlabel=True, show_ylabel=True):
    """
    Render a single risk map onto a provided axes (no legend, no saving).

    Shared by plot_risk_map() and plot_risk_map_grid(). Pass pre-loaded
    boundaries / colormap to avoid recomputing them for every panel.
    """
    if country is None or departments is None:
        country, departments = _load_boundaries()
    if cmap is None or norm is None:
        cmap, norm = _make_colormap()

    raster, lon_edges, lat_edges = _build_raster(df, risk_col)

    ax.pcolormesh(lon_edges, lat_edges, raster, cmap=cmap, norm=norm,
                  shading="flat", rasterized=True)
    _add_boundaries(ax, country, departments)

    ax.set_xlim(GT_LON_MIN, GT_LON_MAX)
    ax.set_ylim(GT_LAT_MIN, GT_LAT_MAX)
    if show_xlabel:
        ax.set_xlabel("Longitude", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.tick_params(labelsize=8)


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

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _render_risk_map(ax, df, title=title, risk_col=risk_col)
    _add_legend(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)


# =============================================================================
# MULTI-PANEL GRID (model comparison + monthly climatology)
# =============================================================================

def plot_risk_map_grid(panels, output_path, nrows, ncols,
                       suptitle=None, panel_letters=False, risk_col="risk_class"):
    """
    Render several pre-classified risk maps as a single multi-panel figure
    with one shared legend.

    Parameters
    ----------
    panels        : list of dict — each {"df": DataFrame, "title": str}, where
                    df has columns lat, lon, <risk_col> already aggregated to one
                    integer risk class (1–5) per cell.
    output_path   : str — output path without extension (PNG + PDF + SVG saved)
    nrows, ncols  : int — grid layout. nrows * ncols must be >= len(panels).
    suptitle      : str or None — overall figure title.
    panel_letters : bool — if True, prefix each panel title with "A) ", "B) " …
                    for cross-referencing in the text.
    risk_col      : str — column containing the integer risk class (1–5).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    country, departments = _load_boundaries()   # cached after first call
    cmap, norm = _make_colormap()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(PANEL_W * ncols, PANEL_H * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i >= len(panels):
            ax.axis("off")
            continue

        panel = panels[i]
        title = panel["title"]
        if panel_letters:
            title = f"{letters[i]}) {title}"

        row, col = divmod(i, ncols)
        print(f"  Rendering panel {i + 1}/{len(panels)}: {title}")
        _render_risk_map(
            ax, panel["df"], title=title, risk_col=risk_col,
            country=country, departments=departments, cmap=cmap, norm=norm,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )

    # Single shared legend for the whole figure
    patches = [
        mpatches.Patch(facecolor=RISK_COLORS[i], edgecolor="#aaaaaa",
                       linewidth=0.5, label=RISK_LABELS[i])
        for i in range(1, 6)
    ]
    fig.legend(
        handles=patches,
        title="Fire Risk",
        loc="lower center",
        ncol=5,
        fontsize=10,
        title_fontsize=10,
        frameon=True,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.02),
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.0)

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    _save_figure(fig, output_path)