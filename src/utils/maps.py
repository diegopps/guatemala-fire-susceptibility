"""
maps.py
-------
Map helpers for the paper figures. Maps are drawn on the analysis grid (CHELSA 30'',
WGS 84 lat/lon), with an aspect ratio that keeps distances true at Guatemala's latitude.
Boundaries: Natural Earth 10m admin-1 (paths.inputs.boundaries, files.boundaries).

cells_to_raster(grid, cell_id, values)   2-D array on the grid (NaN outside the cells)
draw_raster(ax, grid, raster, **kw)      the grid cells, rasterized in the PDF
setup_map(ax, left=True, bottom=True)    extent, aspect, coordinate ticks (15° N, 90° W)
draw_boundaries(ax, departments=True)    Guatemala, its departments and neighbouring countries
add_scale_bar(ax, length_km, x, y)       scale bar at axes position (x, y)
draw_locator(ax)                         Central America, with Guatemala highlighted
"""

from functools import lru_cache

import geopandas as gpd
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Rectangle

from load_config import cfg

COUNTRY = "Guatemala"
PAD_DEG = 0.1                                   # map margin around the country
LOCATOR_EXTENT = (-98.0, -76.5, 6.5, 22.5)      # lon_min, lon_max, lat_min, lat_max

COUNTRY_LINE = {"color": "black", "linewidth": 0.6}
DEPARTMENT_LINE = {"color": "#7f7f7f", "linewidth": 0.3}
NEIGHBOUR_LINE = {"color": "#9e9e9e", "linewidth": 0.4}


# =============================================================================
# GRID
# =============================================================================

def cells_to_raster(grid, cell_id, values):
    """Values of the given cells (row-major cell ids) as a 2-D array; NaN elsewhere."""
    raster = np.full(grid.shape, np.nan)
    cell_id = np.asarray(cell_id)
    raster[cell_id // grid.lon.size, cell_id % grid.lon.size] = values
    return raster


def draw_raster(ax, grid, raster, **kwargs):
    """Grid cells as a mesh, rasterized at figures.dpi when saved as PDF."""
    return ax.pcolormesh(grid.lon, grid.lat, raster, shading="nearest", rasterized=True,
                         linewidth=0, **kwargs)


# =============================================================================
# BOUNDARIES
# =============================================================================

@lru_cache(maxsize=1)
def boundaries():
    """Admin-1 boundaries of the region: (departments, country outline, all countries)."""
    path = cfg.paths.inputs.boundaries / cfg.files.boundaries
    lon_min, lon_max, lat_min, lat_max = LOCATOR_EXTENT
    admin1 = gpd.read_file(path, bbox=(lon_min, lat_min, lon_max, lat_max))[["admin", "geometry"]]
    departments = admin1[admin1.admin == COUNTRY]
    countries = admin1.dissolve(by="admin").reset_index()
    country = countries[countries.admin == COUNTRY]
    return departments, country, countries


def map_extent():
    """lon_min, lon_max, lat_min, lat_max of the national maps."""
    lon_min, lat_min, lon_max, lat_max = boundaries()[1].total_bounds
    return lon_min - PAD_DEG, lon_max + PAD_DEG, lat_min - PAD_DEG, lat_max + PAD_DEG


def draw_boundaries(ax, departments=True):
    """Neighbouring countries, Guatemala's departments (optional) and its outline."""
    deps, country, countries = boundaries()
    countries[countries.admin != COUNTRY].boundary.plot(ax=ax, **NEIGHBOUR_LINE)
    if departments:
        deps.boundary.plot(ax=ax, **DEPARTMENT_LINE)
    country.boundary.plot(ax=ax, **COUNTRY_LINE)


# =============================================================================
# MAP FURNITURE
# =============================================================================

def _lon_label(x, _pos=None):
    return f"{abs(x):g}° {'W' if x < 0 else 'E'}" if x else "0°"


def _lat_label(y, _pos=None):
    return f"{abs(y):g}° {'S' if y < 0 else 'N'}" if y else "0°"


def setup_map(ax, left=True, bottom=True, extent=None):
    """National map extent, true-distance aspect and degree ticks (shown on the left/bottom)."""
    lon_min, lon_max, lat_min, lat_max = extent or map_extent()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1 / np.cos(np.deg2rad((lat_min + lat_max) / 2)))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_lon_label))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_lat_label))
    ax.tick_params(labelleft=left, labelbottom=bottom, direction="out")
    ax.set_xlabel("")
    ax.set_ylabel("")


def add_scale_bar(ax, length_km=100, x=0.62, y=0.05, height_frac=0.012):
    """Two-segment scale bar starting at axes position (x, y), true at that latitude."""
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    lon0 = lon_min + x * (lon_max - lon_min)
    lat0 = lat_min + y * (lat_max - lat_min)
    deg = length_km / (111.32 * np.cos(np.deg2rad(lat0)))
    h = height_frac * (lat_max - lat_min)
    for k, face in enumerate(["black", "white"]):
        ax.add_patch(Rectangle((lon0 + k * deg / 2, lat0), deg / 2, h, facecolor=face,
                               edgecolor="black", linewidth=0.4, zorder=5))
    for frac, text in [(0, "0"), (0.5, f"{length_km / 2:g}"), (1, f"{length_km:g} km")]:
        ax.text(lon0 + frac * deg, lat0 + 1.6 * h, text, ha="center", va="bottom",
                fontsize=cfg.figures.font_size - 1, zorder=5)


def draw_locator(ax):
    """Locator map of Central America with Guatemala highlighted."""
    _, country, countries = boundaries()
    countries.plot(ax=ax, facecolor="#e6e6e6", edgecolor="#9e9e9e", linewidth=0.2)
    country.plot(ax=ax, facecolor="#4d4d4d", edgecolor="black", linewidth=0.3)
    setup_map(ax, left=False, bottom=False, extent=LOCATOR_EXTENT)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
