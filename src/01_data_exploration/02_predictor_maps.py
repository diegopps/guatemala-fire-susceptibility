"""
02_predictor_maps.py
--------------------
Maps of the predictors as the model sees them: for each grid cell, the mean over all
rows of the training set (all GCMs, all months of the training years). Static
predictors are constant per cell, so their map is the layer itself.

Input:  paths.inputs.training_set
Output: results/mainfigs/01_predictor_maps.pdf            mean temperature, precipitation,
                                                          elevation, distance to urban areas
        results/appendix/01_predictor_maps_climate.pdf    the 7 climate predictors
        results/appendix/01_predictor_maps_static.pdf     the 7 static predictors
        results/<data_exploration>/predictor_means.parquet — the mapped values
        (cell_id, lat, lon and one column per predictor, in display units). Reused unless
        --recompute is given.

Units: temperatures °C, precipitation, PET and climatic water balance mm month-1 (real
month lengths), solar radiation W m-2, elevation and distances m, slope degrees.

Usage:
    python 02_predictor_maps.py [--recompute]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import reference_grid
from climate_derived import month_lengths
from figures import (CLIMATE_VARIABLES, STATIC_VARIABLES, add_panel_label, cm, label,
                     save_figure, setup_style, to_display_units)
from maps import cells_to_raster, draw_boundaries, draw_raster, map_extent, setup_map

MAIN_VARIABLES = ["tas", "pr", "elev", "distUrban"]

# Colour map per predictor (perceptually uniform or ColorBrewer, colour-blind safe);
# "diverging" maps are centred on zero.
STYLE = {
    "tas":        ("YlOrRd", False),
    "tasmax":     ("YlOrRd", False),
    "tasmin":     ("YlOrRd", False),
    "pr":         ("YlGnBu", False),
    "PET":        ("YlOrBr", False),
    "climbal":    ("BrBG", True),
    "rsds":       ("magma", False),
    "elev":       ("cividis", False),
    "slope":      ("viridis", False),
    "aspect_sin": ("PuOr", True),
    "aspect_cos": ("PuOr", True),
    "distRoads":  ("viridis", False),
    "distUrban":  ("viridis", False),
    "distWater":  ("viridis", False),
}

# Layout (cm)
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP = 1.1, 0.3, 0.45
GAP_X = 0.5
BELOW_MAP = 1.75          # longitude labels, colour bar and its label
CBAR_OFFSET, CBAR_HEIGHT = 0.55, 0.2


# =============================================================================
# DATA
# =============================================================================

def compute_cell_means(grid):
    """Mean of every predictor per cell over all rows of the training set (display units)."""
    features = cfg.data.features
    n = grid.lat.size * grid.lon.size                  # cell ids are row-major grid indices
    sums = {v: np.zeros(n) for v in features}
    counts = np.zeros(n)
    pf = pq.ParquetFile(cfg.paths.inputs.training_set)
    for k in range(pf.num_row_groups):
        df = pf.read_row_group(k, columns=["cell_id", "month", "year"] + features).to_pandas()
        days = np.empty(len(df))
        for year in df.year.unique():
            rows = (df.year == year).to_numpy()
            days[rows] = month_lengths(int(year))[df.month.to_numpy()[rows] - 1]
        cid = df.cell_id.to_numpy()
        counts += np.bincount(cid, minlength=n)
        for v in features:
            sums[v] += np.bincount(cid, weights=to_display_units(v, df[v].to_numpy(), days), minlength=n)
        if (k + 1) % 20 == 0 or k + 1 == pf.num_row_groups:
            print(f"  row group {k + 1}/{pf.num_row_groups}")

    kept = np.flatnonzero(counts)
    out = pd.DataFrame({"cell_id": kept, "lat": grid.lat[kept // grid.lon.size],
                        "lon": grid.lon[kept % grid.lon.size]})
    for v in features:
        out[v] = sums[v][kept] / counts[kept]
    print(f"  {len(out):,} cells, {int(counts.sum()):,} rows")
    return out


def load_cell_means(grid, recompute):
    path = cfg.paths.results.data_exploration / "predictor_means.parquet"
    if path.exists() and not recompute:
        print(f"Reading {path}")
        return pd.read_parquet(path)
    print("Computing per-cell means from the training set...")
    means = compute_cell_means(grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    means.to_parquet(path, index=False)
    print(f"  Saved: {path}")
    return means


# =============================================================================
# PLOT
# =============================================================================

def norm_for(var, values):
    lo, hi = np.nanmin(values), np.nanmax(values)
    if STYLE[var][1]:
        if var.startswith("aspect"):
            m = max(abs(lo), abs(hi))
            return Normalize(-m, m)
        if lo < 0 < hi:
            return TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
    return Normalize(lo, hi)


def panel_positions(n, ncols, panel_w, panel_h):
    """(left, bottom) in cm of each panel; an incomplete last row is centred."""
    nrows = -(-n // ncols)
    row_h = MARGIN_TOP + panel_h + BELOW_MAP
    height = nrows * row_h
    pos = []
    for k in range(n):
        r, c = divmod(k, ncols)
        in_row = min(ncols, n - r * ncols)
        shift = (ncols - in_row) * (panel_w + GAP_X) / 2
        left = MARGIN_LEFT + shift + c * (panel_w + GAP_X)
        bottom = height - (r + 1) * row_h + BELOW_MAP
        pos.append((left, bottom, c == 0))
    return pos, height


def plot_maps(grid, means, variables, ncols, width_cm, name, appendix):
    lon_min, lon_max, lat_min, lat_max = map_extent()
    ratio = (lat_max - lat_min) / (lon_max - lon_min) / np.cos(np.deg2rad((lat_min + lat_max) / 2))
    panel_w = (width_cm - MARGIN_LEFT - MARGIN_RIGHT - (ncols - 1) * GAP_X) / ncols
    panel_h = panel_w * ratio
    positions, height_cm = panel_positions(len(variables), ncols, panel_w, panel_h)

    fig = plt.figure(figsize=(cm(width_cm), cm(height_cm)))
    frac = lambda l, b, w, h: [l / width_cm, b / height_cm, w / width_cm, h / height_cm]
    for k, (var, (left, bottom, first_col)) in enumerate(zip(variables, positions)):
        ax = fig.add_axes(frac(left, bottom, panel_w, panel_h))
        values = means[var].to_numpy()
        im = draw_raster(ax, grid, cells_to_raster(grid, means.cell_id, values),
                         cmap=STYLE[var][0], norm=norm_for(var, values))
        draw_boundaries(ax, departments=False)
        setup_map(ax, left=first_col, bottom=True)
        add_panel_label(ax, k)

        cax = fig.add_axes(frac(left, bottom - CBAR_OFFSET - CBAR_HEIGHT, panel_w, CBAR_HEIGHT))
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label(label(var), labelpad=2)
        cbar.locator = mticker.MaxNLocator(5)
        cbar.update_ticks()
        cbar.outline.set_linewidth(0.4)

    save_figure(fig, name, appendix=appendix)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recompute", action="store_true", help="recompute the per-cell means")
    args = parser.parse_args()

    grid = reference_grid()
    means = load_cell_means(grid, args.recompute)
    setup_style()
    full = cfg.figures.width_cm.full
    plot_maps(grid, means, MAIN_VARIABLES, 2, 12.0, "01_predictor_maps", appendix=False)
    plot_maps(grid, means, CLIMATE_VARIABLES, 3, full, "01_predictor_maps_climate", appendix=True)
    plot_maps(grid, means, STATIC_VARIABLES, 3, full, "01_predictor_maps_static", appendix=True)


if __name__ == "__main__":
    main()
