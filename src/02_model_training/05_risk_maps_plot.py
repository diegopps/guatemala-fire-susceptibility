"""
05_risk_maps_plot.py
--------------------
Monthly maps of the fire-susceptibility classes of the final models (tables of
05_risk_maps.py): the mean probability of each calendar month over the training years,
classified with the frozen whole-year breaks of each model (equal-area quintiles of its mean
probability over all months of the training years).

Main figure (results/mainfigs/02_risk_maps_monthly.pdf, full width): per cell and month the
median class of the three models.
Appendix (results/appendix/02_risk_maps_monthly_<model>.pdf, one per model): the classes of
each model.

Cells outside the pipeline's cell set (lakes) are left blank.

Input:  results/<model_training risk_maps>/cell_months.parquet
Output: the figures above

Usage:
    python 05_risk_maps_plot.py
"""

import calendar
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import reference_grid
from figures import cm, save_figure, setup_style
from maps import (COUNTRY_LINE, DEPARTMENT_LINE, cells_to_raster, draw_boundaries, draw_raster,
                  map_extent, setup_map)

MODELS = cfg.model_training.models
IN_DIR = cfg.model_training.paths.results.risk_maps
NAME = "02_risk_maps_monthly"
LABELS = cfg.risk_classes.labels

# ColorBrewer YlOrRd (5 classes): sequential, colour-blind safe.
CLASS_COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
CLASS_CMAP = (ListedColormap(CLASS_COLORS),
              BoundaryNorm(np.arange(0.5, len(CLASS_COLORS) + 1), len(CLASS_COLORS)))


def map_ratio():
    """Width / height of a national map (true distances at its mid-latitude)."""
    lon_min, lon_max, lat_min, lat_max = map_extent()
    return (lon_max - lon_min) * np.cos(np.deg2rad((lat_min + lat_max) / 2)) / (lat_max - lat_min)


def panel_grid(fig_w, nrows, ncols, left, right, top, bottom, wgap, hgap):
    """Figure height and axes rectangles (figure fractions) of an nrows x ncols grid of maps;
    sizes in cm."""
    w = (fig_w - left - right - (ncols - 1) * wgap) / ncols
    h = w / map_ratio()
    fig_h = top + bottom + nrows * h + (nrows - 1) * hgap
    rects = [[(left + c * (w + wgap)) / fig_w, (bottom + (nrows - 1 - r) * (h + hgap)) / fig_h,
              w / fig_w, h / fig_h] for r in range(nrows) for c in range(ncols)]
    return fig_h, rects


def draw_map(ax, grid, cell_id, values, cmap, left, bottom, step=1):
    cmap, norm = cmap
    draw_raster(ax, grid, cells_to_raster(grid, cell_id, values), cmap=cmap, norm=norm)
    draw_boundaries(ax)
    setup_map(ax, left=left, bottom=bottom)
    if step != 1:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(step))


def class_handles():
    return [Patch(facecolor=c, edgecolor="#7f7f7f", linewidth=0.3, label=l)
            for c, l in zip(CLASS_COLORS, LABELS)]


def border_handles():
    return [Line2D([], [], label="National border", **COUNTRY_LINE),
            Line2D([], [], label="Department border", **DEPARTMENT_LINE)]


def monthly_figure(grid, months, column, name, appendix):
    """12 calendar-month maps (3 x 4) of one class column of cell_months.parquet."""
    fig_w = cfg.figures.width_cm.full
    fig_h, rects = panel_grid(fig_w, 3, 4, left=1.1, right=0.2, top=0.4, bottom=1.4,
                              wgap=0.3, hgap=0.45)
    fig = plt.figure(figsize=(cm(fig_w), cm(fig_h)))
    for m, rect in enumerate(rects, start=1):
        ax = fig.add_axes(rect)
        sub = months[months.month == m]
        draw_map(ax, grid, sub.cell_id, sub[column], CLASS_CMAP,
                 left=(m - 1) % 4 == 0, bottom=m > 8, step=2)
        ax.text(0.0, 1.02, calendar.month_name[m], transform=ax.transAxes, ha="left", va="bottom")
    fig.legend(handles=class_handles() + border_handles(), ncol=7, loc="upper center",
               bbox_to_anchor=(0.5, 0.7 / fig_h), handlelength=1.2, handleheight=0.9,
               columnspacing=1.0, borderaxespad=0)
    save_figure(fig, name, appendix=appendix)
    plt.close(fig)


def main():
    setup_style()
    grid = reference_grid()
    months = pd.read_parquet(IN_DIR / "cell_months.parquet")
    monthly_figure(grid, months, "median_class", NAME, appendix=False)
    for model in MODELS:
        monthly_figure(grid, months, f"{model}_class", f"{NAME}_{model}", appendix=True)


if __name__ == "__main__":
    main()
