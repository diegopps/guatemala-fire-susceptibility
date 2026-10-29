"""
05_risk_maps_plot.py
--------------------
Maps of the fire-susceptibility classes of the final models (tables of 05_risk_maps.py).

Main figure (results/mainfigs/02_risk_maps.pdf, full width): the fire-season classes of
each model (model_training.models order), and per cell the median of the three classes.
Classes from the frozen breaks of each model (equal-area quintiles of its fire-season mean
probability over the training years).

Appendix (results/appendix/02_risk_maps_monthly_<model>.pdf, one per model): the mean of
each calendar month over the training years, classified with the same fire-season breaks.

Cells outside the pipeline's cell set (lakes) are left blank.

Input:  results/<model_training risk_maps>/cells.parquet, cell_months.parquet
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
from figures import MODELS as MODEL_STYLE, add_panel_label, cm, save_figure, setup_style
from maps import (COUNTRY_LINE, DEPARTMENT_LINE, add_scale_bar, cells_to_raster,
                  draw_boundaries, draw_raster, map_extent, setup_map)

MODELS = cfg.model_training.models
IN_DIR = cfg.model_training.paths.results.risk_maps
NAME = "02_risk_maps"
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


def main_figure(grid, cells):
    fig_w = cfg.figures.width_cm.full
    fig_h, rects = panel_grid(fig_w, 2, 2, left=1.1, right=0.2, top=0.5, bottom=1.6,
                              wgap=0.5, hgap=0.6)
    fig = plt.figure(figsize=(cm(fig_w), cm(fig_h)))
    panels = [(f"{m}_class", MODEL_STYLE[m]["label"]) for m in MODELS]
    panels.append(("median_class", "Median of the three models"))
    for i, (rect, (column, text)) in enumerate(zip(rects, panels)):
        ax = fig.add_axes(rect)
        draw_map(ax, grid, cells.cell_id, cells[column], CLASS_CMAP, left=i % 2 == 0,
                 bottom=i >= 2)
        add_panel_label(ax, i)
        ax.text(0.075, 1.02, text, transform=ax.transAxes, ha="left", va="bottom")
        if i == 0:
            add_scale_bar(ax, length_km=100, x=0.62, y=0.04)

    y = 0.95 / fig_h
    fig.legend(handles=class_handles(), title="Fire susceptibility (fire season)", ncol=5,
               loc="upper left", bbox_to_anchor=(1.1 / fig_w, y), handlelength=1.2,
               handleheight=0.9, columnspacing=1.0, borderaxespad=0,
               title_fontsize=cfg.figures.font_size)
    fig.legend(handles=border_handles(), ncol=1, loc="upper right",
               bbox_to_anchor=(1 - 0.2 / fig_w, y), handlelength=1.6, borderaxespad=0)
    save_figure(fig, NAME)
    plt.close(fig)


def monthly_figure(grid, months, model):
    fig_w = cfg.figures.width_cm.full
    fig_h, rects = panel_grid(fig_w, 3, 4, left=1.1, right=0.2, top=0.4, bottom=1.4,
                              wgap=0.3, hgap=0.45)
    fig = plt.figure(figsize=(cm(fig_w), cm(fig_h)))
    for m, rect in enumerate(rects, start=1):
        ax = fig.add_axes(rect)
        sub = months[months.month == m]
        draw_map(ax, grid, sub.cell_id, sub[f"{model}_class"], CLASS_CMAP,
                 left=(m - 1) % 4 == 0, bottom=m > 8, step=2)
        ax.text(0.0, 1.02, calendar.month_name[m], transform=ax.transAxes, ha="left", va="bottom")
    fig.legend(handles=class_handles() + border_handles(), ncol=7, loc="upper center",
               bbox_to_anchor=(0.5, 0.7 / fig_h), handlelength=1.2, handleheight=0.9,
               columnspacing=1.0, borderaxespad=0)
    save_figure(fig, f"{NAME}_monthly_{model}", appendix=True)
    plt.close(fig)


def main():
    setup_style()
    grid = reference_grid()
    cells = pd.read_parquet(IN_DIR / "cells.parquet")
    months = pd.read_parquet(IN_DIR / "cell_months.parquet")
    main_figure(grid, cells)
    for model in MODELS:
        monthly_figure(grid, months, model)


if __name__ == "__main__":
    main()
