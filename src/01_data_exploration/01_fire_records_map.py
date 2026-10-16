"""
01_fire_records_map.py
----------------------
Map of the grid cells with at least one forest-fire record in the training years
(Fig. 1 of the paper, study area and fire data). Cells outside the pipeline's cell set
(lakes, where no aspect is defined; utils/chelsa_grid.py) are left blank.

Input:  paths.inputs.training_set — the fire labels (identical for every GCM; the first
        GCM's rows are read).
Output: results/mainfigs/00_fire_records_map.pdf
        results/<data_exploration>/fire_cells.csv — cells with a fire record
        (cell_id, lat, lon, n_fire_months), the data behind the map.

Usage:
    python 01_fire_records_map.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import reference_grid
from figures import cm, save_figure, setup_style
from maps import (COUNTRY_LINE, DEPARTMENT_LINE, NEIGHBOUR_LINE, add_scale_bar,
                  cells_to_raster, draw_boundaries, draw_locator, draw_raster, setup_map)

NAME = "00_fire_records_map"     # introduction figure, hence stage prefix 00
LAND_COLOR = "#d9d9d9"
FIRE_COLOR = "#b2182b"


def load_fire_cells():
    """Per cell (all kept cells): number of months with a fire record, plus lat/lon."""
    table = pq.read_table(cfg.paths.inputs.training_set,
                          columns=["cell_id", "lat", "lon", cfg.data.target],
                          filters=[("gcm", "==", cfg.gcms.models[0])])
    df = table.to_pandas()
    return (df.groupby("cell_id")
              .agg(lat=("lat", "first"), lon=("lon", "first"),
                   n_fire_months=(cfg.data.target, "sum"))
              .reset_index())


def plot(grid, cells):
    setup_style()
    fig = plt.figure(figsize=(cm(13), cm(9.2)))
    ax = fig.add_axes([0.065, 0.06, 0.6, 0.92])
    ax_loc = fig.add_axes([0.69, 0.64, 0.3, 0.33])

    has_fire = (cells.n_fire_months > 0).astype(float)
    raster = cells_to_raster(grid, cells.cell_id, has_fire)
    draw_raster(ax, grid, raster, cmap=ListedColormap([LAND_COLOR, FIRE_COLOR]),
                norm=BoundaryNorm([-0.5, 0.5, 1.5], 2))
    draw_boundaries(ax)
    setup_map(ax)
    ax.set_anchor("W")
    add_scale_bar(ax, length_km=100, x=0.66, y=0.04)

    draw_locator(ax_loc)
    ax_loc.set_anchor("N")

    handles = [
        Patch(facecolor=FIRE_COLOR, label="Fire recorded"),
        Patch(facecolor=LAND_COLOR, label="No fire recorded"),
        Line2D([], [], label="National border", **COUNTRY_LINE),
        Line2D([], [], label="Department border", **DEPARTMENT_LINE),
        Line2D([], [], label="Neighbouring country", **NEIGHBOUR_LINE),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.69, 0.58),
               handlelength=1.2, handleheight=0.9, borderaxespad=0)

    save_figure(fig, NAME)
    plt.close(fig)


def main():
    grid = reference_grid()
    cells = load_fire_cells()
    n_fire = int((cells.n_fire_months > 0).sum())
    print(f"Cells: {len(cells):,}; with at least one fire record: {n_fire:,} "
          f"({cells.n_fire_months.sum():,} fire cell-months)")

    out_dir = cfg.paths.results.data_exploration
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fire_cells.csv"
    cells[cells.n_fire_months > 0].to_csv(out, index=False)
    print(f"  Saved: {out}")

    plot(grid, cells)


if __name__ == "__main__":
    main()
