"""
chelsa_grid.py
--------------
The analysis grid (CHELSA, 30 arc-seconds) and the set of cells used throughout the
pipeline.

monthly_climate_file(gcm, var, scenario, year)
    Path of a monthly per-GCM climate file (paths.generated.climate_monthly_gcm).
reference_grid()
    The grid of the monthly climate files.
Grid(nc_path)
    Cell lookup on the grid of a climate NetCDF; cell centres are its coordinates.
load_static_layers(grid)
    Values of every 30'' static layer (preprocessing.static_files, except aspect_sincos)
    for the kept cells: cells where every layer has a value. The layers are clipped to the national boundary, and
    aspect is undefined on water surfaces, so the kept cells are Guatemala's land cells.
"""

import numpy as np
import pandas as pd
import xarray as xr

from load_config import cfg

# Static files that are not 30'' layers and do not define the cell set
NOT_MASK_LAYERS = {"aspect_sincos"}


def monthly_climate_file(gcm, var, scenario, year):
    """Path of the monthly climate of one GCM, variable, scenario and year."""
    return cfg.paths.generated.climate_monthly_gcm / cfg.files.climate_monthly_gcm.format(
        gcm=gcm, var=var, scenario=scenario, year=year)


def reference_grid():
    """Grid of the monthly climate files (all share it)."""
    return Grid(monthly_climate_file(cfg.gcms.models[0], cfg.gcms.climate_vars[0],
                                     cfg.training.climate_scenario, cfg.training.years[0]))


class Grid:
    """Cell lookup on the climate grid; cell centres are the NetCDF coordinates."""

    def __init__(self, nc_path):
        with xr.open_dataset(nc_path) as ds:
            self.lat = ds.lat.values
            self.lon = ds.lon.values

    @property
    def shape(self):
        return self.lat.size, self.lon.size

    def matches(self, lat, lon, tol=1e-6):
        """True if the coordinates describe the same cells (tolerance as a fraction of a cell)."""
        step = abs(self.lat[1] - self.lat[0])
        return (lat.shape == self.lat.shape and lon.shape == self.lon.shape
                and np.abs(lat - self.lat).max() < tol * step
                and np.abs(lon - self.lon).max() < tol * step)

    def index(self, lat, lon):
        """Array indices (i, j) of the cells containing each point, and an in-grid mask."""
        fi = (np.asarray(lat) - self.lat[0]) / (self.lat[1] - self.lat[0])
        fj = (np.asarray(lon) - self.lon[0]) / (self.lon[1] - self.lon[0])
        i, j = np.rint(fi).astype(int), np.rint(fj).astype(int)
        if max(np.abs(fi - i).max(), np.abs(fj - j).max()) > 0.01:
            raise ValueError("Input points do not lie on the climate-grid cell centres.")
        inside = (i >= 0) & (i < self.lat.size) & (j >= 0) & (j < self.lon.size)
        return i, j, inside

    def cell_id(self, i, j):
        """Row-major cell index on the grid."""
        return np.asarray(i) * self.lon.size + np.asarray(j)


def load_static_layers(grid, verbose=False):
    """DataFrame with columns i, j and one column per static layer, for the kept cells."""
    cells = None
    for name, fname in cfg.preprocessing.static_files.items():
        if name in NOT_MASK_LAYERS:
            continue
        df = pd.read_csv(cfg.paths.raw.static_vars / fname).dropna(subset=[name])
        i, j, inside = grid.index(df.lat, df.lon)
        layer = pd.DataFrame({"i": i, "j": j, name: df[name].to_numpy()})[inside]
        if layer.duplicated(["i", "j"]).any():
            raise ValueError(f"{fname}: more than one value per grid cell.")
        cells = layer if cells is None else cells.merge(layer, on=["i", "j"])
        if verbose:
            print(f"  {fname:<22} {len(layer):>8,} cells -> {len(cells):,} with all layers so far")
    return cells.reset_index(drop=True)
