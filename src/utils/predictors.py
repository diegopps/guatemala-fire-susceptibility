"""
predictors.py
-------------
The predictors of a GCM-year, computed from the monthly climate archive and the static
layers. Shared by the training set (00_preprocessing/01_build_training_set.py) and the
projections (src/03_future_classification), so both use the same values.

load_static(grid)                       static predictors of the kept cells
load_climate(grid, cells, gcm, year, scenario)
                                        monthly climate of one GCM-year for all cells
add_pet_and_water_balance(df, year)     adds PET and climbal (utils/climate_derived.py)
predictor_matrix(grid, cells, gcm, year, scenario)
                                        predictors of one GCM-year as a model matrix
scenario_years(scenario)                years of a scenario in the archive
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from load_config import cfg
from chelsa_grid import load_static_layers, monthly_climate_file as climate_file
from climate_derived import month_lengths, pet_priestley_taylor, climatic_water_balance

CLIMATE_VARS = cfg.gcms.climate_vars
MONTHS = 12


def load_static(grid):
    """Static predictors for the kept cells (see chelsa_grid.load_static_layers)."""
    cells = load_static_layers(grid, verbose=True)

    # Aspect degrees only define where aspect exists; the predictors are its components.
    cells = cells.drop(columns="aspect").reset_index(drop=True)
    cells["lat"] = grid.lat[cells.i]
    cells["lon"] = grid.lon[cells.j]
    cells["cell_id"] = grid.cell_id(cells.i, cells.j)

    a = pd.read_csv(cfg.paths.raw.static_vars / cfg.preprocessing.static_files.aspect_sincos)
    alat, alon = np.sort(a.lat.unique()), np.sort(a.lon.unique())
    points = np.c_[cells.lat, cells.lon]
    for comp in ["aspect_sin", "aspect_cos"]:
        values = a.pivot(index="lat", columns="lon", values=comp).reindex(index=alat, columns=alon)
        cells[comp] = RegularGridInterpolator((alat, alon), values.to_numpy(), method="nearest")(points)
        if cells[comp].isna().any():
            raise ValueError(f"{comp}: no value for {cells[comp].isna().sum()} cells.")
    return cells


def load_climate(grid, cells, gcm, year, scenario):
    """Monthly climate of one GCM and year for all cells: columns i, j, month and CLIMATE_VARS."""
    n = len(cells)
    out = {"i": np.tile(cells.i.to_numpy(), 12),
           "j": np.tile(cells.j.to_numpy(), 12),
           "month": np.repeat(np.arange(1, 13), n)}
    for var in CLIMATE_VARS:
        path = climate_file(gcm, var, scenario, year)
        with xr.open_dataset(path) as ds:
            if not grid.matches(ds.lat.values, ds.lon.values):
                raise ValueError(f"{path.name}: grid differs from the reference grid.")
            if not np.array_equal(ds.time.dt.month.values, np.arange(1, 13)):
                raise ValueError(f"{path.name}: expected 12 monthly steps, January to December.")
            values = ds[var].values[:, cells.i.to_numpy(), cells.j.to_numpy()].astype(np.float64)
        if np.isnan(values).any():
            raise ValueError(f"{path.name}: missing values in {np.isnan(values).any(axis=0).sum()} cells.")
        out[var] = values.ravel()
    return pd.DataFrame(out)


def add_pet_and_water_balance(df, year):
    """Adds PET and climbal (mm month-1) in place."""
    days = month_lengths(year)[df.month.to_numpy() - 1]
    df["PET"] = pet_priestley_taylor(df.tas.to_numpy(), df.rsds.to_numpy(), days)
    df["climbal"] = climatic_water_balance(df.pr.to_numpy(), df.PET.to_numpy(), days)


def predictor_matrix(grid, cells, gcm, year, scenario):
    """Predictors of all cells in the 12 months of one GCM-year: float32 array of shape
    (12, cells, predictors), predictors in the order of data.features, cells in the order of
    the argument. Computed in float64 and cast at the end, as the model matrices of
    00_split_data.py. An error if any value is missing."""
    df = load_climate(grid, cells, gcm, year, scenario)
    add_pet_and_water_balance(df, year)
    columns = {c: df[c].to_numpy() if c in df else np.tile(cells[c].to_numpy(), MONTHS)
               for c in cfg.data.features}
    X = pd.DataFrame(columns, copy=False).to_numpy(np.float32)
    missing = np.isnan(X).any(axis=0)
    if missing.any():
        raise ValueError(f"{gcm} {scenario} {year}: missing values in "
                         f"{[c for c, m in zip(cfg.data.features, missing) if m]}.")
    return X.reshape(MONTHS, len(cells), len(columns))


def scenario_years(scenario):
    """Years of a scenario in the monthly archive."""
    return cfg.periods.historical if scenario == cfg.scenarios.historical else cfg.periods.future
