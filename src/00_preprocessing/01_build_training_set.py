"""
01_build_training_set.py
------------------------
Builds the model training table: one row per GCM, grid cell and month of the training
years, with the binary target (forest fire in that cell-month) and the 14 predictors.
The climate of every GCM in gcms.models is included (pooled training, one model per
algorithm); the fire labels are therefore repeated once per GCM.

Inputs (paths and settings in config.toml):
  - Fire records      paths.raw.fire_records — cell-months with at least one forest-fire report.
  - Monthly climate   paths.generated.climate_monthly_gcm — monthly means per GCM of the
                      CHELSA-downscaled ISIMIP3b data (tas, tasmax, tasmin, pr, rsds), built
                      by 00_aggregate_monthly_gcm.py (stored as float32, read as float64),
                      for the scenario training.climate_scenario.
  - Static layers     paths.raw.static_vars — elevation, slope, distances to roads, urban
                      areas and water, and aspect.

Grid and cells (utils/chelsa_grid.py)
  The grid is that of the climate NetCDFs (CHELSA, 30 arc-seconds). A cell is kept only
  if every static layer has a value; this also excludes water surfaces, where aspect is
  undefined. Aspect enters as sine and cosine components (static_files.aspect_sincos),
  sampled at each cell centre by nearest neighbour.

Derived variables (formulas in utils/climate_derived.py)
  Potential evapotranspiration PET (mm month-1), Priestley–Taylor.
  Climatic water balance (mm month-1): climbal = P - PET, with P = pr converted from
  kg m-2 s-1 to mm month-1, using the real number of days in each month.

Output columns
  Identifiers (data.id_columns, never used as predictors): lat, lon, month, year,
  cell_id (row-major index of the cell on the climate grid) and gcm.
  Target: fire (0/1). Predictors: data.features.
  Units: tas/tasmax/tasmin K; pr kg m-2 s-1; rsds W m-2; PET and climbal mm month-1;
  distances m; elev m; slope degrees; aspect components dimensionless.

Output: paths.inputs.training_set (Parquet, written per GCM and year; pyarrow splits each
        GCM-year into row groups of at most 1,048,576 rows)

Usage:
    python 01_build_training_set.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import Grid, load_static_layers, monthly_climate_file as climate_file
from climate_derived import month_lengths, pet_priestley_taylor, climatic_water_balance

CLIMATE_VARS = cfg.gcms.climate_vars
COLUMNS = cfg.data.id_columns + [cfg.data.target] + cfg.data.features


# =============================================================================
# LOADERS
# =============================================================================

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


def load_fires(grid):
    """Fire cell-months as grid indices (i, j, year, month)."""
    f = pd.read_csv(cfg.paths.raw.fire_records)
    i, j, inside = grid.index(f.lat, f.lon)
    return pd.DataFrame({"i": i, "j": j, "year": f.year, "month": f.month})[inside].drop_duplicates()


# =============================================================================
# MAIN
# =============================================================================

def main():
    years = cfg.training.years
    gcms = cfg.gcms.models
    scenario = cfg.training.climate_scenario

    grid = Grid(climate_file(gcms[0], CLIMATE_VARS[0], scenario, years[0]))
    print(f"Grid: {grid.lon.size} x {grid.lat.size} cells")

    print("Static layers:")
    cells = load_static(grid)
    print(f"  cells kept: {len(cells):,}")

    fires = load_fires(grid).assign(fire=np.int8(1))
    gcm_type = pd.CategoricalDtype(gcms)

    out = cfg.paths.inputs.training_set
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    writer, n_rows, n_fire = None, 0, 0
    try:
        for gcm in gcms:
            for year in years:
                df = load_climate(grid, cells, gcm, year, scenario)
                df["year"] = year
                add_pet_and_water_balance(df, year)
                df = (df.merge(cells, on=["i", "j"])
                        .merge(fires, on=["i", "j", "year", "month"], how="left"))
                df["fire"] = df.fire.fillna(0).astype("int8")
                df["gcm"] = pd.Categorical([gcm] * len(df), dtype=gcm_type)
                df = (df[COLUMNS]
                      .astype({"month": "int8", "year": "int16", "cell_id": "int32"})
                      .sort_values(["month", "lat", "lon"]))
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                writer.write_table(table)
                n_rows += len(df)
                n_fire += int(df.fire.sum())
            print(f"  {gcm}: {len(years)} years ({scenario})")
    finally:
        if writer is not None:
            writer.close()
    tmp.rename(out)

    fires_per_gcm = n_fire // len(gcms)
    print(f"\nRows: {n_rows:,} ({len(gcms)} GCMs x {len(cells):,} cells x {12 * len(years)} months)")
    print(f"Fire cell-months per GCM: {fires_per_gcm:,} "
          f"({len(fires) - fires_per_gcm} fire cell-months fall outside the kept cells)")
    print(f"Written: {out} ({out.stat().st_size / 1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
