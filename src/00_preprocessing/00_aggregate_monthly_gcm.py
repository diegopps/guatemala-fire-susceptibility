"""
00_aggregate_monthly_gcm.py
---------------------------
Aggregates the daily CHELSA-downscaled ISIMIP3b climate of one GCM and scenario to
monthly means: one NetCDF per variable and year, with 12 monthly time steps.

Input   paths.raw.climate_gcms          files.climate_daily_gcm   (daily, 30'' grid)
Output  paths.generated.climate_monthly_gcm   files.climate_monthly_gcm

- All variables are averaged over the days of each month, so units are unchanged
  (pr stays a mean flux in kg m-2 s-1).
- Values are stored as float32. Cells outside the pipeline's cell set (Guatemala's
  land cells, see utils/chelsa_grid.py) are set to missing.
- A month may lack at most one day (the raw data miss 1 January in three GCM-years);
  the missing day is simply left out of that month's mean. It is reported as a warning
  and in a "missing_days" file attribute. Larger gaps stop the script.
- Existing output files are skipped, so an interrupted run can be restarted.

By default all years of the scenario are processed: periods.historical for
"historical", periods.future for the SSPs. A year range can be given instead.

Usage:
    python 00_aggregate_monthly_gcm.py <gcm> <scenario> [<first_year> <last_year>]

Examples:
    python 00_aggregate_monthly_gcm.py IPSL-CM6A-LR historical
    python 00_aggregate_monthly_gcm.py IPSL-CM6A-LR ssp370 2017 2024
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import Grid, load_static_layers

# A month may lack at most this many days; missing days are left out of the monthly mean
# (the gap is printed and stored in the file attribute "missing_days").
MAX_MISSING_DAYS_PER_MONTH = 1

ENCODING = {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}


def daily_file(gcm, var, scenario, year):
    pattern = cfg.paths.raw.climate_gcms / cfg.files.climate_daily_gcm.format(
        gcm=gcm, var=var, scenario=scenario, year=year)
    matches = sorted(glob.glob(str(pattern)))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one daily file for {pattern}, found {len(matches)}.")
    return Path(matches[0])


def monthly_file(gcm, var, scenario, year):
    return cfg.paths.generated.climate_monthly_gcm / cfg.files.climate_monthly_gcm.format(
        gcm=gcm, var=var, scenario=scenario, year=year)


def cell_mask(grid):
    """Boolean (lat, lon) array: True for the pipeline's cells."""
    cells = load_static_layers(grid)
    mask = np.zeros(grid.shape, dtype=bool)
    mask[cells.i.to_numpy(), cells.j.to_numpy()] = True
    return mask


def missing_days(times, year):
    """
    Dates of `year` absent from `times`, as ISO strings. Raises if a date is duplicated
    or outside the year, or if any month lacks more than MAX_MISSING_DAYS_PER_MONTH days.
    """
    days = pd.to_datetime(times).normalize()
    if days.duplicated().any() or (days.year != year).any():
        raise ValueError(f"Time axis has duplicated dates or dates outside {year}.")
    missing = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").difference(days)
    per_month = missing.month.value_counts()
    if (per_month > MAX_MISSING_DAYS_PER_MONTH).any():
        raise ValueError(f"Too many missing days in {year}: {per_month.to_dict()} (month: days).")
    return [d.strftime("%Y-%m-%d") for d in missing]


def aggregate(src, dst, var, year, grid, mask):
    with xr.open_dataset(src) as ds:
        if not grid.matches(ds.lat.values, ds.lon.values):
            raise ValueError(f"{src.name}: grid differs from the reference grid.")
        da = ds[var]
        missing = missing_days(da.time.values, year)
        if missing:
            print(f"    WARNING {src.name}: missing day(s) {', '.join(missing)}; "
                  f"left out of the monthly mean")
        months = da.time.dt.month
        monthly = da.groupby(months).mean("time").rename(month="time")
        monthly = monthly.where(xr.DataArray(mask, dims=("lat", "lon")))
        monthly = monthly.assign_coords(
            time=np.array([f"{year}-{m:02d}-01" for m in range(1, 13)], dtype="datetime64[ns]"))
        monthly.attrs = da.attrs
        out = monthly.to_dataset(name=var)
        out.attrs = {**ds.attrs,
                     "aggregation": f"monthly mean of daily values ({src.name})",
                     "masking": "cells outside the pipeline cell set (Guatemala land cells) set to missing"}
        if missing:
            out.attrs["missing_days"] = ", ".join(missing)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".nc.tmp")
        out.to_netcdf(tmp, encoding={var: ENCODING})
        tmp.rename(dst)


def main(gcm, scenario, first_year, last_year):
    if gcm not in cfg.gcms.models:
        raise ValueError(f"Unknown GCM {gcm!r}; expected one of {cfg.gcms.models}.")
    grid = Grid(daily_file(gcm, cfg.gcms.climate_vars[0], scenario, first_year))
    mask = cell_mask(grid)
    print(f"{gcm} {scenario}: grid {grid.shape[1]} x {grid.shape[0]}, {mask.sum():,} cells kept")

    done = skipped = 0
    for year in range(first_year, last_year + 1):
        for var in cfg.gcms.climate_vars:
            dst = monthly_file(gcm, var, scenario, year)
            if dst.exists():
                skipped += 1
                continue
            aggregate(daily_file(gcm, var, scenario, year), dst, var, year, grid, mask)
            done += 1
        print(f"  {gcm} {scenario} {year} done")
    print(f"{gcm} {scenario} {first_year}-{last_year}: {done} written, {skipped} already present")


def default_years(scenario):
    if scenario == cfg.scenarios.historical:
        return cfg.periods.historical
    if scenario in cfg.scenarios.ssp:
        return cfg.periods.future
    raise ValueError(f"Unknown scenario {scenario!r}.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        years = default_years(sys.argv[2])
        main(sys.argv[1], sys.argv[2], years.start, years.stop - 1)
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        print(__doc__)
        sys.exit(1)
