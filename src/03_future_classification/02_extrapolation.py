"""
02_extrapolation.py
-------------------
How much of the projected climate lies outside the climate the final models were trained on.
Nothing is fitted or predicted: the climate predictors are computed straight from the monthly
climate archive with the code of the training set (utils/predictors.py, float64 as the training
set) and compared with the training envelope.

Training envelope: the minimum and maximum of each of the 7 climate predictors (PET, climbal,
pr, rsds, tas, tasmax, tasmin) over all rows of the training set (training.years, the 9 GCMs of
gcms.models, scenario training.climate_scenario). A cell-month of a GCM lies outside the envelope
if at least one climate predictor is below the minimum or above the maximum. This is the same as
a negative multivariate environmental similarity surface (MESS; Elith et al., 2010): MESS is the
lowest similarity over the predictors, and the similarity of a predictor is negative exactly when
its value lies outside the reference range. The static predictors are the same in training and
projections (every cell is in the training set), so they never lie outside.

Envelopes: "all_months", the training envelope; "same_month", the envelope of the training rows
of the same calendar month (one per calendar month).

Periods: periods.baseline of the historical scenario; periods.mid_century and periods.end_century
of each SSP; and, as a check, the training years of the training scenario (training_years), which
must lie inside both envelopes (the same climate as the training set, recomputed).

Shares: per GCM, the share of the cell-months of a period (cells x 12 months x years) outside the
envelope; "share" pools the cell-months of all GCMs (the mean of the GCM shares, as every GCM has
the same number), share_gcm_min and share_gcm_max are the range over the GCMs. Per predictor:
outside, above the maximum and below the minimum (outside = above + below); predictor "any": at
least one predictor outside (MESS < 0).

Regions (all_months envelope): departments (Natural Earth admin-1, paths.inputs.boundaries; the
polygon that contains the cell centre, else the nearest, for cells on the coast and the borders)
and elevation bands (the static elev, edges future_classification.extrapolation.elevation_bands_m).
Per region the share of its cell-months outside, and share_of_outside: its part of all the
outside cell-months of the country (pooled over the GCMs), to compare with cell_share, its part of
the cells.

Seasons: the shares of the calendar months of each season of
future_classification.extrapolation.seasons, under both envelopes.

Input:  training set (paths.inputs.training_set); monthly climate
        (paths.generated.climate_monthly_gcm); static layers (paths.raw.static_vars);
        boundaries (paths.inputs.boundaries)
Output: paths.results.future_classification/
          extrapolation_envelope.csv  per predictor and month ("all" or 1-12): units, rows of the
                                      training set, min and max
          extrapolation_shares.csv    per scenario, period, predictor and direction: first_year,
                                      last_year, cell_months (per GCM), share, share_gcm_min,
                                      share_gcm_max (all_months envelope)
          extrapolation_regions.csv   per region type (department, elevation_band), region,
                                      scenario, period, predictor and direction: cells,
                                      cell_share, share, share_gcm_min, share_gcm_max and
                                      share_of_outside (all_months envelope)
          extrapolation_seasons.csv   per scenario, period, season, envelope, predictor and
                                      direction: months, share, share_gcm_min, share_gcm_max
--trial reads the first GCM only and writes to <folder>_trial (timing and memory of the full run).

Processes: one per CPU of the allocation, one GCM-year each.

Usage:
    python 02_extrapolation.py [--trial]
"""

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import load_static_layers, reference_grid
from maps import COUNTRY
from predictors import MONTHS, add_pet_and_water_balance, load_climate
from resources import n_cpus, report_usage

SETTINGS = cfg.future_classification.extrapolation
GCMS = cfg.gcms.models
HIST = cfg.scenarios.historical
SSPS = cfg.scenarios.ssp
OUT_DIR = cfg.paths.results.future_classification
CLIMATE = [v for v in cfg.data.features if v in cfg.gcms.climate_vars + ["PET", "climbal"]]
UNITS = {"PET": "mm month-1", "climbal": "mm month-1", "pr": "kg m-2 s-1", "rsds": "W m-2",
         "tas": "K", "tasmax": "K", "tasmin": "K"}
ENVELOPES = ["all_months", "same_month"]
CHECK = "training_years"
# Flags counted per cell-month: any predictor outside, then each predictor above, then below.
N_FLAGS = 1 + 2 * len(CLIMATE)
UTM = "EPSG:32615"        # metric CRS for the nearest department (UTM 15N)

_STATE = {}               # grid, cells, envelopes, region ids; set before the worker processes start


# =============================================================================
# ENVELOPE, CELLS, PERIODS
# =============================================================================

def training_envelope():
    """Minimum and maximum of each climate predictor over the training rows of each calendar
    month (arrays months x predictors), and the rows per month."""
    f = pq.ParquetFile(cfg.paths.inputs.training_set)
    lo = np.full((MONTHS, len(CLIMATE)), np.inf)
    hi = np.full((MONTHS, len(CLIMATE)), -np.inf)
    rows = np.zeros(MONTHS, dtype=np.int64)
    for batch in f.iter_batches(batch_size=2**20, columns=["month"] + CLIMATE):
        df = batch.to_pandas()
        if df[CLIMATE].isna().any().any():
            raise ValueError("Missing climate values in the training set.")
        g = df.groupby("month")[CLIMATE]
        m = g.min().index.to_numpy() - 1
        lo[m] = np.minimum(lo[m], g.min().to_numpy())
        hi[m] = np.maximum(hi[m], g.max().to_numpy())
        rows[m] += g.size().to_numpy()
    if rows.sum() != f.metadata.num_rows or (rows == 0).any():
        raise ValueError(f"Training rows per month {rows.tolist()} do not add up to "
                         f"{f.metadata.num_rows:,} or a month is missing.")
    return lo, hi, rows


def envelope_table(lo, hi, rows):
    """Rows of extrapolation_envelope.csv."""
    out = []
    for k, var in enumerate(CLIMATE):
        out.append({"predictor": var, "month": "all", "units": UNITS[var], "rows": rows.sum(),
                    "min": lo[:, k].min(), "max": hi[:, k].max()})
        out += [{"predictor": var, "month": str(m + 1), "units": UNITS[var], "rows": rows[m],
                 "min": lo[m, k], "max": hi[m, k]} for m in range(MONTHS)]
    return pd.DataFrame(out)


def departments(lat, lon):
    """Department of each cell: the polygon that contains its centre, else the nearest one."""
    deps = gpd.read_file(cfg.paths.inputs.boundaries / cfg.files.boundaries,
                         where=f"admin = '{COUNTRY}'")[["name", "geometry"]]
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=deps.crs)
    hit = gpd.sjoin(points, deps, predicate="within", how="left")
    name = hit[~hit.index.duplicated()]["name"].reindex(points.index)
    missing = name.isna().to_numpy()
    nearest = gpd.sjoin_nearest(points[missing].to_crs(UTM), deps.to_crs(UTM), how="left")
    name.loc[missing] = nearest[~nearest.index.duplicated()]["name"].reindex(points.index[missing])
    print(f"  departments: {len(deps)}; {missing.sum():,} cells assigned to the nearest one")
    labels = sorted(deps["name"])
    return name.map({d: k for k, d in enumerate(labels)}).to_numpy().astype(int), labels


def elevation_bands(elev):
    """Elevation band (index) of each cell and the band labels (m), lowest first."""
    edges = list(SETTINGS.elevation_bands_m)
    labels = ([f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])]
              + [f">={edges[-1]}"])
    return np.digitize(elev, edges), labels


def periods():
    """(scenario, period, years) compared with the envelope; the check last."""
    out = [(HIST, "baseline", cfg.periods.baseline)]
    out += [(s, p, cfg.periods[p]) for s in SSPS for p in ["mid_century", "end_century"]]
    out.append((cfg.training.climate_scenario, CHECK, cfg.training.years))
    return out


# =============================================================================
# COUNTS OF ONE GCM-YEAR
# =============================================================================

def year_counts(unit):
    """Flag counts of one GCM-year: per envelope and calendar month (envelopes x months x flags,
    summed over the cells), and per region (regions x flags, all_months envelope)."""
    g, p, gcm, scenario, year = unit
    s = _STATE
    n = len(s["cells"])
    df = load_climate(s["grid"], s["cells"], gcm, year, scenario)
    add_pet_and_water_balance(df, year)
    if not np.array_equal(df.month.to_numpy()[::n], np.arange(1, MONTHS + 1)):
        raise ValueError("Unexpected row order of load_climate.")
    X = df[CLIMATE].to_numpy().reshape(MONTHS, n, len(CLIMATE))
    months = np.empty((len(ENVELOPES), MONTHS, N_FLAGS), dtype=np.int64)
    for e, (lo, hi) in enumerate(s["envelopes"]):
        above, below = X > hi[:, None, :], X < lo[:, None, :]
        flags = np.concatenate([(above | below).any(axis=2, keepdims=True), above, below], axis=2)
        months[e] = flags.sum(axis=1)
        if e == 0:
            per_cell = flags.sum(axis=0, dtype=np.int64)
            regions = np.zeros((s["n_regions"], N_FLAGS), dtype=np.int64)
            for ids in s["region_id"]:      # one region per cell and region type
                np.add.at(regions, ids, per_cell)
    return g, p, months, regions


# =============================================================================
# TABLES
# =============================================================================

def by_predictor(counts):
    """{(predictor, direction): counts[..., GCM]} from the flag axis (last) of counts."""
    out = {("any", "outside"): counts[..., 0]}
    for k, var in enumerate(CLIMATE):
        above, below = counts[..., 1 + k], counts[..., 1 + len(CLIMATE) + k]
        out[var, "outside"], out[var, "above"], out[var, "below"] = above + below, above, below
    return out


def share_columns(counts, total):
    """share (pooled over the GCMs), share_gcm_min and share_gcm_max of counts (one per GCM) of
    total cell-months per GCM."""
    share = counts / total
    return {"share": counts.sum() / (total * counts.size), "share_gcm_min": share.min(),
            "share_gcm_max": share.max()}


def tables(month_counts, region_counts, n_cells, region_labels, region_cells):
    """extrapolation_shares, _regions and _seasons; counts indexed [GCM, period, ...]."""
    shares, regions, seasons = [], [], []
    for p, (scenario, period, years) in enumerate(periods()):
        head = {"scenario": scenario, "period": period}
        per_gcm = n_cells * MONTHS * len(years)
        national = by_predictor(month_counts[:, p, 0].sum(axis=1))
        for (var, direction), c in national.items():
            shares.append({**head, "first_year": min(years), "last_year": max(years),
                           "predictor": var, "direction": direction, "cell_months": per_gcm,
                           **share_columns(c, per_gcm)})
        for r, (rtype, region) in enumerate(region_labels):
            total = region_cells[r] * MONTHS * len(years)
            for (var, direction), c in by_predictor(region_counts[:, p, r]).items():
                outside = national[var, direction].sum()
                regions.append({"region_type": rtype, "region": region, **head,
                                "predictor": var, "direction": direction,
                                "cells": region_cells[r], "cell_share": region_cells[r] / n_cells,
                                **share_columns(c, total),
                                "share_of_outside": c.sum() / outside if outside else np.nan})
        for season, months in SETTINGS.seasons.items():
            idx = np.array(months) - 1
            for e, envelope in enumerate(ENVELOPES):
                c = by_predictor(month_counts[:, p, e, idx].sum(axis=1))
                for (var, direction), v in c.items():
                    seasons.append({**head, "season": season, "envelope": envelope,
                                    "months": "-".join(map(str, months)), "predictor": var,
                                    "direction": direction,
                                    **share_columns(v, n_cells * len(months) * len(years))})
    return pd.DataFrame(shares), pd.DataFrame(regions), pd.DataFrame(seasons)


# =============================================================================
# MAIN
# =============================================================================

def main(trial):
    out_dir = OUT_DIR.with_name(OUT_DIR.name + "_trial") if trial else OUT_DIR
    gcms = GCMS[:1] if trial else GCMS

    print("Training envelope...", flush=True)
    lo, hi, rows = training_envelope()
    envelope = envelope_table(lo, hi, rows)
    print(envelope[envelope.month == "all"].to_string(index=False), flush=True)
    envelopes = [(np.broadcast_to(lo.min(axis=0), lo.shape), np.broadcast_to(hi.max(axis=0), hi.shape)),
                 (lo, hi)]

    grid = reference_grid()
    cells = load_static_layers(grid)[["i", "j", "elev"]]
    dep, dep_labels = departments(grid.lat[cells.i], grid.lon[cells.j])
    band, band_labels = elevation_bands(cells.elev.to_numpy())
    region_labels = ([("department", d) for d in dep_labels]
                     + [("elevation_band", b) for b in band_labels])
    region_id = [dep, len(dep_labels) + band]            # per region type: region of each cell
    region_cells = sum(np.bincount(ids, minlength=len(region_labels)) for ids in region_id)

    _STATE.update(grid=grid, cells=cells, envelopes=envelopes, n_regions=len(region_labels),
                  region_id=region_id)
    units = [(g, p, gcm, scenario, year) for g, gcm in enumerate(gcms)
             for p, (scenario, _, years) in enumerate(periods()) for year in years]
    month_counts = np.zeros((len(gcms), len(periods()), len(ENVELOPES), MONTHS, N_FLAGS), np.int64)
    region_counts = np.zeros((len(gcms), len(periods()), len(region_labels), N_FLAGS), np.int64)
    n_proc = n_cpus()
    print(f"  {len(cells):,} cells; {len(units):,} GCM-years on {n_proc} processes", flush=True)
    with get_context("fork").Pool(n_proc) as pool:      # workers inherit _STATE
        for k, (g, p, months, regions) in enumerate(pool.imap_unordered(year_counts, units), 1):
            month_counts[g, p] += months
            region_counts[g, p] += regions
            if k % 50 == 0 or k == len(units):
                print(f"  {k}/{len(units)}", flush=True)

    shares, regions, seasons = tables(month_counts, region_counts, len(cells), region_labels,
                                      region_cells)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [("envelope", envelope), ("shares", shares), ("regions", regions),
                        ("seasons", seasons)]:
        table.to_csv(out_dir / f"extrapolation_{name}.csv", index=False, float_format="%.6g")

    summary = shares[(shares.predictor == "any")]
    print("\nShare of the cell-months outside the training envelope (MESS < 0), pooled over the "
          "GCMs [GCM range]:")
    for r in summary.itertuples():
        print(f"  {r.scenario:<10} {r.period:<14} {r.share:8.2%}  [{r.share_gcm_min:.2%}, "
              f"{r.share_gcm_max:.2%}]")
    outside = month_counts[:, -1, :, :, 0].sum(axis=(0, 2))     # training years, per envelope
    if outside.any():
        print(f"\nWARNING: cell-months of the training years ({cfg.training.climate_scenario}) "
              f"outside the envelopes {dict(zip(ENVELOPES, outside.tolist()))}: the recomputed "
              "climate differs from the training set.")
    else:
        print("\nCheck passed: no cell-month of the training years lies outside either envelope.")
    print(f"\nSaved: {out_dir}/extrapolation_envelope.csv, _shares.csv, _regions.csv, _seasons.csv")
    report_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trial", action="store_true",
                        help="first GCM only, output to <folder>_trial")
    a = parser.parse_args()
    main(a.trial)
