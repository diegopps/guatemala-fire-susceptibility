"""
03_climate_trends.py
--------------------
Climate trends over Guatemala, 1961–2100, for the 7 climate predictors: the national
mean of each GCM and year, smoothed with a running mean (figures.running_mean_years,
centred, complete windows only), shown as the median of the GCMs and their range, for
the historical simulation and each SSP scenario.

Inputs: paths.generated.climate_monthly_gcm — monthly climate per GCM (historical and the
        SSPs in scenarios.ssp); the national mean is over the pipeline's cells
        (utils/chelsa_grid.py), as in the training set. PET and the climatic water balance
        are computed per cell and month (utils/climate_derived.py), then averaged.
Output: results/mainfigs/01_climate_trends.pdf
        results/<data_exploration>/climate_spatial_means.csv — national monthly means per
        GCM, scenario, year and month (display units), the data behind the figure. Reused
        unless --recompute is given.
        results/<data_exploration>/climate_changes.csv — changes across the GCMs (median,
        minimum, maximum): linear trend over the historical period, and the mid- and
        end-century means of each SSP relative to the baseline (periods in config.toml),
        absolute and in % of the baseline mean.

Smoothing at the junction: each GCM's historical series is continued by each SSP; the
running means of the historical years are averaged over the SSPs (they differ only in
the last few years, whose windows reach into 2015–2019).

Usage:
    python 03_climate_trends.py [--recompute]     (uses SLURM_CPUS_PER_TASK processes)
"""

import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import load_static_layers, monthly_climate_file, reference_grid
from climate_derived import climatic_water_balance, month_lengths, pet_priestley_taylor
from figures import (CLIMATE_VARIABLES, SCENARIOS, VARIABLES, add_panel_label, cm, label,
                     save_figure, setup_style, to_display_units)

HIST = cfg.scenarios.historical
SSPS = cfg.scenarios.ssp
WINDOW = cfg.figures.running_mean_years
OUT_DIR = cfg.paths.results.data_exploration
MEANS_FILE = OUT_DIR / "climate_spatial_means.csv"
CHANGES_FILE = OUT_DIR / "climate_changes.csv"

_CELLS = None             # (i, j) of the pipeline cells, set before the worker processes start


# =============================================================================
# NATIONAL MEANS
# =============================================================================

def national_means(unit):
    """National monthly means of the 7 climate predictors for one GCM, scenario and year."""
    gcm, scenario, year = unit
    i, j = _CELLS
    raw = {}
    for var in cfg.gcms.climate_vars:
        path = monthly_climate_file(gcm, var, scenario, year)
        with xr.open_dataset(path) as ds:
            values = ds[var].values[:, i, j].astype(np.float64)
        if np.isnan(values).any():
            raise ValueError(f"{path.name}: missing values in {np.isnan(values).any(axis=0).sum()} cells.")
        raw[var] = values
    days = month_lengths(year)[:, None]
    pet = pet_priestley_taylor(raw["tas"], raw["rsds"], days)
    fields = {
        "tas": to_display_units("tas", raw["tas"]),
        "tasmax": to_display_units("tasmax", raw["tasmax"]),
        "tasmin": to_display_units("tasmin", raw["tasmin"]),
        "pr": to_display_units("pr", raw["pr"], days),
        "PET": pet,
        "climbal": climatic_water_balance(raw["pr"], pet, days),
        "rsds": raw["rsds"],
    }
    out = pd.DataFrame({var: fields[var].mean(axis=1) for var in CLIMATE_VARIABLES})
    out.insert(0, "month", np.arange(1, 13))
    out.insert(0, "year", year)
    out.insert(0, "scenario", scenario)
    out.insert(0, "gcm", gcm)
    return out


def compute_national_means():
    global _CELLS
    cells = load_static_layers(reference_grid())
    _CELLS = (cells.i.to_numpy(), cells.j.to_numpy())
    units = [(g, HIST, y) for g in cfg.gcms.models for y in cfg.periods.historical]
    units += [(g, s, y) for g in cfg.gcms.models for s in SSPS for y in cfg.periods.future]
    n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"  {len(cells):,} cells; {len(units):,} GCM-scenario-years on {n_proc} processes")
    with get_context("fork").Pool(n_proc) as pool:      # workers inherit _CELLS
        parts = []
        for k, part in enumerate(pool.imap(national_means, units, chunksize=4), 1):
            parts.append(part)
            if k % 200 == 0 or k == len(units):
                print(f"  {k}/{len(units)}")
    return pd.concat(parts, ignore_index=True)


def load_national_means(recompute):
    if MEANS_FILE.exists() and not recompute:
        print(f"Reading {MEANS_FILE}")
        return pd.read_csv(MEANS_FILE)
    print("Computing national monthly means from the monthly climate...")
    means = compute_national_means()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    means.to_csv(MEANS_FILE, index=False, float_format="%.6g")
    print(f"  Saved: {MEANS_FILE}")
    return means


# =============================================================================
# TRENDS AND CHANGES
# =============================================================================

def yearly_by_gcm(means, var):
    """{scenario: DataFrame year x GCM} of yearly means (mean of the monthly values)."""
    yearly = means.groupby(["scenario", "year", "gcm"])[var].mean()
    return {s: yearly.loc[s].unstack("gcm") for s in [HIST] + SSPS}


def smoothed(yearly):
    """Running means per GCM: {scenario: DataFrame}; the historical years are averaged
    over the SSP continuations (see module docstring)."""
    roll = lambda df: df.rolling(WINDOW, center=True, min_periods=WINDOW).mean()
    hist_end = yearly[HIST].index.max()
    cont = {s: roll(pd.concat([yearly[HIST], yearly[s]])) for s in SSPS}
    out = {HIST: sum(cont[s].loc[:hist_end] for s in SSPS) / len(SSPS)}
    for s in SSPS:
        out[s] = cont[s].loc[hist_end:]          # starts at the last historical year, joining the lines
    return out


def plain_unit(var):
    unit = VARIABLES[var][1] or ""
    return unit.replace("$^{-1}$", "-1").replace("$^{-2}$", "-2")


def climate_changes(means):
    """Changes across the GCMs (median, min, max), absolute and relative to the baseline."""
    base = cfg.periods.baseline
    rows = []
    for var in CLIMATE_VARIABLES:
        yearly = yearly_by_gcm(means, var)
        base_mean = yearly[HIST].loc[base.start:base.stop - 1].mean()
        hist = yearly[HIST]
        slope = hist.apply(lambda col: np.polyfit(hist.index, col, 1)[0])
        cases = [(HIST, f"{hist.index.min()}–{hist.index.max()} (linear trend)",
                  slope * (hist.index.max() - hist.index.min()))]
        for s in SSPS:
            for name in ("mid_century", "end_century"):
                p = cfg.periods[name]
                cases.append((s, f"{p.start}–{p.stop - 1}",
                              yearly[s].loc[p.start:p.stop - 1].mean() - base_mean))
        for scenario, period, change in cases:
            pct = 100 * change / base_mean
            rows.append({"variable": var, "unit": plain_unit(var), "scenario": scenario,
                         "period": period, "baseline": f"{base.start}–{base.stop - 1}",
                         "change_median": change.median(), "change_min": change.min(),
                         "change_max": change.max(), "change_pct_median": pct.median(),
                         "change_pct_min": pct.min(), "change_pct_max": pct.max()})
    return pd.DataFrame(rows)


# =============================================================================
# PLOT
# =============================================================================

def plot_trends(means):
    setup_style()
    fig, axes = plt.subplots(4, 2, figsize=(cm(cfg.figures.width_cm.full), cm(20)), sharex=True)
    axes = axes.ravel()
    n_gcm = means.gcm.nunique()

    for k, var in enumerate(CLIMATE_VARIABLES):
        ax = axes[k]
        series = smoothed(yearly_by_gcm(means, var))
        for scenario, df in series.items():
            color = SCENARIOS[scenario]["color"]
            ok = df.notna().all(axis=1)
            df = df[ok]
            ax.fill_between(df.index, df.min(axis=1), df.max(axis=1), color=color, alpha=0.18,
                            linewidth=0, zorder=1)
            ax.plot(df.index, df.median(axis=1), color=color, zorder=3)
        ax.axvline(cfg.periods.historical.stop - 0.5, color="#9e9e9e", linewidth=0.5,
                   linestyle=(0, (3, 2)), zorder=0)
        ax.set_ylabel(label(var, wrap=True))     # unit on a second line: same layout in every panel
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.set_xlim(cfg.periods.historical.start, cfg.periods.future.stop - 1)
        ax.grid(axis="y", linewidth=0.3, color="#d9d9d9")
        ax.spines[["top", "right"]].set_visible(False)
        add_panel_label(ax, k, x=-0.02, y=1.0)

    for ax in axes[len(CLIMATE_VARIABLES) - 2:len(CLIMATE_VARIABLES)]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Year")

    leg_ax = axes[-1]
    leg_ax.axis("off")
    scen = [(Patch(facecolor=SCENARIOS[s]["color"], alpha=0.18, linewidth=0),
             Line2D([], [], color=SCENARIOS[s]["color"])) for s in [HIST] + SSPS]
    first = leg_ax.legend(scen, [SCENARIOS[s]["label"] for s in [HIST] + SSPS],
                          loc="upper left", bbox_to_anchor=(0.05, 0.95), handlelength=2)
    leg_ax.add_artist(first)
    key = [Line2D([], [], color="#4d4d4d"),
           Patch(facecolor="#4d4d4d", alpha=0.18, linewidth=0)]
    leg_ax.legend(key, [f"Median of {n_gcm} GCMs", f"Range of {n_gcm} GCMs"],
                  title=f"{WINDOW}-year running means", alignment="left",
                  loc="upper left", bbox_to_anchor=(0.05, 0.45), handlelength=2)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.05, top=0.98, wspace=0.25, hspace=0.18)
    save_figure(fig, "01_climate_trends")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recompute", action="store_true", help="recompute the national means")
    args = parser.parse_args()

    means = load_national_means(args.recompute)

    changes = climate_changes(means)
    changes.to_csv(CHANGES_FILE, index=False, float_format="%.4f")
    print(f"  Saved: {CHANGES_FILE}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(changes[changes.variable.isin(["tas", "pr"])]
              [["variable", "scenario", "period", "change_median", "change_min", "change_max",
                "change_pct_median"]].round(2).to_string(index=False))

    plot_trends(means)


if __name__ == "__main__":
    main()
