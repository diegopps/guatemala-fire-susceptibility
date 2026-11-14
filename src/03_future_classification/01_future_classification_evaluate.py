"""
01_future_classification_evaluate.py
------------------------------------
Tables behind the figures of the projections, from the predictions of
01_future_classification.py. 01_future_classification_plot.py draws them.

Periods: periods.baseline for the historical scenario; periods.mid_century and
periods.end_century for each SSP.

High-risk share (per model, GCM, scenario and year): the share of the cells in
risk_classes.high_risk, each cell classified by its mean probability of the year with the frozen
breaks of step 05. Per period: the mean over the years of the period of each GCM, then the
median, minimum and maximum over the GCMs.

Period means: per model, the mean probability over the GCMs and the years of the period, per
cell and calendar month; the mean over the 12 months is the whole-year value of the cell. The
GCMs are averaged first, as the susceptibility of step 05.

Classes: the period means classified with the frozen breaks of each model
(utils/risk_thresholds.py), the monthly maps with the same whole-year breaks; and the median
class of the three models per cell and map, as in step 05. The classes of a period are not 20%
each: only the surface of step 05 is.

Class change: future class - baseline class of one model, per cell, for the whole year and per
calendar month (each month against the same month of the baseline), from -4 to +4. The
probabilities are not calibrated, so the projections are compared through the classes, which
rank the cells of each model on its own probability scale.

Input:  predictions of 01_future_classification.py (paths.generated.future_predictions); frozen
        breaks of 05_risk_maps.py
Output: paths.results.future_classification/
          national_yearly.csv   per model, GCM, scenario and year: high_share
          high_risk_shares.csv  per model, scenario and period (with the training years of the
                                training scenario): first_year, last_year and high_share
                                median, min and max over the GCMs
          class_shares.csv      per model and for the median class (summary), scenario, period,
                                map (year, month_1-12) and class: cells and share of the cells
          class_changes.csv     per model, SSP, future period, map (year, month_1-12) and class
                                change (-4 to +4): cells and share of the cells
          cells.parquet         per cell: cell_id, lat, lon; per model the baseline class
                                (<model>_baseline_class) and per model, SSP and future period the
                                class (<model>_<scenario>_<period>_class) and the class change
                                (<model>_<scenario>_<period>_class_change), whole year
          cell_months.parquet   per cell and calendar month: cell_id, month; per model the
                                baseline class (<model>_baseline_class) and per model and SSP the
                                class and class change of periods.end_century
                                (<model>_<scenario>_class, <model>_<scenario>_class_change)
          period_means.parquet  the period means: per model, scenario and period (baseline of the
                                historical scenario; mid_century and end_century of each SSP) and
                                cell: model, scenario, period, cell_id, year (whole year) and
                                month_1-12 (float32; the classes are computed in float64). The
                                column names are the maps of class_shares.csv

Usage:
    python 01_future_classification_evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import reference_grid
from predictors import MONTHS, scenario_years
from resources import report_usage
from risk_thresholds import classify, load_breaks

MODELS = cfg.model_training.models
GCMS = cfg.gcms.models
HIST = cfg.scenarios.historical
SSPS = cfg.scenarios.ssp
PRED_DIR = cfg.paths.generated.future_predictions
OUT_DIR = cfg.paths.results.future_classification
LABELS = cfg.risk_classes.labels
BASELINE = "baseline"
FUTURE_PERIODS = ["mid_century", "end_century"]
MONTHLY_PERIOD = "end_century"      # period of the monthly maps
MAX_CHANGE = len(LABELS) - 1        # largest class change


def prediction_file(model, gcm, scenario):
    return PRED_DIR / model / f"{gcm}_{scenario}.npy"


def periods_of(scenario):
    """Names of the periods of a scenario."""
    return [BASELINE] if scenario == HIST else FUTURE_PERIODS


def read_model(model, n_cells, breaks):
    """Yearly national rows of a model, and its period means {(scenario, period): array
    (months, cells)} over the GCMs."""
    rows, means = [], {}
    for gcm in GCMS:
        for scenario in [HIST] + SSPS:
            path, years = prediction_file(model, gcm, scenario), scenario_years(scenario)
            prob = np.load(path, mmap_mode="r")
            if prob.shape != (len(years), MONTHS, n_cells):
                raise ValueError(f"{path}: shape {prob.shape}, expected "
                                 f"{(len(years), MONTHS, n_cells)}.")
            for k, year in enumerate(years):
                annual = prob[k].mean(axis=0, dtype=np.float64)
                rows.append({"model": model, "gcm": gcm, "scenario": scenario, "year": year,
                             "high_share": np.isin(classify(annual, breaks),
                                                   cfg.risk_classes.high_risk).mean()})
            for period in periods_of(scenario):
                p = cfg.periods[period]
                mean = prob[p.start - years.start:p.stop - years.start].mean(axis=0, dtype=np.float64)
                means[scenario, period] = means.get((scenario, period), 0) + mean / len(GCMS)
        print(f"  {model} {gcm}", flush=True)
    return rows, means


def maps_of(year_values, month_values):
    """(name, values) of the whole-year map and of each monthly map (rows of month_values)."""
    return [("year", year_values)] + [(f"month_{m + 1}", month_values[m]) for m in range(MONTHS)]


def class_rows(summary, scenario, period, year_classes, month_classes):
    """Cells and share per class of the whole-year map and of each monthly map."""
    return [{"summary": summary, "scenario": scenario, "period": period, "map": name, "class": c,
             "label": LABELS[c - 1], "cells": int((classes == c).sum()),
             "share": float((classes == c).mean())}
            for name, classes in maps_of(year_classes, month_classes)
            for c in range(1, len(LABELS) + 1)]


def change_rows(model, scenario, period, year_change, month_change):
    """Cells and share per class change of the whole-year map and of each monthly map."""
    return [{"model": model, "scenario": scenario, "period": period, "map": name,
             "class_change": d, "cells": int((change == d).sum()),
             "share": float((change == d).mean())}
            for name, change in maps_of(year_change, month_change)
            for d in range(-MAX_CHANGE, MAX_CHANGE + 1)]


def period_table(model, scenario, period, cells, year_mean, mean):
    """Rows of period_means.parquet of one model, scenario and period: the whole-year and
    monthly period means of each cell."""
    maps = dict(maps_of(year_mean, mean))
    return pd.DataFrame({"model": model, "scenario": scenario, "period": period,
                         "cell_id": cells, **{k: v.astype(np.float32) for k, v in maps.items()}})


def median_class(classes):
    """Median of the classes of the models (axis 0: models)."""
    return np.sort(np.stack(classes), axis=0)[len(MODELS) // 2]


def high_risk_shares(yearly):
    """Share of the cells in risk_classes.high_risk per model, scenario and period: the mean over
    the years of the period of each GCM, then its median, minimum and maximum over the GCMs."""
    cases = [(HIST, BASELINE, cfg.periods[BASELINE])]
    cases += [(s, p, cfg.periods[p]) for s in SSPS for p in FUTURE_PERIODS]
    cases += [(cfg.training.climate_scenario, "training_years", cfg.training.years)]
    rows = []
    for model in MODELS:
        for scenario, period, years in cases:
            share = yearly[(yearly.model == model) & (yearly.scenario == scenario)
                           & yearly.year.isin(years)].groupby("gcm").high_share.mean()
            rows.append({"model": model, "scenario": scenario, "period": period,
                         "first_year": min(years), "last_year": max(years),
                         "high_share_median": share.median(), "high_share_min": share.min(),
                         "high_share_max": share.max()})
    return pd.DataFrame(rows)


def main():
    if len(MODELS) % 2 == 0:
        raise ValueError("The median class needs an odd number of models (model_training.models).")
    missing = [prediction_file(m, g, s) for m in MODELS for g in GCMS for s in [HIST] + SSPS
               if not prediction_file(m, g, s).exists()]
    if missing:
        sys.exit(f"{len(missing)} prediction files missing, e.g. {missing[0]}: run "
                 "01_future_classification.py first.")

    cells = np.load(PRED_DIR / "cells.npy")
    grid = reference_grid()
    breaks = load_breaks()
    cell_table = pd.DataFrame({"cell_id": cells, "lat": grid.lat[cells // grid.lon.size],
                               "lon": grid.lon[cells % grid.lon.size]})
    month_table = pd.DataFrame({"cell_id": np.repeat(cells, MONTHS),
                                "month": np.tile(np.arange(1, MONTHS + 1), cells.size)})
    yearly, shares, changes, year_classes, month_classes, period_means = [], [], [], {}, {}, []

    for model in MODELS:
        rows, means = read_model(model, cells.size, breaks[model])
        yearly += rows
        classes = {}
        for (scenario, period), mean in means.items():
            year_mean = mean.mean(axis=0)
            period_means.append(period_table(model, scenario, period, cells, year_mean, mean))
            year_class = classify(year_mean, breaks[model])
            month_class = classify(mean.ravel(), breaks[model]).reshape(mean.shape)
            classes[scenario, period] = year_class, month_class
            year_classes.setdefault((scenario, period), []).append(year_class)
            month_classes.setdefault((scenario, period), []).append(month_class)
            shares += class_rows(model, scenario, period, year_class, month_class)

        base_year, base_month = classes[HIST, BASELINE]
        cell_table[f"{model}_baseline_class"] = base_year
        month_table[f"{model}_baseline_class"] = base_month.T.ravel()
        for scenario in SSPS:
            for period in FUTURE_PERIODS:
                year_class, month_class = classes[scenario, period]
                year_change, month_change = year_class - base_year, month_class - base_month
                cell_table[f"{model}_{scenario}_{period}_class"] = year_class
                cell_table[f"{model}_{scenario}_{period}_class_change"] = year_change
                changes += change_rows(model, scenario, period, year_change, month_change)
                if period == MONTHLY_PERIOD:
                    month_table[f"{model}_{scenario}_class"] = month_class.T.ravel()
                    month_table[f"{model}_{scenario}_class_change"] = month_change.T.ravel()
                print(f"  {model} {scenario} {period}, whole year:{(year_change >= 1).mean():.1%} of the "
                      f"cells up >= 1 class, {(year_change >= 2).mean():.1%} up >= 2, "
                      f"{(year_change <= -1).mean():.1%} down >= 1")

    for (scenario, period), classes in year_classes.items():
        shares += class_rows("median", scenario, period, median_class(classes),
                             median_class(month_classes[scenario, period]))

    yearly = pd.DataFrame(yearly)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(OUT_DIR / "national_yearly.csv", index=False, float_format="%.6g")
    high_risk_shares(yearly).to_csv(OUT_DIR / "high_risk_shares.csv", index=False,
                                    float_format="%.6g")
    pd.DataFrame(shares).to_csv(OUT_DIR / "class_shares.csv", index=False, float_format="%.6g")
    pd.DataFrame(changes).to_csv(OUT_DIR / "class_changes.csv", index=False, float_format="%.6g")
    cell_table.to_parquet(OUT_DIR / "cells.parquet", index=False)
    month_table.to_parquet(OUT_DIR / "cell_months.parquet", index=False)
    period_means = pd.concat(period_means, ignore_index=True)
    for name, values in [("model", MODELS), ("scenario", [HIST] + SSPS),
                         ("period", [BASELINE] + FUTURE_PERIODS)]:
        period_means[name] = pd.Categorical(period_means[name], categories=values)
    period_means.to_parquet(OUT_DIR / "period_means.parquet", index=False)
    print(f"\nSaved: {OUT_DIR}/national_yearly.csv, high_risk_shares.csv, class_shares.csv, "
          f"class_changes.csv, cells.parquet, cell_months.parquet, period_means.parquet")
    for name in ["cells.parquet", "cell_months.parquet", "period_means.parquet"]:
        print(f"  {name}: {(OUT_DIR / name).stat().st_size / 1024**2:.1f} MB")
    print(f"  period_means.parquet: {len(period_means):,} rows "
          f"({period_means.groupby(['model', 'scenario', 'period'], observed=True).ngroups} "
          f"model x scenario x period)")
    report_usage()


if __name__ == "__main__":
    main()
