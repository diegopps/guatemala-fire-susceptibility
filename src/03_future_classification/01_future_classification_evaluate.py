"""
01_future_classification_evaluate.py
------------------------------------
Tables behind the figures of the projections, from the predictions of
01_future_classification.py. 01_future_classification_plot.py draws them.

Periods: periods.baseline for the historical scenario; periods.mid_century and
periods.end_century for each SSP.

National means (per model, GCM, scenario and year): the mean probability of all cells and
months of the year; and the share of the cells in risk_classes.high_risk, each cell classified
by its mean probability of the year with the frozen breaks of step 05 (for the text only).

Period means: per model, the mean probability over the GCMs and the years of the period, per
cell and calendar month; the mean over the 12 months is the whole-year value of the cell. The
GCMs are averaged first, as the susceptibility of step 05.

Classes: the period means classified with the frozen breaks of each model
(utils/risk_thresholds.py), and the median class of the three models per cell and map, as in
step 05. The classes of a period are not 20% each: only the surface of step 05 is.

Relative change: 100 x (future / baseline - 1) of the period means of one model, per cell (and
calendar month). Probabilities are not calibrated, so the change is in the probability scale of each model and only its relative
value is comparable. It is 0 where both means are 0, and infinite where only the baseline is 0
(the number of these cell-months is printed).

Input:  predictions of 01_future_classification.py (paths.generated.future_predictions); frozen
        breaks of 05_risk_maps.py
Output: paths.results.future_classification/
          national_yearly.csv   per model, GCM, scenario and year: mean_probability, high_share
          national_changes.csv  per model, scenario and period (with the training years of the
                                training scenario): change of the national mean relative to the
                                baseline and high_share, median, min and max over the GCMs
          class_shares.csv      per model and for the median class (summary), scenario, period,
                                map (year, month_1-12) and class: cells and share of the cells
          cells.parquet         per cell: cell_id, lat, lon; per model the baseline class
                                (<model>_baseline_class) and the relative change
                                (<model>_<scenario>_<period>), whole year
          cell_months.parquet   per cell and calendar month: cell_id, month and <model>_<scenario>
                                (relative change, periods.end_century)

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
                             "mean_probability": annual.mean(),
                             "high_share": np.isin(classify(annual, breaks),
                                                   cfg.risk_classes.high_risk).mean()})
            for period in periods_of(scenario):
                p = cfg.periods[period]
                mean = prob[p.start - years.start:p.stop - years.start].mean(axis=0, dtype=np.float64)
                means[scenario, period] = means.get((scenario, period), 0) + mean / len(GCMS)
        print(f"  {model} {gcm}", flush=True)
    return rows, means


def relative_change(future, base):
    """100 x (future / base - 1); 0 where both are 0, infinite where only base is."""
    with np.errstate(divide="ignore", invalid="ignore"):
        change = 100 * (future / base - 1)
    change[(future == 0) & (base == 0)] = 0
    return change


def class_rows(summary, scenario, period, year_classes, month_classes):
    """Cells and share per class of the whole-year map and of each monthly map."""
    maps = [("year", year_classes)] + [(f"month_{m + 1}", month_classes[m]) for m in range(MONTHS)]
    return [{"summary": summary, "scenario": scenario, "period": period, "map": name, "class": c,
             "label": LABELS[c - 1], "cells": int((classes == c).sum()),
             "share": float((classes == c).mean())}
            for name, classes in maps for c in range(1, len(LABELS) + 1)]


def median_class(classes):
    """Median of the classes of the models (axis 0: models)."""
    return np.sort(np.stack(classes), axis=0)[len(MODELS) // 2]


def national_changes(yearly):
    """Change of the national mean relative to the baseline and share in High or Very high,
    per model, scenario and period, over the GCMs."""
    is_base = (yearly.scenario == HIST) & yearly.year.isin(cfg.periods[BASELINE])
    base = yearly[is_base].groupby(["model", "gcm"]).mean_probability.mean()
    cases = [(HIST, BASELINE, cfg.periods[BASELINE])]
    cases += [(s, p, cfg.periods[p]) for s in SSPS for p in FUTURE_PERIODS]
    cases += [(cfg.training.climate_scenario, "training_years", cfg.training.years)]
    rows = []
    for model in MODELS:
        for scenario, period, years in cases:
            sub = yearly[(yearly.model == model) & (yearly.scenario == scenario)
                         & yearly.year.isin(years)].groupby("gcm")[["mean_probability", "high_share"]].mean()
            change = 100 * (sub.mean_probability / base.loc[model] - 1)
            rows.append({"model": model, "scenario": scenario, "period": period,
                         "change_median": change.median(), "change_min": change.min(),
                         "change_max": change.max(), "high_share_median": sub.high_share.median(),
                         "high_share_min": sub.high_share.min(),
                         "high_share_max": sub.high_share.max()})
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
    yearly, shares, year_classes, month_classes = [], [], {}, {}

    for model in MODELS:
        rows, means = read_model(model, cells.size, breaks[model])
        yearly += rows
        base = means[HIST, BASELINE]
        print(f"{model}: baseline cell-months with probability 0: {(base == 0).mean():.3%}")
        for (scenario, period), mean in means.items():
            year_mean = mean.mean(axis=0)
            year_class = classify(year_mean, breaks[model])
            month_class = classify(mean.ravel(), breaks[model]).reshape(mean.shape)
            year_classes.setdefault((scenario, period), []).append(year_class)
            month_classes.setdefault((scenario, period), []).append(month_class)
            shares += class_rows(model, scenario, period, year_class, month_class)
            if scenario == HIST:
                cell_table[f"{model}_baseline_class"] = year_class
                continue
            cell_table[f"{model}_{scenario}_{period}"] = relative_change(
                year_mean, base.mean(axis=0)).astype(np.float32)
            if period == MONTHLY_PERIOD:
                change = relative_change(mean, base)
                month_table[f"{model}_{scenario}"] = change.T.ravel().astype(np.float32)
                finite = change[np.isfinite(change)]
                print(f"  {scenario}, {period}: infinite change in {np.isinf(change).mean():.3%} of "
                      f"the cell-months; percentiles 1, 50, 99 of the change: "
                      f"{', '.join(f'{v:.1f}' for v in np.percentile(finite, [1, 50, 99]))} %")

    for (scenario, period), classes in year_classes.items():
        shares += class_rows("median", scenario, period, median_class(classes),
                             median_class(month_classes[scenario, period]))

    yearly = pd.DataFrame(yearly)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(OUT_DIR / "national_yearly.csv", index=False, float_format="%.6g")
    national_changes(yearly).to_csv(OUT_DIR / "national_changes.csv", index=False, float_format="%.6g")
    pd.DataFrame(shares).to_csv(OUT_DIR / "class_shares.csv", index=False, float_format="%.6g")
    cell_table.to_parquet(OUT_DIR / "cells.parquet", index=False)
    month_table.to_parquet(OUT_DIR / "cell_months.parquet", index=False)
    print(f"\nSaved: {OUT_DIR}/national_yearly.csv, national_changes.csv, class_shares.csv, "
          f"cells.parquet, cell_months.parquet")
    report_usage()


if __name__ == "__main__":
    main()
