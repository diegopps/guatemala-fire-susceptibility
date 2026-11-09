"""
05_risk_maps.py
---------------
Fire-susceptibility classes of the final models (step 04) over the training years: the
per-cell susceptibility, the frozen class breaks, the classes of every cell, and the
agreement between the models. 05_risk_maps_plot.py draws the maps.

Susceptibility of a cell (the one definition used here and in stage 3): the mean fire
probability of the final model over the 9 GCMs and all months of the training years
(training.years), i.e. the whole year. The predictions are those of 04_final_models.py for
every row of the train, test and validation subsets, so every cell-month of the training
years enters; the train rows are in-sample, as in any susceptibility map. No score is
computed here: the performance of the models is that of the validation subset
(04_final_models_evaluate.py).

Classes (utils/risk_thresholds.py, config risk_classes): the breaks are the percentiles
risk_classes.quintiles of the susceptibility of all cells, one set per model, so each class
holds 20% of the cells (equal-area classes, Verde and Zêzere, 2010). They are frozen here and
applied unchanged to the monthly means below and to the projections (stage 3). The
probabilities are averaged as the models give them (no calibration): the classes rank the
cells of one model against each other. All cells of the grid are classified (no land-cover
mask), so each class is 20% of the national territory.

Monthly means: the same mean for each calendar month, classified with the same breaks.

Agreement between two models (whole-year classes of all cells): share of cells in the same
class, share within one class, and Cohen's kappa with quadratic weights (classes are
ordered); and, per cell, the number of models placing it in risk_classes.high_risk.

Median class: per cell, the median of the whole-year classes of the models (the middle one
of the sorted classes; model_training.models has an odd number of models), the class the
models agree on; likewise per cell and calendar month.

Checks: every cell has the same number of rows in every calendar month (years x GCMs), so
each mean weighs every cell-month and GCM equally; the breaks are strictly ascending; class
shares and cells equal to a break are printed.

Input:  predictions of 04_final_models.py (<model>_<subset>.npy, model_training.paths.
        generated.final_models); model data of 00_split_data.py (cell and month of each row)
Output: paths.breaks         breaks.json in the repository root (tracked by git: the classes
                              of the published models need it): definition, percentiles,
                              labels; per model the breaks, cells per class and cells equal
                              to a break (read by stage 3, risk_thresholds.load_breaks)
        results/<model_training risk_maps>/
          cells.parquet       per cell: cell_id, lat, lon; per model the susceptibility
                              (<model>) and its class (<model>_class); n_models_high;
                              median_class
          cell_months.parquet per cell and calendar month: cell_id, month; per model the mean
                              probability and its class; median_class
          classes.csv         per model and map (year, month_1-12): cells and share of
                              the cells per class, range of the values
          agreement.csv       per pair of models: same class, within one class, weighted kappa
                              (+ one row per number of models in High or Very high)

Usage:
    python 05_risk_maps.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import reference_grid
from resources import report_usage
import model_data
from risk_thresholds import BREAKS_FILE, classify, quintile_breaks

MODELS = cfg.model_training.models
PRED_DIR = cfg.model_training.paths.generated.final_models
OUT_DIR = cfg.model_training.paths.results.risk_maps
LABELS = cfg.risk_classes.labels
MONTHS = 12


def cell_month_keys():
    """Cells (sorted cell ids) and, per subset, the key cell index x 12 + month - 1 of each row;
    checks that every cell has the same number of rows in every month."""
    arrays = {s: model_data.load_subset(s, arrays=("cell_id", "month")) for s in model_data.SUBSETS}
    cells = np.unique(np.concatenate([np.unique(a["cell_id"]) for a in arrays.values()]))
    keys = {s: (np.searchsorted(cells, a["cell_id"]).astype(np.int32) * MONTHS
                + np.asarray(a["month"], dtype=np.int32) - 1) for s, a in arrays.items()}
    counts = sum(np.bincount(k, minlength=cells.size * MONTHS) for k in keys.values())
    expected = len(cfg.training.years) * len(cfg.gcms.models)
    if not (counts == expected).all():
        raise ValueError(f"Rows per cell and month range from {counts.min()} to {counts.max()}, "
                         f"expected {expected} (years x GCMs) everywhere.")
    print(f"{cells.size:,} cells x {MONTHS} months, {expected} rows each "
          f"({len(cfg.training.years)} years x {len(cfg.gcms.models)} GCMs); "
          f"{sum(k.size for k in keys.values()):,} rows")
    return cells, keys, expected


def monthly_means(model, keys, n_cells, rows_per_key):
    """Mean probability of each cell (rows) and calendar month (columns) over its rows."""
    sums = np.zeros(n_cells * MONTHS)
    for subset, key in keys.items():
        path = PRED_DIR / f"{model}_{subset}.npy"
        prob = np.load(path)
        if prob.size != key.size:
            raise ValueError(f"{path}: {prob.size:,} predictions for {key.size:,} {subset} rows.")
        sums += np.bincount(key, weights=prob, minlength=sums.size)
    return (sums / rows_per_key).reshape(n_cells, MONTHS)


def class_rows(model, map_name, values, classes):
    """Cells and value range per class of one map."""
    return [{"model": model, "map": map_name, "class": c, "label": LABELS[c - 1],
             "cells": int((classes == c).sum()), "share": float((classes == c).mean()),
             "min": float(values[classes == c].min()) if (classes == c).any() else np.nan,
             "max": float(values[classes == c].max()) if (classes == c).any() else np.nan}
            for c in range(1, len(LABELS) + 1)]


def agreement(classes):
    """Pairwise agreement of the whole-year classes, and cells per number of models in
    risk_classes.high_risk."""
    rows = []
    for a, b in itertools.combinations(classes, 2):
        ca, cb = classes[a], classes[b]
        rows.append({"model_a": a, "model_b": b, "same_class": float((ca == cb).mean()),
                     "within_one_class": float((np.abs(ca.astype(int) - cb) <= 1).mean()),
                     "kappa_quadratic": cohen_kappa_score(ca, cb, weights="quadratic")})
        print(f"  {a} / {b}: same class {rows[-1]['same_class']:.1%}, within one class "
              f"{rows[-1]['within_one_class']:.1%}, quadratic-weighted kappa "
              f"{rows[-1]['kappa_quadratic']:.3f}")
    n_high = sum(np.isin(c, cfg.risk_classes.high_risk) for c in classes.values())
    for n in range(len(classes) + 1):
        rows.append({"models_high": n, "cells": int((n_high == n).sum()),
                     "share": float((n_high == n).mean())})
    print("  cells in High or Very high in n models: " + ", ".join(
        f"{n}: {(n_high == n).mean():.1%}" for n in range(len(classes) + 1)))
    return rows, n_high.astype(np.int8)


def main():
    cells, keys, rows_per_key = cell_month_keys()
    grid = reference_grid()
    cell_table = pd.DataFrame({"cell_id": cells, "lat": grid.lat[cells // grid.lon.size],
                               "lon": grid.lon[cells % grid.lon.size]})
    month_table = pd.DataFrame({"cell_id": np.repeat(cells, MONTHS),
                                "month": np.tile(np.arange(1, MONTHS + 1), cells.size)})
    breaks_info, class_table, year_classes = {}, [], {}

    for model in MODELS:
        missing = [s for s in keys if not (PRED_DIR / f"{model}_{s}.npy").exists()]
        if missing:
            raise FileNotFoundError(f"{model}: no predictions for {missing} in {PRED_DIR}: run "
                                    f"04_final_models.py (fit, predict) first.")
        means = monthly_means(model, keys, cells.size, rows_per_key)
        susceptibility = means.mean(axis=1)
        breaks = quintile_breaks(susceptibility)
        classes = classify(susceptibility, breaks)
        monthly_classes = classify(means.ravel(), breaks)
        ties = [int((susceptibility == b).sum()) for b in breaks]
        counts = [int((classes == c).sum()) for c in range(1, len(LABELS) + 1)]
        print(f"{model}: susceptibility {susceptibility.min():.4g} to {susceptibility.max():.4g}; "
              f"breaks {', '.join(f'{b:.6g}' for b in breaks)}\n  cells per class "
              f"{', '.join(f'{n:,} ({n / cells.size:.1%})' for n in counts)}; cells equal to a "
              f"break {ties}")

        cell_table[model], cell_table[f"{model}_class"] = susceptibility, classes
        month_table[model], month_table[f"{model}_class"] = means.ravel(), monthly_classes
        year_classes[model] = classes
        class_table += class_rows(model, "year", susceptibility, classes)
        for m in range(MONTHS):
            class_table += class_rows(model, f"month_{m + 1}", means[:, m], monthly_classes.reshape(
                cells.size, MONTHS)[:, m])
        breaks_info[model] = {"breaks": breaks, "cells_per_class": counts,
                              "cells_equal_to_break": ties,
                              "predictions": [str(PRED_DIR / f"{model}_{s}.npy") for s in keys]}

    print("Agreement of the whole-year classes:")
    agreement_rows, n_high = agreement(year_classes)
    cell_table["n_models_high"] = n_high
    if len(MODELS) % 2 == 0:
        raise ValueError("The median class needs an odd number of models (model_training.models).")
    median = np.sort(np.stack(list(year_classes.values())), axis=0)[len(MODELS) // 2]
    cell_table["median_class"] = median
    print("Median class: " + ", ".join(f"{LABELS[c - 1]} {(median == c).mean():.1%}"
                                       for c in range(1, len(LABELS) + 1)))
    month_table["median_class"] = np.sort(month_table[[f"{m}_class" for m in MODELS]].to_numpy(),
                                          axis=1)[:, len(MODELS) // 2]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BREAKS_FILE.write_text(json.dumps({
        "susceptibility": "mean probability of the final model over the GCMs and all months "
                          "of the training years (whole year), per cell",
        "months": list(range(1, MONTHS + 1)), "years": list(cfg.training.years),
        "gcms": list(cfg.gcms.models), "cells": int(cells.size),
        "percentiles": list(cfg.risk_classes.quintiles), "labels": list(LABELS),
        "class_rule": "class = 1 + number of breaks below the value (a value equal to a break "
                      "falls in the lower class)",
        "models": breaks_info}, indent=2))
    cell_table.to_parquet(OUT_DIR / "cells.parquet", index=False)
    month_table.to_parquet(OUT_DIR / "cell_months.parquet", index=False)
    pd.DataFrame(class_table).to_csv(OUT_DIR / "classes.csv", index=False, float_format="%.6g")
    pd.DataFrame(agreement_rows).to_csv(OUT_DIR / "agreement.csv", index=False, float_format="%.6g")
    print(f"\nSaved: {BREAKS_FILE}\n       {OUT_DIR}/cells.parquet, cell_months.parquet, "
          f"classes.csv, agreement.csv")
    report_usage()


if __name__ == "__main__":
    main()
