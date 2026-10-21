"""
03_hyperparameter_tuning_evaluate.py
------------------------------------
Collects the CV scores of the hyperparameter tuning (03_hyperparameter_tuning.py) and
compares the tuned and the base settings of each selected algorithm on the test set.

Base = configuration 0 (the settings of utils/models.py) with the imbalance correction of the
model (model_training.samplers); it is the model of step 02 (or of step 01, with class
weights), so its test scores are read from 02_resampling/metrics.csv. Tuned = the selected
configuration refitted on the whole train subset (03_hyperparameter_tuning.py refit); if
configuration 0 is selected, tuned = base.

Test scoring as in steps 01-02 (utils/classification_metrics.py): per real test cell-month,
on the mean probability of its GCM rows; AP, ROC-AUC and the MCC curve over all thresholds,
reported at its peak (peak MCC, threshold, and the balanced accuracy, precision, recall and
confusion matrix there).

Can run at any time: it summarises the fits scored so far (configurations not yet scored on
every fold get no rank) and reports models without a refit as such.

Input:  output of 03_hyperparameter_tuning.py; model data of 00_split_data.py (test labels);
        results/<model_training resampling>/metrics.csv (base)
Output: results/<model_training hyperparameter_tuning>/
          cv_results.csv              one row per model, configuration and fold scored
          <model>/cv_summary.csv      per configuration: search-space parameters, folds
                                      scored, mean and standard deviation of the scores over
                                      the folds, rank by mean peak MCC
          metrics.csv                 per model, base and tuned: configuration, CV peak MCC
                                      (mean and standard deviation over the folds), test scores
          <model>/threshold_curve_tuned.csv  thinned for plotting (all thresholds near the peak)

--trial evaluates the trial runs (<...>_trial folders); only the cell-months they predicted
are scored, and the base test scores are left empty.

Usage:
    python 03_hyperparameter_tuning_evaluate.py [--trial]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from classification_metrics import plot_points, scores, threshold_curve
from resources import report_usage
import model_data
import tuning

MODELS = cfg.model_training.models
N_FOLDS = cfg.model_training.tuning.folds
BASE_METRICS = cfg.model_training.paths.results.resampling / "metrics.csv"


def test_cell_months():
    test = model_data.load_subset("test", arrays=("y", "cell_id", "year", "month", "gcm"))
    rows, y = model_data.cell_month_rows(test)
    print(f"Test: {rows.shape[1]:,} cell-months ({int(y.sum()):,} fire) x {rows.shape[0]} GCMs")
    return rows, y, len(test["y"])


def main(trial):
    gen, out_dir = tuning.directories(trial)
    base_metrics = None
    if not trial:
        if BASE_METRICS.exists():
            base_metrics = pd.read_csv(BASE_METRICS).set_index(["model", "sampler"])
        else:
            print(f"{BASE_METRICS} not found: base test scores left empty.")
    test = None
    cv_tables, results = [], []
    for model in MODELS:
        cv = tuning.cv_scores(model, trial)
        if cv.empty:
            print(f"{model}: no fits scored yet")
            continue
        summary = tuning.cv_summary(model, cv)
        (out_dir / model).mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / model / "cv_summary.csv", index=False, float_format="%.6g")
        cv_tables.append(cv.assign(params=cv.params.map(json.dumps)))
        complete = int((summary.folds_scored == N_FOLDS).sum())
        print(f"{model}: {len(cv):,} of {len(summary) * N_FOLDS:,} fits scored; "
              f"{complete} of {len(summary)} configurations on all {N_FOLDS} folds")
        if complete:
            top = summary.sort_values("rank").iloc[0]
            print(f"  highest mean CV peak MCC so far: configuration {int(top.config)} "
                  f"{top.peak_mcc_mean:.4f} (defaults: {summary.peak_mcc_mean.iloc[0]:.4f})")

        sampler = tuning.sampler(model)
        cv_by_config = summary.set_index("config")
        base = {"model": model, "setting": "base", "sampler": sampler, "config": 0,
                "cv_peak_mcc_mean": cv_by_config.peak_mcc_mean.get(0),
                "cv_peak_mcc_std": cv_by_config.peak_mcc_std.get(0)}
        if base_metrics is not None and (model, sampler) in base_metrics.index:
            base.update(base_metrics.loc[(model, sampler)].to_dict())
        elif not trial:
            print(f"  no step-02 test scores for {model} / {sampler}: base test scores left empty")
        results.append(base)

        selection_file = out_dir / model / "selection.json"
        if not selection_file.exists():
            print("  not refitted yet")
            continue
        config = json.loads(selection_file.read_text())["config"]
        tuned = {"model": model, "setting": "tuned", "sampler": sampler, "config": config,
                 "cv_peak_mcc_mean": cv_by_config.peak_mcc_mean[config],
                 "cv_peak_mcc_std": cv_by_config.peak_mcc_std[config]}
        if config == 0:
            tuned.update({k: v for k, v in base.items() if k not in tuned})
            print("  defaults selected: tuned = base")
        else:
            path = gen / f"{model}.npy"
            if not path.exists():
                print(f"  {path} not found: tuned test scores left empty")
                results.append(tuned)
                continue
            if test is None:
                test = test_cell_months()
            rows, y, n_rows = test
            prob = np.load(path)
            if prob.size != n_rows:
                raise ValueError(f"{path}: {prob.size:,} predictions for {n_rows:,} test rows.")
            P = prob[rows].astype(np.float64)
            scored = np.isfinite(P).all(axis=0)          # trial runs predict some cell-months only
            p, y_scored = P[:, scored].mean(axis=0), y[scored]
            curve = threshold_curve(y_scored, p)
            tuned.update(scores(y_scored, p, curve))
            plot_points(curve).to_csv(out_dir / model / "threshold_curve_tuned.csv", index=False)
            print(f"  tuned (configuration {config}): test peak MCC {tuned['peak_mcc']:.4f}, "
                  f"AP {tuned['average_precision']:.4f}, ROC-AUC {tuned['roc_auc']:.4f}"
                  + (f"; base {base['peak_mcc']:.4f}" if "peak_mcc" in base else ""))
        results.append(tuned)

    if not cv_tables:
        print("No fits scored.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(cv_tables).to_csv(out_dir / "cv_results.csv", index=False, float_format="%.6g")
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False, float_format="%.6g")
    print(f"\nSaved: {out_dir}/cv_results.csv, metrics.csv and <model>/cv_summary.csv")
    report_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trial", action="store_true", help="evaluate the trial runs")
    main(parser.parse_args().trial)
