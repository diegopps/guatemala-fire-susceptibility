"""
02_resampling_evaluate.py
-------------------------
Scores the test predictions of the selected algorithms trained with each sampler
(02_resampling.py), next to the reference: the same algorithm with balanced class weights
and no sampler, i.e. its result in the model testing (01_model_testing_evaluate.py).

Scoring as in the model testing (utils/classification_metrics.py): per real test
cell-month, on the mean probability of its GCM rows; AP, ROC-AUC and the MCC curve over all
thresholds, reported at its peak (peak MCC, threshold, and the balanced accuracy,
precision, recall and confusion matrix there).

Input:  model data of 00_split_data.py (see utils/model_data.py); predictions of 02_resampling.py;
        results/<model_training model_testing>/metrics.csv (reference)
Output: results/<model_training resampling>/
          metrics.csv                       one row per algorithm and sampler, plus the
                                            reference rows (sampler "class_weights")
          ranking.csv                       per algorithm, the samplers ranked by peak MCC,
                                            with the reference's peak MCC for comparison
          <model>/threshold_curve_<sampler>.csv  thinned for plotting (all thresholds near
                                            the peak)

Usage:
    python 02_resampling_evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from classification_metrics import plot_points, scores, threshold_curve
from resources import report_usage
import model_data
import resampling

MODELS = cfg.model_training.models
REFERENCE = "class_weights"
PRED_DIR = cfg.model_training.paths.generated.resampling
OUT_DIR = cfg.model_training.paths.results.resampling
REFERENCE_METRICS = cfg.model_training.paths.results.model_testing / "metrics.csv"


def main():
    test = model_data.load_subset("test", arrays=("y", "cell_id", "year", "month", "gcm"))
    rows, y = model_data.cell_month_rows(test)
    print(f"Test: {rows.shape[1]:,} cell-months ({int(y.sum()):,} fire) x {rows.shape[0]} GCMs")

    reference = pd.read_csv(REFERENCE_METRICS).set_index("model")
    results = []
    for model in MODELS:
        results.append({"model": model, "sampler": REFERENCE, **reference.loc[model].to_dict()})
        for sampler in resampling.SAMPLERS:
            path = PRED_DIR / model / f"{sampler}.npy"
            if not path.exists():
                print(f"  {model} / {sampler}: no predictions, skipped")
                continue
            prob = np.load(path)
            if prob.size != len(test["y"]):
                raise ValueError(f"{path}: {prob.size:,} predictions for {len(test['y']):,} test rows.")
            p = prob[rows].astype(np.float64).mean(axis=0)
            curve = threshold_curve(y, p)
            s = scores(y, p, curve)
            results.append({"model": model, "sampler": sampler, **s})
            (OUT_DIR / model).mkdir(parents=True, exist_ok=True)
            plot_points(curve).to_csv(OUT_DIR / model / f"threshold_curve_{sampler}.csv", index=False)
            print(f"  {model} / {sampler}: peak MCC {s['peak_mcc']:.4f} at {s['threshold']:.4g}, "
                  f"AP {s['average_precision']:.4f}, ROC-AUC {s['roc_auc']:.4f}")

    table = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.6g")

    samplers = table[table.sampler != REFERENCE].copy()
    if samplers.empty:
        print("No predictions found.")
        return
    samplers["rank"] = samplers.groupby("model").peak_mcc.rank(ascending=False, method="min").astype(int)
    ref_mcc = table[table.sampler == REFERENCE].set_index("model").peak_mcc
    samplers["reference_peak_mcc"] = samplers.model.map(ref_mcc)
    ranking = (samplers.sort_values(["model", "rank"])
               [["model", "rank", "sampler", "peak_mcc", "reference_peak_mcc", "average_precision", "roc_auc"]])
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False, float_format="%.6g")
    print("\nBest sampler per model (peak MCC; reference = balanced class weights):")
    for model, r in ranking[ranking["rank"] == 1].set_index("model").iterrows():
        print(f"  {model}: {r.sampler} {r.peak_mcc:.4f} (reference {r.reference_peak_mcc:.4f})")
    print(f"\nSaved: {OUT_DIR}/metrics.csv, ranking.csv and <model>/threshold_curve_<sampler>.csv")
    report_usage()


if __name__ == "__main__":
    main()
