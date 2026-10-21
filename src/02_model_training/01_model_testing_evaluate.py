"""
01_model_testing_evaluate.py
----------------------------
Scores the test predictions of the candidate algorithms (01_model_testing.py) for every
model whose predictions are complete.

Unit: the real cell-month. The probabilities of the GCM rows of a test cell-month are
averaged, so every cell-month, and every fire, counts once.

Metrics (all months; utils/classification_metrics.py):
- AP (average precision) and ROC-AUC. The no-skill AP equals the fire share.
- The MCC curve: MCC at every distinct probability used as the decision threshold. Each
  model is reported at the peak of its curve: peak MCC, the threshold, and the balanced
  accuracy, precision, recall and confusion matrix there.

Input:  model data of 00_split_data.py (see utils/model_data.py); predictions of 01_model_testing.py
Output: results/<model_training model_testing>/
          metrics.csv                 one row per model
          <model>/threshold_curve.csv thinned for plotting (all thresholds near the peak):
                                      threshold, counts, precision, recall, FPR, MCC,
                                      balanced accuracy

Usage:
    python 01_model_testing_evaluate.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from classification_metrics import plot_points, scores, threshold_curve
from resources import report_usage
import model_data
from models import MODELS

PRED_DIR = cfg.model_training.paths.generated.model_testing
OUT_DIR = cfg.model_training.paths.results.model_testing


def load_predictions(model, n_rows):
    """Test-row probabilities of a model (joining parts), or None if incomplete."""
    whole = PRED_DIR / f"{model}.npy"
    if whole.exists():
        prob = np.load(whole)
    else:
        parts = {}
        for f in PRED_DIR.glob(f"{model}.part*of*.npy"):
            i, n = map(int, re.fullmatch(rf"{model}\.part(\d+)of(\d+)\.npy", f.name).groups())
            parts.setdefault(n, {})[i] = f
        complete = [n for n, found in parts.items() if set(found) == set(range(1, n + 1))]
        if not complete:
            return None
        n = max(complete)
        prob = np.concatenate([np.load(parts[n][i]) for i in range(1, n + 1)])
    if prob.size != n_rows:
        raise ValueError(f"{model}: {prob.size:,} predictions for {n_rows:,} test rows.")
    return prob


def main():
    test = model_data.load_subset("test", arrays=("y", "cell_id", "year", "month", "gcm"))
    rows, y = model_data.cell_month_rows(test)
    print(f"Test: {rows.shape[1]:,} cell-months ({int(y.sum()):,} fire) x {rows.shape[0]} GCMs")

    results = []
    for model in MODELS:
        prob = load_predictions(model, len(test["y"]))
        if prob is None:
            print(f"  {model}: no complete predictions, skipped")
            continue
        p = prob[rows].astype(np.float64).mean(axis=0)
        curve = threshold_curve(y, p)
        s = scores(y, p, curve)
        results.append({"model": model, **s})
        (OUT_DIR / model).mkdir(parents=True, exist_ok=True)
        plot_points(curve).to_csv(OUT_DIR / model / "threshold_curve.csv", index=False)
        print(f"  {model}: peak MCC {s['peak_mcc']:.4f} at {s['threshold']:.4g}, "
              f"AP {s['average_precision']:.4f}, ROC-AUC {s['roc_auc']:.4f}")

    if not results:
        print("No predictions found.")
        return
    table = pd.DataFrame(results)
    table.to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.6g")
    print(f"\nSaved: {OUT_DIR}/metrics.csv and <model>/threshold_curve.csv")
    report_usage()


if __name__ == "__main__":
    main()
