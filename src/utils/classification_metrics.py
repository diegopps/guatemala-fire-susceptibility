"""
classification_metrics.py
-------------------------
Scores of a binary classifier from labels y (0/1) and predicted probabilities p.

    from classification_metrics import threshold_curve, scores, at_threshold, plot_points
    s = scores(y, p)        # dict: AP, ROC-AUC, peak MCC and the confusion matrix there
    c = threshold_curve(y, p)
    t = at_threshold(y, p, 0.2)   # dict: MCC, balanced accuracy, ... at a given threshold

Decision thresholds: a cell is predicted positive if p >= threshold. Candidate
thresholds are all distinct values of p, so the curve covers every operating point the
probabilities allow and no grid has to be chosen. Ties are never split: the counts at a
threshold are exactly those of the rule p >= threshold.

Peak MCC: the maximum of the Matthews correlation coefficient over all thresholds,
reported with its threshold, the balanced accuracy and the confusion matrix there.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_curve(y, p):
    """Counts and metrics at every distinct threshold, from the highest to the lowest.

    Columns: threshold, tp, fp, fn, tn, precision, recall (= TPR), fpr, mcc,
    balanced_accuracy.
    """
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Both classes are needed ({n_pos} positives, {n_neg} negatives).")

    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    tp = np.cumsum(y[order]).astype(np.float64)
    fp = np.arange(1, y.size + 1, dtype=np.float64) - tp
    last_of_tie = np.r_[p_sorted[:-1] != p_sorted[1:], True]
    p_sorted, tp, fp = p_sorted[last_of_tie], tp[last_of_tie], fp[last_of_tie]

    fn, tn = n_pos - tp, n_neg - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)
    recall, fpr = tp / n_pos, fp / n_neg
    return pd.DataFrame({
        "threshold": p_sorted, "tp": tp.astype(np.int64), "fp": fp.astype(np.int64),
        "fn": fn.astype(np.int64), "tn": tn.astype(np.int64),
        "precision": tp / (tp + fp), "recall": recall, "fpr": fpr, "mcc": mcc,
        "balanced_accuracy": (recall + 1 - fpr) / 2,
    })


def scores(y, p, curve=None):
    """AP, ROC-AUC and the peak of the MCC curve (with its threshold and confusion matrix)."""
    if curve is None:
        curve = threshold_curve(y, p)
    peak = curve.loc[curve.mcc.idxmax()]
    n, n_pos = len(y), int(np.sum(y))
    return {
        "n": n, "n_positive": n_pos, "positive_share": n_pos / n,
        "average_precision": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "peak_mcc": peak.mcc, "threshold": peak.threshold,
        "balanced_accuracy": peak.balanced_accuracy,
        "precision": peak.precision, "recall": peak.recall,
        "tp": int(peak.tp), "fp": int(peak.fp), "fn": int(peak.fn), "tn": int(peak.tn),
    }


def at_threshold(y, p, threshold):
    """MCC, balanced accuracy, precision, recall and confusion matrix at a given threshold."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recall = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
            "balanced_accuracy": (recall + tnr) / 2,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def plot_points(curve, n_log=3000, n_lin=1000):
    """A thinned copy of a threshold curve for plotting: points spaced evenly in log rank
    (dense among the highest probabilities, where the peak lies) and in rank, plus the peak."""
    n = len(curve)
    keep = np.unique(np.r_[np.geomspace(1, n, n_log).astype(np.int64) - 1,
                           np.linspace(0, n - 1, n_lin).astype(np.int64),
                           curve.mcc.to_numpy().argmax()])
    return curve.iloc[keep].reset_index(drop=True)
