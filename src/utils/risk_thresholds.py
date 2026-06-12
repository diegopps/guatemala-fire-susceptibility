"""
risk_thresholds.py
------------------
Compute and apply FROZEN fire-risk classification thresholds based on the
prediction/success-rate curve (Chung & Fabbri, 2003), as an alternative to
Jenks natural breaks that is invariant to the per-model probability scale
induced by over/under-sampling.

Idea
----
Rank cells by predicted susceptibility (descending) and place class boundaries
where the ranked cells cumulatively capture target fractions of OBSERVED fires
("capture milestones"). Because only the ranking of cells enters, the boundaries
are invariant to monotonic rescaling of the scores (calibration, RUS inflation,
NCL compression). The boundaries are then stored as fixed PROBABILITY thresholds
so the exact same standard can be applied to every map — present and future —
which is what lets class areas grow/shrink between scenarios (the basis for
temporal comparison). Future projections have no observed fires, so the standard
must be frozen from a reference period.

The thresholds should be derived from HELD-OUT data (test + validation), never
training: training probabilities are in-sample and overconfident, which biases
the thresholds (markedly so for calibrated models such as XGBoost).

Functions
---------
compute_risk_thresholds(susc, fires, milestones) -> dict
    Fit thresholds on reference per-cell susceptibility + fire counts.
apply_thresholds(y_prob, breaks) -> np.ndarray of int (1..5)
    Classify probabilities with frozen breaks.
save_thresholds(data, path) / load_thresholds(path) -> dict

Reference
---------
Chung, C.-J. F., & Fabbri, A. G. (2003). Validation of spatial prediction models
for landslide hazard mapping. Natural Hazards, 30(3), 451-472.
"""

import os
import json
import numpy as np

RISK_LABELS = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}

# Default cumulative fire-capture milestones marking the boundaries
# very high / high / medium / low (very low is the remainder).
DEFAULT_MILESTONES = [0.50, 0.75, 0.90, 0.99]

_TRAPZ = getattr(np, "trapezoid", np.trapz)   # np.trapz deprecated in newer numpy


def apply_thresholds(y_prob, breaks):
    """
    Classify probabilities into risk classes 1..5 using ascending `breaks`
    (4 values). class = 1 + number of breaks the probability exceeds, so
    prob > breaks[-1] -> class 5 (very high), prob <= breaks[0] -> class 1.
    """
    y   = np.asarray(y_prob, dtype=float)
    cls = np.ones(len(y), dtype=int)
    for i, b in enumerate(sorted(breaks)):
        cls[y > b] = i + 2
    return cls


def compute_risk_thresholds(susc, fires, milestones=DEFAULT_MILESTONES):
    """
    Derive frozen probability thresholds from a reference set.

    Parameters
    ----------
    susc      : array-like of float — per-cell susceptibility (mean predicted
                probability) on the reference (held-out) data.
    fires     : array-like of float/int — observed fire count per cell, aligned
                with `susc`.
    milestones: list of float (ascending) — cumulative fire-capture fractions at
                the very-high / high / medium / low boundaries.

    Returns
    -------
    dict with keys: breaks (ascending list of 4 thresholds, for apply_thresholds),
    capture_milestones, prediction_rate_auc, n_cells, n_fires, class_labels,
    class_summary (per-class area% and fire-capture%).
    """
    susc  = np.asarray(susc, dtype=float)
    fires = np.asarray(fires, dtype=float)
    n     = len(susc)

    order      = np.argsort(-susc, kind="mergesort")   # highest susceptibility first
    susc_s     = susc[order]
    fires_s    = fires[order]
    total      = fires_s.sum()
    if total <= 0:
        raise ValueError("Reference data contains no fires; cannot anchor thresholds.")

    cum_capture = np.cumsum(fires_s) / total           # y of the prediction-rate curve
    cum_area    = np.arange(1, n + 1) / n              # x of the curve

    # Probability value at each capture milestone (susc of the boundary cell).
    breaks = []
    for m in milestones:
        i = min(int(np.searchsorted(cum_capture, m)), n - 1)
        breaks.append(float(susc_s[i]))
    breaks = sorted(breaks)                             # ascending: [L, M, H, VH]

    auc = float(_TRAPZ(cum_capture, cum_area))

    cls = apply_thresholds(susc, breaks)
    summary = []
    for c in [5, 4, 3, 2, 1]:
        mask = cls == c
        summary.append({
            "class":       c,
            "label":       RISK_LABELS[c],
            "area_pct":    round(100 * mask.sum() / n, 2),
            "capture_pct": round(100 * fires[mask].sum() / total, 2),
            "n_cells":     int(mask.sum()),
        })

    return {
        "breaks":              breaks,
        "capture_milestones":  list(milestones),
        "prediction_rate_auc": round(auc, 4),
        "n_cells":             int(n),
        "n_fires":             int(total),
        "class_labels":        RISK_LABELS,
        "class_summary":       summary,
    }


def save_thresholds(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def load_thresholds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Risk thresholds file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Loaded risk thresholds from: {path}")
    print(f"    Breaks: {data['breaks']}")
    return data
