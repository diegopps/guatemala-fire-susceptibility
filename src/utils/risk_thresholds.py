"""
risk_thresholds.py
------------------
Compute and apply FROZEN fire-risk classification thresholds using the
area-quintile method (Verde & Zêzere, 2010): cells are divided into five equal-
area bands by susceptibility rank (20% of cells per class), so class boundaries
are the 20th, 40th, 60th, and 80th percentiles of the predicted-probability
distribution.

Because only the RANKING of cells enters, the breaks are invariant to monotonic
rescaling of the scores (calibration, RUS inflation, NCL compression). Storing
the resulting PROBABILITY thresholds lets the exact same standard be applied to
every map — present and future — which is what lets class areas grow/shrink
between scenarios (the basis for temporal comparison). Future projections have no
observed fires, so the standard must be frozen from a reference period.

Thresholds are derived from HELD-OUT data (test + validation), never training:
training probabilities are in-sample and overconfident, which biases the
thresholds (markedly so for calibrated models such as XGBoost).

A prediction-rate curve (ranked cumulative fire capture vs cumulative area) is
computed alongside the quintile breaks — it is retained as a diagnostic and
used in the step-06 figure that visually justifies the quintile choice.

Functions
---------
compute_quintile_thresholds(susc, fires) -> dict
    Fit quintile thresholds on reference per-cell susceptibility + fire counts.
    Returns breaks, per-class summary, and the full prediction-rate curve data.
apply_thresholds(y_prob, breaks) -> np.ndarray of int (1..5)
    Classify probabilities with frozen breaks.
save_thresholds(data, path) / load_thresholds(path) -> dict

Reference
---------
Verde, A., & Zêzere, J. L. (2010). Assessment and validation of wildfire
susceptibility and hazard in Portugal. Natural Hazards and Earth System Sciences,
10(3), 485–497. https://doi.org/10.5194/nhess-10-485-2010
"""

import os
import json
import numpy as np

RISK_LABELS = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}

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


def compute_quintile_thresholds(susc, fires):
    """
    Derive frozen probability thresholds by splitting cells into equal-area
    quintiles (each ~20% of cells by susceptibility rank), following Verde &
    Zêzere (2010).

    The four break-points are the 20th, 40th, 60th, and 80th percentiles of
    the susceptibility distribution on the reference (held-out) data. Class
    areas are equal by construction; observed fire capture per class is reported
    as an interpretive diagnostic (Verde & Zêzere, 2010).

    A prediction-rate curve is also computed: cells ranked by descending
    susceptibility, cumulative fraction of fires captured (y) vs cumulative
    fraction of area (x). This curve is stored in the output dict so that step
    06 can plot it without reloading the raw predictions.

    Parameters
    ----------
    susc  : array-like of float — per-cell susceptibility (mean predicted prob)
            on the reference (held-out) data.
    fires : array-like of float/int — observed fire count per cell, aligned
            with `susc`.

    Returns
    -------
    dict with keys:
        breaks               : list of 4 ascending probability thresholds
        quintile_percentiles : [20, 40, 60, 80]
        prediction_rate_auc  : float — area under the prediction-rate curve
        n_cells              : int
        n_fires              : int
        class_labels         : {1..5 -> label str}
        class_summary        : list of per-class dicts (class, label, area_pct,
                               capture_pct, n_cells)
        curve_area           : list — cumulative fraction of area (x axis)
        curve_capture        : list — cumulative fraction of fires captured (y axis)
    """
    susc  = np.asarray(susc,  dtype=float)
    fires = np.asarray(fires, dtype=float)
    n     = len(susc)
    total = fires.sum()

    if total <= 0:
        raise ValueError("Reference data contains no fires; cannot compute fire-capture diagnostics.")

    # --- Quintile break-points (area-equal by construction) ---
    breaks = [float(np.percentile(susc, p)) for p in [20, 40, 60, 80]]

    # --- Prediction-rate curve (diagnostic; not used to place breaks) ---
    order       = np.argsort(-susc, kind="mergesort")   # descending susceptibility
    fires_s     = fires[order]
    cum_capture = np.cumsum(fires_s) / total
    cum_area    = np.arange(1, n + 1) / n

    auc = float(_TRAPZ(cum_capture, cum_area))

    # --- Per-class summary ---
    cls     = apply_thresholds(susc, breaks)
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
        "breaks":                breaks,
        "quintile_percentiles":  [20, 40, 60, 80],
        "prediction_rate_auc":   round(auc, 4),
        "n_cells":               int(n),
        "n_fires":               int(total),
        "class_labels":          RISK_LABELS,
        "class_summary":         summary,
        "curve_area":            cum_area.tolist(),
        "curve_capture":         cum_capture.tolist(),
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
