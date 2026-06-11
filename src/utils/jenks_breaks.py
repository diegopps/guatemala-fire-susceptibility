"""
jenks_breaks.py
----------------
Generic utility for computing and applying Jenks natural breaks to
predicted probability scores, producing a 5-class fire risk classification.

Functions
---------
compute_jenks_breaks(y_prob, n_classes=5)
    Compute Jenks natural breaks on a probability array.
    Returns a list of 4 threshold values defining 5 classes.

apply_jenks_breaks(y_prob, breaks)
    Apply pre-computed Jenks breaks to a probability array.
    Returns an integer array of risk classes (1=very low … 5=very high).

save_breaks(breaks, path)
    Save break values to a JSON file.

load_breaks(path)
    Load break values from a JSON file.

Notes
-----
- Breaks should be computed on the training set predicted probabilities only,
  then applied to all other subsets and future runs to ensure consistency.
- The 4 break values define 5 intervals:
    class 1 (very low)  : y_prob <= breaks[0]
    class 2 (low)       : breaks[0] < y_prob <= breaks[1]
    class 3 (medium)    : breaks[1] < y_prob <= breaks[2]
    class 4 (high)      : breaks[2] < y_prob <= breaks[3]
    class 5 (very high) : y_prob > breaks[3]
"""

import os
import json
import numpy as np
import jenkspy

# =============================================================================
# RISK CLASS LABELS
# =============================================================================

RISK_LABELS = {
    1: "very low",
    2: "low",
    3: "medium",
    4: "high",
    5: "very high",
}

N_CLASSES = 5


# =============================================================================
# FUNCTIONS
# =============================================================================

def compute_jenks_breaks(y_prob, n_classes=N_CLASSES):
    """
    Compute Jenks natural breaks on predicted probability scores.

    Parameters
    ----------
    y_prob    : array-like of float — predicted probabilities (0–1)
    n_classes : int — number of classes (default: 5)

    Returns
    -------
    breaks : list of float — (n_classes - 1) threshold values defining
             the boundaries between classes, in ascending order.
             e.g. [0.021, 0.085, 0.213, 0.487] for 5 classes.
    """
    y_prob = np.array(y_prob, dtype=float)

    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN values. Remove or impute before computing breaks.")

    if len(y_prob) == 0:
        raise ValueError("y_prob is empty.")

    print(f"  Computing Jenks natural breaks on {len(y_prob):,} probability scores "
          f"({n_classes} classes)...")

    # jenkspy expects a list and returns n_classes + 1 values
    # (including min and max), so we strip those to get the inner breaks only
    all_breaks = jenkspy.jenks_breaks(y_prob.tolist(), n_classes=n_classes)
    breaks     = all_breaks[1:-1]  # drop min and max, keep inner boundaries

    print(f"  Jenks breaks computed:")
    print(f"    Class 1 (very low)  : prob <= {breaks[0]:.6f}")
    print(f"    Class 2 (low)       : {breaks[0]:.6f} < prob <= {breaks[1]:.6f}")
    print(f"    Class 3 (medium)    : {breaks[1]:.6f} < prob <= {breaks[2]:.6f}")
    print(f"    Class 4 (high)      : {breaks[2]:.6f} < prob <= {breaks[3]:.6f}")
    print(f"    Class 5 (very high) : prob > {breaks[3]:.6f}")

    return breaks


def apply_jenks_breaks(y_prob, breaks):
    """
    Apply pre-computed Jenks breaks to predicted probability scores.

    Parameters
    ----------
    y_prob  : array-like of float — predicted probabilities (0–1)
    breaks  : list of float — 4 threshold values from compute_jenks_breaks()

    Returns
    -------
    risk_class : np.ndarray of int — integer risk class per cell (1–5)
    """
    y_prob     = np.array(y_prob, dtype=float)
    risk_class = np.ones(len(y_prob), dtype=int)  # start all at class 1

    for i, threshold in enumerate(breaks):
        risk_class[y_prob > threshold] = i + 2  # classes 2, 3, 4, 5

    return risk_class


def save_breaks(breaks, path):
    """
    Save Jenks break values to a JSON file.

    Parameters
    ----------
    breaks : list of float — 4 threshold values
    path   : str — full output file path (e.g. .../jenks_breaks.json)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "n_classes":   N_CLASSES,
        "breaks":      breaks,
        "class_labels": RISK_LABELS,
        "description": (
            "Jenks natural breaks computed on training set predicted probabilities. "
            "Apply to all subsets and future runs for consistent risk classification."
        ),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Jenks breaks saved to: {path}")


def load_breaks(path):
    """
    Load Jenks break values from a JSON file.

    Parameters
    ----------
    path : str — full path to jenks_breaks.json

    Returns
    -------
    breaks : list of float — 4 threshold values
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Jenks breaks file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    breaks = data["breaks"]
    print(f"  Loaded Jenks breaks from: {path}")
    print(f"    Breaks: {breaks}")

    return breaks