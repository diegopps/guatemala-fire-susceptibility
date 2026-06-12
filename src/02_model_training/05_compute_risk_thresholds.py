"""
05_compute_risk_thresholds.py
-----------------------------
Computes FROZEN fire-risk classification thresholds for each final model and
saves them as risk_thresholds.json under each model's results directory.

Method (replaces Jenks natural breaks, which was distorted by the per-model
probability scale from over/under-sampling): cells are ranked by predicted
susceptibility and class boundaries are placed where the ranked cells
cumulatively capture target fractions of observed fires (the prediction-rate
curve; Chung & Fabbri, 2003). The resulting probability thresholds are frozen
and applied unchanged to every map in step 06 — and to future scenarios later —
so class areas can grow/shrink between scenarios.

Thresholds are derived from the HELD-OUT data (test + validation), NOT training:
training probabilities are in-sample and overconfident, which biases the
thresholds (markedly for calibrated models such as XGBoost). The reference
quantity is the per-cell mean predicted probability over the dry (fire) season,
matching what step 06 maps.

Usage:
    python 05_compute_risk_thresholds.py

No CLI arguments required — runs all three models in sequence.

Output:
    results/04_final_model_training/<MODEL_SHORT>/risk_thresholds.json
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR = f"{PROJECT_DIR}/results/04_final_model_training"

# Models and their short codes — must match 04_final_model_training.py
MODELS = {
    "random_forest": "RF",
    "xgboost":       "XGB",
    "extra_trees":   "XT",
}

# Held-out subsets the thresholds are derived from (NOT train — see module docs).
REFERENCE_SUBSETS = ["test", "val"]

# Cumulative fire-capture milestones for the VH / H / M / L boundaries.
CAPTURE_MILESTONES = [0.50, 0.75, 0.90, 0.99]

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import pandas as pd

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from risk_thresholds import compute_risk_thresholds, save_thresholds
from plot_risk_map import DRY_MONTHS

# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def load_reference_cells(model_short):
    """
    Per-cell dry-season susceptibility (mean prob) and fire count over the
    held-out reference subsets. Returns a DataFrame (susc, fires) or None.
    """
    model_dir = os.path.join(RESULTS_DIR, model_short)
    frames = []
    for subset in REFERENCE_SUBSETS:
        path = os.path.join(model_dir, f"predictions_{subset}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: predictions_{subset}.csv not found for "
                  f"'{model_short}' — skipping that subset.")
            continue
        frames.append(pd.read_csv(path, usecols=["lat", "lon", "month", "y_true", "y_prob"]))
        print(f"  Loaded {subset} predictions.")
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True).dropna(subset=["y_prob"])
    df = df[df["month"].isin(DRY_MONTHS)]
    cell = df.groupby(["lat", "lon"], as_index=False).agg(
        susc=("y_prob", "mean"), fires=("y_true", "sum"))
    return cell


# =============================================================================
# MAIN
# =============================================================================

def main():
    for model_key, model_short in MODELS.items():
        section(f"Computing risk thresholds — {model_short}")

        model_dir   = os.path.join(RESULTS_DIR, model_short)
        out_path    = os.path.join(model_dir, "risk_thresholds.json")

        if os.path.exists(out_path):
            print(f"  NOTE: risk_thresholds.json already exists for '{model_short}' "
                  f"— skipping recomputation.")
            print(f"  Delete {out_path} to force recomputation.")
            continue

        cell = load_reference_cells(model_short)
        if cell is None or cell.empty:
            print(f"  ERROR: no reference predictions for '{model_short}' — skipping.")
            continue

        print(f"  Reference: {REFERENCE_SUBSETS} (held-out), dry season "
              f"({len(cell):,} cells, {int(cell['fires'].sum()):,} fire events)")

        data = compute_risk_thresholds(
            cell["susc"].to_numpy(),
            cell["fires"].to_numpy(),
            milestones=CAPTURE_MILESTONES,
        )
        # Record provenance alongside the thresholds.
        data = {
            "model":              model_key,
            "model_short":        model_short,
            "reference_subsets":  REFERENCE_SUBSETS,
            "reference_window":   "dry season (per-cell mean probability)",
            "description": ("Frozen prediction-rate fire-risk thresholds. Apply "
                            "unchanged to all maps and future scenarios. Derived "
                            "from held-out data; see risk_thresholds.py."),
            **data,
        }

        print(f"  Prediction-rate AUC: {data['prediction_rate_auc']}")
        print(f"  Breaks (ascending) : {[round(b, 6) for b in data['breaks']]}")
        print(f"  {'class':<11}{'area %':>9}{'fires %':>10}")
        for row in data["class_summary"]:
            print(f"  {row['label']:<11}{row['area_pct']:>8.1f}%{row['capture_pct']:>9.1f}%")

        save_thresholds(data, out_path)

    print("\nAll risk thresholds computed and saved.")


if __name__ == "__main__":
    main()
