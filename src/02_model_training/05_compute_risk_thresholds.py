"""
05_compute_risk_thresholds.py
-----------------------------
Computes FROZEN fire-risk classification thresholds for each final model and
saves them as risk_thresholds.json under each model's results directory.

Method (Verde & Zêzere, 2010): cells are divided into five equal-area quintiles
by susceptibility rank (20% of cells per class). The four break-points are the
20th, 40th, 60th, and 80th percentiles of the predicted-probability distribution
on the reference (held-out) data. Because only the RANKING of cells enters, the
breaks are invariant to monotonic rescaling of the scores (calibration, RUS
inflation, NCL compression). The resulting probability thresholds are then FROZEN
and applied unchanged to every map in step 06 — and to future scenarios later —
so risk-class areas can grow/shrink between scenarios (the basis for temporal
comparison).

Thresholds are derived from the HELD-OUT data (test + validation), NOT training:
training probabilities are in-sample and overconfident, which biases the
thresholds (markedly for calibrated models such as XGBoost). The reference
quantity is the per-cell mean predicted probability over the dry (fire) season,
matching what step 06 maps.

A prediction-rate curve (cumulative fire capture vs cumulative area, ranked by
descending susceptibility) is stored alongside the breaks — it is retained as a
diagnostic and plotted in step 06 to visually justify the quintile choice.

Usage:
    python 05_compute_risk_thresholds.py

No CLI arguments required — runs all three models in sequence.

Output:
    results/04_final_model_training/<MODEL_SHORT>/risk_thresholds.json

Reference:
    Verde, A., & Zêzere, J. L. (2010). Assessment and validation of wildfire
    susceptibility and hazard in Portugal. Natural Hazards and Earth System
    Sciences, 10(3), 485–497. https://doi.org/10.5194/nhess-10-485-2010
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

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import pandas as pd

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from risk_thresholds import compute_quintile_thresholds, save_thresholds
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

        model_dir = os.path.join(RESULTS_DIR, model_short)
        out_path  = os.path.join(model_dir, "risk_thresholds.json")

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

        data = compute_quintile_thresholds(
            cell["susc"].to_numpy(),
            cell["fires"].to_numpy(),
        )

        # Record provenance alongside the thresholds.
        data = {
            "model":             model_key,
            "model_short":       model_short,
            "reference_subsets": REFERENCE_SUBSETS,
            "reference_window":  "dry season (per-cell mean probability)",
            "method":            "area-quintile (Verde & Zêzere, 2010)",
            "description": (
                "Frozen area-quintile fire-risk thresholds (Verde & Zêzere, 2010). "
                "Break-points are the 20/40/60/80th percentiles of per-cell mean "
                "susceptibility on the held-out (test+val) dry-season data. Apply "
                "unchanged to all maps and future scenarios; never recompute per "
                "period (that erases the change signal)."
            ),
            **data,
        }

        # --- Sanity checks (printed; do not raise — let the user inspect) ---
        breaks = data["breaks"]
        print(f"\n  Sanity checks:")
        print(f"    Breaks (ascending) : {[round(b, 6) for b in breaks]}")
        strictly_asc = all(breaks[i] < breaks[i+1] for i in range(len(breaks)-1))
        print(f"    Strictly ascending : {'YES' if strictly_asc else 'WARNING — collapsed breaks!'}")
        print(f"    Prediction-rate AUC: {data['prediction_rate_auc']}")
        print(f"\n  {'class':<11}{'area %':>9}{'fires %':>10}")
        for row in data["class_summary"]:
            area_ok = abs(row['area_pct'] - 20.0) < 1.5
            print(f"  {row['label']:<11}{row['area_pct']:>8.1f}%{row['capture_pct']:>9.1f}%"
                  f"  {'(OK)' if area_ok else '(WARNING: area != ~20%)'}")

        save_thresholds(data, out_path)

    print("\nAll risk thresholds computed and saved.")
    print("\nPost-run checks to verify correctness:")
    print("  1. Baseline class shares should be ~20% each (quintile sanity).")
    print("  2. Breaks must be strictly ascending (collapsed = probability skew).")
    print("  3. Future class shares should depart from 20% (freezing is working).")


if __name__ == "__main__":
    main()
