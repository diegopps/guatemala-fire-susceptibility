"""
06_plot_risk_maps.py
---------------------
Applies pre-computed Jenks natural breaks to predicted probabilities and
produces fire risk classification maps for each final trained model.

For each model this script:
  1. Loads jenks_breaks.json from step 05.
  2. Loads predictions_train.csv, predictions_test.csv, predictions_val.csv.
  3. Applies Jenks breaks to y_prob → integer risk class (1–5).
  4. Calls plot_seasonal_maps() → 2 figures per model per subset
     (dry season + wet season) for the results section.
  5. Calls plot_monthly_climatology() → 12 figures per model per subset
     for the appendix.

Usage:
    python 06_plot_risk_maps.py

No CLI arguments required — runs all three models and all subsets.

Output directory:
    results/06_risk_maps/<MODEL_SHORT>/
      seasonal/    — dry + wet season maps (results section)
      climatology/ — 12 monthly climatology maps (appendix)
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR  = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
RESULTS_DIR  = f"{PROJECT_DIR}/results/04_final_model_training"
OUTPUT_DIR   = f"{PROJECT_DIR}/results/06_risk_maps"

# Models and their short codes — must match 04_final_model_training.py
MODELS = {
    "random_forest": "RF",
    "xgboost":       "XGB",
    "extra_trees":   "XT",
}

# Subsets to map — must match subset labels used in step 04
SUBSETS = ["train", "test", "val"]

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import pandas as pd

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from jenks_breaks import load_breaks, apply_jenks_breaks
from plot_risk_map import plot_seasonal_maps, plot_monthly_climatology

# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():

    for model_key, model_short in MODELS.items():
        section(f"Processing risk maps — {model_short}")

        model_results_dir = os.path.join(RESULTS_DIR, model_short)
        breaks_path       = os.path.join(model_results_dir, "jenks_breaks.json")

        # --- Load Jenks breaks ---
        if not os.path.exists(breaks_path):
            print(f"  ERROR: jenks_breaks.json not found for '{model_short}' "
                  f"at {breaks_path}.")
            print(f"  Run 05_compute_jenks.py first — skipping.")
            continue

        breaks = load_breaks(breaks_path)

        # --- Process each subset ---
        for subset in SUBSETS:
            preds_path = os.path.join(model_results_dir, f"predictions_{subset}.csv")

            if not os.path.exists(preds_path):
                print(f"\n  WARNING: predictions_{subset}.csv not found for "
                      f"'{model_short}' — skipping.")
                continue

            print(f"\n  Loading predictions: {subset} set...")
            preds_df = pd.read_csv(preds_path)

            # Validate required columns
            required_cols = ["lat", "lon", "month", "year", "y_prob"]
            missing = [c for c in required_cols if c not in preds_df.columns]
            if missing:
                print(f"  ERROR: Missing columns {missing} in predictions_{subset}.csv "
                      f"for '{model_short}' — skipping.")
                continue

            if preds_df["y_prob"].isna().all():
                print(f"  ERROR: y_prob is all NaN for '{model_short}' "
                      f"({subset}) — skipping.")
                continue

            print(f"  Loaded {len(preds_df):,} rows.")

            # --- Apply Jenks breaks ---
            print(f"  Applying Jenks breaks...")
            preds_df["risk_class"] = apply_jenks_breaks(
                preds_df["y_prob"].values, breaks
            )

            # Summary of risk class distribution
            counts = preds_df["risk_class"].value_counts().sort_index()
            print(f"  Risk class distribution ({subset}):")
            risk_labels = {1: "very low", 2: "low", 3: "medium",
                           4: "high", 5: "very high"}
            for cls, count in counts.items():
                pct = count / len(preds_df) * 100
                print(f"    Class {cls} ({risk_labels.get(cls, '?'):>9}): "
                      f"{count:>10,} cells ({pct:.2f}%)")

            # --- Output directories ---
            model_label   = f"{model_short}_{subset}"
            seasonal_dir  = os.path.join(OUTPUT_DIR, model_short, subset, "seasonal")
            climatology_dir = os.path.join(OUTPUT_DIR, model_short, subset, "climatology")

            # --- Seasonal maps (results section) ---
            print(f"\n  Plotting seasonal maps ({subset})...")
            plot_seasonal_maps(
                df=preds_df,
                model_label=model_label,
                output_dir=seasonal_dir,
            )

            # --- Monthly climatology maps (appendix) ---
            print(f"\n  Plotting monthly climatology maps ({subset})...")
            plot_monthly_climatology(
                df=preds_df,
                model_label=model_label,
                output_dir=climatology_dir,
            )

    print("\nAll risk maps complete.")


if __name__ == "__main__":
    main()