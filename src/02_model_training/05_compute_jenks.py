"""
05_compute_jenks.py
--------------------
Computes Jenks natural breaks on the training set predicted probabilities
for each final trained model (RF, XGB, XT) and saves the break values
as jenks_breaks.json under each model's results directory.

Breaks are computed on the training set only, then applied in step 06
to all subsets (test, val) and future runs to ensure consistent risk
classification across scenarios.

Usage:
    python 05_compute_jenks.py

No CLI arguments required — runs all three models in sequence.

Output:
    results/04_final_model_training/<MODEL_SHORT>/jenks_breaks.json
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

N_CLASSES = 5

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import pandas as pd

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from jenks_breaks import compute_jenks_breaks, save_breaks

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
        section(f"Computing Jenks breaks — {model_short}")

        model_dir  = os.path.join(RESULTS_DIR, model_short)
        preds_path = os.path.join(model_dir, "predictions_train.csv")
        breaks_path = os.path.join(model_dir, "jenks_breaks.json")

        # --- Check predictions file exists ---
        if not os.path.exists(preds_path):
            print(f"  WARNING: predictions_train.csv not found for '{model_short}' "
                  f"at {preds_path} — skipping.")
            continue

        # --- Check if breaks already exist ---
        if os.path.exists(breaks_path):
            print(f"  NOTE: jenks_breaks.json already exists for '{model_short}' "
                  f"— skipping recomputation.")
            print(f"  Delete {breaks_path} to force recomputation.")
            continue

        # --- Load training predictions ---
        print(f"  Loading training predictions from: {preds_path}")
        preds_df = pd.read_csv(preds_path)

        if "y_prob" not in preds_df.columns:
            print(f"  ERROR: y_prob column not found in predictions_train.csv "
                  f"for '{model_short}' — skipping.")
            continue

        if preds_df["y_prob"].isna().all():
            print(f"  ERROR: y_prob is all NaN for '{model_short}' — skipping.")
            continue

        y_prob = preds_df["y_prob"].dropna().values
        print(f"  Loaded {len(y_prob):,} probability scores.")

        # --- Compute Jenks breaks ---
        breaks = compute_jenks_breaks(y_prob, n_classes=N_CLASSES)

        # --- Save breaks ---
        save_breaks(breaks, breaks_path)

    print("\nAll Jenks breaks computed and saved.")


if __name__ == "__main__":
    main()