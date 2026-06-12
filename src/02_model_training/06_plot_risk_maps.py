"""
06_plot_risk_maps.py
---------------------
Applies pre-computed Jenks natural breaks to predicted probabilities and
produces fire risk classification maps for the final trained models.

Two figure types are produced:

  1. Fire-season comparison (main text):
       A single figure with one panel per model (RF, XGB, XT), each showing the
       dry-season (Nov–Apr, the official fire season) risk climatology. Panels
       are lettered A), B), C) for cross-referencing.
         -> results/06_risk_maps/fire_season_comparison.{png,pdf,svg}

  2. Monthly climatology (appendix):
       One 3x4 panel figure per model, showing the 12 calendar-month
       climatologies.
         -> results/06_risk_maps/<MODEL_SHORT>/<MODEL_SHORT>_monthly_climatology.{...}

Data
----
Maps are built from ALL prediction subsets combined (train + test + val), giving
complete spatial coverage and every cell's full month history. This is a
visualization of the model's risk surface — quantitative performance is reported
separately on the held-out test/validation sets (steps 02–04). The risk classes
come from the frozen prediction-rate thresholds fit on HELD-OUT data (step 05),
so the classification scheme is not biased by in-sample training probabilities.

Aggregation
-----------
For each cell, the predicted probability is averaged over the relevant months
(and all years), then the frozen risk thresholds (step 05) are applied to that
mean probability to assign a risk class. Probabilities are averaged (not class
labels), since the classes are unequal-width bins. The same fixed thresholds are
used for every map, so seasons/months (and, later, future scenarios) are
directly comparable.

Usage:
    python 06_plot_risk_maps.py

No CLI arguments required — runs all three models.
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

# Display names used in panel titles
MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "extra_trees":   "Extra Trees",
}

# Subsets combined to form the full prediction set per model
SUBSETS = ["train", "test", "val"]

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import pandas as pd

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from risk_thresholds import load_thresholds, apply_thresholds
from plot_risk_map import plot_risk_map_grid, DRY_MONTHS, MONTH_NAMES

# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def load_all_predictions(model_short):
    """
    Load and concatenate train/test/val predictions for one model into the full
    prediction set. Returns a DataFrame with columns lat, lon, month, y_prob
    (rows with NaN probabilities dropped), or None if nothing could be loaded.
    """
    model_results_dir = os.path.join(RESULTS_DIR, model_short)
    frames = []
    for subset in SUBSETS:
        path = os.path.join(model_results_dir, f"predictions_{subset}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: predictions_{subset}.csv not found for "
                  f"'{model_short}' — skipping that subset.")
            continue
        df = pd.read_csv(path, usecols=["lat", "lon", "month", "y_prob"])
        frames.append(df)
        print(f"  Loaded {len(df):,} rows from {subset}.")

    if not frames:
        return None

    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["y_prob"])
    print(f"  Combined total: {len(full):,} rows.")
    return full


def aggregate_risk(df, months, breaks):
    """
    Aggregate to one risk class per cell over the given months:
    mean y_prob per (lat, lon), then apply the frozen risk thresholds.

    Returns a DataFrame with columns lat, lon, risk_class.
    """
    subset = df[df["month"].isin(months)]
    agg = (subset.groupby(["lat", "lon"], as_index=False)["y_prob"].mean())
    agg["risk_class"] = apply_thresholds(agg["y_prob"].values, breaks)
    return agg[["lat", "lon", "risk_class"]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fire_season_panels = []   # one dry-season panel per model -> combined figure

    for model_key, model_short in MODELS.items():
        section(f"Processing risk maps — {model_short}")

        model_results_dir = os.path.join(RESULTS_DIR, model_short)
        breaks_path       = os.path.join(model_results_dir, "risk_thresholds.json")

        # --- Load frozen risk thresholds (computed in step 05) ---
        if not os.path.exists(breaks_path):
            print(f"  ERROR: risk_thresholds.json not found for '{model_short}'. "
                  f"Run 05_compute_risk_thresholds.py first — skipping.")
            continue
        breaks = load_thresholds(breaks_path)["breaks"]

        # --- Load combined predictions ---
        df = load_all_predictions(model_short)
        if df is None or df.empty:
            print(f"  ERROR: no predictions available for '{model_short}' — skipping.")
            continue

        # --- Fire-season (dry) panel for the combined comparison figure ---
        dry_agg = aggregate_risk(df, DRY_MONTHS, breaks)
        fire_season_panels.append({
            "df":    dry_agg,
            "title": MODEL_DISPLAY[model_key],
        })

        # --- Appendix: 12-month climatology grid for this model ---
        print(f"\n  Building monthly climatology grid for {model_short}...")
        monthly_panels = []
        for month_num, month_name in MONTH_NAMES.items():
            agg = aggregate_risk(df, [month_num], breaks)
            monthly_panels.append({"df": agg, "title": month_name})

        monthly_path = os.path.join(OUTPUT_DIR, model_short,
                                    f"{model_short}_monthly_climatology")
        plot_risk_map_grid(
            panels=monthly_panels,
            output_path=monthly_path,
            nrows=3, ncols=4,
            suptitle=f"{MODEL_DISPLAY[model_key]} — Monthly Fire Risk Climatology",
            panel_letters=False,
        )

        del df  # free the large combined frame before the next model

    # --- Main-text figure: fire-season comparison across models ---
    if fire_season_panels:
        section("Building fire-season comparison figure (all models)")
        comparison_path = os.path.join(OUTPUT_DIR, "fire_season_comparison")
        plot_risk_map_grid(
            panels=fire_season_panels,
            output_path=comparison_path,
            nrows=1, ncols=len(fire_season_panels),
            suptitle="Fire-Season (Dry, Nov–Apr) Fire Risk by Model",
            panel_letters=True,
        )
    else:
        print("\nNo models produced fire-season panels — comparison figure skipped.")

    print("\nAll risk maps complete.")


if __name__ == "__main__":
    main()
