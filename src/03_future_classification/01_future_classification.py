"""
07_future_classification.py
----------------------------
Applies a single pre-trained final model to all preprocessed future climate
and historical CSVs, generating monthly fire probability predictions for
every grid cell in Guatemala across all scenarios and years.

For each scenario × year combination, the script loads the preprocessed CSV,
runs predict_proba() using the saved .joblib model, and writes a lightweight
output CSV containing only lat, lon, year, month, and y_prob.

Jenks risk classification is NOT applied here — it is applied downstream
in the plotting script, keeping this step fast and output files compact.

Usage:
    python 01_future_classification.py <model_name>

Examples:
    python 01_future_classification.py random_forest
    python 01_future_classification.py xgboost
    python 01_future_classification.py extra_trees

Each model is intended to be submitted as a separate SLURM job.

Output structure:
        <scenario>_<year>_predictions.csv
    e.g.
        01_future_classification/RF/ssp126/ssp126_2015_predictions.csv
        01_future_classification/RF/historical/historical_1961_predictions.csv
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR        = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
PROCESSED_DATA_DIR = f"{PROJECT_DIR}/data/future_climate_processed"
MODELS_DIR         = f"{PROJECT_DIR}/models"
OUTPUT_DIR         = f"{PROJECT_DIR}/data/indata/01_future_classification"

# Scenarios and their year ranges — subdirectory name must match key
SCENARIOS = {
    "historical": range(1961, 2015),   # 1961–2014 inclusive
    "ssp126":     range(2015, 2101),   # 2015–2100 inclusive
    "ssp370":     range(2015, 2101),
    "ssp585":     range(2015, 2101),
}

# File naming pattern for preprocessed input CSVs
INPUT_FILENAME_PATTERN  = "fullvarset_{scenario}_{year}.csv"

# File naming pattern for output predictions
OUTPUT_FILENAME_PATTERN = "{scenario}_{year}_predictions.csv"

# Features used during training — must match exactly
FEATURE_COLUMNS = [
    "PET", "climbal", "pr", "rsds",
    "tas", "tasmax", "tasmin",
    "aspect_sin", "aspect_cos",
    "distRoads", "distUrban", "distWater",
    "elev", "slope",
]

# Columns to carry through to the output CSV
OUTPUT_COLUMNS = ["lat", "lon", "year", "month", "y_prob"]

# Model registry — short codes and .joblib filenames
MODELS = {
    "random_forest": {
        "short":        "RF",
        "joblib_file":  "RF_final.joblib",
    },
    "xgboost": {
        "short":        "XGB",
        "joblib_file":  "XGB_final.joblib",
    },
    "extra_trees": {
        "short":        "XT",
        "joblib_file":  "XT_final.joblib",
    },
}

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os
import joblib
import numpy as np
import pandas as pd

# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def load_model(model_key):
    """Load the saved .joblib model for the given model key."""
    joblib_file = MODELS[model_key]["joblib_file"]
    model_path  = os.path.join(MODELS_DIR, joblib_file)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"  Model loaded: {type(model).__name__}")
    return model


def validate_features(df, filepath):
    """
    Check all required feature columns are present in the dataframe.
    Returns True if valid, False if any are missing.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"    WARNING: Missing feature columns {missing} in "
              f"{os.path.basename(filepath)} — skipping.")
        return False
    return True


def process_year(model, in_path, out_path, year):
    """
    Load one preprocessed CSV, run predictions, save output.
    Returns 'processed', 'skipped', or 'error'.
    """
    # Skip if output already exists
    if os.path.exists(out_path):
        return "skipped"

    if not os.path.exists(in_path):
        return "missing"

    # --- Load data ---
    df = pd.read_csv(in_path)

    # --- Validate features ---
    if not validate_features(df, in_path):
        return "error"

    # --- Validate coordinate and time columns ---
    for col in ["lat", "lon", "month"]:
        if col not in df.columns:
            print(f"    WARNING: Missing column '{col}' in "
                  f"{os.path.basename(in_path)} — skipping.")
            return "error"

    # --- Predict probabilities ---
    X = df[FEATURE_COLUMNS]

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        print(f"    WARNING: Model does not support predict_proba — skipping "
              f"{os.path.basename(in_path)}.")
        return "error"

    # --- Assemble output ---
    out_df = pd.DataFrame({
        "lat":    df["lat"].values,
        "lon":    df["lon"].values,
        "year":   year,
        "month":  df["month"].values,
        "y_prob": y_prob,
    })

    # --- Save ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return "processed"


# =============================================================================
# MAIN
# =============================================================================

def main(model_key):
    if model_key not in MODELS:
        print(f"ERROR: Unknown model '{model_key}'.")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_short = MODELS[model_key]["short"]

    print(f"\n{'='*60}")
    print(f"Model   : {model_key}  ({model_short})")
    print(f"Output  : {OUTPUT_DIR}/{model_short}/")
    print(f"{'='*60}")

    # --- Load model once — reused across all scenario × year combinations ---
    model = load_model(model_key)

    total_processed = 0
    total_skipped   = 0
    total_missing   = 0
    total_errors    = 0

    for scenario, year_range in SCENARIOS.items():
        section(f"Scenario: {scenario}  ({min(year_range)}–{max(year_range)})")

        in_scenario_dir  = os.path.join(PROCESSED_DATA_DIR, scenario)
        out_scenario_dir = os.path.join(OUTPUT_DIR, model_short, scenario)

        if not os.path.isdir(in_scenario_dir):
            print(f"  WARNING: Preprocessed input directory not found — skipping: "
                  f"{in_scenario_dir}")
            continue

        n_proc = 0
        n_skip = 0
        n_miss = 0
        n_err  = 0

        for year in year_range:
            in_filename  = INPUT_FILENAME_PATTERN.format(
                scenario=scenario, year=year)
            out_filename = OUTPUT_FILENAME_PATTERN.format(
                scenario=scenario, year=year)

            in_path  = os.path.join(in_scenario_dir,  in_filename)
            out_path = os.path.join(out_scenario_dir, out_filename)

            result = process_year(model, in_path, out_path, year)

            if result == "processed":
                print(f"  OK      : {out_filename}")
                n_proc += 1
            elif result == "skipped":
                print(f"  SKIP    : {out_filename}  (output already exists)")
                n_skip += 1
            elif result == "missing":
                print(f"  MISSING : {in_filename}  (preprocessed file not found)")
                n_miss += 1
            else:
                print(f"  ERROR   : {in_filename}")
                n_err += 1

        print(f"\n  Summary for {scenario}:")
        print(f"    Processed : {n_proc}")
        print(f"    Skipped   : {n_skip}  (output already exists)")
        print(f"    Missing   : {n_miss}  (preprocessed file not found)")
        print(f"    Errors    : {n_err}")

        total_processed += n_proc
        total_skipped   += n_skip
        total_missing   += n_miss
        total_errors    += n_err

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Model           : {model_key}  ({model_short})")
    print(f"  Total processed : {total_processed}")
    print(f"  Total skipped   : {total_skipped}")
    print(f"  Total missing   : {total_missing}")
    print(f"  Total errors    : {total_errors}")
    print(f"  Output dir      : {OUTPUT_DIR}/{model_short}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 07_future_classification.py <model_name>")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    main(sys.argv[1])