"""
06_plot_risk_maps.py
---------------------
Applies the frozen area-quintile thresholds (Verde & Zêzere, 2010) from step 05
to predicted probabilities and produces fire risk classification maps for the
final trained models.

Three figure types are produced:

  1. Fire-season comparison (main text):
       A single figure with one panel per model (RF, XGB, XT), each showing the
       dry-season (Nov–Apr, the official fire season) risk climatology. Panels
       are lettered A), B), C) for cross-referencing.
         -> results/06_risk_maps/fire_season_comparison.{png,pdf,svg}

  2. Monthly climatology (appendix):
       One 3x4 panel figure per model, showing the 12 calendar-month
       climatologies.
         -> results/06_risk_maps/<MODEL_SHORT>/<MODEL_SHORT>_monthly_climatology.{...}

  3. Prediction-rate curve (diagnostic / methods):
       A 3-row panel figure (one row per model) — x = cumulative fraction of area
       ranked by descending predicted susceptibility, y = cumulative fraction of
       observed fires captured. The four frozen quintile breaks are overlaid as
       vertical dashed lines, each annotated with the class fire-capture %. This
       figure visually justifies the quintile choice (smooth curve → quintiles)
       and shows what each class captures. Computed on held-out (test+val) data
       only.
         -> results/06_risk_maps/prediction_rate_curves.{png,pdf,svg}

Data
----
Maps (figures 1 & 2) are built from ALL prediction subsets combined
(train + test + val), giving complete spatial coverage and every cell's full
month history. This is a visualization of the model's risk surface — quantitative
performance is reported separately on the held-out test/validation sets
(steps 02–04). The risk classes come from the frozen quintile thresholds fit on
HELD-OUT data (step 05), so the classification scheme is not biased by in-sample
training probabilities.

The prediction-rate curve (figure 3) is computed on HELD-OUT data only
(test + val), using the same dry-season per-cell mean probabilities as step 05.

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

Reference:
    Verde, A., & Zêzere, J. L. (2010). Assessment and validation of wildfire
    susceptibility and hazard in Portugal. Natural Hazards and Earth System
    Sciences, 10(3), 485–497. https://doi.org/10.5194/nhess-10-485-2010
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

# Subsets combined to form the full prediction set per model (maps only)
SUBSETS = ["train", "test", "val"]

# Held-out subsets used for the prediction-rate curve (matches step 05)
HELD_OUT_SUBSETS = ["test", "val"]

# Prediction-rate curve plot settings (styled after plot_roccurve_panel.py)
_CURVE_COLOR    = "#d6604d"
_BASELINE_COLOR = "#888888"
_BREAK_COLOR    = "#333333"
_DPI            = 300
_FORMATS        = ["png", "pdf", "svg"]
_SUPTITLE_FS    = 18
_TITLE_FS       = 14
_LABEL_FS       = 12
_TICK_FS        = 11
_ANNOT_FS       = 10

# Risk class labels for annotation (ascending class order 1..5)
_CLASS_LABELS = {1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Very high"}

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from risk_thresholds import load_thresholds, apply_thresholds
from plot_risk_map import plot_risk_map_grid, DRY_MONTHS, MONTH_NAMES

# =============================================================================
# HELPERS — MAPS
# =============================================================================

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def load_all_predictions(model_short, subsets):
    """
    Load and concatenate predictions for one model from `subsets`. Returns a
    DataFrame with columns lat, lon, month, y_prob (NaN rows dropped), or None.
    """
    model_results_dir = os.path.join(RESULTS_DIR, model_short)
    frames = []
    for subset in subsets:
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

    full = pd.concat(frames, ignore_index=True).dropna(subset=["y_prob"])
    print(f"  Combined total: {len(full):,} rows.")
    return full


def aggregate_risk(df, months, breaks):
    """
    Aggregate to one risk class per cell over the given months:
    mean y_prob per (lat, lon), then apply the frozen risk thresholds.

    Returns a DataFrame with columns lat, lon, risk_class.
    """
    subset = df[df["month"].isin(months)]
    agg = subset.groupby(["lat", "lon"], as_index=False)["y_prob"].mean()
    agg["risk_class"] = apply_thresholds(agg["y_prob"].values, breaks)
    return agg[["lat", "lon", "risk_class"]]


# =============================================================================
# HELPERS — PREDICTION-RATE CURVE
# =============================================================================

def load_held_out_cells(model_short):
    """
    Per-cell dry-season mean susceptibility and fire count from held-out data.
    Returns a DataFrame with columns susc, fires, or None.
    """
    model_dir = os.path.join(RESULTS_DIR, model_short)
    frames = []
    for subset in HELD_OUT_SUBSETS:
        path = os.path.join(model_dir, f"predictions_{subset}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: predictions_{subset}.csv not found for "
                  f"'{model_short}' — skipping.")
            continue
        frames.append(pd.read_csv(
            path, usecols=["lat", "lon", "month", "y_true", "y_prob"]))

    if not frames:
        return None

    df   = pd.concat(frames, ignore_index=True).dropna(subset=["y_prob"])
    df   = df[df["month"].isin(DRY_MONTHS)]
    cell = df.groupby(["lat", "lon"], as_index=False).agg(
        susc=("y_prob", "mean"), fires=("y_true", "sum"))
    return cell


def _build_curve(susc, fires):
    """Return (cum_area, cum_capture) arrays for the prediction-rate curve."""
    n     = len(susc)
    order = np.argsort(-susc, kind="mergesort")
    cum_capture = np.cumsum(fires[order]) / fires.sum()
    cum_area    = np.arange(1, n + 1) / n
    return cum_area, cum_capture


def plot_prediction_rate_curves(curve_data_list):
    """
    Draw a prediction-rate curve panel figure (one row per model), styled after
    plot_roccurve_panel.py. Overlays the four quintile break positions as
    vertical dashed lines and annotates each class's fire-capture %.

    Parameters
    ----------
    curve_data_list : list of dict, each with keys:
        model_name   : str
        cum_area     : np.ndarray
        cum_capture  : np.ndarray
        auc          : float
        breaks       : list of 4 probability thresholds (ascending)
        class_summary: list of per-class dicts (class, label, area_pct, capture_pct)
        susc         : np.ndarray  (per-cell mean prob, for locating breaks on x-axis)
    """
    n   = len(curve_data_list)
    fig, axes = plt.subplots(n, 1, figsize=(7, 6 * n), constrained_layout=False)
    fig.subplots_adjust(hspace=0.45, top=0.93)
    if n == 1:
        axes = [axes]

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, (ax, entry) in enumerate(zip(axes, curve_data_list)):
        cum_area    = entry["cum_area"]
        cum_capture = entry["cum_capture"]
        susc        = entry["susc"]
        breaks      = entry["breaks"]
        summary     = {row["class"]: row for row in entry["class_summary"]}

        # --- Diagonal reference (random classifier) ---
        ax.plot([0, 1], [0, 1], color=_BASELINE_COLOR, linewidth=1.5,
                linestyle="--", label="Random (AUC = 0.500)", zorder=1)

        # --- Prediction-rate curve ---
        ax.plot(cum_area, cum_capture, color=_CURVE_COLOR, linewidth=2.0,
                label="Prediction-rate curve", zorder=2)

        # --- Quintile break lines: locate each break on the x-axis ---
        # On the x-axis cells are ranked by DESCENDING susceptibility, so the
        # highest-prob cells are at x=0 (class 5) and lowest at x=1 (class 1).
        # x_break = fraction of cells with prob >= break = left edge of that band.
        # Breaks are sorted descending so boundaries run left→right: 0, VH/H, H/M, M/L, L/VL, 1.
        n_cells = len(susc)
        x_breaks = [float(np.sum(susc >= brk)) / n_cells
                    for brk in sorted(breaks, reverse=True)]
        # Boundaries: x=0 (start of VH), then each break x, then x=1 (end of VL)
        class_boundaries = [0.0] + x_breaks + [1.0]

        # Annotate each of the 5 class bands between consecutive boundaries
        # Classes run 5→1 as x goes 0→1 (descending susceptibility order)
        for cls_idx, cls in enumerate([5, 4, 3, 2, 1]):
            x_left  = class_boundaries[cls_idx]
            x_right = class_boundaries[cls_idx + 1]
            x_mid   = 0.5 * (x_left + x_right)

            # Vertical boundary line at the right edge of each band (skip x=1)
            if x_right < 1.0:
                ax.axvline(x_right, color=_BREAK_COLOR, linewidth=1.0,
                           linestyle=":", alpha=0.7, zorder=3)

            cap_pct = summary[cls]["capture_pct"]
            y_annot = 0.97 - cls_idx * 0.10
            ax.text(x_mid, y_annot, f"{_CLASS_LABELS[cls]}\n{cap_pct:.1f}%",
                    transform=ax.get_xaxis_transform(),
                    fontsize=_ANNOT_FS - 1, ha="center", va="top",
                    color=_BREAK_COLOR, zorder=4)

        # --- AUC box ---
        ax.text(0.97, 0.15,
                f"AUC = {entry['auc']:.3f}",
                transform=ax.transAxes,
                fontsize=_ANNOT_FS,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

        # --- Panel letter ---
        ax.text(-0.18, 1.08, f"{letters[idx]})",
                transform=ax.transAxes,
                fontsize=_TITLE_FS, fontweight="bold", va="top", ha="left")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Cumulative fraction of area (ranked by descending susceptibility)",
                      fontsize=_LABEL_FS)
        ax.set_ylabel("Cumulative fraction of fires captured", fontsize=_LABEL_FS)
        ax.set_title(entry["model_name"], fontsize=_TITLE_FS, fontweight="bold", pad=10)
        ax.legend(fontsize=_TICK_FS - 2, loc="lower right", frameon=True)
        ax.tick_params(axis="both", labelsize=_TICK_FS)
        sns.despine(ax=ax)

    fig.suptitle(
        "Prediction-Rate Curves with Quintile Class Boundaries\n"
        "(Verde & Zêzere, 2010 — held-out test+val, dry season)",
        fontsize=_SUPTITLE_FS, fontweight="bold",
    )

    out_stem = os.path.join(OUTPUT_DIR, "prediction_rate_curves")
    for fmt in _FORMATS:
        fig.savefig(f"{out_stem}.{fmt}", format=fmt, dpi=_DPI, bbox_inches="tight")
        print(f"  Saved: {out_stem}.{fmt}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fire_season_panels = []   # one dry-season panel per model -> combined figure
    curve_data_list    = []   # one entry per model -> prediction-rate curve figure

    for model_key, model_short in MODELS.items():
        section(f"Processing risk maps — {model_short}")

        model_results_dir = os.path.join(RESULTS_DIR, model_short)
        breaks_path       = os.path.join(model_results_dir, "risk_thresholds.json")

        # --- Load frozen risk thresholds (computed in step 05) ---
        if not os.path.exists(breaks_path):
            print(f"  ERROR: risk_thresholds.json not found for '{model_short}'. "
                  f"Run 05_compute_risk_thresholds.py first — skipping.")
            continue
        thr_data = load_thresholds(breaks_path)
        breaks   = thr_data["breaks"]

        # --- Load combined predictions (all subsets — maps only) ---
        df = load_all_predictions(model_short, SUBSETS)
        if df is None or df.empty:
            print(f"  ERROR: no predictions available for '{model_short}' — skipping.")
            continue

        # --- Fire-season panel for the combined comparison figure ---
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

        del df  # free the large combined frame before next model

        # --- Held-out cells for the prediction-rate curve ---
        print(f"\n  Loading held-out data for prediction-rate curve ({model_short})...")
        held = load_held_out_cells(model_short)
        if held is not None and not held.empty:
            susc_arr   = held["susc"].to_numpy()
            fires_arr  = held["fires"].to_numpy()
            cum_area, cum_capture = _build_curve(susc_arr, fires_arr)

            # Use AUC from the frozen file (avoids floating-point drift)
            auc = thr_data.get("prediction_rate_auc", float(
                np.trapz(cum_capture, cum_area)))

            curve_data_list.append({
                "model_name":   MODEL_DISPLAY[model_key],
                "cum_area":     cum_area,
                "cum_capture":  cum_capture,
                "auc":          auc,
                "breaks":       breaks,
                "class_summary": thr_data["class_summary"],
                "susc":         susc_arr,
            })
        else:
            print(f"  WARNING: no held-out data for prediction-rate curve — skipping {model_short}.")

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

    # --- Prediction-rate curve panel figure ---
    if curve_data_list:
        section("Building prediction-rate curve panel figure")
        plot_prediction_rate_curves(curve_data_list)
    else:
        print("\nNo curve data available — prediction-rate figure skipped.")

    print("\nAll risk maps complete.")


if __name__ == "__main__":
    main()
