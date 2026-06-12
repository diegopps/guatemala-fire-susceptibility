"""
01_plot_future_classification.py
--------------------------------
Summarizes the future fire-risk projections (step 01_future_classification) into
publication figures, applying the FROZEN risk thresholds (step 05) so present and
future maps are directly comparable.

Two figures are produced:

  Figure 1 — Temporal trend (results)
    fig1_trend.{png,pdf,svg}
    Three rows (one per model, labelled A/B/C). x = year (1961–2100),
    y = % of national area in the High+Very-High fire-risk classes (fire season).
    Three SSP scenario lines per row + the shared historical baseline, all as
    10-year rolling means.

  Figure 2 — Spatial change (results / appendix), generated for ALL THREE models
    in two versions each:
      fig2_<MODEL>_normal.{png,pdf,svg}  — absolute fire-season risk classes
      fig2_<MODEL>_change.{png,pdf,svg}  — Δ risk class vs the baseline
    Layout: one centered baseline map (1991–2014) on the left, then a 3×2 grid
    (rows = SSP scenarios, with vertical labels; columns = Mid-century 2041–2060,
    End-century 2081–2100). In the change version the baseline keeps the absolute
    5-class colors (the reference) while the Mid/End cells show Δ risk class.

Aggregation: per cell, the predicted probability is averaged over the fire-season
months across the period's years, then the frozen thresholds are applied.

Usage:
    python 01_plot_future_classification.py
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR     = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
# Predictions written by 01_future_classification.py:
PREDICTIONS_DIR = f"{PROJECT_DIR}/data/classified_future"
THRESHOLDS_DIR  = f"{PROJECT_DIR}/results/04_final_model_training"
OUTPUT_DIR      = f"{PROJECT_DIR}/results/03_future_classification"

# Models — order determines row order in Figure 1
MODELS = {
    "random_forest": "RF",
    "xgboost":       "XGB",
    "extra_trees":   "XT",
}
MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "extra_trees":   "Extra Trees",
}

# SSP scenarios — order = row order (top→bottom) in Figure 2
SSP_SCENARIOS  = ["ssp126", "ssp370", "ssp585"]
SCENARIO_SHORT = {"ssp126": "SSP1-2.6", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
SCENARIO_COLORS = {"ssp126": "#1d6996", "ssp370": "#e8a33d", "ssp585": "#cc3311"}
HIST_COLOR = "#444444"

# Year ranges
HIST_YEARS   = range(1961, 2015)   # historical scenario (1961–2014)
FUTURE_YEARS = range(2015, 2101)   # ssp scenarios (2015–2100)

# Periods for Figure 2 (per-cell fire-season mean over these years)
BASELINE_YEARS = range(1991, 2015)             # 1991–2014 (historical)
FUTURE_PERIODS = {                              # from each ssp scenario
    "mid": (range(2041, 2061), "Mid-century\n(2041–2060)"),
    "end": (range(2081, 2101), "End-century\n(2081–2100)"),
}
BASELINE_LABEL = "Baseline\n(1991–2014)"

# Trend metric: classes counted as "high risk"
HIGH_RISK_CLASSES = (4, 5)        # High + Very High
ROLLING_WINDOW    = 10            # years

PRED_FILENAME_PATTERN = "{scenario}_{year}_predictions.csv"
SAVE_FORMATS = ["png", "pdf", "svg"]
DPI = 300

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
import matplotlib.patches as mpatches
import matplotlib.cm as cm

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from risk_thresholds import load_thresholds, apply_thresholds
from plot_risk_map import (
    _render_risk_map, _load_boundaries, _make_colormap, make_delta_cmap,
    RISK_COLORS, RISK_LABELS, DRY_MONTHS,
)

# =============================================================================
# DATA
# =============================================================================

def section(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def load_year(model_short, scenario, year):
    """Load one prediction CSV (lat, lon, month, y_prob) or None if missing."""
    path = os.path.join(PREDICTIONS_DIR, model_short, scenario,
                        PRED_FILENAME_PATTERN.format(scenario=scenario, year=year))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, usecols=["lat", "lon", "month", "y_prob"])


def process_model_scenario(model_short, scenario, breaks, years, period_windows):
    """
    Single pass over a scenario's years. Returns:
      trend  : {year: % area high-risk}  (fire season, that year)
      cells  : {period_name: DataFrame(lat, lon, y_prob)}  per-cell fire-season
               mean probability averaged over each period's years.
    """
    trend  = {}
    plists = {name: [] for name in period_windows}

    for year in years:
        df = load_year(model_short, scenario, year)
        if df is None:
            continue
        dry = df[df["month"].isin(DRY_MONTHS)]
        if dry.empty:
            continue
        cell = dry.groupby(["lat", "lon"], as_index=False)["y_prob"].mean()

        cls = apply_thresholds(cell["y_prob"].to_numpy(), breaks)
        trend[year] = 100.0 * float(np.isin(cls, HIGH_RISK_CLASSES).mean())

        for name, yr_range in period_windows.items():
            if year in yr_range:
                plists[name].append(cell)

    cells = {}
    for name, lst in plists.items():
        if lst:
            cells[name] = (pd.concat(lst, ignore_index=True)
                           .groupby(["lat", "lon"], as_index=False)["y_prob"].mean())
    return trend, cells


def classify(cell_df, breaks):
    """Per-cell mean prob -> risk_class (1..5)."""
    out = cell_df[["lat", "lon"]].copy()
    out["risk_class"] = apply_thresholds(cell_df["y_prob"].to_numpy(), breaks)
    return out


def rolling(series_dict):
    """Sorted pandas Series with a centered rolling mean over ROLLING_WINDOW years."""
    s = pd.Series(series_dict).sort_index()
    return s.rolling(ROLLING_WINDOW, center=True, min_periods=1).mean()


def save_figure(fig, stem):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fmt in SAVE_FORMATS:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}.{fmt}"),
                    dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {stem}.{{{','.join(SAVE_FORMATS)}}}")
    plt.close(fig)


# =============================================================================
# FIGURE 1 — TEMPORAL TREND
# =============================================================================

def render_trend_figure(all_trends):
    section("Figure 1 — temporal trend")
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 11), sharex=True)
    letters = "ABCDEFG"

    for i, (model_key, short) in enumerate(MODELS.items()):
        ax = axes[i]
        tr = all_trends[short]

        if tr.get("historical"):
            h = rolling(tr["historical"])
            ax.plot(h.index, h.values, color=HIST_COLOR, lw=1.8, label="Historical")
        for ssp in SSP_SCENARIOS:
            if tr.get(ssp):
                s = rolling(tr[ssp])
                ax.plot(s.index, s.values, color=SCENARIO_COLORS[ssp], lw=1.8,
                        label=SCENARIO_SHORT[ssp])

        ax.axvline(2015, color="grey", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(f"{letters[i]})  {MODEL_DISPLAY[model_key]}",
                     loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("% area High+Very-High")
        ax.set_xlim(1961, 2100)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=11)
    fig.suptitle(
        f"Fire-season area at High/Very-High risk, 1961–2100 "
        f"({ROLLING_WINDOW}-yr rolling mean)",
        fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_figure(fig, "fig1_trend")


# =============================================================================
# FIGURE 2 — SPATIAL CHANGE (per model, normal + change)
# =============================================================================

def render_spatial_figure(model_key, model_short, period_cells, breaks, mode):
    """
    mode: "normal" (absolute classes) or "change" (Δ class vs baseline).
    period_cells keys: "baseline", (ssp, "mid"), (ssp, "end").
    """
    country, departments = _load_boundaries()
    risk_cmap,  risk_norm  = _make_colormap()
    delta_cmap, delta_norm = make_delta_cmap(4)

    if "baseline" not in period_cells:
        print(f"  WARNING: no baseline data for {model_short} — skipping {mode} figure.")
        return
    base_cls = classify(period_cells["baseline"], breaks)

    fig = plt.figure(figsize=(13, 13))
    gs  = fig.add_gridspec(3, 3, wspace=0.08, hspace=0.10)

    # --- Baseline map: single, centered on the left (middle row) ---
    ax_base = fig.add_subplot(gs[1, 0])
    _render_risk_map(ax_base, base_cls, title="", risk_col="risk_class",
                     country=country, departments=departments,
                     cmap=risk_cmap, norm=risk_norm,
                     show_xlabel=False, show_ylabel=False)
    ax_base.set_title(BASELINE_LABEL, fontsize=11, fontweight="bold", pad=6)

    # --- 3×2 grid: rows = SSP, cols = mid / end ---
    period_keys = list(FUTURE_PERIODS.keys())   # ["mid", "end"]
    for r, ssp in enumerate(SSP_SCENARIOS):
        for c, pkey in enumerate(period_keys):
            ax = fig.add_subplot(gs[r, c + 1])
            key = (ssp, pkey)
            if key not in period_cells:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.axis("off")
            else:
                fut_cls = classify(period_cells[key], breaks)
                if mode == "normal":
                    _render_risk_map(ax, fut_cls, title="", risk_col="risk_class",
                                     country=country, departments=departments,
                                     cmap=risk_cmap, norm=risk_norm,
                                     show_xlabel=False, show_ylabel=False)
                else:  # change
                    m = fut_cls.merge(base_cls, on=["lat", "lon"],
                                      suffixes=("_fut", "_base"))
                    m["delta"] = m["risk_class_fut"] - m["risk_class_base"]
                    _render_risk_map(ax, m[["lat", "lon", "delta"]], title="",
                                     risk_col="delta",
                                     country=country, departments=departments,
                                     cmap=delta_cmap, norm=delta_norm,
                                     show_xlabel=False, show_ylabel=False)

            if r == 0:
                ax.set_title(FUTURE_PERIODS[pkey][1], fontsize=11, fontweight="bold", pad=6)
            if c == 0:
                # Vertical scenario label, to the right of the baseline map.
                ax.set_ylabel(SCENARIO_SHORT[ssp], rotation=90, fontsize=11,
                              fontweight="bold", labelpad=8)

    # --- Legends ---
    risk_patches = [mpatches.Patch(facecolor=RISK_COLORS[i], edgecolor="#aaaaaa",
                                   linewidth=0.5, label=RISK_LABELS[i])
                    for i in range(1, 6)]
    fig.subplots_adjust(bottom=0.10)

    if mode == "normal":
        fig.legend(handles=risk_patches, title="Fire Risk", loc="lower center",
                   ncol=5, frameon=True, edgecolor="#cccccc",
                   bbox_to_anchor=(0.5, 0.02), fontsize=10, title_fontsize=10)
        mode_label = "by period"
    else:
        # Baseline (absolute) legend on the left, Δ colorbar on the right.
        fig.legend(handles=risk_patches, title="Baseline risk", loc="lower left",
                   ncol=2, frameon=True, edgecolor="#cccccc",
                   bbox_to_anchor=(0.08, 0.02), fontsize=9, title_fontsize=9)
        sm = cm.ScalarMappable(cmap=delta_cmap, norm=delta_norm); sm.set_array([])
        cax = fig.add_axes([0.55, 0.06, 0.35, 0.018])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                           ticks=range(-4, 5))
        cb.set_label("Δ risk class (future − baseline)", fontsize=9)
        mode_label = "change vs baseline"

    fig.suptitle(f"{MODEL_DISPLAY[model_key]} — Fire-season fire risk ({mode_label})",
                 fontsize=14, fontweight="bold", y=0.97)
    save_figure(fig, f"fig2_{model_short}_{mode}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_trends = {}

    for model_key, model_short in MODELS.items():
        section(f"Processing {MODEL_DISPLAY[model_key]} ({model_short})")

        thr_path = os.path.join(THRESHOLDS_DIR, model_short, "risk_thresholds.json")
        if not os.path.exists(thr_path):
            print(f"  ERROR: risk_thresholds.json not found for {model_short} "
                  f"— run 05_compute_risk_thresholds.py. Skipping model.")
            continue
        breaks = load_thresholds(thr_path)["breaks"]

        # Historical: trend (1961–2014) + baseline period (1991–2014).
        trend_hist, cells_hist = process_model_scenario(
            model_short, "historical", breaks, HIST_YEARS,
            {"baseline": BASELINE_YEARS})

        trends      = {"historical": trend_hist}
        period_cells = {"baseline": cells_hist.get("baseline")}

        # Each SSP: trend (2015–2100) + mid/end periods.
        for ssp in SSP_SCENARIOS:
            t, cells = process_model_scenario(
                model_short, ssp, breaks, FUTURE_YEARS,
                {name: rng for name, (rng, _) in FUTURE_PERIODS.items()})
            trends[ssp] = t
            for pkey in FUTURE_PERIODS:
                if pkey in cells:
                    period_cells[(ssp, pkey)] = cells[pkey]

        all_trends[model_short] = trends
        period_cells = {k: v for k, v in period_cells.items() if v is not None}

        # Figure 2 (both versions) for this model.
        render_spatial_figure(model_key, model_short, period_cells, breaks, "normal")
        render_spatial_figure(model_key, model_short, period_cells, breaks, "change")

    if all_trends:
        render_trend_figure(all_trends)

    print("\nAll future-classification figures complete. Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
