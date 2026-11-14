"""
01_future_classification_plot.py
--------------------------------
Figures of the projections, from the tables of 01_future_classification_evaluate.py.

Main figures (results/mainfigs/, full width):
  03_change_maps.pdf      Whole year, one group of columns per model: the class of the baseline
                          above, then the class change (future - baseline class) in five bins
                          (<= -2, -1, 0, +1, >= +2), SSPs in rows and periods in columns.
  03_high_risk_share.pdf  Share of the country in risk_classes.high_risk (High and Very high)
                          per year, one panel per model: median and range of the GCMs, running
                          means of figures.running_mean_years complete years, the historical
                          years averaged over the SSP continuations (as 01_data_exploration/
                          03_climate_trends.py); the baseline level (median of the GCMs) marked.
  03_classes_monthly.pdf  Share of the country in each class per calendar month, laid out as
                          03_change_maps.pdf: one group of columns per model (panel labels
                          (a)–(c)), the baseline above, then SSPs in rows and periods in columns,
                          with the class limits of the model's baseline.
Appendix (results/appendix/):
  03_change_monthly_<model>_<scenario>.pdf
                          Class change per calendar month (against the same month of the
                          baseline; every change from -4 to +4), periods.end_century, one
                          figure per model and SSP.

Cells outside the pipeline's cell set (lakes) are left blank.

Usage:
    python 01_future_classification_plot.py
"""

import calendar
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from chelsa_grid import reference_grid
from figures import MODELS as MODEL_STYLES, SCENARIOS, add_panel_label, cm, save_figure, setup_style
from maps import (COUNTRY_LINE, DEPARTMENT_LINE, cells_to_raster, draw_boundaries, draw_raster,
                  map_extent, setup_map)

MODELS = cfg.model_training.models
GCMS = cfg.gcms.models
HIST = cfg.scenarios.historical
SSPS = cfg.scenarios.ssp
IN_DIR = cfg.paths.results.future_classification
LABELS = cfg.risk_classes.labels
WINDOW = cfg.figures.running_mean_years
WIDTH = cfg.figures.width_cm.full
BASELINE = cfg.periods.baseline
BASELINE_LABEL = f"{BASELINE.start}–{BASELINE.stop - 1}"
FUTURE_PERIODS = ["mid_century", "end_century"]
MONTHLY_PERIOD = "end_century"

# ColorBrewer YlOrRd (5 classes): sequential, colour-blind safe (as step 05).
CLASS_COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
CLASS_CMAP = (ListedColormap(CLASS_COLORS),
              BoundaryNorm(np.arange(0.5, len(CLASS_COLORS) + 1), len(CLASS_COLORS)))

# Class change (future - baseline class, -4 to +4): ColorBrewer RdBu, colour-blind safe, blue
# for lower and red for higher classes, with a grey middle (no change) so that it differs from
# the blank cells outside the country. Colour map, norm, tick positions and tick labels.
# Monthly maps (appendix): every change, RdBu 9 classes.
CHANGES = np.arange(-(len(LABELS) - 1), len(LABELS))
CHANGE_SCALE = (ListedColormap(["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#e6e6e6",
                                "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]),
                BoundaryNorm(np.r_[CHANGES - 0.5, CHANGES[-1] + 0.5], len(CHANGES)),
                CHANGES, [f"{d:+d}".replace("-", "−") if d else "0" for d in CHANGES])
# Whole-year maps (03_change_maps.pdf): five bins, the ends open (<= -2, >= +2), RdBu 5 classes.
BINNED_CHANGE_SCALE = (ListedColormap(["#0571b0", "#92c5de", "#e6e6e6", "#f4a582", "#ca0020"]),
                       BoundaryNorm([CHANGES[0] - 0.5, -1.5, -0.5, 0.5, 1.5, CHANGES[-1] + 0.5], 5),
                       [(CHANGES[0] - 2) / 2, -1, 0, 1, (CHANGES[-1] + 2) / 2],   # bin centres
                       ["≤ −2", "−1", "0", "+1", "≥ +2"])


def period_label(period):
    years = cfg.periods[period]
    return f"{years.start}–{years.stop - 1}"


# =============================================================================
# LAYOUT
# =============================================================================

def map_ratio():
    """Width / height of a national map (true distances at its mid-latitude)."""
    lon_min, lon_max, lat_min, lat_max = map_extent()
    return (lon_max - lon_min) * np.cos(np.deg2rad((lat_min + lat_max) / 2)) / (lat_max - lat_min)


def draw_map(ax, grid, cell_id, values, cmap, left, bottom, step=1):
    cmap, norm = cmap
    draw_raster(ax, grid, cells_to_raster(grid, cell_id, values), cmap=cmap, norm=norm)
    draw_boundaries(ax)
    setup_map(ax, left=left, bottom=bottom)
    if step != 1:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(step))


def class_handles():
    return [Patch(facecolor=c, edgecolor="#7f7f7f", linewidth=0.3, label=l)
            for c, l in zip(CLASS_COLORS, LABELS)]


def border_handles():
    return [Line2D([], [], label="National border", **COUNTRY_LINE),
            Line2D([], [], label="Department border", **DEPARTMENT_LINE)]


def change_colorbar(fig, rect, scale):
    """Discrete colour bar of the class change, one box per bin of the scale."""
    cmap, norm, ticks, labels = scale
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(rect),
                       orientation="horizontal", spacing="uniform", ticks=ticks)
    bar.ax.set_xticklabels(labels)
    bar.ax.tick_params(length=0)
    bar.outline.set_linewidth(0.3)
    bar.set_label(f"Class change relative to {BASELINE_LABEL} (number of classes)")


# =============================================================================
# HIGH-RISK SHARE
# =============================================================================

def yearly_high_share(yearly, model):
    """{scenario: DataFrame year x GCM} of the share of the country in High or Very high (%)."""
    sub = yearly[yearly.model == model]
    return {s: 100 * sub[sub.scenario == s].pivot(index="year", columns="gcm", values="high_share")
            for s in [HIST] + SSPS}


def smoothed(yearly):
    """Running means per GCM: {scenario: DataFrame}; the historical years are averaged over the
    SSP continuations (as 03_climate_trends.smoothed)."""
    roll = lambda df: df.rolling(WINDOW, center=True, min_periods=WINDOW).mean()
    hist_end = yearly[HIST].index.max()
    cont = {s: roll(pd.concat([yearly[HIST], yearly[s]])) for s in SSPS}
    out = {HIST: sum(cont[s].loc[:hist_end] for s in SSPS) / len(SSPS)}
    for s in SSPS:
        out[s] = cont[s].loc[hist_end:]          # starts at the last historical year, joining the lines
    return out


def high_risk_figure(yearly, summary):
    base = summary[summary.period == "baseline"].set_index("model").high_share_median * 100
    fig, axes = plt.subplots(1, len(MODELS), figsize=(cm(WIDTH), cm(6.5)), sharex=True, sharey=True)
    for k, (ax, model) in enumerate(zip(axes, MODELS)):
        for scenario, df in smoothed(yearly_high_share(yearly, model)).items():
            color = SCENARIOS[scenario]["color"]
            df = df[df.notna().all(axis=1)]
            ax.fill_between(df.index, df.min(axis=1), df.max(axis=1), color=color, alpha=0.18,
                            linewidth=0, zorder=1)
            ax.plot(df.index, df.median(axis=1), color=color, zorder=3)
        ax.axvline(cfg.periods.historical.stop - 0.5, color="#9e9e9e", linewidth=0.5,
                   linestyle=(0, (3, 2)), zorder=0)
        ax.axhline(base[model], color="black", linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=2)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(40))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.set_xlim(cfg.periods.historical.start, cfg.periods.future.stop - 1)
        ax.set_xlabel("Year")
        ax.grid(axis="y", linewidth=0.3, color="#d9d9d9")
        ax.spines[["top", "right"]].set_visible(False)
        add_panel_label(ax, k)
        ax.text(0.5, 1.02, MODEL_STYLES[model]["label"], transform=ax.transAxes, ha="center", va="bottom")
    high = " and ".join(LABELS[c - 1] for c in cfg.risk_classes.high_risk)
    axes[0].set_ylabel(f"Share of the country in the\n{high} classes (%)")
    handles = [(Patch(facecolor=SCENARIOS[s]["color"], alpha=0.18, linewidth=0),
                Line2D([], [], color=SCENARIOS[s]["color"])) for s in [HIST] + SSPS]
    labels = [SCENARIOS[s]["label"] for s in [HIST] + SSPS]
    handles += [Line2D([], [], color="#4d4d4d"), Patch(facecolor="#4d4d4d", alpha=0.18, linewidth=0),
                Line2D([], [], color="black", linewidth=0.6, linestyle=(0, (1, 1.5)))]
    labels += [f"Median of {len(GCMS)} GCMs", f"Range of {len(GCMS)} GCMs",
               f"Baseline level, {BASELINE_LABEL}"]
    fig.legend(handles, labels, ncol=4, loc="lower center", handlelength=2,
               columnspacing=1.2, borderaxespad=0.2)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.3, wspace=0.08)
    save_figure(fig, "03_high_risk_share")
    plt.close(fig)


# =============================================================================
# MONTHLY CLASSES
# =============================================================================

def monthly_shares(shares, model, scenario, period):
    """Share of the country (%) per calendar month (rows) and class (columns) of one model."""
    sub = shares[(shares.summary == model) & (shares.scenario == scenario) & (shares.period == period)]
    return 100 * np.array([sub[sub["map"] == f"month_{m}"].sort_values("class").share.to_numpy()
                           for m in range(1, 13)])


def stacked_bars(ax, values, reference=None):
    """Stacked classes per month, Very high at the bottom; reference: the class limits of
    another period as lines."""
    x = np.arange(12)
    values = values[:, ::-1]
    colors = CLASS_COLORS[::-1]
    bottom = np.zeros(12)
    for k, color in enumerate(colors):
        ax.bar(x, values[:, k], bottom=bottom, width=0.86, color=color, linewidth=0)
        bottom += values[:, k]
    if reference is not None:
        for limit in np.cumsum(reference[:, ::-1], axis=1)[:, :-1].T:
            ax.step(np.r_[x - 0.5, 11.5], np.r_[limit, limit[-1]], where="post", color="black",
                    linewidth=0.5)
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylim(0, 100)
    ax.set_xticks(x[::3], [calendar.month_abbr[m + 1] for m in x[::3]])     # Jan, Apr, Jul, Oct
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)


def classes_figure(shares):
    """Laid out as change_maps_figure: one group of columns per model, the baseline above, then
    SSPs in rows and periods in columns; tick labels on the outer panels only."""
    left, right, bottom, top, inner, between, hgap, h = 1.3, 0.1, 1.0, 0.8, 0.15, 0.4, 0.45, 2.4
    below_baseline = 0.35     # extra gap: the baseline row has month labels
    periods = len(FUTURE_PERIODS)
    w = ((WIDTH - left - right - (len(MODELS) - 1) * between - len(MODELS) * (periods - 1) * inner)
         / (len(MODELS) * periods))
    group_w = periods * w + (periods - 1) * inner
    rows = len(SSPS) + 1
    fig_h = bottom + top + rows * h + (rows - 1) * hgap + below_baseline
    fig = plt.figure(figsize=(cm(WIDTH), cm(fig_h)))

    def row_y(r):
        return bottom + (rows - 1 - r) * (h + hgap) + (below_baseline if r == 0 else 0)

    def add_bars(x, r, text, values, reference=None):
        ax = fig.add_axes([x / WIDTH, row_y(r) / fig_h, w / WIDTH, h / fig_h])
        stacked_bars(ax, values, reference)
        ax.text(0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom")
        return ax

    for m, model in enumerate(MODELS):
        x0 = left + m * (group_w + between)
        fig.text((x0 + group_w / 2) / WIDTH, (fig_h - 0.1) / fig_h,
                 f"({'abc'[m]}) {MODEL_STYLES[model]['label']}", ha="center", va="top")
        reference = monthly_shares(shares, model, HIST, "baseline")
        ax = add_bars(x0 + (group_w - w) / 2, 0, f"Baseline, {BASELINE_LABEL}", reference)
        if m == 0:
            ax.set_ylabel("Share of the\ncountry (%)")
        for r, scenario in enumerate(SSPS, start=1):
            for c, period in enumerate(FUTURE_PERIODS):
                ax = add_bars(x0 + c * (w + inner), r, period_label(period),
                              monthly_shares(shares, model, scenario, period), reference)
                ax.tick_params(labelleft=m == 0 and c == 0, labelbottom=r == len(SSPS))
    for r, scenario in enumerate(SSPS, start=1):
        fig.text(0.1 / WIDTH, (row_y(r) + h / 2) / fig_h, SCENARIOS[scenario]["label"], rotation=90,
                 ha="left", va="center")
    fig.legend(handles=class_handles() + [Line2D([], [], color="black", linewidth=0.5,
                                                 label="Class limits of the baseline")],
               ncol=len(LABELS) + 1, loc="lower center", bbox_to_anchor=(0.5, 0.1 / fig_h),
               handlelength=1.2, handleheight=0.9, columnspacing=1.0, borderaxespad=0)
    save_figure(fig, "03_classes_monthly")
    plt.close(fig)


# =============================================================================
# MAPS
# =============================================================================

def change_maps_figure(grid, cells):
    left, right, bottom, top, inner, between, hgap = 1.2, 0.1, 1.9, 0.8, 0.1, 0.35, 0.42
    periods = len(FUTURE_PERIODS)
    w = ((WIDTH - left - right - (len(MODELS) - 1) * between - len(MODELS) * (periods - 1) * inner)
         / (len(MODELS) * periods))
    h = w / map_ratio()
    group_w = periods * w + (periods - 1) * inner
    rows = len(SSPS) + 1
    fig_h = bottom + top + rows * h + (rows - 1) * hgap
    fig = plt.figure(figsize=(cm(WIDTH), cm(fig_h)))

    def row_y(r):
        return bottom + (rows - 1 - r) * (h + hgap)

    def add_map(x, r, text):
        ax = fig.add_axes([x / WIDTH, row_y(r) / fig_h, w / WIDTH, h / fig_h])
        ax.text(0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom")
        return ax

    for m, model in enumerate(MODELS):
        x0 = left + m * (group_w + between)
        fig.text((x0 + group_w / 2) / WIDTH, (fig_h - 0.1) / fig_h,
                 f"({'abc'[m]}) {MODEL_STYLES[model]['label']}", ha="center", va="top")
        ax = add_map(x0 + (group_w - w) / 2, 0, f"Baseline, {BASELINE_LABEL}")
        draw_map(ax, grid, cells.cell_id, cells[f"{model}_baseline_class"], CLASS_CMAP, left=False,
                 bottom=False, step=2)
        for r, scenario in enumerate(SSPS, start=1):
            for c, period in enumerate(FUTURE_PERIODS):
                ax = add_map(x0 + c * (w + inner), r, period_label(period))
                draw_map(ax, grid, cells.cell_id, cells[f"{model}_{scenario}_{period}_class_change"],
                         BINNED_CHANGE_SCALE[:2], left=m == 0 and c == 0, bottom=r == len(SSPS),
                         step=2)
    for r, scenario in enumerate(SSPS, start=1):
        fig.text(0.1 / WIDTH, (row_y(r) + h / 2) / fig_h, SCENARIOS[scenario]["label"], rotation=90,
                 ha="left", va="center")
    fig.legend(handles=class_handles(), ncol=3, loc="lower left",
               bbox_to_anchor=(left / WIDTH, 0.25 / fig_h), handlelength=1.2, handleheight=0.9,
               columnspacing=1.0, borderaxespad=0)
    change_colorbar(fig, [9.5 / WIDTH, 0.85 / fig_h, 7.2 / WIDTH, 0.28 / fig_h], BINNED_CHANGE_SCALE)
    save_figure(fig, "03_change_maps")
    plt.close(fig)


def monthly_change_figure(grid, months, model, scenario):
    """12 calendar-month maps (3 x 4) of the class change of one model and SSP."""
    left, bottom, top, wgap, hgap = 1.1, 1.7, 0.4, 0.3, 0.45
    w = (WIDTH - left - 0.2 - 3 * wgap) / 4
    h = w / map_ratio()
    fig_h = bottom + top + 3 * h + 2 * hgap
    fig = plt.figure(figsize=(cm(WIDTH), cm(fig_h)))
    column = months[f"{model}_{scenario}_class_change"].to_numpy()
    for m in range(1, 13):
        r, c = (m - 1) // 4, (m - 1) % 4
        ax = fig.add_axes([(left + c * (w + wgap)) / WIDTH, (bottom + (2 - r) * (h + hgap)) / fig_h,
                           w / WIDTH, h / fig_h])
        keep = months.month.to_numpy() == m
        draw_map(ax, grid, months.cell_id[keep], column[keep], CHANGE_SCALE[:2], left=c == 0,
                 bottom=r == 2, step=2)
        ax.text(0.0, 1.02, calendar.month_name[m], transform=ax.transAxes, ha="left", va="bottom")
    change_colorbar(fig, [left / WIDTH, 0.85 / fig_h, 0.6, 0.28 / fig_h], CHANGE_SCALE)
    fig.legend(handles=border_handles(), loc="lower right", bbox_to_anchor=(0.99, 0.35 / fig_h),
               borderaxespad=0)
    save_figure(fig, f"03_change_monthly_{model}_{scenario}", appendix=True)
    plt.close(fig)


def main():
    setup_style()
    grid = reference_grid()
    high_risk_figure(pd.read_csv(IN_DIR / "national_yearly.csv"),
                     pd.read_csv(IN_DIR / "high_risk_shares.csv"))
    classes_figure(pd.read_csv(IN_DIR / "class_shares.csv"))
    change_maps_figure(grid, pd.read_parquet(IN_DIR / "cells.parquet"))
    months = pd.read_parquet(IN_DIR / "cell_months.parquet")
    for model in MODELS:
        for scenario in SSPS:
            monthly_change_figure(grid, months, model, scenario)


if __name__ == "__main__":
    main()
