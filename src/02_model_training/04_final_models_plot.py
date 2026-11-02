"""
04_final_models_plot.py
-----------------------
Figures of the final models (step 04) on the validation subset, from the tables of
04_final_models_evaluate.py.

Main figure 02_final_models_validation.pdf (full width)
(a) Validation scores per model over all cell-months (labelled with its correction of the
    class imbalance): MCC and balanced accuracy at the committed threshold; average precision;
    ROC-AUC. Circles, labelled with their value: the scores; whiskers: 95% bootstrap
    intervals. Dashed lines: the no-skill values (MCC 0, balanced accuracy and ROC-AUC 0.5,
    average precision the fire share); the pale stems start there.
(b) Top-k coverage over all cell-months: share of the fire cell-months among the k% of the
    cell-months with the highest probability. Dashed line: a model without skill (captures
    k%).

Appendix figure 02_final_models_importance.pdf (single column)
Permutation importance of the predictor groups: decrease of the AP when the predictors of a
group are permuted, in % of the unpermuted AP of the model; bars: mean over the repeats,
error bars: standard deviation. Groups sorted by the mean over the models.

Input:  results/<model_training final_models>/metrics.csv, topk.csv, importance.csv
Output: results/mainfigs/02_final_models_validation.pdf
        results/appendix/02_final_models_importance.pdf

Usage:
    python 04_final_models_plot.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from figures import (IMBALANCE_LABELS, MODELS, SCORES, add_panel_label, cm, save_figure,
                     score_axis, setup_style)

TABLES = cfg.model_training.paths.results.final_models
NAMES = {"validation": "02_final_models_validation", "importance": "02_final_models_importance"}
MARKER = 3.6            # marker size (points)
BAR = 0.26              # height of a model's bar in the importance figure (groups are 1 apart)
GREY = "0.35"           # neutral colour of the legend markers
LEGEND = {"handlelength": 1.4, "columnspacing": 1.5, "handletextpad": 0.4}
SCOPE_LABEL = "All months"

# Scores of panel (a), keyed as the columns of metrics.csv: title and no-skill value (None:
# the fire share)
VALIDATION_SCORES = {
    "mcc":               ("MCC",                 0.0),
    "balanced_accuracy": SCORES["balanced_accuracy"],
    "average_precision": SCORES["average_precision"],
    "roc_auc":           SCORES["roc_auc"],
}
SCORE_TICKS = {"mcc": [0, 0.03, 0.06], "balanced_accuracy": [0.5, 0.6],
               "average_precision": [0, 0.01], "roc_auc": [0.5, 0.7, 0.9]}


def model_labels(models):
    return [f"{MODELS[m]['label']}\n({IMBALANCE_LABELS[cfg.model_training.samplers[m]]})"
            for m in models]


# =============================================================================
# MAIN FIGURE: VALIDATION SCORES AND TOP-K COVERAGE
# =============================================================================

def plot_scores(axes, metrics, models):
    rows = metrics[metrics.scope == "all"].set_index("model").loc[models]
    y = range(len(models))
    colors = [MODELS[m]["color"] for m in models]
    for ax, (column, (title, ref)) in zip(axes, VALIDATION_SCORES.items()):
        values, highs = rows[column], rows[f"{column}_high"]
        ref = rows.fire_share.iloc[0] if ref is None else ref
        ax.axvline(ref, color="0.6", lw=0.5, ls="--", zorder=0)
        ax.hlines(y, ref, values, color="0.85", lw=0.6, zorder=1)
        ax.hlines(y, rows[f"{column}_low"], highs, color=colors, lw=0.8, zorder=2)
        ax.scatter(values, y, c=colors, s=MARKER ** 2, lw=0, zorder=3)
        for v, yi in zip(values, y):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(0, 3.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=cfg.figures.font_size - 2)
        start = 0.0 if column == "average_precision" else ref    # AP from 0, not the fire share
        ax.set_xlim(start, min(highs.max() + 0.15 * (highs.max() - start), 1.0))
        ax.set_xticks(SCORE_TICKS[column], [f"{t:g}" for t in SCORE_TICKS[column]])
        score_axis(ax, title)


def plot_topk(ax, topk, models):
    ks = sorted(topk.k_percent.unique())
    ax.plot([0, max(ks)], [0, max(ks)], color="0.6", lw=0.5, ls="--", zorder=0)
    for model in models:
        rows = topk[(topk.model == model) & (topk.scope == "all")].sort_values("k_percent")
        ax.plot(rows.k_percent, 100 * rows.fire_captured_share, "-o", ms=MARKER - 0.8,
                lw=0.8, color=MODELS[model]["color"], mec="white", mew=0.3, zorder=2)
    ax.text(0.02, 0.98, SCOPE_LABEL, transform=ax.transAxes, ha="left", va="top",
            fontsize=cfg.figures.font_size - 1)
    ax.set_xticks(ks)
    ax.set_xlim(0, max(ks) * 1.05)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color="0.9", lw=0.5, zorder=0)   # guides to read off the values
    ax.set_axisbelow(True)
    ax.set_xlabel("Cell-months flagged (%)")
    ax.set_ylabel("Fire cell-months\ncaptured (%)")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def legend_scores():
    return [Line2D([], [], ls="none", marker="o", ms=MARKER, color=GREY, mew=0,
                   label=SCOPE_LABEL),
            Line2D([], [], color=GREY, lw=0.8, label="95% bootstrap interval")]


def legend_topk(models):
    handles = [Line2D([], [], color=MODELS[m]["color"], lw=0.8, marker="o", ms=MARKER - 0.8,
                      mec="white", mew=0.3, label=MODELS[m]["label"]) for m in models]
    return handles + [Line2D([], [], color="0.6", lw=0.5, ls="--", label="No skill")]


def figure_validation(metrics, topk, models):
    fig = plt.figure(figsize=(cm(cfg.figures.width_cm.full), cm(12.0)))
    top = fig.add_gridspec(1, len(VALIDATION_SCORES), wspace=0.25,
                           left=0.22, right=0.98, bottom=0.62, top=0.92)
    bottom = fig.add_gridspec(1, 1, left=0.22, right=0.98, bottom=0.14, top=0.44)
    ax_scores = [fig.add_subplot(top[0])]
    ax_scores += [fig.add_subplot(top[k], sharey=ax_scores[0])
                  for k in range(1, len(VALIDATION_SCORES))]
    ax_topk = fig.add_subplot(bottom[0])

    plot_scores(ax_scores, metrics, models)
    for ax in ax_scores[1:]:
        ax.tick_params(axis="y", labelleft=False)
    ax_scores[0].set_yticks(range(len(models)), model_labels(models))
    ax_scores[0].set_ylim(len(models) - 0.5, -0.5)
    add_panel_label(ax_scores[0], 0, x=-1.25, y=1.12)
    left, right = ax_scores[0].get_position(), ax_scores[-1].get_position()
    fig.legend(handles=legend_scores(), loc="upper center", ncol=2,
               bbox_to_anchor=((left.x0 + right.x1) / 2, left.y0 - 0.06), **LEGEND)

    plot_topk(ax_topk, topk, models)
    add_panel_label(ax_topk, 1, x=-0.26, y=1.02)
    box = ax_topk.get_position()
    fig.legend(handles=legend_topk(models), loc="lower center", ncol=len(models) + 1,
               bbox_to_anchor=((box.x0 + box.x1) / 2, 0.0), **LEGEND)
    return fig


# =============================================================================
# APPENDIX FIGURE: PERMUTATION IMPORTANCE
# =============================================================================

def figure_importance(importance, models):
    fig, ax = plt.subplots(figsize=(cm(cfg.figures.width_cm.single), cm(7.0)),
                           gridspec_kw={"left": 0.25, "right": 0.97, "bottom": 0.3, "top": 0.97})
    groups = importance[importance.kind == "group"].copy()
    groups["percent_std"] = 100 * groups.drop_std / groups.average_precision_base
    order = (groups.groupby("name", sort=False).drop_relative_percent.mean()
             .sort_values(ascending=False).index)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * BAR
    for k, model in enumerate(models):
        rows = groups[groups.model == model].set_index("name").reindex(order)
        ax.barh(np.arange(len(order)) + offsets[k], rows.drop_relative_percent, height=BAR,
                color=MODELS[model]["color"], lw=0, zorder=2)
        ax.errorbar(rows.drop_relative_percent, np.arange(len(order)) + offsets[k],
                    xerr=rows.percent_std, fmt="none", ecolor="black", elinewidth=0.5, capsize=1,
                    capthick=0.5, zorder=3)
    ax.axvline(0, color="black", lw=0.5, zorder=1)
    ax.set_yticks(range(len(order)), [g.capitalize() for g in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Decrease in average precision when the\ngroup is permuted (% of unpermuted)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    score_axis(ax, "")
    fig.legend(handles=[Patch(color=MODELS[m]["color"], lw=0, label=MODELS[m]["label"])
                        for m in models],
               loc="lower center", ncol=len(models), bbox_to_anchor=(0.55, 0.0), **LEGEND)
    return fig


def main():
    setup_style()
    metrics = pd.read_csv(TABLES / "metrics.csv")
    topk = pd.read_csv(TABLES / "topk.csv")
    models = [m for m in cfg.model_training.models if m in set(metrics.model)]
    save_figure(figure_validation(metrics, topk, models), NAMES["validation"])
    importance_file = TABLES / "importance.csv"
    if importance_file.exists():
        importance = pd.read_csv(importance_file)
        save_figure(figure_importance(importance, models), NAMES["importance"], appendix=True)
    else:
        print(f"{importance_file} not found: no importance figure.")


if __name__ == "__main__":
    main()
