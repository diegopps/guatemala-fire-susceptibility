"""
01_model_testing_plot.py
------------------------
Figure of the model testing (step 01): the test scores of the candidate algorithms, from
the tables of 01_model_testing_evaluate.py.

(a) Peak MCC, balanced accuracy at the peak, average precision and ROC-AUC of each
    algorithm, sorted by peak MCC; the colours identify the algorithms in both panels.
    Dashed lines: the no-skill values (MCC 0, balanced accuracy and ROC-AUC 0.5, average
    precision the fire share).
(b) MCC against the share of test cell-months predicted as fire, i.e. the MCC curve
    with each probability threshold expressed as the share of cell-months at or above it.
    Unlike the threshold itself, this share means the same for every algorithm, whatever
    the scale of its probabilities. Dots mark the peak MCC. A curve ends where the next
    lower threshold is a probability of 0, i.e. where the algorithm cannot rank the
    remaining cell-months.

Input:  results/<model_training model_testing>/metrics.csv and <model>/threshold_curve.csv
Output: results/mainfigs/02_model_testing.pdf

Usage:
    python 01_model_testing_plot.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from figures import MODELS, add_panel_label, cm, save_figure, setup_style

RESULTS = cfg.model_training.paths.results.model_testing
SHARE_RANGE = (3e-3, 50)        # % of cell-months shown in (a)

# (a): column of metrics.csv, title, axis range (None: from the data), ticks, no-skill
# value (None: the fire share)
METRICS = [
    ("peak_mcc",          "Peak MCC",            (0, None),     [0, 0.05],      0.0),
    ("balanced_accuracy", "Balanced\naccuracy", (0.5, None),   [0.5, 0.7],     0.5),
    ("average_precision", "Average\nprecision", (0, None),     [0, 0.01],      None),
    ("roc_auc",           "ROC-AUC",             (0.5, 1.12),   [0.5, 0.75, 1], 0.5),
]


def load():
    metrics = pd.read_csv(RESULTS / "metrics.csv").sort_values("peak_mcc", ascending=False)
    curves = {}
    for model in metrics.model:
        c = pd.read_csv(RESULTS / model / "threshold_curve.csv")
        c = c[c.threshold > 0]                       # the jump to flagging every cell-month
        n = c.tp + c.fp + c.fn + c.tn
        curves[model] = c.assign(share=100 * (c.tp + c.fp) / n)
    return metrics.reset_index(drop=True), curves


def plot_curves(ax, metrics, curves):
    for model in metrics.model:
        c, style = curves[model], MODELS[model]
        ax.plot(c.share, c.mcc, color=style["color"], lw=0.8, label=style["label"])
        peak = c.loc[c.mcc.idxmax()]
        ax.plot(peak.share, peak.mcc, "o", color=style["color"], ms=3, mec="white", mew=0.4)
    ax.set_xscale("log")
    ax.set_xlim(SHARE_RANGE)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Test cell-months predicted as fire (%)")
    ax.set_ylabel("MCC")


def plot_dots(axes, metrics):
    y = range(len(metrics))
    colors = [MODELS[m]["color"] for m in metrics.model]
    no_skill_ap = metrics.positive_share.iloc[0]
    for ax, (column, title, (lo, hi), ticks, ref) in zip(axes, METRICS):
        values = metrics[column]
        ref = no_skill_ap if ref is None else ref
        ax.axvline(ref, color="0.6", lw=0.5, ls="--", zorder=0)
        ax.hlines(y, ref, values, color="0.85", lw=0.6, zorder=1)
        ax.scatter(values, y, c=colors, s=14, zorder=2, edgecolors="white", linewidths=0.4)
        ax.set_xlim(lo, hi if hi is not None else values.max() + 0.6 * (values.max() - lo))
        for v, yi in zip(values, y):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=cfg.figures.font_size - 2)
        ax.set_title(title, fontsize=cfg.figures.font_size, va="bottom")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.tick_params(axis="y", length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(list(y), [MODELS[m]["label"] for m in metrics.model])
    axes[0].set_ylim(len(metrics) - 0.5, -0.5)


def main():
    setup_style()
    metrics, curves = load()
    fig = plt.figure(figsize=(cm(cfg.figures.width_cm.full), cm(7.0)))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.32,
                            left=0.15, right=0.99, bottom=0.15, top=0.86)
    dots = grid[0].subgridspec(1, len(METRICS), wspace=0.3)
    ax_curve = fig.add_subplot(grid[1])
    ax_dots = [fig.add_subplot(dots[0])]
    ax_dots += [fig.add_subplot(dots[k], sharey=ax_dots[0]) for k in range(1, len(METRICS))]
    for ax in ax_dots[1:]:
        ax.tick_params(axis="y", labelleft=False)

    plot_curves(ax_curve, metrics, curves)
    plot_dots(ax_dots, metrics)
    add_panel_label(ax_dots[0], 0, x=-1.5, y=1.1)
    add_panel_label(ax_curve, 1, x=-0.16, y=1.1)
    save_figure(fig, "02_model_testing")


if __name__ == "__main__":
    main()
