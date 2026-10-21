"""
02_resampling_plot.py
---------------------
Figure of the resampling step: the test scores of each selected algorithm trained with
each sampler, from the table of 02_resampling_evaluate.py.

One panel per metric (peak MCC, balanced accuracy at the peak, average precision, ROC-AUC),
each with its axis zoomed to the range of the values, so that small differences between
samplers are visible (the no-skill values lie outside these ranges). One band per algorithm; in
it, one marker per sampler at a fixed height (circles: oversampling, squares:
undersampling), so the best sampler of an algorithm is the rightmost marker of its band.
The black vertical line in each band is the reference: the algorithm with balanced class
weights and no sampler (its model-testing result).

Input:  results/<model_training resampling>/metrics.csv
Output: results/mainfigs/02_resampling.pdf

Usage:
    python 02_resampling_plot.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from figures import MODELS, SAMPLERS, add_panel_label, cm, save_figure, setup_style

METRICS_FILE = cfg.model_training.paths.results.resampling / "metrics.csv"
REFERENCE = "class_weights"
BAND = 0.62             # height of a model's band (models are 1 apart)

PAD = 0.12              # margin around the values, as a share of their range

# column of metrics.csv, title
METRICS = [
    ("peak_mcc",          "Peak MCC"),
    ("balanced_accuracy", "Balanced\naccuracy"),
    ("average_precision", "Average\nprecision"),
    ("roc_auc",           "ROC-AUC"),
]


def plot_metric(ax, table, models, column, title):
    samplers = [s for s in SAMPLERS if s in set(table.sampler)]
    offsets = np.linspace(-BAND / 2, BAND / 2, len(samplers))
    for i, model in enumerate(models):
        rows = table[table.model == model].set_index("sampler")[column]
        ax.vlines(rows[REFERENCE], i - BAND / 2 - 0.08, i + BAND / 2 + 0.08, color="black", lw=0.9, zorder=1)
        for sampler, dy in zip(samplers, offsets):
            if sampler in rows:
                style = SAMPLERS[sampler]
                ax.plot(rows[sampler], i + dy, style["marker"], color=style["color"], ms=3.6,
                        mec="0.25", mew=0.3, zorder=2)
        if i:
            ax.axhline(i - 0.5, color="0.9", lw=0.5, zorder=0)
    values = table[table.model.isin(models)][column]
    span = values.max() - values.min()
    ax.set_xlim(values.min() - PAD * span, values.max() + PAD * span)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_title(title, fontsize=cfg.figures.font_size, va="bottom")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def legend_handles(table):
    """Handles in the order that puts the oversamplers in the first and the undersamplers in
    the second row of a 4-column legend (filled column by column), the reference last."""
    marker = {key: Line2D([], [], ls="none", marker=s["marker"], color=s["color"], mec="0.25",
                          mew=0.3, ms=3.6, label=s["label"])
              for key, s in SAMPLERS.items() if key in set(table.sampler)}
    order = [k for pair in zip(["ros", "smote", "adasyn"], ["rus", "tomek", "ncr"]) for k in pair]
    handles = [marker[k] for k in order if k in marker]
    return handles + [Line2D([], [], color="black", lw=0.9, label="Class weights (reference)")]


def main():
    setup_style()
    table = pd.read_csv(METRICS_FILE)
    models = [m for m in cfg.model_training.models if m in set(table.model)]

    fig, axes = plt.subplots(1, len(METRICS), sharey=True,
                             figsize=(cm(cfg.figures.width_cm.full), cm(7.0)),
                             gridspec_kw={"wspace": 0.18, "left": 0.13, "right": 0.99,
                                          "bottom": 0.27, "top": 0.88})
    for ax, (column, title) in zip(axes, METRICS):
        plot_metric(ax, table, models, column, title)
    axes[0].set_yticks(range(len(models)), [MODELS[m]["label"] for m in models])
    for k, ax in enumerate(axes):
        add_panel_label(ax, k, x=-0.05, y=1.12)
    fig.legend(handles=legend_handles(table), loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.0),
               handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    save_figure(fig, "02_resampling")


if __name__ == "__main__":
    main()
