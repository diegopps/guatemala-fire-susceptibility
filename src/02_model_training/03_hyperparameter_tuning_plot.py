"""
03_hyperparameter_tuning_plot.py
--------------------------------
Figure of the hyperparameter tuning (step 03), from the tables of
03_hyperparameter_tuning_evaluate.py. One band per selected algorithm, labelled with its
correction of the class imbalance (model_training.samplers).

(a) Test scores of the base and the tuned settings: peak MCC, balanced accuracy at the peak,
    average precision and ROC-AUC. Open circle: base (configuration 0, the settings of
    utils/models.py, i.e. the model of step 02); filled circle: tuned (the selected
    configuration, refitted on the whole train subset); the line joins the two. Each value
    is written on the outer side of its marker. Dashed lines: the no-skill values (MCC 0,
    balanced accuracy and ROC-AUC 0.5, average precision the fire share).
(b) The search: mean peak MCC over the CV folds of every configuration (grey dots, spread
    vertically at random within the band so that they do not overlap). Black line: the
    defaults (configuration 0), with their mean ± one standard deviation over the folds
    shaded; filled circle: the selected configuration, with the same range as error bar.
(a) is drawn above (b), both over the full width.

Input:  results/<model_training hyperparameter_tuning>/metrics.csv and <model>/cv_summary.csv
Output: results/mainfigs/02_hyperparameter_tuning.pdf
        (--trial: the figure of the trial runs, in their results folder)

Usage:
    python 03_hyperparameter_tuning_plot.py [--trial]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from figures import (IMBALANCE_LABELS, MODELS, SCORES, add_panel_label, cm, save_figure,
                     score_axis, setup_style)
import tuning

NAME = "02_hyperparameter_tuning"
MARKER = 3.6            # marker size (points)
BAND = 0.6              # height of a model's band in (b) (models are 1 apart)
SEED = 0                # vertical spread of the configurations in (b)


def load(trial):
    results = tuning.directories(trial)[1]
    metrics = pd.read_csv(results / "metrics.csv")
    models = [m for m in cfg.model_training.models if m in set(metrics.model)]
    summaries = {m: pd.read_csv(results / m / "cv_summary.csv") for m in models}
    return metrics, models, summaries


def model_labels(metrics, models):
    sampler = metrics.groupby("model").sampler.first()
    return [f"{MODELS[m]['label']}\n({IMBALANCE_LABELS[sampler[m]]})" for m in models]


def plot_scores(axes, metrics, models):
    shares = metrics.positive_share.dropna() if "positive_share" in metrics else pd.Series(dtype=float)
    font = cfg.figures.font_size - 2
    for ax, (column, (title, ref)) in zip(axes, SCORES.items()):
        ref = shares.iloc[0] if ref is None and len(shares) else ref
        if ref is not None:
            ax.axvline(ref, color="0.6", lw=0.5, ls="--", zorder=0)
        values = []
        for i, model in enumerate(models):
            rows = metrics[metrics.model == model].set_index("setting")
            base = rows[column].get("base", np.nan) if column in rows else np.nan
            tuned = rows[column].get("tuned", np.nan) if column in rows else np.nan
            color = MODELS[model]["color"]
            pair = [v for v in (base, tuned) if np.isfinite(v)]
            if len(pair) == 2:
                ax.plot(pair, [i, i], color="0.6", lw=0.8, zorder=1)
            if np.isfinite(base):
                ax.plot(base, i, "o", ms=MARKER, mfc="white", mec=color, mew=0.8, zorder=2)
            if np.isfinite(tuned):
                ax.plot(tuned, i, "o", ms=MARKER, color=color, mec="white", mew=0.4, zorder=3)
            # values on the outer side of their markers (one label if they are equal)
            labels = sorted({round(v, 3) for v in pair})
            for k, v in enumerate(labels):
                left = len(labels) == 2 and k == 0
                ax.annotate(f"{v:.3f}", (v, i), xytext=(-3 if left else 3, 0), textcoords="offset points",
                            ha="right" if left else "left", va="center", fontsize=font)
            values += pair
        start = 0.0 if column == "average_precision" else ref     # axis starts at no skill
        if values:
            ax.set_xlim(start, max(values) + 0.7 * (max(values) - start))
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        score_axis(ax, title)


def plot_search(ax, models, summaries):
    rng = np.random.default_rng(SEED)
    for i, model in enumerate(models):
        s = summaries[model].set_index("config")
        color = MODELS[model]["color"]
        others = s.drop(index=0).peak_mcc_mean.dropna()
        ax.scatter(others, i + rng.uniform(-BAND / 2, BAND / 2, others.size), s=3, color="0.65",
                   linewidths=0, zorder=1)
        default = s.loc[0]
        if np.isfinite(default.peak_mcc_mean):
            span = [i - BAND / 2 - 0.08, i + BAND / 2 + 0.08]
            ax.fill_betweenx(span, default.peak_mcc_mean - default.peak_mcc_std,
                             default.peak_mcc_mean + default.peak_mcc_std, color="0.9", lw=0, zorder=0)
            ax.vlines(default.peak_mcc_mean, *span, color="black", lw=0.9, zorder=2)
        ranked = s[s["rank"] == 1]
        if len(ranked):
            best = ranked.iloc[0]
            ax.errorbar(best.peak_mcc_mean, i, xerr=best.peak_mcc_std, fmt="o", ms=MARKER,
                        color=color, mec="white", mew=0.4, elinewidth=0.6, capsize=1.2,
                        capthick=0.6, zorder=3)
        if i:
            ax.axhline(i - 0.5, color="0.9", lw=0.5, zorder=0)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlabel(f"Mean peak MCC over the {cfg.model_training.tuning.folds} CV folds")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    score_axis(ax, "")


def legend_handles():
    grey = "0.35"
    return [Line2D([], [], ls="none", marker="o", ms=MARKER, mfc="white", mec=grey, mew=0.8,
                   label="Base (default settings)"),
            Line2D([], [], ls="none", marker="o", ms=MARKER, color=grey, mec="white", mew=0.4,
                   label="Tuned (selected configuration)"),
            Line2D([], [], ls="none", marker="o", ms=2, color="0.65", label="Other configurations"),
            Line2D([], [], color="black", lw=0.9, label="Default settings (mean ± SD shaded)")]


def main(trial):
    setup_style()
    metrics, models, summaries = load(trial)
    labels = model_labels(metrics, models)
    fig = plt.figure(figsize=(cm(cfg.figures.width_cm.full), cm(12.0)))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.45,
                            left=0.22, right=0.98, bottom=0.17, top=0.92)
    scores = grid[0].subgridspec(1, len(SCORES), wspace=0.25)
    ax_scores = [fig.add_subplot(scores[0])]
    ax_scores += [fig.add_subplot(scores[k], sharey=ax_scores[0]) for k in range(1, len(SCORES))]
    ax_search = fig.add_subplot(grid[1])

    plot_scores(ax_scores, metrics, models)
    plot_search(ax_search, models, summaries)
    for ax in ax_scores[1:]:
        ax.tick_params(axis="y", labelleft=False)
    ax_scores[0].set_yticks(range(len(models)), labels)
    ax_scores[0].set_ylim(len(models) - 0.5, -0.5)
    ax_search.set_yticks(range(len(models)), labels)
    add_panel_label(ax_scores[0], 0, x=-1.25, y=1.12)
    add_panel_label(ax_search, 1, x=-0.27, y=1.02)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.6, 0.0),
               handlelength=1.4, columnspacing=1.5, handletextpad=0.4)
    if trial:
        path = tuning.directories(trial)[1] / f"{NAME}.{cfg.figures.format}"
        fig.savefig(path, dpi=cfg.figures.dpi)
        print(f"  Saved: {path}")
    else:
        save_figure(fig, NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trial", action="store_true", help="figure of the trial runs")
    main(parser.parse_args().trial)
