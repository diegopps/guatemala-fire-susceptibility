"""
04_correlation.py
-----------------
Spearman rank correlation between the 14 predictors, computed on all rows of the
training set (all GCMs, cells and months; ties get their average rank), shown as a
lower-triangle heatmap.

Input:  paths.inputs.training_set
Output: results/appendix/01_correlation_spearman.pdf
        results/<data_exploration>/correlation_spearman.csv — the correlation matrix.
        Reused unless --recompute is given.

Usage:
    python 04_correlation.py [--recompute]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from figures import VARIABLES, cm, save_figure, setup_style

MATRIX_FILE = cfg.paths.results.data_exploration / "correlation_spearman.csv"
ORDER = [v for v in VARIABLES if v in cfg.data.features]     # display order


def spearman_matrix():
    """Spearman correlation = Pearson correlation of the ranks, one column read at a time."""
    path = cfg.paths.inputs.training_set
    n = pq.ParquetFile(path).metadata.num_rows
    z = np.empty((n, len(ORDER)))
    for k, var in enumerate(ORDER):
        values = pq.read_table(path, columns=[var]).column(var).to_numpy()
        r = rankdata(values)
        z[:, k] = (r - r.mean()) / r.std()
        print(f"  ranked {var}")
    return pd.DataFrame(z.T @ z / n, index=ORDER, columns=ORDER)


def load_matrix(recompute):
    if MATRIX_FILE.exists() and not recompute:
        print(f"Reading {MATRIX_FILE}")
        return pd.read_csv(MATRIX_FILE, index_col=0)
    print("Computing the Spearman correlation on the training set...")
    corr = spearman_matrix()
    MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(MATRIX_FILE, float_format="%.6f")
    print(f"  Saved: {MATRIX_FILE}")
    return corr


def plot(corr):
    setup_style()
    names = [VARIABLES[v][0] for v in ORDER]
    n = len(ORDER)
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    values = np.where(lower, corr.loc[ORDER, ORDER].to_numpy(), np.nan)[1:, :-1]

    fig, ax = plt.subplots(figsize=(cm(14), cm(12)))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="none")
    for (r, c), v in np.ndenumerate(values):
        if np.isfinite(v):
            text = f"{round(v, 2) + 0.0:.2f}".replace("-", "−")   # + 0.0: no "-0.00"
            ax.text(c, r, text, ha="center", va="center",
                    fontsize=cfg.figures.font_size - 1.5,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(n - 1), names[:-1], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n - 1), names[1:])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Spearman rank correlation coefficient")
    cbar.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.2, top=0.98)
    save_figure(fig, "01_correlation_spearman", appendix=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recompute", action="store_true", help="recompute the correlation matrix")
    args = parser.parse_args()

    corr = load_matrix(args.recompute)
    pairs = corr.where(np.tril(np.ones(corr.shape, dtype=bool), k=-1)).stack()
    strong = pairs[pairs.abs() > 0.6].sort_values(key=abs, ascending=False)
    print("Pairs with |rho| > 0.6:")
    print(strong.round(2).to_string())
    plot(corr)


if __name__ == "__main__":
    main()
