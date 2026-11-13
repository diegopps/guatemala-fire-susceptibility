"""
kde_baseline.py
---------------
A model-free baseline from the fire records alone, for the cross-validation test of step 04b
(src/02_model_training/04b_cv_test*.py): the score of cell c in calendar month m is

    KDE_h(c) x s(m)

  KDE_h(c)  Gaussian kernel density (per km^2) of the training fire cell-months at the centre
            of cell c: (1 / (N 2 pi h^2)) sum_j exp(-d_cj^2 / (2 h^2)), j over the N training
            fire cell-months (a cell with several counts once per fire cell-month), d the
            straight-line distance between the cell centres (3-D positions of
            utils/cv_folds.Cells; equal to the great-circle distance to < 1 m at these
            distances), no truncation
  s(m)      share of the training fire cell-months in calendar month m

The scores of different folds are normalised by their own number of training fires.

Usage:
    from kde_baseline import FireHistory, scores
    history = FireHistory(cells, fire_cell_ids, fire_months)
    p = scores(cells, history, [1.0, 3.0], cell_ids, months)   # {h: score of each cell-month}
"""

import numpy as np

CHUNK = 1000                 # cells per distance block


class FireHistory:
    """Training fire cell-months per cell and per calendar month."""

    def __init__(self, cells, cell_ids, months):
        pos = cells.pos(cell_ids)
        months = np.asarray(months, dtype=np.int64)
        self.n_fire_cell_months = pos.size
        self.count = np.bincount(pos, minlength=cells.n)
        self.has = self.count > 0
        self.month_share = np.bincount(months, minlength=13)[1:] / pos.size   # s(m), m = 1..12


def density(cells, history, bandwidths):
    """{h: KDE_h of every cell}."""
    src = np.flatnonzero(history.has)
    w = history.count[src].astype(np.float64)
    b = cells.xyz[src]
    out = {h: np.zeros(cells.n) for h in bandwidths}
    for start in range(0, cells.n, CHUNK):
        a = cells.xyz[start:start + CHUNK]
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        for h in bandwidths:
            out[h][start:start + CHUNK] = np.exp(-d2 / (2 * h * h)) @ w
    return {h: v / (history.n_fire_cell_months * 2 * np.pi * h * h) for h, v in out.items()}


def scores(cells, history, bandwidths, cell_ids, months):
    """{h: KDE_h(c) x s(m) of each cell-month (cell_ids, months)}."""
    dens = density(cells, history, bandwidths)
    pos = cells.pos(cell_ids)
    s = history.month_share[np.asarray(months, dtype=np.int64) - 1]
    return {h: dens[h][pos] * s for h in bandwidths}
