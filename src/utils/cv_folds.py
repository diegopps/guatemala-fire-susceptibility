"""
cv_folds.py
-----------
Data and folds of the cross-validation test of the final configurations (step 04b,
src/02_model_training/04b_cv_test*.py, config [model_training.cv_test]): all cell-months of
the training years, with the train, test and validation subsets pooled, split into folds by
year or by spatial block; and the spatial autocorrelation that sets the block size.

Combined model matrices: the arrays of the three subsets of utils/model_data.py concatenated
in the order train, test, validation (each in its own row order), written once to a
directory as <array>.npy with done.json (rows and offset of each subset). load_combined()
returns them memory-mapped, as model_data.load_subset() does for one subset, so that
utils/fitting.py and utils/resampling.py work on them unchanged. The cell-months are those of
model_data.cell_month_rows() of the combined arrays.

Cells: the grid cells of the model data, sorted by cell id, with the coordinates of the
reference grid (chelsa_grid.reference_grid(), the cell centres of the training set), their
indices on the 30'' grid and 3-D positions on a sphere of radius R_EARTH (for distances).

Folds:
  year_folds     one fold per year of training.years (the index of the year)
  block_folds    squares of block_km on a local equirectangular projection at the mean
                 latitude of the cells (x = R cos(lat0) lon, y = R lat), origin at the minimum
                 x and y of the cell centres; a cell belongs to the square that contains its
                 centre, and all its cell-months to the fold of the square. The squares are
                 shuffled and dealt round-robin to n folds (equal numbers of squares, +-1);
                 of `draws` shuffles, the one with the lowest max/min ratio of fire
                 cell-months per fold is kept (ties: the first). The shuffles come from
                 numpy's default_rng(seed); with warmup_km, the generator first makes `draws`
                 permutations of the squares of that size, which are discarded.

Autocorrelation (indicator_correlation): the correlation of a 0/1 indicator per cell (e.g.
"burned at least once") between cells at distance d, over all pairs of cells, by distance
bin: (P(both) - mu^2) / (mu (1 - mu)), with P(both) the share of the pairs in the bin with
both cells burned and mu the share of burned cells among the pair members. The pair counts
come from FFT cross-correlations of the grid rasters; grid offsets are converted to km with
the same local projection.

Usage:
    import cv_folds
    cv_folds.combine_subsets(directory)
    data = cv_folds.load_combined(directory)        # dict of memory-mapped arrays
    cells = cv_folds.Cells(cell_ids)
    folds, info = cv_folds.block_folds(cells, fire_cell_ids, 30, 5, 1000, 107, warmup_km=15)
"""

import json

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.signal import fftconvolve

from load_config import cfg
from chelsa_grid import reference_grid
import model_data

R_EARTH = 6371.0088          # km, mean Earth radius
STEP_DEG = 1 / 120           # 30 arc-seconds
COPY_ROWS = 2_000_000        # rows per copy when combining the subsets


# =============================================================================
# COMBINED MODEL MATRICES
# =============================================================================

def combine_subsets(directory):
    """Writes the arrays of the train, test and validation subsets, concatenated, to
    directory/<array>.npy and done.json; returns the contents of done.json."""
    meta = model_data.load_meta()
    directory.mkdir(parents=True, exist_ok=True)
    subsets = {s: model_data.load_subset(s) for s in model_data.SUBSETS}
    n = {s: len(d["y"]) for s, d in subsets.items()}
    if n != meta["rows"]:
        raise ValueError(f"Rows {n} differ from meta.json {meta['rows']}.")
    total = sum(n.values())
    for name, dtype in model_data.ARRAYS.items():
        shape = (total, len(cfg.data.features)) if name == "X" else (total,)
        out = open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        offset = 0
        for s in model_data.SUBSETS:
            src = subsets[s][name]
            for a in range(0, n[s], COPY_ROWS):
                b = min(a + COPY_ROWS, n[s])
                out[offset + a:offset + b] = src[a:b]
            offset += n[s]
        out.flush()
        del out
    offsets = np.cumsum([0] + [n[s] for s in model_data.SUBSETS[:-1]])
    info = {"rows": n, "offsets": dict(zip(model_data.SUBSETS, map(int, offsets))),
            "total": total}
    (directory / "done.json").write_text(json.dumps(info, indent=2))
    return info


def combined_info(directory):
    """done.json of the combined arrays, checked against the current model data."""
    path = directory / "done.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 04b_cv_test.py folds first.")
    info = json.loads(path.read_text())
    if info["rows"] != model_data.load_meta()["rows"]:
        raise RuntimeError(f"{directory}: the model data changed since the arrays were "
                           "combined: rerun 04b_cv_test.py folds --force.")
    return info


def load_combined(directory, arrays=tuple(model_data.ARRAYS)):
    """The combined arrays as a dict (memory-mapped, read-only), like model_data.load_subset."""
    combined_info(directory)
    return {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in arrays}


def subset_of_rows(rows, info):
    """Subset name of each row index of the combined arrays."""
    off = info["offsets"]
    return np.where(rows < off["test"], "train",
                    np.where(rows < off["validation"], "test", "validation"))


# =============================================================================
# CELLS
# =============================================================================

class Cells:
    """Cells (sorted cell ids) with their coordinates, grid indices and 3-D positions."""

    def __init__(self, cell_ids):
        self.cell_id = np.unique(np.asarray(cell_ids, dtype=np.int64))
        grid = reference_grid()
        self.lat = np.asarray(grid.lat[self.cell_id // grid.lon.size], dtype=np.float64)
        self.lon = np.asarray(grid.lon[self.cell_id % grid.lon.size], dtype=np.float64)
        self.n = self.cell_id.size
        self.lookup = np.full(self.cell_id.max() + 1, -1, dtype=np.int64)
        self.lookup[self.cell_id] = np.arange(self.n)

        steps = np.diff(np.unique(self.lat)).min(), np.diff(np.unique(self.lon)).min()
        if not np.allclose(steps, STEP_DEG, rtol=1e-2):
            raise ValueError(f"Grid steps {steps} differ from 30 arc-seconds.")
        fi = (self.lat - self.lat.min()) / STEP_DEG
        fj = (self.lon - self.lon.min()) / STEP_DEG
        self.i, self.j = np.rint(fi).astype(np.int64), np.rint(fj).astype(np.int64)
        if max(np.abs(fi - self.i).max(), np.abs(fj - self.j).max()) > 0.01:
            raise ValueError("Cell centres are not on a regular 30'' grid.")
        self.shape = (int(self.i.max()) + 1, int(self.j.max()) + 1)

        la, lo = np.radians(self.lat), np.radians(self.lon)
        self.xyz = R_EARTH * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        self.lat_mean = float(self.lat.mean())

    def pos(self, cell_id):
        """Position of each cell id in the sorted cells."""
        p = self.lookup[np.asarray(cell_id, dtype=np.int64)]
        if (p < 0).any():
            raise ValueError("Unknown cell ids.")
        return p


# =============================================================================
# FOLDS
# =============================================================================

def year_folds(years):
    """Fold of each cell-month: the index of its year in training.years."""
    train_years = np.asarray(cfg.training.years)
    folds = np.searchsorted(train_years, years).astype(np.int8)
    if not (train_years[np.minimum(folds, train_years.size - 1)] == years).all():
        raise ValueError("Cell-months outside training.years.")
    return folds


def square_blocks(cells, size_km):
    """Square of each cell (0..n-1, ordered by column, then row of the squares)."""
    lat0 = np.radians(cells.lat.mean())
    x = R_EARTH * np.cos(lat0) * np.radians(cells.lon)
    y = R_EARTH * np.radians(cells.lat)
    bx = np.floor((x - x.min()) / size_km).astype(np.int64)
    by = np.floor((y - y.min()) / size_km).astype(np.int64)
    return pd.factorize(bx * 100000 + by, sort=True)[0]


def deal_blocks(fire_per_block, n_folds, draws, rng):
    """(fold of each block, its max/min ratio, ratios of all draws): the draw with the lowest
    max/min ratio of fire cell-months per fold (ties: the first)."""
    n_blocks = fire_per_block.size
    best, best_ratio, ratios = None, np.inf, []
    for _ in range(draws):
        order = rng.permutation(n_blocks)
        fold_of_block = np.empty(n_blocks, dtype=np.int8)
        fold_of_block[order] = np.arange(n_blocks) % n_folds
        f = np.bincount(fold_of_block, weights=fire_per_block, minlength=n_folds)
        ratio = f.max() / f.min() if f.min() > 0 else np.inf
        ratios.append(ratio)
        if ratio < best_ratio:
            best, best_ratio = fold_of_block, ratio
    if best is None:
        raise RuntimeError("Every draw left a fold without fire cell-months.")
    return best, best_ratio, np.array(ratios)


def block_folds(cells, fire_cell_ids, size_km, n_folds, draws, seed, warmup_km=None):
    """(block and fold of each cell, information on the draws). fire_cell_ids: the cell of
    every fire cell-month."""
    rng = np.random.default_rng(seed)
    if warmup_km:
        n_warmup = int(square_blocks(cells, warmup_km).max()) + 1
        for _ in range(draws):
            rng.permutation(n_warmup)
    block = square_blocks(cells, size_km)
    n_blocks = int(block.max()) + 1
    fire_blocks = block[cells.pos(fire_cell_ids)]
    fire_per_block = np.bincount(fire_blocks, minlength=n_blocks).astype(np.float64)
    fold_of_block, ratio, ratios = deal_blocks(fire_per_block, n_folds, draws, rng)
    finite = ratios[np.isfinite(ratios)]
    info = {"block_km": size_km, "folds": n_folds, "blocks": n_blocks,
            "blocks_with_fire": int(np.unique(fire_blocks).size), "draws": draws,
            "seed": seed, "warmup_block_km": warmup_km,
            "warmup_blocks": n_warmup if warmup_km else None,
            "best_max_min_ratio": float(ratio),
            "median_max_min_ratio": float(np.median(finite)) if finite.size else None,
            "draws_with_empty_fold": int((~np.isfinite(ratios)).sum())}
    return block, fold_of_block[block], info


# =============================================================================
# SPATIAL AUTOCORRELATION
# =============================================================================

def indicator_correlation(cells, indicator, max_km, bin_km):
    """Correlation of a 0/1 indicator per cell by distance bin, over all pairs of cells."""
    M = np.zeros(cells.shape)
    Z = np.zeros(cells.shape)
    M[cells.i, cells.j] = 1.0
    Z[cells.i[indicator], cells.j[indicator]] = 1.0
    xc = lambda A, B: np.rint(fftconvolve(A, B[::-1, ::-1], mode="full"))   # noqa: E731
    Smm, Szz, Szm = xc(M, M), xc(Z, Z), xc(Z, M)
    ni, nj = cells.shape
    dy = np.radians(STEP_DEG) * R_EARTH
    dx = dy * np.cos(np.radians(cells.lat_mean))
    di = np.arange(-(ni - 1), ni)[:, None] * dy
    dj = np.arange(-(nj - 1), nj)[None, :] * dx
    d = np.sqrt(di ** 2 + dj ** 2)
    use = (d > 0) & (d < max_km)
    b = (d[use] // bin_km).astype(int)
    nb = int(np.ceil(max_km / bin_km))
    s = {n: np.bincount(b, weights=S[use], minlength=nb) for n, S in
         [("mm", Smm), ("zz", Szz), ("zm", Szm)]}
    dsum = np.bincount(b, weights=(d * Smm)[use], minlength=nb)
    mu = s["zm"] / s["mm"]
    return pd.DataFrame({"bin_from_km": np.arange(nb) * bin_km,
                         "bin_to_km": (np.arange(nb) + 1) * bin_km,
                         "mean_distance_km": dsum / s["mm"],
                         "pairs": (s["mm"] / 2).astype(np.int64),          # unordered pairs
                         "indicator_correlation": (s["zz"] / s["mm"] - mu ** 2) / (mu * (1 - mu))})
