"""
resampling.py
-------------
Resampling of the training data against the class imbalance, shared by the steps of the
model training (src/02_model_training) that train with a sampler.

Unit: the cell-month. The training data hold one row per GCM for each cell-month, all with
the same fire label and the same static predictors. The samplers act on cell-months, and
every cell-month they keep, repeat or create brings its rows of all GCMs; "balanced" thus
means as many fire as no-fire cell-months, as in the split and the evaluation.

Cell-month table: the predictors of a cell-month are the means over its GCM rows (the
static predictors are the same in all of them), standardized over the training
cell-months. The table only serves the samplers that use distances between cell-months.

Samplers (SAMPLERS), with the imbalanced-learn defaults: sampling_strategy="auto", i.e.
balance the two classes (ros, smote, adasyn, rus) or clean the no-fire class (tomek,
ncr); random_state SEED.
  ros     random oversampling: repeats fire cell-months drawn at random
  smote   SMOTE: synthetic fire cell-months between a fire cell-month and one of its
          5 nearest fire cell-months, drawn at random
  adasyn  ADASYN: as SMOTE, with more synthetic cell-months around the fire cell-months
          that have more no-fire cell-months among their 5 nearest cell-months
  rus     random undersampling: keeps as many no-fire cell-months, drawn at random, as
          there are fire cell-months
  tomek   Tomek links: removes the no-fire cell-month of every Tomek link
  ncr     neighbourhood cleaning rule: removes no-fire cell-months that their 3 nearest
          cell-months classify as fire, and no-fire neighbours that misclassify fire
          cell-months

ros, rus, tomek and ncr are the imbalanced-learn samplers applied to the cell-month table;
their sample_indices_ give the cell-months kept (or repeated). smote and adasyn create
new data; imbalanced-learn does not report which two cell-months and which weight make
each synthetic one, so they are implemented here following imbalanced-learn 0.14 step by
step (same neighbour searches, same random draws in the same order). A synthetic fire
cell-month made from fire cell-months i and j with weight w has, in every GCM g, the
predictors x_i,g + w (x_j,g - x_i,g): the same interpolation as imbalanced-learn, applied
to the rows of each GCM. Interpolation is linear, so the synthetic rows are in the
original units. With a single GCM the result equals that of imbalanced-learn
(python resampling.py --test).

Usage:
    from resampling import cell_month_table, resample, training_rows
    train = model_data.load_subset("train")
    table = cell_month_table(train)
    res = resample("smote", table, cpus)            # Resampled
    X, y = training_rows(train, table.rows, res)
"""

import argparse
from dataclasses import dataclass, field

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule, RandomUnderSampler, TomekLinks
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from load_config import cfg
import model_data

SEED = 42
K_NEIGHBOURS = 5                 # SMOTE and ADASYN (imbalanced-learn default)
SAMPLERS = {
    "ros":    "Random oversampling",
    "smote":  "SMOTE",
    "adasyn": "ADASYN",
    "rus":    "Random undersampling",
    "tomek":  "Tomek links",
    "ncr":    "Neighbourhood cleaning rule",
}


# =============================================================================
# CELL-MONTH TABLE
# =============================================================================

@dataclass
class CellMonthTable:
    rows: np.ndarray        # (GCMs, cell-months): row of each cell-month in each GCM
    y: np.ndarray           # (cell-months,): fire label
    X: np.ndarray           # (cell-months, predictors): standardized GCM-mean predictors


def cell_month_table(subset):
    """Cell-month table of a subset (dict of arrays from model_data.load_subset)."""
    rows, y = model_data.cell_month_rows(subset)
    X = np.zeros((rows.shape[1], subset["X"].shape[1]))
    for gcm_rows in rows:
        X += subset["X"][gcm_rows]
    X /= rows.shape[0]
    return CellMonthTable(rows, y, StandardScaler().fit_transform(X))


# =============================================================================
# SAMPLERS
# =============================================================================

@dataclass
class Resampled:
    cell_months: np.ndarray       # cell-months kept (repeats allowed)
    parent: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    neighbour: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    weight: np.ndarray = field(default_factory=lambda: np.empty(0))
    # synthetic fire cell-months: from parent towards neighbour by weight in [0, 1)

    def counts(self, y):
        """Fire and no-fire cell-months after resampling."""
        fire = int(y[self.cell_months].sum()) + self.parent.size
        return {"fire": fire, "no_fire": self.cell_months.size + self.parent.size - fire}


def _smote(X, y, cpus):
    """SMOTE (imbalanced-learn 0.14) on the fire class: (parent, neighbour, weight)."""
    fire = np.flatnonzero(y == 1)
    n_new = int((y == 0).sum()) - fire.size
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, n_jobs=cpus).fit(X[fire])
    nns = nn.kneighbors(X[fire], return_distance=False)[:, 1:]
    rng = np.random.RandomState(SEED)
    samples = rng.randint(low=0, high=nns.size, size=n_new)
    weight = rng.uniform(size=n_new)
    rows, cols = np.floor_divide(samples, K_NEIGHBOURS), np.mod(samples, K_NEIGHBOURS)
    return fire[rows], fire[nns[rows, cols]], weight


def _adasyn(X, y, cpus):
    """ADASYN (imbalanced-learn 0.14) on the fire class: (parent, neighbour, weight)."""
    rng = np.random.RandomState(SEED)
    fire = np.flatnonzero(y == 1)
    n_new = int((y == 0).sum()) - fire.size
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, n_jobs=cpus).fit(X)
    nns = nn.kneighbors(X[fire], return_distance=False)[:, 1:]
    ratio = np.sum(y[nns] != 1, axis=1) / K_NEIGHBOURS
    if not ratio.sum():
        raise RuntimeError("ADASYN: no fire cell-month has a no-fire cell-month among its neighbours.")
    ratio /= ratio.sum()
    n_generate = np.rint(ratio * n_new).astype(int)
    n_new = int(n_generate.sum())
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, n_jobs=cpus).fit(X[fire])
    nns = nn.kneighbors(X[fire], return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(fire.size), n_generate)
    cols = rng.choice(K_NEIGHBOURS, size=n_new)
    weight = rng.uniform(size=(n_new, 1))[:, 0]
    return fire[rows], fire[nns[rows, cols]], weight


def resample(name, table, cpus=1):
    """Applies a sampler to a cell-month table."""
    X, y = table.X, table.y
    if name in ("smote", "adasyn"):
        parent, neighbour, weight = (_smote if name == "smote" else _adasyn)(X, y, cpus)
        return Resampled(np.arange(len(y)), parent, neighbour, weight)
    sampler = {"ros": lambda: RandomOverSampler(random_state=SEED),
               "rus": lambda: RandomUnderSampler(random_state=SEED),
               "tomek": lambda: TomekLinks(n_jobs=cpus),
               "ncr": lambda: NeighbourhoodCleaningRule(n_jobs=cpus)}[name]()
    sampler.fit_resample(X, y)
    return Resampled(np.asarray(sampler.sample_indices_))



# =============================================================================
# TRAINING ROWS
# =============================================================================

def training_rows(subset, rows, res):
    """Predictors (float32) and labels of the resampled training data: the rows of every GCM
    of the kept cell-months, then those of the synthetic fire cell-months."""
    X_all, y_all = subset["X"], np.asarray(subset["y"])
    keep = rows[:, res.cell_months].ravel()
    X = [np.asarray(X_all[keep])]
    y = [y_all[keep]]
    w = res.weight[:, None]
    for gcm_rows in rows:
        if res.parent.size:
            Xi = np.asarray(X_all[gcm_rows[res.parent]], dtype=np.float64)
            Xj = np.asarray(X_all[gcm_rows[res.neighbour]], dtype=np.float64)
            X.append((Xi + w * (Xj - Xi)).astype(np.float32))
            y.append(np.ones(res.parent.size, dtype=y_all.dtype))
    return np.concatenate(X), np.concatenate(y)


# =============================================================================
# SELF-TEST
# =============================================================================

def _synthetic_subset(n_cells, n_gcm, rng):
    """A small training subset with the layout of model_data.load_subset: GCM blocks of the same
    cell-months, static predictors (first 4 columns) identical across GCMs."""
    n_static, n_climate = 4, 3
    static = rng.normal(size=(n_cells, n_static))
    base = rng.normal(size=(n_cells, n_climate))
    y = (rng.random(n_cells) < 1 / (1 + np.exp(-(-3 + 2 * static[:, 0] + base[:, 0])))).astype(np.int8)
    X = np.concatenate([np.c_[static, base + 0.3 * rng.normal(size=base.shape)]
                        for _ in range(n_gcm)]).astype(np.float32)
    ids = np.arange(n_cells)
    return {"X": X, "y": np.tile(y, n_gcm), "cell_id": np.tile(ids, n_gcm),
            "year": np.tile(2020 + ids % 2, n_gcm), "month": np.tile(1 + ids % 12, n_gcm),
            "gcm": np.repeat(np.arange(n_gcm), n_cells).astype(np.int8)}, n_static


def self_test():
    rng = np.random.default_rng(0)
    gcms = cfg.gcms.models

    # One GCM: our SMOTE and ADASYN reproduce imbalanced-learn.
    cfg.gcms.models = gcms[:1]
    subset, _ = _synthetic_subset(3000, 1, rng)
    table = cell_month_table(subset)
    n = len(table.y)
    for name, reference in [("smote", SMOTE(random_state=SEED)), ("adasyn", ADASYN(random_state=SEED))]:
        res = resample(name, table)
        X_ref, y_ref = reference.fit_resample(table.X, table.y)
        X_new = table.X[res.parent] + res.weight[:, None] * (table.X[res.neighbour] - table.X[res.parent])
        assert X_ref.shape[0] - n == res.parent.size, f"{name}: number of synthetic samples"
        assert np.allclose(X_ref[n:], X_new), f"{name}: synthetic samples differ from imbalanced-learn"
        print(f"  {name}: {res.parent.size:,} synthetic samples, identical to imbalanced-learn")

    # Several GCMs: counts, static predictors and weights of the training rows.
    n_gcm = 3
    cfg.gcms.models = gcms[:n_gcm]
    subset, n_static = _synthetic_subset(3000, n_gcm, rng)
    table = cell_month_table(subset)
    n_fire = int(table.y.sum())
    for name in SAMPLERS:
        res = resample(name, table)
        X, y = training_rows(subset, table.rows, res)
        c = res.counts(table.y)
        assert len(y) == n_gcm * (c["fire"] + c["no_fire"]) and int(y.sum()) == n_gcm * c["fire"]
        if name in ("ros", "smote", "rus"):
            assert c["fire"] == c["no_fire"], f"{name}: classes not balanced"
        if name in ("tomek", "ncr"):
            assert c["fire"] == n_fire and c["no_fire"] <= len(table.y) - n_fire
        if res.parent.size:
            assert ((res.weight >= 0) & (res.weight < 1)).all()
            synthetic = X[n_gcm * res.cell_months.size:].reshape(n_gcm, res.parent.size, -1)
            assert (synthetic[:, :, :n_static] == synthetic[:1, :, :n_static]).all(), \
                f"{name}: static predictors differ between the GCM rows of a synthetic cell-month"
        print(f"  {name}: {c['fire']:,} fire / {c['no_fire']:,} no-fire cell-months, "
              f"{len(y):,} rows ({n_gcm} GCMs)")
    cfg.gcms.models = gcms
    print("Self-test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test", action="store_true", help="run the self-test")
    if parser.parse_args().test:
        self_test()
