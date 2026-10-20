"""
00_split_data.py
----------------
Splits the cell-months of the training set into train, test and validation subsets and
writes the model matrices of each subset (layout in data.py).

The unit of the split is the cell-month (cell_id, year, month). The training set holds one
row per GCM for each cell-month, all with the same fire label; the rows of a cell-month
always go to the same subset. The split is stratified by the fire label and made in two
stages: the validation subset first, then train and test from the rest (fractions
model_training.split, seed model_training.seed). Cell-months are sorted by year, month and cell_id
beforehand, so the split does not depend on the row order of the training set.

Checks: every GCM holds the same cell-months with the same fire labels, each exactly once.

Input:  paths.inputs.training_set
Output: model_training.paths.inputs.split_data — split_index.parquet, meta.json and one folder
        of model matrices per subset (see data.py), and stats/ with
          split_summary.csv  cell-months, fire cell-months, fire share and rows per subset
          split_by_year.csv  fire cell-months per year and subset
          split_by_month.csv fire cell-months per month and subset
Existing output is kept unless --force is given.

Usage:
    python 00_split_data.py [--force]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from resources import report_usage
import data

KEY = ["year", "month", "cell_id"]
FEATURES = cfg.data.features
GCMS = cfg.gcms.models
SOURCE = cfg.paths.inputs.training_set


# =============================================================================
# CELL-MONTHS AND SPLIT
# =============================================================================

def cell_months():
    """Cell-months of the training set with their fire label (checked across GCMs), sorted."""
    df = pq.read_table(SOURCE, columns=KEY + ["gcm", "fire"]).to_pandas()
    found = set(df.gcm.astype(str).unique())
    if found != set(GCMS):
        raise ValueError(f"GCMs in the training set {sorted(found)} differ from gcms.models.")

    per_gcm = {g: d.drop(columns="gcm").sort_values(KEY, ignore_index=True)
               for g, d in df.groupby("gcm", observed=True)}
    cm = per_gcm[GCMS[0]]
    if cm.duplicated(KEY).any():
        raise ValueError(f"{GCMS[0]}: {cm.duplicated(KEY).sum()} duplicated cell-months.")
    for g in GCMS[1:]:
        if not per_gcm[g].equals(cm):
            raise ValueError(f"{g}: cell-months or fire labels differ from {GCMS[0]}.")
    print(f"  {len(cm):,} cell-months, {int(cm.fire.sum()):,} with fire; "
          f"identical in all {len(GCMS)} GCMs")
    return cm


def split(cm):
    """Adds the column subset (categorical: train, test, validation)."""
    frac, seed = cfg.model_training.split, cfg.model_training.seed
    if not np.isclose(frac.train + frac.test + frac.validation, 1.0):
        raise ValueError("model_training.split fractions must sum to 1.")
    fire = cm.fire.to_numpy()
    rest, val = train_test_split(np.arange(len(cm)), test_size=frac.validation,
                                 random_state=seed, stratify=fire)
    train, test = train_test_split(rest, test_size=frac.test / (frac.train + frac.test),
                                   random_state=seed, stratify=fire[rest])
    codes = np.empty(len(cm), dtype=np.int8)
    for code, rows in enumerate([train, test, val]):          # order of data.SUBSETS
        codes[rows] = code
    cm["subset"] = pd.Categorical.from_codes(codes, data.SUBSETS)
    return cm


# =============================================================================
# MODEL MATRICES
# =============================================================================

class CellMonthLookup:
    """Dense lookup from (year, month, cell_id) to a cell-month's subset code and fire label."""

    def __init__(self, cm):
        years = np.array(cfg.training.years)
        self.year_pos = np.full(years.max() + 1, -1, dtype=np.int64)
        self.year_pos[years] = np.arange(len(years))
        self.n_cells = int(cm.cell_id.max()) + 1
        size = len(years) * 12 * self.n_cells
        k = self.keys(cm.year.to_numpy(), cm.month.to_numpy(), cm.cell_id.to_numpy())
        self.subset = np.full(size, -1, dtype=np.int8)
        self.subset[k] = cm.subset.cat.codes.to_numpy()
        self.fire = np.full(size, -1, dtype=np.int8)
        self.fire[k] = cm.fire.to_numpy()
        self.hits = np.zeros(size, dtype=np.int32)       # rows seen per cell-month
        self.cm_keys = k

    def keys(self, year, month, cell_id):
        pos = self.year_pos[year]
        if (pos < 0).any():
            raise ValueError("Rows outside training.years.")
        return (pos * 12 + month.astype(np.int64) - 1) * self.n_cells + cell_id


def write_matrices(cm, out):
    """Writes the arrays of each subset (data.ARRAYS) into out/<subset>/, row group by row group."""
    lookup = CellMonthLookup(cm)
    n_rows = {s: int((cm.subset == s).sum()) * len(GCMS) for s in data.SUBSETS}
    arrays, offset = {}, dict.fromkeys(data.SUBSETS, 0)
    for s in data.SUBSETS:
        (out / s).mkdir(parents=True)
        arrays[s] = {name: open_memmap(out / s / f"{name}.npy", mode="w+", dtype=dtype,
                                       shape=(n_rows[s], len(FEATURES)) if name == "X" else (n_rows[s],))
                     for name, dtype in data.ARRAYS.items()}

    pf = pq.ParquetFile(SOURCE)
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=KEY + ["gcm", "fire"] + FEATURES).to_pandas()
        k = lookup.keys(df.year.to_numpy(), df.month.to_numpy(), df.cell_id.to_numpy())
        subset = lookup.subset[k]
        if (subset < 0).any():
            raise ValueError(f"Row group {rg}: {(subset < 0).sum()} rows of unknown cell-months.")
        if (lookup.fire[k] != df.fire.to_numpy()).any():
            raise ValueError(f"Row group {rg}: fire labels differ between GCMs.")
        lookup.hits += np.bincount(k, minlength=lookup.hits.size).astype(np.int32)

        gcm_index = np.array([GCMS.index(g) for g in df.gcm.cat.categories], dtype=np.int8)
        columns = {"X": df[FEATURES].to_numpy(np.float32), "y": df.fire.to_numpy(np.int8),
                   "cell_id": df.cell_id.to_numpy(np.int32), "year": df.year.to_numpy(np.int16),
                   "month": df.month.to_numpy(np.int8), "gcm": gcm_index[df.gcm.cat.codes.to_numpy()]}
        for code, s in enumerate(data.SUBSETS):
            rows = subset == code
            n = int(rows.sum())
            for name, values in columns.items():
                arrays[s][name][offset[s]:offset[s] + n] = values[rows]
            offset[s] += n
        if (rg + 1) % 12 == 0 or rg + 1 == pf.num_row_groups:
            print(f"  row groups {rg + 1}/{pf.num_row_groups}")

    if offset != n_rows:
        raise ValueError(f"Rows written {offset} differ from expected {n_rows}.")
    if (lookup.hits[lookup.cm_keys] != len(GCMS)).any() or lookup.hits.sum() != sum(n_rows.values()):
        raise ValueError(f"Not every cell-month has exactly {len(GCMS)} rows.")
    for s in data.SUBSETS:
        for a in arrays[s].values():
            a.flush()
    return n_rows


# =============================================================================
# SUMMARY
# =============================================================================

def summarize(cm, n_rows, out):
    """Writes the summary tables into out and returns the per-subset summary."""
    g = cm.groupby("subset", observed=True).fire
    summary = pd.DataFrame({"cell_months": g.size(), "fire_cell_months": g.sum()})
    summary.index = summary.index.astype(str)
    summary.loc["total"] = summary.sum()
    summary["fire_share_pct"] = 100 * summary.fire_cell_months / summary.cell_months
    summary["rows"] = [n_rows[s] for s in data.SUBSETS] + [sum(n_rows.values())]

    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "split_summary.csv", float_format="%.4f")
    for by in ["year", "month"]:
        table = cm[cm.fire == 1].pivot_table(index=by, columns="subset", values="fire",
                                             aggfunc="size", fill_value=0, observed=False)
        table.to_csv(out / f"split_by_{by}.csv")
    print(f"  Saved: {data.STATS.name}/split_summary.csv, split_by_year.csv, split_by_month.csv")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main(force):
    out = data.DIR
    if data.META.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return

    print(f"Cell-months of {SOURCE.name}:")
    cm = split(cell_months())

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)                         # left over from an interrupted run
    tmp.mkdir(parents=True)

    print("Model matrices:")
    n_rows = write_matrices(cm, tmp)
    cm.to_parquet(tmp / "split_index.parquet", index=False)

    print("Summary:")
    summary = summarize(cm, n_rows, tmp / data.STATS.name)
    meta = {"source": data.source_signature(), "features": FEATURES, "gcms": GCMS,
            "seed": cfg.model_training.seed, "split": dict(cfg.model_training.split),
            "rows": n_rows,
            "cell_months": {s: int(summary.loc[s, "cell_months"]) for s in data.SUBSETS},
            "fire_cell_months": {s: int(summary.loc[s, "fire_cell_months"]) for s in data.SUBSETS}}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"\n{summary.to_string(float_format=lambda v: f'{v:,.4f}')}")
    print(f"\nWritten: {out}")
    report_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="recompute existing output")
    main(parser.parse_args().force)
