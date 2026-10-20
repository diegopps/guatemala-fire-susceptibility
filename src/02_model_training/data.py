"""
data.py
-------
Model data of this pipeline: the split of the cell-months into train, test and
validation subsets, and the model matrices of each subset. Written once by
00_split_data.py into model_training.paths.inputs.split_data; read by all later steps.

Layout:
  split_index.parquet   one row per cell-month: year, month, cell_id, fire, subset
  meta.json             source file, features, GCMs, seed, split fractions and counts
  <subset>/X.npy        predictors (rows x data.features, in that order), float32
  <subset>/y.npy        target (fire), int8
  <subset>/cell_id.npy  int32 }
  <subset>/year.npy     int16 }  cell-month of each row
  <subset>/month.npy    int8  }
  <subset>/gcm.npy      int8, index into meta["gcms"]
  stats/                summary tables of the split (see 00_split_data.py)

Each subset keeps the row order of the training set (GCM, year, month, lat, lon). The rows
of the GCMs of one cell-month are always in the same subset.

Usage:
    from data import load_subset
    train = load_subset("train")          # dict of memory-mapped arrays
    X, y = train["X"], train["y"]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg

DIR = cfg.model_training.paths.inputs.split_data
SPLIT_INDEX = DIR / "split_index.parquet"
META = DIR / "meta.json"
STATS = DIR / "stats"

SUBSETS = ("train", "test", "validation")
ARRAYS = {"X": np.float32, "y": np.int8, "cell_id": np.int32, "year": np.int16,
          "month": np.int8, "gcm": np.int8}


def source_signature():
    """Path, size and modification time of the training set, to detect a rebuilt input."""
    path = cfg.paths.inputs.training_set
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_meta():
    """meta.json, checked against the current training set and config."""
    if not META.exists():
        raise FileNotFoundError(f"{META} not found: run 00_split_data.py first.")
    meta = json.loads(META.read_text())
    if meta["source"] != source_signature():
        raise RuntimeError("The training set changed since the model data was written: "
                           "rerun 00_split_data.py --force.")
    if meta["features"] != cfg.data.features or meta["gcms"] != cfg.gcms.models:
        raise RuntimeError("data.features or gcms.models changed since the model data was "
                           "written: rerun 00_split_data.py --force.")
    return meta


def load_subset(subset, arrays=tuple(ARRAYS), mmap=True):
    """Arrays of one subset as a dict; memory-mapped (read-only) unless mmap=False."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, not '{subset}'.")
    load_meta()
    return {name: np.load(DIR / subset / f"{name}.npy", mmap_mode="r" if mmap else None)
            for name in arrays}
