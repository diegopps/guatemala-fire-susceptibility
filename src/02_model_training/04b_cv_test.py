"""
04b_cv_test.py
--------------
Cross-validation test of the final configurations: how well the selected configuration and
sampler of each final model (step 04) predict cell-months of a year, or of an area, that the
fit has not seen, compared with a baseline from the fire records alone.
04b_cv_test_evaluate.py scores the out-of-fold predictions.

Data: all cell-months of the training years, the train, test and validation subsets of
00_split_data.py pooled, with all GCM rows of each cell-month (utils/cv_folds.py: combined
model matrices). Folds (config [model_training.cv_test], schemes):
  loyo     leave-one-year-out: one fold per year of training.years
  blocks   squares of block_km dealt at random to block_folds folds; of block_draws deals,
           the one with the lowest max/min ratio of fire cell-months per fold is kept
           (generator default_rng(seed), after the warm-up of warmup_block_km; see
           utils/cv_folds.py)

Fits: as 04_final_models.py fit, with other training data. The configuration and sampler of
paths.models_dir/<model>.json (checked against step 03 and model_training.samplers); the
sampler is applied to the training-fold cell-months only (utils/fitting.py; cell-month table
standardized over them), or balanced class weights; seeds of step 04 (utils/resampling.py,
utils/models.py). Every row of the held-out fold is predicted; the prediction of a cell-month
is the mean over its GCM rows. Only these out-of-fold predictions are saved, never the fold
models.

Baseline (utils/kde_baseline.py): KDE_h(c) x s(m) for h in bandwidths_km, per fold from the
fire cell-months of the training folds only; and on the validation subset of the random
split, from the fire cell-months of the train subset (as the final models).

Autocorrelation: the correlation of "cell burned at least once in the training years" between
cells, by distance (autocorrelation.max_km, bins of autocorrelation.bin_km), over all pairs
of cells (utils/cv_folds.indicator_correlation); the reference for the block size.

Commands:
  folds                        combined model matrices, cell-month table (checked against
                               the split index), folds of every scheme
  fit <model> <scheme> <fold>  one fold fit; saves the out-of-fold predictions
  baseline <scheme|validation> the baseline scores of every cell-month of the scheme (out of
                               fold), or of the validation cell-months
  autocorrelation              the autocorrelation table

Input:  model data of 00_split_data.py (see utils/model_data.py); paths.models_dir/<model>.json
        of 04_final_models.py; the reference grid (utils/chelsa_grid.py: cell coordinates)
Output: model_training.paths.generated.cv_test/
          combined/<array>.npy, done.json
                              the model matrices of the three subsets concatenated (train,
                              test, validation; 4.8 GB), read by fit
          cell_months.parquet one row per cell-month, in the order of
                              model_data.cell_month_rows of the combined arrays: cell_id,
                              year, month, fire, subset (of the random split)
          folds_<scheme>.npy  fold of each cell-month (int8, same order)
          blocks.parquet      per cell: cell_id, block, fold (blocks scheme)
          folds.json          settings and information on the block deals
          oof/<model>_<scheme>_fold<k>.npz
                              held_out (indices into cell_months.parquet) and p (mean
                              probability over the GCM rows, float64)
          oof/<model>_<scheme>_fold<k>.json
                              configuration, training size, seeds, run times, peak memory,
                              AP and ROC-AUC of the fold, package versions
          baselines/<scheme>.npz, baselines/validation.npz
                              cell_months (indices into cell_months.parquet) and
                              kde_<h>km (score of each, float64)
        results/<model_training cv_test>/
          autocorrelation.csv bin_from_km, bin_to_km, mean_distance_km, pairs,
                              indicator_correlation
Existing output is kept unless --force is given.

Threads: a fit uses at most MAX_THREADS threads (or --threads N), as steps 03-04.

Usage:
    python 04b_cv_test.py folds [--force]
    python 04b_cv_test.py fit <model> <scheme> <fold> [--threads N] [--force]
    python 04b_cv_test.py baseline <scheme|validation> [--force]
    python 04b_cv_test.py autocorrelation [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from resources import n_cpus, peak_memory_gb, report_usage
import cv_folds
import fitting
import kde_baseline
import model_data
import models
import resampling
import tuning

MODELS = cfg.model_training.models
SETTINGS = cfg.model_training.cv_test
SCHEMES = list(SETTINGS.schemes)
GEN = cfg.model_training.paths.generated.cv_test
RES = cfg.model_training.paths.results.cv_test
COMBINED = GEN / "combined"
CELL_MONTHS = GEN / "cell_months.parquet"
MAX_THREADS = 16
BANDWIDTHS = [float(h) for h in SETTINGS.bandwidths_km]

if not set(SCHEMES) <= {"loyo", "blocks"}:
    raise ValueError(f"model_training.cv_test.schemes = {SCHEMES}: only loyo and blocks exist.")


def n_folds(scheme):
    return len(cfg.training.years) if scheme == "loyo" else SETTINGS.block_folds


def fold_label(scheme, k):
    """The held-out year (loyo) or the fold number (blocks)."""
    return str(cfg.training.years[k]) if scheme == "loyo" else str(k)


def folds_file(scheme):
    return GEN / f"folds_{scheme}.npy"


def oof_file(model_name, scheme, k, suffix):
    return GEN / "oof" / f"{model_name}_{scheme}_fold{k}.{suffix}"


def baseline_file(name):
    return GEN / "baselines" / f"{name}.npz"


def baseline_key(h):
    return f"kde_{h:g}km"


def load_cell_months(columns=None):
    if not CELL_MONTHS.exists():
        sys.exit(f"{CELL_MONTHS} not found: run 04b_cv_test.py folds first.")
    return pd.read_parquet(CELL_MONTHS, columns=columns)


def load_folds(scheme):
    if not folds_file(scheme).exists():
        sys.exit(f"{folds_file(scheme)} not found: run 04b_cv_test.py folds first.")
    return np.load(folds_file(scheme))


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.rename(path)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


# =============================================================================
# FOLDS
# =============================================================================

def cell_month_table(info):
    """One row per cell-month of the combined arrays, checked against the split index."""
    data = cv_folds.load_combined(COMBINED, ("y", "cell_id", "year", "month", "gcm"))
    rows, y = model_data.cell_month_rows(data)
    r0 = rows[0]
    cm = pd.DataFrame({"cell_id": np.asarray(data["cell_id"])[r0].astype(np.int64),
                       "year": np.asarray(data["year"])[r0].astype(np.int64),
                       "month": np.asarray(data["month"])[r0].astype(np.int64),
                       "fire": y.astype(np.int8), "subset": cv_folds.subset_of_rows(r0, info)})
    si = pd.read_parquet(model_data.SPLIT_INDEX)
    si["subset"] = si.subset.astype(str)
    m = cm.merge(si.astype({"cell_id": np.int64, "year": np.int64, "month": np.int64}),
                 on=["cell_id", "year", "month"], how="outer", suffixes=("", "_si"),
                 indicator=True)
    if not ((m._merge == "both").all() and (m.fire == m.fire_si).all()
            and (m.subset == m.subset_si).all() and len(m) == len(cm)):
        raise ValueError("The cell-months of the combined arrays differ from the split index.")
    return cm


def make_folds(force):
    out = GEN / "folds.json"
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return
    t0 = time.monotonic()
    info = cv_folds.combine_subsets(COMBINED)
    print(f"Combined model matrices: {info['total']:,} rows ({info['rows']}) in "
          f"{(time.monotonic() - t0) / 60:.1f} min\n  Saved: {COMBINED}")
    cm = cell_month_table(info)
    cm.to_parquet(CELL_MONTHS, index=False)
    print(f"Cell-months: {len(cm):,} ({int(cm.fire.sum()):,} fire), identical to the split "
          f"index\n  Saved: {CELL_MONTHS}")

    record = {"schemes": SCHEMES, "cell_months": len(cm), "fire_cell_months": int(cm.fire.sum())}
    if "loyo" in SCHEMES:
        folds = cv_folds.year_folds(cm.year.to_numpy())
        np.save(folds_file("loyo"), folds)
        record["loyo"] = {"folds": len(cfg.training.years), "years": list(cfg.training.years)}
        print(f"loyo: {len(cfg.training.years)} folds\n  Saved: {folds_file('loyo')}")
    if "blocks" in SCHEMES:
        cells = cv_folds.Cells(cm.cell_id.unique())
        block, fold_of_cell, draws = cv_folds.block_folds(
            cells, cm.cell_id[cm.fire == 1], SETTINGS.block_km, SETTINGS.block_folds,
            SETTINGS.block_draws, SETTINGS.seed, SETTINGS.warmup_block_km)
        np.save(folds_file("blocks"), fold_of_cell[cells.pos(cm.cell_id)].astype(np.int8))
        pd.DataFrame({"cell_id": cells.cell_id, "block": block, "fold": fold_of_cell}).to_parquet(
            GEN / "blocks.parquet", index=False)
        record["blocks"] = draws
        print(f"blocks: {draws['blocks']} squares of {SETTINGS.block_km:g} km "
              f"({draws['blocks_with_fire']} with fire) in {SETTINGS.block_folds} folds; best "
              f"max/min ratio of fire cell-months per fold {draws['best_max_min_ratio']:.4f} "
              f"(median of {draws['draws']} deals {draws['median_max_min_ratio']:.4f})\n"
              f"  Saved: {folds_file('blocks')}, {GEN / 'blocks.parquet'}")
    save_json(out, record)
    print(f"  Saved: {out}")
    report_usage()


# =============================================================================
# FIT
# =============================================================================

def final_configuration(model_name):
    """(configuration, parameters, sampler) of the final model, checked against step 03."""
    path = cfg.paths.models_dir / f"{model_name}.json"
    if not path.exists():
        sys.exit(f"{path} not found: run 04_final_models.py fit {model_name} first.")
    info = json.loads(path.read_text())
    config = info["config"]
    params = tuning.configurations(model_name)[config]
    sampler = tuning.sampler(model_name)
    if params != info["params"] or sampler != info["sampler"]:
        raise RuntimeError(f"{path}: the configuration or sampler differ from the current "
                           "tuning settings or model_training.samplers.")
    return config, params, sampler


def fit(model_name, scheme, k, threads, force):
    if not 0 <= k < n_folds(scheme):
        sys.exit(f"{scheme} has folds 0 to {n_folds(scheme) - 1}.")
    out_npz, out_json = oof_file(model_name, scheme, k, "npz"), oof_file(model_name, scheme, k, "json")
    if out_json.exists() and not force:
        print(f"{out_json} exists; use --force to recompute.")
        return
    config, params, sampler = final_configuration(model_name)
    cpus = min(n_cpus(), threads)
    print(f"Model: {model_name}, sampler: {sampler}; configuration {config}  {params}\n"
          f"Scheme {scheme}, fold {k} ({fold_label(scheme, k)}); {cpus} threads", flush=True)

    data = cv_folds.load_combined(COMBINED)
    folds = load_folds(scheme)
    fit_cms, held_out = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
    t0 = time.monotonic()
    rows, res = fitting.training_data(model_name, data, fit_cms, cpus)
    resample_minutes = round((time.monotonic() - t0) / 60, 2)
    y_fit = np.asarray(data["y"])[rows[0]]
    before, after = (int(y_fit.sum()), int(y_fit.size - y_fit.sum())), res.counts(y_fit)
    synthetic = int(res.parent.size)
    model, info = fitting.fit(model_name, params, data, rows, res, cpus)
    print(f"  {before[0]:,} fire / {before[1]:,} no-fire cell-months -> {after['fire']:,} / "
          f"{after['no_fire']:,} after resampling; fitted on {info['train_rows']:,} rows "
          f"({info['train_fire_rows']:,} fire) in {info['fit_minutes']:.1f} min "
          f"(sampler {resample_minutes:.1f} min)", flush=True)
    del rows, res

    all_rows, y = model_data.cell_month_rows(data)
    held_rows = all_rows[:, held_out]
    del all_rows
    t0 = time.monotonic()
    prob = fitting.predict_rows(model, data["X"], held_rows.ravel()).reshape(held_rows.shape)
    p = prob.astype(np.float64).mean(axis=0)
    predict_minutes = round((time.monotonic() - t0) / 60, 2)
    tree_nodes = fitting.tree_nodes(model)
    del model
    yk = y[held_out]
    ap, auc = average_precision_score(yk, p), roc_auc_score(yk, p)
    print(f"  predicted {held_rows.size:,} held-out rows ({held_out.size:,} cell-months, "
          f"{int(yk.sum())} fire) in {predict_minutes:.1f} min: AP {ap:.4f}, ROC-AUC {auc:.4f}",
          flush=True)

    save_npz(out_npz, held_out=held_out, p=p)
    save_json(out_json, {
        "model": model_name, "scheme": scheme, "fold": k, "fold_label": fold_label(scheme, k),
        "sampler": sampler, "config": config, "params": params,
        "train_cell_months": {"fire": before[0], "no_fire": before[1]},
        "resampled_cell_months": after, "synthetic_cell_months": synthetic, **info,
        "held_out_cell_months": int(held_out.size), "held_out_fire": int(yk.sum()),
        "average_precision": ap, "roc_auc": auc, "resample_minutes": resample_minutes,
        "predict_minutes": predict_minutes, "tree_nodes": tree_nodes,
        "peak_memory_gb": round(peak_memory_gb(), 2),
        "seed_sampler": resampling.SEED, "seed_model": models.SEED,
        "packages": tuning.package_versions()})
    print(f"  Saved: {out_npz}, {out_json.name}")
    report_usage()


# =============================================================================
# BASELINE
# =============================================================================

def baseline(name, force):
    out = baseline_file(name)
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return
    cm = load_cell_months(["cell_id", "month", "fire", "subset"])
    cells = cv_folds.Cells(cm.cell_id.unique())
    fire = cm.fire.to_numpy() == 1
    scores = {baseline_key(h): np.full(len(cm), np.nan) for h in BANDWIDTHS}
    if name == "validation":     # (training fires, cell-months scored, label)
        subset = cm.subset.to_numpy()
        groups = [((subset == "train") & fire, subset == "validation", "validation")]
    else:
        folds = load_folds(name)
        groups = [((folds != k) & fire, folds == k, fold_label(name, k)) for k in range(n_folds(name))]
    for train_fire, target, label in groups:
        history = kde_baseline.FireHistory(cells, cm.cell_id[train_fire], cm.month[train_fire])
        s = kde_baseline.scores(cells, history, BANDWIDTHS, cm.cell_id[target], cm.month[target])
        for h, v in s.items():
            scores[baseline_key(h)][target] = v
        y = fire[target]
        print(f"{name} {label}: {history.n_fire_cell_months:,} training fire cell-months, "
              f"{int(target.sum()):,} cell-months scored; " + ", ".join(
                  f"h {h:g} km AP {average_precision_score(y, v):.4f}" for h, v in s.items()),
              flush=True)
    target = np.flatnonzero(~np.isnan(scores[baseline_key(BANDWIDTHS[0])]))
    save_npz(out, cell_months=target, **{key: v[target] for key, v in scores.items()})
    print(f"  Saved: {out}")
    report_usage()


# =============================================================================
# AUTOCORRELATION
# =============================================================================

def autocorrelation(force):
    out = RES / "autocorrelation.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return
    si = pd.read_parquet(model_data.SPLIT_INDEX, columns=["cell_id", "fire"])
    cells = cv_folds.Cells(si.cell_id.unique())
    burned = np.zeros(cells.n, dtype=bool)
    burned[cells.pos(si.cell_id[si.fire == 1].unique())] = True
    a = SETTINGS.autocorrelation
    table = cv_folds.indicator_correlation(cells, burned, a.max_km, a.bin_km)
    print(f"{burned.sum():,} of {cells.n:,} cells burned at least once in "
          f"{', '.join(map(str, cfg.training.years))} ({burned.mean():.4%}); all pairs of cells")
    for limit in (0.1, 0.05, 0.02):
        below = table.loc[table.indicator_correlation < limit, "bin_to_km"]
        print(f"  indicator correlation first below {limit}: bin ending at "
              f"{below.iloc[0]:g} km" if len(below) else f"  never below {limit}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    RES.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.6g")
    print(f"  Saved: {out}")
    report_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("folds", help="combined model matrices, cell-months and folds")
    p.add_argument("--force", action="store_true", help="recompute existing output")
    p = commands.add_parser("fit", help="one fold fit; saves the out-of-fold predictions")
    p.add_argument("model", choices=MODELS)
    p.add_argument("scheme", choices=SCHEMES)
    p.add_argument("fold", type=int)
    p.add_argument("--threads", type=int, default=MAX_THREADS,
                   help=f"maximum threads of the algorithm (default {MAX_THREADS})")
    p.add_argument("--force", action="store_true", help="recompute existing output")
    p = commands.add_parser("baseline", help="baseline scores of a scheme or the validation subset")
    p.add_argument("scheme", choices=SCHEMES + ["validation"])
    p.add_argument("--force", action="store_true", help="recompute existing output")
    p = commands.add_parser("autocorrelation", help="autocorrelation of the burned cells")
    p.add_argument("--force", action="store_true", help="recompute existing output")
    a = parser.parse_args()
    if a.command == "folds":
        make_folds(a.force)
    elif a.command == "fit":
        fit(a.model, a.scheme, a.fold, a.threads, a.force)
    elif a.command == "baseline":
        baseline(a.scheme, a.force)
    else:
        autocorrelation(a.force)
