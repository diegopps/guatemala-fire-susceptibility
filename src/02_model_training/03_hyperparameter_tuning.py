"""
03_hyperparameter_tuning.py
---------------------------
Random search over the hyperparameters of each selected algorithm (model_training.models)
with cross-validation over the training cell-months; then the selected configuration is
fitted on the whole train subset and predicts the fire probability of every test row (all
GCMs). 03_hyperparameter_tuning_evaluate.py collects the CV scores and scores the test
predictions.

Configurations (utils/tuning.py, config [model_training.tuning]): configuration 0 = the
settings of utils/models.py (the defaults of steps 01-02), configurations 1..n_iter drawn at
random from the search space of the algorithm. Each configuration is fitted once per fold.

Folds: the training cell-months are split into model_training.tuning.folds folds, stratified
by the fire label (seed model_training.seed); all GCM rows of a cell-month are in one fold.

Class imbalance (model_training.samplers, chosen in step 02): the sampler acts only on the
training cell-months of each fit, as in step 02 (utils/resampling.py; the neighbour searches
standardize over those cell-months); the held-out fold is never resampled. With
"class_weights" the algorithm has balanced class weights from its training labels.

Score of a fit: on the held-out fold, the fire probability of each cell-month is the mean
over its GCM rows, and the score is the peak of the MCC curve over all thresholds
(utils/classification_metrics.py), with the balanced accuracy there, AP and ROC-AUC. The
selected configuration has the highest mean peak MCC over the folds (ties: lower number).

Commands:
  prepare    folds, and the resampled training cell-months of every fold for the samplers
             of all models (so slow samplers such as tomek and ncr run once)
  cv <model> <config|all> <fold|all>
             fits configurations on folds, one after the other; each fit writes its scores
  refit <model>
             selects the configuration (all fits must be scored), fits it on the whole
             train subset (resampled as in step 02) and predicts every test row. If
             configuration 0 is selected, its test predictions are those of step 02 (or of
             step 01, with class weights) and nothing is fitted.
  trees <model> [--best]
             for the SLURM wrapper: prints the sampler, the number of folds and the
             configurations grouped by number of trees (--best: the number of trees of the
             selected configuration). Reads no model data.

Input:  model data of 00_split_data.py (see utils/model_data.py)
Output: model_training.paths.generated.hyperparameter_tuning/ (see utils/tuning.py):
          folds.npy, resampled/, cv/<model>/config<i>_fold<k>.json (scores, parameters,
          training size, run times, tree nodes, peak memory),
          <model>.npy (fire probability of each test row, float32, in the row order of the
          test subset)
        results/<model_training hyperparameter_tuning>/
          folds.csv                cell-months and fire cell-months per fold
          resampling.csv           cell-months per sampler and fold before and after
                                   resampling, with the run time
          <model>/selection.json   selected configuration, its parameters and CV scores
          <model>/model_info.json  refit: settings, training size, run times, package versions
Existing output is kept unless --force is given.

Threads: a fit uses at most MAX_THREADS threads (or --threads N), even if the job has more
CPUs, as in step 02 (memory of the tree building grows with the number of threads).

--trial uses 1% of the training cell-months (drawn at random with seed model_training.seed),
and refit predicts only the rows of 1% of the test cell-months (other rows NaN), to time the
runs before the full ones. Its output goes to folders named <...>_trial.

Usage:
    python 03_hyperparameter_tuning.py prepare [--trial] [--force]
    python 03_hyperparameter_tuning.py cv <model> <config|all> <fold|all> [--threads N] [--trial] [--force]
    python 03_hyperparameter_tuning.py refit <model> [--threads N] [--trial] [--force]
    python 03_hyperparameter_tuning.py trees <model> [--best] [--trial]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from classification_metrics import scores
from resources import n_cpus, peak_memory_gb, report_usage
import model_data
import resampling
import tuning

MODELS = cfg.model_training.models
N_FOLDS = cfg.model_training.tuning.folds
MAX_THREADS = 16
TRIAL_FRACTION = 0.01
PREDICT_BLOCK = 1_000_000   # rows per predict_proba call


def save_npy(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.rename(path)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


def counts(y):
    return {"fire": int(y.sum()), "no_fire": int(y.size - y.sum())}


def predict_rows(model, X_all, rows):
    """Fire probability of the given rows of a predictor matrix (float32)."""
    prob = np.empty(rows.size, dtype=np.float32)
    for start in range(0, rows.size, PREDICT_BLOCK):
        block = rows[start:start + PREDICT_BLOCK]
        prob[start:start + block.size] = model.predict_proba(np.asarray(X_all[block]))[:, 1]
    return prob


def tree_nodes(model):
    """Nodes of all trees of a forest (the size of the fitted model); None otherwise."""
    if hasattr(model, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    return None


def fit(model_name, config, train, rows, res, cpus):
    """Fits a configuration on the resampled training cell-months (rows: their rows in each
    GCM); returns the model, the training size and the fit time."""
    X, y = resampling.training_rows(train, rows, res)
    model, fit_kwargs = tuning.estimator(model_name, config, y, cpus)
    t0 = time.monotonic()
    model.fit(X, y, **fit_kwargs)
    info = {"train_rows": len(y), "train_fire_rows": int(y.sum()), "cpus": cpus,
            "fit_minutes": round((time.monotonic() - t0) / 60, 3)}
    return model, info


# =============================================================================
# PREPARE
# =============================================================================

def make_folds(y, trial):
    """Fold of each training cell-month (-1: not used)."""
    folds = np.full(y.size, -1, dtype=np.int8)
    used = np.arange(y.size)
    if trial:
        rng = np.random.default_rng(cfg.model_training.seed)
        used = np.sort(rng.choice(used, int(TRIAL_FRACTION * used.size), replace=False))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg.model_training.seed)
    for k, (_, held_out) in enumerate(skf.split(np.zeros(used.size), y[used])):
        folds[used[held_out]] = k
    return folds


def prepare(trial, force):
    gen, res_dir = tuning.directories(trial)
    samplers = [s for s in dict.fromkeys(tuning.sampler(m) for m in MODELS) if s != tuning.CLASS_WEIGHTS]
    gen.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    cpus = n_cpus()
    print(f"Prepare{' (trial)' if trial else ''}: {N_FOLDS} folds; samplers: {', '.join(samplers) or 'none'}")

    train = model_data.load_subset("train")
    _, y = model_data.cell_month_rows(train)
    folds_file = gen / "folds.npy"
    if folds_file.exists() and not force:
        folds = np.load(folds_file)
        print(f"  {folds_file} exists, kept")
    else:
        folds = make_folds(y, trial)
        save_npy(folds_file, folds)
        print(f"  Saved: {folds_file}")
    table = pd.DataFrame([{"fold": k, "cell_months": int((folds == k).sum()),
                           "fire": int(y[folds == k].sum())} for k in range(N_FOLDS)])
    table["fire_share"] = table.fire / table.cell_months
    table.to_csv(res_dir / "folds.csv", index=False, float_format="%.6g")
    print(table.to_string(index=False))

    stats = []
    for s in samplers:
        for k in range(N_FOLDS):
            path = tuning.resampled_file(s, k, trial)
            cell_months = tuning.training_cell_months(folds, k)
            minutes = np.nan
            if path.exists() and not force:
                res = tuning.load_resampled(path)
            else:
                t0 = time.monotonic()
                res = resampling.resample(s, resampling.cell_month_table(train, cell_months), cpus)
                minutes = round((time.monotonic() - t0) / 60, 2)
                tuning.save_resampled(path, res)
            before, after = counts(y[cell_months]), res.counts(y[cell_months])
            stats.append({"sampler": s, "fold": k, "fire_before": before["fire"],
                          "no_fire_before": before["no_fire"], "fire_after": after["fire"],
                          "no_fire_after": after["no_fire"], "synthetic": int(res.parent.size),
                          "resample_minutes": minutes})
            print(f"  {s}, fold {k}: {before['fire']:,} fire / {before['no_fire']:,} no-fire -> "
                  f"{after['fire']:,} / {after['no_fire']:,} cell-months"
                  + ("" if np.isnan(minutes) else f", {minutes} min"))
    if stats:
        pd.DataFrame(stats).to_csv(res_dir / "resampling.csv", index=False)
    print(f"  Saved: {res_dir}/folds.csv" + (", resampling.csv" if stats else ""))
    report_usage()


# =============================================================================
# CV
# =============================================================================

def cv(model_name, config_arg, fold_arg, threads, trial, force):
    configs = tuning.configurations(model_name)
    config_ids = range(len(configs)) if config_arg == "all" else [int(config_arg)]
    fold_ids = range(N_FOLDS) if fold_arg == "all" else [int(fold_arg)]
    if not all(0 <= c < len(configs) for c in config_ids) or not all(0 <= k < N_FOLDS for k in fold_ids):
        sys.exit(f"Configurations are 0..{len(configs) - 1}, folds 0..{N_FOLDS - 1}.")
    todo = [(c, k) for c in config_ids for k in fold_ids
            if force or not tuning.cv_file(model_name, c, k, trial).exists()]
    if not todo:
        print("All requested fits are scored; use --force to recompute.")
        return
    sampler = tuning.sampler(model_name)
    cpus = min(n_cpus(), threads)
    print(f"Model: {model_name}, sampler: {sampler}" + (" (trial)" if trial else "")
          + f"; {len(todo)} fits, {cpus} CPUs")

    train = model_data.load_subset("train")
    rows, y = model_data.cell_month_rows(train)
    folds = tuning.load_folds(trial)
    for c, k in todo:
        fit_cms, held_out = tuning.training_cell_months(folds, k), tuning.held_out_cell_months(folds, k)
        if sampler == tuning.CLASS_WEIGHTS:
            res = resampling.Resampled(np.arange(fit_cms.size))
        else:
            res = tuning.load_resampled(tuning.resampled_file(sampler, k, trial))
        model, info = fit(model_name, configs[c], train, rows[:, fit_cms], res, cpus)

        t0 = time.monotonic()
        held_rows = rows[:, held_out]
        prob = predict_rows(model, train["X"], held_rows.ravel()).reshape(held_rows.shape)
        s = scores(y[held_out], prob.astype(np.float64).mean(axis=0))
        record = {"model": model_name, "config": c, "fold": k, "sampler": sampler, "trial": trial,
                  "params": configs[c], **s, **info,
                  "predict_minutes": round((time.monotonic() - t0) / 60, 3),
                  "tree_nodes": tree_nodes(model), "peak_memory_gb": round(peak_memory_gb(), 2)}
        save_json(tuning.cv_file(model_name, c, k, trial), record)
        print(f"  config {c:3d}, fold {k}: peak MCC {s['peak_mcc']:.4f}, AP {s['average_precision']:.4f}, "
              f"fit {info['fit_minutes']:.1f} min, {record['predict_minutes']:.1f} min predict, "
              f"peak memory {record['peak_memory_gb']:.1f} GB  {configs[c]}")
        del model
    report_usage()


# =============================================================================
# REFIT
# =============================================================================

def refit(model_name, threads, trial, force):
    gen, res_dir = tuning.directories(trial)
    best, summary = tuning.select(model_name, trial)
    config = int(best.config)
    params = tuning.configurations(model_name)[config]
    default = summary.set_index("config").loc[0]
    sampler = tuning.sampler(model_name)
    selection = {"model": model_name, "sampler": sampler, "config": config, "params": params,
                 "settings": tuning.settings(model_name, params),
                 "configurations": len(summary), "folds": N_FOLDS,
                 "scorer": "peak MCC on the held-out fold (cell-month mean probability over the "
                           "GCMs, all thresholds), mean over the folds",
                 "cv_peak_mcc_mean": best.peak_mcc_mean, "cv_peak_mcc_std": best.peak_mcc_std,
                 "default_cv_peak_mcc_mean": default.peak_mcc_mean,
                 "default_cv_peak_mcc_std": default.peak_mcc_std, "trial": trial}
    save_json(res_dir / model_name / "selection.json", selection)
    print(f"Model: {model_name}" + (" (trial)" if trial else "")
          + f"; selected configuration {config}: CV peak MCC {best.peak_mcc_mean:.4f} "
          f"(defaults {default.peak_mcc_mean:.4f})  {params}")
    if config == 0:
        print("  The defaults are selected: their test predictions are those of step 02 "
              "(step 01 with class weights); nothing to fit.")
        return
    out = gen / f"{model_name}.npy"
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return

    cpus = min(n_cpus(), threads)
    train = model_data.load_subset("train")
    fit_cms = tuning.training_cell_months(tuning.load_folds(trial))
    t0 = time.monotonic()
    if sampler == tuning.CLASS_WEIGHTS:
        rows, _ = model_data.cell_month_rows(train)
        rows, res = rows[:, fit_cms], resampling.Resampled(np.arange(fit_cms.size))
    else:
        table = resampling.cell_month_table(train, fit_cms)
        rows, res = table.rows, resampling.resample(sampler, table, cpus)
        del table
    resample_minutes = round((time.monotonic() - t0) / 60, 2)
    model, info = fit(model_name, params, train, rows, res, cpus)
    print(f"  fitted on {info['train_rows']:,} rows ({info['train_fire_rows']:,} fire) "
          f"in {info['fit_minutes']:.1f} min, {cpus} CPUs")
    del train, rows, res

    test = model_data.load_subset("test")
    test_rows = np.arange(len(test["y"]))
    if trial:
        cm_rows, _ = model_data.cell_month_rows(test)
        rng = np.random.default_rng(cfg.model_training.seed)
        keep = rng.choice(cm_rows.shape[1], int(TRIAL_FRACTION * cm_rows.shape[1]), replace=False)
        test_rows = np.sort(cm_rows[:, keep].ravel())
    prob = np.full(len(test["y"]), np.nan, dtype=np.float32)
    t0 = time.monotonic()
    prob[test_rows] = predict_rows(model, test["X"], test_rows)
    predict_minutes = round((time.monotonic() - t0) / 60, 2)
    gen.mkdir(parents=True, exist_ok=True)
    save_npy(out, prob)
    print(f"  predicted {test_rows.size:,} test rows in {predict_minutes:.1f} min\n  Saved: {out}")

    model_info = {"model": model_name, "sampler": sampler, "config": config, "params": params,
                  "estimator": repr(model), "seed_folds": cfg.model_training.seed,
                  "seed_sampler": resampling.SEED, "trial": trial, **info,
                  "resample_minutes": resample_minutes, "predict_minutes": predict_minutes,
                  "predicted_rows": int(test_rows.size), "tree_nodes": tree_nodes(model),
                  "packages": tuning.package_versions()}
    save_json(res_dir / model_name / "model_info.json", model_info)
    print(f"  Saved: {res_dir / model_name}/selection.json, model_info.json")
    report_usage()


# =============================================================================
# TREES (for the SLURM wrapper)
# =============================================================================

def trees(model_name, best, trial):
    if best:
        config = int(tuning.select(model_name, trial)[0].config)
        print(tuning.settings(model_name, tuning.configurations(model_name)[config])["n_estimators"])
        return
    groups = {}
    for i, config in enumerate(tuning.configurations(model_name)):
        groups.setdefault(tuning.settings(model_name, config)["n_estimators"], []).append(i)
    print(f"sampler {tuning.sampler(model_name)}")
    print(f"folds {N_FOLDS}")
    for n in sorted(groups):
        print(n, " ".join(map(str, groups[n])))


def add_options(p, threads=True, force=True):
    if threads:
        p.add_argument("--threads", type=int, default=MAX_THREADS,
                       help=f"maximum threads of the algorithm (default {MAX_THREADS})")
    p.add_argument("--trial", action="store_true", help="1%% of the cell-months, separate output")
    if force:
        p.add_argument("--force", action="store_true", help="recompute existing output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="folds and resampled training cell-months")
    add_options(p, threads=False)
    p = commands.add_parser("cv", help="fit configurations on folds")
    p.add_argument("model", choices=MODELS)
    p.add_argument("config", help="configuration number or 'all'")
    p.add_argument("fold", help="fold number or 'all'")
    add_options(p)
    p = commands.add_parser("refit", help="fit the selected configuration, predict the test rows")
    p.add_argument("model", choices=MODELS)
    add_options(p)
    p = commands.add_parser("trees", help="configurations by number of trees (for the SLURM wrapper)")
    p.add_argument("model", choices=MODELS)
    p.add_argument("--best", action="store_true", help="number of trees of the selected configuration")
    add_options(p, threads=False, force=False)
    a = parser.parse_args()
    if a.command == "prepare":
        prepare(a.trial, a.force)
    elif a.command == "cv":
        cv(a.model, a.config, a.fold, a.threads, a.trial, a.force)
    elif a.command == "refit":
        refit(a.model, a.threads, a.trial, a.force)
    else:
        trees(a.model, a.best, a.trial)
