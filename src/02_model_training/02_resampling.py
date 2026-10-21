"""
02_resampling.py
----------------
Resamples the training data with one sampler and fits one of the selected algorithms
(model_training.models) on it, then predicts the fire probability of every test row (all
GCMs). 02_resampling_evaluate.py scores the predictions.

Resampling (utils/resampling.py): the sampler acts on the training cell-months; the
training rows are the rows of all GCMs of the cell-months it kept or repeated, and of its
synthetic fire cell-months. The samplers are deterministic for a given seed
(resampling.SEED), so every algorithm is trained on the same resampled data.

The algorithm has its settings of utils/models.py without class weights, so that the
sampler is the only correction of the class imbalance.

Input:  model data of 00_split_data.py (see utils/model_data.py)
Output: model_training.paths.generated.resampling/<model>/<sampler>.npy — fire probability
        of each test row (float32, in the row order of the test subset)
        results/<model_training resampling>/<model>/model_info_<sampler>.json — settings,
        cell-months before and after resampling, training rows, run times
Existing output is kept unless --force is given.

Threads: the algorithm uses at most MAX_THREADS threads (or --threads N), even if the job
has more CPUs. Each thread building a tree holds working arrays of the size of the
training data (about 2 GB per thread on the largest resampled sets), so a small number of
threads keeps the memory of the job low on the shared cluster, at the cost of run time.

--trial resamples 1% of the training cell-months and predicts 1% of the test rows (drawn at
random with seed model_training.seed), to time a run before the full one. Its output goes
to folders named <...>_trial and is never evaluated.

Usage:
    python 02_resampling.py <model> <sampler> [--threads N] [--trial] [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import imblearn
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from resources import n_cpus, report_usage
import model_data
import resampling
from models import build

MAX_THREADS = 16
TRIAL_FRACTION = 0.01
PREDICT_BLOCK = 1_000_000   # test rows per predict_proba call


def counts(y):
    return {"fire": int(y.sum()), "no_fire": int(y.size - y.sum())}


def save_npy(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.rename(path)


def main(model_name, sampler, threads, trial, force):
    pred_dir = cfg.model_training.paths.generated.resampling
    res_dir = cfg.model_training.paths.results.resampling
    if trial:
        pred_dir = pred_dir.with_name(pred_dir.name + "_trial")
        res_dir = res_dir.with_name(res_dir.name + "_trial")
    out = pred_dir / model_name / f"{sampler}.npy"
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    (res_dir / model_name).mkdir(parents=True, exist_ok=True)
    cpus = min(n_cpus(), threads)
    rng = np.random.default_rng(cfg.model_training.seed)
    print(f"Model: {model_name}, sampler: {resampling.SAMPLERS[sampler]}" + (" (trial)" if trial else ""))

    # Resampling
    train = model_data.load_subset("train")
    table = resampling.cell_month_table(train)
    if trial:
        keep = np.sort(rng.choice(len(table.y), int(TRIAL_FRACTION * len(table.y)), replace=False))
        table = resampling.CellMonthTable(table.rows[:, keep], table.y[keep], table.X[keep])
    t0 = time.monotonic()
    res = resampling.resample(sampler, table, cpus)
    resample_seconds = time.monotonic() - t0
    before, after = counts(table.y), res.counts(table.y)
    print(f"  cell-months: {before['fire']:,} fire / {before['no_fire']:,} no-fire -> "
          f"{after['fire']:,} / {after['no_fire']:,} ({res.parent.size:,} synthetic), "
          f"resampled in {resample_seconds / 60:.1f} min")
    X, y = resampling.training_rows(train, table.rows, res)
    n_train, n_train_fire = len(y), int(y.sum())
    print(f"  training rows: {n_train:,} ({n_train_fire:,} fire), {cpus} CPUs")
    del table, train

    # Fit
    model, fit_kwargs = build(model_name, y, cpus, balanced=False)
    t0 = time.monotonic()
    model.fit(X, y, **fit_kwargs)
    fit_seconds = time.monotonic() - t0
    print(f"  fitted in {fit_seconds / 60:.1f} min")
    del X, y, fit_kwargs

    # Prediction
    test_X = model_data.load_subset("test", arrays=("X",))["X"]
    test_rows = np.arange(len(test_X))
    if trial:
        test_rows = np.sort(rng.choice(test_rows, int(TRIAL_FRACTION * test_rows.size), replace=False))
    prob = np.empty(test_rows.size, dtype=np.float32)
    t0 = time.monotonic()
    for start in range(0, test_rows.size, PREDICT_BLOCK):
        block = test_rows[start:start + PREDICT_BLOCK]
        prob[start:start + block.size] = model.predict_proba(np.asarray(test_X[block]))[:, 1]
    predict_seconds = time.monotonic() - t0
    print(f"  predicted {test_rows.size:,} test rows in {predict_seconds / 60:.1f} min")
    save_npy(out, prob)
    print(f"  Saved: {out}")

    info = {"model": model_name, "sampler": sampler, "sampler_label": resampling.SAMPLERS[sampler],
            "seed": resampling.SEED, "imbalanced_learn": imblearn.__version__,
            "estimator": repr(model), "trial": trial,
            "cell_months_before": before, "cell_months_after": after, "synthetic": int(res.parent.size),
            "train_rows": n_train, "train_fire_rows": n_train_fire, "cpus": cpus,
            "resample_minutes": round(resample_seconds / 60, 2), "fit_minutes": round(fit_seconds / 60, 2),
            "predict_minutes": round(predict_seconds / 60, 2), "predicted_rows": int(test_rows.size)}
    info_file = res_dir / model_name / f"model_info_{sampler}.json"
    info_file.write_text(json.dumps(info, indent=2))
    print(f"  Saved: {info_file}")
    report_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", choices=cfg.model_training.models)
    parser.add_argument("sampler", choices=resampling.SAMPLERS)
    parser.add_argument("--threads", type=int, default=MAX_THREADS,
                        help=f"maximum threads of the algorithm (default {MAX_THREADS})")
    parser.add_argument("--trial", action="store_true", help="1%% of the data, separate output")
    parser.add_argument("--force", action="store_true", help="recompute existing output")
    a = parser.parse_args()
    main(a.model, a.sampler, a.threads, a.trial, a.force)
