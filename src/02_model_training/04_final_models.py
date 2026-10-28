"""
04_final_models.py
------------------
Final models: the selected configuration of each selected algorithm (model_training.models;
step 03, results/<model_training hyperparameter_tuning>/<model>/selection.json) is fitted on
the whole train subset, saved, and predicts the fire probability of every test and
validation row (all GCMs). The validation subset is used here for the first time.
04_final_models_evaluate.py scores the predictions.

Fit: the same procedure as the refit of step 03 (utils/fitting.py): the training cell-months
are resampled with the sampler of the model (model_training.samplers) or, with
"class_weights", fitted with balanced class weights; the seeds are those of steps 02-03. The
test predictions therefore reproduce those of step 03 (or of step 02, step 01 with class
weights, if configuration 0 is selected); --check compares them.

Decision threshold: one per model, committed here: the peak of the MCC curve on the test
cell-months, scored on the mean probability over the GCM rows of each cell-month
(utils/classification_metrics.py). A cell-month is predicted as fire if its mean probability
is at least the threshold. It is never changed on the validation subset.

Permutation importance (model_training.final_models): on the validation subset, the drop in
average precision (AP) of the cell-month mean probability when predictors are permuted. Whole
cell-months are permuted: each cell-month takes, in every GCM, the values of the cell-month
the permutation assigns to it, so its GCM rows keep the same static predictors. The
predictors of a group (importance_groups) are permuted together; then each predictor of a
group with more than one, alone. Each repeat draws one permutation of the validation
cell-months (seed final_models.seed), used for every group and predictor of that repeat.

Commands:
  fit <model>           fits the model, saves it, predicts every test and validation row
                        and commits the threshold
  fit <model> --check   only compares the saved test predictions with those of the same
                        configuration in steps 01-03 (largest difference per row and per
                        cell-month, peak MCC and threshold of both); fits nothing
  importance <model>    loads the saved model, predicts the validation rows again (they must
                        equal the saved predictions) and runs the permutations

Input:  model data of 00_split_data.py (see utils/model_data.py); folds and selection of
        03_hyperparameter_tuning.py
Output: paths.models_dir/
          <model>.joblib    fitted model (uncompressed)
          <model>.json      configuration and settings, threshold with the test scores
                            there, training size, seeds, run times, model size, package
                            versions
        model_training.paths.generated.final_models/
          <model>_test.npy, <model>_validation.npy
                            fire probability of each row (float32, in the row order of the
                            subset)
          importance_<model>.csv
                            AP of the validation cell-months, unpermuted and per repeat and
                            permuted group or predictor
Existing output is kept unless --force is given.

Threads: a fit uses at most MAX_THREADS threads (or --threads N), as in steps 02-03.

Usage:
    python 04_final_models.py fit <model> [--threads N] [--force] [--check]
    python 04_final_models.py importance <model> [--threads N] [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from classification_metrics import scores
from resources import n_cpus, peak_memory_gb, report_usage
import fitting
import model_data
import models
import resampling
import tuning

MODELS = cfg.model_training.models
SETTINGS = cfg.model_training.final_models
GEN = cfg.model_training.paths.generated.final_models
MAX_THREADS = 16
MAX_RELOAD_DIFFERENCE = 1e-6   # validation predictions of the saved and the fitted model


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


def model_file(model_name):
    return cfg.paths.models_dir / f"{model_name}.joblib"


def info_file(model_name):
    return cfg.paths.models_dir / f"{model_name}.json"


def prediction_file(model_name, subset):
    return GEN / f"{model_name}_{subset}.npy"


def selection(model_name):
    """(configuration, parameters, selection.json) selected in step 03 for a model."""
    path = tuning.directories()[1] / model_name / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 03_hyperparameter_tuning.py refit "
                                f"{model_name} first.")
    sel = json.loads(path.read_text())
    params = tuning.configurations(model_name)[sel["config"]]
    if sel["trial"] or sel["sampler"] != tuning.sampler(model_name) or sel["params"] != params:
        raise RuntimeError(f"{path} does not match the current tuning settings (trial run, "
                           "model_training.samplers or search space changed): rerun step 03.")
    return sel["config"], params, sel


def cell_month_mean(prob, rows):
    """Mean probability of each cell-month over its GCM rows."""
    return prob[rows].astype(np.float64).mean(axis=0)


# =============================================================================
# FIT
# =============================================================================

def fit(model_name, threads, force):
    config, params, sel = selection(model_name)
    sampler = tuning.sampler(model_name)
    print(f"Model: {model_name}, sampler: {sampler}; configuration {config}  {params}")
    if info_file(model_name).exists() and not force:
        print(f"{info_file(model_name)} exists; use --force to recompute.")
        return

    cpus = min(n_cpus(), threads)
    train = model_data.load_subset("train")
    fit_cms = tuning.training_cell_months(tuning.load_folds())
    t0 = time.monotonic()
    rows, res = fitting.training_data(model_name, train, fit_cms, cpus)
    resample_minutes = round((time.monotonic() - t0) / 60, 2)
    y_fit = np.asarray(train["y"])[rows[0]]
    before, after, synthetic = counts(y_fit), res.counts(y_fit), int(res.parent.size)
    model, info = fitting.fit(model_name, params, train, rows, res, cpus)
    print(f"  {before['fire']:,} fire / {before['no_fire']:,} no-fire cell-months -> "
          f"{after['fire']:,} / {after['no_fire']:,} after resampling; fitted on "
          f"{info['train_rows']:,} rows ({info['train_fire_rows']:,} fire) in "
          f"{info['fit_minutes']:.1f} min, {cpus} CPUs")
    del train, rows, res

    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    path = model_file(model_name)
    tmp = path.with_name(path.name + ".tmp")
    t0 = time.monotonic()
    joblib.dump(model, tmp)
    tmp.rename(path)
    dump_minutes = round((time.monotonic() - t0) / 60, 2)
    size_gb = round(path.stat().st_size / 1024**3, 2)
    print(f"  Saved: {path} ({size_gb} GB, {dump_minutes:.1f} min)")

    GEN.mkdir(parents=True, exist_ok=True)
    predict_minutes = {}
    for subset in ("test", "validation"):
        data = model_data.load_subset(subset)
        t0 = time.monotonic()
        prob = fitting.predict_rows(model, data["X"], np.arange(len(data["y"])))
        predict_minutes[subset] = round((time.monotonic() - t0) / 60, 2)
        save_npy(prediction_file(model_name, subset), prob)
        print(f"  predicted {prob.size:,} {subset} rows in {predict_minutes[subset]:.1f} min\n"
              f"  Saved: {prediction_file(model_name, subset)}")
        if subset == "test":
            cm_rows, y = model_data.cell_month_rows(data)
            test_scores = scores(y, cell_month_mean(prob, cm_rows))
            print(f"  committed threshold {test_scores['threshold']:.6g}: test peak MCC "
                  f"{test_scores['peak_mcc']:.4f}, AP {test_scores['average_precision']:.4f}, "
                  f"ROC-AUC {test_scores['roc_auc']:.4f}")

    model_info = {"model": model_name, "sampler": sampler, "config": config, "params": params,
                  "settings": tuning.settings(model_name, params), "estimator": repr(model),
                  "cv_peak_mcc_mean": sel["cv_peak_mcc_mean"],
                  "cv_peak_mcc_std": sel["cv_peak_mcc_std"],
                  "threshold": test_scores["threshold"],
                  "threshold_rule": "fire if the mean probability over the GCM rows of a "
                                    "cell-month is at least the threshold (peak MCC on the "
                                    "test cell-months)",
                  "test_scores": test_scores,
                  "train_cell_months": before, "resampled_cell_months": after,
                  "synthetic_cell_months": synthetic, **info,
                  "seed_folds": cfg.model_training.seed, "seed_sampler": resampling.SEED,
                  "seed_model": models.SEED, "resample_minutes": resample_minutes,
                  "dump_minutes": dump_minutes, "predict_minutes": predict_minutes,
                  "model_file": str(path), "model_size_gb": size_gb,
                  "tree_nodes": fitting.tree_nodes(model),
                  "peak_memory_gb": round(peak_memory_gb(), 2),
                  "packages": tuning.package_versions()}
    save_json(info_file(model_name), model_info)
    print(f"  Saved: {info_file(model_name)}")
    report_usage()


# =============================================================================
# CHECK
# =============================================================================

def reference_file(model_name, config):
    """Test predictions of the same model and configuration in an earlier step: the refit of
    step 03, or for configuration 0 step 02 (step 01 with class weights)."""
    gen = cfg.model_training.paths.generated
    sampler = tuning.sampler(model_name)
    if config:
        return gen.hyperparameter_tuning / f"{model_name}.npy"
    if sampler == tuning.CLASS_WEIGHTS:
        return gen.model_testing / f"{model_name}.npy"
    return gen.resampling / model_name / f"{sampler}.npy"


def check(model_name):
    config, _, _ = selection(model_name)
    ours, reference = prediction_file(model_name, "test"), reference_file(model_name, config)
    for path in (ours, reference):
        if not path.exists():
            sys.exit(f"{path} not found.")
    a, b = np.load(ours), np.load(reference)
    if a.shape != b.shape:
        sys.exit(f"{ours}: {a.size:,} predictions, {reference}: {b.size:,}.")
    test = model_data.load_subset("test", arrays=("y", "cell_id", "year", "month", "gcm"))
    rows, y = model_data.cell_month_rows(test)
    pa, pb = cell_month_mean(a, rows), cell_month_mean(b, rows)
    print(f"Model: {model_name}, configuration {config}\n  step 04: {ours}\n  reference: {reference}")
    print(f"  largest absolute difference: {np.abs(a.astype(np.float64) - b).max():.3g} per row, "
          f"{np.abs(pa - pb).max():.3g} per cell-month mean; "
          f"{int((a != b).sum()):,} of {a.size:,} rows differ")
    for name, p in [("step 04", pa), ("reference", pb)]:
        s = scores(y, p)
        print(f"  {name:9s}: peak MCC {s['peak_mcc']:.6f} at threshold {s['threshold']:.6g}, "
              f"AP {s['average_precision']:.6f}, ROC-AUC {s['roc_auc']:.6f}")


# =============================================================================
# IMPORTANCE
# =============================================================================

def permutation_sets():
    """(kind, group, name, predictors) of each permutation: each group, then each predictor
    of the groups with more than one."""
    groups = SETTINGS.importance_groups
    listed = [v for variables in groups.values() for v in variables]
    if sorted(listed) != sorted(cfg.data.features):
        raise ValueError("model_training.final_models.importance_groups must list every "
                         "predictor of data.features exactly once.")
    sets = [("group", g, g, variables) for g, variables in groups.items()]
    sets += [("predictor", g, v, [v]) for g, variables in groups.items() if len(variables) > 1
             for v in variables]
    return sets


def importance(model_name, threads, force):
    out = GEN / f"importance_{model_name}.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force to recompute.")
        return
    for path in (model_file(model_name), info_file(model_name)):
        if not path.exists():
            sys.exit(f"{path} not found: run 04_final_models.py fit {model_name} first.")
    cpus = min(n_cpus(), threads)
    t0 = time.monotonic()
    model = joblib.load(model_file(model_name))
    model.set_params(n_jobs=cpus)
    print(f"Model: {model_name}; loaded in {(time.monotonic() - t0) / 60:.1f} min, {cpus} CPUs")

    validation = model_data.load_subset("validation")
    rows, y = model_data.cell_month_rows(validation)
    X_orig = np.array(validation["X"])
    X = X_orig.copy()
    all_rows = np.arange(len(X))
    base = fitting.predict_rows(model, X, all_rows)
    difference = float(np.abs(base - np.load(prediction_file(model_name, "validation"))).max())
    if difference > MAX_RELOAD_DIFFERENCE:
        raise RuntimeError(f"The saved model predicts the validation rows differently from the "
                           f"fitted one (largest difference {difference:.3g}).")
    ap_base = average_precision_score(y, cell_month_mean(base, rows))
    print(f"  validation: {rows.shape[1]:,} cell-months ({int(y.sum()):,} fire) x {rows.shape[0]} "
          f"GCMs; AP {ap_base:.4f} (largest difference to the saved predictions {difference:.3g})")

    columns = {v: i for i, v in enumerate(cfg.data.features)}
    sets = permutation_sets()
    rng = np.random.default_rng(SETTINGS.seed)
    records = []
    for r in range(SETTINGS.permutation_repeats):
        perm = rng.permutation(rows.shape[1])
        source = np.empty(len(X), dtype=np.int64)   # row whose values each row takes
        source[rows.ravel()] = rows[:, perm].ravel()
        for kind, group, name, variables in sets:
            cols = [columns[v] for v in variables]
            t0 = time.monotonic()
            X[:, cols] = X_orig[source[:, None], cols]
            prob = fitting.predict_rows(model, X, all_rows)
            ap = average_precision_score(y, cell_month_mean(prob, rows))
            X[:, cols] = X_orig[:, cols]
            records.append({"model": model_name, "repeat": r, "kind": kind, "group": group,
                            "name": name, "predictors": ",".join(variables),
                            "average_precision": ap, "average_precision_base": ap_base})
            print(f"  repeat {r}, {kind} {name}: AP {ap:.4f} (drop {ap_base - ap:.4f}), "
                  f"{(time.monotonic() - t0) / 60:.1f} min")

    table = pd.DataFrame(records)
    tmp = out.with_name(out.name + ".tmp")
    table.to_csv(tmp, index=False, float_format="%.6g")
    tmp.rename(out)
    print(f"  Saved: {out}")
    report_usage()


def add_options(p, force=True):
    p.add_argument("--threads", type=int, default=MAX_THREADS,
                   help=f"maximum threads of the algorithm (default {MAX_THREADS})")
    if force:
        p.add_argument("--force", action="store_true", help="recompute existing output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fit", help="fit and save the final model, predict test and validation")
    p.add_argument("model", choices=MODELS)
    p.add_argument("--check", action="store_true",
                   help="only compare the saved test predictions with those of steps 01-03")
    add_options(p)
    p = commands.add_parser("importance", help="permutation importance on the validation subset")
    p.add_argument("model", choices=MODELS)
    add_options(p)
    a = parser.parse_args()
    if a.command == "fit" and a.check:
        check(a.model)
    elif a.command == "fit":
        fit(a.model, a.threads, a.force)
    else:
        importance(a.model, a.threads, a.force)
