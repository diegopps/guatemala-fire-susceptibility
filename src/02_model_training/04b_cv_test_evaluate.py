"""
04b_cv_test_evaluate.py
-----------------------
Scores of the cross-validation test (04b_cv_test.py): the out-of-fold predictions of the final
configurations and the baseline KDE_h(c) x s(m) under each scheme of
model_training.cv_test.schemes, next to the validation scores of the random split.

Scores, from the mean probability over the GCM rows of each cell-month (no decision
threshold):
  - average precision (AP) and ROC-AUC;
  - top-k coverage, k in model_training.final_models.topk_percent: the share of the fire
    cell-months among the top k% of the cell-months ranked by score, with ties broken at
    random (expected value, classification_metrics.topk_coverage). The baseline has many tied
    scores far from any training fire; the models practically none, so for them this equals
    the coverage of step 04 (ties in row order).
Pooled: over all out-of-fold cell-months of a scheme (the predictions of the folds as they
are). Per fold (blocks) or year (loyo): within the cell-months of the fold, so the top k%
are those of the fold.

Validation subset: the models' scores of 04_final_models_evaluate.py (metrics.csv and
topk.csv, scope "all"); the baseline from the train-subset fire cell-months
(04b_cv_test.py baseline validation), scored the same way.

Can run at any time: a model or baseline without all its folds is reported and left out.

Input:  output of 04b_cv_test.py; results of 04_final_models_evaluate.py
Output: results/<model_training cv_test>/
          comparison.csv  per method (models, kde_<h>km) and evaluation (validation,
                          loyo, blocks_<block_km>km): cell-months, fire cell-months, AP,
                          ROC-AUC, top-k coverage (top<k>_pct)
          per_fold.csv    the same per method, scheme and fold (fold = held-out year for
                          loyo)
          fold_sizes.csv  per scheme and fold: blocks, cells, cell-months, fire cell-months
                          and their share (%)
          block_draws.csv the deal of the blocks: block size, blocks (with fire), folds,
                          draws, seed, warm-up, best and median max/min ratio of fire
                          cell-months per fold

Usage:
    python 04b_cv_test_evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from classification_metrics import topk_coverage
from resources import report_usage

MODELS = list(cfg.model_training.models)
SETTINGS = cfg.model_training.cv_test
SCHEMES = list(SETTINGS.schemes)
KS = list(cfg.model_training.final_models.topk_percent)
BANDWIDTHS = [float(h) for h in SETTINGS.bandwidths_km]
GEN = cfg.model_training.paths.generated.cv_test
OUT_DIR = cfg.model_training.paths.results.cv_test
FINAL = cfg.model_training.paths.results.final_models
METHODS = MODELS + [f"kde_{h:g}km" for h in BANDWIDTHS]
EVALUATION = {"validation": "validation", "loyo": "loyo",
              "blocks": f"blocks_{SETTINGS.block_km:g}km"}
SCORE_COLUMNS = ["average_precision", "roc_auc"] + [f"top{k}_pct" for k in KS]


def n_folds(scheme):
    return len(cfg.training.years) if scheme == "loyo" else SETTINGS.block_folds


def fold_label(scheme, k):
    return str(cfg.training.years[k]) if scheme == "loyo" else str(k)


def scores(y, p):
    return {"cell_months": int(y.size), "fire_cell_months": int(y.sum()),
            "average_precision": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            **{f"top{k}_pct": v for k, v in topk_coverage(y, p, KS).items()}}


def model_predictions(model, scheme, folds, n):
    """Out-of-fold mean probability of every cell-month, or None if a fold is missing."""
    files = [GEN / "oof" / f"{model}_{scheme}_fold{k}.npz" for k in range(n_folds(scheme))]
    missing = [k for k, f in enumerate(files) if not f.exists()]
    if missing:
        print(f"  {model} {scheme}: no predictions for folds {missing}; left out")
        return None
    p = np.full(n, np.nan)
    for k, f in enumerate(files):
        with np.load(f) as d:
            if not (folds[d["held_out"]] == k).all():
                raise ValueError(f"{f}: held-out cell-months outside fold {k}.")
            p[d["held_out"]] = d["p"]
    if np.isnan(p).any():
        raise ValueError(f"{model} {scheme}: {int(np.isnan(p).sum())} cell-months without prediction.")
    return p


def baseline_scores(name, n):
    """{method: scores} of the baseline file of a scheme or of the validation subset, and
    the cell-months it scores; None if missing."""
    path = GEN / "baselines" / f"{name}.npz"
    if not path.exists():
        print(f"  baseline {name}: {path} not found; left out")
        return None, None
    with np.load(path) as d:
        target = d["cell_months"]
        out = {f"kde_{h:g}km": d[f"kde_{h:g}km"] for h in BANDWIDTHS}
    if name != "validation" and not np.array_equal(target, np.arange(n)):
        raise ValueError(f"{path}: not every cell-month has a score.")
    return out, target


def validation_rows(cm, fire):
    """The validation subset: models from step 04, baseline from 04b_cv_test.py."""
    rows = []
    metrics_path, topk_path = FINAL / "metrics.csv", FINAL / "topk.csv"
    if metrics_path.exists() and topk_path.exists():
        metrics = pd.read_csv(metrics_path)
        topk = pd.read_csv(topk_path)
        for model in MODELS:
            m = metrics[(metrics.model == model) & (metrics.scope == "all")]
            t = topk[(topk.model == model) & (topk.scope == "all")].set_index("k_percent")
            if m.empty or not set(KS) <= set(t.index):
                print(f"  {model} validation: not in {metrics_path.name} / {topk_path.name}; left out")
                continue
            m = m.iloc[0]
            rows.append({"method": model, "evaluation": EVALUATION["validation"],
                         "cell_months": int(m.cell_months),
                         "fire_cell_months": int(m.fire_cell_months),
                         "average_precision": m.average_precision, "roc_auc": m.roc_auc,
                         **{f"top{k}_pct": t.loc[k, "fire_captured_share"] for k in KS}})
    else:
        print(f"  models validation: {FINAL} has no metrics.csv / topk.csv; left out")
    base, target = baseline_scores("validation", len(cm))
    if base is not None:
        if not (cm.subset.to_numpy()[target] == "validation").all():
            raise ValueError("baselines/validation.npz scores cell-months outside the validation subset.")
        for method, p in base.items():
            rows.append({"method": method, "evaluation": EVALUATION["validation"],
                         **scores(fire[target], p)})
    return rows


def fold_sizes(cm, fire):
    rows = []
    blocks = pd.read_parquet(GEN / "blocks.parquet") if "blocks" in SCHEMES else None
    for scheme in SCHEMES:
        folds = np.load(GEN / f"folds_{scheme}.npy")
        for k in range(n_folds(scheme)):
            sel = folds == k
            rows.append({"scheme": scheme, "fold": fold_label(scheme, k),
                         "blocks": int((blocks.loc[blocks.fold == k, "block"]).nunique())
                         if scheme == "blocks" else None,
                         "cells": int(cm.cell_id[sel].nunique()), "cell_months": int(sel.sum()),
                         "fire_cell_months": int(fire[sel].sum()),
                         "fire_share_percent": 100 * fire[sel].mean()})
    return pd.DataFrame(rows).astype({"blocks": "Int64"})


def main():
    cm = pd.read_parquet(GEN / "cell_months.parquet")
    fire = cm.fire.to_numpy().astype(bool)
    print(f"{len(cm):,} cell-months ({fire.sum():,} fire); schemes {', '.join(SCHEMES)}")

    pooled, per_fold = validation_rows(cm, fire), []
    for scheme in SCHEMES:
        folds = np.load(GEN / f"folds_{scheme}.npy")
        predictions = {m: model_predictions(m, scheme, folds, len(cm)) for m in MODELS}
        base, _ = baseline_scores(scheme, len(cm))
        predictions.update(base or {})
        for method, p in predictions.items():
            if p is None:
                continue
            pooled.append({"method": method, "evaluation": EVALUATION[scheme], **scores(fire, p)})
            for k in range(n_folds(scheme)):
                sel = folds == k
                per_fold.append({"method": method, "scheme": scheme, "fold": fold_label(scheme, k),
                                 **scores(fire[sel], p[sel])})
            print(f"  {method} {scheme}: AP {pooled[-1]['average_precision']:.4g}, ROC-AUC "
                  f"{pooled[-1]['roc_auc']:.4g}, top-{KS[-1]}% {pooled[-1][f'top{KS[-1]}_pct']:.4g}")

    order = {e: i for i, e in enumerate(EVALUATION.values())}
    comparison = pd.DataFrame(pooled)
    comparison = comparison.sort_values(
        ["method", "evaluation"], key=lambda s: s.map(
            {m: i for i, m in enumerate(METHODS)} if s.name == "method" else order),
        ignore_index=True)
    per_fold = pd.DataFrame(per_fold)
    sizes = fold_sizes(cm, fire)
    tables = {"comparison.csv": comparison, "per_fold.csv": per_fold, "fold_sizes.csv": sizes}
    if "blocks" in SCHEMES:
        tables["block_draws.csv"] = pd.DataFrame(
            [json.loads((GEN / "folds.json").read_text())["blocks"]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT_DIR / name, index=False, float_format="%.6g")
    print("\nComparison (pooled out of fold; validation: the random split)\n"
          + comparison.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nFold sizes\n" + sizes.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved: {OUT_DIR}/{', '.join(tables)}")
    report_usage()


if __name__ == "__main__":
    main()
