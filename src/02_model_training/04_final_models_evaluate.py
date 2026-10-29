"""
04_final_models_evaluate.py
---------------------------
Scores the final models (04_final_models.py) on the validation subset, which no earlier step
used: the scores at the committed decision threshold and the threshold-free scores, with
bootstrap intervals; the top-k coverage; and the permutation importance.

Scoring as in steps 01-03: per real validation cell-month, on the mean probability of its
GCM rows. Two scopes: all cell-months, and those of the fire season (fire_season.months).
  - At the committed threshold of the model (peak MCC on the test cell-months, <model>.json;
    never changed here): MCC, balanced accuracy, precision, recall and the confusion matrix.
  - Threshold-free: average precision (AP), AP divided by the fire share of the scope (lift
    over a model without skill, whose AP is the fire share) and ROC-AUC.
  - 95% intervals: percentile bootstrap over the validation cell-months
    (final_models.bootstrap_resamples resamples with replacement, seed final_models.seed).
    The resamples are the same for every model; the fire-season scope uses the cell-months
    of each resample that fall in the season.

Top-k coverage: the cell-months of a scope ranked by probability (ties in row order); for
each k in final_models.topk_percent, the share of the fire cell-months among the top k% of
the cell-months. A model without skill captures k%.

Permutation importance: the drop in AP from the unpermuted validation predictions, per
repeat (04_final_models.py importance), summarised over the repeats; also relative to the
unpermuted AP.

Can run at any time: models without validation predictions or importance are reported and
left out.

Input:  output of 04_final_models.py; model data of 00_split_data.py (validation labels)
Output: results/<model_training final_models>/
          metrics.csv      per model and scope: cell-months, fire share, threshold, confusion
                           matrix, scores and their bootstrap intervals (<score>_low/_high)
          topk.csv         per model, scope and k: cell-months flagged, fire cell-months
                           captured and their share
          importance.csv   per model and permuted group or predictor: AP without and with
                           the permutation, mean, standard deviation, minimum and maximum of
                           the drop over the repeats, mean drop relative to the AP (%)

Usage:
    python 04_final_models_evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from load_config import cfg
from classification_metrics import at_threshold
from resources import report_usage
import model_data

MODELS = cfg.model_training.models
SETTINGS = cfg.model_training.final_models
PRED_DIR = cfg.model_training.paths.generated.final_models
OUT_DIR = cfg.model_training.paths.results.final_models
SCOPES = ("all", "fire_season")
SCORES = ("mcc", "balanced_accuracy", "precision", "recall", "average_precision", "ap_lift",
          "roc_auc")


# =============================================================================
# SCORES WITH CELL-MONTH WEIGHTS (bootstrap)
# =============================================================================

class Scorer:
    """Scores of one model in one scope for cell-month weights w (the number of times each
    cell-month is drawn in a bootstrap resample; all 1 for the scores themselves). AP and
    ROC-AUC are computed from the weighted counts at every distinct probability, as
    scikit-learn does, so that a resample needs no sorting."""

    def __init__(self, y, p, threshold):
        self.y = y.astype(np.float64)
        _, self.group = np.unique(-p, return_inverse=True)   # distinct p, highest first
        self.n_groups = self.group.max() + 1
        self.flagged = p >= threshold

    def __call__(self, w):
        fire, no_fire = w * self.y, w * (1 - self.y)
        tp = np.bincount(self.group, weights=fire, minlength=self.n_groups).cumsum()
        fp = np.bincount(self.group, weights=no_fire, minlength=self.n_groups).cumsum()
        flagged = tp + fp
        precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
        recall, fpr = tp / tp[-1], fp / fp[-1]
        ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))
        s = threshold_scores(fire[self.flagged].sum(), no_fire[self.flagged].sum(),
                             fire[~self.flagged].sum(), no_fire[~self.flagged].sum())
        return {**s, "average_precision": ap, "ap_lift": ap / (tp[-1] / (tp[-1] + fp[-1])),
                "roc_auc": float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fpr]))}


def threshold_scores(tp, fp, fn, tn):
    """MCC, balanced accuracy, precision and recall from a (weighted) confusion matrix, as
    classification_metrics.at_threshold."""
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recall = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
            "balanced_accuracy": (recall + tnr) / 2,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": recall}


# =============================================================================
# TOP-K COVERAGE AND IMPORTANCE
# =============================================================================

def topk(y, p):
    """Cell-months flagged and fire cell-months captured among the top k% by probability."""
    order = np.argsort(-p, kind="stable")
    captured = np.cumsum(y[order])
    records = []
    for k in SETTINGS.topk_percent:
        n = int(round(k / 100 * y.size))
        records.append({"k_percent": k, "cell_months_flagged": n,
                        "fire_captured": int(captured[n - 1]),
                        "fire_captured_share": captured[n - 1] / y.sum(), "no_skill": k / 100})
    return records


def importance_summary(model):
    path = PRED_DIR / f"importance_{model}.csv"
    if not path.exists():
        print(f"  {path} not found: no importance")
        return None
    table = pd.read_csv(path)
    table["drop"] = table.average_precision_base - table.average_precision
    summary = (table.groupby(["kind", "group", "name", "predictors"], sort=False)
               .agg(average_precision_base=("average_precision_base", "first"),
                    average_precision=("average_precision", "mean"),
                    repeats=("drop", "size"), drop_mean=("drop", "mean"),
                    drop_std=("drop", "std"), drop_min=("drop", "min"), drop_max=("drop", "max"))
               .reset_index())
    summary["drop_relative_percent"] = 100 * summary.drop_mean / summary.average_precision_base
    summary.insert(0, "model", model)
    for r in summary[summary.kind == "group"].itertuples():
        print(f"  importance {r.name}: AP drop {r.drop_mean:.4f} ± {r.drop_std:.4f} "
              f"({r.drop_relative_percent:.0f}% of {r.average_precision_base:.4f})")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def check_scores(point, y, p, threshold):
    """The scores of Scorer (all weights 1) must equal those of scikit-learn and
    classification_metrics.at_threshold; returns the confusion matrix at the threshold."""
    at = at_threshold(y, p, threshold)
    reference = {"average_precision": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
                 **{k: v for k, v in at.items() if k in SCORES}}
    for name, value in reference.items():
        if not np.isclose(point[name], value, rtol=1e-9, atol=1e-12):
            raise RuntimeError(f"{name}: {point[name]} (bootstrap scorer) != {value}")
    return {k: at[k] for k in ("tp", "fp", "fn", "tn")}


def main():
    validation = model_data.load_subset("validation",
                                        arrays=("y", "cell_id", "year", "month", "gcm"))
    rows, y = model_data.cell_month_rows(validation)
    month = np.asarray(validation["month"])[rows[0]]
    in_scope = {"all": np.ones(y.size, dtype=bool),
                "fire_season": np.isin(month, cfg.fire_season.months)}
    season = in_scope["fire_season"]
    print(f"Validation: {rows.shape[1]:,} cell-months ({int(y.sum()):,} fire) x {rows.shape[0]} "
          f"GCMs; fire season {int(season.sum()):,} ({int(y[season].sum()):,} fire)")

    metrics, coverage, importance, scorers = [], [], [], {}
    for model in MODELS:
        pred_file = PRED_DIR / f"{model}_validation.npy"
        info_file = cfg.paths.models_dir / f"{model}.json"
        if not (pred_file.exists() and info_file.exists()):
            print(f"{model}: {pred_file} or {info_file} not found, left out")
            continue
        threshold = json.loads(info_file.read_text())["threshold"]
        prob = np.load(pred_file)
        if prob.size != len(validation["y"]):
            raise ValueError(f"{pred_file}: {prob.size:,} predictions for "
                             f"{len(validation['y']):,} validation rows.")
        p = prob[rows].astype(np.float64).mean(axis=0)
        print(f"{model}: committed threshold {threshold:.6g}")
        for scope in SCOPES:
            ys, ps = y[in_scope[scope]], p[in_scope[scope]]
            scorer = Scorer(ys, ps, threshold)
            point = scorer(np.ones(ys.size))
            counts = check_scores(point, ys, ps, threshold)
            metrics.append({"model": model, "scope": scope, "cell_months": int(ys.size),
                            "fire_cell_months": int(ys.sum()), "fire_share": ys.mean(),
                            "threshold": threshold, **counts, **point})
            scorers[model, scope] = scorer
            coverage += [{"model": model, "scope": scope, **r} for r in topk(ys, ps)]
            print(f"  {scope}: MCC {point['mcc']:.4f}, balanced accuracy "
                  f"{point['balanced_accuracy']:.3f}, precision {point['precision']:.4f}, "
                  f"recall {point['recall']:.3f}; AP {point['average_precision']:.4f} "
                  f"({point['ap_lift']:.1f} x no skill), ROC-AUC {point['roc_auc']:.3f}")
        summary = importance_summary(model)
        if summary is not None:
            importance.append(summary)

    if not metrics:
        print("No validation predictions.")
        return
    n_boot = SETTINGS.bootstrap_resamples
    print(f"Bootstrap: {n_boot} resamples of the validation cell-months")
    rng = np.random.default_rng(SETTINGS.seed)
    boot = {key: {s: np.empty(n_boot) for s in SCORES} for key in scorers}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, y.size, y.size), minlength=y.size).astype(np.float64)
        for (model, scope), scorer in scorers.items():
            for name, value in scorer(w[in_scope[scope]]).items():
                boot[model, scope][name][b] = value
    for row in metrics:
        for name in SCORES:
            low, high = np.percentile(boot[row["model"], row["scope"]][name], [2.5, 97.5])
            row[f"{name}_low"], row[f"{name}_high"] = low, high

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(metrics)
    order = ["model", "scope", "cell_months", "fire_cell_months", "fire_share", "threshold",
             "tp", "fp", "fn", "tn"] + [c for s in SCORES for c in (s, f"{s}_low", f"{s}_high")]
    table[order].to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.6g")
    pd.DataFrame(coverage).to_csv(OUT_DIR / "topk.csv", index=False, float_format="%.6g")
    print(table[["model", "scope", "mcc", "mcc_low", "mcc_high", "average_precision",
                 "average_precision_low", "average_precision_high"]].to_string(index=False))
    saved = "metrics.csv, topk.csv"
    if importance:
        pd.concat(importance).to_csv(OUT_DIR / "importance.csv", index=False, float_format="%.6g")
        saved += ", importance.csv"
    print(f"\nSaved: {OUT_DIR}/{saved}")
    report_usage()


if __name__ == "__main__":
    main()
