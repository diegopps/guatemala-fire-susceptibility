"""
tuning.py
---------
Hyperparameter tuning of the model training (03_hyperparameter_tuning*.py in
src/02_model_training): the configurations of the random search, the correction of the class
imbalance, the cross-validation folds and the CV scores.

Configurations (config.toml [model_training.tuning]): configuration 0 = the settings of
utils/models.py (the defaults of steps 01-02); configurations 1..n_iter = random draws from
the search space of the algorithm ([model_training.tuning.search_space.<model>]), the same
draws as scikit-learn's RandomizedSearchCV with random_state `seed`. A configuration is a
dict of parameters set on the estimator of models.build(). In the search spaces, max_depth 0
means no depth limit (None).

Class imbalance (model_training.samplers, chosen in step 02): a sampler of
utils/resampling.py, applied to the training cell-months of each fit, or "class_weights"
(balanced class weights of models.build(), no sampler).

Folds: the training cell-months are split into `folds` folds, stratified by the fire label.
All GCM rows of a cell-month are in one fold. The fit of fold k trains on the cell-months of
the other folds and is scored on fold k.

Files (model_training.paths.generated.hyperparameter_tuning; <...>_trial for trial runs):
  folds.npy                          fold of each training cell-month (int8, in the order of
                                     model_data.cell_month_rows; -1 = not used, trial runs)
  resampled/<sampler>_fold<k>.npz    resampling of the training cell-months of fold k
                                     (indices into those cell-months, see resampling.Resampled)
  cv/<model>/config<i>_fold<k>.json  scores of configuration i on the held-out fold k
  <model>.npy                        test predictions of the selected configuration

Usage:
    import tuning
    configs = tuning.configurations("xgboost")          # list of dicts, configs[0] == {}
    model, fit_kwargs = tuning.estimator("xgboost", configs[3], y, cpus)
    summary = tuning.cv_summary("xgboost", tuning.cv_scores("xgboost"))
"""

import json
import platform

import imblearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import ParameterSampler

from load_config import cfg
from models import LIBRARY_DEFAULTS, build
import resampling

CLASS_WEIGHTS = "class_weights"
NO_LIMIT = ("max_depth",)        # parameters whose 0 in the search space means None
SCORES = ("peak_mcc", "balanced_accuracy", "average_precision", "roc_auc")


def directories(trial=False):
    """(generated, results) directories of the tuning."""
    gen = cfg.model_training.paths.generated.hyperparameter_tuning
    res = cfg.model_training.paths.results.hyperparameter_tuning
    if trial:
        gen, res = gen.with_name(gen.name + "_trial"), res.with_name(res.name + "_trial")
    return gen, res


# =============================================================================
# CONFIGURATIONS AND CLASS IMBALANCE
# =============================================================================

def sampler(model):
    """Correction of the class imbalance of a model: a sampler name or CLASS_WEIGHTS."""
    value = cfg.model_training.samplers.get(model, "")
    if value not in (*resampling.SAMPLERS, CLASS_WEIGHTS):
        raise ValueError(f"model_training.samplers.{model} = '{value}': set it to the result of "
                         f"step 02, one of {', '.join([*resampling.SAMPLERS, CLASS_WEIGHTS])}.")
    return value


def search_space(model):
    """Search space of an algorithm: {parameter: list of values}."""
    space = cfg.model_training.tuning.search_space[model]
    return {name: [None if name in NO_LIMIT and v == 0 else v for v in values]
            for name, values in space.items()}


def configurations(model):
    """All configurations of the search: [{}] (the settings of models.py) + n_iter draws."""
    t = cfg.model_training.tuning
    draws = ParameterSampler(search_space(model), n_iter=t.n_iter, random_state=t.seed)
    return [{}] + [dict(d) for d in draws]


def estimator(model, config, y, cpus):
    """(estimator, keyword arguments of fit) of a configuration for training labels y: the
    algorithm of models.build(), with balanced class weights if the model has no sampler, and
    the parameters of the configuration."""
    est, fit_kwargs = build(model, y, cpus, balanced=sampler(model) == CLASS_WEIGHTS)
    est.set_params(**config)
    return est, fit_kwargs


def settings(model, config):
    """Values of the search-space parameters in a configuration (for configuration 0 those of
    models.build(), or the library default of models.LIBRARY_DEFAULTS where models.py sets
    none; None = no depth limit)."""
    est, _ = build(model, np.array([0, 1]), 1, balanced=False)
    est.set_params(**config)
    params = est.get_params()
    defaults = LIBRARY_DEFAULTS.get(model, {})
    return {name: defaults.get(name) if params[name] is None else params[name]
            for name in search_space(model)}


# =============================================================================
# FOLDS
# =============================================================================

def load_folds(trial=False):
    path = directories(trial)[0] / "folds.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 03_hyperparameter_tuning.py prepare first.")
    return np.load(path)


def training_cell_months(folds, k=None):
    """Training cell-months of the fit of fold k; k=None: all cell-months of the search."""
    return np.flatnonzero((folds >= 0) & (folds != (-1 if k is None else k)))


def held_out_cell_months(folds, k):
    return np.flatnonzero(folds == k)


def resampled_file(sampler_name, k, trial=False):
    return directories(trial)[0] / "resampled" / f"{sampler_name}_fold{k}.npz"


def save_resampled(path, res):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, cell_months=res.cell_months, parent=res.parent, neighbour=res.neighbour,
                 weight=res.weight)
    tmp.rename(path)


def load_resampled(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 03_hyperparameter_tuning.py prepare first.")
    with np.load(path) as d:
        return resampling.Resampled(d["cell_months"], d["parent"], d["neighbour"], d["weight"])


# =============================================================================
# CV SCORES
# =============================================================================

def cv_file(model, config, k, trial=False):
    return directories(trial)[0] / "cv" / model / f"config{config:03d}_fold{k}.json"


def cv_scores(model, trial=False):
    """Scores of the fits done so far: one row per configuration and fold."""
    files = sorted((directories(trial)[0] / "cv" / model).glob("config*_fold*.json"))
    return pd.DataFrame([json.loads(f.read_text()) for f in files])


def cv_summary(model, scores):
    """Per configuration: its search-space parameters, the folds scored, the mean and standard
    deviation of each score over the folds, and the rank by mean peak MCC among the
    configurations scored on all folds (ties: lower configuration first)."""
    configs = configurations(model)
    n_folds = cfg.model_training.tuning.folds
    summary = pd.DataFrame([{"config": i, **settings(model, c)} for i, c in enumerate(configs)])
    summary["folds_scored"] = 0
    if not scores.empty:
        grouped = scores.groupby("config")
        stats = grouped[list(SCORES)].agg(["mean", "std"])
        stats.columns = [f"{score}_{stat}" for score, stat in stats.columns]
        summary["folds_scored"] = summary.config.map(grouped.size()).fillna(0).astype(int)
        summary = summary.join(stats, on="config")
    complete = summary.folds_scored == n_folds
    rank = (summary.loc[complete, "peak_mcc_mean"].rank(ascending=False, method="first")
            if complete.any() else pd.Series(dtype=float))
    summary["rank"] = rank.reindex(summary.index).astype("Int64")
    return summary


def select(model, trial=False):
    """(selected configuration, summary): the highest mean peak MCC over the folds. Every
    configuration must be scored on every fold."""
    summary = cv_summary(model, cv_scores(model, trial))
    missing = summary[summary.folds_scored < cfg.model_training.tuning.folds]
    if len(missing):
        raise RuntimeError(f"{model}: {len(missing)} of {len(summary)} configurations not scored "
                           f"on every fold (e.g. {', '.join(map(str, missing.config[:10]))}).")
    return summary.sort_values("rank").iloc[0], summary


def package_versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "imbalanced-learn": imblearn.__version__}
