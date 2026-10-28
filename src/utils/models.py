"""
models.py
---------
The candidate algorithms of the model training, with the settings shared by all steps:
library defaults except those given in build() (100 trees for the ensembles, k = 5 for
k-nearest neighbours, up to 1000 iterations for logistic regression), random_state SEED
and the CPUs of the job.

- Logistic regression and k-nearest neighbours depend on the scale of the predictors and
  standardize them first (scaler fitted on the training data).
- balanced=True corrects the class imbalance by weighting the two classes inversely to
  their frequency in the training labels ("balanced"): class_weight, sample weights or,
  for XGBoost, scale_pos_weight. k-nearest neighbours has no class weights. With
  balanced=False no algorithm is weighted (the imbalance is then corrected by resampling).

Usage:
    from models import MODELS, build
    model, fit_kwargs = build("random_forest", y, cpus)       # balanced
    model.fit(X, y, **fit_kwargs)
"""

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
MODELS = ("logistic_regression", "decision_tree", "random_forest", "gradient_boosting",
          "xgboost", "extra_trees", "knn", "naive_bayes")

# Library defaults of parameters that build() leaves unset and get_params() reports as None
# (XGBoost 3.0.4, read from the booster's configuration). Reference values for the tables
# only; build() does not pass them.
LIBRARY_DEFAULTS = {
    "xgboost": {"max_depth": 6, "learning_rate": 0.3, "subsample": 1.0, "colsample_bytree": 1.0,
                "min_child_weight": 1, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
}


def build(name, y, cpus, balanced=True):
    """(estimator, keyword arguments of fit) for training labels y and a number of CPUs."""
    class_weight = "balanced" if balanced else None
    sample_weight = {"sample_weight": compute_sample_weight("balanced", y)} if balanced else {}
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            class_weight=class_weight, max_iter=1000, random_state=SEED)), {}
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight=class_weight, random_state=SEED), {}
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, class_weight=class_weight,
                                      random_state=SEED, n_jobs=cpus), {}
    if name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, random_state=SEED), sample_weight
    if name == "xgboost":
        weight = {"scale_pos_weight": float((y == 0).sum() / (y == 1).sum())} if balanced else {}
        return XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=SEED,
                             n_jobs=cpus, **weight), {}
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=100, class_weight=class_weight,
                                    random_state=SEED, n_jobs=cpus), {}
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=cpus)), {}
    if name == "naive_bayes":
        return GaussianNB(), sample_weight
    raise ValueError(f"Unknown model '{name}'; one of {', '.join(MODELS)}.")
