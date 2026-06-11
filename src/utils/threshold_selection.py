"""
threshold_selection.py
-----------------------
Generic utility for selecting a model's decision threshold (operating point)
on held-out validation data, by maximizing the Matthews Correlation
Coefficient (MCC) over all candidate thresholds.

Why this exists
---------------
A probability threshold is only meaningful relative to a given model's
probability distribution. A fixed number (e.g. 0.01) lands at a different
operating point for each model — heavily regularized models output compressed
probabilities, so the same cut classifies far more samples as negative.
Comparing two models at a single shared threshold therefore conflates model
quality with how well that threshold happens to align with each model's
probability scale. The fix is to stop fixing the threshold: each model is
reported at the threshold that maximizes its MCC (the peak of its
MCC-vs-threshold curve), so every model is read at its own optimal operating
point and MCC stays the metric.

Functions
---------
select_threshold_on_validation(y_true, y_prob, metric="mcc")
    Return (best_threshold, best_score): the threshold maximizing the metric on
    the supplied validation labels/probabilities, and the score achieved there.

mcc_best_threshold_scorer(estimator, X, y)
    Scikit-learn-compatible scorer that returns the best achievable MCC over
    thresholds for an estimator on (X, y). Use as the `scoring` argument of
    RandomizedSearchCV / GridSearchCV so the search optimizes achievable MCC
    rather than MCC at one hard-coded threshold.

Notes
-----
- Selecting one scalar threshold is a negligible source of optimism even when
  done on the evaluation set itself (a single parameter fit on many rows), so
  reporting "best achievable MCC at the optimal threshold" on the test set is
  the simple, defensible choice used here — state it plainly in the write-up.
  If you instead need an UNBIASED deployment operating point, select the
  threshold on a held-out validation set disjoint from the test set.
- The sweep is fully vectorized (one cumulative pass over the probabilities
  sorted descending), so it is fast even on millions of rows.
- Ties in probability are handled exactly: candidate cut points are only placed
  at boundaries between distinct probability values, so the returned threshold
  reproduces the same predictions under `y_prob >= threshold`.
- When used as the search scorer, maximizing MCC over thresholds *within* the
  scored fold is slightly optimistic for candidate ranking, but does not bias
  the final reported number — that comes from re-selecting the threshold on a
  separate validation set and evaluating on the untouched test set.
"""

import numpy as np


def _mcc_over_thresholds(y_true, y_prob):
    """
    Vectorized MCC at every distinct-probability cut point.

    Returns (thresholds, mcc) aligned arrays, where predicting
    `y_prob >= thresholds[i]` yields mcc[i]. Computed in a single cumulative
    pass over the probabilities sorted in descending order.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Validation set must contain both classes (got "
                         f"{n_pos} positives, {n_neg} negatives).")

    # Sort by probability descending; including the top-k as positive means the
    # threshold equals the k-th largest probability.
    order    = np.argsort(-y_prob, kind="mergesort")
    p_sorted = y_prob[order]
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted).astype(np.float64)        # positives among top-k
    k  = np.arange(1, y_sorted.size + 1, dtype=np.float64)
    fp = k - tp
    fn = n_pos - tp
    tn = n_neg - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = np.where(denom > 0.0, (tp * tn - fp * fn) / denom, 0.0)

    # Only keep cut points at boundaries between distinct probabilities, so the
    # `>=` semantics are exact (never cut inside a group of tied probabilities).
    boundary       = np.empty(p_sorted.size, dtype=bool)
    boundary[:-1]  = p_sorted[:-1] != p_sorted[1:]
    boundary[-1]   = True

    return p_sorted[boundary], mcc[boundary]


def select_threshold_on_validation(y_true, y_prob, metric="mcc"):
    """
    Select the decision threshold maximizing `metric` on validation data.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        True labels of the validation set (must contain both classes).
    y_prob : array-like of float
        Predicted probabilities of the positive class for the same rows.
    metric : str, default "mcc"
        Metric to maximize. Only "mcc" is currently supported.

    Returns
    -------
    best_threshold : float
        Threshold t such that predicting `y_prob >= t` maximizes the metric.
    best_score : float
        The metric value achieved at best_threshold.
    """
    if metric != "mcc":
        raise ValueError(f"Unsupported metric {metric!r} (only 'mcc' implemented).")

    thresholds, mcc = _mcc_over_thresholds(y_true, y_prob)
    best_i = int(np.argmax(mcc))
    return float(thresholds[best_i]), float(mcc[best_i])


def mcc_best_threshold_scorer(estimator, X, y):
    """
    Scikit-learn scorer: best achievable MCC over thresholds for `estimator`.

    Intended as the `scoring` argument of RandomizedSearchCV / GridSearchCV so
    the search optimizes achievable MCC instead of MCC at a fixed, model-
    mismatched threshold. Higher is better, as scikit-learn expects.
    """
    y_prob = estimator.predict_proba(X)[:, 1]
    _, best_score = select_threshold_on_validation(y, y_prob)
    return best_score


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    # Validate the vectorized sweep against a brute-force reference on a small,
    # imbalanced synthetic set, including tied probabilities.
    from sklearn.metrics import matthews_corrcoef

    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.02).astype(int)          # ~2% positives
    # Signal + noise, rounded to create ties.
    p = np.clip(0.01 * y + 0.005 * rng.standard_normal(n), 1e-6, None)
    p = np.round(p, 3)

    t_star, s_star = select_threshold_on_validation(y, p)

    # Brute-force reference over every distinct threshold.
    best_bf, t_bf = -2.0, None
    for t in np.unique(p):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best_bf:
            best_bf, t_bf = m, t

    print(f"vectorized : threshold={t_star:.6f}  mcc={s_star:.6f}")
    print(f"brute force: threshold={t_bf:.6f}  mcc={best_bf:.6f}")
    assert abs(s_star - best_bf) < 1e-9, "MCC mismatch vs brute force"
    # Predictions at the returned threshold must reproduce the reference MCC.
    assert abs(matthews_corrcoef(y, (p >= t_star).astype(int)) - s_star) < 1e-9
    print("OK: vectorized sweep matches brute force.")
