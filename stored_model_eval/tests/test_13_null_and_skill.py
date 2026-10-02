"""Permutation null with the max statistic re-selected (worst pair is biased above 0.5 under the null), and
proper-scoring skill metrics: zero for the fit prior, positive for a direct signal."""
import numpy as np

from stored_model_eval.inference import permutation_null
from stored_model_eval.metrics import brier_skill, logloss_reduction, pairwise_auc, prior_from_fit


def test_permutation_null_for_worst_pair():
    rng = np.random.default_rng(31)
    n, K = 1200, 4
    y = rng.integers(0, K, n)
    P = rng.dirichlet(np.ones(K), size=n)  # scores independent of y
    stat = lambda labels: pairwise_auc(labels, P, 50)["max"]  # noqa: E731
    r = permutation_null(stat, y, np.arange(n), n_perm=60, seed=1)
    assert r["null_mean"] > 0.5  # max over 6 pairs is biased upward under the null
    assert r["p_value"] > 0.05
    Pi = P.copy()
    Pi[np.arange(n), y] += 0.4  # planted signal
    r2 = permutation_null(lambda labels: pairwise_auc(labels, Pi, 50)["max"], y, np.arange(n), n_perm=60, seed=1)
    assert r2["p_value"] < 0.05


def test_skill_metrics_zero_at_prior_positive_with_signal():
    rng = np.random.default_rng(32)
    y = rng.integers(0, 3, 900)
    prior = prior_from_fit(y[:500], 3)
    Pp = np.tile(prior, (400, 1))
    assert abs(brier_skill(y[500:], Pp, prior)) < 1e-12 and abs(logloss_reduction(y[500:], Pp, prior)) < 1e-12
    Ps = 0.5 * np.eye(3)[y[500:]] + 0.5 * Pp
    assert brier_skill(y[500:], Ps, prior) > 0.3 and logloss_reduction(y[500:], Ps, prior) > 0.3
