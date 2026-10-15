"""Paired duplicate-group bootstrap shared across models and paired seeds.

One set of group resamples (seed 2026101501, B = 2048) is drawn per role and
applied to every model and both seeds, so seed and model comparisons are
paired. Intervals are conditional on the fitted models and readers and do not
account for the repeated development history of these people.
Simultaneous intervals: point +/- t(1 - alpha/(2K), G-1) * bootstrap SE
(Bonferroni over the frozen K-endpoint family, alpha = 0.05).
"""
from __future__ import annotations
import numpy as np

SEED = 2026101501
B = 2048


def group_weights(unit, seed=SEED, draws=B):
    unit = np.asarray(unit)
    groups, inverse = np.unique(unit, return_inverse=True)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(draws, len(groups)))
    counts = np.zeros((draws, len(groups)), dtype=np.float64)
    for b in range(draws):
        counts[b] = np.bincount(picks[b], minlength=len(groups))
    return counts[:, inverse], len(groups)


def weighted_mean(v, w):
    v = np.asarray(v, dtype=np.float64)
    return (w * v[None, :]).sum(1) / w.sum(1)


def auc_point_and_draws(y, s, w):
    from task_conditioned_representation.inference import auc, bootstrap_auc
    return auc(y, s), bootstrap_auc(y, s, w)


def critical(K, n_groups, alpha=0.05):
    from scipy import stats
    return float(stats.t.ppf(1 - alpha / (2 * K), n_groups - 1))


def interval(point, draws, crit):
    se = float(np.std(draws, ddof=1))
    return {'point': float(point), 'se': se, 'lower': float(point - crit * se), 'upper': float(point + crit * se), 'critical': crit}


def percentile(point, draws, level=0.95):
    a = (1 - level) / 2
    return {'point': float(point), 'lower': float(np.quantile(draws, a)), 'upper': float(np.quantile(draws, 1 - a)), 'level': level}


def status_upper(iv, cap):
    """No-excess endpoint: PASS if the whole interval is <= cap, VIOLATION if wholly above, else UNRESOLVED."""
    if iv['upper'] <= cap:
        return 'PASS'
    if iv['lower'] > cap:
        return 'ESTABLISHED_VIOLATION'
    return 'UNRESOLVED'


def status_benefit(iv, margin):
    """Benefit endpoint: ESTABLISHED if lower > margin; SHORTFALL if the whole interval is below margin."""
    if iv['lower'] > margin:
        return 'ESTABLISHED'
    if iv['upper'] < margin:
        return 'ESTABLISHED_SHORTFALL'
    return 'UNRESOLVED'
