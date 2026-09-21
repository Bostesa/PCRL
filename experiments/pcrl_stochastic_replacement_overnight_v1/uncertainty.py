"""Paired household uncertainty with shared-household anchor aggregation.

The three anchors evaluate **overlapping people**, so anchors are not independent pseudo-replicates
(`STATISTICAL_PLAN.md` §2). The aggregate therefore resamples the **union household set once** and
averages each household's per-anchor paired difference within the resample. Independent per-anchor
bootstrapping would understate the spread and is not used for any aggregate claim.
"""
from __future__ import annotations

import numpy as np

Z_ONE_SIDED_95 = 1.6448536269514722


def _household_means(serial, diff, weight):
    """Weighted per-household numerator and denominator for one anchor."""
    uniq, inv = np.unique(serial, return_inverse=True)
    num = np.bincount(inv, weights=diff * weight, minlength=len(uniq))
    den = np.bincount(inv, weights=weight, minlength=len(uniq))
    return uniq, num, den


def paired_anchor_aggregate(anchors, *, n_boot: int = 4000, seed: int = 11,
                            z: float = Z_ONE_SIDED_95) -> dict:
    """Aggregate a paired contrast across anchors that share households.

    `anchors` is a list of `(serial, loss_left, loss_right, weight_or_None)`. The estimate is the
    mean across anchors of the weighted paired difference; the bootstrap resamples the union of
    household identifiers once per replicate and reuses that draw for every anchor.
    """
    prepared, union = [], set()
    for serial, left, right, weight in anchors:
        serial = np.asarray(serial)
        diff = np.asarray(left, np.float64) - np.asarray(right, np.float64)
        w = np.ones(len(diff)) if weight is None else np.asarray(weight, np.float64)
        uniq, num, den = _household_means(serial, diff, w)
        prepared.append({'ids': uniq, 'num': num, 'den': den})
        union.update(uniq.tolist())

    households = np.array(sorted(union))
    index = {h: i for i, h in enumerate(households)}
    # Dense per-anchor tables over the union, zero where an anchor lacks that household.
    tables = []
    for p in prepared:
        num = np.zeros(len(households))
        den = np.zeros(len(households))
        pos = np.fromiter((index[h] for h in p['ids'].tolist()), dtype=np.int64, count=len(p['ids']))
        num[pos] = p['num']
        den[pos] = p['den']
        tables.append((num, den))

    def aggregate(weights_per_household):
        vals = []
        for num, den in tables:
            d = float(np.dot(den, weights_per_household))
            if d <= 0:
                continue
            vals.append(float(np.dot(num, weights_per_household)) / d)
        return float(np.mean(vals)) if vals else np.nan

    ones = np.ones(len(households))
    estimate = aggregate(ones)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(households), len(households)),
                             minlength=len(households)).astype(np.float64)
        boot[b] = aggregate(counts)
    se = float(np.nanstd(boot, ddof=1))
    return {'estimate': float(estimate), 'bootstrap_se': se,
            'households_union': int(len(households)), 'anchors': len(anchors),
            'n_boot': int(n_boot), 'critical_value': float(z),
            'upper_one_sided': float(estimate + z * se),
            'lower_one_sided': float(estimate - z * se),
            'unit': 'household', 'aggregation': 'shared-household resample, mean across anchors',
            'note': ('anchors share people, so this is NOT independent per-anchor '
                     'pseudo-replication; release draws and optimizer seeds add no units')}


def simultaneous(bounds: dict, family_size: int, *, z_table=None) -> dict:
    """Bonferroni-adjust a set of one-sided bounds over a frozen family size."""
    from scipy.stats import norm
    alpha = 0.05 / float(family_size)
    z = float(norm.ppf(1 - alpha))
    out = {}
    for name, rec in bounds.items():
        out[name] = dict(rec)
        out[name]['critical_value_adjusted'] = z
        out[name]['upper_one_sided_adjusted'] = rec['estimate'] + z * rec['bootstrap_se']
        out[name]['lower_one_sided_adjusted'] = rec['estimate'] - z * rec['bootstrap_se']
    return {'family_size': int(family_size), 'alpha_per_test': alpha,
            'critical_value_adjusted': z, 'bounds': out}
