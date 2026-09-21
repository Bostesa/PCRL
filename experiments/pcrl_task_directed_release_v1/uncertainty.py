"""Registered paired household bounds, conditional on the already fitted models.

The input list is the frozen endpoint family: its length alone determines the
Bonferroni divisor. Every contrast contains exactly three anchor records. The
caller supplies oriented per-person paired expected-loss differences and either
unit weights or survey weights; no fit, selection, clipping, or orientation
change occurs here.

A replicate draws H multinomial household multiplicities on the common union of
H household identifiers, once for the entire family. Each anchor recomputes its
weighted numerator/denominator ratio, then the three ratios are averaged equally.
A zero denominator anywhere rejects that entire draw, including for all other
contrasts. This rare-event conditioning is explicitly counted, not silently
replaced by a two-anchor estimate. All-zero pairs retain exact zero variance.

Household tables use sparse column storage, duplicate tables are reused exactly,
and at most 128 draws are held at once. Welford batch merging computes sample SEs
without retaining the n_boot-by-endpoint replicate matrix. No household IDs or
per-person data are returned in the public aggregate result.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse
from scipy.stats import norm

BATCH_SIZE = 128
MAX_AUXILIARY_BYTES = 1_500_000_000  # conservative cap below the requested 2 GB


def _hash_column(indices, values):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return h.digest()


def _prepare_anchor(anchor):
    if not isinstance(anchor, dict) or not {'household', 'difference', 'weights'} <= set(anchor):
        raise ValueError('Each anchor requires household, difference, and weights arrays')
    household = np.asarray(anchor['household'])
    if household.ndim != 1 or not len(household):
        raise ValueError('Household IDs must be a nonempty one-dimensional string array')
    if household.dtype.kind != 'U':
        if household.dtype.kind != 'O' or not all(isinstance(v, (str, np.str_)) for v in household):
            raise ValueError('Household IDs must be strings, never missing values or numeric IDs')
        household = household.astype(str)
    if any(not value.strip() for value in household):
        raise ValueError('Household IDs must be nonempty strings')
    difference = np.asarray(anchor['difference'], dtype=np.float64)
    weights = np.asarray(anchor['weights'], dtype=np.float64)
    if difference.shape != household.shape or weights.shape != household.shape:
        raise ValueError('Household, difference, and weights arrays must align')
    if (not np.isfinite(difference).all() or not np.isfinite(weights).all()
            or (weights < 0).any()):
        raise ValueError('Finite differences and finite nonnegative weights are required')
    with np.errstate(over='ignore', invalid='ignore'):
        total = float(weights.sum())
        weighted = difference * weights
    if total <= 0 or not np.isfinite(total) or not np.isfinite(weighted).all():
        raise ValueError('Every anchor needs a positive finite weight total and finite weighted losses')
    ids, inverse = np.unique(household, return_inverse=True)
    numerator = np.bincount(inverse, weights=weighted, minlength=len(ids))
    denominator = np.bincount(inverse, weights=weights, minlength=len(ids))
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise ValueError('Household aggregation produced nonfinite values')
    with np.errstate(over='ignore', invalid='ignore'):
        estimate = float(numerator.sum() / denominator.sum())
    if not np.isfinite(estimate):
        raise ValueError('Anchor estimate is nonfinite')
    return {'ids': ids, 'num': numerator, 'den': denominator, 'estimate': estimate,
            'persons': len(household), 'households': len(ids),
            'positive_weight_households': int(np.count_nonzero(denominator)),
            'identical': bool(np.all(difference == 0))}


class _SparseColumns:
    """Intern exactly equal household vectors so repeated ancestors cost one column."""

    def __init__(self):
        self.columns = []
        self.lookup = {}
        self.nonzero = 0

    def add(self, indices, values):
        keep = values != 0
        indices, values = np.asarray(indices[keep], np.int64), np.asarray(values[keep], np.float64)
        key = _hash_column(indices, values)
        if key in self.lookup:
            column = self.lookup[key]
            previous = self.columns[column]
            if not (np.array_equal(indices, previous[0]) and np.array_equal(values, previous[1])):
                raise RuntimeError('Household aggregate hash collision')
            return column
        column = len(self.columns)
        self.lookup[key] = column
        self.columns.append((indices, values))
        self.nonzero += len(indices)
        return column

    def matrix(self, households):
        # CSC construction from sorted row indices avoids an H-by-endpoint dense table.
        sizes = np.fromiter((len(v) for _, v in self.columns), dtype=np.int64)
        indptr = np.concatenate(([0], np.cumsum(sizes)))
        indices = np.concatenate([i for i, _ in self.columns])
        values = np.concatenate([v for _, v in self.columns])
        return sparse.csc_matrix((values, indices, indptr), shape=(households, len(self.columns)))


def paired_household_bounds(contrasts, *, n_boot=10000, seed=20260921, alpha=.05):
    """Return simultaneous normal bounds using shared-household bootstrap sample SEs.

    Returns ``bounds[id]`` with estimate, bootstrap_se, lower/upper and the same
    corresponding adjusted one-sided bound aliases. ``family_size`` is always
    len(contrasts), and z is norm.isf(alpha/(2*family_size)). Replicate bookkeeping
    includes global zero-denominator rejections. Insufficient valid replicates after
    max(10000,100*n_boot) attempts raises; no reduced-replicate answer is returned.
    """
    if not isinstance(contrasts, (list, tuple)) or not contrasts:
        raise ValueError('A nonempty explicit contrast list is required')
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError('n_boot must be an integer of at least 2')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (isinstance(alpha, (bool, str, bytes)) or not np.isscalar(alpha)
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('alpha must lie strictly between 0 and 1')
    ids, prepared, union = [], [], set()
    for contrast in contrasts:
        if not isinstance(contrast, dict):
            raise ValueError('Each contrast must be a dictionary')
        ident = contrast.get('id')
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError('Contrast IDs must be nonempty unique strings; duplicate ID rejected')
        anchors = contrast.get('anchors')
        if not isinstance(anchors, (list, tuple)) or len(anchors) != 3:
            raise ValueError('Every contrast must have exactly three anchors')
        records = [_prepare_anchor(anchor) for anchor in anchors]
        for record in records:
            union.update(record['ids'].tolist())
        ids.append(ident)
        prepared.append(records)
    households = np.asarray(sorted(union), dtype=str)
    h, m = len(households), len(ids)
    index = {value: j for j, value in enumerate(households)}
    nums, dens = _SparseColumns(), _SparseColumns()
    num_columns, den_columns = np.empty((m, 3), dtype=np.int64), np.empty((m, 3), dtype=np.int64)
    estimates = np.empty(m)
    identical = np.empty(m, dtype=bool)
    for c, records in enumerate(prepared):
        estimates[c] = np.mean([record['estimate'] for record in records])
        identical[c] = all(record['identical'] for record in records)
        for a, record in enumerate(records):
            positions = np.fromiter((index[v] for v in record['ids']), dtype=np.int64)
            num_columns[c, a] = nums.add(positions, record['num'])
            den_columns[c, a] = dens.add(positions, record['den'])
            # Keep only public aggregate bookkeeping after interning the private arrays.
            del record['ids'], record['num'], record['den']
    del union, index, households
    # Include duplicate staging/CSC construction and generous dense-batch temporaries.
    aggregate_bytes_bound = 64 * (nums.nonzero + dens.nonzero) + 32 * (m * 3 + h)
    batch_bytes_bound = BATCH_SIZE * 8 * (2 * h + len(nums.columns) + len(dens.columns) + 20 * m)
    if aggregate_bytes_bound + batch_bytes_bound > MAX_AUXILIARY_BYTES:
        raise MemoryError('Explicit contrast family exceeds the 1.5 GB auxiliary-memory safety bound')
    numerator_matrix, denominator_matrix = nums.matrix(h), dens.matrix(h)
    numerator_groups, denominator_groups = len(nums.columns), len(dens.columns)
    # Release the intermediate per-household arrays before generating any draws.
    del nums, dens
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(h, 1 / h)
    accepted = attempted = rejected = 0
    max_attempts = max(10000, 100 * int(n_boot))
    running_mean, m2 = np.zeros(m), np.zeros(m)
    while accepted < n_boot and attempted < max_attempts:
        batch = min(BATCH_SIZE, int(n_boot) - accepted, max_attempts - attempted)
        multiplicities = rng.multinomial(h, probabilities, size=batch)
        # Sparse-by-dense multiplication gives only batch-by-unique-column matrices.
        denominators = np.asarray(denominator_matrix.T @ multiplicities.T).T
        valid = np.all(denominators > 0, axis=1)
        if not np.isfinite(denominators).all():
            raise FloatingPointError('Nonfinite bootstrap denominators')
        attempted += batch
        rejected += int(np.count_nonzero(~valid))
        if not valid.any():
            continue
        numerators = np.asarray(numerator_matrix.T @ multiplicities[valid].T).T
        denominator_values = denominators[valid][:, den_columns]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            anchor_ratios = numerators[:, num_columns] / denominator_values
            values = anchor_ratios.mean(axis=2)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite weighted-ratio bootstrap value')
        values[:, identical] = 0.  # preserve the exact paired-zero invariant
        count = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        batch_m2 = np.einsum('ij,ij->j', centered, centered)
        delta = mean - running_mean
        total = accepted + count
        m2 += batch_m2 + delta * delta * accepted * count / total
        running_mean += delta * count / total
        accepted = total
    if accepted != n_boot:
        raise RuntimeError(f'Only {accepted}/{n_boot} globally valid replicates after {attempted} draws; '
                           f'{rejected} whole draws had at least one zero anchor denominator')
    se = np.sqrt(np.maximum(m2, 0.) / (accepted - 1))
    estimates[identical] = 0.
    se[identical] = 0.
    z = float(norm.isf(float(alpha) / (2 * m)))
    if not np.isfinite(se).all() or not np.isfinite(z):
        raise FloatingPointError('Nonfinite bootstrap uncertainty')
    bounds = {}
    for j, ident in enumerate(ids):
        lower, upper = float(estimates[j] - z * se[j]), float(estimates[j] + z * se[j])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise FloatingPointError('Nonfinite simultaneous interval bound')
        bounds[ident] = {
            'estimate': float(estimates[j]), 'bootstrap_se': float(se[j]),
            'lower': lower, 'upper': upper,
            'lower_one_sided_adjusted': lower, 'upper_one_sided_adjusted': upper,
            'critical_value_adjusted': z, 'identical_pairs': bool(identical[j]),
            'anchor_estimates': [record['estimate'] for record in prepared[j]],
            'anchor_person_counts': [record['persons'] for record in prepared[j]],
            'anchor_household_counts': [record['households'] for record in prepared[j]],
            'anchor_positive_weight_households': [record['positive_weight_households'] for record in prepared[j]],
        }
    return {
        'family_size': m, 'contrast_ids': ids, 'alpha': float(alpha),
        'alpha_per_tail': float(alpha) / (2 * m), 'critical_value_adjusted': z,
        'households_union': h, 'anchors': 3, 'bounds': bounds,
        'bootstrap': {'requested': int(n_boot), 'accepted': accepted, 'attempted': attempted,
                      'rejected_zero_denominator': rejected, 'seed': int(seed),
                      'batch_size': BATCH_SIZE, 'max_attempts': max_attempts,
                      'numerator_aggregate_groups': numerator_groups,
                      'denominator_aggregate_groups': denominator_groups,
                      'auxiliary_memory_bound_bytes': aggregate_bytes_bound + batch_bytes_bound,
                      'zero_denominator_policy': 'reject entire common draw for every endpoint',
                      'SE_ddof': 1},
        'resampling': 'one common union-household multinomial multiplicity across all contrasts and anchors',
        'estimator': 'equal mean of three anchor weighted ratios; numerator and denominator recomputed per draw',
        'correction': 'Bonferroni simultaneous two-sided normal bounds, z_(1-alpha/(2*M))',
        'scope': 'conditional on fitted models; no training-seed uncertainty or fresh-population guarantee',
    }
