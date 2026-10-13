"""Exact single-query AUC of fixed finite-wire readers.

No fitting, loader, release sampler or scientific selection occurs here. Each
row's score distribution is obtained by evaluating the reader on the ACTUAL
emitted wires. Equal scores retain half credit, including different wire IDs
that the reader maps to exactly equal floating-point scores.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def validate_score_law(scores, mass, sex, groups=None):
    score = np.asarray(scores, dtype=np.float64)
    prob = np.asarray(mass, dtype=np.float64)
    y = np.asarray(sex)
    if score.ndim != 2 or score.shape != prob.shape or score.shape[1] == 0:
        raise ValueError("finite score/mass law dimensions differ")
    if y.shape != (len(score),) or y.dtype.kind not in "iu" or not np.isin(y, (0, 1)).all():
        raise ValueError("SEX must be an aligned binary vector")
    if not np.isfinite(score).all() or not np.isfinite(prob).all() or np.any(prob < 0):
        raise ValueError("finite score/mass law contains invalid values")
    if not np.allclose(prob.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("per-person conditional law is not normalized")
    if np.unique(y).size != 2:
        raise ValueError("AUC requires both classes")
    if groups is not None:
        group = np.asarray(groups)
        if group.shape != y.shape or group.dtype.kind not in "iu":
            raise ValueError("duplicate groups must be aligned integers")
        for value in np.unique(group):
            indexes = np.flatnonzero(group == value)
            if np.unique(y[indexes]).size != 1:
                raise ValueError("a shared-coin duplicate group crosses SEX classes")
            # Equal raw records share one release cache entry. Its exact score
            # law must therefore be the same, modulo support/action aliases.
            if len(indexes) > 1:
                first = indexes[0]
                ref = aggregate_score_mass(score[first], prob[first])
                for row in indexes[1:]:
                    other = aggregate_score_mass(score[row], prob[row])
                    if not np.array_equal(ref[0], other[0]) or not np.allclose(ref[1], other[1], atol=1e-12, rtol=0):
                        raise ValueError("duplicate cache records have different score laws")
    return score, prob, y.astype(np.int64, copy=False)


def aggregate_score_mass(scores, mass):
    score = np.asarray(scores, dtype=np.float64)
    prob = np.asarray(mass, dtype=np.float64)
    positive = prob > 0
    values, index = np.unique(score[positive], return_inverse=True)
    totals = np.bincount(index, weights=prob[positive], minlength=len(values))
    return values, totals


def weighted_rank_auc(positive_mass, negative_mass):
    """Ascending, exactly tied score bins; supports bootstrap batch axes."""
    pos = np.asarray(positive_mass, dtype=np.float64)
    neg = np.asarray(negative_mass, dtype=np.float64)
    if pos.shape != neg.shape or pos.ndim not in (1, 2) or pos.shape[-1] == 0:
        raise ValueError("positive/negative ascending score masses differ")
    if not np.isfinite(pos).all() or not np.isfinite(neg).all() or np.any(pos < 0) or np.any(neg < 0):
        raise ValueError("weighted rank masses are invalid")
    denom = pos.sum(axis=-1) * neg.sum(axis=-1)
    if np.any(denom <= 0):
        raise ValueError("weighted AUC needs positive mass in both classes")
    below = np.cumsum(neg, axis=-1) - neg
    return np.sum(pos * (below + 0.5 * neg), axis=-1) / denom


def integrated_auc(scores, mass, sex, weights=None, groups=None):
    """Expected AUC of one independently cached release per person/group.

    Scores and mass have [people, emitted-action] dimensions. Aliases may
    occupy several action columns; aggregation combines their score masses.
    Bootstrap weights count people via their duplicate group, never seeds.
    """
    score, prob, y = validate_score_law(scores, mass, sex, groups)
    w = np.ones(len(score), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("aligned nonnegative person weights required")
    support = prob > 0
    _, index = np.unique(score[support], return_inverse=True)
    labels = np.broadcast_to(y[:, None], prob.shape)[support]
    weighted = (prob * w[:, None])[support]
    length = int(index.max()) + 1
    pos = np.bincount(index[labels == 1], weights=weighted[labels == 1], minlength=length)
    neg = np.bincount(index[labels == 0], weights=weighted[labels == 0], minlength=length)
    return float(weighted_rank_auc(pos, neg))


def score_group_matrices(scores, mass, sex, groups):
    """Sparse group x exact-score mass, for paired bootstrap reuse."""
    score, prob, y = validate_score_law(scores, mass, sex, groups)
    group_values, group_index = np.unique(np.asarray(groups), return_inverse=True)
    support = prob > 0
    values, score_index = np.unique(score[support], return_inverse=True)
    row = np.broadcast_to(group_index[:, None], prob.shape)[support]
    labels = np.broadcast_to(y[:, None], prob.shape)[support]
    weights = prob[support]
    shape = (len(group_values), len(values))
    matrices = []
    for label in (1, 0):
        chosen = labels == label
        matrix = sparse.coo_matrix((weights[chosen], (row[chosen], score_index[chosen])), shape=shape).tocsr()
        matrix.sum_duplicates()
        matrices.append(matrix)
    return {"groups": group_values, "scores": values, "positive": matrices[0], "negative": matrices[1]}


def matrix_auc(matrix, group_counts):
    counts = np.asarray(group_counts, dtype=np.float64)
    if counts.ndim not in (1, 2) or counts.shape[-1] != len(matrix["groups"]):
        raise ValueError("bootstrap count/group matrix dimensions differ")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("bootstrap counts must be nonnegative")
    positive = np.asarray(counts @ matrix["positive"])
    negative = np.asarray(counts @ matrix["negative"])
    return weighted_rank_auc(positive, negative)
