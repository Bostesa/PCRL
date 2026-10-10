"""Small independent arithmetic oracle for the D and A release contracts.

No candidate, runner, shared evaluator, or selection module is imported here.
This module deliberately uses explicit finite sums so that tiny fixtures can
challenge vectorized fitting implementations. Real-data execution is subject
to the campaign's sealed R03 numerical queue.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def _binary_matrix(values: Sequence[Sequence[int]]) -> np.ndarray:
    bits = np.asarray(values)
    if bits.ndim != 2 or bits.shape[0] == 0 or bits.shape[1] == 0:
        raise ValueError("nonempty binary matrix required")
    if not np.issubdtype(bits.dtype, np.integer) or not np.isin(bits, (0, 1)).all():
        raise ValueError("hard code must contain only integer bits")
    return bits.astype(np.int8, copy=False)


def gaussian_binary_kernel(a: Sequence[int], b: Sequence[int], bandwidth: float) -> float:
    """RBF on actual hard bits, exp(-||a-b||^2 / (2 sigma^2))."""
    x = _binary_matrix([a, b])
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("positive finite bandwidth required")
    distance = sum(int(left != right) for left, right in zip(x[0], x[1]))
    return math.exp(-distance / (2.0 * bandwidth * bandwidth))


def multiscale_binary_kernel(
    a: Sequence[int], b: Sequence[int], bandwidths: Iterable[float]
) -> float:
    widths = tuple(float(x) for x in bandwidths)
    if not widths:
        raise ValueError("at least one bandwidth required")
    return sum(gaussian_binary_kernel(a, b, width) for width in widths) / len(widths)


def biased_mmd2(
    first: Sequence[Sequence[int]],
    second: Sequence[Sequence[int]],
    bandwidths: Iterable[float],
) -> float:
    """All ordered pairs, including diagonals: nonnegative biased MMD^2."""
    x, y = _binary_matrix(first), _binary_matrix(second)
    if x.shape[1] != y.shape[1]:
        raise ValueError("group code widths differ")
    widths = tuple(float(v) for v in bandwidths)
    xx = sum(multiscale_binary_kernel(a, b, widths) for a in x for b in x)
    yy = sum(multiscale_binary_kernel(a, b, widths) for a in y for b in y)
    xy = sum(multiscale_binary_kernel(a, b, widths) for a in x for b in y)
    value = xx / len(x) ** 2 + yy / len(y) ** 2 - 2.0 * xy / (len(x) * len(y))
    if value < -1e-12:
        raise ArithmeticError("kernel MMD squared became negative")
    return max(0.0, value)


def product_pair_mmd2(
    first: Sequence[tuple[Sequence[int], Sequence[int]]],
    second: Sequence[tuple[Sequence[int], Sequence[int]]],
    bandwidths: Iterable[float],
) -> float:
    """Biased MMD on aligned pairs with k_pair = k_1 * k_2."""
    x, y = tuple(first), tuple(second)
    if not x or not y:
        raise ValueError("both SEX groups required")
    widths = tuple(float(v) for v in bandwidths)

    def kernel(pair_a, pair_b):
        return multiscale_binary_kernel(pair_a[0], pair_b[0], widths) * multiscale_binary_kernel(
            pair_a[1], pair_b[1], widths
        )

    xx = sum(kernel(a, b) for a in x for b in x) / len(x) ** 2
    yy = sum(kernel(a, b) for a in y for b in y) / len(y) ** 2
    xy = sum(kernel(a, b) for a in x for b in y) / (len(x) * len(y))
    value = xx + yy - 2.0 * xy
    if value < -1e-12:
        raise ArithmeticError("pair MMD squared became negative")
    return max(0.0, value)


def expected_task_cost(
    action_probabilities: Sequence[float],
    public_probability_vectors: Sequence[Sequence[float]],
    label: int,
    *,
    clip: float = 1e-12,
) -> tuple[float, float]:
    """Expected single-draw natural-log NLL and summed multiclass Brier."""
    q = np.asarray(action_probabilities, dtype=np.float64)
    p = np.asarray(public_probability_vectors, dtype=np.float64)
    if (q.ndim != 1 or p.ndim != 2 or p.shape[0] != len(q) or not len(q)
            or p.shape[1] < 2 or not np.isfinite(q).all() or not np.isfinite(p).all()
            or np.any(q < 0) or np.any(p < 0) or np.any(p > 1)
            or abs(float(q.sum()) - 1) > 1e-12
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
            or label < 0 or label >= p.shape[1] or clip <= 0):
        raise ValueError("invalid finite channel or task vectors")
    target = np.eye(p.shape[1])[label]
    nll = sum(float(qi) * -math.log(max(float(pi[label]), clip)) for qi, pi in zip(q, p))
    brier = sum(float(qi) * float(np.sum((pi - target) ** 2)) for qi, pi in zip(q, p))
    return nll, brier


def aggregate_public_actions(
    action_probabilities: Sequence[float], public_ids: Sequence[object]
) -> dict[object, float]:
    """Merge input-local actions by their global transmitted public identity."""
    q = np.asarray(action_probabilities, dtype=np.float64)
    if (q.ndim != 1 or len(q) != len(public_ids) or not len(q)
            or not np.isfinite(q).all() or np.any(q < 0)
            or abs(float(q.sum()) - 1) > 1e-12):
        raise ValueError("invalid action law")
    law: dict[object, float] = defaultdict(float)
    for probability, code in zip(q, public_ids):
        law[code] += float(probability)
    return dict(law)


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Weighted-free Mann-Whitney AUC with half credit for tied scores."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if (s.ndim != 1 or y.ndim != 1 or len(s) != len(y) or not len(s)
            or not np.isfinite(s).all() or not np.isin(y, (0, 1)).all()
            or len(np.unique(y)) != 2):
        raise ValueError("aligned finite scores and two binary classes required")
    pos, neg = s[y == 1], s[y == 0]
    return float(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)))


def pack_ten_bit_pair(first: Sequence[int], second: Sequence[int]) -> np.ndarray:
    """Collision-free ordered pair identity, with promotion before multiply."""
    a, b = np.asarray(first), np.asarray(second)
    if (a.ndim != 1 or b.shape != a.shape or not np.issubdtype(a.dtype, np.integer)
            or not np.issubdtype(b.dtype, np.integer) or np.any(a < 0) or np.any(a >= 1024)
            or np.any(b < 0) or np.any(b >= 1024)):
        raise ValueError("two aligned ten-bit code ID arrays required")
    return a.astype(np.int64) * 1024 + b.astype(np.int64)
