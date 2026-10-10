"""Empirical task-only rate of actual public score-channel pair identities.

This uses no protected labels and is an objective on the supplied FIT people,
not a population information or finite-attacker privacy certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class RateReceipt:
    rate_nats: float
    marginal_entropy_nats: float
    conditional_entropy_nats: float
    occupied_pairs: int
    people: int
    max_row_sum_residual: float


def _entropy(masses) -> float:
    return -math.fsum(p * math.log(p) for p in masses if p > 0)


def empirical_public_pair_rate(rows, *, weights=None) -> RateReceipt:
    """Compute H(Z_pair)−E_X H(Z_pair|X) after global pair aggregation.

    ``rows`` holds one actual sparse public output law per admitted FIT person.
    Duplicate local actions naming the same public pair are merged before
    conditional entropy. Tiny float row-sum residuals are recorded and each
    row is normalized only for this arithmetic; deployed q stays unchanged.
    """
    laws = tuple(tuple(row) for row in rows)
    if not laws:
        raise ValueError("task-only empirical rate needs fitting people")
    n = len(laws)
    if weights is None:
        person_weights = (1.0 / n,) * n
    else:
        raw = tuple(weights)
        if (len(raw) != n or any(isinstance(w, bool) or not isinstance(w, Real)
                                or not math.isfinite(float(w)) or w < 0 for w in raw)):
            raise ValueError("person weights must be finite, nonnegative and aligned")
        total_weight = math.fsum(float(w) for w in raw)
        if total_weight <= 0:
            raise ValueError("person weights need positive total mass")
        person_weights = tuple(float(w) / total_weight for w in raw)

    marginal: dict[tuple[int, int], float] = {}
    conditional_entropy = 0.0
    max_residual = 0.0
    for row_index, (row, person_weight) in enumerate(zip(laws, person_weights)):
        if not row:
            raise ValueError(f"row {row_index}: empty public output law")
        grouped: dict[tuple[int, int], list[float]] = {}
        for id1, id2, mass in row:
            if (isinstance(id1, bool) or isinstance(id2, bool)
                    or not isinstance(id1, Integral) or not isinstance(id2, Integral)
                    or id1 < 0 or id2 < 0):
                raise ValueError(f"row {row_index}: invalid public bank ID")
            if (isinstance(mass, bool) or not isinstance(mass, Real)
                    or not math.isfinite(float(mass)) or mass < 0):
                raise ValueError(f"row {row_index}: invalid action mass")
            grouped.setdefault((int(id1), int(id2)), []).append(float(mass))
        grouped_mass = {key: math.fsum(values) for key, values in grouped.items()}
        row_total = math.fsum(grouped_mass.values())
        residual = abs(row_total - 1.0)
        max_residual = max(max_residual, residual)
        if residual > 1e-12 or row_total <= 0:
            raise ValueError(f"row {row_index}: public law is not normalized")
        q = {key: mass / row_total for key, mass in grouped_mass.items() if mass > 0}
        conditional_entropy += person_weight * _entropy(q.values())
        for key, mass in q.items():
            marginal[key] = marginal.get(key, 0.0) + person_weight * mass
    marginal_entropy = _entropy(marginal.values())
    rate = marginal_entropy - conditional_entropy
    if rate < -1e-10:
        raise ArithmeticError("empirical information rate became materially negative")
    return RateReceipt(max(rate, 0.0), marginal_entropy, conditional_entropy,
                       sum(mass > 0 for mass in marginal.values()), n, max_residual)
