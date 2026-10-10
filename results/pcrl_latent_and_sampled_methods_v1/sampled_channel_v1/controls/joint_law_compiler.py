"""Factor a fitted joint *public-output* law in either sampling order.

The two factorizations are engineering aliases of the same release law. They
are not freeze-first learned sequential controls, and cannot establish a
joint-sampling advantage. IDs here are globally canonical public bank IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral
from typing import Iterable


@dataclass(frozen=True)
class SequentialCompilation:
    order: str
    first_ids: tuple[int, ...]
    second_ids: tuple[int, ...]
    marginal: tuple[Fraction | float, ...]
    conditional: tuple[tuple[Fraction | float, ...], ...]

    def conditional_for(self, first_id: int) -> dict[int, Fraction | float]:
        try:
            row = self.first_ids.index(first_id)
        except ValueError as exc:
            raise ValueError("first output ID is outside the admitted bank choices") from exc
        return dict(zip(self.second_ids, self.conditional[row]))

    def replay(self) -> dict[tuple[int, int], Fraction | float]:
        """Reconstruct the same public pair law, omitting zero-mass symbols."""
        reconstructed = {}
        for first, mass, row in zip(self.first_ids, self.marginal, self.conditional):
            for second, conditional_mass in zip(self.second_ids, row):
                value = mass * conditional_mass
                if value:
                    pair = (first, second) if self.order == "12" else (second, first)
                    reconstructed[pair] = value
        return reconstructed


def _ids(values: Iterable[int], label: str) -> tuple[int, ...]:
    ids = tuple(values)
    if (not ids or len(set(ids)) != len(ids)
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                   for value in ids)):
        raise ValueError(f"{label} must contain distinct nonnegative public IDs")
    return tuple(sorted(int(value) for value in ids))


def compile_joint_actions(
    actions: Iterable[tuple[int, int, Fraction | float]],
    allowed1: Iterable[int],
    allowed2: Iterable[int],
    *,
    order: str,
) -> SequentialCompilation:
    """Compile q(z1,z2|x) into q(first|x)q(second|first,x).

    A zero-marginal conditional is one-hot on the lowest admitted second ID.
    This convention has no effect on the public pair law but makes a replay
    table total. Rational masses replay exactly; float masses use usual roundoff.
    """
    if order not in ("12", "21"):
        raise ValueError("sampling order must be '12' or '21'")
    ids1, ids2 = _ids(allowed1, "recipient 1 choices"), _ids(allowed2, "recipient 2 choices")
    rows = tuple(actions)
    if not rows:
        raise ValueError("joint law needs at least one public pair action")
    rational = all(isinstance(item[2], Fraction) for item in rows)
    zero = Fraction(0) if rational else 0.0
    one = Fraction(1) if rational else 1.0
    joint: dict[tuple[int, int], Fraction | float] = {}
    for id1, id2, weight in rows:
        if (isinstance(id1, bool) or isinstance(id2, bool)
                or not isinstance(id1, Integral) or not isinstance(id2, Integral)):
            raise ValueError("joint action IDs must be integers, not booleans")
        if id1 not in ids1 or id2 not in ids2:
            raise ValueError("joint action names an inadmissible public bank ID")
        if (isinstance(weight, bool) or not isinstance(weight, (int, float, Fraction))
                or not math.isfinite(float(weight)) or weight < 0):
            raise ValueError("joint action has invalid probability")
        pair = (int(id1), int(id2))
        joint[pair] = joint.get(pair, zero) + weight
    total = sum(joint.values(), zero)
    if (total != one if rational else abs(total - one) > 1e-12):
        raise ValueError("joint public-output law must sum to one")

    first_ids, second_ids = (ids1, ids2) if order == "12" else (ids2, ids1)
    marginal = []
    conditional = []
    for first in first_ids:
        values = tuple(
            joint.get((first, second) if order == "12" else (second, first), zero)
            for second in second_ids
        )
        first_mass = sum(values, zero)
        marginal.append(first_mass)
        if first_mass:
            conditional.append(tuple(value / first_mass for value in values))
        else:
            conditional.append(tuple(one if second == second_ids[0] else zero
                                     for second in second_ids))
    return SequentialCompilation(order, first_ids, second_ids,
                                 tuple(marginal), tuple(conditional))


def compile_both_orders(
    actions: Iterable[tuple[int, int, Fraction | float]],
    allowed1: Iterable[int],
    allowed2: Iterable[int],
) -> tuple[SequentialCompilation, SequentialCompilation]:
    """Return the 1→2 and 2→1 sampling aliases of one fitted joint law."""
    rows = tuple(actions)
    ids1, ids2 = tuple(allowed1), tuple(allowed2)
    return (compile_joint_actions(rows, ids1, ids2, order="12"),
            compile_joint_actions(rows, ids1, ids2, order="21"))


def verify_fitted_rows(
    rows: Iterable[tuple[
        Iterable[tuple[int, int, Fraction | float]], Iterable[int], Iterable[int]
    ]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, int | float | tuple[str, str]]:
    """Stream every fitted input row through both aliases and report residuals.

    Each item is ``(public_pair_actions, allowed_ids_1, allowed_ids_2)``.
    The caller supplies *all* admitted fitted rows; this function never
    selects a convenient subset. Fractions are checked exactly. Float
    probability and replay discrepancies are measured and bounded.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite nonnegative replay tolerance required")
    count = exact = zero_conditionals = 0
    max_pair_residual = max_mass_residual = 0.0
    for row_index, (actions, allowed1, allowed2) in enumerate(rows):
        try:
            action_list = tuple(actions)
            ids1, ids2 = tuple(allowed1), tuple(allowed2)
            if len(action_list) > 16 or len(ids1) > 4 or len(ids2) > 4:
                raise ValueError("row exceeds the fixed 4-by-4 action capacity")
            compiled = compile_both_orders(action_list, ids1, ids2)
            rational = all(isinstance(item[2], Fraction) for item in action_list)
            zero = Fraction(0) if rational else 0.0
            expected: dict[tuple[int, int], Fraction | float] = {}
            for id1, id2, mass in action_list:
                key = (int(id1), int(id2))
                expected[key] = expected.get(key, zero) + mass
            expected = {key: mass for key, mass in expected.items() if mass}
            if rational:
                exact += 1
            mass_residual = abs(float(sum(expected.values(), zero)) - 1.0)
            max_mass_residual = max(max_mass_residual, mass_residual)
            for factorization in compiled:
                zero_conditionals += sum(mass == 0 for mass in factorization.marginal)
                replay = factorization.replay()
                keys = set(expected) | set(replay)
                residual = max((abs(float(expected.get(key, zero) - replay.get(key, zero)))
                                for key in keys), default=0.0)
                max_pair_residual = max(max_pair_residual, residual)
                if (rational and expected != replay) or residual > tolerance:
                    raise ValueError("sequential compiler changed the public pair law")
            count += 1
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {row_index}: {exc}") from exc
    if count == 0:
        raise ValueError("at least one fitted law row is required")
    return {
        "rows": count,
        "orders": ("12", "21"),
        "exact_rational_rows": exact,
        "zero_marginal_conditionals": zero_conditionals,
        "max_row_sum_residual": max_mass_residual,
        "max_pair_replay_residual": max_pair_residual,
        "tolerance": tolerance,
    }
