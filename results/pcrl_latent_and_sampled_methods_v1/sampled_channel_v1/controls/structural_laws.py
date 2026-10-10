"""Pure final-law composition for matched sampled-channel controls.

These arithmetic functions accept only globally canonical public code IDs.
Their inputs are fitted per-person factors; they do not train those factors.
"""

from __future__ import annotations

from fractions import Fraction
import math
from numbers import Integral, Real


def _factor(rows):
    values = tuple(rows)
    if not values:
        raise ValueError("public factor cannot be empty")
    rational = all(isinstance(row[1], Fraction) for row in values)
    zero = Fraction(0) if rational else 0.0
    grouped = {}
    for code, mass in values:
        if (isinstance(code, bool) or not isinstance(code, Integral) or code < 0
                or isinstance(mass, bool) or not isinstance(mass, Real)
                or not math.isfinite(float(mass)) or mass < 0):
            raise ValueError("invalid public code ID or factor mass")
        code = int(code)
        grouped[code] = grouped.get(code, zero) + mass
    total = sum(grouped.values(), zero)
    if (total != 1 if rational else abs(float(total) - 1.0) > 1e-12):
        raise ValueError("public factor must sum to one")
    return {code: mass for code, mass in grouped.items() if mass}


def product_public_law(first, second):
    """Independent conditional draws given the SAME central or local input."""
    q1, q2 = _factor(first), _factor(second)
    return tuple((i, j, q1[i] * q2[j]) for i in sorted(q1) for j in sorted(q2))


def compose_frozen_first(first, second_by_first, *, order):
    """Combine a frozen first law with a code-conditional second law.

    During deployment, second receives the first code actually emitted.
    This function enumerates the resulting joint law for expected-cost and
    privacy arithmetic; it does not choose a favorable first draw for fit.
    """
    if order not in ("12", "21"):
        raise ValueError("order must be '12' or '21'")
    first_rows = tuple(first)
    q_first = _factor(first_rows)
    first_ids = {int(code) for code, _ in first_rows}
    if set(second_by_first) != first_ids:
        raise ValueError("every first public code needs one conditional law")
    pair_mass = {}
    for first_code in sorted(first_ids):
        q_second = _factor(second_by_first[first_code])
        for second_code, conditional_mass in q_second.items():
            pair = ((first_code, second_code) if order == "12"
                    else (second_code, first_code))
            pair_mass[pair] = pair_mass.get(pair, 0) + q_first.get(first_code, 0) * conditional_mass
    return tuple((i, j, mass) for (i, j), mass in sorted(pair_mass.items()) if mass)
