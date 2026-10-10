"""Fixed same-bank identity and decision-only diagnostic code selection.

These deterministic laws use only the frozen public bank, allowed action map,
and one person's permitted fine cell. They do not read scores, task labels,
SEX, or any private release coin.
"""

from __future__ import annotations

import numpy as np

from ..impl.bank import FixedBank, class_medoid


ARMS = frozenset({"A_IDENTITY", "A_CLASS_ONLY"})


def select_codes(bank: FixedBank, actions: tuple[tuple[int, ...], ...],
                 cells: np.ndarray, *, arm_id: str) -> np.ndarray:
    """Return canonical public code IDs in admitted role order.

    CLASS_ONLY transmits the fixed unweighted within-class medoid in the same
    bank. Its emitted probability is therefore a function of the decision.
    """
    if (type(bank) is not FixedBank or type(actions) is not tuple or
            arm_id not in ARMS):
        raise ValueError("fixed diagnostic bank/action/arm identity differs")
    cells = np.asarray(cells)
    if (cells.dtype != np.int64 or cells.ndim != 1 or len(cells) < 1 or
            np.any((cells < 0) | (cells >= len(actions)))):
        raise ValueError("fixed diagnostic fine cells differ")
    for row in actions:
        if (type(row) is not tuple or not 1 <= len(row) <= 4 or
                any(type(code) is not int or not 0 <= code < len(bank.probability)
                    for code in row) or len(set(row)) != len(row) or
                any(bank.decision[code] != bank.decision[row[0]] for code in row)):
            raise ValueError("fixed diagnostic action map differs")
    identity = np.asarray([actions[int(cell)][0] for cell in cells], dtype=np.int64)
    if arm_id == "A_IDENTITY":
        return identity
    representative = {int(label): class_medoid(bank, int(label))
                      for label in np.unique(bank.decision)}
    selected = np.asarray([representative[int(bank.decision[code])]
                           for code in identity], dtype=np.int64)
    if any(int(code) not in actions[int(cell)]
           for cell, code in zip(cells, selected)):
        raise ValueError("class representative is absent from sealed action map")
    return selected


def assert_group_coherent(unit: np.ndarray, code1: np.ndarray,
                          code2: np.ndarray) -> None:
    """A repeated private unit must receive the same deterministic pair."""
    groups, first, second = (np.asarray(value) for value in (unit, code1, code2))
    if (any(value.dtype != np.int64 or value.ndim != 1 for value in
            (groups, first, second)) or not len(groups) or
            first.shape != groups.shape or second.shape != groups.shape):
        raise ValueError("diagnostic group/code alignment differs")
    observed: dict[int, tuple[int, int]] = {}
    for group, c1, c2 in zip(groups, first, second):
        pair = (int(c1), int(c2))
        earlier = observed.setdefault(int(group), pair)
        if earlier != pair:
            raise ValueError("duplicate unit received different diagnostic codes")
