"""Prospective finite bank-output comparison family; no data imports."""

from __future__ import annotations

from itertools import combinations


ARMS = ("B0", "T", "D", "P", "L", "J", "S12", "S21", "W_Q", "P_STAR")
TASKS = ("income", "occupation")
REFERENCES = ("U0", "Ucal")
METRICS = ("accuracy", "nll", "brier")
VIEWS = ("r1", "r2", "pair")
UTILITY_SUMMARIES = ("s0", "s1", "mean")
RECOVERY_SUMMARIES = ("s0_d0", "s0_d1", "s0_d2",
                      "s1_d0", "s1_d1", "s1_d2",
                      "s0_mean", "s1_mean", "mean")


def ordered_clause_ids() -> tuple[str, ...]:
    utility = tuple(
        f"{summary}:utility:{arm}:{task}:{reference}:{metric}"
        for summary in UTILITY_SUMMARIES
        for arm in ARMS for task in TASKS for reference in REFERENCES
        for metric in METRICS
    )
    recovery = tuple(
        f"{summary}:recovery:{first}:{second}:{view}:auc"
        for summary in RECOVERY_SUMMARIES
        for first, second in combinations(ARMS, 2) for view in VIEWS
    )
    names = utility + recovery
    if len(utility) != 360 or len(recovery) != 1215 or len(set(names)) != 1575:
        raise AssertionError("bank-output prospective family changed")
    return names
