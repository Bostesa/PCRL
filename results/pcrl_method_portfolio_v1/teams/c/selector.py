"""Deterministic Team C selection from a precomputed, fitting-role scorecard.

This module performs no data access or model fit. The scorecard values are
surrogates computed elsewhere from allowed defense-fit rows. Final utility and
privacy claims require retrained heads and fresh complete-view attackers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Mapping


TASKS = ("income", "occupation_group")


@dataclass(frozen=True)
class TaskDelta:
    """Removal minus base for loss/Brier, removal minus base for accuracy."""

    nll: float
    brier: float
    accuracy: float


@dataclass(frozen=True)
class Scorecard:
    optional_atoms: tuple[str, ...]
    auc_drop: Mapping[str, float]
    task_deltas: Mapping[str, Mapping[str, TaskDelta]]

    def __post_init__(self) -> None:
        atoms = tuple(sorted(self.optional_atoms))
        if not atoms or len(atoms) != len(set(atoms)):
            raise ValueError("scorecard atoms must be unique and nonempty")
        if set(self.auc_drop) != set(atoms):
            raise ValueError("SEX AUC scorecard must cover every optional atom")
        if set(self.task_deltas) != set(TASKS):
            raise ValueError("task scorecard must cover income and occupation_group")
        for task in TASKS:
            if set(self.task_deltas[task]) != set(atoms):
                raise ValueError(f"{task} scorecard must cover every optional atom")
            for delta in self.task_deltas[task].values():
                if not all(isfinite(x) for x in (delta.nll, delta.brier, delta.accuracy)):
                    raise ValueError("task deltas must be finite")
        if not all(isfinite(x) for x in self.auc_drop.values()):
            raise ValueError("SEX AUC deltas must be finite")
        object.__setattr__(self, "optional_atoms", atoms)


@dataclass(frozen=True)
class Selection:
    """Mask choice plus arithmetic search receipts; model-fit count is zero here.

    `evaluated_pairs` counts pair eligibility/objective checks. Independent
    selection evaluates local masks only, so its pair count is zero.
    `feasible_pairs` counts eligible pairs in each method's defined search:
    the cartesian product for independent local choices, the compatible
    second choices for sequential search, or eligible diagonal pairs for a
    common mask. Task-only eligibility means its requested capacities.
    """

    income_removed: tuple[str, ...]
    occupation_removed: tuple[str, ...]
    objective: float
    evaluated_pairs: int
    feasible_pairs: int
    evaluated_single_masks: int = 0

    @property
    def union_removed(self) -> tuple[str, ...]:
        """Atoms omitted from both releases, hence absent from raw union."""
        return tuple(sorted(set(self.income_removed) & set(self.occupation_removed)))


def _removal_sets(atoms: tuple[str, ...], max_removals: int) -> tuple[tuple[str, ...], ...]:
    if max_removals < 0 or max_removals > len(atoms):
        raise ValueError("max_removals outside the scorecard atom count")
    return tuple(
        subset
        for size in range(max_removals + 1)
        for subset in combinations(atoms, size)
    )


def _eligible(
    scorecard: Scorecard,
    task: str,
    removed: tuple[str, ...],
    nll_cap: float,
    brier_cap: float,
    accuracy_floor: float,
) -> bool:
    delta = scorecard.task_deltas[task]
    return (
        sum(delta[a].nll for a in removed) <= nll_cap
        and sum(delta[a].brier for a in removed) <= brier_cap
        and sum(delta[a].accuracy for a in removed) >= accuracy_floor
    )


def _eligible_sets(
    scorecard: Scorecard,
    task: str,
    subsets: tuple[tuple[str, ...], ...],
    nll_cap: float,
    brier_cap: float,
    accuracy_floor: float,
) -> set[tuple[str, ...]]:
    return {
        subset for subset in subsets
        if _eligible(scorecard, task, subset, nll_cap, brier_cap, accuracy_floor)
    }


def select_joint(
    scorecard: Scorecard,
    *,
    max_removals: int = 2,
    nll_cap: float = 0.010,
    brier_cap: float = 0.005,
    accuracy_floor: float = -0.010,
    local_weight: float = 0.25,
) -> Selection:
    """Enumerate fixed removal pairs and maximize union plus local AUC drop."""
    subsets = _removal_sets(scorecard.optional_atoms, max_removals)
    income_ok = _eligible_sets(
        scorecard, "income", subsets, nll_cap, brier_cap, accuracy_floor
    )
    occupation_ok = _eligible_sets(
        scorecard, "occupation_group", subsets, nll_cap, brier_cap,
        accuracy_floor,
    )
    candidates = []
    for income in subsets:
        for occupation in subsets:
            if income not in income_ok or occupation not in occupation_ok:
                continue
            intersection = set(income) & set(occupation)
            score = sum(scorecard.auc_drop[a] for a in intersection)
            score += local_weight * (
                sum(scorecard.auc_drop[a] for a in income)
                + sum(scorecard.auc_drop[a] for a in occupation)
            )
            candidates.append((income, occupation, score))
    if not candidates:
        raise ValueError("no utility-feasible removal pair")
    income, occupation, objective = min(
        candidates, key=lambda x: (-x[2], len(x[0]) + len(x[1]), x[0], x[1])
    )
    return Selection(income, occupation, objective, len(subsets) ** 2, len(candidates))


def select_independent(
    scorecard: Scorecard,
    *,
    max_removals: int = 2,
    nll_cap: float = 0.010,
    brier_cap: float = 0.005,
    accuracy_floor: float = -0.010,
) -> Selection:
    """Choose each local mask alone, then expose and attack their raw union."""
    subsets = _removal_sets(scorecard.optional_atoms, max_removals)
    selections = []
    eligible_count = []
    for task in TASKS:
        eligible = _eligible_sets(
            scorecard, task, subsets, nll_cap, brier_cap, accuracy_floor
        )
        if not eligible:
            raise ValueError(f"no utility-feasible {task} removal")
        chosen = min(
            eligible,
            key=lambda removed: (
                -sum(scorecard.auc_drop[a] for a in removed),
                len(removed), removed,
            ),
        )
        selections.append(chosen)
        eligible_count.append(len(eligible))
    income, occupation = selections
    objective = (
        sum(scorecard.auc_drop[a] for a in income)
        + sum(scorecard.auc_drop[a] for a in occupation)
    )
    return Selection(
        income, occupation, objective, 0,
        eligible_count[0] * eligible_count[1], 2 * len(subsets),
    )


def _joint_score(
    scorecard: Scorecard,
    income: tuple[str, ...],
    occupation: tuple[str, ...],
    local_weight: float,
) -> float:
    return (
        sum(scorecard.auc_drop[a] for a in set(income) & set(occupation))
        + local_weight * (
            sum(scorecard.auc_drop[a] for a in income)
            + sum(scorecard.auc_drop[a] for a in occupation)
        )
    )


def select_sequential(
    scorecard: Scorecard,
    *,
    first: str,
    max_removals: int = 2,
    nll_cap: float = 0.010,
    brier_cap: float = 0.005,
    accuracy_floor: float = -0.010,
    local_weight: float = 0.25,
    removal_counts: tuple[int, int] | None = None,
    union_removed_count: int | None = None,
) -> Selection:
    """Local first-stage choice, then a coalition-aware second-stage choice.

    `removal_counts` and `union_removed_count` permit a predeclared
    candidate-capacity-matched control. These counts are fixed before its
    outcome is scored. Search uses the same immutable scorecard as the joint
    selector and makes no extra model fit.
    """
    if first not in TASKS:
        raise ValueError("first must be income or occupation_group")
    subsets = _removal_sets(scorecard.optional_atoms, max_removals)
    eligible = {
        task: _eligible_sets(
            scorecard, task, subsets, nll_cap, brier_cap, accuracy_floor
        )
        for task in TASKS
    }
    if removal_counts is not None:
        if len(removal_counts) != 2 or any(n < 0 for n in removal_counts):
            raise ValueError("removal_counts must give both nonnegative counts")
        for task, count in zip(TASKS, removal_counts):
            eligible[task] = {r for r in eligible[task] if len(r) == count}
    second = TASKS[1] if first == TASKS[0] else TASKS[0]
    first_options = eligible[first]
    if not first_options:
        raise ValueError(f"no feasible first-stage mask for {first}")
    first_removed = min(
        first_options,
        key=lambda removed: (
            -sum(scorecard.auc_drop[a] for a in removed), len(removed), removed
        ),
    )
    second_options = eligible[second]
    if union_removed_count is not None:
        second_options = {
            r for r in second_options
            if len(set(r) & set(first_removed)) == union_removed_count
        }
    if not second_options:
        raise ValueError(f"no feasible second-stage mask for {second}")

    def pair(removed: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (first_removed, removed) if first == "income" else (removed, first_removed)

    second_removed = min(
        second_options,
        key=lambda removed: (
            -_joint_score(scorecard, *pair(removed), local_weight),
            len(first_removed) + len(removed), *pair(removed),
        ),
    )
    income, occupation = pair(second_removed)
    return Selection(
        income, occupation, _joint_score(scorecard, income, occupation, local_weight),
        len(second_options), len(second_options), len(first_options),
    )


def select_common_mask(
    scorecard: Scorecard,
    *,
    max_removals: int = 2,
    nll_cap: float = 0.010,
    brier_cap: float = 0.005,
    accuracy_floor: float = -0.010,
    local_weight: float = 0.25,
    removal_count: int | None = None,
) -> Selection:
    """One shared mask, with both task caps checked on that mask."""
    subsets = _removal_sets(scorecard.optional_atoms, max_removals)
    eligible = [
        removed for removed in subsets
        if (removal_count is None or len(removed) == removal_count)
        and all(
            _eligible(scorecard, task, removed, nll_cap, brier_cap, accuracy_floor)
            for task in TASKS
        )
    ]
    if not eligible:
        raise ValueError("no common utility-feasible mask")
    removed = min(
        eligible,
        key=lambda r: (-_joint_score(scorecard, r, r, local_weight), 2 * len(r), r),
    )
    return Selection(
        removed, removed, _joint_score(scorecard, removed, removed, local_weight),
        len(subsets), len(eligible),
    )


def select_task_only(
    scorecard: Scorecard,
    *,
    income_removals: int,
    occupation_removals: int,
    union_removed_count: int | None = None,
    nll_scale: float = 0.010,
    brier_scale: float = 0.005,
    accuracy_scale: float = 0.010,
) -> Selection:
    """Choose task-best masks at fixed counts, without using SEX deltas.

    The optional intersection count matches the candidate's raw union
    capacity where attainable. The objective is negative normalized proxy
    utility cost; final calibrated utility is assessed separately.
    """
    if min(income_removals, occupation_removals) < 0:
        raise ValueError("removal counts must be nonnegative")
    if min(nll_scale, brier_scale, accuracy_scale) <= 0:
        raise ValueError("utility scales must be positive")
    income_options = tuple(combinations(scorecard.optional_atoms, income_removals))
    occupation_options = tuple(combinations(scorecard.optional_atoms, occupation_removals))
    pairs = [
        (income, occupation)
        for income in income_options for occupation in occupation_options
        if union_removed_count is None
        or len(set(income) & set(occupation)) == union_removed_count
    ]
    if not pairs:
        raise ValueError("requested mask/union capacity is unattainable")

    def task_cost(task: str, removed: tuple[str, ...]) -> float:
        deltas = scorecard.task_deltas[task]
        return sum(
            deltas[a].nll / nll_scale + deltas[a].brier / brier_scale
            - deltas[a].accuracy / accuracy_scale
            for a in removed
        )

    income, occupation = min(
        pairs,
        key=lambda pair: (
            task_cost("income", pair[0]) + task_cost("occupation_group", pair[1]),
            pair[0], pair[1],
        ),
    )
    cost = task_cost("income", income) + task_cost("occupation_group", occupation)
    return Selection(income, occupation, -cost,
                     len(income_options) * len(occupation_options), len(pairs))
