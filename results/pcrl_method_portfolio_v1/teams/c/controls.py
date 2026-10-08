"""Pure Team C control planner; no data access, model fit or outcome scoring.

The candidate's removal set is relative to the eight atoms left after the
predeclared relationship exclusion. An unrestricted nine-atom control must
remove one additional atom to match the recipient/union semantic capacities.
All choices use fitting-role scorecards only. INNER_SELECTION is never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite

from prototype import ATOM_ORDER, AdultGroupedSchema
from selector import (
    Scorecard, Selection, TaskDelta, select_common_mask, select_joint,
    select_sequential, select_task_only,
)


REL0_ATOMS = tuple(a for a in ATOM_ORDER if a != "relationship")
ARM_IDS = (
    "C-JOINT-GROUP", "C-INDEP-GROUP", "C-FULL-FEATURE",
    "C-DROP-RELATIONSHIP", "C-TASK-ONLY-REL0", "C-TASK-ONLY-ALL",
    "C-INDEP-ALL", "C-SEQ-INCOME-FIRST", "C-SEQ-OCCUPATION-FIRST",
    "C-COMMON-MASK", "C-PRIOR-COALITION-SELECTOR",
)


@dataclass(frozen=True)
class Budget:
    semantic_counts: tuple[int, int, int]
    encoded_counts: tuple[int, int, int]


@dataclass(frozen=True)
class BudgetMatch:
    semantic_local: bool
    semantic_union: bool
    encoded_local: bool
    encoded_union: bool

    @property
    def semantic_all(self) -> bool:
        return self.semantic_local and self.semantic_union

    @property
    def encoded_all(self) -> bool:
        return self.encoded_local and self.encoded_union

    @property
    def exact(self) -> bool:
        return self.semantic_all and self.encoded_all


@dataclass(frozen=True)
class ArmPlan:
    arm_id: str
    income_atoms: tuple[str, ...]
    occupation_atoms: tuple[str, ...]
    selection: Selection | None
    budget: Budget
    budget_match: BudgetMatch
    selector_basis: str
    capacity_status: str
    alias_of: str | None = None
    mask_equivalent_to: str | None = None
    note: str = ""

    @property
    def mask_pair(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.income_atoms, self.occupation_atoms


@dataclass(frozen=True)
class ControlPlan:
    arms: tuple[ArmPlan, ...]
    candidate_budget: Budget
    scorecard_fit_units: int
    selector_operations: int

    def by_id(self, arm_id: str) -> ArmPlan:
        matches = [arm for arm in self.arms if arm.arm_id == arm_id]
        if len(matches) != 1:
            raise KeyError(arm_id)
        return matches[0]


def _validate_cards(rel0: Scorecard, full: Scorecard) -> None:
    if set(rel0.optional_atoms) != set(REL0_ATOMS):
        raise ValueError("relationship-excluded scorecard must cover exactly eight atoms")
    if set(full.optional_atoms) != set(ATOM_ORDER):
        raise ValueError("full scorecard must cover exactly nine atoms")
    if "relationship" in rel0.optional_atoms:
        raise ValueError("relationship leaked into restricted scorecard")


def _budget(
    schema: AdultGroupedSchema, income: tuple[str, ...], occupation: tuple[str, ...]
) -> Budget:
    union = schema.normalize_atoms(tuple(set(income) | set(occupation)))
    return Budget(
        (len(income), len(occupation), len(union)),
        (
            len(schema.column_indices(income)),
            len(schema.column_indices(occupation)),
            len(schema.column_indices(union)),
        ),
    )


def _matching(actual: Budget, target: Budget) -> BudgetMatch:
    return BudgetMatch(
        actual.semantic_counts[:2] == target.semantic_counts[:2],
        actual.semantic_counts[2] == target.semantic_counts[2],
        actual.encoded_counts[:2] == target.encoded_counts[:2],
        actual.encoded_counts[2] == target.encoded_counts[2],
    )


def _selected(schema: AdultGroupedSchema, base: tuple[str, ...], removed: tuple[str, ...]) -> tuple[str, ...]:
    if not set(removed) <= set(base):
        raise ValueError("selector removed an atom absent from its base")
    return schema.normalize_atoms(tuple(set(base) - set(removed)))


def _local_eligible(card: Scorecard, task: str, removed: tuple[str, ...]) -> bool:
    deltas = card.task_deltas[task]
    return (
        sum(deltas[a].nll for a in removed) <= 0.010
        and sum(deltas[a].brier for a in removed) <= 0.005
        and sum(deltas[a].accuracy for a in removed) >= -0.010
    )


def _cost(deltas: dict[str, TaskDelta], removed: tuple[str, ...]) -> float:
    return sum(
        deltas[a].nll / 0.010 + deltas[a].brier / 0.005
        - deltas[a].accuracy / 0.010 for a in removed
    )


def _independent_at_counts(
    card: Scorecard, income_count: int, occupation_count: int
) -> tuple[Selection, str]:
    """Select each purpose independently at fixed recipient capacity.

    If no mask at the required size meets a surrogate utility cap, choose the
    least-cost mask using task labels on the fitting role only and mark the
    proxy shortfall. Actual utility eligibility is decided by S2 later.
    """
    picked = []
    counts = []
    notes = []
    for task, n in (("income", income_count), ("occupation_group", occupation_count)):
        if n < 0 or n > len(card.optional_atoms):
            raise ValueError("requested independent mask capacity is impossible")
        choices = tuple(combinations(card.optional_atoms, n))
        feasible = tuple(r for r in choices if _local_eligible(card, task, r))
        if feasible:
            chosen = min(
                feasible,
                key=lambda r: (-sum(card.auc_drop[a] for a in r),
                               _cost(card.task_deltas[task], r), r),
            )
            notes.append("SURROGATE_FEASIBLE")
        else:
            chosen = min(
                choices,
                key=lambda r: (_cost(card.task_deltas[task], r),
                               -sum(card.auc_drop[a] for a in r), r),
            )
            notes.append("NO_FEASIBLE_MASK_AT_MATCHED_LOCAL_CAPACITY")
        picked.append(chosen)
        counts.append(len(choices))
    income, occupation = picked
    objective = sum(card.auc_drop[a] for a in income + occupation)
    if not isfinite(objective):
        raise ValueError("nonfinite independent score")
    return Selection(
        income, occupation, objective, 0, counts[0] * counts[1],
        counts[0] + counts[1],
    ), ";".join(notes)


def _sequential_at_capacity(
    card: Scorecard, *, first: str, income_count: int,
    occupation_count: int, common_removed_count: int,
) -> tuple[Selection | None, str]:
    kwargs = dict(
        first=first, max_removals=2,
        removal_counts=(income_count, occupation_count),
    )
    try:
        selected = select_sequential(
            card, union_removed_count=common_removed_count, **kwargs,
        )
        return selected, "MATCHED_AFTER_LOCAL_FIRST"
    except ValueError as exc:
        # Preserve the committed local-first stage. This fallback changes only
        # the second-stage capacity constraint, never the first-stage choice.
        try:
            unmatched = select_sequential(card, union_removed_count=None, **kwargs)
        except ValueError:
            # This cannot arise from a valid joint candidate on the same
            # scorecard, because its own masks witness local feasibility at
            # the requested sizes. Keep a deterministic control if a future
            # registered capacity rule nevertheless reaches this case.
            counts = {"income": income_count, "occupation_group": occupation_count}
            second = "occupation_group" if first == "income" else "income"
            first_options = tuple(combinations(card.optional_atoms, counts[first]))
            second_options = tuple(combinations(card.optional_atoms, counts[second]))
            if not first_options or not second_options:
                return None, "NOT_RUN_IMPOSSIBLE_SEMANTIC_CAPACITY"
            first_removed = min(
                first_options,
                key=lambda r: (_cost(card.task_deltas[first], r),
                               -sum(card.auc_drop[a] for a in r), r),
            )
            second_feasible = tuple(
                r for r in second_options if _local_eligible(card, second, r)
            )
            second_pool = second_feasible or second_options
            matched = tuple(
                r for r in second_pool
                if len(set(first_removed) & set(r)) == common_removed_count
            )
            pool = matched or second_pool

            def pair(r):
                return (first_removed, r) if first == "income" else (r, first_removed)

            def score(r):
                income, occupation = pair(r)
                intersection = set(income) & set(occupation)
                return (
                    sum(card.auc_drop[a] for a in intersection)
                    + 0.25 * sum(card.auc_drop[a] for a in income + occupation)
                )

            second_removed = min(
                pool,
                key=lambda r: (-score(r), _cost(card.task_deltas[second], r), r),
            )
            income, occupation = pair(second_removed)
            status = "SURROGATE_FIRST_STAGE_SHORTFALL"
            if not matched:
                status += ";UNATTAINABLE_AFTER_FIRST"
            if not second_feasible:
                status += ";SURROGATE_SECOND_STAGE_SHORTFALL"
            return Selection(
                income, occupation, score(second_removed), len(second_options),
                len(pool), len(first_options),
            ), status
        if len(unmatched.union_removed) == common_removed_count:
            raise AssertionError("matched sequential result unexpectedly failed") from exc
        return unmatched, "UNATTAINABLE_AFTER_FIRST"


def build_control_plan(
    rel0_card: Scorecard,
    full_card: Scorecard,
    *,
    schema: AdultGroupedSchema | None = None,
) -> ControlPlan:
    """Build all predeclared Team C arms from immutable fitting-role cards.

    A compatible published coalition-aware selector requires a separately
    registered implementation. The prior-selector slot is explicitly skipped
    here rather than pretending that a common/task-only mask reproduces it.
    """
    _validate_cards(rel0_card, full_card)
    schema = schema or AdultGroupedSchema()
    joint = select_joint(rel0_card)
    r_income = len(joint.income_removed)
    r_occupation = len(joint.occupation_removed)
    r_common = len(joint.union_removed)
    candidate_income = _selected(schema, REL0_ATOMS, joint.income_removed)
    candidate_occupation = _selected(schema, REL0_ATOMS, joint.occupation_removed)
    target = _budget(schema, candidate_income, candidate_occupation)
    plans: list[ArmPlan] = []

    def skip(arm_id: str, status: str, note: str) -> None:
        empty = Budget((0, 0, 0), (0, 0, 0))
        plans.append(ArmPlan(
            arm_id, (), (), None, empty, _matching(empty, target),
            "NOT_APPLICABLE", status, note=note,
        ))

    def add(
        arm_id: str, base: tuple[str, ...], selection: Selection | None,
        *, income_removed: tuple[str, ...] = (),
        occupation_removed: tuple[str, ...] = (), basis: str,
        status: str = "PLANNED", alias_of: str | None = None,
        note: str = "",
    ) -> ArmPlan:
        if arm_id in {p.arm_id for p in plans}:
            raise ValueError(f"duplicate arm {arm_id}")
        if selection is not None:
            income_removed = selection.income_removed
            occupation_removed = selection.occupation_removed
        income = _selected(schema, base, income_removed)
        occupation = _selected(schema, base, occupation_removed)
        budget = _budget(schema, income, occupation)
        match = _matching(budget, target)
        prior = next((p.arm_id for p in plans if p.mask_pair == (income, occupation)), None)
        arm = ArmPlan(
            arm_id, income, occupation, selection, budget, match, basis,
            status, alias_of, prior, note,
        )
        plans.append(arm)
        return arm

    add("C-JOINT-GROUP", REL0_ATOMS, joint, basis="REL0_PAIR")
    indep_rel, indep_rel_note = _independent_at_counts(
        rel0_card, r_income, r_occupation,
    )
    add("C-INDEP-GROUP", REL0_ATOMS, indep_rel, basis="REL0_LOCAL",
        note=indep_rel_note)
    add("C-FULL-FEATURE", ATOM_ORDER, None, basis="FIXED_FULL",
        status="EXTERNAL_CAPACITY_REFERENCE")
    add("C-DROP-RELATIONSHIP", REL0_ATOMS, None, basis="FIXED_REL0",
        status="ATTRIBUTION_REFERENCE")
    task_rel = select_task_only(
        rel0_card, income_removals=r_income,
        occupation_removals=r_occupation, union_removed_count=r_common,
    )
    add("C-TASK-ONLY-REL0", REL0_ATOMS, task_rel, basis="REL0_TASK")
    task_full = select_task_only(
        full_card, income_removals=1 + r_income,
        occupation_removals=1 + r_occupation,
        union_removed_count=1 + r_common,
    )
    add("C-TASK-ONLY-ALL", ATOM_ORDER, task_full, basis="FULL_TASK")
    indep_full, indep_full_note = _independent_at_counts(
        full_card, 1 + r_income, 1 + r_occupation,
    )
    add("C-INDEP-ALL", ATOM_ORDER, indep_full, basis="FULL_LOCAL",
        note=indep_full_note)
    for first, arm_id in (
        ("income", "C-SEQ-INCOME-FIRST"),
        ("occupation_group", "C-SEQ-OCCUPATION-FIRST"),
    ):
        selected, status = _sequential_at_capacity(
            rel0_card, first=first, income_count=r_income,
            occupation_count=r_occupation, common_removed_count=r_common,
        )
        if selected is None:
            skip(arm_id, status, "No utility-feasible first-stage mask at candidate capacity")
        else:
            add(arm_id, REL0_ATOMS, selected, basis="REL0_SEQUENTIAL",
                status=status)
    try:
        shared = select_common_mask(
            rel0_card, removal_count=r_common,
        )
        shared_status = "UNION_CAPACITY_MATCHED_LOCAL_MISMATCH_POSSIBLE"
    except ValueError:
        shared = select_common_mask(rel0_card)
        shared_status = "UNION_CAPACITY_UNATTAINABLE_FOR_SHARED_MASK"
    add("C-COMMON-MASK", REL0_ATOMS, shared, basis="REL0_SHARED",
        status=shared_status)
    skip(
        "C-PRIOR-COALITION-SELECTOR",
        "SKIPPED_NO_COMPATIBLE_PRIOR_IMPLEMENTATION",
        "No published selector with matched raw-value union recovery, task caps, "
        "and two-purpose deployment was identified before the seal; standard "
        "task-only and shared-mask controls are reported under their own IDs.",
    )
    if tuple(p.arm_id for p in plans) != ARM_IDS:
        raise AssertionError("control arm registry changed")
    return ControlPlan(tuple(plans), target, scorecard_fit_units=72,
                       selector_operations=14)
