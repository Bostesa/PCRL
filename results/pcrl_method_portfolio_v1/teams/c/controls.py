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
from typing import Mapping

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
SIMPLE_ARM_IDS = (
    "C-JOINT-GROUP", "C-INDEP-GROUP",
    "C-SIMPLE-FULL", "C-SIMPLE-REL0", "C-SIMPLE-TASK-SHARED1",
    "C-SIMPLE-INDEP1", "C-SIMPLE-COMMON1", "C-SIMPLE-SEQ-IO1",
    "C-SIMPLE-SEQ-OI1",
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


def select_task_shared1(
    atoms: tuple[str, ...],
    task_deltas: Mapping[str, Mapping[str, TaskDelta]],
) -> tuple[str, str, float]:
    """Task-only one-atom removal; SEX scores are absent from this API."""
    atoms = tuple(sorted(atoms))
    if (set(task_deltas) != {"income", "occupation_group"}
            or not atoms or len(atoms) != len(set(atoms))
            or any(set(task_deltas[task]) != set(atoms)
                   for task in ("income", "occupation_group"))):
        raise ValueError("task-only card requires both tasks and exact atom keys")

    def cost(task: str, atom: str) -> float:
        d = task_deltas[task][atom]
        return d.nll / 0.010 + d.brier / 0.005 - d.accuracy / 0.010

    def eligible(task: str, atom: str) -> bool:
        d = task_deltas[task][atom]
        return d.nll <= 0.010 and d.brier <= 0.005 and d.accuracy >= -0.010

    chosen = min(atoms, key=lambda atom: (
        cost("income", atom) + cost("occupation_group", atom), atom,
    ))
    total_cost = cost("income", chosen) + cost("occupation_group", chosen)
    if not isfinite(total_cost):
        raise ValueError("nonfinite task-only proxy cost")
    status = ("TASK_ONLY_SURROGATE_FEASIBLE" if all(
        eligible(task, chosen) for task in ("income", "occupation_group")
    ) else "TASK_ONLY_SURROGATE_CAP_SHORTFALL")
    return chosen, status, -total_cost


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


def build_simple_plan(
    full_card: Scorecard,
    *,
    task_only_deltas: Mapping[str, Mapping[str, TaskDelta]] | None = None,
    schema: AdultGroupedSchema | None = None,
) -> ControlPlan:
    """Seven fixed grouped-mask baselines after the joint XOR premise failed.

    Privacy-aware arms use the nine-atom FULL scorecard. The task-only arm
    requires a separate task-label-only scorecard fitted on its own split;
    without that card its slot is deferred to the separate task-only worker.
    Every active selector removes exactly one semantic atom per recipient.
    Different encoded widths remain explicit capacity mismatches; neither the
    scorecard nor these baselines support a joint-design attribution.
    """
    if set(full_card.optional_atoms) != set(ATOM_ORDER):
        raise ValueError("C-SIMPLE requires exactly the full nine-atom scorecard")
    schema = schema or AdultGroupedSchema()
    target = _budget(schema, REL0_ATOMS, REL0_ATOMS)
    plans: list[ArmPlan] = []
    atoms = tuple(sorted(full_card.optional_atoms))

    def selected(removed: str | None) -> tuple[str, ...]:
        return _selected(schema, ATOM_ORDER, () if removed is None else (removed,))

    def add(
        arm_id: str, removed_i: str | None, removed_o: str | None,
        *, basis: str, status: str, note: str = "", objective: float = 0.0,
        arithmetic: int = 0,
    ) -> None:
        if status.startswith(("PREFIT", "SKIPPED", "DEFERRED")):
            income = occupation = ()
            selection = None
        else:
            income, occupation = selected(removed_i), selected(removed_o)
            selection = None if removed_i is None and removed_o is None else Selection(
                () if removed_i is None else (removed_i,),
                () if removed_o is None else (removed_o,),
                objective, arithmetic, arithmetic, arithmetic,
            )
        budget = _budget(schema, income, occupation)
        prior = (None if status.startswith(("PREFIT", "SKIPPED", "DEFERRED")) else
                 next((a.arm_id for a in plans
                       if not a.capacity_status.startswith(("PREFIT", "SKIPPED", "DEFERRED"))
                       and a.mask_pair == (income, occupation)), None))
        plans.append(ArmPlan(
            arm_id, income, occupation, selection, budget,
            _matching(budget, target), basis, status, None, prior, note,
        ))

    add("C-JOINT-GROUP", None, None, basis="FAILED_PHASE_I_LAW",
        status="PREFIT_PREMISE_FAILED",
        note="Queued XOR scorecard proxy AUC 0.6123299693, below the predeclared 0.90 check")
    add("C-INDEP-GROUP", None, None, basis="FAILED_PROTOTYPE_ABLATION",
        status="SKIPPED_DEPENDS_FAILED_PROTOTYPE")
    add("C-SIMPLE-FULL", None, None, basis="FIXED_FULL",
        status="EXTERNAL_CAPACITY_REFERENCE")
    add("C-SIMPLE-REL0", "relationship", "relationship", basis="FIXED_REL0",
        status="ATTRIBUTION_REFERENCE", arithmetic=0)

    def task_cost(task: str, atom: str) -> float:
        return _cost(full_card.task_deltas[task], (atom,))

    def both_cost(atom: str) -> float:
        return task_cost("income", atom) + task_cost("occupation_group", atom)

    if task_only_deltas is None:
        add("C-SIMPLE-TASK-SHARED1", None, None,
            basis="SEPARATE_TASK_ONLY_WORKER",
            status="DEFERRED_TO_TASK_ONLY_WORKER",
            note="No privacy-worker SEX access is transferred to task-only selection")
    else:
        task_atom, task_status, task_objective = select_task_shared1(
            atoms, task_only_deltas,
        )
        add("C-SIMPLE-TASK-SHARED1", task_atom, task_atom,
            basis="FULL_TASK_SHARED_ONE", status=task_status,
            objective=task_objective, arithmetic=len(atoms))

    def local_choice(task: str) -> tuple[str, str]:
        feasible = tuple(atom for atom in atoms
                         if _local_eligible(full_card, task, (atom,)))
        if feasible:
            chosen = min(feasible, key=lambda atom: (
                -full_card.auc_drop[atom], task_cost(task, atom), atom,
            ))
            return chosen, "SURROGATE_FEASIBLE"
        chosen = min(atoms, key=lambda atom: (
            task_cost(task, atom), -full_card.auc_drop[atom], atom,
        ))
        return chosen, "SURROGATE_CAP_SHORTFALL"

    local_i, status_i = local_choice("income")
    local_o, status_o = local_choice("occupation_group")
    independent_status = ";".join((status_i, status_o))
    if local_i != local_o:
        independent_status += ";UNION_CAPACITY_MISMATCH"
    add("C-SIMPLE-INDEP1", local_i, local_o, basis="FULL_LOCAL_ONE",
        status=independent_status,
        objective=full_card.auc_drop[local_i] + full_card.auc_drop[local_o],
        arithmetic=2 * len(atoms))

    both_feasible = tuple(atom for atom in atoms if all(
        _local_eligible(full_card, task, (atom,))
        for task in ("income", "occupation_group")
    ))
    if both_feasible:
        common_atom = min(both_feasible, key=lambda atom: (
            -full_card.auc_drop[atom], both_cost(atom), atom,
        ))
        common_status = "SURROGATE_FEASIBLE"
    else:
        common_atom = min(atoms, key=lambda atom: (
            both_cost(atom), -full_card.auc_drop[atom], atom,
        ))
        common_status = "SURROGATE_CAP_SHORTFALL"
    add("C-SIMPLE-COMMON1", common_atom, common_atom,
        basis="FULL_COMMON_ONE", status=common_status,
        objective=full_card.auc_drop[common_atom], arithmetic=len(atoms))

    for first, arm_id in (
        ("income", "C-SIMPLE-SEQ-IO1"),
        ("occupation_group", "C-SIMPLE-SEQ-OI1"),
    ):
        second = "occupation_group" if first == "income" else "income"
        first_atom, first_status = local_choice(first)
        if _local_eligible(full_card, second, (first_atom,)):
            second_atom, second_status = first_atom, "SECOND_MATCHED_UNION"
        else:
            second_atom, second_status = local_choice(second)
        removed_i, removed_o = (
            (first_atom, second_atom) if first == "income"
            else (second_atom, first_atom)
        )
        status = ";".join((first_status, second_status))
        if first_atom != second_atom:
            status += ";UNION_CAPACITY_MISMATCH_AFTER_FIRST"
        add(arm_id, removed_i, removed_o,
            basis="FULL_SEQUENTIAL_ONE", status=status,
            objective=full_card.auc_drop[removed_i] + full_card.auc_drop[removed_o],
            arithmetic=2 * len(atoms))
    if tuple(arm.arm_id for arm in plans) != SIMPLE_ARM_IDS:
        raise AssertionError("C-SIMPLE arm registry changed")
    return ControlPlan(tuple(plans), target,
                       scorecard_fit_units=40 if task_only_deltas is None else 60,
                       selector_operations=7 if task_only_deltas is None else 8)


def build_task_only_plan(
    task_deltas: Mapping[str, Mapping[str, TaskDelta]],
    *, schema: AdultGroupedSchema | None = None,
) -> ControlPlan:
    """One-arm plan that can be constructed without a SEX-aware scorecard."""
    schema = schema or AdultGroupedSchema()
    atoms = tuple(sorted(ATOM_ORDER))
    removed, status, objective = select_task_shared1(atoms, task_deltas)
    selected = _selected(schema, ATOM_ORDER, (removed,))
    budget = _budget(schema, selected, selected)
    target = _budget(schema, REL0_ATOMS, REL0_ATOMS)
    arm = ArmPlan(
        "C-SIMPLE-TASK-SHARED1", selected, selected,
        Selection((removed,), (removed,), objective, len(atoms), len(atoms), len(atoms)),
        budget, _matching(budget, target), "TASK_ONLY_SEPARATE_WORKER", status,
        None, None, "Task-label-only split and proxy fits; no SEX file request",
    )
    return ControlPlan((arm,), target, scorecard_fit_units=20, selector_operations=1)
