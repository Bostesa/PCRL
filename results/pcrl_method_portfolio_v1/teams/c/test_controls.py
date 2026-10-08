"""Synthetic, label-free checks of Team C matched-control planning."""

from controls import (
    build_control_plan, build_simple_plan, build_task_only_plan,
    select_task_shared1,
)
from selector import Scorecard, TaskDelta


def card(atoms, drops, income_costs=None, occupation_costs=None):
    zero = TaskDelta(0.0, 0.0, 0.0)
    return Scorecard(
        tuple(atoms), dict(drops),
        {
            "income": {a: (income_costs or {}).get(a, zero) for a in atoms},
            "occupation_group": {a: (occupation_costs or {}).get(a, zero) for a in atoms},
        },
    )


def full_and_rel_cards():
    atoms = (
        "age", "education_fact", "capital-gain", "capital-loss",
        "hours-per-week", "workclass", "marital-status", "relationship",
        "native-country",
    )
    rel0 = tuple(a for a in atoms if a != "relationship")
    drops = {a: 0.0 for a in atoms}
    drops["age"] = 0.5
    drops["capital-gain"] = 0.4
    drops["relationship"] = 0.8
    return card(rel0, {a: drops[a] for a in rel0}), card(atoms, drops)


def test_registers_full_and_all_matched_controls_without_fits():
    rel0, full = full_and_rel_cards()
    result = build_control_plan(rel0, full)
    ids = {arm.arm_id for arm in result.arms}
    assert ids == {
        "C-JOINT-GROUP", "C-INDEP-GROUP", "C-FULL-FEATURE",
        "C-DROP-RELATIONSHIP", "C-TASK-ONLY-REL0", "C-TASK-ONLY-ALL",
        "C-INDEP-ALL", "C-SEQ-INCOME-FIRST", "C-SEQ-OCCUPATION-FIRST",
        "C-COMMON-MASK", "C-PRIOR-COALITION-SELECTOR",
    }
    assert set(result.by_id("C-FULL-FEATURE").income_atoms) == set(full.optional_atoms)
    assert "relationship" not in result.by_id("C-JOINT-GROUP").income_atoms
    assert result.by_id("C-PRIOR-COALITION-SELECTOR").capacity_status == (
        "SKIPPED_NO_COMPATIBLE_PRIOR_IMPLEMENTATION"
    )
    assert result.by_id("C-PRIOR-COALITION-SELECTOR").selection is None


def test_unrestricted_task_only_matches_candidate_atom_and_union_capacity():
    rel0, full = full_and_rel_cards()
    result = build_control_plan(rel0, full)
    joint = result.by_id("C-JOINT-GROUP")
    task = result.by_id("C-TASK-ONLY-ALL")
    assert task.budget.semantic_counts == joint.budget.semantic_counts
    assert task.budget_match.semantic_all
    assert task.selection is not None
    assert "relationship" in full.optional_atoms


def test_independent_controls_match_local_counts_and_report_union_difference():
    rel0, full = full_and_rel_cards()
    result = build_control_plan(rel0, full)
    joint = result.by_id("C-JOINT-GROUP")
    for arm_id in ("C-INDEP-GROUP", "C-INDEP-ALL"):
        arm = result.by_id(arm_id)
        assert arm.budget.semantic_counts[:2] == joint.budget.semantic_counts[:2]
        assert arm.budget_match.semantic_local
        assert arm.budget_match.semantic_union == (
            arm.budget.semantic_counts[2] == joint.budget.semantic_counts[2]
        )


def test_prior_selector_is_not_fabricated_as_an_exact_alias():
    rel0, full = full_and_rel_cards()
    result = build_control_plan(rel0, full)
    prior = result.by_id("C-PRIOR-COALITION-SELECTOR")
    assert prior.alias_of is None
    assert prior.mask_pair == ((), ())
    assert result.selector_operations == 14


def test_local_first_sequential_capacity_failure_is_reported_and_retained():
    atoms = (
        "age", "education_fact", "capital-gain", "capital-loss",
        "hours-per-week", "workclass", "marital-status", "native-country",
    )
    drops = {a: 0.0 for a in atoms}
    drops["age"] = 0.5
    drops["capital-gain"] = 0.9
    rel0 = card(
        atoms, drops,
        income_costs={
            "age": TaskDelta(0.006, 0.0, 0.0),
            "capital-gain": TaskDelta(0.006, 0.0, 0.0),
        },
        occupation_costs={"capital-gain": TaskDelta(0.020, 0.0, 0.0)},
    )
    all_atoms = atoms + ("relationship",)
    full = card(all_atoms, {**drops, "relationship": 0.0})
    result = build_control_plan(rel0, full)
    joint = result.by_id("C-JOINT-GROUP")
    sequential = result.by_id("C-SEQ-INCOME-FIRST")
    assert joint.selection.income_removed == ("age",)
    assert joint.selection.occupation_removed == ("age",)
    assert sequential.capacity_status == "UNATTAINABLE_AFTER_FIRST"
    assert sequential.selection.income_removed == ("capital-gain",)
    assert sequential.selection.occupation_removed == ("age",)
    assert not sequential.budget_match.semantic_union


def test_simple_plan_preserves_failed_prototypes_and_runs_seven_full_card_arms():
    _, full = full_and_rel_cards()
    plan = build_simple_plan(full, task_only_deltas=full.task_deltas)
    assert {a.arm_id for a in plan.arms if not a.capacity_status.startswith((
        "PREFIT", "SKIPPED",
    ))} == {
        "C-SIMPLE-FULL", "C-SIMPLE-REL0", "C-SIMPLE-TASK-SHARED1",
        "C-SIMPLE-INDEP1", "C-SIMPLE-COMMON1", "C-SIMPLE-SEQ-IO1",
        "C-SIMPLE-SEQ-OI1",
    }
    assert plan.by_id("C-JOINT-GROUP").capacity_status == "PREFIT_PREMISE_FAILED"
    assert plan.by_id("C-INDEP-GROUP").capacity_status == (
        "SKIPPED_DEPENDS_FAILED_PROTOTYPE"
    )
    assert plan.scorecard_fit_units == 60
    assert plan.by_id("C-SIMPLE-FULL").budget.semantic_counts == (9, 9, 9)
    assert plan.by_id("C-SIMPLE-REL0").budget.semantic_counts == (8, 8, 8)
    for name in ("C-SIMPLE-TASK-SHARED1", "C-SIMPLE-COMMON1",
                 "C-SIMPLE-SEQ-IO1", "C-SIMPLE-SEQ-OI1"):
        assert plan.by_id(name).budget.semantic_counts[:2] == (8, 8)


def test_simple_independent_and_sequential_keep_native_union_mismatch():
    atoms = (
        "age", "education_fact", "capital-gain", "capital-loss",
        "hours-per-week", "workclass", "marital-status", "relationship",
        "native-country",
    )
    drops = {a: 0.0 for a in atoms}
    drops["age"] = 0.8
    drops["capital-gain"] = 0.7
    full = card(
        atoms, drops,
        income_costs={"capital-gain": TaskDelta(0.020, 0.0, 0.0)},
        occupation_costs={"age": TaskDelta(0.020, 0.0, 0.0)},
    )
    plan = build_simple_plan(full, task_only_deltas=full.task_deltas)
    independent = plan.by_id("C-SIMPLE-INDEP1")
    assert independent.selection.income_removed == ("age",)
    assert independent.selection.occupation_removed == ("capital-gain",)
    assert independent.budget.semantic_counts == (8, 8, 9)
    assert not independent.budget_match.semantic_union
    for name in ("C-SIMPLE-SEQ-IO1", "C-SIMPLE-SEQ-OI1"):
        sequential = plan.by_id(name)
        assert sequential.budget.semantic_counts == (8, 8, 9)
        assert "UNION_CAPACITY_MISMATCH_AFTER_FIRST" in sequential.capacity_status


def test_task_shared_selector_accepts_no_sensitive_score_input():
    _, full = full_and_rel_cards()
    chosen = select_task_shared1(full.optional_atoms, full.task_deltas)
    altered_sex = card(
        full.optional_atoms,
        {atom: 100.0 - full.auc_drop[atom] for atom in full.optional_atoms},
    )
    assert select_task_shared1(altered_sex.optional_atoms,
                               altered_sex.task_deltas) == chosen
    assert build_simple_plan(altered_sex, task_only_deltas=full.task_deltas).by_id(
        "C-SIMPLE-TASK-SHARED1"
    ).mask_pair == build_simple_plan(full, task_only_deltas=full.task_deltas).by_id(
        "C-SIMPLE-TASK-SHARED1"
    ).mask_pair


def test_simple_plan_requires_separate_task_card_and_uses_its_values():
    _, full = full_and_rel_cards()
    privacy = build_simple_plan(full)
    assert privacy.scorecard_fit_units == 40
    assert privacy.selector_operations == 7
    assert privacy.by_id("C-SIMPLE-TASK-SHARED1").capacity_status == (
        "DEFERRED_TO_TASK_ONLY_WORKER"
    )
    detached = {
        task: {atom: TaskDelta(0.010, 0.005, -0.010)
               for atom in full.optional_atoms}
        for task in ("income", "occupation_group")
    }
    for task in detached:
        detached[task]["relationship"] = TaskDelta(-1.0, -1.0, 1.0)
    plan = build_simple_plan(full, task_only_deltas=detached)
    assert plan.by_id("C-SIMPLE-TASK-SHARED1").selection.income_removed == (
        "relationship",
    )
    separate = build_task_only_plan(detached)
    assert separate.scorecard_fit_units == 20
    assert tuple(a.arm_id for a in separate.arms) == ("C-SIMPLE-TASK-SHARED1",)
    assert separate.arms[0].mask_pair == plan.by_id(
        "C-SIMPLE-TASK-SHARED1"
    ).mask_pair
