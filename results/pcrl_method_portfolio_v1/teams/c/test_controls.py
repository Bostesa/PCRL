"""Synthetic, label-free checks of Team C matched-control planning."""

from controls import build_control_plan
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
