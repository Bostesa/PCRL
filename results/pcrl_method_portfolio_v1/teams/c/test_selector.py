"""Synthetic tests for Team C's fixed, cached-scorecard selector."""

from selector import (
    Scorecard, TaskDelta, select_common_mask, select_independent,
    select_joint, select_sequential, select_task_only,
)


def card(atoms, auc_drop, income=None, occupation=None):
    zero = TaskDelta(0.0, 0.0, 0.0)
    return Scorecard(
        optional_atoms=tuple(atoms),
        auc_drop=auc_drop,
        task_deltas={
            "income": {a: (income or {}).get(a, zero) for a in atoms},
            "occupation_group": {a: (occupation or {}).get(a, zero) for a in atoms},
        },
    )


def test_joint_overlap_protects_union_when_local_choices_diverge():
    scorecard = card(
        ("a", "b", "c"), {"a": 0.20, "b": 0.40, "c": 0.40},
        income={"c": TaskDelta(0.02, 0.0, 0.0)},
        occupation={"b": TaskDelta(0.02, 0.0, 0.0)},
    )
    joint = select_joint(scorecard, max_removals=1)
    independent = select_independent(scorecard, max_removals=1)
    assert (joint.income_removed, joint.occupation_removed) == (("a",), ("a",))
    assert (independent.income_removed, independent.occupation_removed) == (
        ("b",), ("c",)
    )
    assert joint.union_removed == ("a",)
    assert independent.union_removed == ()


def test_utility_boundary_is_inclusive_and_excess_blocks_removal():
    scorecard = card(
        ("a", "b"), {"a": 0.3, "b": 0.4},
        income={
            "a": TaskDelta(0.010, 0.005, -0.010),
            "b": TaskDelta(0.01001, 0.0, 0.0),
        },
        occupation={
            "a": TaskDelta(0.010, 0.005, -0.010),
            "b": TaskDelta(0.01001, 0.0, 0.0),
        },
    )
    selected = select_joint(scorecard, max_removals=1)
    assert (selected.income_removed, selected.occupation_removed) == (("a",), ("a",))


def test_zero_gain_prefers_empty_masks_over_unnecessary_removals():
    scorecard = card(("a", "b"), {"a": 0.0, "b": 0.0})
    selected = select_joint(scorecard, max_removals=2)
    assert selected.income_removed == ()
    assert selected.occupation_removed == ()
    assert selected.evaluated_pairs == 16


def test_eight_atom_two_removal_search_has_declared_unit_count():
    atoms = tuple("abcdefgh")
    scorecard = card(atoms, {a: 0.0 for a in atoms})
    selected = select_joint(scorecard, max_removals=2)
    assert selected.evaluated_pairs == 1369
    assert selected.income_removed == ()


def test_sequential_orders_condition_second_mask_on_first_and_can_differ():
    scorecard = card(
        ("a", "b", "c"), {"a": 0.55, "b": 0.40, "c": 0.30},
        income={
            "a": TaskDelta(0.003, 0, 0),
            "b": TaskDelta(0.009, 0, 0),
            "c": TaskDelta(0.003, 0, 0),
        },
        occupation={
            "a": TaskDelta(0.009, 0, 0),
            "b": TaskDelta(0.003, 0, 0),
            "c": TaskDelta(0.003, 0, 0),
        },
    )
    income_first = select_sequential(scorecard, first="income", max_removals=2)
    occupation_first = select_sequential(
        scorecard, first="occupation_group", max_removals=2
    )
    assert (income_first.income_removed, income_first.occupation_removed) == (
        ("a", "c"), ("a",)
    )
    assert (occupation_first.income_removed, occupation_first.occupation_removed) == (
        ("a", "c"), ("b", "c")
    )


def test_common_mask_omits_same_raw_atoms_for_both_recipients():
    scorecard = card(
        ("a", "b", "c"), {"a": 0.20, "b": 0.40, "c": 0.40},
        income={"c": TaskDelta(0.02, 0, 0)},
        occupation={"b": TaskDelta(0.02, 0, 0)},
    )
    selected = select_common_mask(scorecard, max_removals=1)
    assert selected.income_removed == ("a",)
    assert selected.occupation_removed == ("a",)
    assert selected.union_removed == ("a",)


def test_task_only_control_matches_mask_counts_and_optional_union_capacity():
    scorecard = card(
        ("a", "b"), {"a": 0.9, "b": 0.1},
        income={"b": TaskDelta(0.004, 0, 0)},
        occupation={"a": TaskDelta(0.004, 0, 0)},
    )
    unconstrained_union = select_task_only(
        scorecard, income_removals=1, occupation_removals=1
    )
    same_union_budget = select_task_only(
        scorecard, income_removals=1, occupation_removals=1,
        union_removed_count=1,
    )
    assert (unconstrained_union.income_removed, unconstrained_union.occupation_removed) == (
        ("a",), ("b",)
    )
    assert (same_union_budget.income_removed, same_union_budget.occupation_removed) == (
        ("a",), ("a",)
    )
