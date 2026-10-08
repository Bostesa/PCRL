"""Skeptical synthetic checks for Team C control interpretation.

These tests use no Adult files or role exports. Queue them with the other
post-amendment numerical checks before a real-data seal.
"""

from controls import _sequential_at_capacity
from selector import Scorecard, TaskDelta


def test_strict_local_first_mismatch_is_not_global_capacity_impossibility():
    zero = TaskDelta(0.0, 0.0, 0.0)
    too_costly = TaskDelta(0.020, 0.0, 0.0)
    card = Scorecard(
        optional_atoms=("a", "b", "c"),
        auc_drop={"a": 0.4, "b": 0.3, "c": 0.1},
        task_deltas={
            "income": {"a": zero, "b": zero, "c": zero},
            "occupation_group": {"a": too_costly, "b": zero, "c": zero},
        },
    )
    selected, status = _sequential_at_capacity(
        card, first="income", income_count=1, occupation_count=1,
        common_removed_count=1,
    )
    assert status == "UNATTAINABLE_AFTER_FIRST"
    assert selected is not None
    assert selected.income_removed == ("a",)
    assert selected.occupation_removed == ("b",)
    assert selected.union_removed == ()

    # A matched pair exists in the frozen feasible sets. The status therefore
    # describes the committed strict local-first path, not global impossibility.
    assert card.task_deltas["income"]["b"] == zero
    assert card.task_deltas["occupation_group"]["b"] == zero
