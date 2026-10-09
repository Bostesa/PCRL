"""Data-free checks of independent FIT grouping and role allowlist."""

import numpy as np

from results.pcrl_utility_reference_repair_v1.r1.fit_splits import (
    inner_split, outer_split,
)
from results.pcrl_utility_reference_repair_v1.r3.task_gate_replay_worker import (
    _constant_task_gain, _own_split,
)


def test_independent_group_split_matches_sealed_recipe_geometry():
    row = np.arange(600, dtype=np.int64)
    unit = row // 2
    income = row % 2
    occupation = row % 6
    for seed in (0, 1):
        own_fit, own_val = _own_split(unit, income, occupation,
                                      seed, 0.25, 26000)
        r1_fit, r1_val, _ = outer_split(unit, income, occupation, seed)
        assert np.array_equal(own_fit, r1_fit)
        assert np.array_equal(own_val, r1_val)
        inner_fit, inner_val = _own_split(unit[own_fit], income[own_fit],
                                          occupation[own_fit], seed, 0.15, 17001)
        r1_inner_fit, r1_inner_val, _ = inner_split(
            unit[own_fit], income[own_fit], occupation[own_fit], seed)
        assert np.array_equal(inner_fit, r1_inner_fit)
        assert np.array_equal(inner_val, r1_inner_val)


def test_full_fit_majority_constant_is_scored_on_common_role():
    # FIT has a class-count tie, so the lower class must be the fixed choice.
    fit = np.array([0, 0, 1, 1], dtype=np.int64)
    common = np.array([0, 1, 1, 1], dtype=np.int64)
    result = _constant_task_gain(fit, common, 0.75, 2)
    assert result == {
        "FIT_class_counts": [2, 2],
        "FIT_majority_class": 0,
        "INNER_constant_accuracy": 0.25,
        "F_accuracy_gain": 0.5,
        "F_useful_at_0_010": True,
    }
