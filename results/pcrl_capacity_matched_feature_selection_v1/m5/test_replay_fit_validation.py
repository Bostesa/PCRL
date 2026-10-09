"""Data-free checks of the independent pre-sensitive FIT replay."""

import numpy as np
import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_fit_validation import (
    task_only_split, compare_task_row,
)


def test_task_split_is_group_disjoint_and_reproducible():
    unit = np.repeat(np.arange(24, dtype=np.int64), 2)
    income = np.tile(np.array([0, 1], dtype=np.int64), 24)
    occupation = np.tile(np.arange(6, dtype=np.int64), 8)
    fit, val, receipt = task_only_split(unit, income, occupation, 1)
    assert len(np.intersect1d(unit[fit], unit[val])) == 0
    assert len(val) > 0
    assert np.array_equal(task_only_split(unit, income, occupation, 1)[0], fit)
    assert receipt["validation_row_index_sha256"] == receipt["validation_row_index_sha256"]


def test_task_row_rejects_changed_probability_metric():
    observed = {"accuracy": 1.0, "nll": 0.1, "brier": 0.2}
    saved = {"metrics": dict(observed), "utility_margins_by_ref_metric": {
        "U0": {"accuracy": 0.01, "nll": 0.01, "brier": 0.005},
        "Ucal": {"accuracy": 0.01, "nll": 0.01, "brier": 0.005},
        "F": {"accuracy": 0.01, "nll": 0.01, "brier": 0.005},
    }, "utility_feasible": True}
    refs = {key: {"accuracy": 1.0, "nll": 0.1, "brier": 0.2}
            for key in ("U0", "Ucal", "F")}
    assert compare_task_row(saved, observed, refs)
    with pytest.raises(ValueError, match="metric"):
        compare_task_row(saved, {**observed, "nll": 0.1001}, refs)
