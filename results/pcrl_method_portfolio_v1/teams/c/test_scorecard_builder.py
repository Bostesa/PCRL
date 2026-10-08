"""Synthetic numerical checks for the in-memory Team C scorecard builder."""

from itertools import product

import numpy as np
import pytest

from prototype import AdultGroupedSchema
from scorecard_builder import (
    ScorecardProxyCache, build_scorecard, group_disjoint_split,
    preserve_fit_class_support,
)


def synthetic_rows():
    # Independent task and SEX facts. Each stable group has two aligned rows.
    states = list(product(range(6), range(2), range(2), range(8)))
    unit = np.repeat(np.arange(len(states), dtype=np.int64), 2)
    X = np.zeros((len(unit), 83), dtype=np.float32)
    income = np.empty(len(unit), dtype=np.int64)
    occupation = np.empty(len(unit), dtype=np.int64)
    sex = np.empty(len(unit), dtype=np.int64)
    for group, (occ, inc, sensitive, _) in enumerate(states):
        rows = slice(2 * group, 2 * group + 2)
        X[rows, 0] = inc
        X[rows, 2] = sensitive
        X[rows, 4] = occ / 5
        income[rows] = inc
        occupation[rows] = occ
        sex[rows] = sensitive
    return X, income, occupation, sex, unit


def test_internal_split_keeps_paired_groups_together_and_repeats_by_seed():
    _, _, _, _, unit = synthetic_rows()
    fit1, selection1 = group_disjoint_split(unit, seed=17)
    fit2, selection2 = group_disjoint_split(unit, seed=17)
    np.testing.assert_array_equal(fit1, fit2)
    np.testing.assert_array_equal(selection1, selection2)
    assert not set(unit[fit1]) & set(unit[selection1])
    assert len(fit1) + len(selection1) == len(unit)


def test_rare_class_is_moved_whole_group_to_fit_without_replenishing_validation():
    unit = np.repeat(np.arange(12, dtype=np.int64), 2)
    fit, selection = group_disjoint_split(unit, seed=31)
    y = np.zeros(len(unit), dtype=np.int64)
    rare_group = int(unit[selection[0]])
    y[unit == rare_group] = 1
    safe_fit, safe_selection, moved = preserve_fit_class_support(
        unit, fit, selection, {"task": y}, {"task": 2}
    )
    assert moved == (rare_group,)
    assert set(unit[unit == rare_group]) <= set(unit[safe_fit])
    assert not set(unit[safe_fit]) & set(unit[safe_selection])
    assert len(safe_selection) == len(selection) - 2


def test_class_safe_repair_chooses_smallest_group_and_rechecks_missing_classes():
    unit = np.array([0, 0, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    y = np.array([0, 0, 0, 0, 1, 2, 2, 1], dtype=np.int64)
    fit = np.array([0, 1])
    selection = np.arange(2, len(unit))
    safe_fit, safe_selection, moved = preserve_fit_class_support(
        unit, fit, selection, {"task": y}, {"task": 3}
    )
    # Group 3 is smaller for class 1. Group 2 then supplies class 2.
    assert moved == (3, 2)
    assert len(safe_fit) == 6
    assert len(safe_selection) == 2


def test_one_moved_group_can_supply_two_missing_classes():
    unit = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    y = np.array([0, 0, 0, 0, 1, 2, 2], dtype=np.int64)
    fit, selection, moved = preserve_fit_class_support(
        unit, np.array([0, 1]), np.arange(2, len(unit)),
        {"task": y}, {"task": 3},
    )
    assert moved == (2,)
    assert len(fit) == 5
    assert len(selection) == 2


def test_raw_sensitive_proxy_detects_removable_feature_without_task_cost():
    schema = AdultGroupedSchema()
    X, income, occupation, sex, unit = synthetic_rows()
    base_atoms = tuple(a for a in schema.atoms if a != "relationship")
    fitted = build_scorecard(
        X, schema.feature_names, income, occupation, sex, unit,
        base_atoms=base_atoms, optional_atoms=base_atoms, seed=17,
    )
    assert len(fitted.mask_metrics) == 9
    assert fitted.fit_counts == {
        "task_proxy": 18, "sex_logistic_proxy": 9,
        "sex_tree_proxy": 9, "total": 36,
    }
    assert fitted.card.auc_drop["capital-gain"] > 0.20
    assert abs(fitted.card.task_deltas["income"]["capital-gain"].nll) < 0.05
    assert fitted.split_receipt["overlap_groups"] == 0


def test_scorecard_refuses_target_column_even_when_numeric_matrix_has_extra_column():
    schema = AdultGroupedSchema()
    X, income, occupation, sex, unit = synthetic_rows()
    base_atoms = tuple(a for a in schema.atoms if a != "relationship")
    with pytest.raises(ValueError, match="ordered 83-column schema"):
        build_scorecard(
            np.column_stack((X, income)), (*schema.feature_names, "income"),
            income, occupation, sex, unit,
            base_atoms=base_atoms, optional_atoms=base_atoms, seed=17,
        )


def test_full_and_relationship_removed_scorecards_share_exact_mask_fit():
    schema = AdultGroupedSchema()
    X, income, occupation, sex, unit = synthetic_rows()
    full_atoms = tuple(schema.atoms)
    rel0_atoms = tuple(a for a in full_atoms if a != "relationship")
    cache = ScorecardProxyCache()
    full = build_scorecard(
        X, schema.feature_names, income, occupation, sex, unit,
        base_atoms=full_atoms, optional_atoms=full_atoms, seed=17, cache=cache,
    )
    rel0 = build_scorecard(
        X, schema.feature_names, income, occupation, sex, unit,
        base_atoms=rel0_atoms, optional_atoms=rel0_atoms, seed=17, cache=cache,
    )
    assert full.fit_counts["total"] == 40
    assert rel0.fit_counts["total"] == 32
    assert rel0.alias_masks == ("base",)
    assert len(cache.mask_metrics) == 18
    assert full.fit_counts["total"] + rel0.fit_counts["total"] == 72


def test_depth_two_raw_proxy_detects_complementary_xor_clue():
    schema = AdultGroupedSchema()
    states = list(product(range(6), range(2), range(2), range(2), range(8)))
    X = np.zeros((len(states) * 2, 83), dtype=np.float32)
    income = np.empty(len(X), dtype=np.int64)
    occupation = np.empty(len(X), dtype=np.int64)
    sex = np.empty(len(X), dtype=np.int64)
    unit = np.repeat(np.arange(len(states), dtype=np.int64), 2)
    for group, (occ, inc, clue, sensitive, _) in enumerate(states):
        rows = slice(2 * group, 2 * group + 2)
        X[rows, 0] = inc
        X[rows, 2] = clue
        X[rows, 3] = clue ^ sensitive
        X[rows, 4] = occ / 5
        income[rows] = inc
        occupation[rows] = occ
        sex[rows] = sensitive
    fitted = build_scorecard(
        X, schema.feature_names, income, occupation, sex, unit,
        base_atoms=("age", "capital-gain", "capital-loss", "hours-per-week"),
        optional_atoms=("capital-gain", "capital-loss"), seed=23,
    )
    base = fitted.mask_metrics[0]
    assert base["sex_readers"]["tree_depth2"]["reversible_auc"] > 0.90
    assert fitted.card.auc_drop["capital-gain"] > 0.25
    assert fitted.card.auc_drop["capital-loss"] > 0.25
