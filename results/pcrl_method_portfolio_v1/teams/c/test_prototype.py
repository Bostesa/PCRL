"""Synthetic contract checks for the Team C grouped feature release."""

import numpy as np
import pytest

from prototype import (
    AdultGroupedSchema, IdentityCalibrator, PurposeRelease, candidate_pair,
    release_pair,
)


class SumHead:
    """Small deterministic head used to check the release boundary."""

    def predict_proba(self, selected):
        logits = selected.sum(axis=1)
        p1 = 1 / (1 + np.exp(-logits))
        return np.column_stack((1 - p1, p1))


class SixClassHead:
    def predict_proba(self, selected):
        p0 = 1 / (1 + np.exp(-selected.sum(axis=1)))
        return np.column_stack((p0, *([(1 - p0) / 5] * 5)))


def make_release(schema, atoms, head, purpose="income"):
    return PurposeRelease(
        schema, atoms, head, purpose=purpose,
        head_id=f"test-head-{purpose}", calibrator_id="test-calibrator",
        calibrator=IdentityCalibrator(),
    )


def make_candidate(schema, income_atoms, occupation_atoms):
    return candidate_pair(
        schema, income_atoms, occupation_atoms, SumHead(), SixClassHead(),
        "i-head", "o-head", ("i-cal", "o-cal"),
        IdentityCalibrator(), IdentityCalibrator(),
    )


def test_adult_schema_has_exact_pinned_groups_and_no_label_columns():
    schema = AdultGroupedSchema()
    assert len(schema.feature_names) == 83
    assert {name: len(indices) for name, indices in schema.atoms.items()} == {
        "age": 1,
        "capital-gain": 1,
        "capital-loss": 1,
        "hours-per-week": 1,
        "workclass": 8,
        "education_fact": 17,
        "marital-status": 7,
        "relationship": 6,
        "native-country": 41,
    }
    assert not any(
        name == "sex" or name == "race" or name == "income"
        or name == "occupation" or name == "fnlwgt"
        for name in schema.feature_names
    )


def test_schema_refuses_reordered_or_added_label_predictors():
    schema = AdultGroupedSchema()
    with pytest.raises(ValueError, match="ordered 83-column schema"):
        schema.validate_feature_names(schema.feature_names[::-1])
    with pytest.raises(ValueError, match="ordered 83-column schema"):
        schema.validate_feature_names((*schema.feature_names, "income"))


def test_one_categorical_choice_releases_every_one_hot_column_and_no_other_values():
    schema = AdultGroupedSchema()
    X = np.arange(2 * 83, dtype=float).reshape(2, 83)
    release = make_release(schema, ("workclass",), SumHead())
    view = release.view(X, schema.feature_names)
    assert view["feature_names"] == schema.feature_names[5:13]
    np.testing.assert_array_equal(view["selected_values"], X[:, 5:13])
    assert view["selected_groups"] == ("workclass",)
    assert view["probabilities"].shape == (2, 2)
    assert view["decisions"].shape == (2,)


def test_education_number_and_one_hot_are_one_indivisible_selection_atom():
    schema = AdultGroupedSchema()
    X = np.arange(83, dtype=float).reshape(1, 83)
    with pytest.raises(ValueError, match="unknown selection atom"):
        make_release(schema, ("education-num",), SumHead())
    with pytest.raises(ValueError, match="unknown selection atom"):
        make_release(schema, ("education",), SumHead())
    view = make_release(schema, ("education_fact",), SumHead()).view(
        X, schema.feature_names
    )
    assert len(view["feature_names"]) == 17
    assert view["feature_names"][0] == "education-num"
    assert view["feature_names"][1:] == schema.feature_names[13:29]


def test_head_cannot_see_unselected_columns_and_coalition_sees_raw_union():
    schema = AdultGroupedSchema()
    X = np.zeros((2, 83), dtype=float)
    X[:, 0] = (0.0, 2.0)
    X[:, 2] = (1.0, 3.0)
    first = make_release(schema, ("age",), SumHead())
    second = make_release(schema, ("capital-gain",), SixClassHead(), "occupation_group")
    views = release_pair(X, schema.feature_names, first, second)
    np.testing.assert_array_equal(views["coalition"]["selected_values"], X[:, (0, 2)])
    assert views["coalition"]["feature_names"] == ("age", "capital-gain")
    assert views["coalition"]["selected_groups"] == ("age", "capital-gain")
    altered = X.copy()
    altered[:, (1, *range(3, 83))] = 999.0
    other_views = release_pair(altered, schema.feature_names, first, second)
    for recipient in ("income", "occupation_group"):
        np.testing.assert_array_equal(
            views[recipient]["probabilities"], other_views[recipient]["probabilities"]
        )


def test_mask_is_global_and_not_selected_per_person():
    schema = AdultGroupedSchema()
    X = np.zeros((2, 83), dtype=float)
    X[0, 5] = 1.0
    X[1, 12] = 1.0
    release = make_release(schema, ("workclass",), SumHead())
    view = release.view(X, schema.feature_names)
    assert view["selected_values"].shape == (2, 8)
    assert view["feature_names"] == schema.feature_names[5:13]


def test_candidate_pair_refuses_relationship_proxy_in_either_mask():
    schema = AdultGroupedSchema()
    with pytest.raises(ValueError, match="relationship"):
        make_candidate(schema, ("age", "relationship"), ("capital-gain",))
    with pytest.raises(ValueError, match="relationship"):
        make_candidate(schema, ("age",), ("relationship",))
    first, second = make_candidate(schema, ("age", "education_fact"), ("capital-gain",))
    assert "relationship" not in first.selected_atoms
    assert "relationship" not in second.selected_atoms


def test_complete_views_expose_head_calibrator_and_mask_identifiers():
    schema = AdultGroupedSchema()
    X = np.zeros((1, 83), dtype=float)
    first, second = candidate_pair(
        schema, ("age",), ("capital-gain",),
        SumHead(), SixClassHead(), "income-fit-0", "occupation-fit-0",
        ("income-cal-0", "occupation-cal-0"),
        IdentityCalibrator(), IdentityCalibrator(),
    )
    views = release_pair(X, schema.feature_names, first, second)
    assert views["income"]["head_id"] == "income-fit-0"
    assert views["occupation_group"]["calibrator_id"] == "occupation-cal-0"
    assert views["income"]["mask_id"] == "age"
    assert views["coalition"]["income_head_id"] == "income-fit-0"
    assert views["coalition"]["occupation_group_mask_id"] == "capital-gain"
    assert views["coalition"]["income_calibrator_id"] == "income-cal-0"
    assert views["coalition"]["occupation_group_calibrator_id"] == "occupation-cal-0"
    assert views["coalition"]["flags"] == ()


def test_wrong_purpose_head_width_is_rejected_before_release():
    schema = AdultGroupedSchema()
    X = np.zeros((1, 83), dtype=float)
    wrong = make_release(schema, ("age",), SumHead(), "occupation_group")
    with pytest.raises(ValueError, match="six class"):
        wrong.view(X, schema.feature_names)


def test_pair_refuses_swapped_purpose_heads():
    schema = AdultGroupedSchema()
    X = np.zeros((1, 83), dtype=float)
    first = make_release(schema, ("age",), SixClassHead(), "occupation_group")
    second = make_release(schema, ("capital-gain",), SumHead())
    with pytest.raises(ValueError, match="income and occupation_group"):
        release_pair(X, schema.feature_names, first, second)


def test_transmitted_scores_are_actual_calibrated_selected_head_outputs():
    class FixedCalibrator:
        def transform(self, raw_probabilities):
            assert raw_probabilities.shape == (1, 2)
            return np.array([[0.8, 0.2]])

    schema = AdultGroupedSchema()
    X = np.zeros((1, 83), dtype=float)
    release = PurposeRelease(
        schema, ("age",), SumHead(), purpose="income", head_id="raw-head",
        calibrator_id="fixed-cal", calibrator=FixedCalibrator(),
    )
    view = release.view(X, schema.feature_names)
    np.testing.assert_array_equal(view["probabilities"], np.array([[0.8, 0.2]]))
    np.testing.assert_array_equal(view["decisions"], np.array([0]))
