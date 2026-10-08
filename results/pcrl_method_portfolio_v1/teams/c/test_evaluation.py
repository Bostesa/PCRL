"""Synthetic checks that attacks receive the complete Team C feature views."""

import numpy as np

from evaluation import attack_views_from_release
from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease, release_pair


class BinaryHead:
    def predict_proba(self, selected):
        p = np.clip(0.2 + 0.3 * selected[:, 0], 0.05, 0.95)
        return np.column_stack((1 - p, p))


class SixHead:
    def predict_proba(self, selected):
        p = np.full((len(selected), 6), 0.1)
        p[:, 0] = 0.5
        return p


def sample_views():
    schema = AdultGroupedSchema()
    X = np.zeros((4, 83), dtype=float)
    X[:, 0] = [0, 1, 0, 1]
    X[:, 2] = [0, 0, 1, 1]
    income = PurposeRelease(
        schema, ("age",), BinaryHead(), "income", "head1", "cal1",
        IdentityCalibrator(),
    )
    occupation = PurposeRelease(
        schema, ("capital-gain",), SixHead(), "occupation_group",
        "head2", "cal2", IdentityCalibrator(),
    )
    return release_pair(X, schema.feature_names, income, occupation)


def alignment():
    return {
        name: {"row_id": np.array([11, 12, 13, 14]),
               "unit": np.array([100, 100, 101, 102])}
        for name in ("income", "occupation_group", "coalition")
    }


def test_raw_union_and_both_outputs_present_in_complete_attack_view():
    release = sample_views()
    views = attack_views_from_release(release, alignment())
    assert set(views) == {"income", "occupation_group", "coalition"}
    pair = views["coalition"]
    assert pair.raw_matrix.shape == (4, 2)
    assert pair.complete_matrix.shape[1] > pair.raw_matrix.shape[1]
    assert "age" in pair.raw_names and "capital-gain" in pair.raw_names
    assert any("income_probability" in n for n in pair.complete_names)
    assert any("occupation_group_probability" in n for n in pair.complete_names)
    assert pair.ignore_recipient_aliases == ("income", "occupation_group")
    assert not any("row_id" in n or "unit" in n for n in pair.complete_names)


def test_rejects_coalition_value_that_does_not_equal_local_union():
    release = sample_views()
    release["coalition"]["selected_values"][0, 0] = 777
    try:
        attack_views_from_release(release, alignment())
    except ValueError as exc:
        assert "union" in str(exc).lower()
    else:
        raise AssertionError("incorrect coalition union accepted")


def test_rejects_coalition_output_that_does_not_equal_local_output():
    release = sample_views()
    release["coalition"]["income_probabilities"] = release["coalition"][
        "income_probabilities"
    ].copy()
    release["coalition"]["income_probabilities"][0, 0] = 777
    try:
        attack_views_from_release(release, alignment())
    except ValueError as exc:
        assert "probabilit" in str(exc).lower()
    else:
        raise AssertionError("incorrect coalition output accepted")


def test_rejects_unencoded_person_level_flags():
    release = sample_views()
    release["income"]["flags"] = [0, 1, 0, 1]
    try:
        attack_views_from_release(release, alignment())
    except ValueError as exc:
        assert "flags" in str(exc).lower()
    else:
        raise AssertionError("person-level flags were silently omitted")


def test_rejects_label_column_or_partial_categorical_group():
    release = sample_views()
    release["income"]["feature_names"] = ("sex",)
    try:
        attack_views_from_release(release, alignment())
    except ValueError as exc:
        assert "permitted" in str(exc).lower() or "atom" in str(exc).lower()
    else:
        raise AssertionError("forbidden predictor name was accepted")

    release = sample_views()
    release["income"]["selected_groups"] = ("workclass",)
    try:
        attack_views_from_release(release, alignment())
    except ValueError as exc:
        assert "atom" in str(exc).lower()
    else:
        raise AssertionError("partial categorical block was accepted")


def test_disjoint_local_masks_still_require_identical_private_person_order():
    release = sample_views()
    identity = alignment()
    identity["occupation_group"]["row_id"] = np.array([12, 11, 13, 14])
    identity["occupation_group"]["unit"] = np.array([100, 100, 101, 102])
    try:
        attack_views_from_release(release, identity)
    except ValueError as exc:
        assert "misaligned" in str(exc).lower()
    else:
        raise AssertionError("person permutation was accepted")
