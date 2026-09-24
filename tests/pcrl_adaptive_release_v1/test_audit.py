"""Independent access and exact-token audit fixtures."""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def audit():
    try:
        return importlib.import_module("experiments.pcrl_adaptive_release_v1.audit")
    except ModuleNotFoundError as error:
        pytest.fail(f"adaptive audit implementation missing: {error}")


def test_private_leaf_law_is_evaluation_only_and_uses_seventeen_tokens():
    a = audit()
    q = np.zeros((3, 17))
    q[0, 2] = q[1, 4] = q[2, 6] = 1
    law = a.person_token_law(q, np.array([2, 0, 1, 2]))
    assert law.shape == (4, 17)
    assert np.array_equal(law, q[[2, 0, 1, 2]])
    with pytest.raises(ValueError):
        a.person_token_law(q, np.array([3]))
    with pytest.raises(ValueError):
        a.person_token_law(np.ones((3, 18))/18, np.array([0]))


def test_exact_expected_loss_is_not_loss_of_average_prediction():
    a = audit()
    law = np.zeros((1, 17)); law[0, :2] = .5
    pred = np.full((1, 17, 2), .5)
    pred[0, 0] = [.9, .1]
    pred[0, 1] = [.1, .9]
    exact = a.expected_token_loss(pred, law, np.array([1]))[0]
    assert exact == pytest.approx(-.5*np.log(.1)-.5*np.log(.9))
    assert exact > -np.log(.5)


def test_a_and_coalition_views_exclude_hidden_state_and_preserve_ancestors():
    a = audit()
    rows = {"ha": np.arange(8.).reshape(2, 4),
            "hb": np.arange(4.).reshape(2, 2)}
    assert a.view_features(rows, "A").shape == (2, 4)
    assert a.view_features(rows, "B").shape == (2, 2)
    assert a.view_features(rows, "AB").shape == (2, 6)
    assert np.array_equal(a.view_features(rows, "AB")[:, :4], rows["ha"])
    with pytest.raises(ValueError):
        a.view_features({"ha": rows["ha"], "hb": rows["hb"],
                         "hidden_leaf": np.array([0, 1])}, "A", strict=True)


def test_validation_selection_uses_balanced_loss_and_lexical_tie():
    a = audit()
    losses = {"z": np.array([.1, .9]), "a": np.array([.1, .9]),
              "worse": np.array([.4, 1.])}
    selected = a.select_validation(losses, np.array([1., 3.]))
    assert selected["selected"] == "a"


def test_full_nine_race_classes_stay_declared_with_missing_fit_support():
    a = audit()
    rows = {"ha": np.zeros((3, 4)), "hb": np.zeros((3, 2)),
            "labels": {"RAC1P": np.array([0, 1, 8])},
            "weights": np.ones(3), "ids": np.arange(3),
            "households": np.array(["a", "b", "c"])}
    law = np.ones((3, 17))/17
    arrays = a.role_arrays(rows, law, "attack:AB/RAC1P")
    assert arrays["n_classes"] == 9
    assert arrays["missing_classes"] == [2, 3, 4, 5, 6, 7]


def test_undeclared_auxiliary_service_cannot_enter_registered_recipient_view():
    a = audit()
    rows = {"ha": np.zeros((3, 4)), "hb": np.zeros((3, 2)),
            "aux": np.ones((3, 1)),
            "labels": {"SEX": np.array([0, 1, 0])},
            "weights": np.ones(3), "ids": np.arange(3),
            "households": np.array(["a", "b", "c"])}
    with pytest.raises(ValueError):
        a.role_arrays(rows, np.ones((3, 17))/17, "attack:A/SEX")


def test_household_overlap_is_rejected_before_fit_or_inference():
    a = audit()
    with pytest.raises(ValueError):
        a.assert_household_disjoint(np.array(["a", "b"]), np.array(["b", "c"]))


def test_coalition_route_bank_includes_own_h_a_and_b_ancestors():
    a = audit()
    pin = "a" * 64
    def registry(role, release, view, wire):
        return {"role": role, "release_id": release, "slate": "standard",
                "models": {"m": {"kind": "model", "target": "SEX",
                                  "source_view": view, "wire": wire,
                                  "source_release_id": release,
                                  "model_sha256": pin}}}
    own = registry("attack:AB/SEX", "candidate", "AB", "release")
    h = registry("attack:AB/SEX", "H", "AB", "H")
    a_same = registry("attack:A/SEX", "candidate", "A", "release")
    b = registry("attack:B/SEX", "H", "B", "H")
    routes = a.candidate_route_bank("attack:AB/SEX", "candidate", own, h,
                                    a_same=a_same, b_h_only=b)
    assert set(routes) == {"own/m", "H/m", "A/m", "B/m"}
    with pytest.raises(ValueError):
        a.candidate_route_bank("attack:AB/SEX", "candidate", own, h,
                               a_same=a_same)


def test_interaction_positive_control_beats_h_only_even_when_marginals_hide_it():
    a = audit()
    # Sensitive S = public A bit XOR released token bit; either alone is null.
    h_bit = np.array([0, 0, 1, 1])
    token_bit = np.array([0, 1, 0, 1])
    sensitive = h_bit ^ token_bit
    law = np.zeros((4, 17)); law[np.arange(4), token_bit] = 1.
    joint = np.full((4, 17, 2), .5)
    for person, h in enumerate(h_bit):
        for z in (0, 1):
            predicted = h ^ z
            joint[person, z] = [.99, .01] if predicted == 0 else [.01, .99]
    joint_loss = a.expected_token_loss(joint, law, sensitive)
    marginal_loss = a.expected_token_loss(np.full_like(joint, .5), law, sensitive)
    assert joint_loss.mean() == pytest.approx(-np.log(.99))
    assert marginal_loss.mean() == pytest.approx(np.log(2))
    assert joint_loss.mean() < marginal_loss.mean()
