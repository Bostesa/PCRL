"""Behavioral fixtures for matched finite 17-token controls."""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def controls():
    try:
        return importlib.import_module("experiments.pcrl_adaptive_release_v1.controls")
    except ModuleNotFoundError as error:
        pytest.fail(f"new matched-controls implementation missing: {error}")


def _one_hot(actions, n_tokens=17):
    q = np.zeros((len(actions), n_tokens), dtype=np.float64)
    q[np.arange(len(actions)), actions] = 1.0
    return q


def test_lift_parent_channel_repeats_exact_parent_rows_without_reencoding():
    c = controls()
    parent = _one_hot([1, 16])
    child = c.lift_parent_channel(parent, np.array([1, 0, 1, 1]))
    assert child.shape == (4, 17)
    assert np.array_equal(child, parent[[1, 0, 1, 1]])
    with pytest.raises(ValueError):
        c.lift_parent_channel(parent, np.array([0, 2]))
    almost = parent.copy()
    almost[0, 0] = -1e-13
    almost[0, 1] += 1e-13
    with pytest.raises(ValueError):
        c.lift_parent_channel(almost, np.array([0]))


def test_constant_replacement_stays_on_seventeen_token_wire():
    c = controls()
    base = _one_hot([0, 1])
    result = c.constant_replacement(base, replace_probability=.25, token=16)
    assert result.shape == (2, 17)
    assert result[0, 0] == pytest.approx(.75)
    assert result[0, 16] == pytest.approx(.25)
    assert result[1, 1] == pytest.approx(.75)
    assert result[1, 16] == pytest.approx(.25)
    assert np.array_equal(result.sum(axis=1), np.ones(2))
    with pytest.raises(ValueError):
        c.constant_replacement(base, replace_probability=.2, token=17)


def test_randomized_response_is_actual_token_mixture_not_prediction_average():
    c = controls()
    base = _one_hot([0])
    result = c.randomized_response(base, publish_probability=.5)
    assert result.shape == (1, 17)
    assert result[0, 0] == pytest.approx(.5 + .5 / 17)
    assert result[0, 1] == pytest.approx(.5 / 17)
    assert result.sum() == pytest.approx(1.)


def test_replay_cost_and_cut_use_already_weighted_coefficients_once():
    c = controls()
    cost = np.zeros((2, 17))
    cost[0, 0], cost[1, 1] = .2, .3
    coeff = np.zeros_like(cost)
    coeff[0, 0], coeff[1, 1] = .4, .5
    q = _one_hot([0, 1])
    bank = c.make_bank(cost, [{"id": "risk", "role": "A/SEX", "weighting": "U",
                               "coeff": coeff, "floor": .85}])
    replay = c.replay(q, bank, deterministic=True)
    assert replay["objective"] == pytest.approx(.5)
    assert replay["cut_losses"]["risk"] == pytest.approx(.9)
    assert replay["cut_slacks"]["risk"] == pytest.approx(.05)
    assert replay["feasible"]


def test_one_state_milp_closes_deterministic_bound_with_real_randomization_gap():
    c = controls()
    cost = np.full((1, 17), 2.)
    cost[0, 0], cost[0, 1] = 0., 1.
    coeff = np.zeros_like(cost)
    coeff[0, 1] = 1.
    cuts = [{"id": "risk", "role": "A/SEX", "weighting": "U",
             "coeff": coeff, "floor": .5}]
    solved = c.solve_deterministic_p1(cost, cuts, time_limit_seconds=10.)
    assert solved["incumbent_valid"]
    assert solved["incumbent_objective"] == pytest.approx(1., abs=1e-7)
    assert solved["lower_bound"] == pytest.approx(1., abs=1e-7)
    assert solved["Q"].shape == (1, 17)
    assert solved["Q"][0, 1] == 1.


def test_gradient_control_uses_exact_cost_and_reduces_unconstrained_loss():
    c = controls()
    cost = np.ones((1, 17))
    cost[0, 0] = 0.
    result = c.optimize_gradient_bank(cost, [], steps=100,
                                      learning_rate=.1, penalty=0., seed=3)
    assert result["Q"].shape == (1, 17)
    assert result["replay"]["objective"] < .25
    assert result["Q"].sum() == pytest.approx(1.)


def test_child_coefficient_aggregation_matches_person_token_enumeration():
    c = controls()
    leaves = np.array([0, 2, 0, 1])
    per_token = np.arange(4 * 17, dtype=float).reshape(4, 17) / 100
    weights = np.array([1., 2., 3., 4.])
    coeff = c.aggregate_coefficients(leaves, per_token, weights, n_states=4)
    assert coeff.shape == (4, 17)
    assert np.array_equal(coeff[3], np.zeros(17))
    q = np.ones((4, 17))/17
    q[0] = np.eye(17)[2]
    direct = np.sum((weights / weights.sum()) *
                    np.sum(per_token * q[leaves], axis=1))
    assert np.sum(coeff * q) == pytest.approx(direct)
    with pytest.raises(ValueError):
        c.aggregate_coefficients(leaves, per_token, weights,
                                 n_states=4, normalized_weights=True)
