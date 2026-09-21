"""Hand-checked finite-channel fixtures, independent of the CVXPY expression."""

import importlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq


def module():
    # A local import makes the first red run report missing implementation as a test failure.
    return importlib.import_module("experiments.pcrl_task_directed_release_v1.finite")


def private_fixture():
    # State order: (s,y) = (0,0), (0,1), (1,0), (1,1).
    p = np.zeros((2, 1, 4))
    p[0, 0, :2] = 0.25
    p[1, 0, 2:] = 0.25
    q = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    cost = -0.25 * np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    return p, q, np.vstack([cost, cost])


def test_joint_table_normalizes_weights_and_keeps_empty_cells():
    finite = module()
    got = finite.joint_table([0, 0, 1], [0, 1, 0], [0, 1, 0], 2, 2, 3, weights=[1, 2, 1])
    want = np.zeros((2, 2, 3))
    want[0, 0, 0], want[0, 1, 1], want[1, 0, 0] = 0.25, 0.5, 0.25
    assert_allclose(got, want, atol=0)
    assert finite.joint_table([0, 1], [0, 0], [0, 1], 2, 1, 2).sum() == 1


def test_costs_include_joint_mass_once_with_individual_action_probabilities():
    finite = module()
    got = finite.cost_table([0, 1, 1], [0, 1, 0], [[0.1, 0.7], [0.2, 0.8], [0.3, 0.9]], 3, weights=[2, 1, 1])
    want = np.array([
        [-0.5 * np.log(0.9), -0.5 * np.log(0.3)],
        [-0.25 * (np.log(0.2) + np.log(0.7)), -0.25 * (np.log(0.8) + np.log(0.1))],
        [0, 0],
    ])
    assert_allclose(got, want)
    result = finite.solve(got, {}, None, state_mass=np.array([2, 2, 0]))
    assert result["feasible"]
    assert result["objective"] == pytest.approx(want[0, 0] + want[1, 0], abs=1e-9)
    assert_allclose(result["Q"][2], [1, 0], atol=1e-10)


def test_cmi_constant_and_exact_private_channel():
    finite = module()
    p, q, _ = private_fixture()
    assert finite.cmi(p, q) == pytest.approx(0, abs=1e-14)
    assert finite.cmi(p, np.tile([0.2, 0.8], (4, 1))) == pytest.approx(0, abs=1e-14)
    reveals_s = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    assert finite.cmi(p, reveals_s) == pytest.approx(np.log(2))


def test_cmi_is_convex_under_channel_mixing():
    finite = module()
    rng = np.random.default_rng(422)
    p = rng.uniform(size=(3, 2, 5))
    p /= p.sum()
    q1, q2 = rng.dirichlet([0.3, 1, 0.2], size=5), rng.dirichlet([1, 0.2, 0.6], size=5)
    mix = 0.37
    assert finite.cmi(p, mix * q1 + (1 - mix) * q2) <= mix * finite.cmi(p, q1) + (1 - mix) * finite.cmi(p, q2) + 1e-13


def test_zero_budget_lp_finds_nonconstant_private_action_and_checks_cmi():
    finite = module()
    p, _, cost = private_fixture()
    result = finite.solve(cost, {"A/SEX/U": p, "A/SEX/PWGTP": p}, 0.0)
    assert result["feasible"] and result["status"] == "optimal"
    assert result["objective"] == pytest.approx(-np.log(0.9), abs=1e-8)
    assert finite.cmi(p, result["Q"]) < 1e-12
    assert_allclose(finite.privacy_matrix(p) @ result["Q"], 0, atol=1e-10)
    assert result["rank"] >= 1
    assert result["attempts"]


def test_zero_budget_does_not_amplify_independence_roundoff_into_infeasibility():
    # Scaling a cancellation residual of 1e-18 to unit size can falsely rule
    # out even constant channels. This product law allows every channel.
    finite = module()
    rng = np.random.default_rng(12)
    ps, pt = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(4))
    p = np.outer(ps, pt)[:, None, :]
    cost = pt[:, None] * np.array([[0.1, 2.0], [2.0, 0.1], [0.1, 2.0], [2.0, 0.1]])
    result = finite.solve(cost, {"A/SEX/U": p}, 0.0)
    assert result["feasible"], result
    assert result["objective"] == pytest.approx(0.1, abs=1e-9)
    assert finite.cmi(p, result["Q"]) < 1e-14


def test_positive_budget_matches_binary_symmetric_channel_solution():
    finite = module()
    p = np.zeros((2, 1, 2))
    p[0, 0, 0] = p[1, 0, 1] = 0.5
    cost = -0.5 * np.log([[0.9, 0.1], [0.1, 0.9]])
    budget = 0.04
    result = finite.solve(cost, {"A/SEX/U": p}, budget)
    error = brentq(lambda r: np.log(2) + r*np.log(r) + (1-r)*np.log(1-r) - budget, 1e-9, 0.5)
    want = -(1-error)*np.log(0.9) - error*np.log(0.1)
    assert result["feasible"], result
    assert result["objective"] == pytest.approx(want, abs=2e-6)
    assert finite.cmi(p, result["Q"]) <= budget + 1e-7
    assert result["residuals"]["cmi"]["A/SEX/U"] == pytest.approx(finite.cmi(p, result["Q"]), abs=1e-14)


def test_positive_budget_keeps_all_views_with_empty_context_cells():
    finite = module()
    local = np.full((2, 1, 2), 0.25)
    coalition = np.zeros((2, 3, 2))
    for s in range(2):
        for h in range(2):
            coalition[s, h, s ^ h] = 0.25
    # The third context is declared but never observed; no smoothing is needed.
    cost = -0.5 * np.log([[0.9, 0.1], [0.1, 0.9]])
    roles = {"A/SEX/U": local, "AB/SEX/U": coalition, "AB/SEX/PWGTP": coalition}
    result = finite.solve(cost, roles, 0.04)
    assert result["feasible"], result
    error = brentq(lambda r: np.log(2)+r*np.log(r)+(1-r)*np.log(1-r)-0.04, 1e-9, 0.5)
    want = -(1-error)*np.log(0.9)-error*np.log(0.1)
    assert result["objective"] == pytest.approx(want, abs=2e-6)
    assert set(result["residuals"]["cmi"]) == set(roles)
    assert finite.cmi(local, result["Q"]) < 1e-12
    assert all(finite.cmi(p, result["Q"]) <= 0.04+1e-7 for p in roles.values())


def test_each_weighted_role_is_constrained():
    finite = module()
    p, _, cost = private_fixture()
    # Under a second law the sensitive label equals the task label.
    weighted = np.zeros_like(p)
    weighted[0, 0, [0, 2]] = 0.25
    weighted[1, 0, [1, 3]] = 0.25
    result = finite.solve(cost, {"A/SEX/U": p, "A/SEX/PWGTP": weighted}, 0.0)
    assert result["feasible"]
    assert finite.cmi(p, result["Q"]) < 1e-12
    assert finite.cmi(weighted, result["Q"]) < 1e-12
    assert result["objective"] == pytest.approx(-0.5*np.log(0.9)-0.5*np.log(0.1), abs=1e-8)


def test_refinement_aggregation_embedding_and_objective_ordering():
    finite = module()
    p, _, cost = private_fixture()
    fine = np.repeat(p, 2, axis=2) * 0.5
    fine_cost = np.repeat(cost, 2, axis=0) * 0.5
    fine_parent = np.repeat(np.arange(4), 2)
    coarse = finite.solve(cost, {"A/SEX/U": p}, 0.0)
    embedded = coarse["Q"][fine_parent]
    assert_allclose(fine.reshape(2, 1, 4, 2).sum(axis=3), p)
    assert_allclose(fine_cost.reshape(4, 2, 2).sum(axis=1), cost)
    assert finite.cmi(fine, embedded) == pytest.approx(finite.cmi(p, coarse["Q"]), abs=1e-14)
    assert np.sum(fine_cost * embedded) == pytest.approx(coarse["objective"])
    result = finite.solve(fine_cost, {"A/SEX/U": fine}, 0.0, parent=fine_parent, state_mass=np.ones(8), embedded_q=coarse["Q"])
    assert result["feasible"]
    assert result["witness"]["feasible"]
    assert result["objective"] <= coarse["objective"] + 1e-8


def test_unsupported_child_is_frequency_weighted_affine_sibling_mixture():
    finite = module()
    cost = np.array([[0, 2], [3, 0], [0, 0], [0, 0], [0, 0]], dtype=float)
    result = finite.solve(cost, {}, None, parent=np.array([0, 0, 0, 1, 1]), state_mass=np.array([1, 3, 0, 0, 0]), zero_action=1)
    assert result["feasible"]
    assert_allclose(result["Q"][0], [1, 0], atol=1e-9)
    assert_allclose(result["Q"][1], [0, 1], atol=1e-9)
    assert_allclose(result["Q"][2], [0.25, 0.75], atol=1e-9)
    assert_allclose(result["Q"][3:], [[0, 1], [0, 1]], atol=1e-9)
    assert result["support"]["unsupported_states"] == [2, 3, 4]


def test_coalition_cannot_improve_over_local_under_the_same_objective():
    finite = module()
    p, _, cost = private_fixture()
    coalition = np.zeros_like(p)
    coalition[0, 0, [0, 2]] = 0.25
    coalition[1, 0, [1, 3]] = 0.25
    local = finite.solve(cost, {"A/SEX/U": p}, 0.0)
    both = finite.solve(cost, {"A/SEX/U": p, "AB/SEX/U": coalition}, 0.0)
    assert local["feasible"] and both["feasible"]
    assert both["objective"] >= local["objective"] - 1e-9


def test_xor_exposes_coarse_conditioning_limitation():
    finite = module()
    # S independent H, T = S xor H. Coarse H is constant; full H unlocks S.
    coarse = np.full((2, 1, 2), 0.25)
    full = np.zeros((2, 2, 2))
    for s in range(2):
        for h in range(2):
            full[s, h, s ^ h] = 0.25
    assert finite.cmi(coarse, np.eye(2)) == pytest.approx(0, abs=1e-14)
    assert finite.cmi(full, np.eye(2)) == pytest.approx(np.log(2))


def test_invalid_witness_is_not_used_as_a_privacy_certificate():
    finite = module()
    p, _, cost = private_fixture()
    leaking = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    with pytest.raises(ValueError, match="witness"):
        finite.solve(cost, {"A/SEX/U": p}, 0.0, embedded_q=leaking)


def test_solver_retry_and_failure_diagnostics_are_json_safe(monkeypatch):
    finite = module()
    p = np.zeros((2, 1, 2))
    p[0, 0, 0] = p[1, 0, 1] = 0.5
    cost = -0.5 * np.log([[0.9, 0.1], [0.1, 0.9]])
    # Real solver failure at its documented options boundary; no fake solver.
    monkeypatch.setitem(finite.SOLVER_SETTINGS, "CLARABEL", {"max_iter": -1})
    result = finite.solve(cost, {"A/SEX/U": p}, 0.04)
    assert result["feasible"]
    assert [attempt["solver"] for attempt in result["attempts"]] == ["CLARABEL", "SCS"]
    assert result["attempts"][0]["status"] == "solver_error"
    json.dumps({key: value for key, value in result.items() if key != "Q"}, allow_nan=False)


def test_solver_failure_can_return_verified_witness_but_never_as_optimal(monkeypatch):
    finite = module()
    p, _, cost = private_fixture()
    monkeypatch.setitem(finite.SOLVER_SETTINGS, "CLARABEL", {"max_iter": -1})
    monkeypatch.setitem(finite.SOLVER_SETTINGS, "SCS", {"max_iters": 0})
    witness = np.tile([0.5, 0.5], (4, 1))
    result = finite.solve(cost, {"A/SEX/U": p}, 0.04, embedded_q=witness)
    assert result["feasible"] and not result["optimal"]
    assert result["status"] == "feasible_witness"
    assert_allclose(result["Q"], witness)
    assert len(result["attempts"]) == 2
    assert all(not attempt["accepted"] for attempt in result["attempts"])


@pytest.mark.parametrize("invalid", [np.array([-1.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, np.nan])])
def test_joint_table_rejects_invalid_weights(invalid):
    with pytest.raises(ValueError):
        module().joint_table([0, 1], [0, 0], [0, 1], 2, 1, 2, weights=invalid)


def test_invalid_probabilities_indices_and_laws_are_rejected():
    finite = module()
    with pytest.raises(ValueError):
        finite.joint_table([0.5], [0], [0], 2, 1, 1)
    with pytest.raises(ValueError):
        finite.cost_table([0], [1], [[1.1]], 1)
    with pytest.raises(ValueError):
        finite.cmi(np.ones((2, 1, 2)), np.eye(2))
    with pytest.raises(ValueError):
        finite.solve(np.ones((2, 2)), {}, -0.1)
