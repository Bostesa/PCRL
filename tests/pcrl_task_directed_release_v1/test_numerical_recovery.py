"""Synthetic algebra, constraints, and single-attempt recovery checks."""

import importlib
import json

import cvxpy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq
from scipy.special import rel_entr

from experiments.pcrl_task_directed_release_v1 import finite


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.numerical_recovery')


def binary():
    p = np.zeros((2, 1, 2))
    p[0, 0, 0] = p[1, 0, 1] = .5
    return p, -.5 * np.log([[.9, .1], [.1, .9]])


def test_normalized_terms_equal_original_cmi_with_rare_and_empty_cells():
    rng = np.random.default_rng(44)
    p = rng.uniform(size=(3, 4, 5))
    p[:, 0] *= 1e-10
    p[1, 2] = 0
    p[:, 3] = 0
    p /= p.sum()
    q = rng.dirichlet([.3, .7, 1.2], size=5)
    mass, a, b, details = module().normalized_entropy_coefficients(p)
    assert_allclose(a.sum(axis=1), 1, atol=2e-15)
    assert_allclose(b.sum(axis=1), 1, atol=2e-15)
    want = finite.cmi(p, q)
    got = np.sum(mass[:, None] * rel_entr(a @ q, b @ q))
    assert got == pytest.approx(want, abs=3e-16)
    assert details['omitted_zero_mass_cells'] == 4
    assert details['positive_mass_minimum'] < 1e-10
    assert_allclose(mass, p.sum(axis=2).ravel()[p.sum(axis=2).ravel() > 0], atol=0)


def test_closed_entropy_boundary_and_rare_supported_context_are_not_smoothed():
    p = np.zeros((2, 3, 4))
    p[0, 0, 0] = p[1, 0, 1] = (.5-1e-12)
    p[0, 1, 2] = p[1, 1, 3] = 1e-12
    q = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 1., 0.]])
    mass, a, b, details = module().normalized_entropy_coefficients(p)
    assert len(mass) == 4 and details['omitted_zero_mass_cells'] == 2
    assert mass.min() == 1e-12
    assert np.count_nonzero(a) == 4
    assert np.all((a @ q)[:, 2] == 0) and np.all((b @ q)[:, 2] == 0)
    value = np.sum(mass[:, None] * rel_entr(a @ q, b @ q))
    assert np.isfinite(value)
    assert value == pytest.approx(finite.cmi(p, q), abs=1e-15)


def test_binary_optimum_one_attempt_and_fixed_solver_settings():
    p, cost = binary()
    witness = np.full((2, 2), .5)
    result = module().solve_normalized_retry(cost, {'A/SEX/U': p}, .04, embedded_q=witness)
    error = brentq(lambda r: np.log(2)+r*np.log(r)+(1-r)*np.log1p(-r)-.04, 1e-9, .5)
    optimum = -(1-error)*np.log(.9)-error*np.log(.1)
    assert result['accepted'] and result['feasible'] and result['optimal']
    assert result['status'] == 'optimal'
    assert result['objective'] == pytest.approx(optimum, abs=2e-6)
    assert len(result['attempts']) == 1
    assert result['attempts'][0]['solver'] == 'CLARABEL'
    assert result['solver_settings'] == finite.SOLVER_SETTINGS['CLARABEL']
    assert result['tolerances'] == finite.ACCEPTANCE_TOLERANCES
    assert result['diagnostics']['witness_objective_constraint_retained'] is True
    assert result['residuals']['witness_objective_excess'] <= 1e-7
    json.dumps({k: v for k, v in result.items() if k != 'Q'}, allow_nan=False)


def test_all_local_coalition_and_weighted_laws_are_kept():
    p, cost = binary()
    local = np.full((2, 1, 2), .25)
    coalition = np.zeros((2, 3, 2))
    for s in range(2):
        for c in range(2):
            coalition[s, c, s ^ c] = .25
    roles = {'A/SEX/U': local, 'A/SEX/PWGTP': p,
             'AB/SEX/U': coalition, 'AB/SEX/PWGTP': coalition}
    result = module().solve_normalized_retry(cost, roles, .04)
    assert result['feasible'], result
    assert set(result['residuals']['cmi']) == set(roles)
    assert set(result['diagnostics']['roles']) == set(roles)
    for name, law in roles.items():
        assert result['residuals']['cmi'][name] == pytest.approx(finite.cmi(law, result['Q']), abs=1e-14)
        assert finite.cmi(law, result['Q']) <= .04 + 1e-7


def test_retry_matches_original_problem_with_rare_contexts_and_witness_bound():
    rng = np.random.default_rng(422)
    law = rng.uniform(size=(2, 3, 4))
    law[:, 0] *= 1e-5
    law[:, 2] = 0
    law /= law.sum()
    mass = law.sum(axis=(0, 1))
    cost = mass[:, None] * np.array([[.1, 1.2], [1.4, .2], [.3, 1.3], [1.1, .1]])
    witness = np.full((4, 2), .5)
    original = finite.solve(cost, {'A/SEX/U': law}, .001, embedded_q=witness)
    recovered = module().solve_normalized_retry(cost, {'A/SEX/U': law}, .001, embedded_q=witness)
    assert original['feasible'] and recovered['feasible']
    assert recovered['objective'] == pytest.approx(original['objective'], abs=2e-6)
    assert recovered['residuals']['cmi']['A/SEX/U'] <= .001 + finite.ACCEPTANCE_TOLERANCES['cmi']


def test_support_ties_absent_parents_and_coarse_witness_are_unchanged():
    p = np.zeros((2, 1, 5))
    p[:, 0, :2] = [[.2, .3], [.2, .3]]
    cost = np.array([[0., .8], [1.2, 0.], [0., 0.], [0., 0.], [0., 0.]])
    parent = np.array([0, 0, 0, 1, 1])
    mass = np.array([2., 3., 0., 0., 0.])
    coarse = np.array([[.5, .5], [0., 1.]])
    originals = [a.copy() for a in (p, cost, parent, mass, coarse)]
    result = module().solve_normalized_retry(cost, {'A/SEX/U': p}, .04,
        parent=parent, state_mass=mass, zero_action=1, embedded_q=coarse)
    assert result['feasible'], result
    assert_allclose(result['Q'][2], .4*result['Q'][0]+.6*result['Q'][1], atol=1e-14)
    assert_allclose(result['Q'][3:], [[0., 1.], [0., 1.]], atol=0)
    for actual, original in zip((p, cost, parent, mass, coarse), originals):
        assert_allclose(actual, original, atol=0)


def test_solver_failure_makes_one_call_and_does_not_return_witness(monkeypatch):
    p, cost = binary()
    calls = []
    def fail(problem, **options):
        calls.append(options)
        raise cp.error.SolverError('synthetic forced failure')
    monkeypatch.setattr(cp.Problem, 'solve', fail)
    result = module().solve_normalized_retry(cost, {'A/SEX/U': p}, .04, embedded_q=np.full((2, 2), .5))
    assert len(calls) == 1 and calls[0]['solver'] == 'CLARABEL'
    assert not result['accepted'] and not result['feasible'] and not result['optimal']
    assert result['Q'] is None and result['status'] == 'failed'
    assert result['attempts'][0]['status'] == 'solver_error'
    assert result['witness']['feasible']
    json.dumps({k: v for k, v in result.items() if k != 'Q'}, allow_nan=False)


def test_feasible_inaccurate_status_never_claims_optimality(monkeypatch):
    original_solve = cp.Problem.solve
    def inaccurate(problem, **options):
        answer = original_solve(problem, **options)
        problem._status = 'optimal_inaccurate'
        return answer
    monkeypatch.setattr(cp.Problem, 'solve', inaccurate)
    p, cost = binary()
    result = module().solve_normalized_retry(cost, {'A/SEX/U': p}, .04)
    assert result['accepted'] and result['feasible']
    assert not result['optimal'] and result['status'] == 'optimal_inaccurate'
    assert len(result['attempts']) == 1


def test_independent_check_rejects_false_solver_privacy_claim(monkeypatch):
    def false_optimal(problem, **options):
        problem.variables()[0].value = np.eye(2)
        problem._status = 'optimal'
        return 0.
    monkeypatch.setattr(cp.Problem, 'solve', false_optimal)
    p, cost = binary()
    result = module().solve_normalized_retry(cost, {'A/SEX/U': p}, .04)
    assert not result['accepted'] and not result['optimal'] and result['Q'] is None
    assert result['attempts'][0]['residuals']['cmi_excess']['A/SEX/U'] > .6


def test_independent_witness_objective_check_remains_required(monkeypatch):
    def false_optimal(problem, **options):
        problem.variables()[0].value = np.full((2, 2), .5)
        problem._status = 'optimal'
        return 0.
    monkeypatch.setattr(cp.Problem, 'solve', false_optimal)
    law = np.full((2, 1, 2), .25)
    cost = np.array([[0., 1.], [1., 0.]])
    result = module().solve_normalized_retry(cost, {'A/SEX/U': law}, .04, embedded_q=np.eye(2))
    assert not result['accepted'] and result['Q'] is None
    assert result['attempts'][0]['residuals']['feasible']
    assert result['attempts'][0]['residuals']['witness_objective_excess'] == 1.


@pytest.mark.parametrize('budget', [None, 0., -.1, np.nan, np.inf])
def test_retry_rejects_nonpositive_or_nonfinite_budget(budget):
    p, cost = binary()
    with pytest.raises(ValueError, match='positive'):
        module().solve_normalized_retry(cost, {'A/SEX/U': p}, budget)


def test_retry_rejects_invalid_witness_and_empty_roles():
    p, cost = binary()
    with pytest.raises(ValueError, match='witness'):
        module().solve_normalized_retry(cost, {'A/SEX/U': p}, .04, embedded_q=np.eye(2))
    with pytest.raises(ValueError, match='roles'):
        module().solve_normalized_retry(cost, {}, .04)
