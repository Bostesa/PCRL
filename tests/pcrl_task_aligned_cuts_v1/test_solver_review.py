"""Independent fixed-bank signs, feasibility, and solver-bound counterexamples."""
import numpy as np
import pytest
import json

from experiments.pcrl_task_aligned_cuts_v1.audit import expected_token_loss, score_weightings
from experiments.pcrl_task_aligned_cuts_v1.controls import solve_deterministic_p1
from experiments.pcrl_task_aligned_cuts_v1.method import coefficient_pair, score_expected
from experiments.pcrl_task_aligned_cuts_v1.solver import add_violated_cuts, phase_one_p1, relax_cuts, replay_p1, solve_p1


def test_coefficient_dot_matches_independent_expected_token_enumeration():
    rng = np.random.default_rng(18)
    for people, tokens, classes, states in ((3, 2, 2, 2), (11, 17, 9, 32)):
        t = rng.integers(0, states, people)
        y = rng.integers(0, classes, people)
        q = rng.dirichlet(np.ones(tokens), size=states)
        predictor = rng.dirichlet(np.ones(classes), size=(people, tokens))
        weights = rng.uniform(.1, 3, people)
        coefficients = coefficient_pair(t, y, predictor, states, weights)
        independent = score_weightings(expected_token_loss(predictor, q[t], y), weights)
        assert np.sum(coefficients['U'] * q) == pytest.approx(independent['U'], abs=1e-14)
        assert np.sum(coefficients['W'] * q) == pytest.approx(independent['PWGTP'], abs=1e-14)
        direct = score_expected(q, t, y, predictor, weights)
        assert direct['U'] == pytest.approx(independent['U'], abs=1e-14)
        assert direct['W'] == pytest.approx(independent['PWGTP'], abs=1e-14)


def test_fixed_bank_has_real_stochastic_deterministic_gap_and_correct_sign():
    cost = np.array([[0., 1.]])
    cut = {'id': 'risk-floor', 'coeff': np.array([[0., 1.]]),
           'rho': .5, 'delta': 0., 'floor': .5}
    stochastic = solve_p1(cost, [cut])
    deterministic = solve_deterministic_p1(cost, [cut], time_limit_seconds=5)
    assert stochastic['feasible']
    assert stochastic['objective'] == pytest.approx(.5, abs=1e-8)
    assert stochastic['dual_lower_bound'] == pytest.approx(.5, abs=1e-7)
    assert deterministic['incumbent_valid']
    assert deterministic['incumbent_objective'] == pytest.approx(1., abs=1e-8)
    assert deterministic['lower_bound'] == pytest.approx(1., abs=1e-8)
    assert replay_p1(stochastic['Q'], cost, [cut])['maximum_cut_violation'] < 1e-8
    # Positive delta relaxes the sensitive-loss floor and permits more recovery.
    relaxed = dict(cut, delta=.25, floor=.25)
    assert solve_p1(cost, [relaxed])['objective'] == pytest.approx(.25, abs=1e-8)


def test_phase_one_preserves_infeasible_registered_bank():
    cost = np.array([[0., 1.]])
    cuts = [{'id': 'a', 'coeff': np.array([[1., 0.]]), 'floor': .75},
            {'id': 'b', 'coeff': np.array([[0., 1.]]), 'floor': .75}]
    phase = phase_one_p1(cost, cuts)
    assert phase['minimum_common_violation'] == pytest.approx(.25, abs=1e-8)
    original = solve_p1(cost, cuts)
    assert original['status'] == 'registered_bank_infeasible'
    assert original['Q'] is None
    fallback = solve_p1(cost, relax_cuts(cuts, .25 + 1e-7, cost.shape))
    assert fallback['feasible']


def test_reference_selection_considers_legal_coalition_ancestors():
    from experiments.pcrl_task_aligned_cuts_v1.fit import _selected_reference
    def registry(score):
        return {'own_selection': 'winner', 'models': {'winner': {}},
                'own_scores': {'winner': {'balanced': score, 'unweighted': score, 'weighted': score}}}
    routes = [('AB', 'H', registry(.5)), ('AB', 'D17', registry(.4)),
              ('A', 'D17', registry(.3)), ('B', 'H', registry(.35))]
    view, source, cid, chosen = _selected_reference(routes, 'unweighted')
    assert (view, source, cid) == ('A', 'D17', 'winner')
    assert chosen is routes[2][2]


def test_reference_selection_is_per_weighting_and_uses_all_fitted_candidates():
    from experiments.pcrl_task_aligned_cuts_v1.fit import _selected_reference
    h = {'own_selection': 'balanced_h', 'models': {'balanced_h': {}, 'best_u': {}}, 'own_scores': {
        'balanced_h': {'balanced': .45, 'unweighted': .40, 'weighted': .50},
        'best_u': {'balanced': .46, 'unweighted': .20, 'weighted': .72}}}
    d17 = {'own_selection': 'balanced_d', 'models': {'balanced_d': {}, 'best_w': {}}, 'own_scores': {
        'balanced_d': {'balanced': .41, 'unweighted': .50, 'weighted': .32},
        'best_w': {'balanced': .42, 'unweighted': .70, 'weighted': .14}}}
    routes = [('AB', 'H', h), ('A', 'D17', d17)]
    assert _selected_reference(routes, 'unweighted')[:3] == ('AB', 'H', 'best_u')
    assert _selected_reference(routes, 'weighted')[:3] == ('A', 'D17', 'best_w')


def test_coalition_ancestor_coefficients_use_only_their_released_service_view(monkeypatch):
    from experiments.pcrl_task_aligned_cuts_v1.fit import _candidate_coefficients
    from experiments.pcrl_task_directed_release_v1 import audits as inherited
    seen = []
    class Predictor:
        n_classes = 2
        n_tokens = 17
        def predict_token_proba(self, h, n_tokens):
            seen.append((h.shape[1], n_tokens))
            return np.full((len(h), n_tokens, 2), .5)
    monkeypatch.setattr(inherited, 'load_candidate', lambda directory: Predictor())
    rows = {'ha': np.zeros((3, 4)), 'hb': np.ones((3, 2)),
            'labels': {'SEX': np.array([0, 1, 0])}, 'weights': np.ones(3)}
    t = np.array([0, 1, 2])
    mask = np.ones(3, dtype=bool)
    for view, expected in (('A', 4), ('AB', 6)):
        pair = _candidate_coefficients('/synthetic', 'D17', view, 'AB/SEX', rows, t, mask)
        assert seen[-1] == (expected, 17)
        assert pair['U'].shape == (32, 17)
        assert np.sum(pair['U'] * np.full((32, 17), 1/17)) == pytest.approx(np.log(2))
    Predictor.n_tokens = 1
    pair = _candidate_coefficients('/synthetic', 'H', 'B', 'AB/SEX', rows, t, mask)
    assert seen[-1] == (2, 1)
    assert np.sum(pair['U'] * np.full((32, 17), 1/17)) == pytest.approx(np.log(2))


def test_saved_p1_bank_round_trip_replays_identical_cuts_and_rejects_tamper(tmp_path, monkeypatch):
    from experiments.pcrl_task_aligned_cuts_v1 import fit, data, solver
    bank_dir = tmp_path / 'private' / 'bank'
    bank_dir.mkdir(parents=True)
    cost = np.zeros((32, 17))
    cost[0, 1] = 1.
    coefficient = np.zeros_like(cost)
    coefficient[0, 1] = 1.
    cut = {'id': 'synthetic/A/SEX/U', 'coeff': coefficient,
           'rho': .5, 'delta': 0., 'floor': .5}
    bank_hash = solver.bank_sha256([cut], cost.shape)
    archive = bank_dir / 'coefficients.npz'
    np.savez_compressed(archive, cost_U=cost, cost_W=cost, cost=cost, cut_0000=coefficient)
    (bank_dir / 'bank.json').write_text(json.dumps({
        'bank_sha256': bank_hash,
        'cuts': [{k: v for k, v in cut.items() if k != 'coeff'}],
        'coefficient_archive_sha256': data.sha256_file(archive),
    }))
    (bank_dir / 'FIT_P1_CENTER.json').write_text(json.dumps({
        'anchor': 0, 'arm': 'U0P1', 'bank_sha256': bank_hash,
        'selection_household_assignment_sha256': 'synthetic-split',
        'attack_slate': 'standard', 'source_channels': ['H', 'D17', 'Q', 'coverage'],
        'bank_complete': True, 'role_support': {},
    }))
    monkeypatch.setattr(fit.data, 'index', lambda path: {})
    monkeypatch.setattr(fit, '_selection_rows',
                        lambda value, anchor, mask: ({}, np.array([0]), 'synthetic-split'))
    monkeypatch.setattr(fit.data, 'member_record',
                        lambda value, anchor, kind: {'sha256': '0'*64})
    record = fit.solve_from_bank(0, 'synthetic-index', bank_dir,
                                 tmp_path / 'private' / 'arm', arm='U0P1', delta=0.)
    assert record['status'] == 'INITIAL_BANK_FEASIBLE'
    assert record['frozen_source_bank_sha256'] == bank_hash
    assert record['solution']['objective'] == pytest.approx(.5, abs=1e-8)
    assert record['solution']['replay']['maximum_cut_violation'] < 1e-8
    with np.load(archive, allow_pickle=False) as saved:
        changed = {key: saved[key].copy() for key in saved.files}
    changed['cut_0000'][0, 1] += .1
    np.savez_compressed(archive, **changed)
    with pytest.raises(ValueError, match='hash mismatch'):
        fit.solve_from_bank(0, 'synthetic-index', bank_dir,
                            tmp_path / 'private' / 'second', arm='U0P1', delta=0.)


def test_exchange_union_retains_old_and_both_lp_milp_specific_violations():
    shape = (1, 2)
    old = {'id': 'old', 'coeff': np.array([[1., 1.]]), 'floor': .5}
    proposals = [
        {'id': 'lp-only', 'coeff': np.array([[1., 0.]]), 'floor': .6},
        {'id': 'milp-only', 'coeff': np.array([[0., 1.]]), 'floor': .4},
        {'id': 'not-violated', 'coeff': np.array([[1., 1.]]), 'floor': .9},
    ]
    channels = {'LP': np.array([[.5, .5]]), 'MILP': np.array([[1., 0.]])}
    result = add_violated_cuts([old], proposals, channels, shape, tolerance=1e-5)
    assert [cut['id'] for cut in result['cuts']] == ['old', 'lp-only', 'milp-only']
    assert result['added_ids'] == ['lp-only', 'milp-only']
    log = {row['id']: row for row in result['oracle_log']}
    assert log['lp-only']['violations']['LP'] == pytest.approx(.1)
    assert log['lp-only']['violations']['MILP'] < 0
    assert log['milp-only']['violations']['MILP'] == pytest.approx(.4)
    assert log['milp-only']['violations']['LP'] < 0
    assert not log['not-violated']['added']
    with pytest.raises(ValueError, match='same cut ID'):
        add_violated_cuts([old], [{'id': 'old', 'coeff': np.array([[0., 1.]]), 'floor': .5}],
                          channels, shape)
