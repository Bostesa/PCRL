"""Tests of the scientific hazards named in the protocol (synthetic fixtures only)."""
import hashlib
import hmac

import numpy as np
import pytest

from experiments.pcrl_final_prospective_v1 import inference_panel as ip
from experiments.pcrl_final_prospective_v1.audit_panel import dependencies, all_units, roles_for, role_seed
from experiments.pcrl_final_prospective_v1.common import PANEL, CANDIDATES
from experiments.pcrl_task_directed_release_v1.audits import expected_token_loss, TokenCandidate
from experiments.pcrl_task_directed_release_v1.evaluation import _wire
from experiments.pcrl_task_directed_release_v1.uncertainty import paired_household_bounds


def _bounds(values):
    return {e['id']: {'estimate': v[0], 'bootstrap_se': v[1]} for e, v in values}


def test_one_failing_clause_fails_conjunction_despite_nine_passes():
    prim = ip.primary_endpoints()
    vals = []
    for e in prim:
        good = (-.01, .001) if e['clause'] == 'task' else (-.01, .001)
        vals.append((e, good))
    q_priv = [i for i, (e, _) in enumerate(vals) if e['candidate'] == 'Q' and e['clause'] == 'privacy']
    vals[q_priv[0]] = (vals[q_priv[0]][0], (.0, .002))   # UB = .0039 > .001
    d = ip.decide(prim, _bounds(vals))
    assert d['Q']['clauses_passed'] == 9 and d['Q']['decision'] == 'fail'
    assert d['D17']['clauses_passed'] == 10 and d['D17']['decision'] == 'pass'


def test_task_clause_requires_material_gain_and_sign_convention():
    prim = ip.primary_endpoints()
    e = [x for x in prim if x['candidate'] == 'Q' and x['clause'] == 'task'][0]
    z = ip.z_primary()
    assert abs(z-1.959964) < 1e-5
    # estimate -.004 with SE .0001: UB = -.0038 <= -.003 passes; estimate -.002 fails.
    assert ip.clause_passes(e, ip.upper_bound(-.004, .0001, z))
    assert not ip.clause_passes(e, ip.upper_bound(-.002, .0001, z))


def test_zero_variance_bound_equals_estimate():
    assert ip.upper_bound(.0005, 0., 1.96) == .0005


def test_missing_measurement_is_failure():
    prim = ip.primary_endpoints()
    vals = [(e, (-.01, .001)) for e in prim if not (e['candidate'] == 'Q' and e['clause'] == 'task')]
    d = ip.decide(prim, _bounds(vals))
    assert d['Q']['decision'] == 'fail'
    assert any(c['status'] == 'invalid_or_missing_measurement' for c in d['Q']['clauses'])


def test_status_distinguishes_unresolved_from_adverse():
    prim = [e for e in ip.primary_endpoints() if e['candidate'] == 'Q']
    vals = [(e, (-.01, .0001)) for e in prim]
    priv = [i for i, (e, _) in enumerate(vals) if e['clause'] == 'privacy']
    vals[priv[0]] = (vals[priv[0]][0], (-.001, .002))   # unresolved
    vals[priv[1]] = (vals[priv[1]][0], (.01, .001))     # demonstrated adverse vs J
    d = ip.decide(prim, _bounds(vals))['Q']['clauses']
    statuses = {c['id']: c['status'] for c in d}
    assert statuses[vals[priv[0]][0]['id']] == 'unresolved'
    assert statuses[vals[priv[1]][0]['id']] == 'failed_demonstrated_adverse_vs_J'


def test_secondary_family_generated_not_hand_counted():
    sec = ip.secondary_endpoints()
    assert len(sec) == len(ip.SECONDARY_COMPARATORS)*(2+4*2) == 70
    assert len({e['id'] for e in sec}) == 70
    from scipy.stats import norm
    assert abs(ip.z_secondary(70)-norm.ppf(1-.05/140)) < 1e-12 and 3.38 < ip.z_secondary(70) < 3.39


def test_primary_family_counts():
    prim = ip.primary_endpoints()
    assert len(prim) == 20 and len({e['id'] for e in prim}) == 20
    assert set(CANDIDATES) == {'Q', 'D17'}


def _anchor(h, d, w):
    return {'household': np.asarray(h), 'difference': np.asarray(d, float), 'weights': np.asarray(w, float)}


def test_exact_aliases_have_zero_variance_and_estimate():
    h = [f'h{i//2}' for i in range(40)]
    c = [{'id': 'alias', 'anchors': [_anchor(h, np.zeros(40), np.ones(40))]*3}]
    r = paired_household_bounds(c, n_boot=50, seed=1)
    assert r['bounds']['alias']['estimate'] == 0 and r['bounds']['alias']['bootstrap_se'] == 0


def test_replicated_person_does_not_increase_household_count():
    rng = np.random.default_rng(0)
    h = [f'h{i}' for i in range(30)]
    d = rng.normal(size=30)
    base = paired_household_bounds([{'id': 'a', 'anchors': [_anchor(h, d, np.ones(30))]*3}], n_boot=20, seed=2)
    h2, d2 = h+['h0'], np.r_[d, d[0]]
    dup = paired_household_bounds([{'id': 'a', 'anchors': [_anchor(h2, d2, np.ones(31))]*3}], n_boot=20, seed=2)
    assert base['households_union'] == dup['households_union'] == 30


def test_weighted_ratio_not_mean_of_weighted_losses():
    h = ['a', 'b', 'c']
    d, w = np.array([1., 0., 0.]), np.array([10., 1., 1.])
    r = paired_household_bounds([{'id': 'x', 'anchors': [_anchor(h, d, w)]*3}], n_boot=10, seed=3)
    assert abs(r['bounds']['x']['estimate']-10/12) < 1e-12


def test_expected_loss_is_not_loss_of_mean_probability():
    q = np.array([[[.9, .1], [.1, .9]]])       # person, token, class
    p = np.array([[.5, .5]])
    y = np.array([0])
    exact = expected_token_loss(q, p, y)[0]
    assert abs(exact-(.5*-np.log(.9)+.5*-np.log(.1))) < 1e-12
    assert abs(exact-(-np.log(.5))) > .1


def test_attacker_sees_one_hot_token_not_q_row():
    h = np.zeros((3, 2))
    rel = {'aux': None, 'token_probs': np.array([[.2, .8], [1., 0.], [.5, .5]])}
    data = {'ha': h, 'hb': np.zeros((3, 2))}
    x, p = _wire(data, rel, 'A', 'release')
    assert x.shape == (3, 2)             # the Q row is not a feature
    assert np.array_equal(p, rel['token_probs'])
    cand = TokenCandidate('prior', np.array([.5, .5]), np.zeros(2), np.ones(2), 2, 2, False,
                          {'class_order': [0, 1]})
    q = cand.predict_token_proba(x, 2)
    assert q.shape == (3, 2, 2)


def test_h_parity_columns_first_in_every_view():
    ha, hb = np.arange(8.).reshape(2, 4), np.arange(4.).reshape(2, 2)
    rel = {'aux': np.ones((2, 3)), 'token_probs': np.ones((2, 1))}
    xa, _ = _wire({'ha': ha, 'hb': hb}, rel, 'A', 'release')
    xab, _ = _wire({'ha': ha, 'hb': hb}, rel, 'AB', 'release')
    assert xa[:, :4].tobytes() == ha.tobytes()
    assert xab[:, :4].tobytes() == ha.tobytes() and xab[:, 4:6].tobytes() == hb.tobytes()


def test_b_view_never_reads_release():
    rel = {'aux': np.ones((2, 3)), 'token_probs': np.full((2, 2), .5)}
    x, p = _wire({'ha': np.zeros((2, 4)), 'hb': np.ones((2, 2))}, rel, 'B', 'release')
    assert x.shape == (2, 2) and p.shape == (2, 1)


def test_ancestor_routing_and_no_j_ancestor():
    assert dependencies('Q', 0, 'attack:A/SEX') == [('H', 0, 'attack:A/SEX')]
    assert set(dependencies('Q', 1, 'attack:AB/RAC1P')) == {('H', 1, 'attack:AB/RAC1P'), ('Q', 1, 'attack:A/RAC1P'),
                                                          ('H', 1, 'attack:B/RAC1P')}
    assert all(dep[0] in ('H', u[0]) for u in all_units() for dep in dependencies(*u))
    assert len(all_units()) == 7*3+9*5*3 == 156


def test_constant_extension_ancestor_route_equals_h():
    """A constant auxiliary column adds no information; the H-routed ancestor ignores it exactly."""
    rng = np.random.default_rng(4)
    ha = rng.random((50, 4)); hb = rng.random((50, 2)); y = rng.integers(0, 2, 50)
    rel = {'aux': np.ones((50, 1)), 'token_probs': np.ones((50, 1))}
    x_h, _ = _wire({'ha': ha, 'hb': hb}, rel, 'A', 'H')
    assert np.array_equal(x_h, ha)
    cand = TokenCandidate('prior', np.array([.3, .7]), np.zeros(4), np.ones(4), 1, 2, False, {'class_order': [0, 1]})
    l_h = expected_token_loss(cand.predict_token_proba(x_h, 1), np.ones((50, 1)), y)
    l_h2 = expected_token_loss(cand.predict_token_proba(ha, 1), np.ones((50, 1)), y)
    assert np.array_equal(l_h, l_h2)


def test_seeds_common_across_releases():
    assert role_seed(0, 'attack:A/SEX') == role_seed(0, 'attack:A/SEX')
    assert role_seed(0, 'attack:A/SEX') != role_seed(1, 'attack:A/SEX')


def persistent_token(secret, person_id, row):
    u = int.from_bytes(hmac.new(secret, person_id.encode(), hashlib.sha256).digest(), 'big')/2**256
    c = np.cumsum(row); c[-1] = 1.
    return int(np.sum(u >= c))


def test_one_token_contract_repeat_requests_identical_and_mc_converges():
    row = np.array([.1, .6, .3])
    assert persistent_token(b's', 'p1', row) == persistent_token(b's', 'p1', row)
    losses = np.array([1., 2., 5.])
    exact = float(row @ losses)
    draws = [losses[persistent_token(f'k{k}'.encode(), 'p1', row)] for k in range(4000)]
    assert abs(np.mean(draws)-exact) < 4*np.std(draws)/np.sqrt(len(draws))


def test_final_labels_sealed_without_lock(tmp_path, monkeypatch):
    from experiments.pcrl_final_prospective_v1 import transport
    monkeypatch.setattr(transport, 'LOCK', tmp_path/'EVALUATION_LOCK.json')
    with pytest.raises(PermissionError):
        transport.read_labels(np.array([0, 1]), pool='final_evaluation')


def test_class_order_safety_absent_fit_class_keeps_full_schema():
    """A logistic fit that never saw classes 3..8 still predicts on the full 9-class schema."""
    from sklearn.linear_model import LogisticRegression
    x = np.random.default_rng(5).random((30, 2)); y = np.array([0, 1, 2]*10)
    m = LogisticRegression().fit(x, y)
    cand = TokenCandidate('logistic', m, np.zeros(2), np.ones(2), 1, 9, False, {'class_order': list(range(9))})
    q = cand.predict_token_proba(x, 1)
    assert q.shape == (30, 1, 9) and np.allclose(q.sum(2), 1) and (q[:, :, 3:] > 0).all()
    loss = expected_token_loss(q, np.ones((30, 1)), np.full(30, 8))
    assert np.isfinite(loss).all() and (loss > 15).all()
