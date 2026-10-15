"""Unit tests for the independent core (synthetic data only; no person-level data)."""
import numpy as np
import pytest

from task_headroom_independent import core


def test_auc_matches_bruteforce_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + 0.7 * y, 1)  # many ties
    w = rng.integers(0, 4, 300).astype(float)
    assert core.auc(y, s) == pytest.approx(core.auc_bruteforce(y, s), abs=1e-12)
    assert core.auc_weighted(y, s, w[None, :])[0] == pytest.approx(core.auc_bruteforce(y, s, w), abs=1e-12)


def test_auc_extremes_and_constant():
    y = np.array([0, 0, 1, 1])
    assert core.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert core.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert core.auc(y, np.full(4, 0.5)) == 0.5


def test_sklearn_auc_agreement():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500); s = rng.random(500) + 0.2 * y
    assert core.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_bootstrap_multiplicities_groupwise():
    unit = np.array([5, 5, 7, 9, 9, 9, 11])
    W, G = core.group_multiplicities(unit, 50, 3)
    assert G == 4 and W.shape == (50, 7)
    # rows of one group share weight; per draw groups drawn sum to G
    assert np.all(W[:, 0] == W[:, 1]) and np.all(W[:, 3] == W[:, 5])
    assert np.all(W[:, [0, 2, 3, 6]].sum(1) == 4)


def test_bootstrap_auc_draws_center():
    rng = np.random.default_rng(2)
    n = 400; y = rng.integers(0, 2, n); s = rng.normal(size=n) + y; unit = np.arange(n) // 2
    W, G = core.group_multiplicities(unit, 400, 9)
    d = core.auc_weighted(y, s, W)
    assert abs(d.mean() - core.auc(y, s)) < 0.01


def test_intervals_and_status():
    crit = core.t_bonferroni(1, 1000)
    assert crit == pytest.approx(1.962, abs=1e-3)
    iv = core.sim_interval(0.0, np.array([-1.0, 1.0]), 1.0)
    assert iv['se'] == pytest.approx(np.sqrt(2)) and iv['lower'] == pytest.approx(-np.sqrt(2))
    assert core.status_cap({'lower': -1, 'upper': 0.01}, 0.01) == 'PASS'
    assert core.status_cap({'lower': 0.011, 'upper': 0.02}, 0.01) == 'ESTABLISHED_VIOLATION'
    assert core.status_cap({'lower': 0.0, 'upper': 0.02}, 0.01) == 'UNRESOLVED'
    assert core.status_margin({'lower': 0.03, 'upper': 0.04}, 0.02) == 'ESTABLISHED'
    assert core.status_margin({'lower': -0.01, 'upper': 0.019}, 0.02) == 'ESTABLISHED_SHORTFALL'
    assert core.status_margin({'lower': 0.0, 'upper': 0.03}, 0.02) == 'UNRESOLVED'


def _payload(task='occupation'):
    dim = 1 if task == 'income' else 6
    rng = np.random.default_rng(4)
    T, depth = 5, 2
    feats = rng.integers(0, 83, (T, depth)).tolist()
    borders = rng.normal(size=(T, depth)).astype(np.float32)
    borders[4, 1] = np.finfo(np.float32).max  # padding split
    return {'task': task, 'depth': depth, 'split_features': feats, 'split_borders': borders.tolist(),
            'leaf_values': rng.normal(size=(T, 4, dim)).tolist(), 'scale': 1.0, 'bias': [0.1] * dim,
            'feature_names': [f'f{i}' for i in range(83)]}


def test_router_bits_and_padding():
    pl = _payload()
    t = core.Teacher(pl)
    X = np.random.default_rng(5).normal(size=(50, 83)).astype(np.float32)
    L = t.leaves(X)
    for r in range(50):
        for k in range(t.T):
            expect = sum((1 << i) for i in range(t.depth) if X[r, pl['split_features'][k][i]] > np.float32(pl['split_borders'][k][i]))
            assert L[r, k] == expect
    assert np.all(L[:, 4] < 2)  # padding bit never set


def test_router_float32_strict_border():
    pl = _payload('income')
    pl['split_features'] = [[0, 1]] * 5
    pl['split_borders'] = [[0.5, np.finfo(np.float32).max]] * 5
    t = core.Teacher(pl)
    X = np.zeros((3, 83), dtype=np.float32); X[0, 0] = 0.5; X[1, 0] = np.nextafter(np.float32(0.5), np.float32(1))
    assert t.leaves(X)[:, 0].tolist() == [0, 1, 0]


def test_zero_theta_identity_and_correction():
    for task in ('income', 'occupation'):
        t = core.Teacher(_payload(task))
        X = np.random.default_rng(6).normal(size=(40, 83)).astype(np.float32)
        L = t.leaves(X); raw = t.teacher_raw(L)
        z = t.correction(L, np.zeros((t.T * t.L, t.dim)))
        assert np.array_equal(raw + z, raw)
        th = np.random.default_rng(7).normal(size=(t.T * t.L, t.dim))
        brute = np.stack([th[[k * t.L + L[r, k] for k in range(t.T)]].sum(0) / t.T for r in range(40)])
        assert np.allclose(t.correction(L, th), brute, atol=1e-14, rtol=0)


def test_release_conventions():
    p, d = core.release('income', np.array([[0.0], [2.0], [-1.0]]))
    assert np.allclose(p[0], [0.5, 0.5]) and p[1, 1] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert d.tolist() == [0, 1, 0]  # tie -> first class
    raw = np.array([[1.0, 2, 3, 4, 5, 6]])
    p2, d2 = core.release('occupation', raw)
    e = np.exp(raw - raw.max()); assert np.allclose(p2, e / e.sum()) and d2[0] == 5


def test_metrics_and_clauses():
    y = np.array([0, 1, 1])
    p = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    m = core.means(y, p)
    assert m['accuracy'] == pytest.approx(2 / 3)
    assert m['nll'] == pytest.approx(-(np.log(0.8) + np.log(0.6) + np.log(0.3)) / 3)
    assert m['brier'] == pytest.approx((2 * 0.04 + 2 * 0.16 + 2 * 0.49) / 3)
    c = core.clause_values(y, p, p)
    assert all(v == 0 for v in c.values())


def test_proxy_fold_and_att_split_deterministic():
    f = core.proxy_fold(np.arange(1000))
    assert set(f.tolist()) == {0, 1} and 400 < f.sum() < 600
    unit = np.repeat(np.arange(2000), 2)
    tr, va = core.att_split(unit)
    assert len(np.unique(unit[va])) == 1015 and not set(unit[tr]) & set(unit[va])


def test_nomination_replay_rules():
    def row(feas, prox, nll, viol=0.5):
        return {'internal_point_feasible': feas, 'proxy': {'pair': {'value': prox}}, 'summed_nll': nll, 'max_normalized_violation': viol}
    tabs = {0: {}, 1: {}}
    for rho in ('1.0', '3.0', '10.0'):
        for lab in core.candidate_labels('J_TREE'):
            for s in (0, 1):
                tabs[s][f'J_TREE|{rho}|{lab}'] = row(False, 0.6, 1.0, viol=2.0)
    for s in (0, 1):
        tabs[s]['J_TREE|3.0|E5'] = row(True, 0.60, 1.0)
        tabs[s]['J_TREE|1.0|E10'] = row(True, 0.60, 1.0)
    st, best, _ = core.replay_family(tabs, 'J_TREE')
    assert st == 'POINT_FEASIBLE_NOMINEE' and best['key'] == 'J_TREE|1.0|E10'  # tie -> lower strength
    tabs[1]['J_TREE|1.0|E10'] = row(True, 0.65, 1.0)  # worst seed proxy higher
    assert core.replay_family(tabs, 'J_TREE')[1]['key'] == 'J_TREE|3.0|E5'
    assert core.family_size(2, 'J_TREE') == 42 and core.family_size(2, 'L_TREE') == 66
