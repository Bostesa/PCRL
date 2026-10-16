"""Data-free tests for acs_replication/fagtb.py (synthetic fixtures only; no ACS or Adult data).

Run either way from the worktree root:
    python -m pytest acs_replication/test_fagtb.py -q
    python acs_replication/test_fagtb.py
(a) binary residual = -d/dF [task log-loss - lam * adversary summed CE], finite differences
(b) adversary input gradient and parameter gradient = finite differences; official update order
(c) lam = 0 is ordinary gradient boosting on log-loss with the same trees; predict_proba = cumulative sum
(d) multiclass adaptation: residual = finite differences, K=2 behaviour, probabilities sum to 1
(e) official-environment parity (official class + TensorFlow in an isolated venv); skipped if absent
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # Script mode puts acs_replication/ first on sys.path. That directory holds campaign modules whose names
    # shadow the standard library (queue.py, select.py), which would break `import subprocess`. Drop it.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]

import inspect  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import numpy as np
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(HERE)
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

from acs_replication import fagtb as fg  # noqa: E402

try:
    import pytest

    _skip = pytest.skip
except ImportError:  # plain-python runner
    class _Skip(Exception):
        pass

    def _skip(msg):
        raise _Skip(msg)

PARITY_PY = os.path.join(WORKTREE, ".private/acs_replication/vendor/parity_venv/bin/python")
PARITY_HARNESS = os.path.join(WORKTREE, ".private/acs_replication/vendor/parity/official_parity.py")
PARITY_OUT = os.path.join(WORKTREE, ".private/acs_replication/vendor/parity/parity_result_test.json")


# ------------------------------------------------------------------------------------------------
# fixtures and independent reference implementations (deliberately not using fagtb internals)
# ------------------------------------------------------------------------------------------------

def _fixture(n=240, d=6, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.45, n)
    X = rng.normal(size=(n, d))
    X[:, 0] += s
    logit = 1.1 * X[:, 0] - 0.7 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + 0.5 * s - 0.2
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    return X, y, s


def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # independent of fagtb's 1/(1+exp(-z))


def _softplus(z):
    return np.logaddexp(0.0, z)


def _binary_rows(F, y, s, w, b, lam):
    """Per-row J_i(F_i) = L_F,i - lam * L_A,i (paper eq. 6, 8, 9; demographic-parity adversary input)."""
    a = w * _sig(F) + b
    return (_softplus(F) - y * F) - lam * (_softplus(a) - s * a)


def _mc_rows(F, y, s, w, b, lam):
    """Per-row J_i(F_i) = CE_task,i - lam * CE_adv,i for the multiclass adaptation."""
    lse = np.logaddexp.reduce(F, axis=1)
    P = np.exp(F - lse[:, None])
    a = P @ w + b
    return (lse - F[np.arange(F.shape[0]), y]) - lam * (_softplus(a) - s * a)


def _fd_rows(fun, F, h):
    """Central differences of a row-separable objective: one column of F perturbed at a time."""
    F = np.asarray(F, dtype=np.float64)
    if F.ndim == 1:
        return (fun(F + h) - fun(F - h)) / (2.0 * h)
    G = np.empty_like(F)
    for k in range(F.shape[1]):
        E = np.zeros_like(F)
        E[:, k] = h
        G[:, k] = (fun(F + E) - fun(F - E)) / (2.0 * h)
    return G


def _ref_tree(max_depth, max_features):
    return DecisionTreeRegressor(criterion="friedman_mse", max_depth=max_depth, max_features=max_features,
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                                 min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=0)


# ------------------------------------------------------------------------------------------------
# (a) binary residual = -dJ/dF
# ------------------------------------------------------------------------------------------------

def test_a_binary_residual_matches_finite_difference():
    rng = np.random.default_rng(1)
    n = 500
    F = rng.normal(scale=2.0, size=n)
    y = rng.binomial(1, 0.4, n).astype(np.float64)
    s = rng.binomial(1, 0.5, n).astype(np.float64)
    worst = 0.0
    for (w, b, lam, h, atol) in [(1.7, -0.4, 0.0, 1e-6, 1e-8), (1.7, -0.4, 0.15, 1e-6, 1e-8),
                                  (-2.3, 0.9, 3.0, 1e-6, 1e-8), (40.0, -15.0, 0.15, 1e-6, 1e-6)]:
        u = fg.binary_pseudo_residual(F, y, s, w, b, lam)
        fd = -_fd_rows(lambda Z: _binary_rows(Z, y, s, w, b, lam), F, h)
        np.testing.assert_allclose(u, fd, rtol=1e-7, atol=atol)
        worst = max(worst, float(np.max(np.abs(u - fd))))
        # also against the total (non-separable view): perturb single coordinates of sum_i J_i
        for i in (0, 7, 311):
            e = np.zeros(n)
            e[i] = h
            tot = (np.sum(_binary_rows(F + e, y, s, w, b, lam)) - np.sum(_binary_rows(F - e, y, s, w, b, lam))) / (2 * h)
            assert abs(-tot - u[i]) < 1e-5 * max(1.0, abs(u[i]))
    print(f"   (a) max |u - FD| = {worst:.2e}")


# ------------------------------------------------------------------------------------------------
# (b) adversary gradients and official update order
# ------------------------------------------------------------------------------------------------

def test_b_adversary_input_and_param_gradients_match_finite_difference():
    rng = np.random.default_rng(2)
    n = 400
    p = rng.uniform(0.02, 0.98, n)
    s = rng.binomial(1, 0.5, n).astype(np.float64)
    worst_in, worst_par = 0.0, 0.0
    for w, b in [(0.8, -0.3), (-3.1, 1.2), (25.0, -9.0)]:
        def total(pp, ww=w, bb=b):
            a = ww * pp + bb
            return _softplus(a) - s * a  # per-row CE, summed by the caller

        g = fg.binary_adversary_input_grad(p, s, w, b)
        fd = _fd_rows(total, p, 1e-6)
        np.testing.assert_allclose(g, fd, rtol=1e-7, atol=1e-7)
        worst_in = max(worst_in, float(np.max(np.abs(g - fd))))
        gw, gb = fg.binary_adversary_param_grad(p, s, w, b)
        h = 1e-6
        fw = (np.sum(total(p, w + h, b)) - np.sum(total(p, w - h, b))) / (2 * h)
        fb = (np.sum(total(p, w, b + h)) - np.sum(total(p, w, b - h))) / (2 * h)
        assert abs(gw - fw) < 1e-6 * max(1.0, abs(fw)) and abs(gb - fb) < 1e-6 * max(1.0, abs(fb))
        worst_par = max(worst_par, abs(gw - fw) / max(1.0, abs(fw)), abs(gb - fb) / max(1.0, abs(fb)))
    print(f"   (b) max |g_input - FD| = {worst_in:.2e}; max rel param-grad error = {worst_par:.2e}")


def test_b_fit_uses_official_update_order():
    """Iteration 0 recomputed by hand: adversary step FIRST, gradient with the UPDATED adversary, then tree."""
    X, y, s = _fixture(n=200, d=5, seed=3)
    lam, lr, w0, b0 = 2.0, 0.1, 0.37, -0.21
    m = fg.FAGTBBinary(n_estimators=1, learning_rate=lr, max_features=None, adversary_init=(w0, b0)).fit(X, y, s, lam)
    yf, sf = y.astype(float), s.astype(float)
    F0 = np.log(yf.sum() / (1 - yf).sum())
    p0 = np.full(len(y), 1.0 / (1.0 + np.exp(-F0)))
    r = _sig(w0 * p0 + b0) - sf
    w1, b1 = w0 - 0.01 * np.sum(r * p0), b0 - 0.01 * np.sum(r)  # GD on the SUMMED CE, lr 0.01
    assert abs(m.history_["adversary_w"][0] - w1) < 1e-12 and abs(m.history_["adversary_b"][0] - b1) < 1e-12

    def f1_with(wa, ba):
        u = yf - p0 + lam * (_sig(wa * p0 + ba) - sf) * wa * p0 * (1 - p0)
        return F0 + lr * _ref_tree(9, None).fit(X, u).predict(X)

    np.testing.assert_allclose(m.decision_function(X), f1_with(w1, b1), rtol=0, atol=1e-12)
    # negative control: the pre-update adversary gives a different model, so the test detects order errors
    assert np.max(np.abs(m.decision_function(X) - f1_with(w0, b0))) > 1e-4


# ------------------------------------------------------------------------------------------------
# (c) lam = 0 sanity and predict_proba definition
# ------------------------------------------------------------------------------------------------

def test_c_lambda_zero_is_plain_gradient_boosting_with_same_trees():
    X, y, s = _fixture(n=300, d=6, seed=4)
    T, lr, depth, mf = 40, 0.05, 9, 4
    m = fg.FAGTBBinary(n_estimators=T, learning_rate=lr, max_depth=depth, max_features=mf).fit(X, y, s, 0.0)
    yf = y.astype(float)
    F = np.full(len(y), np.log(yf.sum() / (1 - yf).sum()))
    for t in range(T):
        tree = _ref_tree(depth, mf).fit(X, yf - 1.0 / (1.0 + np.exp(-F)))  # plain log-loss gradient boosting
        upd = tree.predict(X)
        assert np.array_equal(upd, m.estimators_[t].predict(X)), f"tree {t} differs"
        F = F + lr * upd
    assert np.array_equal(m.decision_function(X), F)
    # the adversary has no influence at lam = 0: a different adversary seed gives the identical model
    m2 = fg.FAGTBBinary(n_estimators=T, learning_rate=lr, max_depth=depth, max_features=mf, seed=99).fit(X, y, s, 0.0)
    assert np.array_equal(m2.decision_function(X), F)
    assert not np.array_equal(m2.history_["adversary_w"], m.history_["adversary_w"])


def test_c_predict_proba_matches_cumulative_sum_definition():
    X, y, s = _fixture(n=250, d=6, seed=5)
    Xn = _fixture(n=90, d=6, seed=6)[0]
    m = fg.FAGTBBinary(n_estimators=25, learning_rate=0.1, max_features=None).fit(X, y, s, 0.3)
    for Z in (X, Xn):
        F = np.full(Z.shape[0], m.init_)
        for tree in m.estimators_:
            F = F + m.learning_rate * tree.predict(Z)
        P = m.predict_proba(Z)
        assert P.shape == (Z.shape[0], 2)
        assert np.array_equal(P[:, 1], 1.0 / (1.0 + np.exp(-F)))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-15)
    # fit-time F equals predict-time F on the fitting rows (same accumulation order)
    pf = np.clip(m.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
    ll = float(np.mean(-y * np.log(pf) - (1 - y) * np.log(1 - pf)))
    assert ll == m.history_["task_logloss"][-1]


# ------------------------------------------------------------------------------------------------
# (d) multiclass adaptation (NOT official)
# ------------------------------------------------------------------------------------------------

def test_d_multiclass_residual_matches_finite_difference():
    rng = np.random.default_rng(7)
    n, K = 300, 4
    F = rng.normal(scale=1.5, size=(n, K))
    y = rng.integers(0, K, n)
    Y = np.eye(K)[y]
    s = rng.binomial(1, 0.5, n).astype(np.float64)
    worst = 0.0
    for (w, b, lam, atol) in [(rng.normal(size=K), 0.3, 0.0, 1e-8), (rng.normal(size=K), -0.5, 0.15, 1e-8),
                               (rng.normal(scale=30.0, size=K), 4.0, 0.5, 1e-6)]:
        U = fg.multiclass_pseudo_residual(F, Y, s, w, b, lam)
        fd = -_fd_rows(lambda Z: _mc_rows(Z, y, s, w, b, lam), F, 1e-6)
        np.testing.assert_allclose(U, fd, rtol=1e-7, atol=atol)
        worst = max(worst, float(np.max(np.abs(U - fd))))
        # adversary input gradient w.r.t. the probability vector, and parameter gradient
        P = np.exp(F - np.logaddexp.reduce(F, axis=1)[:, None])
        G = fg.multiclass_adversary_input_grad(P, s, w, b)

        def adv_rows(Q, ww=w, bb=b):
            a = Q @ ww + bb
            return _softplus(a) - s * a

        np.testing.assert_allclose(G, _fd_rows(adv_rows, P, 1e-6), rtol=1e-7, atol=atol)
        gw, gb = fg.multiclass_adversary_param_grad(P, s, w, b)
        for k in range(K):
            e = np.zeros(K)
            e[k] = 1e-6
            fwk = (np.sum(adv_rows(P, w + e)) - np.sum(adv_rows(P, w - e))) / 2e-6
            assert abs(gw[k] - fwk) < 1e-6 * max(1.0, abs(fwk))
        fb = (np.sum(adv_rows(P, w, b + 1e-6)) - np.sum(adv_rows(P, w, b - 1e-6))) / 2e-6
        assert abs(gb - fb) < 1e-6 * max(1.0, abs(fb))
    print(f"   (d) max |U - FD| = {worst:.2e}")


def test_d_multiclass_probabilities_sum_to_one_and_cumulative_definition():
    X, y, s = _fixture(n=240, d=6, seed=8)
    yk = np.digitize(X[:, 1] + 0.5 * X[:, 0], [-0.7, 0.0, 0.7])
    m = fg.FAGTBMulticlassAdaptation(n_estimators=15, learning_rate=0.1, max_features=None).fit(X, yk, s, 0.2)
    Xn = _fixture(n=50, d=6, seed=9)[0]
    for Z in (X, Xn):
        P = m.predict_proba(Z)
        assert P.shape == (Z.shape[0], 4) and np.all(P >= 0)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        F = np.tile(m.init_, (Z.shape[0], 1))
        for rnd in m.estimators_:
            for k, tree in enumerate(rnd):
                F[:, k] = F[:, k] + m.learning_rate * tree.predict(Z)
        np.testing.assert_array_equal(m.decision_function(Z), F)
    np.testing.assert_allclose(m.init_, np.log(np.bincount(yk) / len(yk)))


def test_d_tree_learner_sign_symmetry_and_tie_sensitivity():
    """Facts behind the K=2 relation and behind the free-running parity limit (FAGTB_AUDIT.md section 6).

    (i) The official tree is exactly sign-symmetric: tree(-u) = -tree(u), bitwise.
    (ii) Characterisation: a depth-9 tree on a 2-valued residual (iteration 0 has a constant F_0) has
    exactly tied candidate splits, and floating-point rounding decides between them. A 1e-15 relative
    perturbation of the residual changes its predictions by O(0.1-1).
    """
    X, y, s = _fixture(n=260, d=6, seed=10)
    u = np.random.default_rng(0).normal(size=len(y))
    u2 = y - np.mean(y)  # 2-valued, like the iteration-0 residual at lam = 0
    for v in (u, u2):
        assert np.array_equal(fg.official_tree(9, 4).fit(X, -v).predict(X), -fg.official_tree(9, 4).fit(X, v).predict(X))
    base = fg.official_tree(9, 4).fit(X, u2).predict(X)
    pert = fg.official_tree(9, 4).fit(X, u2 * (1.0 + 1e-15)).predict(X)
    assert np.max(np.abs(base - pert)) > 1e-3


def test_d_multiclass_K2_lambda0_equals_binary_with_doubled_learning_rate():
    """Documented K=2 behaviour at lam = 0: the margin F_1 - F_0 moves by 2*lr*h, i.e. binary FAGTB with 2*lr.

    The per-step identities are exact (U_0 = -U_1 to 1 ulp, U_1 = binary residual, tree(-u) = -tree(u)).
    Over many iterations with deep trees the two paths are equal only up to CART tie-breaking (see the
    sign-symmetry/tie test). So the end-to-end check runs in a tie-free configuration: a single
    continuous feature with depth <= 3 trees.
    """
    lr = 0.05
    for seed in (10, 20, 30):
        X, y, s = _fixture(n=260, d=6, seed=seed)
        X1 = X[:, :1]
        for depth in (1, 3):
            mc = fg.FAGTBMulticlassAdaptation(n_estimators=30, learning_rate=lr, max_depth=depth,
                                              max_features=None).fit(X1, y, s, 0.0)
            bi = fg.FAGTBBinary(n_estimators=30, learning_rate=2 * lr, max_depth=depth,
                                max_features=None).fit(X1, y, s, 0.0)
            Fm = mc.decision_function(X1)
            np.testing.assert_allclose(Fm[:, 1] - Fm[:, 0], bi.decision_function(X1), rtol=0, atol=1e-9)
            for rnd in mc.estimators_:
                np.testing.assert_allclose(rnd[0].predict(X1), -rnd[1].predict(X1), rtol=0, atol=1e-12)
    # per-step identity at arbitrary scores, deep-tree configuration: U_0 = -U_1 and U_1 = binary residual
    rng = np.random.default_rng(21)
    F = rng.normal(size=(260, 2))
    y = rng.integers(0, 2, 260)
    U = fg.multiclass_pseudo_residual(F, np.eye(2)[y], np.zeros(260), np.zeros(2), 0.0, 0.0)
    np.testing.assert_allclose(U[:, 0], -U[:, 1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(U[:, 1], y - 1.0 / (1.0 + np.exp(-(F[:, 1] - F[:, 0]))), rtol=0, atol=1e-15)


def test_d_multiclass_K2_residual_is_binary_residual_under_reparameterisation():
    """At fixed parameters, K=2 residual column 1 = binary residual with W = w1 - w0, B = w0 + b."""
    rng = np.random.default_rng(11)
    n = 300
    F = rng.normal(scale=1.5, size=(n, 2))
    y = rng.integers(0, 2, n)
    s = rng.binomial(1, 0.5, n).astype(np.float64)
    w, b, lam = np.array([0.7, -1.9]), 0.4, 0.8
    U = fg.multiclass_pseudo_residual(F, np.eye(2)[y], s, w, b, lam)
    ub = fg.binary_pseudo_residual(F[:, 1] - F[:, 0], y.astype(float), s, w[1] - w[0], w[0] + b, lam)
    np.testing.assert_allclose(U[:, 1], ub, rtol=0, atol=1e-12)
    np.testing.assert_allclose(U[:, 0], -ub, rtol=0, atol=1e-12)


def test_d_multiclass_K2_lambda_positive_is_not_the_binary_algorithm():
    """Documented non-equivalence: the 3-parameter K=2 adversary takes a different GD path."""
    X, y, s = _fixture(n=200, d=5, seed=12)
    W, B = 0.3, -0.2
    mc = fg.FAGTBMulticlassAdaptation(n_estimators=3, learning_rate=0.05, max_features=None,
                                      adversary_init=(np.array([0.0, W]), B)).fit(X, y, s, 0.5)
    bi = fg.FAGTBBinary(n_estimators=3, learning_rate=0.1, max_features=None, adversary_init=(W, B)).fit(X, y, s, 0.5)
    w_mc = mc.history_["adversary_w"]
    eff_W = w_mc[:, 1] - w_mc[:, 0]
    assert np.max(np.abs(eff_W - bi.history_["adversary_w"])) > 1e-3


def test_d_multiclass_requires_every_class():
    X, y, s = _fixture(n=100, d=4, seed=13)
    try:
        fg.FAGTBMulticlassAdaptation(n_estimators=2, max_features=None, n_classes=3).fit(X, y, s, 0.1)
    except ValueError:
        return
    raise AssertionError("missing class accepted")


# ------------------------------------------------------------------------------------------------
# interface, defaults, determinism
# ------------------------------------------------------------------------------------------------

def test_fit_interfaces_accept_only_fitting_arrays():
    for cls in (fg.FAGTBBinary, fg.FAGTBMulticlassAdaptation):
        assert list(inspect.signature(cls.fit).parameters) == ["self", "X", "y", "s", "lam"]
        X, y, s = _fixture(n=60, d=4, seed=14)
        try:
            cls(n_estimators=1, max_features=None).fit(X, y, s, 0.1, Xtest=X, yt=y, sensitivet=s)
        except TypeError:
            continue
        raise AssertionError("test-label arguments were accepted")


def test_input_validation():
    X, y, s = _fixture(n=80, d=4, seed=15)
    bad = [dict(s=s + 1), dict(y=np.ones_like(y)), dict(y=y[:-1]), dict(lam=-0.1), dict(lam=float("nan")),
           dict(X=np.where(np.arange(X.size).reshape(X.shape) == 3, np.nan, X))]
    for kw in bad:
        args = dict(X=X, y=y, s=s, lam=0.1)
        args.update(kw)
        try:
            fg.FAGTBBinary(n_estimators=1, max_features=None).fit(**args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input {list(kw)}")


def test_defaults_record_official_values():
    m = fg.FAGTBBinary()
    assert m.max_depth == fg.OFFICIAL_MAX_DEPTH_USED == 9 and fg.NOTEBOOK_MAX_DEPTH_PASSED == 10
    assert (m.n_estimators, m.learning_rate, m.max_features, m.adversary_lr) == (300, 0.01, 20, 0.01)
    assert m.tree_random_state == 0
    want = {"criterion": "friedman_mse", "max_depth": 9, "max_features": 20, "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2,
            "min_weight_fraction_leaf": 0.0, "random_state": 0}
    got = fg.official_tree().get_params()
    assert {k: got[k] for k in want} == want
    w, b = fg.OFFICIAL_TF_SEED7_ADVERSARY_INIT
    assert w == b and float(np.float32(w)) == w


def test_determinism_given_seed():
    X, y, s = _fixture(n=200, d=6, seed=16)
    a = fg.FAGTBBinary(n_estimators=20, learning_rate=0.1, max_features=3, seed=5).fit(X, y, s, 0.4)
    b = fg.FAGTBBinary(n_estimators=20, learning_rate=0.1, max_features=3, seed=5).fit(X, y, s, 0.4)
    c = fg.FAGTBBinary(n_estimators=20, learning_rate=0.1, max_features=3, seed=6).fit(X, y, s, 0.4)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    assert all(np.array_equal(a.history_[k], b.history_[k]) for k in a.history_)
    assert a.adversary_init_ != c.adversary_init_
    yk = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)
    m1 = fg.FAGTBMulticlassAdaptation(n_estimators=5, learning_rate=0.1, max_features=None, seed=3).fit(X, yk, s, 0.4)
    m2 = fg.FAGTBMulticlassAdaptation(n_estimators=5, learning_rate=0.1, max_features=None, seed=3).fit(X, yk, s, 0.4)
    assert np.array_equal(m1.predict_proba(X), m2.predict_proba(X))


# ------------------------------------------------------------------------------------------------
# (e) official-environment parity (isolated venv with TensorFlow; skipped when absent)
# ------------------------------------------------------------------------------------------------

def test_e_official_environment_parity():
    if not (os.path.exists(PARITY_PY) and os.path.exists(PARITY_HARNESS)):
        _skip("official-environment parity venv/harness not present (see FAGTB_AUDIT.md section 6)")
    env = dict(os.environ, OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([PARITY_PY, "-I", PARITY_HARNESS, "--worktree", WORKTREE, "--out", PARITY_OUT],
                          capture_output=True, text=True, timeout=1800, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    vendor = os.path.join(WORKTREE, ".private/acs_replication/vendor/FAGTB")
    dirty = subprocess.run(["git", "-C", vendor, "status", "--porcelain"], capture_output=True, text=True).stdout
    assert dirty.strip() == "", f"pinned official clone modified: {dirty}"
    rep = json.load(open(PARITY_OUT))
    assert rep["versions"]["tensorflow"] and rep["official_functions_sha256"] == fg.OFFICIAL_FUNCTIONS_SHA256
    for sc in rep["scenarios"]:
        name, fr, tf_, cc = sc["scenario"], sc["free_running"], sc["teacher_forced"], \
            sc["self_sensitivity_control_port_vs_port_init_plus_1e-12"]
        assert sc["init_F0_equal"], name
        assert (sc["tf_initial_W1"], sc["tf_initial_b1"]) == fg.OFFICIAL_TF_SEED7_ADVERSARY_INIT, name
        kw = {k: v for k, v in sc["official_tree_kwargs"].items() if k != "min_impurity_split"}
        assert all(sc["port_tree_params"][k] == v for k, v in kw.items()), name
        # every official step reproduced at float32 precision; the port's step is exact (fsum reference)
        assert tf_["adversary_step_port_vs_exact_fsum_rel_max"] < 1e-12, name
        assert tf_["adversary_step_tf_float32_vs_exact_fsum_rel_max"] < 1e-4, name
        assert tf_["adversary_input_grad_rel_max"] < 1e-6, name
        assert tf_["residual_abs_max"] < 1e-6, name
        if sc["lam"] == 0.0:  # no adversary influence: bitwise identical trajectories and predictions
            assert fr["residual_maxabs_per_iter_max"] == 0.0 and fr["final_pred_train_maxabs"] == 0.0, name
            assert fr["final_pred_holdout_maxabs"] == 0.0, name
        else:  # free-running gap is no larger than the algorithm's own 1e-12-perturbation sensitivity
            assert fr["residual_maxabs_iter0"] < 1e-6, name
            assert fr["final_pred_train_mean_abs"] <= 2.0 * cc["final_pred_train_mean_abs"] + 1e-3, name
    print("   (e) official parity:", "; ".join(
        f"{sc['scenario']}: TF-step rel {sc['teacher_forced']['adversary_step_tf_float32_vs_exact_fsum_rel_max']:.1e}, "
        f"free-run mean|dp| {sc['free_running']['final_pred_train_mean_abs']:.1e} vs control "
        f"{sc['self_sensitivity_control_port_vs_port_init_plus_1e-12']['final_pred_train_mean_abs']:.1e}"
        for sc in rep["scenarios"]))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed, skipped = 0, 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except BaseException as e:  # noqa: BLE001 - report every outcome
            if type(e).__name__ in ("Skipped", "_Skip"):
                print(f"SKIP {name}: {e}")
                skipped += 1
            else:
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
