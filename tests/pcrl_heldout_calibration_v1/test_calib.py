"""Role C tests for hcal.calib (held-out shared calibration of frozen releases). SYNTHETIC data only: no Adult row, no
task label, no SEX; nothing is loaded or unsealed.

    cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m hcal.sema --label C:test -- \\
        env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q \\
        tests/pcrl_heldout_calibration_v1/test_calib.py

Independent references do not reuse the calibrators' arithmetic: the token objective is re-evaluated row by row from
expanded labels exactly as prompt section 7 states it and minimised by scipy (bounded Brent for K = 2, SLSQP with the
class-dominance constraints for K = 3); temperature optima come from closed forms (K = 2, one token: logit(freq) /
logit(q0)) and the NLL derivatives are checked by central finite differences of an independent logsumexp.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from dpc import utility as DU
from hcal import calib as CB
from lra import decoder as DC

EPS = 1e-12
KAPPA = 32.0


# ----------------------------------------------------------------------------------------------- references
def q_of(u, d):
    K = u.shape[0]
    e = np.zeros(K)
    e[d] = EPS
    return (u + EPS + e) / (1 + (K + 1) * EPS)


def full_objective_rows(u, y, pbar, d):
    """sum_i [-log q(Y_i) + 0.5 ||q - onehot(Y_i)||^2] + 32 KL(pbar || q), from expanded rows (prompt section 7)."""
    q = q_of(np.asarray(u, dtype=np.float64), d)
    K = q.shape[0]
    labels = np.repeat(np.arange(K), np.asarray(y, dtype=np.int64))
    ll = float(np.sum(-np.log(q[labels])))
    br = float(np.sum(np.sum((q[None, :] - np.eye(K)[labels]) ** 2, 1)))
    kl = float(np.sum(np.where(pbar > 0, pbar * (np.log(np.where(pbar > 0, pbar, 1)) - np.log(q)), 0.0)))
    return ll + 0.5 * br + KAPPA * kl


def brute_k2(y, pbar, d):
    """K = 2: u_d = t in [0.5, 1], u_other = 1 - t; bounded Brent on the row-wise objective."""
    def f(t):
        u = np.zeros(2)
        u[d], u[1 - d] = t, 1.0 - t
        return full_objective_rows(u, y, pbar, d)
    r = minimize_scalar(f, bounds=(0.5, 1.0), method="bounded", options={"xatol": 1e-14, "maxiter": 2000})
    t = min((float(r.x), 0.5, 1.0), key=f)                     # Brent never evaluates the bounds themselves
    u = np.zeros(2)
    u[d], u[1 - d] = t, 1.0 - t
    return u, f(t)


def slsqp(y, pbar, d, starts=4, seed=0):
    K = len(y)
    n = float(np.sum(y))
    sc = 1.0 / (n + KAPPA)
    f = lambda u: sc * full_objective_rows(np.clip(u, 0, None), y, pbar, d)  # noqa: E731
    cons = [{"type": "eq", "fun": lambda u: np.sum(u) - 1.0}]
    cons += [{"type": "ineq", "fun": lambda u, k=k: u[d] - u[k]} for k in range(K) if k != d]
    rng = np.random.default_rng(seed)
    best = None
    for j in range(starts):
        u0 = np.full(K, 1.0 / K) if j == 0 else DC.project_class_simplex(rng.dirichlet(np.ones(K)), d)[0]
        r = minimize(f, u0, method="SLSQP", bounds=[(0, 1)] * K, constraints=cons,
                     options={"ftol": 1e-15, "maxiter": 2000})
        u = DC.project_class_simplex(np.asarray(r.x), d)[0]
        val = full_objective_rows(u, y, pbar, d)
        if best is None or val < best[1]:
            best = (u, val)
    return best


def teacher_rows(rng, K, n, d, conc):
    P = rng.dirichlet(np.ones(K) * conc, size=n)
    am = P.argmax(1)
    for i in range(n):
        P[i, [d, am[i]]] = P[i, [am[i], d]]
    return P


def random_token(rng, K, n):
    """(y counts, S, n, d) with S = sum of teacher rows whose argmax is d (lra-style kinds of labels)."""
    d = int(rng.integers(K))
    P = teacher_rows(rng, K, n, d, float(rng.choice([0.05, 0.7, 50.0])))
    kind = int(rng.integers(4))
    if kind == 0:
        y = np.bincount([rng.choice(K, p=p / p.sum()) for p in P], minlength=K)
    elif kind == 1:
        y = np.bincount(np.where(rng.random(n) < 0.8, (d + 1) % K, rng.integers(0, K, n)), minlength=K)
    elif kind == 2:
        y = np.zeros(K, dtype=np.int64)
        y[d] = n
    else:
        y = np.bincount(rng.integers(0, K, n), minlength=K)
    return y.astype(np.float64), P.sum(0), n, d


def batch(rng, K, M, nmax=80):
    toks = [random_token(rng, K, int(rng.integers(1, nmax))) for _ in range(M)]
    return (np.array([t[0] for t in toks]), np.array([t[1] for t in toks]), np.array([t[2] for t in toks]),
            np.array([t[3] for t in toks]))


def sm(u, d):
    return DC.smooth(np.asarray(u, dtype=np.float64), d)


def logit(p):
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------------------------- H-TOKEN32 solve
@pytest.mark.parametrize("K", [2, 3, 6])
def test_token32_bitwise_equal_to_lra_solve_batch(K):
    rng = np.random.default_rng(10 + K)
    Y, S, n, d = batch(rng, K, 250)
    U1, Q1, o1, c1 = DC.solve_batch(Y, S, n, d)
    U2, Q2, o2, c2 = CB.token32_solve(Y, S / n[:, None].astype(np.float64), n, d)
    assert np.array_equal(U1, U2) and np.array_equal(Q1, Q2)
    assert np.array_equal(o1, o2)
    assert set(c1) == set(c2) and all(np.array_equal(c1[f], c2[f]) for f in c1)


def test_token32_rows_independent_of_batch():
    rng = np.random.default_rng(3)
    Y, S, n, d = batch(rng, 6, 25)
    PB = S / n[:, None].astype(np.float64)
    U, Q, o, _ = CB.token32_solve(Y, PB, n, d)
    for i in range(len(n)):
        Ui, Qi, oi, _ = CB.token32_solve(Y[i:i + 1], PB[i:i + 1], n[i:i + 1], d[i:i + 1])
        assert np.array_equal(Ui[0], U[i]) and np.array_equal(Qi[0], Q[i]) and oi[0] == o[i]


def _good():
    return np.array([[3.0, 1.0, 0.0]]), np.array([[0.6, 0.3, 0.1]]), np.array([4]), np.array([0])


@pytest.mark.parametrize("mutate", ["pbar_sum", "pbar_neg", "pbar_nan", "d_not_max", "n0", "y_frac", "y_sum", "kappa",
                                    "eps", "shape"])
def test_token32_input_checks(mutate):
    Y, PB, n, d = _good()
    kw = {}
    if mutate == "pbar_sum":
        PB = PB * (1 + 2e-9)
    elif mutate == "pbar_neg":
        PB = np.array([[0.7, 0.4, -0.1]])
    elif mutate == "pbar_nan":
        PB = np.array([[np.nan, 0.5, 0.5]])
    elif mutate == "d_not_max":
        d = np.array([1])
    elif mutate == "n0":
        Y, n = Y * 0, np.array([0])
    elif mutate == "y_frac":
        Y = np.array([[2.5, 1.5, 0.0]])
    elif mutate == "y_sum":
        n = np.array([5])
    elif mutate == "kappa":
        kw = {"kappa": 16.0}
    elif mutate == "eps":
        kw = {"eps": 1e-10}
    elif mutate == "shape":
        PB = PB[:, :2]
    CB.token32_solve(*_good())                                  # the unmutated inputs solve
    with pytest.raises(ValueError):
        CB.token32_solve(Y, PB, n, d, **kw)


def test_token32_refuses_uncertified(monkeypatch):
    rng = np.random.default_rng(4)
    Y, S, n, d = batch(rng, 3, 5)
    monkeypatch.setattr(CB, "BISECT_ITERS", 3)                  # bracket far wider than 2 ulp -> not certified
    with pytest.raises(DC.DecoderError):
        CB.token32_solve(Y, S / n[:, None], n, d)


def test_token32_k2_against_bounded_brent():
    rng = np.random.default_rng(5)
    cases = [(np.array([7.0, 3.0]), np.array([0.7, 0.3]), 0),     # interior
             (np.array([1.0, 9.0]), np.array([0.55, 0.45]), 0),    # labels on the other class: tie u_d = u_k active
             (np.array([0.0, 2.0]), np.array([0.0, 1.0]), 1),      # mu_other = 0, no labels on it: zero active
             (np.array([1.0, 0.0]), np.array([0.9, 0.1]), 0),      # n_cal = 1
             (np.array([0.0, 2.0]), np.array([0.2, 0.8]), 1)]      # n_cal = 2
    for _ in range(6):
        d = int(rng.integers(2))
        mu = teacher_rows(rng, 2, 20, d, 0.7).mean(0)
        cases.append((np.bincount(rng.integers(0, 2, int(rng.integers(1, 60))), minlength=2).astype(float), mu, d))
    for y, mu, d in cases:
        U, Q, obj, certs = CB.token32_solve(y[None], mu[None], np.array([int(y.sum())]), np.array([d]))
        cert = DC.cert_row(certs, 0)
        u_ref, f_ref = brute_k2(y, mu, d)
        assert np.max(np.abs(Q[0] - q_of(u_ref, d))) <= 1e-8
        f_ours = full_objective_rows(U[0], y, mu, d)
        assert f_ours <= f_ref + 1e-10 * max(1.0, abs(f_ref))
        assert abs(cert["obj_full"] - f_ours) <= 1e-10 * max(1.0, abs(f_ours))
        assert cert["converged"] and cert["stationarity_rel"] <= DC.STAT_TOL and cert["dual_infeas_rel"] <= DC.DUAL_TOL
        assert cert["margin"] > 0 and cert["bracket_ulps"] <= 2.0


def test_token32_k3_against_slsqp():
    rng = np.random.default_rng(6)
    for j in range(8):
        d = int(rng.integers(3))
        mu = teacher_rows(rng, 3, 30, d, [0.05, 0.7, 50.0][j % 3]).mean(0)
        y = np.bincount(rng.integers(0, 3, int(rng.integers(1, 40))), minlength=3).astype(float)
        U, Q, obj, certs = CB.token32_solve(y[None], mu[None], np.array([int(y.sum())]), np.array([d]))
        u_ref, f_ref = slsqp(y, mu, d, seed=j)
        f_ours = full_objective_rows(U[0], y, mu, d)
        assert f_ours <= f_ref + 1e-9 * max(1.0, abs(f_ref))
        assert np.max(np.abs(Q[0] - q_of(u_ref, d))) <= 1e-5
        assert bool(certs["converged"][0])


# ----------------------------------------------------------------------------------------------- fit_token32
def _token_fixture(K=3):
    """Six tokens: 0 reserved WITH calibration rows, 1 n_cal = 0, 2 n_cal = 1, 3 n_cal = 2, 4 n_cal = 40, 5 reserved
    without rows."""
    rng = np.random.default_rng(7)
    tc = np.array([0, 1, 2, 0, 1, 2])
    n_fit = np.array([0, 12, 30, 25, 80, 0])
    mu = np.full((6, K), np.nan)
    q0 = np.empty((6, K))
    for t in range(6):
        if n_fit[t]:
            mu[t] = teacher_rows(rng, K, n_fit[t], tc[t], 0.7).mean(0)
            q0[t] = sm(mu[t], tc[t])
        else:
            q0[t] = sm(np.full(K, 1.0 / K), tc[t])
    tok = np.r_[[0, 0, 0], [2], [3, 3], np.full(40, 4)]
    y = np.r_[[1, 2, 0], [1], [2, 0], rng.integers(0, K, 40)]
    return tok, y, mu, q0, tc, n_fit


def test_fit_token32_statuses_counts_and_fallbacks():
    tok, y, mu, q0, tc, n_fit = _token_fixture()
    tab = CB.fit_token32(tok, y, mu, q0, tc, n_fit, 3)
    assert tab["status"] == [CB.RESERVED, CB.NO_CAL, CB.FITTED, CB.FITTED, CB.FITTED, CB.RESERVED]
    assert tab["n_cal"].tolist() == [3, 0, 1, 2, 40, 0]
    assert tab["status_counts"] == {CB.FITTED: 3, CB.NO_CAL: 1, CB.RESERVED: 2}
    assert tab["parameter_count"] == 3 * 2
    for t in (0, 1, 5):                                          # fallbacks: q0 exactly, never fitted
        assert np.array_equal(tab["q"][t], q0[t]) and np.isnan(tab["u"][t]).all()
        assert "fallback" in tab["certificate"]["tokens"][t]
    fit = np.array([2, 3, 4])
    U, Q, _, _ = CB.token32_solve(tab["y_cal"][fit].astype(float), mu[fit], tab["n_cal"][fit], tc[fit])
    assert np.array_equal(tab["q"][fit], Q) and np.array_equal(tab["u"][fit], U)
    for t in fit:
        c = tab["certificate"]["tokens"][t]
        assert c["status"] == CB.FITTED and c["converged"] and set(DC.CERT_FIELDS) <= set(c)
        assert not np.array_equal(tab["q"][t], q0[t])
    s = tab["certificate"]["summary"]
    assert s is tab["summary"] and tab["certs"] is tab["certificate"]["tokens"]
    assert s["violations"] == [] and s["supervised_tokens"] == 3 and s["fallback_tokens"] == 3
    assert np.max(np.abs(tab["q"].sum(1) - 1)) <= 1e-12
    assert np.array_equal(tab["q"].argmax(1), tc)
    assert len(tab["content_sha256"]) == 64


def test_fit_token32_uses_the_fixed_prior_only():
    tok, y, mu, q0, tc, n_fit = _token_fixture()
    a = CB.fit_token32(tok, y, mu, q0, tc, n_fit, 3)
    b = CB.fit_token32(tok, y, mu, q0, tc, n_fit, 3)
    assert np.array_equal(a["q"], b["q"]) and a["content_sha256"] == b["content_sha256"]
    mu2 = mu.copy()
    mu2[4] = 0.5 * mu[4] + 0.5 * np.eye(3)[tc[4]]                # a different (still valid) prior for token 4
    c = CB.fit_token32(tok, y, mu2, q0, tc, n_fit, 3)
    assert not np.array_equal(c["q"][4], a["q"][4]) and np.array_equal(c["q"][:4], a["q"][:4])
    assert c["prior_sha256"] != a["prior_sha256"] and c["content_sha256"] != a["content_sha256"]
    t = CB.fit_token32(tok, y, mu, q0, tc, n_fit, 3, kind="T-TOKEN32")
    assert np.array_equal(t["q"], a["q"]) and t["kind"] == "T-TOKEN32" and t["content_sha256"] != a["content_sha256"]


@pytest.mark.parametrize("bad", ["mu_reserved_finite", "mu_nan_fitted", "q0_class", "neg_label", "tok_range",
                                 "misaligned", "kind"])
def test_fit_token32_validation(bad):
    tok, y, mu, q0, tc, n_fit = _token_fixture()
    kw = {}
    err = ValueError
    if bad == "mu_reserved_finite":
        mu = mu.copy()
        mu[0] = np.array([0.5, 0.3, 0.2])
    elif bad == "mu_nan_fitted":
        mu = mu.copy()
        mu[2] = np.nan
    elif bad == "q0_class":
        tc = tc.copy()
        tc[3] = 1
    elif bad == "neg_label":
        y = y.copy()
        y[0] = -1
        err = PermissionError
    elif bad == "tok_range":
        tok = tok.copy()
        tok[0] = 6
    elif bad == "misaligned":
        y = y[:-1]
    elif bad == "kind":
        kw = {"kind": "H-GLOBAL-TEMP"}
    with pytest.raises(err):
        CB.fit_token32(tok, y, mu, q0, tc, n_fit, 3, **kw)


# ----------------------------------------------------------------------------------------------- temperatures
def test_temperature_known_optimum_k2():
    q0 = sm([0.8, 0.2], 0)[None]
    tok = np.zeros(100, dtype=np.int64)
    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    tab = CB.fit_global_temp(tok, y, q0, np.array([0]), 2)
    a_true = logit(0.9) / np.log(q0[0, 0] / q0[0, 1])
    c = tab["certificate"]
    assert abs(tab["alpha"] - a_true) <= 1e-9
    assert c["status"] == "INTERIOR" and c["certified"] and c["bracket_ulps"] <= 2 and c["grad_rel"] <= 1e-9
    assert c["nll_alpha"] < c["nll_one"] and c["curvature_alpha"] > 0
    assert abs(tab["q"][0, 0] - 0.9) <= 1e-12 and tab["parameter_count"] == 1


@pytest.mark.parametrize("freq,status", [(0.999, "BOUNDARY_HIGH"), (0.55, "BOUNDARY_LOW"), (0.3, "BOUNDARY_LOW")])
def test_temperature_boundaries_kkt_sign(freq, status):
    q0 = sm([0.8, 0.2], 0)[None]
    n1 = int(round(1000 * freq))
    y = np.r_[np.zeros(n1, int), np.ones(1000 - n1, int)]
    L = np.log(q0[np.zeros(1000, dtype=np.int64)])
    c = CB.solve_temperature(L, y)
    assert c["status"] == status and c["certified"]
    if status == "BOUNDARY_HIGH":
        assert c["alpha"] == 4.0 and c["g_alpha"] <= 0
        assert logit(freq) / np.log(q0[0, 0] / q0[0, 1]) > 4
    else:
        assert c["alpha"] == 0.25 and c["g_alpha"] >= 0
    grid = np.linspace(0.25, 4, 61)
    assert all(c["nll_alpha"] <= CB.temp_terms(L, y, a)[0] + 1e-15 for a in grid)


def _random_logits(seed, N=300, K=6):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(K) * 0.5, size=N)
    y = np.array([rng.choice(K, p=p) for p in P])
    return np.log(np.maximum(P, 1e-300)), y


def test_nll_derivative_curvature_and_convexity():
    L, y = _random_logits(8)
    rows = np.arange(len(y))
    nll_ref = lambda a: float(np.mean(-a * L[rows, y] + logsumexp(a * L, axis=1)))  # noqa: E731
    for a in (0.3, 0.9, 1.0, 1.7, 3.5):
        nll, g, curv = CB.temp_terms(L, y, a)
        h = 1e-5
        assert abs(nll - nll_ref(a)) <= 1e-12 * max(1, abs(nll))
        assert abs(g - (nll_ref(a + h) - nll_ref(a - h)) / (2 * h)) <= 1e-6 * max(1, abs(g))
        gp, gm = CB.temp_terms(L, y, a + h)[1], CB.temp_terms(L, y, a - h)[1]
        assert curv >= 0 and abs(curv - (gp - gm) / (2 * h)) <= 1e-5 * max(1, curv)
    grid = np.linspace(0.25, 4, 121)
    vals = np.array([CB.temp_terms(L, y, a)[0] for a in grid])
    scale = float(np.mean(np.abs(L).max(1)))
    assert np.all(vals[:-2] - 2 * vals[1:-1] + vals[2:] >= -1e-12 * scale)
    gs = np.array([CB.temp_terms(L, y, a)[1] for a in grid])
    assert np.all(np.diff(gs) >= -1e-12 * scale)


def test_identity_paths_tokens_and_u():
    tok, y, mu, q0, tc, n_fit = _token_fixture()
    out = CB.temp_apply_tokens(q0, 1.0, tc)
    assert np.array_equal(out, q0) and out is not q0
    assert np.array_equal(CB.temp_apply_tokens_by_class(q0, [1.0, 1.0, 1.0], tc), q0)
    near = CB.temp_apply_tokens(q0, np.nextafter(1.0, 2.0), tc)
    assert not np.array_equal(near, q0) and np.max(np.abs(near - q0)) <= 1e-12
    rng = np.random.default_rng(9)
    P = rng.dirichlet(np.ones(6) * 0.3, size=200)
    P[0] = np.r_[1.0, np.zeros(5)]                               # zeros and values below 1e-12 survive identity
    P[1] = np.r_[1.0 - 5e-14, 5e-14, np.zeros(4)]
    assert np.array_equal(CB.temp_apply_probs(P, 1.0), P)
    tab = {"kind": "H-CLASS-TEMP", "target": "U", "K": 6, "alpha": None, "alphas": [1.0] * 6}
    assert np.array_equal(CB.apply_u(P, tab), P)
    g = {**tab, "kind": "H-GLOBAL-TEMP", "alpha": 1.0, "alphas": None}
    assert np.array_equal(CB.apply_u(P, g), P)


def test_zero_calibration_rows():
    tok, y, mu, q0, tc, n_fit = _token_fixture()
    e = np.zeros(0, dtype=np.int64)
    tab = CB.fit_token32(e, e, mu, q0, tc, n_fit, 3)             # every token keeps q0: no statistics, no fit
    assert np.array_equal(tab["q"], q0) and tab["parameter_count"] == 0
    assert tab["status_counts"] == {CB.FITTED: 0, CB.NO_CAL: 4, CB.RESERVED: 2}
    for f in (CB.fit_global_temp, CB.fit_class_temp):
        with pytest.raises(CB.CalibrationError):
            f(e, e, q0, tc, 3)
    with pytest.raises(CB.CalibrationError):
        CB.fit_u_temp(np.zeros((0, 3)), e, "H-CLASS-TEMP")


def test_solve_returning_exactly_one_uses_identity(monkeypatch):
    tok, y, mu, q0, tc, n_fit = _token_fixture()
    real = CB.solve_temperature

    def fake(L, yy):
        c = real(L, yy)
        return {**c, "alpha": 1.0}
    monkeypatch.setattr(CB, "solve_temperature", fake)
    tab = CB.fit_global_temp(tok, y, q0, tc, 3)
    assert np.array_equal(tab["q"], q0)


def test_global_temp_transforms_every_token():
    tok, y, mu, q0, tc, n_fit = _token_fixture()
    tok = np.r_[tok, np.full(60, 4)]
    y = np.r_[y, np.full(60, tc[4])]                             # mostly-correct labels -> alpha != 1
    tab = CB.fit_global_temp(tok, y, q0, tc, 3)
    a = tab["alpha"]
    assert a != 1.0
    ref = np.exp(a * np.log(q0)) / np.exp(a * np.log(q0)).sum(1, keepdims=True)
    assert np.max(np.abs(tab["q"] - ref)) <= 1e-14
    for t in (0, 1, 5):                                          # reserved / calibration-free tokens: transformed
        assert not np.array_equal(tab["q"][t], q0[t])
    assert np.array_equal(tab["q"].argmax(1), tc)


def test_class_temp_fallbacks_absent_and_threshold():
    rng = np.random.default_rng(11)
    K = 3
    tc = np.array([0, 0, 1, 1, 2])
    q0 = np.array([sm([0.7, 0.2, 0.1], 0), sm([0.5, 0.3, 0.2], 0), sm([0.3, 0.6, 0.1], 1), sm([0.2, 0.45, 0.35], 1),
                   sm([0.3, 0.3, 0.4], 2)])
    for n1, st1 in ((49, CB.CLASS_FALLBACK), (50, CB.FITTED)):
        tok = np.r_[rng.choice([0, 1], 150), rng.choice([2, 3], n1)]      # class 2 absent
        y = np.r_[np.where(rng.random(150) < 0.9, 0, 1), np.where(rng.random(n1) < 0.95, 1, 2)]
        tab = CB.fit_class_temp(tok, y, q0, tc, K)
        recs = tab["certificate"]["classes"]
        assert [r["status"] for r in recs] == [CB.FITTED, st1, CB.CLASS_FALLBACK]
        assert recs[2]["absent"] and recs[2]["n"] == 0 and tab["alphas"][2] == 1.0
        assert np.array_equal(tab["q"][4], q0[4])
        m = tc[tok] == 0
        assert tab["alphas"][0] == CB.solve_temperature(np.log(q0[tok[m]]), y[m])["alpha"]
        if st1 == CB.CLASS_FALLBACK:
            assert tab["alphas"][1] == 1.0 and np.array_equal(tab["q"][2:4], q0[2:4]) and tab["parameter_count"] == 1
        else:
            assert tab["alphas"][1] != 1.0 and tab["parameter_count"] == 2
        pc = tab["parameter_count"]
        assert tab["status_counts"] == {CB.FITTED: pc, CB.CLASS_FALLBACK: K - pc}
        assert np.array_equal(tab["q"].argmax(1), tc)


# ----------------------------------------------------------------------------------------------- continuous U
def _u_data(seed, n=600, K=6):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(K) * 0.4, size=n)
    sharp = P ** 2.0 / (P ** 2.0).sum(1, keepdims=True)          # labels from a sharper model -> alpha > 1
    y = np.array([rng.choice(K, p=p) for p in sharp])
    return P, y


def test_u_log_input_rule_and_global_fit():
    P, y = _u_data(12)
    P[:3] = np.array([[1.0, 0, 0, 0, 0, 0], [0.5, 0.5 - 1e-13, 1e-13, 0, 0, 0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]])
    tab = CB.fit_u_temp(P[:400], y[:400], "H-GLOBAL-TEMP")
    a = tab["alpha"]
    assert a > 1.0 and tab["parameter_count"] == 1
    Pp = np.maximum(P, 1e-12)
    Pp = Pp / Pp.sum(1, keepdims=True)
    assert np.max(np.abs(CB.u_log_inputs(P) - np.log(Pp))) <= 1e-13
    assert abs(a - CB.solve_temperature(np.log(Pp[:400]), y[:400])["alpha"]) <= 1e-12
    q = CB.apply_u(P, tab)                                        # ALL rows
    ref = np.exp(a * np.log(Pp)) / np.exp(a * np.log(Pp)).sum(1, keepdims=True)
    assert np.max(np.abs(q - ref)) <= 1e-14 and np.all(q > 0)
    assert np.array_equal(q.argmax(1), P.argmax(1))


def test_u_class_temp_fallback():
    P, y = _u_data(13, n=500, K=3)
    d = P.argmax(1)
    keep = np.r_[np.flatnonzero(d == 0)[:200], np.flatnonzero(d == 1)[:30]]       # class 1: 30 rows; class 2 absent
    tab = CB.fit_u_temp(P[keep], y[keep], "H-CLASS-TEMP")
    assert [r["status"] for r in tab["certificate"]["classes"]] == [CB.FITTED, CB.CLASS_FALLBACK, CB.CLASS_FALLBACK]
    assert tab["alphas"][1:] == [1.0, 1.0] and tab["parameter_count"] == 1
    q = CB.apply_u(P, tab)
    assert np.array_equal(q[d != 0], P[d != 0])                  # fallback classes: P exactly
    assert not np.array_equal(q[d == 0], P[d == 0])
    assert np.array_equal(q.argmax(1), d)
    with pytest.raises(ValueError):
        CB.apply_u(P, tab, d=(d + 1) % 3)                         # decisions must be the teacher argmax


def test_argmax_preservation_near_ties():
    tc = np.array([0, 2, 1])
    q0 = np.array([sm([0.5, 0.5, 0.0], 0), sm([0.0, 0.5, 0.5], 2), sm([1 / 3, 1 / 3, 1 / 3], 1)])  # u ties: eps margin
    for a in (0.25, 0.6, 2.5, 4.0):
        q = CB.temp_apply_tokens(q0, a, tc)
        assert np.all(q[np.arange(3), tc] > np.max(np.where(np.eye(3)[tc] > 0, -1, q), 1))
        assert np.array_equal(CB.temp_apply_tokens_by_class(q0, [a, 1.0, a], tc).argmax(1), tc)
    P = np.array([[0.5 - 5e-13, 0.5 + 5e-13], [0.3, 0.7], [0.5 + 1e-12, 0.5 - 1e-12]])
    for a in (0.25, 4.0):
        assert np.array_equal(CB.temp_apply_probs(P, a).argmax(1), P.argmax(1))
    b = 0.5
    tie_q0 = np.array([[np.nextafter(b, 0), b]])                 # 1-ulp margin collapses under alpha = 0.25
    with pytest.raises(CB.CalibrationError, match="1 token"):
        CB.temp_apply_tokens(tie_q0, 0.25, np.array([1]))
    with pytest.raises(CB.CalibrationError, match="1 row"):
        CB.temp_apply_probs(np.array([[np.nextafter(b, 0), b], [0.3, 0.7]]), 0.25)
    with pytest.raises(ValueError):
        CB.temp_apply_tokens(q0, 5.0, tc)                         # outside the registered bounds


def test_unclipped_vs_clipped_recording():
    """Token A (q0 ~ (1, 1e-12)) carries one label-1 row: its scored probability is below 1e-12 (already at alpha = 1
    through the eps smoothing, far below at alpha > 1). The record keeps both the unclipped objective and the clipped
    score."""
    q0 = np.array([sm([1.0, 0.0], 0), sm([0.6, 0.4], 0)])
    tc = np.array([0, 0])
    tok = np.r_[np.zeros(51, int), np.ones(300, int)]
    y = np.r_[np.zeros(50, int), [1], np.zeros(297, int), np.ones(3, int)]
    tab = CB.fit_global_temp(tok, y, q0, tc, 2)
    c = tab["certificate"]
    a = tab["alpha"]
    assert c["status"] == "INTERIOR" and 1.0 < a < 4.0
    L = np.log(q0[tok])
    rows = np.arange(len(y))
    assert abs(c["nll_alpha"] - np.mean(-a * L[rows, y] + logsumexp(a * L, 1))) <= 1e-12
    assert abs(c["nll_one"] - np.mean(-L[rows, y] + logsumexp(L, 1))) <= 1e-12
    qa = tab["q"][tok][rows, y]
    assert c["clipped_count_alpha"] == int(np.sum(qa < 1e-12)) == 1
    assert c["clipped_count_one"] == int(np.sum(q0[tok][rows, y] < 1e-12)) == 1
    assert abs(c["score_ll_alpha"] - np.mean(-np.log(np.clip(qa, 1e-12, 1)))) <= 1e-12
    assert c["nll_alpha"] - c["score_ll_alpha"] > 0.5 * 27.6 * (a - 1) / len(y)   # the clip hides the tail
    # U: a zero true-label probability: unclipped scoring NLL is infinite (recorded as None), clipped is finite
    P = np.array([[1.0, 0.0]] + [[0.8, 0.2]] * 99)
    yu = np.r_[[1], np.zeros(99, int)]
    tu = CB.fit_u_temp(P, yu, "H-GLOBAL-TEMP")
    s = tu["calibration_scores"]
    assert s["nll_unclipped_original"] is None and s["clipped_count_original"] == 1
    assert np.isfinite(s["ll_clipped_original"]) and tu["certificate"]["clipped_count_one"] == 1
    assert np.isfinite(tu["certificate"]["nll_one"])               # solver objective uses the P' log-input rule


# ----------------------------------------------------------------------------------------------- helpers / records
def test_row_losses_and_unclipped_nll():
    rng = np.random.default_rng(14)
    P = rng.dirichlet(np.ones(6), size=50)
    P[0] = np.r_[1.0, np.zeros(5)]
    y = rng.integers(0, 6, 50)
    y[0] = 3
    ll, br = CB.row_losses(P, y)
    pr = DU.per_row(P, y, 6)
    assert np.array_equal(ll, pr["ll"]) and np.array_equal(br, pr["br"])
    assert ll[0] == -np.log(1e-12) and br[0] == 2.0
    un = CB.nll_unclipped(P, y)
    assert np.isinf(un[0]) and np.allclose(un[1:], -np.log(P[np.arange(1, 50), y[1:]]), rtol=0, atol=0)


def _bank(seed):
    """Synthetic frozen bank (hcal.admit.build_bank keys) for recipients K=2 (T=6) and K=6 (T=20)."""
    rng = np.random.default_rng(seed)
    N, n_fit_rows = 3000, 2000
    bank = {}
    for i, (K, T) in ((1, (2, 6)), (2, (6, 20))):
        tc = np.arange(T) % K
        tok = rng.integers(0, T - 1, N)                          # the last token never receives rows: reserved
        tok[n_fit_rows:][rng.random(N - n_fit_rows) < 0.05] = T - 1      # ... but some non-fit rows reach it
        P = np.array([teacher_rows(rng, K, 1, tc[t], 0.6)[0] for t in tok])
        fit = np.arange(n_fit_rows)
        n = np.bincount(tok[fit], minlength=T).astype(np.int64)
        S = np.stack([np.bincount(tok[fit], weights=P[fit, k], minlength=T) for k in range(K)], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(n[:, None] > 0, S / np.where(n > 0, n, 1)[:, None], np.nan)
        q0 = np.array([sm(mu[t], tc[t]) if n[t] else sm(np.full(K, 1.0 / K), tc[t]) for t in range(T)])
        bank.update({f"tok{i}": tok, f"hard{i}": tc[tok], f"class{i}": tc, f"n_fit{i}": n, f"S{i}": S, f"mu{i}": mu,
                     f"q0{i}": q0, f"P{i}": P})
    return bank


@pytest.mark.parametrize("family", list(CB.TOKEN_FAMILIES))
def test_calibrate_partition_records_and_decisions(family):
    bank = _bank(15)
    rng = np.random.default_rng(16)
    cal = np.sort(rng.choice(np.arange(2000, 3000), 600, replace=False))
    if family == "T-TOKEN32":
        cal = np.sort(rng.choice(np.arange(2000), 600, replace=False))
    yb = {"income": np.array([rng.choice(2, p=p) for p in bank["P1"][cal]]),
          "occupation": np.array([rng.choice(6, p=p) for p in bank["P2"][cal]])}
    out = CB.calibrate_partition(bank, cal, yb, family)
    assert sorted(out) == [1, 2]
    for i, K in ((1, 2), (2, 6)):
        tab = out[i]
        assert tab["kind"] == family and tab["K"] == K
        rows_q = CB.apply_table(tab["q"], bank[f"tok{i}"])
        assert np.array_equal(rows_q, tab["q"][bank[f"tok{i}"]])
        assert np.array_equal(rows_q.argmax(1), bank[f"hard{i}"])          # every released decision unchanged
        rec = CB.decoder_table(family, i, tab)
        s = json.dumps(rec, allow_nan=False)
        back = json.loads(s)
        assert np.array_equal(np.array(back["q"]), tab["q"]) and back["content_sha256"] == tab["content_sha256"]
        assert back["recipient"] == i and back["task"] == CB.TASK_OF[i]
        if family in ("H-TOKEN32", "T-TOKEN32"):
            st = tab["status"]
            assert st[-1] == CB.RESERVED and np.array_equal(tab["q"][-1], bank[f"q0{i}"][-1])
            if family == "H-TOKEN32":
                assert tab["n_cal"][-1] > 0                       # reserved token reached by calibration rows
            direct = CB.fit_token32(bank[f"tok{i}"][cal], yb[CB.TASK_OF[i]], bank[f"mu{i}"], bank[f"q0{i}"],
                                    bank[f"class{i}"], bank[f"n_fit{i}"], K, kind=family)
            assert direct["content_sha256"] == tab["content_sha256"]
            assert rec["parameter_count"] == st.count(CB.FITTED) * (K - 1)
        elif family == "H-CLASS-TEMP":
            assert len(rec["alphas"]) == K and rec["parameter_count"] <= K
        else:
            assert rec["parameter_count"] == 1 and CB.TEMP_LO <= rec["alpha"] <= CB.TEMP_HI
    with pytest.raises(ValueError):
        CB.decoder_table(family, 2, out[1])                       # recipient 2 needs K = 6
    bad = {**out[1], "q": out[1]["q"].copy()}
    bad["q"][0, 0] += 1e-15
    with pytest.raises(CB.CalibrationError):
        CB.decoder_table(family, 1, bad)                          # tampered content
    with pytest.raises(ValueError):
        CB.calibrate_partition({**bank, "hard1": 1 - bank["hard1"]}, cal, yb, family)
    with pytest.raises(ValueError):
        CB.calibrate_partition(bank, cal, {**yb, "income": yb["income"][:-1]}, family)


def test_u_record_json_safe():
    P, y = _u_data(17, n=300, K=2)
    for fam in CB.TEMP_FAMILIES:
        tab = CB.fit_u_temp(P, y, fam)
        rec = CB.decoder_table(fam, 1, tab)
        json.dumps(rec, allow_nan=False)
        assert rec["target"] == "U" and rec["q"] is None and rec["content_sha256"] == CB.table_sha256(tab)
    bad = {**tab, "certificate": {**tab["certificate"], "nll_alpha": float("nan")}}
    with pytest.raises(CB.CalibrationError):
        CB.decoder_table(tab["kind"], 1, bad)                      # non-finite values are refused, never dropped
