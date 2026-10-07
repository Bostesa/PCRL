"""Role B tests for lcr.decoder (the common learned decoder D1). SYNTHETIC data only: no Adult row, label or SEX.

    cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lcr.sema --label B:test-decoder -- \\
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python \\
        -m pytest -q lcr/tests/test_decoder.py

The independent references here do NOT reuse the decoder's pooling/closed-form arithmetic: the objective is recomputed
ROW BY ROW from expanded rows exactly as the prompt states it, the reference optimum comes from scipy SLSQP (a
general constrained solver), and KKT optimality is re-derived from finite differences and feasible-direction probes.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
from scipy.optimize import minimize

from lcr import decoder as DC
from qpc import kmeans as KM
from qpc import release as RL

EPS = 1e-12
KAPPA = 32.0


# ----------------------------------------------------------------------------------------------- references
def q_of(u, d):
    K = u.shape[0]
    e = np.zeros(K)
    e[d] = EPS
    return (u + EPS + e) / (1 + (K + 1) * EPS)


def full_objective_rows(u, y, s, n, d):
    """The prompt's objective evaluated row by row on n expanded rows (independent of the decoder's algebra)."""
    q = q_of(np.asarray(u, dtype=np.float64), d)
    K = q.shape[0]
    labels = np.repeat(np.arange(K), np.asarray(y, dtype=np.int64))
    ll = float(np.sum(-np.log(q[labels])))
    br = float(np.sum(np.sum((q[None, :] - np.eye(K)[labels]) ** 2, 1)))
    pbar = np.asarray(s) / n
    kl = float(np.sum(np.where(pbar > 0, pbar * (np.log(np.where(pbar > 0, pbar, 1)) - np.log(q)), 0.0)))
    return ll + 0.5 * br + KAPPA * kl


def full_objective_closed(u, y, s, n, d):
    q = q_of(np.asarray(u, dtype=np.float64), d)
    pbar = np.asarray(s) / n
    a = np.asarray(y) + KAPPA * pbar
    const = 0.5 * n + KAPPA * float(np.sum(np.where(pbar > 0, pbar * np.log(np.where(pbar > 0, pbar, 1)), 0)))
    return float(-np.sum(np.where(a > 0, a * np.log(q), 0)) + 0.5 * n * np.sum(q * q) - np.dot(y, q)) + const


def slsqp(y, s, n, d, starts=3, seed=0):
    """Reference optimum of the full objective over the class-dominant simplex by scipy SLSQP (scaled by 1/(n+kappa)),
    best of several starts."""
    K = len(y)
    sc = 1.0 / (n + KAPPA)
    f = lambda u: sc * full_objective_closed(np.clip(u, 0, None), y, s, n, d)  # noqa: E731
    cons = [{"type": "eq", "fun": lambda u: np.sum(u) - 1.0}]
    for k in range(K):
        if k != d:
            cons.append({"type": "ineq", "fun": lambda u, k=k: u[d] - u[k]})
    rng = np.random.default_rng(seed)
    best = None
    for j in range(starts):
        u0 = np.full(K, 1.0 / K) if j == 0 else rng.dirichlet(np.ones(K))
        u0 = DC.project_class_simplex(u0, d)[0]
        r = minimize(f, u0, method="SLSQP", bounds=[(0, 1)] * K, constraints=cons,
                     options={"ftol": 1e-15, "maxiter": 2000})
        u = DC.project_class_simplex(np.asarray(r.x), d)[0]
        val = full_objective_closed(u, y, s, n, d)
        if best is None or val < best[1]:
            best = (u, val)
    return best


def random_token(rng, K, n, kind="dirichlet"):
    d = int(rng.integers(K))
    if kind == "dirichlet":
        P = rng.dirichlet(np.ones(K) * 0.7, size=n)
    elif kind == "peaked":
        P = rng.dirichlet(np.ones(K) * 0.05, size=n)
    elif kind == "flat":
        P = rng.dirichlet(np.ones(K) * 50.0, size=n)
    am = P.argmax(1)
    for i in range(n):
        P[i, [d, am[i]]] = P[i, [am[i], d]]
    if kind == "flat":
        P = P / P.sum(1, keepdims=True)
    lab_kind = rng.integers(4)
    if lab_kind == 0:          # labels from the teacher (roughly calibrated)
        y = np.bincount([rng.choice(K, p=p / p.sum()) for p in P], minlength=K)
    elif lab_kind == 1:        # labels concentrated on a non-d class (miscalibrated; tie constraint likely active)
        o = (d + 1) % K
        y = np.bincount(np.where(rng.random(n) < 0.8, o, rng.integers(0, K, n)), minlength=K)
    elif lab_kind == 2:        # all labels on d
        y = np.zeros(K, dtype=np.int64)
        y[d] = n
    else:                      # uniform labels
        y = np.bincount(rng.integers(0, K, n), minlength=K)
    return y.astype(np.float64), P.sum(0), n, d


def kkt_by_feasible_probes(u, y, s, n, d, h=1e-7):
    """Directional derivatives of the full objective along every feasible pairwise transfer u_i -> u_j that keeps
    the class-dominant simplex (independent first-order optimality check): none may be materially negative."""
    K = len(u)
    f0 = full_objective_closed(u, y, s, n, d)
    worst = 0.0
    for i, j in itertools.permutations(range(K), 2):
        v = u.copy()
        step = min(h, v[i])
        if step <= 0:
            continue
        v[i] -= step
        v[j] += step
        if v[j] > v[d] + 0.0 and j != d:
            continue
        if i == d and np.any(np.delete(v, d) > v[d]):
            continue
        worst = min(worst, (full_objective_closed(v, y, s, n, d) - f0) / step)
    return worst


# ----------------------------------------------------------------------------------------------- the solver
def test_objective_algebra_matches_prompt_rowwise():
    rng = np.random.default_rng(1)
    for K in (2, 6):
        for _ in range(5):
            y, s, n, d = random_token(rng, K, int(rng.integers(1, 60)))
            sol = DC.solve_token(y, s, n, d)
            a = full_objective_rows(sol.u, y, s, n, d)
            assert abs(sol.cert["obj_full"] - a) <= 1e-10 * max(1.0, abs(a))


@pytest.mark.parametrize("K", [2, 6])
def test_matches_independent_slsqp_reference(K):
    rng = np.random.default_rng(10 + K)
    for case in range(12):
        n = int(rng.choice([1, 2, 5, 37, 400]))
        y, s, n, d = random_token(rng, K, n, kind=["dirichlet", "peaked", "flat"][case % 3])
        sol = DC.solve_token(y, s, n, d)
        u_ref, f_ref = slsqp(y, s, n, d)
        f_us = full_objective_closed(sol.u, y, s, n, d)
        assert f_us <= f_ref + 1e-9 * max(1.0, abs(f_ref)), (case, f_us, f_ref)
        assert np.max(np.abs(sol.u - u_ref)) <= 2e-4, (case, sol.u, u_ref)


def test_kkt_certificate_and_feasible_direction_probes():
    rng = np.random.default_rng(3)
    for K in (2, 6):
        for case in range(25):
            y, s, n, d = random_token(rng, K, int(rng.integers(1, 300)), kind=["dirichlet", "peaked", "flat"][case % 3])
            sol = DC.solve_token(y, s, n, d)
            c = sol.cert
            assert c["converged"] and c["stationarity_rel"] <= DC.STAT_TOL and c["dual_infeas_rel"] <= DC.DUAL_TOL
            assert c["projection_magnitude"] <= DC.PROJ_TOL and c["sum_q_residual"] <= DC.PROTO_SUM_TOL
            assert c["bracket_ulps"] <= 2.0
            worst = kkt_by_feasible_probes(np.array(sol.u), y, s, n, d)
            assert worst >= -1e-4 * (n + KAPPA), (case, worst)


@pytest.mark.parametrize("K", [2, 6])
def test_fuzz_extremes_class_constraint_on_actual_q(K):
    rng = np.random.default_rng(100 + K)
    cases = []
    for _ in range(150):
        cases.append(random_token(rng, K, int(rng.choice([1, 2, 3, 10, 1000, 15434])),
                                  kind=["dirichlet", "peaked", "flat"][int(rng.integers(3))]))
    # hand-made extremes: tiny teacher entries, exact teacher ties, n = 1 against the teacher, huge n
    for d in range(K):
        p = np.full(K, 1e-15)
        p[d] = 1 - (K - 1) * 1e-15
        cases.append((np.eye(K)[(d + 1) % K], p, 1, d))                     # one row labelled against a sure teacher
        cases.append((np.eye(K)[d] * 10 ** 6, p * 10 ** 6, 10 ** 6, d))
        pu = np.full(K, 1.0 / K)
        cases.append((np.full(K, 7.0), pu * 7 * K, 7 * K, d))                 # uniform teacher (exact ties in pbar)
        yy = np.zeros(K)
        yy[(d + 1) % K] = 5000
        cases.append((yy, p * 5000, 5000, d))                                 # all labels on another class
    Y = np.array([c[0] for c in cases], dtype=float)
    S = np.array([c[1] for c in cases], dtype=float)
    n = np.array([c[2] for c in cases])
    d = np.array([c[3] for c in cases])
    U, Q, obj, certs = DC.solve_batch(Y, S, n, d)
    rows = np.arange(len(cases))
    assert np.all(certs["converged"])
    assert np.all(U >= 0) and np.all(U[rows, d][:, None] >= U)            # class-dominant base vector, exactly
    other = Q.copy()
    other[rows, d] = -np.inf
    assert np.all(Q[rows, d] > other.max(1)) and np.all(certs["margin"] > 0)   # strict argmax d on the ACTUAL q
    assert np.max(np.abs(Q.sum(1) - 1)) <= DC.PROTO_SUM_TOL
    assert np.array_equal(Q, KM.smooth(U, d, check=True))                  # q is the registered smoothing of u
    # labels entirely against the teacher: the tie constraint binds (u_d = u_other), still argmax d
    tied = [i for i, c in enumerate(cases) if c[0][c[3]] == 0 and c[0].max() >= 5000]
    assert tied and all(certs["n_tie"][i] >= 1 for i in tied)


def test_batch_equals_scalar_bitwise():
    rng = np.random.default_rng(7)
    toks = [random_token(rng, 6, int(rng.integers(1, 200))) for _ in range(60)]
    Y = np.array([t[0] for t in toks])
    S = np.array([t[1] for t in toks])
    n = np.array([t[2] for t in toks])
    d = np.array([t[3] for t in toks])
    U, Q, obj, certs = DC.solve_batch(Y, S, n, d)
    perm = rng.permutation(len(toks))
    U2, Q2, obj2, _ = DC.solve_batch(Y[perm], S[perm], n[perm], d[perm])  # batch composition / order irrelevant
    assert np.array_equal(U2, U[perm]) and np.array_equal(Q2, Q[perm])
    for i in range(len(toks)):
        sol = DC.solve_token(Y[i], S[i], n[i], d[i])
        assert np.array_equal(sol.u, U[i]) and np.array_equal(sol.q, Q[i])
        assert abs(sol.obj - obj[i]) <= DC.TOL_BATCH_OBJ * max(1.0, abs(obj[i]))


def test_refusals():
    y, s = np.array([3.0, 1.0]), np.array([2.5, 1.5])
    with pytest.raises(ValueError):
        DC.solve_token(y, s, 0, 0)                                 # n = 0 -> fallback, not a solve
    with pytest.raises(ValueError):
        DC.solve_token(np.array([3.0, 2.0]), s, 4, 0)              # labels do not sum to n
    with pytest.raises(ValueError):
        DC.solve_token(np.array([2.5, 1.5]), s, 4, 0)              # non-integer labels
    with pytest.raises(ValueError):
        DC.solve_token(y, np.array([1.5, 2.5]), 4, 0)              # d is not the teacher class
    with pytest.raises(ValueError):
        DC.solve_token(y, s, 4, 0, kappa=16.0)                     # kappa fixed
    with pytest.raises(ValueError):
        DC.solve_token(y, s, 4, 0, eps=1e-10)                      # eps fixed


def test_projection_repairs_tiny_residuals_and_keeps_strict_argmax():
    d = 2
    u = np.array([0.2, 0.3 + 1e-13, 0.3, 0.2 - 1e-13 + 2e-16])      # u_1 exceeds u_d by a solver-size residual
    v, mag = DC.project_class_simplex(u, d)
    assert v[1] <= v[d] and np.all(v >= 0) and abs(v.sum() - 1) <= 4e-16 and mag < 1e-12
    q = KM.smooth(v, d, check=True)                                  # strict argmax holds after the repair
    assert q.argmax() == d
    v2, mag2 = DC.project_class_simplex(np.array([0.1, 0.6, 0.3]), 2)
    assert mag2 > DC.PROJ_TOL                                         # a large repair is reported (refused in solve)
    # exact tie u_d == u_k gives a strict q margin of eps / Z
    q = KM.smooth(np.array([0.5, 0.5]), 0, check=True)
    assert q[0] > q[1]


def test_convexity_numerical_midpoints():
    rng = np.random.default_rng(11)
    for K in (2, 6):
        y, s, n, d = random_token(rng, K, 50)
        for _ in range(200):
            u1 = DC.project_class_simplex(rng.dirichlet(np.ones(K)), d)[0]
            u2 = DC.project_class_simplex(rng.dirichlet(np.ones(K)), d)[0]
            f1, f2 = full_objective_closed(u1, y, s, n, d), full_objective_closed(u2, y, s, n, d)
            fm = full_objective_closed(0.5 * (u1 + u2), y, s, n, d)
            assert fm <= 0.5 * (f1 + f2) + 1e-9 * max(1, abs(fm))


def test_calibrated_teacher_null_no_population_improvement():
    """Teacher = true conditional and labels = the exact expectation (dyadic, so y = s exactly): D1's optimum is the
    mean within float tolerance, and no proper-loss improvement over the D0 mean exists in the population."""
    rows = [np.array([0.75, 0.25]), np.array([0.625, 0.375]), np.array([0.5625, 0.4375])]
    counts = [64, 128, 256]
    for K, base in ((2, rows), (6, [np.array([0.375, 0.125, 0.125, 0.125, 0.125, 0.125]),
                                    np.array([0.5, 0.25, 0.0625, 0.0625, 0.0625, 0.0625])])):
        P = np.vstack([np.repeat(p[None], c, 0) for p, c in zip(base, counts)])
        s = P.sum(0)
        y = s.copy()                                       # exact expected counts (integers by construction)
        assert np.all(y == np.round(y))
        n = P.shape[0]
        d = int(s.argmax())
        sol = DC.solve_token(y, s, n, d)
        q0 = KM.smooth(s / n, d)                           # D0 decoded vector
        assert np.max(np.abs(sol.q - q0)) <= 1e-10
        pbar = s / n                                       # the true conditional label law within the token
        pop_ll = lambda q: float(-np.sum(pbar * np.log(q)))          # noqa: E731
        pop_br = lambda q: float(np.sum(pbar * np.sum((q[None] - np.eye(K)) ** 2, 1)))  # noqa: E731
        assert pop_ll(sol.q) >= pop_ll(q0) - 1e-10
        assert pop_br(sol.q) >= pop_br(q0) - 1e-10


def test_miscalibrated_teacher_improves_fixed_token_fit():
    """Overconfident teacher (pbar = 0.9) but true rate 0.6: D1 moves toward the labels, keeps argmax d."""
    n = 1000
    s = np.array([900.0, 100.0])
    y = np.array([600.0, 400.0])
    sol = DC.solve_token(y, s, n, 0)
    q0 = KM.smooth(s / n, 0)
    assert sol.q[0] > sol.q[1] and sol.q[0] < q0[0]
    ll1, br1 = DC.token_losses(y[None], sol.q[None])
    ll0, br0 = DC.token_losses(y[None], q0[None])
    assert ll1[0] < ll0[0] and br1[0] < br0[0]
    # prior strength: the optimum lies between the empirical rate and the teacher mean
    assert 0.6 < sol.q[0] < 0.9


# ----------------------------------------------------------------------------------------------- cache
def test_cache_keys_and_no_stale_reuse():
    rng = np.random.default_rng(5)
    y, s, n, d = random_token(rng, 6, 80)
    cache = DC.TokenCache()
    a = cache.get_or_solve(y, s, n, d)
    b = cache.get_or_solve(y.copy(), s.copy(), n, d)
    assert b is a and cache.hits == 1 and cache.misses == 1
    y2 = y.copy()
    j = int(np.flatnonzero(y2 > 0)[0])
    y2[j] -= 1
    y2[(j + 1) % 6] += 1                                     # changed label counts, same n
    s2 = s.copy()
    s2[d] += 1e-13
    s2[(d + 1) % 6] -= 1e-13                                 # changed teacher sums by a roundoff-size amount
    for args in ((y2, s, n, d), (y, s2, n, d)):
        k0 = DC.cache_key(y, s, n, d)
        assert DC.cache_key(*args) != k0
        before = cache.misses
        r = cache.get_or_solve(*args)
        assert cache.misses == before + 1 and r is not a
        ref = DC.solve_token(*args)
        assert np.array_equal(r.q, ref.q)
    assert DC.cache_key(y, s, n, d) != DC.cache_key(y, s, n, d, kappa=16.0)
    assert DC.cache_key(np.array([0.0, 1.0]), np.array([0.2, 0.8]), 1, 1) == \
        DC.cache_key(np.array([-0.0, 1.0]), np.array([0.2, 0.8]), 1, 1)
    with pytest.raises(ValueError):
        a.q[0] = 1.0                                        # cached arrays are read-only
    # batch path: identical to fresh solves, independent of what was cached
    toks = [random_token(rng, 6, int(rng.integers(1, 90))) for _ in range(20)] + [(y, s, n, d)]
    Y, S = np.array([t[0] for t in toks]), np.array([t[1] for t in toks])
    nn, dd = np.array([t[2] for t in toks]), np.array([t[3] for t in toks])
    U1, Q1, _ = cache.solve_many(Y, S, nn, dd)
    U2, Q2, _, _ = DC.solve_batch(Y, S, nn, dd)
    assert np.array_equal(U1, U2) and np.array_equal(Q1, Q2)
    U3, Q3, _ = cache.solve_many(Y, S, nn, dd)
    assert np.array_equal(Q3, Q2) and cache.stats()["hits"] >= len(toks) + 1
    key = DC.cache_key(y, s, n, d)                           # a tampered entry is detected on its next hit
    cache._d[key] = DC.TokenSolution(a.u, a.q, a.obj, {**a.cert, "_key": b"other"})
    with pytest.raises(AssertionError):
        cache.get_or_solve(y, s, n, d)


# ----------------------------------------------------------------------------------------------- losses
def test_loss_helpers_match_row_level():
    rng = np.random.default_rng(9)
    for K in (2, 6):
        T = 12
        tok = rng.integers(0, T, 3000)
        y = rng.integers(0, K, 3000)
        Qm = rng.dirichlet(np.ones(K), size=T)
        Qm[0] = KM.smooth(np.eye(K)[0], 0)                   # includes entries at the eps floor (log-loss clip)
        Y = np.bincount(tok * K + y, minlength=T * K).reshape(T, K).astype(float)
        L, B = DC.fit_losses(Y, Qm, 3000)
        Lr, Br = DC.row_losses(Qm[tok], y)
        assert abs(L - Lr) <= 1e-12 * max(1, Lr) and abs(B - Br) <= 1e-12 * max(1, Br)
        ll, br = DC.token_losses(Y, Qm)
        for t in range(T):
            m = tok == t
            from dpc.utility import per_row
            pr = per_row(Qm[tok[m]], y[m], K)
            assert abs(ll[t] - pr["ll"].sum()) <= 1e-9 and abs(br[t] - pr["br"].sum()) <= 1e-9


# ----------------------------------------------------------------------------------------------- policies
def _synthetic_policy(rng, K, N, absent=None, recipient=1, m=4, merge=True):
    d_target = rng.integers(0, K, N)
    if absent is not None:
        d_target[d_target == absent] = (absent + 1) % K
    P = rng.dirichlet(np.ones(K), size=N)
    am = P.argmax(1)
    for i in range(N):
        P[i, [d_target[i], am[i]]] = P[i, [am[i], d_target[i]]]
    d = P.argmax(1)
    fit = KM.fit_recipient(P, d, K, m, starts=("source",))
    fine = fit.partition
    lab = np.arange(fine.F)
    if merge:                                                  # merge the first two cells of each class
        for c in range(K):
            idx = fine.cells_of(c)
            if idx.size >= 2:
                lab[idx[1]] = idx[0]
    pol = RL.make_policy(recipient, fine, lab, family="SYN")
    y = np.array([rng.choice(K, p=0.5 * p + 0.5 * np.eye(K)[(k + 1) % K]) for p, k in zip(P, d)])
    return pol, P, d, y


def test_decode_policy_fallback_validate_roundtrip(tmp_path):
    rng = np.random.default_rng(21)
    pol, P, d, y = _synthetic_policy(rng, 6, 900, absent=3)
    tok, q0, _ = RL.encode(pol, P, d)
    tab = DC.decode_policy(pol, tok, P, y, config="U|FINE-TASK|i8o64|D1")
    fb = np.flatnonzero(tab.fallback)
    assert fb.size == 1 and tab.token_class[fb[0]] == 3 and tab.n[fb[0]] == 0
    assert np.array_equal(tab.q[fb], pol.token_proto[fb])                # pinned D0 vector, bitwise
    assert np.all(tab.y[fb] == 0)                                        # no labels invented
    assert np.array_equal(tab.s, pol.token_S) and np.array_equal(tab.n, pol.token_n)
    tab.validate(pol, verify_solve=True)
    z = json.loads(json.dumps(tab.to_dict(), allow_nan=False))
    tab2 = DC.DecoderTable.from_dict(z)
    assert tab2.content_hash() == tab.content_hash() and np.array_equal(tab2.q, tab.q)
    z["q"][0][0] += 1e-9
    with pytest.raises(ValueError):
        DC.DecoderTable.from_dict(z)
    # wrong labels (stale statistics) are refused by the re-solve check
    z = tab.to_dict()
    sup = int(np.flatnonzero(~tab.fallback)[0])
    yy = np.array(z["y"])
    k1 = int(np.flatnonzero(yy[sup] > 0)[0])
    yy[sup, k1] -= 1
    yy[sup, (k1 + 1) % 6] += 1
    z["y"] = yy.tolist()
    t3 = DC.DecoderTable(recipient=1, K=6, token_class=tab.token_class, n=tab.n, y=yy, s=tab.s, u=tab.u, q=tab.q,
                         fallback=tab.fallback, certs=tab.certs, policy_fingerprint=tab.policy_fingerprint)
    with pytest.raises(ValueError):
        t3.validate(pol, verify_solve=True)
    with pytest.raises(ValueError):
        DC.decode_policy(pol, np.roll(tok, 1), P, y)                    # misaligned tokens refused
    # the label counts are exactly those of the rows
    for t in range(pol.T):
        assert np.array_equal(tab.y[t], np.bincount(y[tok == t], minlength=6).astype(float))


def test_token_stats_canonical_and_cell_counts():
    rng = np.random.default_rng(22)
    pol, P, d, y = _synthetic_policy(rng, 6, 700)
    Yc = DC.cell_label_counts(pol.fine, P, d, y)
    n, S, Yt = DC.token_stats(pol.fine, pol.cell_token, Yc)
    tok, _, _ = RL.encode(pol, P, d)
    assert np.array_equal(n, pol.token_n) and np.array_equal(S, pol.token_S)       # bitwise = token_tables order
    assert np.array_equal(Yt, np.bincount(tok * 6 + y, minlength=pol.T * 6).reshape(pol.T, 6))


def test_release_arrays_d1_keys_invariants_and_fixed_token_information():
    rng = np.random.default_rng(23)
    p1, P1, d1, y1 = _synthetic_policy(rng, 2, 600, recipient=1)
    rng2 = np.random.default_rng(24)
    p2, P2, d2, y2 = _synthetic_policy(rng2, 6, 600, recipient=2)
    meta = {"teacher_model_sha256": "a" * 64, "feature_names_sha256": "b" * 64, "config": "U|FINE-TASK|i8o64"}
    pair = RL.make_pair(p1, p2, "FINE-TASK", 8, 64, meta=meta)
    t1, _, _ = RL.encode(pair.p1, P1, d1)
    t2, _, _ = RL.encode(pair.p2, P2, d2)
    dec1 = DC.decode_policy(pair.p1, t1, P1, y1)
    dec2 = DC.decode_policy(pair.p2, t2, P2, y2)
    rel = DC.release_arrays_d1(pair, dec1, dec2, np.arange(600), P1, d1, P2, d2)
    assert list(rel) == ["row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"]
    rel0 = RL.release_arrays(pair, np.arange(600), P1, d1, P2, d2)
    assert list(rel0) == list(rel)
    for i in (1, 2):
        assert np.array_equal(rel[f"tok{i}"], rel0[f"tok{i}"])           # identical tokens
        assert np.array_equal(rel[f"hard{i}"], rel0[f"hard{i}"])         # identical decisions = teacher
        assert int(rel[f"alpha{i}"]) == int(rel0[f"alpha{i}"])
    assert not np.array_equal(rel["q2"], rel0["q2"])                     # the decoder changed the vectors
    # full-token information: q is a deterministic function of the token, so (token, q) and token induce the same
    # partition of the rows, for D0 and D1 alike -> identical plug-in MI with any variable (here a synthetic SEX)
    from dpc.compress import mi_plugin
    sex = rng.integers(0, 2, 600)
    for i in (1, 2):
        joint_d1 = np.unique(np.column_stack([rel[f"tok{i}"], rel[f"q{i}"]]), axis=0, return_inverse=True)[1]
        joint_d0 = np.unique(np.column_stack([rel0[f"tok{i}"], rel0[f"q{i}"]]), axis=0, return_inverse=True)[1]
        mi_t = mi_plugin(sex, rel[f"tok{i}"])
        assert mi_plugin(sex, joint_d1.ravel()) == pytest.approx(mi_t, abs=1e-15)
        assert mi_plugin(sex, joint_d0.ravel()) == pytest.approx(mi_t, abs=1e-15)
    # decoder.json round trip and binding
    body = DC.decoder_pair_dict("U|FINE-TASK|i8o64|D1", pair, dec1, dec2)
    cid, e1, e2, sha = DC.load_decoder_pair(json.loads(json.dumps(body)), pair)
    assert cid == "U|FINE-TASK|i8o64|D1" and sha == body["decoder_sha256"]
    with pytest.raises(ValueError):
        DC.decoder_pair_dict("U|FINE-TASK|i8o64", pair, dec1, dec2)      # not a D1 id
    bad = json.loads(json.dumps(body))
    bad["r2"]["q"][0][0] += 1e-6
    with pytest.raises(ValueError):
        DC.load_decoder_pair(bad, pair)
    # a decoder whose released vector loses strict argmax is refused at release time
    dec_bad = DC.decode_policy(pair.p2, t2, P2, y2)
    t = int(np.flatnonzero(~dec_bad.fallback)[0])
    c = int(dec_bad.token_class[t])
    dec_bad.q = dec_bad.q.copy()
    dec_bad.q[t, (c + 1) % 6] = dec_bad.q[t, c]
    with pytest.raises((ValueError, AssertionError)):
        DC.release_arrays_d1(pair, dec1, dec_bad, np.arange(600), P1, d1, P2, d2)
    # a decoder of another policy is refused
    with pytest.raises((ValueError, AssertionError)):
        DC.release_arrays_d1(pair, dec2, dec1, np.arange(600), P1, d1, P2, d2)
