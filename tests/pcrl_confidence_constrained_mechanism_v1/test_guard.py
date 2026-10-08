"""Synthetic tests of ccm.guard (role C; no real data, no labels, no SEX, no TOY_LAWS.json).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q tests/pcrl_confidence_constrained_mechanism_v1/test_guard.py

Independent re-checks below use plain Python loops and math.log (not the vectorised code under test).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ccm import guard as GD

D, B = GD.CONFIG["d"], GD.CONFIG["b"]


def literal_G(q, p, dec):
    """Independent loop implementation of G: returns (ok, list of reasons)."""
    K = len(p)
    bad = []
    for k in range(K):
        if not q[k] >= math.exp(-D) * p[k]:
            bad.append(("nll", k))
    sq = sum(float(v) * float(v) for v in q) - sum(float(v) * float(v) for v in p)
    for y in range(K):
        if not sq - 2.0 * (q[y] - p[y]) <= B:
            bad.append(("brier", y))
    for k in range(K):
        if k != dec and not q[dec] > q[k]:
            bad.append(("class", k))
    if not (all(math.isfinite(v) and v >= 0 for v in q) and abs(sum(q) - 1.0) <= 1e-12):
        bad.append(("simplex",))
    return not bad, bad


def literal_Gexp(q, p, dec):
    kl = sum(pk * (math.log(pk) - math.log(qk)) if pk > 0 else 0.0 for pk, qk in zip(p, q) if not (pk > 0 and qk == 0))
    if any(pk > 0 and qk == 0 for pk, qk in zip(p, q)):
        kl = math.inf
    sq = sum((qk - pk) ** 2 for pk, qk in zip(p, q))
    cls = all(q[dec] > q[k] for k in range(len(p)) if k != dec)
    simp = all(v >= 0 for v in q) and abs(sum(q) - 1.0) <= 1e-12
    return kl <= D and sq <= B and cls and simp


def test_config_registered_and_capacity():
    assert GD.CONFIG["d"] == 0.005 and GD.CONFIG["b"] == 0.0025
    assert GD.CONFIG["capacity_per_class"] == {"income": 8, "occupation": 64}
    assert GD.capacity_for("income") == 8 and GD.capacity_for(K=6) == 64 and GD.capacity_for(K=2) == 8
    with pytest.raises(ValueError):
        GD.capacity_for(K=3)


def test_brier_every_label_matches_squared_distance_identity():
    rng = np.random.default_rng(0)
    for _ in range(300):
        K = int(rng.integers(2, 7))
        p = rng.dirichlet(np.ones(K))
        q = rng.dirichlet(np.ones(K))
        out = GD.check_release(q, p)
        for y in range(K):
            e = np.eye(K)[y]
            lit = float(np.sum((q - e) ** 2) - np.sum((p - e) ** 2))
            assert abs(out["brier_excess"][y] - lit) <= 1e-12
        exc = [sum(q * q) - sum(p * p) - 2 * (q[y] - p[y]) for y in range(K)]
        assert bool(out["brier"]) == all(v <= B for v in exc)
        ok, _ = literal_G(q.tolist(), p.tolist(), int(np.argmax(p)))
        assert bool(out["ok"]) == ok


def test_nll_condition_implies_every_label_and_clipped_log_loss():
    rng = np.random.default_rng(1)
    n_pass = 0
    for _ in range(2000):
        K = int(rng.integers(2, 7))
        p = rng.dirichlet(np.ones(K) * 0.5)
        q = p * np.exp(rng.uniform(-1.2 * D, 0.3 * D, K))
        q = q / q.sum()
        out = GD.check_release(q, p)
        if out["nll"]:
            n_pass += 1
            for y in range(K):
                if p[y] > 0:
                    assert math.log(p[y]) - math.log(q[y]) <= D + 1e-12
                lp = math.log(max(p[y], 1e-12))
                lq = math.log(max(q[y], 1e-12))
                assert -lq <= -lp + D + 1e-12          # clipped-score log loss of label y
    assert n_pass > 200


def test_zero_probability_coordinates():
    p = np.array([0.7, 0.3, 0.0])
    assert GD.check_release(np.array([0.697, 0.303, 0.0]), p)["nll"]
    assert not GD.check_release(np.array([0.69, 0.31, 0.0]), p)["nll"]           # 0.69 < e^-d 0.7
    assert not GD.check_release(np.array([0.7, 0.3, 0.0]), np.array([0.7, 0.29, 0.01]))["nll"]
    assert GD.kl_div(p, np.array([0.69, 0.31, 0.0])) < 1e-3          # 0 log 0 = 0
    assert np.isinf(GD.kl_div(np.array([0.7, 0.29, 0.01]), np.array([0.7, 0.3, 0.0])))
    assert not GD.check_release_exp(np.array([0.7, 0.3, 0.0]), np.array([0.7, 0.29, 0.01]))["ok"]
    q, cert = GD.bin_representative(np.array([[0.7, 0.3, 0.0], [0.701, 0.299, 0.0]]))
    assert cert["status"] == "CERTIFIED" and literal_G(q.tolist(), [0.7, 0.3, 0.0], 0)[0]


def test_ties_and_strict_class():
    p = np.array([0.4, 0.4, 0.2])
    assert int(GD.decisions(p)) == 0                                  # first-index tie rule
    out = GD.check_release(p, p)
    assert out["nll"] and out["brier"] and not out["class"] and not out["ok"]
    assert not GD.check_release(p, p, dec=1)["class"]
    q, cert = GD.bin_representative(p)                                # NLL slack breaks the tie toward dec 0
    assert cert["status"] == "CERTIFIED" and q[0] > q[1] and literal_G(q.tolist(), p.tolist(), 0)[0]
    with pytest.raises(ValueError):
        GD.bin_representative(np.array([[0.6, 0.4], [0.4, 0.6]]))     # mixed decisions without dec


def test_simplex_and_finiteness():
    p = np.array([0.6, 0.4])
    assert GD.check_release(np.array([0.6 + 5e-13, 0.4]), p)["simplex"]
    assert not GD.check_release(np.array([0.6 + 2e-12, 0.4]), p)["simplex"]
    assert not GD.check_release(np.array([1.1, -0.1]), p)["simplex"]
    assert not GD.check_release(np.array([np.nan, 0.4]), p)["ok"]
    assert not GD.check_release_exp(np.array([np.inf, 0.4]), p)["ok"]


def test_three_members_pairwise_overlap_but_no_common_q():
    h = 0.003                         # pairwise sum max = 1 + h <= e^d < 1 + 2h = triple sum max
    c = (0.5 - h) / 3
    T = np.array([[0.5, c + h, c, c], [0.5, c, c + h, c], [0.5, c, c, c + h]])
    assert 1 + h <= math.exp(D) < 1 + 2 * h
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert GD.nll_necessary(T[[i, j]])[0]
        q, cert = GD.bin_representative(T[[i, j]])
        assert cert["status"] == "CERTIFIED"
        third = ({0, 1, 2} - {i, j}).pop()
        assert not literal_G(q.tolist(), T[third].tolist(), 0)[0]   # the pair's q never serves the third
    q, cert = GD.bin_representative(T)
    assert q is None and cert["status"] == "INFEASIBLE_NLL" and cert["nll_sum"] > math.exp(D)


def test_nll_feasible_but_brier_infeasible_is_not_found_never_infeasible():
    P = np.array([[0.6, 0.4], [0.604, 0.396]])
    assert GD.nll_necessary(P)[0]
    q, cert = GD.bin_representative(P)
    assert q is None and cert["status"] == "NOT_FOUND"
    a = np.linspace(0.55, 0.65, 200001)                               # independent K=2 grid: no common q exists
    Q = np.stack([a, 1 - a], axis=1)
    ok = GD.check_release(Q[:, None, :], P[None, :, :], dec=0)["ok"].all(axis=1)
    assert not ok.any()
    q, cert = GD.bin_representative(np.array([[0.6, 0.4], [0.602, 0.398]]))
    assert cert["status"] == "CERTIFIED"


def test_empty_and_single_member_bins():
    with pytest.raises(ValueError):
        GD.bin_representative(np.zeros((0, 3)))
    rng = np.random.default_rng(2)
    for _ in range(50):
        p = rng.dirichlet(np.ones(int(rng.integers(2, 7))))
        for con in ("G", "G_exp"):
            q, cert = GD.bin_representative(p, contract=con)
            assert cert["status"] == "CERTIFIED" and np.max(np.abs(q - p)) < 1e-6
        q, cert = GD.bin_representative(p)
        assert cert["worst"]["brier"] <= -B + 1e-9                  # q = p is the min-max point: excess 0


def test_identical_reference_vectors_always_mergeable():
    rng = np.random.default_rng(3)
    for _ in range(30):
        p = rng.dirichlet(np.ones(int(rng.integers(2, 7))))
        P = np.repeat(p[None, :], 40, axis=0)
        for con in ("G", "G_exp"):
            q, cert = GD.bin_representative(P, contract=con)
            assert cert["status"] == "CERTIFIED"


def _bin_around(rng, qs, scale, n, contract, frac):
    dec = int(np.argmax(qs))
    out = []
    while len(out) < n:
        p = qs * np.exp(rng.normal(0, scale, len(qs)))
        p = p / p.sum()
        if np.argmax(p) != dec:
            continue
        chk = GD.check(qs, p, contract, D * frac, B * frac, dec)
        if chk["ok"]:
            out.append(p)
    return np.array(out), dec


@pytest.mark.parametrize("contract,scale", [("G", 0.004), ("G_exp", 0.08)])
def test_solver_completeness_on_bins_feasible_by_construction(contract, scale):
    """Members generated so that a KNOWN q* serves all of them at 90% strength: the solver must certify."""
    rng = np.random.default_rng(4)
    for it in range(40):
        K = int(rng.integers(2, 7))
        qs = rng.dirichlet(np.ones(K) * 2)
        P, dec = _bin_around(rng, qs, scale, int(rng.integers(2, 30)), contract, 0.9)
        q, cert = GD.bin_representative(P, dec=dec, contract=contract)
        assert cert["status"] == "CERTIFIED", (it, cert)
        lit = literal_G if contract == "G" else (lambda a, b_, c: (literal_Gexp(a, b_, c), None))
        for p in P:
            assert lit(q.tolist(), p.tolist(), dec)[0]


def test_certificates_independently_rechecked_and_infeasibility_only_closed_form():
    rng = np.random.default_rng(5)
    seen = set()
    for _ in range(400):
        K = int(rng.integers(2, 7))
        c = rng.dirichlet(np.ones(K))
        m = int(rng.integers(2, 6))
        P = c * np.exp(rng.normal(0, rng.choice([0.002, 0.01, 0.05]), (m, K)))
        P = P / P.sum(axis=1, keepdims=True)
        dec = GD.decisions(P)
        if len(set(dec.tolist())) != 1:
            continue
        q, cert = GD.bin_representative(P)
        seen.add(cert["status"])
        if cert["status"] == "CERTIFIED":
            for p in P:
                assert literal_G(q.tolist(), p.tolist(), int(dec[0]))[0]
        elif cert["status"] == "INFEASIBLE_NLL":
            assert float(np.sum(P.max(axis=0))) > math.exp(D)
        else:
            assert cert["status"] == "NOT_FOUND" and GD.nll_necessary(P)[0]
    assert {"CERTIFIED", "INFEASIBLE_NLL"} <= seen


def test_certified_q_gives_clipped_log_loss_bound_for_every_member_and_label():
    rng = np.random.default_rng(6)
    for _ in range(100):
        K = int(rng.integers(2, 7))
        qs = rng.dirichlet(np.ones(K) * 0.7)
        P, dec = _bin_around(rng, qs, 0.003, 4, "G", 0.9)
        q, cert = GD.bin_representative(P, dec=dec)
        assert cert["status"] == "CERTIFIED"
        for p in P:
            for y in range(K):
                assert -math.log(max(q[y], 1e-12)) <= -math.log(max(p[y], 1e-12)) + D + 1e-12


def test_g_exp_closed_form_bounds_are_valid_and_certify_infeasibility():
    rng = np.random.default_rng(7)
    for _ in range(200):
        K = int(rng.integers(2, 6))
        P = rng.dirichlet(np.ones(K) * 3, size=int(rng.integers(2, 6)))
        v = GD.exp_bounds(P)
        for _ in range(5):
            q = rng.dirichlet(np.ones(K))
            assert np.max(GD.kl_div(P, q[None, :])) >= v["gjs"] - 1e-12
            assert np.max(np.sum((P - q) ** 2, axis=1)) >= v["var"] - 1e-12
    far = np.array([[0.9, 0.1], [0.6, 0.4]])
    q, cert = GD.bin_representative(far, contract="G_exp")
    assert q is None and cert["status"] == "INFEASIBLE_BOX"
    near = np.array([[0.70, 0.30], [0.72, 0.28], [0.69, 0.31]])
    q, cert = GD.bin_representative(near, contract="G_exp")
    assert cert["status"] == "CERTIFIED" and all(literal_Gexp(q.tolist(), p.tolist(), 0) for p in near)


def test_determinism():
    rng = np.random.default_rng(8)
    P = rng.dirichlet(np.ones(4), size=1)[0] * np.exp(rng.normal(0, 0.002, (5, 4)))
    P = P / P.sum(axis=1, keepdims=True)
    for con in ("G", "G_exp"):
        q1, _ = GD.bin_representative(P, contract=con)
        q2, _ = GD.bin_representative(P, contract=con)
        assert q1 is not None and np.array_equal(q1, q2)


def test_canonical_predicate_is_the_docstring_formula():
    """B2: check_release IS E = np.exp(-d); q >= E*p; np.sum(q*q,-1) - np.sum(p*p,-1) - 2*(q - p) <= b; strict class;
    q >= 0 and |sum q - 1| <= 1e-12 (bitwise-identical verdicts on random and boundary inputs)."""
    rng = np.random.default_rng(9)
    E = np.exp(-D)
    for _ in range(300):
        K = int(rng.integers(2, 7))
        p = rng.dirichlet(np.ones(K), size=20)
        q = p * np.exp(rng.uniform(-1.1 * D, 0.2 * D, (20, K)))
        q[:10] = E * p[:10]                                       # NLL-tight rows
        q = q / q.sum(axis=1, keepdims=True)
        dec = np.argmax(p, axis=1)
        nll = np.all(q >= E * p, axis=-1)
        br = np.all(np.sum(q * q, -1)[:, None] - np.sum(p * p, -1)[:, None] - 2 * (q - p) <= B, axis=-1)
        cls = np.array([all(q[i, dec[i]] > q[i, k] for k in range(K) if k != dec[i]) for i in range(20)])
        simp = np.all(q >= 0, axis=-1) & (np.abs(np.sum(q, -1) - 1) <= 1e-12)
        out = GD.check_release(q, p)
        assert np.array_equal(out["nll"], nll) and np.array_equal(out["brier"], br)
        assert np.array_equal(out["class"], cls) and np.array_equal(out["simplex"], simp)
        assert np.array_equal(out["ok"], nll & br & cls & simp)
    assert GD.nll_constant() == np.exp(-D)


def test_construction_margin_makes_every_float_form_agree():
    """B2.2: certified representatives satisfy the tightened targets, so the multiplicative, ratio, log and clipped NLL
    forms and the dot, per-row and fsum Brier forms all pass at the true d and b."""
    rng = np.random.default_rng(10)
    n_bins = 0
    for _ in range(150):
        K = int(rng.integers(2, 7))
        qs = rng.dirichlet(np.ones(K))
        P, dec = _bin_around(rng, qs, 0.003, int(rng.integers(1, 6)), "G", 0.95)
        q, cert = GD.bin_representative(P, dec=dec)
        assert cert["status"] == "CERTIFIED"
        assert np.all(GD.check_construction(q, P, "G", dec=dec))
        n_bins += 1
        for p in P:
            for k in range(K):
                assert q[k] >= np.exp(-D) * p[k]
                if p[k] > 0:
                    assert p[k] / q[k] <= math.exp(D)
                    assert math.log(p[k]) - math.log(q[k]) <= D
                assert -math.log(min(1.0, max(q[k], 1e-12))) <= -math.log(min(1.0, max(p[k], 1e-12))) + D
            for y in range(K):
                e = np.eye(K)[y]
                forms = [q @ q - p @ p - 2 * (q[y] - p[y]),
                         float(np.sum((q - e) ** 2) - np.sum((p - e) ** 2)),
                         math.fsum(list(q * q) + list(-p * p) + [-2 * q[y], 2 * p[y]])]
                assert all(v <= B for v in forms)
            assert q[dec] - max(q[k] for k in range(K) if k != dec) >= 1e-9
    assert n_bins == 150


def test_closed_form_infeasibility_is_conservative():
    """B2.3: INFEASIBLE_NLL only if sum max > exp(d)(1 + 1e-12)."""
    bound = math.exp(D) * (1 + 1e-12)
    assert GD.nll_bound() == bound
    a = 0.6
    x = (math.exp(D) - 1.0) / 2.0                     # binary pair with sum max = 1 + 2x = e^d (up to rounding)
    P = np.array([[a + x, 1 - a - x], [a - x, 1 - a + x]])
    ok, v = GD.nll_necessary(P)
    assert abs(v - math.exp(D)) < 1e-12 and ok
    P2 = np.array([[a + x + 1e-9, 1 - a - x - 1e-9], [a - x, 1 - a + x]])
    assert not GD.nll_necessary(P2)[0]
    q, cert = GD.bin_representative(P2)
    assert cert["status"] == "INFEASIBLE_NLL"


def test_fallback_release_strict_rows_unchanged_tied_rows_nudged():
    rng = np.random.default_rng(11)
    P = rng.dirichlet(np.ones(4), size=50)
    P[:5] = [0.4, 0.4, 0.1, 0.1]
    P[5:8] = [0.3, 0.3, 0.3, 0.1]
    P[8] = [0.25, 0.25, 0.25, 0.25]
    dec = np.argmax(P, axis=1)
    assert int(GD.tied_top(P).sum()) == 9
    Q, info = GD.fallback_release(P, dec)
    assert info["n_tied"] == 9 and info["n_fail_G"] == 0
    assert np.array_equal(Q[9:], P[9:])                                   # strict rows: Ucal itself, bitwise
    assert np.all(GD.check_release(Q, P, dec=dec)["ok"])
    assert np.allclose(Q[:9], (1 - 1e-6) * P[:9] + 1e-6 * np.eye(4)[dec[:9]], atol=0, rtol=1e-15)
    assert not GD.check_release(P[:9], P[:9], dec=dec[:9])["class"].any()  # q = p itself fails strict class (B1)
    Qm, _ = GD.fallback_release(P[:9], dec[:9], eta=GD.ETA_MAX)            # R3.3 bound still admissible
    assert np.all(GD.check_release(Qm, P[:9], dec=dec[:9])["ok"])
    with pytest.raises(ValueError):
        GD.fallback_release(P, dec, eta=0.003)
    with pytest.raises(ValueError):
        GD.fallback_release(np.array([[0.6, 0.4]]), np.array([1]))         # decision not a top index
