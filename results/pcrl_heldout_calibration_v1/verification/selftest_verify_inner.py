"""Synthetic self-test of role E's verify_inner.py (no real data, no labels, no hcal import).

Checks the verifier's own numerics against independent references before it is trusted on the study outputs:
  AUC (own Mann-Whitney with average ranks) vs sklearn roc_auc_score, with ties; CE with the 1e-12 clip;
  the token32 KKT certificate accepts lra.decoder's certified solution (K = 2, 3, 6; prior given as S = mu n) and
  rejects perturbed or infeasible vectors; the temperature solve hits the K = 2 closed form logit(f) / log(q0 ratio) and
  the boundaries; within-bank selection and the union keep the earlier bank on ties within 1e-12; the T* / P* ranking
  applies the inclusive guard and benefit gates.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python <WT>/results/pcrl_heldout_calibration_v1/verification/selftest_verify_inner.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
WT = HERE.parents[3]
sys.path.insert(0, str(WT))
spec = importlib.util.spec_from_file_location("verify_inner", HERE.parent / "verify_inner.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def test_auc_ce():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(50, 3000))
        y = rng.integers(0, 2, n)
        s = np.round(rng.random(n), int(rng.integers(1, 6)))            # many ties
        assert abs(V.auc_fixed(y, s) - roc_auc_score(y == 1, s)) <= 1e-12
        assert abs(V.auc_fixed(y, 1 - s) - (1 - V.auc_fixed(y, s))) <= 1e-12   # orientation is fixed, never flipped
    p = np.array([0.0, 1.0, 0.3])
    y = np.array([1, 0, 1])
    want = -np.mean(np.log(np.clip([0.0, 0.0, 0.3], 1e-12, 1)))
    assert abs(V.ce_fixed(y, p) - want) <= 1e-15


def test_token32_kkt():
    from lra import decoder as DC
    rng = np.random.default_rng(1)
    worst = 0.0
    for K in (2, 3, 6):
        for _ in range(60):
            d = int(rng.integers(K))
            n_prior = int(rng.integers(1, 300))
            P = rng.dirichlet(np.ones(K) * rng.choice([0.05, 0.7, 50.0]), size=n_prior)
            am = P.argmax(1)
            for i in range(n_prior):
                P[i, [d, am[i]]] = P[i, [am[i], d]]
            mu = P.mean(0)
            n = int(rng.integers(1, 120))
            y = np.bincount(rng.integers(0, K, n), minlength=K).astype(float)
            if rng.random() < 0.3:
                y = np.zeros(K)
                y[(d + 1) % K] = n                                     # labels all on another class: tie active
            S = mu * n
            U, Q, _, _ = DC.solve_batch(y[None], S[None], np.array([n]), np.array([d]))
            pbar = S / n
            gap, feas, marg = V.token32_kkt(Q[0], y, pbar, d, K)
            worst = max(worst, gap)
            assert gap <= V.KKT_REL_TOL and feas <= 1e-12 and marg > 0, (K, gap, feas, marg)
            # a feasible perturbation towards a different vertex (e_d, or the uniform vector) is detected
            u = Q[0] * (1 + (K + 1) * V.EPS) - V.EPS - V.EPS * np.eye(K)[d]
            v = np.eye(K)[d] if np.abs(u - np.eye(K)[d]).max() > 1e-3 else np.full(K, 1.0 / K)
            u2 = 0.99 * u + 0.01 * v
            g2, f2, _ = V.token32_kkt(V.smooth(u2, d, K), y, pbar, d, K)
            assert f2 <= 1e-12 and g2 > 1e-7, (K, g2)
            assert V.token32_objective(Q[0], y, pbar) <= V.token32_objective(V.smooth(u2, d, K), y, pbar)
    return worst


def test_temperature():
    q0 = V.smooth(np.array([0.8, 0.2]), 0, 2)[None]
    for f, st in ((0.9, "INTERIOR"), (0.999, "BOUNDARY_HIGH"), (0.55, "BOUNDARY_LOW")):
        n1 = int(round(1000 * f))
        y = np.r_[np.zeros(n1, int), np.ones(1000 - n1, int)]
        L = np.log(q0[np.zeros(1000, int)])
        a, s, _, _ = V.solve_temp(L, y)
        assert s == st, (f, s)
        if st == "INTERIOR":
            want = np.log(f / (1 - f)) / np.log(q0[0, 0] / q0[0, 1])
            assert abs(a - want) <= 1e-12, (a, want)
            chk = V.Check("t", "t")
            V.certify_alpha(L, y, want, "INTERIOR", chk, "t")
            assert chk.nfail == 0, chk.fail
            chk = V.Check("t", "t")
            V.certify_alpha(L, y, want + 1e-6, "INTERIOR", chk, "t")
            assert chk.nfail > 0                                       # a wrong alpha fails the KKT check
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.ones(6) * 0.5, size=500)
    y = np.array([rng.choice(6, p=p ** 2 / (p ** 2).sum()) for p in P])
    L = np.log(P)
    a, s, _, _ = V.solve_temp(L, y)
    grid = np.linspace(0.25, 4, 2001)
    nll = np.array([V.temp_terms(L, y, x)[0] for x in grid])
    assert V.temp_terms(L, y, a)[0] <= nll.min() + 1e-12 and abs(V.temp_terms(L, y, a)[1]) <= 1e-12
    assert np.array_equal(V.temp_table(P, 1.0), P)


def test_selection_and_union():
    rng = np.random.default_rng(3)
    n = 400
    ys = rng.integers(0, 2, n)
    keys = ["v1:A", "v1:B", "v2:A", "v2:B", "pair:A", "pair:B"]
    base = rng.random(n)
    P = np.stack([base, base.copy(), rng.random(n), rng.random(n), base.copy(), rng.random(n)])
    chk = V.Check("s", "s")
    # rows for pair: coalition (pair:A, pair:B), ignore_recipient_2 (v1:A, v1:B), ignore_recipient_1 (v2:A, v2:B);
    # v1:A, v1:B and pair:A carry identical predictions: the first in bank order must win every tie
    rows = V.bank_lists(keys)
    assert [r[0] for r in rows["pair"]] == ["coalition"] * 2 + ["ignore_recipient_2"] * 2 + ["ignore_recipient_1"] * 2
    V._ENTRY.clear()
    stored_sel = {}
    # build SEL arrays equal to the expected selected key's predictions
    Pk = dict(zip(keys, P))
    expect = {}
    for w in V.VIEWS:
        r = rows[w]
        a = [V.auc_fixed(ys, Pk[x[3]]) for x in r]
        c = [V.ce_fixed(ys, Pk[x[3]]) for x in r]
        ia = int(np.argmax(a))                                          # numpy argmax = first maximum
        ic = int(np.argmin(c))
        expect[("auc", w)], expect[("ce", w)] = r[ia][3], r[ic][3]
        stored_sel[("auc", w)] = np.stack([Pk[r[ia][3]]] * 3)
        stored_sel[("ce", w)] = np.stack([Pk[r[ic][3]]] * 3)
    e = V.replay_family("syn", keys, P, stored_sel, ys, chk)
    assert chk.nfail == 0, chk.fail
    for kk, v in expect.items():
        assert e["pred_key"][kk] == v, (kk, e["pred_key"][kk], v)
    # union: ties within 1e-12 keep the earlier bank; > 1e-12 switches
    mk = lambda a, c=0.6: {"auc": {w: a for w in V.VIEWS}, "ce": {w: c for w in V.VIEWS},  # noqa: E731
                           "auc_seed0": {w: a for w in V.VIEWS}, "ce_seed0": {w: c for w in V.VIEWS},
                           "auc_per_seed": {w: [a] * 3 for w in V.VIEWS}, "ce_per_seed": {w: [c] * 3 for w in V.VIEWS}}
    u = V.union([("L0", mk(0.7)), ("L1", mk(0.7 + 1e-9)), ("F", mk(0.7 + 1e-9 + 5e-13, 0.6 - 1e-9))])
    assert all(u["winner"][w] == "L1" for w in V.VIEWS) and all(u["ce_winner"][w] == "F" for w in V.VIEWS)
    u = V.union([("L0", mk(0.7)), ("L1", mk(0.7)), ("F", mk(0.7))])
    assert all(u["winner"][w] == "L0" for w in V.VIEWS) and all(u["ce_winner"][w] == "L0" for w in V.VIEWS)


def test_rank_gates():
    V.TABLE.clear()
    V.REC.clear()
    t_rid, p_ok, p_guard, p_ben = "U|FINE-TASK|i8o64|H-GLOBAL-TEMP", "U|JOINT|i8o64|l0.1|H-TOKEN32", \
        "U|JOINT|i8o64|l0.08|D1", "U|JOINT|i8o64|l0.06"
    for rid in (t_rid, p_ok, p_guard, p_ben):
        V.TABLE[rid] = {"eligible_all_seeds": True, "worst_norm_excess": 0.1, "mean_summed_nll": 1.6}
    for k in V.SEEDS:
        V.REC[(k, "U|FINE-TASK|i8o64")] = {"auc": {"v1": 0.70, "v2": 0.80, "pair": 0.85}}
        V.REC[(k, "U|JOINT|i8o64|l0.1")] = {"auc": {"v1": 0.70 + 0.005, "v2": 0.80, "pair": 0.83}}   # inclusive guard
        V.REC[(k, "U|JOINT|i8o64|l0.08")] = {"auc": {"v1": 0.70 + 0.0051, "v2": 0.80, "pair": 0.80}}
        V.REC[(k, "U|JOINT|i8o64|l0.06")] = {"auc": {"v1": 0.70, "v2": 0.80, "pair": 0.85 - 0.019}}
    parts = ["U|FINE-TASK|i8o64", "U|JOINT|i8o64|l0.1", "U|JOINT|i8o64|l0.08", "U|JOINT|i8o64|l0.06"]
    t, _ = V.rank([t_rid], audited=parts)
    rows, rej = V.rank([p_ok, p_guard, p_ben], ref=t[0]["auc_per_seed"], audited=parts)
    assert [r["release"] for r in rows] == [p_ok], rows
    assert dict(rej) == {p_guard: "LOCAL_GUARD_FAILURE", p_ben: "PAIR_BENEFIT_BELOW_0.02"}, rej


if __name__ == "__main__":
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    test_auc_ce()
    w = test_token32_kkt()
    test_temperature()
    test_selection_and_union()
    test_rank_gates()
    loaded = sorted(m for m in sys.modules if m == "hcal" or m.startswith("hcal."))
    assert not loaded, loaded
    print(f"selftest PASS (worst token32 KKT gap_rel on certified lra solutions: {w:.2e}; no hcal module loaded)")
