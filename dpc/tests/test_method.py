"""Method tests for dpc (synthetic data only; no Adult rows are read).

    cd <worktree> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q dpc/tests/test_method.py
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.tests.test_method --timing   # synthetic bank timing
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from collections import Counter

import numpy as np
import pytest

from dpc import compress as CP
from dpc import partition as PT
from dpc import release as RL


# ----------------------------------------------------------------------------------------------- synthetic data
def synth(N=15434, seed=0, unseen_class=5, underflow=195):
    """Real-shaped synthetic teacher outputs: income K=2 (exact 0/1 underflow rows), occupation K=6 with one class
    never predicted (tiny but positive probabilities), binary S correlated with both scores."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, N)
    z = rng.normal(-1.4 + 0.9 * S, 1.6, N)
    p = 1 / (1 + np.exp(-z))
    P1 = np.stack([1 - p, p], 1)
    if underflow:
        u = rng.choice(N, underflow, replace=False)
        P1[u[: underflow // 2]] = [1.0, 0.0]
        P1[u[underflow // 2:]] = [0.0, 1.0]
    bias = np.array([0.0, 0.6, 0.1, -1.2, 1.0, 0.2])
    if unseen_class is not None:
        bias[unseen_class] = -14.0
    L = rng.normal(size=(N, 6)) * 1.2 + bias + np.outer(S - 0.5, [0.8, -0.6, 0.3, 0.0, -0.4, 0.0])
    E = np.exp(L - L.max(1, keepdims=True))
    P2 = E / E.sum(1, keepdims=True)
    return P1, P2, S


@pytest.fixture(scope="module")
def small():
    P1, P2, S = synth(N=2500, seed=3, underflow=40)
    d1, d2 = P1.argmax(1), P2.argmax(1)
    f1 = PT.fit_fine(P1, d1, 2, max_cells=8)
    f2 = PT.fit_fine(P2, d2, 6, max_cells=6)
    return dict(P1=P1, P2=P2, S=S, d1=d1, d2=d2, f1=f1, f2=f2)


@pytest.fixture(scope="module")
def small_bank(small):
    s = small
    bank = CP.fit_bank(s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], ms=(2, 3), lams=(1.0, 10.0))
    return bank


def adversarial_rows(K, rng, n_random=200):
    """Ties at every pair of positions, one-hot rows for every class, underflow zeros, uniform, random."""
    rows = []
    for a, b in itertools.combinations(range(K), 2):
        r = np.zeros(K)
        r[a] = r[b] = 0.5
        rows.append(r)
        r = np.full(K, 0.1 / max(K - 2, 1)) if K > 2 else np.zeros(K)
        r[a] = r[b] = (1 - r.sum() + r[a] + r[b]) / 2
        rows.append(r / r.sum())
    rows += list(np.eye(K))
    rows.append(np.full(K, 1.0 / K))
    for c in range(K):
        r = np.full(K, 1e-300)
        r[c] = 1.0 - (K - 1) * 1e-300
        rows.append(r)
        r = np.zeros(K)
        r[c] = 0.6
        r[(c + 1) % K] = 0.4
        rows.append(r)
    R = rng.dirichlet(np.full(K, 0.3), n_random)
    rows += list(R)
    P = np.array(rows, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    return P


# ----------------------------------------------------------------------------------------------- smoothing / KL
@pytest.mark.parametrize("K", [2, 6])
def test_smoothing_rule(K):
    rng = np.random.default_rng(0)
    eps = 1e-12
    for c in range(K):
        for mean in [np.full(K, 1.0 / K), np.eye(K)[c], rng.dirichlet(np.ones(K))]:
            if mean.argmax() != c and not np.isclose(mean.max(), mean[c]):
                continue
            q = PT.smooth(mean, c)
            ref = (mean + eps * np.ones(K) + eps * np.eye(K)[c]) / (1 + (K + 1) * eps)
            assert np.array_equal(q, ref)
            assert np.all(np.isfinite(q)) and np.all(q >= 0) and abs(q.sum() - 1) <= 1e-12
            assert q.argmax() == c and all(q[c] > q[k] for k in range(K) if k != c)
    tie = np.zeros(K)
    tie[0] = tie[K - 1] = 0.5
    for c in (0, K - 1):
        assert PT.smooth(tie, c).argmax() == c
    bad = np.eye(K)[0]
    with pytest.raises(ValueError):
        PT.smooth(bad, 1)


def test_kl_sufficient_statistics_identity():
    rng = np.random.default_rng(1)
    for K in (2, 6):
        P = rng.dirichlet(np.full(K, 0.4), 300)
        P[:20] = np.eye(K)[rng.integers(0, K, 20)]                  # exact zeros
        q = PT.smooth(P.mean(0), int(P.mean(0).argmax()))
        kl = PT.kl_rows(P, np.repeat(q[None], len(P), 0))
        n, S, A = PT.cell_stats(P, np.zeros(len(P), dtype=int), 1)
        assert np.all(np.isfinite(kl)) and np.all(kl >= -1e-15)
        manual = sum(sum(p[k] * math.log(p[k] / q[k]) for k in range(K) if p[k] > 0) for p in P)
        assert abs(kl.sum() - manual) < 1e-9
        assert abs(kl.sum() - (A[0] - S[0] @ np.log(q))) < 1e-9


def test_input_validation():
    with pytest.raises(ValueError):
        PT.check_probs(np.array([[0.5, np.nan]]))
    with pytest.raises(ValueError):
        PT.check_probs(np.array([[1.2, -0.2]]))
    with pytest.raises(ValueError):
        PT.check_probs(np.array([[0.5, 0.6]]))


# ----------------------------------------------------------------------------------------------- fine partitions
def test_init_rule_order_statistics():
    Pc = np.array([[0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.55, 0.45], [0.9, 0.1]])
    C, nU, pos = PT._init_centroids(Pc, 0, 2)
    # distinct sorted by p_0 descending: .9 .8 .7 .6 .55 ; positions floor((2j+1)*5/4) = 1, 3
    assert nU == 5 and pos == [1, 3]
    assert np.array_equal(C, np.array([[0.8, 0.2], [0.6, 0.4]]))


def test_fit_fine_deterministic_and_consistent(small):
    s = small
    g = PT.fit_fine(s["P2"], s["d2"], 6, max_cells=6)
    assert g.fingerprint() == s["f2"].fingerprint()
    cell = PT.assign_fine(s["P2"], s["d2"], g)
    n, S, A = PT.cell_stats(s["P2"], cell, g.F)
    assert np.array_equal(n, g.n) and np.array_equal(S, g.S) and np.array_equal(A, g.A)
    for c in range(6):
        idx = g.cells_of(c)
        assert 1 <= idx.size <= 6
        assert np.all(g.cell_class[cell[s["d2"] == c]] == c)
    assert g.receipt["fallback_classes"] == [5]
    fb = g.cells_of(5)
    assert g.fallback[fb].all() and g.n[fb[0]] == 0
    assert np.array_equal(g.centroid[fb[0]], PT.smooth(np.full(6, 1 / 6), 5))
    # assignment equals brute-force nearest smoothed centroid within class (ties lowest index)
    for r in range(0, len(cell), 97):
        c = s["d2"][r]
        idx = g.cells_of(c)
        kls = [sum(p * math.log(p / q) for p, q in zip(s["P2"][r], g.centroid[j]) if p > 0) for j in idx]
        best = min(kls)
        assert cell[r] == idx[[i for i, v in enumerate(kls) if v <= best + 1e-15][0]]


def test_fewer_cells_than_cap():
    P = np.array([[0.9, 0.1]] * 5 + [[0.7, 0.3]] * 3 + [[0.2, 0.8]] * 4)
    f = PT.fit_fine(P, None, 2, max_cells=16)
    assert f.receipt["effective_cells_per_class"] == [2, 1]
    assert f.receipt["per_class"][0]["converged"]


def test_assign_ties_lowest_index():
    K = 2
    q = PT.smooth(np.array([0.8, 0.2]), 0)
    f = PT.FinePartition(K=K, cell_class=np.array([0, 0, 1]), centroid=np.stack([q, q, PT.smooth([0.3, 0.7], 1)]),
                         mean=np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]]), n=np.array([1, 1, 1]),
                         S=np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]]), A=np.zeros(3),
                         fallback=np.zeros(3, bool))
    cell = PT.assign_fine(np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]]), None, f)
    assert cell.tolist() == [0, 0, 2]


def test_refuses_label_keyed_partition(small):
    s = small
    rng = np.random.default_rng(5)
    y_true = s["d1"].copy()
    flip = rng.choice(len(y_true), 50, replace=False)
    y_true[flip] = 1 - y_true[flip]                            # a "true label" array that is not argmax P
    with pytest.raises(ValueError, match="teacher decision"):
        PT.fit_fine(s["P1"], y_true, 2)
    with pytest.raises(ValueError, match="teacher decision"):
        PT.assign_fine(s["P1"], y_true, s["f1"])
    with pytest.raises(ValueError, match="teacher decision"):
        CP.fit_policy_pair("FINE-TASK", s["f1"], s["f2"], s["P1"], y_true, s["P2"], s["d2"], s["S"], 2, None)
    pol = CP.fit_policy_pair("CLASS-ONLY", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 1,
                             None)[0].p1
    with pytest.raises(ValueError, match="teacher decision"):
        RL.encode(pol, s["P1"], y_true)
    with pytest.raises(ValueError, match="teacher decision"):     # SEX as a partition key is refused the same way
        PT.fit_fine(s["P1"], s["S"], 2)


def test_planted_true_label_grouping_not_representable(small):
    """Two rows with identical teacher scores but different labels always share a cell; a fine partition whose stored
    statistics encode a label grouping (not nearest-centroid deployment) is refused."""
    s = small
    P = s["P1"].copy()
    P[1] = P[0]
    cell = PT.assign_fine(P, None, s["f1"])
    assert cell[0] == cell[1]
    # plant: rewrite the stored stats of class 0 as if its cells were the true-label groups
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, len(P))
    f = PT.FinePartition.from_dict(s["f1"].to_dict())
    c0 = f.cells_of(0)
    rows0 = np.flatnonzero(s["d1"] == 0)
    planted = np.where(y[rows0] == 1, c0[0], c0[1])
    n, S, A = PT.cell_stats(s["P1"][rows0], planted - c0[0], 2)
    f.n[c0[:2]], f.S[c0[:2]], f.A[c0[:2]] = n, S, A
    f.n[c0[2:]] = 1                       # keep validate() shape rules satisfied; contents are wrong anyway
    with pytest.raises(ValueError, match="refused"):
        PT.validate_fine_against_rows(f, s["P1"], s["d1"])
    with pytest.raises(ValueError, match="refused"):
        CP.fit_policy_pair("FINE-TASK", f, s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 2, None)
    # no public fitting function accepts a label/group array
    import inspect
    for fn in (PT.fit_fine, PT.fit_fine_pair, CP.fit_policy_pair, CP.fit_class_only):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"y", "labels", "groups", "y_true", "assignment"}


# ----------------------------------------------------------------------------------------------- policies / release
def test_class_preservation_all_families(small, small_bank):
    s = small
    rng = np.random.default_rng(11)
    A1, A2 = adversarial_rows(2, rng), adversarial_rows(6, rng)
    n = min(len(A1), len(A2))
    A1, A2 = A1[:n], A2[:n]
    assert np.any(A2.argmax(1) == 5)                       # unseen predicted class is exercised
    for cid, (pair, rec) in small_bank.items():
        for pol, X, P in ((pair.p1, A1, s["P1"]), (pair.p2, A2, s["P2"])):
            for M in (X, P):
                tok, probs, dec = RL.encode(pol, M)
                assert np.array_equal(dec, M.argmax(1)), cid
                assert np.array_equal(probs.argmax(1), M.argmax(1)), cid
                assert np.all(np.abs(probs.sum(1) - 1) <= 1e-12)
            for c in range(pol.K):
                assert pol.tokens_per_class()[c] <= max(rec["m"], 1)
        # unseen occupation class decodes to its fallback token = smooth(uniform, 5)
        tok, probs, _ = RL.encode(pair.p2, np.eye(6)[[5]])
        assert pair.p2.token_fallback[tok[0]]
        assert np.array_equal(probs[0], PT.smooth(np.full(6, 1 / 6), 5))


def test_serialization_roundtrip_exact(small_bank, tmp_path, small):
    s = small
    for cid in ("JOINT__m3__lam10", "DIRECT-TASK__m2", "CLASS-ONLY__m1"):
        pair = small_bank[cid][0]
        p = tmp_path / f"{cid}.json"
        RL.save_policy(pair, p)
        back = RL.load_policy(p)
        assert back.fingerprint() == pair.fingerprint()
        for a, b in ((pair.p1, back.p1), (pair.p2, back.p2)):
            assert np.array_equal(a.token_proto, b.token_proto) and np.array_equal(a.fine.centroid, b.fine.centroid)
            assert np.array_equal(a.cell_token, b.cell_token)
            for M in (s["P1"] if a.K == 2 else s["P2"],):
                x, y = RL.encode(a, M), RL.encode(b, M)
                assert all(np.array_equal(u, v) for u, v in zip(x, y))
        q = tmp_path / f"{cid}_r2.npz"
        RL.save_policy_npz(pair.p2, q)
        b2 = RL.load_policy_npz(q)
        assert b2.fingerprint() == pair.p2.fingerprint() and np.array_equal(b2.fine.S, pair.p2.fine.S)
    z = small_bank["JOINT__m3__lam10"][0].to_dict()
    z["p2"]["token_proto"][0][0] += 1e-9
    with pytest.raises(ValueError):
        RL.PolicyPair.from_dict(z)
    z = small_bank["JOINT__m3__lam10"][0].to_dict()
    z["p1"]["fine"]["centroid"][0][0] += 1e-9
    with pytest.raises(ValueError):
        RL.PolicyPair.from_dict(z)


def test_canonical_ids_and_renumbering(small_bank, small):
    s = small
    pol = small_bank["LOCAL__m3__lam1"][0].p2
    rng = np.random.default_rng(2)
    perm = rng.permutation(pol.T)
    oh, probs, dec = RL.recipient_view(pol, s["P2"])["token_onehot"], None, None
    oh2, probs2, dec2 = RL.renumbered_view(pol, perm, s["P2"])
    assert np.array_equal(oh2[:, perm], oh)
    with pytest.raises(ValueError, match="canonical"):
        RL.Policy(recipient=2, fine=pol.fine, cell_token=perm[pol.cell_token])
    views = RL.release_views(small_bank["LOCAL__m3__lam1"][0].p1, pol, s["P1"], None, s["P2"], None)
    assert views["pair"]["token_onehot"].shape[1] == views["r1"]["alphabet"] + views["r2"]["alphabet"]
    assert np.array_equal(views["pair"]["tokens"][:, 1], views["r2"]["tokens"])
    assert views["r2"]["token_onehot"].shape[1] == pol.T                # full alphabet incl. fallback


def test_decoder_collision_keeps_token_identity():
    """Two tokens with identical decoded prototypes stay distinct one-hot columns."""
    P = np.array([[0.8, 0.2]] * 4 + [[0.3, 0.7]] * 4)
    q = PT.smooth(np.array([0.8, 0.2]), 0)
    f = PT.FinePartition(K=2, cell_class=np.array([0, 0, 1]), centroid=np.stack([q, q, PT.smooth([0.3, 0.7], 1)]),
                         mean=np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]]), n=np.array([2, 2, 4]),
                         S=np.array([[1.6, 0.4], [1.6, 0.4], [1.2, 2.8]]), A=np.zeros(3), fallback=np.zeros(3, bool))
    pol = RL.Policy(recipient=1, fine=f, cell_token=np.array([0, 1, 2]))
    assert np.array_equal(pol.token_proto[0], pol.token_proto[1])
    v = RL.recipient_view(pol, P[:1])
    assert v["token_onehot"].shape[1] == 3 and v["token_onehot"][0, 0] == 1 and v["token_onehot"][0, 1] == 0


def test_runner_compatibility_shims(small, small_bank):
    s = small
    pair = small_bank["SEQ-12__m2__lam1"][0]
    assert pair[0] is pair.p1 and pair[1] is pair.p2 and len(pair) == 2 and list(pair) == [pair.p1, pair.p2]
    assert RL.alphabet_size(pair[1]) == pair.p2.T and RL.fingerprint(pair) == pair.fingerprint()
    back = RL.policy_pair_from_dict(json.loads(json.dumps(RL.policy_pair_to_dict(pair))))
    assert back.fingerprint() == pair.fingerprint() and back.config == pair.config
    g = PT.from_dict(json.loads(json.dumps(PT.to_dict(s["f2"]))))
    assert g.fingerprint() == s["f2"].fingerprint()


def test_tokens_never_cross_classes(small):
    s = small
    with pytest.raises(ValueError, match="mixes predicted classes"):
        RL.Policy(recipient=1, fine=s["f1"], cell_token=np.zeros(s["f1"].F, dtype=int))


# ----------------------------------------------------------------------------------------------- objectives
def _independent_objective(pair, P1, P2, S):
    """Independent brute force (own loops/Counters): D_i, I_i, I_12."""
    t1, q1, _ = RL.encode(pair.p1, P1)
    t2, q2, _ = RL.encode(pair.p2, P2)
    N = len(S)

    def D(P, Q):
        return sum(sum(p * math.log(p / q) for p, q in zip(pr, qr) if p > 0) for pr, qr in zip(P, Q)) / N

    def I(keys):
        cnt = Counter(zip(S.tolist(), keys))
        ns = Counter(S.tolist())
        nc = Counter(keys)
        return sum(n / N * math.log(n * N / (ns[a] * nc[b])) for (a, b), n in cnt.items())

    return {"D1": D(P1, q1), "D2": D(P2, q2), "I1": I(t1.tolist()), "I2": I(t2.tolist()),
            "I12": I(list(zip(t1.tolist(), t2.tolist())))}


def test_objective_matches_bruteforce_all_families(small, small_bank):
    s = small
    for cid in ("CLASS-ONLY__m1", "FINE-TASK__m2", "DIRECT-TASK__m3", "LOCAL__m2__lam10", "SEQ-12__m3__lam1",
                "SEQ-21__m2__lam10", "JOINT__m3__lam10"):
        pair, rec = small_bank[cid]
        ind = _independent_objective(pair, s["P1"], s["P2"], s["S"])
        for k, v in ind.items():
            assert abs(rec["final"][k] - v) < 1e-10, (cid, k)
        assert rec["row_level_max_abs_diff"] < 1e-12


def _random_state(s, rng):
    f1, f2 = s["f1"], s["f2"]
    T = CP.fine_table(PT.assign_fine(s["P1"], s["d1"], f1), PT.assign_fine(s["P2"], s["d2"], f2), s["S"], f1.F,
                      f2.F)
    return CP.State(f1, f2, T, np.arange(f1.F), np.arange(f2.F)), T


def _rebuilt_terms(st, s):
    """Rebuild the objective from scratch from rows via a Policy pair built on the current labels."""
    p1 = RL.Policy(1, st.R[1].fine, RL.canonical_tokens(st.R[1].fine, st.labels(1)))
    p2 = RL.Policy(2, st.R[2].fine, RL.canonical_tokens(st.R[2].fine, st.labels(2)))
    return _independent_objective(RL.PolicyPair(p1, p2), s["P1"], s["P2"], s["S"])


def test_merge_and_move_deltas_vs_bruteforce(small):
    s = small
    rng = np.random.default_rng(7)
    st, T = _random_state(s, rng)
    lam = 3.0
    W = CP.W_joint(lam)

    def F(terms):
        return terms["D1"] + terms["D2"] + lam * ((terms["I1"] + terms["I2"]) / 2 + terms["I12"])

    for step in range(14):
        r = 1 + step % 2
        classes = [c for c in range(st.R[r].K) if st.R[r].coarse_count(c) > 2]
        if not classes:
            continue
        c = classes[rng.integers(len(classes))]
        labs, ia, ib, delta, dD, dI, dI12 = st.merge_deltas(r, c, W)
        j = int(rng.integers(len(delta)))
        before = _rebuilt_terms(st, s)
        st.apply_merge(r, int(labs[ia[j]]), int(labs[ib[j]]))
        after = _rebuilt_terms(st, s)
        assert abs((F(after) - F(before)) - delta[j]) < 1e-11
        assert abs((after[f"D{r}"] - before[f"D{r}"]) - dD[j]) < 1e-11
        assert abs((after[f"I{r}"] - before[f"I{r}"]) - dI[j]) < 1e-11
        assert abs((after["I12"] - before["I12"]) - dI12[j]) < 1e-11
        _check_tables(st, T)
    moved = 0
    for step in range(40):
        r = 1 + step % 2
        f = int(rng.integers(st.R[r].F))
        G = st.G(r)
        res = st.move_deltas(r, f, W, G)
        if res is None:
            continue
        B, delta, dD, dI, dI12 = res
        j = int(rng.integers(len(B)))
        before = _rebuilt_terms(st, s)
        st.apply_move(r, f, int(B[j]), G)
        after = _rebuilt_terms(st, s)
        assert abs((F(after) - F(before)) - delta[j]) < 1e-11
        assert abs((after["I12"] - before["I12"]) - dI12[j]) < 1e-11
        _check_tables(st, T)
        moved += 1
    assert moved > 10
    # engine terms equal the from-scratch rebuild
    e, b = st.terms(), _rebuilt_terms(st, s)
    assert max(abs(e[k] - b[k]) for k in b) < 1e-12


def _check_tables(st, T):
    assert st.T12.sum() == st.N == T.sum()
    assert np.array_equal(st.T12.sum(2), st.Tr[1]) and np.array_equal(st.T12.sum(1), st.Tr[2])
    ref = np.zeros_like(st.T12)
    np.add.at(ref, (slice(None), st.R[1].lab[:, None], st.R[2].lab[None, :]), T)
    assert np.array_equal(ref, st.T12)
    for r in (1, 2):
        for l, mem in st.R[r].members.items():
            assert st.R[r].n[l] == st.R[r].fine.n[mem].sum()


def test_seq_stage_one_coefficient_identity(small, small_bank):
    s = small
    rng = np.random.default_rng(4)
    st, T = _random_state(s, rng)
    lam = 2.5
    # arbitrary recipient-1 map, recipient 2 constant
    W = CP.W_seq_stage1(lam, 1)
    for _ in range(6):
        c = 0 if st.R[1].coarse_count(0) > 1 else 1
        labs = st.R[1].class_labels(c)
        if labs.size > 1:
            st.apply_merge(1, int(labs[0]), int(labs[1]))
    t = st.terms()
    I1c, I2c, I12c = CP.mi_terms_from_labels(T, st.labels(1), np.zeros(st.R[2].F, dtype=int))
    assert abs(I2c) < 1e-14 and abs(I12c - t["I1"]) < 1e-14
    fj_minus_const = t["D1"] + lam * ((I1c + I2c) / 2 + I12c)
    assert abs(fj_minus_const - (t["D1"] + 1.5 * lam * t["I1"])) < 1e-14
    assert abs(st.value(W) - (t["D1"] + 1.5 * lam * t["I1"])) < 1e-15
    # every SEQ unit records the identity and its coefficient; the first recipient equals a standalone stage-one run
    for cid in ("SEQ-12__m2__lam10", "SEQ-21__m3__lam1"):
        pair, rec = small_bank[cid]
        assert rec["stage1_identity"]["abs_diff"] <= 1e-12
        assert rec["stage1_coefficient"] == 1.5 * rec["lam"]
        a = rec["order"][0]
        st2, _ = _random_state(s, rng)
        W1 = CP.W_seq_stage1(rec["lam"], a)
        CP.greedy(st2, (a,), W1, rec["m"], [])
        CP.refine(st2, (a,), W1, [])
        first = pair.p1 if a == 1 else pair.p2
        assert np.array_equal(RL.canonical_tokens(first.fine, st2.labels(a)), first.cell_token)
        assert rec["stages"][1]["weights"] == CP.W_joint(rec["lam"]).__dict__


def test_joint_witness_dominance(small, small_bank):
    s = small
    for m in (2, 3):
        for lam in (1.0, 10.0):
            pair, rec = small_bank[f"JOINT__m{m}__lam{lam:g}"]
            for name, v in rec["witness_dominance"].items():
                assert v["final_minus_witness"] <= 0.0
            wits = {f: small_bank[CP.config_id(f, m, None if f == "FINE-TASK" else lam)][0]
                    for f in CP.WITNESS_FAMILIES}
            assert CP.check_joint_dominance(pair, wits, s["P1"], s["d1"], s["P2"], s["d2"], s["S"], lam) == []
            assert set(rec["starts"]) == set(CP.JOINT_START_ORDER)
            assert rec["winner"]["F_joint"] == min(x["F_joint"] for x in rec["candidates"])
    # a deliberately weak "joint" solution (class-only) must be flagged against the witnesses
    weak = small_bank["CLASS-ONLY__m1"][0]
    wits = {f: small_bank[CP.config_id(f, 2, None if f == "FINE-TASK" else 1.0)][0] for f in CP.WITNESS_FAMILIES}
    assert CP.check_joint_dominance(weak, wits, s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 1.0)
    # mismatched witnesses are refused
    with pytest.raises(ValueError):
        CP.fit_policy_pair("JOINT", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 2, 1.0,
                           witnesses={"LOCAL": small_bank["LOCAL__m3__lam1"][0]})
    with pytest.raises(ValueError):
        CP.fit_policy_pair("JOINT", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 2, 1.0,
                           witnesses={"EXTRA": small_bank["LOCAL__m2__lam1"][0]})


def test_joint_recomputed_witnesses_match_passed(small, small_bank):
    s = small
    pair, rec = CP.fit_policy_pair("JOINT", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 2, 1.0)
    assert pair.fingerprint() == small_bank["JOINT__m2__lam1"][0].fingerprint()
    assert all(v["source"] == "recomputed" for k, v in rec["starts"].items() if k != "JOINT-GREEDY")


def test_cap_effective_states_and_receipts(small_bank):
    for cid, (pair, rec) in small_bank.items():
        for r, pol in ((1, pair.p1), (2, pair.p2)):
            assert all(t <= rec["m"] for t in pol.tokens_per_class())
            assert rec[f"r{r}"]["effective_states"] == int((pol.token_n > 0).sum())
            assert rec[f"r{r}"]["max_abs_unsmoothed_vs_smoothed"] < 1e-11
        assert pair.p2.token_fallback.sum() == 1
        assert rec["wall_seconds"] > 0 and rec["cpu_seconds"] >= 0
        if rec["family"] in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"):
            for stg in rec["stages"]:
                assert stg["sweeps"] <= CP.SWEEPS
                assert stg["stage_objective_after_refine"] <= stg["stage_objective_after_greedy"] + 1e-12
                assert all(mv["delta"] < -CP.TOL for mv in stg["moves"])


def test_receipts_json_safe_and_complete(small_bank):
    for cid, (pair, rec) in small_bank.items():
        txt = json.dumps(rec, allow_nan=False)                         # strict: no numpy scalars, NaN or inf
        back = json.loads(txt)
        for k in ("family", "m", "lam", "final", "after_greedy", "r1", "r2", "work", "summary", "wall_seconds",
                  "cpu_seconds", "pair_fingerprint", "row_level_max_abs_diff"):
            assert k in back, (cid, k)
        for k in ("D1", "D2", "I1", "I2", "I12", "F_task"):
            assert k in back["final"]
        if rec["lam"] is not None:
            assert "F_joint" in back["final"] and "F_local" in back["final"]
            assert "F_joint" in back["after_greedy"]
        json.dumps(pair.to_dict(), allow_nan=False)


def test_task_families_ignore_sensitive_labels(small):
    s = small
    rng = np.random.default_rng(8)
    S2 = rng.permutation(s["S"])
    for fam in ("FINE-TASK", "DIRECT-TASK", "CLASS-ONLY"):
        m = 1 if fam == "CLASS-ONLY" else 3
        a = CP.fit_policy_pair(fam, s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], m, None)[0]
        b = CP.fit_policy_pair(fam, s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], S2, m, None)[0]
        assert a.fingerprint() == b.fingerprint()
    with pytest.raises(ValueError):
        CP.fit_policy_pair("FINE-TASK", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 3, 1.0)
    with pytest.raises(ValueError):
        CP.fit_policy_pair("LOCAL", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 3, None)


def test_alias_detection(small, small_bank):
    s = small
    loc0 = CP.fit_policy_pair("LOCAL", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 2, 0.0)[0]
    al = CP.find_aliases({"FINE-TASK__m2": small_bank["FINE-TASK__m2"][0], "LOCAL__m2__lam0": loc0,
                          "JOINT__m2__lam1": small_bank["JOINT__m2__lam1"][0]})
    assert al["pair"]["LOCAL__m2__lam0"] == "FINE-TASK__m2"
    assert al["pair"]["FINE-TASK__m2"] == "FINE-TASK__m2"


def test_greedy_records_positive_increments_and_ties():
    """Identical fine cells give exactly tied merge increments: the lexicographic rule picks the lowest pair."""
    P = np.array([[0.9, 0.1]] * 6 + [[0.7, 0.3]] * 6 + [[0.6, 0.4]] * 6 + [[0.2, 0.8]] * 6)
    S = np.array([0, 1] * 12)
    f = PT.fit_fine(P, None, 2)
    assert f.receipt["effective_cells_per_class"] == [3, 1]
    st = CP.State(f, f, CP.fine_table(PT.assign_fine(P, None, f), PT.assign_fine(P, None, f), S, f.F, f.F),
                  np.arange(f.F), np.arange(f.F))
    log = []
    CP.greedy(st, (1,), CP.W_task(), 1, log)
    assert [(x["a"], x["b"]) for x in log][0] == (1, 2)          # 0.7 and 0.6 are the closest pair
    assert all(x["increment"] > 0 for x in log)


# ----------------------------------------------------------------------------------------------- fixtures
def _enumerate_partitions(items, maxblocks):
    """All set partitions of ``items`` into at most maxblocks blocks (as label lists = lowest member)."""
    items = list(items)
    if not items:
        yield {}
        return

    def rec(i, blocks):
        if i == len(items):
            yield blocks
            return
        for b in range(len(blocks)):
            yield from rec(i + 1, blocks[:b] + [blocks[b] + [items[i]]] + blocks[b + 1:])
        if len(blocks) < maxblocks:
            yield from rec(i + 1, blocks + [[items[i]]])

    for blocks in rec(0, []):
        yield {f: min(b) for b in blocks for f in b}


def _all_maps(fine, m):
    per_class = [list(_enumerate_partitions(fine.cells_of(c).tolist(), m)) for c in range(fine.K)]
    for combo in itertools.product(*per_class):
        lab = np.zeros(fine.F, dtype=np.int64)
        for dct in combo:
            for f, l in dct.items():
                lab[f] = l
        yield lab


def test_exhaustive_fixture_bracket():
    """Tiny finite fixture: enumerate every class-preserving map with <= m cells per class; JOINT's final objective is
    bracketed by the exhaustive optimum (a fixture gap is reported, not erased)."""
    rng = np.random.default_rng(12)
    vals1 = [0.95, 0.8, 0.65, 0.55, 0.3, 0.1]
    vals2 = [0.9, 0.75, 0.6, 0.4, 0.2]
    N = 600
    a = rng.integers(0, len(vals1), N)
    b = rng.integers(0, len(vals2), N)
    S = ((a + b + rng.integers(0, 3, N)) % 2 == 0).astype(int)
    P1 = np.stack([np.array(vals1)[a], 1 - np.array(vals1)[a]], 1)
    P2 = np.stack([np.array(vals2)[b], 1 - np.array(vals2)[b]], 1)
    f1, f2, _ = PT.fit_fine_pair(P1, None, P2, None)
    assert f1.F == 6 and f2.F == 5
    T = CP.fine_table(PT.assign_fine(P1, None, f1), PT.assign_fine(P2, None, f2), S, f1.F, f2.F)
    m, lam = 2, 5.0
    best_any, best_exact, count = np.inf, np.inf, 0
    for l1 in _all_maps(f1, m):
        for l2 in _all_maps(f2, m):
            st = CP.State(f1, f2, T, l1, l2)
            v = CP.F_values(st.terms(), lam)["F_joint"]
            count += 1
            best_any = min(best_any, v)
            full = all(st.R[r].coarse_count(c) == min(m, st.R[r].fine.cells_of(c).size)
                       for r in (1, 2) for c in range(2))
            if full:
                best_exact = min(best_exact, v)
    pair, rec = CP.fit_policy_pair("JOINT", f1, f2, P1, None, P2, None, S, m, lam)
    fj = rec["final"]["F_joint"]
    assert fj >= best_any - 1e-12 and fj >= best_exact - 1e-12
    gap = fj - best_exact
    print(json.dumps({"exhaustive_maps": count, "global_best_le_m": best_any, "global_best_exactly_cap": best_exact,
                      "joint_final": fj, "gap_to_exact_cap_optimum": gap, "joint_globally_best": gap <= 1e-12}))


def _xor_fixture(N=4000, seed=0):
    """Each recipient's confidence clue u (resp. v) alone is independent of S; S = u XOR v. The clue is a large score
    difference (task-relevant); a second small difference w carries nothing."""
    rng = np.random.default_rng(seed)
    u, v, w1, w2 = (rng.integers(0, 2, N) for _ in range(4))
    S = u ^ v
    p1 = np.array([0.95, 0.65])[u] + np.array([0.0, 0.02])[w1]
    p2 = np.array([0.93, 0.62])[v] + np.array([0.0, 0.02])[w2]
    P1 = np.stack([p1, 1 - p1], 1)
    P2 = np.stack([p2, 1 - p2], 1)
    return P1, P2, S


def test_coalition_positive_fixture():
    P1, P2, S = _xor_fixture()
    f1, f2, _ = PT.fit_fine_pair(P1, None, P2, None)
    fine_pair = CP.fit_policy_pair("FINE-TASK", f1, f2, P1, None, P2, None, S, 16, None)[1]["final"]
    assert fine_pair["I1"] < 0.003 and fine_pair["I2"] < 0.003 and fine_pair["I12"] > 0.6
    lam = 10.0
    loc = CP.fit_policy_pair("LOCAL", f1, f2, P1, None, P2, None, S, 2, lam)[1]["final"]
    jnt = CP.fit_policy_pair("JOINT", f1, f2, P1, None, P2, None, S, 2, lam)[1]["final"]
    assert loc["I12"] > 0.6                                    # local criteria cannot see the coalition
    assert jnt["I12"] < 0.05 and jnt["F_joint"] < loc["F_joint"]


def test_null_fixture():
    P1, P2, _ = synth(N=3000, seed=21, underflow=0)
    S = np.random.default_rng(99).integers(0, 2, 3000)
    f1 = PT.fit_fine(P1, None, 2, max_cells=6)
    f2 = PT.fit_fine(P2, None, 6, max_cells=4)
    pair, rec = CP.fit_policy_pair("JOINT", f1, f2, P1, None, P2, None, S, 2, 1.0)
    assert rec["final"]["I12"] < 0.02 and rec["final"]["I1"] < 0.005


# ----------------------------------------------------------------------------------------------- deploy
def _synthetic_unit(tmp_path, seed=0):
    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from jcv.finalize import save_unit
    from jcv.train import Model
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1200, 83)).astype(np.float64)
    names = [f"f{j:02d}" for j in range(83)]
    model = Model(83, [2, 6], seed)
    with torch.no_grad():
        R = [model.encode(i, torch.from_numpy(X.astype(np.float32))).double().numpy() for i in (0, 1)]
    y1 = (R[0][:, 0] > np.median(R[0][:, 0])).astype(int)
    y2 = np.digitize(R[1][:, 1], np.quantile(R[1][:, 1], [0.2, 0.4, 0.6, 0.8, 0.999]))
    heads = [make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[0], y1),
             make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[1], y2)]
    unit = tmp_path / "rel__s0__SYNTH"
    state = model.state_dict()
    save_unit(unit, {"model.pt": lambda p: torch.save(state, p),
                     "head_0.joblib": lambda p: joblib.dump(heads[0], p),
                     "head_1.joblib": lambda p: joblib.dump(heads[1], p)}, {"seed": seed})
    return unit, X, names


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    from dpc import deploy as DP
    tmp = tmp_path_factory.mktemp("deploy")
    unit, X, names = _synthetic_unit(tmp)
    model, heads, msha = DP.load_teacher(unit)
    P1, P2 = DP.teacher_probs(model, heads, X)
    S = (X[:, 0] > 0).astype(int)
    f1, f2, _ = PT.fit_fine_pair(P1, None, P2, None, max_cells=6)
    pair, _ = CP.fit_policy_pair("JOINT", f1, f2, P1, None, P2, None, S, 2, 1.0,
                                 meta={"teacher_model_sha256": msha})
    pol = tmp / "pair.json"
    RL.save_policy(pair, pol)
    np.savez(tmp / "schema.npz", feature_names=np.array(names))
    np.savez(tmp / "in.npz", X=X, feature_names=np.array(names))
    return dict(tmp=tmp, unit=unit, X=X, names=names, P1=P1, P2=P2, pair=pair, pol=pol, msha=msha)


def _args(D, **kw):
    a = {"--unit": str(D["unit"]), "--policy": str(D["pol"]), "--X": str(D["tmp"] / "in.npz"),
         "--schema": str(D["tmp"] / "schema.npz"), "--out": str(D["tmp"] / "out.npz")}
    a.update(kw)
    out = []
    for k, v in a.items():
        out += [k] if v is True else [k, v]
    return out


def test_deploy_end_to_end(deployed):
    from dpc import deploy as DP
    D = deployed
    DP.main(_args(D))
    with np.load(D["tmp"] / "out.npz", allow_pickle=False) as z:
        assert sorted(z.files) == sorted(DP.ALLOWED_OUTPUT)
        assert np.array_equal(z["decision_1"], D["P1"].argmax(1))
        assert np.array_equal(z["decision_2"], D["P2"].argmax(1))
        t, q, d = RL.encode(D["pair"].p2, D["P2"])
        assert np.array_equal(z["tokens_2"], t) and np.array_equal(z["probs_2"], q)


def _expect_refusal(D, capsys, args, needle):
    from dpc import deploy as DP
    with pytest.raises(SystemExit) as e:
        DP.main(args)
    assert e.value.code == 2
    assert needle in capsys.readouterr().err


def test_deploy_refuses_schema_violations(deployed, capsys):
    D = deployed
    X, names = D["X"], D["names"]
    np.savez(D["tmp"] / "extra.npz", X=np.hstack([X, X[:, :1]]), feature_names=np.array(names + ["SEX"]))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "extra.npz")}), "refused")
    np.savez(D["tmp"] / "missing.npz", X=X[:, :82], feature_names=np.array(names[:82]))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "missing.npz")}), "refused")
    perm = list(range(83))
    perm[3], perm[4] = perm[4], perm[3]
    np.savez(D["tmp"] / "reorder.npz", X=X[:, perm], feature_names=np.array([names[j] for j in perm]))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "reorder.npz")}), "reordered")
    renamed = names[:]
    renamed[7] = "relationship_hidden"
    np.savez(D["tmp"] / "renamed.npz", X=X, feature_names=np.array(renamed))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "renamed.npz")}), "missing")
    np.savez(D["tmp"] / "bundled.npz", X=X, feature_names=np.array(names), sex=np.zeros(len(X)))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "bundled.npz")}), "exactly X and feature_names")


def test_deploy_refuses_export_attempts(deployed, capsys):
    D = deployed
    for flag in ("--export-fine-ids", "--fine-cells", "--raw-scores", "--logits", "--features", "--teacher-probs",
                 "--export=all", "--centred-logits", "--debug"):
        _expect_refusal(D, capsys, _args(D) + [flag], "refused")
    from dpc import deploy as DP
    out, _ = DP.release(D["unit"], D["pair"], D["X"].astype(np.float32))
    assert set(out) == set(DP.ALLOWED_OUTPUT)
    bad = dict(out)
    bad["fine_1"] = np.zeros(len(D["X"]), dtype=np.int64)
    with pytest.raises(DP.Refused):
        DP.write_release(D["tmp"] / "bad.npz", bad)
    bad = dict(out)
    bad["p_teacher_1"] = D["P1"]
    with pytest.raises(DP.Refused):
        DP.write_release(D["tmp"] / "bad.npz", bad)


def test_deploy_refuses_unbound_or_mismatched_policy(deployed, capsys, tmp_path):
    D = deployed
    z = D["pair"].to_dict()
    z["config"].pop("teacher_model_sha256")
    for side in ("p1", "p2"):
        z[side]["meta"].pop("teacher_model_sha256")
    pp = RL.PolicyPair.from_dict(z)
    RL.save_policy(pp, tmp_path / "unbound.json")
    _expect_refusal(D, capsys, _args(D, **{"--policy": str(tmp_path / "unbound.json")}), "not bound")
    from dpc import deploy as DP
    DP.main(_args(D, **{"--policy": str(tmp_path / "unbound.json"), "--allow-unbound-policy": True}))
    pp.config["teacher_model_sha256"] = "0" * 64
    RL.save_policy(pp, tmp_path / "wrong.json")
    _expect_refusal(D, capsys, _args(D, **{"--policy": str(tmp_path / "wrong.json")}), "different teacher")
    pp.config["teacher_model_sha256"] = D["msha"]
    pp.config["feature_names_sha256"] = DP.schema_sha256(list(reversed(D["names"])))
    RL.save_policy(pp, tmp_path / "schema.json")
    _expect_refusal(D, capsys, _args(D, **{"--policy": str(tmp_path / "schema.json")}), "different pinned feature")
    _expect_refusal(D, capsys, _args(D, **{"--schema-sha256": "0" * 64}), "schema hash mismatch")


# ----------------------------------------------------------------------------------------------- timing
def run_timing(N=15434, seed=0, ms=(2, 4, 8), lams=(0.1, 1.0, 10.0)):
    P1, P2, S = synth(N=N, seed=seed)
    t0, c0 = time.perf_counter(), time.process_time()
    f1, f2, fr = PT.fit_fine_pair(P1, None, P2, None)
    fine_wall, fine_cpu = time.perf_counter() - t0, time.process_time() - c0
    t1, c1 = time.perf_counter(), time.process_time()
    bank = CP.fit_bank(f1, f2, P1, None, P2, None, S, ms=ms, lams=lams)
    bank_wall, bank_cpu = time.perf_counter() - t1, time.process_time() - c1
    per_family = {}
    for cid, (pair, rec) in bank.items():
        x = per_family.setdefault(rec["family"], {"units": 0, "wall": 0.0, "cpu": 0.0, "max_wall": 0.0})
        x["units"] += 1
        x["wall"] += rec["wall_seconds"]
        x["cpu"] += rec["cpu_seconds"]
        x["max_wall"] = max(x["max_wall"], rec["wall_seconds"])
    n_units = len(bank)
    return {"rows": N, "fine_cells": [f1.F, f2.F], "fine_wall_s": fine_wall, "fine_cpu_s": fine_cpu,
            "fine_rounds": [[r.get("rounds_used") for r in fr[k]["per_class"]] for k in ("r1", "r2")],
            "bank_units": n_units, "bank_wall_s": bank_wall, "bank_cpu_s": bank_cpu,
            "per_unit_mean_wall_s": bank_wall / n_units,
            "per_family": {k: {**v, "mean_wall": v["wall"] / v["units"]} for k, v in per_family.items()},
            "projected_full_study_wall_s_6_teacher_seeds": 6 * (fine_wall + bank_wall)}


def test_synthetic_bank_timing_real_shape():
    """One full configuration bank (1 class-only + 6 task-only + 36 privacy units) on real-shaped synthetic data."""
    res = run_timing()
    assert res["bank_units"] == 43 and res["fine_cells"] == [32, 81]
    print(json.dumps(res, indent=1))
    assert res["bank_wall_s"] < 600


if __name__ == "__main__":
    if "--timing" in sys.argv:
        print(json.dumps(run_timing(), indent=1))
