"""Role E (math and claims reviewer) tests for the confidence-budgeted privacy study (cbp).

Owner: role E; this file only. It lives in cbp/review_tests/, which no lock globs. Synthetic fixtures only: no Adult
row, no SEX value, no task label and no receipt of a real fit is read.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema \
        --label E:tests -- ~/PCRL/.venv/bin/python -m pytest cbp/review_tests/test_math_review.py -q

Layout (section numbers follow results/pcrl_confidence_budgeted_privacy_v1/MATH_REVIEW.md):
  REFERENCE   own re-implementations (math.log KL, dict plug-in MI, first-index argmax, smoothing); imports nothing
              from qpc, dpc or cbp.
  S1          decision containment: the floating-point lemmas, the deployed release on adversarial inputs for EVERY
              family and for arbitrary within-class groupings, the guards, absent classes, the input boundary, and why
              sampled row checks are not a proof.
  S2          data processing and what it does not imply (a fitted reader can gain from coarsening).
  S3          privacy objectives: decision-floor decomposition, plug-in MI null bias at the i8o64 alphabets, fitted
              MI below its own permutation null under a true null (selection, not protection).
  S4          the corrected sequential design: CLASS-ONLY counterpart in stage one, never a constant; the correction
              term is lambda * I(S; d_b | C_a).
  S5          JOINT: dominance over the unchanged witnesses on the fitting F_joint only (registered lambda grid);
              no dominance in components, on held-out rows, or over DIRECT-TASK; the start asymmetry.
  S6/S7       cbp.fit / cbp.audit review checks (skip until those modules exist).
qpc / cbp modules are imported lazily, so the file runs before the cbp modules exist (those tests skip).
"""
from __future__ import annotations

import importlib
import itertools
import json
import math

import numpy as np
import pytest

EPS = 1e-12
LAM_GRID = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)          # prompt section 7, registered grid


def _mod(name):
    """Import a module or skip (owned by another role; may not exist yet)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            pytest.skip(f"{name} not present yet")
        raise


# ==================================================================================================================
# REFERENCE (no qpc / dpc / cbp imports)
# ==================================================================================================================

def argmax_first(p):
    """Teacher decision: the LOWEST index attaining the maximum (pure Python, independent of numpy's argmax)."""
    v = [float(x) for x in p]
    best = max(v)
    return next(k for k, x in enumerate(v) if x == best)


def argmax_last(p):
    v = [float(x) for x in p]
    best = max(v)
    return max(k for k, x in enumerate(v) if x == best)


def ref_smooth(mean_p, c):
    """(m + eps*1 + eps*e_c) / (1 + (K+1) eps) with the registered operation order: ((m + eps) + eps e_c) / den."""
    m = np.asarray(mean_p, np.float64)
    K = m.shape[-1]
    e = np.zeros(K)
    e[int(c)] = EPS
    return (m + EPS * np.ones(K) + e) / (1.0 + (K + 1) * EPS)


def ref_kl(p, q):
    acc = 0.0
    for pk, qk in zip(np.asarray(p, float).tolist(), np.asarray(q, float).tolist()):
        if pk > 0:
            acc += pk * (math.log(pk) - math.log(qk))
    return acc


def ref_mi(s, *codes):
    """Plug-in I(S; codes) in nats from exact counts (dict, math.log); empty cells contribute 0."""
    s = np.asarray(s)
    N = len(s)
    if N == 0 or not codes:
        return 0.0
    keys = list(zip(*[np.asarray(c).tolist() for c in codes]))
    joint, cc, cs = {}, {}, {}
    for k, sv in zip(keys, s.tolist()):
        joint[(k, sv)] = joint.get((k, sv), 0) + 1
        cc[k] = cc.get(k, 0) + 1
        cs[sv] = cs.get(sv, 0) + 1
    return float(sum(n / N * math.log(n * N / (cc[k] * cs[sv])) for (k, sv), n in joint.items()))


def ref_cmi(s, cond, *codes):
    """Plug-in I(S; codes | cond) = sum_m P(cond = m) I(S; codes | cond = m)."""
    s, cond = np.asarray(s), np.asarray(cond)
    tot = 0.0
    for m in np.unique(cond):
        w = cond == m
        tot += w.mean() * ref_mi(s[w], *[np.asarray(c)[w] for c in codes])
    return tot


def fast_mi(s, tok):
    """Vectorised plug-in I(S; tok) for large N (same definition as ref_mi)."""
    s = np.asarray(s, np.int64)
    _, t = np.unique(np.asarray(tok), return_inverse=True)
    L = int(t.max()) + 1
    tab = np.bincount(s * L + t, minlength=2 * L).reshape(2, L).astype(float)
    N = tab.sum()
    ns, nc = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    m = tab > 0
    return float(np.sum(np.where(m, tab / N * np.log(np.where(m, tab * N, 1) / np.where(m, ns * nc, 1)), 0.0)))


def teacher_like(n, K, seed, conc=0.6, ties=0, zeros=0, absent=None):
    """Teacher-like rows on the simplex; optional exact two-way ties with a LATER index, exact zeros, an absent class."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.full(K, conc), size=n)
    P /= P.sum(1, keepdims=True)
    for r in range(min(ties, n)):
        i, j = sorted(rng.choice(K, 2, replace=False).tolist())
        v = np.full(K, 0.1 / max(K - 2, 1)) if K > 2 else np.zeros(K)
        v[i] = v[j] = 0.45 if K > 2 else 0.5
        P[r] = v / v.sum()
    for r in range(ties, min(ties + zeros, n)):
        v = rng.dirichlet(np.ones(K))
        v[rng.integers(K)] = 0.0
        P[r] = v / v.sum()
    if absent is not None:
        P = P[np.array([argmax_first(p) != absent for p in P])]
    return P


def ref_terms_rows(pol1, pol2, P1, P2, s, enc):
    """D1, D2, I1, I2, I12 recomputed from the RELEASED rows (tokens, decoded vectors) with own KL and MI."""
    t1, q1, _ = enc(pol1, P1)
    t2, q2, _ = enc(pol2, P2)
    out = {"D1": float(np.mean([ref_kl(p, q) for p, q in zip(P1, q1)])),
           "D2": float(np.mean([ref_kl(p, q) for p, q in zip(P2, q2)])),
           "I1": ref_mi(s, t1), "I2": ref_mi(s, t2), "I12": ref_mi(s, t1, t2)}
    return out, t1, t2


def F_joint(t, lam):
    return t["D1"] + t["D2"] + lam * ((t["I1"] + t["I2"]) / 2 + t["I12"])


def F_local(t, lam):
    return t["D1"] + t["D2"] + lam * (t["I1"] + t["I2"]) / 2


# ---------------------------------------------------------------- finite fixtures (fine cells = distinct vectors)

class RefRecip:
    """One recipient of a finite fixture: fine cells = distinct vectors V listed in class order; rows -> fine cell f."""

    def __init__(self, V, f):
        self.V = np.asarray(V, np.float64)
        self.K = self.V.shape[1]
        self.cls = np.array([argmax_first(v) for v in self.V])
        self.f = np.asarray(f, np.int64)
        self.P = self.V[self.f]
        self.F = len(self.V)

    def class_labels(self):
        first = {}
        return np.array([first.setdefault(int(c), j) for j, c in enumerate(self.cls)], np.int64)


def fine_from_recip(R):
    """qpc FinePartition whose cells are exactly the fixture's distinct vectors."""
    KM = _mod("qpc.kmeans")
    DPT = _mod("dpc.partition")
    assert np.all(np.diff(R.cls) >= 0), "fixture vectors must be listed in class order"
    n, S, A = DPT.cell_stats(R.P, R.f, R.F)
    fine = KM.FinePartition(K=R.K, cell_class=R.cls.astype(np.int64), centroid=DPT.smooth(S / n[:, None], R.cls),
                            mean=S / n[:, None], n=n, S=S, A=A, fallback=np.zeros(R.F, bool))
    fine.validate()
    assert np.array_equal(KM.assign_fine(R.P, R.P.argmax(1), fine), R.f)
    return fine


def fixture_six(seed=0):
    """Source review fixture 'six' (income 6 cells, occupation 6 classes x 2 levels; S on both)."""
    rng = np.random.default_rng(seed)
    V1 = [[1 - q, q] for q in (0.1, 0.25, 0.4, 0.6, 0.75, 0.9)]
    V2 = []
    for c in range(6):
        for h in (0.45, 0.85):
            v = np.full(6, (1 - h) / 5)
            v[c] = h
            V2.append(v.tolist())
    N = 300
    S = rng.integers(0, 2, N)
    f1 = np.where(rng.random(N) < 0.6, 3 * S + rng.integers(0, 3, N), rng.integers(0, 6, N))
    f2 = 2 * rng.integers(0, 6, N) + np.where(rng.random(N) < 0.7, S, rng.integers(0, 2, N))
    return RefRecip(V1, f1), RefRecip(V2, f2), np.asarray(S, np.int64)


def fixture_decision_carries_s(seed=2):
    """Recipient 2's DECISION carries S; recipient 1's within-class level is partly redundant with d_2 (source review
    designed fixture for the sequential correction)."""
    rng = np.random.default_rng(seed)
    N = 600
    Sx = rng.integers(0, 2, N)
    V1 = [[0.95, 0.05], [0.8, 0.2], [0.65, 0.35], [0.55, 0.45], [0.4, 0.6], [0.2, 0.8]]
    V2 = [[0.7, 0.3], [0.3, 0.7]]
    d2 = np.where(rng.random(N) < 0.85, Sx, 1 - Sx)
    lvl = np.where(rng.random(N) < 0.75, d2 ^ (rng.random(N) < 0.5), rng.integers(0, 2, N))
    f1 = np.where(lvl == 1, rng.integers(0, 2, N), 2 + rng.integers(0, 2, N))
    f1 = np.where(rng.random(N) < 0.2, 4 + rng.integers(0, 2, N), f1)
    return RefRecip(V1, f1), RefRecip(V2, d2), np.asarray(Sx, np.int64)


def fit_recip(fam, R1, R2, s, m1, m2, lam, **kw):
    CP = _mod("qpc.compress")
    f1, f2 = fine_from_recip(R1), fine_from_recip(R2)
    return CP.fit_policy_pair(fam, f1, f2, R1.P, R1.P.argmax(1), R2.P, R2.P.argmax(1), s, m1, m2, lam, **kw)


def labels_of(pair):
    CP = _mod("qpc.compress")
    return CP.labels_from_policy(pair.p1), CP.labels_from_policy(pair.p2)


def terms_from_labels(R1, R2, s, lab1, lab2):
    """Own objective terms for fine-cell groupings (tokens = group label; prototypes = smooth(group mean))."""
    out = {}
    for name, R, lab in (("1", R1, lab1), ("2", R2, lab2)):
        tok = np.asarray(lab)[R.f]
        D = 0.0
        for t in np.unique(tok):
            w = tok == t
            q = ref_smooth(R.P[w].mean(0), R.cls[R.f[w][0]])
            D += sum(ref_kl(p, q) for p in R.P[w])
        out["D" + name] = D / len(s)
    t1, t2 = np.asarray(lab1)[R1.f], np.asarray(lab2)[R2.f]
    out["I1"], out["I2"], out["I12"] = ref_mi(s, t1), ref_mi(s, t2), ref_mi(s, t1, t2)
    return out


# ==================================================================================================================
# S1  DECISION CONTAINMENT
# ==================================================================================================================

MAGNITUDES = (0.0, 5e-324, 1e-300, 1e-17, 1e-13, 1e-12, 1e-9, 0.1, 1 / 6, 0.25, 1 / 3, 0.5, 1 - 1e-16, 1.0,
              1 + 1e-9, 1.5, 1.999)


@pytest.mark.parametrize("K", [2, 6])
def test_lemma_strict_smoothing_margin_for_any_weakly_ordered_mean(K):
    """Lemma 3 (MATH_REVIEW 1.3). If 0 <= m_k <= m_c < 2 for every k (ties allowed, any magnitude, denormals), the
    registered float64 smoothing has STRICT argmax c with margin q_c - max_{k != c} q_k > 9e-13. Exact later-index ties
    are the hard case: they are where the eps*e_c term is needed."""
    DPT = _mod("dpc.partition")
    rng = np.random.default_rng(K)
    for c in range(K):
        for a in MAGNITUDES:                                       # every coordinate tied with the class coordinate
            m = np.full(K, a)
            for q in (ref_smooth(m, c), DPT.smooth(m, c, check=False)):
                others = np.delete(q, c)
                assert q[c] > others.max() and q[c] - others.max() > 9e-13, (K, c, a, q)
                assert argmax_first(q) == c and argmax_last(q) == c
        for _ in range(400):                                       # random weak orders with exact ties at c
            m = rng.choice(MAGNITUDES, K)
            top = m.max()
            m[c] = top
            if rng.random() < 0.5:
                m[rng.integers(K)] = top                           # another exact tie (any index)
            q = DPT.smooth(m, c, check=False)
            assert np.array_equal(q, ref_smooth(m, c))             # registered operation order, bitwise
            others = np.delete(q, c)
            assert q[c] - others.max() > 9e-13, (m, c)
            if abs(m.sum() - 1.0) < 1e-13:                          # a genuine mean also passes the full check
                DPT.check_prototypes(q[None], c)


def test_lemma_row_order_sums_and_means_keep_the_weak_order():
    """Lemma 2 (MATH_REVIEW 1.3). Rows whose first-index argmax is c satisfy x_c >= x_k (strict for k < c). IEEE-754
    round-to-nearest addition and division by the same positive n are monotone, so sums accumulated in ONE fixed row
    order (np.bincount, then token_tables' sequential accumulation over member cells) keep S_c >= S_k and m_c >= m_k.
    Adversarial rows: later-index exact ties, values that differ by one ulp, 1e-300 entries, many rows."""
    DPT = _mod("dpc.partition")
    DRL = _mod("dpc.release")
    rng = np.random.default_rng(7)
    for K in (2, 6):
        for trial in range(60):
            c = int(rng.integers(K))
            n = int(rng.choice([1, 2, 3, 50, 2000]))
            X = rng.dirichlet(np.full(K, 0.3), size=n)
            X[:, c] = X.max(1)                                     # make c a maximum
            j = [k for k in range(c + 1, K)]
            if j:
                tie = rng.random(n) < 0.5                          # exact ties with a LATER index
                X[tie, int(rng.choice(j))] = X[tie, c]
            near = rng.random(n) < 0.2
            for k in range(K):
                if k != c:
                    X[near, k] = np.nextafter(X[near, c], -np.inf) if k < c else X[near, k]
            X[rng.random(n) < 0.05] = np.eye(K)[c] * (1 - 1e-300) + 1e-300 / K
            X /= X.sum(1, keepdims=True)
            keep = np.array([argmax_first(x) == c for x in X])
            X = X[keep]
            if X.shape[0] == 0:
                continue
            cell = rng.integers(0, 3, X.shape[0])                  # three fine cells of class c
            nn, S, A = DPT.cell_stats(X, cell, 3)
            for f in range(3):
                if nn[f]:
                    assert np.all(S[f, c] >= S[f]), (K, c, S[f])
                    assert np.all(S[f, c] / nn[f] >= S[f] / nn[f])
            used = nn > 0
            fine = DPT.FinePartition(K=K, cell_class=np.full(int(used.sum()), c), centroid=DPT.smooth(
                S[used] / nn[used, None], c), mean=S[used] / nn[used, None], n=nn[used], S=S[used], A=A[used],
                fallback=np.zeros(int(used.sum()), bool))
            # pad the other classes with fallback cells so the partition is valid
            blocks = []
            for cc in range(K):
                if cc == c:
                    blocks.append(fine)
                else:
                    u = np.full(K, 1.0 / K)
                    blocks.append(DPT.FinePartition(K=K, cell_class=np.array([cc]), centroid=DPT.smooth(u, cc)[None],
                                                    mean=u[None], n=np.array([0]), S=np.zeros((1, K)),
                                                    A=np.zeros(1), fallback=np.array([True])))
            cat = lambda a: np.concatenate([getattr(b, a) for b in blocks])  # noqa: E731
            full = DPT.FinePartition(K=K, cell_class=cat("cell_class"), centroid=cat("centroid"), mean=cat("mean"),
                                     n=cat("n"), S=cat("S"), A=cat("A"), fallback=cat("fallback"))
            full.validate()
            lab = np.arange(full.F)
            lab[full.cell_class == c] = int(np.flatnonzero(full.cell_class == c)[0])      # merge all cells of c
            pol = DRL.Policy(recipient=1, fine=full, cell_token=DRL.canonical_tokens(full, lab))
            t = int(pol.cell_token[np.flatnonzero(full.cell_class == c)[0]])
            assert np.all(pol.token_S[t, c] >= pol.token_S[t])
            q = pol.token_proto[t]
            assert q[c] - np.delete(q, c).max() > 9e-13


def _adversarial_inputs(K, absent, rng):
    """Deployment inputs never seen in fitting: one-hots (incl. the absent class), every exact 2-way tie, the K-way
    tie, ties involving the absent class, 1-ulp tie breaks, denormals, -0.0, accepted sum errors, near-uniform and
    spiky rows, rows far from any centroid."""
    rows = [np.eye(K)[j] for j in range(K)]
    for i, j in itertools.combinations(range(K), 2):
        v = np.zeros(K)
        v[i] = v[j] = 0.5
        rows.append(v)
        w = v.copy()
        w[j] = np.nextafter(0.5, 1.0)                              # a later index wins by one ulp
        rows.append(w / w.sum())
    rows.append(np.full(K, 1.0 / K))
    v = np.full(K, 5e-324)
    v[K - 1] = 1.0
    rows.append(v)
    v = -0.0 * np.ones(K)
    v[0] = 1.0
    rows.append(v)
    v = np.full(K, 1.0 / K)
    v[0] += 9e-10                                                  # sum 1 + 9e-10: accepted (INPUT_SUM_TOL 1e-9)
    rows.append(v)
    v = np.full(K, 1.0 / K)
    v[K - 1] -= 9e-10                                              # sum 1 - 9e-10: accepted; class 0 by first index
    rows.append(v)
    if absent is not None:
        v = np.full(K, 0.1 / max(K - 2, 1))
        v[absent] = v[(absent + 1) % K] = 0.45
        rows.append(v / v.sum())                                   # tie between absent class and another
        v = np.full(K, 0.5 / (K - 1))
        v[absent] = 0.5
        rows.append(v)                                             # absent class strictly predicted
    rows += list(rng.dirichlet(np.full(K, 0.02), size=200))        # spiky
    rows += list(rng.dirichlet(np.full(K, 200.0), size=200))       # near-uniform, far from typical centroids
    return np.array(rows, np.float64)


def _check_release(RL, pol, X):
    tok, q, dec = RL.encode(pol, X)
    for p, t, qq, dd in zip(X, tok, q, dec):
        d = argmax_first(p)
        assert dd == d, (p, dd, d)                                  # released decision = teacher decision
        assert pol.token_class[t] == d                              # the decision is the token's class
        assert np.array_equal(qq, pol.token_proto[t])               # decoded vector is the token prototype
        assert qq[d] > np.delete(qq, d).max()                       # strict: any tie convention recovers d
        assert argmax_first(qq) == d and argmax_last(qq) == d
    return tok, q, dec


def _decision_fixture(seed=0, n=900):
    KM = _mod("qpc.kmeans")
    rng = np.random.default_rng(seed)
    P1 = teacher_like(n, 2, 10 + seed, ties=20, zeros=10)
    P2 = teacher_like(int(n * 1.3), 6, 20 + seed, ties=30, zeros=20, absent=5)[:n]
    s = rng.integers(0, 2, n)
    s = np.where(rng.random(n) < 0.6, (P2[:, 0] > P2[:, 1]).astype(int), s)
    f1 = KM.fit_recipient(P1, None, 2, 8).partition
    f2 = KM.fit_recipient(P2, None, 6, 8).partition
    return P1, P2, s, f1, f2


def test_decision_preserved_on_adversarial_inputs_for_every_family_and_lambda():
    """Theorem 1 (MATH_REVIEW 1.3) on the deployed path qpc.release.encode -> dpc assign_fine / token tables: for
    CLASS, FINE-TASK, LOCAL, SEQ-12, SEQ-21 and JOINT at every registered lambda and at an extreme lambda, the released
    decision equals the first-index teacher argmax and the decoded vector has that class as STRICT argmax, on fitting
    rows and on adversarial inputs never seen in fitting (ties, absent class, denormals, out-of-support rows). The
    optimiser and the privacy weight cannot affect the property: it depends only on within-class grouping."""
    CP = _mod("qpc.compress")
    RL = _mod("qpc.release")
    P1, P2, s, f1, f2 = _decision_fixture()
    rng = np.random.default_rng(1)
    X1, X2 = _adversarial_inputs(2, None, rng), _adversarial_inputs(6, 5, rng)
    assert 5 not in {argmax_first(p) for p in P2}                  # class 5 is absent from fitting rows
    checked = 0
    for fam in ("CLASS", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT"):
        lams = [None] if fam in ("CLASS", "FINE-TASK") else [LAM_GRID[0], LAM_GRID[3], 50.0]
        for lam in lams:
            pair, _ = CP.fit_policy_pair(fam, f1, f2, P1, None, P2, None, s, 3, 4, lam)
            for pol, P, X in ((pair.p1, P1, X1), (pair.p2, P2, X2)):
                _check_release(RL, pol, P)
                tok, q, dec = _check_release(RL, pol, X)
                checked += len(P) + len(X)
            fb = pair.p2.token_fallback
            assert fb.sum() == 1 and pair.p2.token_class[fb][0] == 5
            assert np.array_equal(pair.p2.token_proto[fb][0], ref_smooth(np.full(6, 1 / 6), 5))
    assert checked > 20000


def test_decision_preserved_for_arbitrary_within_class_groupings():
    """The proof does not use the objective: ANY grouping of fine cells that stays within predicted classes (here 200
    random groupings, including all-in-one per class and identity) yields a release with exact decisions and strict
    decoded argmax on adversarial inputs."""
    RL = _mod("qpc.release")
    P1, P2, s, f1, f2 = _decision_fixture(seed=3)
    rng = np.random.default_rng(5)
    X2 = _adversarial_inputs(6, 5, rng)
    for g in range(200):
        lab = np.empty(f2.F, np.int64)
        for c in range(6):
            cells = np.flatnonzero(f2.cell_class == c)
            k = int(rng.integers(1, len(cells) + 1))
            blocks = rng.integers(0, k, len(cells))
            first = {}
            for j, b in zip(cells, blocks):
                lab[j] = first.setdefault(int(b), int(j))
        pol = RL.make_policy(2, f2, lab)
        _check_release(RL, pol, P2)
        _check_release(RL, pol, X2)


def test_routing_never_leaves_the_predicted_class():
    """Lemma 1: deployment chooses among the cells of the row's own predicted class (KL arithmetic only picks WHICH
    cell); a class with a single (or fallback) cell maps to it. Checked against the stored partition directly."""
    DPT = _mod("dpc.partition")
    P1, P2, s, f1, f2 = _decision_fixture(seed=4)
    X = np.vstack([P2, _adversarial_inputs(6, 5, np.random.default_rng(9))])
    cell = DPT.assign_fine(X, None, f2)
    d = np.array([argmax_first(p) for p in X])
    assert np.array_equal(f2.cell_class[cell], d)


def test_guards_refuse_class_mixing_nonstrict_prototypes_and_tampering():
    """The runtime guards that make the invariants hold for every constructed or loaded policy: a token mixing two
    predicted classes is refused; a class without a cell is refused; statistics whose mean does not have the cell's
    class as argmax are refused; a tampered stored prototype is refused at load; a wrong decision array is refused."""
    DPT = _mod("dpc.partition")
    RL = _mod("qpc.release")
    KM = _mod("qpc.kmeans")
    P1, P2, s, f1, f2 = _decision_fixture(seed=6)
    lab = np.arange(f2.F)
    c0, c1 = np.flatnonzero(f2.cell_class == 0)[0], np.flatnonzero(f2.cell_class == 1)[0]
    lab[c1] = c0                                                   # a token spanning classes 0 and 1
    with pytest.raises(ValueError):
        RL.make_policy(2, f2, lab)
    with pytest.raises(ValueError):                                # class 5 has no cell at all
        KM.FinePartition(K=6, cell_class=f2.cell_class[:-1], centroid=f2.centroid[:-1], mean=f2.mean[:-1],
                         n=f2.n[:-1], S=f2.S[:-1], A=f2.A[:-1], fallback=f2.fallback[:-1]).validate()
    bad_S = f2.S.copy()
    bad_S[c0] = bad_S[c0][::-1]                                   # statistics of a cell whose mean is not class 0
    bad = KM.FinePartition(K=6, cell_class=f2.cell_class, centroid=f2.centroid, mean=f2.mean, n=f2.n, S=bad_S,
                           A=f2.A, fallback=f2.fallback)
    with pytest.raises(ValueError):
        RL.make_policy(2, bad)
    pol = RL.make_policy(2, f2)
    z = pol.to_dict()
    z["token_proto"][0][0] = float(np.nextafter(z["token_proto"][0][0], 1.0))
    with pytest.raises(ValueError):
        RL.Policy.from_dict(z)
    with pytest.raises(Exception):
        RL.encode(pol, P2, (P2.argmax(1) + 1) % 6)
    assert DPT.INPUT_SUM_TOL == 1e-9 and DPT.EPS == 1e-12


def test_input_boundary_accepted_rows_preserve_and_others_are_refused():
    """The guarantee is stated for ACCEPTED inputs (finite, nonnegative, |sum - 1| <= 1e-9). Rows outside that set
    are refused loudly, never released with a silently changed decision."""
    RL = _mod("qpc.release")
    P1, P2, s, f1, f2 = _decision_fixture(seed=8)
    pol = RL.make_policy(1, f1)
    ok = np.array([[0.5 + 4.9e-10, 0.5 + 4.9e-10], [0.5, 0.5], [1.0 + 9e-10, 0.0]])
    _check_release(RL, pol, ok)
    for bad in ([0.5 + 2e-9, 0.5], [np.nan, 1.0], [1.1, -0.1], [np.inf, 0.0]):
        with pytest.raises(ValueError):
            RL.encode(pol, np.array([bad]))


def test_absent_class_zero_recall_is_inherited_not_repaired():
    """Class preservation copies the teacher's decisions, so every confusion matrix, accuracy and recall equals the
    teacher's: a class the teacher never predicts keeps recall 0 under every code. An assessment row the teacher
    does predict as the absent class goes to the reserved fallback token, whose decoded vector is smooth(uniform)."""
    CP = _mod("qpc.compress")
    RL = _mod("qpc.release")
    P1, P2, s, f1, f2 = _decision_fixture(seed=9)
    rng = np.random.default_rng(2)
    y = np.where(rng.random(len(P2)) < 0.15, 5, P2.argmax(1))     # true class 5 exists, the teacher never says 5
    pair, _ = CP.fit_policy_pair("JOINT", f1, f2, P1, None, P2, None, s, 3, 4, 0.06)
    _, _, dec = RL.encode(pair.p2, P2)
    teacher = P2.argmax(1)
    cm_t = np.zeros((6, 6), int)
    cm_r = np.zeros((6, 6), int)
    np.add.at(cm_t, (y, teacher), 1)
    np.add.at(cm_r, (y, dec), 1)
    assert np.array_equal(cm_t, cm_r)
    assert cm_r[5, 5] == 0 and cm_r[5].sum() > 0                  # recall of class 5 is 0 and stays 0
    x = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])
    tok, q, dec = _check_release(RL, pair.p2, x)
    assert pair.p2.token_fallback[tok[0]] and dec[0] == 5
    assert q[0].max() - np.sort(q[0])[-2] < 1e-11                  # only an eps margin: no confidence for class 5


def test_sampled_row_checks_cannot_distinguish_a_construction_without_the_eps_ec_term():
    """Why passing many sampled row checks is not a proof: a variant smoothing WITHOUT the eps*e_c term keeps the
    strict argmax on every one of 200,000 random continuous rows (exact ties have probability 0 there), yet fails on a
    cell whose members tie with a later index exactly. Only the argument over ALL inputs separates the two."""
    rng = np.random.default_rng(12)

    def smooth_no_ec(m):
        return (m + EPS) / (1.0 + len(m) * EPS)

    X = rng.dirichlet(np.full(6, 0.7), size=200_000)
    d = X.argmax(1)
    for c in range(6):
        w = d == c
        for blk in np.array_split(np.flatnonzero(w), 40):
            q = smooth_no_ec(X[blk].mean(0))
            assert q[c] > np.delete(q, c).max()                    # passes every sampled check
    tie = np.array([[0.1, 0.45, 0.45, 0.0, 0.0, 0.0]])             # first-index decision 1, exact tie with 2
    q = smooth_no_ec(tie.mean(0))
    assert not q[1] > q[2]                                         # the variant fails on the adversarial input
    q = ref_smooth(tie.mean(0), 1)
    assert q[1] > q[2]                                             # the registered rule does not


# ==================================================================================================================
# S2  DATA PROCESSING AND WHAT IT DOES NOT IMPLY
# ==================================================================================================================

def test_data_processing_and_containment_on_a_fitted_code():
    """On the fitting law of a privacy-trained code: I(S; C) <= I(S; fine) (merging), I(S; C) >= I(S; d) (containment),
    pair likewise; the chain rule I(S; C1, C2) = I(S; d1, d2) + I(S; C1, C2 | d1, d2) holds exactly."""
    CP = _mod("qpc.compress")
    RL = _mod("qpc.release")
    DPT = _mod("dpc.partition")
    P1, P2, s, f1, f2 = _decision_fixture(seed=11, n=700)
    pair, _ = CP.fit_policy_pair("SEQ-21", f1, f2, P1, None, P2, None, s, 3, 4, 0.08)
    t1, _, d1 = RL.encode(pair.p1, P1)
    t2, _, d2 = RL.encode(pair.p2, P2)
    c1, c2 = DPT.assign_fine(P1, None, f1), DPT.assign_fine(P2, None, f2)
    assert ref_mi(s, t1) <= ref_mi(s, c1) + 1e-15 and ref_mi(s, t2) <= ref_mi(s, c2) + 1e-15
    assert ref_mi(s, t1, t2) <= ref_mi(s, c1, c2) + 1e-15
    assert ref_mi(s, t2) >= ref_mi(s, d2) - 1e-15 and ref_mi(s, t1, t2) >= ref_mi(s, d1, d2) - 1e-15
    lhs = ref_mi(s, t1, t2)
    rhs = ref_mi(s, d1, d2) + ref_cmi(s, d1 * 6 + d2, t1, t2)
    assert abs(lhs - rhs) < 1e-12


def _laplace_reader_auc(train_code, train_s, test_code, test_s, alpha=0.5):
    """Exact finite-code reader with additive smoothing (as in the source slate), held-out Mann-Whitney AUC."""
    prior = train_s.mean()
    score = {}
    for c in np.unique(train_code):
        w = train_code == c
        score[c] = (train_s[w].sum() + alpha) / (w.sum() + 2 * alpha)
    sc = np.array([score.get(c, prior) for c in test_code])
    pos, neg = sc[test_s == 1], sc[test_s == 0]
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def test_data_processing_does_not_bound_a_fitted_readers_auc():
    """Data processing orders POPULATION information, not the held-out AUC of a FITTED reader: a coarsened code
    (strictly less population information) gives a finite-sample exact-code reader a HIGHER held-out AUC than the
    fine code it was derived from. So no SEX-AUC bound follows from the data-processing facts."""
    rng = np.random.default_rng(21)
    L, G = 400, 4                                                  # fine cells, coarse groups (cell // 100)
    wins = 0
    for rep in range(10):
        n_tr, n_te = 600, 4000
        cell_tr, cell_te = rng.integers(0, L, n_tr), rng.integers(0, L, n_te)
        base = np.array([0.2, 0.4, 0.6, 0.8])
        jitter = rng.normal(0, 0.05, L)                            # fine cells carry slightly MORE information
        p = lambda c: np.clip(base[c * G // L] + jitter[c], 0.01, 0.99)  # noqa: E731
        s_tr = (rng.random(n_tr) < p(cell_tr)).astype(int)
        s_te = (rng.random(n_te) < p(cell_te)).astype(int)
        fine = _laplace_reader_auc(cell_tr, s_tr, cell_te, s_te)
        coarse = _laplace_reader_auc(cell_tr * G // L, s_tr, cell_te * G // L, s_te)
        wins += coarse > fine + 0.01
    assert wins >= 9


# ==================================================================================================================
# S3  PRIVACY OBJECTIVES AND FITTED MI
# ==================================================================================================================

def test_objectives_are_decision_floor_plus_within_class_terms():
    """For every class-preserving map on the same fitting rows, I_i = I(S; d_i) + I(S; C_i | d_i) and
    I12 = I(S; d1, d2) + I(S; C1, C2 | d1, d2), with the decision terms identical across maps. So F_local and F_joint
    differ between maps only through D and the within-class (conditional) terms: the objective cannot, and does not
    try to, remove the decision floor. Also pins the registered weights."""
    CP = _mod("qpc.compress")
    R1, R2, s = fixture_six()
    rng = np.random.default_rng(3)
    floors = set()
    for _ in range(25):
        labs = []
        for R in (R1, R2):
            lab = np.empty(R.F, np.int64)
            for c in range(R.K):
                cells = np.flatnonzero(R.cls == c)
                first = {}
                for j, b in zip(cells, rng.integers(0, 2, len(cells))):
                    lab[j] = first.setdefault(int(b), int(j))
            labs.append(lab)
        t = terms_from_labels(R1, R2, s, *labs)
        t1, t2 = labs[0][R1.f], labs[1][R2.f]
        d1, d2 = R1.cls[R1.f], R2.cls[R2.f]
        assert abs(t["I1"] - (ref_mi(s, d1) + ref_cmi(s, d1, t1))) < 1e-12
        assert abs(t["I12"] - (ref_mi(s, d1, d2) + ref_cmi(s, d1 * 6 + d2, t1, t2))) < 1e-12
        floors.add(round(ref_mi(s, d1, d2), 15))
        for lam in LAM_GRID:
            fv = CP.F_values(t, lam)
            assert abs(fv["F_joint"] - F_joint(t, lam)) < 1e-15 and abs(fv["F_local"] - F_local(t, lam)) < 1e-15
    assert len(floors) == 1
    for lam in LAM_GRID:
        assert CP.W_joint(lam).__dict__ == {"wD1": 1.0, "wD2": 1.0, "wI1": lam / 2, "wI2": lam / 2, "w12": lam}
        assert CP.W_local(lam).__dict__ == {"wD1": 1.0, "wD2": 1.0, "wI1": lam / 2, "wI2": lam / 2, "w12": 0.0}


def test_plugin_mi_null_bias_at_the_i8o64_alphabets():
    """Under a true null (S independent of the code; N = 15,434 fitting rows; P(S=1) = 0.33) plug-in MI is positive
    and close to (L - 1)/(2N) at the i8o64 alphabets (income 2 x 8 = 16 tokens, occupation 5 x 64 = 320 tokens); the
    pair alphabet is far larger and its null MI is of order 0.1 nats. qpc's permutation-null receipt measures the same
    quantity at a given code. At the registered lambdas, lambda times the pair null MI is the same order as the
    0.006-nat headroom rule, so lambda * I12 partly prices alphabet size."""
    CP = _mod("qpc.compress")
    N = 15434
    rng = np.random.default_rng(31)
    s = (rng.random(N) < 0.33).astype(np.int64)
    t1 = rng.integers(0, 16, N)
    t2 = rng.integers(0, 320, N)
    i1, i2, i12 = fast_mi(s, t1), fast_mi(s, t2), fast_mi(s, t1 * 321 + t2)
    occ12 = len(np.unique(t1 * 321 + t2))
    assert 0.5 * 15 / (2 * N) < i1 < 2.0 * 15 / (2 * N)
    assert 0.7 * 319 / (2 * N) < i2 < 1.3 * 319 / (2 * N)
    assert 0.05 < i12 < 1.5 * (occ12 - 1) / (2 * N)
    pn = CP.perm_null_mi(t1, t2, 321, s, n_perm=20)
    assert abs(pn["I2"]["mean"] - 319 / (2 * N)) < 0.003
    assert abs(pn["I12"]["mean"] - i12) < 0.01
    weighted = {lam: lam * pn["I12"]["mean"] for lam in LAM_GRID}
    assert weighted[0.025] > 0.001 and weighted[0.1] > 0.006       # comparable to the 0.006-nat headroom


def test_fitted_mi_below_its_permutation_null_is_selection_not_protection():
    """Under a TRUE null (S independent of every teacher score), privacy-trained maps reach fitted MI far below the
    permutation-null mean AT THEIR OWN CODE (they select groupings whose fitting-row counts happen to balance S). On
    fresh null rows the same codes show ordinary null-size MI. A fitted MI decrease, or a fitted value below its null,
    is therefore not evidence of removed information."""
    KM = _mod("qpc.kmeans")
    CP = _mod("qpc.compress")
    RL = _mod("qpc.release")
    N = 1200
    for seed in range(2):
        P1, P2 = teacher_like(2 * N, 2, 10 + seed), teacher_like(2 * N, 6, 20 + seed)
        s = np.random.default_rng(30 + seed).integers(0, 2, 2 * N)
        fit, ho = slice(0, N), slice(N, 2 * N)
        f1 = KM.fit_recipient(P1[fit], None, 2, 8).partition
        f2 = KM.fit_recipient(P2[fit], None, 6, 16).partition
        pair, rec = CP.fit_policy_pair("JOINT", f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 6, 5.0)
        pn = rec["perm_null_mi_fit"]
        assert rec["final"]["I12"] < 0.5 * pn["I12"]["mean"]
        t1, _, _ = RL.encode(pair.p1, P1[ho])
        t2, _, _ = RL.encode(pair.p2, P2[ho])
        held = fast_mi(s[ho], t1 * pair.p2.T + t2)
        assert held > 2 * rec["final"]["I12"]
        lpair, lrec = CP.fit_policy_pair("LOCAL", f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 6, 1.0)
        assert lrec["final"]["I2"] < 0.6 * lrec["perm_null_mi_fit"]["I2"]["mean"]


# ==================================================================================================================
# S4  THE CORRECTED SEQUENTIAL DESIGN
# ==================================================================================================================

def _stage1_value(R1, R2, s, a, lab_a, lam, counterpart):
    """Stage-one F_joint with recipient b at its CLASS-ONLY release ('class') or, for contrast, the old surrogate
    D_a + 1.5 lam I_a (b treated as releasing nothing; 'old')."""
    b = 2 if a == 1 else 1
    Rb = (R1, R2)[b - 1]
    labs = {a: lab_a, b: Rb.class_labels()}
    t = terms_from_labels(R1, R2, s, labs[1], labs[2])
    if counterpart == "class":
        return F_joint(t, lam), t
    return t[f"D{a}"] + 1.5 * lam * t[f"I{a}"], t


@pytest.mark.parametrize("fixture", ["six", "decision"])
def test_sequential_stage_one_uses_the_class_only_counterpart_never_a_constant(fixture):
    """SEQ-ab stage one is fitted under the actual F_joint with recipient b at its CLASS-ONLY release: the receipt's
    stage-one value equals my row recomputation with b = one token per predicted class; b's disclosed decision term
    I(S; d_b) is present (nonzero), i.e. the counterpart is not a constant/no-release view; the stage weights are
    W_joint (w12 = lam), not the dpc surrogate; the first map is frozen in stage two."""
    CP = _mod("qpc.compress")
    R1, R2, s = fixture_six() if fixture == "six" else fixture_decision_carries_s()
    caps = (2, 1)
    for lam in LAM_GRID + (1.0,):
        for fam, a in (("SEQ-12", 1), ("SEQ-21", 2)):
            b = 2 if a == 1 else 1
            pair, rec = fit_recip(fam, R1, R2, s, *caps, lam)
            corr = rec["baseline_correction"]
            assert corr["counterpart"] == "CLASS-ONLY"
            assert rec["stages"][0]["weights"] == CP.W_joint(lam).__dict__
            assert rec["stages"][1]["weights"] == CP.W_joint(lam).__dict__
            lab_a = labels_of(pair)[a - 1]
            v, t = _stage1_value(R1, R2, s, a, lab_a, lam, "class")
            assert abs(v - corr["stage1_F_joint_with_class_counterpart"]) < 1e-12
            assert abs(t["I12"] - corr["I12_with_class_counterpart"]) < 1e-12
            Ib = ref_mi(s, (R1, R2)[b - 1].cls[(R1, R2)[b - 1].f])
            assert abs(corr[f"I{b}_class"] - Ib) < 1e-12
            if fixture == "decision" and b == 2:
                assert Ib > 0.1                                    # d_2 carries S: a constant would hide this
            assert rec["stages"][0]["recipients"] == [a] and rec["stages"][1]["recipients"] == [b]


def test_sequential_correction_term_is_lambda_times_conditional_mi_of_the_other_decision():
    """Up to map-independent constants, corrected stage one minus the old surrogate D_a + 1.5 lam I_a equals
    lam * I(S; d_b | C_a) >= 0 for EVERY first map C_a (checked on random maps). The corrected rule therefore penalises
    first-map clues that are complementary to the disclosed d_b and penalises less those redundant with it. Since every
    admissible C_b refines d_b, I(S; C_a, C_b) >= I(S; C_a, d_b): the class-only view is the least the contract ever
    discloses. On the designed fixture the correction changes the stage-one map and lowers its F_joint."""
    R1, R2, s = fixture_decision_carries_s()
    rng = np.random.default_rng(4)
    const = None
    for _ in range(30):
        lab = np.empty(R1.F, np.int64)
        for c in range(R1.K):
            cells = np.flatnonzero(R1.cls == c)
            first = {}
            for j, bb in zip(cells, rng.integers(0, 3, len(cells))):
                lab[j] = first.setdefault(int(bb), int(j))
        lam = 0.7
        vc, t = _stage1_value(R1, R2, s, 1, lab, lam, "class")
        vo, _ = _stage1_value(R1, R2, s, 1, lab, lam, "old")
        cmi = ref_cmi(s, lab[R1.f], R2.cls[R2.f])
        k = vc - vo - lam * cmi
        const = k if const is None else const
        assert abs(k - const) < 1e-12                              # same constant for every map
        assert cmi >= -1e-15
        lab2 = np.arange(R2.F)                                     # finest admissible C_2 refines d_2
        assert ref_mi(s, lab[R1.f], lab2[R2.f]) >= ref_mi(s, lab[R1.f], R2.cls[R2.f]) - 1e-15
    changed = False
    for lam in (0.3, 1.0, 3.0):
        pair, rec = fit_recip("SEQ-12", R1, R2, s, 2, 1, lam)
        old = rec["baseline_correction"]["old_rule_stage1"]
        assert old["corrected_minus_old_F_joint"] <= 1e-12
        changed |= not old["same_map_as_corrected"]
    assert changed


# ==================================================================================================================
# S5  JOINT: WITNESS DOMINANCE AND ITS LIMITS
# ==================================================================================================================

def _joint_fixture(seed=4, N=900, Nh=1800):
    KM = _mod("qpc.kmeans")
    RL = _mod("qpc.release")
    rng = np.random.default_rng(seed)
    P1 = teacher_like(N + Nh, 2, 100 + seed)
    P2 = teacher_like(N + Nh, 6, 200 + seed)
    lin = 2.0 * (P1[:, 1] - 0.5) + 1.5 * (P2.max(1) - 0.5) + 0.8 * (P2.argmax(1) == 0)
    s = (rng.random(N + Nh) < 1 / (1 + np.exp(-lin))).astype(np.int64)
    fit, ho = slice(0, N), slice(N, N + Nh)
    f1 = KM.fit_recipient(P1[fit], None, 2, 16).partition
    f2 = KM.fit_recipient(P2[fit], None, 6, 16).partition
    direct = (RL.make_policy(1, KM.fit_recipient(P1[fit], None, 2, 4).partition),
              RL.make_policy(2, KM.fit_recipient(P2[fit], None, 6, 4).partition))
    return P1, P2, s, fit, ho, f1, f2, direct


def _fast_terms(pol1, pol2, P1, P2, s):
    RL = _mod("qpc.release")
    t1, q1, _ = RL.encode(pol1, P1)
    t2, q2, _ = RL.encode(pol2, P2)
    lp = lambda P, Q: float(np.mean(np.sum(np.where(P > 0, P * (np.log(np.where(P > 0, P, 1)) - np.log(Q)), 0), 1)))  # noqa: E731
    return {"D1": lp(P1, q1), "D2": lp(P2, q2), "I1": fast_mi(s, t1), "I2": fast_mi(s, t2),
            "I12": fast_mi(s, t1 * pol2.T + t2)}


def test_witness_dominance_holds_on_the_fitting_F_joint_over_the_registered_grid():
    """Theorem 2 (MATH_REVIEW 5.1): at every registered lambda, JOINT's final F_joint (recomputed by me from the
    released fitting rows) is <= that of each UNCHANGED witness passed in (FINE-TASK, LOCAL, SEQ-12, SEQ-21 at the
    same caps, lambda and fine partitions). The candidate set is 5 refined starts + 4 unchanged witnesses; passed-in
    and recomputed witnesses give the same JOINT pair."""
    CP = _mod("qpc.compress")
    RL = _mod("qpc.release")
    P1, P2, s, fit, ho, f1, f2, _ = _joint_fixture()
    for lam in LAM_GRID:
        wits = {f: CP.fit_policy_pair(f, f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 4,
                                      None if f == "FINE-TASK" else lam)[0]
                for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")}
        J, rec = CP.fit_policy_pair("JOINT", f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 4, lam, witnesses=wits)
        assert len(rec["candidates"]) == 9 and rec["summary"]["starts"] == 5
        assert [c["kind"] for c in rec["candidates"]].count("unchanged") == 4
        fj = F_joint(ref_terms_rows(J.p1, J.p2, P1[fit], P2[fit], s[fit], RL.encode)[0], lam)
        for f, w in wits.items():
            fw = F_joint(ref_terms_rows(w.p1, w.p2, P1[fit], P2[fit], s[fit], RL.encode)[0], lam)
            assert fj <= fw + 1e-12, (lam, f, fj, fw)
            assert rec["witness_dominance"][f]["final_minus_witness"] <= 0.0
        J2, _ = CP.fit_policy_pair("JOINT", f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 4, lam)
        assert J2.fingerprint() == J.fingerprint()


def test_dominance_is_only_on_the_fitting_objective_not_components_heldout_or_direct_task():
    """What witness dominance does NOT give (MATH_REVIEW 5.2), each shown on synthetic data:
    (a) DIRECT-TASK at the same caps (direct k-means, outside the fine-state family) has LOWER fitting F_joint than
        JOINT at every registered lambda on this fixture;
    (b) on held-out rows JOINT's F_joint exceeds some witness's at some registered lambda;
    (c) a lower F_joint can come with HIGHER distortion D or a higher individual/pair MI than a witness."""
    CP = _mod("qpc.compress")
    P1, P2, s, fit, ho, f1, f2, direct = _joint_fixture()
    held_rev, comp_rev = 0, 0
    for lam in LAM_GRID:
        wits = {f: CP.fit_policy_pair(f, f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 4,
                                      None if f == "FINE-TASK" else lam)[0]
                for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")}
        J, rec = CP.fit_policy_pair("JOINT", f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 4, lam, witnesses=wits)
        tJ = _fast_terms(J.p1, J.p2, P1[fit], P2[fit], s[fit])
        tD = _fast_terms(*direct, P1[fit], P2[fit], s[fit])
        assert F_joint(tD, lam) < F_joint(tJ, lam), lam                       # (a)
        hJ = _fast_terms(J.p1, J.p2, P1[ho], P2[ho], s[ho])
        for f, w in wits.items():
            tw = _fast_terms(w.p1, w.p2, P1[fit], P2[fit], s[fit])
            assert F_joint(tJ, lam) <= F_joint(tw, lam) + 1e-12
            hw = _fast_terms(w.p1, w.p2, P1[ho], P2[ho], s[ho])
            held_rev += F_joint(hJ, lam) > F_joint(hw, lam) + 1e-9            # (b)
            comp_rev += ((tJ["D1"] + tJ["D2"] > tw["D1"] + tw["D2"] + 1e-9)
                         or any(tJ[k] > tw[k] + 1e-9 for k in ("I1", "I2", "I12")))  # (c)
    assert held_rev >= 1 and comp_rev >= 1


def test_joint_search_has_more_starts_than_the_sequential_arms():
    """The computational asymmetry (prompt section 7): JOINT refines five starts (four of them the fitted witnesses)
    and keeps the four unchanged witnesses; each SEQ arm runs one greedy + refinement path per stage. JOINT's
    candidate set contains the SEQ solutions, so JOINT <= SEQ on F_joint is by construction, not a finding."""
    CP = _mod("qpc.compress")
    P1, P2, s, fit, ho, f1, f2, _ = _joint_fixture(N=600, Nh=10)
    lam = 0.06
    seq = {f: CP.fit_policy_pair(f, f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 4, lam) for f in
           ("SEQ-12", "SEQ-21")}
    J, rec = CP.fit_policy_pair("JOINT", f1, f2, P1[fit], None, P2[fit], None, s[fit], 4, 4, lam)
    assert tuple(rec["starts"]) == CP.JOINT_START_ORDER and len(CP.JOINT_START_ORDER) == 5
    for f, (pp, r) in seq.items():
        assert r["summary"]["starts"] is None and len(r["stages"]) == 2
        assert rec["work"]["refine_candidates"] > r["work"]["refine_candidates"]
        assert rec["winner"]["F_joint"] <= r["final"]["F_joint"] + 1e-12


def test_registered_search_constants_unchanged():
    """No optimiser change: the constants cbp reuses from qpc.compress / qpc.kmeans are the registered ones."""
    CP = _mod("qpc.compress")
    KM = _mod("qpc.kmeans")
    assert CP.TOL == 1e-12 and CP.TIE_TOL == 1e-12 and CP.SWEEPS == 5
    assert CP.FAMILIES == ("CLASS", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT")
    assert CP.WITNESSES == ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
    assert CP.JOINT_START_ORDER == ("JOINT-GREEDY", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
    assert CP.PERM_SEED == 20261006 and CP.N_PERM == 100
    assert KM.EPS == 1e-12
    assert [CP.config_id("U", "JOINT", 8, 64, lam) for lam in LAM_GRID] == [
        "U|JOINT|i8o64|l0.01", "U|JOINT|i8o64|l0.025", "U|JOINT|i8o64|l0.04", "U|JOINT|i8o64|l0.06",
        "U|JOINT|i8o64|l0.08", "U|JOINT|i8o64|l0.1"]


# ==================================================================================================================
# S6  REVIEW OF cbp.fit (role B): reuse parity, pass-through, no optimiser change (skips until cbp.fit exists)
# ==================================================================================================================

_META = {"teacher": "U", "seed": 1, "teacher_model_sha256": "a" * 64, "feature_names_sha256": "b" * 64}
_CACHE = {}


def _cbp_synth():
    """B's real-shaped synthetic generator at a reduced size (5,000 rows, 2,500 fitting rows); fine partitions income
    16 / occupation 96 per predicted class, so the (8, 64) caps bind. Cached for the module."""
    if "synth" not in _CACHE:
        import json as _json
        import tempfile
        from pathlib import Path
        FT = _mod("cbp.fit")
        PT = _mod("qpc.partition")
        T, tr, S = FT.synthetic_teacher(N=5000, n_fit=2500, seed=1)
        _, files = PT.fine_unit(T, tr, caps={1: 16, 2: 96})
        with tempfile.TemporaryDirectory() as d:
            files["fine.json"](Path(d) / "fine.json")
            fine = _json.loads((Path(d) / "fine.json").read_text())
        _CACHE["synth"] = (T, tr, S, fine)
    return _CACHE["synth"]


def _policy_text(files):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        files["policy.json"](Path(d) / "policy.json")
        return (Path(d) / "policy.json").read_text()


def _release_terms(out, tr, T, S, T2, lam):
    """D, I terms recomputed by me from a stored release (all rows) on the fitting rows."""
    t1, t2 = np.asarray(out["tok1"])[tr], np.asarray(out["tok2"])[tr]
    res = {}
    for i, t in ((1, t1), (2, t2)):
        P, q = np.asarray(T[f"p{i}"])[tr], np.asarray(out[f"q{i}"])[tr]
        res[f"D{i}"] = float(np.mean(np.sum(np.where(P > 0, P * (np.log(np.where(P > 0, P, 1)) - np.log(q)), 0), 1)))
        res[f"I{i}"] = fast_mi(S, t)
    res["I12"] = fast_mi(S, t1 * T2 + t2)
    return res


def test_cbp_fit_unit_is_the_unchanged_qpc_unit_plus_receipts():
    """cbp.fit.fit_unit = qpc.compress.fit_unit(family, fine, T, tr, S, 8, 64, lam, meta, witnesses): identical policy
    JSON, bitwise identical release arrays and an identical qpc record apart from the added "cbp" key and timing. So no
    optimiser, objective, tie rule, sweep cap or start differs from the source."""
    FT = _mod("cbp.fit")
    CP = _mod("qpc.compress")
    T, tr, S, fine = _cbp_synth()
    for fam, lam in (("LOCAL", 0.04), ("SEQ-21", 0.06)):
        cid = FT.config_id(fam, lam)
        r1, f1 = FT.fit_unit(fam, fine, T, tr, S, lam, {**_META, "config": cid})
        r0, f0 = CP.fit_unit(fam, fine, T, tr, S, 8, 64, lam, {**_META, "config": cid})
        assert _policy_text(f1) == _policy_text(f0)
        assert set(f1["release.npz"]) == set(f0["release.npz"])
        for k in f0["release.npz"]:
            assert np.asarray(f1["release.npz"][k]).tobytes() == np.asarray(f0["release.npz"][k]).tobytes(), k
        strip = lambda r: {k: v for k, v in r.items() if k not in ("cbp", "wall_seconds", "cpu_seconds")}  # noqa: E731
        assert json.dumps(strip(r1), sort_keys=True, default=str) == json.dumps(strip(r0), sort_keys=True,
                                                                               default=str)
        assert r1["cbp"]["qpc_call"] == {"function": "qpc.compress.fit_unit", "m1": 8, "m2": 64, "lam": lam,
                                         "witness_slots": []}


def test_cbp_fit_refuses_off_grid_reused_and_witnessless_jobs():
    FT = _mod("cbp.fit")
    T, tr, S, fine = _cbp_synth()
    with pytest.raises(ValueError):
        FT.fit_unit("LOCAL", fine, T, tr, S, 0.05, dict(_META))                  # off the registered grid
    with pytest.raises(ValueError):
        FT.fit_unit("LOCAL", fine, T, tr, S, 0.1, dict(_META))                   # reused endpoint, no refit reason
    with pytest.raises(ValueError):
        FT.fit_unit("JOINT", fine, T, tr, S, 0.06, dict(_META))                  # JOINT without its four witnesses
    with pytest.raises(ValueError):
        FT.fit_unit("FINE-TASK", fine, T, tr, S, 0.06, dict(_META))              # references are reused, not refit
    with pytest.raises(ValueError):
        FT.fit_unit("SEQ-12", fine, T, tr, S, 0.06, dict(_META), witnesses={"FINE-TASK": {}})
    assert FT.NEW_LAMS == (0.025, 0.04, 0.06, 0.08) and FT.REUSED_LAMS == (0.01, 0.1) and (FT.M1, FT.M2) == (8, 64)


def test_cbp_new_units_sequential_counterpart_and_joint_dominance_from_the_deployed_release():
    """At the real rate (8, 64) on B's synthetic generator, from the STORED release arrays (my recomputation):
    (i) SEQ-21 / SEQ-12 stage one equals F_joint with the other recipient's CLASS-ONLY release (its decision);
    (ii) the new JOINT unit's F_joint <= each unchanged witness (reused-style FINE-TASK + the three new same-lambda
    fits), and its asymmetry receipt records 5 search paths / 9 candidates against 1 / 1 for SEQ."""
    FT = _mod("cbp.fit")
    CP = _mod("qpc.compress")
    RL = _mod("qpc.release")
    T, tr, S, fine = _cbp_synth()
    lam = 0.06
    cls_rec, cls_files = CP.fit_unit("CLASS", fine, T, tr, S, 8, 64, None, {**_META, "config": "U|CLASS|i1o1"})
    cls_out = cls_files["release.npz"]
    ft_rec, ft_files = CP.fit_unit("FINE-TASK", fine, T, tr, S, 8, 64, None, {**_META, "config": FT.config_id(
        "FINE-TASK")})
    wit = {"FINE-TASK": json.loads(_policy_text(ft_files))}
    outs = {"FINE-TASK": ft_files["release.npz"]}
    T2 = None
    for fam, a in (("LOCAL", None), ("SEQ-12", 1), ("SEQ-21", 2)):
        rec, files = FT.fit_unit(fam, fine, T, tr, S, lam, {**_META, "config": FT.config_id(fam, lam)})
        wit[fam] = json.loads(_policy_text(files))
        outs[fam] = files["release.npz"]
        T2 = int(np.asarray(files["release.npz"]["alpha2"]))
        if a is not None:
            b = 2 if a == 1 else 1
            o = files["release.npz"]
            mixed = {f"tok{a}": o[f"tok{a}"], f"q{a}": o[f"q{a}"], f"tok{b}": cls_out[f"tok{b}"],
                     f"q{b}": cls_out[f"q{b}"]}
            t = _release_terms(mixed, tr, T, S, int(np.asarray(cls_out["alpha2"])) if b == 2 else T2, lam)
            v = F_joint(t, lam)
            got = rec["baseline_correction"]["stage1_F_joint_with_class_counterpart"]
            assert abs(v - got) <= 1e-12 * max(1.0, abs(v)), (fam, v, got)
            assert t[f"I{b}"] > 0                                   # the counterpart discloses d_b (not constant)
            asym = rec["cbp"]["computational_asymmetry"]
            assert asym["search_paths"] == 1 and asym["candidates_compared"] == 1
    jrec, jfiles = FT.fit_unit("JOINT", fine, T, tr, S, lam, {**_META, "config": FT.config_id("JOINT", lam)},
                               witnesses=wit)
    fj = F_joint(_release_terms(jfiles["release.npz"], tr, T, S, int(np.asarray(jfiles["release.npz"]["alpha2"])),
                                lam), lam)
    for f, o in outs.items():
        fw = F_joint(_release_terms(o, tr, T, S, int(np.asarray(o["alpha2"])), lam), lam)
        assert fj <= fw + 1e-12, (f, fj, fw)
    asym = jrec["cbp"]["computational_asymmetry"]
    assert asym["search_paths"] == 5 and asym["candidates_compared"] == 9 and asym["witnesses_consumed"] == 4
    assert all(v == "passed_in" for v in asym["witness_sources"].values())
    assert jrec["cbp"]["section13"]["reconstruction"]["structure_ok"]


def test_endpoint_parity_accepts_an_exact_replica_and_refuses_defects(tmp_path):
    """Reuse parity (lambda 0.01 / 0.1 endpoints): an exact replica passes every gate; each defect is caught by the
    gate that should catch it: other fitting SEX (objective terms), a one-ulp decoded value re-saved with a valid
    COMPLETE.json (release bitwise), other fine partitions, a wrong expected config, a wrong binding; for JOINT, a
    witness record whose F_joint differs from the one dominated, and a JOINT fitted with RECOMPUTED witnesses."""
    FT = _mod("cbp.fit")
    CP = _mod("qpc.compress")
    PT = _mod("qpc.partition")
    T, tr, S, fine = _cbp_synth()

    def make(fam, lam, root, wit=None):
        cid = FT.config_id(fam, lam) if fam not in ("FINE-TASK",) else FT.config_id(fam)
        rec, files = CP.fit_unit(fam, fine, T, tr, S, 8, 64, lam, {**_META, "config": cid}, witnesses=wit)
        rec.update({"seed": 1, "config": cid})
        u = root / FT.unit_name(1, cid)
        FT.save_unit(u, files, rec)
        return u, rec, cid

    u, rec, cid = make("LOCAL", 0.1, tmp_path)
    ok = FT.endpoint_parity(u, T, tr, S, fine, expected_config=cid, meta=_META)
    assert ok["ok"], ok["failed"]
    S2 = S.copy()
    S2[:40] = 1 - S2[:40]
    assert "fitting_terms_within_rtol" in FT.endpoint_parity(u, T, tr, S2, fine)["failed"]
    with np.load(u / "release.npz") as z:
        arr = {k: z[k].copy() for k in z.files}
    arr["q2"][0, 0] = np.nextafter(arr["q2"][0, 0], 1.0)
    ub = tmp_path / "ulp" / u.name
    FT.save_unit(ub, {"policy.json": (u / "policy.json").read_text(), "release.npz": arr},
                 json.loads((u / "record.json").read_text()))
    bad = FT.endpoint_parity(ub, T, tr, S, fine)
    assert bad["checks"]["complete_json_hashes"] and "release_bitwise" in bad["failed"]
    _, ff = PT.fine_unit(T, tr, caps={1: 16, 2: 80})
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        ff["fine.json"](Path(d) / "f.json")
        other = json.loads((Path(d) / "f.json").read_text())
    assert "fine_partition" in FT.endpoint_parity(u, T, tr, S, other)["failed"]
    assert "config_consistent" in FT.endpoint_parity(u, T, tr, S, fine, expected_config=FT.config_id("LOCAL",
                                                                                                   0.01))["failed"]
    assert "binding" in FT.endpoint_parity(u, T, tr, S, fine, meta={**_META, "teacher_model_sha256": "c" * 64})[
        "failed"]
    # JOINT reuse: dominance must be over the admitted witness units
    lam = 0.1
    wrec, wpol = {}, {}
    for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"):
        uw, rw, _ = make(f, None if f == "FINE-TASK" else lam, tmp_path / "w")
        wrec[f], wpol[f] = rw, json.loads((uw / "policy.json").read_text())
    uj, rj, cj = make("JOINT", lam, tmp_path / "j", wit=wpol)
    good = FT.endpoint_parity(uj, T, tr, S, fine, expected_config=cj, meta=_META, witness_records=wrec)
    assert good["ok"], good["failed"]
    tam = json.loads(json.dumps(wrec))
    tam["SEQ-12"]["final"]["F_joint"] += 1e-9
    assert "joint_witness_dominance" in FT.endpoint_parity(uj, T, tr, S, fine, witness_records=tam)["failed"]
    ur, _, _ = make("JOINT", lam, tmp_path / "recomputed")                        # witnesses recomputed, not passed
    assert "joint_witness_dominance" in FT.endpoint_parity(ur, T, tr, S, fine)["failed"]


# ==================================================================================================================
# S7  REVIEW OF cbp.audit (role D): composition rule and envelope; MI diagnostic (skips until cbp.audit exists)
# ==================================================================================================================

def _bank_rec(rng, views, base, sel_keys):
    a0 = {w: float(base + rng.choice([0.0, 0.01, 0.02, 0.05])) for w in views}
    return {"auc_seed0": a0, "auc": {w: a0[w] - float(rng.choice([0.0, 0.004, 0.03])) for w in views},
            "ce_seed0": {w: float(0.6 - rng.choice([0.0, 0.01, 0.02])) for w in views},
            "ce": {w: float(0.6 + rng.choice([0.0, 0.002])) for w in views},
            "selected": {w: f"att{int(rng.integers(5))}" for w in views},
            "ce_selected": {w: f"att{int(rng.integers(5))}" for w in views}, **sel_keys}


def test_composed_source_bank_rule_and_envelope(monkeypatch):
    """composed_source_bank on synthetic records against my transcription: per (family, view, criterion) the winner is
    the FIRST bank (own, then policies in the given order) beating the running best by > 1e-12 at the seed-0
    selection value; the reported value is that bank's seed 0-2 mean; decisions compose with the class-only code only;
    freeze = every winning policy. Envelope: the composed SELECTION value is >= every bank's (so, at seed 0, the source
    reader dominates every code reader of the bank), but the REPORTED seed-mean value can fall below the source's own
    reported mean (the envelope does not hold for the reported statistic)."""
    AU = _mod("cbp.audit")
    views = tuple(AU.PRIMARY_VIEWS)
    sel = {"sel_row_id_sha256": "x" * 64, "fit_row_id_sha256": "y" * 64, "slate": "final", "attacker_seeds": [0, 1, 2]}
    rng = np.random.default_rng(0)
    cids = ["U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1"] + [
        f"U|{f}|i8o64|l{lam:g}" for lam in LAM_GRID for f in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")]
    below_seen = 0
    for trial in range(30):
        own = {fam: _bank_rec(rng, views, 0.70, sel) for fam in ("interface", "probs", "decisions")}
        store = {}
        for c in cids:
            r = _bank_rec(rng, views, 0.69, sel)
            if rng.random() < 0.2:                                  # exact seed-0 tie with the source
                r["auc_seed0"] = dict(own["interface"]["auc_seed0"])
            store[AU.unit_of(0, c)] = {"recovery": r, "seed": 0}
        monkeypatch.setattr(AU, "_closure", lambda k, t, e, units_dir=None: {"ok": True})
        monkeypatch.setattr(AU, "load_inner", lambda name, units_dir=None: (store[name], {}))
        out = AU.composed_source_bank(0, "U", own, None, policy_cids=cids)
        freeze = set()
        for fam, src in own.items():
            banks = [("source", src)] + [(c, store[AU.unit_of(0, c)]["recovery"]) for c in cids
                                         if fam != "decisions" or c == "U|CLASS|i1o1"]
            for w in views:
                ia = ic = 0
                for j, (_, b) in enumerate(banks):
                    if b["auc_seed0"][w] > banks[ia][1]["auc_seed0"][w] + 1e-12:
                        ia = j
                    if b["ce_seed0"][w] < banks[ic][1]["ce_seed0"][w] - 1e-12:
                        ic = j
                rec = out[fam]
                assert rec["winner"][w] == banks[ia][0] and rec["ce_winner"][w] == banks[ic][0]
                assert rec["auc"][w] == banks[ia][1]["auc"][w] and rec["ce"][w] == banks[ic][1]["ce"][w]
                assert rec["auc_seed0"][w] >= max(b["auc_seed0"][w] for _, b in banks) - 1e-12
                assert rec["ce_seed0"][w] <= min(b["ce_seed0"][w] for _, b in banks) + 1e-12
                below_seen += rec["auc"][w] < src["auc"][w] - 1e-12
                freeze |= {banks[ia][0], banks[ic][0]} - {"source"}
        assert set(out["_all"]["freeze"]) == freeze
        assert out["decisions"]["policies"] == ["U|CLASS|i1o1"]
    assert below_seen > 0


def test_audit_mi_diagnostic_formula():
    """cbp.audit._mi equals the dict plug-in MI; the permutation list is a fixed function of its seed."""
    AU = _mod("cbp.audit")
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(5, 400))
        t = rng.integers(0, int(rng.integers(1, 40)), n)
        s = rng.integers(0, 2, n)
        assert abs(AU._mi(t, s) - ref_mi(s, t)) < 1e-12
    P1, h1 = AU.permutation_list(50, perms=4, seed=AU.MI_PERM_SEED)
    P2, h2 = AU.permutation_list(50, perms=4, seed=AU.MI_PERM_SEED)
    assert h1 == h2 and np.array_equal(P1, P2)
