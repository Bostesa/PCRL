"""Math/design reviewer tests for the decision-preserving joint compression study (role 3; owner of this file only).

Independent brute-force recomputation of the spec's formulas (sections 6-10, 12-14) and adversarial fixtures. The
REFERENCE section below imports nothing from dpc; the integration sections import the method/audit/selection modules
only to call their public entry points and then re-derive every checked quantity from rows with the reference code.

    cd <worktree> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest dpc/tests/test_math_review.py -q
    cd <worktree> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.tests.test_math_review --exhaustive

The exhaustive runner enumerates every allowed class-preserving coarse map on small finite fixtures, reports the
objective brackets (min/max) and whether each family's greedy/refined solution is globally best ON THAT FIXTURE.
A gap is retained and labelled; it is never promoted to an Adult solver certificate.
No real Adult data is read by this file.
"""
from __future__ import annotations

import importlib
import inspect
import itertools
import json
import math
import sys

import numpy as np
import pytest

EPS = 1e-12
TOL = 1e-12


# ==================================================================================================================
# REFERENCE (no dpc imports)
# ==================================================================================================================

def ref_argmax(p):
    """Pinned tie rule assumed here: numpy first-index argmax (checked against the method separately)."""
    return np.argmax(np.asarray(p, float), axis=-1)


def ref_smooth(mean_p, d):
    """(mean_p + eps*1 + eps*e_d) / (1 + (K+1) eps), eps = 1e-12 (spec section 7)."""
    mean_p = np.asarray(mean_p, np.float64)
    K = mean_p.shape[-1]
    e = np.zeros(K)
    e[int(d)] = 1.0
    return (mean_p + EPS + EPS * e) / (1.0 + (K + 1) * EPS)


def ref_kl_rows(P, Q):
    """Row-wise KL(p || q) in nats with 0 log 0 = 0 (q > 0 wherever p > 0 is required; inf otherwise)."""
    P = np.atleast_2d(np.asarray(P, np.float64))
    Q = np.atleast_2d(np.asarray(Q, np.float64))
    Q = np.broadcast_to(Q, P.shape)
    out = np.zeros(P.shape[0])
    for r in range(P.shape[0]):
        acc = 0.0
        for k in range(P.shape[1]):
            if P[r, k] > 0:
                acc += P[r, k] * (math.log(P[r, k]) - (math.log(Q[r, k]) if Q[r, k] > 0 else -math.inf))
        out[r] = acc
    return out


def ref_mi(s, *codes):
    """Plug-in I(S; codes...) in nats from exact counts (empty cells contribute zero; no smoothing)."""
    s = np.asarray(s)
    N = len(s)
    if not codes:
        return 0.0
    keys = [tuple(int(np.asarray(c)[r]) for c in codes) for r in range(N)]
    joint, cnt_c, cnt_s = {}, {}, {}
    for k, sv in zip(keys, s.tolist()):
        joint[(k, sv)] = joint.get((k, sv), 0) + 1
        cnt_c[k] = cnt_c.get(k, 0) + 1
        cnt_s[sv] = cnt_s.get(sv, 0) + 1
    tot = 0.0
    for (k, sv), n in joint.items():
        tot += n / N * math.log(n * N / (cnt_c[k] * cnt_s[sv]))
    return max(tot, 0.0) if abs(tot) < 1e-15 else tot


def ref_cmi(s, cond, *codes):
    """Plug-in I(S; codes | cond) = sum_m P(m) I(S; codes | cond = m)."""
    s, cond = np.asarray(s), np.asarray(cond)
    N, tot = len(s), 0.0
    for m in np.unique(cond):
        w = cond == m
        tot += w.sum() / N * ref_mi(s[w], *[np.asarray(c)[w] for c in codes])
    return tot


def set_partitions(n, kmin, kmax):
    """All set partitions of range(n) as restricted-growth label tuples with kmin <= #blocks <= kmax."""
    out = []

    def rec(prefix, nb):
        if len(prefix) == n:
            if kmin <= nb <= kmax:
                out.append(tuple(prefix))
            return
        for b in range(min(nb + 1, kmax)):
            rec(prefix + [b], max(nb, b + 1))
    if n == 0:
        return [()]
    rec([], 0)
    return out


class Recip:
    """One recipient's finite fixture: fine cells = distinct probability vectors, rows point at fine cells."""

    def __init__(self, V, f):
        self.V = np.asarray(V, np.float64)                  # n_fine x K
        self.K = self.V.shape[1]
        self.cls = ref_argmax(self.V)                       # predicted class of each fine cell
        self.f = np.asarray(f, int)                         # row -> fine cell
        self.P = self.V[self.f]                             # row probability vectors
        self.d = self.cls[self.f]                           # row teacher decisions
        self.classes = sorted(set(self.cls.tolist()))
        self.cells_of = {c: [j for j in range(len(self.V)) if self.cls[j] == c] for c in self.classes}

    def tokens(self, a):
        """a: coarse id per fine cell (globally unique per (class, block)) -> per-row token."""
        return np.asarray(a, int)[self.f]

    def prototypes(self, a):
        """Coarse prototype per coarse id: smoothed weighted mean of member ROWS' vectors; class = members' class."""
        a = np.asarray(a, int)
        tok = self.tokens(a)
        proto = {}
        for c in np.unique(tok):
            rows = tok == c
            dset = set(self.d[rows].tolist())
            assert len(dset) == 1, "coarse cell mixes predicted classes"
            proto[int(c)] = ref_smooth(self.P[rows].mean(0), dset.pop())
        return proto

    def D(self, a):
        tok = self.tokens(a)
        pr = self.prototypes(a)
        Q = np.stack([pr[int(t)] for t in tok])
        return float(ref_kl_rows(self.P, Q).mean())


def assemble(recip, per_class_labels):
    """per_class_labels: {class: restricted-growth tuple over recip.cells_of[class]} -> global coarse id per fine cell."""
    a = np.full(len(recip.V), -1, int)
    nxt = 0
    for c in recip.classes:
        lab = per_class_labels[c]
        for j, b in zip(recip.cells_of[c], lab):
            a[j] = nxt + b
        nxt += max(lab) + 1
    assert (a >= 0).all()
    return a


def ref_terms(r1, r2, s, a1, a2):
    """a_i = None means recipient i is unreleased (constant code): D_i is a constant (reported as 0), I_i = 0."""
    t1 = r1.tokens(a1) if a1 is not None else np.zeros(len(s), int)
    t2 = r2.tokens(a2) if a2 is not None else np.zeros(len(s), int)
    return {"D1": r1.D(a1) if a1 is not None else 0.0, "D2": r2.D(a2) if a2 is not None else 0.0,
            "I1": ref_mi(s, t1), "I2": ref_mi(s, t2), "I12": ref_mi(s, t1, t2)}


def ref_F(terms, lam, kind):
    D = terms["D1"] + terms["D2"]
    if kind == "task":
        return D
    if kind == "local":
        return D + lam * (terms["I1"] + terms["I2"]) / 2
    if kind == "joint":
        return D + lam * ((terms["I1"] + terms["I2"]) / 2 + terms["I12"])
    raise ValueError(kind)


# ---------------------------------------------------------------- reference greedy / refinement (spec section 9)

def _blocks(a, recip, c):
    return sorted(set(int(a[j]) for j in recip.cells_of[c]))


def _merge(a, x, y):
    b = np.array(a)
    b[b == max(x, y)] = min(x, y)
    return b


def ref_greedy(recips, a0, m, score, which=(0, 1)):
    """Agglomerate same-class coarse cells on the recipients in `which` until every class has <= m cells.
    score(list_of_maps) -> objective. Smallest increment; tie -> lexicographic (recipient, class, x, y)."""
    a = [np.array(x) for x in a0]
    log = []
    while True:
        cur = score(a)
        best = None
        for i in which:
            for c in recips[i].classes:
                bl = _blocks(a[i], recips[i], c)
                if len(bl) <= m:
                    continue
                for x, y in itertools.combinations(bl, 2):
                    b = list(a)
                    b[i] = _merge(a[i], x, y)
                    inc = score(b) - cur
                    key = (inc, i, c, x, y)
                    if best is None or key < best[0]:
                        best = (key, b)
        if best is None:
            return a, log
        log.append(best[0])
        a = best[1]


def ref_refine(recips, a0, score, which=(0, 1), sweeps=5, tol=TOL):
    """Move one fine cell to another existing coarse cell of its class; never empty a cell; strict decrease > tol;
    best target per fine cell; fixed order (recipient, fine index); at most `sweeps` full sweeps."""
    a = [np.array(x) for x in a0]
    for _ in range(sweeps):
        moved = False
        for i in which:
            for j in range(len(recips[i].V)):
                c = recips[i].cls[j]
                src = int(a[i][j])
                if (a[i][recips[i].cells_of[c]] == src).sum() <= 1:
                    continue
                cur = score(a)
                best = None
                for t in _blocks(a[i], recips[i], c):
                    if t == src:
                        continue
                    b = list(a)
                    b[i] = np.array(a[i])
                    b[i][j] = t
                    v = score(b)
                    if v < cur - tol and (best is None or v < best[0]):
                        best = (v, b)
                if best is not None:
                    a, moved = best[1], True
        if not moved:
            break
    return a


def fine_map(recip):
    return np.arange(len(recip.V))


def ref_families(r1, r2, s, m, lam):
    """Reference implementation of the six fine-state families (DIRECT-TASK is not on the fine-state family)."""
    R = (r1, r2)
    T = lambda a: ref_terms(r1, r2, s, a[0], a[1])
    const2 = np.zeros(len(r2.V), int)                 # recipient 2 'unreleased' (constant) in SEQ-12 stage one
    const1 = np.zeros(len(r1.V), int)
    out = {}
    sc_task = lambda a: ref_F(T(a), lam, "task")
    a, _ = ref_greedy(R, [fine_map(r1), fine_map(r2)], m, sc_task)
    out["FINE-TASK"] = ref_refine(R, a, sc_task)
    sc_loc = lambda a: ref_F(T(a), lam, "local")
    a, _ = ref_greedy(R, [fine_map(r1), fine_map(r2)], m, sc_loc)
    out["LOCAL"] = ref_refine(R, a, sc_loc)

    def sc1_unreleased(a):        # F_joint with recipient 2 constant == D1 + 1.5 lam I1 + const
        return r1.D(a[0]) + 1.5 * lam * ref_mi(s, r1.tokens(a[0]))

    def sc2_unreleased(a):
        return r2.D(a[1]) + 1.5 * lam * ref_mi(s, r2.tokens(a[1]))
    sc_joint = lambda a: ref_F(T(a), lam, "joint")
    a, _ = ref_greedy(R, [fine_map(r1), const2], m, sc1_unreleased, which=(0,))
    a = ref_refine(R, a, sc1_unreleased, which=(0,))
    a, _ = ref_greedy(R, [a[0], fine_map(r2)], m, sc_joint, which=(1,))
    out["SEQ-12"] = ref_refine(R, a, sc_joint, which=(1,))
    a, _ = ref_greedy(R, [const1, fine_map(r2)], m, sc2_unreleased, which=(1,))
    a = ref_refine(R, a, sc2_unreleased, which=(1,))
    a, _ = ref_greedy(R, [fine_map(r1), a[1]], m, sc_joint, which=(0,))
    out["SEQ-21"] = ref_refine(R, a, sc_joint, which=(0,))
    a, _ = ref_greedy(R, [fine_map(r1), fine_map(r2)], m, sc_joint)
    cands = [("greedy", ref_refine(R, a, sc_joint))]
    for w in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"):
        cands.append((w + ":init", out[w]))
        cands.append((w + ":refined", ref_refine(R, out[w], sc_joint)))
    best = min(cands, key=lambda kv: (sc_joint(kv[1]), kv[0]))
    out["JOINT"] = best[1]
    out["_joint_start"] = best[0]
    return out


# ---------------------------------------------------------------- reference fixtures

def fixture_small(seed=0, N=360):
    """Recipient 1 binary (4 + 4 fine cells), recipient 2 three-class (3 + 3 + 3 fine cells), S coupled to both
    confidence levels with a coalition (xor-like) component. Fine cells are distinct probability vectors."""
    rng = np.random.default_rng(seed)
    V1 = [[1 - q, q] for q in (0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95)]
    V2 = []
    for c in range(3):
        for h in (0.40, 0.65, 0.90):
            v = np.full(3, (1 - h) / 2)
            v[c] = h
            V2.append(v.tolist())
    S = rng.integers(0, 2, N)
    f1 = np.empty(N, int)
    f2 = np.empty(N, int)
    for r in range(N):
        cls1 = rng.integers(0, 2)
        cls2 = rng.integers(0, 3)
        a = rng.integers(0, 2)                       # recipient-1 confidence half
        b = a ^ S[r] if rng.random() < 0.8 else rng.integers(0, 2)
        lvl1 = 2 * a + rng.integers(0, 2)            # 4 levels, a = high/low half
        f1[r] = 4 * cls1 + lvl1
        lvl2 = 2 * b if rng.random() < 0.7 else 1    # 3 levels; b encoded at levels 0 / 2
        f2[r] = 3 * cls2 + lvl2
    return Recip(V1, f1), Recip(V2, f2), S


def fixture_coalition(n_rep=25, sub=True):
    """Each recipient's confidence clue alone is independent of S; the pair reveals S exactly (S = A xor B).
    sub=True: each clue level has two sub-levels (4 fine cells per recipient in class 0), so a fixed-rate m = 2 map can
    either keep the clue (task-optimal) or cut across it (coalition-safe). sub=False: 2 fine cells, m = 2 is forced."""
    if sub:
        V1 = [[0.95, 0.05], [0.85, 0.15], [0.65, 0.35], [0.55, 0.45]]     # clue A = (cell >= 2)
        V2 = [[0.93, 0.07], [0.83, 0.17], [0.63, 0.37], [0.53, 0.47]]     # clue B = (cell >= 2)
    else:
        V1 = [[0.9, 0.1], [0.6, 0.4]]
        V2 = [[0.8, 0.2], [0.55, 0.45]]
    L = len(V1) // 2
    f1, f2, S = [], [], []
    for A in (0, 1):
        for Bv in (0, 1):
            for u in range(L):
                for v in range(L):
                    f1 += [A * L + u] * n_rep
                    f2 += [Bv * L + v] * n_rep
                    S += [A ^ Bv] * n_rep
    return Recip(V1, f1), Recip(V2, f2), np.array(S)


def fixture_six(seed=0, N=300):
    """Recipient 1 binary (3 + 3 fine cells), recipient 2 six-class (2 fine cells per class), S tied to recipient-2
    confidence and to recipient-1 class. Used where the registered purpose shapes (K = 2, 6) are required."""
    rng = np.random.default_rng(seed)
    V1 = [[1 - q, q] for q in (0.1, 0.25, 0.4, 0.6, 0.75, 0.9)]
    V2 = []
    for c in range(6):
        for h in (0.45, 0.85):
            v = np.full(6, (1 - h) / 5)
            v[c] = h
            V2.append(v.tolist())
    S = rng.integers(0, 2, N)
    f1 = np.where(rng.random(N) < 0.6, 3 * S + rng.integers(0, 3, N), rng.integers(0, 6, N))
    f2 = 2 * rng.integers(0, 6, N) + np.where(rng.random(N) < 0.7, S, rng.integers(0, 2, N))
    return Recip(V1, f1), Recip(V2, f2), S


def fixture_wide(seed=20, N=300):
    """Recipient 1 binary with 7 + 7 fine cells, recipient 2 three-class with 4 + 4 + 4: at m = 3 refinement has
    several targets per move (separates best-improvement from first-improvement)."""
    rng = np.random.default_rng(seed)
    qs = np.r_[np.linspace(0.03, 0.47, 7), np.linspace(0.53, 0.97, 7)]
    V1 = [[1 - q, q] for q in qs]
    S = rng.integers(0, 2, N)
    f1 = np.where(rng.random(N) < 0.5, rng.integers(0, 14, N),
                  7 * rng.integers(0, 2, N) + np.clip(S * 4 + rng.integers(0, 3, N), 0, 6))
    V2 = []
    for c in range(3):
        for h in (0.4, 0.55, 0.7, 0.9):
            v = np.full(3, (1 - h) / 2)
            v[c] = h
            V2.append(v.tolist())
    f2 = 4 * rng.integers(0, 3, N) + np.where(rng.random(N) < 0.6, 2 * S + rng.integers(0, 2, N), rng.integers(0, 4, N))
    return Recip(V1, f1), Recip(V2, f2), S


def fixture_tie(n=40):
    """Exact (to an ulp) merge-cost tie inside one class: recipient-2 class 0 has a = [0.8, 0.1, 0.1] and two mirror
    images b = [0.6, 0.3, 0.1], c = [0.6, 0.1, 0.3] with equal counts, so merges (a, b) and (a, c) cost the same and the
    lexicographic tie rule alone decides. S is balanced and independent of every cell."""
    V1 = [[0.9, 0.1], [0.2, 0.8]]
    V2 = [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.6, 0.1, 0.3], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    f1, f2, S = [], [], []
    for j in range(5):
        for k in range(n):
            f2.append(j)
            f1.append(k % 2)
            S.append((k // 2) % 2)
    return Recip(V1, f1), Recip(V2, f2), np.array(S)


def fixture_null(seed=1, N=400):
    """Same score structure as fixture_small, S drawn from an INDEPENDENT stream (independent of both scores)."""
    rng = np.random.default_rng(10_000 + seed)
    r1, r2, _ = fixture_small(seed=seed, N=N)
    return r1, r2, rng.integers(0, 2, N)


# ==================================================================================================================
# STRUCTURAL STATEMENTS AND FORMULA CHECKS (reference only)
# ==================================================================================================================

@pytest.mark.parametrize("K", [2, 6])
def test_smoothing_rule_sum_argmax_ties_underflow(K):
    rng = np.random.default_rng(K)
    cases = []
    for _ in range(200):
        p = rng.dirichlet(np.full(K, 0.3))
        cases.append(p)
    for d in range(K):                                       # exact 0/1 (underflow) vectors
        e = np.zeros(K)
        e[d] = 1.0
        cases.append(e)
    tie = np.zeros(K)
    tie[0] = tie[K - 1] = 0.5                                # two-way tie
    cases.append(tie)
    cases.append(np.full(K, 1.0 / K))                        # K-way tie
    for p in cases:
        for d in [int(np.argmax(p))] + [j for j in range(K) if p[j] == p.max()]:
            q = ref_smooth(p, d)
            assert np.all(np.isfinite(q)) and np.all(q > 0)
            assert abs(q.sum() - 1.0) < 1e-15 * K
            assert int(np.argmax(q)) == d, (p, d, q)         # tie broken toward the cell's class by eps*e_d
    # the smoothed prototype is NOT bit-identical to the barycentre; the deviation is O(K eps)
    p = rng.dirichlet(np.ones(K))
    assert 0 < np.abs(ref_smooth(p, int(np.argmax(p))) - p).max() <= (K + 2) * EPS


def test_mean_of_same_class_vectors_keeps_class_in_float64():
    """Elementwise p[d] >= p[j] for every member implies mean[d] >= mean[j] in float64 when every column is summed in
    the same order (IEEE rounding is monotone); smoothing then makes d the strict argmax."""
    rng = np.random.default_rng(3)
    for K in (2, 6):
        for _ in range(300):
            n = int(rng.integers(1, 40))
            X = rng.dirichlet(np.full(K, 0.2), size=n)
            d = int(rng.integers(0, K))
            X[:, [0, d]] = X[:, [d, 0]]
            X = X[np.argmax(X, 1) == d] if (np.argmax(X, 1) == d).any() else np.eye(K)[[d]]
            q = ref_smooth(X.mean(0), d)
            assert int(np.argmax(q)) == d


def test_kl_zero_terms_and_merge_cost_sufficient_statistics():
    """KL with exact zeros has no 0*log(0) NaN; cell objective from (count, sum p, sum p log p) equals the row sum."""
    P = np.array([[1.0, 0.0], [0.7, 0.3], [0.9, 0.1], [0.55, 0.45]])
    q = ref_smooth(P.mean(0), 0)
    rows = ref_kl_rows(P, q)
    assert np.all(np.isfinite(rows)) and rows[0] == pytest.approx(-math.log(q[0]), abs=0)
    negent = sum(sum(x * math.log(x) for x in p if x > 0) for p in P)
    suff = negent - float(P.sum(0) @ np.log(q))
    assert suff == pytest.approx(rows.sum(), rel=1e-13, abs=1e-15)
    # unsmoothed merge cost of two cells = n_a KL(m_a||m_ab) + n_b KL(m_b||m_ab) (generalised JS; AIB form)
    A, B = P[:2], P[2:]
    ma, mb, mab = A.mean(0), B.mean(0), P.mean(0)
    def cell_sum(X, c):
        return float(sum(ref_kl_rows(X, c)))
    js = 2 * ref_kl_rows(ma, mab)[0] + 2 * ref_kl_rows(mb, mab)[0]
    assert cell_sum(P, mab) - cell_sum(A, ma) - cell_sum(B, mb) == pytest.approx(js, rel=1e-12)


def test_plugin_mi_matches_closed_form_and_ignores_empty_cells():
    s = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    c = np.array([0, 0, 0, 1, 1, 1, 1, 2])
    # closed form via entropies
    def H(*cols):
        _, n = np.unique(np.stack(cols, 1), axis=0, return_counts=True)
        p = n / n.sum()
        return float(-(p * np.log(p)).sum())
    assert ref_mi(s, c) == pytest.approx(H(s) + H(c) - H(s, c), abs=1e-15)
    # relabelling tokens (any bijection) leaves MI unchanged; adding unused alphabet symbols changes nothing
    assert ref_mi(s, (c * 7 + 3) % 11) == pytest.approx(ref_mi(s, c), abs=1e-15)


def test_chain_rule_identities_on_fitting_law():
    """I(S;C_i) = I(S;d_i) + I(S;C_i|d_i) and I(S;C_1,C_2) = I(S;d_1,d_2) + I(S;C_1,C_2|d_1,d_2), exactly, on the same
    plug-in law, because each token determines its decision."""
    r1, r2, s = fixture_small()
    a1 = assemble(r1, {0: (0, 1, 1, 0), 1: (0, 0, 1, 1)})
    a2 = assemble(r2, {0: (0, 1, 0), 1: (0, 0, 1), 2: (0, 1, 1)})
    t1, t2 = r1.tokens(a1), r2.tokens(a2)
    for t, d in ((t1, r1.d), (t2, r2.d)):
        assert ref_mi(s, t) == pytest.approx(ref_mi(s, d) + ref_cmi(s, d, t), abs=1e-14)
    dd = r1.d * 10 + r2.d
    assert ref_mi(s, t1, t2) == pytest.approx(ref_mi(s, r1.d, r2.d) + ref_cmi(s, dd, t1, t2), abs=1e-14)
    # the pair decision baseline is a floor that confidence coarsening cannot remove
    assert ref_mi(s, t1, t2) >= ref_mi(s, r1.d, r2.d) - 1e-15


def test_data_processing_conditional_on_public_state():
    """I(S;C_i|M) <= I(S;p_i|M) and the pair analogue, plug-in on a discretised p, with M a public fitted state drawn
    from a small finite set (the maps differ by M)."""
    r1, r2, s = fixture_small(seed=5, N=600)
    rng = np.random.default_rng(9)
    M = rng.integers(0, 3, len(s))
    maps1 = [assemble(r1, {0: x, 1: y}) for x, y in (((0, 0, 1, 1), (0, 1, 1, 1)), ((0, 1, 0, 1), (0, 0, 0, 1)),
                                                     ((0, 0, 0, 0), (0, 0, 1, 1)))]
    maps2 = [assemble(r2, {0: x, 1: y, 2: z}) for x, y, z in (((0, 0, 1), (0, 1, 1), (0, 0, 0)),
                                                              ((0, 1, 1), (0, 0, 0), (0, 1, 0)),
                                                              ((0, 0, 0), (0, 0, 0), (0, 0, 0)))]
    t1 = np.array([maps1[M[r]][r1.f[r]] for r in range(len(s))])
    t2 = np.array([maps2[M[r]][r2.f[r]] for r in range(len(s))])
    assert ref_cmi(s, M, t1) <= ref_cmi(s, M, r1.f) + 1e-15
    assert ref_cmi(s, M, t2) <= ref_cmi(s, M, r2.f) + 1e-15
    assert ref_cmi(s, M, t1, t2) <= ref_cmi(s, M, r1.f, r2.f) + 1e-15
    # and the bound is not a useful absolute secrecy bound: on this fixture the pair score information is positive
    assert ref_cmi(s, M, r1.f, r2.f) > 0.01


def test_stage_one_coefficient_is_one_point_five_lambda():
    """With C_2 constant, F_joint = D_1 + D_2 + lam*(I_1/2 + 0 + I_1) = D_1 + 1.5 lam I_1 + const."""
    r1, r2, s = fixture_small(seed=2)
    const2 = None
    lam = 0.7
    vals = []
    for x in set_partitions(4, 1, 2)[:6]:
        for y in set_partitions(4, 1, 2)[:6]:
            a1 = assemble(r1, {0: x, 1: y})
            T = ref_terms(r1, r2, s, a1, const2)
            assert T["I2"] == 0.0 and T["I12"] == pytest.approx(T["I1"], abs=1e-15)
            vals.append((ref_F(T, lam, "joint"), T["D1"] + 1.5 * lam * T["I1"]))
    v = np.array(vals)
    assert np.ptp(v[:, 0] - v[:, 1]) < 1e-13


def test_coalition_fixture_terms():
    """Each clue alone: I_1 = I_2 = 0 for EVERY map; the pair: I_12 = log 2 at the fine state. F_local is blind to it;
    F_joint is not. With 4 fine cells and m = 2 the joint family can cut across one clue; with 2 fine cells and m = 2
    the fixed rate forces the leak (documented as a NOTE)."""
    r1, r2, s = fixture_coalition()
    fine = (fine_map(r1), fine_map(r2))
    T = ref_terms(r1, r2, s, *fine)
    assert T["I1"] == pytest.approx(0, abs=1e-15) and T["I2"] == pytest.approx(0, abs=1e-15)
    assert T["I12"] == pytest.approx(math.log(2), abs=1e-14)
    for a1 in enumerate_maps(r1, 2, exact=False):
        assert ref_mi(s, r1.tokens(a1)) == pytest.approx(0, abs=1e-15)
    keep = assemble(r1, {0: (0, 0, 1, 1)})                       # keeps clue A
    cut = assemble(r1, {0: (0, 1, 0, 1)})                        # independent of clue A
    Tk, Tc = ref_terms(r1, r2, s, keep, fine[1]), ref_terms(r1, r2, s, cut, fine[1])
    assert Tk["I12"] == pytest.approx(math.log(2), abs=1e-14) and Tc["I12"] == pytest.approx(0, abs=1e-15)
    lam = 1.0
    assert ref_F(Tk, lam, "local") < ref_F(Tc, lam, "local")     # local/task objective keeps the clue
    assert ref_F(Tk, lam, "joint") > ref_F(Tc, lam, "joint")     # joint objective cuts it
    fam = ref_families(r1, r2, s, m=2, lam=lam)
    TT = {k: ref_terms(r1, r2, s, *v) for k, v in fam.items() if not k.startswith("_")}
    assert TT["LOCAL"]["I12"] == pytest.approx(math.log(2), abs=1e-12)
    assert TT["FINE-TASK"]["I12"] == pytest.approx(math.log(2), abs=1e-12)
    assert TT["JOINT"]["I12"] < 1e-12
    fam1 = ref_families(r1, r2, s, m=1, lam=lam)                 # class-only: no leak beyond the decisions
    assert all(ref_terms(r1, r2, s, *v)["I12"] < 1e-12 for k, v in fam1.items() if not k.startswith("_"))
    # fixed rate: with 2 fine cells and m = 2 no family can merge, so the coalition leak is forced (NOTE N-rate)
    q1, q2, sq = fixture_coalition(sub=False)
    famq = ref_families(q1, q2, sq, m=2, lam=10.0)
    assert ref_terms(q1, q2, sq, *famq["JOINT"])["I12"] == pytest.approx(math.log(2), abs=1e-12)


def test_reference_joint_dominates_its_witnesses():
    r1, r2, s = fixture_small()
    for m, lam in ((2, 1.0), (2, 10.0), (3, 0.1)):
        fam = ref_families(r1, r2, s, m, lam)
        F = {k: ref_F(ref_terms(r1, r2, s, *v), lam, "joint") for k, v in fam.items() if not k.startswith("_")}
        for w in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"):
            assert F["JOINT"] <= F[w] + 1e-15


# ==================================================================================================================
# INTEGRATION: dpc.partition / dpc.release (public entry points only; every checked quantity re-derived from rows)
# ==================================================================================================================

def _mod(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            pytest.skip(f"{name} not written yet")
        raise


def ref_kmeans_class(Pc, c, max_cells=16, rounds=20):
    """Independent re-implementation of the documented fine k-means rule (partition.py docstring / spec section 7)."""
    Pc = np.asarray(Pc, np.float64)
    U = np.unique(Pc, axis=0)
    order = sorted(range(len(U)), key=lambda i: (-U[i, c], i))
    U = U[order]
    k = min(max_cells, len(U), len(Pc))
    C = np.array([U[((2 * j + 1) * len(U)) // (2 * k)] for j in range(k)])

    def assign(C):
        Q = np.array([ref_smooth(x, c) for x in C])
        out = []
        for p in Pc:
            v = [ref_kl_rows(p, q)[0] for q in Q]
            out.append(int(np.argmin(v)))            # first index on ties
        return np.array(out)
    prev, conv = None, False
    for r in range(1, rounds + 1):
        a = assign(C)
        if prev is not None and np.array_equal(a, prev):
            conv = True
            break
        for j in range(k):
            if (a == j).any():
                C[j] = Pc[a == j].mean(0)
        prev = a
    if not conv:
        a = assign(C)
    keep = [j for j in range(k) if (a == j).any()]
    remap = {j: i for i, j in enumerate(keep)}
    return np.array([remap[x] for x in a]), C[keep]


def _probs(rng, n, K, conc=0.4, quant=None):
    P = rng.dirichlet(np.full(K, conc), size=n)
    if quant:
        P = np.round(P * quant) / quant
        P[:, -1] = 1.0 - P[:, :-1].sum(1)
        P = np.clip(P, 0, 1)
        P = P / P.sum(1, keepdims=True)
    return P


def _edge_rows(K):
    rows = [np.eye(K)[j] for j in range(K)]                          # exact 0/1 (underflow) vectors
    t = np.zeros(K)
    t[0] = t[1] = 0.5
    rows.append(t)                                                    # two-way tie -> class 0
    rows.append(np.full(K, 1.0 / K))                                  # K-way tie -> class 0
    if K > 2:
        t = np.zeros(K)
        t[K - 2] = t[K - 1] = 0.5
        rows.append(t)                                                # tie -> class K-2
    return np.array(rows)


@pytest.mark.parametrize("K,absent", [(2, ()), (6, ()), (6, (2, 5)), (2, (1,))])
def test_fine_partition_matches_documented_rule_and_row_statistics(K, absent):
    PT = _mod("dpc.partition")
    rng = np.random.default_rng(10 + K + len(absent))
    P = np.vstack([_probs(rng, 220, K, quant=40), _edge_rows(K)])
    d = np.argmax(P, 1)
    keep = ~np.isin(d, absent)
    P, d = P[keep], d[keep]
    fine = PT.fit_fine(P, d, K)
    cell = PT.assign_fine(P, d, fine)
    for c in range(K):
        idx = fine.cells_of(c)
        if c in absent:
            assert idx.size == 1 and fine.fallback[idx[0]] and fine.n[idx[0]] == 0
            continue
        rows = np.flatnonzero(d == c)
        a_ref, C_ref = ref_kmeans_class(P[rows], c)
        assert idx.size <= 16
        # same partition of the class rows (cell ids local to the class, same order)
        assert np.array_equal(cell[rows] - idx[0], a_ref), f"class {c}: assignment differs from documented rule"
        assert np.allclose(fine.mean[idx], C_ref, rtol=0, atol=1e-15) or not fine.receipt["per_class"][c]["converged"]
    # stored statistics == brute-force row statistics of the deployment assignment
    for f in range(fine.F):
        m = cell == f
        assert fine.n[f] == m.sum()
        assert np.allclose(fine.S[f], P[m].sum(0), rtol=1e-15, atol=1e-13)
        negent = sum(sum(x * math.log(x) for x in p if x > 0) for p in P[m])
        assert fine.A[f] == pytest.approx(negent, rel=1e-13, abs=1e-13)
    # each fitting row is at its nearest smoothed centroid (brute force; ties -> lowest index)
    for r in range(len(P)):
        idx = fine.cells_of(d[r])
        v = [ref_kl_rows(P[r], fine.centroid[j])[0] for j in idx]
        assert cell[r] == idx[int(np.argmin(v))]


@pytest.mark.parametrize("K", [2, 6])
def test_class_preservation_pointwise_ties_underflow_unseen_and_roundtrip(K, tmp_path):
    PT = _mod("dpc.partition")
    RL = _mod("dpc.release")
    rng = np.random.default_rng(K)
    absent = (K - 1,)
    Pf = _probs(rng, 300, K, quant=25)
    Pf = Pf[np.argmax(Pf, 1) != absent[0]]
    fine = PT.fit_fine(Pf, np.argmax(Pf, 1), K)
    # an arbitrary class-respecting coarse map: pair up consecutive fine cells of each class
    lab = np.array([(int(fine.cell_class[f]), int(np.sum(fine.cell_class[:f] == fine.cell_class[f])) // 2)
                    for f in range(fine.F)], dtype=object)
    keys = [tuple(x) for x in lab]
    cell_token = RL.canonical_tokens(fine, np.array([hash(k) for k in keys]))
    pol = RL.Policy(recipient=1, fine=fine, cell_token=cell_token, family="TEST")
    # deployment vectors: fresh draws, exact 0/1, ties, unseen predicted class, values not in the fitting set
    Pd = np.vstack([_probs(rng, 500, K, conc=0.2), _edge_rows(K), np.eye(K)[[absent[0]] * 3]])
    tok, q, dec = RL.encode(pol, Pd)
    assert np.array_equal(dec, np.argmax(Pd, 1))
    assert np.array_equal(np.argmax(q, 1), dec)
    assert np.all(q > 0) and np.allclose(q.sum(1), 1.0, rtol=0, atol=1e-12)
    # decision is a FUNCTION of the token (pointwise structural statement)
    for t in np.unique(tok):
        assert len(set(dec[tok == t].tolist())) == 1
    # a supplied decision array that is not argmax P is refused (never silently re-keyed)
    wrong = np.argmax(Pd, 1)
    wrong[0] = (wrong[0] + 1) % K
    with pytest.raises((ValueError, AssertionError)):
        RL.encode(pol, Pd, wrong)
    # unseen class -> its single fallback token, decoded argmax = class, decision = class
    u = np.argmax(Pd, 1) == absent[0]
    assert len(set(tok[u].tolist())) == 1 and pol.token_fallback[tok[u][0]]
    # prototypes == smoothed weighted mean of fitting ROWS routed to the token (brute force from rows)
    ft, _, _ = RL.encode(pol, Pf)
    for t in range(pol.T):
        m = ft == t
        if m.any():
            assert np.allclose(pol.token_proto[t], ref_smooth(Pf[m].mean(0), pol.token_class[t]), rtol=0, atol=1e-15)
    # serialization round trips (JSON and npz) are bit-exact on deployment outputs
    p1 = tmp_path / "pol.json"
    RL.save_policy(pol, p1)
    pj = RL.load_policy(p1)
    p2 = tmp_path / "pol.npz"
    RL.save_policy_npz(pol, p2)
    pn = RL.load_policy_npz(p2)
    for back in (pj, pn):
        t2, q2, d2 = RL.encode(back, Pd)
        assert np.array_equal(t2, tok) and np.array_equal(q2, q) and np.array_equal(d2, dec)
        assert back.fingerprint() == pol.fingerprint()


def test_partition_refuses_label_keyed_groupings():
    """True Y, SEX or loss bins cannot key the partition: non-argmax label arrays are refused, and rows with identical
    probability vectors but different labels always receive the same fine cell and token (non-representable)."""
    PT = _mod("dpc.partition")
    rng = np.random.default_rng(4)
    base = _probs(rng, 40, 2, quant=10)
    P = np.vstack([base, base])                                # every vector appears twice
    y_true = np.r_[np.zeros(40, int), np.ones(40, int)]         # planted true-label split of identical vectors
    sex = rng.integers(0, 2, 80)
    d = np.argmax(P, 1)
    for bad in (y_true, sex, (np.arange(80) % 2)):
        if not np.array_equal(bad, d):
            with pytest.raises(ValueError):
                PT.fit_fine(P, bad, 2)
    # a label array that differs from argmax P on ONE borderline row, inside a cell whose mean keeps the right argmax
    # (so no downstream prototype check can catch it), is still refused at the gate
    Pb = np.vstack([np.tile([0.1, 0.9], (30, 1)), np.tile([0.2, 0.8], (30, 1)), np.tile([0.8, 0.2], (30, 1)),
                    [[0.55, 0.45]]])
    db = np.argmax(Pb, 1)
    db[-1] = 1
    with pytest.raises(ValueError):
        PT.fit_fine(Pb, db, 2, max_cells=1)
    fine = PT.fit_fine(P, d, 2)
    cell = PT.assign_fine(P, d, fine)
    assert np.array_equal(cell[:40], cell[40:])               # the planted split is not representable
    # a hand-built partition that splits identical vectors by y_true is refused at validation
    if hasattr(PT, "validate_fine_against_rows"):
        cls = np.argmax(P, 1)
        grp = cls * 2 + y_true
        F = 4
        n, S, A = PT.cell_stats(P, grp, F)
        cc = np.array([0, 0, 1, 1])
        ok = n > 0
        mean = S[ok] / n[ok, None]
        planted = PT.FinePartition(K=2, cell_class=cc[ok], centroid=PT.smooth(mean, cc[ok]), mean=mean, n=n[ok],
                                   S=S[ok], A=A[ok], fallback=np.zeros(ok.sum(), bool))
        with pytest.raises(ValueError):
            PT.validate_fine_against_rows(planted, P, d)


# ==================================================================================================================
# INTEGRATION: dpc.compress (families, objectives, merge/exchange arithmetic, sequential freezing, JOINT witnesses)
# ==================================================================================================================

TASK_ONLY_FAMS = ("FINE-TASK", "DIRECT-TASK")
PRIV_FAMS = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")


def aligned_recips(r1, r2):
    """Re-index a reference fixture in the METHOD's fine-cell order (method fine partition on the fixture rows; every
    fixture fine cell is a distinct vector, so the method keeps one fine cell per vector). Returns (fine1, fine2, A1,
    A2) with A_i reference recipients whose fine index j is the method's fine cell j."""
    PT = _mod("dpc.partition")
    out = []
    for r in (r1, r2):
        fine = PT.fit_fine(r.P, r.d, r.K)
        cell = PT.assign_fine(r.P, r.d, fine)
        assert not fine.fallback.any(), "aligned fixtures must cover every class"
        V = np.array([r.P[np.flatnonzero(cell == f)[0]] for f in range(fine.F)])
        A = Recip(V, cell)
        assert np.array_equal(A.cls, fine.cell_class) and np.array_equal(A.P, r.P)
        out += [fine, A]
    return out[0], out[2], out[1], out[3]


def method_pairs(r1, r2, s, m, lam, fams=None, fine=None):
    """{family: (PolicyPair, receipts)} from the method's public entry point on a fixture."""
    CP = _mod("dpc.compress")
    PT = _mod("dpc.partition")
    if fine is None:
        fine = (PT.fit_fine(r1.P, r1.d, r1.K), PT.fit_fine(r2.P, r2.d, r2.K))
    if m == 1:
        fams = fams or ("CLASS-ONLY",)
    else:
        fams = fams or TASK_ONLY_FAMS + PRIV_FAMS
    out = {}
    for fam in fams:
        lm = None if fam in TASK_ONLY_FAMS + ("CLASS-ONLY",) else lam
        out[fam] = CP.fit_policy_pair(fam, fine[0], fine[1], r1.P, r1.d, r2.P, r2.d, s, m, lm)
    return out


def row_tokens(pair, r1, r2):
    RL = _mod("dpc.release")
    t1, q1, h1 = RL.encode(pair.p1, r1.P, r1.d)
    t2, q2, h2 = RL.encode(pair.p2, r2.P, r2.d)
    return (t1, q1, h1), (t2, q2, h2)


def to_cellmap(recip, tok):
    """Per reference fine cell, the token of its rows (asserts the token is a function of the fine cell)."""
    a = np.full(len(recip.V), -1, int)
    for j in range(len(recip.V)):
        rows = recip.f == j
        if rows.any():
            v = np.unique(tok[rows])
            assert v.size == 1, "token is not a function of the probability vector"
            a[j] = int(v[0])
    assert (a >= 0).all(), "fixture fine cell without rows"
    return a


def METHOD_SOLVER(r1, r2, s, m, lam):
    sol = {}
    for fam, (pair, _) in method_pairs(r1, r2, s, m, lam).items():
        (t1, _, _), (t2, _, _) = row_tokens(pair, r1, r2)
        sol[fam] = (to_cellmap(r1, t1), to_cellmap(r2, t2))
    return sol


def ref_terms_rows(s, t1, q1, P1, t2, q2, P2):
    """Objective terms straight from released rows: tokens, decoded probabilities, teacher vectors."""
    return {"D1": float(ref_kl_rows(P1, q1).mean()), "D2": float(ref_kl_rows(P2, q2).mean()),
            "I1": ref_mi(s, t1), "I2": ref_mi(s, t2), "I12": ref_mi(s, t1, t2)}


@pytest.mark.parametrize("seed,m,lam", [(0, 2, 1.0), (3, 2, 10.0), (5, 3, 0.1), (1, 3, 1.0)])
def test_method_objectives_match_row_recomputation(seed, m, lam):
    r1, r2, s = fixture_small(seed, N=240)
    for fam, (pair, rec) in method_pairs(r1, r2, s, m, lam).items():
        (t1, q1, h1), (t2, q2, h2) = row_tokens(pair, r1, r2)
        T = ref_terms_rows(s, t1, q1, r1.P, t2, q2, r2.P)
        for k in ("D1", "D2", "I1", "I2", "I12"):
            assert rec["final"][k] == pytest.approx(T[k], rel=1e-10, abs=1e-12), (fam, k)
        assert rec["final"]["F_task"] == pytest.approx(ref_F(T, lam, "task"), abs=1e-12)
        if rec.get("lam") is not None:
            assert rec["final"]["F_local"] == pytest.approx(ref_F(T, lam, "local"), abs=1e-12)
            assert rec["final"]["F_joint"] == pytest.approx(ref_F(T, lam, "joint"), abs=1e-12)
        # class preservation and the class cap, from rows
        assert np.array_equal(h1, r1.d) and np.array_equal(h2, r2.d)
        for t, h, cap in ((t1, h1, m), (t2, h2, m)):
            for c in np.unique(h):
                assert len(np.unique(t[h == c])) <= cap
            for tt in np.unique(t):
                assert len(np.unique(h[t == tt])) == 1


@pytest.mark.parametrize("seed,m,lam", [(0, 2, 1.0), (1, 2, 1.0), (2, 2, 1.0), (3, 2, 1.0), (8, 2, 10.0),
                                        (15, 2, 10.0), (5, 3, 0.1)])
def test_method_equals_independent_reference_algorithm(seed, m, lam):
    """With fine cells indexed identically, the method's families reproduce the reference implementation of the
    spec's greedy/refinement rules (section 9) exactly (partition equality)."""
    r1, r2, s = fixture_small(seed, N=240)
    fine1, fine2, A1, A2 = aligned_recips(r1, r2)
    ref = ref_families(A1, A2, s, m, lam)
    got = method_pairs(A1, A2, s, m, lam, fams=("FINE-TASK",) + PRIV_FAMS, fine=(fine1, fine2))
    bad = []
    for fam, (pair, rec) in got.items():
        (t1, _, _), (t2, _, _) = row_tokens(pair, A1, A2)
        mine = (to_cellmap(A1, t1), to_cellmap(A2, t2))
        for i in (0, 1):
            if canon_partition(mine[i]) != canon_partition(ref[fam][i]):
                Fm = ref_F(ref_terms(A1, A2, s, *mine), lam, "joint")
                Fr = ref_F(ref_terms(A1, A2, s, *ref[fam]), lam, "joint")
                bad.append((fam, i + 1, Fm, Fr))
    assert not bad, f"method differs from the reference algorithm: {bad}"


def test_merge_and_exchange_deltas_and_tables_against_rows():
    """Every merge / move increment (dD, dI_own, dI12) and the pair table, row totals and local marginals after an
    applied merge or exchange on EITHER recipient, against brute-force recomputation from rows."""
    CP = _mod("dpc.compress")
    r1, r2, s = fixture_small(4, N=240)
    fine1, fine2, A1, A2 = aligned_recips(r1, r2)
    Tf = CP.fine_table(A1.f, A2.f, s, fine1.F, fine2.F)
    lam = 1.3
    W = CP.W_joint(lam)
    rng = np.random.default_rng(0)
    st = CP.State(fine1, fine2, Tf, np.arange(fine1.F), np.arange(fine2.F))

    def brute(lab1, lab2):
        return ref_terms(A1, A2, s, lab1, lab2)

    def check_tables():
        l1, l2 = st.labels(1), st.labels(2)
        t1, t2 = l1[A1.f], l2[A2.f]
        T12 = np.zeros_like(st.T12)
        np.add.at(T12, (s, t1, t2), 1)
        assert np.array_equal(st.T12, T12)
        assert np.array_equal(st.Tr[1], T12.sum(2)) and np.array_equal(st.Tr[2], T12.sum(1))
        assert st.T12.sum() == len(s) and np.array_equal(st.T12.sum((1, 2)), np.bincount(s, minlength=2))
        for r, R, lab in ((1, A1, l1), (2, A2, l2)):
            for l in set(lab.tolist()):
                rows = lab[R.f] == l
                assert st.R[r].n[l] == rows.sum()
                assert np.allclose(st.R[r].S[l], R.P[rows].sum(0), rtol=0, atol=1e-12)
        T = st.terms()
        B = brute(l1, l2)
        for k in B:
            assert T[k] == pytest.approx(B[k], rel=1e-11, abs=1e-13), k

    check_tables()
    for step in range(6):
        r = 1 + step % 2
        R = st.R[r]
        cls = [c for c in range(R.K) if R.coarse_count(c) > 1]
        c = cls[int(rng.integers(len(cls)))]
        labs, ia, ib, delta, dD, dI, dI12 = st.merge_deltas(r, c, W)
        before = brute(st.labels(1), st.labels(2))
        for j in range(len(ia)):
            a, b = int(labs[ia[j]]), int(labs[ib[j]])
            l = [st.labels(1), st.labels(2)]
            l[r - 1] = _merge(l[r - 1], a, b)
            after = brute(*l)
            assert dD[j] == pytest.approx(after[f"D{r}"] - before[f"D{r}"], abs=1e-13)
            assert dI[j] == pytest.approx(after[f"I{r}"] - before[f"I{r}"], abs=1e-13)
            assert dI12[j] == pytest.approx(after["I12"] - before["I12"], abs=1e-13)
            assert delta[j] == pytest.approx(ref_F(after, lam, "joint") - ref_F(before, lam, "joint"), abs=1e-12)
        j = int(rng.integers(len(ia)))
        st.apply_merge(r, int(labs[ia[j]]), int(labs[ib[j]]))
        check_tables()
    for step in range(10):
        r = 1 + step % 2
        G = st.G(r)
        cand = [f for f in range(st.R[r].F) if st.move_deltas(r, f, W, G) is not None]
        f = cand[int(rng.integers(len(cand)))]
        B, delta, dD, dI, dI12 = st.move_deltas(r, f, W, G)
        before = brute(st.labels(1), st.labels(2))
        for j, b in enumerate(B):
            l = [st.labels(1), st.labels(2)]
            l[r - 1] = np.array(l[r - 1])
            l[r - 1][f] = b
            after = brute(*l)
            assert dD[j] == pytest.approx(after[f"D{r}"] - before[f"D{r}"], abs=1e-13)
            assert dI[j] == pytest.approx(after[f"I{r}"] - before[f"I{r}"], abs=1e-13)
            assert dI12[j] == pytest.approx(after["I12"] - before["I12"], abs=1e-13)
            assert delta[j] == pytest.approx(ref_F(after, lam, "joint") - ref_F(before, lam, "joint"), abs=1e-12)
        st.apply_move(r, f, int(B[int(rng.integers(len(B)))]), G)
        check_tables()


@pytest.mark.parametrize("seed,m,lam", [(1, 2, 1.0), (1, 2, 3.0), (2, 2, 1.0)])
def test_sequential_freezing_and_stage_one_coefficient(seed, m, lam):
    """SEQ-12's first map depends on recipient 1 and S only (any change of recipient 2's scores leaves it unchanged),
    equals the reference optimum of D_1 + 1.5 lam I_1 (greedy + refinement), and differs from the 1.0 lam solution on
    these fixtures (so the coefficient has teeth). Mirror statements for SEQ-21."""
    r1, r2, s = fixture_small(seed, N=240)
    fine1, fine2, A1, A2 = aligned_recips(r1, r2)
    R = (A1, A2)
    for order, fam in (((0, 1), "SEQ-12"), ((1, 0), "SEQ-21")):
        a, b = order
        X = R[a]

        def stage1(coef):
            f = lambda q: X.D(q[a]) + coef * lam * ref_mi(s, X.tokens(q[a]))
            st = [fine_map(A1), fine_map(A2)]
            g, _ = ref_greedy(R, st, m, f, which=(a,))
            return canon_partition(ref_refine(R, g, f, which=(a,))[a])
        pair, rec = method_pairs(A1, A2, s, m, lam, fams=(fam,), fine=(fine1, fine2))[fam]
        toks = row_tokens(pair, A1, A2)
        first = canon_partition(to_cellmap(X, toks[a][0]))
        assert first == stage1(1.5), f"{fam}: first map is not the D + 1.5 lam I stage-one solution"
        if (seed, m, lam) in ((1, 2, 1.0), (1, 2, 3.0), (2, 2, 1.0)) and fam == "SEQ-12":
            assert stage1(1.0) != stage1(1.5)
        # replace the second recipient's scores by an unrelated fixture: the first map must not move
        q1, q2, _ = fixture_small(seed + 100, N=240)
        other = q2 if b == 1 else q1
        Pb = other.P
        rr = [A1, A2]
        rr[b] = Recip(np.unique(Pb, axis=0), np.unique(Pb, axis=0, return_inverse=True)[1].ravel())
        pair2, _ = method_pairs(rr[0], rr[1], s, m, lam, fams=(fam,))[fam]
        toks2 = row_tokens(pair2, rr[0], rr[1])
        assert canon_partition(to_cellmap(X, toks2[a][0])) == first, f"{fam}: first map revised by the second"


def test_greedy_lexicographic_tie_rule():
    """Tied merges (within TIE_TOL) are resolved by the lowest (recipient, class, a, b) labels, where a coarse cell's
    label is its lowest member fine index (compress.py docstring / spec 'fixed lexicographic tie rule')."""
    r1, r2, s = fixture_tie()
    fine1, fine2, A1, A2 = aligned_recips(r1, r2)
    c0 = list(np.flatnonzero(fine2.cell_class == 0))
    assert len(c0) == 3
    a = [j for j in c0 if A2.V[j][0] == 0.8][0]
    pair, _ = method_pairs(A1, A2, s, 2, None, fams=("FINE-TASK",), fine=(fine1, fine2))["FINE-TASK"]
    (_, _, _), (t2, _, _) = row_tokens(pair, A1, A2)
    m2 = to_cellmap(A2, t2)
    D_ab = A2.D(_merge(fine_map(A2), a, [j for j in c0 if j != a][0]))
    D_ac = A2.D(_merge(fine_map(A2), a, [j for j in c0 if j != a][1]))
    assert abs(D_ab - D_ac) < 1e-12, "fixture lost its tie"
    lo = min(j for j in c0 if j != a)
    expect = tuple(sorted((min(a, lo), max(a, lo))))                # lexicographically first tied pair
    grouped = tuple(sorted(j for j in c0 if m2[j] == m2[expect[0]]))
    assert grouped == expect, f"tie broken toward {grouped}, expected {expect}"


def test_refinement_takes_the_best_move_wide_fixture():
    """On a fixture where several targets improve, the method's refined starts equal the reference best-improvement
    refinement (a first-improvement rule gives a different JOINT-GREEDY refined state here)."""
    r1, r2, s = fixture_wide(20)
    fine1, fine2, A1, A2 = aligned_recips(r1, r2)
    m, lam = 3, 10.0
    R = (A1, A2)
    sc = lambda q: ref_F(ref_terms(A1, A2, s, q[0], q[1]), lam, "joint")
    pair, rec = method_pairs(A1, A2, s, m, lam, fams=("JOINT",), fine=(fine1, fine2))["JOINT"]
    g, _ = ref_greedy(R, [fine_map(A1), fine_map(A2)], m, sc)
    ref_start = sc(ref_refine(R, g, sc))
    assert rec["starts"]["JOINT-GREEDY"]["refined_F_joint"] == pytest.approx(ref_start, abs=1e-12)
    ref = ref_families(A1, A2, s, m, lam)
    for fam in ("LOCAL", "SEQ-12", "SEQ-21"):
        pp, _ = method_pairs(A1, A2, s, m, lam, fams=(fam,), fine=(fine1, fine2))[fam]
        (t1, _, _), (t2, _, _) = row_tokens(pp, A1, A2)
        assert canon_partition(to_cellmap(A1, t1)) == canon_partition(ref[fam][0]), fam
        assert canon_partition(to_cellmap(A2, t2)) == canon_partition(ref[fam][1]), fam


JOINT_UNIQUE_WINNER_FIXTURES = [(1, 2, 1.0), (2, 2, 1.0), (3, 2, 3.0), (35, 2, 10.0), (39, 2, 10.0)]
# unique best start (method receipts, re-derived independently in the test): SEQ-12, JOINT-GREEDY, SEQ-21, FINE-TASK, LOCAL


@pytest.mark.parametrize("seed,m,lam", JOINT_UNIQUE_WINNER_FIXTURES)
def test_joint_final_not_worse_than_any_witness_or_refined_start(seed, m, lam):
    """Final F_joint <= every unchanged witness AND <= the reference joint refinement of every start (greedy-joint,
    FINE-TASK, LOCAL, SEQ-12, SEQ-21). These fixtures were chosen so that each start is the unique best on at least
    one of them: a JOINT that silently drops a start (strong initialisation) fails here."""
    r1, r2, s = fixture_small(seed, N=240)
    fine1, fine2, A1, A2 = aligned_recips(r1, r2)
    R = (A1, A2)
    sc = lambda q: ref_F(ref_terms(A1, A2, s, q[0], q[1]), lam, "joint")
    got = method_pairs(A1, A2, s, m, lam, fams=("FINE-TASK",) + PRIV_FAMS, fine=(fine1, fine2))
    sol = {}
    for fam, (pair, _) in got.items():
        (t1, _, _), (t2, _, _) = row_tokens(pair, A1, A2)
        sol[fam] = [to_cellmap(A1, t1), to_cellmap(A2, t2)]
    FJ = sc(sol["JOINT"])
    g, _ = ref_greedy(R, [fine_map(A1), fine_map(A2)], m, sc)
    starts = {"greedy": ref_refine(R, g, sc)}
    for w in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"):
        assert FJ <= sc(sol[w]) + 1e-12, f"JOINT worse than unchanged witness {w}"
        starts[w] = ref_refine(R, sol[w], sc)
    for k, v in starts.items():
        assert FJ <= sc(v) + 1e-12, f"JOINT worse than the refined {k} start ({FJ} > {sc(v)})"


def test_class_only_aliases_and_label_refusal():
    r1, r2, s = fixture_small(6, N=240)
    pair, rec = method_pairs(r1, r2, s, 1, None)["CLASS-ONLY"]
    (t1, q1, h1), (t2, q2, h2) = row_tokens(pair, r1, r2)
    assert canon_partition(t1) == canon_partition(r1.d) and canon_partition(t2) == canon_partition(r2.d)
    T = ref_terms_rows(s, t1, q1, r1.P, t2, q2, r2.P)
    assert T["I12"] == pytest.approx(ref_mi(s, r1.d, r2.d), abs=1e-15)
    # m >= every class's fine count: all fine-state families are exact aliases of the fine identity map
    big = method_pairs(r1, r2, s, 16, 1.0, fams=("FINE-TASK",) + PRIV_FAMS)
    for fam, (pp, _) in big.items():
        (u1, _, _), (u2, _, _) = row_tokens(pp, r1, r2)
        assert canon_partition(u1) == canon_partition(r1.f) and canon_partition(u2) == canon_partition(r2.f), fam
    # a true-label array in place of the teacher decision is refused by every entry point
    CP = _mod("dpc.compress")
    PT = _mod("dpc.partition")
    f1, f2 = PT.fit_fine(r1.P, r1.d, 2), PT.fit_fine(r2.P, r2.d, 3)
    y = (r1.d + (np.arange(len(s)) % 7 == 0)) % 2
    with pytest.raises(ValueError):
        CP.fit_policy_pair("JOINT", f1, f2, r1.P, y, r2.P, r2.d, s, 2, 1.0)
    with pytest.raises(ValueError):
        CP.fit_policy_pair("LOCAL", f1, f2, r1.P, r1.d, r2.P, r2.d, s[:-1], 2, 1.0)


def test_coalition_fixture_through_method():
    """Each confidence clue alone is independent of S; the pair reveals S. LOCAL and FINE-TASK keep the clue, JOINT
    cuts it at m = 2 (lam = 1), exactly as the reference."""
    r1, r2, s = fixture_coalition()
    got = method_pairs(r1, r2, s, 2, 1.0, fams=("FINE-TASK", "LOCAL", "JOINT"))
    I12 = {}
    for fam, (pair, rec) in got.items():
        (t1, _, _), (t2, _, _) = row_tokens(pair, r1, r2)
        I12[fam] = ref_mi(s, t1, t2)
        assert ref_mi(s, t1) == pytest.approx(0, abs=1e-15) and ref_mi(s, t2) == pytest.approx(0, abs=1e-15)
    assert I12["FINE-TASK"] == pytest.approx(math.log(2), abs=1e-12)
    assert I12["LOCAL"] == pytest.approx(math.log(2), abs=1e-12)
    assert I12["JOINT"] < 1e-12


def test_null_fixture_plugin_mi_is_finite_sample_optimism_only():
    """S independent of the scores: every family's plug-in I_12 is small and positive (finite-sample optimism);
    JOINT's fitted I_12 is <= LOCAL's, which is NOT evidence of a held-out benefit."""
    r1, r2, s = fixture_null()
    got = method_pairs(r1, r2, s, 2, 1.0, fams=("LOCAL", "JOINT"))
    v = {}
    for fam, (pair, rec) in got.items():
        (t1, _, _), (t2, _, _) = row_tokens(pair, r1, r2)
        v[fam] = ref_mi(s, t1, t2)
    n_cells = 4 * 6                                   # at most (2 m) x (3 m) pair cells
    assert 0 <= v["JOINT"] <= v["LOCAL"] + 1e-12
    assert v["LOCAL"] < 3 * (n_cells - 1) / (2 * len(s))  # of the order of the chi-square bias (|C|-1)/(2N)


# ==================================================================================================================
# INTEGRATION: dpc.audit finite readers, token identity, orientation (tiny slate where a fitted slate is needed)
# ==================================================================================================================

def small_D(n_fit=600, n_sel=400, seed=0, p1=0.4):
    rng = np.random.default_rng(seed)
    n = n_fit + n_sel
    return {"sex": (rng.random(n) < p1).astype(np.int64), "row_id": np.arange(n, dtype=np.int64) * 7 + 3,
            "unit": np.arange(n, dtype=np.int64), "role": np.array(["AUDIT_FIT"] * n_fit + ["INNER_SELECTION"] * n_sel),
            "sealed": True, "idx": {"AUDIT_FIT": np.arange(n_fit), "INNER_SELECTION": np.arange(n_fit, n)}}


def _auc(y, p):
    """Reference Mann-Whitney AUC with midranks (fixed orientation: higher score -> S = 1)."""
    y = np.asarray(y) == 1
    p = np.asarray(p, float)
    pos, neg = p[y], p[~y]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def tiny_slate(monkeypatch, AU):
    from sklearn.linear_model import LogisticRegression
    sl = lambda: [("LR_C1", lambda seed: LogisticRegression(C=1.0, max_iter=500))]   # noqa: E731
    monkeypatch.setattr(AU, "SLATES", {"final": sl, "inner": sl})


def test_cell_reader_formula_unseen_prior_and_renumbering():
    AU = _mod("dpc.audit")
    D = small_D()
    rng = np.random.default_rng(1)
    y = D["sex"]
    f, sl = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    tok = rng.integers(0, 12, len(y))
    tok[sl[:30]] = 40                                   # a token unseen in AUDIT_FIT
    out, cov = AU.cc_local(tok, y, f, sl)
    pi = y[f].mean()
    for a in (0.5, 1.0, 5.0):
        ref = np.array([((y[f][tok[f] == t] == 1).sum() + a * pi) / ((tok[f] == t).sum() + a)
                        if (tok[f] == t).any() else pi for t in tok[sl]])
        assert np.allclose(out[f"CC_alpha{a}"], ref, rtol=0, atol=1e-15)
    assert np.all(out["CC_alpha1.0"][:30] == pi)
    perm = rng.permutation(64)
    out2, _ = AU.cc_local(perm[tok], y, f, sl)
    for k in out:
        assert np.array_equal(out[k], out2[k])


def test_pair_reader_exact_tuple_fallback_rule_and_no_key_collisions():
    AU = _mod("dpc.audit")
    D = small_D(seed=2)
    rng = np.random.default_rng(2)
    y = D["sex"]
    f, sl = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    t1 = rng.integers(0, 5, len(y))
    t2 = rng.integers(0, 9, len(y))
    t1[sl[:40]], t2[sl[:40]] = 7, 13                    # unseen local tokens and unseen tuples
    t2[sl[40:80]] = 8 - t2[sl[40:80]] % 3               # tuples that may be unseen while both locals are seen
    pred = np.r_[f, sl]
    sel_pos = np.arange(len(f), len(pred))
    out, cov = AU.cc_pair(t1, t2, y, f, pred, sel_pos)
    pi = y[f].mean()
    tab = {}
    for r in f:
        k = (int(t1[r]), int(t2[r]))
        n, n1 = tab.get(k, (0, 0))
        tab[k] = (n + 1, n1 + int(y[r] == 1))
    loc1, _ = AU.cc_local(t1, y, f, pred)
    loc2, _ = AU.cc_local(t2, y, f, pred)
    for a in (0.5, 1.0, 5.0):
        key = f"CCpair_alpha{a}"
        rule = cov["fallback_rule"][key]
        fb = {"local_1": loc1[f"CC_alpha{a}"], "local_2": loc2[f"CC_alpha{a}"], "prior": np.full(len(pred), pi)}
        unseen = np.array([(int(t1[r]), int(t2[r])) not in tab for r in pred])
        # the rule is the lowest inner CE over the unseen inner rows (ties -> local_1, local_2, prior)
        us = unseen[sel_pos]
        ys = y[pred][sel_pos]
        def ce(p):
            pt = np.where(ys[us] == 1, p[sel_pos][us], 1 - p[sel_pos][us])
            return -np.mean(np.log(np.clip(pt, 1e-12, 1)))
        ces = [ce(fb[k]) for k in ("local_1", "local_2", "prior")]
        assert rule == ("local_1", "local_2", "prior")[int(np.argmin(ces))]
        for j, r in enumerate(pred):
            k = (int(t1[r]), int(t2[r]))
            want = (tab[k][1] + a * pi) / (tab[k][0] + a) if k in tab else fb[rule][j]
            assert out[key][j] == pytest.approx(want, abs=1e-15)
    # key construction never collides distinct tuples, whatever the alphabet sizes
    for _ in range(20):
        a1 = rng.integers(0, 50, 300)
        a2 = rng.integers(0, int(rng.integers(1, 60)), 300)
        k = AU._pair_keys(a1, a2)
        assert len(set(zip(a1.tolist(), a2.tolist()))) == len(set(k.tolist()))


def _collision_release(D, rng, eta=0.0, carry=True):
    """Recipient 1: two tokens of the SAME predicted class decoding to IDENTICAL probabilities; the token ID carries
    noisy S. Recipient 2: one token per class, independent of S."""
    n = len(D["row_id"])
    S = D["sex"]
    bit = np.where(rng.random(n) < 0.1, rng.integers(0, 2, n), S) if carry else rng.integers(0, 2, n)
    cls = rng.integers(0, 2, n)
    tok1 = 2 * cls + bit                                 # tokens {0,1} class 0, {2,3} class 1
    Q1 = np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7], [0.3, 0.7]])
    cls2 = rng.integers(0, 6, n)
    Q2 = np.full((6, 6), 0.1)
    Q2[np.arange(6), np.arange(6)] = 0.5
    return {"row_id": D["row_id"], "tok1": tok1, "q1": Q1[tok1], "hard1": cls, "alpha1": np.asarray(4),
            "tok2": cls2, "q2": Q2[cls2], "hard2": cls2, "alpha2": np.asarray(6)}


def test_decoder_collision_token_secretly_carrying_S_is_detected(monkeypatch):
    AU = _mod("dpc.audit")
    tiny_slate(monkeypatch, AU)
    D = small_D(seed=3)
    rng = np.random.default_rng(3)
    z = _collision_release(D, rng)
    V = AU.policy_views(z, D)
    sl = D["idx"]["INNER_SELECTION"]
    rec = AU.inner_audit(V, D)
    assert rec["auc"]["v1"] > 0.85 and rec["auc"]["pair"] > 0.85        # token identity is audited
    # BOTH the exact-identity cell readers and the slate (one-hot token columns) see it on their own
    cc = [r["inner_auc"] for r in rec["tables"]["v1"] if r["attacker"].startswith("CC_")]
    sl_ = [r["inner_auc"] for r in rec["tables"]["v1"] if not r["attacker"].startswith("CC_")]
    assert cc and max(cc) > 0.85, "cell-conditional readers do not key on the token identity"
    assert sl_ and max(sl_) > 0.85, "the slate's design matrix does not carry the token identity"
    # a reader that only sees decoded probabilities (the collision) cannot see it
    qtok = AU.value_tokens(z["q1"])
    out, _ = AU.cc_local(qtok, D["sex"], D["idx"]["AUDIT_FIT"], sl)
    assert abs(_auc(D["sex"][sl], out["CC_alpha1.0"]) - 0.5) < 0.08
    # both colliding IDs remain distinct one-hot columns of the slate's design matrix
    X = V["X"]["v1"]
    assert X.shape[1] == 4 + 2 + 2 and np.linalg.matrix_rank(X[:, :4]) == 4
    # renumbering the token IDs leaves the design matrix and every prediction bit-identical
    perm = np.array([3, 0, 2, 1])
    z2 = dict(z, tok1=perm[z["tok1"]])
    V2 = AU.policy_views(z2, D)
    assert np.array_equal(V2["X"]["v1"], V["X"]["v1"]) and np.array_equal(V2["X"]["pair"], V["X"]["pair"])
    rec2 = AU.inner_audit(V2, D)
    assert rec2["auc"] == rec["auc"]
    # a per-person state hidden behind one token (two decoded vectors for one ID) is refused
    z3 = dict(z)
    q = z["q1"].copy()
    q[0] = [0.9, 0.1] if z["hard1"][0] == 0 else [0.1, 0.9]
    z3["q1"] = q
    with pytest.raises(ValueError):
        AU.policy_views(z3, D)


def test_coalition_and_null_releases_through_finite_readers(monkeypatch):
    """XOR coalition: each local token independent of S (local AUC ~ 0.5), the exact tuple reveals S. Null: nothing."""
    AU = _mod("dpc.audit")
    tiny_slate(monkeypatch, AU)
    D = small_D(n_fit=1200, n_sel=800, seed=4)
    rng = np.random.default_rng(4)
    n, S = len(D["row_id"]), D["sex"]
    b1 = rng.integers(0, 2, n)
    noisyS = np.where(rng.random(n) < 0.05, rng.integers(0, 2, n), S)
    b2 = b1 ^ noisyS
    Qb = np.array([[0.8, 0.2], [0.8, 0.2]])
    Q6 = np.array([[0.5] + [0.1] * 5] * 2)
    z = {"row_id": D["row_id"], "tok1": b1, "q1": Qb[b1], "hard1": np.zeros(n, int), "alpha1": np.asarray(2),
         "tok2": b2, "q2": Q6[b2], "hard2": np.zeros(n, int), "alpha2": np.asarray(2)}
    rec = AU.inner_audit(AU.policy_views(z, D), D)
    assert rec["auc"]["v1"] < 0.58 and rec["auc"]["v2"] < 0.58
    assert rec["auc"]["pair"] > 0.9
    zn = dict(z, tok2=rng.integers(0, 2, n))
    zn["q2"] = Q6[zn["tok2"]]
    recn = AU.inner_audit(AU.policy_views(zn, D), D)
    assert max(recn["auc"].values()) < 0.6


def test_auc_orientation_is_fixed_and_never_flipped():
    AU = _mod("dpc.audit")
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 300), 0.001, 0.999)
    assert AU.auc1(y, p) == pytest.approx(_auc(y, p), abs=1e-12)
    assert AU.auc1(y, 1 - p) == pytest.approx(1 - _auc(y, p), abs=1e-12)
    # selection over an anti-informative bank reports the AUC below 0.5 (no flipping)
    preds = {"v1:A": 1 - p, "v2:A": 1 - p, "pair:A": 1 - p}
    sel, _ = AU._select(preds, [""], y, np.arange(len(y)))
    assert sel["v1"]["auc"]["inner_auc"] < 0.5 and sel["pair"]["auc"]["inner_auc"] < 0.5
    # stored-probability AUC used by dpc.infer (bootstrap-weighted Mann-Whitney) matches, with unit weights
    from stored_model_eval.bench_infer import class_auc
    P = np.stack([1 - p, p], 1)
    assert float(class_auc(y, P, 1)(np.ones((300, 1)))[0]) == pytest.approx(_auc(y, p), abs=1e-12)
    w = rng.integers(0, 3, 300).astype(float)
    from sklearn.metrics import roc_auc_score
    assert float(class_auc(y, P, 1)(w[:, None])[0]) == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)


# ==================================================================================================================
# INTEGRATION: dpc.select (gates per seed, C_match / C_global / T* / J* / P*, tie order, missing comparators)
# ==================================================================================================================

SEL_TEACHERS = ("U", "RAW-J_b0.3")
SEL_RATES = (2, 4)
SEL_LAMS = (0.1, 1.0)
U_UTIL = {0: {"acc": 0.85, "logloss": 0.34, "brier": 0.22, "const_acc": 0.76},
          1: {"acc": 0.47, "logloss": 1.27, "brier": 0.70, "const_acc": 0.30}}


def sel_ids():
    ids = []
    for t in SEL_TEACHERS:
        for m in SEL_RATES:
            ids += [f"{t}|FINE-TASK|m{m}", f"{t}|DIRECT-TASK|m{m}"]
            ids += [f"{t}|{f}|m{m}|l{lam:g}" for lam in SEL_LAMS for f in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")]
        ids.append(f"{t}|CLASS|m1")
    return ids + [f"SRC|{t}" for t in SEL_TEACHERS] + ["REF|E", "REF|F", "REF|F0"]


def random_records(rng, ids, p_ok=0.6):
    """Inner records {(seed, cid): record} with random recoveries and utilities that pass/fail gates at random."""
    recs = {}
    for cid in ids:
        base = rng.uniform(0.6, 0.9)
        ok = rng.random() < p_ok or cid == "SRC|U"
        for k in (0, 1, 2):
            a = {w: float(base + rng.normal(0, 0.01) + (0.05 if w == "pair" else 0)) for w in ("v1", "v2", "pair")}
            u = {}
            for i in (0, 1):
                U = U_UTIL[i]
                bad = (not ok) and rng.random() < 0.5
                if cid == "SRC|U":
                    u[str(i)] = dict(U)
                    continue
                u[str(i)] = {"acc": U["acc"] - rng.uniform(0, 0.009), "const_acc": U["const_acc"],
                             "logloss": U["logloss"] + (rng.uniform(0.0105, 0.03) if bad else rng.uniform(0, 0.0095)),
                             "brier": U["brier"] + rng.uniform(0, 0.0045)}
            recs[(k, cid)] = {"recovery": {"auc": a}, "utility": u, "class_preservation_ok": bool(rng.random() > 0.03),
                              "token_states": int(rng.integers(4, 60))}
    return recs


def install_selection(monkeypatch, tmp_path, recs, ids):
    SE = _mod("dpc.select")
    R = SE.R
    monkeypatch.setattr(R, "locked_bank_ids", lambda: list(ids))
    monkeypatch.setattr(R, "RATES", SEL_RATES)
    monkeypatch.setattr(R, "TEACHERS", SEL_TEACHERS)
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path)
    monkeypatch.setattr(R, "event", lambda *a, **k: None)
    names = {f"inner__{R.unit_for(k, cid)}": v for (k, cid), v in recs.items()}
    monkeypatch.setattr(R, "rec", lambda name: names[name])
    return SE


def ref_select(recs, ids):
    """Independent implementation of spec sections 12-13."""
    fam = lambda c: "SRC" if c.startswith("SRC|") else ("REF" if c.startswith("REF|") else c.split("|")[1])  # noqa
    teacher = lambda c: c.split("|")[1] if c.startswith("SRC|") else (None if c.startswith("REF|") else c.split("|")[0])  # noqa
    rate = lambda c: int(c.split("|")[2][1:]) if fam(c) not in ("SRC", "REF") else None  # noqa
    pol = [c for c in ids if fam(c) not in ("SRC", "REF")]
    Uu = {k: {int(i): v for i, v in recs[(k, "SRC|U")]["utility"].items()} for k in (0, 1, 2)}

    def auc(k, c):
        a = dict(recs[(k, c)]["recovery"]["auc"])
        if fam(c) == "SRC":
            for p in pol:
                if teacher(p) == teacher(c):
                    for w in a:
                        a[w] = max(a[w], recs[(k, p)]["recovery"]["auc"][w])
        return a

    def eligible(c):
        for k in (0, 1, 2):
            if not recs[(k, c)].get("class_preservation_ok", True):
                return False
            for i in (0, 1):
                u, U = recs[(k, c)]["utility"][str(i)], Uu[k][i]
                cst = u["const_acc"]
                if not (u["acc"] >= U["acc"] - 0.01 and u["logloss"] <= U["logloss"] + 0.01 and
                        u["brier"] <= U["brier"] + 0.005 and (u["acc"] - cst) >= 0.8 * (U["acc"] - cst) and
                        (u["acc"] - cst) >= 0.03):
                    return False
        return True

    def key(c):
        mp = np.mean([auc(k, c)["pair"] for k in (0, 1, 2)])
        ml = np.mean([(recs[(k, c)]["utility"]["0"]["logloss"] + recs[(k, c)]["utility"]["1"]["logloss"]) / 2
                      for k in (0, 1, 2)])
        ts = sum(recs[(k, c)]["token_states"] for k in (0, 1, 2)) if fam(c) not in ("SRC", "REF") else float("inf")
        return (round(mp, 12), round(ml, 12), ts, c)

    def guarded(c, g):
        return all(auc(k, c)[w] - (auc(k, g)[w] + 0.005) <= 0 for k in (0, 1, 2) for w in ("v1", "v2"))

    best = lambda cs: min(cs, key=key) if cs else None  # noqa
    E = {c: eligible(c) for c in ids}
    nonjoint = [c for c in pol if fam(c) in ("FINE-TASK", "DIRECT-TASK", "LOCAL", "SEQ-12", "SEQ-21")]
    cm = {(t, m): best([c for c in nonjoint if teacher(c) == t and rate(c) == m and E[c]]) for t in SEL_TEACHERS
          for m in SEL_RATES}
    cg = best([c for c in nonjoint + [c for c in ids if fam(c) in ("CLASS", "SRC", "REF")] if E[c]])
    ts = best([c for c in ids if (fam(c) in ("FINE-TASK", "DIRECT-TASK", "CLASS", "SRC") or c == "REF|F0") and E[c]])  # R1
    jok = [c for c in pol if fam(c) == "JOINT" and E[c] and cm[(teacher(c), rate(c))] and cg and
           guarded(c, cm[(teacher(c), rate(c))]) and guarded(c, cg)]
    J = best(jok)
    P = best([c for c in pol if fam(c) in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT") and E[c] and ts and guarded(c, ts)])
    return {"J*": J, "P*": P, "T*": ts, "C_global": cg, "C_match": cm, "eligible": E}


@pytest.mark.parametrize("seed", range(12))
def test_selection_matches_independent_reference(monkeypatch, tmp_path, seed):
    rng = np.random.default_rng(100 + seed)
    ids = sel_ids()
    recs = random_records(rng, ids, p_ok=[0.3, 0.6, 0.9][seed % 3])
    SE = install_selection(monkeypatch, tmp_path, recs, ids)
    out = SE.select_all(None)
    ref = ref_select(recs, ids)
    st = out["statuses"]
    for role in ("J*", "P*", "T*", "C_global"):
        got = st[role]["config"] if st[role]["status"] == "NOMINEE" else None
        assert got == ref[role], (role, st[role]["status"], got, ref[role])
    for (t, m), c in ref["C_match"].items():
        g = out["C_match_all"][f"{t}|m{m}"]
        assert (g["config"] if g["status"] == "NOMINEE" else None) == c
    for cid, row in out["rows"].items():
        assert row["task_eligible"] == ref["eligible"][cid], cid
    if st["J*"]["status"] == "NOMINEE":
        t, _, m, _ = st["J*"]["config"].split("|")
        assert st["C_match"]["config"] == ref["C_match"][(t, int(m[1:]))]


def test_selection_designed_edge_cases(monkeypatch, tmp_path):
    """Per-seed gates (a mean pass with one failing seed is ineligible), JOINT never in C_global, missing C_match
    blocks J* (INVALID_MISSING_COMPARATOR, not a dropped guard), and the full tie order."""
    ids = sel_ids()
    rng = np.random.default_rng(7)
    recs = random_records(rng, ids, p_ok=1.0)
    for (k, c), r in recs.items():                      # make every non-U-anchored release clearly eligible first
        if c != "SRC|U":
            for i in ("0", "1"):
                r["utility"][i].update(acc=U_UTIL[int(i)]["acc"], logloss=U_UTIL[int(i)]["logloss"] + 0.001,
                                       brier=U_UTIL[int(i)]["brier"])
            r["class_preservation_ok"] = True
    # JOINT has the lowest pair AUC of all, but must never become C_global
    for k in (0, 1, 2):
        recs[(k, "U|JOINT|m2|l1")]["recovery"]["auc"] = {"v1": 0.70, "v2": 0.70, "pair": 0.10}
    # one failing seed by 1e-6 nats makes LOCAL ineligible although its mean passes
    recs[(1, "U|LOCAL|m2|l0.1")]["utility"]["1"]["logloss"] = U_UTIL[1]["logloss"] + 0.01 + 1e-6
    for k in (0, 2):
        recs[(k, "U|LOCAL|m2|l0.1")]["utility"]["1"]["logloss"] = U_UTIL[1]["logloss"]
    SE = install_selection(monkeypatch, tmp_path, recs, ids)
    out = SE.select_all(None)
    assert out["rows"]["U|LOCAL|m2|l0.1"]["task_eligible"] is False
    assert out["statuses"]["C_global"]["config"] != "U|JOINT|m2|l1"
    # every nonjoint control of U at m = 2 becomes ineligible: the m = 2 JOINT configs lose their guard
    for c in ids:
        if c.startswith("U|") and "|m2" in c and "JOINT" not in c:
            for k in (0, 1, 2):
                recs[(k, c)]["class_preservation_ok"] = False
    for c in ids:                                       # and every other JOINT is task-ineligible
        if "JOINT" in c and not c.startswith("U|JOINT|m2"):
            for k in (0, 1, 2):
                recs[(k, c)]["utility"]["0"]["acc"] = 0.5
        elif c.startswith("U|JOINT|m2"):                # the m = 2 JOINTs pass the C_global guard easily
            for k in (0, 1, 2):
                recs[(k, c)]["recovery"]["auc"].update(v1=0.0, v2=0.0)
    out = SE.select_all(None)
    assert out["C_match_all"]["U|m2"]["status"] == "NO_FEASIBLE_CONTROL"
    assert out["statuses"]["J*"]["status"] == "INVALID_MISSING_COMPARATOR"
    assert out["statuses"]["J*"]["config"] is None
    # tie order: pair AUC, then mean log loss, then token states, then id
    rows = []
    for cid, ll, tsx in (("b", 0.5, 10), ("a", 0.5, 10), ("c", 0.4, 99), ("d", 0.5, 5)):
        rows.append({"config": cid, "mean_pair": 0.7, "mean_logloss": ll, "token_states": tsx, "task_eligible": True,
                     "gate_shortfall": 0.0, "seeds": {k: {"auc": {"v1": 0.5, "v2": 0.5, "pair": 0.7}} for k in (0, 1, 2)}})
    assert SE.pick(rows)["config"] == "c"
    assert SE.pick([r for r in rows if r["config"] != "c"])["config"] == "d"
    assert SE.pick([r for r in rows if r["config"] in ("a", "b")])["config"] == "a"
    lower = dict(rows[0], config="z", mean_pair=0.6999)
    assert SE.pick(rows + [lower])["config"] == "z"


# ==================================================================================================================
# INTEGRATION: dpc.family + dpc.infer (33 slots, strict thresholds, gain-retention algebra, shared group bootstrap)
# ==================================================================================================================

def test_family_slots_z_and_claim_rule():
    FA = _mod("dpc.family")
    from statistics import NormalDist
    assert FA.PRIMARY_SIZE == 33 and len(FA.PRIMARY) == 33 and len({e["id"] for e in FA.PRIMARY}) == 33
    assert FA.Z_PRIMARY == NormalDist().inv_cdf(1 - 0.05 / 66) and repr(FA.Z_PRIMARY) == "3.1717657833516224"
    assert (FA.B, FA.BOOT_SEED) == (1999, 20261007)
    want = [("coalition", 0.02, "lower>"), ("local", 0.01, "upper<"), ("local", 0.01, "upper<"),
            ("acc", -0.01, "lower>"), ("acc", -0.01, "lower>"), ("logloss", 0.01, "upper<"),
            ("logloss", 0.01, "upper<"), ("brier", 0.005, "upper<"), ("brier", 0.005, "upper<"),
            ("retain", 0.0, "lower>"), ("retain", 0.0, "lower>")]
    for ci, (claim, (nom, ref)) in enumerate((("A", ("J*", "C_match")), ("B", ("J*", "C_global")), ("C", ("P*", "T*")))):
        rows = [e for e in FA.PRIMARY if e["claim"] == claim]
        assert [(e["kind"], e["target"], e["side"]) for e in rows] == want
        assert all(e["nominee"] == nom for e in rows) and all(e.get("ref", ref) == ref for e in rows)
        assert [e.get("task") for e in rows[3:]] == [0, 1] * 4
    ok = {e["id"]: "PASS" for e in FA.PRIMARY}
    st = {x: {"status": "NOMINEE", "config": "c"} for x in ("J*", "C_match", "C_global", "P*", "T*")}
    assert all(FA.claim_decision(c, ok, st)["decision"] == "PASS" for c in "ABC")
    one = dict(ok, P03="NOT_ESTABLISHED")
    assert FA.claim_decision("A", one, st)["decision"] == "NOT_ESTABLISHED"
    assert FA.claim_decision("B", one, st)["decision"] == "PASS"
    for bad in ("DESCRIPTIVE_ONLY", "NO_FEASIBLE_NOMINEE", "INVALID_MISSING_COMPARATOR"):
        st2 = dict(st, **{"T*": {"status": bad, "config": "c"}})
        assert FA.claim_decision("C", ok, st2)["decision"] == "NOT_ESTABLISHED"
    assert FA.overall_label({c: {"decision": "PASS"} for c in "ABC"}, complete=False) == "INCOMPLETE_OR_INVALID"
    # R3: validity is per claim; a favourable label needs its own claim(s) PASS and valid
    P_, N_ = "PASS", "NOT_ESTABLISHED"
    lab = lambda a, b, c, va=True, vb=True, vc=True: FA.overall_label(  # noqa: E731
        {"A": {"decision": a, "valid": va}, "B": {"decision": b, "valid": vb}, "C": {"decision": c, "valid": vc}})
    assert lab(N_, N_, P_, va=False) == "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET"
    assert lab(P_, P_, N_, vc=False) == "JOINT_DEVELOPMENT_CRITERION_MET"
    assert lab(P_, P_, P_) == "JOINT_DEVELOPMENT_CRITERION_MET + PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET"
    assert lab(P_, P_, N_, vb=False) == "INCOMPLETE_OR_INVALID"          # A passes but B invalid: no joint label
    assert lab(N_, N_, P_, vc=False) == "INCOMPLETE_OR_INVALID"
    assert lab(N_, N_, N_) == "EXPERIMENTAL_NO_ADVANTAGE"
    assert lab(N_, N_, N_, vb=False) == "INCOMPLETE_OR_INVALID"
    assert lab(P_, N_, N_) == "EXPERIMENTAL_NO_ADVANTAGE"
    # gain-retention algebra: Acc_N - 0.8 Acc_U - 0.2 c > 0  <=>  (Acc_N - c) > 0.8 (Acc_U - c)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        aN, aU, c = rng.uniform(0.2, 0.95, 3)
        assert ((aN - 0.8 * aU - 0.2 * c) > 0) == ((aN - c) > 0.8 * (aU - c) + 1e-15) or \
            abs((aN - c) - 0.8 * (aU - c)) < 1e-12


def _fake_preds(rng, n, units, sex, y1, y2, const, strength, acc_shift):
    P = {}
    for v, a in (("v1", strength[0]), ("v2", strength[1]), ("pair", strength[2])):
        P[f"P_auc_{v}"] = np.stack([np.stack([1 - q, q], 1) for q in
                                    [1 / (1 + np.exp(-(a * (2 * sex - 1) + rng.normal(0, 1, n)))) for _ in range(3)]])
    h1 = np.where(rng.random(n) < 0.85 - acc_shift, y1, 1 - y1)
    h2 = np.where(rng.random(n) < 0.47 - acc_shift, y2, rng.integers(0, 6, n))
    q1 = np.clip(0.7 + 0.2 * rng.random(n), 0, 1)
    p1 = np.where(h1[:, None] == np.arange(2), q1[:, None], 1 - q1[:, None])
    p2 = np.full((n, 6), 0.0)
    q2 = 0.4 + 0.3 * rng.random(n)
    p2[:] = ((1 - q2) / 5)[:, None]
    p2[np.arange(n), h2] = q2
    return {"assess_row_id": np.arange(n) * 5 + 1, "assess_unit": units, "sex": sex, "y_income": y1, "y_occ": y2,
            "const_class": np.asarray(const), "hard1": h1, "hard2": h2, "prob1": p1, "prob2": p2, **P}


def _wauc_matrix(y, p, W):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    C = (p[pos][:, None] > p[neg][None, :]).astype(float) + 0.5 * (p[pos][:, None] == p[neg][None, :])
    num = ((C @ W[neg]) * W[pos]).sum(0)
    return num / (W[pos].sum(0) * W[neg].sum(0))


def test_infer_end_to_end_against_independent_bootstrap(monkeypatch, tmp_path):
    """Synthetic assessment predictions -> dpc.infer.main; every primary point, SE, bound and decision recomputed
    with an independent multinomial exact-record-group bootstrap (B = 1999, seed 20261007, ONE draw sequence for all
    arms and seeds), pairwise-definition AUC, per-seed paired differences averaged over seeds."""
    IN = _mod("dpc.infer")
    FA = _mod("dpc.family")
    import dpc.eval_lock as EL_mod
    import dpc.data as DA_mod
    R = IN.R
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path)
    monkeypatch.setattr(EL_mod, "prior_hash", lambda D: "h")
    monkeypatch.setattr(DA_mod, "load", lambda: None)
    rng = np.random.default_rng(11)
    n = 240
    units = np.sort(np.r_[np.arange(220), rng.integers(0, 220, 20)])          # 20 exact-record duplicates
    sex = rng.integers(0, 2, n)
    y1, y2 = rng.integers(0, 2, n), rng.integers(0, 6, n)
    const = [0, 2]
    labs = {"SRC|U": ((1.0, 1.0, 1.6), 0.0), "U|JOINT|m4|l1": ((0.6, 0.6, 0.3), 0.0),
            "U|SEQ-12|m4|l1": ((0.6, 0.65, 1.2), 0.0), "U|DIRECT-TASK|m8": ((0.7, 0.7, 1.4), 0.0),
            "U|LOCAL|m8|l0.1": ((0.5, 0.5, 0.6), 0.05), "U|FINE-TASK|m2": ((0.6, 0.6, 1.0), 0.0)}
    preds = {}
    for k in (0, 1, 2):
        for lab, (stren, shift) in labs.items():
            z = _fake_preds(rng, n, units, sex, y1, y2, const, stren, shift)
            preds[(k, lab)] = z
            d = R.U(f"outer__s{k}__{IN.safe(lab)}")
            d.mkdir(parents=True)
            np.savez(d / "preds.npz", **z)
    resolved = {"J*": "U|JOINT|m4|l1", "C_match": "U|SEQ-12|m4|l1", "C_global": "U|DIRECT-TASK|m8",
                "P*": "U|LOCAL|m8|l0.1", "T*": "U|FINE-TASK|m2"}
    statuses = {x: {"status": "NOMINEE", "config": c} for x, c in resolved.items()}
    statuses["P*"] = {"status": "NO_FEASIBLE_NOMINEE", "config": None, "descriptive_config": resolved["P*"]}
    EL = {"seeds": {str(k): {"score": {lab: {} for lab in labs}} for k in (0, 1, 2)}, "resolved": resolved,
          "statuses": statuses, "sex_prior_defense_fit_sha256": "h", "U_valid": True}
    lockp = tmp_path / "EVALUATION_LOCK.json"
    lockp.write_text(json.dumps(EL))
    out = IN.main(["--evaluation-lock", str(lockp)])
    # ---- independent recomputation
    G_u, gidx = np.unique(units, return_inverse=True)
    rngb = np.random.default_rng(20261007)
    W = np.stack([rngb.multinomial(len(G_u), np.full(len(G_u), 1 / len(G_u))) for _ in range(1999)], 1)[gidx]
    W = W.astype(float)
    ones = np.ones((n, 1))
    WW = np.hstack([ones, W])

    def rec(k, lab, view):
        P3 = preds[(k, lab)][f"P_auc_{view}"]
        return np.mean([_wauc_matrix(sex, P3[s][:, 1], WW) for s in range(3)], 0)

    def wmean(v):
        return (v @ WW) / WW.sum(0)

    def acc(k, lab, j):
        z = preds[(k, lab)]
        return wmean((z[f"hard{j + 1}"] == (y1 if j == 0 else y2)).astype(float))

    def loss(k, lab, j, kind):
        z = preds[(k, lab)]
        y = y1 if j == 0 else y2
        P = z[f"prob{j + 1}"]
        if kind == "ll":
            return wmean(-np.log(np.clip(P[np.arange(n), y], 1e-12, 1)))
        return wmean(((P - np.eye(P.shape[1])[y]) ** 2).sum(1))

    cst = {j: wmean(((y1 if j == 0 else y2) == const[j]).astype(float)) for j in (0, 1)}
    U = "SRC|U"
    z = FA.Z_PRIMARY
    got = {e["id"]: e for e in out["primary"]}
    for e in FA.PRIMARY:
        N, C = resolved[e["nominee"]], resolved.get(e.get("ref"))
        j = e.get("task")
        per = []
        for k in (0, 1, 2):
            if e["kind"] == "coalition":
                per.append(rec(k, C, "pair") - rec(k, N, "pair"))
            elif e["kind"] == "local":
                per.append(rec(k, N, e["view"]) - rec(k, C, e["view"]))
            elif e["kind"] == "acc":
                per.append(acc(k, N, j) - acc(k, U, j))
            elif e["kind"] in ("logloss", "brier"):
                kd = "ll" if e["kind"] == "logloss" else "br"
                per.append(loss(k, N, j, kd) - loss(k, U, j, kd))
            else:
                per.append(acc(k, N, j) - 0.8 * acc(k, U, j) - 0.2 * cst[j])
        v = np.mean(per, 0)
        pt, se = v[0], np.std(v[1:], ddof=1)
        lo, hi = pt - z * se, pt + z * se
        g = got[e["id"]]
        assert g["point"] == pytest.approx(pt, abs=1e-12), e["id"]
        assert g["se"] == pytest.approx(se, rel=1e-9, abs=1e-14), e["id"]
        assert g["lower"] == pytest.approx(lo, abs=1e-11) and g["upper"] == pytest.approx(hi, abs=1e-11)
        num = ("PASS" if lo > e["target"] else "NOT_ESTABLISHED") if e["side"] == "lower>" else \
              ("PASS" if hi < e["target"] else "NOT_ESTABLISHED")
        if e["claim"] == "C":
            assert g["decision"] == "DESCRIPTIVE_ONLY" and g["decision_numeric"] == num
        else:
            assert g["decision"] == num, e["id"]
    assert out["claimC"]["decision"] == "NOT_ESTABLISHED"
    assert all(out[f"claim{c}"]["valid"] for c in "ABC") and out["label"] in ("EXPERIMENTAL_NO_ADVANTAGE",
                                                                                 "JOINT_DEVELOPMENT_CRITERION_MET")
    # R2: an arm whose stored SEX differs is refused
    z = dict(preds[(1, "U|SEQ-12|m4|l1")])
    z["sex"] = 1 - z["sex"]
    np.savez(R.U(f"outer__s1__{IN.safe('U|SEQ-12|m4|l1')}") / "preds.npz", **z)
    with pytest.raises(AssertionError):
        IN.main(["--evaluation-lock", str(lockp)])
    assert out["B"] == 1999 and out["seed"] == 20261007 and out["n_groups"] == len(G_u)
    # strict thresholds: a bound exactly at the target never passes
    assert IN.decide({"side": "lower>", "target": 0.02}, 0.02, 0.5) == "NOT_ESTABLISHED"
    assert IN.decide({"side": "upper<", "target": 0.01}, -0.5, 0.01) == "NOT_ESTABLISHED"


# ==================================================================================================================
# INTEGRATION: dpc.deploy (83-column schema, protected interface only) and dpc.utility (metric formulas)
# ==================================================================================================================

def _schema(tmp_path):
    names = [f"f{j:02d}" for j in range(83)]
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(names))
    return names, p


def test_deploy_schema_and_interface_refusals(tmp_path, monkeypatch):
    DP = _mod("dpc.deploy")
    names, sp = _schema(tmp_path)
    pinned = DP.schema_names(sp)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 83))
    good = tmp_path / "good.npz"
    np.savez(good, X=X, feature_names=np.array(names))
    assert DP.load_permitted_input(good, pinned).shape == (20, 83)
    cases = {"extra_col": dict(X=np.hstack([X, X[:, :1]]), feature_names=np.array(names + ["SEX"])),
             "reordered": dict(X=X[:, ::-1], feature_names=np.array(names[::-1])),
             "swapped_two": dict(X=X, feature_names=np.array([names[1], names[0]] + names[2:])),
             "renamed": dict(X=X, feature_names=np.array(names[:-1] + ["fnlwgt"])),
             "extra_array": dict(X=X, feature_names=np.array(names), sex=np.zeros(20)),
             "missing_col": dict(X=X[:, :82], feature_names=np.array(names[:82]))}
    for nm, arrs in cases.items():
        p = tmp_path / f"{nm}.npz"
        np.savez(p, **arrs)
        with pytest.raises(DP.Refused):
            DP.load_permitted_input(p, pinned)
    for flag in ("--export-fine-ids", "--raw-scores", "--include-teacher-probs", "--logits", "--fine", "--dump",
                 "--latent", "--distance", "--cell", "--sex", "--p1", "--probabilities-unrounded", "--foo"):
        with pytest.raises(DP.Refused):
            DP.refuse_forbidden_flags(["--unit", "u", flag])
    # the written release contains ONLY tokens, decoded probabilities and decisions
    out = {"tokens_1": np.zeros(3, np.int64), "probs_1": np.full((3, 2), 0.5), "decision_1": np.zeros(3, np.int64),
           "tokens_2": np.zeros(3, np.int64), "probs_2": np.full((3, 6), 1 / 6), "decision_2": np.zeros(3, np.int64)}
    DP.write_release(tmp_path / "r.npz", out)
    with np.load(tmp_path / "r.npz") as z:
        assert set(z.files) == set(DP.ALLOWED_OUTPUT)
    for extra in ("fine_1", "p_1", "teacher_probs_1"):
        with pytest.raises(DP.Refused):
            DP.write_release(tmp_path / "bad.npz", dict(out, **{extra: np.zeros(3)}))
    # end to end with a stand-in teacher (binary + six-class): decisions preserved pointwise, outputs restricted
    r1, r2, s = fixture_six(3)
    pair, _ = method_pairs(r1, r2, s, 2, 1.0, fams=("JOINT",))["JOINT"]
    Pd1 = np.vstack([r1.P, _probs(rng, 50, 2, conc=0.3), _edge_rows(2)])
    Pd2 = np.vstack([r2.P, _probs(rng, 50, 6, conc=0.3), _edge_rows(6)[:len(_edge_rows(2))]])
    Xd = rng.normal(size=(len(Pd1), 83)).astype(np.float32)
    monkeypatch.setattr(DP, "load_teacher", lambda unit_dir, seed=None: (None, None, "model-sha"))
    monkeypatch.setattr(DP, "teacher_probs", lambda model, heads, X: [Pd1, Pd2])
    with pytest.raises(DP.Refused):
        DP.release(tmp_path, pair, Xd)                   # unbound policy refused by default
    out, info = DP.release(tmp_path, pair, Xd, allow_unbound=True)
    assert set(out) == set(DP.ALLOWED_OUTPUT)
    assert np.array_equal(out["decision_1"], Pd1.argmax(1)) and np.array_equal(out["decision_2"], Pd2.argmax(1))
    assert np.array_equal(out["probs_1"].argmax(1), out["decision_1"])
    assert np.array_equal(out["probs_2"].argmax(1), out["decision_2"])


def test_deploy_requires_the_registered_purpose_shapes(tmp_path):
    DP = _mod("dpc.deploy")
    r1, r2, s = fixture_small(2, N=240)
    pair, _ = method_pairs(r1, r2, s, 2, 1.0, fams=("FINE-TASK",))["FINE-TASK"]
    with pytest.raises(DP.Refused):                     # recipient 2 of this fixture has K = 3, not 6
        DP.release(tmp_path, pair, np.zeros((2, 83), np.float32))


def test_utility_metric_formulas_against_sklearn():
    UT = _mod("dpc.utility")
    from sklearn.metrics import log_loss, balanced_accuracy_score
    rng = np.random.default_rng(1)
    n, K = 500, 6
    P = rng.dirichlet(np.ones(K) * 0.5, n)
    P[0] = np.eye(K)[0]                                    # exact 0/1 vector: clipped log loss
    y = rng.integers(0, K, n)
    y[0] = 1
    hard = P.argmax(1)
    m = UT.metrics(P, hard, y, K, const=2)
    ll = -np.mean(np.log(np.clip(P[np.arange(n), y], 1e-12, 1)))
    assert m["logloss"] == pytest.approx(ll, rel=1e-13)
    assert m["logloss"] == pytest.approx(log_loss(y, np.clip(P, 1e-12, 1) / np.clip(P, 1e-12, 1).sum(1, keepdims=True),
                                                  labels=range(K)), rel=1e-6)
    assert m["brier"] == pytest.approx(np.mean(((P - np.eye(K)[y]) ** 2).sum(1)), rel=1e-13)
    assert m["acc"] == pytest.approx(np.mean(hard == y)) and m["const_acc"] == pytest.approx(np.mean(y == 2))
    assert m["balanced_acc"] == pytest.approx(balanced_accuracy_score(y, hard), rel=1e-12)
    # gate margins are non-strict and per task
    u = {"0": dict(acc=0.85, logloss=0.34, brier=0.22, const_acc=0.76), "1": dict(acc=0.47, logloss=1.27, brier=0.7,
                                                                                    const_acc=0.3)}
    c = {"0": dict(u["0"], acc=0.85 - 0.0099), "1": dict(u["1"], logloss=1.27 + 0.0099)}
    assert UT.gate(c, u)["pass"]
    c["1"]["logloss"] = 1.27 + 0.0101
    assert not UT.gate(c, u)["pass"]


# ==================================================================================================================
# EXHAUSTIVE FIXTURE RUNNER
# ==================================================================================================================

def enumerate_maps(recip, m, exact):
    """Every class-preserving coarse map with exactly min(m, n_c) (exact=True) or at most m blocks per class."""
    per = []
    for c in recip.classes:
        n = len(recip.cells_of[c])
        lo = min(m, n) if exact else 1
        per.append(set_partitions(n, lo, min(m, n)))
    return [assemble(recip, dict(zip(recip.classes, combo))) for combo in itertools.product(*per)]


def exhaustive(r1, r2, s, m, lam, exact=True):
    A1, A2 = enumerate_maps(r1, m, exact), enumerate_maps(r2, m, exact)
    D1 = np.array([r1.D(a) for a in A1])
    D2 = np.array([r2.D(a) for a in A2])
    T1 = [r1.tokens(a) for a in A1]
    T2 = [r2.tokens(a) for a in A2]
    I1 = np.array([ref_mi(s, t) for t in T1])
    I2 = np.array([ref_mi(s, t) for t in T2])
    I12 = np.empty((len(A1), len(A2)))
    N = len(s)
    for x, t1 in enumerate(T1):
        for y, t2 in enumerate(T2):
            I12[x, y] = _fast_mi_pair(s, t1, t2, N)
    Ft = D1[:, None] + D2[None, :]
    Fl = Ft + lam * (I1[:, None] + I2[None, :]) / 2
    Fj = Fl + lam * I12
    return {"A1": A1, "A2": A2, "D1": D1, "D2": D2, "I1": I1, "I2": I2, "I12": I12, "task": Ft, "local": Fl,
            "joint": Fj}


def _fast_mi_pair(s, t1, t2, N):
    k = (t1.astype(np.int64) * 100003 + t2) * 2 + s
    _, inv_k, n_k = np.unique(k, return_inverse=True, return_counts=True)
    c = t1.astype(np.int64) * 100003 + t2
    _, inv_c, n_c = np.unique(c, return_inverse=True, return_counts=True)
    n_s = np.bincount(s, minlength=2)
    # per joint cell: representative row
    first = np.unique(inv_k, return_index=True)[1]
    nk = n_k
    nc = n_c[inv_c[first]]
    ns = n_s[s[first]]
    return float(np.sum(nk / N * np.log(nk * N / (nc * ns))))


def canon_partition(x):
    """Partition of fine cells as first-appearance labels (independent of the label values)."""
    seen = {}
    return tuple(seen.setdefault(int(v), len(seen)) for v in np.asarray(x))


def _index_of(maps, a):
    tgt = canon_partition(a)
    for i, b in enumerate(maps):
        if canon_partition(b) == tgt:
            return i
    return None


def exhaustive_report(r1, r2, s, m, lam, solutions, label):
    """solutions: {family: (a1, a2)} on the fine-state family. Report brackets and per-family global optimality."""
    E = exhaustive(r1, r2, s, m, lam, exact=True)
    Ecap = exhaustive(r1, r2, s, m, lam, exact=False)
    rep = {"fixture": label, "m": m, "lambda": lam, "n_maps_exact": [len(E["A1"]), len(E["A2"])],
           "n_maps_cap": [len(Ecap["A1"]), len(Ecap["A2"])], "families": {}}
    for kind in ("task", "local", "joint"):
        rep[f"bracket_{kind}_exact"] = [float(E[kind].min()), float(E[kind].max())]
        rep[f"bracket_{kind}_cap"] = [float(Ecap[kind].min()), float(Ecap[kind].max())]
    # stage-one sequential objectives (single recipient)
    s1 = E["D1"] + 1.5 * lam * E["I1"]
    s2 = E["D2"] + 1.5 * lam * E["I2"]
    own = {"FINE-TASK": "task", "DIRECT-TASK": "task", "CLASS-ONLY": "task", "LOCAL": "local", "SEQ-12": "joint",
           "SEQ-21": "joint", "JOINT": "joint"}
    for fam, (a1, a2) in solutions.items():
        x, y = _index_of(E["A1"], a1), _index_of(E["A2"], a2)
        row = {"in_exact_space": x is not None and y is not None}
        if x is None or y is None:
            rep["families"][fam] = row
            continue
        kind = own.get(fam, "joint")
        val = float(E[kind][x, y])
        row.update({"objective": kind, "value": val, "global_min_exact": float(E[kind].min()),
                    "gap_exact": val - float(E[kind].min()), "global_min_cap": float(Ecap[kind].min()),
                    "gap_cap": val - float(Ecap[kind].min()), "F_joint": float(E["joint"][x, y]),
                    "F_joint_gap_exact": float(E["joint"][x, y] - E["joint"].min())})
        if fam == "SEQ-12":
            row["stage1_gap"] = float(s1[x] - s1.min())
            row["stage2_conditional_gap"] = float(E["joint"][x, y] - E["joint"][x, :].min())
        if fam == "SEQ-21":
            row["stage1_gap"] = float(s2[y] - s2.min())
            row["stage2_conditional_gap"] = float(E["joint"][x, y] - E["joint"][:, y].min())
        row["globally_best_on_fixture"] = bool(row["gap_exact"] <= 1e-12)
        rep["families"][fam] = row
    return rep


def run_exhaustive(method_solver=None, out=sys.stdout):
    """method_solver(r1, r2, s, m, lam) -> {family: (a1, a2)} for the METHOD's families (if available); the reference
    families are always reported as well."""
    reports = []
    fixtures = [("small_seed0", fixture_small(0)), ("small_seed3", fixture_small(3)), ("coalition", fixture_coalition()),
                ("null", fixture_null())]
    for label, (r1, r2, s) in fixtures:
        for m in (1, 2, 3):
            for lam in (0.1, 1.0, 10.0):
                ref = {k: tuple(v) for k, v in ref_families(r1, r2, s, m, lam).items() if not k.startswith("_")}
                rep = exhaustive_report(r1, r2, s, m, lam, ref, label + "|reference")
                reports.append(rep)
                if method_solver is not None:
                    try:
                        sol = method_solver(r1, r2, s, m, lam)
                        reports.append(exhaustive_report(r1, r2, s, m, lam, sol, label + "|method"))
                    except Exception as exc:                                  # retained, never hidden
                        reports.append({"fixture": label + "|method", "m": m, "lambda": lam, "error": repr(exc)})
    json.dump(reports, out, indent=1, default=float)
    return reports


def exhaustive_summary(reports):
    """Per (fixture, source, family): number of (m, lam) settings where the family is globally best on its own
    objective (exact-m space), the largest gap, and gaps against the at-most-m (cap) space."""
    out = {}
    for rep in reports:
        if "error" in rep:
            out.setdefault("errors", []).append(rep)
            continue
        for fam, row in rep["families"].items():
            key = f"{rep['fixture']}|{fam}"
            d = out.setdefault(key, {"settings": 0, "globally_best": 0, "max_gap_exact": 0.0, "max_gap_cap": 0.0,
                                     "max_F_joint_gap_exact": 0.0, "not_in_exact_space": 0, "gap_settings": []})
            d["settings"] += 1
            if not row.get("in_exact_space", False):
                d["not_in_exact_space"] += 1
                continue
            d["globally_best"] += int(row["globally_best_on_fixture"])
            d["max_gap_exact"] = max(d["max_gap_exact"], row["gap_exact"])
            d["max_gap_cap"] = max(d["max_gap_cap"], row["gap_cap"])
            d["max_F_joint_gap_exact"] = max(d["max_F_joint_gap_exact"], row["F_joint_gap_exact"])
            for g in ("stage1_gap", "stage2_conditional_gap"):
                if g in row:
                    d["max_" + g] = max(d.get("max_" + g, 0.0), row[g])
            if not row["globally_best_on_fixture"]:
                d["gap_settings"].append([rep["m"], rep["lambda"], round(row["gap_exact"], 8)])
    return out


if __name__ == "__main__":
    if "--exhaustive" in sys.argv:
        solver = METHOD_SOLVER if "--reference-only" not in sys.argv else None
        out = sys.stdout
        if "--out" in sys.argv:
            out = open(sys.argv[sys.argv.index("--out") + 1], "w")
        reps = run_exhaustive(solver, out=out)
        summary = exhaustive_summary(reps)
        print(json.dumps(summary, indent=1), file=sys.stderr)
