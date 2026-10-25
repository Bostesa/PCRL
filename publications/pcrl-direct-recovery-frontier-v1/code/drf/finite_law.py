"""Exact optimal-reader AUC of a SPECIFIED finite binary-attribute law (campaign `pcrl_direct_recovery_frontier_v1`, DRF1;
investigator A, construction / optimizer). Synthetic-data module: it reads no file and no label; `selftest` uses invented laws only.

Object. Each person emits ONE observable wire w: an income probability vector, an occupation probability vector, or the linked
pair of both. A finite law is a pair of nonnegative mass vectors over the DISTINCT observable wires,
    m1[w] = P(W = w, S = 1),   m0[w] = P(W = w, S = 0)
(any positive multiples work: only P1 = m1 / sum m1 and P0 = m0 / sum m0 enter). Per-person SEX weights give the law: w1 = 1[S = 1],
w0 = 1 - w1 for an empirical law; w1 = pi, w0 = 1 - pi for an estimated law. Identical emitted wires (exact float64 equality of the
whole vector; for a pair, equality of both components) are ONE observable value: `merge_wires` combines their masses BEFORE any
recovery computation, because the reader never sees an internal cell identifier. Integer ids passed instead of vectors must be ids
of distinct EMITTED wires (e.g. the inverse returned by `merge_wires`), never internal cell ids that may share an emitted vector.

Optimal-reader AUC. For a score f of the wire, AUC(f) = P(f(W1) > f(W0)) + 1/2 P(f(W1) = f(W0)), W1 ~ P1, W0 ~ P0 independent.
With Delta(a, b) = P1(a) P0(b) - P1(b) P0(a), split the double sum over (a, b) into the diagonal and the unordered pairs:
    AUC(f) = 1/2 sum_a P1(a) P0(a) + sum_{a < b} c_f(a, b),
    c_f(a, b) = P1(a) P0(b) if f(a) > f(b);   P1(b) P0(a) if f(a) < f(b);   their mean if f(a) = f(b).
Each c_f(a, b) <= max(P1(a) P0(b), P1(b) P0(a)) = mean + |Delta(a, b)| / 2, and sum over all (a, b) of P1(a) P0(b) is 1, so
    AUC(f) <= AUC* = 1/2 + 1/4 sum_{a != b} |P1(a) P0(b) - P1(b) P0(a)|                              (closed form, O(M^2))
with equality iff f orders every pair with Delta(a, b) != 0 as the likelihood ratio m1/m0 does. Pairs with Delta = 0 (equal
ratios) contribute the same whatever f does, so ties among equal ratios may be kept (1/2) or broken. The bound is attained by the
likelihood-ratio ranking (equivalently the posterior P(S = 1 | w)). Pure wires are exact: m0 = 0 has ratio +inf (above every other
wire; pure wires tie among themselves at Delta = 0), m1 = 0 has ratio 0; wires with m1 = m0 = 0 carry no mass and are dropped.
Sorted O(M log M) form: sort by ratio, group equal ratios g, AUC* = sum_g P1_g (P0_below_g + P0_g / 2).
Properties (all verified in `selftest`): AUC* in [1/2, 1]; a one-wire law has AUC* = 1/2; AUC* is invariant to permuting /
relabelling the wires, to scaling m1 and m0 by separate positive constants and to swapping the two classes (|Delta| is symmetric);
merging wires (any deterministic function of the wire) never raises it (data processing: every score of the merged wire is a score
of the original wire); hence AUC*_pair >= max(AUC*_income, AUC*_occupation), with equality when the pair is a function of one view.

Scope. These are exact values of the SPECIFIED finite law only. For fitted census models they are not a population privacy
guarantee: an empirical or estimated law is itself a fitted object (in-sample optimistic or pessimistic), and a population reader
faces the population law, other people and other side information. Lower mutual information, lower covariance between SEX and the
(Bayes) score, a lower optimal critic loss or lower linear separability are NOT substitutes for lower AUC*. Counterexample (exact,
in `selftest`): law A, m1 = (3, 2), m0 = (2, 3), has MI 0.0201 nats, Cov(S, P(S=1|W)) = Var(P(S=1|W)) = 1/100 and AUC* = 3/5;
law B, m1 = (1, 9), m0 = (0, 10), same prior 1/2, has the HIGHER MI 0.0360 nats, the higher covariance 1/76 and the lower (more
leaking) optimal log-loss and Brier critic losses, yet the LOWER AUC* = 11/20. An XOR law (vectors (0,0), (1,1) pure S = 1 and
(0,1), (1,0) pure S = 0, equal masses) has AUC exactly 1/2 for EVERY linear score of its vectors (for f = a x + b y the four
comparisons pair up to total 2) yet AUC* = 1, while law A on a line has best linear AUC 3/5 and AUC* 3/5.

CLI: python3 -m drf.finite_law selftest   (prints JSON; exit status 0 iff every check passes)
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

__all__ = ['merge_wires', 'pair_ids', 'masses_from_ids', 'lr_scores', 'auc_star', 'auc_star_people', 'auc_star_closed_form',
           'auc_of_score', 'plugin_mi', 'plugin_mi_people', 'auc_star_exact', 'auc_star_closed_form_exact',
           'auc_of_score_exact', 'plugin_mi_exact_terms', 'best_auc_by_enumeration', 'weak_orders', 'selftest', 'MAX_ENUM']

MAX_ENUM = 7          # ordered set partitions of 7 wires: Fubini number 47293
FUBINI = (1, 1, 3, 13, 75, 541, 4683, 47293)


# ------------------------------------------------------------------------------------------------ observable wires and laws
def _component(c):
    """(n, d) contiguous array of one view and its row bytes; floats canonicalized so byte equality == float64 equality."""
    a = np.asarray(c)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError('a wire component must be a (n,) or (n, d) array')
    if a.dtype.kind == 'f':
        a = np.asarray(a, dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError('emitted wires must be finite')
        a = a + 0.0          # -0.0 -> +0.0 (the only float64 value with two encodings among finite numbers)
    elif a.dtype.kind in 'iub':
        a = np.asarray(a, dtype=np.int64)
    else:
        raise TypeError('wire components must be float or integer arrays')
    a = np.ascontiguousarray(a)
    return a, a.view(np.uint8).reshape(a.shape[0], -1)


def _unique_rows(components):
    """first index and dense inverse of the distinct rows of the concatenated components (exact equality of every component)."""
    arrs, byts = zip(*[_component(c) for c in components])
    n = arrs[0].shape[0]
    if n == 0 or any(a.shape[0] != n for a in arrs):
        raise ValueError('wire components must be non-empty and have the same number of people')
    key = np.ascontiguousarray(np.hstack(byts))
    void = key.view(np.dtype((np.void, key.shape[1]))).reshape(-1)
    _, first, inv = np.unique(void, return_index=True, return_inverse=True)
    return arrs, first, np.asarray(inv).reshape(-1)


def _weights(w1, w0, n):
    w1 = np.asarray(w1, dtype=np.float64).reshape(-1)
    w0 = (1.0 - w1) if w0 is None else np.asarray(w0, dtype=np.float64).reshape(-1)
    if w1.shape[0] != n or w0.shape[0] != n:
        raise ValueError('need one SEX = 1 weight and one SEX = 0 weight per person')
    if not (np.isfinite(w1).all() and np.isfinite(w0).all()) or (w1 < 0).any() or (w0 < 0).any():
        raise ValueError('SEX weights must be finite and nonnegative')
    return w1, w0


def merge_wires(wires, w1, w0=None, return_inverse=False):
    """Combine identical emitted wires into one observable value and return (unique_wires, m1, m0[, inverse]).

    wires: (n, d) array of emitted vectors (or (n,) ids of distinct emitted wires); a TUPLE of such arrays is a linked pair (or
    tuple) of views and two people share a pair wire iff every component is identical. Equality is exact float64 equality of the
    whole vector (-0.0 == +0.0; non-finite wires are refused). w1, w0: per-person SEX = 1 / SEX = 0 masses (w0 defaults to 1 - w1).
    unique_wires keeps the input structure (array, or tuple of arrays); m1[j], m0[j] are the summed masses of the people emitting
    wire j; inverse[i] is person i's wire index."""
    is_tuple = isinstance(wires, tuple)
    comps = list(wires) if is_tuple else [wires]
    arrs, first, inv = _unique_rows(comps)
    w1, w0 = _weights(w1, w0, arrs[0].shape[0])
    K = len(first)
    m1 = np.bincount(inv, weights=w1, minlength=K)
    m0 = np.bincount(inv, weights=w0, minlength=K)
    uniq = []
    for c, a in zip(comps, arrs):
        u = a[first]
        uniq.append(u[:, 0] if np.ndim(c) == 1 else u)
    uniq = tuple(uniq) if is_tuple else uniq[0]
    return (uniq, m1, m0, inv) if return_inverse else (uniq, m1, m0)


def pair_ids(inc_ids, occ_ids):
    """Dense per-person ids of the linked pair: two people share a pair id iff both their income and occupation wire ids agree.
    Takes integer ids of DISTINCT emitted wires per view (e.g. the inverses of merge_wires); for vectors pass a tuple to merge_wires."""
    a, b = np.asarray(inc_ids), np.asarray(occ_ids)
    if a.dtype.kind not in 'iub' or b.dtype.kind not in 'iub':
        raise TypeError('pair_ids takes integer view ids; pass emitted vectors to merge_wires as a tuple instead')
    if a.reshape(-1).shape != b.reshape(-1).shape:
        raise ValueError('one income id and one occupation id per person')
    return _unique_rows([a.reshape(-1), b.reshape(-1)])[2]


def masses_from_ids(ids, w1, w0=None, n_ids=None):
    """(m1, m0) per id (dense integer ids 0..n_ids-1 of distinct emitted wires) from per-person SEX weights."""
    ids = np.asarray(ids).reshape(-1)
    if ids.dtype.kind not in 'iu' or (ids < 0).any():
        raise TypeError('masses_from_ids takes nonnegative integer ids')
    w1, w0 = _weights(w1, w0, ids.shape[0])
    K = int(ids.max()) + 1 if n_ids is None else int(n_ids)
    return np.bincount(ids, weights=w1, minlength=K), np.bincount(ids, weights=w0, minlength=K)


def _as_masses(m1, m0):
    m1 = np.asarray(m1, dtype=np.float64).reshape(-1)
    m0 = np.asarray(m0, dtype=np.float64).reshape(-1)
    if m1.shape != m0.shape:
        raise ValueError('m1 and m0 must have one entry per distinct wire')
    if not (np.isfinite(m1).all() and np.isfinite(m0).all()) or (m1 < 0).any() or (m0 < 0).any():
        raise ValueError('masses must be finite and nonnegative')
    if not (m1.sum() > 0 and m0.sum() > 0):
        raise ValueError('both SEX classes need positive total mass (the AUC is undefined otherwise)')
    keep = (m1 > 0) | (m0 > 0)
    return m1[keep], m0[keep], keep


# ------------------------------------------------------------------------------------------------ float64 path
def _sorted_auc(s, p1, p0):
    """AUC of score s (ties 1/2) under P1 = p1, P0 = p0 (normalized), by one sort and a grouped cumulative sum."""
    order = np.argsort(s, kind='stable')
    s, p1, p0 = s[order], p1[order], p0[order]
    new = np.ones(s.shape[0], dtype=bool)
    new[1:] = s[1:] != s[:-1]
    g = np.cumsum(new) - 1
    G = int(g[-1]) + 1
    p1g = np.bincount(g, weights=p1, minlength=G)
    p0g = np.bincount(g, weights=p0, minlength=G)
    below = np.concatenate(([0.0], np.cumsum(p0g)[:-1]))
    return float(math.fsum(p1g * (below + 0.5 * p0g)))


def lr_scores(m1, m0):
    """The optimal reader's score of each distinct wire: the likelihood ratio m1 / m0 (+inf where m0 = 0 < m1, 0 where m1 = 0 < m0,
    NaN where both are 0: such a wire carries no mass)."""
    m1 = np.asarray(m1, dtype=np.float64).reshape(-1)
    m0 = np.asarray(m0, dtype=np.float64).reshape(-1)
    out = np.full(m1.shape, np.nan)
    pos = m0 > 0
    out[pos] = m1[pos] / m0[pos]
    out[(~pos) & (m1 > 0)] = np.inf
    return out


def auc_star(m1, m0):
    """Exact optimal-reader AUC (float64) of the finite law (m1, m0) over distinct wires: likelihood-ratio ranking, ties 1/2,
    O(M log M). Masses must already be merged over identical emitted wires (merge_wires)."""
    a, b, _ = _as_masses(m1, m0)
    return _sorted_auc(lr_scores(a, b), a / a.sum(), b / b.sum())


def auc_star_people(wire_ids_or_vectors, w1, w0=None):
    """AUC* of the law of the emitted wire given per-person SEX weights (w1 = 1[S = 1] for the empirical law; w1 = pi, w0 = 1 - pi
    for an estimated law). Accepts (n, d) vectors, a tuple of view arrays (a linked pair) or ids of distinct emitted wires; identical
    wires are merged first."""
    _, m1, m0 = merge_wires(wire_ids_or_vectors, w1, w0)
    return auc_star(m1, m0)


def auc_star_closed_form(m1, m0, block=1024):
    """1/2 + 1/4 sum_{a != b} |P1(a) P0(b) - P1(b) P0(a)| in float64, O(M^2) in row blocks (an independent check of auc_star)."""
    a, b, _ = _as_masses(m1, m0)
    p1, p0 = a / a.sum(), b / b.sum()
    tot = 0.0
    for i in range(0, p1.shape[0], block):
        tot += float(np.abs(np.outer(p1[i:i + block], p0) - np.outer(p0[i:i + block], p1)).sum())
    return 0.5 + 0.25 * tot


def auc_of_score(score, m1, m0):
    """AUC of an arbitrary score of the distinct wires (one value per wire, aligned with m1, m0): P(s(W1) > s(W0)) + 1/2 P(tie)."""
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    a, b, keep = _as_masses(m1, m0)
    if s.shape[0] != keep.shape[0]:
        raise ValueError('one score per distinct wire')
    s = s[keep]
    if np.isnan(s).any():
        raise ValueError('scores of wires with mass must not be NaN')
    return _sorted_auc(s, a / a.sum(), b / b.sum())


def plugin_mi(m1, m0, clip=None):
    """Plug-in mutual information I(S; W) in nats, the PRC definition sum_w P(w) KL(r_w || rbar), P(w) = (m1 + m0)[w] / total,
    r_w = m1[w] / (m1 + m0)[w], rbar = sum m1 / total, 0 log 0 = 0. clip=1e-9 reproduces nrm.fa.prc.plugin_mi's clipping of r_w
    (with w0 = 1 - w1); the default is the exact plug-in value."""
    a, b, _ = _as_masses(m1, m0)
    n = a + b
    T = float(n.sum())
    r = a / n
    rb = float(a.sum()) / T
    if clip is not None:
        r = np.clip(r, clip, 1.0 - clip)

    def xlog(x, y):
        return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / y), 0.0)
    kl = xlog(r, rb) + xlog(1.0 - r, 1.0 - rb)
    return float(math.fsum(n / T * kl))


def plugin_mi_people(wire_ids_or_vectors, w1, w0=None, clip=None):
    """plugin_mi of the merged law of per-person wires and SEX weights."""
    _, m1, m0 = merge_wires(wire_ids_or_vectors, w1, w0)
    return plugin_mi(m1, m0, clip=clip)


# ------------------------------------------------------------------------------------------------ exact (Fraction) path
def _frac(v):
    if isinstance(v, Fraction):
        return v
    if isinstance(v, (int, np.integer)):
        return Fraction(int(v))
    f = float(v)
    if not math.isfinite(f):
        raise ValueError('masses must be finite')
    return Fraction(f)                 # the exact binary value of the float


def _exact_masses(m1, m0):
    a = [_frac(v) for v in np.asarray(m1, dtype=object).reshape(-1)]
    b = [_frac(v) for v in np.asarray(m0, dtype=object).reshape(-1)]
    if len(a) != len(b):
        raise ValueError('m1 and m0 must have one entry per distinct wire')
    if any(v < 0 for v in a + b):
        raise ValueError('masses must be nonnegative')
    if not (sum(a) > 0 and sum(b) > 0):
        raise ValueError('both SEX classes need positive total mass (the AUC is undefined otherwise)')
    keep = [i for i in range(len(a)) if a[i] > 0 or b[i] > 0]
    return [a[i] for i in keep], [b[i] for i in keep], keep


def _exact_sorted_auc(keys, a, b):
    groups = {}
    for k, x, y in zip(keys, a, b):
        g = groups.setdefault(k, [Fraction(0), Fraction(0)])
        g[0] += x
        g[1] += y
    below, tot = Fraction(0), Fraction(0)
    for k in sorted(groups):
        x, y = groups[k]
        tot += x * (below + y / 2)
        below += y
    return tot / (sum(a) * sum(b))


def auc_star_exact(m1, m0):
    """Exact AUC* as a Fraction (masses: ints, Fractions or floats taken at their exact binary value); likelihood-ratio ranking
    with exact rational ratios (pure wires keyed above every finite ratio, exactly tied among themselves)."""
    a, b, _ = _exact_masses(m1, m0)
    keys = [(1, Fraction(0)) if y == 0 else (0, x / y) for x, y in zip(a, b)]
    return _exact_sorted_auc(keys, a, b)


def auc_star_closed_form_exact(m1, m0):
    """Exact closed form 1/2 + 1/4 sum_{a != b} |P1(a) P0(b) - P1(b) P0(a)| as a Fraction, O(M^2)."""
    a, b, _ = _exact_masses(m1, m0)
    n = len(a)
    tot = sum(abs(a[i] * b[j] - a[j] * b[i]) for i in range(n) for j in range(n) if i != j)
    return Fraction(1, 2) + Fraction(tot) / (4 * sum(a) * sum(b))


def auc_of_score_exact(score, m1, m0):
    """Exact AUC (Fraction) of an arbitrary score with ties 1/2; scores may be ints, Fractions or floats (compared exactly)."""
    s = list(np.asarray(score, dtype=object).reshape(-1))
    a, b, keep = _exact_masses(m1, m0)
    if len(s) != len(np.asarray(m1, dtype=object).reshape(-1)):
        raise ValueError('one score per distinct wire')
    return _exact_sorted_auc([_frac(s[i]) for i in keep], a, b)


def plugin_mi_exact_terms(m1, m0):
    """Independent route to the plug-in MI: sum_{w,s} p(w,s) log(p(w,s) / (p(w) p(s))) from the joint table (float, exact inputs)."""
    a, b, _ = _exact_masses(m1, m0)
    T = sum(a) + sum(b)
    ps = {1: sum(a) / T, 0: sum(b) / T}
    tot = []
    for x, y in zip(a, b):
        pw = (x + y) / T
        for s, m in ((1, x), (0, y)):
            if m > 0:
                pws = m / T
                tot.append(float(pws) * math.log(float(pws / (pw * ps[s]))))
    return math.fsum(tot)


def weak_orders(k):
    """Every ordered set partition (weak order) of range(k) exactly once, as a rank tuple (rank 0 = lowest score). Built by adding
    element i either to one of the r existing blocks or as a new singleton block in one of the r + 1 gaps; removing the last element
    inverts the step, so each weak order is produced once (count = Fubini number)."""
    def rec(i, ranks, r):
        if i == k:
            yield tuple(ranks)
            return
        for j in range(r):
            ranks.append(j)
            yield from rec(i + 1, ranks, r)
            ranks.pop()
        for p in range(r + 1):
            yield from rec(i + 1, [x + 1 if x >= p else x for x in ranks] + [p], r + 1)
    yield from rec(0, [], 0)


def best_auc_by_enumeration(m1, m0, return_argmax=False):
    """Brute force: the maximum AUC over ALL weak orders (scores with arbitrary ties) of the k <= MAX_ENUM distinct wires with mass,
    exact (integer arithmetic on a common denominator; returned as a Fraction). return_argmax also returns every maximizing order."""
    a, b, _ = _exact_masses(m1, m0)
    k = len(a)
    if k > MAX_ENUM:
        raise ValueError(f'enumeration is limited to k <= {MAX_ENUM} distinct wires (got {k})')
    den = 1
    for v in a + b:
        den = den * v.denominator // math.gcd(den, v.denominator)
    A = [int(v * den) for v in a]
    B = [int(v * den) for v in b]
    best, arg = -1, []
    for ranks in weak_orders(k):
        r = max(ranks) + 1
        ga, gb = [0] * r, [0] * r
        for i, g in enumerate(ranks):
            ga[g] += A[i]
            gb[g] += B[i]
        below, s2 = 0, 0
        for g in range(r):
            s2 += ga[g] * (2 * below + gb[g])
            below += gb[g]
        if s2 > best:
            best, arg = s2, [ranks]
        elif s2 == best and return_argmax:
            arg.append(ranks)
    val = Fraction(best, 2 * sum(A) * sum(B))
    return (val, arg) if return_argmax else val


# ------------------------------------------------------------------------------------------------ self-test (synthetic laws only)
def _random_law(rng, k, max_mass=6, p_pure=0.25, p_tie=0.3):
    """Random small integer law on k wires with pure wires, proportional (tied) wires and unequal class totals."""
    m1, m0 = [], []
    while len(m1) < k:
        u = rng.random()
        if m1 and u < p_tie:                               # a wire with the same likelihood ratio as an earlier one
            j = int(rng.integers(len(m1)))
            c = int(rng.integers(1, 3))
            m1.append(m1[j] * c)
            m0.append(m0[j] * c)
        elif u < p_tie + p_pure:                           # a pure wire
            if rng.random() < 0.5:
                m1.append(int(rng.integers(1, max_mass + 1)))
                m0.append(0)
            else:
                m1.append(0)
                m0.append(int(rng.integers(1, max_mass + 1)))
        else:
            m1.append(int(rng.integers(0, max_mass + 1)))
            m0.append(int(rng.integers(0, max_mass + 1)))
            if m1[-1] == 0 and m0[-1] == 0:
                m1[-1] = 1
    if sum(m1) == 0:
        m1[0] += 1
    if sum(m0) == 0:
        m0[-1] += 1
    return m1, m0


def selftest():
    import time
    from sklearn.metrics import roc_auc_score
    t0 = time.process_time()
    rng = np.random.default_rng(2026102501)
    checks = []

    def check(name, ok, **detail):
        checks.append({'name': name, 'pass': bool(ok), **detail})

    def guarded(name, fn):
        try:
            fn()
        except Exception as e:                             # a crash is a failed check, never a silent pass
            check(name, False, error=f'{type(e).__name__}: {e}')

    # 1. the weak-order generator is complete and duplicate-free (Fubini numbers)
    def t_fubini():
        counts, distinct = [], True
        for k in range(MAX_ENUM + 1):
            orders = list(weak_orders(k))
            counts.append(len(orders))
            distinct &= len(set(orders)) == len(orders)
            canon = {tuple(sorted(set(o))) == tuple(range(len(set(o)))) for o in orders}
            distinct &= canon <= {True}
        check('weak_orders_fubini', tuple(counts) == FUBINI and distinct, counts=counts)
    guarded('weak_orders_fubini', t_fubini)

    # 2. random small laws (ties, pure wires, unequal class totals): closed form == sorted == enumeration, exactly
    def t_random():
        n_laws, bad, worst = 0, [], 0.0
        for trial in range(70):
            k = int(rng.integers(1, 7)) if trial < 64 else MAX_ENUM
            m1, m0 = _random_law(rng, k)
            if sum(m1) == 0 or sum(m0) == 0:
                continue
            ex = auc_star_exact(m1, m0)
            cf = auc_star_closed_form_exact(m1, m0)
            en = best_auc_by_enumeration(m1, m0)
            fl, fc = auc_star(m1, m0), auc_star_closed_form(m1, m0)
            lr = auc_of_score(lr_scores(m1, m0), m1, m0)
            post = auc_of_score(np.array(m1, float) / np.maximum(np.array(m1, float) + np.array(m0, float), 1e-300), m1, m0)
            worst = max(worst, abs(fl - float(ex)), abs(fc - float(ex)), abs(lr - float(ex)), abs(post - float(ex)))
            n_laws += 1
            if not (ex == cf == en and Fraction(1, 2) <= ex <= 1):
                bad.append({'m1': m1, 'm0': m0, 'sorted': str(ex), 'closed': str(cf), 'enum': str(en)})
        check('random_laws_closed_sorted_enumeration_exact', not bad and worst < 1e-12, laws=n_laws, mismatches=bad[:3],
              max_float_error=worst)
    guarded('random_laws_closed_sorted_enumeration_exact', t_random)

    # 3. pure wires: a fully separable law has AUC* = 1; pure wires tie among themselves; mixed laws match the enumeration
    def t_pure():
        sep = auc_star_exact([2, 0, 1], [0, 3, 0]) == 1 and auc_star([2, 0, 1], [0, 3, 0]) == 1.0
        cases = [([1, 1, 2, 0], [0, 0, 3, 2]), ([4, 0, 3], [0, 5, 3]), ([1, 2, 3], [0, 0, 1]), ([0, 2, 5], [7, 2, 5])]
        agree = all(auc_star_exact(a, b) == best_auc_by_enumeration(a, b) == auc_star_closed_form_exact(a, b) for a, b in cases)
        # one pure S = 1 wire of mass eps above an uninformative wire: AUC* = 1/2 + eps / 2 (closed form)
        eps = Fraction(1, 10)
        tiny = auc_star_exact([1, 9], [0, 10]) == Fraction(1, 2) + eps / 2
        check('pure_wires', sep and agree and tiny, separable=sep, agree_enumeration=agree, eps_law=tiny)
    guarded('pure_wires', t_pure)

    # 4. a one-wire law has AUC* = 1/2; zero-mass wires are ignored
    def t_one():
        one = auc_star_exact([5], [3]) == Fraction(1, 2) and auc_star([5.0], [3.0]) == 0.5
        ppl = auc_star_people(np.tile([[0.3, 0.7]], (9, 1)), rng.random(9)) == 0.5
        zero = auc_star_exact([1, 2, 0], [3, 1, 0]) == auc_star_exact([1, 2], [3, 1]) and \
            best_auc_by_enumeration([1, 2, 0], [3, 1, 0]) == auc_star_exact([1, 2], [3, 1])
        check('one_wire_half_and_zero_mass_ignored', one and ppl and zero, one_wire=one, people=ppl, zero_mass=zero)
    guarded('one_wire_half_and_zero_mass_ignored', t_one)

    # 5. invariance: permutation / relabelling of wires, separate scaling of m1 and m0, swapping the classes
    def t_inv():
        ok = True
        for _ in range(40):
            m1, m0 = _random_law(rng, int(rng.integers(2, 8)))
            ex = auc_star_exact(m1, m0)
            perm = rng.permutation(len(m1))
            p1, p0 = [m1[i] for i in perm], [m0[i] for i in perm]
            c1, c0 = Fraction(int(rng.integers(1, 9)), int(rng.integers(1, 9))), Fraction(int(rng.integers(1, 9)), 7)
            ok &= auc_star_exact(p1, p0) == ex
            ok &= auc_star_exact([c1 * v for v in m1], [c0 * v for v in m0]) == ex
            ok &= auc_star_exact(m0, m1) == ex                       # swapping S = 1 and S = 0
            ok &= abs(auc_star(np.array(m1) * 3.7, np.array(m0) * 0.013) - float(ex)) < 1e-12
            # relabelling the emitted vectors (any injective recoding) leaves the merged law unchanged: exactly (Fractions of the
            # merged masses) and in float64 up to summation order
            ids = np.repeat(np.arange(len(m1)), 2)
            w = np.array([[m1[i], m0[i]] for i in range(len(m1)) for _ in range(2)], float) / 2
            reps = (np.stack([ids * 0.1, 1 - ids * 0.1], 1), np.stack([np.sqrt(ids + 2.0), ids ** 2 + 0.5], 1), ids)
            exs = {auc_star_exact(*merge_wires(r, w[:, 0], w[:, 1])[1:]) for r in reps}
            fls = [auc_star_people(r, w[:, 0], w[:, 1]) for r in reps]
            ok &= exs == {ex} and max(fls) - min(fls) < 1e-14
        check('invariance_relabel_scale_swap', ok)
    guarded('invariance_relabel_scale_swap', t_inv)

    # 6. merged outputs: two internal cells emitting identical vectors are ONE observable wire (incl. -0.0 == +0.0)
    def t_merge():
        table = np.array([[0.2, 0.8], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1], [-0.0, 1.0], [0.0, 1.0]])   # cells 0 = 2, 4 = 5
        cell = np.repeat(np.arange(6), [3, 4, 5, 2, 3, 6])
        s = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], float)
        uw, m1, m0, inv = merge_wires(table[cell], s, return_inverse=True)
        n_ok = uw.shape[0] == 4 and np.array_equal(np.sort(m1 + m0), np.sort(np.array([8, 4, 2, 9], float)))
        same = len({tuple(r) for r in uw.tolist()}) == 4 and all(np.array_equal(uw[inv[i]], table[cell[i]] + 0.0)
                                                                  for i in range(len(cell)))
        _, c1, c0 = merge_wires(cell, s)                                 # the (illegal) internal-cell law
        a_obs, a_cell = auc_star_exact(m1, m0), auc_star_exact(c1, c0)
        exact_obs = auc_star_exact([c1[0] + c1[2], c1[1], c1[3], c1[4] + c1[5]], [c0[0] + c0[2], c0[1], c0[3], c0[4] + c0[5]])
        # data processing on many random coarsenings: merging never raises AUC*
        dp_ok, n_dp = True, 0
        for _ in range(150):
            a, b = _random_law(rng, int(rng.integers(2, 9)))
            g = rng.integers(0, int(rng.integers(1, len(a) + 1)), size=len(a))
            ga = [sum(a[i] for i in range(len(a)) if g[i] == j) for j in set(g.tolist())]
            gb = [sum(b[i] for i in range(len(a)) if g[i] == j) for j in set(g.tolist())]
            dp_ok &= auc_star_exact(ga, gb) <= auc_star_exact(a, b)
            n_dp += 1
        ok = n_ok and same and a_obs <= a_cell and a_obs == exact_obs and dp_ok
        check('merge_identical_wires_and_data_processing', ok, unique_wires=int(uw.shape[0]), auc_observable=str(a_obs),
              auc_internal_cells=str(a_cell), random_coarsenings=n_dp)
    guarded('merge_identical_wires_and_data_processing', t_merge)

    # 7. linked pairs: AUC*_pair >= max(inc, occ); a pair that is a function of one view equals that view
    def t_pairs():
        ok_ge, ok_fn, ok_ids, n = True, True, True, 0
        for trial in range(60):
            N = int(rng.integers(20, 200))
            I, J = int(rng.integers(1, 5)), int(rng.integers(1, 7))
            tI = rng.dirichlet(np.ones(2), size=I)
            tO = rng.dirichlet(np.ones(5), size=J)
            if trial % 7 == 0:
                tI[-1] = tI[0]                                             # identical emitted income vectors
            ci, co = rng.integers(0, I, N), rng.integers(0, J, N)
            w1 = (rng.random(N) < 0.35 + 0.3 * (ci % 2)).astype(float) if trial % 2 else rng.random(N)
            if w1.sum() == 0 or (1 - w1).sum() == 0:
                continue
            inc, occ = tI[ci], tO[co]
            a_i, a_o, a_p = auc_star_people(inc, w1), auc_star_people(occ, w1), auc_star_people((inc, occ), w1)
            ok_ge &= a_p >= max(a_i, a_o) - 1e-12
            uI, _, _, ii = merge_wires(inc, w1, return_inverse=True)
            _, _, _, io = merge_wires(occ, w1, return_inverse=True)
            ok_ids &= abs(auc_star_people(pair_ids(ii, io), w1) - a_p) < 1e-14
            g = rng.integers(0, J, size=uI.shape[0])                       # occupation vector a function of the income WIRE
            occ_f = tO[g[ii]]
            _, mf1, mf0 = merge_wires((inc, occ_f), w1)
            _, mi1, mi0 = merge_wires(inc, w1)
            ok_fn &= (auc_star_exact(mf1, mf0) == auc_star_exact(mi1, mi0)
                      and abs(auc_star_people((inc, occ_f), w1) - a_i) < 1e-14)
            n += 1
        ex_ok = True                                                      # exact version on integer weights
        for _ in range(30):
            N = int(rng.integers(10, 60))
            ci, co = rng.integers(0, 3, N), rng.integers(0, 4, N)
            s = rng.integers(0, 2, N)
            if s.sum() in (0, N):
                continue
            def law(ids):
                return ([int(((ids == u) & (s == 1)).sum()) for u in np.unique(ids)],
                        [int(((ids == u) & (s == 0)).sum()) for u in np.unique(ids)])
            pid = pair_ids(ci, co)
            ex_ok &= auc_star_exact(*law(pid)) >= max(auc_star_exact(*law(ci)), auc_star_exact(*law(co)))
        check('linked_pair_dominates_views', ok_ge and ok_fn and ok_ids and ex_ok, populations=n, pair_ge_views=ok_ge,
              pair_function_of_one_view_equal=ok_fn, pair_ids_agree=ok_ids, exact_integer=ex_ok)
    guarded('linked_pair_dominates_views', t_pairs)

    # 8. sklearn roc_auc_score on an expanded integer population scored by the posterior (and by arbitrary tied scores)
    def t_sklearn():
        worst, worst_s, n = 0.0, 0.0, 0
        for _ in range(50):
            m1, m0 = _random_law(rng, int(rng.integers(1, 9)), max_mass=9)
            if sum(m1) == 0 or sum(m0) == 0:
                continue
            post = np.array([x / (x + y) if x + y else 0.0 for x, y in zip(m1, m0)])
            y = np.concatenate([np.r_[np.ones(x), np.zeros(z)] for x, z in zip(m1, m0)])
            sc = np.concatenate([np.full(x + z, p) for x, z, p in zip(m1, m0, post)])
            worst = max(worst, abs(roc_auc_score(y, sc) - auc_star(m1, m0)))
            arb = rng.integers(0, 3, len(m1)).astype(float)                # an arbitrary score with ties
            sa = np.concatenate([np.full(x + z, v) for x, z, v in zip(m1, m0, arb)])
            worst_s = max(worst_s, abs(roc_auc_score(y, sa) - auc_of_score(arb, m1, m0)))
            n += 1
        check('sklearn_roc_auc_agreement', worst < 1e-12 and worst_s < 1e-12, laws=n, max_abs_diff_posterior=worst,
              max_abs_diff_arbitrary_score=worst_s)
    guarded('sklearn_roc_auc_agreement', t_sklearn)

    # 9. AUC* >= AUC of any fixed score (exactly), and the likelihood-ratio score attains it
    def t_fixed():
        ok, n = True, 0
        for _ in range(60):
            m1, m0 = _random_law(rng, int(rng.integers(1, 8)))
            ex = auc_star_exact(m1, m0)
            for _ in range(8):
                sc = rng.integers(-2, 3, len(m1)).tolist()
                ok &= auc_of_score_exact(sc, m1, m0) <= ex
                ok &= auc_of_score(np.array(sc, float), m1, m0) <= float(ex) + 1e-12
                n += 1
            lr = [Fraction(10 ** 9) if y == 0 else Fraction(x, y) for x, y in zip(m1, m0)]
            ok &= auc_of_score_exact(lr, m1, m0) == ex
        check('auc_star_dominates_fixed_scores', ok, scores=n)
    guarded('auc_star_dominates_fixed_scores', t_fixed)

    # 10. plug-in MI: equals the joint-table formula, 0 for a one-wire law, nonnegative, PRC clip difference tiny
    def t_mi():
        worst, nonneg, clipd = 0.0, True, 0.0
        for _ in range(40):
            m1, m0 = _random_law(rng, int(rng.integers(1, 8)))
            mi = plugin_mi(m1, m0)
            worst = max(worst, abs(mi - plugin_mi_exact_terms(m1, m0)))
            nonneg &= mi >= -1e-15
            clipd = max(clipd, abs(plugin_mi(m1, m0, clip=1e-9) - mi))
        zero = plugin_mi([4], [9]) == 0.0
        check('plugin_mi_definition', worst < 1e-12 and nonneg and zero and clipd < 1e-6, max_abs_diff_joint_formula=worst,
              one_wire_zero=zero, max_clip_effect=clipd)
    guarded('plugin_mi_definition', t_mi)

    # 11. counterexample: lower MI / covariance / critic loss coexisting with HIGHER AUC*
    def t_counter():
        A1, A0, B1, B0 = [3, 2], [2, 3], [1, 9], [0, 10]
        aA, aB = auc_star_exact(A1, A0), auc_star_exact(B1, B0)
        miA, miB = plugin_mi(A1, A0), plugin_mi(B1, B0)

        def var_post(m1, m0):                                              # Cov(S, P(S=1|W)) = Var(P(S=1|W)), exact
            T = sum(m1) + sum(m0)
            rb = Fraction(sum(m1), T)
            return sum(Fraction(x + y, T) * (Fraction(x, x + y) - rb) ** 2 for x, y in zip(m1, m0))
        vA, vB = var_post(A1, A0), var_post(B1, B0)
        same_prior = Fraction(sum(A1), sum(A1) + sum(A0)) == Fraction(sum(B1), sum(B1) + sum(B0)) == Fraction(1, 2)
        brierA, brierB = Fraction(1, 4) - vA, Fraction(1, 4) - vB         # optimal Brier critic loss
        llA, llB = math.log(2) - miA, math.log(2) - miB                   # optimal log-loss critic loss (H(S) - MI)
        ok = (aA == Fraction(3, 5) and aB == Fraction(11, 20) and miA < miB and vA == Fraction(1, 100) and vB == Fraction(1, 76)
              and same_prior and brierA > brierB and llA > llB and aA > aB)
        check('counterexample_lower_mi_higher_auc', ok, law_A={'m1': A1, 'm0': A0, 'auc_star': str(aA), 'mi_nats': miA,
              'cov_S_bayes_score': str(vA), 'brier_critic': str(brierA), 'logloss_critic': llA},
              law_B={'m1': B1, 'm0': B0, 'auc_star': str(aB), 'mi_nats': miB, 'cov_S_bayes_score': str(vB),
                     'brier_critic': str(brierB), 'logloss_critic': llB})
    guarded('counterexample_lower_mi_higher_auc', t_counter)

    # 12. counterexample: lower linear separability coexisting with higher AUC* (XOR law)
    def t_xor():
        V = [(0, 0), (1, 1), (0, 1), (1, 0)]
        m1, m0 = [1, 1, 0, 0], [0, 0, 1, 1]
        lin = {auc_of_score_exact([a * x + b * y for x, y in V], m1, m0)
               for a in range(-4, 5) for b in range(-4, 5) if (a, b) != (0, 0)}
        xs = auc_star_exact(m1, m0)
        VA = [(Fraction(3, 5), Fraction(2, 5)), (Fraction(2, 5), Fraction(3, 5))]
        linA = max(auc_of_score_exact([a * x + b * y for x, y in VA], [3, 2], [2, 3])
                   for a in range(-4, 5) for b in range(-4, 5) if (a, b) != (0, 0))
        ok = lin == {Fraction(1, 2)} and xs == 1 and linA == Fraction(3, 5) == auc_star_exact([3, 2], [2, 3])
        check('counterexample_linear_separability', ok, xor_linear_aucs=sorted(str(v) for v in lin), xor_auc_star=str(xs),
              law_A_best_linear=str(linA))
    guarded('counterexample_linear_separability', t_xor)

    # 13. a larger estimated (pi-weighted) law: sorted O(M log M) path == O(M^2) closed form; people path == merged path
    def t_large():
        N, K = 20000, 1500
        cell = rng.integers(0, K, N)
        table = np.round(rng.dirichlet(np.ones(5), size=K), 3)              # rounding creates identical emitted vectors
        pi = np.clip(rng.beta(2, 2, N) * 0.6 + 0.2 * (cell % 3 == 0), 1e-6, 1 - 1e-6)
        _, m1, m0 = merge_wires(table[cell], pi)
        s1 = time.process_time()
        a = auc_star(m1, m0)
        t_sorted = time.process_time() - s1
        b = auc_star_closed_form(m1, m0)
        c = auc_star_people(table[cell], pi, 1 - pi)
        ok = abs(a - b) < 1e-10 and a == c and 0.5 <= a <= 1
        check('estimated_law_sorted_vs_closed_form', ok, distinct_wires=int(m1.shape[0]), sorted=a, closed_form=b,
              abs_diff=abs(a - b), sorted_cpu_s=round(t_sorted, 4))
    guarded('estimated_law_sorted_vs_closed_form', t_large)

    # 14. float masses (estimated laws) agree with their exact binary values; input validation refuses ill-posed laws
    def t_float_exact():
        ok = True
        for _ in range(15):
            k = int(rng.integers(1, 7))
            m1 = rng.random(k) * (rng.random(k) < 0.8)
            m0 = rng.random(k) * (rng.random(k) < 0.8)
            if m1.sum() == 0 or m0.sum() == 0:
                continue
            ex = auc_star_exact(m1, m0)
            ok &= ex == best_auc_by_enumeration(m1, m0) == auc_star_closed_form_exact(m1, m0)
            ok &= abs(auc_star(m1, m0) - float(ex)) < 1e-12
        refused = 0
        for bad in (([1, -1], [1, 1]), ([0, 0], [1, 2]), ([1, 2], [0, 0]), ([1, np.nan], [1, 1])):
            try:
                auc_star(*bad)
            except ValueError:
                refused += 1
        try:
            merge_wires(np.array([[0.5, np.nan]]), [1.0])
        except ValueError:
            refused += 1
        check('float_exact_agreement_and_validation', ok and refused == 5, refused=refused)
    guarded('float_exact_agreement_and_validation', t_float_exact)

    all_pass = all(c['pass'] for c in checks)
    return {'module': 'drf.finite_law', 'checks': checks, 'n_checks': len(checks),
            'n_pass': sum(c['pass'] for c in checks), 'all_pass': all_pass, 'cpu_s': round(time.process_time() - t0, 2)}


def _main(argv):
    import json
    if len(argv) >= 1 and argv[0] == 'selftest':
        out = selftest()
        print(json.dumps(out, indent=1, default=str))
        return 0 if out['all_pass'] else 1
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
