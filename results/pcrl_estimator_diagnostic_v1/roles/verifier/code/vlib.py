"""EDG1 independent verifier, round 1: own exact AUC arithmetic, own wire merging, own sampler and own known-law constructions
built directly from PROTOCOL.md section 3. Synthetic data only. Nothing here is imported by the study code.

Independence from the study: AUC values are computed with exact rational arithmetic (fractions.Fraction / Python integers) by
BRUTE-FORCE enumeration over every pair of wire values (population) or every (positive, negative) pair of people (sample), ties
counted 1/2. The study's sort-and-cumulate float code (drf.finite_law._sorted_auc, drf.solver.auc_star, edg.estimators.auc_if)
is only ever the object under test. Ranks of exact keys are obtained by an exact sort (equal keys share a rank) and then compared
pair by pair.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

VCODE = Path(__file__).resolve().parent                 # D/roles/verifier/code
VER = VCODE.parent                                       # D/roles/verifier
D = VER.parent.parent                                    # results/pcrl_estimator_diagnostic_v1
W = D.parent.parent                                      # the worktree
OUT = VER / 'out'
STUDY_CODE = D / 'code'


def study_imports():
    """Make the study package importable (edg/__init__ then adds the publication code and the vendored recipes)."""
    if str(STUDY_CODE) not in sys.path:
        sys.path.insert(0, str(STUDY_CODE))


# ---------------------------------------------------------------- bookkeeping
def cpu_s():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def utc_now():
    """The clock, read with `date -u`."""
    return subprocess.run(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], capture_output=True, text=True, check=True).stdout.strip()


def _default(o):
    if isinstance(o, Fraction):
        return {'fraction': f'{o.numerator}/{o.denominator}' if len(str(o.denominator)) < 60 else 'long', 'float': float(o)}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(type(o))


def write_out(name, rec, started_utc=None):
    """Write OUT/name; refuses to overwrite (a failed output is kept and a corrected run gets a new name)."""
    rec = dict(rec)
    rec['cpu_s_rusage_user_plus_sys'] = round(cpu_s(), 3)
    if started_utc:
        rec['started_utc'] = started_utc
    rec['finished_utc'] = utc_now()
    rec['interpreter'] = sys.executable
    rec['numpy'] = np.__version__
    path = OUT / name
    if path.exists():
        raise FileExistsError(f'{path} exists: keep it and write a corrected run beside it')
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(rec, fh, indent=1, default=_default, allow_nan=True)
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- exact arithmetic
def F(x):
    """Exact rational value of an int, Fraction, Decimal or float (the float's exact binary value)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (bool, np.bool_)):
        return Fraction(int(x))
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    if isinstance(x, Decimal):
        return Fraction(x)
    f = float(x)
    if not math.isfinite(f):
        raise ValueError('non-finite value')
    return Fraction(f)


def dense_rank(keys):
    """Exact dense ranks (0 = lowest); equal keys (exact equality) share a rank. keys: floats, ints or Fractions (not mixed)."""
    keys = list(keys)
    order = sorted(set(keys))
    r = {k: i for i, k in enumerate(order)}
    return np.array([r[k] for k in keys], dtype=np.int64)


def _common_int(vals):
    fr = [F(v) for v in vals]
    den = 1
    for v in fr:
        den = den * v.denominator // math.gcd(den, v.denominator)
    return np.array([int(v * den) for v in fr], dtype=object), den


def pop_auc_brute(keys, m1, m0):
    """Exact population AUC of a fixed score of the wire values: sum over EVERY ordered pair (a, b) of wire values of
    P1(a) P0(b) ([k(a) > k(b)] + 1/2 [k(a) = k(b)]), P1 = m1 / sum m1, P0 = m0 / sum m0 (masses of any positive scale).
    keys: one exactly comparable score per wire value. Returns a Fraction (None if a class has zero mass)."""
    A, da = _common_int(m1)
    B, db = _common_int(m0)
    sa, sb = int(A.sum()), int(B.sum())
    if sa == 0 or sb == 0:
        return None
    r = dense_rank(keys)
    if len(r) != len(A) or len(A) != len(B):
        raise ValueError('one key and one mass pair per wire value')
    tot = 0
    for a in range(len(A)):
        if A[a] == 0:
            continue
        lt = r < r[a]
        eq = r == r[a]
        tot += A[a] * (2 * int(B[lt].sum()) + int(B[eq].sum()))
    return Fraction(int(tot), 2 * sa * sb)


def sample_auc_brute(y, score=None, exact_keys=None):
    """Exact sample AUC over EVERY (positive, negative) pair of people, ties 1/2: (#{s_i > s_j} + #{s_i = s_j} / 2) / (n1 n0).
    score: float array compared exactly, or exact_keys: one exact key per person. Returns a Fraction, or None (UNDEFINED) when a
    class is missing."""
    y = np.asarray(y).astype(bool)
    s = dense_rank(exact_keys) if exact_keys is not None else np.asarray(score, dtype=np.float64)
    pos, neg = s[y], s[~y]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None
    gt = eq = 0
    step = max(1, 2_000_000 // max(n0, 1))
    for i in range(0, n1, step):
        blk = pos[i:i + step, None]
        gt += int(np.count_nonzero(blk > neg[None, :]))
        eq += int(np.count_nonzero(blk == neg[None, :]))
    return Fraction(2 * gt + eq, 2 * n1 * n0)


def sample_auc_by_wire(keys, n1w, n0w):
    """Exact sample AUC from per-wire class counts by brute force over EVERY ordered pair (a, b) of wire values:
    sum n1_a n0_b ([k(a) > k(b)] + 1/2 [k(a) = k(b)]) / (n1 n0) (people sharing a wire share its score). keys: one exact key per
    wire (floats compared exactly, or Fractions). Integer arithmetic (int64 is safe: the double sum is at most 2 n1 n0)."""
    n1w = np.asarray(n1w, dtype=np.int64)
    n0w = np.asarray(n0w, dtype=np.int64)
    N1, N0 = int(n1w.sum()), int(n0w.sum())
    if N1 == 0 or N0 == 0:
        return None
    r = dense_rank(keys)
    gt = r[:, None] > r[None, :]
    eq = r[:, None] == r[None, :]
    prod = n1w[:, None] * n0w[None, :]
    return Fraction(2 * int(prod[gt].sum()) + int(prod[eq].sum()), 2 * N1 * N0)


def placements_brute(y, score):
    """Brute-force DeLong placement values (ties 1/2), as Fractions: for a positive i, V1_i = (#neg below + #neg tied / 2) / n0;
    for a negative j, V0_j = (#pos above + #pos tied / 2) / n1. Returns a list aligned with the people."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    pos, neg = s[y], s[~y]
    n1, n0 = len(pos), len(neg)
    out = []
    for i in range(len(s)):
        if y[i]:
            out.append(Fraction(2 * int(np.count_nonzero(neg < s[i])) + int(np.count_nonzero(neg == s[i])), 2 * n0))
        else:
            out.append(Fraction(2 * int(np.count_nonzero(pos > s[i])) + int(np.count_nonzero(pos == s[i])), 2 * n1))
    return out


def max_auc_by_enumeration(n1w, n0w):
    """Exact maximum sample AUC over ALL rankings (every map of the k wires to ranks 0..k-1, which covers every weak order) of a
    sample with per-wire class counts n1w, n0w (k <= 6). Independent check that the empirical-posterior ranking is the maximizer."""
    k = len(n1w)
    if k > 6:
        raise ValueError('k <= 6')
    N1, N0 = int(sum(n1w)), int(sum(n0w))
    best = -1
    import itertools
    for f in itertools.product(range(k), repeat=k):
        t = 0
        for a in range(k):
            if n1w[a] == 0:
                continue
            for b in range(k):
                if f[a] > f[b]:
                    t += 2 * n1w[a] * n0w[b]
                elif f[a] == f[b]:
                    t += n1w[a] * n0w[b]
        best = max(best, t)
    return Fraction(best, 2 * N1 * N0)


# ---------------------------------------------------------------- own merging and own sampler
def merge_own(vectors):
    """Observable wire id of every internal cell: cells share a wire iff the exact float64 bytes of their emitted vector(s) agree
    (a pair: both components). -0.0 is canonicalized to +0.0. Returns (wire id per cell in first-occurrence order, K)."""
    comps = vectors if isinstance(vectors, tuple) else (vectors,)
    rows = [np.ascontiguousarray(np.asarray(c, dtype=np.float64) + 0.0) for c in comps]
    n = rows[0].shape[0]
    keys = {}
    wid = np.empty(n, dtype=np.int64)
    for i in range(n):
        k = b'|'.join(r[i].tobytes() for r in rows)
        wid[i] = keys.setdefault(k, len(keys))
    return wid, len(keys)


def wire_masses_exact(p, q, wid, K):
    """Exact masses m1[w] = sum_{c in w} p_c q_c, m0[w] = sum_{c in w} p_c (1 - q_c) (inputs taken at their exact values)."""
    m1 = [Fraction(0)] * K
    m0 = [Fraction(0)] * K
    for c in range(len(p)):
        pc, qc = F(p[c]), F(q[c])
        m1[wid[c]] += pc * qc
        m0[wid[c]] += pc * (1 - qc)
    return m1, m0


def posterior_exact(m1, m0):
    return [a / (a + b) for a, b in zip(m1, m0)]


def pop_optimal_auc_exact(m1, m0):
    """A: the exact population AUC of the exact posterior (likelihood-ratio) ranking, brute-force pair sum."""
    return pop_auc_brute(posterior_exact(m1, m0), m1, m0)


def draw_people(p, q, n, rng):
    """Own sampler: n independent people, internal cell by inverse CDF of the cumulative masses, S ~ Bernoulli(q[cell])."""
    p = np.asarray(p, dtype=np.float64)
    cdf = np.cumsum(p)
    cdf = cdf / cdf[-1]
    c = np.minimum(np.searchsorted(cdf, rng.random(n), side='right'), len(p) - 1)
    s = (rng.random(n) < np.asarray(q, dtype=np.float64)[c]).astype(np.int64)
    return c, s


def counts(wid_people, y, K):
    n1 = np.bincount(wid_people, weights=y, minlength=K).astype(np.int64)
    n = np.bincount(wid_people, minlength=K).astype(np.int64)
    return n1, n - n1


def c_same_exact(wid_people, y, K):
    """Own empirical optimal AUC: rank wires by the exact empirical posterior n1_w / n_w, sample AUC by brute force over pairs."""
    n1, n0 = counts(wid_people, y, K)
    keys = [Fraction(int(a), int(a + b)) if a + b > 0 else Fraction(-1) for a, b in zip(n1, n0)]
    return sample_auc_brute(y, exact_keys=[keys[w] for w in wid_people])


# ---------------------------------------------------------------- own known-law constructions (PROTOCOL.md section 3)
getcontext().prec = 60
CHILD_SKEW_OWN = (Fraction(55, 100), Fraction(25, 100), Fraction(13, 100), Fraction(7, 100))


def own_occupancy(k, occ, rng):
    """Uniform 1/k, or Zipf weights 1/k (normalized) over a fixed (own-seeded) permutation of the atoms; exact rationals."""
    if occ == 'uniform':
        return [Fraction(1, k)] * k
    H = sum(Fraction(1, i) for i in range(1, k + 1))
    w = [Fraction(1, i) / H for i in range(1, k + 1)]
    perm = rng.permutation(k)
    return [w[int(j)] for j in perm]


def own_child_shares(occ, rng):
    if occ == 'uniform':
        return [Fraction(1, 4)] * 4
    perm = rng.permutation(4)
    return [CHILD_SKEW_OWN[int(j)] for j in perm]


def _dec(fr):
    return Decimal(fr.numerator) / Decimal(fr.denominator)


def _dsig(x):
    return Decimal(1) / (Decimal(1) + (-x).exp())


def own_coarse_signal(pi, occ, rng, beta=Fraction(1)):
    """16 coarse atoms, q_g = sigmoid(alpha + beta z_g), z on 8 equally spaced levels in [-1, 1] (atoms 2k, 2k+1 share level k),
    alpha solved by bisection in 60-digit decimal arithmetic so that sum_g w_g q_g = pi; q_g taken as the exact rational of that
    decimal. Returns (w, z, q, alpha, marginal error as a Fraction)."""
    w = own_occupancy(16, occ, rng)
    levels = [Fraction(-1) + Fraction(2 * k, 7) for k in range(8)]
    z = [levels[g // 2] for g in range(16)]
    Wd, Zd, Bd, Pd = [_dec(x) for x in w], [_dec(x) for x in z], _dec(beta), _dec(pi)
    lo, hi = Decimal(-30), Decimal(30)
    for _ in range(230):
        mid = (lo + hi) / 2
        if sum(wi * _dsig(mid + Bd * zi) for wi, zi in zip(Wd, Zd)) < Pd:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    q = [Fraction(_dsig(alpha + Bd * zi)) for zi in Zd]
    err = sum(wi * qi for wi, qi in zip(w, q)) - pi
    return w, z, q, alpha, err


def _softmax_rows(t):
    t = np.asarray(t, dtype=np.float64)
    e = np.exp(t - t.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _inc_rows(tau):
    s = 1.0 / (1.0 + np.exp(-np.asarray(tau, dtype=np.float64)))
    return np.column_stack((1.0 - s, s))


def own_law(family, pi, occ, seed):
    """Own exact-rational law: internal masses p (Fractions), posteriors q (Fractions), and per-view emitted float vectors of every
    internal cell (identity is what matters; built per the protocol's recipe with an own RNG)."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    pi = Fraction(pi).limit_denominator(1000) if isinstance(pi, float) else Fraction(pi)      # 0.5 -> 1/2, 0.2 -> 1/5 exactly
    out = {'family': family, 'pi': pi, 'occ': occ}
    if family == 'IND':
        a, b = own_occupancy(16, occ, rng), own_occupancy(64, occ, rng)
        ii, jj = np.repeat(np.arange(16), 64), np.tile(np.arange(64), 16)
        p = [a[i] * b[j] for i, j in zip(ii, jj)]
        q = [pi] * len(p)
        inc = _inc_rows(rng.normal(0, 1, 16))
        occv = _softmax_rows(rng.normal(0, 0.8, (64, 5)))
        views = {'income': inc[ii], 'occupation': occv[jj], 'pair': (inc[ii], occv[jj])}
    elif family == 'REF':
        w, z, qg, alpha, err = own_coarse_signal(pi, occ, rng)
        out['alpha'], out['marginal_error'], out['q_coarse'] = alpha, err, qg
        # internal cell c in [0, 1024): level-1024 index; ancestors by integer division (as the protocol's refinement tree)
        sh = [[own_child_shares(occ, rng) for _ in range(16 * 4 ** l)] for l in range(3)]
        p, q = [], []
        for c in range(1024):
            m = w[c // 64]
            for l in range(3):
                parent = c // (4 ** (3 - l))
                child = (c // (4 ** (2 - l))) % 4
                m = m * sh[l][parent][child]
            p.append(m)
            q.append(qg[c // 64])
        th = [np.column_stack((rng.normal(0, 1, 16), rng.normal(0, 0.7, (16, 4))))]
        for l in range(3):
            th.append(np.repeat(th[-1], 4, axis=0) + rng.normal(0, 0.15, (16 * 4 ** (l + 1), 5)))
        c = np.arange(1024)
        views = {f'L{16 * 4 ** l}': _softmax_rows(th[l])[c // (4 ** (3 - l))] for l in range(4)}
        out['node_of_cell'] = {f'L{16 * 4 ** l}': (c // (4 ** (3 - l))) for l in range(4)}
    elif family == 'RED':
        w, z, qg, alpha, err = own_coarse_signal(pi, occ, rng)
        out['alpha'], out['marginal_error'], out['q_coarse'] = alpha, err, qg
        e = own_occupancy(16, occ, rng)
        gg, zz = np.repeat(np.arange(16), 16), np.tile(np.arange(16), 16)
        p = [w[g] * e[k] for g, k in zip(gg, zz)]
        q = [qg[g] for g in gg]
        inc = _inc_rows(rng.normal(0, 1, 16))
        occv = _softmax_rows(rng.normal(0, 0.7, (64, 5)))
        views = {'income': inc[gg], 'occupation': occv[(gg // 4) * 16 + zz], 'pair': (inc[gg], occv[(gg // 4) * 16 + zz])}
        out['g_of_cell'] = gg
    elif family == 'COMP':
        a, b = own_occupancy(16, occ, rng), own_occupancy(16, occ, rng)
        U = [Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)] * 4
        V = [Fraction(1), Fraction(-1), Fraction(-1, 2), Fraction(1, 2)] * 4
        mu = sum(x * y for x, y in zip(a, U))
        u = [x - mu for x in U]
        mx = max(abs(x) for x in u)
        u = [x / mx for x in u]
        mv = sum(x * y for x, y in zip(b, V))
        v = [x - mv for x in V]
        mx = max(abs(x) for x in v)
        v = [x / mx for x in v]
        root = {Fraction(1, 2): Fraction(1, 2), Fraction(1, 5): Fraction(2, 5)}[pi]      # sqrt(pi (1 - pi)), exact
        delta = Fraction(24, 100) * root
        ii, jj = np.repeat(np.arange(16), 16), np.tile(np.arange(16), 16)
        p = [a[i] * b[j] for i, j in zip(ii, jj)]
        q = [pi + delta * u[i] * v[j] for i, j in zip(ii, jj)]
        inc = _inc_rows(rng.normal(0, 1, 16))
        th = np.column_stack((rng.normal(0, 1, 16), rng.normal(0, 0.7, (16, 4))))
        th[15] = th[14]                                   # occupation cells 14 and 15 emit the identical vector
        occv = _softmax_rows(th)
        views = {'income': inc[ii], 'occupation': occv[jj], 'pair': (inc[ii], occv[jj])}
        out.update({'a': a, 'b': b, 'u': u, 'v': v, 'delta': delta, 'weighted_mean_u': sum(x * y for x, y in zip(a, u)),
                    'weighted_mean_v': sum(x * y for x, y in zip(b, v))})
    else:
        raise ValueError(family)
    out['p'], out['q'] = p, q
    out['views'] = {}
    for name, vec in views.items():
        wid, K = merge_own(vec)
        m1, m0 = wire_masses_exact(p, q, wid, K)
        out['views'][name] = {'wid': wid, 'K': K, 'm1': m1, 'm0': m0}
    return out


def study_law_exact_view(law, name):
    """Own merging and exact masses of one view of a STUDY law (edg.laws.build output): wire ids from the emitted vectors."""
    if law['family'] == 'REF':
        vec = law['vectors']['levels'][name]
    elif name == 'pair':
        vec = (law['vectors']['income'], law['vectors']['occupation'])
    else:
        vec = law['vectors'][name]
    wid, K = merge_own(vec)
    m1, m0 = wire_masses_exact(law['p'], law['q'], wid, K)
    return wid, K, m1, m0
