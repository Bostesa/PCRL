"""Role C (mathematics, invariants and protocol review) tests for the confidence-capacity study (qpc).

Owner: role C; this file only. Synthetic fixtures only: no real Adult data, no masked labels, no attackers fitted on
real rows.

    cd <worktree> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest qpc/tests/test_math_review.py -q

Layout:
  REFERENCE      independent re-implementations of the registered formulas (imports nothing from qpc or dpc).
  STRUCTURAL     standard statements (data processing, decision containment, plug-in MI bias) on fitting laws.
  STAGE A        qpc.kmeans / qpc.stagea / qpc.release / qpc.gate checked against the reference: source-rule
                 reproduction (A1), KL k-means++ determinism, convergence and best coherent iterate, coherence of
                 deployed statistics, empty cells, class preservation, larger capacity -> new cells, rate selection.
  STAGE B        (added at the Stage B review)
  SELECTION      (added at the selection review)
qpc modules are imported lazily inside the tests, so this file also runs before they exist (those tests skip).
"""
from __future__ import annotations

import importlib
import itertools
import math

import numpy as np
import pytest

EPS = 1e-12
CLIP = 1e-12


def _qpc(name):
    """Import qpc.<name> or skip (the module is owned by another role and may not exist yet)."""
    try:
        return importlib.import_module(f"qpc.{name}")
    except ModuleNotFoundError as e:
        if e.name == f"qpc.{name}":
            pytest.skip(f"qpc.{name} not present yet")
        raise


# ==================================================================================================================
# REFERENCE (no qpc / dpc imports)
# ==================================================================================================================

def ref_smooth(mean_p, c):
    """(mean_p + eps*1 + eps*e_c) / (1 + (K+1) eps), eps = 1e-12, float64 (prompt section 6)."""
    m = np.asarray(mean_p, np.float64)
    K = m.shape[-1]
    e = np.zeros(K)
    e[int(c)] = EPS
    return (m + EPS + e) / (1.0 + (K + 1) * EPS)


def ref_kl(p, q):
    """KL(p || q) in nats, 0 log 0 = 0, summed over k in increasing order (math.log; scalar loop)."""
    acc = 0.0
    for pk, qk in zip(np.asarray(p, float).tolist(), np.asarray(q, float).tolist()):
        if pk > 0:
            acc += pk * (math.log(pk) - math.log(qk))
    return acc


def ref_kl_matrix(P, Q):
    """Vectorised KL(p_r || q_j) with the same 0 log 0 rule (for larger fixtures)."""
    P = np.asarray(P, np.float64)
    Q = np.atleast_2d(np.asarray(Q, np.float64))
    lp = np.log(np.where(P > 0, P, 1.0))
    out = np.zeros((P.shape[0], Q.shape[0]))
    for k in range(P.shape[1]):
        pk = P[:, k:k + 1]
        out += np.where(pk > 0, pk * (lp[:, k:k + 1] - np.log(Q[None, :, k])), 0.0)
    return out


def ref_assign(P, centroids):
    """Nearest centroid by KL(p || centroid), ties -> lowest index (numpy argmin first-index rule)."""
    return np.argmin(ref_kl_matrix(P, centroids), axis=1)


def ref_mi(s, *codes):
    """Plug-in I(S; codes) in nats from exact counts; empty cells contribute 0; no smoothing."""
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


# ---------------------------------------------------------------- source (dpc) k-means rule, re-implemented

def ref_source_init(Pc, c, k):
    """Distinct vectors (ascending lexicographic), stably re-sorted by p_c descending, block-midpoint order statistics
    floor((2j+1) nU / (2k)), j = 0..k-1 (dpc METHOD_CARD / partition.py docstring)."""
    U = np.unique(Pc, axis=0)
    U = U[np.argsort(-U[:, c], kind="stable")]
    nU = U.shape[0]
    return U[[((2 * j + 1) * nU) // (2 * k) for j in range(k)]].copy()


def ref_source_kmeans(Pc, c, max_cells, rounds):
    """The dpc 20-round rule: assign (KL to smoothed centroid, first index); stop when the assignment repeats; else
    update centroid = member mean (empty keeps the old one). Cap reached -> one final assignment with the last
    centroids. Returns (centroids used by the final assignment (smoothed), final assignment, converged, rounds)."""
    n = Pc.shape[0]
    nU = np.unique(Pc, axis=0).shape[0]
    k = int(min(max_cells, nU, n))
    C = ref_source_init(Pc, c, k)
    Q = np.stack([ref_smooth(x, c) for x in C])
    prev, conv, used = None, False, 0
    for r in range(1, rounds + 1):
        a = ref_assign(Pc, Q)
        used = r
        if prev is not None and np.array_equal(a, prev):
            conv = True
            break
        cnt = np.bincount(a, minlength=k)
        S = np.stack([np.bincount(a, weights=Pc[:, t], minlength=k) for t in range(Pc.shape[1])], 1)
        for j in range(k):
            if cnt[j] > 0:                                   # empty cell keeps its previous centroid
                C[j] = S[j] / cnt[j]
        Q = np.stack([ref_smooth(x, c) for x in C])
        prev = a
    if not conv:
        a = ref_assign(Pc, Q)
        conv = bool(prev is not None and np.array_equal(a, prev))
    return Q, a, conv, used


def ref_coherent_policy(Pc, c, centroids):
    """The deployed policy defined by a set of assignment centroids on the fitting rows of class c: assignment,
    nonempty cells, member counts / sums (row order) and decoded prototypes smooth(S/n). Returns a dict."""
    a = ref_assign(Pc, centroids)
    keep = [j for j in range(centroids.shape[0]) if np.any(a == j)]
    n = np.array([int(np.sum(a == j)) for j in keep])
    S = np.stack([Pc[a == j].sum(0) for j in keep]) if keep else np.zeros((0, Pc.shape[1]))
    proto = np.stack([ref_smooth(S[i] / n[i], c) for i in range(len(keep))]) if keep else S
    remap = {j: i for i, j in enumerate(keep)}
    cell = np.array([remap[int(x)] for x in a])
    D = float(np.sum([ref_kl(Pc[r], proto[cell[r]]) for r in range(Pc.shape[0])]))
    return {"assign": a, "cell": cell, "keep": keep, "n": n, "S": S, "proto": proto, "sum_kl": D}


# ---------------------------------------------------------------- synthetic teacher-like fixtures

def teacher_like(n, K, seed, conc=0.6, sharp=1.0, ties=0, zeros=0):
    """Rows on the simplex with an argmax distribution like a task head; optional exact ties and exact zeros."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.full(K, conc), size=n) ** sharp
    P /= P.sum(1, keepdims=True)
    for r in range(min(ties, n)):
        i, j = rng.choice(K, 2, replace=False)
        v = np.full(K, 0.1 / max(K - 2, 1)) if K > 2 else np.zeros(K)
        v[i] = v[j] = 0.45 if K > 2 else 0.5
        P[r] = v / v.sum()
    for r in range(ties, min(ties + zeros, n)):
        v = rng.dirichlet(np.ones(K))
        v[rng.integers(K)] = 0.0
        P[r] = v / v.sum()
    return P


def binary_continuum(n, seed, lo=0.02, hi=0.98):
    """Binary rows [1-q, q] with q spread over a continuum (slow Lloyd convergence at many cells)."""
    rng = np.random.default_rng(seed)
    q = np.sort(rng.beta(0.7, 1.4, n) * (hi - lo) + lo)
    rng.shuffle(q)
    return np.stack([1 - q, q], 1)


# ==================================================================================================================
# STRUCTURAL STATEMENTS (standard facts, checked on fitting laws; not new theorems)
# ==================================================================================================================

def _class_code(P, m, seed):
    """An arbitrary deterministic class-preserving code: within each predicted class, bin by the top probability."""
    d = P.argmax(1)
    top = P.max(1)
    tok = np.zeros(len(P), int)
    nxt = 0
    for c in range(P.shape[1]):
        w = d == c
        if not w.any():
            continue
        edges = np.quantile(top[w], np.linspace(0, 1, m + 1)[1:-1]) if m > 1 else []
        b = np.searchsorted(edges, top[w], side="right")
        tok[w] = nxt + b
        nxt += m
    return tok, d


def test_decision_containment_and_chain_rule_on_fitting_law():
    """For any class-preserving code (decision = function of token): I(S;C) = I(S;d) + I(S;C|d) >= I(S;d), and the
    same for the pair; the empirical law is a law, so this holds exactly for plug-in quantities. It does NOT bound
    attacker AUC of a fixed parametric family; it says the decision is always recoverable from the release."""
    rng = np.random.default_rng(3)
    P1 = teacher_like(3000, 2, 1)
    P2 = teacher_like(3000, 6, 2)
    s = (rng.random(3000) < 0.3 + 0.4 * P2[:, 0]).astype(int)
    for m in (1, 2, 4, 8, 16):
        t1, d1 = _class_code(P1, m, 0)
        t2, d2 = _class_code(P2, m, 1)
        lhs = ref_mi(s, t1)
        assert abs(lhs - (ref_mi(s, d1) + ref_cmi(s, d1, t1))) < 1e-12
        assert lhs >= ref_mi(s, d1) - 1e-15
        pair = ref_mi(s, t1, t2)
        assert abs(pair - (ref_mi(s, d1, d2) + ref_cmi(s, d1 * 6 + d2, t1, t2))) < 1e-12
        assert pair >= ref_mi(s, d1, d2) - 1e-15
        assert pair >= max(lhs, ref_mi(s, t2)) - 1e-15       # monotonicity of MI in the released variables


def test_data_processing_for_deterministic_public_maps():
    """C = g(p) deterministic: on the fitting law I(S; g(p)) <= I(S; p) (p discretised to its exact distinct values),
    and coarsening a code (merging tokens) never increases plug-in MI. Standard data processing; the map reads only
    p, never S. Composition with the public map is what lets a source-score attacker simulate any code reader."""
    rng = np.random.default_rng(5)
    base = rng.integers(0, 40, 2000)
    s = (rng.random(2000) < 0.2 + 0.6 * (base % 7 == 0)).astype(int)
    full = ref_mi(s, base)
    for m in (20, 10, 5, 2, 1):
        g = (base * m) // 40
        assert ref_mi(s, g) <= full + 1e-15
        g2 = g // 2
        assert ref_mi(s, g2) <= ref_mi(s, g) + 1e-15


def test_plugin_mi_bias_grows_with_alphabet_under_null():
    """S independent of the code: the true MI is 0, but plug-in MI is positive and grows roughly like
    (cells - 1) / (2N). A fitted MI decrease at a larger alphabet is therefore not protection (prompt section 9)."""
    N = 15434
    rng = np.random.default_rng(11)
    vals = {}
    for L in (8, 64, 512, 2048):
        reps = []
        for _ in range(12):
            s = (rng.random(N) < 0.32).astype(int)
            c = rng.integers(0, L, N)
            reps.append((ref_mi(s, c), (len(np.unique(c)) - 1) / (2 * N)))
        vals[L] = float(np.mean([r[0] for r in reps]))
        approx = float(np.mean([r[1] for r in reps]))
        assert 0.6 * approx < vals[L] < 1.5 * approx, (L, vals[L], approx)
    assert vals[8] < vals[64] < vals[512] < vals[2048]
    assert vals[2048] > 0.05                         # comparable to the lambda-weighted differences being optimised


@pytest.mark.parametrize("K", [2, 6])
def test_smoothed_mean_of_same_class_vectors_keeps_the_class(K):
    """Means of same-predicted-class vectors (first-index ties, exact zeros, near-ties) plus the eps rule have that
    class as STRICT argmax in float64; this is the class-preservation identity the release relies on."""
    rng = np.random.default_rng(K + 40)
    for trial in range(300):
        c = int(rng.integers(K))
        n = int(rng.integers(1, 30))
        V = rng.dirichlet(np.full(K, 0.5), size=n)
        V[:, c] = V.max(1) + rng.choice([0.0, 1e-17, 1e-9, 0.2], size=n)
        if trial % 7 == 0:                                         # exact two-way tie with a LATER index
            j = (c + 1) % K
            if j > c:
                V[:, j] = V[:, c]
        V /= V.sum(1, keepdims=True)
        keep = V.argmax(1) == c
        if not keep.any():
            continue
        m = V[keep].mean(0)
        q = ref_smooth(m, c)
        assert np.all(q > 0) and abs(q.sum() - 1) < 1e-15 * K
        others = np.delete(q, c)
        assert q[c] > others.max(), (m, q)


# ==================================================================================================================
# STAGE A: KL k-means, starts, convergence, coherence, capacity (qpc.kmeans, qpc.release)
# ==================================================================================================================

def ref_kpp(Pc, c, k, seed, K):
    """Independent transcription of the registered KL k-means++ rule (distinct vectors with multiplicities; one
    rng.random() draw per centre from default_rng([seed, K, c]); weights w * min KL(u || smooth(chosen)); chosen set to
    0; roundoff in [-1e-12, 0) clipped; degenerate -> multiplicities of the unchosen)."""
    U, w = np.unique(Pc, axis=0, return_counts=True)
    w = w.astype(np.float64)
    rng = np.random.default_rng([int(seed), int(K), int(c)])

    def draw(v):
        cum = np.cumsum(v)
        t = rng.random() * cum[-1]
        i = int(np.searchsorted(cum, t, side="right"))
        return i if i < len(v) else int(np.flatnonzero(v > 0)[-1])
    chosen = [draw(w)]
    dmin = np.full(len(U), np.inf)
    for _ in range(1, k):
        q = ref_smooth(U[chosen[-1]], c)
        dk = np.array([ref_kl(u, q) for u in U])
        assert np.all(dk >= -1e-12)
        dk[dk < 0] = 0.0
        dmin = np.minimum(dmin, dk)
        v = w * dmin
        v[chosen] = 0.0
        if not v.sum() > 0:
            v = w.copy()
            v[chosen] = 0.0
        chosen.append(draw(v))
    return U[chosen], chosen


def ref_qpc_kmeans(Pc, c, C0, rounds=200, rtol=1e-9, patience=3):
    """Independent transcription of the registered qpc iteration (prompt section 7 A2 + METHOD_CARD): coherent iterate
    per assignment pass, stop at an assignment fixed point, or after `patience` successive relative objective changes
    below rtol (then one final update + assignment), or at the cap (final update + assignment); return the best
    coherent iterate (lowest objective, later pass wins exact ties). Empty cells keep their previous centroid."""
    Pc = np.asarray(Pc, np.float64)
    K = Pc.shape[1]
    C = np.array(C0, np.float64)
    k = C.shape[0]
    negent = float(sum(sum(p * math.log(p) for p in row if p > 0) for row in Pc.tolist()))

    def Q_of(Cm):
        return np.stack([ref_smooth(x, c) for x in Cm])

    def stats(a):
        n = np.bincount(a, minlength=k)
        S = np.stack([np.bincount(a, weights=Pc[:, t], minlength=k) for t in range(K)], 1)
        return n, S

    def J_of(n, S):
        tot = negent
        for j in range(k):
            if n[j] > 0:
                q = ref_smooth(S[j] / n[j], c)
                tot -= float(np.dot(S[j], np.log(q)))
        return tot
    Q = Q_of(C)
    traj, best, prev_a, prev_J, streak = [], None, None, None, 0
    reason = "cap"

    def rec(Qm, a):
        nonlocal best
        n, S = stats(a)
        J = J_of(n, S)
        traj.append(J)
        if best is None or J <= best[2]:
            best = (Qm.copy(), a.copy(), J, len(traj))
        return n, S, J
    for r in range(1, rounds + 1):
        a = ref_assign(Pc, Q)
        n, S, J = rec(Q, a)
        if prev_a is not None and np.array_equal(a, prev_a):
            reason = "assignment_fixed_point"
            break
        if prev_J is not None:
            streak = streak + 1 if abs(prev_J - J) / max(abs(prev_J), 1e-300) < rtol else 0
            if streak >= patience:
                C[n > 0] = S[n > 0] / n[n > 0, None]
                Q = Q_of(C)
                rec(Q, ref_assign(Pc, Q))
                reason = "relative_tolerance"
                break
        C[n > 0] = S[n > 0] / n[n > 0, None]
        Q = Q_of(C)
        prev_a, prev_J = a, J
    else:
        a2 = ref_assign(Pc, Q)
        rec(Q, a2)
        reason = "assignment_fixed_point" if np.array_equal(a2, prev_a) else "cap"
    return {"Q": best[0], "a": best[1], "J": best[2], "returned_pass": best[3], "traj": traj, "reason": reason,
            "rounds_used": r}


def _fixtures_stage_a():
    """(name, P, K) synthetic fixtures: binary continuum (slow Lloyd), six-class teacher-like rows with exact ties and
    exact zeros, and a six-class fixture with an ABSENT predicted class (class 5 never the argmax)."""
    out = [("bin_continuum", binary_continuum(1500, 0), 2),
           ("six_ties_zeros", teacher_like(1200, 6, 3, conc=0.5, ties=40, zeros=60), 6)]
    P = teacher_like(1200, 6, 4, conc=0.7)
    P[:, 5] *= 0.05                                            # class 5 never predicted (like Adult occupation)
    P /= P.sum(1, keepdims=True)
    P = P[P.argmax(1) != 5]
    out.append(("six_absent5", P, 6))
    return out


@pytest.mark.parametrize("m", [4, 8])
def test_a1_source_rule_reproduces_dpc_fit_fine_bitwise(m):
    """A1 semantics: rule 'dpc' with the single source start and 20 rounds is dpc.partition.fit_fine exactly
    (centroids, counts, sums, entropies, classes, fallbacks), and the qpc 'source' start passes 1..20 are the same
    passes (identical objective trajectory prefix), so the 200-round run isolates optimisation time only."""
    KM = _qpc("kmeans")
    from dpc import partition as DPT
    for name, P, K in _fixtures_stage_a():
        d = P.argmax(1)
        f20 = KM.fit_recipient(P, d, K, m, starts=("source",), rounds=20, rule="dpc")
        g = DPT.fit_fine(P, d, K, max_cells=m, rounds=20)
        for key in ("centroid", "n", "S", "A", "cell_class", "fallback", "mean"):
            assert np.array_equal(getattr(f20.partition, key), getattr(g, key)), (name, key)
        f200 = KM.fit_recipient(P, d, K, m, starts=("source",), rounds=200, rule="qpc")
        for c in range(K):
            pa, pb = f20.receipt["per_class"][c], f200.receipt["per_class"][c]
            if pa["fallback"]:
                assert pb["fallback"]
                continue
            ra, rb = pa["starts"][0], pb["starts"][0]
            n = min(20, len(ra["objective_trajectory"]) - (0 if ra["converged"] else 1), len(rb["objective_trajectory"]))
            assert ra["objective_trajectory"][:n] == rb["objective_trajectory"][:n], (name, c)
            # the longer run never returns a worse coherent iterate than the 20-round source code
            assert rb["objective"] <= ra["objective"] + 1e-12 * max(abs(ra["objective"]), 1.0), (name, c)


def test_a1_source_kmeans_matches_independent_reference():
    """dpc's rule (and hence the A1 reproduction) agrees with an independent transcription of the documented rule."""
    KM = _qpc("kmeans")
    for name, P, K in _fixtures_stage_a():
        d = P.argmax(1)
        for c in range(K):
            Pc = P[d == c]
            if Pc.shape[0] == 0:
                continue
            Q, a, rec = KM.kmeans_class(Pc, c, 8, "source", 20, "dpc")
            Qr, ar, conv, used = ref_source_kmeans(Pc, c, 8, 20)
            assert np.array_equal(a, ar), (name, c)
            assert np.allclose(Q, Qr, rtol=0, atol=1e-15), (name, c)
            assert rec["converged"] == conv and rec["rounds_used"] == used


@pytest.mark.parametrize("start", ["source", "kpp:20261006", "kpp:20261007"])
def test_qpc_iteration_matches_independent_reference_and_returns_best_coherent_iterate(start):
    """Registered convergence rule and best-coherent-iterate retention, against an independent transcription: same
    returned assignment, pass, stop reason and rounds; the returned objective is the minimum of the trajectory; the
    returned centroids reproduce the returned assignment when deployed (coherence) and the objective equals the
    row-level KL to the decoded prototypes; at an assignment fixed point the centroid IS the prototype."""
    KM = _qpc("kmeans")
    for name, P, K in _fixtures_stage_a():
        d = P.argmax(1)
        for c in range(K):
            Pc = P[d == c]
            if Pc.shape[0] == 0:
                continue
            kind, seed = KM.parse_start(start)
            k = int(min(8, np.unique(Pc, axis=0).shape[0], Pc.shape[0]))
            C0 = ref_source_init(Pc, c, k) if kind == "source" else ref_kpp(Pc, c, k, seed, K)[0]
            Q, a, rec = KM.kmeans_class(Pc, c, 8, start)
            ref = ref_qpc_kmeans(Pc, c, C0)
            assert np.array_equal(a, ref["a"]), (name, c, start)
            assert np.allclose(Q, ref["Q"], rtol=0, atol=1e-15)
            assert rec["stop_reason"] == ref["reason"] and rec["rounds_used"] == ref["rounds_used"]
            assert rec["returned_pass"] == ref["returned_pass"]
            assert abs(rec["objective"] - ref["J"]) <= 1e-11 * max(1.0, abs(ref["J"]))
            assert rec["objective"] == min(rec["objective_trajectory"])
            pol = ref_coherent_policy(Pc, c, Q)
            assert np.array_equal(pol["assign"], a)                     # deployed centroids reproduce the cells
            assert abs(pol["sum_kl"] - rec["objective"]) <= 1e-10 * max(1.0, abs(rec["objective"]))
            if rec["stop_reason"] == "assignment_fixed_point" and rec["returned_pass"] == len(rec["objective_trajectory"]):
                for i, j in enumerate(pol["keep"]):
                    assert np.array_equal(Q[j], ref_smooth(pol["S"][i] / pol["n"][i], c))


def test_kpp_definition_matches_reference_and_is_deterministic():
    """KL k-means++: chosen distinct vectors equal the independent transcription; repeated calls are identical;
    row order does not matter (distinct vectors with multiplicities); different registered seeds give different
    starts on a generic fixture; no distinct vector is chosen twice."""
    KM = _qpc("kmeans")
    P = teacher_like(800, 6, 9, conc=0.5, ties=20, zeros=30)
    d = P.argmax(1)
    perm = np.random.default_rng(0).permutation(len(P))
    differ = 0
    for c in range(6):
        Pc = P[d == c]
        if len(Pc) < 10:
            continue
        for seed in (20261006, 20261007):
            k = min(8, len(np.unique(Pc, axis=0)))
            C1, _, r1 = KM.kpp_init(Pc, c, k, seed, 6)
            C2, _, r2 = KM.kpp_init(Pc, c, k, seed, 6)
            Cp, _, _ = KM.kpp_init(P[perm][d[perm] == c], c, k, seed, 6)
            Cr, idx = ref_kpp(Pc, c, k, seed, 6)
            assert np.array_equal(C1, C2) and r1 == r2
            assert np.array_equal(C1, Cp)
            assert np.array_equal(C1, Cr)
            assert len(set(idx)) == len(idx)
        Ca = KM.kpp_init(Pc, c, min(8, len(np.unique(Pc, axis=0))), 20261006, 6)[0]
        Cb = KM.kpp_init(Pc, c, min(8, len(np.unique(Pc, axis=0))), 20261007, 6)[0]
        differ += int(not np.array_equal(Ca, Cb))
    assert differ >= 3


def test_kpp_sampling_law_is_divergence_weighted():
    """Brute-force law of the first two KL k-means++ centres on a 5-vector fixture with multiplicities, over 6000
    seeds: P(first = i) = w_i / W and P(second = j | first = i) = w_j KL(u_j || smooth(u_i)) / sum. Chi-square check
    (this tests the mathematical definition, not just the transcription)."""
    KM = _qpc("kmeans")
    U = np.array([[0.51, 0.49], [0.62, 0.38], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])   # np.unique (ascending) order
    w = np.array([5, 1, 3, 2, 4])
    Pc = np.repeat(U, w, axis=0)
    exp = np.zeros((5, 5))
    for i in range(5):
        kl = np.array([ref_kl(U[j], ref_smooth(U[i], 0)) for j in range(5)])
        kl[i] = 0.0
        v = w * np.maximum(kl, 0)
        exp[i] = w[i] / w.sum() * v / v.sum()
    n = 6000
    obs = np.zeros((5, 5))
    for s in range(n):
        _, _, r = KM.kpp_init(Pc, 0, 2, s, 2)
        i, j = r["chosen_distinct_index"]
        obs[i, j] += 1
    m = exp > 0
    assert obs[~m].sum() == 0                                   # never a zero-weight (already chosen) vector
    chi2 = float((((obs - n * exp) ** 2)[m] / (n * exp[m])).sum())
    assert chi2 < 60.0, chi2                                    # 19 dof; p ~ 3e-6 at 60


def test_kpp_roundoff_guard_and_degenerate_draws():
    """Initialisation KL in [-1e-12, 0) is clipped and counted, never changes the optimised objective; a real
    negative divergence (< -1e-12) is refused; near-duplicate vectors with zero clipped weight use the recorded
    degenerate rule (multiplicities of the unchosen), never a chosen vector twice."""
    KM = _qpc("kmeans")
    M, n = KM.guarded_kl(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
    assert M.min() >= 0
    with pytest.raises(ValueError):
        KM.guarded_kl(np.array([[0.5, 0.5]]), np.array([[0.4, 0.4]]) / 0.8 * 1.0 + np.array([[0.0, 1e-9]]))
    base = np.array([0.7, 0.3])
    U = np.stack([base, base + np.array([1e-17, -1e-17]) * 0, base + np.array([-2**-53, 2**-53])])
    Pc = np.unique(U, axis=0)
    C, _, r = KM.kpp_init(np.repeat(Pc, [3, 2][:len(Pc)], axis=0), 0, len(Pc), 20261006, 2)
    assert len(set(r["chosen_distinct_index"])) == len(r["chosen_distinct_index"])


def test_start_selection_is_per_class_lowest_fitting_kl_with_fixed_ties():
    """fit_recipient: per class, the winner is the lowest class-total fitting KL over the three registered starts
    (ties within 1e-12 relative -> earlier start), recomputed here from each start's deployed partition; the winner
    partition's total equals the sum of the winners; no label or SEX enters (the API only takes P and d = argmax P)."""
    KM = _qpc("kmeans")
    for name, P, K in _fixtures_stage_a():
        d = P.argmax(1)
        fit = KM.fit_recipient(P, d, K, 8)
        assert tuple(fit.receipt["starts"]) == ("source", "kpp:20261006", "kpp:20261007")
        tot = 0.0
        for c in range(K):
            Pc = P[d == c]
            pc = fit.receipt["per_class"][c]
            if Pc.shape[0] == 0:
                assert pc["fallback"] and pc["winner"] is None
                continue
            J = {}
            for s, part in fit.by_start.items():
                cen = part.centroid[part.cell_class == c]
                J[s] = ref_coherent_policy(Pc, c, cen)["sum_kl"]
            order = list(fit.receipt["starts"])
            w = order[0]
            for s in order[1:]:
                if J[s] < J[w] - 1e-12 * max(abs(J[w]), 1.0) - 1e-11 * max(abs(J[w]), 1.0):
                    w = s
            # tolerant comparison: the method compares sufficient-statistic objectives; ours are row sums
            assert pc["winner"] == w or abs(J[pc["winner"]] - J[w]) <= 2e-11 * max(abs(J[w]), 1.0), (name, c, J)
            assert abs(J[pc["winner"]] - pc["winner_objective"]) <= 1e-10 * max(1.0, abs(J[w]))
            tot += J[pc["winner"]]
        assert abs(tot / P.shape[0] - fit.receipt["mean_kl_fit"]) <= 1e-12 * max(1.0, tot)
        with pytest.raises(ValueError):
            KM.fit_recipient(P, (d + 1) % K, K, 8)                  # a non-argmax label array is refused


def test_training_statistics_equal_deployed_policy_and_release_rows():
    """The stored statistics (n, S, A) of the winner partition equal those of deploying its own centroids on the
    fitting rows (independent nearest-centroid code), the release prototypes are smooth(S/n) by the registered
    formula, the release decision is the teacher decision on every row, and the row-level mean KL of the released
    vectors equals the fitted objective (training statistics = deployed assignment statistics)."""
    KM = _qpc("kmeans")
    RL = _qpc("release")
    for name, P, K in _fixtures_stage_a():
        d = P.argmax(1)
        fit = KM.fit_recipient(P, d, K, 8)
        fine = fit.partition
        for c in range(K):
            idx = np.flatnonzero(fine.cell_class == c)
            Pc = P[d == c]
            if Pc.shape[0] == 0:
                assert idx.size == 1 and fine.fallback[idx[0]] and fine.n[idx[0]] == 0
                continue
            a = ref_assign(Pc, fine.centroid[idx])
            for jj, j in enumerate(idx):
                w = a == jj
                assert fine.n[j] == int(w.sum()) > 0
                assert np.allclose(fine.S[j], Pc[w].sum(0), rtol=0, atol=1e-12)
        pol = RL.make_policy(1, fine)
        tok, q, dec = RL.encode(pol, P, d)
        assert np.array_equal(dec, d)
        for t in np.unique(tok):
            c = int(pol.token_class[t])
            assert np.array_equal(q[tok == t][0], ref_smooth(pol.token_S[t] / pol.token_n[t], c))
        D = float(np.mean([ref_kl(P[r], q[r]) for r in range(len(P))]))
        assert abs(D - fit.receipt["mean_kl_fit"]) <= 1e-12 * max(1.0, D) + 1e-15


def test_larger_capacity_creates_new_cells_and_never_pretends():
    """m2 = 8 -> 64 on rows with many distinct vectors per class produces strictly more ACTUAL cells (<= m) and lower
    fitting KL; a class with only 10 distinct vectors gives exactly 10 cells at m = 64 (the alphabet reports actual
    states, not the cap); caps hold per class; the absent class keeps exactly one fallback token."""
    KM = _qpc("kmeans")
    RL = _qpc("release")
    P = teacher_like(6000, 6, 21, conc=0.8)
    P[:, 5] *= 0.05
    P /= P.sum(1, keepdims=True)
    P = P[P.argmax(1) != 5]
    d = P.argmax(1)
    few = np.flatnonzero(d == 4)
    lib = P[few[:10]]
    P[few] = lib[np.arange(few.size) % 10]                         # class 4: exactly 10 distinct vectors
    d = P.argmax(1)
    cells, Ds = {}, {}
    for m in (8, 16, 32, 64):
        fit = KM.fit_recipient(P, d, 6, m)
        pol = RL.make_policy(2, fit.partition)
        assert pol.caps_ok(m)
        cells[m] = pol.tokens_per_class()
        Ds[m] = fit.receipt["mean_kl_fit"]
        assert cells[m][5] == 1 and pol.token_fallback[pol.token_class == 5].all()
        assert cells[m][4] == min(m, 10)
        assert pol.T == sum(cells[m])
    for c in range(4):
        assert cells[8][c] < cells[16][c] < cells[32][c] < cells[64][c] <= 64, (c, cells)
        assert cells[64][c] >= 48                                    # most of the cap is realised on rich classes
    assert Ds[8] > Ds[16] > Ds[32] > Ds[64]


def test_empty_cells_keep_centroid_are_reported_and_removed():
    """An empty cell keeps its previous centroid during iteration (registered rule), is reported per pass, and is
    removed from the returned partition, so deployment can never route a fitting row to a cell without rows."""
    KM = _qpc("kmeans")
    found = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        q = np.r_[rng.uniform(0.01, 0.05, 30), rng.uniform(0.4, 0.5, 5), rng.uniform(0.6, 0.62, 3),
                  rng.uniform(0.9, 0.99, 30)]
        P = np.stack([1 - q, q], 1)
        P = P[P.argmax(1) == 0] if seed % 2 else P[P.argmax(1) == 1]
        c = int(P.argmax(1)[0])
        for start in ("source", "kpp:20261006", "kpp:20261007"):
            Q, a, rec = KM.kmeans_class(P, c, 12, start)
            if rec["empty_cell_events"] == 0:
                continue
            found += 1
            kind, sd = KM.parse_start(start)
            k = rec["initial_cells"]
            C0 = ref_source_init(P, c, k) if kind == "source" else ref_kpp(P, c, k, sd, 2)[0]
            ref = ref_qpc_kmeans(P, c, C0)
            assert np.array_equal(a, ref["a"])
            fit = KM.fit_recipient(P, np.full(len(P), c), 2, 12, starts=(start,))
            assert np.all(fit.partition.n[~fit.partition.fallback] > 0)
            pc = fit.receipt["per_class"][c]["starts"][0]
            assert pc["cells"] == rec["initial_cells"] - rec["empty_in_returned"]
        if found >= 3:
            break
    assert found >= 1, "no empty-cell fixture found"


@pytest.mark.parametrize("K", [2, 6])
def test_release_class_preservation_ties_underflow_and_absent_class(K):
    """Pointwise class preservation of a Stage A code on rows the fit never saw: exact first-index ties, exact 0/1
    vectors, tiny probabilities, and rows of a class that was absent from the fitting rows (fallback token)."""
    KM = _qpc("kmeans")
    RL = _qpc("release")
    P = teacher_like(600, K, 30 + K, conc=0.5)
    P = P[P.argmax(1) != K - 1]                                    # last class absent from fitting rows
    d = P.argmax(1)
    fit = KM.fit_recipient(P, d, K, 4)
    pol = RL.make_policy(1, fit.partition)
    new = [np.eye(K)[j] for j in range(K)]                         # exact 0/1, including the absent class
    for j in range(K - 1):
        v = np.zeros(K)
        v[j] = v[j + 1] = 0.5                                      # first-index tie -> class j
        new.append(v)
    v = np.full(K, 1e-300)
    v[K - 1] = 1 - (K - 1) * 1e-300
    new.append(v)
    new.append(np.full(K, 1.0 / K))                                 # K-way tie -> class 0
    X = np.array(new)
    tok, q, dec = RL.encode(pol, X)
    assert np.array_equal(dec, X.argmax(1))
    assert np.array_equal(q.argmax(1), dec)
    fb = pol.token_fallback[tok]
    assert np.array_equal(fb, X.argmax(1) == K - 1)
    assert np.allclose(q[fb], ref_smooth(np.full(K, 1.0 / K), K - 1), rtol=0, atol=0)
    with pytest.raises(Exception):
        RL.encode(pol, X, (X.argmax(1) + 1) % K)                    # a wrong decision array is refused


# ---------------------------------------------------------------- gate (A3 / A4) against the prompt's text

def _summ(elig, head, states, ll_occ, ll_inc, excess=0.5):
    return {"eligible_all_seeds": elig, "headroom_all_seeds": head and elig, "mean_total_states": states,
            "worst_seed_ll": {"occupation": ll_occ, "income": ll_inc}, "worst_seed_norm_excess": excess}


def ref_select_rates(summary):
    """Prompt section 7 A4, transcribed: headroom-on-all-seeds eligible configurations first, ordered by (fewer
    actual total states averaged over seeds, lower worst-seed occupation log loss, lower income log loss, ID); take
    the first two; otherwise fill the remaining slots with the other eligible configurations, same ordering."""
    key = lambda c: (summary[c]["mean_total_states"], summary[c]["worst_seed_ll"]["occupation"],  # noqa: E731
                     summary[c]["worst_seed_ll"]["income"], c)
    H = sorted([c for c, s in summary.items() if s["eligible_all_seeds"] and s["headroom_all_seeds"]], key=key)
    E = sorted([c for c, s in summary.items() if s["eligible_all_seeds"] and not s["headroom_all_seeds"]], key=key)
    return H[:2] if len(H) >= 2 else H + E[:2 - len(H)]


def test_gate_rate_selection_matches_the_registered_rule():
    GT = _qpc("gate")
    ids = GT.stagea_ids()
    assert len(ids) == 8 and ids[0] == "U|DIRECT-TASK|i4o8" and "U|DIRECT-TASK|i8o64" in ids
    rng = np.random.default_rng(7)
    for trial in range(400):
        summ = {}
        for cid in ids:
            e = bool(rng.random() < 0.4)
            summ[cid] = _summ(e, bool(rng.random() < 0.5), float(rng.choice([13, 17, 21, 25])),
                              float(rng.choice([1.20, 1.21])), float(rng.choice([0.33, 0.34])), float(rng.random()))
        dec = GT.select_rates(summ)
        assert dec["selected_rates"] == ref_select_rates(summ), trial
        if not dec["selected_rates"]:
            assert dec["gate"] == "CAPACITY_GATE_NOT_MET"
            assert dec["best_shortfall_config"] in ids
        else:
            assert dec["gate"] == "CAPACITY_GATE_MET" and dec["best_shortfall_config"] is None


def test_gate_eligibility_needs_every_seed_and_every_gate():
    """summarize: eligibility requires all three seeds eligible (no mean across seeds); headroom requires all seeds;
    the shortfall ordering uses the worst seed. One failing seed, or a missing seed, makes the configuration
    ineligible."""
    GT = _qpc("gate")
    UT = _qpc("utility")

    def metrics(acc, ll, br, const=0.6):
        return {"acc": acc, "logloss": ll, "brier": br, "const_acc": const}
    U = {"income": metrics(0.84, 0.338, 0.214), "occupation": metrics(0.475, 1.269, 0.652, 0.28)}

    def code(dll_o, dbr_o=0.0):
        return {"income": metrics(0.84, 0.339, 0.2142), "occupation": metrics(0.475, 1.269 + dll_o, 0.652 + dbr_o, 0.28)}
    ok = {1: True, 2: True}
    per = {k: {"gate": UT.gate_record(code(0.005), U, ok), "util": code(0.005), "alpha1": 9, "alpha2": 33}
           for k in (0, 1, 2)}
    s = GT.summarize(per)
    assert s["eligible_all_seeds"] and s["headroom_all_seeds"]
    per[2] = {"gate": UT.gate_record(code(0.0101), U, ok), "util": code(0.0101), "alpha1": 9, "alpha2": 33}
    s = GT.summarize(per)
    assert not s["eligible_all_seeds"] and abs(s["worst_seed_norm_excess"] - 1.01) < 1e-9
    per[2] = {"gate": UT.gate_record(code(0.008), U, ok), "util": code(0.008), "alpha1": 9, "alpha2": 33}
    s = GT.summarize(per)
    assert s["eligible_all_seeds"] and not s["headroom_all_seeds"]            # 0.008 > 0.0075 headroom
    per[2] = {"gate": UT.gate_record(code(0.0), U, {1: True, 2: False}), "util": code(0.0), "alpha1": 9, "alpha2": 33}
    assert not GT.summarize(per)["eligible_all_seeds"]                        # decision not preserved
    del per[2]
    assert not GT.summarize(per)["eligible_all_seeds"]                        # a missing seed fails


# ---------------------------------------------------------------- A1 unit, runner bindings, release format

def _synthetic_teacher_and_dpc_release(n=2400, n_fit=1600, seed=1):
    """Synthetic all-rows teacher T, fitting rows, and an 'admitted' dpc DIRECT-TASK m8 release built with dpc's own
    code on the fitting rows (as the source study did)."""
    from dpc import compress as DC
    from dpc import release as DRL
    P1 = binary_continuum(n, seed)
    P2 = teacher_like(n, 6, seed + 1)
    P2[:, 5] *= 0.05
    P2 /= P2.sum(1, keepdims=True)
    T = {"row_id": np.arange(n, dtype=np.int64) * 3 + 1, "p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}
    tr = np.arange(n_fit)
    s = np.random.default_rng(seed).integers(0, 2, n_fit)
    pair, _ = DC.fit_direct_task(P1[tr], T["d1"][tr], P2[tr], T["d2"][tr], s, 8)
    t1, q1, h1 = DRL.encode(pair.p1, P1, T["d1"])
    t2, q2, h2 = DRL.encode(pair.p2, P2, T["d2"])
    src = {"row_id": T["row_id"], "tok1": t1, "q1": q1, "hard1": h1, "alpha1": np.int64(pair.p1.T),
           "tok2": t2, "q2": q2, "hard2": h2, "alpha2": np.int64(pair.p2.T)}
    meta = {"teacher": "U", "seed": 0, "teacher_model_sha256": "a" * 64, "feature_names_sha256": "b" * 64}
    return T, tr, src, meta


def test_a1_unit_parity_with_admitted_source_release_and_alias_of_a2_source_start():
    """a1_unit reproduces the admitted dpc DIRECT-TASK m8 release bitwise (token IDs and decoded vectors on all
    rows); its 200-round variant IS A2's source-start receipt at (8, 8) (same partition fingerprint), so it is an
    alias, not an extra nominal fit; a perturbed admitted release is reported as a mismatch."""
    SA = _qpc("stagea")
    KM = _qpc("kmeans")
    T, tr, src, meta = _synthetic_teacher_and_dpc_release()
    rec, files = SA.a1_unit(T, tr, src, meta)
    par = rec["parity_with_admitted_release"]
    assert par["ok"] and par["q_rule"] == "bitwise" and par["ids_rule"] == "exact_ids"
    for i, K in ((1, 2), (2, 6)):
        P, d = T[f"p{i}"][tr], T[f"d{i}"][tr]
        a2 = KM.fit_recipient(P, d, K, 8)
        assert a2.by_start["source"].fingerprint() == rec["r200"]["per_recipient"][i]["partition_fingerprint"]
    bad = dict(src)
    bad["q2"] = src["q2"].copy()
    bad["q2"][0] = bad["q2"][0] + np.r_[1e-9, -1e-9, 0, 0, 0, 0]          # a 1e-9 change in one decoded vector
    rec2, _ = SA.a1_unit(T, tr, bad, meta)
    assert not rec2["parity_with_admitted_release"]["ok"] and "ENGINEERING_BLOCKER" in rec2


def test_runner_reads_the_keys_that_stagea_writes():
    """REQUIRED-1 / REQUIRED-2 fixture (Stage A review): qpc.run must read the A1 parity verdict and the A1
    convergence fields from keys that qpc.stagea.a1_unit actually writes. Otherwise a bitwise-parity A1 aborts as
    'REPRODUCTION MISMATCH' and CONVERGENCE_DIAGNOSTIC.csv silently gets empty fit-objective, convergence and work
    columns."""
    import inspect
    import json as _json
    SA = _qpc("stagea")
    RUN = _qpc("run")
    T, tr, src, meta = _synthetic_teacher_and_dpc_release()
    rec, _ = SA.a1_unit(T, tr, src, meta)
    rec = _json.loads(_json.dumps(rec))                                   # as read back from the unit record
    s_a = inspect.getsource(RUN.stage_stagea)
    s_g = inspect.getsource(RUN.stage_gate)
    if '"source_parity"' in s_a:
        assert "source_parity" in rec and rec["source_parity"]["ok"], "runner reads a parity key a1_unit never writes"
    else:
        assert "parity_with_admitted_release" in s_a
    if '"versions"' in s_g:
        assert "versions" in rec, "runner reads A1 convergence fields from a 'versions' key a1_unit never writes"
    for ver in ("src20", "r200"):
        for i in ("1", "2"):
            pr = rec[ver]["per_recipient"][i]
            assert pr["objective_total"] is not None and "work" in pr and "per_class" in pr


def test_asymmetric_caps_save_restore_and_no_hidden_ids():
    """Per-recipient caps are enforced separately (m1 for income, m2 for occupation); a pair over a cap is refused;
    JSON save/restore gives the identical assignment and decoded vectors bitwise; dpc records are refused as qpc
    releases; the release arrays carry only row_id, token, decoded vector, decision and alphabet size."""
    import json as _json
    SA = _qpc("stagea")
    RL = _qpc("release")
    T, tr, src, meta = _synthetic_teacher_and_dpc_release()
    p1d, _ = SA.recipient_fit(T["p1"][tr], T["d1"][tr], 2, 4, 1)
    p2d, _ = SA.recipient_fit(T["p2"][tr], T["d2"][tr], 6, 16, 2)
    rec, files = SA.pair_unit(p1d, p2d, T, {**meta, "config": "U|DIRECT-TASK|i4o16"}, tr)
    out = files["release.npz"]
    assert set(out) == {"row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"}
    p1, p2 = RL.Policy.from_dict(p1d), RL.Policy.from_dict(p2d)
    assert max(p1.tokens_per_class()) <= 4 and max(p2.tokens_per_class()) <= 16
    assert max(p2.tokens_per_class()) > 8
    with pytest.raises(ValueError):
        RL.make_pair(p1, p2, "DIRECT-TASK", 4, 8)                        # occupation exceeds m2 = 8
    with pytest.raises(ValueError):
        RL.make_pair(p2, p1, "DIRECT-TASK", 16, 4)                       # recipients swapped
    pair = RL.make_pair(p1, p2, "DIRECT-TASK", 4, 16, None, meta)
    z = _json.loads(_json.dumps(pair.to_dict(), allow_nan=False))
    back = RL.from_dict(z)
    for i, (a, b) in enumerate(((pair.p1, back.p1), (pair.p2, back.p2)), 1):
        ta, qa, da = RL.encode(a, T[f"p{i}"], T[f"d{i}"])
        tb, qb, db = RL.encode(b, T[f"p{i}"], T[f"d{i}"])
        assert np.array_equal(ta, tb) and np.array_equal(qa, qb) and np.array_equal(da, db)
        assert np.array_equal(ta, out[f"tok{i}"]) and np.array_equal(qa, out[f"q{i}"])
    z_dpc = dict(z)
    z_dpc["kind"] = "dpc.PolicyPair"
    with pytest.raises(ValueError):
        RL.from_dict(z_dpc)
    assert rec["alpha1"] == pair.p1.T and rec["alpha2"] == pair.p2.T == sum(pair.p2.tokens_per_class())
