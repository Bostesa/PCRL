# Post-AUDIT_AND_SELECTION_LOCK copy of qpc/tests/test_math_review.py with tests strengthened during mutation testing (math
# review role). The locked file stays byte-identical so stages verify; this file is not loaded by any stage.
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
import json
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
            # the registered tie rule, applied exactly to the method's own (verified) per-start objectives
            objs = [r["objective"] for r in pc["starts"]]
            for s_, o_ in zip(order, objs):
                assert abs(o_ - J[s_]) <= 1e-10 * max(1.0, abs(o_))
            wi = 0
            for i in range(1, len(objs)):
                if objs[i] < objs[wi] - 1e-12 * max(abs(objs[wi]), 1.0):
                    wi = i
            assert pc["winner"] == order[wi], (name, c, objs)
        assert abs(tot / P.shape[0] - fit.receipt["mean_kl_fit"]) <= 1e-12 * max(1.0, tot)
        with pytest.raises(ValueError):
            KM.fit_recipient(P, (d + 1) % K, K, 8)                  # a non-argmax label array is refused
    # every start reaches the same cells (few distinct vectors): an exact tie, so the FIRST start must win
    rng = np.random.default_rng(3)
    lib = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
    P = lib[rng.integers(0, 6, 500)]
    fit = KM.fit_recipient(P, P.argmax(1), 2, 8)
    for pc in fit.receipt["per_class"]:
        assert pc["winner"] == "source", pc["winner"]


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


# ==================================================================================================================
# SELECTION, VALIDITY AND INFERENCE (qpc.select, qpc.family, LABEL_TRUTH_TABLE.json) -- synthetic inner records only
# ==================================================================================================================

U_ANCHOR = {"income": {"acc": 0.8440, "logloss": 0.3380, "brier": 0.2140, "const_acc": 0.7600},
            "occupation": {"acc": 0.4750, "logloss": 1.2690, "brier": 0.6520, "const_acc": 0.2800}}
STAGE_A_RATES = [(a, b) for a in (4, 8) for b in (8, 16, 32, 64)]
LAMS = (0.01, 0.1, 1.0)


def _cid(fam, m1=None, m2=None, lam=None):
    if fam == "CLASS":
        return "U|CLASS|i1o1"
    return f"U|{fam}|i{m1}o{m2}" + (f"|l{lam:g}" if fam in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT") else "")


def _bank_ids(rates_b):
    ids = [_cid("DIRECT-TASK", a, b) for a, b in STAGE_A_RATES]
    for a, b in rates_b:
        ids.append(_cid("FINE-TASK", a, b))
        ids += [_cid(f, a, b, lam) for lam in LAMS for f in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")]
    return ids + ["U|CLASS|i1o1", "SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]


def _fam(cid):
    return cid.split("|")[0] if cid.startswith(("SRC", "REF")) else cid.split("|")[1]


def _rate(cid):
    if _fam(cid) in ("SRC", "REF", "CLASS"):
        return None
    r = cid.split("|")[2]
    a, b = r[1:].split("o")
    return int(a), int(b)


def _random_bank(seed, rates_b=((8, 32), (4, 64)), p_elig=0.5, priv_shift=0.0):
    """Synthetic inner records {(k, cid): record} in the qpc.audit inner-record format used by qpc.select."""
    rng = np.random.default_rng(seed)
    ids = _bank_ids(rates_b)
    recs = {}
    for cid in ids:
        fam = _fam(cid)
        elig_cfg = rng.random() < p_elig
        if fam == "DIRECT-TASK" and _rate(cid) in rates_b:
            elig_cfg = True                          # Stage B rates are eligible Stage A rates (as registered)
        base_pair = {"SRC": 0.858, "REF": 0.80, "CLASS": 0.739}.get(fam, 0.80)
        for k in (0, 1, 2):
            u = {t: dict(v) for t, v in U_ANCHOR.items()}
            if cid != "SRC|U":
                for t in u:
                    if fam == "REF":
                        u[t]["acc"] -= float(rng.choice([0.0, 0.004, 0.03]))
                    bad = (not elig_cfg) and (k == int(rng.integers(3)) or rng.random() < 0.3)
                    u[t]["logloss"] += float(rng.uniform(0.0105, 0.03) if bad and t == "occupation"
                                             else rng.uniform(0.0, 0.0095))
                    u[t]["brier"] += float(rng.uniform(0.0, 0.0045))
            pair = base_pair + float(rng.normal(0, 0.015))
            sh = priv_shift if fam in ("LOCAL", "SEQ-12", "SEQ-21") else 0.0
            sh = priv_shift * 1.5 if fam == "JOINT" else sh
            recs[(k, cid)] = {"recovery": {"auc": {"v1": 0.69 + sh + float(rng.normal(0, 0.006)),
                                                   "v2": pair - 0.05 + sh + float(rng.normal(0, 0.008)), "pair": pair}},
                              "utility": u, "preserved": {"1": True, "2": True},
                              "token_states": None if fam in ("SRC", "REF") else int(rng.integers(8, 80))}
    return ids, recs


def ref_eligible_seed(u, U, pres=True):
    for t in ("income", "occupation"):
        c, a = u[t], U[t]
        g, ga = c["acc"] - c["const_acc"], a["acc"] - a["const_acc"]
        if not (c["acc"] >= a["acc"] - 0.01 and c["logloss"] <= a["logloss"] + 0.01 and c["brier"] <= a["brier"] + 0.005
                and g >= 0.8 * ga and g >= 0.03):
            return False
    return bool(pres)


def ref_select(ids, recs):
    """PROTOCOL.md section 11 transcribed (clean banks: every unit present and finite)."""
    S = (0, 1, 2)
    U = {k: recs[(k, "SRC|U")]["utility"] for k in S}
    row = {}
    for c in ids:
        rs = [recs[(k, c)] for k in S]
        el = all(ref_eligible_seed(r["utility"], U[k]) for k, r in zip(S, rs))
        ne = max(max((r["utility"][t]["logloss"] - U[k][t]["logloss"]) / 0.01,
                     (r["utility"][t]["brier"] - U[k][t]["brier"]) / 0.005)
                 for k, r in zip(S, rs) for t in ("income", "occupation"))
        st = [math.inf if r["token_states"] is None else r["token_states"] for r in rs]
        row[c] = {"el": el, "ne": ne, "pair": np.mean([r["recovery"]["auc"]["pair"] for r in rs]),
                  "ll": np.mean([r["utility"]["income"]["logloss"] + r["utility"]["occupation"]["logloss"] for r in rs]),
                  "states": float(np.mean(st)), "auc": {k: rs[k]["recovery"]["auc"] for k in S}}
    key = lambda c: (row[c]["pair"], row[c]["ll"], row[c]["states"], c)  # noqa: E731

    def best(cands):
        e = [c for c in cands if row[c]["el"]]
        return min(e, key=key) if e else None

    def guard_ok(c, g):
        return all(row[c]["auc"][k][w] <= row[g]["auc"][k][w] + 0.005 for k in S for w in ("v1", "v2"))
    out = {}
    da = [c for c in ids if _fam(c) == "DIRECT-TASK"]
    qe = [c for c in da if row[c]["el"]]
    out["Q*"] = min(qe, key=lambda c: (row[c]["ne"], row[c]["states"], c)) if qe else None
    nonjoint = [c for c in ids if _fam(c) in ("DIRECT-TASK", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")]
    out["C_global"] = best(nonjoint + [c for c in ids if _fam(c) in ("CLASS", "SRC", "REF")])
    out["T*"] = best([c for c in ids if _fam(c) in ("DIRECT-TASK", "FINE-TASK", "CLASS", "SRC") or c == "REF|F0"])
    crate = {r: best([c for c in nonjoint if _rate(c) == r]) for r in {_rate(c) for c in ids if _fam(c) == "JOINT"}}
    J = [c for c in ids if _fam(c) == "JOINT" and row[c]["el"] and crate[_rate(c)] is not None
         and guard_ok(c, crate[_rate(c)]) and guard_ok(c, out["C_global"])]
    out["J*"] = min(J, key=key) if J else None

    def shortfall(c, guards):
        xs = [row[c]["auc"][k][w] - (row[g]["auc"][k][w] + 0.005) for g in guards if g for k in S
              for w in ("v1", "v2")]
        return max(0.0, max(xs)) if xs else 0.0

    def fallback(cands, guards_of):
        return min(cands, key=lambda c: (0.0 if row[c]["el"] else row[c]["ne"], shortfall(c, guards_of(c))) + key(c))
    jall = [c for c in ids if _fam(c) == "JOINT"]
    out["J*_fallback"] = None if J else fallback(jall, lambda c: (crate[_rate(c)], out["C_global"]))
    pall = [c for c in ids if _fam(c) in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")]
    P = [c for c in pall if row[c]["el"] and guard_ok(c, out["T*"])]
    out["P*"] = min(P, key=key) if P else None
    out["P*_fallback"] = None if P else fallback(pall, lambda c: (out["T*"],))
    jcell = out["J*"] or out["J*_fallback"]
    out["C_rate"] = crate.get(_rate(jcell))
    return out, row


@pytest.fixture
def synthetic_select(monkeypatch, tmp_path):
    """Run qpc.select.select_all on synthetic inner records (qpc.run record access monkeypatched; outputs to tmp)."""
    RUN = _qpc("run")
    SEL = _qpc("select")

    def go(ids, recs):
        store = {f"inner__{RUN.unit_for(k, c)}": r for (k, c), r in recs.items()}
        monkeypatch.setattr(RUN, "scored_ids", lambda protocol=None: list(ids))
        monkeypatch.setattr(RUN, "rec", lambda n: store[n])
        monkeypatch.setattr(RUN, "done", lambda n: n in store)
        monkeypatch.setattr(RUN, "RUN", tmp_path)
        monkeypatch.setattr(RUN, "PKG", tmp_path)
        monkeypatch.setattr(RUN, "event", lambda *a, **k: None)
        return SEL.select_all()
    return go


def test_selection_matches_independent_protocol_transcription(synthetic_select):
    """Q*, C_global, T*, C_rate(J*), J* and P* chosen by qpc.select equal an independent transcription of PROTOCOL
    section 11 on 40 random clean banks (eligibility recomputed here from the raw metrics, not via qpc.utility)."""
    seen = {"J*": 0, "P*": 0}
    for seed in range(60):
        ids, recs = _random_bank(seed, priv_shift=0.0 if seed < 20 else -0.03, p_elig=0.5 if seed < 40 else 0.8)
        out = synthetic_select(ids, recs)
        ref, _ = ref_select(ids, recs)
        st = out["statuses"]
        for role in ("Q*", "C_global", "T*", "J*", "P*", "C_rate"):
            got = st[role].get("config") if st[role]["status"] == "NOMINEE" else None
            assert got == ref[role], (seed, role, st[role]["status"], got, ref[role])
        for role in ("J*", "P*"):                                  # deterministic DESCRIPTIVE_ONLY fallbacks
            if ref[role] is None:
                assert st[role]["status"] == "NO_ELIGIBLE_NOMINEE"
                assert st[role]["descriptive_config"] == ref[role + "_fallback"], (seed, role)
        assert st["C_global"]["config"] is not None and st["T*"]["config"] is not None
        if ref["P*"]:
            assert st["P*"]["winning_family"] == _fam(ref["P*"])
        for role in seen:
            seen[role] += ref[role] is not None
    assert seen["J*"] >= 5 and seen["P*"] >= 5, seen


def test_selection_guards_are_per_seed_per_recipient_and_never_dropped(synthetic_select):
    """A JOINT code eligible and best on pair AUC but 0.006 above C_global on ONE recipient of ONE seed is not J*;
    a JOINT code exactly at the + 0.005 bound passes (<=). The fallback is DESCRIPTIVE (NO_ELIGIBLE_NOMINEE)."""
    ids, recs = _random_bank(3, p_elig=0.0)
    for k in (0, 1, 2):
        for c in ids:
            if _fam(c) in ("LOCAL", "SEQ-12", "SEQ-21", "FINE-TASK", "JOINT", "CLASS", "REF"):
                for t in ("income", "occupation"):
                    recs[(k, c)]["utility"][t]["logloss"] = U_ANCHOR[t]["logloss"] + 0.02   # ineligible
    j = _cid("JOINT", 8, 32, 0.1)
    for k in (0, 1, 2):
        recs[(k, j)]["utility"] = {t: dict(v) for t, v in U_ANCHOR.items()}
        recs[(k, j)]["recovery"]["auc"]["pair"] = 0.70
    out0 = synthetic_select(ids, recs)
    cg = out0["statuses"]["C_global"]["config"]
    for k in (0, 1, 2):
        for w in ("v1", "v2"):
            recs[(k, j)]["recovery"]["auc"][w] = recs[(k, cg)]["recovery"]["auc"][w] - 0.05
    for k in (0, 1, 2):
        cr = _cid("DIRECT-TASK", 8, 32)
        for w in ("v1", "v2"):
            recs[(k, cr)]["recovery"]["auc"][w] = max(recs[(k, cr)]["recovery"]["auc"][w],
                                                      recs[(k, j)]["recovery"]["auc"][w])
    out = synthetic_select(ids, recs)
    assert out["statuses"]["J*"]["status"] == "NOMINEE" and out["statuses"]["J*"]["config"] == j
    recs[(2, j)]["recovery"]["auc"]["v2"] = recs[(2, cg)]["recovery"]["auc"]["v2"] + 0.006
    out = synthetic_select(ids, recs)
    assert out["statuses"]["J*"]["status"] == "NO_ELIGIBLE_NOMINEE"
    assert out["statuses"]["J*"]["descriptive_config"] is not None


def test_guard_blocked_nominee_status_is_registered_in_protocol_and_truth_table(synthetic_select):
    """SEL-R1 fixture (selection review). An ELIGIBLE JOINT code whose own-rate C_rate has candidates but none
    eligible: qpc.select returns J* INVALID_NOMINEE ('blocked only by a missing guard comparator') and claim A's
    comparator INVALID_COMPARATOR ('no J* cell'). PROTOCOL section 11's status table and LABEL_TRUTH_TABLE.json
    define TECHNICAL_FAILURE as a missing/uncomputable candidate set and map 'candidates computed, none eligible' to
    NO_ELIGIBLE -> NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR; neither states the guard-blocked rule. The executable and
    the registered text must agree exactly, so the text must state this case (or the code must follow the text)."""
    import json as _json
    import pathlib
    ids, recs = _random_bank(5)
    rate = (4, 64)
    for k in (0, 1, 2):
        for c in ids:
            if _rate(c) == rate and _fam(c) != "JOINT":
                recs[(k, c)]["utility"]["occupation"]["logloss"] = U_ANCHOR["occupation"]["logloss"] + 0.02
        for c in ids:
            if _fam(c) == "JOINT":
                recs[(k, c)]["utility"]["occupation"]["logloss"] = U_ANCHOR["occupation"]["logloss"] + 0.02
        j = _cid("JOINT", 4, 64, 0.1)
        recs[(k, j)]["utility"] = {t: dict(v) for t, v in U_ANCHOR.items()}
        for w in ("v1", "v2"):
            recs[(k, j)]["recovery"]["auc"][w] = 0.5
    out = synthetic_select(ids, recs)
    st = out["statuses"]
    assert st["J*"]["status"] == "INVALID_NOMINEE" and st["J*"].get("blocked") == [_cid("JOINT", 4, 64, 0.1)]
    assert st["C_rate"]["status"] == "INVALID_COMPARATOR"
    root = pathlib.Path(__file__).resolve().parents[2] / "results" / "pcrl_confidence_capacity_v1"
    tt = _json.dumps(_json.loads((root / "LABEL_TRUTH_TABLE.json").read_text())).lower()
    proto = (root / "PROTOCOL.md").read_text().lower()
    assert "guard" in tt, "LABEL_TRUTH_TABLE.json does not register the guard-blocked nominee rule"
    sec11 = proto.split("## 11.")[1].split("## 12.")[0]
    assert "missing guard" in sec11 or "guard comparator" in sec11 or "blocked" in sec11, \
        "PROTOCOL.md section 11 does not register the guard-blocked nominee rule"


def _protocol_sec11():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "results" / "pcrl_confidence_capacity_v1"
    return (root / "PROTOCOL.md").read_text().split("## 11.")[1].split("## 12.")[0]


def test_claim_status_and_label_match_protocol_text_exhaustively():
    """Every (nominee, comparator, clauses) combination and every overall-label input combination agree with an
    independent transcription of PROTOCOL.md section 11 AS WRITTEN (status table, precedence, labels 1-8 after the
    SEL-C2 amendment), and the STAGE_A_LOCK rule (reported beside it) agrees with its own registered text. Whether
    an unresolved Q* (NOT_APPLICABLE_NO_Q despite a met gate) counts as missing coverage is read from label item 6
    of the text, so a text/code disagreement on that case fails here (SEL-R2)."""
    FAM = _qpc("family")

    def ref_claim(n, c, cl):
        if n == "TECHNICAL_FAILURE" or c == "TECHNICAL_FAILURE":
            return "INVALID_COMPARATOR" if c == "TECHNICAL_FAILURE" else "INVALID_NOMINEE"
        if n == "NO_ELIGIBLE":
            return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
        if c == "NO_ELIGIBLE":
            return "NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR"
        if cl == "ANY_INVALID":
            return "INVALID"
        return "PASS" if cl == "ALL_PASS" else "NOT_ESTABLISHED"
    for n, c, cl in itertools.product(FAM.ROLE_STATES, FAM.ROLE_STATES, FAM.CLAUSE_STATES):
        assert FAM.claim_status(n, c, cl) == ref_claim(n, c, cl), (n, c, cl)
    sec = _protocol_sec11()
    item6 = next(ln for ln in sec.splitlines() if ln.strip().startswith("6."))
    q_unresolved_missing = "unresolved" in item6.lower() or "not_applicable_no_q" in item6.lower()
    statuses = ["PASS", "NOT_ESTABLISHED", "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE",
                "NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR", "INVALID_COMPARATOR", "INVALID_NOMINEE", "INVALID"]
    missing = {"NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR", "INVALID_COMPARATOR", "INVALID_NOMINEE", "INVALID"}
    for sa, gm, tv in itertools.product([True, False], repeat=3):
        for a, b, c in itertools.product(statuses, repeat=3):
            for q in ("PASS", "NOT_ESTABLISHED", "INVALID", "NOT_APPLICABLE_NO_Q"):
                fav = []
                if a == "PASS" and b == "PASS":
                    fav.append("JOINT_DEVELOPMENT_CRITERION_MET")
                if c == "PASS":
                    fav.append("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (LOCAL)")
                gap = bool({a, b, c} & missing) or q == "INVALID" or (q == "NOT_APPLICABLE_NO_Q" and q_unresolved_missing)
                # amended rule (section 11 items 1-8)
                if not sa:
                    exp = "INCOMPLETE_OR_INVALID"
                elif not gm:
                    exp = "CAPACITY_GATE_NOT_MET"
                elif not tv:
                    exp = "INCOMPLETE_OR_INVALID"
                elif fav:
                    exp = " + ".join(fav)
                elif gap:
                    exp = "INCOMPLETE_OR_INVALID"
                elif q == "PASS":
                    exp = "CONFIDENCE_FEASIBILITY_ESTABLISHED"
                else:
                    exp = "EXPERIMENTAL_NO_ADVANTAGE"
                lab, miss = FAM.overall_label(sa, gm, tv, {"A": a, "B": b, "C": c}, q, "LOCAL")
                assert lab == exp, ("amended", sa, gm, tv, a, b, c, q, lab, exp)
                if sa and gm and tv and fav:
                    for cl_, st_ in (("A", a), ("B", b), ("C", c)):
                        if st_ in missing:
                            assert any(cl_ in x for x in miss), "a coverage gap beside a favourable label is listed"
                # STAGE_A_LOCK rule (any gap first; an unresolved Q* is a gap there, as registered)
                gap0 = bool({a, b, c} & missing) or q in ("INVALID", "NOT_APPLICABLE_NO_Q") or not tv
                if not sa:
                    exp0 = "INCOMPLETE_OR_INVALID"
                elif not gm:
                    exp0 = "CAPACITY_GATE_NOT_MET"
                elif gap0:
                    exp0 = "INCOMPLETE_OR_INVALID"
                elif fav:
                    exp0 = " + ".join(fav)
                else:
                    exp0 = "CONFIDENCE_FEASIBILITY_ESTABLISHED" if q == "PASS" else "EXPERIMENTAL_NO_ADVANTAGE"
                lab0, _ = FAM.overall_label_stage_a_rule(sa, gm, tv, {"A": a, "B": b, "C": c}, q, "LOCAL")
                assert lab0 == exp0, ("stage_a_rule", sa, gm, tv, a, b, c, q, lab0, exp0)


def test_family_37_fixed_slots_and_z():
    """37 fixed slots P01..P37 (A: 11, B: 11, C: 11, Q: 4), z = NormalDist().inv_cdf(1 - 0.05/74) exactly, strict
    clause sides and targets as registered."""
    from statistics import NormalDist
    FAM = _qpc("family")
    z = NormalDist().inv_cdf(1 - 0.05 / 74)
    assert z == FAM.Z_PRIMARY and repr(z) == "3.2048452050105634"
    assert [e["id"] for e in FAM.PRIMARY] == [f"P{i:02d}" for i in range(1, 38)]
    cnt = {c: sum(e["claim"] == c for e in FAM.PRIMARY) for c in "ABCQ"}
    assert cnt == {"A": 11, "B": 11, "C": 11, "Q": 4}
    exp = [("coalition", 0.02, "lower>"), ("local", 0.01, "upper<"), ("local", 0.01, "upper<"),
           ("acc", -0.01, "lower>"), ("acc", -0.01, "lower>"), ("logloss", 0.01, "upper<"), ("logloss", 0.01, "upper<"),
           ("brier", 0.005, "upper<"), ("brier", 0.005, "upper<"), ("retain", 0.0, "lower>"), ("retain", 0.0, "lower>")]
    for c in "ABC":
        got = [(e["kind"], e["target"], e["side"]) for e in FAM.PRIMARY if e["claim"] == c]
        assert got == exp, c
    q = [(e["kind"], e["task"], e["target"], e["side"]) for e in FAM.PRIMARY if e["claim"] == "Q"]
    assert q == [("logloss", 0, 0.01, "upper<"), ("logloss", 1, 0.01, "upper<"), ("brier", 0, 0.005, "upper<"),
                 ("brier", 1, 0.005, "upper<")]


# ==================================================================================================================
# STAGE B: objectives, deltas, search rules, sequential correction, JOINT witnesses (qpc.partition, qpc.compress)
# ==================================================================================================================

class RefRecip:
    """One recipient of a finite fixture: fine cells = distinct vectors V (class = argmax), rows -> fine cell f."""

    def __init__(self, V, f):
        self.V = np.asarray(V, np.float64)
        self.K = self.V.shape[1]
        self.cls = self.V.argmax(1)
        self.f = np.asarray(f, np.int64)
        self.P = self.V[self.f]
        self.F = len(self.V)
        self.n = np.bincount(self.f, minlength=self.F)
        self.Ssum = np.stack([self.P[self.f == j].sum(0) for j in range(self.F)])
        self.negent = np.array([sum(p * math.log(p) for p in self.V[j] if p > 0) for j in range(self.F)])
        self.cells_of = {c: [j for j in range(self.F) if self.cls[j] == c] for c in range(self.K)}


def ref_terms_fine(R1, R2, s, lab1, lab2):
    """D1, D2, I1, I2, I12 from fine-level statistics and exact count tables (own implementation)."""
    N = len(s)
    out = {}
    for name, R, lab in (("1", R1, lab1), ("2", R2, lab2)):
        D = float(np.dot(R.n, R.negent))
        for g in np.unique(lab):
            mem = np.flatnonzero(lab == g)
            n = R.n[mem].sum()
            if n == 0:
                continue
            S = R.Ssum[mem].sum(0)
            q = ref_smooth(S / n, R.cls[mem[0]])
            D -= float(np.dot(S, np.log(q)))
        out["D" + name] = D / N
    t1 = np.asarray(lab1)[R1.f]
    t2 = np.asarray(lab2)[R2.f]
    out["I1"], out["I2"], out["I12"] = ref_mi(s, t1), ref_mi(s, t2), ref_mi(s, t1, t2)
    return out


def ref_terms_rows(R1, R2, s, lab1, lab2):
    """The same terms recomputed from the released ROWS (tokens and decoded vectors), the brute-force definition."""
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


def ref_value(t, w):
    """w = (wD1, wD2, wI1, wI2, w12)."""
    return w[0] * t["D1"] + w[1] * t["D2"] + w[2] * t["I1"] + w[3] * t["I2"] + w[4] * t["I12"]


def W_ref(kind, lam=None, r=None):
    if kind == "task":
        return (float(r in (None, 1)), float(r in (None, 2)), 0.0, 0.0, 0.0)
    if kind == "local":
        return (float(r in (None, 1)), float(r in (None, 2)), lam / 2 if r in (None, 1) else 0.0,
                lam / 2 if r in (None, 2) else 0.0, 0.0)
    if kind == "joint":
        return (1.0, 1.0, lam / 2, lam / 2, lam)
    raise ValueError(kind)


class RefSearch:
    """Transcription of the registered Stage B search on label vectors (label = a member fine index at creation;
    merges keep the lower label; a move gives the moved cell the target label, so a label can become virtual)."""

    def __init__(self, R1, R2, s, labs, w, tol=1e-12, tie=1e-12):
        self.R = {1: R1, 2: R2}
        self.s = s
        self.lab = {1: np.array(labs[0], np.int64), 2: np.array(labs[1], np.int64)}
        self.w = w
        self.tol, self.tie = tol, tie

    def val(self, lab=None):
        lab = lab or self.lab
        return ref_value(ref_terms_fine(self.R[1], self.R[2], self.s, lab[1], lab[2]), self.w)

    def groups(self, r, c):
        return sorted({int(self.lab[r][j]) for j in self.R[r].cells_of[c]})

    def _merged(self, r, a, b):
        lab = {1: self.lab[1].copy(), 2: self.lab[2].copy()}
        lab[r][lab[r] == b] = a
        return lab

    def merge_step(self, recips, caps, improving):
        cur = self.val()
        cands = []
        for r in recips:
            for c in range(self.R[r].K):
                g = self.groups(r, c)
                if (improving and len(g) >= 2) or (not improving and len(g) > caps[r]):
                    for a, b in itertools.combinations(g, 2):
                        cands.append((self.val(self._merged(r, a, b)) - cur, r, c, a, b))
        if not cands:
            return False
        gmin = min(x[0] for x in cands)
        if improving and not gmin < -self.tol:
            return False
        for x in cands:                                          # (r, c, a, b) lexicographic generation order
            if x[0] <= gmin + self.tie and (not improving or x[0] < -self.tol):
                self.lab = self._merged(x[1], x[3], x[4])
                return True
        return False

    def greedy(self, recips, caps):
        while self.merge_step(recips, caps, improving=False):
            pass
        while self.merge_step(recips, caps, improving=True):
            pass

    def refine(self, recips, sweeps=5):
        for _ in range(sweeps):
            moved = 0
            for r in recips:
                for f in range(self.R[r].F):
                    a = int(self.lab[r][f])
                    if np.sum(self.lab[r] == a) <= 1:
                        continue
                    c = int(self.R[r].cls[f])
                    B = [g for g in self.groups(r, c) if g != a]
                    if not B:
                        continue
                    cur = self.val()
                    deltas = []
                    for b in B:
                        lab = {1: self.lab[1].copy(), 2: self.lab[2].copy()}
                        lab[r][f] = b
                        deltas.append(self.val(lab) - cur)
                    j = int(np.argmin(deltas))
                    if deltas[j] < -self.tol:
                        self.lab[r][f] = B[j]
                        moved += 1
            merged = 0
            while self.merge_step(recips, None, improving=True):
                merged += 1
            if moved == 0 and merged == 0:
                return True
        return False

    def canon(self, r):
        out = np.empty(self.R[r].F, np.int64)
        for g in np.unique(self.lab[r]):
            mem = np.flatnonzero(self.lab[r] == g)
            out[mem] = mem.min()
        return out


def ref_class_labels(R):
    first = {}
    return np.array([first.setdefault(int(c), j) for j, c in enumerate(R.cls)], np.int64)


def ref_family(fam, R1, R2, s, m1, m2, lam):
    """Reference FINE-TASK, LOCAL, SEQ-12, SEQ-21 (corrected: stage one under F_joint with the other recipient at its
    CLASS-ONLY release) and JOINT (5 refined starts + 4 unchanged witnesses, lowest F_joint, fixed tie order)."""
    caps = {1: m1, 2: m2}
    ident = (np.arange(R1.F), np.arange(R2.F))
    if fam in ("FINE-TASK", "LOCAL"):
        kind = "task" if fam == "FINE-TASK" else "local"
        X = RefSearch(R1, R2, s, ident, None)
        for r in (1, 2):
            X.w = W_ref(kind, lam, r)
            X.greedy((r,), caps)
        for r in (1, 2):
            X.w = W_ref(kind, lam, r)
            X.refine((r,))
        return X.canon(1), X.canon(2)
    if fam in ("SEQ-12", "SEQ-21"):
        a, b = (1, 2) if fam == "SEQ-12" else (2, 1)
        labs = {a: np.arange((R1, R2)[a - 1].F), b: ref_class_labels((R1, R2)[b - 1])}
        X = RefSearch(R1, R2, s, (labs[1], labs[2]), W_ref("joint", lam))
        X.greedy((a,), caps)
        X.refine((a,))
        frozen = X.canon(a)
        labs = {a: frozen, b: np.arange((R1, R2)[b - 1].F)}
        Y = RefSearch(R1, R2, s, (labs[1], labs[2]), W_ref("joint", lam))
        Y.greedy((b,), caps)
        Y.refine((b,))
        assert np.array_equal(Y.canon(a), frozen)
        return Y.canon(1), Y.canon(2)
    if fam == "JOINT":
        W = W_ref("joint", lam)
        X = RefSearch(R1, R2, s, ident, W)
        X.greedy((1, 2), caps)
        X.refine((1, 2))
        cands = [("JOINT-GREEDY", "refined", (X.canon(1), X.canon(2)))]
        wit = {f: ref_family(f, R1, R2, s, m1, m2, None if f == "FINE-TASK" else lam)
               for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")}
        for f, lw in wit.items():
            Y = RefSearch(R1, R2, s, lw, W)
            Y.refine((1, 2))
            cands.append((f, "refined", (Y.canon(1), Y.canon(2))))
        cands += [(f, "unchanged", lw) for f, lw in wit.items()]
        vals = [ref_value(ref_terms_fine(R1, R2, s, *c[2]), W) for c in cands]
        k = int(np.argmin(vals))
        return cands[k][2]
    raise ValueError(fam)


def _fine_from_recip(R):
    """qpc FinePartition whose cells are exactly the fixture's distinct vectors (cells in class order)."""
    KM = _qpc("kmeans")
    from dpc import partition as DPT
    order = np.argsort(R.cls, kind="stable")
    assert np.array_equal(order, np.arange(R.F)), "fixture vectors must be listed in class order"
    n, S, A = DPT.cell_stats(R.P, R.f, R.F)
    fine = KM.FinePartition(K=R.K, cell_class=R.cls.astype(np.int64), centroid=DPT.smooth(S / n[:, None], R.cls),
                            mean=S / n[:, None], n=n, S=S, A=A, fallback=np.zeros(R.F, bool))
    fine.validate()
    assert np.array_equal(KM.assign_fine(R.P, R.P.argmax(1), fine), R.f)
    return fine


def sb_fixture(name, seed=0):
    """Finite Stage B fixtures (rows drawn from distinct vectors listed in class order)."""
    rng = np.random.default_rng(seed)
    if name in ("small", "null"):
        V1 = [[1 - q, q] for q in (0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95)]
        V2 = []
        for c in range(3):
            for h in (0.40, 0.65, 0.90):
                v = np.full(3, (1 - h) / 2)
                v[c] = h
                V2.append(v.tolist())
        N = 360
        S = rng.integers(0, 2, N)
        f1 = 4 * rng.integers(0, 2, N) + 2 * rng.integers(0, 2, N) + rng.integers(0, 2, N)
        b = np.where(rng.random(N) < 0.8, (f1 % 4 >= 2).astype(int) ^ S, rng.integers(0, 2, N))
        f2 = 3 * rng.integers(0, 3, N) + np.where(rng.random(N) < 0.7, 2 * b, 1)
        if name == "null":
            S = np.random.default_rng(10_000 + seed).integers(0, 2, N)
    elif name == "xor":
        V1 = [[0.95, 0.05], [0.85, 0.15], [0.65, 0.35], [0.55, 0.45], [0.2, 0.8]]
        V2 = [[0.93, 0.07], [0.83, 0.17], [0.63, 0.37], [0.53, 0.47], [0.25, 0.75]]
        f1, f2, S = [], [], []
        for A in (0, 1):
            for B_ in (0, 1):
                for u in range(2):
                    for v in range(2):
                        f1 += [2 * A + u] * 20
                        f2 += [2 * B_ + v] * 20
                        S += [A ^ B_] * 20
        f1 += [4] * 30
        f2 += [4] * 30
        S += [0] * 15 + [1] * 15
        f1, f2, S = np.array(f1), np.array(f2), np.array(S)
    elif name == "six":
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
    else:
        raise ValueError(name)
    R1, R2 = RefRecip(V1, f1), RefRecip(V2, f2)
    return R1, R2, np.asarray(S, np.int64)


def _fit(fam, R1, R2, s, m1, m2, lam, **kw):
    CP = _qpc("compress")
    f1, f2 = _fine_from_recip(R1), _fine_from_recip(R2)
    return CP.fit_policy_pair(fam, f1, f2, R1.P, R1.P.argmax(1), R2.P, R2.P.argmax(1), s, m1, m2, lam, **kw)


def _labels(CP, pair):
    return CP.labels_from_policy(pair.p1), CP.labels_from_policy(pair.p2)


SB_SETTINGS = [("small", 2, 2, 0.1), ("small", 2, 3, 1.0), ("small", 1, 2, 10.0), ("xor", 2, 2, 1.0),
               ("xor", 2, 2, 10.0), ("six", 2, 1, 1.0), ("null", 2, 2, 1.0), ("small", 3, 2, 0.01)]


@pytest.mark.parametrize("name,m1,m2,lam", SB_SETTINGS)
def test_stageb_families_equal_independent_reference(name, m1, m2, lam):
    """FINE-TASK, LOCAL, SEQ-12, SEQ-21 (corrected) and JOINT solutions equal the independent transcription
    (identical canonical maps) at asymmetric caps; every receipt term equals the row-level brute force; the caps hold
    ('at most'); SEQ freezes its first map."""
    CP = _qpc("compress")
    R1, R2, s = sb_fixture(name)
    for fam in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT"):
        lm = None if fam == "FINE-TASK" else lam
        pair, rec = _fit(fam, R1, R2, s, m1, m2, lm)
        got = _labels(CP, pair)
        exp = ref_family(fam, R1, R2, s, m1, m2, lm)
        assert np.array_equal(got[0], exp[0]) and np.array_equal(got[1], exp[1]), (name, fam, got, exp)
        rows = ref_terms_rows(R1, R2, s, *got)
        for k in ("D1", "D2", "I1", "I2", "I12"):
            assert abs(rows[k] - rec["final"][k]) <= 1e-12 * max(1.0, abs(rows[k])), (fam, k)
        for pol, m in ((pair.p1, m1), (pair.p2, m2)):
            assert max(pol.tokens_per_class()) <= m


def test_merge_and_move_deltas_and_tables_against_brute_force():
    """Every merge and every single-cell move increment (dD, dI_own, dI12 and the weighted total) of
    qpc.compress.State equals the brute-force difference of row-level objectives, at random intermediate states
    including virtual labels; tables after apply_merge / apply_move equal recomputation from labels."""
    CP = _qpc("compress")
    rng = np.random.default_rng(4)
    for name in ("small", "six", "xor"):
        R1, R2, s = sb_fixture(name)
        f1, f2 = _fine_from_recip(R1), _fine_from_recip(R2)
        T = CP.fine_table(R1.f, R2.f, s, R1.F, R2.F)
        st = CP.State(f1, f2, T, np.arange(R1.F), np.arange(R2.F))
        W = CP.W_joint(0.7)
        w = (W.wD1, W.wD2, W.wI1, W.wI2, W.w12)
        for step in range(12):
            r = int(rng.integers(1, 3))
            R = (R1, R2)[r - 1]
            base = {1: st.labels(1), 2: st.labels(2)}
            t0 = ref_terms_rows(R1, R2, s, base[1], base[2])
            # merges
            for c in range(R.K):
                if st.count(r, c) < 2:
                    continue
                labs, ia, ib, delta, dD, dI, dI12 = st.merge_deltas(r, c, W)
                for j in range(len(ia)):
                    a, b = int(labs[ia[j]]), int(labs[ib[j]])
                    lab = {1: st.lab[1].copy(), 2: st.lab[2].copy()}
                    lab[r][lab[r] == b] = a
                    t1 = ref_terms_rows(R1, R2, s, lab[1], lab[2])
                    assert abs(dD[j] - (t1[f"D{r}"] - t0[f"D{r}"])) <= 1e-12
                    assert abs(dI[j] - (t1[f"I{r}"] - t0[f"I{r}"])) <= 1e-12
                    assert abs(dI12[j] - (t1["I12"] - t0["I12"])) <= 1e-12
                    assert abs(delta[j] - (ref_value(t1, w) - ref_value(t0, w))) <= 1e-12
            # moves
            G = st.G(r)
            for f in range(R.F):
                res = st.move_deltas(r, f, W, G)
                if res is None:
                    continue
                B, delta, dD, dI, dI12 = res
                for j, b in enumerate(B):
                    lab = {1: st.lab[1].copy(), 2: st.lab[2].copy()}
                    lab[r][f] = b
                    t1 = ref_terms_rows(R1, R2, s, lab[1], lab[2])
                    assert abs(delta[j] - (ref_value(t1, w) - ref_value(t0, w))) <= 1e-12, (name, r, f, b)
                    assert abs(dI12[j] - (t1["I12"] - t0["I12"])) <= 1e-12
            # random update (merge or move), then tables vs recomputation
            if rng.random() < 0.5:
                cs = [c for c in range(R.K) if st.count(r, c) >= 2]
                if cs:
                    labs = st.class_labels(r, int(rng.choice(cs)))
                    a, b = sorted(rng.choice(labs, 2, replace=False).tolist())
                    st.apply_merge(r, int(a), int(b))
            else:
                G = st.G(r)
                movable = [f for f in range(R.F) if st.move_deltas(r, f, W, G) is not None]
                if movable:
                    f = int(rng.choice(movable))
                    B = st.move_deltas(r, f, W, G)[0]
                    st.apply_move(r, f, int(rng.choice(B)), G)
            t = st.terms()
            tr = ref_terms_rows(R1, R2, s, st.labels(1), st.labels(2))
            for k in t:
                assert abs(t[k] - tr[k]) <= 1e-12, (name, step, k)
            T12 = np.zeros_like(st.T12)
            np.add.at(T12, (s, st.lab[1][R1.f], st.lab[2][R2.f]), 1)
            assert np.array_equal(T12, st.T12)


def test_at_most_cap_extra_merges_and_local_optimality():
    """'At most the cap': after the search no class exceeds its cap, and when refinement converged no remaining
    single merge or single-cell move improves the family's objective by more than TOL (so objective-improving merges
    below the cap were taken). A fixture where an extra merge below the cap is strictly improving shows fewer tokens
    than the cap."""
    CP = _qpc("compress")
    for name, m1, m2, lam in SB_SETTINGS:
        R1, R2, s = sb_fixture(name)
        for fam in ("LOCAL", "JOINT"):
            pair, rec = _fit(fam, R1, R2, s, m1, m2, lam)
            l1, l2 = _labels(CP, pair)
            if not rec["summary"]["converged"] or fam == "LOCAL":
                continue
            W = W_ref("joint", lam)
            X = RefSearch(R1, R2, s, (l1, l2), W)
            assert not X.merge_step((1, 2), None, improving=True), (name, fam)
            v0 = X.val()
            for r in (1, 2):
                R = (R1, R2)[r - 1]
                for f in range(R.F):
                    if np.sum(X.lab[r] == X.lab[r][f]) <= 1:
                        continue
                    for b in X.groups(r, int(R.cls[f])):
                        lab = {1: X.lab[1].copy(), 2: X.lab[2].copy()}
                        lab[r][f] = b
                        assert X.val(lab) >= v0 - 1e-12
    R1, R2, s = sb_fixture("small")
    pair, rec = _fit("LOCAL", R1, R2, s, 4, 3, 50.0)                 # strong privacy weight: merges below the cap
    assert max(pair.p1.tokens_per_class()) < 4 or max(pair.p2.tokens_per_class()) < 3
    assert rec["summary"]["extra_merges"] > 0


def test_sequential_correction_stage_one_is_F_joint_with_class_only_counterpart():
    """SEQ stage one optimises the first recipient under the actual F_joint with the other recipient at its CLASS-ONLY
    release (so the decision it always discloses is accounted for), not dpc's D + 1.5 lam I. On a fixture where the
    other recipient's decision is informative about S given the first code, the corrected first map differs from the
    old-rule map and has lower F_joint against the class-only counterpart; the receipt reports the correction."""
    CP = _qpc("compress")
    R1, R2, s = sb_fixture("six")
    lam = 1.0
    for fam, a, b in (("SEQ-12", 1, 2), ("SEQ-21", 2, 1)):
        pair, rec = _fit(fam, R1, R2, s, 2, 1, lam)
        corr = rec["baseline_correction"]
        assert corr["counterpart"] == "CLASS-ONLY"
        got = _labels(CP, pair)
        # stage-one value recomputed from rows with the class-only counterpart
        lab = {a: got[a - 1], b: ref_class_labels((R1, R2)[b - 1])}
        t = ref_terms_rows(R1, R2, s, lab[1], lab[2])
        fj = ref_value(t, W_ref("joint", lam))
        assert abs(fj - corr["stage1_F_joint_with_class_counterpart"]) <= 1e-12
        assert abs(corr["I12_with_class_counterpart"] - t["I12"]) <= 1e-12
        old = corr["old_rule_stage1"]
        assert old["corrected_minus_old_F_joint"] <= 1e-12                    # corrected is no worse on F_joint
    # a fixture where the corrections matter: the old rule ignores I(S; d_other | C_first)
    rng = np.random.default_rng(2)
    N = 600
    Sx = rng.integers(0, 2, N)
    V1 = [[0.95, 0.05], [0.8, 0.2], [0.65, 0.35], [0.55, 0.45], [0.4, 0.6], [0.2, 0.8]]
    V2 = [[0.7, 0.3], [0.3, 0.7]]
    d2 = np.where(rng.random(N) < 0.85, Sx, 1 - Sx)                          # recipient 2's DECISION carries S
    lvl = np.where(rng.random(N) < 0.75, d2 ^ (rng.random(N) < 0.5), rng.integers(0, 2, N))
    f1 = np.where(lvl == 1, rng.integers(0, 2, N), 2 + rng.integers(0, 2, N))
    f1 = np.where(rng.random(N) < 0.2, 4 + rng.integers(0, 2, N), f1)
    R1x, R2x = RefRecip(V1, f1), RefRecip(V2, d2)
    seen_diff = False
    for lam in (0.3, 1.0, 3.0):
        pair, rec = _fit("SEQ-12", R1x, R2x, Sx, 2, 1, lam)
        old = rec["baseline_correction"]["old_rule_stage1"]
        assert old["corrected_minus_old_F_joint"] <= 1e-12
        seen_diff |= not old["same_map_as_corrected"]
    assert seen_diff, "the correction should change the stage-one map on this fixture (it does at lam = 3)"
    exp = ref_family("SEQ-12", R1x, R2x, Sx, 2, 1, 1.0)
    pair, _ = _fit("SEQ-12", R1x, R2x, Sx, 2, 1, 1.0)
    assert np.array_equal(_labels(CP, pair)[0], exp[0])


def test_joint_candidates_and_witness_dominance():
    """JOINT keeps 9 candidates (5 refined starts + 4 unchanged witnesses), its final F_joint is <= every unchanged
    witness and every refined start (recomputed from rows), witnesses are the fitted FINE-TASK, LOCAL, SEQ-12 and
    SEQ-21 policies at the same caps and lam, and unresolved local optima are reported."""
    CP = _qpc("compress")
    for name, m1, m2, lam in SB_SETTINGS:
        R1, R2, s = sb_fixture(name)
        wits = {f: _fit(f, R1, R2, s, m1, m2, None if f == "FINE-TASK" else lam)[0]
                for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")}
        pair, rec = _fit("JOINT", R1, R2, s, m1, m2, lam, witnesses=wits)
        assert len(rec["candidates"]) == 9
        assert all(rec["starts"][f]["source"] == "passed_in" for f in wits)
        W = W_ref("joint", lam)
        fj = ref_value(ref_terms_rows(R1, R2, s, *_labels(CP, pair)), W)
        for f, w in wits.items():
            fw = ref_value(ref_terms_rows(R1, R2, s, *_labels(CP, w)), W)
            assert fj <= fw + 1e-12, (name, f)
        for n, srec in rec["starts"].items():
            assert fj <= srec["refined_F_joint"] + 1e-12
        assert isinstance(rec["unresolved_local_optima"], list)
        pair2, rec2 = _fit("JOINT", R1, R2, s, m1, m2, lam)                    # witnesses recomputed: same answer
        assert pair2.fingerprint() == pair.fingerprint()


# ---------------------------------------------------------------- exhaustive optimiser gaps (tiny fixtures only)

def _rgs(n, kmax):
    """Restricted-growth strings of length n with at most kmax blocks."""
    out = []

    def rec(prefix, nb):
        if len(prefix) == n:
            out.append(tuple(prefix))
            return
        for b in range(min(nb + 1, kmax)):
            rec(prefix + [b], max(nb, b + 1))
    rec([], 0) if n else out.append(())
    return out


def all_maps(R, m):
    """Every class-preserving map of R's fine cells with at most m coarse cells per class (canonical labels)."""
    per = []
    for c in range(R.K):
        cells = R.cells_of[c]
        per.append([(cells, g) for g in _rgs(len(cells), m)] if cells else [([], ())])
    out = []
    for combo in itertools.product(*per):
        lab = np.empty(R.F, np.int64)
        for cells, g in combo:
            first = {}
            for j, b in zip(cells, g):
                first.setdefault(b, j)
                lab[j] = first[b]
        out.append(lab)
    return out


def _fast_parts(R, s, labs):
    """Per map: D (row mean KL to decoded) and plug-in I(S; C) from fine-level sums; also row tokens."""
    N = len(s)
    res = []
    for lab in labs:
        D = float(np.dot(R.n, R.negent))
        for g in np.unique(lab):
            mem = lab == g
            n = R.n[mem].sum()
            S = R.Ssum[mem].sum(0)
            D -= float(np.dot(S, np.log(ref_smooth(S / n, R.cls[np.flatnonzero(mem)[0]]))))
        tok = lab[R.f]
        res.append((D / N, _fast_mi(s, tok), tok))
    return res


def _fast_mi(s, tok):
    _, t = np.unique(tok, return_inverse=True)
    tab = np.bincount(s * (t.max() + 1) + t, minlength=2 * (t.max() + 1)).reshape(2, -1).astype(float)
    N = tab.sum()
    ns, nc = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    m = tab > 0
    return float(np.sum(np.where(m, tab / N * np.log(np.where(m, tab * N, 1) / np.where(m, ns * nc, 1)), 0.0)))


def exhaustive_gaps(name, m1, m2, lam):
    """Global minima over the at-most-cap map space of each family's own objective, and the method's gaps."""
    CP = _qpc("compress")
    R1, R2, s = sb_fixture(name)
    L1, L2 = all_maps(R1, m1), all_maps(R2, m2)
    p1, p2 = _fast_parts(R1, s, L1), _fast_parts(R2, s, L2)
    i12 = np.array([[_fast_mi(s, a[2] * (R2.F + 1) + b[2]) for b in p2] for a in p1])
    D1 = np.array([x[0] for x in p1])[:, None]
    D2 = np.array([x[0] for x in p2])[None, :]
    I1 = np.array([x[1] for x in p1])[:, None]
    I2 = np.array([x[1] for x in p2])[None, :]
    Fj = D1 + D2 + lam * ((I1 + I2) / 2 + i12)
    Fl = D1 + D2 + lam * (I1 + I2) / 2
    Ft = D1 + D2
    idx1 = {tuple(l): i for i, l in enumerate(L1)}
    idx2 = {tuple(l): i for i, l in enumerate(L2)}
    cl1, cl2 = idx1[tuple(ref_class_labels(R1))], idx2[tuple(ref_class_labels(R2))]
    out = {"fixture": name, "m1": m1, "m2": m2, "lam": lam, "space": [len(L1), len(L2)],
           "global_min": {"F_task": float(Ft.min()), "F_local": float(Fl.min()), "F_joint": float(Fj.min())}}
    for fam in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT"):
        pair, rec = _fit(fam, R1, R2, s, m1, m2, None if fam == "FINE-TASK" else lam)
        l1, l2 = _labels(CP, pair)
        i, j = idx1[tuple(l1)], idx2[tuple(l2)]
        own = {"FINE-TASK": Ft, "LOCAL": Fl}.get(fam, Fj)
        e = {"value_own": float(own[i, j]), "gap_own": float(own[i, j] - own.min()),
             "gap_F_joint": float(Fj[i, j] - Fj.min()), "globally_best_own": bool(own[i, j] - own.min() <= 1e-12)}
        if fam.startswith("SEQ"):
            if fam == "SEQ-12":
                st1 = Fj[:, cl2]                                    # stage one: recipient 1 vs class-only 2
                e["stage1_gap"] = float(st1[i] - st1.min())
                e["stage2_conditional_gap"] = float(Fj[i, j] - Fj[i, :].min())
            else:
                st1 = Fj[cl1, :]
                e["stage1_gap"] = float(st1[j] - st1.min())
                e["stage2_conditional_gap"] = float(Fj[i, j] - Fj[:, j].min())
        out[fam] = e
    return out


EXHAUSTIVE = [("small", 2, 2, 0.1), ("small", 2, 2, 1.0), ("small", 2, 2, 10.0), ("small", 1, 2, 1.0),
              ("small", 2, 1, 10.0), ("xor", 2, 2, 0.1), ("xor", 2, 2, 1.0), ("xor", 2, 2, 10.0), ("null", 2, 2, 1.0),
              ("null", 2, 2, 10.0), ("six", 2, 1, 1.0), ("six", 3, 1, 10.0)]


@pytest.mark.parametrize("setting", EXHAUSTIVE[:6:5] + EXHAUSTIVE[7:8])
def test_exhaustive_gaps_are_nonnegative_and_fine_task_is_exact(setting):
    """On tiny exhaustive fixtures no family beats the global minimum of its own objective (sanity of the search
    space: at most the cap, class-preserving); FINE-TASK at m = 2 attains its global optimum on these fixtures; gaps
    of the privacy families are recorded (MATH_REVIEW.md), never promoted to an Adult certificate."""
    g = exhaustive_gaps(*setting)
    for fam in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT"):
        assert g[fam]["gap_own"] >= -1e-12, (setting, fam, g[fam])
        if fam.startswith("SEQ"):
            assert g[fam]["stage1_gap"] >= -1e-12 and g[fam]["stage2_conditional_gap"] >= -1e-12
    assert g["JOINT"]["value_own"] <= min(g[f]["value_own"] if f != "FINE-TASK" else
                                          g[f]["gap_F_joint"] + g["global_min"]["F_joint"]
                                          for f in ("FINE-TASK", "SEQ-12", "SEQ-21")) + 1e-12


def test_xor_fixture_requires_coordinated_moves():
    """S = A xor B (A, B = confidence sub-level clues on recipients 1 and 2; each alone independent of S). At m = 2
    per class JOINT removes the coalition leak by cross-mixing one recipient, but lands on the costlier pairing: its
    solution is a strict local optimum for every single-cell move and every merge, while the global optimum of
    F_joint (exhaustive) is lower and differs only by SWAPPING two fine cells between the two coarse cells of one
    class, i.e. it needs a coordinated two-cell move. The gap is the registered search's limitation on this fixture,
    recorded in MATH_REVIEW.md; it is not a defect and not an Adult certificate."""
    CP = _qpc("compress")
    R1, R2, s = sb_fixture("xor")
    L1, L2 = all_maps(R1, 2), all_maps(R2, 2)
    for lam in (1.0, 10.0):
        W = W_ref("joint", lam)
        best = min(((ref_value(ref_terms_fine(R1, R2, s, a, b), W), a, b) for a in L1 for b in L2),
                   key=lambda x: x[0])
        pair, rec = _fit("JOINT", R1, R2, s, 2, 2, lam)
        l1, l2 = _labels(CP, pair)
        X = RefSearch(R1, R2, s, (l1, l2), W)
        v = X.val()
        assert v - best[0] > 1e-3                                       # a real gap ...
        assert not X.merge_step((1, 2), None, improving=True)          # ... at a strict local optimum
        for r in (1, 2):
            R = (R1, R2)[r - 1]
            for f in range(R.F):
                if np.sum(X.lab[r] == X.lab[r][f]) <= 1:
                    continue
                for b in X.groups(r, int(R.cls[f])):
                    lab = {1: X.lab[1].copy(), 2: X.lab[2].copy()}
                    lab[r][f] = b
                    assert X.val(lab) >= v - 1e-12

        def pairs(lab):
            return {(i, j) for i in range(len(lab)) for j in range(i + 1, len(lab)) if lab[i] == lab[j]}
        d1, d2 = len(pairs(l1) ^ pairs(best[1])), len(pairs(l2) ^ pairs(best[2]))
        assert sorted((d1, d2)) == [0, 4]                               # a two-cell swap in one class
        assert rec["unresolved_local_optima"]                           # reported, not hidden
        assert v <= min(ref_value(ref_terms_fine(R1, R2, s, *_labels(CP, _fit(f, R1, R2, s, 2, 2, lam)[0])), W)
                        for f in ("LOCAL", "SEQ-12", "SEQ-21")) + 1e-12


def run_exhaustive(out_path=None):
    res = [exhaustive_gaps(*st) for st in EXHAUSTIVE]
    if out_path:
        import json as _json
        with open(out_path, "w") as fh:
            _json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    import sys
    if "--exhaustive" in sys.argv:
        path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
        for r in run_exhaustive(path):
            print(r["fixture"], r["m1"], r["m2"], r["lam"], r["space"],
                  {f: (round(r[f]["gap_own"], 6), round(r[f].get("stage1_gap", 0), 6),
                       round(r[f].get("stage2_conditional_gap", 0), 6)) for f in
                   ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT")})


def test_fine_unit_caps_starts_rows_and_deployment():
    """qpc.partition.fine_unit: income <= 32 and occupation <= 128 cells per predicted class, fitted on the
    DEFENSE_FIT rows only, with the Stage A starts and rule (identical to kmeans.fit_recipient at those caps); the
    private assignment of ALL rows equals deployment, its fitting-row counts equal the stored statistics; support
    receipts are those of the stored cells; caps are realised (more than 16 / 64 cells on rich classes)."""
    PT = _qpc("partition")
    KM = _qpc("kmeans")
    n, n_fit = 9000, 6000
    P1 = binary_continuum(n, 7)
    P2 = teacher_like(n, 6, 8, conc=0.9)
    P2[:, 5] *= 0.05
    P2 /= P2.sum(1, keepdims=True)
    T = {"row_id": np.arange(n) * 5 + 2, "p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}
    tr = np.arange(1000, 1000 + n_fit)
    rec, files = PT.fine_unit(T, tr)
    import json as _json
    fd = {}
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as td:
        files["fine.json"](os.path.join(td, "f.json"))
        fd = _json.loads(open(os.path.join(td, "f.json")).read())
    f1, f2 = PT.load_fine(fd)
    for f, P, d, K, cap in ((f1, P1, T["d1"], 2, 32), (f2, P2, T["d2"], 6, 128)):
        ref = KM.fit_recipient(P[tr], d[tr], K, cap)
        assert f.fingerprint() == ref.partition.fingerprint()
        cnt = [int(np.sum(f.cell_class == c)) for c in range(K)]
        assert max(cnt) <= cap
        rich = [c for c in range(K) if np.sum(d[tr] == c) > 4 * cap]
        assert rich and all(cnt[c] > cap // 2 for c in rich), cnt
    a = files["assign.npz"]
    assert np.array_equal(a["f1"], KM.assign_fine(P1, T["d1"], f1)) and np.array_equal(a["f2"], KM.assign_fine(P2, T["d2"], f2))
    assert np.array_equal(np.bincount(a["f1"][tr], minlength=f1.F), f1.n)
    assert np.array_equal(np.bincount(a["f2"][tr], minlength=f2.F), f2.n)
    for c, sup in enumerate(rec["support"][1]):
        assert sup["cells"] == int(np.sum(f1.cell_class == c)) and sup["rows"] == int(f1.n[f1.cell_class == c].sum())
    assert rec["caps"] == {1: 32, 2: 128}


# ---------------------------------------------------------------- attackers (qpc.audit; role D) -- definitions only

def _mw_auc(y, p):
    """Mann-Whitney AUC with ties counted 1/2 (independent of sklearn); never flipped."""
    y, p = np.asarray(y), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def test_attacker_auc_orientation_ce_clip_and_null_threshold():
    """auc1 is the fixed-orientation AUC of P(S = 1) (an anti-informative reader stays below 0.5, never flipped or
    clamped), ce1 clips at 1e-12, and the control null threshold is 0.5 + 3.5 x the Mann-Whitney null SD
    sqrt((n0 + n1 + 1) / (12 n0 n1)) on held-out half B; a noisy-S plant (20% re-drawn) has perfect-reader AUC 0.9."""
    AU = _qpc("audit")
    from dpc import audit as DA
    from smf import audit as SA
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    good = y + rng.normal(0, 0.8, 500)
    for p in (good, -good, np.round(good, 1)):
        assert abs(DA.auc1(y, p) - _mw_auc(y, p)) < 1e-12
    assert DA.auc1(y, -good) < 0.5
    p = np.clip(rng.random(500), 0, 1)
    p[:3] = [0.0, 1.0, 0.5]
    pt = np.where(y == 1, p, 1 - p)
    assert abs(DA.ce1(y, p) + np.mean(np.log(np.clip(pt, 1e-12, 1)))) < 1e-12
    assert SA.NULL_Z == 3.5 and SA.PLANT_MIN == 0.75 and AU.NULL_Z == 3.5 and AU.PLANT_MIN == 0.75
    for n0, n1 in ((300, 700), (512, 605)):
        yy = np.r_[np.zeros(n0, int), np.ones(n1, int)]
        assert abs(SA.null_sd(yy) - math.sqrt((n0 + n1 + 1) / (12 * n0 * n1))) < 1e-15
    S = rng.integers(0, 2, 200000)
    noisy = SA.noisy_sex(S, SA.CONTROL_SEED)
    assert abs(_mw_auc(S[:20000], noisy[:20000]) - 0.9) < 0.01


def ref_composed(own, pols, tie=1e-12):
    """Independent transcription of the composed source rule (PROTOCOL section 10; audit docstring): per family and
    view, the AUC winner is the first bank (own, then policies in scored_ids order) with the highest SEED-0 selected
    AUC (strictly better by > 1e-12 to replace), the CE winner likewise on the lowest seed-0 CE, separately; the
    reported value is the winning bank's seed 0-2 mean; the decisions family composes only with the class-only code."""
    out, freeze = {}, set()
    for fam, src in own.items():
        banks = [("source", src)] + [(c, r) for c, r in pols if fam != "decisions" or "|CLASS|" in c]
        e = {"auc": {}, "ce": {}, "winner": {}, "ce_winner": {}}
        for w in ("v1", "v2", "pair"):
            ia = max(range(len(banks)), key=lambda j: (banks[j][1]["auc_seed0"][w], -j))
            ic = min(range(len(banks)), key=lambda j: (banks[j][1]["ce_seed0"][w], j))
            e["auc"][w], e["winner"][w] = banks[ia][1]["auc"][w], banks[ia][0]
            e["ce"][w], e["ce_winner"][w] = banks[ic][1]["ce"][w], banks[ic][0]
            freeze |= {banks[ia][0], banks[ic][0]} - {"source"}
        out[fam] = e
    return out, freeze


def test_composed_source_bank_winner_rule_matches_transcription(monkeypatch):
    """qpc.audit.composed_source_bank on synthetic bank records (closure and record loading monkeypatched): the
    AUC and CE winners, reported values and the freeze list equal the independent transcription on 40 random banks,
    including banks where a policy's seed-0 value beats the source while its 3-seed mean does not (selection is on
    the seed-0 bank, the report is the winner's seed mean) and exact seed-0 ties (the earlier bank keeps it)."""
    AU = _qpc("audit")
    RUN = _qpc("run")
    fams = ("interface", "complete", "scores", "probs", "decisions")
    cids = ["U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|LOCAL|i8o64|l0.1", "U|JOINT|i8o64|l1", "U|CLASS|i1o1"]
    meta = {"sel_row_id_sha256": "a", "fit_row_id_sha256": "b", "slate": "final", "attacker_seeds": [0, 1, 2]}

    def rec(rng):
        r = {"auc_seed0": {}, "ce_seed0": {}, "auc": {}, "ce": {}, "selected": {}, "ce_selected": {}, **meta}
        for w in ("v1", "v2", "pair"):
            a0 = float(np.round(rng.uniform(0.6, 0.9), 3))               # rounded: exact ties occur
            r["auc_seed0"][w], r["auc"][w] = a0, a0 - float(rng.uniform(0, 0.02))
            c0 = float(np.round(rng.uniform(0.4, 0.7), 3))
            r["ce_seed0"][w], r["ce"][w] = c0, c0 + float(rng.uniform(0, 0.02))
            r["selected"][w], r["ce_selected"][w] = f"att{rng.integers(9)}", f"att{rng.integers(9)}"
        return r
    for trial in range(40):
        rng = np.random.default_rng(trial)
        store = {RUN.unit_for(0, c): {"recovery": rec(rng), "seed": 0} for c in cids}
        own = {f: rec(rng) for f in fams}
        monkeypatch.setattr(AU, "_closure", lambda k, t, e, u=None: {"ok": True, "expected": len(e)})
        monkeypatch.setattr(AU, "load_inner", lambda name, units_dir=None: (store[name], {}))
        got = AU.composed_source_bank(0, "U", own, None, policy_cids=cids)
        ref, freeze = ref_composed(own, [(c, store[RUN.unit_for(0, c)]["recovery"]) for c in cids])
        for f in fams:
            for w in ("v1", "v2", "pair"):
                assert got[f]["winner"][w] == ref[f]["winner"][w], (trial, f, w)
                assert got[f]["ce_winner"][w] == ref[f]["ce_winner"][w], (trial, f, w)
                assert got[f]["auc"][w] == ref[f]["auc"][w] and got[f]["ce"][w] == ref[f]["ce"][w]
        assert set(got["_all"]["freeze"]) == freeze
        assert got["decisions"]["policies"] == ["U|CLASS|i1o1"]
    bad = dict(store[RUN.unit_for(0, cids[0])])
    bad["recovery"] = {**bad["recovery"], "slate": "inner"}
    store[RUN.unit_for(0, cids[0])] = bad
    with pytest.raises(SystemExit):                                     # a record from another slate is refused
        AU.composed_source_bank(0, "U", own, None, policy_cids=cids)


def test_tolerance_and_cap_stops_match_reference():
    """The two non-fixed-point stop rules, exercised deliberately (the registered RTOL = 1e-9 rarely binds on small
    fixtures): with a loose tolerance the 'relative_tolerance' stop (PATIENCE = 3 successive passes, then one final
    update + assignment) and with a small round cap the 'cap' stop (final update + assignment, converged False) give
    exactly the reference's assignment, returned pass, stop reason and rounds. Also pins the registered constants."""
    KM = _qpc("kmeans")
    assert KM.RTOL == 1e-9 and KM.PATIENCE == 3 and KM.ROUNDS == 200 and KM.SELECT_TOL == 1e-12
    seen = set()
    for seed in range(4):
        P = binary_continuum(3000, seed)
        d = P.argmax(1)
        for c in (0, 1):
            Pc = P[d == c]
            for start in KM.STARTS:
                kind, sd = KM.parse_start(start)
                k = int(min(12, np.unique(Pc, axis=0).shape[0], Pc.shape[0]))
                C0 = ref_source_init(Pc, c, k) if kind == "source" else ref_kpp(Pc, c, k, sd, 2)[0]
                for rtol, rounds in ((3e-4, 200), (1e-9, 6)):
                    Q, a, rec = KM.kmeans_class(Pc, c, 12, start, rounds, "qpc", rtol=rtol)
                    ref = ref_qpc_kmeans(Pc, c, C0, rounds=rounds, rtol=rtol, patience=3)
                    assert rec["stop_reason"] == ref["reason"] and rec["rounds_used"] == ref["rounds_used"]
                    assert np.array_equal(a, ref["a"]) and rec["returned_pass"] == ref["returned_pass"]
                    assert rec["objective"] == min(rec["objective_trajectory"])
                    if rec["stop_reason"] == "cap":
                        assert rec["converged"] is False and rec["rounds_used"] == rounds
                    seen.add(rec["stop_reason"])
    assert {"relative_tolerance", "cap"} <= seen, seen


# ---------------------------------------------------------------- inference end to end vs an independent bootstrap

def _w_auc(y, s, w):
    """Weighted Mann-Whitney AUC of score s for y == 1 with integer weights w (ties 1/2); never flipped."""
    pos, neg = y == 1, y == 0
    gt = (s[pos][:, None] > s[neg][None, :]).astype(float)
    eq = (s[pos][:, None] == s[neg][None, :]).astype(float)
    num = w[pos] @ (gt + 0.5 * eq) @ w[neg]
    den = w[pos].sum() * w[neg].sum()
    return num / den if den > 0 else np.nan


def ref_inference(preds, roles, B, seed, z):
    """Independent 37-slot inference: endpoint = mean over model seeds of the per-seed paired statistic; recovery =
    mean over attacker seeds of the fixed-orientation AUC of P(S=1); multinomial bootstrap of exact-record groups
    (np.unique order), one sequential draw stream shared by every statistic; SE ddof 1; bounds point -+ z SE;
    strict decisions; a role that is not a NOMINEE makes the slot DESCRIPTIVE_ONLY."""
    FAM = _qpc("family")
    p0 = preds[(0, "SRC|U")]
    units, idx = np.unique(p0["assess_unit"], return_inverse=True)
    G = len(units)
    sex = p0["sex"]
    ys = {0: p0["y_income"], 1: p0["y_occ"]}
    const = p0["const_class"]

    def stats(w):
        def R(k, lab, view):
            P = preds[(k, lab)][f"P_auc_{view}"]
            return np.mean([_w_auc(sex, P[a][:, 1], w) for a in range(3)])

        def mean(x):
            return float(np.dot(w, x) / w.sum())

        def acc(k, lab, j):
            return mean(preds[(k, lab)][f"hard{j + 1}"] == ys[j])

        def loss(k, lab, j, kind):
            P = preds[(k, lab)][f"prob{j + 1}"]
            y = ys[j]
            if kind == "ll":
                return mean(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)))
            return mean(((P - np.eye(P.shape[1])[y]) ** 2).sum(1))
        out = {}
        for e in FAM.PRIMARY:
            nom = roles[e["nominee"]]
            ref = roles.get(e.get("ref")) if "ref" in e else None
            per = []
            for k in (0, 1, 2):
                j = e.get("task")
                if e["kind"] == "coalition":
                    per.append(R(k, ref, "pair") - R(k, nom, "pair"))
                elif e["kind"] == "local":
                    per.append(R(k, nom, e["view"]) - R(k, ref, e["view"]))
                elif e["kind"] == "acc":
                    per.append(acc(k, nom, j) - acc(k, "SRC|U", j))
                elif e["kind"] in ("logloss", "brier"):
                    kd = "ll" if e["kind"] == "logloss" else "br"
                    per.append(loss(k, nom, j, kd) - loss(k, "SRC|U", j, kd))
                else:
                    per.append(acc(k, nom, j) - 0.8 * acc(k, "SRC|U", j) - 0.2 * mean(ys[j] == const[j]))
            out[e["id"]] = float(np.mean(per))
        return out
    pts = stats(np.ones(len(sex)))
    rng = np.random.default_rng(seed)
    reps = {i: [] for i in pts}
    for _ in range(B):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[idx].astype(float)
        for i, v in stats(w).items():
            reps[i].append(v)
    res = {}
    for e in FAM.PRIMARY:
        r = np.array(reps[e["id"]])
        se = float(np.std(r, ddof=1))
        lo, hi = pts[e["id"]] - z * se, pts[e["id"]] + z * se
        dec = ("PASS" if lo > e["target"] else "NOT_ESTABLISHED") if e["side"] == "lower>" else \
            ("PASS" if hi < e["target"] else "NOT_ESTABLISHED")
        res[e["id"]] = {"point": pts[e["id"]], "se": se, "lower": lo, "upper": hi, "decision": dec}
    return res


def test_inference_end_to_end_against_independent_bootstrap(tmp_path, monkeypatch):
    """qpc.infer.main on a synthetic assessment with duplicate exact-record groups: every one of the 37 primary
    slots (point, SE, bounds, decision, DESCRIPTIVE_ONLY override) equals the independent bootstrap above with the
    same B, seed 20261007 and z = 3.2048452050105634; arms with different SEX or labels are refused."""
    from jcv.finalize import save_unit
    RUN = _qpc("run")
    INF = _qpc("infer")
    FAM = _qpc("family")
    monkeypatch.setattr(RUN, "UNITS", tmp_path / "units")
    monkeypatch.setattr(RUN, "RUN", tmp_path)
    monkeypatch.setattr(RUN, "PKG", tmp_path / "pkg")
    (tmp_path / "pkg").mkdir()
    B = 80
    monkeypatch.setattr(FAM, "B", B)
    rng = np.random.default_rng(1)
    n = 240
    groups = np.repeat(np.arange(200), [2] * 40 + [1] * 160)                # 40 duplicated exact-record groups
    sex = rng.integers(0, 2, 200)[groups]
    yI, yO = rng.integers(0, 2, 200)[groups], rng.integers(0, 6, 200)[groups]
    labs = {"SRC|U": 1.2, "U|JOINT|i8o64|l0.1": 0.5, "U|FINE-TASK|i8o64": 0.9, "U|LOCAL|i8o64|l1": 0.7,
            "U|DIRECT-TASK|i8o64": 1.0}
    preds = {}
    for k in (0, 1, 2):
        for li, (lab, st) in enumerate(labs.items()):
            g = np.random.default_rng([k, li])
            p = {"assess_row_id": np.arange(n) * 3, "assess_unit": groups, "sex": sex, "y_income": yI, "y_occ": yO,
                 "const_class": np.array([0, 4])}
            for w, s_ in (("v1", 0.4 * st), ("v2", 0.8 * st), ("pair", st)):
                P = np.empty((3, n, 2))
                for a in range(3):
                    zz = s_ * (2 * sex - 1) + g.normal(size=n)
                    q = 1 / (1 + np.exp(-zz))
                    P[a] = np.stack([1 - q, q], 1)
                p[f"P_auc_{w}"] = P
            p1 = g.dirichlet([2, 2], n)
            p2 = g.dirichlet([1] * 6, n)
            p.update(prob1=p1, prob2=p2, hard1=p1.argmax(1), hard2=p2.argmax(1))
            preds[(k, lab)] = p
            save_unit(RUN.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {})
    st = {"J*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.1"},
          "C_rate": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "C_global": {"status": "NOMINEE", "config": "SRC|U"}, "T*": {"status": "NOMINEE", "config": "SRC|U"},
          "P*": {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": "U|LOCAL|i8o64|l1"},
          "Q*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    roles = {x: s.get("config") or s.get("descriptive_config") for x, s in st.items()}
    lock = {"statuses": st, "resolved": roles, "seeds": {str(k): {"score": {lab: {} for lab in labs}} for k in (0, 1, 2)},
            "stage_a_valid": True, "gate_met": True, "technical_validity": {"ok": True}}
    lp = tmp_path / "EL.json"
    lp.write_text(json.dumps(lock))
    out = INF.main(["--evaluation-lock", str(lp)], check_prior=False)
    ref = ref_inference(preds, roles, B, 20261007, 3.2048452050105634)
    assert [e["id"] for e in out["primary"]] == [f"P{i:02d}" for i in range(1, 38)]
    for e in out["primary"]:
        r = ref[e["id"]]
        assert e["z"] == 3.2048452050105634
        assert abs(e["point"] - r["point"]) <= 1e-12, e["id"]
        assert abs(e["se"] - r["se"]) <= 1e-9 * max(1.0, r["se"]), e["id"]
        assert abs(e["lower"] - r["lower"]) <= 1e-9 and abs(e["upper"] - r["upper"]) <= 1e-9
        if e["claim"] == "C":
            assert e["decision"] == "DESCRIPTIVE_ONLY" and e["decision_numeric"] == r["decision"]
        else:
            assert e["decision"] == r["decision"], (e["id"], e["decision"], r)
    assert out["claim_status"]["C"] == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    # an arm whose SEX differs is refused
    bad = dict(preds[(1, "U|FINE-TASK|i8o64")])
    bad["sex"] = 1 - bad["sex"]
    save_unit(RUN.U("outer__s1__U_FINE-TASK_i8o64__x"), {"preds.npz": lambda q: np.savez(q, **bad)}, {})
    import shutil
    shutil.rmtree(RUN.U("outer__s1__U_FINE-TASK_i8o64"))
    shutil.move(str(RUN.U("outer__s1__U_FINE-TASK_i8o64__x")), str(RUN.U("outer__s1__U_FINE-TASK_i8o64")))
    with pytest.raises(AssertionError):
        INF.main(["--evaluation-lock", str(lp)], check_prior=False)
