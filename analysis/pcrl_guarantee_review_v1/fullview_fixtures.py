"""Fixtures for the full-view guarantee review.

Notation: S sensitive, Y authorised task, H the recipient's actual service view, B=b(H) a finite bin,
T the encoder code, Z the sampled release, Q(z|t) the public kernel. Exact fixtures use rational
tables and return information in the basis {log 2, log 3}. The "numerical" fixtures are spot checks
over random finite instances: coverage evidence, not proofs. Proofs are in FULL_VIEW_GUARANTEE_REVIEW.md.

Run: python analysis/pcrl_guarantee_review_v1/fullview_fixtures.py [--json OUT]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.pcrl_claims_foundation_v1.exact_fixtures import mi, mi_float  # noqa: E402


def ln(v):
    return mi_float(v)


# ------------------------------------------------------------------ numeric helpers

def kl(p, q):
    s = 0.0
    for a, b in zip(p, q):
        if a > 0:
            if b <= 0:
                return math.inf
            s += a * math.log(a / b)
    return s


def cmi_float(joint, a, b, c):
    def marg(idx):
        m = {}
        for k, p in joint.items():
            kk = tuple(k[i] for i in idx)
            m[kk] = m.get(kk, 0.0) + p
        return m
    pabc, pac, pbc, pc = marg(a + b + c), marg(a + c), marg(b + c), marg(c)
    na, nb = len(a), len(b)
    tot = 0.0
    for k, p in pabc.items():
        if p <= 0:
            continue
        ka, kb, kc = k[:na], k[na:na + nb], k[na + nb:]
        tot += p * math.log(p * pc[kc] / (pac[ka + kc] * pbc[kb + kc]))
    return tot


def rand_simplex(rng, n, sparse=0.0):
    v = [0.0 if rng.random() < sparse else rng.random() for _ in range(n)]
    if sum(v) == 0:
        v[rng.randrange(n)] = 1.0
    s = sum(v)
    return [x / s for x in v]


def blahut_arimoto(Q, iters=4000):
    nt, nz = len(Q), len(Q[0])
    p = [1.0 / nt] * nt
    for _ in range(iters):
        q = [sum(p[t] * Q[t][z] for t in range(nt)) for z in range(nz)]
        d = [kl(Q[t], q) for t in range(nt)]
        w = [p[t] * math.exp(d[t]) for t in range(nt)]
        s = sum(w)
        p = [x / s for x in w]
    q = [sum(p[t] * Q[t][z] for t in range(nt)) for z in range(nz)]
    cap = sum(p[t] * kl(Q[t], q) for t in range(nt))
    radius = max(kl(Q[t], q) for t in range(nt))
    return cap, radius, q, p


# ------------------------------------------------------------------ G1 chain rule

def g1_chain_rule():
    """I(S;Z|H) = I(S;Z|B) + I(H;Z|S,B) - I(H;Z|B), B=b(H). Exact XOR, exact overstatement,
    numerical random tables."""
    # (a) XOR under-statement: S,H fair independent, Z=S xor H, B constant
    xor = {(s, h, 0, s ^ h): F(1, 4) for s in (0, 1) for h in (0, 1)}  # (S,H,B,Z)
    lhs = mi(xor, [0], [3], [1])
    t1, t2, t3 = mi(xor, [0], [3], [2]), mi(xor, [1], [3], [0, 2]), mi(xor, [1], [3], [2])
    assert lhs == {"log2": 1, "log3": 0} and t1["log2"] == 0 and t2["log2"] == 1 and t3["log2"] == 0
    # (b) over-statement: S = H, Z = S, B constant: binned CMI log 2, full-view CMI 0
    over = {(s, s, 0, s): F(1, 2) for s in (0, 1)}
    lhs_o = mi(over, [0], [3], [1])
    t1o, t2o, t3o = mi(over, [0], [3], [2]), mi(over, [1], [3], [0, 2]), mi(over, [1], [3], [2])
    assert lhs_o["log2"] == 0 and t1o["log2"] == 1 and t2o["log2"] == 0 and t3o["log2"] == 1
    # (c) numerical: random finite (S,H,Z) with B = H mod 2, 300 tables
    rng = random.Random(1)
    worst = 0.0
    for _ in range(300):
        nh, ns, nz = rng.randint(2, 5), rng.randint(2, 3), rng.randint(2, 4)
        w = {(s, h, h % 2, z): rng.random() for s in range(ns) for h in range(nh) for z in range(nz)}
        tot = sum(w.values())
        j = {k: v / tot for k, v in w.items()}
        L = cmi_float(j, (0,), (3,), (1,))
        R = cmi_float(j, (0,), (3,), (2,)) + cmi_float(j, (1,), (3,), (0, 2)) - cmi_float(j, (1,), (3,), (2,))
        worst = max(worst, abs(L - R))
    assert worst < 1e-12
    return {"xor": {"I(S;Z|H)": "log2", "I(S;Z|B)": "0", "I(H;Z|S,B)": "log2", "I(H;Z|B)": "0"},
            "overstatement": {"I(S;Z|H)": "0", "I(S;Z|B)": "log2", "I(H;Z|S,B)": "0", "I(H;Z|B)": "log2"},
            "random_tables": 300, "max_abs_identity_error": worst}


# ------------------------------------------------------------------ G2 row-radius bound

def g2_row_radius_bound(trials: int = 400):
    """For (S,W) -> T -> Z with Z ~ Q(.|T): I(S;Z|W) <= max_t KL(Q_t || r) for every r.
    Checked for random laws, random W (any side information), r = uniform and r = BA output."""
    rng = random.Random(2)
    viol = 0
    worst_ratio = 0.0
    for _ in range(trials):
        nt, nz, ns, nw = rng.randint(2, 6), rng.randint(2, 5), rng.randint(2, 3), rng.randint(1, 4)
        Q = [rand_simplex(rng, nz) for _ in range(nt)]
        cap, radius, qstar, _ = blahut_arimoto(Q, 600)
        ru = [1.0 / nz] * nz
        rad_u = max(kl(Q[t], ru) for t in range(nt))
        # arbitrary law of (S, W, T); adversarial variant ties S to T
        law = {}
        for s in range(ns):
            for w in range(nw):
                pt = rand_simplex(rng, nt, sparse=0.3)
                psw = rng.random()
                for t in range(nt):
                    law[(s, w, t)] = psw * pt[t]
        tot = sum(law.values())
        j = {}
        for (s, w, t), p in law.items():
            for z in range(nz):
                j[(s, w, z)] = j.get((s, w, z), 0.0) + p / tot * Q[t][z]
        I = cmi_float(j, (0,), (2,), (1,))
        if I > radius + 1e-9 or I > rad_u + 1e-9:
            viol += 1
        worst_ratio = max(worst_ratio, I / max(radius, 1e-15))
    assert viol == 0
    # exact tightness: BSC(1/4), S = T uniform, W constant: I = capacity = radius
    Qb = [[F(3, 4), F(1, 4)], [F(1, 4), F(3, 4)]]
    j = {(s, 0, z): F(1, 2) * Qb[s][z] for s in (0, 1) for z in (0, 1)}
    I_exact = mi(j, [0], [2], [1])
    r = [F(1, 2), F(1, 2)]
    # KL(Q_t || r) = 3/4 log(3/2) + 1/4 log(1/2) = 3/4 log3 - log2 (exactly, both rows)
    rad_exact = {"log2": F(-1), "log3": F(3, 4)}
    assert I_exact == rad_exact
    # support condition: r with a zero where a row has mass makes the radius infinite
    assert kl([0.5, 0.5], [1.0, 0.0]) == math.inf
    return {"random_trials": trials, "violations": viol, "max_I_over_radius": worst_ratio,
            "tightness_bsc_quarter": {"I(S;Z|W)": "3/4 log3 - log2", "radius": "3/4 log3 - log2",
                                      "nats": ln(rad_exact)}}


def g2b_public_state_dependence():
    """Q may depend on a public bin B=b(H_A). The row bound must hold within each public state.
    Rows constant within a state carry nothing about T given H, even if they differ across states."""
    # H in {0,1} = its own bin; S = H (fully public); T independent noise bit; Q_b(z|t) = 1[z=b]
    j = {}
    for h in (0, 1):
        for t in (0, 1):
            j[(h, h, t, h)] = j.get((h, h, t, h), F(0)) + F(1, 4)  # (S,H,T,Z) with Z=b(H)
    cond = mi(j, [0], [3], [1])
    uncond = mi(j, [0], [3])
    assert cond["log2"] == 0 and uncond["log2"] == 1
    return {"I(S;Z|H)": "0 (per-state radius 0)", "I(S;Z)": "log2",
            "lesson": "state-dependent kernels are bounded only for views that contain the state; the local encoder may use b(H_A) but not H_B"}


# ------------------------------------------------------------------ G3 utility ceiling

def g3_utility_ceiling():
    """The same radius caps I(Y;Z|H), which equals the Bayes log-loss gain of adding Z to H."""
    # exact: H constant, Y = T fair bit, Q = BSC(1/4); Bayes gain = H(Y) - H(Y|Z) = I(Y;Z)
    Qb = [[F(3, 4), F(1, 4)], [F(1, 4), F(3, 4)]]
    j = {(y, 0, z): F(1, 2) * Qb[y][z] for y in (0, 1) for z in (0, 1)}
    I = mi(j, [0], [2], [1])
    # Bayes posterior P(Y=z|Z=z) = 3/4, log loss = h(1/4); prior loss = log 2
    hb = {"log2": F(2), "log3": F(-3, 4)}  # h(1/4) = 2 log2 - (3/4) log3 nats
    bayes_gain = {"log2": F(1) - hb["log2"], "log3": -hb["log3"]}
    assert I == bayes_gain
    return {"I(Y;Z|H)": "3/4 log3 - log2", "bayes_log_loss_gain": "log2 - h(1/4) = 3/4 log3 - log2",
            "nats": ln(I), "reading": "a kappa-radius release can add at most kappa nats of Bayes task log-loss over H"}


# ------------------------------------------------------------------ G4 SDPI alternative

def dobrushin(Q):
    return max(0.5 * sum(abs(a - b) for a, b in zip(Q[t], Q[u])) for t in range(len(Q)) for u in range(len(Q)))


def g4_sdpi_bound(trials: int = 300):
    """Attribute-specific distribution-free bound: I(S;Z|H) <= eta_TV(Q) * I(S;T|H) <= eta_TV(Q) H(S|H).
    Spot-checked numerically; compared with the radius bound."""
    rng = random.Random(4)
    viol = 0
    sdpi_tighter = radius_tighter = 0
    for _ in range(trials):
        nt, nz, nh = rng.randint(2, 6), rng.randint(2, 5), rng.randint(1, 3)
        Q = [rand_simplex(rng, nz) for _ in range(nt)]
        eta = dobrushin(Q)
        _, radius, _, _ = blahut_arimoto(Q, 400)
        law = {}
        for s in (0, 1):
            for h in range(nh):
                pt = rand_simplex(rng, nt, sparse=0.3)
                psh = rng.random()
                for t in range(nt):
                    law[(s, h, t)] = psh * pt[t]
        tot = sum(law.values())
        law = {k: v / tot for k, v in law.items()}
        j = {}
        for (s, h, t), p in law.items():
            for z in range(nz):
                j[(s, h, z)] = j.get((s, h, z), 0.0) + p * Q[t][z]
        I = cmi_float(j, (0,), (2,), (1,))
        IST = cmi_float(law, (0,), (2,), (1,))
        if I > eta * IST + 1e-10:
            viol += 1
        b_sdpi, b_rad = eta * math.log(2), radius
        sdpi_tighter += b_sdpi < b_rad
        radius_tighter += b_rad < b_sdpi
    assert viol == 0
    # deterministic kernels (D17-like): eta = 1 and radius = log(#used actions): vacuous for SEX
    D = [[1.0 if z == t else 0.0 for z in range(17)] for t in range(17)]
    return {"trials": trials, "violations": viol, "sdpi_log2_bound_tighter": sdpi_tighter,
            "radius_bound_tighter": radius_tighter,
            "deterministic_17_action_kernel": {"eta_TV": dobrushin(D), "radius_nats": math.log(17),
                                               "log2_ceiling_SEX": math.log(2), "log9_ceiling_RAC1P": math.log(9)}}


# ------------------------------------------------------------------ G5 envelope / robust constraint

def binary_mi(prior, rows):
    """I(S;Z) for binary S with P(Z|S=s) = rows[s]."""
    j = {(s, 0, z): (prior if s else 1 - prior) * rows[s][z] for s in (0, 1) for z in range(len(rows[0]))}
    return cmi_float(j, (0,), (2,), (1,))


def g5_envelope():
    """(a) sup over a law envelope of I_P(S;Z|H) is convex in Q (pointwise sup of convex);
    (b) for fixed Q it is a convex function of P(T|S,h), so the worst case sits at a vertex and
        interior sampling under-estimates it; (c) a bin-level radius of 0 need not cover H-level laws."""
    rng = random.Random(5)
    nt = 17
    Q = [[1.0 if z == t else 0.0 for z in range(nt)] for t in range(nt)]  # deterministic, D17-like
    m = [1.0 / nt] * nt
    eps = 0.05
    # TV-ball vertices around m for each s: move eps mass from atom a to atom b
    def move(a, b):
        v = list(m)
        v[a] -= eps
        v[b] += eps
        return v
    vertices = [move(a, b) for a in range(nt) for b in range(nt) if a != b]
    # exact worst case: MI is convex in each conditional row, so the max over the product of the two
    # TV balls is attained at a pair of vertices; enumerate all vertex pairs
    worst_vertex = max(binary_mi(0.5, [v0, v1]) for v0 in vertices for v1 in vertices)
    # sampled interior scenarios (random convex combinations of vertices) under-estimate it
    sampled = 0.0
    for _ in range(2000):
        def interior():
            w = rand_simplex(rng, 5)
            vs = [vertices[rng.randrange(len(vertices))] for _ in range(5)]
            return [sum(wi * v[k] for wi, v in zip(w, vs)) for k in range(nt)]
        sampled = max(sampled, binary_mi(0.5, [interior(), interior()]))
    assert sampled < worst_vertex
    # eps needed for a 0.01-nat worst case with a deterministic 17-action kernel (bisection)
    lo, hi = 0.0, 1.0 / nt
    for _ in range(60):
        mid = (lo + hi) / 2
        v0, v1 = list(m), list(m)
        v0[0] -= mid; v0[1] += mid; v1[0] += mid; v1[1] -= mid
        if binary_mi(0.5, [v0, v1]) > 0.01:
            hi = mid
        else:
            lo = mid
    # (a) convexity in Q of the worst case over a finite law set: midpoint check
    laws = [[vertices[rng.randrange(len(vertices))], vertices[rng.randrange(len(vertices))]] for _ in range(20)]
    def worst(Qk):
        return max(binary_mi(0.5, [[sum(p[t] * Qk[t][z] for t in range(nt)) for z in range(nt)] for p in law])
                   for law in laws)
    gap = 0.0
    for _ in range(30):
        Q1 = [rand_simplex(rng, nt) for _ in range(nt)]
        Q2 = [rand_simplex(rng, nt) for _ in range(nt)]
        Qm = [[(a + b) / 2 for a, b in zip(r1, r2)] for r1, r2 in zip(Q1, Q2)]
        gap = max(gap, worst(Qm) - (worst(Q1) + worst(Q2)) / 2)
    assert gap <= 1e-12
    # (c) bin-level radius 0 but H-level law deterministic: S,H fair bits, one bin, T = S xor H
    j = {(s, h, 0, s ^ h): F(1, 4) for s in (0, 1) for h in (0, 1)}  # (S,H,B,T); Q = identity so Z = T
    at_bin = mi(j, [0], [3], [2])
    at_h = mi(j, [0], [3], [1])
    assert at_bin["log2"] == 0 and at_h["log2"] == 1
    return {"deterministic_kernel_tv_radius": eps, "exact_vertex_worst_case_nats": worst_vertex,
            "max_over_2000_interior_samples_nats": sampled,
            "tv_radius_for_0.01_nats_worst_case": lo, "midpoint_convexity_violation": gap,
            "bin_level_vs_h_level": {"I(S;T|B) with P(T|S,B) calibrated exactly": "0", "I(S;T|H)": "log2"}}


def g6_unrestricted_impossibility():
    """With an unrestricted law envelope, any kernel with two reachable distinct rows has a law with
    I(S;Z|H) > 0: map S=0 to t and S=1 to t'. The worst case equals the binary-input channel MI."""
    Q = [[F(3, 4), F(1, 4)], [F(1, 4), F(3, 4)]]
    j = {(s, 0, z): F(1, 2) * Q[s][z] for s in (0, 1) for z in (0, 1)}
    I = mi(j, [0], [2], [1])
    assert I["log3"] != 0 or I["log2"] != 0
    return {"worst_case_I(S;Z|H)": "3/4 log3 - log2 > 0", "nats": ln(I),
            "consequence": "a nonzero useful kernel cannot be certified for every law; restrict the envelope or bound all information (radius)"}


def g7_constant_channel():
    """Correctness fixture only: a constant kernel has radius 0, so I(S;Z|H)=I(Y;Z|H)=0 (no utility)."""
    Q = [[F(1, 3)] * 3 for _ in range(4)]
    r = [F(1, 3)] * 3
    assert all(q == rr for row in Q for q, rr in zip(row, r))
    return {"radius": "0", "I(S;Z|H)": "0", "I(Y;Z|H)": "0", "use": "correctness fixture, not a useful method"}


def g8_conditional_homogeneity():
    """If T is independent of H given (S, B) then Z is too, I(H;Z|S,B)=0, and the chain rule gives
    I(S;Z|H) = I(S;Z|B) - I(H;Z|B) <= I(S;Z|B): the binned CMI is then an UPPER bound.
    Exact check on a table satisfying the assumption, and on one that violates it (T depends on H
    within the bin, as the residual-logit code does by construction)."""
    # satisfies: H in {0,1,2,3}, B = H // 2, S depends on H, T depends on (S, B) only
    ps_h = {0: F(1, 5), 1: F(2, 5), 2: F(3, 5), 3: F(4, 5)}
    pt_sb = {(0, 0): F(1, 4), (1, 0): F(3, 4), (0, 1): F(1, 2), (1, 1): F(1, 3)}
    Q = [[F(2, 3), F(1, 3)], [F(1, 6), F(5, 6)]]
    def table(pt):
        j = {}
        for h in range(4):
            b = h // 2
            for s in (0, 1):
                psh = F(1, 4) * (ps_h[h] if s else 1 - ps_h[h])
                p1 = pt(s, b, h)
                for t in (0, 1):
                    ptt = p1 if t else 1 - p1
                    for z in (0, 1):
                        key = (s, h, b, z)
                        j[key] = j.get(key, F(0)) + psh * ptt * Q[t][z]
        return j
    ok = table(lambda s, b, h: pt_sb[(s, b)])
    full, binned = cmi_float(ok, (0,), (3,), (1,)), cmi_float(ok, (0,), (3,), (2,))
    within = cmi_float(ok, (1,), (3,), (0, 2))
    assert abs(within) < 1e-15 and full <= binned + 1e-15
    # violates: T depends on H inside the bin (T=1 more likely for odd H), S independent of H
    bad = table(lambda s, b, h: F(9, 10) if (h % 2) ^ s else F(1, 10))
    full_b, binned_b = cmi_float(bad, (0,), (3,), (1,)), cmi_float(bad, (0,), (3,), (2,))
    assert full_b > binned_b
    return {"assumption_holds": {"I(S;Z|H)": full, "I(S;Z|B)": binned, "I(H;Z|S,B)": within,
                                 "binned_is_upper_bound": True},
            "assumption_violated": {"I(S;Z|H)": full_b, "I(S;Z|B)": binned_b, "binned_is_upper_bound": False}}


FIXTURES = [g1_chain_rule, g2_row_radius_bound, g2b_public_state_dependence, g3_utility_ceiling,
            g4_sdpi_bound, g5_envelope, g6_unrestricted_impossibility, g7_constant_channel,
            g8_conditional_homogeneity]


def run_all():
    return {f.__name__: f() for f in FIXTURES}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    res = run_all()
    txt = json.dumps(res, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(txt + "\n")
    print(txt)
