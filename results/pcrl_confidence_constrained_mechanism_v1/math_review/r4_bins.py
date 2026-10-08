"""R4 -- bin feasibility. (a) NLL-only: sum_k M_k <= e^d is necessary AND sufficient; NLL+Class closed form.
(b) 3-member counterexamples (every pair admissible, whole bin not): NLL alone (K=3) and full G with one decision (K=6),
certified in exact rational arithmetic. (c) the NLL closed form is NOT sufficient for G (binary pair), certified by a
rational Brier dual vector. (d) Brier-only 3-member counterexample (K=3). (e) Helly: pairwise suffices for K=2, K-wise
in general. (f) the exact joint dual formula. Synthetic only. Run: python -P r4_bins.py (about 20 s)."""
import itertools
import math
import os
import sys
from fractions import Fraction as F

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(11)


def ratq(qf, d):
    """Rationalise a float q onto the simplex exactly (q_d absorbs the residual)."""
    q = [F(float(x)).limit_denominator(10 ** 12) for x in qf]
    q[d] = 1 - sum(q[k] for k in range(len(q)) if k != d)
    return q


def certify_pair_or_bin(Pr, d, qf):
    q = ratq(qf, d)
    res = [M.g_ok_rational(p, q, d) for p in Pr]
    return all(all(r) for r in res), q, res


def nll_sumM_exact(Pr):
    K = len(Pr[0])
    return sum(max(p[k] for p in Pr) for k in range(K))


print("== R4.a  NLL-only: sum M <= e^d  <=>  exists q in simplex with q >= e^-d p for all members ==")
bad = 0
for t in range(20000):
    K = int(rng.choice([2, 3, 6]))
    ctr = rng.dirichlet(np.ones(K))
    P = np.abs(ctr + rng.normal(0, 0.002, (int(rng.integers(2, 6)), K)))
    P = P / P.sum(1, keepdims=True)
    Mx = P.max(0)
    if Mx.sum() <= M.ED:
        r = rng.dirichlet(np.ones(K))
        q = M.EMD * Mx + (1 - M.EMD * Mx.sum()) * r      # every such q works (sufficiency)
        bad += int(not all(np.all(q >= M.EMD * p - 1e-17) for p in P))
print(f"sufficiency construction failures: {bad}  (necessity: summing q_k >= e^-d M_k gives 1 >= e^-d sum M)")
print("NLL+Class closed form: admissible iff sum M < e^d, or sum M = e^d and M_d > max_{k!=d} M_k (q = e^-d M + slack"
      " e_d); PROVED in MATH_REVIEW R4.")

print("\n== R4.b1  NLL alone, K=3: every pair NLL-feasible, the triple is not (exact rationals) ==")
s = F(12, 10000)
third = F(1, 3)
P3 = [[third + 2 * s if k == i else third - s for k in range(3)] for i in range(3)]
for a, b in itertools.combinations(range(3), 2):
    SM = nll_sumM_exact([P3[a], P3[b]])
    print(f"pair ({a},{b}): sum M = {float(SM):.6f} <= e^d ({float(M.ED_LO):.7f})? {SM <= M.ED_LO}")
SM = nll_sumM_exact(P3)
print(f"triple: sum M = {float(SM):.6f} > e^d? {SM > M.ED_UP}  -> NLL-infeasible (certified by the closed form)")

print("\n== R4.b2  full G (NLL + Brier + strict Class), K=6, ONE decision: pairs admissible, triple infeasible ==")
s = F(2, 1000)
bb = F(1, 10)
P6 = []
for i in (1, 2, 3):
    p = [F(1, 2), bb, bb, bb, bb - s, bb - s]
    p[i] = bb + 2 * s
    P6.append(p)
assert all(sum(p) == 1 for p in P6)
d = 0
for a, b in itertools.combinations(range(3), 2):
    Pf = np.array([[float(x) for x in P6[a]], [float(x) for x in P6[b]]])
    val, qf, ok = M.joint_minimax(Pf, d)
    fl = [M.g_ok(p, qf, d) for p in Pf]
    good0, _, res0 = certify_pair_or_bin([P6[a], P6[b]], d, qf)
    # the solver output is NLL-tight (q_k = e^-d M_k); move 1e-9 relative inside the NLL box, q_d absorbs the rest
    qi = qf.copy()
    Mx = Pf.max(0)
    for k in range(6):
        if k != d:
            qi[k] = max(qi[k], M.EMD * Mx[k] * (1 + 1e-9))
    qi[d] = 1 - (qi.sum() - qi[d])
    good, q, res = certify_pair_or_bin([P6[a], P6[b]], d, qi)
    print(f"pair ({a},{b}): sum M = {float(nll_sumM_exact([P6[a], P6[b]])):.4f}; solver max Brier excess {val:.6f};"
          f" float64 G (as written) {fl}; EXACT check of the same q {res0} -> {good0}  [boundary issue, see R1.d]")
    print(f"    after a 1e-9 interior shift: exact (NLL, Brier, Class) per member {res} -> admissible {good};"
          f" q = {[f'{float(x):.9f}' for x in q]}")
SM = nll_sumM_exact(P6)
print(f"triple: sum M = {float(SM):.4f} > e^d? {SM > M.ED_UP} -> infeasible (closed-form certificate)")

print("\n== R4.c  the NLL closed form is NOT sufficient for G: a binary pair (income-like) ==")
pa = [F(52, 100), F(48, 100)]
pb = [F(5245, 10000), F(4755, 10000)]
SM = nll_sumM_exact([pa, pb])
print(f"sum M = {float(SM):.5f} <= e^d: {SM <= M.ED_LO}  (NLL+Class closed form: admissible)")
# Brier-only dual certificate: V >= 1 - ||c||^2 - c.beta for EVERY c in the simplex; beta_y = min_i Brier(p_i, y)
E = [[F(1), F(0)], [F(0), F(1)]]
beta = [min(sum((p[k] - E[y][k]) ** 2 for k in range(2)) for p in (pa, pb)) for y in range(2)]
Vf, cf, _ = M.brier_minimax(np.array([[float(x) for x in pa], [float(x) for x in pb]]))
c = [F(float(cf[0])).limit_denominator(10 ** 9)]
c.append(1 - c[0])
lb = 1 - sum(x * x for x in c) - sum(cy * by for cy, by in zip(c, beta))
print(f"rational dual c = {[str(x) for x in c]}: certified lower bound on min_q max Brier excess = {float(lb):.6f}"
      f" > b: {lb > M.B_FR}  -> no q exists (Brier-infeasible although NLL-feasible)")
print("exact binary member intervals of feasible c (=q_0):",
      [tuple(round(v, 6) for v in iv) for iv in (
          (max(M.EMD * 0.52, 1 - math.sqrt(0.48 ** 2 + M.B / 2)), min(1 - M.EMD * 0.48, math.sqrt(0.52 ** 2 + M.B / 2))),
          (max(M.EMD * 0.5245, 1 - math.sqrt(0.4755 ** 2 + M.B / 2)),
           min(1 - M.EMD * 0.4755, math.sqrt(0.5245 ** 2 + M.B / 2))))])

print("\n== R4.d  Brier-only, K=3: every pair Brier-feasible, the triple is not (closed form + rational dual) ==")
found = None
for t in range(200000):
    ctr = rng.dirichlet(np.ones(3) * 3)
    P = np.abs(ctr + rng.normal(0, 0.02, (3, 3)))
    P = P / P.sum(1, keepdims=True)
    if any(M.brier_minimax(P[[i, j]])[0] > M.B * 0.98 for i, j in ((0, 1), (0, 2), (1, 2))):
        continue
    V3 = M.brier_minimax(P)[0]
    if V3 > M.B * 1.05:
        found = P
        break
if found is None:
    print("no Brier-only 3-member counterexample found")
else:
    Pr = [[F(float(x)).limit_denominator(10 ** 6) for x in row] for row in found]
    for row in Pr:
        row[2] = 1 - row[0] - row[1]
    Pf = np.array([[float(x) for x in r] for r in Pr])
    print("members:", [[f"{float(x):.6f}" for x in r] for r in Pr])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        V, cq, _ = M.brier_minimax(Pf[[i, j]])
        q = ratq(cq, 0)
        exc = max(max(M.brier_excess_exact([float(x) for x in Pr[m]], [float(x) for x in q])) for m in (i, j))
        print(f"pair ({i},{j}): closed-form V = {V:.6f}; exact max excess at rationalised q = c* : {float(exc):.6f}"
              f" <= b: {exc <= M.B_FR}")
    V, cq, _ = M.brier_minimax(Pf)
    beta = [min(sum((p[k] - (1 if k == y else 0)) ** 2 for k in range(3)) for p in Pr) for y in range(3)]
    c = [F(float(x)).limit_denominator(10 ** 9) for x in cq]
    c[2] = 1 - c[0] - c[1]
    lb = 1 - sum(x * x for x in c) - sum(cy * by for cy, by in zip(c, beta))
    print(f"triple: closed-form V = {V:.6f}; rational dual lower bound {float(lb):.6f} > b: {lb > M.B_FR}")

print("\n== R4.e  Helly. K=2: each member's feasible c-interval I(a) is an interval -> pairwise <=> whole bin ==")


def interval(a):
    lo = max(M.EMD * a, 1 - math.sqrt((1 - a) ** 2 + M.B / 2), 0.5)
    hi = min(1 - M.EMD * (1 - a), math.sqrt(a * a + M.B / 2))
    return lo, hi


bad = 0
for t in range(50000):
    a = rng.uniform(0.5, 1, int(rng.integers(2, 7)))
    a = a.min() + (a - a.min()) * rng.uniform(0, 0.012)
    iv = [interval(x) for x in a]
    pair = all(max(i1[0], i2[0]) <= min(i1[1], i2[1]) and min(i1[1], i2[1]) > 0.5 for i1, i2 in
               itertools.combinations(iv, 2))
    whole = max(i[0] for i in iv) <= min(i[1] for i in iv) and min(i[1] for i in iv) > 0.5
    bad += int(pair and not whole)
    if whole:  # spot-check the interval logic against the vector predicate at the midpoint
        cmid = 0.5 * (max(i[0] for i in iv) + min(i[1] for i in iv))
        bad += int(not all(M.g_ok(np.array([x, 1 - x]), np.array([cmid, 1 - cmid]), 0) for x in a
                           if cmid > max(i[0] for i in iv) + 1e-15 and cmid < min(i[1] for i in iv) - 1e-15))
print(f"K=2 pairwise-admissible but whole-inadmissible bins: {bad} / 50000 (PROVED: Helly in dimension 1)")
print("general K: each member's feasible set is convex in the (K-1)-dim plane sum q = 1 -> by Helly a bin is admissible"
      " iff every K members are (K=6: every 6-subset). R4.b1/b2 show K-subsets cannot be replaced by pairs for K>=3.")

print("\n== R4.f  joint (NLL+Brier+Class-closure) value = max_c [dist^2(c, P) + 1 - ||c||^2 - c.beta] (Sion) ==")


def proj(c, lo, d):
    K = c.size
    cons = [{"type": "eq", "fun": lambda z: z.sum() - 1}, {"type": "ineq", "fun": lambda z: z - lo},
            {"type": "ineq", "fun": lambda z: np.delete(z[d] - z, d)}]
    z0 = lo + (1 - lo.sum()) * np.eye(K)[d]
    r = minimize(lambda z: np.sum((z - c) ** 2), z0, method="SLSQP", constraints=cons,
                 options={"ftol": 1e-16, "maxiter": 300})
    return float(np.sum((r.x - c) ** 2))


gaps = []
for t in range(6):
    K = 3
    ctr = np.array([0.55, 0.3, 0.15]) + rng.normal(0, 0.03, 3)
    ctr = np.abs(ctr) / np.abs(ctr).sum()
    P = np.abs(ctr + rng.normal(0, 0.0015, (3, K)))
    P = P / P.sum(1, keepdims=True)
    d = M.decision(ctr)
    if len({M.decision(p) for p in P}) > 1 or P.max(0).sum() > M.ED:
        continue
    primal, qf, ok = M.joint_minimax(P, d)
    lo = M.EMD * P.max(0)
    Eb = np.eye(K)
    beta = np.array([min(np.sum((p - Eb[y]) ** 2) for p in P) for y in range(K)])
    best = -math.inf
    g = np.linspace(0, 1, 61)
    for x in g:
        for y_ in g:
            if x + y_ <= 1:
                c = np.array([x, y_, 1 - x - y_])
                best = max(best, proj(c, lo, d) + 1 - c @ c - c @ beta)
    gaps.append((primal, best))
print("primal (SLSQP) vs dual (grid 1/60 lower bound) per bin:", [(round(a, 7), round(b, 7)) for a, b in gaps])
print("(dual grid value <= primal always; equality up to grid resolution confirms the formula)")
