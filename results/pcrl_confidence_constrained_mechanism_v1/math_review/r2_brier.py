"""R2 -- the all-label Brier guard: identity with the source per-row Brier, expected (y ~ p) excess, ball form and
convexity, the exact max-label formula, the deficit identity, the closed-form Brier-only bin value (water-filling),
and float-form disagreement at the boundary. Synthetic only. Run: python -P r2_brier.py (about 15 s)."""
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(7)


def src_brier(P, y):
    """Source per-row convention: sum_k (P_k - 1[y = k])^2 (dpc.utility.per_row, re-written)."""
    return np.sum((P - np.eye(P.size)[y]) ** 2)


print("== R2.a  contract excess == source Brier(q,y) - Brier(p,y); expected-excess identities ==")
e1 = e2 = e3 = e4 = 0.0
for t in range(20000):
    K = int(rng.choice([2, 3, 6]))
    p = rng.dirichlet(np.ones(K))
    q = rng.dirichlet(np.ones(K))
    ex = M.brier_excess(p, q)
    e1 = max(e1, max(abs(ex[y] - (src_brier(q, y) - src_brier(p, y))) for y in range(K)))
    dl = q - p
    e2 = max(e2, abs(p @ ex - dl @ dl))                         # y ~ p : ||q - p||^2
    e3 = max(e3, abs(q @ ex + dl @ dl))                         # y ~ q : -||q - p||^2
    e4 = max(e4, abs(ex.max() - (dl @ dl + 2 * (p @ dl - dl.min()))))  # max over labels
print(f"max |contract - source| {e1:.2e}; |E_p excess - ||q-p||^2| {e2:.2e}; |E_q excess + ||q-p||^2| {e3:.2e};"
      f" |max_y excess - (||d||^2 + 2(p.d - min d))| {e4:.2e}")

print("\n== R2.b  all-label => expected (y~p); expected does NOT imply all-label ==")
p = np.array([0.99, 0.01])
q = np.array([0.995, 0.005])
ex = M.brier_excess(p, q)
print(f"p={p} q={q}: ||q-p||^2 = {np.sum((q-p)**2):.2e} <= b, but excess at y=1 = {ex[1]:.5f} > b={M.B}")

print("\n== R2.c  ball form ||q - e_y||^2 <= ||p - e_y||^2 + b  <=> label-y guard; per-member feasible set convex ==")
bad = 0
for t in range(5000):
    K = int(rng.choice([2, 6]))
    p = rng.dirichlet(np.ones(K))
    d = M.decision(p)
    # two random G-admissible q's for p (rejection around p), then their midpoint must be admissible
    qs = []
    for _ in range(400):
        r = rng.dirichlet(np.ones(K))
        eta = rng.uniform(0, 0.004)
        q = (1 - eta) * p + eta * r
        if M.g_ok(p, q, d):
            qs.append(q)
        if len(qs) == 2:
            break
    if len(qs) == 2:
        mid = 0.5 * (qs[0] + qs[1])
        if not M.g_ok(p, mid, d):
            bad += 1
print(f"midpoint of two admissible q's inadmissible: {bad} cases (PROVED convex: NLL box, K balls, open half-spaces)")

print("\n== R2.d  deficit identity: label-y guard <=> p_y - q_y <= (b - S)/2, S = ||q||^2 - ||p||^2 ==")
mism = 0
for t in range(20000):
    K = int(rng.choice([2, 6]))
    p = rng.dirichlet(np.ones(K))
    q = (1 - 0.01) * p + 0.01 * rng.dirichlet(np.ones(K))
    S = q @ q - p @ p
    a = M.brier_excess(p, q) <= M.B
    b2 = (p - q) <= (M.B - S) / 2
    mism += int(np.any(a != b2))
print(f"mismatches {mism} (identity; float ties aside)")

print("\n== R2.e  closed-form Brier-only bin value V = max_c [1 - ||c||^2 - c.beta] vs direct min-max (SLSQP) ==")
worst = 0.0
nfail = 0
lower_ok = True
for t in range(300):
    K = int(rng.choice([2, 3, 6]))
    n = int(rng.integers(1, 5))
    ctr = rng.dirichlet(np.ones(K))
    P = np.abs(ctr + rng.normal(0, 0.03, (n, K)))
    P = P / P.sum(1, keepdims=True)
    V, c, beta = M.brier_minimax(P)
    E = np.eye(K)
    Bp = np.array([[np.sum((p_ - E[y]) ** 2) for y in range(K)] for p_ in P])

    def obj(z):
        return z[K]

    cons = [{"type": "eq", "fun": lambda z: z[:K].sum() - 1},
            {"type": "ineq", "fun": lambda z: z[:K]},
            {"type": "ineq", "fun": lambda z: np.array([z[K] - (np.sum((z[:K] - E[y]) ** 2) - Bp[i, y])
                                                         for i in range(n) for y in range(K)])}]
    best = math.inf
    for s in range(8):
        q0 = rng.dirichlet(np.ones(K)) if s else P.mean(0)
        t0 = max(np.sum((q0 - E[y]) ** 2) - Bp[i, y] for i in range(n) for y in range(K))
        res = minimize(obj, np.r_[q0, t0], method="SLSQP", constraints=cons, options={"ftol": 1e-15, "maxiter": 500})
        if res.success:
            best = min(best, res.x[K])
    # the closed-form q = c attains V exactly:
    att = max(np.sum((c - E[y]) ** 2) - Bp[i, y] for i in range(n) for y in range(K))
    if not math.isfinite(best):
        nfail += 1
        worst = max(worst, abs(att - V))
        continue
    lower_ok = lower_ok and best >= V - 1e-9          # no feasible q beats the closed form
    worst = max(worst, abs(best - V), abs(att - V))
print(f"max |closed form - SLSQP|, |value at q=c* - V| over 300 random bins: {worst:.2e};"
      f" SLSQP never below V: {lower_ok}; SLSQP failures (all starts) {nfail}")
print("single member: V = 0 at c = p (check):", M.brier_minimax(np.array([[0.7, 0.2, 0.1]]))[0])

print("\n== R2.f  float forms of the Brier guard disagree at the boundary ==")
dis = 0
nt = 0
for t in range(20000):
    K = 6
    p = rng.dirichlet(np.ones(K))
    dirn = rng.normal(size=K)
    dirn -= dirn.mean()
    # bisection on s so that max-label excess of q = p + s*dirn equals b (as closely as float allows)
    lo, hi = 0.0, 1.0
    if M.brier_excess(p, p + hi * dirn).max() <= M.B:
        continue
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if M.brier_excess(p, p + mid * dirn).max() <= M.B:
            lo = mid
        else:
            hi = mid
    q = p + lo * dirn
    if q.min() < 0:
        continue
    y = int(np.argmax(M.brier_excess(p, q)))
    f_contract = (q @ q - p @ p - 2 * (q[y] - p[y])) <= M.B
    f_source = (src_brier(q, y) - src_brier(p, y)) <= M.B
    f_fsum = (math.fsum(q * q) - math.fsum(p * p) - 2 * (q[y] - p[y])) <= M.B
    nt += 1
    dis += int(len({bool(f_contract), bool(f_source), bool(f_fsum)}) > 1)
print(f"boundary-tight q (K=6): the three float forms (dot, source per-row difference, fsum) disagree on {dis}/{nt}")
