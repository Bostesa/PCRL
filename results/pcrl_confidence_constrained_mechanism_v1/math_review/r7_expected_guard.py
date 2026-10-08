"""R7 -- pointwise G vs the teacher-expected guard G_exp (KL(p||q) <= d, ||q - p||^2 <= b, class): G is contained in
G_exp; G_exp's member shift and bin diameter (binary exact, K=6 numerical); its observed-label meaning depends on
calibration; averages over rows are not pointwise. Synthetic only. Run: python -P r7_expected_guard.py (about 30 s)."""
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(9)


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def gexp_ok(p, q, d):
    return kl(p, q) <= M.D and np.sum((q - p) ** 2) <= M.B and M.class_ok(q, d)


print("== R7.a  G => G_exp (KL = E_{y~p} log(p_y/q_y) <= max_y; ||q-p||^2 = E_{y~p} Brier excess <= max_y) ==")
bad = n = 0
for t in range(100000):
    K = int(rng.choice([2, 6]))
    p = rng.dirichlet(np.ones(K))
    d = M.decision(p)
    q = (1 - 0.004) * p + 0.004 * rng.dirichlet(np.ones(K))
    if M.g_ok(p, q, d):
        n += 1
        bad += int(not gexp_ok(p, q, d))
print(f"G-admissible pairs {n}; not G_exp-admissible {bad}")

print("\n== R7.b  binary: member shift and bin width under G_exp vs G ==")


def gexp_interval(c):
    """members a = p_0 with KL((a,1-a)||(c,1-c)) <= d, 2(a-c)^2 <= b, a >= 1/2 (class 0)."""
    def ok(a):
        return kl([a, 1 - a], [c, 1 - c]) <= M.D and 2 * (a - c) ** 2 <= M.B
    lo_, hi_ = 0.5, c
    if ok(0.5):
        L = 0.5
    else:
        for _ in range(100):
            mid = 0.5 * (lo_ + hi_)
            lo_, hi_ = (lo_, mid) if ok(mid) else (mid, hi_)
        L = hi_
    lo_, hi_ = c, 1.0
    if ok(1.0):
        R = 1.0
    else:
        for _ in range(100):
            mid = 0.5 * (lo_ + hi_)
            lo_, hi_ = (mid, hi_) if ok(mid) else (lo_, mid)
        R = lo_
    return L, R


print(" c      W_G      W_Gexp   ratio")
best = 0.0
for c in (0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    Lg, Rg = M.bin_interval(c)
    Le, Re = gexp_interval(c)
    print(f"{c:4.2f}  {Rg - Lg:.5f}  {Re - Le:.5f}  {(Re - Le) / (Rg - Lg):5.1f}x")
for c in np.linspace(0.501, 0.999, 999):
    Le, Re = gexp_interval(c)
    best = max(best, Re - Le)
print(f"sup_c W_Gexp(c) = {best:.5f}  (G: e^d - 1 = {M.ED - 1:.5f}; Brier-part bound 2 sqrt(b/2) = {2*math.sqrt(M.B/2):.5f};"
      f" Pinsker+triangle bound 2 sqrt(d/2) = {2*math.sqrt(M.D/2):.4f})")

print("\n== R7.c  K=6: largest TV diameter of a G_exp bin (multi-start SLSQP over p, p', q) ==")
K = 6
bestv = 0.0
for s in range(40):
    q0 = rng.dirichlet(np.ones(K) * float(rng.choice([0.7, 2, 6])))
    d = M.decision(q0)
    dirn = rng.normal(size=K)
    dirn -= dirn.mean()
    dirn *= 0.02 / np.abs(dirn).sum()
    z0 = np.r_[q0 + dirn, q0 - dirn, q0]
    if z0.min() <= 0:
        continue

    def f(z):
        return -0.5 * np.sum(np.sqrt((z[:K] - z[K:2 * K]) ** 2 + 1e-14))

    cons = [{"type": "eq", "fun": lambda z: np.array([z[:K].sum() - 1, z[K:2 * K].sum() - 1, z[2 * K:].sum() - 1])},
            {"type": "ineq", "fun": lambda z: z - 1e-9},
            {"type": "ineq", "fun": lambda z: np.array([M.D - kl(z[:K], z[2 * K:]), M.D - kl(z[K:2 * K], z[2 * K:]),
                                                        M.B - np.sum((z[:K] - z[2 * K:]) ** 2),
                                                        M.B - np.sum((z[K:2 * K] - z[2 * K:]) ** 2)])},
            {"type": "ineq", "fun": lambda z: np.r_[np.delete(z[2 * K + d] - z[2 * K:], d),
                                                    np.delete(z[d] - z[:K], d), np.delete(z[K + d] - z[K:2 * K], d)]}]
    r = minimize(f, z0, method="SLSQP", constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    z = r.x
    if r.success and gexp_ok(z[:K], z[2 * K:], d) is not None:
        ok = (kl(z[:K], z[2 * K:]) <= M.D + 1e-9 and kl(z[K:2 * K], z[2 * K:]) <= M.D + 1e-9 and
              np.sum((z[:K] - z[2 * K:]) ** 2) <= M.B + 1e-9 and np.sum((z[K:2 * K] - z[2 * K:]) ** 2) <= M.B + 1e-9)
        if ok:
            bestv = max(bestv, 0.5 * np.abs(z[:K] - z[K:2 * K]).sum())
print(f"largest G_exp bin TV diameter found (K=6): {bestv:.4f}  (G: <= {M.ED - 1:.4f}; ratio ~{bestv / (M.ED - 1):.0f}x;"
      f" Pinsker+triangle upper bound {2 * math.sqrt(M.D / 2):.3f})")

print("\n== R7.d  observed-label meaning of G_exp depends on calibration; G does not ==")
p = np.array([0.99, 0.01])
q = np.array([0.995, 0.005])
print(f"p={p.tolist()} q={q.tolist()}: KL(p||q)={kl(p, q):.5f} <= d, ||q-p||^2={np.sum((q-p)**2):.1e} <= b ->"
      f" G_exp-admissible {gexp_ok(p, q, 0)}; G-admissible {M.g_ok(p, q, 0)}")
for pi in (np.array([0.99, 0.01]), np.array([0.95, 0.05]), np.array([0.9, 0.1])):
    ll = float(pi @ (np.log(p) - np.log(q)))
    br = float(pi @ M.brier_excess(p, q))
    print(f"  true label law pi={pi.tolist()}: expected log-loss excess {ll:+.5f} ({ll / M.D:4.1f} d),"
          f" Brier excess {br:+.5f} ({br / M.B:4.1f} b)")
qg = np.array([0.99, 0.01]) * M.EMD
qg[0] = 1 - qg[1]
print(f"  a G-admissible q = {qg.tolist()}: per-label log excess {(np.log(p) - np.log(qg)).round(6).tolist()} <= d for"
      f" EVERY y, so <= d under ANY label law (calibrated or not)")

print("\n== R7.e  averages over rows are not pointwise ==")
print("two rows, excess 2d and 0: mean = d satisfies an average allowance, row 1 violates the pointwise guard; the"
      " converse (pointwise => average) holds.")
