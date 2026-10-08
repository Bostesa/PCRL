"""R5 -- geometry of admissible bins. Sum-max = 1 + TV; TV and l_inf diameter <= e^d - 1 (tight under the FULL G in
K=2); exact binary member intervals [L(c), R(c)] and single-member shift bounds; minimal number of bins covering the
binary class range; per-coordinate disclosure interval of p given any admissible output; volume bound for K=6;
non-convexity of the compatible set in p (K>=3); exact binary maximum coverage by C bins (DP) vs the F3u bound.
Synthetic only. Run: python -P r5_geometry.py (about 15 s)."""
import math
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(5)
ED1 = M.ED - 1.0

print("== R5.a  sum_k max(p_k, p'_k) = 1 + TV(p, p') ==")
err = 0.0
for t in range(20000):
    K = int(rng.choice([2, 6]))
    p, pp = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
    err = max(err, abs(np.maximum(p, pp).sum() - 1 - 0.5 * np.abs(p - pp).sum()))
print(f"max error {err:.1e}. Hence NLL-admissible bin => TV diam <= e^d - 1 = {ED1:.7f}; also l_inf diam <= e^d - 1"
      " (|p_k - p'_k| <= TV).")

print("\n== R5.b  binary exact structure (decision 0, a = p_0, representative q = (c, 1-c)) ==")
print(" c      L(c)      R(c)      W=R-L     W_NLL-only   binding")
for c in (0.5005, 0.51, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995):
    L, R = M.bin_interval(c)
    Ln, Rn = M.bin_interval_nll(c)
    print(f"{c:6.4f} {L:9.6f} {R:9.6f} {R - L:9.6f} {Rn - Ln:11.7f}   {'NLL' if abs((R-L)-(Rn-Ln))<1e-12 else 'Brier'}")
cs = np.linspace(0.5 + 1e-9, 1.0, 200001)
W = np.array([M.bin_interval(c)[1] - M.bin_interval(c)[0] for c in cs])
full = cs[np.abs(W - ED1) < 1e-12]
print(f"sup_c W(c) = {W.max():.7f} (e^d - 1 = {ED1:.7f}); attained for c in [{full.min():.4f}, {full.max():.4f}]")

# exact certificate of tightness at c = 0.9: members a = L(0.9), R(0.9) (shrunk 1e-12 inward), q = (0.9, 1 - 0.9)
# (1 - x is exact in float64 for x in [1/2, 1] (Sterbenz), so p and q sum to 1 exactly as rationals)
L, R = M.bin_interval(0.9)
a1, a2 = L + 1e-12, R - 1e-12
q = np.array([0.9, 1 - 0.9])
ok = [M.g_ok_exact(np.array([a, 1 - a]), q, 0) for a in (a1, a2)]
print(f"tightness: members a = {a1:.9f}, {a2:.9f} share q = (0.9, 0.1) under EXACT G: {ok};"
      f" TV = {a2 - a1:.9f} vs e^d - 1 = {ED1:.9f}")
mid = (cs > 0.51) & (cs < 0.99)
print(f"min over c in (0.51, 0.99) of W(c) = {W[mid].min():.7f} at c = {cs[mid][W[mid].argmin()]:.4f}"
      " (Brier-binding region c < 0.854: W ~ b/(4c(1-c))... see table)")

print("\n== R5.c  exact binary single-member shift bounds (a = p_0 >= 1/2) ==")
print(" a     s+ (toward 0)           s- (toward 1)")
for a in (0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    sp = min((1 - M.EMD) * (1 - a), math.sqrt(a * a + M.B / 2) - a)
    sm = min((1 - M.EMD) * a, math.sqrt((1 - a) ** 2 + M.B / 2) - (1 - a), a - 0.5 if a > 0.5 else 0.0)
    print(f"{a:4.2f}  {sp:.6f} ({'NLL' if sp == (1 - M.EMD) * (1 - a) else 'Brier'})"
          f"    {sm:.6f}")
print("(s- also capped by strict Class at a - 1/2.)  General K (PROVED): -min{(1-e^-d) p_k, (b - S)/2} <= q_k - p_k"
      " <= min{(1-e^-d)(1-p_k), b/(2 p_k)}, S = ||q||^2 - ||p||^2; TV(p, q) <= 1 - e^-d.")

print("\n== R5.d  minimal number of admissible bins covering the whole binary class range a in [1/2, 1] ==")


def chain(interval_fn):
    a0, n = 0.5, 0
    while True:
        lo, hi = 0.5 + 1e-15, 1.0
        for _ in range(200):          # largest c with L(c) <= a0 (L is non-decreasing)
            mid = 0.5 * (lo + hi)
            if interval_fn(mid)[0] <= a0:
                lo = mid
            else:
                hi = mid
        n += 1
        R = interval_fn(lo)[1]
        if R >= 1.0 - 1e-15:
            return n
        a0 = R + 1e-15


n_joint = chain(M.bin_interval)
n_nll = chain(M.bin_interval_nll)
print(f"G (NLL+Brier+Class): {n_joint} bins per class;  NLL only: {n_nll} bins per class;  capacity: 8 (income)")
print("(greedy chaining is optimal for covering an interval by intervals whose endpoints L(c), R(c) are monotone)")

print("\n== R5.e  disclosure: any admissible output q pins every coordinate of p to an interval of width e^d - 1 ==")
worst_tv = 0.0
viol = 0
for t in range(3000):
    K = 6
    q = rng.dirichlet(np.ones(K) * 2)
    d = M.decision(q)
    comp = []
    for _ in range(300):
        p = q + rng.normal(0, 0.003, K)
        p -= p.mean() - 1.0 / K * 0  # keep sum: adjust below
        p = p - (p.sum() - 1.0) / K
        if p.min() < 0 or M.decision(p) != d:
            continue
        if M.g_ok(p, q, d):
            comp.append(p)
            lo, hi = M.ED * q - ED1, M.ED * q
            viol += int(np.any(p < lo - 1e-15) or np.any(p > hi + 1e-15))
    for i in range(len(comp)):
        for j in range(i + 1, len(comp)):
            worst_tv = max(worst_tv, 0.5 * np.abs(comp[i] - comp[j]).sum())
print(f"interval violations {viol}; largest TV between two inputs compatible with the same q: {worst_tv:.6f}"
      f" (<= e^d - 1 = {ED1:.6f})")
print(f"volume bound: bins needed to cover a whole class region >= 1/(K (e^d-1)^(K-1)) = K=2: {1/(2*ED1):.1f};"
      f" K=6: {1/(6*ED1**5):.3e}")

print("\n== R5.f  the compatible set A(q) = {p : q admissible for p} is NOT convex in p for K >= 3 ==")
# Brier part: ||p - e_y||^2 >= ||q - e_y||^2 - b, i.e. p OUTSIDE a ball (reverse convex). Targeted search: two points
# on that circle (K=3 plane) symmetric about the ray through q; their midpoint lies inside the ball.
u_ = np.array([1, -1, 0]) / math.sqrt(2)
v_ = np.array([1, 1, -2]) / math.sqrt(6)
best = None
for t in range(100000):
    q = rng.dirichlet(np.ones(3) * float(rng.choice([0.5, 2, 8])))
    d = M.decision(q)
    y = int(rng.integers(3))
    e = np.eye(3)[y]
    R2 = np.sum((q - e) ** 2) - M.B
    if R2 <= 0:
        continue
    r = math.sqrt(R2)
    w = q - e
    th0 = math.atan2(w @ v_, w @ u_)
    phi = rng.uniform(0, 0.02)
    p1, p2 = (e + r * (1 + 1e-10) * (math.cos(th0 + sg * phi) * u_ + math.sin(th0 + sg * phi) * v_) for sg in (1, -1))
    if min(p1.min(), p2.min()) < 0 or not (M.decision(p1) == d == M.decision(p2)):
        continue
    if not (M.g_ok(p1, q, d) and M.g_ok(p2, q, d)):
        continue
    mid = 0.5 * (p1 + p2)
    ex = M.brier_excess(mid, q).max()
    if ex > M.B and (best is None or ex > best[0]):
        best = (ex, q, p1, p2, d)
if best:
    ex, q, p1, p2, d = best
    print(f"q={np.round(q, 6).tolist()}, p1={np.round(p1, 6).tolist()}, p2={np.round(p2, 6).tolist()}: q admissible"
          f" for p1 and p2 (float G: {M.g_ok(p1, q, d)}, {M.g_ok(p2, q, d)}), NOT for their midpoint (max Brier excess"
          f" {ex:.6f} > b = {M.B}); TV(p1, p2) = {0.5 * np.abs(p1 - p2).sum():.5f}")
    print("K = 2: A(q) = [L(c), R(c)] is an interval (convex).")
else:
    print("no non-convex example found")

print("\n== R5.g  binary: exact maximum coverage by C bins (DP over maximal windows) vs the registered F3u bound ==")


def reach(a0):
    lo, hi = 0.5 + 1e-15, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if M.bin_interval(mid)[0] <= a0:
            lo = mid
        else:
            hi = mid
    return M.bin_interval(lo)[1]


def exact_cover(a, C):
    a = np.sort(a)
    n = a.size
    nxt = np.searchsorted(a, [reach(x) for x in a], side="right")
    f = np.zeros((n + 1, C + 1), dtype=int)
    for i in range(n - 1, -1, -1):
        for m in range(1, C + 1):
            f[i, m] = max(f[i + 1, m], (nxt[i] - i) + f[nxt[i], m - 1])
    return f[0, C]


def f3u(a, C):
    """Registered F3u for one class: sum of the C largest N(r), N(r) = #{r' : max(a,a') + max(1-a,1-a') <= e^d}."""
    N = np.array([np.sum(np.maximum(x, a) + np.maximum(1 - x, 1 - a) <= M.ED) for x in a])
    return min(int(np.sort(N)[::-1][:C].sum()), a.size)


for name, a in (("uniform on [0.5,1], n=2000", rng.uniform(0.5, 1, 2000)),
                ("concentrated: 70% in [0.97,1], n=2000",
                 np.r_[rng.uniform(0.97, 1, 1400), rng.uniform(0.5, 0.97, 600)]),
                ("clustered near 0.55 (Brier-tight zone), n=2000", 0.55 + rng.normal(0, 0.004, 2000))):
    a = np.clip(a, 0.5, 1.0)
    for C in (8,):
        ex = exact_cover(a, C)
        print(f"{name}: exact max coverage with C={C} bins {ex / a.size:.3f};  F3u bound {f3u(a, C) / a.size:.3f}")
