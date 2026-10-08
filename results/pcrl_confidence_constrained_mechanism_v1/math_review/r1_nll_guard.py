"""R1 -- NLL guard q_k >= e^{-d} p_k: unclipped and clipped (1e-12) log-loss implications, zeros, underflow, and the
float64 semantics of "no tolerance". Synthetic only. Run: python -P r1_nll_guard.py  (about 10 s)."""
import math
import os
import sys
from decimal import Decimal

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(20261008)
c = M.CLIP
DEC_D = Decimal(5) / Decimal(1000)
DEC_C = Decimal(c)          # exact binary value of the float 1e-12 (the clip as numpy uses it)


def dlog(x):
    return Decimal(float(x)).ln()


def clipped_excess_exact(py, qy):
    """-log clip(q_y) + log clip(p_y) in 60-digit decimal arithmetic on the exact float inputs."""
    cp = max(min(Decimal(float(py)), Decimal(1)), DEC_C)
    cq = max(min(Decimal(float(qy)), Decimal(1)), DEC_C)
    return cp.ln() - cq.ln()


print("== R1.a  exact check: real NLL guard => unclipped and clipped excess <= d (all labels) ==")
viol_unclip = viol_clip = n_pairs = 0
regimes = {"p_y < clip": 0, "p_y >= clip, q_y < clip": 0, "both >= clip": 0, "p_y = 0": 0}
for t in range(4000):
    K = int(rng.choice([2, 6]))
    # reference with some coordinates far below the clip, some zeros, some near the clip
    logs = rng.uniform(-35, 0, K)
    p = np.exp(logs)
    if t % 7 == 0:
        p[rng.integers(K)] = 0.0
    if t % 5 == 0:
        p[rng.integers(K)] = c * rng.uniform(0.5, 1.5)
    p = p / p.sum()
    # q = e^{-d} p + (1 - e^{-d}) r, r in simplex; this parametrises every q in the simplex with q >= e^{-d} p
    r = rng.dirichlet(np.full(K, 0.3))
    if t % 3 == 0:
        r = np.eye(K)[rng.integers(K)]
    q = M.EMD * p + (1 - M.EMD) * r
    # also push some coordinates of q exactly onto / just above the boundary and below the clip
    if not M.nll_ok_exact(p, q):
        continue  # only test pairs that satisfy the REAL-arithmetic guard
    for y in range(K):
        n_pairs += 1
        if p[y] == 0:
            regimes["p_y = 0"] += 1
        elif p[y] < c:
            regimes["p_y < clip"] += 1
        elif q[y] < c:
            regimes["p_y >= clip, q_y < clip"] += 1
        else:
            regimes["both >= clip"] += 1
        if p[y] > 0 and q[y] > 0:
            if dlog(p[y]) - dlog(q[y]) > DEC_D:
                viol_unclip += 1
        if clipped_excess_exact(p[y], q[y]) > DEC_D:
            viol_clip += 1
# targeted regime p_y in [c, e^d c), q_y in [e^-d p_y, c): the only way q_y < c <= p_y under the guard
for t in range(3000):
    py = c * (1 + rng.uniform(0, M.ED - 1))
    qy = py * M.EMD * (1 + rng.uniform(0, 1) * (c / (py * M.EMD) - 1))
    if not (qy < c <= py) or not M.nll_ok_exact([py], [qy]):
        continue
    n_pairs += 1
    regimes["p_y >= clip, q_y < clip"] += 1
    if dlog(py) - dlog(qy) > DEC_D:
        viol_unclip += 1
    if clipped_excess_exact(py, qy) > DEC_D:
        viol_clip += 1
print(f"label-pairs tested {n_pairs}; regimes {regimes}")
print(f"violations: unclipped {viol_unclip}, clipped {viol_clip}   (PROVED: expected 0, 0)")

print("\n== R1.b  the clipped all-label bound does NOT imply the NLL guard (G is strictly stronger) ==")
p = np.array([1 - 1e-13, 1e-13])
q = np.array([1.0, 0.0])
ex = [float(clipped_excess_exact(p[y], q[y])) for y in range(2)]
print(f"p={p.tolist()} q={q.tolist()}: clipped excess per label {ex} (both <= d) but NLL guard holds? {M.nll_ok(p, q)}")

print("\n== R1.c  zeros and underflow ==")
print("p_y = 0: NLL requires q_y >= 0 (vacuous); log(p_y/q_y) = -inf if q_y > 0 and 0/0 (undefined) if q_y = 0;")
print("         the clipped excess is exactly 0 in both cases:",
      float(clipped_excess_exact(0.0, 0.0)), float(clipped_excess_exact(0.0, 1e-300)))
# smallest Ucal entry under the frozen text rule: alpha in [0.25, 4], floor 1e-12
for K in (2, 6):
    P = np.full(K, 1e-12)
    P[0] = 1 - (K - 1) * 1e-12
    for a in (0.25, 1.0001, 4.0):
        u = M.ucal_text_rule(P, a)
        print(f"K={K} alpha={a}: min Ucal entry {u.min():.3e} (normal float64: {u.min() >= np.finfo(float).tiny});"
              f" e^-d * min = {M.EMD * u.min():.3e}")
sub = 5e-324
print(f"subnormal p_k = {sub}: fl(e^-d * p_k) = {M.EMD * sub} -> the float predicate then needs q_k >= p_k exactly"
      " (harmless; cannot arise from Ucal with alpha != 1, see above)")

print("\n== R1.d  float64 semantics of 'no tolerance': algebraically equivalent predicates disagree at the boundary ==")
p = np.concatenate([rng.random(100000) ** 3, 10.0 ** rng.uniform(-14, 0, 100000)])
q = M.EMD * p                                   # boundary construction q_k = fl(fl(e^-d) * p_k)
forms = {
    "mult  q >= exp(-d)*p": q >= np.exp(-M.D) * p,
    "ratio p/q <= exp(d)": p / q <= np.exp(M.D),
    "log   log p - log q <= d": np.log(p) - np.log(q) <= M.D,
    "clip  -log clip q + log clip p <= d": (-np.log(np.clip(q, c, 1)) + np.log(np.clip(p, c, 1))) <= M.D,
}
for k, v in forms.items():
    print(f"  {k:40s} passes {v.mean():.4f}")
idx = np.arange(0, p.size, 499)
real_fail = 0
worst = Decimal(0)
for i in idx:
    lhs = Decimal(float(q[i]))
    rhs = (-DEC_D).exp() * Decimal(float(p[i]))
    if lhs < rhs:
        real_fail += 1
    worst = max(worst, dlog(p[i]) - dlog(q[i]) - DEC_D)
print(f"  real arithmetic (60 digits) violates q >= e^-d p on {real_fail}/{idx.size} sampled boundary coordinates;"
      f" worst real log-excess over d = {float(worst):.3e}")
fl_worst = np.max((np.log(p) - np.log(q)) - M.D)
print(f"  worst float-evaluated (log p - log q) - d = {fl_worst:.3e}")

# realistic: bin representatives built as q = e^-d * M + slack on the decision coordinate (NLL-tight)
dis = 0
nb = 20000
for t in range(nb):
    K = 6
    ctr = rng.dirichlet(np.ones(K))
    P = np.abs(ctr + rng.normal(0, 4e-4, (3, K)))
    P = P / P.sum(1, keepdims=True)
    d = M.decision(ctr)
    Mx = P.max(0)
    if Mx.sum() > M.ED:
        continue
    qq = M.EMD * Mx
    qq[d] += 1.0 - qq.sum()
    a = all(M.nll_ok(p_, qq) for p_ in P)
    b = all(np.all(np.log(p_) - np.log(qq) <= M.D) for p_ in P)
    if a != b:
        dis += 1
print(f"  NLL-tight bin representatives (q = e^-d M + slack e_d, K=6): mult-form vs log-form certification disagree on"
      f" {dis}/{nb} bins")
