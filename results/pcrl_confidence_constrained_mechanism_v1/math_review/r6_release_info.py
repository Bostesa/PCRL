"""R6 -- SEX information in ANY admissible release (deterministic or randomised). Every supported output q confines the
input to A(q) with TV-diameter <= D = e^d - 1. Under (i) S - p - T Markov (the release reads p and independent
randomness only) and (ii) eta(p) = P(S = 1 | p) L-Lipschitz in TV:
    Bacc(S | p) - L D <= Bacc(S | T) <= Bacc(S | p),     I(S; p) - min{(L D)^2 / kappa, h(L D)} <= I(S; T) <= I(S; p),
kappa = min_t eta_t (1 - eta_t), h = binary entropy (nats; L D <= 1/2). Without (ii): a pathological law loses ALL
information in one admissible bin. Without (i): admissibility does not bound leakage from above. Synthetic only. Run: python -P r6_release_info.py (about 20 s)."""
import math
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(6)
DD = M.ED - 1.0


def reach(a0):
    lo, hi = 0.5 + 1e-15, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if M.bin_interval(mid)[0] <= a0:
            lo = mid
        else:
            hi = mid
    return M.bin_interval(lo)[1], lo


def hb(x):
    return 0.0 if x <= 0 or x >= 1 else -(x * math.log(x) + (1 - x) * math.log(1 - x))


def law(n, beta, spacing):
    a = 0.5 + spacing * np.arange(n)
    w = np.exp(-((a - 0.8) / 0.12) ** 2)
    w /= w.sum()
    eta = 1 / (1 + np.exp(-beta * (a - 0.75)))
    return a, w, eta


def min_mi_partition(a, w, eta):
    """DP over contiguous admissible bins (a_j <= reach(a_i)) maximising sum_t P(t) h(eta_t) = minimising I(S;T)."""
    n = a.size
    R = np.array([reach(x)[0] for x in a])
    cw = np.r_[0, np.cumsum(w)]
    cs = np.r_[0, np.cumsum(w * eta)]

    def h(x):
        return 0.0 if x <= 0 or x >= 1 else -(x * math.log(x) + (1 - x) * math.log(1 - x))
    best = np.full(n + 1, -np.inf)
    best[n] = 0.0
    arg = np.zeros(n, int)
    for i in range(n - 1, -1, -1):
        j = i
        while j < n and a[j] <= R[i]:
            pt = cw[j + 1] - cw[i]
            val = pt * h((cs[j + 1] - cs[i]) / pt) + best[j + 1]
            if val > best[i]:
                best[i], arg[i] = val, j + 1
            j += 1
    bins, i = [], 0
    while i < n:
        bins.append((i, arg[i]))
        i = arg[i]
    return bins


def stats_partition(w, eta, bins):
    J = np.zeros((2, len(bins)))
    for t, (i, j) in enumerate(bins):
        J[1, t] = np.sum(w[i:j] * eta[i:j])
        J[0, t] = np.sum(w[i:j] * (1 - eta[i:j]))
    return M.mi_bayes(J), J


print("== R6.a  smooth law, binary: the most SEX-destroying admissible partition vs the bound ==")
for beta in (4.0, 20.0, 80.0):
    a, w, eta = law(5001, beta, 0.0001)
    L = beta / 4.0                      # Lipschitz constant of eta in a; TV(p, p') = |a - a'| for K = 2
    Jp = np.vstack([w * (1 - eta), w * eta])
    Ip, Bp = M.mi_bayes(Jp)
    bins = min_mi_partition(a, w, eta)
    (It, Bt), J = stats_partition(w, eta, bins)
    et = J[1] / J.sum(0)
    kappa = np.min(et * (1 - et))
    # every bin is admissible: certify with its representative c (members inside [L(c), R(c)])
    okb = all(a[j - 1] <= reach(a[i])[0] for i, j in bins)
    bound = min((L * DD) ** 2 / kappa, hb(L * DD))
    print(f"beta={beta:5.1f} (L={L:5.1f}): I(S;p)={Ip:.6f} I(S;T*)={It:.6f} loss={Ip-It:.2e} <= min((LD)^2/kappa,"
          f" h(LD))={bound:.2e};  Bacc(p)={Bp:.5f} Bacc(T*)={Bt:.5f} loss={Bp-Bt:.2e} <= L D={L*DD:.2e};"
          f" bins={len(bins)} admissible={okb}")

print("\n== R6.b  randomised admissible channel (LP: minimise Bayes accuracy) on a coarser grid ==")
a, w, eta = law(501, 20.0, 0.001)
L = 5.0
n = a.size
# outputs: maximal windows starting at each grid point (WLOG maximal admissible sets, see R9)
wins = []
for i in range(n):
    R = reach(a[i])[0]
    j = i
    while j < n and a[j] <= R:
        j += 1
    wins.append((i, j))
wins = sorted(set(wins))
T = len(wins)
var = [(x, t) for t, (i, j) in enumerate(wins) for x in range(i, j)]
nv = len(var)
# variables: W(t|x) for admissible (x, t), then z_t ; minimise sum z_t ; z_t >= sum_x P(x) P(s|x) W(t|x), s = 0, 1
c = np.r_[np.zeros(nv), np.ones(T)]
rows, b_ub = [], []
A_ub = np.zeros((2 * T, nv + T))
for k, (x, t) in enumerate(var):
    A_ub[2 * t, k] = w[x] * (1 - eta[x])
    A_ub[2 * t + 1, k] = w[x] * eta[x]
for t in range(T):
    A_ub[2 * t, nv + t] = -1
    A_ub[2 * t + 1, nv + t] = -1
A_eq = np.zeros((n, nv + T))
for k, (x, t) in enumerate(var):
    A_eq[x, k] = 1
res = linprog(c, A_ub=A_ub, b_ub=np.zeros(2 * T), A_eq=A_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
Wt = res.x[:nv]
J = np.zeros((2, T))
for k, (x, t) in enumerate(var):
    J[0, t] += w[x] * (1 - eta[x]) * Wt[k]
    J[1, t] += w[x] * eta[x] * Wt[k]
I_lp, B_lp = M.mi_bayes(J)
Ip, Bp = M.mi_bayes(np.vstack([w * (1 - eta), w * eta]))
# posterior support diameter of every used output
diam = max(a[j - 1] - a[i] for t, (i, j) in enumerate(wins) if J[:, t].sum() > 1e-15)
print(f"LP status {res.status}; Bacc(p)={Bp:.5f}  min Bacc over randomised admissible channels={B_lp:.5f}"
      f" (re-check {J.max(0).sum():.5f}); loss {Bp - B_lp:.2e} <= L D = {L * DD:.2e}; max posterior-support"
      f" width {diam:.6f} <= D = {DD:.6f}")

print("\n== R6.c  without smoothness: one admissible bin removes ALL information (pathological law) ==")
h = 0.0005
a = np.linspace(0.9, 0.904, 4001)[:-1]
s = (np.floor((a - 0.9) / h + 1e-9) % 2).astype(float)       # S is a deterministic, alternating function of p
w = np.full(a.size, 1 / a.size)
Ip, Bp = M.mi_bayes(np.vstack([w * (1 - s), w * s]))
c_ = 0.9
Lc, Rc = M.bin_interval(c_)
q = np.array([c_, 1 - c_])
allok = all(M.g_ok(np.array([x, 1 - x]), q, 0) for x in a)
It, Bt = M.mi_bayes(np.array([[np.sum(w * (1 - s))], [np.sum(w * s)]]))
print(f"I(S;p) = {Ip:.4f} (log 2 = {math.log(2):.4f}), Bacc(p) = {Bp:.3f}; ONE token q = (0.9, 0.1) admissible for all"
      f" {a.size} inputs: {allok} -> I(S;T) = {It:.4f}, Bacc(T) = {Bt:.3f}")

print("\n== R6.d  without the Markov assumption: admissibility does not bound leakage from above ==")
# p constant (a = 0.9), S independent of p (I(S;p) = 0); the release picks one of two admissible q's BY SEX
p = np.array([0.9, 1 - 0.9])
q1 = np.array([0.9, 1 - 0.9])
q2 = np.array([0.9004, 1 - 0.9004])
print(f"q1, q2 admissible for p: {M.g_ok(p, q1, 0)}, {M.g_ok(p, q2, 0)}; T = q_(S+1): I(S;p) = 0, I(S;T) = log 2 ="
      f" {math.log(2):.4f}, Bacc 1.0")
print("Robust (non-Markov) Bayes-accuracy lower bound, PROVED in MATH_REVIEW R6: Bacc(S|T) >= Bacc(S|p) -"
      " P(|eta(p) - 1/2| <= L D).")
