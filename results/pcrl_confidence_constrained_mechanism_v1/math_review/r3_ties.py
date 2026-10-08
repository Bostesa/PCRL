"""R3 -- strict class preservation, the source tie rule and exact ties: numpy argmax first index; tempering under the
frozen log-input rule preserves ties and can CREATE exact top ties from strictly ordered U0 (a few ulps apart); then
Ucal(p) itself fails the strict Class condition, so the fallback 'release Ucal(p), which satisfies G trivially' is false
for that row. The nudge q = (1 - eta) p + eta e_d is admissible for every p and 0 < eta <= eta0. Synthetic only.
Run: python -P r3_ties.py (about 10 s)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(3)

print("== R3.a  source tie rule ==")
p = np.array([0.4, 0.4, 0.2])
print(f"numpy argmax {p.tolist()} -> {np.argmax(p)} (first index). Strict Class for q = p: {M.class_ok(p, 0)}")

print("\n== R3.b  tempering (frozen text rule) preserves exact ties and can create them ==")
for a in (0.5, 1.0, 1.7):
    u = M.ucal_text_rule(p, a)
    print(f"alpha={a}: Ucal={u.tolist()}  tie kept: {u[0] == u[1]}")
created = rev = n = 0
example = None
for t in range(100000):
    K = 6 if t % 2 else 3
    a = float(rng.uniform(0.25, 4.0))
    if a == 1.0:
        continue
    base = rng.dirichlet(np.ones(K))
    i = int(base.argmax())
    j = int((i + 1 + rng.integers(K - 1)) % K)
    k = [m for m in range(K) if m not in (i, j)][0]
    q0 = base.copy()
    delta = (base[i] - int(rng.integers(1, 4)) * np.spacing(base[i])) - q0[j]   # 1-3 ulps below the top
    q0[j] += delta
    q0[k] -= delta                                                               # keep the vector normalised
    if q0[k] <= 0 or abs(q0.sum() - 1.0) > 1e-15:
        continue
    d = M.decision(q0)
    u = M.ucal_text_rule(q0, a)
    n += 1
    s = np.sort(u)[::-1]
    if s[0] == s[1]:
        if M.decision(u) == d:          # hcal's check (numpy argmax == d) PASSES, strict Class FAILS
            created += 1
            if example is None and K == 6:
                example = (a, q0.tolist(), u.tolist(), d)
        else:
            rev += 1                    # hcal would raise (argmax(u) != d)
print(f"normalised near-tie probes {n}: strict-U0 -> exactly tied Ucal top with argmax == d (passes the hcal check,"
      f" fails strict Class) {created}; tied with argmax != d (hcal raises) {rev}")
print("K=6 example (alpha, U0, Ucal, d):", example)
if example is not None:
    a, q0, u, d = example
    u = np.array(u)
    print(f"  U0 strict top? {np.sort(q0)[-1] > np.sort(q0)[-2]}; fallback q = Ucal(p) admissible under G?"
          f" {M.g_ok(u, u, d)}  (NLL {M.nll_ok(u, u)}, Brier {M.brier_ok(u, u)}, strict Class {M.class_ok(u, d)})")
for k_ in range(1, 6):                 # binary: the top two are p0 and 1 - p0, never within a few ulps
    p0 = 0.5 + k_ * 2.0 ** -53
    assert all(M.ucal_text_rule(np.array([p0, 1 - p0]), float(a_))[0] > M.ucal_text_rule(
        np.array([p0, 1 - p0]), float(a_))[1] for a_ in np.arange(0.25, 4.0, 0.05) if a_ != 1.0)
print("binary (income): no tie created for p0 = 1/2 + k 2^-53, k = 1..5, alpha grid -> ties need an exact U0 tie"
      " (p0 = 1/2)")

print("\n== R3.c  nudge q = (1-eta) p + eta e_d is G-admissible for all p (incl. ties) when 0 < eta <= eta0 ==")
eta0 = (math.sqrt(1 + 8 * M.B) - 1) / 4      # from max_y excess <= 2 eta^2 + eta (PROVED in MATH_REVIEW R3)
print(f"eta0 = (sqrt(1+8b)-1)/4 = {eta0:.7f}; NLL needs eta <= 1-e^-d = {1 - M.EMD:.7f}")
bad = 0
worst = -1.0
for t in range(200000):
    K = int(rng.choice([2, 3, 6]))
    p = rng.dirichlet(np.full(K, float(rng.choice([0.05, 1.0, 20.0]))))
    if t % 4 == 0:      # force an exact top tie
        i, j = rng.choice(K, 2, replace=False)
        p[j] = p[i] = max(p[i], p[j])
        p = p / p.sum()
    d = M.decision(p)
    for eta in (1e-9, 1e-6, eta0 * 0.999):
        q = (1 - eta) * p
        q[d] += eta
        ok = M.g_ok(p, q, d)
        worst = max(worst, M.brier_excess(p, q).max() / M.B)
        bad += int(not ok)
print(f"failures {bad} over 600000 (p, eta) cases; worst Brier excess / b = {worst:.4f}")

print("\n== R3.d  within a same-decision bin, M_d >= M_k for every k (so strict Class is never blocked by the maxima) ==")
bad = 0
for t in range(20000):
    K = 6
    ctr = rng.dirichlet(np.ones(K))
    P = np.abs(ctr + rng.normal(0, 0.05, (4, K)))
    P = P / P.sum(1, keepdims=True)
    ds = [M.decision(r) for r in P]
    if len(set(ds)) != 1:
        continue
    Mx = P.max(0)
    bad += int(np.any(Mx > Mx[ds[0]]))
print(f"violations {bad} (PROVED: the member attaining M_k has p_d >= p_k = M_k)")
