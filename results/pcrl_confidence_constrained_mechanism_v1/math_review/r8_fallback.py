"""R8 -- the disclosed fallback. Ucal(p) satisfies G iff Ucal(p) has a strict top (float64 included); it discloses p;
the fallback flag alone can leak SEX; the flag is recoverable from the output itself; the choice rule among several
admissible registered representatives changes leakage and must be fixed. Synthetic only.
Run: python -P r8_fallback.py (about 5 s)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(8)

print("== R8.a  q = p: NLL and Brier hold exactly in float64; strict Class iff strict top ==")
bad_nll = bad_br = 0
for t in range(200000):
    K = int(rng.choice([2, 6]))
    p = rng.dirichlet(np.full(K, float(rng.choice([0.05, 1.0]))))
    if t % 10 == 0:
        p[rng.integers(K)] = 5e-324 * rng.integers(0, 3)
    bad_nll += int(not M.nll_ok(p, p))
    bad_br += int(not M.brier_ok(p, p))
print(f"NLL failures {bad_nll}, Brier failures {bad_br} (fl(e^-d p) <= p since e^-d < 1; excess is exactly 0)")
p = np.array([0.4, 0.4, 0.2])
print(f"tied top {p.tolist()}: strict Class fails -> fallback NOT admissible (see R3 for tempering-created ties)")

print("\n== R8.b  the fallback flag alone leaks SEX ==")
# four inputs, equal mass. x1, x2 covered by one registered token (one admissible bin); x3, x4 fall back.
px = np.full(4, 0.25)
eta = np.array([0.3, 0.3, 0.7, 0.7])
flag = np.array([0, 0, 1, 1])
J = np.zeros((2, 2))
for x in range(4):
    J[1, flag[x]] += px[x] * eta[x]
    J[0, flag[x]] += px[x] * (1 - eta[x])
I_flag, B_flag = M.mi_bayes(J)
print(f"I(S; flag) = {I_flag:.4f} nats, Bayes accuracy from the flag alone = {B_flag:.3f} (prior 0.5): tokens that carry"
      " no SEX information do not make the release private if coverage correlates with SEX")
J2 = np.zeros((2, 3))
for x, t in zip(range(4), (0, 0, 1, 2)):   # covered token, fallback p(x3), fallback p(x4)
    J2[1, t] += px[x] * eta[x]
    J2[0, t] += px[x] * (1 - eta[x])
print(f"with the fallback vectors (distinct p(x3), p(x4)): I(S; T) = {M.mi_bayes(J2)[0]:.4f} >= I(S; flag)")

print("\n== R8.c  the flag is recoverable from the output: fallback outputs are generically not registered vectors ==")
reg = [np.array([0.7, 0.3]), np.array([0.9, 0.1])]
outs = [reg[0], np.array([0.62345, 0.37655]), reg[1], np.array([0.80011, 0.19989])]
print("recipient's test 'output in registered set':", [any(np.array_equal(o, r) for r in reg) for o in outs],
      "-> equals NOT flag, except when a fallback p coincides exactly with a registered q (then q is admissible for p,"
      " so the input would not have fallen back)")

print("\n== R8.d  the choice among several admissible registered representatives changes leakage ==")
# inputs a = p_0 (binary), registered representatives c1, c2 with overlapping member intervals
c1, c2 = 0.900, 0.9045
I1, I2 = M.bin_interval(c1), M.bin_interval(c2)
a = np.array([0.8998, 0.9042, 0.9070])
mass = np.array([1 / 3, 1 / 3, 1 / 3])
eta = np.array([0.2, 0.8, 0.8])
adm = [[I1[0] <= x <= I1[1], I2[0] <= x <= I2[1]] for x in a]
print("admissible (t1, t2) per input:", adm)
for name, rule in (("first registered", lambda r: r.index(True)), ("last registered", lambda r: len(r) - 1 -
                                                                   r[::-1].index(True))):
    J = np.zeros((2, 2))
    for x in range(3):
        t = rule(adm[x])
        J[1, t] += mass[x] * eta[x]
        J[0, t] += mass[x] * (1 - eta[x])
    I, Bacc = M.mi_bayes(J)
    print(f"rule '{name}': I(S;T) = {I:.4f}, Bayes acc = {Bacc:.3f}")
