"""R9 -- oracle formulations. Decision-only is confidence-ineligible; the decision is a function of EVERY admissible
output, so decision-only is a leakage FLOOR for every admissible arm (local and coalition); CLASS eligibility closed
form; merging never increases MI or Bayes accuracy; for the stochastic arm the outputs may WLOG be the maximal
admissible sets (finite exact LP), randomisation can strictly beat every deterministic partition, and a per-class
token cap makes the problem a cardinality-constrained (mixed-integer) program. Synthetic only.
Run: python -P r9_oracles.py (about 10 s)."""
import itertools
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mr as M  # noqa: E402

rng = np.random.default_rng(99)

print("== R9.a  decision-only: the one-hot e_d violates NLL whenever some p_k > 0, k != d ==")
p = np.array([0.7, 0.2, 0.1])
print(f"p={p.tolist()}, q=e_0: NLL {M.nll_ok(p, np.eye(3)[0])}; a release with no vector is outside G's domain ->"
      " confidence-INELIGIBLE (premise check holds)")

print("\n== R9.b  floor: argmax of every admissible output = d(p) => I(S;T) >= I(S;d), Bacc(S|T) >= Bacc(S|d) ==")
viol = 0
for t in range(3000):
    n = 8
    a = np.sort(rng.uniform(0.0, 1.0, n))         # binary p_0; decision 0 iff a >= 1/2
    px = rng.dirichlet(np.ones(n))
    eta = rng.uniform(0, 1, n)
    dec = (a < 0.5).astype(int)
    Jd = np.zeros((2, 2))
    for x in range(n):
        Jd[1, dec[x]] += px[x] * eta[x]
        Jd[0, dec[x]] += px[x] * (1 - eta[x])
    Id, Bd = M.mi_bayes(Jd)
    # a random admissible deterministic partition: merge neighbours of the same class when the pair interval fits
    lab, cur = [], 0
    for x in range(n):
        if x and dec[x] == dec[x - 1] and abs(a[x] - a[x - 1]) <= 0.002 and rng.random() < 0.7:
            lab.append(cur)
        else:
            cur += 1
            lab.append(cur)
    J = np.zeros((2, cur + 1))
    for x in range(n):
        J[1, lab[x]] += px[x] * eta[x]
        J[0, lab[x]] += px[x] * (1 - eta[x])
    It, Bt = M.mi_bayes(J)
    viol += int(It < Id - 1e-12 or Bt < Bd - 1e-12)
print(f"violations {viol} (PROVED: d = argmax(T) is a function of T; data processing; coalition: (d1, d2) is a"
      " function of (T1, T2))")

print("\n== R9.c  CLASS eligibility: whole class must be one admissible bin ==")
for name, P in (("class spread 0.003 (K=2)", np.array([[0.900, 0.100], [0.903, 0.097]])),
                ("class spread 0.30 (K=2)", np.array([[0.60, 0.40], [0.90, 0.10]]))):
    SM = P.max(0).sum()
    val, q, ok = M.joint_minimax(P, 0)
    print(f"{name}: sum M = {SM:.4f} (<= e^d {SM <= M.ED}); joint min-max Brier excess {val:.5f} -> eligible"
          f" {SM <= M.ED and val <= M.B}")
print("binary: CLASS eligible iff a_max - a_min <= W at the best c, i.e. at most e^d - 1 = 0.0050 (R5.b)")

print("\n== R9.d  merging two outputs never increases MI or Bayes accuracy ==")
bad = 0
for t in range(20000):
    J = rng.dirichlet(np.ones(10)).reshape(2, 5)
    I0, B0 = M.mi_bayes(J)
    i, j = rng.choice(5, 2, replace=False)
    Jm = np.delete(J, j, axis=1)
    Jm[:, i if i < j else i - 1] += J[:, j]
    I1, B1 = M.mi_bayes(Jm)
    bad += int(I1 > I0 + 1e-12 or B1 > B0 + 1e-12)
print(f"violations {bad} (data processing; max_s(u_s+v_s) <= max_s u_s + max_s v_s)")

print("\n== R9.e  stochastic arm: outputs WLOG = maximal admissible sets; randomisation can beat every partition ==")
# 3 inputs on a line: {x1,x2} and {x2,x3} admissible, {x1,x2,x3} not (binary, members a)
a = np.array([0.9000, 0.9040, 0.9080])
c12, c23 = 0.9000, 0.9040
adm12 = all(M.bin_interval(c12)[0] <= x <= M.bin_interval(c12)[1] for x in a[:2])
adm23 = all(M.bin_interval(c23)[0] <= x <= M.bin_interval(c23)[1] for x in a[1:])
adm123 = (a[2] - a[0]) <= M.ED - 1
print(f"admissible: {{x1,x2}} {adm12}, {{x2,x3}} {adm23}, {{x1,x2,x3}} {adm123} (TV {a[2]-a[0]:.4f} > e^d - 1)")
best_gap, best_case = 0.0, None
for t in range(20000):
    px = rng.dirichlet(np.ones(3))
    eta = rng.uniform(0, 1, 3)
    P1 = px * eta
    P0 = px * (1 - eta)
    sets = [(0, 1), (1, 2), (0,), (1,), (2,)]          # maximal sets plus singletons (for the partitions)
    parts = [[(0, 1), (2,)], [(0,), (1, 2)], [(0,), (1,), (2,)]]

    def bacc(part):
        return sum(max(sum(P0[x] for x in b), sum(P1[x] for x in b)) for b in part)
    det = min(bacc(pt) for pt in parts)
    # LP over outputs = maximal sets {x1,x2}, {x2,x3}: variables W(t|x) for x in t, then z_t
    var = [(0, 0), (1, 0), (1, 1), (2, 1)]
    c = np.r_[np.zeros(4), 1, 1]
    A = np.zeros((4, 6))
    for k, (x, tt) in enumerate(var):
        A[2 * tt, k] = P0[x]
        A[2 * tt + 1, k] = P1[x]
    A[0, 4] = A[1, 4] = -1
    A[2, 5] = A[3, 5] = -1
    Aeq = np.zeros((3, 6))
    for k, (x, tt) in enumerate(var):
        Aeq[x, k] = 1
    r = linprog(c, A_ub=A, b_ub=np.zeros(4), A_eq=Aeq, b_eq=np.ones(3), bounds=(0, None), method="highs")
    W = r.x[:4]
    J = np.zeros((2, 2))
    for k, (x, tt) in enumerate(var):
        J[0, tt] += P0[x] * W[k]
        J[1, tt] += P1[x] * W[k]
    sto = M.mi_bayes(J)[1]                              # independent re-check of the LP value
    if abs(sto - r.fun) > 1e-9:
        print("LP re-check mismatch", sto, r.fun)
    if det - sto > best_gap:
        best_gap, best_case = det - sto, (px.round(4).tolist(), eta.round(4).tolist(), W.round(4).tolist(), det, sto)
print(f"largest Bayes-accuracy gap deterministic - stochastic: {best_gap:.4f}; case (P(x), eta, W(t|x), det, sto):"
      f" {best_case}")
print("WLOG maximal sets (PROVED): relabel each output with a maximal admissible set containing its support set, then"
      " merge outputs with equal labels (merging cannot increase Bayes accuracy or MI).")
print("Capacity: a cap of C outputs per class is a cardinality constraint (|{t : W(t|.) > 0}| <= C): exact only by"
      " enumerating output subsets of size <= C with one LP each (or a MILP); the LP alone is exact iff the number of"
      " maximal admissible sets per class is <= C.")
