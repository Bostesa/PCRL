"""Mathematical-review reproduction script (pure synthetic laws, no ACS data).

Run from the worktree root:
    python3 results/pcrl_shared_context_release_v1/agents/math/reproduce_examples.py

Sections
  A  Example A (committed fixture_b1) re-derived exactly, plus nullspace test,
     private-channel vertex enumeration, and its representation as a K=1
     two-policy mixture.
  B  Example B (committed fixture_b2) re-derived exactly.
  N  Nested parameterization: exact (Fraction) checks of row sums, affinity,
     T32 embedding at eta=0, compact reduction at eta=1, linear loss replay,
     both D17 witnesses, and the D17 alias direction.
  R  Degrees of freedom and identifiable rank on a 32x17, K=4, M=6 instance.
  C  Counterexample: a single shared eta with K=1 loses the whole richer-policy
     gain; D17-as-a-column does not fix it; aligned contexts or a D17-anchored
     (switched) column do.
  V  LP vertex structure: one binding cut -> time-sharing of two
     deterministic releases; the same kernel at eta=1 and eta=3/4 (aliasing).
  M  Nonalias metric, task-sufficiency invariance, fresh-draw failure.
  S  Stale frozen attackers overstate mixture privacy; concavity gives a
     conservative component-refit lower bound.
  P  Rebasing: adding a stronger attacker can make an infeasible candidate
     feasible.
"""
from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.pcrl_adaptive_release_v1 import refinement  # noqa: E402


def mi(joint):
    """Mutual information (nats) of an exact finite joint {(a,b): Fraction}."""
    pa, pb = {}, {}
    for (a, b), p in joint.items():
        pa[a] = pa.get(a, 0) + p
        pb[b] = pb.get(b, 0) + p
    return sum(float(p) * math.log(float(p / (pa[a] * pb[b])))
               for (a, b), p in joint.items() if p)


def h(p):
    p = float(p)
    return -sum(x * math.log(x) for x in (p, 1 - p) if x > 0)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------- Example A
section("A. Example A (fixture_b1), exact")
m = tuple(map(F, ("2/5", "1/10", "1/10", "2/5")))
S = (0, 1, 0, 1)
ty = tuple(map(F, ("1/10", "1/10", "9/10", "9/10")))
T = tuple(0 if p == F(1, 10) else 1 for p in ty)  # T32 analog: 2 states
assert sum(m) == 1
pS = [sum(m[i] for i in range(4) if S[i] == s) for s in (0, 1)]
print("P(S=0), P(S=1) =", pS)

# T-only privacy: P(T=lo|S=s)
pT_S = [[sum(m[i] for i in range(4) if S[i] == s and T[i] == t) / pS[s] for t in (0, 1)]
        for s in (0, 1)]
det = pT_S[0][0] * pT_S[1][1] - pT_S[0][1] * pT_S[1][0]
print("P(T|S) =", pT_S, " det =", det)
print("=> for ANY token alphabet, P(Z|S=0)=P(Z|S=1) iff (P(T=lo|S=0)-P(T=lo|S=1))"
      "(q_lo-q_hi)=0 iff q_lo=q_hi (factor", pT_S[0][0] - pT_S[1][0], "!= 0);"
      " then Z is independent of X, so I(Y;Z)=0.")

q1 = tuple(map(F, ("0", "0", "1", "1/4")))
pz1_s = [sum(m[i] * q1[i] for i in range(4) if S[i] == s) / pS[s] for s in (0, 1)]
joint_yz, joint_sz = {}, {}
for i in range(4):
    for z, pz in ((0, 1 - q1[i]), (1, q1[i])):
        for y, py in ((0, 1 - ty[i]), (1, ty[i])):
            joint_yz[y, z] = joint_yz.get((y, z), 0) + m[i] * pz * py
        joint_sz[S[i], z] = joint_sz.get((S[i], z), 0) + m[i] * pz
info_rich = mi(joint_yz)
print("richer channel P(Z=1|X) =", q1, " P(Z=1|S=0), P(Z=1|S=1) =", pz1_s)
print(f"I(S;Z) = {mi(joint_sz):.3e} (exact independence: {pz1_s[0] == pz1_s[1]})")
print(f"I(Y;Z) = {info_rich:.11f} nats")


def rgs(n):
    def walk(prefix, hi):
        if len(prefix) == n:
            yield prefix
        else:
            for lab in range(hi + 2):
                yield from walk(prefix + (lab,), max(hi, lab))
    yield from walk((0,), 0)


parts = list(rgs(4))
private = []
for p in parts:
    if all(sum(m[i] for i in range(4) if S[i] == 0 and p[i] == z) / pS[0]
           == sum(m[i] for i in range(4) if S[i] == 1 and p[i] == z) / pS[1]
           for z in set(p)):
        jy = {}
        for i in range(4):
            for y, py in ((0, 1 - ty[i]), (1, ty[i])):
                jy[y, p[i]] = jy.get((y, p[i]), 0) + m[i] * py
        private.append((p, mi(jy)))
print(f"deterministic partitions enumerated: {len(parts)}; exactly private: {len(private)}")
for p, v in private:
    print(f"   private partition {p}: I(Y;Z) = {v:.3e}")

# Rassouli-Gunduz Proposition 1 (general observation W): feasible iff some v in
# Null(P_{S|W}) is not in Null(P_{Y|W}).
v = (1, 0, -1, 0)
PS_X = [[1 if S[i] == s else 0 for i in range(4)] for s in (0, 1)]
PY_X = [[1 - ty[i] for i in range(4)], [ty[i] for i in range(4)]]
print("RG Prop.1, W=X: v=(1,0,-1,0): P_{S|X}v =",
      [sum(r[i] * v[i] for i in range(4)) for r in PS_X],
      " P_{Y|X}v =", [sum(r[i] * v[i] for i in range(4)) for r in PY_X], "-> feasible")
print("RG Prop.1, W=T: P_{S|T} is 2x2 with det", det, "-> Null={0} -> infeasible")

# Binary private channels: box [0,1]^4 with one linear equality. Vertices
# have <=1 fractional coordinate (vertex counting); I(Y;Z) is convex in the
# channel so its max over the polytope is at a vertex.
coef = [m[i] / pS[S[i]] * (1 if S[i] == 0 else -1) for i in range(4)]
verts = set()
for free in range(4):
    others = [j for j in range(4) if j != free]
    for bits in itertools.product((0, 1), repeat=3):
        rhs = -sum(coef[j] * b for j, b in zip(others, bits))
        val = rhs / coef[free]
        if 0 <= val <= 1:
            qv = [None] * 4
            qv[free] = val
            for j, b in zip(others, bits):
                qv[j] = F(b)
            verts.add(tuple(qv))


def task_info(qv):
    jy = {}
    for i in range(4):
        for z, pz in ((0, 1 - qv[i]), (1, qv[i])):
            for y, py in ((0, 1 - ty[i]), (1, ty[i])):
                jy[y, z] = jy.get((y, z), 0) + m[i] * pz * py
    return mi(jy)


best = max(verts, key=task_info)
print(f"private binary-channel polytope: {len(verts)} vertices; "
      f"max I(Y;Z) = {task_info(best):.11f} at {tuple(str(x) for x in best)}")
print("   (the fixture channel (0,0,1,1/4) is a vertex with ONE fractional row)")
assert q1 in verts and abs(task_info(best) - info_rich) < 1e-15

# K=1 compact mixture of two deterministic policies reproduces q1.
d17 = (0, 0, 1, 1)          # T-only deterministic ("D17 analog")
dnew = (0, 0, 1, 0)         # richer deterministic policy 1{x=2}
mix = tuple(F(1, 4) * d17[i] + F(3, 4) * dnew[i] for i in range(4))
assert mix == q1
for name, d in (("D17-analog", d17), ("d_new", dnew)):
    ps = [sum(m[i] * d[i] for i in range(4) if S[i] == s) / pS[s] for s in (0, 1)]
    print(f"component {name}: P(Z=1|S=0,1) = {ps} -> private: {ps[0] == ps[1]}")
print("1/4*D17-analog + 3/4*d_new == (0,0,1,1/4): True  (neither component is private)")

fb1 = refinement.fixture_b1()
assert fb1["private_stochastic_z1"] == q1 and fb1["p_z1_given_s"] == tuple(pz1_s)
assert fb1["partitions_enumerated"] == 15 and fb1["deterministic_private_partitions"] == 2
assert abs(fb1["stochastic_task_information_nats"] - info_rich) < 1e-15
print("committed fixture_b1 agrees (note: its 't_only_private_implies_row_equality'"
      " field is a hard-coded constant, re-derived above).")

# ---------------------------------------------------------------- Example B
section("B. Example B (fixture_b2), exact")
rows = []  # (h, t, s, y, mass)
for t in (0, 1):
    for s in (0, 1):
        rows.append((0, t, s, t, F(1, 8)))
    rows.append((1, t, t, 0, F(1, 4)))
assert sum(r[-1] for r in rows) == 1


def cond_info(zmap, which):
    tot = 0.0
    for hh in (0, 1):
        j = {}
        ph = sum(r[-1] for r in rows if r[0] == hh)
        for (h_, t, s, y, w) in rows:
            if h_ == hh:
                key = (y if which == "y" else s, zmap(h_, t))
                j[key] = j.get(key, 0) + w / ph
        tot += float(ph) * mi(j)
    return tot


ctx = lambda h_, t: t if h_ == 0 else 0
print(f"contextual Z = T if H=0 else 0: I(Y;Z|H) = {cond_info(ctx, 'y'):.14f} "
      f"(log2/2 = {math.log(2) / 2:.14f}); I(S;Z|H) = {cond_info(ctx, 's'):.1e}")
print("context-blind q(z|T): at H=1, S=T, so P(Z|S=s,H=1)=q(.|T=s); privacy forces"
      " q(.|0)=q(.|1) -> Z independent of (H,T) -> I(Y;Z|H)=0.")
print("nested representation: hard contexts phi_k=1{H=k}, bank {Z=T, Z=0},"
      " eta=1, A[0,Z=T]=1, A[1,Z=0]=1 -> deterministic.")
fb2 = refinement.fixture_b2()
assert abs(fb2["conditional_task_information_nats"] - math.log(2) / 2) < 1e-15
assert fb2["exact_sensitive_independence"] is True
print("committed fixture_b2 agrees (its 'context_blind_private_implies_constant'"
      " field is hard-coded True; re-derived above).")

# ---------------------------------------------------- nested parameterization
section("N. Nested parameterization: exact Fraction checks")
rng = np.random.default_rng(20260924)
nT, nZ, K, M, N = 4, 3, 2, 3, 12
Tx = [i % nT for i in range(N)]
d17_rows = [int(rng.integers(nZ)) for _ in range(nT)]
pol = [[d17_rows[Tx[i]] for i in range(N)]]  # column 0 = D17 exactly
for _ in range(M - 1):
    pol.append([int(rng.integers(nZ)) for _ in range(N)])
phi = []
for i in range(N):
    a = F(int(rng.integers(1, 5)), 1)
    b = F(int(rng.integers(0, 5)), 1)
    phi.append((a / (a + b), b / (a + b)))  # soft, exact, sums to 1


def rand_simplex(k, total):
    raw = [F(int(rng.integers(0, 7))) for _ in range(k)]
    raw[0] += 1
    s = sum(raw)
    return [x * total / s for x in raw]


def emit(B, A):
    q = []
    for i in range(N):
        row = list(B[Tx[i]])
        for k in range(K):
            for mm in range(M):
                row[pol[mm][i]] += phi[i][k] * A[k][mm]
        q.append(row)
    return q


def params(eta):
    B = [rand_simplex(nZ, 1 - eta) for _ in range(nT)]
    A = [rand_simplex(M, eta) for _ in range(K)]
    return B, A


ok_rows = True
for eta in (F(0), F(1, 3), F(1)):
    B, A = params(eta)
    q = emit(B, A)
    ok_rows &= all(sum(r) == 1 and min(r) >= 0 for r in q)
print("row sums exactly 1 and nonnegative at eta in {0,1/3,1}:", ok_rows)

(B1, A1), (B2, A2) = params(F(1, 5)), params(F(4, 5))
lam = F(2, 7)
Bm = [[lam * x + (1 - lam) * y for x, y in zip(r1, r2)] for r1, r2 in zip(B1, B2)]
Am = [[lam * x + (1 - lam) * y for x, y in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
qa, qb, qm = emit(B1, A1), emit(B2, A2), emit(Bm, Am)
aff = all(qm[i][z] == lam * qa[i][z] + (1 - lam) * qb[i][z] for i in range(N) for z in range(nZ))
etam = lam * F(1, 5) + (1 - lam) * F(4, 5)
feas = all(sum(r) == 1 - etam for r in Bm) and all(sum(r) == etam for r in Am)
print("affine: mixture of parameters (eta=1/5, 4/5) is feasible with eta="
      f"{etam} and emits the mixture of kernels: {aff and feas}")

K32 = [rand_simplex(nZ, 1) for _ in range(nT)]
q0 = emit(K32, [[F(0)] * M for _ in range(K)])
print("eta=0 embeds an arbitrary T-kernel exactly:",
      all(q0[i] == K32[Tx[i]] for i in range(N)))

w = [F(1, N)] * N
loss = [[F(int(rng.integers(1, 20)), 10) for _ in range(nZ)] for _ in range(N)]
B, A = params(F(2, 5))
q = emit(B, A)
L_person = sum(w[i] * sum(q[i][z] * loss[i][z] for z in range(nZ)) for i in range(N))
cB = [[sum(w[i] * loss[i][z] for i in range(N) if Tx[i] == t) for z in range(nZ)] for t in range(nT)]
cA = [[sum(w[i] * phi[i][k] * loss[i][pol[mm][i]] for i in range(N)) for mm in range(M)] for k in range(K)]
L_coef = (sum(B[t][z] * cB[t][z] for t in range(nT) for z in range(nZ))
          + sum(A[k][mm] * cA[k][mm] for k in range(K) for mm in range(M)))
print("expected-token loss replay: person-level == sum B*cB + sum A*cA exactly:", L_person == L_coef)

d17_q = [[F(int(z == d17_rows[Tx[i]])) for z in range(nZ)] for i in range(N)]
wB = emit([[F(int(z == d17_rows[t])) for z in range(nZ)] for t in range(nT)],
          [[F(0)] * M for _ in range(K)])
wA = emit([[F(0)] * nZ for _ in range(nT)], [[F(1)] + [F(0)] * (M - 1) for _ in range(K)])
print("D17 witness (B=D17,A=0,eta=0) emits D17 exactly:", wB == d17_q)
print("D17 witness (A[k,D17]=1,B=0,eta=1), SOFT contexts, emits D17 exactly:", wA == d17_q)
# alias direction: dB[t,D17(t)]=-1, dA[k,D17]=+1, deta=+1
dq_zero = True
for i in range(N):
    dq = [F(0)] * nZ
    dq[d17_rows[Tx[i]]] -= 1
    dq[pol[0][i]] += sum(phi[i][k] for k in range(K))
    dq_zero &= all(x == 0 for x in dq)
print("alias direction (move mass B->A on the D17 token, eta+=eps) leaves q unchanged:", dq_zero)

# --------------------------------------------------------- rank / DOF count
section("R. Degrees of freedom and identifiable rank (float, synthetic)")


def rank_report(nT, nZ, K, M, N, t32_cols, seed):
    g = np.random.default_rng(seed)
    Tx = g.integers(nT, size=N)
    d17 = g.integers(nZ, size=nT)
    cols = [d17[Tx]]
    for _ in range(t32_cols):              # extra T32-measurable columns
        cols.append(g.integers(nZ, size=nT)[Tx])
    while len(cols) < M:
        cols.append(g.integers(nZ, size=N))
    ph = g.dirichlet(np.ones(K), size=N) if K > 1 else np.ones((N, 1))
    nb, na = nT * nZ, K * M
    nv = nb + na + 1
    E = np.zeros((nT + K, nv))
    for t in range(nT):
        E[t, t * nZ:(t + 1) * nZ] = 1
        E[t, -1] = 1                       # sum_z B[t,z] + eta = 1
    for k in range(K):
        E[nT + k, nb + k * M: nb + (k + 1) * M] = 1
        E[nT + k, -1] = -1                 # sum_m A[k,m] - eta = 0
    tangent = null_space(E)
    Q = np.zeros((N * nZ, nv))
    for i in range(N):
        for z in range(nZ):
            Q[i * nZ + z, Tx[i] * nZ + z] = 1
        for k in range(K):
            for mm in range(M):
                Q[i * nZ + cols[mm][i], nb + k * M + mm] += ph[i, k]
    dim = tangent.shape[1]
    rank = np.linalg.matrix_rank(Q @ tangent, tol=1e-9)
    return dim, rank


for (K_, M_, extra) in ((1, 6, 0), (4, 6, 0), (4, 6, 1)):
    dim, rank = rank_report(32, 17, K_, M_, 4000, extra, 7)
    print(f"32x17, K={K_}, M={M_} (D17 column + {extra} extra T32-measurable column): "
          f"polytope dim = {dim} (=512+K(M-1)+1={512 + K_ * (M_ - 1) + 1}); "
          f"rank of map to q on rows = {rank}; alias dim = {dim - rank}")

# ---------------------------------------------------- shared-eta counterexample
section("C. Counterexample: single shared eta at K=1 loses the richer-policy gain")
# states t1={x0,x1}, t2={x2,x3}; tokens {0,1,2}; masses 1/4; frozen costs
Tc = (0, 0, 1, 1)
cost = [(F(0), F(1, 2), F(1, 2)), (F(0), F(1, 2), F(1, 2)),
        (F(2, 5), F(0), F(1)), (F(2, 5), F(1), F(0))]
wc = [F(1, 4)] * 4
D17c = (0, 0, 0, 0)          # rowwise T-minimizer (row costs t2: 2/5 vs 1/2 vs 1/2)
NEWc = (1, 2, 1, 2)          # richer policy: good on t2, bad on t1
SWc = (0, 0, 1, 2)           # D17-anchored switched policy
Cpol = lambda d: sum(wc[i] * cost[i][d[i]] for i in range(4))
t32_det = min(sum(wc[i] * cost[i][r[Tc[i]]] for i in range(4))
              for r in itertools.product(range(3), repeat=2))


def nested_value(bank, contexts):
    """Exact LP value: vertices of the Cayley polytope are the vertices of the
    eta=0 face (deterministic T-kernels) and of the eta=1 face (hard-context
    switched policies)."""
    Kc = max(contexts) + 1
    comp = min(sum(wc[i] * cost[i][bank[sel[contexts[i]]][i]] for i in range(4))
               for sel in itertools.product(range(len(bank)), repeat=Kc))
    return min(t32_det, comp)


print("C(D17) =", Cpol(D17c), " C(d_new) =", Cpol(NEWc), " best deterministic T-kernel =", t32_det)
print("nested, K=1, bank {D17,d_new}          :", nested_value([D17c, NEWc], (0, 0, 0, 0)))
print("nested, K=2 aligned with T, same bank  :", nested_value([D17c, NEWc], (0, 0, 1, 1)))
print("nested, K=2 contexts CROSSING T (x0,x2|x1,x3):", nested_value([D17c, NEWc], (0, 1, 0, 1)))
print("nested, K=1, bank {D17,d_new,switched} :", nested_value([D17c, NEWc, SWc], (0, 0, 0, 0)))
# confirm the K=1 shared-eta value with an LP solver over (B,A,eta)
nb = 2 * 3
c = [float(sum(wc[i] * cost[i][z] for i in range(4) if Tc[i] == t)) for t in range(2) for z in range(3)]
c += [float(Cpol(D17c)), float(Cpol(NEWc)), 0.0]
Aeq = [[1] * 3 + [0] * 3 + [0, 0, 1], [0] * 3 + [1] * 3 + [0, 0, 1], [0] * 6 + [1, 1, -1]]
res = linprog(c, A_eq=Aeq, b_eq=[1, 1, 0], bounds=[(0, None)] * 9, method="highs")
print(f"HiGHS LP over (B,A,eta), K=1: value {res.fun:.6f} (exact 1/5)")

# ------------------------------------------------------ LP vertex structure
section("V. One binding cut: LP optimum is time-sharing of two deterministic releases")
# Example-A law, T-states {x0,x1},{x2,x3}, binary tokens, bank {D17, d_new},
# K=1.  Constraint: exact S-independence (one linear equality) replaced by the
# frozen-attacker-free linear form P(Z=1|S=0)-P(Z=1|S=1)=0.  Objective:
# frozen decoder fitted on D17, P(Y=1|z)=(1/10,9/10).  Minimize task loss.
dec = (0.1, 0.9)
ell = [[-(float(ty[i]) * math.log(dec[z]) + (1 - float(ty[i])) * math.log(1 - dec[z]))
        for z in (0, 1)] for i in range(4)]
banks = [d17, dnew]
# variables: B[t,z] (4), A[m] (2), eta
cvec = [sum(float(m[i]) * ell[i][z] for i in range(4) if T[i] == t) for t in (0, 1) for z in (0, 1)]
cvec += [sum(float(m[i]) * ell[i][d[i]] for i in range(4)) for d in banks] + [0.0]
g = lambda i: float(m[i] / pS[S[i]]) * (1 if S[i] == 0 else -1)
priv = [sum(g(i) for i in range(4) if T[i] == t) if z == 1 else 0.0 for t in (0, 1) for z in (0, 1)]
priv += [sum(g(i) * d[i] for i in range(4)) for d in banks] + [0.0]
Aeq = [[1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1, -1], priv]
res = linprog(cvec, A_eq=Aeq, b_eq=[1, 1, 0, 0], bounds=[(0, None)] * 7, method="highs")
x = res.x
print("optimum: B =", np.round(x[:4], 6), " A =", np.round(x[4:6], 6), " eta =", round(x[6], 6))
qopt = [x[2 * T[i] + 1] + sum(x[4 + j] * banks[j][i] for j in range(2)) for i in range(4)]
print("emitted P(Z=1|x) =", np.round(qopt, 6), " (fixture (0,0,1,1/4))")
print(f"frozen-decoder task loss {res.fun:.6f} vs D17 {float(sum(m[i]*ell[i][d17[i]] for i in range(4))):.6f}"
      f" vs constant {float(sum(m[i]*ell[i][0] for i in range(4))):.6f}")
print(f"refit (Bayes) decoder on the optimum: H(Y|Z) = {h(0.5) - info_rich:.6f} "
      f"< frozen {res.fun:.6f}: decoder staleness is conservative for task loss")
# Same law, bank {d_new} only: the SAME emitted kernel is reached with eta=3/4.
Aeq2 = [[1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 1, -1],
        priv[:4] + [priv[5], 0.0]]
res2 = linprog(cvec[:4] + [cvec[5], 0.0], A_eq=Aeq2, b_eq=[1, 1, 0, 0],
               bounds=[(0, None)] * 6, method="highs")
x2 = res2.x
q2 = [x2[2 * T[i] + 1] + x2[4] * dnew[i] for i in range(4)]
print("bank {d_new} only: B =", np.round(x2[:4], 6), " A =", np.round(x2[4:5], 6),
      " eta =", round(x2[5], 6), " emitted =", np.round(q2, 6))
print("=> identical release at eta=1 (A on D17 column) and eta=3/4 (D17 in B):"
      " eta is not identified; interior eta = global time-sharing of a"
      " deterministic T-kernel (weight 1/4) and d_new (weight 3/4)")

section("M. Nonalias metric: within-T variation and T-projection value")


def within_T(qz1):
    """Weighted mean TV to the state mean, and max pairwise TV, binary tokens."""
    out, mx = 0.0, 0.0
    for t in (0, 1):
        idx = [i for i in range(4) if T[i] == t]
        wt = sum(float(m[i]) for i in idx)
        bar = sum(float(m[i]) * float(qz1[i]) for i in idx) / wt
        out += sum(float(m[i]) * abs(float(qz1[i]) - bar) for i in idx)
        mx = max([mx] + [abs(float(qz1[i]) - float(qz1[j])) for i in idx for j in idx])
    return out, mx


def projT(qz1):
    bars = {}
    for t in (0, 1):
        idx = [i for i in range(4) if T[i] == t]
        bars[t] = sum(m[i] * qz1[i] for i in idx) / sum(m[i] for i in idx)
    return tuple(bars[T[i]] for i in range(4))


for name, qq in (("D17-analog (eta=0 witness)", tuple(map(F, d17))),
                 ("D17 via A-column, eta=1 (alias)", tuple(map(F, d17))),
                 ("fixture channel (0,0,1,1/4)", q1)):
    tv, mx = within_T(qq)
    pj = projT(qq)
    ps = [sum(m[i] * pj[i] for i in range(4) if S[i] == s) / pS[s] for s in (0, 1)]
    print(f"{name:34s}: mean within-T TV {tv:.4f}, max pairwise {mx:.4f}; "
          f"T-projection {tuple(str(x) for x in pj)} private: {ps[0] == ps[1]}, "
          f"task info {task_info(pj):.5f}")
print("=> T is task-sufficient in Example A, so the T-projection has IDENTICAL task"
      " information (lemma: if Y is independent of X given (T,H), every decoder's"
      " expected loss depends on q only through its (T,H)-cell average); the"
      " within-T variation buys privacy, not task information.")
# Persistent vs fresh draws: two fresh draws of the fixture channel.
p2 = [sum(m[i] * q1[i] ** 2 for i in range(4) if S[i] == s) / pS[s] for s in (0, 1)]
print("two FRESH draws of (0,0,1,1/4): P(Z1=Z2=1|S=0), P(Z1=Z2=1|S=1) =", p2,
      "-> exact privacy fails; one persistent draw per record is required")

# ------------------------------------------------ stale frozen attackers
section("S. Frozen attackers overstate mixture privacy; concavity lower bound")
flip = tuple(1 - z for z in d17)
att = (0.2, 0.8)  # attacker P(S=1|z) fitted on D17 (Bayes for D17)


def frozen_attack(qz1, a):
    return sum(float(m[i]) * sum(pz * -math.log(a[z] if S[i] == 1 else 1 - a[z])
               for z, pz in ((0, 1 - qz1[i]), (1, qz1[i]))) for i in range(4))


def bayes_attack(qz1):
    j = {}
    for i in range(4):
        for z, pz in ((0, 1 - qz1[i]), (1, qz1[i])):
            j[S[i], z] = j.get((S[i], z), 0) + float(m[i]) * float(pz)
    return h(0.5) - mi({k: F(v).limit_denominator(10**12) for k, v in j.items()})


for lam in (0.0, 0.05, 0.1, 0.2):
    qz = [(1 - lam) * d17[i] + lam * flip[i] for i in range(4)]
    fa, ba = frozen_attack(qz, att), bayes_attack(qz)
    print(f"lambda={lam:4.2f} on the relabelled D17: frozen-attacker loss {fa:.4f}, "
          f"refit (Bayes) loss {ba:.4f}, overstatement {fa - ba:+.4f} nats; "
          f"component-refit concavity bound {(1-lam)*bayes_attack(d17)+lam*bayes_attack(flip):.4f}")

# ------------------------------------------------------------ rebasing
section("P. Rebasing after adding a stronger attacker can relax a constraint")
a1 = (0.55, 0.45)   # weak, anti-calibrated attacker in the old bank
delta = 0.001
rho_old = frozen_attack(d17, a1)
rho_new = min(rho_old, frozen_attack(d17, att))
cand = flip
print(f"rho(old bank {{a1}}) = {rho_old:.4f}; rho(new bank {{a1,a2}}) = {rho_new:.4f}")
print(f"candidate (relabelled D17): L_a1 = {frozen_attack(cand, a1):.4f}, L_a2 = {frozen_attack(cand, att):.4f}")
print("feasible under old bank:", frozen_attack(cand, a1) >= rho_old - delta,
      "| under new bank:", all(frozen_attack(cand, a) >= rho_new - delta for a in (a1, att)))
print("\nAll assertions passed.")
