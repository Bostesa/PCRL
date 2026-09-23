"""Independent re-evaluation of the full-view method's synthetic suite.

Pinned evidence: research/pcrl-full-view-protection-v1 @ 0e90d0b3da97bec28cf200abe2213803c2fb3b71.

For every fixture, budget and optimised method (robust / nominal / capacity):
* rebuild each law EXACTLY in rationals from the committed generator (verifying the float law's SHA-256);
* take the reported channel as the exact rational row-normalisation of its stored floats;
* evaluate every full-view CMI I(S;Z|H_A), I(S;Z|H_A,H_B) in mpmath interval arithmetic and classify
  primal feasibility as `certified` (upper end <= budget), `within_guard` (<= budget + 1e-7) or `violated`;
* compute the exact rational task cost;
* for the CMI-constrained methods, derive an interval-certified Lagrangian lower bound on the optimal
  cost: for any lambda >= 0, OPT >= sum_k lambda_k (f_k(q*) - grad f_k(q*).q* - delta)
  + sum_t min_z [c_tz + sum_k lambda_k grad f_k(q*)_tz], valid because each f_k is convex in q
  (supporting hyperplane) and every row ranges over a simplex. The lambda are chosen by a float search;
  any lambda gives a valid bound. The final evaluation is in interval arithmetic.
Also solves the rational_separation zero-budget case exactly as a linear programme in rationals.

Usage: python review_t4_synthetic.py > T4_SYNTHETIC_REVIEW.json   (run inside the PCRL repository)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
import types
from fractions import Fraction as F

import numpy as np
from mpmath import iv

iv.dps = 40
PIN = "0e90d0b3da97bec28cf200abe2213803c2fb3b71"
GUARD = 1e-7
ROLES = {"A": (0, 1), "AB": (0, 1, 2)}  # law axes: S, H_A, H_B, T, Y


def show(path):
    return subprocess.check_output(["git", "show", f"{PIN}:{path}"]).decode()


def load_generator(exact: bool):
    src = show("experiments/pcrl_full_view_protection_v1/synthetic.py")
    if exact:
        src = src.replace("dtype=float", "dtype=object").replace("float(", "(")
        src = src.replace("np.zeros((2, 2, 1, 2, 2))", "np.zeros((2, 2, 1, 2, 2), dtype=object)")
        src = src.replace("np.zeros((2, 2, 2, 2, 2))", "np.zeros((2, 2, 2, 2, 2), dtype=object)")
        src = src.replace("np.zeros((2, 1, 1, 3, 2))", "np.zeros((2, 1, 1, 3, 2), dtype=object)")
    mod = types.ModuleType("t4syn_exact" if exact else "t4syn")
    exec(compile(src, "synthetic.py", "exec"), mod.__dict__)
    return mod


def to_exact(arr):
    out = np.empty(arr.shape, dtype=object)
    for idx, v in np.ndenumerate(arr):
        out[idx] = v if isinstance(v, F) else F(v)
    assert sum(out.flat) == 1, "exact law must sum to 1"
    return out


def iv_of(x):
    return iv.mpf(x.numerator) / iv.mpf(x.denominator)


def marg(law, keep):
    """p(keep..., t) summing Y and non-kept axes; returns dict key -> Fraction."""
    d = {}
    for idx, v in np.ndenumerate(law):
        if v == 0:
            continue
        key = tuple(idx[a] for a in keep) + (idx[3],)
        d[key] = d.get(key, F(0)) + v
    return d


def joint_z(pst, q):
    """pst: {(s,h...,t): p} -> {(s,h...,z): p} as Fractions."""
    out = {}
    nz = len(q[0])
    for key, p in pst.items():
        t = key[-1]
        for z in range(nz):
            if q[t][z] == 0:
                continue
            k = key[:-1] + (z,)
            out[k] = out.get(k, F(0)) + p * q[t][z]
    return out


def cmi_iv(pshz):
    """I(S;Z|H) with keys (s, h..., z)."""
    ph, psh, phz = {}, {}, {}
    for k, p in pshz.items():
        s, h, z = k[0], k[1:-1], k[-1]
        ph[h] = ph.get(h, F(0)) + p
        psh[(s,) + h] = psh.get((s,) + h, F(0)) + p
        phz[h + (z,)] = phz.get(h + (z,), F(0)) + p
    tot = iv.mpf(0)
    for k, p in pshz.items():
        if p == 0:
            continue
        s, h, z = k[0], k[1:-1], k[-1]
        ratio = p * ph[h] / (psh[(s,) + h] * phz[h + (z,)])
        tot += iv_of(p) * iv.log(iv_of(ratio))
    return tot


def grad_float(pst, q):
    """d I(S;Z|H) / d q[t][z] = sum_{s,h} p(s,h,t) log p(s|h,z); None if infinite."""
    nt, nz = len(q), len(q[0])
    pz = joint_z(pst, q)
    phz = {}
    for k, p in pz.items():
        phz[k[1:]] = phz.get(k[1:], F(0)) + p
    g = [[0.0] * nz for _ in range(nt)]
    for key, p in pst.items():
        s, h, t = key[0], key[1:-1], key[-1]
        for z in range(nz):
            num = pz.get((s,) + h + (z,), F(0))
            den = phz.get(h + (z,), F(0))
            if num == 0:
                if den == 0:
                    continue  # (h,z) never occurs: the term's weight is zero along feasible q
                return None
            g[t][z] += float(p) * math.log(float(num / den))
    return g


def grad_iv(pst, q):
    nt, nz = len(q), len(q[0])
    pz = joint_z(pst, q)
    phz = {}
    for k, p in pz.items():
        phz[k[1:]] = phz.get(k[1:], F(0)) + p
    g = [[iv.mpf(0)] * nz for _ in range(nt)]
    for key, p in pst.items():
        s, h, t = key[0], key[1:-1], key[-1]
        for z in range(nz):
            num = pz.get((s,) + h + (z,), F(0))
            den = phz.get(h + (z,), F(0))
            if num == 0:
                if den == 0:
                    continue
                return None
            g[t][z] = g[t][z] + iv_of(p) * iv.log(iv_of(num / den))
    return g


def lagrangian_lb(cost, constraints, q, delta, lam, use_iv):
    """constraints: list of (pst). Returns the lower bound for multipliers lam."""
    nt, nz = len(q), len(q[0])
    if use_iv:
        C = [[iv_of(cost[t][z]) for z in range(nz)] for t in range(nt)]
        const = iv.mpf(0)
    else:
        C = [[float(cost[t][z]) for z in range(nz)] for t in range(nt)]
        const = 0.0
    for pst, lk in zip(constraints, lam):
        if lk == 0:
            continue
        f = cmi_iv(joint_z(pst, q)) if use_iv else float(cmi_iv(joint_z(pst, q)).mid)
        g = grad_iv(pst, q) if use_iv else grad_float(pst, q)
        if g is None:
            return None
        L = iv.mpf(lk) if use_iv else lk
        gq = sum(g[t][z] * (iv_of(q[t][z]) if use_iv else float(q[t][z])) for t in range(nt) for z in range(nz))
        const = const + L * (f - gq - (iv_of(F(delta)) if use_iv else delta))
        for t in range(nt):
            for z in range(nz):
                C[t][z] = C[t][z] + L * g[t][z]
    if use_iv:
        rowmins = []
        for t in range(nt):
            lows = [c.a for c in C[t]]
            rowmins.append(min(lows))
        return float((const + sum(iv.mpf(x) for x in rowmins)).a)
    return const + sum(min(C[t]) for t in range(nt))


def lp_lambda(cost, constraints, q, delta):
    """Maximise the concave piecewise-linear bound LB(lambda) exactly as an LP (float gradients):
    max sum_k lambda_k a_k + sum_t u_t  s.t.  u_t <= c_tz + sum_k lambda_k g_ktz,  lambda >= 0."""
    from scipy.optimize import linprog
    nt, nz, K = len(q), len(q[0]), len(constraints)
    a, G = [], []
    for pst in constraints:
        f = float(cmi_iv(joint_z(pst, q)).mid)
        g = grad_float(pst, q)
        if g is None:
            return None
        gq = sum(g[t][z] * float(q[t][z]) for t in range(nt) for z in range(nz))
        a.append(f - gq - delta)
        G.append(g)
    nv = K + nt
    obj = [-x for x in a] + [-1.0] * nt
    A, b = [], []
    for t in range(nt):
        for z in range(nz):
            row = [-G[k][t][z] for k in range(K)] + [0.0] * nt
            row[K + t] = 1.0
            A.append(row)
            b.append(float(cost[t][z]))
    bounds = [(0, 1e6)] * K + [(None, None)] * nt
    res = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return list(res.x[:K]) if res.status == 0 else None


def feasible_upper(cost, constraints, q, delta):
    """Rigorous upper bound on OPT: mix q toward the best constant row c (CMI 0) until the interval
    CMI of every constraint is <= delta; by convexity f(mix) <= (1-theta) f(q)."""
    nt, nz = len(q), len(q[0])
    zc = min(range(nz), key=lambda z: sum(cost[t][z] for t in range(nt)))
    for exp in range(40, 0, -1):
        theta = F(1, 2 ** exp)
        qm = [[(1 - theta) * q[t][z] + theta * (F(1) if z == zc else F(0)) for z in range(nz)] for t in range(nt)]
        if all(float(cmi_iv(joint_z(p, qm)).b) <= delta for p in constraints):
            c = sum(cost[t][z] * qm[t][z] for t in range(nt) for z in range(nz))
            return float(c), float(theta)
    return None, None


def search_lambda(cost, constraints, q, delta, active):
    lam = [0.0] * len(constraints)
    best = lagrangian_lb(cost, constraints, q, delta, lam, False)
    if best is None:
        return lam, None
    grid = [0.0] + [10 ** e for e in np.linspace(-4, 3, 57)]
    for _ in range(4):
        for k in active:
            cand = []
            for v in grid:
                trial = list(lam)
                trial[k] = v
                val = lagrangian_lb(cost, constraints, q, delta, trial, False)
                if val is not None:
                    cand.append((val, v))
            if cand:
                val, v = max(cand)
                if val > best:
                    best, lam[k] = val, v
        # local refinement
        for k in active:
            lo, hi = lam[k] * 0.5, lam[k] * 2 + 1e-9
            for _ in range(60):
                m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                t1, t2 = list(lam), list(lam)
                t1[k], t2[k] = m1, m2
                v1 = lagrangian_lb(cost, constraints, q, delta, t1, False)
                v2 = lagrangian_lb(cost, constraints, q, delta, t2, False)
                if v1 is None or v2 is None:
                    break
                if v1 < v2:
                    lo = m1
                else:
                    hi = m2
            trial = list(lam)
            trial[k] = (lo + hi) / 2
            v = lagrangian_lb(cost, constraints, q, delta, trial, False)
            if v is not None and v > best:
                best, lam = v, trial
    return lam, best


def exact_channel(ch):
    rows = []
    for r in ch:
        fr = [F(x) for x in r]
        s = sum(fr)
        rows.append([x / s for x in fr])
    return rows


def rational_zero_budget_lp(law):
    """Exact LP: min cost s.t. S indep Z (H trivial), Z binary, q_t = P(Z=1|t) in [0,1]."""
    p_st = {}
    for idx, v in np.ndenumerate(law):
        p_st[(idx[0], idx[3])] = p_st.get((idx[0], idx[3]), F(0)) + v
    p_ty = {}
    for idx, v in np.ndenumerate(law):
        p_ty[(idx[3], idx[4])] = p_ty.get((idx[3], idx[4]), F(0)) + v
    nt = law.shape[3]
    ps = [sum(p_st.get((s, t), F(0)) for t in range(nt)) for s in (0, 1)]
    a = [p_st.get((1, t), F(0)) / ps[1] - p_st.get((0, t), F(0)) / ps[0] for t in range(nt)]
    # cost = sum_t [P(t,Y=1)(1-q_t) + P(t,Y=0) q_t] (z=1 predicts y=1)
    base = sum(p_ty.get((t, 1), F(0)) for t in range(nt))
    slope = [p_ty.get((t, 0), F(0)) - p_ty.get((t, 1), F(0)) for t in range(nt)]
    best = None
    for free in range(nt):
        others = [t for t in range(nt) if t != free]
        for bounds in itertools.product((F(0), F(1)), repeat=nt - 1):
            q = [None] * nt
            for t, b in zip(others, bounds):
                q[t] = b
            rest = -sum(a[t] * q[t] for t in others)
            if a[free] == 0:
                if rest != 0:
                    continue
                cands = [F(0), F(1)]
            else:
                cands = [rest / a[free]]
            for v in cands:
                if 0 <= v <= 1:
                    q[free] = v
                    c = base + sum(slope[t] * q[t] for t in range(nt))
                    if best is None or c < best[0]:
                        best = (c, list(q))
    det = []
    for bits in itertools.product((0, 1), repeat=nt):
        if sum(a[t] * bits[t] for t in range(nt)) == 0:
            det.append(base + sum(slope[t] * bits[t] for t in range(nt)))
    return {"lp_optimum_cost": str(best[0]), "lp_optimum_q_Z1_given_T": [str(x) for x in best[1]],
            "best_zero_leakage_deterministic_cost": str(min(det))}


def main():
    gen_f, gen_x = load_generator(False), load_generator(True)
    fx_f, fx_x = gen_f.fixtures(), gen_x.fixtures()
    report = {"pin": PIN, "fixtures": {}}
    for name in fx_f:
        rec = json.loads(show(f"results/pcrl_full_view_protection_v1/synthetic/{name}.json"))
        laws_f = fx_f[name]["laws"]
        hashes = [hashlib.sha256(L.tobytes()).hexdigest() for L in laws_f]
        laws_x = [to_exact(L) for L in fx_x[name]["laws"]]
        # the exact rebuild must round to the committed float law
        rounds = all(np.array_equal(np.vectorize(float)(Lx).astype(float), Lf) for Lx, Lf in zip(laws_x, laws_f))
        cost = gen_x.task_coefficients(laws_x[0])
        costx = [[F(c) if not isinstance(c, F) else c for c in row] for row in cost.tolist()]
        pst = [(marg(L, ROLES[r])) for L in laws_x for r in ROLES]
        fx_out = {"law_hashes_match": hashes == rec["law_sha256"], "exact_rebuild_rounds_to_committed": rounds,
                  "budgets": {}}
        for b, brec in rec["budgets"].items():
            delta = float(b)
            out_b = {}
            for mname, m in brec["methods"].items():
                if "channel" not in m:
                    continue
                q = exact_channel(m["channel"])
                cmis = [cmi_iv(joint_z(p, q)) for p in pst]
                up = max(float(c.b) for c in cmis)
                exact_cost = sum(costx[t][z] * q[t][z] for t in range(len(q)) for z in range(len(q[0])))
                entry = {"reported_cost": m["cost"], "exact_cost": float(exact_cost),
                         "cmi_interval_max_upper": up,
                         "feasibility": ("certified" if up <= delta else
                                         "within_guard" if up <= delta + GUARD else "violated"),
                         "T4_lower": m.get("objective_lower_bound"), "T4_upper": m.get("objective_upper_bound")}
                if mname in ("robust", "nominal") and delta > 0:
                    lam = lp_lambda(costx, pst, q, delta)
                    lb = lagrangian_lb(costx, pst, q, delta, lam, True) if lam is not None else None
                    ub, theta = feasible_upper(costx, pst, q, delta)
                    entry.update({"certified_lower_bound": lb, "certified_feasible_upper_bound": ub,
                                  "mixing_theta": theta,
                                  "certified_gap": (ub - lb) if (lb is not None and ub is not None) else None,
                                  "T4_lower_exceeds_certified": (m.get("objective_lower_bound") is not None
                                                                 and lb is not None
                                                                 and m["objective_lower_bound"] > lb + 1e-12)})
                out_b[mname] = entry
            fx_out["budgets"][b] = out_b
        if name == "rational_separation":
            fx_out["exact_zero_budget_lp"] = rational_zero_budget_lp(laws_x[0])
        report["fixtures"][name] = fx_out
    json.dump(report, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    main()
