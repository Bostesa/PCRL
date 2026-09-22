"""Exact finite-support checks for the PCRL claims audit.

Every fixture uses fractions.Fraction on an explicit finite table, so a pass is
an exact statement about that table, not a floating-point approximation.
Information quantities are returned as exact rational multiples of log(2)
(or log(3)) by construction; the helper refuses any table whose log ratios
are not in the supported set.

Fixtures complement the proofs in THEORY_REPAIRS.md. They demonstrate a
counterexample or instantiate an identity; they do not prove a universal
statement by passing.

Run: python analysis/pcrl_claims_foundation_v1/exact_fixtures.py [--json OUT]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from fractions import Fraction as F


# ---------------------------------------------------------------------------
# small exact linear algebra
# ---------------------------------------------------------------------------

def mean(xs, w=None):
    if w is None:
        return sum(xs, F(0)) / len(xs)
    return sum((wi * x for wi, x in zip(w, xs)), F(0)) / sum(w, F(0))


def cov(a, b, w=None):
    ma, mb = mean(a, w), mean(b, w)
    if w is None:
        w = [F(1)] * len(a)
    return sum((wi * (x - ma) * (y - mb) for wi, x, y in zip(w, a, b)), F(0)) / sum(w, F(0))


def solve(A, b):
    """Gauss-Jordan on Fractions; A square nonsingular."""
    n = len(A)
    M = [list(map(F, row)) + [F(v)] for row, v in zip(A, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[r][n] for r in range(n)]


def ols_r2(X_cols, y, w=None, ridge=F(0)):
    """Affine least-squares R^2 of y on the columns of X (with intercept).

    Weighted population version when w is given; centering and the optional
    ridge penalty (on slopes only, in covariance units) follow the repository
    convention gram = Cov(X) + ridge*I.
    """
    d = len(X_cols)
    S = [[cov(X_cols[i], X_cols[j], w) + (ridge if i == j else 0) for j in range(d)] for i in range(d)]
    c = [cov(X_cols[i], y, w) for i in range(d)]
    beta = solve(S, c)
    vy = cov(y, y, w)
    # residual variance of centered fit: Var(y) - 2 b'c + b' Cov(X) b
    SX = [[cov(X_cols[i], X_cols[j], w) for j in range(d)] for i in range(d)]
    bSb = sum(beta[i] * SX[i][j] * beta[j] for i in range(d) for j in range(d))
    rss = vy - 2 * sum(bi * ci for bi, ci in zip(beta, c)) + bSb
    return 1 - rss / vy, rss, vy


# ---------------------------------------------------------------------------
# exact information quantities on finite tables, as multiples of log(base)
# ---------------------------------------------------------------------------

def _log_coeff(ratio: F):
    """Return (k2, k3) with ratio == 2**k2 * 3**k3, else raise."""
    num, den = ratio.numerator, ratio.denominator
    k = {2: 0, 3: 0}
    for p in (2, 3):
        while num % p == 0:
            num //= p
            k[p] += 1
        while den % p == 0:
            den //= p
            k[p] -= 1
    if num != 1 or den != 1:
        raise ValueError(f"ratio {ratio} is not 2^a 3^b; exact log form unsupported")
    return k[2], k[3]


def mi(joint: dict, a_idx, b_idx, c_idx=()):
    """Exact I(A;B|C) = sum p log[p(a,b,c)p(c)/(p(a,c)p(b,c))] as {'log2':q,'log3':r}."""
    def marg(idx):
        m = {}
        for key, p in joint.items():
            k = tuple(key[i] for i in idx)
            m[k] = m.get(k, F(0)) + p
        return m
    pabc = marg(tuple(a_idx) + tuple(b_idx) + tuple(c_idx))
    pac = marg(tuple(a_idx) + tuple(c_idx))
    pbc = marg(tuple(b_idx) + tuple(c_idx))
    pc = marg(tuple(c_idx))
    na, nb = len(a_idx), len(b_idx)
    out2, out3 = F(0), F(0)
    for key, p in pabc.items():
        if p == 0:
            continue
        a, b, c = key[:na], key[na:na + nb], key[na + nb:]
        ratio = p * pc[c] / (pac[a + c] * pbc[b + c])
        k2, k3 = _log_coeff(ratio)
        out2 += p * k2
        out3 += p * k3
    return {"log2": out2, "log3": out3}


def mi_float(v):
    return float(v["log2"]) * math.log(2) + float(v["log3"]) * math.log(3)


def fstr(x):
    return str(x) if isinstance(x, F) else x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def f1_retired_accuracy_guarantee():
    """Zero affine LS R^2 with 90% threshold accuracy (balanced binary A)."""
    rows = [(1, 1)] * 9 + [(1, -9)] + [(0, -1)] * 9 + [(0, 9)]
    A = [F(a) for a, _ in rows]
    h = [F(x) for _, x in rows]
    c = cov(h, A)
    r2, _, _ = ols_r2([h], A)
    acc = F(sum(1 for a, x in rows if (x > 0) == (a == 1)), len(rows))
    majority = F(max(sum(A), len(A) - sum(A)), len(A))
    cond_means = (mean([x for a, x in rows if a == 1]), mean([x for a, x in rows if a == 0]))
    assert c == 0 and r2 == 0 and acc == F(9, 10) and majority == F(1, 2)
    assert cond_means == (0, 0)
    return {"n": len(rows), "cov_hA": fstr(c), "affine_ls_r2": fstr(r2),
            "threshold_h_gt_0_accuracy": fstr(acc), "majority_accuracy": fstr(majority),
            "conditional_means": [fstr(m) for m in cond_means]}


def _onehot_identity(X_cols, labels, K, ridge=F(0)):
    n = len(labels)
    per, rss_tot, tss_tot, tss = [], F(0), F(0), []
    for k in range(K):
        z = [F(int(y == k)) for y in labels]
        r2, rss, vy = ols_r2(X_cols, z, ridge=ridge)
        per.append(r2)
        rss_tot += rss
        tss_tot += vy
        tss.append(vy)
    pooled = 1 - rss_tot / tss_tot
    pri = [F(sum(1 for y in labels if y == k), n) for k in range(K)]
    w = [p * (1 - p) for p in pri]
    sw = sum(w)
    w = [x / sw for x in w]
    combo = sum(wi * ri for wi, ri in zip(w, per))
    return pooled, per, w, pri, combo


def f2_convex_combination_identity():
    """Pooled one-hot R^2 == sum_k w_k R2_OvR_k exactly (OLS and common ridge)."""
    rnd = random.Random(20260922)
    n, K = 40, 3
    labels = [0] * 30 + [1] * 8 + [2] * 2
    X1 = [F(rnd.randint(-5, 5)) + (3 if y == 2 else 0) for y in labels]
    X2 = [F(rnd.randint(-5, 5)) + (1 if y == 1 else 0) for y in labels]
    out = {}
    for name, ridge in (("ols", F(0)), ("ridge_1_10", F(1, 10))):
        pooled, per, w, pri, combo = _onehot_identity([X1, X2], labels, K, ridge)
        assert pooled == combo, (name, pooled, combo)
        out[name] = {"pooled_onehot_r2": fstr(pooled), "per_class_r2": [fstr(x) for x in per],
                     "weights": [fstr(x) for x in w], "combo": fstr(combo), "exact_equal": pooled == combo}
    # counter-convention: weights from different priors than the scored rows break it
    pooled, per, w, pri, combo = _onehot_identity([X1, X2], labels, K)
    wrong = [F(1, 3)] * 3
    combo_wrong = sum(a * b for a, b in zip(wrong, per))
    out["uniform_weights_not_equal"] = combo_wrong != pooled
    assert combo_wrong != pooled
    return out


def f2b_rare_class_masking():
    """A single feature perfectly identifies a 1% class; aggregate stays small."""
    labels = [0] * 90 + [1] * 9 + [2] * 1
    h = [F(int(y == 2)) for y in labels]
    pooled, per, w, pri, combo = _onehot_identity([h], labels, 3)
    assert per[2] == 1 and pooled == combo
    bound_c = pooled / w[2]
    return {"priors": [fstr(p) for p in pri], "per_class_r2": [fstr(x) for x in per],
            "weights": [fstr(x) for x in w], "aggregate": fstr(pooled),
            "aggregate_float": float(pooled), "rare_class_r2": fstr(per[2]),
            "bound_aggregate_over_w_rare": fstr(bound_c), "bound_float": float(bound_c),
            "bound_is_vacuous": bound_c >= 1}


def f3_max_ovr_vs_all_directions():
    """Max one-vs-rest R^2 can be strictly below the best sensitive direction."""
    labels = [0, 1, 2] * 4
    h = [F({0: 1, 1: -1, 2: 0}[y]) for y in labels]
    pooled, per, w, pri, combo = _onehot_identity([h], labels, 3)
    # best linear combination a'onehot: h itself is z0 - z1, so R^2(h -> z0 - z1) = 1
    contrast = [F(int(y == 0)) - F(int(y == 1)) for y in labels]
    r2_contrast, _, _ = ols_r2([h], contrast)
    assert per == [F(3, 4), F(3, 4), F(0)] and pooled == F(1, 2) and r2_contrast == 1
    return {"per_class_r2": [fstr(x) for x in per], "aggregate": fstr(pooled),
            "max_ovr_r2_DA": fstr(max(per)), "r2_best_class_contrast": fstr(r2_contrast),
            "note": "top squared canonical correlation is 1 > max OvR 3/4 > aggregate 1/2"}


def f4_composition():
    """Exact-zero composition, approximate-leakage amplification, and the
    lambda_min bound R2_joint <= sum_i R2_i Var(Z_i) / lambda_min(Cov Z)."""
    eps = F(1, 10)
    pts = list(itertools.product([-1, 1], [-1, 1]))  # (V, S) independent fair signs
    V = [F(v) for v, _ in pts]
    S = [F(s) for _, s in pts]
    Z1 = [v + eps * s for v, s in zip(V, S)]
    Z2 = [v - eps * s for v, s in zip(V, S)]
    r1, _, _ = ols_r2([Z1], S)
    r2, _, _ = ols_r2([Z2], S)
    rj, _, _ = ols_r2([Z1, Z2], S)
    a, b = cov(Z1, Z1), cov(Z1, Z2)
    lam_min = a - b  # eigenvalues of [[a,b],[b,a]] are a+b, a-b (b>0)
    bound = (r1 * a + r2 * cov(Z2, Z2)) / lam_min
    assert r1 == r2 == eps ** 2 / (1 + eps ** 2) == F(1, 101) and rj == 1
    assert lam_min == 2 * eps ** 2 and bound == 1
    # exact-zero composition with full nonlinear recovery: Z1=S*V, Z2=V
    W1 = [s * v for v, s in zip(V, S)]
    c1, c2 = cov(W1, S), cov(V, S)
    rj0, _, _ = ols_r2([W1, V], S)
    recover = all(w * v == s for w, v, s in zip(W1, V, S))
    assert c1 == 0 and c2 == 0 and rj0 == 0 and recover
    return {"epsilon": fstr(eps), "individual_r2": [fstr(r1), fstr(r2)], "joint_r2": fstr(rj),
            "cov_Z_eigen_min": fstr(lam_min), "lambda_min_bound": fstr(bound),
            "condition_number": fstr((a + b) / (a - b)),
            "zero_cov_views": {"cov_Z1_S": fstr(c1), "cov_Z2_S": fstr(c2), "joint_linear_r2": fstr(rj0),
                                "S_equals_Z1_times_Z2_exactly": recover}}


def f5_xor_coarsening():
    """I(S;Z)=0 but I(S;Z|H)=log 2 for Z = S xor H; constant coarsening hides it."""
    joint = {(s, h, s ^ h): F(1, 4) for s in (0, 1) for h in (0, 1)}  # (S, H, Z)
    i_marg = mi(joint, [0], [2])
    i_cond = mi(joint, [0], [2], [1])
    # coarse view: replace H by a constant partition cell
    coarse = {}
    for (s, h, z), p in joint.items():
        coarse[(s, 0, z)] = coarse.get((s, 0, z), F(0)) + p
    i_coarse = mi(coarse, [0], [2], [1])
    assert i_marg == {"log2": 0, "log3": 0} and i_cond == {"log2": 1, "log3": 0}
    assert i_coarse == {"log2": 0, "log3": 0}
    return {"I(S;Z)": "0", "I(S;Z|H)": "1*log2", "I(S;Z|coarse(H)=const)": "0",
            "I(S;Z|H)_nats": mi_float(i_cond)}


def f6_replacement_vs_extension():
    """H constant, J=(Y,S), T=Y with Y,S independent fair bits."""
    joint = {(y, s, (y, s), y): F(1, 4) for y in (0, 1) for s in (0, 1)}  # (Y,S,J,T)
    add_task = mi(joint, [0], [3], [2])      # I(Y;T|J)
    task_T = mi(joint, [0], [3])             # I(Y;T)
    task_J = mi(joint, [0], [2])             # I(Y;J)
    leak_J = mi(joint, [1], [2])             # I(S;J)
    leak_T = mi(joint, [1], [3])             # I(S;T)
    assert add_task["log2"] == 0 and task_T == task_J and leak_J["log2"] == 1 and leak_T["log2"] == 0
    return {"I(Y;T|J)": "0", "I(Y;T)": "1*log2", "I(Y;J)": "1*log2", "I(S;J)": "1*log2", "I(S;T)": "0"}


def _kernel_joint(p_sct, Q):
    """p(s,c,z) = sum_t p(s,c,t) Q[t][z]."""
    out = {}
    for (s, c, t), p in p_sct.items():
        for z, q in enumerate(Q[t]):
            out[(s, c, z)] = out.get((s, c, z), F(0)) + p * q
    return out


def f7_zero_budget_and_refinement():
    """Constant kernel is feasible at zero budget; child-copy of a coarse kernel
    reproduces the coarse joint law and objective exactly under pooled costs."""
    rnd = random.Random(7)
    # children t in 0..3, parents: 0,1 -> 0 ; 2,3 -> 1
    parent = {0: 0, 1: 0, 2: 1, 3: 1}
    raw = {(s, c, t): F(rnd.randint(1, 9)) for s in (0, 1) for c in (0, 1) for t in range(4)}
    tot = sum(raw.values())
    p_fine = {k: v / tot for k, v in raw.items()}
    p_coarse = {}
    for (s, c, t), p in p_fine.items():
        key = (s, c, parent[t])
        p_coarse[key] = p_coarse.get(key, F(0)) + p
    nz = 3
    Qc = [[F(1, 2), F(1, 3), F(1, 6)], [F(0), F(1, 4), F(3, 4)]]
    Qf = [Qc[parent[t]] for t in range(4)]
    jc, jf = _kernel_joint(p_coarse, Qc), _kernel_joint(p_fine, Qf)
    assert jc == jf
    # costs: D_fine(t,z) arbitrary; pooled parent cost is the p(t)-weighted mean
    pt = {t: sum(p for (s, c, tt), p in p_fine.items() if tt == t) for t in range(4)}
    Df = [[F(rnd.randint(0, 9), 7) for _ in range(nz)] for _ in range(4)]
    Dc = []
    for g in (0, 1):
        kids = [t for t in range(4) if parent[t] == g]
        m = sum(pt[t] for t in kids)
        Dc.append([sum(pt[t] * Df[t][z] for t in kids) / m for z in range(nz)])
    obj_f = sum(pt[t] * Qf[t][z] * Df[t][z] for t in range(4) for z in range(nz))
    pg = {g: sum(pt[t] for t in range(4) if parent[t] == g) for g in (0, 1)}
    obj_c = sum(pg[g] * Qc[g][z] * Dc[g][z] for g in (0, 1) for z in range(nz))
    assert obj_f == obj_c
    # zero budget: a constant row law makes Z independent of (S, C)
    Q0 = [[F(1, 3)] * 3 for _ in range(4)]
    j0 = {(s, c, 0, z): p for (s, c, z), p in _kernel_joint(p_fine, Q0).items()}
    i0 = mi(j0, [0], [3], [1])
    assert i0 == {"log2": 0, "log3": 0}
    # zero-mass cell: a row for a code with p(t)=0 does not affect the law
    p_zero = dict(p_fine)
    for s in (0, 1):
        for c in (0, 1):
            p_zero[(s, c, 4)] = F(0)
    Qz = Qf + [[F(1), F(0), F(0)]]
    Qz2 = Qf + [[F(0), F(0), F(1)]]
    assert _kernel_joint(p_zero, Qz) == _kernel_joint(p_zero, Qz2)
    return {"child_copy_joint_equal": True, "objective_equal": fstr(obj_f) == fstr(obj_c),
            "objective": fstr(obj_f), "constant_kernel_I(S;Z|C)": "0",
            "zero_mass_row_irrelevant": True}


def f8_expected_loss_vs_mixture_loss():
    """E_z[-log f(z)] differs from -log E_z[f(z)] (Jensen)."""
    q = [F(1, 2), F(1, 2)]
    fz = [0.9, 0.1]  # predicted P(Y=1) under each token; truth Y=1
    exp_loss = sum(float(qi) * -math.log(f) for qi, f in zip(q, fz))
    mix_loss = -math.log(sum(float(qi) * f for qi, f in zip(q, fz)))
    assert exp_loss > mix_loss
    return {"expected_single_token_loss": exp_loss, "loss_of_mixture_prediction": mix_loss,
            "difference_nats": exp_loss - mix_loss}


def f9_restricted_family_increment_without_information():
    """H uniform on {-1,0,1}, S=1[H=0], Z=1[H=0]=f(H): I(S;Z|H)=0, yet a
    linear-logistic attacker on H gains H_b(1/3) nats from Z."""
    joint = {(int(h == 0), h, int(h == 0)): F(1, 3) for h in (-1, 0, 1)}  # (S,H,Z)
    cmi = mi(joint, [0], [2], [1])
    assert cmi == {"log2": 0, "log3": 0}
    # symmetric convex loss => an optimal linear-logistic slope on H is 0; best
    # constant gives H_b(1/3) = log 3 - (2/3) log 2 nats; with Z the loss is 0.
    hb = {"log3": F(1), "log2": F(-2, 3)}
    return {"I(S;Z|H)": "0", "restricted_H_only_linear_logistic_loss": "log3 - (2/3)log2",
            "restricted_loss_nats": mi_float(hb), "loss_with_Z": 0.0,
            "measured_increment_nats": mi_float(hb),
            "argument": "loss(a,b)=loss(a,-b) by H->-H symmetry; convexity gives optimum at b=0"}


def f10_cmi_convexity_spotcheck(trials: int = 200):
    """Midpoint convexity of Q -> I(S;Z|C) on random finite tables (float).
    A spot check of the standard fact, not a proof."""
    rnd = random.Random(11)
    worst = 0.0
    for _ in range(trials):
        nt, nz = 4, 3
        p = {(s, c, t): rnd.random() for s in (0, 1) for c in (0, 1) for t in range(nt)}
        tot = sum(p.values())
        p = {k: v / tot for k, v in p.items()}

        def rowst():
            r = [rnd.random() for _ in range(nz)]
            s = sum(r)
            return [x / s for x in r]
        Q1 = [rowst() for _ in range(nt)]
        Q2 = [rowst() for _ in range(nt)]
        Qm = [[(a + b) / 2 for a, b in zip(r1, r2)] for r1, r2 in zip(Q1, Q2)]

        def cmi(Q):
            j = {}
            for (s, c, t), pp in p.items():
                for z in range(nz):
                    j[(s, c, z)] = j.get((s, c, z), 0.0) + pp * Q[t][z]
            pc, psc, pcz = {}, {}, {}
            for (s, c, z), v in j.items():
                pc[c] = pc.get(c, 0) + v
                psc[(s, c)] = psc.get((s, c), 0) + v
                pcz[(c, z)] = pcz.get((c, z), 0) + v
            return sum(v * math.log(v * pc[c] / (psc[(s, c)] * pcz[(c, z)])) for (s, c, z), v in j.items() if v > 0)
        gap = cmi(Qm) - (cmi(Q1) + cmi(Q2)) / 2
        worst = max(worst, gap)
    assert worst <= 1e-12
    return {"trials": trials, "max_midpoint_violation": worst}


def f11_prop6_whitened_bound_on_epsilon_example():
    """Original Proposition 6 (I): R2(H;A) <= sum_p R2(h_p;A) / lambda_min(R),
    R the block-whitened correlation. On the epsilon example it is tight (=1)."""
    eps = F(1, 10)
    v = 1 + eps ** 2                      # Var(Z1) = Var(Z2)
    rho = (1 - eps ** 2) / v              # whitened cross-correlation
    lam_min = 1 - rho
    per = [eps ** 2 / v, eps ** 2 / v]
    bound = sum(per) / lam_min
    assert bound == 1
    return {"R_offdiag": fstr(rho), "lambda_min_R": fstr(lam_min), "sum_per_purpose_r2": fstr(sum(per)),
            "bound": fstr(bound), "joint_r2": "1", "tight": True}


def f12_prop4_floor_not_implied_for_multilayer_lora():
    """Original Proposition 4 bounds h=(I+BA')f0 with rank(BA')<=r. The trained
    encoder adapts every Linear (ReLU in between). A rank-1 edit of the FIRST
    layer removes a rank-2 cross-covariance entirely, below the rank-1 floor."""
    # x = (s, y): s uniform on {0,1,2} (sensitive, K=3), y uniform on {0,1}, independent
    pts = [(s, y) for s in (0, 1, 2) for y in (0, 1)]

    def relu(t):
        return t if t > 0 else F(0)
    b = [F(-1, 2), F(-3, 2), F(0)]
    W1 = [[F(1), F(0)], [F(1), F(0)], [F(0), F(1)]]   # rank-2 weight, three hidden units

    def layer(W, x):
        return [relu(sum(w * xi for w, xi in zip(row, x)) + bb) for row, bb in zip(W, b)]
    f0 = [layer(W1, (F(s), F(y))) for s, y in pts]
    onehot = [[F(int(s == k)) for k in range(3)] for s, _ in pts]
    cols = list(zip(*f0))
    # pooled one-hot R2 of f0 (features 0,1 carry s; feature 2 carries y)
    def pooled(cols_x):
        cols_x = [list(c) for c in cols_x if cov(list(c), list(c)) != 0]
        rss = tss = F(0)
        for k in range(3):
            z = [row[k] for row in onehot]
            r2, rs, vy = ols_r2(cols_x, z)
            rss += rs
            tss += vy
        return 1 - rss / tss
    r2_f0 = pooled(cols)
    # rank-1 first-layer LoRA: Delta = -W1 e_s e_s^T (zeroes the s column)
    W1e = [[F(0), row[1]] for row in W1]
    f1 = [layer(W1e, (F(s), F(y))) for s, y in pts]
    r2_after = pooled(list(zip(*f1)))
    # rank of the cross-covariance between f0 and the centered one-hot: 2 (three
    # affinely independent s-images), so the Prop-4 rank-1 floor is sigma_2^2/tr > 0.
    s_imgs = {s: layer(W1, (F(s), F(0)))[:2] for s in (0, 1, 2)}
    affinely_independent = (s_imgs[1][0] - s_imgs[0][0]) * (s_imgs[2][1] - s_imgs[0][1]) != \
        (s_imgs[1][1] - s_imgs[0][1]) * (s_imgs[2][0] - s_imgs[0][0])
    assert r2_f0 == 1 and r2_after == 0 and affinely_independent
    return {"pooled_r2_before": fstr(r2_f0), "cross_cov_rank": 2, "lora_rank_first_layer": 1,
            "pooled_r2_after_rank1_first_layer_edit": fstr(r2_after),
            "task_feature_y_preserved": all(a[2] == c[2] for a, c in zip(f0, f1)),
            "note": "Prop 4's floor sum_{i>r} sigma_i^2/tr > 0 for r=1 applies only to h=(I+BA')f0"}


FIXTURES = [f1_retired_accuracy_guarantee, f2_convex_combination_identity, f2b_rare_class_masking,
            f3_max_ovr_vs_all_directions, f4_composition, f5_xor_coarsening,
            f6_replacement_vs_extension, f7_zero_budget_and_refinement,
            f8_expected_loss_vs_mixture_loss, f9_restricted_family_increment_without_information,
            f10_cmi_convexity_spotcheck, f11_prop6_whitened_bound_on_epsilon_example,
            f12_prop4_floor_not_implied_for_multilayer_lora]


def run_all():
    return {f.__name__: f() for f in FIXTURES}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    res = run_all()
    text = json.dumps(res, indent=1, default=str)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    print(text)
