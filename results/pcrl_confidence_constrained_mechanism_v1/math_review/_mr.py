"""Shared helpers for role B's independent math review (ccm). numpy / scipy / stdlib only; SYNTHETIC data only.

Nothing here imports ccm.* or hcal.*: every predicate is re-derived from the TEXT of PROTOCOL.md section 3 and
UTILITY_CONTRACT.md section 2. Exact checks use fractions.Fraction (rational arithmetic) and decimal.Decimal (60 digits)
with directed rational bounds on exp(+-d).
"""
from __future__ import annotations

import math
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize

getcontext().prec = 60

D = 0.005            # NLL constant d (contract)
B = 0.0025           # Brier constant b (contract)
CLIP = 1e-12         # scoring clip (dpc.utility.per_row)
EMD = math.exp(-D)   # float64 exp(-d)
ED = math.exp(D)     # float64 exp(d)

# Exact decimal values of exp(-0.005) and exp(0.005) for the REAL constant d = 5/1000 (60 significant digits), and
# rational bounds that bracket them (|error| < 1e-50), used for exact certificates.
_DEXP_M = (Decimal(-5) / Decimal(1000)).exp()
_DEXP_P = (Decimal(5) / Decimal(1000)).exp()
EMD_UP = Fraction(_DEXP_M) + Fraction(1, 10 ** 50)     # > exp(-d)
EMD_LO = Fraction(_DEXP_M) - Fraction(1, 10 ** 50)     # < exp(-d)
ED_UP = Fraction(_DEXP_P) + Fraction(1, 10 ** 50)      # > exp(d)
ED_LO = Fraction(_DEXP_P) - Fraction(1, 10 ** 50)      # < exp(d)
D_FR = Fraction(5, 1000)
B_FR = Fraction(25, 10000)


# ------------------------------------------------------------------ float predicates (canonical forms as written)
def nll_ok(p, q):
    """(NLL) q_k >= exp(-d) p_k for every k, evaluated as written (multiplicative form)."""
    return bool(np.all(np.asarray(q) >= EMD * np.asarray(p)))


def brier_excess(p, q):
    """Vector over labels y of sum_k q_k^2 - sum_k p_k^2 - 2 (q_y - p_y) (full Brier, no 1/2)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return float(q @ q - p @ p) - 2.0 * (q - p)


def brier_ok(p, q):
    return bool(np.all(brier_excess(p, q) <= B))


def class_ok(q, d):
    q = np.asarray(q, float)
    return bool(np.all(np.delete(q, d) < q[d]))


def g_ok(p, q, d):
    return nll_ok(p, q) and brier_ok(p, q) and class_ok(q, d)


def decision(p):
    """Source tie rule: numpy argmax, first index on ties."""
    return int(np.argmax(np.asarray(p)))


# ------------------------------------------------------------------ exact (rational) predicates
def fr(v):
    return [Fraction(float(x)) for x in v]


def nll_ok_exact(p, q):
    """q_k >= exp(-d) p_k in real arithmetic (exact rationals of the float inputs; exp(-d) bracketed)."""
    p, q = fr(p), fr(q)
    # sufficient: q_k >= EMD_UP p_k ; necessary: q_k >= EMD_LO p_k. Report True only when the sufficient form holds.
    return all(qk >= EMD_UP * pk for pk, qk in zip(p, q))


def brier_excess_exact(p, q):
    p, q = fr(p), fr(q)
    S = sum(x * x for x in q) - sum(x * x for x in p)
    return [S - 2 * (qy - py) for py, qy in zip(p, q)]


def g_ok_exact(p, q, d):
    """Exact real-arithmetic G for float-valued p, q (exact rationals of the floats). Simplex: q sums to 1 exactly."""
    qf = fr(q)
    if sum(qf) != 1 or min(qf) < 0:
        return False
    if not nll_ok_exact(p, q):
        return False
    if max(brier_excess_exact(p, q)) > B_FR:
        return False
    return all(qf[k] < qf[d] for k in range(len(qf)) if k != d)


def g_ok_rational(p, q, d):
    """G for rational p, q (lists of Fraction), exact; p and q must each sum to 1 exactly."""
    assert sum(p) == 1 and sum(q) == 1 and min(q) >= 0
    nll = all(qk >= EMD_UP * pk for pk, qk in zip(p, q))
    S = sum(x * x for x in q) - sum(x * x for x in p)
    br = max(S - 2 * (qy - py) for py, qy in zip(p, q)) <= B_FR
    cl = all(q[k] < q[d] for k in range(len(q)) if k != d)
    return nll, br, cl


# ------------------------------------------------------------------ closed forms
def nll_minimax(P):
    """min over q in simplex of max_{i,y} log(p_iy / q_y) = log sum_k max_i p_ik (attained at q = M / sum M)."""
    M = np.max(np.asarray(P, float), axis=0)
    return math.log(M.sum()), M / M.sum()


def waterfill(beta):
    """argmax over c in simplex of 1 - ||c||^2 - c.beta ; c_y = max(0, (tau - beta_y)/2), sum c = 1."""
    beta = np.asarray(beta, float)
    K = beta.size
    s = np.sort(beta)
    for m in range(K, 0, -1):
        tau = (2.0 + s[:m].sum()) / m
        if tau - s[m - 1] > 0:
            break
    c = np.maximum(0.0, (tau - beta) / 2.0)
    c = c / c.sum()
    return c


def brier_minimax(P):
    """Exact min over q of max_{i,y} [B(q, y) - B(p_i, y)] (Brier-only bin value):
    = max_{c in simplex} [1 - ||c||^2 - c.beta], beta_y = min_i B(p_i, y), attained at q = c* (water-filling)."""
    P = np.asarray(P, float)
    K = P.shape[1]
    E = np.eye(K)
    beta = np.array([min(np.sum((p - E[y]) ** 2) for p in P) for y in range(K)])
    c = waterfill(beta)
    return float(1.0 - c @ c - c @ beta), c, beta


def joint_minimax(P, d, cls_closure=True, x0=None):
    """Numerical min over q in {simplex, q >= e^{-d} M, (q_d >= q_k)} of max_{i,y} Brier excess, by SLSQP on the
    epigraph. Returns (value, q, success). Admissibility (closure of Class) <=> value <= b and the polytope nonempty."""
    P = np.asarray(P, float)
    n, K = P.shape
    M = P.max(0)
    lo = EMD * M
    if lo.sum() > 1.0:
        return math.inf, None, False
    E = np.eye(K)
    Bp = np.array([[np.sum((p - E[y]) ** 2) for y in range(K)] for p in P])
    beta = Bp.min(0)

    def g(z):
        q = z[:K]
        return np.array([z[K] - (np.sum((q - E[y]) ** 2) - beta[y]) for y in range(K)])

    cons = [{"type": "eq", "fun": lambda z: np.sum(z[:K]) - 1.0},
            {"type": "ineq", "fun": g},
            {"type": "ineq", "fun": lambda z: z[:K] - lo}]
    if cls_closure:
        cons.append({"type": "ineq", "fun": lambda z: np.delete(z[d] - z[:K], d)})
    if x0 is None:
        q0 = lo + (1 - lo.sum()) * E[d]
    else:
        q0 = np.asarray(x0, float)
    t0 = max(np.sum((q0 - E[y]) ** 2) - beta[y] for y in range(K))
    res = minimize(lambda z: z[K], np.r_[q0, t0], method="SLSQP", constraints=cons,
                   options={"ftol": 1e-15, "maxiter": 500})
    return float(res.x[K]), res.x[:K], bool(res.success)


# ------------------------------------------------------------------ binary (K = 2) exact bin structure, class 0
def bin_interval(c):
    """For K = 2, decision 0 (a = p_0 in [1/2, 1]), representative q = (c, 1 - c) with c > 1/2: the exact set of members
    a admissible under G (NLL + Brier + Class) is the interval [L(c), R(c)].
      a <= c : NLL(1) a >= 1 - e^d (1 - c) ; Brier(label 1) 2(c^2 - a^2) <= b  <=>  a >= sqrt(c^2 - b/2)
      a >= c : NLL(0) a <= e^d c          ; Brier(label 0) 2((1-c)^2 - (1-a)^2) <= b <=> a <= 1 - sqrt((1-c)^2 - b/2)
    """
    L = max(1.0 - ED * (1.0 - c), math.sqrt(max(c * c - B / 2.0, 0.0)), 0.5)
    R = min(ED * c, 1.0 - math.sqrt(max((1.0 - c) ** 2 - B / 2.0, 0.0)), 1.0)
    return L, R


def bin_interval_nll(c):
    L = max(1.0 - ED * (1.0 - c), 0.5)
    R = min(ED * c, 1.0)
    return L, R


# ------------------------------------------------------------------ information measures (plug-in, exact)
def mi_bayes(joint):
    """joint[s, t] probabilities; returns (MI in nats, Bayes accuracy of S from T)."""
    J = np.asarray(joint, float)
    ps = J.sum(1, keepdims=True)
    pt = J.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(J > 0, J * np.log(J / (ps * pt)), 0.0)
    return float(r.sum()), float(J.max(0).sum())


def ucal_text_rule(P, alpha):
    """Independent re-implementation of the TEXT of the frozen log-input rule (UTILITY_CONTRACT 1, hcal.calib doc):
    alpha == 1 -> P exactly; else P' = max(P, 1e-12) / sum, q = exp(alpha L - max) / sum."""
    P = np.asarray(P, float)
    if alpha == 1.0:
        return P.copy()
    Pp = np.maximum(P, 1e-12)
    Pp = Pp / Pp.sum()
    z = alpha * np.log(Pp)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
