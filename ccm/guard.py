"""Pointwise utility contract G(d, b) and the teacher-expected guard G_exp(d, b) (ccm; role C; PROTOCOL sections 3, 5;
MATH_REVIEW.md B1, B2, R3.3, R4, R7.4).

SYNTHETIC tests only (tests/pcrl_confidence_constrained_mechanism_v1/test_guard.py). No data is loaded here, no label
and no SEX value is read: every function is a pure function of reference vectors p (Ucal, PRIMARY), released
vectors q and decisions.

SCIENTIFIC CONFIGURATION (CONFIG, registered; imported by ccm.geometry and ccm.oracle; nothing in it is chosen from a
result): d = 0.005, b = 0.0025, capacity 8 tokens per predicted class for income (K = 2) and 64 for occupation (K = 6),
simplex tolerance 1e-12, clipping floor 1e-12, fallback nudge eta = 1e-6. ENGINEERING holds solver constants and
float-semantics margins only (iteration budgets, construction margins, conservative closed-form margins).

CANONICAL PREDICATE G(d, b) (B2; THE predicate used for every certificate; float64, evaluated exactly as written, no
tolerance). Inputs p (reference, ..., K), q (release, ..., K), dec (default numpy argmax of p, first index):
    E      = np.exp(-d)                                         one float64 constant
    NLL    : np.all(q >= E * p, axis=-1)
    Brier  : np.all(np.sum(q*q, -1)[..., None] - np.sum(p*p, -1)[..., None] - 2*(q - p) <= b, axis=-1)
             (sum over k in index order; the full multiclass Brier with no factor 1/2, every label y)
    Class  : q[dec] > q[k] for every k != dec (strict)
    Simplex: np.all(q >= 0, axis=-1) and abs(np.sum(q, -1) - 1) <= 1e-12   (non-finite q fails)
  The log-form statements (log(p_y/q_y) <= d for every label; the clipped (1e-12) score version) hold in real arithmetic
  and, evaluated in float, to about 3e-15 (MATH_REVIEW R1.5).
G_exp(d, b) (secondary diagnostic): KL(p||q) = sum_{k: p_k > 0} p_k (log p_k - log q_k) <= d (0 log 0 = 0; q_k = 0 < p_k
  -> +inf), np.sum((q - p)**2, -1) <= b, the same strict class and simplex conditions.
Residual convention: <= 0 passes (class: < 0 passes).

CONSTRUCTION MARGIN (B2.2). Every representative this module returns is BUILT against tightened targets and then
CERTIFIED with the canonical predicate at the true d and b (both must pass, for every member):
    G:     q >= (E*p)*(1 + 1e-9);  Brier excess <= b - 1e-9;  q[dec] - q[k] >= 1e-9
    G_exp: KL <= d - 1e-9;          sum (q-p)^2 <= b - 1e-9;    q[dec] - q[k] >= 1e-9
  so that every algebraically equivalent float form agrees (check_construction). Checks stay at d and b.
CONSERVATIVE CLOSED FORMS (B2.3). Ucal row sums are only within 1e-12 of 1, so the G closed form declares a bin
  infeasible only if sum_k max_member p_k > exp(d) (1 + 1e-12) (nll_necessary). G_exp closed forms use an absolute
  margin 1e-9 (exp_necessary). The same margins are used by ccm.geometry F2 / F3u.

BIN REPRESENTATIVE (bin_representative; deterministic; fixed iteration budget)
  G:  1. empty bin -> ValueError. dec: given, else the common first-index argmax of the members (ValueError if they
         disagree).
      2. nll_necessary fails -> INFEASIBLE_NLL (closed form; pairwise overlap is not sufficient for K >= 3).
      3. start q0 = M / sum(M) (M = coordinatewise max). This is L + (1 - sum L) w with L = E M and w = M / sum M, the
         point maximising the uniform relative NLL slack min_k q_k / M_k.
      4. SLSQP (analytic gradients, maxiter ENGINEERING['slsqp_maxiter']) on the epigraph form of
             min_q max_y [q.q - 2 q_y - c_y],  c_y = min_member (p.p - 2 p_y)   (= max_{members, y} Brier excess)
             s.t. sum q = 1,  q_k >= E M_k (1 + 1e-9),  q_dec >= q_k + 1e-8   (convex; the solver aims 10x inside
             the 1e-9 construction class margin because SLSQP meets linear constraints only to ~1e-10)
      5. polish (clip to the solver lower bounds; the float residual of sum q goes to dec, or to the coordinate with the
         largest slack when negative), then check_construction AND the canonical predicate against EVERY member. The
         SLSQP point is tried first, then q0; the first that passes is returned (CERTIFIED); otherwise NOT_FOUND (never
         a claim of infeasibility).
  G_exp: exp_necessary (closed form -> INFEASIBLE_BOX: coordinatewise member range > 2 sqrt(b), two Brier balls of
      radius sqrt(b) cannot meet; generalised Jensen-Shannon mean_j KL(p_j || pbar) > d; member variance
      mean_j ||p_j - pbar||^2 > b: a max over members is at least their mean, minimised at q = pbar), start q0 = member
      mean, SLSQP on min_q max_j max(KL(p_j||q)/d, ||q - p_j||^2/b) - 1 in softmax coordinates (q = softmax(z),
      z_dec = 0, strict class as the bound z_k <= -2 K class_margin) over a working set of members (initial:
      coordinatewise extreme members plus the worst at q0; the most violated members are added between rounds; at most
      exp_ws_rounds rounds), then polish and the same two checks against ALL members (SLSQP point first, then q0).

FALLBACK (B1, R3.3). fallback_release(P, dec) releases Ucal itself where its top is strict at dec, and
  (1 - eta) Ucal + eta e_dec where the top is tied (admissible for 0 < eta <= (sqrt(1 + 8b) - 1)/4 = 0.0024876).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

CONFIG = {
    "study": "pcrl_confidence_constrained_mechanism_v1",
    "primary_contract": "G",
    "secondary_contract": "G_exp",
    "d": 0.005,
    "b": 0.0025,
    "capacity_per_class": {"income": 8, "occupation": 64},
    "K": {"income": 2, "occupation": 6},
    "simplex_tol": 1e-12,
    "clip": 1e-12,
    "fallback_eta": 1e-6,
}

ENGINEERING = {
    "slsqp_maxiter": 100,
    "slsqp_ftol": 1e-15,
    "nll_rel_margin": 1e-9,               # construction: q >= (E*p)(1 + 1e-9)
    "brier_margin": 1e-9,                 # construction: Brier excess <= b - 1e-9 (G_exp: sum (q-p)^2 <= b - 1e-9)
    "class_margin": 1e-9,                 # construction: q_dec - q_k >= 1e-9
    "solver_class_margin": 1e-8,          # G solver constraint q_dec >= q_k + 1e-8 (SLSQP meets linear constraints only
                                          # to ~1e-10, so it aims 10x inside the construction margin)
    "exp_kl_margin": 1e-9,                # construction (G_exp): KL <= d - 1e-9
    "closed_form_rel_margin": 1e-12,      # G infeasible / pairwise infeasible only if sum max > exp(d)(1 + 1e-12)
    "exp_closed_form_abs_margin": 1e-9,   # G_exp closed forms: infeasible only if value > bound + 1e-9
    "prefilter_tol": 1e-12,               # permissive slack on NECESSARY prefilters (never prune a certifiable bin)
    "exp_z_floor": -700.0,                # G_exp solver (softmax coordinates, z_dec = 0): z_k >= -700 (q_k ~ 1e-304)
    "exp_ws_rounds": 12,
    "exp_ws_add": 8,
    "exp_ws_init_worst": 4,
}

CONTRACTS = ("G", "G_exp")
STATUSES = ("CERTIFIED", "INFEASIBLE_NLL", "INFEASIBLE_BOX", "NOT_FOUND")
ETA_MAX = (math.sqrt(1.0 + 8.0 * CONFIG["b"]) - 1.0) / 4.0     # R3.3 admissible nudge bound (0.0024876 at b = 0.0025)


def capacity_for(recipient=None, K=None):
    """Registered capacity (tokens per predicted class): by recipient name or by K (2 -> income, 6 -> occupation)."""
    if recipient is not None:
        return int(CONFIG["capacity_per_class"][recipient])
    for name, k in CONFIG["K"].items():
        if int(K) == k:
            return int(CONFIG["capacity_per_class"][name])
    raise ValueError(f"no registered capacity for K={K}; pass the capacity explicitly")


def _dk(d, b):
    return (float(CONFIG["d"]) if d is None else float(d)), (float(CONFIG["b"]) if b is None else float(b))


def nll_constant(d=None):
    """E = np.exp(-d): the ONE float64 constant of the canonical NLL predicate q >= E * p."""
    d, _ = _dk(d, None)
    return np.exp(-d)


def decisions(P):
    """Source tie rule: numpy argmax, first index."""
    return np.argmax(np.asarray(P, dtype=np.float64), axis=-1)


def _broadcast(q, p, dec):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q, p = np.broadcast_arrays(q, p)
    if dec is None:
        dec = np.argmax(p, axis=-1)
    dec = np.broadcast_to(np.asarray(dec, dtype=np.intp), q.shape[:-1])
    return q, p, dec


def _class_and_simplex(q, dec, margin=None):
    """Strict class (margin None: q_dec > q_k; else q_dec - q_k >= margin) and the simplex condition."""
    K = q.shape[-1]
    qd = np.take_along_axis(q, dec[..., None], axis=-1)[..., 0]
    if K > 1:
        other = q.copy()
        np.put_along_axis(other, dec[..., None], -np.inf, axis=-1)
        class_res = other.max(axis=-1) - qd
        if margin is None:
            class_ok = np.all(other < qd[..., None], axis=-1)
        else:
            class_ok = np.all(qd[..., None] - other >= margin, axis=-1)
    else:
        class_res = np.full(q.shape[:-1], -np.inf)
        class_ok = np.ones(q.shape[:-1], dtype=bool)
    s = np.sum(q, axis=-1)
    tol = float(CONFIG["simplex_tol"])
    simplex_ok = np.all(q >= 0.0, axis=-1) & (np.abs(s - 1.0) <= tol)
    simplex_res = np.maximum(np.abs(s - 1.0) - tol, -q.min(axis=-1))
    simplex_res = np.where(np.all(np.isfinite(q), axis=-1), simplex_res, np.inf)
    return class_ok, class_res, simplex_ok, simplex_res


def _worst(out, names):
    w = {}
    for n in names:
        r = out[n + "_residual"]
        w[n] = float(np.max(r)) if r.size else float("-inf")
    return w


def brier_excess(q, p):
    """Canonical per-label Brier excess: np.sum(q*q,-1) - np.sum(p*p,-1) - 2*(q - p)  (shape (..., K))."""
    return np.sum(q * q, axis=-1)[..., None] - np.sum(p * p, axis=-1)[..., None] - 2.0 * (q - p)


def check_release(q, p, d=None, b=None, dec=None):
    """The CANONICAL predicate G (module docstring), rowwise over the leading axes (q and p broadcast).

    Returns a dict of arrays: ok, nll, brier, class, simplex (booleans) and the residuals nll_residual
    (max_k E p_k - q_k), brier_residual (max_y Brier excess - b), brier_excess (..., K: every label),
    class_residual (max_{k != dec} q_k - q_dec), simplex_residual; plus 'worst' (max residual of each condition)."""
    d, b = _dk(d, b)
    q, p, dec = _broadcast(q, p, dec)
    E = np.exp(-d)
    lower = E * p
    nll_ok = np.all(q >= lower, axis=-1)
    nll_res = np.max(lower - q, axis=-1)
    excess = brier_excess(q, p)
    brier_ok = np.all(excess <= b, axis=-1)
    brier_res = np.max(excess, axis=-1) - b
    class_ok, class_res, simplex_ok, simplex_res = _class_and_simplex(q, dec)
    out = {"ok": nll_ok & brier_ok & class_ok & simplex_ok, "nll": nll_ok, "brier": brier_ok, "class": class_ok,
           "simplex": simplex_ok, "nll_residual": nll_res, "brier_residual": brier_res, "brier_excess": excess,
           "class_residual": class_res, "simplex_residual": simplex_res, "contract": "G", "d": d, "b": b}
    out["worst"] = _worst(out, ("nll", "brier", "class", "simplex"))
    return out


def kl_div(p, q):
    """KL(p || q) along the last axis, natural logs, 0 log 0 = 0, q_k = 0 < p_k -> +inf."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p, q = np.broadcast_arrays(p, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0.0, p * (np.log(np.where(p > 0.0, p, 1.0)) - np.log(q)), 0.0)
    return np.sum(t, axis=-1)


def check_release_exp(q, p, d=None, b=None, dec=None):
    """Teacher-expected guard G_exp: KL(p||q) <= d, sum (q-p)^2 <= b, strict class, simplex. Same layout as
    check_release (kl / sq conditions in place of nll / brier)."""
    d, b = _dk(d, b)
    q, p, dec = _broadcast(q, p, dec)
    kl = kl_div(p, q)
    kl_ok = kl <= d
    sqd = np.sum((q - p) ** 2, axis=-1)
    sq_ok = sqd <= b
    class_ok, class_res, simplex_ok, simplex_res = _class_and_simplex(q, dec)
    out = {"ok": kl_ok & sq_ok & class_ok & simplex_ok, "kl": kl_ok, "sq": sq_ok, "class": class_ok,
           "simplex": simplex_ok, "kl_residual": kl - d, "sq_residual": sqd - b, "class_residual": class_res,
           "simplex_residual": simplex_res, "contract": "G_exp", "d": d, "b": b}
    out["worst"] = _worst(out, ("kl", "sq", "class", "simplex"))
    return out


def check(q, p, contract="G", d=None, b=None, dec=None):
    """The canonical predicate of the contract (G or G_exp) at the true d and b."""
    if contract == "G":
        return check_release(q, p, d, b, dec)
    if contract == "G_exp":
        return check_release_exp(q, p, d, b, dec)
    raise ValueError(f"unknown contract {contract!r}")


def check_construction(q, p, contract="G", d=None, b=None, dec=None):
    """The TIGHTENED construction targets (module docstring); returns the boolean array 'ok' (rowwise)."""
    d, b = _dk(d, b)
    q, p, dec = _broadcast(q, p, dec)
    E = ENGINEERING
    class_ok, _, simplex_ok, _ = _class_and_simplex(q, dec, margin=E["class_margin"])
    if contract == "G":
        nll = np.all(q >= (np.exp(-d) * p) * (1.0 + E["nll_rel_margin"]), axis=-1)
        br = np.all(brier_excess(q, p) <= b - E["brier_margin"], axis=-1)
        return nll & br & class_ok & simplex_ok
    if contract == "G_exp":
        kl = kl_div(p, q) <= d - E["exp_kl_margin"]
        sq = np.sum((q - p) ** 2, axis=-1) <= b - E["brier_margin"]
        return kl & sq & class_ok & simplex_ok
    raise ValueError(f"unknown contract {contract!r}")


def accepts(q, p, contract="G", d=None, b=None, dec=None):
    """A representative q ACCEPTS p iff the construction targets AND the canonical predicate both hold (rowwise)."""
    return check_construction(q, p, contract, d, b, dec) & check(q, p, contract, d, b, dec)["ok"]


def nll_bound(d=None):
    """Conservative closed-form bound exp(d)(1 + closed_form_rel_margin) (B2.3)."""
    d, _ = _dk(d, None)
    return math.exp(d) * (1.0 + ENGINEERING["closed_form_rel_margin"])


def nll_necessary(members, d=None):
    """Closed-form necessary condition of a G bin, conservative (B2.3): sum_k max_member p_k <= exp(d)(1 + 1e-12).
    Returns (ok, value). A failure certifies infeasibility (INFEASIBLE_NLL)."""
    P = np.atleast_2d(np.asarray(members, dtype=np.float64))
    v = float(np.sum(P.max(axis=0)))
    return bool(v <= nll_bound(d)), v


def js_div(p, q):
    """Jensen-Shannon divergence 0.5 KL(p||m) + 0.5 KL(q||m), m = (p + q) / 2, along the last axis (natural logs)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def exp_bounds(members):
    """Closed-form lower bounds for a G_exp bin (members P (m, K), mean pbar):
        range = max_k (max_j p_jk - min_j p_jk)         (feasible needs <= 2 sqrt(b): two Brier balls must meet)
        gjs   = mean_j KL(p_j || pbar)                   (max_j KL(p_j||q) >= mean_j KL(p_j||q) >= gjs for every q)
        var   = mean_j ||p_j - pbar||^2                  (max_j ||q - p_j||^2 >= mean_j ||q - p_j||^2 >= var)"""
    P = np.atleast_2d(np.asarray(members, dtype=np.float64))
    pbar = P.mean(axis=0)
    return {"range": float((P.max(axis=0) - P.min(axis=0)).max()), "gjs": float(np.mean(kl_div(P, pbar[None, :]))),
            "var": float(np.mean(np.sum((P - pbar[None, :]) ** 2, axis=1)))}


def exp_necessary(members, d=None, b=None):
    """Closed-form necessary conditions of a G_exp bin (exp_bounds), conservative by an absolute 1e-9:
    range <= 2 sqrt(b) + m, gjs <= d + m, var <= b + m. Returns (ok, bounds). A failure certifies infeasibility."""
    d, b = _dk(d, b)
    m = ENGINEERING["exp_closed_form_abs_margin"]
    v = exp_bounds(members)
    ok = v["range"] <= 2.0 * math.sqrt(b) + m and v["gjs"] <= d + m and v["var"] <= b + m
    return bool(ok), v


# ------------------------------------------------------------------------------------------------------- fallback

def tied_top(P):
    """Rows whose maximum is attained at two or more classes (label-free)."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    return np.sum(P == P.max(axis=1, keepdims=True), axis=1) >= 2


def fallback_release(P, dec=None, eta=None, d=None, b=None):
    """Disclosed fallback (B1): Ucal itself where its top is strict at dec, (1 - eta) Ucal + eta e_dec where the top is
    tied at dec. dec must be a top index of its row (ValueError otherwise). Returns (Q, info) with info = counts of tied
    rows and of released rows failing the canonical predicate (expected 0)."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    dec = decisions(P) if dec is None else np.asarray(dec, dtype=np.intp)
    eta = float(CONFIG["fallback_eta"]) if eta is None else float(eta)
    if not 0.0 < eta <= ETA_MAX:
        raise ValueError(f"eta must lie in (0, {ETA_MAX}]")
    top = P.max(axis=1)
    pd = P[np.arange(P.shape[0]), dec]
    if np.any(pd != top):
        raise ValueError(f"{int(np.sum(pd != top))} rows have a decision that is not a top index")
    tied = tied_top(P)
    Q = P.copy()
    if tied.any():
        Q[tied] = (1.0 - eta) * P[tied]
        Q[np.flatnonzero(tied), dec[tied]] += eta
    chk = check_release(Q, P, d, b, dec)
    return Q, {"n": int(P.shape[0]), "n_tied": int(tied.sum()), "eta": eta, "n_fail_G": int(np.sum(~chk["ok"]))}


# ---------------------------------------------------------------------------------------------------------------- G

def g_stats(P, d):
    """(M, L, c): coordinatewise max, NLL lower bounds E M, c_y = min_member (p.p - 2 p_y)."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    M = P.max(axis=0)
    L = np.exp(-d) * M
    c = (np.sum(P * P, axis=1)[:, None] - 2.0 * P).min(axis=0)
    return M, L, c


def polish(q, lb, dec):
    """Clip to lb, then move the float residual of sum q onto dec (positive) or the coordinate with the most slack."""
    q = np.maximum(np.asarray(q, dtype=np.float64), lb)
    r = 1.0 - np.sum(q)
    if r >= 0.0:
        q[dec] += r
    else:
        k = int(np.argmax(q - lb))
        q[k] += r
    return q


def g_slsqp(L, c, dec, q0, b=None):
    """min_q max_y [q.q - 2 q_y - c_y] over {sum q = 1, q >= L (1 + nll_rel_margin), q_dec >= q_k + solver margin}.
    Returns (q, info); q is the raw SLSQP point (not polished, not certified); info['t'] = max Brier excess - b."""
    _, b = _dk(None, b)
    E = ENGINEERING
    K = len(L)
    lb = L * (1.0 + E["nll_rel_margin"])
    eye2 = 2.0 * np.eye(K)
    others = [k for k in range(K) if k != dec]
    A = np.zeros((len(others), K + 1))
    for i, k in enumerate(others):
        A[i, dec] = 1.0
        A[i, k] = -1.0
    m = E["solver_class_margin"]

    def h(x):
        q = x[:K]
        return x[K] - (q @ q - 2.0 * q - c)

    def h_j(x):
        q = x[:K]
        J = np.empty((K, K + 1))
        J[:, :K] = eye2 - 2.0 * q[None, :]
        J[:, K] = 1.0
        return J

    cons = [{"type": "ineq", "fun": h, "jac": h_j},
            {"type": "eq", "fun": lambda x: np.array([x[:K].sum() - 1.0]),
             "jac": lambda x: np.concatenate([np.ones(K), [0.0]])[None, :]}]
    if others:
        cons.append({"type": "ineq", "fun": lambda x: A @ x - m, "jac": lambda x: A})
    q0 = np.maximum(np.asarray(q0, dtype=np.float64), lb)
    x0 = np.concatenate([q0, [float(np.max(q0 @ q0 - 2.0 * q0 - c))]])
    bounds = [(float(lb[k]), 1.0) for k in range(K)] + [(None, None)]
    res = minimize(lambda x: x[K], x0, jac=lambda x: np.concatenate([np.zeros(K), [1.0]]), constraints=cons,
                   bounds=bounds, method="SLSQP", options={"maxiter": E["slsqp_maxiter"], "ftol": E["slsqp_ftol"]})
    return np.asarray(res.x[:K], dtype=np.float64), {"nit": int(res.nit), "success": bool(res.success),
                                                       "message": str(res.message), "t": float(res.x[K]) - b}


def _certify(q, P, dec, contract, d, b):
    """Construction targets AND canonical predicate for every member. Returns (ok, canonical worst residuals)."""
    chk = check(q[None, :], P, contract, d, b, dec)
    ok = bool(np.all(chk["ok"])) and bool(np.all(check_construction(q[None, :], P, contract, d, b, dec)))
    return ok, chk["worst"]


def g_try(P, dec, d=None, b=None, order=("slsqp", "start"), stats=None):
    """Try candidate representatives for a G bin in the given order; return (q, method, worst, info) of the first one
    that passes _certify against every member, or (None, None, worst_of_last, info)."""
    d, b = _dk(d, b)
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    M, L, c = g_stats(P, d) if stats is None else stats
    q0 = M / np.sum(M)
    lb = L * (1.0 + ENGINEERING["nll_rel_margin"])
    info = {}
    worst = None
    for meth in order:
        if meth == "start":
            q = polish(q0, lb, dec)
        elif meth == "slsqp":
            raw, info = g_slsqp(L, c, dec, q0, b)
            q = polish(raw, lb, dec)
        else:
            raise ValueError(meth)
        ok, worst = _certify(q, P, dec, "G", d, b)
        if ok:
            return q, meth, worst, info
    return None, None, worst, info


# ------------------------------------------------------------------------------------------------------------ G_exp

def _exp_excess(q, P, d, b):
    """Normalised excess per member: max(KL(p_j||q)/d, ||q - p_j||^2/b) - 1 (<= 0 iff both G_exp bounds hold)."""
    return np.maximum(kl_div(P, q[None, :]) / d, np.sum((q[None, :] - P) ** 2, axis=1) / b) - 1.0


def exp_slsqp(P, dec, q0, d=None, b=None):
    """Working-set SLSQP for min_q max_j max(KL(p_j||q)/d, ||q - p_j||^2/b) - 1 over the class simplex, in softmax
    coordinates q = softmax(z) with z_dec = 0 pinned (KL_j = sum p log p - p_j.z + s_j LSE(z) is convex and smooth, all
    gradients are bounded, sum q = 1 holds by construction, the strict class is the bound z_k <= -2 K class_margin, and
    z_k >= exp_z_floor). A smooth reparametrisation of a convex problem: local minima are global. Returns (q, info)."""
    d, b = _dk(d, b)
    E = ENGINEERING
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    n, K = P.shape
    free = np.array([k for k in range(K) if k != dec], dtype=np.intp)
    zf, mz = E["exp_z_floor"], 2.0 * K * E["class_margin"]
    pos = P > 0.0
    plogp = np.sum(np.where(pos, P * np.log(np.where(pos, P, 1.0)), 0.0), axis=1)
    srow = P.sum(axis=1)

    def zfull(x):
        z = np.zeros(K)
        z[free] = x[: K - 1]
        return z

    def soft(z):
        e = np.exp(z - z.max())
        return e / e.sum(), z.max() + math.log(e.sum())

    def q_of(x):
        return soft(zfull(x))[0]

    q = np.maximum(np.asarray(q0, dtype=np.float64), 0.0)
    q = q / q.sum()
    with np.errstate(divide="ignore"):
        z0 = np.log(q) - math.log(q[dec]) if q[dec] > 0 else np.zeros(K)
    x = np.concatenate([np.clip(z0[free], zf, -mz), [0.0]])
    ex0 = _exp_excess(q_of(x), P, d, b)
    ws = set(np.argmax(P, axis=0).tolist()) | set(np.argmin(P, axis=0).tolist())
    ws |= set(np.argsort(-ex0, kind="stable")[: E["exp_ws_init_worst"]].tolist())
    info = {"rounds": 0, "ws": 0, "nit": 0, "success": False, "t": float(np.max(ex0))}
    bounds = [(zf, -mz)] * (K - 1) + [(None, None)]
    for rnd in range(E["exp_ws_rounds"]):
        W = np.array(sorted(ws), dtype=np.intp)
        PW, plW, sW = P[W], plogp[W], srow[W]

        def cons(x, PW=PW, plW=plW, sW=sW):
            z = zfull(x)
            q, lse = soft(z)
            kl = plW - PW @ z + sW * lse
            sq = np.sum((q[None, :] - PW) ** 2, axis=1)
            return np.concatenate([x[-1] - kl / d + 1.0, x[-1] - sq / b + 1.0])

        def cons_j(x, PW=PW, sW=sW):
            q, _ = soft(zfull(x))
            gkl = (-PW + sW[:, None] * q[None, :]) / d                     # d KL_j / dz
            r = q[None, :] - PW
            gsq = 2.0 * (q[None, :] * r - q[None, :] * (r @ q)[:, None]) / b  # d ||q - p_j||^2 / dz
            J = np.zeros((2 * len(PW), K))
            J[: len(PW), :K - 1] = -gkl[:, free]
            J[len(PW):, :K - 1] = -gsq[:, free]
            J[:, K - 1] = 1.0
            return J

        x[-1] = float(np.max(_exp_excess(q_of(x), PW, d, b)))
        res = minimize(lambda x: x[-1], x, jac=lambda x: np.concatenate([np.zeros(K - 1), [1.0]]),
                       constraints=[{"type": "ineq", "fun": cons, "jac": cons_j}], bounds=bounds, method="SLSQP",
                       options={"maxiter": E["slsqp_maxiter"], "ftol": E["slsqp_ftol"]})
        x = np.asarray(res.x, dtype=np.float64)
        q = q_of(x)
        info = {"rounds": rnd + 1, "ws": len(W), "nit": info["nit"] + int(res.nit), "success": bool(res.success),
                "message": str(res.message)}
        ex = _exp_excess(q, P, d, b)
        tW = float(np.max(ex[W]))
        info["t"] = float(np.max(ex))
        viol = np.flatnonzero(ex > tW + E["prefilter_tol"])
        viol = [j for j in viol[np.argsort(-ex[viol], kind="stable")].tolist() if j not in ws]
        if not viol:
            break
        ws |= set(viol[: E["exp_ws_add"]])
    return q, info


def exp_try(P, dec, d=None, b=None, order=("slsqp", "start")):
    """Candidate representatives for a G_exp bin (SLSQP optimum, member mean); first one passing _certify wins."""
    d, b = _dk(d, b)
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    q0 = P.mean(axis=0)
    lb = np.zeros(P.shape[1])
    info = {}
    worst = None
    for meth in order:
        if meth == "start":
            q = polish(q0, lb, dec)
        elif meth == "slsqp":
            raw, info = exp_slsqp(P, dec, q0, d, b)
            q = polish(raw, lb, dec)
        else:
            raise ValueError(meth)
        ok, worst = _certify(q, P, dec, "G_exp", d, b)
        if ok:
            return q, meth, worst, info
    return None, None, worst, info


# ------------------------------------------------------------------------------------------------------- public API

def bin_dec(P, dec=None):
    """The bin's decision: given, else the common first-index argmax of every member (ValueError if they differ)."""
    if dec is not None:
        return int(dec)
    dd = np.unique(decisions(P))
    if len(dd) != 1:
        raise ValueError(f"members have different decisions {dd.tolist()}; pass dec explicitly")
    return int(dd[0])


def bin_representative(members, dec=None, contract="G", d=None, b=None):
    """Find ONE q satisfying the contract for ALL members, or report why not.

    Returns (q or None, certificate). certificate['status'] in STATUSES: CERTIFIED (q built against the tightened
    construction targets and checked with the canonical predicate against every member and label; 'worst' = canonical
    worst residuals), INFEASIBLE_NLL (G: sum_k max p_k > exp(d)(1 + 1e-12)), INFEASIBLE_BOX (G_exp closed forms),
    NOT_FOUND (no certificate; NOT a claim of infeasibility)."""
    d, b = _dk(d, b)
    P = np.asarray(members, dtype=np.float64)
    if P.ndim == 1:
        P = P[None, :]
    if P.ndim != 2 or P.shape[0] == 0:
        raise ValueError("bin_representative needs a non-empty (n, K) member array")
    dec = bin_dec(P, dec)
    cert = {"contract": contract, "d": d, "b": b, "n_members": int(P.shape[0]), "dec": dec}
    if contract == "G":
        ok, v = nll_necessary(P, d)
        cert.update(nll_sum=v, nll_bound=nll_bound(d))
        if not ok:
            cert["status"] = "INFEASIBLE_NLL"
            return None, cert
        q, meth, worst, info = g_try(P, dec, d, b)
    elif contract == "G_exp":
        ok, v = exp_necessary(P, d, b)
        cert.update(exp_bounds=v)
        if not ok:
            cert["status"] = "INFEASIBLE_BOX"
            return None, cert
        q, meth, worst, info = exp_try(P, dec, d, b)
    else:
        raise ValueError(f"unknown contract {contract!r}")
    cert.update(method=meth, worst=worst, solver=info, status="CERTIFIED" if q is not None else "NOT_FOUND")
    return q, cert
