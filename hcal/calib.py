"""Held-out, shared calibration of frozen releases (hcal; role C; prompt section 7). SYNTHETIC tests only
(tests/pcrl_heldout_calibration_v1/test_calib.py); no data is loaded here and no SEX enters any objective. Every rule
below was fixed before any real fit: no grid search, no kappa sweep, no tolerance tuning.

Every calibrator changes only the DECODED VECTORS of a frozen release. The partition's tokens, the token classes and
therefore every released decision are unchanged (checked on every output; a mismatch raises, it is never hidden).
Fitting rows: the CALIBRATION_HELDOUT representatives (H-*) or the CALIBRATION_TRAIN_MATCHED representatives
(T-TOKEN32, diagnostic only). Labels are the caller's (hcal.data.task_labels through the allowlist); none is read here.

H-TOKEN32 / T-TOKEN32 (token t of one recipient; K classes; d = the token class = the teacher decision of every row of
the token; y_t (K,) calibration label counts, n_t = sum_k y_t)
    minimise over C_d = {u : sum u = 1, u >= 0, u_d >= u_k for every k}
        sum_i [-log q(Y_i) + 0.5 ||q - onehot(Y_i)||^2] + 32 KL(mu_t || q),
        q = smooth(u, d) = (u + eps 1 + eps e_d) / (1 + (K+1) eps),   eps = 1e-12   (natural logs, float64)
  mu_t = the FIXED ORIGINAL UNSMOOTHED TEACHER MEAN of the token (frozen bank mu = S / n_fit), passed in explicitly and
  never recomputed from calibration rows. This is exactly lra.decoder's per-token objective with pbar := mu_t and
  n := n_t (calibration count). token32_solve WRAPS lra.decoder's certified exact solver: lra.decoder._state, _rowsum,
  project_class_simplex, _certify, smooth, EPS, BISECT_ITERS and its tolerances are imported unchanged and
  solve_batch's control flow is replicated step for step with A = Y + 32 PBAR. With PBAR = S / n it is BITWISE identical
  to lra.decoder.solve_batch(Y, S, n, d) for U and Q (obj equal; tested). The certificate is lra.decoder.cert_row of the
  FINAL released vector; an uncertified token raises lra.decoder.DecoderError (never released). Inputs: Y nonnegative
  integer counts with row sums == n >= 1; PBAR finite, >= 0, |sum - 1| <= 1e-9, PBAR[d] >= max PBAR - 1e-9; kappa = 32
  and eps = 1e-12 fixed.
  Fallbacks (registered): a RESERVED EMPTY token (n_fit = 0) keeps q0[t] exactly, RESERVED_EMPTY_ORIGINAL_FALLBACK
  (never fitted, even when calibration rows reach it); n_cal = 0 keeps q0[t] exactly, NO_CALIBRATION_OBSERVATIONS; else
  FITTED. Parameter count = FITTED tokens x (K - 1) free coordinates.

H-GLOBAL-TEMP (temperature scaling is PRIOR WORK: Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural
Networks", ICML 2017; it is applied here unchanged, not proposed as a new algorithm). One inverse temperature alpha per
(task / recipient, map, seed):
    q_alpha(t)_k = softmax_k(alpha log q0_tk),   alpha in [0.25, 4]
  fitted on the calibration representatives by minimising the UNCLIPPED mean NLL
    NLL(alpha) = mean_i [-alpha L_i,y_i + logsumexp_k(alpha L_ik)],   L = log q0[tok]   (float64; LSE minus row max)
  No SEX, no Brier, no ECE, no regulariser, no grid. NLL is convex: g = NLL' = mean_i [sum_k p_ik L_ik - L_i,y_i] is
  non-decreasing and NLL'' = mean_i Var_{p_i}(L_i) >= 0 (p_i = softmax(alpha L_i); row means by math.fsum).
  Deterministic bounded solve: g(0.25) >= 0 -> alpha = 0.25 (BOUNDARY_LOW); elif g(4) <= 0 -> alpha = 4
  (BOUNDARY_HIGH); else bisection on g over [0.25, 4] (invariant g(lo) <= 0 < g(hi)), at most TEMP_BISECT_ITERS = 100
  halvings, stopping early once lo, hi are adjacent float64 values; alpha = the bracket end with the smaller |g|
  (ties -> lo) (INTERIOR). Certificate (else CalibrationError): interior |g(alpha)| <= TEMP_GRAD_TOL scale and
  bracket <= 2 ulp;
  boundary sign condition as above; curvature >= 0; NLL(alpha) <= NLL(1) + TEMP_NLL_TOL scale (alpha = 1 is feasible).
  scale = mean_i max_k |L_ik| (> 0): g has the units of L and |g_i| <= 2 max_k |L_ik|, so 1e-9 scale is a relative
  stationarity bound, ~1e6 above the float64 noise of g at an adjacent-float bracket. Recorded on the calibration rows:
  NLL at alpha and at 1 (unclipped, log space), the CLIPPED (1e-12) scoring log loss of the APPLIED release at alpha and
  at 1, and the count of scored (true-label) probabilities below 1e-12 at alpha and at 1.
H-CLASS-TEMP: the same scalar fitted separately within each predicted class c (calibration rows whose token class /
  decision is c), shared by every token of class c; a class with fewer than CLASS_MIN = 50 calibration representatives
  (0 = absent class included) has alpha = 1, predetermined (CLASS_FALLBACK_LT50). At most K parameters (2 income,
  6 occupation); no per-token temperatures. Both temperature families transform EVERY token of the map (reserved and
  calibration-free tokens included): the temperature is shared, there is no per-token fallback.

IDENTITY PATHS: alpha == 1.0 exactly returns the input exactly (a float64 copy): temp_apply_tokens(q0, 1.0) == q0,
temp_apply_probs(P, 1.0) == P bitwise; the class fallback and a solve returning exactly 1.0 take this path. Otherwise
q = exp(alpha L - max_k alpha L) / sum_k (explicit k-ordered sum), float64. Positive alpha preserves class order; after
applying, the strict argmax of every token must equal its token class, and argmax(q) of every U row must equal the
teacher decision (raise otherwise, with the mismatch count).

CONTINUOUS U (per-row teacher probabilities P (n, K)): the same two families. LOG-INPUT RULE (frozen): alpha == 1 -> P
returned exactly; alpha != 1 -> P' = max(P, 1e-12), P' = P' / sum_k P', q = softmax(alpha log P'). Fitting uses
L = log P' for every alpha (the solver's alpha = 1 evaluation included); APPLYING alpha == 1 returns P exactly.
H-CLASS-TEMP classes for U = argmax(P) (= the teacher decision d). Applied to ALL rows.

TOLERANCES (registered; TOLERANCES): token32 = lra.decoder.TOLERANCES (STAT_TOL = DUAL_TOL = PROJ_TOL = 1e-9,
PROTO_SUM_TOL = 1e-12, BISECT_ITERS = 64, bracket <= 2 ulp, PBAR sum / class tol 1e-9); temperatures: bounds
[0.25, 4], TEMP_BISECT_ITERS = 100, TEMP_GRAD_TOL = 1e-9, TEMP_NLL_TOL = 1e-15, bracket <= 2 ulp, CLASS_MIN = 50,
LOG_FLOOR = 1e-12 (U log-input rule); released tables: |sum q - 1| <= 1e-12, strict argmax == token class; scoring clip
LOSS_CLIP = 1e-12 (dpc.utility.per_row: natural log; multiclass Brier sum_k (p_k - 1[y = k])^2).

API
  token32_solve(Y, PBAR, n, d)                         -> (U, Q, obj, certs)    lra.decoder solve with explicit PBAR
  fit_token32(tok, y, mu, q0, token_class, n_fit, K)   -> token table (H-TOKEN32 / T-TOKEN32 via kind=)
  solve_temperature(L, y)                              -> scalar temperature certificate (alpha, status, ...)
  temp_apply_tokens(q0, alpha, token_class=None)       -> (T, K) table;  temp_apply_tokens_by_class(q0, alphas, tc)
  fit_global_temp(tok, y, q0, token_class, K) / fit_class_temp(...)            -> token tables
  u_log_inputs(P); temp_apply_probs(P, alpha); fit_u_temp(P, y, family, d=None); apply_u(P, tab, d=None)
  row_losses(prob, y); nll_unclipped(prob, y); apply_table(table_q, tok); table_sha256(tab)
  decoder_table(kind, recipient, tab)                  -> uniform JSON-safe record
  calibrate_partition(bank, cal_rows, y_by_task, family) -> {1: income table (K=2), 2: occupation table (K=6)}
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from dpc import utility as DU
from lra import decoder as DC
from qpc.kmeans import check_decisions, check_probs

SCHEMA = "hcal-calib-v1"
# ----- token32: lra.decoder's constants and internals, imported unchanged (never re-defined here)
KAPPA = DC.KAPPA
EPS = DC.EPS
BISECT_ITERS = DC.BISECT_ITERS
PROJ_TOL = DC.PROJ_TOL
PROTO_SUM_TOL = DC.PROTO_SUM_TOL
PBAR_SUM_TOL = DC.STATS_SUM_TOL              # |sum PBAR - 1| (lra: |sum s - n| <= 1e-9 n)
TEACHER_CLASS_TOL = DC.TEACHER_CLASS_TOL     # PBAR[d] >= max PBAR - 1e-9
DecoderError = DC.DecoderError
smooth = DC.smooth
assert (KAPPA, EPS, BISECT_ITERS, PROJ_TOL, PROTO_SUM_TOL, PBAR_SUM_TOL, TEACHER_CLASS_TOL) == \
    (32.0, 1e-12, 64, 1e-9, 1e-12, 1e-9, 1e-9), "lra.decoder constants differ from the registered values"
# ----- temperatures
TEMP_LO, TEMP_HI = 0.25, 4.0
TEMP_BISECT_ITERS = 100
TEMP_GRAD_TOL = 1e-9          # interior: |g(alpha)| <= TEMP_GRAD_TOL * scale
TEMP_NLL_TOL = 1e-15          # NLL(alpha) <= NLL(1) + TEMP_NLL_TOL * scale
MAX_BRACKET_ULPS = 2.0
CLASS_MIN = 50
LOG_FLOOR = 1e-12             # U log-input rule
LOSS_CLIP = DU.CLIP           # 1e-12 scoring clip (dpc.utility.per_row)
assert LOSS_CLIP == 1e-12
RECIPIENT_K = {1: 2, 2: 6}
TASK_OF = {1: "income", 2: "occupation"}
TOKEN_FAMILIES = ("H-TOKEN32", "T-TOKEN32", "H-GLOBAL-TEMP", "H-CLASS-TEMP")
TEMP_FAMILIES = ("H-GLOBAL-TEMP", "H-CLASS-TEMP")
FITTED = "FITTED"
NO_CAL = "NO_CALIBRATION_OBSERVATIONS"
RESERVED = "RESERVED_EMPTY_ORIGINAL_FALLBACK"
CLASS_FALLBACK = "CLASS_FALLBACK_LT50"
TOKEN_STATUSES = (FITTED, NO_CAL, RESERVED)
TEMP_STATUSES = ("INTERIOR", "BOUNDARY_LOW", "BOUNDARY_HIGH")
TOLERANCES = {"token32": dict(DC.TOLERANCES), "PBAR_SUM_TOL": PBAR_SUM_TOL, "TEACHER_CLASS_TOL": TEACHER_CLASS_TOL,
              "temp_bounds": [TEMP_LO, TEMP_HI], "TEMP_BISECT_ITERS": TEMP_BISECT_ITERS, "TEMP_GRAD_TOL": TEMP_GRAD_TOL,
              "TEMP_NLL_TOL": TEMP_NLL_TOL, "max_bracket_ulps": MAX_BRACKET_ULPS, "CLASS_MIN": CLASS_MIN,
              "LOG_FLOOR": LOG_FLOOR, "LOSS_CLIP": LOSS_CLIP, "table_sum_tol": PROTO_SUM_TOL}


class CalibrationError(ValueError):
    """A calibration output that fails its certificate or a release invariant (refused; never released)."""


# ----------------------------------------------------------------------------------------------- validation
def _labels(y, K, name="y"):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if y.size and not (np.issubdtype(y.dtype, np.integer) or np.all(y == np.round(y))):
        raise ValueError(f"{name} must be integer class labels")
    y = y.astype(np.int64)
    if y.size and y.min() < 0:
        raise PermissionError(f"REFUSED: negative (sealed) labels in {name}")
    if y.size and y.max() >= K:
        raise ValueError(f"{name} out of range [0, {K})")
    return y


def _tokens(tok, T, name="tok"):
    tok = np.asarray(tok)
    if tok.ndim != 1 or not (np.issubdtype(tok.dtype, np.integer) or np.all(tok == np.round(tok))):
        raise ValueError(f"{name} must be 1-D integer token ids")
    tok = tok.astype(np.int64)
    if tok.size and (tok.min() < 0 or tok.max() >= T):
        raise ValueError(f"{name} out of range [0, {T})")
    return tok


def _strict_argmax_ok(Q, cls):
    """Boolean per row: Q[cls] > Q[k] for every k != cls (strict, float64)."""
    rows = np.arange(Q.shape[0])
    other = Q.copy()
    other[rows, cls] = -np.inf
    return Q[rows, cls] > other.max(1)


def _check_table(Q, tc, name):
    """Released-table invariants: finite, >= 0, |sum - 1| <= 1e-12 (k-ordered sum), strict argmax == token class."""
    T, K = Q.shape
    if not np.all(np.isfinite(Q)) or np.any(Q < 0):
        raise CalibrationError(f"{name}: non-finite or negative released probabilities")
    if T and np.max(np.abs(DC._rowsum(Q, K) - 1.0)) > PROTO_SUM_TOL:
        raise CalibrationError(f"{name}: a released vector is not normalised within {PROTO_SUM_TOL}")
    bad = ~_strict_argmax_ok(Q, tc)
    if bad.any():
        raise CalibrationError(f"{name}: {int(bad.sum())} token(s) whose strict argmax is not the token class "
                               f"(first token {int(np.flatnonzero(bad)[0])})")


def _check_q0(q0, token_class, K=None):
    q0 = np.atleast_2d(np.asarray(q0, dtype=np.float64))
    T, Kq = q0.shape
    if K is not None and Kq != K:
        raise ValueError(f"q0 has {Kq} columns; expected K={K}")
    if Kq < 2:
        raise ValueError("K must be >= 2")
    if token_class is None:
        tc = q0.argmax(1).astype(np.int64)
    else:
        tc = np.asarray(token_class)
        if tc.shape != (T,) or not (np.issubdtype(tc.dtype, np.integer) or np.all(tc == np.round(tc))):
            raise ValueError("token_class must be (T,) integer classes")
        tc = tc.astype(np.int64)
        if T and (tc.min() < 0 or tc.max() >= Kq):
            raise ValueError("token_class out of range")
    if not np.all(np.isfinite(q0)) or np.any(q0 <= 0):
        raise ValueError("q0 must be finite and strictly positive (smoothed vectors)")
    _check_table(q0, tc, "q0 (frozen original vectors)")
    return q0 + 0.0, tc, T, Kq


# ----------------------------------------------------------------------------------------------- H-TOKEN32
def _check_token_inputs(Y, PBAR, n, d, kappa, eps):
    """lra.decoder._check_inputs with PBAR given explicitly instead of S / n (same conversions, same order)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64)) + 0.0
    PBAR = np.atleast_2d(np.asarray(PBAR, dtype=np.float64)) + 0.0
    n = np.atleast_1d(np.asarray(n))
    d = np.atleast_1d(np.asarray(d))
    M, K = Y.shape
    if PBAR.shape != (M, K) or n.shape != (M,) or d.shape != (M,):
        raise ValueError(f"shapes: Y {Y.shape}, PBAR {PBAR.shape}, n {n.shape}, d {d.shape} are not aligned")
    if K < 2:
        raise ValueError("K must be >= 2")
    if float(kappa) != KAPPA or float(eps) != EPS:
        raise ValueError(f"kappa and eps are fixed by the protocol ({KAPPA}, {EPS}); got ({kappa}, {eps})")
    if not (np.issubdtype(n.dtype, np.integer) or np.all(n == np.round(n))):
        raise ValueError("n must be integer counts")
    n = n.astype(np.int64)
    if not (np.issubdtype(d.dtype, np.integer) or np.all(d == np.round(d))):
        raise ValueError("d must be integer class indices")
    d = d.astype(np.int64)
    if np.any(n < 1):
        raise ValueError("n_t = 0 has no calibration statistics: the registered fallback applies (fit_token32)")
    if np.any((d < 0) | (d >= K)):
        raise ValueError("d out of range")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(PBAR))):
        raise ValueError("non-finite statistics (a reserved empty token's NaN prior must never be solved)")
    if np.any(Y < 0) or np.any(Y != np.round(Y)):
        raise ValueError("label counts y must be nonnegative integers")
    if np.any(DC._rowsum(Y, K) != n):
        raise ValueError("label counts must sum to n exactly (no labels are invented or dropped)")
    if np.any(PBAR < 0) or np.any(np.abs(DC._rowsum(PBAR, K) - 1.0) > PBAR_SUM_TOL):
        raise ValueError(f"PBAR rows must be nonnegative and sum to 1 within {PBAR_SUM_TOL}")
    if np.any(PBAR[np.arange(M), d] < PBAR.max(1) - TEACHER_CLASS_TOL):
        raise ValueError("d is not the teacher-predicted class of the token (PBAR[d] is not maximal)")
    return Y, PBAR, n, d, M, K


def token32_solve(Y, PBAR, n, d, kappa=KAPPA, eps=EPS):
    """M independent H-TOKEN32 token problems: lra.decoder.solve_batch's exact certified solve with the prior PBAR
    (M, K) given explicitly (A = Y + 32 PBAR). Returns (U, Q, obj, certs) in solve_batch's format; raises
    DecoderError if any row is not certified. Bitwise equal to solve_batch(Y, S, n, d) when PBAR = S / n[:, None]."""
    Y, PBAR, n, d, M, K = _check_token_inputs(Y, PBAR, n, d, kappa, eps)
    nf = n.astype(np.float64)
    Z = 1.0 + (K + 1) * EPS
    rows = np.arange(M)
    C = np.full((M, K), EPS)
    C[rows, d] = 2.0 * EPS
    A = Y + KAPPA * PBAR
    lo = np.full(M, 1.0 / K)
    hi = np.ones(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            Um, _nu = DC._state(mid, A, Y, nf, C, d, Z, K, M)
            R = DC._rowsum(Um, K) - 1.0
            neg = R <= 0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
        Ul, _nul = DC._state(lo, A, Y, nf, C, d, Z, K, M)
        Uh, _nuh = DC._state(hi, A, Y, nf, C, d, Z, K, M)
    Rl, Rh = np.abs(DC._rowsum(Ul, K) - 1.0), np.abs(DC._rowsum(Uh, K) - 1.0)
    take_hi = Rh < Rl
    t = np.where(take_hi, hi, lo)
    Uraw = np.where(take_hi[:, None], Uh, Ul)
    ulps = (hi - lo) / np.spacing(lo)
    Ufin, proj = DC.project_class_simplex(Uraw, d)
    Ufin = np.atleast_2d(Ufin)
    proj = np.atleast_1d(proj)
    Qm = smooth(Ufin, d, check=False)
    obj, certs = DC._certify(Ufin, Qm, A, Y, nf, PBAR, C, d, Z, K, M)
    certs["t"] = t
    certs["bracket_ulps"] = ulps
    certs["projection_magnitude"] = proj
    conv = (certs["converged_kkt"] & (ulps <= MAX_BRACKET_ULPS) & (proj <= PROJ_TOL))
    certs["converged"] = conv
    del certs["converged_kkt"]
    if not np.all(conv):
        bad = np.flatnonzero(~conv)
        i = int(bad[0])
        raise DecoderError(f"{bad.size} H-TOKEN32 token solve(s) not certified; first row {i}: "
                           + json.dumps({f: DC._py(certs[f][i]) for f in DC.CERT_FIELDS}))
    return Ufin, Qm, obj, certs


def _token_summary(status, certs, K):
    """Counts by status, parameter count and the worst certificate values (lra.decoder.certificate_summary keys, so
    lra.decoder.summary_violations applies unchanged)."""
    sup = [c for c in certs if c["status"] == FITTED]
    mx = lambda f: max((c[f] for c in sup), default=None)  # noqa: E731
    counts = {s: int(sum(1 for x in status if x == s)) for s in TOKEN_STATUSES}
    out = {"tokens": len(status), "status_counts": counts, "supervised_tokens": len(sup),
           "fallback_tokens": len(status) - len(sup), "parameter_count": len(sup) * (K - 1),
           "all_converged": all(c["converged"] for c in sup),
           "max_stationarity_rel": mx("stationarity_rel"), "max_dual_infeas_rel": mx("dual_infeas_rel"),
           "max_projection_magnitude": mx("projection_magnitude"), "max_sum_q_residual": mx("sum_q_residual"),
           "min_margin": min((c["margin"] for c in sup), default=None),
           "tokens_with_tie": int(sum(1 for c in sup if c["n_tie"] > 0)),
           "tokens_with_zero": int(sum(1 for c in sup if c["n_zero"] > 0)),
           "max_bracket_ulps": mx("bracket_ulps")}
    out["violations"] = DC.summary_violations(out)
    return out


def fit_token32(tok, y, mu, q0, token_class, n_fit, K, kind="H-TOKEN32"):
    """H-TOKEN32 (CALIBRATION_HELDOUT rows) or T-TOKEN32 (CALIBRATION_TRAIN_MATCHED rows) table of one recipient.
    tok, y: the calibration representatives' tokens and labels; mu (T, K): fixed original unsmoothed teacher means
    (NaN rows exactly for reserved empty tokens, n_fit == 0); q0 (T, K): frozen smoothed D0 vectors."""
    if kind not in ("H-TOKEN32", "T-TOKEN32"):
        raise ValueError(f"fit_token32 kind must be H-TOKEN32 or T-TOKEN32; got {kind!r}")
    q0, tc, T, K = _check_q0(q0, token_class, K)
    n_fit = np.asarray(n_fit)
    if n_fit.shape != (T,) or not np.issubdtype(n_fit.dtype, np.integer) or np.any(n_fit < 0):
        raise ValueError("n_fit must be (T,) nonnegative integer fitting counts")
    n_fit = n_fit.astype(np.int64)
    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (T, K):
        raise ValueError(f"mu has shape {mu.shape}; expected {(T, K)}")
    res = n_fit == 0
    if not np.all(np.isnan(mu[res])) or not np.all(np.isfinite(mu[~res])):
        raise ValueError("mu must be NaN exactly on the reserved empty tokens (n_fit == 0) and finite elsewhere")
    tok = _tokens(tok, T)
    y = _labels(y, K)
    if y.shape != tok.shape:
        raise ValueError("labels not aligned with the calibration tokens")
    n_cal = np.bincount(tok, minlength=T).astype(np.int64)
    y_cal = np.bincount(tok * K + y, minlength=T * K).reshape(T, K).astype(np.int64)
    status = [RESERVED if res[t] else (NO_CAL if n_cal[t] == 0 else FITTED) for t in range(T)]
    fit = np.asarray([s == FITTED for s in status], dtype=bool)
    q = q0.copy()
    U = np.full((T, K), np.nan)
    certs = [None] * T
    sup = np.flatnonzero(fit)
    if sup.size:
        Us, Qs, _obj, cb = token32_solve(y_cal[sup].astype(np.float64), mu[sup], n_cal[sup], tc[sup])
        U[sup], q[sup] = Us, Qs
        for j, t in enumerate(sup):
            certs[t] = {"status": FITTED, "token": int(t), "n_cal": int(n_cal[t]), **DC.cert_row(cb, j)}
    for t in np.flatnonzero(~fit):
        certs[t] = {"status": status[t], "token": int(t), "n_cal": int(n_cal[t]),
                    "fallback": "q0[t] exactly (frozen original vector); never fitted"}
    if not np.array_equal(q[~fit], q0[~fit]):
        raise CalibrationError("a fallback token does not carry its frozen original vector exactly")
    _check_table(q, tc, kind)
    summ = _token_summary(status, certs, K)
    if summ["violations"]:
        raise DecoderError(f"{kind}: certificate violations {summ['violations']}")
    tab = {"kind": kind, "target": "tokens", "K": int(K), "T": int(T), "q": q, "u": U, "token_class": tc,
           "n_fit": n_fit, "n_cal": n_cal, "y_cal": y_cal, "status": status, "status_counts": summ["status_counts"],
           "parameter_count": summ["parameter_count"], "alpha": None, "alphas": None, "certs": certs, "summary": summ,
           "certificate": {"summary": summ, "tokens": certs}, "prior_sha256": _sha_arrays([("mu", mu)]),
           "calibration_scores": _scores(q[tok], q0[tok], y)}
    tab["content_sha256"] = table_sha256(tab)
    return tab


# ----------------------------------------------------------------------------------------------- temperatures
def _softmax_rows(Z):
    """exp(Z - max) / sum (k-ordered); row-independent (elementwise + explicit k loop)."""
    K = Z.shape[1]
    E = np.exp(Z - Z.max(1)[:, None])
    return E / DC._rowsum(E, K)[:, None]


def _mean(x):
    return math.fsum(np.asarray(x, dtype=np.float64).tolist()) / x.shape[0]


def temp_terms(L, y, alpha):
    """(NLL, g, curvature) at alpha: UNCLIPPED mean NLL, its derivative and its second derivative (module docstring)."""
    N, K = L.shape
    rows = np.arange(N)
    Z = alpha * L
    zmax = Z.max(1)
    E = np.exp(Z - zmax[:, None])
    s = DC._rowsum(E, K)
    P = E / s[:, None]
    lse = zmax + np.log(s)
    m = DC._rowsum(P * L, K)
    v = DC._rowsum(P * (L - m[:, None]) ** 2, K)
    return _mean(lse - Z[rows, y]), _mean(m - L[rows, y]), _mean(v)


def _check_log_inputs(L, y):
    L = np.atleast_2d(np.asarray(L, dtype=np.float64))
    N, K = L.shape
    if N < 1:
        raise CalibrationError("a temperature needs at least one calibration row")
    if K < 2 or not np.all(np.isfinite(L)) or np.any(L > 0):
        raise ValueError("L must be finite log-probabilities (<= 0), K >= 2")
    return L, _labels(y, K), N, K


def solve_temperature(L, y):
    """Bounded deterministic fit of one inverse temperature on log-probabilities L (N, K) with labels y. Returns the
    certificate dict (alpha, status, g, curvature, NLLs, bracket); raises CalibrationError if not certified."""
    L, y, N, K = _check_log_inputs(L, y)
    scale = _mean(np.abs(L).max(1))
    if not scale > 0:
        raise CalibrationError("degenerate log-probabilities (scale = 0)")
    g_lo_end = temp_terms(L, y, TEMP_LO)[1]
    g_hi_end = temp_terms(L, y, TEMP_HI)[1]
    lo = hi = None
    halvings = 0
    if g_lo_end >= 0:
        alpha, status = TEMP_LO, "BOUNDARY_LOW"
    elif g_hi_end <= 0:
        alpha, status = TEMP_HI, "BOUNDARY_HIGH"
    else:
        lo, hi, glo, ghi = TEMP_LO, TEMP_HI, g_lo_end, g_hi_end
        for _ in range(TEMP_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if not (lo < mid < hi):
                break
            gm = temp_terms(L, y, mid)[1]
            halvings += 1
            if gm <= 0:
                lo, glo = mid, gm
            else:
                hi, ghi = mid, gm
        alpha = hi if abs(ghi) < abs(glo) else lo
        status = "INTERIOR"
    nll_a, g_a, curv_a = temp_terms(L, y, alpha)
    nll_1, g_1, _c1 = temp_terms(L, y, 1.0)
    ulps = float((hi - lo) / np.spacing(lo)) if status == "INTERIOR" else None
    if status == "INTERIOR":
        kkt_ok = abs(g_a) <= TEMP_GRAD_TOL * scale and ulps <= MAX_BRACKET_ULPS
        kkt = "|g(alpha)| <= TEMP_GRAD_TOL * scale and bracket <= 2 ulp"
    elif status == "BOUNDARY_LOW":
        kkt_ok, kkt = g_a >= 0, "g(0.25) >= 0"
    else:
        kkt_ok, kkt = g_a <= 0, "g(4) <= 0"
    nll_ok = nll_a <= nll_1 + TEMP_NLL_TOL * scale
    cert = {"alpha": float(alpha), "status": status, "n": int(N), "K": int(K), "scale": scale, "g_alpha": g_a,
            "grad_rel": abs(g_a) / scale, "g_at_low_bound": g_lo_end, "g_at_high_bound": g_hi_end, "g_at_one": g_1,
            "curvature_alpha": curv_a, "nll_alpha": nll_a, "nll_one": nll_1, "nll_decrease_vs_one": nll_1 - nll_a,
            "bracket": None if lo is None else [float(lo), float(hi)], "bracket_ulps": ulps, "halvings": halvings,
            "kkt_condition": kkt, "kkt_ok": bool(kkt_ok), "curvature_ok": bool(curv_a >= 0), "nll_ok": bool(nll_ok),
            "identity": bool(alpha == 1.0), "method": "bounded bisection on the convex NLL derivative"}
    cert["certified"] = bool(kkt_ok and curv_a >= 0 and nll_ok)
    if not cert["certified"]:
        raise CalibrationError("temperature not certified: " + json.dumps(cert))
    return cert


def _check_alpha(alpha):
    a = float(alpha)
    if not (np.isfinite(a) and TEMP_LO <= a <= TEMP_HI):
        raise ValueError(f"alpha {alpha!r} outside the registered bounds [{TEMP_LO}, {TEMP_HI}]")
    return a


def temp_apply_tokens(q0, alpha, token_class=None):
    """Shared-temperature token table q_alpha = softmax(alpha log q0) (alpha == 1: q0 exactly, a copy). The strict
    argmax of every token must stay its token class (token_class, or the strict argmax of q0); raises otherwise."""
    q0, tc, T, K = _check_q0(q0, token_class)
    a = _check_alpha(alpha)
    if a == 1.0:
        return q0.copy()
    q = _softmax_rows(a * np.log(q0))
    _check_table(q, tc, f"temperature alpha={a!r}")
    return q


def temp_apply_tokens_by_class(q0, alphas, token_class):
    """Per-class temperature table: token t gets alphas[token_class[t]] (alpha == 1: q0[t] exactly)."""
    q0, tc, T, K = _check_q0(q0, token_class)
    al = [_check_alpha(a) for a in alphas]
    if len(al) != K:
        raise ValueError(f"alphas must have K={K} entries")
    q = q0.copy()
    for c in range(K):
        m = tc == c
        if al[c] != 1.0 and m.any():
            q[m] = _softmax_rows(al[c] * np.log(q0[m]))
    _check_table(q, tc, "class temperatures")
    return q


def _class_fit(L, y, cls, K):
    """Per-class temperatures on calibration rows with predicted class cls: (alphas, per-class records)."""
    if L.shape[0] < 1:
        raise CalibrationError("a temperature family needs at least one calibration row (no silent all-identity fit)")
    alphas, recs = [], []
    for c in range(K):
        m = cls == c
        nc = int(m.sum())
        if nc < CLASS_MIN:
            alphas.append(1.0)
            recs.append({"class": c, "n": nc, "status": CLASS_FALLBACK, "alpha": 1.0, "absent": nc == 0,
                         "rule": f"fewer than {CLASS_MIN} calibration representatives: alpha = 1 (identity)"})
        else:
            cert = solve_temperature(L[m], y[m])
            alphas.append(cert["alpha"])
            recs.append({**cert, "class": c, "n": nc, "status": FITTED, "solve_status": cert["status"]})
    return alphas, recs


def _temp_table(kind, q0, tc, tok, y, alpha, alphas, cert, status, counts, nparam):
    """Token table of a temperature family, with the scoring receipts of the APPLIED release on the calibration rows
    (clipped 1e-12 log loss and clipped counts at the fitted alpha(s) and at alpha = 1, i.e. q0 exactly)."""
    T, K = q0.shape
    q = temp_apply_tokens(q0, alpha, tc) if alphas is None else temp_apply_tokens_by_class(q0, alphas, tc)
    sc = _scores(q[tok], q0[tok], y)
    cert.update({"score_ll_alpha": sc["ll_clipped_fitted"], "score_ll_one": sc["ll_clipped_original"],
                 "clipped_count_alpha": sc["clipped_count_fitted"], "clipped_count_one": sc["clipped_count_original"]})
    tab = {"kind": kind, "target": "tokens", "K": int(K), "T": int(T), "q": q, "u": None, "token_class": tc,
           "n_fit": None, "n_cal": np.bincount(tok, minlength=T).astype(np.int64),
           "y_cal": np.bincount(tok * K + y, minlength=T * K).reshape(T, K).astype(np.int64), "status": status,
           "status_counts": counts, "parameter_count": int(nparam), "alpha": alpha, "alphas": alphas,
           "certificate": cert, "prior_sha256": None, "calibration_scores": sc}
    tab["content_sha256"] = table_sha256(tab)
    return tab


def fit_global_temp(tok, y, q0, token_class, K):
    """H-GLOBAL-TEMP table of one recipient: one alpha on all calibration representatives, applied to every token."""
    q0, tc, T, K = _check_q0(q0, token_class, K)
    tok = _tokens(tok, T)
    y = _labels(y, K)
    if y.shape != tok.shape:
        raise ValueError("labels not aligned with the calibration tokens")
    cert = solve_temperature(np.log(q0[tok]), y)
    return _temp_table("H-GLOBAL-TEMP", q0, tc, tok, y, cert["alpha"], None, cert, [cert["status"]],
                       {cert["status"]: 1}, 1)


def fit_class_temp(tok, y, q0, token_class, K):
    """H-CLASS-TEMP table of one recipient: one alpha per predicted class (>= 50 representatives, else alpha = 1)."""
    q0, tc, T, K = _check_q0(q0, token_class, K)
    tok = _tokens(tok, T)
    y = _labels(y, K)
    if y.shape != tok.shape:
        raise ValueError("labels not aligned with the calibration tokens")
    alphas, recs = _class_fit(np.log(q0[tok]), y, tc[tok], K)
    status = [r["status"] for r in recs]
    counts = {FITTED: status.count(FITTED), CLASS_FALLBACK: status.count(CLASS_FALLBACK)}
    cert = {"classes": recs}
    return _temp_table("H-CLASS-TEMP", q0, tc, tok, y, None, alphas, cert, status, counts, counts[FITTED])


# ----------------------------------------------------------------------------------------------- continuous U
def u_log_inputs(P):
    """Frozen log-input rule: log P', P' = max(P, 1e-12) / sum_k max(P, 1e-12) (k-ordered)."""
    P = check_probs(P)
    Pp = np.maximum(P, LOG_FLOOR)
    Pp = Pp / DC._rowsum(Pp, Pp.shape[1])[:, None]
    return np.log(Pp)


def _check_u_out(q, d, name, transformed=None):
    """argmax(q) == d on EVERY row; finiteness and |sum q - 1| <= PROTO_SUM_TOL on the TRANSFORMED rows (rows with
    alpha == 1 are the caller's P exactly, validated by check_probs at its own 1e-9 tolerance; role A integration of
    MATH_REVIEW finding 2, before ENGINEERING_LOCK)."""
    mis = int(np.sum(q.argmax(1) != d))
    if mis:
        raise CalibrationError(f"{name}: argmax(q) differs from the teacher decision on {mis} row(s)")
    qt = q if transformed is None else q[np.asarray(transformed, dtype=bool)]
    if not np.all(np.isfinite(qt)) or (qt.shape[0] and np.max(np.abs(DC._rowsum(qt, qt.shape[1]) - 1.0))
                                       > PROTO_SUM_TOL):
        raise CalibrationError(f"{name}: released U probabilities not finite / not normalised within {PROTO_SUM_TOL}")
    return mis


def temp_apply_probs(P, alpha, d=None):
    """Shared temperature on continuous U rows: alpha == 1 -> P exactly (copy); else softmax(alpha log P').
    argmax(q) must equal the teacher decision d (= argmax P) on every row; raises otherwise."""
    Pc = check_probs(P)
    d = check_decisions(Pc, d)
    a = _check_alpha(alpha)
    if a == 1.0:
        return np.array(P, dtype=np.float64, copy=True)
    q = _softmax_rows(a * u_log_inputs(Pc))
    _check_u_out(q, d, f"U temperature alpha={a!r}")
    return q


def apply_u(P, tab, d=None):
    """Apply a fitted U calibration record (fit_u_temp) to ALL rows of P (n, K)."""
    if tab.get("target") != "U":
        raise ValueError("not a U calibration table")
    if tab["alphas"] is None:
        return temp_apply_probs(P, tab["alpha"], d)
    Pc = check_probs(P, tab["K"])
    d = check_decisions(Pc, d)
    al = [_check_alpha(a) for a in tab["alphas"]]
    q = np.array(P, dtype=np.float64, copy=True)
    L = u_log_inputs(Pc)
    tr = np.zeros(len(d), dtype=bool)
    for c in range(tab["K"]):
        m = d == c
        if al[c] != 1.0 and m.any():
            q[m] = _softmax_rows(al[c] * L[m])
            tr |= m
    _check_u_out(q, d, "U class temperatures", transformed=tr)
    return q


def fit_u_temp(P, y, family, d=None):
    """H-GLOBAL-TEMP / H-CLASS-TEMP of continuous U on the calibration representatives (P (n, K), labels y). The U
    record has no token table (q = None, T = None): n_cal (K,) = calibration rows per predicted class and y_cal (K, K)
    = predicted class x true label counts. Apply with apply_u (every row)."""
    if family not in TEMP_FAMILIES:
        raise ValueError(f"U family must be one of {TEMP_FAMILIES}")
    Pc = check_probs(P)
    K = Pc.shape[1]
    d = check_decisions(Pc, d)
    y = _labels(y, K)
    if y.shape != d.shape:
        raise ValueError("labels not aligned with the calibration rows")
    L = u_log_inputs(Pc)
    if family == "H-GLOBAL-TEMP":
        cert = solve_temperature(L, y)
        alpha, alphas, status, counts, nparam = cert["alpha"], None, [cert["status"]], {cert["status"]: 1}, 1
    else:
        alphas, recs = _class_fit(L, y, d, K)
        status = [r["status"] for r in recs]
        counts = {FITTED: status.count(FITTED), CLASS_FALLBACK: status.count(CLASS_FALLBACK)}
        alpha, nparam, cert = None, counts[FITTED], {"classes": recs}
    tab = {"kind": family, "target": "U", "K": int(K), "T": None, "q": None, "u": None, "token_class": None,
           "n_fit": None, "n_cal": np.bincount(d, minlength=K).astype(np.int64),
           "y_cal": np.bincount(d * K + y, minlength=K * K).reshape(K, K).astype(np.int64), "status": status,
           "status_counts": counts, "parameter_count": int(nparam), "alpha": alpha, "alphas": alphas,
           "certificate": cert, "prior_sha256": None}
    q = apply_u(Pc, tab, d)
    tab["calibration_scores"] = _scores(q, np.array(P, dtype=np.float64), y)
    sc = tab["calibration_scores"]
    cert.update({"score_ll_alpha": sc["ll_clipped_fitted"], "score_ll_one": sc["ll_clipped_original"],
                 "clipped_count_alpha": sc["clipped_count_fitted"], "clipped_count_one": sc["clipped_count_original"]})
    tab["content_sha256"] = table_sha256(tab)
    return tab


# ----------------------------------------------------------------------------------------------- losses / helpers
def row_losses(prob, y):
    """Per-row (ll, brier) with the source convention (dpc.utility.per_row): ll = -log clip(p_y, 1e-12, 1) (natural
    log), brier = sum_k (p_k - 1[y = k])^2."""
    P = np.asarray(prob, dtype=np.float64)
    pr = DU.per_row(P, _labels(y, P.shape[1]), P.shape[1])
    return pr["ll"], pr["br"]


def nll_unclipped(prob, y):
    """Per-row UNCLIPPED true-label negative log-likelihood -log p_y (natural log; +inf where p_y = 0)."""
    P = np.asarray(prob, dtype=np.float64)
    y = _labels(y, P.shape[1])
    with np.errstate(divide="ignore"):
        return -np.log(P[np.arange(y.shape[0]), y])


def _scores(Qrows, Q0rows, y):
    """Descriptive calibration-row receipts: clipped scoring log loss of the fitted and the original release, unclipped
    NLL, and the count of scored (true-label) probabilities below the 1e-12 clip."""
    N = int(y.shape[0])
    if N == 0:
        return {"n": 0}
    out = {"n": N}
    for nm, Q in (("fitted", Qrows), ("original", Q0rows)):
        ll, _br = row_losses(Q, y)
        py = Q[np.arange(N), y]
        un = nll_unclipped(Q, y)
        out[f"ll_clipped_{nm}"] = _mean(ll)
        out[f"nll_unclipped_{nm}"] = _mean(un) if np.all(np.isfinite(un)) else None
        out[f"clipped_count_{nm}"] = int(np.sum(py < LOSS_CLIP))
    return out


def apply_table(table_q, tok):
    """Per-row released vectors table_q[tok] (float64 copy)."""
    Q = np.asarray(table_q, dtype=np.float64)
    return Q[_tokens(tok, Q.shape[0])].copy()


def _sha_arrays(items, header=None):
    h = hashlib.sha256()
    if header is not None:
        h.update(json.dumps(header, sort_keys=True, allow_nan=False).encode())
    for name, a in items:
        if a is None:
            h.update(f"{name}:None|".encode())
            continue
        a = np.ascontiguousarray(np.asarray(a))
        a = a.astype("<f8") if a.dtype.kind == "f" else a.astype("<i8")
        h.update(f"{name}:{a.dtype.str}:{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def table_sha256(tab):
    """Content hash: kind, target, K, T, alpha(s), statuses, parameter count, prior hash and the arrays q, token_class,
    n_cal, y_cal (recipient excluded)."""
    header = {"schema": SCHEMA, "kind": tab["kind"], "target": tab["target"], "K": tab["K"], "T": tab["T"],
              "alpha": None if tab["alpha"] is None else float(tab["alpha"]).hex(),
              "alphas": None if tab["alphas"] is None else [float(a).hex() for a in tab["alphas"]],
              "status": list(tab["status"]), "parameter_count": tab["parameter_count"],
              "prior_sha256": tab["prior_sha256"]}
    return _sha_arrays([("q", tab["q"]), ("token_class", tab["token_class"]), ("n_cal", tab["n_cal"]),
                        ("y_cal", tab["y_cal"])], header)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if not np.isfinite(v):
            raise CalibrationError("non-finite value in a calibration record (never silently dropped)")
        return v
    return x


def decoder_table(kind, recipient, tab):
    """Uniform JSON-safe record of one fitted table: {schema, kind, target, recipient, task, K, T, alpha / alphas, q,
    token_class, n_cal, y_cal, u (fitted token32 rows; None elsewhere), status, status_counts, parameter_count,
    certificate, calibration_scores, prior_sha256, tolerances, content_sha256}. The content hash is re-verified."""
    if tab.get("kind") != kind:
        raise ValueError(f"table kind {tab.get('kind')!r} is not {kind!r}")
    if int(recipient) not in RECIPIENT_K or RECIPIENT_K[int(recipient)] != tab["K"]:
        raise ValueError(f"recipient {recipient} requires K={RECIPIENT_K.get(int(recipient))}; table has K={tab['K']}")
    if table_sha256(tab) != tab["content_sha256"]:
        raise CalibrationError("table content hash mismatch")
    u = None
    if tab["u"] is not None:
        u = [None if np.isnan(r).any() else r.tolist() for r in tab["u"]]
    rec = {"schema": SCHEMA, "kind": kind, "target": tab["target"], "recipient": int(recipient),
           "task": TASK_OF[int(recipient)], "K": tab["K"], "T": tab["T"], "alpha": tab["alpha"],
           "alphas": tab["alphas"],
           "q": None if tab["q"] is None else tab["q"].tolist(),
           "token_class": None if tab["token_class"] is None else tab["token_class"].tolist(),
           "n_cal": tab["n_cal"].tolist(), "y_cal": tab["y_cal"].tolist(), "u": u, "status": list(tab["status"]),
           "status_counts": tab["status_counts"], "parameter_count": tab["parameter_count"],
           "certificate": _jsonable(tab["certificate"]), "calibration_scores": _jsonable(tab["calibration_scores"]),
           "prior_sha256": tab["prior_sha256"], "tolerances": TOLERANCES, "content_sha256": tab["content_sha256"]}
    json.dumps(rec, allow_nan=False)
    return rec


def calibrate_partition(bank, cal_rows, y_by_task, family):
    """Both recipients' tables of one family for one frozen-bank partition (hcal.admit.build_bank arrays: tok_i,
    class_i, n_fit_i, mu_i, q0_i; hard_i checked when present). cal_rows: row positions of the calibration
    representatives (CALIBRATION_HELDOUT for H-*, CALIBRATION_TRAIN_MATCHED for T-TOKEN32);
    y_by_task = {"income": y, "occupation": y} aligned with cal_rows.
    Returns {1: income table (K=2), 2: occupation table (K=6)}."""
    if family not in TOKEN_FAMILIES:
        raise ValueError(f"family must be one of {TOKEN_FAMILIES}")
    cal_rows = np.asarray(cal_rows)
    if cal_rows.ndim != 1 or not np.issubdtype(cal_rows.dtype, np.integer):
        raise ValueError("cal_rows must be 1-D integer row positions")
    cal_rows = cal_rows.astype(np.int64)
    if np.unique(cal_rows).size != cal_rows.size:
        raise ValueError("cal_rows must be distinct representatives")
    out = {}
    for i in (1, 2):
        K = RECIPIENT_K[i]
        tok_all = np.asarray(bank[f"tok{i}"], dtype=np.int64)
        tc = np.asarray(bank[f"class{i}"], dtype=np.int64)
        if cal_rows.size and (cal_rows.min() < 0 or cal_rows.max() >= tok_all.shape[0]):
            raise ValueError("cal_rows out of range of the bank rows")
        if f"hard{i}" in bank and not np.array_equal(tc[tok_all], np.asarray(bank[f"hard{i}"])):
            raise ValueError(f"bank recipient {i}: token classes differ from the released decisions")
        y = np.asarray(y_by_task[TASK_OF[i]])
        if y.shape != cal_rows.shape:
            raise ValueError(f"{TASK_OF[i]} labels not aligned with cal_rows")
        tok = tok_all[cal_rows]
        q0 = bank[f"q0{i}"]
        if family in ("H-TOKEN32", "T-TOKEN32"):
            out[i] = fit_token32(tok, y, bank[f"mu{i}"], q0, tc, bank[f"n_fit{i}"], K, kind=family)
        elif family == "H-GLOBAL-TEMP":
            out[i] = fit_global_temp(tok, y, q0, tc, K)
        else:
            out[i] = fit_class_temp(tok, y, q0, tc, K)
    return out
