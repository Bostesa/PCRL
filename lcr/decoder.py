"""Common learned decoder D1 of the lcr study (role B; prompt sec. 6). Shared by EVERY new arm and every fixed-map
calibration control. SYNTHETIC tests only in this module's test file; no data is loaded here.

PER-TOKEN PROBLEM (token t of recipient i; K classes; fitting rows of the token only)
    n    fitting count (int, >= 1 for a supervised solve)
    y    fitting true-label class counts (float64[K], integer-valued, sum y = n)
    s    sum of the teacher probability vectors over the token's fitting rows; pbar = s / n
    d    the teacher-predicted class common to the token
  Base vector u in the CLASS-DOMINANT SIMPLEX  C_d = {u : sum u = 1, u >= 0, u[d] >= u[k] for every k}.
  Released vector (the source smoothing, registered affine map; qpc.kmeans.smooth):
      q = (u + eps*1 + eps*e_d) / (1 + (K+1) eps),   eps = 1e-12     (Z := 1 + (K+1) eps, c_k := eps (1 + [k = d]))
  Objective (natural logs, float64):
      sum_k y_k (-log q_k) + 0.5 sum_rows ||q - onehot(Y)||^2 + kappa KL(pbar || q),   kappa = 32 (fixed; not tuned)
    = f(u) + const,   f(u) = -sum_k a_k log q_k + 0.5 n sum_k q_k^2 - sum_k y_k q_k,   a = y + kappa pbar,
      const = 0.5 n + kappa sum_{pbar_k > 0} pbar_k log pbar_k.
  No SEX enters this objective. (0 log 0 = 0: a class with a_k = 0 contributes only its quadratic terms.)

EXACT METHOD (deterministic; no general-purpose solver in the loop; PKG/MATH_REVIEW.md sec. 2-3)
  f is separable: f = sum_k phi_k(u_k), phi_k strictly convex (phi_k'' = (a_k / q_k^2 + n) / Z^2 >= n / Z^2 > 0).
  g_k(t) := phi_k'(t) = (-a_k / q_k(t) + n q_k(t) - y_k) / Z.   KKT with t = u_d and multiplier nu of sum u = 1:
    nu(t)  = max_{m = 0..K-1} -(g_d(t) + sum of the m smallest g_k(t), k != d) / (1 + m)      (exact pooling;
             = the unique root of g_d + nu + sum_{k != d} min(0, g_k + nu) = 0, piecewise linear in nu)
    u_k(t) = min(t, max(0, w_k(nu(t)))),  k != d,  where w_k(nu) solves phi_k'(w) = -nu in closed form:
             q = positive root of n q^2 + (nu Z - y_k) q - a_k = 0  (cancellation-free branch), w = Z q - c_k
    R(t)   = t + sum_{k != d} u_k(t) - 1 is continuous and strictly increasing on [1/K, 1], R(1/K) <= 0 <= R(1).
  The unique root t* of R is found by BISECTION with exactly BISECT_ITERS = 64 halvings from [1/K, 1] (this brackets
  t* between adjacent float64 values); t = the bracket end with the smaller |R| (ties -> the lower end). Every coordinate
  is then a KKT point of the strictly convex problem; the only primal residual left is R(t) (float rounding).
  Every reduction over classes is an explicit k-ordered loop and every array operation is elementwise, so a row's
  result does not depend on the batch it is solved in: solve_batch is bitwise identical to solve_token for u and q
  (solve_token IS solve_batch with M = 1; tested). The objective uses np.log (TOL_BATCH_OBJ registered for it).

FROZEN PROJECTION (feasibility repair of tiny residuals) and FINAL CERTIFICATE (prompt sec. 6)
  project_class_simplex(u, d): (1) u <- max(u, 0); (2) u_k <- min(u_k, u_d) for k != d; (3) u <- u / sum_k u_k
  (explicit k-ordered sum). Each step keeps the previous properties exactly in float64 (division by one positive
  float is monotone), so the result satisfies u >= 0 and u_d >= u_k EXACTLY; then q = smooth(u, d) has q_d > q_k
  strictly in float64 for u in [0, 1] (eps = 1e-12 >> ulp(1)). projection_magnitude = max |u_out - u_in|; a value
  above PROJ_TOL refuses the solve (a solver defect, not roundoff). The certificate is computed on the FINAL released
  q (and its u), never on an intermediate iterate:
    primal   sum_u_residual |sum u - 1|, sum_q_residual |sum q - 1| (must be <= PROTO_SUM_TOL = 1e-12),
             min_u (>= 0 exactly), margin = q_d - max_{k != d} q_k (must be > 0: strict argmax d; else REFUSED)
    active   zero set {k != d : u_k == 0}, tie set {k != d : u_k == u_d} (exact float equality)
    nu       = -(sum_{k in G} g_k) / |G|, G = {d} U tie set  (pooled multiplier at the final point)
    stationarity_rel = max_{free k} |g_k + nu| / scale; dual_infeas_rel = max(0, max_{zero} -(g_k + nu),
             max_{tie} (g_k + nu)) / scale; scale = max_k (a_k / q_k + n q_k + y_k) / Z (largest gradient term)
    converged = bracket width <= 2 ulp AND stationarity_rel <= STAT_TOL AND dual_infeas_rel <= DUAL_TOL AND
             projection_magnitude <= PROJ_TOL AND margin > 0 AND sum_q_residual <= PROTO_SUM_TOL.
  A solve that is not converged raises DecoderError (never released).

FALLBACKS (registered before any fit): a token with n_t = 0 on the fitting rows (an absent-class fallback token or any
other token without fitting rows) keeps the pinned D0 decoded vector token_proto (smooth(uniform, class) for a qpc
fallback); no labels are invented and no supervised statistic is attached.

SUFFICIENT STATISTICS (canonical accumulation, the same order as qpc/dpc token_tables): per token
  n_t = sum fine.n[f], s_t = sum fine.S[f] (float64, starting from zeros, increasing fine index f), y_t = sum of the
  per-cell label counts Ycell[f] (exact integers). decode_policy uses exactly these and checks them against the rows.

API (role C and the runner code against this):
  solve_token(y, s, n, d, kappa=32.0, eps=1e-12) -> TokenSolution(u, q, obj, cert)
  solve_batch(Y, S, n, d, kappa=32.0, eps=1e-12) -> (U, Q, obj, certs)      certs: dict of per-row arrays
  cache_key(y, s, n, d, kappa, eps) -> bytes;  TokenCache.get_or_solve(...), TokenCache.solve_many(...)
  cell_label_counts(fine, P_fit, d_fit, y_fit) -> Ycell (F, K) int64
  token_stats(fine, cell_token, Ycell) -> (n (T,), S (T, K), Y (T, K))
  token_losses(Y, Q) -> (ll_total (T,), brier_total (T,));  fit_losses(Y, Q, N) -> (L, B)
  decode_policy(pol, tok_fit, P_fit, y_fit, config=None, cache=None) -> DecoderTable
  decoder_pair_dict(cid, pair, dec1, dec2) / save_decoder_pair(path, ...) / load_decoder_pair(path, pair=None)
  release_arrays_d1(pair, dec1, dec2, row_id, P1, d1, P2, d2) -> the release.npz dict (TEAM_PLAN keys)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from qpc import release as RL
from qpc.kmeans import EPS, check_decisions, check_probs, smooth

KAPPA = 32.0
BISECT_ITERS = 64
PROJ_TOL = 1e-9              # max |u_out - u_in| of the frozen projection; above -> REFUSED (solver defect)
STAT_TOL = 1e-9              # relative stationarity residual on the final released vector
DUAL_TOL = 1e-9              # relative dual infeasibility (tie / zero multipliers of the wrong sign)
PROTO_SUM_TOL = RL.PROTO_SUM_TOL     # 1e-12: |sum q - 1| of a released vector
STATS_SUM_TOL = 1e-9         # |sum s - n| <= STATS_SUM_TOL * max(n, 1) (teacher rows sum to 1 within 1e-9)
TEACHER_CLASS_TOL = 1e-9     # pbar[d] >= max_k pbar[k] - TEACHER_CLASS_TOL (d is the teacher class of every row)
STATS_ROW_TOL = 1e-9         # decode_policy: |canonical s_t - row-level s_t| <= STATS_ROW_TOL * max(n_t, 1)
TOL_BATCH_OBJ = 1e-12        # relative |obj_batch - obj_scalar| allowance (np.log only; u and q are bitwise equal)
LOSS_CLIP = 1e-12            # source log-loss clip (dpc.utility.CLIP)
SCHEMA = "lcr-decoder-v1"
TOLERANCES = {"PROJ_TOL": PROJ_TOL, "STAT_TOL": STAT_TOL, "DUAL_TOL": DUAL_TOL, "PROTO_SUM_TOL": PROTO_SUM_TOL,
              "STATS_SUM_TOL": STATS_SUM_TOL, "TEACHER_CLASS_TOL": TEACHER_CLASS_TOL, "STATS_ROW_TOL": STATS_ROW_TOL,
              "TOL_BATCH_OBJ": TOL_BATCH_OBJ, "BISECT_ITERS": BISECT_ITERS, "LOSS_CLIP": LOSS_CLIP,
              "kappa": KAPPA, "eps": EPS}
CERT_FIELDS = ("obj", "obj_full", "t", "nu", "bracket_ulps", "sum_u_residual", "sum_q_residual", "min_u", "margin",
               "stationarity_rel", "dual_infeas_rel", "scale", "projection_magnitude", "n_zero", "n_tie", "converged")


class DecoderError(ValueError):
    """A decoded vector that is not certified (refused; never released)."""


@dataclass(frozen=True)
class TokenSolution:
    u: np.ndarray
    q: np.ndarray
    obj: float
    cert: dict


def _ro(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


# ----------------------------------------------------------------------------------------------- validation
def _check_inputs(Y, S, n, d, kappa, eps):
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64)) + 0.0
    S = np.atleast_2d(np.asarray(S, dtype=np.float64)) + 0.0
    n = np.atleast_1d(np.asarray(n))
    d = np.atleast_1d(np.asarray(d))
    M, K = Y.shape
    if S.shape != (M, K) or n.shape != (M,) or d.shape != (M,):
        raise ValueError(f"shapes: Y {Y.shape}, S {S.shape}, n {n.shape}, d {d.shape} are not aligned")
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
        raise ValueError("n_t = 0 has no supervised statistics: the pinned D0 fallback applies (decode_policy)")
    if np.any((d < 0) | (d >= K)):
        raise ValueError("d out of range")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(S))):
        raise ValueError("non-finite statistics")
    if np.any(Y < 0) or np.any(Y != np.round(Y)):
        raise ValueError("label counts y must be nonnegative integers")
    ysum = Y[:, 0].copy()
    ssum = S[:, 0].copy()
    for k in range(1, K):
        ysum = ysum + Y[:, k]
        ssum = ssum + S[:, k]
    if np.any(ysum != n):
        raise ValueError("label counts must sum to n exactly (no labels are invented or dropped)")
    if np.any(S < 0) or np.any(np.abs(ssum - n) > STATS_SUM_TOL * np.maximum(n, 1)):
        raise ValueError("teacher sums s must be nonnegative and sum to n")
    pbar = S / n[:, None].astype(np.float64)
    if np.any(pbar[np.arange(M), d] < pbar.max(1) - TEACHER_CLASS_TOL):
        raise ValueError("d is not the teacher-predicted class of the token (pbar[d] is not maximal)")
    return Y, S, n, d, pbar, M, K


# ----------------------------------------------------------------------------------------------- core
def _w_of_nu(nu, ak, yk, nf, ck, Z):
    """Closed-form unconstrained response: w solves phi_k'(w) = -nu (positive root, cancellation-free). Elementwise
    on (M, K) arrays (nu, nf broadcast as (M, 1)); callers hold np.errstate(divide/invalid = ignore)."""
    b = nu * Z - yk
    disc = np.sqrt(b * b + 4.0 * nf * ak)
    den = b + disc
    pos = b >= 0
    q_pos = np.where(den > 0, 2.0 * ak / np.where(den > 0, den, 1.0), 0.0)
    q_neg = (disc - b) / (2.0 * nf)
    q = np.where(pos, q_pos, q_neg)
    return Z * q - ck


def _grad(t_or_u, ak, yk, nf, ck, Z):
    q = (t_or_u + ck) / Z
    return (-ak / q + nf * q - yk) / Z


def _state(t, A, Y, nf, C, d, Z, K, M):
    """(u (M, K), nu (M,)) at a given t (u_d = t) by exact pooling + closed-form responses. All class-wise arithmetic
    is elementwise on (M, K) arrays; the only reductions (pooling prefix sums) are explicit k-ordered loops."""
    rows = np.arange(M)
    tc = t[:, None]
    G = _grad(tc, A, Y, nf[:, None], C, Z)
    gd = G[rows, d].copy()
    Gs = G.copy()
    Gs[rows, d] = np.inf
    Gs = np.sort(Gs, axis=1)                      # the K-1 non-d gradients ascending, +inf (class d) last
    pref = gd.copy()
    nu = -pref
    for m in range(1, K):
        pref = pref + Gs[:, m - 1]
        nu = np.maximum(nu, -pref / (1.0 + m))
    W = _w_of_nu(nu[:, None], A, Y, nf[:, None], C, Z)
    U = np.minimum(tc, np.maximum(0.0, W))
    U[rows, d] = t
    return U, nu


def _rowsum(X, K):
    s = X[:, 0].copy()
    for k in range(1, K):
        s = s + X[:, k]
    return s


def project_class_simplex(u, d):
    """Frozen deterministic repair onto the class-dominant simplex (module docstring). u (K,) or (M, K)."""
    U = np.atleast_2d(np.asarray(u, dtype=np.float64))
    M, K = U.shape
    d = np.broadcast_to(np.asarray(d, dtype=np.int64), (M,))
    rows = np.arange(M)
    V = np.maximum(U, 0.0)
    ud = V[rows, d].copy()
    V = np.minimum(V, ud[:, None])
    V[rows, d] = ud
    s = _rowsum(V, K)
    if np.any(~(s > 0)):
        raise DecoderError("projection: base vector has no positive mass")
    V = V / s[:, None]
    mag = np.max(np.abs(V - U), axis=1)
    return (V[0] if np.ndim(u) == 1 else V), (float(mag[0]) if np.ndim(u) == 1 else mag)


def _objective(U, Qm, A, Y, nf, K):
    """f(u) = -sum a_k log q_k + 0.5 n sum q_k^2 - sum y_k q_k (0 log q = 0 where a_k = 0), k-ordered."""
    out = np.zeros(U.shape[0])
    for k in range(K):
        a, q = A[:, k], Qm[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(a > 0, a * np.log(np.where(a > 0, q, 1.0)), 0.0)
        out = out + (-lg + 0.5 * nf * q * q - Y[:, k] * q)
    return out


def solve_batch(Y, S, n, d, kappa=KAPPA, eps=EPS):
    """Vectorised exact solve of M independent token problems. Returns (U (M, K), Q (M, K), obj (M,), certs dict of
    per-row arrays, CERT_FIELDS). Rows are independent (bitwise equal to solve_token). Raises DecoderError if any
    row is not certified (the failing rows are named)."""
    Y, S, n, d, pbar, M, K = _check_inputs(Y, S, n, d, kappa, eps)
    nf = n.astype(np.float64)
    Z = 1.0 + (K + 1) * EPS
    rows = np.arange(M)
    C = np.full((M, K), EPS)
    C[rows, d] = 2.0 * EPS
    A = Y + KAPPA * pbar
    lo = np.full(M, 1.0 / K)
    hi = np.ones(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            Um, _nu = _state(mid, A, Y, nf, C, d, Z, K, M)
            R = _rowsum(Um, K) - 1.0
            neg = R <= 0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
        Ul, nul = _state(lo, A, Y, nf, C, d, Z, K, M)
        Uh, nuh = _state(hi, A, Y, nf, C, d, Z, K, M)
    Rl, Rh = np.abs(_rowsum(Ul, K) - 1.0), np.abs(_rowsum(Uh, K) - 1.0)
    take_hi = Rh < Rl
    t = np.where(take_hi, hi, lo)
    Uraw = np.where(take_hi[:, None], Uh, Ul)
    ulps = (hi - lo) / np.spacing(lo)
    Ufin, proj = project_class_simplex(Uraw, d)
    Ufin = np.atleast_2d(Ufin)
    proj = np.atleast_1d(proj)
    Qm = smooth(Ufin, d, check=False)
    obj, certs = _certify(Ufin, Qm, A, Y, nf, pbar, C, d, Z, K, M)
    certs["t"] = t
    certs["bracket_ulps"] = ulps
    certs["projection_magnitude"] = proj
    conv = (certs["converged_kkt"] & (ulps <= 2.0) & (proj <= PROJ_TOL))
    certs["converged"] = conv
    del certs["converged_kkt"]
    if not np.all(conv):
        bad = np.flatnonzero(~conv)
        i = int(bad[0])
        raise DecoderError(f"{bad.size} token solve(s) not certified; first row {i}: "
                           + json.dumps({f: _py(certs[f][i]) for f in CERT_FIELDS}))
    return Ufin, Qm, obj, certs


def _py(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return float(v)


def _certify(U, Qm, A, Y, nf, pbar, C, d, Z, K, M):
    rows = np.arange(M)
    G = np.empty((M, K))
    scale = np.zeros(M)
    for k in range(K):
        q = Qm[:, k]
        G[:, k] = (-A[:, k] / q + nf * q - Y[:, k]) / Z
        scale = np.maximum(scale, (A[:, k] / q + nf * q + Y[:, k]) / Z)
    ud = U[rows, d]
    notd = np.ones((M, K), dtype=bool)
    notd[rows, d] = False
    zero = notd & (U == 0.0)
    tie = notd & (U == ud[:, None]) & ~zero
    free = notd & ~zero & ~tie
    grp = tie.copy()
    grp[rows, d] = True
    gsum = np.zeros(M)
    gcnt = np.zeros(M)
    for k in range(K):
        gsum = gsum + np.where(grp[:, k], G[:, k], 0.0)
        gcnt = gcnt + grp[:, k]
    nu = -gsum / gcnt
    r = G + nu[:, None]
    stat = np.max(np.where(free, np.abs(r), 0.0), axis=1)
    dual = np.maximum(0.0, np.maximum(np.max(np.where(zero, -r, 0.0), axis=1),
                                      np.max(np.where(tie, r, 0.0), axis=1)))
    sum_u = np.abs(_rowsum(U, K) - 1.0)
    sum_q = np.abs(Qm.sum(1) - 1.0)
    other = np.where(notd, Qm, -np.inf)
    margin = Qm[rows, d] - other.max(1)
    obj = _objective(U, Qm, A, Y, nf, K)
    const = 0.5 * nf
    for k in range(K):
        p = pbar[:, k]
        const = const + KAPPA * np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    st_rel, du_rel = stat / scale, dual / scale
    ok = ((st_rel <= STAT_TOL) & (du_rel <= DUAL_TOL) & (margin > 0) & (sum_q <= PROTO_SUM_TOL)
          & (np.min(U, axis=1) >= 0.0))
    certs = {"obj": obj, "obj_full": obj + const, "nu": nu, "sum_u_residual": sum_u, "sum_q_residual": sum_q,
             "min_u": np.min(U, axis=1), "margin": margin, "stationarity_rel": st_rel, "dual_infeas_rel": du_rel,
             "scale": scale, "n_zero": zero.sum(1), "n_tie": tie.sum(1), "zero_mask": zero, "tie_mask": tie,
             "converged_kkt": ok}
    return obj, certs


def cert_row(certs, i):
    """JSON-safe certificate of row i of a solve_batch certs dict."""
    out = {f: _py(certs[f][i]) for f in CERT_FIELDS}
    out["zero_set"] = [int(k) for k in np.flatnonzero(certs["zero_mask"][i])]
    out["tie_set"] = [int(k) for k in np.flatnonzero(certs["tie_mask"][i])]
    out["method"] = f"exact pooling + closed-form responses + {BISECT_ITERS}-step bisection on t = u_d"
    return out


def solve_token(y, s, n, d, kappa=KAPPA, eps=EPS):
    """One token (y, s float64[K]; n int; d int) -> TokenSolution(u, q, obj, cert). Identical to solve_batch."""
    U, Qm, obj, certs = solve_batch(np.asarray(y, dtype=np.float64)[None], np.asarray(s, dtype=np.float64)[None],
                                    np.array([n]), np.array([d]), kappa, eps)
    return TokenSolution(_ro(U[0]), _ro(Qm[0]), float(obj[0]), cert_row(certs, 0))


# ----------------------------------------------------------------------------------------------- cache
def cache_key(y, s, n, d, kappa=KAPPA, eps=EPS):
    """Exact bytes of (y, s, n, d, K, kappa, eps): a stale reuse (different label counts, teacher sums, count, class,
    K, kappa or eps) cannot hit. -0.0 is mapped to +0.0 first (the solve is identical)."""
    y = np.ascontiguousarray(np.asarray(y, dtype="<f8") + 0.0)
    s = np.ascontiguousarray(np.asarray(s, dtype="<f8") + 0.0)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError("cache_key takes one token's y and s (K,)")
    return b"".join((b"lcr-D1|", y.tobytes(), s.tobytes(), np.int64(n).tobytes(), np.int64(d).tobytes(),
                     np.int64(y.shape[0]).tobytes(), np.float64(kappa).tobytes(), np.float64(eps).tobytes()))


class TokenCache:
    """Solve cache keyed on the EXACT bytes of the sufficient statistics (cache_key). Entries also store the key's
    inputs and are re-verified on every hit, so a stale or colliding reuse is impossible. Values are read-only."""

    def __init__(self, maxsize=2_000_000):
        self.maxsize = int(maxsize)
        self._d = {}
        self.hits = 0
        self.misses = 0
        self.clears = 0

    def __len__(self):
        return len(self._d)

    def _put(self, key, sol):
        if len(self._d) >= self.maxsize:
            self._d.clear()
            self.clears += 1
        self._d[key] = sol

    def _hit(self, key):
        sol = self._d.get(key)
        if sol is not None and sol.cert.get("_key") != key:
            raise AssertionError("cache entry does not match its key")
        return sol

    def get_or_solve(self, y, s, n, d, kappa=KAPPA, eps=EPS):
        key = cache_key(y, s, n, d, kappa, eps)
        sol = self._hit(key)
        if sol is not None:
            self.hits += 1
            return sol
        self.misses += 1
        sol = solve_token(y, s, n, d, kappa, eps)
        sol = TokenSolution(sol.u, sol.q, sol.obj, {**sol.cert, "_key": key})
        self._put(key, sol)
        return sol

    def solve_many(self, Y, S, n, d, kappa=KAPPA, eps=EPS):
        """Batch form: (U, Q, obj) for M tokens; misses are solved together with solve_batch (bitwise equal to the
        scalar path), so results never depend on what was cached before."""
        Y = np.asarray(Y, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        n = np.asarray(n)
        d = np.asarray(d)
        M, K = Y.shape
        keys = [cache_key(Y[i], S[i], n[i], d[i], kappa, eps) for i in range(M)]
        Uo, Qo, ob = np.empty((M, K)), np.empty((M, K)), np.empty(M)
        miss = []
        for i, kk in enumerate(keys):
            sol = self._hit(kk)
            if sol is None:
                miss.append(i)
            else:
                self.hits += 1
                Uo[i], Qo[i], ob[i] = sol.u, sol.q, sol.obj
        if miss:
            mi = np.asarray(miss)
            self.misses += mi.size
            U, Qm, obj, certs = solve_batch(Y[mi], S[mi], n[mi], d[mi], kappa, eps)
            for j, i in enumerate(miss):
                Uo[i], Qo[i], ob[i] = U[j], Qm[j], obj[j]
                self._put(keys[i], TokenSolution(_ro(U[j]), _ro(Qm[j]), float(obj[j]),
                                                 {**cert_row(certs, j), "_key": keys[i]}))
        return Uo, Qo, ob

    def stats(self):
        return {"entries": len(self._d), "hits": self.hits, "misses": self.misses, "clears": self.clears}


# ----------------------------------------------------------------------------------------------- statistics
def _labels(y, K):
    y = np.asarray(y)
    if y.ndim != 1 or not np.all(y == np.round(y)) or np.any(y < 0) or np.any(y >= K):
        raise ValueError(f"labels must be integer classes in [0, {K})")
    return y.astype(np.int64)


def cell_label_counts(fine, P_fit, d_fit, y_fit):
    """Per fine cell true-label counts (F, K) int64 of the fitting rows under the deployed routing."""
    from qpc.kmeans import assign_fine
    cell = assign_fine(P_fit, d_fit, fine)
    y = _labels(y_fit, fine.K)
    if y.shape[0] != cell.shape[0]:
        raise ValueError("labels not aligned with the fitting rows")
    return np.bincount(cell * fine.K + y, minlength=fine.F * fine.K).reshape(fine.F, fine.K).astype(np.int64)


def token_stats(fine, cell_token, Ycell):
    """Canonical token sufficient statistics: n (T,) int64, S (T, K) float64 accumulated from zeros over member cells
    in increasing fine index (= qpc token_tables), Y (T, K) float64 integer-valued label counts."""
    cell_token = np.asarray(cell_token, dtype=np.int64)
    Ycell = np.asarray(Ycell)
    T = int(cell_token.max()) + 1
    K = fine.K
    n = np.zeros(T, dtype=np.int64)
    S = np.zeros((T, K))
    Yt = np.zeros((T, K))
    for f in range(fine.F):
        t = cell_token[f]
        n[t] += fine.n[f]
        S[t] = S[t] + fine.S[f]
        Yt[t] = Yt[t] + Ycell[f]
    return n, S, Yt


# ----------------------------------------------------------------------------------------------- losses
def _brier_by_label(Qm):
    """b[:, k] = sum_j (q_j - [j = k])^2, computed exactly as dpc.utility.per_row does for a row labelled k."""
    T, K = Qm.shape
    out = np.empty((T, K))
    I = np.eye(K)
    for k in range(K):
        out[:, k] = np.sum((Qm - I[k][None]) ** 2, 1)
    return out


def token_losses(Y, Qm):
    """Per-token TOTALS over its fitting rows from sufficient statistics: source log loss sum_k y_k * -log clip(q_k,
    1e-12, 1) and source multiclass Brier sum_k y_k * sum_j (q_j - [j = k])^2 (dpc.utility.per_row conventions)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    Qm = np.atleast_2d(np.asarray(Qm, dtype=np.float64))
    if Y.shape != Qm.shape:
        raise ValueError("Y and Q must be aligned (T, K)")
    nll = -np.log(np.clip(Qm, LOSS_CLIP, 1.0))
    ll = np.zeros(Y.shape[0])
    br = np.zeros(Y.shape[0])
    B = _brier_by_label(Qm)
    for k in range(Y.shape[1]):
        ll = ll + np.where(Y[:, k] > 0, Y[:, k] * nll[:, k], 0.0)
        br = br + Y[:, k] * B[:, k]
    return ll, br


def fit_losses(Y, Qm, N):
    """(L, B): mean source log loss and mean source multiclass Brier over the N fitting rows of one recipient."""
    ll, br = token_losses(Y, Qm)
    if int(np.asarray(Y).sum()) != int(N):
        raise ValueError("label counts do not cover the N fitting rows")
    return float(ll.sum() / N), float(br.sum() / N)


def row_losses(Qrows, y):
    """Row-level reference (dpc.utility.per_row): (mean log loss, mean Brier)."""
    from dpc.utility import per_row
    pr = per_row(np.asarray(Qrows, dtype=np.float64), y, np.asarray(Qrows).shape[1])
    return float(pr["ll"].mean()), float(pr["br"].mean())


# ----------------------------------------------------------------------------------------------- decoder tables
def _sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@dataclass
class DecoderTable:
    """D1 decoder of one recipient for one policy (assignment map). Released vector of token t = q[t]."""
    recipient: int
    K: int
    token_class: np.ndarray
    n: np.ndarray
    y: np.ndarray
    s: np.ndarray
    u: np.ndarray
    q: np.ndarray
    fallback: np.ndarray
    certs: list
    policy_fingerprint: str
    config: str = None
    meta: dict = field(default_factory=dict)

    @property
    def T(self):
        return int(self.token_class.shape[0])

    def stats_hash(self):
        return _sha(np.int64(self.K), np.float64(KAPPA), np.float64(EPS), self.token_class.astype("<i8"),
                    self.n.astype("<i8"), self.y.astype("<f8"), self.s.astype("<f8"))

    def content_hash(self):
        return hashlib.sha256((self.stats_hash() + _sha(self.u.astype("<f8"), self.q.astype("<f8"),
                                                        self.fallback.astype(np.uint8))
                               + self.policy_fingerprint + SCHEMA).encode()).hexdigest()

    def fit_losses(self):
        N = int(self.n.sum())
        return fit_losses(self.y, self.q, N)

    def to_dict(self):
        return {"kind": "lcr.DecoderTable", "schema": SCHEMA, "recipient": int(self.recipient), "K": int(self.K),
                "kappa": KAPPA, "eps": EPS, "token_class": self.token_class.tolist(), "n": self.n.tolist(),
                "y": self.y.tolist(), "s": self.s.tolist(), "u": self.u.tolist(), "q": self.q.tolist(),
                "fallback": [bool(x) for x in self.fallback], "certs": self.certs,
                "policy_fingerprint": self.policy_fingerprint, "config": self.config, "meta": self.meta,
                "tolerances": TOLERANCES, "stats_hash": self.stats_hash(), "content_hash": self.content_hash()}

    @classmethod
    def from_dict(cls, z, verify_solve=True):
        if z.get("kind") != "lcr.DecoderTable" or z.get("schema") != SCHEMA:
            raise ValueError("not an lcr.DecoderTable record")
        if float(z["kappa"]) != KAPPA or float(z["eps"]) != EPS:
            raise ValueError("decoder kappa/eps differ from the fixed protocol values")
        K = int(z["K"])
        tab = cls(recipient=int(z["recipient"]), K=K, token_class=np.asarray(z["token_class"], dtype=np.int64),
                  n=np.asarray(z["n"], dtype=np.int64), y=np.asarray(z["y"], dtype=np.float64).reshape(-1, K),
                  s=np.asarray(z["s"], dtype=np.float64).reshape(-1, K),
                  u=np.asarray(z["u"], dtype=np.float64).reshape(-1, K),
                  q=np.asarray(z["q"], dtype=np.float64).reshape(-1, K),
                  fallback=np.asarray(z["fallback"], dtype=bool), certs=z["certs"],
                  policy_fingerprint=z["policy_fingerprint"], config=z.get("config"), meta=z.get("meta", {}))
        if z.get("stats_hash") != tab.stats_hash() or z.get("content_hash") != tab.content_hash():
            raise ValueError("decoder table hash mismatch after load")
        tab.validate(verify_solve=verify_solve)
        return tab

    def validate(self, pol=None, verify_solve=True):
        """Release invariants of every token; with pol, the binding to the policy; with verify_solve, every supervised
        token is re-solved from its stored statistics and must reproduce u and q bitwise."""
        T, K = self.T, self.K
        for a, shp in ((self.n, (T,)), (self.y, (T, K)), (self.s, (T, K)), (self.u, (T, K)), (self.q, (T, K)),
                       (self.fallback, (T,))):
            if a.shape != shp:
                raise ValueError("decoder table arrays have inconsistent shapes")
        if not np.array_equal(self.fallback, self.n == 0):
            raise ValueError("fallback flags must be exactly the tokens without fitting rows")
        ref = smooth(self.u, self.token_class, check=False)
        if not np.array_equal(ref, self.q):
            raise ValueError("released q differs from the registered smoothing of u")
        rows = np.arange(T)
        other = self.q.copy()
        other[rows, self.token_class] = -np.inf
        if T and not np.all(self.q[rows, self.token_class] > other.max(1)):
            raise ValueError("a released vector's strict argmax is not its token class")
        if T and np.max(np.abs(self.q.sum(1) - 1.0)) > PROTO_SUM_TOL:
            raise ValueError("a released vector is not normalised")
        if pol is not None:
            if pol.fingerprint() != self.policy_fingerprint:
                raise ValueError("decoder table is bound to a different policy")
            if not (np.array_equal(pol.token_class, self.token_class) and np.array_equal(pol.token_n, self.n)):
                raise ValueError("decoder table token classes/counts differ from the policy")
            fb = self.fallback
            if fb.any() and not np.array_equal(self.q[fb], pol.token_proto[fb]):
                raise ValueError("a fallback token does not carry the pinned D0 vector")
            if not np.allclose(self.s, pol.token_S, rtol=0, atol=STATS_ROW_TOL * max(int(self.n.max(initial=1)), 1)):
                raise ValueError("decoder teacher sums differ from the policy's token sums")
        if verify_solve:
            sup = ~self.fallback
            if sup.any():
                U, Qm, _, _ = solve_batch(self.y[sup], self.s[sup], self.n[sup], self.token_class[sup])
                if not (np.array_equal(U, self.u[sup]) and np.array_equal(Qm, self.q[sup])):
                    raise ValueError("stored decoded vectors are not the registered solve of the stored statistics")
        return True


def decode_policy(pol, tok_fit, P_fit, y_fit, config=None, cache=None, meta=None):
    """D1 decoder table of one qpc Policy from its fitting rows. tok_fit must be exactly the deployed tokens of P_fit
    (checked by re-encoding); y_fit the true labels of the same rows. Fallback: tokens without fitting rows keep the
    pinned D0 vector (pol.token_proto). Statistics: canonical token sums (pol.token_n / pol.token_S, i.e. fine-cell
    accumulation order) checked against the row-level sums; label counts from the rows (exact integers)."""
    K, T = pol.K, pol.T
    P_fit = check_probs(P_fit, K)
    d_fit = check_decisions(P_fit, None)
    tok_fit = np.asarray(tok_fit, dtype=np.int64)
    tok_ref, _, _ = RL.encode(pol, P_fit, d_fit)
    if not np.array_equal(tok_ref, tok_fit):
        raise ValueError("tok_fit is not the deployed token assignment of P_fit under this policy")
    y = _labels(y_fit, K)
    if y.shape[0] != tok_fit.shape[0]:
        raise ValueError("labels not aligned with the fitting rows")
    n_rows = np.bincount(tok_fit, minlength=T).astype(np.int64)
    if not np.array_equal(n_rows, pol.token_n):
        raise ValueError("row-level token counts differ from the policy's token counts")
    Yt = np.bincount(tok_fit * K + y, minlength=T * K).reshape(T, K).astype(np.float64)
    S_rows = np.stack([np.bincount(tok_fit, weights=P_fit[:, k], minlength=T) for k in range(K)], 1)
    S = np.asarray(pol.token_S, dtype=np.float64).copy()
    if np.any(np.abs(S - S_rows) > STATS_ROW_TOL * np.maximum(n_rows, 1)[:, None]):
        raise ValueError("canonical token teacher sums differ from the row-level sums")
    tc = np.asarray(pol.token_class, dtype=np.int64)
    fb = n_rows == 0
    U = np.array(pol.token_S, dtype=np.float64) * 0.0
    Qm = np.array(pol.token_proto, dtype=np.float64)
    for t in np.flatnonzero(fb):
        U[t] = np.full(K, 1.0 / K)        # the D0 fallback base vector (qpc: uniform); its q is the pinned token_proto
    certs = [None] * T
    sup = np.flatnonzero(~fb)
    if sup.size:
        if cache is not None:
            Us, Qs, _ = cache.solve_many(Yt[sup], S[sup], n_rows[sup], tc[sup])
            Us2, Qs2, obj, cb = solve_batch(Yt[sup], S[sup], n_rows[sup], tc[sup])
            if not (np.array_equal(Us, Us2) and np.array_equal(Qs, Qs2)):
                raise AssertionError("cached decoder solve differs from a fresh solve")
        else:
            Us2, Qs2, obj, cb = solve_batch(Yt[sup], S[sup], n_rows[sup], tc[sup])
        U[sup], Qm[sup] = Us2, Qs2
        for j, t in enumerate(sup):
            certs[t] = cert_row(cb, j)
    for t in np.flatnonzero(fb):
        certs[t] = {"fallback": "D0_PINNED_NO_FITTING_ROWS", "converged": True}
    tab = DecoderTable(recipient=int(pol.recipient), K=K, token_class=tc.copy(), n=n_rows, y=Yt, s=S, u=U, q=Qm,
                       fallback=fb, certs=certs, policy_fingerprint=pol.fingerprint(), config=config,
                       meta=dict(meta or {}))
    tab.validate(pol, verify_solve=False)
    return tab


def certificate_summary(tab: DecoderTable):
    """Aggregate certificate of one table (for DECODER_CERTIFICATES.json)."""
    sup = [c for c in tab.certs if c and "fallback" not in c]
    mx = lambda f: max((c[f] for c in sup), default=None)  # noqa: E731
    return {"recipient": tab.recipient, "tokens": tab.T, "supervised_tokens": len(sup),
            "fallback_tokens": int(tab.fallback.sum()), "all_converged": all(c["converged"] for c in tab.certs),
            "max_stationarity_rel": mx("stationarity_rel"), "max_dual_infeas_rel": mx("dual_infeas_rel"),
            "max_projection_magnitude": mx("projection_magnitude"), "max_sum_q_residual": mx("sum_q_residual"),
            "min_margin": min((c["margin"] for c in sup), default=None),
            "tokens_with_tie": int(sum(1 for c in sup if c["n_tie"] > 0)),
            "tokens_with_zero": int(sum(1 for c in sup if c["n_zero"] > 0)),
            "max_bracket_ulps": mx("bracket_ulps"), "stats_hash": tab.stats_hash(), "content_hash": tab.content_hash(),
            "tolerances": TOLERANCES}


# ----------------------------------------------------------------------------------------------- decoder.json
def decoder_pair_dict(cid, pair, dec1, dec2):
    """decoder.json content for a D1 configuration: both recipient tables bound to the policy pair (and through it to
    the teacher model and the feature schema)."""
    from lcr import run as R
    p = R.parse_id(cid)
    if p.get("kind") != "policy" or p["decoder"] != "D1":
        raise ValueError(f"{cid!r} is not a D1 configuration")
    for pol, dec in ((pair.p1, dec1), (pair.p2, dec2)):
        dec.validate(pol, verify_solve=False)
    b = RL.binding(pair)
    body = {"kind": "lcr.DecoderPair", "schema": SCHEMA, "config": cid, "policy_pair_fingerprint": pair.fingerprint(),
            "policy_config": pair.config.get("config"), "binding": b, "r1": dec1.to_dict(), "r2": dec2.to_dict()}
    body["decoder_sha256"] = decoder_hash(body)
    return body


def decoder_hash(body):
    """sha256 of the canonical JSON of the decoder pair body (excluding the hash field itself)."""
    z = {k: v for k, v in body.items() if k != "decoder_sha256"}
    return hashlib.sha256(json.dumps(z, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_decoder_pair(path, cid, pair, dec1, dec2):
    body = decoder_pair_dict(cid, pair, dec1, dec2)
    with open(path, "w") as fh:
        json.dump(body, fh, allow_nan=False)
    return body["decoder_sha256"]


def load_decoder_pair(path_or_body, pair=None, verify_solve=True):
    """(cid, dec1, dec2, sha). Refuses a hash mismatch, a different policy pair or a table that fails validation."""
    if isinstance(path_or_body, dict):
        z = path_or_body
    else:
        with open(path_or_body) as fh:
            z = json.load(fh)
    if z.get("kind") != "lcr.DecoderPair" or z.get("schema") != SCHEMA:
        raise ValueError("not an lcr.DecoderPair record")
    if z.get("decoder_sha256") != decoder_hash(z):
        raise ValueError("decoder.json hash mismatch")
    d1 = DecoderTable.from_dict(z["r1"], verify_solve=verify_solve)
    d2 = DecoderTable.from_dict(z["r2"], verify_solve=verify_solve)
    if d1.recipient != 1 or d2.recipient != 2:
        raise ValueError("decoder tables must be (recipient 1, recipient 2)")
    if pair is not None:
        if z["policy_pair_fingerprint"] != pair.fingerprint():
            raise ValueError("decoder.json is bound to a different policy pair")
        if z.get("binding") != RL.binding(pair):
            raise ValueError("decoder.json teacher/schema binding differs from the policy pair")
        d1.validate(pair.p1, verify_solve=False)
        d2.validate(pair.p2, verify_solve=False)
    return z["config"], d1, d2, z["decoder_sha256"]


# ----------------------------------------------------------------------------------------------- release
def encode_d1(pol, dec: DecoderTable, P, d=None):
    """(tokens, D1 released vectors, decisions) with every invariant checked on the rows."""
    if dec.policy_fingerprint != pol.fingerprint():
        raise ValueError("decoder is bound to a different policy")
    tok, _q0, hard = RL.encode(pol, P, d)
    q = dec.q[tok]
    dd = np.asarray(P).argmax(1)
    if not np.array_equal(hard, dd) or not np.array_equal(dec.token_class[tok], hard):
        raise AssertionError("decision preservation violated")
    if q.shape[0]:
        if np.max(np.abs(q.sum(1) - 1.0)) > PROTO_SUM_TOL:
            raise AssertionError("released vector not normalised")
        rows = np.arange(q.shape[0])
        other = q.copy()
        other[rows, hard] = -np.inf
        if not np.all(q[rows, hard] > other.max(1)):
            raise AssertionError("released vector argmax is not strictly the decision")
    return tok, q, hard


def release_arrays_d1(pair, dec1, dec2, row_id, P1, d1, P2, d2):
    """release.npz for a D1 configuration: exactly row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2 (the qpc
    keys; tokens are the policy's canonical token IDs, q the D1 decoded vectors, hard the teacher decisions)."""
    out = {"row_id": np.asarray(row_id)}
    for i, (pol, dec, P, d) in enumerate(((pair.p1, dec1, P1, d1), (pair.p2, dec2, P2, d2)), 1):
        dec.validate(pol, verify_solve=False)
        tok, q, hard = encode_d1(pol, dec, P, d)
        if tok.shape[0] != out["row_id"].shape[0]:
            raise ValueError("row_id is not aligned with the encoded rows")
        out.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": hard, f"alpha{i}": np.int64(pol.T)})
    return out
