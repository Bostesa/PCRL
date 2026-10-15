"""Independent projection solver: argmin ||d-p||^2 s.t. a_j.d <= b_j (j<=3), ||d|| <= r.

Formulation (written independently of the campaign solver): for each candidate active
halfspace set S, the equality-constrained ridge problem min ||d-p||^2 + nu||d||^2,
A_S d = b_S has solution d(nu) = Proj_{A_S d=b_S}(p/(1+nu)) computed by least squares;
nu = 0 if ||d(0)|| <= r, otherwise nu solves ||d(nu)|| = r by bracketed bisection
(||d(nu)|| is monotone). Multipliers mu_S = (1+nu) * lambda_S must be >= 0, all
halfspaces satisfied; the feasible KKT point of least objective is returned (the
problem is strictly convex, so the KKT point is the unique optimum).
"""
from __future__ import annotations
from itertools import combinations


def _affine_proj(y, A, b):
    import numpy as np
    if len(A) == 0:
        return y.copy(), np.zeros(0)
    # lambda = argmin || A^T lambda - (y - d) || with A d = b  ->  (A A^T) lambda = A y - b (least squares, rank-safe)
    lam, *_ = np.linalg.lstsq(A @ A.T, A @ y - b, rcond=None)
    return y - A.T @ lam, lam


def solve(p, A, b, r, tol=1e-10):
    import numpy as np
    p = np.asarray(p, float); A = np.asarray(A, float).reshape(-1, p.size); b = np.asarray(b, float)
    assert np.all(b >= 0) and r > 0
    nrm = np.linalg.norm(A, axis=1)
    keep = nrm > 0
    An = A[keep] / nrm[keep, None]; bn = b[keep] / nrm[keep]
    best = None; ncand = 0
    m = len(An)
    for size in range(m + 1):
        for S in combinations(range(m), size):
            S = list(S); AS = An[S]; bS = bn[S]
            if size:
                # affine set must be consistent
                d_test, _ = _affine_proj(np.zeros_like(p), AS, bS)
                if np.max(np.abs(AS @ d_test - bS)) > 1e-9:
                    continue
            d, lam = _affine_proj(p, AS, bS); nu = 0.0
            if np.linalg.norm(d) > r:
                dmin, _ = _affine_proj(np.zeros_like(p), AS, bS)
                if np.linalg.norm(dmin) > r * (1 + 1e-12):
                    continue
                lo, hi = 0.0, 1.0
                while np.linalg.norm(_affine_proj(p / (1 + hi), AS, bS)[0]) > r:
                    hi *= 2.0
                    if hi > 1e300:
                        break
                for _ in range(400):
                    mid = 0.5 * (lo + hi)
                    if mid <= lo or mid >= hi:
                        break
                    if np.linalg.norm(_affine_proj(p / (1 + mid), AS, bS)[0]) > r:
                        lo = mid
                    else:
                        hi = mid
                nu = hi
                d, lam = _affine_proj(p / (1 + nu), AS, bS)
            mu = (1 + nu) * lam
            if size and np.min(mu) < -tol * max(1.0, float(np.max(np.abs(mu)))):
                continue
            if m and np.max(An @ d - bn) > tol:
                continue
            if np.linalg.norm(d) > r * (1 + 1e-12):
                continue
            ncand += 1
            obj = float(np.sum((d - p) ** 2))
            if best is None or obj < best[0] - 1e-18:
                best = (obj, d, S, mu, nu)
    if best is None:
        return np.zeros_like(p), {'active_set': [], 'ball_active': False, 'fallback_zero': True, 'candidates': 0}
    obj, d, S, mu, nu = best
    nd = np.linalg.norm(d)
    if nd > r:
        d = d * (r / nd)
    return d, {'active_set': [int(np.flatnonzero(keep)[j]) for j in S], 'ball_active': bool(nu > 0), 'ball_multiplier': float(nu),
               'multipliers': [float(x) for x in mu], 'fallback_zero': False, 'candidates': ncand, 'objective': obj,
               'max_violation': float(max(0.0, np.max(An @ d - bn))) if m else 0.0, 'norm': float(np.linalg.norm(d)),
               'normalized_rows': An, 'normalized_slacks': bn}
