"""Exact Euclidean projection of a privacy direction onto at most two task half-spaces (the proposed update).

    minimize_d 0.5 * ||d - p||^2   subject to   a_j . d <= 0   for each active task guard j (|A| <= 2)

This is a GEM-style projection (Lopez-Paz & Ranzato 2017 solve the same QP with constraints <g, g_k> >= 0); here it is
used to keep a privacy step from increasing active task losses to first order. Nothing about it is new.

Solution: strictly convex objective over a closed convex polyhedron -> unique minimiser. Enumerate active sets
T in {}, {1}, {2}, {1,2}; for each, d_T = p - A_T^T lam_T with A_T d_T = 0; accept the first T with lam_T >= 0
(within tol) and A d_T <= 0 (within tol) -- that is the KKT point. Degenerate cases handled explicitly:
zero gradients (constraint 0 <= 0 is vacuous, dropped), parallel gradients (one constraint), anti-parallel gradients
(feasible set is the hyperplane a.d = 0). Every result carries certified residuals.
"""
from __future__ import annotations

import numpy as np

TOL = 1e-10


def project(p, A, tol=TOL):
    """p: (n,) direction; A: (m, n) active task gradients, m <= 2. Returns (d, info)."""
    p = np.asarray(p, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64).reshape(-1, p.shape[0])
    if A.shape[0] > 2:
        raise ValueError("at most two task guards")
    scale = max(1.0, float(np.linalg.norm(p)))
    norms = np.linalg.norm(A, axis=1)
    keep = norms > tol * scale
    info = {"m_active": int(A.shape[0]), "zero_guards": int((~keep).sum())}
    A = A[keep]
    norms = norms[keep]
    if A.shape[0] == 2:
        cos = float(A[0] @ A[1] / (norms[0] * norms[1]))
        info["cos_a1_a2"] = cos
        if cos >= 1 - 1e-12:              # parallel: identical half-spaces
            A = A[:1]
            info["case"] = "parallel"
        elif cos <= -1 + 1e-12:           # anti-parallel: feasible set is the hyperplane a1.d = 0
            a = A[0]
            d = p - a * (a @ p) / (a @ a)
            info.update(case="antiparallel", active_set=[0, 1])
            return d, _certify(d, p, np.vstack([A[0], A[1]]), info, tol)
    if A.shape[0] == 0:
        info.update(case=info.get("case", "unconstrained"), active_set=[])
        return p.copy(), _certify(p.copy(), p, A, info, tol)
    m = A.shape[0]
    subsets = [()] + [(j,) for j in range(m)] + ([(0, 1)] if m == 2 else [])
    for T in subsets:
        if not T:
            d = p.copy()
            lam = np.zeros(0)
        else:
            AT = A[list(T)]
            G = AT @ AT.T
            lam = np.linalg.solve(G, AT @ p)
            d = p - AT.T @ lam
        viol = A @ d
        if (lam >= -tol * scale).all() and (viol <= tol * scale * norms).all():
            info.update(case=info.get("case", "kkt"), active_set=list(T), lam=lam.tolist())
            return d, _certify(d, p, A, info, tol)
    raise RuntimeError("projection: no KKT active set found (should be impossible)")


def _certify(d, p, A, info, tol):
    viol = A @ d if A.size else np.zeros(0)
    info["max_constraint_violation"] = float(max(0.0, viol.max())) if viol.size else 0.0
    info["moved"] = float(np.linalg.norm(d - p))
    info["certified"] = bool(info["max_constraint_violation"] <= 1e-8 * max(1.0, float(np.linalg.norm(p))) *
                             max([1.0] + [float(np.linalg.norm(a)) for a in A]))
    return info
