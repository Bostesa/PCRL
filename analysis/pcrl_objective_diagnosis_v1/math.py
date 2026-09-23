"""Pure, post hoc calculations on already frozen finite channels."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.special import rel_entr


def cmi(joint: np.ndarray, channel: np.ndarray) -> float:
    p, q = np.asarray(joint, float), np.asarray(channel, float)
    a = np.einsum("sct,tz->scz", p, q)
    p_sc = p.sum(axis=2)
    p_c = p_sc.sum(axis=0)
    b = np.divide(p_sc[:, :, None], p_c[None, :, None],
                  out=np.zeros_like(a), where=p_c[None, :, None] > 0) * a.sum(axis=0)[None]
    return max(0.0, float(rel_entr(a, b).sum()))


def cmi_gradient(joint: np.ndarray, interior_channel: np.ndarray) -> np.ndarray:
    """Gradient of I(S;Z|C) in ambient channel coordinates at positive Q."""
    p, q = np.asarray(joint, float), np.asarray(interior_channel, float)
    if np.any(q <= 0):
        raise ValueError("CMI gradient requires a positive channel")
    a = np.einsum("sct,tz->scz", p, q)
    p_sc = p.sum(axis=2)
    p_c = p_sc.sum(axis=0)
    b = np.divide(p_sc[:, :, None], p_c[None, :, None],
                  out=np.zeros_like(a), where=p_c[None, :, None] > 0) * a.sum(axis=0)[None]
    log_ratio = np.zeros_like(a)
    supported = p_sc > 0
    log_ratio[supported] = np.log(a[supported] / b[supported])
    return np.einsum("sct,scz->tz", p, log_ratio)


def affine_dual_certificate(cost: np.ndarray, laws: dict[str, np.ndarray],
                            budgets: dict[str, float], channel: np.ndarray,
                            mixture: float = 1e-4) -> dict:
    """Valid supporting-hyperplane lower bound; optimizes multipliers, not Q.

    For any nonnegative multipliers, L(Q)=C·Q+sum lambda_j(I_j(Q)-b_j)
    is convex. Its tangent at an interior channel underestimates L everywhere.
    Minimizing that tangent over row simplices gives a lower bound on the
    constrained primal. The small LP maximizes only this certificate.
    """
    cost, q = np.asarray(cost, float), np.asarray(channel, float)
    if cost.shape != q.shape or not 0 < mixture < 1:
        raise ValueError("bad certificate inputs")
    q0 = (1-mixture)*q + mixture/cost.shape[1]
    names = sorted(laws)
    gradients = np.stack([cmi_gradient(laws[k], q0) for k in names])
    g_values = np.array([cmi(laws[k], q0) for k in names])
    budget = np.array([budgets[k] for k in names])
    const = g_values-budget-np.einsum("jtz,tz->j", gradients, q0)
    nt, nz = cost.shape
    # variables: lambda_j >= 0 and unconstrained eta_t <= cost_tz+sum_j lambda_j grad_jtz
    objective = np.r_[-const, -np.ones(nt)]
    a = np.zeros((nt*nz, len(names)+nt))
    a[:, :len(names)] = -gradients.transpose(1, 2, 0).reshape(nt*nz, len(names))
    for t in range(nt):
        a[t*nz:(t+1)*nz, len(names)+t] = 1.0
    result = linprog(objective, A_ub=a, b_ub=cost.ravel(),
                     bounds=[(0, None)]*len(names)+[(None, None)]*nt,
                     method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                              "dual_feasibility_tolerance": 1e-9})
    if not result.success:
        return {"status": "certificate_lp_failed", "message": result.message}
    multipliers = np.maximum(result.x[:len(names)], 0)
    g = cost + np.einsum("j,jtz->tz", multipliers, gradients)
    # Re-evaluate at returned multipliers, bypassing eta's solver precision.
    lower = float(g.min(axis=1).sum() + multipliers @ const)
    primal = float(np.sum(cost*q))
    return {"status": "dual_affine_bound", "interior_mixture": mixture,
            "multipliers": dict(zip(names, map(float, multipliers))),
            "lower_bound": lower, "primal_objective": primal,
            "absolute_gap": primal-lower, "relative_gap": (primal-lower)/max(1, abs(primal)),
            "lp_primal_residual": float(np.max(a@result.x-cost.ravel())),
            "unconstrained_lower_bound": float(cost.min(axis=1).sum())}


def information_radius_bound(channel: np.ndarray, state_mass: np.ndarray) -> float:
    """E_T KL(Q_T || r), r=E_T Q_T; bounds I(S;Z|H) for S,H -> T -> Z."""
    q = np.asarray(channel, float)
    p = np.asarray(state_mass, float)
    p = p/p.sum()
    r = p@q
    return float((p[:, None]*rel_entr(q, r[None, :])).sum())
