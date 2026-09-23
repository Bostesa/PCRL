"""Exact finite-law evaluations and primal/dual information-radius brackets.

Array convention: law[S, H_A, optional H_B, T, Y]. All masses sum to one.
The public Q has one row for every allowed T, including unsupported training T.
"""

from __future__ import annotations

import math
import numpy as np


def _validate(law: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    law = np.asarray(law, dtype=float)
    q = np.asarray(q, dtype=float)
    if law.ndim < 4 or q.ndim != 2 or law.shape[-2] != q.shape[0]:
        raise ValueError("expected law[S,H...,T,Y] and Q[T,Z]")
    if not np.isfinite(law).all() or np.min(law) < 0 or not np.isclose(law.sum(), 1, atol=1e-10):
        raise ValueError("law must be a finite probability table")
    if not np.isfinite(q).all() or np.min(q) < 0 or not np.allclose(q.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Q must contain finite stochastic rows")
    return law, q


def released_joint(law: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return P[S,H...,Z] after summing over Y and T."""
    law, q = _validate(law, q)
    return np.tensordot(law.sum(axis=-1), q, axes=([-1], [0]))


def _mi(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("MI expects a two-dimensional joint table")
    if joint.sum() <= 0:
        return 0.0
    p = joint / joint.sum()
    base = p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :]
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / base[mask])))


def conditional_mi(law: np.ndarray, q: np.ndarray, view_axes: tuple[int, ...]) -> float:
    """I(S;Z|H_view) for axes from the original law."""
    p = released_joint(law, q)
    h_axes = tuple(range(1, p.ndim - 1))
    if any(a not in h_axes for a in view_axes) or len(set(view_axes)) != len(view_axes):
        raise ValueError("view_axes must name distinct H axes")
    keep = (0,) + tuple(view_axes) + (p.ndim - 1,)
    p = p.sum(axis=tuple(a for a in range(p.ndim) if a not in keep))
    if not view_axes:
        return _mi(p)
    # After summation, NumPy retains the original axes in sorted order.
    current = sorted(keep)
    p = np.transpose(p, [current.index(a) for a in keep])
    ans = 0.0
    for state in np.ndindex(*p.shape[1:-1]):
        table = p[(slice(None),) + state + (slice(None),)]
        ans += table.sum() * _mi(table)
    return float(ans)


def chain_rule_terms(law: np.ndarray, q: np.ndarray, bins: np.ndarray) -> dict[str, float]:
    """Return the four terms of the B=b(H_A) identity for one H axis."""
    if law.ndim != 4:
        raise ValueError("chain_rule_terms currently expects one H axis")
    p = released_joint(law, q)
    bins = np.asarray(bins, dtype=int)
    if bins.shape != (p.shape[1],) or np.min(bins) < 0:
        raise ValueError("one nonnegative bin per H state required")
    b_count = int(np.max(bins)) + 1
    pb = np.zeros((p.shape[0], b_count, p.shape[-1]))
    for h, b in enumerate(bins):
        pb[:, b, :] += p[:, h, :]
    coarse = sum(pb[:, b, :].sum() * _mi(pb[:, b, :]) for b in range(b_count))
    positive = 0.0
    negative = 0.0
    for b in range(b_count):
        hs = np.flatnonzero(bins == b)
        bhz = p[:, hs, :].sum(axis=0)
        negative += bhz.sum() * _mi(bhz)
        for s in range(p.shape[0]):
            shz = p[s, hs, :]
            positive += shz.sum() * _mi(shz)
    return {"full": conditional_mi(law, q, (1,)), "coarse": float(coarse),
            "positive": float(positive), "negative": float(negative)}


def _row_kl(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    if np.any((q > 0) & (r[None, :] == 0)):
        raise ValueError("reference assigns zero to an allowed output")
    ratio = np.ones_like(q)
    np.divide(q, r[None, :], out=ratio, where=q > 0)
    return np.sum(np.where(q > 0, q * np.log(ratio), 0), axis=1)


def information_radius_bracket(q: np.ndarray, tol: float = 1e-10,
                               max_iter: int = 100000) -> dict:
    """BA capacity lower bound and reference-row-KL upper bound.

    The printed floating point bracket carries a separate numerical guard.
    A certificate is valid for *all supplied rows*, not only observed rows.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] == 0 or not np.isfinite(q).all() or np.min(q) < 0:
        raise ValueError("invalid channel")
    if not np.allclose(q.sum(axis=1), 1, atol=1e-10):
        raise ValueError("channel rows must sum to one")
    active = np.any(q > 0, axis=0)
    rows = q[:, active]
    p = np.full(rows.shape[0], 1 / rows.shape[0])
    guard = 5e-13
    for iteration in range(max_iter):
        r = p @ rows
        d = _row_kl(rows, r)
        lower = float(p @ d)
        upper = float(np.max(d))
        if upper - lower <= tol:
            break
        weights = p * np.exp(d - np.max(d))
        p = np.maximum(weights, 1e-300)
        p /= p.sum()
    full_r = np.zeros(q.shape[1])
    full_r[active] = r
    return {"lower": max(0.0, lower - guard), "upper": upper + guard,
            "raw_lower": lower, "raw_upper": upper, "gap": upper - lower,
            "iterations": iteration + 1, "converged": upper - lower <= tol,
            "reference": full_r.tolist(), "input_prior": p.tolist(),
            "allowed_rows": int(q.shape[0]), "active_outputs": int(active.sum()),
            "roundoff_guard": guard}
