"""Mechanism diagnostics on already-fitted maps. No new interface, no new audit unit.

The central diagnostic answers a question raised by Terminal B's independent
mathematical review (finding A2): a penalty that is nonlinear in ``Z`` is
provably not a trace form ``tr(W' A W)`` for any ``W``-independent ``A``, because
a trace form is invariant under ``W -> W Q`` for orthogonal ``Q`` and a nonlinear
feature map of ``Z = V W`` is not.

That has a consequence the review did not draw out, and it is testable without
any attacker:

* The utility term ``tr(W' U W)`` **is** rotation invariant.
* The original linear penalty ``tr(W' P W)`` **is** rotation invariant.
* The released channel's information content is rotation invariant too: ``Z`` and
  ``Z Q`` are related by an invertible linear map, so any attacker able to use one
  is able to use the other.
* The new nonlinear penalty is **not** rotation invariant.

So the optimiser can lower the new penalty by rotating within a fixed subspace,
at zero utility cost and with **zero change in what is recoverable from the
release**. Whatever share of the training-objective improvement is reachable that
way is, provably, surrogate movement rather than disclosure reduction. This
module measures that share.
"""
from __future__ import annotations

import numpy as np

from .nonlinear_moment import MAX_UPDATES
from .objective import PolicyObjective, canonicalize, qr_retract, stiefel_descent


class RotatedObjective:
    """``L(W_fixed Q)`` as a function of an orthogonal ``Q``: same subspace, new basis."""

    def __init__(self, objective: PolicyObjective, w_fixed: np.ndarray):
        self.objective = objective
        self.w_fixed = w_fixed
        self.utility = objective.utility           # only for shape-compatible callers

    def loss(self, q: np.ndarray) -> float:
        return self.objective.loss(self.w_fixed @ q)

    def loss_and_grad(self, q: np.ndarray):
        loss, grad, parts = self.objective.loss_and_grad(self.w_fixed @ q)
        return loss, self.w_fixed.T @ grad, parts


def subspace_geometry(a: np.ndarray, b: np.ndarray) -> dict:
    """Principal angles between two orthonormal bases, and the projector distance.

    The projector Frobenius distance is the numerically reliable "same subspace"
    measure. ``arccos`` of a singular value near 1 has unbounded derivative, so a
    1e-16 rounding error there becomes ~1e-8 in the reported angle; read the angles
    as descriptive and the projector distance as the test.
    """
    if a.shape[1] == 0:
        return {'rank': 0, 'projector_frobenius_distance': 0.0, 'max_principal_angle_rad': 0.0,
                'mean_principal_angle_rad': 0.0, 'singular_values': []}
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    angles = np.arccos(singular)
    pa, pb = a @ a.T, b @ b.T
    return {'rank': int(a.shape[1]),
            'projector_frobenius_distance': float(np.linalg.norm(pa - pb)),
            'max_principal_angle_rad': float(angles.max()),
            'mean_principal_angle_rad': float(angles.mean()),
            'max_principal_angle_deg': float(np.degrees(angles.max())),
            'singular_values': singular.tolist()}


def rotation_share(objective: PolicyObjective, w_original: np.ndarray, w_refined: np.ndarray,
                   *, max_updates: int = MAX_UPDATES) -> dict:
    """How much of the training-objective gain needs no change in released information.

    ``rotation_only_share`` near 1 means the refinement is moving the surrogate
    without moving the release: decisive evidence of surrogate mismatch that does
    not depend on any fitted attacker.
    """
    r = w_original.shape[1]
    if r == 0:
        return {'rank': 0, 'applicable': False}
    start = objective.loss(w_original)
    refined = objective.loss(w_refined)
    rotated = stiefel_descent(RotatedObjective(objective, w_original), np.eye(r),
                              max_updates=max_updates, label='rotation_only')
    rotation_best = rotated['loss']
    total_gain = start - refined
    rotation_gain = start - rotation_best
    # None, not NaN: the JSON writers use allow_nan=False, and an undefined share
    # must be reported as undefined rather than crash the run.
    share = float(rotation_gain / total_gain) if abs(total_gain) > 1e-15 else None
    q = rotated['W']
    return {
        'rank': int(r), 'applicable': True,
        'loss_at_original_moment_solution': start,
        'loss_at_nonlinear_solution': refined,
        'loss_at_best_rotation_of_original_subspace': rotation_best,
        'total_training_gain': total_gain,
        'rotation_only_training_gain': rotation_gain,
        'rotation_only_share': share,
        'rotation_updates': rotated['updates'],
        'rotation_stop_reason': rotated['stop_reason'],
        'rotation_feasibility_max_abs': float(np.max(abs(q.T @ q - np.eye(r)))),
        'utility_at_original': float(np.trace(w_original.T @ objective.utility @ w_original)),
        'utility_at_best_rotation': float(
            np.trace((w_original @ q).T @ objective.utility @ (w_original @ q))),
        'utility_at_nonlinear': float(np.trace(w_refined.T @ objective.utility @ w_refined)),
        'subspace_original_vs_nonlinear': subspace_geometry(w_original, w_refined),
        'subspace_original_vs_rotation': subspace_geometry(w_original, canonicalize(w_original @ q)),
        'interpretation': (
            'Utility, the original linear penalty and the information content of the release are '
            'all invariant under W -> W Q. Any training-objective gain reachable by rotation alone '
            'is therefore surrogate movement with provably no change in what an attacker can '
            'recover from the released channel.'),
    }


def rotation_invariance_check(objective: PolicyObjective, w: np.ndarray, seed: int = 0) -> dict:
    """Confirms numerically that the two families differ exactly as the algebra predicts."""
    r = w.shape[1]
    rng = np.random.default_rng(20260921 + seed)
    q = qr_retract(rng.normal(size=(r, r)))
    rotated = w @ q
    utility_before = float(np.trace(w.T @ objective.utility @ w))
    utility_after = float(np.trace(rotated.T @ objective.utility @ rotated))
    linear = PolicyObjective(utility=objective.utility, local=objective.local,
                             coalition=objective.coalition, policy=objective.policy,
                             family='original', features=objective.features, chunk=objective.chunk)
    return {
        'utility_change_abs': abs(utility_after - utility_before),
        'original_penalty_change_abs': abs(linear.penalty_value(rotated) - linear.penalty_value(w)),
        'nonlinear_penalty_change_abs': abs(objective.penalty_value(rotated)
                                            - objective.penalty_value(w)),
        'expected': ('utility and original penalty invariant (~0); nonlinear penalty not '
                     'invariant (> 0), so the nonlinear objective is not a trace form'),
    }
