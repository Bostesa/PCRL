"""Policy objectives for the rotation-invariant refinement.

Two penalty families share one aggregation and one optimiser:

* ``original`` -- the historical projected linear-moment term only. The objective is
  then exactly ``tr(W' (U - lambda P) W)`` and the closed-form top-eigenvector
  solution is the global optimum of that fixed matrix. Unchanged claim, and the only
  place a global-optimality statement is made.
* ``rotation_invariant`` -- an equal-weight average of the historical linear term and
  the calibrated repaired nonlinear term (Frobenius-weighted quadratic block + exact
  radial-kernel block). Generally nonconvex; the global-optimality statement does NOT
  transfer.

Sign convention, unchanged: everything reports a *training objective* to be
MINIMISED, ``L(W) = penalty(W) - utility(W)``.

Because every term is now rotation invariant (see ``METHOD.md`` section 5.1),
``L(W Q) = L(W)`` for every orthogonal ``Q``: ``L`` is a function on the Grassmannian.
The Stiefel solver is kept unchanged so the comparison with the predecessor stays
interpretable, but its search direction has no rotational component.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The solver, retraction, tangent projection, sign convention, closed-form spectral
# step, policy weights and the perturbed second start are REUSED UNCHANGED from the
# predecessor so that the only difference between the two studies is the penalty.
from experiments.pcrl_nonlinear_rank_v1.objective import (COALITION_ROLES, LOCAL_ROLES, POLICIES,
                                                          canonicalize, perturbed_start,
                                                          qr_retract, spectral_solution,
                                                          stiefel_descent, tangent_project)

from .invariant_moment import (BLOCK_WEIGHT, COMBINE_WEIGHT, DEFAULT_CHUNK, NORMALIZER_FLOOR,
                               FeatureBundle, KernelBlock, QuadraticFeatures, build_features,
                               quadratic_value, quadratic_value_and_dz)

FAMILY_ORIGINAL = 'original'
FAMILY_INVARIANT = 'rotation_invariant'

__all__ = ['FAMILY_ORIGINAL', 'FAMILY_INVARIANT', 'POLICIES', 'LOCAL_ROLES', 'COALITION_ROLES',
           'InvariantRolePenalty', 'PolicyObjective', 'canonicalize', 'qr_retract',
           'tangent_project', 'spectral_solution', 'stiefel_descent', 'perturbed_start',
           'optimise_policy']


# ------------------------------------------------------------------ role penalty
@dataclass
class InvariantRolePenalty:
    """One protected role's linear term plus its calibrated rotation-invariant term.

    The two block normalisers and ``linear_reference`` are measured once at the common
    reference projection and then frozen. Nothing is normalised by a ``W``-dependent
    quantity, which would destroy the objective.

    Degenerate references follow the **nondividing** conventions declared in
    ``METHOD.md`` section 4: a degenerate block is dropped and the survivor reweighted
    to 1.0; if both are degenerate the nonlinear term is zero; if the linear reference
    is degenerate the calibration is zero. Nothing is ever divided by the floor.
    """

    role: object                        # ProtectedRole (frozen, from the predecessor)
    quadratic: QuadraticFeatures
    kernel: KernelBlock
    mask: np.ndarray | None
    quad_normalizer: float
    kernel_normalizer: float
    quad_weight: float
    kernel_weight: float
    linear_reference: float
    calibration: float
    flags: list = field(default_factory=list)
    chunk: int = DEFAULT_CHUNK
    reference_raw: dict = field(default_factory=dict)

    # -------------------------------------------------------------- construction
    @classmethod
    def build(cls, role, quadratic: QuadraticFeatures, kernel: KernelBlock, v: np.ndarray,
              mask, w_reference: np.ndarray, chunk: int = DEFAULT_CHUNK) -> "InvariantRolePenalty":
        flags: list = []
        linear_reference = float(np.trace(w_reference.T @ role.linear_gram @ w_reference))
        bundle = build_features(quadratic, v, w_reference, chunk)
        weights = role.weights()
        quad_raw = quadratic_value(bundle, weights, mask, role.n_valid)
        kernel_raw = kernel.value(bundle.z)

        quad_ok = quad_raw > NORMALIZER_FLOOR
        kernel_ok = kernel_raw > NORMALIZER_FLOOR
        if quad_ok and kernel_ok:
            quad_weight, kernel_weight = BLOCK_WEIGHT, BLOCK_WEIGHT
        elif quad_ok:
            flags.append('kernel_block_reference_degenerate_dropped')
            quad_weight, kernel_weight = 1.0, 0.0
        elif kernel_ok:
            flags.append('quadratic_block_reference_degenerate_dropped')
            quad_weight, kernel_weight = 0.0, 1.0
        else:
            flags.append('nonlinear_block_fully_degenerate')
            quad_weight, kernel_weight = 0.0, 0.0

        if linear_reference > NORMALIZER_FLOOR:
            calibration = linear_reference
        else:
            flags.append('linear_reference_degenerate_nonlinear_term_dropped')
            calibration = 0.0

        obj = cls(role=role, quadratic=quadratic, kernel=kernel, mask=mask,
                  quad_normalizer=quad_raw if quad_ok else 0.0,
                  kernel_normalizer=kernel_raw if kernel_ok else 0.0,
                  quad_weight=quad_weight, kernel_weight=kernel_weight,
                  linear_reference=linear_reference, calibration=calibration,
                  flags=flags, chunk=chunk)
        obj.reference_raw = {
            'quadratic_block_raw': quad_raw, 'kernel_block_raw': kernel_raw,
            'linear_reference': linear_reference,
            'nonlinear_at_reference': obj.nonlinear_value(bundle)[0],
        }
        return obj

    # -------------------------------------------------------------- evaluation
    def nonlinear_value(self, bundle: FeatureBundle):
        quad_raw = (quadratic_value(bundle, self.role.weights(), self.mask, self.role.n_valid)
                    if self.quad_weight else 0.0)
        kernel_raw = self.kernel.value(bundle.z) if self.kernel_weight else 0.0
        quad_norm = (self.quad_weight * quad_raw / self.quad_normalizer) if self.quad_weight else 0.0
        kernel_norm = ((self.kernel_weight * kernel_raw / self.kernel_normalizer)
                       if self.kernel_weight else 0.0)
        return quad_norm + kernel_norm, {
            'quadratic_block_raw': quad_raw, 'kernel_block_raw': kernel_raw,
            'quadratic_block_normalized': quad_norm, 'kernel_block_normalized': kernel_norm}

    def linear_value(self, w: np.ndarray) -> float:
        return float(np.trace(w.T @ self.role.linear_gram @ w))

    def value(self, w: np.ndarray, family: str, bundle: FeatureBundle | None = None) -> float:
        lin = self.linear_value(w)
        if family == FAMILY_ORIGINAL:
            return lin
        nl, _ = self.nonlinear_value(bundle)
        return COMBINE_WEIGHT * lin + COMBINE_WEIGHT * self.calibration * nl

    def value_and_grad(self, w: np.ndarray, family: str, bundle: FeatureBundle | None = None):
        """Returns ``(value, dPenalty/dW from the linear term, dPenalty/dZ, detail)``."""
        lin = self.linear_value(w)
        if family == FAMILY_ORIGINAL:
            return lin, 2.0 * (self.role.linear_gram @ w), None, {'linear': lin}
        lin_grad = COMBINE_WEIGHT * 2.0 * (self.role.linear_gram @ w)
        outer = COMBINE_WEIGHT * self.calibration
        dz = np.zeros_like(bundle.z)
        quad_norm = kernel_norm = 0.0
        quad_raw = kernel_raw = 0.0
        if self.quad_weight and outer:
            scale = outer * self.quad_weight / self.quad_normalizer
            value, dz_quad = quadratic_value_and_dz(self.quadratic, bundle, self.role.weights(),
                                                    self.mask, self.role.n_valid, scale)
            dz += dz_quad
            quad_norm = value / outer if outer else 0.0
            quad_raw = value / scale if scale else 0.0
        if self.kernel_weight and outer:
            scale = outer * self.kernel_weight / self.kernel_normalizer
            value, dz_kernel = self.kernel.value_and_dz(bundle.z, scale)
            dz += dz_kernel
            kernel_norm = value / outer if outer else 0.0
            kernel_raw = value / scale if scale else 0.0
        nl = quad_norm + kernel_norm
        total = COMBINE_WEIGHT * lin + outer * nl
        return total, lin_grad, dz, {
            'linear': lin, 'nonlinear_normalized': nl, 'nonlinear_calibrated': self.calibration * nl,
            'quadratic_block_raw': quad_raw, 'kernel_block_raw': kernel_raw,
            'quadratic_block_normalized': quad_norm, 'kernel_block_normalized': kernel_norm}

    def metadata(self) -> dict:
        return {
            'quadratic_reference_normalizer': self.quad_normalizer,
            'kernel_reference_normalizer': self.kernel_normalizer,
            'quadratic_block_weight': self.quad_weight,
            'kernel_block_weight': self.kernel_weight,
            'linear_reference': self.linear_reference,
            'calibration_constant': self.calibration,
            'flags': list(self.flags),
            'reference_raw': dict(self.reference_raw),
            'kernel': dict(self.kernel.diagnostics),
        }


# ------------------------------------------------------------------ policy objective
@dataclass
class PolicyObjective:
    """``L(W) = penalty(W) - tr(W' U W)`` for one policy and penalty family."""

    utility: np.ndarray                  # trace-normalised U (q, q)
    local: dict                          # role name -> InvariantRolePenalty
    coalition: dict
    policy: str
    family: str
    features: np.ndarray | None = None   # the frozen whitened V
    quadratic: QuadraticFeatures | None = None
    chunk: int = DEFAULT_CHUNK

    def bundle_for(self, w: np.ndarray) -> FeatureBundle | None:
        """One shared ``Z`` and packed-monomial evaluation for every role."""
        if self.family == FAMILY_ORIGINAL:
            return None
        return build_features(self.quadratic, self.features, w, self.chunk)

    def aggregate(self, w: np.ndarray, bundle: FeatureBundle | None = None):
        wl, wc = POLICIES[self.policy]
        if self.family != FAMILY_ORIGINAL and bundle is None:
            bundle = self.bundle_for(w)
        total, grad, parts = 0.0, np.zeros_like(w), {}
        dz_total = None
        for weight, group in ((wl, self.local), (wc, self.coalition)):
            if not weight:
                continue
            acc, acc_grad = 0.0, np.zeros_like(w)
            acc_dz = None
            for name, role in group.items():
                v, g, dz, detail = role.value_and_grad(w, self.family, bundle)
                acc += v / len(group)
                acc_grad += g / len(group)
                if dz is not None:
                    acc_dz = dz / len(group) if acc_dz is None else acc_dz + dz / len(group)
                parts[name] = detail
            total += weight * acc
            grad += weight * acc_grad
            if acc_dz is not None:
                dz_total = weight * acc_dz if dz_total is None else dz_total + weight * acc_dz
            parts['local_aggregate' if group is self.local else 'coalition_aggregate'] = acc
        if dz_total is not None:
            # One V' dZ product finishes the nonlinear part of every role at once.
            grad = grad + self.features.T @ dz_total
        return total, grad, parts

    def loss_and_grad(self, w: np.ndarray):
        penalty, penalty_grad, parts = self.aggregate(w)
        util = float(np.trace(w.T @ self.utility @ w))
        util_grad = 2.0 * (self.utility @ w)
        parts['utility'] = util
        parts['penalty'] = penalty
        return penalty - util, penalty_grad - util_grad, parts

    def penalty_value(self, w: np.ndarray) -> float:
        wl, wc = POLICIES[self.policy]
        bundle = self.bundle_for(w)
        total = 0.0
        for weight, group in ((wl, self.local), (wc, self.coalition)):
            if not weight:
                continue
            total += weight * sum(r.value(w, self.family, bundle) for r in group.values()) / len(group)
        return total

    def loss(self, w: np.ndarray) -> float:
        return self.penalty_value(w) - float(np.trace(w.T @ self.utility @ w))

    def spectral_matrix(self) -> np.ndarray:
        """``U - lambda P`` for the original-moment family (exact historical matrix)."""
        if self.family != FAMILY_ORIGINAL:
            raise ValueError('a fixed matrix exists only for the original-moment family')
        wl, wc = POLICIES[self.policy]
        a = self.utility.copy()
        if wl:
            a -= wl * sum(r.role.linear_gram for r in self.local.values()) / len(self.local)
        if wc:
            a -= wc * sum(r.role.linear_gram for r in self.coalition.values()) / len(self.coalition)
        return a


def optimise_policy(objective: PolicyObjective, w_initial: np.ndarray, seed: int, r: int):
    """Two deterministic starts; keep the lowest training objective, initial points included.

    Identical control flow to the predecessor. Selection consults the training
    objective and nothing else: no attacker, residence, commute, development outcome
    or transport table is reachable from here.
    """
    if objective.family == FAMILY_ORIGINAL:
        a = objective.spectral_matrix()
        w, values = spectral_solution(a, r)
        loss = objective.loss(w) if r else 0.0
        return {'W': w, 'selected_start': 'closed_form_spectral', 'loss': loss,
                'eigenvalues': values.tolist(),
                'top_eigenvalue_sum': float(values[:r].sum()) if r else 0.0,
                'objective_gap': abs(-loss - float(values[:r].sum())) if r else 0.0,
                'starts': {}, 'unique_new_fit': True,
                'feasibility_max_abs': float(np.max(abs(w.T @ w - np.eye(r)))) if r else 0.0}
    starts, candidates = {}, []
    for name, w_start in (('original_moment_spectral', w_initial),
                          ('perturbed_retracted', perturbed_start(w_initial, seed, r))):
        result = stiefel_descent(objective, w_start, label=name)
        starts[name] = {k: v for k, v in result.items() if k != 'W'}
        candidates.append((result['loss'], name, result))
        # The unmoved initial point itself is always an eligible checkpoint.
        candidates.append((result['initial_loss'], name + '__initial',
                           {'W': canonicalize(qr_retract(w_start)), 'loss': result['initial_loss'],
                            'feasibility_max_abs': 0.0}))
    candidates.sort(key=lambda item: (item[0], item[1]))
    loss, name, best = candidates[0]
    return {'W': best['W'], 'selected_start': name, 'loss': loss, 'starts': starts,
            'unique_new_fit': True,
            'restart_loss_spread': float(max(c[0] for c in candidates)
                                         - min(c[0] for c in candidates)),
            'feasibility_max_abs': best.get('feasibility_max_abs', 0.0),
            'returned_initial_point': name.endswith('__initial')}
