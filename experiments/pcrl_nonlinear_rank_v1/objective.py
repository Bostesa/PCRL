"""Policy objectives and Stiefel optimisation for the nonlinear moment refinement.

Two penalty families share one aggregation and one optimiser:

* ``original`` — the historical projected linear-moment term only. The objective
  is then exactly ``tr(W' (U - lambda P) W)`` and the closed-form top-eigenvector
  solution is the global optimum of that fixed matrix (unchanged claim).
* ``nonlinear`` — an equal-weight average of the historical linear term and the
  calibrated nonlinear conditional-moment term. Generally nonconvex; the global
  optimality statement does NOT transfer.

Sign convention. Everything below reports a *training objective* to be
MINIMISED:

    L(W) = penalty(W) - utility(W)

so "lowest training objective" is unambiguously "best". The maximised
matrix objective of the historical baseline is ``-L(W)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nonlinear_moment import (BLOCK_WEIGHT, COMBINE_WEIGHT, DEFAULT_CHUNK, INITIAL_STEP,
                               MAX_BACKTRACK, MAX_UPDATES, MIN_GRAD_NORM, MIN_IMPROVEMENT,
                               NORMALIZER_FLOOR, PERTURB_MAGNITUDE, PERTURB_SEED_BASE, SHRINK,
                               ChiFeatures, ProtectedRole, block_squared_norms, build_features,
                               role_moment_gradient, role_moments, _seed)

# Policy definitions: (local aggregate weight, coalition aggregate weight).
POLICIES = {'L1': (1.0, 0.0), 'L2': (2.0, 0.0), 'C1': (1.0, 1.0)}
LOCAL_ROLES = ('A/SEX', 'A/RAC1P', 'A/public_coverage')
COALITION_ROLES = ('AB/SEX', 'AB/RAC1P')


def canonicalize(w: np.ndarray) -> np.ndarray:
    """Historical sign convention: largest-absolute component of each column is nonnegative."""
    w = np.asarray(w, dtype=np.float64).copy()
    if w.size:
        signs = np.sign(w[np.argmax(abs(w), axis=0), np.arange(w.shape[1])])
        w *= np.where(signs == 0, 1, signs)
    return w


def qr_retract(m: np.ndarray) -> np.ndarray:
    """Q factor of a thin QR with positive diagonal R: a well-defined Stiefel retraction."""
    q, r = np.linalg.qr(m)
    d = np.sign(np.diag(r))
    d = np.where(d == 0, 1.0, d)
    return q * d


def tangent_project(w: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Project ``g`` onto the tangent space of the Stiefel manifold at ``w``."""
    wg = w.T @ g
    return g - w @ ((wg + wg.T) / 2.0)


# ------------------------------------------------------------------ role penalty
@dataclass
class RolePenalty:
    """One protected role's combined linear + calibrated nonlinear penalty.

    ``linear_reference`` and the two block normalisers are measured once at the
    common reference projection and then frozen. Nothing here is normalised by a
    W-dependent quantity, which would destroy the objective.
    """

    role: ProtectedRole
    chi: ChiFeatures | None
    mask: np.ndarray | None
    quad_normalizer: float
    rff_normalizer: float
    linear_reference: float
    calibration: float
    flags: list = field(default_factory=list)
    chunk: int = DEFAULT_CHUNK

    @property
    def block_scale(self) -> np.ndarray:
        """Per-feature weight folded into the squared-moment sum."""
        s = np.empty(self.chi.width)
        s[:self.chi.n_quadratic] = BLOCK_WEIGHT / self.quad_normalizer
        s[self.chi.n_quadratic:] = BLOCK_WEIGHT / self.rff_normalizer
        return s

    # ------------------------------------------------------------- construction
    @classmethod
    def build(cls, role: ProtectedRole, chi: ChiFeatures, v: np.ndarray, mask,
              w_reference: np.ndarray, chunk: int = DEFAULT_CHUNK) -> "RolePenalty":
        flags = []
        linear_reference = float(np.trace(w_reference.T @ role.linear_gram @ w_reference))
        bundle = build_features(chi, v, w_reference, chunk)
        m = role_moments(role, bundle, mask)
        quad_raw, rff_raw = block_squared_norms(m, chi)
        quad = quad_raw
        rff = rff_raw
        if quad <= NORMALIZER_FLOOR:
            flags.append('quadratic_block_reference_below_floor')
            quad = NORMALIZER_FLOOR
        if rff <= NORMALIZER_FLOOR:
            flags.append('fourier_block_reference_below_floor')
            rff = NORMALIZER_FLOOR
        # After these normalisers the nonlinear term equals 1.0 at the reference
        # projection (0.5 + 0.5), so calibration to the original role term is a
        # single multiplicative constant.
        calibration = linear_reference
        if linear_reference <= NORMALIZER_FLOOR:
            flags.append('linear_reference_below_floor_calibration_degenerate')
            calibration = NORMALIZER_FLOOR
        obj = cls(role=role, chi=chi, mask=mask, quad_normalizer=quad, rff_normalizer=rff,
                  linear_reference=linear_reference, calibration=calibration, flags=flags, chunk=chunk)
        obj.reference_raw = {'quadratic_block_raw': quad_raw, 'fourier_block_raw': rff_raw,
                             'linear_reference': linear_reference,
                             'nonlinear_at_reference': obj.nonlinear_value(bundle)[0]}
        return obj

    # ------------------------------------------------------------- evaluation
    def nonlinear_value(self, bundle):
        m = role_moments(self.role, bundle, self.mask)
        quad_raw, rff_raw = block_squared_norms(m, self.chi)
        value = BLOCK_WEIGHT * quad_raw / self.quad_normalizer + BLOCK_WEIGHT * rff_raw / self.rff_normalizer
        return value, {'quadratic_block_raw': quad_raw, 'fourier_block_raw': rff_raw,
                       'quadratic_block_normalized': BLOCK_WEIGHT * quad_raw / self.quad_normalizer,
                       'fourier_block_normalized': BLOCK_WEIGHT * rff_raw / self.rff_normalizer}

    def linear_value(self, w: np.ndarray) -> float:
        return float(np.trace(w.T @ self.role.linear_gram @ w))

    def value(self, w: np.ndarray, family: str, bundle=None) -> float:
        """Value only — the line search never needs a gradient."""
        lin = self.linear_value(w)
        if family == 'original':
            return lin
        nl, _ = self.nonlinear_value(bundle)
        return COMBINE_WEIGHT * lin + COMBINE_WEIGHT * self.calibration * nl

    def value_and_grad(self, w: np.ndarray, family: str, bundle=None):
        """Returns (value, linear-part gradient wrt W, dZ gradient or None, detail)."""
        lin = self.linear_value(w)
        lin_grad = COMBINE_WEIGHT * 2.0 * (self.role.linear_gram @ w)
        if family == 'original':
            return lin, 2.0 * (self.role.linear_gram @ w), None, {'linear': lin}
        nl, gz, _ = role_moment_gradient(self.role, self.chi, bundle, self.mask, self.block_scale)
        value = COMBINE_WEIGHT * lin + COMBINE_WEIGHT * self.calibration * nl
        return value, lin_grad, COMBINE_WEIGHT * self.calibration * gz, {
            'linear': lin, 'nonlinear_normalized': nl,
            'nonlinear_calibrated': self.calibration * nl}


# ------------------------------------------------------------------ policy objective
@dataclass
class PolicyObjective:
    """``L(W) = penalty(W) - tr(W' U W)`` for one policy and penalty family."""

    utility: np.ndarray                  # trace-normalised U (q, q)
    local: dict                          # role name -> RolePenalty
    coalition: dict
    policy: str
    family: str
    features: np.ndarray | None = None   # the frozen whitened V, needed by the nonlinear family
    chunk: int = DEFAULT_CHUNK

    def bundle_for(self, w: np.ndarray):
        """One shared ``chi_r(VW)`` evaluation for every role."""
        if self.family == 'original':
            return None
        any_role = next(iter({**self.local, **self.coalition}.values()))
        return build_features(any_role.chi, self.features, w, self.chunk)

    def aggregate(self, w: np.ndarray, bundle=None):
        wl, wc = POLICIES[self.policy]
        if self.family != 'original' and bundle is None:
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
        if self.family != 'original':
            raise ValueError('a fixed matrix exists only for the original-moment family')
        wl, wc = POLICIES[self.policy]
        a = self.utility.copy()
        if wl:
            p = sum(r.role.linear_gram for r in self.local.values()) / len(self.local)
            a -= wl * p
        if wc:
            p = sum(r.role.linear_gram for r in self.coalition.values()) / len(self.coalition)
            a -= wc * p
        return a


def spectral_solution(a: np.ndarray, r: int):
    """Top-``r`` eigenvectors of a symmetric matrix; the global optimum at fixed rank."""
    a = (a + a.T) / 2.0
    values, vectors = np.linalg.eigh(a)
    if r == 0:
        return np.zeros((a.shape[0], 0)), values[::-1]
    return canonicalize(vectors[:, -r:][:, ::-1]), values[::-1]


# ------------------------------------------------------------------ Stiefel optimiser
def stiefel_descent(objective: PolicyObjective, w0: np.ndarray, *, max_updates: int = MAX_UPDATES,
                    initial_step: float = INITIAL_STEP, label: str = ''):
    """Projected-gradient descent on ``St(q, r)`` with QR retraction and backtracking.

    One "update" is one accepted or rejected full-objective step, so the recorded
    ``updates`` is the full-objective budget actually consumed.
    """
    w = qr_retract(np.asarray(w0, dtype=np.float64)) if w0.shape[1] else np.asarray(w0, np.float64)
    loss, grad, _ = objective.loss_and_grad(w)
    history = [{'update': 0, 'loss': loss, 'step': None, 'accepted': True,
                'riemannian_grad_norm': float(np.linalg.norm(tangent_project(w, grad)))}]
    best_w, best_loss = w.copy(), loss
    step = initial_step
    stop = 'budget_exhausted'
    objective_evaluations = 1
    for update in range(1, max_updates + 1):
        rgrad = tangent_project(w, grad)
        gnorm = float(np.linalg.norm(rgrad))
        if gnorm < MIN_GRAD_NORM:
            stop = 'riemannian_gradient_below_tolerance'
            break
        accepted = False
        trial_step = step
        for _ in range(MAX_BACKTRACK):
            candidate = qr_retract(w - trial_step * rgrad)
            trial_loss = objective.loss(candidate)
            objective_evaluations += 1
            if trial_loss < loss - MIN_IMPROVEMENT:
                accepted = True
                break
            trial_step *= SHRINK
        if not accepted:
            stop = 'line_search_failed_no_improvement'
            history.append({'update': update, 'loss': loss, 'step': trial_step,
                            'accepted': False, 'riemannian_grad_norm': gnorm})
            break
        w = candidate
        loss, grad, _ = objective.loss_and_grad(w)
        objective_evaluations += 1
        step = trial_step / SHRINK           # mild re-expansion so the step can grow back
        history.append({'update': update, 'loss': loss, 'step': trial_step, 'accepted': True,
                        'riemannian_grad_norm': float(np.linalg.norm(tangent_project(w, grad)))})
        if loss < best_loss:
            best_w, best_loss = w.copy(), loss
    r = w.shape[1]
    feasibility = float(np.max(abs(best_w.T @ best_w - np.eye(r)))) if r else 0.0
    return {'W': canonicalize(best_w), 'loss': best_loss, 'initial_loss': history[0]['loss'],
            'updates': len(history) - 1, 'objective_evaluations': objective_evaluations,
            'stop_reason': stop, 'feasibility_max_abs': feasibility,
            'final_riemannian_grad_norm': history[-1]['riemannian_grad_norm'],
            'history': history, 'label': label}


def perturbed_start(w0: np.ndarray, seed: int, r: int) -> np.ndarray:
    """Fixed small perturbed/retracted second start; magnitude frozen before fitting."""
    rng = np.random.default_rng(_seed(PERTURB_SEED_BASE, seed, r))
    noise = rng.normal(size=w0.shape)
    norm = np.linalg.norm(noise)
    if norm <= 0:
        return qr_retract(w0)
    return qr_retract(w0 + PERTURB_MAGNITUDE * noise / norm * np.linalg.norm(w0))


def optimise_policy(objective: PolicyObjective, w_initial: np.ndarray, seed: int, r: int):
    """Two deterministic starts; keep the lowest training objective, initial points included."""
    if objective.family == 'original':
        a = objective.spectral_matrix()
        w, values = spectral_solution(a, r)
        loss = objective.loss(w) if r else 0.0
        return {'W': w, 'selected_start': 'closed_form_spectral', 'loss': loss,
                'eigenvalues': values.tolist(), 'top_eigenvalue_sum': float(values[:r].sum()) if r else 0.0,
                'objective_gap': abs(-loss - float(values[:r].sum())) if r else 0.0,
                'starts': {}, 'unique_new_fit': True,
                'feasibility_max_abs': float(np.max(abs(w.T @ w - np.eye(r)))) if r else 0.0}
    starts = {}
    candidates = []
    for name, w_start in (('original_moment_spectral', w_initial),
                          ('perturbed_retracted', perturbed_start(w_initial, seed, r))):
        result = stiefel_descent(objective, w_start, label=name)
        starts[name] = {k: v for k, v in result.items() if k != 'W'}
        candidates.append((result['loss'], name, result))
        # The unmoved initial point itself is always an eligible checkpoint.
        initial_loss = result['initial_loss']
        candidates.append((initial_loss, name + '__initial',
                           {'W': canonicalize(qr_retract(w_start)), 'loss': initial_loss,
                            'feasibility_max_abs': 0.0}))
    candidates.sort(key=lambda item: (item[0], item[1]))
    loss, name, best = candidates[0]
    return {'W': best['W'], 'selected_start': name, 'loss': loss, 'starts': starts,
            'unique_new_fit': True,
            'restart_loss_spread': float(max(c[0] for c in candidates) - min(c[0] for c in candidates)),
            'feasibility_max_abs': best.get('feasibility_max_abs', 0.0),
            'returned_initial_point': name.endswith('__initial')}
