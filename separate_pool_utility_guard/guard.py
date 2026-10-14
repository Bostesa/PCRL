"""Absolute whole-pool task safeguard: slack-aware projection and finite checks.

Importing performs no numerical work. The guard pool is a fixed fitting pool
(GUARD or UPDATE). Metrics and gradients are exact over the whole pool under a
fixed chunk rule; budgets are anchored to the exact starting metrics and are
never reset to the previous accepted state.
"""
from __future__ import annotations
from itertools import combinations

from .strict import QP_TOL, FINITE_TOL, BACKTRACK, NORM_REPAIR_SAFETY, NORM_REPAIR_ATTEMPTS
from .strict import parameter_vector, _assign, flat_gradient

CHUNK = 1024
BUDGETS = {'nll': 0.010, 'brier': 0.005, 'accuracy': 0.010}


def project_slack(proposal, gradients, slacks, max_norm):
    """Exact Euclidean projection onto {a_j.d<=b_j} intersect {||d||<=r}.

    KKT: d=(p-A^T mu)/(1+nu), mu,nu>=0. Enumerate halfspace active sets S and
    ball state. Ball inactive: mu=G^-1(A_S p-b_S). Ball active: with q the
    minimum-norm point of {A_S d=b_S} and P the orthogonal residual of p,
    ||P p||/(1+nu)=sqrt(r^2-||q||^2). The feasible KKT point with nonnegative
    multipliers is the unique optimum of this strictly convex problem; the
    minimum-objective valid candidate is returned. b>=0 keeps zero feasible.
    """
    import numpy as np
    p = np.asarray(proposal, dtype=np.float64); a = np.asarray(gradients, dtype=np.float64)
    b = np.asarray(slacks, dtype=np.float64); r = float(max_norm)
    if p.ndim != 1 or a.ndim != 2 or a.shape[1] != p.size or b.shape != (len(a),) or len(a) > 2:
        raise ValueError('at most two aligned halfspaces')
    if not (np.isfinite(p).all() and np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(r) and r > 0):
        raise ValueError('finite proposal, gradients, slacks and positive radius required')
    if np.any(b < 0):
        raise ValueError('negative slack must be resolved before projection')
    scale = np.max(np.abs(a), axis=1) if len(a) else np.zeros(0)
    keep = scale > 0
    aa = a[keep] / scale[keep, None]; norms = np.linalg.norm(aa, axis=1)
    aa = aa / norms[:, None]; bb = b[keep] / (scale[keep] * norms)
    tol = QP_TOL
    candidates = []
    for size in range(len(aa) + 1):
        for active in combinations(range(len(aa)), size):
            idx = list(active); at = aa[idx]; bt = bb[idx]
            if size:
                g = at @ at.T; ginv = np.linalg.pinv(g)
                q = at.T @ (ginv @ bt)
                pp = p - at.T @ (ginv @ (at @ p))
            else:
                ginv = np.zeros((0, 0)); q = np.zeros_like(p); pp = p.copy()
            # Ball inactive.
            if size:
                mu = ginv @ (at @ p - bt); d = p - at.T @ mu
            else:
                mu = np.zeros(0); d = p.copy()
            nu = 0.0
            options = [(d, mu, nu, False)]
            # Ball active.
            qq = float(q @ q); ppn = float(np.linalg.norm(pp))
            if r * r - qq > 0 and ppn > 0:
                one_plus = ppn / np.sqrt(r * r - qq)
                if one_plus >= 1 - tol:
                    nu_b = max(0.0, one_plus - 1.0)
                    d_b = pp / one_plus + q
                    mu_b = ginv @ (at @ p - one_plus * bt) if size else np.zeros(0)
                    options.append((d_b, mu_b, nu_b, True))
            for d_c, mu_c, nu_c, ball in options:
                if size and np.any(mu_c < -tol * max(1.0, float(np.abs(mu_c).max()))):
                    continue
                if len(aa) and np.max(aa @ d_c - bb) > tol:
                    continue
                if size and np.max(np.abs(at @ d_c - bt)) > tol * 10:
                    continue
                nd = float(np.linalg.norm(d_c))
                if nd > r * (1 + 1e-12):
                    continue
                candidates.append((float(np.sum((d_c - p) ** 2)), d_c, idx, mu_c, nu_c, ball))
    fallback = not candidates
    if fallback:
        d = np.zeros_like(p); idx = []; mu = np.zeros(0); nu = 0.0; ball = False
    else:
        _, d, idx, mu, nu, ball = min(candidates, key=lambda c: c[0])
    nd = float(np.linalg.norm(d))
    if nd > r:  # floating radius rounding only
        d = d * (r / nd)
    residual = (aa @ d - bb) if len(aa) else np.zeros(0)
    violation = float(max(0.0, float(residual.max()))) if len(aa) else 0.0
    return d, {'zero_feasible': True, 'zero_gradients': int((~keep).sum()), 'guard_count': len(a),
               'active_set': idx, 'multipliers': [float(x) for x in mu], 'ball_multiplier': float(nu),
               'norm_bound_active': bool(ball), 'proposal_norm': float(np.linalg.norm(p)),
               'displacement_norm': float(np.linalg.norm(d)), 'normalized_slacks': bb.tolist(),
               'normalized_constraint_residuals': residual.tolist(), 'max_normalized_violation': violation,
               'valid_kkt_candidates': len(candidates), 'conservative_zero_fallback': fallback,
               'certified': bool(violation <= tol and np.linalg.norm(d) <= r + tol)}


def pool_pass(model, x, y, geometry, parameters=None, *, want=('nll', 'brier', 'ce')):
    """Exact whole-pool occupation metrics and (optionally) gradients.

    Fixed chunks of CHUNK rows in stored row order; float64 sums of row losses
    and float64 gradient sums, divided by pool rows. Uses the canonical wire.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .wires import corrected_trial_wires
    n = len(x)
    if n < 1 or len(y) != n:
        raise ValueError('nonempty aligned pool required')
    sums = {'nll': 0.0, 'brier': 0.0, 'ce': 0.0}; correct = 0
    grads = {k: None for k in want} if parameters is not None else None
    for start in range(0, n, CHUNK):
        xb = x[start:start + CHUNK]; yb = y[start:start + CHUNK]
        ctx = torch.enable_grad() if parameters is not None else torch.no_grad()
        with ctx:
            w = corrected_trial_wires(model, xb, geometry)[1]
            p = w['probabilities']; rows = torch.arange(len(xb))
            losses = {'nll': -p[rows, yb].clamp_min(1e-12).log().sum(),
                      'brier': (p - F.one_hot(yb, 6).to(torch.float64)).square().sum(),
                      'ce': F.cross_entropy(w['centered_logits'], yb, reduction='sum')}
            correct += int((w['decision'] == yb).sum())
            for k in sums:
                sums[k] += float(losses[k].detach())
            if parameters is not None:
                for k in want:
                    g = flat_gradient(losses[k], parameters)
                    grads[k] = g if grads[k] is None else grads[k] + g
    out = {'nll': sums['nll'] / n, 'brier': sums['brier'] / n, 'ce': sums['ce'] / n,
           'correct': correct, 'rows': n, 'accuracy': correct / n}
    if parameters is not None:
        out['gradients'] = {k: v / n for k, v in grads.items()}
    return out


class Anchor:
    """Frozen absolute budgets from exact starting metrics on one fixed pool."""
    def __init__(self, start):
        self.start = {k: start[k] for k in ('nll', 'brier', 'correct', 'rows', 'accuracy')}
        self.nll_limit = start['nll'] + BUDGETS['nll']
        self.brier_limit = start['brier'] + BUDGETS['brier']
        self.allowed_drop = int(BUDGETS['accuracy'] * start['rows'] + 1e-9)
        self.correct_floor = start['correct'] - self.allowed_drop

    def record(self):
        return dict(start=self.start, nll_limit=self.nll_limit, brier_limit=self.brier_limit,
                    allowed_correct_drop=self.allowed_drop, correct_floor=self.correct_floor,
                    budgets=dict(BUDGETS))

    def slacks(self, current):
        return [self.nll_limit - current['nll'], self.brier_limit - current['brier']]

    def within(self, metrics):
        return (metrics['nll'] <= self.nll_limit + FINITE_TOL and metrics['brier'] <= self.brier_limit + FINITE_TOL
                and metrics['correct'] >= self.correct_floor)


def global_step(parameters, proposal, current, anchor, evaluate, max_norm):
    """Apply the slack-projected displacement; accept only inside frozen budgets.

    current: exact full-pool metrics plus NLL/Brier gradients at this state.
    evaluate(): exact full-pool metrics after the real mixed-dtype update.
    Failed fractions restore the exact saved parameters.
    """
    import numpy as np
    before = parameter_vector(parameters)
    raw = anchor.slacks(current)
    resolved = []
    for value in raw:
        if value < -FINITE_TOL:
            return {'status': 'INFEASIBLE_CURRENT_STATE', 'accepted_fraction': None,
                    'actual_displacement_norm': 0.0, 'raw_slacks': raw, 'attempts': [], 'projection': None,
                    'current': {k: current[k] for k in ('nll', 'brier', 'correct')}, 'actual': np.zeros_like(before)}
        resolved.append(max(0.0, value))
    tiny = [bool(-FINITE_TOL <= v < 0) for v in raw]
    a = np.vstack((current['gradients']['nll'], current['gradients']['brier']))
    d, qp = project_slack(proposal, a, np.asarray(resolved), max_norm)
    scale = np.max(np.abs(a), axis=1); keep = scale > 0
    normal = a[keep] / scale[keep, None]; nn = np.linalg.norm(normal, axis=1)
    normal = normal / nn[:, None]; nb = np.asarray(resolved)[keep] / (scale[keep] * nn)
    attempts = []; accepted = None; actual = np.zeros_like(before); metric = None
    for fraction in BACKTRACK:
        norm_trials = []; norm_scale = 1.0; norm_ok = False
        for attempt in range(NORM_REPAIR_ATTEMPTS):
            _assign(parameters, before + fraction * norm_scale * d)
            actual = parameter_vector(parameters) - before
            actual_norm = float(np.linalg.norm(actual))
            norm_ok = bool(np.isfinite(actual_norm) and actual_norm <= max_norm)
            norm_trials.append({'attempt': attempt, 'scale': norm_scale, 'actual_norm': actual_norm, 'bounded': norm_ok})
            if norm_ok or not np.isfinite(actual_norm) or attempt + 1 == NORM_REPAIR_ATTEMPTS:
                break
            norm_scale *= min(1.0, max_norm / actual_norm) * (1.0 - NORM_REPAIR_SAFETY)
        products = normal @ actual if len(normal) else np.zeros(0)
        linear = not len(products) or bool(np.max(products - nb) <= QP_TOL)
        metric = evaluate()
        finite = all(np.isfinite([metric['nll'], metric['brier']]))
        inside = finite and anchor.within(metric)
        passed = bool(finite and norm_ok and linear and inside)
        attempts.append({'fraction': fraction, 'norm_repair_scale': norm_scale,
                         'actual_effective_fraction': fraction * norm_scale, 'actual_norm': float(np.linalg.norm(actual)),
                         'norm_repair_attempts': norm_trials,
                         'actual_normalized_products_minus_slack': (products - nb).tolist(),
                         'linearized_check': linear, 'metrics': {k: metric[k] for k in ('nll', 'brier', 'correct')},
                         'within_absolute_budget': inside, 'accepted': passed})
        if passed:
            accepted = fraction
            break
        _assign(parameters, before)
    if accepted is None:
        _assign(parameters, before); actual = np.zeros_like(before); status = 'REJECTED_ZERO_STEP'; metric = None
    elif not np.any(actual):
        status = 'ROUNDING_ZERO_STEP'
    else:
        status = 'ACCEPTED'
    return {'status': status, 'accepted_fraction': accepted, 'actual_displacement_norm': float(np.linalg.norm(actual)),
            'raw_slacks': raw, 'resolved_slacks': resolved, 'tiny_negative_slack_resolved': tiny,
            'current': {k: current[k] for k in ('nll', 'brier', 'correct')},
            'post_metrics': None if metric is None else {k: metric[k] for k in ('nll', 'brier', 'correct', 'accuracy')},
            'attempts': attempts, 'projection': qp, 'finite_step_tolerance': FINITE_TOL, 'backtrack': list(BACKTRACK),
            'actual': actual}
