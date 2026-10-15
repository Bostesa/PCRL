"""Absolute whole-pool per-recipient task safeguard for the residual adapters.

Importing performs no numerical work. The task fitting pool is UPDATE+GUARD
(10,223 rows). Budgets are separate per recipient, anchored to that
recipient's exact starting (zero-residual) metrics and never reset (not at a
sequential phase boundary either). Every arm enforces the empirical
teacher-function anchor K_i(theta) = mean_x KL(p0_i(x) || p_theta_i(x)) <= 0.001
with immutable p0_i. Income metrics depend only on the income residual and
occupation metrics only on the occupation residual, so the constraints separate.
"""
from __future__ import annotations
from itertools import combinations

QP_TOL = 1e-10
FINITE_TOL = 1e-10
BACKTRACK = (1.0, 0.5, 0.25, 0.125)
NORM_REPAIR_SAFETY = 1e-6
NORM_REPAIR_ATTEMPTS = 16
CHUNK = 1024
BUDGETS = {'nll': 0.004, 'brier': 0.002, 'accuracy_fraction': 0.005, 'kl': 0.001}
NLL_FLOOR = 1e-12


def project_slack(proposal, gradients, slacks, max_norm):
    """Exact Euclidean projection onto {a_j.d<=b_j, j<=3} intersect {||d||<=r}.

    KKT: d=(p-A_S^T mu)/(1+nu), mu,nu>=0. Enumerate every halfspace active set
    S and both ball states. Ball inactive: mu=G^+(A_S p-b_S). Ball active: q is
    the minimum-norm point of {A_S d=b_S}, P p the residual of p orthogonal to
    row(A_S); ||P p||/(1+nu)=sqrt(r^2-||q||^2). Zero-norm rows (e.g. the KL
    gradient at the start) are dropped: with nonnegative slack they contribute
    no first-order direction. b>=0 keeps d=0 feasible. Returns the feasible
    KKT candidate of minimum objective (the unique optimum).
    """
    import numpy as np
    p = np.asarray(proposal, dtype=np.float64); a = np.asarray(gradients, dtype=np.float64)
    b = np.asarray(slacks, dtype=np.float64); r = float(max_norm)
    if p.ndim != 1 or a.ndim != 2 or a.shape[1] != p.size or b.shape != (len(a),) or len(a) > 3:
        raise ValueError('at most three aligned halfspaces')
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
                ginv = np.linalg.pinv(at @ at.T)
                q = at.T @ (ginv @ bt)
                pp = p - at.T @ (ginv @ (at @ p))
                mu = ginv @ (at @ p - bt); d = p - at.T @ mu
            else:
                ginv = np.zeros((0, 0)); q = np.zeros_like(p); pp = p.copy(); mu = np.zeros(0); d = p.copy()
            options = [(d, mu, 0.0, False)]
            qq = float(q @ q); ppn = float(np.linalg.norm(pp))
            if r * r - qq > 0 and ppn > 0:
                one_plus = ppn / np.sqrt(r * r - qq)
                if one_plus >= 1 - tol:
                    d_b = pp / one_plus + q
                    mu_b = ginv @ (at @ p - one_plus * bt) if size else np.zeros(0)
                    options.append((d_b, mu_b, max(0.0, one_plus - 1.0), True))
            for d_c, mu_c, nu_c, ball in options:
                if size and np.any(mu_c < -tol * max(1.0, float(np.abs(mu_c).max()))):
                    continue
                if len(aa) and np.max(aa @ d_c - bb) > tol:
                    continue
                if size and np.max(np.abs(at @ d_c - bt)) > tol * 10:
                    continue
                if float(np.linalg.norm(d_c)) > r * (1 + 1e-12):
                    continue
                candidates.append((float(np.sum((d_c - p) ** 2)), d_c, idx, mu_c, nu_c, ball))
    fallback = not candidates
    if fallback:
        d = np.zeros_like(p); idx = []; mu = np.zeros(0); nu = 0.0; ball = False
    else:
        _, d, idx, mu, nu, ball = min(candidates, key=lambda c: c[0])
    nd = float(np.linalg.norm(d))
    if nd > r:
        d = d * (r / nd)
    residual = (aa @ d - bb) if len(aa) else np.zeros(0)
    violation = float(max(0.0, float(residual.max()))) if len(aa) else 0.0
    return d, {'zero_feasible': True, 'zero_gradient_rows': int((~keep).sum()), 'halfspaces': len(a),
               'active_set': idx, 'multipliers': [float(x) for x in mu], 'ball_multiplier': float(nu),
               'norm_bound_active': bool(ball), 'proposal_norm': float(np.linalg.norm(p)),
               'displacement_norm': float(np.linalg.norm(d)), 'normalized_slacks': bb.tolist(),
               'normalized_constraint_residuals': residual.tolist(), 'max_normalized_violation': violation,
               'valid_kkt_candidates': len(candidates), 'conservative_zero_fallback': fallback,
               'certified': bool(violation <= tol and np.linalg.norm(d) <= r + tol)}


def flat(tensors):
    import numpy as np
    return np.concatenate([t.detach().reshape(-1).numpy().astype(np.float64) for t in tensors])


def pool_pass(adapter, hbar, q0, y, logp0=None, params=None, *, want=('nll', 'brier', 'ce', 'kl')):
    """Exact whole-pool metrics (and gradients w.r.t. R, b_res) in fixed chunks.

    float64 sums of per-row losses in stored row order, divided by pool rows.
    NLL uses the source probability floor 1e-12; CE/KL use stable log-softmax.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    n = len(y)
    if n < 1 or len(hbar) != n or len(q0) != n:
        raise ValueError('nonempty aligned pool required')
    K = adapter.K
    if logp0 is not None and logp0.shape != (n, K):
        raise ValueError('teacher log-probabilities misaligned')
    sums = {'nll': 0.0, 'brier': 0.0, 'ce': 0.0, 'kl': 0.0}; correct = 0
    grads = {k: None for k in want} if params is not None else None
    R, b = adapter.R, adapter.b
    if params is not None:
        R = R.detach().clone().requires_grad_(True); b = b.detach().clone().requires_grad_(True)
    for start in range(0, n, CHUNK):
        sl = slice(start, start + CHUNK)
        ctx = torch.enable_grad() if params is not None else torch.no_grad()
        with ctx:
            w = adapter.wire(hbar[sl], q0[sl], R, b)
            logits = w['centered_logits']; p = w['probabilities']; yb = y[sl]; rows = torch.arange(len(yb))
            logp = F.log_softmax(logits, 1)
            losses = {'nll': -p[rows, yb].clamp_min(NLL_FLOOR).log().sum(),
                      'brier': (p - F.one_hot(yb, K).to(torch.float64)).square().sum(),
                      'ce': -logp[rows, yb].sum()}
            if logp0 is not None:
                lp0 = logp0[sl]
                losses['kl'] = (lp0.exp() * (lp0 - logp)).sum()
            correct += int((w['decision'] == yb).sum())
            for k, v in losses.items():
                sums[k] += float(v.detach())
            if params is not None:
                for k in want:
                    if k not in losses:
                        continue
                    gr = torch.autograd.grad(losses[k], (R, b), retain_graph=True)
                    g = flat(gr)
                    grads[k] = g if grads[k] is None else grads[k] + g
    out = {k: v / n for k, v in sums.items()}
    out.update(correct=correct, rows=n, accuracy=correct / n)
    if logp0 is None:
        out['kl'] = None
    if params is not None:
        out['gradients'] = {k: v / n for k, v in grads.items() if v is not None}
    return out


class Anchor:
    """Frozen absolute budgets from the exact starting metrics on the task pool."""
    def __init__(self, start, trust):
        self.start = {k: start[k] for k in ('nll', 'brier', 'correct', 'rows', 'accuracy')}
        self.trust = bool(trust)
        self.nll_limit = start['nll'] + BUDGETS['nll']
        self.brier_limit = start['brier'] + BUDGETS['brier']
        self.allowed_drop = int(BUDGETS['accuracy_fraction'] * start['rows'] + 1e-9)
        self.correct_floor = start['correct'] - self.allowed_drop
        self.kl_limit = BUDGETS['kl']

    def record(self):
        return dict(start=self.start, nll_limit=self.nll_limit, brier_limit=self.brier_limit,
                    allowed_correct_drop=self.allowed_drop, correct_floor=self.correct_floor,
                    kl_limit=self.kl_limit if self.trust else None, trust=self.trust, budgets=dict(BUDGETS))

    def slacks(self, current):
        s = [self.nll_limit - current['nll'], self.brier_limit - current['brier']]
        if self.trust:
            s.append(self.kl_limit - current['kl'])
        return s

    def within(self, m):
        ok = (m['nll'] <= self.nll_limit + FINITE_TOL and m['brier'] <= self.brier_limit + FINITE_TOL
              and m['correct'] >= self.correct_floor)
        if self.trust:
            ok = ok and m['kl'] <= self.kl_limit + FINITE_TOL
        return bool(ok)


def guarded_step(adapter, proposal, current, anchor, evaluate, max_norm):
    """Slack-projected displacement with exact rollback and the finite ladder.

    current: exact pool metrics plus NLL/Brier(/KL) gradients at this state.
    evaluate(): exact pool metrics after the real float64 update.
    """
    import numpy as np
    import torch
    before = adapter.state(); vec0 = flat([before['R'], before['b']])
    raw = anchor.slacks(current)
    for value in raw:
        if value < -FINITE_TOL:
            return {'status': 'INFEASIBLE_CURRENT_STATE', 'accepted_fraction': None, 'actual_displacement_norm': 0.0,
                    'raw_slacks': raw, 'attempts': [], 'projection': None, 'actual': np.zeros_like(vec0)}
    resolved = [max(0.0, v) for v in raw]
    keys = ['nll', 'brier'] + (['kl'] if anchor.trust else [])
    a = np.vstack([current['gradients'][k] for k in keys])
    d, qp = project_slack(proposal, a, np.asarray(resolved), max_norm)
    scale = np.max(np.abs(a), axis=1); keep = scale > 0
    normal = a[keep] / scale[keep, None]; nn = np.linalg.norm(normal, axis=1)
    normal = normal / nn[:, None]; nb = np.asarray(resolved)[keep] / (scale[keep] * nn)
    nR = adapter.R.numel()
    attempts = []; accepted = None; actual = np.zeros_like(vec0); metric = None

    def assign(vec):
        adapter.R = torch.from_numpy(vec[:nR].reshape(adapter.R.shape).copy())
        adapter.b = torch.from_numpy(vec[nR:].copy())

    for fraction in BACKTRACK:
        trials = []; norm_scale = 1.0; norm_ok = False
        for attempt in range(NORM_REPAIR_ATTEMPTS):
            assign(vec0 + fraction * norm_scale * d)
            actual = flat([adapter.R, adapter.b]) - vec0
            an = float(np.linalg.norm(actual))
            norm_ok = bool(np.isfinite(an) and an <= max_norm)
            trials.append({'attempt': attempt, 'scale': norm_scale, 'actual_norm': an, 'bounded': norm_ok})
            if norm_ok or not np.isfinite(an) or attempt + 1 == NORM_REPAIR_ATTEMPTS:
                break
            norm_scale *= min(1.0, max_norm / an) * (1.0 - NORM_REPAIR_SAFETY)
        products = normal @ actual if len(normal) else np.zeros(0)
        linear = not len(products) or bool(np.max(products - nb) <= QP_TOL)
        metric = evaluate()
        finite = all(np.isfinite([metric['nll'], metric['brier']] + ([metric['kl']] if anchor.trust else [])))
        inside = finite and anchor.within(metric)
        passed = bool(finite and norm_ok and linear and inside)
        attempts.append({'fraction': fraction, 'norm_repair_scale': norm_scale, 'actual_norm': float(np.linalg.norm(actual)),
                         'norm_repair_attempts': trials, 'normalized_products_minus_slack': (products - nb).tolist(),
                         'linearized_check': linear,
                         'metrics': {k: metric[k] for k in ('nll', 'brier', 'correct', 'kl')},
                         'within_absolute_budget': inside, 'accepted': passed})
        if passed:
            accepted = fraction
            break
        assign(vec0)
    if accepted is None:
        assign(vec0); actual = np.zeros_like(vec0); status = 'REJECTED_ZERO_STEP'; metric = None
    elif qp['conservative_zero_fallback']:
        status = 'QP_FALLBACK_ZERO_STEP'
    elif not np.any(d):
        status = 'ZERO_PROJECTION'
    elif not np.any(actual):
        status = 'ROUNDING_ZERO_STEP'
    else:
        status = 'ACCEPTED'
    return {'status': status, 'accepted_fraction': accepted, 'actual_displacement_norm': float(np.linalg.norm(actual)),
            'raw_slacks': raw, 'resolved_slacks': resolved, 'constraint_keys': keys,
            'current': {k: current[k] for k in ('nll', 'brier', 'correct', 'kl')},
            'post_metrics': None if metric is None else {k: metric[k] for k in ('nll', 'brier', 'correct', 'accuracy', 'kl')},
            'attempts': attempts, 'projection': qp, 'finite_step_tolerance': FINITE_TOL, 'backtrack': list(BACKTRACK),
            'actual': actual}
