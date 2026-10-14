"""Independent forward, metrics, critic and projection code (written from the spec only).

No module of the campaign's numerical code (runner, scoring, selection, projection)
is imported. torch is used only as a tensor loader and for autograd.
"""
import hashlib
import json

import numpy as np
import scipy.optimize
import scipy.special
import torch

PROBE_SALT = 'SPG1|UPDATE-wire-probe|20261013|'
NLL_FLOOR = 1e-12
BUDGET = {'nll': 0.004, 'brier': 0.002, 'kl': 0.001, 'acc_frac': 0.005}
LR = 0.01
MAX_NORM = 0.01
LADDER = (1.0, 0.5, 0.25, 0.125)
TOL = 1e-10


# ----------------------------------------------------------------- loading
def tload(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


class Source:
    """Frozen source encoders/heads from a TCR1 'model' state dict."""

    def __init__(self, sd):
        self.sd = {k: v.detach().clone() for k, v in sd.items()}

    def _lin(self, x, name):
        return torch.nn.functional.linear(x, self.sd[name + '.weight'], self.sd[name + '.bias'])

    def hidden(self, i, X):
        x = torch.as_tensor(X).to(torch.float32)
        with torch.no_grad():
            h = torch.relu(self._lin(x, f'enc.{i}.0'))
            return torch.relu(self._lin(h, f'enc.{i}.2'))

    def final(self, i, h):
        with torch.no_grad():
            return self._lin(h, f'enc.{i}.4')

    def head(self, i):
        return self.sd[f'head.{i}.weight'].to(torch.float64), self.sd[f'head.{i}.bias'].to(torch.float64)


def own_rowspace(W, rel=1e-8):
    """My SVD of the class-centered head; rank by relative cutoff."""
    A = np.asarray(W, dtype=np.float64)
    A = A - A.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    rank = int((s > rel * s[0]).sum())
    return vt[:rank], s, rank, A


def outputs_from_z(z, W, b):
    logits = z @ W.T + b
    cl = logits - logits.mean(1, keepdim=True)
    p = torch.softmax(cl, 1)
    d = torch.argmax(p, 1)  # first maximal index
    return cl, p, d


def occ_wire(hbar, q0, R, b, B, W, bias):
    q = (q0 + hbar @ R.T) + b
    z = q @ B
    cl, p, d = outputs_from_z(z, W, bias)
    return {'representation': z, 'centered_logits': cl, 'probabilities': p, 'decision': d,
            'full': torch.cat((z, cl, p, d.to(torch.float64)[:, None]), 1)}


def raw_proj_wire(z0, B, W, bias):
    z = (z0.to(torch.float64) @ B.T) @ B
    cl, p, d = outputs_from_z(z, W, bias)
    return {'representation': z, 'centered_logits': cl, 'probabilities': p, 'decision': d,
            'full': torch.cat((z, cl, p, d.to(torch.float64)[:, None]), 1)}


def np_wire(w):
    return {k: w[k].detach().numpy().copy() for k in ('representation', 'centered_logits', 'probabilities', 'decision', 'full')}


def wire_diff(a, b):
    out = {k: float(np.max(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64))))
           for k in ('representation', 'centered_logits', 'probabilities')}
    out['decision_changes'] = int((np.asarray(a['decision']) != np.asarray(b['decision'])).sum())
    return out


def probe_positions(row_ids, n=512):
    keys = [hashlib.sha256((PROBE_SALT + str(r)).encode()).hexdigest() for r in row_ids]
    return sorted(range(len(row_ids)), key=lambda i: (keys[i], i))[:n]


# ----------------------------------------------------------------- metrics (numpy)
def np_metrics(cl, y, logp0=None):
    """Mean NLL (floored prob), mean summed Brier, correct, mean KL(p0||p) and CE from centered logits."""
    cl = np.asarray(cl, np.float64); y = np.asarray(y, np.int64); n = len(y)
    logp = scipy.special.log_softmax(cl, axis=1)
    p = scipy.special.softmax(cl, axis=1)
    onehot = np.zeros_like(p); onehot[np.arange(n), y] = 1.0
    out = {'nll': float(np.mean(-np.log(np.maximum(p[np.arange(n), y], NLL_FLOOR)))),
           'brier': float(np.mean(((p - onehot) ** 2).sum(1))),
           'ce': float(np.mean(-logp[np.arange(n), y])),
           'correct': int((np.argmax(p, 1) == y).sum()), 'rows': n}
    out['accuracy'] = out['correct'] / n
    out['kl'] = None if logp0 is None else float(np.mean((np.exp(logp0) * (logp0 - logp)).sum(1)))
    return out


def prob_metrics(p, y):
    p = np.asarray(p, np.float64); y = np.asarray(y, np.int64); n = len(y)
    onehot = np.zeros_like(p); onehot[np.arange(n), y] = 1.0
    return {'accuracy': float(np.mean(np.argmax(p, 1) == y)),
            'nll': float(np.mean(-np.log(np.maximum(p[np.arange(n), y], NLL_FLOOR)))),
            'brier': float(np.mean(((p - onehot) ** 2).sum(1)))}


# ----------------------------------------------------------------- autograd pieces
def flatten(R, b):
    return np.concatenate([R.detach().numpy().reshape(-1), b.detach().numpy().reshape(-1)]).astype(np.float64)


def unflatten(v, rank, width=64):
    v = np.asarray(v, np.float64)
    return torch.from_numpy(v[:rank * width].reshape(rank, width).copy()), torch.from_numpy(v[rank * width:].copy())


def grad_of(fn, R, b):
    R = R.detach().clone().requires_grad_(True); b = b.detach().clone().requires_grad_(True)
    val = fn(R, b)
    g = torch.autograd.grad(val, (R, b))
    return float(val.detach()), np.concatenate([g[0].reshape(-1).numpy(), g[1].reshape(-1).numpy()]).astype(np.float64)


def pool_losses(G, R, b, hbar, q0, y, logp0):
    w = occ_wire(hbar, q0, R, b, G['B'], G['W'], G['bias'])
    cl = w['centered_logits']; p = w['probabilities']; n = len(y); idx = torch.arange(n)
    logp = torch.log_softmax(cl, 1)
    onehot = torch.nn.functional.one_hot(y, cl.shape[1]).to(torch.float64)
    return {'nll': -p[idx, y].clamp_min(NLL_FLOOR).log().mean(),
            'brier': (p - onehot).square().sum(1).mean(),
            'kl': (logp0.exp() * (logp0 - logp)).sum(1).mean(),
            'ce': -logp[idx, y].mean()}


def pool_gradients(G, Rt, bt, hbar, q0, y, logp0, keys=('nll', 'brier', 'kl')):
    return {k: grad_of(lambda R, b, k=k: pool_losses(G, R, b, hbar, q0, y, logp0)[k], Rt, bt)[1] for k in keys}


def critic_apply(state, x):
    layers = sorted({k.rsplit('.', 1)[0] for k in state}, key=lambda s: int(s) if s.isdigit() else -1)
    if layers == ['weight'] or 'weight' in state:
        return x @ state['weight'].to(torch.float64).T + state['bias'].to(torch.float64)
    for j, name in enumerate(layers):
        x = x @ state[name + '.weight'].to(torch.float64).T + state[name + '.bias'].to(torch.float64)
        if j + 1 < len(layers):
            x = torch.relu(x)
    return x


def my_ce(logits, y):
    return -torch.log_softmax(logits, 1)[torch.arange(len(y)), y].mean()


def privacy_risk(G, critic_state, R, b, hbar, q0, inc_full, sex, kind='joint'):
    occ = occ_wire(hbar, q0, R, b, G['B'], G['W'], G['bias'])['full']
    views = [occ, torch.cat((inc_full, occ), 1)]
    H = float(critic_state['entropy'])
    risks, ces = [], []
    for v, (bank, ref) in enumerate(zip(critic_state['critics'], critic_state['references'])):
        x = (views[v] - ref['mean'].to(torch.float64)) / ref['scale'].to(torch.float64)
        losses = [my_ce(critic_apply(c, x), sex) for c in bank]
        m = losses[0]
        for L in losses[1:]:
            m = torch.minimum(m, L)
        m = torch.clamp(m, max=H)
        risks.append(1 - m / H); ces.append([float(L.detach()) for L in losses])
    total = risks[0] + risks[1] if kind == 'joint' else risks[0]
    return total, ces


# ----------------------------------------------------------------- projection (own solver)
def _poly_proj(v, A, s):
    """Exact projection of v onto {A d <= s} (A rows unit norm, s>=0) via NNLS dual + active-set polish."""
    if len(A) == 0:
        return v.copy(), np.zeros(0)
    Gm = A @ A.T
    c = A @ v - s
    # dual: min_{mu>=0} 0.5 mu'G mu - mu'c ; G = L L'  ->  0.5||L' mu - L^{-1} c||^2
    ev, U = np.linalg.eigh(Gm)
    ev = np.maximum(ev, 1e-300)
    Lt = (U * np.sqrt(ev)).T          # L' with L L' = G
    rhs = (U / np.sqrt(ev)).T @ c      # L^{-1} c (in the same factorisation)
    mu, _ = scipy.optimize.nnls(Lt, rhs)
    act = np.where(mu > 0)[0]
    if len(act):
        As = A[act]
        mu_s = np.linalg.solve(As @ As.T, As @ v - s[act])
        if np.all(mu_s >= 0):
            mu = np.zeros(len(A)); mu[act] = mu_s
    return v - A.T @ mu, mu


def project(p, grads, slacks, r=MAX_NORM):
    """argmin ||d-p||^2 s.t. g_k.d <= slack_k (normalized), ||d|| <= r. Bisection on ball multiplier."""
    p = np.asarray(p, np.float64)
    G = np.asarray(grads, np.float64); s = np.asarray(slacks, np.float64)
    nr = np.linalg.norm(G, axis=1); keep = nr > 0
    A = G[keep] / nr[keep, None]; sn = s[keep] / nr[keep]
    d, mu = _poly_proj(p, A, sn); nu = 0.0
    if np.linalg.norm(d) > r:
        f = lambda t: np.linalg.norm(_poly_proj(t * p, A, sn)[0]) - r
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-17:
                break
        t = lo
        d, mu_t = _poly_proj(t * p, A, sn)
        nu = 1.0 / t - 1.0; mu = mu_t / t
    kkt = {'primal_max_violation': float(max(0.0, (A @ d - sn).max())) if len(A) else 0.0,
           'dual_min': float(mu.min()) if len(mu) else 0.0,
           'complementarity_max': float(np.max(np.abs(mu * (A @ d - sn)))) if len(A) else 0.0,
           'stationarity_max': float(np.max(np.abs((1 + nu) * d - p + A.T @ mu))),
           'norm': float(np.linalg.norm(d)), 'ball_multiplier': float(nu),
           'ball_complementarity': float(nu * (r - np.linalg.norm(d))), 'active': [int(i) for i in np.where(mu > 0)[0]]}
    return d, kkt, (A, sn)


def slsqp_project(p, A, sn, r=MAX_NORM):
    cons = [{'type': 'ineq', 'fun': lambda d, a=a, c=c: c - a @ d, 'jac': lambda d, a=a: -a} for a, c in zip(A, sn)]
    cons.append({'type': 'ineq', 'fun': lambda d: r * r - d @ d, 'jac': lambda d: -2 * d})
    x0 = p * min(1.0, 0.5 * r / max(np.linalg.norm(p), 1e-300))
    res = scipy.optimize.minimize(lambda d: 0.5 * np.sum((d - p) ** 2), x0, jac=lambda d: d - p,
                                  constraints=cons, method='SLSQP', options={'ftol': 1e-16, 'maxiter': 500})
    return res.x, bool(res.success)


def ladder(vec0, d, metrics_at, limits, trust, A, sn, r=MAX_NORM):
    """Fractions 1,.5,.25,.125 with norm repair; accept first within budgets and linearized check."""
    attempts = []
    for frac in LADDER:
        scale = 1.0
        for k in range(16):
            vec = vec0 + frac * scale * d
            actual = vec - vec0
            an = float(np.linalg.norm(actual))
            if an <= r or k == 15:
                break
            scale *= min(1.0, r / an) * (1.0 - 1e-6)
        lin = bool(len(A) == 0 or np.max(A @ actual - sn) <= TOL)
        m = metrics_at(vec)
        inside = (m['nll'] <= limits['nll'] + TOL and m['brier'] <= limits['brier'] + TOL and m['correct'] >= limits['correct']
                  and (not trust or m['kl'] <= limits['kl'] + TOL))
        ok = bool(an <= r and lin and inside)
        attempts.append({'fraction': frac, 'actual_norm': an, 'linearized': lin, 'inside': inside, 'metrics': m})
        if ok:
            return actual, frac, attempts, m, vec
    return np.zeros_like(vec0), None, attempts, None, vec0.copy()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
