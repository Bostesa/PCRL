"""Independent verifier helpers (separate from amd.report / amd.evaluate / amd.promote / amd.devrun).

Own implementations: oblivious-tree routing and raw scores from the admitted teacher JSON, release arithmetic of the
family-B uncoupled coarse-code finalist, task loss vectors, unweighted and replicate-weighted AUC (sparse tie aggregation),
household bootstrap weights, Wald bounds and statuses. Only amd.common's hash-checked loaders are used for raw arrays.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

WT = Path('<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1')
if str(WT) not in sys.path:
    sys.path.insert(0, str(WT))
PRIV = WT / '.private' / 'pcrl_autonomous_method_development_v1'
RES = WT / 'results' / 'pcrl_autonomous_method_development_v1'
OUT = PRIV / 'verifier' / 'out'
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD = 0.005
KCLS = {'income': 2, 'occupation': 5}
TASKS = ('income', 'occupation')


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n')


# ------------------------------------------------------------------ teacher routing (own implementation)
class Teacher:
    def __init__(self, payload):
        self.task = payload['task']
        self.feat = np.asarray(payload['split_features'], dtype=np.int64)          # (T, depth)
        self.bord = np.asarray(payload['split_borders'], dtype=np.float32)         # (T, depth)
        self.leaf = np.asarray(payload['leaf_values'], dtype=np.float64)           # (T, L, dim)
        self.scale = float(payload['scale'])
        self.dim = self.leaf.shape[2]
        self.bias = np.asarray(payload['bias'], dtype=np.float64).reshape(self.dim)
        self.names = tuple(payload['feature_names'])
        self.T, self.depth = self.feat.shape
        self.L = self.leaf.shape[1]

    def leaves(self, X, names):
        if tuple(map(str, names)) != self.names:
            raise ValueError('feature schema differs')
        X32 = np.asarray(X, dtype=np.float32)
        idx = np.zeros((len(X32), self.T), dtype=np.int64)
        for t in range(self.T):                       # per tree, bit i set iff float32 x > float32 border
            for i in range(self.depth):
                idx[:, t] += (X32[:, self.feat[t, i]] > self.bord[t, i]).astype(np.int64) * (1 << i)
        return idx

    def raw(self, idx, theta=None):
        n = idx.shape[0]
        acc = np.zeros((n, self.dim), dtype=np.float64)
        accth = np.zeros((n, self.dim), dtype=np.float64)
        th = None if theta is None else np.asarray(theta, dtype=np.float64).reshape(self.T, self.L, self.dim)
        for t in range(self.T):                       # sequential accumulation (differs in order from the source's .sum(1))
            acc += self.leaf[t][idx[:, t]]
            if th is not None:
                accth += th[t][idx[:, t]]
        out = self.scale * acc + self.bias
        if th is not None:
            out = out + accth / self.T
        return out


def probs(task, raw):
    raw = np.asarray(raw, dtype=np.float64)
    if task == 'income':
        d = raw.reshape(-1)
        lg = np.stack([-0.5 * d, 0.5 * d], axis=1)
    else:
        lg = raw - raw.mean(axis=1, keepdims=True)
    lg = lg - lg.max(axis=1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ task loss vectors (own implementation)
def loss_vectors(y, p):
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    n, k = p.shape
    onehot = np.zeros((n, k)); onehot[np.arange(n), y] = 1.0
    acc = (np.argmax(p, axis=1) == y).astype(np.float64)
    nll = -np.log(np.maximum(p[np.arange(n), y], 1e-12))
    brier = ((p - onehot) ** 2).sum(axis=1)
    return {'accuracy': acc, 'nll': nll, 'brier': brier}


def clause_vectors(y, p_f, p_ref):
    f, r = loss_vectors(y, p_f), loss_vectors(y, p_ref)
    return {'accuracy_loss': r['accuracy'] - f['accuracy'], 'nll_excess': f['nll'] - r['nll'], 'brier_excess': f['brier'] - r['brier']}


# ------------------------------------------------------------------ AUC (own implementation)
def auc_point(y, s):
    """Mann-Whitney AUC with ties counted 1/2 (average ranks)."""
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    r = rankdata(s, method='average')
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_weighted(y, s, W):
    """Replicate-weighted AUC: for each replicate b, sum_{i pos, j neg} w_bi w_bj [1(s_i > s_j) + 1/2 1(s_i = s_j)] / (W+ W-).
    Ties are aggregated with a sparse person-by-distinct-score indicator (no reduceat)."""
    import scipy.sparse as sp
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64); W = np.asarray(W, dtype=np.float64)
    u, inv = np.unique(s, return_inverse=True)
    n = len(s)
    M = sp.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, len(u)))
    pos = (W * (y == 1)[None, :]) @ M          # (B, U) weight of positives at each distinct score
    neg = (W * (y == 0)[None, :]) @ M
    pos = np.asarray(pos); neg = np.asarray(neg)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * (below + 0.5 * neg)).sum(axis=1)
    return num / (pos.sum(axis=1) * neg.sum(axis=1))


# ------------------------------------------------------------------ household bootstrap weights (own implementation)
def household_weights(unit, seed, B):
    unit = np.asarray(unit)
    groups = np.unique(unit)
    gi = np.searchsorted(groups, unit)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(B, len(groups)))
    cnt = np.zeros((B, len(groups)), dtype=np.float64)
    rows = np.repeat(np.arange(B), len(groups))
    np.add.at(cnt, (rows, draws.reshape(-1)), 1.0)
    return cnt[:, gi], len(groups)


def tcrit(G, one_sided):
    from scipy import stats
    return float(stats.t.ppf(1 - one_sided, G - 1))


def status(kind, L, U, m):
    if kind == 'benefit':
        return 'ESTABLISHED' if L > m else ('ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED')
    if kind == 'no_excess':
        return 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')
    return 'ABOVE_ZERO' if L > m else ('BELOW_ZERO' if U < m else 'UNRESOLVED')


def category(kind, point, L1, U1, Lf, Uf, m):
    if kind == 'benefit':
        if L1 > m:
            return 'PASS'
        return 'ESTABLISHED_SHORTFALL' if Uf < m else ('POINT_SHORTFALL' if point <= m else 'UNRESOLVED')
    if U1 <= m:
        return 'PASS'
    return 'ESTABLISHED_VIOLATION' if Lf > m else ('POINT_EXCESS' if point > m else 'UNRESOLVED')
