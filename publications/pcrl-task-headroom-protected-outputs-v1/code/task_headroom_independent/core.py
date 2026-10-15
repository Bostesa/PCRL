"""Independent numerical core: router, release, task metrics, AUC, bootstrap, intervals, readers, rules.

Written from the SCIENCE_LOCK text; no code shared with task_headroom or
task_conditioned_representation.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

TASKS = ('income', 'occupation')
NCLASS = {'income': 2, 'occupation': 6}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
NLL_FLOOR = 1e-12


# ----------------------------------------------------------------------------- hashing
def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_digest(a) -> str:
    """dtype + shape + C-order bytes (the campaign's array fingerprint convention)."""
    z = np.ascontiguousarray(a)
    return sha_bytes(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes())


def wire_digest(recipients) -> str:
    """SHA-256 over (probability float64 bytes, decision int64 bytes) per recipient in order."""
    h = hashlib.sha256()
    for p, d in recipients:
        h.update(np.ascontiguousarray(p, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(d, dtype=np.int64).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- oblivious trees
class Teacher:
    """Oblivious-tree teacher from the explicit JSON payload (own router)."""

    def __init__(self, payload: dict):
        self.task = payload['task']
        self.dim = 1 if self.task == 'income' else 6
        self.depth = int(payload['depth'])
        self.feat = np.array(payload['split_features'], dtype=np.int64)
        self.border = np.array(payload['split_borders'], dtype=np.float32)
        self.leaf = np.array(payload['leaf_values'], dtype=np.float64)
        self.scale = float(payload['scale'])
        self.bias = np.array(payload['bias'], dtype=np.float64).reshape(-1)
        self.names = [str(x) for x in payload['feature_names']]
        self.T = self.feat.shape[0]
        self.L = 1 << self.depth
        assert self.feat.shape == (self.T, self.depth) == self.border.shape
        assert self.leaf.shape == (self.T, self.L, self.dim), self.leaf.shape
        assert self.bias.shape == (self.dim,)

    def leaves(self, X) -> np.ndarray:
        """Leaf index per (row, tree): bit i set iff float32 x[feature_i] > float32 border_i."""
        X = np.asarray(X)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        n = X.shape[0]
        idx = np.zeros((n, self.T), dtype=np.int64)
        for i in range(self.depth):
            bit = X[:, self.feat[:, i]] > self.border[:, i][None, :]
            idx += bit.astype(np.int64) * (1 << i)
        return idx

    def teacher_raw(self, leaves: np.ndarray) -> np.ndarray:
        n = leaves.shape[0]
        acc = np.zeros((n, self.dim))
        for t in range(self.T):
            acc += self.leaf[t][leaves[:, t]]
        return self.scale * acc + self.bias[None, :]

    def correction(self, leaves: np.ndarray, theta) -> np.ndarray:
        """sum_t theta[t*L + leaf_t(x)] / T."""
        th = np.asarray(theta, dtype=np.float64).reshape(self.T * self.L, self.dim)
        n = leaves.shape[0]
        acc = np.zeros((n, self.dim))
        flat = leaves + (np.arange(self.T, dtype=np.int64) * self.L)[None, :]
        for t in range(self.T):
            acc += th[flat[:, t]]
        return acc / self.T


def release_probabilities(task: str, raw) -> np.ndarray:
    """Income: scalar log-odds d -> softmax(-d/2, d/2). Occupation: softmax(raw - mean(raw))."""
    raw = np.asarray(raw, dtype=np.float64)
    if task == 'income':
        d = raw.reshape(-1)
        z = np.stack([-0.5 * d, 0.5 * d], axis=1)
    else:
        z = raw - raw.mean(axis=1, keepdims=True)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def release(task, raw):
    p = release_probabilities(task, raw)
    return p, np.argmax(p, axis=1).astype(np.int64)


# ----------------------------------------------------------------------------- task metrics
def per_row(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n, k = p.shape
    assert y.shape == (n,) and np.all(np.abs(p.sum(1) - 1) < 1e-9)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    return {'accuracy': (np.argmax(p, 1) == y).astype(np.float64),
            'nll': -np.log(np.clip(p[np.arange(n), y], NLL_FLOOR, None)),
            'brier': np.sum((p - onehot) ** 2, axis=1)}


def means(y, p):
    return {k: float(v.mean()) for k, v in per_row(y, p).items()}


def clause_values(y, p_service, p_reference):
    """Excess of the service over the reference: accuracy loss, NLL excess, summed-Brier excess."""
    a, r = means(y, p_service), means(y, p_reference)
    return {'accuracy_loss': r['accuracy'] - a['accuracy'], 'nll_excess': a['nll'] - r['nll'],
            'brier_excess': a['brier'] - r['brier']}


def clause_rows(labels, probs, ref_probs, seed, ref_name):
    rows = []
    for i, t in enumerate(TASKS):
        v = clause_values(labels[t], probs[i], ref_probs[i])
        for m, cap in CAPS.items():
            rows.append({'seed': seed, 'task': t, 'reference': ref_name, 'metric': m, 'value': v[m], 'cap': cap,
                         'normalized': v[m] / cap, 'point_pass': bool(v[m] <= cap)})
    return rows


# ----------------------------------------------------------------------------- AUC
def auc_weighted(y, s, W=None) -> np.ndarray:
    """Weighted Mann-Whitney AUC (ties count 1/2). W: (B, n) nonnegative weights or None -> (1,) array.

    For each draw: sum over positive i of w_i * (weighted negatives with score < s_i + 1/2 weighted
    negatives with score == s_i), divided by (sum w_pos)(sum w_neg).
    """
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    if W is None:
        W = np.ones((1, len(y)))
    W = np.asarray(W, dtype=np.float64)
    assert set(np.unique(y)) <= {0, 1} and s.shape == y.shape and W.shape[1] == len(y)
    uniq, inv = np.unique(s, return_inverse=True)
    U = len(uniq)
    pos = y == 1
    out = np.empty(W.shape[0])
    for b0 in range(0, W.shape[0], 256):
        Wb = W[b0:b0 + 256]
        negw = np.zeros((Wb.shape[0], U))
        posw = np.zeros((Wb.shape[0], U))
        # accumulate by unique-score bin
        for j in range(Wb.shape[0]):
            negw[j] = np.bincount(inv[~pos], weights=Wb[j, ~pos], minlength=U)
            posw[j] = np.bincount(inv[pos], weights=Wb[j, pos], minlength=U)
        below = np.cumsum(negw, axis=1) - negw
        num = (posw * (below + 0.5 * negw)).sum(1)
        den = posw.sum(1) * negw.sum(1)
        if np.any(den <= 0):
            raise ValueError('a draw lacks one class')
        out[b0:b0 + 256] = num / den
    return out


def auc(y, s) -> float:
    return float(auc_weighted(y, s)[0])


def auc_bruteforce(y, s, w=None) -> float:
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    P = np.flatnonzero(y == 1); N = np.flatnonzero(y == 0)
    num = 0.0
    for i in P:
        num += w[i] * (w[N] * ((s[i] > s[N]) + 0.5 * (s[i] == s[N]))).sum()
    return float(num / (w[P].sum() * w[N].sum()))


# ----------------------------------------------------------------------------- bootstrap and intervals
def group_multiplicities(unit, draws, seed):
    """Row weights = number of times the row's duplicate group is drawn (groups resampled with replacement)."""
    unit = np.asarray(unit)
    groups, inv = np.unique(unit, return_inverse=True)
    G = len(groups)
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=draws).astype(np.float64)
    return counts[:, inv], G


def weighted_means(v, W):
    v = np.asarray(v, dtype=np.float64)
    return (W @ v) / W.sum(1)


def t_bonferroni(K, groups, alpha=0.05):
    from scipy.stats import t
    return float(t.ppf(1.0 - alpha / (2.0 * K), groups - 1))


def sim_interval(point, draws, crit):
    se = float(np.std(np.asarray(draws), ddof=1))
    return {'point': float(point), 'se': se, 'lower': float(point) - crit * se, 'upper': float(point) + crit * se}


def pct_interval(point, draws, level=0.95):
    lo, hi = np.quantile(np.asarray(draws), [(1 - level) / 2, 1 - (1 - level) / 2])
    return {'point': float(point), 'lower': float(lo), 'upper': float(hi)}


def status_cap(iv, cap):
    if iv['upper'] <= cap:
        return 'PASS'
    if iv['lower'] > cap:
        return 'ESTABLISHED_VIOLATION'
    return 'UNRESOLVED'


def status_margin(iv, margin):
    if iv['lower'] > margin:
        return 'ESTABLISHED'
    if iv['upper'] < margin:
        return 'ESTABLISHED_SHORTFALL'
    return 'UNRESOLVED'


# ----------------------------------------------------------------------------- splits
def proxy_fold(unit) -> np.ndarray:
    return np.array([int(hashlib.sha256(('THP1|FIT2-proxy|' + str(int(u))).encode()).hexdigest(), 16) % 2 for u in unit],
                    dtype=np.int64)


ATT_SALT = 'pcrl-method-portfolio-v1|attack-tune|20261008|'
ATT_VALIDATION_GROUPS = 1015


def att_split(unit):
    """Inherited ATT split: groups ordered by sha256(salt + group id); the first 1015 are validation."""
    unit = np.asarray(unit)
    groups = sorted({int(g) for g in unit}, key=lambda g: hashlib.sha256((ATT_SALT + str(g)).encode()).hexdigest())
    val_groups = set(groups[:ATT_VALIDATION_GROUPS])
    is_val = np.array([int(g) in val_groups for g in unit])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


# ----------------------------------------------------------------------------- readers
class Standardizer:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mu = x.mean(axis=0)
        sd = x.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def __call__(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sd


def make_reader(name, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if 'HGB' in name:
        return HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False,
                                              random_state=seed)
    if 'MLP' in name:
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300,
                             early_stopping=False, random_state=seed)
    return LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)


def fit_predict(name, seed, x_fit, y_fit, *x_eval):
    import warnings
    st = Standardizer().fit(x_fit)
    m = make_reader(name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(st(x_fit), np.asarray(y_fit))
    assert list(m.classes_) == [0, 1]
    return [m.predict_proba(st(x))[:, 1].astype(np.float64) for x in x_eval]


def view_features(p, d, kind):
    p = np.asarray(p, dtype=np.float64); d = np.asarray(d, dtype=np.int64)
    k = p.shape[1]
    oh = np.eye(k)[d]
    if kind == 'out':
        return np.column_stack([p, d.astype(np.float64)])
    if kind == 'log':
        return np.column_stack([np.log(np.maximum(p, 1e-300)), oh])
    if kind == 'dec':
        return oh
    if kind == 'conf':
        return p.max(axis=1, keepdims=True)
    raise ValueError(kind)


SLATE = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}


def reader_input(release_pair, view, kind):
    (p1, d1), (p2, d2) = release_pair
    if view == 'income':
        return view_features(p1, d1, kind)
    if view == 'occupation':
        return view_features(p2, d2, kind)
    return np.column_stack([view_features(p1, d1, kind), view_features(p2, d2, kind)])


# ----------------------------------------------------------------------------- nomination rule (SCIENCE_LOCK text)
PROTECTED_ORDER = ('J_TREE', 'L_TREE', 'L_STRONG', 'S12_TREE', 'S21_TREE')


def candidate_labels(family):
    return [('P2E%d' if family.startswith('S') else 'E%d') % e for e in (2, 5, 10, 20)]


def label_epoch(label):
    return int(label.rsplit('E', 1)[1])


def replay_family(tables, family, strengths=('1.0', '3.0', '10.0')):
    """tables: {seed: {key: row with internal_point_feasible, proxy.pair.value, max_normalized_violation, summed_nll}}."""
    cands = []
    for rho in strengths:
        for lab in candidate_labels(family):
            key = f'{family}|{rho}|{lab}'
            rows = [tables[s][key] for s in sorted(tables)]
            cands.append({'key': key, 'rho': float(rho), 'epoch': label_epoch(lab),
                          'feasible': all(r['internal_point_feasible'] for r in rows),
                          'worst_proxy': max((r['proxy']['pair']['value'] if 'proxy' in r else float('inf')) for r in rows),
                          'worst_violation': max(r['max_normalized_violation'] for r in rows),
                          'summed_nll': sum(r['summed_nll'] for r in rows)})
    feas = [c for c in cands if c['feasible']]
    if feas:
        best = sorted(feas, key=lambda c: (c['worst_proxy'], c['summed_nll'], c['rho'], c['epoch']))[0]
        return 'POINT_FEASIBLE_NOMINEE', best, cands
    best = sorted(cands, key=lambda c: (c['worst_violation'], c['summed_nll'], c['rho'], c['epoch']))[0]
    return 'NO_FEASIBLE_FALLBACK_DIAGNOSTIC', best, cands


def replay_primary(fams):
    feas = [f for f in PROTECTED_ORDER if fams[f][0] == 'POINT_FEASIBLE_NOMINEE']
    if not feas:
        return None
    return sorted(feas, key=lambda f: (fams[f][1]['worst_proxy'], fams[f][1]['summed_nll'], PROTECTED_ORDER.index(f)))[0]


def family_size(n_required_b, primary):
    return 24 + 3 * n_required_b + 12 + (24 if primary != 'J_TREE' else 0)
