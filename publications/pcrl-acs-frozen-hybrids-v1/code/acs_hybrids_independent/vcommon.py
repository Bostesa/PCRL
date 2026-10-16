"""Shared helpers written for the independent verification (own AUC, metrics, household bootstrap, statuses).

Nothing here imports acs_hybrids.report / readers / lock, acs_replication.report or task_conditioned_representation.inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1'
OUT = RES / 'independent'
SRC_RES = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = ROOT / '.private' / 'acs_frozen_hybrids'
SRCP = PRIV / 'source'

SEEDS = (0, 1)
TASKS = ('income', 'occupation')
NCLASS = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
REFS = ('U0', 'Ucal')
ROSTER = ('H_LL', 'H_LJ', 'H_JL', 'H_JJ', 'NEURAL_RAW', 'TASK_REFIT', 'T_TREE')
HYBRIDS = ('H_LL', 'H_LJ', 'H_JL', 'H_JJ')
VIEWS = ('income', 'occupation', 'pair')
ROLES = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'MODEL_SELECT')
SLATE = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB')
MODEL_SEED_BASE = 2026101205        # registered reader rng rule: MODEL_SEED_BASE + model seed
NULL_SEED = 2026101602              # registered shuffled-null permutation seed base
BOOT_B = 2048
BOOT_SEED = 2026101603
PUB_COMMIT = 'f4ea19982bce3dc3c88c5951ad34c4cf84620a20'
PUB_PATH = 'publications/pcrl-acs-protected-tree-replication-v1/results'


def jload(p):
    return json.loads(Path(p).read_text())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def analysis_lock():
    return jload(RES / 'ANALYSIS_LOCK.json')


def reporting_lock():
    return jload(RES / 'REPORTING_LOCK.json')


def components():
    """Per-service (income source label, occupation source label), read from the ANALYSIS_LOCK bank section."""
    bank = analysis_lock()['bank']
    out = {h: (v['income'], v['occupation']) for h, v in bank['combinations'].items()}
    out.update({b: (s, s) for b, s in bank['controls'].items()})
    out.update({r: (r, r) for r in bank['references_for_utility_only']})
    return out


# ------------------------------------------------------------------ reader inputs (definitions must match the slate)
def reader_kind(name):
    if name.startswith('OUT_'):
        return 'out'
    if name.startswith('LOG_'):
        return 'log'
    if name == 'DEC_LR':
        return 'dec'
    if name == 'CONF_HGB':
        return 'conf'
    raise ValueError(f'unknown reader {name}')


def wire_features(p, k, kind):
    """One recipient's released wire turned into a reader input (probabilities, decision index, one-hot, max prob)."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != k:
        raise ValueError('probability shape differs')
    dec = np.argmax(p, axis=1)
    onehot = (dec[:, None] == np.arange(k)[None, :]).astype(np.float64)
    if kind == 'out':
        return np.column_stack((p, dec.astype(np.float64)))
    if kind == 'log':
        return np.column_stack((np.log(np.maximum(p, 1e-300)), onehot))
    if kind == 'dec':
        return onehot
    if kind == 'conf':
        return np.max(p, axis=1).reshape(-1, 1)
    raise ValueError(kind)


def view_features(bank_svc_role, view, kind):
    """Reader input for one service, role and view; the pair view is [income wire | occupation wire]."""
    if view in TASKS:
        return wire_features(bank_svc_role[view], NCLASS[view], kind)
    a = wire_features(bank_svc_role['income'], 2, kind)
    b = wire_features(bank_svc_role['occupation'], 5, kind)
    return np.column_stack((a, b))


def saved_predict(record, x):
    """Positive-class probability of a saved reader record ({'model', 'scaler'}) computed here from its parts."""
    model, sc = record['model'], record['scaler']
    if list(model.classes_) != [0, 1]:
        raise ValueError('reader class order differs')
    z = (np.asarray(x, dtype=np.float64) - sc.mean_) / sc.scale_
    return model.predict_proba(z)[:, 1].astype(np.float64)


# ------------------------------------------------------------------ AUC (own implementation)
def auc(y, s):
    """Mann-Whitney AUC from midranks (ties count one half). Exact rational result rounded once."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError('AUC needs aligned 1-d arrays')
    if not np.isfinite(s).all():
        raise ValueError('non-finite score')
    pos = y == 1
    neg = y == 0
    if not np.all(pos | neg):
        raise ValueError('labels must be 0/1')
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        raise ValueError('both classes needed')
    r = stats.rankdata(s, method='average')
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_weighted_draws(y, s, W):
    """AUC for every row of a nonnegative person-weight matrix W (B x n), via per-person cumulative negative weight.

    For each positive person: weight x (negative weight strictly below its score + half the negative weight tied with it)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError('non-finite score')
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    yy = y[order] == 1
    Ws = W[:, order]
    negw = Ws * (~yy)[None, :]
    cneg = np.zeros((W.shape[0], len(ss) + 1))
    np.cumsum(negw, axis=1, out=cneg[:, 1:])
    new = np.r_[True, ss[1:] != ss[:-1]]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(ss)]                 # exclusive
    pidx = np.flatnonzero(yy)
    st = starts[gid[pidx]]
    en = ends[gid[pidx]]
    below = cneg[:, st]
    upto = cneg[:, en]
    num = (Ws[:, pidx] * (below + 0.5 * (upto - below))).sum(1)
    P = Ws[:, pidx].sum(1)
    N = negw.sum(1)
    if np.any(P * N <= 0):
        raise ValueError('a draw lacks one class')
    return num / (P * N)


# ------------------------------------------------------------------ household bootstrap (own implementation)
def household_weights(unit, B=BOOT_B, seed=BOOT_SEED):
    """Paired nonparametric household bootstrap: households in sorted id order; each draw picks G households with
    replacement (default_rng(seed).integers(0, G, size=(B, G))); a person's weight is its household's pick count."""
    unit = np.asarray(unit)
    households, inverse = np.unique(unit, return_inverse=True)
    G = len(households)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, G, size=(B, G))
    counts = np.zeros((B, G), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(B), G), picks.ravel()), 1.0)
    return counts[:, inverse], counts, G


def boot_mean(W, d):
    d = np.asarray(d, dtype=np.float64)
    return (W @ d) / W.sum(1)


def sd(draws):
    return float(np.std(np.asarray(draws, dtype=np.float64), ddof=1))


def tcrit(one_sided_level, df):
    return float(stats.t.ppf(1.0 - one_sided_level, df))


# ------------------------------------------------------------------ task metrics (own implementation)
def metric_vectors(y, p, k):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or np.any(p < 0) or np.max(np.abs(p.sum(1) - 1)) > 1e-9:
        raise ValueError('probability convention differs')
    if y.min() < 0 or y.max() >= k:
        raise ValueError('label outside class range')
    onehot = (y[:, None] == np.arange(k)[None, :]).astype(np.float64)
    py = p[np.arange(len(y)), y]
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(py, 1e-12)),
            'brier': np.sum((p - onehot) ** 2, axis=1),
            'min_true_class_prob': float(py.min())}


# ------------------------------------------------------------------ statuses (registered rule, own code)
def status_at(kind, L, U, m):
    if kind == 'benefit':
        return 'ESTABLISHED' if L > m else ('ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED')
    if kind == 'no_excess':
        return 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')
    if kind == 'difference':
        return 'ABOVE_ZERO' if L > m else ('BELOW_ZERO' if U < m else 'UNRESOLVED')
    raise ValueError(kind)


# ------------------------------------------------------------------ output
def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else repr(v)
    if isinstance(x, np.ndarray):
        if x.size > 64:
            raise ValueError('refusing to write an array with more than 64 entries into a result file')
        return clean(x.tolist())
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False)
    if '/Users/' in text:
        raise ValueError('refusing to write an absolute home path into a result file')
    path.write_text(text + '\n')
    return sha256_file(path)


def scratch_dir(arg=None):
    p = Path(arg or os.environ.get('ACSH_VERIFY_SCRATCH', ''))
    if not str(p) or str(p) == '.':
        raise SystemExit('scratch directory required (--scratch or ACSH_VERIFY_SCRATCH)')
    p.mkdir(parents=True, exist_ok=True)
    return p


def maxabs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0
