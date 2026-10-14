"""Independent numpy arithmetic for the separate-pool Stage A verifier.

Written without importing or copying runner arithmetic. Two forward paths:
'f32' emulates the deployed dtypes (float32 encoder, float64 head) with numpy;
'f64' runs the whole network in float64 from the (exactly representable)
float32 weights. Their difference is reported as arithmetic discrepancy.
"""
from __future__ import annotations
import hashlib

import numpy as np

OCC_KEYS = ('enc.1.0.weight', 'enc.1.0.bias', 'enc.1.2.weight', 'enc.1.2.bias',
            'enc.1.4.weight', 'enc.1.4.bias', 'head.1.weight', 'head.1.bias')
OCC_SHAPES = ((64, 83), (64,), (64, 64), (64,), (16, 64), (16,), (6, 16), (6,))
INC_KEYS = ('enc.0.0.weight', 'enc.0.0.bias', 'enc.0.2.weight', 'enc.0.2.bias',
            'enc.0.4.weight', 'enc.0.4.bias', 'head.0.weight', 'head.0.bias')
N_OCC = sum(int(np.prod(s)) for s in OCC_SHAPES)
FOLD_SALT = 'IVR1|FIT-crossfold|20261010|'
SPLIT_SALT = 'SPG1|UPDATE-GUARD-split|20261013|'
PROBE_SALT = 'SPG1|UPDATE-wire-probe|20261013|'


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def fold_of(unit):
    return int(h(FOLD_SALT + str(unit)), 16) % 3


def unflatten_occ(vec):
    """Trace row (float64) -> dict of occupation tensors in deployed dtypes."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape != (N_OCC,):
        raise ValueError('trace row length')
    out = {}; o = 0
    for k, s in zip(OCC_KEYS, OCC_SHAPES):
        n = int(np.prod(s)); part = vec[o:o + n].reshape(s); o += n
        out[k] = part.astype(np.float32) if k.startswith('enc.') else part.copy()
    return out


def flatten_occ(state):
    return np.concatenate([np.asarray(state[k], dtype=np.float64).reshape(-1) for k in OCC_KEYS])


def encoder(state, i, X, mode):
    dt = np.float32 if mode == 'f32' else np.float64
    a = np.asarray(X, dtype=dt)
    for j, relu in ((0, True), (2, True), (4, False)):
        W = np.asarray(state[f'enc.{i}.{j}.weight']).astype(dt); b = np.asarray(state[f'enc.{i}.{j}.bias']).astype(dt)
        a = a @ W.T + b
        if relu:
            a = np.maximum(a, dt(0))
    return a


def head_probs(z, W, b):
    z = np.asarray(z, dtype=np.float64)
    l = z @ np.asarray(W, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
    c = l - l.mean(axis=1, keepdims=True)
    e = np.exp(c - c.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return c, p, np.argmax(p, axis=1)


def occupation_wire(state, X, mode):
    z = encoder(state, 1, X, mode)
    c, p, d = head_probs(z, state['head.1.weight'], state['head.1.bias'])
    return {'representation': z, 'centered_logits': c, 'probability': p, 'decision': d}


def income_wire(state, X, B, mode, project=True):
    z = encoder(state, 0, X, mode).astype(np.float64)
    if project:
        z = (z @ B.T) @ B
    c, p, d = head_probs(z, state['head.0.weight'], state['head.0.bias'])
    return {'representation': z, 'centered_logits': c, 'probability': p, 'decision': d}


def pool_metrics(p, d, y):
    y = np.asarray(y).astype(np.int64); n = len(y); K = p.shape[1]
    pt = p[np.arange(n), y]
    onehot = np.zeros_like(p); onehot[np.arange(n), y] = 1.0
    correct = int(np.sum(d == y))
    return {'nll': float(np.mean(-np.log(np.maximum(pt, 1e-12)))),
            'brier': float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
            'correct': correct, 'rows': n, 'accuracy': correct / n}


def full_metrics(p, y):
    y = np.asarray(y).astype(np.int64); K = p.shape[1]
    d = np.argmax(p, axis=1)
    m = pool_metrics(p, d, y)
    recall = []
    for k in range(K):
        sel = y == k
        recall.append(float(np.mean(d[sel] == k)) if sel.any() else 0.0)
    m.update(balanced_accuracy=float(np.mean(recall)), per_class_recall=recall,
             class_counts=[int(np.sum(y == k)) for k in range(K)], confidence=float(np.mean(np.max(p, axis=1))))
    return m


def deltas(cand, ref):
    return {'accuracy_loss': ref['accuracy'] - cand['accuracy'], 'nll_excess': cand['nll'] - ref['nll'],
            'brier_excess': cand['brier'] - ref['brier']}


def l2(v):
    v = np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.sum(v * v)))
