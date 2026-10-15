"""Boosted-tree task teachers and the exact leaf-score correction family.

A fitted CatBoost model is converted to an explicit numpy teacher: oblivious
tree splits (float32 feature > float32 border, bit i for split i), leaf values
(float64, leaf-major per tree), scale and bias. Routing is verified bitwise
against CatBoost's own calc_leaf_indexes and raw scores against
RawFormulaVal, so deployment needs only numpy.

Corrected recipient r: f_r(x) = f_teacher,r(x) + sum_t delta[r,t,leaf_t(x)],
with delta = theta / T (T = tree count). Income is binary: f is the scalar
CatBoost log-odds d, released as softmax(-d/2, +d/2). Occupation: f is the
6-vector, released as softmax(center(f)). Zero theta is the teacher exactly.
Leaf ids are private intermediates and are never transmitted.
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile

import numpy as np

CATBOOST_VERSION = '1.2.8'
REGISTRY = {'depth': (4, 6), 'learning_rate': 0.05, 'max_trees': 800, 'l2_leaf_reg': 3.0,
            'task_type': 'CPU', 'thread_count': 1}
LOSS = {'income': 'Logloss', 'occupation': 'MultiClass'}
CLASSES = {'income': 2, 'occupation': 6}


def seed_for(*items):
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:8], 16)


def catboost():
    import catboost as cb
    if cb.__version__ != CATBOOST_VERSION:
        raise RuntimeError('pinned CatBoost version differs')
    return cb


def make_classifier(task, depth, iterations, seed):
    cb = catboost()
    return cb.CatBoostClassifier(loss_function=LOSS[task], depth=depth, learning_rate=REGISTRY['learning_rate'],
                                 iterations=int(iterations), l2_leaf_reg=REGISTRY['l2_leaf_reg'], task_type='CPU',
                                 thread_count=1, random_seed=seed, verbose=0, allow_writing_files=False)


class TreeTeacher:
    """Explicit oblivious-tree teacher; numpy only at inference."""

    def __init__(self, payload):
        self.task = payload['task']; self.K = CLASSES[self.task]
        self.dim = 1 if self.task == 'income' else 6
        self.depth = int(payload['depth'])
        self.features = np.asarray(payload['split_features'], dtype=np.int64)   # (T, depth)
        self.borders = np.asarray(payload['split_borders'], dtype=np.float32)   # (T, depth)
        self.leaf_values = np.asarray(payload['leaf_values'], dtype=np.float64)  # (T, 2**depth, dim)
        self.scale = float(payload['scale']); self.bias = np.asarray(payload['bias'], dtype=np.float64).reshape(self.dim)
        self.feature_names = tuple(payload['feature_names'])
        self.T = self.features.shape[0]; self.L = 2 ** self.depth
        if (self.features.shape != (self.T, self.depth) or self.borders.shape != self.features.shape
                or self.leaf_values.shape != (self.T, self.L, self.dim) or len(self.feature_names) != 83):
            raise ValueError('teacher payload shape differs')

    def payload(self):
        return {'schema': 'THP1-oblivious-tree-teacher-v1', 'task': self.task, 'depth': self.depth,
                'split_features': self.features.tolist(), 'split_borders': [[float(b) for b in r] for r in self.borders],
                'leaf_values': self.leaf_values.tolist(), 'scale': self.scale, 'bias': self.bias.tolist(),
                'feature_names': list(self.feature_names), 'class_order': list(range(self.K)),
                'decision_rule': 'argmax released probabilities; ties choose the first class'}

    def sha(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def check_inputs(self, X, names):
        if tuple(map(str, names)) != self.feature_names:
            raise ValueError('wrong or reordered raw83 schema')
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 83 or not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
            raise ValueError('finite raw83 matrix required')
        return np.ascontiguousarray(X, dtype=np.float32)

    def route(self, X, names):
        X = self.check_inputs(X, names)
        out = np.zeros((len(X), self.T), dtype=np.int64)
        for i in range(self.depth):
            out |= (X[:, self.features[:, i]] > self.borders[None, :, i]).astype(np.int64) << i
        return out

    def raw_from_leaves(self, leaves, theta=None):
        """Teacher raw scores plus theta/T, float64, shape (n, dim)."""
        t = np.arange(self.T)[None, :]
        raw = self.scale * self.leaf_values[t, leaves].sum(1) + self.bias
        if theta is not None:
            theta = np.asarray(theta, dtype=np.float64).reshape(self.T, self.L, self.dim)
            raw = raw + theta[t, leaves].sum(1) / self.T
        return raw

    def probabilities_from_raw(self, raw):
        return probabilities(self.task, raw)


def probabilities(task, raw):
    raw = np.asarray(raw, dtype=np.float64)
    if task == 'income':
        d = raw.reshape(-1)
        logits = np.column_stack((-d / 2, d / 2))
    else:
        logits = raw - raw.mean(1, keepdims=True)
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def release(task, raw):
    """The only transmitted object: full-precision probability vector and decision."""
    p = probabilities(task, raw)
    return {'probability': p, 'decision': p.argmax(1).astype(np.int64)}


def from_catboost(model, task, depth, feature_names):
    """Convert a fitted CatBoostClassifier into an explicit TreeTeacher."""
    path = tempfile.mktemp(suffix='.json')
    try:
        model.save_model(path, format='json')
        with open(path) as stream:
            J = json.load(stream)
    finally:
        if os.path.exists(path):
            os.remove(path)
    ff = J['features_info']['float_features']
    dim = 1 if task == 'income' else 6
    feats, borders, leaves = [], [], []
    never = float(np.finfo(np.float32).max)  # padding split: finite x > float32 max is never true (bit stays 0)
    for tree in J['oblivious_trees']:
        d = len(tree['splits'])
        if d > depth or any(s['split_type'] != 'FloatFeature' for s in tree['splits']):
            raise ValueError('tree deeper than registered or non-float split')
        feats.append([ff[s['float_feature_index']]['flat_feature_index'] for s in tree['splits']] + [0] * (depth - d))
        borders.append([s['border'] for s in tree['splits']] + [never] * (depth - d))
        lv = np.zeros((2 ** depth, dim)); lv[:2 ** d] = np.asarray(tree['leaf_values'], dtype=np.float64).reshape(2 ** d, dim)
        leaves.append(lv)
    scale, bias = J['scale_and_bias'][0], J['scale_and_bias'][1]
    return TreeTeacher({'task': task, 'depth': depth, 'split_features': feats, 'split_borders': borders,
                        'leaf_values': np.stack(leaves), 'scale': scale, 'bias': bias,
                        'feature_names': list(map(str, feature_names))})


def verify_against_catboost(teacher, model, X, names):
    """Bitwise routing and float64 raw-score parity with CatBoost on one role."""
    cb = catboost()
    leaves = teacher.route(X, names)
    ref_leaves = np.asarray(model.calc_leaf_indexes(cb.Pool(np.asarray(X, dtype=np.float32))), dtype=np.int64)
    raw = teacher.raw_from_leaves(leaves)
    ref_raw = np.asarray(model.predict(np.asarray(X, dtype=np.float32), prediction_type='RawFormulaVal'), dtype=np.float64).reshape(len(X), -1)
    p = teacher.probabilities_from_raw(raw)
    ref_p = np.asarray(model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)
    return {'routing_identical': bool(np.array_equal(leaves, ref_leaves)),
            'raw_max_abs_diff': float(np.max(np.abs(raw - ref_raw))),
            'probability_max_abs_diff': float(np.max(np.abs(p - ref_p))),
            'decision_identical': bool(np.array_equal(p.argmax(1), ref_p.argmax(1)))}
