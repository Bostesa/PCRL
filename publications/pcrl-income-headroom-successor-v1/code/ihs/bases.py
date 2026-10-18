"""Shared, verified base outputs for every family (no label is read here; label-free features only).

The replication's frozen CatBoost teachers T_TREE route every person (145-column admitted encoding of the six permitted raw
fields) to one leaf per tree; LOCAL_TREE / JOINT_TREE / TASK_REFIT add stored leaf-score corrections theta (Stage B, epochs
E0, E2, E5, E10, E20 admitted). Released probability = acs_replication.trees.probabilities(task, raw score).
`base_outputs` recomputes them for any role and checks the E20 / teacher outputs against the stored replication outputs
(<= 1e-12, identical decisions). The accepted occupation output of H_LJ (and of the retired finalist) is
base_outputs(seed, 'JOINT_TREE', ...)['occupation'] at E20.

Also: the source campaign's decision-aligned coarse-code helpers (label-free thresholds) and the standardized L2 logistic map
used for calibration (numpy-only deployment), unchanged from amd.fam_b.
"""
from __future__ import annotations
import warnings

import numpy as np

from acs_replication import trees
from . import common as C

ARMS = ('T_TREE', 'LOCAL_TREE', 'JOINT_TREE', 'TASK_REFIT')


def load_X(role):
    F = C.load_role(role, ('features',))
    return F['X'], F['feature_names'], F['row_id'], F['unit']


def base_raw(seed, arm, X, names, label='E20', tasks=C.TASKS):
    thetas = None if arm == 'T_TREE' else C.stage_b_theta(seed, arm, label)
    out = {}
    for i, t in enumerate(C.TASKS):
        if t not in tasks:
            continue
        te = C.teacher(seed, t)
        out[t] = te.raw_from_leaves(te.route(X, names), None if thetas is None else thetas[i])
    return out


def base_outputs(seed, arm, roles, label='E20'):
    """{role: {'raw': {task: raw}, 'prob': {task: prob}, 'row_id', 'unit'}}; E20 / teacher outputs checked against storage."""
    sa, sb = C.stage_a_outputs(seed), C.stage_b_outputs(seed)
    out = {}
    for role in roles:
        X, names, rid, unit = load_X(role)
        raw = base_raw(seed, arm, X, names, label)
        prob = {t: trees.probabilities(t, raw[t]) for t in C.TASKS}
        if label == 'E20' or arm == 'T_TREE':
            if not np.array_equal(sa[f'row_id__{role}'], rid):
                raise AssertionError(f'{role}: row order differs from the stored Stage A outputs')
            for t in C.TASKS:
                ref = sa[f'T_TREE__{role}__{t}'] if arm == 'T_TREE' else sb[f'{arm}__{role}__{t}']
                if float(np.max(np.abs(prob[t] - ref))) > 1e-12 or not np.array_equal(prob[t].argmax(1), ref.argmax(1)):
                    raise AssertionError(f'{arm} {role} {t}: recomputed output differs from the stored output')
        out[role] = {'raw': raw, 'prob': prob, 'row_id': rid, 'unit': unit}
    return out


def joint_occupation(seed, roles):
    """The accepted, frozen H_LJ / JOINT_TREE occupation output (E20) for each role."""
    bo = base_outputs(seed, 'JOINT_TREE', roles)
    return {r: bo[r]['prob']['occupation'] for r in roles}


def stored_reference(seed, name, role):
    """U0 / Ucal / T_TREE stored Stage A outputs for any role (e.g. U0 on MODEL_FIT for an internal guard diagnostic)."""
    sa = C.stage_a_outputs(seed)
    return {t: sa[f'{name}__{role}__{t}'] for t in C.TASKS}


# ---------------------------------------------------------------- source coarse-code helpers (label-free thresholds)
def coarse_bins(d, thresholds):
    return np.searchsorted(np.asarray(thresholds, dtype=np.float64), np.asarray(d, dtype=np.float64).reshape(-1), side='left')


def thresholds(d_fit, m, kind):
    """'quantile' (m quantiles), 'decision' (threshold 0), 'aligned' (m even: m/2 quantile bins each side of 0)."""
    d_fit = np.asarray(d_fit, dtype=np.float64).reshape(-1)
    if kind == 'decision':
        return [0.0]
    if kind == 'quantile':
        return [float(q) for q in np.quantile(d_fit, np.arange(1, m) / m)]
    if kind == 'aligned':
        if m % 2:
            raise ValueError('aligned bins need even m')
        k = m // 2
        neg, pos = d_fit[d_fit <= 0], d_fit[d_fit > 0]
        return ([float(q) for q in np.quantile(neg, np.arange(1, k) / k)] + [0.0] + [float(q) for q in np.quantile(pos, np.arange(1, k) / k)])
    raise ValueError(kind)


class H:
    """Standardized L2 logistic map (stored as plain coefficients so deployment is numpy only)."""

    def __init__(self, Creg=1.0):
        self.Creg = Creg

    def fit(self, Z, y):
        from sklearn.linear_model import LogisticRegression
        self.mu = Z.mean(0); self.sd = Z.std(0); self.sd[self.sd == 0] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = LogisticRegression(C=self.Creg, max_iter=5000, solver='lbfgs', tol=1e-10).fit((Z - self.mu) / self.sd, y)
        self.w = m.coef_.reshape(-1).astype(np.float64); self.b = float(m.intercept_[0])
        return self

    def logit(self, Z):
        return ((Z - self.mu) / self.sd) @ self.w + self.b

    def prob(self, Z):
        return trees.probabilities('income', self.logit(Z))

    def state(self):
        return {'C': self.Creg, 'mu': self.mu.tolist(), 'sd': self.sd.tolist(), 'w': self.w.tolist(), 'b': self.b}
