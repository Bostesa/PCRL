"""Independent verifier helpers (own code; numpy only). No campaign module is imported here.

Only aggregates are ever written by the verifier scripts; person-level arrays stay in memory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / 'results' / 'pcrl_scm_utility_matched_v1'
VDIR = RES / 'roles' / 'verifier'
PRIV = ROOT / '.private' / 'pcrl_scm_utility_matched_v1'
SRC = PRIV / 'source'
SB_RES = ROOT / 'results' / 'pcrl_scm_engineering_benchmark_v1'
SB_PRIV = ROOT / '.private' / 'pcrl_scm_engineering_benchmark_v1'
REPL_RES = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
HYB_RES = ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1'
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
OUT_ROLES = ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
TASKS = ('income', 'occupation')
KCLS = {'income': 2, 'occupation': 5}
SEEDS = (0, 1)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def public_export_sha():
    return jload(REPL_RES / 'EXTRACTION_SUMMARY.json')['export_file_sha256']


def export(role, field):
    """Admitted export copy, after an own hash check against the public extraction summary. DEV_REPORT truth is refused."""
    if role == 'DEV_REPORT' and field in ('income', 'occupation', 'sex'):
        raise PermissionError('DEV_REPORT truth is not read through this helper')
    if role not in ROLES6:
        raise PermissionError(f'role {role} refused')
    p = SRC / 'exports' / role / f'{field}.npz'
    if sha_file(p) != public_export_sha()[role][field]:
        raise PermissionError(f'hash differs: {p}')
    return npz(p)


def seed_for(*items):
    """Own re-implementation of the registered seed derivation: first 8 hex digits of sha256('|'.join(items))."""
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:8], 16)


def write_json(name, obj):
    p = VDIR / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default))
    os.replace(tmp, p)
    return p


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def auc_mw(y, s, w=None):
    """Own Mann-Whitney AUC with midrank ties: P(s_pos > s_neg) + 0.5 P(tie), optional nonnegative weights."""
    y = np.asarray(y).astype(np.int64); s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(s, kind='mergesort'); ss = s[order]; yy = y[order]; ww = w[order]
    # group equal scores
    brk = np.flatnonzero(np.diff(ss) != 0) + 1
    starts = np.concatenate(([0], brk)); ends = np.concatenate((brk, [len(ss)]))
    wp = ww * (yy == 1); wn = ww * (yy == 0)
    P = np.array([wp[a:b].sum() for a, b in zip(starts, ends)])
    Nn = np.array([wn[a:b].sum() for a, b in zip(starts, ends)])
    neg_below = np.concatenate(([0.0], np.cumsum(Nn)[:-1]))
    num = (P * (neg_below + 0.5 * Nn)).sum()
    return float(num / (P.sum() * Nn.sum()))


def task_vectors(y, p, k):
    """Own per-person accuracy (argmax, first max), NLL -log max(p_y, 1e-12) and summed Brier."""
    y = np.asarray(y).astype(np.int64); p = np.asarray(p, dtype=np.float64)
    assert p.shape == (len(y), k)
    py = p[np.arange(len(y)), y]
    onehot = np.zeros_like(p); onehot[np.arange(len(y)), y] = 1.0
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(py, 1e-12)),
            'brier': ((p - onehot) ** 2).sum(axis=1)}
