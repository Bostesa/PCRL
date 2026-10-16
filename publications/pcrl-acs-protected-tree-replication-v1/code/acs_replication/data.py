"""Role loading with access guards, task metrics and the inherited utility clauses (ACS benchmark).

Guards: SEALED_ASSESSMENT has no export and cannot be loaded. DEV_REPORT truth (labels, SEX, weights)
lives under exports/DEV_REPORT_TRUTH and is refused unless the committed REPORTING_LOCK exists.
Fields are verified against the extraction receipt's sha256 values.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
RESULTS = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
FIELDS = {'features': ('row_id', 'unit', 'X', 'feature_names'), 'income': ('row_id', 'unit', 'income'),
          'occupation': ('row_id', 'unit', 'occupation'), 'sex': ('row_id', 'unit', 'sex'), 'weights': ('row_id', 'unit', 'pwgtp')}
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def receipt():
    return json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())


def lock_committed(name):
    p = RESULTS / name
    if not p.is_file():
        return False
    out = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%H', '--', str(p)], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', str(p)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not dirty.stdout.strip()


def _load(item, fields):
    p = ROOT / item['path']
    if sha_file(p) != item['sha256']:
        raise PermissionError(f'export hash differs: {p.name}')
    with ZipFile(p) as z:
        if {x.filename for x in z.infolist()} != {f + '.npy' for f in fields}:
            raise PermissionError('unexpected export fields: ' + p.name)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in fields}


def load_role(role, wanted=('features',)):
    if role == 'SEALED_ASSESSMENT' or role not in ROLES:
        raise PermissionError('the sealed assessment is closed in this campaign')
    if role == 'DEV_REPORT' and set(wanted) & {'income', 'occupation', 'sex', 'weights'} and not lock_committed('REPORTING_LOCK.json'):
        raise PermissionError('DEV_REPORT truth opens only after the committed REPORTING_LOCK')
    files = receipt()['files'][role]
    out = _load(files['features'], FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    for name in wanted:
        if name == 'features':
            continue
        z = _load(files[name], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'{role} {name} pairing differs')
        out[name] = z['pwgtp' if name == 'weights' else name].astype(np.int64)
    return out


def internal_split(unit, labels, seed):
    """Recipe-A internal early-stopping split: 15% of HOUSEHOLDS (label-free), whole-household class-support repair."""
    from task_headroom.data import group_disjoint_split
    fit, val = group_disjoint_split(unit, seed + 17001, 0.15)
    fit = list(fit); val = list(val); moved = []
    for task in sorted(labels):
        y = labels[task]
        for c in range(K[task]):
            if np.any(y[fit] == c):
                continue
            cand = sorted({int(unit[i]) for i in val if y[i] == c}, key=lambda g: (int((unit[val] == g).sum()), g))
            if not cand:
                raise ValueError(f'class {c} of {task} absent from fitting pool')
            g = cand[0]; moving = [i for i in val if unit[i] == g]
            val = [i for i in val if unit[i] != g]; fit = sorted(fit + moving); moved.append(g)
    return np.asarray(fit, dtype=np.int64), np.asarray(val, dtype=np.int64), moved


def task_vectors(y, p, k):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
        raise ValueError('probability convention differs')
    target = y[:, None] == np.arange(k)[None, :]
    return {'accuracy': (p.argmax(1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - target) ** 2).sum(1)}


def summary(y, p, k, w=None):
    v = task_vectors(y, p, k); d = np.asarray(p).argmax(1)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    mean = lambda a: float(np.sum(w * a) / np.sum(w))
    prior = np.bincount(y, weights=w, minlength=k) / w.sum()
    cv = task_vectors(y, np.repeat(prior[None, :], len(y), 0), k)
    recall = [float(np.sum(w[(y == c)] * (d[y == c] == c)) / np.sum(w[y == c])) if np.any(y == c) else None for c in range(k)]
    present = [r for r in recall if r is not None]
    return {'accuracy': mean(v['accuracy']), 'nll': mean(v['nll']), 'brier': mean(v['brier']),
            'balanced_accuracy': float(np.mean(present)), 'per_class_recall': recall,
            'constant_accuracy': mean(cv['accuracy']), 'constant_nll': mean(cv['nll']), 'constant_brier': mean(cv['brier']),
            'accuracy_gain_over_constant': mean(v['accuracy']) - mean(cv['accuracy']),
            'nll_gain_over_constant': mean(cv['nll']) - mean(v['nll']),
            'brier_gain_over_constant': mean(cv['brier']) - mean(v['brier']),
            'mean_confidence': mean(np.asarray(p).max(1)), 'distinct_decisions': int(len(np.unique(d)))}


def clauses(labels, probs, refs, seed, ref_names=('U0', 'Ucal')):
    """Per seed: task x reference x (accuracy loss, NLL excess, summed Brier excess), inclusive caps."""
    out = []
    for i, task in enumerate(TASKS):
        k = K[task]; own = summary(labels[task], probs[i], k)
        for ref in ref_names:
            r = summary(labels[task], refs[ref][i], k)
            diff = {'accuracy_loss': r['accuracy'] - own['accuracy'], 'nll_excess': own['nll'] - r['nll'], 'brier_excess': own['brier'] - r['brier']}
            for metric, cap in CAPS.items():
                out.append({'seed': seed, 'task': task, 'reference': ref, 'metric': metric, 'value': diff[metric], 'cap': cap,
                            'normalized': diff[metric] / cap, 'point_pass': bool(diff[metric] <= cap)})
    return out


def arr_sha(a):
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def order_rows(row_id, salt, n):
    ranked = sorted(range(len(row_id)), key=lambda i: hashlib.sha256(f'{salt}|{int(row_id[i])}'.encode()).hexdigest())
    return np.asarray(ranked[:n], dtype=np.int64)
