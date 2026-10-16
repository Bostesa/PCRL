"""Role loading through the active queue, task metrics and the original utility clauses.

Roles (inherited, manifests authoritative): UPDATE 8180 and GUARD 2043 form
the fitting pool (FIT folds 0/1); FIT2 = VALIDATION 5211 (FIT fold 2) is the
internal selection role; ATT 4064 trains/orients fresh readers; INNER 2235 is
the once-opened development reporting role. INNER labels are not loadable here.
"""
from __future__ import annotations
from zipfile import ZipFile

import numpy as np

COUNTS = {'UPDATE': (8180, 8176), 'GUARD': (2043, None), 'FIT2': (5211, None), 'ATT': (4064, 4061), 'INNER': (2235, 2234)}
FIELDS = {'features': ('row_id', 'unit', 'X', 'feature_names'), 'income': ('row_id', 'unit', 'income'),
          'occupation': ('row_id', 'unit', 'occupation'), 'sex': ('row_id', 'unit', 'sex'),
          'references_s0': ('row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2'),
          'references_s1': ('row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2')}
FORBIDDEN_INPUTS = {'sex', 'race', 'income', 'occupation', 'fnlwgt', 'row_id', 'unit', 'id', 'hidden_flag'}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 6}


def load_npz(item, verify, fields):
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as archive:
        if {x.filename for x in archive.infolist()} != {f + '.npy' for f in fields}:
            raise PermissionError('unexpected export fields: ' + p.name)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in fields}


def load_role(role, items, verify, wanted):
    """items: {field: {path, sha256}}; wanted: subset of FIELDS keys. INNER labels refused."""
    if role == 'INNER' and set(wanted) & {'income', 'occupation', 'sex'}:
        raise PermissionError('INNER labels open only in the committed reporting stage')
    f = load_npz(items['features'], verify, FIELDS['features'])
    n, groups = COUNTS[role]
    if len(f['row_id']) != n or len(np.unique(f['row_id'])) != n or (groups and len(np.unique(f['unit'])) != groups):
        raise PermissionError(f'{role} row/group counts differ')
    names = tuple(map(str, f['feature_names']))
    if (f['X'].shape != (n, 83) or f['X'].dtype != np.float32 or not np.isfinite(f['X']).all() or len(set(names)) != 83
            or any(x.lower().split('=')[0] in FORBIDDEN_INPUTS for x in names)):
        raise PermissionError(f'{role} raw83 schema differs')
    out = dict(f)
    for name in wanted:
        if name == 'features':
            continue
        z = load_npz(items[name], verify, FIELDS[name])
        if not (np.array_equal(z['row_id'], f['row_id']) and np.array_equal(z['unit'], f['unit'])):
            raise PermissionError(f'{role} {name} pairing differs')
        if name.startswith('references'):
            out[name] = z
        else:
            a = z[name]; k = 2 if name in ('income', 'sex') else 6
            if a.shape != (n,) or a.dtype.kind not in 'iu' or a.min() < 0 or a.max() >= k:
                raise PermissionError(f'{role} {name} label convention differs')
            out[name] = a.astype(np.int64)
    return out


def group_disjoint_split(unit, seed, fraction=0.15):
    """Deterministic label-free group split (inherited Recipe-A convention)."""
    import math
    groups = np.unique(unit)
    perm = np.random.default_rng(seed).permutation(len(groups))
    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * fraction)))
    chosen = groups[perm[:count]]
    sel = np.isin(unit, chosen)
    return np.flatnonzero(~sel), np.flatnonzero(sel)


def preserve_support(unit, fit, val, labels):
    """Move the smallest whole validation group for each class absent from the fit part."""
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


def internal_split(unit, labels, seed):
    fit, val = group_disjoint_split(unit, seed + 17001, 0.15)
    return preserve_support(unit, fit, val, labels)


def task_vectors(y, p, k):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
        raise ValueError('probability convention differs')
    target = y[:, None] == np.arange(k)[None, :]
    return {'accuracy': (p.argmax(1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - target) ** 2).sum(1)}


def summary(y, p, k):
    v = task_vectors(y, p, k); d = np.asarray(p).argmax(1)
    prior = np.bincount(y, minlength=k) / len(y)
    const = np.repeat(prior[None, :], len(y), 0); cv = task_vectors(y, const, k)
    recall = [float((d[y == c] == c).mean()) if np.any(y == c) else None for c in range(k)]
    present = [r for r in recall if r is not None]
    return {'accuracy': float(v['accuracy'].mean()), 'nll': float(v['nll'].mean()), 'brier': float(v['brier'].mean()),
            'balanced_accuracy': float(np.mean(present)), 'per_class_recall': recall,
            'class_counts': np.bincount(y, minlength=k).tolist(), 'mean_confidence': float(np.asarray(p).max(1).mean()),
            'constant_accuracy': float(cv['accuracy'].mean()), 'constant_nll': float(cv['nll'].mean()),
            'constant_brier': float(cv['brier'].mean()),
            'accuracy_gain_over_constant': float(v['accuracy'].mean() - cv['accuracy'].mean()),
            'nll_gain_over_constant': float(cv['nll'].mean() - v['nll'].mean()),
            'distinct_decisions': int(len(np.unique(d)))}


def clauses(labels, probs, refs, seed, ref_names):
    """Per seed: task x reference x (accuracy loss, NLL excess, summed Brier excess) with inclusive caps."""
    out = []
    for i, task in enumerate(TASKS):
        k = K[task]; own = summary(labels[task], probs[i], k)
        for ref in ref_names:
            r = summary(labels[task], refs[ref][i], k)
            diff = {'accuracy_loss': r['accuracy'] - own['accuracy'], 'nll_excess': own['nll'] - r['nll'],
                    'brier_excess': own['brier'] - r['brier']}
            for metric, cap in CAPS.items():
                out.append({'seed': seed, 'task': task, 'reference': ref, 'metric': metric, 'value': diff[metric],
                            'cap': cap, 'normalized': diff[metric] / cap, 'point_pass': bool(diff[metric] <= cap)})
    return out


def arr_sha(a):
    import hashlib
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()
