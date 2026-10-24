"""Independent verifier helpers for the strong-anchor release campaign (SAR1).

Written by the verifier from the stated definitions (PROTOCOL.md 5-8, roles/statistical_reviewer/RULES.md, the cards). Nothing here
imports sar.rules, sar.claims, nrm.rules or sar.devreaders, and nothing here reads a label: label reads happen only in the queue-unit
entry scripts of this folder, through sar.common.load(..., operation='verify').
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
CAMP = 'pcrl_strong_anchor_release_v1'
RES = ROOT / 'results' / CAMP
PRIV = ROOT / '.private' / CAMP
RUNS = PRIV / 'runs'
VDIR = RES / 'roles' / 'verifier'
WORK = VDIR / 'work'

TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD = 0.005
REFS = ('U0', 'Ucal')
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
CONDS = tuple(f'{t}|{r}|{m}' for t in TASKS for r in REFS for m in METRICS) + ('guard|sex0', 'guard|sex1')
Z = 2.0
SE_FLOOR = 0.001
RECIPIENT_LIMIT = 0.010
TARGET = 0.020
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
REG_CELLS = tuple(f'S{v}{r}k{k}' for v in (0, 1) for r in ('R1', 'R2') for k in (0, 1))

# grid constants (READER_SPLITS.json / PROTOCOL 7)
GRID_VERSION = 'SAR1-devgrid-v1'
MODEL_SEED_BASE = 2026101205
SEED_OFFSET = 1000
BOOT_SEED = 2026102400
B_BOOT = 400
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
RECIPES = SRC8 + NEW8


# ------------------------------------------------------------------ hashing
def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ahash(a):
    """SHA-256 over dtype, shape and bytes (the campaign's array-hash convention, used in the reader-cache key)."""
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def view_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    arrs = (inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def wire_hash(p):
    """float64 probabilities then int64 argmax (the recipient's wire)."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes())
    h.update(np.ascontiguousarray(p.argmax(1).astype(np.int64)).tobytes())
    return h.hexdigest()


def cache_key(rep, reader_seed, frac, vh, sex_hash):
    tag = rep if rep in ('R1', 'R2') else f'{rep}:SAR1_devgrid_{rep}'
    return hashlib.sha256(f'{GRID_VERSION}|{tag}|{reader_seed}|{float(frac)}|{vh}|{sex_hash}'.encode()).hexdigest()


# ------------------------------------------------------------------ splits and bootstrap (my implementation of the stated rule)
def household_parts(unit, k, tag):
    groups = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{int(g)}'.encode()).hexdigest() for g in groups]
    order = sorted(range(len(groups)), key=lambda i: keys[i])
    part_of_group = np.empty(len(groups), dtype=np.int64)
    for rank, i in enumerate(order):
        part_of_group[i] = rank % k
    return part_of_group[np.searchsorted(groups, unit)]


def split_roles(unit, rep):
    part = household_parts(unit, 10, f'SAR1_devgrid_{rep}')
    return {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


def reader_seed(v, k):
    return MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def boot_weights(unit_score, v, rep):
    groups, inv = np.unique(unit_score, return_inverse=True)
    G = len(groups)
    rng = np.random.default_rng(BOOT_SEED + 100 * int(v) + REP_INDEX[rep])
    W = np.zeros((B_BOOT, len(unit_score)))
    for b in range(B_BOOT):
        counts = np.bincount(rng.integers(0, G, G), minlength=G)
        W[b] = counts[inv]
    return W


def auc_weighted(y, s, w=None):
    """P(score of a women's record > score of a men's record) + 0.5 P(tie) (SEX 1 positive), with case weights."""
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    vals, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=w * (y == 1), minlength=len(vals))
    wn = np.bincount(inv, weights=w * (y == 0), minlength=len(vals))
    below = np.concatenate(([0.0], np.cumsum(wn)[:-1]))
    return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))


def auc_draws(y, s, W):
    """auc_weighted under each row of W (the unique-value pass is shared across draws)."""
    y = np.asarray(y).astype(np.int64)
    vals, inv = np.unique(np.asarray(s, dtype=np.float64), return_inverse=True)
    pos, neg = (y == 1).astype(np.float64), (y == 0).astype(np.float64)
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        wp = np.bincount(inv, weights=W[b] * pos, minlength=len(vals))
        wn = np.bincount(inv, weights=W[b] * neg, minlength=len(vals))
        below = np.concatenate(([0.0], np.cumsum(wn)[:-1]))
        out[b] = np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum())
    return out


# ------------------------------------------------------------------ per-person task metrics and clustered SEs
def person_metrics(y, p, k):
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or (p < 0).any() or np.abs(p.sum(1) - 1).max() > 1e-9:
        raise ValueError('not a probability-vector release')
    n = len(y)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    return {'acc': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(n), y], 1e-12)),
            'brier': np.sum((p - onehot) ** 2, axis=1)}


def clustered_se(d, unit):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    _, inv = np.unique(unit, return_inverse=True)
    G = int(inv.max()) + 1
    sums = np.bincount(inv, weights=d - d.mean(), minlength=G)
    return float(math.sqrt(G / (G - 1) * float(np.sum(sums ** 2))) / n)


def subgroup_clustered_se(d, mask, unit):
    d = np.asarray(d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = len(d)
    m = d[mask].mean()
    p = mask.mean()
    z = np.where(mask, (d - m) / p, 0.0)
    _, inv = np.unique(unit, return_inverse=True)
    G = int(inv.max()) + 1
    sums = np.bincount(inv, weights=z, minlength=G)
    return float(math.sqrt(G / (G - 1) * float(np.sum(sums ** 2))) / n)


def utility(labels, rel, refs, sex, unit):
    """{condition: {'value', 'se', 'cap'}} for the 12 clauses and the 2 guards of one version."""
    own = {t: person_metrics(labels[t], rel[t], K[t]) for t in TASKS}
    out = {}
    for t in TASKS:
        for r in REFS:
            ref = person_metrics(labels[t], refs[r][t], K[t])
            for m in METRICS:
                if m == 'accuracy_loss':
                    d = ref['acc'] - own[t]['acc']
                elif m == 'nll_excess':
                    d = own[t]['nll'] - ref['nll']
                else:
                    d = own[t]['brier'] - ref['brier']
                out[f'{t}|{r}|{m}'] = {'value': float(d.mean()), 'se': clustered_se(d, unit), 'cap': CAPS[m]}
    u0 = person_metrics(labels['occupation'], refs['U0']['occupation'], 5)
    d = own['occupation']['brier'] - u0['brier']
    for g in (0, 1):
        mask = sex == g
        out[f'guard|sex{g}'] = {'value': float(d[mask].mean()), 'se': subgroup_clustered_se(d, mask, unit), 'cap': GUARD}
    return out, own


def paired_costs(a, b, sex, unit):
    """loss(a) - loss(b) per person (accuracy as a loss), means and clustered SEs."""
    out = {}
    for t in TASKS:
        for m, key in (('acc_loss', 'acc'), ('nll', 'nll'), ('brier', 'brier')):
            d = (1.0 - a[t]['acc']) - (1.0 - b[t]['acc']) if key == 'acc' else a[t][key] - b[t][key]
            out[f'{t}|{m}'] = {'value': float(d.mean()), 'se': clustered_se(d, unit)}
            if m == 'brier':
                for g, name in ((0, 'men_brier'), (1, 'women_brier')):
                    mask = sex == g
                    out[f'{t}|{name}'] = {'value': float(d[mask].mean()), 'se': subgroup_clustered_se(d, mask, unit)}
    tot = (a['income']['nll'] + a['occupation']['nll']) - (b['income']['nll'] + b['occupation']['nll'])
    out['total|nll'] = {'value': float(tot.mean()), 'se': clustered_se(tot, unit)}
    return out


# ------------------------------------------------------------------ the inherited aggregate screen (my implementation)
def parse_cell(c):
    return int(c[1]), c[2:4], int(c[5:])


def aggregate(cells, ses, mcs=None, z=Z, floor=SE_FLOOR):
    mcs = mcs or {}
    blocks = {}
    for c, x in cells.items():
        v, r, k = parse_cell(c)
        blocks.setdefault((v, r), []).append((float(x), float(ses[c]), float(mcs.get(c, 0.0) or 0.0)))
    alias = all(t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0 for b in blocks.values() for t in b)

    def summ(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b]) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        se_hh = math.sqrt(sum(t[1] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b]))
        se_mc = math.sqrt(sum(t[2] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b]))
        se_rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        raw = math.sqrt(se_hh ** 2 + se_rs ** 2 + se_mc ** 2)
        se = 0.0 if alias else max(raw, floor)
        return {'mean': x, 'se_agg': se, 'se_household': se_hh, 'se_reader_split': se_rs, 'se_mc': se_mc,
                'lower': x - z * se, 'upper': x + z * se, 'block_means': {f'S{b[0]}{b[1]}': m for b, m in sorted(mb.items())}}
    keys = sorted(blocks)
    tot = summ(keys)
    tot['per_version'] = {v: summ([b for b in keys if b[0] == v]) for v in (0, 1)}
    tot['alias'] = alias
    return tot


def benefit_label(agg, threshold=0.0):
    names = (('AGG_POSITIVE', 'AGG_POSITIVE_POINT_ONLY', 'AGG_NOT_POSITIVE') if threshold == 0 else
             ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY', 'TARGET_AGG_NOT_MET'))
    if agg['alias']:
        return 'ALIAS_ZERO' if threshold == 0 else names[2]
    if agg['lower'] > threshold and all(agg['per_version'][v]['mean'] > threshold for v in (0, 1)):
        return names[0]
    if agg['mean'] > threshold:
        return names[1]
    return names[2]


def recipient_label(agg, limit=RECIPIENT_LIMIT):
    pv = agg['per_version']
    if agg['alias']:
        lab = 'ALIAS_ZERO'
    elif any(pv[v]['lower'] > limit for v in (0, 1)):
        lab = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif not all(pv[v]['mean'] <= limit for v in (0, 1)):
        lab = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(pv[v]['upper'] <= limit for v in (0, 1)):
        lab = 'RECIPIENT_AGG_RESOLVED'
    else:
        lab = 'RECIPIENT_AGG_EDGE'
    return lab, lab in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO')


def pooled(per_version):
    vals = [float(per_version[v]['value']) for v in (0, 1)]
    ses = [math.sqrt(float(per_version[v]['se']) ** 2 + float(per_version[v].get('se_mc', 0.0) or 0.0) ** 2) for v in (0, 1)]
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2, 'by_version': {0: vals[0], 1: vals[1]}}


def close(a, b, tol):
    return a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and abs(float(a) - float(b)) <= tol


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default) + '\n')


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
