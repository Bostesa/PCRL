"""Independent verifier helpers for the PCRL overnight occupation-headroom campaign (exploratory development; verification only).

Own implementations of every checked quantity: file and wire hashes, household partitions from the registered tags, per-person
task losses, household-clustered standard errors (paired means and linearized subgroup means), the whole-release reserve and the
witness labels. Nothing here imports ohn.evaluate, ohn.devreaders, ohn.rules, ohn.grid_summary, ihs.evaluate or ihs.reserve.
Data access uses ohn.common / ohn.devrun loaders only (hash-checked, label reads logged under the verifier's unit ids with
purposes starting "independent verification:").

Census context: public-use ACS 2019 1-Year PUMS person records for Florida (development roles only). Income class (2 classes) and
occupation group (5 broad civilian groups) are predicted from six permitted fields; SEX is the dataset's recorded demographic
column (sex0 = men, sex1 = women), used here only as an evaluation label on MODEL_SELECT.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAMP = 'pcrl_occupation_headroom_overnight_v1'
RES = ROOT / 'results' / CAMP
PRIV = ROOT / '.private' / CAMP
RUNS = PRIV / 'runs'
VER = RES / 'roles' / 'verifier'
RECORDS = VER / 'records'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
SRC_RESULTS = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'
PURPOSE = 'independent verification: '

TASKS = ('income', 'occupation')
NCLASS = {'income': 2, 'occupation': 5}
CAP = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD_CAP = 0.005
MS_HOUSEHOLDS = 5306


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


# ------------------------------------------------------------------------------------------------------------------ hashing
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def wire_hash(prob):
    """Released wire of one recipient: the float64 probability vectors (C order) followed by the int64 argmax decisions."""
    a = np.array(prob, dtype=np.float64, order='C', copy=True)
    d = np.array(np.argmax(a, axis=1), dtype=np.int64, order='C')
    h = hashlib.sha256()
    h.update(memoryview(a).cast('B'))
    h.update(memoryview(d).cast('B'))
    return h.hexdigest()


def array_hash(a):
    """dtype + shape + bytes (the lineage's array hash definition)."""
    z = np.array(a, order='C', copy=True)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + memoryview(z).cast('B').tobytes()).hexdigest()


# --------------------------------------------------------------------------------------------- household partitions (own code)
def household_parts(unit, k, tag):
    """Registered rule: households ordered by SHA-256('<tag>|<household>') hex digest, dealt round-robin into k parts."""
    hh = sorted({int(g) for g in np.asarray(unit).tolist()})
    keyed = sorted(hh, key=lambda g: hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest())
    part_of = {g: i % k for i, g in enumerate(keyed)}
    return np.fromiter((part_of[int(g)] for g in np.asarray(unit).tolist()), dtype=np.int64, count=len(unit))


def household_set_sha(unit_subset):
    hh = sorted({int(g) for g in np.asarray(unit_subset).tolist()})
    return hashlib.sha256(','.join(str(g) for g in hh).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------- task losses (own code)
def check_prob(p, k):
    p = np.asarray(p)
    out = {'dtype': str(p.dtype), 'shape': list(p.shape), 'ndim_ok': p.ndim == 2 and p.shape[1] == k}
    if out['ndim_ok']:
        pf = p.astype(np.float64)
        out.update({'finite': bool(np.isfinite(pf).all()), 'nonnegative': bool((pf >= 0).all()),
                    'max_abs_rowsum_minus_1': float(np.max(np.abs(pf.sum(axis=1) - 1.0))) if len(pf) else 0.0})
        srt = np.sort(pf, axis=1)
        out['argmax_exact_ties'] = int(np.sum(srt[:, -1] == srt[:, -2]))
        out['contract_ok'] = bool(out['finite'] and out['nonnegative'] and out['max_abs_rowsum_minus_1'] <= 1e-9 and p.dtype == np.float64)
    else:
        out['contract_ok'] = False
    return out


def losses(y, p, k):
    """Per-person accuracy indicator, NLL (probability floored at 1e-12, the lineage convention) and summed multiclass Brier."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    n = y.shape[0]
    if p.shape != (n, k):
        raise ValueError('shape')
    rows = np.arange(n)
    acc = np.equal(np.argmax(p, axis=1), y).astype(np.float64)
    nll = -np.log(np.maximum(p[rows, y], 1e-12))
    e = p.copy()
    e[rows, y] -= 1.0
    brier = np.einsum('ij,ij->i', e, e)
    return {'acc': acc, 'nll': nll, 'brier': brier}


def _cluster_sums(v, unit):
    order = np.argsort(np.asarray(unit), kind='stable')
    us = np.asarray(unit)[order]
    starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
    return np.add.reduceat(np.asarray(v, dtype=np.float64)[order], starts), len(starts)


def clustered_se(d, unit):
    """SE of a person-level mean with household clustering (CR1-type G/(G-1) factor)."""
    d = np.asarray(d, dtype=np.float64)
    s, G = _cluster_sums(d - d.mean(), unit)
    return math.sqrt(G / (G - 1) * float(np.dot(s, s))) / d.shape[0]


def subgroup_mean_se(d, mask, unit):
    """Ratio-estimator mean of d over mask (sum(d*m) / sum(m)) with its delta-method household-clustered SE."""
    d = np.asarray(d, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    n = d.shape[0]
    R = float(np.dot(d, m) / m.sum())
    xbar = m.sum() / n
    infl = (d * m - R * m) / xbar
    s, G = _cluster_sums(infl, unit)
    return R, math.sqrt(G / (G - 1) * float(np.dot(s, s))) / n


def clause_table(labels, outs, refs, sex, unit):
    """12 clauses (task x U0/Ucal x metric) and the two within-SEX occupation Brier guards vs U0, with clustered SEs."""
    res = {}
    own = {t: losses(labels[t], outs[t], NCLASS[t]) for t in TASKS}
    for t in TASKS:
        for r in ('U0', 'Ucal'):
            ref = losses(labels[t], refs[r][t], NCLASS[t])
            for metric, d in (('accuracy_loss', ref['acc'] - own[t]['acc']), ('nll_excess', own[t]['nll'] - ref['nll']),
                              ('brier_excess', own[t]['brier'] - ref['brier'])):
                res[f'{t}|{r}|{metric}'] = {'value': float(d.mean()), 'se': clustered_se(d, unit), 'cap': CAP[metric]}
    u0 = losses(labels['occupation'], refs['U0']['occupation'], 5)
    d = own['occupation']['brier'] - u0['brier']
    for g in (0, 1):
        v, se = subgroup_mean_se(d, sex == g, unit)
        res[f'guard|sex{g}'] = {'value': v, 'se': se, 'cap': GUARD_CAP}
    return res


# ------------------------------------------------------------------------------------------------------- reserve (own code)
def load_rules():
    return json.loads((RES / 'roles' / 'statistical_reviewer' / 'RULES.json').read_text())


def reserve_seed(table, seed, rules, households):
    """Per condition: margin = cap - value; se_used = max(own SE, H_LJ floor x sqrt(5306 / households)); met iff margin >= k se_used."""
    k = float(rules['reserve']['multiplier_k'])
    floors = rules['reserve']['floor']['se'][str(seed)]
    scale = math.sqrt(rules['reserve']['floor']['households'] / households)
    out = {}
    for cond in rules['reserve']['conditions_per_seed']:
        c = table[cond]
        floor = floors[cond] * scale
        own = c['se']
        if not (math.isfinite(floor) and floor > 0):
            raise ValueError('floor')
        margin = c['cap'] - c['value']
        finite = math.isfinite(c['value']) and math.isfinite(own)
        se_used = max(own, floor) if finite else float('nan')
        ratio = margin / se_used if finite else float('-inf')
        out[cond] = {'value': c['value'], 'own_se': own, 'floor_se': floor, 'se_used': se_used, 'margin': margin, 'ratio': ratio,
                     'met': bool(finite and margin >= k * se_used), 'point_pass': bool(finite and c['value'] <= c['cap']),
                     'se_source': ('own' if own >= floor else 'floor') if finite else 'nonfinite'}
    return out


def leximin_key(ratios):
    return tuple(sorted(ratios))


def reserve_status(per_seed_met):
    """per_seed_met: {seed: bool} for screened seeds."""
    if len(per_seed_met) == 2:
        n = sum(per_seed_met.values())
        return 'RESERVE_MET' if n == 2 else 'RESERVE_ONE_SEED_ONLY' if n == 1 else 'RESERVE_NOT_MET'
    if len(per_seed_met) == 1:
        return 'RESERVE_PENDING_SECOND_SEED' if list(per_seed_met.values())[0] else 'RESERVE_NOT_MET'
    return 'NOT_SCREENED'


OCC_SIDE = ('occupation|U0|accuracy_loss', 'occupation|U0|nll_excess', 'occupation|U0|brier_excess', 'occupation|Ucal|accuracy_loss',
            'occupation|Ucal|nll_excess', 'occupation|Ucal|brier_excess', 'guard|sex0', 'guard|sex1')


def witness_label(task_only, per_seed_reserve):
    """RULES.json witness eligibility, first applicable label (both seeds needed)."""
    if not task_only:
        return 'NOT_TASK_ONLY'
    if set(per_seed_reserve) != {0, 1}:
        return 'NOT_CONSIDERED_ONE_SEED_ONLY'
    allc = [c for s in (0, 1) for c in per_seed_reserve[s].items()]
    if any(not v['point_pass'] for k, v in allc if k in OCC_SIDE):
        return 'OCCUPATION_POINT_INELIGIBLE'
    if any(not v['point_pass'] for k, v in allc if k not in OCC_SIDE):
        return 'RELEASE_POINT_INELIGIBLE'
    if any(not v['met'] for k, v in allc if k in OCC_SIDE):
        return 'OCCUPATION_RESERVE_NOT_MET'
    return 'PROMISING_WITNESS'


# ------------------------------------------------------------------------------------------------------------- AUC (own code)
def auc_mw(y, s, w=None):
    """Weighted Mann-Whitney AUC (ties count 1/2) by sorting scores and accumulating negative weight below each tie block."""
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(s, kind='mergesort')
    ss, yy, ww = s[order], y[order], w[order]
    blk = np.r_[True, ss[1:] != ss[:-1]]
    bid = np.cumsum(blk) - 1
    nb = bid[-1] + 1
    wp = np.bincount(bid, weights=ww * (yy == 1), minlength=nb)
    wn = np.bincount(bid, weights=ww * (yy == 0), minlength=nb)
    below = np.cumsum(wn) - wn
    return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))


# ------------------------------------------------------------------------------------------------------------- records
def write_record(name, obj):
    """Latest record (records/<name>.json) PLUS an immutable history copy (records/history/<name>__<UTC>.json, written with
    mode 'x', never overwritten): every run's result stays on file, so a later change of a check's acceptance rule cannot hide
    the earlier result (history kept from 2026-10-18T07:24Z; earlier runs are reconstructed in ACCESS_CHECK_AMENDMENT.json)."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, sort_keys=True, default=_jdefault) + '\n'
    hist = RECORDS / 'history'
    hist.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    i = 0
    while True:
        hp = hist / f'{name}__{stamp}{"" if i == 0 else f"_{i}"}.json'
        try:
            with open(hp, 'x') as f:
                f.write(text)
            break
        except FileExistsError:
            i += 1
    p = RECORDS / f'{name}.json'
    p.write_text(text)
    return p


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def unit_id():
    return os.environ.get('OHN_UNIT_ID')
