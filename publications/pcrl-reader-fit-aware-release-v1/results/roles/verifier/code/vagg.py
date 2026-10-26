"""RFA1 verifier: aggregate screens and decision conditions, re-implemented from the frozen definitions (RULES.md sections 0, 2, 4,
6 and nrm/rules.py aggregate / benefit_screen / target_screen / recipient_screen / comparison / _pooled / _primary /
frontier_comparison, read by the verifier). Pure functions on per-cell inputs; no data access, no campaign import.

Entries: {'cells': {cell: value}, 'cell_se': {cell: se}[, 'cell_se_mc': {...}]} over the 8 registered cells S<v><R>k<k>.
Pair benefit b = comparator AUC - candidate AUC; recipient r = candidate AUC - comparator AUC; costs loss(CAND) - loss(point).
"""
from __future__ import annotations

import math
import re

import numpy as np

Z = 2.0
SE_FLOOR = 0.001
TARGET = 0.020
RECIPIENT_LIMIT = 0.010
VERSIONS = (0, 1)
REGISTERED = tuple(f'S{v}{r}k{k}' for v in VERSIONS for r in ('R1', 'R2') for k in (0, 1))
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
BENCHMARK = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
TARGET_REFS = ('NEURAL_RAW', 'H_LJ')
_CELL = re.compile(r'^S([01])(R[1-4])k([0-9])$')


def auc_fast(y, s):
    """Mann-Whitney AUC, average ranks for ties (vectorized: unique values, cumulative counts)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def aggregate(entry, z=Z):
    vals, ses = (entry or {}).get('cells') or {}, (entry or {}).get('cell_se') or {}
    mcs = (entry or {}).get('cell_se_mc') or {}
    if not all(c in vals and c in ses for c in REGISTERED):
        return {'complete': False}
    blocks = {}
    for c in REGISTERED:
        m = _CELL.match(c)
        v, r, k = int(m.group(1)), m.group(2), int(m.group(3))
        blocks.setdefault((v, r), {})[k] = (float(vals[c]), float(ses[c]), float(mcs.get(c) or 0.0))

    def summ(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b].values()) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        hh = math.sqrt(sum(t[1] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b].values()))
        mc = math.sqrt(sum(t[2] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b].values()))
        rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        return {'mean': x, 'se_hh': hh, 'se_rs': rs, 'se_mc': mc, 'raw': math.sqrt(hh * hh + rs * rs + mc * mc),
                'block_means': {f'S{b[0]}{b[1]}': m for b, m in sorted(mb.items())}}
    keys = sorted(blocks)
    tot = summ(keys)
    per = {v: summ([b for b in keys if b[0] == v]) for v in VERSIONS}
    alias = all(t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0 for b in blocks.values() for t in b.values())
    for s in [tot] + list(per.values()):
        s['se_agg'] = 0.0 if alias else max(s['raw'], SE_FLOOR)
        s['lower'], s['upper'] = s['mean'] - z * s['se_agg'], s['mean'] + z * s['se_agg']
    return {'complete': True, 'alias': alias, **tot, 'per_version': per}


def benefit_screen(entry, threshold=0.0):
    a = aggregate(entry)
    names = ('AGG_POSITIVE', 'AGG_POSITIVE_POINT_ONLY', 'AGG_NOT_POSITIVE') if threshold == 0 else \
        ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY', 'TARGET_AGG_NOT_MET')
    if not a['complete']:
        return {'label': 'INCOMPLETE', 'resolved': False, 'aggregate': a}
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif a['lower'] > threshold and all(a['per_version'][v]['mean'] > threshold for v in VERSIONS):
        lab = names[0]
    elif a['mean'] > threshold:
        lab = names[1]
    else:
        lab = names[2]
    return {'label': lab, 'resolved': lab == names[0], 'aggregate': a}


def target_screen(entry):
    r = benefit_screen(entry, TARGET)
    if r['label'] in ('INCOMPLETE', 'ALIAS_ZERO'):
        r['label'] = 'TARGET_AGG_NOT_MET'
    return r


def recipient_screen(entry, limit=RECIPIENT_LIMIT):
    a = aggregate(entry)
    if not a['complete']:
        return {'label': 'INCOMPLETE', 'gate_pass': False, 'aggregate': a}
    pv = a['per_version']
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif any(pv[v]['lower'] > limit for v in VERSIONS):
        lab = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif not all(pv[v]['mean'] <= limit for v in VERSIONS):
        lab = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(pv[v]['upper'] <= limit for v in VERSIONS):
        lab = 'RECIPIENT_AGG_RESOLVED'
    else:
        lab = 'RECIPIENT_AGG_EDGE'
    return {'label': lab, 'gate_pass': lab in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO'), 'aggregate': a}


def comparison(entry3):
    b = benefit_screen((entry3 or {}).get('pair'))
    rc = {v: recipient_screen((entry3 or {}).get(v)) for v in ('income', 'occupation')}
    return {'benefit': b, 'recipient': rc, 'benefit_positive': b['resolved'], 'recipient_gate_pass': all(x['gate_pass'] for x in rc.values()),
            'passes': b['resolved'] and all(x['gate_pass'] for x in rc.values())}


def pooled(per):
    """{v: {'value', 'se'[, 'se_mc']}} -> value = mean over versions, se = mean of version SEs (sqrt(se^2 + se_mc^2))."""
    vals = [float(per[v]['value']) for v in VERSIONS]
    ses = [math.sqrt(float(per[v]['se']) ** 2 + float(per[v].get('se_mc') or 0.0) ** 2) for v in VERSIONS]
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2, 'by_version': dict(zip(VERSIONS, vals))}


def frontier(costs_vs, path, grid_vs):
    """costs_vs[id][v] = paired costs of CAND vs id ('total|nll' and the 8 secondary metrics); path = [ids] (START first);
    Q* = the cheapest point spending at least as much as CAND (d = pooled total-NLL cost(CAND) - cost(point) <= 0, largest d first).
    Label DOMINATES_CONTROL_FRONTIER / SEPARATED_ON_PRIMARY_COST_ONLY / NOT_SEPARATED_FROM_TASK_SPENDING (+ reason); the RFA1
    conjuncts c2 (recipient gate vs Q*) and c3 (no cheaper point resolved LESS recoverable than CAND) are returned beside."""
    prim = {q: pooled({v: costs_vs[q][v]['total|nll'] for v in VERSIONS}) for q in path}
    above = sorted([q for q in path if prim[q]['value'] <= 0], key=lambda q: -prim[q]['value'])
    out = {'primary': {q: prim[q]['value'] for q in path}, 'path': list(path)}
    if not above:
        out.update({'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'NO_BRACKET', 'bracket': None})
        return out
    qs = above[0]
    ben = benefit_screen((grid_vs.get(qs) or {}).get('pair'))
    sec = {m: pooled({v: costs_vs[qs][v][m] for v in VERSIONS}) for m in SECONDARY}
    more = sorted(m for m, x in sec.items() if x['value'] - Z * x['se'] > 0)
    rec_q = comparison(grid_vs.get(qs))
    cheaper = [q for q in path if prim[q]['value'] > 0]
    c3_fail = []
    for q in cheaper:
        a = benefit_screen((grid_vs.get(q) or {}).get('pair'))['aggregate']
        if a.get('complete') and not a['alias'] and a['upper'] < 0 and all(a['per_version'][v]['mean'] < 0 for v in VERSIONS):
            c3_fail.append(q)
    lab = ('NOT_SEPARATED_FROM_TASK_SPENDING' if not ben['resolved'] else
           'SEPARATED_ON_PRIMARY_COST_ONLY' if more else 'DOMINATES_CONTROL_FRONTIER')
    out.update({'label': lab, 'bracket': qs, 'benefit_label': ben['label'], 'benefit_mean': ben['aggregate'].get('mean'),
                'benefit_se_agg': ben['aggregate'].get('se_agg'),
                'benefit_version_means': {v: ben['aggregate']['per_version'][v]['mean'] for v in VERSIONS} if ben['aggregate'].get('complete') else None,
                'secondary_more_spending': more, 'c2_recipient_gate_vs_Q': rec_q['recipient_gate_pass'],
                'c3_cheaper_points_resolved_less_recoverable': c3_fail,
                'passes': lab == 'DOMINATES_CONTROL_FRONTIER' and rec_q['recipient_gate_pass'] and not c3_fail})
    return out


def point_positive(entry):
    """Reader targeting (RULES 4.1 d): the non-table winner's point pair benefit > 0 in both version means."""
    cells = (entry or {}).get('cells') or {}
    if not all(c in cells for c in REGISTERED):
        return None
    vm = {v: float(np.mean([cells[c] for c in REGISTERED if c.startswith(f'S{v}')])) for v in VERSIONS}
    return {'version_means': vm, 'pass': all(x > 0 for x in vm.values())}
