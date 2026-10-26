"""RFA1 verifier: aggregate screens over an arbitrary cell set (registered / robustness / all), re-implemented from nrm/rules.py
aggregate / _select_cells / benefit_screen / recipient_screen / robustness (read by the verifier), plus the D2R per-comparator
reading of rfa/rules.py d2r (RULES.md 8). Pure functions; no data access, no campaign import."""
from __future__ import annotations

import math
import re

Z, SE_FLOOR, LIMIT = 2.0, 0.001, 0.010
VERSIONS = (0, 1)
REGISTERED = tuple(f'S{v}{r}k{k}' for v in VERSIONS for r in ('R1', 'R2') for k in (0, 1))
_CELL = re.compile(r'^S([01])(R[1-4])k([0-9])$')


def select(entry, which):
    vals, ses = (entry or {}).get('cells'), (entry or {}).get('cell_se')
    if not isinstance(vals, dict) or not isinstance(ses, dict):
        return None
    names = [c for c in vals if _CELL.match(c)]
    if which == 'registered':
        names = [c for c in names if c in REGISTERED]
        if sorted(names) != sorted(REGISTERED):
            return None
    elif which == 'robustness':
        names = [c for c in names if c not in REGISTERED]
    if not names:
        return None
    mcs = (entry or {}).get('cell_se_mc') or {}
    return {c: (float(vals[c]), float(ses[c]), float(mcs.get(c) or 0.0)) for c in names}


def aggregate(entry, which='registered', z=Z):
    sel = select(entry, which)
    if sel is None:
        return {'complete': False}
    blocks = {}
    for c, t in sel.items():
        m = _CELL.match(c)
        blocks.setdefault((int(m.group(1)), m.group(2)), {})[int(m.group(3))] = t
    if sorted({v for v, _ in blocks}) != list(VERSIONS):
        return {'complete': False}

    def summ(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b].values()) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        hh = math.sqrt(sum(t[1] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b].values()))
        mc = math.sqrt(sum(t[2] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b].values()))
        rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        return {'mean': x, 'raw': math.sqrt(hh * hh + rs * rs + mc * mc)}
    keys = sorted(blocks)
    tot = summ(keys)
    per = {v: summ([b for b in keys if b[0] == v]) for v in VERSIONS}
    alias = all(t == (0.0, 0.0, 0.0) for b in blocks.values() for t in b.values())
    for s in [tot] + list(per.values()):
        s['se_agg'] = 0.0 if alias else max(s['raw'], SE_FLOOR)
        s['lower'], s['upper'] = s['mean'] - z * s['se_agg'], s['mean'] + z * s['se_agg']
    return {'complete': True, 'alias': alias, **tot, 'per_version': per, 'n_cells': len(sel)}


def benefit(entry, which='registered', threshold=0.0):
    a = aggregate(entry, which)
    if not a['complete']:
        return {'label': 'INCOMPLETE', 'resolved': False, 'aggregate': a}
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif a['lower'] > threshold and all(a['per_version'][v]['mean'] > threshold for v in VERSIONS):
        lab = 'AGG_POSITIVE'
    elif a['mean'] > threshold:
        lab = 'AGG_POSITIVE_POINT_ONLY'
    else:
        lab = 'AGG_NOT_POSITIVE'
    return {'label': lab, 'resolved': lab == 'AGG_POSITIVE', 'aggregate': a}


def recipient(entry, which='registered', limit=LIMIT):
    a = aggregate(entry, which)
    if not a['complete']:
        return {'label': 'INCOMPLETE', 'gate_pass': False}
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
    return {'label': lab, 'gate_pass': lab in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO')}


def d2r_comparator(entry3, refit_effective):
    """rfa.rules.d2r per comparator: ROBUSTNESS_CONFIRMED iff AGG_POSITIVE on the robustness cells and on all cells, both recipient
    gates on the robustness cells, the registered cells present and the refit genuinely effective; sign change of the robustness mean
    against the registered mean; a failed robustness recipient gate."""
    rob = benefit(entry3.get('pair'), 'robustness')
    if rob['label'] == 'INCOMPLETE':
        return {'label': 'NOT_RUN'}
    allc = benefit(entry3.get('pair'), 'all')
    reg = aggregate(entry3.get('pair'), 'registered')
    rg = {w: recipient(entry3.get(w), 'robustness') for w in ('income', 'occupation')}
    rg_ok = all(x['gate_pass'] for x in rg.values())
    rg_missing = any(x['label'] == 'INCOMPLETE' for x in rg.values())
    sgn = lambda x: (x > 0) - (x < 0)
    sign_change = reg.get('complete') and sgn(reg['mean']) != 0 and sgn(rob['aggregate']['mean']) != 0 and sgn(reg['mean']) != sgn(rob['aggregate']['mean'])
    reasons = ([] if (rob['resolved'] and allc['resolved']) else ['NOT_AGG_POSITIVE_ON_ROBUSTNESS_OR_ALL_CELLS']) + \
              ([] if refit_effective else ['REFIT_NOT_GENUINELY_EFFECTIVE']) + ([] if reg.get('complete') else ['REGISTERED_CELLS_ABSENT']) + \
              (['RECIPIENT_ROBUSTNESS_NOT_RECORDED'] if rg_missing else ([] if rg_ok else ['RECIPIENT_GATE_FAILED_ON_ROBUSTNESS']))
    return {'label': 'ROBUSTNESS_CONFIRMED' if not reasons else 'ROBUSTNESS_NOT_CONFIRMED', 'reasons': reasons,
            'robustness_mean': rob['aggregate']['mean'], 'robustness_se_agg': rob['aggregate']['se_agg'], 'robustness_label': rob['label'],
            'all_cells_label': allc['label'], 'all_cells_mean': allc['aggregate'].get('mean'), 'registered_mean': reg.get('mean'),
            'sign_change': bool(sign_change), 'recipient_failed': (not rg_ok and not rg_missing),
            'recipient_labels': {w: x['label'] for w, x in rg.items()}}
