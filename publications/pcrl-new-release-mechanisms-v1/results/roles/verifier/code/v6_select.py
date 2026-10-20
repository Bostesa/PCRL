"""V6 candidate / control selection replay: the statistical reviewer's frozen generic claims and selection rules (RULES.md /
RULES.json, NRM1-rules-v1) re-implemented from their TEXT with the verifier's own code (no import of nrm.rules, ohn.rules or
tsz.rules), run on the RECORDED inputs of the coordinator's selection, and compared with the recorded outcome.

Implemented (RULES.md sections): aggregate estimator and its SE components (5.1-5.2), pair-benefit / 0.020 / per-version recipient
labels (5.3), old every-cell diagnostics (5.4), point eligibility and reserve with the MC term (4, 9), frontier bracketing (7),
claims G1-G7, tiers and the useful-component label (1), selection with aliases, output mismatches, near ties and tie-breaks (8).

    python3 .../v6_select.py --tag V1 --input SELECTION_INPUT.json --record SELECTION.json      (no label read)
    python3 .../v6_select.py --selftest                                                          (invented known-answer cases)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

RULES = V.RESULTS / 'roles' / 'statistical_reviewer' / 'RULES.json'
OHN_RULES = V.ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles' / 'statistical_reviewer' / 'RULES.json'
VERS = (0, 1)
REG = tuple(f'S{v}{r}k{k}' for v in VERS for r in ('R1', 'R2') for k in (0, 1))
Z, FLOOR, LIMIT, TARGET, NEAR = 2.0, 0.001, 0.010, 0.020, 0.005
BENCH = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
TREFS = ('NEURAL_RAW', 'H_LJ')
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
TIERS = ('T2', 'T3', 'T4', 'T5')
NOVELTY = ('NOT_CLAIMED', 'NOVELTY_REVIEW_PENDING', 'NOVEL_PER_SEPARATE_REVIEW', 'PRIOR_ART_FOUND')


def fnum(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return math.nan
    return y


def finite(*xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def vget(d, v):
    if isinstance(d, dict):
        if v in d:
            return d[v]
        if str(v) in d:
            return d[str(v)]
    return None


def cell_parts(c):
    import re
    m = re.fullmatch(r'S([01])(R[1-4])k([0-9])', str(c))
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


# ---------------------------------------------------------------- 5.1-5.3 aggregate screen
def my_aggregate(entry, cells='registered'):
    vals, ses, mcs = (entry or {}).get('cells'), (entry or {}).get('cell_se'), (entry or {}).get('cell_se_mc') or {}
    if not isinstance(vals, dict) or not isinstance(ses, dict):
        return None
    names = [c for c in vals if cell_parts(c)]
    if cells == 'registered':
        if set(REG) - set(names):
            return None
        names = list(REG)
    elif cells == 'robustness':
        names = [c for c in names if c not in REG]
    if not names:
        return None
    cell = {}
    for c in names:
        x, s = fnum(vals[c]), fnum(ses.get(c))
        m = 0.0 if mcs.get(c) is None else fnum(mcs.get(c))
        if not finite(x, s, m) or s < 0 or m < 0:
            return None
        cell[c] = (x, s, m)
    blocks = {}
    for c, t in cell.items():
        v, r, _ = cell_parts(c)
        blocks.setdefault((v, r), []).append(t)
    if {v for v, _ in blocks} != set(VERS):
        return None

    def summ(keys):
        nb = len(keys)
        means = {b: sum(t[0] for t in blocks[b]) / len(blocks[b]) for b in keys}
        x = sum(means.values()) / nb
        hh = math.sqrt(sum(t[1] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b]))
        mc = math.sqrt(sum(t[2] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b]))
        rs = math.sqrt(sum((m - x) ** 2 for m in means.values()) / nb)
        return {'mean': x, 'hh': hh, 'rs': rs, 'mc': mc, 'raw': math.sqrt(hh ** 2 + rs ** 2 + mc ** 2)}
    keys = sorted(blocks)
    tot = summ(keys)
    per = {v: summ([b for b in keys if b[0] == v]) for v in VERS}
    alias = all(t == (0.0, 0.0, 0.0) for t in cell.values())
    for s in [tot] + list(per.values()):
        s['se'] = 0.0 if alias else max(s['raw'], FLOOR)
        s['lo'], s['hi'] = s['mean'] - Z * s['se'], s['mean'] + Z * s['se']
    return {'alias': alias, 'tot': tot, 'per': per, 'cells': {c: t[0] for c, t in cell.items()}}


def my_benefit(entry, thr=0.0, cells='registered'):
    a = my_aggregate(entry, cells)
    names = ('AGG_POSITIVE', 'AGG_POSITIVE_POINT_ONLY', 'AGG_NOT_POSITIVE') if thr == 0 else \
        ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY', 'TARGET_AGG_NOT_MET')
    if a is None:
        return {'label': 'INCOMPLETE', 'resolved': False, 'agg': None}
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif a['tot']['lo'] > thr and all(a['per'][v]['mean'] > thr for v in VERS):
        lab = names[0]
    elif a['tot']['mean'] > thr:
        lab = names[1]
    else:
        lab = names[2]
    return {'label': lab, 'resolved': lab == names[0], 'agg': a}


def my_target(entry):
    r = my_benefit(entry, TARGET)
    if r['label'] in ('INCOMPLETE', 'ALIAS_ZERO'):
        r['label'] = 'TARGET_AGG_NOT_MET'
    return r


def my_recipient(entry):
    a = my_aggregate(entry)
    if a is None:
        return {'label': 'INCOMPLETE', 'gate': False}
    p = a['per']
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif any(p[v]['lo'] > LIMIT for v in VERS):
        lab = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif not all(p[v]['mean'] <= LIMIT for v in VERS):
        lab = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(p[v]['hi'] <= LIMIT for v in VERS):
        lab = 'RECIPIENT_AGG_RESOLVED'
    else:
        lab = 'RECIPIENT_AGG_EDGE'
    return {'label': lab, 'gate': lab in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO')}


def my_comparison(g):
    b = my_benefit((g or {}).get('pair'))
    rc = {v: my_recipient((g or {}).get(v)) for v in ('income', 'occupation')}
    return {'benefit': b, 'recipient': rc, 'pos': b['resolved'], 'gate': all(x['gate'] for x in rc.values())}


# ---------------------------------------------------------------- 5.4 old every-cell rules (diagnostics)
def old_stable(entry, thr=0.0):
    vals, ses = (entry or {}).get('cells') or {}, (entry or {}).get('cell_se') or {}
    x = {c: fnum(vals.get(c)) for c in REG}
    s = {c: fnum(ses.get(c)) for c in REG}
    if not all(finite(x[c], s[c]) and s[c] >= 0 for c in REG):
        return None
    d = {v: sum(x[c] for c in REG if c.startswith(f'S{v}')) / 4 for v in VERS}
    dist = {v: [] for v in VERS}
    for v in VERS:
        for r in ('R1', 'R2'):
            a, b = x[f'S{v}{r}k0'], x[f'S{v}{r}k1']
            dist[v] += [a, b] if a != b else [a]
    n_eff = sum(len(t) for t in dist.values())
    vrep = sum((y - d[v]) ** 2 for v in VERS for y in dist[v]) / (n_eff - 2)
    sd = math.sqrt(vrep + sum(y ** 2 for y in s.values()) / 8)
    return all(y > thr for y in x.values()) and all(d[v] - Z * sd > thr for v in VERS)


def old_recipient(entry):
    vals = (entry or {}).get('cells') or {}
    x = [fnum(vals.get(c)) for c in REG]
    return all(finite(y) for y in x) and all(y <= LIMIT for y in x)


# ---------------------------------------------------------------- 4 / 9 point eligibility and reserve
def conds_of(u, v):
    u = u.get('utility', u) if isinstance(u, dict) else u
    out = {}
    for c in u['clauses']:
        if 'seed' in c and int(c['seed']) != v:
            raise ValueError('seed mismatch')
        n = f"{c['task']}|{c['reference']}|{c['metric']}"
        if fnum(c['cap']) != V.CAPS[c['metric']] or n in out:
            raise ValueError(f'bad clause {n}')
        out[n] = c
    for g in u['guards']:
        n = f"guard|{g['group']}"
        if fnum(g['cap']) != V.GUARD_CAP or n in out:
            raise ValueError(f'bad guard {n}')
        out[n] = g
    if set(out) != set(V.CONDITIONS):
        raise ValueError('condition set')
    return out


def my_point(evals):
    failed = []
    for v in VERS:
        e = vget(evals, v)
        if e is None:
            return 'INCOMPLETE', None
        try:
            cs = conds_of(e, v)
        except (KeyError, ValueError, TypeError):
            return 'INCOMPLETE', None
        if not all(finite(fnum(c['value'])) for c in cs.values()):
            return 'INCOMPLETE', None
        failed += [f'v{v}|{n}' for n, c in cs.items() if not fnum(c['value']) <= fnum(c['cap'])]
    return ('POINT_ELIGIBLE' if not failed else 'POINT_INELIGIBLE'), failed


def my_reserve(evals, with_mc=True):
    floor = json.loads(OHN_RULES.read_text())['reserve']['floor']
    ratios, per_ok, mins = [], {}, {}
    present = [v for v in VERS if vget(evals, v) is not None]
    for v in present:
        e = vget(evals, v)
        u = e.get('utility', e)
        cs = conds_of(u, v)
        G = int(u.get('households', V.FLOOR_HOUSEHOLDS))
        scale = math.sqrt(int(floor['households']) / G)
        ok, rs = True, []
        for n in V.CONDITIONS:
            c = cs[n]
            val, cap, own = fnum(c['value']), fnum(c['cap']), fnum(c.get('se'))
            if with_mc and finite(fnum(c.get('se_mc'))):
                own = math.sqrt(own ** 2 + fnum(c['se_mc']) ** 2)
            fl = fnum(floor['se'][str(v)][n]) * scale
            if not finite(val, own):
                rs.append(-math.inf); ok = False; continue
            used = max(own, fl)
            r = (cap - val) / used
            rs.append(r)
            ok = ok and (cap - val >= 2.0 * used)
        per_ok[v] = ok; mins[v] = min(rs); ratios += rs
    qualified = len(present) == 2 and all(per_ok.values())
    return {'qualified': qualified, 'ratio_vector': sorted(ratios), 'min_by_version': mins}


# ---------------------------------------------------------------- 7 frontier
def pooled(d):
    if not all(isinstance(vget(d, v), dict) for v in VERS):
        return None
    vals = [fnum(vget(d, v).get('value')) for v in VERS]
    ses = []
    for v in VERS:
        x = vget(d, v)
        mc = fnum(x.get('se_mc'))
        ses.append(math.sqrt(fnum(x.get('se')) ** 2 + (mc ** 2 if finite(mc) else 0.0)))
    if not all(finite(y) for y in vals + ses):
        return None
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2}


def primary(costs):
    per = {}
    for v in VERS:
        c = vget(costs, v)
        if not isinstance(c, dict):
            return None
        if 'total|nll' in c:
            per[v] = c['total|nll']
        elif 'income|nll' in c and 'occupation|nll' in c:
            a, b = c['income|nll'], c['occupation|nll']
            per[v] = {'value': fnum(a['value']) + fnum(b['value']), 'se': fnum(a['se']) + fnum(b['se']),
                      'se_mc': fnum(a.get('se_mc') or 0) + fnum(b.get('se_mc') or 0)}
        else:
            return None
    return pooled(per)


def my_frontier(point, path):
    if not path:
        return {'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'CONTROL_PATH_MISSING'}
    prim = {}
    for q in path:
        d = primary((point.get('costs_vs') or {}).get(q['id']))
        if d is None:
            return {'label': 'INCOMPLETE'}
        prim[q['id']] = d
    above = [q for q in path if prim[q['id']]['value'] <= 0]
    if not above:
        return {'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'}
    qs = max(above, key=lambda q: prim[q['id']]['value'])        # smallest pooled cost among those spending at least as much
    ben = my_benefit(((point.get('grid_vs') or {}).get(qs['id']) or {}).get('pair'))
    if ben['label'] == 'INCOMPLETE':
        return {'label': 'INCOMPLETE', 'bracket': qs['id']}
    more = []
    for m in SECONDARY:
        per = {v: (vget(point['costs_vs'][qs['id']], v) or {}).get(m) for v in VERS}
        p = pooled(per) if all(isinstance(x, dict) for x in per.values()) else None
        if p is None:
            return {'label': 'INCOMPLETE', 'bracket': qs['id']}
        if p['value'] - Z * p['se'] > 0:
            more.append(m)
    if not ben['resolved']:
        return {'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL', 'bracket': qs['id']}
    return {'label': 'SEPARATED_ON_PRIMARY_COST_ONLY' if more else 'DOMINATES_CONTROL_FRONTIER', 'bracket': qs['id'], 'more': sorted(more)}


# ---------------------------------------------------------------- 1 claims, tiers
def eligible_refs():
    pub = json.loads(RULES.read_text())['references']['published_labels']
    return tuple(r for r in BENCH if pub[r]['point'] == 'POINT_ELIGIBLE')


def my_claims(rec, frontier_override=None):
    gv = rec.get('grid_valid')
    pl, _ = my_point(rec.get('evals') or {})
    feas = pl == 'POINT_ELIGIBLE'
    g1 = 'TASK_FEASIBLE' if feas else ('INCOMPLETE' if pl == 'INCOMPLETE' else 'TASK_INFEASIBLE')
    if gv is None or not bool(gv):
        why = 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN' if gv is None else 'NOT_COMPUTED_GRID_INVALID'
        return {'G1': g1, **{g: why for g in ('G2', 'G3', 'G4', 'G5', 'G6')}, 'G7': 'NOT_CLAIMED', 'tier': None, 'score': None,
                'useful': None, 'reserve': {'qualified': False, 'ratio_vector': [-math.inf]}, 'comps': {}}
    try:
        res = my_reserve(rec.get('evals') or {})
    except (KeyError, ValueError, TypeError):
        res = {'qualified': False, 'ratio_vector': [-math.inf]}
    grid = rec.get('grid') or {}
    comps = {n: my_comparison(e) for n, e in grid.items()}

    def ok(n):
        return n in comps and comps[n]['pos'] and comps[n]['gate']
    m = rec.get('matched') or {}
    if not m.get('twin') or not m.get('alternative') or m['twin'] not in comps or m['alternative'] not in comps:
        g2 = 'NO_MATCHED_CONTROLS'
    elif ok(m['twin']) and ok(m['alternative']):
        g2 = 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS' if feas else 'MET_WITHOUT_TASK_FEASIBILITY'
    else:
        g2 = 'NOT_MET'
    fr = frontier_override if frontier_override is not None else (rec.get('frontier') or {})
    if fr.get('label') == 'DOMINATES_CONTROL_FRONTIER':
        g3 = 'STABLE_FRONTIER_IMPROVEMENT' if feas else 'MET_WITHOUT_TASK_FEASIBILITY'
    elif not fr:
        g3 = 'NOT_ASSESSED'
    else:
        g3 = fr.get('label', 'INCOMPLETE')
    refs = eligible_refs()
    missing = [r for r in refs if r not in comps]
    fail = [r for r in refs if r in comps and not ok(r)]
    g4 = 'INCOMPLETE' if missing else 'NOT_SUPERIOR' if fail else ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if feas else 'MET_WITHOUT_TASK_FEASIBILITY')
    tg = [my_target((grid.get(r) or {}).get('pair'))['label'] for r in TREFS]
    rgate = all(r in comps and comps[r]['gate'] for r in TREFS)
    if all(x == 'TARGET_AGG_RESOLVED' for x in tg):
        level = 'MATERIAL_0020_RESOLVED' if rgate else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in tg):
        level = 'MATERIAL_0020_POINT_ONLY'
    elif all(r in comps and comps[r]['pos'] for r in TREFS):
        level = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        level = 'NOT_MET'
    g5 = level if (feas or level == 'NOT_MET') else 'MET_WITHOUT_TASK_FEASIBILITY'
    j = rec.get('joint')
    if not j:
        g6 = 'NOT_APPLICABLE_NOT_JOINT'
    else:
        ch = set(j.get('outputs_changed') or []) & {'income', 'occupation'}
        if len(ch) < 2:
            g6 = 'NOT_APPLICABLE_ONE_OUTPUT_CHANGED'
        elif not j.get('jointly_designed'):
            g6 = 'NOT_APPLICABLE_NOT_JOINT'
        else:
            need = [j.get('separate'), j.get('income_first'), j.get('occupation_first')]
            if not all(need) or any(n not in comps for n in need):
                g6 = 'NOT_ASSESSED_MISSING_CONTROLS'
            else:
                cs = [primary((j.get('costs_vs') or {}).get(n)) for n in need]
                cheap = all(c is not None and not (c['value'] - Z * c['se'] > 0) for c in cs)
                g6 = (('JOINT_CONTRIBUTION_SUPPORTED' if feas else 'MET_WITHOUT_TASK_FEASIBILITY')
                      if all(ok(n) for n in need) and cheap else 'NOT_SUPPORTED')
    nov = rec.get('novelty') or 'NOT_CLAIMED'
    g7 = nov if nov in NOVELTY else 'NOT_CLAIMED'
    useful = 'USEFUL_COMPONENT_NOT_COMPETITIVE' if g2 == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS' and \
        g4 != 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' else None
    tier = None
    if feas and g2 == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS':
        tier = 'T2'
        if g3 == 'STABLE_FRONTIER_IMPROVEMENT':
            tier = 'T3'
            if g4 == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES':
                tier = 'T4'
                if g5 == 'MATERIAL_0020_RESOLVED':
                    tier = 'T5'
    scores = [comps[r]['benefit']['agg']['tot']['mean'] if comps[r]['benefit']['agg'] else math.nan for r in refs if r in comps]
    score = min(scores) if scores and not missing and all(finite(s) for s in scores) else None
    return {'G1': g1, 'G2': g2, 'G3': g3, 'G4': g4, 'G5': g5, 'G6': g6, 'G7': g7, 'tier': tier, 'score': score, 'useful': useful,
            'reserve': res, 'comps': comps, 'g4_failing': fail, 'g4_missing': missing}


# ---------------------------------------------------------------- 8 selection
def wires_eq(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and all(vget(a, v) is not None and vget(a, v) == vget(b, v) for v in VERS)


def components(x):
    if isinstance(x, dict):
        vals = [fnum(vget(x, v)) if vget(x, v) is not None else math.inf for v in VERS]
        return sum(vals) if not any(math.isnan(y) for y in vals) else math.inf
    y = fnum(x)
    return y if finite(y) else math.inf


def entry_no(x):
    y = fnum(x)
    return y if finite(y) else math.inf


def leximin_key(vec):
    """Ascending ratio vector; larger elementwise first. Encoded for sorting: negate (1e-9 ties handled in the comparator)."""
    return vec


def better(a, b):
    """True iff a precedes b under the tie-breaks of RULES.md section 8."""
    if a['qual'] != b['qual']:
        return a['qual']
    for x, y in zip(a['ratios'], b['ratios']):
        if x == y or (finite(x, y) and abs(x - y) <= 1e-9):
            continue
        return x > y
    if a['comp'] != b['comp']:
        return a['comp'] < b['comp']
    if a['entry'] != b['entry']:
        return a['entry'] < b['entry']
    return str(a['config_id']) < str(b['config_id'])


def my_select(ledger, ref_wires):
    if not isinstance(ref_wires, dict) or any(r not in ref_wires for r in BENCH):
        raise ValueError('reference wires of all nine benchmark references are required')
    out, seen = [], {}
    for c in sorted(ledger, key=lambda x: (entry_no(x.get('entry')), str(x.get('config_id')))):
        r = {'config_id': c.get('config_id'), 'kind': c.get('kind'), 'entry': entry_no(c.get('entry')),
             'comp': components(c.get('n_fitted_components'))}
        we, wg = c.get('wire_sha256_eval'), c.get('wire_sha256_grid')
        if not wires_eq(we, wg):
            r['status'] = 'NOT_SELECTABLE_OUTPUT_MISMATCH'; out.append(r); continue
        ref = next((n for n, w in ref_wires.items() if wires_eq(we, w)), None)
        if ref:
            r['status'] = f'ALIAS_OF_{ref}'; out.append(r); continue
        key = json.dumps({str(v): vget(we, v) for v in VERS}, sort_keys=True)
        if key in seen:
            r['status'] = f'ALIAS_OF_{seen[key]}'; out.append(r); continue
        seen[key] = c.get('config_id')
        cl = my_claims(c)
        zero = [n for n in BENCH if n in cl['comps'] and cl['comps'][n]['benefit']['label'] == 'ALIAS_ZERO']
        if zero:
            r['status'] = f'ALIAS_OF_{zero[0]}'; out.append(r); continue
        r.update({'claims': {g: cl[g] for g in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7')}, 'tier': cl['tier'], 'score': cl['score'],
                  'useful': cl['useful'], 'qual': bool(cl['reserve']['qualified']), 'ratios': cl['reserve']['ratio_vector'] or [-math.inf],
                  'status': 'BELOW_MINIMUM_TIER' if not cl['tier'] else ('SELECTABLE' if cl['score'] is not None
                                                                          else 'NOT_SELECTABLE_INCOMPLETE_REFERENCE_GRID')})
        out.append(r)
    sel = [r for r in out if r['status'] == 'SELECTABLE']
    if not sel:
        return {'outcome': 'NO_DEVELOPMENT_CANDIDATE', 'selected': None, 'records': out, 'near': []}
    top = max(TIERS.index(r['tier']) for r in sel)
    tier = [r for r in sel if TIERS.index(r['tier']) == top]
    best = max(r['score'] for r in tier)
    near = [r for r in tier if r['score'] >= best - NEAR]
    choice = near[0]
    for r in near[1:]:
        if better(r, choice):
            choice = r
    for r in sel:
        r['status'] = 'SELECTED_DEVELOPMENT_CANDIDATE' if r is choice else 'SELECTABLE_NOT_CHOSEN'
    return {'outcome': 'DEVELOPMENT_CANDIDATE_SELECTED', 'selected': choice['config_id'], 'tier': choice['tier'], 'best': best,
            'near': [r['config_id'] for r in near], 'records': out,
            'control_designated': choice.get('kind') == 'control'}


# ---------------------------------------------------------------- comparison with the recorded selection
def compare_selection(mine, rec):
    mm = []
    if mine['outcome'] != rec.get('outcome'):
        mm.append({'what': 'outcome', 'mine': mine['outcome'], 'recorded': rec.get('outcome')})
    if mine.get('selected') != rec.get('selected'):
        mm.append({'what': 'selected', 'mine': mine.get('selected'), 'recorded': rec.get('selected')})
    if sorted(mine.get('near') or []) != sorted(rec.get('near_tie_set') or []):
        mm.append({'what': 'near-tie set', 'mine': mine.get('near'), 'recorded': rec.get('near_tie_set')})
    their = {r['config_id']: r for r in rec.get('records', [])}
    for r in mine['records']:
        t = their.get(r['config_id'])
        if t is None:
            mm.append({'what': f"{r['config_id']} missing in the record"}); continue
        if r['status'] != t.get('status'):
            mm.append({'what': f"{r['config_id']} status", 'mine': r['status'], 'recorded': t.get('status')})
        if 'claims' in r:
            for g, lab in r['claims'].items():
                if (t.get('claims') or {}).get(g) != lab:
                    mm.append({'what': f"{r['config_id']} {g}", 'mine': lab, 'recorded': (t.get('claims') or {}).get(g)})
            if r['tier'] != t.get('tier'):
                mm.append({'what': f"{r['config_id']} tier", 'mine': r['tier'], 'recorded': t.get('tier')})
            a, b = r['score'], t.get('score')
            if (a is None) != (b is None) or (a is not None and abs(a - fnum(b)) > 1e-12):
                mm.append({'what': f"{r['config_id']} score", 'mine': a, 'recorded': b})
            if r['qual'] != bool(t.get('reserve_qualified')):
                mm.append({'what': f"{r['config_id']} reserve qualified", 'mine': r['qual'], 'recorded': t.get('reserve_qualified')})
            if r['useful'] != t.get('useful_component'):
                mm.append({'what': f"{r['config_id']} useful component", 'mine': r['useful'], 'recorded': t.get('useful_component')})
    extra = sorted(set(their) - {r['config_id'] for r in mine['records']})
    if extra:
        mm.append({'what': 'records only in the coordinator output', 'ids': extra})
    return mm


# ---------------------------------------------------------------- invented known-answer selftest
def _entry_const(xs, s=0.002, mc=None):
    e = {'cells': {c: x for c, x in zip(REG, xs)}, 'cell_se': {c: s for c in REG}}
    if mc is not None:
        e['cell_se_mc'] = {c: mc for c in REG}
    return e


def selftest():
    ok = {}
    a = my_aggregate(_entry_const([0.01] * 8))
    ok['agg_constant_mean'] = abs(a['tot']['mean'] - 0.01) < 1e-15 and abs(a['tot']['hh'] - 0.002) < 1e-15 and a['tot']['rs'] == 0.0
    ok['agg_positive'] = my_benefit(_entry_const([0.01] * 8))['label'] == 'AGG_POSITIVE'
    ok['agg_point_only'] = my_benefit(_entry_const([0.003] * 8))['label'] == 'AGG_POSITIVE_POINT_ONLY'
    ok['agg_version_disagrees'] = my_benefit(_entry_const([0.03] * 4 + [-0.001] * 4, s=0.0001))['label'] == 'AGG_POSITIVE_POINT_ONLY'
    ok['alias_zero'] = my_benefit(_entry_const([0.0] * 8, s=0.0))['label'] == 'ALIAS_ZERO'
    ok['floor_binds'] = my_aggregate(_entry_const([0.0015] * 8, s=0.0))['tot']['se'] == FLOOR and \
        my_benefit(_entry_const([0.0015] * 8, s=0.0))['label'] == 'AGG_POSITIVE_POINT_ONLY'
    ok['incomplete'] = my_benefit({'cells': {c: 0.01 for c in REG[:7]}, 'cell_se': {c: 0.001 for c in REG[:7]}})['label'] == 'INCOMPLETE'
    # se_rs: block means 0.01, 0.03 (v0), 0.01, 0.03 (v1) -> x 0.02, se_rs 0.01
    e = _entry_const([0.01, 0.01, 0.03, 0.03, 0.01, 0.01, 0.03, 0.03], s=0.0)
    ok['se_rs'] = abs(my_aggregate(e)['tot']['rs'] - 0.01) < 1e-15
    ok['target_resolved'] = my_target(_entry_const([0.05] * 8))['label'] == 'TARGET_AGG_RESOLVED'
    ok['target_alias_not_met'] = my_target(_entry_const([0.0] * 8, s=0.0))['label'] == 'TARGET_AGG_NOT_MET'
    ok['recipient_resolved'] = my_recipient(_entry_const([0.0] * 8, s=0.001))['label'] == 'RECIPIENT_AGG_RESOLVED'
    ok['recipient_edge'] = my_recipient(_entry_const([0.008] * 8, s=0.002))['label'] == 'RECIPIENT_AGG_EDGE'
    ok['recipient_point_failed'] = my_recipient(_entry_const([0.011] * 8, s=0.002))['label'] == 'RECIPIENT_AGG_POINT_FAILED'
    ok['recipient_worse'] = my_recipient(_entry_const([0.03] * 8, s=0.001))['label'] == 'RECIPIENT_AGG_RESOLVED_WORSE'
    ok['old_stable'] = old_stable(_entry_const([0.01] * 8)) is True and old_stable(_entry_const([0.01] * 7 + [-0.001])) is False
    # MC component enters
    ok['mc_component'] = abs(my_aggregate(_entry_const([0.01] * 8, s=0.0, mc=0.003))['tot']['mc'] - 0.003) < 1e-15
    # claims and selection on invented records
    def evals(val=0.0, se=0.001):
        def u(v):
            return {'clauses': [{'seed': v, 'task': t, 'reference': r, 'metric': m, 'value': val, 'cap': V.CAPS[m], 'se': se}
                                for t in V.TASKS for r in V.REFS for m in V.METRICS],
                    'guards': [{'seed': v, 'group': g, 'value': val, 'cap': 0.005, 'se': se} for g in ('sex0', 'sex1')],
                    'households': 5306}
        return {v: {'utility': u(v)} for v in VERS}
    good = {'pair': _entry_const([0.01] * 8), 'income': _entry_const([0.0] * 8, s=0.001), 'occupation': _entry_const([0.0] * 8, s=0.001)}
    weak = {'pair': _entry_const([0.004] * 8), 'income': good['income'], 'occupation': good['occupation']}
    wires = lambda tag: {v: {'income': f'i{tag}', 'occupation': f'o{tag}'} for v in VERS}  # noqa: E731
    base = {'kind': 'candidate', 'grid_valid': True, 'matched': {'twin': 'TW', 'alternative': 'ALT'},
            'frontier': {'label': 'DOMINATES_CONTROL_FRONTIER'}, 'n_fitted_components': 2}
    A = {**base, 'config_id': 'A', 'entry': 1, 'evals': evals(), 'grid': {**{r: good for r in BENCH}, 'TW': good, 'ALT': good},
         'wire_sha256_eval': wires('A'), 'wire_sha256_grid': wires('A')}
    B = {**base, 'config_id': 'B', 'entry': 2, 'evals': evals(), 'grid': {**{r: good for r in BENCH}, 'TW': good, 'ALT': good},
         'wire_sha256_eval': wires('B'), 'wire_sha256_grid': wires('B'), 'n_fitted_components': 1}
    Cm = {**A, 'config_id': 'C', 'entry': 3, 'wire_sha256_eval': wires('C'), 'wire_sha256_grid': wires('Cx')}
    D = {**A, 'config_id': 'D', 'entry': 4, 'wire_sha256_eval': wires('A'), 'wire_sha256_grid': wires('A')}
    E = {**A, 'config_id': 'E', 'entry': 5, 'evals': evals(val=0.02), 'wire_sha256_eval': wires('E'), 'wire_sha256_grid': wires('E')}
    F = {**A, 'config_id': 'F', 'entry': 6, 'grid': {**{r: weak for r in BENCH}, 'TW': weak, 'ALT': weak},
         'wire_sha256_eval': wires('F'), 'wire_sha256_grid': wires('F')}
    refw = {r: wires(f'ref{r}') for r in BENCH}
    s = my_select([A, B, Cm, D, E, F], refw)
    st = {r['config_id']: r['status'] for r in s['records']}
    ok['select_tie_break_fewer_components'] = s['selected'] == 'B' and sorted(s['near']) == ['A', 'B']
    ok['select_output_mismatch'] = st['C'] == 'NOT_SELECTABLE_OUTPUT_MISMATCH'
    ok['select_alias_earlier'] = st['D'] == 'ALIAS_OF_A'
    ok['select_infeasible_below_tier'] = st['E'] == 'BELOW_MINIMUM_TIER'
    ok['select_weak_below_tier'] = st['F'] == 'BELOW_MINIMUM_TIER'
    strong = {**good, 'pair': _entry_const([0.05] * 8)}
    ok['claims_tiers'] = my_claims(A)['tier'] == 'T4' and my_claims(A)['G5'] == 'POSITIVE_BELOW_MATERIAL_TARGET' and \
        my_claims({**A, 'grid': {**A['grid'], 'NEURAL_RAW': strong, 'H_LJ': strong}})['tier'] == 'T5' and \
        my_claims({**A, 'grid': {**A['grid'], 'NEURAL_RAW': strong, 'H_LJ': strong}})['G5'] == 'MATERIAL_0020_RESOLVED' and \
        my_claims({**A, 'frontier': {'label': 'NOT_SEPARATED_FROM_TASK_SPENDING'}})['tier'] == 'T2'
    ok['claims_grid_unknown'] = my_claims({**A, 'grid_valid': None})['G2'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    ok['claims_useful_component'] = my_claims({**A, 'grid': {**A['grid'], 'W': weak}})['useful'] == 'USEFUL_COMPONENT_NOT_COMPETITIVE'
    try:
        my_select([A], {}); ok['select_needs_reference_wires'] = False
    except ValueError:
        ok['select_needs_reference_wires'] = True
    alias_ref = {**A, 'config_id': 'G', 'entry': 7, 'wire_sha256_eval': refw['W'], 'wire_sha256_grid': refw['W']}
    ok['select_alias_reference'] = my_select([alias_ref], refw)['records'][0]['status'] == 'ALIAS_OF_W'
    # frontier: bracket = cheapest control with cost >= point; benefit resolved; secondary spending flags
    costs = lambda d: {v: {**{'total|nll': {'value': d, 'se': 0.001}}, **{m: {'value': 0.0, 'se': 0.001} for m in SECONDARY}} for v in VERS}  # noqa: E731
    point = {'costs_vs': {'Q1': costs(0.01), 'Q2': costs(-0.002), 'Q3': costs(-0.02)},
             'grid_vs': {'Q1': {'pair': _entry_const([0.0] * 8)}, 'Q2': {'pair': _entry_const([0.01] * 8)}, 'Q3': {'pair': _entry_const([0.0] * 8)}}}
    fr = my_frontier(point, [{'id': 'Q1'}, {'id': 'Q2'}, {'id': 'Q3'}])
    ok['frontier_bracket_dominates'] = fr['label'] == 'DOMINATES_CONTROL_FRONTIER' and fr['bracket'] == 'Q2'
    ok['frontier_spends_more'] = my_frontier({'costs_vs': {'Q1': costs(0.01)}}, [{'id': 'Q1'}])['reason'] == \
        'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag'); ap.add_argument('--input'); ap.add_argument('--record'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        chk = {'id': f'V6.SELECT_SELFTEST__{a.tag or "V1"}', 'what': 'own re-implementation of the NRM1 aggregate screen, recipient gate, '
               'old every-cell rule, claims, tiers, frontier bracketing and selection on invented known-answer cases',
               'inputs': V.inputs_sha([Path(__file__), RULES]), 'result': ok,
               'numerical_status': 'PASS' if ok['all_ok'] else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
        print(json.dumps(V.append_checks([chk]), indent=1)); print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    inp = json.loads(Path(a.input).read_text())
    rec = json.loads(Path(a.record).read_text())
    ledger, refw = inp['ledger'], inp['reference_wires']
    mine = my_select(ledger, refw)
    mm = compare_selection(mine, rec)
    frs = {}
    for c in ledger:
        fin = c.get('frontier_inputs')
        if fin:
            f = my_frontier(fin['point'], fin['path'])
            frs[c['config_id']] = {'mine': f, 'recorded': (c.get('frontier') or {}).get('label')}
            if f['label'] != (c.get('frontier') or {}).get('label'):
                mm.append({'what': f"{c['config_id']} frontier", 'mine': f['label'], 'recorded': (c.get('frontier') or {}).get('label')})
    summary = {'outcome': mine['outcome'], 'selected': mine.get('selected'), 'tier': mine.get('tier'), 'near_tie_set': mine.get('near'),
               'statuses': {r['config_id']: r['status'] for r in mine['records']},
               'scores': {r['config_id']: r.get('score') for r in mine['records']},
               'tiers': {r['config_id']: r.get('tier') for r in mine['records']},
               'claims': {r['config_id']: r.get('claims') for r in mine['records'] if r.get('claims')},
               'frontier_replays': frs, 'mismatches': mm}
    V.write_record(f'V6_SELECTION__{a.tag}.json', summary)
    chk = {'id': f'V6.SELECTION__{a.tag}', 'what': 'own replay of the frozen NRM1 claims and selection rules on the recorded selection inputs, '
           'compared with the recorded selection (outcome, selected id, near-tie set, per-configuration status, G1-G7, tier, score, '
           'reserve qualification, useful-component label, frontier labels where inputs are recorded)',
           'inputs': V.inputs_sha([a.input, a.record, RULES, Path(__file__)]),
           'result': summary, 'numerical_status': 'PASS' if not mm else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps({k: summary[k] for k in ('outcome', 'selected', 'tier', 'near_tie_set')}, indent=1))
    for m in mm[:20]:
        print('  MISMATCH', json.dumps(m, default=V._jd)[:300])
    return 0


if __name__ == '__main__':
    sys.exit(main())
