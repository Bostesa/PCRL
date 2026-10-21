"""Verifier check 4 (label-free): re-derive the D2 conjuncts (and D1, D3 / D4 descriptively) of the registered readings DA_REG and
DB_REG from the saved grid records with own arithmetic, and compare with claims/<plan>/DECISIONS.json.

Inputs (read-only): runs/GRID/DDR1_MAIN/S<v>/GRID.json (per-cell paired differences and bootstrap cell SEs, the cells' non-table
winner AUCs, the extras' recipe validation AUCs), runs/<F>/<C>/EVAL_MS.json paired costs (verified separately to 1e-9 by part U
of v2_label.py), the verifier's own utility record (G1 of CAND and TWIN), the committed plans and the NRM1 published reference labels.
Own implementations (typed from RULES.md 3 and NRM1 RULES.md): aggregate over blocks (version x repetition; block means over reader
seeds; se_hh = RMS of cell SEs with the estimator weights, se_rs = RMS deviation of block means, se_agg = max(sqrt(se_hh^2 +
se_rs^2 + se_mc^2), 0.001), per-version the same on that version's blocks); AGG_POSITIVE = mean - 2 se_agg > 0 and both version
means > 0; recipient gate = both per-version means <= 0.010 (single views; r = candidate - comparator); bracket Q* = the path point
(TWIN + saved points, aliases once) with the largest pooled total-NLL cost difference d = cost(CAND) - cost(point) <= 0; secondary
metrics resolved higher = pooled value - 2 pooled SE > 0 (pooled = mean of versions; SE = mean of version SEs); (c3) every cheaper
path point (d > 0) not resolved negative (upper < 0 and both version means < 0); (d) non-table winner's per-cell pair benefit,
version means (reader-seed block means, then repetition means) > 0 in both versions vs TWIN and every Q*.

    python3 -m ddr.jobqueue run --unit DDR_VER_DECISIONS_V<n> --stage verify --rss-gib 1 -- python3 <this file> --v2-record <records/V2_LABEL__...json>
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CAMPAIGN = 'pcrl_dual_design_release_v1'
RESULTS = ROOT / 'results' / CAMPAIGN
RUNS = ROOT / '.private' / CAMPAIGN / 'runs'
GRID = RUNS / 'GRID' / 'DDR1_MAIN'
VERSIONS = (0, 1)
REPS = ('R1', 'R2')
KS = (0, 1)
REG = [f'S{v}{r}k{k}' for v in VERSIONS for r in REPS for k in KS]
FLOOR, Z, LIMIT, TARGET = 0.001, 2.0, 0.010, 0.020
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
BENCH = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')


def parse(c):
    return int(c[1]), c[2:4], int(c[5:])


def agg(entry, cells=REG):
    """entry = {'cells': {cell: x}, 'cell_se': {cell: se}}; registered cells only."""
    if not entry or any(c not in entry.get('cells', {}) or c not in entry.get('cell_se', {}) for c in cells):
        return None
    blocks = {}
    for c in cells:
        v, r, k = parse(c)
        blocks.setdefault((v, r), {})[k] = (float(entry['cells'][c]), float(entry['cell_se'][c]), float((entry.get('cell_se_mc') or {}).get(c, 0.0) or 0.0))

    def summ(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b].values()) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        se_hh = math.sqrt(sum(blocks[b][k][1] ** 2 / (nb * len(blocks[b])) for b in keys for k in blocks[b]))
        se_mc = math.sqrt(sum(blocks[b][k][2] ** 2 / (nb * len(blocks[b])) for b in keys for k in blocks[b]))
        se_rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        return {'mean': x, 'se_hh': se_hh, 'se_rs': se_rs, 'se_mc': se_mc, 'raw': math.sqrt(se_hh ** 2 + se_rs ** 2 + se_mc ** 2),
                'block_means': {f'S{b[0]}{b[1]}': m for b, m in sorted(mb.items())}}
    keys = sorted(blocks)
    tot = summ(keys)
    per = {v: summ([b for b in keys if b[0] == v]) for v in VERSIONS}
    alias = all(t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0 for b in blocks.values() for t in b.values())
    for s in [tot] + list(per.values()):
        s['se_agg'] = 0.0 if alias else max(s['raw'], FLOOR)
        s['lower'], s['upper'] = s['mean'] - Z * s['se_agg'], s['mean'] + Z * s['se_agg']
    tot['per_version'] = per
    tot['alias'] = alias
    return tot


def benefit(entry, threshold=0.0):
    a = agg(entry)
    if a is None:
        return {'label': 'INCOMPLETE', 'resolved': False}
    pos = ('AGG_POSITIVE', 'AGG_POSITIVE_POINT_ONLY', 'AGG_NOT_POSITIVE') if threshold == 0 else \
        ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY', 'TARGET_AGG_NOT_MET')
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif a['lower'] > threshold and all(a['per_version'][v]['mean'] > threshold for v in VERSIONS):
        lab = pos[0]
    elif a['mean'] > threshold:
        lab = pos[1]
    else:
        lab = pos[2]
    return {'label': lab, 'resolved': lab == pos[0], 'mean': a['mean'], 'se_agg': a['se_agg'],
            'version_means': {v: a['per_version'][v]['mean'] for v in VERSIONS}, 'se_hh': a['se_hh'], 'se_rs': a['se_rs']}


def recipient(entry):
    a = agg(entry)
    if a is None:
        return {'gate': None, 'label': 'INCOMPLETE'}
    pv = a['per_version']
    if a['alias']:
        return {'gate': True, 'label': 'ALIAS_ZERO', 'version_means': {v: 0.0 for v in VERSIONS}}
    if any(pv[v]['lower'] > LIMIT for v in VERSIONS):
        lab = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif not all(pv[v]['mean'] <= LIMIT for v in VERSIONS):
        lab = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(pv[v]['upper'] <= LIMIT for v in VERSIONS):
        lab = 'RECIPIENT_AGG_RESOLVED'
    else:
        lab = 'RECIPIENT_AGG_EDGE'
    return {'gate': lab in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE'), 'label': lab,
            'version_means': {v: pv[v]['mean'] for v in VERSIONS}}


def comparison(e):
    b = benefit((e or {}).get('pair'))
    rc = {w: recipient((e or {}).get(w)) for w in ('income', 'occupation')}
    gate = None if any(x['gate'] is None for x in rc.values()) else all(x['gate'] for x in rc.values())
    return {'benefit': b, 'recipient': rc, 'gate': gate, 'pass': bool(b['resolved'] and gate)}


def pooled(costs, metric):
    vals, ses = [], []
    for v in VERSIONS:
        c = costs[str(v)][metric] if str(v) in costs else costs[v][metric]
        vals.append(float(c['value']))
        ses.append(math.sqrt(float(c['se']) ** 2 + float(c.get('se_mc', 0.0) or 0.0) ** 2))
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2, 'by_version': vals}


def nolookup_cells(cells, cand, comp):
    out = {}
    for c in REG:
        try:
            a = cells[c][cand]['pair']['extras']['nolookup']['score_auc']
            b = cells[c][comp]['pair']['extras']['nolookup']['score_auc']
        except (KeyError, TypeError):
            return None
        out[c] = float(b) - float(a)
    return out


def version_means(cellvals):
    out = {}
    for v in VERSIONS:
        blocks = {}
        for c in REG:
            vv, r, _ = parse(c)
            if vv == v:
                blocks.setdefault(r, []).append(cellvals[c])
        out[v] = sum(sum(b) / len(b) for b in blocks.values()) / len(blocks)
    return out


def load_grid():
    g = {'paired': {}, 'cells': {}, 'members': {}}
    for v in VERSIONS:
        x = json.loads((GRID / f'S{v}' / 'GRID.json').read_text())
        g['members'][v] = x['members']
        g['cells'].update(x['cells'])
        for cand, comps in x['paired'].items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = g['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells']); d['cell_se'].update(e['cell_se'])
    return g


def nolookup_winner_checks(cells, members):
    """Label-free consistency of the recorded non-table winner (pair view): its validation AUC equals the maximum over the
    non-LOOKUP locked candidates (pair recipes and both ignore-one alternatives, from the extras' recipe validation AUCs); when it
    equals the main winner, its SCORE AUC equals the main winner's (which part G of v2_label replays exactly)."""
    bad, same, differ = [], 0, 0
    for c in REG:
        for m in members:
            ent = cells[c][m]
            ex = {w: ent[w]['extras']['recipe_val_auc'] for w in ('income', 'occupation', 'pair')}
            cand = [val for w in ex for n, val in ex[w].items() if n != 'LOOKUP']
            nl = ent['pair']['extras']['nolookup']
            if abs(nl['val_auc'] - max(cand)) > 1e-15:
                bad.append({'cell': c, 'member': m, 'val': nl['val_auc'], 'max': max(cand)})
            if nl['winner'] == ent['pair']['winner']:
                same += 1
                if abs(nl['score_auc'] - ent['pair']['score_auc']) > 1e-15:
                    bad.append({'cell': c, 'member': m, 'issue': 'same winner, different SCORE AUC'})
            else:
                differ += 1
    return {'problems': bad, 'nolookup_equals_main_winner': same, 'nolookup_differs_from_main_winner': differ}


SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
SEED_INV = ('OUT_LR', 'OUT_HGB', 'LOG_LR', 'DEC_LR', 'CONF_HGB', 'LOOKUP', 'HGB15', 'HGB31', 'CATI01', 'CATI1')
MIN_LEAF = {'ET5': 5, 'ET25': 25}


def distinct_counts(member, v):
    import numpy as np
    fam = member.split('_')[0]
    with np.load(RUNS / fam / member / f'V{v}' / 'RELEASE.npz', allow_pickle=False) as z:
        inc, occ = z['MODEL_SELECT__income'], z['MODEL_SELECT__occupation']
    return {'income': int(np.unique(inc, axis=0).shape[0]), 'occupation': int(np.unique(occ, axis=0).shape[0]),
            'pair': int(np.unique(np.column_stack((inc, occ)), axis=0).shape[0])}


def seed_validity(cells, minimum):
    """RULES.md 9.3 on the minimum set (own implementation): every seed-dependent recipe's k0 / k1 score fingerprints; an equality
    is explained only for ET5 / ET25 with (a) another seed-dependent recipe differing in the same wire, view, version, repetition,
    (b) the same recipe differing on another view of the same wire, (c) distinct inputs of the view <= 8001 // min_leaf."""
    fp = {}
    for c, cres in cells.items():
        v, r, k = parse(c)
        for m in minimum:
            if m not in cres:
                continue
            for w in ('income', 'occupation', 'pair'):
                fp[(m, v, r, w, k)] = (cres[m][w].get('extras') or {}).get('score_sha256') or {}
    dc = {(m, v): distinct_counts(m, v) for m in minimum for v in VERSIONS}
    eqs, unexplained, nondet, checked = [], [], [], 0
    for m in minimum:
        for v in VERSIONS:
            for r in REPS:
                for w in ('income', 'occupation', 'pair'):
                    f0, f1 = fp.get((m, v, r, w, 0)), fp.get((m, v, r, w, 1))
                    if f0 is None or f1 is None:
                        unexplained.append({'member': m, 'v': v, 'rep': r, 'view': w, 'why': 'fingerprints missing'})
                        continue
                    for rec in SEED_DEP:
                        if rec not in f0 or rec not in f1:
                            unexplained.append({'member': m, 'v': v, 'rep': r, 'view': w, 'recipe': rec, 'why': 'recipe unchecked'})
                            continue
                        checked += 1
                        if f0[rec] == f1[rec]:
                            a = any(f0.get(o) != f1.get(o) for o in SEED_DEP if o != rec and o in f0 and o in f1)
                            b = any((fp.get((m, v, r, ww, 0)) or {}).get(rec) != (fp.get((m, v, r, ww, 1)) or {}).get(rec)
                                    for ww in ('income', 'occupation', 'pair') if ww != w)
                            cnt = dc[(m, v)][w]
                            ok = rec in MIN_LEAF and a and b and cnt <= 8001 // MIN_LEAF[rec]
                            item = {'member': m, 'v': v, 'rep': r, 'view': w, 'recipe': rec, 'a': a, 'b': b, 'distinct': cnt,
                                    'explained': ok}
                            eqs.append(item)
                            if not ok:
                                unexplained.append(item)
                    for rec in SEED_INV:
                        if rec in f0 and rec in f1 and f0[rec] != f1[rec]:
                            nondet.append({'member': m, 'v': v, 'rep': r, 'view': w, 'recipe': rec})
    return {'checked_seed_dependent': checked, 'equalities': eqs, 'unexplained': unexplained,
            'nondeterministic_seed_invariant': nondet, 'valid_recipe_part': not unexplained,
            'distinct_counts': {f'{m}|V{v}': x for (m, v), x in dc.items()}}


def reading(plan, grid, g1, published):
    cand, twin = plan['cand'], plan['twin']
    mem = plan['members']
    ev = json.loads((RUNS / mem[cand].split('/')[0] / mem[cand].split('/')[1] / 'EVAL_MS.json').read_text())
    costs_of = lambda m: {str(v): ev['versions'][str(v)]['paired_costs'][mem[m]]['costs'] for v in VERSIONS}
    pc = grid['paired'][cand]
    out = {'cand': cand, 'twin': twin, 'G1_cand': g1.get(cand), 'G1_twin': g1.get(twin)}
    out['D1'] = 'TASK_FEASIBLE' if g1.get(cand) and g1.get(twin) else ('CAND_INFEASIBLE' if g1.get(cand) is False else 'OTHER')
    ct = comparison(pc[twin])
    out['TWIN'] = ct
    comps = {}
    targets = ['TWIN']
    for name, key in (('SBC_FRONTIER', 'SBC'), ('SPEND_FRONTIER', 'SPEND')):
        ids = ['TWIN'] + [i for i in plan['paths'][key]]
        ids = list(dict.fromkeys((plan.get('alias_of') or {}).get(i, i) for i in ids))
        prim = {}
        for i in ids:
            m = twin if i == 'TWIN' else i
            prim[i] = pooled(costs_of(m), 'total|nll')
        above = sorted((i for i in ids if prim[i]['value'] <= 0), key=lambda i: -prim[i]['value'])
        rec = {'path': ids, 'primary': {i: prim[i]['value'] for i in ids}}
        if not above:
            rec.update({'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT', 'Q': None,
                        'pass': False})
        else:
            q = above[0]
            qm = twin if q == 'TWIN' else q
            cq = comparison(pc[qm])
            sec = {m: pooled(costs_of(qm), m) for m in SECONDARY}
            more = sorted(m for m, x in sec.items() if x['value'] - Z * x['se'] > 0)
            lab = ('NOT_SEPARATED_FROM_TASK_SPENDING' if not cq['benefit']['resolved'] else
                   'SEPARATED_ON_PRIMARY_COST_ONLY' if more else 'DOMINATES_CONTROL_FRONTIER')
            dominated = []
            for i in ids:
                if prim[i]['value'] > 0:
                    a = agg(pc[twin if i == 'TWIN' else i]['pair'])
                    if a is not None and not a['alias'] and a['upper'] < 0 and all(a['per_version'][v]['mean'] < 0 for v in VERSIONS):
                        dominated.append(i)
            rec.update({'label': lab, 'Q': q, 'benefit_vs_Q': cq['benefit'], 'recipient_gate_vs_Q': cq['gate'],
                        'recipient_vs_Q': {w: cq['recipient'][w]['label'] for w in cq['recipient']},
                        'secondary_more_spending': more, 'dominated_by_cheaper': dominated,
                        'pass': lab == 'DOMINATES_CONTROL_FRONTIER' and cq['gate'] is True and not dominated})
            targets.append(q)
        comps[name] = rec
    out['frontiers'] = comps
    tg = {}
    for q in dict.fromkeys(targets):
        qm = twin if q == 'TWIN' else q
        cells = nolookup_cells(grid['cells'], cand, qm)
        if cells is None:
            tg[q] = None
            continue
        vm = version_means(cells)
        tg[q] = {'version_means': vm, 'pass': all(vm[v] > 0 for v in VERSIONS)}
    out['targeting'] = tg
    passed = [n for n, ok in (('TWIN', ct['pass']), ('SBC_FRONTIER', comps['SBC_FRONTIER']['pass']),
                              ('SPEND_FRONTIER', comps['SPEND_FRONTIER']['pass'])) if ok]
    targeted = not all(x and x['pass'] for x in tg.values())
    if not out['G1_cand']:
        lab = 'NOT_ASSESSED_TASK_INFEASIBLE'
    elif len(passed) == 3 and targeted:
        lab = 'READER_TARGETED'
    elif len(passed) == 3:
        lab = 'IMPROVEMENT_BEYOND_MATCHED_CONTROLS'
    elif not passed:
        lab = 'NOT_SEPARATED'
    else:
        lab = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    out['D2'] = lab
    out['D2_passed'] = passed
    # D3 / D4 (descriptive; the reader-budget credit flags are taken from DECISIONS.json)
    elig = [r for r in BENCH if published[r]['point'] == 'POINT_ELIGIBLE']
    refc = {r: comparison(pc.get(r)) for r in elig}
    out['D3_refs'] = {r: {'benefit': c['benefit']['label'], 'mean': c['benefit'].get('mean'), 'se_agg': c['benefit'].get('se_agg'),
                          'gate': c['gate'], 'pass': c['pass']} for r, c in refc.items()}
    out['D3_G4'] = 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if all(c['pass'] for c in refc.values()) else 'NOT_SUPERIOR'
    tgt = {r: benefit((pc.get(r) or {}).get('pair'), threshold=TARGET)['label'] for r in ('NEURAL_RAW', 'H_LJ')}
    rg = all(refc[r]['gate'] for r in ('NEURAL_RAW', 'H_LJ'))
    if all(x == 'TARGET_AGG_RESOLVED' for x in tgt.values()):
        lev = 'MATERIAL_0020_RESOLVED' if rg else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in tgt.values()):
        lev = 'MATERIAL_0020_POINT_ONLY'
    elif all(refc[r]['benefit']['resolved'] for r in ('NEURAL_RAW', 'H_LJ')):
        lev = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        lev = 'NOT_MET'
    out['D4_level'] = lev
    out['D4_target_labels'] = tgt
    return out


def compare(mine, dec):
    d = dec['decisions']
    D2 = d['D2']
    res = {'D1': (mine['D1'], d['D1'].get('label_unprefixed', d['D1']['label'])),
           'D2': (mine['D2'], D2.get('label_unprefixed', D2['label'])),
           'D3_G4_vs_D3': (mine['D3_G4'], d['D3'].get('label_unprefixed', d['D3']['label'])),
           'D4_level': (mine['D4_level'], d['D4'].get('level', d['D4'].get('label'))),
           'outcome_class': (None, d['outcome_class'].get('label_unprefixed', d['outcome_class']['label'])),
           'registered': d.get('registered'), 'adaptive_tag': d.get('adaptive_tag'), 'flags': d['outcome_class'].get('flags')}
    num = []
    comps = D2.get('comparisons') or {}
    if 'TWIN' in comps:
        c = comps['TWIN']
        num.append(('TWIN benefit_mean', mine['TWIN']['benefit']['mean'], c.get('benefit_mean')))
        num.append(('TWIN benefit_se', mine['TWIN']['benefit']['se_agg'], c.get('benefit_se')))
        res['TWIN_label'] = (mine['TWIN']['benefit']['label'], c.get('benefit'))
        res['TWIN_gate'] = (mine['TWIN']['gate'], c.get('recipient_gate_pass'))
        res['TWIN_pass'] = (mine['TWIN']['pass'], c.get('pass'))
    for n in ('SBC_FRONTIER', 'SPEND_FRONTIER'):
        if n in comps:
            c, m = comps[n], mine['frontiers'][n]
            res[f'{n}_label'] = (m['label'], c.get('label'))
            res[f'{n}_Q'] = (m.get('Q'), c.get('point'))
            res[f'{n}_gate'] = (m.get('recipient_gate_vs_Q'), c.get('recipient_gate_vs_bracket'))
            res[f'{n}_dominated'] = (m.get('dominated_by_cheaper'), c.get('dominated_by_cheaper'))
            res[f'{n}_pass'] = (m.get('pass'), c.get('pass'))
            if m.get('benefit_vs_Q'):
                num.append((f'{n} benefit_mean', m['benefit_vs_Q']['mean'], c.get('benefit_mean')))
                num.append((f'{n} benefit_se', m['benefit_vs_Q']['se_agg'], c.get('benefit_se')))
            fr = c.get('frontier') or {}
            for i, val in (m.get('primary') or {}).items():
                rv = ((fr.get('primary_cost_differences') or {}).get(i) or {}).get('value')
                num.append((f'{n} primary cost {i}', val, rv))
            res[f'{n}_secondary_more'] = (m.get('secondary_more_spending'), fr.get('secondary_more_spending'))
    tg = D2.get('targeting') or {}
    for q, x in (mine['targeting'] or {}).items():
        if x is None:
            continue
        y = tg.get(q) or {}
        for v in VERSIONS:
            rv = (y.get('version_means') or {}).get(v, (y.get('version_means') or {}).get(str(v)))
            num.append((f'targeting {q} v{v}', x['version_means'][v], rv))
        res[f'targeting_{q}_pass'] = (x['pass'], y.get('pass'))
    res['numbers'] = [{'what': w, 'mine': a, 'recorded': b, 'absdiff': (abs(a - b) if (a is not None and b is not None) else None)}
                      for w, a, b in num]
    res['max_absdiff'] = max((x['absdiff'] for x in res['numbers'] if x['absdiff'] is not None), default=None)
    res['labels_agree'] = all(a == b for k, (a, b) in ((k, v) for k, v in res.items() if isinstance(v, tuple) and k != 'outcome_class'
                                                        and k != 'D3_G4_vs_D3' and k != 'D4_level'))
    return res


def main():
    v2 = None
    if '--v2-record' in sys.argv:
        v2 = json.loads(Path(sys.argv[sys.argv.index('--v2-record') + 1]).read_text())
    g1 = {}
    if v2 and 'U' in v2:
        g1 = {k: x['G1_24_plus_4'] for k, x in v2['U']['G1'].items()}
    else:
        for fam, cfgs in (('DA', ('DA_TWIN', 'DA_STD')), ('DB', ('DB_PTC',))):
            for c in cfgs:
                ev = json.loads((RUNS / fam / c / 'EVAL_MS.json').read_text())
                g1[c] = all(ev['versions'][str(v)]['point']['point_pass_12_plus_2'] for v in VERSIONS)
    published = json.loads((ROOT / 'results' / 'pcrl_new_release_mechanisms_v1' / 'roles' / 'statistical_reviewer' /
                            'RULES.json').read_text())['references']['published_labels']
    grid = load_grid()
    out = {'g1_source': 'verifier V2 record' if v2 else 'EVAL_MS.json (coordinator)', 'readings': {}}
    members_all = sorted({m for v in VERSIONS for m in grid['members'][v]})
    out['nolookup_consistency'] = nolookup_winner_checks(grid['cells'], members_all)
    for pid in ('DA_REG', 'DB_REG'):
        plan = json.loads((RESULTS / 'claims' / f'PLAN_{pid}.json').read_text())
        mine = reading(plan, grid, g1, published)
        minimum = [plan['cand'], plan['twin']] + [x for k in ('SBC', 'SPEND') for x in plan['paths'][k]]
        mine['seed_validity'] = seed_validity(grid['cells'], minimum)
        rec = {'mine': mine}
        dp = RESULTS / 'claims' / pid / 'DECISIONS.json'
        if dp.is_file():
            dec = json.loads(dp.read_text())
            rec['comparison'] = compare(mine, dec)
            rec['recorded_grid_validity'] = {k: dec['grid_validity'].get(k) for k in ('effective', 'label') if k in dec['grid_validity']}
            rec['recorded_seeds'] = dec.get('seeds'); rec['recorded_boot_seeds'] = dec.get('boot_seeds')
        else:
            rec['comparison'] = 'NOT_RUN: DECISIONS.json missing'
        out['readings'][pid] = rec
    # registered seeds (READER_SPLITS.json) and bootstrap seeds (PROTOCOL 5) against the decision records
    man = json.loads((RESULTS / 'READER_SPLITS.json').read_text())
    want_seeds = {f'S{v}k{k}': int(man['reader_seeds'][f'V{v}k{k}']) for v in VERSIONS for k in KS}
    want_boot = {f'S{v}{r}': 2026102200 + 100 * v + {'R1': 0, 'R2': 1}[r] for v in VERSIONS for r in REPS}
    out['seed_records_match'] = {pid: (r.get('recorded_seeds') == want_seeds and r.get('recorded_boot_seeds') == want_boot)
                                 for pid, r in out['readings'].items() if 'recorded_seeds' in r}
    out['registered_seed_values'] = {'reader': want_seeds, 'boot': want_boot,
                                     'reader_formula_ok': all(want_seeds[f'S{v}k{k}'] == 2026101205 + 800 + 10 * v + k
                                                              for v in VERSIONS for k in KS)}
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import vlib as V
        uid = os.environ.get('DDR_UNIT_ID') or ('NOQUEUE_' + V.utc().replace(':', '').replace('+0000', 'Z'))
        p = V.write_record(f'V4_DECISIONS__{uid}.json', out)
        print(json.dumps({'record': str(p)}))
    except ImportError:
        pass
    print(json.dumps({pid: (r['mine']['D2'], (r['comparison'] if isinstance(r['comparison'], str) else
                                              {'labels_agree': r['comparison']['labels_agree'],
                                               'max_absdiff': r['comparison']['max_absdiff']}))
                      for pid, r in out['readings'].items()}, default=str))


if __name__ == '__main__':
    main()
