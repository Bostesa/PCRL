"""Verifier check 4 (label-free): the decisions of the registered readings IA_REG and IB_REG re-derived with own arithmetic.

Inputs: the verifier's own records (V2 utility: G1 and the paired MODEL_SELECT costs of CAND vs TWIN and every path point; V3 grid:
replayed per-cell SCORE AUC differences, bootstrap cell SEs and non-table winner AUCs) and, for comparison, the recorded GRID.json
'paired' / 'cells' entries and the claims' DECISIONS.json. Nothing here imports ipr.rules, ipr.claims, nrm.rules or ipr.evaluate.

D2 conjuncts (RULES.md 3.1): (b) vs TWIN: aggregate pair benefit (blocks version x repetition; se_agg = max(sqrt(se_hh^2 + se_rs^2 +
se_mc^2), 0.001)) resolved (mean - 2 se_agg > 0, both version means > 0) and the recipient gate (income and occupation views: r_v <= 0.010
at the per-version point, not resolved worse); (c) per gating path (TWIN + saved points): Q* = the cheapest path point whose pooled
MODEL_SELECT paired total-NLL cost (CAND - point; mean over versions) is <= 0 (ties keep path order); (c1) AGG_POSITIVE vs Q* and no
secondary metric resolved higher (pooled value - 2 x mean version SE > 0); (c2) the recipient gate vs Q*; (c3) no cheaper path point
(cost > 0) with a resolved negative pair benefit (upper < 0 and both version means < 0); (d) the non-table winner's point benefit > 0
in both version means vs TWIN and every gating Q*. D3 (nine references: AGG_POSITIVE + recipient gate each), D4 (0.020 vs NEURAL_RAW
and H_LJ), D5 (card declarations), the outcome class.

    python3 -m ipr.jobqueue run --unit IPR_VER_DECISIONS_V1 --stage verify --rss-gib 1 -- python3 <this file>
"""
from __future__ import annotations
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vlib as V  # noqa: E402

READINGS = {
    'IA_REG': {'cand': 'IA_SFG', 'paths': {'SBC': ['IA_SBC_HI'], 'SPEND': ['IA_SPD_HI'], 'RO': ['IA_RO_HI']}, 'sex_blind': False,
               'gating': ('TWIN', 'SBC_FRONTIER', 'SPEND_FRONTIER', 'RO_FRONTIER'),
               'met_label': 'IMPROVEMENT_BEYOND_MATCHED_CONTROLS'},
    'IB_REG': {'cand': 'IB_CSF', 'paths': {'SPEND': ['IB_SPD_LO', 'IB_SPD_HI']}, 'sex_blind': True,
               'gating': ('TWIN', 'SPEND_FRONTIER'), 'met_label': 'SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND'},
}
PATH_NAME = {'SBC': 'SBC_FRONTIER', 'SPEND': 'SPEND_FRONTIER', 'RO': 'RO_FRONTIER'}
NINE = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')


def latest(prefix):
    ps = sorted(V.RECORDS.glob(f'{prefix}__*.json'), key=lambda p: p.stat().st_mtime)
    return ps[-1] if ps else None


def grid_entries():
    """Recorded paired entries and cells merged over both versions."""
    paired, cells = {}, {}
    for v in (0, 1):
        g = json.loads((V.GRID_DIR / f'S{v}' / 'GRID.json').read_text())
        cells.update(g['cells'])
        for cand, comps in g['paired'].items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = paired.setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells'])
                    d['cell_se'].update(e['cell_se'])
    return paired, cells


def own_entries(v3):
    """Replayed per-cell differences and SEs from the verifier's own V3 record."""
    out = {}
    for r in (v3.get('G') or {}).get('paired', []):
        if 'error' in r:
            continue
        d = out.setdefault(r['cand'], {}).setdefault(r['comp'], {}).setdefault(r['view'], {'cells': {}, 'cell_se': {}})
        d['cells'][r['cell']] = r['diff']
        d['cell_se'][r['cell']] = r['se']
    nol = {}
    for r in (v3.get('G') or {}).get('cells', []):
        if r['view'] == 'pair' and 'nolookup' in r:
            nol[(r['cell'], r['member'])] = r['nolookup']['score_auc']
    return out, nol


def comparison(entry):
    b = V.aggregate(entry['pair']['cells'], entry['pair']['cell_se'])
    lab = V.benefit_label(b)
    rec = {'benefit': lab, 'benefit_mean': b.get('mean'), 'benefit_se': b.get('se_agg'),
           'version_means': {v: b['per_version'][v]['mean'] for v in (0, 1)} if b.get('complete') else None, 'recipient': {}}
    gate = True
    for w in ('income', 'occupation'):
        a = V.aggregate(entry[w]['cells'], entry[w]['cell_se'])
        lw, gw = V.recipient_label(a)
        rec['recipient'][w] = {'label': lw, 'version_means': {v: a['per_version'][v]['mean'] for v in (0, 1)} if a.get('complete') else None}
        gate = gate and gw
    rec['recipient_gate'] = gate
    rec['pass'] = lab == 'AGG_POSITIVE' and gate
    rec['resolved_negative'] = (b.get('complete') and not b['alias'] and b['upper'] < 0
                                and all(b['per_version'][v]['mean'] < 0 for v in (0, 1)))
    rec['agg'] = {k: b.get(k) for k in ('mean', 'se_hh', 'se_rs', 'se_mc', 'se_agg', 'lower', 'upper')}
    return rec


def pooled(costs, metric):
    vals = [costs[str(v)][metric]['value'] for v in (0, 1)]
    ses = [costs[str(v)][metric]['se'] for v in (0, 1)]
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2, 'by_version': vals}


def version_means(cells):
    out = {}
    for v in (0, 1):
        blocks = {}
        for c in V.REG_CELLS:
            if int(c[1]) == v:
                blocks.setdefault(c[2:4], []).append(cells[c])
        out[v] = sum(sum(b) / len(b) for b in blocks.values()) / len(blocks)
    return out


def derive(rid, spec, entries, nol_of, costs_of, util):
    cand = spec['cand']
    res = {'cand': cand}
    res['G1_cand'] = util['members'][cand]['g1_both_versions']
    res['G1_twin'] = util['members']['CTL_TWIN']['g1_both_versions']
    res['D1'] = 'TASK_FEASIBLE' if res['G1_cand'] and res['G1_twin'] else ('CAND_INFEASIBLE' if not res['G1_cand'] else
                                                                           'CAND_FEASIBLE_TWIN_INFEASIBLE')
    if not res['G1_cand']:
        res['D2'] = {'label': 'NOT_ASSESSED_TASK_INFEASIBLE'}
        return res
    name_of = {'TWIN': 'CTL_TWIN'}
    comps = {}
    tw = comparison(entries[cand]['CTL_TWIN'])
    comps['TWIN'] = {**tw, 'point': 'TWIN'}
    nol = {}

    def nol_entry(member):
        cells = {}
        for c in V.REG_CELLS:
            a, b = nol_of(c, cand), nol_of(c, member)
            if a is None or b is None:
                return None
            cells[c] = b - a
        return {'cells': cells, 'version_means': version_means(cells)}
    for key, pts in spec['paths'].items():
        path = ['TWIN'] + [p for p in pts if p != 'TWIN']
        prim = {}
        for p in path:
            m = name_of.get(p, p)
            prim[p] = pooled(costs_of[cand][m], 'total|nll')
        above = sorted((p for p in path if prim[p]['value'] <= 0), key=lambda p: -prim[p]['value'])
        c = {'path': path, 'primary': {p: prim[p]['value'] for p in path}, 'point': above[0] if above else None}
        if not above:
            c.update({'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'NO_BRACKET', 'pass': False, 'dominated_by_cheaper': []})
        else:
            q = above[0]
            cq = comparison(entries[cand][name_of.get(q, q)])
            sec = {m: pooled(costs_of[cand][name_of.get(q, q)], m) for m in SECONDARY}
            more = sorted(m for m, x in sec.items() if x['value'] - 2 * x['se'] > 0)
            if cq['benefit'] != 'AGG_POSITIVE':
                lab = 'NOT_SEPARATED_FROM_TASK_SPENDING'
            elif more:
                lab = 'SEPARATED_ON_PRIMARY_COST_ONLY'
            else:
                lab = 'DOMINATES_CONTROL_FRONTIER'
            dominated = [p for p in path if prim[p]['value'] > 0 and comparison(entries[cand][name_of.get(p, p)])['resolved_negative']]
            c.update({'label': lab, 'benefit': cq['benefit'], 'benefit_mean': cq['benefit_mean'], 'benefit_se': cq['benefit_se'],
                      'version_means': cq['version_means'], 'recipient_gate_vs_bracket': cq['recipient_gate'],
                      'recipient_vs_bracket': {w: cq['recipient'][w]['label'] for w in ('income', 'occupation')},
                      'secondary_more_spending': more, 'dominated_by_cheaper': dominated,
                      'pass': lab == 'DOMINATES_CONTROL_FRONTIER' and cq['recipient_gate'] and not dominated})
        comps[PATH_NAME[key]] = c
    tg = {}
    for n in spec['gating']:
        q = comps[n].get('point')
        if q and q not in tg:
            tg[q] = nol_entry(name_of.get(q, q))
    passed = [n for n in spec['gating'] if comps[n]['pass']]
    targeted = not all(x is not None and all(x['version_means'][v] > 0 for v in (0, 1)) for x in tg.values())
    if len(passed) == len(spec['gating']) and targeted:
        lab = 'READER_TARGETED'
    elif len(passed) == len(spec['gating']):
        lab = spec['met_label']
    elif not passed:
        lab = 'NOT_SEPARATED'
    else:
        lab = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    res['D2'] = {'label': lab, 'passed': passed, 'comparisons': comps, 'targeting': tg, 'met': lab == spec['met_label']}
    # D3 / D4
    d3 = {}
    for r in NINE:
        e = entries[cand].get(r)
        d3[r] = comparison(e) if e else None
    fail = [r for r in NINE if d3[r] is not None and not d3[r]['pass']]
    missing = [r for r in NINE if d3[r] is None]
    res['D3'] = {'label_before_budget_credit': 'NOT_SUPERIOR' if fail else ('INCOMPLETE_IN_THIS_SOURCE' if missing else
                                                                            'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES'),
                 'failing': fail, 'missing_in_this_source': missing, 'per_reference': {r: (None if d3[r] is None else {k: d3[r][k] for k in ('benefit', 'benefit_mean',
                                                                                                          'benefit_se', 'recipient_gate')})
                                                     for r in NINE}}
    t = {}
    for r in ('NEURAL_RAW', 'H_LJ'):
        b = V.aggregate(entries[cand][r]['pair']['cells'], entries[cand][r]['pair']['cell_se'])
        t[r] = V.benefit_label(b, threshold=V.TARGET)
    if all(x == 'TARGET_AGG_RESOLVED' for x in t.values()):
        lvl = 'MATERIAL_0020_RESOLVED' if all(d3[r]['recipient_gate'] for r in t) else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in t.values()):
        lvl = 'MATERIAL_0020_POINT_ONLY'
    elif all(d3[r]['benefit'] == 'AGG_POSITIVE' for r in t):
        lvl = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        lvl = 'NOT_MET'
    res['D4'] = {'level': lvl, 'target_screens': t}
    res['outcome_class'] = ('COMPONENT' if res['D2']['met'] and res['D3']['label_before_budget_credit'] != 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES'
                            else 'CANDIDATE' if res['D2']['met'] else 'TASK_ONLY_FEASIBLE_NO_PROTECTION')
    return res


def compare(mine, dec):
    d = dec['decisions']
    out = {}
    out['D1'] = {'mine': mine['D1'], 'recorded': d['D1']['label'], 'match': d['D1'].get('label_unprefixed', d['D1']['label']) == mine['D1']}
    D2 = d['D2']
    out['D2_label'] = {'mine': mine['D2']['label'], 'recorded': D2['label'],
                       'match': D2.get('label_unprefixed', D2['label']) == mine['D2']['label']}
    cc = {}
    for n, c in (mine['D2'].get('comparisons') or {}).items():
        rc = (D2.get('comparisons') or {}).get(n) or {}
        e = {'mine_point': c.get('point'), 'recorded_point': rc.get('point'), 'point_match': c.get('point') == rc.get('point'),
             'mine_pass': c.get('pass'), 'recorded_pass': rc.get('pass'), 'pass_match': c.get('pass') == rc.get('pass'),
             'mine_benefit_mean': c.get('benefit_mean'), 'recorded_benefit_mean': rc.get('benefit_mean'),
             'mine_benefit_se': c.get('benefit_se'), 'recorded_benefit_se': rc.get('benefit_se')}
        if c.get('benefit_mean') is not None and rc.get('benefit_mean') is not None:
            e['abs_diff_mean'] = abs(c['benefit_mean'] - rc['benefit_mean'])
            e['abs_diff_se'] = abs(c['benefit_se'] - rc['benefit_se'])
        if n != 'TWIN':
            e['mine_label'] = c.get('label'); e['recorded_label'] = rc.get('label')
            e['label_match'] = c.get('label') == rc.get('label')
            e['mine_dominated'] = c.get('dominated_by_cheaper'); e['recorded_dominated'] = rc.get('dominated_by_cheaper')
            e['mine_gate'] = c.get('recipient_gate_vs_bracket'); e['recorded_gate'] = rc.get('recipient_gate_vs_bracket')
            fr = rc.get('frontier') or {}
            e['recorded_primary'] = {k: (x or {}).get('value') for k, x in (fr.get('primary_cost_differences') or {}).items()}
            e['mine_primary'] = c.get('primary')
            if e['recorded_primary'] and e['mine_primary']:
                e['primary_max_abs_diff'] = max(abs(e['recorded_primary'][k] - e['mine_primary'][k]) for k in e['mine_primary']
                                                if k in e['recorded_primary'])
        else:
            e['mine_benefit'] = c.get('benefit'); e['recorded_benefit'] = rc.get('benefit')
            e['mine_gate'] = c.get('recipient_gate'); e['recorded_gate'] = rc.get('recipient_gate_pass')
        cc[n] = e
    out['D2_comparisons'] = cc
    tgm = {q: (None if x is None else {'version_means': x['version_means'], 'pass': all(x['version_means'][v] > 0 for v in (0, 1))})
           for q, x in (mine['D2'].get('targeting') or {}).items()}
    out['D2_targeting'] = {'mine': tgm, 'recorded': D2.get('targeting')}
    out['D2_flags_recorded'] = D2.get('flags')
    out['D3'] = {'mine_before_credit': mine['D3']['label_before_budget_credit'], 'recorded': d['D3']['label'],
                 'recorded_credit': d['D3'].get('reader_budget_credit')}
    out['D4'] = {'mine_level': mine['D4']['level'], 'recorded': d['D4']['label'], 'recorded_level': d['D4'].get('level')}
    out['D5'] = {'recorded': d['D5']['label']}
    out['class'] = {'mine': mine['outcome_class'], 'recorded': d['outcome_class']['label'],
                    'recorded_flags': d['outcome_class'].get('flags')}
    out['registered'] = d.get('registered')
    return out


def main():
    t0 = time.process_time()
    v2p, v3p = latest('V2_UTILITY'), latest('V3_GRID')
    util = json.loads(v2p.read_text())
    v3 = json.loads(v3p.read_text()) if v3p else {}
    rec_paired, rec_cells = grid_entries()
    own_paired, own_nol = own_entries(v3)
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'inputs': {'v2': v2p.name, 'v3': v3p.name if v3p else None}, 'readings': {}}
    costs_of = util['decisive_costs']
    for rid, spec in READINGS.items():
        ent = {}
        dp = V.RESULTS / 'claims' / rid / 'DECISIONS.json'
        dec = json.loads(dp.read_text()) if dp.is_file() else None
        for src, entries, nolf in (
                ('recorded_grid', rec_paired,
                 lambda c, m: ((((rec_cells.get(c) or {}).get(m) or {}).get('pair') or {}).get('extras') or {}).get('nolookup', {}).get('score_auc')),
                ('replayed_grid', own_paired, lambda c, m: own_nol.get((c, m)))):
            try:
                mine = derive(rid, spec, entries, nolf, costs_of, util)
                ent[src] = {'mine': mine, 'vs_decisions': compare(mine, dec) if dec else 'DECISIONS.json missing'}
            except Exception as e:
                import traceback
                ent[src] = {'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-2000:]}
        if dec:
            ent['decisions_meta'] = {k: dec.get(k) for k in ('utc', 'plan', 'bank', 'rules_sha256', 'registration_status',
                                                             'reader_budget', 'inputs_summary', 'seeds', 'boot_seeds')}
            ent['grid_validity_effective'] = (dec.get('grid_validity') or {}).get('effective')
            dd = dec['decisions']
            ent['recorded_D5'] = dd.get('D5')
            ent['recorded_D1'] = dd.get('D1')
            ent['recorded_fineness_flags'] = [f for f in (dd['D2'].get('flags') or []) if 'FINENESS' in f or 'COARSE' in f
                                              or 'BRACKET' in f or 'POINT_RULE' in f]
        out['readings'][rid] = ent
    # registration status: card first commit vs first MODEL_SELECT read
    ev = [json.loads(l) for l in V.ACCESS_LOG.read_text().splitlines() if l.strip()]
    first_ms = min((V.parse_utc(e['utc']) for e in ev if e.get('role') == 'MODEL_SELECT'), default=None)
    reg = {}
    for card in ('IA_SFG_v1.md', 'IB_CSF_v1.md'):
        r = subprocess.run(['git', '-C', str(V.ROOT), 'log', '--diff-filter=A', '--format=%cI', '--',
                            f'results/{V.CAMPAIGN}/cards/{card}'], capture_output=True, text=True)
        first = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
        reg[card] = {'card_first_commit': first, 'first_model_select_read': str(first_ms),
                     'committed_before_first_ms': bool(first and first_ms and V.parse_utc(first) < first_ms)}
    out['registration'] = reg
    out['cpu_s'] = round(time.process_time() - t0, 3)
    out['utc_end'] = V.utc()
    p = V.write_record(f'V4_DECISIONS__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'record': str(p), 'registration': reg}
    for rid, ent in out['readings'].items():
        for src in ('recorded_grid', 'replayed_grid'):
            x = ent.get(src) or {}
            if 'error' in x:
                summ[f'{rid}|{src}'] = x['error']
                continue
            vs = x.get('vs_decisions')
            if isinstance(vs, dict):
                summ[f'{rid}|{src}'] = {'D1': vs['D1'], 'D2': vs['D2_label'], 'class': vs['class'], 'D3': vs['D3'], 'D4': vs['D4'],
                                        'D5': vs['D5'],
                                        'comparisons': {n: {k: e.get(k) for k in ('mine_point', 'recorded_point', 'mine_pass', 'recorded_pass',
                                                                                  'abs_diff_mean', 'abs_diff_se', 'mine_label', 'recorded_label',
                                                                                  'primary_max_abs_diff')}
                                                        for n, e in vs['D2_comparisons'].items()}}
            else:
                summ[f'{rid}|{src}'] = {'mine_D2': x['mine']['D2']['label'], 'decisions': vs}
    print(json.dumps(summ, default=V._default))
    return 0


if __name__ == '__main__':
    sys.exit(main())
