"""Addendum checks A5 (IA_R1 decisions) and A6 (the three D2R readings); label-free arithmetic on the verifier's own records.

Inputs: A3 (own utility, reserve and paired costs of the IA revision members), A4 (own replays of IPR1_R1, IPR1_ROB_R1, IPR1_ROB), the
first report's V3 replay of IPR1_MAIN (registered cells of IA_REG / IB_REG), the recorded GRID.json files (D3 references that were not
replayed), label-free MODEL_SELECT household ids (split records), and for comparison claims/IA_R1/DECISIONS.json, the D2R records,
claims/STRONGEST.json and claims/CAMPAIGN_SUMMARY.json. Nothing here imports ipr.rules, ipr.claims, nrm.rules or ipr.evaluate.

A5: D1 (G1 of TWIN and CAND; the reserve beside), D2 (b) TWIN, (c) per path TWIN + LO + HI: Q* from the pooled MODEL_SELECT total-NLL
costs, (c1) AGG_POSITIVE at Q* and no secondary metric resolved higher, (c2) the recipient gate at Q*, (c3) no cheaper point resolved
less recoverable, (d) non-table targeting; D3 (nine references), D4 (0.020), D5 (joint null), D6, the class, the flags I can derive
(CAND_STRENGTH_DIFFERS_BY_VERSION, CONTROL_GRID_COARSE_<path>, NO_BRACKET_<path>, POINT_FEASIBLE_NOT_RESERVE_QUALIFIED); the
adaptive tag from the card's first commit against the first MODEL_SELECT read; grid validity (seed formulas, seed fingerprints on the
minimum set).
A6: per D2R reading, refit effectiveness (own split assignment hashes of R1-R4, SCORE household overlaps, the eight reader seeds and
bootstrap seeds, seed fingerprints for k0/k1 and k2/k3 on the minimum set) and the per-comparator robustness labels (AGG_POSITIVE on the
24 robustness cells and on all 32 cells of the merged registered + robustness entries); the strongest-release selection.

    python3 -m ipr.jobqueue run --unit IPR_VER_A5_DECISIONS_V1 --stage verify --rss-gib 1 -- python3 <this file>
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402
import a_lib as A  # noqa: E402

PATH_NAME = {'SBC': 'SBC_FRONTIER', 'SPEND': 'SPEND_FRONTIER', 'RO': 'RO_FRONTIER'}
R1 = {'cand': 'IA_OSG', 'twin': 'CTL_TWIN',
      'paths': {'SBC': ['IA_OSG_SBC_LO', 'IA_OSG_SBC_HI'], 'SPEND': ['IA_OSG_SPD_LO', 'IA_OSG_SPD_HI'], 'RO': ['IA_OSG_RO_LO', 'IA_OSG_RO_HI']},
      'gating': ('TWIN', 'SBC_FRONTIER', 'SPEND_FRONTIER', 'RO_FRONTIER'), 'met': 'IMPROVEMENT_BEYOND_MATCHED_CONTROLS',
      'minimum': ('IA_OSG', 'CTL_TWIN', 'IA_OSG_SBC_LO', 'IA_OSG_SBC_HI', 'IA_OSG_SPD_LO', 'IA_OSG_SPD_HI', 'IA_OSG_RO_LO', 'IA_OSG_RO_HI')}
D2R = {'IA_REG': {'release': 'IA_SFG', 'main': 'IPR1_MAIN', 'rob': 'IPR1_ROB',
                  'brackets': {'SBC': 'IA_SBC_HI', 'SPEND': 'IA_SPD_HI', 'RO': 'IA_RO_HI'}},
       'IB_REG': {'release': 'IB_CSF', 'main': 'IPR1_MAIN', 'rob': 'IPR1_ROB', 'brackets': {'SBC': None, 'SPEND': 'IB_SPD_HI', 'RO': None}},
       'IA_R1': {'release': 'IA_OSG', 'main': 'IPR1_R1', 'rob': 'IPR1_ROB_R1',
                 'brackets': {'SBC': 'IA_OSG_SBC_HI', 'SPEND': None, 'RO': 'IA_OSG_RO_HI'}}}


def latest(prefix):
    ps = sorted(V.RECORDS.glob(f'{prefix}__*.json'), key=lambda p: p.stat().st_mtime)
    return ps[-1] if ps else None


def entries(paired_rows):
    out = {}
    for r in paired_rows:
        if 'error' in r:
            continue
        d = out.setdefault(r['cand'], {}).setdefault(r['comp'], {}).setdefault(r['view'], {'cells': {}, 'cell_se': {}})
        d['cells'][r['cell']] = r['diff']
        d['cell_se'][r['cell']] = r['se']
    return out


def nolookup(cell_rows):
    return {(r['cell'], r['member']): r['nolookup']['score_auc'] for r in cell_rows if r['view'] == 'pair' and 'nolookup' in r}


def grid_recorded(bank):
    paired, cells = {}, {}
    for v in (0, 1):
        g = json.loads((V.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json').read_text())
        cells.update(g['cells'])
        for cand, comps in g['paired'].items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = paired.setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells'])
                    d['cell_se'].update(e['cell_se'])
    return paired, cells


def comparison(entry, which=V.REG_CELLS):
    b = A.aggregate(entry['pair']['cells'], entry['pair']['cell_se'], which)
    rec = {'benefit': A.benefit_label(b), 'mean': b.get('mean'), 'se_agg': b.get('se_agg'),
           'version_means': {v: b['per_version'][v]['mean'] for v in (0, 1)} if b.get('complete') else None,
           'resolved_negative': A.resolved_negative(b), 'recipient': {}}
    gate = True
    for w in ('income', 'occupation'):
        a = A.aggregate(entry[w]['cells'], entry[w]['cell_se'], which)
        lab, g = A.recipient_label(a)
        rec['recipient'][w] = lab
        gate = gate and g
    rec['gate'] = gate
    rec['pass'] = rec['benefit'] == 'AGG_POSITIVE' and gate
    return rec


def pooled(costs, metric):
    vals = [costs[str(v)][metric]['value'] for v in (0, 1)]
    ses = [costs[str(v)][metric]['se'] for v in (0, 1)]
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2}


def a5(out, a3, a4):
    ent = entries(a4['banks']['IPR1_R1']['paired'])
    nol = nolookup(a4['banks']['IPR1_R1']['cells'])
    rp, rc = grid_recorded('IPR1_R1')
    costs = a3['decisive_costs']['IA_OSG']
    cand, twin = R1['cand'], R1['twin']
    res = {}
    g1c, g1t = a3['members']['IA_OSG']['g1_both_versions'], a3['members']['CTL_TWIN']['g1_both_versions']
    res['D1'] = {'label': 'TASK_FEASIBLE' if g1c and g1t else 'CAND_INFEASIBLE' if not g1c else 'CAND_FEASIBLE_TWIN_INFEASIBLE',
                 'reserve_cand': a3['members']['IA_OSG'].get('reserve_status'), 'reserve_twin': a3['members']['CTL_TWIN'].get('reserve_status')}
    comps = {'TWIN': {**comparison(ent[cand][twin]), 'point': 'TWIN'}}
    comps['TWIN']['pass'] = comps['TWIN']['benefit'] == 'AGG_POSITIVE' and comps['TWIN']['gate']
    flags = []
    name_of = {'TWIN': twin}
    for key, pts in R1['paths'].items():
        path = ['TWIN'] + pts
        prim = {p: pooled(costs[name_of.get(p, p)], 'total|nll')['value'] for p in path}
        above = sorted((p for p in path if prim[p] <= 0), key=lambda p: -prim[p])
        c = {'path': path, 'primary': prim, 'point': above[0] if above else None}
        dominated = [p for p in path if prim[p] > 0 and comparison(ent[cand][name_of.get(p, p)])['resolved_negative']]
        if not above:
            c.update({'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT',
                      'pass': False, 'dominated_by_cheaper': dominated})
            flags.append(f'NO_BRACKET_{key}')
        else:
            q = above[0]
            cq = comparison(ent[cand][name_of.get(q, q)])
            sec = {m: pooled(costs[name_of.get(q, q)], m) for m in A.SECONDARY}
            more = sorted(m for m, x in sec.items() if x['value'] - 2 * x['se'] > 0)
            lab = ('NOT_SEPARATED_FROM_TASK_SPENDING' if cq['benefit'] != 'AGG_POSITIVE' else
                   'SEPARATED_ON_PRIMARY_COST_ONLY' if more else 'DOMINATES_CONTROL_FRONTIER')
            c.update({'label': lab, 'benefit': cq['benefit'], 'mean': cq['mean'], 'se_agg': cq['se_agg'],
                      'version_means': cq['version_means'], 'gate': cq['gate'], 'recipient': cq['recipient'],
                      'secondary_more_spending': more, 'dominated_by_cheaper': dominated,
                      'pass': lab == 'DOMINATES_CONTROL_FRONTIER' and cq['gate'] and not dominated})
            if lab == 'DOMINATES_CONTROL_FRONTIER' and not cq['gate']:
                flags.append(f'RECIPIENT_GATE_FAILED_VS_{q}')
        comps[PATH_NAME[key]] = c
    tg = {}
    for n in R1['gating']:
        q = comps[n].get('point')
        if q and q not in tg:
            m = name_of.get(q, q)
            cells = {c: nol[(c, m)] - nol[(c, cand)] for c in V.REG_CELLS}
            vm = A.version_means(cells)
            tg[q] = {'version_means': vm, 'pass': all(vm[v] > 0 for v in (0, 1))}
    passed = [n for n in R1['gating'] if comps[n]['pass']]
    targeted = not all(x['pass'] for x in tg.values())
    lab = ('READER_TARGETED' if len(passed) == 4 and targeted else R1['met'] if len(passed) == 4 else
           'NOT_SEPARATED' if not passed else 'PARTIAL_' + '+'.join(passed) + '_ONLY')
    fin = {}
    plan = json.loads((V.RESULTS / 'claims' / 'PLAN_IA_R1.json').read_text())
    for key, per in plan['fineness_inputs'].items():
        fin[key] = {v: A.fineness(fi) for v, fi in per.items()}
        if any(x.startswith('COARSE') for x in fin[key].values()):
            flags.append(f'CONTROL_GRID_COARSE_{key}')
    st = plan['strengths'][0]['strength']
    if st.get('0') != st.get('1'):
        flags.append('CAND_STRENGTH_DIFFERS_BY_VERSION')
    if res['D1']['reserve_cand'] != 'RESERVE_MET':
        flags.append('POINT_FEASIBLE_NOT_RESERVE_QUALIFIED')
    res['D2'] = {'label': lab, 'passed': passed, 'comparisons': comps, 'targeting': tg, 'fineness': fin}
    d3 = {}
    for r in A.NINE:
        d3[r] = comparison(rp[cand][r])
    d3own = {r: comparison(ent[cand][r]) for r in A.REFS_FROZEN}
    fail = [r for r in A.NINE if not d3[r]['pass']]
    res['D3'] = {'label': 'NOT_SUPERIOR' if fail else 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES', 'failing': fail,
                 'per_reference': {r: {k: d3[r][k] for k in ('benefit', 'mean', 'se_agg', 'gate')} for r in A.NINE},
                 'replayed_targets': {r: {k: d3own[r][k] for k in ('benefit', 'mean', 'se_agg', 'gate')} for r in A.REFS_FROZEN}}
    tl = {}
    for r in A.REFS_FROZEN:
        b = A.aggregate(ent[cand][r]['pair']['cells'], ent[cand][r]['pair']['cell_se'], V.REG_CELLS)
        tl[r] = A.benefit_label(b, threshold=V.TARGET)
    lvl = ('MATERIAL_0020_RESOLVED' if all(x == 'TARGET_AGG_RESOLVED' for x in tl.values()) and all(d3own[r]['gate'] for r in tl) else
           'TARGET_RESOLVED_RECIPIENT_GATE_FAILED' if all(x == 'TARGET_AGG_RESOLVED' for x in tl.values()) else
           'MATERIAL_0020_POINT_ONLY' if all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in tl.values()) else
           'POSITIVE_BELOW_MATERIAL_TARGET' if all(d3own[r]['benefit'] == 'AGG_POSITIVE' for r in tl) else 'NOT_MET')
    res['D4'] = {'level': lvl, 'target_screens': tl}
    res['D5'] = 'NOT_APPLICABLE_NOT_JOINT' if plan.get('joint') is None else 'SEE_RECORD'
    res['D6'] = 'NOT_CLAIMED' if plan.get('novelty') is None else plan.get('novelty')
    res['class'] = ('TASK_INFEASIBLE' if res['D1']['label'] == 'CAND_INFEASIBLE' else
                    'CANDIDATE' if lab == R1['met'] and not fail else 'COMPONENT' if lab == R1['met'] else 'TASK_ONLY_FEASIBLE_NO_PROTECTION')
    res['flags_derived'] = sorted(set(flags + ['RO_GENERIC_ZTILT']))
    # adaptive tag
    acc = [json.loads(l) for l in V.ACCESS_LOG.read_text().splitlines() if l.strip()]
    first_ms = min(V.parse_utc(e['utc']) for e in acc if e.get('role') == 'MODEL_SELECT')
    r_ = subprocess.run(['git', '-C', str(V.ROOT), 'log', '--diff-filter=A', '--format=%cI', '--', plan['card']], capture_output=True,
                        text=True)
    first_card = V.parse_utc(r_.stdout.strip().splitlines()[-1])
    early = first_card < first_ms
    res['tag'] = {'card_first_commit': str(first_card), 'first_model_select_read': str(first_ms), 'card_before_first_ms': early,
                  'revision': plan['revision'], 'derived': None if (early and plan['revision'] == 0) else
                  f"{'REVISED' if early else 'ADAPTIVE'}_R{plan['revision']}"}
    # grid validity of IPR1_R1
    fp = A.seed_fingerprint_checks(rc, R1['minimum'], pairs=((0, 1),))
    dec = json.loads((V.RESULTS / 'claims' / 'IA_R1' / 'DECISIONS.json').read_text())
    want = {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in (0, 1)}
    wantb = {f'S{v}{r}': V.BOOT_BASE + 100 * v + V.REP_INDEX[r] for v in (0, 1) for r in ('R1', 'R2')}
    forbidden = {V.MODEL_SEED_BASE + o + 10 * v + k for o in (300, 400, 500, 600, 700, 800) for v in (0, 1) for k in range(10)}
    res['grid_validity'] = {'fingerprint_checks': fp['checks'], 'equal_seed_dependent': fp['equal'],
                            'seeds_match': dec['seeds'] == want, 'boot_match': dec['boot_seeds'] == wantb,
                            'any_forbidden': any(s in forbidden for s in dec['seeds'].values()),
                            'effective_own': not fp['equal'] and dec['seeds'] == want and dec['boot_seeds'] == wantb,
                            'recorded_effective': dec['grid_validity'].get('effective')}
    # compare with the recorded decisions
    dd = dec['decisions']
    cmpd = {'D1': (res['D1']['label'], dd['D1'].get('label_unprefixed')),
            'D1_reserve': ([res['D1']['reserve_cand'], res['D1']['reserve_twin']],
                           [dd['D1'].get('reserve', {}).get('cand', {}).get('status'), dd['D1'].get('reserve', {}).get('twin', {}).get('status')]),
            'D2': (lab, dd['D2'].get('label_unprefixed')), 'D3': (res['D3']['label'], dd['D3'].get('label_unprefixed')),
            'D4': (lvl, dd['D4'].get('level')), 'D5': (res['D5'], dd['D5'].get('label_unprefixed')),
            'D6': (res['D6'], dd['D6'].get('label_unprefixed')), 'class': (res['class'], dd['outcome_class'].get('label_unprefixed')),
            'tag': (res['tag']['derived'], dd.get('adaptive_tag')), 'registered': (False, dd.get('registered')),
            'labels_prefixed': all(str(dd[k]['label']).startswith('ADAPTIVE_R1__') for k in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'outcome_class'))}
    rc2 = {}
    for n, c in comps.items():
        r_c = dd['D2']['comparisons'][n]
        e = {'point': (c.get('point'), r_c.get('point')), 'pass': (c.get('pass'), r_c.get('pass'))}
        if c.get('mean') is not None and r_c.get('benefit_mean') is not None:
            e['abs_diff_mean'] = abs(c['mean'] - r_c['benefit_mean'])
            e['abs_diff_se'] = abs(c['se_agg'] - r_c['benefit_se'])
        if n != 'TWIN':
            e['label'] = (c.get('label'), r_c.get('label'))
            e['dominated'] = (c.get('dominated_by_cheaper'), r_c.get('dominated_by_cheaper'))
            fr = r_c.get('frontier') or {}
            rp_ = {k: x['value'] for k, x in (fr.get('primary_cost_differences') or {}).items()}
            e['primary_max_abs_diff'] = max(abs(rp_[k] - c['primary'][k]) for k in c['primary']) if rp_ else None
            e['secondary_more'] = (c.get('secondary_more_spending'), fr.get('secondary_more_spending'))
            e['gate'] = (c.get('gate'), r_c.get('recipient_gate_vs_bracket'))
        else:
            e['benefit'] = (c['benefit'], r_c.get('benefit'))
            e['gate'] = (c['gate'], r_c.get('recipient_gate_pass'))
            e['recipient'] = (c['recipient'], r_c.get('recipient'))
        rc2[n] = e
    cmpd['D2_comparisons'] = rc2
    cmpd['targeting'] = {q: (x['version_means'], (dd['D2'].get('targeting') or {}).get(q, {}).get('version_means')) for q, x in tg.items()}
    cmpd['flags'] = {'derived': res['flags_derived'], 'recorded_D2': dd['D2'].get('flags'), 'recorded_class': dd['outcome_class'].get('flags')}
    cmpd['D3_failing'] = (fail, dd['D3'].get('failing'))
    out['A5'] = {'mine': res, 'vs_recorded': cmpd}


def split_records(unit):
    out = {}
    m = {r: V.split_masks(unit, r) for r in ('R1', 'R2', 'R3', 'R4')}
    for r, x in m.items():
        a = np.where(x['FIT'], 0, np.where(x['VALIDATION'], 1, 2)).astype(np.int8)
        sc = set(np.unique(unit[x['SCORE']]).tolist())
        ov = {}
        for q, y in m.items():
            if q != r:
                sq = set(np.unique(unit[y['SCORE']]).tolist())
                ov[q] = len(sc & sq) / len(sc)
        out[r] = {'assignment_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'score_overlap': ov, 'score_households': len(sc)}
    return out


def a6(out, a4, v3, unit):
    sp = split_records(unit)
    hs = [sp[r]['assignment_sha256'] for r in ('R1', 'R2', 'R3', 'R4')]
    ovs = [sp[r]['score_overlap'][q] for r in ('R3', 'R4') for q in ('R1', 'R2', 'R3', 'R4') if q != r]
    splits_ok = len(set(hs)) == 4 and all(0 <= o < 1 for o in ovs)
    seeds8 = {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in range(4)}
    boot8 = {f'S{v}{r}': V.BOOT_BASE + 100 * v + V.REP_INDEX[r] for v in (0, 1) for r in ('R1', 'R2', 'R3', 'R4')}
    forbidden = {V.MODEL_SEED_BASE + o + 10 * v + k for o in (300, 400, 500, 600, 700, 800) for v in (0, 1) for k in range(10)}
    main_rows = {'IPR1_MAIN': (v3['G']['paired'], v3['G']['cells']), 'IPR1_R1': (a4['banks']['IPR1_R1']['paired'], a4['banks']['IPR1_R1']['cells'])}
    res = {'splits': sp, 'splits_ok': splits_ok, 'score_overlaps_R3_R4': ovs, 'seeds_ok': not any(s in forbidden for s in seeds8.values())}
    for rid, spec in D2R.items():
        rec = json.loads((V.RESULTS / 'claims' / f'{rid}_D2R' / 'D2R.json').read_text())
        mainp, _ = main_rows[spec['main']]
        robp = a4['banks'][spec['rob']]['paired']
        em, er = entries(mainp), entries(robp)
        rel = spec['release']
        _, cells_m = grid_recorded(spec['main'])
        _, cells_r = grid_recorded(spec['rob'])
        merged_cells = {**cells_m, **cells_r}
        minimum = [rel, 'CTL_TWIN'] + [b for b in spec['brackets'].values() if b]
        fp = A.seed_fingerprint_checks(merged_cells, minimum, pairs=((0, 1), (2, 3)))
        refit_ok = splits_ok and not fp['equal'] and rec['seeds'] == seeds8 and rec['boot_seeds'] == boot8
        decisive = ['CTL_TWIN'] + [b for b in (spec['brackets']['SBC'], spec['brackets']['SPEND'], spec['brackets']['RO']) if b]
        per = {}
        reg_consistency = []
        for c in decisive + [r for r in A.REFS_FROZEN]:
            pm, pr = em.get(rel, {}).get(c, {}).get('pair'), er.get(rel, {}).get(c, {}).get('pair')
            if not pm or not pr:
                per[c] = {'label': 'NOT_RUN'}
                continue
            for cell in V.REG_CELLS:
                if cell in pr['cells']:
                    reg_consistency.append(abs(pr['cells'][cell] - pm['cells'][cell]))
            cells = {**pm['cells'], **pr['cells']}
            ses = {**pm['cell_se'], **pr['cell_se']}
            rob_cells = sorted(x for x in cells if x not in V.REG_CELLS)
            ar = A.aggregate(cells, ses, rob_cells)
            aa = A.aggregate(cells, ses, sorted(cells))
            lr, la = A.benefit_label(ar), A.benefit_label(aa)
            has_reg = all(x in cells for x in V.REG_CELLS)
            ok = lr == 'AGG_POSITIVE' and la == 'AGG_POSITIVE' and refit_ok and has_reg
            key = 'TWIN' if c == 'CTL_TWIN' else c
            r_rec = rec['d2r']['per_comparator'].get(key, {})
            per[key] = {'label': 'ROBUSTNESS_CONFIRMED' if ok else 'ROBUSTNESS_NOT_CONFIRMED', 'rob_label': lr, 'all_label': la,
                        'n_rob_cells': len(rob_cells), 'n_all_cells': len(cells), 'rob_mean': ar['mean'], 'rob_se': ar['se_agg'],
                        'n_eff': ar['n_eff'], 'qualifier': ar['qualifier'],
                        'recorded': {k: r_rec.get(k) for k in ('label', 'robustness_mean', 'robustness_se', 'n_eff_robustness_cells',
                                                               'reader_seed_qualifier', 'all_cells_label')},
                        'match': (r_rec.get('label') == ('ROBUSTNESS_CONFIRMED' if ok else 'ROBUSTNESS_NOT_CONFIRMED')
                                  and abs((r_rec.get('robustness_mean') or 0) - ar['mean']) <= 1e-12
                                  and abs((r_rec.get('robustness_se') or 0) - ar['se_agg']) <= 1e-12
                                  and r_rec.get('n_eff_robustness_cells') == ar['n_eff']
                                  and r_rec.get('reader_seed_qualifier') == ar['qualifier'] and r_rec.get('all_cells_label') == la)}
        dec_labels = [per['TWIN' if c == 'CTL_TWIN' else c]['label'] for c in decisive]
        fam = ('NOT_RUN' if 'NOT_RUN' in dec_labels else 'ROBUSTNESS_CONFIRMED' if all(x == 'ROBUSTNESS_CONFIRMED' for x in dec_labels)
               else 'ROBUSTNESS_NOT_CONFIRMED')
        res[rid] = {'release': rel, 'decisive': ['TWIN' if c == 'CTL_TWIN' else c for c in decisive],
                    'recorded_decisive': rec['d2r'].get('decisive'), 'label': fam, 'recorded_label': rec['d2r'].get('label'),
                    'refit_own': 'REFIT_GENUINELY_EFFECTIVE' if refit_ok else 'REFIT_NOT_GENUINELY_EFFECTIVE',
                    'refit_recorded': rec['refit_effectiveness'].get('label'), 'fingerprint_checks': fp['checks'],
                    'fingerprint_equal': fp['equal'], 'seeds_match': rec['seeds'] == seeds8, 'boot_match': rec['boot_seeds'] == boot8,
                    'registered_cells_main_vs_rob_max_abs': max(reg_consistency) if reg_consistency else None,
                    'per_comparator': per, 'all_per_comparator_match': all(x.get('match') for x in per.values() if 'match' in x)}
    # strongest release
    sres = {}
    v3e = entries(v3['G']['paired'])
    r1e = entries(a4['banks']['IPR1_R1']['paired'])
    xs = {'IA_SFG': A.aggregate(v3e['IA_SFG']['CTL_TWIN']['pair']['cells'], v3e['IA_SFG']['CTL_TWIN']['pair']['cell_se'], V.REG_CELLS)['mean'],
          'IA_OSG': A.aggregate(r1e['IA_OSG']['CTL_TWIN']['pair']['cells'], r1e['IA_OSG']['CTL_TWIN']['pair']['cell_se'], V.REG_CELLS)['mean'],
          'IB_CSF': A.aggregate(v3e['IB_CSF']['CTL_TWIN']['pair']['cells'], v3e['IB_CSF']['CTL_TWIN']['pair']['cell_se'], V.REG_CELLS)['mean']}
    srec = json.loads((V.RESULTS / 'claims' / 'STRONGEST.json').read_text())
    for fam, ids in (('IA', ('IA_SFG', 'IA_OSG')), ('IB', ('IB_CSF',))):
        pick = max(ids, key=lambda i: xs[i])
        recx = {r_['config_id']: r_['x'] for r_ in srec['families'][fam]['records']}
        sres[fam] = {'own_x': {i: xs[i] for i in ids}, 'recorded_x': recx, 'own_selected': pick,
                     'recorded_selected': srec['families'][fam]['selected'],
                     'x_max_abs_diff': max(abs(xs[i] - recx[i]) for i in ids)}
    res['strongest'] = sres
    cs = json.loads((V.RESULTS / 'claims' / 'CAMPAIGN_SUMMARY.json').read_text())['summary']
    labs = [json.loads((V.RESULTS / 'claims' / r / 'DECISIONS.json').read_text())['decisions']['D2'].get('label_unprefixed')
            for r in ('IA_REG', 'IB_REG', 'IA_R1')]
    res['campaign_summary'] = {'d2_looks': cs['d2_looks'], 'd2_met': cs['d2_met'], 'own_looks': len(labs), 'own_labels': labs,
                               'consistent': cs['d2_looks'] == 3 and not cs['d2_met'] and all(x == 'NOT_SEPARATED' for x in labs)}
    out['A6'] = res


def main():
    t0 = time.process_time()
    from ipr import common as C                       # label-free household ids only
    unit = np.asarray(C.features('MODEL_SELECT')['unit'])
    a3p, a4p = latest('A3_UTILITY'), latest('A4_GRID')
    v3p = V.RECORDS / 'V3_GRID__IPR_VER_GRID_G_V1.json'
    a3, a4, v3 = json.loads(a3p.read_text()), json.loads(a4p.read_text()), json.loads(v3p.read_text())
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'inputs': [a3p.name, a4p.name, v3p.name], 'errors': []}
    for name, fn in (('A5', lambda: a5(out, a3, a4)), ('A6', lambda: a6(out, a4, v3, unit))):
        try:
            fn()
        except Exception as e:
            out['errors'].append({'part': name, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]})
    out['cpu_s'] = round(time.process_time() - t0, 3)
    out['utc_end'] = V.utc()
    p = V.write_record(f'A5_DECISIONS__{V.unit_id() or "NOQUEUE"}.json', out)
    s = {'record': str(p), 'errors': [e['part'] + ': ' + e['error'] for e in out['errors']], 'cpu_s': out['cpu_s']}
    if 'A5' in out:
        s['A5'] = {k: x for k, x in out['A5']['vs_recorded'].items() if k not in ('D2_comparisons', 'targeting')}
        s['A5_comparisons'] = out['A5']['vs_recorded']['D2_comparisons']
        s['A5_grid_validity'] = out['A5']['mine']['grid_validity']
    if 'A6' in out:
        s['A6'] = {rid: {k: out['A6'][rid][k] for k in ('label', 'recorded_label', 'refit_own', 'refit_recorded', 'decisive',
                                                        'recorded_decisive', 'all_per_comparator_match', 'fingerprint_checks',
                                                        'registered_cells_main_vs_rob_max_abs')}
                   | {'n_equal': len(out['A6'][rid]['fingerprint_equal'])} for rid in D2R}
        s['A6_splits_ok'] = out['A6']['splits_ok']
        s['A6_strongest'] = out['A6']['strongest']
        s['A6_summary'] = out['A6']['campaign_summary']
    print(json.dumps(s, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
