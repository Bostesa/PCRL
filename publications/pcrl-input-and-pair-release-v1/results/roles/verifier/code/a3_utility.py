"""Addendum check A3 (MODEL_SELECT labels, read once through ipr.common.load, operation 'verify', inside a queue unit).

For every member evaluated since the first report (IA_OSG, IA_OSG_SBC_LO / HI, IA_OSG_SPD_LO / HI, IA_OSG_RO_LO / HI, IA_OSG_BLIND),
and for CTL_TWIN, IA_SFG and IB_CSF as references of the reserve: the 12 clauses + 2 guards per version (24 + 4) with own metric code
(vlib.losses: first argmax, NLL floor 1e-12, summed Brier; U0_S<v> / Ucal_S<v> from the frozen bank, MODEL_SELECT role only), G1 at
the point, the two-SE reserve (own arithmetic of the OHN1 rule: margin >= 2 max(own SE, H_LJ floor SE), floor SEs read as data from the
occupation-headroom RULES.json) per version for the whole release, the occupation side and the income side; all compared with the saved
EVAL_MS.json records (point values and SEs to 1e-9; reserve flags exactly, ratios to 1e-9). Every paired cost recorded in the new
EVAL_MS records against a comparator of this campaign (W skipped) is recomputed (11 metrics). The decisive costs of the IA_R1 reading
(IA_OSG vs CTL_TWIN and every path point) are written for check A5.

    python3 -m ipr.jobqueue run --unit IPR_VER_A3_UTILITY_V1 --stage verify --rss-gib 2 -- python3 <this file>
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402
import a_lib as A  # noqa: E402

TOL = 1e-9
NEW = ('IA_OSG', 'IA_OSG_SBC_LO', 'IA_OSG_SBC_HI', 'IA_OSG_SPD_LO', 'IA_OSG_SPD_HI', 'IA_OSG_RO_LO', 'IA_OSG_RO_HI', 'IA_OSG_BLIND')
REF_MEMBERS = (('CTL', 'CTL_TWIN'), ('IA', 'IA_SFG'), ('IB', 'IB_CSF'))
DECISIVE = {'IA_OSG': ('CTL_TWIN', 'IA_OSG_SBC_LO', 'IA_OSG_SBC_HI', 'IA_OSG_SPD_LO', 'IA_OSG_SPD_HI', 'IA_OSG_RO_LO', 'IA_OSG_RO_HI',
                       'IA_OSG_BLIND', 'IA_SFG')}


def per_person(y, rel):
    return {t: V.losses(y[t], rel[t], V.K[t]) for t in V.TASKS}


def utility(own, ref_l, sex, unit):
    clauses = []
    for task in V.TASKS:
        for r in V.REFS:
            for m in V.METRICS:
                key = {'accuracy_loss': 'err', 'nll_excess': 'nll', 'brier_excess': 'brier'}[m]
                d = own[task][key] - ref_l[r][task][key]
                clauses.append({'task': task, 'reference': r, 'metric': m, 'value': float(d.mean()), 'se': V.clustered_se(d, unit),
                                'cap': V.CAPS[m]})
    d = own['occupation']['brier'] - ref_l['U0']['occupation']['brier']
    guards = []
    for g, name in ((0, 'sex0'), (1, 'sex1')):
        m_, se_ = V.subgroup_mean_se(d, sex == g, unit)
        guards.append({'group': name, 'value': m_, 'se': se_, 'cap': V.GUARD_CAP})
    for c in clauses + guards:
        c['point_pass'] = bool(c['value'] <= c['cap'])
    return clauses, guards


def costs(a, b, sex, unit):
    out = {}
    for task in V.TASKS:
        for m, key in (('acc_loss', 'err'), ('nll', 'nll'), ('brier', 'brier')):
            d = a[task][key] - b[task][key]
            out[f'{task}|{m}'] = {'value': float(d.mean()), 'se': V.clustered_se(d, unit)}
            if m == 'brier':
                for g, name in ((0, 'men_brier'), (1, 'women_brier')):
                    mv, ms_ = V.subgroup_mean_se(d, sex == g, unit)
                    out[f'{task}|{name}'] = {'value': mv, 'se': ms_}
    tot = (a['income']['nll'] + a['occupation']['nll']) - (b['income']['nll'] + b['occupation']['nll'])
    out['total|nll'] = {'value': float(tot.mean()), 'se': V.clustered_se(tot, unit)}
    return out


def main():
    t0 = time.process_time()
    from ipr import common as C
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'independent verifier addendum: 24 clauses + 4 guards, the two-SE reserve and the paired costs of the IA revision R1 '
               'configurations and the reserve references (own metric code)')
    y = {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation'])}
    sex = np.asarray(S['sex']).astype(int)
    unit = np.asarray(S['unit'])
    G = int(len(np.unique(unit)))
    from ohn import common as OC
    refs = {}
    for v in (0, 1):
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank MODEL_SELECT row order differs')
        refs[v] = {r: per_person(y, {t: bank[r]['MODEL_SELECT'][t] for t in V.TASKS}) for r in V.REFS}
    members = [('IA', c) for c in NEW] + list(REF_MEMBERS)
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'people': int(len(unit)), 'households': G, 'members': {}, 'paired': [],
           'errors': [], 'decisive_costs': {}}
    own, worst = {}, {'value': 0.0, 'se': 0.0, 'reserve_ratio': 0.0}
    for fam, cfg in members:
        ev_path = V.RUNS / fam / cfg / 'EVAL_MS.json'
        ev = json.loads(ev_path.read_text()) if ev_path.is_file() else None
        ent = {'eval_exists': ev is not None, 'versions': {}}
        res_pv = {}
        for v in (0, 1):
            try:
                rel, rec = A.load_ms(fam, cfg, v)
                o = per_person(y, rel)
                own[(cfg, v)] = o
                cl, gu = utility(o, refs[v], sex, unit)
                rs = A.reserve({f"{c['task']}|{c['reference']}|{c['metric']}": (c['value'], c['se'], c['cap']) for c in cl},
                               {g['group']: (g['value'], g['se'], g['cap']) for g in gu}, G, v)
                res_pv[v] = rs
                e = {'release_sha256': rec['release_sha256'], 'g1_version_pass': all(c['point_pass'] for c in cl + gu),
                     'max_value_over_cap': max(c['value'] / c['cap'] for c in cl + gu), 'reserve': rs,
                     'guards': {g['group']: g['value'] for g in gu}}
                if ev is not None:
                    ve = ev['versions'][str(v)]
                    e['release_sha_matches_eval'] = ve.get('release_sha256') == rec['release_sha256']
                    cmp_, wv, ws = [], 0.0, 0.0
                    for c in cl:
                        hit = [x for x in ve['utility']['clauses'] if x['task'] == c['task'] and x['reference'] == c['reference']
                               and x['metric'] == c['metric']]
                        dv, ds = abs(hit[0]['value'] - c['value']), abs(hit[0]['se'] - c['se'])
                        wv, ws = max(wv, dv), max(ws, ds)
                        cmp_.append(dv <= TOL and bool(hit[0]['point_pass']) == c['point_pass'])
                    for g in gu:
                        hit = [x for x in ve['utility']['guards'] if x['group'] == g['group']]
                        dv, ds = abs(hit[0]['value'] - g['value']), abs(hit[0]['se'] - g['se'])
                        wv, ws = max(wv, dv), max(ws, ds)
                        cmp_.append(dv <= TOL and bool(hit[0]['point_pass']) == g['point_pass'])
                    worst['value'] = max(worst['value'], wv); worst['se'] = max(worst['se'], ws)
                    e['n_values'] = len(cmp_)
                    e['all_within_tol'] = all(cmp_) and len(cmp_) == 14
                    e['max_abs_value_diff'] = wv; e['max_abs_se_diff'] = ws
                    e['point_pass_agrees'] = bool(ve['point']['point_pass_12_plus_2']) == e['g1_version_pass']
                    rr = ve.get('reserve_without_mc') or {}
                    dr = {'whole_release_met': (rr.get('whole_release_met'), rs['whole_met']),
                          'occ_side_met': (rr.get('occ_side_met'), rs['occ_met']),
                          'income_reserve_met': (rr.get('income_reserve_met'), rs['income_met'])}
                    e['reserve_flags_agree'] = all(a == b for a, b in dr.values())
                    rd = {k: abs(float(rr[k]) - float(x)) for k, x in (('min_whole_ratio', rs['min_whole_ratio']),
                                                                       ('min_occ_ratio', rs['min_occ_ratio']),
                                                                       ('min_income_ratio', rs['min_income_ratio'])) if k in rr}
                    e['reserve_ratio_max_abs_diff'] = max(rd.values()) if rd else None
                    if rd:
                        worst['reserve_ratio'] = max(worst['reserve_ratio'], max(rd.values()))
                    e['reserve_binding_agree'] = rr.get('occ_binding') == rs['occ_binding']
                ent['versions'][str(v)] = e
            except Exception as ex:
                ent['versions'][str(v)] = {'error': f'{type(ex).__name__}: {ex}'}
                out['errors'].append({'member': cfg, 'version': v, 'traceback': traceback.format_exc()[-1500:]})
        ent['g1_both_versions'] = all(ent['versions'].get(str(v), {}).get('g1_version_pass') for v in (0, 1))
        if len(res_pv) == 2:
            ent['reserve_status'] = A.reserve_status(res_pv)
        out['members'][cfg] = ent
    worst_pc = {'value': 0.0, 'se': 0.0}
    for cfg in NEW:
        ev = json.loads((V.RUNS / 'IA' / cfg / 'EVAL_MS.json').read_text())
        for v in (0, 1):
            for comp, ent in ev['versions'][str(v)].get('paired_costs', {}).items():
                if comp == 'W':
                    continue
                cname = comp.split('/')[1]
                if (cname, v) not in own:
                    out['paired'].append({'release': cfg, 'version': v, 'comparator': comp, 'error': 'comparator not recomputed'})
                    continue
                mine = costs(own[(cfg, v)], own[(cname, v)], sex, unit)
                rc = ent['costs']
                diffs = {k: (abs(rc[k]['value'] - mine[k]['value']), abs(rc[k]['se'] - mine[k]['se'])) for k in mine if k in rc}
                missing = sorted(set(mine) - set(rc))
                wv = max(d[0] for d in diffs.values()); ws = max(d[1] for d in diffs.values())
                worst_pc['value'] = max(worst_pc['value'], wv); worst_pc['se'] = max(worst_pc['se'], ws)
                out['paired'].append({'release': cfg, 'version': v, 'comparator': comp, 'max_abs_value_diff': wv, 'max_abs_se_diff': ws,
                                      'missing': missing, 'within_tol': wv <= TOL and not missing})
    for cand, comps in DECISIVE.items():
        out['decisive_costs'][cand] = {c: {str(v): costs(own[(cand, v)], own[(c, v)], sex, unit) for v in (0, 1)} for c in comps}
    # income utility of every IA_OSG* member = the twin's (bitwise income => identical income clauses)
    out['income_clauses_equal_twin'] = {
        cfg: all(np.array_equal(own[(cfg, v)]['income'][k], own[('CTL_TWIN', v)]['income'][k]) for v in (0, 1) for k in ('err', 'nll', 'brier'))
        for cfg in NEW}
    out['worst_clause'] = worst
    out['worst_paired'] = worst_pc
    out['all_clauses_within_tol'] = all(e.get('all_within_tol') for m in out['members'].values() for e in m['versions'].values())
    out['all_paired_within_tol'] = all(p.get('within_tol') for p in out['paired'])
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'A3_UTILITY__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'members': {k: {'g1': m['g1_both_versions'], 'reserve': m.get('reserve_status'),
                            'tol': [m['versions'].get(str(v), {}).get('all_within_tol') for v in (0, 1)],
                            'reserve_agree': [m['versions'].get(str(v), {}).get('reserve_flags_agree') for v in (0, 1)],
                            'max_over_cap': [round(m['versions'].get(str(v), {}).get('max_value_over_cap', float('nan')), 4) for v in (0, 1)]}
                        for k, m in out['members'].items()},
            'worst_clause': worst, 'worst_paired': worst_pc, 'n_paired': len(out['paired']),
            'all_clauses_within_tol': out['all_clauses_within_tol'], 'all_paired_within_tol': out['all_paired_within_tol'],
            'income_equal_twin': all(out['income_clauses_equal_twin'].values()), 'errors': len(out['errors']), 'cpu_s': out['cpu_s'],
            'record': str(p)}
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
