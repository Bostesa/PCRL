"""Unit VER_COMPARE_V1 (checks 5-7 comparison step). Label-free: compares the verifier's saved DEV replay results (aggregates
only) with the coordinator's report/DEV_REPORT_RESULTS.json and the source campaign's published DEV report.

Public-use ACS 2019 Florida person records; readers infer the recorded SEX column from released predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V

FIELDS_NUM = ('point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1')
FIELDS_CAT = ('status', 'category', 'point_pass', 'clause_pass', 'availability')


def num_diff(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float('inf')
    return abs(float(a) - float(b))


def main():
    mine = V.jload(V.VPRIV / 'dev_results.json')
    co = V.jload(V.RES / 'report' / 'DEV_REPORT_RESULTS.json')
    out = {'unit': 'VER_COMPARE_V1', 'labels_read': 'none'}
    # ---------------------------------------------------------------- endpoints and claims, both scorecards
    sc = {}
    for SC in ('ORIGINAL', 'ENHANCED'):
        me, ce = mine['scorecards'][SC]['endpoints'], co['scorecards'][SC]['endpoints']
        mx = {f: 0.0 for f in FIELDS_NUM}
        worst = {f: None for f in FIELDS_NUM}
        cat_mism = []
        for k in me:
            for f in FIELDS_NUM:
                d = num_diff(me[k].get(f), ce[k].get(f))
                if d > mx[f]:
                    mx[f], worst[f] = d, k
            for f in FIELDS_CAT:
                if me[k].get(f) != ce[k].get(f):
                    cat_mism.append([k, f, me[k].get(f), ce[k].get(f)])
        mc, cc = mine['scorecards'][SC]['claims'], co['scorecards'][SC]['claims']
        claims = {n: {'mine': mc[n]['label'], 'coordinator': cc[n]['label'], 'equal': mc[n]['label'] == cc[n]['label'],
                      'holds_equal': mc[n].get('holds') == cc[n].get('holds')} for n in cc}
        claims['B_nested_0020'] = {'mine': mc['B']['nested_0020']['label'], 'coordinator': cc['B']['nested_0020']['label'],
                                   'equal': mc['B']['nested_0020']['label'] == cc['B']['nested_0020']['label']}
        ff = {}
        for n in ('B', 'J'):
            a = (mc[n].get('first_failing') or {}).get('condition')
            b = (cc[n].get('first_failing') or {}).get('condition')
            ff[n] = {'mine': a, 'coordinator': b, 'equal': a == b}
        ms = [(x['name'], x['holds']) for x in mc['M']['steps']]
        cs = [(x['name'], x['holds']) for x in cc['M']['steps']]
        sc[SC] = {'n_endpoints': len(me), 'same_keys': sorted(me) == sorted(ce), 'max_abs_diff': mx, 'worst_key': worst,
                  'categorical_mismatches': cat_mism, 'claims': claims, 'first_failing': ff, 'M_steps_equal': ms == cs, 'M_steps': ms,
                  'E_fields_equal': mine['scorecards'][SC]['E_fields'] == co['scorecards'][SC]['E_fields'],
                  'E_fields': mine['scorecards'][SC]['E_fields'], 'CV_equal': mine['CV'][SC] == co['control_validity']['CV'][SC]}
    out['scorecards'] = sc
    # ---------------------------------------------------------------- interpretation and side
    it = {}
    for n, v in co['interpretation'].items():
        m = mine['interpretation'][n]
        e = {'interpretation_equal': m['interpretation'] == v['interpretation'], 'mine': m['interpretation'], 'coordinator': v['interpretation'],
             'labels_equal': (m['original_label'], m['enhanced_label']) == (v['original_label'], v['enhanced_label'])}
        if v.get('side') or m.get('side'):
            ms_, cs_ = m.get('side') or {}, v.get('side') or {}
            e['side_equal'] = ms_.get('label') == cs_.get('label')
            e['side_change_diffs'] = [num_diff(ms_.get('finalist_change'), cs_.get('finalist_change')),
                                      num_diff(ms_.get('comparator_change'), cs_.get('comparator_change'))]
            e['side'] = ms_
        if n == 'M':
            e['steps_equal'] = {k: (m['steps'][k]['interpretation'] == s['interpretation'] and
                                    (m['steps'][k]['side'] or {}).get('label') == (s['side'] or {}).get('label')) for k, s in v['steps'].items()}
        it[n] = e
    out['interpretation'] = it
    # ---------------------------------------------------------------- controls
    ctl = []
    for s in V.SEEDS:
        ps = co['per_seed'][f'S{s}']
        for view in V.VIEWS:
            a = mine['controls']['ORIGINAL'][f'S{s}|OUT_HGB|{view}']
            b = ps['controls_cv'][f'source_run|{view}']
            ctl.append(max(num_diff(a['dev_shuffled'], b['shuffled']), num_diff(a['dev_planted'], b['planted']),
                           num_diff(a['av_shuffled'], b['av_shuffled']), num_diff(a['av_planted'], b['av_planted'])))
            a = mine['controls']['ENHANCED'][f'S{s}|OUT_HGB|{view}']
            b = ps['controls_cv'][f'this_run|{view}']
            ctl.append(max(num_diff(a['dev_shuffled'], b['shuffled']), num_diff(a['dev_planted'], b['planted'])))
            for name in ('ET5', 'HGB31', 'MLP6432', 'CATI1'):
                a = mine['controls']['ENHANCED'][f'S{s}|{name}|{view}']
                b = ps['controls_diagnostic'][f'{name}|{view}']
                ctl.append(max(num_diff(a['dev_shuffled'], b['shuffled']), num_diff(a['dev_planted'], b['planted'])))
    cv_in = {k: v['in_band'] for k, v in mine['controls']['ENHANCED'].items()}
    ext = {k: (min(v['dev_planted'], v['av_planted']), max(abs(v['dev_shuffled'] - 0.5), abs(v['av_shuffled'] - 0.5))) for k, v in mine['controls']['ENHANCED'].items()}
    out['controls'] = {'n_compared': len(ctl), 'max_abs_auc_diff': max(ctl), 'CV_mine': mine['CV'], 'CV_coordinator': co['control_validity']['CV'],
                       'enhanced_in_band_equal': all(cv_in[f'{k.split("|")[0]}|{k.split("|")[1]}|{k.split("|")[2]}'] == v for k, v in co['control_validity']['ENHANCED_controls_in_band'].items()),
                       'enhanced_min_planted': min(x[0] for x in ext.values()), 'enhanced_max_abs_shuffled_minus_half': max(x[1] for x in ext.values()),
                       'original_min_planted': min(min(v['dev_planted'], v['av_planted']) for v in mine['controls']['ORIGINAL'].values()),
                       'original_max_abs_shuffled_minus_half': max(max(abs(v['dev_shuffled'] - 0.5), abs(v['av_shuffled'] - 0.5)) for v in mine['controls']['ORIGINAL'].values())}
    # ---------------------------------------------------------------- recovery AUCs, winners, reader-class changes
    rec, win_m, rc = [], [], []
    for s in V.SEEDS:
        for svc in V.BANK:
            cs_ = co['per_seed'][f'S{s}']['services'][svc]
            for view in V.VIEWS:
                for tag, key in (('orig', 'recovery_original'), ('enh', 'recovery_enhanced')):
                    m = mine['recovery'][f'{svc}|S{s}|{view}|{tag}']
                    c = cs_[key][view]
                    rec.append(max(num_diff(m['auc'], c['auc']), num_diff(m['se'], c['auc_se'])))
                    w = mine['winners'][f'{svc}|S{s}|{view}']
                    wm = w['orig'] if tag == 'orig' else w['enh'].replace(' [new]', '')
                    if wm != c['winner']:
                        win_m.append([svc, s, view, tag, wm, c['winner']])
                m = mine['reader_class_change'][f'{svc}|S{s}|{view}']
                c = cs_['reader_class_change'][view]
                rc.append(max(num_diff(m['enhanced_minus_original'], c['enhanced_minus_original']), num_diff(m['se'], c['se'])) if m['same_reader'] == c['same_reader'] else float('inf'))
    out['recovery'] = {'n': len(rec), 'max_abs_diff_auc_or_se': max(rec), 'winner_mismatches': win_m}
    out['reader_class_change'] = {'n': len(rc), 'max_abs_diff': max(rc)}
    bc = [max(num_diff(mine['benefit_change'][k]['enhanced_minus_original_benefit'], v['enhanced_minus_original_benefit']), num_diff(mine['benefit_change'][k]['se'], v['se']))
          for k, v in co['benefit_change'].items()]
    out['benefit_change'] = {'n': len(bc), 'max_abs_diff': max(bc)}
    # ---------------------------------------------------------------- reader-matched benefits and variation
    rm = []
    for s in V.SEEDS:
        for k, v in co['per_seed'][f'S{s}']['reader_matched_benefits_descriptive'].items():
            comp, n = k.split('|')
            m = mine['reader_matched_benefits'][f'S{s}|{comp}|{n}']
            rm.append(max(num_diff(m['benefit'], v['benefit']), num_diff(m['se'], v['se'])))
    out['reader_matched_benefits'] = {'n': len(rm), 'max_abs_diff': max(rm)}
    vr = []
    for s in V.SEEDS:
        cv_ = co['per_seed'][f'S{s}']['variation']
        for svc in ('FINALIST', 'NEURAL_RAW', 'H_LJ'):
            for kind in ('VAR', 'AUX'):
                if kind in cv_.get(svc, {}):
                    a = mine['variation'][f'S{s}|{kind}|{svc}']['dev_auc_by_refit']
                    b = cv_[svc][kind]['dev_auc_by_refit']
                    vr.append(max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else float('inf'))
        for k, v in cv_.get('benefit_matched_refits', {}).items():
            kind, ref = k.split('|')
            vr.append(num_diff(mine['variation'][f'S{s}|{kind}|benefit_vs_{ref}']['sd'], v['sd']))
    out['variation'] = {'n': len(vr), 'max_abs_diff': max(vr)}
    # ---------------------------------------------------------------- utility of every service (point gates) and reproduction of the source
    ut = []
    for s in V.SEEDS:
        for svc in V.BANK:
            cs_ = co['per_seed'][f'S{s}']['services'][svc]
            mu = mine['utility_all_services'][f'{svc}|S{s}']
            for k, v in cs_['utility'].items():
                ut.append(num_diff(mu['points'][k], v['point']))
            for g in (0, 1):
                ut.append(num_diff(mu['guards'][f'sex{g}'], cs_['subgroup'][f'sex{g}']['occupation_brier_excess_vs_U0']))
            ut.append(0.0 if (mu['utility_point_pass_12'] == cs_['utility_point_feasible_12_this_seed'] and mu['guard_point_pass_2'] == cs_['guard_point_pass_this_seed']) else float('inf'))
    out['utility_all_services'] = {'n': len(ut), 'max_abs_diff': max(ut)}
    out['original_vs_source_report'] = {'mine': mine.get('original_vs_source_report'), 'coordinator': co['original_reproduces_source']}
    out['refit_dev_auc'] = {'n': len(mine['refit_dev_auc']), 'max_abs_diff': max(abs(v['diff']) for v in mine['refit_dev_auc'].values()),
                            'all_bitwise': all(v['bitwise'] for v in mine['refit_dev_auc'].values())}
    out['critical'] = {'mine': mine['critical'], 'coordinator': co['critical']}
    V.jdump(V.VPRIV / 'check6_compare.json', out)
    import json
    print(json.dumps({k: v for k, v in out.items() if k != 'scorecards'}, indent=1, default=str)[:6000])
    for SC, v in sc.items():
        print(SC, {k: v[k] for k in ('n_endpoints', 'same_keys', 'max_abs_diff', 'worst_key', 'M_steps_equal', 'E_fields_equal', 'CV_equal')})
        print('  cat mismatches', v['categorical_mismatches'][:10])
        print('  claims', {n: (c['mine'], c['equal']) for n, c in v['claims'].items()}, v['first_failing'])


if __name__ == '__main__':
    main()
