"""Verifier check 4: occupation-NLL matching, comparator selection, same-count roles, nearest diagnostics, effort (own code).

Inputs: the verifier's own per-release MODEL_SELECT measurements (check3_model_select.json: expenditure, 12-clause pass, proxy,
task summaries) and the checkpoint label -> release map (TRAJ_SEED.json, which check 2 reproduced from own labels and tables).
Rules re-applied from PROTOCOL.md section 7 and the FROZEN DESIGN; every role, status, candidate list, effort count, coverage
figure, service alias and SCM-minus-role difference is compared with SELECTION_LOCK.json. No data file is read beyond these.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

TOL = 0.001
GAMMAS = ('IBSI_g0.5', 'IBSI_g1.0', 'IBSI_g2.0')
SHUFS = tuple(f'SHUF_r{r}' for r in range(5))
ROLE_ORDER = ('SCM', 'ORD_NLLM', 'ORD_CNT', 'IBSI_NLLM', 'IBSI_CNT', 'SHUF_NLLM', 'NEURAL_RAW', 'H_LJ', 'C0', 'HIST_SCM')


def main():
    t0 = time.process_time()
    sel = V.jload(V.RES / 'SELECTION_LOCK.json')
    c3 = V.jload(V.VDIR / 'check3_model_select.json')
    meas = {s: c3['own_measurements'][str(s)] for s in V.SEEDS}
    cps = {s: V.jload(V.PRIV / 'runs' / f'TRAJ_{s}' / 'TRAJ_SEED.json')['checkpoints'] for s in V.SEEDS}
    t_star = c3['nomination']['own_t_star']; lab_star = f'N{t_star:.3f}'
    scm = {s: cps[s]['SCM'][lab_star]['release'] for s in V.SEEDS}
    e_star = {s: meas[s][scm[s]]['expenditure'] for s in V.SEEDS}
    out = {'check': 4, 'title': 'matching and comparator selection', 'discrepancies': [], 'own': {}, 'comparison': {}}

    def candidates(s, name):
        rows = {}
        for label, cp in cps[s][name].items():
            if cp['release'] is None:
                continue
            r = rows.setdefault(cp['release'], {'release': cp['release'], 'labels': [], 'm': cp['m'], 'cells': None})
            r['labels'].append(label); r['m'] = min(r['m'], cp['m'])
        out_ = []
        for rid, r in rows.items():
            mm = meas[s][rid]; d = mm['expenditure'] - e_star[s]
            out_.append({**r, 'cells': 512 - r['m'], 'proxy': mm['proxy'], 'delta': d, 'abs_delta': abs(d), 'match': abs(d) <= TOL, 'pass12': mm['point_pass_12']})
        return sorted(out_, key=lambda c: c['m'])

    def best(c):
        el = [x for x in c if x['match'] and x['pass12']]
        return min(el, key=lambda x: (x['proxy'], x['abs_delta'], x['m'])) if el else None

    def nearest(c):
        el = [x for x in c if x['pass12']]
        return min(el, key=lambda x: (x['abs_delta'], x['proxy'], x['m'])) if el else None

    roles = {s: {} for s in V.SEEDS}; effort = {}
    cand = {name: {s: candidates(s, name) for s in V.SEEDS} for name in ('ORD',) + GAMMAS + SHUFS}
    effort['ORD_NLLM'] = {s: len(cand['ORD'][s]) for s in V.SEEDS}
    margins = {}
    for s in V.SEEDS:
        b = best(cand['ORD'][s])
        el = sorted([x for x in cand['ORD'][s] if x['match'] and x['pass12']], key=lambda x: (x['proxy'], x['abs_delta'], x['m']))
        margins[f'ORD_NLLM_s{s}'] = {'eligible_matched_candidates': [(x['release'], x['labels'], round(x['proxy'], 6), x['delta']) for x in el],
                                     'proxy_gap_to_runner_up': (el[1]['proxy'] - el[0]['proxy']) if len(el) > 1 else None}
        roles[s]['ORD_NLLM'] = ({**b, 'status': 'MATCHED'} if b else
                                ({**nearest(cand['ORD'][s]), 'status': 'NEAREST_DIAGNOSTIC_NOT_MATCHED'} if nearest(cand['ORD'][s]) else {'release': None, 'status': 'MISSING'}))

    def common(names, label):
        effort[label] = {n: {s: len(cand[n][s]) for s in V.SEEDS} for n in names}
        adm = []
        for i, n in enumerate(names):
            picks = {s: best(cand[n][s]) for s in V.SEEDS}
            if all(picks.values()):
                adm.append((float(np.mean([picks[s]['proxy'] for s in V.SEEDS])), i, n, picks))
        if adm:
            sc, i, n, picks = min(adm, key=lambda x: (x[0], x[1]))
            srt = sorted(adm, key=lambda x: (x[0], x[1]))
            margins[label] = {'admissible': [(a[2], round(a[0], 6)) for a in srt], 'gap_to_runner_up': (srt[1][0] - srt[0][0]) if len(srt) > 1 else None}
            return n, {s: {**picks[s], 'status': 'MATCHED'} for s in V.SEEDS}, True, [a[2] for a in adm]
        near = []
        for i, n in enumerate(names):
            picks = {s: nearest(cand[n][s]) for s in V.SEEDS}
            if all(picks.values()):
                near.append((max(picks[s]['abs_delta'] for s in V.SEEDS), float(np.mean([picks[s]['proxy'] for s in V.SEEDS])), i, n, picks))
        if near:
            _, _, i, n, picks = min(near, key=lambda x: (x[0], x[1], x[2]))
            return n, {s: {**picks[s], 'status': 'NEAREST_DIAGNOSTIC_NOT_MATCHED'} for s in V.SEEDS}, False, []
        return None, {s: {'release': None, 'status': 'MISSING'} for s in V.SEEDS}, False, []
    ib = common(GAMMAS, 'IBSI_NLLM'); sh = common(SHUFS, 'SHUF_NLLM')
    for s in V.SEEDS:
        roles[s]['IBSI_NLLM'] = {**ib[1][s], 'trajectory': ib[0]}; roles[s]['SHUF_NLLM'] = {**sh[1][s], 'trajectory': sh[0]}
    lab_s = f'S{t_star:.3f}'
    for s in V.SEEDS:
        cp = cps[s]['ORD'][lab_s]
        roles[s]['ORD_CNT'] = ({'release': cp['release'], 'm': cp['m'], 'cells': cp['cells'], 'pass12': meas[s][cp['release']]['point_pass_12'],
                                'proxy': meas[s][cp['release']]['proxy'], 'status': 'SAME_COUNT'} if cp['release'] else {'release': None, 'status': 'UNMATCHED_CAPACITY'})
    ibc = []
    for i, n in enumerate(GAMMAS):
        rids = {s: cps[s][n][lab_s]['release'] for s in V.SEEDS}
        ibc.append({'trajectory': n, 'i': i, 'releases': rids, 'eligible_24': all(meas[s][rids[s]]['point_pass_12'] for s in V.SEEDS),
                    'mean_proxy': float(np.mean([meas[s][rids[s]]['proxy'] for s in V.SEEDS])),
                    'worst_normalized_24': max(meas[s][rids[s]]['max_normalized_12'] for s in V.SEEDS)})
    effort['IBSI_CNT'] = len(ibc)
    el = [x for x in ibc if x['eligible_24']]
    pick = min(el, key=lambda x: (x['mean_proxy'], x['i'])) if el else min(ibc, key=lambda x: (x['worst_normalized_24'], x['i']))
    margins['IBSI_CNT'] = [(x['trajectory'], x['eligible_24'], round(x['mean_proxy'], 6), round(x['worst_normalized_24'], 4)) for x in ibc]
    for s in V.SEEDS:
        roles[s]['IBSI_CNT'] = {'release': pick['releases'][s], 'trajectory': pick['trajectory'], 'status': 'SAME_COUNT' if pick['eligible_24'] else 'SAME_COUNT_INELIGIBLE_FALLBACK'}
        roles[s]['SCM'] = {'release': scm[s], 'status': c3['nomination']['own_status']}
        roles[s]['C0'] = {'release': cps[s]['SCM']['C512']['release'], 'status': 'REFERENCE'}
        roles[s]['HIST_SCM'] = {'release': cps[s]['SCM']['TERM']['release'], 'status': 'HISTORICALLY_SELECTED_DIAGNOSTIC'}
        for n in ('NEURAL_RAW', 'H_LJ'):
            fm = meas[s][f'FROZEN_{n}']
            roles[s][n] = {'release': f'FROZEN_{n}', 'status': 'FROZEN_REFERENCE', 'proxy': fm['proxy'], 'delta': fm['expenditure'] - e_star[s], 'pass12': fm['point_pass_12']}
        sc = meas[s][scm[s]]['task_summary']
        for rn, r in roles[s].items():
            rid = r.get('release')
            if rid and not rid.startswith('FROZEN_'):
                mm = meas[s][rid]
                r['expenditure'] = mm['expenditure']; r['delta'] = mm['expenditure'] - e_star[s]; r['proxy'] = mm['proxy']; r['pass12'] = mm['point_pass_12']
                r['SCM_minus_role'] = {f'{t}_{m}': sc[t][m] - mm['task_summary'][t][m] for t in V.TASKS for m in ('nll', 'brier', 'accuracy')}
                r['income_qualifier'] = 'INCOME_EXPENDITURE_NOT_MATCHED' if r['SCM_minus_role']['income_nll'] > TOL else 'income NLL within 0.001 or lower for SCM'
    # services
    services = {}
    for s in V.SEEDS:
        by = {}
        for rn in ROLE_ORDER:
            rid = roles[s][rn].get('release')
            if rid is not None:
                by.setdefault(rid, []).append(rn)
        services[s] = {'service_of_role': {rn: names[0] for names in by.values() for rn in names}, 'aliases': {names[0]: names for names in by.values()},
                       'missing_roles': [rn for rn in ROLE_ORDER if roles[s][rn].get('release') is None]}
    # coverage
    coverage = {}
    for n in ('ORD',) + GAMMAS + SHUFS:
        coverage[n] = {}
        for s in V.SEEDS:
            xs = sorted(meas[s][c['release']]['expenditure'] for c in cand[n][s])
            br = [b - a for a, b in zip(xs, xs[1:]) if a <= e_star[s] <= b]
            coverage[n][s] = {'n_candidates': len(xs), 'largest_gap_bracketing_e_star': max(br) if br else None, 'max_candidate_expenditure': max(xs),
                              'n_within_tolerance': sum(abs(x - e_star[s]) <= TOL for x in xs)}
    # ---------------- comparison with SELECTION_LOCK
    cmpres = {}
    R = sel['roles']
    for s in V.SEEDS:
        for rn in ROLE_ORDER:
            mine, rec = roles[s][rn], R[str(s)][rn]
            item = {'release_equal': mine.get('release') == rec.get('release'), 'status_equal': mine.get('status') == rec.get('status')}
            if 'trajectory' in mine:
                item['trajectory_equal'] = mine['trajectory'] == rec.get('trajectory')
            for f_m, f_r in (('delta', 'delta'), ('proxy', 'proxy'), ('expenditure', 'expenditure')):
                if mine.get(f_m) is not None and rec.get(f_r) is not None:
                    item[f'{f_m}_abs_diff'] = abs(mine[f_m] - rec[f_r])
            if 'pass12' in mine and 'point_pass_12' in rec:
                item['pass12_equal'] = mine['pass12'] == rec['point_pass_12']
            if 'SCM_minus_role' in mine and 'SCM_minus_role_MODEL_SELECT' in rec:
                item['SCM_minus_role_max_abs_diff'] = max(abs(mine['SCM_minus_role'][k_] - rec['SCM_minus_role_MODEL_SELECT'][k_]) for k_ in mine['SCM_minus_role'])
                item['income_qualifier_equal'] = mine['income_qualifier'] == rec['income_expenditure_qualifier']
            if rn in ('ORD_NLLM', 'IBSI_NLLM', 'SHUF_NLLM', 'ORD_CNT', 'IBSI_CNT') and mine.get('release'):
                item['labels_equal'] = sorted(mine.get('labels', rec.get('labels', []))) == sorted(rec.get('labels', [])) if 'labels' in mine else None
                item['m_equal'] = (mine.get('m') == rec.get('m')) if 'm' in mine else None
                item['cells_equal'] = (mine.get('cells') == rec.get('cells')) if 'cells' in mine else None
            cmpres[f's{s}|{rn}'] = item
    m_rec = sel['matching']
    cand_cmp = {}
    for n in ('ORD',) + GAMMAS + SHUFS:
        for s in V.SEEDS:
            a = cand[n][s]; b = m_rec['candidates'][n][str(s)]
            cand_cmp[f'{n}|s{s}'] = {'n_equal': len(a) == len(b),
                                     'releases_labels_m_equal': [(x['release'], sorted(x['labels']), x['m'], x['cells']) for x in a]
                                     == [(y['release'], sorted(y['labels']), y['m'], y['cells']) for y in b],
                                     'match_pass_equal': [(x['match'], x['pass12']) for x in a] == [(y['match'], y['point_pass_12']) for y in b],
                                     'max_abs_delta_diff': max(abs(x['delta'] - y['delta']) for x, y in zip(a, b)) if len(a) == len(b) else None,
                                     'max_abs_proxy_diff': max(abs(x['proxy'] - y['proxy']) for x, y in zip(a, b)) if len(a) == len(b) else None}
    eff_cmp = {'ORD_NLLM': {str(s): effort['ORD_NLLM'][s] for s in V.SEEDS} == sel['selection_effort']['ORD_NLLM'],
               'IBSI_NLLM': {n: {str(s): v for s, v in d.items()} for n, d in effort['IBSI_NLLM'].items()} == sel['selection_effort']['IBSI_NLLM'],
               'SHUF_NLLM': {n: {str(s): v for s, v in d.items()} for n, d in effort['SHUF_NLLM'].items()} == sel['selection_effort']['SHUF_NLLM'],
               'IBSI_CNT': effort['IBSI_CNT'] == sel['selection_effort']['IBSI_CNT']}
    cov_cmp = {}
    for n in coverage:
        for s in V.SEEDS:
            a = coverage[n][s]; b = sel['matching_coverage'][n][str(s)]
            cov_cmp[f'{n}|s{s}'] = {'n_equal': a['n_candidates'] == b['n_candidates'],
                                    'gap_abs_diff': None if (a['largest_gap_bracketing_e_star'] is None or b['largest_gap_bracketing_e_star'] is None)
                                    else abs(a['largest_gap_bracketing_e_star'] - b['largest_gap_bracketing_e_star']),
                                    'gap_both_none': a['largest_gap_bracketing_e_star'] is None and b['largest_gap_bracketing_e_star'] is None}
    match_cmp = {'ORD_NLLM_available': {str(s): roles[s]['ORD_NLLM']['status'] == 'MATCHED' for s in V.SEEDS} == m_rec['ORD_NLLM_available'],
                 'IBSI_gamma': ib[0] == m_rec['IBSI_NLLM']['gamma_trajectory'], 'IBSI_admissible': ib[3] == m_rec['IBSI_NLLM']['admissible'],
                 'IBSI_available_both': ib[2] == m_rec['IBSI_NLLM']['available_both_seeds'],
                 'SHUF_r': sh[0] == m_rec['SHUF_NLLM']['r_trajectory'], 'SHUF_admissible': sh[3] == m_rec['SHUF_NLLM']['admissible'],
                 'SHUF_available_both': sh[2] == m_rec['SHUF_NLLM']['available_both_seeds'],
                 'ORD_CNT_available_both': all(roles[s]['ORD_CNT']['release'] is not None for s in V.SEEDS) == m_rec['ORD_CNT_available_both_seeds'],
                 'IBSI_CNT_trajectory': pick['trajectory'] == m_rec['IBSI_CNT']['trajectory'], 'IBSI_CNT_eligible': pick['eligible_24'] == m_rec['IBSI_CNT']['eligible_24'],
                 'IBSI_CNT_releases': {str(s): v for s, v in pick['releases'].items()} == m_rec['IBSI_CNT']['releases']}
    svc_cmp = {str(s): ({k_: v for k_, v in services[s].items()} == sel['services'][str(s)]) for s in V.SEEDS}
    out['own'] = {'t_star': t_star, 'e_star': e_star, 'K_star': {s: 512 - cps[s]['SCM'][lab_star]['m'] for s in V.SEEDS},
                  'roles': {str(s): {rn: {k_: v for k_, v in r.items() if k_ not in ('SCM_minus_role',)} for rn, r in roles[s].items()} for s in V.SEEDS},
                  'effort': {k_: v for k_, v in effort.items()}, 'coverage': coverage, 'decision_margins': margins}
    out['comparison'] = {'roles': cmpres, 'candidates': cand_cmp, 'effort': eff_cmp, 'coverage': cov_cmp, 'matching': match_cmp, 'services': svc_cmp}
    bad_roles = {k_: v for k_, v in cmpres.items() if not all(x for kk, x in v.items() if isinstance(x, bool))
                 or any(kk.endswith('abs_diff') and x is not None and x > 1e-9 for kk, x in v.items())}
    bad_cand = {k_: v for k_, v in cand_cmp.items() if not (v['n_equal'] and v['releases_labels_m_equal'] and v['match_pass_equal'])
                or (v['max_abs_delta_diff'] or 0) > 1e-12 or (v['max_abs_proxy_diff'] or 0) > 1e-9}
    bad_cov = {k_: v for k_, v in cov_cmp.items() if not v['n_equal'] or (v['gap_abs_diff'] is not None and v['gap_abs_diff'] > 1e-12)}
    out['problems'] = {'roles': bad_roles, 'candidates': bad_cand, 'coverage': bad_cov,
                       'effort': [k_ for k_, v in eff_cmp.items() if not v], 'matching': [k_ for k_, v in match_cmp.items() if not v],
                       'services': [k_ for k_, v in svc_cmp.items() if not v]}
    ok = not any(out['problems'].values())
    out['status'] = 'PASS' if ok else 'FAIL_OR_REVIEW'
    out['cpu_s'] = round(time.process_time() - t0, 2)
    V.write_json('check4_matching.json', out)
    print(json.dumps({'status': out['status'], 'problems': out['problems'], 'own_roles': {s: {rn: (r.get('release'), r.get('status'), r.get('trajectory'),
                                                                                                    None if r.get('delta') is None else round(r['delta'], 6),
                                                                                                    None if r.get('proxy') is None else round(r['proxy'], 5))
                                                                                               for rn, r in rr.items()} for s, rr in out['own']['roles'].items()},
                      'margins': margins, 'effort': effort}, indent=1, default=str))


if __name__ == '__main__':
    main()
