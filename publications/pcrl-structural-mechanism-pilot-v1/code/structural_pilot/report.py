"""Report: re-read the already-exposed DEV_REPORT truth (after this campaign's committed REPORTING_LOCK) and evaluate the
endpoints fixed in ANALYSIS_LOCK for the frozen service bank and the frozen readers (exploratory development pilot).

Inference (inherited from the source and the frozen-hybrids follow-up): one nonparametric household bootstrap over
DEV_REPORT households (B and seed from ANALYSIS_LOCK), shared across every service, endpoint and both model seeds. Bounds
are Wald-type: point +/- t(G-1) x bootstrap SE. Claim clauses are one-sided at alpha/3 (alpha = 0.025, three claims, IUT
within each claim). Reporting family: two-sided 95% Bonferroni-t intervals over the enumerated primary roster (K_primary);
a separate secondary descriptive roster (K_secondary). Failing-direction labels come only from the primary simultaneous
intervals. Intervals condition on the fitted defense models and fitted readers; the two seeds share people; the household
bootstrap is not ACS survey-design variance; these people have been used repeatedly in development, so everything is
exploratory and error rates are nominal.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import common as C
from .readers import load_service_bank


def crit(G, level_one_sided):
    from scipy import stats
    return float(stats.t.ppf(1 - level_one_sided, G - 1))


def status_at(kind, L, U, m):
    if kind == 'benefit':
        return 'ESTABLISHED' if L > m else ('ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED')
    if kind == 'no_excess':
        return 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')
    return 'ABOVE_ZERO' if L > m else ('BELOW_ZERO' if U < m else 'UNRESOLVED')


def endpoint(point, draws, spec, c1, cK, family):
    draws = np.asarray(draws, dtype=np.float64)
    se = float(np.std(draws, ddof=1)); kind = spec['kind']
    L, U = float(point - cK * se), float(point + cK * se)
    rec = {'point': float(point), 'se': se, 'kind': kind, 'family': family, 'claim': spec.get('claim'), 'margin': spec.get('margin'),
           'refs': spec['refs'], 'family_lower': L, 'family_upper': U,
           'construction': 'point +/- t(G-1) x paired household-bootstrap SE; conditional on fitted models and readers',
           'finite': bool(np.isfinite(point) and np.isfinite(se)), 'at': {}}
    for m in spec['refs']:
        rec['at'][str(m)] = {'point_minus_m': float(point - m), 'lower_minus_m': L - m, 'upper_minus_m': U - m, 'status_at_m': status_at(kind, L, U, m)}
    if spec.get('claim'):
        m = spec['margin']
        rec['clause_L1'] = float(point - c1 * se); rec['clause_U1'] = float(point + c1 * se)
        rec['clause_pass'] = bool(rec['clause_L1'] > m) if kind == 'benefit' else bool(rec['clause_U1'] <= m)
        rec['point_pass'] = bool(point > m) if kind == 'benefit' else bool(point <= m)
        rec['status'] = status_at(kind, L, U, m)
    else:
        rec['status'] = {str(m): rec['at'][str(m)]['status_at_m'] for m in spec['refs']}
    return rec


def run(out_dir):
    from task_headroom.inference import group_weights
    from task_conditioned_representation.inference import auc, bootstrap_auc
    for name in ('ANALYSIS_LOCK.json', 'SELECTION_LOCK.json', 'REPORTING_LOCK.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'report requires the committed {name}')
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    rlock = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text())
    inf = lock['inference']
    T = C.dev_report_truth('structural-mechanism pilot report: evaluate the ANALYSIS_LOCK endpoints for the frozen service bank and readers')
    y = {t: T[t] for t in C.TASKS}; sex = T['sex']
    W, G = group_weights(T['unit'], seed=inf['bootstrap_seed'], draws=inf['bootstrap_B'])
    c1 = crit(G, inf['alpha_family_one_sided'] / inf['n_claims'])
    cs = crit(G, 0.05 / (2 * inf['K_primary'])); cs2 = crit(G, 0.05 / (2 * inf['K_secondary'])); cm = crit(G, 0.025)
    hh_per_sex = {int(g): int(len(np.unique(T['unit'][sex == g]))) for g in (0, 1)}
    res = {'schema': 'SMP1-dev-report-v1', 'utc': C.now(), 'status': 'EXPLORATORY DEVELOPMENT PILOT (already-exposed DEV_REPORT people)',
           'households': int(G), 'persons': int(len(sex)), 'subgroup_households_per_sex': hh_per_sex,
           'bootstrap': {'B': inf['bootstrap_B'], 'seed': inf['bootstrap_seed'], 'unit': 'DEV_REPORT household', 'shared_across': 'services, endpoints, seeds'},
           'critical': {'t_one_sided_alpha_over_3': c1, 't_simultaneous_primary': cs, 'K_primary': inf['K_primary'], 't_simultaneous_secondary': cs2,
                        'K_secondary': inf['K_secondary'], 't_marginal_95': cm},
           'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'), 'reporting_lock_sha256': C.sha(C.RESULTS / 'REPORTING_LOCK.json'), 'per_seed': {}}
    auc_pt, auc_dr = {}, {}
    util = {}
    roster = lock['readers']['attack_roster']; util_services = lock['utility_services']
    for s in C.SEEDS:
        bank, ids, man = load_service_bank(s)
        if not np.array_equal(ids['row_id__DEV_REPORT'], T['row_id']):
            raise AssertionError('bank / DEV_REPORT row order differs')
        rd = C.PRIV / 'runs' / f'READERS_{s}'
        if C.sha(rd / 'DEV_REPORT_SCORES.npz') != rlock['readers'][f'S{s}']['dev_report_scores_sha256']:
            raise PermissionError('reader scores differ from REPORTING_LOCK')
        rr = json.loads((rd / 'READERS_SEED.json').read_text())
        with np.load(rd / 'DEV_REPORT_SCORES.npz') as z:
            scores = {k: z[k].copy() for k in z.files}
        if not np.array_equal(scores['row_id'], T['row_id']):
            raise AssertionError('score / DEV_REPORT row order differs')
        srec = {'services': {}, 'controls': {}, 'learning_curve': {}}
        for svc in roster:
            v = {'recovery': {}}
            for view in ('income', 'occupation', 'pair'):
                w = rr['services'][svc]['winner'][view]
                sc = scores[w['reader_id']]
                auc_pt[(svc, s, view)] = auc(sex, sc); auc_dr[(svc, s, view)] = bootstrap_auc(sex, sc, W)
                se = float(np.std(auc_dr[(svc, s, view)], ddof=1))
                v['recovery'][view] = {'winner': w['candidate'], 'reader_id': w['reader_id'], 'auc': auc_pt[(svc, s, view)], 'auc_se': se,
                                       'marginal95': [auc_pt[(svc, s, view)] - cm * se, auc_pt[(svc, s, view)] + cm * se]}
            v['recovery']['pair_reader_slate_auc_descriptive'] = {k.split('__', 2)[2]: auc(sex, a) for k, a in scores.items() if k.startswith(f'PAIR__{svc}__')}
            srec['services'][svc] = v
        for svc in util_services:
            v = srec['services'].setdefault(svc, {})
            v['utility'] = {}; v['task_summary'] = {}; v['subgroup'] = {}
            for t in C.TASKS:
                k = C.K[t]; vv = C.task_vectors(y[t], bank[svc]['DEV_REPORT'][t], k)
                v['task_summary'][t] = {m: float(vv[m].mean()) for m in vv}
                for ref in ('U0', 'Ucal'):
                    rv = C.task_vectors(y[t], bank[ref]['DEV_REPORT'][t], k)
                    for metric, d in {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'], 'brier_excess': vv['brier'] - rv['brier']}.items():
                        dr = (W * d[None, :]).sum(1) / W.sum(1)
                        util[(svc, s, t, ref, metric)] = (float(d.mean()), dr)
                        v['utility'][f'{t}|{ref}|{metric}'] = {'point': float(d.mean()), 'se': float(np.std(dr, ddof=1)), 'cap': C.CAPS[metric], 'point_pass': bool(d.mean() <= C.CAPS[metric])}
            rv = C.task_vectors(y['occupation'], bank['U0']['DEV_REPORT']['occupation'], 5)
            vv = C.task_vectors(y['occupation'], bank[svc]['DEV_REPORT']['occupation'], 5)
            for g in (0, 1):
                m = sex == g; d = vv['brier'] - rv['brier']; Wg = W[:, m]
                dr = (Wg * d[m][None, :]).sum(1) / Wg.sum(1)
                util[(svc, s, 'subgroup', g)] = (float(d[m].mean()), dr)
                v['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d[m].mean()), 'se': float(np.std(dr, ddof=1)), 'group_persons': int(m.sum()),
                                            'group_households': hh_per_sex[g]}
        nom = lock['readers']['control_wire']
        for view in ('income', 'occupation', 'pair'):
            pl = np.where(sex == 1, scores[f'CONTROL__{view}__PLANTED_1'], scores[f'CONTROL__{view}__PLANTED_0'])
            srec['controls'][view] = {'wire': nom, 'shuffled_null_auc': auc(sex, scores[f'CONTROL__{view}__SHUFFLED']), 'planted_bit_auc': auc(sex, pl),
                                      'planted_attack_validation_auc': rr['controls'][view]['planted_attack_validation_auc'],
                                      'shuffled_attack_validation_auc': rr['controls'][view]['shuffled_attack_validation_auc']}
        for arm, a in rr['learning_curve']['arms'].items():
            sc = scores[a['winner']['reader_id']]
            auc_pt[(arm, s, 'pair_lc')] = auc(sex, sc); auc_dr[(arm, s, 'pair_lc')] = bootstrap_auc(sex, sc, W)
            srec['learning_curve'][arm] = {'winner': a['winner']['candidate'], 'auc': auc_pt[(arm, s, 'pair_lc')],
                                           'household_fraction': rr['learning_curve']['household_fraction']}
        res['per_seed'][f'S{s}'] = srec

    def evaluate(spec):
        e = spec['estimand']; s = spec['seed']
        if e['type'] == 'auc_diff':
            return auc_pt[(e['plus'], s, e['view'])] - auc_pt[(e['minus'], s, e['view'])], auc_dr[(e['plus'], s, e['view'])] - auc_dr[(e['minus'], s, e['view'])]
        if e['type'] == 'utility':
            return util[(e['service'], s, e['task'], e['reference'], e['metric'])]
        if e['type'] == 'subgroup_occupation_brier_vs_U0':
            return util[(e['service'], s, 'subgroup', e['group'])]
        raise ValueError(f'unknown estimand {e}')
    E = {sp['key']: endpoint(*evaluate(sp), sp, c1, cs, f"primary (K = {inf['K_primary']})") for sp in lock['primary_roster']}
    E2 = {sp['key']: endpoint(*evaluate(sp), sp, c1, cs2, f"secondary descriptive (K = {inf['K_secondary']})") for sp in lock['secondary_roster']}
    if len(E) != inf['K_primary'] or len(E2) != inf['K_secondary'] or not all(v['finite'] for v in list(E.values()) + list(E2.values())):
        raise SystemExit('HALT before any decision: roster size or finiteness check failed')
    res['endpoints_primary'] = E; res['endpoints_secondary'] = E2
    # controls (pipeline validity; never evidence of privacy)
    cr = lock['controls']; lo, hi = cr['null_band']
    vals = [c for s in C.SEEDS for c in res['per_seed'][f'S{s}']['controls'].values()]
    planted = all(c['planted_bit_auc'] >= cr['planted_min_auc'] and c['planted_attack_validation_auc'] >= cr['planted_min_auc'] for c in vals)
    null = all(lo <= c['shuffled_null_auc'] <= hi and lo <= c['shuffled_attack_validation_auc'] <= hi for c in vals)
    valid = bool(planted and null and len(vals) == 6)
    res['controls'] = {'PLANTED_PASS': bool(planted), 'NULL_PASS': bool(null), 'CONTROLS_VALID': valid}
    # claims (kept separate)
    ck = lock['claim_keys']
    gate24 = all(E[k]['point_pass'] for k in ck['utility_24'])
    ms_ok = bool(sel['nominee']['model_select_point_feasible_24'])
    A = {'coalition_benefit_clause_pass': {k: E[k]['clause_pass'] for k in ck['recipe_coalition']},
         'recipient_worsening_clause_pass': {k: E[k]['clause_pass'] for k in ck['recipe_worsening']}, 'utility_point_gate_24': gate24}
    estA = all(A['coalition_benefit_clause_pass'].values()) and all(A['recipient_worsening_clause_pass'].values()) and gate24 and ms_ok
    A['decision'] = 'CONTROL_FAILURE' if not valid else ('RECIPE_EFFECT_ESTABLISHED' if estA else 'RECIPE_EFFECT_NOT_ESTABLISHED')
    B = {'eligible_set': lock['claims']['B']['eligible_set'], 'clause_pass_vs_0': {k: E[k]['clause_pass'] for k in ck['competitive']},
         'clause_L1_gt_0.020': {k: E[k]['clause_L1'] > 0.020 for k in ck['competitive']},
         'status_at_0_and_0.020': {k: {m: E[k]['at'][m]['status_at_m'] for m in E[k]['at']} for k in ck['competitive']}}
    if not valid:
        B['decision'] = 'CONTROL_FAILURE'
    elif not (ms_ok and gate24):
        B['decision'] = 'NOMINEE_FAILS_UTILITY_GATE'
    else:
        B['decision'] = 'SUPERIOR_TO_ALL_ELIGIBLE' if all(B['clause_pass_vs_0'].values()) else 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'
    B['nested_material_superiority_0.020'] = ('MATERIAL_SUPERIORITY_ESTABLISHED' if B['decision'] == 'SUPERIOR_TO_ALL_ELIGIBLE' and all(B['clause_L1_gt_0.020'].values())
                                              else 'MATERIAL_SUPERIORITY_NOT_ESTABLISHED')
    sg = [E[k] for k in ck['subgroup']]
    Cc = {'guards_clause_pass': {k: E[k]['clause_pass'] for k in ck['subgroup']}, 'guards_status': {k: E[k]['status'] for k in ck['subgroup']},
          'decision': 'SUBGROUP_QUALIFIED' if all(v['clause_pass'] for v in sg) else ('SUBGROUP_HARM' if any(v['family_lower'] > 0.005 for v in sg) else 'SUBGROUP_UNRESOLVED')}
    res['claims'] = {'A_RECIPE_BENEFIT': A, 'B_COMPETITIVE': B, 'C_SUBGROUP': Cc,
                     'D_JOINT_COMPONENT': lock['claims'].get('D_joint', 'NOT_CLAIMED'), 'E_NOVELTY': 'judged against prior work only; see the prior-work review'}
    ut_status = {k: E[k]['status'] for k in ck['utility_24']}
    res['separate_status_fields'] = {
        'point_feasibility': {'DEV_REPORT': 'POINT_FEASIBLE_24' if gate24 else 'POINT_INFEASIBLE', 'DEV_REPORT_failed_keys': [k for k in ck['utility_24'] if not E[k]['point_pass']],
                              'MODEL_SELECT': 'POINT_FEASIBLE_24' if ms_ok else 'POINT_INFEASIBLE'},
        'confidence_status': {'status': ('CONFIDENCE_FEASIBLE_24' if all(v == 'PASS' for v in ut_status.values()) else 'ESTABLISHED_UTILITY_VIOLATION'
                                         if any(v == 'ESTABLISHED_VIOLATION' for v in ut_status.values()) else 'CONFIDENCE_UNRESOLVED'),
                              'counts': {x: sum(v == x for v in ut_status.values()) for x in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
                              'clause_bounds_passing_of_24': int(sum(E[k]['clause_pass'] for k in ck['utility_24']))},
        'subgroup_status': Cc['decision']}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'DEV_REPORT_RESULTS.json', 'x') as f:
        json.dump(res, f, indent=2, sort_keys=True, default=float)
    cols = ['key', 'family', 'claim', 'kind', 'margin', 'point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1', 'clause_pass', 'point_pass', 'status']
    for name, EE in (('PRIMARY_ENDPOINTS.csv', E), ('SECONDARY_ENDPOINTS.csv', E2)):
        with open(out / name, 'x', newline='') as f:
            w = csv.writer(f); w.writerow(cols)
            for k, v in EE.items():
                w.writerow([k] + [json.dumps(v.get(c)) if isinstance(v.get(c), dict) else v.get(c) for c in cols[1:]])
    with open(out / 'SERVICE_TABLE.csv', 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'service'] + [f'{v}_{x}' for v in ('pair', 'income', 'occupation') for x in ('auc', 'se', 'winner')])
        for s in C.SEEDS:
            for svc in roster:
                rc = res['per_seed'][f'S{s}']['services'][svc]['recovery']
                w.writerow([s, svc] + [x for v in ('pair', 'income', 'occupation') for x in (rc[v]['auc'], rc[v]['auc_se'], rc[v]['winner'])])
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    r = run(ap.parse_args().out)
    print(json.dumps({'claims': {k: (v['decision'] if isinstance(v, dict) else v) for k, v in r['claims'].items()}, 'controls': r['controls'],
                      'separate': r['separate_status_fields']}, indent=1, default=str))
