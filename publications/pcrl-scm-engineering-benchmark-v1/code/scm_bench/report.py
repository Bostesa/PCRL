"""Report: one logged re-read of the already-exposed DEV_REPORT truth (after the committed REPORTING_LOCK), then the endpoints
fixed in ANALYSIS_LOCK for the frozen final bank and frozen readers (PROTOCOL.md section 9).

Household bootstrap over DEV_REPORT households (B and seed from ANALYSIS_LOCK), shared across releases, endpoints and seeds.
Wald bounds: point +/- t(G-1) x bootstrap SE. Claim clauses one-sided at alpha/3 (alpha = 0.025). Primary family: two-sided
95% Bonferroni-t over K_primary (fixed; an endpoint whose comparator is absent is reported MISSING and still counted in K).
Conditional on the fitted models and readers; the two seeds share people; not ACS survey-design variance; exploratory.
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
           'family_lower': L, 'family_upper': U, 'finite': bool(np.isfinite(point) and np.isfinite(se)), 'at': {}}
    for m in spec['refs']:
        rec['at'][str(m)] = {'point_minus_m': float(point - m), 'lower_minus_m': L - m, 'upper_minus_m': U - m, 'status_at_m': status_at(kind, L, U, m)}
    m = spec['margin']
    rec['clause_L1'] = float(point - c1 * se); rec['clause_U1'] = float(point + c1 * se)
    rec['clause_pass'] = bool(rec['clause_L1'] > m) if kind == 'benefit' else bool(rec['clause_U1'] <= m)
    rec['point_pass'] = bool(point > m) if kind == 'benefit' else bool(point <= m)
    rec['status'] = status_at(kind, L, U, m)
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
    T = C.dev_report_truth('SCM engineering benchmark report: evaluate the ANALYSIS_LOCK endpoints for the frozen final bank and readers')
    y = {t: T[t] for t in C.TASKS}; sex = T['sex']
    W, G = group_weights(T['unit'], seed=inf['bootstrap_seed'], draws=inf['bootstrap_B'])
    c1 = crit(G, inf['alpha_family_one_sided'] / inf['n_claims'])
    cs = crit(G, 0.05 / (2 * inf['K_primary'])); cs2 = crit(G, 0.05 / (2 * inf['K_secondary'])); cm = crit(G, 0.025)
    hh = {int(g): int(len(np.unique(T['unit'][sex == g]))) for g in (0, 1)}
    res = {'schema': 'SCM1-dev-report-v1', 'utc': C.now(), 'status': 'EXPLORATORY DEVELOPMENT EVIDENCE (already-exposed DEV_REPORT people)',
           'households': int(G), 'persons': int(len(sex)), 'subgroup_households_per_sex': hh,
           'critical': {'t_one_sided_alpha_over_3': c1, 't_simultaneous_primary': cs, 'K_primary': inf['K_primary'], 't_simultaneous_secondary': cs2,
                        'K_secondary': inf['K_secondary'], 't_marginal_95': cm},
           'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'), 'selection_lock_sha256': C.sha(C.RESULTS / 'SELECTION_LOCK.json'),
           'reporting_lock_sha256': C.sha(C.RESULTS / 'REPORTING_LOCK.json'), 'per_seed': {}}
    auc_pt, auc_dr, util = {}, {}, {}
    present = set()
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
        srec = {'services': {}, 'controls': {}}
        for svc, sv in rr['services'].items():
            present.add(svc); v = {'recovery': {}, 'recovery_slate7': {}}
            for key, wkey in (('recovery', 'winner'), ('recovery_slate7', 'winner_slate7')):
                for view in ('income', 'occupation', 'pair'):
                    w = sv[wkey][view]; sc = scores[w['reader_id']]
                    tag = view if key == 'recovery' else view + '_slate7'
                    auc_pt[(svc, s, tag)] = auc(sex, sc); auc_dr[(svc, s, tag)] = bootstrap_auc(sex, sc, W)
                    se = float(np.std(auc_dr[(svc, s, tag)], ddof=1))
                    v[key][view] = {'winner': w['candidate'], 'auc': auc_pt[(svc, s, tag)], 'auc_se': se,
                                    'marginal95': [auc_pt[(svc, s, tag)] - cm * se, auc_pt[(svc, s, tag)] + cm * se]}
            v['pair_reader_slate_auc_descriptive'] = {k.split('__', 2)[2]: auc(sex, a) for k, a in scores.items() if k.startswith(f'PAIR__{svc}__')}
            v['utility'] = {}; v['task_summary'] = {}; v['subgroup'] = {}
            for t in C.TASKS:
                k = C.K[t]; vv = C.task_vectors(y[t], bank[svc]['DEV_REPORT'][t], k)
                v['task_summary'][t] = {m_: float(vv[m_].mean()) for m_ in vv}
                for ref in ('U0', 'Ucal'):
                    rv = C.task_vectors(y[t], bank[ref]['DEV_REPORT'][t], k)
                    for metric, d in {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'], 'brier_excess': vv['brier'] - rv['brier']}.items():
                        dr = (W * d[None, :]).sum(1) / W.sum(1)
                        util[(svc, s, t, ref, metric)] = (float(d.mean()), dr)
                        v['utility'][f'{t}|{ref}|{metric}'] = {'point': float(d.mean()), 'se': float(np.std(dr, ddof=1)), 'cap': C.CAPS[metric], 'point_pass': bool(d.mean() <= C.CAPS[metric])}
            rv = C.task_vectors(y['occupation'], bank['U0']['DEV_REPORT']['occupation'], 5)
            vv = C.task_vectors(y['occupation'], bank[svc]['DEV_REPORT']['occupation'], 5)
            for g in (0, 1):
                m_ = sex == g; d = vv['brier'] - rv['brier']; Wg = W[:, m_]
                dr = (Wg * d[m_][None, :]).sum(1) / Wg.sum(1)
                util[(svc, s, 'subgroup', g)] = (float(d[m_].mean()), dr)
                v['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d[m_].mean()), 'se': float(np.std(dr, ddof=1))}
            v['utility_point_feasible_24_this_seed'] = all(x['point_pass'] for x in v['utility'].values())
            srec['services'][svc] = v
        for view in ('income', 'occupation', 'pair'):
            pl = np.where(sex == 1, scores[f'CONTROL__{view}__PLANTED_1'], scores[f'CONTROL__{view}__PLANTED_0'])
            srec['controls'][view] = {'wire': rr['controls'][view]['wire'], 'shuffled_null_auc': auc(sex, scores[f'CONTROL__{view}__SHUFFLED']), 'planted_bit_auc': auc(sex, pl),
                                      'planted_attack_validation_auc': rr['controls'][view]['planted_attack_validation_auc'],
                                      'shuffled_attack_validation_auc': rr['controls'][view]['shuffled_attack_validation_auc']}
        res['per_seed'][f'S{s}'] = srec

    def evaluate(spec):
        e = spec['estimand']; s = spec['seed']
        if e['type'] in ('auc_diff', 'auc_diff_slate7'):
            if e['plus'] not in present or e['minus'] not in present:
                return None
            tag = e['view'] if e['type'] == 'auc_diff' else e['view'] + '_slate7'
            return auc_pt[(e['plus'], s, tag)] - auc_pt[(e['minus'], s, tag)], auc_dr[(e['plus'], s, tag)] - auc_dr[(e['minus'], s, tag)]
        if e['type'] == 'utility':
            return util[(e['service'], s, e['task'], e['reference'], e['metric'])]
        if e['type'] == 'subgroup_occupation_brier_vs_U0':
            return util[(e['service'], s, 'subgroup', e['group'])]
        raise ValueError(f'unknown estimand {e}')
    E, E2 = {}, {}
    for roster, fam, cK, store in ((lock['primary_roster'], f"primary (K = {inf['K_primary']})", cs, E),
                                   (lock['secondary_roster'], f"secondary (K = {inf['K_secondary']})", cs2, E2)):
        for sp in roster:
            v = evaluate(sp)
            store[sp['key']] = ({'status': 'MISSING', 'note': 'comparator absent from the final bank (coverage gap)', 'claim': sp.get('claim')} if v is None
                                else endpoint(*v, sp, c1, cK, fam))
    if len(E) != inf['K_primary'] or len(E2) != inf['K_secondary']:
        raise SystemExit('HALT: roster size check failed')
    res['endpoints_primary'] = E; res['endpoints_secondary'] = E2
    cr = lock['controls']; lo, hi = cr['null_band']
    vals = [c for s in C.SEEDS for c in res['per_seed'][f'S{s}']['controls'].values()]
    planted = all(c['planted_bit_auc'] >= cr['planted_min_auc'] and c['planted_attack_validation_auc'] >= cr['planted_min_auc'] for c in vals)
    null = all(lo <= c['shuffled_null_auc'] <= hi and lo <= c['shuffled_attack_validation_auc'] <= hi for c in vals)
    valid = bool(planted and null and len(vals) == 6)
    res['controls'] = {'PLANTED_PASS': bool(planted), 'NULL_PASS': bool(null), 'CONTROLS_VALID': valid}
    ck = lock['claim_keys']
    ok = lambda k: E[k].get('clause_pass') is True
    gate24 = all(E[k].get('point_pass') for k in ck['utility_24'])
    eligible = bool(sel['nominee']['model_select_point_feasible_24'])
    A = {'coalition_benefit_clause_pass': {k: E[k].get('clause_pass') for k in ck['recipe_coalition']},
         'recipient_worsening_clause_pass': {k: E[k].get('clause_pass') for k in ck['recipe_worsening']}, 'dev_utility_point_gate_24': gate24,
         'twin_matched_capacity': sel.get('twin_matched_capacity')}
    if not valid:
        A['decision'] = 'CONTROL_FAILURE'
    elif not eligible:
        A['decision'] = 'NOMINEE_INELIGIBLE'
    elif not sel.get('twin_matched_capacity'):
        A['decision'] = 'UNMATCHED_CAPACITY'          # twin stopped above the nominee's cell count on a seed: not established
    else:
        A['decision'] = 'RECIPE_EFFECT_ESTABLISHED' if all(ok(k) for k in ck['recipe_coalition'] + ck['recipe_worsening']) and gate24 else 'RECIPE_EFFECT_NOT_ESTABLISHED'
    # Attribution: tested in fixed sequence inside A's alpha/3 slice, only if A is established; otherwise descriptive.
    ibsi_keys = [k for k in ck['attribution'] if 'vs_IBSI' in k]; shuf_keys = [k for k in ck['attribution'] if 'vs_SHUF' in k]
    parts = {'IBSI': ('COMPARATOR_INELIGIBLE' if not sel['ibsi']['model_select_point_feasible_24'] else
                      ('PASS' if all(ok(k) for k in ibsi_keys) else 'FAIL')),
             'SHUF': ('MISSING' if any(E[k].get('status') == 'MISSING' for k in shuf_keys) else ('PASS' if all(ok(k) for k in shuf_keys) else 'FAIL'))}
    if A['decision'] != 'RECIPE_EFFECT_ESTABLISHED':
        att_dec = 'NOT_TESTED_RECIPE_NOT_ESTABLISHED'
    elif any(v in ('COMPARATOR_INELIGIBLE', 'MISSING') for v in parts.values()):
        att_dec = 'INCOMPLETE: ' + ', '.join(f'{k} {v}' for k, v in parts.items() if v in ('COMPARATOR_INELIGIBLE', 'MISSING'))
    else:
        att_dec = 'ATTRIBUTION_ESTABLISHED' if all(v == 'PASS' for v in parts.values()) else 'ATTRIBUTION_NOT_ESTABLISHED'
    ATT = {'parts': parts, 'clause_pass_vs_0': {k: E[k].get('clause_pass') for k in ck['attribution']}, 'decision': att_dec,
           'rule': "sequential inside claim A's alpha/3 slice; descriptive unless A is established; an ineligible comparator is never counted as beaten"}
    B = {'eligible_set': lock['claims']['B']['eligible_set'], 'clause_pass_vs_0': {k: E[k].get('clause_pass') for k in ck['competitive']},
         'clause_L1_gt_0.020': {k: (E[k].get('clause_L1', -1) > 0.020) for k in ck['competitive']}}
    B['decision'] = ('CONTROL_FAILURE' if not valid else 'NOMINEE_INELIGIBLE' if not eligible else 'NOMINEE_FAILS_DEV_UTILITY_GATE' if not gate24 else
                     'SUPERIOR_TO_ALL_ELIGIBLE' if all(ok(k) for k in ck['competitive']) else 'NOT_SUPERIOR_TO_ALL_ELIGIBLE')
    B['nested_material_superiority_0.020'] = 'MATERIAL_SUPERIORITY_ESTABLISHED' if B['decision'] == 'SUPERIOR_TO_ALL_ELIGIBLE' and all(B['clause_L1_gt_0.020'].values()) else 'NOT_ESTABLISHED'
    sg = [E[k] for k in ck['subgroup']]
    Cc = {'guards_clause_pass': {k: E[k]['clause_pass'] for k in ck['subgroup']}, 'guards_status': {k: E[k]['status'] for k in ck['subgroup']},
          'decision': 'SUBGROUP_QUALIFIED' if all(v['clause_pass'] for v in sg) else ('SUBGROUP_HARM' if any(v['family_lower'] > 0.005 for v in sg) else 'SUBGROUP_UNRESOLVED')}
    ut = {k: E[k]['status'] for k in ck['utility_24']}
    res['claims'] = {'A_RECIPE_VS_TWIN': A, 'ATTRIBUTION_VS_PUB_IBSI_AND_SHUF': ATT, 'B_COMPETITIVE': B, 'C_SUBGROUP': Cc,
                     'D_JOINT_COMPONENT': 'NOT_CLAIMED', 'E_NOVELTY': 'NOT_CLAIMED (existing-method engineering benchmark)'}
    res['separate_status_fields'] = {
        'point_feasibility': {'MODEL_SELECT': 'POINT_FEASIBLE_24' if eligible else 'POINT_INFEASIBLE', 'DEV_REPORT': 'POINT_FEASIBLE_24' if gate24 else 'POINT_INFEASIBLE',
                              'DEV_REPORT_failed_keys': [k for k in ck['utility_24'] if not E[k]['point_pass']]},
        'confidence_status': {'counts': {x: sum(v == x for v in ut.values()) for x in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
                              'status': ('CONFIDENCE_FEASIBLE_24' if all(v == 'PASS' for v in ut.values()) else 'ESTABLISHED_UTILITY_VIOLATION'
                                         if any(v == 'ESTABLISHED_VIOLATION' for v in ut.values()) else 'CONFIDENCE_UNRESOLVED')},
        'subgroup_status': Cc['decision']}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'DEV_REPORT_RESULTS.json', 'x') as f:
        json.dump(res, f, indent=2, sort_keys=True, default=float)
    cols = ['key', 'family', 'claim', 'kind', 'margin', 'point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1', 'clause_pass', 'point_pass', 'status']
    for name, EE in (('PRIMARY_ENDPOINTS.csv', E), ('SECONDARY_ENDPOINTS.csv', E2)):
        with open(out / name, 'x', newline='') as f:
            w = csv.writer(f); w.writerow(cols)
            for k, v in EE.items():
                w.writerow([k] + [v.get(c) for c in cols[1:]])
    with open(out / 'RECOVERY_AND_UTILITY_TABLE.csv', 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'service'] + [f'{v}_{x}' for v in ('pair', 'income', 'occupation') for x in ('auc', 'se', 'winner')]
                   + ['pair_auc_slate7'] + [f'{t}|{r}|{m}' for t in C.TASKS for r in ('U0', 'Ucal') for m in C.CAPS] + ['dev_point_feasible_24_this_seed'])
        for s in C.SEEDS:
            for svc, v in res['per_seed'][f'S{s}']['services'].items():
                rc = v['recovery']
                w.writerow([s, svc] + [x for vv in ('pair', 'income', 'occupation') for x in (rc[vv]['auc'], rc[vv]['auc_se'], rc[vv]['winner'])]
                           + [v['recovery_slate7']['pair']['auc']] + [v['utility'][f'{t}|{r}|{m}']['point'] for t in C.TASKS for r in ('U0', 'Ucal') for m in C.CAPS]
                           + [v['utility_point_feasible_24_this_seed']])
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    r = run(ap.parse_args().out)
    print(json.dumps({'claims': {k: (v['decision'] if isinstance(v, dict) else v) for k, v in r['claims'].items()}, 'controls': r['controls'],
                      'separate': r['separate_status_fields']}, indent=1, default=str))
