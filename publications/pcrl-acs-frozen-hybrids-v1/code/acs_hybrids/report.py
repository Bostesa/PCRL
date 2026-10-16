"""Report: re-read the already-exposed DEV_REPORT truth (after this campaign's committed REPORTING_LOCK) and evaluate the
endpoints fixed in ANALYSIS_LOCK for the frozen bank and the frozen readers.

Inference (ANALYSIS_LOCK 'inference', inherited from the source): one nonparametric household bootstrap (B = 2048, seed
2026101603) over DEV_REPORT households, shared across every service, endpoint and both model seeds. Bounds are Wald-type:
point +/- t(G-1) x bootstrap SE. Decision family: three claims, IUT within each claim at one-sided alpha/3 per clause
(alpha = 0.025). Reporting families: two-sided 95% Bonferroni-t intervals over the enumerated primary roster (K_primary) and,
separately, over the secondary descriptive roster (K_secondary). Failing-direction labels come only from the primary
simultaneous intervals; secondary labels are descriptive. Intervals condition on the frozen defense models and fitted
readers; the two seeds share people and are not independent populations. The household bootstrap is not ACS survey-design
variance; PWGTP-weighted values are point summaries only. Everything here is exploratory development evidence.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import common as C


HALT_ON_SOURCE_REPRODUCTION = True   # engineering-rehearsal hook (amendment E1): only the synthetic-label rehearsal may switch it off


def crit(G, level_one_sided):
    from scipy import stats
    return float(stats.t.ppf(1 - level_one_sided, G - 1))


def status_at(kind, L, U, m):
    if kind == 'benefit':
        return 'ESTABLISHED' if L > m else ('ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED')
    if kind == 'no_excess':
        return 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')
    return 'ABOVE_ZERO' if L > m else ('BELOW_ZERO' if U < m else 'UNRESOLVED')


def endpoint(point, draws, spec, c1, cK, family, alpha_clause):
    """Wald-type: clause bounds point -/+ c1 x SE (claim clauses only); family interval point +/- cK x SE.
    Statuses at every reference value m of the endpoint (STATISTICAL_REVIEW_PRELOCK (d) 1 and 9)."""
    draws = np.asarray(draws, dtype=np.float64)
    se = float(np.std(draws, ddof=1)); kind = spec['kind']
    L, U = float(point - cK * se), float(point + cK * se)
    is_clause = spec.get('claim') in ('A', 'B', 'C')
    rec = {'point': float(point), 'se': se, 'kind': kind, 'family': family, 'claim': spec.get('claim'), 'margin': spec.get('margin'),
           'refs': spec['refs'], 'family_lower': L, 'family_upper': U, 'value_source': spec.get('value_source'),
           'construction': 'point +/- t(G-1) x paired household-bootstrap SE; conditional on frozen models and fitted readers',
           'bootstrap_skewness': float(((draws - draws.mean()) ** 3).mean() / max(draws.std() ** 3, 1e-300)), 'finite': bool(np.isfinite(point) and np.isfinite(se))}
    rec['at'] = {}
    for m in spec['refs']:
        row = {'point_minus_m': float(point - m), 'lower_minus_m': L - m, 'upper_minus_m': U - m, 'status_at_m': status_at(kind, L, U, m)}
        if is_clause:
            row['clause_bound_minus_m'] = float((point - c1 * se) - m) if kind == 'benefit' else float((point + c1 * se) - m)
        rec['at'][str(m)] = row
    if is_clause:
        m = spec['margin']
        rec['clause_L1'] = float(point - c1 * se); rec['clause_U1'] = float(point + c1 * se)
        rec['clause_pass'] = bool(rec['clause_L1'] > m) if kind == 'benefit' else bool(rec['clause_U1'] <= m)
        rec['point_pass'] = bool(point > m) if kind == 'benefit' else bool(point <= m)
        rec['status'] = status_at(kind, L, U, m)
        q = float(np.quantile(draws, alpha_clause)) if kind == 'benefit' else float(np.quantile(draws, 1 - alpha_clause))
        rec['sensitivity_percentile_bound'] = q
        rec['sensitivity_pass'] = bool(q > m) if kind == 'benefit' else bool(q <= m)
        rec['sensitivity_disagrees_with_clause'] = rec['sensitivity_pass'] != rec['clause_pass']
    else:
        rec['status'] = status_at(kind, L, U, spec['refs'][0]) if kind == 'difference' else {str(m): rec['at'][str(m)]['status_at_m'] for m in spec['refs']}
    return rec


def run(out_dir):
    from task_headroom.inference import group_weights
    from task_conditioned_representation.inference import auc, bootstrap_auc
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    rl = C.RESULTS / 'REPORTING_LOCK.json'
    if not (C.committed(C.RESULTS / 'ANALYSIS_LOCK.json') and C.committed(rl)):
        raise PermissionError('report requires the committed ANALYSIS_LOCK and REPORTING_LOCK')
    rlock = json.loads(rl.read_text())
    inf = lock['inference']
    T = C.dev_report_truth('frozen-hybrids follow-up report: evaluate the ANALYSIS_LOCK endpoints for the frozen bank and readers')
    y = {t: T[t] for t in C.TASKS}; sex = T['sex']; pw = T['weights'].astype(np.float64)
    W, G = group_weights(T['unit'], seed=inf['bootstrap_seed'], draws=inf['bootstrap_B'])
    alpha = inf['alpha_family_one_sided']; n_claims = inf['n_claims']
    c1 = crit(G, alpha / n_claims)
    cs = crit(G, 0.05 / (2 * inf['K_primary'])); cs2 = crit(G, 0.05 / (2 * inf['K_secondary'])); cm = crit(G, 0.025)
    reg = inf['critical_values']
    if not (G - 1 == reg['df'] and round(c1, 4) == reg['clause_one_sided_alpha_over_3'] and round(cs, 4) == reg['primary_K38'] and round(cs2, 4) == reg['secondary_K22'] and round(cm, 4) == reg['marginal_95']):
        raise SystemExit('HALT: critical values differ from the ANALYSIS_LOCK registration')
    hh_per_sex = {int(g): int(len(np.unique(T['unit'][sex == g]))) for g in (0, 1)}
    estimable = all(v >= inf['subgroup_min_households_per_sex'] for v in hh_per_sex.values())
    res = {'schema': 'ACSH1-dev-report-v1', 'utc': C.now(), 'status': 'EXPLORATORY DEVELOPMENT FOLLOW-UP (already-exposed DEV_REPORT people)',
           'households': int(G), 'persons': int(len(sex)), 'subgroup_households_per_sex': hh_per_sex, 'subgroup_estimable': estimable,
           'bootstrap': {'B': inf['bootstrap_B'], 'seed': inf['bootstrap_seed'], 'unit': 'DEV_REPORT household', 'shared_across': 'services, endpoints, seeds'},
           'critical': {'t_one_sided_alpha_over_3': c1, 't_simultaneous_primary': cs, 'K_primary': inf['K_primary'], 't_simultaneous_secondary': cs2,
                        'K_secondary': inf['K_secondary'], 't_marginal_95': cm},
           'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'), 'reporting_lock_sha256': C.sha(rl), 'per_seed': {}}
    auc_pt, auc_dr, vec = {}, {}, {}
    for s in C.SEEDS:
        bank, ids = C.load_bank(s, roles=('DEV_REPORT',))
        if not np.array_equal(ids['row_id__DEV_REPORT'], T['row_id']):
            raise AssertionError('bank / DEV_REPORT row order differs')
        rd = C.PRIV / 'runs' / f'READERS_{s}'
        rr = json.loads((rd / 'READERS_SEED.json').read_text())
        if C.sha(rd / 'DEV_REPORT_SCORES.npz') != rlock['readers'][f'S{s}']['dev_report_scores_sha256']:
            raise PermissionError('reader scores differ from REPORTING_LOCK')
        with np.load(rd / 'DEV_REPORT_SCORES.npz') as z:
            scores = {k: z[k].copy() for k in z.files}
        if not np.array_equal(scores['row_id'], T['row_id']):
            raise AssertionError('score / DEV_REPORT row order differs')
        srec = {'services': {}, 'controls': {}}
        for svc in C.ATTACK_ROSTER:
            v = {'recovery': {}, 'utility': {}, 'task_summary': {}, 'subgroup': {}}
            for view in ('income', 'occupation', 'pair'):
                w = rlock['readers'][f'S{s}']['winners'][svc][view]
                sc = scores[w['reader_id']]
                auc_pt[(svc, s, view)] = auc(sex, sc); auc_dr[(svc, s, view)] = bootstrap_auc(sex, sc, W)
                se = float(np.std(auc_dr[(svc, s, view)], ddof=1))
                v['recovery'][view] = {'winner': w['candidate'], 'reader_id': w['reader_id'], 'auc': auc_pt[(svc, s, view)], 'auc_se': se,
                                       'marginal95': [auc_pt[(svc, s, view)] - cm * se, auc_pt[(svc, s, view)] + cm * se],
                                       'pwgtp_weighted_auc_point': auc(sex, sc, pw)}
            v['recovery']['pair_reader_slate_auc_descriptive'] = {k.split('__', 2)[2]: auc(sex, a) for k, a in scores.items() if k.startswith(f'PAIR__{svc}__')}
            for t in C.TASKS:
                k = C.K[t]; p = bank[svc]['DEV_REPORT'][t]
                vv = C.task_vectors(y[t], p, k); vec[(svc, s, t)] = vv
                v['task_summary'][t] = {m: float(vv[m].mean()) for m in vv}
                v['task_summary'][t]['pwgtp_weighted'] = {m: float(np.sum(pw * vv[m]) / pw.sum()) for m in vv}
                for ref in C.REFERENCES:
                    rv = C.task_vectors(y[t], bank[ref]['DEV_REPORT'][t], k)
                    for metric, d in {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'], 'brier_excess': vv['brier'] - rv['brier']}.items():
                        dr = (W * d[None, :]).sum(1) / W.sum(1); se = float(np.std(dr, ddof=1))
                        v['utility'][f'{t}|{ref}|{metric}'] = {'point': float(d.mean()), 'se': se, 'cap': C.CAPS[metric], 'point_pass': bool(d.mean() <= C.CAPS[metric]),
                                                              'marginal95': [float(d.mean() - cm * se), float(d.mean() + cm * se)],
                                                              'pwgtp_weighted_point': float(np.sum(pw * d) / pw.sum()), '_draws': dr}
            rv = C.task_vectors(y['occupation'], bank['U0']['DEV_REPORT']['occupation'], 5); vv = vec[(svc, s, 'occupation')]
            for g in (0, 1):
                m = sex == g; d = vv['brier'] - rv['brier']; Wg = W[:, m]
                dr = (Wg * d[m][None, :]).sum(1) / Wg.sum(1); se = float(np.std(dr, ddof=1))
                pm = bank[svc]['DEV_REPORT']['occupation'][m]; ym = y['occupation'][m]; dec = pm.argmax(1)
                v['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d[m].mean()), 'se': se, '_draws': dr,
                                            'marginal95': [float(d[m].mean() - cm * se), float(d[m].mean() + cm * se)],
                                            'group_households': hh_per_sex[g], 'group_persons': int(m.sum()),
                                            'group_accuracy': float(vv['accuracy'][m].mean()), 'group_nll': float(vv['nll'][m].mean()),
                                            'group_accuracy_loss_vs_U0': float((rv['accuracy'] - vv['accuracy'])[m].mean()),
                                            'group_nll_excess_vs_U0': float((vv['nll'] - rv['nll'])[m].mean()),
                                            'group_per_class_recall': [float((dec[ym == c] == c).mean()) if np.any(ym == c) else None for c in range(5)],
                                            'group_class_support': [int((ym == c).sum()) for c in range(5)]}
            srec['services'][svc] = v
        for view in ('income', 'occupation', 'pair'):
            pl = np.where(sex == 1, scores[f'CONTROL__{view}__PLANTED_1'], scores[f'CONTROL__{view}__PLANTED_0'])
            nd = bootstrap_auc(sex, scores[f'CONTROL__{view}__SHUFFLED'], W)
            srec['controls'][view] = {'wire': C.PRIMARY, 'shuffled_null_auc': auc(sex, scores[f'CONTROL__{view}__SHUFFLED']), 'shuffled_null_auc_se': float(np.std(nd, ddof=1)),
                                      'planted_bit_auc': auc(sex, pl), 'planted_attack_validation_auc': rr['controls'][view]['planted_attack_validation_auc'],
                                      'shuffled_attack_validation_auc': rr['controls'][view]['shuffled_attack_validation_auc']}
        res['per_seed'][f'S{s}'] = srec

    # ---------------- checks at the truth reading (any failure halts; no decision is issued)
    halts = []
    inh = {}
    for s in C.SEEDS:
        for svc, (src_i, src_o) in C.COMPONENTS.items():
            for t, comp in zip(C.TASKS, (src_i, src_o)):
                ref_svc = 'H_LL' if comp == 'LOCAL_TREE' else 'H_JJ'
                inh[f'S{s}|{svc}|{t}=={comp}'] = all(np.array_equal(vec[(svc, s, t)][m], vec[(ref_svc, s, t)][m]) for m in ('accuracy', 'nll', 'brier'))
    if not all(inh.values()):
        halts.append(f'per-person utility vectors differ from components: {[k for k, v in inh.items() if not v]}')
    src_res = json.loads((C.SRC_RESULTS / 'DEV_REPORT_RESULTS.json').read_text())
    rep = {}
    for s in C.SEEDS:   # bootstrap reproduction: recomputed source values keep their published point and SE
        for svc, s_label in (('H_LL', 'LOCAL_TREE'), ('H_JJ', 'JOINT_TREE'), ('TASK_REFIT', 'TASK_REFIT'), ('NEURAL_RAW', 'NEURAL_RAW_rho1.0_E2'), ('T_TREE', 'T_TREE')):
            for key, u in res['per_seed'][f'S{s}']['services'][svc]['utility'].items():
                th = src_res['per_seed'][str(s)]['services'][s_label]['utility'][key]
                rep[f'S{s}|{svc}|utility|{key}'] = bool(u['point'] == th['point'] and abs(u['se'] - th['se']) <= 1e-12)
            for g in ('sex0', 'sex1'):
                u = res['per_seed'][f'S{s}']['services'][svc]['subgroup'][g]; th = src_res['per_seed'][str(s)]['services'][s_label]['subgroup'][g]
                rep[f'S{s}|{svc}|subgroup|{g}'] = bool(u['occupation_brier_excess_vs_U0'] == th['point'] and abs(u['se'] - th['se']) <= 1e-12)
    replay_bitwise = all(rlock['readers'][f'S{s}']['replay_vs_source']['bitwise'] == rlock['readers'][f'S{s}']['replay_vs_source']['compared'] for s in C.SEEDS)
    if replay_bitwise:
        for s in C.SEEDS:
            for v in ('income', 'occupation'):
                th = src_res['endpoints'][f'RECIPE|{v}_recipient_worsening_vs_TASK_REFIT|s{s}']
                d = auc_dr[('H_LL', s, v)] - auc_dr[('TASK_REFIT', s, v)]
                rep[f'S{s}|LOCAL_{v}_worsening_endpoint'] = bool(auc_pt[('H_LL', s, v)] - auc_pt[('TASK_REFIT', s, v)] == th['point'] and abs(float(np.std(d, ddof=1)) - th['se']) <= 1e-12)
    if not HALT_ON_SOURCE_REPRODUCTION and not getattr(C, '_SYNTHETIC_REHEARSAL', False):
        raise SystemExit('HALT: the source-reproduction halt may be disabled only in the synthetic-label rehearsal')
    if not all(rep.values()) and HALT_ON_SOURCE_REPRODUCTION:
        halts.append(f'bootstrap / source reproduction failed: {[k for k, v in rep.items() if not v]}')
    res['checks_at_truth'] = {'inheritance_equalities': {'n': len(inh), 'all_pass': all(inh.values())},
                              'source_reproduction': {'n': len(rep), 'all_pass': all(rep.values()), 'reader_replay_bitwise': replay_bitwise,
                                                      'note': 'income/occupation worsening reproduction is checked only when the reader replay held bitwise'}}

    # ---------------- endpoints from the lock (generic evaluator)
    def evaluate(spec):
        e = spec['estimand']; s = spec['seed']
        if e['type'] == 'auc_diff':
            return auc_pt[(e['plus'], s, e['view'])] - auc_pt[(e['minus'], s, e['view'])], auc_dr[(e['plus'], s, e['view'])] - auc_dr[(e['minus'], s, e['view'])]
        if e['type'] == 'auc_interaction':
            a = lambda x: (auc_pt[(x, s, e['view'])], auc_dr[(x, s, e['view'])])
            (ll, dll), (lj, dlj), (jl, djl), (jj, djj) = a('H_LL'), a('H_LJ'), a('H_JL'), a('H_JJ')
            return (ll - lj) - (jl - jj), (dll - dlj) - (djl - djj)
        if e['type'] == 'utility':
            u = res['per_seed'][f'S{s}']['services'][e['service']]['utility'][f"{e['task']}|{e['reference']}|{e['metric']}"]
            return u['point'], u['_draws']
        if e['type'] == 'subgroup_occupation_brier_vs_U0':
            g = res['per_seed'][f'S{s}']['services'][e['service']]['subgroup'][f"sex{e['group']}"]
            return g['occupation_brier_excess_vs_U0'], g['_draws']
        raise ValueError(f'unknown estimand {e}')
    a_clause = alpha / n_claims
    E = {sp['key']: endpoint(*evaluate(sp), sp, c1, cs, 'primary (K = 38)', a_clause) for sp in lock['primary_roster']}
    E2 = {sp['key']: endpoint(*evaluate(sp), sp, c1, cs2, 'secondary descriptive (K = 22)', a_clause) for sp in lock['secondary_roster']}
    if len(E) != inf['K_primary'] or len(E2) != inf['K_secondary'] or not all(v['finite'] for v in list(E.values()) + list(E2.values())):
        halts.append('roster size or finiteness check failed')
    if halts:
        raise SystemExit('HALT before any decision: ' + ' | '.join(halts))
    # source K = 40 status beside the K = 38 status (multiplicity only)
    def src_record(cp, s):
        if not cp:
            return None
        if cp['type'] == 'endpoint':
            return src_res['endpoints'][cp['key']]
        if cp['type'] == 'service_utility':
            return src_res['per_seed'][str(s)]['services'][cp['service']]['utility'][cp['key']]
        if cp['type'] == 'service_subgroup':
            return src_res['per_seed'][str(s)]['services'][cp['service']]['subgroup'][cp['group']]
        return None
    for fam_E, roster in ((E, lock['primary_roster']), (E2, lock['secondary_roster'])):
        for sp in roster:
            rec40 = src_record(sp.get('source_counterpart'), sp['seed'])
            if rec40 is None or rec40.get('sim_status') is None:
                continue
            m40 = rec40.get('margin'); at = fam_E[sp['key']]['at'].get(str(m40))
            mine = at['status_at_m'] if at else None
            fam_E[sp['key']]['source_K40_status'] = rec40['sim_status']; fam_E[sp['key']]['source_K40_margin'] = m40
            fam_E[sp['key']]['K38_status_at_source_margin'] = mine
            fam_E[sp['key']]['status_changed_by_multiplicity_only'] = (bool(mine != rec40['sim_status']) if mine is not None else None)
            if mine is not None and mine != rec40['sim_status']:
                fam_E[sp['key']]['status_change_label'] = 'STATUS_CHANGED_BY_MULTIPLICITY_ONLY' if sp.get('value_source') == 'P' or sp.get('source_counterpart', {}).get('type') == 'endpoint' else 'STATUS_DIFFERS'
    res['endpoints_primary'] = E; res['endpoints_secondary'] = E2

    # ---------------- controls, stage 2 (pipeline validity; never evidence of privacy)
    cr = lock['controls']; lo, hi = cr['null_band']; wlo, whi = cr['null_wide_band_disclosure_only']
    vals = []
    for s in C.SEEDS:
        for view, c in res['per_seed'][f'S{s}']['controls'].items():
            c['null_z'] = (c['shuffled_null_auc'] - 0.5) / c['shuffled_null_auc_se'] if c['shuffled_null_auc_se'] > 0 else None
            c['NULL_WIDE'] = bool(not (wlo <= c['shuffled_null_auc'] <= whi) and lo <= c['shuffled_null_auc'] <= hi)
            vals.append(c)
    planted = all(np.isfinite(c['planted_bit_auc']) and c['planted_bit_auc'] >= cr['planted_min_auc'] and np.isfinite(c['planted_attack_validation_auc'])
                  and c['planted_attack_validation_auc'] >= cr['planted_min_auc'] for c in vals) and len(vals) == 6
    null = all(np.isfinite(c['shuffled_null_auc']) and lo <= c['shuffled_null_auc'] <= hi and lo <= c['shuffled_attack_validation_auc'] <= hi for c in vals) and len(vals) == 6
    valid = bool(planted and null)
    res['controls'] = {'PLANTED_PASS': bool(planted), 'NULL_PASS': bool(null), 'CONTROLS_VALID': valid, 'stage_1_passed_before_truth': rlock['stage_1_controls']['CONTROLS_STAGE1_VALID'],
                       'null_wide_flags': [f'S{s}|{v}' for s in C.SEEDS for v, c in res['per_seed'][f'S{s}']['controls'].items() if c['NULL_WIDE']]}
    if not valid:
        for fam_E in (E, E2):
            for k, v in fam_E.items():
                if v['family'] and any(x in k for x in ('coalition', 'worsening', 'INCREMENT', 'INTERACTION')):
                    v['recovery_status_invalid'] = True

    # ---------------- claims
    ck = lock['claim_keys']; labA = lock['claims']['A']; labB = lock['claims']['B']['labels']; labC = lock['claims']['C']['labels']
    util = [E[k] for k in ck['A_utility']]; gate24 = all(v['point_pass'] for v in util)
    ben_ok = {k: E[k]['clause_pass'] for k in ck['A_coalition_benefit']}; wor_ok = {k: E[k]['clause_pass'] for k in ck['A_recipient_worsening']}
    A = {'coalition_benefit_clause_pass': ben_ok, 'recipient_worsening_clause_pass': wor_ok, 'utility_point_gate_24': gate24,
         'failed_clauses': [k for k, ok in {**ben_ok, **wor_ok}.items() if not ok] + [k for k in ck['A_utility'] if not E[k]['point_pass']],
         'scope': labA['scope']}
    estA = all(ben_ok.values()) and all(wor_ok.values()) and gate24
    A['decision'] = labA['invalid_label'] if not valid else (labA['labels'][0] if estA else labA['labels'][1])
    A['nested_decision_utility_bounds'] = (labA['invalid_label'] if not valid else
                                           labA['nested_labels'][0] if estA and all(v['clause_pass'] for v in util) else labA['nested_labels'][1])
    adm = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())['model_select_point_feasibility']
    ms_ok = bool(adm[C.PRIMARY]['point_feasible_24'])
    sel = json.loads((C.SRC_RESULTS / 'SELECTION_LOCK.json').read_text())
    comp_keys = ck['B_competitive']
    nr_complete = all(np.isfinite(auc_pt[('NEURAL_RAW', s, 'pair')]) for s in C.SEEDS)
    sup = all(E[k]['clause_pass'] for k in comp_keys); mat = all(E[k]['clause_L1'] > 0.020 for k in comp_keys)
    if not nr_complete:
        dB = labB['incomplete']
    elif not valid:
        dB = labB['invalid']
    elif sel['frozen']['NEURAL_RAW']['status'] != 'ELIGIBLE':
        dB = labB['no_eligible']
    elif not (ms_ok and gate24):
        dB = labB['fails_gate']
    else:
        dB = labB['superior'] if sup else labB['not_superior']
    Bc = {'eligible_set': ['NEURAL_RAW'], 'note': 'FAGTB and FARE had no MODEL_SELECT-feasible configuration (source SELECTION_LOCK) and were not compared; T_TREE is never part of claim B',
          'h_lj_same_utility_requirements': {'model_select_point_feasible_24': ms_ok, 'dev_report_point_feasible_24': gate24},
          'clause_pass_vs_0': {k: E[k]['clause_pass'] for k in comp_keys}, 'clause_L1_gt_0.020': {k: E[k]['clause_L1'] > 0.020 for k in comp_keys},
          'K38_status_at_0_and_0.020': {k: {m: E[k]['at'][m]['status_at_m'] for m in E[k]['at']} for k in comp_keys}, 'decision': dB}
    Bc['nested_material_superiority_0.020'] = lock['claims']['B']['nested_labels'][0] if (dB == labB['superior'] and mat) else lock['claims']['B']['nested_labels'][1]
    sg = [E[k] for k in ck['C_subgroup']]
    dC = (labC['not_estimable'] if not estimable else labC['qualified'] if all(v['clause_pass'] for v in sg)
          else labC['harm'] if any(v['family_lower'] > 0.005 for v in sg) else labC['unresolved'])
    Cc = {'estimable': estimable, 'households_per_sex': hh_per_sex, 'guards_clause_pass': {k: E[k]['clause_pass'] for k in ck['C_subgroup']},
          'guards_K38_status': {k: E[k]['status'] for k in ck['C_subgroup']}, 'harm_guards': [k for k in ck['C_subgroup'] if E[k]['family_lower'] > 0.005], 'decision': dC}
    res['claims'] = {'A_RECIPE_BENEFIT': A, 'B_COMPETITIVE': Bc, 'C_SUBGROUP': Cc}
    settled = lock['settled_before_readers']
    res['settled_vs_reported'] = {'utility_point_gate_24': [settled['dev_report_utility_point_gate_24'], gate24],
                                  'utility_clause_bounds_passing_of_24': [settled['utility_clause_bounds_passing_of_24'], int(sum(v['clause_pass'] for v in util))],
                                  'claim_C': [settled['claim_C_outcome_implied'], dC],
                                  'income_worsening_clause_passes': [settled['income_worsening_clause_passes'], {k: E[k]['clause_pass'] for k in ck['A_recipient_worsening'] if 'income' in k}],
                                  'note': 'these outcomes were fixed by published values before readers ran; they are not findings of this campaign'}

    # ---------------- separate status fields (never merged, never in the conjunction)
    ut_status = {k: E[k]['status'] for k in ck['A_utility']}
    res['separate_status_fields'] = {
        'overall_point_feasibility': {'DEV_REPORT': 'POINT_FEASIBLE_24' if gate24 else 'POINT_INFEASIBLE', 'DEV_REPORT_failed_keys': [k for k in ck['A_utility'] if not E[k]['point_pass']],
                                      'MODEL_SELECT': 'POINT_FEASIBLE_24' if ms_ok else 'POINT_INFEASIBLE'},
        'overall_confidence_status': {'status': ('CONFIDENCE_FEASIBLE_24' if all(v == 'PASS' for v in ut_status.values()) else 'ESTABLISHED_UTILITY_VIOLATION'
                                                 if any(v == 'ESTABLISHED_VIOLATION' for v in ut_status.values()) else 'CONFIDENCE_UNRESOLVED'),
                                      'violation_keys': [k for k, v in ut_status.items() if v == 'ESTABLISHED_VIOLATION'],
                                      'counts': {x: sum(v == x for v in ut_status.values()) for x in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
                                      'clause_bounds_passing_of_24': int(sum(v['clause_pass'] for v in util))},
        'subgroup_status': {'decision': dC, 'guards_K38_status': Cc['guards_K38_status']}}

    # ---------------- overall conjunction (reported last)
    unmet = []
    if not valid:
        unmet.append('CONTROL_FAILURE')
    if not gate24:
        unmet.append('DEV_REPORT utility point gate: ' + ', '.join(k for k in ck['A_utility'] if not E[k]['point_pass']))
    if not ms_ok:
        unmet.append('MODEL_SELECT feasibility')
    unmet += [f'coalition benefit clause {k}' for k in ck['A_coalition_benefit'] if not E[k]['clause_pass']]
    unmet += [f'recipient worsening clause {k}' for k in ck['A_recipient_worsening'] if not E[k]['clause_pass']]
    if dB in (labB['incomplete'], labB['no_eligible']):
        unmet.append(f'claim B {dB}')
    unmet += [f'COMPETITIVE benefit clause {k}' for k in comp_keys if not E[k]['clause_pass']]
    supported = valid and A['decision'] == labA['labels'][0] and dB == labB['superior']
    res['central_question'] = {'Q1_point_utility_inherited': bool(gate24), 'Q2_material_improvement_over_TASK_REFIT': A['decision'] == labA['labels'][0],
                               'Q3_beats_neural_control': dB == labB['superior'],
                               'overall': lock['central_question']['labels'][0] if supported else lock['central_question']['labels'][1],
                               'unmet_components_in_order': unmet, 'scope_sentence': lock['central_question']['scope_sentence']}
    res['sensitivity_disagreements'] = [k for k, v in E.items() if v.get('sensitivity_disagrees_with_clause')]
    res['descriptive'] = {'pair_slate_envelope_selected_on_report_data': {
        f'S{s}|{svc}': {'by_reader': res['per_seed'][f'S{s}']['services'][svc]['recovery']['pair_reader_slate_auc_descriptive'],
                        'envelope_max_selected_on_report_data': max(res['per_seed'][f'S{s}']['services'][svc]['recovery']['pair_reader_slate_auc_descriptive'].values()),
                        'attack_validation_winner': res['per_seed'][f'S{s}']['services'][svc]['recovery']['pair']['winner']}
        for s in C.SEEDS for svc in ('H_LJ', 'H_LL', 'TASK_REFIT', 'NEURAL_RAW')},
        'per_recipient_differences_unadjusted': {f'S{s}|{h}|{v}|vs_{c}': auc_pt[(h, s, v)] - auc_pt[(c, s, v)]
                                                 for s in C.SEEDS for h in C.COMPONENTS for v in ('income', 'occupation') for c in ('NEURAL_RAW', 'T_TREE')}}

    # strip draws, write outputs
    for s in C.SEEDS:
        for svc, v in res['per_seed'][f'S{s}']['services'].items():
            for u in v['utility'].values():
                u.pop('_draws', None)
            for g in v['subgroup'].values():
                g.pop('_draws', None)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'DEV_REPORT_RESULTS.json', 'x') as f:
        json.dump(res, f, indent=2, sort_keys=True, default=float)
    write_tables(res, out, lock)
    return res


def write_tables(res, out, lock):
    cols = ['key', 'family', 'claim', 'kind', 'margin', 'value_source', 'point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1', 'clause_pass',
            'point_pass', 'status', 'source_K40_status', 'status_changed_by_multiplicity_only', 'sensitivity_percentile_bound', 'sensitivity_disagrees_with_clause',
            'bootstrap_skewness', 'recovery_status_invalid']
    for name, E in (('PRIMARY_ENDPOINTS.csv', res['endpoints_primary']), ('SECONDARY_ENDPOINTS.csv', res['endpoints_secondary'])):
        with open(out / name, 'x', newline='') as f:
            w = csv.writer(f); w.writerow(cols)
            for k, v in E.items():
                w.writerow([k] + [json.dumps(v.get(c)) if isinstance(v.get(c), dict) else v.get(c) for c in cols[1:]])
    with open(out / 'DISTANCES.csv', 'x', newline='') as f:
        w = csv.writer(f); w.writerow(['key', 'family', 'm', 'point_minus_m', 'lower_minus_m', 'upper_minus_m', 'clause_bound_minus_m', 'status_at_m'])
        for E in (res['endpoints_primary'], res['endpoints_secondary']):
            for k, v in E.items():
                for m, r in v['at'].items():
                    w.writerow([k, v['family'], m] + [None if r.get(c) is None else round(r[c], 4) if isinstance(r[c], float) else r[c]
                                                     for c in ('point_minus_m', 'lower_minus_m', 'upper_minus_m', 'clause_bound_minus_m', 'status_at_m')])
    adm = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())['model_select_point_feasibility']
    with open(out / 'SERVICE_TABLE.csv', 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'service', 'income_source', 'occupation_source', 'model_select_point_feasible_24', 'dev_report_point_feasible_24']
                   + [f'{v}_{x}' for v in ('pair', 'income', 'occupation') for x in ('auc', 'auc_minus_0.5', 'se', 'marginal95_lower', 'marginal95_upper', 'winner')]
                   + ['pair_auc_pwgtp_point'] + [f'{t}|{r}|{m}' for t in C.TASKS for r in C.REFERENCES for m in C.CAPS])
        for s in C.SEEDS:
            for svc in C.ATTACK_ROSTER:
                v = res['per_seed'][f'S{s}']['services'][svc]; rc = v['recovery']; src = C.SOURCE_OF[svc]
                row = [s, svc, src[0], src[1], adm[svc]['point_feasible_24'], all(u['point_pass'] for u in v['utility'].values())]
                for view in ('pair', 'income', 'occupation'):
                    r = rc[view]; row += [r['auc'], r['auc'] - 0.5, r['auc_se'], r['marginal95'][0], r['marginal95'][1], r['winner']]
                row += [rc['pair']['pwgtp_weighted_auc_point']] + [v['utility'][f'{t}|{r}|{m}']['point'] for t in C.TASKS for r in C.REFERENCES for m in C.CAPS]
                w.writerow(row)
    # four-hybrid display: each difference carries the interval of the family that holds it
    E, E2 = res['endpoints_primary'], res['endpoints_secondary']
    where = {}
    for s in C.SEEDS:
        where[('H_LJ', s, 'coalition_vs_TASK_REFIT')] = E[f'RECIPE|H_LJ|coalition_benefit_vs_TASK_REFIT|s{s}']
        where[('H_LJ', s, 'coalition_vs_NEURAL_RAW')] = E[f'COMPETITIVE|H_LJ|coalition_benefit_vs_NEURAL_RAW|s{s}']
        for h in ('H_LL', 'H_JL', 'H_JJ'):
            where[(h, s, 'coalition_vs_TASK_REFIT')] = E2[f'SEC|{h}|coalition_benefit_vs_TASK_REFIT|s{s}']
            where[(h, s, 'coalition_vs_NEURAL_RAW')] = E2[f'SEC|{h}|coalition_benefit_vs_NEURAL_RAW|s{s}']
        for h in ('H_LL', 'H_LJ'):
            where[(h, s, 'income_worsening')] = E[f'RECIPE|H_LJ|income_recipient_worsening_vs_TASK_REFIT|s{s}']
        for h in ('H_JL', 'H_JJ'):
            where[(h, s, 'income_worsening')] = E2[f'SEC|JOINT_income_wire|recipient_worsening_vs_TASK_REFIT|s{s}']
        for h in ('H_LJ', 'H_JJ'):
            where[(h, s, 'occupation_worsening')] = E[f'RECIPE|H_LJ|occupation_recipient_worsening_vs_TASK_REFIT|s{s}']
        for h in ('H_LL', 'H_JL'):
            where[(h, s, 'occupation_worsening')] = E2[f'SEC|LOCAL_occupation_wire|recipient_worsening_vs_TASK_REFIT|s{s}']
    with open(out / 'FOUR_HYBRID_TABLE.csv', 'x', newline='') as f:
        w = csv.writer(f)
        hdr = ['seed', 'hybrid', 'income_source', 'occupation_source']
        for v in ('pair', 'income', 'occupation'):
            hdr += [f'{v}_auc', f'{v}_marginal95_lower', f'{v}_marginal95_upper']
        for q in ('coalition_vs_TASK_REFIT', 'coalition_vs_NEURAL_RAW', 'income_worsening', 'occupation_worsening'):
            hdr += [f'{q}_point', f'{q}_family', f'{q}_family_lower', f'{q}_family_upper', f'{q}_status']
        w.writerow(hdr)
        for s in C.SEEDS:
            for h in C.COMPONENTS:
                rc = res['per_seed'][f'S{s}']['services'][h]['recovery']
                row = [s, h, *C.COMPONENTS[h]]
                for v in ('pair', 'income', 'occupation'):
                    row += [rc[v]['auc'], rc[v]['marginal95'][0], rc[v]['marginal95'][1]]
                for q in ('coalition_vs_TASK_REFIT', 'coalition_vs_NEURAL_RAW', 'income_worsening', 'occupation_worsening'):
                    e = where[(h, s, q)]
                    row += [e['point'], e['family'], e['family_lower'], e['family_upper'], json.dumps(e['status']) if isinstance(e['status'], dict) else e['status']]
                w.writerow(row)
    with open(out / 'SUBGROUP_TABLE.csv', 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'service', 'group', 'sex_code', 'households', 'persons', 'occupation_brier_excess_vs_U0', 'se', 'marginal95_lower', 'marginal95_upper',
                    'group_accuracy', 'group_nll', 'group_accuracy_loss_vs_U0', 'group_nll_excess_vs_U0', 'per_class_recall', 'class_support'])
        for s in C.SEEDS:
            for svc in C.ATTACK_ROSTER:
                for g in (0, 1):
                    x = res['per_seed'][f'S{s}']['services'][svc]['subgroup'][f'sex{g}']
                    w.writerow([s, svc, f'sex{g}', g + 1, x['group_households'], x['group_persons'], x['occupation_brier_excess_vs_U0'], x['se'],
                                x['marginal95'][0], x['marginal95'][1], x['group_accuracy'], x['group_nll'], x['group_accuracy_loss_vs_U0'],
                                x['group_nll_excess_vs_U0'], json.dumps(x['group_per_class_recall']), json.dumps(x['group_class_support'])])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    r = run(ap.parse_args().out)
    print(json.dumps({'claims': {k: v['decision'] for k, v in r['claims'].items()}, 'central': r['central_question'],
                      'controls': r['controls'], 'separate': r['separate_status_fields']}, indent=1, default=str))
