"""Unit UM_REPORT: the one logged coordinator re-read of the already-exposed DEV_REPORT truth (after the committed
REPORTING_LOCK), then every endpoint, label, utility vector and bounded diagnostic registered in PROTOCOL.md sections 10-12 and
UNIT_INVENTORY.json, computed in this one run.

Household bootstrap over DEV_REPORT households (B and seed from the lock), shared across releases, endpoints and seeds; paired
differences are person-level within each replicate. Wald bounds: point +/- t(G-1) x bootstrap SE. Claim clauses one-sided at
alpha / n_claims (alpha = 0.025, four slices). Primary family: two-sided 95% Bonferroni-t over K_primary; secondary over
K_secondary. Endpoints whose comparator role is unavailable are MISSING (no release) or DIAGNOSTIC_NOT_MATCHED (nearest
diagnostic) and stay in K. Conditional on the fitted releases and readers; the seeds share people; exploratory.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import common as C
from .readers import load_service_bank

ALPHA = 0.025


def crit(G, level_one_sided):
    from scipy import stats
    return float(stats.t.ppf(1 - level_one_sided, G - 1))


def status_at(kind, L, U, m):
    if kind == 'benefit':
        return 'ESTABLISHED' if L > m else ('ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED')
    if kind == 'no_excess':
        return 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')
    if kind == 'match':
        return 'EQUIVALENCE_ESTABLISHED' if (L >= -m and U <= m) else ('MISMATCH_ESTABLISHED' if (L > m or U < -m) else 'EQUIVALENCE_UNRESOLVED')
    return 'ABOVE_ZERO' if L > m else ('BELOW_ZERO' if U < m else 'UNRESOLVED')


def endpoint(point, draws, spec, c1, cK, family, availability, identical=False):
    """Frozen design: clause bounds exist only for AVAILABLE benefit / no-excess endpoints; match and difference kinds have none."""
    draws = np.asarray(draws, dtype=np.float64)
    se = float(np.std(draws, ddof=1)); kind = spec['kind']; m = spec['margin']
    L, U = float(point - cK * se), float(point + cK * se)
    rec = {'point': float(point), 'se': se, 'kind': kind, 'family': family, 'claim': spec.get('claim'), 'margin': m, 'availability': availability,
           'identical_release': bool(identical), 'family_lower': L, 'family_upper': U, 'finite': bool(np.isfinite(point) and np.isfinite(se)), 'at': {},
           'clause_L1': None, 'clause_U1': None, 'clause_pass': None}
    if kind in ('benefit', 'no_excess') and availability == 'AVAILABLE':
        rec['clause_L1'] = float(point - c1 * se); rec['clause_U1'] = float(point + c1 * se)
        rec['clause_pass'] = bool(rec['clause_L1'] > m) if kind == 'benefit' else bool(rec['clause_U1'] <= m)
    if kind == 'benefit':
        rec['point_pass'] = bool(point > m)
    elif kind == 'no_excess':
        rec['point_pass'] = bool(point <= m)
    elif kind == 'match':
        rec['point_pass'] = bool(abs(point) <= m); rec['point_status'] = 'POINT_MATCH' if rec['point_pass'] else 'POINT_MISMATCH'
    else:
        rec['point_pass'] = None
    if kind != 'match':
        for r in spec['refs']:
            rec['at'][str(r)] = {'point_minus_m': float(point - r), 'lower_minus_m': L - r, 'upper_minus_m': U - r, 'status_at_m': status_at(kind, L, U, r)}
    rec['status'] = status_at(kind, L, U, m)
    if availability != 'AVAILABLE':
        rec['status'] = f'{availability}:{rec["status"]}'
    if identical:
        rec['flag'] = 'IDENTICAL_RELEASE'
    return rec


def run(out_dir, truth=None, event_purpose=None, event_name='DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP'):
    from task_headroom.inference import group_weights
    from task_conditioned_representation.inference import auc, bootstrap_auc
    names = ('ANALYSIS_LOCK.json', 'TRAJECTORY_COMPLETE_LOCK.json', 'SELECTION_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'REPORTING_LOCK.json')
    if truth is None:
        for name in names:
            if not C.committed(C.RESULTS / name):
                raise PermissionError(f'report requires the committed {name}')
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    inv = json.loads((C.RESULTS / 'UNIT_INVENTORY.json').read_text())
    if C.sha(C.RESULTS / 'UNIT_INVENTORY.json') != lock['unit_inventory_sha256']:
        raise PermissionError('UNIT_INVENTORY differs from ANALYSIS_LOCK')
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    rlock = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text()) if truth is None else None
    out = Path(out_dir)
    if (out / 'DEV_REPORT_RESULTS.json').exists():
        raise PermissionError('report results exist (one evaluation)')
    inf = lock['inference']
    T = truth if truth is not None else C.dev_report_truth(event_purpose or
        'SCM utility-matched follow-up: the one coordinator evaluation of the registered endpoints for the frozen bank and readers', event_name)
    y = {t: T[t] for t in C.TASKS}; sex = T['sex']
    W, G = group_weights(T['unit'], seed=inf['bootstrap_seed'], draws=inf['bootstrap_B'])
    if G != 3955 or len(sex) != 5943:
        raise SystemExit(f'HALT: DEV_REPORT has {G} households and {len(sex)} persons, not the registered 3955 / 5943')
    Wsum = W.sum(1)
    c1 = crit(G, inf['alpha_family_one_sided'] / inf['n_claims'])
    cs = crit(G, 0.05 / (2 * inf['K_primary'])); cs2 = crit(G, 0.05 / (2 * inf['K_secondary'])); cm = crit(G, 0.025)
    res = {'schema': 'SCMUM1-dev-report-v1', 'utc': C.now(), 'status': 'EXPLORATORY DEVELOPMENT EVIDENCE (already-exposed DEV_REPORT people; previously used records)',
           'households': int(G), 'persons': int(len(sex)), 'subgroup_households_per_sex': {int(g): int(len(np.unique(T['unit'][sex == g]))) for g in (0, 1)},
           'critical': {'t_one_sided_alpha_over_n_claims': c1, 'n_claims': inf['n_claims'], 't_simultaneous_primary': cs, 'K_primary': inf['K_primary'],
                        't_simultaneous_secondary': cs2, 'K_secondary': inf['K_secondary'], 't_marginal_95': cm},
           'locks_sha256': {n: C.sha(C.RESULTS / n) for n in names if (C.RESULTS / n).exists()}, 'per_seed': {}}
    auc_pt, auc_dr, per_nll, util, per_vec = {}, {}, {}, {}, {}
    svc_of, role_status, rr_services = {}, {}, {}
    for s in C.SEEDS:
        bank, ids, man = load_service_bank(s)
        if not np.array_equal(ids['row_id__DEV_REPORT'], T['row_id']):
            raise AssertionError('bank / DEV_REPORT row order differs')
        rd = C.PRIV / 'runs' / f'READERS_{s}'
        if rlock is not None and C.sha(rd / 'DEV_REPORT_SCORES.npz') != rlock['readers'][f'S{s}']['dev_report_scores_sha256']:
            raise PermissionError('reader scores differ from REPORTING_LOCK')
        rr = json.loads((rd / 'READERS_SEED.json').read_text())
        with np.load(rd / 'DEV_REPORT_SCORES.npz') as z:
            scores = {k: z[k].copy() for k in z.files}
        if not np.array_equal(scores['row_id'], T['row_id']):
            raise AssertionError('score / DEV_REPORT row order differs')
        ps = man['per_seed'][f'S{s}']
        svc_of[s] = ps['service_of_role']
        role_status[s] = {rn: v.get('status') for rn, v in sel['roles'][str(s)].items()}
        rr_services[s] = set(rr['services'])
        srec = {'service_of_role': ps['service_of_role'], 'aliases': ps['aliases'], 'missing_roles': ps['missing_roles'], 'services': {}, 'controls': {}}
        u0 = {t: C.task_vectors(y[t], bank['U0']['DEV_REPORT'][t], C.K[t]) for t in C.TASKS}
        for svc in list(rr['services']) + ['U0', 'Ucal']:
            v = {'recovery': {}, 'recovery_slate7': {}}
            if svc in rr['services']:
                sv = rr['services'][svc]
                for key, wkey in (('recovery', 'winner'), ('recovery_slate7', 'winner_slate7')):
                    for view in ('income', 'occupation', 'pair'):
                        w = sv[wkey][view]; sc = scores[w['reader_id']]
                        tag = view if key == 'recovery' else view + '_slate7'
                        auc_pt[(svc, s, tag)] = auc(sex, sc); auc_dr[(svc, s, tag)] = bootstrap_auc(sex, sc, W)
                        se = float(np.std(auc_dr[(svc, s, tag)], ddof=1))
                        v[key][view] = {'winner': w['candidate'], 'auc': auc_pt[(svc, s, tag)], 'auc_se': se,
                                        'marginal95': [auc_pt[(svc, s, tag)] - cm * se, auc_pt[(svc, s, tag)] + cm * se]}
                v['pair_reader_slate_auc_descriptive'] = {k.split('__', 2)[2]: auc(sex, a) for k, a in scores.items() if k.startswith(f'PAIR__{svc}__')}
            v['utility'] = {}; v['task_summary'] = {}; v['subgroup'] = {}; v['by_sex'] = {}
            for t in C.TASKS:
                k = C.K[t]; vv = C.task_vectors(y[t], bank[svc]['DEV_REPORT'][t], k)
                per_nll[(svc, s, t)] = vv['nll']; per_vec[(svc, s, t)] = vv
                v['task_summary'][t] = {m_: float(vv[m_].mean()) for m_ in vv}
                v['by_sex'][t] = {f'sex{g}': {m_: float(vv[m_][sex == g].mean()) for m_ in vv} for g in (0, 1)}
                for ref in ('U0', 'Ucal'):
                    rv = C.task_vectors(y[t], bank[ref]['DEV_REPORT'][t], k)
                    for metric, d in {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'], 'brier_excess': vv['brier'] - rv['brier']}.items():
                        dr = (W * d[None, :]).sum(1) / Wsum
                        util[(svc, s, t, ref, metric)] = (float(d.mean()), dr)
                        v['utility'][f'{t}|{ref}|{metric}'] = {'point': float(d.mean()), 'se': float(np.std(dr, ddof=1)), 'cap': C.CAPS[metric],
                                                               'point_pass': bool(d.mean() <= C.CAPS[metric])}
            vv = C.task_vectors(y['occupation'], bank[svc]['DEV_REPORT']['occupation'], 5)
            for g in (0, 1):
                m_ = sex == g; d = vv['brier'] - u0['occupation']['brier']; Wg = W[:, m_]
                dr = (Wg * d[m_][None, :]).sum(1) / Wg.sum(1)
                util[(svc, s, 'subgroup', g)] = (float(d[m_].mean()), dr)
                v['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d[m_].mean()), 'se': float(np.std(dr, ddof=1)),
                                            'occupation_nll_excess_vs_U0': float((vv['nll'] - u0['occupation']['nll'])[m_].mean())}
            v['utility_point_feasible_12_this_seed'] = all(x['point_pass'] for x in v['utility'].values())
            # bounded diagnostics: classwise occupation NLL contributions and predicted-class frequencies
            yo = y['occupation']; po = bank[svc]['DEV_REPORT']['occupation']
            v['diagnostics'] = {'occupation_classwise_nll_contribution': [float(vv['nll'][yo == c].sum() / len(yo)) for c in range(5)],
                                'occupation_classwise_mean_nll': [float(vv['nll'][yo == c].mean()) if (yo == c).any() else None for c in range(5)],
                                'occupation_true_class_frequency': [float((yo == c).mean()) for c in range(5)],
                                'occupation_predicted_class_frequency': [float((po.argmax(1) == c).mean()) for c in range(5)],
                                'occupation_mean_probability': [float(x) for x in po.mean(0)],
                                'income_predicted_positive_rate': float((bank[svc]['DEV_REPORT']['income'].argmax(1) == 1).mean())}
            srec['services'][svc] = v
        for view in ('income', 'occupation', 'pair'):
            pl = np.where(sex == 1, scores[f'CONTROL__{view}__PLANTED_1'], scores[f'CONTROL__{view}__PLANTED_0'])
            srec['controls'][view] = {'wire': rr['controls'][view]['wire'], 'shuffled_null_auc': auc(sex, scores[f'CONTROL__{view}__SHUFFLED']),
                                      'planted_bit_auc': auc(sex, pl), 'planted_attack_validation_auc': rr['controls'][view]['planted_attack_validation_auc'],
                                      'shuffled_attack_validation_auc': rr['controls'][view]['shuffled_attack_validation_auc']}
        srec['coverage_attack_roles'] = rr['coverage']
        res['per_seed'][f'S{s}'] = srec

    SEV = ('AVAILABLE', 'DIAGNOSTIC_NOT_MATCHED', 'COMPARATOR_INELIGIBLE', 'MISSING')

    def role_avail(s, role):
        r = sel['roles'][str(s)].get(role, {}); st = r.get('status')
        if role not in svc_of[s] or st in (None, 'MISSING', 'UNMATCHED_CAPACITY'):
            return 'MISSING'
        if st == 'NEAREST_DIAGNOSTIC_NOT_MATCHED':
            return 'DIAGNOSTIC_NOT_MATCHED'
        if st == 'SAME_COUNT_INELIGIBLE_FALLBACK' or (role in ('ORD_CNT', 'NEURAL_RAW', 'H_LJ') and not r.get('point_pass_12', True)):
            return 'COMPARATOR_INELIGIBLE'
        return 'AVAILABLE'

    def worst(*a):
        return max(a, key=SEV.index)

    def evaluate(spec):
        e = spec['estimand']; s = spec['seed']
        if e['type'] in ('auc_diff', 'auc_diff_slate7', 'nll_diff'):
            a = worst(role_avail(s, e['plus']), role_avail(s, e['minus']))
            if e['plus'] not in svc_of[s] or e['minus'] not in svc_of[s]:
                return None, 'MISSING', False
            p, m = svc_of[s][e['plus']], svc_of[s][e['minus']]
            if e['type'] == 'nll_diff':
                d = per_nll[(p, s, e['task'])] - per_nll[(m, s, e['task'])]
                return (float(d.mean()), (W * d[None, :]).sum(1) / Wsum), a, p == m
            tag = e['view'] if e['type'] == 'auc_diff' else e['view'] + '_slate7'
            return (auc_pt[(p, s, tag)] - auc_pt[(m, s, tag)], auc_dr[(p, s, tag)] - auc_dr[(m, s, tag)]), a, p == m
        if e['type'] == 'auc_view_diff':
            svc = svc_of[s][e['service']]
            return (auc_pt[(svc, s, e['plus_view'])] - auc_pt[(svc, s, e['minus_view'])],
                    auc_dr[(svc, s, e['plus_view'])] - auc_dr[(svc, s, e['minus_view'])]), 'AVAILABLE', False
        if e['type'] == 'utility':
            return util[(svc_of[s][e['service']], s, e['task'], e['reference'], e['metric'])], 'AVAILABLE', False
        if e['type'] == 'subgroup_occupation_brier_vs_U0':
            return util[(svc_of[s][e['service']], s, 'subgroup', e['group'])], 'AVAILABLE', False
        raise ValueError(f'unknown estimand {e}')
    ms_delta = {}
    for s in C.SEEDS:
        for rn, r in sel['roles'][str(s)].items():
            ms_delta[(s, rn)] = r.get('delta')
    E, E2 = {}, {}
    for roster, fam, cK, store in ((lock['primary_roster'], f"primary (K = {inf['K_primary']})", cs, E),
                                   (lock['secondary_roster'], f"secondary (K = {inf['K_secondary']})", cs2, E2)):
        for sp in roster:
            v, a, ident = evaluate(sp)
            if v is None:
                store[sp['key']] = {'status': 'MISSING', 'availability': 'MISSING', 'note': 'comparator role empty in the final bank (coverage gap)',
                                    'claim': sp.get('claim'), 'kind': sp['kind'], 'margin': sp['margin'], 'clause_L1': None, 'clause_U1': None, 'clause_pass': None,
                                    'point_pass': None}
                continue
            rec = endpoint(*v, sp, c1, cK, fam, a, ident)
            if sp['kind'] == 'match':
                rec['MODEL_SELECT_achieved_mismatch_SCM_minus_comparator'] = (-ms_delta.get((sp['seed'], sp['estimand']['minus']))
                                                                            if ms_delta.get((sp['seed'], sp['estimand']['minus'])) is not None else None)
            store[sp['key']] = rec
    if len(E) != inf['K_primary'] or len(E2) != inf['K_secondary']:
        raise SystemExit('HALT: roster size check failed')
    res['endpoints_primary'] = E; res['endpoints_secondary'] = E2
    EA = {**E, **E2}
    # ---------------- controls
    lo, hi = inv['readers']['controls']['null_band']; pmin = inv['readers']['controls']['planted_min_auc']
    vals = [c for s in C.SEEDS for c in res['per_seed'][f'S{s}']['controls'].values()]
    planted = all(c['planted_bit_auc'] >= pmin and c['planted_attack_validation_auc'] >= pmin for c in vals)
    null = all(lo <= c['shuffled_null_auc'] <= hi and lo <= c['shuffled_attack_validation_auc'] <= hi for c in vals)
    CV = bool(planted and null and len(vals) == 6)
    res['controls'] = {'PLANTED_PASS': bool(planted), 'NULL_PASS': bool(null), 'CONTROLS_VALID': CV}
    # ---------------- conditions with failure categories (frozen design MF-4)
    ck = lock['claim_keys']

    def c_ben(k, m=0.0):
        e = E[k]
        if e.get('availability') != 'AVAILABLE':
            return {'value': False, 'category': 'UNAVAILABLE', 'detail': e.get('availability')}
        if e['point'] - c1 * e['se'] > m:
            return {'value': True, 'clause_L1': e['point'] - c1 * e['se']}
        return {'value': False, 'clause_L1': e['point'] - c1 * e['se'],
                'category': 'ESTABLISHED_SHORTFALL' if e['family_upper'] < m else ('POINT_SHORTFALL' if e['point'] <= m else 'UNRESOLVED')}

    def c_nw(k, m=0.010):
        e = E[k]
        if e.get('availability') != 'AVAILABLE':
            return {'value': False, 'category': 'UNAVAILABLE', 'detail': e.get('availability')}
        if e['point'] + c1 * e['se'] <= m:
            return {'value': True, 'clause_U1': e['point'] + c1 * e['se']}
        return {'value': False, 'clause_U1': e['point'] + c1 * e['se'],
                'category': 'ESTABLISHED_VIOLATION' if e['family_lower'] > m else ('POINT_EXCESS' if e['point'] > m else 'UNRESOLVED')}

    def c_flag(ok, cat):
        return {'value': bool(ok)} if ok else {'value': False, 'category': cat}

    def c_pm(k):
        e = EA[k]
        if e.get('availability') != 'AVAILABLE':
            return {'value': False, 'category': 'UNAVAILABLE', 'detail': e.get('availability')}
        return c_flag(abs(e['point']) <= 0.001, 'POINT_FAILURE') | {'point': e['point'], 'interval_status': e['status']}
    NOM = sel['status'] == 'NOMINEE_SELECTED'
    IS_C0 = any(sel['roles'][str(s)]['SCM']['release'] == sel['roles'][str(s)]['C0']['release'] for s in C.SEEDS)
    UG = all(E[k].get('point_pass') for k in ck['utility_24'])
    SG = all(E[k].get('point_pass') for k in ck['subgroup'])
    head_conds = [('CONTROLS_VALID', c_flag(CV, 'CONTROL_FAILURE'), 'CONTROL_FAILURE'), ('NOMINEE_SELECTED', c_flag(NOM, 'UNAVAILABLE'), 'NOMINEE_INELIGIBLE'),
                  ('NOMINEE_NOT_C0', c_flag(not IS_C0, 'UNAVAILABLE'), 'NOMINEE_IS_C0')]
    tail_conds = [('DEV_UTILITY_POINT_GATE_24', c_flag(UG, 'POINT_FAILURE'), 'NOMINEE_FAILS_DEV_UTILITY_GATE'),
                  ('DEV_SUBGROUP_POINT_GUARDS_4', c_flag(SG, 'POINT_FAILURE'), 'NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD')]

    def label(conds, success, failure):
        for name, c, lab in conds:
            if not c['value'] and lab is not None:
                return lab
        return success if all(c['value'] for _, c, _ in conds) else failure

    def cvec(conds):
        return {name: c for name, c, _ in conds}
    # ---- claim M (primary)
    ms_match = {s: role_avail(s, 'ORD_NLLM') == 'AVAILABLE' for s in C.SEEDS}
    condsM = head_conds + [('ORD_NLLM_MODEL_SELECT_match_both_seeds', c_flag(all(ms_match.values()), 'UNAVAILABLE'), 'MATCH_UNAVAILABLE')]
    condsM += [(f'DEV_point_match|{k}', c_pm(k), 'MATCH_DID_NOT_CARRY_OVER') for k in ck['match_nll']]
    condsM += tail_conds + [(f'benefit_L1_gt_0|{k}', c_ben(k), None) for k in ck['match_benefit']] + [(f'worsening_U1_le_0.010|{k}', c_nw(k), None) for k in ck['match_worsening']]
    M = {'conditions': cvec(condsM), 'decision': label(condsM, 'MATCHED_ADVANTAGE_ESTABLISHED', 'MATCHED_ADVANTAGE_NOT_ESTABLISHED')}
    M['material_0.020'] = {'status': 'MATERIAL_ADVANTAGE_ESTABLISHED' if (M['decision'] == 'MATCHED_ADVANTAGE_ESTABLISHED' and all(c_ben(k, 0.020)['value'] for k in ck['match_benefit']))
                           else 'NOT_ESTABLISHED', 'per_seed_L1_gt_0.020': {k: c_ben(k, 0.020) for k in ck['match_benefit']}}
    comp_only = all(c_ben(k)['value'] for k in ck['match_benefit']) and all(c_nw(k)['value'] for k in ck['match_worsening'])
    M['comparison_only_status'] = (('BENEFIT_AND_NON_WORSENING_CLAUSES_PASS' if comp_only else 'COMPARISON_CLAUSES_NOT_ALL_PASS')
                                   + (' (not cost-matched on DEV_REPORT)' if M['decision'] in ('MATCH_UNAVAILABLE', 'MATCH_DID_NOT_CARRY_OVER') else ''))

    def direction(k):
        e = EA[k]
        if e.get('availability') != 'AVAILABLE':
            return e.get('availability')
        return 'CARRIED_OVER' if abs(e['point']) <= 0.001 else ('COMPARATOR_SPENT_MORE' if e['point'] < -0.001 else 'SCM_SPENT_MORE')
    M['carry_over_direction'] = {k: direction(k) for k in ck['match_nll']}
    M['carry_over_note'] = ('per-seed direction reported in every case (CARRIED_OVER, SCM_SPENT_MORE, COMPARATOR_SPENT_MORE); it states only who spent more '
                            'occupation NLL on DEV_REPORT; a drift in either direction is MATCH_DID_NOT_CARRY_OVER')
    # ---- matched components: strict fixed sequence M -> IBSI -> SHUF inside M's slice (MF-1)
    def component(keys, role):
        if not all(role_avail(s, role) == 'AVAILABLE' for s in C.SEEDS):
            return 'UNAVAILABLE'
        if not all(c_pm(k)['value'] for k in keys if 'occupation_nll' in k):
            return 'MATCH_DID_NOT_CARRY_OVER'
        return 'PASS' if all(c_ben(k)['value'] for k in keys if 'coalition_benefit' in k) else 'FAIL'
    if M['decision'] != 'MATCHED_ADVANTAGE_ESTABLISHED':
        mc = {'IBSI': 'NOT_TESTED_M_NOT_ESTABLISHED', 'SHUF': 'NOT_TESTED_M_NOT_ESTABLISHED'}
    else:
        ib = component(ck['component_ibsi'], 'IBSI_NLLM')
        mc = {'IBSI': ib, 'SHUF': component(ck['component_shuf'], 'SHUF_NLLM') if ib == 'PASS' else 'NOT_TESTED_SEQUENCE_STOPPED'}
    MC = {**mc, 'decision': 'COMPONENTS_ESTABLISHED' if (mc['IBSI'] == 'PASS' and mc['SHUF'] == 'PASS') else
          ('NOT_TESTED_M_NOT_ESTABLISHED' if mc['IBSI'].startswith('NOT_TESTED') else 'COMPONENTS_NOT_ESTABLISHED: first non-PASS ' + ('IBSI' if mc['IBSI'] != 'PASS' else 'SHUF')),
          'descriptive_status': {'IBSI': component(ck['component_ibsi'], 'IBSI_NLLM'), 'SHUF': component(ck['component_shuf'], 'SHUF_NLLM')},
          'rule': "strict fixed sequence M -> IBSI -> SHUF inside claim M's alpha/4 slice; descriptive unless tested; shuffled controls are component "
                  "comparisons, not permutation tests; a PASS is about the linked pair at the achieved match only"}
    # ---- claim K (same capacity)
    cnt_reached = all(role_avail(s, 'ORD_CNT') != 'MISSING' for s in C.SEEDS)
    cnt_elig = all(role_avail(s, 'ORD_CNT') == 'AVAILABLE' for s in C.SEEDS)
    condsK = head_conds + [('ORD_CNT_reached_K_star_both_seeds', c_flag(cnt_reached, 'UNAVAILABLE'), 'UNMATCHED_CAPACITY'),
                           ('ORD_CNT_MODEL_SELECT_point_eligible_both_seeds', c_flag(cnt_elig, 'UNAVAILABLE'), 'COMPARATOR_INELIGIBLE')]
    condsK += tail_conds + [(f'benefit_L1_gt_0|{k}', c_ben(k), None) for k in ck['capacity_benefit']] + [(f'worsening_U1_le_0.010|{k}', c_nw(k), None) for k in ck['capacity_worsening']]
    K = {'conditions': cvec(condsK), 'decision': label(condsK, 'SAME_CAPACITY_ADVANTAGE_ESTABLISHED', 'SAME_CAPACITY_ADVANTAGE_NOT_ESTABLISHED')}
    K['material_0.020'] = {'status': 'MATERIAL_ADVANTAGE_ESTABLISHED' if (K['decision'] == 'SAME_CAPACITY_ADVANTAGE_ESTABLISHED' and all(c_ben(k, 0.020)['value'] for k in ck['capacity_benefit']))
                           else 'NOT_ESTABLISHED', 'per_seed_L1_gt_0.020': {k: c_ben(k, 0.020) for k in ck['capacity_benefit']}}
    K['comparison_only_status'] = 'BENEFIT_AND_NON_WORSENING_CLAUSES_PASS' if (all(c_ben(k)['value'] for k in ck['capacity_benefit']) and all(c_nw(k)['value'] for k in ck['capacity_worsening'])) else 'COMPARISON_CLAUSES_NOT_ALL_PASS'
    if K['decision'] != 'SAME_CAPACITY_ADVANTAGE_ESTABLISHED':
        K['IBSI_CNT'] = 'NOT_TESTED_K_NOT_ESTABLISHED'
    elif not all(role_avail(s, 'IBSI_CNT') == 'AVAILABLE' for s in C.SEEDS):
        K['IBSI_CNT'] = 'COMPARATOR_INELIGIBLE'
    else:
        K['IBSI_CNT'] = 'PASS' if all(c_ben(k)['value'] for k in ck['capacity_ibsi']) else 'FAIL'
    # ---- claim B (competitive)
    condsB = head_conds + tail_conds + [(f'benefit_L1_gt_0|{k}', c_ben(k), None) for k in ck['competitive_benefit']] + \
        [(f'worsening_U1_le_0.010|{k}', c_nw(k), None) for k in ck['competitive_worsening']]
    B = {'eligible_set': ['NEURAL_RAW', 'H_LJ'], 'conditions': cvec(condsB), 'decision': label(condsB, 'SUPERIOR_TO_ALL_ELIGIBLE', 'NOT_SUPERIOR_TO_ALL_ELIGIBLE')}
    B['material_0.020'] = {'status': 'MATERIAL_SUPERIORITY_ESTABLISHED' if (B['decision'] == 'SUPERIOR_TO_ALL_ELIGIBLE' and all(c_ben(k, 0.020)['value'] for k in ck['competitive_benefit']))
                           else 'NOT_ESTABLISHED', 'per_seed_L1_gt_0.020': {k: c_ben(k, 0.020) for k in ck['competitive_benefit']}}
    B['comparison_only_status'] = 'BENEFIT_AND_NON_WORSENING_CLAUSES_PASS' if (all(c_ben(k)['value'] for k in ck['competitive_benefit']) and all(c_nw(k)['value'] for k in ck['competitive_worsening'])) else 'COMPARISON_CLAUSES_NOT_ALL_PASS'
    # ---- claim C (subgroup)
    sg = [E[k] for k in ck['subgroup']]
    Cc = {'about': 'the nominee' if NOM else 'the ineligible diagnostic fallback',
          'guards_clause_pass': {k: c_nw(k, 0.005)['value'] for k in ck['subgroup']}, 'guards_family_status': {k: E[k]['status'] for k in ck['subgroup']},
          'point_status': 'SUBGROUP_POINT_GUARDS_PASS' if SG else 'SUBGROUP_POINT_GUARD_FAILED',
          'decision': 'SUBGROUP_QUALIFIED' if all(c_nw(k, 0.005)['value'] for k in ck['subgroup']) else ('SUBGROUP_HARM' if any(v['family_lower'] > 0.005 for v in sg) else 'SUBGROUP_UNRESOLVED')}
    # ---- full utility-vector, subgroup and recipient differences beside each comparison (descriptive, MARGINAL_DESCRIPTIVE intervals)
    diffs = {}
    for s in C.SEEDS:
        scm = svc_of[s]['SCM']
        for role in ('ORD_NLLM', 'ORD_CNT', 'IBSI_NLLM', 'IBSI_CNT', 'SHUF_NLLM', 'NEURAL_RAW', 'H_LJ', 'C0', 'HIST_SCM'):
            if role not in svc_of[s]:
                continue
            o = svc_of[s][role]; row = {'availability': role_avail(s, role), 'identical_release': o == scm}

            def put(name, d=None, pt=None, dr=None):
                if d is not None:
                    pt, dr = float(d.mean()), (W * d[None, :]).sum(1) / Wsum
                se = float(np.std(dr, ddof=1))
                row[name] = {'point': float(pt), 'se': se, 'marginal95': [float(pt - cm * se), float(pt + cm * se)], 'kind': 'MARGINAL_DESCRIPTIVE'}
            for t in C.TASKS:
                for m_ in ('nll', 'brier', 'accuracy'):
                    put(f'SCM_minus_role|{t}_{m_}', per_vec[(scm, s, t)][m_] - per_vec[(o, s, t)][m_])
            for g in (0, 1):
                msk = sex == g; Wg = W[:, msk]
                for m_ in ('brier', 'nll'):
                    d = (per_vec[(scm, s, 'occupation')][m_] - per_vec[(o, s, 'occupation')][m_])[msk]
                    put(f'SCM_minus_role|occupation_{m_}|sex{g}', pt=d.mean(), dr=(Wg * d[None, :]).sum(1) / Wg.sum(1))
            if o in rr_services[s]:
                for view in ('income', 'occupation', 'pair'):
                    put(f'SCM_minus_role|auc_{view}', pt=auc_pt[(scm, s, view)] - auc_pt[(o, s, view)], dr=auc_dr[(scm, s, view)] - auc_dr[(o, s, view)])
            row['income_expenditure_qualifier'] = 'INCOME_EXPENDITURE_NOT_MATCHED' if row['SCM_minus_role|income_nll']['point'] > 0.001 else 'income NLL within 0.001 or lower for SCM'
            dev12 = res['per_seed'][f'S{s}']['services'][o]['utility']
            row['comparator_DEV_point_feasible_12'] = all(x['point_pass'] for x in dev12.values())
            row['comparator_DEV_failed_point_clauses'] = [k for k, x in dev12.items() if not x['point_pass']]
            diffs[f'{role}|s{s}'] = row
    # ---- separate status fields (MF-5)
    ut = {k: E[k]['status'] for k in ck['utility_24']}
    t_star = sel['nomination']['t_star']; gstar = [g for g in sel['nomination']['grid'] if g['t'] == t_star][0]
    ms_orig = {s: sel['per_seed_measurements'][str(s)][sel['roles'][str(s)]['SCM']['release']] for s in C.SEEDS}
    res['separate_status_fields'] = {
        'MODEL_SELECT_original_point_eligibility_SCM': {'status': 'ORIGINAL_POINT_FEASIBLE_24' if all(ms_orig[s]['point_pass_12'] for s in C.SEEDS) else 'ORIGINAL_POINT_INFEASIBLE',
                                                        'failed_keys': [f"s{s}|{c['task']}|{c['reference']}|{c['metric']}" for s in C.SEEDS for c in ms_orig[s]['clauses'] if not c['point_pass']]},
        'MODEL_SELECT_conservative_nomination': {'status': 'NOMINATION_ELIGIBLE' if NOM else 'INELIGIBLE_FALLBACK', 'failed_conditions': gstar['failed_conditions']},
        'DEV_REPORT_point': {'status': 'POINT_FEASIBLE_24' if UG else 'POINT_INFEASIBLE', 'failed_keys': [k for k in ck['utility_24'] if not E[k]['point_pass']]},
        'DEV_REPORT_confidence': {'counts': {x: sum(v == x for v in ut.values()) for x in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
                                  'status': ('CONFIDENCE_FEASIBLE_24' if all(v == 'PASS' for v in ut.values()) else 'ESTABLISHED_UTILITY_VIOLATION'
                                             if any(v == 'ESTABLISHED_VIOLATION' for v in ut.values()) else 'CONFIDENCE_UNRESOLVED')},
        'subgroup_point': Cc['point_status'], 'subgroup_family_status': Cc['guards_family_status'], 'subgroup_claim': Cc['decision'],
        'per_role_point_status': {f'{rn}|s{s}': {'MODEL_SELECT_12': sel['roles'][str(s)][rn].get('point_pass_12'),
                                                 'DEV_REPORT_12': res['per_seed'][f'S{s}']['services'][svc_of[s][rn]]['utility_point_feasible_12_this_seed']}
                                  for s in C.SEEDS for rn in svc_of[s]}}
    res['claims'] = {'M_MATCHED_ORDINARY': M, 'M_COMPONENTS_IBSI_SHUF': MC, 'K_SAME_CAPACITY': K, 'B_COMPETITIVE': B, 'C_SUBGROUP': Cc,
                     'D_JOINT_COMPONENT': 'NOT_CLAIMED', 'E_NOVELTY': 'NOT_CLAIMED (existing-method engineering comparison on previously used development records)'}
    res['comparison_differences'] = diffs
    # ---- research decision (MF-6) with mandatory qualifiers
    d = M['decision']
    res['research_decision'] = ('ENGINEERING_REFERENCE_FROZEN_AT_ORIGINAL_TARGET' if M['material_0.020']['status'] == 'MATERIAL_ADVANTAGE_ESTABLISHED' else
                                'ENGINEERING_REFERENCE_FROZEN_POSITIVE_BELOW_ORIGINAL_TARGET' if d == 'MATCHED_ADVANTAGE_ESTABLISHED' else
                                'FRONTIER_DELIVERED_MATCH_UNAVAILABLE' if d == 'MATCH_UNAVAILABLE' else
                                'COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE' if d == 'MATCH_DID_NOT_CARRY_OVER' else
                                'COMPARISON_CLOSED_CONTROL_FAILURE' if d == 'CONTROL_FAILURE' else f'COMPARISON_CLOSED_{d}')
    res['research_decision_qualifiers'] = {'DEV_utility_confidence': res['separate_status_fields']['DEV_REPORT_confidence']['status'], 'claim_C': Cc['decision'],
                                           'ORD_NLLM_DEV_match': {k: {'status': EA[k].get('status'), 'point_status': EA[k].get('point_status'), 'direction': direction(k)} for k in ck['match_nll']},
                                           'claim_B': B['decision'],
                                           'note': 'a frozen engineering reference never authorises opening the sealed assessment; competitive only under SUPERIOR_TO_ALL_ELIGIBLE'}
    # ---- registered bets (scored; never used for a decision)
    roles0 = sel['roles']
    p2 = all(roles0[str(s)]['ORD_NLLM'].get('status') == 'MATCHED' for s in C.SEEDS)
    bets = {'P1': ('HELD' if all(roles0[str(s)]['SCM']['release'] != roles0[str(s)]['HIST_SCM']['release'] for s in C.SEEDS) else 'NOT_HELD', 0.95),
            'P2': ('HELD' if p2 else 'NOT_HELD', 0.70),
            'P3': (('HELD' if all(roles0[str(s)]['ORD_NLLM']['proxy'] - roles0[str(s)]['SCM']['proxy'] >= 0.020 for s in C.SEEDS) else 'NOT_HELD') if p2 else 'VOID', 0.70),
            'P4': ('HELD' if p2 and all(c_pm(k)['value'] for k in ck['match_nll']) else 'NOT_HELD', 0.35),
            'P5': ('HELD' if p2 and all(E[k].get('point', -1) >= 0.020 for k in ck['match_benefit']) else 'NOT_HELD', 0.55),
            'P6': ('HELD' if UG else 'NOT_HELD', 0.75), 'P7': ('HELD' if SG else 'NOT_HELD', 0.50),
            'P8': ('HELD' if M['decision'] == 'MATCHED_ADVANTAGE_ESTABLISHED' else 'NOT_HELD', inv['predictions']['P8'])}
    scored = [(p, 1.0 if o == 'HELD' else 0.0) for o, p in bets.values() if o != 'VOID']
    res['registered_bets'] = {'outcomes': {k: {'outcome': o, 'probability': p} for k, (o, p) in bets.items()},
                              'mean_brier_non_void': float(np.mean([(p - y_) ** 2 for p, y_ in scored])) if scored else None}
    # ---------------- bounded diagnostics (label-free parts and fitted vs held-out costs)
    tsum = json.loads((C.RESULTS / 'TRAJECTORY_SUMMARY.json').read_text())
    diag = {}
    for s in C.SEEDS:
        rel = tsum['per_seed'][f'S{s}']['releases']; srv = res['per_seed'][f'S{s}']['services']
        c0 = svc_of[s]['C0']; dd = {}
        for role, svc in svc_of[s].items():
            rid = sel['roles'][str(s)][role].get('release')
            row = {'service': svc, 'release': rid}
            if rid and not str(rid).startswith('FROZEN_'):
                r = rel[rid]
                row['plugin_vs_C0'] = r['plugin_costs_vs_C0']; row['capacity'] = r['capacity']
                ms = sel['per_seed_measurements'][str(s)][rid]
                ms0 = sel['per_seed_measurements'][str(s)][sel['roles'][str(s)]['C0']['release']]
                row['MODEL_SELECT_vs_C0'] = {f'{t}_{m}': ms['task_summary'][t][m] - ms0['task_summary'][t][m] for t in C.TASKS for m in ('nll', 'brier', 'accuracy')}
            row['DEV_REPORT_vs_C0'] = {f'{t}_{m}': srv[svc]['task_summary'][t][m] - srv[c0]['task_summary'][t][m] for t in C.TASKS for m in ('nll', 'brier', 'accuracy')}
            row['DEV_REPORT_occupation_by_sex'] = srv[svc]['by_sex']['occupation']
            if rid and not str(rid).startswith('FROZEN_'):
                cap = rel[rid]['capacity']
                row['common_cell_check'] = {r_: {'occupied_final_cells': cap[r_]['occupied_cells'], 'distinct_occupation_vectors': cap[r_]['distinct_occupation_vectors'],
                                                 'distinct_pair_vectors': cap[r_]['distinct_pair_vectors'],
                                                 'occupation_output_identifies_cell': cap[r_]['distinct_occupation_vectors'] == cap[r_]['occupied_cells'],
                                                 'pair_adds_no_distinction_beyond_occupation': cap[r_]['distinct_pair_vectors'] == cap[r_]['distinct_occupation_vectors']}
                                            for r_ in C.OUT_ROLES}
            dd[role] = row
        diag[f'S{s}'] = dd
    res['diagnostics'] = {'per_role': diag, 'routing_and_teacher_prediction_differences': {s: v['source_routing_diagnostics'] for s, v in C.admission()['routing'].items()},
                          'note': 'diagnostics locate where prediction quality changes; they authorise no calibration, class-weight, capacity, teacher or merge-objective search; '
                                  'the lookup reader is not population-optimal'}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'DEV_REPORT_RESULTS.json', 'x') as f:
        json.dump(res, f, indent=2, sort_keys=True, default=float)
    cols = ['key', 'family', 'claim', 'kind', 'margin', 'availability', 'point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1', 'clause_pass', 'point_pass', 'status']
    for name, EE in (('PRIMARY_ENDPOINTS.csv', E), ('SECONDARY_ENDPOINTS.csv', E2)):
        with open(out / name, 'x', newline='') as f:
            w = csv.writer(f, lineterminator='\n'); w.writerow(cols)
            for k, v in EE.items():
                w.writerow([k] + [v.get(c) for c in cols[1:]])
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--crash-repair', action='store_true')
    a = ap.parse_args()
    r = run(a.out, event_name='DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP_CRASH_REPAIR' if a.crash_repair else 'DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP')
    print(json.dumps({'claims': {k: (v['decision'] if isinstance(v, dict) and 'decision' in v else v) for k, v in r['claims'].items()}, 'controls': r['controls'],
                      'separate': r['separate_status_fields'], 'research_decision': r['research_decision']}, indent=1, default=str))
