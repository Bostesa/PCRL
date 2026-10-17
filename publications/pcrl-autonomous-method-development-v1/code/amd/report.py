"""Unit AMD_REPORT: the one logged coordinator re-read of the already-exposed DEV_REPORT truth (after the committed and
published REPORTING_LOCK), then every endpoint and claim label of the frozen roster in ANALYSIS_LOCK.json, in this one run.

Household bootstrap over DEV_REPORT households (B and seed from the lock), shared across services, endpoints and seeds;
paired differences are person-level within each replicate. Wald bounds point +/- t(G-1) x bootstrap SE. Clause bounds are
one-sided at alpha / n_claims; families are two-sided 95% Bonferroni-t over their K. Conditional on the fitted releases and
readers; the seeds share people; intervals exclude the adaptive development history. Adapted from scm_um/report.py.

Endpoint kinds (sign convention: a positive `benefit` favours the finalist):
- benefit: point = AUC(comparator) - AUC(finalist) on a view; condition = clause_L1 > margin (0 unless stated).
- no_excess: recipient non-worsening, point = AUC(finalist) - AUC(comparator) on a single view, condition clause_U1 <= 0.010;
  utility clauses, point = finalist metric excess over U0 / Ucal, condition point <= cap (point gate) with the family status
  reported; subgroup guards, point = within-SEX occupation summed-Brier excess over U0, condition point <= 0.005.
- difference: descriptive paired difference with interval (no condition).
"""
from __future__ import annotations
import argparse
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


def endpoint(point, draws, kind, margin, c1, cK, refs=()):
    draws = np.asarray(draws, dtype=np.float64)
    se = float(np.std(draws, ddof=1)); L, U = float(point - cK * se), float(point + cK * se)
    rec = {'point': float(point), 'se': se, 'kind': kind, 'margin': margin, 'family_lower': L, 'family_upper': U,
           'clause_L1': float(point - c1 * se), 'clause_U1': float(point + c1 * se), 'status': status_at(kind, L, U, margin), 'at': {}}
    if kind == 'benefit':
        rec['point_pass'] = bool(point > margin); rec['clause_pass'] = bool(rec['clause_L1'] > margin)
        rec['category'] = 'PASS' if rec['clause_pass'] else ('ESTABLISHED_SHORTFALL' if U < margin else ('POINT_SHORTFALL' if not rec['point_pass'] else 'UNRESOLVED'))
    elif kind == 'no_excess':
        rec['point_pass'] = bool(point <= margin); rec['clause_pass'] = bool(rec['clause_U1'] <= margin)
        rec['category'] = 'PASS' if rec['clause_pass'] else ('ESTABLISHED_VIOLATION' if L > margin else
                                                              ('POINT_EXCESS' if not rec['point_pass'] else 'UNRESOLVED'))
    else:
        rec['point_pass'] = None; rec['clause_pass'] = None; rec['category'] = 'DESCRIPTIVE'
    for r in refs:
        rec['at'][str(r)] = {'point_minus_r': float(point - r), 'family_status_at_r': status_at(kind, L, U, r),
                             'clause_status_at_r': ('PASS' if (rec['clause_L1'] > r if kind == 'benefit' else rec['clause_U1'] <= r) else 'NOT_PASS')}
    return rec


def run(out_dir, truth=None, event_purpose=None, event_name='DEV_REPORT_TRUTH_REREAD_BY_AMD_COORDINATOR'):
    from task_headroom.inference import group_weights
    from task_conditioned_representation.inference import auc, bootstrap_auc
    names = ('FINALIST_LOCK.json', 'ANALYSIS_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'REPORTING_LOCK.json')
    if truth is None:
        for name in names:
            if not C.committed(C.RESULTS / name):
                raise PermissionError(f'report requires the committed {name}')
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    rlock = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text()) if truth is None else None
    out = Path(out_dir)
    if (out / 'DEV_REPORT_RESULTS.json').exists():
        raise PermissionError('report results exist (one evaluation)')
    inf = lock['inference']
    T = truth if truth is not None else C.dev_report_truth(event_purpose or
        'autonomous method-development campaign: the one coordinator evaluation of the frozen finalist, bank and readers', event_name)
    y = {t: T[t] for t in C.TASKS}; sex = T['sex']
    W, G = group_weights(T['unit'], seed=inf['bootstrap_seed'], draws=inf['bootstrap_B'])
    if G != 3955 or len(sex) != 5943:
        raise SystemExit(f'HALT: DEV_REPORT has {G} households and {len(sex)} persons, not 3955 / 5943')
    Wsum = W.sum(1)
    c1 = crit(G, inf['alpha_family_one_sided'] / inf['n_claims'])
    cK = {fam: crit(G, 0.05 / (2 * k)) for fam, k in inf['K'].items()}; cm = crit(G, 0.025)
    res = {'schema': 'AMD1-dev-report-v1', 'utc': C.now(),
           'status': 'EXPLORATORY DEVELOPMENT EVIDENCE (already-exposed DEV_REPORT people; previously used records; adaptive development history not covered)',
           'households': int(G), 'persons': int(len(sex)), 'subgroup_households_per_sex': {int(g): int(len(np.unique(T['unit'][sex == g]))) for g in (0, 1)},
           'critical': {'t_one_sided_alpha_over_n_claims': c1, 'n_claims': inf['n_claims'], 't_family': cK, 'K': inf['K'], 't_marginal_95': cm},
           'locks_sha256': {n: C.sha(C.RESULTS / n) for n in names if (C.RESULTS / n).exists()}, 'per_seed': {}}
    auc_pt, auc_dr, util, per_vec = {}, {}, {}, {}
    for s in C.SEEDS:
        bank, ids, man = load_service_bank(s)
        frozen, fids = C.frozen_bank(s, roles=('DEV_REPORT',))
        if not (np.array_equal(ids['row_id__DEV_REPORT'], T['row_id']) and np.array_equal(fids['row_id__DEV_REPORT'], T['row_id'])):
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
        srec = {'service_of_role': ps['service_of_role'], 'aliases': ps['aliases'], 'services': {}, 'controls': {}}
        refs = {r: {t: frozen[r]['DEV_REPORT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
        u0 = {t: C.task_vectors(y[t], refs['U0'][t], C.K[t]) for t in C.TASKS}
        for svc in rr['services']:
            sv = rr['services'][svc]; v = {'recovery': {}, 'recovery_slate7': {}, 'utility': {}, 'subgroup': {}, 'task_summary': {}, 'by_sex': {}}
            for key, wkey in (('recovery', 'winner'), ('recovery_slate7', 'winner_slate7')):
                for view in ('income', 'occupation', 'pair'):
                    w = sv[wkey][view]; sc = scores[w['reader_id']]
                    tag = view if key == 'recovery' else view + '_slate7'
                    auc_pt[(svc, s, tag)] = auc(sex, sc); auc_dr[(svc, s, tag)] = bootstrap_auc(sex, sc, W)
                    se = float(np.std(auc_dr[(svc, s, tag)], ddof=1))
                    v[key][view] = {'winner': w['candidate'], 'attack_validation_auc': w['attack_validation_auc'], 'auc': auc_pt[(svc, s, tag)],
                                    'auc_se': se, 'marginal95': [auc_pt[(svc, s, tag)] - cm * se, auc_pt[(svc, s, tag)] + cm * se]}
            v['pair_reader_slate_auc_descriptive'] = {k.split('__', 2)[2]: auc(sex, a) for k, a in scores.items() if k.startswith(f'PAIR__{svc}__')}
            for t in C.TASKS:
                k = C.K[t]; vv = C.task_vectors(y[t], bank[svc]['DEV_REPORT'][t], k); per_vec[(svc, s, t)] = vv
                v['task_summary'][t] = {m_: float(vv[m_].mean()) for m_ in vv}
                v['by_sex'][t] = {f'sex{g}': {m_: float(vv[m_][sex == g].mean()) for m_ in vv} for g in (0, 1)}
                for ref in ('U0', 'Ucal'):
                    rv = C.task_vectors(y[t], refs[ref][t], k)
                    for metric, d in {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'], 'brier_excess': vv['brier'] - rv['brier']}.items():
                        dr = (W * d[None, :]).sum(1) / Wsum
                        util[(svc, s, t, ref, metric)] = (float(d.mean()), dr)
                        v['utility'][f'{t}|{ref}|{metric}'] = {'point': float(d.mean()), 'se': float(np.std(dr, ddof=1)), 'cap': C.CAPS[metric],
                                                               'point_pass': bool(d.mean() <= C.CAPS[metric])}
            vv = C.task_vectors(y['occupation'], bank[svc]['DEV_REPORT']['occupation'], 5)
            for g in (0, 1):
                m_ = sex == g; d = vv['brier'] - u0['occupation']['brier']; Wg = W[:, m_]
                dr = (Wg * d[m_][None, :]).sum(1) / Wg.sum(1)
                util[(svc, s, 'guard', g)] = (float(d[m_].mean()), dr)
                v['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d[m_].mean()), 'se': float(np.std(dr, ddof=1)),
                                            'occupation_nll_excess_vs_U0': float((vv['nll'] - u0['occupation']['nll'])[m_].mean()),
                                            'income_nll_excess_vs_U0': float((C.task_vectors(y['income'], bank[svc]['DEV_REPORT']['income'], 2)['nll'] - u0['income']['nll'])[m_].mean())}
            v['utility_point_feasible_12_this_seed'] = all(x['point_pass'] for x in v['utility'].values())
            v['guard_point_pass_this_seed'] = all(x['occupation_brier_excess_vs_U0'] <= C.GUARD for x in v['subgroup'].values())
            yo = y['occupation']; po = bank[svc]['DEV_REPORT']['occupation']
            v['diagnostics'] = {'occupation_classwise_mean_nll': [float(vv['nll'][yo == c].mean()) for c in range(5)],
                                'occupation_true_class_frequency': [float((yo == c).mean()) for c in range(5)],
                                'occupation_predicted_class_frequency': [float((po.argmax(1) == c).mean()) for c in range(5)],
                                'income_predicted_positive_rate': float((bank[svc]['DEV_REPORT']['income'].argmax(1) == 1).mean())}
            srec['services'][svc] = v
        for view in ('income', 'occupation', 'pair'):
            pl = np.where(sex == 1, scores[f'CONTROL__{view}__PLANTED_1'], scores[f'CONTROL__{view}__PLANTED_0'])
            srec['controls'][view] = {'wire': rr['controls'][view]['wire'], 'shuffled_null_auc': auc(sex, scores[f'CONTROL__{view}__SHUFFLED']),
                                      'planted_bit_auc': auc(sex, pl), 'planted_attack_validation_auc': rr['controls'][view]['planted_attack_validation_auc'],
                                      'shuffled_attack_validation_auc': rr['controls'][view]['shuffled_attack_validation_auc']}
        srec['coverage_attack_roles'] = rr['coverage']
        res['per_seed'][f'S{s}'] = srec

    # ---------------- endpoints of the frozen roster (ANALYSIS_LOCK; statistical reviewer's frozen design)
    if inf['K'] != {'P': 52, 'S': 20} or inf['n_claims'] != 3:
        raise AssertionError('registered counts differ (K_P 52, K_S 20, n_claims 3)')
    for name, val in (('c1', c1), ('c_P', cK['P']), ('c_S', cK['S']), ('marginal', cm)):
        if abs(val - inf['critical_values_expected'][name]) > 1e-6:
            raise AssertionError(f'critical value {name} {val} differs from the registered {inf["critical_values_expected"][name]}')
    svc_of = {s: res['per_seed'][f'S{s}']['service_of_role'] for s in C.SEEDS}
    wires = {}
    for s in C.SEEDS:
        bank, _, _ = load_service_bank(s)
        wires[s] = {svc: {t: C.wire_sha(bank[svc]['DEV_REPORT'][t]) for t in C.TASKS} for svc in bank}
    eps = {}
    for e in lock['endpoints']:
        s = e['seed']; fam = e['family']; identical = False
        if e['type'] == 'auc':
            a, b = svc_of[s][e['a']], svc_of[s][e['b']]
            point = auc_pt[(a, s, e['view'])] - auc_pt[(b, s, e['view'])]
            draws = auc_dr[(a, s, e['view'])] - auc_dr[(b, s, e['view'])]
            identical = a == b or (e['view'] in C.TASKS and wires[s][a][e['view']] == wires[s][b][e['view']])
        elif e['type'] == 'auc_views':
            a = svc_of[s][e['service']]
            point = auc_pt[(a, s, e['plus'])] - auc_pt[(a, s, e['minus'])]
            draws = auc_dr[(a, s, e['plus'])] - auc_dr[(a, s, e['minus'])]
        elif e['type'] == 'utility':
            a = svc_of[s][e['service']]
            point, draws = util[(a, s, e['task'], e['ref'], e['metric'])]
        elif e['type'] == 'guard':
            a = svc_of[s][e['service']]
            point, draws = util[(a, s, 'guard', e['group'])]
        elif e['type'] == 'loss_diff':
            a, b = svc_of[s][e['a']], svc_of[s][e['b']]
            d = per_vec[(a, s, e['task'])][e['metric']] - per_vec[(b, s, e['task'])][e['metric']]
            point, draws = float(d.mean()), (W * d[None, :]).sum(1) / Wsum
            identical = wires[s][a][e['task']] == wires[s][b][e['task']]
        else:
            raise ValueError('unregistered endpoint type')
        if identical:
            rec = {'point': 0.0, 'se': 0.0, 'kind': e['kind'], 'margin': e['margin'], 'family_lower': 0.0, 'family_upper': 0.0,
                   'clause_L1': 0.0 if fam == 'P' else None, 'clause_U1': 0.0 if fam == 'P' else None,
                   'status': 'PASS' if e['kind'] == 'no_excess' else 'IDENTICAL_RELEASE', 'category': 'PASS' if e['kind'] == 'no_excess' else 'IDENTICAL_RELEASE',
                   'point_pass': e['kind'] == 'no_excess', 'clause_pass': e['kind'] == 'no_excess', 'at': {}}
        else:
            rec = endpoint(point, draws, e['kind'], e['margin'], c1, cK[fam], e.get('refs', ()))
            if fam != 'P':
                rec['clause_L1'] = rec['clause_U1'] = rec['clause_pass'] = None
        flags = list(e.get('flags', [])) + (['IDENTICAL_RELEASE'] if identical and 'IDENTICAL_RELEASE' not in e.get('flags', []) else [])
        rec.update({'family': fam, 'claim': e.get('claim'), 'flags': flags, 'availability': 'COMPARATOR_INELIGIBLE' if 'COMPARATOR_INELIGIBLE' in flags else 'AVAILABLE',
                    'spec': {k: v for k, v in e.items() if k != 'flags'}})
        eps[e['key']] = rec
    res['endpoints'] = eps

    # ---------------- control validity (frozen bands) and availability
    bands = lock['control_validity_bands']
    cv_detail = {}
    for s in C.SEEDS:
        for view, c in res['per_seed'][f'S{s}']['controls'].items():
            ok = (c['planted_bit_auc'] >= bands['planted_bit_auc_min'] and abs(c['shuffled_null_auc'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max']
                  and c['planted_attack_validation_auc'] >= bands['planted_bit_auc_min']
                  and abs(c['shuffled_attack_validation_auc'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max'])
            cv_detail[f's{s}|{view}'] = {'valid': bool(ok), **c}
    CV = all(v['valid'] for v in cv_detail.values())
    avail = lock['availability_ms_point_eligible']

    def category(ep, test):
        if test == 'point_le':
            return 'PASS' if ep['point'] <= ep['margin'] + 0.0 else 'POINT_FAILURE'
        if ep['availability'] != 'AVAILABLE':
            return 'UNAVAILABLE'
        if ep['category'] in ('PASS', 'IDENTICAL_RELEASE'):
            return 'PASS'
        return ep['category']

    def evaluate(c, claims_so_far):
        if 'claim_ref' in c:
            ok = claims_so_far[c['claim_ref']]['holds']
            return {'condition': f"claim {c['claim_ref']}", 'holds': ok, 'category': 'PASS' if ok else 'POINT_FAILURE'}
        if c['test'] == 'cv':
            return {'condition': 'CV (planted and shuffled controls inside the frozen bands, every view and seed)', 'holds': CV,
                    'category': 'PASS' if CV else 'CONTROL_FAILURE'}
        if c['test'] == 'av':
            ok = c['role'] == 'TASKONLY' or bool(avail.get(c['role']))
            return {'condition': f"AV {c['role']}", 'holds': ok, 'category': 'PASS' if ok else 'UNAVAILABLE'}
        ep = eps[c['endpoint']]
        if c['test'] == 'point_le':
            ok = bool(ep['point'] <= ep['margin'])
        elif c['test'] == 'clause':
            ok = bool(ep['clause_pass']) and ep['availability'] == 'AVAILABLE'
        elif c['test'] == 'clause_at_0020':
            ok = ep['clause_L1'] is not None and bool(ep['clause_L1'] > 0.020)
        else:
            raise ValueError('unregistered condition test')
        cat = 'PASS' if ok else (category(ep, c['test']) if c['test'] != 'clause_at_0020' else
                                 ('ESTABLISHED_SHORTFALL' if ep['family_upper'] < 0.020 else ('POINT_SHORTFALL' if ep['point'] <= 0.020 else 'UNRESOLVED')))
        return {'condition': c['endpoint'], 'test': c['test'], 'gate': c.get('gate'), 'holds': ok, 'category': cat, 'point': ep['point'],
                'clause_L1': ep.get('clause_L1'), 'clause_U1': ep.get('clause_U1'), 'family': [ep.get('family_lower'), ep.get('family_upper')]}

    claims = {}
    for cl in lock['claims']:
        name = cl['name']
        if cl['kind'] in ('not_claimed', 'descriptive'):
            claims[name] = {'label': cl['label'], 'holds': None}
            continue
        if cl['kind'] == 'gate':
            conds = [evaluate(c, claims) for c in cl['conditions']]
            ok = all(c['holds'] for c in conds)
            failed = [c for c in conds if not c['holds']]
            failed.sort(key=lambda c: cl['failure_order'].index(c['gate']))
            claims[name] = {'label': cl['success_label'] if ok else cl['failure_label'], 'holds': ok, 'conditions': conds,
                            'failed_in_order': [c['condition'] for c in failed]}
            continue
        if cl['kind'] == 'fixed_sequence':
            steps, stop = [], False
            for st in cl['steps']:
                if stop:
                    steps.append({'name': st['name'], 'status': 'NOT_TESTED_SEQUENCE_STOPPED', 'holds': False}); continue
                conds = [evaluate(c, claims) for c in st['conditions']]
                ok = all(c['holds'] for c in conds)
                steps.append({'name': st['name'], 'holds': ok, 'conditions': conds,
                              'first_failing': next((c for c in conds if not c['holds']), None)})
                stop = not ok
            lab = cl['labels']['both'] if all(s_['holds'] for s_ in steps) else (cl['labels']['to_only'] if steps[0]['holds'] else cl['labels']['none'])
            claims[name] = {'label': lab, 'holds': lab == cl['labels']['both'], 'steps': steps}
            continue
        conds = [evaluate(c, claims) for c in cl['conditions']]
        ok = all(c['holds'] for c in conds)
        if name == 'B':
            if ok:
                lab = cl['success_label']
            elif not CV:
                lab = 'CONTROL_FAILURE'
            elif not claims['E']['holds']:
                gates = {c['gate'] for c in claims['E']['conditions'] if not c['holds']}
                lab = ('FINALIST_FAILS_DEV_UTILITY_GATE' if 'UG' in gates else 'FINALIST_FAILS_DEV_SUBGROUP_POINT_GUARD' if 'SG' in gates
                       else 'FINALIST_FAILS_DEV_RECIPIENT_POINT_GATE')
            else:
                lab = 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'
            nested = [evaluate(c, claims) for c in cl['nested']['adds']]
            claims[name] = {'label': lab, 'holds': ok, 'conditions': conds, 'first_failing': next((c for c in conds if not c['holds']), None),
                            'nested_0020': {'label': cl['nested']['name'] if (ok and all(c['holds'] for c in nested)) else 'NOT_ESTABLISHED',
                                            'conditions': nested}}
        else:
            claims[name] = {'label': cl['success_label'] if ok else cl['failure_label'], 'holds': ok, 'conditions': conds,
                            'first_failing': next((c for c in conds if not c['holds']), None)}
    res['claims'] = claims
    res['control_validity'] = {'valid': CV, 'bands': bands, 'detail': cv_detail}

    # ---------------- marginal descriptive set (no status, no claim role; marginal 95%)
    md = {}
    for s in C.SEEDS:
        Fs = svc_of[s]['FINALIST']
        for role, X in svc_of[s].items():
            if X == Fs:
                continue
            for t in C.TASKS:
                for met, name in (('accuracy', 'accuracy_loss'), ('nll', 'nll'), ('brier', 'brier')):
                    d = per_vec[(Fs, s, t)][met] - per_vec[(X, s, t)][met]
                    if met == 'accuracy':
                        d = -d                                  # accuracy loss: positive means the finalist is worse
                    dr = (W * d[None, :]).sum(1) / Wsum; se = float(np.std(dr, ddof=1)); pt = float(d.mean())
                    md[f'F_minus_{role}|{t}|{name}|s{s}'] = {'point': pt, 'se': se, 'marginal95': [pt - cm * se, pt + cm * se]}
            for g in (0, 1):
                m_ = sex == g; Wg = W[:, m_]
                for met in ('brier', 'nll'):
                    d = per_vec[(Fs, s, 'occupation')][met] - per_vec[(X, s, 'occupation')][met]
                    dr = (Wg * d[m_][None, :]).sum(1) / Wg.sum(1); se = float(np.std(dr, ddof=1)); pt = float(d[m_].mean())
                    md[f'F_minus_{role}|occupation_{met}|sex{g}|s{s}'] = {'point': pt, 'se': se, 'marginal95': [pt - cm * se, pt + cm * se]}
    res['marginal_descriptive'] = md
    res['mandatory_qualifiers'] = {f's{s}': {'F_minus_TASKONLY_occupation_nll': md[f'F_minus_TASKONLY|occupation|nll|s{s}'],
                                             'F_minus_B2TWIN_income_brier': md[f'F_minus_B2TWIN|income|brier|s{s}']} for s in C.SEEDS}

    # ---------------- E separate fields (always reported)
    P = {k: v for k, v in eps.items() if v['family'] == 'P'}
    uk = [k for k in P if k.startswith(('UTILITY|', 'SUBGROUP|'))]
    fam_status = [P[k]['status'] for k in uk]
    rk = [k for k in P if k.startswith('COMPETITIVE|') and '_worsening_' in k]
    res['E_fields'] = {
        'model_select_selection_label': 'EDGE_SELECTED',
        'dev_point_status': {'utility_24': sum(eps[k]['point'] <= eps[k]['margin'] for k in uk if k.startswith('UTILITY|')),
                             'guards_4': sum(eps[k]['point'] <= eps[k]['margin'] for k in uk if k.startswith('SUBGROUP|')),
                             'recipient_8': sum(eps[k]['point'] <= eps[k]['margin'] for k in rk)},
        'simultaneous_utility_confidence': ('CONFIDENCE_FEASIBLE' if all(x == 'PASS' for x in fam_status) else
                                            'ESTABLISHED_UTILITY_VIOLATION' if any(x == 'ESTABLISHED_VIOLATION' for x in fam_status) else 'CONFIDENCE_UNRESOLVED'),
        'simultaneous_utility_counts': {x: fam_status.count(x) for x in sorted(set(fam_status))},
        'recipient_clause_counts': {x: [P[k]['category'] for k in rk].count(x) for x in sorted({P[k]['category'] for k in rk})},
        'flags': {k: v['flags'] for k, v in eps.items() if v['flags']}}
    res['comparator_dev_point_status_12'] = {f'S{s}': {svc: v['utility_point_feasible_12_this_seed'] for svc, v in res['per_seed'][f'S{s}']['services'].items()}
                                            for s in C.SEEDS}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'DEV_REPORT_RESULTS.json', 'x') as f:
        json.dump(res, f, indent=1, sort_keys=True, default=float)
    print(json.dumps({k: v['label'] for k, v in claims.items()}, indent=1))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    a = ap.parse_args(); run(a.out)
