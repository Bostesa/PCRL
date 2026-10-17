"""Verifier unit AMD_VERIFY_DEV_V1 (the one logged independent-verifier DEV_REPORT truth re-read): checks 3 (DEV part), 5, 6.

Own code for: household weights, AUC point and replicate AUCs, loss vectors, endpoint points / SEs / bounds / statuses /
categories, control-validity bands and every claim label from the conjunctions in ANALYSIS_LOCK.json. Reader scores are the
frozen DEV_REPORT_SCORES.npz (winner per release and view from READERS_SEED.json; winner rule re-checked label-free in
AMD_VERIFY_READERS_V1). Person-level labels stay in memory; only aggregates are written.

--dry: invented labels (seeded random), no truth access, to exercise the code path; output goes to V_DEV_DRY.json.
"""
from __future__ import annotations
import json
import sys

sys.path.insert(0, '<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1/.private/pcrl_autonomous_method_development_v1/verifier/independent')
import numpy as np
import vcommon as V
from amd import common as C

DRY = '--dry' in sys.argv
lock = json.loads((V.RES / 'ANALYSIS_LOCK.json').read_text())
rep = json.loads((V.RES / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
inf = lock['inference']
SEED, BB = inf['bootstrap_seed'], inf['bootstrap_B']

if DRY:
    F = C.load_role('DEV_REPORT', ('features',))
    rng = np.random.default_rng(7)
    n = len(F['row_id'])
    T = {'row_id': F['row_id'], 'unit': F['unit'], 'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n), 'sex': rng.integers(0, 2, n)}
else:
    T = C.dev_report_truth('independent verifier replay of the frozen finalist DEV_REPORT evaluation (checks 3, 5, 6)',
                           'DEV_REPORT_TRUTH_REREAD_BY_AMD_INDEPENDENT_VERIFIER')
sex = np.asarray(T['sex']); unit = np.asarray(T['unit'])
out = {'unit': 'AMD_VERIFY_DEV_V1' + ('_DRY' if DRY else ''), 'utc': C.now(), 'dry_run_invented_labels': DRY}

# ---------------------------------------------------------------- weights and critical values
W, G = V.household_weights(unit, SEED, BB)
from task_headroom.inference import group_weights
W_src, G_src = group_weights(unit, seed=SEED, draws=BB)
out['weights'] = {'G': int(G), 'n': int(len(sex)), 'B': int(W.shape[0]), 'equal_to_group_weights_bitwise': bool(np.array_equal(W, W_src)),
                  'max_abs_diff_vs_group_weights': float(np.max(np.abs(W - W_src))), 'G_src': int(G_src),
                  'row_sums_equal_n_persons_on_average': float(W.sum(1).mean())}
if G != 3955 or len(sex) != 5943:
    raise SystemExit('HALT: household / person counts differ')
c1 = V.tcrit(G, inf['alpha_family_one_sided'] / inf['n_claims'])
cP = V.tcrit(G, 0.05 / (2 * inf['K']['P'])); cS = V.tcrit(G, 0.05 / (2 * inf['K']['S'])); cm = V.tcrit(G, 0.025)
out['critical'] = {'c1': c1, 'c_P': cP, 'c_S': cS, 'marginal': cm, 'expected': inf['critical_values_expected'],
                   'recorded': rep['critical']}
cK = {'P': cP, 'S': cS}

# ---------------------------------------------------------------- per-seed AUCs (winner readers) and loss vectors
auc_pt, auc_dr, lossv, util = {}, {}, {}, {}
svc_list = None
for s in (0, 1):
    with np.load(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') as z:
        B = {k: z[k].copy() for k in z.files}
    fz, fids = C.frozen_bank(s, roles=('DEV_REPORT',))
    if not (np.array_equal(B['row_id__DEV_REPORT'], T['row_id']) and np.array_equal(fids['row_id__DEV_REPORT'], T['row_id'])):
        raise AssertionError('row order differs')
    rd = V.PRIV / 'runs' / f'READERS_{s}'
    rr = json.loads((rd / 'READERS_SEED.json').read_text())
    with np.load(rd / 'DEV_REPORT_SCORES.npz') as z:
        sc = {k: z[k].copy() for k in z.files}
    if not np.array_equal(sc['row_id'], T['row_id']):
        raise AssertionError('score order differs')
    svcs = sorted({k.split('__')[0] for k in B if k.count('__') == 2})
    svc_list = svcs
    for svc in svcs:
        for view in ('income', 'occupation', 'pair'):
            rid = rr['services'][svc]['winner'][view]['reader_id']
            auc_pt[(svc, s, view)] = V.auc_point(sex, sc[rid])
            auc_dr[(svc, s, view)] = V.auc_weighted(sex, sc[rid], W)
        for t in V.TASKS:
            lossv[(svc, s, t)] = V.loss_vectors(T[t], B[f'{svc}__DEV_REPORT__{t}'])
            for ref in ('U0', 'Ucal'):
                for m, d in V.clause_vectors(T[t], B[f'{svc}__DEV_REPORT__{t}'], fz[ref]['DEV_REPORT'][t]).items():
                    util[(svc, s, t, ref, m)] = (float(d.mean()), (W @ d) / W.sum(1))
        d = lossv[(svc, s, 'occupation')]['brier'] - V.loss_vectors(T['occupation'], fz['U0']['DEV_REPORT']['occupation'])['brier']
        for g in (0, 1):
            mk = sex == g
            util[(svc, s, 'guard', g)] = (float(d[mk].mean()), (W[:, mk] @ d[mk]) / W[:, mk].sum(1))
    # controls (DEV part recomputed; ATTACK_VALIDATION parts are the recorded reader-unit values)
    for view in ('income', 'occupation', 'pair'):
        pl = np.where(sex == 1, sc[f'CONTROL__{view}__PLANTED_1'], sc[f'CONTROL__{view}__PLANTED_0'])
        out.setdefault('controls', {})[f's{s}|{view}'] = {
            'planted_bit_auc': V.auc_point(sex, pl), 'shuffled_null_auc': V.auc_point(sex, sc[f'CONTROL__{view}__SHUFFLED']),
            'planted_attack_validation_auc_recorded': rr['controls'][view]['planted_attack_validation_auc'],
            'shuffled_attack_validation_auc_recorded': rr['controls'][view]['shuffled_attack_validation_auc'],
            'report_planted_bit_auc': rep['control_validity']['detail'][f's{s}|{view}']['planted_bit_auc'],
            'report_shuffled_null_auc': rep['control_validity']['detail'][f's{s}|{view}']['shuffled_null_auc'], 'wire': rr['controls'][view]['wire']}
bands = lock['control_validity_bands']
CV = all(c['planted_bit_auc'] >= bands['planted_bit_auc_min'] and abs(c['shuffled_null_auc'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max']
         and c['planted_attack_validation_auc_recorded'] >= bands['planted_bit_auc_min'] and abs(c['shuffled_attack_validation_auc_recorded'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max']
         and c['wire'] == 'FINALIST' for c in out['controls'].values())
out['CV'] = CV

# per-service AUC comparison with the report
auc_cmp = []
for (svc, s, view), p in auc_pt.items():
    r = rep['per_seed'][f'S{s}']['services'][svc]['recovery'][view]
    se = float(np.std(auc_dr[(svc, s, view)], ddof=1))
    auc_cmp.append({'svc': svc, 'seed': s, 'view': view, 'auc': p, 'auc_recorded': r['auc'], 'se': se, 'se_recorded': r['auc_se'],
                    'abs_diff': abs(p - r['auc']), 'se_rel_diff': abs(se - r['auc_se']) / max(r['auc_se'], 1e-300)})
out['auc_by_service'] = auc_cmp

# ---------------------------------------------------------------- endpoints (my sign convention from the estimand table)
def my_point_draws(e):
    s = e['seed']; key = e['key']; typ = e['type']
    if typ == 'utility':
        return util[(e['service'], s, e['task'], e['ref'], e['metric'])]
    if typ == 'guard':
        return util[(e['service'], s, 'guard', e['group'])]
    if typ == 'auc' and e['kind'] == 'benefit':           # benefit_X = AUC_pair(X) - AUC_pair(F)
        X = key.split('_vs_')[1].split('|')[0]
        return auc_pt[(X, s, 'pair')] - auc_pt[('FINALIST', s, 'pair')], auc_dr[(X, s, 'pair')] - auc_dr[('FINALIST', s, 'pair')]
    if typ == 'auc' and e['kind'] == 'no_excess':         # worsening_X,r = AUC_r(F) - AUC_r(X)
        X = key.split('_vs_')[1].split('|')[0]; r = key.split('|')[1].split('_worsening')[0]
        return auc_pt[('FINALIST', s, r)] - auc_pt[(X, s, r)], auc_dr[('FINALIST', s, r)] - auc_dr[(X, s, r)]
    if typ == 'auc' and e['kind'] == 'difference':        # SEC|<X>_minus_F|pair : AUC_pair(X) - AUC_pair(F)
        X = e['a']
        return auc_pt[(X, s, e['view'])] - auc_pt[('FINALIST', s, e['view'])], auc_dr[(X, s, e['view'])] - auc_dr[('FINALIST', s, e['view'])]
    if typ == 'auc_views':
        return auc_pt[('FINALIST', s, e['plus'])] - auc_pt[('FINALIST', s, e['minus'])], auc_dr[('FINALIST', s, e['plus'])] - auc_dr[('FINALIST', s, e['minus'])]
    if typ == 'loss_diff':                                # nll_diff = mean [NLL(F) - NLL(X)]
        d = lossv[('FINALIST', s, e['task'])][e['metric']] - lossv[(e['b'], s, e['task'])][e['metric']]
        return float(d.mean()), (W @ d) / W.sum(1)
    raise ValueError(typ)


eps, cmp_rows = {}, []
enc_sign_ok = True
for e in lock['endpoints']:
    point, draws = my_point_draws(e)
    if e['type'] == 'auc' and e['kind'] in ('benefit', 'no_excess'):  # encoding check: lock a - b equals my estimand sign
        s = e['seed']; v = e['view']
        enc_sign_ok &= abs((auc_pt[(e['a'], s, v)] - auc_pt[(e['b'], s, v)]) - point) == 0.0
    se = float(np.std(draws, ddof=1)); fam = e['family']; m = e['margin']
    Lf, Uf = point - cK[fam] * se, point + cK[fam] * se
    rec = {'point': point, 'se': se, 'family_lower': Lf, 'family_upper': Uf, 'status': V.status(e['kind'], Lf, Uf, m), 'kind': e['kind'], 'margin': m}
    if se == 0.0 and point == 0.0:                       # exactly identical released vectors on this view / task (no sampling variation)
        rec['identical_release_observed'] = True
        if e['kind'] == 'difference':
            rec['status'] = 'IDENTICAL_RELEASE'
    if fam == 'P' and e['kind'] in ('benefit', 'no_excess'):
        rec['clause_L1'] = point - c1 * se; rec['clause_U1'] = point + c1 * se
        rec['category'] = V.category(e['kind'], point, rec['clause_L1'], rec['clause_U1'], Lf, Uf, m)
        rec['clause_pass'] = rec['category'] == 'PASS'
        rec['point_pass'] = (point > m) if e['kind'] == 'benefit' else (point <= m)
        if e['kind'] == 'benefit':
            rec['clause_L1_gt_0020'] = rec['clause_L1'] > 0.020
    eps[e['key']] = rec
    r = rep['endpoints'][e['key']]
    row = {'key': e['key'], 'point': point, 'point_rec': r['point'], 'se': se, 'se_rec': r['se'], 'status': rec['status'], 'status_rec': r['status'],
           'point_abs_diff': abs(point - r['point']), 'se_rel_diff': (abs(se - r['se']) / r['se']) if r['se'] > 0 else abs(se - r['se'])}
    if 'clause_L1' in rec:
        row.update({'L1': rec['clause_L1'], 'L1_rec': r['clause_L1'], 'U1': rec['clause_U1'], 'U1_rec': r['clause_U1'],
                    'category': rec['category'], 'category_rec': r['category'], 'Lf': Lf, 'Lf_rec': r['family_lower'], 'Uf': Uf, 'Uf_rec': r['family_upper']})
    cmp_rows.append(row)
out['endpoints'] = cmp_rows
out['lock_ab_sign_encoding_matches_estimands'] = bool(enc_sign_ok)
pt_tol, se_tol = 1e-12, 1e-9
bad = [r for r in cmp_rows if r['point_abs_diff'] > pt_tol or r['se_rel_diff'] > se_tol or r['status'] != r['status_rec']
       or ('category' in r and r['category'] != r['category_rec'])]
out['endpoint_mismatches'] = bad
out['max_point_abs_diff'] = max(r['point_abs_diff'] for r in cmp_rows)
out['max_se_rel_diff'] = max(r['se_rel_diff'] for r in cmp_rows)

# ---------------------------------------------------------------- comparators' 12-clause point status and finalist utility summary
comp = {}
for s in (0, 1):
    for svc in svc_list:
        comp[f'S{s}|{svc}'] = all(util[(svc, s, t, ref, m)][0] <= V.CAPS[m] for t in V.TASKS for ref in ('U0', 'Ucal') for m in V.CAPS)
out['comparator_dev_point_status_12'] = comp
out['comparator_dev_point_status_12_matches'] = all(comp[f'S{s}|{svc}'] == rep['comparator_dev_point_status_12'][f'S{s}'][svc] for s in (0, 1) for svc in svc_list)
out['finalist_utility_max_normalized'] = {f'S{s}': max(util[('FINALIST', s, t, ref, m)][0] / V.CAPS[m] for t in V.TASKS for ref in ('U0', 'Ucal') for m in V.CAPS) for s in (0, 1)}

# ---------------------------------------------------------------- claims from the lock conjunctions
def keyhold(k, test):
    r = eps[k]
    if test == 'point_le':
        return r['point'] <= r['margin']
    if test == 'clause':
        return r['category'] == 'PASS'
    if test == 'clause_at_0020':
        return r['clause_L1'] > 0.020
    raise ValueError(test)


avail = lock['availability_ms_point_eligible']
in_bank = {svc: all(f'S{s}|{svc}' in comp for s in (0, 1)) for svc in svc_list}


def cond(c, results):
    if c.get('test') == 'cv':
        return CV
    if 'claim_ref' in c:
        return results[c['claim_ref']] in ('DEV_POINT_ELIGIBLE',)
    if c.get('test') == 'av':
        role = c['role']
        return in_bank.get(role, False) and (role == 'TASKONLY' or avail.get(role, False))
    return keyhold(c['endpoint'], c['test'])


claims = {}
results = {}
for cl in lock['claims']:
    name = cl['name']
    if cl['kind'] == 'gate':
        fails = [c['endpoint'] for c in cl['conditions'] if not keyhold(c['endpoint'], c['test'])]
        results[name] = cl['success_label'] if not fails else cl['failure_label']
        claims[name] = {'label': results[name], 'failed': fails, 'n_conditions': len(cl['conditions'])}
    elif cl['kind'] == 'intersection_union':
        vals = [(c.get('endpoint') or c.get('test') or c.get('claim_ref') or c.get('role'), cond(c, results)) for c in cl['conditions']]
        ok = all(v for _, v in vals)
        if ok:
            results[name] = cl['success_label']
        elif 'failure_labels_in_order' in cl:                 # B: first failing item in the registered order
            Ec = claims['E']['failed']
            gate_of = {c['endpoint']: c['gate'] for c in next(x for x in lock['claims'] if x['name'] == 'E')['conditions']}
            fl = cl['failure_labels_in_order']
            if not CV:
                results[name] = fl[0]
            elif any(gate_of[k] == 'UG' for k in Ec):
                results[name] = fl[1]
            elif any(gate_of[k] == 'SG' for k in Ec):
                results[name] = fl[2]
            elif any(gate_of[k] == 'RG' for k in Ec):
                results[name] = fl[3]
            else:
                results[name] = fl[4]
        else:
            results[name] = cl['failure_label']
        rec = {'label': results[name], 'failed': [k for k, v in vals if not v], 'n_conditions': len(vals)}
        if 'nested' in cl:
            rec['nested'] = {cl['nested']['name']: ok and all(keyhold(a['endpoint'], a['test']) for a in cl['nested']['adds']),
                             'L1_values': {a['endpoint']: eps[a['endpoint']]['clause_L1'] for a in cl['nested']['adds']}}
        claims[name] = rec
    elif cl['kind'] == 'fixed_sequence':
        st = cl['steps']; res_steps = {}
        to_ok = all(cond(c, results) for c in st[0]['conditions'])
        res_steps[st[0]['name']] = to_ok
        ss_ok = all(cond(c, results) for c in st[1]['conditions']) if to_ok else None
        res_steps[st[1]['name']] = ss_ok if to_ok else 'NOT_TESTED_SEQUENCE_STOPPED'
        lab = cl['labels']['both'] if (to_ok and ss_ok) else (cl['labels']['to_only'] if to_ok else cl['labels']['none'])
        results[name] = lab
        claims[name] = {'label': lab, 'steps': res_steps}
out['claims'] = claims
rec_labels = {k: (v.get('label') if isinstance(v, dict) else v) for k, v in rep['claims'].items()}
out['claims_recorded'] = rec_labels
out['claims_match'] = all(claims[k]['label'] == rec_labels.get(k) for k in claims)
out['B_nested_recorded'] = rep['claims']['B'].get('nested_0020')

# ---------------------------------------------------------------- E fields: 28-endpoint simultaneous utility confidence, recipient clause counts
ug_keys = [e['key'] for e in lock['endpoints'] if e['type'] in ('utility', 'guard')]
fam_status = [eps[k]['status'] for k in ug_keys]
out['E_simultaneous_utility'] = {'n': len(ug_keys), 'PASS': fam_status.count('PASS'), 'ESTABLISHED_VIOLATION': fam_status.count('ESTABLISHED_VIOLATION'),
                                 'UNRESOLVED': fam_status.count('UNRESOLVED'),
                                 'label': 'CONFIDENCE_FEASIBLE' if all(x == 'PASS' for x in fam_status) else ('ESTABLISHED_UTILITY_VIOLATION' if 'ESTABLISHED_VIOLATION' in fam_status else 'CONFIDENCE_UNRESOLVED'),
                                 'unresolved_keys': [k for k in ug_keys if eps[k]['status'] != 'PASS']}
rkeys = [e['key'] for e in lock['endpoints'] if e['key'].startswith('COMPETITIVE|') and 'worsening' in e['key']]
out['E_recipient_clause_counts'] = {c: sum(eps[k]['category'] == c for k in rkeys) for c in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION', 'POINT_EXCESS')}
out['E_fields_recorded'] = rep['E_fields']

# marginal qualifiers
mq = {}
for s in (0, 1):
    for name, (a, b, t, m) in {'F_minus_TASKONLY_occupation_nll': ('FINALIST', 'TASKONLY', 'occupation', 'nll'),
                               'F_minus_B2TWIN_income_brier': ('FINALIST', 'B2TWIN', 'income', 'brier')}.items():
        d = lossv[(a, s, t)][m] - lossv[(b, s, t)][m]; dr = (W @ d) / W.sum(1); se = float(np.std(dr, ddof=1))
        r = rep['mandatory_qualifiers'][f's{s}'][name]
        mq[f's{s}|{name}'] = {'point': float(d.mean()), 'point_rec': r['point'], 'se': se, 'se_rec': r['se'],
                              'marginal95': [float(d.mean()) - cm * se, float(d.mean()) + cm * se], 'marginal95_rec': r['marginal95']}
out['mandatory_qualifiers'] = mq
V.dump('V_DEV_DRY.json' if DRY else 'V_DEV.json', out)
print(json.dumps({'dry': DRY, 'weights_equal': out['weights']['equal_to_group_weights_bitwise'], 'CV': CV, 'claims': {k: v['label'] for k, v in claims.items()},
                  'claims_recorded': rec_labels, 'claims_match': out['claims_match'], 'n_endpoint_mismatch': len(bad),
                  'max_point_abs_diff': out['max_point_abs_diff'], 'max_se_rel_diff': out['max_se_rel_diff'],
                  'comparators_match': out['comparator_dev_point_status_12_matches'], 'enc_sign_ok': out['lock_ab_sign_encoding_matches_estimands'],
                  'E_simul': out['E_simultaneous_utility']['label']}, default=str))
