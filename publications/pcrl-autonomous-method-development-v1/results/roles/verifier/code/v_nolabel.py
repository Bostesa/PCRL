"""Verifier unit AMD_VERIFY_NOLABEL_V1: checks 2 (runtime outputs), 3 (MODEL_SELECT utility / guards of the finalist),
4 (PROTOCOL section 6 selection) and the programmatic part of 1 (fit-role exposure of the finalist). No DEV_REPORT label."""
from __future__ import annotations
import csv
import json
import sys
import warnings

sys.path.insert(0, '<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1/.private/pcrl_autonomous_method_development_v1/verifier/independent')
import numpy as np
import vcommon as V
from amd import common as C   # hash-checked loaders only

FIN = 'B3CTL_UNCOUPLED_MIX50_C6A'
out = {'unit': 'AMD_VERIFY_NOLABEL_V1', 'utc': C.now()}


def bank(seed):
    man = json.loads((V.RES / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = V.PRIV / 'bank' / f'SERVICE_BANK_S{seed}.npz'
    sh = C.sha(p)
    if sh != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('bank hash differs')
    with np.load(p) as z:
        return {k: z[k].copy() for k in z.files}, sh


# ---------------------------------------------------------------- check 2: recompute finalist outputs from frozen pieces
feats = {r: C.load_role(r, ('features',)) for r in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'DEV_REPORT')}
chk2 = {}
recomputed = {}
for s in (0, 1):
    rel = json.loads((V.PRIV / 'runs' / 'B' / FIN / f'S{s}' / 'RELEASE.json').read_text())
    te = {t: V.Teacher(json.loads(C.admitted(f'stage_a/S{s}/TREE_TEACHER_{t}.json').read_text())) for t in V.TASKS}
    th = {}
    for arm in ('LOCAL_TREE', 'JOINT_TREE'):
        with np.load(C.admitted(f'stage_b/S{s}/{arm}__E20.npz')) as z:
            th[arm] = {'income': z['theta_income'].copy(), 'occupation': z['theta_occupation'].copy()}
    hrec = rel['h']; thr = np.asarray(rel['thresholds'], dtype=np.float64)
    mu = np.asarray(hrec['mu'], dtype=np.float64); sd = np.asarray(hrec['sd'], dtype=np.float64)
    w = np.asarray(hrec['w'], dtype=np.float64); b = float(hrec['b'])
    B, bsha = bank(s)
    with np.load(V.PRIV / 'runs' / 'B' / FIN / f'S{s}' / 'RELEASE.npz') as z:
        RN = {k: z[k].copy() for k in z.files}
    srec = {'release_json_mechanism': rel['mechanism'], 'phi': rel['phi'], 'mix_alpha': rel['mix_alpha'], 'coarse_m': rel['coarse_m'],
            'base_occupation': rel['base_occupation'], 'bank_sha256': bsha, 'roles': {}}
    d_by_role = {}
    for role, F in feats.items():
        X, names = F['X'], F['feature_names']
        lv = {t: te[t].leaves(X, names) for t in V.TASKS}
        rawL = {t: te[t].raw(lv[t], th['LOCAL_TREE'][t]) for t in V.TASKS}
        rawJ = {t: te[t].raw(lv[t], th['JOINT_TREE'][t]) for t in V.TASKS}
        d = 0.5 * rawL['income'].reshape(-1) + 0.5 * rawJ['income'].reshape(-1)
        d_by_role[role] = d
        c = (d[:, None] > thr[None, :]).sum(1)                       # bin = number of thresholds strictly below d
        Z = np.zeros((len(d), 6)); Z[np.arange(len(d)), c] = 1.0; Z = Z[:, 1:]
        sc = ((Z - mu) / sd) @ w + b
        p_inc = V.probs('income', sc)
        p_occ = V.probs('occupation', rawJ['occupation'])
        recomputed[(s, role)] = {'income': p_inc, 'occupation': p_occ, 'cells': np.bincount(c, minlength=6).tolist(), 'row_id': F['row_id']}
        rr = {}
        for t, p in (('income', p_inc), ('occupation', p_occ)):
            e = {'vs_release_npz_max_abs': float(np.max(np.abs(p - RN[f'{role}__{t}']))), 'vs_release_npz_bitwise': bool(np.array_equal(p, RN[f'{role}__{t}'])),
                 'decisions_equal_release_npz': bool(np.array_equal(p.argmax(1), RN[f'{role}__{t}'].argmax(1)))}
            if role in ('MODEL_SELECT', 'DEV_REPORT'):
                bk = B[f'FINALIST__{role}__{t}']
                if not np.array_equal(B[f'row_id__{role}'], F['row_id']):
                    raise AssertionError('bank row order differs')
                e.update({'vs_bank_max_abs': float(np.max(np.abs(p - bk))), 'vs_bank_bitwise': bool(np.array_equal(p, bk)),
                          'decisions_equal_bank': bool(np.array_equal(p.argmax(1), bk.argmax(1))), 'n': int(len(p))})
            rr[t] = e
        rr['coarse_cells_recomputed'] = recomputed[(s, role)]['cells']
        rr['coarse_cells_recorded'] = rel['coarse_cell_counts'].get(role)
        srec['roles'][role] = rr
    # thresholds from label-free MODEL_FIT log-odds ('aligned', m = 6)
    dfit = d_by_role['MODEL_FIT']; neg, pos = dfit[dfit <= 0], dfit[dfit > 0]
    thr_re = [float(q) for q in np.quantile(neg, [1 / 3, 2 / 3])] + [0.0] + [float(q) for q in np.quantile(pos, [1 / 3, 2 / 3])]
    srec['thresholds_recomputed_from_MODEL_FIT_features'] = thr_re
    srec['thresholds_recorded'] = rel['thresholds']
    srec['thresholds_max_abs_diff'] = float(np.max(np.abs(np.asarray(thr_re) - thr)))
    # h refit on CALIBRATION income labels only (no SEX)
    Cal = C.load_role('CALIBRATION', ('features', 'income'))
    if not np.array_equal(Cal['row_id'], feats['CALIBRATION']['row_id']):
        raise AssertionError('CALIBRATION order differs')
    dc = d_by_role['CALIBRATION']; cc = (dc[:, None] > thr[None, :]).sum(1)
    Zc = np.zeros((len(dc), 6)); Zc[np.arange(len(dc)), cc] = 1.0; Zc = Zc[:, 1:]
    mu2 = Zc.mean(0); sd2 = Zc.std(0); sd2[sd2 == 0] = 1.0
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', tol=1e-10).fit((Zc - mu2) / sd2, Cal['income'])
    srec['h_refit_on_CALIBRATION'] = {'mu_max_abs': float(np.max(np.abs(mu2 - mu))), 'sd_max_abs': float(np.max(np.abs(sd2 - sd))),
                                      'w_max_abs': float(np.max(np.abs(lr.coef_.reshape(-1) - w))), 'b_abs': float(abs(lr.intercept_[0] - b)),
                                      'w_refit': lr.coef_.reshape(-1).tolist(), 'b_refit': float(lr.intercept_[0])}
    # occupation output identical to H_LJ's and to B2TWIN's (IDENTICAL_RELEASE flag basis), DEV_REPORT and MODEL_SELECT
    srec['occupation_identical_to'] = {f'{svc}|{role}': bool(np.array_equal(B[f'FINALIST__{role}__occupation'], B[f'{svc}__{role}__occupation']))
                                       for svc in ('H_LJ', 'B2TWIN', 'H_JJ', 'TASKONLY', 'NEURAL_RAW') for role in ('MODEL_SELECT', 'DEV_REPORT')}
    srec['distinct_income_vectors_DEV'] = int(len(np.unique(B['FINALIST__DEV_REPORT__income'], axis=0)))
    chk2[f'S{s}'] = srec
tol = 1e-12
ok2 = all(r[t]['vs_bank_max_abs'] <= tol and r[t]['decisions_equal_bank']
          for s in chk2.values() for role, r in s['roles'].items() if role in ('MODEL_SELECT', 'DEV_REPORT') for t in V.TASKS)
ok2 = ok2 and all(s['thresholds_max_abs_diff'] == 0.0 for s in chk2.values())
out['check2'] = {'detail': chk2, 'pass': bool(ok2), 'tolerance': tol}
print('check2', ok2, flush=True)

# ---------------------------------------------------------------- check 3 (MODEL_SELECT part): 12 clauses + 2 guards of the finalist
MS = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'),
                 purpose='independent verifier: finalist MODEL_SELECT 12 clauses and 2 guards recomputation (AMD_VERIFY_NOLABEL_V1)')
chk3 = {}
ok3 = True
for s in (0, 1):
    B, _ = bank(s)
    fz, fids = C.frozen_bank(s, roles=('MODEL_SELECT',))
    if not (np.array_equal(B['row_id__MODEL_SELECT'], MS['row_id']) and np.array_equal(fids['row_id__MODEL_SELECT'], MS['row_id'])):
        raise AssertionError('MODEL_SELECT order differs')
    ev = json.loads((V.PRIV / 'runs' / 'B' / FIN / f'S{s}' / 'EVAL_MS.json').read_text())
    recd = {(c['task'], c['reference'], c['metric']): c['value'] for c in ev['clauses']}
    rows = []
    for t in V.TASKS:
        pf = B[f'FINALIST__MODEL_SELECT__{t}']
        for ref in ('U0', 'Ucal'):
            cv = V.clause_vectors(MS[t], pf, fz[ref]['MODEL_SELECT'][t])
            for m, v in cv.items():
                val = float(v.mean())
                rows.append({'task': t, 'ref': ref, 'metric': m, 'recomputed': val, 'recorded': recd[(t, ref, m)],
                             'abs_diff': abs(val - recd[(t, ref, m)]), 'point_pass': val <= V.CAPS[m], 'normalized': val / V.CAPS[m]})
    lf = V.loss_vectors(MS['occupation'], B['FINALIST__MODEL_SELECT__occupation'])['brier']
    l0 = V.loss_vectors(MS['occupation'], fz['U0']['MODEL_SELECT']['occupation'])['brier']
    guards = {}
    for g in (0, 1):
        m_ = MS['sex'] == g
        val = float((lf - l0)[m_].mean())
        guards[f'sex{g}'] = {'recomputed': val, 'recorded': ev['guard_occupation_brier_excess_vs_U0'][f'sex{g}'],
                             'abs_diff': abs(val - ev['guard_occupation_brier_excess_vs_U0'][f'sex{g}']), 'pass': val <= V.GUARD}
    maxdiff = max([r['abs_diff'] for r in rows] + [g['abs_diff'] for g in guards.values()])
    chk3[f'S{s}'] = {'clauses': rows, 'guards': guards, 'all_12_point_pass': all(r['point_pass'] for r in rows),
                     'guards_pass': all(g['pass'] for g in guards.values()), 'max_abs_diff_vs_EVAL_MS': maxdiff,
                     'max_normalized_12': max(r['normalized'] for r in rows), 'robust_0p8': max(r['normalized'] for r in rows) <= 0.8 and all(g['recomputed'] <= 0.8 * V.GUARD for g in guards.values())}
    ok3 = ok3 and maxdiff <= tol and chk3[f'S{s}']['all_12_point_pass'] and chk3[f'S{s}']['guards_pass']
out['check3_model_select'] = {'detail': chk3, 'pass': bool(ok3)}
print('check3 MS', ok3, flush=True)

# ---------------------------------------------------------------- check 4: PROTOCOL section 6 over every fitted configuration's EVAL_MS.json
base = {s: json.loads((V.PRIV / 'runs' / 'BASELINES' / f'BASELINES_MS_S{s}.json').read_text()) for s in (0, 1)}
refbar = {s: {ref: {v: base[s][ref][str(s)]['proxy'][v]['max'] for v in ('income', 'occupation', 'pair')} for ref in ('NEURAL_RAW', 'H_LJ')} for s in (0, 1)}
promo = json.loads((V.RES / 'PROMOTION.json').read_text())
refbar_match = all(abs(refbar[s][ref][v] - promo['reference_bar_model_select'][str(s)][ref][v]) == 0.0 for s in (0, 1) for ref in refbar[s] for v in refbar[s][ref])
configs = {}
for fam in ('A', 'B', 'C'):
    for d in sorted((V.PRIV / 'runs' / fam).iterdir()):
        if not d.is_dir():
            continue
        evs = {}
        for s in (0, 1):
            p = d / f'S{s}' / 'EVAL_MS.json'
            if p.is_file():
                evs[s] = json.loads(p.read_text())
        if evs:
            configs[(fam, d.name)] = evs
trial = {}
for (fam, cid), evs in configs.items():
    r = {'family': fam, 'seeds': sorted(evs)}
    if sorted(evs) != [0, 1]:
        r['status'] = 'ONE_SEED_ONLY'; trial[cid] = r; continue
    per = {}
    for s, ev in evs.items():
        ug = all(c['value'] <= c['cap'] for c in ev['clauses']) and len(ev['clauses']) == 12
        gg = all(v <= V.GUARD for v in ev['guard_occupation_brier_excess_vs_U0'].values())
        screen = all(ev['proxy'][v]['max'] <= refbar[s][ref][v] + 0.010 for ref in ('NEURAL_RAW', 'H_LJ') for v in ('income', 'occupation'))
        below = all(ev['proxy']['pair']['max'] < refbar[s][ref]['pair'] for ref in ('NEURAL_RAW', 'H_LJ'))
        rob = all(c['value'] <= 0.8 * c['cap'] for c in ev['clauses']) and all(v <= 0.8 * V.GUARD for v in ev['guard_occupation_brier_excess_vs_U0'].values())
        per[s] = {'UG': ug, 'G': gg, 'R_screen': screen, 'pair_below_both': below, 'robust': rob, 'pair_proxy': ev['proxy']['pair']['max']}
    elig = all(p['UG'] and p['G'] and p['R_screen'] for p in per.values())
    prom = elig and all(p['pair_below_both'] for p in per.values())
    r.update({'per_seed': per, 'eligible': elig, 'promotable': prom, 'robust_both': all(p['robust'] for p in per.values()),
              'mean_pair_proxy': float(np.mean([per[0]['pair_proxy'], per[1]['pair_proxy']]))})
    trial[cid] = r
promotable = [c for c, r in trial.items() if r.get('promotable')]
robust_prom = [c for c in promotable if trial[c]['robust_both']]
pool = robust_prom if robust_prom else promotable
choice = min(pool, key=lambda c: trial[c]['mean_pair_proxy']) if pool else None
label = ('ROBUST_SELECTED' if robust_prom else 'EDGE_SELECTED') if choice else None
# compare with PROMOTION.json
ptr = {t['config_id']: t for t in promo['trials']}
cmp_rows = []
for c, r in trial.items():
    p = ptr.get(c, {})
    cmp_rows.append({'config': c, 'mine_promotable': r.get('promotable'), 'recorded_promotable': p.get('promotable'),
                     'mine_mean_pair_proxy': r.get('mean_pair_proxy'), 'recorded_mean_pair_proxy': p.get('mean_pair_proxy'),
                     'mine_robust_both': r.get('robust_both'), 'recorded_robust_both': p.get('robust_both'),
                     'mine_eligible': r.get('eligible'), 'recorded_eligible_both': p.get('eligible_both'), 'recorded_screen_both': p.get('screen_both')})
mismatch = [x for x in cmp_rows if (x['mine_promotable'] or False) != (x['recorded_promotable'] or False)
            or (x['mine_mean_pair_proxy'] is not None and x['recorded_mean_pair_proxy'] is not None and abs(x['mine_mean_pair_proxy'] - x['recorded_mean_pair_proxy']) > 1e-15)]
ok4 = (choice == promo['finalist']['config_id'] and label == promo['finalist']['selection_label'] and len(promotable) == promo['n_promotable']
       and len(robust_prom) == promo['n_promotable_robust'] and not mismatch and refbar_match)
# ledger counts
led = list(csv.DictReader(open(V.RES / 'EXPERIMENT_LEDGER.csv')))
cnt36 = sum(1 for r in led if str(r.get('counts_toward_36', '')).strip().lower() in ('1', 'true', 'yes'))
out['check4'] = {'trials_recomputed': trial, 'promotable': promotable, 'robust_promotable': robust_prom, 'choice': choice, 'label': label,
                 'recorded_finalist': promo['finalist'], 'recorded_n_promotable': promo['n_promotable'], 'recorded_n_promotable_robust': promo['n_promotable_robust'],
                 'comparison': cmp_rows, 'mismatches': mismatch, 'reference_bar_matches_baselines': refbar_match, 'reference_bar': refbar,
                 'ledger_rows': len(led), 'ledger_counts_toward_36': cnt36, 'ledger_columns': list(led[0].keys()) if led else [],
                 'n_configs_with_eval_ms': len(configs), 'pass': bool(ok4)}
print('check4', ok4, choice, label, promotable, robust_prom, flush=True)

# ---------------------------------------------------------------- 3 promotable configs: MODEL_SELECT utility clause re-check from their RELEASE.npz
extra = {}
for (fam, cid) in [('B', c) for c in promotable]:
    for s in (0, 1):
        with np.load(V.PRIV / 'runs' / fam / cid / f'S{s}' / 'RELEASE.npz') as z:
            pr = {t: z[f'MODEL_SELECT__{t}'].copy() for t in V.TASKS}
        fz, _ = C.frozen_bank(s, roles=('MODEL_SELECT',))
        ev = json.loads((V.PRIV / 'runs' / fam / cid / f'S{s}' / 'EVAL_MS.json').read_text())
        recd = {(c['task'], c['reference'], c['metric']): c['value'] for c in ev['clauses']}
        md = 0.0
        for t in V.TASKS:
            for ref in ('U0', 'Ucal'):
                for m, v in V.clause_vectors(MS[t], pr[t], fz[ref]['MODEL_SELECT'][t]).items():
                    md = max(md, abs(float(v.mean()) - recd[(t, ref, m)]))
        lf = V.loss_vectors(MS['occupation'], pr['occupation'])['brier']; l0 = V.loss_vectors(MS['occupation'], fz['U0']['MODEL_SELECT']['occupation'])['brier']
        for g in (0, 1):
            md = max(md, abs(float((lf - l0)[MS['sex'] == g].mean()) - ev['guard_occupation_brier_excess_vs_U0'][f'sex{g}']))
        extra[f'{cid}|S{s}'] = md
out['check4_promotable_eval_ms_recheck_max_abs_diff'] = extra
V.dump('V_NOLABEL.json', out)
print(json.dumps({'check2': ok2, 'check3_ms': ok3, 'check4': ok4, 'extra': extra}))
