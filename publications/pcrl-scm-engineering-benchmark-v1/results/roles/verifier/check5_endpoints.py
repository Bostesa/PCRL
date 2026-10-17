"""Independent verifier, check 5: all 42 primary (and 10 secondary) endpoints, intervals, statuses and claim decisions.

Own code: household bootstrap (B = 2048, seed 2026101705, households resampled with replacement through
numpy default_rng(seed).integers(0, G, size=(B, G)), the registered draw), weighted tie-aware AUC (own implementation,
cross-checked against sklearn.roc_auc_score with sample weights on the point and on a subset of draws), task metrics,
Wald t(G-1) bounds, clause (one-sided alpha/3) and family (two-sided Bonferroni-t over K) critical values, status rules
and claim logic per PROTOCOL.md section 9. No scm_bench / task_headroom / task_conditioned_representation import for the
quantities verified; the only scm_bench call is the registered, logged truth re-read
scm_bench.common.dev_report_truth(..., event_name='DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER').

Modes: --dry  uses invented truth (random SEX/labels on the real DEV_REPORT households) to test the pipeline; nothing logged.
       --real performs the single logged re-read (once), keeps a private session copy outside the repository only as crash
              insurance (deleted after the run), and writes aggregates to check5_endpoints.json.
       --cached reuses that private session copy (no new re-read).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path('<repository>/.claude/worktrees/pcrl-scm-engineering-benchmark-v1')
PRIV = ROOT / '.private' / 'pcrl_scm_engineering_benchmark_v1'
RES = ROOT / 'results' / 'pcrl_scm_engineering_benchmark_v1'
OUT = RES / 'roles' / 'verifier'
CACHE = Path('<session scratch>')
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
K = {'income': 2, 'occupation': 5}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def boot_weights(unit, seed, B):
    hh, inv = np.unique(unit, return_inverse=True)
    G = len(hh)
    picks = np.random.default_rng(seed).integers(0, G, size=(B, G))
    cnt = np.zeros((B, G))
    np.add.at(cnt, (np.repeat(np.arange(B), G), picks.ravel()), 1.0)
    return cnt[:, inv], G


def wauc(y, s, W):
    """Weighted AUC per row of W: P(score_pos > score_neg) + 0.5 P(tie), weights on persons."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
    vals, grp = np.unique(s, return_inverse=True)     # tie groups, ascending
    ng = len(vals)
    out = np.empty(W.shape[0])
    for b0 in range(0, W.shape[0], 256):
        Wb = W[b0:b0 + 256]
        P = np.zeros((Wb.shape[0], ng)); N = np.zeros((Wb.shape[0], ng))
        for b in range(Wb.shape[0]):
            P[b] = np.bincount(grp[y], weights=Wb[b, y], minlength=ng)
            N[b] = np.bincount(grp[~y], weights=Wb[b, ~y], minlength=ng)
        below = np.cumsum(N, axis=1) - N
        out[b0:b0 + 256] = (P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1))
    return out


def metrics(y, p, k):
    p = np.asarray(p, dtype=np.float64)
    assert p.shape == (len(y), k) and np.allclose(p.sum(1), 1, atol=1e-9, rtol=0) and (p >= 0).all()
    onehot = np.eye(k)[y]
    return {'accuracy': (np.argmax(p, 1) == y).astype(float),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - onehot) ** 2).sum(1)}


def status_at(kind, L, U, m):
    if kind == 'benefit':
        return 'ESTABLISHED' if L > m else ('ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED')
    if kind == 'no_excess':
        return 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')
    return 'ABOVE_ZERO' if L > m else ('BELOW_ZERO' if U < m else 'UNRESOLVED')


def get_truth(mode, unit_dev, row_dev):
    if mode == 'dry':
        rng = np.random.default_rng(7)
        n = len(unit_dev)
        return {'row_id': row_dev, 'unit': unit_dev, 'sex': rng.integers(0, 2, n), 'income': rng.integers(0, 2, n),
                'occupation': rng.integers(0, 5, n)}
    if mode == 'cached':
        with np.load(CACHE) as z:
            return {k: z[k].copy() for k in z.files}
    sys.path.insert(0, str(ROOT))
    from scm_bench.common import dev_report_truth
    T = dev_report_truth('Independent verifier: recompute the 42 primary and 10 secondary ANALYSIS_LOCK endpoints, intervals, statuses and claims (check 5)',
                         event_name='DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER')
    T = {k: T[k] for k in ('row_id', 'unit', 'sex', 'income', 'occupation')}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, 'wb') as f:
        np.savez(f, **T)
    os.chmod(CACHE, 0o600)
    return T


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry', action='store_true'); g.add_argument('--real', action='store_true'); g.add_argument('--cached', action='store_true')
    a = ap.parse_args(); mode = 'dry' if a.dry else ('real' if a.real else 'cached')
    lock = json.loads((RES / 'ANALYSIS_LOCK.json').read_text())
    rlock = json.loads((RES / 'REPORTING_LOCK.json').read_text())
    sel = json.loads((RES / 'SELECTION_LOCK.json').read_text())
    rep = json.loads((RES / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    inf = lock['inference']
    hashes = {'analysis_lock': sha_file(RES / 'ANALYSIS_LOCK.json') == rlock['analysis_lock_sha256'] == rep['analysis_lock_sha256'],
              'selection_lock': sha_file(RES / 'SELECTION_LOCK.json') == rlock['selection_lock_sha256'] == rep['selection_lock_sha256'],
              'reporting_lock_in_report': sha_file(RES / 'REPORTING_LOCK.json') == rep['reporting_lock_sha256'],
              'service_bank_manifest': sha_file(RES / 'SERVICE_BANK_MANIFEST.json') == rlock['service_bank_manifest_sha256']}
    banks, scores, rseed = {}, {}, {}
    for s in (0, 1):
        bp = PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz'
        hashes[f'bank_S{s}'] = sha_file(bp) == rlock['bank_files_sha256'][f'S{s}']
        sp = PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz'
        hashes[f'scores_S{s}'] = sha_file(sp) == rlock['readers'][f'S{s}']['dev_report_scores_sha256']
        jp = PRIV / 'runs' / f'READERS_{s}' / 'READERS_SEED.json'
        hashes[f'readers_seed_S{s}'] = sha_file(jp) == rlock['readers'][f'S{s}']['readers_seed_sha256']
        with np.load(bp) as z:
            banks[s] = {k: z[k].copy() for k in z.files if 'DEV_REPORT' in k}
        with np.load(sp) as z:
            scores[s] = {k: z[k].copy() for k in z.files}
        rseed[s] = json.loads(jp.read_text())
    unit_dev, row_dev = banks[0]['unit__DEV_REPORT'], banks[0]['row_id__DEV_REPORT']
    T = get_truth(mode, unit_dev, row_dev)
    align = all(np.array_equal(T['row_id'], banks[s]['row_id__DEV_REPORT']) and np.array_equal(T['unit'], banks[s]['unit__DEV_REPORT'])
                and np.array_equal(T['row_id'], scores[s]['row_id']) for s in (0, 1))
    sex = T['sex'].astype(np.int64); y = {'income': T['income'].astype(np.int64), 'occupation': T['occupation'].astype(np.int64)}
    W, G = boot_weights(T['unit'], inf['bootstrap_seed'], inf['bootstrap_B'])
    c1 = float(stats.t.ppf(1 - inf['alpha_family_one_sided'] / inf['n_claims'], G - 1))
    cK = float(stats.t.ppf(1 - 0.05 / (2 * inf['K_primary']), G - 1))
    cK2 = float(stats.t.ppf(1 - 0.05 / (2 * inf['K_secondary']), G - 1))
    cm = float(stats.t.ppf(0.975, G - 1))
    out = {'mode': mode, 'hashes_ok': hashes, 'rows_aligned': bool(align), 'households': int(G), 'persons': int(len(sex)),
           'subgroup_households_per_sex': {str(g): int(len(np.unique(T['unit'][sex == g]))) for g in (0, 1)},
           'critical': {'c_clause_alpha_over_3': c1, 'c_family_K42': cK, 'c_family_K10': cK2, 't_marginal_95': cm}}
    # bootstrap implementation facts
    hh, inv = np.unique(T['unit'], return_inverse=True)
    out['bootstrap_checks'] = {'weights_constant_within_household': bool(all(np.all(W[:5, inv == i] == W[:5, [np.flatnonzero(inv == i)[0]]]) for i in range(0, G, 97))),
                               'row_sums_households_equal_G': bool(np.allclose(np.bincount(inv, minlength=G) @ np.zeros(G) + W[:, np.unique(inv, return_index=True)[1]].sum(1), G)),
                               'mean_household_weight': float(W[:, np.unique(inv, return_index=True)[1]].mean())}
    # recovery AUCs
    aucp, aucd = {}, {}
    sk_check = []
    for s in (0, 1):
        for svc, sv in rseed[s]['services'].items():
            for slate, wkey in (('', 'winner'), ('_slate7', 'winner_slate7')):
                for view in ('income', 'occupation', 'pair'):
                    rid = sv[wkey][view]['reader_id']; sc = scores[s][rid]
                    aucp[(svc, s, view + slate)] = float(roc_auc_score(sex, sc))
                    aucd[(svc, s, view + slate)] = wauc(sex, sc, W)
                    own_pt = float(wauc(sex, sc, np.ones((1, len(sex))))[0])
                    sk_check.append(abs(own_pt - aucp[(svc, s, view + slate)]))
                    if slate == '' and view == 'pair':
                        for b in range(0, 2048, 512):
                            sk_check.append(abs(aucd[(svc, s, view)][b] - roc_auc_score(sex, sc, sample_weight=W[b])))
    out['own_auc_vs_sklearn_max_abs'] = float(max(sk_check))
    # utility
    util = {}
    for s in (0, 1):
        b = banks[s]
        for svc in ('SCM', 'TWIN', 'IBSI', 'SHUF', 'NEURAL_RAW', 'H_LJ', 'C0'):
            for t in ('income', 'occupation'):
                v = metrics(y[t], b[f'{svc}__DEV_REPORT__{t}'], K[t])
                for ref in ('U0', 'Ucal'):
                    r = metrics(y[t], b[f'{ref}__DEV_REPORT__{t}'], K[t])
                    for met, d in (('accuracy_loss', r['accuracy'] - v['accuracy']), ('nll_excess', v['nll'] - r['nll']), ('brier_excess', v['brier'] - r['brier'])):
                        util[(svc, s, t, ref, met)] = (float(d.mean()), (W @ d) / W.sum(1))
            v = metrics(y['occupation'], b[f'{svc}__DEV_REPORT__occupation'], 5); r = metrics(y['occupation'], b['U0__DEV_REPORT__occupation'], 5)
            d = v['brier'] - r['brier']
            for g in (0, 1):
                m = sex == g
                util[(svc, s, 'subgroup', g)] = (float(d[m].mean()), (W[:, m] @ d[m]) / W[:, m].sum(1))

    def evaluate(e, s):
        if e['type'] in ('auc_diff', 'auc_diff_slate7'):
            tag = e['view'] + ('' if e['type'] == 'auc_diff' else '_slate7')
            return aucp[(e['plus'], s, tag)] - aucp[(e['minus'], s, tag)], aucd[(e['plus'], s, tag)] - aucd[(e['minus'], s, tag)]
        if e['type'] == 'utility':
            return util[(e['service'], s, e['task'], e['reference'], e['metric'])]
        return util[(e['service'], s, 'subgroup', e['group'])]

    cmp_fields = ('point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1')
    E = {}
    maxdiff = {f: 0.0 for f in cmp_fields}
    mism = []
    for roster, cfam, repkey in ((lock['primary_roster'], cK, 'endpoints_primary'), (lock['secondary_roster'], cK2, 'endpoints_secondary')):
        for sp in roster:
            pt, dr = evaluate(sp['estimand'], sp['seed'])
            se = float(np.std(dr, ddof=1)); kind = sp['kind']; m = sp['margin']
            L, U = pt - cfam * se, pt + cfam * se
            rec = {'point': pt, 'se': se, 'family_lower': L, 'family_upper': U, 'clause_L1': pt - c1 * se, 'clause_U1': pt + c1 * se,
                   'status': status_at(kind, L, U, m),
                   'clause_pass': bool(pt - c1 * se > m) if kind == 'benefit' else bool(pt + c1 * se <= m),
                   'point_pass': bool(pt > m) if kind == 'benefit' else bool(pt <= m),
                   'status_at_refs': {str(r): status_at(kind, L, U, r) for r in sp['refs']}}
            E[sp['key']] = rec
            if mode != 'dry':
                R = rep[repkey][sp['key']]
                for f in cmp_fields:
                    maxdiff[f] = max(maxdiff[f], abs(rec[f] - R[f]))
                for f in ('status', 'clause_pass', 'point_pass'):
                    if rec[f] != R[f]:
                        mism.append(f"{sp['key']}:{f} verifier={rec[f]} report={R[f]}")
                for r_, st in rec['status_at_refs'].items():
                    if R['at'][r_]['status_at_m'] != st:
                        mism.append(f"{sp['key']}:status_at_{r_} verifier={st} report={R['at'][r_]['status_at_m']}")
    out['endpoints_primary_n'] = len(lock['primary_roster']); out['endpoints_secondary_n'] = len(lock['secondary_roster'])
    out['endpoints'] = {k: {kk: (round(vv, 10) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in E.items()}
    out['max_abs_diff_vs_report'] = maxdiff; out['field_mismatches_vs_report'] = mism
    # recovery table and controls vs report
    rec_diff = 0.0; se_diff = 0.0; win_mism = []
    ctrl = {}
    for s in (0, 1):
        for svc, sv in rseed[s]['services'].items():
            for slate, key in (('', 'recovery'), ('_slate7', 'recovery_slate7')):
                for view in ('income', 'occupation', 'pair'):
                    R = rep['per_seed'][f'S{s}']['services'][svc][key][view]
                    rec_diff = max(rec_diff, abs(R['auc'] - aucp[(svc, s, view + slate)]))
                    se_diff = max(se_diff, abs(R['auc_se'] - float(np.std(aucd[(svc, s, view + slate)], ddof=1))))
                    wk = 'winner' if slate == '' else 'winner_slate7'
                    if R['winner'] != sv[wk][view]['candidate']:
                        win_mism.append(f'S{s} {svc} {view}{slate}')
            for t in ('income', 'occupation'):
                for ref in ('U0', 'Ucal'):
                    for met in CAPS:
                        R = rep['per_seed'][f'S{s}']['services'][svc]['utility'][f'{t}|{ref}|{met}']
                        rec_diff = max(rec_diff, abs(R['point'] - util[(svc, s, t, ref, met)][0]))
            feas = all(util[(svc, s, t, ref, met)][0] <= CAPS[met] for t in ('income', 'occupation') for ref in ('U0', 'Ucal') for met in CAPS)
            if feas != rep['per_seed'][f'S{s}']['services'][svc]['utility_point_feasible_24_this_seed']:
                win_mism.append(f'S{s} {svc} dev_point_feasible')
        for view in ('income', 'occupation', 'pair'):
            pl = np.where(sex == 1, scores[s][f'CONTROL__{view}__PLANTED_1'], scores[s][f'CONTROL__{view}__PLANTED_0'])
            ctrl[f'S{s}|{view}'] = {'shuffled_null_auc': float(roc_auc_score(sex, scores[s][f'CONTROL__{view}__SHUFFLED'])),
                                    'planted_auc': float(roc_auc_score(sex, pl)),
                                    'planted_av': rseed[s]['controls'][view]['planted_attack_validation_auc'],
                                    'shuffled_av': rseed[s]['controls'][view]['shuffled_attack_validation_auc']}
            R = rep['per_seed'][f'S{s}']['controls'][view]
            rec_diff = max(rec_diff, abs(R['shuffled_null_auc'] - ctrl[f'S{s}|{view}']['shuffled_null_auc']), abs(R['planted_bit_auc'] - ctrl[f'S{s}|{view}']['planted_auc']))
    lo, hi = lock['controls']['null_band']; pm = lock['controls']['planted_min_auc']
    planted = all(c['planted_auc'] >= pm and c['planted_av'] >= pm for c in ctrl.values())
    null = all(lo <= c['shuffled_null_auc'] <= hi and lo <= c['shuffled_av'] <= hi for c in ctrl.values())
    out['controls'] = {'per_view': ctrl, 'PLANTED_PASS': planted, 'NULL_PASS': null, 'CONTROLS_VALID': bool(planted and null and len(ctrl) == 6)}
    out['recovery_utility_table_max_abs_diff_vs_report'] = rec_diff; out['recovery_se_max_abs_diff_vs_report'] = se_diff
    out['winner_or_feasibility_mismatches'] = win_mism
    # claims per PROTOCOL section 9 (and the frozen report rule where the protocol text is silent)
    ck = lock['claim_keys']
    cp = lambda k: E[k]['clause_pass']
    gate24 = all(E[k]['point_pass'] for k in ck['utility_24'])
    elig = bool(sel['nominee']['model_select_point_feasible_24']); twin_mc = bool(sel.get('twin_matched_capacity'))
    valid = out['controls']['CONTROLS_VALID']
    recipe_clauses = all(cp(k) for k in ck['recipe_coalition'] + ck['recipe_worsening'])
    A = ('CONTROL_FAILURE' if not valid else 'NOMINEE_INELIGIBLE' if not elig else 'UNMATCHED_CAPACITY' if not twin_mc else
         'RECIPE_EFFECT_ESTABLISHED' if (recipe_clauses and gate24) else 'RECIPE_EFFECT_NOT_ESTABLISHED')
    ATT = 'NOT_TESTED_RECIPE_NOT_ESTABLISHED' if A != 'RECIPE_EFFECT_ESTABLISHED' else (
        'ATTRIBUTION_ESTABLISHED' if all(cp(k) for k in ck['attribution']) and sel['ibsi']['model_select_point_feasible_24'] else 'ATTRIBUTION_NOT_ESTABLISHED')
    comp_clauses = all(cp(k) for k in ck['competitive'])
    B_frozen_code = ('CONTROL_FAILURE' if not valid else 'NOMINEE_INELIGIBLE' if not elig else 'NOMINEE_FAILS_DEV_UTILITY_GATE' if not gate24 else
                     'SUPERIOR_TO_ALL_ELIGIBLE' if comp_clauses else 'NOT_SUPERIOR_TO_ALL_ELIGIBLE')
    B_protocol_text_only = 'SUPERIOR_TO_ALL_ELIGIBLE' if (valid and elig and comp_clauses) else 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'
    material = all(E[k]['clause_L1'] > 0.020 for k in ck['competitive'])
    sg = [E[k] for k in ck['subgroup']]
    C = 'SUBGROUP_QUALIFIED' if all(v['clause_pass'] for v in sg) else ('SUBGROUP_HARM' if any(v['family_lower'] > 0.005 for v in sg) else 'SUBGROUP_UNRESOLVED')
    ut = [E[k]['status'] for k in ck['utility_24']]
    claims = {'A_RECIPE_VS_TWIN': A, 'ATTRIBUTION_VS_PUB_IBSI_AND_SHUF': ATT, 'B_COMPETITIVE': B_frozen_code,
              'B_if_DEV_gate_not_applied': B_protocol_text_only, 'B_nested_material_0.020': 'MATERIAL_SUPERIORITY_ESTABLISHED' if (B_frozen_code == 'SUPERIOR_TO_ALL_ELIGIBLE' and material) else 'NOT_ESTABLISHED',
              'C_SUBGROUP': C, 'dev_gate24': gate24, 'dev_failed_keys': [k for k in ck['utility_24'] if not E[k]['point_pass']],
              'recipe_clauses_all_pass': recipe_clauses, 'competitive_clauses_all_pass_vs_0': comp_clauses,
              'competitive_clause_L1_gt_0.020_any': any(E[k]['clause_L1'] > 0.020 for k in ck['competitive']),
              'confidence_counts': {x: ut.count(x) for x in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')}}
    out['claims_recomputed'] = claims
    if mode != 'dry':
        rc = rep['claims']
        out['claims_match_report'] = {'A': rc['A_RECIPE_VS_TWIN']['decision'] == A, 'ATT': rc['ATTRIBUTION_VS_PUB_IBSI_AND_SHUF']['decision'] == ATT,
                                      'B': rc['B_COMPETITIVE']['decision'] == B_frozen_code,
                                      'B_nested': rc['B_COMPETITIVE']['nested_material_superiority_0.020'] == claims['B_nested_material_0.020'],
                                      'C': rc['C_SUBGROUP']['decision'] == C,
                                      'controls': rep['controls'] == {k: out['controls'][k] for k in ('PLANTED_PASS', 'NULL_PASS', 'CONTROLS_VALID')},
                                      'separate_point_feasibility': rep['separate_status_fields']['point_feasibility']['DEV_REPORT_failed_keys'] == claims['dev_failed_keys'],
                                      'confidence_counts': rep['separate_status_fields']['confidence_status']['counts'] == claims['confidence_counts'],
                                      'households': rep['households'] == G, 'persons': rep['persons'] == len(sex),
                                      'subgroup_households': {k: int(v) for k, v in rep['subgroup_households_per_sex'].items()} == out['subgroup_households_per_sex'],
                                      'critical_values_max_abs_diff': max(abs(rep['critical']['t_one_sided_alpha_over_3'] - c1), abs(rep['critical']['t_simultaneous_primary'] - cK),
                                                                          abs(rep['critical']['t_simultaneous_secondary'] - cK2), abs(rep['critical']['t_marginal_95'] - cm))}
    fn = OUT / ('check5_endpoints.json' if mode != 'dry' else 'check5_endpoints_DRYRUN_invented_truth.json')
    fn.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(json.dumps({k: out[k] for k in ('mode', 'hashes_ok', 'rows_aligned', 'households', 'own_auc_vs_sklearn_max_abs', 'max_abs_diff_vs_report',
                                           'field_mismatches_vs_report', 'recovery_utility_table_max_abs_diff_vs_report', 'recovery_se_max_abs_diff_vs_report',
                                           'winner_or_feasibility_mismatches', 'claims_recomputed', 'bootstrap_checks')}, indent=1, default=float))
    if mode != 'dry':
        print(json.dumps(out['claims_match_report'], default=float))


if __name__ == '__main__':
    main()
