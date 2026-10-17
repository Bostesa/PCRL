"""Verifier check 5: primary results, intervals and labels (own implementation of the FROZEN DESIGN; one logged replay).

Modes
  --mode dry   invented labels (no truth file is opened, no access event); output to the session scratchpad, deleted after review.
  --mode real  exactly one call of scm_um.common.dev_report_truth(<purpose>, event_name='DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER')
               (object under test: the logged access route). A crash copy of the truth arrays is written to the session scratchpad
               (mode 600) immediately after the read and deleted at the end of a successful run; --from-crash-copy reuses it
               without a second logged read.
Own code: household bootstrap (re-implemented; compared bitwise with task_headroom.inference.group_weights, imported only for that
comparison), weighted Mann-Whitney AUC replicates, utility / subgroup / NLL replicates, critical values (scipy t quantiles),
statuses, availabilities, IDENTICAL_RELEASE flags, claim conjunctions with failure categories, separate status fields, research
decision with qualifiers, registered bets, descriptive companions, and the label-free replay against the source's stored
DEV_REPORT figures. Inputs are the locked bank (re-derived bitwise from own tables and routing), the locked reader scores and the
stored winners (winner rule re-applied). Every number is compared with the coordinator's report/DEV_REPORT_RESULTS.json and CSVs.
Only aggregates are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
sys.path.insert(1, str(V.ROOT))

SCRATCH =Path('<session scratch>')
CRASH = SCRATCH / 'VERIFIER_DEV_TRUTH_CRASH_COPY.npz'
PURPOSE = 'SCM utility-matched follow-up: independent verifier replay of the registered DEV_REPORT endpoints, statuses and labels (check 5)'
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
SEV = ('AVAILABLE', 'DIAGNOSTIC_NOT_MATCHED', 'COMPARATOR_INELIGIBLE', 'MISSING')
SLATE8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')


# ---------------------------------------------------------------- own statistics
def own_group_weights(unit, seed, B):
    groups, inv = np.unique(unit, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(B, len(groups)))
    cnt = np.zeros((B, len(groups)))
    for b in range(B):
        np.add.at(cnt[b], draws[b], 1.0)
    return cnt[:, inv], len(groups)


def wauc(y, s, W):
    """Weighted Mann-Whitney AUC for every row of W (midrank ties)."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    o = np.argsort(s, kind='mergesort'); ss = s[o]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ss) != 0) + 1))
    Wo = W[:, o]; pos = (y[o] == 1)
    P = np.add.reduceat(Wo * pos[None, :], starts, axis=1); N = np.add.reduceat(Wo * (~pos)[None, :], starts, axis=1)
    below = np.cumsum(N, axis=1) - N
    return (P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1))


def wmean(d, W):
    return (W * d[None, :]).sum(1) / W.sum(1)


def tv(y, p, k):
    return V.task_vectors(y, p, k)


# ---------------------------------------------------------------- inputs (label-free)
def load_inputs():
    lock = V.jload(V.RES / 'ANALYSIS_LOCK.json'); inv = V.jload(V.RES / 'UNIT_INVENTORY.json'); sel = V.jload(V.RES / 'SELECTION_LOCK.json')
    rl = V.jload(V.RES / 'REPORTING_LOCK.json'); sbm = V.jload(V.RES / 'SERVICE_BANK_MANIFEST.json')
    fbm = V.jload(V.HYB_RES / 'FROZEN_BANK_MANIFEST.json')['bank_files_sha256']
    R = V.export('DEV_REPORT', 'features')
    chk = {'unit_inventory_in_analysis_lock': V.sha_file(V.RES / 'UNIT_INVENTORY.json') == lock['unit_inventory_sha256'],
           'selection_lock_in_reporting_lock': V.sha_file(V.RES / 'SELECTION_LOCK.json') == rl['selection_lock_sha256'],
           'bank_manifest_in_reporting_lock': V.sha_file(V.RES / 'SERVICE_BANK_MANIFEST.json') == rl['service_bank_manifest_sha256'],
           'analysis_lock_in_reporting_lock': V.sha_file(V.RES / 'ANALYSIS_LOCK.json') == rl['analysis_lock_sha256'],
           'trajectory_lock_in_reporting_lock': V.sha_file(V.RES / 'TRAJECTORY_COMPLETE_LOCK.json') == rl['trajectory_lock_sha256'],
           'report_py_frozen': V.sha_file(V.ROOT / 'scm_um' / 'report.py') == rl['code_sha256']['scm_um/report.py']}
    seeds = {}
    for s in V.SEEDS:
        bp = V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz'; rdir = V.PRIV / 'runs' / f'READERS_{s}'
        chk[f'S{s}_bank_hash'] = V.sha_file(bp) == sbm['bank_files_sha256'][f'S{s}'] == rl['bank_files_sha256'][f'S{s}']
        chk[f'S{s}_scores_hash'] = V.sha_file(rdir / 'DEV_REPORT_SCORES.npz') == rl['readers'][f'S{s}']['dev_report_scores_sha256']
        chk[f'S{s}_readers_seed_hash'] = V.sha_file(rdir / 'READERS_SEED.json') == rl['readers'][f'S{s}']['readers_seed_sha256']
        BK = V.npz(bp); SC = V.npz(rdir / 'DEV_REPORT_SCORES.npz'); RS = V.jload(rdir / 'READERS_SEED.json')
        assert np.array_equal(BK['row_id__DEV_REPORT'], R['row_id']) and np.array_equal(SC['row_id'], R['row_id'])
        bank = {}
        for k in BK:
            p = k.split('__')
            if len(p) == 3 and p[1] == 'DEV_REPORT':
                bank.setdefault(p[0], {})[p[2]] = BK[k]
        # independent re-derivation of every DEV_REPORT wire of the bank
        RTAB = V.npz(V.PRIV / 'runs' / f'TRAJ_{s}' / 'RELEASE_TABLES.npz'); RT = V.npz(V.SRC / 'routing' / f'ROUTING_S{s}.npz')
        FB = V.npz(V.SRC / 'bank' / f'FROZEN_BANK_S{s}.npz'); assert V.sha_file(V.SRC / 'bank' / f'FROZEN_BANK_S{s}.npz') == fbm[f'S{s}']
        init = RT['init__DEV_REPORT']; wires_ok = {}
        svc_of = sel['services'][str(s)]['service_of_role']
        for role, svc in svc_of.items():
            if svc != role:
                continue
            rid = sel['roles'][str(s)][role]['release']
            if str(rid).startswith('FROZEN_'):
                own = {t: FB[f'{role}__DEV_REPORT__{t}'] for t in V.TASKS}
            else:
                own = {'income': RTAB[f'{rid}__v1'][RTAB[f'{rid}__group_of_initial'][init]], 'occupation': RTAB[f'{rid}__v2'][RTAB[f'{rid}__cell_of_initial'][init]]}
            wires_ok[svc] = all(np.array_equal(own[t], bank[svc][t]) for t in V.TASKS)
        for ref in ('U0', 'Ucal'):
            wires_ok[ref] = all(np.array_equal(FB[f'{ref}__DEV_REPORT__{t}'], bank[ref][t]) for t in V.TASKS)
        chk[f'S{s}_bank_wires_rederived_bitwise'] = all(wires_ok.values())
        # winner rule re-applied from stored ATTACK_VALIDATION AUCs (pair: 8 + 16 ignore-one; recipient: 8), ties to the earlier candidate
        win_ok = True
        for svc, sr in RS['services'].items():
            for key, names in (('winner', SLATE8), ('winner_slate7', SLATE8[:7])):
                for view in ('income', 'occupation'):
                    mb = RS['marginal_banks'][sr[f'{view}_wire']]['readers']
                    c = [(n, mb[n]['attack_validation_auc']) for n in names]
                    i = max(range(len(c)), key=lambda j: (c[j][1], -j)); win_ok &= c[i][0] == sr[key][view]['candidate']
                c = [(n, sr['pair_readers'][n]['attack_validation_auc']) for n in names]
                c += [(f'IGNORE_occupation__{n}', RS['marginal_banks'][sr['income_wire']]['readers'][n]['attack_validation_auc']) for n in names]
                c += [(f'IGNORE_income__{n}', RS['marginal_banks'][sr['occupation_wire']]['readers'][n]['attack_validation_auc']) for n in names]
                i = max(range(len(c)), key=lambda j: (c[j][1], -j)); win_ok &= c[i][0] == sr[key]['pair']['candidate']
        chk[f'S{s}_winner_rule_reapplied'] = bool(win_ok)
        seeds[s] = {'bank': bank, 'scores': SC, 'readers': RS, 'svc_of': svc_of}
    return {'lock': lock, 'inv': inv, 'sel': sel, 'R': R, 'seeds': seeds, 'input_checks': chk}


# ---------------------------------------------------------------- analysis (own rules)
def analyse(T, I):
    lock, inv, sel = I['lock'], I['inv'], I['sel']
    infd = lock['inference']
    y = {t: T[t] for t in V.TASKS}; sex = T['sex']; unit = T['unit']
    W, G = own_group_weights(unit, infd['bootstrap_seed'], infd['bootstrap_B'])
    out = {'households': G, 'persons': int(len(sex)), 'halt_check_3955_5943': bool(G == 3955 and len(sex) == 5943)}
    try:
        from task_headroom.inference import group_weights          # object under test (bitwise comparison only)
        Wt, Gt = group_weights(unit, seed=infd['bootstrap_seed'], draws=infd['bootstrap_B'])
        out['bootstrap_weights_bitwise_equal_campaign_function'] = bool(np.array_equal(W, Wt) and G == Gt)
        del Wt
    except Exception as e:  # pragma: no cover
        out['bootstrap_weights_bitwise_equal_campaign_function'] = f'not compared: {e!r}'
    out['subgroup_households_per_sex'] = {g: int(len(np.unique(unit[sex == g]))) for g in (0, 1)}
    c1 = float(stats.t.ppf(1 - infd['alpha_family_one_sided'] / infd['n_claims'], G - 1))
    cP = float(stats.t.ppf(1 - 0.05 / (2 * infd['K_primary']), G - 1)); cS = float(stats.t.ppf(1 - 0.05 / (2 * infd['K_secondary']), G - 1))
    cM = float(stats.t.ppf(0.975, G - 1))
    out['critical'] = {'clause': c1, 'primary': cP, 'secondary': cS, 'marginal': cM}
    Wsum = W.sum(1)
    aucp, aucd, vec, util, svc_rec = {}, {}, {}, {}, {}
    controls = {}
    for s in V.SEEDS:
        S = I['seeds'][s]; bank, SC, RS = S['bank'], S['scores'], S['readers']
        u0 = {t: tv(y[t], bank['U0'][t], V.KCLS[t]) for t in V.TASKS}
        svc_rec[s] = {}
        for svc in list(RS['services']) + ['U0', 'Ucal']:
            rec = {'recovery': {}, 'recovery_slate7': {}, 'utility': {}, 'subgroup': {}, 'task_summary': {}, 'by_sex': {}}
            if svc in RS['services']:
                for key, wkey in (('recovery', 'winner'), ('recovery_slate7', 'winner_slate7')):
                    for view in ('income', 'occupation', 'pair'):
                        w = RS['services'][svc][wkey][view]; sc = SC[w['reader_id']]
                        tag = view if key == 'recovery' else view + '_slate7'
                        aucp[(svc, s, tag)] = V.auc_mw(sex, sc); aucd[(svc, s, tag)] = wauc(sex, sc, W)
                        rec[key][view] = {'winner': w['candidate'], 'auc': aucp[(svc, s, tag)], 'auc_se': float(np.std(aucd[(svc, s, tag)], ddof=1))}
            for t in V.TASKS:
                vv = tv(y[t], bank[svc][t], V.KCLS[t]); vec[(svc, s, t)] = vv
                rec['task_summary'][t] = {m: float(vv[m].mean()) for m in vv}
                rec['by_sex'][t] = {f'sex{g}': {m: float(vv[m][sex == g].mean()) for m in vv} for g in (0, 1)}
                for ref in ('U0', 'Ucal'):
                    rv = tv(y[t], bank[ref][t], V.KCLS[t])
                    for met, d in (('accuracy_loss', rv['accuracy'] - vv['accuracy']), ('nll_excess', vv['nll'] - rv['nll']), ('brier_excess', vv['brier'] - rv['brier'])):
                        util[(svc, s, t, ref, met)] = (float(d.mean()), wmean(d, W))
                        rec['utility'][f'{t}|{ref}|{met}'] = {'point': float(d.mean()), 'se': float(np.std(util[(svc, s, t, ref, met)][1], ddof=1)),
                                                             'point_pass': bool(d.mean() <= CAPS[met])}
            vo = vec[(svc, s, 'occupation')]
            for g in (0, 1):
                msk = sex == g; d = (vo['brier'] - u0['occupation']['brier'])[msk]
                util[(svc, s, 'subgroup', g)] = (float(d.mean()), wmean(d, W[:, msk]))
                rec['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d.mean()), 'se': float(np.std(util[(svc, s, 'subgroup', g)][1], ddof=1)),
                                              'occupation_nll_excess_vs_U0': float((vo['nll'] - u0['occupation']['nll'])[msk].mean())}
            rec['utility_point_feasible_12_this_seed'] = all(x['point_pass'] for x in rec['utility'].values())
            yo = y['occupation']; po = bank[svc]['occupation']
            rec['diagnostics'] = {'occupation_classwise_nll_contribution': [float(vo['nll'][yo == c].sum() / len(yo)) for c in range(5)],
                                  'occupation_true_class_frequency': [float((yo == c).mean()) for c in range(5)],
                                  'occupation_predicted_class_frequency': [float((np.argmax(po, 1) == c).mean()) for c in range(5)],
                                  'income_predicted_positive_rate': float((np.argmax(bank[svc]['income'], 1) == 1).mean())}
            svc_rec[s][svc] = rec
        for view in ('income', 'occupation', 'pair'):
            pl = np.where(sex == 1, SC[f'CONTROL__{view}__PLANTED_1'], SC[f'CONTROL__{view}__PLANTED_0'])
            controls[(s, view)] = {'null_dev': V.auc_mw(sex, SC[f'CONTROL__{view}__SHUFFLED']), 'planted_dev': V.auc_mw(sex, pl),
                                   'null_av': RS['controls'][view]['shuffled_attack_validation_auc'], 'planted_av': RS['controls'][view]['planted_attack_validation_auc'],
                                   'wire': RS['controls'][view]['wire']}
    lo, hi = inv['readers']['controls']['null_band']; pmin = inv['readers']['controls']['planted_min_auc']
    CV = len(controls) == 6 and all(c['planted_dev'] >= pmin and c['planted_av'] >= pmin and lo <= c['null_dev'] <= hi and lo <= c['null_av'] <= hi for c in controls.values())
    out['controls'] = {'CONTROLS_VALID': bool(CV), 'per_view': {f'S{s}|{v}': c for (s, v), c in controls.items()}}

    def avail(s, role):
        r = sel['roles'][str(s)].get(role, {}); st = r.get('status')
        if r.get('release') is None or st in ('MISSING', 'UNMATCHED_CAPACITY'):
            return 'MISSING'
        if st == 'NEAREST_DIAGNOSTIC_NOT_MATCHED':
            return 'DIAGNOSTIC_NOT_MATCHED'
        if st == 'SAME_COUNT_INELIGIBLE_FALLBACK' or (role in ('ORD_CNT', 'NEURAL_RAW', 'H_LJ') and r.get('point_pass_12') is False):
            return 'COMPARATOR_INELIGIBLE'
        return 'AVAILABLE'

    def value(spec):
        e = spec['estimand']; s = spec['seed']; so = I['seeds'][s]['svc_of']
        if e['type'] in ('auc_diff', 'auc_diff_slate7', 'nll_diff'):
            a = max(avail(s, e['plus']), avail(s, e['minus']), key=SEV.index)
            p, m = so[e['plus']], so[e['minus']]
            if e['type'] == 'nll_diff':
                d = vec[(p, s, e['task'])]['nll'] - vec[(m, s, e['task'])]['nll']
                return float(d.mean()), wmean(d, W), a, p == m
            tag = e['view'] if e['type'] == 'auc_diff' else e['view'] + '_slate7'
            return aucp[(p, s, tag)] - aucp[(m, s, tag)], aucd[(p, s, tag)] - aucd[(m, s, tag)], a, p == m
        if e['type'] == 'auc_view_diff':
            svc = so[e['service']]
            return aucp[(svc, s, e['plus_view'])] - aucp[(svc, s, e['minus_view'])], aucd[(svc, s, e['plus_view'])] - aucd[(svc, s, e['minus_view'])], 'AVAILABLE', False
        if e['type'] == 'utility':
            pt, dr = util[(so[e['service']], s, e['task'], e['reference'], e['metric'])]; return pt, dr, 'AVAILABLE', False
        if e['type'] == 'subgroup_occupation_brier_vs_U0':
            pt, dr = util[(so[e['service']], s, 'subgroup', e['group'])]; return pt, dr, 'AVAILABLE', False
        raise ValueError(e)

    def status(kind, L, U, m):
        if kind == 'benefit':
            return 'ESTABLISHED' if L > m else 'ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED'
        if kind == 'no_excess':
            return 'PASS' if U <= m else 'ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED'
        if kind == 'match':
            return 'EQUIVALENCE_ESTABLISHED' if (L >= -m and U <= m) else 'MISMATCH_ESTABLISHED' if (L > m or U < -m) else 'EQUIVALENCE_UNRESOLVED'
        return 'ABOVE_ZERO' if L > m else 'BELOW_ZERO' if U < m else 'UNRESOLVED'
    EP = {}
    for roster, cK, fam in ((lock['primary_roster'], cP, 'primary'), (lock['secondary_roster'], cS, 'secondary')):
        for sp in roster:
            pt, dr, a, ident = value(sp)
            se = float(np.std(dr, ddof=1)); kind, m = sp['kind'], sp['margin']
            L, U = pt - cK * se, pt + cK * se
            e = {'point': pt, 'se': se, 'family_lower': L, 'family_upper': U, 'availability': a, 'identical_release': bool(ident), 'kind': kind, 'family': fam,
                 'margin': m, 'claim': sp.get('claim'), 'clause_L1': None, 'clause_U1': None, 'clause_pass': None}
            if kind in ('benefit', 'no_excess') and a == 'AVAILABLE':
                e['clause_L1'] = pt - c1 * se; e['clause_U1'] = pt + c1 * se
                e['clause_pass'] = (e['clause_L1'] > m) if kind == 'benefit' else (e['clause_U1'] <= m)
            e['point_pass'] = (pt > m) if kind == 'benefit' else (pt <= m) if kind == 'no_excess' else (abs(pt) <= m) if kind == 'match' else None
            st = status(kind, L, U, m)
            e['status'] = st if a == 'AVAILABLE' else f'{a}:{st}'
            if kind == 'match':
                e['point_status'] = 'POINT_MATCH' if abs(pt) <= m else 'POINT_MISMATCH'
                e['direction'] = 'CARRIED_OVER' if abs(pt) <= 0.001 else ('SCM_SPENT_MORE' if pt > 0.001 else 'COMPARATOR_SPENT_MORE')
                d_ = sel['roles'][str(sp['seed'])][sp['estimand']['minus']].get('delta')
                e['MODEL_SELECT_achieved_mismatch_SCM_minus_comparator'] = None if d_ is None else -d_
            if kind != 'match':
                e['at'] = {str(r): status(kind, L, U, r) for r in sp['refs']}
            EP[sp['key']] = e
    out['endpoints'] = EP
    out['n_primary'] = len(lock['primary_roster']); out['n_secondary'] = len(lock['secondary_roster'])
    ck = lock['claim_keys']

    def ben(k, m=0.0):
        e = EP[k]
        if e['availability'] != 'AVAILABLE':
            return False, 'UNAVAILABLE'
        L1 = e['point'] - c1 * e['se']
        if L1 > m:
            return True, None
        return False, 'ESTABLISHED_SHORTFALL' if e['family_upper'] < m else 'POINT_SHORTFALL' if e['point'] <= m else 'UNRESOLVED'

    def nw(k, m=0.010):
        e = EP[k]
        if e['availability'] != 'AVAILABLE':
            return False, 'UNAVAILABLE'
        U1 = e['point'] + c1 * e['se']
        if U1 <= m:
            return True, None
        return False, 'ESTABLISHED_VIOLATION' if e['family_lower'] > m else 'POINT_EXCESS' if e['point'] > m else 'UNRESOLVED'

    def pm(k):
        e = EP[k]
        if e['availability'] != 'AVAILABLE':
            return False, 'UNAVAILABLE'
        return (True, None) if abs(e['point']) <= 0.001 else (False, 'POINT_FAILURE')
    NOM = sel['status'] == 'NOMINEE_SELECTED'
    NC0 = all(sel['roles'][str(s)]['SCM']['release'] != sel['roles'][str(s)]['C0']['release'] for s in V.SEEDS)
    UG = all(EP[k]['point'] <= EP[k]['margin'] for k in ck['utility_24']); SG = all(EP[k]['point'] <= 0.005 for k in ck['subgroup'])
    head = [('CONTROLS_VALID', (CV, None if CV else 'CONTROL_FAILURE'), 'CONTROL_FAILURE'),
            ('NOMINEE_SELECTED', (NOM, None if NOM else 'UNAVAILABLE'), 'NOMINEE_INELIGIBLE'),
            ('NOMINEE_NOT_C0', (NC0, None if NC0 else 'UNAVAILABLE'), 'NOMINEE_IS_C0')]
    tail = [('DEV_UTILITY_POINT_GATE_24', (UG, None if UG else 'POINT_FAILURE'), 'NOMINEE_FAILS_DEV_UTILITY_GATE'),
            ('DEV_SUBGROUP_POINT_GUARDS_4', (SG, None if SG else 'POINT_FAILURE'), 'NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD')]

    def decide(conds, ok_label, fail_label):
        for _, (v, _c), lab in conds:
            if not v and lab is not None:
                return lab
        return ok_label if all(v for _, (v, _c), _ in conds) else fail_label
    ms_match = all(avail(s, 'ORD_NLLM') == 'AVAILABLE' for s in V.SEEDS)
    cM = head + [('ORD_NLLM_MODEL_SELECT_match_both_seeds', (ms_match, None if ms_match else 'UNAVAILABLE'), 'MATCH_UNAVAILABLE')]
    cM += [(f'DEV_point_match|{k}', pm(k), 'MATCH_DID_NOT_CARRY_OVER') for k in ck['match_nll']] + tail
    cM += [(f'benefit_L1_gt_0|{k}', ben(k), None) for k in ck['match_benefit']] + [(f'worsening_U1_le_0.010|{k}', nw(k), None) for k in ck['match_worsening']]
    Mdec = decide(cM, 'MATCHED_ADVANTAGE_ESTABLISHED', 'MATCHED_ADVANTAGE_NOT_ESTABLISHED')
    Mmat = 'MATERIAL_ADVANTAGE_ESTABLISHED' if Mdec == 'MATCHED_ADVANTAGE_ESTABLISHED' and all(ben(k, 0.020)[0] for k in ck['match_benefit']) else 'NOT_ESTABLISHED'
    comp_only = all(ben(k)[0] for k in ck['match_benefit']) and all(nw(k)[0] for k in ck['match_worsening'])

    def component(keys, role):
        if not all(avail(s, role) == 'AVAILABLE' for s in V.SEEDS):
            return 'UNAVAILABLE'
        if not all(pm(k)[0] for k in keys if k.startswith('SEC|')):
            return 'MATCH_DID_NOT_CARRY_OVER'
        return 'PASS' if all(ben(k)[0] for k in keys if k.startswith('MATCH_COMPONENT|')) else 'FAIL'
    if Mdec != 'MATCHED_ADVANTAGE_ESTABLISHED':
        mc = {'IBSI': 'NOT_TESTED_M_NOT_ESTABLISHED', 'SHUF': 'NOT_TESTED_M_NOT_ESTABLISHED'}
    else:
        ib = component(ck['component_ibsi'], 'IBSI_NLLM')
        mc = {'IBSI': ib, 'SHUF': component(ck['component_shuf'], 'SHUF_NLLM') if ib == 'PASS' else 'NOT_TESTED_SEQUENCE_STOPPED'}
    mc_desc = {'IBSI': component(ck['component_ibsi'], 'IBSI_NLLM'), 'SHUF': component(ck['component_shuf'], 'SHUF_NLLM')}
    cnt_reached = all(avail(s, 'ORD_CNT') != 'MISSING' for s in V.SEEDS); cnt_elig = all(avail(s, 'ORD_CNT') == 'AVAILABLE' for s in V.SEEDS)
    cK = head + [('ORD_CNT_reached_K_star_both_seeds', (cnt_reached, None if cnt_reached else 'UNAVAILABLE'), 'UNMATCHED_CAPACITY'),
                 ('ORD_CNT_MODEL_SELECT_point_eligible_both_seeds', (cnt_elig, None if cnt_elig else 'UNAVAILABLE'), 'COMPARATOR_INELIGIBLE')] + tail
    cK += [(f'benefit_L1_gt_0|{k}', ben(k), None) for k in ck['capacity_benefit']] + [(f'worsening_U1_le_0.010|{k}', nw(k), None) for k in ck['capacity_worsening']]
    Kdec = decide(cK, 'SAME_CAPACITY_ADVANTAGE_ESTABLISHED', 'SAME_CAPACITY_ADVANTAGE_NOT_ESTABLISHED')
    Kmat = 'MATERIAL_ADVANTAGE_ESTABLISHED' if Kdec == 'SAME_CAPACITY_ADVANTAGE_ESTABLISHED' and all(ben(k, 0.020)[0] for k in ck['capacity_benefit']) else 'NOT_ESTABLISHED'
    if Kdec != 'SAME_CAPACITY_ADVANTAGE_ESTABLISHED':
        kib = 'NOT_TESTED_K_NOT_ESTABLISHED'
    elif not all(avail(s, 'IBSI_CNT') == 'AVAILABLE' for s in V.SEEDS):
        kib = 'COMPARATOR_INELIGIBLE'
    else:
        kib = 'PASS' if all(ben(k)[0] for k in ck['capacity_ibsi']) else 'FAIL'
    cB = head + tail + [(f'benefit_L1_gt_0|{k}', ben(k), None) for k in ck['competitive_benefit']] + [(f'worsening_U1_le_0.010|{k}', nw(k), None) for k in ck['competitive_worsening']]
    Bdec = decide(cB, 'SUPERIOR_TO_ALL_ELIGIBLE', 'NOT_SUPERIOR_TO_ALL_ELIGIBLE')
    Bmat = 'MATERIAL_SUPERIORITY_ESTABLISHED' if Bdec == 'SUPERIOR_TO_ALL_ELIGIBLE' and all(ben(k, 0.020)[0] for k in ck['competitive_benefit']) else 'NOT_ESTABLISHED'
    Cdec = ('SUBGROUP_QUALIFIED' if all(nw(k, 0.005)[0] for k in ck['subgroup']) else
            'SUBGROUP_HARM' if any(EP[k]['family_lower'] > 0.005 for k in ck['subgroup']) else 'SUBGROUP_UNRESOLVED')
    ut = [EP[k]['status'] for k in ck['utility_24']]
    conf = 'CONFIDENCE_FEASIBLE_24' if all(x == 'PASS' for x in ut) else 'ESTABLISHED_UTILITY_VIOLATION' if any(x == 'ESTABLISHED_VIOLATION' for x in ut) else 'CONFIDENCE_UNRESOLVED'
    rd = ('ENGINEERING_REFERENCE_FROZEN_AT_ORIGINAL_TARGET' if Mmat == 'MATERIAL_ADVANTAGE_ESTABLISHED' else
          'ENGINEERING_REFERENCE_FROZEN_POSITIVE_BELOW_ORIGINAL_TARGET' if Mdec == 'MATCHED_ADVANTAGE_ESTABLISHED' else
          'FRONTIER_DELIVERED_MATCH_UNAVAILABLE' if Mdec == 'MATCH_UNAVAILABLE' else
          'COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE' if Mdec == 'MATCH_DID_NOT_CARRY_OVER' else
          'COMPARISON_CLOSED_CONTROL_FAILURE' if Mdec == 'CONTROL_FAILURE' else f'COMPARISON_CLOSED_{Mdec}')
    ms_scm = {s: sel['per_seed_measurements'][str(s)][sel['roles'][str(s)]['SCM']['release']] for s in V.SEEDS}
    gstar = [g for g in sel['nomination']['grid'] if g['t'] == sel['nomination']['t_star']][0]
    p2 = all(sel['roles'][str(s)]['ORD_NLLM'].get('status') == 'MATCHED' for s in V.SEEDS)
    bets = {'P1': ('HELD' if sel['nomination']['t_star'] != 0.010 else 'NOT_HELD', 0.95), 'P2': ('HELD' if p2 else 'NOT_HELD', 0.70),
            'P3': (('HELD' if all(sel['roles'][str(s)]['ORD_NLLM']['proxy'] - sel['roles'][str(s)]['SCM']['proxy'] >= 0.020 for s in V.SEEDS) else 'NOT_HELD') if p2 else 'VOID', 0.70),
            'P4': ('HELD' if p2 and all(pm(k)[0] for k in ck['match_nll']) else 'NOT_HELD', 0.35),
            'P5': ('HELD' if p2 and all(EP[k]['point'] >= 0.020 for k in ck['match_benefit']) else 'NOT_HELD', 0.55),
            'P6': ('HELD' if UG else 'NOT_HELD', 0.75), 'P7': ('HELD' if SG else 'NOT_HELD', 0.50),
            'P8': ('HELD' if Mdec == 'MATCHED_ADVANTAGE_ESTABLISHED' else 'NOT_HELD', inv['predictions']['P8'])}
    scored = [(p, 1.0 if o == 'HELD' else 0.0) for o, p in bets.values() if o != 'VOID']
    out['claims'] = {
        'M': {'decision': Mdec, 'material': Mmat, 'comparison_only_pass': comp_only, 'conditions': {n: {'value': v, 'category': c} for n, (v, c), _ in cM},
              'directions': {k: EP[k]['direction'] for k in ck['match_nll']}},
        'MC': {**mc, 'descriptive': mc_desc}, 'K': {'decision': Kdec, 'material': Kmat, 'IBSI_CNT': kib, 'conditions': {n: {'value': v, 'category': c} for n, (v, c), _ in cK}},
        'B': {'decision': Bdec, 'material': Bmat, 'conditions': {n: {'value': v, 'category': c} for n, (v, c), _ in cB}},
        'C': {'decision': Cdec, 'point_status': 'SUBGROUP_POINT_GUARDS_PASS' if SG else 'SUBGROUP_POINT_GUARD_FAILED',
              'guards_clause_pass': {k: nw(k, 0.005)[0] for k in ck['subgroup']}, 'guards_family_status': {k: EP[k]['status'] for k in ck['subgroup']}}}
    out['separate'] = {'MODEL_SELECT_original_24': 'ORIGINAL_POINT_FEASIBLE_24' if all(ms_scm[s]['point_pass_12'] for s in V.SEEDS) else 'ORIGINAL_POINT_INFEASIBLE',
                       'nomination': 'NOMINATION_ELIGIBLE' if NOM else 'INELIGIBLE_FALLBACK', 'nomination_failed_conditions': gstar['failed_conditions'],
                       'DEV_point': 'POINT_FEASIBLE_24' if UG else 'POINT_INFEASIBLE', 'DEV_point_failed_keys': [k for k in ck['utility_24'] if EP[k]['point'] > EP[k]['margin']],
                       'DEV_confidence': conf, 'DEV_confidence_counts': {x: ut.count(x) for x in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
                       'subgroup_point': 'SUBGROUP_POINT_GUARDS_PASS' if SG else 'SUBGROUP_POINT_GUARD_FAILED'}
    out['research_decision'] = rd
    out['qualifiers'] = {'DEV_utility_confidence': conf, 'claim_C': Cdec, 'claim_B': Bdec,
                         'ORD_NLLM_DEV_match': {k: {'status': EP[k]['status'], 'point_status': EP[k]['point_status'], 'direction': EP[k]['direction']} for k in ck['match_nll']}}
    out['bets'] = {'outcomes': {k: {'outcome': o, 'probability': p} for k, (o, p) in bets.items()},
                   'mean_brier_non_void': float(np.mean([(p - h) ** 2 for p, h in scored])) if scored else None}
    # descriptive companions
    diffs = {}
    for s in V.SEEDS:
        so = I['seeds'][s]['svc_of']; scm = so['SCM']; RSv = I['seeds'][s]['readers']['services']
        for role in ('ORD_NLLM', 'ORD_CNT', 'IBSI_NLLM', 'IBSI_CNT', 'SHUF_NLLM', 'NEURAL_RAW', 'H_LJ', 'C0', 'HIST_SCM'):
            o = so[role]; row = {}
            for t in V.TASKS:
                for m in ('nll', 'brier', 'accuracy'):
                    d = vec[(scm, s, t)][m] - vec[(o, s, t)][m]; dr = wmean(d, W)
                    row[f'SCM_minus_role|{t}_{m}'] = (float(d.mean()), float(np.std(dr, ddof=1)))
            for g in (0, 1):
                msk = sex == g
                for m in ('brier', 'nll'):
                    d = (vec[(scm, s, 'occupation')][m] - vec[(o, s, 'occupation')][m])[msk]
                    row[f'SCM_minus_role|occupation_{m}|sex{g}'] = (float(d.mean()), float(np.std(wmean(d, W[:, msk]), ddof=1)))
            if o in RSv:
                for view in ('income', 'occupation', 'pair'):
                    row[f'SCM_minus_role|auc_{view}'] = (aucp[(scm, s, view)] - aucp[(o, s, view)], float(np.std(aucd[(scm, s, view)] - aucd[(o, s, view)], ddof=1)))
            row['income_qualifier'] = 'INCOME_EXPENDITURE_NOT_MATCHED' if row['SCM_minus_role|income_nll'][0] > 0.001 else 'income NLL within 0.001 or lower for SCM'
            diffs[f'{role}|s{s}'] = row
    out['differences'] = diffs
    out['services'] = svc_rec
    return out


# ---------------------------------------------------------------- comparison with the coordinator's report
def compare(mine, rep):
    C = {'endpoints': {}, 'mismatches': []}
    allrep = {**rep['endpoints_primary'], **rep['endpoints_secondary']}
    if set(allrep) != set(mine['endpoints']):
        C['mismatches'].append({'what': 'endpoint key sets differ', 'only_mine': sorted(set(mine['endpoints']) - set(allrep)), 'only_report': sorted(set(allrep) - set(mine['endpoints']))})
    maxd = {'point': 0.0, 'se': 0.0, 'family_lower': 0.0, 'family_upper': 0.0, 'clause_L1': 0.0, 'clause_U1': 0.0}
    exact = {k: 0 for k in maxd}
    for k, e in mine['endpoints'].items():
        r = allrep.get(k)
        if r is None:
            continue
        for f in maxd:
            if e.get(f) is None or r.get(f) is None:
                if (e.get(f) is None) != (r.get(f) is None):
                    C['mismatches'].append({'key': k, 'field': f, 'mine': e.get(f), 'report': r.get(f)})
                continue
            d = abs(e[f] - r[f]); maxd[f] = max(maxd[f], d); exact[f] += int(e[f] == r[f])
        for f in ('availability', 'identical_release', 'status', 'clause_pass', 'point_pass', 'kind', 'margin'):
            if e.get(f) != r.get(f):
                C['mismatches'].append({'key': k, 'field': f, 'mine': e.get(f), 'report': r.get(f)})
        if e['kind'] == 'match':
            for f in ('point_status', 'MODEL_SELECT_achieved_mismatch_SCM_minus_comparator'):
                if e.get(f) != r.get(f):
                    C['mismatches'].append({'key': k, 'field': f, 'mine': e.get(f), 'report': r.get(f)})
        else:
            for ref, st in e['at'].items():
                if r['at'].get(ref, {}).get('status_at_m') != st:
                    C['mismatches'].append({'key': k, 'field': f'at[{ref}]', 'mine': st, 'report': r['at'].get(ref)})
    C['endpoints'] = {'max_abs_diff': maxd, 'exact_counts': exact, 'n': len(mine['endpoints'])}
    cr = rep['critical']
    C['critical'] = {'clause': mine['critical']['clause'] - cr['t_one_sided_alpha_over_n_claims'], 'primary': mine['critical']['primary'] - cr['t_simultaneous_primary'],
                     'secondary': mine['critical']['secondary'] - cr['t_simultaneous_secondary'], 'marginal': mine['critical']['marginal'] - cr['t_marginal_95'],
                     'registered_values_match_7dp': [round(mine['critical']['clause'], 7) == 2.4988491, round(mine['critical']['primary'], 7) == 3.3346075,
                                                     round(mine['critical']['secondary'], 7) == 3.0801279, round(mine['critical']['marginal'], 7) == 1.9605641]}
    C['households'] = (mine['households'], rep['households'], mine['persons'], rep['persons'])
    C['subgroup_households'] = ({str(k): v for k, v in mine['subgroup_households_per_sex'].items()} == {str(k): v for k, v in rep['subgroup_households_per_sex'].items()})
    rc = rep['controls']
    C['controls'] = {'CV_equal': mine['controls']['CONTROLS_VALID'] == rc['CONTROLS_VALID'], 'max_abs_diff': max(
        max(abs(c['null_dev'] - rep['per_seed'][f'S{sv.split("|")[0][1]}']['controls'][sv.split('|')[1]]['shuffled_null_auc']),
            abs(c['planted_dev'] - rep['per_seed'][f'S{sv.split("|")[0][1]}']['controls'][sv.split('|')[1]]['planted_bit_auc'])) for sv, c in mine['controls']['per_view'].items())}
    cl = rep['claims']; mc = mine['claims']
    C['claims'] = {'M_decision': (mc['M']['decision'], cl['M_MATCHED_ORDINARY']['decision']), 'M_material': (mc['M']['material'], cl['M_MATCHED_ORDINARY']['material_0.020']['status']),
                   'M_directions': (mc['M']['directions'], cl['M_MATCHED_ORDINARY']['carry_over_direction']),
                   'MC': ((mc['MC']['IBSI'], mc['MC']['SHUF']), (cl['M_COMPONENTS_IBSI_SHUF']['IBSI'], cl['M_COMPONENTS_IBSI_SHUF']['SHUF'])),
                   'MC_descriptive': (mc['MC']['descriptive'], cl['M_COMPONENTS_IBSI_SHUF']['descriptive_status']),
                   'K_decision': (mc['K']['decision'], cl['K_SAME_CAPACITY']['decision']), 'K_material': (mc['K']['material'], cl['K_SAME_CAPACITY']['material_0.020']['status']),
                   'K_IBSI_CNT': (mc['K']['IBSI_CNT'], cl['K_SAME_CAPACITY']['IBSI_CNT']),
                   'B_decision': (mc['B']['decision'], cl['B_COMPETITIVE']['decision']), 'B_material': (mc['B']['material'], cl['B_COMPETITIVE']['material_0.020']['status']),
                   'C_decision': (mc['C']['decision'], cl['C_SUBGROUP']['decision']), 'C_point': (mc['C']['point_status'], cl['C_SUBGROUP']['point_status']),
                   'C_guards_clause': (mc['C']['guards_clause_pass'], cl['C_SUBGROUP']['guards_clause_pass']),
                   'C_guards_family': (mc['C']['guards_family_status'], cl['C_SUBGROUP']['guards_family_status'])}
    C['claims_equal'] = {k: (a == b) for k, (a, b) in C['claims'].items()}
    cond = {}
    for mk, rk in (('M', 'M_MATCHED_ORDINARY'), ('K', 'K_SAME_CAPACITY'), ('B', 'B_COMPETITIVE')):
        rconds = cl[rk]['conditions']
        for n, c in mc[mk]['conditions'].items():
            r = rconds.get(n)
            ok = r is not None and bool(r['value']) == bool(c['value']) and (c['value'] or r.get('category') == c['category'])
            if not ok:
                cond[f'{mk}|{n}'] = {'mine': c, 'report': r}
        if set(rconds) != set(mc[mk]['conditions']):
            cond[f'{mk}|names'] = sorted(set(rconds) ^ set(mc[mk]['conditions']))
    C['condition_mismatches'] = cond
    ss = rep['separate_status_fields']; ms = mine['separate']
    C['separate'] = {'MODEL_SELECT_original_24': (ms['MODEL_SELECT_original_24'], ss['MODEL_SELECT_original_point_eligibility_SCM']['status']),
                     'nomination': (ms['nomination'], ss['MODEL_SELECT_conservative_nomination']['status']),
                     'DEV_point': (ms['DEV_point'], ss['DEV_REPORT_point']['status']), 'DEV_point_failed_keys': (ms['DEV_point_failed_keys'], ss['DEV_REPORT_point']['failed_keys']),
                     'DEV_confidence': (ms['DEV_confidence'], ss['DEV_REPORT_confidence']['status']), 'DEV_confidence_counts': (ms['DEV_confidence_counts'], ss['DEV_REPORT_confidence']['counts']),
                     'subgroup_point': (ms['subgroup_point'], ss['subgroup_point'])}
    C['separate_equal'] = {k: a == b for k, (a, b) in C['separate'].items()}
    C['research_decision'] = (mine['research_decision'], rep['research_decision'])
    rq = rep['research_decision_qualifiers']; mq = mine['qualifiers']
    C['qualifiers_equal'] = {'DEV_utility_confidence': mq['DEV_utility_confidence'] == rq['DEV_utility_confidence'], 'claim_C': mq['claim_C'] == rq['claim_C'],
                             'claim_B': mq['claim_B'] == rq['claim_B'], 'ORD_NLLM_DEV_match': mq['ORD_NLLM_DEV_match'] == rq['ORD_NLLM_DEV_match']}
    C['bets'] = {'equal': {k: v['outcome'] == rep['registered_bets']['outcomes'][k]['outcome'] for k, v in mine['bets']['outcomes'].items()},
                 'brier': (mine['bets']['mean_brier_non_void'], rep['registered_bets']['mean_brier_non_void'])}
    # per-service blocks
    sv_max = 0.0; sv_mism = []
    for s in V.SEEDS:
        for svc, r in mine['services'][s].items():
            rr = rep['per_seed'][f'S{s}']['services'][svc]
            for key in ('recovery', 'recovery_slate7'):
                for view, x in r[key].items():
                    y_ = rr[key][view]
                    if x['winner'] != y_['winner']:
                        sv_mism.append((s, svc, key, view, 'winner'))
                    sv_max = max(sv_max, abs(x['auc'] - y_['auc']), abs(x['auc_se'] - y_['auc_se']))
            for k, x in r['utility'].items():
                y_ = rr['utility'][k]; sv_max = max(sv_max, abs(x['point'] - y_['point']), abs(x['se'] - y_['se']))
                if x['point_pass'] != y_['point_pass']:
                    sv_mism.append((s, svc, 'utility', k))
            for g in ('sex0', 'sex1'):
                for f in ('occupation_brier_excess_vs_U0', 'se', 'occupation_nll_excess_vs_U0'):
                    sv_max = max(sv_max, abs(r['subgroup'][g][f] - rr['subgroup'][g][f]))
            for t in V.TASKS:
                for m in ('accuracy', 'nll', 'brier'):
                    sv_max = max(sv_max, abs(r['task_summary'][t][m] - rr['task_summary'][t][m]))
            if r['utility_point_feasible_12_this_seed'] != rr['utility_point_feasible_12_this_seed']:
                sv_mism.append((s, svc, 'feasible12'))
            dg = rr['diagnostics']
            for f in ('occupation_classwise_nll_contribution', 'occupation_true_class_frequency', 'occupation_predicted_class_frequency'):
                sv_max = max(sv_max, max(abs(a - b) for a, b in zip(r['diagnostics'][f], dg[f])))
            sv_max = max(sv_max, abs(r['diagnostics']['income_predicted_positive_rate'] - dg['income_predicted_positive_rate']))
    C['services'] = {'max_abs_diff': sv_max, 'mismatches': sv_mism}
    dmax = 0.0; dm = []
    for k, row in mine['differences'].items():
        rr = rep['comparison_differences'][k]
        for f, (pt, se) in ((f, v) for f, v in row.items() if f != 'income_qualifier'):
            dmax = max(dmax, abs(pt - rr[f]['point']), abs(se - rr[f]['se']))
        if row['income_qualifier'] != rr['income_expenditure_qualifier']:
            dm.append(k)
    C['differences'] = {'max_abs_diff': dmax, 'qualifier_mismatches': dm}
    return C


def csv_check(rep_dir, mine):
    out = {}
    for name, fam in (('PRIMARY_ENDPOINTS.csv', 'primary'), ('SECONDARY_ENDPOINTS.csv', 'secondary')):
        with open(rep_dir / name) as f:
            rows = list(csv.DictReader(f))
        bad = []
        for r in rows:
            e = mine['endpoints'].get(r['key'])
            if e is None:
                bad.append(r['key']); continue
            try:
                num_ok = abs(float(r['point']) - e['point']) <= 1e-12 and abs(float(r['se']) - e['se']) <= 1e-12
            except (TypeError, ValueError):
                num_ok = False
            if r['status'] != e['status'] or r['availability'] != e['availability'] or not num_ok:
                bad.append(r['key'])
        out[name] = {'rows': len(rows), 'rows_disagreeing': bad}
    return out


def source_replay(mine):
    """Statistical review S2: services whose DEV scores replay the source bitwise must reproduce the source AUC point and SE exactly."""
    src = V.jload(V.SB_RES / 'report' / 'DEV_REPORT_RESULTS.json')
    name = {'HIST_SCM': 'SCM', 'C0': 'C0', 'NEURAL_RAW': 'NEURAL_RAW', 'H_LJ': 'H_LJ'}
    out = {}
    for s in V.SEEDS:
        for role, sn in name.items():
            r = mine['services'][s].get(role); q = src['per_seed'][f'S{s}']['services'].get(sn)
            if r is None or q is None:
                continue
            for view in ('income', 'occupation', 'pair'):
                a, b = r['recovery'][view], q['recovery'][view]
                out[f'S{s}|{role}|{view}'] = {'winner_mine': a['winner'], 'winner_source': b['winner'], 'same_winner': a['winner'] == b['winner'],
                                             'auc_equal': a['auc'] == b['auc'], 'se_equal': a['auc_se'] == b['auc_se'],
                                             'auc_abs_diff': abs(a['auc'] - b['auc']), 'se_abs_diff': abs(a['auc_se'] - b['auc_se'])}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=('dry', 'real'), required=True); ap.add_argument('--from-crash-copy', action='store_true')
    a = ap.parse_args()
    t0 = time.process_time()
    I = load_inputs()
    if a.mode == 'dry':
        R = I['R']; rng = np.random.default_rng(20261017)
        T = {'row_id': R['row_id'], 'unit': R['unit'], 'income': rng.integers(0, 2, len(R['row_id'])), 'occupation': rng.integers(0, 5, len(R['row_id'])),
             'sex': rng.integers(0, 2, len(R['row_id']))}
        event = None
    elif a.from_crash_copy:
        z = V.npz(CRASH); T = {k: z[k] for k in z}; event = 'crash copy reused (no second logged read)'
        assert np.array_equal(T['row_id'], I['R']['row_id'])
    else:
        sys.path.insert(0, str(V.ROOT))
        from scm_um import common as UMC                  # object under test: the logged access route
        if CRASH.exists():
            raise PermissionError('crash copy exists: use --from-crash-copy instead of a second logged read')
        TT = UMC.dev_report_truth(PURPOSE, event_name='DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER')
        T = {k: np.asarray(TT[k]) for k in ('row_id', 'unit', 'income', 'occupation', 'sex')}
        fd = os.open(CRASH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, **T)
        event = 'DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER'
        assert np.array_equal(T['row_id'], I['R']['row_id']) and np.array_equal(T['unit'], I['R']['unit'])
    mine = analyse(T, I)
    rep_dir = V.RES / 'report'
    rep = V.jload(rep_dir / 'DEV_REPORT_RESULTS.json')
    C = compare(mine, rep)
    csvc = csv_check(rep_dir, mine)
    srcr = source_replay(mine)
    res = {'check': 5, 'mode': a.mode, 'event': event, 'input_checks': I['input_checks'],
           'bootstrap': {'households': mine['households'], 'persons': mine['persons'], 'halt_check': mine['halt_check_3955_5943'],
                         'weights_bitwise_equal_campaign_function': mine['bootstrap_weights_bitwise_equal_campaign_function'], 'B': I['lock']['inference']['bootstrap_B'],
                         'seed': I['lock']['inference']['bootstrap_seed']},
           'critical': mine['critical'], 'controls': mine['controls'],
           'own_claims': {k: {kk: vv for kk, vv in v.items() if kk != 'conditions'} for k, v in mine['claims'].items()},
           'own_separate': mine['separate'], 'own_research_decision': mine['research_decision'], 'own_qualifiers': mine['qualifiers'], 'own_bets': mine['bets'],
           'own_endpoints': {k: {f: e.get(f) for f in ('point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1', 'clause_pass', 'point_pass', 'availability',
                                                       'identical_release', 'status', 'point_status', 'direction', 'MODEL_SELECT_achieved_mismatch_SCM_minus_comparator')}
                             for k, e in mine['endpoints'].items()},
           'comparison_with_report': C, 'csv_check': csvc, 'source_label_free_replay': srcr}
    ok = (all(I['input_checks'].values()) and mine['halt_check_3955_5943'] and mine['bootstrap_weights_bitwise_equal_campaign_function'] is True
          and not C['mismatches'] and all(v <= 1e-12 for v in C['endpoints']['max_abs_diff'].values()) and all(C['claims_equal'].values())
          and not C['condition_mismatches'] and all(C['separate_equal'].values()) and C['research_decision'][0] == C['research_decision'][1]
          and all(C['qualifiers_equal'].values()) and all(C['bets']['equal'].values()) and C['services']['max_abs_diff'] <= 1e-12 and not C['services']['mismatches']
          and C['differences']['max_abs_diff'] <= 1e-12 and not C['differences']['qualifier_mismatches'] and C['controls']['CV_equal']
          and all(not v['rows_disagreeing'] for v in csvc.values()))
    res['status'] = ('PASS' if ok else 'FAIL_OR_REVIEW') if a.mode == 'real' else 'DRY_RUN_COMPLETED (invented labels; values meaningless)'
    res['cpu_s'] = round(time.process_time() - t0, 2)
    if a.mode == 'dry':
        p = SCRATCH / 'check5_dry_run.json'; p.write_text(json.dumps(res, indent=1, sort_keys=True, default=V._default))
        print(json.dumps({'status': res['status'], 'cpu_s': res['cpu_s'], 'input_checks': res['input_checks'], 'bootstrap': res['bootstrap'],
                          'n_endpoints': len(mine['endpoints']), 'n_mismatch_records': len(C['mismatches'])}, indent=1, default=V._default))
    else:
        V.write_json('check5_dev_report.json', res)
        if CRASH.exists():
            CRASH.unlink()
        print(json.dumps({'status': res['status'], 'cpu_s': res['cpu_s'], 'input_checks': res['input_checks'], 'bootstrap': res['bootstrap'],
                          'claims_equal': C['claims_equal'], 'endpoint_max_abs_diff': C['endpoints']['max_abs_diff'], 'mismatches': C['mismatches'][:20],
                          'condition_mismatches': C['condition_mismatches'], 'separate_equal': C['separate_equal'], 'research_decision': C['research_decision'],
                          'qualifiers_equal': C['qualifiers_equal'], 'bets': C['bets'], 'services': C['services'], 'differences': C['differences'], 'csv': csvc,
                          'crash_copy_deleted': not CRASH.exists()}, indent=1, default=V._default))


if __name__ == '__main__':
    main()
