"""Stage CONTROLS (INNER): independent recomputation of the K=11 completion family, utility clauses and T_TREE replay.

Definitions taken from COMPLETION_LOCK.json (endpoints, inference) and the original SCIENCE_LOCK status rules:
  pair benefit     = mean_s[AUC_pair(comparator) - AUC_pair(candidate)]          (winner reader per view)
  recipient excess = mean_s[AUC_r(candidate) - AUC_r(comparator)], cap 0.010
  interval         = point +/- t_{1-0.05/(2K), G-1} * SE_boot, SE from an own paired duplicate-group bootstrap
  benefit status   = ESTABLISHED (lower > margin) / ESTABLISHED_SHORTFALL (upper < margin) / UNRESOLVED
  cap status       = PASS (upper <= cap) / ESTABLISHED_VIOLATION (lower > cap) / UNRESOLVED
"""
from __future__ import annotations

import argparse
import csv
import io
import json

import numpy as np

from . import core

NEW = ('T_REFIT_E20', 'RAW_J_NEQ', 'T_NEURAL', 'FARE_TC', 'FARE_TC_Z')
SLATE = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB')   # frozen reader slate order
INNER_NULL_SEED = 2026101207                                                        # registered INNER null-label seed
OLD_LABEL = {'L_TREE': 'L_TREE', 'T_TREE': 'T_TREE', 'T_REFIT_E5': 'T_REFIT', 'J_TREE': 'J_TREE', 'RAW_PROJ': 'RAW_PROJ',
             'N_EQ': 'N_EQ', 'H_JOINT_CODE': 'H_JOINT_CODE', 'H_SEQ21_CODE': 'H_SEQ21_CODE', 'H_DPC_JOINT': 'H_DPC_JOINT'}
VIEWS = ('income', 'occupation', 'pair')
FAMILY = [  # (group, candidate, comparator, parts)
    ('C1', 'L_TREE', 'T_REFIT_E20', ('pair_benefit', 'income_excess', 'occupation_excess')),
    ('C2', 'L_TREE', 'RAW_J_NEQ', ('pair_benefit', 'income_excess', 'occupation_excess')),
    ('C3', 'RAW_J_NEQ', 'T_NEURAL', ('pair_benefit',)),
    ('C4', 'L_TREE', 'FARE_TC', ('pair_benefit', 'income_excess', 'occupation_excess')),
    ('C5', 'FARE_TC', 'FARE_TC_Z', ('pair_benefit',)),
]
EXACT_TOL = 1e-12
INTERVAL_TOL = 0.003


def bank_of(svc):
    return ('new', svc) if svc in NEW else ('old', OLD_LABEL[svc])


def run(spec):
    from tree_completion.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec, 'verification', allow_inner_truth=True)
    R = core.Receipt('CONTROLS', active, cfg)
    cfg['_code_sha256'] = core.code_hashes(verify, cfg)
    J = lambda it: core.load_json(verify, it)

    # ---------------------------------------------------------------- locks, receipts, report
    rlock = J(core.lock_item(cfg, 'reporting_lock')); slock = J(core.lock_item(cfg, 'selection_lock')); clock = J(cfg['completion_lock'])
    manifest = J(cfg['manifest']); report = J(cfg['report'])
    rep_cfg = J(cfg['tc_report_v2_config'])
    receipts = {b: {s: J(cfg['banks'][b][s]['receipt']) for s in ('0', '1')} for b in ('old', 'new')}
    R.check('reporting_lock_frozen', rlock.get('status') == 'ALL_READERS_FROZEN' and
            rlock.get('INNER_labels_opened_in_this_campaign_before_this_lock') is False)
    R.check('selection_lock_frozen', slock.get('status') == 'COMPLETION_ROSTER_FROZEN' and slock.get('INNER_truth_accessed') is False)
    R.check('completion_lock_K11', clock['endpoints']['family_K'] == 11 and cfg['K'] == 11)
    for s in ('0', '1'):
        R.check(f'new_bank_s{s}_hashes_equal_REPORTING_LOCK',
                cfg['banks']['new'][s]['receipt']['sha256'] == rlock['new_banks'][s]['receipt_sha256'] and
                cfg['banks']['new'][s]['scores']['sha256'] == rlock['new_banks'][s]['scores_sha256'] and
                receipts['new'][s]['inner_scores_sha256'] == rlock['new_banks'][s]['scores_sha256'])
        R.check(f'old_bank_s{s}_hashes_equal_REPORTING_LOCK_and_manifest',
                cfg['banks']['old'][s]['receipt']['sha256'] == rlock['old_banks'][s]['receipt_sha256'] ==
                manifest['frozen_reader_banks'][s]['receipt'] and
                cfg['banks']['old'][s]['scores']['sha256'] == rlock['old_banks'][s]['scores_sha256'] ==
                manifest['frozen_reader_banks'][s]['scores'])
        for b in ('old', 'new'):
            R.check(f'{b}_bank_s{s}_INNER_labels_not_opened_by_readers', receipts[b][s].get('INNER_labels_opened_or_hashed') is False)
        # winners in the new receipts equal the REPORTING_LOCK roster; replay the winner rule from the validation table
        for lab in rlock['new_banks'][s]['winners']:
            w = {v: receipts['new'][s]['services'][lab]['winner'][v]['reader'] for v in VIEWS}
            R.check(f'winners_s{s}_{lab}_equal_REPORTING_LOCK', w == rlock['new_banks'][s]['winners'][lab])
            a = {v: receipts['new'][s]['services'][lab]['winner'][v]['att_validation_auc'] for v in VIEWS}
            R.check(f'winner_auc_s{s}_{lab}_equal_REPORTING_LOCK', a == rlock['new_banks'][s]['att_validation_winner_auc'][lab])
        for b in ('old', 'new'):
            for lab, srec in receipts[b][s]['services'].items():
                ok = True
                ties = []
                for v in VIEWS:
                    # candidate order = the frozen reader slate order (receipts store keys sorted, so order is rebuilt);
                    # pair candidates add the 14 ignore-one-recipient aliases (income readers first); ties -> first in order
                    val = srec['validation'][v]
                    if sorted(val) != sorted(SLATE):
                        ok = False; continue
                    cands = [(n, val[n]['fixed_orientation_auc']) for n in SLATE]
                    if v == 'pair':
                        cands += [(f'IGNORE_occupation__{n}', srec['validation']['income'][n]['fixed_orientation_auc']) for n in SLATE]
                        cands += [(f'IGNORE_income__{n}', srec['validation']['occupation'][n]['fixed_orientation_auc']) for n in SLATE]
                    best = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
                    ok &= cands[best][0] == srec['winner'][v]['reader'] and cands[best][1] == srec['winner'][v]['att_validation_auc']
                    tied = [c[0] for c in cands if c[1] == cands[best][1]]
                    if len(tied) > 1:
                        ties.append({'view': v, 'tied': tied, 'chosen': cands[best][0]})
                R.check(f'winner_rule_replay_{b}_s{s}_{lab}', ok, exact_validation_ties=ties)

    # service -> bank mapping and release paths agree with the TC_REPORT_V2 configuration
    for svc, it in cfg['releases'].items():
        if svc in ('T_TREE_new',):
            continue
        name = 'T_TREE' if svc == 'T_TREE_old' else svc
        R.check(f'release_binding_{svc}_equals_TC_REPORT_V2', all(rep_cfg['releases'][name][s] == it[s] for s in ('0', '1')))
    R.check('service_bank_mapping_equals_TC_REPORT_V2', all(
        rep_cfg['services'][svc] == ({'bank': 'new', 'bank_label': svc} if svc in NEW else {'bank': 'old', 'bank_label': OLD_LABEL[svc]})
        for svc in list(NEW) + list(OLD_LABEL)))

    # ---------------------------------------------------------------- INNER data
    guard()
    truth = {k: core.load_npz(verify, cfg['inner_truth'][k], ['row_id', 'unit', k]) for k in ('sex', 'income', 'occupation')}
    feats = core.load_npz(verify, cfg['inner_features'], ['row_id', 'unit', 'X', 'feature_names'])
    row_id, unit = feats['row_id'], feats['unit']
    n = len(row_id)
    R.check('truth_rows_aligned', all(np.array_equal(truth[k]['row_id'], row_id) and np.array_equal(truth[k]['unit'], unit) for k in truth))
    y = truth['sex']['sex'].astype(np.int64); yi = truth['income']['income'].astype(np.int64); yo = truth['occupation']['occupation'].astype(np.int64)
    R.check('label_conventions', set(np.unique(y)) <= {0, 1} and set(np.unique(yi)) <= {0, 1} and set(np.unique(yo)) <= set(range(6)))
    gidx, G = core.group_index(unit)
    sizes = np.bincount(gidx, minlength=G)
    sex_pure = all(len(np.unique(y[gidx == g])) == 1 for g in np.flatnonzero(sizes > 1))
    R.check('INNER_rows_2235_groups_2234', n == 2235 and G == 2234 and len(np.unique(row_id)) == n, rows=n, groups=G,
            multi_row_groups=int((sizes > 1).sum()), max_group_size=int(sizes.max()))
    R.check('duplicate_groups_sex_pure', sex_pure)
    R.numbers['sex_prevalence_pi1_rows'] = float(y.mean())
    refs = {s: core.load_npz(verify, cfg['inner_references'][s], ['row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2']) for s in ('0', '1')}
    R.check('references_rows_aligned', all(np.array_equal(refs[s]['row_id'], row_id) and np.array_equal(refs[s]['unit'], unit) for s in refs))
    scores = {b: {s: core.load_npz(verify, cfg['banks'][b][s]['scores']) for s in ('0', '1')} for b in ('old', 'new')}
    R.check('scores_rows_aligned', all(np.array_equal(scores[b][s]['row_id'], row_id) and np.array_equal(scores[b][s]['unit'], unit)
                                       for b in scores for s in scores[b]))
    rel = {}
    for svc, items in cfg['releases'].items():
        rel[svc] = {}
        for s in ('0', '1'):
            z = core.load_npz(verify, items[s])
            rel[svc][s] = z
            R.check(f'release_{svc}_s{s}_fields_and_rows', sorted(z) == ['d1', 'd2', 'p1', 'p2', 'row_id'] and np.array_equal(z['row_id'], row_id))

    # ---------------------------------------------------------------- AUC points (winner reader per service, seed, view)
    def winner(svc, s, v):
        b, lab = bank_of(svc)
        return receipts[b][s]['services'][lab]['winner'][v]['reader']

    def score(svc, s, v):
        b, lab = bank_of(svc)
        return scores[b][s][f'{lab}__{v}__{winner(svc, s, v)}']

    services = ['L_TREE'] + list(NEW)
    diag_services = ['T_TREE', 'T_REFIT_E5', 'RAW_PROJ', 'H_SEQ21_CODE', 'H_JOINT_CODE', 'N_EQ', 'H_DPC_JOINT']
    A = {(svc, s, v): core.auc(y, score(svc, s, v)) for svc in services + diag_services for s in ('0', '1') for v in VIEWS}
    # rank formula vs brute-force pairwise oracle on the six family services
    oracle = max(abs(core.auc_pairs(y, score(svc, s, v)) - A[(svc, s, v)]) for svc in services for s in ('0', '1') for v in VIEWS)
    R.check('auc_rank_formula_equals_pairwise_oracle', oracle < 1e-12, max_abs=oracle)

    # ---------------------------------------------------------------- family endpoints (points)
    K = int(cfg['K']); df = G - 1
    crit = core.t_critical(K, df)
    R.check('critical_value_equals_report', abs(crit - report['critical_value']) < 1e-12, mine=crit, report=report['critical_value'])
    R.check('report_groups_draws', report['groups'] == G and report['K'] == K, report_groups=report['groups'])
    req = {'RAW_J_NEQ': slock['required_status']['RAW_J_NEQ']['required_comparator_for_L_TREE'],
           'FARE_TC': slock['required_status']['FARE_TC']['required_comparator_for_L_TREE']}
    R.check('required_status_report_equals_selection_lock', report['required_status'] == slock['required_status'])
    margins = {'C1': 0.0, 'C2': 0.020 if req['RAW_J_NEQ'] else 0.0, 'C3': 0.0, 'C4': 0.020 if req['FARE_TC'] else 0.0, 'C5': 0.0}

    def per_seed_value(part, cand, comp, s, Aget):
        if part == 'pair_benefit':
            return Aget(comp, s, 'pair') - Aget(cand, s, 'pair')
        v = part.split('_')[0]
        return Aget(cand, s, v) - Aget(comp, s, v)

    endpoints = []
    for grp, cand, comp, parts in FAMILY:
        for part in parts:
            ps = {s: per_seed_value(part, cand, comp, s, lambda a, b, c: A[(a, b, c)]) for s in ('0', '1')}
            endpoints.append({'endpoint': f'{grp}|{cand}_vs_{comp}|{part}', 'group': grp, 'cand': cand, 'comp': comp, 'part': part,
                              'per_seed': ps, 'point': (ps['0'] + ps['1']) / 2.0})
    R.check('family_has_11_endpoints', len(endpoints) == 11 == len(report['family']))
    rep_fam = {e['endpoint']: e for e in report['family']}
    max_point_diff = 0.0; bitwise = 0
    for e in endpoints:
        r = rep_fam.get(e['endpoint'])
        if r is None:
            R.check(f'endpoint_present_{e["endpoint"]}', False); continue
        d = max(abs(e['point'] - r['point']), abs(e['per_seed']['0'] - r['per_seed']['0']), abs(e['per_seed']['1'] - r['per_seed']['1']))
        max_point_diff = max(max_point_diff, d); bitwise += int(e['point'] == r['point'])
        R.check(f'point_exact_{e["endpoint"]}', d <= EXACT_TOL, mine=e['point'], report=r['point'], abs_diff=d)
    R.numbers['family_point_max_abs_diff'] = max_point_diff
    R.numbers['family_points_bitwise_equal'] = f'{bitwise}/11'

    # ---------------------------------------------------------------- own paired duplicate-group bootstrap
    guard()
    B = int(cfg['bootstrap']['draws']); rng = np.random.default_rng(int(cfg['bootstrap']['seed']))
    M = core.multiplicities(rng, G, B)                       # B x G
    keys = [(svc, s, v) for svc in services for s in ('0', '1') for v in VIEWS]
    wauc = {k: core.WeightedAUC(y, score(*k)) for k in keys}
    boot = {k: np.empty(B) for k in keys}
    # utility per-row differences for L_TREE + the five new services
    util_specs, util_cols, util_points = [], [], {}
    for svc in services:
        b, lab = bank_of(svc)
        relname = svc
        for s in ('0', '1'):
            for ti, task in enumerate(core.TASKS):
                k = core.KCLS[task]; yt = yi if task == 'income' else yo
                p_svc = rel[relname][s]['p1' if task == 'income' else 'p2']
                for ref in ('U0', 'Ucal'):
                    p_ref = refs[s][f'{ref}_{ti + 1}']
                    point, rows = core.clause_point_and_rows(yt, p_svc, p_ref, k)
                    for m in core.METRICS:
                        name = f'U|{svc}|s{s}|{task}|{ref}|{m}'
                        util_specs.append(name); util_cols.append(rows[m]); util_points[name] = point[m]
    D = np.column_stack(util_cols)
    boot_util = np.empty((B, D.shape[1]))
    chunk = 512
    for a in range(0, B, chunk):
        Wr = M[a:a + chunk][:, gidx]
        for k in keys:
            boot[k][a:a + chunk] = wauc[k](Wr)
        boot_util[a:a + chunk] = (Wr @ D) / Wr.sum(1, keepdims=True)
    # sanity: unit weights reproduce the point AUCs
    ones = np.ones((1, n))
    R.check('weighted_auc_at_unit_weights_equals_point', max(abs(wauc[k](ones)[0] - A[k]) for k in keys) < 1e-12)

    def boot_endpoint(e):
        out = np.zeros(B)
        for s in ('0', '1'):
            out += per_seed_value(e['part'], e['cand'], e['comp'], s, lambda a, b, c: boot[(a, b, c)]) / 2.0
        return out

    max_end_diff = 0.0; status_rows = []
    for e in endpoints:
        r = rep_fam[e['endpoint']]
        se = float(np.std(boot_endpoint(e), ddof=1))
        lo, hi = e['point'] - crit * se, e['point'] + crit * se
        e.update(se=se, lower=lo, upper=hi)
        dlo, dhi = abs(lo - r['lower']), abs(hi - r['upper'])
        max_end_diff = max(max_end_diff, dlo, dhi)
        R.check(f'interval_within_0.003_{e["endpoint"]}', dlo <= INTERVAL_TOL and dhi <= INTERVAL_TOL,
                mine=[lo, hi], report=[r['lower'], r['upper']], se_mine=se, se_report=r['se'])
        if e['part'] == 'pair_benefit':
            m = margins[e['group']]
            e['margin'] = m
            e['status'] = core.benefit_status(lo, hi, m)
            e['status_at_0'] = core.benefit_status(lo, hi, 0.0)
            e['status_at_0.020'] = core.benefit_status(lo, hi, 0.020)
            ok = (abs(r['margin'] - m) < 1e-15 and e['status'] == r['status'] and e['status_at_0'] == r['status_at_0']
                  and e['status_at_0.020'] == r['status_at_0.020'])
            R.check(f'status_{e["endpoint"]}', ok, mine=[m, e['status'], e['status_at_0'], e['status_at_0.020']],
                    report=[r['margin'], r['status'], r['status_at_0'], r['status_at_0.020']])
        else:
            e['status'] = core.cap_status(lo, hi, 0.010)
            R.check(f'status_{e["endpoint"]}', e['status'] == r['status'] and r['cap'] == 0.01, mine=e['status'], report=r['status'])
        status_rows.append({k: e[k] for k in ('endpoint', 'point', 'per_seed', 'se', 'lower', 'upper', 'status')})
    R.numbers['family_interval_end_max_abs_diff'] = max_end_diff
    R.numbers['family'] = status_rows
    R.numbers['bootstrap'] = {'seed': int(cfg['bootstrap']['seed']), 'draws': B, 'critical': crit, 'df': df, 'unit': 'INNER duplicate groups, shared by both seeds and all services'}

    # ---------------------------------------------------------------- labels (lock vocabulary)
    E = {e['endpoint']: e for e in endpoints}
    c1 = E['C1|L_TREE_vs_T_REFIT_E20|pair_benefit']
    c2 = E['C2|L_TREE_vs_RAW_J_NEQ|pair_benefit']
    c2x = [E['C2|L_TREE_vs_RAW_J_NEQ|income_excess'], E['C2|L_TREE_vs_RAW_J_NEQ|occupation_excess']]
    c3 = E['C3|RAW_J_NEQ_vs_T_NEURAL|pair_benefit']
    c4 = E['C4|L_TREE_vs_FARE_TC|pair_benefit']
    c5 = E['C5|FARE_TC_vs_FARE_TC_Z|pair_benefit']

    def rel_label(e, x, required, ctl):
        if required and e['lower'] > 0.020 and all(z['status'] != 'ESTABLISHED_VIOLATION' for z in x):
            return 'MARGIN'
        if e['lower'] > 0:
            return 'SUPERIORITY_ONLY'
        if e['upper'] < 0:
            return 'LOWER_RECOVERY'
        return 'NOT_DISTINGUISHABLE'
    mine = {'C1': 'ESTABLISHED' if c1['lower'] > 0 else 'NOT_ESTABLISHED',
            'C2': rel_label(c2, c2x, req['RAW_J_NEQ'], 'NEURAL_RAWJ'),
            'C3': 'ESTABLISHED' if c3['lower'] > 0 else 'NOT_ESTABLISHED',
            'C4': ('REQUIRED:' if req['FARE_TC'] else 'DESCRIPTIVE_ONLY:') + rel_label(c4, [], req['FARE_TC'], 'FARE'),
            'C5': 'ESTABLISHED' if c5['lower'] > 0 else 'NOT_ESTABLISHED'}
    L = report['labels']

    def parse(lbl):
        if 'NOT_ESTABLISHED' in lbl:
            return 'NOT_ESTABLISHED'
        if 'MARGIN' in lbl:
            return 'MARGIN'
        if 'SUPERIORITY' in lbl:
            return 'SUPERIORITY_ONLY'
        if 'LOWER_RECOVERY' in lbl:
            return 'LOWER_RECOVERY'
        if 'NOT_DISTINGUISHABLE' in lbl:
            return 'NOT_DISTINGUISHABLE'
        return 'ESTABLISHED' if 'ESTABLISHED' in lbl else lbl
    rep_cls = {'C1': parse(L['C1']), 'C2': parse(L['C2']), 'C3': parse(L['C3']),
               'C4': ('DESCRIPTIVE_ONLY:' if L['C4'].startswith('FARE_TC_DESCRIPTIVE_ONLY') else 'REQUIRED:') + parse(L['C4'].split(':')[-1]),
               'C5': parse(L['C5'])}
    for c in ('C1', 'C2', 'C3', 'C4', 'C5'):
        R.check(f'label_{c}_semantics_match', mine[c] == rep_cls[c], mine=mine[c], report=L[c])
    vocab = clock['endpoints']['labels']
    naming = {'C1': L['C1'] in vocab['C1'], 'C2': any(v.split(' ')[0] == L['C2'] for v in vocab['C2'])}
    R.numbers['label_vocabulary_naming'] = {'C1_in_lock_vocabulary': naming['C1'], 'C2_in_lock_vocabulary': naming['C2'],
                                            'C2_report': L['C2'], 'C2_lock_vocabulary': vocab['C2'],
                                            'C3_C5_lock_vocabulary': 'none given in COMPLETION_LOCK',
                                            'C4_report': L['C4'], 'C4_lock_rule': vocab['C4']}
    if not naming['C2']:
        R.notes.append(f"Naming only: report C2 label '{L['C2']}' is not verbatim in the COMPLETION_LOCK vocabulary "
                       f"({vocab['C2'][0].split(' ')[0]}); semantics (lower > 0.020, no excess violations, required comparator) match.")

    # ---------------------------------------------------------------- utility clauses (24 per new service, + L_TREE)
    rep_util = {}
    for svc, lst in report['utility_all_services'].items():
        for r in lst:
            rep_util[r['endpoint']] = r
    new_rep = {r['endpoint']: r for svc in NEW for r in report['utility_new_services'][svc]}
    R.check('utility_new_services_24_each', all(len(report['utility_new_services'][svc]) == 24 for svc in NEW))
    R.check('utility_new_equals_all_services_copy', all(new_rep[k] == rep_util[k] for k in new_rep))
    max_u = 0.0; max_ui = 0.0; u_status_mismatch = []; u_bitwise = 0; u_count = 0; my_sim_pass = {svc: 0 for svc in services}
    for j, name in enumerate(util_specs):
        r = rep_util.get(name)
        if r is None:
            R.check(f'utility_present_{name}', False); continue
        p = util_points[name]; d = abs(p - r['point'])
        u_count += 1; u_bitwise += int(p == r['point'])
        max_u = max(max_u, d)
        se = float(np.std(boot_util[:, j], ddof=1)); lo, hi = p - crit * se, p + crit * se
        max_ui = max(max_ui, abs(lo - r['lower']), abs(hi - r['upper']))
        st = core.cap_status(lo, hi, core.CAPS[name.split('|')[-1]])
        my_sim_pass[name.split('|')[1]] += int(st == 'PASS')
        if st != r['status']:
            u_status_mismatch.append({'endpoint': name, 'mine': st, 'report': r['status'], 'mine_upper': hi, 'report_upper': r['upper'], 'cap': r['cap']})
        if (p <= core.CAPS[name.split('|')[-1]]) != r['point_pass']:
            R.check(f'utility_point_pass_{name}', False, mine=p, report=r['point'])
    svc_ok = {svc: all(abs(util_points[nm] - rep_util[nm]['point']) <= EXACT_TOL for nm in util_specs if nm.split('|')[1] == svc) for svc in services}
    for svc in services:
        R.check(f'utility_24_points_exact_{svc}', svc_ok[svc] and sum(1 for nm in util_specs if nm.split('|')[1] == svc) == 24)
    R.check('utility_intervals_within_0.003', max_ui <= INTERVAL_TOL, max_abs=max_ui)
    R.check('utility_interval_statuses_match', not u_status_mismatch, mismatches=u_status_mismatch)
    R.numbers['utility_point_max_abs_diff'] = max_u
    R.numbers['utility_points_bitwise_equal'] = f'{u_bitwise}/{u_count}'
    R.numbers['utility_interval_end_max_abs_diff'] = max_ui
    R.numbers['utility_point_pass_counts'] = {svc: f"{sum(1 for nm in util_specs if nm.split('|')[1] == svc and util_points[nm] <= core.CAPS[nm.split('|')[-1]])}/24" for svc in services}
    R.numbers['utility_simultaneous_pass_counts_own_bootstrap'] = {svc: f'{my_sim_pass[svc]}/24' for svc in services}

    # service table: AUCs, readers and task metrics per seed
    st_rows = {(r['service'], str(r['seed'])): r for r in report['service_table']}
    max_st = 0.0; st_ok = True; max_env = 0.0
    for svc in services:
        for s in ('0', '1'):
            r = st_rows[(svc, s)]
            for v in VIEWS:
                max_st = max(max_st, abs(A[(svc, s, v)] - r[f'{v}_auc']))
                st_ok &= r[f'{v}_reader'] == winner(svc, s, v)
                b, lab = bank_of(svc)
                env = max(core.auc(y, scores[b][s][k]) for k in scores[b][s] if k.startswith(f'{lab}__{v}__'))
                max_env = max(max_env, abs(env - r[f'{v}_envelope']))
            for task in core.TASKS:
                k = core.KCLS[task]; yt = yi if task == 'income' else yo
                tr = core.task_rows(yt, rel[svc][s]['p1' if task == 'income' else 'p2'], k)
                for m in ('accuracy', 'nll', 'brier'):
                    max_st = max(max_st, abs(float(np.mean(tr[m])) - r[f'{task}_{m}']))
    R.check('service_table_aucs_readers_metrics_exact', st_ok and max_st <= EXACT_TOL, max_abs=max_st)
    R.check('service_table_envelopes_exact (max INNER AUC over all stored readers incl. aliases)', max_env <= EXACT_TOL, max_abs=max_env)

    # diagnostics (descriptive): points exact
    max_dg = 0.0
    for r in report['diagnostic']:
        _, pair, v = r['endpoint'].split('|')
        a, b = pair.split('_minus_')
        ps = {s: A[(a, s, v)] - A[(b, s, v)] for s in ('0', '1')}
        max_dg = max(max_dg, abs((ps['0'] + ps['1']) / 2 - r['point']), abs(ps['0'] - r['per_seed']['0']), abs(ps['1'] - r['per_seed']['1']))
    R.check('diagnostic_42_points_exact', len(report['diagnostic']) == 42 and max_dg <= EXACT_TOL, max_abs=max_dg)

    # reader-control gates on the new banks (shuffled <= 0.60, planted >= 0.99). The shuffled-label reader is scored
    # against group-shuffled INNER SEX (own permutation within duplicate-group-size buckets, registered null seed).
    rng0 = np.random.default_rng(INNER_NULL_SEED); ynull = np.empty_like(y); buckets = {}
    for g in range(G):
        buckets.setdefault(int(sizes[g]), []).append(np.flatnonzero(gidx == g))
    for size in sorted(buckets):
        arr = buckets[size]
        for target, source in zip(arr, rng0.permutation(len(arr))):
            ynull[target] = y[arr[int(source)]]
    max_ctl = 0.0; gates = True; ctl = {}
    for s in ('0', '1'):
        sc = scores['new'][s]
        for v in VIEWS:
            a_sh = core.auc(ynull, sc[f'CONTROL__{v}__SHUFFLED']); a_sh = max(a_sh, 1 - a_sh)
            a_true = core.auc(y, sc[f'CONTROL__{v}__SHUFFLED']); a_true = max(a_true, 1 - a_true)
            planted = np.where(y == 1, sc[f'CONTROL__{v}__PLANTED_1'], sc[f'CONTROL__{v}__PLANTED_0'])
            a_pl = core.auc(y, planted)
            r = report['controls_new_banks'][f's{s}|{v}']
            max_ctl = max(max_ctl, abs(a_sh - r['shuffled_auc_abs']), abs(a_pl - r['planted_auc']))
            gates &= a_sh <= 0.60 and a_pl >= 0.99 and a_true <= 0.60
            ctl[f's{s}|{v}'] = {'shuffled_vs_null_labels': a_sh, 'shuffled_vs_true_labels': a_true, 'planted': a_pl}
    R.check('reader_controls_recomputed_and_gates_pass', gates and max_ctl <= EXACT_TOL and report['all_controls_pass'] is True, max_abs=max_ctl)
    R.numbers['reader_controls'] = ctl

    # ---------------------------------------------------------------- T_TREE replay (new bank vs original bank)
    for s in ('0', '1'):
        new_k = sorted(k for k in scores['new'][s] if k.startswith('T_TREE__'))
        old_k = sorted(k for k in scores['old'][s] if k.startswith('T_TREE__'))
        same = new_k == old_k and all(scores['new'][s][k].dtype == scores['old'][s][k].dtype and
                                      scores['new'][s][k].tobytes() == scores['old'][s][k].tobytes() for k in new_k)
        wn = receipts['new'][s]['services']['T_TREE']['winner']; wo = receipts['old'][s]['services']['T_TREE']['winner']
        vn = receipts['new'][s]['services']['T_TREE']['validation']; vo = receipts['old'][s]['services']['T_TREE']['validation']
        R.check(f'T_TREE_replay_s{s}_scores_bitwise', same and len(new_k) == 35, keys=len(new_k))
        R.check(f'T_TREE_replay_s{s}_winners_and_validation_equal', wn == wo and vn == vo)
        a, b = rel['T_TREE_new'][s], rel['T_TREE_old'][s]
        R.check(f'T_TREE_replay_s{s}_release_bitwise', all(a[f].tobytes() == b[f].tobytes() and a[f].dtype == b[f].dtype for f in a))
        R.check(f'T_TREE_replay_s{s}_report_flag', report['replay_T_TREE'][s] == {'all_bitwise': True, 'keys': 35, 'winners_equal': True})

    # ---------------------------------------------------------------- published copies of the report
    rep_sha = core.file_sha(verify, cfg['report'])
    R.check('results_COMPLETION_REPORT_is_byte_copy', core.file_sha(verify, cfg['report_results_copy']) == rep_sha)
    for name in ('CONTROL_COMPLETION_RESULTS', 'COMPLETION_UTILITY_BOUNDS', 'COMPLETION_SERVICE_TABLE'):
        R.check(f'results_{name}_csv_is_byte_copy', core.file_sha(verify, cfg['report_csv'][name]) == core.file_sha(verify, cfg['results_csv'][name]))
    rows = list(csv.DictReader(io.StringIO(verify(cfg['report_csv']['CONTROL_COMPLETION_RESULTS']['path'],
                                                  cfg['report_csv']['CONTROL_COMPLETION_RESULTS']['sha256']).read_text())))
    fam_csv = {r['endpoint']: r for r in rows}
    R.check('CONTROL_COMPLETION_RESULTS_csv_family_consistent_with_json', all(
        e['endpoint'] in fam_csv and float(fam_csv[e['endpoint']]['point']) == rep_fam[e['endpoint']]['point'] and
        fam_csv[e['endpoint']]['status'] == rep_fam[e['endpoint']]['status'] for e in endpoints))

    R.numbers['labels'] = {'mine': mine, 'report': L}
    R.numbers['auc_points'] = {f'{a}|s{b}|{c}': v for (a, b, c), v in A.items() if a in services}
    return R.write()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    run(ap.parse_args().spec)
