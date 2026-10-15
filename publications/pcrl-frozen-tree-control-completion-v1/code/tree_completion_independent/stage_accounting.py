"""Stage ACCOUNTING: sample sizes in independent duplicate groups, no seed doubling, and internal consistency of
COMPARATOR_EXPOSURE_MATRIX.csv and POPULATION_COMPATIBILITY_AND_USAGE.csv with the locks and reports.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import re

import numpy as np
from scipy import stats

from . import core

EXPO_SERVICE = {'L_TREE (frozen candidate)': 'L_TREE', 'T_TREE': 'T_TREE', 'T_REFIT_E5': 'T_REFIT_E5', 'T_REFIT_E20': 'T_REFIT_E20',
                'J_TREE': 'J_TREE', 'N_EQ': 'N_EQ', 'RAW_PROJ (= historical RAW-J alias)': 'RAW_PROJ', 'RAW_J_NEQ (completion)': 'RAW_J_NEQ',
                'T_NEURAL (completion)': 'T_NEURAL', 'H_JOINT_CODE': 'H_JOINT_CODE', 'H_SEQ21_CODE': 'H_SEQ21_CODE', 'H_DPC_JOINT': 'H_DPC_JOINT',
                'FARE (stored c1)': 'FARE_STORED', 'FARE_Z1 (stored)': 'FARE_Z1_STORED', 'FARE_TC (completion)': 'FARE_TC',
                'FARE_TC_Z (completion)': 'FARE_TC_Z'}


def num(s):
    return int(s.replace(',', ''))


def run(spec):
    from tree_completion.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec, 'verification')
    R = core.Receipt('ACCOUNTING', active, cfg)
    cfg['_code_sha256'] = core.code_hashes(verify, cfg)
    J = lambda it: core.load_json(verify, it)
    T = lambda it: verify(it['path'], it['sha256']).read_text()

    # ---------------------------------------------------------------- role sizes in duplicate groups (row_id/unit only)
    roles = {}
    for name, it in cfg['roles'].items():
        z = core.load_npz(verify, it)
        roles[name] = {'row_id': z['row_id'], 'unit': z['unit'], 'rows': int(len(z['row_id'])), 'groups': int(len(np.unique(z['unit'])))}
    sizes = {k: (v['rows'], v['groups']) for k, v in roles.items()}
    expect = {'UPDATE': (8180, 8176), 'GUARD': (2043, 2043), 'FIT2': (5211, 5209), 'ATT': (4064, 4061), 'INNER': (2235, 2234)}
    R.check('role_rows_and_duplicate_groups', sizes == expect, observed=sizes)
    ids = {k: set(map(int, v['row_id'])) for k, v in roles.items()}
    units = {k: set(map(int, v['unit'])) for k, v in roles.items()}
    names = list(roles)
    overlap_r = {f'{a}&{b}': len(ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    overlap_u = {f'{a}&{b}': len(units[a] & units[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    R.check('roles_disjoint_in_rows_and_groups', all(v == 0 for v in overlap_r.values()) and all(v == 0 for v in overlap_u.values()),
            row_overlaps=overlap_r, group_overlaps=overlap_u)
    pool_rows = sum(roles[k]['rows'] for k in ('UPDATE', 'GUARD', 'FIT2'))
    pool_groups = len(units['UPDATE'] | units['GUARD'] | units['FIT2'])
    R.check('UPDATE+GUARD+FIT2_equals_OSF_DEFENSE_FIT_15434_15428', (pool_rows, pool_groups) == (15434, 15428), rows=pool_rows, groups=pool_groups)
    R.check('teacher_pool_UPDATE+GUARD_10223', roles['UPDATE']['rows'] + roles['GUARD']['rows'] == 10223)
    inner_unit = roles['INNER']['unit']; G = roles['INNER']['groups']; n = roles['INNER']['rows']
    gs = np.unique(inner_unit, return_counts=True)[1]
    R.check('INNER_one_duplicate_group_of_two', int((gs == 2).sum()) == 1 and int(gs.max()) == 2)

    # ---------------------------------------------------------------- seeds share people (no doubling)
    rels = {s: core.load_npz(verify, cfg['inner_release'][s]) for s in ('0', '1')}
    scs = {(b, s): core.load_npz(verify, cfg['scores'][b][s]) for b in ('old', 'new') for s in ('0', '1')}
    same = (np.array_equal(rels['0']['row_id'], rels['1']['row_id']) and np.array_equal(rels['0']['row_id'], roles['INNER']['row_id'])
            and all(np.array_equal(v['row_id'], roles['INNER']['row_id']) and np.array_equal(v['unit'], inner_unit) for v in scs.values()))
    R.check('both_seeds_score_the_same_INNER_people_in_the_same_order', same)
    rep = J(cfg['report']); fin = J(cfg['feasibility_inputs']); meta = J(cfg['power_meta'])
    R.check('report_counts_groups_not_rows_or_seed_copies', rep['groups'] == G == 2234 and abs(rep['critical_value'] - core.t_critical(11, G - 1)) < 1e-12
            and abs(rep['critical_value'] - core.t_critical(11, 2 * G - 1)) > 1e-6, groups=rep['groups'], critical=rep['critical_value'])
    R.check('feasibility_inputs_G_2234_rows_2235', fin['G'] == G and fin['n_rows'] == n)
    R.check('power_model_states_groups_not_seeds', 'NOT\nmodel seeds' in meta['model'] or 'NOT model seeds' in meta['model'].replace('\n', ' '))
    nm = fin['names']; C = np.array(fin['cov_phi']); ix = {k: i for i, k in enumerate(nm)}
    corr = {}
    for svc in ('L_TREE', 'RAW_PROJ', 'H_SEQ21_CODE', 'RAW_J_NEQ'):
        a, b = ix[f'AUC|{svc}|0|pair'], ix[f'AUC|{svc}|1|pair']
        corr[svc] = float(C[a, b] / math.sqrt(C[a, a] * C[b, b]))
    a = np.zeros(len(nm))
    for s in ('0', '1'):
        a[ix[f'AUC|RAW_PROJ|{s}|pair']] += 0.5; a[ix[f'AUC|L_TREE|{s}|pair']] -= 0.5
    se_joint = math.sqrt(a @ C @ a / G)
    seed_of = np.array([k.split('|')[2] if k.startswith('AUC') else k.split('|')[1] for k in nm])
    Cind = np.where(seed_of[:, None] == seed_of[None, :], C, 0.0)   # what treating the seeds as different people would assume
    se_doubled = math.sqrt(a @ Cind @ a / G)
    R.check('cross_seed_dependence_retained', all(v > 0.3 for v in corr.values()) and se_joint > se_doubled,
            cross_seed_corr=corr, se_benefit_vs_RAW_PROJ_joint=se_joint, se_if_seeds_were_independent_people=se_doubled,
            understatement_if_doubled=1 - se_doubled / se_joint)

    # ---------------------------------------------------------------- comparator exposure matrix
    E = {r['service']: r for r in csv.DictReader(io.StringIO(T(cfg['exposure_matrix'])))}
    R.check('exposure_matrix_lists_all_services', set(EXPO_SERVICE) <= set(E) and {'U0', 'Ucal'} <= set(E), missing=sorted(set(EXPO_SERVICE) - set(E)))
    util = rep['utility_all_services']
    stc = J(cfg['stage_controls']) if 'stage_controls' in cfg else None
    mism = []
    for row, svc in EXPO_SERVICE.items():
        m = re.search(r'INNER (?:only )?(\d+)/24 points(?: \((\d+) simultaneous PASS\))?', E[row]['current_utility_eligibility'])
        pts = sum(1 for r in util[svc] if r['point_pass']); sim = sum(1 for r in util[svc] if r['status'] == 'PASS')
        if m is None or int(m.group(1)) != pts or (m.group(2) is not None and int(m.group(2)) != sim):
            mism.append({'service': svc, 'matrix': E[row]['current_utility_eligibility'], 'report_points': pts, 'report_simultaneous': sim})
        if stc and svc in stc['numbers']['utility_point_pass_counts']:
            own_p = int(stc['numbers']['utility_point_pass_counts'][svc].split('/')[0])
            own_s = int(stc['numbers']['utility_simultaneous_pass_counts_own_bootstrap'][svc].split('/')[0])
            if own_p != pts or (m and m.group(2) is not None and own_s != int(m.group(2))):
                mism.append({'service': svc, 'own_points': own_p, 'own_simultaneous': own_s, 'matrix': E[row]['current_utility_eligibility']})
    R.check('exposure_matrix_INNER_utility_counts_match_report_and_own_recompute', not mism, mismatches=mism)
    old = J(cfg['old_inner_report']); oldf = {e['endpoint']: e for e in old['family']}
    fam = {e['endpoint']: e for e in rep['family']}
    diag = {e['endpoint']: e for e in rep['diagnostic']}
    checks = []

    def has(row, text):
        return text in E[row]['current_comparison_status']
    for row, c, val in (('T_TREE', 'T_TREE', '0.087085'), ('N_EQ', 'N_EQ', '0.080048'), ('RAW_PROJ (= historical RAW-J alias)', 'RAW_PROJ', '0.050532')):
        e = oldf[f'B|L_TREE_vs_{c}|pair_benefit']
        checks.append((f'{c}_margin_{val}_ESTABLISHED', has(row, f'margin {val} ESTABLISHED') and f'{e["point"]:.6f}' == val and e['status'] == 'ESTABLISHED'))
    for c, val in (('H_JOINT_CODE', '0.017740'), ('H_SEQ21_CODE', '0.015040'), ('H_DPC_JOINT', '0.013811')):
        e = oldf[f'B|L_TREE_vs_{c}|pair_benefit']
        checks.append((f'{c}_{val}_UNRESOLVED', has(c, f'{val} UNRESOLVED') and f'{e["point"]:.6f}' == val and e['status'] == 'UNRESOLVED'
                       and abs(diag[f'D|{c}_minus_L_TREE|pair']['point'] - e['point']) < 1e-12))
    for row, ep, txt in (('T_REFIT_E20', 'C1|L_TREE_vs_T_REFIT_E20|pair_benefit', '0.0852 ESTABLISHED'),
                         ('RAW_J_NEQ (completion)', 'C2|L_TREE_vs_RAW_J_NEQ|pair_benefit', '0.0664 ESTABLISHED'),
                         ('T_NEURAL (completion)', 'C3|RAW_J_NEQ_vs_T_NEURAL|pair_benefit', '0.0149 ESTABLISHED'),
                         ('FARE_TC_Z (completion)', 'C5|FARE_TC_vs_FARE_TC_Z|pair_benefit', '0.172 ESTABLISHED')):
        dec = len(txt.split(' ')[0].split('.')[1])
        checks.append((f'{ep}_{txt}', has(row, txt) and f'{fam[ep]["point"]:.{dec}f}' == txt.split(' ')[0] and fam[ep]['status'] == 'ESTABLISHED'))
    st = {(r['service'], str(r['seed'])): r for r in rep['service_table']}
    fare_pair = (st[('FARE_TC', '0')]['pair_auc'] + st[('FARE_TC', '1')]['pair_auc']) / 2
    checks.append(('FARE_TC_lower_recovery_pair_0.694_descriptive', has('FARE_TC (completion)', 'pair 0.694') and f'{fare_pair:.3f}' == '0.694'
                   and fam['C4|L_TREE_vs_FARE_TC|pair_benefit']['status'] == 'ESTABLISHED_SHORTFALL' and 'descriptive' in E['FARE_TC (completion)']['current_comparison_status']))
    seq = (st[('H_SEQ21_CODE', '0')]['pair_auc'] + st[('H_SEQ21_CODE', '1')]['pair_auc']) / 2
    code_ok = {c: sum(1 for r in util[c] if r['point_pass']) == 24 for c in ('H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT')}
    code_auc = {c: (st[(c, '0')]['pair_auc'] + st[(c, '1')]['pair_auc']) / 2 for c in code_ok}
    strongest = min((c for c in code_ok if code_ok[c]), key=lambda c: code_auc[c])
    checks.append(('H_SEQ21_strongest_utility_feasible_code_AUC_0.80014', has('H_SEQ21_CODE', 'coalition AUC 0.80014') and f'{seq:.5f}' == '0.80014' and strongest == 'H_SEQ21_CODE'))
    slock = J(core.lock_item(cfg, 'selection_lock')); oldsel = J(cfg['old_selection_lock']); fman = J(cfg['manifest'])
    ro = slock['attack_roster']; oro = oldsel['attack_roster']
    checks += [('checkpoint_L_TREE', 'L_TREE|1.0|E20' in E['L_TREE (frozen candidate)']['checkpoint'] and fman['candidate']['nominee_key'] == 'L_TREE|1.0|E20'),
               ('checkpoint_T_REFIT_E20', E['T_REFIT_E20']['checkpoint'].startswith(ro['T_REFIT_E20'])),
               ('checkpoint_RAW_J_NEQ', E['RAW_J_NEQ (completion)']['checkpoint'] == ro['RAW_J_NEQ']),
               ('checkpoint_T_NEURAL', E['T_NEURAL (completion)']['checkpoint'] == ro['T_NEURAL']),
               ('checkpoint_J_TREE_and_T_REFIT_E5_old_roster', E['J_TREE']['checkpoint'] == oro['J_TREE'] and E['T_REFIT_E5']['checkpoint'] == oro['T_REFIT'])]
    frec = J(cfg['fare_record'])
    checks += [('checkpoint_FARE_TC_F1', E['FARE_TC (completion)']['checkpoint'] == frec['units']['F1']['setting'] and ro['FARE_TC'] == 'FARE|F1'),
               ('checkpoint_FARE_TC_Z_Z1', E['FARE_TC_Z (completion)']['checkpoint'].startswith('Z1') and ro['FARE_TC_Z'] == 'FARE|Z1'),
               ('FARE_TC_access_UPDATE_only_GUARD_SEX_unused', 'UPDATE only' in E['FARE_TC (completion)']['training_records']
                and frec['access']['GUARD_SEX_used'] is False and frec['access']['tree_fit_rows'].startswith('UPDATE (8180)'))]
    t0, t1 = J(cfg['fit2_tables']['0'])['table'], J(cfg['fit2_tables']['1'])['table']
    feas = lambda k: t0[k]['internal_point_feasible'] and t1[k]['internal_point_feasible']
    rawj = [k for k in t0 if k.startswith('RAW_J|') and k.split('|')[2] != 'E0']
    checks += [('FIT2_L_TREE_feasible_both_seeds', feas('L_TREE|1.0|E20') and 'FIT2 point-feasible both seeds' in E['L_TREE (frozen candidate)']['current_utility_eligibility']),
               ('FIT2_T_REFIT_E20_feasible', feas('T_REFIT|none|E20') and 'FIT2 feasible' in E['T_REFIT_E20']['current_utility_eligibility']),
               ('FIT2_RAW_J_E2_only_1_of_12', [k for k in rawj if feas(k)] == ['RAW_J|1.0|E2'] and len(rawj) == 12
                and '1 of 12 candidates feasible' in E['RAW_J_NEQ (completion)']['selection_records']),
               ('FIT2_FARE_none_feasible', not any(feas(f'FARE|F{i}') for i in (1, 2, 3)) and 'none feasible; F1 fallback' in E['FARE_TC (completion)']['selection_records']),
               ('FIT2_J_TREE_E5_feasible_E20_not', all(slock['common_checkpoint_descriptive_FIT2_only']['J_TREE|1.0|E5'][s]['feasible'] for s in ('0', '1'))
                and not any(slock['common_checkpoint_descriptive_FIT2_only']['J_TREE|1.0|E20'][s]['feasible'] for s in ('0', '1'))),
               ('required_status_RAW_J_required_FARE_not', '(required comparator)' in E['RAW_J_NEQ (completion)']['current_comparison_status']
                and slock['required_status']['RAW_J_NEQ']['required_comparator_for_L_TREE'] and not slock['required_status']['FARE_TC']['required_comparator_for_L_TREE'])]
    for name, ok in checks:
        R.check(f'exposure_matrix_{name}', ok)

    # ---------------------------------------------------------------- population compatibility and usage
    P = list(csv.DictReader(io.StringIO(T(cfg['population_csv']))))
    prow = {}
    for r in P:
        m = re.match(r'Adult (UPDATE|GUARD|FIT2|ATT / ATTACK_FIT_NEW|INNER_SELECTION) \(([\d,]+)(?: rows)? / ([\d,]+)', r['population'])
        if m:
            prow[{'ATT / ATTACK_FIT_NEW': 'ATT', 'INNER_SELECTION': 'INNER'}.get(m.group(1), m.group(1))] = (num(m.group(2)), num(m.group(3)), r)
    R.check('population_csv_role_counts_equal_role_files', set(prow) == set(expect) and all(prow[k][:2] == sizes[k] for k in expect),
            parsed={k: v[:2] for k, v in prow.items()})
    adult = [r for r in P if r['population'].startswith('Adult') and 'dropna' not in r['population'] and 'PCRL val split' not in r['population']]
    R.check('population_csv_adult_roles_used_and_zero_independent', all(r['status'] == 'USED' and r['independent_evaluation_groups_available'].startswith('0') for r in adult),
            rows=len(adult))
    inner_txt = prow['INNER'][2]['training_and_attacker_overhead']
    b = oldf['B|L_TREE_vs_RAW_PROJ|pair_benefit']
    hw = (b['upper'] - b['lower']) / 2; t78 = core.t_critical(78, G - 1)
    R.check('population_csv_INNER_precision_statement', '~0.025' in inner_txt and 't=3.418' in inner_txt and 'SE ~0.0073' in inner_txt and 'K=78' in inner_txt
            and round(hw, 3) == 0.025 and f'{t78:.3f}' == '3.418' and round(b['se'], 4) == 0.0073 and old['K'] == 78,
            half_width=hw, t78=t78, se=b['se'])
    R.check('population_csv_FIT2_note_15434_15428', '15,434 (15,428 groups)' in prow['FIT2'][2]['notes'])
    R.check('population_csv_UPDATE_teacher_pool_10223', '10,223' in prow['UPDATE'][2]['training_and_attacker_overhead'])
    R.check('population_csv_ATT_AUDIT_FIT_arithmetic', 'AUDIT_FIT 6,065 = ATT 4,064 + CALIBRATION_HELDOUT group rows 2,001' in prow['ATT'][2]['notes'] and 4064 + 2001 == 6065)
    aud = T(cfg['population_audit'])
    R.check('population_audit_39170_row_space_arithmetic', '39,170' in aud and 15434 + 1500 + 6065 + 2235 + 13936 == 39170)
    R.numbers['role_sizes'] = sizes
    R.numbers['seed_dependence'] = {'cross_seed_corr_pair_auc': corr, 'se_joint': se_joint, 'se_if_independent': se_doubled}
    return R.write()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    run(ap.parse_args().spec)
