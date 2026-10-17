"""Assemble INDEPENDENT_VERIFICATION.json from the verifier's private aggregate outputs (no label, no person-level value).

Public-use ACS 2019 Florida person records; readers infer the recorded SEX column from the released income and occupation
predictions. Every number below was produced by the verifier's own code in roles/verifier/code/.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V

DISCREPANCIES = [
    {'id': 'V-D1', 'severity': 'minor', 'check': 9, 'location': 'OUTPUT_MAPPING_CHECK.md, Part 1 "Only repeated vectors can test it"',
     'description': 'States "seed 0: 6 such vectors in the union" and "seed 1: 49 such vectors". The verifier counts 5 (seed 0) and 46 (seed 1) '
                    'occupation vectors shared by distinct encoded inputs; 6 and 49 are the excess distinct inputs (inputs minus vectors: seed 0 '
                    'has one vector shared by 3 inputs; seed 1 has one shared by 3 and one by 4).',
     'effect': 'none on any conclusion; the 2 seed-1 conflicts are confirmed'},
    {'id': 'V-D2', 'severity': 'minor', 'check': 9, 'location': 'OUTPUT_MAPPING_CHECK.md, Part 2 "Routing cells"',
     'description': 'States "587 depth-4 income trees and 800 occupation trees per seed". True for seed 0; seed 1 has 529 income trees and 799 '
                    'occupation trees (admitted teacher JSONs).',
     'effect': 'none; the routing-cell counts were computed from the actual teachers and are reproduced exactly'},
    {'id': 'V-D3', 'severity': 'minor', 'check': 9,
     'location': 'CONFIRMATION_FEASIBILITY.md conclusion 1; RESEARCH_DECISION.md section 4 (and the "about 17,500 households" sentence it repeats)',
     'description': 'States that at about 17,500 households joint power is "at most about 0.2" for the source scorecard (D1). Under the plug-in, '
                    'ORIGINAL D1 power at 17,510 households is 0.25 (V2 curve, interpolated), 0.23 (verifier bootstrap rescaling) and 0.24 '
                    '(verifier normal Monte Carlo); 0.20 is the value at about 10,000 households (the sealed 20% alone). ENHANCED (governing) is '
                    'about 0.01 at 17,510.',
     'effect': 'qualitative conclusion (no realistic confirmation) unchanged'},
    {'id': 'V-D4', 'severity': 'minor', 'check': 8,
     'location': 'CONFIRMATION_FEASIBILITY.md ("No status label or conclusion changed"), RESEARCH_DECISION.md section 4 ("with no label changed"), '
                 'PROTOCOL_DEVIATIONS.md item 9 ("no status changed")',
     'description': 'V1 -> V2 changed the S6 assurance status_80 label from NOT_REACHED_BY_1E8 (V1, NaN limit) to LIMIT_BELOW_TARGET in all 6 '
                    'scorecard x definition cells, and the governing-scorecard assignment in 8 scenario x definition cells (new rule when neither '
                    'scorecard reaches 80%). No n80 or n90 value changed.',
     'effect': 'conclusions unchanged; the statement should say which labels changed'},
    {'id': 'V-D5', 'severity': 'minor', 'check': 9, 'location': 'CONFIRMATION_FEASIBILITY.md results table header "ENHANCED governs; ORIGINAL in brackets where it differs"',
     'description': "Under V2's own governing rule, ORIGINAL governs the D3 column at slack 0.0005, 0.001, 0.002 and the D3 plug-in "
                    '(ORIGINAL limit 0), and the D2 cell at slack 0. The bracketed ORIGINAL values are shown, so no number is wrong, but the header '
                    'is inaccurate for those cells.',
     'effect': 'none on the numbers'},
    {'id': 'V-D6', 'severity': 'minor', 'check': 9, 'location': 'RESEARCH_DECISION.md section 4 "What binds ... the recovery clauses alone pass at DEV\'s size"',
     'description': "Only the four pair-benefit clauses pass at DEV's size in ENHANCED. The ENHANCED seed-1 income recipient non-worsening clause "
                    'against H_LJ, also an AUC recovery clause, fails at DEV size (U1 0.0235 > 0.010) and needs about 318,000 households alone for '
                    '80% (verifier analytic 317,958; CONFIRMATION_FEASIBILITY.md states this correctly).',
     'effect': 'the claim that utility binds still holds (318,000 is far below 4.6 million)'},
    {'id': 'V-D7', 'severity': 'minor', 'check': 9, 'location': 'RESEARCH_DECISION.md "Output structure", second bullet',
     'description': '"two occupation vectors carry different income codes, one on each side of the income decision" is ambiguous: each of the two '
                    'vectors accompanies income levels 2 and 3 (decisions 0 and 1), so each straddles the decision threshold.',
     'effect': 'wording only'},
    {'id': 'V-D8', 'severity': 'informational', 'check': 9, 'location': 'RESEARCH_DECISION.md "Status of the evidence"',
     'description': 'Says DEV_REPORT "was read once more" (22:27:16Z). The text predates the separately logged verifier replay '
                    '(DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER, 22:50:37Z, unit VER_DEVREPORT_V1), which the protocol registered; '
                    'the campaign has two logged re-reads.',
     'effect': 'disclosure completeness only'},
    {'id': 'V-D9', 'severity': 'cosmetic', 'check': 6, 'location': 'report/DEV_REPORT_RESULTS.json, both scorecards',
     'description': 'The two IDENTICAL_RELEASE secondary keys SEC|occupation_nll_diff_vs_H_LJ|s0 and |s1 (family S, descriptive) carry '
                    'clause_pass = false where the other family-S keys carry null.',
     'effect': 'none; family-S keys enter no claim'},
    {'id': 'V-D10', 'severity': 'informational', 'check': 6, 'location': 'PROTOCOL.md section 8 side attribution; report interpretation of B and M_SS',
     'description': 'The side attribution is defined "from the paired reader-class changes of the failing benefit", but the first failing ENHANCED '
                    'condition of B (and of M step 2) is a recipient clause (COMPETITIVE|income_worsening_vs_H_LJ|s1). The report applies the same '
                    'labels to that clause: FINALIST income winner unchanged (0.0), H_LJ income winner -0.01112 (LOG_MLP -> HGB15 on an AV gap '
                    'of 0.00062), giving COMPARATOR_READ_LOWER. The verifier reproduces these numbers; the label meaning carries over.',
     'effect': 'none on numbers; the extension of the rule should be stated'},
    {'id': 'V-D11', 'severity': 'informational', 'check': 8, 'location': 'governing-scorecard assignment in near-tie scenarios',
     'description': 'Where neither scorecard reaches 80% and the powers at 1e8 differ by Monte Carlo noise (S3 halfway D2: 0.4194 vs 0.4189; at-cap '
                    'D2: 0.0004 vs 0.0003), and for the S4 combined HYPOTHETICAL D2 tie at 59.9 M, the verifier methods pick the other scorecard. '
                    'No decision depends on these cells.',
     'effect': 'none'},
    {'id': 'V-D12', 'severity': 'resolved', 'check': 8, 'location': 'CONFIRMATION_FEASIBILITY_RESULTS.json (V1, kept)',
     'description': 'V1 limits used an independence product (D1 at the cap 0.0625; D2 4.8e-9; assurance NaN). The verifier analytic values are '
                    '0.2647 (orthant probability, correlations from its own replicates) and 0.00034 (MVN CDF at -c1), in agreement with V2 '
                    '(0.2657 / 0.2681 and 0.0004 / 0.0003).',
     'effect': 'corrected in V2'},
]


def main():
    c1 = V.jload(V.VPRIV / 'check1_identity.json')
    c2 = V.jload(V.VPRIV / 'check2_exposure.json')
    c3 = V.jload(V.VPRIV / 'check3_mapping.json')
    c5 = {s: V.jload(V.VPRIV / f'check5_refit_S{s}.json') for s in V.SEEDS}
    dev = V.jload(V.VPRIV / 'dev_results.json')
    cmp_ = V.jload(V.VPRIV / 'check6_compare.json')
    c8 = V.jload(V.VPRIV / 'check8_power.json')
    fu = V.jload(V.VPRIV / 'check4_9_followup.json')
    ev = V.jload(V.VPRIV / 'dev_truth_event.json')
    led = [json.loads(l) for l in (V.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
    units = [{k: e[k] for k in ('unit_id', 'status', 'returncode', 'charged_cpu_s', 'wall_s', 'utc')} for e in led
             if e.get('event') == 'unit_end' and str(e.get('unit_id', '')).startswith('VER_')]
    occ = c1['occupation_wire_identity']
    checks = {}
    checks['1_hashes_and_identity'] = {'status': 'PASS', 'numbers': {
        'admitted_copies_rehashed': c1['admission_rehash']['n_rehashed'], 'listed': c1['admission_rehash']['n_copies_listed'],
        'sha256_mismatches': len(c1['admission_rehash']['sha256_mismatch']), 'with_expected_sha256': c1['admission_rehash']['n_with_expected_sha256'],
        'expected_sha256_mismatches': len(c1['admission_rehash']['expected_sha256_mismatch']),
        'cross_checks_against_source_records_all_true': True,
        'source_reader_model_files_rehashed': [c1['cross_checks_against_source_records'][f'source_reader_model_files_S{s}']['n'] for s in V.SEEDS],
        'enhanced_reader_model_files_rehashed': [c1['cross_checks_against_source_records'][f'enhanced_readers_S{s}']['n_model_files'] for s in V.SEEDS],
        'release_sha256_equals_FINALIST_LOCK': {s: m['release_sha256_manifest_equals_FINALIST_LOCK'] and m['release_file_sha256_equals_manifest'] for s, m in c1['frozen_candidate_manifest'].items()},
        'released_income_rows_equal_a_per_level_vector_bitwise': {s: m['released_income_rows_all_equal_a_per_level_vector_bitwise'] for s, m in c1['frozen_candidate_manifest'].items()},
        'level_counts_equal_release_record_all_roles': {s: all(v['level_counts_equal_release_record_coarse_cell_counts'] for v in m['per_role'].values()) for s, m in c1['frozen_candidate_manifest'].items()},
        'own_6row_per_level_recompute_last_bit': {s: m['own_6row_recompute_from_H'] for s, m in c1['frozen_candidate_manifest'].items()},
        'occupation_wire_FINALIST_eq_H_LJ_eq_B2TWIN_bank_roles': {k: v for k, v in occ.items() if k.endswith('bank_roles')},
        'occupation_wire_FINALIST_eq_B2TWIN_release_all_six_roles': {k: all(v.values()) for k, v in occ.items() if 'release_FINALIST_vs_B2TWIN' in k},
        'own_JOINT_TREE_E20_occupation_recompute_bitwise_all_six_roles': {k: all(x['own_joint_tree_occupation_bitwise_equal_release'] for x in v.values()) for k, v in occ.items() if 'own_recompute' in k},
        'own_income_level_recompute_rows_differing': {k: sum(x['own_income_level_differs'] for x in v.values()) for k, v in occ.items() if 'own_recompute' in k},
        'unique_single_recipient_wires': {k: v for k, v in occ.items() if 'unique' in k}}}
    ho = c2['household_overlap']
    checks['2_exposure_ledger'] = {'status': 'PASS', 'numbers': {
        'spot_checks': c2['spot_check_summary']['n'], 'spot_checks_matched': c2['spot_check_summary']['matched'],
        'spot_check_claims': [x['claim'] for x in c2['spot_checks']],
        'role_pairs_checked': ho['n_pairs'], 'role_pairs_sharing_households_or_persons': sum(1 for v in ho['pairs'].values() if v['shared_households'] or v['shared_persons']),
        'per_role_persons_households_largest_match_audit_table': {r: v['audit_table_match'] for r, v in ho['per_role'].items()},
        'union_persons': ho['union_distinct_persons'], 'union_households': ho['union_distinct_households'],
        'equals_extraction_summary': all(ho['equals_extraction_summary'].values()), 'sealed_assessment_overlap': 'UNKNOWN (not opened)'}}
    m3 = {}
    for s, v in c3['per_seed'].items():
        m3[s] = {'per_role_count_differences_vs_reported': {r: x for r, x in v['differences_vs_reported_per_role'].items() if x},
                 'union_differences_vs_reported': v['differences_vs_reported_union'], 'union': v['union'],
                 'cross_role_conflicts': v['conflicts'], 'routing_cells': v['routing_cells'],
                 'routing_cell_differences_vs_reported': v['differences_vs_reported_cells'],
                 'per_variable_class_differences_vs_reported': {k: x for k, x in v['differences_vs_reported_per_variable'].items() if x},
                 'reported_invented_examples_confirmed_through_package': sum(1 for e in v['reported_examples_confirmed'] if e['package_occupation_bitwise_equal']
                                                                             and (e['kind'].startswith('examples_distinct') or e['package_income_differs'])),
                 'reported_examples_total': len(v['reported_examples_confirmed']),
                 'own_bounded_search': {k: x for k, x in v['own_bounded_search'].items() if k != 'value_pairs_same_occ_class_diff_income_class'}}
    checks['3_output_mapping'] = {'status': 'PASS', 'numbers': m3, 'note': 'reported income strings of some examples differ in the last bit under a '
                                  '2-row package request (characterized batch-size dependence); levels, decisions and occupation bitwise equality agree'}
    checks['4_reader_feature_access'] = {'status': 'PASS', 'numbers': {
        **{s: {k: fu['check4'][s][k] for k in ('design_width_mismatches', 'n_designs', 'occupation_wire_alias', 'aliased_occupation_winners_share_one_reader',
                                             'source_origin_winner_auc_equals_source_record', 'n_source_origin_winners', 'DEV_REPORT_labels_opened')} for s in ('S0', 'S1')},
        'sex_read_log': fu['check4']['sex_read_log'],
        'code_review': 'ffv/readers.py deletes X and feature_names from the three role exports before any fit; features.py builds every design from '
                       'released probability vectors only; design_for and FiniteScaler are fitted on ATTACK_FIT rows (trip[0], X3[0]); unique wires '
                       'are fitted once and aliased; source candidates enter with SRC:: keys, stored AV AUCs and stored DEV scores (report.score_of)',
        'refits_bitwise_with_ATTACK_FIT_only_fitting': True}}
    rf = {}
    for s in V.SEEDS:
        r = c5[s]
        rf[f'S{s}'] = {'refits': r['C_summary']['n_refits'], 'dev_scores_bitwise': r['C_summary']['dev_bitwise'], 'new_reader_av_scores_bitwise': r['C_summary']['av_bitwise_new'],
                       'max_abs_dev_score_diff': r['C_summary']['max_abs_dev_diff'], 'max_abs_av_auc_diff': r['C_summary']['max_abs_av_auc_diff'],
                       'roster_fixed': r['C_roster']['fixed'], 'enhanced_pair_winners': r['C_roster']['enhanced_pair_winners'],
                       'cati_state_equals_lock': [v.get('cati_state_equals_lock') for v in r['C_refits'].values() if v.get('cati_state_equals_lock') is not None],
                       'estimator_params_equal_lock': all(r['estimator_params_equal_lock'].values()), 'versions_equal_lock': all(r['versions_equal_lock'].values()),
                       'av_auc_per_refit': {k: v['own_attack_validation_auc'] for k, v in r['C_refits'].items()}}
    rf['dev_auc_diff_own_minus_stored_max_abs'] = cmp_['refit_dev_auc']['max_abs_diff']
    rf['dev_auc_per_refit'] = {k: v['own_dev_auc'] for k, v in dev['refit_dev_auc'].items()}
    checks['5_representative_reader_refits'] = {'status': 'PASS', 'numbers': rf}
    sc6 = {}
    for SC, v in cmp_['scorecards'].items():
        sc6[SC] = {'endpoints': v['n_endpoints'], 'max_abs_diff': v['max_abs_diff'], 'categorical_mismatches': v['categorical_mismatches'],
                   'claims': {n: c['mine'] for n, c in v['claims'].items()}, 'claims_equal_coordinator': all(c['equal'] for c in v['claims'].values()),
                   'first_failing': {n: x['mine'] for n, x in v['first_failing'].items()}, 'M_steps': v['M_steps'], 'E_fields': v['E_fields'],
                   'E_fields_equal': v['E_fields_equal'], 'CV_equal': v['CV_equal']}
    key = {}
    for SC in ('ORIGINAL', 'ENHANCED'):
        E = dev['scorecards'][SC]['endpoints']
        key[SC] = {k: {'point': E[k]['point'], 'se': E[k]['se'], 'clause_L1': E[k]['clause_L1'], 'clause_U1': E[k]['clause_U1'], 'status': E[k]['status']}
                   for k in sorted(E) if k.startswith(('COMPETITIVE|pair', 'MECH_TO|pair', 'JOINT|pair', 'SEC|IBSI', 'COMPETITIVE|income_worsening_vs_H_LJ|s1'))}
    w = {}
    for s in V.SEEDS:
        b = c5[s]['B_winners']
        w[f'S{s}'] = {'re_derived': b['n'], 'mismatches': b['mismatches'], 'enhanced_differs_from_original': sum(1 for x in b['per_release_view'].values() if x['enhanced_key'] != 'SRC::' + x['original_key']),
                      'near_tie_flags': sum(1 for x in b['per_release_view'].values() if x['near_tie']),
                      'av_aucs_recomputed_exactly': [c5[s]['A_attack_validation_auc_recomputation']['n_exactly_equal'], c5[s]['A_attack_validation_auc_recomputation']['n_compared']]}
    checks['6_selection_and_primary_endpoints'] = {'status': 'PASS', 'numbers': {
        'winners': w, 'scorecards': sc6, 'key_endpoints_verifier': key, 'interpretation': cmp_['interpretation'], 'controls': cmp_['controls'],
        'recovery_aucs': cmp_['recovery'], 'reader_class_change': cmp_['reader_class_change'], 'benefit_change': cmp_['benefit_change'],
        'reader_matched_benefits': cmp_['reader_matched_benefits'], 'variation': cmp_['variation'], 'original_vs_source_report': cmp_['original_vs_source_report'],
        'critical_values': cmp_['critical'], 'bootstrap': 'household-level, B = 2048, seed 2026101705, task_headroom.inference.group_weights; own AUC and ratio-mean code'}}
    uh = dev['utility_headroom']
    checks['7_utility_and_subgroup_headroom'] = {'status': 'PASS', 'numbers': {
        'n_utility': uh['n_utility'], 'n_guards': uh['n_guards'], 'all_point_gates_pass': uh['all_point_pass'], 'n_U1_le_cap': uh['n_U1_le_cap'],
        'binding_ascending_slack_over_se': uh['ascending_slack_over_se'][:5],
        'binding': {k: {f: uh['endpoints'][k][f] for f in ('point', 'se', 'cap', 'slack', 'slack_over_se', 'clause_U1')} for k in
                    ('UTILITY|income|Ucal|brier_excess|s0', 'UTILITY|income|Ucal|brier_excess|s1', 'UTILITY|income|U0|accuracy_loss|s1', 'UTILITY|income|Ucal|accuracy_loss|s1')},
        'guards': {k: {f: uh['endpoints'][k][f] for f in ('point', 'se', 'slack_over_se')} for k in uh['endpoints'] if k.startswith('SUBGROUP|')},
        'utility_all_services_vs_coordinator_max_abs_diff': cmp_['utility_all_services']['max_abs_diff'],
        'family_status_counts': dev['scorecards']['ENHANCED']['E_fields']['simultaneous_utility_counts']}}
    p8 = {}
    for SC, r in c8['scorecards'].items():
        cm = r['comparisons']
        p8[SC] = {'conditions': r['conditions'], 'by_kind': r['by_kind'],
                  'S1_S2_D1_D2': {k: {f: cm[k][f] for f in ('coordinator_V2_n80', 'boot_n80', 'mvn_n80', 'boot_grid_steps_from_V2', 'mvn_grid_steps_from_V2',
                                                              'coordinator_V2_n90', 'boot_n90', 'mvn_n90')} for k in ('S1_plugin|D1', 'S1_plugin|D2', 'S2_benefit_x0.75|D1', 'S2_benefit_x0.75|D2')},
                  'max_grid_step_disagreement_all_scenarios': max(abs(x) for k, v in cm.items() if isinstance(v, dict) and 'boot_grid_steps_from_V2' in v
                                                                 for x in (v['boot_grid_steps_from_V2'], v['mvn_grid_steps_from_V2']) if x is not None),
                  'n80_none_agreement_all_scenarios': all((v['coordinator_V2_n80'] is None) == (v['boot_n80'] is None) == (v['mvn_n80'] is None)
                                                          for k, v in cm.items() if isinstance(v, dict) and 'boot_n80' in v),
                  'analytic_single_clause_n80': {k: v['n80'] for k, v in r['A_single_clause'].items() if ('UTILITY' in k) or ('H_LJ|s1|D1D2' in k and 'worsening' in k)
                                                 or ('NEURAL_RAW|s1|S1|D3' in k)},
                  'limits_at_cap': {k: v for k, v in r['A_limits_at_cap'].items() if k != 'correlations'},
                  'assurance_limits': {k: v for k, v in cm.items() if k.startswith('S6')},
                  'same_size_replication': cm['same_size_replication_plugin'],
                  'tau_reader_refit_vs_V2_max_abs_diff': r['tau_vs_coordinator_V2_max_abs_diff']}
    checks['8_power_and_sample_size'] = {'status': 'PASS', 'numbers': {
        'verified_file': 'CONFIRMATION_FEASIBILITY_RESULTS_V2.json (V1 compared: identical n80 / n90 in every cell; limits and S6 labels differ, see V-D4, V-D12)',
        'methods': 'A analytic n = 3955((z + c) SE / slack)^2 and orthant / MVN-CDF limits; B nonparametric bootstrap-deviation rescaling of the verifier '
                   'replicates; C own MVN Monte Carlo (seed 61017, 20,000 draws, X = Z D_c / sqrt(B - 1), no factorization)',
        'grid_equals_registered': c8['grid_equals_registered_logspace'], 'grid_step_ratio': c8['grid_step_ratio'], 'scorecards': p8,
        'governing_disagreements': c8['governing_disagreements'], 'power_at_quoted_sizes_plugin': fu['power_at_quoted_sizes_plugin']}}
    checks['9_reporting_labels_and_limitations'] = {'status': 'PARTIAL', 'numbers': {
        'documents_read': ['CONFIRMATION_FEASIBILITY.md', 'OUTPUT_MAPPING_CHECK.md', 'RESEARCH_DECISION.md', 'HANDOFF_DECISION.json', 'START_HERE.md (skimmed)',
                           'PROTOCOL_DEVIATIONS.md'],
        'unsupported_or_imprecise_statements': [d['id'] for d in DISCREPANCIES if d['check'] in (8, 9) and d['severity'] in ('minor',)],
        'supported_examples': ['binding slack table (0.027 / 0.042 / 0.110 SE; P(slack > 0) 0.51 / 0.52 / 0.54)', 'n80 tables (all cells)',
                               'assurance limits 0.102 / 0.161 / 0.053 / 0.064', 'same-size replication 0.003 / 0.115', 'recipient clause n80 about 318,000; '
                               'NEURAL_RAW s1 0.020 n80 about 522,000', 'reader-matched: 14 of 15 non-LOOKUP recipes positive per seed and reference, range '
                               '+0.005 to +0.034, CONF_HGB -0.015 to -0.002, LOOKUP about 0', 'H_LJ s1 income winner LOG_MLP 0.5517 -> HGB15 0.5523 (AV), DEV 0.549 -> 0.538',
                               'all 24 bootstrap refits read lower; benefit shrank in 3 of 4 cells', 'Part 3 reconstruction reproduced exactly',
                               'pair-benefit and recipient numbers in RESEARCH_DECISION.md', 'men = SEX code 0 (schema SEX_CODE) for the 0.0032 / 0.0040 guard'],
        'documents_reviewed_sha256': {n: V.sha_file(V.RES / n) for n in ('CONFIRMATION_FEASIBILITY.md', 'OUTPUT_MAPPING_CHECK.md', 'RESEARCH_DECISION.md',
                                                                          'HANDOFF_DECISION.json', 'PROTOCOL_DEVIATIONS.md', 'CONFIRMATION_FEASIBILITY_RESULTS_V2.json',
                                                                          'CONFIRMATION_FEASIBILITY_RESULTS.json', 'report/DEV_REPORT_RESULTS.json', 'OUTPUT_MAPPING_RESULTS.json')
                                      if (V.RES / n).is_file()},
        'reader_matched': fu['reader_matched'], 'H_LJ_S1_income': fu['H_LJ_S1_income'], 'union_shared_vectors': fu['union_shared_vectors'],
        'reconstruction_reproduced': all(abs(fu['reconstruction']['own'][s][r]['level_accuracy'] - fu['reconstruction']['reported'][s][r]['level_accuracy']) == 0
                                         for s in ('S0', 'S1') for r in ('ATTACK_VALIDATION', 'DEV_REPORT'))}}
    # final re-check of the revised documents (VER_RECHECK units): per-item resolution and the residual items
    rc_path = V.VPRIV / 'check9_recheck.json'
    if rc_path.is_file():
        rc = V.jload(rc_path)
        by_item = {}
        for t in rc['phrase_tests']:
            by_item.setdefault(t['item'], []).append(t['resolved'])
        for d in DISCREPANCIES:
            if d['id'] in by_item:
                d['status_at_final_recheck'] = 'RESOLVED_IN_CURRENT_TEXT' if all(by_item[d['id']]) else 'PARTLY_OR_NOT_RESOLVED'
            elif d['id'] in ('V-D9',):
                d['status_at_final_recheck'] = 'ACKNOWLEDGED (frozen evaluation output; not regenerated)'
            elif d['id'] in ('V-D10', 'V-D11'):
                d['status_at_final_recheck'] = 'ACKNOWLEDGED_IN_CURRENT_TEXT'
            elif d['id'] == 'V-D12':
                d['status_at_final_recheck'] = 'RESOLVED_IN_V2'
        residual = [t for t in rc['phrase_tests'] if not t['resolved']]
        checks['9_reporting_labels_and_limitations']['status'] = 'PASS' if not residual else 'PARTIAL'
        checks['9_reporting_labels_and_limitations']['numbers']['final_recheck'] = {
            'utc': rc['utc'], 'phrase_tests': rc['phrase_tests'], 'residual': residual, 'document_sha256_at_recheck': rc['document_sha256'],
            'mapping_conflicts_json_verified': rc['mapping_conflicts_json'],
            'new_statements_verified': ['H_LJ s1 income change marginal 95% interval -0.024 to +0.001 (verifier -0.0236 to +0.0014)',
                                        "H_LJ s0 income winner change -0.0085 (verifier -0.00854)",
                                        'FINALIST s1 income winner on an exact AV tie (gap 0.0) broken by the source-first rule',
                                        'bootstrap-refit benefits range -0.001 to +0.031 (verifier -0.0011 to +0.0314)',
                                        '2 (seed 0) and 13 (seed 1) within-role vectors shared by distinct inputs',
                                        'power at about 17,500 households 0.013 [0.252] (V2 interpolation; verifier 0.009-0.013 [0.23-0.24])',
                                        'tau_r 0.0036-0.0134; tau_h 0.0017 / 0.0037 ENHANCED and 0.0024 / 0.0050 ORIGINAL',
                                        'governing-scorecard description (ORIGINAL in the D3 column, the D2 boundary cells and the combined-term D1 cell)',
                                        'MAPPING_CONFLICTS.json: 5 / 46 shared vectors, excess 6 / 49, maximum 3 / 4 inputs, both seed-1 conflicts (roles, levels, POBP only)']}
        if any(t['item'] == 'V-D4' and t['document'] == 'RESEARCH_DECISION.md' and not t['old_phrase_absent'] and t['corrected_phrase_present'] is None
               for t in residual):
            DISCREPANCIES.append({'id': 'V-D13', 'severity': 'minor', 'check': 9, 'location': 'RESEARCH_DECISION.md, "Why validation is not incomplete (option 4)"',
                                  'description': 'Residual of V-D4 after the revisions: "through a disclosed V2 that changed no label". V2 changed no claim label '
                                                 'and no n80 / n90 value, but it did change 6 feasibility status labels and 8 governing assignments, as the same '
                                                 'document (section 4) and PROTOCOL_DEVIATIONS.md item 9 now state.',
                                  'effect': 'wording only', 'status_at_final_recheck': 'OPEN'})
    code = {p.name: V.sha_file(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}
    out = {'schema': 'FFV1-independent-verification-v1', 'campaign': V.CAMPAIGN, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'role': 'independent numerical verifier (separate agent); own code in roles/verifier/code/ (no reuse of ffv/report.py, ffv/feasibility*.py or ffv/mapping.py)',
           'status': {k: v['status'] for k, v in checks.items()}, 'checks': checks, 'discrepancies': DISCREPANCIES,
           'major_discrepancies': 0,
           'dev_truth_event': {'event': ev['event'], 'utc': ev['utc'], 'purpose': ev['purpose'], 'unit_id': ev['unit_id'],
                               'reporting_lock_sha256': ev['reporting_lock_sha256'], 'uses': 1},
           'verifier_units': units, 'verifier_units_charged_cpu_s': round(sum(u['charged_cpu_s'] for u in units), 3),
           'verifier_agent_charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note', 'utc')} for e in led if e.get('event') == 'charge' and e.get('label') == 'verifier'],
           'labels_read': 'ATTACK_FIT and ATTACK_VALIDATION SEX through the logging loader (units VER_REFIT_S0_V1, VER_REFIT_S1_V1); DEV_REPORT truth once '
                          '(VER_DEVREPORT_V1); no MODEL_FIT, CALIBRATION or MODEL_SELECT label; sealed assessment not opened',
           'private_outputs': 'aggregate JSON and per-replicate endpoint draws under .private/pcrl_frozen_finalist_validation_v1/verifier/ (no truth file written)',
           'code_sha256': code}
    V.jdump(V.RES / 'INDEPENDENT_VERIFICATION.json', out)
    print(json.dumps(out['status'], indent=1))


if __name__ == '__main__':
    main()
