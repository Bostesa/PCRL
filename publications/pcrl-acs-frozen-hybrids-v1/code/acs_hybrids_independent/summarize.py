"""Assemble results/pcrl_acs_frozen_hybrids_v1/independent/INDEPENDENT_VERIFICATION.json from the per-check JSON files."""
from __future__ import annotations

import argparse
import json

from . import vcommon as V

CHECKS = (('V1', 'V1_source_identities.json'), ('V2', 'V2_row_pairing_and_inheritance.json'), ('V3', 'V3_utility.json'),
          ('V4', 'V4_subgroups.json'), ('V5', 'V5_reader_selection.json'), ('V6', 'V6_refit_sample.json'),
          ('V7', 'V7_household_bootstrap.json'), ('V8', 'V8_endpoints_and_multiplicity.json'), ('V9', 'V9_decisions_and_labels.json'),
          ('V10', 'V10_deployment.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--concerns', default=None, help='JSON file with a list of concern strings')
    a = ap.parse_args()
    per, files = {}, {}
    for cid, fn in CHECKS:
        p = V.OUT / fn
        if not p.exists():
            per[cid] = {'status': 'MISSING'}
            continue
        d = V.jload(p)
        files[cid] = {'file': f'independent/{fn}', 'sha256': V.sha256_file(p)}
        per[cid] = {'status': d['status'], 'name': d['check']}
    supplements = {'V2': 'V2b_comparator_row_alignment_recompute.json', 'V8': 'V8b_multiplicity_labels.json'}
    for cid, fn in supplements.items():
        p = V.OUT / fn
        st = V.jload(p)['status'] if p.exists() else 'MISSING'
        files[cid + '_supplement'] = {'file': f'independent/{fn}', 'sha256': V.sha256_file(p) if p.exists() else None}
        per[cid]['supplement'] = {'file': fn, 'status': st}
        per[cid]['status_main_file'] = per[cid]['status']
        per[cid]['status'] = 'PASS' if (per[cid]['status'] == 'PASS' and st == 'PASS') else 'FAIL'
    v2b = V.jload(V.OUT / supplements['V2']) if (V.OUT / supplements['V2']).exists() else {}
    v8b = V.jload(V.OUT / supplements['V8']) if (V.OUT / supplements['V8']).exists() else {}
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = (V.jload(V.OUT / fn) if (V.OUT / fn).exists() else {} for _, fn in CHECKS)
    key = {}
    if v1:
        key['V1'] = {'published_docs_identical': f"{v1['A_inherited_documents_vs_published_commit']['identical']}/{v1['A_inherited_documents_vs_published_commit']['published_files']}",
                     'extra_inherited': v1['A_inherited_documents_vs_published_commit']['extra_inherited_files_not_in_published_commit'],
                     'restored_files_pinned_and_matching': f"{v1['B_restored_files_vs_published_hashes']['pinned_files'] - len(v1['B_restored_files_vs_published_hashes']['mismatched'])}/{v1['B_restored_files_vs_published_hashes']['restored_files']}",
                     'pins_checked': v1['B_restored_files_vs_published_hashes']['pins_checked']}
    if v2:
        key['V2'] = {s: {'bitwise_identity': f"{r['recipient_bitwise_identity']['identical']}/{r['recipient_bitwise_identity']['arrays']}",
                         'other_seed_equal': len(r['seed_provenance']['arrays_equal_to_other_seed_source']),
                         'complete_wires': r['complete_wires']} for s, r in v2['per_seed'].items()}
    if v2b:
        key['V2_supplement'] = {k: {'arrays_recomputed': r['arrays'], 'bitwise': r['bitwise'], 'max_abs_diff': r['max_abs_diff'],
                                    'all_decisions_equal': r['all_decisions_equal'],
                                    'TASK_REFIT_bitwise': f"{r['by_source_model']['TASK_REFIT']['bitwise']}/{r['by_source_model']['TASK_REFIT']['arrays']}",
                                    'NEURAL_RAW_bitwise': f"{r['by_source_model']['NEURAL_RAW_rho1.0_E2']['bitwise']}/{r['by_source_model']['NEURAL_RAW_rho1.0_E2']['arrays']}"}
                                for k, r in v2b['per_seed'].items()}
    if v8b:
        key['V8_supplement'] = {'endpoints_with_source_K40_status': v8b['endpoints_with_source_status'], 'label_failures': v8b['label_failures'],
                                'settled_clause_bounds': v8b['settled_clause_bounds']}
    if v3:
        key['V3'] = {'H_LJ_points_passing': f"{v3['H_LJ_points_passing']}/24", 'H_LJ_model_select_point_feasible_24': v3['H_LJ_model_select_point_feasible_24'],
                     'vs_campaign_max_abs_diff': v3['all_services_vs_campaign']['max_abs_diff'], 'vs_campaign_compared': v3['all_services_vs_campaign']['compared'],
                     'campaign_vs_source_points_exact': f"{v3['campaign_vs_source_published_points_exact']['bitwise_or_equal']}/{v3['campaign_vs_source_published_points_exact']['compared']}",
                     'campaign_vs_source_se_bitwise': f"{v3['campaign_vs_source_published_se']['bitwise_or_equal']}/{v3['campaign_vs_source_published_se']['compared']}"}
    if v4:
        key['V4'] = {k: {'point': g['point'], 'se': g['se'], 'U1': g['clause_U1'], 'K38_status': g['K38_status']} for k, g in v4['H_LJ_guards'].items()}
        key['V4']['households_per_sex'] = v4['households_per_sex']
    if v5:
        key['V5'] = {s: {'av_auc_max_abs_diff': r['attack_validation_auc_recomputed']['max_abs_diff_vs_READERS_SEED'],
                         'dev_report_replay_bitwise': f"{r['dev_report_score_replay']['bitwise']}/{r['dev_report_score_replay']['models']}",
                         'winner_disagreements': r['winner_disagreements'], 'fit_counts': r['inventory']['fit_counts_recomputed']['effective_numerical_fits']}
                     for s, r in v5['per_seed'].items()}
    if v6:
        key['V6'] = {k: {'reader_id': r['reader_id'], 'bitwise': r['refit_vs_stored_dev_report_scores']['bitwise'],
                         'max_abs_diff': r['refit_vs_stored_dev_report_scores']['max_abs_diff']} for k, r in v6['refits'].items()}
    if v7:
        key['V7'] = {'matrix_equals_group_weights_bitwise': v7['cross_check_task_headroom_group_weights']['bitwise_equal'], 'G': v7['G'], 'B': v7['B'],
                     'campaign_ses_reproduced': f"{v7['se_reproduction_all_campaign_ses']['compared'] - v7['se_reproduction_all_campaign_ses']['n_failures']}/{v7['se_reproduction_all_campaign_ses']['compared']}",
                     'se_max_abs_diff': v7['se_reproduction_all_campaign_ses']['max_abs_diff']}
    if v8:
        key['V8'] = {'critical_values': {k: v.get('rounded_4', v.get('value')) for k, v in v8['critical_values'].items() if isinstance(v, dict) and 'value' in v},
                     'endpoint_fields_compared': v8['endpoints_vs_DEV_REPORT_RESULTS']['compared'], 'endpoint_max_abs_diff': v8['endpoints_vs_DEV_REPORT_RESULTS']['max_abs_diff'],
                     'distance_rows_compared': v8['DISTANCES_csv']['compared'], 'distance_failures': v8['DISTANCES_csv']['n_failures'],
                     'claim_endpoints': {k: v for k, v in v8['primary'].items() if k.startswith(('RECIPE|H_LJ|coalition', 'COMPETITIVE', 'INCREMENT'))}}
    if v9:
        key['V9'] = {'controls_valid': v9['controls']['CONTROLS_VALID'], 'claim_A': v9['claim_A']['decision'], 'claim_B': v9['claim_B']['decision'],
                     'claim_B_nested': v9['claim_B']['nested_material_superiority_0.020'], 'claim_C': v9['claim_C']['decision'],
                     'overall': v9['conjunction']['overall'], 'unmet': v9['conjunction']['unmet_components_in_order'],
                     'confidence_status': v9['separate_status_fields']['overall_confidence_status']}
    if v10:
        key['V10'] = {s: {'parity': {t: v['bitwise'] for t, v in r['parity_vs_bank_H_LJ_DEV_REPORT'].items()},
                          'isolation': r['recipient_isolation']} for s, r in v10['per_seed'].items()}
    acc = [json.loads(l) for l in (V.PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl').read_text().splitlines() if l.strip()]
    mine = [e for e in acc if str(e.get('unit_id') or '').startswith('VERIFY_')]
    ledger = [json.loads(l) for l in (V.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
    ends = [e for e in ledger if e.get('event') == 'unit_end' and str(e['unit_id']).startswith('VERIFY_')]
    concerns = json.loads(open(a.concerns).read()) if a.concerns else []
    statuses = [v['status'] for v in per.values()]
    out = {'schema': 'ACSH1-independent-verification-v1', 'campaign': 'pcrl_acs_frozen_hybrids_v1',
           'role': 'independent verifier (own metric, AUC, bootstrap, endpoint, critical-value and decision code)',
           'per_check': per, 'files': files, 'key_numbers': key,
           'pass_count': statuses.count('PASS'), 'fail_count': statuses.count('FAIL'),
           'other_count': len(statuses) - statuses.count('PASS') - statuses.count('FAIL'),
           'refit_vs_replay_scope': {
               'refits_from_scratch': ['V6: the 5 declared pair winners (H_LJ s0, H_LJ s1, H_LL s0, TASK_REFIT s0, NEURAL_RAW s1) via fit_reader on ATTACK_FIT',
                                       'V9 support: the 12 control readers (2 seeds x 3 views x {shuffled null, planted leak}) for Stage-1 ATTACK_VALIDATION AUCs'],
               'replays_of_saved_models_or_stored_scores': ['V5: all 238 saved reader models (119 per seed) re-applied to ATTACK_VALIDATION (AUC, orientation, winners) and DEV_REPORT (scores)',
                                                            'V5: campaign stored DEV_REPORT scores vs restored source scores on 105 shared wires per seed',
                                                            'V8/V9: every DEV_REPORT AUC, SE and endpoint recomputed from the stored winner scores and control scores',
                                                            'V10: H_LJ packages released from raw permitted fields',
                                                            'V1/V2: hash and array identity replays'],
               'not_refit': 'of the 250 campaign reader fits, 17 were refit from scratch (5 declared pair winners + all 12 controls, which were never saved); the other 233 saved models were replayed, not refit'},
           'dev_report_truth_rereads_by_verifier': len(mine),
           'dev_report_truth_reread_events': [{'unit_id': e.get('unit_id'), 'utc': e.get('utc'), 'purpose': e.get('purpose')} for e in mine],
           'verifier_queue_units': [{'unit_id': e['unit_id'], 'status': e['status'], 'cpu_s': e['charged_cpu_s'], 'wall_s': e['wall_s']} for e in ends],
           'verifier_cpu_s_total': round(sum(float(e['charged_cpu_s']) for e in ends), 3),
           'concerns': concerns,
           'sealed_assessment_touched': False, 'raw_census_file_read': False, 'extractor_run': False}
    V.write_json(V.OUT / 'INDEPENDENT_VERIFICATION.json', out)
    print(json.dumps({'pass': out['pass_count'], 'fail': out['fail_count'], 'other': out['other_count'], 'per_check': per,
                      'rereads': out['dev_report_truth_rereads_by_verifier'], 'cpu_s': out['verifier_cpu_s_total']}, indent=1))


if __name__ == '__main__':
    main()
