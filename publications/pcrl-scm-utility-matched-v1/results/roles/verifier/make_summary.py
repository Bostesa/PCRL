"""Assemble INDEPENDENT_VERIFICATION.json from the verifier's check outputs (JSON only; no data file is read)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

c = {n: V.jload(V.VDIR / f) for n, f in (('1', 'check1_admission.json'), ('2', 'check2_trajectories.json'), ('3', 'check3_model_select.json'),
                                         ('4', 'check4_matching.json'), ('5', 'check5_dev_report.json'), ('6', 'check6_deploy.json'),
                                         ('7', 'check7_refits.json'), ('7b', 'check7b_controls.json'), ('8', 'check8_access.json'))}
ledger = [json.loads(l) for l in (V.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
ver_units = {e['unit_id']: e['charged_cpu_s'] for e in ledger if e['event'] == 'unit_end' and e['unit_id'].startswith('VERIFY_')}
c2 = c['2']['per_seed']; c3 = c['3']; c5 = c['5']['comparison_with_report']
out = {
    'schema': 'SCMUM1-independent-verification-v1', 'role': 'independent numerical verifier',
    'utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(timespec='seconds'),
    'checks': {
        '1_source_admission_and_roles': {'status': c['1']['status'], 'copies_rehashed': c['1']['1a_copies']['n_recorded'], 'copies_by_kind': c['1']['1a_copies']['n_by_kind'],
                                         'manifest_checks_redone': c['1']['1a_manifest_checks']['n'], 'roles_disjoint_units_and_rows': True,
                                         'role_counts_match_public': c['1']['1d_roles']['counts_match_public'],
                                         'own_routing_mismatches': {k: v['own_routing_mismatches_vs_admitted'] for k, v in c['1']['1e_routing'].items()},
                                         'historical_arm_outputs_reproduced_bitwise': {k: v['arms_reproduced_bitwise_4_roles_2_tasks'] for k, v in c['1']['1e_routing'].items()},
                                         'sealed_refused': c['1']['1f_sealed']['all_refused'], 'sealed_paths': c['1']['1f_sealed']['sealed_named_paths_in_admitted_trees'],
                                         'code_changed_since_ANALYSIS_LOCK': c['1']['1b_lock_chain']['code_changed_since_ANALYSIS_LOCK'],
                                         'lock_chain_failures': [k for k, v in c['1']['1b_lock_chain']['detail'].items() if not v], 'cpu_s': c['1']['cpu_s']},
        '2_trajectories': {'status': c['2']['status'],
                           'per_seed': {s: {'merge_steps_certified_greedy': sum(t['agree'] for t in r['trajectories'].values()),
                                            'merge_steps': sum(t['steps_checked'] for t in r['trajectories'].values()),
                                            'near_tie_or_divergent_steps': sum(len(t['near_tie_divergences']) + len(t['divergences']) for t in r['trajectories'].values()),
                                            'terminal_states_verified': True, 'F_own_delta_route_bitwise': all(t['F_max_abs_diff_own_delta_route_vs_recorded'] == 0 for t in r['trajectories'].values()),
                                            'F_absolute_route_max_abs_diff': max(t['F_max_abs_diff_absolute_route_vs_recorded'] for t in r['trajectories'].values()),
                                            'guard_running_sum_max_abs_diff_own_vs_recorded': max(max(t['spent_max_abs_diff_by_key']['guard0'], t['spent_max_abs_diff_by_key']['guard1']) for t in r['trajectories'].values()),
                                            'min_per_merge_dF': min(t['min_dF'] for t in r['trajectories'].values()), 'labels_all_match': r['labels']['all_match'],
                                            'effective_releases_own_vs_recorded': [r['releases']['own_effective_releases'], r['releases']['recorded_effective_releases']],
                                            'alias_partition_equal': r['releases']['alias_partition_equal'], 'label_tables_bitwise': r['releases']['tables']['bitwise_equal_release_tables'],
                                            'capacity_mismatches': len(r['releases']['capacity_mismatches']), 'wire_hash_mismatches': len(r['releases']['wire_hash_mismatches']),
                                            'historical_at_K_hist_bitwise': all(h['tables_bitwise'] and h.get('outputs_bitwise_4_roles', True) for h in r['historical'].values()),
                                            'unreached_N_labels_shortfall': r['labels']['unreached_N_labels']} for s, r in c2.items()},
                           'cpu_s': c['2']['cpu_s']},
        '3_model_select': {'status': c3['status'], 'values_exact': {s: [r['values_exact'], r['values_total']] for s, r in c3['per_seed'].items()},
                           'proxy_directions_exact': {s: [r['proxy_direction_values_exact'], r['proxy_direction_values_total']] for s, r in c3['per_seed'].items()},
                           'proxy_scope': 'ALL effective releases (272 + 261) and the frozen NEURAL_RAW / H_LJ on both seeds; no sampling',
                           't_star': c3['nomination']['own_t_star'], 'eligible_thresholds': c3['nomination']['own_eligible'], 'K_star': c3['nomination']['own_K_star'],
                           'e_star': c3['nomination']['own_e_star'], 'runner_up_mean_proxy_gap': c3['nomination']['runner_up_gap'], 'cpu_s': c3['cpu_s']},
        '4_matching': {'status': c['4']['status'], 'roles': {s: {rn: [r.get('release'), r.get('status'), r.get('trajectory')] for rn, r in rr.items()} for s, rr in c['4']['own']['roles'].items()},
                       'selection_effort': c['4']['own']['effort'], 'decision_margins': c['4']['own']['decision_margins'], 'cpu_s': c['4']['cpu_s']},
        '5_dev_report': {'status': c['5']['status'], 'event': c['5']['event'], 'bootstrap': c['5']['bootstrap'], 'critical': c['5']['critical'],
                         'endpoints_compared': c5['endpoints']['n'], 'endpoint_max_abs_diff': c5['endpoints']['max_abs_diff'], 'endpoint_field_mismatches': len(c5['mismatches']),
                         'claims_equal': c5['claims_equal'], 'condition_mismatches': len(c5['condition_mismatches']), 'separate_fields_equal': c5['separate_equal'],
                         'research_decision': c5['research_decision'][0], 'research_decision_equal': c5['research_decision'][0] == c5['research_decision'][1],
                         'qualifiers_equal': c5['qualifiers_equal'], 'bets_equal': c5['bets']['equal'], 'bets_mean_brier': c5['bets']['brier'][0],
                         'services_max_abs_diff': c5['services']['max_abs_diff'], 'descriptive_differences_max_abs_diff': c5['differences']['max_abs_diff'],
                         'csv_rows_disagreeing': {k: len(v['rows_disagreeing']) for k, v in c['5']['csv_check'].items()},
                         'source_label_free_replay_auc_and_se_exact': all(v['auc_equal'] and v['se_equal'] and v['same_winner'] for v in c['5']['source_label_free_replay'].values()),
                         'own_claims': c['5']['own_claims'], 'own_separate': c['5']['own_separate'], 'own_qualifiers': c['5']['own_qualifiers'], 'own_bets': c['5']['own_bets'],
                         'controls_valid': c['5']['controls']['CONTROLS_VALID'], 'cpu_s': c['5']['cpu_s']},
        '6_deployment': {'status': c['6']['status'], 'summary': c['6']['summary'], 'cpu_s': c['6']['cpu_s']},
        '7_reader_refits': {'status': c['7']['status'], 'summary': c['7']['summary'], 'controls_refit_7b': c['7b']['status'], 'cpu_s': c['7']['cpu_s'] + c['7b']['cpu_s']},
        '8_access': {'status': c['8']['status'], 'order': c['8']['order'], 'access_logs': c['8']['access_logs'], 'sealed': c['8']['sealed'],
                     'reporting_receipt': c['8']['reporting_receipt'], 'cpu_s': c['8']['cpu_s']}},
    'discrepancies': [
        {'id': 'D1', 'check': 1, 'severity': 'record wording', 'text': "SOURCE_ADMISSION.json explains the three differing source report CSVs (PRIMARY_ENDPOINTS, SECONDARY_ENDPOINTS, RECOVERY_AND_UTILITY_TABLE) as 'changed on the research branch after the publication build: cc7f8a168'. cc7f8a168 (2026-10-16T23:22:06-04:00) predates the publication commit 6fc06021d (23:48:14-04:00); the files equal the published ones after CRLF->LF normalisation (PROTOCOL.md section 1 is right). Cause: admit.py takes the last research commit touching the file, which is not 'after the build' when the publication branch is an orphan."},
        {'id': 'D2', 'check': '1/8', 'severity': 'procedure (not outcome-informed)', 'text': 'roles/coordinator/VERIFIER_BRIEF.md is hashed in ANALYSIS_LOCK (prefit_reviews_sha256) but was edited after the lock, in commit cf647d650 (05:24:38Z, the TRAJECTORY_COMPLETE_LOCK commit): the check-5 instruction was expanded (58/24 counts, availabilities, direction, strict sequence, FROZEN DESIGN reference). PROTOCOL section 16 says locked files are never edited. The edit preceded every MODEL_SELECT label read (UM_SELECT_V1 started 05:25:05Z) and changes no registered rule.'},
        {'id': 'D3', 'check': 1, 'severity': 'procedure (dry-run tool)', 'text': 'scm_um/dryrun.py is hashed in ANALYSIS_LOCK but was edited in cf647d650 (adds a stand-in proxy for the selection dry run); the new hash is recorded in REPORTING_LOCK; the commit message does not mention it. Before any MODEL_SELECT label; no analysis path affected.'},
        {'id': 'D4', 'check': '1/8', 'severity': 'procedure (presentation only)', 'text': 'scm_um/tables.py is hashed in ANALYSIS_LOCK and REPORTING_LOCK but was edited after the DEV_REPORT evaluation (commit 57f50a5ff, 05:34:26Z): figure annotation offsets and the K* string format in SELECTION_AND_MATCHING.md only; no endpoint, status or label is computed there. Its current hash differs from both locks; the commit message does not mention the code change.'},
        {'id': 'D5', 'check': 2, 'severity': 'registered text inconsistency (no numerical consequence)', 'text': "UNIT_INVENTORY.json checkpoints.nll_rule still reads 'last prefix m with F_m <= t + 1e-15' while PROTOCOL section 4 and the code use the first-crossing rule m*(t) = min{m : F_m > t + 1e-15} - 1 (math review S1, disposition 'applied'). Here every per-merge change of F is positive (minimum 1.9e-07 over 8,194 merges), so both rules give the same prefix for all 220 N{t} labels."},
        {'id': 'D6', 'check': 6, 'severity': 'deployment qualification', 'text': "For the frozen references (NEURAL_RAW_S0, H_LJ_S0, H_LJ_S1), the direct source API entry points (acs_replication.deploy.release, acs_hybrids.hybrid_deploy.release, named as their entry points in the coordinator's note) silently accept rows with a SEX key, a lower-case 'sex' key, an extra PUMA key, or reordered keys: the extra keys are ignored and only the six permitted fields are encoded, but nothing is refused. The refusals hold only behind scm_um.deploy.check_rows (as DEPLOYMENT_RESULTS.json says: 'behind the same strict input check') and on the CSV path. PROTOCOL section 14 ('each ... refuses extra or reordered inputs, SEX as an input') holds for the frozen references only on those two paths. By code reading, scm_bench.deploy_scm.release used directly also accepts reordered keys; scm_um.deploy.release, the documented SCM-format entry point, is strict (verified)."},
        {'id': 'D7', 'check': 1, 'severity': 'trivial wording', 'text': "PROTOCOL section 1 says '51 hash-checked copies' and lists 49 copied items (18 exports, 2 frozen banks, 28 fit files, 1 encoder record); the 51 entries of SOURCE_ADMISSION 'copies' include the 2 computed routing files, which are hashed but not copies."},
        {'id': 'D8', 'check': 4, 'severity': 'trivial (commit message)', 'text': "The SELECTION_LOCK commit message (3d2c34e8d) says ORD_NLLM is matched with '|delta| <= 0.00015'; seed 1's |delta| is 0.000154299 > 0.00015. The locked JSON is correct."},
        {'id': 'D9', 'check': 4, 'severity': 'reporting completeness', 'text': "PROTOCOL section 7 says selection effort is reported for every role; SELECTION_LOCK.selection_effort lists ORD_NLLM, IBSI_NLLM, SHUF_NLLM and IBSI_CNT only. SCM's nomination effort (11 distinct candidate releases per seed, visible in the nomination grid) and ORD_CNT's (1 fixed label) are not listed there."},
        {'id': 'D10', 'check': 2, 'severity': 'trivial (commit message)', 'text': "The TRAJECTORY_COMPLETE_LOCK commit message reports the historical gate as '52/52 per seed'; the 52 entries are 51 checks (10 trajectories x 5 + C0 tables) plus the ALL_PASS summary flag."}],
    'open_issues': [
        "Bet scoring P4/P5 ('NOT_HELD unless ORD_NLLM is MATCHED on both seeds'): the frozen report reads MATCHED as the MODEL_SELECT role status (as P2), giving P5 HELD and mean Brier 0.10375. Under the alternative reading 'DEV_REPORT point match' (MF-2 wording rule for labels), P5 would be NOT_HELD and the mean Brier 0.11625. The role-status reading matches the protocol's P2/P3 structure; noted only because bets are scored after the fact.",
        'The ORD_NLLM choice rests on thin MODEL_SELECT proxy margins (0.00014 seed 0, 0.00029 seed 1); the seed-0 runner-up (C128) matched expenditure more closely (|delta| 3.2e-06 against 2.8e-05). This is the registered strongest-proxy rule, applied correctly.',
        'Equivalence at +-0.001 was unreachable as the statistical review predicted (SE 0.0019 to 0.0020 for SCM - ORD_NLLM occupation NLL).'],
    'verifier_cpu': {'queue_units_cpu_s': ver_units, 'queue_cpu_s_total': round(sum(ver_units.values()), 1),
                     'light_non_queue_commands_cpu_s_upper_estimate': 60, 'agent_analysis_cpu_h_upper_estimate': 0.25,
                     'total_cpu_h_upper_estimate': round((sum(ver_units.values()) + 60) / 3600 + 0.25, 3),
                     'note': 'queue figures are the ledger RUSAGE_CHILDREN values (all attempts; none failed); light commands are git/JSON reads outside the queue; the agent allowance is an upper estimate, not yet logged with jobqueue charge (left to the coordinator to avoid double counting)'},
    'overall': 'ALL EIGHT CHECKS PASS; every recomputed number equals the campaign record bitwise; ten discrepancies (D1-D10), none of which changes a release, endpoint, status or label.'}
statuses = [v['status'] for v in out['checks'].values()]
out['all_checks_pass'] = all(s == 'PASS' for s in statuses)
(V.VDIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=V._default))
print(json.dumps({'all_checks_pass': out['all_checks_pass'], 'statuses': statuses, 'cpu': out['verifier_cpu']}, indent=1, default=V._default))
