"""Stage C public summary and consolidated INDEPENDENT_VERIFICATION.json (JSON only)."""
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PRIV = W / '.private/minimal_task_wire_adapter'
RES = W / 'results/pcrl_minimal_task_wire_adapter_v1'
TOL = 1e-10  # admitted wire tolerance (ADMISSION wire_tolerance); decisions must be identical


def main(unit):
    d = json.loads((PRIV / f'independent/{unit}.json').read_text())
    assert 'error' not in d, d['error']
    r = d['restore']; rep = d['replay']
    checks = {}
    checks['C1_restore_from_snapshot_alone'] = {
        'status': 'PASS' if (r['files_in_manifest'] == r['manifest_file_count'] == 1301 and not r['source_hash_failures'] and not r['restored_hash_failures']
                             and not r['size_mismatches'] and r['bytes_restored'] == r['manifest_total_bytes'] and d['restore_copy_deleted']) else 'FAIL',
        'files': r['files_in_manifest'], 'bytes': r['bytes_restored'], 'hash_failures': len(r['source_hash_failures']) + len(r['restored_hash_failures']),
        'restore_copy_deleted': d['restore_copy_deleted'], 'manifest_sha256': '806ada7cd044965b18e9cd49a1f5cee84b69333836beb7602127a7d0c5091b8e'}
    worst = max(max(v[k]['r1_maxabs'], v[k]['r2_maxabs']) for v in rep.values() for k in ('expected', 'reader_release'))
    ok = (len(rep) == 10 and worst <= TOL and all(v['package_hashes_ok'] and v['feature_names_ok'] and v['package_input_equals_snapshot_input'] is True
                                                  and v['reader_release_row_ids_match_input'] and v['expected']['decisions_equal'] and v['reader_release']['decisions_equal']
                                                  and v.get('raw_proj_zero_adapter_bitwise', True) for v in rep.values()))
    checks['C2_raw_input_deployment_replay'] = {
        'status': 'PASS' if ok else 'FAIL', 'packages': sorted(rep), 'max_abs_wire_diff': worst, 'tolerance': TOL,
        'bitwise': 'not bitwise: own head uses z@W.T+b (matmul) instead of torch Linear/addmm; differences at 1 ulp level, all decisions identical',
        'per_package': {k: {'kind': v['kind'], 'vs_expected': v['expected'], 'vs_reader_release': v['reader_release']} for k, v in rep.items()}}
    import numpy, scipy, sklearn, torch, platform
    versions = {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__}
    pub_c = {'schema': 'MTAV-stage-C-v1', 'campaign': 'pcrl_minimal_task_wire_adapter_v1', 'queue_unit': unit,
             'overall': 'PASS' if all(c['status'] == 'PASS' for c in checks.values()) else 'FAIL', 'checks': checks, 'versions': versions,
             'scope': 'all 1301 snapshot files copied by manifest into a fresh $PRIV/independent/restore_<unit>/ inside the queue unit, re-hashed, then 10 deployment packages (J_TRUST_E10, TASK_TRUST_E10, RAW_PROJ, L_TRUST_E10, PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ x 2 seeds) replayed from restored inputs/INNER_features.npz (2235 raw83 rows) with own forward; restore copy deleted',
             'not_imported': 'minimal_task_wire_adapter.deployment'}
    (RES / 'independent/STAGE_C.json').write_text(json.dumps(pub_c, indent=1, sort_keys=True))
    A = json.loads((RES / 'independent/STAGE_A.json').read_text()); B = json.loads((RES / 'independent/STAGE_B.json').read_text())
    allc = {**{k: v['status'] for k, v in A['checks'].items()}, **{k: v['status'] for k, v in B['checks'].items()}, **{k: v['status'] for k, v in checks.items()}}
    a3 = [A['checks'][f'A3_S{s}_counterfactual_resolve_matches'] for s in (0, 1)]
    lab = B['checks']['B3_scientific_label']
    consolidated = {
        'schema': 'MTAV-independent-verification-v1', 'campaign': 'pcrl_minimal_task_wire_adapter_v1', 'role': 'independent verification',
        'overall': 'PASS' if all(v == 'PASS' for v in allc.values()) else 'FAIL',
        'checks': allc, 'stage_files': ['independent/STAGE_A.json', 'independent/STAGE_B.json', 'independent/STAGE_C.json'],
        'independent_scientific_label': lab['independent_label'], 'campaign_scientific_label': lab['campaign_label'],
        'nominee_condition_privacy_driven_update': {'verified': lab['privacy_driven_update_verified_each_seed'],
                                                    'how': 'all 8 saved J_TRUST counterfactual states per seed (updates 1-8, epoch 1) re-solved; privacy changed the accepted displacement (min max-abs diff S0 %.2e, S1 %.2e) and the probe wire beyond 1e-10' % (
                                                        A['checks']['A3_S0_privacy_changes_displacement_and_wire']['min_privacy_minus_cf_maxabs'],
                                                        A['checks']['A3_S1_privacy_changes_displacement_and_wire']['min_privacy_minus_cf_maxabs'])},
        're_solved': ['A3: 16 J_TRUST privacy/no-privacy projections (own gradients from role data + saved critic state, own exact projection solver with KKT check and SLSQP cross-check, own ladder); max displacement diff vs stored %.1e' % max(x['privacy_displacement_maxabs_vs_stored'] for x in a3),
                      'A1: own SVD row space / rank / scale; own raw83 forward on UPDATE/GUARD/VALIDATION',
                      'A2: own pool metrics at all 50 checkpoints and all 3210 trace states (10 arms x 321)',
                      'A4: 24 FIT2 clauses for every arm/epoch + selection rule',
                      'B1: 18 independent reader refits (<=36)',
                      'B2: K=39 primary, K=45 diagnostic, control gates, service-table AUCs from frozen INNER scores/releases with own AUC/bootstrap/intervals',
                      'B3: scientific label and queue accounting'],
        'replayed_not_resolved': ['C: deployment wires replayed from raw inputs after restore from snapshot alone (no retraining)',
                                  'B2 uses the campaign\'s frozen INNER reader scores (only 18 of 972 reader fits were independently refit in B1)',
                                  'training trajectories themselves were audited (A2) but not re-run from scratch; only the 16 counterfactual steps were re-solved'],
        'tolerances': {'forward/metrics exact': 1e-12, 're-solve': 1e-10, 'wire replay': TOL, 'AUC refit': 1e-9, 'budgets finite step': 1e-10, 'decisions': 'exact'},
        'notes': ['In the 16 re-solved states no budget halfspace was active (ball active once, S0 update 6); the solver re-solve therefore exercises unconstrained and ball branches only.',
                  'Scientific label rule text is not formalised in SCIENCE_LOCK; the reading applied is recorded in STAGE_B.json B3_scientific_label.rule_reading.',
                  'Failed/superseded verification units: MTAV_B2_V1 (config key collision with queue guard), MTAV_C_V1 (manifest entry format misread; restore aborted before any comparison, copy deleted). Both rerun as _V2.',
                  'B1 refit units ran stage_b.py before an edit that only renamed a report-path config key (refit code unchanged).'],
        'versions': versions,
        'queue_units': A['queue_units'] + B['queue_units'] + [unit],
        'code': 'minimal_task_wire_adapter_independent/ (core.py, stage_a.py, stage_b.py, stage_c.py, merge_a.py, merge_b.py, merge_c.py)'}
    (RES / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(consolidated, indent=1, sort_keys=True))
    print(pub_c['overall'], consolidated['overall'])


if __name__ == '__main__':
    main(sys.argv[1])
