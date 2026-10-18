"""Collect the verifier's per-check records (records/*.json) into roles/verifier/CHECKS.json (machine-readable summary; light).

Numerical status (PASS / FAIL / PARTIAL / PENDING) is kept separate from documentation observations; every unresolved item stays
listed. Aggregate numbers only (no person-level data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402


def load(name):
    p = V.RECORDS / f'{name}.json'
    return json.loads(p.read_text()) if p.is_file() else None


def main():
    notes_p = V.VER / 'records' / 'NOTES.json'
    notes = json.loads(notes_p.read_text()) if notes_p.is_file() else {}
    out = {'schema': 'OHN1-verifier-checks-v1', 'utc': V.utc(), 'campaign': V.CAMP,
           'code': {p.name: V.file_sha256(p) for p in sorted((V.VER / 'code').glob('*.py'))},
           'separation': 'numerical status is separate from documentation observations (doc_notes); every unresolved item stays visible',
           'checks': {}}
    s = load('VER_SOURCE')
    if s:
        b = s['1b_committed_records']
        out['checks']['1_source_and_roles'] = {
            'status': s['status'], 'unit_id': s['unit_id'], 'utc': s['utc'],
            'mirror_copies_rehashed': s['1a_mirror']['n_copies'], 'mirror_hash_mismatches': len(s['1a_mirror']['hash_mismatch_vs_admission']),
            'mirror_vs_original_source_mismatches': len(s['1a_mirror']['mirror_differs_from_original_source_file']),
            'committed_record_checks': {k: v for k, v in b.items() if k not in ('problems', 'copies_without_expected_hash')},
            'copies_without_committed_pin': {'n': b['copies_without_expected_hash']['n'],
                                             'pinned_elsewhere': b['copies_without_expected_hash']['hash_found_in_committed_source_records']},
            'roles': {k: [v['people'], v['households']] for k, v in s['1c_roles']['roles'].items()},
            'roles_disjoint': s['1c_roles']['all_disjoint'], 'devgrid_problems': s['1d_devgrid']['problems'],
            'crossfit_stored_vectors_equal': s['1e_crossfit']['all_stored_equal'], 'crossfit_n_stored': s['1e_crossfit']['n_stored'],
            'locks': {k: v for k, v in s['1f_locks'].items() if k in ('code_lock_files_unchanged', 'dev_grid_sha_equals_lock', 'registration_files_unchanged',
                                                                      'research_commit_is_ancestor', 'problems')}}
    a = load('VER_ACCESS')
    if a:
        out['checks']['2_exposure_and_access'] = {'status': a['status'], 'status_registered_rule': a.get('status_registered_rule'),
                                                  'status_amended_rule': a.get('status_amended_rule'),
                                                  'n_reads_outside_registered_entry_points': a.get('n_reads_outside_registered_entry_points'),
                                                  'amendment_record': 'records/ACCESS_CHECK_AMENDMENT.json',
                                                  'utc': a['utc'], 'issues': a['issues'], 'notes': a['notes'],
                                                  'log_lines': a['log_lines'], 'first_label_read_utc': a['first_label_read_utc'],
                                                  'receipt_verified_utc': a['receipt_verified_utc'],
                                                  'model_select_reads_by_kind': a['model_select_reads']['by_kind'],
                                                  'sex_reads_on_model_fit': [[r['unit_id'], r['stage']] for r in a['sex_reads_in_training_log']],
                                                  'n_release_records_checked': len(a['release_records']),
                                                  'mirror_access_dir_exists': a['mirror_access_dir_exists'],
                                                  'source_logs_unchanged': {k: v['unchanged'] for k, v in a['source_logs_vs_admission_snapshot'].items()}}
    u = load('VER_UTILITY_REFS')
    if u:
        out['checks']['3a_utility_references'] = {k: u[k] for k in ('status', 'unit_id', 'utc', 'max_abs_value_diff', 'max_abs_se_diff',
                                                                     'max_reserve_ratio_diff', 'all_point_pass_equal', 'all_reserve_met_equal',
                                                                     'all_wires_equal', 'max_side_diff')}
    c = load('VER_UTILITY_CONFIGS')
    if c:
        out['checks']['3b_utility_configurations'] = {
            'n_entries': c['n_entries'], 'n_pass': c['n_pass'],
            'status': 'PASS' if c['n_pass'] == c['n_entries'] else 'ATTENTION',
            'entries': {k: {'status': e['status'], 'max_abs_value_diff': e['max_abs_value_diff'], 'max_abs_se_diff': e['max_abs_se_diff'],
                            'reserve_ratio_diff': e.get('reserve_max_abs_ratio_diff'), 'my_min_ratio': e['my_min_ratio'], 'my_binding': e['my_binding'],
                            'crossfit': e.get('crossfit')} for k, e in c['entries'].items()}}
    r = load('VER_RESERVE')
    if r:
        out['checks']['4_reserve_witness_branch'] = {'utc': r['utc'], 'witness_ranking': r['witness_ranking'],
                                                     'selected_by_my_rule': r['selected_witnesses_by_my_rule'], 'income_branch': r['income_branch'],
                                                     'configs': {k: {kk: v.get(kk) for kk in ('status', 'min_ratio', 'binding', 'occ_side_min_ratio',
                                                                                             'witness_label', 'point_pass_all', 'bonferroni_unresolved')}
                                                                 for k, v in r['configs'].items()},
                                                     'witness_selection_compared': r.get('witness_selection_compared'),
                                                     'ledger_consistency': r.get('ledger_consistency')}
        wsc = r.get('witness_selection_compared')
        lc_ok = not (r.get('ledger_consistency') or {}).get('mismatches')
        out['checks']['4_reserve_witness_branch']['status'] = (
            ('PASS' if (wsc['agree'] and lc_ok) else 'ATTENTION') if wsc else ('PARTIAL (WITNESS_SELECTION.json not yet written)' if lc_ok else 'ATTENTION'))
    o = load('VER_OUTPUTS')
    if o:
        out['checks']['5a_outputs_contract'] = {'n': o['n_contract'], 'ok': o['n_contract_ok'],
                                                'status': 'PASS' if o['n_contract'] == o['n_contract_ok'] else 'ATTENTION',
                                                'argmax_ties': {k: v['argmax_ties_total'] for k, v in o['contract'].items() if v['argmax_ties_total']}}
    op = load('VER_OUTPUTS_PREDICT')
    if op:
        out['checks']['5b_outputs_predict'] = {'entries': {k: v['all_bitwise'] for k, v in op['predict'].items()},
                                               'status': 'PASS' if all(v['all_bitwise'] for v in op['predict'].values()) else 'ATTENTION'}
    g = load('VER_GRID_PAIRED')
    if g:
        out['checks']['6a_grid_paired_and_labels'] = {'status': g['status'], 'utc': g['utc'], 'max_abs_cell_value_diff': g['max_abs_cell_value_diff'],
                                                      'max_abs_cell_se_diff': g['max_abs_cell_se_diff'], 'summary_comparisons': g['summary_comparisons']}
    for p in sorted(V.RECORDS.glob('VER_GRID_CACHE*.json')):
        gc = json.loads(p.read_text())
        out['checks'][f'6b_grid_cache_{p.stem}'] = {'status': gc['status'], 'unit_id': gc['unit_id'], 'score_auc_max_abs_diff': gc['score_auc_max_abs_diff'],
                                                    'boot_max_abs_diff': gc['boot_max_abs_diff'], 'n_boot_vectors_checked': gc['n_boot_vectors_checked'],
                                                    'missing_or_mismatch': {b: [len(v['missing_cache']), len(v['winner_mismatch'])] for b, v in gc['banks'].items()}}
    gr = load('VER_GRID_REFIT')
    if gr:
        out['checks']['6c_grid_independent_refits'] = {
            'entries': {k: {'winners_same': {v: w['same_winner'] for v, w in e['winners'].items()},
                            'winner_score_auc_abs_diff': {v: w['score_auc_abs_diff'] for v, w in e['winners'].items()},
                            'max_val_auc_abs_diff': e['max_val_auc_abs_diff'], 'recipes_bitwise_equal': f"{e['recipes_bitwise_equal']}/{e['n_recipes']}",
                            'max_score_abs_diff': e['max_score_abs_diff']} for k, e in gr['entries'].items()}}
        allok = all(all(w['same_winner'] and w['score_auc_abs_diff'] < 1e-12 for w in e['winners'].values()) for e in gr['entries'].values())
        out['checks']['6c_grid_independent_refits']['status'] = 'PASS' if allok else 'ATTENTION'
        out['checks']['6c_grid_independent_refits']['n_entries'] = len(gr['entries'])
    for name in sorted(p.stem for p in V.RECORDS.glob('VER_*.json')):
        if name.startswith(('VER_SOURCE', 'VER_ACCESS', 'VER_UTILITY', 'VER_RESERVE', 'VER_OUTPUTS', 'VER_GRID')):
            continue
        x = load(name)
        out['checks'][name] = {k: x.get(k) for k in ('status', 'utc', 'unit_id', 'summary') if k in x}
        if name.startswith('VER_DEPLOY'):
            out['checks'][name] = {'status': 'PASS' if x.get('all_ok') else 'ATTENTION' if x.get('all_ok') is False else 'NO_PACKAGES',
                                   'root': x.get('root'), 'n_packages': len(x.get('packages', {})),
                                   'packages': {k: {kk: v.get(kk) for kk in ('anchor', 'occupation', 'seed', 'all_refused', 'own_encoder_equals_admitted_X')}
                                                | {'dependency_mismatches': len(v.get('dependency_mismatches', [])),
                                                   'runtime_data_files_not_in_manifest': v.get('runtime_data_files_not_in_manifest'),
                                                   'bitwise_all': all(all(t.values()) for t in v.get('bitwise_equal_to_stored_DEV_REPORT_release', {}).values())}
                                                for k, v in x.get('packages', {}).items()}}
    for extra in ('CHECK_CHANGES', 'ACCESS_CHECK_AMENDMENT'):
        x = load(extra)
        if x:
            out[extra.lower()] = x
    out['doc_notes'] = notes.get('doc_notes', [])
    out['unresolved'] = notes.get('unresolved', [])
    out['interim_messages_to_coordinator'] = notes.get('messages', [])
    V.write_record('CHECKS_SNAPSHOT', out)
    (V.VER / 'CHECKS.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=V._jdefault) + '\n')
    print(json.dumps({k: v.get('status') for k, v in out['checks'].items()}, indent=1))


if __name__ == '__main__':
    main()
