"""Assemble INDEPENDENT_VERIFICATION.json from unit outputs (JSON summaries only; no person-level data)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from .common import OUT, RES, tokenize

ROSTER = ('RAW_PROJ', 'T_BOTH', 'L_BOTH', 'S12', 'S21')


def label(*, gates_ok, nominee, eligible, util_all_pass, priv_eligible_all_pass, movement_ok, verification_complete):
    """Registered rule, re-implemented: gates -> nominee -> competitive (all 24 utility PASS, every eligible comparator's
    privacy endpoints PASS, movement, verification) -> full roster or incomplete; else point-feasible without criterion."""
    if not gates_ok:
        return 'NOT_ESTIMABLE_CONTROL_GATE'
    if not nominee:
        return 'NO_POINT_FEASIBLE_NOMINEE'
    competitive = util_all_pass and bool(eligible) and priv_eligible_all_pass and movement_ok and verification_complete
    if competitive:
        return 'TWO_RECIPIENT_COMPETITIVE_CRITERION_ESTABLISHED' if len(eligible) == len(ROSTER) else 'COMPETITIVE_CLAIM_INCOMPLETE_COMPARATOR_INELIGIBLE'
    return 'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION'


def main(units, disclosures):
    L = {k: json.loads((OUT / f'{u}.json').read_text()) for k, u in units.items() if k != 'FIX'}
    fx = json.loads((OUT / units['FIX']).read_text())
    pub = RES / 'independent'; pub.mkdir(exist_ok=True)
    for k, u in units.items():
        src = OUT / (u if k == 'FIX' else f'{u}.json')
        (pub / f'STAGE_{k}.json').write_text(json.dumps(tokenize(json.loads(src.read_text())), indent=2, sort_keys=True))
    checks = {}
    A = L['A']
    checks['A_zero_residual_parity_frozen_source_service_wires_contract'] = {
        'status': 'PASS' if A['pass'] else 'FAIL',
        'detail': {s: {'checkpoints_checked': r['checkpoints_checked'], 'epoch0_checkpoints_bitwise_RAW_PROJ_INNER_FIT2': all(all(all(v) for v in e['bitwise_raw_proj'].values()) and e['states_exactly_zero'] for e in r['epoch0_parity'].values()),
                       'services_bitwise_equal_stored_INNER_RELEASE': sum(all(v['bitwise_equal_stored_INNER_RELEASE']) for v in r['services'].values()),
                       'services': r['n_services'], 'contract_failures': sum(len(v['contract_failures']) for v in r['services'].values()),
                       'heads_bound_by_geometry': r['heads_match_geometry_binding'], 'basis_sha_match_binding': r['basis_sha_match_binding'],
                       'failures': r['failures']} for s, r in A['seeds'].items()}}
    B = L['B']
    checks['B_training_traces_budgets_freezing_accounting'] = {
        'status': 'PASS' if B['pass'] else 'FAIL',
        'detail': {s: {arm: {'failures': a['failures'], 'critic_rounds': a['critic_rounds'], 'opportunities': a['opportunities'],
                             'own_recomputed_violations': {n: a.get(f'{n}_own_recomputed_violations') for n in ('income', 'occupation')},
                             'own_vs_recorded_post_max_abs': {n: a.get(f'{n}_own_vs_recorded_post_max_abs') for n in ('income', 'occupation')}}
                       for arm, a in r['arms'].items()} for s, r in B['seeds'].items()}}
    C = L['C']
    mv = {s: r['movement'] for s, r in C['seeds'].items()}
    checks['C_counterfactual_resolve_own_solver'] = {
        'status': 'PASS' if C['pass'] else 'FAIL',
        'detail': {'states_reproduced': {s: f"{sum(x['reproduced'] for x in r['states'])}/{r['n_states']}" for s, r in C['seeds'].items()},
                   'max_privacy_displacement_diff': max(x['privacy_actual_max_abs_vs_stored'] for r in C['seeds'].values() for x in r['states']),
                   'max_counterfactual_displacement_diff': max(x['counterfactual_actual_max_abs_vs_stored'] for r in C['seeds'].values() for x in r['states']),
                   'movement': mv, 'movement_rule_met_all_recipient_seeds': C['movement_ok_all'],
                   'active_constraint_counts_own_privacy_solve': {f'S{s}|{n}': {
                       'states': sum(x['recipient'] == n for x in r['states']),
                       'nonempty_halfspace_active_set': sum(x['recipient'] == n and bool(x['privacy_active_set_own']) for x in r['states']),
                       'ladder_shortened': sum(x['recipient'] == n and (x['privacy_fraction_own'] or 1) < 1 for x in r['states']),
                       'ball_active': sum(x['recipient'] == n and bool(x['privacy_ball_active_own']) for x in r['states']),
                       'counterfactual_nonempty_active_set': sum(x['recipient'] == n and bool(x['counterfactual_active_set_own']) for x in r['states']),
                       'active_halfspace_ids_seen': sorted({j for x in r['states'] if x['recipient'] == n for j in x['privacy_active_set_own']})}
                       for s, r in C['seeds'].items() for n in ('income', 'occupation')},
                   'all_states_at_or_before_nominated_J_BOTH_E10': all(x['at_or_before_nominated_checkpoint'] for r in C['seeds'].values() for x in r['states']),
                   'campaign_scan_records_agree': sum(bool(x.get('campaign_record_agrees')) for r in C['seeds'].values() for x in r['states']),
                   'failures': [f for r in C['seeds'].values() for f in r['failures']]}}
    checks['C_solver_data_free_fixtures_vs_campaign_and_SLSQP'] = {'status': 'PASS' if fx['summary']['pass'] else 'FAIL', 'detail': fx['summary']}
    D = L['D']
    checks['D_fit2_24_clause_rule_and_selection'] = {'status': 'PASS' if D['pass'] else 'FAIL',
                                                     'detail': {'entries_recomputed': D['entries_recomputed'], 'max_abs_clause_value_diff': D['max_abs_clause_value_diff'],
                                                                'selection': {a: v['selected_label'] for a, v in D['selection'].items()}, 'mismatches': D['mismatches']}}
    E = L['E']
    checks['E_inner_gates_primary39_diagnostic45'] = {'status': 'PASS' if E['pass'] else 'FAIL',
                                                      'detail': {'control_gates_all_passed': E['control_gates_all_passed'], 'gate_value_max_abs_diff': E['control_gate_value_max_abs_diff'],
                                                                 'primary_K': E['primary']['K'], 'primary_max_abs_diff': E['primary']['max_abs_diff_vs_report'],
                                                                 'primary_status_mismatches': E['primary']['status_mismatches'], 'primary_status_counts': E['primary']['status_counts'],
                                                                 'diagnostic_K': E['diagnostic']['K'], 'diagnostic_max_abs_diff': E['diagnostic']['max_abs_diff_vs_report'],
                                                                 'diagnostic_status_mismatches': E['diagnostic']['status_mismatches'], 'diagnostic_status_counts': E['diagnostic']['status_counts'],
                                                                 'service_table_complete_auc_max_abs_diff': E['service_table_complete_auc_max_abs_diff']}}
    F = L['F']
    checks['F_reader_refits'] = {'status': 'PASS' if F['pass'] else 'FAIL',
                                 'detail': {'refits': F['reader_refits_used'], 'bitwise': sum(r['bitwise'] for r in F['refits']),
                                            'max_abs_diff': max(r['inner_score_max_abs_diff'] for r in F['refits']), 'failures': F['failures'],
                                            'readers': [f"S{r['seed']} {r['service']} {r['view']} {r['reader']}" for r in F['refits']]}}
    others_pass = all(c['status'] == 'PASS' for c in checks.values())
    sel = D['selection']
    nominee = sel['J_BOTH']['selected_label'] is not None
    eligible = [c for c in ROSTER if (D['RAW_PROJ_point_eligible'] if c == 'RAW_PROJ' else sel[c]['selected_label'] is not None)]
    lab = label(gates_ok=E['control_gates_all_passed'], nominee=nominee, eligible=eligible, util_all_pass=E['primary_utility_all_pass'],
                priv_eligible_all_pass=E['primary_privacy_eligible_all_pass'], movement_ok=C['movement_ok_all'], verification_complete=others_pass)
    checks['E_scientific_label_recomputed'] = {'status': 'PASS', 'detail': {
        'label': lab, 'inputs': {'gates_ok': E['control_gates_all_passed'], 'nominee_J_BOTH': sel['J_BOTH']['selected_label'],
                                 'comparators_point_eligible': eligible, 'primary_utility_all_pass': E['primary_utility_all_pass'],
                                 'primary_privacy_eligible_all_pass': E['primary_privacy_eligible_all_pass'],
                                 'movement_ok': C['movement_ok_all'], 'verification_complete_other_checks': others_pass}}}
    out = {'schema': 'TRA1-independent-verification-v1', 'checks': checks,
           'n_pass': sum(c['status'] == 'PASS' for c in checks.values()), 'n_fail': sum(c['status'] == 'FAIL' for c in checks.values()),
           'recomputed_scientific_label': lab, 'reader_refits_used': F['reader_refits_used'],
           'units': [u for k, u in units.items() if k != 'FIX'], 'disclosures': disclosures}
    (RES / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(tokenize(out), indent=2, sort_keys=True))
    return out


if __name__ == '__main__':
    cfg = json.loads(Path(sys.argv[1]).read_text())
    o = main(cfg['units'], cfg['disclosures'])
    print(o['n_pass'], o['n_fail'], o['recomputed_scientific_label'])
