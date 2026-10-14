"""Merge per-seed Stage A unit outputs (JSON only, no data) into detail + public PASS/FAIL summary."""
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PRIV = W / '.private/minimal_task_wire_adapter'
RES = W / 'results/pcrl_minimal_task_wire_adapter_v1/independent'
ARMS = ('J_TRUST', 'J_FREE', 'L_TRUST', 'L_FREE', 'TASK_TRUST')
EPOCHS = (1, 3, 5, 10)
TOL_EXACT = 1e-12
TOL_RESOLVE = 1e-10


def main(units):
    d = {s: json.loads((PRIV / f'independent/{u}.json').read_text()) for s, u in zip((0, 1), units)}
    checks = {}

    def add(name, ok, **nums):
        checks[name] = {'status': 'PASS' if ok else 'FAIL', **nums}
    for s, r in d.items():
        assert not r['errors'], r['errors']
        a1, a2, a3, a4 = (r['parts'][k] for k in ('A1', 'A2', 'A3', 'A4'))
        # A1
        occ, inc = a1['occupation'], a1['income']
        add(f'A1_S{s}_own_SVD_rowspace_equals_bound', occ['projector_maxabs_diff'] <= TOL_EXACT and inc['projector_maxabs_diff'] <= TOL_EXACT
            and occ['my_rank'] == occ['bound_rank'] == 5 and inc['my_rank'] == inc['bound_rank'] == 1,
            occ_rank=occ['my_rank'], inc_rank=inc['my_rank'], occ_projector_maxabs=occ['projector_maxabs_diff'],
            inc_projector_maxabs=inc['projector_maxabs_diff'], occ_sv5_over_sv6=occ['rank_gap_ratio'], occ_sv=occ['my_singular_values'],
            A_BtB_minus_A=occ['A_BtB_minus_A_maxabs'])
        add(f'A1_S{s}_rank_param_count_scale', a1['trainable_matches_binding'] and a1['scale_matches_binding'],
            trainable_parameters=a1['trainable_parameters'], scale=a1['hidden']['scale'],
            hidden_vs_admission_maxabs=a1['hidden_vs_admission_maxabs'], admission_sv_maxabs=a1['admission_sv_maxabs'])
        add(f'A1_S{s}_zero_residual_witness_bitwise', all(v['bitwise'] for v in a1['zero_residual_witness'].values()),
            roles=sorted(a1['zero_residual_witness']))
        e0 = [a1['checkpoints'][a]['0'] for a in ARMS]
        add(f'A1_S{s}_epoch0_equals_my_RAW_PROJ', all(x['R_b_all_zero'] and all(v['bitwise'] for v in x['epoch0_vs_my_RAW_PROJ'].values()) for x in e0),
            arms=len(e0), roles=['UPDATE', 'GUARD', 'VALIDATION'])
        allck = [v for a in ARMS for k, v in a1['checkpoints'][a].items() if isinstance(v, dict)]
        add(f'A1_S{s}_frozen_income_and_bases_all_checkpoints', all(v['income_basis_bitwise'] and v['occupation_basis_bitwise'] and v['income_wire_validation_bitwise']
                                                                  and v['source_sha_ok'] and v['arm_epoch_ok'] and v['rank'] == 5 and v['scale'] == 1.0 for v in allck),
            checkpoints=len(allck))
        diffkeys = {tuple(a1['checkpoints'][a]['keys_differing_from_epoch0']) for a in ARMS}
        add(f'A1_S{s}_only_R_b_trainable_differ', diffkeys == {('R', 'b_res', 'critic_state', 'epoch', 'trainer_metadata')},
            keys_differing=sorted(diffkeys)[0], note='critic_state/epoch/trainer_metadata are bookkeeping, not deployed parameters')
        add(f'A1_S{s}_probe_order', r['probe_position_sha256_matches_split'])
        # A2
        add(f'A2_S{s}_start_pool_metrics', a2['start_vs_seed_summary_maxabs'] <= TOL_EXACT and a2['start_correct_equal'],
            maxabs=a2['start_vs_seed_summary_maxabs'], start=r['my_start_pool'])
        for arm in ARMS:
            x = a2[arm]
            ok = (x['checkpoint_metric_maxabs_vs_summary'] <= TOL_EXACT and x['checkpoint_correct_mismatches'] == 0 and x['checkpoints_equal_trace']
                  and x['current_metric_maxabs_vs_log'] <= TOL_EXACT and x['current_correct_mismatches'] == 0
                  and x['post_metric_maxabs_vs_log'] <= TOL_EXACT and x['post_correct_mismatches'] == 0
                  and not x['accepted_budget_violations_mine'] and x['accepted_step_norm_over_0.01'] == 0 and x['nonaccepted_steps_that_moved'] == 0
                  and x['status_counts_match'] and x['budget_trace_limits_constant'] and x['anchor_vs_mine_maxabs'] <= TOL_EXACT
                  and x['anchor_correct_floor_equal'] and x['anchor_trust'] == (arm != 'J_FREE' and arm != 'L_FREE')
                  and x['raw_slack_lengths'] == ([3] if x['anchor_trust'] else [2]) and x['trace0_zero'])
            add(f'A2_S{s}_{arm}_metrics_and_budget_audit', ok,
                checkpoint_metric_maxabs=x['checkpoint_metric_maxabs_vs_summary'], every_state_metric_maxabs=max(x['current_metric_maxabs_vs_log'], x['post_metric_maxabs_vs_log']),
                status_counts=x['status_counts_mine'], max_accepted_kl=x['max_accepted_kl'], max_accepted_nll=x['max_accepted_nll'],
                max_accepted_brier=x['max_accepted_brier'], min_accepted_correct=x['min_accepted_correct'], limits=x['budget_trace_limits'],
                kl_constrained=x['anchor_trust'])
        # A3
        st = a3['states']
        mx = lambda f: max(f(x) for x in st)
        resolve_ok = all(x['pre_equals_trace_n_minus_1'] and x['coefficient_matches_summary'] and x['current_correct_equal']
                         and x['counterfactual']['status_matches'] and x['counterfactual']['fraction_matches'] and x['probe_decision_changes_match']
                         and x['privacy']['status'] == 'ACCEPTED' for x in st)
        nums = {'states': len(st),
                'g_task_maxabs': mx(lambda x: x['g_task_maxabs']), 'g_privacy_maxabs': mx(lambda x: x['g_privacy_maxabs']),
                'pool_gradient_maxabs': mx(lambda x: max(x['pool_gradient_maxabs'].values())), 'guard_ce_gradient_maxabs': mx(lambda x: x['guard_ce_gradient_maxabs']),
                'proposal_maxabs': mx(lambda x: x['proposal_maxabs']),
                'privacy_displacement_maxabs_vs_stored': mx(lambda x: x['privacy']['actual_maxabs_vs_stored']),
                'privacy_displacement_maxabs_vs_trace': mx(lambda x: x['privacy']['actual_maxabs_vs_trace']),
                'counterfactual_displacement_maxabs_vs_stored': mx(lambda x: x['counterfactual']['actual_maxabs_vs_stored']),
                'probe_wire_maxabs_vs_stored': mx(lambda x: max(x['probe_wire_vs_stored_maxabs'].values())),
                'kkt_worst': mx(lambda x: max(x[l]['kkt'][k] for l in ('privacy', 'counterfactual') for k in ('primal_max_violation', 'complementarity_max', 'stationarity_max'))),
                'slsqp_vs_exact_maxabs': mx(lambda x: max(x[l]['slsqp_maxabs_vs_exact'] for l in ('privacy', 'counterfactual'))),
                'active_halfspaces': sorted({(x['update_number'], l, tuple(x[l]['kkt']['active'])) for x in st for l in ('privacy', 'counterfactual') if x[l]['kkt']['active']}),
                'ball_active_updates': [x['update_number'] for x in st if x['privacy']['kkt']['ball_multiplier'] > 0],
                'fractions': sorted({x[l]['fraction'] for x in st for l in ('privacy', 'counterfactual')})}
        numeric_ok = all(nums[k] <= TOL_RESOLVE for k in ('g_task_maxabs', 'g_privacy_maxabs', 'pool_gradient_maxabs', 'guard_ce_gradient_maxabs', 'proposal_maxabs',
                                                          'privacy_displacement_maxabs_vs_stored', 'privacy_displacement_maxabs_vs_trace',
                                                          'counterfactual_displacement_maxabs_vs_stored', 'probe_wire_maxabs_vs_stored', 'kkt_worst'))
        add(f'A3_S{s}_counterfactual_resolve_matches', resolve_ok and numeric_ok and len(st) == 8, **nums)
        add(f'A3_S{s}_privacy_changes_displacement_and_wire', all(x['privacy_changes_displacement_beyond_1e-10'] and x['privacy_changes_probe_wire_beyond_1e-10'] for x in st),
            min_privacy_minus_cf_maxabs=min(x['privacy_minus_cf_maxabs'] for x in st),
            min_probe_repr_diff=min(x['probe_wire_privacy_vs_cf']['representation'] for x in st),
            probe_decision_changes=[x['probe_wire_privacy_vs_cf']['decision_changes'] for x in st])
        # A4 per seed
        add(f'A4_S{s}_clause_values_match_campaign', a4['clause_value_maxabs_vs_campaign'] <= TOL_EXACT and a4['pass_flag_mismatches'] == 0
            and all(v['n_clauses'] == 12 and v['all_pass'] == v['campaign_all_pass'] for v in a4['table'].values()),
            services=len(a4['table']), clause_value_maxabs=a4['clause_value_maxabs_vs_campaign'])
    # A4 selection across seeds
    tab = {k: v for s in d for k, v in d[s]['parts']['A4']['table'].items()}
    camp = d[0]['parts']['A4']['campaign_selection']
    sel = {}
    for arm in ARMS:
        feas = [e for e in EPOCHS if all(tab[f'{s}|{arm}|{e}']['all_pass'] for s in (0, 1))]
        sel[arm] = {'feasible': feas, 'selected': max(feas) if feas else None,
                    'campaign_feasible': camp[arm]['feasible_common_epochs_gt0'], 'campaign_selected': camp[arm]['selected_epoch']}
    add('A4_selection_rule_reapplied', all(v['feasible'] == v['campaign_feasible'] and v['selected'] == v['campaign_selected'] for v in sel.values())
        and all(tab[f'{s}|RAW_PROJ|0']['all_pass'] for s in (0, 1)), selection=sel,
        failing=[{'service': k, 'failed': [(c['task'], c['reference'], c['metric'], c['value']) for c in v['failed']]} for k, v in tab.items() if not v['all_pass']])
    import numpy, scipy, sklearn, torch, platform
    versions = {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                'torch': torch.__version__, 'sklearn': sklearn.__version__}
    overall = 'PASS' if all(c['status'] == 'PASS' for c in checks.values()) else 'FAIL'
    public = {'schema': 'MTAV-stage-A-v1', 'campaign': 'pcrl_minimal_task_wire_adapter_v1', 'role': 'independent verification',
              'overall': overall, 'checks': checks, 'versions': versions, 'queue_units': list(units),
              'tolerances': {'exact_forward_metrics': TOL_EXACT, 'resolve': TOL_RESOLVE, 'budget_finite': 1e-10},
              'independence': 'own forward/metrics/critics/projection in minimal_task_wire_adapter_independent (core.py); only minimal_task_wire_adapter.evaluation_queue/scheduler imported for queue authentication; no campaign runner/scoring/selection/solver imported',
              'scope': {'A1': 'both seeds; own raw83 forward on UPDATE(8180)/GUARD(2043)/VALIDATION(5211) features; own SVD rank (relative cutoff 1e-8); 25 checkpoints per seed',
                        'A2': 'both seeds x 5 arms: metrics at 5 checkpoints (TASK_POOL/UPDATE/GUARD) and at all 321 STATE_TRACE states vs UPDATE_LOG current/post metrics; budgets with own start anchor',
                        'A3': 'all 8 saved J_TRUST counterfactual states per seed (updates 1-8): g_task, g_privacy (own critic forward), pool gradients, own projection (NNLS dual + ball bisection; SLSQP cross-check), own ladder, probe wire (512 rows)',
                        'A4': 'all 24 FIT2 point clauses (12 per seed) for RAW_PROJ and 5 arms x 5 epochs; latest-common-epoch>0 selection re-applied'},
              'notes': ['In all 16 re-solved states no budget halfspace was active; the ball was active only where noted, so the re-solve exercises the unconstrained/ball branches of the solver, not active halfspaces.',
                        'FREE arms are not KL-constrained by design; their KL exceeds 0.001 (see max_accepted_kl).'],
              'private_detail': '$PRIV/independent/STAGE_A_DETAIL.json'}
    (PRIV / 'independent/STAGE_A_DETAIL.json').write_text(json.dumps({'public': public, 'per_seed_unit_outputs': d}, indent=1, sort_keys=True))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / 'STAGE_A.json').write_text(json.dumps(public, indent=1, sort_keys=True))
    print(overall)
    for k, v in checks.items():
        if v['status'] != 'PASS':
            print('FAIL', k, json.dumps(v)[:600])


if __name__ == '__main__':
    main(sys.argv[1:3])
