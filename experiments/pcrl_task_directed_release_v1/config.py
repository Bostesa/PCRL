"""Outcome-independent configuration and complete nominal release ledger."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

STUDY = 'pcrl_task_directed_release_v1'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / STUDY
PRIMARY = ('A/SEX', 'A/RAC1P', 'AB/SEX', 'AB/RAC1P')
SECONDARY = ('A/public_coverage', 'A/commute_over20', 'B/income_binary',
             'B/civilian_at_work', 'B/same_residence', 'B/SEX', 'B/RAC1P')
UTILITY = ('A/same_residence', 'A/income_binary', 'A/civilian_at_work',
           'B/public_coverage', 'B/commute_over20')
INPUTS = ('T0', 'Ttask', 'Trisk')
POSITIVE_BUDGETS = (.0005, .002, .01)
PINNED = {
    'replacement': 'e3415b94deb8d71c4870d4c392bb8ad4b464a847',
    'stochastic': 'cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a',
    'manuscript': '0176f149e91c02b8e2d202eb25ea9cba8ae019dc',
    'utility_pilot': 'ba531ab424c593fe8573cd320d2bcef379907cf1',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def mechanism_id(input_code, policy, budget, actions=17):
    b = 'unconstrained' if budget is None else format(budget, '.4g')
    return f'{input_code}_{policy}_{b}_a{actions}'


def configuration():
    maps = []
    for code in INPUTS:
        for policy in ('L', 'C'):
            for budget in (*POSITIVE_BUDGETS, 0.):
                for anchor in (0, 1, 2):
                    cid = mechanism_id(code, policy, budget)
                    maps.append({'id': f'{cid}/anchor_{anchor}', 'configuration': cid,
                                 'input': code, 'policy': policy, 'budget': budget,
                                 'anchor': anchor, 'max_actions': 17})
        for anchor in (0, 1, 2):
            cid = mechanism_id(code, 'U', None)
            maps.append({'id': f'{cid}/anchor_{anchor}', 'configuration': cid,
                         'input': code, 'policy': 'U', 'budget': None,
                         'anchor': anchor, 'max_actions': 17})
    return {
        'study': STUDY, 'version': 1, 'pinned_shas': PINNED,
        'started_utc': '2026-09-21T22:26:05Z',
        'science_deadline_utc': '2026-09-22T07:26:05Z',
        'absolute_deadline_utc': '2026-09-22T08:26:05Z',
        'target_useful_hours': 8, 'incremental_usd_ceiling': 50,
        'compute_network_target_usd': 35, 'threads': 1,
        'physical_memory_worker_ceiling_gib': 3.5,
        'seeds': [0, 1, 2], 'maps': maps,
        'roles': {'primary': PRIMARY, 'secondary': SECONDARY, 'utility': UTILITY},
        'report_weightings': ['unweighted', 'PWGTP'],
        'fit_weighting': '0.5 + 0.5 * PWGTP / mean(PWGTP), normalized per original person',
        'selection_rule': 'minimum 0.5*unweighted_CE + 0.5*PWGTP_CE, then lexical candidate id; epoch ties earliest',
        'probability_clip': [1e-5, 1-1e-5],
        'split': {'teacher_fraction_RF': .60, 'teacher_internal_fit_fraction': .80,
                  'mechanism_fraction_RF': .40,
                  'salt': 'pcrl_task_directed_release_v1/households/v1',
                  'probe_and_attacker_fit': ['downstream_fit', 'attacker_fit'],
                  'utility_and_mechanism_validation': 'downstream_validation',
                  'attacker_validation': 'attacker_validation',
                  'current_run_evaluation': 'test'},
        'teacher_slate': {'logistic_C': [.03, .3, 3.], 'mlp_hidden': [64,32],
                          'mlp_weight_decay': [0., .0001], 'mlp_epochs': 200,
                          'batch_original_people': 256, 'lr': .001,
                          'validation_interval': 5, 'refit_after_internal_selection': False},
        'risk_model': {'family': 'logistic', 'C': 1., 'max_iter': 2000,
                       'schema': {'SEX': 2, 'RAC1P': 9},
                       'missing_training_class_probability_floor': 1e-8},
        'code': {'coarse_quantiles': 32, 'refined_children': 2,
                 'tie_projection': 'cos(1..32)/sqrt(sum(cos(1..32)^2)) on permitted PCA32',
                 'risk_cell_min_fit': 40, 'risk_kmeans_n_init': 1,
                 'risk_initialization': 'farthest pair initialized by min/max fixed projection; canonical lexicographic centers',
                 'absent_child_rule': 'unweighted mechanism-frequency weighted supported sibling Q; absent parent zero-offset action'},
        'actions': {'primary_max': 17, 'expanded_max': 33,
                    'rule': '16 (32 expanded) residual-quantile groups, balanced-weight soft-label CE stationary roots plus exactly zero',
                    'offset_bound': 12., 'duplicate_tolerance': 1e-10},
        'conditioning': {'local': 'KMeans2 on H_A[:,[1,3]], train features only',
                         'coalition': 'local cell crossed with median H_B[:,1]',
                         'robustness': 'intersect primary with KMeans4 and same H_B split'},
        'audit': {'slate': 'catchup', 'logistic_C': 1.,
                  'full_class_probability_floor': 1e-9,
                  'mlp_hidden': [64,32], 'mlp_epochs': [120,360],
                  'mlp_lr': .001, 'mlp_weight_decay': .0001,
                  'histgb_iterations': 150, 'histgb_leaves': 15,
                  'histgb_min_original_people': [20,5],
                  'histgb_l2': 1., 'exact_expansion': True,
                  'supplementary_histgb': 'same two trees on one fixed sampled token per original person; exact expected validation; preserves native leaf-count semantics',
                  'sampled_tree_seed': '20260921 + supplied anchor/role seed',
                  'token_interactions': 'onehot and H x onehot for logistic; H plus onehot for nonlinear learners',
                  'ancestor_candidates': 'all compatible H-only and actual A/B singleton predictions; no J ancestor for replacement'},
        'withholding_publish_probability': [.25,.5,.75],
        'randomized_response_deterministic_mix': [.25,.5,.75],
        'inference': {'alpha': .05, 'bootstrap_replicates': 10000, 'seed': 20260921,
                      'resampling': 'multinomial common household multiplicities across all arms and anchors',
                      'estimator': 'equal mean of three anchor estimates; actual weighted ratios per replicate',
                      'correction': 'Bonferroni two-sided normal bounds from household bootstrap SE; alpha/(2*len(CONTRASTS.endpoints))',
                      'identical_pair_variance': 'exactly zero',
                      'validation_screens': 'point estimates',
                      'evaluation_claims': 'adjusted one-sided bounds represented by simultaneous two-sided intervals',
                      'utility_material_gain': .003, 'utility_allowance': .001,
                      'sensitive_allowance': .001,
                      'historical_utility_gain': .01,
                      'historical_half_headroom': 'descriptive separate inherited criterion'},
        'selection': {
            'utility_first': 'screen all eight J-relative sensitive increments <=.001 and both utility deltas <=-.003; choose smallest mean balanced task CE then config id',
            'protection_first': 'screen both J utility deltas<=.001, all eight sensitive deltas<=.001 and at least one<0; minimize maximum sensitive delta then balanced task CE then config id',
            'one_common_configuration': True,
            'family_UF_nominee': 'best balanced utility among family settings passing all eight J privacy caps; tie lexical id',
            'family_PF_nominee': 'minimum maximum eight J sensitive deltas among settings passing both J utility allowances; tie utility then id',
            'no_eligible_control': 'report entire attempted frontier; empty set cannot establish competitive superiority',
            'if_no_screen_passes': 'freeze nearest descriptive nominee per route by lexicographic violation then route objective; no scientific pass'},
        'recovery_branches': {
            'A': 'after mandatory T0 audits, expanded33 if any input family unconstrained fixed-action validation balanced CE minus frozen teacher CE >.001; paired L/C and all codes priority middle, low, high budgets across all anchors; original retained',
            'B': 'Ttask and Trisk run regardless of T0 outcome under resource schedule',
            'C': 'optional finer conditioning if numerically feasible Q has validation sensitive increment over H > max(.003,2*budget) on any primary weighting; choose middle-budget Trisk C family in advance; both L/C all budgets/anchors',
            'D': 'bounded two compatible solvers plus one deterministic corrected-input retry; quarantine integrity failures; no budget/role weakening',
            'E': 'validation screen pass prioritizes symmetric replay and fresh attacks; 2016 remains sealed',
            'F': 'complete registered schedule even without candidate; diagnose exact limiting stage'},
    }


def release_ledger(cfg=None):
    cfg = cfg or configuration()
    records = [{**m, 'kind': 'finite_map', 'status': 'registered',
                'audit_roles': list(PRIMARY+SECONDARY), 'utility_roles': list(UTILITY)}
               for m in cfg['maps']]
    names = ['H', 'J', 'leace_A0', 'splince_A0', 'optnet16_L1', 'optnet16_L2',
             'optnet16_C1', 'continuous_task', 'constant_best', 'independent_token',
             'leace_supervised', 'splince_supervised']
    for code in INPUTS:
        names += [f'{code}_code']
        names += [f'{code}_withhold_{p:g}' for p in cfg['withholding_publish_probability']]
        names += [f'{code}_rr_{p:g}' for p in cfg['randomized_response_deterministic_mix']]
    for name in names:
        for anchor in cfg['seeds']:
            records.append({'id': f'{name}/anchor_{anchor}', 'configuration': name,
                            'anchor': anchor, 'kind': 'control', 'status': 'registered',
                            'audit_roles': list(PRIMARY+SECONDARY), 'utility_roles': list(UTILITY)})
    return {'nominal_primary_Q_maps': 81, 'records': records,
            'audit_unit_definition': 'release/anchor/role; identical B and exact duplicate releases may reuse verified objects',
            'nominal_release_anchor_records': len(records)}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    for name,value in [('CONFIGS.json',cfg),('ALL_RELEASES.json',release_ledger(cfg))]:
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
