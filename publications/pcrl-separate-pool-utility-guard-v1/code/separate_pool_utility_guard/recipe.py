"""Frozen separate-pool utility guard recipe; import performs no numerical work.

One fixed five-arm, five-epoch recipe. These constants are not a search space.
Every arm shares the population-weighted UPDATE-minibatch + whole-GUARD task
objective, critics, random streams, step bound and checkpoints. Arms differ
only in the applied privacy gradient and the task acceptance safeguard.
"""
from __future__ import annotations
import hashlib
import json

from .wires import WIRE_CONTRACT, NEW_CAMPAIGN as CAMPAIGN

ARMS = ('SEPARATE_GUARD', 'GLOBAL_UPDATE_GUARD', 'OLD_BATCH_GUARD', 'ORDINARY', 'TASK_ONLY')
GLOBAL_GUARD_POOL = {'SEPARATE_GUARD': 'GUARD', 'GLOBAL_UPDATE_GUARD': 'UPDATE'}
SPLIT_SALT = 'SPG1|UPDATE-GUARD-split|20261013|'
PROBE_SALT = 'SPG1|UPDATE-wire-probe|20261013|'

RECIPE = {
    'schema': 'SPG1-separate-pool-utility-guard-recipe-v1',
    'campaign': CAMPAIGN,
    'source_family': 'RAW_JOINT', 'source_strength': 0.3, 'source_epoch': 5,
    'arms': list(ARMS),
    'epochs': 5, 'checkpoint_epochs': [0, 1, 3, 5], 'batch_size': 256,
    'proposals_per_epoch_rule': 'ceil(N_UPDATE/256), locked after the group split',
    'roles': {'continuation_pool_folds': [0, 1], 'validation_fold': 2,
              'split_salt': SPLIT_SALT, 'guard_fraction': 0.2,
              'guard_rule': 'first floor(0.2*eligible groups) in SHA256(salt+str(unit)) order; UPDATE is the remainder',
              'same_partition_both_seeds': True},
    'task_objective': '(N_UPDATE/N_TOTAL)*mean CE(UPDATE minibatch) + (N_GUARD/N_TOTAL)*mean CE(whole GUARD pool)',
    'task_objective_all_arms': True,
    'full_pool_chunk_rows': 1024,
    'full_pool_reduction': 'float64 sum of per-chunk row-sum losses in fixed row order, divided by pool rows',
    'optimizer': 'SGD_no_momentum_actual_displacement', 'learning_rate': 0.01,
    'maximum_actual_step_norm': 0.01,
    'critic_views': ['occupation_full29', 'aligned_pair_projected_income_full_occupation50'],
    'critic_kinds': ['linear', 'mlp32', 'mlp64'],
    'critic_learning_rate': 0.003, 'critic_optimizer': 'Adam',
    'critic_data': 'UPDATE rows and UPDATE SEX only',
    'warm_updates_per_critic': 80, 'critic_updates_per_model_step': 5,
    'critics_continue_across_model_rejection': True,
    'reference_rows': 256, 'reference_pool': 'UPDATE', 'reference_refresh': 'initial_and_each_epoch',
    'reference_absolute_scale_floor': 1e-12, 'reference_initial_relative_scale_floor': 1e-8,
    'normalization_draws': 16, 'normalization_target_ratio': 3.0,
    'normalization_task_gradient': 'shared UPDATE-minibatch plus whole-GUARD task objective',
    'normalization_min_median_norm': 1e-12, 'maximum_coefficient': 1e4,
    'coefficient_scope': 'one coefficient per source seed shared by all five arms',
    'sensitive_penalty': 'sum_views(1-min(critic_cross_entropies,sex_entropy)/sex_entropy)',
    'proposal': 'd=-0.01*(g_shared_task+lambda*g_sensitive); TASK_ONLY applies g_sensitive=0',
    'global_guard': {
        'arms': GLOBAL_GUARD_POOL,
        'anchor': 'exact starting metrics on the fixed guard pool; never reset',
        'budgets': {'nll_increase': 0.010, 'summed_brier_increase': 0.005, 'accuracy_loss': 0.010},
        'accuracy_rule': 'correct_count >= correct_count_0 - floor(0.010*N_pool + 1e-9)',
        'qp': 'min ||d-p||^2 s.t. g_NLL.d<=b_NLL, g_Brier.d<=b_Brier, ||d||<=0.01; exact KKT active-set enumeration',
        'slack_tolerance_rule': 'slack in [-1e-10,0) treated as 0 and logged; slack < -1e-10 is a real infeasible state',
        'finite_acceptance': 'recompute full-pool NLL, Brier, accuracy after actual mixed-dtype displacement; NLL/Brier <= budget + 1e-10',
    },
    'old_batch_guard': 'strict current-UPDATE-minibatch NLL/Brier nonincrease cone projection and finite check (predecessor rule)',
    'controls': {'ORDINARY': 'same objective/coefficient/step bound; no task acceptance',
                 'TASK_ONLY': 'zero applied privacy gradient; no task acceptance; critics still trained'},
    'task_guard_metrics': ['nll', 'summed_multiclass_brier', 'accuracy'],
    'nll_probability_floor': 1e-12, 'QP_tolerance': 1e-10,
    'finite_step_tolerance': 1e-10, 'backtrack_fractions': [1.0, 0.5, 0.25, 0.125],
    'actual_norm_repair_safety_fraction': 1e-6, 'actual_norm_repair_maximum_attempts': 16,
    'actual_norm_repair_scope': 'all_arms_without_task_metrics',
    'income_encoder_and_head': 'bitwise_frozen_every_critic_and_model_update',
    'income_basis': 'bitwise_frozen_every_critic_and_model_update',
    'wire_contract': WIRE_CONTRACT,
    'geometry_scope': 'inherited per-seed admitted start checkpoint head and basis bytes, unchanged',
    'privacy_movement': {
        'probe': 'first 512 UPDATE rows in SHA256(probe salt+str(row_id)) order',
        'probe_salt': PROBE_SALT, 'probe_rows': 512,
        'wire_tolerance': 'max(1e-10, 10*independently measured repeat-forward max abs noise), per coordinate block',
        'counterfactual_scope': 'SEPARATE_GUARD; first eight accepted nonzero updates per source seed, in update order',
        'counterfactual_rule': 'remove applied privacy gradient; identical task gradient, pool, budgets, solver and ladder; restore state',
        'contribution_tolerance': 'max(1e-10, 10*repeat-computation max abs displacement noise)',
    },
    'validation_gate': {
        'opened_after': 'TRAINING_COMPLETE_LOCK committed',
        'slice': 'inherited FIT fold 2 (VALIDATION)',
        'reference': 'unchanged starting RAW_HYBRID (epoch 0)',
        'caps': {'accuracy_loss': 0.010, 'nll_increase': 0.010, 'brier_increase': 0.005},
        'scope': 'both tasks and both source seeds',
        'selection': 'latest common saved epoch passing every incremental point proxy in both seeds; epoch 0 fallback',
        'separate_guard_pass_requires': ['chosen common epoch > 0', 'released occupation wire changed beyond frozen tolerance',
                                         'accepted nonzero privacy-bearing update evidence in both seeds',
                                         'all incremental validation point proxies pass in both seeds',
                                         'training, source and independent feasibility checks complete'],
    },
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_recipe(value):
    if value != RECIPE:
        raise PermissionError('trainer requires the exact one frozen recipe')
    return canonical_sha(RECIPE)


def validate_science_lock(lock, *, seed, checkpoint_sha256, geometry_sha256, role_split_sha256):
    """The closed predecessor lock or capability cannot authorize this trial."""
    if (lock.get('schema') != 'SPG1-science-lock-v1' or lock.get('campaign') != CAMPAIGN
            or lock.get('status') != 'DIRECT_TRIAL_AUTHORIZED' or lock.get('owner') != 'ROOT'
            or lock.get('INNER_truth_accessed') is not False
            or lock.get('VALIDATION_outcomes_opened') is not False
            or lock.get('recipe_sha256') != canonical_sha(RECIPE)
            or lock.get('role_split_sha256') != role_split_sha256
            or lock.get('starting_checkpoints', {}).get(str(seed)) != checkpoint_sha256
            or lock.get('geometry_bindings', {}).get(str(seed)) != geometry_sha256):
        raise PermissionError('separate-pool SCIENCE_LOCK capability required')
    return True
