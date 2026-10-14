"""Frozen minimal task-wire adapter recipe; import performs no numerical work.

One fixed five-arm bank. These constants are not a search space. Every arm
shares the residual architecture, source tensors, projected wires, labels,
batches, task objective, budgets, normalization procedure, proposal
opportunities, critic bank/updates and checkpoint schedule. Arms differ only
in the applied privacy objective (joint, local, none) and the teacher-function
trust constraint.
"""
from __future__ import annotations

from .adapter import ADAPTER_CONTRACT, CAMPAIGN, canonical_sha
from .guard import BUDGETS, BACKTRACK, QP_TOL, FINITE_TOL, NLL_FLOOR, CHUNK

ARMS = ('J_TRUST', 'J_FREE', 'L_TRUST', 'L_FREE', 'TASK_TRUST')
ARM_RULES = {'J_TRUST': {'privacy': 'joint', 'trust': True},
             'J_FREE': {'privacy': 'joint', 'trust': False},
             'L_TRUST': {'privacy': 'local', 'trust': True},
             'L_FREE': {'privacy': 'local', 'trust': False},
             'TASK_TRUST': {'privacy': None, 'trust': True}}
REFERENCE_SERVICES = ('RAW_PROJ', 'RAW_HYBRID')
PRIOR_GUARD_DIAGNOSTICS = ('PRIOR_SEPARATE_GUARD_E5', 'PRIOR_GLOBAL_UPDATE_GUARD_E5', 'PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ')

RECIPE = {
    'schema': 'MTA1-minimal-task-wire-adapter-recipe-v1',
    'campaign': CAMPAIGN,
    'source_family': 'RAW_JOINT', 'source_strength': 0.3, 'source_epoch': 5,
    'arms': list(ARMS), 'arm_rules': ARM_RULES,
    'trainable': 'R (rank x 64) and b_res (rank) only; every source tensor, both heads, bases, class orders and preprocessing frozen',
    'initialization': 'R = 0, b_res = 0 exactly (zero-residual witness = RAW_PROJ)',
    'adapter_contract': ADAPTER_CONTRACT,
    'epochs': 10, 'proposals_per_epoch': 32, 'checkpoint_epochs': [0, 1, 3, 5, 10], 'batch_size': 256,
    'roles': {'task_pool': 'inherited UPDATE (8180) + GUARD (2043) = 10223 rows; FIT folds 0/1',
              'critic_data': 'UPDATE rows and UPDATE SEX only (GUARD has no SEX export)',
              'excluded_from_training': ['FIT fold 2', 'ATT', 'INNER']},
    'task_objective': '(8180/10223)*mean CE(current UPDATE minibatch) + (2043/10223)*mean CE(whole GUARD pool)',
    'task_objective_all_arms': True,
    'optimizer': 'SGD_no_momentum_actual_displacement', 'learning_rate': 0.01, 'maximum_actual_step_norm': 0.01,
    'proposal': 'p = -0.01*(g_task + lambda_arm*g_privacy_arm); TASK_TRUST applies g_privacy = 0',
    'critic_views': ['occupation_full29', 'aligned_pair_projected_income21_occupation29'],
    'critic_kinds': ['linear', 'mlp32', 'mlp64'],
    'critic_learning_rate': 0.003, 'critic_optimizer': 'Adam', 'warm_updates_per_critic': 80,
    'critic_updates_per_model_step': 5, 'critics_continue_across_model_rejection': True,
    'critic_bank_all_arms': True,
    'reference_rows': 256, 'reference_pool': 'UPDATE', 'reference_refresh': 'initial_and_each_epoch',
    'reference_scale_floors': 'inherited DiagonalReference: max(raw std, max(initial*1e-8, 1e-12))',
    'sensitive_risk_per_view': '1 - min(critic CEs, SEX entropy)/SEX entropy',
    'privacy_objectives': {'joint': 'risk(occupation29) + risk(pair50)', 'local': 'risk(occupation29)'},
    'normalization': {'draws': 16, 'target_ratio': 3.0, 'statistic': 'median gradient norm over draws at matched warm zero-residual state',
                      'task_gradient': 'shared UPDATE-minibatch plus whole-GUARD task objective',
                      'coefficients': 'lambda_J from joint risk shared by J_TRUST/J_FREE; lambda_L from local risk shared by L_TRUST/L_FREE; TASK_TRUST none',
                      'unavailable_if': 'median nonfinite or <=1e-12, or coefficient > 1e4; recorded, no search'},
    'maximum_coefficient': 1e4, 'normalization_min_median_norm': 1e-12,
    'budgets': {'pool': 'full 10223-row task pool', 'nll_increase': BUDGETS['nll'],
                'summed_brier_increase': BUDGETS['brier'],
                'correct_rule': 'correct >= start_correct - floor(0.005*N)',
                'anchor': 'exact zero-residual start; never reset',
                'disclosure': 'outcome-informed conservative internal development budgets, not the original criterion or a bound'},
    'trust': {'arms': [a for a in ARMS if ARM_RULES[a]['trust']], 'K': 'mean_x KL(p0(x)||p_theta(x)) on full task pool',
              'limit': BUDGETS['kl'], 'p0': 'immutable zero-residual start distribution',
              'free_arms': 'omit KL halfspace and KL acceptance; K logged as diagnostic'},
    'qp': 'min ||d-p||^2 s.t. g_NLL.d<=s_NLL, g_Brier.d<=s_Brier, [g_K.d<=s_K for TRUST], ||d||<=0.01; exact KKT active-set enumeration',
    'slack_rule': 'slack < -1e-10 is an infeasible current state (zero step); [-1e-10,0) resolved to 0 and logged',
    'finite_acceptance': 'recompute full-pool NLL, Brier, correct (and K for TRUST) after real update; NLL/Brier/K <= limit + 1e-10',
    'backtrack_fractions': list(BACKTRACK), 'QP_tolerance': QP_TOL, 'finite_step_tolerance': FINITE_TOL,
    'nll_probability_floor': NLL_FLOOR, 'full_pool_chunk_rows': CHUNK,
    'full_pool_reduction': 'float64 sum of per-chunk row-sum losses in fixed row order, divided by pool rows',
    'income': 'bitwise frozen factorized RAW income projection; checked after every update and in every wire',
    'privacy_movement': {
        'probe': 'inherited fixed 512-row UPDATE probe (label-independent SHA256 order; same rows both seeds)',
        'wire_tolerance': 'max(1e-10, 10*measured repeat-forward max abs noise) per coordinate block; any decision change counts',
        'counterfactual_scope': 'J_TRUST first eight accepted nonzero updates per seed, in update order',
        'counterfactual_rule': 'at saved pre-step state remove applied privacy gradient; identical task gradient, pool metrics/gradients, budgets, solver and ladder',
        'contribution_tolerance': 'max(1e-10, 10*repeat-proposal max abs noise)',
        'saved_state': 'pre-step R/b, minibatch rows, critic state, references, gradients for verifier re-solve',
    },
    'no_sweeps': 'no lambda, KL, temperature, capacity, head architecture, mask, calibration, controller or refitted-critic family',
}


def validate_recipe(value):
    if value != RECIPE:
        raise PermissionError('trainer requires the exact one frozen recipe')
    return canonical_sha(RECIPE)


def validate_science_lock(lock, *, seed, checkpoint_sha256, geometry_sha256, adapter_binding_sha256):
    if (lock.get('schema') != 'MTA1-science-lock-v1' or lock.get('campaign') != CAMPAIGN
            or lock.get('status') != 'BANK_FITS_AUTHORIZED' or lock.get('owner') != 'ROOT'
            or lock.get('INNER_truth_accessed') is not False or lock.get('FIT2_outcomes_opened_by_campaign') is not False
            or lock.get('recipe_sha256') != canonical_sha(RECIPE)
            or lock.get('starting_checkpoints', {}).get(str(seed)) != checkpoint_sha256
            or lock.get('geometry_bindings', {}).get(str(seed)) != geometry_sha256
            or lock.get('adapter_bindings', {}).get(str(seed)) != adapter_binding_sha256):
        raise PermissionError('minimal task-wire adapter SCIENCE_LOCK capability required')
    return True
