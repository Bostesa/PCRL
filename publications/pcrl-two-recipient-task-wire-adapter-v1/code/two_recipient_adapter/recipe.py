"""Frozen two-recipient task-wire adapter recipe; import performs no numerical work.

One fixed six-arm bank per source seed. These constants are not a search space.
Every arm shares the residual architecture per recipient, source tensors,
projected wires, task supervision, per-recipient budgets (including the KL
anchor), critic types, initialization, proposal opportunities per active
recipient and checkpoint schedule. Simultaneous arms differ only in the applied
privacy objective (none / local / joint). Sequential and income-only arms apply
the joint objective restricted to the active recipient's parameters.
S12/S21 are learned sequential controls; they are not implementations of any
published finite-law sequential algorithm.
"""
from __future__ import annotations

from .adapter import ADAPTER_CONTRACT, CAMPAIGN, canonical_sha
from .guard import BUDGETS, BACKTRACK, QP_TOL, FINITE_TOL, NLL_FLOOR, CHUNK

ARMS = ('T_BOTH', 'L_BOTH', 'J_BOTH', 'S12', 'S21', 'I_ONLY')
ARM_RULES = {
    'T_BOTH': {'schedule': 'simultaneous', 'phases': [[0, 1]], 'privacy': None},
    'L_BOTH': {'schedule': 'simultaneous', 'phases': [[0, 1]], 'privacy': 'local'},
    'J_BOTH': {'schedule': 'simultaneous', 'phases': [[0, 1]], 'privacy': 'joint'},
    'S12': {'schedule': 'sequential', 'phases': [[0], [1]], 'privacy': 'joint'},
    'S21': {'schedule': 'sequential', 'phases': [[1], [0]], 'privacy': 'joint'},
    'I_ONLY': {'schedule': 'single', 'phases': [[0]], 'privacy': 'joint'},
}
CHECKPOINT_EPOCHS = (0, 1, 3, 5, 10)


def checkpoint_labels(arm):
    """Checkpoint keys in save order. Sequential arms save complete release pairs in both phases."""
    if ARM_RULES[arm]['schedule'] == 'sequential':
        return [f'P1E{e}' for e in CHECKPOINT_EPOCHS] + [f'P2E{e}' for e in CHECKPOINT_EPOCHS]
    return [f'E{e}' for e in CHECKPOINT_EPOCHS]


def selectable_labels(arm):
    """Positive checkpoints eligible for selection (sequential: second-phase only)."""
    if ARM_RULES[arm]['schedule'] == 'sequential':
        return [f'P2E{e}' for e in CHECKPOINT_EPOCHS if e > 0]
    return [f'E{e}' for e in CHECKPOINT_EPOCHS if e > 0]


def fallback_label(arm):
    return 'P2E10' if ARM_RULES[arm]['schedule'] == 'sequential' else 'E10'


REFERENCE_SERVICES = ('RAW_PROJ', 'O_PREVIOUS')
HYBRIDS = {'HYB_JI_RO': 'J_BOTH audited income residual + RAW_PROJ (zero) occupation',
           'HYB_RI_JO': 'RAW_PROJ (zero) income + J_BOTH audited occupation residual'}

RECIPE = {
    'schema': 'TRA1-two-recipient-task-wire-adapter-recipe-v1',
    'campaign': CAMPAIGN,
    'source_family': 'RAW_JOINT', 'source_strength': 0.3, 'source_epoch': 5,
    'arms': list(ARMS), 'arm_rules': ARM_RULES,
    'trainable': 'per recipient R_i (rank_i x 64) and b_i (rank_i) only; every source tensor, both heads, bases, class orders and preprocessing frozen',
    'initialization': 'R_i = 0, b_i = 0 exactly (zero-residual witness = RAW_PROJ for both recipients)',
    'adapter_contract': ADAPTER_CONTRACT,
    'epochs_per_active_recipient': 10, 'proposals_per_epoch': 32, 'proposal_opportunities_per_active_recipient': 320,
    'checkpoint_epochs': list(CHECKPOINT_EPOCHS), 'batch_size': 256,
    'simultaneous_rule': 'both proposals computed from the same pre-update state (same minibatch rows, same critic state); each recipient projected/accepted on its own separable budgets; applied together as one recorded macro-step',
    'sequential_rule': 'first recipient receives 320 opportunities (10 epochs) while the second stays at its zero residual; the first is then frozen bitwise and the second receives 320 opportunities against the fixed first release; anchors are never reset',
    'single_rule': 'I_ONLY adapts income for 320 opportunities; occupation stays the exact RAW_PROJ zero residual',
    'roles': {'task_pool': 'inherited UPDATE (8180) + GUARD (2043) = 10223 rows; FIT folds 0/1',
              'task_labels': 'income and occupation labels for UPDATE and GUARD (new authorized export)',
              'critic_data': 'UPDATE rows and UPDATE SEX only (GUARD has no SEX export)',
              'excluded_from_training': ['FIT fold 2 (FIT2)', 'ATT', 'INNER']},
    'task_objective': 'per task: (8180/10223)*mean CE(current UPDATE minibatch) + (2043/10223)*mean CE(whole GUARD pool)',
    'optimizer': 'SGD_no_momentum_actual_displacement', 'learning_rate': 0.01, 'maximum_actual_step_norm_per_recipient': 0.01,
    'proposal': 'p_i = -0.01*(g_task_i + lambda_i*g_privacy_i); T_BOTH applies g_privacy = 0',
    'critic_views': ['income_full21', 'occupation_full29', 'aligned_pair_income21_occupation29'],
    'critic_kinds': ['linear', 'mlp32', 'mlp64'],
    'critic_learning_rate': 0.003, 'critic_optimizer': 'Adam', 'warm_updates_per_critic': 80,
    'critic_rounds_per_recipient_opportunity': 5,
    'critic_rounds_per_step': {'simultaneous': 10, 'sequential': 5, 'single': 5},
    'critic_round': 'one Adam step for each of the 9 critics on one fresh UPDATE minibatch of the current complete wires',
    'critics_continue_across_rejection_and_phases': True,
    'expected_critic_rounds': {'simultaneous': 80 + 10 * 320, 'sequential': 80 + 5 * 640, 'single': 80 + 5 * 320},
    'reference_rows': 256, 'reference_pool': 'UPDATE', 'reference_refresh': 'initial and at each epoch start (each phase)',
    'reference_scale_floors': 'inherited DiagonalReference: max(raw std, max(initial*1e-8, 1e-12))',
    'sensitive_risk_per_view': '1 - min(critic CEs, SEX entropy)/SEX entropy (best-of-bank, entropy normalized)',
    'privacy_objectives': {'joint': 'risk(income21) + risk(occupation29) + risk(pair50)',
                           'local': 'risk(income21) + risk(occupation29)'},
    'per_recipient_gradient': 'gradient of the arm objective w.r.t. that recipient residual only (joint: own view + pair; local: own view)',
    'normalization': {'draws': 16, 'target_ratio': 3.0, 'statistic': 'median gradient norm over draws at the matched warm zero-residual state',
                      'per_recipient': True,
                      'task_gradient': 'shared UPDATE-minibatch plus whole-GUARD task objective of that recipient',
                      'coefficients': 'lambda_J_i from the joint objective (J_BOTH, S12, S21, I_ONLY); lambda_L_i from the local objective (L_BOTH); T_BOTH none',
                      'unavailable_if': 'median nonfinite or <=1e-12, or coefficient > 1e4; recorded, no search',
                      'disclosure': 'equal target ratio does not imply equal force; achieved per-step strength ratios are logged'},
    'maximum_coefficient': 1e4, 'normalization_min_median_norm': 1e-12,
    'budgets': {'pool': 'full 10223-row task pool, per recipient', 'nll_increase': BUDGETS['nll'],
                'summed_brier_increase': BUDGETS['brier'],
                'correct_rule': 'correct_i >= start_correct_i - floor(0.005*N)',
                'kl_limit': BUDGETS['kl'], 'kl': 'mean_x KL(p0_i(x)||p_theta_i(x)) with immutable zero-residual p0_i',
                'anchor': 'exact zero-residual start of each recipient; never reset (including at the sequential phase boundary)',
                'disclosure': 'empirical fitting constraints, not held-out utility guarantees; KL initial gradient is zero and never chooses a coefficient'},
    'qp': 'per recipient: min ||d-p||^2 s.t. g_NLL.d<=s_NLL, g_Brier.d<=s_Brier, g_K.d<=s_K, ||d||<=0.01; exact KKT active-set enumeration',
    'slack_rule': 'slack < -1e-10 is an infeasible current state (zero step); [-1e-10,0) resolved to 0 and logged',
    'finite_acceptance': 'recompute that recipient full-pool NLL, Brier, correct and K after real update; NLL/Brier/K <= limit + 1e-10',
    'backtrack_fractions': list(BACKTRACK), 'QP_tolerance': QP_TOL, 'finite_step_tolerance': FINITE_TOL,
    'nll_probability_floor': NLL_FLOOR, 'full_pool_chunk_rows': CHUNK,
    'full_pool_reduction': 'float64 sum of per-chunk row-sum losses in fixed row order, divided by pool rows',
    'frozen_checks': 'all source tensors bitwise after every step; inactive recipient residual bitwise unchanged after every step',
    'privacy_movement': {
        'probe': 'inherited fixed 512-row UPDATE probe (label-independent SHA256 order; same rows both seeds)',
        'wire_tolerance': 'per recipient: max(1e-10, 10*measured repeat-forward max abs noise) per coordinate block; any decision change counts',
        'counterfactual_scope': 'J_BOTH first eight accepted nonzero updates per recipient per seed, in update order',
        'counterfactual_rule': 'at saved pre-step state remove that recipient applied privacy gradient; identical task gradient, pool metrics/gradients, budgets, solver and ladder',
        'active_budget_states': 'J_BOTH: first 16 accepted updates per recipient whose projection had an active halfspace or whose ladder shortened, with their privacy-removed re-solve',
        'contribution_tolerance': 'max(1e-10, 10*repeat-proposal max abs noise)',
        'meaningful_rule': 'for each seed and recipient, at least 6 of the 8 counterfactuals change the deployed probe wire beyond tolerance and the displacement beyond noise; all at or before the nominated (or fixed epoch-10) checkpoint',
    },
    'no_sweeps': 'no lambda, KL, temperature, capacity, trust, head architecture, mask, calibration, controller or refitted-critic family; no extra epochs for rejected steps',
}


def validate_recipe(value):
    if value != RECIPE:
        raise PermissionError('trainer requires the exact one frozen recipe')
    return canonical_sha(RECIPE)


def validate_science_lock(lock, *, seed, checkpoint_sha256, geometry_sha256, adapter_binding_sha256):
    if (lock.get('schema') != 'TRA1-science-lock-v1' or lock.get('campaign') != CAMPAIGN
            or lock.get('status') != 'BANK_FITS_AUTHORIZED' or lock.get('owner') != 'ROOT'
            or lock.get('INNER_truth_accessed') is not False or lock.get('FIT2_outcomes_opened_by_campaign') is not False
            or lock.get('recipe_sha256') != canonical_sha(RECIPE)
            or lock.get('starting_checkpoints', {}).get(str(seed)) != checkpoint_sha256
            or lock.get('geometry_bindings', {}).get(str(seed)) != geometry_sha256
            or lock.get('adapter_bindings', {}).get(str(seed)) != adapter_binding_sha256):
        raise PermissionError('two-recipient task-wire adapter SCIENCE_LOCK capability required')
    return True
