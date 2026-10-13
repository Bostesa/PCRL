"""Frozen occupation repair contract; import performs no numerical work.

Only the coordinator may authorize a protection-stage worker. These constants
are one recipe, never a search space. The actual-displacement solver and real
finite-step guard are exported from R2's implementation, lazily on request.
"""
from __future__ import annotations
import hashlib
import json

ARMS = ('CONSTRAINED', 'ORDINARY', 'LOCAL_TWIN', 'TASK_ONLY')
RECIPE = {
    'schema': 'OB1-occupation-bounded-SGD-recipe-v1',
    'source_family': 'RAW_JOINT', 'source_strength': 0.3, 'source_epoch': 5,
    'epochs': 20, 'checkpoint_epochs': [0, 5, 10, 20], 'batch_size': 256,
    'update_folds': [0, 1], 'selection_fold': 2,
    'optimizer': 'SGD_no_momentum_actual_displacement', 'learning_rate': 0.01,
    'maximum_actual_step_norm': 0.01,
    'critic_views': ['occupation_full29', 'aligned_pair_full50'],
    'critic_kinds': ['linear', 'mlp32', 'mlp64'],
    'critic_learning_rate': 0.003, 'critic_optimizer': 'Adam',
    'warm_updates_per_critic': 80, 'critic_updates_per_model_step': 5,
    'reference_rows': 256, 'reference_refresh': 'initial_and_each_epoch',
    'reference_absolute_scale_floor': 1e-12,
    'reference_initial_relative_scale_floor': 1e-8,
    'normalization_draws': 16, 'normalization_target_ratio': 3.0,
    'normalization_min_median_norm': 1e-12, 'maximum_coefficient': 1e4,
    'task_guard_metrics': ['nll', 'summed_multiclass_brier'],
    'nll_probability_floor': 1e-12, 'QP_tolerance': 1e-10,
    'finite_step_tolerance': 1e-10, 'backtrack_fractions': [1.0, 0.5, 0.25, 0.125],
    'actual_norm_repair_safety_fraction': 1e-6,
    'actual_norm_repair_maximum_attempts': 16,
    'actual_norm_repair_scope': 'all_arms_without_task_metrics',
    'income_encoder_and_head': 'bitwise_frozen_every_update',
    'sensitive_penalty': 'sum_views(1-min(critic_cross_entropies,sex_entropy)/sex_entropy)',
    'coefficient_scope': 'same_full_local_plus_pair_coefficient_all_arms_within_source_seed',
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_recipe(value):
    if value != RECIPE:
        raise PermissionError('pilot requires the exact one frozen recipe')
    return canonical_sha(RECIPE)


def displacement_api():
    """No optimizer state: constraints apply to proposed SGD displacement.

    min_d .5||d-p||², a_NLL.d<=0, a_Brier.d<=0, ||d||<=.01.
    p=-.01(g_CE+lambda*g_sensitive). Sensitive penalties are minimized.
    Zero is feasible. Applying d and reading back mixed-dtype parameters is
    required before both gradient products and finite-batch metrics certify it.
    """
    from .gradient_worker import project_displacement, finite_step
    return project_displacement, finite_step
