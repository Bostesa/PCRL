"""Build the repaired interface maps from the frozen 2018 spectral objects.

Nothing historical is refitted. ``build_roles`` is imported unchanged from the
predecessor study, so ``V``, ``U``, the polynomial bases, the trace-normalised moment
Grams and the fifteen fitted out-of-fold nuisance models are bit-for-bit the objects
the completed studies used. This module only chooses new ``W`` maps under the
repaired objective.

Only the **rotation_invariant** family is fitted here. The matched-rank ``original``
controls already exist (at rank 16 they *are* the historical ``spectral_L1/L2/C1``;
at rank 8 they are the predecessor's ``spectral_lin8_*``), and the defective
``nonlinear`` controls exist in ``results/pcrl_nonlinear_rank_v1/``. Both are reused,
never refitted. The closed-form original solution is still *computed* here because it
is the first optimiser start, and it is asserted against the stored historical map.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

# Frozen input construction, imported unchanged from the completed study.
from experiments.pcrl_nonlinear_rank_v1.maps import ROLE_SPEC, build_roles
from experiments.pcrl_nonlinear_rank_v1.inputs import Registry, load_representation_labels

from .invariant_moment import (DEFAULT_CHUNK, KernelBlock, QuadraticFeatures, ROLE_ORDER,
                               reference_bandwidth, subset_class_support)
from .objective import (FAMILY_INVARIANT, FAMILY_ORIGINAL, POLICIES, InvariantRolePenalty,
                        PolicyObjective, optimise_policy, spectral_solution)

RANKS = (16, 8)
FAMILY_TAG = {FAMILY_ORIGINAL: 'lin', FAMILY_INVARIANT: 'riv'}


def condition_name(family: str, rank: int, policy: str) -> str:
    """Arm identifier. The ``spectral_`` prefix keeps the historical wire conventions."""
    return f'spectral_{FAMILY_TAG[family]}{rank}_{policy}'


# Conditions that are reused rather than fitted, with the arm they alias.
HISTORICAL_ALIAS = {condition_name(FAMILY_ORIGINAL, 16, p): f'spectral_{p}' for p in POLICIES}
PREDECESSOR_CONTROLS = (
    tuple(f'spectral_lin8_{p}' for p in POLICIES)
    + tuple(f'spectral_nlr{r}_{p}' for r in RANKS for p in POLICIES))


# ------------------------------------------------------------------ per-rank setup
@dataclass
class RankContext:
    rank: int
    w_reference: np.ndarray
    quadratic: QuadraticFeatures
    penalties: dict
    diagnostics: dict


def build_rank_context(state: dict, rank: int, seed: int, labels: dict) -> RankContext:
    """Reference projection, frozen bandwidth, frozen per-role subsets and penalties."""
    w_ref, eigenvalues = spectral_solution(state['U'], rank)
    quadratic = QuadraticFeatures.build(rank)
    bandwidth = reference_bandwidth(state['V'] @ w_ref, seed, rank)
    sigma = bandwidth['sigma_reference']

    penalties, details = {}, {}
    for name, basis_key, attribute, classes in ROLE_SPEC:
        role = state['roles'][name]
        valid = role.valid_mask
        mask = valid if not valid.all() else None
        valid_rows = np.flatnonzero(valid)
        block = KernelBlock.build(name, role, valid_rows, sigma, seed, rank,
                                  role_index=ROLE_ORDER.index(name))
        penalty = InvariantRolePenalty.build(role, quadratic, block, state['V'], mask, w_ref,
                                             state['chunk'])
        penalties[name] = penalty
        details[name] = {
            'basis_width': role.basis_width, 'classes': role.classes, 'n_valid': role.n_valid,
            'population_support': role.support.tolist(),
            'population_unsupported_classes': list(role.unsupported),
            'historical_raw_trace': role.raw_trace, 'historical_zero_trace': role.zero_trace,
            'subset_support': subset_class_support(role, block, labels[attribute][valid]),
            **penalty.metadata(),
        }
    return RankContext(rank=rank, w_reference=w_ref, quadratic=quadratic, penalties=penalties,
                       diagnostics={
                           'rank': rank,
                           'utility_eigenvalues_top': eigenvalues[:rank].tolist(),
                           'bandwidth': bandwidth,
                           'quadratic_feature_width': quadratic.width,
                           'packed_offdiagonal_weight': float(np.sqrt(2.0)),
                           'feature_centering': 'none',
                           'roles': details})


def make_objective(state: dict, context: RankContext, family: str, policy: str) -> PolicyObjective:
    local = {k: context.penalties[k] for k, _, _, _ in ROLE_SPEC if k.startswith('A/')}
    coalition = {k: context.penalties[k] for k, _, _, _ in ROLE_SPEC if k.startswith('AB/')}
    return PolicyObjective(utility=state['U'], local=local, coalition=coalition, policy=policy,
                           family=family, features=state['V'], quadratic=context.quadratic,
                           chunk=state['chunk'])


# ------------------------------------------------------------------ the fitted model
@dataclass
class InvariantRankModel:
    """Serialisable interface model: frozen features plus the new maps.

    ``transform`` needs only ``T`` and ``hA`` -- no ``hB``, no labels, no refit. This
    is the legal-input-dependence property, asserted in the validation suite.
    """

    spectral: object
    maps: dict
    rank_of: dict
    seed: int

    def features(self, t, ha):
        return self.spectral.features(t, ha)

    def transform(self, t, ha, arm):
        return self.features(t, ha) @ self.maps[arm]

    @property
    def arms(self):
        return tuple(sorted(self.maps))


def fit_seed(seed: int, registry: Registry, *, chunk: int = DEFAULT_CHUNK, ranks=RANKS,
             policies=tuple(POLICIES)):
    """Fit the 6 repaired conditions of one seed. Returns (model, diagnostics)."""
    tick = time.perf_counter()
    state = build_roles(seed, registry, chunk)
    labels, _households = load_representation_labels(seed, registry)

    maps, rank_of, records, contexts = {}, {}, {}, {}
    closed_form = {}
    for rank in ranks:
        context = build_rank_context(state, rank, seed, labels)
        contexts[str(rank)] = context.diagnostics

        # Closed-form original-moment solution: the first optimiser start. This is not
        # a new fit -- it reproduces an arm that already exists and is asserted against
        # it in the validation stage.
        starts = {}
        for policy in policies:
            original = make_objective(state, context, FAMILY_ORIGINAL, policy)
            starts[policy] = optimise_policy(original, context.w_reference, seed, rank)
            closed_form[f'{rank}_{policy}'] = starts[policy]['W']

        for policy in policies:
            name = condition_name(FAMILY_INVARIANT, rank, policy)
            objective = make_objective(state, context, FAMILY_INVARIANT, policy)
            result = optimise_policy(objective, starts[policy]['W'], seed, rank)
            w = result['W']
            maps[name] = w
            rank_of[name] = rank

            z = state['V'] @ w
            explained = float(np.sum((w.T @ state['cross']) ** 2))
            direct = np.linalg.lstsq(z, state['R'], rcond=1e-10)[0]
            direct_mse = float(np.sum((state['R'] - z @ direct) ** 2) / len(z))
            identity_mse = state['replay']['residual_frobenius_over_n'] - explained
            shared = objective.bundle_for(w)
            per_role = {}
            for role_name, penalty in context.penalties.items():
                nl, blocks = penalty.nonlinear_value(shared)
                per_role[role_name] = {'linear': penalty.linear_value(w),
                                       'nonlinear_normalized': nl,
                                       'nonlinear_calibrated': penalty.calibration * nl, **blocks}
            records[name] = {
                'condition': name, 'family': FAMILY_INVARIANT, 'rank': rank, 'policy': policy,
                'selected_start': result['selected_start'],
                'training_objective': result['loss'],
                'training_objective_original_family': make_objective(
                    state, context, FAMILY_ORIGINAL, policy).loss(w),
                'closed_form_start_objective_invariant_family': objective.loss(starts[policy]['W']),
                'initial_training_objective': result['starts']['original_moment_spectral'][
                    'initial_loss'],
                'restart_loss_spread': result['restart_loss_spread'],
                'returned_initial_point': result['returned_initial_point'],
                'starts': result['starts'],
                'feasibility_max_abs': result['feasibility_max_abs'],
                'orthogonality_max_abs': float(np.max(abs(w.T @ w - np.eye(rank)))),
                'utility_normalized': float(np.trace(w.T @ state['U'] @ w)),
                'utility_raw_explained': explained,
                'reconstruction_direct_ls_mse': direct_mse,
                'reconstruction_identity_mse': identity_mse,
                'reconstruction_identity_abs_error': abs(direct_mse - identity_mse),
                'released_covariance_max_abs': float(
                    np.max(abs(z.T @ z / len(z) - np.eye(rank)))),
                'per_role': per_role,
            }
    model = InvariantRankModel(spectral=state['model'], maps=maps, rank_of=rank_of, seed=seed)
    diagnostics = {
        'seed': seed, 'runtime_seconds': time.perf_counter() - tick, 'chunk_size': chunk,
        'rows': int(len(state['V'])), 'whitened_rank': int(state['V'].shape[1]),
        'frozen_replay': state['replay'], 'moment_parity': state['moment_parity'],
        'rank_contexts': contexts, 'conditions': records,
        'unique_new_fits': len(records),
        'reused_historical_conditions': sorted(HISTORICAL_ALIAS),
        'reused_predecessor_controls': sorted(PREDECESSOR_CONTROLS),
        'claim_scope': ('Finite empirical moment objective on a bounded row subset. The '
                        'original-moment family is the global optimum of its fixed matrix; the '
                        'rotation-invariant family is a nonconvex objective solved by iterative '
                        'Stiefel descent and is at best locally optimal. Rotation invariance is '
                        'exact by construction and is NOT a privacy guarantee.'),
    }
    return model, diagnostics, closed_form
