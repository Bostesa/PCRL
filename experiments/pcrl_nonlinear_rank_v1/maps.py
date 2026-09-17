"""Build the new interface maps from the frozen 2018 spectral objects.

Nothing historical is refitted. The frozen ``SpectralModel`` supplies the
whitened features ``V``, the polynomial bases, the utility matrix, the trace
normalised moment Grams and the fifteen fitted out-of-fold nuisance models; this
module only chooses new ``W`` maps under new objectives.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .inputs import (Registry, load_matrix_diagnostics, load_pools, load_representation_labels,
                     load_spectral_model)
from .nonlinear_moment import ChiFeatures, DEFAULT_CHUNK, ProtectedRole
from .objective import (POLICIES, PolicyObjective, RolePenalty, canonicalize, optimise_policy,
                        spectral_solution)

# Protected roles: (name, basis key, attribute, class count)
ROLE_SPEC = (
    ('A/SEX', 'local', 'SEX', 2),
    ('A/RAC1P', 'local', 'RAC1P', 9),
    ('A/public_coverage', 'local', 'public_coverage', 2),
    ('AB/SEX', 'coalition', 'SEX', 2),
    ('AB/RAC1P', 'coalition', 'RAC1P', 9),
)
FAMILIES = ('original', 'nonlinear')
RANKS = (16, 8)


def condition_name(family: str, rank: int, policy: str) -> str:
    """Arm identifier. The ``spectral_`` prefix keeps the historical wire conventions."""
    tag = 'lin' if family == 'original' else 'nlr'
    return f'spectral_{tag}{rank}_{policy}'


HISTORICAL_ALIAS = {condition_name('original', 16, p): 'spectral_' + p for p in POLICIES}


# ------------------------------------------------------------------ inputs
def oof_predictions(model, basis: np.ndarray, folds: np.ndarray, role: str, attribute: str,
                    classes: int) -> np.ndarray:
    """Reconstruct the original three-fold household OOF predictions exactly.

    Fold ``f``'s rows are predicted by the saved fold-``f`` model, which is what
    ``acs_residual_spectral._oof`` produced when the moments were first built.
    """
    predictions = np.zeros((len(basis), classes))
    fitted = model.nuisance_models[role][attribute]
    for f in range(3):
        mask = folds == f
        if mask.any():
            predictions[mask] = fitted[f].predict(basis[mask])
    return predictions


def build_roles(seed: int, registry: Registry, chunk: int = DEFAULT_CHUNK):
    """Frozen V, U, bases, OOF residuals and historical moment Grams for one seed."""
    from experiments.acs_residual_spectral import moment_penalty

    model = load_spectral_model(seed, registry)
    diag = load_matrix_diagnostics(seed, registry)
    _, teacher, anchors = load_pools(seed, registry)
    labels, _households = load_representation_labels(seed, registry)

    t = np.asarray(teacher['representation_fit'], dtype=np.float64)
    ha = np.asarray(anchors['representation_fit/A'], dtype=np.float64)
    hb = np.asarray(anchors['representation_fit/B'], dtype=np.float64)
    v = model.features(t, ha)
    n, q = v.shape

    # Frozen whitening replay: V must reproduce the stored diagnostics.
    replay = {
        'whitening_covariance_max_abs': float(np.max(abs(v.T @ v / n - np.eye(q)))),
        'whitening_mean_max_abs': float(np.max(abs(v.mean(0)))),
        'stored_whitened_rank': int(diag['whitened_rank']),
        'observed_whitened_rank': int(q),
        'rows': int(n),
    }

    bases = {'local': model.qA.transform(ha[:, [1, 3]]),
             'coalition': model.qAB.transform(np.column_stack((ha[:, [1, 3]], hb[:, 1])))}
    folds = np.asarray(diag['fold_assignments'], dtype=int)

    roles, parity = {}, {}
    for name, basis_key, attribute, classes in ROLE_SPEC:
        basis = bases[basis_key]
        y = labels[attribute]
        predicted = oof_predictions(model, basis, folds, basis_key, attribute, classes)
        gram, record = moment_penalty(v, basis, y, predicted, classes)
        stored = diag['penalties'][basis_key]['attributes'][attribute]
        parity[name] = {
            'raw_trace_recomputed': record['raw_trace'],
            'raw_trace_stored': stored['raw_trace'],
            'raw_trace_abs_error': abs(record['raw_trace'] - stored['raw_trace']),
            'class_squared_norm_max_abs_error': max(
                abs(a['squared_norm'] - b['squared_norm'])
                for a, b in zip(record['classes'], stored['classes'])),
            'valid_rows_recomputed': record['valid_rows'],
            'valid_rows_stored': stored['valid_rows'],
        }
        valid = y >= 0
        roles[name] = ProtectedRole(
            name=name, view=name.split('/')[0], basis=basis[valid],
            residual=np.column_stack([(y[valid] == c).astype(float) - predicted[valid, c]
                                      for c in range(classes)]),
            linear_gram=gram, n_valid=int(valid.sum()), classes=classes,
            support=np.bincount(y[valid], minlength=classes),
            unsupported=record['unsupported_classes'], raw_trace=record['raw_trace'],
            zero_trace=record['zero_trace'])
        roles[name].valid_mask = valid

    # Trace-normalised utility matrix, rebuilt from the frozen residual teacher.
    q_local = model.qA.transform(ha[:, [1, 3]])
    teacher_coeff = np.linalg.lstsq(q_local, t, rcond=1e-10)[0]
    r = t - q_local @ teacher_coeff
    r -= r.mean(0)
    cross = v.T @ r / n
    raw = cross @ cross.T
    utility_trace = float(np.trace(raw))
    if utility_trace <= 1e-12:
        raise ValueError(f'utility trace diagnostic failure: {utility_trace}')
    utility = raw / utility_trace
    replay['utility_raw_trace_recomputed'] = utility_trace
    replay['utility_raw_trace_stored'] = diag['utility_raw_trace']
    replay['utility_raw_trace_abs_error'] = abs(utility_trace - diag['utility_raw_trace'])
    replay['residual_frobenius_over_n'] = float(np.sum(r * r) / n)

    return {'model': model, 'diagnostics': diag, 'V': v, 'U': utility, 'U_raw': raw,
            'cross': cross, 'R': r, 'roles': roles, 'moment_parity': parity, 'replay': replay,
            'chunk': chunk, 'teacher': t, 'anchors_A': ha, 'anchors_B': hb}


# ------------------------------------------------------------------ per-rank setup
@dataclass
class RankContext:
    rank: int
    w_reference: np.ndarray
    chi: ChiFeatures
    penalties: dict
    diagnostics: dict


def build_rank_context(state: dict, rank: int, seed: int) -> RankContext:
    """Reference projection, frozen feature family and per-role calibrated penalties."""
    w_ref, eigenvalues = spectral_solution(state['U'], rank)
    chi = ChiFeatures.fit(state['V'] @ w_ref, seed)
    penalties, details = {}, {}
    for name, role in state['roles'].items():
        mask = role.valid_mask if not role.valid_mask.all() else None
        penalty = RolePenalty.build(role, chi, state['V'], mask, w_ref, state['chunk'])
        penalties[name] = penalty
        details[name] = {
            'basis_width': role.basis_width, 'classes': role.classes, 'n_valid': role.n_valid,
            'support': role.support.tolist(), 'unsupported_classes': list(role.unsupported),
            'historical_raw_trace': role.raw_trace, 'historical_zero_trace': role.zero_trace,
            'quadratic_reference_normalizer': penalty.quad_normalizer,
            'fourier_reference_normalizer': penalty.rff_normalizer,
            'linear_reference': penalty.linear_reference,
            'calibration_constant': penalty.calibration,
            'flags': list(penalty.flags), 'reference_raw': penalty.reference_raw,
        }
    return RankContext(rank=rank, w_reference=w_ref, chi=chi, penalties=penalties,
                       diagnostics={'rank': rank,
                                    'utility_eigenvalues_top': eigenvalues[:rank].tolist(),
                                    'chi': chi.metadata(), 'roles': details})


def make_objective(state: dict, context: RankContext, family: str, policy: str) -> PolicyObjective:
    local = {k: context.penalties[k] for k, _, _, _ in ROLE_SPEC if k.startswith('A/')}
    coalition = {k: context.penalties[k] for k, _, _, _ in ROLE_SPEC if k.startswith('AB/')}
    return PolicyObjective(utility=state['U'], local=local, coalition=coalition,
                           policy=policy, family=family, features=state['V'],
                           chunk=state['chunk'])


# ------------------------------------------------------------------ the fitted model
@dataclass
class NonlinearRankModel:
    """Serialisable interface model: frozen features plus the new maps.

    ``transform`` needs only ``T`` and ``hA`` — no ``hB``, no labels, no refit.
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


def fit_seed(seed: int, registry: Registry, *, chunk: int = DEFAULT_CHUNK,
             ranks=RANKS, families=FAMILIES, policies=tuple(POLICIES)):
    """Fit every new condition for one seed. Returns (model, diagnostics)."""
    tick = time.perf_counter()
    state = build_roles(seed, registry, chunk)
    maps, rank_of, records, contexts = {}, {}, {}, {}
    for rank in ranks:
        context = build_rank_context(state, rank, seed)
        contexts[str(rank)] = context.diagnostics
        # Closed-form original-moment solution per policy, also the nonlinear start.
        starts = {}
        for policy in policies:
            original = make_objective(state, context, 'original', policy)
            starts[policy] = optimise_policy(original, context.w_reference, seed, rank)
        for family in families:
            for policy in policies:
                name = condition_name(family, rank, policy)
                objective = make_objective(state, context, family, policy)
                if family == 'original':
                    result = starts[policy]
                else:
                    result = optimise_policy(objective, starts[policy]['W'], seed, rank)
                w = result['W']
                maps[name] = w
                rank_of[name] = rank
                z = state['V'] @ w
                linear_loss = make_objective(state, context, 'original', policy).loss(w)
                nonlinear_loss = make_objective(state, context, 'nonlinear', policy).loss(w)
                explained = float(np.sum((w.T @ state['cross']) ** 2))
                direct = np.linalg.lstsq(z, state['R'], rcond=1e-10)[0]
                direct_mse = float(np.sum((state['R'] - z @ direct) ** 2) / len(z))
                identity_mse = state['replay']['residual_frobenius_over_n'] - explained
                per_role = {}
                shared = make_objective(state, context, 'nonlinear', policy).bundle_for(w)
                for role_name, penalty in context.penalties.items():
                    nl, blocks = penalty.nonlinear_value(shared)
                    per_role[role_name] = {
                        'linear': penalty.linear_value(w),
                        'nonlinear_normalized': nl,
                        'nonlinear_calibrated': penalty.calibration * nl, **blocks}
                records[name] = {
                    'condition': name, 'family': family, 'rank': rank, 'policy': policy,
                    'selected_start': result['selected_start'],
                    'training_objective': result['loss'],
                    'training_objective_original_family': linear_loss,
                    'training_objective_nonlinear_family': nonlinear_loss,
                    'initial_training_objective': result.get('starts', {}).get(
                        'original_moment_spectral', {}).get('initial_loss'),
                    'restart_loss_spread': result.get('restart_loss_spread'),
                    'returned_initial_point': result.get('returned_initial_point', False),
                    'starts': result.get('starts', {}),
                    'eigenvalues_top_sum': result.get('top_eigenvalue_sum'),
                    'closed_form_objective_gap': result.get('objective_gap'),
                    'feasibility_max_abs': result['feasibility_max_abs'],
                    'orthogonality_max_abs': float(np.max(abs(w.T @ w - np.eye(rank)))),
                    'utility_normalized': float(np.trace(w.T @ state['U'] @ w)),
                    'utility_raw_explained': explained,
                    'reconstruction_direct_ls_mse': direct_mse,
                    'reconstruction_identity_mse': identity_mse,
                    'reconstruction_identity_abs_error': abs(direct_mse - identity_mse),
                    'released_covariance_max_abs': float(np.max(abs(z.T @ z / len(z) - np.eye(rank)))),
                    'per_role': per_role,
                    'historical_alias': HISTORICAL_ALIAS.get(name),
                }
    model = NonlinearRankModel(spectral=state['model'], maps=maps, rank_of=rank_of, seed=seed)
    diagnostics = {
        'seed': seed, 'runtime_seconds': time.perf_counter() - tick,
        'chunk_size': chunk, 'rows': int(len(state['V'])), 'whitened_rank': int(state['V'].shape[1]),
        'frozen_replay': state['replay'], 'moment_parity': state['moment_parity'],
        'rank_contexts': contexts, 'conditions': records,
        'unique_new_fits': sum(1 for n in records if n not in HISTORICAL_ALIAS),
        'reused_alias_conditions': sorted(n for n in records if n in HISTORICAL_ALIAS),
        'claim_scope': ('Finite empirical moment objective only. The original-moment family is the '
                        'global optimum of its fixed matrix; the nonlinear family is a nonconvex '
                        'objective solved by iterative Stiefel descent and is at best locally '
                        'optimal. No marginal or conditional privacy guarantee.'),
    }
    return model, diagnostics
