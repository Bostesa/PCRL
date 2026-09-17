"""Rank diagnostic: is eigenvalue-sign rank selection a live issue for this interface?

Outcome-free. Reads only the frozen 2018 matrix diagnostics of the completed
residual spectral study (already-saved eigenvalues of every arm matrix) and the
saved spectral maps. No ACS outcome, no reserved label, no attack and no fit.

The proposal under test (from the completed study's own next-step note) is to
replace the fixed ``r = 16`` with

    r_plus = min(16, #{eigenvalues of U - (P_L + P_AB) above tol})
    tol    = max(1e-12, 1e-10 * max|eigenvalues|)

Note the scope limit this diagnostic cannot escape: ``r_plus`` chosen this way is
the rank of the OLD fixed linear-moment objective. It is not the optimal rank of
the new nonlinear objective, whose matrix depends on ``W``.
"""
from __future__ import annotations

import numpy as np

from .inputs import Registry, load_matrix_diagnostics

TARGET_RANK = 16
ORIGINAL_OBJECTIVES = {'L1': 'spectral_L1', 'L2': 'spectral_L2', 'C1': 'spectral_C1'}
RANK_SELECTOR_ARM = 'spectral_C1'          # U - (P_L + P_AB)
SENSITIVITY_RANK = 8                        # predeclared, fires only if every seed has r_plus == 16


def tolerance(eigenvalues: np.ndarray) -> float:
    """The existing prospectively specified relative convention, reused for replay."""
    return max(1e-12, 1e-10 * float(np.max(np.abs(eigenvalues))))


def spectrum_record(eigenvalues, name: str) -> dict:
    ev = np.asarray(eigenvalues, dtype=np.float64)
    tol = tolerance(ev)
    positive = int((ev > tol).sum())
    negative = int((ev < -tol).sum())
    within = int((np.abs(ev) <= tol).sum())
    top = ev[:TARGET_RANK]
    return {
        'objective': name,
        'dimension': int(ev.size),
        'tolerance': tol,
        'eigenvalues_descending': ev.tolist(),
        'positive_count': positive,
        'negative_count': negative,
        'within_tolerance_count': within,
        'retained_rank_r_plus': int(min(TARGET_RANK, positive)),
        'nonpositive_among_top_16': int((top <= tol).sum()),
        'negative_among_top_16': int((top < -tol).sum()),
        'max_eigenvalue': float(ev[0]),
        'eigenvalue_16': float(ev[TARGET_RANK - 1]) if ev.size >= TARGET_RANK else None,
        'eigenvalue_17': float(ev[TARGET_RANK]) if ev.size > TARGET_RANK else None,
        'min_eigenvalue': float(ev[-1]),
    }


def diagnose(seeds=(0, 1, 2), registry: Registry | None = None) -> dict:
    registry = registry or Registry.new()
    per_seed = {}
    for seed in seeds:
        diag = load_matrix_diagnostics(seed, registry)
        objectives = {}
        for label, arm in ORIGINAL_OBJECTIVES.items():
            objectives[label] = spectrum_record(diag['arms'][arm]['eigenvalues'], label)
        utility = spectrum_record(diag['arms']['spectral_S0']['eigenvalues'], 'U_only')
        selector = objectives[[k for k, v in ORIGINAL_OBJECTIVES.items() if v == RANK_SELECTOR_ARM][0]]
        per_seed[str(seed)] = {
            'seed': seed,
            'representation_fit_rows': int(diag['n_rows']),
            'whitened_rank': int(diag['whitened_rank']),
            'historical_output_rank': int(diag['output_rank']),
            'utility_matrix_spectrum': utility,
            'objectives': objectives,
            'r_plus': selector['retained_rank_r_plus'],
            'r_plus_source_objective': RANK_SELECTOR_ARM,
        }
    r_plus = {s: v['r_plus'] for s, v in per_seed.items()}
    all_full = all(v == TARGET_RANK for v in r_plus.values())
    any_zero = any(v == 0 for v in r_plus.values())
    return {
        'target_rank': TARGET_RANK,
        'tolerance_convention': 'max(1e-12, 1e-10 * max|eigenvalues|)',
        'rank_selector_objective': RANK_SELECTOR_ARM,
        'per_seed': per_seed,
        'r_plus_by_seed': r_plus,
        'all_seeds_r_plus_equals_16': all_full,
        'any_seed_r_plus_zero': any_zero,
        'reduced_rank_recipe_is_alias_of_rank_16': all_full,
        'predeclared_sensitivity_rank': SENSITIVITY_RANK if all_full else None,
        'sensitivity_branch_fires': all_full,
        'sensitivity_label': 'compression, not eigenvalue-sign selection' if all_full else None,
        'structural_note': (
            'U_raw = (V\'R/n)(R\'V/n) with R of width 32, so rank(U) <= 32 and the utility '
            'matrix has at most 32 positive eigenvalues. Because the penalties are positive '
            'semidefinite, #positive(U - lambda P) <= 32 as well. r_plus can therefore only '
            'fall below 16 if a penalty drives more than 16 of U\'s 32 positive directions '
            'to or below zero.'),
        'scope_note': (
            'r_plus is the retained rank of the OLD fixed linear-moment matrix. It is not the '
            'optimal rank of the new nonlinear objective, which has no fixed matrix.'),
        'inputs': registry.dump(),
    }
