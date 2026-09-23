"""Independent phase-I fallback fixture; no ACS rows or refits."""
from __future__ import annotations

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import fit, solver


def test_conflicting_union_keeps_registered_failure_and_labels_fallback():
    cost = np.array([[0., 1.]])
    cuts = [
        {'id': 'attack-a', 'coeff': np.array([[1., 0.]]),
         'rho': .8, 'delta': 0., 'floor': .8},
        {'id': 'attack-b', 'coeff': np.array([[0., 1.]]),
         'rho': .8, 'delta': 0., 'floor': .8},
    ]
    result = fit._solve_with_descriptive_fallback(cost, cuts)
    registered = result['registered_solution']
    fallback = result['solution']
    assert result['registered_feasible'] is False
    assert registered['status'] == 'registered_bank_infeasible'
    assert registered['Q'] is None
    assert abs(registered['phase_one']['minimum_common_violation'] - .3) < 1e-9
    amount = result['fallback_common_relaxation']
    assert abs(amount - (.3 + solver.PRIMAL_TOL)) < 1e-8
    assert fallback['feasible'] and fallback['Q'] is not None
    assert fallback['replay']['maximum_cut_violation'] <= solver.PRIMAL_TOL
    original_violations = [float(cut['floor'] - np.sum(cut['coeff'] * fallback['Q']))
                           for cut in cuts]
    assert max(original_violations) > .29
    assert max(original_violations) <= amount + solver.PRIMAL_TOL
