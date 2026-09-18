"""Stage 4b: the outcome-free mechanism gate, plus the cross-family objective matrix.

Runs on already-fitted maps. **No 2018 or 2017 score is read here** -- this module
cannot reach an attacker, a residence label or a transport table.

Two things are measured:

1. **The predeclared acceptance gate.** ``rotation_share`` is reused unchanged from the
   predecessor. The repaired arm is eligible to be scored only if the measured
   rotation-only share is below 0.10 in every seed, rank and policy; forecast Q1
   sharpens that to 0.01. Because every term of the repaired objective is rotation
   invariant, the expected value is ~0 and the measured value is reported with its
   absolute training gain beside it -- a share is unstable when the gain is tiny, so
   the direct invariance identity is what carries the argument.

2. **The cross-family objective matrix.** Each map is evaluated under **both**
   objectives. Values from differently normalised objectives are **not** directly
   comparable as measures of privacy, so the table is read only within a column.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np

# The rotation diagnostic is REUSED UNCHANGED; it only needs .loss/.loss_and_grad.
from experiments.pcrl_nonlinear_rank_v1.diagnostics import (rotation_invariance_check,
                                                            rotation_share, subspace_geometry)
from experiments.pcrl_nonlinear_rank_v1.inputs import (Registry, load_representation_labels,
                                                       write_json)
from experiments.pcrl_nonlinear_rank_v1.maps import build_roles
from experiments.pcrl_nonlinear_rank_v1.run_fit import limit_threads, machine_state

from .maps import RANKS, build_rank_context, condition_name, make_objective
from .objective import FAMILY_INVARIANT, FAMILY_ORIGINAL, POLICIES, optimise_policy
from .run_fit import OUT, predecessor_maps

GATE_THRESHOLD = 0.10
REGISTERED_FORECAST = 0.01


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    results = {}
    tick = time.perf_counter()
    for seed in seeds:
        registry = Registry.new()
        state = build_roles(seed, registry)
        labels, _ = load_representation_labels(seed, registry)
        mine = joblib.load(out / f'seed_{seed}' / 'maps.joblib')
        theirs, theirs_path = predecessor_maps(seed)

        seed_record = {'predecessor_maps': str(theirs_path), 'ranks': {}}
        for rank in RANKS:
            context = build_rank_context(state, rank, seed, labels)
            rank_record = {}
            for policy in POLICIES:
                invariant = make_objective(state, context, FAMILY_INVARIANT, policy)
                original = make_objective(state, context, FAMILY_ORIGINAL, policy)
                # The closed-form original-moment solution at this rank and policy; it is
                # the same map the fit stage asserted bitwise against the historical arm.
                w_original = optimise_policy(original, context.w_reference, seed, rank)['W']

                riv = condition_name(FAMILY_INVARIANT, rank, policy)
                nlr = f'spectral_nlr{rank}_{policy}'
                w_riv = mine.maps[riv]
                w_nlr = theirs.maps[nlr]

                share = rotation_share(invariant, w_original, w_riv)
                identity = rotation_invariance_check(invariant, w_riv, seed=seed)

                rank_record[policy] = {
                    'condition': riv,
                    'gate': {
                        **share,
                        'gate_threshold': GATE_THRESHOLD,
                        'registered_forecast': REGISTERED_FORECAST,
                        'passes_gate': (share.get('rotation_only_share') is None
                                        or abs(share['rotation_only_share']) < GATE_THRESHOLD),
                        'note': ('A share is unstable when the total gain is tiny and is not a '
                                 'causal decomposition. The direct invariance identity below is '
                                 'the load-bearing evidence.'),
                    },
                    'direct_invariance_identity': identity,
                    # Each map under BOTH objectives. Read only within a column: the two
                    # objectives are normalised differently and their VALUES are not
                    # comparable measures of privacy.
                    'cross_family_objective': {
                        'under_rotation_invariant_objective': {
                            'original_moment_map': invariant.loss(w_original),
                            'defective_nonlinear_map': invariant.loss(w_nlr),
                            'rotation_invariant_map': invariant.loss(w_riv),
                        },
                        'under_original_objective': {
                            'original_moment_map': original.loss(w_original),
                            'defective_nonlinear_map': original.loss(w_nlr),
                            'rotation_invariant_map': original.loss(w_riv),
                        },
                        'utility_normalized': {
                            'original_moment_map': float(
                                np.trace(w_original.T @ state['U'] @ w_original)),
                            'defective_nonlinear_map': float(
                                np.trace(w_nlr.T @ state['U'] @ w_nlr)),
                            'rotation_invariant_map': float(
                                np.trace(w_riv.T @ state['U'] @ w_riv)),
                        },
                        'reading_rule': ('Values from differently normalised objectives are NOT '
                                         'directly comparable measures of privacy. Compare within '
                                         'a column only; the attack comparison on the actual '
                                         'released wires is what carries weight.'),
                    },
                    'subspace_repaired_vs_defective': subspace_geometry(w_riv, w_nlr),
                    'subspace_repaired_vs_original': subspace_geometry(w_riv, w_original),
                    'subspace_defective_vs_original': subspace_geometry(w_nlr, w_original),
                }
                print('DIAG', seed, rank, policy,
                      'share=', rank_record[policy]['gate'].get('rotation_only_share'),
                      'gain=', round(rank_record[policy]['gate'].get('total_training_gain', 0), 6),
                      flush=True)
            seed_record['ranks'][str(rank)] = rank_record
        results[str(seed)] = seed_record

    cells = [(s, r, p, cell)
             for s, sr in results.items()
             for r, rr in sr['ranks'].items()
             for p, cell in rr.items()]
    shares = [c['gate'].get('rotation_only_share') for _, _, _, c in cells]
    finite = [abs(v) for v in shares if v is not None]
    record = {
        'gate': {
            'threshold': GATE_THRESHOLD, 'registered_forecast': REGISTERED_FORECAST,
            'cells': len(cells),
            'cells_with_defined_share': len(finite),
            'max_abs_share': max(finite) if finite else None,
            'passes_gate': all(c['gate']['passes_gate'] for _, _, _, c in cells),
            'meets_registered_forecast': (max(finite) < REGISTERED_FORECAST) if finite else None,
            'timing': 'computed after fitting and BEFORE any 2018 or 2017 score was read',
        },
        'max_direct_invariance_change': max(
            c['direct_invariance_identity']['nonlinear_penalty_change_abs']
            for _, _, _, c in cells),
        'seeds': results,
        'runtime_seconds': time.perf_counter() - tick,
        'machine': machine_state(),
    }
    write_json(out / 'MECHANISM_GATE.json', record)
    print('GATE passes=', record['gate']['passes_gate'],
          'max_abs_share=', record['gate']['max_abs_share'],
          'max_direct_change=', record['max_direct_invariance_change'], flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
