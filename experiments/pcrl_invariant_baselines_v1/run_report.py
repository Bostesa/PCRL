"""Stage 8: 2018 development endpoints, uncertainty and decisions.

Fits nothing and selects nothing. The bootstrap, the studentized max-|t| adjustment,
the advantage rule, the point-estimate coordination rule, the noninferiority and
equivalence reporting and every endpoint definition are **imported unchanged** from the
completed study. Only the path resolution, the comparison slate and the contrast list
are new.

All uncertainty here is **descriptive development uncertainty conditional on fitted
systems**. It does not undo repeated use of the 2018 pools and it is not Census
replicate-weight variance or retraining variability. Every seed and both weightings are
reported, never only pooled means.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT, Registry, write_json
from experiments.pcrl_nonlinear_rank_v1.run_fit import limit_threads, machine_state
from experiments.run_acs_residual_spectral import load_labels

# Bootstrap, adjustment, contrast machinery and CSV writer: REUSED unchanged.
from experiments.pcrl_nonlinear_rank_v1.run_report import (_losses, _replay_row, _weights_for,
                                                           contrast_intervals, csvout)

from .report import (ALL_CONDITIONS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, ERASURE,
                     FAMILY_ENDPOINTS, FORBIDDEN, MAIN_BUDGET, MAIN_SCOPE, MAIN_SPLIT, NEW_ARMS,
                     OPTNET, OUT, PREDECESSOR, REPAIRED, WEIGHTS, ClusterBootstrap, advantage,
                     condition_file, coordination, flat_rows, load_points, noninferiority)

RANKS = (16, 8)
POLICIES = ('L1', 'L2', 'C1')


def build_bootstrap(seeds, registry: Registry):
    """Household clusters over the cohort; see ClusterBootstrap for why not per seed."""
    frames = {}
    for seed in seeds:
        frames[seed] = load_labels(HIST_ROOT, seed)
    cohort = frames[seeds[0]][0].SERIALNO.to_numpy()
    boot = ClusterBootstrap(cohort, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    targets = sorted({e.split('/')[1] for e in FORBIDDEN} | set(frames[seeds[0]][2]['test']))
    for seed in seeds:
        _frame, pools, labels, weights = frames[seed]
        rows = np.asarray(pools['test'])
        for target in targets:
            boot.register(seed, target, rows, labels['test'][target] >= 0, weights['test'])
    return boot, frames


def person_loss_vectors(out: Path, seeds, conditions, points, frames, registry: Registry):
    """Per-person test-pool losses of the SELECTED candidate for every endpoint.

    Reimplemented here only because it must resolve through this study's three-source
    ``condition_file``; the loss, weighting and replay helpers are imported unchanged.
    """
    vectors, replay = {}, []
    for seed in seeds:
        _frame, _pools, labels, _weights = frames[seed]
        for condition in conditions:
            path = condition_file(out, seed, condition, 'predictions.npz')
            registry.add(path)
            with np.load(path) as store:
                for weight in WEIGHTS:
                    point = points[seed, condition, MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE]
                    for task in point['utility']:
                        row = point['selected']['utility/' + task]
                        key = f"utility/{row['view']}/{task}/None/{row['candidate_id']}/test"
                        y = labels['test'][task]
                        losses, _ = _losses(store[key], y)
                        vectors[seed, condition, weight, 'utility/' + task] = (task, losses)
                        replay.append(_replay_row(seed, condition, weight, 'utility/' + task,
                                                  losses, point['utility'][task],
                                                  _weights_for(frames[seed], y, weight)))
                    for endpoint in FORBIDDEN:
                        row = point['selected']['audit/' + endpoint]
                        target = endpoint.split('/')[1]
                        key = (f"audit/{row['view']}/{target}/{row['audit_budget']}/"
                               f"{row['candidate_id']}/test")
                        y = labels['test'][target]
                        losses, _ = _losses(store[key], y)
                        vectors[seed, condition, weight, 'recovery/' + endpoint] = (target, losses)
                        replay.append(_replay_row(seed, condition, weight, 'loss/' + endpoint,
                                                  losses, point['losses'][endpoint],
                                                  _weights_for(frames[seed], y, weight)))
    return vectors, replay


def contrasts() -> list:
    """The predeclared comparison list.

    Grouped into families only to label the output; the max-|t| adjustment is applied
    across the five family endpoints within each (left, right, weighting) contrast,
    exactly as the completed study did.
    """
    out = []
    for rank in RANKS:
        for policy in POLICIES:
            riv = f'spectral_riv{rank}_{policy}'
            nlr = f'spectral_nlr{rank}_{policy}'
            lin = f'spectral_lin{rank}_{policy}' if rank == 8 else f'spectral_{policy}'
            # Q3: does removing the provably inert slack change MEASURED recovery?
            out.append({'left': riv, 'right': nlr, 'family': 'repair_vs_defective'})
            # the repaired arm against its own linear-moment control at matched rank
            out.append({'left': riv, 'right': lin, 'family': 'repair_vs_original'})
            # Q5: against the competing frozen neural channel
            out.append({'left': riv, 'right': 'J', 'family': 'repair_vs_J'})
        # Q4: the coordination cells, C1 against both local controls
        out.append({'left': f'spectral_riv{rank}_C1', 'right': f'spectral_riv{rank}_L1',
                    'family': 'coordination_local'})
        out.append({'left': f'spectral_riv{rank}_C1', 'right': f'spectral_riv{rank}_L2',
                    'family': 'coordination_local'})
    # external adaptations, each against J and against the best-placed repaired arm
    for arm in ERASURE + OPTNET:
        out.append({'left': arm, 'right': 'J', 'family': 'external_vs_J'})
        out.append({'left': arm, 'right': 'spectral_riv16_C1', 'family': 'external_vs_repaired'})
    for arm in ERASURE:
        out.append({'left': arm, 'right': 'A0', 'family': 'erasure_vs_its_own_source'})
    return out


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    tick = time.perf_counter()
    registry = Registry.new()

    conditions = [c for c in ALL_CONDITIONS]
    points, _raw, _prior = load_points(out, seeds, conditions, registry)
    boot, frames = build_bootstrap(seeds, registry)
    vectors, replay = person_loss_vectors(out, seeds, conditions, points, frames, registry)

    rows = flat_rows(points, seeds, conditions)
    specs = contrasts()
    intervals = contrast_intervals(boot, vectors, seeds, specs, FAMILY_ENDPOINTS)

    # Decisions, per (left, right, weighting).
    decisions = {}
    grouped = {}
    for row in intervals:
        grouped.setdefault((row['left'], row['right'], row['weight']), []).append(row)
    for (left, right, weight), group in grouped.items():
        decisions[f'{left} vs {right} / {weight}'] = {
            **advantage(group), 'noninferiority_and_equivalence': noninferiority(group)}

    registered = {}
    for rank in RANKS:
        left = f'spectral_riv{rank}_C1'
        registered[f'coordination_repaired_rank{rank}_C1_vs_local'] = coordination(
            decisions, left, [f'spectral_riv{rank}_L1', f'spectral_riv{rank}_L2'])
        registered[f'coordination_repaired_rank{rank}_C1_vs_J'] = coordination(
            decisions, left, ['J'])

    max_replay = max((r['abs_difference'] for r in replay), default=0.0)
    record = {
        'seeds': list(seeds),
        'conditions': conditions,
        'new_arms_scored_here': list(NEW_ARMS),
        'reused_never_rescored': list(PREDECESSOR) + [c for c in conditions
                                                      if c not in NEW_ARMS
                                                      and c not in PREDECESSOR],
        'bootstrap': {'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
                      'clusters': int(boot.n_groups), 'unit': 'cohort SERIALNO household',
                      'adjustment': 'single-step studentized max-|t| within each contrast family',
                      'note': ClusterBootstrap.__doc__.strip()},
        'score_replay': {'checks': len(replay), 'max_abs_difference': max_replay},
        'registered_decisions': registered,
        'decisions': decisions,
        'runtime_seconds': time.perf_counter() - tick,
        'machine': machine_state(),
        'inputs': registry.dump(),
        'uncertainty_scope': (
            'DEVELOPMENT uncertainty conditional on fitted systems. The paired household '
            'bootstrap quantifies sampling variability for fixed fitted systems; it cannot undo '
            'repeated use of the 2018 pools, and it is neither Census replicate-weight variance '
            'nor retraining variability.'),
        'coordination_rule_scope': (
            'The coordination rule is a one-sided POINT-ESTIMATE rule reused for continuity. A '
            'pass is NOT statistical equivalence and NOT noninferiority within .001. Equivalence '
            'and noninferiority are reported separately per contrast; absence of significance is '
            'neither.'),
    }
    csvout(out / 'PER_SEED.csv', rows)
    csvout(out / 'PAIRED_INTERVALS.csv', intervals)
    write_json(out / 'DEVELOPMENT_2018.json', record)
    csvout(out / 'SCORE_REPLAY.csv', replay)
    print('REPORT done', round(record['runtime_seconds'], 1), 's; replay max abs diff',
          f'{max_replay:.2e}', flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
