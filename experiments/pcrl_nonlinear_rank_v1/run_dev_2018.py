"""Phase 2: 2018 development audits, utility probes and scores for the new interfaces.

Maximum reuse of the immutable historical machinery. Every attack family,
selection rule, role, mask, subset index, seed formula and scorer is the
historical one; this module only points them at the new wires. Nothing in the
completed studies is written to.

Attack scopes stay distinct: the new interfaces get fresh five-candidate and
kernel families for their A/AB roles, the canonical B candidates and the legally
routed H ancestors — exactly the slate the historical spectral arms received. No
new candidate family is introduced, so attack opportunities are matched without
adding anything to the controls. No spectral saved observer exists, so no
catch-up scope is claimed for them.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np

from experiments import acs_fixed_predictions_audits as oldaudit
from experiments import acs_spectral_audits as audit
from experiments.run_acs_coalition import TARGETS as COALITION_TARGETS
from experiments.run_acs_protection import TASKS
from experiments.run_acs_residual_spectral import load_labels, reusable_utilities, score, wires
from experiments.run_acs_transfer import subset_indices

from .inputs import (DEV_NAME, FIXED_NAME, HIST_ROOT, OUT, Registry, array_hash, read_json,
                     resolve, sha_file, write_json)
from .maps import HISTORICAL_ALIAS
from .run_fit import limit_threads, machine_state

DEV_H_AUDITS = f'results/{DEV_NAME}/seed_{{seed}}/H/audits'
FIXED_SEED = f'results/{FIXED_NAME}/seed_{{seed}}'


def _fit_pools(space: dict) -> dict:
    return {view: {pool: space[view][pool] for pool in oldaudit.FIT_POOLS} for view in space}


def evaluate_seed(out: Path, seed: int, conditions, registry: Registry) -> dict:
    frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
    ti = {t: subset_indices(labels['downstream_fit'][t], 2048, 1230000 + 100 * seed + j)
          for j, t in enumerate(TASKS)}
    ai = {t: subset_indices(labels['attacker_fit'][t], 4096, 1240000 + 100 * seed + j)
          for j, t in enumerate(COALITION_TARGETS)}

    # Compatibility: the subset indices must be the historical ones, bit for bit.
    stored = read_json(registry.resolve(FIXED_SEED.format(seed=seed) + '/indices.json'))
    observed = {'utility': {t: array_hash(ix) for t, ix in ti.items()},
                'attacker': {t: array_hash(ix) for t, ix in ai.items()}}
    if observed != stored:
        raise AssertionError(f'subset index hashes differ from the historical study for seed {seed}')

    ancestor = audit.load_audits(resolve(DEV_H_AUDITS.format(seed=seed)))
    h_utility = resolve(FIXED_SEED.format(seed=seed) + '/H')
    _, teacher, anchors = None, None, None
    results = {}
    for condition in conditions:
        if condition in HISTORICAL_ALIAS:
            # Objective identical to a historical arm: reuse its frozen unit, never refit.
            results[condition] = {'condition': condition, 'disposition': 'REUSED HISTORICAL',
                                  'aliases': HISTORICAL_ALIAS[condition], 'new_fits': 0}
            print('DEV_ALIAS', seed, condition, '->', HISTORICAL_ALIAS[condition], flush=True)
            continue
        dest = out / f'seed_{seed}' / condition
        marker = dest / 'complete.json'
        if marker.exists():
            results[condition] = read_json(marker)
            print('DEV_REUSED', seed, condition, flush=True)
            continue
        tick = time.perf_counter()
        dest.mkdir(parents=True, exist_ok=True)
        release = out / f'seed_{seed}' / 'releases' / condition / 'releases.npz'
        w, d = wires(release)
        if d is not None:
            raise AssertionError('the new interfaces have no derived space')

        # Anchor parity on every released pool, against the frozen historical anchors.
        fixed_anchors = dict(np.load(resolve(FIXED_SEED.format(seed=seed) + '/anchors.npz')))
        for pool in w['A']:
            if not np.array_equal(w['A'][pool][:, :4], fixed_anchors[pool + '/A']):
                raise AssertionError(f'H_A parity failed: {condition} {pool}')
            if not np.array_equal(w['B'][pool], fixed_anchors[pool + '/B']):
                raise AssertionError(f'H_B parity failed: {condition} {pool}')

        utility_path = dest / 'utility_state.joblib'
        if utility_path.exists():
            u = joblib.load(utility_path)
        else:
            u = reusable_utilities(w, labels, ti, seed, dest, h_utility)
            joblib.dump(u, utility_path)

        attacks = audit.build_audits(
            _fit_pools(w), None, {p: labels[p] for p in oldaudit.FIT_POOLS},
            ai, seed, dest / 'audits', historical=None, ancestor=ancestor)

        # Selections are on disk BEFORE the development test pool is read.
        write_json(dest / 'selection_before_test.json', {
            'seed': seed, 'condition': condition,
            'utility': u['selection'], 'utility_metadata': u['metadata'],
            'audits': attacks['selection'],
            'fit_complete_sha256': sha_file(out / f'seed_{seed}' / 'fit_complete.json'),
            'release_sha256': sha_file(release),
            'development_test_pool_read': False})

        score(dest, seed, condition, w, d, u, attacks, labels, weights, None)

        record = {'seed': seed, 'condition': condition, 'year': 2018,
                  'runtime_seconds': time.perf_counter() - tick,
                  'audit_counts': attacks['metadata']['counts'],
                  'projection_checks': len(attacks['metadata']['projection_checks']),
                  'A_width': int(w['A']['test'].shape[1]),
                  'AB_width': int(w['AB']['test'].shape[1]),
                  'new_utility_fits': 6,
                  'metrics_sha256': sha_file(dest / 'metrics.json'),
                  'predictions_sha256': sha_file(dest / 'predictions.npz'),
                  'selection_sha256': sha_file(dest / 'selection_before_test.json'),
                  'evaluation_status': 'DEVELOPMENT EVALUATION (2018 pools, repeatedly used)'}
        write_json(marker, record)
        results[condition] = record
        print('DEV_DONE', seed, condition, round(record['runtime_seconds'], 1), 's', flush=True)
    return results


def run(out: Path = OUT, seeds=(0, 1, 2), conditions=None) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        names = conditions or read_json(out / f'seed_{seed}' / 'fit_complete.json')['conditions']
        summary[str(seed)] = evaluate_seed(out, seed, names, registry)
        write_json(out / f'seed_{seed}' / 'dev_2018_inputs.json', registry.dump())
    write_json(out / 'DEV_2018_SUMMARY.json', {
        'seeds': summary, 'machine': machine_state(),
        'scope_note': ('2018 pools are the original development resource of the residual spectral '
                       'study and have been used repeatedly. These are development numbers.'),
        'attack_scope_note': ('Fresh five-candidate and kernel families for A/AB roles, canonical B '
                              'candidates and legally routed H ancestors. No spectral saved '
                              'observer exists, so no catch-up scope is claimed; historical '
                              'catch-up exposure is never merged into the common fresh scope.')})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--only', nargs='+')
    a = p.parse_args()
    run(a.out, tuple(a.seeds), a.only)


if __name__ == '__main__':
    main()
