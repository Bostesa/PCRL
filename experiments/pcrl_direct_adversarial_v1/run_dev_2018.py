"""Stage 5: 2018 development audits, utility probes and scores for the new interfaces.

Maximum reuse of the immutable historical machinery. Every attack family, selection
rule, role, mask, subset index, seed formula and scorer is the historical one; this
module only points them at the new wires. Nothing in the completed studies is written
to, and no already-scored arm is rescored.

Attack scopes stay distinct: the new interfaces get fresh five-candidate and kernel
families for their A/AB roles, the canonical B candidates and the legally routed H
ancestors -- exactly the slate the historical spectral arms received. **No new
candidate family is introduced.** No saved observer exists for these arms, so no
catch-up scope is claimed and historical catch-up exposure is never merged into the
common fresh scope.

The training attacker ensemble of `train.py` is **not** reused here in any form. Every
audit model is fitted fresh by the historical harness.
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

from experiments.pcrl_nonlinear_rank_v1.inputs import DEV_NAME, FIXED_NAME, HIST_ROOT

from .inputs import OUT, Registry, array_hash, read_json, resolve, sha_file, write_json
from .run_fit import limit_threads, machine_state

DEV_H_AUDITS = f'results/{DEV_NAME}/seed_{{seed}}/H/audits'
FIXED_SEED = f'results/{FIXED_NAME}/seed_{{seed}}'


def _fit_pools(space: dict) -> dict:
    return {view: {pool: space[view][pool] for pool in oldaudit.FIT_POOLS} for view in space}


def new_conditions(out: Path, seed: int) -> list:
    root = out / f'seed_{seed}' / 'releases'
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / 'releases.npz').exists())


def evaluate_seed(out: Path, seed: int, conditions, registry: Registry,
                  catchup: int = None) -> dict:
    frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
    ti = {t: subset_indices(labels['downstream_fit'][t], 2048, 1230000 + 100 * seed + j)
          for j, t in enumerate(TASKS)}
    ai = {t: subset_indices(labels['attacker_fit'][t], 4096, 1240000 + 100 * seed + j)
          for j, t in enumerate(COALITION_TARGETS)}

    stored = read_json(registry.resolve(FIXED_SEED.format(seed=seed) + '/indices.json'))
    observed = {'utility': {t: array_hash(ix) for t, ix in ti.items()},
                'attacker': {t: array_hash(ix) for t, ix in ai.items()}}
    if observed != stored:
        raise AssertionError(f'subset index hashes differ from the historical study for seed {seed}')

    ancestor = audit.load_audits(resolve(DEV_H_AUDITS.format(seed=seed)))
    h_utility = resolve(FIXED_SEED.format(seed=seed) + '/H')
    fixed_anchors = dict(np.load(resolve(FIXED_SEED.format(seed=seed) + '/anchors.npz')))

    results = {}
    for condition in conditions:
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

        write_json(dest / 'selection_before_test.json', {
            'seed': seed, 'condition': condition,
            'utility': u['selection'], 'utility_metadata': u['metadata'],
            'audits': attacks['selection'],
            'release_sha256': sha_file(release),
            'development_test_pool_read': False})

        score(dest, seed, condition, w, d, u, attacks, labels, weights, None)
        validate_predictions(dest / 'predictions.npz')

        record = {'seed': seed, 'condition': condition, 'year': 2018,
                  'runtime_seconds': time.perf_counter() - tick,
                  'audit_counts': attacks['metadata']['counts'],
                  'A_width': int(w['A']['test'].shape[1]),
                  'AB_width': int(w['AB']['test'].shape[1]),
                  'metrics_sha256': sha_file(dest / 'metrics.json'),
                  'predictions_sha256': sha_file(dest / 'predictions.npz'),
                  'selection_sha256': sha_file(dest / 'selection_before_test.json'),
                  'evaluation_status': 'DEVELOPMENT EVALUATION (2018 pools, repeatedly used)'}
        write_json(marker, record)
        results[condition] = record
        print('DEV_DONE', seed, condition, round(record['runtime_seconds'], 1), 's', flush=True)
    return results


def validate_predictions(path: Path):
    """Finiteness, bounds, nonzero row mass and row sums, on every stored matrix."""
    with np.load(path, allow_pickle=False) as store:
        for key in store.files:
            p = np.asarray(store[key], np.float64)
            if not np.isfinite(p).all():
                raise AssertionError(f'non-finite probabilities in {path}:{key}')
            if (p < -1e-12).any() or (p > 1 + 1e-8).any():
                raise AssertionError(f'out-of-range probabilities in {path}:{key}')
            mass = p.sum(1)
            if (mass <= 0).any():
                raise AssertionError(f'zero-mass probability rows in {path}:{key}')
            if not np.allclose(mass, 1.0, atol=1e-6, rtol=0):
                raise AssertionError(f'unnormalised probability rows in {path}:{key}')


def run(out: Path = OUT, seeds=(0, 1, 2), conditions=None) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        names = conditions or new_conditions(out, seed)
        summary[str(seed)] = evaluate_seed(out, seed, names, registry)
        write_json(out / f'seed_{seed}' / 'dev_2018_inputs.json', registry.dump())
    write_json(out / 'DEV_2018_SUMMARY.json', {
        'seeds': summary, 'machine': machine_state(),
        'scope_note': ('2018 pools are the original development resource of this line of work '
                       'and have been used repeatedly. These are DEVELOPMENT numbers. The '
                       'paired household bootstrap quantifies sampling variability for fixed '
                       'fitted systems; it cannot undo repeated use.'),
        'attack_scope_note': ('Fresh five-candidate and kernel families for A/AB roles, canonical '
                              'B candidates and legally routed H ancestors. No saved observer '
                              'exists for these arms, so no catch-up scope is claimed; historical '
                              'catch-up exposure is never merged into the common fresh scope.'),
        'training_ensemble_note': ('The differentiable training ensemble of train.py is not '
                                   'reused here in any form. Every audit model is fitted fresh '
                                   'by the historical harness, and trees and kernels were never '
                                   'in the training gradient at all.'),
        'reuse_note': ('Historical H, E, A0, L025, L20, J, spectral_*, leace_A0, splince_A0 and '
                       'optnet16_* arms are reused by exact hash and recipe. None is refitted or '
                       'rescored here.')})
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
