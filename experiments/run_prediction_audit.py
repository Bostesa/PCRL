"""Close the scalar-release audit without changing any learned release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments import run_nonlinear_conflict as old
from experiments.run_nonlinear_release import prediction_views, diagnostic_views, direct_scores
from experiments.nonlinear_release_training import load_reference
from experiments.run_frozen_release_diagnostic import (
    freeze, tensor_records, load_frozen, integrity, apply_final, file_hash,
)
from experiments.frozen_release_adversaries import (
    SavedAdversaryPredictor, SelectedAdversaryPredictor, TARGET_NAMES,
)

torch.set_num_threads(1)
PRIOR = ROOT/'results/redesign_20260907_nonlinear_upstream_v1'
RELEASE = ROOT/'results/redesign_20260907_nonlinear_release_v1'
DIAGNOSTIC = ROOT/'results/redesign_20260907_frozen_adversary_v1'
CONFIG = {
    'seeds': [2, 0, 1], 'test_rng_seeds': {str(s): 1100004+100*s for s in (0, 1, 2)},
    'test_n': 4096, 'utility_floor_each': .99,
    'thresholds': dict(zip(TARGET_NAMES, (.05, .05, .05, .05, .10))),
    'attack_seed_base': 1200000, 'task_seed_base': 1300000,
    'families': ['linear', 'mlp', 'histgb'],
    'full_reference': {'seed': 2, 'key': 'E_dual_0.01'},
}
TARGET_GROUPS = (('p1', (1, 2)), ('p2', (0, 2)), ('combined', (2,)))
dump = old.dump


def scalar_integrity(model):
    return {name: tensor_records(model[name]) for name in ('encoder', 'heads')}


def fresh_test(seed, manifest, selection_path):
    """The fitting API has no access to these arrays; require a saved choice."""
    selection_path = Path(selection_path)
    if not selection_path.exists():
        raise RuntimeError('Save all attacker selections before generating final test')
    saved = json.loads(selection_path.read_text())
    rng_seed = CONFIG['test_rng_seeds'].get(str(seed), 1100004+100*seed)
    if saved.get('test_rng_seed') != rng_seed or saved.get('test_generated') is not False:
        raise ValueError('Selection record does not authorize this fresh test stream')
    latent = np.random.default_rng(rng_seed).normal(size=(CONFIG['test_n'], 8))
    q1, q2, offset = old.generator_parameters()
    r = latent@q1
    x = ((r+.2*r**3)@q2+offset-np.array(manifest['preprocessing_mean']))/np.array(manifest['preprocessing_std'])
    return {'x': x, 'y': latent[:, :3],
            'ids': np.arange(len(x), dtype=np.int64)+np.int64(rng_seed)*100000}


def cache(directory, views, data):
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, hs in views.items():
        arrays = {f'view_{p}': h for p, h in enumerate(hs)}
        arrays.update(y=data[split]['y'], ids=data[split]['ids'])
        np.savez_compressed(directory/f'{split}.npz', **arrays)
        hashes[split] = {k: old.digest_array(v) for k, v in arrays.items()}
    return hashes


def assess(evaluation):
    utility = {k: evaluation['task']['mlp'][k]['r2'] for k in ('p1_U', 'p2_V')}
    utility_pass = all(v is not None and np.isfinite(v) and v >= .99 for v in utility.values())
    checks = {f'{family}/{target}': score['r2'] is not None and np.isfinite(score['r2'])
              and score['r2'] <= CONFIG['thresholds'][target]
              for family, row in evaluation['attack'].items() for target, score in row.items()}
    protection_pass = len(checks) == 15 and all(checks.values())
    return {'utility': utility, 'utility_pass': bool(utility_pass),
            'protection_pass': bool(protection_pass), 'feasible': bool(utility_pass and protection_pass),
            'constraint_checks': checks}


def fit_probes(views, data, seed, out):
    from experiments.prediction_release_attackers import fit_attackers
    fitted = {'attack': {}, 'task': {}}
    for p, (view_name, targets) in enumerate(TARGET_GROUPS):
        names = [f'{view_name}_{old.SIGNALS[j]}' for j in targets]
        fitted['attack'][p] = fit_attackers(
            views['attacker_fit'][p], data['attacker_fit']['y'][:, targets],
            views['validation'][p], data['validation']['y'][:, targets],
            target_names=names, seed=CONFIG['attack_seed_base']+100*seed+10*p,
            out_dir=out/'probes'/f'attack_{view_name}')
    for p, target in enumerate(('p1_U', 'p2_V')):
        fitted['task'][p] = fit_attackers(
            views['representation_train'][p], data['representation_train']['y'][:, p:p+1],
            views['validation'][p], data['validation']['y'][:, p:p+1],
            target_names=[target], seed=CONFIG['task_seed_base']+100*seed+10*p,
            out_dir=out/'probes'/f'task_p{p+1}', kinds=('linear', 'mlp'))
    return fitted


def evaluate(probes, views, y):
    result = {'attack': {kind: {} for kind in CONFIG['families']}, 'task': {'linear': {}, 'mlp': {}}}
    for p, (_, targets) in enumerate(TARGET_GROUPS):
        for kind, probe in probes['attack'][p].items():
            result['attack'][kind].update(probe.score(views[p], y[:, targets]))
    for p in range(2):
        for kind, probe in probes['task'][p].items():
            result['task'][kind].update(probe.score(views[p], y[:, p:p+1]))
    return result


def selected_families(validation):
    return {target: min(CONFIG['families'], key=lambda family: (
                -validation['attack'][family][target]['r2'], family)) for target in TARGET_NAMES}


def full_views(model, data):
    return {split: apply_final([old.encode(model['encoder'], row['x'], p) for p in range(2)], model['erasers'])
            for split, row in data.items()}


def continued_reference(model):
    source = DIAGNOSTIC/'seed_2/E_dual_0.01/fitting/B'
    c = model['checkpoint']
    a = SavedAdversaryPredictor(model['adversaries'], c['training_release'], c['target_mean'], c['target_std'])
    states = []
    for j, name in enumerate(TARGET_NAMES):
        record = torch.load(source/f'selected_target_{j}.pt', weights_only=True, map_location='cpu')
        if record['target_name'] != name or record['target_index'] != j:
            raise ValueError('Continued adversary target mismatch')
        for coordinate, value in record['preprocessing'].items():
            if not torch.equal(value, a.preprocessing.state()[coordinate]):
                raise ValueError('Continued adversary preprocessing changed')
        states.append(record['adversaries'])
    b = SelectedAdversaryPredictor(model['adversaries'], a.preprocessing, states,
                                   json.loads((source/'metadata.json').read_text()))
    return {'saved_training': a, 'saved_continued': b}


def run_seed(seed, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    data, manifest = old.make_data(seed, include_test=False)
    original_manifest = json.loads((RELEASE/f'seed_{seed}/split_manifest_fit.json').read_text())
    if manifest != original_manifest:
        raise ValueError('Fitting examples or observation preprocessing changed')
    ids = [row['ids'] for row in data.values()]
    if len(np.unique(np.concatenate(ids))) != sum(map(len, ids)):
        raise AssertionError('Fitting split IDs overlap')
    dump(out/'split_manifest_fit.json', manifest)
    np.savez_compressed(out/'split_ids_fit.npz', **{k: v['ids'] for k, v in data.items()})
    scalar = load_reference(PRIOR/f'seed_{seed}', 'C_task_only')
    for name in ('encoder', 'heads'):
        freeze(scalar[name])
    source_state = scalar_integrity(scalar)
    # Calibration examples are not needed by any audit or task fitting API.
    data = {k: v for k, v in data.items() if k != 'calibration'}
    prepared, rows = {}, {}
    keys = ['oracle', 'prediction_only', 'exposed_target']
    full = None
    if seed == 2:
        full = load_frozen(RELEASE/'seed_2/E_dual_0.01')
        full_before = integrity(full)
        keys.append('full_E_dual_0.01')
    for key in keys:
        tick = time.perf_counter()
        print(f'seed={seed}: fitting frozen {key}', flush=True)
        directory = out/key
        directory.mkdir()
        if key == 'prediction_only':
            views = prediction_views(scalar, data)
        elif key.startswith('full_'):
            views = full_views(full, data)
        else:
            views = diagnostic_views(data, exposed=key == 'exposed_target')
        hashes = cache(directory/'release_cache', views, data)
        probes = fit_probes(views, data, seed, directory)
        validation = evaluate(probes, views['validation'], data['validation']['y'])
        row = {'validation': validation, 'assessment_validation': assess(validation),
               'selected_family': selected_families(validation),
               'release_dimensions': [h.shape[1] for h in views['validation']],
               'serialized_float32_bytes_per_row': [4*h.shape[1] for h in views['validation']],
               'actual_cached_bytes_per_row': [h.dtype.itemsize*h.shape[1] for h in views['validation']],
               'cached_dtype': str(views['validation'][0].dtype), 'cache_hashes': hashes,
               'probes': {role: {str(p): {family: probe.metadata for family, probe in families.items()}
                                  for p, families in arms.items()} for role, arms in probes.items()},
               'audit_and_task_fit_seconds': time.perf_counter()-tick}
        if key in ('prediction_only', 'oracle'):
            row['direct_task'] = {'validation': direct_scores(views['validation'], data['validation']['y'])}
        if key == 'prediction_only':
            row['provenance'] = scalar['metadata']
            historical = json.loads((RELEASE/f'seed_{seed}/B_prediction_only/validation.json').read_text())
            expected = historical['direct_prediction']['validation']
            delta = {name: row['direct_task']['validation'][name]['r2']-expected[name]['r2'] for name in expected}
            if max(abs(v) for v in delta.values()) > 1e-12:
                raise AssertionError('Exact saved scalar predictions did not replay')
            row['historical_validation_replay_delta'] = delta
        extras = continued_reference(full) if key.startswith('full_') else {}
        if extras:
            row['prior_adversary_references'] = {'validation': {k: a.score(views['validation'], data['validation']['y']) for k, a in extras.items()}}
            row['provenance'] = {str(p.relative_to(ROOT)): file_hash(p) for p in (
                RELEASE/'seed_2/E_dual_0.01/training/final.pt', RELEASE/'seed_2/E_dual_0.01/erasers.npz')}
        dump(directory/'validation.json', row)
        rows[key] = row
        prepared[key] = (probes, views, extras)
        print(f"  validation {key}: feasible={row['assessment_validation']['feasible']}", flush=True)
    if source_state != scalar_integrity(scalar) or (full is not None and full_before != integrity(full)):
        raise AssertionError('Frozen release changed during fitting')
    selection_path = out/'selection_before_test.json'
    dump(selection_path, {'test_generated': False, 'test_rng_seed': CONFIG['test_rng_seeds'][str(seed)],
         'selected': {k: {'family': v['selected_family'], 'probes': v['probes'],
                          'assessment': v['assessment_validation']} for k, v in rows.items()}})
    selection_hash = file_hash(selection_path)
    # No fitting below this boundary.
    final = fresh_test(seed, manifest, selection_path)
    if any(np.intersect1d(final['ids'], split_ids).size for split_ids in ids):
        raise AssertionError('Test IDs overlap a fitting split')
    np.savez_compressed(out/'fresh_test.npz', **final)
    for key, (probes, previous_views, extras) in prepared.items():
        inference_tick = time.perf_counter()
        if key == 'prediction_only':
            views = prediction_views(scalar, {'test': final})
        elif key.startswith('full_'):
            views = full_views(full, {'test': final})
        else:
            views = diagnostic_views({'test': final}, exposed=key == 'exposed_target')
        row = rows[key]
        row['release_inference_seconds_test_4096_rows'] = time.perf_counter()-inference_tick
        row['cache_hashes'].update(cache(out/key/'release_cache', views, {'test': final}))
        audit_tick = time.perf_counter()
        row['test'] = evaluate(probes, views['test'], final['y'])
        row['audit_and_task_inference_seconds_test_4096_rows'] = time.perf_counter()-audit_tick
        row['assessment_test'] = assess(row['test'])
        row['validation_selected_family_test'] = {k: row['test']['attack'][family][k] for k, family in row['selected_family'].items()}
        if 'direct_task' in row:
            row['direct_task']['test'] = direct_scores(views['test'], final['y'])
        if extras:
            row['prior_adversary_references']['test'] = {k: a.score(views['test'], final['y']) for k, a in extras.items()}
        # Recompute representative validation releases and compare all cached views.
        if key == 'prediction_only':
            repeated = prediction_views(scalar, {'validation': data['validation']})['validation']
        elif key.startswith('full_'):
            repeated = full_views(full, {'validation': data['validation']})['validation']
        else:
            repeated = diagnostic_views({'validation': data['validation']}, exposed=key == 'exposed_target')['validation']
        if not all(np.array_equal(a, b) for a, b in zip(repeated, previous_views['validation'])):
            raise AssertionError('Frozen release outputs changed')
        row['frozen_output_unchanged'] = True
        dump(out/key/'metrics.json', row)
    after = scalar_integrity(scalar)
    if source_state != after or (full is not None and full_before != integrity(full)):
        raise AssertionError('Frozen parameters or buffers changed')
    result = {'seed': seed, 'methods': rows, 'frozen_state_before': source_state, 'frozen_state_after': after,
              'frozen_unchanged': True, 'selection_sha256_before_test': selection_hash,
              'selection_sha256_after_test': file_hash(selection_path),
              'test_rng_seed': CONFIG['test_rng_seeds'][str(seed)],
              'test_hashes': {k: old.digest_array(v) for k, v in final.items()},
              'runtime_seconds': time.perf_counter()-started}
    if full is not None:
        result.update(full_state_before=full_before, full_state_after=integrity(full))
    dump(out/'metrics.json', result)
    print(f'seed={seed}: complete {result["runtime_seconds"]:.3f}s', flush=True)


def main():
    from experiments.prediction_release_attackers import ATTACKER_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.out
    if not (out/'PROTOCOL.md').exists():
        raise FileNotFoundError('Write protocol before fitting')
    sources = ['experiments/run_prediction_audit.py', 'experiments/prediction_release_attackers.py',
               'experiments/run_nonlinear_release.py', 'experiments/run_nonlinear_conflict.py',
               'experiments/nonlinear_conflict_training.py', 'experiments/nonlinear_conflict_probes.py',
               'experiments/nonlinear_release_training.py', 'experiments/run_frozen_release_diagnostic.py',
               'experiments/frozen_release_adversaries.py', 'experiments/run_redesign_conflict.py',
               'pcrl/models/lora.py', 'pcrl/training/proxy_lagrangian.py']
    records = {'config.json': {**CONFIG, 'attackers': ATTACKER_CONFIG},
               'frozen_source_hashes.json': {p: file_hash(ROOT/p) for p in sources},
               'protocol_sha256.json': file_hash(out/'PROTOCOL.md')}
    for name, value in records.items():
        path = out/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen evidence changed: {name}')
        if not path.exists():
            dump(path, value)
    for source in sources:
        dest = out/'source_snapshot'/source
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/source).read_bytes())
    invocation = out/('invocation_prepare.json' if args.prepare_only else f'invocation_{"_".join(map(str,args.seeds))}.json')
    if invocation.exists():
        raise FileExistsError('Use a new result directory; preserve completed invocations')
    dump(invocation, {'argv': sys.argv, 'time_ns': time.time_ns()})
    if not args.prepare_only:
        for seed in args.seeds:
            run_seed(seed, out/f'seed_{seed}')


if __name__ == '__main__':
    main()
