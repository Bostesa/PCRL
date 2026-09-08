"""Supplement: frozen three-probability banks to16 teacher coordinates.

Exactly18 fitting-only affine decoders complete the label-free geometry scope
for all six published source banks. No hidden bank representation is read.
Preparation freezes this source and all inputs before fitting; every coefficient
is frozen before source-validation/development inputs are evaluated.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.acs_selective_diagnostics import components, _rank, RCOND, VARIANCE_FLOOR
from experiments.acs_selective_teachers import load_teachers
from experiments.acs_transfer_data import array_hash

DEFAULT_STUDY = ROOT / 'results/redesign_20260908_acs_restricted_inputs_v1'
TARGETS = ('raw', 'E', 'qE')
UNITS = [(teacher, seed) for teacher in ('E', 'S') for seed in (0, 1, 2)]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path, value):
    with Path(path).open('x') as handle: handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def feature_matrix(value):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != 3 or value.dtype != np.float32 or len(value) < 2 or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError('Require the published finite float32 three-probability bank')
    return value.astype(np.float64)


def fit(bank, target):
    x, y = feature_matrix(bank), np.asarray(target, np.float64)
    if y.shape != (len(x), 16) or not np.isfinite(y).all(): raise ValueError('Aligned standardized16 teacher target required')
    design = np.column_stack((x, np.ones(len(x))))
    coefficient, _, rank, singular = np.linalg.lstsq(design, y, rcond=RCOND)
    prior, variance = y.mean(0), np.var(y, axis=0)
    metadata = {'fit_pool': 'representation_fit', 'fit_rows': len(x), 'input_dimension': 3, 'target_dimension': 16,
        'intercept_fitted': True, 'selection': 'none', 'rcond': RCOND, 'rank': int(rank), 'design_rank': int(rank),
        'singular_values': singular.tolist(), 'design_singular_values': singular.tolist(),
        'release_centered_rank': _rank(x - x.mean(0)), 'target_centered_rank': _rank(y - prior), 'target_uncentered_rank': _rank(y),
        'fit_prior': prior.tolist(), 'prior': prior.tolist(), 'fit_variance': variance.tolist(),
        'fit_target_variance': variance.tolist(), 'fit_target_variance_mean': float(variance.mean()),
        'fit_release_sha256': array_hash(x), 'fit_target_sha256': array_hash(y), 'coefficient_sha256': array_hash(coefficient),
        'target_scale': 'original raw PCA16 fitting scales; never target/residual variance', 'variance_floor': VARIANCE_FLOOR}
    return {'coefficient': coefficient, 'prior': prior, 'fit_variance': variance, 'metadata': metadata}


def evaluate(fitted, bank, target):
    x, y = feature_matrix(bank), np.asarray(target, np.float64)
    if y.shape != (len(x), 16) or not np.isfinite(y).all(): raise ValueError('Aligned standardized16 teacher target required')
    prediction = np.column_stack((x, np.ones(len(x)))) @ fitted['coefficient']
    mse = np.mean((prediction - y) ** 2, axis=0)
    prior_mse = np.mean((y - fitted['prior']) ** 2, axis=0)
    variance = np.var(y, axis=0)
    defined = (fitted['fit_variance'] > VARIANCE_FLOOR) & (variance > VARIANCE_FLOOR) & (prior_mse > VARIANCE_FLOOR)
    mean_defined = float(fitted['fit_variance'].mean()) > VARIANCE_FLOOR and float(variance.mean()) > VARIANCE_FLOOR and float(prior_mse.mean()) > VARIANCE_FLOOR
    return {'rows': len(x), 'mean_mse': float(mse.mean()), 'per_coordinate_mse': mse.tolist(),
        'prior_mean_mse': float(prior_mse.mean()), 'prior_per_coordinate_mse': prior_mse.tolist(),
        'target_variance_mean': float(variance.mean()), 'target_variance_per_coordinate': variance.tolist(),
        'mse_over_prior': float(mse.mean() / prior_mse.mean()) if mean_defined else None,
        'per_coordinate_mse_over_prior': [float(a / b) if flag else None for a, b, flag in zip(mse, prior_mse, defined)],
        'ratio_defined': bool(mean_defined), 'per_coordinate_ratio_defined': defined.tolist(), 'variance_floor': VARIANCE_FLOOR,
        'ratio_rule': 'defined only if fitting/evaluation target variance and evaluation fitting-prior MSE all exceed1e-12',
        'evaluation_target_centered_rank': _rank(y - y.mean(0))}


def inputs(study):
    cfg = read(study / 'config.json');paths = [study / 'config.json', study / 'PROTOCOL.md', study / 'protocol_freeze.json']
    for teacher, seed in UNITS:
        unit = study / f'{teacher}_bank' / f'seed_{seed}'
        assert (unit / 'completion.json').exists(), 'All six predetermined banks must finish before supplement preparation'
        manifest = {e['path']: e['sha256'] for e in read(unit / 'local_artifacts.json')}
        for name in ('release_B.npz', 'split_rows.npz'):
            path = unit / name
            assert sha(path) == manifest[str(path.relative_to(ROOT))]
            paths.append(path)
        paths.extend(unit / name for name in ('completion.json', 'training/training.json', 'release_freeze.json', 'local_artifacts.json'))
    for seed in (0, 1, 2):
        paths.append(ROOT / cfg['reference_results'] / f'seed_{seed}/release_E_pca.npz')
        paths.append(ROOT / cfg['init_reference_results'] / f'seed_{seed}/training/training.json')
        teacher_dir = ROOT / cfg['selective_reference_results'] / f'static/seed_{seed}/teachers'
        teacher_meta = read(teacher_dir / 'teachers.json')
        paths.append(teacher_dir / 'teachers.json')
        for name, digest in teacher_meta['files_sha256'].items():
            assert sha(teacher_dir / name) == digest
            paths.append(teacher_dir / name)
    return list(dict.fromkeys(paths))


def prepare(study):
    out = study / 'bank_geometry'
    assert out.is_dir() and (out / 'BANK_GEOMETRY_SCOPE.md').exists()
    cfg = {'study': 'acs_restricted_source_bank_affine_geometry_supplement', 'units': [list(u) for u in UNITS],
           'targets': list(TARGETS), 'number_of_fits': 18, 'input_dimension': 3, 'target_dimension': 16,
           'fit_pool': 'representation_fit', 'evaluation_pools': ['source_validation', 'test'],
           'rcond': RCOND, 'variance_floor': VARIANCE_FLOOR, 'numerical_threads': 1,
           'intercept': 'fitting-only', 'target_scale': 'original saved PCA16 coordinate scales',
           'selection': 'none', 'all_coefficients_frozen_before_heldout_evaluation': True,
           'direct_coordinate_error': None, 'direct_coordinate_error_reason': 'three-probability inputs and16-coordinate targets have different dimensions',
           'hidden_bank_mapper_release_received': False, 'outcome_labels_received': False}
    write(out / 'config.json', cfg)
    source = [Path(__file__), ROOT / 'experiments/acs_selective_diagnostics.py', ROOT / 'experiments/acs_selective_teachers.py',
              ROOT / 'experiments/acs_protection_maps.py', ROOT / 'experiments/acs_transfer_data.py', out / 'config.json', out / 'BANK_GEOMETRY_SCOPE.md']
    write(out / 'protocol_freeze.json', {'created_utc': now(), 'any_fits_started': False,
        'source_and_config_sha256': {str(p.relative_to(ROOT)): sha(p) for p in source},
        'required_input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs(study)},
        'original_protocol_freeze_sha256': sha(study / 'protocol_freeze.json'),
        'scope': 'Coverage completion for all six published bank releases; original protocols and completed unit artifacts remain unchanged'})


def load_pool(study, cfg, teacher, seed, pool):
    with np.load(study / f'{teacher}_bank/seed_{seed}/release_B.npz') as archive: bank = archive[pool].copy()
    with np.load(ROOT / cfg['reference_results'] / f'seed_{seed}/release_E_pca.npz') as archive: raw = archive[pool][:, :16].copy()
    return bank, raw


def run(study):
    started = time.perf_counter();out = study / 'bank_geometry';freeze = read(out / 'protocol_freeze.json')
    for group in ('source_and_config_sha256', 'required_input_sha256'):
        for path, digest in freeze[group].items(): assert sha(ROOT / path) == digest
    cfg = read(study / 'config.json');subcfg = read(out / 'config.json')
    assert subcfg['units'] == [list(u) for u in UNITS] and subcfg['targets'] == list(TARGETS)
    artifact_dir = out / 'fitted';artifact_dir.mkdir(exist_ok=False)
    fits, metadata, contexts = {}, {}, {}
    for teacher, seed in UNITS:
        key = f'{teacher}_bank/seed_{seed}'
        maps = load_teachers(ROOT / cfg['selective_reference_results'] / f'static/seed_{seed}/teachers')['maps']
        pre = read(ROOT / cfg['init_reference_results'] / f'seed_{seed}/training/training.json')['preprocessing']
        current = read(study / f'{teacher}_bank/seed_{seed}/training/training.json')['preprocessing']
        assert pre['mean'] == current['mean'] and pre['scale'] == current['scale']
        contexts[key] = (maps, pre)
        # Only fitting arrays are extracted before all18 coefficients freeze.
        bank, raw = load_pool(study, cfg, teacher, seed, 'representation_fit')
        targets = components(raw, maps, pre)
        fits[key], metadata[key] = {}, {'unit': [teacher, 'bank', seed], 'teacher': teacher, 'seed': seed,
            'release': 'B', 'reported_release': teacher + '_bank', 'input_dimension': 3, 'targets': {},
            'direct_coordinate_error': None, 'direct_coordinate_error_reason': subcfg['direct_coordinate_error_reason']}
        for target in TARGETS:
            fitted = fit(bank, targets[target]);fits[key][target] = fitted
            name = f'fitted/{teacher}_bank_seed_{seed}_{target}.npz'
            with (out / name).open('xb') as handle:
                np.savez_compressed(handle, coefficient=fitted['coefficient'], prior=fitted['prior'], fit_variance=fitted['fit_variance'])
            metadata[key]['targets'][target] = {**fitted['metadata'], 'artifact': name, 'artifact_sha256': sha(out / name)}
    assert sum(len(v) for v in fits.values()) == 18
    write(out / 'affine_freeze.json', {'created_utc': now(), 'heldout_evaluation_started': False,
        'protocol_freeze_sha256': sha(out / 'protocol_freeze.json'), 'number_of_fits': 18, 'snapshots': metadata,
        'files_sha256': {str(p.relative_to(out)): sha(p) for p in artifact_dir.glob('*.npz')}})
    affine_hash, evaluation_started = sha(out / 'affine_freeze.json'), now()
    for teacher, seed in UNITS:
        key = f'{teacher}_bank/seed_{seed}';maps, pre = contexts[key]
        for pool, label in (('representation_fit', 'fit'), ('source_validation', 'source_validation'), ('test', 'development_evaluation')):
            bank, raw = load_pool(study, cfg, teacher, seed, pool);targets = components(raw, maps, pre)
            for target in TARGETS: metadata[key]['targets'][target][label] = evaluate(fits[key][target], bank, targets[target])
    assert sha(out / 'affine_freeze.json') == affine_hash
    for path, digest in read(out / 'affine_freeze.json')['files_sha256'].items(): assert sha(out / path) == digest
    report = {'created_utc': now(), 'evaluation_started_utc': evaluation_started, 'snapshots': metadata,
        'rcond': RCOND, 'variance_floor': VARIANCE_FLOOR, 'fit_pool': 'representation_fit', 'selection': 'none',
        'number_of_fits': 18, 'input_dimension': 3, 'target_dimension': 16, 'bank_interface': 'published three source probabilities only',
        'original_scales': True, 'hidden_bank_mapper_release_received': False, 'outcome_labels_received': False,
        'affine_freeze_sha256': affine_hash, 'protocol_freeze_sha256': sha(out / 'protocol_freeze.json'),
        'scope': 'Affine recovery from final three-probability banks; no coordinate matching, no hidden16-interface, no selection, and no privacy guarantee',
        'removed_component_scope': 'qE is not pure sensitive information; high affine error does not exclude nonlinear recovery',
        'runtime_seconds': time.perf_counter() - started}
    write(out / 'geometry.json', report)
    write(out / 'local_artifacts.json', [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p), 'availability': 'local only'} for p in artifact_dir.glob('*.npz')])
    write(out / 'completion.json', {'created_utc': now(), 'number_of_fits': 18,
        'sha256': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}})
    return {'complete': True, 'number_of_fits': 18, 'runtime_seconds': report['runtime_seconds'], 'geometry_sha256': sha(out / 'geometry.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--prepare', action='store_true');parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.prepare: prepare(args.study.resolve())
        if args.run: print(json.dumps(run(args.study.resolve())), flush=True)


if __name__ == '__main__': main()
