"""Read-only replay of new selective training, frozen releases and gradients.

No representation/auditor fitting is run. Historical initial tensors and cached
PCA inputs establish identity; independent literal tensor algebra checks new
release values and diagnostic gradients. Teacher fitting/geometry and candidate
probability/score replay have separate verifiers.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_bottleneck_artifacts import adam_steps
from scripts.verify_acs_pca16_init import load, map_output
from experiments.acs_bottleneck_training import tree_digest

SOURCES = ('income_binary', 'civilian_at_work', 'public_coverage')
ATTRIBUTES = {'SEX': 2, 'RAC1P': 9}


def unit_path(out, unit):
    teacher, rho, seed = unit
    return out / (teacher + '_rho' + ('0p1' if rho == .1 else '0')) / f'seed_{seed}'


def initial_observer_state(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1290000 + 100 * seed)
        observers = torch.nn.ModuleDict({name: torch.nn.Sequential(
            torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32),
            torch.nn.ReLU(), torch.nn.Linear(32, k)) for name, k in ATTRIBUTES.items()})
    return observers.state_dict()


def linear(x, state, name):
    return F.linear(x, state[name + '.weight'], state[name + '.bias'])


def independent_gradients(state, observers, xraw, teacher, source, attributes, priors, rho, protected):
    """Replay gradient vectors from detached checkpoint tensors, never a fitter."""
    state = {name: value.detach().clone().requires_grad_(name.startswith('mapper.')) for name, value in state.items()}
    x = torch.from_numpy(((xraw.astype(np.float64) - state['input_mean'].numpy()) / state['input_scale'].numpy()).astype(np.float32))
    h = linear(torch.relu(linear(x, state, 'mapper.0')), state, 'mapper.2')
    source_losses = []
    for name in SOURCES:
        y = torch.from_numpy(source[name])
        scores = linear(h, state, 'heads.' + name).reshape(-1)
        mask = y >= 0
        source_losses.append(F.binary_cross_entropy_with_logits(scores[mask], y[mask].float()) if mask.any() else scores.sum() * 0.)
    source_loss = torch.stack(source_losses).mean()
    teacher_loss = ((h - torch.from_numpy(np.array(teacher, dtype=np.float32, copy=True))) / state['input_scale'][:16].float()).square().mean()
    reconstruction = F.mse_loss(linear(torch.relu(linear(h, state, 'decoder.0')), state, 'decoder.2'), x)
    adversary_losses = []
    for name in ATTRIBUTES:
        y = torch.from_numpy(attributes[name])
        scores = h
        for layer in ('0', '2', '4'):
            scores = linear(scores, observers, name + '.' + layer)
            if layer != '4':
                scores = torch.relu(scores)
        mask = y >= 0
        ce = F.cross_entropy(scores[mask], y[mask]) if mask.any() else scores.sum() * 0.
        adversary_losses.append(ce / priors[name]['entropy'])
    protection = torch.stack(adversary_losses).mean()
    losses = {'source': source_loss, 'teacher': teacher_loss, 'reconstruction': reconstruction, 'protection': protection}
    parameters = [state[name] for name in ('mapper.0.weight', 'mapper.0.bias', 'mapper.2.weight', 'mapper.2.bias')]
    raw = {name: torch.cat([g.reshape(-1) for g in torch.autograd.grad(value, parameters, retain_graph=True)]) for name, value in losses.items()}
    coefficients = {'source': 1., 'teacher': 1., 'reconstruction': rho, 'protection': -.1 if protected else 0.}
    applied = {name: coefficients[name] * vector for name, vector in raw.items()}
    result = {'source_loss': float(source_loss.detach()), 'preservation_loss': float(teacher_loss.detach()),
              'reconstruction_mse': float(reconstruction.detach()), 'normalized_adversary_ce': float(protection.detach())}
    for convention, vectors in (('raw', raw), ('applied', applied)):
        for name, vector in vectors.items():
            result[f'{name}_{convention}_mapper_l2'] = float(torch.linalg.vector_norm(vector))
        for a, b in (('teacher', 'reconstruction'), ('teacher', 'protection'), ('source', 'protection')):
            dot = float(torch.dot(vectors[a], vectors[b]))
            denominator = result[f'{a}_{convention}_mapper_l2'] * result[f'{b}_{convention}_mapper_l2']
            result[f'{a}_{b}_{convention}_dot'] = dot
            result[f'{a}_{b}_{convention}_cosine'] = dot / denominator if denominator else None
    return result


def teacher_target(directory, name, pca):
    metadata = check.read(directory / 'teachers.json')
    for filename, digest in metadata['files_sha256'].items():
        assert check.sha(directory / filename) == digest
    if name == 'R':
        target = pca[:, :16].copy()
    else:
        with np.load(directory / f'map_{name}.npz') as saved:
            digest = hashlib.sha256(str(saved['metadata_json']).encode())
            for key in ('matrix', 'mean', 'fit_mask', 'known_masks'):
                digest.update(key.encode())
                digest.update(check.array_hash(saved[key]).encode())
            assert digest.hexdigest() == str(saved['map_sha256']) == metadata['maps'][name]['map_sha256']
            target = ((pca[:, :16].astype(np.float64) - saved['mean']) @ saved['matrix'].T + saved['mean']).astype(np.float32)
    assert check.array_hash(target) == metadata['fit_target_sha256'][name]
    return target, metadata


def verify_unit(out, cfg, unit, raw, frozen):
    teacher_name, rho, seed = unit
    directory = unit_path(out, unit)
    training = directory / 'training'
    meta, release_freeze, measured, selected = (check.read(directory / path) for path in
        ('training/training.json', 'release_freeze.json', 'metrics.json', 'selection_before_test.json'))
    assert measured['unit'] == unit and measured['teacher'] == teacher_name and measured['rho'] == rho
    assert meta['module_source_sha256'] == frozen['sha256']['experiments/acs_bottleneck_training.py']
    assert not meta['miniature'] and meta['fit_pool'] == 'representation_fit'
    assert not meta['reserved_labels_received'] and not meta['final_evaluation_received']
    assert meta['source_label_keys'] == list(SOURCES) and meta['attribute_schema'] == ATTRIBUTES
    assert release_freeze['representation_fitting_complete'] and not release_freeze['reserved_labels_received']
    assert release_freeze['created_utc'] < selected['created_utc'] <= measured['integrity']['evaluation_started_utc']
    assert selected['release_freeze_sha256'] == check.sha(directory / 'release_freeze.json')
    assert selected['protocol_freeze_sha256'] == check.sha(out / 'protocol_freeze.json')
    assert all(value for key, value in measured['integrity'].items() if key.endswith('_unchanged'))
    for name, digest in check.read(directory / 'completion.json')['sha256'].items():
        assert check.sha(directory / name) == digest
    for row in check.read(directory / 'local_artifacts.json'):
        assert check.sha(ROOT / row['path']) == row['sha256']
    olddir = ROOT / cfg['init_reference_results'] / f'seed_{seed}'
    oldmeta = check.read(olddir / 'training/training.json')
    with np.load(directory / 'split_rows.npz') as archive, np.load(olddir / 'split_rows.npz') as oldrows:
        rows = {p: archive[p].copy() for p in archive.files}
        assert set(rows) == set(oldrows.files)
        assert all(np.array_equal(value, oldrows[key]) for key, value in rows.items())
    frames = {p: raw.iloc[ix] for p, ix in rows.items()}
    with np.load(ROOT / cfg['reference_results'] / f'seed_{seed}/release_E_pca.npz') as archive:
        pca = {p: archive[p].copy() for p in archive.files}
    teacher, teacher_meta = teacher_target(out / 'static' / f'seed_{seed}/teachers', teacher_name, pca['representation_fit'])
    assert meta['teacher']['sha256'] == check.array_hash(teacher)
    assert meta['teacher']['attribute_label_informed'] == (teacher_name != 'R')
    assert all(meta['teacher'][key] for key in ('detached', 'immutable_verified', 'saved_statistics_supplied', 'saved_statistics_exact_fit_parity'))
    assert meta['teacher_name'] == teacher_name and meta['reconstruction_weight'] == rho
    assert meta['selective_config']['preservation_beta'] == 1. and meta['selective_config']['preservation_schedule'] == 'persistent'
    for key in ('preprocessing', 'schedules', 'adversary_initialization_hash', 'source_label_hashes',
                'attribute_label_hashes', 'source_validation_label_hashes', 'common_optimizer_counts',
                'fit_input_sha256', 'source_validation_input_sha256'):
        assert meta[key] == oldmeta[key], key
    assert {key: value for key, value in meta['config'].items() if key != 'reconstruction_weight'} == {
        key: value for key, value in oldmeta['config'].items() if key != 'reconstruction_weight'}
    assert meta['config']['reconstruction_weight'] == rho
    labels = {}
    for pool, targets, field in (('representation_fit', SOURCES, 'source_label_hashes'),
                                ('representation_fit', ATTRIBUTES, 'attribute_label_hashes'),
                                ('source_validation', SOURCES, 'source_validation_label_hashes')):
        for name in targets:
            y, mask = check.labels(frames[pool], name)
            y = np.where(mask, y, -1).astype(np.int64)
            assert check.array_hash(y) == meta[field][name]
            labels[pool, name] = y
    for name, classes in ATTRIBUTES.items():
        y = labels['representation_fit', name]
        counts = np.bincount(y[y >= 0], minlength=classes).astype(np.float64)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum())
        assert meta['prior_entropies'][name]['support'] == counts.astype(int).tolist()
        assert meta['prior_entropies'][name]['probabilities'] == probabilities.tolist()
        assert meta['prior_entropies'][name]['entropy'] == entropy
    initial, warm_base, warm = (load(training / name) for name in ('initialization.pt', 'warm_base.pt', 'warm_adversary.pt'))
    assert tree_digest(initial) == tree_digest(load(olddir / 'training/initialization.pt'))
    state = initial['model_state']
    fit = pca['representation_fit'].astype(np.float64)
    mean, scale = fit.mean(0), fit.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    np.testing.assert_array_equal(state['input_mean'].numpy(), mean)
    np.testing.assert_array_equal(state['input_scale'].numpy(), scale)
    np.testing.assert_array_equal(meta['teacher']['coordinate_scale'], scale[:16])
    np.testing.assert_array_equal(teacher_meta['original_scale'], scale[:16])
    assert meta['teacher']['coordinate_scale_sha256'] == check.array_hash(scale[:16])
    n, batches = len(fit), int(np.ceil(len(fit) / 256))
    for phase, epochs, offset in (('warm_base', 60, 0), ('warm_adversary', 20, 100), ('continuation', 80, 200)):
        schedule = meta['schedules'][phase]
        assert schedule['seed'] == 1280000 + 100 * seed + offset
        rng, digest = np.random.default_rng(schedule['seed']), hashlib.sha256()
        for _ in range(epochs):
            digest.update(rng.permutation(n).tobytes())
        assert digest.hexdigest() == schedule['sha256']
    idx = np.random.default_rng(meta['schedules']['warm_base']['seed']).permutation(n)[:256]
    assert check.array_hash(idx) == meta['gradient_diagnostic_indices_sha256']
    assert check.state_hash(warm_base['model_state']) == check.state_hash(warm['model_state']) == meta['snapshots']['W']['model_hash']
    assert meta['snapshots']['W']['unchanged_across_adversary_warmup']
    assert check.state_hash(initial['model_state']) == meta['snapshots']['I']['model_hash']
    assert tree_digest(warm_base['mapper_optimizer_state']) == tree_digest(warm['mapper_optimizer_state'])
    adam_steps(warm_base['mapper_optimizer_state'], 60 * batches)
    adam_steps(warm['adversary_optimizer_state'], 20 * batches)
    assert meta['common_base_row_exposures'] == n * 60 and meta['common_adversary_row_exposures'] == n * 20
    initial_observers = initial_observer_state(seed)
    assert check.state_hash(initial_observers) == meta['adversary_initialization_hash']
    forks = {arm: load(training / arm / 'fork.pt') for arm in ('C', 'D')}
    assert tree_digest(forks['C']) == tree_digest(forks['D'])
    checkpoints = {'I': initial, 'W': warm_base}
    points = [(initial, initial_observers, False, meta['stage_gradient_diagnostics']['initialization']),
              (warm_base, initial_observers, False, meta['stage_gradient_diagnostics']['after_base_warmup']),
              (warm, warm['adversary_state'], True, meta['stage_gradient_diagnostics']['shared_fork'])]
    for arm in ('C', 'D'):
        final = load(training / arm / 'final.pt')
        checkpoints[arm] = final
        record = meta['arms'][arm]
        assert record['fork_hashes'] == meta['shared_fork_hashes']
        historical_arm = 'C_bottleneck' if arm == 'C' else 'D_protected'
        for field in ('schedule_hash', 'continuation_mapper_optimizer_steps', 'continuation_adversary_optimizer_steps',
                      'optimizer_counts_including_common', 'mapper_row_exposures', 'adversary_row_exposures'):
            assert record[field] == oldmeta['arms'][historical_arm][field]
        for key, label in (('model_state', 'model'), ('adversary_state', 'adversaries'),
                           ('mapper_optimizer_state', 'mapper_optimizer'), ('adversary_optimizer_state', 'adversary_optimizer')):
            digest = tree_digest if 'optimizer' in key else check.state_hash
            assert digest(forks[arm][key]) == digest(warm[key]) == meta['shared_fork_hashes'][label]
        assert final['counters'] == {'mapper_optimizer_steps': 140 * batches, 'adversary_optimizer_steps': 260 * batches, 'continuation_epoch': 80}
        adam_steps(final['mapper_optimizer_state'], 140 * batches)
        adam_steps(final['adversary_optimizer_state'], 260 * batches)
        assert record['selected_epoch'] == 80 and record['continuation_preservation_coefficient'] == 1.
        assert record['mapper_loss_uses_protection_gradient'] == (arm == 'D')
        for key, field in (('model_state', 'final_model_hash'), ('adversary_state', 'final_adversary_hash')):
            assert check.state_hash(final[key]) == record[field]
        for key, field in (('mapper_optimizer_state', 'final_mapper_optimizer_hash'), ('adversary_optimizer_state', 'final_adversary_optimizer_hash')):
            assert tree_digest(final[key]) == record[field]
        # Fixed architecture parameter order is mapper (4), source heads (6),
        # decoder (4); do not trust an empty recorded exclusion list.
        assert record['decoder_optimizer_parameter_indices'] == [10, 11, 12, 13]
        assert final['mapper_optimizer_state']['param_groups'][0]['params'] == list(range(14))
        if rho == 0.:
            assert record['decoder_unchanged'] and record['decoder_gradients_absent'] and record['decoder_optimizer_state_entries'] == 0
            for checkpoint in (warm_base, warm, forks[arm], final):
                assert all(key not in checkpoint['mapper_optimizer_state']['state'] for key in record['decoder_optimizer_parameter_indices'])
                assert all(torch.equal(value, checkpoint['model_state'][key]) for key, value in state.items() if key.startswith('decoder.'))
        else:
            assert record['decoder_optimizer_state_entries'] == 4
        points.extend([(forks[arm], forks[arm]['adversary_state'], arm == 'D', record['fixed_batch_gradient_diagnostics_at_shared_fork']),
                       (final, final['adversary_state'], arm == 'D', record['fixed_batch_gradient_diagnostics_at_final'])])
        for curve in record['curve']:
            assert curve['mapper_optimizer_steps'] == (60 + curve['epoch']) * batches
            assert curve['adversary_optimizer_steps'] == (20 + 3 * curve['epoch']) * batches
            assert curve['preservation_coefficient'] == 1. and curve['reconstruction_coefficient'] == rho
            assert curve['protection_coefficient'] == (-.1 if arm == 'D' else 0.)
    maximum_gradient_error = 0.
    for checkpoint, observers, protected, recorded in points:
        actual = independent_gradients(checkpoint['model_state'], observers, pca['representation_fit'][idx], teacher[idx],
            {name: labels['representation_fit', name][idx] for name in SOURCES},
            {name: labels['representation_fit', name][idx] for name in ATTRIBUTES}, meta['prior_entropies'], rho, protected)
        assert recorded['training_state_unchanged'] and recorded['torch_rng_unchanged'] and recorded['optimizer_steps'] == 0
        assert recorded['diagnostic_observer_state_sha256'] == check.state_hash(observers)
        assert recorded['coefficients'] == {'source': 1., 'teacher': 1., 'reconstruction': rho,
                                             'protection': -.1 if protected else 0.}
        assert recorded['preservation_nonmapper_gradients_all_absent']
        for name, value in actual.items():
            if value is None:
                assert recorded[name] is None
            else:
                error = abs(value - recorded[name])
                maximum_gradient_error = max(error, maximum_gradient_error)
                assert np.isclose(value, recorded[name], rtol=2e-6, atol=1e-9), (unit, name, value, recorded[name])
    arrays_replayed = 0
    for arm, checkpoint in checkpoints.items():
        assert check.state_hash(checkpoint['model_state']) == release_freeze['state']['models'][arm]
        if arm in ('C', 'D'):
            assert check.state_hash(checkpoint['adversary_state']) == release_freeze['state']['adversaries'][arm]
        with np.load(directory / f'release_{arm}.npz') as arrays:
            for pool in arrays.files:
                np.testing.assert_array_equal(map_output(checkpoint['model_state'], pca[pool]), arrays[pool])
                recorded_hash = (measured['integrity']['evaluation_output_hashes'][arm] if pool == 'test'
                                 else release_freeze['output_hashes'][arm][pool])
                assert check.array_hash(arrays[pool]) == recorded_hash
                if arm == 'I':
                    np.testing.assert_allclose(arrays[pool], pca[pool][:, :16], atol=1e-5, rtol=1e-5)
                arrays_replayed += 1
    return {'unit': unit, 'passed': True, 'exact_initialization_and_C_D_fork': True,
            'fixed_schedules_labels_and_original_scales': True, 'frozen_teacher_targets_exact': True,
            'decoder_updates_omitted_when_rho_zero': rho == 0., 'all_Adam_steps_checked': True,
            'new_release_arrays_replayed': arrays_replayed, 'independent_gradient_points_replayed': len(points),
            'maximum_gradient_scalar_error': maximum_gradient_error,
            'training_json_sha256': check.sha(training / 'training.json'),
            'metrics_sha256': check.sha(directory / 'metrics.json')}


def verify(out):
    started = time.perf_counter()
    cfg, freeze = (check.read(out / name) for name in ('config.json', 'protocol_freeze.json'))
    for group in ('sha256', 'reference_record_hashes'):
        for path, digest in freeze[group].items():
            assert check.sha(ROOT / path) == digest
    progress = check.read(out / 'progress.json')
    complete = [unit for unit in progress['completed_units'] if unit[0] != 'static']
    assert complete, 'No completed learned units to replay'
    parent = ROOT / cfg['parent_results']
    raw_path = ROOT / check.read(parent / 'config.json')['raw_path']
    assert check.sha(raw_path) == check.read(parent / 'schema_support.json')['raw_sha256']
    raw = pd.read_csv(raw_path, usecols=['PINCP', 'ESR', 'PUBCOV', 'SEX', 'RAC1P'])
    reports = [verify_unit(out, cfg, unit, raw, freeze) for unit in complete]
    return {'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'New learned unit checkpoint/Adam/release and independent diagnostic gradient replay only; no fitting or historical scientific inference',
            'script_sha256': check.sha(__file__), 'completed_learned_units': len(reports),
            'expected_learned_units': 15, 'complete_matrix': len(reports) == 15, 'units': reports,
            'new_release_arrays_replayed': sum(r['new_release_arrays_replayed'] for r in reports),
            'independent_gradient_points_replayed': sum(r['independent_gradient_points_replayed'] for r in reports),
            'maximum_gradient_scalar_error': max(r['maximum_gradient_scalar_error'] for r in reports),
            'runtime_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/redesign_20260908_acs_selective_preservation_v1')
    parser.add_argument('--report', type=Path, help='Fresh report; stdout if omitted')
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError('Preserve completed replay evidence')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = verify(args.out.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.report:
        with args.report.open('x') as handle:
            handle.write(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
