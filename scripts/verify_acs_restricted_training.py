"""Independent read-only replay of restricted-input initialization and training.

Checks all saved full-state forks, Adam counters and zero-block invariants;
reconstructs fixed-batch gradient diagnostics with literal tensor operations.
The separate audit verifier replays published releases, native heads, candidate
scores and teacher-only composed attacks. No fitting is performed here.
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
from scripts.verify_acs_pca16_init import load
from scripts.verify_acs_selective_training import teacher_target, initial_observer_state, linear
from experiments.acs_bottleneck_training import tree_digest

SOURCES = ('income_binary', 'civilian_at_work', 'public_coverage')
ATTRIBUTES = {'SEX': 2, 'RAC1P': 9}


def prepared(state, teacher, raw, access):
    mean, scale = state['input_mean'].numpy(), state['input_scale'].numpy()
    u = ((teacher.astype(np.float64) - mean[:16]) / scale[:16]).astype(np.float32)
    v = ((raw.astype(np.float64) - mean) / scale).astype(np.float32) if access == 'F' else np.zeros((len(u), 32), np.float32)
    return np.concatenate((u, v), 1)


def diagnostic(state, observer_state, x, teacher, source, attributes, priors, beta, protected):
    """Literal independent forward/autograd replay from disposable saved tensors."""
    state = {name: value.detach().clone().requires_grad_(name.startswith('mapper.')) for name, value in state.items()}
    x = torch.from_numpy(x)
    h = linear(torch.relu(linear(x, state, 'mapper.0')), state, 'mapper.2')
    pieces = []
    for name in SOURCES:
        y = torch.from_numpy(source[name])
        scores = linear(h, state, 'heads.' + name).reshape(-1)
        known = y >= 0
        pieces.append(F.binary_cross_entropy_with_logits(scores[known], y[known].float()) if known.any() else scores.sum() * 0.)
    losses = {'source': torch.stack(pieces).mean(),
              'teacher': ((h - torch.from_numpy(teacher.copy())) / state['input_scale'][:16].float()).square().mean()}
    if observer_state:
        pieces = []
        for name in ATTRIBUTES:
            scores = h
            for layer in ('0', '2', '4'):
                scores = linear(scores, observer_state, name + '.' + layer)
                if layer != '4': scores = torch.relu(scores)
            y = torch.from_numpy(attributes[name]);known = y >= 0
            ce = F.cross_entropy(scores[known], y[known]) if known.any() else scores.sum() * 0.
            pieces.append(ce / priors[name]['entropy'])
        losses['protection'] = torch.stack(pieces).mean()
    coefficients = {'source': 1., 'teacher': beta, 'protection': -.1 if protected else 0.}
    result = {'source_loss': float(losses['source'].detach()), 'preservation_loss': float(losses['teacher'].detach()),
              'normalized_adversary_ce': float(losses['protection'].detach()) if observer_state else None,
              'input_teacher_l2': float(torch.linalg.vector_norm(x[:, :16])), 'input_raw_l2': float(torch.linalg.vector_norm(x[:, 16:])),
              'teacher_input_weight_l2': float(torch.linalg.vector_norm(state['mapper.0.weight'][:, :16]).detach()),
              'raw_input_weight_l2': float(torch.linalg.vector_norm(state['mapper.0.weight'][:, 16:]).detach())}
    params = [state[name] for name in ('mapper.0.weight', 'mapper.0.bias', 'mapper.2.weight', 'mapper.2.bias')]
    vectors = {}
    for name, value in losses.items():
        gradients = torch.autograd.grad(value, params, retain_graph=True)
        vectors[name] = torch.cat([g.reshape(-1) for g in gradients])
        for kind, coefficient in (('raw', 1.), ('applied', coefficients[name])):
            for block, vector in (('mapper', vectors[name]), ('teacher_input_weight', gradients[0][:, :16]), ('raw_input_weight', gradients[0][:, 16:])):
                result[f'{name}_{kind}_{block}_l2'] = float(torch.linalg.vector_norm(coefficient * vector))
    for name in ('source', 'teacher', 'protection'):
        if name not in losses:
            for kind in ('raw', 'applied'):
                for block in ('mapper', 'teacher_input_weight', 'raw_input_weight'): result[f'{name}_{kind}_{block}_l2'] = None
    for left, right in (('teacher', 'protection'), ('source', 'protection'), ('source', 'teacher')):
        for kind in ('raw', 'applied'):
            if left not in vectors or right not in vectors:
                dot, cosine = None, None
            else:
                a, b = vectors[left], vectors[right]
                if kind == 'applied': a, b = coefficients[left] * a, coefficients[right] * b
                dot = float(torch.dot(a, b))
                denominator = float(torch.linalg.vector_norm(a)) * float(torch.linalg.vector_norm(b))
                cosine = dot / denominator if denominator else None
            result[f'{left}_{right}_{kind}_dot'] = dot
            result[f'{left}_{right}_{kind}_cosine'] = cosine
    return result


def verify_unit(out, cfg, unit, raw, frozen, matched_initial):
    teacher_name, access, seed = unit;bank = access == 'bank';beta = 0. if bank else 1.
    directory = out / f'{teacher_name}_{access}' / f'seed_{seed}';training = directory / 'training'
    meta = check.read(training / 'training.json');metrics = check.read(directory / 'metrics.json')
    assert meta['module_source_sha256'] == frozen['sha256']['experiments/acs_restricted_training.py']
    assert meta['historical_helper_source_sha256'] == check.sha(ROOT / 'experiments/acs_bottleneck_training.py')
    assert metrics['unit'] == unit and meta['bank'] == bank and meta['access'] == ('K' if bank else access)
    assert meta['fit_pool'] == 'representation_fit' and not meta['miniature']
    assert not meta['reserved_labels_received'] and not meta['final_evaluation_received']
    assert meta['raw_inputs_received'] == (access == 'F') and meta['source_label_keys'] == list(SOURCES)
    assert meta['attribute_schema'] == ({} if bank else ATTRIBUTES)
    assert meta['config']['preservation_beta'] == beta and meta['config']['reconstruction_weight'] == 0.
    for key, expected in (('input_dim', 48), ('mapper_hidden', 64), ('release_dim', 16), ('warm_base_epochs', 60),
                           ('warm_adversary_epochs', 20), ('continuation_epochs', 80), ('batch_size', 256), ('adversary_updates_per_mapper_step', 3)):
        assert meta['config'][key] == expected
    for path, digest in check.read(directory / 'completion.json')['sha256'].items(): assert check.sha(directory / path) == digest
    for entry in check.read(directory / 'local_artifacts.json'): assert check.sha(ROOT / entry['path']) == entry['sha256']
    olddir = ROOT / cfg['init_reference_results'] / f'seed_{seed}'
    oldmeta = check.read(olddir / 'training/training.json')
    with np.load(directory / 'split_rows.npz') as archive, np.load(olddir / 'split_rows.npz') as older:
        rows = {name: archive[name].copy() for name in archive.files}
        assert set(rows) == set(older.files)
        assert all(np.array_equal(value, older[name]) for name, value in rows.items())
    with np.load(ROOT / cfg['reference_results'] / f'seed_{seed}/release_E_pca.npz') as archive:
        pca = {name: archive[name].copy() for name in archive.files}
    teacher, teacher_meta = teacher_target(ROOT / cfg['selective_reference_results'] / f'static/seed_{seed}/teachers', teacher_name, pca['representation_fit'])
    assert meta['teacher']['sha256'] == meta['fit_input_sha256'] == check.array_hash(teacher)
    assert meta['teacher']['immutable_verified'] and meta['teacher']['detached']
    assert meta['source_label_hashes'] == oldmeta['source_label_hashes'] and meta['source_validation_label_hashes'] == oldmeta['source_validation_label_hashes']
    labels = {}
    for pool, names, field in (('representation_fit', SOURCES, 'source_label_hashes'), ('source_validation', SOURCES, 'source_validation_label_hashes'),
                               ('representation_fit', () if bank else ATTRIBUTES, 'attribute_label_hashes')):
        for name in names:
            y, known = check.labels(raw.iloc[rows[pool]], name)
            y = np.where(known, y, -1).astype(np.int64)
            assert check.array_hash(y) == meta[field][name]
            labels[pool, name] = y
    if bank:
        assert meta['attribute_label_hashes'] == meta['prior_entropies'] == meta['attribute_fit_coverage'] == {}
    else:
        assert meta['attribute_label_hashes'] == oldmeta['attribute_label_hashes'] and meta['prior_entropies'] == oldmeta['prior_entropies']
    initial, warm = load(training / 'initialization.pt'), load(training / 'warm_base.pt')
    original = load(olddir / 'training/initialization.pt')['model_state'];state = initial['model_state']
    digest = check.state_hash(state)
    if seed in matched_initial: assert matched_initial[seed] == digest
    else: matched_initial[seed] = digest
    assert digest == meta['initialization_hashes']['model'] == meta['snapshots']['I']['model_hash']
    assert not initial['mapper_optimizer_state']['state'] and initial['adversary_state'] == {} and initial['adversary_optimizer_state'] is None
    for key, value in original.items():
        if not key.startswith('mapper.'): assert torch.equal(state[key], value)
    assert check.state_hash({k[len('heads.'):]: v for k, v in state.items() if k.startswith('heads.')}) == meta['initialization']['heads_initial_sha256']
    q, r = np.linalg.qr(np.random.default_rng(20260909 + seed).standard_normal((16, 16)))
    q *= np.where(np.diag(r) < 0., -1., 1.)[None, :]
    with np.load(out / f'initial_Q_seed_{seed}.npz') as saved: np.testing.assert_array_equal(q, saved['Q'])
    np.testing.assert_array_equal(state['orthogonal_q'].numpy(), q)
    assert check.array_hash(q) == meta['initialization']['q_sha256']
    np.testing.assert_array_equal(state['mapper.0.weight'].numpy()[:, :16], np.concatenate((np.eye(16), -np.eye(16), q, -q)).astype(np.float32))
    np.testing.assert_array_equal(state['mapper.0.weight'].numpy()[:, 16:], np.zeros((64, 32)))
    np.testing.assert_array_equal(state['mapper.0.bias'].numpy(), np.tile(np.concatenate((np.ones(16), -np.ones(16))), 2))
    s = state['input_scale'][:16].float().numpy()
    np.testing.assert_array_equal(state['mapper.2.weight'].numpy(), np.concatenate((np.diag(s), -np.diag(s), np.zeros((16, 32))), 1).astype(np.float32))
    np.testing.assert_array_equal(state['mapper.2.bias'].numpy(), state['input_mean'][:16].float().numpy() - s)
    np.testing.assert_array_equal(meta['teacher']['coordinate_scale'], state['input_scale'][:16].numpy())
    np.testing.assert_array_equal(teacher_meta['original_scale'], state['input_scale'][:16].numpy())
    x = prepared(state, teacher, pca['representation_fit'], access)
    assert check.array_hash(x) == meta['standardized_fit_sha256']
    if access == 'F': assert check.array_hash(pca['representation_fit']) == meta['raw_fit_input_sha256']
    else: assert meta['raw_fit_input_sha256'] is None and meta['raw_source_validation_input_sha256'] is None
    n, batches = len(x), int(np.ceil(len(x) / 256))
    for phase, epochs, offset in (('warm_base', 60, 0), ('warm_adversary', 20, 100), ('continuation', 80, 200)):
        schedule = meta['schedules'][phase]
        assert schedule == oldmeta['schedules'][phase] and schedule['seed'] == 1280000 + 100 * seed + offset
        rng, digest = np.random.default_rng(schedule['seed']), hashlib.sha256()
        for _ in range(epochs): digest.update(rng.permutation(n).tobytes())
        assert digest.hexdigest() == schedule['sha256']
    idx = np.random.default_rng(meta['schedules']['warm_base']['seed']).permutation(n)[:256]
    assert check.array_hash(idx) == meta['gradient_diagnostic_indices_sha256']
    adam_steps(warm['mapper_optimizer_state'], 60 * batches)
    assert meta['common_base_row_exposures'] == n * 60 and meta['common_adversary_row_exposures'] == (0 if bank else n * 20)
    assert meta['native_source_head_training_exposure']['mapper_epochs'] == 140
    for name in SOURCES:
        assert meta['native_source_head_training_exposure']['valid_label_exposures'][name] == int((labels['representation_fit', name] >= 0).sum()) * 140
    if bank:
        shared = warm;initial_observers = {}
        assert not (training / 'warm_adversary.pt').exists() and meta['adversary_initialization_hash'] is None
    else:
        shared = load(training / 'warm_adversary.pt');initial_observers = initial_observer_state(seed)
        assert meta['adversary_initialization_hash'] == check.state_hash(initial_observers) == oldmeta['adversary_initialization_hash']
        assert tree_digest((warm['model_state'], warm['mapper_optimizer_state'])) == tree_digest((shared['model_state'], shared['mapper_optimizer_state']))
        adam_steps(shared['adversary_optimizer_state'], 20 * batches)
    assert check.state_hash(warm['model_state']) == meta['snapshots']['W']['model_hash']
    points = [(initial, initial_observers, False, meta['stage_gradient_diagnostics']['initialization']),
              (warm, initial_observers, False, meta['stage_gradient_diagnostics']['after_base_warmup']),
              (shared, shared['adversary_state'], not bank, meta['stage_gradient_diagnostics']['shared_fork'])]
    stages = [initial, warm] + ([] if bank else [shared]);fork_hashes = []
    for arm in (('B',) if bank else ('C', 'D')):
        fork, final = load(training / arm / 'fork.pt'), load(training / arm / 'final.pt');record = meta['arms'][arm]
        fork_hashes.append(tree_digest(fork));stages += [fork, final]
        for key, field in (('model_state', 'model'), ('adversary_state', 'adversaries'), ('mapper_optimizer_state', 'mapper_optimizer'), ('adversary_optimizer_state', 'adversary_optimizer')):
            digest = tree_digest if 'optimizer' in key else check.state_hash
            value = digest(fork[key]) if fork[key] is not None else None
            assert value == (digest(shared[key]) if shared[key] is not None else None) == record['fork_hashes'][field] == meta['shared_fork_hashes'][field]
        assert final['counters'] == {'mapper_optimizer_steps': 140 * batches, 'adversary_optimizer_steps': 0 if bank else 260 * batches, 'continuation_epoch': 80}
        adam_steps(final['mapper_optimizer_state'], 140 * batches)
        if bank: assert final['adversary_state'] == {} and final['adversary_optimizer_state'] is None
        else: adam_steps(final['adversary_optimizer_state'], 260 * batches)
        assert record['selected_epoch'] == 80 and record['continuation_preservation_coefficient'] == beta
        assert record['mapper_loss_uses_protection_gradient'] == (arm == 'D') and record['schedule_hash'] == meta['schedules']['continuation']['sha256']
        assert record['mapper_row_exposures'] == n * 80 and record['adversary_row_exposures'] == (0 if bank else n * 240)
        for key, field in (('model_state', 'final_model_hash'), ('adversary_state', 'final_adversary_hash')): assert check.state_hash(final[key]) == record[field]
        for key, field in (('mapper_optimizer_state', 'final_mapper_optimizer_hash'), ('adversary_optimizer_state', 'final_adversary_optimizer_hash')):
            assert (tree_digest(final[key]) if final[key] is not None else None) == record[field]
        assert record['decoder_unchanged'] and record['decoder_gradients_absent'] and record['decoder_optimizer_state_entries'] == 0
        assert record['decoder_optimizer_parameter_indices'] == [10, 11, 12, 13]
        for curve in record['curve']:
            assert curve['mapper_optimizer_steps'] == (60 + curve['epoch']) * batches
            assert curve['adversary_optimizer_steps'] == (0 if bank else (20 + 3 * curve['epoch']) * batches)
            assert curve['preservation_coefficient'] == beta and curve['reconstruction_coefficient'] == 0.
            assert curve['protection_coefficient'] == (-.1 if arm == 'D' else 0.)
        points.extend([(fork, fork['adversary_state'], arm == 'D', record['fixed_batch_gradient_diagnostics_at_shared_fork']),
                       (final, final['adversary_state'], arm == 'D', record['fixed_batch_gradient_diagnostics_at_final'])])
    assert len(set(fork_hashes)) == 1
    for checkpoint in stages:
        assert checkpoint['mapper_optimizer_state']['param_groups'][0]['params'] == list(range(14))
        assert all(i not in checkpoint['mapper_optimizer_state']['state'] for i in (10, 11, 12, 13))
        assert all(torch.equal(value, checkpoint['model_state'][key]) for key, value in state.items() if key.startswith('decoder.'))
        if access != 'F':
            assert torch.count_nonzero(checkpoint['model_state']['mapper.0.weight'][:, 16:]) == 0
            if checkpoint['mapper_optimizer_state']['state']:
                for key in ('exp_avg', 'exp_avg_sq'): assert torch.count_nonzero(checkpoint['mapper_optimizer_state']['state'][0][key][:, 16:]) == 0
    maximum = 0.
    for checkpoint, observer_state, protected, recorded in points:
        actual = diagnostic(checkpoint['model_state'], observer_state, x[idx], teacher[idx],
            {name: labels['representation_fit', name][idx] for name in SOURCES},
            {} if bank else {name: labels['representation_fit', name][idx] for name in ATTRIBUTES}, meta['prior_entropies'], beta, protected)
        assert recorded['coefficients'] == {'source': 1., 'teacher': beta, 'protection': -.1 if protected else 0.}
        assert recorded['training_state_unchanged'] and recorded['torch_rng_unchanged'] and recorded['optimizer_steps'] == 0
        for key, value in actual.items():
            if value is None: assert recorded[key] is None
            else:
                maximum = max(maximum, abs(value - recorded[key]))
                assert np.isclose(value, recorded[key], rtol=2e-6, atol=1e-9), (unit, key, value, recorded[key])
    return {'unit': unit, 'passed': True, 'checkpoint_stages_checked': len(stages), 'gradient_points_replayed': len(points),
            'maximum_gradient_scalar_error': maximum, 'full_initial_tensors_matched_by_seed': True,
            'all_Adam_counters_forks_and_decoder_exclusions': True, 'restricted_raw_weights_and_moments_zero': access != 'F',
            'source_only_bank_exposure_and_observer_omission': bank, 'training_json_sha256': check.sha(training / 'training.json')}


def verify(out):
    started = time.perf_counter();cfg, frozen = (check.read(out / name) for name in ('config.json', 'protocol_freeze.json'))
    for group in ('sha256', 'reference_record_hashes', 'required_local_reference_hashes'):
        for path, digest in frozen[group].items(): assert check.sha(ROOT / path) == digest
    complete = check.read(out / 'progress.json')['completed_units'];assert complete
    parent = ROOT / cfg['parent_results'];rawpath = ROOT / check.read(parent / 'config.json')['raw_path']
    assert check.sha(rawpath) == check.read(parent / 'schema_support.json')['raw_sha256']
    raw = pd.read_csv(rawpath, usecols=['PINCP', 'ESR', 'PUBCOV', 'SEX', 'RAC1P'])
    initial, reports = {}, []
    for unit in complete: reports.append(verify_unit(out, cfg, unit, raw, frozen, initial))
    return {'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'Independent saved training/initialization/fork/Adam/boundary/gradient replay; no fitting; separate verifier handles release/audit/native/composition predictions',
            'script_sha256': check.sha(__file__), 'completed_units': len(complete), 'expected_units': 18,
            'complete_matrix': len(complete) == 18, 'units': reports,
            'gradient_points_replayed': sum(r['gradient_points_replayed'] for r in reports),
            'maximum_gradient_scalar_error': max(r['maximum_gradient_scalar_error'] for r in reports),
            'runtime_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/redesign_20260908_acs_restricted_inputs_v1')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists(): raise FileExistsError('Preserve existing evidence')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result = verify(args.out.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.report:
        with args.report.open('x') as handle: handle.write(rendered)
    else: print(rendered, end='')


if __name__ == '__main__': main()
