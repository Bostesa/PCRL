"""Independent source-guard proposal, geometry, release and auditor replay.

This read-only program never invokes the experimental guarded update or an
optimizer step.  Literal tensor algebra replays Adam from the saved current
state; a separate least-squares active-set calculation checks the projection.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import itertools
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_coalition_strength_training import literal_components
from experiments.acs_bottleneck_training import tree_digest
from scripts.verify_acs_coalition_strength_training import seed_context
from scripts.verify_acs_coalition_training import literal_parts

TASKS = ('income_binary', 'civilian_at_work', 'public_coverage')
SVD_RCOND = 1e-12
KKT_RTOL = 1e-12
CONDITIONS = ('F_T', 'P_T', 'F_G_J', 'F_G_L025', 'F_G_L20', 'P_G_J', 'P_G_L025', 'P_G_L20')
PARENT_COMMIT = '02a069213e095ed9e8a890c4cbd92e9ef84edb41'


def condition_identity(name):
    if name not in CONDITIONS:
        raise ValueError('Only the eight predeclared new source-guard conditions are accepted')
    interface, role = name.split('_', 1)
    family, beta = {'T': ('T', 0.), 'G_J': ('J', .1),
                    'G_L025': ('Iplus', .025), 'G_L20': ('Iplus', .2)}[role]
    return interface, family, beta


def literal_adam(parameters, optimizer_state, gradients):
    """Replay the installed CPU Adam algebra without constructing an optimizer.

    None skips both a parameter and its moment clock. A real zero gradient
    advances its clock and inherited momentum. Input objects remain untouched.
    """
    state = copy.deepcopy(optimizer_state)
    if len(state['param_groups']) != 1:
        raise ValueError('Only the fixed single Adam group is supported')
    group = state['param_groups'][0]
    assert group['lr'] == .001 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
    assert group['weight_decay'] == 0
    assert all(not group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay', 'fused'))
    assert group.get('foreach') in (None, False)
    assert len(parameters) == len(gradients) == len(group['params'])
    proposed = [p.detach().clone() for p in parameters]
    for parameter, gradient, key in zip(proposed, gradients, group['params']):
        assert parameter.dtype == torch.float32 and parameter.device.type == 'cpu'
        if gradient is None:
            continue
        assert gradient.shape == parameter.shape and gradient.dtype == parameter.dtype
        assert bool(torch.isfinite(gradient).all())
        if key not in state['state']:
            state['state'][key] = {'step': torch.tensor(0., dtype=torch.float32),
                                  'exp_avg': torch.zeros_like(parameter),
                                  'exp_avg_sq': torch.zeros_like(parameter)}
        item = state['state'][key]
        assert set(item) == {'step', 'exp_avg', 'exp_avg_sq'}
        item['step'].add_(1)
        beta1, beta2 = group['betas']
        item['exp_avg'].lerp_(gradient, 1-beta1)
        item['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
        step = item['step'].item()
        denominator = (item['exp_avg_sq'].sqrt() / ((1-beta2**step)**.5)).add_(group['eps'])
        parameter.addcdiv_(item['exp_avg'], denominator, value=-group['lr']/(1-beta1**step))
    return proposed, state


def independent_projection(raw, gradients):
    """Enumerate legal KKT solutions using a different rectangular LS solve."""
    raw = np.asarray(raw, dtype=np.float64)
    g = np.asarray(gradients, dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or g.shape != (3, len(raw)):
        raise ValueError('Expected three gradients on a nonempty parameter support')
    if not np.isfinite(raw).all() or not np.isfinite(g).all():
        raise FloatingPointError('Nonfinite projection input')
    subsets = sorted(a for n in range(4) for a in itertools.combinations(range(3), n))
    row_norm = np.linalg.norm(g, axis=1)
    candidates = []
    for active in subsets:
        multipliers = np.zeros(3, dtype=np.float64)
        if active:
            a = g[list(active)]
            lam, _, rank, singular = np.linalg.lstsq(a.T, raw, rcond=SVD_RCOND)
            multipliers[list(active)] = lam
            normal = a.T @ lam
            value = raw-normal
        else:
            rank, singular, value = 0, np.empty(0), raw.copy()
        normal = g.T @ multipliers
        dots = g @ value
        primal = KKT_RTOL*np.maximum(1., row_norm*np.linalg.norm(value))
        dual = KKT_RTOL*max(1., np.linalg.norm(multipliers))
        stationary = KKT_RTOL*max(1., np.linalg.norm(value), np.linalg.norm(raw), np.linalg.norm(normal))
        complementary = KKT_RTOL*np.maximum(1., np.abs(multipliers)*row_norm*np.linalg.norm(value))
        if (np.all(dots <= primal) and np.all(multipliers >= -dual)
                and np.linalg.norm(value-raw+normal) <= stationary
                and np.all(np.abs(multipliers*dots) <= complementary)):
            candidates.append({'active': list(active), 'value': value, 'multipliers': multipliers,
                               'rank': int(rank), 'singular_values': singular,
                               'distance_squared': float(np.dot(value-raw, value-raw))})
    if not candidates:
        raise FloatingPointError('No independently valid KKT solution')
    best = min(c['distance_squared'] for c in candidates)
    return next(c for c in candidates if c['distance_squared'] <= best+KKT_RTOL*max(1., best))


def flat(values, parameters=None):
    if parameters is not None:
        values = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, values)]
    return np.concatenate([value.detach().numpy().reshape(-1).astype(np.float64) for value in values])


def assert_gradient_lists(actual, recorded):
    assert len(actual) == len(recorded)
    for left, right in zip(actual, recorded):
        assert (left is None) == (right is None), 'None gradient was converted to an Adam zero step'
        if left is not None:
            torch.testing.assert_close(left, right, atol=0., rtol=0.)


def assert_dot_reduction(actual, recorded, matrix, vector):
    """Permit only the sum of two float64 dot-product roundoff bounds."""
    unit = np.finfo(np.float64).eps/2.
    gamma = len(vector)*unit/(1-len(vector)*unit)
    bound = 2*gamma*(np.abs(matrix)@np.abs(vector))+4*unit*np.abs(recorded)
    assert np.all(np.abs(np.asarray(actual)-recorded) <= bound)


def verify_guard_evidence(evidence, standardized, labels, priors, interface, family, beta):
    """Replay one actual saved diagnostic step, including stored float32 bounds."""
    names = evidence['parameter_names']
    pre = evidence['pre_model_state']
    observers = evidence['observers_after_scheduled_updates']
    parameters = [pre[name] for name in names]
    state, losses, tasks, _, _, _ = literal_components(
        pre, observers, standardized, labels, priors, interface)
    literal_parameters = [state[name] for name in names]
    source_grad = torch.autograd.grad(losses['source'], literal_parameters, retain_graph=True, allow_unused=True)
    task_grad = {t: torch.autograd.grad(tasks[t], literal_parameters, retain_graph=True, allow_unused=True) for t in TASKS}
    coefficients = {'individual': -.1, 'extra_local': -beta if family == 'Iplus' else 0.,
                    'coalition': -beta if family == 'J' else 0.}
    full = losses['source']
    for name, coefficient in coefficients.items():
        if coefficient:
            full = full + coefficient*losses[name]
    full_grad = torch.autograd.grad(full, literal_parameters, allow_unused=True)
    assert_gradient_lists(source_grad, evidence['source_gradients'])
    assert_gradient_lists(full_grad, evidence['full_gradients'])
    for task in TASKS:
        assert_gradient_lists(task_grad[task], evidence['task_gradients'][task])
    source, source_adam = literal_adam(parameters, evidence['pre_optimizer'], source_grad)
    complete, complete_adam = literal_adam(parameters, evidence['pre_optimizer'], full_grad)
    assert_gradient_lists(source, evidence['source_proposal'])
    assert_gradient_lists(complete, evidence['full_proposal'])
    assert tree_digest(source_adam) == tree_digest(evidence['source_proposal_optimizer'])
    assert tree_digest(complete_adam) == tree_digest(evidence['full_proposal_optimizer']) == tree_digest(evidence['post_optimizer'])
    mask = np.concatenate([np.full(p.numel(), interface == 'P' or '.mapper.' in name) for name, p in zip(names, parameters)])
    np.testing.assert_array_equal(mask, evidence['support_mask'].numpy())
    source_flat, full_flat = flat(source), flat(complete)
    np.testing.assert_array_equal(source_flat[~mask], full_flat[~mask])
    raw = (full_flat-source_flat)[mask]
    gradients = np.stack([flat(task_grad[t], parameters)[mask] for t in TASKS])
    np.testing.assert_array_equal(raw, evidence['raw_increment'].numpy())
    np.testing.assert_array_equal(gradients, evidence['task_gradient_matrix'].numpy())
    independent = independent_projection(raw, gradients)
    ideal = evidence['ideal_increment'].numpy()
    np.testing.assert_allclose(independent['value'], ideal, atol=1e-11, rtol=1e-8)
    scalar = evidence['scalar']; projection = scalar['projection']
    assert independent['active'] == projection['active_subset'], 'Fixed lexicographic active-set choice differs'
    assert independent['rank'] == projection['active_rank']
    assert abs(independent['distance_squared']-projection['distance_squared']) <= projection['distance_tie_tolerance']
    multipliers = np.asarray(projection['multipliers'])
    norm = np.linalg.norm(gradients, axis=1)
    dots = gradients @ ideal
    primal = KKT_RTOL*np.maximum(1., norm*np.linalg.norm(ideal))
    assert np.all(dots <= primal)
    assert np.all(multipliers >= -KKT_RTOL*max(1., np.linalg.norm(multipliers)))
    assert np.linalg.norm(ideal-raw+gradients.T@multipliers) <= projection['stationarity_tolerance']
    assert np.all(np.abs(multipliers*dots) <= np.asarray(projection['complementarity_tolerances']))
    proposed = source_flat.copy(); proposed[mask] += ideal
    stored = proposed.astype(np.float32).astype(np.float64)
    post = flat([evidence['post_model_state'][name] for name in names])
    np.testing.assert_array_equal(post, stored)
    np.testing.assert_array_equal(flat(evidence['accepted_parameters']), stored)
    np.testing.assert_array_equal(stored[~mask], source_flat[~mask])
    realized = (stored-source_flat)[mask]; error = realized-ideal
    np.testing.assert_array_equal(realized, evidence['realized_increment'].numpy())
    np.testing.assert_array_equal(error, evidence['cast_error'].numpy())
    actual_dot = gradients @ realized; cast_bound = np.abs(gradients) @ np.abs(error)
    rounding = np.asarray(scalar['float64_roundoff_bounds'])
    unit = np.finfo(np.float64).eps/2.
    gamma_n = len(ideal)*unit/(1-len(ideal)*unit)
    gamma_one = unit/(1-unit)
    absolute_realized = np.abs(gradients) @ np.abs(realized)
    absolute_ideal = np.abs(gradients) @ np.abs(ideal)
    independently_bounded = (gamma_n*(absolute_realized+absolute_ideal+cast_bound)
        + gamma_one*(absolute_realized+absolute_ideal+np.abs(actual_dot)+np.abs(dots)))
    # Fresh NumPy buffers can change the last reduction bit through BLAS
    # alignment. The bound itself is checked to eight float64 epsilons; this
    # does not alter either the production KKT or stored-step inequalities.
    np.testing.assert_allclose(rounding, independently_bounded, atol=0., rtol=8*np.finfo(np.float64).eps)
    np.testing.assert_array_equal(scalar['kkt_primal_bounds'], primal)
    assert np.all(actual_dot-dots <= cast_bound+rounding)
    assert np.all(actual_dot <= primal+cast_bound+rounding)
    assert_dot_reduction(actual_dot, scalar['actual_dots'], gradients, realized)
    assert_dot_reduction(cast_bound, scalar['cast_dot_bounds'], np.abs(gradients), np.abs(error))
    ds = (source_flat-flat(parameters))[mask]
    assert_dot_reduction(gradients@ds, scalar['source_displacement_dots'], gradients, ds)
    assert_dot_reduction(gradients@(ds+ideal), scalar['ideal_total_displacement_dots'], gradients, ds+ideal)
    assert_dot_reduction(gradients@(ds+realized), scalar['actual_total_displacement_dots'], gradients, ds+realized)
    whole_g = np.stack([flat(task_grad[t], parameters) for t in TASKS])
    theta = flat(parameters)
    for field, vector in [('full_source_displacement_dots', source_flat-theta),
                          ('full_ideal_total_displacement_dots', proposed-theta),
                          ('full_actual_total_displacement_dots', stored-theta)]:
        assert_dot_reduction(whole_g@vector, scalar[field], whole_g, vector)
    assert scalar['source_objective'] == float(losses['source'].detach())
    assert scalar['full_objective'] == float(full.detach())
    assert scalar['loss_components'] == {k: float(v.detach()) for k, v in losses.items()}
    assert not scalar['cast_correction_applied'] and not scalar['projection_changes_optimizer_moments']
    for name in pre:
        if name not in names:
            torch.testing.assert_close(pre[name], evidence['post_model_state'][name], atol=0., rtol=0.)
    native_replay = {}
    for stage, values in [('pre', parameters), ('source', source), ('full', complete),
                          ('accepted', [evidence['post_model_state'][name] for name in names])]:
        updated = dict(pre); updated.update(zip(names, values))
        _, values, native, _, _, _ = literal_components(updated, observers, standardized, labels, priors, interface)
        native_replay[stage] = {'source': float(values['source'].detach()),
                               'tasks': {t: float(native[t].detach()) for t in TASKS},
                               'support': {t: int(np.sum(labels[t] >= 0)) for t in TASKS}}
    check.compare(native_replay, evidence['native_losses'], 'guard_native_losses')
    return {'projection_max_abs_error': float(np.max(np.abs(independent['value']-ideal))),
            'independent_active_subset_exact': True,
            'ideal_max_task_dot': float(np.max(dots)), 'actual_max_task_dot': float(np.max(actual_dot)),
            'float32_actual_bounds_passed': True, 'literal_adam_exact': True,
            'literal_gradients_exact': True, 'full_moments_retained_once': True}


def verify_source_evidence(evidence, standardized, labels, priors, interface):
    pre = evidence['pre_model_state']
    names = [name for name in pre if name.startswith('branches.')]
    state, losses, tasks, _, _, _ = literal_components(
        pre, evidence['observers_after_scheduled_updates'], standardized, labels, priors, interface)
    parameters = [state[n] for n in names]
    task_gradients = [torch.autograd.grad(tasks[t], parameters, retain_graph=True, allow_unused=True) for t in TASKS]
    gradients = torch.autograd.grad(losses['source'], parameters, allow_unused=True)
    assert_gradient_lists(gradients, evidence['source_gradients'])
    values, optimizer = literal_adam([pre[n] for n in names], evidence['pre_optimizer'], gradients)
    assert_gradient_lists(values, [evidence['post_model_state'][n] for n in names])
    assert tree_digest(optimizer) == tree_digest(evidence['post_optimizer'])
    scalar = evidence['scalar']
    assert scalar['literal_source_only'] and not scalar['float64_parameter_round_trip']
    assert scalar['coefficients'] == {'source': 1., 'individual': 0., 'extra_local': 0., 'coalition': 0.}
    assert scalar['source_objective'] == float(losses['source'].detach())
    assert scalar['source_task_losses'] == {t: float(tasks[t].detach()) for t in TASKS}
    after = dict(pre); after.update(zip(names, values))
    _, new_losses, new_tasks, _, _, _ = literal_components(after, evidence['observers_after_scheduled_updates'], standardized, labels, priors, interface)
    before_native = {'source': float(losses['source'].detach()), 'tasks': {t: float(tasks[t].detach()) for t in TASKS}}
    after_native = {'source': float(new_losses['source'].detach()), 'tasks': {t: float(new_tasks[t].detach()) for t in TASKS}}
    displacement = flat(values)-flat([pre[n] for n in names])
    full_g = np.stack([flat(g, parameters) for g in task_gradients])
    return {'literal_gradients_exact': True, 'literal_source_Adam_exact': True,
            'all_protection_coefficients_zero': True, 'float64_parameter_round_trip': False,
            'source_displacement_l2': float(np.linalg.norm(displacement)), 'raw_increment_l2': 0., 'ideal_increment_l2': 0.,
            'actual_increment_l2': 0., 'projection_applied': False,
            'full_source_displacement_dots': (full_g@displacement).tolist(),
            'native_losses': {'pre': before_native, 'source': after_native, 'full': after_native, 'accepted': after_native},
            'native_finite_step_task_changes': {t: after_native['tasks'][t]-before_native['tasks'][t] for t in TASKS},
            'derivation': 'Read-only exact saved T pre/post state; source/full/accepted are aliases of the ordinary source step'}


def optimizer_clock(optimizer, expected):
    assert len(optimizer['param_groups']) == 1 and optimizer['state']
    group = optimizer['param_groups'][0]
    assert group['lr'] == .001 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
    assert group['weight_decay'] == 0
    assert set(optimizer['state']) == set(group['params'])
    for value in optimizer['state'].values():
        assert int(value['step']) == expected
        assert all(bool(torch.isfinite(value[k]).all()) for k in ('step', 'exp_avg', 'exp_avg_sq'))


def verify_step_stream(path, metadata, context):
    """Read every compact update record; independently check row schedule/bounds.

    Detailed vector/gradient reconstruction is limited to the fixed five saved
    steps. This stream check does not pretend to reconstruct unsaved vectors.
    """
    seed = metadata['seed']; n = len(context['pca']); batches = (n+255)//256
    rng = np.random.default_rng(1280000+100*seed+200)
    total = 0; projection_steps = 0; epoch_summaries = []
    with gzip.open(path, 'rt') as stream:
        for epoch in range(1, 81):
            order = rng.permutation(n); records = []
            for start in range(0, n, 256):
                raw = stream.readline(); assert raw, 'Missing compact guarded step'
                row = json.loads(raw); ix = order[start:start+256]; total += 1
                assert (row['epoch'], row['minibatch_start']) == (epoch, start)
                assert row['batch_indices_sha256'] == check.array_hash(ix)
                assert row['counts'] == {'mapper_optimizer_steps': 60*batches+total,
                                         'adversary_optimizer_steps': 20*batches+3*total}
                assert row['source_support'] == {t: int(np.sum(context['labels'][t][ix]>=0)) for t in TASKS}
                assert row['coefficients'] == {'source': 1., **metadata['coefficients']}
                if metadata['guarded']:
                    projection_steps += 1; p = row['projection']
                    assert row['support_dimension'] == (6304 if metadata['interface']=='F' else 6355)
                    assert row['outside_support_dimension'] == 6355-row['support_dimension']
                    assert np.all(np.asarray(p['ideal_dots']) <= np.asarray(p['primal_tolerances']))
                    assert np.all(np.asarray(p['multipliers']) >= -p['dual_tolerance'])
                    assert p['stationarity_l2'] <= p['stationarity_tolerance']
                    assert np.all(np.asarray(p['complementarity_abs']) <= np.asarray(p['complementarity_tolerances']))
                    assert np.all(np.asarray(row['actual_minus_ideal_dots']) <= np.asarray(row['cast_plus_roundoff_bounds']))
                    assert np.all(np.asarray(row['actual_dots']) <= np.asarray(row['actual_dot_excess_bounds']))
                    assert row['zero_constraint_tasks'] == [t for t in TASKS if not row['source_support'][t]]
                else:
                    assert row['literal_source_only'] and row['projection_steps'] == 0
                records.append(row)
            summary = metadata['epoch_update_summaries'][epoch-1]
            assert summary['epoch'] == epoch and summary['mapper_steps'] == batches
            if not metadata['guarded']:
                assert summary['literal_source_only'] and summary['projection_steps'] == 0
                assert summary['mean_source_objective'] == float(np.mean([r['source_objective'] for r in records]))
            else:
                assert summary['projection_steps'] == batches
                active, ranks = {}, {}
                for r in records:
                    key = ','.join(map(str, r['projection']['active_subset'])) or 'empty'
                    active[key] = active.get(key, 0)+1
                    rank = str(r['projection']['active_rank']); ranks[rank] = ranks.get(rank, 0)+1
                assert summary['active_subset_counts'] == active and summary['active_rank_counts'] == ranks
                for field, container in [('raw_dots', 'projection'), ('ideal_dots', 'projection'), ('actual_dots', None),
                    ('cast_dot_bounds', None), ('actual_dot_minus_bound', None), ('source_displacement_dots', None),
                    ('ideal_total_displacement_dots', None), ('actual_total_displacement_dots', None)]:
                    v = np.array([(r[container] if container else r)[field] for r in records])
                    assert summary[field] == {'min': v.min(0).tolist(), 'max': v.max(0).tolist(), 'mean': v.mean(0).tolist()}
                for field, container in [(f, None) for f in ('source_displacement_l2','full_displacement_l2','realized_increment_l2','cast_error_l2','cast_error_max_abs')]+[(f,'projection') for f in ('raw_l2','ideal_l2','projection_distance','stationarity_l2','distance_squared')]:
                    v = [(r[container] if container else r)[field] for r in records]
                    assert summary[field] == {'min': min(v), 'max': max(v), 'mean': float(np.mean(v))}
            epoch_summaries.append(summary['epoch'])
        assert not stream.readline(), 'Extra undeclared update after fixed epoch80'
    return {'records': total, 'projection_steps': projection_steps, 'epoch_summaries': len(epoch_summaries),
            'all_schedules_and_saved_scalar_bounds_passed': True}


def verify_training_unit(out, entry, context, report, shared_seen):
    seed, condition = entry['seed'], entry['condition']
    interface, family, beta = condition_identity(condition)
    directory = out/f'seed_{seed}'/'training'/condition
    metadata = check.read(directory/'training.json')
    assert metadata['study'] == 'acs_source_guard_v1' and not metadata['miniature']
    assert (metadata['interface'], metadata['regime'], metadata['beta']) == (interface, family, beta)
    assert metadata['guarded'] == (family != 'T')
    assert not any(metadata[k] for k in ('reserved_labels_received', 'final_evaluation_received', 'warmup_refitted'))
    assert metadata['historical_fork_unchanged'] and metadata['caller_rng_unchanged']
    assert metadata['module_source_sha256'] == check.sha(ROOT/'experiments/acs_source_guard_training.py')
    previous = context['meta']
    for key in ('source_label_hashes', 'attribute_label_hashes', 'source_validation_label_hashes',
                'fit_input_sha256', 'standardized_fit_sha256', 'source_validation_input_sha256',
                'preprocessing', 'prior_entropies', 'schedules', 'observer_optimization',
                'source_fit_coverage', 'attribute_fit_coverage'):
        assert metadata[key] == previous[key], key
    assert metadata['historical_training_module_sha256'] == previous['module_source_sha256']
    assert metadata['observer_roles'] == previous['observer_roles'][interface]
    original = context['forks'][interface]
    fork = torch.load(directory/'fork.pt', map_location='cpu', weights_only=True)
    final = torch.load(directory/'final.pt', map_location='cpu', weights_only=True)
    assert tree_digest(fork) == tree_digest(original) == metadata['historical_fork_tree_sha256']
    assert metadata['historical_fork_sha256'] == check.sha(ROOT/entry['historical_fork'])
    assert metadata['fork_hashes'] == previous['arms'][interface+'_I']['fork_hashes']
    assert metadata['final_hashes'] == {'model': check.state_hash(final['model_state']),
        'adversaries': check.state_hash(final['adversary_state']), 'mapper_optimizer': tree_digest(final['mapper_optimizer']),
        'adversary_optimizer': tree_digest(final['adversary_optimizer'])}
    n = len(context['pca']); batches = (n+255)//256
    assert metadata['counts'] == final['counts'] == {'mapper_optimizer_steps': 140*batches, 'adversary_optimizer_steps': 260*batches}
    optimizer_clock(final['mapper_optimizer'], 140*batches)
    optimizer_clock(final['adversary_optimizer'], 260*batches)
    schedule = copy.deepcopy(fork['schedule_state']); schedule['completed_epochs'] = 80
    assert final['schedule_state'] == schedule and final['stage'] == 'fixed final iterate'
    assert torch.equal(final['torch_rng_state'], fork['torch_rng_state'])
    assert metadata['selected_epoch'] == 80 and metadata['mapper_exposure_per_row'] == 140 and metadata['observer_exposure_per_row'] == 260
    assert metadata['diagnostic_epochs'] == [1,20,40,60,80]
    for name in ('input_mean', 'input_scale'):
        torch.testing.assert_close(final['model_state'][name], fork['model_state'][name], atol=0., rtol=0.)
    stream_meta = metadata['all_step_diagnostics'] if family!='T' else metadata['shared_source_only']['all_step_diagnostics']
    step_path = directory/'step_diagnostics.jsonl.gz' if family!='T' else directory.parent/'T_shared_step_diagnostics.jsonl.gz'
    assert Path(stream_meta['path']).resolve() == step_path.resolve() and check.sha(step_path) == stream_meta['sha256']
    if family=='T' and (seed, 'stream') in shared_seen:
        stream = shared_seen[seed, 'stream']
    else:
        stream = verify_step_stream(step_path, metadata, context)
        report['compact_step_records'] += stream['records']; report['compact_guard_projection_records'] += stream['projection_steps']
        report['epoch_summaries_replayed'] += stream['epoch_summaries']
        if family=='T': shared_seen[seed, 'stream'] = stream
    assert stream_meta['rows'] == stream['records'] == 80*batches
    rng = np.random.default_rng(1280000+100*seed+200)
    orders = [rng.permutation(n) for _ in range(80)]
    detailed = []
    for item in metadata['detailed_steps']:
        epoch = item['epoch']; path = directory/f'diagnostic_epoch_{epoch:03d}.pt'
        assert Path(item['path']).resolve() == path.resolve() and check.sha(path) == item['sha256']
        evidence = torch.load(path, map_location='cpu', weights_only=True)
        ix = orders[epoch-1][:256]
        np.testing.assert_array_equal(evidence['batch_indices'].numpy(), ix)
        assert evidence['batch_indices_sha256'] == item['batch_indices_sha256'] == check.array_hash(ix)
        assert evidence['interface'] == interface and evidence['epoch'] == epoch
        assert evidence['scalar'] == item['scalar']
        steps = (epoch-1)*batches+1
        assert evidence['counts_after_update'] == {'mapper_optimizer_steps': 60*batches+steps,
                                                  'adversary_optimizer_steps': 20*batches+3*steps}
        optimizer_clock(evidence['pre_optimizer'], 60*batches+steps-1)
        optimizer_clock(evidence['post_optimizer'], 60*batches+steps)
        optimizer_clock(evidence['observer_optimizer_after_scheduled_updates'], 20*batches+3*steps)
        assert torch.equal(evidence['torch_rng_state'], fork['torch_rng_state'])
        if epoch == 1:
            assert tree_digest(evidence['pre_model_state']) == tree_digest(fork['model_state'])
            assert tree_digest(evidence['pre_optimizer']) == tree_digest(fork['mapper_optimizer'])
        x = context['standardized'][ix]; labels = {k: y[ix] for k, y in context['labels'].items()}
        if family == 'T':
            key = seed, epoch
            if key not in shared_seen:
                value = verify_source_evidence(evidence, x, labels, context['priors'], interface)
                report['unique_source_only_steps_replayed'] += 1
                shared_seen[key] = {k: tree_digest(evidence[k]) for k in ('pre_model_state','pre_optimizer','source_gradients','post_model_state','post_optimizer')}
                shared_seen[seed, epoch, 'diagnostic'] = value
            else:
                assert shared_seen[key] == {k: tree_digest(evidence[k]) for k in shared_seen[key]}
                value = {**shared_seen[seed, epoch, 'diagnostic'], 'shared_forward_and_Adam_alias_exact': True}
                report['source_only_alias_steps_checked'] += 1
        else:
            value = verify_guard_evidence(evidence, x, labels, context['priors'], interface, family, beta)
            report['guard_steps_replayed'] += 1
        detailed.append({'epoch': epoch, 'sha256': item['sha256'], **value})
    assert len(detailed) == 5
    if family == 'T':
        alias = metadata['shared_source_only']
        assert alias['forward_alias'] == f'seed_{seed}/T_shared'
        assert alias['unique_new_forward_continuations'] == 1 and alias['interface_systems'] == 2
        assert alias['new_forward_optimizer_steps'] == 80*batches
        assert alias['initial_forward_state_sha256'] == tree_digest((fork['model_state'],fork['mapper_optimizer']))
        assert alias['final_forward_state_sha256'] == tree_digest((final['model_state'],final['mapper_optimizer']))
        other = directory.parent/('P_T' if interface=='F' else 'F_T')
        other_final = torch.load(other/'final.pt', map_location='cpu', weights_only=True)
        for key in ('model_state', 'mapper_optimizer', 'counts', 'schedule_state', 'torch_rng_state'):
            assert tree_digest(final[key]) == tree_digest(other_final[key])
        assert final['adversary_state']['A__SEX.0.weight'].shape != other_final['adversary_state']['A__SEX.0.weight'].shape
    report['training_units_verified'] += 1
    return {'final_checkpoint_sha256': check.sha(directory/'final.pt'), 'fork_full_state_exact': True,
            'final_optimizer_clocks_and_schedule_exact': True, 'step_stream': stream, 'diagnostic_steps': detailed}


# The immutable strength replay body is retained here because its public entry
# point rejects this separately named matrix. No historical globals are patched.
from scripts.verify_acs_coalition import (
    TARGETS, TASKS as UTILITY_TASKS, AUTHORIZED, ROLES, FRESH, SCOPES, literal_wires,
    public_probabilities, candidate_input, expected_pool, check_curves, verify_native,
)
from scripts.verify_acs_coalition_strength import candidate_belongs_to_condition, check_completion, published_hash

def verify_release_arrays(releases,checkpoint,pca,condition,outputs,report):
    interface,_,_=condition_identity(condition)
    for pool,x in pca.items():
        wires,derived,_=literal_wires(checkpoint['model_state'],x,interface)
        for space,values in [('wire',wires),('derived',derived)]:
            for view,value in values.items():
                np.testing.assert_array_equal(value,releases[space+'/'+view+'/'+pool])
                report['literal_release_arrays']+=1
                if pool!='test': assert check.array_hash(value)==outputs[space][view][pool]
        if interface=='F':
            composed,_=public_probabilities(checkpoint['model_state'],{v:releases['wire/'+v+'/'+pool] for v in ('A','B')})
            for view in ('A','B','AB'):
                np.testing.assert_array_equal(composed[view],releases['derived/'+view+'/'+pool])
                report['public_head_composition_arrays']+=1
        else:
            for view in ('A','B','AB'):
                np.testing.assert_array_equal(releases['wire/'+view+'/'+pool],releases['derived/'+view+'/'+pool])

def verify_condition(dest,seed,condition,checkpoint,releases,release_freeze,release_freeze_hash,
                     completion,fit_indices,frames,label,rows,report,cache,score_cache,seen,terminals):
    condition_identity(condition)
    selected = check.read(dest/'selection_before_test.json'); audits = check.read(dest/'audits/audit_selection.json')
    interface,family,beta=condition_identity(condition)
    assert selected['coefficient_identity']=={'interface':interface,'family':family,'beta':beta}
    assert selected['release_freeze_sha256'] == release_freeze_hash
    assert release_freeze['created_utc'] <= selected['created_utc'] <= completion['completed_utc']
    assert not audits['development_received'] and audits['interface'] == condition[0]
    report['new_trajectory_counts'][str(seed)+'/'+condition] = audits['counts']
    expected_roles_count = 22 if condition.startswith('F') else 11
    assert audits['counts']['new_five_candidate_roles'] == expected_roles_count
    assert audits['counts']['new_fresh_mlp_trajectories'] == expected_roles_count*2
    assert audits['counts']['own_catchup_trajectories'] == 9
    assert audits['counts']['reused_five_candidate_roles'] == 0
    assert selected['audits'] == audits['selections']
    for role, record in selected['utility_metadata'].items():
        choices = record['candidates']
        assert selected['utility'][role] == min(choices,key=lambda c:(choices[c]['validation_scores']['log_loss'],c))
    expected_roles = {v+'/'+t for v,ts in ROLES.items() for t in ts}
    for budget in ('120','360'):
        assert set(audits['candidates'][budget]) == expected_roles
        for role, candidates in audits['candidates'][budget].items():
            view,target = role.split('/'); own = {'wire__'+c for c in FRESH}
            if condition.startswith('F'): own |= {'derived__'+c for c in FRESH}
            if target not in ('same_residence','commute_over20'): own |= {'saved_adversary','catchup'}
            inherited = {f'inherited_{v}__{cid}' for v in ('A','B') for cid in audits['candidates'][budget][v+'/'+target]} if view == 'AB' else set()
            assert set(candidates) == own|inherited
            for scope in SCOPES:
                eligible = expected_pool(candidates,scope)
                assert eligible == audits['selection_pools'][budget][role][scope]
                winner = min(eligible,key=lambda c:(candidates[c]['validation_scores']['log_loss'],c))
                assert winner == audits['selections'][budget][role][scope]; report['selection_pools'] += 1
                if view == 'AB': assert all(candidates[winner]['validation_scores']['log_loss'] <= candidates[c]['validation_scores']['log_loss'] for c in eligible)
    measurements = check.read(dest/'metrics.json')['raw_metrics']
    expected_rows = {('audit',*role.split('/'),int(budget),cid) for budget,roles in audits['candidates'].items() for role,candidates in roles.items() for cid in candidates}
    expected_rows |= {('utility',*role.split('/'),None,cid) for role,record in selected['utility_metadata'].items() for cid in record['candidates']}
    assert len(measurements) == len(expected_rows)
    assert {(r['role'],r['view'],r['target'],r['audit_budget'],r['candidate_id']) for r in measurements} == expected_rows
    with np.load(dest/'predictions.npz') as predictions:
        assert len(predictions.files) == len(measurements)*2
        for row in measurements:
            role,view,target,cid,budget = (row[k] for k in ('role','view','target','candidate_id','audit_budget'))
            assert row['seed'] == seed and row['condition'] == condition
            if role == 'audit':
                meta = audits['candidates'][str(budget)][view+'/'+target][cid]
                path = candidate_belongs_to_condition(meta['base_candidate_directory'],dest); base_meta = check.read(path/'metadata.json')
                assert check.sha(path/'metadata.json') == meta['base_metadata_sha256']
                assert all(meta[k] == v for k,v in base_meta.items() if k not in ('candidate_id',))
                scopes = [s for s,c in audits['selections'][str(budget)][view+'/'+target].items() if c == cid]
                if meta['inherited_singleton']:
                    origin = meta['source_view']; original = audits['candidates'][str(budget)][origin+'/'+target][meta['source_candidate_id']]
                    assert original['base_candidate_directory'] == str(path) and meta['space'] == original['space']
                    widths = (2,1) if meta['space'] == 'derived' or condition.startswith('P') else (16,16)
                    expected_columns = list(range(widths[0])) if origin == 'A' else list(range(widths[0],sum(widths)))
                    assert meta['projection_columns'] == expected_columns
                assert meta['actual_auditor_input_dimension'] == base_meta['input_dim']
            else:
                assert target in AUTHORIZED[view] and budget is None
                meta = selected['utility_metadata'][view+'/'+target]['candidates'][cid]; base_meta = meta
                path = dest/'fitted/utility'/view/target/cid
                assert check.read(path/'metadata.json') == meta
                scopes = ['utility'] if selected['utility'][view+'/'+target] == cid else []
            assert row['selected_scopes'] == scopes
            classes = 9 if target == 'RAC1P' else 2
            fitpool,valpool = ('downstream_fit','downstream_validation') if role == 'utility' else ('attacker_fit','attacker_validation')
            ix = fit_indices[role,target]; yf,mf = label[fitpool,target]; yv,mv = label[valpool,target]
            inputs = (lambda p,m: candidate_input(releases,meta,p,m)) if role == 'audit' else (lambda p,m: np.ascontiguousarray(releases['wire/'+view+'/'+p][m]))
            xf,xv = inputs(fitpool,ix).astype(np.float64),inputs(valpool,mv).astype(np.float64)
            unique = str(path)
            if unique not in cache: cache[unique] = check.load_inference(path,base_meta)
            predict,mean,scale,modelstate = cache[unique]
            if unique not in seen:
                seen.add(unique); report['unique_saved_candidate_paths'] += 1
                assert base_meta['n_classes'] == classes
                assert base_meta['validation_hashes'] == {'x':check.array_hash(xv),'y':check.array_hash(yv[mv].astype(np.int64))}
                saved = base_meta.get('audit_kind') == 'saved'; caught = base_meta.get('audit_kind') == 'catchup'
                if not saved:
                    assert base_meta['fit_hashes'] == {'x':check.array_hash(xf),'y':check.array_hash(yf[ix].astype(np.int64))}
                    assert base_meta['fit_support'] == np.bincount(yf[ix],minlength=classes).tolist()
                if saved or caught:
                    np.testing.assert_array_equal(mean,np.zeros(xf.shape[1])); np.testing.assert_array_equal(scale,np.ones(xf.shape[1]))
                    source_view = meta['source_view']; prefix = source_view+'__'+target+'.'
                    original = {k.removeprefix(prefix):v for k,v in checkpoint['adversary_state'].items() if k.startswith(prefix)}
                    assert check.state_hash(original) == base_meta['source_state_hash']
                    exposure = base_meta['inherited_exposure']; ty,tm = label['representation_fit',target]; ty = np.where(tm,ty,-1).astype(np.int64)
                    assert exposure['source_fit_raw_row_hash'] == check.array_hash(rows['representation_fit'])
                    assert exposure['fit_label_hash'] == check.array_hash(ty)
                    assert exposure['representation_fitting_passes'] == 260 and exposure['known_fitting_labels'] == int(tm.sum())
                    assert exposure['label_presentations'] == int(tm.sum())*260; report['inherited_exposure_checks'] += 1
                    if saved:
                        assert check.state_hash(modelstate) == check.state_hash(original)
                        np.testing.assert_array_equal(predict(xv),check.network_probability(original,xv)); report['saved_observer_fidelity'] += 1
                    else:
                        assert base_meta['initial_state_hash'] == check.state_hash(original)
                        assert base_meta['optimizer']['initial_state_entries'] == 0 and not base_meta['optimizer']['restored_state']
                else:
                    np.testing.assert_array_equal(mean,xf.mean(0)); std = xf.std(0)
                    np.testing.assert_array_equal(scale,np.where(std>1e-12,std,1.))
                original_cid = path.name
                expected_seed = ((1250000+100*seed+UTILITY_TASKS.index(target)) if role == 'utility' else (1300000 if caught or saved else 1260000)+100*seed+10*('A','B','AB').index(meta['source_view'])+TARGETS.index(target))
                if original_cid == 'mlp_1': expected_seed += 10000
                assert base_meta['seed'] == expected_seed
                check_curves(path,base_meta,len(ix),expected_seed,budget,report,terminals)
            for split,pool in [('validation',valpool),('test','test')]:
                y,mask = label[pool,target]; p = predictions[f'{role}/{view}/{target}/{budget}/{cid}/{split}']
                np.testing.assert_array_equal(predict(inputs(pool,mask)),p); report['prediction_sets'] += 1
                for suffix,weighted in [('',False),('_person_weighted',True)]:
                    skey = (seed,pool,target,check.array_hash(p),weighted)
                    if skey not in score_cache: score_cache[skey] = check.independent_scores(y[mask],p,classes,frames[pool].PWGTP.to_numpy(float)[mask] if weighted else None)
                    check.compare(score_cache[skey],row['scores'][split+suffix],f'{seed}/{condition}/{role}/{view}/{target}/{budget}/{cid}/{split}{suffix}'); report['score_dictionaries'] += 1
                if role == 'audit' and meta['inherited_singleton']:
                    direct = predictions[f'audit/{meta["source_view"]}/{target}/{budget}/{meta["source_candidate_id"]}/{split}']
                    np.testing.assert_array_equal(p,direct); report['inherited_prediction_sets'] += 1
            check.compare(base_meta['validation_scores'],row['scores']['validation'],f'{seed}/{condition}/{role}/{view}/{target}/{budget}/{cid}/validation_metadata')
            report['candidate_records'] += 1
        history = check.read(dest/'history_attack_replay.json')
        expected_history = {(r['view']+'/'+r['target'],r['candidate_id'],k) for r in measurements if r['role']=='audit' and r['audit_budget']==360 and r['selected_scopes'] for k in (1,2,4)}
        assert {(r['role'],r['candidate_id'],r['calls']) for r in history['records']} == expected_history
        for r in history['records']:
            view,target = r['role'].split('/'); meta = audits['candidates']['360'][r['role']][r['candidate_id']]
            _,mask = label['test',target]; x = candidate_input(releases,meta,'test',mask)
            repeated = np.stack([x]*r['calls'],1)
            assert all(np.array_equal(repeated[:,i],x) for i in range(r['calls']))
            predict = cache[meta['base_candidate_directory']][0]
            p = predict(np.ascontiguousarray(repeated[:,0]))
            assert check.array_hash(p) == r['prediction_sha256']; report['history_prediction_sets'] += 1


def expected_system_identities():
    result = {}
    for seed in (0,1,2):
        for condition in CONDITIONS:
            result[seed,condition] = (*condition_identity(condition),False,condition)
        for interface in ('F','P'):
            for family,beta,encoded in [('J',.1,'0p1'),('Iplus',.025,'0p025'),('Iplus',.2,'0p2'),('I',0.,None)]:
                condition = interface+'_I' if family=='I' else f'{interface}_{family}_b{encoded}'
                original = interface+'_'+family if beta==.1 else condition
                result[seed,condition] = (interface,family,beta,True,original)
    return result


def verify_global_gate(record, completions, fit_starts):
    assert record['training_permanently_closed'] and not record['new_reserved_fitting_started']
    assert record['historical_reserved_outcomes_previously_known']
    expected = expected_system_identities()
    entries = record['systems']; missing = record['missing']
    assert len(entries)+len(missing)==48
    assert len({(e['seed'],e['condition']) for e in entries+missing})==48
    for e in entries+missing:
        assert (e['interface'],e['family'],e['beta'],e['reused'],e['original_condition']) == expected[e['seed'],e['condition']]
    assert all(not e['reused'] for e in missing)
    assert record['all_intended_releases_frozen'] == (not missing)
    assert record['explicit_partial_prefix'] == bool(missing)
    ordered = [(s,c) for s in (0,1,2) for c in CONDITIONS]
    included = {(e['seed'],e['condition']) for e in entries if not e['reused']}
    assert included == set(ordered[:len(included)]), 'Partial training must be the predeclared completed prefix'
    assert len(completions)==len(included)
    for complete in completions:
        assert complete['training_complete'] and not complete['reserved_labels_accessed']
        assert complete['completed_utc'] <= record['created_utc']
    for start in fit_starts:
        assert start['all_intended_frozen'] == record['all_intended_releases_frozen']
        assert start['first_reserved_labels_after_global_freeze'] and record['created_utc'] <= start['started_utc']
    return {'frozen_systems':len(entries),'new_training_completions':len(included),
            'observed_fit_starts':len(fit_starts),'explicit_partial_prefix':bool(missing),'passed':True}


def verify_reuse(out, cfg, report):
    reuse = check.read(out/'REUSE_MANIFEST.json')
    assert reuse['original_commit']==PARENT_COMMIT
    assert reuse['historical_training_or_audit_refits']==0 and reuse['regeneration']==[]
    assert reuse['original_scientific_hashes_preserved']
    assert len(reuse['systems'])==48 and sum(e['reused'] for e in reuse['systems'])==24
    expected=expected_system_identities()
    for e in reuse['systems']:
        assert (e['interface'],e['family'],e['beta'],e['reused'],e['original_condition'])==expected[e['seed'],e['condition']]
        if e['reused']:
            assert e['original_fitting_exposure']=={'source_epochs':140,'observer_passes':260,
                'fresh_attacker_epochs':[120,360],'catchup_epochs':[120,360],'saved_start_epoch0_eligible':True,
                'utility_labels':2048,'utility_mlp_epochs':40}
            for name,h in e['original_files_sha256'].items():
                assert reuse['historical_files_sha256'][name]==h
    certificates={}
    for directory,names in [(ROOT/cfg['strength_reference_results'],('SCORE_REPLAY.json','TRAINING_REPLAY.json',
        'FINAL_COMPARISON_REPLAY_PUBLICATION.json','protocol_freeze.json','COMPACT_UNIT_EVIDENCE.json')),
        (ROOT/cfg['coalition_reference_results'],('SCORE_REPLAY.json','TRAINING_REPLAY.json','protocol_freeze.json','EXECUTION_AMENDMENTS.json'))]:
        for name in names:
            path=directory/name;certificates[str(path.relative_to(ROOT))]=published_hash(path,PARENT_COMMIT)
        score=check.read(directory/'SCORE_REPLAY.json')
        assert score['passed'] and score['complete_matrix']
        prior_verifier='verify_acs_coalition_strength.py' if directory==ROOT/cfg['strength_reference_results'] else 'verify_acs_coalition.py'
        assert score['script_sha256']==check.sha(ROOT/'scripts'/prior_verifier)
        assert score['metric_helper_sha256']==check.sha(check.__file__)
        assert score['checkpoint_helper_sha256']==check.sha(ROOT/'scripts/verify_acs_preservation_extended.py')
        assert check.read(directory/'TRAINING_REPLAY.json')['passed']
    for name,h in reuse['historical_files_sha256'].items():
        assert check.sha(ROOT/name)==h,('historical evidence changed',name)
    report['historical_reuse']={'paired_systems':24,'historical_files_hashed':len(reuse['historical_files_sha256']),
        'new_historical_control_fits':0,'mature_published_certificates':certificates,
        'historical_fitting_or_model_score_inference_repeated':False}
    return reuse


def verify(out, seeds=None, max_units=None, mode='all'):
    tick=time.perf_counter();check.errors.clear()
    check.max_error,check.max_error_path,check.numeric_comparisons=0.,None,0
    cfg=check.read(out/'config.json');freeze=check.read(out/'protocol_freeze.json')
    for name,h in freeze['scientific_and_protocol_sha256'].items():
        assert check.sha(ROOT/name)==h,('frozen scientific source',name)
    assert freeze['reuse_manifest_sha256']==check.sha(out/'REUSE_MANIFEST.json')
    assert freeze['prefit_identity_sha256']==check.sha(out/'PREFIT_IDENTITY.json')
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':mode,'script_sha256':check.sha(__file__),'metric_helper_sha256':check.sha(check.__file__),
        'literal_replay_helper_sha256':check.sha(ROOT/'scripts/verify_acs_coalition.py'),
        'literal_gradient_helper_sha256':check.sha(ROOT/'scripts/verify_acs_coalition_strength_training.py'),
        'checkpoint_helper_sha256':check.sha(ROOT/'scripts/verify_acs_preservation_extended.py'),
        'protocol_freeze_sha256':check.sha(out/'protocol_freeze.json'),'evaluation_status':cfg['evaluation_status'],
        'absolute_metric_tolerance':check.TOL,'seed_evidence':{},'training_evidence':{},'new_trajectory_counts':{},
        'guard_replay_policy':{'gradient_parameters_Adam_state':'exact equality including None versus zero',
            'independent_projection_comparison':{'atol':1e-11,'rtol':1e-8},
            'stored_projection_KKT':'original frozen tolerances, checked independently',
            'scalar_dot_reduction_equality':'sum of two standard float64 dot summation bounds',
            'roundoff_bound_relative_replay_tolerance':8*np.finfo(np.float64).eps,
            'actual_float32_inequalities':'original cast/roundoff bounds; no correction or relaxed feasibility'},
        'limitations':['Full saved gradient/proposal reconstruction is at the five fixed diagnostic steps per guarded trajectory; other steps have schedule and saved-scalar checks.',
            'Source-only current proposals inherit the actual full-objective Adam history.',
            'Independent attacker, native-head composition and saved-observer exposure remain separate.',
            'No representation retraining, historical model replay or optimization step is performed.',
            'Absent independent race-category support remains unassessable.']}
    for key in ('candidate_records','score_dictionaries','prediction_sets','unique_saved_candidate_paths','selection_pools',
        'inherited_prediction_sets','literal_release_arrays','public_head_composition_arrays','native_score_dictionaries',
        'native_predictions','mlp_schedules','actual_terminal_checkpoints','nested_trajectory_prefixes','saved_observer_fidelity',
        'inherited_exposure_checks','history_prediction_sets','local_artifact_hashes','training_units_verified',
        'guard_steps_replayed','unique_source_only_steps_replayed','source_only_alias_steps_checked',
        'compact_step_records','compact_guard_projection_records','epoch_summaries_replayed'):
        report[key]=0
    reuse=verify_reuse(out,cfg,report);hashed={}
    global_freeze=check.read(out/'RELEASE_MANIFEST.json') if (out/'RELEASE_MANIFEST.json').exists() else None
    if mode!='training': assert global_freeze is not None,'Globally frozen releases required before score replay'
    if global_freeze:
        global_hash=check.sha(out/'RELEASE_MANIFEST.json');report['global_release_manifest_sha256']=global_hash
        assert global_freeze['protocol_sha256']==check.sha(out/'protocol_freeze.json')
        completions,fit_starts=[],[]
        for e in global_freeze['systems']:
            assert check.sha(ROOT/e['checkpoint_path'])==e['checkpoint_sha256']
            assert check.sha(ROOT/e['release_record_path'])==e['release_record_sha256']
            source=check.read(ROOT/e['release_record_path'])
            outputs=source['outputs'] if 'wire' in source['outputs'] else source['outputs'][e['original_condition']]
            assert outputs==e['outputs']
            if not e['reused']:
                directory=out/f"seed_{e['seed']}"
                completions.append(check_completion(directory/'training'/e['condition']/'complete.json',report,hashed))
                assert source['checkpoint_sha256']==e['checkpoint_sha256']
                assert source['protocol_sha256']==check.sha(out/'protocol_freeze.json')
                if (directory/e['condition']/'fit_start.json').exists():
                    start=check.read(directory/e['condition']/'fit_start.json')
                    assert start['global_release_manifest_sha256']==global_hash;fit_starts.append(start)
        report['global_release_gate']=verify_global_gate(global_freeze,completions,fit_starts)
    selected_seeds=cfg['seeds'] if seeds is None else seeds
    assert selected_seeds and len(set(selected_seeds))==len(selected_seeds) and set(selected_seeds)<=set(cfg['seeds'])
    source_entries=global_freeze['systems'] if global_freeze else reuse['systems']
    units=[e for seed in selected_seeds for name in CONDITIONS for e in source_entries
        if e['seed']==seed and e['condition']==name and not e['reused']
        and (out/f'seed_{seed}'/('training' if mode=='training' else '')/name/'complete.json').exists()]
    if max_units is not None:
        assert max_units>0;units=units[:max_units]
    assert units,'No immutable completed new unit to replay'
    parent=ROOT/cfg['parent_results'];raw_path=ROOT/check.read(parent/'config.json')['raw_path']
    raw_hash=check.sha(raw_path)
    assert raw_hash==check.read(parent/'schema_support.json')['raw_sha256'];report['raw_sha256']=raw_hash
    columns=['PINCP','ESR','PUBCOV','SEX','RAC1P'] if mode=='training' else ['MIG','JWMNP','PINCP','ESR','PUBCOV','SEX','RAC1P','PWGTP']
    raw=pd.read_csv(raw_path,usecols=columns,low_memory=False)
    cache,score_cache,seen,shared_seen={},{},set(),{}
    for seed in selected_seeds:
        local=[e for e in units if e['seed']==seed]
        if not local:continue
        directory=out/f'seed_{seed}'
        with np.load(directory/'split_rows.npz') as z: rows={p:z[p].copy() for p in z.files}
        with np.load(parent/f'seed_{seed}/split_rows.npz') as z:
            assert set(rows)==set(z.files) and all(np.array_equal(v,z[p]) for p,v in rows.items())
        frames={p:raw.iloc[ix] for p,ix in rows.items()}
        with np.load(ROOT/cfg['reference_results']/f'seed_{seed}/release_E_pca.npz') as z:pca={p:z[p].copy() for p in z.files}
        if mode!='scores':
            context=seed_context(out,cfg,seed,raw,reuse)
            for e in local:
                check_completion(directory/'training'/e['condition']/'complete.json',report,hashed)
                report['training_evidence'][f"seed_{seed}/{e['condition']}"]=verify_training_unit(out,e,context,report,shared_seen)
        if mode=='training':continue
        labels={(p,t):check.labels(frame,t) for p,frame in frames.items() for t in TARGETS}
        indices=check.read(directory/'indices.json');fit_indices={}
        assert indices==check.read(ROOT/cfg['coalition_reference_results']/f'seed_{seed}/indices.json')
        for kind,pool,order,cap,base in [('utility','downstream_fit',UTILITY_TASKS,2048,1230000),('audit','attacker_fit',TARGETS,4096,1240000)]:
            for j,target in enumerate(order):
                _,valid=labels[pool,target];ix=np.random.default_rng(base+100*seed+j).permutation(np.flatnonzero(valid))[:cap]
                fit_indices[kind,target]=ix
                assert indices['task_fit_indices' if kind=='utility' else 'audit_fit_indices'][target]=={
                    'n':len(ix),'pool_indices_sha256':check.array_hash(ix),'raw_rows_sha256':check.array_hash(rows[pool][ix])}
        terminals={};unit_evidence={}
        for e in local:
            condition=e['condition'];dest=directory/condition
            complete=check_completion(dest/'complete.json',report,hashed)
            assert complete['seed']==seed and complete['condition']==condition and complete['evaluation_complete']
            assert complete['global_release_manifest_sha256']==global_hash
            checkpoint=torch.load(ROOT/e['checkpoint_path'],map_location='cpu',weights_only=True)
            training=check.read(directory/'training'/condition/'training.json')
            assert check.state_hash(checkpoint['model_state'])==training['final_hashes']['model']
            assert check.state_hash(checkpoint['adversary_state'])==training['final_hashes']['adversaries']
            assert (training['interface'],training['regime'],training['beta'])==condition_identity(condition)
            with np.load(dest/'releases.npz') as z:releases={p:z[p].copy() for p in z.files}
            verify_release_arrays(releases,checkpoint,pca,condition,e['outputs'],report)
            verify_condition(dest,seed,condition,checkpoint,releases,global_freeze,global_hash,complete,
                fit_indices,frames,labels,rows,report,cache,score_cache,seen,terminals)
            verify_native(dest,{condition:checkpoint},pca,frames,report)
            unit_evidence[condition]={'complete_sha256':check.sha(dest/'complete.json'),'checkpoint_sha256':e['checkpoint_sha256']}
            cache.clear();score_cache.clear()
        report['seed_evidence'][str(seed)]={'units':unit_evidence,'actual_training_checkpoints':terminals}
    if mode!='training':
        n_f=sum(e['interface']=='F' for e in units);n_p=len(units)-n_f
        expected_counts={'candidate_records':362*n_f+212*n_p,'prediction_sets':2*(362*n_f+212*n_p),
            'score_dictionaries':4*(362*n_f+212*n_p),'unique_saved_candidate_paths':257*n_f+147*n_p,
            'selection_pools':66*len(units),'inherited_prediction_sets':192*n_f+112*n_p,
            'literal_release_arrays':42*len(units),'public_head_composition_arrays':21*n_f,
            'native_predictions':6*len(units),'native_score_dictionaries':12*len(units),
            'mlp_schedules':111*n_f+67*n_p,'actual_terminal_checkpoints':106*n_f+62*n_p,
            'nested_trajectory_prefixes':53*n_f+31*n_p,'saved_observer_fidelity':9*len(units),
            'inherited_exposure_checks':27*len(units)}
        assert all(report[k]==v for k,v in expected_counts.items()),{k:(report[k],v) for k,v in expected_counts.items() if report[k]!=v}
        report['expected_score_scope_counts']=expected_counts
    if mode!='scores':
        assert report['training_units_verified']==len(units)
        assert report['guard_steps_replayed']==5*sum(e['family']!='T' for e in units)
        assert report['unique_source_only_steps_replayed']+report['source_only_alias_steps_checked']==5*sum(e['family']=='T' for e in units)
    report.update(passed=not check.errors,errors=check.errors,seeds=sorted({e['seed'] for e in units}),
        new_units_verified=len(units),complete_matrix=len(units)==24,numeric_values_compared=check.numeric_comparisons,
        max_absolute_metric_error=check.max_error,max_error_path=check.max_error_path,runtime_seconds=time.perf_counter()-tick)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_source_guard_v1')
    parser.add_argument('--report',type=Path);parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--max-units',type=int);parser.add_argument('--mode',choices=('all','training','scores'),default='all')
    args=parser.parse_args()
    if args.report and args.report.exists():raise FileExistsError('Preserve completed replay evidence')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=verify(args.out.resolve(),args.seeds,args.max_units,args.mode)
    rendered=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle:handle.write(rendered)
    else:print(rendered,end='')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
