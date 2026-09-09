"""Current-displacement source guards from the exact reviewed coalition fork.

The guard changes parameters after two disposable literal Adam proposals.  Only
the full proposal's once-advanced moments are retained.  Historical trainers and
their objective/configuration contracts are not modified.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import acs_bottleneck_training as old
from experiments import acs_coalition_training as historical
from experiments import acs_coalition_strength_training as strength
from experiments.acs_transfer_models import array_digest, state_digest

TASKS = ('income_binary', 'civilian_at_work', 'public_coverage')
GUARDED = (('J', .1), ('Iplus', .025), ('Iplus', .2))
DIAGNOSTIC_EPOCHS = (1, 20, 40, 60, 80)
SVD_RCOND = 1e-12
KKT_RTOL = 1e-12
ACTIVE_SUBSETS = tuple(sorted(tuple(i for i in range(3) if mask & (1 << i)) for mask in range(8)))


def project_three_halfspaces(r, gradients):
    """Nearest v to r with Gv<=0; enumerate eight fixed active subsets.

    Rank is resolved on the active *gradient matrix*, never its Gram matrix.
    All inputs/calculations are float64.  Failure is explicit, with no fallback.
    """
    r = np.asarray(r, dtype=np.float64)
    g = np.asarray(gradients, dtype=np.float64)
    if r.ndim != 1 or g.shape != (3, len(r)) or not len(r):
        raise ValueError('Three source gradients and a nonempty displacement are required')
    if not np.isfinite(r).all() or not np.isfinite(g).all():
        raise FloatingPointError('Nonfinite projection inputs')
    row_norms = np.linalg.norm(g, axis=1)
    candidates, rejected = [], []
    for active in ACTIVE_SUBSETS:
        multipliers = np.zeros(3, dtype=np.float64)
        if active:
            a = g[list(active)]
            u, singular, vt = np.linalg.svd(a, full_matrices=False)
            keep = singular > (SVD_RCOND * singular[0] if singular.size and singular[0] else 0.)
            rank = int(keep.sum())
            coordinates = vt[keep] @ r
            v = r - vt[keep].T @ coordinates
            multipliers[list(active)] = u[:, keep] @ (coordinates / singular[keep])
        else:
            singular = np.empty(0, dtype=np.float64)
            rank, v = 0, r.copy()
        if not np.isfinite(v).all() or not np.isfinite(multipliers).all():
            raise FloatingPointError('Nonfinite active-set solution')
        dots = g @ v
        primal_tolerance = KKT_RTOL * np.maximum(1., row_norms * np.linalg.norm(v))
        dual_tolerance = KKT_RTOL * max(1., float(np.linalg.norm(multipliers)))
        normal = g.T @ multipliers
        stationarity = float(np.linalg.norm(v-r+normal))
        stationarity_tolerance = KKT_RTOL * max(1., float(np.linalg.norm(v)), float(np.linalg.norm(r)), float(np.linalg.norm(normal)))
        complementary = np.abs(multipliers*dots)
        complementarity_tolerance = KKT_RTOL * np.maximum(1., np.abs(multipliers)*row_norms*np.linalg.norm(v))
        valid = (bool(np.all(dots <= primal_tolerance)) and bool(np.all(multipliers >= -dual_tolerance))
                 and stationarity <= stationarity_tolerance
                 and bool(np.all(complementary <= complementarity_tolerance)))
        if not valid:
            rejected.append(list(active))
            continue
        distance_squared = float(np.dot(v-r, v-r))
        candidates.append((active, v, {'active_subset': list(active), 'active_rank': rank,
            'active_singular_values': singular.tolist(), 'multipliers': multipliers.tolist(),
            'ideal_dots': dots.tolist(), 'primal_tolerances': primal_tolerance.tolist(),
            'dual_tolerance': dual_tolerance, 'stationarity_l2': stationarity,
            'stationarity_tolerance': stationarity_tolerance, 'complementarity_abs': complementary.tolist(),
            'complementarity_tolerances': complementarity_tolerance.tolist(),
            'distance_squared': distance_squared}))
    if not candidates:
        raise FloatingPointError('No active-set candidate passed all fixed KKT checks')
    best = min(item[2]['distance_squared'] for item in candidates)
    tie_tolerance = KKT_RTOL * max(1., best)
    chosen = next(item for item in candidates if item[2]['distance_squared'] <= best+tie_tolerance)
    _, v, record = chosen
    record.update(svd_rcond=SVD_RCOND, numerical_tolerance=KKT_RTOL,
        active_subset_order=[list(a) for a in ACTIVE_SUBSETS], valid_candidate_count=len(candidates),
        rejected_subsets=rejected, minimum_candidate_distance_squared=best,
        distance_tie_tolerance=tie_tolerance, tie_break='fixed lexicographic active subset',
        gradient_row_l2=row_norms.tolist(), raw_dots=(g @ r).tolist(),
        raw_l2=float(np.linalg.norm(r)), ideal_l2=float(np.linalg.norm(v)),
        projection_distance=float(np.linalg.norm(v-r)))
    return v, record


def disposable_adam(parameters, optimizer, gradients):
    """One literal torch Adam step on copies, retaining None versus zero grads."""
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise ValueError('The reviewed single-group literal Adam optimizer is required')
    if len(parameters) != len(gradients):
        raise ValueError('Parameter/gradient length mismatch')
    copies = [torch.nn.Parameter(p.detach().clone(), requires_grad=True) for p in parameters]
    proposal = torch.optim.Adam(copies, lr=optimizer.param_groups[0]['lr'])
    proposal.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for p, grad in zip(copies, gradients):
        p.grad = None if grad is None else grad.detach().clone()
    proposal.step()
    return [p.detach().clone() for p in copies], copy.deepcopy(proposal.state_dict())


def _flat(values, parameters=None):
    if parameters is None:
        return torch.cat([v.detach().reshape(-1).to(torch.float64) for v in values]).numpy()
    return torch.cat([(torch.zeros_like(p) if v is None else v).detach().reshape(-1).to(torch.float64)
                      for p, v in zip(parameters, values)]).numpy()


def _support(named_parameters, interface):
    if interface not in ('F', 'P'):
        raise ValueError('Frozen F/P interface required')
    included = [interface == 'P' or '.mapper.' in name for name, _ in named_parameters]
    mask = np.concatenate([np.full(p.numel(), use, dtype=bool) for (_, p), use in zip(named_parameters, included)])
    return mask, [name for (name, _), use in zip(named_parameters, included) if use]


def _unflat(vector, parameters):
    result, start = [], 0
    for p in parameters:
        end = start+p.numel()
        result.append(torch.from_numpy(np.array(vector[start:end], copy=True)).reshape(p.shape).to(p.dtype))
        start = end
    if start != len(vector):
        raise ValueError('Parameter vector length mismatch')
    return result


@torch.no_grad()
def _proposal_losses(model, parameters, proposed, x, source):
    current = [p.detach().clone() for p in parameters]
    try:
        for p, value in zip(parameters, proposed):
            p.copy_(value)
        loss, tasks, _, support = historical.source_loss(model.forward_parts(x)[1], source)
        return {'source': float(loss), 'tasks': {k: float(v) for k, v in tasks.items()}, 'support': support}
    finally:
        for p, value in zip(parameters, current):
            p.copy_(value)


def guard_step(model, observers, optimizer, x, source, attributes, priors, interface, regime, beta, *, capture=False):
    """Commit theta_source+projected current increment and full moments once."""
    if (regime, beta) not in GUARDED:
        raise ValueError('Only the three predeclared guarded conditions are allowed')
    optimizer.zero_grad(set_to_none=True)
    observers.zero_grad(set_to_none=True)
    named = list(model.named_parameters()); parameters = [p for _, p in named]
    mask, support_names = _support(named, interface)
    initial = [p.detach().clone() for p in parameters]
    theta = _flat(initial)
    rng = torch.get_rng_state().clone()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    flags = [p.requires_grad for p in observers.parameters()]
    observers.requires_grad_(False)
    try:
        losses, detail = historical.components(model, observers, x, source, attributes, priors, interface)
        full = losses['source']
        coeff = strength.coefficients(regime, beta)
        for name, coefficient in coeff.items():
            if coefficient:
                full = full + coefficient*losses[name]
        if not all(torch.isfinite(v) for v in (*losses.values(), full)):
            raise FloatingPointError('Nonfinite source/full objective')
        source_grad = torch.autograd.grad(losses['source'], parameters, retain_graph=True, allow_unused=True)
        task_grad = {task: torch.autograd.grad(detail['tasks'][task], parameters, retain_graph=True, allow_unused=True) for task in TASKS}
        full_grad = torch.autograd.grad(full, parameters, allow_unused=True)
    finally:
        for p, flag in zip(observers.parameters(), flags):
            p.requires_grad_(flag)
    theta_source, source_state = disposable_adam(parameters, optimizer, source_grad)
    theta_full, full_state = disposable_adam(parameters, optimizer, full_grad)
    source_flat, full_flat = _flat(theta_source), _flat(theta_full)
    if not np.isfinite(source_flat).all() or not np.isfinite(full_flat).all():
        raise FloatingPointError('Nonfinite Adam proposal')
    if not np.array_equal(source_flat[~mask], full_flat[~mask]):
        raise AssertionError('Protection Adam proposals differ outside the original gradient support')
    r = (full_flat-source_flat)[mask]
    full_task_gradients = np.stack([_flat(task_grad[task], parameters) for task in TASKS])
    gradients = full_task_gradients[:, mask]
    v, projection = project_three_halfspaces(r, gradients)
    accepted64 = source_flat.copy(); accepted64[mask] += v
    accepted = _unflat(accepted64, parameters)
    stored = _flat(accepted)
    if not np.isfinite(stored).all() or not np.array_equal(stored[~mask], source_flat[~mask]):
        raise FloatingPointError('Invalid stored update/support')
    realized = (stored-source_flat)[mask]
    cast_error = realized-v
    actual_dots = gradients @ realized
    cast_bound = np.abs(gradients) @ np.abs(cast_error)
    ideal_dots = gradients @ v
    # Standard gamma_n dot-product bound, also allowing rounding in e=actual-v
    # and in the scalar subtraction actual_dot-ideal_dot.  KKT feasibility has
    # its own separately recorded tolerance; it is not called cast roundoff.
    unit_roundoff = float(np.finfo(np.float64).eps/2.)
    gamma_n = len(v)*unit_roundoff/(1.-len(v)*unit_roundoff)
    gamma_one = unit_roundoff/(1.-unit_roundoff)
    absolute_realized = np.abs(gradients) @ np.abs(realized)
    absolute_ideal = np.abs(gradients) @ np.abs(v)
    floating_bound = (gamma_n*(absolute_realized+absolute_ideal+cast_bound)
        + gamma_one*(absolute_realized+absolute_ideal+np.abs(actual_dots)+np.abs(ideal_dots)))
    primal_bound = np.asarray(projection['primal_tolerances'], dtype=np.float64)
    if np.any(actual_dots-ideal_dots > cast_bound+floating_bound):
        raise FloatingPointError('Stored-minus-ideal dot exceeds its declared casting/summation bound')
    if np.any(actual_dots > primal_bound+cast_bound+floating_bound):
        raise FloatingPointError('Stored float32 guard violates its declared casting bound')
    record = {'coefficients': {'source': 1., **coeff}, 'interface': interface, 'regime': regime, 'beta': float(beta),
        'source_support': detail['source_support'], 'zero_constraint_tasks': [task for task in TASKS if not detail['source_support'][task]],
        'parameter_names': [name for name, _ in named], 'support_parameter_names': support_names,
        'support_dimension': int(mask.sum()), 'outside_support_dimension': int((~mask).sum()),
        'source_displacement_l2': float(np.linalg.norm(source_flat-theta)),
        'full_displacement_l2': float(np.linalg.norm(full_flat-theta)),
        'source_displacement_dots': (gradients @ (source_flat-theta)[mask]).tolist(),
        'ideal_total_displacement_dots': (gradients @ ((source_flat-theta)[mask]+v)).tolist(),
        'actual_total_displacement_dots': (gradients @ ((stored-theta)[mask])).tolist(),
        'displacement_dot_support': 'Original protection support only; F excludes ordinary source-head displacement',
        'full_source_displacement_dots': (full_task_gradients @ (source_flat-theta)).tolist(),
        'full_ideal_total_displacement_dots': (full_task_gradients @ (accepted64-theta)).tolist(),
        'full_actual_total_displacement_dots': (full_task_gradients @ (stored-theta)).tolist(),
        'full_displacement_dot_support': 'All forward parameters, including ordinary F source-head displacement',
        'realized_increment_l2': float(np.linalg.norm(realized)), 'cast_error_l2': float(np.linalg.norm(cast_error)),
        'cast_error_max_abs': float(np.max(np.abs(cast_error))), 'actual_dots': actual_dots.tolist(),
        'cast_dot_bounds': cast_bound.tolist(), 'float64_roundoff_bounds': floating_bound.tolist(),
        'float64_unit_roundoff': unit_roundoff, 'float64_dot_gamma_n': gamma_n,
        'kkt_primal_bounds': primal_bound.tolist(), 'actual_minus_ideal_dots': (actual_dots-ideal_dots).tolist(),
        'cast_plus_roundoff_bounds': (cast_bound+floating_bound).tolist(),
        'actual_dot_excess_bounds': (primal_bound+cast_bound+floating_bound).tolist(),
        'actual_dot_minus_bound': (actual_dots-primal_bound-cast_bound-floating_bound).tolist(),
        'projection': projection, 'full_objective': float(full.detach()),
        'source_objective': float(losses['source'].detach()),
        'loss_components': {k: float(value.detach()) for k, value in losses.items()},
        'source_task_losses': {k: float(value.detach()) for k, value in detail['tasks'].items()},
        'source_proposal_is_inherited_moment_proposal': True, 'outside_support_proposals_identical': True,
        'retained_optimizer': 'full-objective Adam moments and once-incremented steps',
        'projection_changes_optimizer_moments': False, 'cast_correction_applied': False}
    evidence = None
    if capture:
        evidence = {'parameter_names': record['parameter_names'], 'support_mask': torch.from_numpy(mask.copy()),
            'pre_model_state': copy.deepcopy(model.state_dict()), 'pre_optimizer': optimizer_before,
            'source_gradients': [None if g is None else g.detach().clone() for g in source_grad],
            'full_gradients': [None if g is None else g.detach().clone() for g in full_grad],
            'task_gradients': {task: [None if g is None else g.detach().clone() for g in grads] for task, grads in task_grad.items()},
            'source_proposal': theta_source, 'full_proposal': theta_full,
            'source_proposal_optimizer': source_state, 'full_proposal_optimizer': full_state,
            'source_displacement': torch.from_numpy(source_flat-theta), 'raw_increment': torch.from_numpy(r),
            'task_gradient_matrix': torch.from_numpy(gradients), 'ideal_increment': torch.from_numpy(v),
            'accepted_parameters': accepted, 'realized_increment': torch.from_numpy(realized),
            'cast_error': torch.from_numpy(cast_error),
            'native_losses': {name: _proposal_losses(model, parameters, values, x, source)
                              for name, values in (('pre', initial), ('source', theta_source), ('full', theta_full), ('accepted', accepted))},
            'torch_rng_state': rng.clone(), 'scalar': record}
        record['pre_optimizer_sha256'] = old.tree_digest(optimizer_before)
        record['source_proposal_optimizer_sha256'] = old.tree_digest(source_state)
        record['retained_optimizer_sha256'] = old.tree_digest(full_state)
    if not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError('Guard proposal/projection changed RNG')
    with torch.no_grad():
        for parameter, value, grad in zip(parameters, accepted, full_grad):
            parameter.copy_(value)
            parameter.grad = None if grad is None else grad.detach().clone()
    optimizer.load_state_dict(full_state)
    if capture:
        evidence['post_model_state'] = copy.deepcopy(model.state_dict())
        evidence['post_optimizer'] = copy.deepcopy(optimizer.state_dict())
        if old.tree_digest(evidence['post_optimizer']) != record['retained_optimizer_sha256']:
            raise AssertionError('Full proposal moments were not retained exactly')
    return record, evidence


def source_step(model, optimizer, x, source, *, capture=False):
    """The literal ordinary source-only Adam path, with no float64 round trip."""
    evidence = {'pre_model_state': copy.deepcopy(model.state_dict()),
                'pre_optimizer': copy.deepcopy(optimizer.state_dict()),
                'torch_rng_state': torch.get_rng_state().clone()} if capture else None
    optimizer.zero_grad(set_to_none=True)
    loss, tasks, _, support = historical.source_loss(model.forward_parts(x)[1], source)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite source-only objective')
    loss.backward()
    if capture:
        evidence['source_gradients'] = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    optimizer.step()
    record = {'coefficients': {'source': 1., 'individual': 0., 'extra_local': 0., 'coalition': 0.},
        'source_objective': float(loss.detach()), 'source_task_losses': {k: float(v.detach()) for k, v in tasks.items()},
        'source_support': support, 'literal_source_only': True, 'float64_parameter_round_trip': False}
    if capture:
        evidence.update(post_model_state=copy.deepcopy(model.state_dict()), post_optimizer=copy.deepcopy(optimizer.state_dict()), scalar=record)
    return record, evidence


def _prepare(pca_fit, source_y, attr_y, pre, path, metadata, seed, interface, miniature, pca_val, source_val):
    if interface not in ('F', 'P') or seed not in (0, 1, 2):
        raise ValueError('Frozen interface/seed required')
    if metadata['seed'] != seed or metadata['miniature'] != miniature:
        raise ValueError('Historical seed/test mode mismatch')
    raw = old._features(pca_fit)
    source = old._labels(source_y, len(raw), historical.SOURCE_SCHEMA, 'source fitting')
    attributes = old._labels(attr_y, len(raw), historical.ATTRIBUTE_SCHEMA, 'attribute fitting')
    if (pca_val is None) != (source_val is None):
        raise ValueError('Source validation inputs/labels must both be provided')
    valraw = old._features(pca_val) if pca_val is not None else None
    validation = old._labels(source_val, len(valraw), historical.SOURCE_SCHEMA, 'source validation') if valraw is not None else None
    for name, actual in (('fit_input_sha256', array_digest(raw)),
                        ('source_label_hashes', {k: array_digest(y.numpy()) for k, y in source.items()}),
                        ('attribute_label_hashes', {k: array_digest(y.numpy()) for k, y in attributes.items()})):
        if metadata[name] != actual:
            raise ValueError('Historical fitting identity mismatch: '+name)
    if valraw is not None and (metadata['source_validation_input_sha256'] != array_digest(valraw)
            or metadata['source_validation_label_hashes'] != {k: array_digest(y.numpy()) for k, y in validation.items()}):
        raise ValueError('Historical source validation identity mismatch')
    config = copy.deepcopy(historical.CONFIG)
    config.update(study='acs_source_guard_v1', historical_training_recipe='acs_coalition_v1')
    if miniature:
        config.update(warm_base_epochs=1, warm_adversary_epochs=1, continuation_epochs=2, batch_size=16, curve_interval=1)
    for key in ('continuation_epochs', 'batch_size', 'lr', 'betas', 'eps', 'weight_decay', 'adversary_updates_per_mapper_step'):
        if metadata['config'][key] != config[key]:
            raise ValueError('Historical recipe mismatch: '+key)
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    schedule = checkpoint['schedule_state']
    if schedule['phase'] != 'continuation' or schedule['completed_epochs'] != 0 or schedule['next_minibatch_index'] != 0:
        raise ValueError('Only the exact pre-continuation historical fork is eligible')
    if schedule['schedules'] != metadata['schedules'] or schedule['rows'] != len(raw) or schedule['batch_size'] != config['batch_size']:
        raise ValueError('Historical complete schedule state mismatch')
    actual_schedule = metadata['schedules']['continuation']
    if actual_schedule['seed'] != 1280000+100*seed+200:
        raise ValueError('Historical continuation schedule seed mismatch')
    orders, order_hash = old._orders(len(raw), config['continuation_epochs'], actual_schedule['seed'])
    if order_hash != actual_schedule['sha256']:
        raise ValueError('Historical continuation permutation mismatch')
    model, observers, optimizer, observer_optimizer = strength.restore_fork(checkpoint, pre, interface, seed)
    hashes = historical._hashes(model, observers, optimizer, observer_optimizer)
    if hashes != metadata['arms'][interface+'_I']['fork_hashes']:
        raise ValueError('Not the exact reviewed common interface fork')
    x, xv = model.standardize(raw), model.standardize(valraw) if valraw is not None else None
    if array_digest(x.numpy()) != metadata['standardized_fit_sha256']:
        raise ValueError('Fitting preprocessing mismatch')
    priors = historical.label_priors(source, attributes)
    if priors != metadata['prior_entropies']:
        raise ValueError('Historical fitting prior mismatch')
    n_batch = (len(raw)+config['batch_size']-1)//config['batch_size']
    expected_counts = {'mapper_optimizer_steps': config['warm_base_epochs']*n_batch,
                       'adversary_optimizer_steps': config['warm_adversary_epochs']*n_batch}
    if checkpoint['counts'] != expected_counts:
        raise ValueError('Historical prefix exposure mismatch')
    return {'raw': raw, 'source': source, 'attributes': attributes, 'validation': validation,
        'valraw': valraw, 'x': x, 'xv': xv, 'priors': priors, 'checkpoint': checkpoint,
        'model': model, 'observers': observers, 'optimizer': optimizer, 'observer_optimizer': observer_optimizer,
        'config': config, 'orders': orders, 'order_hash': order_hash, 'counts': copy.deepcopy(expected_counts),
        'fork_hashes': hashes, 'fork_path': path, 'fork_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'fork_tree_sha256': old.tree_digest(checkpoint)}


def _epoch_summary(epoch, records):
    summary = {'epoch': epoch, 'mapper_steps': len(records)}
    if records[0].get('literal_source_only'):
        summary.update(literal_source_only=True, projection_steps=0,
            mean_source_objective=float(np.mean([r['source_objective'] for r in records])))
        return summary
    scalar_keys = ('source_displacement_l2', 'full_displacement_l2', 'realized_increment_l2', 'cast_error_l2', 'cast_error_max_abs')
    projection_keys = ('raw_l2', 'ideal_l2', 'projection_distance', 'stationarity_l2', 'distance_squared')
    for key in scalar_keys:
        values = [r[key] for r in records]
        summary[key] = {'min': min(values), 'max': max(values), 'mean': float(np.mean(values))}
    for key in projection_keys:
        values = [r['projection'][key] for r in records]
        summary[key] = {'min': min(values), 'max': max(values), 'mean': float(np.mean(values))}
    for field, container in (('raw_dots', 'projection'), ('ideal_dots', 'projection'),
                              ('actual_dots', None), ('cast_dot_bounds', None), ('actual_dot_minus_bound', None),
                              ('source_displacement_dots', None), ('ideal_total_displacement_dots', None),
                              ('actual_total_displacement_dots', None), ('full_source_displacement_dots', None),
                              ('full_ideal_total_displacement_dots', None), ('full_actual_total_displacement_dots', None)):
        values = np.array([(r[container] if container else r)[field] for r in records], dtype=np.float64)
        summary[field] = {'min': values.min(0).tolist(), 'max': values.max(0).tolist(), 'mean': values.mean(0).tolist()}
    summary['active_subset_counts'] = {}
    summary['active_rank_counts'] = {}
    for r in records:
        active = ','.join(map(str, r['projection']['active_subset'])) or 'empty'
        rank = str(r['projection']['active_rank'])
        summary['active_subset_counts'][active] = summary['active_subset_counts'].get(active, 0)+1
        summary['active_rank_counts'][rank] = summary['active_rank_counts'].get(rank, 0)+1
    summary['zero_eligible_task_steps'] = {task: sum(task in r['zero_constraint_tasks'] for r in records) for task in TASKS}
    summary['support_dimension'] = records[0]['support_dimension']
    summary['outside_support_proposals_identical_all_steps'] = all(r['outside_support_proposals_identical'] for r in records)
    summary['projection_steps'] = len(records)
    return summary


def _step_record(record, epoch, start, index, counts):
    result = {'epoch': epoch, 'minibatch_start': start, 'batch_indices_sha256': array_digest(index),
        'counts': copy.deepcopy(counts), 'source_support': record['source_support'],
        'coefficients': record['coefficients'],
        'source_objective': record['source_objective'], 'source_task_losses': record['source_task_losses']}
    if record.get('literal_source_only'):
        return {**result, 'literal_source_only': True, 'projection_steps': 0}
    for key in ('zero_constraint_tasks', 'support_dimension', 'outside_support_dimension',
        'source_displacement_l2', 'full_displacement_l2', 'realized_increment_l2', 'cast_error_l2', 'cast_error_max_abs',
        'source_displacement_dots', 'ideal_total_displacement_dots', 'actual_total_displacement_dots',
        'full_source_displacement_dots', 'full_ideal_total_displacement_dots', 'full_actual_total_displacement_dots',
        'actual_dots', 'cast_dot_bounds', 'float64_roundoff_bounds', 'kkt_primal_bounds', 'actual_minus_ideal_dots',
        'cast_plus_roundoff_bounds', 'actual_dot_excess_bounds', 'actual_dot_minus_bound', 'full_objective', 'loss_components'):
        result[key] = record[key]
    result['projection'] = {key: record['projection'][key] for key in ('active_subset', 'active_rank',
        'multipliers', 'raw_dots', 'ideal_dots', 'primal_tolerances', 'dual_tolerance',
        'stationarity_l2', 'stationarity_tolerance', 'complementarity_abs', 'complementarity_tolerances',
        'raw_l2', 'ideal_l2', 'projection_distance', 'distance_squared')}
    return result


def _save_detail(out, epoch, index, evidence, observers, observer_optimizer, counts, *, interface):
    path = out/f'diagnostic_epoch_{epoch:03d}.pt'
    if path.exists():
        raise FileExistsError(path)
    evidence.update(epoch=epoch, batch_indices=torch.from_numpy(index.copy()),
        batch_indices_sha256=array_digest(index), interface=interface,
        observers_after_scheduled_updates=copy.deepcopy(observers.state_dict()),
        observer_optimizer_after_scheduled_updates=copy.deepcopy(observer_optimizer.state_dict()),
        counts_after_update=copy.deepcopy(counts))
    torch.save(evidence, path)
    return {'epoch': epoch, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'batch_indices_sha256': array_digest(index), 'scalar': evidence['scalar'],
            'native_losses': evidence.get('native_losses')}


@torch.no_grad()
def _curve(ctx, interface, regime, beta, epoch):
    if regime != 'T':
        return strength.curve(ctx['model'], ctx['observers'], ctx['x'], ctx['source'], ctx['attributes'],
            ctx['priors'], interface, regime, beta, epoch, ctx['counts'], ctx['xv'], ctx['validation'])
    record = historical._curve(ctx['model'], ctx['observers'], ctx['x'], ctx['source'], ctx['attributes'],
        ctx['priors'], interface, None, epoch, ctx['counts'], ctx['xv'], ctx['validation'])
    record['coefficients'] = {'source': 1., 'individual': 0., 'extra_local': 0., 'coalition': 0.}
    return record


def _finish_metadata(ctx, previous, seed, interface, regime, beta, trajectory, summaries, detailed, started):
    config = ctx['config']; source = ctx['source']; attrs = ctx['attributes']
    source_passes = config['warm_base_epochs']+config['continuation_epochs']
    observer_passes = config['warm_adversary_epochs']+3*config['continuation_epochs']
    return {'study': 'acs_source_guard_v1', 'seed': int(seed), 'interface': interface, 'regime': regime,
        'beta': float(beta), 'miniature': previous['miniature'], 'config': config,
        'coefficients': {'individual': 0., 'extra_local': 0., 'coalition': 0.} if regime == 'T' else strength.coefficients(regime, beta),
        'guarded': regime != 'T', 'fit_pool': 'representation_fit', 'reserved_labels_received': False,
        'final_evaluation_received': False, 'historical_fork_path': str(ctx['fork_path']),
        'historical_fork_sha256': ctx['fork_sha256'], 'historical_fork_tree_sha256': ctx['fork_tree_sha256'],
        'historical_training_module_sha256': previous['module_source_sha256'], 'fork_hashes': ctx['fork_hashes'],
        'final_hashes': historical._hashes(ctx['model'], ctx['observers'], ctx['optimizer'], ctx['observer_optimizer']),
        'counts': ctx['counts'], 'schedules': previous['schedules'], 'schedule_sha256': ctx['order_hash'],
        'fit_input_sha256': array_digest(ctx['raw']), 'standardized_fit_sha256': array_digest(ctx['x'].numpy()),
        'source_label_hashes': previous['source_label_hashes'], 'attribute_label_hashes': previous['attribute_label_hashes'],
        'source_validation_input_sha256': array_digest(ctx['valraw']) if ctx['valraw'] is not None else None,
        'source_validation_label_hashes': previous['source_validation_label_hashes'] if ctx['validation'] is not None else {},
        'preprocessing': previous['preprocessing'], 'prior_entropies': ctx['priors'],
        'source_fit_coverage': previous['source_fit_coverage'], 'attribute_fit_coverage': previous['attribute_fit_coverage'],
        'observer_roles': previous['observer_roles'][interface], 'observer_optimization': previous['observer_optimization'],
        'curve': trajectory, 'epoch_update_summaries': summaries, 'detailed_steps': detailed,
        'diagnostic_epochs': sorted({d['epoch'] for d in detailed}), 'caller_rng_unchanged': True,
        'historical_fork_unchanged': True, 'warmup_refitted': False, 'selected_epoch': config['continuation_epochs'],
        'selection': 'fixed final iterate; source validation diagnostic only', 'mapper_exposure_per_row': source_passes,
        'observer_exposure_per_row': observer_passes,
        'source_valid_label_exposures': {target: int((y>=0).sum())*source_passes for target, y in source.items()},
        'observer_valid_label_exposures': {role: int(({**source, **attrs})[role.split('__')[1]].ge(0).sum())*observer_passes for role in historical.ROLE_SCHEMA},
        'module_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runtime_seconds': time.perf_counter()-started}


def _save_failure(out, ctx, epoch, start, error):
    path = out/'failure.pt'
    if not path.exists():
        torch.save({'model_state': ctx['model'].state_dict(), 'adversary_state': ctx['observers'].state_dict(),
            'mapper_optimizer': ctx['optimizer'].state_dict(), 'adversary_optimizer': ctx['observer_optimizer'].state_dict(),
            'counts': ctx['counts'], 'torch_rng_state': torch.get_rng_state(), 'epoch': epoch,
            'next_minibatch_start': start, 'schedule_state': ctx['checkpoint']['schedule_state']}, path)
    (out/'failure.json').write_text(json.dumps({'error_type': type(error).__name__, 'message': str(error),
        'epoch': epoch, 'minibatch_start': start, 'diagnostic_checkpoint': str(path),
        'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scientifically_complete': False}, indent=2)+'\n')


def train_guarded_continuation(pca_fit, source_y, attr_y, pre, fork_checkpoint, historical_metadata, seed, out, *,
                              interface, regime, beta, pca_val=None, source_val=None, miniature=False):
    started = time.perf_counter()
    if (regime, beta) not in GUARDED:
        raise ValueError('Only the three predeclared guarded conditions are allowed')
    ctx = _prepare(pca_fit, source_y, attr_y, pre, fork_checkpoint, historical_metadata, seed, interface, miniature, pca_val, source_val)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    model, observers, optimizer, observer_optimizer = (ctx[k] for k in ('model', 'observers', 'optimizer', 'observer_optimizer'))
    config = ctx['config']; counts = ctx['counts']; rng = torch.get_rng_state().clone()
    details, summaries, trajectory = [], [], []
    epoch, start = 0, 0
    step_path = out/'step_diagnostics.jsonl.gz'
    with gzip.open(step_path, 'xt') as step_stream, torch.random.fork_rng(devices=[]):
        torch.set_rng_state(ctx['checkpoint']['torch_rng_state'])
        historical._checkpoint(out/'fork.pt', model, observers, optimizer, observer_optimizer, counts,
                               'exact interface fork', schedule_state=ctx['checkpoint']['schedule_state'])
        if old.tree_digest(torch.load(out/'fork.pt', weights_only=True)) != ctx['fork_tree_sha256']:
            raise AssertionError('Full inherited fork was not preserved exactly')
        trajectory.append(_curve(ctx, interface, regime, beta, 0))
        diagnostic_epochs = set(DIAGNOSTIC_EPOCHS if not miniature else (1, config['continuation_epochs']))
        try:
            for epoch, order in enumerate(ctx['orders'], 1):
                records = []
                for start in range(0, len(order), config['batch_size']):
                    index = order[start:start+config['batch_size']]
                    x = ctx['x'][index]; source = old._batch(ctx['source'], index); attrs = old._batch(ctx['attributes'], index)
                    for _ in range(3):
                        historical.observer_update(model, observers, observer_optimizer, x, {**source, **attrs}, ctx['priors'], interface)
                        counts['adversary_optimizer_steps'] += 1
                    capture = start == 0 and epoch in diagnostic_epochs
                    record, evidence = guard_step(model, observers, optimizer, x, source, attrs, ctx['priors'], interface, regime, beta, capture=capture)
                    counts['mapper_optimizer_steps'] += 1
                    records.append(record)
                    step_stream.write(json.dumps(_step_record(record, epoch, start, index, counts), separators=(',', ':'), allow_nan=False)+'\n')
                    if capture:
                        details.append(_save_detail(out, epoch, index, evidence, observers, observer_optimizer, counts, interface=interface))
                summaries.append(_epoch_summary(epoch, records))
                if epoch % config['curve_interval'] == 0 or epoch == config['continuation_epochs']:
                    trajectory.append(_curve(ctx, interface, regime, beta, epoch))
        except Exception as error:
            _save_failure(out, ctx, epoch, start, error)
            raise
        final_schedule = copy.deepcopy(ctx['checkpoint']['schedule_state']); final_schedule['completed_epochs'] = config['continuation_epochs']
        historical._checkpoint(out/'final.pt', model, observers, optimizer, observer_optimizer, counts,
                               'fixed final iterate', schedule_state=final_schedule)
        if not torch.equal(ctx['checkpoint']['torch_rng_state'], torch.get_rng_state()):
            raise AssertionError('Deterministic training RNG advanced unexpectedly')
    if not torch.equal(rng, torch.get_rng_state()) or hashlib.sha256(ctx['fork_path'].read_bytes()).hexdigest() != ctx['fork_sha256']:
        raise AssertionError('Caller RNG or historical checkpoint changed')
    metadata = _finish_metadata(ctx, historical_metadata, seed, interface, regime, beta, trajectory, summaries, details, started)
    metadata['all_step_diagnostics'] = {'path': str(step_path), 'sha256': hashlib.sha256(step_path.read_bytes()).hexdigest(),
        'format': 'gzip JSONL, one compact record per actual forward optimizer update',
        'rows': sum(s['mapper_steps'] for s in summaries)}
    (out/'training.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return {'model': model.freeze(), 'observers': observers.eval().requires_grad_(False), 'metadata': metadata}


def train_source_only_shared(pca_fit, source_y, attr_y, pre, fork_checkpoints, historical_metadata, seed, out, *,
                             pca_val=None, source_val=None, miniature=False):
    """One forward/Adam trajectory; independent F/P observers on each pre-step model."""
    started = time.perf_counter()
    if set(fork_checkpoints) != {'F', 'P'}:
        raise ValueError('Both exact F/P observer forks are required')
    contexts = {interface: _prepare(pca_fit, source_y, attr_y, pre, fork_checkpoints[interface], historical_metadata,
        seed, interface, miniature, pca_val, source_val) for interface in ('F', 'P')}
    f, p = contexts['F'], contexts['P']
    for key in ('model_state', 'mapper_optimizer', 'torch_rng_state', 'schedule_state'):
        if old.tree_digest(f['checkpoint'][key]) != old.tree_digest(p['checkpoint'][key]):
            raise AssertionError('F/P source-only common '+key+' identity is absent')
    if f['order_hash'] != p['order_hash'] or f['counts']['mapper_optimizer_steps'] != p['counts']['mapper_optimizer_steps']:
        raise AssertionError('F/P source schedule/forward exposure differs')
    model, optimizer = f['model'], f['optimizer']
    p['model'], p['optimizer'] = model, optimizer
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    paths = {interface: out/(interface+'_T') for interface in ('F', 'P')}
    for path in paths.values(): path.mkdir(exist_ok=False)
    config = f['config']; rng = torch.get_rng_state().clone()
    interface_rng = {interface: ctx['checkpoint']['torch_rng_state'].clone() for interface, ctx in contexts.items()}
    details = {i: [] for i in contexts}; trajectories = {i: [] for i in contexts}; summaries = []
    epoch, start = 0, 0
    step_path = out/'T_shared_step_diagnostics.jsonl.gz'
    with gzip.open(step_path, 'xt') as step_stream, torch.random.fork_rng(devices=[]):
        torch.set_rng_state(f['checkpoint']['torch_rng_state'])
        for interface, ctx in contexts.items():
            historical._checkpoint(paths[interface]/'fork.pt', model, ctx['observers'], optimizer, ctx['observer_optimizer'],
                ctx['counts'], 'exact interface fork', schedule_state=ctx['checkpoint']['schedule_state'])
            if old.tree_digest(torch.load(paths[interface]/'fork.pt', weights_only=True)) != ctx['fork_tree_sha256']:
                raise AssertionError('Shared T full interface fork was not preserved')
            trajectories[interface].append(_curve(ctx, interface, 'T', 0., 0))
        diagnostic_epochs = set(DIAGNOSTIC_EPOCHS if not miniature else (1, config['continuation_epochs']))
        try:
            for epoch, order in enumerate(f['orders'], 1):
                records = []
                for start in range(0, len(order), config['batch_size']):
                    index = order[start:start+config['batch_size']]
                    x = f['x'][index]; source = old._batch(f['source'], index); attrs = old._batch(f['attributes'], index)
                    for interface, ctx in contexts.items():
                        with torch.random.fork_rng(devices=[]):
                            torch.set_rng_state(interface_rng[interface])
                            for _ in range(3):
                                historical.observer_update(model, ctx['observers'], ctx['observer_optimizer'], x, {**source, **attrs}, ctx['priors'], interface)
                                ctx['counts']['adversary_optimizer_steps'] += 1
                            interface_rng[interface] = torch.get_rng_state().clone()
                    capture = start == 0 and epoch in diagnostic_epochs
                    for ctx in contexts.values():
                        ctx['observers'].zero_grad(set_to_none=True)
                    record, evidence = source_step(model, optimizer, x, source, capture=capture)
                    records.append(record)
                    for interface, ctx in contexts.items():
                        ctx['counts']['mapper_optimizer_steps'] += 1
                        if capture:
                            details[interface].append(_save_detail(paths[interface], epoch, index, copy.deepcopy(evidence),
                                ctx['observers'], ctx['observer_optimizer'], ctx['counts'], interface=interface))
                    step_stream.write(json.dumps(_step_record(record, epoch, start, index, f['counts']), separators=(',', ':'), allow_nan=False)+'\n')
                summaries.append(_epoch_summary(epoch, records))
                if epoch % config['curve_interval'] == 0 or epoch == config['continuation_epochs']:
                    for interface, ctx in contexts.items():
                        trajectories[interface].append(_curve(ctx, interface, 'T', 0., epoch))
        except Exception as error:
            for interface, ctx in contexts.items(): _save_failure(paths[interface], ctx, epoch, start, error)
            raise
        for interface, ctx in contexts.items():
            if not torch.equal(interface_rng[interface], ctx['checkpoint']['torch_rng_state']):
                raise AssertionError('Detached observer unexpectedly advanced its isolated prescribed RNG')
            final_schedule = copy.deepcopy(ctx['checkpoint']['schedule_state']); final_schedule['completed_epochs'] = config['continuation_epochs']
            historical._checkpoint(paths[interface]/'final.pt', model, ctx['observers'], optimizer, ctx['observer_optimizer'],
                ctx['counts'], 'fixed final iterate', schedule_state=final_schedule)
        if not torch.equal(f['checkpoint']['torch_rng_state'], torch.get_rng_state()):
            raise AssertionError('Shared forward training RNG changed')
    if not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError('Shared source-only changed caller RNG')
    shared = {'forward_alias': f'seed_{seed}/T_shared', 'unique_new_forward_continuations': 1,
        'interface_systems': 2, 'same_pre_step_forward_for_both_observer_sets': True,
        'isolated_interface_rng': True, 'new_forward_optimizer_steps': len(f['orders'])*len(range(0, len(f['x']), config['batch_size'])),
        'initial_forward_state_sha256': old.tree_digest((f['checkpoint']['model_state'], f['checkpoint']['mapper_optimizer'])),
        'final_forward_state_sha256': old.tree_digest((model.state_dict(), optimizer.state_dict())),
        'source_only_no_float64_roundtrip': True, 'runtime_seconds': time.perf_counter()-started,
        'all_step_diagnostics': {'path': str(step_path), 'sha256': hashlib.sha256(step_path.read_bytes()).hexdigest(),
            'rows': sum(s['mapper_steps'] for s in summaries), 'format': 'gzip JSONL; one actual shared source-only step per row'}}
    arms = {}
    for interface, ctx in contexts.items():
        if hashlib.sha256(ctx['fork_path'].read_bytes()).hexdigest() != ctx['fork_sha256']:
            raise AssertionError('Historical source-only fork changed')
        metadata = _finish_metadata(ctx, historical_metadata, seed, interface, 'T', 0., trajectories[interface], summaries, details[interface], started)
        metadata['shared_source_only'] = shared
        (paths[interface]/'training.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        arms[interface] = {'model': model, 'observers': ctx['observers'].eval().requires_grad_(False), 'metadata': metadata}
    model.freeze()
    return {'arms': arms, 'metadata': shared}
