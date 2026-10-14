"""ROOT-authorized, queue-only five-arm separate-pool continuation for one seed.

Receives only the physical UPDATE and GUARD exports. VALIDATION is never a
source of this worker; there are no in-training utility selection metrics.
Checkpoints 0/1/3/5 and label-free diagnostics are saved for a later worker.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from .recipe import ARMS, RECIPE, GLOBAL_GUARD_POOL, canonical_sha, validate_recipe, validate_science_lock
from .wires import corrected_trial_wires, prepare_geometry, FrozenIncomeGuard, tensor_sha

PARAM_PREFIX = ('enc.1.', 'head.1.')


def _seed(source_seed, purpose):
    return int(hashlib.sha256(f'SPG1-separate-pool|{source_seed}|{purpose}'.encode()).hexdigest()[:8], 16)


def _guard():
    from .scheduler import assert_active_from_env
    active = assert_active_from_env()
    if active['stage'] not in ('admission', 'protection'):
        raise PermissionError('trainer numerical work requires admission or protection queue')
    return active


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_export(item, fields, verify):
    """Read one physical role export; VALIDATION is not a trainer input."""
    import numpy as np
    from zipfile import ZipFile
    if 'VALIDATION' in Path(item['path']).as_posix().upper().split('/'):
        raise PermissionError('trainer cannot load VALIDATION')
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as archive:
        names = {x.filename for x in archive.infolist()}
    if names != {f + '.npy' for f in fields}:
        raise PermissionError('unexpected role export fields')
    with np.load(p, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in fields}


def load_roles(spec, verify):
    import numpy as np
    roles = {}
    want = {'UPDATE': ('features', 'occupation', 'sex'), 'GUARD': ('features', 'occupation')}
    if set(spec['roles']) != set(want):
        raise PermissionError('trainer receives exactly UPDATE and GUARD exports')
    for role, names in want.items():
        if set(spec['roles'][role]) != set(names):
            raise PermissionError(f'{role} export roster differs (GUARD SEX is not a fitting input)')
        f = load_export(spec['roles'][role]['features'], ('row_id', 'unit', 'X', 'feature_names'), verify)
        out = dict(f)
        for name in names[1:]:
            z = load_export(spec['roles'][role][name], ('row_id', 'unit', name), verify)
            if not (np.array_equal(z['row_id'], f['row_id']) and np.array_equal(z['unit'], f['unit'])):
                raise PermissionError('role export person pairing differs')
            out[name] = z[name]
        n = spec['role_rows'][role]
        if len(f['row_id']) != n or f['X'].shape != (n, 83) or f['X'].dtype != np.float32 or not np.isfinite(f['X']).all():
            raise PermissionError('role export rows differ from frozen split')
        roles[role] = out
    if set(map(str, roles['UPDATE']['unit'])) & set(map(str, roles['GUARD']['unit'])):
        raise PermissionError('UPDATE/GUARD groups overlap')
    return roles


class OccupationCritics:
    """Six actual-wire critics on UPDATE rows and UPDATE SEX, in every arm."""
    def __init__(self, source_seed, sex, geometry, frozen):
        import torch
        prior = torch.bincount(sex, minlength=2).double() / len(sex)
        if (prior <= 0).any():
            raise ValueError('both sensitive groups required in UPDATE')
        self.geometry, self.frozen, self.prior = geometry, frozen, prior
        self.entropy = float(-(prior * prior.log()).sum())
        self.banks, self.optimizers, self.references = [], [], None
        self.optimizer_steps = 0
        for view, width in enumerate((29, 50)):
            bank, optimizers = [], []
            for kind in RECIPE['critic_kinds']:
                torch.manual_seed(_seed(source_seed, f'critic|{view}|{kind}'))
                if kind == 'linear':
                    critic = torch.nn.Linear(width, 2, dtype=torch.float64)
                elif kind == 'mlp32':
                    critic = torch.nn.Sequential(torch.nn.Linear(width, 32, dtype=torch.float64),
                                                 torch.nn.ReLU(), torch.nn.Linear(32, 2, dtype=torch.float64))
                else:
                    critic = torch.nn.Sequential(torch.nn.Linear(width, 64, dtype=torch.float64), torch.nn.ReLU(),
                                                 torch.nn.Linear(64, 64, dtype=torch.float64), torch.nn.ReLU(),
                                                 torch.nn.Linear(64, 2, dtype=torch.float64))
                bank.append(critic)
                optimizers.append(torch.optim.Adam(critic.parameters(), lr=RECIPE['critic_learning_rate']))
            self.banks.append(bank); self.optimizers.append(optimizers)

    def refresh(self, model, fixed_x):
        import torch
        from task_conditioned_representation.kernels import DiagonalReference
        with torch.no_grad():
            self.frozen.verify()
            wires = corrected_trial_wires(model, fixed_x, self.geometry)
            values = [wires[1]['full'], torch.cat([w['full'] for w in wires], 1)]
            prior = self.references or [None, None]
            self.references = [DiagonalReference.fit(v, old) for v, old in zip(values, prior)]
        for r in self.references:
            if (not torch.isfinite(r.mean).all() or not torch.isfinite(r.scale).all() or (r.scale <= 0).any()):
                raise ValueError('finite positive full-coordinate reference required')
        return [{k: v.tolist() if hasattr(v, 'tolist') else v for k, v in r.snapshot().items()} for r in self.references]

    def views(self, model, x):
        import torch
        wires = corrected_trial_wires(model, x, self.geometry)
        values = [wires[1]['full'], torch.cat([w['full'] for w in wires], 1)]
        return [r.normalize(v) for r, v in zip(self.references, values)]

    def update(self, model, x, sex):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            views = self.views(model, x)
        for bank, optimizers, view in zip(self.banks, self.optimizers, views):
            for critic, optimizer in zip(bank, optimizers):
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(critic(view), sex)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite critic loss')
                loss.backward(); optimizer.step(); self.optimizer_steps += 1
        self.frozen.verify()

    def terms(self, model, x, sex):
        import torch
        import torch.nn.functional as F
        views = self.views(model, x)
        terms, ces = [], []
        for bank, view in zip(self.banks, views):
            for critic in bank:
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
            losses = torch.stack([F.cross_entropy(c(view), sex) for c in bank] + [view.sum() * 0 + self.entropy])
            terms.append(1 - losses.min() / self.entropy); ces.append(losses.detach().tolist())
        self.frozen.verify()
        return {'local1': terms[0], 'pair': terms[1], 'critic_ces': ces}

    def state_sha(self):
        return canonical_sha([[{k: tensor_sha(v) for k, v in c.state_dict().items()} for c in bank] for bank in self.banks])

    def state_dict(self):
        return {'critics': [[c.state_dict() for c in bank] for bank in self.banks],
                'optimizers': [[o.state_dict() for o in bank] for bank in self.optimizers],
                'references': self.references, 'prior': self.prior, 'entropy': self.entropy,
                'critic_optimizer_steps': self.optimizer_steps}


def _freeze(model, geometry):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PARAM_PREFIX))
    return FrozenIncomeGuard(model, geometry)


def _params(model):
    return list(model.enc[1].parameters()) + list(model.head[1].parameters())


def _draw(rng, n):
    return rng.sample(range(n), RECIPE['batch_size'])


class Task:
    """Shared population-weighted task objective, identical in every arm."""
    def __init__(self, n_update, n_guard):
        self.n_update, self.n_guard = n_update, n_guard
        self.w_update = n_update / (n_update + n_guard); self.w_guard = n_guard / (n_update + n_guard)

    def gradient(self, model, xb, yb, guard_pool, geometry):
        import torch.nn.functional as F
        from .strict import flat_gradient
        params = _params(model)
        ce = F.cross_entropy(corrected_trial_wires(model, xb, geometry)[1]['centered_logits'], yb)
        g_mb = flat_gradient(ce, params)
        return self.w_update * g_mb + self.w_guard * guard_pool['gradients']['ce'], float(ce.detach()), g_mb


def minibatch_guards(model, xb, yb, geometry):
    import torch
    import torch.nn.functional as F
    from .strict import flat_gradient
    import numpy as np
    params = _params(model)
    p = corrected_trial_wires(model, xb, geometry)[1]['probabilities']; rows = torch.arange(len(xb))
    nll = -p[rows, yb].clamp_min(1e-12).log().mean()
    brier = (p - F.one_hot(yb, 6).to(torch.float64)).square().sum(1).mean()
    a = np.vstack((flat_gradient(nll, params), flat_gradient(brier, params)))

    def evaluate():
        with torch.no_grad():
            pp = corrected_trial_wires(model, xb, geometry)[1]['probabilities']
            return {'nll': float(-pp[rows, yb].clamp_min(1e-12).log().mean()),
                    'brier': float((pp - F.one_hot(yb, 6).to(torch.float64)).square().sum(1).mean())}
    return a, evaluate


def probe_wire(model, x, geometry):
    import torch
    with torch.no_grad():
        w = corrected_trial_wires(model, x, geometry)[1]
        return {k: w[k].detach().cpu().numpy().copy() for k in ('representation', 'centered_logits', 'probabilities', 'decision')}


def wire_change(a, b):
    import numpy as np
    out = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in ('representation', 'centered_logits', 'probabilities')}
    out['decision_changes'] = int((a['decision'] != b['decision']).sum())
    return out


def changed(change, tolerance):
    return any(change[k] > tolerance[k] for k in ('representation', 'centered_logits', 'probabilities')) or change['decision_changes'] > 0


def normalization(model, critics, task, x, y, sex, guard_pool_fn, source_seed, private_dir):
    import numpy as np
    import torch
    from .strict import flat_gradient
    params = _params(model)
    rng = random.Random(_seed(source_seed, 'normalization'))
    guard_pool = guard_pool_fn()
    records, tensors = [], []
    for draw in range(RECIPE['normalization_draws']):
        rows = _draw(rng, len(x))
        terms = critics.terms(model, x[rows], sex[rows])
        g_task, _, _ = task.gradient(model, x[rows], y[rows], guard_pool, critics.geometry)
        g_sens = flat_gradient(terms['local1'] + terms['pair'], params)
        records.append({'draw': draw, 'task_norm': float(np.linalg.norm(g_task)), 'sensitive_norm': float(np.linalg.norm(g_sens)),
                        'rows_sha256': canonical_sha(rows)})
        tensors.append({'rows': rows, 'task': torch.from_numpy(g_task), 'sensitive': torch.from_numpy(g_sens)})
    t = float(np.median([r['task_norm'] for r in records])); s = float(np.median([r['sensitive_norm'] for r in records]))
    coefficient = RECIPE['normalization_target_ratio'] * t / s if s > 0 else None
    path = private_dir / 'NORMALIZATION_TENSORS.pt'
    torch.save(tensors, path); path.chmod(0o600)
    valid = (math.isfinite(t) and math.isfinite(s) and min(t, s) > RECIPE['normalization_min_median_norm']
             and coefficient is not None and math.isfinite(coefficient) and coefficient <= RECIPE['maximum_coefficient'])
    receipt = {'task_median_norm': t, 'sensitive_median_norm': s, 'coefficient': coefficient, 'draws': records,
               'task_gradient': RECIPE['normalization_task_gradient'], 'tensor_sha256': _sha(path),
               'status': 'FIXED' if valid else 'UNAVAILABLE_FIXED_NORMALIZATION'}
    (private_dir / 'NORMALIZATION.json').write_text(json.dumps(receipt, indent=2, sort_keys=True))
    if not valid:
        raise ValueError('fixed normalization unavailable; recorded dependency failure, no coefficient tuning')
    return coefficient, receipt


def save_checkpoint(model, source_payload, critics, arm_dir, epoch, metadata):
    import torch
    payload = {k: copy.deepcopy(source_payload[k]) for k in ('schema', 'feature_names', 'class_orders', 'source_bindings')}
    payload.update(model={k: v.detach().clone() for k, v in model.state_dict().items()},
                   family='SEPARATE_POOL_UTILITY_GUARD', seed=metadata['source_seed'], epoch=epoch,
                   wire_binding=copy.deepcopy(critics.geometry.binding), wire_contract=copy.deepcopy(RECIPE['wire_contract']),
                   wire_variant=metadata['arm'], recipe=copy.deepcopy(RECIPE), coefficient=metadata['coefficient'],
                   objective_state=critics.state_dict(), trainer_metadata=dict(metadata, epoch=epoch),
                   optimizer={'type': 'SGD', 'momentum': 0.0, 'learning_rate': 0.01},
                   selection_history='NONE_VALIDATION_NOT_LOADED_IN_TRAINING')
    path = arm_dir / f'epoch_{epoch}.pt'
    with path.open('xb') as stream:
        torch.save(payload, stream)
    path.chmod(0o600)
    return {'epoch': epoch, 'checkpoint_sha256': _sha(path), 'path_name': path.name}


def run_arm(arm, ctx):
    """One matched arm; returns summary. ctx carries shared frozen inputs."""
    import numpy as np
    import torch
    from .guard import pool_pass, Anchor, global_step
    from .strict import strict_finite_step, flat_gradient, parameter_vector, _assign
    seed = ctx['seed']; geometry = ctx['geometry']; task = ctx['task']
    xU, yU, sU, xG, yG = ctx['xU'], ctx['yU'], ctx['sU'], ctx['xG'], ctx['yG']
    model = copy.deepcopy(ctx['source_model']); model.eval(); frozen = _freeze(model, geometry)
    params = _params(model)
    critics = OccupationCritics(seed, sU, geometry, frozen)
    critics.refresh(model, xU[ctx['reference_rows']])
    warm_rng = random.Random(_seed(seed, 'warm'))
    for _ in range(RECIPE['warm_updates_per_critic']):
        rows = _draw(warm_rng, len(xU)); critics.update(model, xU[rows], sU[rows])
    warm_sha = critics.state_sha()
    arm_dir = ctx['out'] / arm; arm_dir.mkdir(mode=0o700)
    pool = GLOBAL_GUARD_POOL.get(arm)
    px, py = (xG, yG) if pool == 'GUARD' else (xU, yU)
    anchor = Anchor(ctx['anchors'][pool]) if pool else None
    meta = dict(ctx['common'], arm=arm, guard_pool=pool, anchor=None if anchor is None else anchor.record(),
                warm_critic_state_sha256=warm_sha)
    checkpoints = [save_checkpoint(model, ctx['source_payload'], critics, arm_dir, 0, meta)]
    probe0 = probe_wire(model, ctx['probe_x'], geometry)
    state_trace = [parameter_vector(params)]
    batch_rng = random.Random(_seed(seed, 'model_and_critic_draws'))
    statuses = {}; steps = 0; guard_passes = {'shared_task_GUARD_full_passes': 0, 'global_guard_gradient_passes': 0,
                                               'global_guard_metric_evaluations': 0, 'counterfactual_guard_evaluations': 0,
                                               'counterfactual_gradient_passes': 0}
    scans = []; diagnostics = []; epoch_rows = []
    P = ctx['common']['proposals_per_epoch']
    t0 = time.monotonic()
    log = (arm_dir / 'UPDATE_LOG.jsonl').open('x'); os.chmod(arm_dir / 'UPDATE_LOG.jsonl', 0o600)
    for epoch in range(1, RECIPE['epochs'] + 1):
        _guard()
        ref = critics.refresh(model, xU[ctx['reference_rows']])
        counts = {'accepted': 0, 'shortened': 0, 'rejected': 0, 'zero_displacement': 0, 'infeasible_state': 0}
        for step_in_epoch in range(P):
            for _ in range(RECIPE['critic_updates_per_model_step']):
                rows = _draw(batch_rng, len(xU)); critics.update(model, xU[rows], sU[rows])
            rows = _draw(batch_rng, len(xU))
            terms = critics.terms(model, xU[rows], sU[rows])
            if pool == 'GUARD':
                gpool = pool_pass(model, xG, yG, geometry, params, want=('ce', 'nll', 'brier'))
            else:
                gpool = pool_pass(model, xG, yG, geometry, params, want=('ce',))
            guard_passes['shared_task_GUARD_full_passes'] += 1
            if gpool['rows'] != len(xG):
                raise AssertionError('shared task requires the whole GUARD pool')
            g_task, ce_mb, _ = task.gradient(model, xU[rows], yU[rows], gpool, geometry)
            g_sens = flat_gradient(terms['local1'] + terms['pair'], params)
            applied = np.zeros_like(g_sens) if arm == 'TASK_ONLY' else g_sens
            proposal = -RECIPE['learning_rate'] * (g_task + ctx['coefficient'] * applied)
            before = parameter_vector(params)
            current = None
            if pool:
                current = gpool if pool == 'GUARD' else pool_pass(model, px, py, geometry, params, want=('nll', 'brier'))
                guard_passes['global_guard_gradient_passes'] += 1
                if current['rows'] != len(px):
                    raise AssertionError('sampled guard cannot replace the whole fixed pool')

                def evaluate():
                    guard_passes['global_guard_metric_evaluations'] += 1
                    return pool_pass(model, px, py, geometry)
                result = global_step(params, proposal, current, anchor, evaluate, RECIPE['maximum_actual_step_norm'])
                actual = result.pop('actual')
                if result['status'] == 'INFEASIBLE_CURRENT_STATE':
                    _assign(params, before)
            else:
                yb = yU[rows]
                a, evaluate_mb = minibatch_guards(model, xU[rows], yb, geometry)
                result = strict_finite_step(params, proposal, a, evaluate_mb, RECIPE['maximum_actual_step_norm'],
                                            guarded=(arm == 'OLD_BATCH_GUARD'))
                actual = parameter_vector(params) - before
            frozen.verify()
            after = parameter_vector(params)
            if result['status'] != 'ACCEPTED' and not np.array_equal(after, before):
                raise AssertionError('rejected proposal did not restore exact weights')
            accepted = result['status'] == 'ACCEPTED'
            # Fixed counterfactual scan: SEPARATE_GUARD, first eight accepted nonzero updates.
            scan = None
            if arm == 'SEPARATE_GUARD' and accepted and len(scans) < 8:
                scan = counterfactual(model, params, before, after, proposal, g_task, gpool, anchor, ctx, guard_passes,
                                      terms, rows, critics)
                scans.append(dict(scan, update_number=steps + 1, epoch=epoch))
            steps += 1; statuses[result['status']] = statuses.get(result['status'], 0) + 1
            counts['accepted'] += int(accepted)
            counts['shortened'] += int(accepted and (result.get('accepted_fraction') or 1) < 1)
            counts['rejected'] += int(result['status'] == 'REJECTED_ZERO_STEP')
            counts['zero_displacement'] += int(not np.any(actual))
            counts['infeasible_state'] += int(result['status'] == 'INFEASIBLE_CURRENT_STATE')
            state_trace.append(after)
            record = {'epoch': epoch, 'step_in_epoch': step_in_epoch, 'update_number': steps, 'status': result['status'],
                      'accepted_fraction': result.get('accepted_fraction'),
                      'proposed_norm': float(np.linalg.norm(proposal)),
                      'projected_norm': None if result.get('projection') is None else result['projection']['displacement_norm'],
                      'actual_norm': float(np.linalg.norm(actual)),
                      'task_gradient_norm': float(np.linalg.norm(g_task)), 'sensitive_gradient_norm': float(np.linalg.norm(g_sens)),
                      'applied_sensitive_norm': float(np.linalg.norm(applied)),
                      'task_sensitive_cosine': float(g_task @ g_sens / max(1e-300, np.linalg.norm(g_task) * np.linalg.norm(g_sens))),
                      'critic_terms': {'local1': float(terms['local1'].detach()), 'pair': float(terms['pair'].detach())},
                      'critic_ces': terms['critic_ces'], 'minibatch_ce': ce_mb, 'guard_pool_ce': gpool['ce'],
                      'rows_sha256': canonical_sha(rows), 'critic_optimizer_steps': critics.optimizer_steps,
                      'elapsed_s': time.monotonic() - t0, 'result': result}
            log.write(json.dumps(record, sort_keys=True, allow_nan=False, default=float) + '\n'); log.flush()
            if pool:
                ctx['budget_rows'].append({'seed': seed, 'arm': arm, 'pool': pool, 'epoch': epoch, 'update': steps,
                                           'status': result['status'], 'fraction': result.get('accepted_fraction'),
                                           'actual_norm': float(np.linalg.norm(actual)),
                                           'slack_nll': result['raw_slacks'][0], 'slack_brier': result['raw_slacks'][1],
                                           'current_nll': current['nll'], 'current_brier': current['brier'],
                                           'current_correct': current['correct'],
                                           'post_nll': (result.get('post_metrics') or {}).get('nll'),
                                           'post_brier': (result.get('post_metrics') or {}).get('brier'),
                                           'post_correct': (result.get('post_metrics') or {}).get('correct'),
                                           'nll_limit': anchor.nll_limit, 'brier_limit': anchor.brier_limit,
                                           'correct_floor': anchor.correct_floor,
                                           'tiny_negative_slack_resolved': any(result.get('tiny_negative_slack_resolved', []))})
        epoch_rows.append(dict(epoch=epoch, proposals=P, reference_scale_floor_counts=[r['floor_count'] for r in ref], **counts))
        if epoch in RECIPE['checkpoint_epochs']:
            frozen.verify()
            checkpoints.append(save_checkpoint(model, ctx['source_payload'], critics, arm_dir, epoch, dict(meta, model_updates=steps)))
            wire = probe_wire(model, ctx['probe_x'], geometry)
            ch = wire_change(wire, probe0)
            with torch.no_grad():
                gm = pool_pass(model, xG, yG, geometry); um = pool_pass(model, xU, yU, geometry)
            diagnostics.append({'epoch': epoch, 'probe_wire_change_vs_epoch0': ch,
                                'probe_wire_changed_beyond_tolerance': changed(ch, ctx['wire_tolerance']),
                                'fitting_pool_diagnostics': {'GUARD': {k: gm[k] for k in ('nll', 'brier', 'accuracy', 'correct')},
                                                             'UPDATE': {k: um[k] for k in ('nll', 'brier', 'accuracy', 'correct')}},
                                'parameter_displacement_from_start': float(np.linalg.norm(state_trace[-1] - state_trace[0]))})
    log.close()
    expected = (RECIPE['warm_updates_per_critic'] + RECIPE['critic_updates_per_model_step'] * steps) * 6
    if critics.optimizer_steps != expected:
        raise AssertionError('matched critic budget differs')
    trace_path = arm_dir / 'STATE_TRACE.npy'
    with trace_path.open('xb') as stream:
        np.save(stream, np.stack(state_trace))
    trace_path.chmod(0o600)
    if scans:
        with (arm_dir / 'COUNTERFACTUAL_SCAN.json').open('x') as stream:
            json.dump(scans, stream, sort_keys=True, default=float)
    summary = dict(meta, model_updates=steps, critic_optimizer_steps=critics.optimizer_steps, warm_updates_per_critic=80,
                   update_status_counts=statuses, epoch_accounting=epoch_rows, checkpoints=checkpoints,
                   label_free_and_fitting_pool_diagnostics=diagnostics, guard_pass_accounting=guard_passes,
                   counterfactual_scan=scans, update_log_sha256=_sha(arm_dir / 'UPDATE_LOG.jsonl'),
                   state_trace_sha256=_sha(trace_path), runtime_s=time.monotonic() - t0)
    (arm_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


def counterfactual(model, params, before, after, proposal, g_task, gpool, anchor, ctx, guard_passes, terms, rows, critics):
    """Same frozen state, identical task gradient/pool/budgets/solver/ladder, no privacy term."""
    import numpy as np
    from .guard import pool_pass, global_step
    from .strict import parameter_vector, _assign, flat_gradient
    geometry = ctx['geometry']; xG, yG = ctx['xG'], ctx['yG']
    actual_priv = after - before
    post_wire = probe_wire(model, ctx['probe_x'], geometry)
    _assign(params, before)
    pre_wire = probe_wire(model, ctx['probe_x'], geometry)
    # Repeat-computation noise of the privacy proposal at the frozen state.
    g_sens2 = flat_gradient(critics.terms(model, ctx['xU'][rows], ctx['sU'][rows])['local1']
                            + critics.terms(model, ctx['xU'][rows], ctx['sU'][rows])['pair'], params)
    gpool2 = pool_pass(model, xG, yG, geometry, params, want=('ce', 'nll', 'brier'))
    guard_passes['counterfactual_gradient_passes'] += 1
    g_task2, _, _ = ctx['task'].gradient(model, ctx['xU'][rows], ctx['yU'][rows], gpool2, geometry)
    proposal2 = -RECIPE['learning_rate'] * (g_task2 + ctx['coefficient'] * g_sens2)
    repeat_noise = float(np.max(np.abs(proposal2 - proposal)))
    cf_proposal = -RECIPE['learning_rate'] * g_task

    def evaluate():
        guard_passes['counterfactual_guard_evaluations'] += 1
        return pool_pass(model, xG, yG, geometry)
    cf = global_step(params, cf_proposal, gpool, anchor, evaluate, RECIPE['maximum_actual_step_norm'])
    actual_cf = cf.pop('actual')
    _assign(params, after)
    if not np.array_equal(parameter_vector(params), after):
        raise AssertionError('counterfactual scan did not restore recorded state')
    tol = max(1e-10, 10 * repeat_noise)
    diff = actual_priv - actual_cf
    wire_ch = wire_change(post_wire, pre_wire)
    return {'privacy_actual_norm': float(np.linalg.norm(actual_priv)), 'counterfactual_status': cf['status'],
            'counterfactual_actual_norm': float(np.linalg.norm(actual_cf)),
            'privacy_minus_counterfactual_norm': float(np.linalg.norm(diff)),
            'privacy_minus_counterfactual_max_abs': float(np.max(np.abs(diff))),
            'repeat_computation_proposal_max_abs_noise': repeat_noise, 'contribution_tolerance': tol,
            'privacy_contribution_beyond_noise': bool(np.max(np.abs(diff)) > tol),
            'cosine_privacy_vs_counterfactual': float(actual_priv @ actual_cf / max(1e-300, np.linalg.norm(actual_priv) * np.linalg.norm(actual_cf))),
            'update_probe_wire_change': wire_ch, 'update_changed_probe_wire': changed(wire_ch, ctx['wire_tolerance']),
            'counterfactual_fraction': cf.get('accepted_fraction'), 'counterfactual_post_metrics': cf.get('post_metrics')}


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, queue_guard = queue_config(spec_path, 'protection')
    for key in ('geometry', 'checkpoint', 'role_split'):
        verify(spec[key]['path'], spec[key]['sha256'])
    verify(spec['science_lock'], spec['science_lock_sha256'])
    if spec.get('operation') != 'separate_pool_utility_guard_trial' or spec.get('seed') not in (0, 1):
        raise PermissionError('unregistered trainer operation or seed')
    if any('VALIDATION' in Path(s['path']).as_posix().upper().split('/') for s in
           json.loads((Path(os.environ['PCRL_QUEUE_ROOT']) / 'state.json').read_text())['units'][active['unit_id']]['spec']['sources']):
        raise PermissionError('VALIDATION is not a trainer source')
    validate_recipe(spec['recipe'])
    lock = json.loads(Path(spec['science_lock']).read_text())
    split = json.loads(Path(spec['role_split']['path']).read_text())
    validate_science_lock(lock, seed=spec['seed'], checkpoint_sha256=spec['checkpoint']['sha256'],
                          geometry_sha256=spec['geometry_binding_sha256'], role_split_sha256=spec['role_split']['sha256'])
    queue_guard()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .guard import pool_pass
    from .roles_worker import probe_key
    source_model, source_payload = load_checkpoint(verify(spec['checkpoint']['path'], spec['checkpoint']['sha256']))
    if (source_payload.get('family') != 'RAW_JOINT' or source_payload.get('strength') != 0.3
            or source_payload.get('epoch') != 5 or source_payload.get('seed') != spec['seed']):
        raise PermissionError('starting package must be admitted RAW strength .3 epoch 5')
    geometry = prepare_geometry(json.loads(verify(spec['geometry']['path'], spec['geometry']['sha256']).read_text()),
                                expected_sha256=spec['geometry_binding_sha256'], source_seed=spec['seed'],
                                checkpoint_sha256=spec['checkpoint']['sha256'])
    roles = load_roles(spec, verify)
    for role in ('UPDATE', 'GUARD'):
        if tuple(map(str, roles[role]['feature_names'])) != tuple(source_model.names):
            raise PermissionError('feature order differs')
    xU = torch.from_numpy(roles['UPDATE']['X']); yU = torch.from_numpy(roles['UPDATE']['occupation']).long()
    sU = torch.from_numpy(roles['UPDATE']['sex']).long()
    xG = torch.from_numpy(roles['GUARD']['X']); yG = torch.from_numpy(roles['GUARD']['occupation']).long()
    probe_pos = sorted(range(len(xU)), key=lambda i: (probe_key(roles['UPDATE']['row_id'][i]), i))[:512]
    if hashlib.sha256(json.dumps(probe_pos).encode()).hexdigest() != split['probe_position_sha256']:
        raise PermissionError('frozen probe differs')
    out = Path(spec['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private trainer directory required')
    out.mkdir(parents=True, mode=0o700)
    P = math.ceil(len(xU) / RECIPE['batch_size'])
    if P != lock['proposals_per_epoch']:
        raise PermissionError('proposal count differs from locked split value')
    task = Task(len(xU), len(xG))
    frozen0 = _freeze(source_model, geometry)
    anchors = {'GUARD': pool_pass(source_model, xG, yG, geometry), 'UPDATE': pool_pass(source_model, xU, yU, geometry)}
    reference_rows = _draw(random.Random(_seed(spec['seed'], 'reference')), len(xU))
    common = {'source_seed': spec['seed'], 'starting_checkpoint_sha256': spec['checkpoint']['sha256'],
              'science_lock_sha256': spec['science_lock_sha256'], 'recipe_sha256': canonical_sha(RECIPE),
              'role_split_sha256': spec['role_split']['sha256'], 'update_rows': len(xU), 'guard_rows': len(xG),
              'task_weights': {'update': task.w_update, 'guard': task.w_guard}, 'proposals_per_epoch': P,
              'VALIDATION_loaded': False, 'INNER_loaded': False, 'ATT_loaded': False,
              'geometry_binding_sha256': geometry.binding_sha256,
              'income_wire_alias': {'income_tensor_sha256': geometry.binding['head_tensor_sha256'][0],
                                    'geometry_binding_sha256': geometry.binding_sha256}}
    # Coefficient once per seed at the matched warm initial state.
    norm_model = copy.deepcopy(source_model); nf = _freeze(norm_model, geometry)
    norm_critics = OccupationCritics(spec['seed'], sU, geometry, nf)
    norm_critics.refresh(norm_model, xU[reference_rows])
    warm_rng = random.Random(_seed(spec['seed'], 'warm'))
    for _ in range(RECIPE['warm_updates_per_critic']):
        rows = _draw(warm_rng, len(xU)); norm_critics.update(norm_model, xU[rows], sU[rows])
    coefficient, receipt = normalization(norm_model, norm_critics, task, xU, yU, sU,
                                         lambda: pool_pass(norm_model, xG, yG, geometry, _params(norm_model), want=('ce',)),
                                         spec['seed'], out)
    ctx = dict(seed=spec['seed'], geometry=geometry, task=task, xU=xU, yU=yU, sU=sU, xG=xG, yG=yG,
               source_model=source_model, source_payload=source_payload, reference_rows=reference_rows,
               out=out, coefficient=coefficient, probe_x=xU[probe_pos], wire_tolerance=lock['wire_tolerance'],
               anchors=anchors, budget_rows=[],
               common=dict(common, coefficient=coefficient, normalization_receipt_sha256=canonical_sha(receipt),
                           normalization_warm_critic_state_sha256=norm_critics.state_sha()))
    summaries = {}
    for arm in ARMS:
        queue_guard()
        summaries[arm] = run_arm(arm, ctx)
        if summaries[arm]['warm_critic_state_sha256'] != ctx['common']['normalization_warm_critic_state_sha256']:
            raise AssertionError('arms do not share the matched warm critic state')
    frozen0.verify()
    (out / 'GUARD_BUDGET_TRACE.json').write_text(json.dumps(ctx['budget_rows'], sort_keys=True, default=float))
    result = {'schema': 'SPG1-separate-pool-training-v1', 'queue_unit_id': active['unit_id'], 'seed': spec['seed'],
              'recipe_sha256': canonical_sha(RECIPE), 'coefficient': coefficient, 'normalization': receipt,
              'anchors': {k: {kk: v[kk] for kk in ('nll', 'brier', 'correct', 'rows', 'accuracy')} for k, v in anchors.items()},
              'arms': summaries, 'VALIDATION_loaded': False, 'INNER_loaded': False,
              'scientific_status': 'TRAINING_ARTIFACTS_ONLY_NO_VALIDATION_OR_PRIVACY_CLAIM',
              'budget_trace_sha256': _sha(out / 'GUARD_BUDGET_TRACE.json')}
    (out / 'SUMMARY.json').write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
    Path(spec['output']).parent.mkdir(parents=True, exist_ok=True)
    with Path(spec['output']).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, default=float)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
