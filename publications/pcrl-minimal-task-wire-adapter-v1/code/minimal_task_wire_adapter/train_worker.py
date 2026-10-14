"""ROOT-authorized, queue-only five-arm residual-adapter bank for one source seed.

Receives only the physical UPDATE and GUARD exports. FIT fold 2, ATT and INNER
are never sources. Checkpoints 0/1/3/5/10 and label-free movement diagnostics
are saved for later workers; no held-out metric exists here.
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

from .recipe import ARMS, ARM_RULES, RECIPE, validate_recipe, validate_science_lock
from .adapter import canonical_sha, tensor_sha, Adapter


def _seed(source_seed, purpose):
    return int(hashlib.sha256(f'MTA1-adapter|{source_seed}|{purpose}'.encode()).hexdigest()[:8], 16)


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
    import numpy as np
    from zipfile import ZipFile
    parts = Path(item['path']).as_posix().upper().split('/')
    if any(x in parts for x in ('VALIDATION', 'ATTACK_FIT_NEW', 'INNER_SELECTION')):
        raise PermissionError('trainer cannot load FIT2, ATT or INNER')
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as archive:
        names = {x.filename for x in archive.infolist()}
    if names != {f + '.npy' for f in fields}:
        raise PermissionError('unexpected role export fields')
    with np.load(p, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in fields}


def load_roles(spec, verify):
    import numpy as np
    want = {'UPDATE': ('features', 'occupation', 'sex'), 'GUARD': ('features', 'occupation')}
    if set(spec['roles']) != set(want):
        raise PermissionError('trainer receives exactly UPDATE and GUARD exports')
    roles = {}
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


class Critics:
    """Six actual-wire critics (occupation29, pair50) on UPDATE rows/SEX, every arm."""
    def __init__(self, source_seed, sex):
        import torch
        prior = torch.bincount(sex, minlength=2).double() / len(sex)
        if (prior <= 0).any():
            raise ValueError('both sensitive groups required in UPDATE')
        self.prior = prior; self.entropy = float(-(prior * prior.log()).sum())
        self.banks, self.optimizers, self.references = [], [], None
        self.optimizer_steps = 0
        for view, width in enumerate((29, 50)):
            bank, optimizers = [], []
            for kind in RECIPE['critic_kinds']:
                torch.manual_seed(_seed(source_seed, f'critic|{view}|{kind}'))
                if kind == 'linear':
                    critic = torch.nn.Linear(width, 2, dtype=torch.float64)
                elif kind == 'mlp32':
                    critic = torch.nn.Sequential(torch.nn.Linear(width, 32, dtype=torch.float64), torch.nn.ReLU(),
                                                 torch.nn.Linear(32, 2, dtype=torch.float64))
                else:
                    critic = torch.nn.Sequential(torch.nn.Linear(width, 64, dtype=torch.float64), torch.nn.ReLU(),
                                                 torch.nn.Linear(64, 64, dtype=torch.float64), torch.nn.ReLU(),
                                                 torch.nn.Linear(64, 2, dtype=torch.float64))
                bank.append(critic)
                optimizers.append(torch.optim.Adam(critic.parameters(), lr=RECIPE['critic_learning_rate']))
            self.banks.append(bank); self.optimizers.append(optimizers)

    @staticmethod
    def raw_views(occ_full, income_full):
        import torch
        return [occ_full, torch.cat((income_full, occ_full), 1)]

    def refresh(self, occ_full, income_full):
        import torch
        from task_conditioned_representation.kernels import DiagonalReference
        with torch.no_grad():
            values = self.raw_views(occ_full, income_full)
            prior = self.references or [None, None]
            self.references = [DiagonalReference.fit(v, old) for v, old in zip(values, prior)]
        for r in self.references:
            if not torch.isfinite(r.mean).all() or not torch.isfinite(r.scale).all() or (r.scale <= 0).any():
                raise ValueError('finite positive full-coordinate reference required')
        return [{'floor_count': r.floor_count, 'scale_min': float(r.scale.min()), 'scale_max': float(r.scale.max())}
                for r in self.references]

    def views(self, occ_full, income_full):
        return [r.normalize(v) for r, v in zip(self.references, self.raw_views(occ_full, income_full))]

    def update(self, occ_full, income_full, sex):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            views = self.views(occ_full.detach(), income_full)
        for bank, optimizers, view in zip(self.banks, self.optimizers, views):
            for critic, optimizer in zip(bank, optimizers):
                for p in critic.parameters():
                    p.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(critic(view), sex)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite critic loss')
                loss.backward(); optimizer.step(); self.optimizer_steps += 1

    def terms(self, occ_full, income_full, sex):
        import torch
        import torch.nn.functional as F
        views = self.views(occ_full, income_full)
        terms, ces = [], []
        for bank, view in zip(self.banks, views):
            for critic in bank:
                for p in critic.parameters():
                    p.requires_grad_(False)
            losses = torch.stack([F.cross_entropy(c(view), sex) for c in bank] + [view.sum() * 0 + self.entropy])
            terms.append(1 - losses.min() / self.entropy); ces.append(losses.detach().tolist())
        return {'local': terms[0], 'pair': terms[1], 'critic_ces': ces}

    def state_sha(self):
        return canonical_sha([[{k: tensor_sha(v) for k, v in c.state_dict().items()} for c in bank] for bank in self.banks])

    def state_dict(self):
        return {'critics': [[copy.deepcopy(c.state_dict()) for c in bank] for bank in self.banks],
                'references': [None if r is None else r.snapshot() for r in (self.references or [])],
                'prior': self.prior.clone(), 'entropy': self.entropy, 'critic_optimizer_steps': self.optimizer_steps}


def privacy_objective(terms, kind):
    if kind == 'joint':
        return terms['local'] + terms['pair']
    if kind == 'local':
        return terms['local']
    raise ValueError('unregistered privacy objective')


class Task:
    """Shared population-weighted task objective, identical in every arm."""
    def __init__(self, n_update, n_guard):
        self.n_update, self.n_guard = n_update, n_guard
        self.w_update = n_update / (n_update + n_guard); self.w_guard = n_guard / (n_update + n_guard)

    def gradient(self, adapter, hb, q0b, yb, guard_ce_gradient):
        import torch
        import torch.nn.functional as F
        from .guard import flat
        R = adapter.R.detach().clone().requires_grad_(True); b = adapter.b.detach().clone().requires_grad_(True)
        logits = adapter.wire(hb, q0b, R, b)['centered_logits']
        ce = -F.log_softmax(logits, 1)[torch.arange(len(yb)), yb].mean()
        g_mb = flat(torch.autograd.grad(ce, (R, b)))
        return self.w_update * g_mb + self.w_guard * guard_ce_gradient, float(ce.detach())


def privacy_gradient(adapter, critics, hb, q0b, incb, sb, kind):
    """Gradient of the arm's privacy objective w.r.t. (R, b_res) through the actual wire."""
    import torch
    from .guard import flat
    R = adapter.R.detach().clone().requires_grad_(True); b = adapter.b.detach().clone().requires_grad_(True)
    occ = adapter.wire(hb, q0b, R, b)['full']
    terms = critics.terms(occ, incb, sb)
    if kind is None:
        return None, terms
    return flat(torch.autograd.grad(privacy_objective(terms, kind), (R, b))), terms


def _draw(rng, n):
    return rng.sample(range(n), RECIPE['batch_size'])


def probe_wire(adapter, hb, q0b):
    import torch
    with torch.no_grad():
        w = adapter.wire(hb, q0b)
        return {k: w[k].detach().numpy().copy() for k in ('representation', 'centered_logits', 'probabilities', 'decision')}


def wire_change(a, b):
    import numpy as np
    out = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in ('representation', 'centered_logits', 'probabilities')}
    out['decision_changes'] = int((a['decision'] != b['decision']).sum())
    return out


def changed(change, tolerance):
    return any(change[k] > tolerance[k] for k in ('representation', 'centered_logits', 'probabilities')) or change['decision_changes'] > 0


def warm_critics(ctx, adapter):
    import torch
    critics = Critics(ctx['seed'], ctx['sU'])
    ref = ctx['reference_rows']
    with torch.no_grad():
        critics.refresh(adapter.wire(ctx['hU'][ref], ctx['qU'][ref])['full'], ctx['iU'][ref])
    rng = random.Random(_seed(ctx['seed'], 'warm'))
    for _ in range(RECIPE['warm_updates_per_critic']):
        rows = _draw(rng, len(ctx['yU']))
        with torch.no_grad():
            occ = adapter.wire(ctx['hU'][rows], ctx['qU'][rows])['full']
        critics.update(occ, ctx['iU'][rows], ctx['sU'][rows])
    return critics


def normalization(ctx, private_dir):
    """lambda_J and lambda_L at the matched warm zero-residual state; no search."""
    import numpy as np
    import torch
    from .guard import pool_pass
    adapter = Adapter(ctx['source_model'], ctx['geometry'], ctx['scale'])
    critics = warm_critics(ctx, adapter)
    gpool = pool_pass(adapter, ctx['hG'], ctx['qG'], ctx['yG'], params=True, want=('ce',))
    rng = random.Random(_seed(ctx['seed'], 'normalization'))
    records, tensors = [], []
    for draw in range(RECIPE['normalization']['draws']):
        rows = _draw(rng, len(ctx['yU']))
        g_task, _ = ctx['task'].gradient(adapter, ctx['hU'][rows], ctx['qU'][rows], ctx['yU'][rows], gpool['gradients']['ce'])
        gj, _ = privacy_gradient(adapter, critics, ctx['hU'][rows], ctx['qU'][rows], ctx['iU'][rows], ctx['sU'][rows], 'joint')
        gl, _ = privacy_gradient(adapter, critics, ctx['hU'][rows], ctx['qU'][rows], ctx['iU'][rows], ctx['sU'][rows], 'local')
        records.append({'draw': draw, 'task_norm': float(np.linalg.norm(g_task)), 'joint_norm': float(np.linalg.norm(gj)),
                        'local_norm': float(np.linalg.norm(gl)), 'rows_sha256': canonical_sha(rows)})
        tensors.append({'rows': rows, 'task': torch.from_numpy(g_task), 'joint': torch.from_numpy(gj), 'local': torch.from_numpy(gl)})
    t = float(np.median([r['task_norm'] for r in records]))
    out = {'task_median_norm': t, 'draws': records, 'warm_critic_state_sha256': critics.state_sha(), 'coefficients': {}}
    for kind in ('joint', 'local'):
        s = float(np.median([r[f'{kind}_norm'] for r in records]))
        coef = RECIPE['normalization']['target_ratio'] * t / s if s > 0 else None
        valid = (math.isfinite(t) and math.isfinite(s) and min(t, s) > RECIPE['normalization_min_median_norm']
                 and coef is not None and math.isfinite(coef) and coef <= RECIPE['maximum_coefficient'])
        out['coefficients'][kind] = {'sensitive_median_norm': s, 'coefficient': coef if valid else None,
                                     'raw_coefficient': coef, 'status': 'FIXED' if valid else 'UNAVAILABLE_FIXED_NORMALIZATION'}
    path = private_dir / 'NORMALIZATION_TENSORS.pt'
    torch.save(tensors, path); path.chmod(0o600)
    out['tensor_sha256'] = _sha(path)
    (private_dir / 'NORMALIZATION.json').write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def save_checkpoint(adapter, ctx, arm, arm_dir, epoch, critics, extra):
    import torch
    payload = {'schema': 'MTA1-adapter-checkpoint-v1', 'campaign': RECIPE['campaign'], 'seed': ctx['seed'], 'arm': arm,
               'epoch': epoch, 'R': adapter.R.detach().clone(), 'b_res': adapter.b.detach().clone(),
               'rank': adapter.rank, 'scale': adapter.scale, 'trainable_parameters': adapter.n_parameters(),
               'source_checkpoint_sha256': ctx['checkpoint_sha256'], 'geometry_binding_sha256': ctx['geometry'].binding_sha256,
               'adapter_binding_sha256': ctx['adapter_binding_sha256'], 'occupation_basis': adapter.B.clone(),
               'income_basis': adapter.Bi.clone(), 'adapter_contract': RECIPE['adapter_contract'],
               'recipe_sha256': canonical_sha(RECIPE), 'arm_rule': ARM_RULES[arm],
               'coefficient': ctx['coefficients'].get(arm), 'critic_state': critics.state_dict(),
               'trainer_metadata': dict(extra, epoch=epoch), 'selection_history': 'NONE_FIT2_ATT_INNER_NOT_LOADED'}
    path = arm_dir / f'epoch_{epoch}.pt'
    with path.open('xb') as stream:
        torch.save(payload, stream)
    path.chmod(0o600)
    return {'epoch': epoch, 'checkpoint_sha256': _sha(path), 'path_name': path.name,
            'R_sha256': tensor_sha(adapter.R), 'b_res_sha256': tensor_sha(adapter.b)}


def run_arm(arm, ctx):
    import numpy as np
    import torch
    from .guard import pool_pass, Anchor, guarded_step, flat
    rule = ARM_RULES[arm]
    adapter = Adapter(ctx['source_model'], ctx['geometry'], ctx['scale'])
    critics = warm_critics(ctx, adapter)
    warm_sha = critics.state_sha()
    arm_dir = ctx['out'] / arm; arm_dir.mkdir(mode=0o700)
    anchor = Anchor(ctx['start_pool'], trust=rule['trust'])
    coef = ctx['coefficients'].get(arm)
    meta = dict(ctx['common'], arm=arm, arm_rule=rule, anchor=anchor.record(), coefficient=coef, warm_critic_state_sha256=warm_sha)
    checkpoints = [save_checkpoint(adapter, ctx, arm, arm_dir, 0, critics, dict(meta, model_updates=0))]
    probe0 = probe_wire(adapter, ctx['probe_h'], ctx['probe_q'])
    trace = [flat([adapter.R, adapter.b])]
    batch_rng = random.Random(_seed(ctx['seed'], 'model_and_critic_draws'))
    statuses = {}; steps = 0; scans = []; scan_states = []; diagnostics = []; epoch_rows = []; budget_rows = []
    P = RECIPE['proposals_per_epoch']; t0 = time.monotonic()
    log = (arm_dir / 'UPDATE_LOG.jsonl').open('x'); os.chmod(arm_dir / 'UPDATE_LOG.jsonl', 0o600)
    hU, qU, iU, yU, sU = ctx['hU'], ctx['qU'], ctx['iU'], ctx['yU'], ctx['sU']
    ref = ctx['reference_rows']
    for epoch in range(1, RECIPE['epochs'] + 1):
        _guard()
        with torch.no_grad():
            refinfo = critics.refresh(adapter.wire(hU[ref], qU[ref])['full'], iU[ref])
        counts = {'accepted': 0, 'shortened': 0, 'rejected': 0, 'zero_displacement': 0, 'infeasible_state': 0}
        for step_in_epoch in range(P):
            for _ in range(RECIPE['critic_updates_per_model_step']):
                rows = _draw(batch_rng, len(yU))
                with torch.no_grad():
                    occ = adapter.wire(hU[rows], qU[rows])['full']
                critics.update(occ, iU[rows], sU[rows])
            rows = _draw(batch_rng, len(yU))
            current = pool_pass(adapter, ctx['hP'], ctx['qP'], ctx['yP'], ctx['lp0P'], params=True, want=('nll', 'brier', 'kl'))
            gpool = pool_pass(adapter, ctx['hG'], ctx['qG'], ctx['yG'], params=True, want=('ce',))
            g_task, ce_mb = ctx['task'].gradient(adapter, hU[rows], qU[rows], yU[rows], gpool['gradients']['ce'])
            g_priv, terms = privacy_gradient(adapter, critics, hU[rows], qU[rows], iU[rows], sU[rows], rule['privacy'])
            applied = np.zeros_like(g_task) if g_priv is None else coef * g_priv
            proposal = -RECIPE['learning_rate'] * (g_task + applied)
            before = adapter.state(); vec0 = flat([before['R'], before['b']])

            def evaluate():
                return pool_pass(adapter, ctx['hP'], ctx['qP'], ctx['yP'], ctx['lp0P'])
            result = guarded_step(adapter, proposal, current, anchor, evaluate, RECIPE['maximum_actual_step_norm'])
            actual = result.pop('actual')
            after = flat([adapter.R, adapter.b])
            if result['status'] != 'ACCEPTED' and not np.array_equal(after, vec0):
                raise AssertionError('rejected proposal did not restore exact residual')
            ctx['income_guard']()
            accepted = result['status'] == 'ACCEPTED'
            scan = None
            if arm == 'J_TRUST' and accepted and len(scans) < 8:
                scan, state = counterfactual(adapter, ctx, before, after, actual, proposal, g_task, g_priv, coef, current,
                                             anchor, rows, critics, gpool)
                scans.append(dict(scan, update_number=steps + 1, epoch=epoch, privacy_accepted_fraction=result.get('accepted_fraction')))
                scan_states.append(dict(state, update_number=steps + 1, epoch=epoch))
            steps += 1; statuses[result['status']] = statuses.get(result['status'], 0) + 1
            counts['accepted'] += int(accepted)
            counts['shortened'] += int(accepted and (result.get('accepted_fraction') or 1) < 1)
            counts['rejected'] += int(result['status'] == 'REJECTED_ZERO_STEP')
            counts['zero_displacement'] += int(not np.any(actual))
            counts['infeasible_state'] += int(result['status'] == 'INFEASIBLE_CURRENT_STATE')
            trace.append(after)
            gp_norm = 0.0 if g_priv is None else float(np.linalg.norm(g_priv))
            record = {'epoch': epoch, 'step_in_epoch': step_in_epoch, 'update_number': steps, 'status': result['status'],
                      'accepted_fraction': result.get('accepted_fraction'), 'proposed_norm': float(np.linalg.norm(proposal)),
                      'projected_norm': None if result.get('projection') is None else result['projection']['displacement_norm'],
                      'actual_norm': float(np.linalg.norm(actual)), 'task_gradient_norm': float(np.linalg.norm(g_task)),
                      'privacy_gradient_norm': gp_norm, 'applied_privacy_norm': float(np.linalg.norm(applied)),
                      'achieved_strength_ratio': float(np.linalg.norm(applied) / max(1e-300, np.linalg.norm(g_task))),
                      'task_privacy_cosine': None if g_priv is None else float(g_task @ g_priv / max(1e-300, np.linalg.norm(g_task) * gp_norm)),
                      'critic_terms': {'local': float(terms['local'].detach()), 'pair': float(terms['pair'].detach())},
                      'critic_ces': terms['critic_ces'], 'minibatch_ce': ce_mb, 'guard_pool_ce': gpool['ce'],
                      'rows_sha256': canonical_sha(rows), 'critic_optimizer_steps': critics.optimizer_steps,
                      'elapsed_s': time.monotonic() - t0, 'result': result}
            log.write(json.dumps(record, sort_keys=True, allow_nan=False, default=float) + '\n'); log.flush()
            post = result.get('post_metrics') or {}
            budget_rows.append({'seed': ctx['seed'], 'arm': arm, 'epoch': epoch, 'update': steps, 'status': result['status'],
                                'fraction': result.get('accepted_fraction'), 'actual_norm': float(np.linalg.norm(actual)),
                                'raw_slacks': result['raw_slacks'], 'current_nll': current['nll'], 'current_brier': current['brier'],
                                'current_correct': current['correct'], 'current_kl': current['kl'],
                                'post_nll': post.get('nll'), 'post_brier': post.get('brier'), 'post_correct': post.get('correct'),
                                'post_kl': post.get('kl'), 'nll_limit': anchor.nll_limit, 'brier_limit': anchor.brier_limit,
                                'correct_floor': anchor.correct_floor, 'kl_limit': anchor.kl_limit if anchor.trust else None})
        epoch_rows.append(dict(epoch=epoch, proposals=P, reference=refinfo, **counts))
        if epoch in RECIPE['checkpoint_epochs']:
            checkpoints.append(save_checkpoint(adapter, ctx, arm, arm_dir, epoch, critics, dict(meta, model_updates=steps)))
            wire = probe_wire(adapter, ctx['probe_h'], ctx['probe_q']); ch = wire_change(wire, probe0)
            pm = pool_pass(adapter, ctx['hP'], ctx['qP'], ctx['yP'], ctx['lp0P'])
            um = pool_pass(adapter, hU, qU, yU, ctx['lp0U']); gm = pool_pass(adapter, ctx['hG'], ctx['qG'], ctx['yG'], ctx['lp0G'])
            diagnostics.append({'epoch': epoch, 'probe_wire_change_vs_epoch0': ch,
                                'probe_wire_changed_beyond_tolerance': changed(ch, ctx['wire_tolerance']),
                                'fitting_pool': {name: {k: m[k] for k in ('nll', 'brier', 'accuracy', 'correct', 'kl', 'ce')}
                                                 for name, m in (('TASK_POOL', pm), ('UPDATE', um), ('GUARD', gm))},
                                'residual_norm': float(np.linalg.norm(trace[-1])),
                                'parameter_displacement_from_start': float(np.linalg.norm(trace[-1] - trace[0]))})
    log.close()
    expected = (RECIPE['warm_updates_per_critic'] + RECIPE['critic_updates_per_model_step'] * steps) * 6
    if critics.optimizer_steps != expected:
        raise AssertionError('matched critic budget differs')
    trace_path = arm_dir / 'STATE_TRACE.npy'
    with trace_path.open('xb') as stream:
        np.save(stream, np.stack(trace))
    trace_path.chmod(0o600)
    extra = {}
    if scan_states:
        p = arm_dir / 'COUNTERFACTUAL_STATES.pt'
        torch.save(scan_states, p); p.chmod(0o600); extra['counterfactual_states_sha256'] = _sha(p)
    bpath = arm_dir / 'BUDGET_TRACE.json'; bpath.write_text(json.dumps(budget_rows, sort_keys=True, default=float))
    summary = dict(meta, model_updates=steps, critic_optimizer_steps=critics.optimizer_steps,
                   update_status_counts=statuses, qp_conservative_zero_fallbacks=statuses.get('QP_FALLBACK_ZERO_STEP', 0),
                   counts_note='zero_displacement overlaps rejected/infeasible/zero-projection; categories are not additive', epoch_accounting=epoch_rows, checkpoints=checkpoints,
                   label_free_and_fitting_pool_diagnostics=diagnostics, counterfactual_scan=scans,
                   update_log_sha256=_sha(arm_dir / 'UPDATE_LOG.jsonl'), state_trace_sha256=_sha(trace_path),
                   budget_trace_sha256=_sha(bpath), runtime_s=time.monotonic() - t0, **extra)
    (arm_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


def counterfactual(adapter, ctx, before, after, actual_priv, proposal, g_task, g_priv, coef, current, anchor, rows, critics, gpool):
    """Same saved state, identical task gradient/pool/budgets/solver/ladder, privacy removed."""
    import numpy as np
    import torch
    from .guard import pool_pass, guarded_step, flat
    post_wire = probe_wire(adapter, ctx['probe_h'], ctx['probe_q'])
    adapter.R, adapter.b = before['R'].clone(), before['b'].clone()
    pre_wire = probe_wire(adapter, ctx['probe_h'], ctx['probe_q'])
    # Repeat-computation noise of the full privacy proposal at the frozen state.
    gpool2 = pool_pass(adapter, ctx['hG'], ctx['qG'], ctx['yG'], params=True, want=('ce',))
    g_task2, _ = ctx['task'].gradient(adapter, ctx['hU'][rows], ctx['qU'][rows], ctx['yU'][rows], gpool2['gradients']['ce'])
    g_priv2, _ = privacy_gradient(adapter, critics, ctx['hU'][rows], ctx['qU'][rows], ctx['iU'][rows], ctx['sU'][rows], 'joint')
    proposal2 = -RECIPE['learning_rate'] * (g_task2 + coef * g_priv2)
    repeat_noise = float(np.max(np.abs(proposal2 - proposal)))
    cf_proposal = -RECIPE['learning_rate'] * g_task

    def evaluate():
        return pool_pass(adapter, ctx['hP'], ctx['qP'], ctx['yP'], ctx['lp0P'])
    cf = guarded_step(adapter, cf_proposal, current, anchor, evaluate, RECIPE['maximum_actual_step_norm'])
    actual_cf = cf.pop('actual')
    cf_wire = probe_wire(adapter, ctx['probe_h'], ctx['probe_q'])
    adapter.R = torch.from_numpy(after[:adapter.R.numel()].reshape(adapter.R.shape).copy())
    adapter.b = torch.from_numpy(after[adapter.R.numel():].copy())
    if not np.array_equal(flat([adapter.R, adapter.b]), after):
        raise AssertionError('counterfactual scan did not restore recorded state')
    tol = max(1e-10, 10 * repeat_noise)
    diff = actual_priv - actual_cf
    priv_vs_cf_wire = wire_change(post_wire, cf_wire)
    scan = {'privacy_actual_norm': float(np.linalg.norm(actual_priv)), 'counterfactual_status': cf['status'],
            'counterfactual_actual_norm': float(np.linalg.norm(actual_cf)),
            'privacy_minus_counterfactual_norm': float(np.linalg.norm(diff)),
            'privacy_minus_counterfactual_max_abs': float(np.max(np.abs(diff))),
            'repeat_computation_proposal_max_abs_noise': repeat_noise, 'contribution_tolerance': tol,
            'privacy_contribution_beyond_noise': bool(np.max(np.abs(diff)) > tol),
            'cosine_privacy_vs_counterfactual': float(actual_priv @ actual_cf / max(1e-300, np.linalg.norm(actual_priv) * np.linalg.norm(actual_cf))),
            'update_probe_wire_change': wire_change(post_wire, pre_wire),
            'privacy_vs_counterfactual_probe_wire_difference': priv_vs_cf_wire,
            'privacy_changes_deployed_wire_beyond_tolerance': changed(priv_vs_cf_wire, ctx['wire_tolerance']),
            'counterfactual_fraction': cf.get('accepted_fraction'), 'counterfactual_post_metrics': cf.get('post_metrics')}
    state = {'pre_R': before['R'].clone(), 'pre_b': before['b'].clone(), 'rows': list(rows), 'coefficient': coef,
             'critic_state': critics.state_dict(), 'g_task': torch.from_numpy(g_task.copy()),
             'g_privacy': torch.from_numpy(g_priv.copy()), 'proposal': torch.from_numpy(proposal.copy()),
             'pool_current': {k: current[k] for k in ('nll', 'brier', 'correct', 'kl', 'rows')},
             'pool_gradients': {k: torch.from_numpy(v.copy()) for k, v in current['gradients'].items()},
             'guard_ce_gradient': torch.from_numpy(gpool['gradients']['ce'].copy()), 'anchor': anchor.record(),
             'actual_privacy_displacement': torch.from_numpy(actual_priv.copy()),
             'actual_counterfactual_displacement': torch.from_numpy(actual_cf.copy()),
             'counterfactual_status': cf['status'], 'counterfactual_fraction': cf.get('accepted_fraction')}
    return scan, state


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, queue_guard = queue_config(spec_path, 'protection')
    for key in ('geometry', 'checkpoint', 'adapter_binding', 'role_split', 'admission'):
        verify(spec[key]['path'], spec[key]['sha256'])
    verify(spec['science_lock'], spec['science_lock_sha256'])
    if spec.get('operation') != 'minimal_task_wire_adapter_bank' or spec.get('seed') not in (0, 1):
        raise PermissionError('unregistered trainer operation or seed')
    state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT']) / 'state.json').read_text())
    for s in state['units'][active['unit_id']]['spec']['sources']:
        parts = Path(s['path']).as_posix().upper().split('/')
        if any(x in parts for x in ('VALIDATION', 'ATTACK_FIT_NEW', 'INNER_SELECTION')):
            raise PermissionError('FIT2/ATT/INNER are not trainer sources')
    validate_recipe(spec['recipe'])
    lock = json.loads(Path(spec['science_lock']).read_text())
    seed = spec['seed']
    validate_science_lock(lock, seed=seed, checkpoint_sha256=spec['checkpoint']['sha256'],
                          geometry_sha256=spec['geometry_binding_sha256'], adapter_binding_sha256=spec['adapter_binding']['sha256'])
    binding = json.loads(Path(spec['adapter_binding']['path']).read_text())
    admission = json.loads(Path(spec['admission']['path']).read_text())
    split = json.loads(Path(spec['role_split']['path']).read_text())
    queue_guard()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .wires import prepare_geometry, FrozenIncomeGuard
    from .wires import probe_key
    from .guard import pool_pass
    model, payload = load_checkpoint(verify(spec['checkpoint']['path'], spec['checkpoint']['sha256']))
    if payload.get('family') != 'RAW_JOINT' or payload.get('strength') != 0.3 or payload.get('epoch') != 5 or payload.get('seed') != seed:
        raise PermissionError('starting package must be admitted RAW strength .3 epoch 5')
    for p in model.parameters():
        p.requires_grad_(False)
    geometry = prepare_geometry(json.loads(verify(spec['geometry']['path'], spec['geometry']['sha256']).read_text()),
                                expected_sha256=spec['geometry_binding_sha256'], source_seed=seed,
                                checkpoint_sha256=spec['checkpoint']['sha256'])
    if (binding.get('source_checkpoint_sha256') != spec['checkpoint']['sha256'] or binding.get('seed') != seed
            or binding.get('geometry_binding_sha256') != geometry.binding_sha256):
        raise PermissionError('adapter binding differs from source/geometry')
    scale = float(binding['scale'])
    frozen_state = {k: tensor_sha(v) for k, v in model.state_dict().items()}

    def income_guard():
        if any(tensor_sha(v) != frozen_state[k] for k, v in model.state_dict().items()):
            raise AssertionError('a frozen source tensor changed')
        return True
    roles = load_roles(spec, verify)
    for role in ('UPDATE', 'GUARD'):
        if tuple(map(str, roles[role]['feature_names'])) != tuple(model.names):
            raise PermissionError('feature order differs')
    adapter0 = Adapter(model, geometry, scale)
    if adapter0.rank != binding['rank'] or adapter0.n_parameters() != binding['trainable_parameters']:
        raise PermissionError('measured rank/parameter count differs from binding')
    xU = torch.from_numpy(roles['UPDATE']['X']); xG = torch.from_numpy(roles['GUARD']['X'])
    hU, qU, iU = adapter0.frozen_features(xU); hG, qG, iG = adapter0.frozen_features(xG)
    yU = torch.from_numpy(roles['UPDATE']['occupation']).long(); yG = torch.from_numpy(roles['GUARD']['occupation']).long()
    sU = torch.from_numpy(roles['UPDATE']['sex']).long()
    probe_pos = sorted(range(len(xU)), key=lambda i: (probe_key(roles['UPDATE']['row_id'][i]), i))[:512]
    if hashlib.sha256(json.dumps(probe_pos).encode()).hexdigest() != split['probe_position_sha256']:
        raise PermissionError('frozen probe differs')
    hP = torch.cat((hU, hG)); qP = torch.cat((qU, qG)); yP = torch.cat((yU, yG))
    if len(yP) != 10223:
        raise PermissionError('task pool rows differ')
    with torch.no_grad():
        lp0U = torch.log_softmax(adapter0.wire(hU, qU)['centered_logits'], 1)
        lp0G = torch.log_softmax(adapter0.wire(hG, qG)['centered_logits'], 1)
    lp0P = torch.cat((lp0U, lp0G))
    out = Path(spec['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private trainer directory required')
    out.mkdir(parents=True, mode=0o700)
    task = Task(len(yU), len(yG))
    start_pool = pool_pass(adapter0, hP, qP, yP, lp0P)
    if abs(start_pool['kl']) > 1e-12:
        raise AssertionError('teacher anchor KL nonzero at the zero residual')
    ref_rows = _draw(random.Random(_seed(seed, 'reference')), len(yU))
    ctx = dict(seed=seed, geometry=geometry, scale=scale, task=task, source_model=model, hU=hU, qU=qU, iU=iU, yU=yU, sU=sU,
               hG=hG, qG=qG, yG=yG, hP=hP, qP=qP, yP=yP, lp0P=lp0P, lp0U=lp0U, lp0G=lp0G, reference_rows=ref_rows,
               probe_h=hU[probe_pos], probe_q=qU[probe_pos], out=out, start_pool=start_pool,
               wire_tolerance=admission['wire_tolerance'], income_guard=income_guard,
               checkpoint_sha256=spec['checkpoint']['sha256'], adapter_binding_sha256=spec['adapter_binding']['sha256'])
    norm = normalization(ctx, out)
    coefs = {'J_TRUST': norm['coefficients']['joint']['coefficient'], 'J_FREE': norm['coefficients']['joint']['coefficient'],
             'L_TRUST': norm['coefficients']['local']['coefficient'], 'L_FREE': norm['coefficients']['local']['coefficient'],
             'TASK_TRUST': None}
    ctx['coefficients'] = coefs
    ctx['common'] = {'source_seed': seed, 'starting_checkpoint_sha256': spec['checkpoint']['sha256'],
                     'science_lock_sha256': spec['science_lock_sha256'], 'recipe_sha256': canonical_sha(RECIPE),
                     'adapter_binding_sha256': spec['adapter_binding']['sha256'], 'update_rows': len(yU), 'guard_rows': len(yG),
                     'task_weights': {'update': task.w_update, 'guard': task.w_guard}, 'proposals_per_epoch': RECIPE['proposals_per_epoch'],
                     'FIT2_loaded': False, 'INNER_loaded': False, 'ATT_loaded': False, 'scale': scale,
                     'trainable_parameters': adapter0.n_parameters(), 'geometry_binding_sha256': geometry.binding_sha256,
                     'normalization_warm_critic_state_sha256': norm['warm_critic_state_sha256']}
    summaries = {}
    for arm in ARMS:
        queue_guard()
        rule = ARM_RULES[arm]
        if rule['privacy'] is not None and coefs[arm] is None:
            summaries[arm] = {'arm': arm, 'status': 'UNAVAILABLE_FIXED_NORMALIZATION', 'checkpoints': []}
            continue
        summaries[arm] = run_arm(arm, ctx)
        if summaries[arm]['warm_critic_state_sha256'] != norm['warm_critic_state_sha256']:
            raise AssertionError('arms do not share the matched warm critic state')
    income_guard()
    result = {'schema': 'MTA1-adapter-bank-training-v1', 'queue_unit_id': active['unit_id'], 'seed': seed,
              'recipe_sha256': canonical_sha(RECIPE), 'normalization': norm, 'coefficients': coefs,
              'start_pool': {k: start_pool[k] for k in ('nll', 'brier', 'correct', 'rows', 'accuracy', 'kl', 'ce')},
              'arms': summaries, 'FIT2_loaded': False, 'ATT_loaded': False, 'INNER_loaded': False,
              'scientific_status': 'TRAINING_ARTIFACTS_ONLY_NO_HELDOUT_OR_PRIVACY_CLAIM'}
    (out / 'SUMMARY.json').write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
    Path(spec['output']).parent.mkdir(parents=True, exist_ok=True)
    with Path(spec['output']).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, default=float)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
