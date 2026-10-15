"""ROOT-authorized, queue-only six-arm two-recipient adapter bank for one source seed.

Receives only the physical UPDATE and GUARD exports (features, income and
occupation labels; SEX for UPDATE only). FIT2, ATT and INNER are never
sources. Saves complete release-pair checkpoints and label-free movement
diagnostics; no held-out metric exists here.
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
from .adapter import canonical_sha, tensor_sha, RecipientAdapter, RECIPIENTS

VIEWS = ('income', 'occupation', 'pair')


def _seed(source_seed, purpose):
    return int(hashlib.sha256(f'TRA1-adapter|{source_seed}|{purpose}'.encode()).hexdigest()[:8], 16)


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
    want = {'UPDATE': ('features', 'income', 'occupation', 'sex'), 'GUARD': ('features', 'income', 'occupation')}
    if set(spec['roles']) != set(want):
        raise PermissionError('trainer receives only UPDATE and GUARD')
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
    """Nine actual-wire critics (income21, occupation29, pair50 x linear/mlp32/mlp64) on UPDATE rows/SEX."""
    def __init__(self, source_seed, sex):
        import torch
        prior = torch.bincount(sex, minlength=2).double() / len(sex)
        if (prior <= 0).any():
            raise ValueError('both sensitive groups required in UPDATE')
        self.prior = prior; self.entropy = float(-(prior * prior.log()).sum())
        self.banks, self.optimizers, self.references = [], [], None
        self.optimizer_steps = 0; self.rounds = 0
        for view, width in zip(VIEWS, (21, 29, 50)):
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
    def raw_views(inc_full, occ_full):
        import torch
        return [inc_full, occ_full, torch.cat((inc_full, occ_full), 1)]

    def refresh(self, inc_full, occ_full):
        import torch
        from task_conditioned_representation.kernels import DiagonalReference
        with torch.no_grad():
            values = self.raw_views(inc_full, occ_full)
            prior = self.references or [None, None, None]
            self.references = [DiagonalReference.fit(v, old) for v, old in zip(values, prior)]
        for r in self.references:
            if not torch.isfinite(r.mean).all() or not torch.isfinite(r.scale).all() or (r.scale <= 0).any():
                raise ValueError('finite positive full-coordinate reference required')
        return [{'floor_count': r.floor_count, 'scale_min': float(r.scale.min()), 'scale_max': float(r.scale.max())}
                for r in self.references]

    def views(self, inc_full, occ_full):
        return [r.normalize(v) for r, v in zip(self.references, self.raw_views(inc_full, occ_full))]

    def update(self, inc_full, occ_full, sex):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            views = self.views(inc_full.detach(), occ_full.detach())
        for bank, optimizers, view in zip(self.banks, self.optimizers, views):
            for critic, optimizer in zip(bank, optimizers):
                for p in critic.parameters():
                    p.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(critic(view), sex)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite critic loss')
                loss.backward(); optimizer.step(); self.optimizer_steps += 1
        self.rounds += 1

    def terms(self, inc_full, occ_full, sex):
        import torch
        import torch.nn.functional as F
        views = self.views(inc_full, occ_full)
        terms, ces = {}, []
        for name, bank, view in zip(VIEWS, self.banks, views):
            for critic in bank:
                for p in critic.parameters():
                    p.requires_grad_(False)
            losses = torch.stack([F.cross_entropy(c(view), sex) for c in bank] + [view.sum() * 0 + self.entropy])
            terms[name] = 1 - losses.min() / self.entropy; ces.append(losses.detach().tolist())
        terms['critic_ces'] = ces
        return terms

    def state_sha(self):
        return canonical_sha([[{k: tensor_sha(v) for k, v in c.state_dict().items()} for c in bank] for bank in self.banks])

    def state_dict(self):
        return {'critics': [[copy.deepcopy(c.state_dict()) for c in bank] for bank in self.banks],
                'references': [None if r is None else r.snapshot() for r in (self.references or [])],
                'prior': self.prior.clone(), 'entropy': self.entropy, 'critic_optimizer_steps': self.optimizer_steps,
                'critic_rounds': self.rounds}


def privacy_objective(terms, kind):
    if kind == 'joint':
        return terms['income'] + terms['occupation'] + terms['pair']
    if kind == 'local':
        return terms['income'] + terms['occupation']
    raise ValueError('unregistered privacy objective')


class Task:
    """Shared population-weighted task objective per recipient, identical in every arm."""
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


def privacy_gradients(adapters, critics, feats, rows, sb, kind, active):
    """Gradients of the arm objective w.r.t. each active recipient residual, from one shared state."""
    import torch
    from .guard import flat
    params = {}; wires = []
    for i, ad in enumerate(adapters):
        if i in active and kind is not None:
            R = ad.R.detach().clone().requires_grad_(True); b = ad.b.detach().clone().requires_grad_(True)
            params[i] = (R, b)
            wires.append(ad.wire(feats[i]['hU'][rows], feats[i]['qU'][rows], R, b)['full'])
        else:
            with torch.no_grad():
                wires.append(ad.wire(feats[i]['hU'][rows], feats[i]['qU'][rows])['full'])
    terms = critics.terms(wires[0], wires[1], sb)
    if kind is None:
        return {i: None for i in active}, terms
    obj = privacy_objective(terms, kind)
    order = sorted(params)
    grads = torch.autograd.grad(obj, [t for i in order for t in params[i]], allow_unused=True)
    out = {}
    for k, i in enumerate(order):
        gR, gb = grads[2 * k], grads[2 * k + 1]
        gR = torch.zeros_like(params[i][0]) if gR is None else gR
        gb = torch.zeros_like(params[i][1]) if gb is None else gb
        out[i] = flat([gR, gb])
    return out, terms


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


def critic_round(critics, adapters, feats, rng, sU):
    import torch
    rows = _draw(rng, len(sU))
    with torch.no_grad():
        w = [ad.wire(feats[i]['hU'][rows], feats[i]['qU'][rows])['full'] for i, ad in enumerate(adapters)]
    critics.update(w[0], w[1], sU[rows])


def refresh(critics, adapters, feats, ref):
    import torch
    with torch.no_grad():
        w = [ad.wire(feats[i]['hU'][ref], feats[i]['qU'][ref])['full'] for i, ad in enumerate(adapters)]
    return critics.refresh(w[0], w[1])


def fresh_adapters(ctx):
    return [RecipientAdapter(ctx['source_model'], ctx['geometry'], i, ctx['scales'][i]) for i in (0, 1)]


def warm_critics(ctx, adapters):
    critics = Critics(ctx['seed'], ctx['sU'])
    refresh(critics, adapters, ctx['feats'], ctx['reference_rows'])
    rng = random.Random(_seed(ctx['seed'], 'warm'))
    for _ in range(RECIPE['warm_updates_per_critic']):
        critic_round(critics, adapters, ctx['feats'], rng, ctx['sU'])
    return critics


def normalization(ctx, private_dir):
    """Per-recipient lambda_J_i and lambda_L_i at the matched warm zero-residual state; no search."""
    import numpy as np
    import torch
    from .guard import pool_pass
    adapters = fresh_adapters(ctx)
    critics = warm_critics(ctx, adapters)
    F = ctx['feats']
    gpool = [pool_pass(adapters[i], F[i]['hG'], F[i]['qG'], F[i]['yG'], params=True, want=('ce',))['gradients']['ce'] for i in (0, 1)]
    rng = random.Random(_seed(ctx['seed'], 'normalization'))
    records, tensors = [], []
    for draw in range(RECIPE['normalization']['draws']):
        rows = _draw(rng, len(ctx['sU']))
        rec = {'draw': draw, 'rows_sha256': canonical_sha(rows)}; ten = {'rows': rows}
        gj, _ = privacy_gradients(adapters, critics, F, rows, ctx['sU'][rows], 'joint', (0, 1))
        gl, _ = privacy_gradients(adapters, critics, F, rows, ctx['sU'][rows], 'local', (0, 1))
        for i in (0, 1):
            gt, _ = ctx['task'].gradient(adapters[i], F[i]['hU'][rows], F[i]['qU'][rows], F[i]['yU'][rows], gpool[i])
            rec[f'task_norm_{i}'] = float(np.linalg.norm(gt)); rec[f'joint_norm_{i}'] = float(np.linalg.norm(gj[i]))
            rec[f'local_norm_{i}'] = float(np.linalg.norm(gl[i]))
            ten[f'task_{i}'] = torch.from_numpy(gt); ten[f'joint_{i}'] = torch.from_numpy(gj[i]); ten[f'local_{i}'] = torch.from_numpy(gl[i])
        records.append(rec); tensors.append(ten)
    out = {'draws': records, 'warm_critic_state_sha256': critics.state_sha(), 'coefficients': {}, 'task_median_norm': {}}
    for i in (0, 1):
        t = float(np.median([r[f'task_norm_{i}'] for r in records])); out['task_median_norm'][RECIPIENTS[i]] = t
        for kind in ('joint', 'local'):
            s = float(np.median([r[f'{kind}_norm_{i}'] for r in records]))
            coef = RECIPE['normalization']['target_ratio'] * t / s if s > 0 else None
            valid = (math.isfinite(t) and math.isfinite(s) and min(t, s) > RECIPE['normalization_min_median_norm']
                     and coef is not None and math.isfinite(coef) and coef <= RECIPE['maximum_coefficient'])
            out['coefficients'][f'{kind}|{RECIPIENTS[i]}'] = {'sensitive_median_norm': s, 'task_median_norm': t,
                                                               'coefficient': coef if valid else None, 'raw_coefficient': coef,
                                                               'status': 'FIXED' if valid else 'UNAVAILABLE_FIXED_NORMALIZATION'}
    path = private_dir / 'NORMALIZATION_TENSORS.pt'
    torch.save(tensors, path); path.chmod(0o600)
    out['tensor_sha256'] = _sha(path)
    (private_dir / 'NORMALIZATION.json').write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def save_checkpoint(adapters, ctx, arm, arm_dir, label, critics, extra):
    import torch
    payload = {'schema': 'TRA1-adapter-checkpoint-v1', 'campaign': RECIPE['campaign'], 'seed': ctx['seed'], 'arm': arm,
               'label': label, 'states': [ad.state() for ad in adapters], 'scales': list(ctx['scales']),
               'ranks': [ad.rank for ad in adapters], 'trainable_parameters': [ad.n_parameters() for ad in adapters],
               'source_checkpoint_sha256': ctx['checkpoint_sha256'], 'geometry_binding_sha256': ctx['geometry'].binding_sha256,
               'adapter_binding_sha256': ctx['adapter_binding_sha256'], 'bases': [ad.B.clone() for ad in adapters],
               'adapter_contract': RECIPE['adapter_contract'], 'recipe_sha256': canonical_sha(RECIPE), 'arm_rule': ARM_RULES[arm],
               'coefficients': ctx['coefficients'][arm], 'critic_state': critics.state_dict(),
               'trainer_metadata': dict(extra, label=label), 'selection_history': 'NONE_FIT2_ATT_INNER_NOT_LOADED'}
    path = arm_dir / f'{label}.pt'
    with path.open('xb') as stream:
        torch.save(payload, stream)
    path.chmod(0o600)
    return {'label': label, 'checkpoint_sha256': _sha(path), 'path_name': path.name,
            'R_sha256': [tensor_sha(ad.R) for ad in adapters], 'b_sha256': [tensor_sha(ad.b) for ad in adapters]}


def _assign(ad, vec):
    import torch
    n = ad.R.numel()
    ad.R = torch.from_numpy(vec[:n].reshape(ad.R.shape).copy()); ad.b = torch.from_numpy(vec[n:].copy())


def recipient_step(i, adapters, ctx, rows, g_priv, coef):
    """Pool metrics, task gradient and the per-recipient guarded step at the current shared state."""
    import numpy as np
    from .guard import pool_pass, guarded_step, flat
    F = ctx['feats'][i]; ad = adapters[i]
    current = pool_pass(ad, F['hP'], F['qP'], F['yP'], F['lp0P'], params=True, want=('nll', 'brier', 'kl'))
    gpool = pool_pass(ad, F['hG'], F['qG'], F['yG'], params=True, want=('ce',))
    g_task, ce_mb = ctx['task'].gradient(ad, F['hU'][rows], F['qU'][rows], F['yU'][rows], gpool['gradients']['ce'])
    applied = np.zeros_like(g_task) if g_priv is None else coef * g_priv
    proposal = -RECIPE['learning_rate'] * (g_task + applied)
    return {'current': current, 'gpool': gpool, 'g_task': g_task, 'ce_mb': ce_mb, 'applied': applied, 'proposal': proposal,
            'pre': flat([ad.R, ad.b])}


def apply_step(i, adapters, ctx, prep):
    import numpy as np
    from .guard import pool_pass, guarded_step, flat
    F = ctx['feats'][i]; ad = adapters[i]

    def evaluate():
        return pool_pass(ad, F['hP'], F['qP'], F['yP'], F['lp0P'])
    result = guarded_step(ad, prep['proposal'], prep['current'], ctx['anchors'][i], evaluate, RECIPE['maximum_actual_step_norm_per_recipient'])
    actual = result.pop('actual')
    after = flat([ad.R, ad.b])
    if result['status'] != 'ACCEPTED' and not np.array_equal(after, prep['pre']):
        raise AssertionError('rejected proposal did not restore exact residual')
    return result, actual, after


def counterfactual(i, adapters, ctx, pre_vecs, post_vecs, prep, actual_priv, g_priv, coef, rows, critics, kind):
    """Recipient i at its saved pre-step state: identical task gradient/pool/budgets/solver/ladder, privacy removed."""
    import numpy as np
    import torch
    from .guard import pool_pass, guarded_step, flat
    F = ctx['feats'][i]; ad = adapters[i]
    post_wire = probe_wire(ad, F['probe_h'], F['probe_q'])
    for k in (0, 1):
        _assign(adapters[k], pre_vecs[k])
    pre_wire = probe_wire(ad, F['probe_h'], F['probe_q'])
    rep = recipient_step(i, adapters, ctx, rows, None, None)
    gp2, _ = privacy_gradients(adapters, critics, ctx['feats'], rows, ctx['sU'][rows], kind, (i,))
    proposal2 = -RECIPE['learning_rate'] * (rep['g_task'] + coef * gp2[i])
    repeat_noise = float(np.max(np.abs(proposal2 - prep['proposal'])))
    cf_proposal = -RECIPE['learning_rate'] * prep['g_task']

    def evaluate():
        return pool_pass(ad, F['hP'], F['qP'], F['yP'], F['lp0P'])
    cf = guarded_step(ad, cf_proposal, prep['current'], ctx['anchors'][i], evaluate, RECIPE['maximum_actual_step_norm_per_recipient'])
    actual_cf = cf.pop('actual')
    cf_wire = probe_wire(ad, F['probe_h'], F['probe_q'])
    for k in (0, 1):
        _assign(adapters[k], post_vecs[k])
    if not all(np.array_equal(flat([adapters[k].R, adapters[k].b]), post_vecs[k]) for k in (0, 1)):
        raise AssertionError('counterfactual scan did not restore recorded state')
    tol = max(1e-10, 10 * repeat_noise)
    diff = actual_priv - actual_cf
    pvc = wire_change(post_wire, cf_wire)
    scan = {'recipient': RECIPIENTS[i], 'privacy_actual_norm': float(np.linalg.norm(actual_priv)), 'counterfactual_status': cf['status'],
            'counterfactual_actual_norm': float(np.linalg.norm(actual_cf)),
            'privacy_minus_counterfactual_norm': float(np.linalg.norm(diff)),
            'privacy_minus_counterfactual_max_abs': float(np.max(np.abs(diff))),
            'repeat_computation_proposal_max_abs_noise': repeat_noise, 'contribution_tolerance': tol,
            'privacy_contribution_beyond_noise': bool(np.max(np.abs(diff)) > tol),
            'cosine_privacy_vs_counterfactual': float(actual_priv @ actual_cf / max(1e-300, np.linalg.norm(actual_priv) * np.linalg.norm(actual_cf))),
            'update_probe_wire_change': wire_change(post_wire, pre_wire),
            'privacy_vs_counterfactual_probe_wire_difference': pvc,
            'privacy_changes_deployed_wire_beyond_tolerance': changed(pvc, ctx['wire_tolerance'][i]),
            'counterfactual_fraction': cf.get('accepted_fraction'), 'counterfactual_post_metrics': cf.get('post_metrics'),
            'counterfactual_active_set': (cf.get('projection') or {}).get('active_set')}
    state = {'recipient': i, 'pre_R': torch.from_numpy(pre_vecs[i][:ad.R.numel()].reshape(ad.R.shape).copy()),
             'pre_b': torch.from_numpy(pre_vecs[i][ad.R.numel():].copy()),
             'other_pre': torch.from_numpy(pre_vecs[1 - i].copy()), 'rows': list(rows), 'coefficient': coef,
             'critic_state': critics.state_dict(), 'g_task': torch.from_numpy(prep['g_task'].copy()),
             'g_privacy': torch.from_numpy(g_priv.copy()), 'proposal': torch.from_numpy(prep['proposal'].copy()),
             'pool_current': {k: prep['current'][k] for k in ('nll', 'brier', 'correct', 'kl', 'rows')},
             'pool_gradients': {k: torch.from_numpy(v.copy()) for k, v in prep['current']['gradients'].items()},
             'guard_ce_gradient': torch.from_numpy(prep['gpool']['gradients']['ce'].copy()), 'anchor': ctx['anchors'][i].record(),
             'actual_privacy_displacement': torch.from_numpy(actual_priv.copy()),
             'actual_counterfactual_displacement': torch.from_numpy(actual_cf.copy()),
             'counterfactual_status': cf['status'], 'counterfactual_fraction': cf.get('accepted_fraction')}
    return scan, state


def run_arm(arm, ctx):
    import numpy as np
    import torch
    from .guard import pool_pass, flat
    rule = ARM_RULES[arm]; kind = rule['privacy']; sched = rule['schedule']
    adapters = fresh_adapters(ctx)
    critics = warm_critics(ctx, adapters)
    warm_sha = critics.state_sha()
    arm_dir = ctx['out'] / arm; arm_dir.mkdir(mode=0o700)
    coefs = ctx['coefficients'][arm]
    meta = dict(ctx['common'], arm=arm, arm_rule=rule, anchors=[a.record() for a in ctx['anchors']], coefficients=coefs,
                warm_critic_state_sha256=warm_sha)
    F = ctx['feats']; sU = ctx['sU']; ref = ctx['reference_rows']
    probe0 = [probe_wire(adapters[i], F[i]['probe_h'], F[i]['probe_q']) for i in (0, 1)]
    checkpoints = []; diagnostics = []; epoch_rows = []; budget_rows = []
    statuses = {RECIPIENTS[i]: {} for i in (0, 1)}; opportunities = [0, 0]; macro_steps = 0
    scans = {0: [], 1: []}; scan_states = []; active_scans = {0: [], 1: []}
    P = RECIPE['proposals_per_epoch']; t0 = time.monotonic()
    rounds_per_step = RECIPE['critic_rounds_per_step'][sched]
    log = (arm_dir / 'UPDATE_LOG.jsonl').open('x'); os.chmod(arm_dir / 'UPDATE_LOG.jsonl', 0o600)
    batch_rng = random.Random(_seed(ctx['seed'], 'model_and_critic_draws'))
    trace = {0: [flat([adapters[0].R, adapters[0].b])], 1: [flat([adapters[1].R, adapters[1].b])]}
    frozen_records = []
    seq = sched == 'sequential'

    def checkpoint(label, epoch, phase_idx):
        checkpoints.append(save_checkpoint(adapters, ctx, arm, arm_dir, label, critics,
                                           dict(meta, opportunities=list(opportunities), macro_steps=macro_steps, phase=phase_idx + 1, epoch_in_phase=epoch)))
        d = {'label': label, 'phase': phase_idx + 1, 'epoch_in_phase': epoch, 'recipients': {}}
        for i in (0, 1):
            ad = adapters[i]; w = probe_wire(ad, F[i]['probe_h'], F[i]['probe_q']); ch = wire_change(w, probe0[i])
            pm = pool_pass(ad, F[i]['hP'], F[i]['qP'], F[i]['yP'], F[i]['lp0P'])
            um = pool_pass(ad, F[i]['hU'], F[i]['qU'], F[i]['yU'], F[i]['lp0U'])
            gm = pool_pass(ad, F[i]['hG'], F[i]['qG'], F[i]['yG'], F[i]['lp0G'])
            v = flat([ad.R, ad.b])
            d['recipients'][RECIPIENTS[i]] = {
                'probe_wire_change_vs_start': ch, 'probe_wire_changed_beyond_tolerance': changed(ch, ctx['wire_tolerance'][i]),
                'fitting_pool': {name: {k: m[k] for k in ('nll', 'brier', 'accuracy', 'correct', 'kl', 'ce')}
                                 for name, m in (('TASK_POOL', pm), ('UPDATE', um), ('GUARD', gm))},
                'residual_norm': float(np.linalg.norm(v)), 'is_zero': ad.is_zero()}
        diagnostics.append(d)

    for phase_idx, active in enumerate(rule['phases']):
        inactive = [i for i in (0, 1) if i not in active]
        frozen = {i: flat([adapters[i].R, adapters[i].b]) for i in inactive}
        frozen_records.append({'phase': phase_idx + 1, 'active': [RECIPIENTS[i] for i in active],
                               'frozen': {RECIPIENTS[i]: {'R_sha256': tensor_sha(adapters[i].R), 'b_sha256': tensor_sha(adapters[i].b),
                                                          'is_zero': adapters[i].is_zero()} for i in inactive}})
        prefix = f'P{phase_idx + 1}' if seq else ''
        checkpoint(f'{prefix}E0', 0, phase_idx)
        for epoch in range(1, RECIPE['epochs_per_active_recipient'] + 1):
            _guard()
            refinfo = refresh(critics, adapters, F, ref)
            counts = {RECIPIENTS[i]: {'accepted': 0, 'shortened': 0, 'rejected': 0, 'zero_displacement': 0, 'infeasible_state': 0} for i in active}
            for step_in_epoch in range(P):
                for _ in range(rounds_per_step):
                    critic_round(critics, adapters, F, batch_rng, sU)
                rows = _draw(batch_rng, len(sU))
                pre_vecs = {k: flat([adapters[k].R, adapters[k].b]) for k in (0, 1)}
                gpriv, terms = privacy_gradients(adapters, critics, F, rows, sU[rows], kind, tuple(active))
                preps = {i: recipient_step(i, adapters, ctx, rows, gpriv[i], None if coefs is None else coefs[RECIPIENTS[i]]) for i in active}
                results = {}
                for i in active:
                    results[i] = apply_step(i, adapters, ctx, preps[i])
                post_vecs = {k: flat([adapters[k].R, adapters[k].b]) for k in (0, 1)}
                ctx['source_guard']()
                for i in inactive:
                    if not np.array_equal(post_vecs[i], frozen[i]):
                        raise AssertionError('inactive recipient residual changed')
                macro_steps += 1
                record = {'phase': phase_idx + 1, 'epoch': epoch, 'step_in_epoch': step_in_epoch, 'macro_step': macro_steps,
                          'rows_sha256': canonical_sha(rows), 'critic_optimizer_steps': critics.optimizer_steps, 'critic_rounds': critics.rounds,
                          'critic_terms': {v: float(terms[v].detach()) for v in VIEWS}, 'critic_ces': terms['critic_ces'],
                          'elapsed_s': time.monotonic() - t0, 'recipients': {}}
                for i in active:
                    result, actual, after = results[i]; prep = preps[i]; name = RECIPIENTS[i]
                    opportunities[i] += 1
                    accepted = result['status'] == 'ACCEPTED'
                    statuses[name][result['status']] = statuses[name].get(result['status'], 0) + 1
                    c = counts[name]
                    c['accepted'] += int(accepted); c['shortened'] += int(accepted and (result.get('accepted_fraction') or 1) < 1)
                    c['rejected'] += int(result['status'] == 'REJECTED_ZERO_STEP'); c['zero_displacement'] += int(not np.any(actual))
                    c['infeasible_state'] += int(result['status'] == 'INFEASIBLE_CURRENT_STATE')
                    trace[i].append(after)
                    g_priv = gpriv[i]; coef = None if coefs is None else coefs[name]
                    gp_norm = 0.0 if g_priv is None else float(np.linalg.norm(g_priv))
                    tn = float(np.linalg.norm(prep['g_task']))
                    active_budget = bool(accepted and ((result.get('projection') or {}).get('active_set') or (result.get('accepted_fraction') or 1) < 1))
                    entry = {'opportunity': opportunities[i], 'status': result['status'], 'accepted_fraction': result.get('accepted_fraction'),
                             'proposed_norm': float(np.linalg.norm(prep['proposal'])),
                             'projected_norm': None if result.get('projection') is None else result['projection']['displacement_norm'],
                             'actual_norm': float(np.linalg.norm(actual)), 'task_gradient_norm': tn, 'privacy_gradient_norm': gp_norm,
                             'applied_privacy_norm': float(np.linalg.norm(prep['applied'])),
                             'achieved_strength_ratio': float(np.linalg.norm(prep['applied']) / max(1e-300, tn)),
                             'task_privacy_cosine': None if g_priv is None else float(prep['g_task'] @ g_priv / max(1e-300, tn * gp_norm)),
                             'active_budget_case': active_budget, 'minibatch_ce': prep['ce_mb'], 'guard_pool_ce': prep['gpool']['ce'],
                             'result': result}
                    if arm == 'J_BOTH' and accepted and np.any(actual):
                        want_cf = len(scans[i]) < 8
                        want_active = active_budget and len(active_scans[i]) < 16
                        if want_cf or want_active:
                            scan, state = counterfactual(i, adapters, ctx, pre_vecs, post_vecs, prep, actual, g_priv, coef, rows, critics, kind)
                            tag = dict(opportunity=opportunities[i], macro_step=macro_steps, epoch=epoch, phase=phase_idx + 1,
                                       privacy_accepted_fraction=result.get('accepted_fraction'), active_set=(result.get('projection') or {}).get('active_set'),
                                       active_budget_case=active_budget)
                            kinds = []
                            if want_cf:
                                scans[i].append(dict(scan, **tag)); kinds.append('first_eight')
                            if want_active:
                                active_scans[i].append(dict(scan, **tag)); kinds.append('active_budget')
                            scan_states.append(dict(state, **tag, scan_kinds=kinds))
                    record['recipients'][name] = entry
                    post = result.get('post_metrics') or {}
                    cur = prep['current']; an = ctx['anchors'][i]
                    budget_rows.append({'seed': ctx['seed'], 'arm': arm, 'recipient': name, 'phase': phase_idx + 1, 'epoch': epoch,
                                        'opportunity': opportunities[i], 'status': result['status'], 'fraction': result.get('accepted_fraction'),
                                        'actual_norm': float(np.linalg.norm(actual)), 'raw_slacks': result['raw_slacks'],
                                        'current_nll': cur['nll'], 'current_brier': cur['brier'], 'current_correct': cur['correct'], 'current_kl': cur['kl'],
                                        'post_nll': post.get('nll'), 'post_brier': post.get('brier'), 'post_correct': post.get('correct'),
                                        'post_kl': post.get('kl'), 'nll_limit': an.nll_limit, 'brier_limit': an.brier_limit,
                                        'correct_floor': an.correct_floor, 'kl_limit': an.kl_limit, 'active_budget_case': active_budget})
                log.write(json.dumps(record, sort_keys=True, allow_nan=False, default=float) + '\n'); log.flush()
            epoch_rows.append(dict(phase=phase_idx + 1, epoch=epoch, proposals=P, reference=refinfo, counts=counts))
            if epoch in RECIPE['checkpoint_epochs']:
                checkpoint(f'{prefix}E{epoch}', epoch, phase_idx)
    log.close()
    per_phase = RECIPE['epochs_per_active_recipient'] * P
    expected_rounds = RECIPE['warm_updates_per_critic'] + rounds_per_step * per_phase * len(rule['phases'])
    for i in (0, 1):
        if opportunities[i] != per_phase * sum(i in ph for ph in rule['phases']):
            raise AssertionError('proposal opportunities differ from the matched budget')
    if macro_steps != per_phase * len(rule['phases']) or critics.rounds != expected_rounds or critics.optimizer_steps != 9 * expected_rounds:
        raise AssertionError('matched critic budget differs')
    tpath = arm_dir / 'STATE_TRACE.npz'
    with tpath.open('xb') as stream:
        np.savez(stream, income=np.stack(trace[0]), occupation=np.stack(trace[1]))
    tpath.chmod(0o600)
    extra = {}
    if scan_states:
        p = arm_dir / 'COUNTERFACTUAL_STATES.pt'
        torch.save(scan_states, p); p.chmod(0o600); extra['counterfactual_states_sha256'] = _sha(p)
    bpath = arm_dir / 'BUDGET_TRACE.json'; bpath.write_text(json.dumps(budget_rows, sort_keys=True, default=float))
    summary = dict(meta, status='TRAINED', opportunities={RECIPIENTS[i]: opportunities[i] for i in (0, 1)}, macro_steps=macro_steps,
                   critic_rounds=critics.rounds, critic_optimizer_steps=critics.optimizer_steps,
                   update_status_counts=statuses, phase_freezing=frozen_records,
                   counts_note='zero_displacement overlaps rejected/infeasible/zero-projection; categories are not additive',
                   epoch_accounting=epoch_rows, checkpoints=checkpoints, label_free_and_fitting_pool_diagnostics=diagnostics,
                   counterfactual_scan={RECIPIENTS[i]: scans[i] for i in (0, 1)},
                   active_budget_scan={RECIPIENTS[i]: active_scans[i] for i in (0, 1)},
                   active_budget_cases={RECIPIENTS[i]: sum(r['active_budget_case'] for r in budget_rows if r['recipient'] == RECIPIENTS[i]) for i in (0, 1)},
                   update_log_sha256=_sha(arm_dir / 'UPDATE_LOG.jsonl'), state_trace_sha256=_sha(tpath),
                   budget_trace_sha256=_sha(bpath), runtime_s=time.monotonic() - t0, **extra)
    (arm_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


def build_context(spec, verify, model, geometry, binding, admission, split, seed):
    import numpy as np
    import torch
    from .wires import probe_key
    from .guard import pool_pass, Anchor
    roles = load_roles(spec, verify)
    for role in ('UPDATE', 'GUARD'):
        if tuple(map(str, roles[role]['feature_names'])) != tuple(model.names):
            raise PermissionError('feature order differs')
    scales = [float(s) for s in binding['scales']]
    adapters0 = [RecipientAdapter(model, geometry, i, scales[i]) for i in (0, 1)]
    if [a.rank for a in adapters0] != binding['ranks'] or [a.n_parameters() for a in adapters0] != binding['trainable_parameters']:
        raise PermissionError('measured rank/parameter count differs from binding')
    xU = torch.from_numpy(roles['UPDATE']['X']); xG = torch.from_numpy(roles['GUARD']['X'])
    probe_pos = sorted(range(len(xU)), key=lambda k: (probe_key(roles['UPDATE']['row_id'][k]), k))[:512]
    if hashlib.sha256(json.dumps(probe_pos).encode()).hexdigest() != split['probe_position_sha256']:
        raise PermissionError('frozen probe differs')
    feats = []; anchors = []; starts = []
    for i, task in enumerate(RECIPIENTS):
        ad = adapters0[i]
        hU, qU = ad.frozen_features(xU); hG, qG = ad.frozen_features(xG)
        yU = torch.from_numpy(roles['UPDATE'][task]).long(); yG = torch.from_numpy(roles['GUARD'][task]).long()
        hP = torch.cat((hU, hG)); qP = torch.cat((qU, qG)); yP = torch.cat((yU, yG))
        if len(yP) != 10223:
            raise PermissionError('task pool rows differ')
        with torch.no_grad():
            lp0U = torch.log_softmax(ad.wire(hU, qU)['centered_logits'], 1)
            lp0G = torch.log_softmax(ad.wire(hG, qG)['centered_logits'], 1)
        lp0P = torch.cat((lp0U, lp0G))
        f = dict(hU=hU, qU=qU, yU=yU, hG=hG, qG=qG, yG=yG, hP=hP, qP=qP, yP=yP, lp0U=lp0U, lp0G=lp0G, lp0P=lp0P,
                 probe_h=hU[probe_pos], probe_q=qU[probe_pos])
        start = pool_pass(ad, hP, qP, yP, lp0P)
        if abs(start['kl']) > 1e-12:
            raise AssertionError('teacher anchor KL nonzero at the zero residual')
        feats.append(f); anchors.append(Anchor(start, trust=True)); starts.append(start)
    sU = torch.from_numpy(roles['UPDATE']['sex']).long()
    return dict(feats=feats, anchors=anchors, start_pool=[{k: s[k] for k in ('nll', 'brier', 'correct', 'rows', 'accuracy', 'kl', 'ce')} for s in starts],
                sU=sU, scales=scales, task=Task(len(xU), len(xG)), probe_positions_sha256=split['probe_position_sha256'])


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, queue_guard = queue_config(spec_path, 'protection')
    for key in ('geometry', 'checkpoint', 'adapter_binding', 'role_split', 'admission'):
        verify(spec[key]['path'], spec[key]['sha256'])
    verify(spec['science_lock'], spec['science_lock_sha256'])
    if spec.get('operation') != 'two_recipient_task_wire_adapter_bank' or spec.get('seed') not in (0, 1):
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
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .wires import prepare_geometry
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
    frozen_state = {k: tensor_sha(v) for k, v in model.state_dict().items()}

    def source_guard():
        if any(tensor_sha(v) != frozen_state[k] for k, v in model.state_dict().items()):
            raise AssertionError('a frozen source tensor changed')
        return True
    base = build_context(spec, verify, model, geometry, binding, admission, split, seed)
    out = Path(spec['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private trainer directory required')
    out.mkdir(parents=True, mode=0o700)
    ref_rows = _draw(random.Random(_seed(seed, 'reference')), len(base['sU']))
    ctx = dict(base, seed=seed, geometry=geometry, source_model=model, reference_rows=ref_rows, out=out,
               wire_tolerance=admission['wire_tolerance'], source_guard=source_guard,
               checkpoint_sha256=spec['checkpoint']['sha256'], adapter_binding_sha256=spec['adapter_binding']['sha256'])
    norm = normalization(ctx, out)
    cj = {RECIPIENTS[i]: norm['coefficients'][f'joint|{RECIPIENTS[i]}']['coefficient'] for i in (0, 1)}
    cl = {RECIPIENTS[i]: norm['coefficients'][f'local|{RECIPIENTS[i]}']['coefficient'] for i in (0, 1)}
    coefs = {'T_BOTH': None, 'L_BOTH': cl, 'J_BOTH': cj, 'S12': cj, 'S21': cj, 'I_ONLY': cj}
    ctx['coefficients'] = coefs
    ctx['common'] = {'source_seed': seed, 'starting_checkpoint_sha256': spec['checkpoint']['sha256'],
                     'science_lock_sha256': spec['science_lock_sha256'], 'recipe_sha256': canonical_sha(RECIPE),
                     'adapter_binding_sha256': spec['adapter_binding']['sha256'], 'update_rows': 8180, 'guard_rows': 2043,
                     'task_weights': {'update': ctx['task'].w_update, 'guard': ctx['task'].w_guard},
                     'FIT2_loaded': False, 'INNER_loaded': False, 'ATT_loaded': False, 'scales': ctx['scales'],
                     'trainable_parameters': binding['trainable_parameters'], 'ranks': binding['ranks'],
                     'geometry_binding_sha256': geometry.binding_sha256,
                     'normalization_warm_critic_state_sha256': norm['warm_critic_state_sha256']}
    summaries = {}
    for arm in ARMS:
        queue_guard()
        rule = ARM_RULES[arm]
        needed = sorted({i for ph in rule['phases'] for i in ph})
        if rule['privacy'] is not None and any(coefs[arm][RECIPIENTS[i]] is None for i in needed):
            summaries[arm] = {'arm': arm, 'status': 'UNAVAILABLE_FIXED_NORMALIZATION', 'checkpoints': []}
            continue
        summaries[arm] = run_arm(arm, ctx)
        if summaries[arm]['warm_critic_state_sha256'] != norm['warm_critic_state_sha256']:
            raise AssertionError('arms do not share the matched warm critic state')
    source_guard()
    result = {'schema': 'TRA1-adapter-bank-training-v1', 'queue_unit_id': active['unit_id'], 'seed': seed,
              'recipe_sha256': canonical_sha(RECIPE), 'normalization': norm, 'coefficients': coefs,
              'start_pool': dict(zip(RECIPIENTS, ctx['start_pool'])),
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
