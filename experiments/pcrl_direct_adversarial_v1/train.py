"""The alternating fit: a refreshed attacker ensemble against one shared utility objective.

Finite inner optimisation is **not** a solved minimax problem and confers no protection
certificate. What is fitted here is a bounded alternating search whose inner player is a
finite differentiable family; a low measured gain means that family did not find one in
its budget, nothing more.

Shared utility objective, identical in every new neural arm:

    U(theta) = mean_{income, employment} CE( Z-only head )  +  1.0 * teacher_distortion

`teacher_distortion` is the mean squared distance to the **frozen starting channel at
that width**, divided by that channel's fixed training variance. It is a
useful-capability proxy. It is **not** a guarantee that residence capability transfers,
and no claim here treats it as one.

Mapper objective:   `U + beta * policy_penalty(G)`
Attacker objective: each attacker minimises its own cross-entropy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from .attackers import (FAMILIES, RoleEnsemble, SINGLE_FAMILY, baseline_logits,
                        fit_service_baselines, masked_cross_entropy, service_view, service_width)
from .channel import build_channel, state_of
from .inputs import COALITION_ROLES, LOCAL_ROLES, ROLE_CLASSES, ROLE_ORDER, SOURCE_TASKS

POLICIES = ('L1', 'L2', 'C1')
BETAS = (0.1, 0.3, 1.0, 3.0)

# Locked slot schedule. Every policy spends the SAME number of attacker slots and the
# same total attacker optimiser-step budget. The local policies spend their two
# coalition-equivalent slots on independently initialised replicas of the two sensitive
# local roles, so the local controls are strong rather than merely cheaper.
SLOT_SCHEDULE = {
    'C1': tuple((role, 0) for role in ROLE_ORDER),
    'L1': (('A/public_coverage', 0), ('A/SEX', 0), ('A/RAC1P', 0), ('A/SEX', 1), ('A/RAC1P', 1)),
}
SLOT_SCHEDULE['L2'] = SLOT_SCHEDULE['L1']

# Which family is re-initialised at each refresh point. Locked before fitting.
REFRESH_FAMILY = ('mlp64', 'mlp64_32', 'mlp64')


@dataclass
class TrainingConfig:
    mapper_updates: int = 300
    checkpoint_every: int = 50
    attacker_updates_per_mapper: int = 5
    attacker_warmup: int = 100
    refresh_fractions: tuple = (0.25, 0.50, 0.75)
    refresh_catchup: int = 40
    head_warmup: int = 60
    batch: int = 256
    lr: float = 1e-3
    teacher_weight: float = 1.0
    nominal_mapper_updates: int = 600
    reduction_reason: str = ''
    extra_optimizer_seed: int = 0
    families: tuple = FAMILIES

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class FoldTensors:
    x: torch.Tensor
    ha: torch.Tensor
    hb: torch.Tensor
    source: dict
    role_labels: dict
    views: dict
    base: dict = field(default_factory=dict)
    teacher: torch.Tensor = None


def make_fold(state: dict, labels: dict, rows: np.ndarray, a0: dict) -> FoldTensors:
    from .inputs import standardize
    pca = state['pca']['representation_fit'][rows]
    anchors = state['anchors']
    ha = torch.tensor(anchors['representation_fit/A'][rows], dtype=torch.float32)
    hb = torch.tensor(anchors['representation_fit/B'][rows], dtype=torch.float32)
    x = standardize(pca, a0['input_mean'], a0['input_scale'])
    source = {t: torch.from_numpy(labels['source'][t][rows]) for t in SOURCE_TASKS}
    protected = labels['protected']
    role_labels = {role: torch.from_numpy(protected[role.split('/')[1]][rows]) for role in ROLE_ORDER}
    views = {role: service_view(role, ha, hb) for role in ROLE_ORDER}
    return FoldTensors(x=x, ha=ha, hb=hb, source=source, role_labels=role_labels, views=views)


def policy_penalty(gains: dict, policy: str) -> torch.Tensor:
    local = torch.stack([gains[r] for r in LOCAL_ROLES]).mean()
    if policy == 'L1':
        return local
    if policy == 'L2':
        return 2.0 * local
    if policy == 'C1':
        return local + torch.stack([gains[r] for r in COALITION_ROLES]).mean()
    raise ValueError(f'unknown policy {policy}')


def utility(model, fold: FoldTensors, index, teacher_variance: float, weight: float):
    z, logits = model(fold.x[index])
    losses = {}
    for task in SOURCE_TASKS:
        y = fold.source[task][index]
        valid = y >= 0
        scores = logits[task].reshape(-1)
        losses[task] = (F.binary_cross_entropy_with_logits(scores[valid], y[valid].float())
                        if bool(valid.any()) else scores.sum() * 0.0)
    source = torch.stack([losses[t] for t in SOURCE_TASKS]).mean()
    distortion = ((z - fold.teacher[index]) ** 2).mean() / teacher_variance
    return z, source + weight * distortion, {'source': source, 'distortion': distortion,
                                             'tasks': losses}


def role_gains(z, fold: FoldTensors, index, ensembles, config: TrainingConfig,
               detach_attackers: bool):
    """`G_j = max(0, max_k g_jk)` with a hard maximum and a deterministic tie rule.

    `torch.max` returns the FIRST maximiser, and the family order is fixed, so a tie
    resolves to the lowest `(slot, family)` position deterministically.
    """
    per_role, active = {}, {}
    candidates = {role: [] for role in ROLE_ORDER}
    labels_seen = {}
    for (role, slot), ensemble in ensembles.items():
        y = fold.role_labels[role][index]
        base = fold.base[role][index]
        valid = y >= 0
        if not bool(valid.any()):
            continue
        reference = F.cross_entropy(base[valid], y[valid])
        labels_seen[role] = int(valid.sum())
        features = torch.cat((fold.views[role][index], z), 1)
        for family in ensemble.families:
            net = ensemble.nets[family]
            if detach_attackers:
                for p in net.parameters():
                    p.requires_grad_(False)
            logits = base + net(features)
            gain = reference - F.cross_entropy(logits[valid], y[valid])
            candidates[role].append((f'{slot}:{family}', gain))
    for role in ROLE_ORDER:
        if not candidates[role]:
            per_role[role] = torch.zeros((), dtype=torch.float32)
            active[role] = None
            continue
        stacked = torch.stack([g for _, g in candidates[role]])
        best = int(torch.argmax(stacked.detach()))
        per_role[role] = torch.clamp(stacked[best], min=0.0)
        active[role] = candidates[role][best][0]
    return per_role, active, labels_seen


def attacker_loss(z, fold: FoldTensors, index, ensembles):
    total, count = 0.0, 0
    for (role, _slot), ensemble in ensembles.items():
        y = fold.role_labels[role][index]
        base = fold.base[role][index]
        features = torch.cat((fold.views[role][index], z), 1)
        for family in ensemble.families:
            logits = base + ensemble.nets[family](features)
            loss, known = masked_cross_entropy(logits, y)
            if known:
                total = total + loss
                count += 1
    return total, count


@torch.no_grad()
def _monitor_components(model, fold, ensembles, config, teacher_variance, policy, beta):
    index = torch.arange(len(fold.x))
    z, u_value, parts = utility(model, fold, index, teacher_variance, config.teacher_weight)
    gains, active, support = role_gains(z, fold, index, ensembles, config, detach_attackers=False)
    penalty = policy_penalty(gains, policy)
    return {'utility': float(u_value), 'source': float(parts['source']),
            'distortion': float(parts['distortion']),
            'gains': {r: float(v) for r, v in gains.items()},
            'active_attacker': active, 'label_support': support,
            'penalty': float(penalty), 'monitor_score': float(u_value + beta * penalty)}


@torch.no_grad()
def _role_monitor_ce(model, fold, ensembles):
    """Per-attacker monitor cross-entropy, used to keep the stronger of two attackers."""
    index = torch.arange(len(fold.x))
    z = model.encode(fold.x[index])
    out = {}
    for (role, slot), ensemble in ensembles.items():
        y = fold.role_labels[role][index]
        base = fold.base[role][index]
        features = torch.cat((fold.views[role][index], z), 1)
        for family in ensemble.families:
            logits = base + ensemble.nets[family](features)
            loss, known = masked_cross_entropy(logits, y)
            out[(role, slot, family)] = float(loss) if known else float('inf')
    return out


def fit_arm(state, labels, folds, a0, width, policy, beta, seed, config: TrainingConfig,
            baselines, shared, log_prefix='') -> dict:
    """One fitted interface: width x policy x beta x seed (x optimiser repeat)."""
    tick = time.perf_counter()
    torch.manual_seed(20263018 + 1000 * seed + 100 * width + 10 * POLICIES.index(policy)
                      + config.extra_optimizer_seed)

    model = shared['model_factory']()
    train, monitor = folds['mapper_fit'], folds['monitor']
    teacher_variance = shared['teacher_variance'][width]

    slots = SLOT_SCHEDULE[policy]
    hb_width = int(train.hb.shape[1])
    ensembles = {}
    for position, (role, replica) in enumerate(slots):
        ensembles[(role, replica)] = RoleEnsemble(
            role, replica, config.families,
            service_width(role, hb_width) + width,
            seed * 10 + config.extra_optimizer_seed)
        ensembles[(role, replica)].position = position

    attacker_params = [p for e in ensembles.values() for p in e.parameters()]
    attacker_opt = torch.optim.Adam(attacker_params, lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
    mapper_opt = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8)

    order = shared['order']
    cursor = {'value': 0}
    n = len(train.x)

    def next_batch():
        ix = order[cursor['value'] % len(order)]
        cursor['value'] += 1
        return torch.from_numpy(ix)

    def attacker_step(subset=None):
        ix = next_batch()
        with torch.no_grad():
            z = model.encode(train.x[ix])
        for p in attacker_params:
            p.requires_grad_(True)
        attacker_opt.zero_grad(set_to_none=True)
        if subset is None:
            total, count = attacker_loss(z, train, ix, ensembles)
        else:
            total, count = attacker_loss(z, train, ix, subset)
        if count:
            if not torch.isfinite(total):
                raise AssertionError('non-finite attacker loss')
            total.backward()
            attacker_opt.step()
        return (float(total.detach()) if count else 0.0) / max(count, 1)

    warm = [attacker_step() for _ in range(config.attacker_warmup)]

    refresh_steps = {int(round(f * config.mapper_updates)): REFRESH_FAMILY[i]
                     for i, f in enumerate(config.refresh_fractions)}
    checkpoints, trace, refresh_log = [], [], []

    def take_checkpoint(step):
        record = _monitor_components(model, monitor, ensembles, config, teacher_variance,
                                     policy, beta)
        record.update(step=step, state_index=len(checkpoints))
        checkpoints.append({'step': step, 'state': state_of(model), 'monitor': record})
        return record

    take_checkpoint(0)
    for step in range(1, config.mapper_updates + 1):
        if step in refresh_steps:
            family = refresh_steps[step]
            before = _role_monitor_ce(model, monitor, ensembles)
            fresh = {}
            for key, ensemble in ensembles.items():
                role, replica = key
                incumbent = ensemble.nets[family]
                new_seed = (20264018 + 1000 * seed + 100 * step
                            + 10 * ROLE_ORDER.index(role) + replica)
                replacement = ensemble.replace(family, incumbent_input_dim(incumbent), new_seed)
                fresh[key] = (incumbent, replacement)
            catchup_params = [p for _i, r in fresh.values() for p in r.parameters()]
            catch_opt = torch.optim.Adam(catchup_params, lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
            subset = {k: _single_family_view(e, family) for k, e in ensembles.items()}
            for _ in range(config.refresh_catchup):
                ix = next_batch()
                with torch.no_grad():
                    z = model.encode(train.x[ix])
                catch_opt.zero_grad(set_to_none=True)
                total, count = attacker_loss(z, train, ix, subset)
                if count:
                    total.backward()
                    catch_opt.step()
            after = _role_monitor_ce(model, monitor, ensembles)
            decisions = {}
            for key, (incumbent, replacement) in fresh.items():
                role, replica = key
                old_ce = before[(role, replica, family)]
                new_ce = after[(role, replica, family)]
                keep_new = new_ce < old_ce
                if not keep_new:
                    ensembles[key].nets[family] = incumbent
                decisions[f'{role}|{replica}'] = {
                    'incumbent_monitor_ce': old_ce, 'refreshed_monitor_ce': new_ce,
                    'kept': 'refreshed' if keep_new else 'incumbent',
                    'recovery': old_ce - new_ce}
            attacker_params = [p for e in ensembles.values() for p in e.parameters()]
            attacker_opt = torch.optim.Adam(attacker_params, lr=config.lr, betas=(0.9, 0.999),
                                            eps=1e-8)
            refresh_log.append({'step': step, 'family': family, 'decisions': decisions})

        for _ in range(config.attacker_updates_per_mapper):
            attacker_step()

        ix = next_batch()
        for p in attacker_params:
            p.requires_grad_(False)
        mapper_opt.zero_grad(set_to_none=True)
        z, u_value, parts = utility(model, train, ix, teacher_variance, config.teacher_weight)
        gains, active, _support = role_gains(z, train, ix, ensembles, config,
                                             detach_attackers=False)
        penalty = policy_penalty(gains, policy)
        loss = u_value + beta * penalty
        if not torch.isfinite(loss):
            raise AssertionError('non-finite mapper loss')
        loss.backward()
        grad_norm = float(torch.sqrt(sum((p.grad.double() ** 2).sum()
                                         for p in model.parameters() if p.grad is not None)))
        mapper_opt.step()
        for p in attacker_params:
            p.requires_grad_(True)

        if step % 10 == 0 or step == 1:
            trace.append({'step': step, 'train_loss': float(loss), 'utility': float(u_value),
                          'source': float(parts['source']),
                          'distortion': float(parts['distortion']),
                          'penalty': float(penalty), 'mapper_grad_norm': grad_norm,
                          'gains': {r: float(v) for r, v in gains.items()},
                          'active_attacker': active})
        if step % config.checkpoint_every == 0:
            take_checkpoint(step)

    scores = [c['monitor']['monitor_score'] for c in checkpoints]
    chosen = int(np.argmin(scores))
    model.load_state_dict(checkpoints[chosen]['state'])
    model.eval()

    return {'model': model, 'checkpoints': checkpoints, 'selected_index': chosen,
            'selected_step': checkpoints[chosen]['step'],
            'monitor_scores': [{'step': c['step'], **c['monitor']} for c in checkpoints],
            'trace': trace, 'refreshes': refresh_log,
            'attacker_warmup_loss': {'first': warm[0] if warm else None,
                                     'last': warm[-1] if warm else None},
            'slots': [f'{r}|{s}' for r, s in slots],
            'attacker_families': list(config.families),
            'attacker_updates_total': int(config.attacker_warmup
                                          + config.mapper_updates * config.attacker_updates_per_mapper
                                          + config.refresh_catchup * len(config.refresh_fractions)),
            'mapper_updates_total': config.mapper_updates,
            'runtime_seconds': time.perf_counter() - tick}


def incumbent_input_dim(net) -> int:
    first = net if isinstance(net, torch.nn.Linear) else net[0]
    return int(first.in_features)


def _single_family_view(ensemble: RoleEnsemble, family: str) -> RoleEnsemble:
    """A shallow view of one ensemble restricted to a single family, for catch-up."""
    view = RoleEnsemble.__new__(RoleEnsemble)
    view.role = ensemble.role
    view.slot = ensemble.slot
    view.families = (family,)
    view.classes = ensemble.classes
    view.nets = {family: ensemble.nets[family]}
    view.seeds = ensemble.seeds
    view.refreshes = ensemble.refreshes
    return view


__all__ = ['POLICIES', 'BETAS', 'SLOT_SCHEDULE', 'REFRESH_FAMILY', 'TrainingConfig',
           'FoldTensors', 'make_fold', 'policy_penalty', 'utility', 'role_gains', 'fit_arm',
           'fit_service_baselines', 'baseline_logits', 'build_channel']
