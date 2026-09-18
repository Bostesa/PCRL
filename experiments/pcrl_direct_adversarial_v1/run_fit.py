"""Stage 3: fit the declared matrix of directly adversarial interfaces.

Every unit is resumable and idempotent: a completed unit is reused by hash, never
refitted to fill a count. A failed unit is quarantined, not overwritten.
"""
from __future__ import annotations

import argparse
import copy
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .attackers import FAMILIES, SINGLE_FAMILY, baseline_logits, fit_service_baselines
from .channel import build_channel, state_of
from .inputs import (OUT, POOLS, ROLE_ORDER, SOURCE_TASKS, Registry, array_hash, build_wires,
                     household_folds, load_frozen_state, read_json, representation_labels,
                     save_release, sha_file, write_json)
from .train import BETAS, POLICIES, TrainingConfig, fit_arm, make_fold

WIDTHS = (16, 8)
SINGLE_CELLS = tuple((16, p, b) for p in ('L2', 'C1') for b in (0.3, 1.0))
REPEAT_SEEDS = (1, 2)


def beta_tag(beta: float) -> str:
    return f'b{int(round(beta * 100)):03d}'


def main_arm(width, policy, beta) -> str:
    return f'dax{width}_{policy}_{beta_tag(beta)}'


def arm_plan() -> dict:
    """The full declared matrix, as {block: [(arm, spec)]}. Enumerated before fitting."""
    plan = {'main': [], 'no_protection': [], 'single_attacker': [], 'optimizer_repeat': []}
    for width in WIDTHS:
        for policy in POLICIES:
            for beta in BETAS:
                plan['main'].append((main_arm(width, policy, beta),
                                     {'width': width, 'policy': policy, 'beta': beta,
                                      'families': FAMILIES, 'repeat': 0}))
    for width in WIDTHS:
        plan['no_protection'].append((f'dax{width}_none',
                                      {'width': width, 'policy': 'C1', 'beta': 0.0,
                                       'families': FAMILIES, 'repeat': 0}))
    for width, policy, beta in SINGLE_CELLS:
        plan['single_attacker'].append((f'{main_arm(width, policy, beta)}_single',
                                        {'width': width, 'policy': policy, 'beta': beta,
                                         'families': (SINGLE_FAMILY,), 'repeat': 0}))
    for width, policy, beta in SINGLE_CELLS:
        for repeat in REPEAT_SEEDS:
            plan['optimizer_repeat'].append((f'{main_arm(width, policy, beta)}_r{repeat}',
                                             {'width': width, 'policy': policy, 'beta': beta,
                                              'families': FAMILIES, 'repeat': repeat}))
    return plan


def machine_state() -> dict:
    load = os.getloadavg()
    return {'platform': platform.platform(), 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__,
            'cpu_count': os.cpu_count(), 'load_average': list(load),
            'torch_threads': torch.get_num_threads()}


def limit_threads():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(name, '1')
    torch.set_num_threads(1)


# ------------------------------------------------------------------ per-seed shared state
def warm_heads(model, fold, order, config: TrainingConfig):
    """Warm the Z-only source heads with the channel frozen.

    Because only the heads move, the RELEASED channel is bit-identical before and
    after, so `teacher` is unambiguous and identical for every policy at this width.
    """
    for parameter in model.mapper.parameters():
        parameter.requires_grad_(False)
    if model.compress is not None:
        for parameter in model.compress.parameters():
            parameter.requires_grad_(False)
    optimiser = torch.optim.Adam(model.heads.parameters(), lr=config.lr, betas=(0.9, 0.999),
                                 eps=1e-8)
    history = []
    for step in range(config.head_warmup):
        ix = torch.from_numpy(order[step % len(order)])
        optimiser.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = model.encode(fold.x[ix])
        losses = []
        for task in SOURCE_TASKS:
            y = fold.source[task][ix]
            valid = y >= 0
            scores = model.heads[task](z).reshape(-1)
            if bool(valid.any()):
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                    scores[valid], y[valid].float()))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimiser.step()
        if step % 20 == 0:
            history.append({"step": step, "source_loss": float(loss.detach())})
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return history


def seed_context(seed: int, config: TrainingConfig, registry: Registry) -> dict:
    state = load_frozen_state(seed, registry)
    labels = representation_labels(seed, registry)
    folds_rows = household_folds(labels['serials'])
    a0 = state['a0']

    folds = {name: make_fold(state, labels, rows, a0) for name, rows in folds_rows.items()}

    rng = np.random.default_rng(20265018 + seed)
    n = len(folds['mapper_fit'].x)
    needed = (config.attacker_warmup + config.mapper_updates
              * (config.attacker_updates_per_mapper + 1)
              + config.refresh_catchup * len(config.refresh_fractions) + config.head_warmup + 64)
    order = []
    while len(order) < needed:
        permutation = rng.permutation(n)
        order.extend(permutation[start:start + config.batch]
                     for start in range(0, n - config.batch + 1, config.batch))
    order = order[:needed]

    baselines = fit_service_baselines(
        {role: folds['p0_fit'].views[role] for role in ROLE_ORDER},
        {role: folds['p0_fit'].role_labels[role] for role in ROLE_ORDER},
        np.arange(len(folds['p0_fit'].x)), seed)
    for name in ('mapper_fit', 'monitor'):
        folds[name].base = baseline_logits(baselines, folds[name].views)

    shared = {}
    for width in WIDTHS:
        built = build_channel(width, a0, state['channel']['representation_fit'])
        model = built['model']
        head_history = warm_heads(model, folds['mapper_fit'], order, config)
        warmed = state_of(model)
        with torch.no_grad():
            teacher_full = model.encode(
                torch.from_numpy(np.asarray(
                    (state['pca']['representation_fit'] - a0['input_mean']) / a0['input_scale'],
                    np.float32)))
        variance = float(teacher_full.var(0, unbiased=False).mean())
        if not np.isfinite(variance) or variance <= 0:
            raise AssertionError(f'degenerate teacher variance at width {width}')
        for name in ('mapper_fit', 'monitor'):
            with torch.no_grad():
                folds[name].teacher = model.encode(folds[name].x).detach()

        def factory(width=width, warmed=warmed, built=built):
            fresh = build_channel(width, a0, state['channel']['representation_fit'])['model']
            fresh.load_state_dict(copy.deepcopy(warmed))
            return fresh

        shared[width] = {'model_factory': factory, 'order': order,
                         'teacher_variance': {width: variance},
                         'projection': built['projection'],
                         'initialisation': built['initialisation'],
                         'head_warmup': head_history,
                         'warmed_state_hash': array_hash(
                             np.concatenate([v.numpy().ravel() for v in warmed.values()]))}
    # Teacher tensors are width-specific: rebuild per width at fit time.
    teachers = {}
    for width in WIDTHS:
        model = shared[width]['model_factory']()
        with torch.no_grad():
            teachers[width] = {name: model.encode(folds[name].x).detach()
                               for name in ('mapper_fit', 'monitor')}
    return {'state': state, 'labels': labels, 'folds': folds, 'fold_rows': folds_rows,
            'baselines': baselines, 'shared': shared, 'teachers': teachers, 'a0': a0,
            'order_hash': array_hash(np.concatenate(order))}


def fit_unit(context, seed, arm, spec, config: TrainingConfig, out: Path) -> dict:
    width = spec['width']
    dest = out / f'seed_{seed}' / 'fits' / arm
    marker = dest / 'fit_complete.json'
    if marker.exists():
        record = read_json(marker)
        print('FIT_REUSED', seed, arm, flush=True)
        return record
    folds = context['folds']
    for name in ('mapper_fit', 'monitor'):
        folds[name].teacher = context['teachers'][width][name]
    local = TrainingConfig(**{**config.as_dict(),
                              'families': tuple(spec['families']),
                              'extra_optimizer_seed': spec['repeat'],
                              'refresh_fractions': tuple(config.refresh_fractions)})
    result = fit_arm(context['state'], context['labels'], folds, context['a0'], width,
                     spec['policy'], spec['beta'], seed, local, context['baselines'],
                     context['shared'][width])
    dest.mkdir(parents=True, exist_ok=True)

    model = result.pop('model')
    checkpoints = result.pop('checkpoints')
    torch.save({'selected_state': state_of(model),
                'checkpoint_states': [c['state'] for c in checkpoints],
                'checkpoint_steps': [c['step'] for c in checkpoints]},
               dest / 'checkpoints.pt')

    channel = release_channel(context, model, context['a0'])
    cache = build_wires(channel, context['state']['anchors'])
    release = save_release(out / f'seed_{seed}' / 'releases' / arm / 'releases.npz', cache)
    validate_cache(cache)

    record = {'seed': seed, 'arm': arm, **{k: v for k, v in spec.items()
                                           if k != 'families'},
              'families': list(spec['families']),
              'config': local.as_dict(), 'release': release,
              'checkpoints_sha256': sha_file(dest / 'checkpoints.pt'),
              'channel_hash': {pool: array_hash(channel[pool]) for pool in POOLS},
              'teacher_variance': context['shared'][width]['teacher_variance'][width],
              'initialisation': context['shared'][width]['initialisation'],
              **result}
    write_json(dest / 'fit_record.json', record)
    write_json(marker, {'seed': seed, 'arm': arm,
                        'runtime_seconds': record['runtime_seconds'],
                        'selected_step': record['selected_step'],
                        'release_sha256': release['sha256'],
                        'fit_record_sha256': sha_file(dest / 'fit_record.json')})
    print('FIT_DONE', seed, arm, round(record['runtime_seconds'], 1), 's step',
          record['selected_step'], flush=True)
    return read_json(marker)


def release_channel(context, model, a0) -> dict:
    from .inputs import standardize
    out = {}
    with torch.no_grad():
        for pool in POOLS:
            x = standardize(context['state']['pca'][pool], a0['input_mean'], a0['input_scale'])
            out[pool] = model.encode(x).numpy().astype(np.float64)
    return out


def validate_cache(cache: dict):
    for key, value in cache.items():
        if not np.isfinite(value).all():
            raise AssertionError(f'non-finite release values in {key}')
        if value.dtype != np.float64:
            raise AssertionError(f'{key} is not float64')


def run(out: Path = OUT, seeds=(0, 1, 2), blocks=('main', 'no_protection', 'single_attacker'),
        config: TrainingConfig = None) -> dict:
    limit_threads()
    out = Path(out)
    config = config or TrainingConfig()
    plan = arm_plan()
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        tick = time.perf_counter()
        context = seed_context(seed, config, registry)
        write_json(out / f'seed_{seed}' / 'fit_context.json', {
            'seed': seed,
            'a0_parity': context['state']['a0_parity'],
            'a0_checkpoint_sha256': context['a0']['checkpoint_sha256'],
            'fold_rows': {k: int(len(v)) for k, v in context['fold_rows'].items()},
            'fold_hashes': {k: array_hash(v) for k, v in context['fold_rows'].items()},
            'order_hash': context['order_hash'],
            'service_baselines': {r: {k: v for k, v in b.items() if k != 'model'}
                                  for r, b in context['baselines'].items()},
            'widths': {str(w): {'initialisation': context['shared'][w]['initialisation'],
                                'teacher_variance': context['shared'][w]['teacher_variance'][w],
                                'warmed_state_hash': context['shared'][w]['warmed_state_hash'],
                                'projection_explained_variance_ratio':
                                    (context['shared'][w]['projection'] or {}).get(
                                        'explained_variance_ratio')}
                       for w in WIDTHS},
            'context_seconds': time.perf_counter() - tick,
            'inputs': registry.dump()})
        records = {}
        for block in blocks:
            for arm, spec in plan[block]:
                records[arm] = {'block': block, **fit_unit(context, seed, arm, spec, config, out)}
        summary[str(seed)] = records
        write_json(out / f'seed_{seed}' / 'fit_summary.json', records)
    write_json(out / 'FIT_SUMMARY.json', {'seeds': summary, 'machine': machine_state(),
                                          'blocks': list(blocks),
                                          'config': (config or TrainingConfig()).as_dict()})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--blocks', nargs='+',
                   default=['main', 'no_protection', 'single_attacker'])
    p.add_argument('--mapper-updates', type=int, default=300)
    a = p.parse_args()
    config = TrainingConfig(mapper_updates=a.mapper_updates)
    with threadpool_limits(limits=1):
        run(a.out, tuple(a.seeds), tuple(a.blocks), config)


if __name__ == '__main__':
    main()
