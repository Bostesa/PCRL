"""Track N fitting: teacher strength x initialisation x policy x beta, at width 16.

Reuses the direct-adversarial trainer (`fit_arm`) verbatim: residual-logit attackers,
internal household folds, slot schedule with matched local replicas, the refresh
recipe, 600 mapper updates, checkpoints every 100 including step 0, and the equal-budget
fresh-probe checkpoint selection of that study's amendment 1. Only three things are
varied, all prospectively:

* `gamma` (the teacher-distortion coefficient; `TrainingConfig.teacher_weight`)
  in `{0, .01, .1, 1}`;
* initialisation in `{A0, J}` — **J's actual trained mapper and heads**, proved
  bit-exact against J's release, never a distillation or same-name approximation;
* `beta in {1, 3}` for policies `{L1, L2, C1}`, plus `beta = 0` continuations.

The teacher is the frozen **A0** channel for BOTH initialisations, and the distortion is
normalised by A0's training variance, so changing the initialisation does not also
change the target being preserved. J therefore starts at a nonzero distortion; that is
a measured property of the design, not an implementation error.

Heads are warmed identically within an initialisation (60 updates, mapper frozen, the
same minibatch order), so the released channel at step 0 is bitwise the untouched
starting channel and is always one of the selectable checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1.attackers import FAMILIES
from experiments.pcrl_direct_adversarial_v1.channel import AdversarialChannel, state_of
from experiments.pcrl_direct_adversarial_v1.run_fit import (release_channel, seed_context,
                                                            validate_cache, warm_heads)
from experiments.pcrl_direct_adversarial_v1.train import TrainingConfig, fit_arm

from .common import OUT, limit_threads, read_json, sha_file, utcnow, write_json_atomic
from .run_fit_e import frozen_channel, load_mapper

GAMMAS = (0.0, 0.01, 0.1, 1.0)
BETAS = (1.0, 3.0)
POLICIES = ('L1', 'L2', 'C1')
INITS = ('A0', 'J')
WIDTH = 16


def gamma_tag(g: float) -> str:
    return {0.0: 'g000', 0.01: 'g001', 0.1: 'g010', 1.0: 'g100'}[g]


def unit_name(init, gamma, policy, beta) -> str:
    if beta == 0.0:
        return f'N_{init}_{gamma_tag(gamma)}_none'
    return f'N_{init}_{gamma_tag(gamma)}_{policy}_b{int(round(beta * 100)):03d}'


def plan(gammas=GAMMAS) -> list:
    units = []
    for init in INITS:
        for gamma in gammas:
            for policy in POLICIES:
                for beta in BETAS:
                    units.append({'unit': unit_name(init, gamma, policy, beta), 'init': init,
                                  'gamma': gamma, 'policy': policy, 'beta': beta,
                                  'block': 'main'})
            units.append({'unit': unit_name(init, gamma, 'C1', 0.0), 'init': init,
                          'gamma': gamma, 'policy': 'C1', 'beta': 0.0,
                          'block': 'utility_only_continuation'})
    return units


def j_factory(seed: int, context: dict, registry, config: TrainingConfig) -> dict:
    """A width-16 channel initialised at J's trained mapper and heads, heads warmed."""
    state = context['state']
    proof = frozen_channel(seed, registry, state, 'J')
    j = proof['model']
    a0 = context['a0']
    if not (np.array_equal(j['input_mean'], a0['input_mean'])
            and np.array_equal(j['input_scale'], a0['input_scale'])):
        raise AssertionError('J and A0 standardisers differ; a shared input path is required')
    model = AdversarialChannel(WIDTH, j['mapper'], j['heads'], None)
    history = warm_heads(model, context['folds']['mapper_fit'], context['shared'][WIDTH]['order'],
                         config)
    warmed = state_of(model)

    def factory():
        fresh = AdversarialChannel(WIDTH, j['mapper'], j['heads'], None)
        fresh.load_state_dict(copy.deepcopy(warmed))
        return fresh

    with torch.no_grad():
        start = {name: factory().encode(context['folds'][name].x).detach()
                 for name in ('mapper_fit', 'monitor')}
    teacher = context['teachers'][WIDTH]
    variance = context['shared'][WIDTH]['teacher_variance'][WIDTH]
    initial_distortion = {name: float(((start[name] - teacher[name]) ** 2).mean() / variance)
                          for name in start}
    return {'model_factory': factory, 'order': context['shared'][WIDTH]['order'],
            'teacher_variance': {WIDTH: variance},
            'initialisation': "J's trained mapper and heads, bit-exact to J's release; heads "
                              'warmed 60 updates with the mapper frozen',
            'head_warmup': history, 'parity': proof['parity'],
            'initial_distortion_vs_A0_teacher': initial_distortion,
            'checkpoint_sha256': j['checkpoint_sha256']}


def fit_unit(out: Path, seed: int, spec: dict, context: dict, shared: dict,
             config: TrainingConfig) -> dict:
    dest = out / f'seed_{seed}' / 'fits' / spec['unit']
    marker = dest / 'fit_complete.json'
    if marker.exists():
        print('N_REUSED', seed, spec['unit'], flush=True)
        return read_json(marker)
    folds = context['folds']
    for name in ('mapper_fit', 'monitor'):
        folds[name].teacher = context['teachers'][WIDTH][name]        # A0 teacher, always
    local = TrainingConfig(**{**config.as_dict(), 'teacher_weight': spec['gamma'],
                              'families': tuple(FAMILIES), 'extra_optimizer_seed': 0,
                              'refresh_fractions': tuple(config.refresh_fractions)})
    result = fit_arm(context['state'], context['labels'], folds, context['a0'], WIDTH,
                     spec['policy'], spec['beta'], seed, local, context['baselines'], shared)
    dest.mkdir(parents=True, exist_ok=True)
    model = result.pop('model')
    checkpoints = result.pop('checkpoints')
    torch.save({'selected_state': state_of(model),
                'checkpoint_states': [c['state'] for c in checkpoints],
                'checkpoint_steps': [c['step'] for c in checkpoints]}, dest / 'checkpoints.pt')
    channel = release_channel(context, model, context['a0'])
    cache = dax.build_wires(channel, context['state']['anchors'])
    validate_cache(cache)
    release = dax.save_release(out / f'seed_{seed}' / 'releases' / spec['unit'] / 'releases.npz',
                               cache)
    record = {'seed': seed, **spec, 'config': local.as_dict(), 'release': release,
              'checkpoints_sha256': sha_file(dest / 'checkpoints.pt'),
              'channel_hash': {p: dax.array_hash(channel[p]) for p in dax.POOLS},
              'teacher': 'frozen A0 channel (warmed-head state), A0 training variance',
              'teacher_variance': shared['teacher_variance'][WIDTH],
              'initialisation': shared['initialisation'], **result}
    write_json_atomic(dest / 'fit_record.json', record)
    done = {'seed': seed, 'unit': spec['unit'], 'runtime_seconds': record['runtime_seconds'],
            'selected_step': record['selected_step'], 'release_sha256': release['sha256'],
            'channel_hash_test': record['channel_hash']['test'],
            'fit_record_sha256': sha_file(dest / 'fit_record.json')}
    write_json_atomic(marker, done)
    print('N_DONE', seed, spec['unit'], round(record['runtime_seconds'], 1), 's step',
          record['selected_step'], flush=True)
    return done


def run(out: Path = OUT, seeds=(0, 1, 2), gammas=GAMMAS, only=None) -> dict:
    limit_threads()
    out = Path(out)
    config = TrainingConfig(mapper_updates=600)
    units = [u for u in plan(gammas) if only is None or u['unit'] in only]
    summary = {}
    for seed in seeds:
        registry = dax.Registry.new()
        tick = time.perf_counter()
        context = seed_context(seed, config, registry)
        shared = {'A0': context['shared'][WIDTH],
                  'J': j_factory(seed, context, registry, config)}
        write_json_atomic(out / f'seed_{seed}' / 'track_n_context.json', {
            'seed': seed, 'generated_utc': utcnow(),
            'a0_parity': context['state']['a0_parity'],
            'j_parity': shared['J']['parity'],
            'j_checkpoint_sha256': shared['J']['checkpoint_sha256'],
            'j_initial_distortion_vs_A0_teacher': shared['J']['initial_distortion_vs_A0_teacher'],
            'teacher_variance_A0': context['shared'][WIDTH]['teacher_variance'][WIDTH],
            'warmed_state_hash_A0': context['shared'][WIDTH]['warmed_state_hash'],
            'order_hash': context['order_hash'],
            'fold_rows': {k: int(len(v)) for k, v in context['fold_rows'].items()},
            'context_seconds': time.perf_counter() - tick})
        records = {}
        for spec in units:
            records[spec['unit']] = fit_unit(out, seed, spec, context, shared[spec['init']],
                                             config)
        summary[str(seed)] = records
        write_json_atomic(out / f'seed_{seed}' / 'track_n_summary.json', records)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--gammas', type=float, nargs='+', default=list(GAMMAS))
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, tuple(args.seeds), tuple(args.gammas), args.only)


if __name__ == '__main__':
    main()
