"""Exploratory 2017 transport of the fixed panel (PROTOCOL §8).

No 2017 encoder or eraser is fitted. Track N units carry their selected 2018 mapper;
Track E units carry their 2018 affine map applied to the frozen A0 or J channel
computed on 2017 inputs. Only fresh attackers and utility probes are fitted, on the
existing 2017 fit/validation partitions, by the predecessor's Mode B machinery.
Historical references (`J`, `A0`, `leace_A0`, `splince_A0`, `optnet16_C1`) are reused
from the predecessor's completed 2017 panel and transport study, not recomputed.

Every number is EXPLORATORY CROSS-YEAR DEVELOPMENT. 2017 has been used repeatedly; the
first locked 2017 experiment keeps its historical status.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from experiments.pcrl_direct_adversarial_v1 import transport as T
from experiments.pcrl_direct_adversarial_v1.channel import AdversarialChannel
from experiments.pcrl_direct_adversarial_v1.inputs import (Registry, load_frozen_state, read_json,
                                                           resolve)
from experiments.pcrl_direct_adversarial_v1.transport import TransportModel, load_new_channels

from . import evidence as ev_local
from .common import OUT, limit_threads, utcnow, write_json_atomic
from .run_fit_e import load_mapper


def affine_for(seed: int, unit: str, maps: dict) -> dict:
    """Express a Track E map in `apply_affine` form: (x - mean) @ P.T + mean."""
    record = maps[unit]
    base = unit.split('_')[1] if unit.startswith('E_') else 'J'
    base = 'A0' if base == 'A0' else 'ref_J'
    if record['kind'] == 'whitened_projection':
        linear = record['R'] @ record['P'] @ record['S']
        return {'base': base, 'projection': linear.T, 'mean': record['mu']}
    if record['kind'] == 'original_metric_projection':
        return {'base': base, 'projection': record['Q'].T, 'mean': record['mu']}
    if record['kind'] == 'affine_erasure':
        return {'base': base, 'projection': record['projection'], 'mean': record['mean']}
    raise ValueError(record['kind'])


def verify_affine_2018(seed: int, unit: str, spec: dict, state: dict, j_channel: dict):
    """The re-expressed map must rebuild the stored 2018 release (tolerance: float64 reassociation)."""
    from experiments.pcrl_invariant_baselines_v1.erasure_baselines import apply_affine
    stored = np.load(OUT / f'seed_{seed}' / 'releases' / unit / 'releases.npz')
    base = state['channel'] if spec['base'] == 'A0' else j_channel
    worst = 0.0
    for pool in ('representation_fit', 'test'):
        z = apply_affine(base[pool], spec['projection'], spec['mean'])
        worst = max(worst, float(np.abs(z - stored[f'wire/A/{pool}'][:, 4:]).max()))
    if worst > 1e-9:
        raise AssertionError(f'{unit} affine re-expression differs from its 2018 release: {worst}')
    return worst


def prepare(seed: int, registry: Registry, neural: list, affine_units: list):
    from experiments import acs_spectral_transport_eval as ev
    ev.DEV = resolve(f'results/{T.DEV_NAME}/seed_{seed}/maps.joblib').resolve().parents[1]
    ev.LOCAL = resolve('data/acs_spectral_transport/2017_attacker_fit.npz').resolve().parent
    frozen = ev.FrozenSeed(seed)
    state = load_frozen_state(seed, registry)
    a0 = state['a0']
    channels = load_new_channels(OUT, seed, neural, a0, state['channel']['representation_fit'])
    j = load_mapper(seed, registry, 'J')
    channels['ref_J'] = AdversarialChannel(16, j['mapper'], j['heads'], None).eval()
    from .run_fit_e import frozen_channel
    j_channel = frozen_channel(seed, registry, state, 'J')['channel']
    maps = joblib.load(OUT / f'seed_{seed}' / 'track_e_maps.joblib')['maps']
    affine, proofs = {}, {}
    for unit in affine_units:
        affine[unit] = affine_for(seed, unit, maps)
        proofs[unit] = verify_affine_2018(seed, unit, affine[unit], state, j_channel)
    for path in frozen.object_files():
        registry.add(path)
    model = TransportModel(frozen.spectral, a0, channels, affine, {})
    conditions = tuple(sorted(model.arms))
    frozen.spectral = model
    ev.SPECTRAL = conditions
    ev.INTERFACES = ('H',) + conditions
    ev.DEV = OUT.resolve()
    return ev, frozen, conditions, proofs


def link_duplicates(seed: int, units):
    """A duplicate release has no audit directory; point it at its canonical audit."""
    for unit in units:
        canonical = ev_local.canonical(seed, unit)
        link = OUT / f'seed_{seed}' / unit
        if canonical != unit and not link.exists():
            link.symlink_to(OUT / f'seed_{seed}' / canonical)


def run(seed: int, units) -> dict:
    limit_threads()
    torch.set_num_threads(1)
    root = OUT / 'exploratory_2017'
    registry = Registry.new()
    neural = [u for u in units if u.startswith('N_')]
    affine_units = [u for u in units if u.startswith('E_') or u in ('leace_J', 'splince_J')]
    link_duplicates(seed, units)
    ev, frozen, conditions, proofs = prepare(seed, registry, neural, affine_units)
    source = resolve(f'results/{T.TRANSPORT_NAME}/seed_{seed}/H')
    dest = root / f'seed_{seed}' / 'H'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.symlink_to(source)
    rel, labels = {}, {}
    for part in ev.FIT_PARTITIONS:
        data = ev._partition(part)
        path = root / f'seed_{seed}' / 'releases_2017' / f'{part}.npz'
        if not path.exists():
            ev.save_releases(path, ev.build_releases(frozen, data['frame']))
        rel[part] = ev.load_releases(path)
        labels[part] = data['labels']
    fits = {}
    for condition in conditions:
        marker = root / f'seed_{seed}' / condition / 'fit_complete.json'
        if marker.exists():
            fits[condition] = read_json(marker)
            continue
        tick = time.perf_counter()
        fits[condition] = ev.fit_unit(root, seed, condition, rel, labels, frozen)
        print('X2017_FIT', seed, condition, round(time.perf_counter() - tick, 1), flush=True)
    scores = T.score_seed(ev, frozen, root, seed, conditions)
    record = {'seed': seed, 'conditions': list(conditions), 'affine_reexpression_max_abs': proofs,
              'fits': fits, 'scores': scores, 'utc': utcnow(),
              'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT'}
    write_json_atomic(root / f'seed_{seed}' / 'transport_record.json', record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--units', nargs='+', required=True)
    args = parser.parse_args()
    run(args.seed, args.units)


if __name__ == '__main__':
    main()
