"""Read-only recovery of the frozen 2018 objects this study starts from.

Nothing historical is refitted. The frozen `A0` auxiliary channel is recovered from
its saved checkpoint and asserted **bit-exact** against the stored 2018 release wire
before it is used as the common starting point, so a reconstructed channel can never
silently inherit the identity of the released one.

Label boundaries enforced here:

* the protected labels (`SEX`, `RAC1P`, `public_coverage`) come from the four-column
  read that the 2018 representation stage itself used -- no residence, no commute;
* the two **authorised** source tasks (`income_binary`, `civilian_at_work`) are the
  only entries ever taken from the full-cohort label helper, and the reserved
  residence/commute keys are dropped before the dictionary is returned, so a later
  typo cannot reach them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.pcrl_nonlinear_rank_v1.inputs import (FIXED_NAME, Registry, arrays, array_hash,
                                                       load_representation_labels, resolve,
                                                       sha_file, write_json, read_json)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pcrl_direct_adversarial_v1'

POOLS = ('representation_fit', 'source_validation', 'downstream_fit', 'downstream_validation',
         'attacker_fit', 'attacker_validation', 'test')

H_A_WIDTH = 4
H_B_WIDTH = 2
A0_WIDTH = 16
PCA_WIDTH = 32

SOURCE_TASKS = ('income_binary', 'civilian_at_work')
RESERVED_TASKS = ('same_residence', 'commute_over20')

# The five trainable protected roles, in the fixed order used for every seed formula.
ROLE_ORDER = ('A/public_coverage', 'A/SEX', 'A/RAC1P', 'AB/SEX', 'AB/RAC1P')
ROLE_CLASSES = {'A/public_coverage': 2, 'A/SEX': 2, 'A/RAC1P': 9, 'AB/SEX': 2, 'AB/RAC1P': 9}
LOCAL_ROLES = tuple(r for r in ROLE_ORDER if r.startswith('A/'))
COALITION_ROLES = tuple(r for r in ROLE_ORDER if r.startswith('AB/'))

# Internal household folds inside `representation_fit`. Fixed before any fit.
FOLD_EDGES = ((0.00, 0.25, 'p0_fit'), (0.25, 0.80, 'mapper_fit'), (0.80, 1.00, 'monitor'))
FOLD_SALT = 'pcrl_direct_adversarial_v1/internal_household_fold/v1'


def fixed_path(seed: int, *parts) -> str:
    return str(Path('results') / FIXED_NAME / f'seed_{seed}' / Path(*parts))


# ------------------------------------------------------------------ the frozen A0 channel
def load_a0_mapper(seed: int, registry: Registry):
    """The frozen `A0` mapper and its input standardiser, exactly as trained in 2018."""
    path = registry.resolve(fixed_path(seed, 'training/A0/final.pt'))
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state']
    mapper = nn.Sequential(nn.Linear(PCA_WIDTH, 64), nn.ReLU(), nn.Linear(64, A0_WIDTH))
    mapper.load_state_dict({k[len('branch.mapper.'):]: v.detach().clone()
                            for k, v in state.items() if k.startswith('branch.mapper.')})
    heads = {name: nn.Linear(A0_WIDTH, 1) for name in SOURCE_TASKS}
    for name, head in heads.items():
        head.load_state_dict({'weight': state[f'branch.heads.{name}.weight'].detach().clone(),
                              'bias': state[f'branch.heads.{name}.bias'].detach().clone()})
    return {'mapper': mapper.eval(), 'heads': heads,
            'input_mean': state['input_mean'].numpy().copy(),
            'input_scale': state['input_scale'].numpy().copy(),
            'checkpoint': str(path), 'checkpoint_sha256': sha_file(path)}


def standardize(pca: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    """The frozen 2018 convention: float64 arithmetic, float32 tensor."""
    return torch.tensor((np.asarray(pca, np.float64) - mean) / scale, dtype=torch.float32)


def load_frozen_state(seed: int, registry: Registry) -> dict:
    """Pools, anchors, PCA coordinates, the A0 channel and its bit-exactness proof."""
    rows = arrays(registry.resolve(fixed_path(seed, 'split_rows.npz')))
    pca = arrays(registry.resolve(fixed_path(seed, 'pca.npz')))
    anchors = arrays(registry.resolve(fixed_path(seed, 'anchors.npz')))
    a0 = load_a0_mapper(seed, registry)
    released = arrays(registry.resolve(fixed_path(seed, 'training/A0/releases.npz')))

    channel, parity = {}, {}
    with torch.no_grad():
        for pool in POOLS:
            x = standardize(pca[pool], a0['input_mean'], a0['input_scale'])
            value = a0['mapper'](x).numpy().astype(np.float64)
            stored = released[f'wire/A/{pool}'][:, H_A_WIDTH:]
            if stored.shape[1] != A0_WIDTH:
                raise AssertionError(f'A0 auxiliary width is {stored.shape[1]}, not {A0_WIDTH}')
            parity[pool] = {'bitwise_identical': bool(np.array_equal(value, stored)),
                            'max_abs_difference': float(np.abs(value - stored).max()),
                            'rows': int(len(value))}
            if not parity[pool]['bitwise_identical']:
                raise AssertionError(
                    f'reconstructed A0 channel differs from the released wire at {pool}; a '
                    'reconstruction that is not bit-exact is a NEW fit and may not inherit the '
                    'A0 identity')
            channel[pool] = value
    return {'seed': seed, 'rows': rows, 'pca': pca, 'anchors': anchors, 'a0': a0,
            'channel': channel, 'a0_parity': parity,
            'a0_release_sha256': sha_file(resolve(fixed_path(seed, 'training/A0/releases.npz')))}


# ------------------------------------------------------------------ labels and folds
def representation_labels(seed: int, registry: Registry) -> dict:
    """Protected labels (four-column read) plus the two authorised source tasks."""
    protected, serials = load_representation_labels(seed, registry)
    tasks = authorised_source_labels(seed, len(serials))
    return {'protected': protected, 'source': tasks, 'serials': serials}


def authorised_source_labels(seed: int, n: int) -> dict:
    """`income_binary` and `civilian_at_work` on the representation-fitting rows.

    The reserved residence/commute keys are deleted from the helper's output before it
    is returned, so nothing downstream can index them even by mistake.
    """
    from experiments.run_acs_coalition import all_labels
    from experiments.run_acs_residual_spectral import load_labels

    frame, pools, _labels, _weights = load_labels(Path('/Users/nathansamson/PCRL'), seed)
    every = dict(all_labels(frame.iloc[pools['representation_fit']], []))
    for reserved in RESERVED_TASKS:
        every.pop(reserved, None)
    if any(r in every for r in RESERVED_TASKS):
        raise AssertionError('a reserved task label survived the drop')
    out = {}
    for task in SOURCE_TASKS:
        y = np.asarray(every[task]).astype(np.int64)
        if y.shape != (n,):
            raise ValueError(f'{task} has {y.shape}, expected {(n,)}')
        out[task] = y
    return out


def household_folds(serials: np.ndarray) -> dict:
    """Deterministic household-level split of `representation_fit` into internal folds.

    Households never straddle a fold, so an attacker fitted on `mapper_fit` is never
    monitored on a relative of its own training rows.
    """
    unique = np.unique(np.asarray(serials).astype(str))
    draw = {}
    for household in unique:
        digest = hashlib.sha256(f'{FOLD_SALT}|{household}'.encode()).hexdigest()
        draw[household] = int(digest[:8], 16) / 2 ** 32
    value = np.array([draw[s] for s in np.asarray(serials).astype(str)])
    folds = {}
    for low, high, name in FOLD_EDGES:
        folds[name] = np.flatnonzero((value >= low) & (value < high)) if high < 1.0 else \
            np.flatnonzero((value >= low) & (value <= 1.0))
    covered = np.concatenate([folds[n] for _, _, n in FOLD_EDGES])
    if len(np.unique(covered)) != len(value):
        raise AssertionError('internal household folds do not partition representation_fit')
    return folds


# ------------------------------------------------------------------ release assembly
def build_wires(channel: dict, anchors: dict) -> dict:
    """`A = [H_A, Z]`, `B = H_B`, `AB = [A, B]`, float64, anchors bitwise preserved."""
    cache = {}
    for pool in POOLS:
        ha = anchors[f'{pool}/A']
        hb = anchors[f'{pool}/B']
        z = np.asarray(channel[pool], np.float64)
        wa = np.column_stack((ha, z))
        wab = np.column_stack((wa, hb))
        if wa.dtype != np.float64 or wab.dtype != np.float64:
            raise AssertionError('release wires must be float64')
        if not (np.array_equal(wa[:, :H_A_WIDTH], ha) and np.array_equal(wab[:, -H_B_WIDTH:], hb)):
            raise AssertionError(f'anchor parity failed on {pool}')
        cache[f'wire/A/{pool}'] = wa
        cache[f'wire/B/{pool}'] = hb.copy()
        cache[f'wire/AB/{pool}'] = wab
    return cache


def save_release(path: Path, cache: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)
    return {'path': str(path), 'sha256': sha_file(path),
            'A_width': int(cache['wire/A/test'].shape[1]),
            'AB_width': int(cache['wire/AB/test'].shape[1])}


__all__ = ['OUT', 'POOLS', 'ROLE_ORDER', 'ROLE_CLASSES', 'LOCAL_ROLES', 'COALITION_ROLES',
           'SOURCE_TASKS', 'RESERVED_TASKS', 'A0_WIDTH', 'H_A_WIDTH', 'H_B_WIDTH',
           'load_frozen_state', 'representation_labels', 'household_folds', 'build_wires',
           'save_release', 'standardize', 'load_a0_mapper', 'Registry', 'write_json',
           'read_json', 'sha_file', 'array_hash', 'arrays', 'resolve']
