"""Frozen release objects (no fitting) and exact release construction for any rows.

Objects are the task-directed study's archived fits, restored from the pinned
private archive with per-file hash verification. Solutions are additionally
checked against their accepted-map receipts. Releases are built by the
predecessor's unchanged ``build_release``.
"""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.baselines import AffineEraser
from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from experiments.pcrl_task_directed_release_v1.mechanisms import build_release
from .common import PANEL, RESTORED, read_json, sha

MAPS = {'Q': 'T0_L_0.01_a17', 'D17': 'T0_U_unconstrained_a17', 'D33': 'T0_U_unconstrained_a33',
        'RR75': 'T0_U_unconstrained_a17', 'W75': 'T0_U_unconstrained_a17'}
ACTIONS = {'D33': 33}
ERASERS = {'E': ('leace_supervised', 'mechanism40'), 'S': ('splince_supervised', 'mechanism40')}
ENCODED_KEYS = ('p', 'b', 'r', 'global_offsets')


def anchor_dir(anchor):
    return RESTORED/f'anchor_{anchor}'


def object_hashes(anchor):
    """Hashes of every frozen object used by the panel for one anchor."""
    base = anchor_dir(anchor)
    paths = [base/'encoder/encoder.joblib']
    for name in sorted(set(MAPS.values())):
        paths += [base/'maps'/name/'solution.joblib', base/'maps'/name/'ACCEPTED.json']
    for method, scope in ERASERS.values():
        paths += [base/'baseline_supplement'/scope/method/'map.npz',
                  base/'baseline_supplement'/scope/method/'diagnostics.json',
                  base/'baseline_supplement'/scope/'FITTED.json']
    return {str(p.relative_to(RESTORED)): sha(p) for p in sorted(set(paths))}


class FrozenReleases:
    def __init__(self, anchor):
        base = anchor_dir(anchor)
        self.anchor = anchor
        self.encoder = joblib.load(base/'encoder/encoder.joblib')
        fitted = read_json(base/'baseline_supplement/mechanism40/FITTED.json')
        if fitted['encoder_sha256'] != sha(base/'encoder/encoder.joblib'):
            raise ValueError('Encoder differs from the supplement receipt')
        self.maps = {}
        for name in sorted(set(MAPS.values())):
            receipt = read_json(base/'maps'/name/'ACCEPTED.json')
            path = base/'maps'/name/'solution.joblib'
            if receipt['sha256'] != sha(path) or receipt['configuration'] != name or receipt['anchor'] != anchor:
                raise ValueError(f'Accepted map receipt mismatch: {name}')
            solution = joblib.load(path)
            if not solution.get('feasible') or solution.get('configuration') != name:
                raise ValueError(f'Frozen map infeasible or mislabeled: {name}')
            self.maps[name] = solution
        self.erasers = {PANEL[k]: AffineEraser.load(base/'baseline_supplement'/scope/method)
                        for k, (method, scope) in ERASERS.items()}

    def encode(self, x, ha):
        e = self.encoder.encode(RuntimeInputs(x, ha))
        return {'p': e['p'], 'b': e['b'], 'r': e['r'], 'codes': e['codes'], 'actions': e['actions'],
                'global_offsets': e['global_offsets']}

    def release(self, short, pools, encoded):
        """pools: {pool: {'ha','J','x'}}; encoded: {pool: encode(...)} for the same rows."""
        name = PANEL[short]
        mechanism = self.maps.get(MAPS.get(short)) if short in MAPS else None
        ctx = {'anchor': self.anchor, 'pools': pools}
        return build_release(ctx, self.encoder, encoded, name, mechanism=mechanism,
                             erasers=self.erasers, actions=ACTIONS.get(short, 17))


def release_fingerprint(release):
    """Stable digest of the released view (token law, aux, fixed decoder) per pool."""
    from .common import array_hash
    out = {}
    for pool, arm in release.items():
        out[pool] = {k: array_hash(np.asarray(v)) for k, v in arm.items() if v is not None}
    return out
