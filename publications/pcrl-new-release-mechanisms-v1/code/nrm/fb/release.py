"""Family B deployment interface (nrm/deploy.py contract): predict(...) and model_paths(...) for every FB configuration.

    predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64
        From admitted features alone (the encoder's exact 145 names in order; finite, complete one-hot blocks). Reads only the files
        listed by model_paths. A RANDOMIZED configuration (FSC channel) needs out-of-band keys {'role': <stream>, 'row_id': (n,)} and
        returns the keyed one-draw-per-person emission of instance `draw` (0..R-1 are the stored instances); deterministic
        configurations refuse keys and any draw other than 0. predict(config, v, X_role, names_role, draw=0,
        keys={'role': ROLE, 'row_id': row_ids of ROLE}) reproduces RELEASE.npz draw 0 bitwise (checked in the learner selftest).
    model_paths(config_id, version) -> [learner weights, frozen channel file, (randomized only) the private root key file]
        all under .private/<campaign>/runs/FB/MODELS/V<v>/ (write-once files written by nrm.fb.learner.fit_version).
        FB2_* (revision 1, nrm.fb.posthoc): [frozen channel file, (randomized only) root key] under runs/FB/MODELS_FB2/V<v>/ PLUS the
        frozen base's files OUTSIDE this campaign's runs folder (HC_STRONG state / Step-1 learner and admitted stage files in the
        income-headroom mirror; the A1_PO_16 run directory of the occupation-headroom campaign; two governance admissions).
Needs the torch interpreter (the release logits are the torch float32 forward pass of the saved networks).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from . import channel as CH

FAMILY = 'FB'
CODE_MODULES = ('nrm.fb.channel', 'nrm.fb.learner', 'nrm.fb.release', 'nrm.fb.posthoc',
                # FB2 base sources (HC_STRONG income, A1_PO_16 occupation) read by nrm.fb.posthoc.base_probs
                'ihs.fam_h', 'ihs.bases', 'ihs.common', 'ohn.fam_a', 'ohn.common', 'ohn.devrun', 'acs_replication.trees')
TASKS = ('income', 'occupation')
_ROOT_OVERRIDE = None            # selftest only: a temporary model folder in place of runs/FB/MODELS[/<rev>]/V<v>
REVISIONS = ('FB1', 'FB2', 'FB2P')   # FB1 = card FB_FSC_v1 (nrm.fb.learner); FB2 = card v2, FB2P = card v3 (nrm.fb.posthoc)


def revision_of(config_id):
    """The revision tag a family-B configuration id belongs to (ENG1-08: configurations are selected by revision)."""
    from . import learner as L
    if config_id in L.CONFIGS:
        return 'FB1'
    from . import posthoc as PH
    if config_id in PH.REVISION_OF:
        return PH.REVISION_OF[config_id]
    raise ValueError(f'unknown family-B configuration {config_id}')


def model_dir(version, root=None, rev='FB1'):
    """Model folder of one (revision, version). FB1 keeps the layout it was fitted with (runs/FB/MODELS/V<v>, written before
    ENG1-08); every later revision uses runs/FB/MODELS/<rev>/V<v>, with its own write-once root key."""
    if rev not in REVISIONS:
        raise ValueError(f'unknown family-B revision {rev}')
    if root is not None or _ROOT_OVERRIDE is not None:           # selftests: the caller's folder (FB1 flat, later revisions per rev/V)
        base = Path(root if root is not None else _ROOT_OVERRIDE)
        return base if rev == 'FB1' else base / rev / f'V{int(version)}'
    from .. import common as C
    base = C.RUNS / FAMILY / 'MODELS'
    return base / f'V{int(version)}' if rev == 'FB1' else base / rev / f'V{int(version)}'


def channel_file(config_id, version, root=None, rev='FB1'):
    return model_dir(version, root, rev) / f'CHANNEL_{config_id}.json'


def learner_file(learner, version, root=None):
    return model_dir(version, root, 'FB1') / f'LEARNER_{learner.replace(":", "_")}.pt'


def root_key_file(version, root=None, rev='FB1'):
    return model_dir(version, root, rev) / 'RELEASE_ROOT.bin'


def _channel(config_id, version):
    from . import learner as L
    if config_id not in L.CONFIGS or int(version) not in (0, 1):
        raise ValueError(f'unknown family-B configuration {config_id} V{version}')
    rec = json.loads(channel_file(config_id, version).read_text())
    if rec['config_id'] != config_id or int(rec['version']) != int(version):
        raise PermissionError('channel file does not belong to this configuration')
    return rec


def model_paths(config_id, version):
    if revision_of(config_id) in ('FB2', 'FB2P'):         # post-hoc channel on the frozen TWA base (nrm.fb.posthoc)
        from . import posthoc as PH
        return PH.model_paths2(config_id, version)
    rec = _channel(config_id, version)
    paths = [learner_file(rec['learner'], version), channel_file(config_id, version)]
    if rec['randomized']:
        paths.append(root_key_file(version))
    return [str(p) for p in paths]


def _check_inputs(X, feature_names, expected):
    names = tuple(map(str, feature_names))
    if names != tuple(expected):
        raise ValueError('wrong or reordered input schema (the encoder\'s exact names in order are required)')
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(names) or not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
        raise ValueError('finite floating input matrix of the registered width required')
    for b in ('COW', 'SCHL', 'MAR', 'POBP'):
        idx = [i for i, n in enumerate(names) if n.startswith(b + '=')]
        if idx:
            Z = X[:, idx]
            if not (np.all((Z == 0) | (Z == 1)) and np.all(Z.sum(1) == 1)):
                raise ValueError(f'{b}: one-hot block is not complete')
    return X


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    if revision_of(config_id) in ('FB2', 'FB2P'):         # numpy / scikit-learn only (python3)
        from . import posthoc as PH
        return PH.predict2(config_id, version, X, feature_names, draw=draw, keys=keys)
    from . import learner as L
    rec = _channel(config_id, version)
    X = _check_inputs(X, feature_names, rec['feature_names'])
    if not isinstance(draw, (int, np.integer)) or int(draw) < 0:
        raise ValueError('draw must be a non-negative integer')
    if rec['randomized']:
        if keys is None or 'role' not in keys or 'row_id' not in keys or len(keys['row_id']) != len(X):
            raise PermissionError('a randomized FB configuration needs out-of-band per-request draw keys (one row id per row)')
    else:
        if keys is not None:
            raise PermissionError('draw keys are refused by a deterministic FB configuration')
        if int(draw) != 0:
            raise ValueError('a deterministic FB configuration has only draw 0')
    torch = L._torch()
    sd = torch.load(learner_file(rec['learner'], version), weights_only=True)
    nets = {}
    for t in TASKS:
        nets[t] = L.network(X.shape[1], CH.K[t]); nets[t].load_state_dict(sd[t])
    chan = {'mode': rec['mode'], 's': rec['s'], 'cap': rec['cap'], 'geom': rec['geometry']}
    temps = tuple(rec['temperatures'])
    z = L.logits_z(nets, X)
    if rec['mean_vector']:
        return L.emitted(z, chan, temps, None, None, None, 1, mean_vector=True)
    if not rec['randomized']:
        return L.emitted(z, chan, temps, None, None, None, 1)
    key = CH.release_key(root_key_file(version).read_bytes(), config_id, version)
    zs = z / np.array([temps[0]] + [temps[1]] * 4)
    return CH.emit_one(zs, L.scaled(chan, temps), key, keys['role'], np.asarray(keys['row_id']), int(draw))
