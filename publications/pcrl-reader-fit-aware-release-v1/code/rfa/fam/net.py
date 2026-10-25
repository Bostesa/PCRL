"""Deployment forward pass of family FAM (numpy only). `np_forward`, `layers_of` and `release_weights` are verbatim copies of
`sar.fb.pfd` (SAR1, publication c8aaa2df4), so that a deployment package of this campaign loads no SAR1 module."""
from __future__ import annotations

import numpy as np

from .. import famkit as FK

FAMILY = 'FAM'                     # runs/FAM/<config>/V<v>; rfa.deploy.family_module('FAM') -> rfa.fam
TASKS = ('income', 'occupation')


def np_forward(layers, X):
    """Probability vectors of one U0-architecture network from its saved float32 weights, in float64 (the release law)."""
    h = np.asarray(X, dtype=np.float64)
    for j, (w, b) in enumerate(layers):
        h = h @ np.asarray(w, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
        if j < 2:
            h = np.maximum(h, 0.0)
    z = h - h.mean(1, keepdims=True)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def layers_of(state):
    """[(W, b)] float32 numpy copies of the four Linear layers of a U0-architecture state dict (Sequential indices 0, 2, 4, 5)."""
    return [(np.asarray(state[f'{i}.weight'].detach().cpu().numpy() if hasattr(state[f'{i}.weight'], 'detach') else state[f'{i}.weight'],
                        dtype=np.float32).copy(),
             np.asarray(state[f'{i}.bias'].detach().cpu().numpy() if hasattr(state[f'{i}.bias'], 'detach') else state[f'{i}.bias'],
                        dtype=np.float32).copy()) for i in (0, 2, 4, 5)]


def release_weights(W, X):
    return np_forward(W['income'], X), np_forward(W['occupation'], X)


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (rfa.deploy contract); deterministic; numpy only (python3)."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    if P.get('op', 'net') != 'net':
        raise ValueError(f'unknown FAM payload op {P.get("op")!r}')
    X, names = FK.check_X(X, feature_names, P)
    pi, po = release_weights(P['weights'], X)
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)
