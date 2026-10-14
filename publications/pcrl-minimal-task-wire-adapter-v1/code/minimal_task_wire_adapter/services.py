"""Literal recipient wires for every service in the fixed attack roster.

Imports are metadata only. All numerical functions require the active queue.
Services:
  RAW_PROJ        immutable start; factorized income and occupation projections
  RAW_HYBRID      immutable start; projected income, full occupation z16
  ADAPTER         start trunk/heads + saved residual (R, b_res) on occupation
  PRIOR_*_E5      prior campaign checkpoint, its original complete hybrid wire
  PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ
                  prior GLOBAL_UPDATE_GUARD epoch 5 with occupation projected on
                  its own current folded head row space; unchanged compressed income
"""
from __future__ import annotations
import json

OWNPROJ_SCHEMA = 'MTA1-own-head-occupation-projection-v1'


def _queued():
    from .scheduler import assert_active_from_env
    return assert_active_from_env()


def build_ownproj_binding(model, *, checkpoint_sha256, source_seed, start_geometry):
    """Geometry from the checkpoint's actual current occupation head (not the start's)."""
    _queued()
    from .geometry import build_geometry, public_geometry
    from .wires import head_sha
    from .adapter import canonical_sha
    if head_sha(model.head[0]) != start_geometry.binding['head_tensor_sha256'][0]:
        raise PermissionError('prior checkpoint income head differs from the frozen start income head')
    g = build_geometry(model.head[1].weight.detach().cpu().numpy(), model.head[1].bias.detach().cpu().numpy(), model.class_orders[1])
    binding = {'schema': OWNPROJ_SCHEMA, 'checkpoint_sha256': checkpoint_sha256, 'source_seed': source_seed,
               'income_start_geometry_binding_sha256': start_geometry.binding_sha256,
               'occupation_head_tensor_sha256': head_sha(model.head[1]),
               'occupation_geometry': public_geometry(g)}
    binding['binding_sha256'] = canonical_sha(binding)
    return binding


def _ownproj_basis(binding):
    import torch
    from .adapter import canonical_sha
    b = dict(binding); digest = b.pop('binding_sha256')
    if b.get('schema') != OWNPROJ_SCHEMA or canonical_sha(b) != digest:
        raise PermissionError('own-head projection binding hash differs')
    g = binding['occupation_geometry']
    return torch.tensor(g['B'], dtype=torch.float64).reshape(g['rank'], g['source_dimension'])


def service_forward(kind, raw, names, *, model, geometry, adapter_state=None, scale=None, ownproj=None):
    """Return [income dict, occupation dict] with representation/logits/probability/decision/wire arrays."""
    _queued()
    import numpy as np
    import torch
    from .wires import forward_role, head_sha
    from task_conditioned_representation.models import own_wire
    if kind in ('RAW_PROJ', 'RAW_HYBRID', 'PRIOR_HYBRID'):
        return forward_role(model, raw, names, geometry, variant='RAW_PROJ' if kind == 'RAW_PROJ' else 'RAW_HYBRID')
    if tuple(map(str, names)) != tuple(model.names):
        raise ValueError('wrong or reordered raw83 schema')
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 83 or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all() or not len(raw):
        raise ValueError('finite nonempty raw83 matrix required')
    if kind == 'ADAPTER':
        from .adapter import Adapter
        adapter = Adapter(model, geometry, scale, R=adapter_state['R'], b=adapter_state['b'])
    elif kind == 'OWNPROJ':
        if head_sha(model.head[1]) != ownproj['occupation_head_tensor_sha256']:
            raise PermissionError('own-head projection head differs from bound checkpoint head')
        if ownproj['income_start_geometry_binding_sha256'] != geometry.binding_sha256:
            raise PermissionError('own-head projection income geometry differs')
        basis = _ownproj_basis(ownproj)
    else:
        raise PermissionError('undeclared service kind')
    collected = [{}, {}]
    with torch.no_grad():
        for start in range(0, len(raw), 1024):
            x = torch.from_numpy(np.ascontiguousarray(raw[start:start + 1024], dtype=np.float32))
            if kind == 'ADAPTER':
                hbar, q0, _ = adapter.frozen_features(x)
                zi = (model.encode(0, x).to(torch.float64) @ geometry.bases[0].T) @ geometry.bases[0]
                wires = [own_wire(zi, model.head[0], model.class_orders[0]), adapter.wire(hbar, q0)]
            else:
                zi = (model.encode(0, x).to(torch.float64) @ geometry.bases[0].T) @ geometry.bases[0]
                zo = (model.encode(1, x).to(torch.float64) @ basis.T) @ basis
                wires = [own_wire(zi, model.head[0], model.class_orders[0]), own_wire(zo, model.head[1], model.class_orders[1])]
            for i, w in enumerate(wires):
                for target, key in (('representation', 'representation'), ('centered_logits', 'centered_logits'),
                                    ('probability', 'probabilities'), ('decision', 'decision'), ('wire', 'full')):
                    collected[i].setdefault(target, []).append(w[key].detach().cpu().numpy())
    return [{k: np.concatenate(v, 0) for k, v in c.items()} for c in collected]


def load_adapter_checkpoint(path, expected_sha256, *, seed, source_checkpoint_sha256, adapter_binding_sha256):
    import torch
    import hashlib
    from pathlib import Path
    with Path(path).open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected_sha256:
            raise PermissionError('adapter checkpoint hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != 'MTA1-adapter-checkpoint-v1' or payload.get('seed') != seed
            or payload.get('source_checkpoint_sha256') != source_checkpoint_sha256
            or payload.get('adapter_binding_sha256') != adapter_binding_sha256):
        raise PermissionError('adapter checkpoint identity differs')
    return payload
