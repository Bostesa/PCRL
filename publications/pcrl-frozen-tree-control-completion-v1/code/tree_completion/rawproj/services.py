"""Literal recipient wires for every service in the fixed attack roster.

Imports are metadata only. All numerical functions require the active queue.
Services are built from the admitted start (trunks, folded heads, bases) plus
per-recipient residual states:
  RAW_PROJ      both residuals exactly zero (inherited factorized projections)
  arm services  both residuals from one saved TRA1 checkpoint (zero where never active)
  O_PREVIOUS    occupation residual of the previous campaign's J_TRUST epoch-10
                checkpoint (exact manifest); income residual zero
  HYB_JI_RO     J_BOTH audited income residual + zero occupation residual
  HYB_RI_JO     zero income residual + J_BOTH audited occupation residual
"""
from __future__ import annotations
import hashlib
from pathlib import Path

CHECKPOINT_SCHEMA = 'TRA1-adapter-checkpoint-v1'
PREVIOUS_SCHEMA = 'MTA1-adapter-checkpoint-v1'


def _queued():
    from tree_completion.scheduler import assert_active_from_env
    return assert_active_from_env()


def _file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_checkpoint2(path, expected_sha256, *, seed, source_checkpoint_sha256, adapter_binding_sha256):
    import torch
    if _file_sha(path) != expected_sha256:
        raise PermissionError('adapter checkpoint hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('seed') != seed
            or payload.get('source_checkpoint_sha256') != source_checkpoint_sha256
            or payload.get('adapter_binding_sha256') != adapter_binding_sha256):
        raise PermissionError('adapter checkpoint identity differs')
    return payload


def load_previous(path, expected_sha256, *, seed, source_checkpoint_sha256):
    import torch
    if _file_sha(path) != expected_sha256:
        raise PermissionError('previous adapter checkpoint hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != PREVIOUS_SCHEMA or payload.get('seed') != seed or payload.get('arm') != 'J_TRUST'
            or payload.get('epoch') != 10 or payload.get('source_checkpoint_sha256') != source_checkpoint_sha256):
        raise PermissionError('previous J_TRUST epoch-10 identity differs')
    return payload


def resolve(service, *, verify, seed, source_checkpoint_sha256, adapter_binding_sha256, binding_scales):
    """Return (states[2], scales[2], provenance) for one declared service description."""
    states = [None, None]; scales = [float(binding_scales[0]), float(binding_scales[1])]; prov = {}
    kind = service['kind']
    if kind == 'RAW_PROJ':
        return states, scales, {'kind': kind}
    if kind != 'ADAPTER2':
        raise PermissionError('undeclared service kind')
    cache = {}
    for i, name in enumerate(('income', 'occupation')):
        src = service.get(name)
        if src is None:
            prov[name] = 'zero residual (RAW_PROJ)'
            continue
        c = src['checkpoint']
        if src['source'] == 'TRA1':
            if c['sha256'] not in cache:
                cache[c['sha256']] = load_checkpoint2(verify(c['path'], c['sha256']), c['sha256'], seed=seed,
                                                      source_checkpoint_sha256=source_checkpoint_sha256,
                                                      adapter_binding_sha256=adapter_binding_sha256)
            p = cache[c['sha256']]
            if p['arm'] != src['arm'] or p['label'] != src['label']:
                raise PermissionError('attacked checkpoint identity differs from the declared service')
            states[i] = {'R': p['states'][i]['R'], 'b': p['states'][i]['b']}
            if float(p['scales'][i]) != scales[i]:
                raise PermissionError('checkpoint scale differs from binding')
        elif src['source'] == 'MTA1_PREVIOUS' and i == 1:
            p = load_previous(verify(c['path'], c['sha256']), c['sha256'], seed=seed, source_checkpoint_sha256=source_checkpoint_sha256)
            states[i] = {'R': p['R'], 'b': p['b_res']}
            if float(p['scale']) != scales[i]:
                raise PermissionError('previous occupation scale differs from the admitted scale')
        else:
            raise PermissionError('undeclared residual source')
        prov[name] = {k: v for k, v in src.items() if k != 'checkpoint'} | {'checkpoint_sha256': c['sha256']}
    return states, scales, dict(prov, kind=kind)


def service_forward(service, raw, names, *, model, geometry, verify, seed, source_checkpoint_sha256, adapter_binding_sha256, binding_scales):
    """Return [income dict, occupation dict] of representation/centered_logits/probability/decision/wire arrays."""
    _queued()
    from .wires import forward_role
    from .adapter import forward_raw
    states, scales, prov = resolve(service, verify=verify, seed=seed, source_checkpoint_sha256=source_checkpoint_sha256,
                                   adapter_binding_sha256=adapter_binding_sha256, binding_scales=binding_scales)
    if service['kind'] == 'RAW_PROJ':
        return forward_role(model, raw, names, geometry, variant='RAW_PROJ')
    return forward_raw(model, geometry, scales, states, raw, names)


def describe(service):
    return {k: v for k, v in service.items() if k not in ('income', 'occupation')} | {
        n: (None if service.get(n) is None else {k: v for k, v in service[n].items() if k != 'checkpoint'} | {'checkpoint_sha256': service[n]['checkpoint']['sha256']})
        for n in ('income', 'occupation') if service['kind'] == 'ADAPTER2'}
