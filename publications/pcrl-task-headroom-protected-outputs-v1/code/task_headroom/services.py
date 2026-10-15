"""Resolve a registered service to its actual transmitted outputs on one role.

Every service, new or historical, is released under ONE contract: each
recipient receives its own full-precision probability vector and decision.
Returned per recipient: {'probability': (n, K) float64, 'decision': (n,) int64}.
Nothing else (leaf ids, raw scores, representations, tokens, cells, teacher
outputs) leaves this module.
"""
from __future__ import annotations
import json

import numpy as np

from . import trees

TASKS = ('income', 'occupation')


def load_teacher(item, verify):
    return trees.TreeTeacher(json.loads(verify(item['path'], item['sha256']).read_text()))


def load_theta(item, verify):
    with np.load(verify(item['path'], item['sha256']), allow_pickle=False) as z:
        return [z['theta_income'].copy(), z['theta_occupation'].copy()]


def tree_outputs(service, X, names, verify):
    teachers = [load_teacher(service['teachers'][t], verify) for t in TASKS]
    theta = load_theta(service['theta'], verify) if service.get('theta') else [None, None]
    out = []
    for task, teacher, th in zip(TASKS, teachers, theta):
        raw = teacher.raw_from_leaves(teacher.route(X, names), th)
        out.append(trees.release(task, raw))
    return out


def neq_outputs(service, X, names, verify):
    import torch
    from . import neural
    payload = torch.load(verify(service['model']['path'], service['model']['sha256']), map_location='cpu', weights_only=True)
    if tuple(payload['feature_names']) != tuple(map(str, names)):
        raise ValueError('neural reference schema differs')
    out = []
    for task, k in (('income', 2), ('occupation', 6)):
        m = neural.network(k); m.load_state_dict(payload[task]); m.eval()
        p = neural.probabilities(m, X)
        out.append({'probability': p, 'decision': p.argmax(1).astype(np.int64)})
    return out


def raw_proj_outputs(service, X, names, verify, seed):
    from task_conditioned_representation.models import load_checkpoint
    from .rawproj.wires import prepare_geometry
    from .rawproj.services import service_forward
    g = service['geometry']
    geometry = prepare_geometry(json.loads(verify(g['path'], g['sha256']).read_text()), expected_sha256=g['binding_sha256'], source_seed=seed)
    c = service['checkpoint']; model, _ = load_checkpoint(verify(c['path'], c['sha256']), c['sha256'])
    ab = service['adapter_binding']; binding = json.loads(verify(ab['path'], ab['sha256']).read_text())
    w = service_forward({'kind': 'RAW_PROJ'}, X, names, model=model, geometry=geometry, verify=verify, seed=seed,
                        source_checkpoint_sha256=c['sha256'], adapter_binding_sha256=ab['sha256'], binding_scales=binding['scales'])
    return [{'probability': np.asarray(w[i]['probability'], dtype=np.float64), 'decision': np.asarray(w[i]['decision'], dtype=np.int64)} for i in range(2)]


def hist_outputs(service, row_id, verify):
    out = []
    for i, part in enumerate(service['parts']):
        with np.load(verify(part['path'], part['sha256']), allow_pickle=False) as z:
            rid = z['row_id']; p = np.asarray(z[part['probability_field']], dtype=np.float64); d = np.asarray(z[part['decision_field']], dtype=np.int64)
        pos = {int(r): j for j, r in enumerate(rid)}
        ix = np.asarray([pos[int(r)] for r in row_id])
        p, d = p[ix], d[ix]
        if not np.array_equal(p.argmax(1), d) or not np.allclose(p.sum(1), 1, atol=1e-9) or p.shape[1] != (2, 6)[i]:
            raise ValueError('historical release is not a faithful probability+decision release')
        out.append({'probability': p, 'decision': d})
    return out


def outputs(service, role, verify, seed):
    """role: dict with row_id, X, feature_names."""
    kind = service['kind']
    if kind == 'TREE':
        return tree_outputs(service, role['X'], role['feature_names'], verify)
    if kind == 'NEQ':
        return neq_outputs(service, role['X'], role['feature_names'], verify)
    if kind == 'RAW_PROJ':
        return raw_proj_outputs(service, role['X'], role['feature_names'], verify, seed)
    if kind == 'HIST':
        return hist_outputs(service, role['row_id'], verify)
    raise ValueError('unregistered service kind')


def wire(rec):
    """Actual transmitted wire of one recipient: probabilities then decision."""
    return np.column_stack((rec['probability'], rec['decision'].astype(np.float64)))
