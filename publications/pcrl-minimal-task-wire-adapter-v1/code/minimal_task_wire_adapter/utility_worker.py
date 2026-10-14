"""One FIT2 (inherited FIT fold 2) utility pass after the committed TRAINING_COMPLETE_LOCK.

Scores every saved adapter checkpoint (5 arms x 5 epochs x 2 seeds) and the
reference/prior-guard diagnostic services on the original 24 POINT clauses
(both tasks x both references U0/Ucal x accuracy/NLL/Brier x both seeds), and
applies the frozen per-arm latest-common-epoch>0 selection rule. FIT2 is an
exposed development selection role, not pristine validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
TASKS = ('income', 'occupation')
REFS = ('U0', 'Ucal')


def task_vectors(y, p, order):
    import numpy as np
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64); order = np.asarray(order)
    if p.shape != (len(y), len(order)) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError('probability convention differs')
    target = y[:, None] == order
    return {'accuracy': (order[p.argmax(1)] == y).astype(np.float64),
            'nll': -(target * np.log(np.maximum(p, 1e-12))).sum(1),
            'brier': ((p - target) ** 2).sum(1)}


def summary(y, p, order):
    import numpy as np
    v = task_vectors(y, p, order); d = np.asarray(order)[np.asarray(p).argmax(1)]
    present = [bool(np.any(y == k)) for k in order]
    return {**{k: float(a.mean()) for k, a in v.items()},
            'per_class_recall': [float((d[y == k] == k).mean()) if h else None for k, h in zip(order, present)],
            'class_counts': [int((y == k).sum()) for k in order], 'confidence': float(np.asarray(p).max(1).mean()),
            'confidence_by_true_class': [float(np.asarray(p)[y == k].max(1).mean()) if h else None for k, h in zip(order, present)]}


def clauses(y, probs, refs, seed):
    """12 seed clauses: task x reference x metric (accuracy loss, NLL excess, summed Brier excess)."""
    out = []
    for i, task in enumerate(TASKS):
        order = list(range(2 if i == 0 else 6))
        own = summary(y[task], probs[i], order)
        for ref in REFS:
            r = summary(y[task], refs[f'{ref}_{i + 1}'], order)
            diff = {'accuracy_loss': r['accuracy'] - own['accuracy'], 'nll_excess': own['nll'] - r['nll'],
                    'brier_excess': own['brier'] - r['brier']}
            for metric, cap in CAPS.items():
                out.append({'seed': seed, 'task': task, 'reference': ref, 'metric': metric, 'value': diff[metric],
                            'cap': cap, 'headroom': cap - diff[metric], 'point_pass': bool(diff[metric] <= cap)})
    return out


def _lock(spec, verify, key, status):
    p = verify(spec[key], spec[key + '_sha256'])
    root = Path(spec['repo_root'])
    rel = p.resolve().relative_to(root.resolve()).as_posix()
    committed = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=root, capture_output=True, check=False)
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != spec[key + '_sha256']:
        raise PermissionError(f'{key} must be committed at HEAD before FIT2 opens')
    lock = json.loads(p.read_text())
    if lock.get('status') != status:
        raise PermissionError(f'{key} status differs')
    return lock


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, guard = queue_config(spec_path, 'verification')
    lock = _lock(spec, verify, 'training_complete_lock', 'BANK_TRAINING_FROZEN')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .wires import prepare_geometry
    from .services import service_forward, load_adapter_checkpoint
    from .recipe import ARMS, RECIPE
    v = spec['validation']
    with np.load(verify(v['features']['path'], v['features']['sha256']), allow_pickle=False) as z:
        feat = {k: z[k].copy() for k in z.files}
    y = {}
    for task in TASKS:
        with np.load(verify(v[task]['path'], v[task]['sha256']), allow_pickle=False) as z:
            if not np.array_equal(z['row_id'], feat['row_id']):
                raise PermissionError('FIT2 pairing differs')
            y[task] = z[task].copy()
    X, names = feat['X'], feat['feature_names']
    rows = []; table = {}; changes = {}; services = {}
    for seed in (0, 1):
        guard()
        with np.load(verify(v[f'references_s{seed}']['path'], v[f'references_s{seed}']['sha256']), allow_pickle=False) as z:
            refs = {k: z[k].copy() for k in z.files}
        if not np.array_equal(refs['row_id'], feat['row_id']):
            raise PermissionError('reference pairing differs')
        g = spec['geometry'][str(seed)]
        geometry = prepare_geometry(json.loads(verify(g['path'], g['sha256']).read_text()), expected_sha256=g['binding_sha256'], source_seed=seed)
        c = spec['checkpoints'][str(seed)]
        model, _ = load_checkpoint(verify(c['path'], c['sha256']), c['sha256'])
        ab = spec['adapter_bindings'][str(seed)]
        binding = json.loads(verify(ab['path'], ab['sha256']).read_text())
        start = service_forward('RAW_PROJ', X, names, model=model, geometry=geometry)
        start_occ = start[1]

        def record(name, epoch, wires, kind):
            cl = clauses(y, [wires[0]['probability'], wires[1]['probability']], refs, seed)
            ch = {k: float(np.max(np.abs(wires[1][k].astype(np.float64) - start_occ[k].astype(np.float64))))
                  for k in ('representation', 'centered_logits', 'probability')}
            ch['decision_changes'] = int((wires[1]['decision'] != start_occ['decision']).sum())
            ch['income_wire_identical_to_RAW_PROJ'] = bool(np.array_equal(wires[0]['wire'], start[0]['wire']))
            key = f'{seed}|{name}|{epoch}'
            table[key] = {'clauses': cl, 'all_point_pass': all(x['point_pass'] for x in cl),
                          'metrics': {task: summary(y[task], wires[i]['probability'], list(range(2 if i == 0 else 6))) for i, task in enumerate(TASKS)},
                          'occupation_change_vs_RAW_PROJ': ch, 'kind': kind}
            changes[key] = ch
            for x in cl:
                rows.append({'seed': seed, 'service': name, 'epoch': epoch, **x})
        record('RAW_PROJ', 0, start, 'reference')
        record('RAW_HYBRID', 0, service_forward('RAW_HYBRID', X, names, model=model, geometry=geometry), 'reference_diagnostic')
        for label, item in spec['prior_checkpoints'][str(seed)].items():
            pm, _ = load_checkpoint(verify(item['path'], item['sha256']), item['sha256'])
            record(f'PRIOR_{label}_E5', 5, service_forward('PRIOR_HYBRID', X, names, model=pm, geometry=geometry), 'prior_diagnostic')
            if label == 'GLOBAL_UPDATE_GUARD':
                ob = spec['ownproj'][str(seed)]
                own = json.loads(verify(ob['path'], ob['sha256']).read_text())
                record('PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ', 5, service_forward('OWNPROJ', X, names, model=pm, geometry=geometry, ownproj=own), 'prior_diagnostic')
        for arm in ARMS:
            for epoch in RECIPE['checkpoint_epochs']:
                ck = spec['checkpoint_paths'][str(seed)][arm][str(epoch)]
                if ck['sha256'] != lock['checkpoints'][str(seed)][arm][str(epoch)]['sha256']:
                    raise PermissionError('private checkpoint binding differs from committed TRAINING_COMPLETE_LOCK')
                payload = load_adapter_checkpoint(verify(ck['path'], ck['sha256']), ck['sha256'], seed=seed,
                                                  source_checkpoint_sha256=c['sha256'], adapter_binding_sha256=ab['sha256'])
                if payload['arm'] != arm or payload['epoch'] != epoch:
                    raise PermissionError('adapter checkpoint identity differs')
                w = service_forward('ADAPTER', X, names, model=model, geometry=geometry, scale=float(binding['scale']),
                                    adapter_state={'R': payload['R'], 'b': payload['b_res']})
                if epoch == 0 and not all(np.array_equal(w[i]['wire'], start[i]['wire']) for i in range(2)):
                    raise AssertionError('epoch-0 adapter is not the RAW_PROJ zero-residual witness')
                record(arm, epoch, w, 'adapter_arm')
    selection = {}
    for arm in ARMS:
        feasible = [e for e in RECIPE['checkpoint_epochs'] if e > 0 and all(table[f'{s}|{arm}|{e}']['all_point_pass'] for s in (0, 1))]
        chosen = max(feasible) if feasible else None
        selection[arm] = {'feasible_common_epochs_gt0': feasible, 'selected_epoch': chosen,
                          'status': 'POINT_FEASIBLE_NOMINEE' if chosen else 'NO_POINT_FEASIBLE_NOMINEE',
                          'attacked_epoch': chosen if chosen else 10,
                          'attacked_status': 'SELECTED_POINT_ELIGIBLE' if chosen else 'FIXED_EPOCH10_DIAGNOSTIC_INELIGIBLE',
                          'fallback_epoch': 0,
                          'failed_clauses_at_epoch10': [dict(x) for s in (0, 1) for x in table[f'{s}|{arm}|10']['clauses'] if not x['point_pass']]}
    ref_ok = all(table[f'{s}|RAW_PROJ|0']['all_point_pass'] for s in (0, 1))
    result = {'schema': 'MTA1-fit2-utility-v1', 'queue_unit_id': active['unit_id'],
              'training_complete_lock_sha256': spec['training_complete_lock_sha256'], 'caps': CAPS,
              'clauses_per_service': 24, 'rows': int(len(X)), 'groups': int(len(np.unique(feat['unit']))),
              'selection': selection, 'RAW_PROJ_point_eligible': ref_ok, 'table': table,
              'INNER_opened': False, 'ATT_opened': False, 'FIT2_opened_once': True,
              'role_status': 'exposed development selection role, not pristine validation'}
    out = Path(spec['output'])
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    with Path(spec['rows_output']).open('x') as stream:
        json.dump(rows, stream, sort_keys=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
