"""One FIT2 (inherited FIT fold 2) utility pass after the committed TRAINING_COMPLETE_LOCK.

Scores every saved checkpoint of the six-arm bank (both seeds), RAW_PROJ and
O_PREVIOUS on the original 24 POINT clauses (both tasks x U0/Ucal x
accuracy/NLL/Brier x both seeds) and applies the frozen per-arm latest common
positive checkpoint rule (sequential arms: second-phase checkpoints only, both
recipients qualifying together). FIT2 is an exposed development selection role,
not pristine validation.
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
            'class_counts': [int((y == k).sum()) for k in order], 'confidence': float(np.asarray(p).max(1).mean())}


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


def select(table, arms, selectable, fallback):
    selection = {}
    for arm in arms:
        feasible = [lab for lab in selectable(arm) if all(table[f'{s}|{arm}|{lab}']['all_point_pass'] for s in (0, 1))]
        chosen = max(feasible, key=lambda lab: int(lab.split('E')[-1])) if feasible else None
        selection[arm] = {'feasible_common_labels': feasible, 'selected_label': chosen,
                          'status': 'POINT_FEASIBLE_NOMINEE' if chosen else 'NO_POINT_FEASIBLE_NOMINEE',
                          'attacked_label': chosen if chosen else fallback(arm),
                          'attacked_status': 'SELECTED_POINT_ELIGIBLE' if chosen else 'FIXED_EPOCH10_DIAGNOSTIC_INELIGIBLE',
                          'failed_clauses_at_fallback': [dict(x) for s in (0, 1) for x in table[f'{s}|{arm}|{fallback(arm)}']['clauses'] if not x['point_pass']]}
    return selection


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, guard = queue_config(spec_path, 'verification')
    lock = _lock(spec, verify, 'training_complete_lock', 'BANK_TRAINING_FROZEN')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .wires import prepare_geometry
    from .services import service_forward
    from .recipe import ARMS, checkpoint_labels, selectable_labels, fallback_label
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
    rows = []; table = {}
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
        common = dict(model=model, geometry=geometry, verify=verify, seed=seed, source_checkpoint_sha256=c['sha256'],
                      adapter_binding_sha256=ab['sha256'], binding_scales=binding['scales'])
        start = service_forward({'kind': 'RAW_PROJ'}, X, names, **common)

        def record(name, label, wires, kind):
            cl = clauses(y, [wires[0]['probability'], wires[1]['probability']], refs, seed)
            ch = {}
            for i, task in enumerate(TASKS):
                ch[task] = {k: float(np.max(np.abs(wires[i][k].astype(np.float64) - start[i][k].astype(np.float64))))
                            for k in ('representation', 'centered_logits', 'probability')}
                ch[task]['decision_changes'] = int((wires[i]['decision'] != start[i]['decision']).sum())
                ch[task]['wire_identical_to_RAW_PROJ'] = bool(np.array_equal(wires[i]['wire'], start[i]['wire']))
            table[f'{seed}|{name}|{label}'] = {'clauses': cl, 'all_point_pass': all(x['point_pass'] for x in cl),
                                                'metrics': {task: summary(y[task], wires[i]['probability'], list(range(2 if i == 0 else 6))) for i, task in enumerate(TASKS)},
                                                'change_vs_RAW_PROJ': ch, 'kind': kind}
            for x in cl:
                rows.append({'seed': seed, 'service': name, 'label': label, **x})
        record('RAW_PROJ', 'E0', start, 'reference')
        record('O_PREVIOUS', 'E10', service_forward(spec['o_previous'][str(seed)], X, names, **common), 'inherited_component_reference')
        for arm in ARMS:
            for label in checkpoint_labels(arm):
                ck = spec['checkpoint_paths'][str(seed)][arm][label]
                if ck['sha256'] != lock['checkpoints'][str(seed)][arm][label]['sha256']:
                    raise PermissionError('private checkpoint binding differs from committed TRAINING_COMPLETE_LOCK')
                svc = {'kind': 'ADAPTER2', 'income': {'source': 'TRA1', 'arm': arm, 'label': label, 'checkpoint': ck},
                       'occupation': {'source': 'TRA1', 'arm': arm, 'label': label, 'checkpoint': ck}}
                w = service_forward(svc, X, names, **common)
                if label in ('E0', 'P1E0') and not all(np.array_equal(w[i]['wire'], start[i]['wire']) for i in range(2)):
                    raise AssertionError('epoch-0 checkpoint is not the RAW_PROJ zero-residual witness')
                record(arm, label, w, 'adapter_arm')
                if arm == 'J_BOTH':
                    for hname, keep in (('HYB_JI_RO', 'income'), ('HYB_RI_JO', 'occupation')):
                        hs = {'kind': 'ADAPTER2', 'income': None, 'occupation': None}
                        hs[keep] = svc[keep]
                        record(f'{hname}@{label}', label, service_forward(hs, X, names, **common), 'hybrid_diagnostic')
    selection = select(table, ARMS, selectable_labels, fallback_label)
    ref_ok = all(table[f'{s}|RAW_PROJ|E0']['all_point_pass'] for s in (0, 1))
    prev_ok = all(table[f'{s}|O_PREVIOUS|E10']['all_point_pass'] for s in (0, 1))
    result = {'schema': 'TRA1-fit2-utility-v1', 'queue_unit_id': active['unit_id'],
              'training_complete_lock_sha256': spec['training_complete_lock_sha256'], 'caps': CAPS,
              'clauses_per_service': 24, 'rows': int(len(X)), 'groups': int(len(np.unique(feat['unit']))),
              'selection': selection, 'RAW_PROJ_point_eligible': ref_ok, 'O_PREVIOUS_point_eligible': prev_ok, 'table': table,
              'INNER_opened': False, 'ATT_opened': False, 'FIT2_opened_once': True,
              'role_status': 'exposed development selection role, not pristine validation'}
    with Path(spec['output']).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    with Path(spec['rows_output']).open('x') as stream:
        json.dump(rows, stream, sort_keys=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
