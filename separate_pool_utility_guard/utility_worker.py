"""One VALIDATION (inherited FIT2) utility pass after the committed training lock.

Scores every saved checkpoint (4 epochs x 5 arms x 2 seeds), applies the frozen
latest-common-feasible-epoch rule, and classifies the provisional SEPARATE_GUARD
feasibility gate. The final gate also requires the independent replay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .recipe import ARMS, RECIPE, canonical_sha

CAPS = RECIPE['validation_gate']['caps']
EPOCHS = RECIPE['checkpoint_epochs']


def _training_lock(spec, verify):
    p = verify(spec['training_complete_lock'], spec['training_complete_lock_sha256'])
    root = Path(spec['repo_root'])
    rel = p.resolve().relative_to(root.resolve()).as_posix()
    committed = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=root, capture_output=True, check=False)
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != spec['training_complete_lock_sha256']:
        raise PermissionError('TRAINING_COMPLETE_LOCK must be committed at HEAD before VALIDATION opens')
    lock = json.loads(p.read_text())
    if lock.get('status') != 'STAGE_A_TRAINING_FROZEN' or lock.get('VALIDATION_outcomes_opened') is not False:
        raise PermissionError('training lock does not freeze Stage A')
    return lock


def metrics(y, p, order):
    import numpy as np
    from task_conditioned_representation.inference import task_metrics
    out = task_metrics(y, p, order)
    out['confidence_by_true_class'] = [None if not np.any(y == k) else float(p[y == k].max(1).mean()) for k in order]
    return out


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, guard = queue_config(spec_path, 'verification')
    lock = _training_lock(spec, verify)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from task_conditioned_representation.inference import point_utility
    from .wires import forward_role, prepare_geometry
    from .train_worker import load_export
    val = {}
    names = {'features': ('row_id', 'unit', 'X', 'feature_names'), 'income': ('row_id', 'unit', 'income'),
             'occupation': ('row_id', 'unit', 'occupation')}
    vitems = spec['validation']
    for k, f in names.items():
        p = verify(vitems[k]['path'], vitems[k]['sha256'])
        with np.load(p, allow_pickle=False) as z:
            val[k] = {n: z[n].copy() for n in f}
    refs = {}
    for s in (0, 1):
        p = verify(vitems[f'references_s{s}']['path'], vitems[f'references_s{s}']['sha256'])
        with np.load(p, allow_pickle=False) as z:
            refs[s] = {n: z[n].copy() for n in z.files}
    for k in ('income', 'occupation'):
        if not np.array_equal(val[k]['row_id'], val['features']['row_id']):
            raise PermissionError('VALIDATION pairing differs')
    X = val['features']['X']; fn = val['features']['feature_names']
    y = {'income': val['income']['income'], 'occupation': val['occupation']['occupation']}
    guard()
    rows = []; table = {}; wire_changes = {}
    for seed in (0, 1):
        g = spec['geometry'][str(seed)]
        geometry = prepare_geometry(json.loads(verify(g['path'], g['sha256']).read_text()), expected_sha256=g['binding_sha256'],
                                    source_seed=seed)
        src_model, _ = load_checkpoint(verify(spec['source_checkpoints'][str(seed)]['path'], spec['source_checkpoints'][str(seed)]['sha256']))
        start = forward_role(src_model, X, fn, geometry, variant='RAW_HYBRID')
        ref_metrics = {task: metrics(y[task], start[i]['probability'], src_model.class_orders[i]) for i, task in enumerate(('income', 'occupation'))}
        for arm in ARMS:
            for epoch in EPOCHS:
                c = lock['checkpoints'][str(seed)][arm][str(epoch)]
                model, payload = load_checkpoint(verify(c['path'], c['sha256']), c['sha256'])
                if payload['seed'] != seed or payload['epoch'] != epoch or payload['wire_variant'] != arm:
                    raise PermissionError('checkpoint identity differs')
                views = forward_role(model, X, fn, geometry, variant='RAW_HYBRID')
                if not np.array_equal(views[0]['wire'], start[0]['wire']):
                    raise AssertionError('frozen income wire does not alias the start')
                ch = {k: float(np.max(np.abs(views[1][k].astype(np.float64) - start[1][k].astype(np.float64))))
                      for k in ('representation', 'centered_logits', 'probability')}
                ch['decision_changes'] = int((views[1]['decision'] != start[1]['decision']).sum())
                wire_changes[f'{seed}|{arm}|{epoch}'] = ch
                entry = {}
                for i, task in enumerate(('income', 'occupation')):
                    order = model.class_orders[i]
                    m = metrics(y[task], views[i]['probability'], order)
                    inc = point_utility(m, ref_metrics[task])
                    orig = {}
                    for ref in ('U0', 'Ucal'):
                        rm = metrics(y[task], refs[seed][f'{ref}_{i+1}'], order)
                        orig[ref] = point_utility(m, rm)
                    passes = (inc['accuracy_loss'] <= CAPS['accuracy_loss'] and inc['nll_excess'] <= CAPS['nll_increase']
                              and inc['brier_excess'] <= CAPS['brier_increase'])
                    entry[task] = {'metrics': m, 'incremental_vs_start': inc, 'original_reference_deltas': orig,
                                   'incremental_proxy_pass': bool(passes)}
                    rows.append({'seed': seed, 'arm': arm, 'epoch': epoch, 'task': task, 'accuracy': m['accuracy'],
                                 'nll': m['nll'], 'brier': m['brier'], 'balanced_accuracy': m['balanced_accuracy'],
                                 'confidence': m['confidence'], 'per_class_recall': m['per_class_recall'],
                                 'class_counts': m['class_counts'], 'accuracy_loss_vs_start': inc['accuracy_loss'],
                                 'nll_increase_vs_start': inc['nll_excess'], 'brier_increase_vs_start': inc['brier_excess'],
                                 'U0_accuracy_loss': orig['U0']['accuracy_loss'], 'U0_nll_excess': orig['U0']['nll_excess'],
                                 'U0_brier_excess': orig['U0']['brier_excess'], 'Ucal_accuracy_loss': orig['Ucal']['accuracy_loss'],
                                 'Ucal_nll_excess': orig['Ucal']['nll_excess'], 'Ucal_brier_excess': orig['Ucal']['brier_excess'],
                                 'incremental_proxy_pass': bool(passes)})
                table[f'{seed}|{arm}|{epoch}'] = entry
    selection = {}
    for arm in ARMS:
        feasible = [e for e in EPOCHS if all(table[f'{s}|{arm}|{e}'][t]['incremental_proxy_pass'] for s in (0, 1) for t in ('income', 'occupation'))]
        chosen = max(feasible) if feasible else 0
        selection[arm] = {'feasible_common_epochs': feasible, 'chosen_epoch': chosen,
                          'fallback_to_epoch0': chosen == 0}
    sep = selection['SEPARATE_GUARD']['chosen_epoch']
    ev = lock['privacy_movement_evidence']
    criteria = {
        'chosen_common_epoch_positive': sep > 0,
        'released_occupation_wire_changed_both_seeds': sep > 0 and all(
            lock['probe_wire_changed'][str(s)]['SEPARATE_GUARD'][str(sep)] for s in (0, 1)),
        'validation_wire_changed_both_seeds': sep > 0 and all(
            (lambda c: c['decision_changes'] > 0 or max(c['representation'], c['centered_logits'], c['probability']) > 1e-10)(
                wire_changes[f'{s}|SEPARATE_GUARD|{sep}']) for s in (0, 1)),
        'privacy_bearing_accepted_update_both_seeds': all(ev[str(s)]['privacy_bearing_accepted_wire_changing_update'] for s in (0, 1)),
        'all_incremental_validation_proxies_pass_both_seeds': sep > 0,
    }
    provisional = 'PASS_PENDING_INDEPENDENT_REPLAY' if all(criteria.values()) else 'FAIL'
    result = {'schema': 'SPG1-validation-utility-v1', 'queue_unit_id': active['unit_id'],
              'training_complete_lock_sha256': spec['training_complete_lock_sha256'],
              'validation_rows': int(len(X)), 'validation_groups': int(len(np.unique(val['features']['unit']))),
              'caps': CAPS, 'selection': selection, 'separate_guard_criteria': criteria,
              'provisional_gate': provisional, 'table': table, 'validation_wire_change_vs_start': wire_changes,
              'INNER_opened': False, 'ATT_opened': False, 'validation_opened_once': True}
    out = Path(spec['output'])
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    with Path(spec['rows_output']).open('x') as stream:
        json.dump(rows, stream, sort_keys=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
