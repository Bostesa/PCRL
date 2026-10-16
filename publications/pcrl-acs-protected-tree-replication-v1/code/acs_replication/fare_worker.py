"""One seed: official FARE bank (eth-sri/fare pinned, oar.fare_official in its own Python 3.9 env) and matched zero-fairness controls.

Transport of tree_completion/fare_worker.py. Access matches the candidate: the official fair tree (criterion
'fair_gini_dp') is grown on MODEL_FIT people with task labels and SEX. Its own task head (validated
jcv.finalize.fit_head: StandardScaler + LogisticRegression, C in {0.01, ..., 100} chosen by internal-fold log loss) is
fit on one-hot cells of MODEL_FIT using the SAME household-disjoint internal split as the Stage A references.
Released wire: the head's probability vector and argmax decision. Settings: the predecessor's three task-first
configurations F1-F3, each with its zero-fairness compression twin Z1-Z3 (gamma = 0, same k-bar / n_i budget).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import data as D

SETTINGS = [  # tree_completion.fare_worker.SETTINGS (predecessor grid ids 1-3), unchanged
    {"id": 1, "name": "F1_task_first_k100_g0.3", "max_leaf_nodes": 100, "min_samples_leaf": 100, "gamma": 0.3, "criterion": "fair_gini_dp"},
    {"id": 2, "name": "F2_task_first_k50_g0.7", "max_leaf_nodes": 50, "min_samples_leaf": 100, "gamma": 0.7, "criterion": "fair_gini_dp"},
    {"id": 3, "name": "F3_balanced_k20_g0.85", "max_leaf_nodes": 20, "min_samples_leaf": 100, "gamma": 0.85, "criterion": "fair_gini_dp"},
]
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(seed, out_dir):
    from oar import fare_official as FO
    from jcv.finalize import fit_head
    if not D.lock_committed('SCIENCE_LOCK.json'):
        raise PermissionError('FARE fits require the committed SCIENCE_LOCK')
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh FARE directory required')
    out.mkdir(parents=True, mode=0o700)
    F = D.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    roles = {r: (F if r == 'MODEL_FIT' else D.load_role(r, ('features',))) for r in ROLES}
    y = {t: F[t] for t in D.TASKS}
    fit_rows, val_rows, moved = D.internal_split(F['unit'], y, seed)
    verification = FO.verify_official_fare()
    record = {'schema': 'ACSR1-fare-seed-v1', 'seed': seed, 'official_verification': verification,
              'access': {'tree_fit_rows': 'MODEL_FIT with task labels and SEX', 'head_fit_rows': 'MODEL_FIT internal fit part (task labels)',
                         'head_C_selection_rows': 'MODEL_FIT internal validation part'},
              'internal_split': {'fit_rows': int(len(fit_rows)), 'validation_rows': int(len(val_rows))}, 'units': {},
              'DEV_REPORT_labels_opened': False}
    outputs = {}
    for st in SETTINGS:
        for zero in (False, True):
            cfgF = FO.FareConfig(max_leaf_nodes=st['max_leaf_nodes'], min_samples_leaf=st['min_samples_leaf'], gamma=st['gamma'],
                                 criterion=st['criterion'], name=st['name'])
            if zero:
                cfgF = cfgF.zero_fairness()
            label = f"{'Z' if zero else 'F'}{st['id']}"
            meta = {}
            for task in D.TASKS:
                k = D.K[task]
                model = FO.fit(F['X'].astype(np.float64), F[task], F['sex'], cfgF, seed)
                n_cells = int(len(model.meta['leaf_node_ids']))
                cells_fit = FO.encode(model, F['X'].astype(np.float64))
                head, hm = fit_head(np.eye(n_cells)[cells_fit], F[task], fit_rows, val_rows, k)
                for role, R in roles.items():
                    cells = cells_fit if role == 'MODEL_FIT' else FO.encode(model, R['X'].astype(np.float64))
                    P = np.asarray(head.predict_proba(np.eye(n_cells)[cells]), dtype=np.float64)
                    if P.shape[1] != k:
                        raise ValueError('FARE head class coverage differs')
                    outputs[f'FARE_{label}__{role}__{task}'] = P
                meta[task] = {'n_cells': n_cells, 'fingerprint': model.fingerprint, 'head': hm,
                              'config': cfgF.__dict__ if hasattr(cfgF, '__dict__') else str(cfgF), 'fit_rows': int(len(F['X'])),
                              'tree_fit_seconds': (model.runtime or {}).get('tree_fit_seconds')}
                model.save(out / f'{label}_{task}_model')
            record['units'][label] = {'setting': st['name'], 'zero_fairness': zero, 'tasks': meta}
    op = out / 'FARE_OUTPUTS.npz'
    with op.open('xb') as f:
        np.savez(f, **outputs)
    op.chmod(0o600); record['outputs_sha256'] = sha(op)
    with (out / 'FARE_SEED.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'units': {k: {t: v['tasks'][t]['n_cells'] for t in D.TASKS} for k, v in r['units'].items()}}))
