"""One seed: access-honoring OFFICIAL FARE bank and its matched zero-fairness controls.

Access matches the candidate exactly: the official FARE fair tree (eth-sri/fare pinned commit, patched
sklearn criterion 'fair_gini_dp', oar.fare_official in its own Python 3.9 environment) is grown on UPDATE
rows only, the only rows whose SEX the candidate's training ever used (task label + SEX). Its own task head
(the validated jcv.finalize.fit_head: StandardScaler + LogisticRegression, C in {0.01,...,100} chosen by
internal-fold log loss) is fit on one-hot cells of the UPDATE+GUARD task pool (task labels only; no GUARD
SEX) using the same 15% group-disjoint internal split as the Stage A teachers. Released wire: the head's
probability vector and argmax decision. Settings: the registered predecessor grid's three most task-first
configs F1, F2, F3, each with its zero-fairness twin (gamma = 0, same k-bar / n_i budget).
FIT2/ATT/INNER outputs are written in the stored-release format; no metric is computed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SETTINGS = [  # rgj.baselines.FARE_GRID ids 1-3 (predecessor jcv.run.FARE_GRID), unchanged
    {"id": 1, "name": "F1_task_first_k100_g0.3", "max_leaf_nodes": 100, "min_samples_leaf": 100, "gamma": 0.3, "criterion": "fair_gini_dp"},
    {"id": 2, "name": "F2_task_first_k50_g0.7", "max_leaf_nodes": 50, "min_samples_leaf": 100, "gamma": 0.7, "criterion": "fair_gini_dp"},
    {"id": 3, "name": "F3_balanced_k20_g0.85", "max_leaf_nodes": 20, "min_samples_leaf": 100, "gamma": 0.85, "criterion": "fair_gini_dp"},
]


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(spec_path):
    from .evaluation_queue import queue_config
    from task_headroom.data import load_role, internal_split
    from oar import fare_official as FO
    from jcv.finalize import fit_head
    active, cfg, verify, guard = queue_config(spec_path, 'protection')
    lock = json.loads(verify(cfg['completion_lock'], cfg['completion_lock_sha256']).read_text())
    if lock.get('status') != 'COMPLETION_LOCKED':
        raise PermissionError('FARE fits require the committed COMPLETION_LOCK')
    seed = int(cfg['seed']); R = cfg['roles']
    U = load_role('UPDATE', R['UPDATE'], verify, ('features', 'income', 'occupation', 'sex'))
    G = load_role('GUARD', R['GUARD'], verify, ('features', 'income', 'occupation'))
    roles = {r: load_role(r, R[r], verify, ('features',)) for r in ('FIT2', 'ATT', 'INNER')}
    pool_X = np.concatenate((U['X'], G['X'])); pool_unit = np.concatenate((U['unit'], G['unit']))
    pool_y = {t: np.concatenate((U[t], G[t])) for t in ('income', 'occupation')}
    fit_rows, val_rows, moved = internal_split(pool_unit, pool_y, seed)
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private FARE directory required')
    out.mkdir(parents=True, mode=0o700)
    guard()
    verification = FO.verify_official_fare()
    record = {'schema': 'TC1-fare-seed-v1', 'queue_unit_id': active['unit_id'], 'seed': seed, 'official_verification': verification,
              'access': {'tree_fit_rows': 'UPDATE (8180) with task labels and SEX', 'head_fit_rows': 'UPDATE+GUARD internal fit part (task labels only)',
                         'head_C_selection_rows': 'UPDATE+GUARD internal validation part', 'GUARD_SEX_used': False},
              'internal_split': {'fit_rows': int(len(fit_rows)), 'validation_rows': int(len(val_rows))}, 'units': {}}
    rid = np.concatenate([roles[r]['row_id'] for r in ('FIT2', 'ATT', 'INNER')])
    allX = np.concatenate([roles[r]['X'] for r in ('FIT2', 'ATT', 'INNER')])
    for st in SETTINGS:
        for zero in (False, True):
            guard()
            cfgF = FO.FareConfig(max_leaf_nodes=st['max_leaf_nodes'], min_samples_leaf=st['min_samples_leaf'], gamma=st['gamma'],
                                 criterion=st['criterion'], name=st['name'])
            if zero:
                cfgF = cfgF.zero_fairness()
            label = f"{'Z' if zero else 'F'}{st['id']}"
            parts, meta = {}, {}
            for i, (task, k) in enumerate((('income', 2), ('occupation', 6))):
                model = FO.fit(U['X'].astype(np.float64), U[task], U['sex'], cfgF, seed)
                cells_pool = FO.encode(model, pool_X.astype(np.float64)); cells_all = FO.encode(model, allX.astype(np.float64))
                port = FO.encode_portable(model, allX.astype(np.float64))
                n_cells = int(len(model.meta['leaf_node_ids']))
                Rp = np.eye(n_cells)[cells_pool]; Ra = np.eye(n_cells)[cells_all]
                head, hm = fit_head(Rp, pool_y[task], fit_rows, val_rows, k)
                P = np.asarray(head.predict_proba(Ra), dtype=np.float64)
                if P.shape[1] != k:
                    raise ValueError('FARE head class coverage differs')
                parts[f'q{i + 1}'] = P; parts[f'hard{i + 1}'] = P.argmax(1).astype(np.int64)
                meta[task] = {'n_cells': n_cells, 'fingerprint': model.fingerprint, 'head': hm, 'encode_portable_identical': bool(np.array_equal(port, cells_all)),
                              'config': cfgF.__dict__ if hasattr(cfgF, '__dict__') else str(cfgF), 'fit_rows': int(len(U['X'])),
                              'tree_fit_seconds': (model.runtime or {}).get('tree_fit_seconds')}
                model.save(out / f'{label}_{task}_model')
            p = out / f'FARE_RELEASE__{label}.npz'
            with p.open('xb') as f:
                np.savez(f, row_id=rid, **parts)
            record['units'][label] = {'setting': st['name'], 'zero_fairness': zero, 'release_file': p.name, 'release_sha256': _sha(p), 'tasks': meta}
    record['FIT2_labels_opened'] = False; record['INNER_labels_opened'] = False
    with (out / 'FARE_RECORD.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    with Path(cfg['output']).open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
