"""Stage A, one seed: the equally exposed neural reference and the bounded CatBoost registry.

Fitting pool: UPDATE + GUARD (10223 rows). Internal early-stopping fold: the
label-free 15% group-disjoint split (seed + 17001) with whole-group class
support repair; the SAME split for N_EQ and trees. Trees: 800 trees with the
internal fold as eval set, tree count = first argmin of the eval loss + 1,
then a refit on the whole pool with that count and the same seed. Every
registry teacher is converted to an explicit numpy teacher and verified
against CatBoost (routing bitwise, raw/probabilities float64) on every role.
FIT2/ATT/INNER outputs are written privately; no FIT2 metric is computed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(spec_path):
    from .evaluation_queue import queue_config
    from .data import load_role, internal_split, arr_sha
    from . import trees, neural
    active, cfg, verify, guard = queue_config(spec_path, 'witness')
    lock = json.loads(verify(cfg['science_lock'], cfg['science_lock_sha256']).read_text())
    if lock.get('status') != 'SCIENCE_LOCKED':
        raise PermissionError('Stage A requires the committed SCIENCE_LOCK')
    import torch
    torch.set_num_threads(1)
    seed = int(cfg['seed']); R = cfg['roles']
    U = load_role('UPDATE', R['UPDATE'], verify, ('features', 'income', 'occupation'))
    G = load_role('GUARD', R['GUARD'], verify, ('features', 'income', 'occupation'))
    other = {'FIT2': load_role('FIT2', R['FIT2'], verify, ('features',)), 'ATT': load_role('ATT', R['ATT'], verify, ('features',)),
             'INNER': load_role('INNER', R['INNER'], verify, ('features',))}
    names = tuple(map(str, U['feature_names']))
    X = np.concatenate((U['X'], G['X'])); unit = np.concatenate((U['unit'], G['unit']))
    y = {t: np.concatenate((U[t], G[t])) for t in ('income', 'occupation')}
    fit_rows, val_rows, moved = internal_split(unit, y, seed)
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private Stage A directory required')
    out.mkdir(parents=True, mode=0o700)
    record = {'schema': 'THP1-stage-a-seed-v1', 'queue_unit_id': active['unit_id'], 'seed': seed, 'pool_rows': int(len(X)),
              'internal_split': {'fit_rows': int(len(fit_rows)), 'validation_rows': int(len(val_rows)), 'moved_groups': len(moved),
                                 'fit_row_sha256': arr_sha(fit_rows), 'validation_row_sha256': arr_sha(val_rows),
                                 'group_overlap': int(len(np.intersect1d(unit[fit_rows], unit[val_rows])))},
              'neural': {}, 'trees': {}, 'files': {}}
    roles_X = {'POOL': X, **{k: v['X'] for k, v in other.items()}}
    # Equally exposed neural reference.
    guard()
    states = {'feature_names': list(names)}; nprobs = {}
    for task, k in (('income', 2), ('occupation', 6)):
        t0 = time.process_time()
        model, info = neural.fit(X, y[task], k, fit_rows, val_rows, trees.seed_for('THP1neural', seed, task) % 2 ** 31)
        info['cpu_s'] = time.process_time() - t0
        record['neural'][task] = info; states[task] = model.state_dict()
        for role, XX in roles_X.items():
            nprobs[f'{role}|{task}'] = neural.probabilities(model, XX)
    mp = out / 'NEQ_MODEL.pt'; torch.save(states, mp); mp.chmod(0o600)
    record['files']['NEQ_MODEL.pt'] = _sha(mp)
    npath = out / 'NEQ_OUTPUTS.npz'
    with npath.open('xb') as f:
        np.savez(f, **{k.replace('|', '__'): v for k, v in nprobs.items()})
    record['files']['NEQ_OUTPUTS.npz'] = _sha(npath)
    # Bounded CatBoost registry.
    for task in ('income', 'occupation'):
        for depth in trees.REGISTRY['depth']:
            guard()
            rs = trees.seed_for('THP1catboost', seed, task, depth)
            t0 = time.process_time()
            m = trees.make_classifier(task, depth, trees.REGISTRY['max_trees'], rs)
            m.fit(X[fit_rows], y[task][fit_rows], eval_set=(X[val_rows], y[task][val_rows]))
            curve = m.get_evals_result()['validation'][trees.LOSS[task]]
            count = int(np.argmin(curve)) + 1
            internal_nll = float(curve[count - 1])
            final = trees.make_classifier(task, depth, count, rs)
            final.fit(X, y[task])
            teacher = trees.from_catboost(final, task, depth, names)
            parity = {role: trees.verify_against_catboost(teacher, final, XX, names) for role, XX in roles_X.items()}
            ok = all(v['routing_identical'] and v['decision_identical'] and v['raw_max_abs_diff'] <= 1e-9 and v['probability_max_abs_diff'] <= 1e-12
                     for v in parity.values())
            if not ok:
                raise AssertionError(f'teacher conversion parity failed: {task} depth {depth}: {parity}')
            tag = f'{task}_d{depth}'
            tp = out / f'TREE_TEACHER_{tag}.json'
            with tp.open('x') as f:
                json.dump(teacher.payload(), f, sort_keys=True)
            cbm = out / f'CATBOOST_{tag}.cbm'; final.save_model(str(cbm))
            probs = {}
            for role, XX in roles_X.items():
                probs[role] = teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(XX, names)))
            pp = out / f'TREE_OUTPUTS_{tag}.npz'
            with pp.open('xb') as f:
                np.savez(f, **probs)
            record['trees'][tag] = {'task': task, 'depth': depth, 'catboost_seed': rs, 'internal_curve_len': len(curve),
                                    'tree_count': count, 'internal_best_nll': internal_nll, 'internal_curve_every_50': [float(c) for c in curve[::50]],
                                    'cpu_s': time.process_time() - t0, 'parity': parity, 'teacher_sha256': teacher.sha(),
                                    'params': {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in final.get_params().items()}}
            for p in (tp, cbm, pp):
                record['files'][p.name] = _sha(p)
    record['FIT2_labels_opened'] = False; record['INNER_labels_opened'] = False
    record['catboost_version'] = trees.CATBOOST_VERSION
    with (out / 'STAGE_A_SEED.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    with Path(cfg['output']).open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
