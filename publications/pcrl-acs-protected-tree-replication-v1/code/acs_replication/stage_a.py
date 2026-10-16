"""Stage A, one seed: new ACS U0 (equally exposed neural reference), Ucal and the T_TREE CatBoost teachers.

Fitting people: MODEL_FIT only (task labels). Internal early-stopping fold: label-free 15% household-disjoint
split (seed + 17001) with whole-household class-support repair; the SAME split for U0 and the trees.
U0: validated source network, Recipe A (acs_replication.neural). Ucal: one global temperature per task fitted on
CALIBRATION (fixed bounded solver, no sweep). T_TREE: CatBoost depth 4, learning rate 0.05, at most 800 trees,
l2_leaf_reg 3; tree count = first argmin of the internal-fold loss + 1; refit on all MODEL_FIT people with that count
and the same seed; converted to an explicit numpy teacher verified against CatBoost on every development role.
Outputs on MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT are written privately from features only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import neural, trees

OUTPUT_ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(seed, out_dir):
    if not D.lock_committed('SCIENCE_LOCK.json'):
        raise PermissionError('Stage A requires the committed SCIENCE_LOCK')
    import torch
    torch.set_num_threads(1)
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh Stage A directory required')
    out.mkdir(parents=True, mode=0o700)
    F = D.load_role('MODEL_FIT', ('features', 'income', 'occupation'))
    C = D.load_role('CALIBRATION', ('features', 'income', 'occupation'))
    roles = {r: (F if r == 'MODEL_FIT' else C if r == 'CALIBRATION' else D.load_role(r, ('features',))) for r in OUTPUT_ROLES}
    names = F['feature_names']; X = F['X']; y = {t: F[t] for t in D.TASKS}
    fit_rows, val_rows, moved = D.internal_split(F['unit'], y, seed)
    rec = {'schema': 'ACSR1-stage-a-seed-v1', 'seed': seed, 'pool_rows': int(len(X)), 'input_width': int(X.shape[1]),
           'internal_split': {'fit_rows': int(len(fit_rows)), 'validation_rows': int(len(val_rows)), 'moved_households': len(moved),
                              'household_overlap': int(len(np.intersect1d(F['unit'][fit_rows], F['unit'][val_rows])))},
           'U0': {}, 'Ucal': {}, 'T_TREE': {}, 'files': {}, 'catboost_version': trees.CATBOOST_VERSION,
           'DEV_REPORT_labels_opened': False, 'sealed_assessment_touched': False}
    states = {'feature_names': list(names)}; outputs = {}
    for task in D.TASKS:
        k = D.K[task]; t0 = time.process_time()
        model, info = neural.fit(X, y[task], k, fit_rows, val_rows, trees.seed_for('ACSR1neural', seed, task) % 2 ** 31)
        info['cpu_s'] = time.process_time() - t0
        rec['U0'][task] = info; states[task] = model.state_dict()
        z_cal = neural.logits(model, C['X'])
        temp, tinfo = neural.fit_temperature(z_cal, C[task])
        rec['Ucal'][task] = tinfo
        for role, R in roles.items():
            z = neural.logits(model, R['X'])
            outputs[f'U0|{role}|{task}'] = neural.tempered_probabilities(z, 1.0)
            outputs[f'Ucal|{role}|{task}'] = neural.tempered_probabilities(z, temp)
    mp = out / 'U0_MODEL.pt'; torch.save({**states, 'temperatures': {t: rec['Ucal'][t]['temperature'] for t in D.TASKS}}, mp); mp.chmod(0o600)
    rec['files']['U0_MODEL.pt'] = sha(mp)
    teachers = {}
    for task in D.TASKS:
        depth = trees.REGISTRY['depth'][0]
        rs = trees.seed_for('ACSR1catboost', seed, task, depth); t0 = time.process_time()
        m = trees.make_classifier(task, depth, trees.REGISTRY['max_trees'], rs)
        m.fit(X[fit_rows], y[task][fit_rows], eval_set=(X[val_rows], y[task][val_rows]))
        curve = m.get_evals_result()['validation'][trees.LOSS[task]]
        count = int(np.argmin(curve)) + 1
        final = trees.make_classifier(task, depth, count, rs); final.fit(X, y[task])
        teacher = trees.from_catboost(final, task, depth, names)
        parity = {role: trees.verify_against_catboost(teacher, final, R['X'], names) for role, R in roles.items()}
        ok = all(v['routing_identical'] and v['decision_identical'] and v['raw_max_abs_diff'] <= 1e-9 and v['probability_max_abs_diff'] <= 1e-12
                 for v in parity.values())
        if not ok:
            raise AssertionError(f'teacher conversion parity failed: {task}: {parity}')
        tp = out / f'TREE_TEACHER_{task}.json'
        with tp.open('x') as f:
            json.dump(teacher.payload(), f, sort_keys=True)
        cbm = out / f'CATBOOST_{task}.cbm'; final.save_model(str(cbm))
        for role, R in roles.items():
            outputs[f'T_TREE|{role}|{task}'] = teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(R['X'], names)))
        teachers[task] = teacher
        rec['T_TREE'][task] = {'depth': depth, 'catboost_seed': rs, 'tree_count': count, 'internal_curve_len': len(curve),
                               'internal_best_loss': float(curve[count - 1]), 'internal_curve_every_50': [float(c) for c in curve[::50]],
                               'cpu_s': time.process_time() - t0, 'parity': parity, 'teacher_sha256': teacher.sha(),
                               'params': {k: (v.item() if isinstance(v, (np.floating, np.integer)) else v) for k, v in final.get_params().items()}}
        for p in (tp, cbm):
            rec['files'][p.name] = sha(p)
    op = out / 'STAGE_A_OUTPUTS.npz'
    with op.open('xb') as f:
        np.savez(f, **{k.replace('|', '__'): v for k, v in outputs.items()},
                 **{f'row_id__{r}': R['row_id'] for r, R in roles.items()}, **{f'unit__{r}': R['unit'] for r, R in roles.items()})
    op.chmod(0o600); rec['files'][op.name] = sha(op)
    with (out / 'STAGE_A_SEED.json').open('x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'U0_epochs': {t: r['U0'][t]['best_epoch'] for t in D.TASKS},
                      'T_TREE_counts': {t: r['T_TREE'][t]['tree_count'] for t in D.TASKS},
                      'temperatures': {t: r['Ucal'][t]['temperature'] for t in D.TASKS}}))
