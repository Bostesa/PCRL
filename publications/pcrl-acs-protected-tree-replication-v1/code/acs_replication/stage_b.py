"""Stage B, one seed: LOCAL_TREE (rho = 1, E20) and the matched TASK_REFIT (privacy disabled, E20); optional JOINT_TREE.

Transport of task_headroom/train_worker.py. Before training the zero correction is verified to reproduce the
stored Stage A teacher probabilities (<= 1e-12) and decisions (exact) on every development role (features only).
Every arm restarts from theta = 0 with the same deterministic warm critics and FIT-only calibration
(16 draws at the exact teacher; lambda_J_r(rho) = rho * median||grad CE_r|| / median||grad joint_r||).
LOCAL_TREE = source arm L_TREE (local privacy objective, lambda_J coefficient, simultaneous schedule, 20 epochs of 40 updates
per active recipient, AdamW lr 0.01 / wd 0.01 on theta). TASK_REFIT = source arm T_REFIT (identical loop, lambda = 0).
JOINT_TREE (optional diagnostic) = source arm J_TREE at rho = 1 (adds only the pair term).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import leafcorr as lc
from . import trees

ARMS = {'TASK_REFIT': 'T_REFIT', 'LOCAL_TREE': 'L_TREE', 'JOINT_TREE': 'J_TREE'}
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_teacher(stage_a_dir, task):
    return trees.TreeTeacher(json.loads((Path(stage_a_dir) / f'TREE_TEACHER_{task}.json').read_text()))


def run(seed, stage_a_dir, out_dir, arms):
    if not D.lock_committed('SCIENCE_LOCK.json'):
        raise PermissionError('Stage B requires the committed SCIENCE_LOCK')
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh Stage B directory required')
    out.mkdir(parents=True, mode=0o700)
    F = D.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    names = F['feature_names']; X = F['X']; y = [F['income'], F['occupation']]
    teachers = [load_teacher(stage_a_dir, t) for t in D.TASKS]
    with np.load(Path(stage_a_dir) / 'STAGE_A_OUTPUTS.npz') as z:
        stored = {k: z[k].copy() for k in z.files}
    roles_X = {r: (X if r == 'MODEL_FIT' else D.load_role(r, ('features',))['X']) for r in ROLES}
    identity = {}
    for t, teacher in zip(D.TASKS, teachers):
        for role, XX in roles_X.items():
            L = teacher.route(XX, names)
            zero = trees.probabilities(t, teacher.raw_from_leaves(L, np.zeros((teacher.T * teacher.L, teacher.dim))))
            ref = stored[f'T_TREE__{role}__{t}']
            identity[f'{t}|{role}'] = {'max_abs_prob_diff': float(np.max(np.abs(zero - ref))),
                                       'decisions_identical': bool(np.array_equal(zero.argmax(1), ref.argmax(1)))}
    if not all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for v in identity.values()):
        raise AssertionError(f'zero-correction identity failed: {identity}')
    leaves = [teachers[i].route(X, names) for i in range(2)]
    upd = np.arange(len(F['row_id']))
    probe = D.order_rows(F['row_id'], 'ACSR1-probe', 512); reference = D.order_rows(F['row_id'], 'ACSR1-reference', 256)

    def make():
        rec = [lc.Recipient(t, teachers[i], leaves[i], teachers[i].raw_from_leaves(leaves[i])) for i, t in enumerate(D.TASKS)]
        return lc.Trainer(seed, rec, y, upd, F['sex'], probe, reference)
    tr0 = make(); critics = tr0.warm(); calib = tr0.calibrate(critics); warm_sha = critics.state_sha()
    record = {'schema': 'ACSR1-stage-b-seed-v1', 'seed': seed, 'zero_identity': identity, 'calibration': calib,
              'warm_critic_sha256': warm_sha, 'runs': {}, 'checkpoints': {}, 'probe_rows_sha256': D.arr_sha(probe),
              'reference_rows_sha256': D.arr_sha(reference), 'DEV_REPORT_labels_opened': False}
    outputs = {}
    for name in arms:
        src = ARMS[name]; saved = {}

        def on_ck(label, thetas, info, name=name, saved=saved):
            p = out / f'{name}__{label}.npz'
            with p.open('xb') as f:
                np.savez(f, theta_income=thetas[0], theta_occupation=thetas[1])
            p.chmod(0o600)
            saved[label] = {'file': p.name, 'sha256': sha(p), **info}
        tr = make()
        res = lc.run_arm(tr, src, 1.0, calib['coefficients'], on_ck, counterfactual_steps=8 if name == 'LOCAL_TREE' else 0)
        if res['warm_critic_sha256'] != warm_sha:
            raise AssertionError('warm critics differ across runs')
        record['runs'][name] = {k: v for k, v in res.items() if k != 'logs'}
        record['checkpoints'][name] = saved
        with (out / f'{name}__LOGS.json').open('x') as f:
            json.dump(res['logs'], f)
        with np.load(out / saved['E20']['file']) as z:
            thetas = [z['theta_income'].copy(), z['theta_occupation'].copy()]
        for role, XX in roles_X.items():
            for i, (t, teacher) in enumerate(zip(D.TASKS, teachers)):
                outputs[f'{name}__{role}__{t}'] = teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(XX, names), thetas[i]))
    op = out / 'STAGE_B_OUTPUTS_E20.npz'
    with op.open('xb') as f:
        np.savez(f, **outputs)
    op.chmod(0o600)
    record['outputs_E20_sha256'] = sha(op)
    with (out / 'STAGE_B_SEED.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--stage-a', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--arms', default='TASK_REFIT,LOCAL_TREE')
    a = ap.parse_args(); r = run(a.seed, a.stage_a, a.out, a.arms.split(','))
    print(json.dumps({'seed': r['seed'], 'lambda': {k: v['lambda'] for k, v in r['runs'].items()},
                      'updates': {k: v['updates_per_recipient'] for k, v in r['runs'].items()}}))
