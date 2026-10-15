"""Stage B, one seed and one family: leaf-score correction fits at every registered strength.

Before training: the zero correction is verified to reproduce the selected
teacher's stored Stage A probabilities (<= 1e-12) and decisions (exact) on
the pool, FIT2, ATT and INNER (features only). Every arm restarts from theta = 0
with the same deterministic warm critics and FIT-only calibration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def order_rows(row_id, salt, n):
    ranked = sorted(range(len(row_id)), key=lambda i: hashlib.sha256(f'{salt}|{int(row_id[i])}'.encode()).hexdigest())
    return np.asarray(ranked[:n], dtype=np.int64)


def run(spec_path):
    from .evaluation_queue import queue_config
    from .data import load_role
    from . import trees, leafcorr as lc
    from .services import load_teacher
    active, cfg, verify, guard = queue_config(spec_path, 'protection')
    lock = json.loads(verify(cfg['stage_b_lock'], cfg['stage_b_lock_sha256']).read_text())
    if lock.get('status') != 'STAGE_B_LOCKED':
        raise PermissionError('protected fits require the committed STAGE_B_LOCK')
    seed = int(cfg['seed']); family = cfg['family']; R = cfg['roles']
    U = load_role('UPDATE', R['UPDATE'], verify, ('features', 'income', 'occupation', 'sex'))
    G = load_role('GUARD', R['GUARD'], verify, ('features', 'income', 'occupation'))
    names = tuple(map(str, U['feature_names']))
    X = np.concatenate((U['X'], G['X'])); y = [np.concatenate((U[t], G[t])) for t in ('income', 'occupation')]
    teachers = [load_teacher(cfg['teachers'][t], verify) for t in ('income', 'occupation')]
    identity = {}
    roles_X = {'POOL': X}
    for role in ('FIT2', 'ATT', 'INNER'):
        roles_X[role] = load_role(role, R[role], verify, ('features',))['X']
    for t, teacher in zip(('income', 'occupation'), teachers):
        with np.load(verify(cfg['teacher_outputs'][t]['path'], cfg['teacher_outputs'][t]['sha256'])) as z:
            stored = {k: z[k].copy() for k in z.files}
        for role, XX in roles_X.items():
            L = teacher.route(XX, names)
            zero = trees.probabilities(t, teacher.raw_from_leaves(L, np.zeros((teacher.T * teacher.L, teacher.dim))))
            identity[f'{t}|{role}'] = {'max_abs_prob_diff': float(np.max(np.abs(zero - stored[role]))),
                                       'decisions_identical': bool(np.array_equal(zero.argmax(1), stored[role].argmax(1)))}
    if not all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for v in identity.values()):
        raise AssertionError(f'zero-correction identity failed: {identity}')
    leaves = [teachers[i].route(X, names) for i in range(2)]
    upd = np.arange(len(U['row_id']))
    probe = order_rows(U['row_id'], 'THP1-probe', 512); reference = order_rows(U['row_id'], 'THP1-reference', 256)
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private training directory required')
    out.mkdir(parents=True, mode=0o700)

    def make():
        rec = [lc.Recipient(t, teachers[i], leaves[i], teachers[i].raw_from_leaves(leaves[i])) for i, t in enumerate(('income', 'occupation'))]
        return lc.Trainer(seed, rec, y, upd, U['sex'], probe, reference)
    tr0 = make(); critics = tr0.warm(); calib = tr0.calibrate(critics); warm_sha = critics.state_sha()
    strengths = [None] if family == 'T_REFIT' else lc.RECIPE['strength_ratios']
    record = {'schema': 'THP1-stage-b-family-v1', 'queue_unit_id': active['unit_id'], 'seed': seed, 'family': family,
              'zero_identity': identity, 'calibration': calib, 'warm_critic_sha256': warm_sha, 'runs': {}, 'checkpoints': {}}
    for rho in strengths:
        guard()
        tr = make(); tag = 'none' if rho is None else str(rho)
        saved = {}

        def on_ck(label, thetas, info, tag=tag, saved=saved):
            p = out / f'{family}__rho{tag}__{label}.npz'
            with p.open('xb') as f:
                np.savez(f, theta_income=thetas[0], theta_occupation=thetas[1])
            saved[label] = {'file': p.name, 'sha256': _sha(p), **info}
        res = lc.run_arm(tr, family, rho if rho is not None else 1.0, calib['coefficients'], on_ck,
                         counterfactual_steps=8 if (family == 'J_TREE' and rho == 3.0) else 0)
        if res['warm_critic_sha256'] != warm_sha:
            raise AssertionError('warm critics differ across runs')
        record['runs'][tag] = {k: v for k, v in res.items() if k != 'logs'}
        record['checkpoints'][tag] = saved
        with (out / f'{family}__rho{tag}__LOGS.json').open('x') as f:
            json.dump(res['logs'], f)
    record['FIT2_labels_opened'] = False; record['INNER_labels_opened'] = False
    with (out / 'FAMILY_RECORD.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    with Path(cfg['output']).open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
