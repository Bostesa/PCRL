"""One seed: fresh equally exposed neural RAW-J at rho {1,3,10} and its task-only continuation T_NEURAL.

Before training: the admitted N_EQ model must reproduce its stored Stage A outputs (<= 1e-12) on the pool,
FIT2, ATT and INNER (features only). Every run restarts from N_EQ with identical warm critics.
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


def load_neq(path):
    import torch
    from task_headroom import neural
    payload = torch.load(path, map_location='cpu', weights_only=True)
    models = []
    for task, k in (('income', 2), ('occupation', 6)):
        m = neural.network(k); m.load_state_dict(payload[task]); m.eval(); models.append(m)
    return payload['feature_names'], models


def run(spec_path):
    import torch
    from .evaluation_queue import queue_config
    from task_headroom.data import load_role
    from task_headroom import neural
    from . import neural_rawj as nr
    active, cfg, verify, guard = queue_config(spec_path, 'protection')
    lock = json.loads(verify(cfg['completion_lock'], cfg['completion_lock_sha256']).read_text())
    if lock.get('status') != 'COMPLETION_LOCKED':
        raise PermissionError('neural control fits require the committed COMPLETION_LOCK')
    seed = int(cfg['seed']); R = cfg['roles']
    U = load_role('UPDATE', R['UPDATE'], verify, ('features', 'income', 'occupation', 'sex'))
    G = load_role('GUARD', R['GUARD'], verify, ('features', 'income', 'occupation'))
    X = np.concatenate((U['X'], G['X'])); y = [np.concatenate((U[t], G[t])) for t in ('income', 'occupation')]
    names, models = load_neq(verify(cfg['n_eq']['path'], cfg['n_eq']['sha256']))
    if tuple(names) != tuple(map(str, U['feature_names'])):
        raise PermissionError('N_EQ schema differs')
    # N_EQ identity against its stored Stage A outputs.
    with np.load(verify(cfg['neq_outputs']['path'], cfg['neq_outputs']['sha256'])) as z:
        stored = {k: z[k].copy() for k in z.files}
    roles_X = {'POOL': X, **{r: load_role(r, R[r], verify, ('features',))['X'] for r in ('FIT2', 'ATT', 'INNER')}}
    identity = {}
    for (task, m) in zip(('income', 'occupation'), models):
        for role, XX in roles_X.items():
            identity[f'{task}|{role}'] = float(np.max(np.abs(neural.probabilities(m, XX) - stored[f'{role}__{task}'])))
    if max(identity.values()) > 1e-12:
        raise AssertionError(f'N_EQ identity failed: {identity}')
    upd = np.arange(len(U['row_id']))
    probe = order_rows(U['row_id'], 'THP1-probe', 512); reference = order_rows(U['row_id'], 'THP1-reference', 256)
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private training directory required')
    out.mkdir(parents=True, mode=0o700)

    def make():
        _, ms = load_neq(cfg['n_eq']['path'])
        rec = [nr.NeuralRecipient(t, m, X) for t, m in zip(('income', 'occupation'), ms)]
        return nr.NeuralTrainer(seed, rec, y, upd, U['sex'], probe, reference)
    t0 = make(); critics = t0.warm(); calib = t0.calibrate(critics); warm_sha = critics.state_sha()
    record = {'schema': 'TC1-neural-control-seed-v1', 'queue_unit_id': active['unit_id'], 'seed': seed, 'neq_identity_max_abs': identity,
              'calibration': calib, 'warm_critic_sha256': warm_sha, 'recipe': nr.RECIPE, 'runs': {}, 'checkpoints': {},
              'parameter_counts': {t: int(sum(p.numel() for p in m.parameters())) for t, m in zip(('income', 'occupation'), models)}}
    plan = [('RAW_J', rho) for rho in nr.RECIPE['strength_ratios']] + [('T_NEURAL', None)]
    for arm, rho in plan:
        guard()
        tr = make(); tag = 'none' if rho is None else str(rho); saved = {}

        def on_ck(label, states, info, arm=arm, tag=tag, saved=saved):
            p = out / f'{arm}__rho{tag}__{label}.pt'
            torch.save({'feature_names': list(names), **states}, p); p.chmod(0o600)
            saved[label] = {'file': p.name, 'sha256': _sha(p), **info}
        res = nr.run_arm(tr, arm, rho if rho is not None else 1.0, calib['coefficients'], on_ck,
                         counterfactual_steps=8 if (arm == 'RAW_J' and rho == 3.0) else 0)
        if res['warm_critic_sha256'] != warm_sha:
            raise AssertionError('warm critics differ across runs')
        record['runs'][f'{arm}|{tag}'] = {k: v for k, v in res.items() if k != 'logs'}
        record['checkpoints'][f'{arm}|{tag}'] = saved
        with (out / f'{arm}__rho{tag}__LOGS.json').open('x') as f:
            json.dump(res['logs'], f)
    record['FIT2_labels_opened'] = False; record['INNER_labels_opened'] = False
    with (out / 'NEURAL_RECORD.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    with Path(cfg['output']).open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
