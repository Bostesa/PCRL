"""One seed: NEURAL_RAW (ordinary neural adversarial protection from the new ACS U0) at the locked grid rho {1, 3, 10}.

Transport of tree_completion/train_neural_worker.py (T_NEURAL task-only continuation not run: optional, not required).
Before training, U0 must reproduce its stored Stage A outputs (<= 1e-12) on every development role (features only).
Every run restarts from U0 with identical warm critics; nomination candidates = checkpoints E2, E5, E10, E20 per strength
(the source nomination checkpoints), i.e. 12 configurations, frozen later on MODEL_SELECT by the registered rule.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import neural
from . import neural_raw as nr

ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_u0(path, d):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    models = []
    for task in D.TASKS:
        m = neural.network(D.K[task], d); m.load_state_dict(payload[task]); m.eval(); models.append(m)
    return payload['feature_names'], models


def run(seed, stage_a_dir, out_dir):
    import torch
    if not D.lock_committed('SCIENCE_LOCK.json'):
        raise PermissionError('NEURAL_RAW requires the committed SCIENCE_LOCK')
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh directory required')
    out.mkdir(parents=True, mode=0o700)
    F = D.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    X = F['X']; y = [F['income'], F['occupation']]; d = X.shape[1]
    u0_path = Path(stage_a_dir) / 'U0_MODEL.pt'
    names, models = load_u0(u0_path, d)
    if tuple(names) != F['feature_names']:
        raise PermissionError('U0 schema differs')
    with np.load(Path(stage_a_dir) / 'STAGE_A_OUTPUTS.npz') as z:
        stored = {k: z[k].copy() for k in z.files}
    roles_X = {r: (X if r == 'MODEL_FIT' else D.load_role(r, ('features',))['X']) for r in ROLES}
    identity = {}
    for task, m in zip(D.TASKS, models):
        for role, XX in roles_X.items():
            identity[f'{task}|{role}'] = float(np.max(np.abs(neural.probabilities(m, XX) - stored[f'U0__{role}__{task}'])))
    if max(identity.values()) > 1e-12:
        raise AssertionError(f'U0 identity failed: {identity}')
    upd = np.arange(len(F['row_id']))
    probe = D.order_rows(F['row_id'], 'ACSR1-probe', 512); reference = D.order_rows(F['row_id'], 'ACSR1-reference', 256)

    def make():
        _, ms = load_u0(u0_path, d)
        rec = [nr.NeuralRecipient(t, m, X) for t, m in zip(D.TASKS, ms)]
        return nr.NeuralTrainer(seed, rec, y, upd, F['sex'], probe, reference)
    t0 = make(); critics = t0.warm(); calib = t0.calibrate(critics); warm_sha = critics.state_sha()
    record = {'schema': 'ACSR1-neural-raw-seed-v1', 'seed': seed, 'u0_identity_max_abs': identity, 'calibration': calib,
              'warm_critic_sha256': warm_sha, 'recipe': nr.RECIPE, 'runs': {}, 'checkpoints': {},
              'parameter_counts': {t: int(sum(p.numel() for p in m.parameters())) for t, m in zip(D.TASKS, models)},
              'DEV_REPORT_labels_opened': False}
    outputs = {}
    for rho in nr.RECIPE['strength_ratios']:
        tr = make(); tag = str(rho); saved = {}

        def on_ck(label, states, info, tag=tag, saved=saved):
            p = out / f'NEURAL_RAW__rho{tag}__{label}.pt'
            torch.save({'feature_names': list(names), **states}, p); p.chmod(0o600)
            saved[label] = {'file': p.name, 'sha256': sha(p), **info}
            if label in ('E2', 'E5', 'E10', 'E20'):
                for i, task in enumerate(D.TASKS):
                    m = neural.network(D.K[task], d); m.load_state_dict(states[task]); m.eval()
                    for role, XX in roles_X.items():
                        outputs[f'NEURAL_RAW_rho{tag}_{label}__{role}__{task}'] = neural.probabilities(m, XX)
        res = nr.run_arm(tr, 'RAW_J', rho, calib['coefficients'], on_ck)
        if res['warm_critic_sha256'] != warm_sha:
            raise AssertionError('warm critics differ across runs')
        record['runs'][tag] = {k: v for k, v in res.items() if k != 'logs'}
        record['checkpoints'][tag] = saved
        with (out / f'NEURAL_RAW__rho{tag}__LOGS.json').open('x') as f:
            json.dump(res['logs'], f)
    op = out / 'NEURAL_RAW_OUTPUTS.npz'
    with op.open('xb') as f:
        np.savez(f, **outputs)
    op.chmod(0o600); record['outputs_sha256'] = sha(op)
    with (out / 'NEURAL_RAW_SEED.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--stage-a', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.stage_a, a.out)
    print(json.dumps({'seed': r['seed'], 'lambda': {k: v['lambda'] for k, v in r['runs'].items()}}))
