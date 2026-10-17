"""Fit units: fit this campaign's arms for one model seed on MODEL_FIT and write their released outputs for every
development role (exploratory development pilot).

The mechanism itself lives in `structural_pilot.mechanism` (interface: ARMS registry; fit(arm, data, seed) -> model;
release(model, X) -> {'income': (n, 2), 'occupation': (n, 5)} float64 probability vectors). This module only handles
role access, the lock that authorises the fit, output files and hashes.

Access: MODEL_FIT features, both task labels and SEX (SEX only where the arm's spec says it is a training signal);
CALIBRATION features and task labels only for arms whose spec uses calibration; features of MODEL_SELECT, ATTACK_FIT,
ATTACK_VALIDATION and DEV_REPORT only to compute releases (no labels, no SEX).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import common as C

OUT_ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def run(seed, arms, lock_name, out_dir):
    from . import mechanism as M
    if not C.committed(C.RESULTS / lock_name):
        raise PermissionError(f'fits require the committed {lock_name}')
    lock = json.loads((C.RESULTS / lock_name).read_text())
    allowed = set(lock['authorised_arms'])
    if set(arms) - allowed:
        raise PermissionError(f'arms {sorted(set(arms) - allowed)} are not authorised by {lock_name}')
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh fit directory required')
    out.mkdir(parents=True, mode=0o700)
    data = {'MODEL_FIT': C.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex')),
            'CALIBRATION': C.load_role('CALIBRATION', ('features', 'income', 'occupation'))}
    feats = {r: (data[r] if r in data else C.load_role(r, ('features',))) for r in OUT_ROLES}
    rec = {'schema': 'SMP1-fit-v1', 'seed': seed, 'lock': lock_name, 'lock_sha256': C.sha(C.RESULTS / lock_name), 'arms': {}, 'utc_start': C.now()}
    arrays = {}
    for r in OUT_ROLES:
        arrays[f'row_id__{r}'] = feats[r]['row_id']; arrays[f'unit__{r}'] = feats[r]['unit']
    for arm in arms:
        t0 = time.process_time()
        model, info = M.fit(arm, data, seed)
        mp = out / f'MODEL__{arm}.joblib'; joblib.dump(model, mp, compress=3); mp.chmod(0o600)
        for r in OUT_ROLES:
            rel = M.release(model, feats[r]['X'])
            for t in C.TASKS:
                p = np.asarray(rel[t], dtype=np.float64)
                if p.shape != (len(feats[r]['row_id']), C.K[t]) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
                    raise ValueError(f'{arm} {r} {t}: probability convention differs')
                arrays[f'{arm}__{r}__{t}'] = p
        rec['arms'][arm] = {'info': info, 'model_sha256': C.sha(mp), 'cpu_s': round(time.process_time() - t0, 3)}
    p = out / 'ARM_OUTPUTS.npz'
    with p.open('xb') as f:
        np.savez(f, **arrays)
    p.chmod(0o600)
    rec['arm_outputs_sha256'] = C.sha(p); rec['utc_end'] = C.now()
    with (out / 'FIT_SEED.json').open('x') as f:
        json.dump(rec, f, indent=2, sort_keys=True, default=float)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--arms', required=True)
    ap.add_argument('--lock', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.arms.split(','), a.lock, a.out)
    print(json.dumps({'seed': r['seed'], 'arms': {k: {'cpu_s': v['cpu_s']} for k, v in r['arms'].items()}, 'sha': r['arm_outputs_sha256']}, indent=1))
