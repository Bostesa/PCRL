"""One seed: the close published tree baseline (FAGTB) at the locked bounded grid.

FAGTB_BINARY (income): faithful repaired port of the official binary demographic-parity FAGTB (acs_replication.fagtb,
vincent-grari/FAGTB @ 0cd0bb1): probability-input logistic adversary, gradient-descent step on the summed CE before each
tree, pseudo-residual y - p + lam * dL_adv/dF, depth-9 friedman_mse regression trees (the depth the official code
actually uses), notebook n_estimators / learning rate / max_features, official TF-seed-7 adversary initialisation.
FAGTB_MC_ADAPTATION (occupation): the documented softmax extension (NOT an official implementation): K trees per
iteration on the multinomial pseudo-residuals, SEX adversary on the K-probability vector.
Access: MODEL_FIT task labels and SEX only (no other role's labels; no test-label argument exists).
Seed pairing: seed s sets every tree's random_state to s (the official code uses 0 for all trees = seed 0); the
adversary initialisation is the official one for both seeds.
Released wire: each model's probability vector and argmax decision.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import fagtb as FG

ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(seed, grid_path, out_dir):
    import joblib
    if not D.lock_committed('SCIENCE_LOCK.json'):
        raise PermissionError('FAGTB fits require the committed SCIENCE_LOCK')
    grid = json.loads(Path(grid_path).read_text())
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh FAGTB directory required')
    out.mkdir(parents=True, mode=0o700)
    F = D.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    roles = {r: (F if r == 'MODEL_FIT' else D.load_role(r, ('features',))) for r in ROLES}
    X = F['X'].astype(np.float64)
    hp = grid['fixed']
    rec = {'schema': 'ACSR1-fagtb-seed-v1', 'seed': seed, 'grid': grid, 'units': {}, 'binary_valid': True, 'mc_valid': True,
           'access': 'MODEL_FIT task labels and SEX only', 'DEV_REPORT_labels_opened': False}
    outputs = {}
    for lam in grid['lambda']:
        tag = f'FAGTB_lam{lam}'
        unit = {}
        for task in D.TASKS:
            t0 = time.process_time()
            try:
                if task == 'income':
                    m = FG.FAGTBBinary(n_estimators=hp['n_estimators'], learning_rate=hp['learning_rate'], max_depth=hp['max_depth'],
                                       max_features=hp['max_features'], adversary_lr=hp['adversary_lr'],
                                       adversary_init=tuple(FG.OFFICIAL_TF_SEED7_ADVERSARY_INIT), tree_random_state=seed)
                else:
                    m = FG.FAGTBMulticlassAdaptation(n_estimators=hp['n_estimators'], learning_rate=hp['learning_rate'], max_depth=hp['max_depth'],
                                                     max_features=hp['max_features'], adversary_lr=hp['adversary_lr'], seed=seed,
                                                     tree_random_state=seed, n_classes=D.K['occupation'])
                m.fit(X, F[task], F['sex'], float(lam))
                for role, R in roles.items():
                    P = np.asarray(m.predict_proba(R['X'].astype(np.float64)), dtype=np.float64)
                    if P.shape != (len(R['X']), D.K[task]) or not np.all(np.isfinite(P)) or not np.allclose(P.sum(1), 1, atol=1e-9):
                        raise ValueError(f'invalid probabilities for {task}')
                    outputs[f'{tag}__{role}__{task}'] = P
                mp = out / f'{tag}__{task}.joblib'; joblib.dump(m, mp, compress=3); mp.chmod(0o600)
                hist = getattr(m, 'history_', None)
                unit[task] = {'cpu_s': time.process_time() - t0, 'model_sha256': sha(mp), 'n_trees': hp['n_estimators'] * (1 if task == 'income' else D.K[task]),
                              'max_depth': hp['max_depth'], 'final_history': (hist[-1] if isinstance(hist, list) and hist else None), 'valid': True}
            except Exception as e:  # recorded honestly; an invalid occupation adaptation makes the two-task comparison incomplete
                unit[task] = {'valid': False, 'error': repr(e), 'cpu_s': time.process_time() - t0}
                rec['binary_valid' if task == 'income' else 'mc_valid'] = False
        rec['units'][tag] = unit
    op = out / 'FAGTB_OUTPUTS.npz'
    if rec['binary_valid'] and rec['mc_valid']:
        with op.open('xb') as f:
            np.savez(f, **outputs)
        op.chmod(0o600); rec['outputs_sha256'] = sha(op)
    else:
        rec['outputs_sha256'] = None
        rec['note'] = 'two-task FAGTB service incomplete: no combined release file written (an unprotected occupation teacher is never substituted)'
        with (out / 'FAGTB_PARTIAL_OUTPUTS.npz').open('xb') as f:
            np.savez(f, **outputs)
    with (out / 'FAGTB_SEED.json').open('x') as f:
        json.dump(rec, f, indent=2, sort_keys=True, default=str)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--grid', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.grid, a.out)
    print(json.dumps({'seed': r['seed'], 'binary_valid': r['binary_valid'], 'mc_valid': r['mc_valid'],
                      'cpu': {k: {t: v[t].get('cpu_s') for t in v} for k, v in r['units'].items()}}))
