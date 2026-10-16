"""Engineering test of selection.run() and readers.run() on a SYNTHETIC private tree (no ACS value is read)."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    from acs_replication import data as D, registry, selection, readers
    rng = np.random.default_rng(5)
    tmp = Path(tempfile.mkdtemp(prefix='acsr_sel_test_', dir=os.environ.get('ACSR_TEST_TMP')))
    priv, results, runs = tmp / 'priv', tmp / 'results', tmp / 'priv' / 'runs'
    for p in (priv, results, runs):
        p.mkdir(parents=True, exist_ok=True)
    roles = {}
    for role, n in (('MODEL_FIT', 1500), ('MODEL_SELECT', 800), ('ATTACK_FIT', 800), ('ATTACK_VALIDATION', 400), ('DEV_REPORT', 600)):
        unit = np.arange(n) // 2
        roles[role] = {'row_id': np.arange(n, dtype=np.int64) + 1000 * len(roles), 'unit': unit.astype(np.int64), 'X': np.zeros((n, 3), np.float32),
                       'feature_names': ('a', 'b', 'c'), 'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n), 'sex': rng.integers(0, 2, n)}
    svcs = {'stage_a': ['U0', 'Ucal', 'T_TREE'], 'stage_b': ['LOCAL_TREE', 'TASK_REFIT'],
            'neural_raw': [f'NEURAL_RAW_rho{r}_E{e}' for r in (1.0, 3.0, 10.0) for e in (2, 5, 10, 20)],
            'fagtb': [f'FAGTB_lam{l}' for l in (0.03, 0.06, 0.15)], 'fare': [f'FARE_{x}{i}' for x in 'FZ' for i in (1, 2, 3)]}
    for seed in (0, 1):
        for fam, names in svcs.items():
            p = runs / registry.FILES[fam].format(s=seed); p.parent.mkdir(parents=True, exist_ok=True)
            arr = {}
            for s in names:
                for role, R in roles.items():
                    for t, k in (('income', 2), ('occupation', 5)):
                        z = rng.normal(size=(len(R['row_id']), k)); z[np.arange(len(z)), R[t]] += (1.0 if s != 'FARE_F1' else 0.2)
                        arr[f'{s}__{role}__{t}'] = softmax(z)
            np.savez(p, **arr)
        (runs / f'FAGTB_{seed}' / 'FAGTB_SEED.json').write_text(json.dumps({'binary_valid': True, 'mc_valid': True}))
    D.PRIV, D.RESULTS, registry.RUNS, registry.PRIV = priv, results, runs, priv
    D.lock_committed = lambda name: True
    D.load_role = lambda role, wanted=('features',): roles[role]
    lock = selection.run(results / 'SELECTION_LOCK.json', results / 'SELECTION_DETAIL.json')
    rec = readers.run(0, runs / 'READERS_0')
    print(json.dumps({'frozen': {k: (v['status'], v['configuration']) for k, v in lock['frozen'].items()}, 'roster': lock['attack_roster'],
                      'fixed': {k: v['point_feasible_24'] for k, v in lock['fixed_services'].items()},
                      'reader_services': list(rec['services']), 'controls': rec['controls'], 'fits': rec['numerical_fit_count']}, default=str))


if __name__ == '__main__':
    main()
