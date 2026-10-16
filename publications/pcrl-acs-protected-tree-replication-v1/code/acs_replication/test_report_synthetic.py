"""Engineering test of report.run() on a fully SYNTHETIC private tree (no ACS value is read). Exercises every code path
of the single-opening DEV_REPORT worker before the real opening: endpoints, claims, planning, controls, sensitivity."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    from acs_replication import data as D, registry, report
    rng = np.random.default_rng(11)
    import os
    tmp = Path(tempfile.mkdtemp(prefix='acsr_report_test_', dir=os.environ.get('ACSR_TEST_TMP')))
    priv, results, runs = tmp / 'priv', tmp / 'results', tmp / 'priv' / 'runs'
    for p in (priv, results, runs):
        p.mkdir(parents=True, exist_ok=True)
    n = 3000
    unit = np.repeat(np.arange(1200), [3 if i < 600 else 2 for i in range(1200)])[:n]
    sex = rng.integers(0, 2, n); inc = rng.integers(0, 2, n); occ = rng.integers(0, 5, n)
    R = {'row_id': np.arange(n, dtype=np.int64) + 10, 'unit': unit.astype(np.int64), 'X': np.zeros((n, 3), np.float32),
         'feature_names': ('a', 'b', 'c'), 'income': inc, 'occupation': occ, 'sex': sex, 'weights': rng.integers(5, 200, n)}

    def probs(k, strength):
        z = rng.normal(size=(n, k)); z[np.arange(n), inc if k == 2 else occ] += strength
        return softmax(z)
    services = {'U0': 1.0, 'Ucal': 1.0, 'T_TREE': 1.1, 'LOCAL_TREE': 1.05, 'TASK_REFIT': 1.1, 'JOINT_TREE': 1.0,
                'NEURAL_RAW_rho1.0_E2': 0.9, 'FAGTB_lam0.03': 0.8, 'FARE_F1': 0.5, 'FARE_Z1': 0.6}
    files = {'stage_a': ['U0', 'Ucal', 'T_TREE'], 'stage_b': ['LOCAL_TREE', 'TASK_REFIT', 'JOINT_TREE'], 'neural_raw': ['NEURAL_RAW_rho1.0_E2'],
             'fagtb': ['FAGTB_lam0.03'], 'fare': ['FARE_F1', 'FARE_Z1']}
    for seed in (0, 1):
        for fam, svcs in files.items():
            p = runs / registry.FILES[fam].format(s=seed); p.parent.mkdir(parents=True, exist_ok=True)
            arrays = {f'{s}__DEV_REPORT__{t}': probs(k, services[s]) for s in svcs for t, k in (('income', 2), ('occupation', 5))}
            if fam == 'stage_a':
                arrays['row_id__DEV_REPORT'] = R['row_id']
            np.savez(p, **arrays)
        roster = ['T_TREE', 'LOCAL_TREE', 'TASK_REFIT', 'U0', 'JOINT_TREE', 'NEURAL_RAW_rho1.0_E2', 'FAGTB_lam0.03', 'FARE_F1', 'FARE_Z1']
        rd = runs / f'READERS_{seed}'; rd.mkdir(parents=True, exist_ok=True)
        sc = {'row_id': R['row_id'], 'unit': R['unit']}; svc_rec = {}
        for s in roster:
            svc_rec[s] = {'winner': {}}
            for v in ('income', 'occupation', 'pair'):
                for name in ('OUT_LR', 'OUT_HGB'):
                    sc[f'{s}__{v}__{name}'] = np.clip(0.5 + 0.1 * (sex - 0.5) * (1 if s != 'LOCAL_TREE' else 0.4) + rng.normal(scale=0.2, size=n), 0, 1)
                svc_rec[s]['winner'][v] = {'reader': 'OUT_HGB', 'attack_validation_auc': 0.6}
        for v in ('income', 'occupation', 'pair'):
            sc[f'CONTROL__{v}__SHUFFLED'] = rng.random(n)
            sc[f'CONTROL__{v}__PLANTED_0'] = rng.random(n) * 0.1; sc[f'CONTROL__{v}__PLANTED_1'] = 0.9 + rng.random(n) * 0.1
        np.savez(rd / 'DEV_REPORT_SCORES.npz', **sc)
        (rd / 'READERS_SEED.json').write_text(json.dumps({'services': svc_rec}))
    sel = {'attack_roster': roster, 'frozen': {'NEURAL_RAW': {'status': 'ELIGIBLE', 'configuration': 'NEURAL_RAW_rho1.0_E2'},
                                               'FAGTB': {'status': 'INELIGIBLE_FALLBACK', 'configuration': 'FAGTB_lam0.03', 'two_task_valid': True},
                                               'FARE': {'status': 'INELIGIBLE_FALLBACK', 'configuration': 'FARE_F1', 'matched_compression_twin': 'FARE_Z1'}},
           'fixed_services': {'LOCAL_TREE': {'point_feasible_24': True}}}
    (results / 'SELECTION_LOCK.json').write_text(json.dumps(sel)); (results / 'REPORTING_LOCK.json').write_text('{}')
    D.PRIV, D.RESULTS, registry.RUNS, registry.PRIV = priv, results, runs, priv
    report.ACCESS_LOG = priv / 'access' / 'DEV_REPORT_ACCESS.jsonl'
    D.lock_committed = lambda name: True
    D.load_role = lambda role, wanted=('features',): R
    res = report.run(tmp / 'out')
    second = None
    try:
        report.run(tmp / 'out2')
    except PermissionError as e:
        second = str(e)
    print(json.dumps({'claims': {k: v['decision'] for k, v in res['claims'].items()}, 'roster': res['primary_roster_size'],
                      'second_opening_refused': second is not None, 'sensitivity_disagreements': len(res['sensitivity_disagreements']),
                      'controls_s0': res['per_seed']['0']['controls'], 'planning_claims': list(res['planning']['per_claim'])}, default=str))


if __name__ == '__main__':
    main()
