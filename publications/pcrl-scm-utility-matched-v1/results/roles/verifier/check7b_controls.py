"""Verifier check 7b (additional, not predefined): refit of the reader controls on the SCM wire (own code, source settings).

Per seed and view (income, occupation, pair; 'out' features): shuffled-SEX null (OUT_HGB on a fixed permutation of ATTACK_FIT SEX,
numpy default_rng(2026101602 + s)) and planted SEX bit (OUT_HGB on the wire plus the SEX column). ATTACK_VALIDATION AUCs are
compared with READERS_SEED.json controls and the DEV_REPORT control scores (SHUFFLED, PLANTED_0, PLANTED_1) bitwise with the
stored DEV_REPORT_SCORES.npz. Supports condition CV of the claims. No DEV_REPORT label is read here.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

NULL_SEED = 2026101602
MODEL_SEED_BASE = 2026101205


def out_kind(p):
    return np.column_stack((p, np.argmax(p, 1).astype(np.float64)))


def hgb(x, y, seed):
    mu = x.mean(0); sd = x.std(0); sd = np.where(sd > 0, sd, 1.0)
    m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit((x - mu) / sd, y)
    return lambda z: m.predict_proba((z - mu) / sd)[:, 1].astype(np.float64)


def main():
    t0 = time.process_time()
    sel = V.jload(V.RES / 'SELECTION_LOCK.json')
    A = V.export('ATTACK_FIT', 'features'); yA = V.export('ATTACK_FIT', 'sex')['sex'].astype(np.int64)
    Vf = V.export('ATTACK_VALIDATION', 'features'); yV = V.export('ATTACK_VALIDATION', 'sex')['sex'].astype(np.int64)
    out = {'check': '7b', 'title': 'control reader refits (additional)', 'per_seed': {}}
    for s in V.SEEDS:
        RS = V.jload(V.PRIV / 'runs' / f'READERS_{s}' / 'READERS_SEED.json'); SC = V.npz(V.PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz')
        SB = V.npz(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz'); svc = sel['services'][str(s)]['service_of_role']['SCM']
        assert np.array_equal(SB['row_id__ATTACK_FIT'], A['row_id']) and np.array_equal(SB['row_id__ATTACK_VALIDATION'], Vf['row_id'])
        shuffled = np.random.default_rng(NULL_SEED + s).permutation(yA)
        res = {}
        for view in ('income', 'occupation', 'pair'):
            def tab(r):
                a, b = out_kind(SB[f'{svc}__{r}__income']), out_kind(SB[f'{svc}__{r}__occupation'])
                return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))
            x, xv, xr = tab('ATTACK_FIT'), tab('ATTACK_VALIDATION'), tab('DEV_REPORT')
            ps = hgb(x, shuffled, MODEL_SEED_BASE + s); pp = hgb(np.column_stack((x, yA)), yA, MODEL_SEED_BASE + s)
            sh_av = V.auc_mw(yV, ps(xv)); pl_av = V.auc_mw(yV, pp(np.column_stack((xv, yV))))
            rc = RS['controls'][view]
            res[view] = {'wire_is_SCM': rc['wire'] == svc, 'shuffled_av_own': sh_av, 'shuffled_av_stored': rc['shuffled_attack_validation_auc'],
                         'planted_av_own': pl_av, 'planted_av_stored': rc['planted_attack_validation_auc'],
                         'dev_shuffled_bitwise': bool(np.array_equal(ps(xr), SC[f'CONTROL__{view}__SHUFFLED'])),
                         'dev_planted0_bitwise': bool(np.array_equal(pp(np.column_stack((xr, np.zeros(len(xr))))), SC[f'CONTROL__{view}__PLANTED_0'])),
                         'dev_planted1_bitwise': bool(np.array_equal(pp(np.column_stack((xr, np.ones(len(xr))))), SC[f'CONTROL__{view}__PLANTED_1']))}
        out['per_seed'][f'S{s}'] = res
    vals = [v for r in out['per_seed'].values() for v in r.values()]
    out['status'] = 'PASS' if all(v['wire_is_SCM'] and v['dev_shuffled_bitwise'] and v['dev_planted0_bitwise'] and v['dev_planted1_bitwise']
                                  and abs(v['shuffled_av_own'] - v['shuffled_av_stored']) <= 1e-12 and abs(v['planted_av_own'] - v['planted_av_stored']) <= 1e-12
                                  for v in vals) else 'FAIL_OR_REVIEW'
    out['cpu_s'] = round(time.process_time() - t0, 2)
    V.write_json('check7b_controls.json', out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
