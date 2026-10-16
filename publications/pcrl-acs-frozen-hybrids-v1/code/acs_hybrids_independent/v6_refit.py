"""V6 Fixed refit sample (REFITS FROM SCRATCH; declared before any verifier result was seen; never changed).

Declared sample (pair-view winners as recorded in REPORTING_LOCK):
  H_LJ seed 0, H_LJ seed 1, H_LL seed 0, TASK_REFIT seed 0, NEURAL_RAW seed 1.
Each winner is refit from scratch on ATTACK_FIT with SEX through task_conditioned_representation.attacks.fit_reader
(reader definitions must match) with model seed MODEL_SEED_BASE + s, oriented on ATTACK_VALIDATION with the verifier's
own AUC, and used to score DEV_REPORT from released outputs only. Compared with: the stored DEV_REPORT scores (bitwise /
max abs diff), the saved model's predictions (replay), and the READERS_SEED ATTACK_VALIDATION AUC.
"""
from __future__ import annotations

import argparse
import time
import warnings

import joblib
import numpy as np

from acs_hybrids import common as C   # hash-checked data loading only

from . import vcommon as V

SAMPLE = (('H_LJ', 0), ('H_LJ', 1), ('H_LL', 0), ('TASK_REFIT', 0), ('NEURAL_RAW', 1))


def main():
    from task_conditioned_representation.attacks import fit_reader, predict_reader
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(V.OUT / 'V6_refit_sample.json'))
    a = ap.parse_args()
    RL = V.reporting_lock()
    A = C.load_role('ATTACK_FIT', ('features', 'sex'))
    Vv = C.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    R = C.load_role('DEV_REPORT', ('features',))
    out = {'check': 'V6 Fixed refit sample',
           'scope': 'REFIT from scratch for the 5 declared pair winners (fit_reader on ATTACK_FIT with SEX, seed MODEL_SEED_BASE + s); '
                    'saved-model and stored-score comparisons are replays',
           'declared_sample': [f'{svc}|s{s}|pair' for svc, s in SAMPLE], 'refits': {}}
    ok = True
    for svc, s in SAMPLE:
        bank, ids = C.load_bank(s, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
        assert np.array_equal(ids['row_id__ATTACK_FIT'], A['row_id']) and np.array_equal(ids['row_id__DEV_REPORT'], R['row_id'])
        w = RL['readers'][f'S{s}']['winners'][svc]['pair']
        rid, name = w['reader_id'], w['reader_id'].split('__')[-1]
        kind = V.reader_kind(name)
        if rid.startswith('PAIR__'):
            xs = {r: V.view_features(bank[svc][r], 'pair', kind) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
        else:   # an ignore-one alias: refit that marginal reader on its single recipient wire
            t = rid.split('__')[1].split('|')[1]
            xs = {r: V.view_features(bank[svc][r], t, kind) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
        seed_model = V.MODEL_SEED_BASE + s
        t0 = time.process_time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_reader(xs['ATTACK_FIT'], A['sex'], name, seed_model)
        cpu = time.process_time() - t0
        raw_v = predict_reader(m, xs['ATTACK_VALIDATION'])
        a_raw = V.auc(Vv['sex'], raw_v)
        sign = 1 if a_raw >= 0.5 else -1
        a_or = V.auc(Vv['sex'], raw_v if sign == 1 else 1.0 - raw_v)
        raw_r = predict_reader(m, xs['DEV_REPORT'])
        s_r = raw_r if sign == 1 else 1.0 - raw_r
        rd = V.PRIV / 'runs' / f'READERS_{s}'
        with np.load(rd / 'DEV_REPORT_SCORES.npz', allow_pickle=False) as z:
            stored = z[rid].copy()
        rs = V.jload(rd / 'READERS_SEED.json')
        ref = rs['services'][svc]['pair_readers'][name] if rid.startswith('PAIR__') else rs['marginal_banks'][rid.split('__')[1]]['readers'][name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            saved = joblib.load(rd / f'MODEL__{rid}.joblib')
        sv_v = V.saved_predict(saved, xs['ATTACK_VALIDATION'])
        sv_r = V.saved_predict(saved, xs['DEV_REPORT'])
        params = {}
        mod, smod = m['model'], saved['model']
        if hasattr(mod, 'coefs_'):
            params['mlp_weights_bitwise_equal_saved'] = all(np.array_equal(x, y) for x, y in zip(mod.coefs_ + mod.intercepts_, smod.coefs_ + smod.intercepts_))
            params['n_iter_refit'], params['n_iter_saved'] = int(mod.n_iter_), int(smod.n_iter_)
        elif hasattr(mod, 'coef_'):
            params['lr_coef_bitwise_equal_saved'] = bool(np.array_equal(mod.coef_, smod.coef_) and np.array_equal(mod.intercept_, smod.intercept_))
        params['scaler_moments_equal_saved'] = bool(np.array_equal(m['scaler'].mean_, saved['scaler'].mean_) and np.array_equal(m['scaler'].scale_, saved['scaler'].scale_))
        r = {'reader_id': rid, 'reader': name, 'model_seed': seed_model, 'train_rows': int(len(A['sex'])),
             'refit_cpu_s': round(cpu, 3),
             'refit_attack_validation_auc': a_or, 'campaign_attack_validation_auc': ref['attack_validation_auc'],
             'attack_validation_auc_abs_diff': abs(a_or - ref['attack_validation_auc']),
             'refit_orientation': sign, 'campaign_orientation': ref['orientation'],
             'refit_vs_stored_dev_report_scores': {'bitwise': bool(np.array_equal(s_r, stored)), 'max_abs_diff': V.maxabs(s_r, stored)},
             'refit_vs_saved_model_predictions': {'attack_validation_bitwise': bool(np.array_equal(raw_v, sv_v)),
                                                  'dev_report_bitwise': bool(np.array_equal(raw_r, sv_r)),
                                                  'max_abs_diff': max(V.maxabs(raw_v, sv_v), V.maxabs(raw_r, sv_r))},
             'parameters': params}
        r['pass'] = bool(r['attack_validation_auc_abs_diff'] <= 1e-12 and sign == ref['orientation']
                         and r['refit_vs_stored_dev_report_scores']['max_abs_diff'] <= 1e-12)
        ok &= r['pass']
        out['refits'][f'{svc}|s{s}'] = r
    out['bitwise_refits'] = sum(v['refit_vs_stored_dev_report_scores']['bitwise'] for v in out['refits'].values())
    out['status'] = 'PASS' if ok else 'FAIL'
    V.write_json(a.out, out)
    print({'status': out['status'], 'bitwise': out['bitwise_refits'],
           'maxdiff': {k: v['refit_vs_stored_dev_report_scores']['max_abs_diff'] for k, v in out['refits'].items()}})


if __name__ == '__main__':
    main()
