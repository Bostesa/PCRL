"""V9 support: Stage-1 controls (label-free; CONTROL REFITS, declared together with the V6 sample before any result).

The campaign saved no control models, so the 12 control fits (2 seeds x 3 views x {shuffled null, planted leak}) are refit
here with the registered layout: OUT_HGB on the H_LJ wire ('out' transform), model seed MODEL_SEED_BASE + s; the shuffled
null is trained on ATTACK_FIT SEX permuted by default_rng(2026101602 + s); the planted reader is trained on [wire, SEX].
Stage 1 recomputes the 6 planted and 6 null ATTACK_VALIDATION AUCs (no orientation) with the verifier's own AUC; the
refit DEV_REPORT control scores are compared with the stored ones (bitwise). A summary (no person-level data) is written
to the scratch directory for the truth job, which reports it under V9.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from acs_hybrids import common as C   # hash-checked data loading only

from . import vcommon as V


def main():
    from task_conditioned_representation.attacks import fit_reader, predict_reader
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    scratch = V.scratch_dir(a.scratch)
    RL = V.reporting_lock()
    A = C.load_role('ATTACK_FIT', ('features', 'sex'))
    Vv = C.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    R = C.load_role('DEV_REPORT', ('features',))
    out = {'scope': 'REFIT of the 12 control readers (OUT_HGB) from scratch; Stage-1 AUCs on ATTACK_VALIDATION; label-free',
           'per_view': {}}
    for s in V.SEEDS:
        bank, ids = C.load_bank(s, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
        with np.load(V.PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz', allow_pickle=False) as z:
            stored = {k: z[k].copy() for k in z.files if k.startswith('CONTROL__')}
        shuffled = np.random.default_rng(V.NULL_SEED + s).permutation(A['sex'])
        seed_model = V.MODEL_SEED_BASE + s
        for view in V.VIEWS:
            xs = {r: V.view_features(bank['H_LJ'][r], view, 'out') for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                null_m = fit_reader(xs['ATTACK_FIT'], shuffled, 'OUT_HGB', seed_model)
                plant_m = fit_reader(np.column_stack((xs['ATTACK_FIT'], A['sex'])), A['sex'], 'OUT_HGB', seed_model)
            null_av = V.auc(Vv['sex'], predict_reader(null_m, xs['ATTACK_VALIDATION']))
            plant_av = V.auc(Vv['sex'], predict_reader(plant_m, np.column_stack((xs['ATTACK_VALIDATION'], Vv['sex']))))
            rep_null = predict_reader(null_m, xs['DEV_REPORT'])
            rep_pl = {b: predict_reader(plant_m, np.column_stack((xs['DEV_REPORT'], np.full(len(xs['DEV_REPORT']), b)))) for b in (0, 1)}
            camp = RL['stage_1_controls']['per_view'][f'S{s}|{view}']
            out['per_view'][f'S{s}|{view}'] = {
                'refit_null_attack_validation_auc': null_av, 'refit_planted_attack_validation_auc': plant_av,
                'campaign_null_attack_validation_auc': camp['shuffled_attack_validation_auc'],
                'campaign_planted_attack_validation_auc': camp['planted_attack_validation_auc'],
                'null_abs_diff': abs(null_av - camp['shuffled_attack_validation_auc']),
                'planted_abs_diff': abs(plant_av - camp['planted_attack_validation_auc']),
                'stored_dev_report_scores_bitwise': {'SHUFFLED': bool(np.array_equal(rep_null, stored[f'CONTROL__{view}__SHUFFLED'])),
                                                     'PLANTED_0': bool(np.array_equal(rep_pl[0], stored[f'CONTROL__{view}__PLANTED_0'])),
                                                     'PLANTED_1': bool(np.array_equal(rep_pl[1], stored[f'CONTROL__{view}__PLANTED_1']))},
                'stored_dev_report_scores_max_abs_diff': max(V.maxabs(rep_null, stored[f'CONTROL__{view}__SHUFFLED']),
                                                             V.maxabs(rep_pl[0], stored[f'CONTROL__{view}__PLANTED_0']),
                                                             V.maxabs(rep_pl[1], stored[f'CONTROL__{view}__PLANTED_1']))}
    pv = out['per_view'].values()
    out['stage_1'] = {'planted_all_ge_0.99': all(np.isfinite(v['refit_planted_attack_validation_auc']) and v['refit_planted_attack_validation_auc'] >= 0.99 for v in pv),
                      'null_all_in_band_0.45_0.55': all(np.isfinite(v['refit_null_attack_validation_auc']) and 0.45 <= v['refit_null_attack_validation_auc'] <= 0.55 for v in pv),
                      'n_planted': len(out['per_view']), 'n_null': len(out['per_view']),
                      'max_abs_diff_vs_REPORTING_LOCK': max(max(v['null_abs_diff'], v['planted_abs_diff']) for v in pv),
                      'stored_dev_report_control_scores_bitwise': sum(sum(v['stored_dev_report_scores_bitwise'].values()) for v in pv),
                      'stored_dev_report_control_score_arrays': 3 * len(out['per_view'])}
    out['stage_1']['CONTROLS_STAGE1_VALID_recomputed'] = bool(out['stage_1']['planted_all_ge_0.99'] and out['stage_1']['null_all_in_band_0.45_0.55'])
    out['stage_1']['campaign_CONTROLS_STAGE1_VALID'] = RL['stage_1_controls']['CONTROLS_STAGE1_VALID']
    V.write_json(scratch / 'v9_stage1_controls.json', out)
    print(out['stage_1'])


if __name__ == '__main__':
    main()
