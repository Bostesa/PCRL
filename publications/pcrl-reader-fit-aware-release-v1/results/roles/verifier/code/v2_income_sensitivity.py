#!/usr/bin/env python3
"""RFA1 verifier, wave 2 (descriptive; never a label): sensitivity of the decisive START comparison to the income-wire mismatch found
by RFA_VER_USEFULNESS_v1 (RF_* released income = START's network evaluated in float64; the frozen bank's NEURAL_RAW income differs by
<= 6.8e-8, so the grid fits START's readers on a slightly different income wire).

Per registered cell of RFA1_MAIN (8): NEURAL_RAW's pair winner (recipe and view fixed as the grid selected it on VALIDATION) refitted
with the verifier's own sklearn code (vlib.refit) on (a) the bank NEURAL_RAW income + occupation (control: must reproduce the grid's
SCORE AUC) and (b) RF_FULL's released income + NEURAL_RAW's occupation (the income-matched START); Delta = AUC(b) - AUC(a). The START
pair benefit with the income-matched START, b'_cell = b_cell + Delta_cell (cell SEs unchanged), is aggregated with vagg (descriptive).
SEX through rfa.common.load(..., 'verify') from this __main__, in a queue unit.

    python3 -m rfa.jobqueue run --unit RFA_VER_INCOME_v1 --stage verify --rss-gib 3 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_income_sensitivity.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vagg as A  # noqa: E402
import vcore  # noqa: E402
import vlib as V  # noqa: E402

OUT = HERE.parent / 'out' / 'V2_INCOME_SENSITIVITY.json'
RUNS = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1' / 'runs'
BANK = 'RFA1_MAIN'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: labels are read only inside an rfa.jobqueue verify unit')
    t0 = time.process_time()
    from rfa import common as C
    from rfa import devreaders as DR
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               'verify: descriptive sensitivity of the RF_FULL vs NEURAL_RAW pair comparison to the float-rounding income-wire mismatch '
               '(START pair winner refitted on the bank income and on RF_FULL income; verifier wave 2)')
    S.pop('X', None)
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    sx = V.arr_sha(np.asarray(sex))
    res = {'schema': 'RFA1-verifier-income-sensitivity-v2', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'cells': {}}
    b_cells, b_se, b_new = {}, {}, {}
    for v in (0, 1):
        gdir = RUNS / 'GRID' / BANK / f'S{v}'
        grid = json.loads((gdir / 'GRID.json').read_text())
        nr, _ = DR.load_member(grid['spec']['members']['NEURAL_RAW'], v, n=len(row_id), row_id=row_id)
        if {t: V.wire_sha(nr[t]) for t in V.TASKS} != grid['members']['NEURAL_RAW']['wire_sha256']:
            raise PermissionError('NEURAL_RAW wire differs from the grid provenance')
        d = RUNS / 'FAM' / 'RF_FULL' / f'V{v}'
        rec = json.loads((d / 'RELEASE.json').read_text())
        if sha256(d / 'RELEASE.npz') != rec['release_sha256']:
            raise PermissionError('RF_FULL RELEASE.npz differs from its record')
        with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
            rf_inc = np.asarray(z['MODEL_SELECT__income'], dtype=np.float64).copy()
        nr = {t: np.asarray(nr[t], dtype=np.float64) for t in V.TASKS}
        for cell in sorted(grid['cells']):
            ctx = vcore._cell_ctx(cell, sex, unit)
            E = vcore._entries(RUNS / 'devgrid_cache', ctx, nr, sx)
            w = V.winners({vw: {r: E[vw][r][1] for r in V.RECIPES} for vw in V.VIEWS})['pair']
            vw, recipe = w['reader_key'].split('__', 1)
            fit, val, sc = ctx['roles']['FIT'], ctx['roles']['VALIDATION'], ctx['score']
            s_a, va_a, _ = V.refit(recipe, vw, nr['income'], nr['occupation'], sex, fit, val, ctx['rs'])
            s_b, va_b, _ = V.refit(recipe, vw, rf_inc, nr['occupation'], sex, fit, val, ctx['rs'])
            a_a, a_b = V.auc(ctx['y'], s_a[sc]), V.auc(ctx['y'], s_b[sc])
            g = grid['cells'][cell]['NEURAL_RAW']['pair']
            pb = grid['paired']['RF_FULL']['NEURAL_RAW']['pair']
            b_cells[cell], b_se[cell] = pb['cells'][cell], pb['cell_se'][cell]
            b_new[cell] = pb['cells'][cell] + (a_b - a_a)
            res['cells'][cell] = {'winner': g['winner'], 'recipe': recipe, 'view': vw, 'grid_score_auc': g['score_auc'],
                                  'control_refit_auc': a_a, 'control_reproduces_grid': bool(a_a == g['score_auc']),
                                  'income_matched_auc': a_b, 'delta': a_b - a_a, 'val_auc_control': va_a, 'val_auc_income_matched': va_b,
                                  'max_abs_score_change': float(np.max(np.abs(s_b[sc] - s_a[sc])))}
    base = A.benefit_screen({'cells': b_cells, 'cell_se': b_se})
    alt = A.benefit_screen({'cells': b_new, 'cell_se': b_se})
    summ = lambda x: {'label': x['label'], 'mean': x['aggregate']['mean'], 'se_agg': x['aggregate']['se_agg'], 'lower': x['aggregate']['lower'],
                      'version_means': {str(v): x['aggregate']['per_version'][v]['mean'] for v in (0, 1)}}
    res['start_pair_benefit_grid'] = summ(base)
    res['start_pair_benefit_income_matched_start_winner_fixed'] = summ(alt)
    res['deltas'] = {c: x['delta'] for c, x in res['cells'].items()}
    res['mean_abs_delta'] = float(np.mean([abs(x) for x in res['deltas'].values()]))
    res['all_controls_reproduce_grid'] = all(x['control_reproduces_grid'] for x in res['cells'].values())
    res['note'] = ('descriptive: the winner and orientation-selection are held at the grid\'s; cell SEs unchanged; never a label, a '
                   'selection or a replacement of the registered comparison')
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {f: sha256(HERE / f) for f in ('v2_income_sensitivity.py', 'vlib.py', 'vcore.py', 'vagg.py')}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({k: res[k] for k in ('start_pair_benefit_grid', 'start_pair_benefit_income_matched_start_winner_fixed', 'deltas',
                                          'all_controls_reproduce_grid', 'cpu_s')}, indent=1, default=float))
    return 0 if res['all_controls_reproduce_grid'] else 1


if __name__ == '__main__':
    sys.exit(main())
