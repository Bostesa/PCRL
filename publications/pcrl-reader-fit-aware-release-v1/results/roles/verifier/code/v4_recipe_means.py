#!/usr/bin/env python3
"""RFA1 verifier, wave 4 (b): claims/PER_RECIPE_PAIR_MEANS.json recomputed from the saved main grids (label-free).

Per bank (RFA1_MAIN, RFA1_R1_MAIN), member and recipe: the mean of the pair view's per_recipe_score_auc over the 8 registered cells
(S<v>R<r>k<k>, v 0-1, R1-R2, k0-k1) read from GRID.json S0 / S1. (The verifier reproduced every per_recipe_score_auc of both banks from
the cache with its own AUC code in waves 2 and 3: max abs diff 0.0.) Falls against START: mean(candidate) - mean(NEURAL_RAW), same
bank; the R1 card prediction (ET25 / CATI1 / HGB15 fall more than under v1) judged per recipe as R1 fall < v1 fall, met iff all three.

    <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v4_recipe_means.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RUNS = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1' / 'runs' / 'GRID'
REC = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1' / 'claims' / 'PER_RECIPE_PAIR_MEANS.json'
OUT = HERE.parent / 'out' / 'V4_RECIPE_MEANS.json'
REG = tuple(f'S{v}{r}k{k}' for v in (0, 1) for r in ('R1', 'R2') for k in (0, 1))
RECIPES = ('CATI1', 'ET25', 'HGB15', 'LOG_LR', 'MLP64')
PRED = ('ET25', 'CATI1', 'HGB15')


def means(bank, members):
    cells = {}
    for v in (0, 1):
        cells.update(json.loads((RUNS / bank / f'S{v}' / 'GRID.json').read_text())['cells'])
    if sorted(c for c in cells if c in REG) != sorted(REG):
        raise ValueError(f'{bank}: registered cells missing')
    return {m: {r: sum(cells[c][m]['pair']['per_recipe_score_auc'][r] for c in REG) / len(REG) for r in RECIPES} for m in members}


def main():
    t0 = time.process_time()
    rec = json.loads(REC.read_text())
    v1 = means('RFA1_MAIN', ('NEURAL_RAW', 'RF_FULL', 'RF_STOP_HI'))
    r1 = means('RFA1_R1_MAIN', ('NEURAL_RAW', 'RF1_FULL', 'RF1_STOP_HI', 'RF_FULL'))
    falls = {'v1 (RFA1_MAIN)': {r: v1['RF_FULL'][r] - v1['NEURAL_RAW'][r] for r in RECIPES},
             'R1 (RFA1_R1_MAIN)': {r: r1['RF1_FULL'][r] - r1['NEURAL_RAW'][r] for r in RECIPES},
             'v1 candidate in the R1 bank': {r: r1['RF_FULL'][r] - r1['NEURAL_RAW'][r] for r in RECIPES}}
    pred = {r: {'R1_fall': falls['R1 (RFA1_R1_MAIN)'][r], 'v1_fall': falls['v1 (RFA1_MAIN)'][r],
                'R1_fell_more': falls['R1 (RFA1_R1_MAIN)'][r] < falls['v1 (RFA1_MAIN)'][r]} for r in PRED}
    met = all(x['R1_fell_more'] for x in pred.values())
    chk, worst = {}, 0.0
    for bank, mine in (('v1_bank', v1), ('R1_bank', r1)):
        for m, d in mine.items():
            for r, x in d.items():
                y = rec[bank][m]['mean'][r]
                ok = round(x, 5) == y
                worst = max(worst, abs(round(x, 5) - y))
                chk[f'{bank}|{m}|{r}_mean_equals_record_at_5_decimals'] = ok
            chk[f'{bank}|{m}|n_cells_8'] = rec[bank][m]['n_cells'] == 8
    for k, d in falls.items():
        for r, x in d.items():
            chk[f'fall|{k}|{r}_equals_record_at_5_decimals'] = round(x, 5) == rec['fall_vs_START'][k][r]
    for r, x in pred.items():
        rr = rec['R1_card_prediction_ET25_CATI1_HGB15_fall_more'][r]
        chk[f'prediction|{r}_fell_more_flag_equals_record'] = x['R1_fell_more'] == rr['R1_fell_more']
    chk['prediction_met_equals_record'] = met == rec['prediction_met']
    res = {'schema': 'RFA1-verifier-recipe-means-v4', 'v1_bank': v1, 'R1_bank': r1, 'falls_vs_START': falls, 'prediction': pred,
           'prediction_met': met, 'worst_rounded_abs_diff': worst, 'checks': chk, 'all_pass': all(chk.values()),
           'cpu_s': time.process_time() - t0}
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'prediction_met': met,
                      'prediction': pred, 'falls': {k: {r: round(x, 5) for r, x in d.items()} for k, d in falls.items()}}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
