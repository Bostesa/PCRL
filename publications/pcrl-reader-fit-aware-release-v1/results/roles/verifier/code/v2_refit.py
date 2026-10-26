#!/usr/bin/env python3
"""RFA1 verifier, wave 2, task 4b: bounded refit of winning readers of the main grid RFA1_MAIN with the verifier's own sklearn code
(vlib feature tables, scaler and the 16 locked recipe definitions; vcore.refit_core): the pair winners of RF_FULL, RF_STOP_HI and
NEURAL_RAW in four cells (S0R1k0, S0R2k1, S1R1k0, S1R2k1: both versions, both repetitions, both reader seeds), each fitted on the
cell's FIT households, oriented on VALIDATION and scored on SCORE; compared with the grid's SCORE AUC and the cached oriented SCORE
scores. SEX through rfa.common.load(..., 'verify') from this __main__, in a queue unit.

    python3 -m rfa.jobqueue run --unit RFA_VER_REFIT_v1 --stage verify --rss-gib 3 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_refit.py
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

import vcore  # noqa: E402
import vlib as V  # noqa: E402

OUT = HERE.parent / 'out' / 'V2_REFIT.json'
RUNS = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1' / 'runs'
BANK = 'RFA1_MAIN'
CELLS = {0: ('S0R1k0', 'S0R2k1'), 1: ('S1R1k0', 'S1R2k1')}
MEMBERS = ('RF_FULL', 'RF_STOP_HI', 'NEURAL_RAW')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_sex():
    from rfa import common as C
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               f'verify: bounded sklearn refit of pair winners of RF_FULL, RF_STOP_HI and NEURAL_RAW in four cells of grid {BANK} '
               '(verifier wave 2)')
    S.pop('X', None)
    return S


def member_arrays(grid, version, row_id):
    from rfa import devreaders as DR
    out = {}
    for name in MEMBERS:
        spec = grid['spec']['members'][name]
        if spec.get('type') == 'rfa':
            d = RUNS / 'FAM' / spec['config'] / f'V{version}'
            rec = json.loads((d / 'RELEASE.json').read_text())
            if sha256(d / 'RELEASE.npz') != rec['release_sha256']:
                raise PermissionError(f'{name}/V{version}: RELEASE.npz differs from its record')
            with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
                rel = {t: np.asarray(z[f'MODEL_SELECT__{t}'], dtype=np.float64).copy() for t in V.TASKS}
        else:
            rel, _ = DR.load_member(spec, version, n=len(row_id), row_id=row_id)
        if {t: V.wire_sha(rel[t]) for t in V.TASKS} != grid['members'][name]['wire_sha256']:
            raise PermissionError(f'{name}/V{version}: released wire differs from the grid provenance')
        out[name] = {t: np.asarray(rel[t], dtype=np.float64) for t in V.TASKS}
    return out


def main():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: labels are read only inside an rfa.jobqueue verify unit')
    t0 = time.process_time()
    S = load_sex()
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    res = {'schema': 'RFA1-verifier-refit-v2', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'bank': BANK, 'by_version': {}}
    allr = {}
    for v, cells in CELLS.items():
        gdir = RUNS / 'GRID' / BANK / f'S{v}'
        grid = json.loads((gdir / 'GRID.json').read_text())
        mem = member_arrays(grid, v, row_id)
        picks = [(c, m, 'pair') for c in cells for m in MEMBERS]
        r = vcore.refit_core(gdir, RUNS / 'devgrid_cache', v, sex, unit, mem, picks, max_fits=len(picks))
        res['by_version'][str(v)] = r
        allr.update(r['refits'])
    import scipy
    import sklearn
    res['versions'] = {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}
    res['n_refits'] = len(allr)
    res['max_abs_score_auc_diff'] = max(abs(x['score_auc_diff']) for x in allr.values())
    res['max_abs_val_auc_diff'] = max(abs(x['refit_val_auc'] - x['cached_val_auc']) for x in allr.values())
    res['bitwise_scores'] = sum(x['bitwise_scores'] for x in allr.values())
    res['checks'] = {'refit_SCORE_AUC_equals_grid (abs <= 1e-12)': res['max_abs_score_auc_diff'] <= 1e-12,
                     'refit_VALIDATION_AUC_equals_cached (abs <= 1e-12)': res['max_abs_val_auc_diff'] <= 1e-12,
                     'library_versions_equal_grid_lock': res['versions'] == json.loads(
                         (ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1' / 'DEVGRID_CODE_LOCK.json').read_text())['versions']}
    res['all_pass'] = all(res['checks'].values())
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {f: sha256(HERE / f) for f in ('v2_refit.py', 'vcore.py', 'vlib.py')}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'checks': res['checks'], 'n_refits': res['n_refits'], 'bitwise': res['bitwise_scores'],
                      'cpu_s': res['cpu_s']}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
