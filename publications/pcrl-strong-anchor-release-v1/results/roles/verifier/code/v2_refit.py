"""Verifier wave 2 unit (label read; the grid's locked torch interpreter): refit a random sample of winning readers from scratch.

Frames (stated before drawing): (1) every registered-cell winner of SAR1_FA_R2 (members other than U0 / Ucal), 6 drawn; (2) every
robustness-cell winner (R3 / R4, or k2 / k3) of SAR1_FB_R1_ROB, 2 drawn; numpy.random.default_rng(SEED) without replacement, frame 1
first. Refit, orientation and scoring as wave 1 (v_refit.refit_one: the locked recipe modules through the verifier's dispatch; the
verifier's own split rule, reader seeds and AUC). Also the seed-effectiveness spot check on the FA_R2 minimum-set wires.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
import v_ms as W1  # noqa: E402
import v2_ms as W2  # noqa: E402
import v_refit as R1  # noqa: E402

import numpy as np  # noqa: E402
from sar import common as C  # noqa: E402

SEED = 20261025
REG = ('S0R1k0', 'S0R1k1', 'S0R2k0', 'S0R2k1', 'S1R1k0', 'S1R1k1', 'S1R2k0', 'S1R2k1')
MIN_SET = ('FA_TFRB_R2', 'SUCC', 'FA_TFRO_R1_CONT', 'FA_R2_CTRL_LO', 'FA_R2_CTRL_HI')
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')


def members_of(bank):
    spec = json.loads((V.RES / 'grid_specs' / f'{bank}.json').read_text())
    m = spec['members']
    if isinstance(m, list):
        main = json.loads((V.RES / 'grid_specs' / f"{bank.replace('_ROB', '')}.json").read_text())['members']
        m = {n: main[n] for n in m}
    return m


def arrays(m, v):
    W2.VCUR[0] = v
    t = m['type']
    if t == 'sar':
        r = W1.load_sar(m['family'], m['config'], v)
    elif t == 'arrays':
        r = W2.array_member(m['seeds'][str(v)])
    elif t == 'reference':
        r = W1.load_reference({'H3+TASK_REFIT': 'TR100', 'H3+T_TREE': 'TT100'}.get(m['name'], m['name']), v)
    else:
        r = W1.load_reference('W', v)
    return {k: np.ascontiguousarray(np.asarray(r[k], dtype=np.float64)) for k in V.TASKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               'independent verifier wave 2: refit a random sample (seed 20261025) of winning readers of SAR1_FA_R2 (6) and of the '
               'SAR1_FB_R1_ROB robustness cells (2); seed-effectiveness spot check on the FA_R2 minimum set')
    sex, unit = np.asarray(S['sex']), np.asarray(S['unit'])
    del S
    sex_hash = V.ahash(sex)
    cache = V.RUNS / 'devgrid_cache'
    frames, grids, mems = {'SAR1_FA_R2': [], 'SAR1_FB_R1_ROB': []}, {}, {}
    for bank in frames:
        mems[bank] = members_of(bank)
        for v in (0, 1):
            gp = V.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json'
            W1.note_input(gp)
            grids[(bank, v)] = json.loads(gp.read_text())
            for cell in sorted(grids[(bank, v)]['cells']):
                if (bank == 'SAR1_FA_R2') != (cell in REG):
                    continue
                for name in sorted(grids[(bank, v)]['cells'][cell]):
                    if name in ('U0', 'Ucal'):
                        continue
                    for view in ('income', 'occupation', 'pair'):
                        frames[bank].append((bank, v, cell, name, view))
    rng = np.random.default_rng(SEED)
    picks = {'SAR1_FA_R2': sorted(rng.choice(len(frames['SAR1_FA_R2']), size=6, replace=False).tolist()),
             'SAR1_FB_R1_ROB': sorted(rng.choice(len(frames['SAR1_FB_R1_ROB']), size=2, replace=False).tolist())}
    out = {'schema': 'SAR1-verifier-refit2-v1', 'utc_start': C.now(), 'seed': SEED, 'interpreter': sys.executable,
           'versions': R1.versions(), 'frame_sizes': {b: len(f) for b, f in frames.items()}, 'sample_indices': picks,
           'refits': [], 'seed_effectiveness': []}
    for bank, idxs in picks.items():
        for idx in idxs:
            _, v, cell, name, view = frames[bank][idx]
            rep, k = cell[2:4], int(cell[5:])
            e = grids[(bank, v)]['cells'][cell][name][view]
            w = e['winner']
            wview, recipe = (('income', w.split('__', 1)[1]) if w.startswith('IGNORE_occupation__') else
                             ('occupation', w.split('__', 1)[1]) if w.startswith('IGNORE_income__') else (view, w))
            r = arrays(mems[bank][name], v)
            roles = V.split_roles(unit, rep)
            rs = V.reader_seed(v, k)
            t1 = time.process_time()
            s, sign = R1.refit_one(recipe, wview, r['income'], r['occupation'], sex, roles['FIT'], roles['VALIDATION'], rs)
            cpu = time.process_time() - t1
            key = V.cache_key(rep, rs, 1.0, V.view_hash(wview, r['income'], r['occupation']), sex_hash)
            with np.load(cache / f'{key}.npz', allow_pickle=False) as z:
                stored = z[f'score__{recipe}'].copy()
                info = json.loads(str(z['info']))[recipe]
            rec = {'bank': bank, 'version': v, 'cell': cell, 'member': name, 'view': view, 'winner': w, 'refit_view': wview,
                   'recipe': recipe, 'reader_seed': rs, 'orientation': sign, 'orientation_record': info.get('orientation'),
                   'val_auc_refit': V.auc_weighted(sex[roles['VALIDATION']], s[roles['VALIDATION']]), 'val_auc_record': e['val_auc'],
                   'score_auc_refit': V.auc_weighted(sex[roles['SCORE']], s[roles['SCORE']]), 'score_auc_record': e['score_auc'],
                   'score_vector_bitwise_equal_to_stored': bool(np.array_equal(stored, s[roles['SCORE']])),
                   'max_abs_score_diff': float(np.abs(stored - s[roles['SCORE']]).max()), 'cpu_s': cpu}
            out['refits'].append(rec)
            print(json.dumps(rec), flush=True)
    for v in (0, 1):
        for name in MIN_SET:
            r = arrays(mems['SAR1_FA_R2'][name], v)
            for rep in ('R1', 'R2'):
                for view in ('income', 'occupation', 'pair'):
                    vh = V.view_hash(view, r['income'], r['occupation'])
                    ent = []
                    for k in (0, 1):
                        with np.load(cache / f'{V.cache_key(rep, V.reader_seed(v, k), 1.0, vh, sex_hash)}.npz') as z:
                            ent.append({x: z[f'score__{x}'].copy() for x in SEED_DEP})
                    out['seed_effectiveness'].append({'v': v, 'member': name, 'rep': rep, 'view': view,
                                                      'equal_k0_k1': [x for x in SEED_DEP if np.array_equal(ent[0][x], ent[1][x])]})
    out['inputs_sha256'] = W1.INPUTS
    out['cpu_s'] = time.process_time() - t0
    out['utc_end'] = C.now()
    V.jdump(Path(a.out), out)
    print(json.dumps({'refits': len(out['refits']), 'bitwise': [x['score_vector_bitwise_equal_to_stored'] for x in out['refits']],
                      'seed_equalities': sum(bool(x['equal_k0_k1']) for x in out['seed_effectiveness']), 'cpu_s': out['cpu_s']}))


if __name__ == '__main__':
    main()
