"""Verifier stage V5f: the deployed (full-pool, S100) correction recomputed end to end for one version.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Labels: MODEL_FIT occupation (and SEX for the critics) of S100 through tsz.common.load operation 'verify'.

  base       (python3)  W100_v's 5-fold household cross-fitted out-of-fold probabilities on MODEL_FIT, by the verifier's own code (own
                        folds SHA-256('OHN1_crossfit|<household>') and inner folds 'OHN_A_cv_S<v>' restricted per pipeline, own T_c, HGB from
                        the A1_PO_16 constants); vs runs/CORRECT/V<v>/BASE/BASE.npz (oof, fold) and the source A1_PO_16 S<v>
                        CROSSFIT.npz (MODEL_FIT__oof, fold), bitwise
  direction  (torch)    routing = the admitted T_TREE occupation teacher of seed v (file hash checked; own oblivious routing of all
                        MODEL_FIT people); the source P2 recipe ohn.protect.train_correction (guard none, fixed targets, seed 'TSZ1_V<v>',
                        fold 'full') on the verifier's own base for CONT, LOC, PAIR; theta hash vs DIR/<arm>/DIR.json

    python3 .../v5_fullpool.py base --version 0 --tag V1
    <torch venv python> .../v5_fullpool.py direction --version 0 --tag V1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
CORRECT = PRIV / 'runs' / 'CORRECT'
OHN_RUNS = ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
SCRATCH = PRIV / 'verifier_scratch'


def base(v, tag):
    from v5_nestedfold import tstar, hgb_fit
    t0 = time.process_time()
    F = C.load('MODEL_FIT', ('features', 'occupation'), 'verify',
               f'independent verifier V5: recompute the full-pool cross-fitted witness base of version {v} with own code', subset='S100')
    X, names, y, unit = np.asarray(F['X']), [str(x) for x in F['feature_names']], np.asarray(F['occupation'], dtype=np.int64), np.asarray(F['unit'])
    Xr = V.compact_po(X, names)                                   # S100: the FIT-only dictionary collapses no level (verified in V3)
    cf = V.folds(unit, 5, 'OHN1_crossfit')
    inner = V.folds(unit, 5, f'OHN_A_cv_S{v}')
    oof = np.full((len(y), 5), np.nan)
    T_c = {}
    for c in range(5):
        rows = cf != c
        T_c[c], _ = tstar(Xr[rows], y[rows], inner[rows], v)
        m = hgb_fit(Xr[rows], y[rows], T_c[c], v)
        oof[cf == c] = V.softmax(np.asarray(m.decision_function(Xr[cf == c]), dtype=np.float64))
    d = CORRECT / f'V{v}' / 'BASE'
    brec = json.loads((d / 'BASE.json').read_text())
    if V.file_sha(d / 'BASE.npz') != brec['files_sha256']['BASE.npz']:
        raise PermissionError('BASE.npz differs from BASE.json')
    with np.load(d / 'BASE.npz') as z:
        B = {k: z[k].copy() for k in z.files}
    with np.load(OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'CROSSFIT.npz') as z:
        S = {k: z[k].copy() for k in z.files}
    SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez(SCRATCH / f'FULLPOOL_BASE_V{v}.npz', oof=oof)
    obs = {'T_c_mine': {str(c): T_c[c] for c in range(5)}, 'T_k_record': brec['T_k'], 'oof_bitwise_vs_BASE': bool(np.array_equal(oof, B['oof'])),
           'oof_max_abs_vs_BASE': float(np.max(np.abs(oof - B['oof']))), 'folds_equal_BASE': bool(np.array_equal(cf, B['fold'])),
           'oof_bitwise_vs_source_CROSSFIT': bool(np.array_equal(oof, S['MODEL_FIT__oof'])), 'folds_equal_source': bool(np.array_equal(cf, S['fold'])),
           'row_id_equal_BASE': bool(np.array_equal(np.asarray(F['row_id']), B['row_id'])), 'cpu_s': round(time.process_time() - t0, 1)}
    ok = (all(T_c[c] == brec['T_k'][str(c)] for c in range(5)) and obs['oof_bitwise_vs_BASE'] and obs['folds_equal_BASE']
          and obs['oof_bitwise_vs_source_CROSSFIT'] and obs['folds_equal_source'] and obs['row_id_equal_BASE'])
    return {'id': f'V5.FULLPOOL.V{v}.BASE__{tag}', 'kind': 'NUMERICAL_REPLAY',
            'what': f'version {v}: the deployed correction\'s base (W100_v 5-fold household cross-fitted out-of-fold probabilities on all '
                    'MODEL_FIT) recomputed with own code vs CORRECT/BASE.npz and the source A1_PO_16 CROSSFIT.npz, bitwise, with T_k',
            'inputs': V.inputs_sha([d / 'BASE.json', d / 'BASE.npz', OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'CROSSFIT.npz']),
            'expected': {'T_k': brec['T_k']}, 'observed': obs, 'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}


def direction(v, tag):
    from ohn import protect as PR
    from ohn import devrun as OD
    F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
               f'independent verifier V5: recompute the full-pool correction directions of version {v} (source P2 recipe on own base)', subset='S100')
    X, names, y, sex = np.asarray(F['X']), [str(x) for x in F['feature_names']], np.asarray(F['occupation'], dtype=np.int64), np.asarray(F['sex'], dtype=np.int64)
    with np.load(SCRATCH / f'FULLPOOL_BASE_V{v}.npz') as z:
        oof = z['oof'].copy()
    tp = MIRROR / 'admitted' / 'stage_a' / f'S{v}' / 'TREE_TEACHER_occupation.json'
    payload = json.loads(tp.read_text())
    adm = json.loads((V.RESULTS / 'SOURCE_ADMISSION.json').read_text())['releases_used'][f'T_TREE/occupation/S{v}']['teacher_sha256']
    teacher_ok = V.file_sha(tp) == adm == V.file_sha(CORRECT / f'V{v}' / 'ROUTING' / 'TEACHER_occupation.json')
    f = np.asarray(payload['split_features'], dtype=np.int64); b = np.asarray(payload['split_borders'], dtype=np.float32)
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    T, L = f.shape[0], 2 ** f.shape[1]
    leaf = np.zeros((len(X32), T), dtype=np.int64)
    for i in range(f.shape[1]):
        leaf |= (X32[:, f[:, i]] > b[None, :, i]).astype(np.int64) << i
    inc_wire = PR.income_wire(np.asarray(OD.anchor_income('H3', int(v), 'MODEL_FIT'), dtype=np.float64))
    lp = np.log(np.maximum(oof, 1e-300)); lp = lp - lp.mean(1, keepdims=True)
    res = {}
    for arm in ('CONT', 'LOC', 'PAIR'):
        drec = json.loads((CORRECT / f'V{v}' / 'DIR' / arm / 'DIR.json').read_text())
        r = PR.train_correction(lp, leaf, y, sex, inc_wire, T, L, arm, f'TSZ1_V{v}', 'full', {'men': 0.005, 'women': 0.005}, guard='none')
        res[arm] = {'theta_sha_mine': V.arr_sha(np.asarray(r['theta'])), 'theta_sha_record': drec['theta_sha256'],
                    'lambda_mine_record': [float(r['lambda']), drec['lambda']], 'equal': V.arr_sha(np.asarray(r['theta'])) == drec['theta_sha256']}
    ok = teacher_ok and all(x['equal'] for x in res.values())
    return {'id': f'V5.FULLPOOL.V{v}.DIRECTIONS__{tag}', 'kind': 'NUMERICAL_REPLAY',
            'what': f'version {v}: the deployed CONT / LOC / PAIR correction parameters recomputed: routing = the admitted T_TREE occupation '
                    'teacher (file hash = SOURCE_ADMISSION = the correction stage routing file), own oblivious routing of all MODEL_FIT people, '
                    'the source P2 recipe (guard none, fixed targets, seed TSZ1_V<v>, fold full) on the verifier\'s own cross-fitted base; '
                    'theta hash vs DIR.json',
            'inputs': V.inputs_sha([tp] + [CORRECT / f'V{v}' / 'DIR' / arm / 'DIR.json' for arm in ('CONT', 'LOC', 'PAIR')]),
            'expected': {arm: res[arm]['theta_sha_record'] for arm in res}, 'observed': {'routing_teacher_admitted': teacher_ok, **res},
            'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('base', 'direction')); ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    chk = base(a.version, a.tag) if a.mode == 'base' else direction(a.version, a.tag)
    V.write_record(f'V5_FULLPOOL__{a.mode}__V{a.version}__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **{k: x for k, x in chk['observed'].items() if not isinstance(x, dict) or k in ('CONT', 'LOC', 'PAIR')}}, default=str)[:2000])


if __name__ == '__main__':
    main()
