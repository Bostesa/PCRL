"""Verifier stage V5d: one nested outer fold recomputed end to end (exposure proof of the strength-selection estimator).

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Fold choice declared before reading any nested output: version 0, outer fold k = 3 of TSZ1_ncv10, arm LOC (the primary arm).
Labels: MODEL_FIT (subset S100) through tsz.common.load operation 'verify'.

  witness  (python3)   own FIT-only dictionary on the 90% (rows outside fold k), own compact_po, own inner folds
                       (SHA-256('OHN_A_cv_S<v>|<household>') over all MODEL_FIT units, restricted) and cross-fitting folds
                       (SHA-256('OHN1_crossfit|<household>'), restricted), own T* / T_c by the pooled inner-CV NLL, HGB built from the
                       A1_PO_16 constants: the deployed 90% witness applied to fold k and the 5-fold cross-fitted base on the 90%, vs
                       NESTED/V<v>/WITNESS/F<k>/WITNESS.npz (bitwise) and WITNESS.json (T*, T_k, sizes).
  routing  (CatBoost)  own T_TREE occupation refit on the 90% (internal split = the registered recipe function), vs
                       ROUTING/F<k>/TEACHER.json by own oblivious-tree evaluation (routing of the 90% and of fold k).
  correct  (torch)     the source P2 recipe ohn.protect.train_correction (guard 'none', fixed targets) re-run on the verifier's
                       witness base and routing for this fold; own correction sum on fold k vs NESTED_<arm>.npz delta_oof (fold-k rows)
                       and the fold's recorded theta hash.

    python3 .../v5_nestedfold.py witness --version 0 --fold 3 --tag V1
    <catboost venv python> .../v5_nestedfold.py routing --version 0 --fold 3 --tag V1
    <torch venv python> .../v5_nestedfold.py correct --version 0 --fold 3 --arm LOC --tag V1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

NESTED = ROOT / '.private' / 'pcrl_controlled_training_size_v1' / 'runs' / 'NESTED'
CAT_MASK = np.array([False, False, True, False, True, False])
SCRATCH = ROOT / '.private' / 'pcrl_controlled_training_size_v1' / 'verifier_scratch'


def hgb_fit(Xr, y, T, v):
    from ohn import fam_a as FA
    cfg = FA.CONFIGS['A1_PO_16']
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(cfg['lr']), max_iter=int(T), max_leaf_nodes=int(cfg['leaves']),
                                       min_samples_leaf=int(cfg['msl']), l2_regularization=float(cfg['l2']), max_bins=255, early_stopping=False,
                                       categorical_features=CAT_MASK, random_state=int(v))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(Xr, y)
    return m


def tstar(Xr, y, inner, v, T_max=1000):
    cur = np.zeros(T_max)
    idx_all = np.arange(len(y))
    for j in sorted(set(inner.tolist())):
        tr, va = inner != j, inner == j
        m = hgb_fit(Xr[tr], y[tr], T_max, v)
        yv = y[va]; idx = np.arange(len(yv))
        for t, raw in enumerate(m.staged_decision_function(Xr[va])):
            p = V.softmax(raw)
            cur[t] += float(np.sum(-np.log(np.maximum(p[idx, yv], V.NLL_FLOOR))))
    cur /= float(len(y))
    srt = np.sort(cur)
    return int(np.argmin(cur)) + 1, float(srt[1] - srt[0])


def load_fold(v, sub):
    F = C.load('MODEL_FIT', ('features',) + sub, 'verify',
               f'independent verifier V5: end-to-end recomputation of one nested outer fold (version {v}) with own witness / routing code',
               subset='S100')
    return F


def witness(v, k, tag):
    t0 = time.process_time()
    F = load_fold(v, ('occupation',))
    X, names, y, unit = np.asarray(F['X']), [str(x) for x in F['feature_names']], np.asarray(F['occupation'], dtype=np.int64), np.asarray(F['unit'])
    ncv = V.folds(unit, 10, 'TSZ1_ncv10')
    rk, tk = ncv != k, ncv == k
    coll = V.fit_only_dictionary(X[rk], names, 10)
    Xr = V.compact_po(V.apply_dictionary(X[rk], names, coll), names)
    Xt = V.compact_po(V.apply_dictionary(X[tk], names, coll), names)
    yr = y[rk]
    inner = V.folds(unit, 5, f'OHN_A_cv_S{v}')[rk]
    cf = V.folds(unit, 5, 'OHN1_crossfit')[rk]
    T_dep, gap_dep = tstar(Xr, yr, inner, v)
    dep = hgb_fit(Xr, yr, T_dep, v)
    w_fold = V.softmax(np.asarray(dep.decision_function(Xt), dtype=np.float64))
    oof = np.full((len(yr), 5), np.nan)
    T_c, gaps = {}, {}
    for c in range(5):
        rows = cf != c
        T_c[c], gaps[c] = tstar(Xr[rows], yr[rows], inner[rows], v)
        mc = hgb_fit(Xr[rows], yr[rows], T_c[c], v)
        oof[cf == c] = V.softmax(np.asarray(mc.decision_function(Xr[cf == c]), dtype=np.float64))
    d = NESTED / f'V{v}' / 'WITNESS' / f'F{k}'
    rec = json.loads((d / 'WITNESS.json').read_text())
    for n, h in rec['files_sha256'].items():
        if V.file_sha(d / n) != h:
            raise PermissionError(f'{n} differs from its record')
    with np.load(d / 'WITNESS.npz') as z:
        W = {x: z[x].copy() for x in z.files}
    obs = {'train_idx_equal': bool(np.array_equal(W['train_idx'], np.flatnonzero(rk))), 'test_idx_equal': bool(np.array_equal(W['test_idx'], np.flatnonzero(tk))),
           'T_star_mine_record': [T_dep, rec['T_star']], 'T_k_mine_record': {str(c): [T_c[c], rec['T_k'][str(c)]] for c in range(5)},
           'argmin_gaps': {'deployed': gap_dep, **{str(c): gaps[c] for c in range(5)}},
           'witness_fold_bitwise': bool(np.array_equal(w_fold, W['witness_fold'])), 'witness_fold_max_abs': float(np.max(np.abs(w_fold - W['witness_fold']))),
           'base_oof_bitwise': bool(np.array_equal(oof, W['base_oof'])), 'base_oof_max_abs': float(np.max(np.abs(oof - W['base_oof']))),
           'training_people': int(rk.sum()), 'held_out_people': int(tk.sum()), 'n_collapsed': {b: len(coll[b]) for b in V.BLOCKS},
           'cpu_s': round(time.process_time() - t0, 1)}
    ok = (obs['train_idx_equal'] and obs['test_idx_equal'] and T_dep == rec['T_star'] and all(T_c[c] == rec['T_k'][str(c)] for c in range(5))
          and obs['witness_fold_bitwise'] and obs['base_oof_bitwise'])
    SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez(SCRATCH / f'NESTEDFOLD_V{v}_F{k}_WITNESS.npz', base_oof=oof, witness_fold=w_fold)     # private scratch for the correct step
    chk = {'id': f'V5.NESTED_FOLD.V{v}.F{k}.WITNESS__{tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'nested outer fold {k} of version {v}: the 90% witness (own dictionary, folds, T*, HGB) applied to the held-out fold and its '
                   '5-fold cross-fitted base on the 90%, recomputed independently, vs WITNESS.npz bitwise and WITNESS.json T* / T_k',
           'inputs': V.inputs_sha([d / 'WITNESS.json', d / 'WITNESS.npz']), 'expected': {'T_star': rec['T_star'], 'T_k': rec['T_k']},
           'observed': obs, 'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    return chk


def routing(v, k, tag):
    import catboost as cb
    from acs_replication import data as AD
    F = load_fold(v, ('income', 'occupation'))
    X, names, unit = np.asarray(F['X']), [str(x) for x in F['feature_names']], np.asarray(F['unit'])
    y = {'income': np.asarray(F['income'], dtype=np.int64), 'occupation': np.asarray(F['occupation'], dtype=np.int64)}
    ncv = V.folds(unit, 10, 'TSZ1_ncv10')
    rk, tk = ncv != k, ncv == k
    coll = V.fit_only_dictionary(X[rk], names, 10)
    Xtr, Xte = V.apply_dictionary(X[rk], names, coll), V.apply_dictionary(X[tk], names, coll)
    fit_rows, val_rows, moved = AD.internal_split(unit[rk], {t: y[t][rk] for t in y}, v)
    rs = int(hashlib.sha256(f'ACSR1catboost|{v}|occupation|4'.encode()).hexdigest()[:8], 16)

    def make(it):
        return cb.CatBoostClassifier(loss_function='MultiClass', depth=4, learning_rate=0.05, iterations=int(it), l2_leaf_reg=3.0,
                                     task_type='CPU', thread_count=1, random_seed=rs, verbose=0, allow_writing_files=False)
    m = make(800)
    yo = y['occupation'][rk]
    m.fit(Xtr[fit_rows], yo[fit_rows], eval_set=(Xtr[val_rows], yo[val_rows]))
    count = int(np.argmin(m.get_evals_result()['validation']['MultiClass'])) + 1
    final = make(count); final.fit(Xtr, yo)
    d = NESTED / f'V{v}' / 'ROUTING' / f'F{k}'
    rec = json.loads((d / 'ROUTING.json').read_text())
    payload = json.loads((d / 'TEACHER.json').read_text())
    if V.file_sha(d / 'TEACHER.json') != rec['files_sha256']['TEACHER.json']:
        raise PermissionError('TEACHER.json differs from its record')
    f = np.asarray(payload['split_features'], dtype=np.int64); b = np.asarray(payload['split_borders'], dtype=np.float32)
    lv = np.asarray(payload['leaf_values'], dtype=np.float64)
    out = {}
    for nm, Xs in (('training', Xtr), ('held_out', Xte)):
        X32 = np.ascontiguousarray(Xs, dtype=np.float32)
        leaf = np.zeros((len(X32), f.shape[0]), dtype=np.int64)
        for i in range(f.shape[1]):
            leaf |= (X32[:, f[:, i]] > b[None, :, i]).astype(np.int64) << i
        raw = float(payload['scale']) * lv[np.arange(f.shape[0])[None, :], leaf].sum(1) + np.asarray(payload['bias'], dtype=np.float64)
        cb_leaf = np.asarray(final.calc_leaf_indexes(cb.Pool(X32)), dtype=np.int64)
        cb_raw = np.asarray(final.predict(X32, prediction_type='RawFormulaVal'), dtype=np.float64).reshape(len(X32), -1)
        out[nm] = {'routing_identical': bool(leaf.shape == cb_leaf.shape and np.array_equal(leaf, cb_leaf)),
                   'raw_max_abs_diff': float(np.max(np.abs(raw - cb_raw))) if raw.shape == cb_raw.shape else None}
    SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
    obs = {'tree_count_mine_record': [count, rec['tree_count']], 'internal_split_mine': [int(len(fit_rows)), int(len(val_rows)), len(moved)],
           'internal_split_record': [rec.get('internal_fit_rows'), rec.get('internal_validation_rows'), rec.get('moved_households')],
           'catboost_seed': rs, **out}
    ok = (count == rec['tree_count'] and all(x['routing_identical'] and x['raw_max_abs_diff'] is not None and x['raw_max_abs_diff'] <= 1e-9 for x in out.values())
          and obs['internal_split_mine'][:2] == obs['internal_split_record'][:2])
    return {'id': f'V5.NESTED_FOLD.V{v}.F{k}.ROUTING__{tag}', 'kind': 'NUMERICAL_REPLAY',
            'what': f'nested outer fold {k} of version {v}: the T_TREE occupation routing teacher refitted independently on the 90% (own '
                    'dictionary, registered internal split, own CatBoost call) vs ROUTING/F<k>/TEACHER.json (own oblivious-tree evaluation: '
                    'identical routing of the 90% and of the held-out fold, raw within 1e-9) and its tree count',
            'inputs': V.inputs_sha([d / 'ROUTING.json', d / 'TEACHER.json']), 'expected': {'tree_count': rec['tree_count']}, 'observed': obs,
            'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}


def correct(v, k, arm, tag):
    from ohn import protect as PR
    from ohn import devrun as OD
    F = load_fold(v, ('occupation', 'sex'))
    X, names, y, sex, unit = np.asarray(F['X']), [str(x) for x in F['feature_names']], np.asarray(F['occupation'], dtype=np.int64), \
        np.asarray(F['sex'], dtype=np.int64), np.asarray(F['unit'])
    ncv = V.folds(unit, 10, 'TSZ1_ncv10')
    rk, tk = ncv != k, ncv == k
    coll = V.fit_only_dictionary(X[rk], names, 10)
    with np.load(SCRATCH / f'NESTEDFOLD_V{v}_F{k}_WITNESS.npz') as z:
        base_oof = z['base_oof'].copy()
    d = NESTED / f'V{v}' / 'ROUTING' / f'F{k}'
    payload = json.loads((d / 'TEACHER.json').read_text())
    f = np.asarray(payload['split_features'], dtype=np.int64); b = np.asarray(payload['split_borders'], dtype=np.float32)
    T, L = f.shape[0], 2 ** f.shape[1]

    def leaves(Xs):
        X32 = np.ascontiguousarray(V.apply_dictionary(Xs, names, coll), dtype=np.float32)
        leaf = np.zeros((len(X32), T), dtype=np.int64)
        for i in range(f.shape[1]):
            leaf |= (X32[:, f[:, i]] > b[None, :, i]).astype(np.int64) << i
        return leaf
    lv_tr, lv_te = leaves(X[rk]), leaves(X[tk])
    inc = np.asarray(OD.anchor_income('H3', int(v), 'MODEL_FIT'), dtype=np.float64)
    inc_wire = PR.income_wire(inc)
    lp = np.log(np.maximum(base_oof, 1e-300)); lp = lp - lp.mean(1, keepdims=True)
    targets = {'men': 0.005, 'women': 0.005}
    r = PR.train_correction(lp, lv_tr, y[rk], sex[rk], inc_wire[rk], T, L, arm, f'TSZ1_V{v}', f'ncv{k}', targets, guard='none')
    th = np.asarray(r['theta'], dtype=np.float64).reshape(T, L, 5)
    delta = th[np.arange(T)[None, :], lv_te].sum(1) / T
    nrec = json.loads((NESTED / f'V{v}' / f'NESTED_{arm}.json').read_text())
    with np.load(NESTED / f'V{v}' / f'NESTED_{arm}.npz') as z:
        dn = z['delta_oof'].copy()
    obs = {'theta_sha256_mine': V.arr_sha(np.asarray(r['theta'])), 'theta_sha256_record': nrec['folds'][str(k)]['correction']['theta_sha256'],
           'lambda_mine_record': [float(r['lambda']), nrec['folds'][str(k)]['correction']['lambda']],
           'delta_fold_bitwise': bool(np.array_equal(delta, dn[tk])), 'delta_fold_max_abs': float(np.max(np.abs(delta - dn[tk])))}
    ok = obs['theta_sha256_mine'] == obs['theta_sha256_record'] and obs['delta_fold_max_abs'] <= 1e-12
    return {'id': f'V5.NESTED_FOLD.V{v}.F{k}.CORRECT_{arm}__{tag}', 'kind': 'NUMERICAL_REPLAY',
            'what': f'nested outer fold {k} of version {v}, arm {arm}: the source P2 correction recipe (ohn.protect.train_correction, guard none, '
                    'fixed targets {men: 0.005, women: 0.005}) re-run on the verifier\'s own 90% base (log of the cross-fitted base, centred) '
                    'and own routing of the saved fold teacher; theta hash and the held-out correction vs NESTED_<arm> (fold rows)',
            'inputs': V.inputs_sha([NESTED / f'V{v}' / f'NESTED_{arm}.json', NESTED / f'V{v}' / f'NESTED_{arm}.npz', d / 'TEACHER.json']),
            'expected': {'theta_sha256': obs['theta_sha256_record']}, 'observed': obs, 'status': 'PASS' if ok else 'FAIL',
            'code_sha256': V.code_hashes(__file__, V.__file__)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('witness', 'routing', 'correct')); ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True); ap.add_argument('--arm', default='LOC'); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    chk = {'witness': lambda: witness(a.version, a.fold, a.tag), 'routing': lambda: routing(a.version, a.fold, a.tag),
           'correct': lambda: correct(a.version, a.fold, a.arm, a.tag)}[a.mode]()
    V.write_record(f'V5_NESTEDFOLD__{a.mode}__V{a.version}F{a.fold}__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **{kk: vv for kk, vv in chk['observed'].items() if not isinstance(vv, dict)}}, default=str)[:2000])


if __name__ == '__main__':
    main()
