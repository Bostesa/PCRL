"""Verifier checks of the nested out-of-fold variant (PROTOCOL section 7: every upstream layer refitted inside each outer fold).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
  witness  (python3)  for every outer fold k: the A1_PO_16 witness refitted by the verifier on FIT minus fold k (own FIT-minus-k
                      dictionary, own compact_po, recorded T*_k) must reproduce the stored fold-k witness output bitwise; for fold 0
                      the inner cross-fitting base (5 models on FIT minus fold 0, recorded T_k) must reproduce base_oof__0 bitwise;
                      the nested ladder (NESTED_<ARM>.json) is recomputed from NESTED_<ARM>.npz with own metrics / SEs.
  routing  (CatBoost) fold 0: the T_TREE occupation recipe on FIT minus fold 0 must reproduce TEACHER_0.json (routing, raw scores).
Labels through cts.common.load(operation='verify') only (FIT rows of the partition). Nothing person-level is printed or written.

    python3 .../v_nested.py witness --partition D0 --arms PAIR,LOC --tag V1
    <catboost python> .../v_nested.py routing --partition D0 --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
from v_refit import my_recode, my_compact_po, hgb, soft, oblivious_raw  # noqa: E402  (verifier's own helpers)

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

TRACKF = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
TOL = 1e-10


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def fit_rows(P, fields):
    return C.load('MODEL_FIT', ('features',) + tuple(fields), 'verify',
                  f'independent verifier: nested-variant replay on {P} FIT rows (exclusion of outer fold k from its upstream layers)',
                  partition=P, prole='FIT')


def witness(P, arms, tag):
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    Fall = C.features('MODEL_FIT')
    masks = V.partition_masks_from_json(Fall['unit'], spec)
    F = fit_rows(P, ('occupation', 'sex'))
    if not np.array_equal(F['rows'], masks[P]['FIT']):
        raise AssertionError('access layer FIT rows differ from the reconstruction')
    wj = json.loads((TRACKF / P / 'WITNESS' / 'WITNESS.json').read_text())
    cfg, seed = wj['config'], int(wj['seed'])
    nd = TRACKF / P / 'NESTED'
    nw = json.loads((nd / 'WITNESS_NESTED.json').read_text())
    WN = npz(nd / 'WITNESS_NESTED.npz')
    integrity = V.file_sha(nd / 'WITNESS_NESTED.npz') == nw['npz_sha256']
    names, y, unit, X = Fall['feature_names'], F['occupation'], F['unit'], F['X']
    folds = V.deal(unit, 5, wj['outer_fold_tag'])
    res, ok = {'npz_integrity': integrity}, integrity
    for k in range(5):
        rk, tk = folds != k, folds == k
        Xk = my_recode(X[rk], names, X[rk]); Xt = my_recode(X[tk], names, X[rk])
        Rk, Rt = my_compact_po(Xk, names), my_compact_po(Xt, names)
        T = nw['folds'][str(k)]['T_star']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = hgb(cfg, T, seed).fit(Rk, y[rk])
        p = soft(m.decision_function(Rt))
        eq = bool(np.array_equal(p, WN[f'witness_fold__{k}']))
        res[f'fold{k}'] = {'T_star': T, 'n_train': int(rk.sum()), 'witness_fold_bitwise_equal': eq,
                           'max_abs_diff': float(np.max(np.abs(p - WN[f'witness_fold__{k}'])))}
        ok = ok and eq
        if k == 0:                      # inner cross-fitting base of the nested correction on FIT minus fold 0
            ncf = V.deal(unit[rk], 5, f'CTS1_ncf_{P}_{k}')
            base = np.zeros((int(rk.sum()), 5))
            for j in range(5):
                tr, te = ncf != j, ncf == j
                Tj = nw['folds'][str(k)]['T_k'][str(j)]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mj = hgb(cfg, Tj, seed).fit(Rk[tr], y[rk][tr])
                base[te] = soft(mj.decision_function(Rk[te]))
            eqb = bool(np.array_equal(base, WN['base_oof__0']))
            res['fold0']['nested_base_bitwise_equal'] = eqb
            res['fold0']['nested_base_max_abs_diff'] = float(np.max(np.abs(base - WN['base_oof__0'])))
            ok = ok and eqb
    # nested ladders
    lad = {}
    for arm in arms:
        rec = json.loads((nd / f'NESTED_{arm}.json').read_text())
        Z = npz(nd / f'NESTED_{arm}.npz')
        integ = V.file_sha(nd / f'NESTED_{arm}.npz') == rec['npz_sha256']
        same_w = all(np.array_equal(Z['witness_oof'][folds == k], WN[f'witness_fold__{k}']) for k in range(5))
        worst = {'stats': 0.0, 'se': 0.0}
        for s in V.LADDER:
            c = V.incremental_costs(V.release_full(Z['witness_oof'], Z['delta_oof'], s), Z['witness_oof'], y, F['sex'], unit)
            r = rec['ladder_NESTED_OOF'][str(s)]
            for part in ('stats', 'se'):
                worst[part] = max(worst[part], max(abs(c[part][kk] - r[part][kk]) for kk in V.STATS))
        lad[arm] = {'npz_integrity': integ, 'witness_oof_equals_nested_fold_witnesses': same_w,
                    'max_abs_diff_stats': worst['stats'], 'max_abs_diff_se': worst['se']}
        ok = ok and integ and same_w and worst['stats'] <= TOL and worst['se'] <= TOL
    chk = {'id': f'C2.7.{P}.NESTED_VARIANT__{tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{P} nested variant: each outer-fold witness (FIT minus fold k) reproduced bitwise from FIT-minus-k rows alone; the '
                   'nested correction base on FIT minus fold 0 reproduced bitwise; nested OOF ladder costs recomputed',
           'method': 'own FIT-minus-k dictionary and compact_po, scikit-learn HGB from recipe constants with the recorded T*; own metrics '
                     f'and SEs on NESTED_<ARM>.npz (tolerance {TOL})',
           'status': 'PASS' if ok else 'FAIL', 'numbers': {'witness': res, 'ladders': lad}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__, HERE / 'v_refit.py')
    V.write_record(f'V_NESTED_WITNESS__{P}__{tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status']}))


def routing(P, tag):
    import catboost as cb
    from acs_replication import data as AD
    from acs_replication import trees as TR
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    Fall = C.features('MODEL_FIT')
    masks = V.partition_masks_from_json(Fall['unit'], spec)
    F = fit_rows(P, ('income', 'occupation'))
    if not np.array_equal(F['rows'], masks[P]['FIT']):
        raise AssertionError('access layer FIT rows differ from the reconstruction')
    wj = json.loads((TRACKF / P / 'WITNESS' / 'WITNESS.json').read_text())
    seed = int(wj['seed'])
    nd = TRACKF / P / 'NESTED'
    rn = json.loads((nd / 'ROUTING_NESTED.json').read_text())
    names, unit, X = Fall['feature_names'], F['unit'], F['X']
    folds = V.deal(unit, 5, wj['outer_fold_tag'])
    k = 0
    rk, tk = folds != k, folds == k
    Xk = my_recode(X[rk], names, X[rk]); Xt = my_recode(X[tk], names, X[rk])
    fr, vr, moved = AD.internal_split(unit[rk], {'income': F['income'][rk], 'occupation': F['occupation'][rk]}, seed)
    rs = TR.seed_for('ACSR1catboost', seed, 'occupation', 4)

    def make(iters):
        return cb.CatBoostClassifier(loss_function='MultiClass', depth=4, learning_rate=0.05, iterations=int(iters), l2_leaf_reg=3.0,
                                     task_type='CPU', thread_count=1, random_seed=rs, verbose=0, allow_writing_files=False)
    yk = F['occupation'][rk]
    m = make(800); m.fit(Xk[fr], yk[fr], eval_set=(Xk[vr], yk[vr]))
    count = int(np.argmin(m.get_evals_result()['validation']['MultiClass'])) + 1
    final = make(count); final.fit(Xk, yk)
    payload = json.loads((nd / f'TEACHER_{k}.json').read_text())
    raw, leaf = oblivious_raw(payload, Xt)
    ref_raw = np.asarray(final.predict(Xt.astype(np.float32), prediction_type='RawFormulaVal'), dtype=np.float64).reshape(len(Xt), -1)
    ref_leaf = np.asarray(final.calc_leaf_indexes(cb.Pool(Xt.astype(np.float32))), dtype=np.int64)
    res = {'fold': k, 'tree_count_mine': count, 'tree_count_recorded': rn['folds'][str(k)]['tree_count'],
           'internal_split_mine': [int(len(fr)), int(len(vr))],
           'internal_split_recorded': [rn['folds'][str(k)]['internal_fit_rows'], rn['folds'][str(k)]['internal_validation_rows']],
           'fold_k_routing_identical': bool(leaf.shape == ref_leaf.shape and np.array_equal(leaf, ref_leaf)),
           'fold_k_raw_max_abs_diff': float(np.max(np.abs(raw - ref_raw))) if raw.shape == ref_raw.shape else None}
    ok = (res['tree_count_mine'] == res['tree_count_recorded'] and res['internal_split_mine'] == res['internal_split_recorded'] and
          res['fold_k_routing_identical'] and res['fold_k_raw_max_abs_diff'] is not None and res['fold_k_raw_max_abs_diff'] <= 1e-9)
    chk = {'id': f'C2.8.{P}.NESTED_ROUTING_FOLD0__{tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{P} nested variant: the fold-0 routing teacher reproduced from FIT minus fold 0 alone (T_TREE occupation recipe)',
           'method': 'CatBoost 1.2.8 from the registered constants on own FIT-minus-0 dictionary; stored teacher evaluated by own '
                     'oblivious-tree code on the fold-0 rows and compared with CatBoost leaf indexes and raw values',
           'status': 'PASS' if ok else 'FAIL', 'numbers': res}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__, HERE / 'v_refit.py')
    V.write_record(f'V_NESTED_ROUTING__{P}__{tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **res}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('witness', 'routing')); ap.add_argument('--partition', required=True)
    ap.add_argument('--arms', default='PAIR'); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    if a.mode == 'witness':
        witness(a.partition, [x for x in a.arms.split(',') if x], a.tag)
    else:
        routing(a.partition, a.tag)


if __name__ == '__main__':
    main()
