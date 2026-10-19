"""Verifier stage V3 (control exposure): independent refit of the T_TREE occupation teacher of one subset control (CatBoost 1.2.8
venv), compared with the saved teacher payload by the verifier's own oblivious-tree evaluation.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Labels: MODEL_FIT income and occupation of the registered subset only (tsz.common.load operation 'verify'; income only for the
recipe's two-task class-support repair of the internal split). Recipe constants from the RULES / DESIGN_BRIEF text (depth 4, lr 0.05,
l2 3, <= 800 trees, one thread, seed SHA-256('ACSR1catboost|<v>|occupation|4')[:8] as an integer, tree count = first argmin of the
validation loss + 1, refit on the subset); the internal 15% household split is the registered recipe function
acs_replication.data.internal_split (label-free order, class-support repair); the FIT-only dictionary is the verifier's own.

    <catboost venv python> .../v3_ttree.py --subset S50_O1 --version 0 --tag V1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

CONTROL = ROOT / '.private' / 'pcrl_controlled_training_size_v1' / 'runs' / 'CONTROL'


def oblivious(payload, X):
    f = np.asarray(payload['split_features'], dtype=np.int64)
    b = np.asarray(payload['split_borders'], dtype=np.float32)
    lv = np.asarray(payload['leaf_values'], dtype=np.float64)
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    T, depth = f.shape
    leaf = np.zeros((len(X32), T), dtype=np.int64)
    for i in range(depth):
        leaf |= (X32[:, f[:, i]] > b[None, :, i]).astype(np.int64) << i
    raw = float(payload['scale']) * lv[np.arange(T)[None, :], leaf].sum(1) + np.asarray(payload['bias'], dtype=np.float64)
    return raw, leaf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--subset', required=True); ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    import catboost as cb
    from acs_replication import data as AD
    sid, v = a.subset, a.version
    d = CONTROL / f'{sid}__V{v}' / 'TEACHERS'
    trec = json.loads((d / 'TEACHERS.json').read_text())
    payload = json.loads((d / 'TEACHER_occupation.json').read_text())
    if V.file_sha(d / 'TEACHER_occupation.json') != trec['files_sha256']['TEACHER_occupation.json']:
        raise PermissionError('teacher payload differs from its record')
    Fall = C.features('MODEL_FIT'); S = C.features('MODEL_SELECT')
    masks = V.subset_masks(Fall['unit'])
    F = C.load('MODEL_FIT', ('features', 'income', 'occupation'), 'verify',
               f'independent verifier V3: refit the T_TREE occupation teacher of the {sid} V{v} task-only reference on its subset rows only',
               subset=sid)
    rows_ok = bool(np.array_equal(F['rows'], masks[sid]))
    names = [str(x) for x in F['feature_names']]
    coll = V.fit_only_dictionary(F['X'], names, 10)
    Xf = V.apply_dictionary(F['X'], names, coll)
    Xs = V.apply_dictionary(S['X'], names, coll)
    y = {'income': np.asarray(F['income'], dtype=np.int64), 'occupation': np.asarray(F['occupation'], dtype=np.int64)}
    fit_rows, val_rows, moved = AD.internal_split(np.asarray(F['unit']), y, v)
    rs = int(hashlib.sha256(f'ACSR1catboost|{v}|occupation|4'.encode()).hexdigest()[:8], 16)

    def make(it):
        return cb.CatBoostClassifier(loss_function='MultiClass', depth=4, learning_rate=0.05, iterations=int(it), l2_leaf_reg=3.0,
                                     task_type='CPU', thread_count=1, random_seed=rs, verbose=0, allow_writing_files=False)
    m = make(800)
    m.fit(Xf[fit_rows], y['occupation'][fit_rows], eval_set=(Xf[val_rows], y['occupation'][val_rows]))
    curve = m.get_evals_result()['validation']['MultiClass']
    count = int(np.argmin(curve)) + 1
    final = make(count)
    final.fit(Xf, y['occupation'])
    cb_raw = np.asarray(final.predict(Xs.astype(np.float32), prediction_type='RawFormulaVal'), dtype=np.float64).reshape(len(Xs), -1)
    cb_leaf = np.asarray(final.calc_leaf_indexes(cb.Pool(Xs.astype(np.float32))), dtype=np.int64)
    raw, leaf = oblivious(payload, Xs)
    with np.load(d / 'TEACHER_OUTPUTS.npz') as z:
        saved = z['T_TREE__MODEL_SELECT__occupation'].copy()
    zc = raw - raw.mean(1, keepdims=True)
    p_mine = V.softmax(zc)
    info = trec['teachers']['occupation']
    obs = {'access_rows_equal_own_subset': rows_ok, 'dictionary_n_collapsed': {b: len(coll[b]) for b in V.BLOCKS},
           'internal_split_mine': [int(len(fit_rows)), int(len(val_rows)), len(moved)],
           'internal_split_recorded': [trec['internal_split']['fit_rows'], trec['internal_split']['validation_rows'], trec['internal_split']['moved_households']],
           'catboost_seed_mine': rs, 'catboost_seed_recorded': info['catboost_seed'], 'tree_count_mine': count, 'tree_count_recorded': info['T'],
           'MODEL_SELECT_routing_identical': bool(leaf.shape == cb_leaf.shape and np.array_equal(leaf, cb_leaf)),
           'MODEL_SELECT_raw_max_abs_diff_vs_catboost': float(np.max(np.abs(raw - cb_raw))) if raw.shape == cb_raw.shape else None,
           'saved_output_vs_own_payload_softmax_max_abs_diff': float(np.max(np.abs(p_mine - saved))),
           'catboost_version': cb.__version__}
    ok = (rows_ok and count == info['T'] and rs == info['catboost_seed'] and obs['MODEL_SELECT_routing_identical']
          and obs['MODEL_SELECT_raw_max_abs_diff_vs_catboost'] is not None and obs['MODEL_SELECT_raw_max_abs_diff_vs_catboost'] <= 1e-9
          and obs['internal_split_mine'] == obs['internal_split_recorded'] and obs['saved_output_vs_own_payload_softmax_max_abs_diff'] <= 1e-12)
    chk = {'id': f'V3.REFIT.T_TREE.{sid}__V{v}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{sid} V{v} task-only reference: the saved T_TREE occupation teacher is reproduced by an independent CatBoost refit on the '
                   'verifier-reconstructed subset rows only (own FIT-only dictionary; internal 15% household split with class-support repair; '
                   'tree count = first argmin + 1; refit on the subset): identical tree count and seed, identical MODEL_SELECT routing, raw '
                   'scores within 1e-9 of the saved payload evaluated by own oblivious-tree code; the saved MODEL_SELECT T_TREE output equals '
                   'the own softmax of the payload',
           'inputs': V.inputs_sha([d / 'TEACHERS.json', d / 'TEACHER_occupation.json', d / 'TEACHER_OUTPUTS.npz']),
           'expected': {'tree_count': info['T'], 'catboost_seed': info['catboost_seed']}, 'observed': obs,
           'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    import os
    V.write_record(f'V3_TTREE__{sid}__V{v}__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **{k: obs[k] for k in ('tree_count_mine', 'tree_count_recorded', 'MODEL_SELECT_routing_identical',
                                                                         'MODEL_SELECT_raw_max_abs_diff_vs_catboost', 'internal_split_mine')}}))


if __name__ == '__main__':
    main()
