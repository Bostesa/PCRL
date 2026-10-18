"""Verifier check 2 (numerical exclusion proof): refit the Track F witness models and the REFIT routing teacher on the
verifier-reconstructed FIT rows ONLY and compare with the stored artifacts bitwise. A bitwise match shows that the stored witness
(deployed model and the 5 outer-fold models behind the out-of-fold base) and the REFIT routing are functions of FIT rows (given
the recorded tree counts) - no GUARD or CHECK row entered them (tree learners bin and split on their training rows).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Learners are constructed directly from the registered recipe text (scikit-learn HistGradientBoosting for the A1_PO_16 witness;
CatBoost 1.2.8 for the T_TREE occupation routing); the compact_po representation, the FIT-only dictionary and the oblivious-tree
evaluation are the verifier's own code. Labels: FIT occupation (+ income for the routing split's class-support repair) via
cts.common.load(operation='verify'). Nothing person-level is printed or written.

    python3 .../v_refit.py witness --partition D0 --tag V1                      (python3: scikit-learn)
    <catboost venv python> .../v_refit.py routing --partition D0 --tag V1     (CatBoost 1.2.8)
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

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402   (admitted-data loader and the logged label-access layer only)

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKF = CAMP / 'runs' / 'TRACKF'
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def fit_labels(P, fields):
    return C.load('MODEL_FIT', ('features',) + tuple(fields), 'verify',
                  f'independent verifier: refit {P} Track F layers on reconstructed FIT rows only (exclusion proof by bitwise identity)',
                  partition=P, prole='FIT')


def my_recode(X, names, X_fit, min_count=10):
    """FIT-only dictionary (PROTOCOL section 6): a level with fewer than 10 FIT people moves to its block's OTHER column."""
    names = [str(n) for n in names]
    X = np.array(X, dtype=np.float64, copy=True)
    for b in BLOCKS:
        other = names.index(f'{b}=__OTHER__')
        for i, n in enumerate(names):
            if n.startswith(b + '=') and n != f'{b}=__OTHER__' and X_fit[:, i].sum() < min_count:
                on = X[:, i] == 1
                X[on, i] = 0.0
                X[on, other] = 1.0
    return X


def my_compact_po(X, names):
    """compact_po from the recipe text: AGEP, WKHP numeric; COW, MAR native category codes (index inside the one-hot block);
    SCHL ordinal value (OTHER = 3); POBP ordinal place-of-birth code (OTHER = 999). Columns AGEP, WKHP, COW, SCHL, MAR, POBP."""
    names = [str(n) for n in names]
    cols = [X[:, names.index('AGEP')], X[:, names.index('WKHP')]]
    for b in BLOCKS:
        idx = [i for i, n in enumerate(names) if n.startswith(b + '=')]
        code = np.argmax(X[:, idx], axis=1)
        if b in ('COW', 'MAR'):
            cols.append(code.astype(np.float64))
        else:
            other = 3.0 if b == 'SCHL' else 999.0
            vals = np.array([other if names[i].endswith('__OTHER__') else float(names[i].split('=', 1)[1]) for i in idx])
            cols.append(vals[code])
    return np.column_stack(cols)


def hgb(cfg, T, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(cfg['lr']), max_iter=int(T),
                                          max_leaf_nodes=int(cfg['leaves']), min_samples_leaf=int(cfg['msl']),
                                          l2_regularization=float(cfg['l2']), max_bins=255, early_stopping=False,
                                          categorical_features=np.array([False, False, True, False, True, False]),
                                          random_state=int(seed))


def soft(raw):
    z = np.asarray(raw, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def witness(P, tag):
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    Fall = C.features('MODEL_FIT'); S = C.features('MODEL_SELECT')
    masks = V.partition_masks_from_json(Fall['unit'], spec)
    F = fit_labels(P, ('occupation',))
    fit = masks[P]['FIT']
    if not np.array_equal(F['rows'], fit):
        raise AssertionError('access layer FIT rows differ from the reconstruction')
    wj = json.loads((TRACKF / P / 'WITNESS' / 'WITNESS.json').read_text())
    cfg, seed = wj['config'], int(wj['seed'])
    if wj['recipe'] != 'A1_PO_16':
        raise SystemExit('only the A1_PO_16 witness is refitted here')
    W, O = npz(TRACKF / P / 'WITNESS' / 'WITNESS.npz'), npz(TRACKF / P / 'WITNESS' / 'OOF.npz')
    names = Fall['feature_names']
    Xf = my_recode(F['X'], names, F['X'])
    Xs = my_recode(S['X'], names, F['X'])
    Rf, Rs = my_compact_po(Xf, names), my_compact_po(Xs, names)
    y = F['occupation']
    folds = V.deal(F['unit'], 5, wj['outer_fold_tag'])
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = hgb(cfg, wj['T_star'], seed).fit(Rf, y)
    p = soft(m.decision_function(Rs))
    res['deployed'] = {'T': wj['T_star'], 'MS_max_abs_diff': float(np.max(np.abs(p - W['dep__MODEL_SELECT']))),
                       'MS_bitwise_equal': bool(np.array_equal(p, W['dep__MODEL_SELECT']))}
    # contrast: the same recipe with GUARD rows added must NOT reproduce the stored model (shows the check can fail)
    for k in range(5):
        tr, te = folds != k, folds == k
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mk = hgb(cfg, wj['T_k'][str(k)], seed).fit(Rf[tr], y[tr])
        pk = soft(mk.decision_function(Rs))
        po = soft(mk.decision_function(Rf[te]))
        res[f'fold{k}'] = {'T': wj['T_k'][str(k)], 'MS_max_abs_diff': float(np.max(np.abs(pk - W['fold__MODEL_SELECT'][k]))),
                           'OOF_max_abs_diff': float(np.max(np.abs(po - O['oof'][te]))),
                           'bitwise_equal': bool(np.array_equal(pk, W['fold__MODEL_SELECT'][k]) and np.array_equal(po, O['oof'][te]))}
    g = masks[P]['GUARD']
    Fg = C.load('MODEL_FIT', ('features', 'occupation'), 'verify',
                f'independent verifier: negative control for the {P} witness refit (FIT plus GUARD must not reproduce the stored model)',
                partition=P, prole='GUARD')
    Xg = my_recode(Fg['X'], names, F['X'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mc = hgb(cfg, wj['T_star'], seed).fit(np.vstack([Rf, my_compact_po(Xg, names)]), np.concatenate([y, Fg['occupation']]))
    pc = soft(mc.decision_function(Rs))
    res['negative_control_fit_plus_guard'] = {'MS_max_abs_diff': float(np.max(np.abs(pc - W['dep__MODEL_SELECT'])))}
    ok = res['deployed']['MS_bitwise_equal'] and all(res[f'fold{k}']['bitwise_equal'] for k in range(5)) and \
        res['negative_control_fit_plus_guard']['MS_max_abs_diff'] > 1e-6
    near = res['deployed']['MS_max_abs_diff'] <= 1e-12 and all(res[f'fold{k}']['MS_max_abs_diff'] <= 1e-12 and
                                                               res[f'fold{k}']['OOF_max_abs_diff'] <= 1e-12 for k in range(5))
    chk = {'id': f'C2.5.{P}.WITNESS_REFIT_IDENTITY__{tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{P}: the stored deployed witness and its 5 outer-fold models (the out-of-fold base of OOF costs and the ensemble '
                   'members) are reproduced bitwise by the A1_PO_16 recipe fitted on the verifier-reconstructed FIT rows alone (recorded '
                   'T* and T_k); a FIT + GUARD fit is a negative control that must differ',
           'method': 'own FIT-only dictionary, own compact_po representation, scikit-learn HistGradientBoostingClassifier built from the '
                     'recipe constants (lr, leaves, min leaf, l2, max_bins 255, no early stopping, COW / MAR categorical, random_state = '
                     'seed); probabilities = softmax of the decision function; compared with WITNESS.npz / OOF.npz',
           'status': 'PASS' if ok else ('ATTENTION' if near else 'FAIL'), 'numbers': res}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_REFIT_WITNESS__{P}__{tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status']}))


def oblivious_raw(payload, X):
    """Own evaluation of an oblivious-tree teacher payload: leaf_t = sum_i 1[x_{f_ti} > b_ti] << i (float32 comparison);
    raw = scale * sum_t leaf_values[t, leaf_t] + bias."""
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


def routing(P, tag):
    import catboost as cb
    from acs_replication import data as AD          # the registered internal early-stopping split (recipe code, label-free hash)
    from acs_replication import trees as TR          # registered constants and seed derivation (recipe), not metric code
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    Fall = C.features('MODEL_FIT'); S = C.features('MODEL_SELECT')
    masks = V.partition_masks_from_json(Fall['unit'], spec)
    F = fit_labels(P, ('income', 'occupation'))
    if not np.array_equal(F['rows'], masks[P]['FIT']):
        raise AssertionError('access layer FIT rows differ from the reconstruction')
    rj = json.loads((TRACKF / P / 'ROUTING' / 'ROUTING.json').read_text())
    payload = json.loads((TRACKF / P / 'ROUTING' / 'TEACHER_occupation.json').read_text())
    seed = int(rj['seed'])
    names = Fall['feature_names']
    Xf = my_recode(F['X'], names, F['X']); Xs = my_recode(S['X'], names, F['X'])
    fit_rows, val_rows, moved = AD.internal_split(F['unit'], {'income': F['income'], 'occupation': F['occupation']}, seed)
    rs = TR.seed_for('ACSR1catboost', seed, 'occupation', 4)

    def make(iters):
        return cb.CatBoostClassifier(loss_function='MultiClass', depth=4, learning_rate=0.05, iterations=int(iters), l2_leaf_reg=3.0,
                                     task_type='CPU', thread_count=1, random_seed=rs, verbose=0, allow_writing_files=False)
    m = make(800)
    m.fit(Xf[fit_rows], F['occupation'][fit_rows], eval_set=(Xf[val_rows], F['occupation'][val_rows]))      # float64 as the recipe
    curve = m.get_evals_result()['validation']['MultiClass']
    count = int(np.argmin(curve)) + 1
    final = make(count)
    final.fit(Xf, F['occupation'])
    ref_raw = np.asarray(final.predict(Xs.astype(np.float32), prediction_type='RawFormulaVal'), dtype=np.float64).reshape(len(Xs), -1)
    ref_leaf = np.asarray(final.calc_leaf_indexes(cb.Pool(Xs.astype(np.float32))), dtype=np.int64)
    raw, leaf = oblivious_raw(payload, Xs)
    res = {'tree_count_mine': count, 'tree_count_recorded': rj['tree_count'], 'catboost_seed_mine': rs,
           'catboost_seed_recorded': rj['catboost_seed'], 'internal_split': [int(len(fit_rows)), int(len(val_rows)), len(moved)],
           'internal_split_recorded': [rj['internal_fit_rows'], rj['internal_validation_rows'], rj['moved_households']],
           'MS_routing_identical': bool(leaf.shape == ref_leaf.shape and np.array_equal(leaf, ref_leaf)),
           'MS_raw_max_abs_diff': float(np.max(np.abs(raw - ref_raw))) if raw.shape == ref_raw.shape else None}
    ok = (count == rj['tree_count'] and rs == rj['catboost_seed'] and res['MS_routing_identical'] and
          res['MS_raw_max_abs_diff'] is not None and res['MS_raw_max_abs_diff'] <= 1e-9 and
          res['internal_split'] == res['internal_split_recorded'])
    chk = {'id': f'C2.6.{P}.ROUTING_REFIT_IDENTITY__{tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{P}: the stored REFIT routing teacher is reproduced by the T_TREE occupation recipe (CatBoost 1.2.8) fitted on the '
                   'verifier-reconstructed FIT rows alone (FIT-only dictionary; internal 15% household split of FIT; tree count = first '
                   'minimum + 1; refit on FIT)',
           'method': 'CatBoost built from the registered constants (depth 4, lr 0.05, l2 3, <= 800 trees, one thread, seed_for(ACSR1catboost, s, '
                     'occupation, 4)); the stored teacher JSON evaluated by the verifier\'s own oblivious-tree code on MODEL_SELECT and '
                     'compared with CatBoost leaf indexes and raw formula values',
           'status': 'PASS' if ok else 'FAIL', 'numbers': res}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_REFIT_ROUTING__{P}__{tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **res}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('witness', 'routing')); ap.add_argument('--partition', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    (witness if a.mode == 'witness' else routing)(a.partition, a.tag)


if __name__ == '__main__':
    main()
