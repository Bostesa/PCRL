"""Verifier checks of Track R (whole-service refits on R0 / R1 FIT; PROTOCOL section 9).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
  teacher  (CatBoost venv) the refitted T_TREE income teacher reproduced from the verifier-reconstructed FIT rows alone (FIT-only
           dictionary, internal 15% household split of FIT with both tasks' support repair, tree count = first minimum + 1, refit on
           FIT): routing and raw values on MODEL_SELECT compared by the verifier's own oblivious-tree evaluation.
  step1    (python3) the refitted H3 Step-1 income learner (HistGradientBoosting, the recorded selected setting and tree count)
           reproduced from FIT rows alone; decision values on MODEL_SELECT compared with the stored HGB.pkl.
  eval     (python3) the income clauses of the Track R complete releases (H3R, HCR income with the partition's refitted witness) on
           MODEL_SELECT recomputed with own metrics / household-clustered SEs against the frozen-bank U0 / Ucal income outputs.
Labels via cts.common.load(operation='verify') only (R FIT rows; MODEL_SELECT). Nothing person-level is printed or written.

    <catboost python> .../v_trackr.py teacher --partition R0 --tag V1
    python3 .../v_trackr.py step1 --partition R0 --tag V1
    python3 .../v_trackr.py eval --partition R0 --tag V1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
from v_refit import my_recode, oblivious_raw  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKR = CAMP / 'runs' / 'TRACK_R'
TRACKF = CAMP / 'runs' / 'TRACKF'
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def fit_rows(P, fields):
    return C.load('MODEL_FIT', ('features',) + tuple(fields), 'verify',
                  f'independent verifier: Track R {P} refit identity on reconstructed FIT rows (whole-service exclusion check)',
                  partition=P, prole='FIT')


def ms_rows(fields):
    return C.load('MODEL_SELECT', ('features',) + tuple(fields), 'verify',
                  'independent verifier: Track R complete-release income clauses on MODEL_SELECT')


def masks_for(P):
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    Fall = C.features('MODEL_FIT')
    return V.partition_masks_from_json(Fall['unit'], spec), Fall, spec


def teacher(P, tag):
    import catboost as cb
    from acs_replication import data as AD
    from acs_replication import trees as TR
    masks, Fall, spec = masks_for(P)
    S = C.features('MODEL_SELECT')
    F = fit_rows(P, ('income', 'occupation'))
    if not np.array_equal(F['rows'], masks[P]['FIT']):
        raise AssertionError('FIT rows differ from the reconstruction')
    seed = int(spec['partitions'][P]['associated_source_seed'])
    tj = json.loads((TRACKR / P / 'TEACHERS' / 'TEACHERS.json').read_text())
    names = Fall['feature_names']
    Xf = my_recode(F['X'], names, F['X']); Xs = my_recode(S['X'], names, F['X'])
    fr, vr, moved = AD.internal_split(F['unit'], {'income': F['income'], 'occupation': F['occupation']}, seed)
    rs = TR.seed_for('ACSR1catboost', seed, 'income', 4)

    def make(iters):
        return cb.CatBoostClassifier(loss_function='Logloss', depth=4, learning_rate=0.05, iterations=int(iters), l2_leaf_reg=3.0,
                                     task_type='CPU', thread_count=1, random_seed=rs, verbose=0, allow_writing_files=False)
    y = F['income']
    m = make(800); m.fit(Xf[fr], y[fr], eval_set=(Xf[vr], y[vr]))
    count = int(np.argmin(m.get_evals_result()['validation']['Logloss'])) + 1
    final = make(count); final.fit(Xf, y)
    payload = json.loads((TRACKR / P / 'TEACHERS' / 'TEACHER_income.json').read_text())
    raw, leaf = oblivious_raw(payload, Xs)
    ref_raw = np.asarray(final.predict(Xs.astype(np.float32), prediction_type='RawFormulaVal'), dtype=np.float64).reshape(len(Xs), -1)
    ref_leaf = np.asarray(final.calc_leaf_indexes(cb.Pool(Xs.astype(np.float32))), dtype=np.int64)
    rec = tj['teachers']['income']
    res = {'tree_count_mine': count, 'tree_count_recorded': rec['T'], 'catboost_seed_mine': rs, 'catboost_seed_recorded': rec['catboost_seed'],
           'internal_split_mine': [int(len(fr)), int(len(vr)), len(moved)],
           'internal_split_recorded': [tj['internal_split']['fit_rows'], tj['internal_split']['validation_rows'],
                                       tj['internal_split']['moved_households']],
           'MS_routing_identical': bool(leaf.shape == ref_leaf.shape and np.array_equal(leaf, ref_leaf)),
           'MS_raw_max_abs_diff': float(np.max(np.abs(raw.reshape(len(Xs), -1) - ref_raw))) if raw.size == ref_raw.size else None,
           'fit_rows_sha_recorded_equals_reconstruction': tj['fit_rows_sha256'] == V.arr_sha(np.asarray(F['row_id']))}
    ok = (res['tree_count_mine'] == res['tree_count_recorded'] and rs == rec['catboost_seed'] and res['MS_routing_identical'] and
          res['MS_raw_max_abs_diff'] is not None and res['MS_raw_max_abs_diff'] <= 1e-9 and
          res['internal_split_mine'] == res['internal_split_recorded'] and res['fit_rows_sha_recorded_equals_reconstruction'])
    finish(f'C2.9.{P}.TRACKR_INCOME_TEACHER__{tag}',
           f'{P} Track R: the refitted T_TREE income teacher reproduced from the reconstructed FIT rows alone',
           'CatBoost 1.2.8 (Logloss) from the registered constants; own FIT-only dictionary; stored teacher JSON evaluated by own oblivious-'
           'tree code on MODEL_SELECT against CatBoost leaf indexes and raw values', ok, res, f'V_TRACKR_TEACHER__{P}__{tag}.json')


def step1(P, tag):
    from sklearn.ensemble import HistGradientBoostingClassifier
    masks, Fall, spec = masks_for(P)
    S = C.features('MODEL_SELECT')
    F = fit_rows(P, ('income',))
    if not np.array_equal(F['rows'], masks[P]['FIT']):
        raise AssertionError('FIT rows differ from the reconstruction')
    seed = int(spec['partitions'][P]['associated_source_seed'])
    inc = json.loads((TRACKR / P / 'INCOME' / 'INCOME.json').read_text())
    sel = inc['step1']['selected']
    if sel['rep'] != 'ohe':
        raise SystemExit('only the one-hot Step-1 setting is refitted here')
    names = Fall['feature_names']
    with open(TRACKR / P / 'INCOME' / 'HGB.pkl', 'rb') as f:        # the campaign's own saved model (local, hash-checked below)
        stored = pickle.load(f)
    if isinstance(stored, dict):
        stored = stored.get('model', stored.get('hgb', stored))
    hgb_sha_ok = V.file_sha(TRACKR / P / 'INCOME' / 'HGB.pkl') == inc['files_sha256']['HGB.pkl']
    res = {'selected': sel, 'hgb_pkl_sha_ok': hgb_sha_ok}
    for label, Xf, Xs in (('fit_only_dictionary', my_recode(F['X'], names, F['X']), my_recode(S['X'], names, F['X'])),
                          ('admitted_encoding', np.asarray(F['X'], dtype=np.float64), np.asarray(S['X'], dtype=np.float64))):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(sel['lr']), max_iter=int(sel['max_iter']),
                                               max_leaf_nodes=int(sel['leaves']), min_samples_leaf=int(sel['msl']),
                                               l2_regularization=float(sel['l2']), max_bins=255, early_stopping=False,
                                               categorical_features=None, random_state=seed).fit(Xf, F['income'])
        d_mine = np.asarray(m.decision_function(Xs), dtype=np.float64)
        d_stored = np.asarray(stored.decision_function(Xs), dtype=np.float64)
        res[label] = {'MS_max_abs_diff': float(np.max(np.abs(d_mine - d_stored))), 'MS_bitwise_equal': bool(np.array_equal(d_mine, d_stored))}
    ok = hgb_sha_ok and (res['fit_only_dictionary']['MS_bitwise_equal'] or res['admitted_encoding']['MS_bitwise_equal'])
    finish(f'C2.10.{P}.TRACKR_STEP1_HGB__{tag}',
           f'{P} Track R: the refitted H3 Step-1 income learner (HGB, recorded selected setting and tree count) reproduced from the '
           'reconstructed FIT rows alone (decision values on MODEL_SELECT against the stored HGB.pkl)',
           'scikit-learn HistGradientBoostingClassifier from the recorded setting (random_state = seed), fitted on FIT rows with the own '
           'FIT-only dictionary (and, as a recorded alternative, the admitted encoding)', ok, res, f'V_TRACKR_STEP1__{P}__{tag}.json')


def evaluate(P, tag):
    from ohn import common as OC
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    seed = int(spec['partitions'][P]['associated_source_seed'])
    S = ms_rows(('income', 'occupation', 'sex'))
    R = npz(TRACKR / P / 'INCOME' / 'RELEASE.npz')
    if not np.array_equal(R['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('MODEL_SELECT row order differs')
    bank, ids = OC.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('bank row order differs')
    ev = json.loads((TRACKF / P / 'EVAL' / 'TRACKR_EVAL_MS.json').read_text())
    y = S['income']; unit = S['unit']
    worst_v, worst_se, rows = 0.0, 0.0, {}
    for cfg in ('H3R', 'HCR'):
        own = metrics2(y, R[f'{cfg}__MODEL_SELECT__income'])
        recs = {(c['reference'], c['metric']): c for c in ev['results'][cfg]['eval']['utility']['clauses'] if c['task'] == 'income'}
        for ref in ('U0', 'Ucal'):
            r = metrics2(y, np.asarray(bank[ref]['MODEL_SELECT']['income'], dtype=np.float64))
            for mname, d in (('accuracy_loss', r['accuracy'] - own['accuracy']), ('nll_excess', own['nll'] - r['nll']),
                             ('brier_excess', own['brier'] - r['brier'])):
                v, se = float(d.mean()), V.clustered_se(d, unit)
                rc = recs[(ref, mname)]
                worst_v = max(worst_v, abs(v - rc['value'])); worst_se = max(worst_se, abs(se - rc['se']))
                rows[f'{cfg}|{ref}|{mname}'] = {'value': v, 'se': se, 'point_pass': bool(v <= V.CAP[mname]),
                                                'recorded_point_pass': rc['point_pass']}
    passes_ok = all(v['point_pass'] == v['recorded_point_pass'] for v in rows.values())
    distinct = {cfg: V.arr_sha(R[f'{cfg}__MODEL_SELECT__income']) for cfg in ('H3R', 'HCR')}
    ok = worst_v <= 1e-10 and worst_se <= 1e-10 and passes_ok
    finish(f'C3.{P}.TRACKR_INCOME_CLAUSES__{tag}',
           f'{P} Track R complete releases: the 6 income clauses of H3R and HCR (with the partition\'s refitted witness) on MODEL_SELECT',
           'own per-person metrics (1e-12 NLL floor) and household-clustered SEs of the paired differences against the frozen-bank U0 / '
           'Ucal income outputs; compared with TRACKR_EVAL_MS.json (tolerance 1e-10); point passes compared',
           ok, {'max_abs_value_diff': worst_v, 'max_abs_se_diff': worst_se, 'clauses': rows, 'release_sha256': distinct,
                'recorded_reserve_one_seed': {c: ev['results'][c]['reserve_one_seed'] for c in ('H3R', 'HCR')}},
           f'V_TRACKR_EVAL__{P}__{tag}.json')


def metrics2(y, p):
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    idx = np.arange(len(y)); onehot = np.zeros_like(p); onehot[idx, y] = 1.0
    return {'accuracy': (np.argmax(p, 1) == y).astype(np.float64), 'nll': -np.log(np.maximum(p[idx, y], V.NLL_FLOOR)),
            'brier': np.sum((p - onehot) ** 2, 1)}


def finish(cid, what, method, ok, numbers, record):
    chk = {'id': cid, 'kind': 'NUMERICAL_REPLAY', 'what': what, 'method': method, 'status': 'PASS' if ok else 'FAIL', 'numbers': numbers,
           'code_sha256': V.code_hashes(__file__, V.__file__, HERE / 'v_refit.py')}
    V.write_record(record, chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({cid: chk['status']}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('teacher', 'step1', 'eval')); ap.add_argument('--partition', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    {'teacher': teacher, 'step1': step1, 'eval': evaluate}[a.mode](a.partition, a.tag)


if __name__ == '__main__':
    main()
