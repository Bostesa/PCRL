"""Nested versus non-nested out-of-fold cost estimation (PCRL correction-transfer campaign, Track F diagnosis).

The source estimator (OOF in cts.trackf.ladder) trains fold pipeline k on FIT minus fold k, but its UPSTREAM layers are not nested:
the correction's training base on FIT minus fold k comes from witness fold pipelines j != k that trained on fold k, and the routing
(REFIT: fitted on all FIT; FROZEN: on all MODEL_FIT) saw fold k. Here every upstream layer is refitted inside each outer fold k:
  1. witness  (python3)      dictionary, A1_PO_16 witness (inner CV) and its own household cross-fitting, all on FIT minus fold k;
  2. routing  (CatBoost venv) the T_TREE occupation recipe on FIT minus fold k (recoded with that fold's dictionary);
  3. correct  (torch venv)    the correction trained on FIT minus fold k from the nested base (same recipe and random streams as the
                              non-nested fold pipeline k: seed = partition, fold = 'fold<k>'), applied to fold k with the fold-k
                              witness and routing; nested OOF incremental costs on the fixed ladder (FIT labels, operation 'fit').
    python3 -m cts.nested witness --partition D0
    <catboost python> -m cts.nested routing --partition D0
    <torch python> -m cts.nested correct --partition D0 --arm PAIR
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from . import common as C
from . import trackf as TF


def ndir(P):
    return TF.pdir(P) / 'NESTED'


def _fit(P, purpose, fields=('features', 'occupation')):
    return C.load('MODEL_FIT', fields, 'fit', purpose, partition=P, prole='FIT')


def witness(P):
    from ohn import fam_a as FA
    t0 = time.process_time()
    d = ndir(P)
    if (d / 'WITNESS_NESTED.json').exists():
        raise PermissionError('exists (append-only)')
    s = TF.seed_of(P)
    F = _fit(P, f'nested diagnosis {P}: witness pipelines refitted inside each outer fold of FIT (no SEX)')
    wrec, W, O = TF.load_witness(P)
    folds = O['fold']
    if not np.array_equal(F['row_id'], O['row_id']):
        raise AssertionError('FIT rows differ')
    names, y, unit = F['feature_names'], F['occupation'], F['unit']
    cfg = dict(FA.CONFIGS[TF.W_RECIPE])
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    out, recs, fits = {}, {}, 0
    for k in range(5):
        rk, tk = folds != k, folds == k
        dic = TF.fit_dictionary(F['X'][rk], names)
        C.write_json(d / f'DICT_{k}.json', dic)
        Xk = FA.rep_matrix(cfg['rep'], TF.recode(F['X'][rk], names, dic), names)
        Xt = FA.rep_matrix(cfg['rep'], TF.recode(F['X'][tk], names, dic), names)
        inner = C.household_folds(unit[rk], 5, f'CTS1_inner_{P}')
        ncf = C.household_folds(unit[rk], 5, f'CTS1_ncf_{P}_{k}')
        pipe = FA.run_pipeline(cfg, s, Xk, y[rk], ncf, inner, log=lambda m: None)
        oof_k = np.zeros((int(rk.sum()), TF.K_OCC))
        for j in range(pipe['K']):
            oof_k[ncf == j] = FA.combine(pipe['oof_raw'][j])
        out[f'base_oof__{k}'] = oof_k                                  # nested training base on FIT minus fold k
        out[f'witness_fold__{k}'] = FA.release_prob(pipe['deployed'], Xt)  # fold-k witness on fold k
        fits += pipe['n_physical_fits']
        recs[str(k)] = {'T_star': pipe['T_star'], 'T_k': {str(a): b for a, b in pipe['T_k'].items()}, 'n_rows': int(rk.sum()),
                        'n_collapsed': dic['n_collapsed'], 'physical_fits': pipe['n_physical_fits']}
        with open(d / f'WITNESS_{k}.pkl', 'xb') as f:
            pickle.dump(pipe['deployed'][0], f, protocol=5)
        print(json.dumps({'fold': k, **recs[str(k)]}), flush=True)
    sha = TF._save_npz(d / 'WITNESS_NESTED.npz', **out)
    rec = {'schema': 'CTS1-nested-witness-v1', 'partition': P, 'folds': recs, 'npz_sha256': sha, 'physical_fits': fits,
           'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(d / 'WITNESS_NESTED.json', rec)
    C.unit_stats(n_crossfit_fits=fits)
    return rec


def routing(P):
    from acs_replication import trees
    t0 = time.process_time()
    d = ndir(P)
    if (d / 'ROUTING_NESTED.json').exists():
        raise PermissionError('exists (append-only)')
    s = TF.seed_of(P)
    F = _fit(P, f'nested diagnosis {P}: routing (T_TREE occupation recipe) refitted inside each outer fold of FIT (no SEX; income '
             'only for the internal split class-support repair)', fields=('features', 'income', 'occupation'))
    wrec, W, O = TF.load_witness(P)
    folds = O['fold']
    names = F['feature_names']
    recs = {}
    for k in range(5):
        rk = folds != k
        dic = json.loads((d / f'DICT_{k}.json').read_text())
        X = TF.recode(F['X'][rk], names, dic)
        teacher, model, info = TF._catboost_teacher(X, F['occupation'][rk], F['unit'][rk], s, names, ('ACSR1catboost', s, 'occupation', 4),
                                                    y_income=F['income'][rk])
        par = trees.verify_against_catboost(teacher, model, X, names)
        if not (par['routing_identical'] and par['raw_max_abs_diff'] <= 1e-9):
            raise AssertionError(f'teacher parity failed: {par}')
        with open(d / f'TEACHER_{k}.json', 'x') as f:
            json.dump(teacher.payload(), f, sort_keys=True)
        recs[str(k)] = {**info, 'T': teacher.T, 'teacher_sha256': teacher.sha()}
        print(json.dumps({'fold': k, 'tree_count': info['tree_count']}), flush=True)
    rec = {'schema': 'CTS1-nested-routing-v1', 'partition': P, 'folds': recs, 'cpu_s': time.process_time() - t0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(d / 'ROUTING_NESTED.json', rec)
    C.unit_stats(n_routing_fits=10)
    return rec


def correct(P, arm):
    from acs_replication import trees
    from ohn import protect as PR, devrun as OD
    t0 = time.process_time()
    d = ndir(P)
    outp = d / f'NESTED_{arm}.json'
    if outp.exists():
        raise PermissionError('exists (append-only)')
    s = TF.seed_of(P)
    F = _fit(P, f'nested diagnosis {P}: correction {arm} refitted inside each outer fold of FIT from the nested base; nested OOF costs',
             fields=('features', 'occupation', 'sex'))
    wrec, W, O = TF.load_witness(P)
    folds = O['fold']
    names, y, sex, unit = F['feature_names'], F['occupation'], F['sex'], F['unit']
    with np.load(d / 'WITNESS_NESTED.npz') as z:
        WN = {k: z[k].copy() for k in z.files}
    inc_all = OD.anchor_income('H3', s, 'MODEL_FIT')
    inc_wire = PR.income_wire(inc_all[F['rows']])
    pw = np.zeros((len(y), TF.K_OCC)); dn = np.zeros((len(y), TF.K_OCC)); recs = {}
    for k in range(5):
        rk, tk = folds != k, folds == k
        dic = json.loads((d / f'DICT_{k}.json').read_text())
        te = trees.TreeTeacher(json.loads((d / f'TEACHER_{k}.json').read_text()))
        lv_tr = te.route(TF.recode(F['X'][rk], names, dic), names)
        lv_te = te.route(TF.recode(F['X'][tk], names, dic), names)
        r = PR.train_correction(TF.logp(WN[f'base_oof__{k}']), lv_tr, y[rk], sex[rk], inc_wire[rk], te.T, te.L, arm, P, f'fold{k}',
                                {'men': 1.0, 'women': 1.0}, guard='none')
        dn[tk] = PR.correction(r['theta'], lv_te, te.T, te.L)
        pw[tk] = WN[f'witness_fold__{k}']
        recs[str(k)] = {'lambda': r['lambda'], 'theta_sha256': C.arr_sha(r['theta']), 'cpu_s': r['cpu_s']}
    ladder = {str(sv): TF.cost_summary(TF.release_at(pw, dn, sv), pw, y, sex, unit) for sv in TF.LADDER}
    sha = TF._save_npz(d / f'NESTED_{arm}.npz', witness_oof=pw, delta_oof=dn)
    rec = {'schema': 'CTS1-nested-v1', 'partition': P, 'arm': arm, 'routing': 'REFIT (nested)', 'folds': recs, 'ladder_NESTED_OOF': ladder,
           'npz_sha256': sha, 'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(outp, rec)
    C.unit_stats(n_crossfit_fits=5 * 10)
    print(json.dumps({'partition': P, 'arm': arm, 'cpu_s': round(rec['cpu_s'], 1)}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('witness', 'routing'):
        x = sub.add_parser(c); x.add_argument('--partition', required=True)
    x = sub.add_parser('correct'); x.add_argument('--partition', required=True); x.add_argument('--arm', required=True)
    a = ap.parse_args(argv)
    {'witness': lambda: witness(a.partition), 'routing': lambda: routing(a.partition),
     'correct': lambda: correct(a.partition, a.arm)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
