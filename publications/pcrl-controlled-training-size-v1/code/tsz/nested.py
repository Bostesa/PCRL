"""Nested strength-selection estimator of the conditional correction stage (PCRL controlled training-size campaign
`pcrl_controlled_training_size_v1`; EXPLORATORY DEVELOPMENT on previously used records, not confirmation). Runs ONLY if the
registered task-stage trigger fires. Labels: MODEL_FIT, registered subset S100, operation 'select_ncv' (fitting information only;
MODEL_SELECT never enters strength selection).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed). Occupation group (5 official broad civilian groups) from AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column
encoding); SEX (recorded demographic column) is a training label of the critics and an evaluation label of the out-of-fold
within-SEX costs; never a runtime input or output.

Outer folds: tsz.common.ncv_folds (10 household folds of all MODEL_FIT units, tag 'TSZ1_ncv10'). For outer fold k, training =
the other 9 folds (about 18,000 people), and EVERY upstream layer is refitted on it:
  1. witness  (python3)       FIT-only dictionary on the 90% (cts.trackf.fit_dictionary); A1_PO_16 witness with its own 5-fold
                              household cross-fitting base (ohn.fam_a.run_pipeline; outer folds ohn.common.crossfit_folds and inner
                              folds household_folds(.., 5, 'OHN_A_cv_S<v>'), both computed on ALL MODEL_FIT units and restricted to
                              the 90%); the fold-k witness = its deployed refit applied to fold k. Resumable per fold (atomic fold
                              directories); `--folds 0-4` / `--folds 5-9` split the step into two units.
  2. routing  (CatBoost venv) the T_TREE occupation recipe (cts.trackf._catboost_teacher: internal 15% household split with the
                              two-task class-support repair of seed v, CatBoost seed seed_for('ACSR1catboost', v, 'occupation', 4),
                              refit) on the 90%, recoded with that fold's dictionary; parity with CatBoost checked on both sides.
  3. correct  (torch venv)    LOC / PAIR (CONT optional) corrections trained on the 90% from the nested base (ohn.protect.
                              train_correction, P2 recipe, guard 'none', seed 'TSZ1_V<v>', fold 'ncv<k>'; PAIR reads the frozen H3
                              income S<v> wire), applied to fold k with fold k's witness and routing; per-person incremental costs
                              against s = 0 on fold k at every strength, pooled over the 10 folds (all 20,000 people) and summarized
                              per strength by cts.trackf.cost_summary (means + household-clustered SEs of men_brier, women_brier,
                              brier, nll, acc_loss) -> NESTED_<arm>.json (schema TSZ1-nested-v1) + NESTED_<arm>.npz.
  bench       (any)           label-free compute benchmark: one outer fold of a step on the REAL MODEL_FIT features with INVENTED
                              labels / SEX (drawn from frozen model outputs; no label file is opened).
Outputs: .private/<campaign>/runs/NESTED/V<v>/ (write-once). Real-data commands run only through `python3 -m tsz.jobqueue run`.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import correct as CR
from cts import trackf as TF

K_OCC = 5
NCV_ARMS = ('LOC', 'PAIR', 'CONT')
STATS = ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')
DISCLOSURE = ("Pooled cross-fitted costs target 90%-size submodels (each outer fold's dictionary, witness, cross-fitted base, "
              "routing and correction are trained on about 18,000 MODEL_FIT people) and do not guarantee the deployed 100% model's "
              'costs. ' + CR.H3_NOTE + '. MODEL_SELECT never enters this estimator.')
BENCH_DIR = C.RESULTS / 'roles' / 'correction_investigator' / 'bench'


def ndir(v):
    return C.RUNS / 'NESTED' / f'V{int(v)}'


def parse_folds(spec):
    out = set()
    for part in str(spec).split(','):
        if '-' in part:
            a, b = part.split('-'); out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    if not out or min(out) < 0 or max(out) >= C.NCV_K:
        raise ValueError(f'folds must lie in 0..{C.NCV_K - 1}: {spec}')
    return sorted(out)


def _load(fields, purpose):
    return C.load('MODEL_FIT', fields, 'select_ncv', purpose, subset='S100')


def _dict_sha(dic):
    return hashlib.sha256(json.dumps(dic, sort_keys=True).encode()).hexdigest()


def _fold_done(fd, recname):
    p = fd / recname
    if not p.is_file():
        return None
    rec = json.loads(p.read_text())
    for n, h in rec['files_sha256'].items():
        if C.sha(fd / n) != h:
            raise PermissionError(f'{fd.name}/{n} differs from its record')
    return rec


def _atomic_fold(parent, k):
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return Path(tempfile.mkdtemp(prefix=f'.F{k}.partial.', dir=parent))


# ------------------------------------------------------------------ 1. witness (python3)
def witness(v, folds, dry=None):
    from ohn import fam_a as FA
    t0 = time.process_time()
    root = Path(dry['root']) if dry else ndir(v)
    wd = root / 'WITNESS'
    if dry:
        X, names, y, unit = dry['X'], dry['names'], dry['y'], dry['unit']
    else:
        F = _load(('features', 'occupation'), f'nested strength selection V{v}: A1_PO_16 witness and its 5-fold cross-fitted base '
                  f'refitted inside outer folds {folds} of the 10-fold household split (occupation only, no SEX)')
        X, names, y, unit = F['X'], F['feature_names'], F['occupation'], F['unit']
    cfg = dict(dry['cfg']) if dry else dict(FA.CONFIGS[CR.W_RECIPE])
    ncv = C.ncv_folds(unit)
    done = {}
    for k in folds:
        fd = wd / f'F{k}'
        rec = _fold_done(fd, 'WITNESS.json')
        if rec is not None:
            done[k] = 'already_done'; continue
        tf = time.process_time()
        rk, tk = ncv != k, ncv == k
        dic = TF.fit_dictionary(X[rk], names)
        res = CR.base_pipeline(cfg, v, CR.witness_rep(X[rk], names, dic), y[rk], unit, rk)
        pipe = res['pipe']
        w_fold = FA.release_prob(pipe['deployed'], CR.witness_rep(X[tk], names, dic))
        tmp = _atomic_fold(wd, k)
        files = {}
        with open(tmp / 'MODEL.pkl', 'xb') as f:
            pickle.dump(pipe['deployed'][0], f, protocol=5)
        files['MODEL.pkl'] = C.sha(tmp / 'MODEL.pkl')
        C.write_json_x(tmp / 'DICT.json', dic); files['DICT.json'] = C.sha(tmp / 'DICT.json')
        files['WITNESS.npz'] = C.save_npz_x(tmp / 'WITNESS.npz', base_oof=res['oof'], witness_fold=w_fold,
                                            train_idx=np.flatnonzero(rk), test_idx=np.flatnonzero(tk))
        rec = {'schema': 'TSZ1-nested-witness-fold-v1', 'version': int(v), 'fold': int(k), 'recipe': CR.W_RECIPE, 'config': cfg,
               'T_star': pipe['T_star'], 'T_k': {str(a): b for a, b in pipe['T_k'].items()},
               'training': CR.sizes(unit, rk), 'held_out': CR.sizes(unit, tk), 'base_crossfit_fold_sizes': res['crossfit_fold_sizes'],
               'submodel_training_sizes': res['submodel_training_sizes'],
               'dictionary': {'n_collapsed': dic['n_collapsed'], 'dict_sha256': _dict_sha(dic)},
               'n_physical_fits': pipe['n_physical_fits'], 'files_sha256': files, 'labels_used': 'MODEL_FIT occupation (90%)',
               'cpu_s': time.process_time() - tf, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV), 'interpreter': sys.executable}
        C.write_json_x(tmp / 'WITNESS.json', rec)
        os.rename(tmp, fd)                                           # atomic: a fold directory is either complete or absent
        done[k] = {'T_star': pipe['T_star'], 'cpu_s': round(rec['cpu_s'], 1)}
        C.unit_stats(n_nested_witness_fits=pipe['n_physical_fits'])
        print(json.dumps({'version': v, 'fold': k, **done[k]}), flush=True)
    summ = None
    if all(_fold_done(wd / f'F{k}', 'WITNESS.json') is not None for k in range(C.NCV_K)) and not (root / 'WITNESS_NESTED.json').exists():
        recs = {str(k): _fold_done(wd / f'F{k}', 'WITNESS.json') for k in range(C.NCV_K)}
        summ = {'schema': 'TSZ1-nested-witness-v1', 'version': int(v), 'folds': {k: {x: r[x] for x in ('T_star', 'training', 'held_out',
                'dictionary', 'n_physical_fits', 'files_sha256', 'cpu_s')} for k, r in recs.items()},
                'outer_tag': C.NCV_TAG, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
        C.write_json_x(root / 'WITNESS_NESTED.json', summ)
    return {'done': done, 'summary_written': summ is not None, 'cpu_s': time.process_time() - t0}


# ------------------------------------------------------------------ 2. routing (CatBoost venv)
def routing(v, folds=None, dry=None):
    from acs_replication import trees
    t0 = time.process_time()
    folds = list(range(C.NCV_K)) if folds is None else folds
    root = Path(dry['root']) if dry else ndir(v)
    rd = root / 'ROUTING'
    if dry:
        X, names, y, inc, unit = dry['X'], dry['names'], dry['y'], dry['income_label'], dry['unit']
    else:
        F = _load(('features', 'income', 'occupation'), f'nested strength selection V{v}: T_TREE occupation routing refitted inside '
                  'each outer fold of the 10-fold household split (income only for the internal-split class-support repair; no SEX)')
        X, names, y, inc, unit = F['X'], F['feature_names'], F['occupation'], F['income'], F['unit']
    ncv = C.ncv_folds(unit)
    done = {}
    for k in folds:
        fd = rd / f'F{k}'
        if _fold_done(fd, 'ROUTING.json') is not None:
            done[k] = 'already_done'; continue
        tf = time.process_time()
        rk, tk = ncv != k, ncv == k
        dic = TF.fit_dictionary(X[rk], names)
        Xtr, Xte = TF.recode(X[rk], names, dic), TF.recode(X[tk], names, dic)
        te, model, info = TF._catboost_teacher(Xtr, y[rk], unit[rk], int(v), names, ('ACSR1catboost', int(v), 'occupation', 4),
                                               y_income=inc[rk])
        par = {'train': trees.verify_against_catboost(te, model, Xtr, names), 'held_out': trees.verify_against_catboost(te, model, Xte, names)}
        if not all(p['routing_identical'] and p['raw_max_abs_diff'] <= 1e-9 for p in par.values()):
            raise AssertionError(f'teacher conversion parity failed: {par}')
        tmp = _atomic_fold(rd, k)
        with open(tmp / 'TEACHER.json', 'x') as f:
            json.dump(te.payload(), f, sort_keys=True)
        rec = {'schema': 'TSZ1-nested-routing-fold-v1', 'version': int(v), 'fold': int(k), **info, 'T': te.T, 'L': te.L,
               'teacher_sha256': te.sha(), 'files_sha256': {'TEACHER.json': C.sha(tmp / 'TEACHER.json')}, 'parity': par,
               'training': CR.sizes(unit, rk), 'dictionary_sha256': _dict_sha(dic), 'catboost_version': trees.CATBOOST_VERSION,
               'labels_used': 'MODEL_FIT occupation (90%); income only for the internal-split class-support repair',
               'cpu_s': time.process_time() - tf, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV), 'interpreter': sys.executable}
        C.write_json_x(tmp / 'ROUTING.json', rec)
        os.rename(tmp, fd)
        done[k] = {'tree_count': info['tree_count'], 'cpu_s': round(rec['cpu_s'], 1)}
        C.unit_stats(n_routing_fits=2)
        print(json.dumps({'version': v, 'fold': k, **done[k]}), flush=True)
    summ = None
    if all(_fold_done(rd / f'F{k}', 'ROUTING.json') is not None for k in range(C.NCV_K)) and not (root / 'ROUTING_NESTED.json').exists():
        recs = {str(k): _fold_done(rd / f'F{k}', 'ROUTING.json') for k in range(C.NCV_K)}
        summ = {'schema': 'TSZ1-nested-routing-v1', 'version': int(v), 'folds': {k: {x: r[x] for x in ('tree_count', 'T', 'teacher_sha256',
                'internal_fit_rows', 'internal_validation_rows', 'moved_households', 'training', 'cpu_s')} for k, r in recs.items()},
                'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
        C.write_json_x(root / 'ROUTING_NESTED.json', summ)
    return {'done': done, 'summary_written': summ is not None, 'cpu_s': time.process_time() - t0}


# ------------------------------------------------------------------ 3. correct (torch venv) and the pooled cost table
def cost_table(pw, dn, y, sex, unit, strengths):
    """Per strength: cts.trackf.cost_summary of r_s against r_0 on the same people (pooled out-of-fold arrays)."""
    base0 = CR.release(pw, dn, 0.0)                              # = the fold's own witness, exactly
    return {CR.skey(s): TF.cost_summary(CR.release(pw, dn, s), base0, y, sex, unit) for s in strengths}


def fold_stats(pw, dn, y, sex, strengths):
    base0 = CR.release(pw, dn, 0.0)
    out = {}
    for s in strengths:
        c = TF.cost_vectors(CR.release(pw, dn, s), base0, y)
        out[CR.skey(s)] = {'men_brier': float(c['brier'][sex == 0].mean()) if (sex == 0).any() else None,
                           'women_brier': float(c['brier'][sex == 1].mean()) if (sex == 1).any() else None,
                           'brier': float(c['brier'].mean()), 'nll': float(c['nll'].mean()), 'acc_loss': float(c['acc_loss'].mean())}
    return out


def correct(v, arm, dry=None):
    from acs_replication import trees
    from ohn import protect as PR, devrun as OD
    if arm not in NCV_ARMS:
        raise ValueError(arm)
    t0 = time.process_time(); w0 = time.time()
    root = Path(dry['root']) if dry else ndir(v)
    outp = root / f'NESTED_{arm}.json'
    if outp.exists() or (root / f'NESTED_{arm}.npz').exists():
        raise PermissionError(f'{outp} exists (write-once)')
    S, ssrc = dry['strengths'] if dry else CR.strength_set(strict=True)
    if dry:
        X, names, y, sex, unit, inc = dry['X'], dry['names'], dry['y'], dry['sex'], dry['unit'], dry['income']
        epochs, steps = dry.get('epochs'), dry.get('steps')
        row_id = np.arange(len(y))
    else:
        F = _load(('features', 'occupation', 'sex'), f'nested strength selection V{v}: {arm} correction refitted inside each outer fold '
                  'from the nested base; out-of-fold incremental task and within-SEX costs on the held-out fold')
        X, names, y, sex, unit, row_id = F['X'], F['feature_names'], F['occupation'], F['sex'], F['unit'], F['row_id']
        inc = OD.anchor_income('H3', int(v), 'MODEL_FIT')[F['rows']]
        epochs = steps = None
    inc_wire = PR.income_wire(inc)
    targets, tsrc = CR.guard_targets(strict=not dry)
    if tuple(PR.SLACK_OF) != STATS:
        raise AssertionError('the five statistics differ from ohn.protect.SLACK_OF')
    ncv = C.ncv_folds(unit)
    n = len(y)
    pw = np.full((n, K_OCC), np.nan); dn = np.full((n, K_OCC), np.nan)
    folds, inputs = {}, {}
    for k in range(C.NCV_K):
        wrec = _fold_done(root / 'WITNESS' / f'F{k}', 'WITNESS.json')
        rrec = _fold_done(root / 'ROUTING' / f'F{k}', 'ROUTING.json')
        if wrec is None or rrec is None:
            raise FileNotFoundError(f'outer fold {k}: witness or routing step incomplete')
        W = C.load_npz(root / 'WITNESS' / f'F{k}' / 'WITNESS.npz')
        rk, tk = ncv != k, ncv == k
        if not (np.array_equal(W['train_idx'], np.flatnonzero(rk)) and np.array_equal(W['test_idx'], np.flatnonzero(tk))):
            raise AssertionError(f'outer fold {k}: witness rows differ')
        dic = TF.fit_dictionary(X[rk], names)
        if _dict_sha(dic) != wrec['dictionary']['dict_sha256'] or _dict_sha(dic) != rrec['dictionary_sha256']:
            raise AssertionError(f'outer fold {k}: dictionary differs between steps')
        te = trees.TreeTeacher(json.loads((root / 'ROUTING' / f'F{k}' / 'TEACHER.json').read_text()))
        if te.sha() != rrec['teacher_sha256']:
            raise PermissionError(f'outer fold {k}: routing payload differs')
        lv_tr = te.route(TF.recode(X[rk], names, dic), names)
        lv_te = te.route(TF.recode(X[tk], names, dic), names)
        r = PR.train_correction(TF.logp(W['base_oof']), lv_tr, y[rk], sex[rk], inc_wire[rk], te.T, te.L, arm, CR.seed_str(v), f'ncv{k}',
                                targets, epochs=epochs, steps=steps, guard='none')
        dn[tk] = PR.correction(r['theta'], lv_te, te.T, te.L)
        pw[tk] = W['witness_fold']
        folds[str(k)] = {'training': CR.sizes(unit, rk), 'held_out': CR.sizes(unit, tk),
                         'witness': {'T_star': wrec['T_star'], 'training': wrec['training'], 'base_crossfit_fold_sizes': wrec['base_crossfit_fold_sizes'],
                                     'submodel_training_sizes': wrec['submodel_training_sizes']},
                         'routing': {'tree_count': rrec['tree_count'], 'internal_fit_rows': rrec['internal_fit_rows'],
                                     'internal_validation_rows': rrec['internal_validation_rows'], 'teacher_sha256': rrec['teacher_sha256']},
                         'correction': {'lambda': r['lambda'], 'theta_sha256': C.arr_sha(r['theta']), 'cpu_s': r['cpu_s'],
                                        'summary': PR.fold_summary(r)},
                         'held_out_costs': fold_stats(pw[tk], dn[tk], y[tk], sex[tk], S)}
        inputs[str(k)] = {'witness_files_sha256': wrec['files_sha256'], 'routing_files_sha256': rrec['files_sha256']}
        C.unit_stats(n_nested_correction_fits=1, n_nested_critic_fits=9)
        print(json.dumps({'version': v, 'arm': arm, 'fold': k, 'lambda': round(r['lambda'], 4), 'cpu_s': round(r['cpu_s'], 1)}), flush=True)
    if not (np.isfinite(pw).all() and np.isfinite(dn).all()):
        raise AssertionError('pooled out-of-fold arrays incomplete')
    table = cost_table(pw, dn, y, sex, unit, S)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    npz_sha = C.save_npz_x(root / f'NESTED_{arm}.npz', witness_oof=pw, delta_oof=dn, ncv_fold=ncv, row_id=np.asarray(row_id))
    rec = {'schema': 'TSZ1-nested-v1', 'version': int(v), 'arm': arm, 'seed': CR.seed_str(v), 'guard': 'none', 'targets': targets,
           'targets_source': tsrc,
           'estimator': 'nested 10-fold household cross-fitting (every upstream layer refitted on the 90%); per-person incremental costs '
                        'against s = 0 on the held-out fold, pooled over the 10 folds; cts.trackf.cost_summary per strength',
           'outer_folds': {'tag': C.NCV_TAG, 'k': C.NCV_K, 'sizes': {str(k): CR.sizes(unit, ncv == k) for k in range(C.NCV_K)}},
           'strengths': list(S), 'strength_source': ssrc, 'stats': list(STATS), 'per_strength': table,
           'pooled': {**CR.sizes(unit), 'n_men': int((sex == 0).sum()), 'n_women': int((sex == 1).sum())},
           'folds': folds, 'inputs_sha256': inputs, 'npz_sha256': npz_sha,
           'h3_income': {'array_sha256': C.arr_sha(inc), 'frozen': True, 'fitted_on': 'all MODEL_FIT (source campaign)',
                         'used': 'PAIR critic income wire during training only' if arm == 'PAIR' else 'not used by this arm'},
           'disclosure': DISCLOSURE, 'labels_used': "MODEL_FIT occupation and SEX, subset S100, operation 'select_ncv'",
           'interpreter': sys.executable, 'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(outp, rec)
    print(json.dumps({'version': v, 'arm': arm, 'cpu_s': round(rec['cpu_s'], 1),
                      'per_strength': {s: {k: round(t['stats'][k], 5) for k in STATS} for s, t in table.items()}}), flush=True)
    return rec


def load_nested(v, arm, root=None):
    root = Path(root) if root is not None else ndir(v)
    rec = json.loads((root / f'NESTED_{arm}.json').read_text())
    if C.sha(root / f'NESTED_{arm}.npz') != rec['npz_sha256']:
        raise PermissionError(f'NESTED_{arm}.npz differs from its record')
    return rec


# ------------------------------------------------------------------ label-free compute benchmark
def _bench_inputs(seed=7):
    """REAL MODEL_FIT features (label-free) with INVENTED occupation / income labels and SEX drawn from frozen model outputs."""
    from ohn import common as OC, devrun as OD
    F = C.features('MODEL_FIT')
    names, X, unit = F['feature_names'], F['X'], F['unit']
    te = OC.teacher(0, 'occupation')
    lv = te.route(X, names)
    p_t = te.probabilities_from_raw(te.raw_from_leaves(lv))
    rng = np.random.default_rng(seed)
    y = (rng.random(len(X))[:, None] > np.cumsum(p_t, 1)).sum(1).clip(0, K_OCC - 1).astype(np.int64)
    inc_p = OD.anchor_income('H3', 0, 'MODEL_FIT')
    inc_y = (rng.random(len(X)) < inc_p[:, 1]).astype(np.int64)
    wk = X[:, list(names).index('WKHP')]
    sex = (rng.random(len(X)) < 1 / (1 + np.exp(-(-0.8 * wk + 3.0 * (p_t[:, 1] + p_t[:, 2] - p_t[:, 3] - p_t[:, 4]))))).astype(np.int64)
    return {'X': X, 'names': names, 'unit': unit, 'y': y, 'income_label': inc_y, 'income': inc_p, 'sex': sex, 'teacher': te, 'p_t': p_t}


def bench(step, fold=0, arm='PAIR'):
    """Time one outer fold of a step at the real size (label-free); writes bench/BENCH_<step>[_<arm>].json (overwrite allowed)."""
    t0 = time.process_time(); w0 = time.time()
    B = _bench_inputs()
    X, names, unit, y = B['X'], B['names'], B['unit'], B['y']
    ncv = C.ncv_folds(unit)
    rk, tk = ncv != fold, ncv == fold
    setup = time.process_time() - t0
    out = {'step': step, 'fold': fold, 'training': CR.sizes(unit, rk), 'held_out': CR.sizes(unit, tk), 'setup_cpu_s': setup,
           'data': 'REAL MODEL_FIT features (label-free); INVENTED occupation (sampled from the admitted T_TREE S0 probabilities), income '
                   '(sampled from H3 S0 probabilities) and SEX (logistic in WKHP and teacher probabilities); no label file opened',
           'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
           'interpreter': sys.executable}
    tw = time.process_time(); ww = time.time()
    if step == 'witness':
        from ohn import fam_a as FA
        cfg = dict(FA.CONFIGS[CR.W_RECIPE])
        dic = TF.fit_dictionary(X[rk], names)
        res = CR.base_pipeline(cfg, 0, CR.witness_rep(X[rk], names, dic), y[rk], unit, rk)
        FA.release_prob(res['pipe']['deployed'], CR.witness_rep(X[tk], names, dic))
        out.update({'T_star': res['pipe']['T_star'], 'T_k': {str(a): b for a, b in res['pipe']['T_k'].items()},
                    'n_physical_fits': res['pipe']['n_physical_fits'], 'n_collapsed': dic['n_collapsed']})
    elif step == 'routing':
        from acs_replication import trees
        dic = TF.fit_dictionary(X[rk], names)
        Xtr = TF.recode(X[rk], names, dic)
        te, model, info = TF._catboost_teacher(Xtr, y[rk], unit[rk], 0, names, ('ACSR1catboost', 0, 'occupation', 4), y_income=B['income_label'][rk])
        par = trees.verify_against_catboost(te, model, TF.recode(X[tk], names, dic), names)
        out.update({'tree_count': info['tree_count'], 'parity_ok': bool(par['routing_identical'] and par['raw_max_abs_diff'] <= 1e-9)})
    elif step in ('correct', 'direction'):
        from ohn import protect as PR
        rows = rk if step == 'correct' else np.ones(len(y), dtype=bool)
        te = B['teacher']
        lv = te.route(X, names)
        z = TF.logp(B['p_t'])                                       # stand-in base (frozen teacher output; label-free)
        r = PR.train_correction(z[rows], lv[rows], y[rows], B['sex'][rows], PR.income_wire(B['income'])[rows], te.T, te.L, arm,
                                'TSZ1_BENCH', f'bench{fold}', CR.TARGETS, guard='none')
        out.update({'arm': arm, 'lambda': r['lambda'], 'rows': int(rows.sum()), 'epochs': r['epochs'], 'steps_per_epoch': r['steps_per_epoch']})
    else:
        raise ValueError(step)
    out['step_cpu_s'] = time.process_time() - tw; out['step_wall_s'] = time.time() - ww
    out['total_cpu_s'] = time.process_time() - t0; out['total_wall_s'] = time.time() - w0
    out['utc'] = C.now(); out['unit_id'] = os.environ.get(C.UNIT_ENV)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    name = f'BENCH_{step}' + (f'_{arm}' if step in ('correct', 'direction') else '') + '.json'
    C.write_json(BENCH_DIR / name, out)
    print(json.dumps(out, default=str))
    return out


# ------------------------------------------------------------------ selftest (invented data only)
def selftest():
    from ohn import fam_a as FA, common as OC
    import importlib.util as IU
    t0 = time.process_time()
    rng = np.random.default_rng(20261020)
    checks = {}
    checks['parse_folds'] = parse_folds('0-4') == [0, 1, 2, 3, 4] and parse_folds('5-9,2') == [2, 5, 6, 7, 8, 9]
    try:
        parse_folds('0-10'); checks['parse_folds_range'] = False
    except ValueError:
        checks['parse_folds_range'] = True
    D = CR.invented(rng, n=1400)
    X, names, y, unit, sex = D['X'], D['names'], D['y'], D['unit'], D['sex']
    ncv = C.ncv_folds(unit)
    checks['ncv_household_separated'] = bool(all(len(np.unique(ncv[unit == u])) == 1 for u in np.unique(unit))) and len(np.unique(ncv)) == 10
    # cost table: zero at s = 0, pooled per-person vectors equal the per-fold vectors, SE structure
    pw = rng.dirichlet(np.ones(K_OCC), size=len(y)); dn = rng.normal(size=(len(y), K_OCC))
    tab = cost_table(pw, dn, y, sex, unit, (0.0, 0.1, 1.0))
    checks['cost_zero_at_identity'] = all(abs(x) == 0.0 for x in tab['0.0']['stats'].values())
    checks['identity_release_is_witness_exactly'] = bool(np.array_equal(CR.release(pw, dn, 0.0), pw))
    v_all = TF.cost_vectors(CR.release(pw, dn, 0.1), pw, y)
    parts = [TF.cost_vectors(CR.release(pw[ncv == k], dn[ncv == k], 0.1), pw[ncv == k], y[ncv == k])
             for k in range(10)]
    pooled = np.zeros(len(y))
    for k in range(10):
        pooled[ncv == k] = parts[k]['nll']
    checks['pooling_equals_per_fold_vectors'] = bool(np.array_equal(pooled, v_all['nll']))
    checks['cost_table_has_all_stats_and_se'] = set(tab['0.1']['stats']) == set(STATS) and set(tab['0.1']['se']) == set(STATS)
    # the three steps on invented data in a temporary tree (small witness configuration; CatBoost / torch parts when available)
    tmp = Path(tempfile.mkdtemp(prefix='tsz_nested_selftest_'))
    cfg = {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.1, 'leaves': 8, 'msl': 20, 'l2': 1.0, 'max_iter': 20, 'calibrator': None,
           'deploy': 'refit', 'kind': 'task_model'}
    have = {m: IU.find_spec(m) is not None for m in ('sklearn', 'catboost', 'torch')}
    dry = {'root': tmp, 'X': X, 'names': names, 'y': y, 'unit': unit, 'cfg': cfg, 'sex': sex, 'income': D['income'],
           'income_label': (rng.random(len(y)) < D['income'][:, 1]).astype(np.int64), 'strengths': ((0.0, 0.03, 0.3, 1.0), 'selftest'),
           'epochs': 1, 'steps': 3}
    shared = Path(tempfile.gettempdir()) / 'tsz_nested_selftest_shared'        # cross-interpreter hand-off of the invented tree
    if have['sklearn'] and not have['catboost'] and not have['torch']:
        r1 = witness(0, [0, 1, 2, 3, 4], dry=dry)
        r2 = witness(0, [3, 4, 5, 6, 7, 8, 9], dry=dry)                         # resumes: 3, 4 already done
        checks['witness_resumable'] = r2['done'][3] == 'already_done' and r2['done'][4] == 'already_done' and r2['summary_written']
        w5 = C.load_npz(tmp / 'WITNESS' / 'F5' / 'WITNESS.npz')
        rk = ncv != 5
        dic = TF.fit_dictionary(X[rk], names)
        res = CR.base_pipeline(cfg, 0, CR.witness_rep(X[rk], names, dic), y[rk], unit, rk)
        checks['witness_fold_reproducible_bitwise'] = bool(np.array_equal(res['oof'], w5['base_oof']) and np.array_equal(
            FA.release_prob(res['pipe']['deployed'], CR.witness_rep(X[ncv == 5], names, dic)), w5['witness_fold']))
        checks['witness_fold_rows'] = bool(np.array_equal(w5['test_idx'], np.flatnonzero(ncv == 5)))
        if shared.exists():
            shutil.rmtree(shared)
        shutil.copytree(tmp, shared)
        np.savez(shared / 'INVENTED.npz', X=X, y=y, unit=unit, sex=sex, income=D['income'], income_label=dry['income_label'])
        (shared / 'names.json').write_text(json.dumps(list(names)))
        (shared / 'teacher.json').write_text(json.dumps(D['teacher'].payload()))
    elif have['catboost'] or have['torch']:
        if not (shared / 'INVENTED.npz').is_file():
            raise FileNotFoundError('run the python3 selftest first (it writes the invented hand-off tree)')
        Z = C.load_npz(shared / 'INVENTED.npz')
        dry.update({'root': shared, 'X': Z['X'], 'y': Z['y'], 'unit': Z['unit'], 'sex': Z['sex'], 'income': Z['income'],
                    'income_label': Z['income_label'], 'names': tuple(json.loads((shared / 'names.json').read_text()))})
        if have['catboost']:
            routing(0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dry=dry)
            r = routing(0, [0], dry=dry)
            checks['routing_complete_and_resumable'] = r['done'][0] == 'already_done' and (shared / 'ROUTING_NESTED.json').is_file()
        if have['torch']:
            if not (shared / 'ROUTING_NESTED.json').is_file():
                raise FileNotFoundError('run the CatBoost selftest before the torch selftest')
            for f in shared.glob('NESTED_*'):
                f.unlink()
            rec = correct(0, 'LOC', dry=dry)
            checks['correct_table_schema'] = rec['schema'] == 'TSZ1-nested-v1' and set(rec['per_strength']) == {'0.0', '0.03', '0.3', '1.0'}
            checks['correct_zero_at_identity'] = all(x == 0.0 for x in rec['per_strength']['0.0']['stats'].values())
            checks['correct_pooled_all_people'] = rec['per_strength']['0.3']['n'] == len(dry['y']) and len(rec['folds']) == 10
            recP = correct(0, 'PAIR', dry=dry)
            checks['correct_arms_differ'] = recP['folds']['0']['correction']['theta_sha256'] != rec['folds']['0']['correction']['theta_sha256']
            loaded = load_nested(0, 'LOC', shared)
            checks['nested_npz_hash_checked'] = loaded['npz_sha256'] == rec['npz_sha256']
            try:
                correct(0, 'LOC', dry=dry); checks['correct_write_once'] = False
            except PermissionError:
                checks['correct_write_once'] = True
            checks['disclosure_present'] = '90%-size submodels' in rec['disclosure'] and 'FROZEN' in rec['disclosure']
    ok = all(checks.values())
    out = {'selftest': 'PASS' if ok else 'FAIL', 'module': 'tsz.nested', 'interpreter': sys.executable, 'packages': have,
           'checks': checks, 'data': 'invented only (numpy default_rng(20261020)); no real data or label read',
           'cpu_s': round(time.process_time() - t0, 1)}
    print(json.dumps(out, default=str))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    x = sub.add_parser('witness'); x.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    x.add_argument('--folds', default='0-9')
    x = sub.add_parser('routing'); x.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    x.add_argument('--folds', default='0-9')
    x = sub.add_parser('correct'); x.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    x.add_argument('--arm', required=True, choices=NCV_ARMS)
    x = sub.add_parser('bench'); x.add_argument('--step', required=True, choices=('witness', 'routing', 'correct', 'direction'))
    x.add_argument('--fold', type=int, default=0); x.add_argument('--arm', default='PAIR', choices=CR.ARMS)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return selftest()
    if a.cmd == 'witness':
        witness(a.version, parse_folds(a.folds))
    elif a.cmd == 'routing':
        routing(a.version, parse_folds(a.folds))
    elif a.cmd == 'correct':
        correct(a.version, a.arm)
    else:
        bench(a.step, a.fold, a.arm)
    return 0


if __name__ == '__main__':
    sys.exit(main())
