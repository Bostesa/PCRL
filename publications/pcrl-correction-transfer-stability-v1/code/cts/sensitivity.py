"""Bounded sensitivity witness of the PCRL correction-transfer campaign (EXPLORATORY DEVELOPMENT on previously used records; not
confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed). Occupation group (5 official broad civilian groups) is predicted from AGEP, WKHP, COW, SCHL, MAR, POBP through
the admitted 145-column encoding; each recipient receives only its own float64 probability vector and argmax decision. This module
is TASK-ONLY: no SEX and no income enter any fit, input or output here (SEX stays a training-time label of the correction critics,
fitted elsewhere on FIT, and an evaluation label).

PROTOCOL.md section 6: "The sensitivity witness (B2_SINGLE recipe) is a bounded check on one partition." This module refits the
source B2_SINGLE occupation recipe of the occupation-headroom campaign (ohn.fam_b: LEARNERS['TREE_V2'], LEARNERS['ADD_V2'],
CONFIGS['B2_SINGLE']; the source functions are called unchanged) on partition P's FIT rows only (P = D0B: the D0 households):
  TREE_V2  HGB on the compact coding (POBP ordinal code; COW, MAR native categories), leaves {4, 8, 15} x min leaf {50, 200},
           lr 0.05, <= 500 iterations, tree count (checkpoints of 10) by household 5-fold inner CV, refit;
  ADD_V2   L2 multinomial logistic regression on the label-free additive basis (cubic B-splines of AGEP z / WKHP z, frequent-WKHP
           indicators, one-hot blocks, fixed interactions), capacity S / L x C in {0.003, 0.01, 0.03} by the same inner CV, refit;
  layer    B2_SINGLE: the learner with the lower temperature-scaled inner out-of-fold NLL, released as softmax(tau z_L*(x)).
Full nested household cross-fitting as the source (ohn.fam_b.outer_sets / run_base / fit_layer / crossfit_release): tuning, refit,
learner choice and temperature are redone inside each of the 5 outer folds. The outer folds are the main witness's folds,
household_folds(unit_FIT, 5, 'CTS1_cf_<base>') with base = the partition whose rows P shares (D0B -> D0); the inner folds are the
source's (outer_sets tags OHN_B_inner_S<seed>[_out<k>], here over FIT households).

FIT-only label-free preprocessing (no GUARD, CHECK or repair-panel row reaches any fit):
  dictionary  cts.trackf.fit_dictionary / recode on FIT (as the main witness);
  z-map       AGEP z and WKHP z re-standardized with FIT moments (mean, population SD; float64 then float32 as the admitted schema):
              a strictly increasing affine map of the admitted z, which carries all-MODEL_FIT moments. Needed because the ADD learner
              (L2 penalty; linear AGEP / WKHP columns and their products) is not invariant to that map; the tree learner is (up to
              bin-edge rounding);
  featurizer  ohn.fam_b.make_featurizer unchanged on the FIT inputs (spline knots over the FIT ranges; the WKHP values held by at
              least ohn.fam_b.WKHP_MIN_COUNT = 200 FIT people; the absolute count is the recipe's and is not rescaled).
They are fitted once on all FIT rows and shared by the 5 fold pipelines, as the source fitted its label-free steps once on all of its
training role (label-free; disclosed in the record).

Files (the layout of cts.trackf.fit_witness, read by cts.trackf.load_witness / fit_direction / ladder and cts.evaluate):
  pdir(P)/DICT.json; pdir(P)/WITNESS/WITNESS.json, WITNESS.npz (dep__<ROLE> for the six roles; fold__MODEL_FIT, fold__MODEL_SELECT
  = (5, n, 5) stacks of the 5 fold pipelines), OOF.npz (oof, fold, fit_rows, row_id), MODEL.pkl (deployed layer + base learners +
  z-map), FOLD_<k>.pkl (fold pipeline k). Every file is created exclusively (append-only): move a failed attempt's partial
  directory aside before a retry.

Commands:
  python3 -m cts.sensitivity selftest      invented data only (no real data; the access layer and loaders are replaced by fakes)
  real fit (coordinator, through the queue):
    python3 -m cts.jobqueue run --unit CTS_<NAME>_V1 --stage fit --rss-gib 2 -- python3 -m cts.trackf witness --partition D0B
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import trackf as TF
from ohn import fam_b as FB

RECIPE = 'B2_SINGLE'
K_OCC = 5
N_OUTER = 5
ROLES = C.ROLES6
ZCOLS = (0, 1)                                    # AGEP z, WKHP z (admitted column order, checked in fit_zmap)
SCHEMA = 'CTS1-witness-v1'                        # the main witness's record schema (same file layout)
MODEL_SCHEMA = 'CTS1-b2-model-v1'


def base_partition(P):
    return C.PARTITIONS[P].get('same_rows_as', P)


def outer_folds(P, unit_fit):
    """The main witness's outer household folds on FIT (D0B shares D0's)."""
    return C.household_folds(unit_fit, N_OUTER, f'CTS1_cf_{base_partition(P)}')


# ------------------------------------------------------------------ FIT-only label-free preprocessing
def fit_zmap(X_fit, names):
    names = tuple(map(str, names))
    if names[:2] != ('AGEP', 'WKHP'):
        raise ValueError('unexpected admitted encoding')
    X_fit = np.asarray(X_fit)
    out = {'columns': ['AGEP', 'WKHP'], 'mean': [], 'sd': [], 'ddof': 0, 'n_fit': int(len(X_fit)),
           'what': 'z = float32((z_admitted - mean_FIT) / sd_FIT): FIT moments (population SD) of the admitted z columns; a strictly '
                   'increasing affine map of the admitted z (which carries all-MODEL_FIT moments), applied identically to every role'}
    for j in ZCOLS:
        v = X_fit[:, j].astype(np.float64)
        m, s = float(v.mean()), float(v.std())
        if not (np.isfinite(m) and np.isfinite(s) and s > 0):
            raise ValueError(f'{names[j]}: FIT moments not usable')
        out['mean'].append(m); out['sd'].append(s)
    return out


def encode(X, names, dic, zmap):
    """Admitted X -> FIT dictionary (cts.trackf.recode) -> FIT z-map of AGEP / WKHP (float64 then float32, as the admitted schema)."""
    Xr = TF.recode(X, names, dic)
    for j, m, s in zip(ZCOLS, zmap['mean'], zmap['sd']):
        Xr[:, j] = ((Xr[:, j] - float(m)) / float(s)).astype(np.float32).astype(np.float64)
    return Xr


# ------------------------------------------------------------------ source pipeline (ohn.fam_b), keeping the fold models
def run_base_keep(spec, X, names, y, unit, outer, X_extra, seed, log=print):
    """ohn.fam_b.run_base line for line (same source functions, same order), except that it also RETURNS the 5 outer-fold model
    objects, which the source discarded after taking their logits. The selftest checks that its arrays and records equal
    ohn.fam_b.run_base's bitwise on invented data. Returns (main_obj, {k: fold_obj}, arrays, record)."""
    t0 = time.process_time()
    kind = spec['kind']
    arrays, rec, fits, fold_objs = {}, {'sets': {}}, 0, {}
    if kind == 'TREE':
        Xc = FB.compact(X, names); Xc_ex = {t: FB.compact(Xe, names) for t, Xe in X_extra.items()}; fz = None
    else:
        fz = FB.make_featurizer(X, names, spec['fixed']['spline_knots'])
        caps = sorted({g['capacity'] for g in spec['grid']})
        Phi = {cap: FB.add_design(X, names, fz, cap) for cap in caps}
        Phi_ex = {t: {cap: FB.add_design(Xe, names, fz, cap) for cap in caps} for t, Xe in X_extra.items()}
        rec['design_columns'] = {cap: int(Phi[cap].shape[1]) for cap in caps}
        rec['wkhp_values'] = fz['wkhp_values']
    main_obj = None
    for name, rows, fr in FB.outer_sets(unit, outer, seed):
        ts = time.process_time()
        if kind == 'TREE':
            best, table, nf = FB.tune_tree(spec, Xc, y, rows, fr, seed)
            model = FB.refit_tree(spec, best, Xc, y, rows, seed); nf += 1
            obj = {'kind': 'TREE', 'model': model, 'setting': best['setting'], 'n_iter': best['n']}
            z_rows = lambda idx, model=model: FB.center(model.decision_function(Xc[idx]))
            z_ex = lambda t, model=model: FB.center(model.decision_function(Xc_ex[t]))
            refit_info = {'n_iter': best['n']}
        else:
            best, table, nf = FB.tune_add(spec, Phi, y, rows, fr)
            model, it = FB.refit_add(spec, best, Phi, y, rows); nf += 1
            cap = best['setting']['capacity']
            obj = {'kind': 'ADD', 'model': model, 'setting': best['setting'], 'capacity': cap, 'featurizer': fz}
            z_rows = lambda idx, cap=cap, model=model: FB.center(model.decision_function(Phi[cap][idx]))
            z_ex = lambda t, cap=cap, model=model: FB.center(model.decision_function(Phi_ex[t][cap]))
            refit_info = {'lbfgs_iters': it, 'hit_max_iter': bool(it >= spec['fixed']['max_iter'])}
        fits += nf
        if name == 'main':
            main_obj = obj
            arrays['main__oof'] = best['oof']
        else:
            k = int(name[3:])
            held = np.where(outer == k)[0]
            arrays[f'{name}__inner_oof'] = best['oof']
            arrays[f'{name}__fold'] = z_rows(held)
            for t in X_extra:
                arrays[f'{name}__{t}'] = z_ex(t)
            fold_objs[k] = obj
        rec['sets'][name] = {'n_train': int(len(rows)), 'chosen': best['setting'], 'chosen_n_iter': best.get('n'),
                             'chosen_oof_nll': best['nll'], 'chosen_at_edge': bool(best.get('at_edge', False) or
                                                                                   (kind == 'TREE' and best['n'] == spec['fixed']['max_iter'])),
                             'table': table, 'refit': refit_info, 'fits': nf, 'cpu_s': round(time.process_time() - ts, 2)}
        log(json.dumps({'learner': kind, 'set': name, 'chosen': best['setting'], 'n': best.get('n'), 'oof_nll': round(best['nll'], 6),
                        'fits': nf, 'cpu_s': round(time.process_time() - ts, 1)}, default=float))
    rec['n_model_fits_main'] = rec['sets']['main']['fits']
    rec['n_crossfit_fits'] = fits - rec['sets']['main']['fits']
    rec['cpu_s'] = time.process_time() - t0
    return main_obj, fold_objs, arrays, rec


def fit_core(Xe_fit, names, y, unit, outer, extras, seed, learners, log=print):
    """Array-level B2_SINGLE pipeline on the encoded FIT rows. extras = {'mf': encoded MODEL_FIT (all rows), 'ms': encoded
    MODEL_SELECT}: label-free sets on which every fold pipeline's release is returned."""
    cfg = FB.CONFIGS[RECIPE]
    main, fold_objs, arrays, brec = {}, {}, {}, {}
    for lid in cfg['bases']:
        main[lid], fold_objs[lid], arrays[lid], brec[lid] = run_base_keep(learners[lid], Xe_fit, names, y, unit, outer, extras, seed, log)
    params, info = FB.fit_layer(cfg, {i: arrays[i]['main__oof'] for i in cfg['bases']}, y)
    n = len(y); oof = np.zeros((n, K_OCC)); fold_p = {t: [] for t in extras}; per_fold = {}
    for k in range(int(outer.max()) + 1):
        rows = np.where(outer != k)[0]; held = np.where(outer == k)[0]
        pk, ik = FB.fit_layer(cfg, {i: arrays[i][f'out{k}__inner_oof'] for i in cfg['bases']}, y[rows])
        oof[held] = FB.probs(FB.apply_layer(pk, {i: arrays[i][f'out{k}__fold'] for i in cfg['bases']}))
        for t in extras:
            fold_p[t].append(FB.probs(FB.apply_layer(pk, {i: arrays[i][f'out{k}__{t}'] for i in cfg['bases']})))
        per_fold[k] = {'params': pk, 'info': ik}
    fold_p = {t: np.stack(v) for t, v in fold_p.items()}
    oof_ref, ms_ref, _ = FB.crossfit_release(cfg, arrays, y, outer)      # the source's own routine on the same arrays
    checks = {'oof_equals_source_crossfit_release': bool(np.array_equal(oof, oof_ref)),
              'ms_foldavg_equals_source_crossfit_release': bool(np.array_equal(np.mean(fold_p['ms'], axis=0), ms_ref))}
    if not all(checks.values()):
        raise AssertionError(f'cross-fitting differs from ohn.fam_b.crossfit_release: {checks}')
    return {'main': main, 'fold_objs': fold_objs, 'arrays': arrays, 'brec': brec, 'params': params, 'info': info, 'oof': oof,
            'fold_p': fold_p, 'per_fold': per_fold, 'checks': checks}


# ------------------------------------------------------------------ deployment (from admitted features alone)
def predict_model(model, X, names, dic):
    """Occupation release of a saved B2_SINGLE pipeline: FIT dictionary, FIT z-map, the chosen base learner's centered scores
    (ohn.fam_b.base_logits_from) and the layer (ohn.fam_b.apply_layer, ohn.fam_b.probs)."""
    names = tuple(map(str, names))
    Xe = encode(X, names, dic, model['zmap'])
    zs = {lid: FB.base_logits_from(model['objs'][lid], Xe, names) for lid in model['params']['bases']}
    return FB.probs(FB.apply_layer(model['params'], zs))


def _write_pickle(path, obj):
    with open(path, 'xb') as f:
        pickle.dump(obj, f, protocol=5)
    return C.sha(path)


def _load_pickle(path, sha):
    if C.sha(path) != sha:
        raise PermissionError(f'{Path(path).name} differs from its record')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _write_json_x(path, obj):
    """Exclusive create, byte format of cts.common.write_json (so equal dictionaries give equal files)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n')
    return C.sha(path)


def _deployed(root, model_sha, dict_sha):
    root = Path(root)
    if C.sha(root / 'DICT.json') != dict_sha:
        raise PermissionError('DICT.json differs from the witness record')
    dic = json.loads((root / 'DICT.json').read_text())
    model = _load_pickle(root / 'WITNESS' / 'MODEL.pkl', model_sha)
    if model.get('schema') != MODEL_SCHEMA or model.get('recipe') != RECIPE:
        raise PermissionError('MODEL.pkl is not a B2_SINGLE sensitivity witness')
    return model, dic


def predict_b2(P, X, names):
    """Deployed sensitivity-witness probabilities from admitted features alone (bitwise equal to WITNESS.npz dep__<ROLE>)."""
    root = TF.pdir(P)
    rec = json.loads((root / 'WITNESS' / 'WITNESS.json').read_text())
    if rec.get('recipe') != RECIPE:
        raise PermissionError(f'{P}: witness recipe is {rec.get("recipe")}, not {RECIPE}')
    model, dic = _deployed(root, rec['files_sha256']['MODEL.pkl'], rec['dictionary']['dict_sha256'])
    return predict_model(model, X, names, dic)


# ------------------------------------------------------------------ real fit (python3; through cts.trackf witness --partition D0B)
def _versions():
    import scipy
    import sklearn
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}


def _base_crosscheck(P, dic, outer, row_id):
    """D0B vs its base partition's main witness, BEFORE any fit (label-free fields only: the FIT row ids, the outer folds and the
    dictionary must be the base witness's, when it exists). A mismatch stops the unit."""
    base = base_partition(P)
    if base == P:
        return None
    broot = TF.pdir(base)
    out = {'base_partition': base}
    if (broot / 'DICT.json').is_file():
        out['dictionary_equal_to_base'] = json.loads((broot / 'DICT.json').read_text()) == dic
    if (broot / 'WITNESS' / 'OOF.npz').is_file():
        with np.load(broot / 'WITNESS' / 'OOF.npz') as z:
            out['fit_rows_equal_to_base_witness'] = bool(np.array_equal(z['row_id'], row_id))
            out['outer_folds_equal_to_base_witness'] = bool(out['fit_rows_equal_to_base_witness'] and np.array_equal(z['fold'], outer))
    if not all(v for k, v in out.items() if k != 'base_partition'):
        raise AssertionError(f'{P} does not share {base}\'s FIT rows / folds / dictionary: {out}')
    return out


def fit_witness_b2(P, learners=None, log=None):
    """Fit the B2_SINGLE sensitivity witness on partition P's FIT rows and write the main witness's file layout under
    cts.trackf.pdir(P). `learners` is for the invented-data selftest only (None = the source registry, always on real data)."""
    t0 = time.process_time(); w0 = time.time()
    log = log or (lambda m: print(m, flush=True))
    if C.PARTITIONS[P].get('witness') != RECIPE:
        raise ValueError(f'partition {P} does not use the {RECIPE} witness')
    cfg = FB.CONFIGS[RECIPE]
    registry = {lid: FB.LEARNERS[lid] for lid in cfg['bases']}
    learners = registry if learners is None else learners
    root = TF.pdir(P); d = root / 'WITNESS'
    if (d / 'WITNESS.json').exists() or (root / 'DICT.json').exists() or (d / 'MODEL.pkl').exists():
        raise PermissionError(f'{P} witness exists (append-only)')
    s = C.PARTITIONS[P]['seed']
    F = C.load('MODEL_FIT', ('features', 'occupation'), 'fit', f'Track F {P}: bounded sensitivity witness (source B2_SINGLE recipe: '
               'TREE_V2 + ADD_V2 inner CV, refit, learner choice and temperature, nested household cross-fitting) on FIT; no SEX, no income',
               partition=P, prole='FIT')
    names = tuple(map(str, F['feature_names'])); y = np.asarray(F['occupation'], dtype=np.int64); unit = np.asarray(F['unit'])
    fit_rows = np.asarray(F['rows'], dtype=bool)
    role_feats = {r: C.features(r) for r in ROLES}                                       # label-free
    for r in ROLES:
        if tuple(map(str, role_feats[r]['feature_names'])) != names:
            raise ValueError(f'{r}: feature names differ from MODEL_FIT')
    mf = role_feats['MODEL_FIT']
    if (fit_rows.shape != (len(mf['row_id']),) or not np.array_equal(mf['row_id'][fit_rows], F['row_id'])
            or not np.array_equal(np.asarray(mf['unit'])[fit_rows], unit) or not np.array_equal(np.asarray(mf['X'])[fit_rows], np.asarray(F['X']))):
        raise AssertionError('FIT rows are not the masked MODEL_FIT rows')
    # ---- label-free preprocessing, FIT inputs only
    dic = TF.fit_dictionary(F['X'], names)
    zmap = fit_zmap(F['X'], names)
    Xe_fit = encode(F['X'], names, dic, zmap)
    extras = {'mf': encode(mf['X'], names, dic, zmap), 'ms': encode(role_feats['MODEL_SELECT']['X'], names, dic, zmap)}
    if not np.array_equal(extras['mf'][fit_rows], Xe_fit):
        raise AssertionError('encoding is not row-wise')
    outer = outer_folds(P, unit)
    xcheck = _base_crosscheck(P, dic, outer, np.asarray(F['row_id']))
    log(json.dumps({'partition': P, 'recipe': RECIPE, 'n_fit': int(len(y)), 'households': int(len(np.unique(unit))),
                    'outer_fold_sizes': np.bincount(outer).tolist(), 'n_collapsed': dic['n_collapsed'], 'base_crosscheck': xcheck}))
    # ---- the source pipeline on FIT
    core = fit_core(Xe_fit, names, y, unit, outer, extras, s, learners, log)
    params, per_fold = core['params'], core['per_fold']
    # ---- files (exclusive create)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    dict_sha = _write_json_x(root / 'DICT.json', dic)
    files = {}
    files['MODEL.pkl'] = _write_pickle(d / 'MODEL.pkl', {'schema': MODEL_SCHEMA, 'recipe': RECIPE, 'partition': P, 'pipeline': 'deployed',
                                                         'params': params, 'objs': core['main'], 'zmap': zmap})
    for k in sorted(per_fold):
        files[f'FOLD_{k}.pkl'] = _write_pickle(d / f'FOLD_{k}.pkl', {'schema': MODEL_SCHEMA, 'recipe': RECIPE, 'partition': P,
                                                                    'pipeline': f'outer fold {k}', 'params': per_fold[k]['params'],
                                                                    'objs': {lid: core['fold_objs'][lid][k] for lid in cfg['bases']},
                                                                    'zmap': zmap})
    dep_model, dic_saved = _deployed(root, files['MODEL.pkl'], dict_sha)                # the deployment path of predict_b2
    p_dep = {r: predict_model(dep_model, role_feats[r]['X'], names, dic_saved) for r in ROLES}
    checks = dict(core['checks'])
    checks['deployed_in_memory_equals_saved_MODEL_SELECT'] = bool(np.array_equal(
        predict_model({'params': params, 'objs': core['main'], 'zmap': zmap}, role_feats['MODEL_SELECT']['X'], names, dic),
        p_dep['MODEL_SELECT']))
    fold_ok = True
    for k in sorted(per_fold):
        fm = _load_pickle(d / f'FOLD_{k}.pkl', files[f'FOLD_{k}.pkl'])
        fold_ok &= bool(np.array_equal(predict_model(fm, role_feats['MODEL_SELECT']['X'], names, dic_saved), core['fold_p']['ms'][k]))
        fold_ok &= bool(np.array_equal(predict_model(fm, mf['X'], names, dic_saved), core['fold_p']['mf'][k]))
    checks['fold_pickles_reproduce_fold_stacks'] = fold_ok
    if not all(checks.values()):
        raise AssertionError(f'witness consistency checks failed: {checks}')
    files['WITNESS.npz'] = TF._save_npz(d / 'WITNESS.npz', **{f'dep__{r}': p_dep[r] for r in ROLES},
                                        fold__MODEL_FIT=core['fold_p']['mf'], fold__MODEL_SELECT=core['fold_p']['ms'])
    files['OOF.npz'] = TF._save_npz(d / 'OOF.npz', oof=core['oof'], fold=outer, fit_rows=fit_rows, row_id=np.asarray(F['row_id']))
    # ---- record
    brec = core['brec']
    own = C.task_vectors(y, core['oof'], K_OCC)
    fz = core['main']['ADD_V2']['featurizer']
    w_fit = Xe_fit[:, 1]
    _, wcnt = np.unique(w_fit, return_counts=True)
    share_thr = int(np.ceil(FB.WKHP_MIN_COUNT * len(y) / len(mf['row_id'])))
    featurizer = {'n_knots': fz['n_knots'], 'wkhp_min_count': FB.WKHP_MIN_COUNT, 'n_wkhp_values': len(fz['wkhp_values']),
                  'wkhp_values_fit_z': fz['wkhp_values'],
                  'agep_knot_range': [float(Xe_fit[:, 0].min()), float(Xe_fit[:, 0].max())],
                  'wkhp_knot_range': [float(w_fit.min()), float(w_fit.max())],
                  'design_columns': brec['ADD_V2'].get('design_columns'),
                  'not_used_share_preserving_alternative': {'min_count': share_thr, 'n_wkhp_values': int((wcnt >= share_thr).sum()),
                                                            'note': f'{FB.WKHP_MIN_COUNT} x n_FIT / n_MODEL_FIT, rounded up; information only'}}
    n_layer_cf = sum(v['info']['n_fits'] for v in per_fold.values())
    counts = {'n_model_fits': sum(brec[i]['n_model_fits_main'] for i in cfg['bases']), 'n_calibrator_fits': int(core['info']['n_fits']),
              'n_crossfit_fits': sum(brec[i]['n_crossfit_fits'] for i in cfg['bases']) + n_layer_cf}
    ch = params['bases'][0]
    adaptations = [
        f'training set: FIT rows of {P} ({len(y)} people, {len(np.unique(unit))} households) instead of all MODEL_FIT',
        f'outer cross-fitting folds: household_folds(unit_FIT, 5, CTS1_cf_{base_partition(P)}) (the main witness folds) instead of '
        'ohn.common.crossfit_folds (OHN1_crossfit over MODEL_FIT)',
        f'inner folds: ohn.fam_b.outer_sets unchanged (tags OHN_B_inner_S{s} and OHN_B_inner_S{s}_out<k>), computed over the FIT '
        'households, so the assignments differ from the source',
        f'dictionary: FIT-only re-collapse (cts.trackf.fit_dictionary, min count {dic["min_count"]}); collapsed levels {dic["n_collapsed"]}; '
        f'collapsed SCHL levels {dic["collapsed"]["SCHL"]} are coded as SCHL value 3 by ohn.fam_b.compact (SCHL=__OTHER__); '
        'collapsed POBP levels fall into POB6 group 5 of the additive basis',
        'z-map: AGEP z / WKHP z re-standardized with FIT moments (float64 then float32), replacing the all-MODEL_FIT moments of the '
        'admitted z; strictly increasing affine (TREE invariant up to bin-edge rounding; ADD is not invariant: L2 penalty, linear '
        'AGEP / WKHP columns and products)',
        f'featurizer: ohn.fam_b.make_featurizer on FIT inputs (spline knots over the FIT ranges); the recipe\'s absolute WKHP '
        f'threshold {FB.WKHP_MIN_COUNT} people kept, giving {len(fz["wkhp_values"])} indicator values on FIT (13 in the source on '
        f'20,000 people; a share-preserving threshold of {share_thr} would give {int((wcnt >= share_thr).sum())}, not used)',
        'label-free preprocessing (dictionary, z-map, featurizer) fitted once on all FIT and shared by the 5 fold pipelines, as the '
        'source fitted its label-free steps once on all MODEL_FIT (the held-out fold inputs inform them; no labels)',
        'no other recipe constant is rescaled (TREE_V2 / ADD_V2 grids, min leaf 50 / 200, lr, max_iter, C grid, knots, temperature '
        f'bounds); HGB random_state ohn.fam_b.hgb_seed({s}) (the associated source seed)',
        'fold models kept: cts.sensitivity.run_base_keep mirrors ohn.fam_b.run_base and also returns the fold model objects '
        '(the source discarded them); its arrays equal ohn.fam_b.run_base (selftest, invented data)',
        'fold pipelines evaluated on all MODEL_FIT rows and on MODEL_SELECT (stacks kept, as the main witness); the source\'s '
        'CALIBRATION extra (unused by B2_SINGLE) and its cross-fitted diagnostic layers (not released) are not computed',
    ]
    rec = {'schema': SCHEMA, 'partition': P, 'seed': s, 'recipe': RECIPE, 'kind': 'bounded sensitivity witness',
           'config': cfg, 'learners': learners, 'learners_are_source_registry': learners == registry,
           'T_star': brec['TREE_V2']['sets']['main']['chosen_n_iter'],
           'T_k': {str(k): brec['TREE_V2']['sets'][f'out{k}']['chosen_n_iter'] for k in sorted(per_fold)},
           'T_star_note': 'TREE_V2 tree count (main set / outer sets); a release uses it only when TREE_V2 is its chosen learner',
           'deployed_layer': params, 'deployed_layer_info': core['info'], 'chosen_learner': ch, 'tau': params['tau'],
           'fold_layers': {str(k): per_fold[k] for k in sorted(per_fold)},
           'fold_chosen_learners': {str(k): per_fold[k]['params']['chosen'] for k in sorted(per_fold)},
           'base_learners': {i: {k: v for k, v in brec[i].items()} for i in cfg['bases']},
           'n_fit_people': int(len(y)), 'n_fit_households': int(len(np.unique(unit))),
           'dictionary': {'min_count': dic['min_count'], 'n_collapsed': dic['n_collapsed'], 'dict_sha256': dict_sha},
           'zmap': zmap, 'featurizer': featurizer,
           'outer_fold_tag': f'CTS1_cf_{base_partition(P)}', 'outer_fold_sizes': np.bincount(outer).tolist(),
           'inner_fold_tag': f'OHN_B_inner_S{s} (main) / OHN_B_inner_S{s}_out<k> (outer set k): ohn.fam_b.outer_sets over FIT households',
           'hgb_random_state': FB.hgb_seed(s), 'base_partition_crosscheck': xcheck,
           'oof_fit_summary': {m: float(v.mean()) for m, v in own.items()}, 'checks': checks,
           'fit_counts': counts, 'release_specifications': 1, 'adaptations': adaptations,
           'files_sha256': files, 'release_sha256': {r: C.arr_sha(p_dep[r]) for r in ROLES},
           'labels_used': 'FIT occupation only (no SEX, no income)',
           'exposure': 'deployed and fold pipelines fitted on FIT rows only (labels and label-free preprocessing); GUARD, CHECK, the '
                       'repair panels and every other role only receive predictions',
           'interpreter': sys.executable, 'versions': _versions(), 'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0,
           'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_json_x(d / 'WITNESS.json', rec)
    C.unit_stats(**counts)
    log(json.dumps({k: rec[k] for k in ('partition', 'recipe', 'chosen_learner', 'tau', 'T_star', 'fold_chosen_learners',
                                        'oof_fit_summary', 'checks', 'base_partition_crosscheck', 'cpu_s')}, default=float))
    return rec


# ------------------------------------------------------------------ selftest (invented data only)
def _inv_names():
    return (('AGEP', 'WKHP') + tuple(f'COW={i}' for i in (1, 2, 3)) + ('COW=__OTHER__',)
            + tuple(f'SCHL={i:02d}' for i in (4, 16, 21, 22)) + ('SCHL=__OTHER__',)
            + tuple(f'MAR={i}' for i in (1, 3, 5)) + ('MAR=__OTHER__',)
            + tuple(f'POBP={i:03d}' for i in (12, 36, 110, 303)) + ('POBP=__OTHER__',))


def _inv_block(X, names, rng, rows, b, levels, p):
    idx = [names.index(f'{b}={lv}') for lv in levels]
    X[rows, np.asarray(idx)[rng.choice(len(idx), size=len(rows), p=p)]] = 1.0


def _inv_rows(X, names, rng, rows, gc_variant):
    """Invented person features. gc_variant None: FIT-like rows; 'A' / 'B': two different GUARD / CHECK laws, each with values
    outside the FIT ranges, a WKHP value frequent only there and levels absent or rare in FIT."""
    n = len(rows)
    if gc_variant is None:
        X[rows, 0] = rng.choice(np.linspace(-1.8, 1.65, 40), size=n)
        X[rows, 1] = np.where(rng.random(n) < 0.45, -0.0569, rng.choice(np.linspace(-3.5, 2.2, 30), size=n))
        _inv_block(X, names, rng, rows, 'COW', ('1', '2', '3', '__OTHER__'), [0.5, 0.25, 0.2, 0.05])
        _inv_block(X, names, rng, rows, 'SCHL', ('04', '16', '21', '22', '__OTHER__'), [0.003, 0.45, 0.35, 0.177, 0.02])
        _inv_block(X, names, rng, rows, 'MAR', ('1', '3', '5', '__OTHER__'), [0.55, 0.004, 0.4, 0.046])
        _inv_block(X, names, rng, rows, 'POBP', ('012', '036', '110', '__OTHER__'), [0.6, 0.2, 0.15, 0.05])
    else:
        hi = (3.0, 4.0) if gc_variant == 'A' else (2.6, 5.5)
        wv = 2.5 if gc_variant == 'A' else 1.7
        X[rows, 0] = np.where(rng.random(n) < 0.3, rng.choice(hi, size=n), rng.choice(np.linspace(-1.8, 1.65, 40), size=n))
        X[rows, 1] = np.where(rng.random(n) < 0.6, wv, rng.choice(np.linspace(-3.5, 6.0, 30), size=n))
        _inv_block(X, names, rng, rows, 'COW', ('1', '2', '3', '__OTHER__'), [0.25, 0.25, 0.25, 0.25])
        _inv_block(X, names, rng, rows, 'SCHL', ('04', '16', '21', '22', '__OTHER__'), [0.2, 0.2, 0.2, 0.2, 0.2])
        _inv_block(X, names, rng, rows, 'MAR', ('1', '3', '5', '__OTHER__'), [0.3, 0.3, 0.3, 0.1])
        _inv_block(X, names, rng, rows, 'POBP', ('012', '036', '110', '303', '__OTHER__'), [0.2, 0.2, 0.2, 0.3, 0.1])


def _inv_labels(X, names, rng):
    edu = (X[:, names.index('SCHL=21')] + X[:, names.index('SCHL=22')]) > 0
    z = np.column_stack([1.2 * edu + 0.4 * X[:, 0], 0.8 * (X[:, 1] < -0.5) - 0.3 * edu, 0.6 * X[:, names.index('COW=2')] + 0.3 * X[:, 0] * X[:, 1],
                         0.7 * (~edu) - 0.4 * X[:, names.index('MAR=5')], 0.5 * X[:, names.index('POBP=036')] + 0.2 * X[:, 1]])
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return (rng.random(len(X))[:, None] > p.cumsum(1)).sum(1).clip(0, K_OCC - 1).astype(np.int64)


def _inv_world(seed_data, gc_variant, perturb_fold0=False):
    """Invented MODEL_FIT population (FIT / GUARD / CHECK by the D0 household parts) and the five other roles."""
    names = _inv_names()
    rng = np.random.default_rng(20261018)
    n = 1600
    unit = np.repeat(np.arange(n // 2), 2) + 5000
    row_id = np.arange(n) + 100000
    parts = C.household_folds(unit, 10, C.PARTITIONS['D0B']['tag'])
    fit = parts < 6
    X = np.zeros((n, len(names)), dtype=np.float32)
    _inv_rows(X, names, rng, np.where(fit)[0], None)                       # FIT rows: identical in every world
    y = _inv_labels(X, names, rng)
    rng_gc = np.random.default_rng(seed_data)
    _inv_rows(X, names, rng_gc, np.where(~fit)[0], gc_variant)            # GUARD / CHECK rows: the world's own law
    y[~fit] = rng_gc.integers(0, K_OCC, int((~fit).sum()))
    if perturb_fold0:                                                       # labels of outer fold 0 of FIT changed
        f0 = np.where(fit)[0][outer_folds('D0B', unit[fit]) == 0]
        y[f0] = (y[f0] + 1 + rng.integers(0, K_OCC - 1, len(f0))) % K_OCC
    roles = {'MODEL_FIT': {'X': X, 'feature_names': names, 'row_id': row_id, 'unit': unit}}
    rng_r = np.random.default_rng(7)
    for i, (r, m) in enumerate((('CALIBRATION', 150), ('MODEL_SELECT', 300), ('ATTACK_FIT', 120), ('ATTACK_VALIDATION', 90),
                                ('DEV_REPORT', 110))):
        Xr = np.zeros((m, len(names)), dtype=np.float32)
        _inv_rows(Xr, names, rng_r, np.arange(m), None if i % 2 == 0 else 'A')
        roles[r] = {'X': Xr, 'feature_names': names, 'row_id': np.arange(m) + 200000 * (i + 2), 'unit': np.arange(m) + 900000}
    F = {'X': X[fit].copy(), 'feature_names': names, 'row_id': row_id[fit], 'unit': unit[fit], 'occupation': y[fit], 'rows': fit}
    return F, roles, fit


TINY = {'TREE_V2': {'kind': 'TREE', 'fixed': {'lr': 0.1, 'max_iter': 30, 'l2': 1.0, 'max_bins': 255, 'checkpoint': 10, 'cat': 'P_ORD'},
                    'grid': ({'leaves': 4, 'msl': 20}, {'leaves': 8, 'msl': 40})},
        'ADD_V2': {'kind': 'ADD', 'fixed': {'tol': 1e-6, 'max_iter': 2000, 'spline_knots': 6},
                   'grid': ({'capacity': 'S', 'C': 0.01}, {'capacity': 'L', 'C': 0.03})}}


def selftest():
    import inspect
    import tempfile
    t0 = time.process_time()
    checks = {}
    orig = (C.load, C.features, TF.pdir)
    calls = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def run_world(tag, F, roles):
                def fake_load(role, fields, operation, purpose, partition=None, prole=None):
                    calls.append({'role': role, 'fields': tuple(fields), 'operation': operation, 'partition': partition, 'prole': prole})
                    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in F.items()}
                C.load = fake_load
                C.features = lambda r, raw=False: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in roles[r].items()}
                TF.pdir = lambda P: tmp / tag / P
                rec = fit_witness_b2('D0B', learners=TINY, log=lambda m: None)
                return rec, TF.load_witness('D0B')

            FA_, RA, fitA = _inv_world(11, 'A')
            FB_, RB, fitB = _inv_world(29, 'B')
            FC, RC, _ = _inv_world(11, 'A', perturb_fold0=True)
            names = FA_['feature_names']

            def make_base(tag, F, permute=False):               # a stand-in for the D0 main witness (label-free fields only)
                b = tmp / tag / 'D0'; (b / 'WITNESS').mkdir(parents=True)
                (b / 'DICT.json').write_text(json.dumps(TF.fit_dictionary(F['X'], names), indent=1, sort_keys=True) + '\n')
                f = outer_folds('D0B', F['unit'])
                np.savez(b / 'WITNESS' / 'OOF.npz', row_id=F['row_id'], fold=(f + 1) % N_OUTER if permute else f)
            make_base('A', FA_); make_base('B', FB_); make_base('N', FA_, permute=True)
            try:
                run_world('N', FA_, RA); checks['base_crosscheck_rejects_other_folds'] = False
            except AssertionError:
                checks['base_crosscheck_rejects_other_folds'] = not (tmp / 'N' / 'D0B').exists()
            calls.clear()
            recA, (wA, WA, OA) = run_world('A', FA_, RA)
            checks['base_crosscheck_accepts_same_folds'] = wA['base_partition_crosscheck'] == {
                'base_partition': 'D0', 'dictionary_equal_to_base': True, 'fit_rows_equal_to_base_witness': True,
                'outer_folds_equal_to_base_witness': True}
            # --- access: one logged FIT read, occupation only
            checks['access_one_fit_read_occupation_only'] = bool(len(calls) == 1 and calls[0] == {
                'role': 'MODEL_FIT', 'fields': ('features', 'occupation'), 'operation': 'fit', 'partition': 'D0B', 'prole': 'FIT'})
            # --- layout (cts.trackf.load_witness and the arrays its readers use)
            nmf, nms, nfit = len(RA['MODEL_FIT']['row_id']), len(RA['MODEL_SELECT']['row_id']), int(fitA.sum())
            want_W = {f'dep__{r}': (len(RA[r]['row_id']), K_OCC) for r in ROLES}
            want_W.update({'fold__MODEL_FIT': (N_OUTER, nmf, K_OCC), 'fold__MODEL_SELECT': (N_OUTER, nms, K_OCC)})
            checks['layout_witness_npz'] = bool(set(WA) == set(want_W) and all(WA[k].shape == v and WA[k].dtype == np.float64
                                                                              and np.allclose(WA[k].sum(-1), 1, atol=1e-12) for k, v in want_W.items()))
            checks['layout_oof_npz'] = bool(set(OA) == {'oof', 'fold', 'fit_rows', 'row_id'} and OA['oof'].shape == (nfit, K_OCC)
                                            and OA['fold'].shape == (nfit,) and OA['fold'].dtype == np.int64 and OA['fit_rows'].dtype == bool
                                            and OA['fit_rows'].shape == (nmf,) and np.array_equal(OA['fit_rows'], fitA)
                                            and np.array_equal(OA['row_id'], RA['MODEL_FIT']['row_id'][fitA]))
            checks['layout_record'] = bool(wA['recipe'] == RECIPE and wA['schema'] == SCHEMA and 'T_star' in wA and
                                           set(wA['files_sha256']) == {'MODEL.pkl', 'WITNESS.npz', 'OOF.npz'} | {f'FOLD_{k}.pkl' for k in range(N_OUTER)}
                                           and (TF.pdir('D0B') / 'DICT.json').is_file() and not wA['learners_are_source_registry'])
            rows = fitA.copy(); rows[:7] = True
            dl = np.random.default_rng(3).normal(size=(nmf, K_OCC)); dlf = np.stack([dl] * N_OUTER)
            r0, e0 = TF.release_at(WA['dep__MODEL_FIT'][rows], dl[rows], 0.0), TF.ensemble_at(WA['fold__MODEL_FIT'][:, rows], dlf[:, rows], 0.0)
            checks['trackf_readers_accept_arrays'] = bool(np.allclose(r0, WA['dep__MODEL_FIT'][rows], atol=1e-12)
                                                          and np.allclose(e0, WA['fold__MODEL_FIT'][:, rows].mean(0), atol=1e-12)
                                                          and np.isfinite(TF.release_at(WA['dep__MODEL_SELECT'], dl[:nms], 0.7)).all())
            # --- deployment parity through the cts.trackf dispatch (fresh load from the files for every role)
            checks['predict_parity_all_roles_bitwise'] = bool(all(np.array_equal(TF.witness_predict('D0B', RA[r]['X'], names), WA[f'dep__{r}'])
                                                                  for r in ROLES))
            checks['fit_consistency_checks'] = bool(all(wA['checks'].values()) and len(wA['checks']) == 4)
            # --- FIT-only preprocessing (direct)
            dic = json.loads((TF.pdir('D0B') / 'DICT.json').read_text())
            checks['fitonly_dictionary'] = bool(dic == TF.fit_dictionary(FA_['X'], names) and 'POBP=303' in dic['collapsed']['POBP']
                                                and 'MAR=3' in dic['collapsed']['MAR'] and 'SCHL=04' in dic['collapsed']['SCHL']
                                                and dic != TF.fit_dictionary(RA['MODEL_FIT']['X'], names))
            zA = wA['zmap']
            xf = FA_['X'].astype(np.float64); xall = RA['MODEL_FIT']['X'].astype(np.float64)
            checks['fitonly_zmap'] = bool(zA['mean'] == [float(xf[:, 0].mean()), float(xf[:, 1].mean())]
                                          and zA['sd'] == [float(xf[:, 0].std()), float(xf[:, 1].std())]
                                          and abs(zA['mean'][0] - xall[:, 0].mean()) > 1e-3)
            mA = _load_pickle(TF.pdir('D0B') / 'WITNESS' / 'MODEL.pkl', wA['files_sha256']['MODEL.pkl'])
            fz = mA['objs']['ADD_V2']['featurizer']
            Xe_fit = encode(FA_['X'], names, dic, zA); Xe_all = encode(RA['MODEL_FIT']['X'], names, dic, zA)
            tA = fz['agep_spline'].bsplines_[0].t; tW = fz['wkhp_spline'].bsplines_[0].t
            u, c = np.unique(Xe_fit[:, 1], return_counts=True)
            gc_w = float(Xe_all[~fitA][np.isclose(RA['MODEL_FIT']['X'][~fitA, 1], 2.5), 1][0])
            checks['fitonly_featurizer'] = bool(tA[3] == Xe_fit[:, 0].min() and tA[-4] == Xe_fit[:, 0].max()
                                                and tW[3] == Xe_fit[:, 1].min() and tW[-4] == Xe_fit[:, 1].max()
                                                and Xe_all[:, 0].max() > tA[-4] and Xe_all[:, 1].max() > tW[-4]
                                                and fz['wkhp_values'] == [float(v) for v in u[c >= FB.WKHP_MIN_COUNT]]
                                                and len(fz['wkhp_values']) == 1 and gc_w not in fz['wkhp_values']
                                                and int((Xe_all[:, 1] == gc_w).sum()) >= FB.WKHP_MIN_COUNT)
            # --- poisoning: a different GUARD / CHECK law (features and labels) changes nothing that was fitted
            recB, (wB, WB, OB) = run_world('B', FB_, RB)
            volatile = ('cpu_s', 'wall_s', 'utc', 'files_sha256', 'release_sha256', 'base_learners', 'interpreter')
            same_rec = all(wA[k] == wB[k] for k in wA if k not in volatile)
            nocpu = lambda sets: {s: {a: b for a, b in v.items() if a != 'cpu_s'} for s, v in sets.items()}
            same_bases = all(nocpu(wA['base_learners'][i]['sets']) == nocpu(wB['base_learners'][i]['sets']) for i in wA['base_learners'])
            checks['fitonly_guard_check_poisoning_invariance'] = bool(
                np.array_equal(fitA, fitB) and same_rec and same_bases and all(np.array_equal(OA[k], OB[k]) for k in OA)
                and all(np.array_equal(WA[f'dep__{r}'], WB[f'dep__{r}']) for r in ROLES if r != 'MODEL_FIT')
                and np.array_equal(WA['fold__MODEL_SELECT'], WB['fold__MODEL_SELECT'])
                and np.array_equal(WA['dep__MODEL_FIT'][fitA], WB['dep__MODEL_FIT'][fitA])
                and np.array_equal(WA['fold__MODEL_FIT'][:, fitA], WB['fold__MODEL_FIT'][:, fitA])
                and not np.array_equal(WA['dep__MODEL_FIT'][~fitA], WB['dep__MODEL_FIT'][~fitA])
                and (tmp / 'A' / 'D0B' / 'DICT.json').read_bytes() == (tmp / 'B' / 'D0B' / 'DICT.json').read_bytes())
            # --- nested cross-fitting: outer folds, household separation, per-fold layers, held-out labels never reach pipeline k
            folds = OA['fold']; uf = FA_['unit']
            checks['outer_folds_are_main_witness_folds'] = bool(np.array_equal(folds, C.household_folds(uf, N_OUTER, 'CTS1_cf_D0'))
                                                                and wA['outer_fold_tag'] == 'CTS1_cf_D0'
                                                                and all(len(np.unique(folds[uf == h])) == 1 for h in np.unique(uf)))
            sets = FB.outer_sets(uf, folds, 0)
            checks['nested_sets_household_separated'] = bool(len(sets) == N_OUTER + 1 and all(
                not np.isin(uf[r], uf[folds == int(nm[3:])]).any() and np.array_equal(r, np.where(folds != int(nm[3:]))[0]) for nm, r, _ in sets[1:]))
            checks['per_fold_tuning_choice_temperature'] = bool(
                len(wA['fold_layers']) == N_OUTER and all(set(v['params']) >= {'chosen', 'tau'} for v in wA['fold_layers'].values())
                and all(set(wA['base_learners'][i]['sets']) == {'main'} | {f'out{k}' for k in range(N_OUTER)} for i in wA['base_learners']))
            recC, (wC, WC, OC_) = run_world('C', FC, RC)
            f0 = folds == 0
            checks['heldout_labels_never_reach_fold_pipeline'] = bool(
                np.array_equal(WA['fold__MODEL_SELECT'][0], WC['fold__MODEL_SELECT'][0]) and np.array_equal(WA['fold__MODEL_FIT'][0], WC['fold__MODEL_FIT'][0])
                and np.array_equal(OA['oof'][f0], OC_['oof'][f0]) and wA['fold_layers']['0'] == wC['fold_layers']['0']
                and all(not np.array_equal(WA['fold__MODEL_SELECT'][k], WC['fold__MODEL_SELECT'][k]) for k in range(1, N_OUTER))
                and not np.array_equal(WA['dep__MODEL_SELECT'], WC['dep__MODEL_SELECT']))
            # --- run_base_keep is ohn.fam_b.run_base (same arrays and records) on the same inputs
            ext = {'mf': Xe_all, 'ms': encode(RA['MODEL_SELECT']['X'], names, dic, zA)}
            yF = FA_['occupation']
            mirror = True
            for lid, spec in TINY.items():
                _, arr_ref, rec_ref = FB.run_base(spec, Xe_fit, names, yF, uf, folds, ext, 0, log=lambda m: None)
                _, fobj, arr, rk = run_base_keep(spec, Xe_fit, names, yF, uf, folds, ext, 0, log=lambda m: None)
                mirror &= bool(set(arr) == set(arr_ref) and all(np.array_equal(arr[k], arr_ref[k]) for k in arr)
                               and all({a: b for a, b in rk['sets'][s].items() if a != 'cpu_s'} == {a: b for a, b in rec_ref['sets'][s].items() if a != 'cpu_s'}
                                       for s in rk['sets']) and sorted(fobj) == list(range(N_OUTER)))
            checks['run_base_keep_equals_source_run_base'] = mirror
            # --- append-only; dispatch from cts.trackf
            TF.pdir = lambda P: tmp / 'A' / P
            try:
                fit_witness_b2('D0B', learners=TINY, log=lambda m: None); checks['append_only'] = False
            except PermissionError:
                checks['append_only'] = True
            checks['trackf_dispatch'] = bool(C.PARTITIONS['D0B'].get('witness') == RECIPE and C.PARTITIONS['D0B'].get('same_rows_as') == 'D0'
                                             and 'sensitivity.fit_witness_b2' in inspect.getsource(TF.fit_witness)
                                             and 'sensitivity.predict_b2' in inspect.getsource(TF.witness_predict))
            checks['registry_is_source_recipe'] = bool(FB.CONFIGS[RECIPE]['layer'] == 'single' and FB.CONFIGS[RECIPE]['bases'] == ('TREE_V2', 'ADD_V2')
                                                       and FB.LEARNERS['TREE_V2']['fixed']['cat'] == 'P_ORD' and len(FB.LEARNERS['TREE_V2']['grid']) == 6
                                                       and len(FB.LEARNERS['ADD_V2']['grid']) == 6)
            info = {'chosen_learner_invented': wA['chosen_learner'], 'fold_chosen_invented': wA['fold_chosen_learners'],
                    'wkhp_values_invented': len(fz['wkhp_values']), 'n_fit_invented': nfit}
    finally:
        C.load, C.features, TF.pdir = orig
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'info': info if 'info' in locals() else None,
                      'cpu_s': round(time.process_time() - t0, 1)}, indent=1, default=float))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return selftest()
    return 2


if __name__ == '__main__':
    sys.exit(main())
