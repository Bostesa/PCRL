"""Conditional income branch: POBP-ordinal repair of the source's STRONG income learner. PCRL overnight occupation-headroom
campaign (EXPLORATORY DEVELOPMENT on previously used records; not confirmation; DEV-guided: opened through leg B of the frozen
income-branch trigger, a published DEV_REPORT result of the source campaign; no new DEV_REPORT read).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed). The service predicts an income class (2) and an occupation group (5) from six permitted raw fields (AGEP, WKHP,
COW, SCHL, MAR, POBP; admitted 145-column encoding X). Each recipient receives only its own float64 probability vector and argmax
decision. SEX is the dataset's recorded demographic column; readers are separate evaluation classifiers.

Mechanism (results/pcrl_occupation_headroom_overnight_v1/roles/investigator_inc/CARD.md):
  Step 1   the source's Step-1 income learner (ihs.fam_h.step1: 7-setting HGB grid, household 5-fold CV on MODEL_FIT with the
           source folds 'IHS_H_step1_cv_S<seed>', first minimum of the pooled out-of-fold summed Brier, refit) with ONE change: the
           compact settings code POBP as its ordinal place-of-birth code (ohn.fam_a.compact_po) instead of a native category.
           The original-coding learner is recomputed in the same unit (reproduction check against the source STEP1.json and
           HGB.pkl; learner control in the household cross-fitting check).
  INC_TW_PO   matched task-only twin: CALIBRATION stack of the repaired HGB and T_TREE income log-odds + Platt map (= the source's
              HC_STRONG with the repaired base).
  INC_H3_PO   branch configuration: the source's H3 step unchanged on the repaired base, d = a*(s_B + D_L + beta*D_J) + b, beta by
              H3's registered precision-aware CALIBRATION budget rule (ihs.fam_h.budget_select with CONFIGS['H3_STRONG_LJDIR_PB']).
  occupation  the top selected task-only witness (WITNESS below), recomputed by its family's own predict and checked bitwise.
Labels are loaded only through ohn.common.load_role (MODEL_FIT income for Step 1; CALIBRATION income for the stack, budget and
Platt map). The ihs label loaders are never called; ihs.fam_h is imported for its label-free numerical functions only.

Commands: selftest | step1 --seed s | fit --seed s --config ID | verify --seed s --config ID.
`predict(seed, config_id, X, names)` -> {'income': (n, 2), 'occupation': (n, 5)} from features alone.
"""
from __future__ import annotations
import argparse
import importlib
import io
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

from . import common as C
from . import fam_a as A
from ihs import fam_h as FH

FAMILY = 'INC'
ROLES = C.ROLES6
WITNESS = ('B', 'B2_SINGLE')   # top SELECTED witness (WITNESS_SELECTION.json, commit c292353dc); set 06:28Z before the first fit
PO_CATEGORICAL = np.array([False, False, True, False, True, False])     # compact_po: COW, MAR native; POBP ordinal code
STEP1_DIR = C.RUNS / FAMILY / 'STEP1'
SRC_CV_TAG = 'IHS_H_step1_cv_S{seed}'
SRC_CAL_TAG = 'IHS_H_calcv_S{seed}'
SRC_BUDGET_TAG = 'IHS_H_budget_S{seed}'

CONFIGS = {
    'INC_TW_PO': {'source_config': 'HC_STRONG', 'kind': 'control', 'budget_class': 'INC8',
                  'what': 'matched task-only twin: CALIBRATION stack of the POBP-ordinal STRONG HGB income learner and T_TREE income '
                          'log-odds + Platt map (the source HC_STRONG with the repaired base), beside the selected witness occupation'},
    'INC_H3_PO': {'source_config': 'H3_STRONG_LJDIR_PB', 'kind': 'income_branch', 'budget_class': 'INC8',
                  'what': 'branch configuration: the source H3 step unchanged on the repaired base (stack + D_L + beta*D_J, beta by the '
                          'registered precision-aware CALIBRATION budget rule, Platt map), beside the selected witness occupation'},
}


# ---------------------------------------------------------------- grids, representations, learner
def grids():
    """(source grid, repaired grid): identical except that the compact settings use the POBP-ordinal coding."""
    orig = [dict(p) for p in FH.HGB_GRID]
    rep = [{**p, 'rep': 'cmp_po'} if p['rep'] == 'cmp' else dict(p) for p in FH.HGB_GRID]
    return orig, rep


def key(p):
    return f"{p['rep']}|{p['leaves']}|{p['msl']}"


def union_grid(orig, rep):
    out, seen = [], set()
    for p in orig + rep:
        if key(p) not in seen:
            out.append(p); seen.add(key(p))
    return out


def rep_matrix(rep, X, names):
    if rep == 'cmp':
        return FH.compact(X, names)
    if rep == 'cmp_po':
        return A.compact_po(X, names)
    if rep == 'ohe':
        return np.asarray(X, dtype=np.float64)
    raise KeyError(rep)


def make_hgb(params, seed):
    if params['rep'] in ('cmp', 'ohe'):
        return FH.make_hgb(params, seed)                      # the source's own constructor (reproduction)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=params['lr'], max_iter=int(params['max_iter']),
                                          max_leaf_nodes=int(params['leaves']), min_samples_leaf=int(params['msl']),
                                          l2_regularization=float(params['l2']), max_bins=255, early_stopping=False,
                                          categorical_features=PO_CATEGORICAL, random_state=int(seed))


def fit_hgb(params, Xr, y, seed):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return make_hgb(params, seed).fit(Xr, y)


def cv_curves(Xr, y, folds, params, seed):
    """Household K-fold out-of-fold curves (mean summed Brier, NLL, accuracy at every tree count); same arithmetic as
    ihs.fam_h.cv_curves (checked equal in the selftest), with the learner dispatched by representation."""
    T = int(params['max_iter']); n = len(y)
    acc = {'brier': np.zeros(T), 'nll': np.zeros(T), 'accuracy': np.zeros(T)}
    fits = 0
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = fit_hgb(params, Xr[tr], y[tr], seed)
        fits += 1
        for i, d in enumerate(m.staged_decision_function(Xr[te])):
            v = FH.person_metrics(y[te], d)
            for kk in acc:
                acc[kk][i] += v[kk].sum()
    return {kk: v / n for kk, v in acc.items()}, fits


def run_grid(reps, y, folds, grid, seed, rows=None):
    """CV curves for every setting on the given rows; entries as the source's STEP1.json grid records."""
    rows = np.ones(len(y), dtype=bool) if rows is None else rows
    out, fits = {}, 0
    for p in grid:
        curves, nf = cv_curves(reps[p['rep']][rows], y[rows], folds[rows], p, seed); fits += nf
        i = int(np.argmin(curves['brier']))
        out[key(p)] = {'params': p, 'best_iter': i + 1, 'oof_brier': float(curves['brier'][i]), 'oof_nll': float(curves['nll'][i]),
                       'oof_accuracy': float(curves['accuracy'][i]), 'oof_nll_min': float(curves['nll'].min()),
                       'oof_brier_at_600': float(curves['brier'][-1]), 'curve_brier_every_50': [float(c) for c in curves['brier'][49::50]]}
    return out, fits


def select(entries, grid):
    """The source's rule: min over the grid (first in grid order on ties) of the out-of-fold summed Brier at its best tree count."""
    best = min((entries[key(p)] for p in grid), key=lambda g: g['oof_brier'])
    return {**best['params'], 'max_iter': best['best_iter']}, best


def step1_core(reps, y, src_folds, cf_folds, orig, rep, seed, log=None):
    """Main CV and refits for both codings, plus the household cross-fitting check (OHN1 folds) of the entire Step-1 procedure
    for both codings. Returns entries, selections, refitted models and out-of-fold log-odds."""
    t0 = time.process_time()
    grid = union_grid(orig, rep)
    main, fits = run_grid(reps, y, src_folds, grid, seed)
    sel_o, _ = select(main, orig); sel_r, _ = select(main, rep)
    hgb_o = fit_hgb(sel_o, reps[sel_o['rep']], y, seed); hgb_r = fit_hgb(sel_r, reps[sel_r['rep']], y, seed); fits += 2
    if log:
        log(json.dumps({'main_cv_done': True, 'selected_orig': sel_o, 'selected_rep': sel_r, 'cpu_s': round(time.process_time() - t0, 1)}))
    oof = {'orig': np.zeros(len(y)), 'rep': np.zeros(len(y))}; per_fold = {}; cf_fits = 0
    for k in np.unique(cf_folds):
        rows = cf_folds != k; te = cf_folds == k
        ent, nf = run_grid(reps, y, src_folds, grid, seed, rows); cf_fits += nf
        so, _ = select(ent, orig); sr, _ = select(ent, rep)
        mo = fit_hgb(so, reps[so['rep']][rows], y[rows], seed); mr = fit_hgb(sr, reps[sr['rep']][rows], y[rows], seed); cf_fits += 2
        oof['orig'][te] = mo.decision_function(reps[so['rep']][te]).reshape(-1)
        oof['rep'][te] = mr.decision_function(reps[sr['rep']][te]).reshape(-1)
        per_fold[str(int(k))] = {'selected_orig': so, 'selected_rep': sr}
        if log:
            log(json.dumps({'crossfit_fold': int(k), 'selected_orig': so, 'selected_rep': sr, 'cpu_s': round(time.process_time() - t0, 1)}))
    return {'main': main, 'selected_orig': sel_o, 'selected_rep': sel_r, 'hgb_orig': hgb_o, 'hgb_rep': hgb_r, 'oof': oof,
            'per_fold': per_fold, 'n_fits_main': fits, 'n_fits_crossfit': cf_fits, 'cpu_s': time.process_time() - t0}


# ---------------------------------------------------------------- real-data Step 1
def _load(role, wanted, purpose=None):
    if 'sex' in wanted or 'occupation' in wanted:
        raise PermissionError('the income branch reads income labels only (no SEX, no occupation label)')
    if role not in ('MODEL_FIT', 'CALIBRATION') and set(wanted) - {'features'}:
        raise PermissionError(f'the income branch reads only features on {role}')
    return C.load_role(role, wanted, purpose=purpose)


def _metric_summary(v):
    return {m: float(v[m].mean()) for m in ('brier', 'nll', 'accuracy')}


def _paired(va, vb, unit):
    """b minus a (positive = b worse for Brier / NLL; accuracy as loss of b against a) with household-clustered SEs."""
    out = {}
    for m in ('brier', 'nll'):
        d = vb[m] - va[m]; out[m] = {'value': float(d.mean()), 'se': FH.cluster_se(d, unit)}
    d = va['accuracy'] - vb['accuracy']; out['accuracy_loss'] = {'value': float(d.mean()), 'se': FH.cluster_se(d, unit)}
    return out


def source_step1(seed):
    rec = json.loads(C.admitted(f'runs/H/STEP1/S{seed}/STEP1.json').read_text())
    p = C.admitted(f'runs/H/STEP1/S{seed}/HGB.pkl')
    if C.sha(p) != rec['hgb_pickle_sha256']:
        raise PermissionError('source HGB.pkl differs from its STEP1 record')
    with open(p, 'rb') as f:
        return pickle.load(f), rec


def step1(seed):
    t0 = time.process_time()
    from . import devrun as D
    out_dir = STEP1_DIR / f'S{seed}'
    if (out_dir / 'STEP1.json').exists():
        raise PermissionError('Step 1 already recorded for this seed (append-only)')
    F = _load('MODEL_FIT', ('features', 'income'), purpose=f'income branch Step 1 S{seed}: HGB income learner (both POBP codings), household CV and cross-fitting (no SEX)')
    names, y, unit = F['feature_names'], F['income'], F['unit']
    reps = {r: rep_matrix(r, F['X'], names) for r in ('cmp', 'cmp_po', 'ohe')}
    src_folds = C.household_folds(unit, 5, SRC_CV_TAG.format(seed=seed))
    cf_folds = C.crossfit_folds(unit)
    orig, rep = grids()
    core = step1_core(reps, y, src_folds, cf_folds, orig, rep, seed, log=lambda s: print(s, flush=True))
    # reproduction of the source Step 1 with the original coding
    src_hgb, src_rec = source_step1(seed)
    src_grid = {key(g['params']): g for g in src_rec['cv']['grid']}
    repro = {'selected_equal': core['selected_orig'] == {**src_rec['cv']['selected']},
             'grid_best_iter_equal': all(core['main'][kk]['best_iter'] == src_grid[kk]['best_iter'] for kk in src_grid),
             'grid_oof_brier_max_abs_diff': max(abs(core['main'][kk]['oof_brier'] - src_grid[kk]['oof_brier']) for kk in src_grid)}
    feats = {role: (F if role == 'MODEL_FIT' else _load(role, ('features',))) for role in ROLES}
    repro['hgb_bitwise_all_roles'] = bool(all(np.array_equal(core['hgb_orig'].decision_function(rep_matrix('cmp', feats[r]['X'], names)),
                                                             src_hgb.decision_function(rep_matrix('cmp', feats[r]['X'], names))) for r in ROLES)) \
        if core['selected_orig']['rep'] == 'cmp' else None
    # household cross-fitting check of the HGB layer (OHN1), both codings
    vo = FH.person_metrics(y, core['oof']['orig']); vr = FH.person_metrics(y, core['oof']['rep'])
    crossfit = {'oof_orig': _metric_summary(vo), 'oof_rep': _metric_summary(vr), 'rep_minus_orig': _paired(vo, vr, unit),
                'per_fold': core['per_fold'], 'folds': 'ohn.common.crossfit_folds (OHN1_crossfit), inner CV on the source folds restricted to the training rows'}
    # CALIBRATION diagnostic (calibrator side; as the source's Step-1 comparison): raw and household cross-fitted logistic maps
    Cal = _load('CALIBRATION', ('features', 'income'), purpose=f'income branch Step 1 S{seed}: CALIBRATION comparison of cross-fitted stacks (calibrator diagnostic)')
    Xc, yc, uc = Cal['X'], Cal['income'], Cal['unit']
    sc = FH.stored_scores(seed, Xc, names)
    comp = {'T_TREE': sc['T_TREE'], 'HGB_ORIG': core['hgb_orig'].decision_function(rep_matrix(core['selected_orig']['rep'], Xc, names)).reshape(-1),
            'HGB_PO': core['hgb_rep'].decision_function(rep_matrix(core['selected_rep']['rep'], Xc, names)).reshape(-1)}
    cfolds = C.household_folds(uc, 5, SRC_CAL_TAG.format(seed=seed))
    vecs = {'raw_T_TREE': FH.person_metrics(yc, comp['T_TREE']), 'raw_HGB_ORIG': FH.person_metrics(yc, comp['HGB_ORIG']),
            'raw_HGB_PO': FH.person_metrics(yc, comp['HGB_PO'])}
    n_cal = 0
    for name, cols in (('cf_STRONG_ORIG', ['HGB_ORIG', 'T_TREE']), ('cf_STRONG_PO', ['HGB_PO', 'T_TREE']), ('cf_HGB_ORIG', ['HGB_ORIG']),
                       ('cf_HGB_PO', ['HGB_PO']), ('cf_T_TREE', ['T_TREE'])):
        Z = np.column_stack([comp[c] for c in cols]); d = np.zeros(len(yc))
        for k in range(5):
            tr, te = cfolds != k, cfolds == k
            d[te] = FH.apply_logit(FH.fit_logit(Z[tr], yc[tr]), Z[te]); n_cal += 1
        vecs[name] = FH.person_metrics(yc, d)
    cal = {name: {**_metric_summary(v), 'vs_raw_T_TREE': _paired(vecs['raw_T_TREE'], v, uc)} for name, v in vecs.items()}
    cal['cf_STRONG_PO_minus_cf_STRONG_ORIG'] = _paired(vecs['cf_STRONG_ORIG'], vecs['cf_STRONG_PO'], uc)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(out_dir / 'HGB_PO.pkl', 'wb') as f:
        pickle.dump(core['hgb_rep'], f, protocol=5)
    with open(out_dir / 'CROSSFIT_INCOME.npz', 'wb') as f:
        np.savez(f, MODEL_FIT__oof_logodds_orig=core['oof']['orig'], MODEL_FIT__oof_logodds_rep=core['oof']['rep'], fold=cf_folds)
    rec = {'seed': seed, 'schema': 'OHN-INC-step1-v1', 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'roles_read': 'MODEL_FIT (features, income); CALIBRATION (features, income: calibrator-side comparison); features of all six roles '
                         '(reproduction check); stored T_TREE income parameters; admitted source STEP1.json / HGB.pkl',
           'sex_used': False, 'model_select_read': False,
           'cv': {'folds': 5, 'tag': SRC_CV_TAG.format(seed=seed), 'grid_main': core['main'], 'selected_orig': core['selected_orig'],
                  'selected_rep': core['selected_rep']},
           'reproduction_of_source_step1': repro, 'crossfit_hgb_layer': crossfit, 'calibration_comparison': cal,
           'n_model_fits': core['n_fits_main'], 'n_crossfit_fits': core['n_fits_crossfit'], 'n_calibrator_fits': n_cal,
           'cpu_s': time.process_time() - t0, 'hgb_po_pickle_sha256': C.sha(out_dir / 'HGB_PO.pkl'),
           'crossfit_income_sha256': C.sha(out_dir / 'CROSSFIT_INCOME.npz'), 'code_sha256': C.sha(__file__), 'interpreter': sys.executable,
           'sklearn': __import__('sklearn').__version__}
    C.write_json_x(out_dir / 'STEP1.json', rec)
    D.unit_stats(model_fits=core['n_fits_main'], crossfit_fits=core['n_fits_crossfit'], calibrator_fits=n_cal)
    print(json.dumps({'seed': seed, 'selected_orig': core['selected_orig'], 'selected_rep': core['selected_rep'], 'reproduction': repro,
                      'crossfit': {k: v for k, v in crossfit.items() if k != 'per_fold'},
                      'cal': {k: round(v['brier'], 5) for k, v in cal.items() if isinstance(v.get('brier'), float)}}, default=float))
    # (console-summary fix 06:41Z: the first Step-1 units failed here, after every output above was written and hash-recorded)
    return rec


def load_step1(seed):
    d = STEP1_DIR / f'S{seed}'
    rec = json.loads((d / 'STEP1.json').read_text())
    if C.sha(d / 'HGB_PO.pkl') != rec['hgb_po_pickle_sha256']:
        raise PermissionError('repaired HGB file differs from its Step-1 record')
    with open(d / 'HGB_PO.pkl', 'rb') as f:
        return pickle.load(f), rec


# ---------------------------------------------------------------- release maps (the source's H3 step, unchanged)
def components(seed, X, names, hgb, params):
    sc = FH.stored_scores(seed, X, names)
    return {'T_TREE': sc['T_TREE'], 'LOCAL': sc['LOCAL_TREE'], 'D_J': sc['JOINT_TREE'] - sc['T_TREE'], 'D_L': sc['LOCAL_TREE'] - sc['T_TREE'],
            'HGB': hgb.decision_function(rep_matrix(params['rep'], X, names)).reshape(-1)}


def income_step(seed, source_config, hgb, params, Xc, names, yc, uc):
    """The source's step for one configuration on a given base learner: stack, (budgeted beta), Platt map. Returns state."""
    hcfg = FH.CONFIGS[source_config]
    cc = components(seed, Xc, names, hgb, params)
    Zb = FH.base_design(cc, hcfg['base'])
    folds = C.household_folds(uc, 5, SRC_BUDGET_TAG.format(seed=seed))
    table, note, nfit = None, 'no budgeted direction', 0
    if hcfg.get('direction') is None:
        beta = 0.0
    else:
        beta, table, nfit = FH.budget_select(Zb, cc, hcfg, yc, uc, folds, FH.BUDGETS[hcfg['budget']])
        note = 'largest grid beta inside the budget'
        if beta is None:
            beta, note = 0.0, 'OUTSIDE BUDGET even at beta = 0 (the fixed residual alone exceeds it); released at beta = 0'
    pres = bool(hcfg.get('preserve_decision'))
    v0 = FH.person_metrics(yc, FH.crossfit(Zb, np.zeros(len(yc)), yc, folds))
    v1 = FH.person_metrics(yc, FH.crossfit(Zb, FH.offset(cc, hcfg, beta), yc, folds, pres))
    cal_cost = {'brier': float((v1['brier'] - v0['brier']).mean()), 'nll': float((v1['nll'] - v0['nll']).mean()),
                'accuracy': float((v0['accuracy'] - v1['accuracy']).mean()), 'reference': 'cross-fitted task-only base map',
                'base_reference': {m: float(v0[m].mean()) for m in ('brier', 'nll', 'accuracy')}}
    wb, wp = FH.fit_maps(Zb, FH.offset(cc, hcfg, beta), yc)
    return {'source_config': source_config, 'source_step': hcfg, 'beta': beta, 'budget_note': note, 'budget_table': table,
            'stack_w': wb.tolist(), 'platt_w': wp.tolist(), 'calibration_cost_vs_base': cal_cost,
            'n_calibrator_fits': nfit + 4 * 5 + 2}


def income_logodds(seed, state, hgb, params, X, names):
    hcfg = FH.CONFIGS[state['source_config']]
    comp = components(seed, X, names, hgb, params)
    return FH.release_logodds(FH.base_design(comp, hcfg['base']), FH.offset(comp, hcfg, state['beta']), np.asarray(state['stack_w']),
                              np.asarray(state['platt_w']), bool(hcfg.get('preserve_decision')))


def witness_occupation(seed, X, names, witness=None):
    fam, cfg = witness or WITNESS
    mod = importlib.import_module(f'ohn.fam_{fam.lower()}')
    p = mod.predict(seed, cfg, X, names)
    p = p['occupation'] if isinstance(p, dict) else p
    return np.asarray(p, dtype=np.float64)


def state_path(seed, config_id):
    return C.RUNS / FAMILY / config_id / f'S{seed}' / 'STATE.json'


def fit(seed, config_id):
    t0 = time.process_time()
    from . import devrun as D
    if WITNESS is None:
        raise PermissionError('WITNESS is not set: the occupation witness is fixed only after WITNESS_SELECTION.json')
    cfg = CONFIGS[config_id]
    if (C.RUNS / FAMILY / config_id / f'S{seed}' / 'RELEASE.npz').exists():
        raise PermissionError('release already saved (append-only)')
    hgb, s1 = load_step1(seed); params = s1['cv']['selected_rep']
    Cal = _load('CALIBRATION', ('features', 'income'), purpose=f'income branch {config_id} S{seed}: stack, budget rule and Platt map (calibrators)')
    names = Cal['feature_names']; Xc, yc, uc = Cal['X'], Cal['income'], Cal['unit']
    state = income_step(seed, cfg['source_config'], hgb, params, Xc, names, yc, uc)
    # reproduction: the same step on the source's own HGB must reproduce the source state and released income
    src_hgb, src_rec = source_step1(seed)
    src_state = income_step(seed, cfg['source_config'], src_hgb, src_rec['cv']['selected'], Xc, names, yc, uc)
    stored = json.loads(C.admitted(f"runs/H/{cfg['source_config']}/S{seed}/STATE.json").read_text())
    src_rel, _ = C.ihs_release('H', cfg['source_config'], seed)
    repro = {'beta_equal': src_state['beta'] == stored['beta'],
             'stack_w_max_abs_diff': float(np.max(np.abs(np.asarray(src_state['stack_w']) - np.asarray(stored['stack_w'])))),
             'platt_w_max_abs_diff': float(np.max(np.abs(np.asarray(src_state['platt_w']) - np.asarray(stored['platt_w']))))}
    worst = 0.0; dec = True
    for role in ROLES:
        Fr = _load(role, ('features',))
        d_src = income_logodds(seed, src_state, src_hgb, src_rec['cv']['selected'], Fr['X'], Fr['feature_names'])
        p_src = FH.inc_prob(d_src)
        worst = max(worst, float(np.max(np.abs(p_src - src_rel[role]['income'])))); dec &= bool(np.array_equal(p_src.argmax(1), src_rel[role]['income'].argmax(1)))
    repro['released_income_max_abs_diff_all_roles'] = worst; repro['released_income_decisions_identical'] = dec
    repro['ok'] = bool(repro['beta_equal'] and repro['stack_w_max_abs_diff'] <= 1e-9 and repro['platt_w_max_abs_diff'] <= 1e-9 and worst <= 1e-12 and dec)
    # releases (all six roles): new income + the witness occupation (checked bitwise against the witness's saved release)
    fam, wcfg = WITNESS
    w_saved, w_rec, _ = D.load_occupation(fam, wcfg, seed)
    rel, summary = {}, {}
    for role in ROLES:
        Fr = _load(role, ('features',))
        d = income_logodds(seed, state, hgb, params, Fr['X'], Fr['feature_names'])
        occ = witness_occupation(seed, Fr['X'], Fr['feature_names'])
        if not np.array_equal(occ, w_saved[role]):
            raise AssertionError(f'{role}: witness occupation recomputed from features differs from its saved release')
        rel[role] = {'income': FH.inc_prob(d), 'occupation': occ}
        summary[role] = {'n': int(len(d)), 'share_income_class1': float((d > 0).mean()), 'mean_p1': float(rel[role]['income'][:, 1].mean())}
    st = {'seed': seed, 'config_id': config_id, 'config': cfg, 'state': state, 'hgb_params': params, 'hgb_po_pickle_sha256': s1['hgb_po_pickle_sha256'],
          'step1_sha256': C.sha(STEP1_DIR / f'S{seed}' / 'STEP1.json'), 'witness': list(WITNESS), 'witness_occ_sha256': w_rec['occ_sha256']}
    C.write_json(state_path(seed, config_id), st)
    exposure = ('income: MODEL_FIT income labels (repaired HGB, household CV), CALIBRATION income labels (stack, '
                + ('budget rule, ' if state['beta'] or cfg['source_config'] != 'HC_STRONG' else '') + 'Platt map); frozen T_TREE income routing (MODEL_FIT labels, in-sample there)'
                + ('; stored LOCAL_TREE / JOINT_TREE E20 income corrections (replication Stage B: MODEL_FIT task labels and SEX)' if cfg['source_config'] != 'HC_STRONG' else '')
                + f'; occupation: the witness {fam}/{wcfg} (its own exposure in its OCC.json); no MODEL_SELECT / ATTACK / DEV_REPORT label')
    rec = {'kind': cfg['kind'], 'budget_class': cfg['budget_class'],
           'operation': (f"INC {config_id}: {cfg['what']}. Income log-odds d = a*(s_B + off) + b with s_B = w0 + w1*s_PO + w2*s_T "
                         f"(stack_w {state['stack_w']}), off = {'D_L + beta*D_J' if cfg['source_config'] != 'HC_STRONG' else '0'}, beta = {state['beta']} "
                         f"({state['budget_note']}), Platt (b, a) = {state['platt_w']}; s_PO = repaired HGB {params}; income = trees.probabilities('income', d); "
                         f"occupation = {fam}/{wcfg} S{seed} (bitwise its saved release); decisions = argmax"),
           'roles_read': 'MODEL_FIT (via Step 1: features, income); CALIBRATION (features, income); features of all six roles; admitted source states (reproduction)',
           'exposure': exposure, 'dev_guided': 'branch opened through leg B (published DEV_REPORT seed-1 income accuracy of the source anchor); no new DEV_REPORT read',
           'n_model_fits': 1, 'n_model_fits_note': 'the repaired Step-1 HGB refit (Step-1 CV and cross-fitting fits are counted in runs/INC/STEP1/S<s>/STEP1.json)',
           'n_crossfit_fits': 0, 'n_calibrator_fits': state['n_calibrator_fits'], 'fit_cpu_s': time.process_time() - t0,
           'reproduction_of_source_step': repro, 'state_sha256': C.sha(state_path(seed, config_id)), 'role_summary': summary,
           'step1_crossfit_hgb_layer': {k: v for k, v in s1['crossfit_hgb_layer'].items() if k != 'per_fold'},
           'step1_calibration_comparison': s1['calibration_comparison'], 'code_sha256': C.sha(__file__),
           'matched_twin': 'INC_TW_PO' if config_id != 'INC_TW_PO' else 'itself (task-only)',
           'fixed_h3_comparison': f'{fam}/{wcfg} scored with anchor H3 (and TW for the twin)'}
    D.save_complete(FAMILY, config_id, seed, rel, rec)
    D.unit_stats(model_fits=0, calibrator_fits=state['n_calibrator_fits'] + src_state['n_calibrator_fits'])
    print(json.dumps({'config': config_id, 'seed': seed, 'beta': state['beta'], 'stack_w': state['stack_w'], 'platt_w': state['platt_w'],
                      'cal_cost': state['calibration_cost_vs_base'], 'reproduction': repro}, default=float))
    return rec


def predict(seed, config_id, X, names):
    st = json.loads(state_path(seed, config_id).read_text())
    rec = json.loads((C.RUNS / FAMILY / config_id / f'S{seed}' / 'RELEASE.json').read_text())
    if C.sha(state_path(seed, config_id)) != rec['state_sha256']:
        raise PermissionError('STATE.json differs from its record')
    hgb, s1 = load_step1(seed)
    if s1['hgb_po_pickle_sha256'] != st['hgb_po_pickle_sha256']:
        raise PermissionError('repaired HGB differs from the configuration state')
    d = income_logodds(seed, st['state'], hgb, st['hgb_params'], X, names)
    return {'income': FH.inc_prob(d), 'occupation': witness_occupation(seed, X, names, tuple(st['witness']))}


def verify(seed, config_id):
    from . import devrun as D
    rel, rec = D.load_complete(FAMILY, config_id, seed)
    res = {}
    for role in ROLES:
        Fr = _load(role, ('features',))
        p = predict(seed, config_id, Fr['X'], Fr['feature_names'])
        res[role] = {t: {'bitwise': bool(np.array_equal(p[t], rel[role][t])), 'max_abs_diff': float(np.max(np.abs(p[t] - rel[role][t])))} for t in C.TASKS}
    ok = all(v[t]['bitwise'] for v in res.values() for t in C.TASKS)
    C.write_json(C.RUNS / FAMILY / config_id / f'S{seed}' / 'VERIFY.json', {'ok': ok, 'roles': res, 'utc': C.now(), 'release_sha256': rec['release_sha256'],
                                                                            'code_sha256': C.sha(__file__), 'interpreter': sys.executable})
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'config': config_id, 'seed': seed}))
    return ok


# ---------------------------------------------------------------- invented-data self-test (no real data)
def selftest():
    rng = np.random.default_rng(20261018)
    names = list(C.teacher(0, 'income').feature_names)          # schema only (stored teacher payload; no person data)
    checks, info = {}, {}
    orig, rep = grids()
    checks['grids_differ_only_in_compact_coding'] = bool(len(orig) == len(rep) == 7 and all(
        (a['rep'] == 'cmp' and b['rep'] == 'cmp_po' and {**a, 'rep': 'x'} == {**b, 'rep': 'x'}) or (a == b and a['rep'] == 'ohe') for a, b in zip(orig, rep)))
    checks['union_grid_13'] = len(union_grid(orig, rep)) == 13
    checks['make_hgb_orig_equals_source'] = bool(all(make_hgb(p, 3).get_params() == FH.make_hgb(p, 3).get_params() for p in orig))
    po = make_hgb(rep[0], 3).get_params()
    checks['make_hgb_po_differs_only_in_categories'] = bool({k: v for k, v in po.items() if k != 'categorical_features'} ==
                                                            {k: v for k, v in FH.make_hgb(orig[0], 3).get_params().items() if k != 'categorical_features'})
    n = 1600
    X = A._fake_X(n, names, rng)
    unit = np.repeat(np.arange(n // 2), 2)
    from ihs import common as IC
    checks['folds_equal_source_function'] = bool(np.array_equal(C.household_folds(unit, 5, 'IHS_H_step1_cv_S0'), IC.household_folds(unit, 5, 'IHS_H_step1_cv_S0')))
    Xc = FH.compact(X, names)
    logit = 1.1 * Xc[:, 1] + 0.25 * (Xc[:, 3] - 16) - 0.3 * (Xc[:, 5] % 5 == 0) - 0.4
    y = (rng.random(n) < FH.sigmoid(logit)).astype(np.int64)
    reps = {r: rep_matrix(r, X, names) for r in ('cmp', 'cmp_po', 'ohe')}
    folds = C.household_folds(unit, 5, 'OHN_INC_selftest')
    p_small = {'rep': 'cmp', 'lr': 0.1, 'max_iter': 30, 'leaves': 8, 'msl': 20, 'l2': 1.0}
    a_, _ = cv_curves(reps['cmp'], y, folds, p_small, 0); b_, _ = FH.cv_curves(reps['cmp'], y, folds, p_small, 0)
    checks['cv_curves_equal_source'] = bool(all(np.array_equal(a_[k], b_[k]) for k in a_))
    # small grids for the pipeline (same structure: compact settings x2 codings + one shared one-hot setting)
    so = [{'rep': 'cmp', 'lr': 0.1, 'max_iter': 25, 'leaves': L, 'msl': 20, 'l2': 1.0} for L in (4, 8)] + [{'rep': 'ohe', 'lr': 0.1, 'max_iter': 25, 'leaves': 4, 'msl': 20, 'l2': 1.0}]
    sr = [{**p, 'rep': 'cmp_po'} if p['rep'] == 'cmp' else dict(p) for p in so]
    cff = C.household_folds(unit, 5, 'OHN_INC_selftest_cf')
    core = step1_core(reps, y, folds, cff, so, sr, 0)
    # brute force: the selection of each outer pipeline and its out-of-fold predictions from fresh fits
    ok_sel, ok_oof = True, True
    for k in range(5):
        rows = cff != k; te = cff == k
        best = {}
        for name, grid in (('orig', so), ('rep', sr)):
            cand = []
            for p in grid:
                c, _ = FH.cv_curves(reps[p['rep']][rows], y[rows], folds[rows], p, 0) if p['rep'] != 'cmp_po' else cv_curves(reps[p['rep']][rows], y[rows], folds[rows], p, 0)
                i = int(np.argmin(c['brier'])); cand.append((c['brier'][i], {**p, 'max_iter': i + 1}))
            sel = min(cand, key=lambda t: t[0])[1]
            best[name] = sel
            m = fit_hgb(sel, reps[sel['rep']][rows], y[rows], 0)
            ok_oof &= bool(np.array_equal(m.decision_function(reps[sel['rep']][te]).reshape(-1), core['oof'][name][te]))
        ok_sel &= bool(best['orig'] == core['per_fold'][str(k)]['selected_orig'] and best['rep'] == core['per_fold'][str(k)]['selected_rep'])
    checks['crossfit_selection_equals_bruteforce'] = bool(ok_sel)
    checks['crossfit_oof_equals_fresh_refits_bitwise'] = bool(ok_oof)
    checks['fit_counts'] = bool(core['n_fits_main'] == 5 * 5 + 2 and core['n_fits_crossfit'] == 5 * (5 * 5 + 2))
    # the source step on invented scores: budget rule, maps and release validity (stored teacher routing on invented features)
    comp = components(0, X, names, core['hgb_rep'], core['selected_rep'])
    checks['components_finite'] = bool(all(np.isfinite(v).all() for v in comp.values()))
    yc = (rng.random(n) < FH.sigmoid(comp['T_TREE'] * 0.5)).astype(np.int64)
    st_tw = income_step(0, 'HC_STRONG', core['hgb_rep'], core['selected_rep'], X, names, yc, unit)
    st_h3 = income_step(0, 'H3_STRONG_LJDIR_PB', core['hgb_rep'], core['selected_rep'], X, names, yc, unit)
    checks['twin_has_no_residual'] = bool(st_tw['beta'] == 0.0 and st_tw['budget_table'] is None)
    checks['h3_beta_on_grid_and_rule_respected'] = bool(st_h3['beta'] in FH.BETA_GRID and all(r['within'] for r in st_h3['budget_table'] if r['beta'] <= st_h3['beta']))
    d = income_logodds(0, st_h3, core['hgb_rep'], core['selected_rep'], X, names)
    P = FH.inc_prob(d)
    checks['income_release_valid'] = bool(np.isfinite(P).all() and (P >= 0).all() and np.allclose(P.sum(1), 1, atol=1e-12) and np.array_equal(P.argmax(1), (d > 0).astype(np.int64)))
    buf = io.BytesIO(); pickle.dump(core['hgb_rep'], buf, protocol=5); h2 = pickle.loads(buf.getvalue())
    checks['pickled_hgb_reproduces_income_bitwise'] = bool(np.array_equal(FH.inc_prob(income_logodds(0, st_h3, h2, core['selected_rep'], X, names)), P))
    for wanted, role, kk in ((('features', 'sex'), 'MODEL_FIT', 'guard_refuses_sex'), (('features', 'occupation'), 'MODEL_FIT', 'guard_refuses_occupation'),
                             (('features', 'income'), 'MODEL_SELECT', 'guard_refuses_ms_labels')):
        try:
            _load(role, wanted); checks[kk] = False
        except PermissionError:
            checks[kk] = True
    try:
        fit(0, 'INC_TW_PO') if WITNESS is None else None
        checks['fit_refuses_without_witness'] = WITNESS is not None
    except PermissionError:
        checks['fit_refuses_without_witness'] = True
    if WITNESS is not None:                                     # the witness's own predict on invented features (saved models; label-free)
        ok_w = True
        for s in C.SEEDS:
            q = witness_occupation(s, X[:300], names)
            ok_w &= bool(q.shape == (300, 5) and np.isfinite(q).all() and (q >= 0).all() and np.allclose(q.sum(1), 1, atol=1e-9))
            ok_w &= bool(np.array_equal(q, witness_occupation(s, X[:300], names)))
        checks['witness_predict_valid_and_deterministic'] = ok_w
        info['witness'] = list(WITNESS)
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'info': info, 'sklearn': __import__('sklearn').__version__,
                      'interpreter': sys.executable}, default=float))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('step1'); s.add_argument('--seed', type=int, required=True)
    f = sub.add_parser('fit'); f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True)
    v = sub.add_parser('verify'); v.add_argument('--seed', type=int, required=True); v.add_argument('--config', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    if a.cmd == 'step1':
        step1(a.seed); return 0
    if a.cmd == 'verify':
        return 0 if verify(a.seed, a.config) else 1
    fit(a.seed, a.config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
