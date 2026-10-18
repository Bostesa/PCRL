"""Family H: headroom-first income learner. PCRL income-headroom successor development campaign (EXPLORATORY DEVELOPMENT on
previously used records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the development roles of the
protected-tree replication. The service predicts each person's income class (2 classes: PINCP x ADJINC / 10^6 above $50,000)
and occupation group (5 classes: OCCP mapped to the five civilian broad groups) from the six permitted raw fields (AGEP, WKHP,
COW, SCHL, MAR, POBP; admitted 145-column encoding X). Each recipient receives only its own float64 probability vector and
argmax decision. SEX is the dataset's recorded demographic column; separate evaluation classifiers ("readers") estimate SEX
from the released probability vectors (each output alone and the linked pair).

Mechanism (roles/investigator_h/CARD.md):
  occupation release  the frozen, accepted JOINT_TREE occupation output (ihs.bases.joint_occupation, E20), unchanged.
  Step 1 (task learning; no privacy; no SEX)
      s_HGB(x)   scikit-learn HistGradientBoosting income log-odds on a compact, deterministic re-coding of X (AGEP, WKHP,
                 COW / MAR / POBP as native categories, SCHL as its ordinal code; 'cmp') or on X itself ('ohe'). Hyper-
                 parameters and the tree count are chosen by household-separated 5-fold cross-validation on MODEL_FIT
                 (ihs.common.household_folds; income labels only); the chosen setting is refitted on all of MODEL_FIT.
      s_B(x)     the STRONG base: a 3-coefficient logistic stack w0 + w1 s_HGB(x) + w2 s_T(x) of s_HGB and the frozen T_TREE
                 income log-odds s_T, fitted on CALIBRATION (income labels; both scores are out-of-sample there).
                 Released alone (beside the JOINT_TREE occupation output) this is the matched task-only control HC_STRONG.
  Step 2 (privacy operation; the configuration registry CONFIGS below)
      LJDIR      budgeted transplant of the replication's privacy-trained corrections as a residual:
                 d(x) = a * (s_B(x) + D_L(x) + beta * D_J(x)) + b, where D_L = s_L - s_T and D_J = s_J - s_T are the stored
                 LOCAL_TREE and JOINT_TREE E20 income leaf corrections on the teacher routing (Stage B: LOCAL against online
                 SEX critics on each single wire; JOINT adding the linked-pair wire, trained jointly with the very occupation
                 output that is released here). beta is NOT swept on MODEL_SELECT: it is the largest grid value (with every
                 smaller one) whose household-cross-fitted CALIBRATION costs relative to the task-only base (no residual)
                 stay inside a fixed budget (summed Brier, NLL, accuracy; BUDGETS below); (a, b) is a 2-parameter Platt map
                 refitted on CALIBRATION for that beta (one slope for the whole score, so it cannot undo the residual
                 selectively). The learner-isolation control applies the identical step to the old base T_TREE, which gives
                 a * (s_L + beta * D_J) + b: the same privacy step on the old LOCAL_TREE income score.
  Income release      softmax(-d/2, +d/2) (acs_replication.trees.probabilities('income', d)); decision = argmax.
  The income output does not depend on the released occupation output. SEX is never a runtime input or output and is not
  used by any fit in this module; it entered only through the stored JOINT_TREE correction (replication Stage B, MODEL_FIT).
  MODEL_SELECT is never read here (only `python3 -m ihs.devrun score` reads it).

Commands (run through `python3 -m ihs.jobqueue run ...`):
  selftest                         invented-data checks (no real data)
  step1 --seed s                   Step 1: MODEL_FIT CV grid, refit, CALIBRATION comparison (cross-fitted); no SEX
  fit --seed s --config ID         release for all six roles (ihs.devrun.save_release) + state for `predict`
`predict(seed, config_id, X, names)` recomputes a saved configuration's release from features alone.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np

from acs_replication import trees
from . import common as C

FAMILY = 'H'
ROLES = C.ROLES6
H_RUNS = C.RUNS / FAMILY
STEP1_DIR = H_RUNS / 'STEP1'
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
CMP_CATEGORICAL = np.array([False, False, True, False, True, True])      # AGEP, WKHP, COW, SCHL(ordinal), MAR, POBP
N_FOLDS = 5
BETA_GRID = tuple(float(b) for b in np.round(np.arange(0.0, 2.0001, 0.05), 2))

# Step-1 grid (fixed before any fit): learning rate 0.05, up to 600 trees, L2 1.0; tree count chosen on the CV curve.
HGB_GRID = tuple([{'rep': 'cmp', 'lr': 0.05, 'max_iter': 600, 'leaves': L, 'msl': m, 'l2': 1.0} for L in (8, 16, 32) for m in (20, 80)]
                 + [{'rep': 'ohe', 'lr': 0.05, 'max_iter': 600, 'leaves': 16, 'msl': 20, 'l2': 1.0}])
ET_PARAMS = {'n_estimators': 300, 'min_samples_leaf': 20, 'max_features': 0.5}

# Privacy-step budgets: household-cross-fitted CALIBRATION costs relative to the same base at beta = 0 (paired, per person).
# B1 (fixed in CARD.md before any fit): about 0.65-0.72 of the room between T_TREE's registered MODEL_SELECT income values
# (REFERENCE_MS_TABLE.csv) and the reviewer's reserve thresholds (RESERVE_RULE.json: k = 2 x floor SE); never tuned on MODEL_SELECT.
BUDGETS = {
    'B1': {'brier': 0.0045, 'nll': 0.0080, 'accuracy': 0.0035},
}

CONFIGS = {
    'HC_STRONG': {'base': 'STRONG', 'direction': None, 'fixed': None, 'budget': None, 'kind': 'control', 'matched_twin': 'HC_STRONG (itself: task-only)',
                  'what': 'matched task-only control: the Step-1 STRONG learner (CALIBRATION stack of HGB and T_TREE income) beside the frozen JOINT_TREE occupation output'},
    'H1_STRONG_LJDIR': {'base': 'STRONG', 'direction': 'JOINT', 'fixed': {'D_L': 1.0}, 'budget': 'B1', 'kind': 'fit', 'matched_twin': 'HC_STRONG',
                        'what': 'STRONG base + the stored LOCAL_TREE income correction (full) + the budgeted JOINT_TREE income correction direction, CALIBRATION Platt map, JOINT_TREE occupation'},
    'HC_TTREE_LJDIR': {'base': 'TTREE', 'direction': 'JOINT', 'fixed': {'D_L': 1.0}, 'budget': 'B1', 'kind': 'control',
                       'matched_twin': 'T_TREE income base (reference; Platt-mapped T_TREE beside the JOINT_TREE occupation output is the budget reference)',
                       'what': 'learner-isolation control: the same privacy step on the old base, i.e. a*(s_L + beta*D_J) + b (s_L = s_T + D_L, the old LOCAL_TREE income score), budget measured against Platt-mapped T_TREE'},
    # revision 1 (H2): one change from H1 -- the residual may not change the task-only base's decisions
    'H2_STRONG_LJDIR_DP': {'base': 'STRONG', 'direction': 'JOINT', 'fixed': {'D_L': 1.0}, 'budget': 'B1', 'preserve_decision': True,
                           'kind': 'fit', 'matched_twin': 'HC_STRONG (identical income decisions by construction)',
                           'what': 'H1 made decision-preserving: STRONG base + D_L + budgeted beta*D_J, CALIBRATION Platt map, released income decisions held equal to the task-only STRONG decisions (+-TAU on the kept side where the mapped score would cross), JOINT_TREE occupation'},
    'HC_TTREE_LJDIR_DP': {'base': 'TTREE', 'direction': 'JOINT', 'fixed': {'D_L': 1.0}, 'budget': 'B1', 'preserve_decision': True,
                          'kind': 'control', 'matched_twin': 'Platt-mapped T_TREE income (identical decisions by construction)',
                          'what': 'learner-isolation control of H2: the identical decision-preserving step on the old base T_TREE (Platt-mapped)'},
    # revision 2 (H3): H1's release form (decisions may change) with a precision-aware budget rule on the same B1 values
    'H3_STRONG_LJDIR_PB': {'base': 'STRONG', 'direction': 'JOINT', 'fixed': {'D_L': 1.0}, 'budget': 'B1', 'se_multiplier': 1.0,
                           'acc_mode': 'expected', 'kind': 'fit', 'matched_twin': 'HC_STRONG', 'component_control': 'HC_TTREE_LJDIR_PB',
                           'what': 'STRONG base + D_L + beta*D_J (decisions may change), CALIBRATION Platt map, JOINT_TREE occupation; beta by the precision-aware B1 rule (Brier and NLL cost + 1 SE within B1; expected accuracy cost under the task-only base probabilities within B1; realized accuracy reported)'},
    'HC_TTREE_LJDIR_PB': {'base': 'TTREE', 'direction': 'JOINT', 'fixed': {'D_L': 1.0}, 'budget': 'B1', 'se_multiplier': 1.0,
                          'acc_mode': 'expected', 'kind': 'control', 'matched_twin': 'T_TREE income base (reference)',
                          'what': 'learner-isolation control of H3: the identical step and precision-aware rule on the old base T_TREE (Platt-mapped), i.e. about a*(s_L + beta*D_J) + b'},
}


# ---------------------------------------------------------------- small numerics (numpy only at deployment)
def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def inc_prob(d):
    """Released income probability vector for log-odds d (identical to the replication's convention)."""
    return trees.probabilities('income', np.asarray(d, dtype=np.float64).reshape(-1))


def person_metrics(y, d):
    """Per-person accuracy, NLL and summed Brier of the released income vector (scm_bench definitions)."""
    return C.task_vectors(np.asarray(y, dtype=np.int64), inc_prob(d), 2)


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64); n = len(v)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=v - v.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def fit_logit(Z, y, ridge=1e-8, iters=100):
    """Unpenalized (tiny ridge) logistic regression by Newton steps: returns w with d = w[0] + Z @ w[1:]."""
    Z = np.asarray(Z, dtype=np.float64).reshape(len(y), -1); y = np.asarray(y, dtype=np.float64)
    Z1 = np.column_stack((np.ones(len(y)), Z)); w = np.zeros(Z1.shape[1])
    R = ridge * np.eye(Z1.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        p = sigmoid(Z1 @ w)
        g = Z1.T @ (p - y) + R @ w
        H = (Z1 * (p * (1 - p))[:, None]).T @ Z1 + R
        step = np.linalg.solve(H, g); w = w - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return w


def apply_logit(w, Z):
    Z = np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)
    return w[0] + Z @ w[1:]


# ---------------------------------------------------------------- features
def compact(X, names):
    """Deterministic re-coding of the admitted 145-column X: AGEP, WKHP (as admitted), COW / MAR / POBP category codes and the
    SCHL ordinal value (SCHL=__OTHER__, the merged rare pre-grade-3 codes, maps to 3). Every one-hot block must be complete."""
    names = tuple(map(str, names)); X = np.asarray(X, dtype=np.float64)
    if names[:2] != ('AGEP', 'WKHP') or X.shape[1] != len(names):
        raise ValueError('unexpected admitted encoding')
    cols = [X[:, 0], X[:, 1]]
    for b in BLOCKS:
        idx = [i for i, n in enumerate(names) if n.startswith(b + '=')]
        Z = X[:, idx]
        if not (np.all((Z == 0) | (Z == 1)) and np.all(Z.sum(1) == 1)):
            raise ValueError(f'{b}: one-hot block is not complete')
        code = Z.argmax(1)
        if b == 'SCHL':
            vals = np.array([3.0 if names[i].endswith('__OTHER__') else float(names[i].split('=')[1]) for i in idx])
            cols.append(vals[code])
        else:
            cols.append(code.astype(np.float64))
    return np.column_stack(cols)


def rep_matrix(rep, X, names):
    return compact(X, names) if rep == 'cmp' else np.asarray(X, dtype=np.float64)


def make_hgb(params, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=params['lr'], max_iter=int(params['max_iter']),
                                          max_leaf_nodes=int(params['leaves']), min_samples_leaf=int(params['msl']),
                                          l2_regularization=float(params['l2']), max_bins=255, early_stopping=False,
                                          categorical_features=CMP_CATEGORICAL if params['rep'] == 'cmp' else None,
                                          random_state=int(seed))


def cv_curves(Xr, y, folds, params, seed):
    """Household-separated K-fold OOF curves (mean summed Brier, NLL, accuracy at every tree count)."""
    T = int(params['max_iter']); n = len(y)
    acc = {'brier': np.zeros(T), 'nll': np.zeros(T), 'accuracy': np.zeros(T)}
    fits = 0
    for k in range(int(folds.max()) + 1):
        tr, te = folds != k, folds == k
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = make_hgb(params, seed).fit(Xr[tr], y[tr])
        fits += 1
        for i, d in enumerate(m.staged_decision_function(Xr[te])):
            v = person_metrics(y[te], d)
            for key in acc:
                acc[key][i] += v[key].sum()
    return {k: v / n for k, v in acc.items()}, fits


def logodds_from_prob(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


# ---------------------------------------------------------------- stored scores (frozen teacher routing; label-free)
def stored_scores(seed, X, names):
    """Income log-odds of T_TREE, LOCAL_TREE, JOINT_TREE (E20) for any feature matrix."""
    from . import bases as B
    return {arm: B.base_raw(seed, arm, X, names, tasks=('income',))['income'].reshape(-1)
            for arm in ('T_TREE', 'LOCAL_TREE', 'JOINT_TREE')}


# ---------------------------------------------------------------- Step 1
def step1(seed):
    t0 = time.process_time()
    from . import bases as B
    F = C.load_role('MODEL_FIT', ('features', 'income'))
    Cal = C.load_role('CALIBRATION', ('features', 'income'))
    names = F['feature_names']; y = F['income']
    folds = C.household_folds(F['unit'], N_FOLDS, f'IHS_H_step1_cv_S{seed}')
    reps = {r: rep_matrix(r, F['X'], names) for r in ('cmp', 'ohe')}
    grid, fits = [], 0
    for params in HGB_GRID:
        curves, nf = cv_curves(reps[params['rep']], y, folds, params, seed); fits += nf
        i = int(np.argmin(curves['brier']))
        grid.append({'params': params, 'best_iter': i + 1, 'oof_brier': float(curves['brier'][i]), 'oof_nll': float(curves['nll'][i]),
                     'oof_accuracy': float(curves['accuracy'][i]), 'oof_nll_min': float(curves['nll'].min()),
                     'oof_brier_at_600': float(curves['brier'][-1]), 'curve_brier_every_50': [float(c) for c in curves['brier'][49::50]]})
    best = min(grid, key=lambda g: g['oof_brier'])
    params = {**best['params'], 'max_iter': best['best_iter']}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hgb = make_hgb(params, seed).fit(reps[params['rep']], y)
    fits += 1
    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(n_estimators=ET_PARAMS['n_estimators'], min_samples_leaf=ET_PARAMS['min_samples_leaf'],
                              max_features=ET_PARAMS['max_features'], random_state=int(trees.seed_for('IHS_H_ET', seed) % 2 ** 31), n_jobs=1)
    et.fit(reps['ohe'], y); fits += 1
    # CALIBRATION comparison (all scores out-of-sample there; maps cross-fitted on household folds of CALIBRATION)
    Xc = Cal['X']; yc = Cal['income']; uc = Cal['unit']
    sc = stored_scores(seed, Xc, names)
    comp = {'HGB': hgb.decision_function(rep_matrix(params['rep'], Xc, names)).reshape(-1), 'T_TREE': sc['T_TREE'],
            'LOCAL': sc['LOCAL_TREE'], 'JOINT': sc['JOINT_TREE'],
            'TASK_REFIT': B.base_raw(seed, 'TASK_REFIT', Xc, names, tasks=('income',))['income'].reshape(-1),
            'ET': logodds_from_prob(et.predict_proba(rep_matrix('ohe', Xc, names))[:, 1])}
    ref = B.stored_reference(seed, 'Ucal', 'CALIBRATION'); u0 = B.stored_reference(seed, 'U0', 'CALIBRATION')
    cfolds = C.household_folds(uc, N_FOLDS, f'IHS_H_calcv_S{seed}')
    rows = {}
    raw_vectors = {'raw_T_TREE': person_metrics(yc, comp['T_TREE']), 'raw_TASK_REFIT': person_metrics(yc, comp['TASK_REFIT']),
                   'raw_HGB': person_metrics(yc, comp['HGB']), 'raw_LOCAL': person_metrics(yc, comp['LOCAL']),
                   'raw_JOINT': person_metrics(yc, comp['JOINT']),
                   'U0': C.task_vectors(yc, u0['income'], 2), 'Ucal_in_sample': C.task_vectors(yc, ref['income'], 2)}
    stacks = {'cf_T_TREE': ['T_TREE'], 'cf_TASK_REFIT': ['TASK_REFIT'], 'cf_HGB': ['HGB'], 'cf_STRONG_HGB_T': ['HGB', 'T_TREE'],
              'cf_HGB_TR': ['HGB', 'TASK_REFIT'], 'cf_HGB_T_ET': ['HGB', 'T_TREE', 'ET'], 'cf_LOCAL': ['LOCAL']}
    for name, cols in stacks.items():
        Z = np.column_stack([comp[c] for c in cols]); d = np.zeros(len(yc))
        for k in range(N_FOLDS):
            tr, te = cfolds != k, cfolds == k
            d[te] = apply_logit(fit_logit(Z[tr], yc[tr]), Z[te])
        raw_vectors[name] = person_metrics(yc, d)
    base_T = raw_vectors['raw_T_TREE']; base_U = raw_vectors['Ucal_in_sample']
    for name, v in raw_vectors.items():
        r = {'n': int(len(yc))}
        for m in ('brier', 'nll', 'accuracy'):
            r[m] = float(v[m].mean())
            dT = (v[m] - base_T[m]) if m != 'accuracy' else (base_T[m] - v[m])
            dU = (v[m] - base_U[m]) if m != 'accuracy' else (base_U[m] - v[m])
            r[f'{m}_vs_T_TREE'] = float(dT.mean()); r[f'{m}_vs_T_TREE_se'] = cluster_se(dT, uc)
            r[f'{m}_vs_Ucal'] = float(dU.mean()); r[f'{m}_vs_Ucal_se'] = cluster_se(dU, uc)
        rows[name] = r
    out_dir = STEP1_DIR / f'S{seed}'; out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(out_dir / 'HGB.pkl', 'wb') as f:
        pickle.dump(hgb, f)
    rec = {'seed': seed, 'schema': 'IHS-H-step1-v1', 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'roles_read': 'MODEL_FIT (features, income); CALIBRATION (features, income); stored T_TREE / TASK_REFIT / LOCAL_TREE / JOINT_TREE income parameters; stored U0 / Ucal CALIBRATION outputs',
           'sex_used': False, 'model_select_read': False,
           'cv': {'folds': N_FOLDS, 'tag': f'IHS_H_step1_cv_S{seed}', 'grid': grid, 'selected': params},
           'calibration_comparison': rows,
           'note': 'cf_* = household 5-fold cross-fitted CALIBRATION logistic maps of the listed scores (honest on CALIBRATION); raw_* = unmapped scores; '
                   'Ucal_in_sample: Ucal temperature was fitted on CALIBRATION (slightly optimistic there). vs_T_TREE: positive = worse than raw T_TREE '
                   '(Brier, NLL excess; accuracy loss). MODEL_FIT T_TREE outputs are in-sample, so no OOF T_TREE comparison exists on MODEL_FIT.',
           'n_model_fits': fits + len(stacks) * N_FOLDS, 'cpu_s': time.process_time() - t0, 'hgb_pickle_sha256': C.sha(out_dir / 'HGB.pkl')}
    C.write_json(out_dir / 'STEP1.json', rec)
    print(json.dumps({'seed': seed, 'selected': params, 'oof_brier': best['oof_brier'],
                      'cal': {k: {m: round(rows[k][f'{m}_vs_T_TREE'], 5) for m in ('brier', 'nll', 'accuracy')} for k in rows}}, default=float))
    return rec


def load_step1(seed):
    d = STEP1_DIR / f'S{seed}'
    rec = json.loads((d / 'STEP1.json').read_text())
    if C.sha(d / 'HGB.pkl') != rec['hgb_pickle_sha256']:
        raise PermissionError('Step-1 HGB file differs from its record')
    with open(d / 'HGB.pkl', 'rb') as f:
        hgb = pickle.load(f)
    return hgb, rec


# ---------------------------------------------------------------- Step 2: release maps
def components(seed, X, names, hgb=None, params=None):
    sc = stored_scores(seed, X, names)
    out = {'T_TREE': sc['T_TREE'], 'LOCAL': sc['LOCAL_TREE'], 'D_J': sc['JOINT_TREE'] - sc['T_TREE'],
           'D_L': sc['LOCAL_TREE'] - sc['T_TREE']}
    if hgb is not None:
        out['HGB'] = hgb.decision_function(rep_matrix(params['rep'], X, names)).reshape(-1)
    return out


BASE_COLS = {'STRONG': ('HGB', 'T_TREE'), 'TTREE': ('T_TREE',), 'LOCAL': ('LOCAL',)}


def base_design(comp, base):
    return np.column_stack([comp[c] for c in BASE_COLS[base]])


def offset(comp, cfg, beta):
    """Privacy residual added to the stacked task-only base score: fixed stored corrections (cfg['fixed']) plus beta times the
    budgeted direction D_J (cfg['direction'] == 'JOINT')."""
    off = np.zeros(len(comp['T_TREE']))
    for name, coef in (cfg.get('fixed') or {}).items():
        off = off + float(coef) * comp[name]
    if cfg.get('direction') == 'JOINT':
        off = off + float(beta) * comp['D_J']
    return off


def fit_maps(Zb, off, y):
    """Stack weights of the task-only base (offset-free) and the Platt map of s_B + off, both on the given rows."""
    wb = fit_logit(Zb, y)
    wp = fit_logit((apply_logit(wb, Zb) + off).reshape(-1, 1), y)
    return wb, wp


TAU = 0.01     # decision-preserving release: log-odds magnitude on the base decision's side where the mapped score would cross


def release_logodds(Zb, off, wb, wp, preserve=False):
    """a * (s_B + off) + b. Decision-preserving (preserve=True): the task-only base's own decisions keep = [w . Z_B > 0]
    (the fitted base map, offset 0) are kept; every score whose decision would differ is set to +-TAU on the kept side, so
    the released decisions equal the task-only base's decisions exactly and the residual changes only the confidence."""
    d = apply_logit(wp, (apply_logit(wb, Zb) + off).reshape(-1, 1))
    if preserve:
        keep = apply_logit(wb, Zb) > 0
        d = np.where(keep == (d > 0), d, np.where(keep, TAU, -TAU))
    return d


def crossfit(Zb, off, y, folds, preserve=False):
    d = np.zeros(len(y))
    for k in range(int(folds.max()) + 1):
        tr, te = folds != k, folds == k
        wb, wp = fit_maps(Zb[tr], off[tr], y[tr])
        d[te] = release_logodds(Zb[te], off[te], wb, wp, preserve)
    return d


def budget_select(Zb, comp, cfg, y, unit, folds, budget, grid=BETA_GRID):
    """Largest grid beta such that it and every smaller grid value keep the cross-fitted CALIBRATION costs relative to the
    task-only base (the cross-fitted base map, offset 0; paired per person: summed Brier excess, NLL excess, accuracy loss)
    inside the budget. Returns (beta or None if even beta = 0 with the fixed residual is outside), the cost table (means,
    household-clustered SEs), fits."""
    pres = bool(cfg.get('preserve_decision'))
    d0 = crossfit(Zb, np.zeros(len(y)), y, folds)
    v0 = person_metrics(y, d0)
    pB = sigmoid(d0)
    z = float(cfg.get('se_multiplier', 0.0))
    table, chosen = [], None
    for beta in grid:
        d = crossfit(Zb, offset(comp, cfg, beta), y, folds, pres)
        v = person_metrics(y, d)
        cost = {'brier': v['brier'] - v0['brier'], 'nll': v['nll'] - v0['nll'], 'accuracy': v0['accuracy'] - v['accuracy']}
        row = {'beta': beta, **{m: float(c.mean()) for m, c in cost.items()}, **{f'{m}_se': cluster_se(c, unit) for m, c in cost.items()}}
        # expected accuracy cost under the cross-fitted task-only base probabilities: |2 p_B - 1| for every changed decision
        ea = np.abs(2 * pB - 1) * ((d > 0) != (d0 > 0))
        row['expected_accuracy'] = float(ea.mean()); row['changed_decisions'] = float(np.mean((d > 0) != (d0 > 0)))
        if cfg.get('acc_mode') == 'expected':
            gate = {'brier': row['brier'] + z * row['brier_se'], 'nll': row['nll'] + z * row['nll_se'], 'accuracy': row['expected_accuracy']}
        else:
            gate = {m: row[m] + (z * row[f'{m}_se'] if m != 'accuracy' else 0.0) for m in ('brier', 'nll', 'accuracy')}
        row['gate'] = gate
        row['within'] = bool(all(gate[m] <= budget[m] for m in ('brier', 'nll', 'accuracy')))
        table.append(row)
    for row in table:
        if not row['within']:
            break
        chosen = row['beta']
    return chosen, table, (len(grid) + 1) * (int(folds.max()) + 1) * 2


def state_path(seed, config_id):
    return H_RUNS / config_id / f'S{seed}' / 'STATE.json'


def fit(seed, config_id):
    t0 = time.process_time()
    from . import bases as B
    from . import devrun as D
    cfg = CONFIGS[config_id]
    hgb = params = None; s1 = None
    if cfg['base'] == 'STRONG':
        hgb, s1 = load_step1(seed); params = s1['cv']['selected']
    Cal = C.load_role('CALIBRATION', ('features', 'income'))
    names = Cal['feature_names']
    cc = components(seed, Cal['X'], names, hgb, params)
    Zb = base_design(cc, cfg['base']); yc = Cal['income']
    n_fits = 0; table = None; budget_note = 'no budgeted direction'
    folds = C.household_folds(Cal['unit'], N_FOLDS, f'IHS_H_budget_S{seed}')
    if cfg.get('direction') is None:
        beta = 0.0
    else:
        beta, table, nf = budget_select(Zb, cc, cfg, yc, Cal['unit'], folds, BUDGETS[cfg['budget']]); n_fits += nf
        budget_note = 'largest grid beta inside the budget'
        if beta is None:
            beta, budget_note = 0.0, 'OUTSIDE BUDGET even at beta = 0 (the fixed residual alone exceeds it); released at beta = 0'
    # the chosen release's own cross-fitted CALIBRATION costs vs the task-only base (recorded for the transfer check)
    pres = bool(cfg.get('preserve_decision'))
    v0 = person_metrics(yc, crossfit(Zb, np.zeros(len(yc)), yc, folds)); n_fits += 2 * N_FOLDS
    v1 = person_metrics(yc, crossfit(Zb, offset(cc, cfg, beta), yc, folds, pres)); n_fits += 2 * N_FOLDS
    cal_cost = {'brier': float((v1['brier'] - v0['brier']).mean()), 'nll': float((v1['nll'] - v0['nll']).mean()),
                'accuracy': float((v0['accuracy'] - v1['accuracy']).mean()), 'reference': 'cross-fitted task-only base map',
                'base_reference': {m: float(v0[m].mean()) for m in ('brier', 'nll', 'accuracy')}}
    wb, wp = fit_maps(Zb, offset(cc, cfg, beta), yc); n_fits += 2
    outs = {}
    occ = B.joint_occupation(seed, ROLES)
    summary = {}
    for role in ROLES:
        X, nm, rid, unit = B.load_X(role)
        comp = components(seed, X, nm, hgb, params)
        Zr = base_design(comp, cfg['base'])
        d = release_logodds(Zr, offset(comp, cfg, beta), wb, wp, pres)
        keep = apply_logit(wb, Zr) > 0
        summary[role] = {'n': int(len(d)), 'mean_p1': float(inc_prob(d)[:, 1].mean()), 'share_class1': float((d > 0).mean()),
                         'distinct_income': int(np.unique(d).size), 'share_held_at_tau': float(np.mean(np.abs(d) == TAU)) if pres else 0.0,
                         'share_decisions_equal_task_only_base': float(np.mean((d > 0) == keep))}
        outs[role] = {'income': inc_prob(d), 'occupation': occ[role]}
    state = {'seed': seed, 'config_id': config_id, 'config': cfg, 'base_columns': list(BASE_COLS[cfg['base']]),
             'stack_w': wb.tolist(), 'platt_w': wp.tolist(), 'beta': beta, 'budget_note': budget_note,
             'budget': BUDGETS.get(cfg['budget']) if cfg['budget'] else None, 'hgb_params': params,
             'hgb_pickle_sha256': s1['hgb_pickle_sha256'] if s1 else None, 'budget_table': table, 'calibration_cost_vs_base': cal_cost}
    C.write_json(state_path(seed, config_id), state)
    fixed = cfg.get('fixed') or {}
    rec = {'mechanism': f"Family H {config_id}: {cfg['what']}. Income log-odds d = a*(s_B + off) + b; s_B = CALIBRATION stack of "
                        f"{'+'.join(BASE_COLS[cfg['base']])}; off = {' + '.join(f'{c}*{k}' for k, c in fixed.items()) or '0'}"
                        f"{' + beta*D_J' if cfg.get('direction') == 'JOINT' else ''} (D_J / D_L = stored JOINT_TREE / LOCAL_TREE E20 income "
                        f"corrections on the T_TREE routing); beta = {beta} ({budget_note}; budget {cfg['budget']})"
                        + (f"; decision-preserving: where sign(d) differs from the task-only base map's decision sign(w . Z_B), d = +-{TAU} "
                           f"on that side, so the released income decisions equal the task-only base's" if cfg.get('preserve_decision') else '')
                        + "; occupation = frozen JOINT_TREE E20.",
           'calibration_cost_vs_base': cal_cost, 'budget_note': budget_note,
           'matched_twin': cfg.get('matched_twin', ''),
           'inputs_used': 'the six permitted raw fields through the admitted 145-column encoding X (and its deterministic compact re-coding)',
           'roles_read': ('MODEL_FIT (features, income: Step-1 HGB fit and CV) ; ' if hgb is not None else '')
                         + 'CALIBRATION (features, income: stack, budget cross-fitting, Platt map) ; features of all six roles ; '
                         'stored T_TREE / LOCAL_TREE / JOINT_TREE parameters (JOINT_TREE income correction trained by the replication with MODEL_FIT SEX)',
           'labels_used_at_training': ('MODEL_FIT income (HGB); ' if hgb is not None else '') + 'CALIBRATION income (maps); no SEX in this module',
           'n_model_fits': n_fits + (1 if hgb is not None else 0),
           'n_model_fits_note': 'this configuration: CALIBRATION maps incl. budget cross-fitting' + (' + the shared Step-1 HGB refit (Step-1 CV fits reported in STEP1.json)' if hgb is not None else ''),
           'fit_cpu_s': time.process_time() - t0, 'beta': beta, 'stack_w': wb.tolist(), 'platt_w': wp.tolist(),
           'hgb_params': params, 'role_summary': summary, 'state_file': str(state_path(seed, config_id).relative_to(C.ROOT))}
    D.save_release(FAMILY, config_id, seed, outs, rec)
    print(json.dumps({'config': config_id, 'seed': seed, 'beta': beta, 'stack_w': wb.tolist(), 'platt_w': wp.tolist(),
                      'budget_rows': [r for r in (table or []) if r['beta'] in (0.0, 0.25, 0.5, 0.75, 1.0, beta)]}, default=float))


def predict(seed, config_id, X, names):
    """Recompute a saved configuration's release (income and occupation probability vectors) from features alone."""
    from . import bases as B
    st = json.loads(state_path(seed, config_id).read_text())
    hgb = params = None
    if st['config']['base'] == 'STRONG':
        hgb, s1 = load_step1(seed); params = s1['cv']['selected']
    comp = components(seed, X, names, hgb, params)
    d = release_logodds(base_design(comp, st['config']['base']), offset(comp, st['config'], st['beta']),
                        np.asarray(st['stack_w']), np.asarray(st['platt_w']), bool(st['config'].get('preserve_decision')))
    thetas = C.stage_b_theta(seed, 'JOINT_TREE', 'E20')
    te = C.teacher(seed, 'occupation')
    occ = trees.probabilities('occupation', te.raw_from_leaves(te.route(X, names), thetas[1]))
    return {'income': inc_prob(d), 'occupation': occ}


def verify(seed, config_id):
    """Deployment parity: `predict` from each role's features alone against the saved release (label-free)."""
    from . import bases as B
    from . import devrun as D
    outs, rec = D.load_release(FAMILY, config_id, seed)
    res = {}
    for role in outs:
        X, names, _, _ = B.load_X(role)
        p = predict(seed, config_id, X, names)
        res[role] = {t: float(np.max(np.abs(p[t] - outs[role][t]))) for t in C.TASKS}
        res[role]['decisions_identical'] = bool(all(np.array_equal(p[t].argmax(1), outs[role][t].argmax(1)) for t in C.TASKS))
        res[role]['bitwise'] = bool(all(np.array_equal(p[t], outs[role][t]) for t in C.TASKS))
    ok = all(v['income'] <= 1e-12 and v['occupation'] <= 1e-12 and v['decisions_identical'] for v in res.values())
    bitwise = all(v['bitwise'] for v in res.values())
    C.write_json(H_RUNS / config_id / f'S{seed}' / 'VERIFY.json', {'ok': ok, 'bitwise_all_roles': bitwise, 'max_abs_diff': res, 'utc': C.now(),
                                                                   'release_sha256': rec['release_sha256']})
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'bitwise_all_roles': bitwise, 'config': config_id, 'seed': seed}))
    return ok


def verify_all():
    """Final deployment check of every saved family-H release with the current code (label-free)."""
    out = {}
    for d in sorted(H_RUNS.glob('*/S*/RELEASE.json')):
        cfg, seed = d.parent.parent.name, int(d.parent.name[1:])
        out[f'{cfg}/S{seed}'] = verify(seed, cfg)
    print(json.dumps({'verify_all': 'PASS' if all(out.values()) else 'FAIL', 'n': len(out)}))
    return all(out.values())


# ---------------------------------------------------------------- invented-data self-test (no real data)
def _fake_X(n, names, rng):
    X = np.zeros((n, len(names)), dtype=np.float32)
    X[:, 0] = rng.normal(size=n); X[:, 1] = rng.normal(size=n)
    for b in BLOCKS:
        idx = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        X[np.arange(n), rng.choice(idx, size=n)] = 1.0
    return X


def selftest():
    rng = np.random.default_rng(12345)
    names = list(C.teacher(0, 'income').feature_names)          # schema only (stored teacher payload; no person data)
    n = 3000
    X = _fake_X(n, names, rng)
    Xc = compact(X, names)
    checks = {}
    checks['compact_shape'] = Xc.shape == (n, 6)
    checks['compact_codes_in_range'] = bool(Xc[:, 2].max() < 9 and Xc[:, 4].max() < 6 and Xc[:, 5].max() < 106 and Xc[:, 3].min() >= 1)
    bad = X.copy(); bad[0, 2:11] = 0
    try:
        compact(bad, names); checks['compact_rejects_incomplete_block'] = False
    except ValueError:
        checks['compact_rejects_incomplete_block'] = True
    # invented labels depending on two invented columns
    logit_true = 1.2 * Xc[:, 1] + 0.3 * (Xc[:, 3] - 16) - 0.5
    y = (rng.random(n) < sigmoid(logit_true)).astype(np.int64)
    unit = np.repeat(np.arange(n // 2), 2)
    folds = C.household_folds(unit, N_FOLDS, 'IHS_H_selftest')
    checks['folds_household_separated'] = bool(all(len(np.unique(folds[unit == u])) == 1 for u in np.unique(unit)[:200]))
    curves, nf = cv_curves(Xc, y, folds, {'rep': 'cmp', 'lr': 0.1, 'max_iter': 40, 'leaves': 8, 'msl': 20, 'l2': 1.0}, 0)
    checks['cv_curve_improves'] = bool(curves['brier'][-1] < curves['brier'][0] and nf == N_FOLDS)
    v = person_metrics(y, logit_true)
    p = inc_prob(logit_true)
    checks['brier_is_summed'] = bool(np.allclose(v['brier'], 2 * (p[:, 1] - y) ** 2))
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack((logit_true + rng.normal(size=n), rng.normal(size=n)))
    w = fit_logit(Z, y)
    sk = LogisticRegression(C=1e10, max_iter=10000, tol=1e-12).fit(Z, y)
    checks['fit_logit_matches_sklearn'] = bool(np.allclose(w, np.r_[sk.intercept_, sk.coef_.ravel()], atol=1e-5))
    bud = {'brier': 0.004, 'nll': 0.006, 'accuracy': 0.004}
    jcfg = {'direction': 'JOINT', 'fixed': None}
    Dj = -0.5 * logit_true + rng.normal(scale=0.3, size=n)      # an invented 'privacy direction' that costs utility
    fake = {'T_TREE': np.zeros(n), 'D_J': Dj, 'D_L': np.zeros(n)}
    beta, table, _ = budget_select(Z, fake, jcfg, y, unit, folds, bud, grid=(0.0, 0.1, 0.2, 0.4, 0.8))
    checks['budget_zero_cost_at_zero'] = bool(abs(table[0]['brier']) < 1e-15)
    checks['budget_respected'] = bool(beta is not None and all(r['within'] for r in table if r['beta'] <= beta))
    checks['budget_cost_increases'] = bool(table[-1]['brier'] > table[1]['brier'])
    mild = {'T_TREE': np.zeros(n), 'D_J': rng.normal(scale=0.5, size=n), 'D_L': np.zeros(n)}   # unrelated to the label: cheap at small beta
    beta2, table2, _ = budget_select(Z, mild, jcfg, y, unit, folds, bud, grid=(0.0, 0.1, 0.2, 0.4, 0.8, 1.6))
    checks['budget_allows_cheap_direction'] = bool(beta2 is not None and beta2 > 0 and all(r['within'] for r in table2 if r['beta'] <= beta2))
    heavy = {'T_TREE': np.zeros(n), 'D_J': mild['D_J'], 'D_L': -1.0 * logit_true}            # a fixed residual that alone breaks the budget
    beta3, table3, _ = budget_select(Z, heavy, {'direction': 'JOINT', 'fixed': {'D_L': 1.0}}, y, unit, folds, bud, grid=(0.0, 0.1))
    checks['budget_flags_fixed_residual_outside'] = bool(beta3 is None and not table3[0]['within'])
    # precision-aware rule: never admits a larger beta than the plain rule; expected accuracy cost 0 where no decision changes
    pgrid = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6)
    pcfg = {'direction': 'JOINT', 'fixed': None, 'se_multiplier': 1.0, 'acc_mode': 'expected'}
    big = {'brier': 0.004, 'nll': 0.006, 'accuracy': 1.0}
    b_plain, _, _ = budget_select(Z, mild, jcfg, y, unit, folds, big, grid=pgrid)
    b_prec, t_prec, _ = budget_select(Z, mild, pcfg, y, unit, folds, big, grid=pgrid)
    checks['precision_rule_not_looser'] = bool(b_prec is not None and b_plain is not None and b_prec <= b_plain)
    checks['expected_accuracy_zero_without_changes'] = bool(t_prec[0]['expected_accuracy'] == 0.0 and all(r['expected_accuracy'] >= 0 for r in t_prec))
    # decision preservation: released decisions equal the task-only base map's decisions; accuracy cost exactly 0
    dcfg = {'direction': 'JOINT', 'fixed': None, 'preserve_decision': True}
    beta4, table4, _ = budget_select(Z, fake, dcfg, y, unit, folds, {'brier': 1.0, 'nll': 1.0, 'accuracy': 0.0}, grid=(0.0, 0.2, 0.4))
    checks['dp_accuracy_cost_exactly_zero'] = bool(all(r['accuracy'] == 0.0 for r in table4) and beta4 == 0.4)
    off4 = offset(fake, dcfg, 0.4)
    wb4, wp4 = fit_maps(Z, off4, y)
    d4 = release_logodds(Z, off4, wb4, wp4, True)
    d4n = release_logodds(Z, off4, wb4, wp4, False)
    keep4 = apply_logit(wb4, Z) > 0
    checks['dp_decisions_equal_base'] = bool(np.array_equal(d4 > 0, keep4) and not np.array_equal(d4n > 0, keep4))
    checks['dp_only_crossing_scores_changed'] = bool(np.array_equal(d4 == d4n, (d4n > 0) == keep4) and np.all(np.abs(d4[d4 != d4n]) == TAU))
    off = offset(fake, jcfg, beta)
    wb, wp = fit_maps(Z, off, y)
    d = release_logodds(Z, off, wb, wp)
    P = inc_prob(d)
    checks['release_is_probability'] = bool(np.allclose(P.sum(1), 1, atol=1e-12) and (P >= 0).all() and np.isfinite(P).all())
    checks['decision_matches_sign'] = bool(np.array_equal(P.argmax(1), (d > 0).astype(np.int64)))
    # real stored teacher routing on invented features (parameters only): D_J and occupation computed from X alone
    comp = components(0, X[:200], names)
    checks['direction_finite'] = bool(np.isfinite(comp['D_J']).all() and comp['D_J'].shape == (200,))
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'invented_budget_beta': beta}, default=float))
    return ok, checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('step1'); s.add_argument('--seed', type=int, required=True)
    f = sub.add_parser('fit'); f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True)
    f.add_argument('--verify', action='store_true')
    v = sub.add_parser('verify'); v.add_argument('--seed', type=int, required=True); v.add_argument('--config', required=True)
    sub.add_parser('verifyall')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        ok, _ = selftest(); return 0 if ok else 1
    if a.cmd == 'step1':
        step1(a.seed); return 0
    if a.cmd == 'verify':
        return 0 if verify(a.seed, a.config) else 1
    if a.cmd == 'verifyall':
        return 0 if verify_all() else 1
    fit(a.seed, a.config)
    if a.verify:
        return 0 if verify(a.seed, a.config) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
