"""Family A: stronger categorical boosted learning for the OCCUPATION output. PCRL overnight occupation-headroom campaign
(EXPLORATORY DEVELOPMENT on previously used records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed), development roles of the protected-tree replication. The service predicts an income class (2) and an
occupation group (5 official broad civilian groups) from six permitted raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted
145-column encoding X). Each recipient receives only its own float64 probability vector and argmax decision. SEX is the dataset's
recorded demographic column; separate evaluation classifiers ("readers") estimate SEX from the released probability vectors.
This family is TASK-ONLY: no fit here reads or uses SEX.

Mechanism (results/pcrl_occupation_headroom_overnight_v1/roles/investigator_a/CARD.md):
  re-coding   compact(X, names): AGEP, WKHP as admitted; COW / MAR / POBP = index of the active admitted one-hot column (native
              categories); SCHL = ordinal value of its admitted code (__OTHER__ = 3). Deterministic from X (no learned dictionary).
  learner     scikit-learn HistGradientBoostingClassifier, multinomial log loss, native categorical splits, fixed learning rate,
              capacity (max_leaf_nodes, min_samples_leaf), l2; early_stopping=False (deterministic).
  T*          first argmin of the pooled out-of-fold mean occupation NLL over 5 SEED-SPECIFIC household folds of the training
              rows (ohn.common.household_folds(unit, 5, 'OHN_A_cv_S<seed>'), the analogue of the teacher's seed-specific internal
              split; with the seed-independent OHN1 partition a deterministic learner would give bitwise-identical releases on both
              seeds), then refit on all training rows with exactly T* iterations (deploy='refit'), or the 5 inner-fold training
              sets refitted at T* and averaged (deploy='committee').
  calibrator  optional global temperature on CALIBRATION occupation labels (config 'calibrator'); always fitted and cross-fitted
              on CALIBRATION as a recorded diagnostic.
  release     (n, 5) float64 probability vectors for all six roles from X alone; decision = argmax.
  cross-fit   for each registered OHN1_crossfit fold k, the ENTIRE pipeline is rerun on MODEL_FIT minus fold k (inner CV over the
              seed-specific folds restricted to those rows for T*_k, refit / committee, calibrator refitted on CALIBRATION) and
              predicts fold k (MODEL_FIT__oof); the 5 fold pipelines' mean on MODEL_SELECT (MODEL_SELECT__foldavg). 36 physical
              fits per refit configuration (6 deployed + 30 cross-fitting); committee members of cross-fitting pipelines are inner
              models truncated at T*_k, which equal fresh T*_k-iteration fits (bitwise, selftest).

Commands (run through `python3 -m ohn.jobqueue run ...`):
  selftest                     invented-data checks only (no real data)
  fit --seed s --config ID     MODEL_FIT fit + cross-fitting, CALIBRATION calibrator (diagnostic or deployed), six-role release
  verify --seed s --config ID  predict() from features alone reproduces the saved release bitwise on all six roles
`predict(seed, config_id, X, names)` recomputes a saved configuration's occupation release from features alone.
MODEL_SELECT labels are never read here (only `python3 -m ohn.devrun score` reads them).
"""
from __future__ import annotations
import argparse
import io
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

from . import common as C

FAMILY = 'A'
ROLES = C.ROLES6
K_OCC = 5
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
CMP_CATEGORICAL = np.array([False, False, True, False, True, True])      # AGEP, WKHP, COW, SCHL (ordinal), MAR, POBP
TEMP_BOUNDS = (0.5, 2.0)
CAL_TAG = 'OHN_A_caltemp'                                                 # household folds of CALIBRATION (temperature check)

# Configuration registry. Initial design (CARD.md section 2) fixed before any real-data fit; revisions are appended with the
# card's dated sections, never edited afterwards.
CONFIGS = {
    'A0_HGB_S': {'learner': 'hgb', 'rep': 'cmp', 'lr': 0.05, 'leaves': 8, 'msl': 50, 'l2': 1.0, 'max_iter': 1500,
                 'calibrator': None, 'deploy': 'refit', 'kind': 'task_model', 'revision': 0,
                 'what': 'initial design, small capacity: HGB multinomial, native categories (compact coding), 8 leaves, min leaf 50, '
                         'l2 1, lr 0.05, T* <= 1500 by household 5-fold CV on MODEL_FIT, refit on all MODEL_FIT, no calibrator'},
    'A0_HGB_L': {'learner': 'hgb', 'rep': 'cmp', 'lr': 0.05, 'leaves': 31, 'msl': 20, 'l2': 1.0, 'max_iter': 800,
                 'calibrator': None, 'deploy': 'refit', 'kind': 'task_model', 'revision': 0,
                 'what': 'initial design, large capacity: HGB multinomial, native categories (compact coding), 31 leaves, min leaf 20, '
                         'l2 1, lr 0.05, T* <= 800 by household 5-fold CV on MODEL_FIT, refit on all MODEL_FIT, no calibrator'},
    # Revision 1 (CARD.md, decided 05:43Z after the A0 seed-0 results and the MODEL_FIT-only diagnostic): POBP as its ordinal
    # place-of-birth code instead of a native category, at the diagnostic's neutral regularisation; two capacity settings.
    'A1_PO_8': {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.025, 'leaves': 8, 'msl': 100, 'l2': 3.0, 'max_iter': 1200,
                'calibrator': None, 'deploy': 'refit', 'kind': 'task_model', 'revision': 1,
                'what': 'revision 1 (POBP coding replaced), 8 leaves: HGB multinomial on compact_po (AGEP, WKHP numeric; COW, MAR native '
                        'categories; SCHL ordinal; POBP ordinal place-of-birth code), min leaf 100, l2 3, lr 0.025, T* <= 1200 by household '
                        '5-fold CV on MODEL_FIT, refit on all MODEL_FIT, no calibrator'},
    'A1_PO_16': {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.025, 'leaves': 16, 'msl': 100, 'l2': 3.0, 'max_iter': 1000,
                 'calibrator': None, 'deploy': 'refit', 'kind': 'task_model', 'revision': 1,
                 'what': 'revision 1 (POBP coding replaced), 16 leaves: HGB multinomial on compact_po (AGEP, WKHP numeric; COW, MAR native '
                         'categories; SCHL ordinal; POBP ordinal place-of-birth code), min leaf 100, l2 3, lr 0.025, T* <= 1000 by household '
                         '5-fold CV on MODEL_FIT, refit on all MODEL_FIT, no calibrator'},
}


# Diagnostic settings (no release; MODEL_FIT internal household CV only, on the same seed-specific inner folds as the deployed
# pipelines, so the curves are directly comparable with A0's recorded main curves). Fixed 2026-10-18 ~05:20Z after the A0 seed-0
# fits (internal diagnosis: sharp early OOF minimum, smaller trees better, committee gain 4-6 SE => variance), before any
# MODEL_SELECT result of this family was read.
DIAG_SETTINGS = {
    'D1': {'learner': 'hgb', 'rep': 'cmp', 'lr': 0.025, 'leaves': 8, 'msl': 100, 'l2': 3.0, 'max_iter': 1200},     # regularization bundle
    'D2': {'learner': 'hgb', 'rep': 'cmp', 'lr': 0.025, 'leaves': 4, 'msl': 100, 'l2': 3.0, 'max_iter': 2000},     # smaller trees
    'D3': {'learner': 'hgb', 'rep': 'cmp', 'lr': 0.025, 'leaves': 16, 'msl': 200, 'l2': 3.0, 'max_iter': 1000},    # larger trees, larger leaves
    'D4': {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.025, 'leaves': 8, 'msl': 100, 'l2': 3.0, 'max_iter': 1200},  # POBP ordinal code
    'D5': {'learner': 'hgb', 'rep': 'ohe', 'lr': 0.025, 'leaves': 8, 'msl': 100, 'l2': 3.0, 'max_iter': 2000},     # admitted one-hot X
}
# (D5 max_iter raised from 1200 to 2000 at ~05:27Z, after the A0_HGB_S MODEL_SELECT score was read and before any diagnostic
# fit: single-level one-hot splits are weaker learners and need more iterations; the iteration cap is not a selection on MODEL_SELECT.)
# Second diagnostic on the revision-1 coding (fixed ~05:57Z, after the A1 seed-0 MODEL_SELECT scores; MODEL_FIT only, no release):
DIAG_SETTINGS.update({
    'D6': {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.025, 'leaves': 32, 'msl': 100, 'l2': 3.0, 'max_iter': 800},    # more capacity
    'D7': {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.01, 'leaves': 16, 'msl': 100, 'l2': 3.0, 'max_iter': 2000},    # finer steps
    'D8': {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.025, 'leaves': 16, 'msl': 200, 'l2': 10.0, 'max_iter': 1000},  # stronger leaf regularisation
})
# Third diagnostic (fixed ~06:03Z, after the A1 seed-1 scores; MODEL_FIT only, no release): the remaining native categories.
DIAG_SETTINGS.update({
    'D9': {'learner': 'hgb', 'rep': 'cmp_po_cow', 'lr': 0.025, 'leaves': 16, 'msl': 100, 'l2': 3.0, 'max_iter': 1000},  # COW as code
    'D10': {'learner': 'hgb', 'rep': 'cmp_ord', 'lr': 0.025, 'leaves': 16, 'msl': 100, 'l2': 3.0, 'max_iter': 1000},    # COW, MAR as codes
})


def run_dir(seed, config_id):
    return C.RUNS / FAMILY / config_id / f'S{seed}'


# ---------------------------------------------------------------- small numerics
def softmax(raw, tau=None):
    z = np.asarray(raw, dtype=np.float64)
    if tau is not None:
        z = z / float(tau)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metric_sums(y, p, onehot=None):
    """Summed NLL (natural log, floor 1e-12 as scm_bench.task_vectors), summed multiclass Brier and correct-decision count."""
    n = len(y)
    py = np.maximum(p[np.arange(n), y], 1e-12)
    if onehot is None:
        onehot = np.zeros_like(p); onehot[np.arange(n), y] = 1.0
    return float(-np.log(py).sum()), float(((p - onehot) ** 2).sum()), float((p.argmax(1) == y).sum())


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64); n = len(v)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=v - v.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def first_argmin(curve):
    return int(np.argmin(np.asarray(curve))) + 1          # np.argmin returns the first minimum; iterations are 1-based


# ---------------------------------------------------------------- features
def compact(X, names):
    """Deterministic re-coding of the admitted 145-column X (same as ihs.fam_h.compact): AGEP, WKHP as admitted; COW / MAR /
    POBP category codes (index inside the admitted one-hot block); SCHL ordinal value (__OTHER__ = 3). Every block must be a
    complete one-hot block."""
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


def compact_po(X, names):
    """compact() with POBP as its numeric place-of-birth CODE (ordinal geographic code: US states < 60, then world regions in
    code ranges; the admitted __OTHER__ level = 999) instead of a native category. Deterministic from X."""
    names = tuple(map(str, names))
    Z = compact(X, names)
    idx = [i for i, n in enumerate(names) if n.startswith('POBP=')]
    vals = np.array([999.0 if names[i].endswith('__OTHER__') else float(names[i].split('=')[1]) for i in idx])
    Z[:, 5] = vals[Z[:, 5].astype(np.int64)]
    return Z


def compact_codes(X, names, blocks):
    """compact_po() with the listed blocks (COW, MAR) also entered as their numeric admitted codes (__OTHER__ = 99) instead of
    native categories. Deterministic from X (diagnostic representations)."""
    names = tuple(map(str, names))
    Z = compact_po(X, names)
    col = {'COW': 2, 'MAR': 4}
    for b in blocks:
        idx = [i for i, n in enumerate(names) if n.startswith(b + '=')]
        vals = np.array([99.0 if names[i].endswith('__OTHER__') else float(names[i].split('=')[1]) for i in idx])
        Z[:, col[b]] = vals[Z[:, col[b]].astype(np.int64)]
    return Z


CAT_MASK = {'cmp': CMP_CATEGORICAL, 'cmp_po': np.array([False, False, True, False, True, False]), 'ohe': None,
            'cmp_po_cow': np.array([False, False, False, False, True, False]), 'cmp_ord': None}


def rep_matrix(rep, X, names):
    if rep == 'cmp':
        return compact(X, names)
    if rep == 'cmp_po':
        return compact_po(X, names)
    if rep == 'cmp_po_cow':
        return compact_codes(X, names, ('COW',))
    if rep == 'cmp_ord':
        return compact_codes(X, names, ('COW', 'MAR'))
    if rep == 'ohe':
        return np.asarray(X, dtype=np.float64)
    raise KeyError(rep)


# ---------------------------------------------------------------- learner
def make_model(cfg, max_iter, seed):
    if cfg['learner'] != 'hgb':
        raise KeyError(cfg['learner'])
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(cfg['lr']), max_iter=int(max_iter),
                                          max_leaf_nodes=int(cfg['leaves']), min_samples_leaf=int(cfg['msl']),
                                          l2_regularization=float(cfg['l2']), max_bins=255, early_stopping=False,
                                          categorical_features=CAT_MASK[cfg['rep']], random_state=int(seed))


def fit_model(cfg, Xr, y, max_iter, seed):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = make_model(cfg, max_iter, seed).fit(Xr, y)
    if not np.array_equal(m.classes_, np.arange(K_OCC)):
        raise AssertionError('occupation classes differ from 0..4 in a training set')
    if m.n_iter_ != int(max_iter):
        raise AssertionError('unexpected iteration count (early stopping must be off)')
    return m


def raw_full(m, Xr):
    return np.asarray(m.decision_function(Xr), dtype=np.float64)


def raw_at(m, Xr, t):
    """Raw scores of m truncated at t iterations (1-based); equals a fresh t-iteration fit (selftest)."""
    if not 1 <= t <= m.n_iter_:
        raise ValueError('truncation outside the fitted range')
    for i, r in enumerate(m.staged_decision_function(Xr), start=1):
        if i == t:
            return np.asarray(r, dtype=np.float64).copy()
    raise AssertionError('staged predictions ended early')


def staged_curves(m, Xr, y):
    """Per-iteration summed NLL, Brier and correct-decision count of the softmax release of m on (Xr, y)."""
    T = m.n_iter_
    out = np.zeros((3, T))
    onehot = np.zeros((len(y), K_OCC)); onehot[np.arange(len(y)), y] = 1.0
    for i, r in enumerate(m.staged_decision_function(Xr)):
        out[:, i] = metric_sums(y, softmax(r), onehot)
    return out


# ---------------------------------------------------------------- calibrator (global temperature)
def fit_temperature(raw, y):
    from scipy.optimize import minimize_scalar
    yy = np.asarray(y, dtype=np.int64); n = len(yy)

    def f(t):
        return metric_sums(yy, softmax(raw, t))[0] / n
    r = minimize_scalar(f, bounds=TEMP_BOUNDS, method='bounded', options={'xatol': 1e-8, 'maxiter': 500})
    tau = float(r.x)
    return tau, {'tau': tau, 'nll_at_1': f(1.0), 'nll_at_tau': f(tau),
                 'at_bound': bool(min(abs(tau - TEMP_BOUNDS[0]), abs(tau - TEMP_BOUNDS[1])) < 1e-4)}


def temperature_crossfit(raw, y, unit, tag=CAL_TAG, k=5):
    """Household k-fold cross-fitted temperature on a labelled set: per-person (temperature - raw) NLL / Brier / accuracy
    differences with household-clustered SEs. Returns (summary, number of temperature fits)."""
    folds = C.household_folds(unit, k, tag)
    p_raw = softmax(raw); p_cf = np.zeros_like(p_raw); taus = []
    for j in range(k):
        tr, te = folds != j, folds == j
        tau, _ = fit_temperature(raw[tr], y[tr]); taus.append(tau)
        p_cf[te] = softmax(raw[te], tau)
    a = C.task_vectors(y, p_raw, K_OCC); b = C.task_vectors(y, p_cf, K_OCC)
    out = {'fold_taus': taus, 'raw': {m: float(a[m].mean()) for m in a}, 'crossfit_temperature': {m: float(b[m].mean()) for m in b}}
    for m in ('nll', 'brier'):
        d = b[m] - a[m]
        out[f'{m}_change'] = float(d.mean()); out[f'{m}_change_se'] = cluster_se(d, unit)
    d = a['accuracy'] - b['accuracy']
    out['accuracy_loss_change'] = float(d.mean()); out['accuracy_loss_change_se'] = cluster_se(d, unit)
    return out, k


# ---------------------------------------------------------------- the pipeline (deployed + household cross-fitting)
def inner_tag(seed):
    return f'OHN_A_cv_S{seed}'


def cv_select(cfg, seed, Xr, y, inner, rows, log=None):
    """Inner household CV on the training rows `rows` (boolean mask): one model per inner fold j (trained on rows with inner != j,
    to T_max), staged curves on its held-out inner fold; T = first argmin of the pooled mean NLL. Returns (T, pooled curve, models)."""
    T = int(cfg['max_iter']); cur = np.zeros((3, T)); models = []
    for j in np.unique(inner[rows]):
        tr = rows & (inner != j); va = rows & (inner == j)
        if not tr.any() or not va.any():
            raise AssertionError('empty inner fold')
        m = fit_model(cfg, Xr[tr], y[tr], T, seed)
        cur += staged_curves(m, Xr[va], y[va])
        models.append(m)
        if log:
            log(json.dumps({'inner_fold': int(j), 'rows': int(rows.sum())}))
    cur /= float(rows.sum())
    return first_argmin(cur[0]), cur, models


def run_pipeline(cfg, seed, Xr, y, folds, inner, extra=None, log=print):
    """Fit the deployed pipeline and the K household cross-fitting pipelines (full nested refits).

    Xr, y      training rows (MODEL_FIT, compact or encoded)
    folds      registered cross-fitting fold per row (OHN1_crossfit, 0..K-1)
    inner      seed-specific household fold per row for the pipeline's own T* selection (OHN_A_cv_S<seed>)
    extra      {name: Xr_other}: sets on which every cross-fitting pipeline's predictions are returned (raw scores)
    Deployed pipeline: T* by inner CV on all rows; 'refit' = one model on all rows with T* iterations; 'committee' = the K_inner
    inner-fold training sets refitted with T* iterations (mean of probabilities). Cross-fitting pipeline k: the same procedure on
    the rows outside fold k (inner CV over the seed-specific folds restricted to those rows, T*_k, refit or committee); committee
    members of cross-fitting pipelines are the inner models truncated at T*_k (equal to fresh T*_k fits, checked in the selftest).
    """
    K = int(folds.max()) + 1; T = int(cfg['max_iter']); n = len(y)
    extra = extra or {}
    t0 = time.process_time()
    allr = np.ones(n, dtype=bool)
    T_star, main, dep_inner = cv_select(cfg, seed, Xr, y, inner, allr)
    if cfg['deploy'] == 'refit':
        deployed = [fit_model(cfg, Xr, y, T_star, seed)]
    elif cfg['deploy'] == 'committee':
        deployed = [fit_model(cfg, Xr[inner != j], y[inner != j], T_star, seed) for j in np.unique(inner)]
    else:
        raise KeyError(cfg['deploy'])
    log(json.dumps({'deployed_models': len(deployed), 'T_star': T_star, 'cpu_s': round(time.process_time() - t0, 1)}))
    n_phys = len(dep_inner) + len(deployed)
    T_k, inner_curves, oof_raw, comm_raw, outer_refits = {}, {}, {}, {}, {}
    extra_raw = {name: {} for name in extra}
    for k in range(K):
        rows = folds != k; te = folds == k
        T_k[k], inner_curves[k], ms = cv_select(cfg, seed, Xr, y, inner, rows)
        n_phys += len(ms)
        if cfg['deploy'] == 'refit':
            r = fit_model(cfg, Xr[rows], y[rows], T_k[k], seed); n_phys += 1
            outer_refits[k] = r
            oof_raw[k] = [raw_full(r, Xr[te])]
            for name, Z in extra.items():
                extra_raw[name][k] = [raw_full(r, Z)]
            comm_raw[k] = [raw_at(m, Xr[te], T_k[k]) for m in ms]          # committee diagnostic (honest OOF)
        else:
            oof_raw[k] = [raw_at(m, Xr[te], T_k[k]) for m in ms]
            for name, Z in extra.items():
                extra_raw[name][k] = [raw_at(m, Z, T_k[k]) for m in ms]
        log(json.dumps({'crossfit_pipeline': k, 'T_k': T_k[k], 'cpu_s': round(time.process_time() - t0, 1)}))
    n_in = len(dep_inner)
    logical = {'deployed_pipeline': n_in + len(deployed),
               'crossfit_pipelines': K * (n_in + (1 if cfg['deploy'] == 'refit' else n_in))}
    return {'K': K, 'T_max': T, 'T_star': T_star, 'T_k': T_k, 'main_curve': main, 'inner_curves': inner_curves, 'deployed': deployed,
            'deployed_inner_models': dep_inner, 'outer_refits': outer_refits, 'oof_raw': oof_raw, 'comm_raw': comm_raw,
            'extra_raw': extra_raw, 'n_physical_fits': n_phys, 'n_logical_fits': logical, 'cpu_s': time.process_time() - t0}


def combine(raw_list, tau=None):
    """Release probability of one pipeline: softmax(raw / tau) of one model, or the mean of its members' probabilities."""
    return np.mean([softmax(r, tau) for r in raw_list], axis=0) if len(raw_list) > 1 else softmax(raw_list[0], tau)


def release_prob(models, Xr, tau=None):
    return combine([raw_full(m, Xr) for m in models], tau)


# ---------------------------------------------------------------- fit (real data)
def _load(role, wanted, purpose=None):
    """Family-A loader guard: never SEX; labels only on MODEL_FIT / CALIBRATION (MODEL_SELECT labels only via the scorer)."""
    if 'sex' in wanted:
        raise PermissionError('family A is task-only: SEX is never read')
    if role not in ('MODEL_FIT', 'CALIBRATION') and set(wanted) - {'features'}:
        raise PermissionError(f'family A reads only features on {role}')
    return C.load_role(role, wanted, purpose=purpose)


def _curve_summary(curve, every=25):
    T = curve.shape[1]
    idx = sorted(set(list(range(every - 1, T, every)) + [T - 1]))
    return {'iterations': [i + 1 for i in idx], 'nll': [float(curve[0, i]) for i in idx], 'brier': [float(curve[1, i]) for i in idx],
            'accuracy': [float(curve[2, i]) for i in idx]}


def fit(seed, config_id):
    t0 = time.process_time(); w0 = time.time()
    from . import devrun as D
    cfg = CONFIGS[config_id]
    d = run_dir(seed, config_id)
    if (d / 'OCC.npz').exists():
        raise PermissionError(f'{config_id}/S{seed} already saved (append-only)')
    F = _load('MODEL_FIT', ('features', 'occupation'),
              purpose=f'family A {config_id} S{seed}: occupation task-model fit, household CV and cross-fitting (no SEX)')
    names = F['feature_names']; y = F['occupation']; unit = F['unit']
    folds = C.crossfit_folds(unit)
    inner = C.household_folds(unit, 5, inner_tag(seed))
    Xr = rep_matrix(cfg['rep'], F['X'], names)
    feats = {role: (F if role == 'MODEL_FIT' else _load(role, ('features',))) for role in ROLES}
    for role in ROLES:
        if tuple(feats[role]['feature_names']) != tuple(names):
            raise AssertionError(f'{role}: feature schema differs')
    reps = {role: rep_matrix(cfg['rep'], feats[role]['X'], names) for role in ROLES}
    Cal = _load('CALIBRATION', ('features', 'occupation'),
                purpose=f'family A {config_id} S{seed}: occupation temperature calibrator ({"deployed" if cfg["calibrator"] else "diagnostic only"})')
    if not np.array_equal(Cal['row_id'], feats['CALIBRATION']['row_id']):
        raise AssertionError('CALIBRATION row order differs')
    yc, uc = Cal['occupation'], Cal['unit']
    pipe = run_pipeline(cfg, seed, Xr, y, folds, inner, extra={'MODEL_SELECT': reps['MODEL_SELECT'], 'CALIBRATION': reps['CALIBRATION']},
                        log=lambda s: print(s, flush=True))
    n_cal = 0
    # deployed calibrator (if configured) and the fold pipelines' calibrators (refitted per pipeline on CALIBRATION)
    raw_cal_dep = [raw_full(m, reps['CALIBRATION']) for m in pipe['deployed']]
    tau = None; tau_k = {k: None for k in range(pipe['K'])}
    if cfg['calibrator'] == 'temperature':
        if len(raw_cal_dep) > 1:
            raise NotImplementedError('temperature on a committee is not part of any registered configuration')
        tau, _ = fit_temperature(raw_cal_dep[0], yc); n_cal += 1
        for k in range(pipe['K']):
            tau_k[k], _ = fit_temperature(pipe['extra_raw']['CALIBRATION'][k][0], yc); n_cal += 1
    elif cfg['calibrator'] is not None:
        raise KeyError(cfg['calibrator'])
    # releases (all six roles, from features alone)
    occ = {role: release_prob(pipe['deployed'], reps[role], tau) for role in ROLES}
    # cross-fitting arrays
    oof = np.zeros((len(y), K_OCC)); fa = np.zeros((len(feats['MODEL_SELECT']['row_id']), K_OCC))
    for k in range(pipe['K']):
        oof[folds == k] = combine(pipe['oof_raw'][k], tau_k[k])
        fa += combine(pipe['extra_raw']['MODEL_SELECT'][k], tau_k[k])
    fa /= pipe['K']
    # recorded calibrator diagnostics (not applied unless configured): CALIBRATION temperature of the deployed raw output and
    # its household cross-fitted effect; temperature of the cross-fitted MODEL_FIT predictions (internal folds)
    raw_dep_cal = raw_cal_dep[0] if len(raw_cal_dep) == 1 else np.log(release_prob(pipe['deployed'], reps['CALIBRATION']))
    tau_diag, tinfo = fit_temperature(raw_dep_cal, yc); n_cal += 1
    tcf, nf = temperature_crossfit(raw_dep_cal, yc, uc); n_cal += nf
    oof_logp = np.log(np.maximum(oof, 1e-300))
    tau_oof, toof = fit_temperature(oof_logp, y); n_cal += 1
    own = C.task_vectors(y, oof, K_OCC)
    oof_summary = {m: float(own[m].mean()) for m in own}
    oof_by_fold = {str(k): {m: float(own[m][folds == k].mean()) for m in own} for k in range(pipe['K'])}
    # committee diagnostic on MODEL_FIT (honest): each outer pipeline's 4 inner models at T*_k averaged on its held-out fold
    comm = None
    if cfg['deploy'] == 'refit':
        pc = np.zeros_like(oof)
        for k in range(pipe['K']):
            pc[folds == k] = combine(pipe['comm_raw'][k])
        cv = C.task_vectors(y, pc, K_OCC)
        comm = {'oof_committee': {m: float(cv[m].mean()) for m in cv}}
        for m in ('nll', 'brier'):
            dd = cv[m] - own[m]
            comm[f'committee_minus_refit_{m}'] = float(dd.mean()); comm[f'committee_minus_refit_{m}_se'] = cluster_se(dd, unit)
        dd = own['accuracy'] - cv['accuracy']
        comm['committee_minus_refit_accuracy_loss'] = float(dd.mean()); comm['committee_minus_refit_accuracy_loss_se'] = cluster_se(dd, unit)
    # save fitted models (deployed members and the 5 outer-pipeline refits), hash-recorded
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    model_files = {}
    for i, m in enumerate(pipe['deployed']):
        name = 'MODEL.pkl' if len(pipe['deployed']) == 1 else f'MODEL_M{i}.pkl'
        with open(d / name, 'wb') as f:
            pickle.dump(m, f, protocol=5)
        model_files[name] = C.sha(d / name)
    fold_files = {}
    for k, m in pipe['outer_refits'].items():
        name = f'FOLD_{k}.pkl'
        with open(d / name, 'wb') as f:
            pickle.dump(m, f, protocol=5)
        fold_files[name] = C.sha(d / name)
    role_summary = {role: {'n': int(len(p)), 'mean_prob': [float(x) for x in p.mean(0)],
                           'decision_share': [float(x) for x in np.bincount(p.argmax(1), minlength=K_OCC) / len(p)],
                           'distinct_vectors': int(np.unique(p, axis=0).shape[0])} for role, p in occ.items()}
    flags = []
    if pipe['T_star'] >= pipe['T_max']:
        flags.append('T_STAR_AT_T_MAX')
    if any(t >= pipe['T_max'] for t in pipe['T_k'].values()):
        flags.append('A_FOLD_PIPELINE_T_AT_T_MAX')
    if not 1.15 <= oof_summary['nll'] <= 1.27:
        flags.append('OOF_NLL_OUTSIDE_PLAUSIBLE_RANGE')
    state = {'seed': seed, 'config_id': config_id, 'config': cfg, 'T_star': pipe['T_star'], 'tau': tau,
             'model_files': model_files, 'rep': cfg['rep'], 'deploy': cfg['deploy'], 'n_members': len(pipe['deployed'])}
    C.write_json(d / 'STATE.json', state)
    op = (f"Family A {config_id}: {cfg['what']}. Occupation release p(x) = "
          + ("softmax(raw_HGB(x))" if tau is None and len(pipe['deployed']) == 1 else
             ("softmax(raw_HGB(x) / tau)" if len(pipe['deployed']) == 1 else "mean over 5 fold-committee members of softmax(raw_m(x))"))
          + f" on the {'compact native-category' if cfg['rep'] == 'cmp' else 'admitted 145-column'} coding of X; T* = {pipe['T_star']} "
          + "(first argmin of the pooled household 5-fold out-of-fold occupation NLL on MODEL_FIT, OHN1_crossfit partition)"
          + (f"; temperature tau = {tau:.6f} fitted on CALIBRATION occupation labels" if tau is not None else '; no calibrator applied')
          + '; decision = argmax; no SEX, no income label, no frozen in-sample layer')
    rec = {'kind': cfg['kind'], 'operation': op, 'mechanism': op, 'config': cfg,
           'roles_read': 'MODEL_FIT (features, occupation: training, household CV, cross-fitting); CALIBRATION (features, occupation: '
                         + ('deployed temperature calibrator and its diagnostics' if tau is not None else 'temperature calibrator diagnostic only, not applied')
                         + '); features of all six roles (release); no SEX anywhere',
           'exposure': ('release parameters: MODEL_FIT occupation labels (all 20,000 people; T* by household CV on MODEL_FIT)'
                        + ('; CALIBRATION occupation labels (temperature)' if tau is not None else '')
                        + '; no SEX, no income label, no MODEL_SELECT / ATTACK / DEV_REPORT label; no frozen layer (T_TREE, U0, TASK_REFIT '
                          'or Stage B correction) is used, so the cross-fitted MODEL_FIT predictions are out of fold for every label-using step'),
           'frozen_in_sample_layers_on_MODEL_FIT': [], 'crossfit_complete': True,
           'inputs_used': 'the six permitted raw fields through the admitted 145-column encoding X ('
                          + ('deterministic compact native-category re-coding' if cfg['rep'] == 'cmp' else 'used as is') + ')',
           'labels_used_at_training': 'MODEL_FIT occupation' + ('; CALIBRATION occupation (temperature)' if tau is not None else ''),
           'n_model_fits': len(pipe['deployed_inner_models']) + len(pipe['deployed']),
           'n_crossfit_fits': pipe['n_physical_fits'] - len(pipe['deployed_inner_models']) - len(pipe['deployed']),
           'inner_cv_tag': inner_tag(seed), 'crossfit_tag': C.CROSSFIT_TAG,
           'n_calibrator_fits': n_cal, 'n_physical_fits': pipe['n_physical_fits'], 'n_logical_fits': pipe['n_logical_fits'],
           'fit_count_note': ('n_model_fits = deployed pipeline (5 seed-specific inner-CV models to T_max + the deployed refit or 5 '
                              'committee refits at T*); n_crossfit_fits = the 5 cross-fitting pipelines (each 5 inner-CV models on MODEL_FIT '
                              'minus its fold + 1 refit at T*_k for refit configurations; committee members of cross-fitting pipelines are '
                              'the inner models truncated at T*_k, equal to fresh fits); calibrator fits include the recorded CALIBRATION / '
                              'OOF temperature diagnostics'),
           'fit_cpu_s': time.process_time() - t0, 'fit_wall_s': time.time() - w0, 'pipeline_cpu_s': pipe['cpu_s'],
           'T_star': pipe['T_star'], 'T_max': pipe['T_max'], 'T_k': {str(k): v for k, v in pipe['T_k'].items()},
           'curve_main_oof': _curve_summary(pipe['main_curve']),
           'curve_main_oof_min': {'nll': float(pipe['main_curve'][0].min()), 'brier_at_T_star': float(pipe['main_curve'][1, pipe['T_star'] - 1]),
                                  'accuracy_at_T_star': float(pipe['main_curve'][2, pipe['T_star'] - 1])},
           'crossfit_oof_MODEL_FIT': oof_summary, 'crossfit_oof_by_fold': oof_by_fold, 'committee_diagnostic_MODEL_FIT': comm,
           'temperature_diagnostics': {'CALIBRATION_full': tinfo, 'CALIBRATION_crossfit': tcf, 'MODEL_FIT_oof': toof,
                                       'applied': tau is not None,
                                       'note': 'CALIBRATION: temperature of the deployed raw output (in-sample for tau) and its household '
                                               '5-fold cross-fitted effect (temperature minus raw; negative = temperature helps); MODEL_FIT_oof: '
                                               'temperature of the cross-fitted MODEL_FIT predictions (internal folds)'},
           'model_files_sha256': model_files, 'fold_model_files_sha256': fold_files, 'state_sha256': C.sha(d / 'STATE.json'),
           'role_summary': role_summary, 'flags': flags, 'code_sha256': C.sha(__file__),
           'sklearn': __import__('sklearn').__version__, 'scipy': __import__('scipy').__version__}
    D.save_occupation(FAMILY, config_id, seed, occ, rec, crossfit={'MODEL_FIT__oof': oof, 'MODEL_SELECT__foldavg': fa, 'fold': folds})
    D.unit_stats(model_fits=rec['n_model_fits'], crossfit_fits=rec['n_crossfit_fits'], calibrator_fits=n_cal)
    print(json.dumps({'config': config_id, 'seed': seed, 'T_star': pipe['T_star'], 'T_k': pipe['T_k'], 'oof': oof_summary,
                      'committee': comm, 'tau_cal': tinfo, 'tau_cal_crossfit': {k: v for k, v in tcf.items() if k != 'fold_taus'},
                      'tau_oof': toof, 'flags': flags, 'fit_cpu_s': round(rec['fit_cpu_s'], 1)}, default=float))
    return rec


# ---------------------------------------------------------------- internal diagnostic (no release; MODEL_FIT only)
def diag(seed, ids):
    """Pooled out-of-fold occupation NLL curves on MODEL_FIT for diagnostic settings (seed-specific inner folds; 5 fits each)."""
    t0 = time.process_time()
    from . import devrun as D
    F = _load('MODEL_FIT', ('features', 'occupation'),
              purpose=f'family A diagnostic {"+".join(ids)} S{seed}: internal household CV scan on MODEL_FIT only (no release, no MODEL_SELECT)')
    names, y, unit = F['feature_names'], F['occupation'], F['unit']
    inner = C.household_folds(unit, 5, inner_tag(seed))
    allr = np.ones(len(y), dtype=bool)
    out = {}
    for sid in ids:
        cfg = DIAG_SETTINGS[sid]; tc = time.process_time()
        Xr = rep_matrix(cfg['rep'], F['X'], names)
        T, cur, ms = cv_select(cfg, seed, Xr, y, inner, allr)
        del ms
        out[sid] = {'setting': cfg, 'T_star': T, 'at_T_max': bool(T >= cfg['max_iter']), 'oof_nll_min': float(cur[0, T - 1]),
                    'oof_brier_at_T': float(cur[1, T - 1]), 'oof_accuracy_at_T': float(cur[2, T - 1]), 'curve': _curve_summary(cur),
                    'fits': 5, 'cpu_s': time.process_time() - tc}
        print(json.dumps({sid: {k: v for k, v in out[sid].items() if k != 'curve'}}, default=float), flush=True)
    rec = {'seed': seed, 'ids': list(ids), 'inner_cv_tag': inner_tag(seed), 'results': out, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'roles_read': 'MODEL_FIT (features, occupation); no SEX; no MODEL_SELECT', 'n_model_fits': 5 * len(ids),
           'cpu_s': time.process_time() - t0, 'code_sha256': C.sha(__file__), 'interpreter': sys.executable}
    C.write_json_x(C.RUNS / FAMILY / 'DIAG' / f'S{seed}' / f'DIAG_{"_".join(ids)}.json', rec)
    D.unit_stats(model_fits=5 * len(ids))
    return rec


# ---------------------------------------------------------------- deployment
def load_models(seed, config_id):
    d = run_dir(seed, config_id)
    rec = json.loads((d / 'OCC.json').read_text())
    st = json.loads((d / 'STATE.json').read_text())
    if C.sha(d / 'STATE.json') != rec['state_sha256']:
        raise PermissionError('STATE.json differs from its record')
    models = []
    for name, h in sorted(rec['model_files_sha256'].items()):
        if C.sha(d / name) != h or st['model_files'].get(name) != h:
            raise PermissionError(f'{name} differs from its recorded SHA-256')
        with open(d / name, 'rb') as f:
            models.append(pickle.load(f))
    return models, st


def predict(seed, config_id, X, names):
    """Recompute a saved configuration's occupation release (n, 5) from features alone (deterministic)."""
    models, st = load_models(seed, config_id)
    return release_prob(models, rep_matrix(st['rep'], X, names), st['tau'])


def verify(seed, config_id):
    from . import devrun as D
    occ, rec, _ = D.load_occupation(FAMILY, config_id, seed)
    res = {}
    for role in ROLES:
        Fr = _load(role, ('features',))
        p = predict(seed, config_id, Fr['X'], Fr['feature_names'])
        res[role] = {'max_abs_diff': float(np.max(np.abs(p - occ[role]))), 'bitwise': bool(np.array_equal(p, occ[role])),
                     'decisions_identical': bool(np.array_equal(p.argmax(1), occ[role].argmax(1)))}
    ok = all(v['bitwise'] for v in res.values())
    C.write_json(run_dir(seed, config_id) / 'VERIFY.json', {'ok': ok, 'roles': res, 'utc': C.now(), 'occ_sha256': rec['occ_sha256'],
                                                            'unit_id': os.environ.get(C.UNIT_ENV), 'code_sha256': C.sha(__file__),
                                                            'interpreter': sys.executable})
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'config': config_id, 'seed': seed, 'roles': res}))
    return ok


# ---------------------------------------------------------------- invented-data self-test (no real data)
def _fake_X(n, names, rng):
    X = np.zeros((n, len(names)), dtype=np.float32)
    X[:, 0] = rng.normal(size=n); X[:, 1] = rng.normal(size=n)
    for b in BLOCKS:
        idx = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        p = rng.dirichlet(np.ones(len(idx)) * 0.5)                     # skewed level frequencies (rare levels exist)
        X[np.arange(n), rng.choice(idx, size=n, p=p)] = 1.0
    return X


def _fake_labels(Xc, rng):
    """Invented 5-class labels depending on invented features (age, hours, an ordinal and two categorical codes)."""
    n = len(Xc)
    s = np.column_stack([0.8 * Xc[:, 1] + 0.15 * (Xc[:, 3] - 16), -0.6 * Xc[:, 1] + 0.3 * (Xc[:, 2] % 3), 0.3 * Xc[:, 0] - 0.2 * (Xc[:, 4] == 1),
                         -0.15 * (Xc[:, 3] - 16) + 0.5 * (Xc[:, 5] % 7 == 0), 0.4 * (Xc[:, 2] == 1) - 0.4 * Xc[:, 0] * Xc[:, 1]])
    p = softmax(s)
    u = rng.random(n)[:, None]
    return (u > np.cumsum(p, 1)).sum(1).astype(np.int64)


def selftest():
    rng = np.random.default_rng(20261018)
    names = list(C.teacher(0, 'occupation').feature_names)       # schema only (stored teacher payload; no person data)
    checks, info = {}, {}
    n = 2400
    X = _fake_X(n, names, rng)
    Xc = compact(X, names)
    checks['compact_shape'] = Xc.shape == (n, 6)
    checks['compact_codes_in_range'] = bool(Xc[:, 2].max() < 9 and Xc[:, 4].max() < 6 and Xc[:, 5].max() < 106 and Xc[:, 3].min() >= 1)
    try:
        from ihs import fam_h as FH
        checks['compact_equals_fam_h'] = bool(np.array_equal(Xc, FH.compact(X, names)))
    except Exception as e:                                     # pragma: no cover - recorded, not hidden
        checks['compact_equals_fam_h'] = False; info['fam_h_import_error'] = repr(e)[:200]
    bad = X.copy(); bad[0, 2:11] = 0
    try:
        compact(bad, names); checks['compact_rejects_incomplete_block'] = False
    except ValueError:
        checks['compact_rejects_incomplete_block'] = True
    Xpo = compact_po(X, names)
    pob_codes = {999.0 if nm.endswith('__OTHER__') else float(nm.split('=')[1]) for nm in names if nm.startswith('POBP=')}
    pob_idx = [i for i, nm in enumerate(names) if nm.startswith('POBP=')]
    expect_po = np.array([999.0 if names[i].endswith('__OTHER__') else float(names[i].split('=')[1]) for i in pob_idx])[X[:, pob_idx].argmax(1)]
    checks['compact_po_changes_only_pobp_to_codes'] = bool(np.array_equal(Xpo[:, :5], Xc[:, :5]) and set(np.unique(Xpo[:, 5])) <= pob_codes
                                                           and np.array_equal(Xpo[:, 5], expect_po))
    checks['rep_matrix_cmp_is_compact'] = bool(np.array_equal(rep_matrix('cmp', X, names), Xc))
    Xo = rep_matrix('cmp_ord', X, names)
    checks['cmp_ord_changes_only_cow_mar'] = bool(np.array_equal(Xo[:, [0, 1, 3, 5]], Xpo[:, [0, 1, 3, 5]]) and Xo[:, 2].min() >= 1
                                                  and Xo[:, 4].min() >= 1 and np.array_equal(np.unique(Xo[:, 2]).size, np.unique(Xpo[:, 2]).size))
    y = _fake_labels(Xc, rng)
    unit = np.repeat(np.arange(n // 2), 2)                     # invented two-person households
    folds = C.household_folds(unit, 5, 'OHN_A_selftest')
    checks['folds_household_separated'] = bool(all(len(np.unique(folds[unit == u])) == 1 for u in np.unique(unit)))
    checks['all_classes_present'] = bool(np.array_equal(np.unique(y), np.arange(K_OCC)))
    cfg = {'learner': 'hgb', 'rep': 'cmp', 'lr': 0.1, 'leaves': 8, 'msl': 20, 'l2': 1.0, 'max_iter': 40, 'calibrator': None, 'deploy': 'refit'}
    # truncation equals a fresh fit (bitwise), raw scores and released probabilities
    m40 = fit_model(cfg, Xc, y, 40, 0); m17 = fit_model(cfg, Xc, y, 17, 0)
    r_tr = raw_at(m40, Xc, 17); r_fr = raw_full(m17, Xc)
    checks['truncation_equals_fresh_fit_bitwise'] = bool(np.array_equal(r_tr, r_fr) and np.array_equal(softmax(r_tr), softmax(r_fr)))
    checks['softmax_matches_sklearn_proba'] = bool(np.allclose(softmax(r_fr), m17.predict_proba(Xc), atol=1e-12, rtol=0))
    rep_ok = True
    for rep in ('cmp_po', 'ohe'):
        mr_ = fit_model({**cfg, 'rep': rep}, rep_matrix(rep, X, names), y, 10, 0)
        cats = getattr(mr_, 'is_categorical_', None)
        expect = CAT_MASK[rep]
        rep_ok &= bool(np.isfinite(raw_full(mr_, rep_matrix(rep, X, names))).all())
        rep_ok &= bool((cats is None and expect is None) or (cats is not None and expect is not None and np.array_equal(cats, expect))
                       or (expect is None and cats is not None and not cats.any()))
    checks['alternative_representations_fit'] = rep_ok
    checks['cmp_mask_unchanged'] = bool(np.array_equal(m17.is_categorical_, CMP_CATEGORICAL))
    # the pipeline and an independent brute-force replay of the deployed and every cross-fitting pipeline
    inner = C.household_folds(unit, 5, 'OHN_A_selftest_inner')
    checks['inner_folds_household_separated'] = bool(all(len(np.unique(inner[unit == u])) == 1 for u in np.unique(unit)))
    checks['inner_partition_seed_specific'] = bool(not np.array_equal(C.household_folds(unit, 5, inner_tag(0)), C.household_folds(unit, 5, inner_tag(1))))
    Xs = compact(_fake_X(500, names, rng), names)
    pipe = run_pipeline(cfg, 0, Xc, y, folds, inner, extra={'S': Xs}, log=lambda s: None)
    K = pipe['K']

    def brute_T(rows):
        cur = np.zeros(40)
        for j in range(5):
            tr = rows & (inner != j); va = rows & (inner == j)
            mj = fit_model(cfg, Xc[tr], y[tr], 40, 0)                       # fresh inner model
            for t in range(1, 41):
                cur[t - 1] += metric_sums(y[va], softmax(raw_at(mj, Xc[va], t)))[0]
        return int(np.argmin(cur)) + 1

    checks['T_star_is_first_argmin_bruteforce'] = bool(pipe['T_star'] == brute_T(np.ones(n, dtype=bool)))
    oof_ok, tk_ok = True, True
    fa_brute = np.zeros((len(Xs), K_OCC)); oof_pipe = np.zeros((n, K_OCC))
    for k in range(K):
        oof_pipe[folds == k] = combine(pipe['oof_raw'][k])
        rows = folds != k
        tk = brute_T(rows)
        tk_ok &= (tk == pipe['T_k'][k])
        mr = fit_model(cfg, Xc[rows], y[rows], tk, 0)                       # fresh outer refit with exactly T*_k iterations
        oof_ok &= bool(np.array_equal(softmax(raw_full(mr, Xc[folds == k])), oof_pipe[folds == k]))
        fa_brute += softmax(raw_full(mr, Xs))
    fa_brute /= K
    fa_pipe = np.mean([combine(pipe['extra_raw']['S'][k]) for k in range(K)], axis=0)
    checks['crossfit_T_k_equal_fresh_nested_cv'] = bool(tk_ok)
    checks['crossfit_oof_equals_fresh_pipeline_bitwise'] = bool(oof_ok)
    checks['foldavg_equals_mean_of_fresh_pipelines'] = bool(np.allclose(fa_pipe, fa_brute, atol=1e-15, rtol=0))
    checks['deployed_equals_fresh_refit'] = bool(np.array_equal(raw_full(pipe['deployed'][0], Xs),
                                                                raw_full(fit_model(cfg, Xc, y, pipe['T_star'], 0), Xs)))
    checks['fit_counts'] = bool(pipe['n_physical_fits'] == 36 and pipe['n_logical_fits'] == {'deployed_pipeline': 6, 'crossfit_pipelines': 30})
    checks['committee_diagnostic_has_five_inner_members'] = bool(all(len(pipe['comm_raw'][k]) == 5 for k in range(K)))
    # committee variant: deployed members are fresh T* refits that equal the deployed pipeline's inner models truncated at T*;
    # cross-fitting committee members are the outer pipelines' inner models truncated at T*_k
    ccfg = {**cfg, 'deploy': 'committee'}
    cp = run_pipeline(ccfg, 0, Xc, y, folds, inner, extra={'S': Xs}, log=lambda s: None)
    checks['committee_members_equal_truncated_inner_models'] = bool(all(np.array_equal(raw_full(cp['deployed'][j], Xs), raw_at(cp['deployed_inner_models'][j], Xs, cp['T_star'])) for j in range(5)))
    checks['committee_counts_and_members'] = bool(all(len(cp['oof_raw'][k]) == 5 for k in range(K)) and cp['n_physical_fits'] == 35
                                                  and cp['n_logical_fits'] == {'deployed_pipeline': 10, 'crossfit_pipelines': 50} and cp['T_k'] == pipe['T_k'])
    # temperature recovers an invented temperature
    zr = rng.normal(scale=1.5, size=(20000, K_OCC))
    yt = (rng.random(20000)[:, None] > np.cumsum(softmax(zr, 1.3), 1)).sum(1).astype(np.int64)
    tau, tinfo = fit_temperature(zr, yt)
    checks['temperature_recovers_invented'] = bool(abs(tau - 1.3) < 0.05 and tinfo['nll_at_tau'] <= tinfo['nll_at_1'])
    tcf, _ = temperature_crossfit(zr[:3000], yt[:3000], np.repeat(np.arange(1500), 2))
    checks['temperature_crossfit_helps_when_miscalibrated'] = bool(tcf['nll_change'] < 0)
    # release validity and deterministic deployment from a pickled model (protocol 5) and its SHA
    p = release_prob(pipe['deployed'], Xs)
    checks['release_is_probability'] = bool(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-12, rtol=0))
    buf = io.BytesIO(); pickle.dump(pipe['deployed'][0], buf, protocol=5)
    m2 = pickle.loads(buf.getvalue())
    checks['pickled_model_reproduces_bitwise'] = bool(np.array_equal(release_prob([m2], Xs), p) and np.array_equal(release_prob([m2], Xs, 1.1), release_prob(pipe['deployed'], Xs, 1.1)))
    # loader guard: SEX and non-training labels are refused before any data access
    for wanted, role, key in ((('features', 'sex'), 'MODEL_FIT', 'guard_refuses_sex'), (('features', 'occupation'), 'MODEL_SELECT', 'guard_refuses_ms_labels')):
        try:
            _load(role, wanted); checks[key] = False
        except PermissionError:
            checks[key] = True
    # timing on invented data of the real size (20,000 x 6, five classes) for the cost projection
    nt = 20000
    Xt = compact(_fake_X(nt, names, rng), names); yt2 = _fake_labels(Xt, rng)
    for leaves, msl in ((8, 50), (31, 20)):
        tc = time.process_time()
        fit_model({**cfg, 'lr': 0.05, 'leaves': leaves, 'msl': msl}, Xt[:16000], yt2[:16000], 100, 0)
        info[f'cpu_s_per_100_iterations_16000_rows_leaves{leaves}'] = round(time.process_time() - tc, 3)
        mm = fit_model({**cfg, 'lr': 0.05, 'leaves': leaves, 'msl': msl}, Xt[:12000], yt2[:12000], 100, 0)
        tc = time.process_time(); staged_curves(mm, Xt[12000:], yt2[12000:])
        info[f'cpu_s_staged_curves_100_iterations_8000_rows_leaves{leaves}'] = round(time.process_time() - tc, 3)
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'info': info, 'T_star': pipe['T_star'], 'T_k': pipe['T_k'],
                      'sklearn': __import__('sklearn').__version__, 'interpreter': sys.executable}, default=float))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    f = sub.add_parser('fit'); f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True)
    v = sub.add_parser('verify'); v.add_argument('--seed', type=int, required=True); v.add_argument('--config', required=True)
    g = sub.add_parser('diag'); g.add_argument('--seed', type=int, required=True); g.add_argument('--ids', nargs='+', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    if a.cmd == 'diag':
        diag(a.seed, a.ids); return 0
    if a.cmd == 'verify':
        return 0 if verify(a.seed, a.config) else 1
    fit(a.seed, a.config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
