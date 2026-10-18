"""Family B: complementary occupation learners and a household-cross-fitted stack. PCRL overnight occupation-headroom campaign
(EXPLORATORY DEVELOPMENT on previously used records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged
25-64, civilian employed), development roles of the protected-tree replication. The service predicts an income class (2) and an
occupation group (5 official broad civilian groups) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column
encoding X). Each recipient receives only its own float64 probability vector and argmax decision. SEX is the dataset's recorded
demographic column; readers are separate evaluation classifiers that estimate SEX from released probability vectors. This module
is a TASK-ONLY occupation stage: no SEX enters any fit, input or output; MODEL_SELECT is never read here (only through
`python3 -m ohn.devrun score`). Card: results/pcrl_occupation_headroom_overnight_v1/roles/investigator_b/CARD.md.

Mechanism (initial design B1; registry LEARNERS / CONFIGS below)
  TREE   scikit-learn HistGradientBoosting multinomial learner on the compact re-coding of X (AGEP z, WKHP z, COW / MAR / POBP
         codes as native categories, SCHL ordinal value); 6 settings (leaves {15, 31, 63} x min leaf {20, 100}; lr 0.1, up to
         600 iterations, L2 1.0) and the tree count (checkpoints every 10) chosen by household-separated 5-fold inner CV.
  ADD    L2-penalized multinomial logistic regression on a deterministic label-free additive basis (cubic B-splines of AGEP and
         WKHP, the 13 frequent WKHP values, the one-hot blocks of X) with a fixed set of pairwise interactions; capacity S or L
         and C in {0.03, 0.15, 0.75} chosen by the same inner CV (6 settings).
  Final layers fitted on the inner out-of-fold logits of the same training set:
         B1_STACK   log-linear pool softmax(a_T z_T + a_A z_A), a >= 0;
         B1_SINGLE  the learner with the lower temperature-scaled out-of-fold NLL, softmax(tau z_L*).
  Full household cross-fitting: for each fold k of ohn.common.crossfit_folds the ENTIRE label-using pipeline (both learners'
  tuning with nested inner folds inside MODEL_FIT minus fold k, refits, learner choice, final layer on nested inner out-of-fold
  logits) is rerun without fold k -> MODEL_FIT__oof (fold k) and MODEL_SELECT__foldavg (average of the 5 fold pipelines).

Commands (through `python3 -m ohn.jobqueue run ...`):
  selftest                             invented-data checks (no real data)
  bases --seed s --learner TREE|ADD    tuning + nested cross-fitting of one base learner (MODEL_FIT occupation labels only)
  fit --seed s --config ID [--verify]  final layer, release for all six roles (ohn.devrun.save_occupation) + CROSSFIT.npz
  verify --seed s --config ID / verifyall
`predict(seed, config_id, X, names)` recomputes a saved configuration's occupation release from features alone.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from acs_replication import trees
from . import common as C

FAMILY = 'B'
K_OCC = 5
N_INNER = 5
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
CMP_CATEGORICAL = np.array([False, False, True, False, True, True])      # AGEP, WKHP, COW, SCHL(ordinal), MAR, POBP
WKHP_MIN_COUNT = 200                                                      # WKHP values with >= 200 MODEL_FIT people get an indicator

# ------------------------------------------------------------------ registry (fixed in CARD.md before any real-data fit)
LEARNERS = {
    'TREE_V1': {'kind': 'TREE', 'fixed': {'lr': 0.1, 'max_iter': 600, 'l2': 1.0, 'max_bins': 255, 'checkpoint': 10},
                'grid': tuple({'leaves': L, 'msl': m} for L in (15, 31, 63) for m in (20, 100))},
    'ADD_V1': {'kind': 'ADD', 'fixed': {'tol': 1e-6, 'max_iter': 5000, 'spline_knots': 6},
               'grid': tuple({'capacity': cap, 'C': c} for cap in ('S', 'L') for c in (0.03, 0.15, 0.75))},
    # revision B2 (CARD.md, appended before its fit): repaired tree encoding and grid; ADD's C grid extended below the B1 edge
    'TREE_V2': {'kind': 'TREE', 'fixed': {'lr': 0.05, 'max_iter': 500, 'l2': 1.0, 'max_bins': 255, 'checkpoint': 10, 'cat': 'P_ORD'},
                'grid': tuple({'leaves': L, 'msl': m} for L in (4, 8, 15) for m in (50, 200))},
    'ADD_V2': {'kind': 'ADD', 'fixed': {'tol': 1e-6, 'max_iter': 5000, 'spline_knots': 6},
               'grid': tuple({'capacity': cap, 'C': c} for cap in ('S', 'L') for c in (0.003, 0.01, 0.03))},
}
CONFIGS = {
    'B1_STACK': {'layer': 'pool', 'bases': ('TREE_V1', 'ADD_V1'), 'kind': 'task_model', 'design': 'B1',
                 'matched_control': 'B1_SINGLE',
                 'what': 'log-linear pool softmax(a_T z_T + a_A z_A), a >= 0, of the equally tuned HGB (TREE) and additive multinomial '
                         'logistic (ADD) learners, weights fitted on household-separated inner out-of-fold logits of MODEL_FIT'},
    'B1_SINGLE': {'layer': 'single', 'bases': ('TREE_V1', 'ADD_V1'), 'kind': 'control', 'design': 'B1',
                  'matched_control': 'itself (best equally tuned single learner)',
                  'what': 'the best equally tuned single learner (TREE or ADD, lower temperature-scaled inner out-of-fold NLL) with one '
                          'temperature fitted on its inner out-of-fold logits (same data as the stack layer)'},
    'B2_STACK': {'layer': 'pool', 'bases': ('TREE_V2', 'ADD_V2'), 'kind': 'task_model', 'design': 'B2',
                 'matched_control': 'B2_SINGLE',
                 'what': 'revision B2: log-linear pool softmax(a_T z_T + a_A z_A), a >= 0, of the repaired tree learner (HGB, POBP as '
                         'ordinal code, smaller trees, lr 0.05) and the additive learner with its C grid extended below the B1 edge, '
                         'weights fitted on household-separated inner out-of-fold logits of MODEL_FIT'},
    'B3_CALTR': {'layer': 'calpool', 'bases': ('TREE_V2', 'ADD_V2'), 'frozen': 'TASK_REFIT', 'kind': 'task_model', 'design': 'B3',
                 'matched_control': 'B2_STACK (the same pool without the CALIBRATION layer and without the frozen input)',
                 'what': 'revision B3: CALIBRATION-fitted log-linear layer softmax(c_1 s_B2(x) + c_2 z_TR(x)), c >= 0, on top of the B2 pool '
                         's_B2 = a_T z_T + a_A z_A (weights on MODEL_FIT inner OOF logits) and the frozen TASK_REFIT occupation scores z_TR '
                         '(task-only leaf corrections on the teacher routing; fitted on ALL MODEL_FIT people, out of sample on CALIBRATION)'},
    'B2_SINGLE': {'layer': 'single', 'bases': ('TREE_V2', 'ADD_V2'), 'kind': 'control', 'design': 'B2',
                  'matched_control': 'itself (best equally tuned single learner of B2)',
                  'what': 'revision B2 matched control: the better of the two repaired, equally tuned single learners (lower '
                          'temperature-scaled inner out-of-fold NLL) with one inner out-of-fold temperature'},
}


# ------------------------------------------------------------------ small numerics
def logsoftmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def center(z):
    z = np.asarray(z, dtype=np.float64)
    return z - z.mean(1, keepdims=True)


def person_nll(z, y):
    return -logsoftmax(z)[np.arange(len(y)), y]


def mean_nll(z, y):
    return float(person_nll(z, y).mean())


def probs(z):
    """Released probability vector of centered class scores (acs_replication.trees.probabilities convention)."""
    return trees.probabilities('occupation', np.asarray(z, dtype=np.float64))


def person_metrics(y, p):
    return C.task_vectors(np.asarray(y, dtype=np.int64), p, K_OCC)


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64); n = len(v)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=v - v.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def fit_temperature(z, y):
    """tau in [0.05, 20] minimizing the mean NLL of softmax(tau z) (bounded scalar search, the Ucal solver settings)."""
    from scipy.optimize import minimize_scalar
    z = np.asarray(z, dtype=np.float64)
    r = minimize_scalar(lambda t: mean_nll(t * z, y), method='bounded', bounds=(0.05, 20.0), options={'xatol': 1e-6, 'maxiter': 500})
    t = float(r.x)
    if not np.isfinite(t):
        raise FloatingPointError('temperature not finite')
    return t


def fit_pool(zs, y):
    """Log-linear (geometric) pool: a >= 0 minimizing the mean NLL of softmax(sum_i a_i z_i) (L-BFGS-B, analytic gradient)."""
    from scipy.optimize import minimize
    zs = [np.asarray(z, dtype=np.float64) for z in zs]
    n = len(y); onehot = np.zeros((n, K_OCC)); onehot[np.arange(n), y] = 1.0

    def f(a):
        s = sum(ai * zi for ai, zi in zip(a, zs))
        ls = logsoftmax(s); p = np.exp(ls)
        g = np.array([float(((p - onehot) * zi).sum(1).mean()) for zi in zs])
        return float(-ls[np.arange(n), y].mean()), g
    r = minimize(f, x0=np.full(len(zs), 0.5), jac=True, method='L-BFGS-B', bounds=[(0.0, 20.0)] * len(zs),
                 options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
    return [float(a) for a in r.x], {'nll': float(r.fun), 'converged': bool(r.success), 'n_iter': int(r.nit)}


def apply_pool(a, zs):
    return sum(float(ai) * np.asarray(zi, dtype=np.float64) for ai, zi in zip(a, zs))


def fit_linpool(zs, y):
    """Diagnostic linear pool p = b p_1 + (1 - b) p_2, b in [0, 1]."""
    from scipy.optimize import minimize_scalar
    p1, p2 = np.exp(logsoftmax(zs[0])), np.exp(logsoftmax(zs[1]))
    idx = np.arange(len(y))
    r = minimize_scalar(lambda b: float(-np.log(np.maximum(b * p1[idx, y] + (1 - b) * p2[idx, y], 1e-300)).mean()),
                        method='bounded', bounds=(0.0, 1.0), options={'xatol': 1e-7, 'maxiter': 500})
    return float(r.x)


def linpool_prob(b, zs):
    return b * np.exp(logsoftmax(zs[0])) + (1 - b) * np.exp(logsoftmax(zs[1]))


# ------------------------------------------------------------------ deterministic re-codings of the admitted X (label-free)
def onehot_block(X, names, b):
    idx = [i for i, n in enumerate(names) if n.startswith(b + '=')]
    Z = np.asarray(X[:, idx], dtype=np.float64)
    if not (np.all((Z == 0) | (Z == 1)) and np.all(Z.sum(1) == 1)):
        raise ValueError(f'{b}: one-hot block is not complete')
    return Z, [names[i] for i in idx]


def schl_values(level_names):
    """SCHL value of each one-hot level (SCHL=__OTHER__, the merged rare pre-grade-3 codes, maps to 3; as ihs.fam_h.compact)."""
    return np.array([3.0 if n.endswith('__OTHER__') else float(n.split('=')[1]) for n in level_names])


def compact(X, names):
    """AGEP z, WKHP z (as admitted), COW / MAR / POBP category codes, SCHL ordinal value (identical to ihs.fam_h.compact)."""
    names = tuple(map(str, names)); X = np.asarray(X)
    if names[:2] != ('AGEP', 'WKHP') or X.shape[1] != len(names):
        raise ValueError('unexpected admitted encoding')
    cols = [X[:, 0].astype(np.float64), X[:, 1].astype(np.float64)]
    for b in BLOCKS:
        Z, lv = onehot_block(X, names, b)
        code = Z.argmax(1)
        cols.append(schl_values(lv)[code] if b == 'SCHL' else code.astype(np.float64))
    return np.column_stack(cols)


def edu6_index(schl_val):
    v = np.asarray(schl_val)
    return np.select([v <= 15, v <= 17, v <= 19, v == 20, v == 21], [0, 1, 2, 3, 4], default=5)


def pob6_index(level_names):
    out = []
    for n in level_names:
        code = n.split('=')[1]
        if code == '__OTHER__':
            out.append(5); continue
        c = int(code)
        out.append(0 if c <= 56 else 1 if c < 100 else 2 if c < 200 else 3 if c < 300 else 4 if c < 400 else 5)
    return np.array(out)


def indicator(idx, k):
    Z = np.zeros((len(idx), k)); Z[np.arange(len(idx)), idx] = 1.0
    return Z


def prod(A, B):
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    return (A[:, :, None] * B[:, None, :]).reshape(len(A), -1)


def make_featurizer(X, names, n_knots=6):
    """Label-free, frozen basis parameters from the MODEL_FIT features: uniform cubic B-spline knots for AGEP z and WKHP z over
    their MODEL_FIT ranges and the list of WKHP values held by >= WKHP_MIN_COUNT people."""
    from sklearn.preprocessing import SplineTransformer
    a = np.asarray(X[:, 0], dtype=np.float64)[:, None]; w = np.asarray(X[:, 1], dtype=np.float64)[:, None]
    sa = SplineTransformer(n_knots=n_knots, degree=3, knots='uniform', extrapolation='constant', include_bias=False).fit(a)
    sw = SplineTransformer(n_knots=n_knots, degree=3, knots='uniform', extrapolation='constant', include_bias=False).fit(w)
    u, c = np.unique(np.asarray(X[:, 1]), return_counts=True)
    vals = [float(v) for v in u[c >= WKHP_MIN_COUNT]]
    return {'agep_spline': sa, 'wkhp_spline': sw, 'wkhp_values': vals, 'names': list(map(str, names)), 'n_knots': n_knots}


def add_design(X, names, fz, capacity):
    """Deterministic additive basis Phi(x) (CSR, float64) of capacity 'S' or 'L' (CARD.md section 2)."""
    import scipy.sparse as sp
    names = tuple(map(str, names))
    if list(names) != fz['names']:
        raise ValueError('feature names differ from the featurizer')
    X = np.asarray(X)
    a = X[:, 0].astype(np.float64); w = X[:, 1].astype(np.float64)
    oh, lv = {}, {}
    for b in BLOCKS:
        oh[b], lv[b] = onehot_block(X, names, b)
    edu = indicator(edu6_index(schl_values(lv['SCHL'])[oh['SCHL'].argmax(1)]), 6)
    pob = indicator(pob6_index(lv['POBP'])[oh['POBP'].argmax(1)], 6)
    As = fz['agep_spline'].transform(a[:, None]); Ws = fz['wkhp_spline'].transform(w[:, None])
    Wi = (w[:, None] == np.asarray(fz['wkhp_values'], dtype=np.float64)[None, :]).astype(np.float64)
    cow, schl, mar = oh['COW'], oh['SCHL'], oh['MAR']
    blocks = [As, Ws, Wi, a[:, None], w[:, None], cow, schl, mar, oh['POBP'],
              prod(edu, cow), prod(edu, As), prod(cow, w), prod(edu, w), prod(As, w), prod(mar, a), prod(pob, edu)]
    if capacity == 'L':
        blocks += [prod(schl, cow), prod(edu, Ws), prod(cow, Ws), prod(As, Ws), prod(pob, cow), prod(pob, a), prod(pob, w),
                   prod(mar, edu), prod(mar, cow)]
    elif capacity != 'S':
        raise ValueError(capacity)
    return sp.csr_matrix(np.column_stack(blocks))


# ------------------------------------------------------------------ base learners
def hgb_seed(seed):
    return int(trees.seed_for('OHN_B_hgb', seed) % 2 ** 31)


_SPEC_CAT = object()
CAT_MASKS = {'ALL_NATIVE': CMP_CATEGORICAL,                                   # TREE_V1: COW, MAR, POBP native categories
             'P_ORD': np.array([False, False, True, False, True, False]),     # POBP as its ordinal code; COW, MAR native
             'NONE': None}


def make_hgb(setting, fixed, seed, max_iter, cat=_SPEC_CAT):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if cat is _SPEC_CAT:
        cat = CAT_MASKS[fixed.get('cat', 'ALL_NATIVE')]
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=fixed['lr'], max_iter=int(max_iter),
                                          max_leaf_nodes=int(setting['leaves']), min_samples_leaf=int(setting['msl']),
                                          l2_regularization=float(fixed['l2']), max_bins=int(fixed['max_bins']), early_stopping=False,
                                          categorical_features=cat, random_state=hgb_seed(seed))


def cv_curve(Xm, cat, setting, fixed, y, rows, fr, seed):
    """Diagnostic: mean out-of-fold NLL at every checkpoint of one HGB setting on design Xm (categorical mask cat or None)."""
    ck = int(fixed['checkpoint']); M = int(fixed['max_iter']); n_ck = M // ck
    Z = np.zeros((n_ck, len(rows), K_OCC))
    for j in range(N_INNER):
        te_mask = fr == j
        tr, te = rows[~te_mask], rows[te_mask]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = make_hgb(setting, fixed, seed, M, cat).fit(Xm[tr], y[tr])
        _check_classes(m)
        c = 0
        for it, z in enumerate(m.staged_decision_function(Xm[te]), start=1):
            if it % ck == 0:
                Z[c][te_mask] = center(z); c += 1
    return np.array([mean_nll(Z[c], y[rows]) for c in range(n_ck)])


DIAG_VARIANTS = {'P_ORD': 'compact coding, POBP as its ordinal code (COW, MAR native categories)',
                 'ALL_ORD': 'compact coding, no native categories (COW, MAR, POBP ordinal codes)',
                 'OHE': 'the admitted one-hot X (145 columns, no native categories; the teacher input)'}


def diag_tree(seed, variants=tuple(DIAG_VARIANTS)):
    """MODEL_FIT-only diagnosis of TREE_V1's early overfitting: inner-CV curves (the main pipeline's seed-specific inner folds)
    of TREE_V1's chosen main setting (15 leaves, min leaf 100, lr 0.1) under alternative encodings. No MODEL_SELECT, no SEX."""
    from . import devrun as D
    t0 = time.process_time()
    F = C.load_role('MODEL_FIT', ('features', 'occupation'), purpose=f'family B TREE_V1 overfitting diagnosis seed {seed}: inner-CV curves under alternative encodings (occupation labels; no SEX)')
    X, names, y, unit = F['X'], F['feature_names'], F['occupation'], F['unit']
    rows = np.arange(len(y)); fr = C.household_folds(unit, N_INNER, f'OHN_B_inner_S{seed}')
    Xc = compact(X, names)
    designs = {'P_ORD': (Xc, np.array([False, False, True, False, True, False])), 'ALL_ORD': (Xc, None),
               'OHE': (np.asarray(X, dtype=np.float64), None)}
    g = {'leaves': 15, 'msl': 100}; fx = {'lr': 0.1, 'max_iter': 600, 'l2': 1.0, 'max_bins': 255, 'checkpoint': 10}
    out = {'seed': seed, 'setting': g, 'fixed': fx, 'inner_tag': f'OHN_B_inner_S{seed}', 'variants': {}}
    for v in variants:
        ts = time.process_time()
        Xm, cat = designs[v]
        curve = cv_curve(Xm, cat, g, fx, y, rows, fr, seed)
        ci = int(np.argmin(curve))
        out['variants'][v] = {'what': DIAG_VARIANTS[v], 'best_n': (ci + 1) * 10, 'best_oof_nll': float(curve[ci]),
                              'nll_at_600': float(curve[-1]), 'curve_every_50': [float(c) for c in curve[4::5]],
                              'curve_first_10': [float(c) for c in curve[:10]], 'cpu_s': time.process_time() - ts}
        print(json.dumps({v: out['variants'][v]}, default=float), flush=True)
    out['n_model_fits'] = N_INNER * len(variants); out['cpu_s'] = time.process_time() - t0; out['utc'] = C.now()
    out['unit_id'] = os.environ.get(C.UNIT_ENV)
    d = C.RUNS / FAMILY / 'DIAG'; d.mkdir(parents=True, exist_ok=True, mode=0o700)
    C.write_json(d / f'DIAG_TREE_S{seed}_{os.environ.get(C.UNIT_ENV, "local")}.json', out)
    D.unit_stats(n_model_fits=out['n_model_fits'])


def make_lr(C_, fixed, warm):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=float(C_), tol=float(fixed['tol']), max_iter=int(fixed['max_iter']), warm_start=bool(warm))


def _check_classes(m):
    if list(m.classes_) != list(range(K_OCC)):
        raise ValueError(f'training set lacks an occupation class: {m.classes_}')


def tune_tree(spec, Xc, y, rows, fr, seed):
    """Inner CV on `rows` (inner fold ids fr): curve of mean OOF NLL at every checkpoint for every setting; returns the chosen
    setting, tree count, OOF logits at the choice, the table and the number of fits."""
    fx = spec['fixed']; ck = int(fx['checkpoint']); M = int(fx['max_iter']); n_ck = M // ck
    best, table, fits = None, [], 0
    for gi, g in enumerate(spec['grid']):
        Z = np.zeros((n_ck, len(rows), K_OCC))
        for j in range(N_INNER):
            te_mask = fr == j
            tr, te = rows[~te_mask], rows[te_mask]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = make_hgb(g, fx, seed, M).fit(Xc[tr], y[tr])
            fits += 1; _check_classes(m)
            c = 0
            for it, z in enumerate(m.staged_decision_function(Xc[te]), start=1):
                if it % ck == 0:
                    Z[c][te_mask] = center(z); c += 1
        curve = np.array([mean_nll(Z[c], y[rows]) for c in range(n_ck)])
        ci = int(np.argmin(curve))
        table.append({'setting': g, 'best_n': (ci + 1) * ck, 'oof_nll': float(curve[ci]), 'nll_at_max': float(curve[-1]),
                      'curve_every_50': [float(v) for v in curve[4::5]], 'at_edge': bool(ci == n_ck - 1)})
        if best is None or curve[ci] < best['nll']:
            best = {'gi': gi, 'setting': g, 'n': (ci + 1) * ck, 'nll': float(curve[ci]), 'oof': Z[ci].copy()}
        del Z
    return best, table, fits


def refit_tree(spec, best, Xc, y, rows, seed):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = make_hgb(best['setting'], spec['fixed'], seed, best['n']).fit(Xc[rows], y[rows])
    _check_classes(m)
    return m


def tune_add(spec, Phi, y, rows, fr):
    """Inner CV for the additive learner: per capacity, warm-started regularization path over increasing C."""
    fx = spec['fixed']; grid = spec['grid']
    oofs = {gi: np.zeros((len(rows), K_OCC)) for gi in range(len(grid))}
    iters = {gi: [] for gi in range(len(grid))}
    fits = 0
    for cap in dict.fromkeys(g['capacity'] for g in grid):
        path = sorted((gi for gi, g in enumerate(grid) if g['capacity'] == cap), key=lambda gi: grid[gi]['C'])
        for j in range(N_INNER):
            te_mask = fr == j
            tr, te = rows[~te_mask], rows[te_mask]
            lr = make_lr(grid[path[0]]['C'], fx, warm=True)
            for gi in path:
                lr.set_params(C=float(grid[gi]['C']))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    lr.fit(Phi[cap][tr], y[tr])
                fits += 1; _check_classes(lr)
                oofs[gi][te_mask] = center(lr.decision_function(Phi[cap][te]))
                iters[gi].append(int(np.max(lr.n_iter_)))
    table, best = [], None
    for gi, g in enumerate(grid):
        nll = mean_nll(oofs[gi], y[rows])
        table.append({'setting': g, 'oof_nll': nll, 'lbfgs_iters': iters[gi], 'hit_max_iter': bool(max(iters[gi]) >= fx['max_iter'])})
        if best is None or nll < best['nll']:
            best = {'gi': gi, 'setting': g, 'nll': nll, 'oof': oofs[gi]}
    cs = sorted({g['C'] for g in grid})
    best['at_edge'] = bool(best['setting']['C'] in (cs[0], cs[-1]))
    return best, table, fits


def refit_add(spec, best, Phi, y, rows):
    lr = make_lr(best['setting']['C'], spec['fixed'], warm=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr.fit(Phi[best['setting']['capacity']][rows], y[rows])
    _check_classes(lr)
    return lr, int(np.max(lr.n_iter_))


def frozen_logits(seed, arm, X, names):
    """Centered occupation scores of a frozen historical checkpoint on the teacher routing (label-free): T_TREE, or TASK_REFIT
    (stored E20 task-only leaf corrections). Both were fitted on ALL MODEL_FIT people (in-sample there)."""
    te = C.teacher(seed, 'occupation')
    theta = None if arm == 'T_TREE' else C.stage_b_theta(seed, arm, 'E20')[1]
    return center(te.raw_from_leaves(te.route(X, names), theta))


def base_logits_from(obj, X, names):
    """Centered class scores of a saved base learner object for any feature matrix (deployment path)."""
    if obj['kind'] == 'TREE':
        return center(obj['model'].decision_function(compact(X, names)))
    if obj['kind'] == 'ADD':
        return center(obj['model'].decision_function(add_design(X, names, obj['featurizer'], obj['capacity'])))
    raise KeyError(obj['kind'])


# ------------------------------------------------------------------ nested pipeline of one base learner (array level)
def outer_sets(unit, outer, seed):
    """('main', all rows, inner folds) and ('out{k}', rows outside fold k, inner folds inside them); asserts household separation."""
    n = len(unit)
    sets = [('main', np.arange(n), C.household_folds(unit, N_INNER, f'OHN_B_inner_S{seed}'))]
    for k in range(int(outer.max()) + 1):
        rows = np.where(outer != k)[0]
        held = np.where(outer == k)[0]
        if np.intersect1d(unit[rows], unit[held]).size or np.any(outer[rows] == k):
            raise AssertionError(f'outer fold {k}: training rows are not household-separated from the held-out fold')
        sets.append((f'out{k}', rows, C.household_folds(unit[rows], N_INNER, f'OHN_B_inner_S{seed}_out{k}')))
    return sets


def run_base(spec, X, names, y, unit, outer, X_extra, seed, log=print):
    """Tune + refit the learner on the main training set and on each outer training set (nested inner folds). Returns the main
    model object, arrays (main OOF logits; per outer fold: inner OOF on its training rows, held-out fold logits, and the fold
    model's logits on each label-free extra feature matrix in X_extra, e.g. {'ms': MODEL_SELECT X, 'cal': CALIBRATION X}) and the
    record."""
    t0 = time.process_time()
    kind = spec['kind']
    arrays, rec, fits = {}, {'sets': {}}, 0
    if kind == 'TREE':
        Xc = compact(X, names); Xc_ex = {t: compact(Xe, names) for t, Xe in X_extra.items()}; fz = None
    else:
        fz = make_featurizer(X, names, spec['fixed']['spline_knots'])
        caps = sorted({g['capacity'] for g in spec['grid']})
        Phi = {cap: add_design(X, names, fz, cap) for cap in caps}
        Phi_ex = {t: {cap: add_design(Xe, names, fz, cap) for cap in caps} for t, Xe in X_extra.items()}
        rec['design_columns'] = {cap: int(Phi[cap].shape[1]) for cap in caps}
        rec['wkhp_values'] = fz['wkhp_values']
    main_obj = None
    for name, rows, fr in outer_sets(unit, outer, seed):
        ts = time.process_time()
        if kind == 'TREE':
            best, table, nf = tune_tree(spec, Xc, y, rows, fr, seed)
            model = refit_tree(spec, best, Xc, y, rows, seed); nf += 1
            obj = {'kind': 'TREE', 'model': model, 'setting': best['setting'], 'n_iter': best['n']}
            z_rows = lambda idx: center(model.decision_function(Xc[idx]))
            z_ex = lambda t: center(model.decision_function(Xc_ex[t]))
            refit_info = {'n_iter': best['n']}
        else:
            best, table, nf = tune_add(spec, Phi, y, rows, fr)
            model, it = refit_add(spec, best, Phi, y, rows); nf += 1
            cap = best['setting']['capacity']
            obj = {'kind': 'ADD', 'model': model, 'setting': best['setting'], 'capacity': cap, 'featurizer': fz}
            z_rows = lambda idx, cap=cap, model=model: center(model.decision_function(Phi[cap][idx]))
            z_ex = lambda t, cap=cap, model=model: center(model.decision_function(Phi_ex[t][cap]))
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
        rec['sets'][name] = {'n_train': int(len(rows)), 'chosen': best['setting'], 'chosen_n_iter': best.get('n'),
                             'chosen_oof_nll': best['nll'], 'chosen_at_edge': bool(best.get('at_edge', False) or
                                                                                   (kind == 'TREE' and best['n'] == spec['fixed']['max_iter'])),
                             'table': table, 'refit': refit_info, 'fits': nf, 'cpu_s': round(time.process_time() - ts, 2)}
        log(json.dumps({'set': name, 'chosen': best['setting'], 'n': best.get('n'), 'oof_nll': round(best['nll'], 6),
                        'fits': nf, 'cpu_s': round(time.process_time() - ts, 1)}, default=float))
        sys.stdout.flush()
    rec['n_model_fits_main'] = rec['sets']['main']['fits']
    rec['n_crossfit_fits'] = fits - rec['sets']['main']['fits']
    rec['cpu_s'] = time.process_time() - t0
    return main_obj, arrays, rec


# ------------------------------------------------------------------ final layers (array level)
def fit_layer(cfg, oofs, y):
    """Final layer on inner OOF logits {learner_id: (n, 5)}: returns params and a small record."""
    ids = list(cfg['bases'])
    if cfg['layer'] == 'pool':
        a, info = fit_pool([oofs[i] for i in ids], y)
        return {'layer': 'pool', 'bases': ids, 'weights': a}, {**info, 'n_fits': 1}
    if cfg['layer'] == 'single':
        cand = {}
        for i in ids:
            t = fit_temperature(oofs[i], y)
            cand[i] = {'tau': t, 'oof_nll_tempered': mean_nll(t * oofs[i], y), 'oof_nll_raw': mean_nll(oofs[i], y)}
        best = min(ids, key=lambda i: (cand[i]['oof_nll_tempered'], ids.index(i)))
        return {'layer': 'single', 'bases': [best], 'tau': cand[best]['tau'], 'chosen': best}, {'candidates': cand, 'n_fits': len(ids)}
    raise KeyError(cfg['layer'])


def apply_layer(params, zs):
    """Class scores of the final layer from the base logits {learner_id: (n, 5)}."""
    if params['layer'] == 'pool':
        return apply_pool(params['weights'], [zs[i] for i in params['bases']])
    if params['layer'] == 'single':
        return float(params['tau']) * zs[params['bases'][0]]
    if params['layer'] == 'calpool':
        c = params['cal_weights']
        return float(c[0]) * apply_pool(params['weights'], [zs[i] for i in params['bases']]) + float(c[1]) * zs['FROZEN_' + params['frozen']]
    raise KeyError(params['layer'])


def fit_calpool(cfg, oofs, y, cal_zs, cal_frozen, ycal):
    """Two-stage layer: the pool weights a on MODEL_FIT inner OOF logits (as B2_STACK), then c >= 0 for [s_pool, z_frozen] on
    CALIBRATION (out of sample there for the base learners and for the frozen checkpoint)."""
    ids = list(cfg['bases'])
    a, info_a = fit_pool([oofs[i] for i in ids], y)
    s_cal = apply_pool(a, [cal_zs[i] for i in ids])
    c, info_c = fit_pool([s_cal, cal_frozen], ycal)
    return ({'layer': 'calpool', 'bases': ids, 'weights': a, 'frozen': cfg['frozen'], 'cal_weights': c},
            {'pool': info_a, 'cal': info_c, 'n_fits': 2, 'cal_nll_in_sample': info_c['nll'],
             'cal_nll_pool_only_unrefitted': mean_nll(s_cal, ycal)})


def crossfit_release(cfg, arrays, y, outer):
    """Nested cross-fitted predictions: per outer fold k the final layer is fitted on that fold pipeline's inner OOF logits
    (its own training rows only) and applied to its held-out-fold and MODEL_SELECT logits."""
    n = len(y); oof = np.zeros((n, K_OCC)); ms = []; per_fold = {}
    for k in range(int(outer.max()) + 1):
        rows = np.where(outer != k)[0]; held = np.where(outer == k)[0]
        params, info = fit_layer(cfg, {i: arrays[i][f'out{k}__inner_oof'] for i in cfg['bases']}, y[rows])
        oof[held] = probs(apply_layer(params, {i: arrays[i][f'out{k}__fold'] for i in cfg['bases']}))
        ms.append(probs(apply_layer(params, {i: arrays[i][f'out{k}__ms'] for i in cfg['bases']})))
        per_fold[k] = {'params': params, 'info': info}
    return oof, np.mean(ms, axis=0), per_fold


def crossfit_diagnostics(arrays, ids, y, unit, outer):
    """Fully cross-fitted MODEL_FIT comparison of raw / tempered single learners, the log-linear pool, a linear pool and a
    multinomial-logistic stacker (each layer refitted per outer fold on nested inner OOF logits). Paired differences vs the
    cross-fitted B1_SINGLE variant (positive = worse than SINGLE), household-clustered SEs."""
    from sklearn.linear_model import LogisticRegression
    n = len(y); variants = {}
    names = [f'{i}_raw' for i in ids] + [f'{i}_temp' for i in ids] + ['POOL', 'SINGLE', 'LINPOOL', 'MLR_STACK']
    for v in names:
        variants[v] = np.zeros((n, K_OCC))
    chosen = {}
    for k in range(int(outer.max()) + 1):
        rows = np.where(outer != k)[0]; held = np.where(outer == k)[0]
        inner = {i: arrays[i][f'out{k}__inner_oof'] for i in ids}; fold = {i: arrays[i][f'out{k}__fold'] for i in ids}
        yk = y[rows]
        for i in ids:
            variants[f'{i}_raw'][held] = probs(fold[i])
            variants[f'{i}_temp'][held] = probs(fit_temperature(inner[i], yk) * fold[i])
        a, _ = fit_pool([inner[i] for i in ids], yk)
        variants['POOL'][held] = probs(apply_pool(a, [fold[i] for i in ids]))
        sp_, si = fit_layer({'layer': 'single', 'bases': ids}, inner, yk)
        variants['SINGLE'][held] = probs(apply_layer(sp_, fold)); chosen[k] = sp_['chosen']
        b = fit_linpool([inner[i] for i in ids], yk)
        variants['LINPOOL'][held] = linpool_prob(b, [fold[i] for i in ids])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mlr = LogisticRegression(C=1.0, tol=1e-8, max_iter=5000).fit(np.column_stack([inner[i] for i in ids]), yk)
        variants['MLR_STACK'][held] = probs(center(mlr.decision_function(np.column_stack([fold[i] for i in ids]))))
    met = {v: person_metrics(y, p) for v, p in variants.items()}
    out = {'chosen_single_per_fold': chosen}
    for v in names:
        r = {m: float(met[v][m].mean()) for m in ('nll', 'brier', 'accuracy')}
        for m in ('nll', 'brier', 'accuracy'):
            d = (met[v][m] - met['SINGLE'][m]) if m != 'accuracy' else (met['SINGLE'][m] - met[v][m])
            r[f'{m}_vs_SINGLE'] = float(d.mean()); r[f'{m}_vs_SINGLE_se'] = cluster_se(d, unit)
        out[v] = r
    return out


# ------------------------------------------------------------------ storage and real-data steps
def base_dir(seed, learner_id, root=None):
    return Path(root or C.RUNS / FAMILY) / 'BASES' / learner_id / f'S{seed}'


def state_path(seed, config_id, root=None):
    return Path(root or C.RUNS / FAMILY) / config_id / f'S{seed}' / 'STATE.json'


def save_base(seed, learner_id, obj, arrays, rec, outer, root=None):
    d = base_dir(seed, learner_id, root)
    if (d / 'BASE.json').exists():
        raise PermissionError(f'base {learner_id}/S{seed} already exists (append-only; new learner id for a change)')
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(d / 'MODEL.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=5)
    with open(d / 'PRED.npz', 'wb') as f:
        np.savez(f, fold=np.asarray(outer, dtype=np.int64), **arrays)
    rec = {**rec, 'learner_id': learner_id, 'seed': seed, 'utc': C.now(), 'interpreter': sys.executable,
           'python': sys.version.split()[0], 'numpy': np.__version__, 'sklearn': __import__('sklearn').__version__,
           'model_sha256': C.sha(d / 'MODEL.pkl'), 'pred_sha256': C.sha(d / 'PRED.npz'), 'unit_id': os.environ.get(C.UNIT_ENV)}
    (d / 'BASE.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + '\n')
    return d


_OBJ_CACHE = {}


def load_base(seed, learner_id, root=None, arrays=True):
    d = base_dir(seed, learner_id, root)
    rec = json.loads((d / 'BASE.json').read_text())
    key = (str(d), rec['model_sha256'])
    if key not in _OBJ_CACHE:
        if C.sha(d / 'MODEL.pkl') != rec['model_sha256']:
            raise PermissionError(f'base model {learner_id}/S{seed} differs from its record')
        with open(d / 'MODEL.pkl', 'rb') as f:
            _OBJ_CACHE[key] = pickle.load(f)
    arr = None
    if arrays:
        if C.sha(d / 'PRED.npz') != rec['pred_sha256']:
            raise PermissionError(f'base predictions {learner_id}/S{seed} differ from their record')
        with np.load(d / 'PRED.npz') as z:
            arr = {k: z[k].copy() for k in z.files}
    return _OBJ_CACHE[key], arr, rec


def bases(seed, learner_id):
    """Real-data base step: MODEL_FIT features + occupation labels (logged), MODEL_SELECT features (label-free)."""
    from . import devrun as D
    spec = LEARNERS[learner_id]
    F = C.load_role('MODEL_FIT', ('features', 'occupation'), purpose=f'family B base learner {learner_id} seed {seed}: inner CV, refit, nested cross-fitting (occupation labels; no SEX)')
    M = C.load_role('MODEL_SELECT', ('features',))
    Cal = C.load_role('CALIBRATION', ('features',))
    names = F['feature_names']
    if not (tuple(M['feature_names']) == tuple(names) == tuple(Cal['feature_names'])):
        raise ValueError('feature names differ between roles')
    outer = C.crossfit_folds(F['unit'])
    obj, arrays, rec = run_base(spec, F['X'], names, F['occupation'], F['unit'], outer, {'ms': M['X'], 'cal': Cal['X']}, seed)
    rec.update({'spec': spec, 'roles_read': 'MODEL_FIT (features, occupation labels); MODEL_SELECT and CALIBRATION features (label-free); no SEX, no income',
                'crossfit_tag': C.CROSSFIT_TAG, 'inner_tag': f'OHN_B_inner_S{seed}[_out<k>]', 'hgb_random_state': hgb_seed(seed)})
    d = save_base(seed, learner_id, obj, arrays, rec, outer)
    D.unit_stats(n_model_fits=rec['n_model_fits_main'], n_crossfit_fits=rec['n_crossfit_fits'])
    print(json.dumps({'base': learner_id, 'seed': seed, 'dir': str(d.relative_to(C.ROOT)), 'cpu_s': round(rec['cpu_s'], 1),
                      'main': {k: rec['sets']['main'][k] for k in ('chosen', 'chosen_n_iter', 'chosen_oof_nll', 'chosen_at_edge')},
                      'outer_chosen': {s: (v['chosen'], v['chosen_n_iter'], round(v['chosen_oof_nll'], 6)) for s, v in rec['sets'].items() if s != 'main'}},
                     default=float))


def predict(seed, config_id, X, names, root=None):
    """Deterministic deployment: occupation probability vectors from features and the saved, hash-recorded parameters."""
    st = json.loads(state_path(seed, config_id, root).read_text())
    zs = {}
    for lid in st['params']['bases']:
        obj, _, rec = load_base(seed, lid, root, arrays=False)
        if rec['model_sha256'] != st['base_model_sha256'][lid]:
            raise PermissionError(f'{lid}: base model differs from the one recorded in STATE.json')
        zs[lid] = base_logits_from(obj, X, names)
    if st['params'].get('frozen'):
        zs['FROZEN_' + st['params']['frozen']] = frozen_logits(seed, st['params']['frozen'], X, names)
    return probs(apply_layer(st['params'], zs))


def calpool_pipeline(seed, cfg, objs, arrays, y, outer, X_fit, names):
    """B3: CALIBRATION-fitted layer over the B2 pool and a frozen checkpoint. Returns params, info, cross-fitted arrays, per-fold
    layers and a record (frozen-score check against the frozen bank; household cross-fitted CALIBRATION check)."""
    ids = list(cfg['bases']); fz = cfg['frozen']; key = 'FROZEN_' + fz
    Cal = C.load_role('CALIBRATION', ('features', 'occupation'), purpose=f'family B {cfg["design"]} seed {seed}: CALIBRATION-fitted layer weights with the frozen {fz} input and its household cross-fitted CALIBRATION check (occupation labels; no SEX)')
    M = C.load_role('MODEL_SELECT', ('features',))
    ycal, ucal, Xcal = Cal['occupation'], Cal['unit'], Cal['X']
    zf = {'MF': frozen_logits(seed, fz, X_fit, names), 'MS': frozen_logits(seed, fz, M['X'], names), 'CAL': frozen_logits(seed, fz, Xcal, names)}
    bank, _ = C.frozen_bank(seed, roles=('MODEL_SELECT',))
    chk = float(np.max(np.abs(probs(zf['MS']) - bank[fz]['MODEL_SELECT']['occupation'])))
    if chk > 1e-12:
        raise AssertionError(f'recomputed {fz} scores differ from the frozen bank ({chk})')
    cal_main = {i: base_logits_from(objs[i], Xcal, names) for i in ids}
    params, info = fit_calpool(cfg, {i: arrays[i]['main__oof'] for i in ids}, y, cal_main, zf['CAL'], ycal)
    n = len(y); oof = np.zeros((n, K_OCC)); ms = []; per_fold = {}
    for k in range(int(outer.max()) + 1):
        rows = np.where(outer != k)[0]; held = np.where(outer == k)[0]
        pk, ik = fit_calpool(cfg, {i: arrays[i][f'out{k}__inner_oof'] for i in ids}, y[rows], {i: arrays[i][f'out{k}__cal'] for i in ids},
                             zf['CAL'], ycal)
        oof[held] = probs(apply_layer(pk, {**{i: arrays[i][f'out{k}__fold'] for i in ids}, key: zf['MF'][held]}))
        ms.append(probs(apply_layer(pk, {**{i: arrays[i][f'out{k}__ms'] for i in ids}, key: zf['MS']})))
        per_fold[k] = {'params': pk, 'info': ik}
    # household cross-fitted CALIBRATION check: the CALIBRATION-fitted weights evaluated out of fold on CALIBRATION
    s_cal = apply_pool(params['weights'], [cal_main[i] for i in ids])
    cf = C.household_folds(ucal, 5, f'OHN_B_calcheck_S{seed}')
    var = {'POOL_ONLY': s_cal.copy(), 'POOL_TEMP': np.zeros_like(s_cal), 'CALPOOL': np.zeros_like(s_cal), 'FROZEN_ONLY': zf['CAL'].copy()}
    for j in range(5):
        tr, te = cf != j, cf == j
        var['POOL_TEMP'][te] = fit_temperature(s_cal[tr], ycal[tr]) * s_cal[te]
        c, _ = fit_pool([s_cal[tr], zf['CAL'][tr]], ycal[tr])
        var['CALPOOL'][te] = c[0] * s_cal[te] + c[1] * zf['CAL'][te]
    met = {v: person_metrics(ycal, probs(z)) for v, z in var.items()}
    check = {}
    for v in var:
        r = {m: float(met[v][m].mean()) for m in ('nll', 'brier', 'accuracy')}
        for m in ('nll', 'brier', 'accuracy'):
            d = (met['CALPOOL'][m] - met[v][m]) if m != 'accuracy' else (met[v][m] - met['CALPOOL'][m])
            r[f'CALPOOL_minus_this_{m}'] = float(d.mean()); r[f'CALPOOL_minus_this_{m}_se'] = cluster_se(d, ucal)
        check[v] = r
    calrec = {'frozen': fz, 'frozen_bank_check_max_abs_prob_diff_MODEL_SELECT': chk, 'calibration_people': int(len(ycal)),
              'calibration_households': int(len(np.unique(ucal))), 'household_crossfit_calibration_check': check,
              'calcheck_tag': f'OHN_B_calcheck_S{seed}', 'n_calcheck_fits': 10}
    return params, info, oof, np.mean(ms, axis=0), per_fold, calrec


def fit(seed, config_id, verify_after=False):
    t0 = time.process_time()
    from . import devrun as D
    cfg = CONFIGS[config_id]
    if (D.occ_dir(FAMILY, config_id, seed) / 'OCC.npz').exists():
        raise PermissionError(f'{config_id}/S{seed} already released (append-only)')
    F = C.load_role('MODEL_FIT', ('features', 'occupation'), purpose=f'family B {config_id} seed {seed}: final layer on inner out-of-fold logits, cross-fitted diagnostics (occupation labels; no SEX)')
    y, unit = F['occupation'], F['unit']
    outer = C.crossfit_folds(unit)
    objs, arrays, brec = {}, {}, {}
    for lid in cfg['bases']:
        objs[lid], arrays[lid], brec[lid] = load_base(seed, lid)
        if not np.array_equal(arrays[lid]['fold'], outer):
            raise AssertionError('stored base folds differ from the registered cross-fitting folds')
    calrec = None
    if cfg['layer'] == 'calpool':
        params, info, oof, foldavg, per_fold, calrec = calpool_pipeline(seed, cfg, objs, arrays, y, outer, F['X'], F['feature_names'])
    else:
        params, info = fit_layer(cfg, {i: arrays[i]['main__oof'] for i in cfg['bases']}, y)
        oof, foldavg, per_fold = crossfit_release(cfg, arrays, y, outer)
    used = params['bases']
    st = {'seed': seed, 'config_id': config_id, 'config': cfg, 'params': params, 'layer_info': info,
          'base_model_sha256': {i: brec[i]['model_sha256'] for i in cfg['bases']},
          'base_pred_sha256': {i: brec[i]['pred_sha256'] for i in cfg['bases']}, 'crossfit_layers': per_fold}
    sp = state_path(seed, config_id); sp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    C.write_json(sp, st)
    occ, summary = {}, {}
    for role in C.ROLES6:
        R = C.load_role(role, ('features',))
        p = predict(seed, config_id, R['X'], R['feature_names'])
        occ[role] = p
        summary[role] = {'n': int(len(p)), 'class_share_decisions': [float(np.mean(p.argmax(1) == c)) for c in range(K_OCC)],
                         'mean_max_prob': float(p.max(1).mean()), 'distinct_vectors': int(np.unique(p, axis=0).shape[0])}
    # internal consistency: the deployed MODEL_FIT scores equal the stored base logits path (sanity, not parity)
    diag = crossfit_diagnostics(arrays, list(cfg['bases']), y, unit, outer)
    vf = person_metrics(y, oof)
    main_oof_nll = {i: mean_nll(arrays[i]['main__oof'], y) for i in cfg['bases']}
    n_base_main = sum(brec[i]['n_model_fits_main'] for i in cfg['bases'])
    n_base_cf = sum(brec[i]['n_crossfit_fits'] for i in cfg['bases'])
    n_layer_cf = sum(v['info']['n_fits'] for v in per_fold.values())
    rec = {'kind': cfg['kind'], 'design': cfg['design'], 'matched_control': cfg['matched_control'],
           'operation': f"Family B {config_id}: {cfg['what']}. " + (
               f"Deployed p(x) = softmax({params['weights'][0]:.6f} z_{used[0]}(x) + {params['weights'][1]:.6f} z_{used[1]}(x))"
               if params['layer'] == 'pool' else
               f"Deployed p(x) = softmax({params['cal_weights'][0]:.6f} * ({params['weights'][0]:.6f} z_{used[0]}(x) + {params['weights'][1]:.6f} z_{used[1]}(x)) + {params['cal_weights'][1]:.6f} z_{params['frozen']}(x))"
               if params['layer'] == 'calpool' else f"Deployed p(x) = softmax({params['tau']:.6f} z_{used[0]}(x)) (chosen single learner {used[0]})")
               + '; z = centered class scores of the base learners refitted on all MODEL_FIT at their inner-CV settings'
               + ('; z_TASK_REFIT = centered stored task-only leaf-corrected teacher scores (frozen, fitted on all MODEL_FIT)' if params['layer'] == 'calpool' else ''),
           'calibration_layer': calrec,
           'final_layer': params, 'final_layer_info': info,
           'base_learners': {i: {'chosen_main': brec[i]['sets']['main']['chosen'], 'chosen_n_iter_main': brec[i]['sets']['main']['chosen_n_iter'],
                                 'main_inner_oof_nll_at_choice': brec[i]['sets']['main']['chosen_oof_nll'],
                                 'chosen_at_edge_main': brec[i]['sets']['main']['chosen_at_edge'],
                                 'outer_choices': {s: [v['chosen'], v['chosen_n_iter']] for s, v in brec[i]['sets'].items() if s != 'main'},
                                 'model_sha256': brec[i]['model_sha256'], 'pred_sha256': brec[i]['pred_sha256'],
                                 'base_cpu_s': brec[i]['cpu_s'], 'base_unit_id': brec[i].get('unit_id')} for i in cfg['bases']},
           'main_inner_oof_nll_raw': main_oof_nll,
           'inputs_used': 'the six permitted raw fields through the admitted 145-column X and its deterministic re-codings (compact codes; additive basis)',
           'roles_read': ('MODEL_FIT (features, occupation labels: base tuning, refits, final layer, nested cross-fitting); features of all six roles (label-free); no SEX, no income, no CALIBRATION labels'
                          if calrec is None else
                          'MODEL_FIT (features, occupation labels: base tuning, refits, pool weights, nested cross-fitting); CALIBRATION (features, occupation labels: the 2 layer weights and their household cross-fitted check); features of all six roles (label-free); stored TASK_REFIT parameters (teacher routing + E20 leaf corrections; label-free use); no SEX, no income'),
           'exposure': ('deployed base learners and final layer fitted on all MODEL_FIT (in-sample there); out of sample on CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT. '
                        'Cross-fitting: every label-using layer (both learners tuning + refit, learner choice, final layer on nested inner OOF logits) refitted per OHN1_crossfit fold; '
                        'no frozen checkpoint (T_TREE / U0 / TASK_REFIT / Stage B) used; label-free steps (X dictionary, spline knots, WKHP indicator list) from all MODEL_FIT features'
                        if calrec is None else
                        'FROZEN INPUT: TASK_REFIT (task-only leaf corrections on the teacher routing, replication Stage B E20) was fitted on ALL MODEL_FIT people, so the MODEL_FIT__oof '
                        'predictions are NOT fully out of sample (its component is in-sample on every MODEL_FIT person; the B2 learners and pool are refitted per fold; the CALIBRATION weights '
                        'are refitted per fold from that fold pipeline CALIBRATION scores). The CALIBRATION layer is out of sample on CALIBRATION for every input; its honest '
                        'CALIBRATION value is the household cross-fitted check in calibration_layer. Deployed base learners in-sample on MODEL_FIT; every component out of sample on '
                        'MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT; CALIBRATION labels used by the 2 layer weights'),
           'n_model_fits': n_base_main, 'n_calibrator_fits': int(info['n_fits']), 'n_crossfit_fits': n_base_cf + n_layer_cf,
           'fit_counts_note': 'base-learner fits (n_model_fits: main pipeline incl. inner CV; n_crossfit_fits: the 5 outer fold pipelines incl. their inner CV, plus the per-fold final layers) are SHARED by the configurations of the same design and seed (one bases unit per learner; B3 reuses the B2 bases); count them once',
           'n_diagnostic_layer_fits': 5 * (3 * len(cfg['bases']) + 3),
           'n_diagnostic_layer_fits_note': 'cross-fitted MODEL_FIT diagnostics only (per outer fold: one temperature per learner, the pool, the single-learner choice with one temperature per learner, the linear pool, the multinomial-logistic stacker); not released',
           'fit_cpu_s': time.process_time() - t0, 'base_fit_cpu_s_shared': {i: brec[i]['cpu_s'] for i in cfg['bases']},
           'crossfit_model_fit_oof': {m: float(vf[m].mean()) for m in ('nll', 'brier', 'accuracy')},
           'crossfit_diagnostics_model_fit': diag, 'role_summary': summary, 'state_file': str(sp.relative_to(C.ROOT)),
           'state_sha256': C.sha(sp)}
    d = D.save_occupation(FAMILY, config_id, seed, occ, rec, crossfit={'MODEL_FIT__oof': oof, 'MODEL_SELECT__foldavg': foldavg, 'fold': outer})
    D.unit_stats(n_calibrator_fits=int(info['n_fits']), n_crossfit_fits=n_layer_cf)
    print(json.dumps({'config': config_id, 'seed': seed, 'dir': str(d.relative_to(C.ROOT)), 'params': params,
                      'crossfit_oof': rec['crossfit_model_fit_oof'],
                      'diag': {v: {m: round(diag[v][m], 6) for m in ('nll', 'nll_vs_SINGLE', 'nll_vs_SINGLE_se', 'brier', 'accuracy')}
                               for v in diag if v != 'chosen_single_per_fold'}}, default=float))
    if verify_after:
        return verify(seed, config_id)
    return True


def verify(seed, config_id):
    """Deployment parity: predict from each role's features alone against the saved release (bitwise; label-free)."""
    from . import devrun as D
    occ, rec, _ = D.load_occupation(FAMILY, config_id, seed)
    st_sha = C.sha(state_path(seed, config_id))
    res = {}
    for role in C.ROLES6:
        R = C.load_role(role, ('features',))
        p = predict(seed, config_id, R['X'], R['feature_names'])
        res[role] = {'max_abs_diff': float(np.max(np.abs(p - occ[role]))), 'bitwise': bool(np.array_equal(p, occ[role])),
                     'decisions_identical': bool(np.array_equal(p.argmax(1), occ[role].argmax(1)))}
    ok = all(v['bitwise'] for v in res.values()) and st_sha == rec['state_sha256']
    out = {'ok': ok, 'state_sha256_matches_record': st_sha == rec['state_sha256'], 'roles': res, 'utc': C.now(),
           'occ_sha256': rec['occ_sha256'], 'interpreter': sys.executable}
    p = D.occ_dir(FAMILY, config_id, seed) / 'VERIFY.json'
    C.write_json(p, out)
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'config': config_id, 'seed': seed}))
    return ok


def verify_all():
    out = {}
    for d in sorted((C.RUNS / FAMILY).glob('*/S*/OCC.json')):
        cfg, seed = d.parent.parent.name, int(d.parent.name[1:])
        out[f'{cfg}/S{seed}'] = verify(seed, cfg)
    print(json.dumps({'verify_all': 'PASS' if all(out.values()) else 'FAIL', 'n': len(out), 'items': out}))
    return all(out.values())


# ------------------------------------------------------------------ invented-data self-test (no real data)
def _fake_X(n, names, rng):
    X = np.zeros((n, len(names)), dtype=np.float32)
    X[:, 0] = rng.choice(np.linspace(-1.8, 1.65, 40), size=n).astype(np.float32)
    X[:, 1] = rng.choice(np.r_[np.linspace(-3.5, 5.2, 30), np.full(30, -0.0569)], size=n).astype(np.float32)
    for b in BLOCKS:
        idx = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        X[np.arange(n), rng.choice(idx, size=n)] = 1.0
    return X


def selftest():
    import tempfile
    rng = np.random.default_rng(20261018)
    names = list(C.teacher(0, 'occupation').feature_names)       # schema only (stored teacher payload; no person data)
    checks = {}
    n = 2400
    X = _fake_X(n, names, rng)
    Xc = compact(X, names)
    checks['compact_shape_and_codes'] = bool(Xc.shape == (n, 6) and Xc[:, 2].max() < 9 and Xc[:, 4].max() < 6 and Xc[:, 5].max() < 106)
    bad = X.copy(); bad[0, 2:11] = 0
    try:
        compact(bad, names); checks['compact_rejects_incomplete_block'] = False
    except ValueError:
        checks['compact_rejects_incomplete_block'] = True
    fz = make_featurizer(X, names)
    PhiS, PhiL = add_design(X, names, fz, 'S'), add_design(X, names, fz, 'L')
    nwi = len(fz['wkhp_values'])
    main_cols = 7 + 7 + nwi + 2 + 9 + 22 + 6 + 106
    s_cols = main_cols + 54 + 42 + 9 + 6 + 7 + 6 + 36
    l_cols = s_cols + 198 + 42 + 63 + 49 + 54 + 6 + 6 + 36 + 54
    checks['add_design_shapes'] = bool(PhiS.shape == (n, s_cols) and PhiL.shape == (n, l_cols) and np.isfinite(PhiL.data).all())
    checks['add_design_label_free_deterministic'] = bool((add_design(X, names, fz, 'L') != PhiL).nnz == 0)
    checks['spline_partition'] = bool(np.all(PhiS[:, :7].toarray() >= -1e-12))
    checks['edu6_pob6_maps'] = bool(list(edu6_index(np.array([1, 15, 16, 17, 18, 19, 20, 21, 22, 24]))) == [0, 0, 1, 1, 2, 2, 3, 4, 5, 5]
                                    and list(pob6_index(['POBP=001', 'POBP=056', 'POBP=072', 'POBP=110', 'POBP=207', 'POBP=303', 'POBP=440', 'POBP=__OTHER__'])) == [0, 0, 1, 2, 3, 4, 5, 5])
    # invented labels (5 classes) depending on a few invented signals
    edu = edu6_index(Xc[:, 3])
    logits = np.column_stack([0.8 * (edu >= 4) + 0.3 * Xc[:, 0], 0.5 * (Xc[:, 1] < -0.5) - 0.2 * (edu >= 4),
                              0.3 * (Xc[:, 2] == 1) + 0.2 * Xc[:, 0] * Xc[:, 1], 0.6 * (edu <= 1) - 0.3 * (Xc[:, 4] == 2), 0.2 * np.sin(Xc[:, 5])])
    P = np.exp(logits - logits.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > P.cumsum(1)).sum(1).clip(0, 4).astype(np.int64)
    unit = np.repeat(np.arange(n // 2), 2)
    outer = C.crossfit_folds(unit)
    sets = outer_sets(unit, outer, 0)
    checks['nested_sets_household_separated'] = bool(len(sets) == 6 and all(np.intersect1d(unit[r], unit[outer == int(s[3:])]).size == 0 for s, r, _ in sets[1:]))
    checks['inner_folds_household_separated'] = bool(all(all(len(np.unique(fr[unit[r] == u])) == 1 for u in np.unique(unit[r])[:100]) for _, r, fr in sets))
    # staged HGB predictions at n equal a refit with max_iter = n
    fx = {'lr': 0.1, 'max_iter': 30, 'l2': 1.0, 'max_bins': 255, 'checkpoint': 10}
    g = {'leaves': 15, 'msl': 20}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m30 = make_hgb(g, fx, 0, 30).fit(Xc[:1800], y[:1800]); m20 = make_hgb(g, fx, 0, 20).fit(Xc[:1800], y[:1800])
    st20 = [z for i, z in enumerate(m30.staged_decision_function(Xc[1800:]), start=1) if i == 20][0]
    checks['hgb_staged_equals_refit'] = bool(np.array_equal(st20, m20.decision_function(Xc[1800:])))
    checks['tree_v2_pobp_ordinal_mask'] = bool(np.array_equal(make_hgb(g, LEARNERS['TREE_V2']['fixed'], 0, 5).categorical_features,
                                                              np.array([False, False, True, False, True, False]))
                                               and np.array_equal(make_hgb(g, LEARNERS['TREE_V1']['fixed'], 0, 5).categorical_features, CMP_CATEGORICAL))
    # pool recovers known weights; temperature agrees with a grid
    z1 = rng.normal(size=(20000, 5)); z2 = rng.normal(size=(20000, 5))
    s = 1.3 * z1 + 0.4 * z2
    Ps = np.exp(s - s.max(1, keepdims=True)); Ps /= Ps.sum(1, keepdims=True)
    yy = (rng.random(20000)[:, None] > Ps.cumsum(1)).sum(1).clip(0, 4)
    a, info = fit_pool([z1, z2], yy)
    checks['pool_recovers_weights'] = bool(abs(a[0] - 1.3) < 0.08 and abs(a[1] - 0.4) < 0.08 and info['converged'])
    a0, _ = fit_pool([z1, -z2], yy)
    checks['pool_nonnegative'] = bool(a0[1] == 0.0 and a0[0] > 0)
    t = fit_temperature(z1, yy)
    grid = np.linspace(0.05, 3, 2951)
    tg = grid[int(np.argmin([mean_nll(gg * z1, yy) for gg in grid]))]
    checks['temperature_matches_grid'] = bool(abs(t - tg) < 2e-3)
    b = fit_linpool([z1, z2], yy)
    checks['linpool_in_unit_interval'] = bool(0.0 <= b <= 1.0)
    # full mini pipeline (both learners, both configurations) on invented data, written to a temporary directory
    tiny = {'TREE_V1': {'kind': 'TREE', 'fixed': fx, 'grid': ({'leaves': 7, 'msl': 20}, {'leaves': 15, 'msl': 40})},
            'ADD_V1': {'kind': 'ADD', 'fixed': {'tol': 1e-6, 'max_iter': 2000, 'spline_knots': 6},
                       'grid': ({'capacity': 'S', 'C': 0.05}, {'capacity': 'S', 'C': 0.5})},
            'TREE_V2': {'kind': 'TREE', 'fixed': {**fx, 'cat': 'P_ORD'}, 'grid': ({'leaves': 4, 'msl': 50}, {'leaves': 8, 'msl': 50})},
            'ADD_V2': {'kind': 'ADD', 'fixed': {'tol': 1e-6, 'max_iter': 2000, 'spline_knots': 6},
                       'grid': ({'capacity': 'S', 'C': 0.01}, {'capacity': 'L', 'C': 0.01})}}
    Xms = _fake_X(500, names, rng)
    with tempfile.TemporaryDirectory() as tmp:
        arrays, brec = {}, {}
        for lid, spec in tiny.items():
            obj, arr, rec = run_base(spec, X, names, y, unit, outer, {'ms': Xms, 'cal': Xms[:300]}, 0, log=lambda s: None)
            save_base(0, lid, obj, arr, {**rec, 'spec': spec}, outer, root=tmp)
            _, arrays[lid], brec[lid] = load_base(0, lid, root=tmp)
        checks['base_arrays_shapes'] = bool(all(arrays[l]['main__oof'].shape == (n, 5) and arrays[l]['out0__ms'].shape == (500, 5)
                                                and arrays[l]['out4__cal'].shape == (300, 5)
                                                and arrays[l]['out0__inner_oof'].shape == ((outer != 0).sum(), 5) for l in tiny))
        checks['base_fit_counts'] = bool(all(brec[l]['n_model_fits_main'] == 2 * N_INNER + 1 and brec[l]['n_crossfit_fits'] == 5 * (2 * N_INNER + 1) for l in tiny))
        checks['tree_v2_model_uses_pobp_ordinal'] = bool(np.array_equal(load_base(0, 'TREE_V2', root=tmp, arrays=False)[0]['model'].is_categorical_,
                                                                        np.array([False, False, True, False, True, False])))
        ok_release = True; ok_parity = True; ok_cf = True
        Xcal_inv = Xms[:300]
        ycal_inv = (rng.random(300)[:, None] > np.full((300, 5), 0.2).cumsum(1)).sum(1).clip(0, 4).astype(np.int64)
        for cid, cfg in CONFIGS.items():
            if cfg['layer'] == 'calpool':
                objs_inv = {i: load_base(0, i, root=tmp, arrays=False)[0] for i in cfg['bases']}
                cal_zs = {i: base_logits_from(objs_inv[i], Xcal_inv, names) for i in cfg['bases']}
                zfr = frozen_logits(0, cfg['frozen'], Xcal_inv, names)
                params, info = fit_calpool(cfg, {i: arrays[i]['main__oof'] for i in cfg['bases']}, y, cal_zs, zfr, ycal_inv)
                pool_only = apply_pool(params['weights'], [cal_zs[i] for i in cfg['bases']])
                same = apply_layer({**params, 'cal_weights': [1.0, 0.0]}, {**cal_zs, 'FROZEN_' + cfg['frozen']: zfr})
                checks['calpool_reduces_to_pool'] = bool(np.array_equal(pool_only, same) and len(params['cal_weights']) == 2
                                                         and min(params['cal_weights']) >= 0.0)
                checks['frozen_scores_finite'] = bool(np.isfinite(zfr).all() and zfr.shape == (300, 5) and np.allclose(zfr.mean(1), 0, atol=1e-9))
            else:
                params, info = fit_layer(cfg, {i: arrays[i]['main__oof'] for i in cfg['bases']}, y)
            sp = state_path(0, cid, tmp); sp.parent.mkdir(parents=True, exist_ok=True)
            C.write_json(sp, {'params': params, 'base_model_sha256': {i: brec[i]['model_sha256'] for i in cfg['bases']}})
            for XX in (X, Xms):
                p = predict(0, cid, XX, names, root=tmp)
                ok_release &= bool(p.shape == (len(XX), 5) and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-12))
                ok_parity &= bool(np.array_equal(p, predict(0, cid, XX, names, root=tmp)))
            if cfg['layer'] == 'calpool':
                continue                     # its cross-fitting uses the real CALIBRATION / frozen-bank loaders (checked at fit time)
            oof, fa, per_fold = crossfit_release(cfg, arrays, y, outer)
            ok_cf &= bool(oof.shape == (n, 5) and fa.shape == (500, 5) and np.allclose(oof.sum(1), 1) and np.allclose(fa.sum(1), 1))
            # main OOF logits are genuinely out of fold: the deployed in-sample NLL is lower than the OOF NLL for the flexible tree
        checks['release_probability_contract'] = ok_release
        checks['predict_deterministic'] = ok_parity
        checks['crossfit_arrays_valid'] = ok_cf
        diag = crossfit_diagnostics(arrays, ['TREE_V1', 'ADD_V1'], y, unit, outer)
        checks['diagnostics_complete'] = bool(all(v in diag for v in ('TREE_V1_raw', 'ADD_V1_temp', 'POOL', 'SINGLE', 'LINPOOL', 'MLR_STACK'))
                                              and abs(diag['SINGLE']['nll_vs_SINGLE']) == 0.0)
        obj_t, _, _ = load_base(0, 'TREE_V1', root=tmp, arrays=False)
        ins = mean_nll(base_logits_from(obj_t, X, names), y)
        checks['oof_is_out_of_sample'] = bool(mean_nll(arrays['TREE_V1']['main__oof'], y) > ins)
        try:
            save_base(0, 'TREE_V1', obj_t, arrays['TREE_V1'], brec['TREE_V1'], outer, root=tmp); checks['base_append_only'] = False
        except PermissionError:
            checks['base_append_only'] = True
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'pool_weights_invented': a,
                      'design_columns': {'S': int(PhiS.shape[1]), 'L': int(PhiL.shape[1])}}, default=float))
    return ok, checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    b = sub.add_parser('bases'); b.add_argument('--seed', type=int, required=True); b.add_argument('--learner', required=True, choices=tuple(LEARNERS))
    f = sub.add_parser('fit'); f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True, choices=tuple(CONFIGS))
    f.add_argument('--verify', action='store_true')
    v = sub.add_parser('verify'); v.add_argument('--seed', type=int, required=True); v.add_argument('--config', required=True)
    sub.add_parser('verifyall')
    dg = sub.add_parser('diag_tree'); dg.add_argument('--seed', type=int, required=True); dg.add_argument('--variants', default=','.join(DIAG_VARIANTS))
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        ok, _ = selftest(); return 0 if ok else 1
    if a.cmd == 'bases':
        bases(a.seed, a.learner); return 0
    if a.cmd == 'verify':
        return 0 if verify(a.seed, a.config) else 1
    if a.cmd == 'verifyall':
        return 0 if verify_all() else 1
    if a.cmd == 'diag_tree':
        diag_tree(a.seed, tuple(a.variants.split(','))); return 0
    return 0 if fit(a.seed, a.config, a.verify) else 1


if __name__ == '__main__':
    raise SystemExit(main())
