"""Family C of the overnight occupation-headroom campaign: residual task learning for the OCCUPATION output (investigator C).

EXPLORATORY DEVELOPMENT on previously used records (public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for
Florida); not confirmation. The service predicts an occupation group (5 official broad civilian groups) from the six permitted raw
fields (admitted 145-column encoding X); the occupation recipient receives only its float64 probability vector and argmax decision.
No fit in this module reads SEX; MODEL_SELECT labels are never read here (scoring goes through `python3 -m ohn.devrun score`).
Mechanism card: results/pcrl_occupation_headroom_overnight_v1/roles/investigator_c/CARD.md.

Operations (all deployed from X only; z_T = centred raw score of the frozen T_TREE occupation teacher, exact numpy routing):
  residual (C1_RB_D1, C1_RB_D2, CC_CONT_D4):  p = softmax(z_T + alpha * g),  g = xgboost depth-d trees fitted to the teacher's
      residual on MODEL_FIT (multi:softprob, base_margin = z_T), M rounds by household 5-fold CV on MODEL_FIT (registered folds),
      alpha in [0, 2] by NLL on CALIBRATION (calibrator).  Deployment evaluates the saved trees in numpy (float32 x < float32
      threshold -> yes child; float64 accumulation of the float32 leaf values).
  vscal (CC_VSCAL): p = softmax(w * z_T + b), 9 free parameters by NLL on CALIBRATION (calibration-only control).
Household cross-fitting check: every label-using step of this family (CV for M, g, alpha) is refitted without each registered
MODEL_FIT fold; the frozen teacher z_T is NOT refitted (it saw every MODEL_FIT household), so the out-of-fold predictions are out of
fold for the correction layer only. Usage:
  python -m ohn.fam_c selftest                      (invented data only)
  python -m ohn.fam_c fit --seed S --config ID      (real data; label reads gated by ohn.common.load_role)
  python -m ohn.fam_c verify --seed S --config ID   (bitwise reproduction of the saved release on all six roles from features)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C

FAMILY = 'C'
K = 5
M_MAX = 800
ALPHA_BOUNDS = (0.0, 2.0)
XGB_FIXED = {'objective': 'multi:softprob', 'num_class': K, 'eta': 0.05, 'lambda': 3.0, 'alpha': 0.0, 'min_child_weight': 1.0,
             'gamma': 0.0, 'subsample': 1.0, 'colsample_bytree': 1.0, 'tree_method': 'hist', 'max_bin': 256, 'nthread': 1,
             'seed': 0, 'eval_metric': 'mlogloss', 'verbosity': 0}
CONFIGS = {
    'C1_RB_D1': {'kind': 'task_model', 'op': 'residual', 'depth': 1, 'base': 'T_TREE', 'version': 'initial',
                 'role': 'capacity setting 1: additive main-effect residual (boosted stumps)'},
    'C1_RB_D2': {'kind': 'task_model', 'op': 'residual', 'depth': 2, 'base': 'T_TREE', 'version': 'initial',
                 'role': 'capacity setting 2: pairwise-interaction residual (depth-2 trees)'},
    'CC_CONT_D4': {'kind': 'control', 'op': 'residual', 'depth': 4, 'base': 'T_TREE', 'version': 'initial',
                   'role': 'matched continuation control: teacher depth, no capacity limit, same budget, CV and alpha rules'},
    'CC_VSCAL': {'kind': 'control', 'op': 'vscal', 'base': 'T_TREE', 'version': 'initial',
                 'role': 'calibration-only control: vector scaling of the teacher margin on CALIBRATION'},
    # revision 1 (card section appended 2026-10-18 ~05:58Z, before these fits): residual trained against CROSS-FITTED margins
    'C2_XF_D1': {'kind': 'task_model', 'op': 'residual_xf', 'depth': 1, 'base': 'T_TREE', 'version': 'revision1',
                 'role': 'revision 1, capacity setting 1: main-effect residual trained on cross-fitted teacher margins'},
    'C2_XF_D2': {'kind': 'task_model', 'op': 'residual_xf', 'depth': 2, 'base': 'T_TREE', 'version': 'revision1',
                 'role': 'revision 1, capacity setting 2: pairwise residual trained on cross-fitted teacher margins'},
    'CC_XF_CONT_D4': {'kind': 'control', 'op': 'residual_xf', 'depth': 4, 'base': 'T_TREE', 'version': 'revision1',
                      'role': 'matched continuation control of revision 1: teacher depth, cross-fitted margins, same budget and rules'},
    # revision 2 (card section appended 2026-10-18 06:11Z, before these fits): ordered block codes as correction inputs
    'C3_XC_D1': {'kind': 'task_model', 'op': 'residual_xf', 'depth': 1, 'base': 'T_TREE', 'version': 'revision2', 'inputs': 'X+codes',
                 'role': 'revision 2, capacity setting 1: main-effect residual on X plus ordered block codes, cross-fitted margins'},
    'C3_XC_D2': {'kind': 'task_model', 'op': 'residual_xf', 'depth': 2, 'base': 'T_TREE', 'version': 'revision2', 'inputs': 'X+codes',
                 'role': 'revision 2, capacity setting 2: pairwise residual on X plus ordered block codes, cross-fitted margins'},
}
_DRYRUN_INVENTED_LABELS = False                 # set only by dryrun_full (invented labels cannot reproduce the real teacher)
CAL_CHECK_TAG = 'OHN1_famC_calcheck'          # household 5-fold split of CALIBRATION for the cross-fitted held-out check


# ---------------------------------------------------------------- numerics
def probs(logits):
    from acs_replication import trees
    return trees.probabilities('occupation', logits)


def centre(z):
    z = np.asarray(z, dtype=np.float64)
    return z - z.mean(1, keepdims=True)


def lse(L, axis=-1):
    m = L.max(axis, keepdims=True)
    return (m + np.log(np.exp(L - m).sum(axis, keepdims=True))).squeeze(axis)


def nll_mean(y, L):
    L = np.asarray(L, dtype=np.float64)
    return float((lse(L) - L[np.arange(len(y)), y]).mean())


def metrics(y, p):
    return C.task_vectors(y, p, K)


def cluster_se(d, unit):
    """SE of the mean of d with household clustering (cluster sums of centred values, G/(G-1)); same formula as the scorer."""
    d = np.asarray(d, dtype=np.float64); n = len(d)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=d - d.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def compare(y, unit, preds, refs):
    """Absolute metrics of every prediction and paired differences (candidate minus reference; accuracy as a loss) with
    household-clustered SEs."""
    vec = {k: metrics(y, p) for k, p in preds.items()}
    out = {'absolute': {k: {m: float(v[m].mean()) for m in v} for k, v in vec.items()}, 'people': int(len(y)),
           'households': int(len(np.unique(unit)))}
    for r in refs:
        b = vec[r]
        for k, a in vec.items():
            if k == r:
                continue
            out[f'{k}_minus_{r}'] = {
                'nll': {'value': float((a['nll'] - b['nll']).mean()), 'se': cluster_se(a['nll'] - b['nll'], unit)},
                'brier': {'value': float((a['brier'] - b['brier']).mean()), 'se': cluster_se(a['brier'] - b['brier'], unit)},
                'accuracy_loss': {'value': float((b['accuracy'] - a['accuracy']).mean()),
                                  'se': cluster_se(b['accuracy'] - a['accuracy'], unit)}}
    return out


# ---------------------------------------------------------------- calibrators (one strength; vector scaling; temperature)
def fit_alpha(z, g, y, bounds=ALPHA_BOUNDS):
    """alpha = argmin_{alpha in bounds} NLL(softmax(z + alpha g)) (convex in alpha; bounded Brent)."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda a: nll_mean(y, z + a * g), bounds=bounds, method='bounded', options={'xatol': 1e-9, 'maxiter': 500})
    a = float(r.x)
    return a, {'alpha': a, 'nll_at_alpha': float(r.fun), 'nll_at_0': nll_mean(y, z), 'nll_at_1': nll_mean(y, z + g),
               'at_bound': bool(a - bounds[0] < 1e-4 or bounds[1] - a < 1e-4), 'iterations': int(r.nfev)}


def fit_vscal(z, y):
    """Vector scaling softmax(w * z + b): w = exp(v) > 0, b centred after the fit (softmax-invariant shift); L-BFGS-B on NLL."""
    from scipy.optimize import minimize
    n = len(y); rows = np.arange(n)

    def f(theta):
        w = np.exp(theta[:K]); b = theta[K:]
        L = z * w + b
        s = lse(L)
        val = float((s - L[rows, y]).mean())
        G = np.exp(L - s[:, None]); G[rows, y] -= 1.0; G /= n
        return val, np.concatenate([(G * z).sum(0) * w, G.sum(0)])
    r = minimize(f, np.zeros(2 * K), jac=True, method='L-BFGS-B', bounds=[(-3.0, 3.0)] * K + [(-5.0, 5.0)] * K,
                 options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
    w = np.exp(r.x[:K]); b = r.x[K:] - r.x[K:].mean()
    return w, b, {'w': w.tolist(), 'b': b.tolist(), 'nll_fit': float(r.fun), 'nll_identity': nll_mean(y, z), 'converged': bool(r.success),
                  'iterations': int(r.nit)}


def fit_temperature(z, y):
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: nll_mean(y, z / np.exp(lt)), bounds=(np.log(0.25), np.log(4.0)), method='bounded',
                        options={'xatol': 1e-9, 'maxiter': 500})
    return float(np.exp(r.x)), {'temperature': float(np.exp(r.x)), 'nll_fit': float(r.fun), 'nll_identity': nll_mean(y, z)}


# ---------------------------------------------------------------- xgboost residual trees and their numpy evaluator
def xgb_train(X, y, margin, depth, rounds):
    import xgboost as xgb
    d = xgb.DMatrix(np.asarray(X, dtype=np.float32), label=np.asarray(y), base_margin=np.asarray(margin, dtype=np.float32), nthread=1)
    return xgb.train({**XGB_FIXED, 'max_depth': int(depth)}, d, num_boost_round=int(rounds), verbose_eval=False)


def parse_trees(bst):
    """Explicit arrays of an xgboost multi:softprob booster (JSON model): per tree, children, split column, float32 threshold
    (or float32 leaf value at a leaf) and the class the tree adds to."""
    J = json.loads(bytes(bst.save_raw(raw_format='json')).decode())
    model = J['learner']['gradient_booster']['model']
    trees = model['trees']; info = np.asarray(model['tree_info'], dtype=np.int64)
    nt = len(trees)
    if nt % K or not np.array_equal(info, np.tile(np.arange(K), nt // K)):
        raise AssertionError('unexpected tree order (expected one tree per class per round)')
    width = max(len(t['left_children']) for t in trees)
    left = np.full((nt, width), -1, dtype=np.int64); right = np.full((nt, width), -1, dtype=np.int64)
    feat = np.zeros((nt, width), dtype=np.int64); cond = np.zeros((nt, width), dtype=np.float32)
    for i, t in enumerate(trees):
        m = len(t['left_children'])
        left[i, :m] = t['left_children']; right[i, :m] = t['right_children']
        feat[i, :m] = t['split_indices']; cond[i, :m] = np.asarray(t['split_conditions'], dtype=np.float64).astype(np.float32)
        if any(int(x) != 0 for x in t.get('split_type', [])) or len(t.get('categories', [])):
            raise AssertionError('categorical split in a numeric-only model')
    is_leaf = left < 0
    feat[is_leaf] = 0
    depth = 0
    for i in range(nt):                                    # maximum depth over trees (iterations of the evaluator)
        dep = np.zeros(width, dtype=np.int64)
        for j in range(width):
            if not is_leaf[i, j]:
                dep[left[i, j]] = dep[j] + 1; dep[right[i, j]] = dep[j] + 1
        depth = max(depth, int(dep[: len(trees[i]['left_children'])].max()))
    return {'left': left, 'right': right, 'feat': feat, 'cond': cond, 'is_leaf': is_leaf, 'cls': info,
            'max_depth': np.int64(depth), 'rounds': np.int64(nt // K)}


def leaf_index(T, X, chunk=1024):
    """Node id of the leaf each row reaches in every tree (numpy routing)."""
    X32 = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, nt = len(X32), T['cls'].shape[0]
    out = np.empty((n, nt), dtype=np.int64); col = np.arange(nt)[None, :]
    for s in range(0, n, chunk):
        Xc = X32[s:s + chunk]; m = len(Xc); rows = np.arange(m)[:, None]
        node = np.zeros((m, nt), dtype=np.int64)
        for _ in range(int(T['max_depth'])):
            leaf = T['is_leaf'][col, node]
            go_left = Xc[rows, T['feat'][col, node]] < T['cond'][col, node]
            node = np.where(leaf, node, np.where(go_left, T['left'][col, node], T['right'][col, node]))
        if not T['is_leaf'][col, node].all():
            raise AssertionError('routing did not end at leaves')
        out[s:s + m] = node
    return out


def leaf_values(T, X, chunk=1024):
    idx = leaf_index(T, X, chunk)
    return T['cond'][np.arange(idx.shape[1])[None, :], idx]          # float32 leaf values, (n, n_trees)


def correction(T, X, rounds=None, lv=None):
    """g(x) = sum of the first `rounds` rounds of trees, per class, float64 sequential accumulation."""
    lv = leaf_values(T, X) if lv is None else lv
    R = int(T['rounds']) if rounds is None else int(rounds)
    return lv[:, :R * K].astype(np.float64).reshape(len(lv), R, K).sum(1)


def round_curve_sum(T, X, y, z, chunk=1024):
    """Sum over rows of the multinomial NLL of softmax(z + g_m) after each round m = 1..R (held-out curve)."""
    R = int(T['rounds']); out = np.zeros(R)
    for s in range(0, len(X), chunk):
        lv = leaf_values(T, X[s:s + chunk])
        cum = np.cumsum(lv.astype(np.float64).reshape(len(lv), R, K), axis=1)
        L = z[s:s + chunk][:, None, :] + cum
        yy = y[s:s + chunk]
        out += (lse(L) - L[np.arange(len(yy)), :, yy]).sum(0)
    return out


def _sha_arrays(T):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(T):
        a = np.ascontiguousarray(T[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- correction inputs (deterministic re-codings of X)
CODE_BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')


def compact_codes(X, names):
    """Deterministic re-coding of the admitted one-hot blocks into one ordered code each (label-free): COW, MAR, POBP = index of
    the active column inside the block (columns are in code order; the block's __OTHER__ level last), SCHL = its ordinal value
    (__OTHER__, the merged rare pre-grade-3 codes, = 3). Same coding as ihs.fam_h.compact without the two numeric columns."""
    names = tuple(map(str, names)); X = np.asarray(X)
    cols = []
    for b in CODE_BLOCKS:
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
    return np.column_stack(cols).astype(np.float32)


def design(X, names, inputs='X'):
    """Input matrix of the correction trees: 'X' = the admitted encoding itself (unchanged object); 'X+codes' = X plus the four
    ordered block codes (lets a tree split on code ranges, e.g. place-of-birth regions, which one-hot columns cannot)."""
    if inputs == 'X':
        return X
    if inputs == 'X+codes':
        return np.ascontiguousarray(np.column_stack([np.asarray(X, dtype=np.float32), compact_codes(X, names)]), dtype=np.float32)
    raise KeyError(inputs)


# ---------------------------------------------------------------- routing-fixed re-estimation of the teacher (exposure tool)
def reboost(leaves, y, rows, n_trees, n_leaves=16, lr=0.05, l2=3.0):
    """Re-estimate the leaf values of a fixed oblivious-tree routing on the people `rows` with CatBoost-like Plain boosting: per
    tree one full-Hessian multiclass Newton step, delta_l = -(H_l + l2 I)^-1 G_l, times the learning rate; unit weights (the
    teacher's Bayesian bootstrap weights are not reproducible), start at 0 (boost_from_average false). Returns (T, L, K) values."""
    lv = np.asarray(leaves)[rows]; Y = np.eye(K)[np.asarray(y)[rows]]
    m = len(lv); raw = np.zeros((m, K)); vals = np.zeros((n_trees, n_leaves, K)); I = np.eye(K)
    iu = np.triu_indices(K)
    for t in range(n_trees):
        p = probs(raw); g = p - Y; leaf = lv[:, t]
        G = np.stack([np.bincount(leaf, weights=g[:, c], minlength=n_leaves) for c in range(K)], 1)
        H = np.zeros((n_leaves, K, K))
        for a, b in zip(*iu):
            w = np.bincount(leaf, weights=(p[:, a] * (a == b) - p[:, a] * p[:, b]), minlength=n_leaves)
            H[:, a, b] = w; H[:, b, a] = w
        delta = -np.linalg.solve(H + l2 * I, G[..., None])[..., 0]
        vals[t] = lr * delta
        raw += vals[t][leaf]
    return vals


def reboost_raw(vals, leaves):
    t = np.arange(vals.shape[0])[None, :]
    return vals[t, leaves[:, :vals.shape[0]]].sum(1)


def exposure_diagnostic(seed):
    """How in-sample is the frozen teacher on MODEL_FIT? Routing-fixed re-estimation on all MODEL_FIT (fidelity against the
    actual teacher on CALIBRATION) and without each registered household fold (out-of-fold margins for that fold): in-sample
    versus out-of-fold NLL / Brier on MODEL_FIT and held-out NLL on CALIBRATION. Reads MODEL_FIT and CALIBRATION occupation
    labels; no SEX; nothing is released."""
    t0 = time.process_time()
    F = C.load_role('MODEL_FIT', ('features', 'occupation'), purpose=f'family C S{seed}: exposure diagnostic (routing-fixed re-estimation)')
    Cal = C.load_role('CALIBRATION', ('features', 'occupation'), purpose=f'family C S{seed}: exposure diagnostic (held-out reference)')
    te = C.teacher(seed, 'occupation'); names = F['feature_names']
    Lf, Lc = te.route(F['X'], names), te.route(Cal['X'], names)
    yf, yc = F['occupation'], Cal['occupation']
    zt_f = centre(te.raw_from_leaves(Lf)); zt_c = centre(te.raw_from_leaves(Lc))
    folds = C.crossfit_folds(F['unit'])
    full = reboost(Lf, yf, np.arange(len(yf)), te.T)
    zr_f, zr_c = centre(reboost_raw(full, Lf)), centre(reboost_raw(full, Lc))
    oof = np.empty_like(zt_f); per_fold = {}
    for k in range(5):
        v = reboost(Lf, yf, np.where(folds != k)[0], te.T)
        te_k = folds == k
        oof[te_k] = centre(reboost_raw(v, Lf[te_k]))
        per_fold[k] = {'oof_nll': float(metrics(yf[te_k], probs(oof[te_k]))['nll'].mean()),
                       'reest_in_sample_nll': float(metrics(yf[te_k], probs(zr_f[te_k]))['nll'].mean()),
                       'teacher_in_sample_nll': float(metrics(yf[te_k], probs(zt_f[te_k]))['nll'].mean()),
                       'cal_nll_fold_model': float(metrics(yc, probs(centre(reboost_raw(v, Lc))))['nll'].mean())}
    def mm(y, z):
        v = metrics(y, probs(z)); return {k2: float(x.mean()) for k2, x in v.items()}
    stored_tr = stored_base_prob(seed, 'TASK_REFIT', 'MODEL_FIT'); stored_tr_c = stored_base_prob(seed, 'TASK_REFIT', 'CALIBRATION')
    out = {'utc': C.now(), 'seed': seed, 'trees': te.T,
           'fidelity_on_CALIBRATION': {'corr_centred_raw': float(np.corrcoef(zr_c.ravel(), zt_c.ravel())[0, 1]),
                                       'mean_abs_raw_diff': float(np.abs(zr_c - zt_c).mean()),
                                       'max_abs_prob_diff': float(np.abs(probs(zr_c) - probs(zt_c)).max()),
                                       'decision_agreement': float((zr_c.argmax(1) == zt_c.argmax(1)).mean())},
           'MODEL_FIT': {'teacher_in_sample': mm(yf, zt_f), 'reestimate_in_sample': mm(yf, zr_f), 'reestimate_out_of_fold': mm(yf, oof),
                         'TASK_REFIT_in_sample': {k2: float(x.mean()) for k2, x in metrics(yf, stored_tr).items()}},
           'CALIBRATION': {'teacher': mm(yc, zt_c), 'reestimate_full': mm(yc, zr_c),
                           'TASK_REFIT': {k2: float(x.mean()) for k2, x in metrics(yc, stored_tr_c).items()}},
           'per_fold': per_fold, 'cpu_s': time.process_time() - t0,
           'note': 'routing (split structure) of the teacher was learned on all MODEL_FIT; only leaf values are re-estimated per fold'}
    d = C.RUNS / FAMILY / 'DIAG'; d.mkdir(parents=True, exist_ok=True, mode=0o700)
    (d / f'EXPOSURE_S{seed}.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- the residual pipeline (shared by fit and selftest)
def residual_pipeline(Xf, yf, zf, folds, Xc, yc, zc, depth, m_max=M_MAX, log=print, train=xgb_train):
    """CV for M on the registered folds, full refit, alpha on CALIBRATION, and the household cross-fitting replication of every
    label-using step without each fold (M_k from a CV over the other folds, g_k = first M_k rounds of the model trained without
    fold k, alpha_k on CALIBRATION).  Returns the fitted pieces and the fold models (for out-of-fold and fold-average outputs)."""
    folds = np.asarray(folds); kf = int(folds.max()) + 1
    n = len(yf); t0 = time.process_time(); fits = 0
    models4, curve4 = {}, {}
    for k in range(kf):                                    # trained without fold k, evaluated on fold k (CV for M; fold-k refit)
        tr, te = folds != k, folds == k
        bst = train(Xf[tr], yf[tr], zf[tr], depth, m_max); fits += 1
        T = parse_trees(bst)
        models4[k] = T
        curve4[k] = round_curve_sum(T, Xf[te], yf[te], zf[te])
        log(f'  fold model {k}: {int(tr.sum())} people, {int(T["rounds"])} rounds, cpu {time.process_time() - t0:.1f}s')
    curve = sum(curve4.values()) / n
    M = int(np.argmin(curve)) + 1
    pair_curves = {}
    for k in range(kf):                                    # trained without folds k and j: CV of the fold-k pipeline (and fold j's)
        for j in range(k + 1, kf):
            tr = (folds != k) & (folds != j)
            bst = train(Xf[tr], yf[tr], zf[tr], depth, m_max); fits += 1
            T = parse_trees(bst)
            pair_curves[(k, j)] = round_curve_sum(T, Xf[folds == j], yf[folds == j], zf[folds == j])   # validation fold j (for k)
            pair_curves[(j, k)] = round_curve_sum(T, Xf[folds == k], yf[folds == k], zf[folds == k])   # validation fold k (for j)
        log(f'  pair models for fold {k} done, cpu {time.process_time() - t0:.1f}s')
    Mk, curve_k = {}, {}
    for k in range(kf):
        nk = int((folds != k).sum())
        curve_k[k] = sum(pair_curves[(k, j)] for j in range(kf) if j != k) / nk
        Mk[k] = int(np.argmin(curve_k[k])) + 1
    bst_full = train(Xf, yf, zf, depth, M); fits += 1
    T_full = parse_trees(bst_full)
    g_cal = correction(T_full, Xc)
    alpha, ainfo = fit_alpha(zc, g_cal, yc)
    cal_fits = 1
    alpha_k, ainfo_k = {}, {}
    for k in range(kf):
        a, info = fit_alpha(zc, correction(models4[k], Xc, Mk[k]), yc); cal_fits += 1
        alpha_k[k], ainfo_k[k] = a, info
    return {'M': M, 'curve': curve, 'alpha': alpha, 'alpha_info': ainfo, 'T_full': T_full, 'bst_full': bst_full,
            'models4': models4, 'Mk': Mk, 'curve_k': curve_k, 'alpha_k': alpha_k, 'alpha_info_k': ainfo_k,
            'xgb_fits': fits, 'calibrator_fits': cal_fits, 'cpu_s': time.process_time() - t0, 'g_cal': g_cal}


def xf_leaf_values(leaves, y, folds, n_trees, max_excl=3):
    """Exact routing-fixed re-estimations of the teacher's leaf values without every set of at most `max_excl` registered folds
    (the empty set reproduces the frozen teacher). {frozenset(excluded folds): (T, L, K) leaf values}."""
    from itertools import combinations
    kf = int(folds.max()) + 1
    out = {}
    for r in range(max_excl + 1):
        for ex in combinations(range(kf), r):
            out[frozenset(ex)] = reboost(leaves, y, np.where(~np.isin(folds, ex))[0], n_trees)
    return out


def xf_margin(V, excl, L):
    return centre(reboost_raw(V[frozenset(excl)], L))


def oof_margins(V, L, folds, rows, excl):
    """Training margins of the pipeline that excludes the folds `excl`: each person's teacher margin re-estimated without `excl`
    and without the person's own fold (cross-fitted inside the pipeline's own data)."""
    out = np.empty((len(rows), K)); fr = folds[rows]
    for f in np.unique(fr):
        sel = fr == f
        out[sel] = xf_margin(V, set(excl) | {int(f)}, L[rows][sel])
    return out


def residual_pipeline_xf(Xf, yf, Lf, folds, V, Xc, yc, zc, Lc, depth, m_max=M_MAX, log=print, train=xgb_train):
    """Revision 1: the residual pipeline with CROSS-FITTED base margins. Every model of the pipeline that excludes the folds E is
    trained on margins re-estimated without E and without each training person's own fold, and validated on margins re-estimated
    without E and the validation fold; the deployed correction (all folds) is trained on z^(-own fold); alpha on CALIBRATION with
    the frozen teacher z_T; the fold-k pipeline's alpha_k with z^(-k) on CALIBRATION."""
    folds = np.asarray(folds); kf = int(folds.max()) + 1; n = len(yf); t0 = time.process_time(); fits = 0
    models4, curve4 = {}, {}
    for k in range(kf):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        bst = train(Xf[tr], yf[tr], oof_margins(V, Lf, folds, tr, {k}), depth, m_max); fits += 1
        T = parse_trees(bst); models4[k] = T
        curve4[k] = round_curve_sum(T, Xf[te], yf[te], xf_margin(V, {k}, Lf[te]))
        log(f'  xf fold model {k}: {len(tr)} people, cpu {time.process_time() - t0:.1f}s')
    curve = sum(curve4.values()) / n
    M = int(np.argmin(curve)) + 1
    pair_curves = {}
    for k in range(kf):
        for j in range(k + 1, kf):
            tr = np.where((folds != k) & (folds != j))[0]
            bst = train(Xf[tr], yf[tr], oof_margins(V, Lf, folds, tr, {k, j}), depth, m_max); fits += 1
            T = parse_trees(bst)
            for a, b in ((k, j), (j, k)):                  # pipeline a validates on fold b with margins without {k, j}
                te = folds == b
                pair_curves[(a, b)] = round_curve_sum(T, Xf[te], yf[te], xf_margin(V, {k, j}, Lf[te]))
        log(f'  xf pair models for fold {k} done, cpu {time.process_time() - t0:.1f}s')
    Mk, curve_k = {}, {}
    for k in range(kf):
        curve_k[k] = sum(pair_curves[(k, j)] for j in range(kf) if j != k) / int((folds != k).sum())
        Mk[k] = int(np.argmin(curve_k[k])) + 1
    bst_full = train(Xf, yf, oof_margins(V, Lf, folds, np.arange(n), set()), depth, M); fits += 1
    T_full = parse_trees(bst_full)
    g_cal = correction(T_full, Xc)
    alpha, ainfo = fit_alpha(zc, g_cal, yc); cal_fits = 1
    alpha_k, ainfo_k = {}, {}
    for k in range(kf):
        a, info = fit_alpha(xf_margin(V, {k}, Lc), correction(models4[k], Xc, Mk[k]), yc); cal_fits += 1
        alpha_k[k], ainfo_k[k] = a, info
    return {'M': M, 'curve': curve, 'alpha': alpha, 'alpha_info': ainfo, 'T_full': T_full, 'bst_full': bst_full,
            'models4': models4, 'Mk': Mk, 'curve_k': curve_k, 'alpha_k': alpha_k, 'alpha_info_k': ainfo_k,
            'xgb_fits': fits, 'calibrator_fits': cal_fits, 'cpu_s': time.process_time() - t0, 'g_cal': g_cal}


def crossfit_outputs_xf(res, V, Xf, Lf, folds, Xs, Ls):
    """Fully cross-fitted except the teacher's split structure: fold k is predicted by z^(-k) + alpha_k g_k; the MODEL_SELECT
    fold average is the mean of the five fold pipelines z^(-k) + alpha_k g_k."""
    oof = np.empty((len(Xf), K)); fa = np.zeros((len(Xs), K)); kf = len(res['Mk'])
    for k in range(kf):
        te = folds == k
        oof[te] = probs(xf_margin(V, {k}, Lf[te]) + res['alpha_k'][k] * correction(res['models4'][k], Xf[te], res['Mk'][k]))
        fa += probs(xf_margin(V, {k}, Ls) + res['alpha_k'][k] * correction(res['models4'][k], Xs, res['Mk'][k])) / kf
    return oof, fa


def crossfit_outputs(res, Xf, zf, folds, Xs, zs):
    """MODEL_FIT out-of-fold probabilities (fold k by the fold-k pipeline) and the MODEL_SELECT fold average."""
    oof = np.empty((len(Xf), K)); fa = np.zeros((len(Xs), K)); kf = len(res['Mk'])
    gain = {}
    for k in range(kf):
        te = folds == k
        g = correction(res['models4'][k], Xf[te], res['Mk'][k])
        oof[te] = probs(zf[te] + res['alpha_k'][k] * g)
        fa += probs(zs + res['alpha_k'][k] * correction(res['models4'][k], Xs, res['Mk'][k])) / kf
        gain[k] = g
    return oof, fa, gain


def cal_crossfit_check(zc, gc, yc, uc, extra_preds=None, tag=CAL_CHECK_TAG):
    """Held-out CALIBRATION check: alpha fitted on 4 household folds of CALIBRATION and applied to the fifth (and the vector-scaled
    teacher likewise); paired comparisons with the teacher and the other frozen predictions, household-clustered SEs."""
    cf = C.household_folds(uc, 5, tag)
    p_cf = np.empty((len(yc), K)); vs_cf = np.empty((len(yc), K)); alphas, fits = [], 0
    for j in range(5):
        tr, te = cf != j, cf == j
        if gc is not None:
            a, _ = fit_alpha(zc[tr], gc[tr], yc[tr]); fits += 1; alphas.append(a)
            p_cf[te] = probs(zc[te] + a * gc[te])
        w, b, _ = fit_vscal(zc[tr], yc[tr]); fits += 1
        vs_cf[te] = probs(zc[te] * w + b)
    preds = {'teacher': probs(zc), 'vscal_cf': vs_cf}
    if gc is not None:
        preds['corr_alpha1'] = probs(zc + gc)
        preds['corr_alpha_cf'] = p_cf
    preds.update(extra_preds or {})
    refs = ['teacher'] + [r for r in ('TASK_REFIT',) if r in preds]
    out = compare(yc, uc, preds, refs)
    out['alpha_by_cal_fold'] = alphas
    return out, fits


# ---------------------------------------------------------------- real-data plumbing
def run_dir(config_id, seed):
    from .devrun import occ_dir
    return occ_dir(FAMILY, config_id, seed)


def base_margin(seed, base, X, names):
    te = C.teacher(seed, 'occupation')
    theta = C.stage_b_theta(seed, 'TASK_REFIT')[1] if base == 'TASK_REFIT' else None
    if base not in ('T_TREE', 'TASK_REFIT'):
        raise KeyError(base)
    return centre(te.raw_from_leaves(te.route(X, names), theta)), te.sha()


def stored_base_prob(seed, base, role):
    if base == 'T_TREE':
        return C.stage_a_outputs(seed)[f'T_TREE__{role}__occupation']
    return C.stage_b_outputs(seed)[f'{base}__{role}__occupation']


def load_trees(d):
    P = json.loads((d / 'PARAMS.json').read_text())
    if P['op'] not in ('residual', 'residual_xf'):
        return P, None
    p = d / 'TREES.npz'
    if C.sha(p) != P['trees_npz_sha256']:
        raise PermissionError('saved tree arrays differ from their hash in PARAMS.json')
    with np.load(p) as z:
        T = {k: z[k].copy() for k in z.files}
    if _sha_arrays(T) != P['trees_arrays_sha256']:
        raise PermissionError('tree arrays differ from their content hash')
    return P, T


def predict(seed, config_id, X, names):
    """Deterministic deployment from features: the frozen teacher margin plus the saved correction (or the saved vector scaling)."""
    d = run_dir(config_id, seed)
    P, T = load_trees(d)
    z, tsha = base_margin(seed, P['base'], X, names)
    if tsha != P['teacher_sha256']:
        raise PermissionError('frozen teacher differs from the one recorded at fit time')
    if P['op'] in ('residual', 'residual_xf'):
        return probs(z + float(P['alpha']) * correction(T, design(X, names, P.get('inputs', 'X'))))
    if P['op'] == 'vscal':
        return probs(z * np.asarray(P['w'], dtype=np.float64) + np.asarray(P['b'], dtype=np.float64))
    raise KeyError(P['op'])


def _features(role):
    F = C.load_role(role, ('features',))
    return F['X'], F['feature_names'], F['unit']


def _fit_xf(seed, config_id, cfg, F, Cal, Xr, z, yf, yc, folds, names, d, m_max, stored_tr, xgb_check=True):
    """Revision 1 fit: residual trained against cross-fitted teacher margins (exact routing-fixed re-estimation without each
    set of folds), deployed on the frozen teacher; fully nested cross-fitting check (only the split structure stays in-sample)."""
    from . import devrun as DR
    import xgboost as xgb
    t0 = time.process_time()
    te = C.teacher(seed, 'occupation')
    L = {role: te.route(Xr[role], names) for role in C.ROLES6}
    V = xf_leaf_values(L['MODEL_FIT'], yf, folds, te.T, max_excl=3)
    n_reest = len(V)
    fidelity = {role: float(np.max(np.abs(probs(xf_margin(V, set(), L[role])) - probs(z[role])))) for role in C.ROLES6}
    if max(fidelity.values()) > 1e-6 and not _DRYRUN_INVENTED_LABELS:
        raise AssertionError(f'routing-fixed re-estimation does not reproduce the frozen teacher: {fidelity}')
    reest_cpu = time.process_time() - t0
    depth = cfg['depth']; inputs = cfg.get('inputs', 'X')
    Xd = {role: design(Xr[role], names, inputs) for role in C.ROLES6}
    res = residual_pipeline_xf(Xd['MODEL_FIT'], yf, L['MODEL_FIT'], folds, V, Xd['CALIBRATION'], yc, z['CALIBRATION'], L['CALIBRATION'],
                               depth, m_max=m_max)
    T = res['T_full']
    parity = {}
    if xgb_check:
        for role in ('CALIBRATION', 'MODEL_SELECT'):
            dm = xgb.DMatrix(np.asarray(Xd[role], dtype=np.float32), base_margin=np.zeros((len(Xd[role]), K), dtype=np.float32), nthread=1)
            ref = res['bst_full'].predict(dm, output_margin=True).reshape(len(Xd[role]), K)
            leaves = res['bst_full'].predict(dm, pred_leaf=True).reshape(len(Xd[role]), -1)
            parity[role] = {'max_abs_margin_diff': float(np.max(np.abs(correction(T, Xd[role]) - ref))),
                            'leaf_ids_identical': bool(np.array_equal(leaf_index(T, Xd[role]), leaves))}
        if not all(v['leaf_ids_identical'] and v['max_abs_margin_diff'] <= 1e-4 for v in parity.values()):
            raise AssertionError(f'numpy evaluator differs from xgboost: {parity}')
    occ = {role: probs(z[role] + res['alpha'] * correction(T, Xd[role])) for role in C.ROLES6}
    oof, fa = crossfit_outputs_xf(res, V, Xd['MODEL_FIT'], L['MODEL_FIT'], folds, Xd['MODEL_SELECT'], L['MODEL_SELECT'])
    crossfit = {'MODEL_FIT__oof': oof, 'MODEL_SELECT__foldavg': fa, 'fold': folds}
    diag = {}
    for k in range(5):
        tk = folds == k
        zk_f = xf_margin(V, {k}, L['MODEL_FIT'][tk]); zk_c = xf_margin(V, {k}, L['CALIBRATION'])
        gk_f = correction(res['models4'][k], Xd['MODEL_FIT'][tk], res['Mk'][k]); gk_c = correction(res['models4'][k], Xd['CALIBRATION'], res['Mk'][k])
        a = res['alpha_k'][k]
        diag[k] = {'M_k': res['Mk'][k], 'alpha_k': a,
                   'oof_base_nll': nll_mean(yf[tk], zk_f), 'teacher_in_sample_nll': nll_mean(yf[tk], z['MODEL_FIT'][tk]),
                   'mf_oof_gain_alpha1': nll_mean(yf[tk], zk_f) - nll_mean(yf[tk], zk_f + gk_f),
                   'mf_oof_gain_alpha_k': nll_mean(yf[tk], zk_f) - nll_mean(yf[tk], zk_f + a * gk_f),
                   'cal_gain_alpha1': nll_mean(yc, zk_c) - nll_mean(yc, zk_c + gk_c),
                   'cal_gain_alpha_k': nll_mean(yc, zk_c) - nll_mean(yc, zk_c + a * gk_c)}
    check, cfits = cal_crossfit_check(z['CALIBRATION'], res['g_cal'], yc, Cal['unit'], {'TASK_REFIT': stored_tr['CALIBRATION']})
    oof_base = np.empty_like(oof)
    for k in range(5):
        oof_base[folds == k] = probs(xf_margin(V, {k}, L['MODEL_FIT'][folds == k]))
    mf_full = {'MODEL_FIT_in_sample_teacher': metrics(yf, probs(z['MODEL_FIT'])), 'MODEL_FIT_oof_teacher_reestimated': metrics(yf, oof_base),
               'MODEL_FIT_release_in_sample': metrics(yf, occ['MODEL_FIT']), 'MODEL_FIT_oof': metrics(yf, oof)}
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez(d / 'TREES.npz', **T)
    res['bst_full'].save_model(str(d / 'MODEL.json'))
    P = {'op': 'residual_xf', 'base': cfg['base'], 'depth': depth, 'M': res['M'], 'alpha': res['alpha'], 'xgb_params': {**XGB_FIXED, 'max_depth': depth},
         'm_max': m_max, 'alpha_bounds': list(ALPHA_BOUNDS), 'teacher_sha256': te.sha(), 'trees_npz_sha256': C.sha(d / 'TREES.npz'),
         'trees_arrays_sha256': _sha_arrays(T), 'xgb_model_json_sha256': C.sha(d / 'MODEL.json'), 'seed': seed, 'config_id': config_id,
         'xgboost_version': xgb.__version__, 'inputs': inputs,
         'training_margins': 'cross-fitted: exact routing-fixed re-estimation of the teacher leaf values without the person\'s own fold'}
    (d / 'PARAMS.json').write_text(json.dumps(P, indent=1, sort_keys=True) + '\n')
    n_cal = res['calibrator_fits'] + cfits
    DR.unit_stats(model_fits=6, crossfit_fits=10, calibrator_fits=n_cal, reestimations=n_reest)
    upd = {
        'operation': f'p = softmax(z_T + alpha * g), z_T the centred frozen T_TREE occupation raw score (exact numpy routing); g = '
                     f'{res["M"]} rounds x 5 classes of depth-{depth} xgboost regression trees fitted on MODEL_FIT to the residual of '
                     'CROSS-FITTED teacher margins (each person\'s margin from the teacher leaf values re-estimated exactly, routing '
                     'fixed, without that person\'s household fold); M by household 5-fold CV on the registered folds with margins '
                     f're-estimated without the validation fold; alpha = {res["alpha"]:.6f} by NLL on CALIBRATION in [0, 2] with the '
                     'frozen teacher margin; deployed with a numpy evaluator of the saved trees on the frozen teacher margin',
        'exposure': {'T_TREE': 'frozen deployed base; all MODEL_FIT occupation labels; in-sample on MODEL_FIT, out of sample elsewhere',
                     'training_margins': 'teacher leaf values re-estimated without each fold (and, inside the cross-fitting check, '
                                         'without each pair / triple of folds); the split structure (routing) was learned by CatBoost '
                                         'on all MODEL_FIT and is NOT refitted',
                     'correction_g': 'MODEL_FIT occupation labels, residuals against cross-fitted margins',
                     'alpha': 'CALIBRATION occupation labels (1 parameter); in-sample on CALIBRATION for it',
                     'crossfit_note': 'MODEL_FIT__oof: fold k predicted by z^(-k) + alpha_k g_k, where z^(-k) are the teacher leaf values '
                                      're-estimated without fold k, M_k by a CV over the other folds with margins re-estimated without '
                                      '{k, j} (training: without {k, j, own fold}), g_k trained on margins without {k, own fold}, '
                                      'alpha_k on CALIBRATION with z^(-k): out of fold for every label-using step except the '
                                      'teacher\'s split structure; MODEL_SELECT__foldavg = mean of the five fold pipelines'},
        'reestimation': {'count': n_reest, 'max_excluded_folds': 3, 'fidelity_full_vs_frozen_max_abs_prob_diff': fidelity,
                         'cpu_s': reest_cpu},
        'M': res['M'], 'M_at_cap': bool(res['M'] == m_max), 'm_max': m_max, 'alpha': res['alpha'], 'alpha_info': res['alpha_info'],
        'cv_curve_every_25': [float(x) for x in res['curve'][::25]], 'cv_curve_min': float(res['curve'].min()),
        'cv_curve_at_1': float(res['curve'][0]), 'cv_curve_at_M_max': float(res['curve'][-1]),
        'crossfit_folds': {'sizes': [int((folds == k).sum()) for k in range(5)], 'per_fold': diag},
        'cal_check': check, 'numpy_vs_xgboost': parity,
        'model_fit_metrics': {k: {m: float(v[m].mean()) for m in v} for k, v in mf_full.items()},
        'n_model_fits': 6, 'n_crossfit_fits': 10, 'n_reestimations': n_reest,
        'fit_counting_note': '16 xgboost trainings (5 four-fold CV models, also the cross-fitting refits by exact prefix; 10 three-fold '
                             f'models; 1 full refit) and {n_reest} routing-fixed teacher re-estimations (1 full fidelity check, 5 single-, '
                             '10 pair-, 10 triple-fold exclusions)',
        'n_calibrator_fits': n_cal, 'params_sha256': C.sha(d / 'PARAMS.json')}
    return occ, crossfit, upd


def fit(seed, config_id, m_max=M_MAX):
    from . import devrun as DR
    cfg = CONFIGS[config_id]
    t_start = time.process_time()
    d = run_dir(config_id, seed)
    if (d / 'OCC.npz').exists():
        raise PermissionError(f'{config_id}/S{seed} already released (append-only)')
    F = C.load_role('MODEL_FIT', ('features', 'occupation'), purpose=f'family C {config_id} S{seed}: correction fit, CV, cross-fitting')
    Cal = C.load_role('CALIBRATION', ('features', 'occupation'), purpose=f'family C {config_id} S{seed}: strength / calibrator, held-out check')
    names = F['feature_names']
    Xr, Ur = {'MODEL_FIT': F['X'], 'CALIBRATION': Cal['X']}, {'MODEL_FIT': F['unit'], 'CALIBRATION': Cal['unit']}
    for role in C.ROLES6:
        if role not in Xr:
            X, nm, u = _features(role)
            if tuple(nm) != tuple(names):
                raise AssertionError('feature schema differs between roles')
            Xr[role], Ur[role] = X, u
    z, tsha = {}, None
    base_check = {}
    for role in C.ROLES6:
        z[role], tsha = base_margin(seed, cfg['base'], Xr[role], names)
        ref = stored_base_prob(seed, cfg['base'], role)
        diff = float(np.max(np.abs(probs(z[role]) - ref)))
        base_check[role] = diff
        if diff > 1e-12 or not np.array_equal(probs(z[role]).argmax(1), ref.argmax(1)):
            raise AssertionError(f'recomputed {cfg["base"]} occupation output differs from storage on {role}: {diff}')
    yf, yc = F['occupation'], Cal['occupation']
    folds = C.crossfit_folds(F['unit'])
    rec = {'kind': cfg['kind'], 'version': cfg['version'], 'config_role': cfg['role'], 'base': cfg['base'],
           'teacher_sha256': tsha, 'base_recompute_max_abs_diff': base_check, 'code_sha256': C.sha(Path(__file__)),
           'roles_read': 'MODEL_FIT (features, occupation labels: correction fit, CV for M, cross-fitting); CALIBRATION (features, '
                         'occupation labels: strength / vector scaling, held-out checks); features of all six roles (release); '
                         'stored T_TREE / TASK_REFIT outputs (label-free recomputation checks). No SEX, no MODEL_SELECT label.',
           'matched_controls': ['CC_CONT_D4', 'CC_VSCAL', 'frozen T_TREE', 'frozen TASK_REFIT']}
    stored_tr = {r: stored_base_prob(seed, 'TASK_REFIT', r) for r in ('MODEL_FIT', 'CALIBRATION')}
    rec['exposure_diag'] = {
        'teacher_nll_MODEL_FIT_in_sample': float(metrics(yf, probs(z['MODEL_FIT']))['nll'].mean()),
        'teacher_nll_CALIBRATION_held_out': float(metrics(yc, probs(z['CALIBRATION']))['nll'].mean()),
        'teacher_brier_MODEL_FIT_in_sample': float(metrics(yf, probs(z['MODEL_FIT']))['brier'].mean()),
        'teacher_brier_CALIBRATION_held_out': float(metrics(yc, probs(z['CALIBRATION']))['brier'].mean()),
        'TASK_REFIT_nll_MODEL_FIT_in_sample': float(metrics(yf, stored_tr['MODEL_FIT'])['nll'].mean()),
        'TASK_REFIT_nll_CALIBRATION_held_out': float(metrics(yc, stored_tr['CALIBRATION'])['nll'].mean())}
    if cfg['op'] == 'vscal':
        w, b, vinfo = fit_vscal(z['CALIBRATION'], yc)
        temp, tinfo = fit_temperature(z['CALIBRATION'], yc)
        occ = {role: probs(z[role] * w + b) for role in C.ROLES6}
        check, cfits = cal_crossfit_check(z['CALIBRATION'], None, yc, Cal['unit'], {'TASK_REFIT': stored_tr['CALIBRATION']})
        P = {'op': 'vscal', 'base': cfg['base'], 'w': w.tolist(), 'b': b.tolist(), 'teacher_sha256': tsha, 'seed': seed,
             'config_id': config_id}
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        (d / 'PARAMS.json').write_text(json.dumps(P, indent=1, sort_keys=True) + '\n')
        crossfit = {'MODEL_FIT__oof': occ['MODEL_FIT'], 'MODEL_SELECT__foldavg': occ['MODEL_SELECT'], 'fold': folds}
        rec.update({'operation': 'p = softmax(w * z_T + b), z_T the centred frozen T_TREE occupation raw score; w (5, positive) and '
                                 'b (5, centred) by NLL on CALIBRATION (vector scaling; calibration-only control)',
                    'exposure': {'T_TREE': 'all MODEL_FIT occupation labels; in-sample on MODEL_FIT, out of sample elsewhere',
                                 'vector_scaling': 'CALIBRATION occupation labels (9 parameters); in-sample on CALIBRATION',
                                 'crossfit_note': 'no MODEL_FIT label is used by this control, so MODEL_FIT__oof is the deployed '
                                                  'MODEL_FIT output (in-sample teacher margins) and the fold average equals the '
                                                  'deployed MODEL_SELECT output'},
                    'vscal': vinfo, 'temperature_diagnostic': tinfo, 'cal_check': check,
                    'n_model_fits': 0, 'n_calibrator_fits': 1 + 1 + cfits, 'n_crossfit_fits': 0,
                    'params_sha256': C.sha(d / 'PARAMS.json')})
        DR.unit_stats(calibrator_fits=1 + 1 + cfits)
    elif cfg['op'] == 'residual_xf':
        occ, crossfit, upd = _fit_xf(seed, config_id, cfg, F, Cal, Xr, z, yf, yc, folds, names, d, m_max, stored_tr, xgb_check=True)
        rec.update(upd)
    else:
        depth = cfg['depth']
        res = residual_pipeline(Xr['MODEL_FIT'], yf, z['MODEL_FIT'], folds, Xr['CALIBRATION'], yc, z['CALIBRATION'], depth, m_max=m_max)
        T = res['T_full']
        # numpy evaluator against xgboost's own margins and leaf ids (MODEL_SELECT and CALIBRATION features; label-free)
        import xgboost as xgb
        parity = {}
        for role in ('CALIBRATION', 'MODEL_SELECT'):
            dm = xgb.DMatrix(np.asarray(Xr[role], dtype=np.float32), base_margin=np.zeros((len(Xr[role]), K), dtype=np.float32), nthread=1)
            ref = res['bst_full'].predict(dm, output_margin=True).reshape(len(Xr[role]), K)
            leaves = res['bst_full'].predict(dm, pred_leaf=True).reshape(len(Xr[role]), -1)
            g = correction(T, Xr[role])
            parity[role] = {'max_abs_margin_diff': float(np.max(np.abs(g - ref))),
                            'leaf_ids_identical': bool(np.array_equal(leaf_index(T, Xr[role]), leaves))}
        if not all(v['leaf_ids_identical'] and v['max_abs_margin_diff'] <= 1e-4 for v in parity.values()):
            raise AssertionError(f'numpy evaluator differs from xgboost: {parity}')
        occ = {role: probs(z[role] + res['alpha'] * correction(T, Xr[role])) for role in C.ROLES6}
        oof, fa, _ = crossfit_outputs(res, Xr['MODEL_FIT'], z['MODEL_FIT'], folds, Xr['MODEL_SELECT'], z['MODEL_SELECT'])
        crossfit = {'MODEL_FIT__oof': oof, 'MODEL_SELECT__foldavg': fa, 'fold': folds}
        # exposure diagnostic: out-of-fold gain of the correction over the in-sample teacher on MODEL_FIT, against the same fold
        # models' gain on CALIBRATION (both at alpha = 1 and at alpha_k)
        diag = {}
        for k in range(5):
            te = folds == k
            gk_f = correction(res['models4'][k], Xr['MODEL_FIT'][te], res['Mk'][k])
            gk_c = correction(res['models4'][k], Xr['CALIBRATION'], res['Mk'][k])
            zf_k = z['MODEL_FIT'][te]
            diag[k] = {'M_k': res['Mk'][k], 'alpha_k': res['alpha_k'][k],
                       'mf_oof_gain_alpha1': nll_mean(yf[te], zf_k) - nll_mean(yf[te], zf_k + gk_f),
                       'cal_gain_alpha1': nll_mean(yc, z['CALIBRATION']) - nll_mean(yc, z['CALIBRATION'] + gk_c),
                       'mf_oof_gain_alpha_k': nll_mean(yf[te], zf_k) - nll_mean(yf[te], zf_k + res['alpha_k'][k] * gk_f),
                       'cal_gain_alpha_k': nll_mean(yc, z['CALIBRATION']) - nll_mean(yc, z['CALIBRATION'] + res['alpha_k'][k] * gk_c)}
        g_cal = res['g_cal']
        check, cfits = cal_crossfit_check(z['CALIBRATION'], g_cal, yc, Cal['unit'], {'TASK_REFIT': stored_tr['CALIBRATION']})
        mf_full = {'MODEL_FIT_in_sample_teacher': metrics(yf, probs(z['MODEL_FIT'])),
                   'MODEL_FIT_release_in_sample': metrics(yf, occ['MODEL_FIT']), 'MODEL_FIT_oof': metrics(yf, oof)}
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        np.savez(d / 'TREES.npz', **T)
        res['bst_full'].save_model(str(d / 'MODEL.json'))
        P = {'op': 'residual', 'base': cfg['base'], 'depth': depth, 'M': res['M'], 'alpha': res['alpha'], 'xgb_params': {**XGB_FIXED, 'max_depth': depth},
             'm_max': m_max, 'alpha_bounds': list(ALPHA_BOUNDS), 'teacher_sha256': tsha, 'trees_npz_sha256': C.sha(d / 'TREES.npz'),
             'trees_arrays_sha256': _sha_arrays(T), 'xgb_model_json_sha256': C.sha(d / 'MODEL.json'), 'seed': seed,
             'config_id': config_id, 'xgboost_version': xgb.__version__}
        (d / 'PARAMS.json').write_text(json.dumps(P, indent=1, sort_keys=True) + '\n')
        fold_sizes = [int((folds == k).sum()) for k in range(5)]
        rec.update({
            'operation': f'p = softmax(z_T + alpha * g), z_T the centred frozen {cfg["base"]} occupation raw score (exact numpy '
                         f'routing); g = {res["M"]} rounds x 5 classes of depth-{depth} xgboost regression trees fitted to the '
                         f'residual of z_T on MODEL_FIT (multi:softprob, base_margin z_T, eta 0.05, lambda 3), M by household 5-fold '
                         f'CV on the registered MODEL_FIT folds; alpha = {res["alpha"]:.6f} by NLL on CALIBRATION in [0, 2]; '
                         'deployed with a numpy evaluator of the saved trees',
            'exposure': {'T_TREE': 'frozen; all 20,000 MODEL_FIT occupation labels; in-sample on every MODEL_FIT person and fold; '
                                   'out of sample on CALIBRATION, MODEL_SELECT, ATTACK_*, DEV_REPORT; not refitted per fold',
                         'correction_g': 'MODEL_FIT occupation labels, residuals against the IN-SAMPLE teacher margins',
                         'alpha': 'CALIBRATION occupation labels (1 parameter); the release is in-sample on CALIBRATION for it',
                         'crossfit_note': 'MODEL_FIT__oof: fold k is predicted by z_T + alpha_k g_k with M_k (CV over the other '
                                          'folds), g_k and alpha_k refitted without fold k; z_T saw fold k, so the predictions '
                                          'are out of fold for the correction layer only (not fully out of sample); '
                                          'MODEL_SELECT__foldavg = mean of the five fold pipelines\' probability vectors'},
            'M': res['M'], 'M_at_cap': bool(res['M'] == m_max), 'm_max': m_max, 'alpha': res['alpha'], 'alpha_info': res['alpha_info'],
            'cv_curve_every_25': [float(x) for x in res['curve'][::25]], 'cv_curve_min': float(res['curve'].min()),
            'cv_curve_at_1': float(res['curve'][0]), 'cv_curve_at_M_max': float(res['curve'][-1]),
            'crossfit_folds': {'sizes': fold_sizes, 'per_fold': diag},
            'cal_check': check, 'numpy_vs_xgboost': parity,
            'model_fit_metrics': {k: {m: float(v[m].mean()) for m in v} for k, v in mf_full.items()},
            'n_model_fits': 6, 'n_crossfit_fits': 10,
            'fit_counting_note': '16 xgboost trainings: 5 four-fold CV models (also the fold refits of the cross-fitting check, '
                                 'truncated to M_k rounds: an exact prefix of the same boosting run), 10 three-fold models (the CV of '
                                 'each fold pipeline), 1 full refit with M rounds',
            'n_calibrator_fits': res['calibrator_fits'] + cfits, 'params_sha256': C.sha(d / 'PARAMS.json')})
        DR.unit_stats(model_fits=6, crossfit_fits=10, calibrator_fits=res['calibrator_fits'] + cfits)
    rec['fit_cpu_s'] = time.process_time() - t_start
    out = DR.save_occupation(FAMILY, config_id, seed, occ, rec, crossfit=crossfit)
    v = verify(seed, config_id)
    print(json.dumps({'config': config_id, 'seed': seed, 'dir': str(out), 'verify': v['all_bitwise'],
                      'summary': {k: rec.get(k) for k in ('M', 'alpha', 'fit_cpu_s')},
                      'cal_check_vs_teacher': {k: v2 for k, v2 in rec['cal_check'].items() if k.endswith('_minus_teacher')}}, default=float))
    return rec


def verify(seed, config_id):
    from .devrun import load_occupation
    occ, rec, _ = load_occupation(FAMILY, config_id, seed)
    res = {}
    for role in C.ROLES6:
        X, names, _ = _features(role)
        p = predict(seed, config_id, X, names)
        res[role] = {'bitwise': bool(np.array_equal(p, occ[role])), 'max_abs_diff': float(np.max(np.abs(p - occ[role])))}
    out = {'utc': C.now(), 'config_id': config_id, 'seed': seed, 'roles': res, 'all_bitwise': all(v['bitwise'] for v in res.values()),
           'interpreter': sys.executable, 'code_sha256': C.sha(Path(__file__))}
    d = run_dir(config_id, seed)
    with open(d / f'VERIFY_{time.strftime("%H%M%S", time.gmtime())}.json', 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)
    if not out['all_bitwise']:
        raise AssertionError(f'verify failed: {res}')
    return out


# ---------------------------------------------------------------- invented-data self-test (no real data is read)
def _invented(rng, n, n_house):
    num = rng.normal(size=(n, 2))
    cats = [rng.integers(0, m, size=n) for m in (5, 8, 4)]
    X = np.column_stack([num] + [np.eye(m)[c] for m, c in zip((5, 8, 4), cats)]).astype(np.float32)
    unit = np.sort(rng.integers(0, n_house, size=n))
    return X, unit


def _true_logits(X):
    A = np.array([[0.8, -0.3, 0.0, 0.2, -0.5], [0.1, 0.5, -0.4, 0.0, 0.2]])
    E = np.linspace(-0.6, 0.6, X.shape[1] - 2)[:, None] * np.array([1.0, -0.5, 0.3, -0.8, 0.2])[None, :]
    inter = (X[:, [0]] * X[:, [3]]) * np.array([0.0, 0.6, -0.6, 0.0, 0.0])
    return X[:, :2] @ A + X[:, 2:] @ E + inter


def _planted_residual(X):
    return np.outer(np.tanh(X[:, 1]), np.array([0.5, -0.5, 0.0, 0.25, -0.25])) + X[:, [8]] * np.array([-0.4, 0.0, 0.4, 0.0, 0.0])


def _draw(rng, L):
    p = probs(L)
    c = p.cumsum(1); u = rng.random(len(p))[:, None]
    return np.minimum((u > c).sum(1), K - 1)


def selftest():
    import xgboost as xgb
    rng = np.random.default_rng(20261018)
    checks = {}
    t0 = time.process_time()
    Xf, uf = _invented(rng, 6000, 3000); Xc, uc = _invented(rng, 3000, 1500); Xs, us = _invented(rng, 6000, 3000)
    truth = {k: _true_logits(X) for k, X in (('f', Xf), ('c', Xc), ('s', Xs))}
    planted = {k: _planted_residual(X) for k, X in (('f', Xf), ('c', Xc), ('s', Xs))}
    y = {k: _draw(rng, truth[k]) for k in truth}
    zt = {k: centre(truth[k] - planted[k]) for k in truth}           # an under-fit "teacher": truth minus a planted residual
    folds = C.household_folds(uf, 5, 'OHN1_famC_selftest')
    # 1. numpy evaluator equals xgboost margins and leaf ids; deterministic training; prefix property
    b1 = xgb_train(Xf, y['f'], zt['f'], 2, 60); b2 = xgb_train(Xf, y['f'], zt['f'], 2, 60); b3 = xgb_train(Xf, y['f'], zt['f'], 2, 25)
    T1, T3 = parse_trees(b1), parse_trees(b3)
    dm = xgb.DMatrix(Xs, base_margin=np.zeros((len(Xs), K), dtype=np.float32), nthread=1)
    ref = b1.predict(dm, output_margin=True).reshape(len(Xs), K)
    checks['numpy_evaluator_equals_xgboost_margin'] = bool(np.max(np.abs(correction(T1, Xs) - ref)) <= 1e-5)
    checks['numpy_leaf_ids_equal_xgboost'] = bool(np.array_equal(leaf_index(T1, Xs), b1.predict(dm, pred_leaf=True).reshape(len(Xs), -1)))
    checks['training_deterministic'] = bytes(b1.save_raw('json')) == bytes(b2.save_raw('json'))
    checks['prefix_equals_shorter_training'] = bool(np.array_equal(correction(T1, Xs, 25), correction(T3, Xs)))
    for depth in (1, 4):
        Td = parse_trees(xgb_train(Xf, y['f'], zt['f'], depth, 10))
        checks[f'depth{depth}_max_depth_respected'] = bool(int(Td['max_depth']) <= depth)
    # 2. residual pipeline recovers a planted residual (small M_max for speed) and transfers to held-out people
    res = residual_pipeline(Xf, y['f'], zt['f'], folds, Xc, y['c'], zt['c'], 2, m_max=300, log=lambda *a: None)
    g_s = correction(res['T_full'], Xs)
    nll_teacher = nll_mean(y['s'], zt['s']); nll_corr = nll_mean(y['s'], zt['s'] + res['alpha'] * g_s); nll_truth = nll_mean(y['s'], truth['s'])
    checks['planted_residual_improves_heldout_nll'] = bool(nll_corr < nll_teacher - 0.005)
    checks['planted_residual_alpha_near_1'] = bool(0.6 <= res['alpha'] <= 1.8)
    checks['M_selected_inside_range'] = bool(1 <= res['M'] <= 300)
    # 3. null case: the base is already the truth; the strength shrinks the correction and held-out NLL is not materially worse
    zt0 = {k: centre(truth[k]) for k in truth}
    res0 = residual_pipeline(Xf, y['f'], zt0['f'], folds, Xc, y['c'], zt0['c'], 2, m_max=300, log=lambda *a: None)
    nll0 = nll_mean(y['s'], zt0['s']); nll0c = nll_mean(y['s'], zt0['s'] + res0['alpha'] * correction(res0['T_full'], Xs))
    checks['null_case_not_materially_worse'] = bool(nll0c <= nll0 + 0.003)
    checks['null_case_alpha_or_M_small'] = bool(res0['alpha'] < res['alpha'] or res0['M'] < res['M'])
    # 4. alpha = 0 reproduces the base exactly
    checks['alpha0_is_base'] = bool(np.array_equal(probs(zt['s'] + 0.0 * g_s), probs(zt['s'])))
    # 5. cross-fitting never uses the predicted fold: permuting fold-0 labels leaves fold-0 out-of-fold outputs unchanged
    oof, fa, _ = crossfit_outputs(res, Xf, zt['f'], folds, Xs, zt['s'])
    yp = y['f'].copy(); m0 = folds == 0; yp[m0] = rng.permutation(yp[m0])
    resp = residual_pipeline(Xf, yp, zt['f'], folds, Xc, y['c'], zt['c'], 2, m_max=300, log=lambda *a: None)
    oofp, _, _ = crossfit_outputs(resp, Xf, zt['f'], folds, Xs, zt['s'])
    checks['oof_fold0_independent_of_fold0_labels'] = bool(np.array_equal(oof[m0], oofp[m0]))
    checks['oof_other_folds_change'] = bool(not np.array_equal(oof[~m0], oofp[~m0]))
    checks['foldavg_valid'] = bool(np.allclose(fa.sum(1), 1, atol=1e-12) and (fa >= 0).all())
    # 6. vector scaling recovers planted per-class scales and biases; temperature recovers a planted temperature
    Xv, uv = _invented(rng, 40000, 20000); zv = centre(_true_logits(Xv)); wv = np.array([0.7, 1.2, 0.9, 1.4, 1.0]); bv = np.array([0.2, -0.3, 0.1, 0.0, 0.0])
    bv = bv - bv.mean(); yv = _draw(rng, zv * wv + bv)
    w, b, _ = fit_vscal(zv, yv)
    checks['vscal_recovers_planted'] = bool(np.max(np.abs(w - wv)) < 0.1 and np.max(np.abs(b - bv)) < 0.06)
    yt = _draw(rng, zv / 1.3)
    tt, _ = fit_temperature(zv, yt)
    checks['temperature_recovers_planted'] = bool(abs(tt - 1.3) < 0.08)
    # 7. alpha recovers a planted strength
    gv = _planted_residual(Xv); ya = _draw(rng, zv + 0.6 * gv)
    a, _ = fit_alpha(zv, gv, ya)
    checks['alpha_recovers_planted'] = bool(abs(a - 0.6) < 0.08)
    # 8. release contract and determinism
    p1 = probs(zt['s'] + res['alpha'] * correction(res['T_full'], Xs)); p2 = probs(zt['s'] + res['alpha'] * correction(res['T_full'], Xs))
    checks['release_contract'] = bool(p1.shape == (len(Xs), K) and np.allclose(p1.sum(1), 1, atol=1e-12) and (p1 >= 0).all() and np.isfinite(p1).all())
    checks['release_deterministic'] = bool(np.array_equal(p1, p2))
    checks['row_chunking_invariant'] = bool(np.array_equal(correction(res['T_full'], Xs)[:100], correction(res['T_full'], Xs[:100])))
    # 9. clustered SE equals the iid SE when every household has one person (sanity) and the parameter-hash refusal works
    dd = rng.normal(size=500)
    checks['cluster_se_singletons'] = bool(abs(cluster_se(dd, np.arange(500)) - dd.std(ddof=1) / np.sqrt(500)) < 1e-12)
    T_t = dict(res['T_full']); sha0 = _sha_arrays(T_t); T_t['cond'] = T_t['cond'].copy(); T_t['cond'][0, 0] += np.float32(1e-3)
    checks['tamper_changes_hash'] = _sha_arrays(T_t) != sha0
    # 10. timing at the real data size (invented features of the real width; projections for the queue)
    Xb, _ = _invented(rng, 16000, 8000)
    Xb = np.column_stack([Xb] + [np.eye(2)[rng.integers(0, 2, size=16000)] for _ in range(63)]).astype(np.float32)[:, :145]
    yb = _draw(rng, np.zeros((16000, K))); zb = np.zeros((16000, K))
    timing = {}
    for depth in (1, 2, 4):
        s = time.process_time(); Tb = parse_trees(xgb_train(Xb, yb, zb, depth, 40)); fit_s = time.process_time() - s
        s = time.process_time(); round_curve_sum(Tb, Xb[:4000], yb[:4000], zb[:4000]); ev_s = time.process_time() - s
        timing[f'depth{depth}'] = {'fit_s_per_round': fit_s / 40, 'curve_eval_s_per_round_4000_rows': ev_s / 40,
                                   'projected_16_fits_800_rounds_cpu_s': 16 * 800 * (fit_s / 40 + 2.5 * ev_s / 40)}
    out = {'utc': C.now(), 'checks': checks, 'all_pass': all(checks.values()), 'cpu_s': time.process_time() - t0,
           'planted': {'nll_teacher': nll_teacher, 'nll_corrected': nll_corr, 'nll_truth': nll_truth, 'alpha': res['alpha'], 'M': res['M'],
                       'null_nll_base': nll0, 'null_nll_corrected': nll0c, 'null_alpha': res0['alpha'], 'null_M': res0['M']},
           'vscal': {'w': w.tolist(), 'b': b.tolist(), 'temperature': tt, 'alpha_planted_fit': a}, 'timing': timing,
           'interpreter': sys.executable, 'xgboost': xgb.__version__, 'numpy': np.__version__, 'code_sha256': C.sha(Path(__file__)),
           'data': 'invented only (no real record, feature or label read)'}
    print(json.dumps(out, indent=1, default=float))
    if not out['all_pass']:
        raise SystemExit('selftest FAILED: ' + ', '.join(k for k, v in checks.items() if not v))
    return out


def dryrun(seed):
    """Label-free dry run on REAL features with INVENTED labels (no real label is read; nothing is released): recomputation of
    both bases on all six roles against storage, the residual pipeline end to end (small round cap), numpy-vs-xgboost parity on
    real features, cross-fitting outputs and a save / reload round trip of the tree arrays in memory."""
    import io
    import xgboost as xgb
    t0 = time.process_time()
    rng = np.random.default_rng(1000 + seed)
    Xr, Ur, names = {}, {}, None
    base_diff = {}
    for role in C.ROLES6:
        X, nm, u = _features(role)
        names = names or nm
        Xr[role], Ur[role] = X, u
        for base in ('T_TREE', 'TASK_REFIT'):
            zb, _ = base_margin(seed, base, X, nm)
            ref = stored_base_prob(seed, base, role)
            base_diff[f'{base}|{role}'] = float(np.max(np.abs(probs(zb) - ref)))
    z = {r: base_margin(seed, 'T_TREE', Xr[r], names)[0] for r in C.ROLES6}
    plant = lambda X: np.outer(X[:, 0], np.array([0.15, -0.1, 0.0, 0.05, -0.1])) + X[:, [2]] * np.array([0.0, 0.2, -0.2, 0.0, 0.0])
    yf = _draw(rng, z['MODEL_FIT'] + plant(Xr['MODEL_FIT'])); yc = _draw(rng, z['CALIBRATION'] + plant(Xr['CALIBRATION']))
    folds = C.crossfit_folds(Ur['MODEL_FIT'])
    res = residual_pipeline(Xr['MODEL_FIT'], yf, z['MODEL_FIT'], folds, Xr['CALIBRATION'], yc, z['CALIBRATION'], 2, m_max=60)
    T = res['T_full']
    dm = xgb.DMatrix(np.asarray(Xr['MODEL_SELECT'], dtype=np.float32), base_margin=np.zeros((len(Xr['MODEL_SELECT']), K), dtype=np.float32), nthread=1)
    ref = res['bst_full'].predict(dm, output_margin=True).reshape(-1, K)
    g = correction(T, Xr['MODEL_SELECT'])
    buf = io.BytesIO(); np.savez(buf, **T); buf.seek(0)
    with np.load(buf) as zz:
        T2 = {k: zz[k].copy() for k in zz.files}
    oof, fa, _ = crossfit_outputs(res, Xr['MODEL_FIT'], z['MODEL_FIT'], folds, Xr['MODEL_SELECT'], z['MODEL_SELECT'])
    occ = {r: probs(z[r] + res['alpha'] * correction(T, Xr[r])) for r in C.ROLES6}
    out = {'utc': C.now(), 'seed': seed, 'data': 'real features of all six roles, INVENTED labels; no real label read; nothing released',
           'base_recompute_max_abs_diff': base_diff, 'base_ok': all(v <= 1e-12 for v in base_diff.values()),
           'M': res['M'], 'alpha': res['alpha'], 'Mk': res['Mk'], 'alpha_k': res['alpha_k'],
           'parity_max_abs_margin_diff_MODEL_SELECT': float(np.max(np.abs(g - ref))),
           'parity_leaf_ids_MODEL_SELECT': bool(np.array_equal(leaf_index(T, Xr['MODEL_SELECT']), res['bst_full'].predict(dm, pred_leaf=True).reshape(len(g), -1))),
           'reload_bitwise': bool(np.array_equal(correction(T2, Xr['MODEL_SELECT']), g)) and _sha_arrays(T2) == _sha_arrays(T),
           'oof_valid': bool(np.allclose(oof.sum(1), 1, atol=1e-12)), 'foldavg_valid': bool(np.allclose(fa.sum(1), 1, atol=1e-12)),
           'release_valid': all(np.allclose(p.sum(1), 1, atol=1e-12) and np.isfinite(p).all() for p in occ.values()),
           'release_shapes': {r: list(p.shape) for r, p in occ.items()}, 'xgb_fits': res['xgb_fits'], 'cpu_s': time.process_time() - t0}
    print(json.dumps(out, indent=1, default=float))
    if not (out['base_ok'] and out['parity_leaf_ids_MODEL_SELECT'] and out['parity_max_abs_margin_diff_MODEL_SELECT'] <= 1e-4
            and out['reload_bitwise'] and out['oof_valid'] and out['release_valid']):
        raise SystemExit('dry run FAILED')
    return out


def dryrun_full(seed, config_id, scratch, m_max=60):
    """End-to-end dry run of `fit` and `verify` with INVENTED labels on real features, writing only under `scratch` (the run
    directory root and the label loader are redirected in this process; no real label is read; nothing enters the campaign runs)."""
    from . import common as CC
    orig = CC.load_role
    rng = np.random.default_rng(2000 + seed)

    def fake_load_role(role, wanted=('features',), purpose=None):
        lab = [w for w in wanted if w in ('income', 'occupation', 'sex')]
        out = orig(role, tuple(w for w in wanted if w not in lab))
        if lab:
            if lab != ['occupation'] or role not in ('MODEL_FIT', 'CALIBRATION'):
                raise PermissionError('dry run invents occupation labels on MODEL_FIT / CALIBRATION only')
            zz, _ = base_margin(seed, 'T_TREE', out['X'], out['feature_names'])
            out['occupation'] = _draw(rng, zz + 0.1 * np.outer(out['X'][:, 0], np.array([1.0, -1.0, 0.0, 0.5, -0.5])))
        return out
    CC.load_role = fake_load_role
    CC.RUNS = Path(scratch) / 'runs'
    global _DRYRUN_INVENTED_LABELS
    _DRYRUN_INVENTED_LABELS = True
    try:
        rec = fit(seed, config_id, m_max=m_max)
    finally:
        CC.load_role = orig
    print(json.dumps({'dryrun_full': config_id, 'seed': seed, 'scratch': str(CC.RUNS), 'n_calibrator_fits': rec['n_calibrator_fits'],
                      'kind': rec['kind'], 'note': 'INVENTED labels on real features; written to scratch only'}, default=float))


def combo_diagnostic(seed, configs):
    """POST-HOC CALIBRATION-only diagnostic (no fit, nothing released): the frozen TASK_REFIT margin plus a saved family-C correction
    at its saved strength, against TASK_REFIT, T_TREE and the saved release, with household-clustered paired SEs. Reads CALIBRATION
    occupation labels only."""
    Cal = C.load_role('CALIBRATION', ('features', 'occupation'), purpose=f'family C S{seed}: post-hoc combination diagnostic (no fit)')
    X, names, yc, uc = Cal['X'], Cal['feature_names'], Cal['occupation'], Cal['unit']
    zt, _ = base_margin(seed, 'T_TREE', X, names); ztr, _ = base_margin(seed, 'TASK_REFIT', X, names)
    preds = {'teacher': probs(zt), 'TASK_REFIT': probs(ztr)}
    for cfg in configs:
        P, T = load_trees(run_dir(cfg, seed))
        g = correction(T, design(X, names, P.get('inputs', 'X')))
        preds[f'{cfg}'] = probs(zt + float(P['alpha']) * g)
        preds[f'TASK_REFIT+{cfg}'] = probs(ztr + float(P['alpha']) * g)
    out = {'utc': C.now(), 'seed': seed, 'note': 'POST-HOC diagnostic on CALIBRATION (alpha in-sample there); no fit, no release',
           **compare(yc, uc, preds, ['teacher', 'TASK_REFIT'])}
    d = C.RUNS / FAMILY / 'DIAG'; d.mkdir(parents=True, exist_ok=True, mode=0o700)
    (d / f'COMBO_CAL_S{seed}.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(out, indent=1))
    return out


def selftest_reboost():
    """Invented-data checks of the routing-fixed re-estimation (no real data)."""
    rng = np.random.default_rng(7)
    n, T, depth = 3000, 60, 2
    Xb = (rng.random((n, 10)) < 0.4).astype(np.float64)
    feats = rng.integers(0, 10, size=(T, depth))
    leaves = sum(((Xb[:, feats[:, i]] > 0.5).astype(np.int64) << i) for i in range(depth))
    truth = Xb[:, :5] @ rng.normal(size=(5, K)) * 0.8
    y = _draw(rng, truth)
    vals = reboost(leaves, y, np.arange(n), T, n_leaves=2 ** depth)
    checks = {}
    checks['leaf_values_centred'] = bool(np.abs(vals.sum(2)).max() < 1e-12)
    raw = reboost_raw(vals, leaves)
    checks['training_nll_decreases'] = bool(nll_mean(y, raw) < nll_mean(y, np.zeros((n, K))) - 0.01)
    p0 = np.full(K, 1.0 / K); Y = np.eye(K)[y]; ok = True
    for l in range(2 ** depth):                              # first tree: independent per-leaf Newton solution at p = 1/K
        idx = leaves[:, 0] == l
        if not idx.any():
            continue
        Gl = (p0[None, :] - Y[idx]).sum(0); Hl = idx.sum() * (np.diag(p0) - np.outer(p0, p0)) + 3.0 * np.eye(K)
        ok &= bool(np.allclose(vals[0, l], -0.05 * np.linalg.solve(Hl, Gl), atol=1e-12))
    checks['first_tree_newton_step'] = ok
    fold = rng.integers(0, 5, size=n); yp = y.copy(); yp[fold == 0] = rng.permutation(yp[fold == 0])
    a = reboost(leaves, y, np.where(fold != 0)[0], T, n_leaves=2 ** depth); b = reboost(leaves, yp, np.where(fold != 0)[0], T, n_leaves=2 ** depth)
    checks['out_of_fold_independent_of_fold_labels'] = bool(np.array_equal(a, b))
    out = {'utc': C.now(), 'checks': checks, 'all_pass': all(checks.values()), 'data': 'invented only'}
    print(json.dumps(out, indent=1))
    if not out['all_pass']:
        raise SystemExit('reboost selftest FAILED')
    return out


def selftest_xf():
    """Invented-data checks of the revision-1 pipeline (cross-fitted base margins; no real data)."""
    rng = np.random.default_rng(11)
    t0 = time.process_time()
    n, T, depth = 2500, 40, 3
    Xb = np.column_stack([rng.normal(size=(n, 2)), (rng.random((n, 12)) < 0.35)]).astype(np.float32)
    feats = rng.integers(2, 14, size=(T, depth))
    leaves = sum(((Xb[:, feats[:, i]] > 0.5).astype(np.int64) << i) for i in range(depth))
    truth = Xb[:, 2:7] @ rng.normal(size=(5, K)) * 0.7 + np.outer(np.tanh(Xb[:, 0]), np.array([0.4, -0.4, 0.0, 0.2, -0.2]))
    y = _draw(rng, truth)
    unit = np.sort(rng.integers(0, 1500, size=n)); folds = C.household_folds(unit, 5, 'OHN1_famC_selftest_xf')
    Xc = Xb[:600]; yc = _draw(rng, truth[:600]); Lc = leaves[:600]           # invented "calibration" people (re-drawn labels)
    def run(yy):
        V = xf_leaf_values(leaves, yy, folds, T, max_excl=3)
        zc = xf_margin(V, set(), Lc)
        res = residual_pipeline_xf(Xb, yy, leaves, folds, V, Xc, yc, zc, Lc, 2, m_max=40, log=lambda *a: None)
        oof, fa = crossfit_outputs_xf(res, V, Xb, leaves, folds, Xb[:300], leaves[:300])
        return V, res, oof, fa
    V, res, oof, fa = run(y)
    checks = {'reestimation_count_26': len(V) == 26,
              'oof_valid': bool(np.allclose(oof.sum(1), 1, atol=1e-12) and (oof >= 0).all()),
              'foldavg_valid': bool(np.allclose(fa.sum(1), 1, atol=1e-12))}
    zin = xf_margin(V, set(), leaves); zoof = np.empty_like(zin)
    for k in range(5):
        zoof[folds == k] = xf_margin(V, {k}, leaves[folds == k])
    checks['in_sample_optimism_visible'] = bool(nll_mean(y, zoof) > nll_mean(y, zin))
    yp = y.copy(); m0 = folds == 0; yp[m0] = rng.permutation(yp[m0])
    _, resp, oofp, _ = run(yp)
    checks['oof_fold0_independent_of_fold0_labels'] = bool(np.array_equal(oof[m0], oofp[m0]))
    checks['oof_other_folds_change'] = bool(not np.array_equal(oof[~m0], oofp[~m0]))
    checks['fold0_pipeline_choices_unchanged'] = bool(resp['Mk'][0] == res['Mk'][0] and resp['alpha_k'][0] == res['alpha_k'][0])
    out = {'utc': C.now(), 'checks': checks, 'all_pass': all(checks.values()), 'M': res['M'], 'Mk': res['Mk'], 'alpha': res['alpha'],
           'cpu_s': time.process_time() - t0, 'data': 'invented only'}
    print(json.dumps(out, indent=1, default=float))
    if not out['all_pass']:
        raise SystemExit('xf selftest FAILED')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('selftest_reboost')
    sub.add_parser('selftest_xf')
    sub.add_parser('verifyall')
    p = sub.add_parser('exposure'); p.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('combo'); p.add_argument('--seed', type=int, required=True); p.add_argument('--configs', nargs='+', required=True)
    p = sub.add_parser('dryrun'); p.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('dryrun_full'); p.add_argument('--seed', type=int, required=True); p.add_argument('--config', required=True)
    p.add_argument('--scratch', required=True); p.add_argument('--m-max', type=int, default=60)
    for name in ('fit', 'verify'):
        p = sub.add_parser(name); p.add_argument('--seed', type=int, required=True); p.add_argument('--config', required=True, choices=sorted(CONFIGS))
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        selftest()
    elif a.cmd == 'selftest_reboost':
        selftest_reboost()
    elif a.cmd == 'selftest_xf':
        selftest_xf()
    elif a.cmd == 'verifyall':
        res = {}
        for d in sorted((C.RUNS / FAMILY).glob('*/S*')):
            if (d / 'OCC.npz').is_file():
                res[f'{d.parent.name}/S{d.name[1:]}'] = verify(int(d.name[1:]), d.parent.name)['all_bitwise']
        print(json.dumps({'utc': C.now(), 'code_sha256': C.sha(Path(__file__)), 'releases': res, 'all_bitwise': all(res.values())}, indent=1))
        if not all(res.values()):
            raise SystemExit('verifyall FAILED')
    elif a.cmd == 'exposure':
        exposure_diagnostic(a.seed)
    elif a.cmd == 'combo':
        combo_diagnostic(a.seed, a.configs)
    elif a.cmd == 'dryrun':
        dryrun(a.seed)
    elif a.cmd == 'dryrun_full':
        dryrun_full(a.seed, a.config, a.scratch, a.m_max)
    elif a.cmd == 'fit':
        fit(a.seed, a.config)
    else:
        print(json.dumps(verify(a.seed, a.config), default=float))


if __name__ == '__main__':
    main()
