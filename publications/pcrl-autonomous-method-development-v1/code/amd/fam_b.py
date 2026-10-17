"""Family B: coupled shared-statistic release (CSR). PCRL autonomous method-development campaign (EXPLORATORY DEVELOPMENT; not
confirmation).

Census context: public-use Florida ACS 2019 person records (development roles of the protected-tree replication). The service
predicts each person's income class (2 classes) and occupation group (5 classes) from the six permitted raw fields (AGEP, WKHP,
COW, SCHL, MAR, POBP; admitted 145-column encoding) and gives each recipient only its own float64 probability vector and argmax
decision. Evaluation readers try to infer the dataset's recorded SEX column from the released outputs (each output alone, and
the two outputs aligned to the same record: the linked pair).

Mechanism (MECHANISM_CARDS.md, Family B):
  occupation release  p_occ(x) = a BASE occupation output computed from the 145 input columns only: the replication's frozen
                      CatBoost teacher T_TREE (explicit routed trees) plus, for LOCAL_TREE / JOINT_TREE, its stored E20 leaf-score
                      corrections theta (amd.common.stage_b_theta). Wire: probabilities_from_raw(raw_from_leaves(route(X), theta)).
  income release      p_inc(x) = softmax(-s/2, +s/2) with s(x) = h(phi(p_occ(x)), c(x)): h is a low-capacity L2 logistic map
                      fitted on CALIBRATION (features -> base outputs; income labels; no SEX); phi = the occupation output's
                      log-ratios to the last class (4 columns, 'lin') optionally with their degree-2 monomials ('quad');
                      c(x) in {0..m-1} is an optional coarse income statistic: m bins of the LOCAL_TREE income log-odds
                      (thresholds = label-free MODEL_FIT quantiles, or the decision threshold 0 for m = 2), one-hot into h.
  m = 1 (no c): the linked pair is a deterministic function of the occupation output alone.
  m > 1: the pair is a deterministic function of (occupation output, c); pair information <= occupation-output
         information + log m.
SEX is never a runtime input and never a label for h. MODEL_FIT SEX was used only by the stored base corrections (the
replication's LOCAL_TREE / JOINT_TREE critics, Stage B). MODEL_SELECT is never read here; it is scored only by amd.devrun.

Commands
  diag --seed s           internal diagnostic on CALIBRATION (income labels only, household 5-fold cross-fitting) and
                          MODEL_FIT -> CALIBRATION: income utility of h for several bases / feature maps / m. No SEX, no
                          MODEL_SELECT. Writes runs/B/DIAG/DIAG_S<s>.json.
  fit --seed s --config ID --base JOINT_TREE --phi lin --m 1 [--coarse-kind quantile|decision] [--C 1.0]
                          fits h on all of CALIBRATION and saves the release for every role (amd.devrun.save_release).
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np

from acs_replication import trees
from . import common as C

ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
BASES = ('T_TREE', 'LOCAL_TREE', 'JOINT_TREE')
FAMILY = 'B'
DIAG_DIR = C.RUNS / 'B' / 'DIAG'


# ---------------------------------------------------------------- base outputs (teacher routing + stored corrections)
def load_X(role):
    F = C.load_role(role, ('features',))
    return F['X'], F['feature_names'], F['row_id'], F['unit']


def base_raw(seed, arm, X, names, label='E20'):
    """Raw scores of both recipients for base arm (T_TREE: theta = None)."""
    thetas = None if arm == 'T_TREE' else C.stage_b_theta(seed, arm, label)
    out = {}
    for i, t in enumerate(C.TASKS):
        te = C.teacher(seed, t)
        out[t] = te.raw_from_leaves(te.route(X, names), None if thetas is None else thetas[i])
    return out


def stored_outputs(seed):
    sa = C.stage_a_outputs(seed)
    with np.load(C.admitted(f'stage_b/S{seed}/STAGE_B_OUTPUTS_E20.npz')) as z:
        sb = {k: z[k].copy() for k in z.files}
    return sa, sb


def identity_check(seed, arm, role, probs, row_id, sa, sb):
    """Recomputed base probabilities against the stored replication outputs (<= 1e-12, decisions identical)."""
    if not np.array_equal(sa[f'row_id__{role}'], row_id):
        raise AssertionError(f'{role}: row order differs from the stored Stage A outputs')
    rec = {}
    for t in C.TASKS:
        ref = sa[f'T_TREE__{role}__{t}'] if arm == 'T_TREE' else sb[f'{arm}__{role}__{t}']
        d = float(np.max(np.abs(probs[t] - ref)))
        same = bool(np.array_equal(probs[t].argmax(1), ref.argmax(1)))
        if d > 1e-12 or not same:
            raise AssertionError(f'{arm} {role} {t}: recomputed output differs from stored ({d}, decisions {same})')
        rec[t] = d
    return rec


def base_outputs(seed, arm, roles):
    """{role: {'raw': {...}, 'prob': {...}, 'row_id', 'unit'}} plus the identity record."""
    sa, sb = stored_outputs(seed)
    out, ident = {}, {}
    for role in roles:
        X, names, rid, unit = load_X(role)
        raw = base_raw(seed, arm, X, names)
        prob = {t: trees.probabilities(t, raw[t]) for t in C.TASKS}
        ident[role] = identity_check(seed, arm, role, prob, rid, sa, sb)
        out[role] = {'raw': raw, 'prob': prob, 'row_id': rid, 'unit': unit}
    return out, ident


# ---------------------------------------------------------------- h: feature map and logistic fit
def phi(p_occ, kind):
    lp = np.log(np.maximum(np.asarray(p_occ, dtype=np.float64), 1e-300))
    r = lp[:, :4] - lp[:, 4:5]                      # log-ratios to the last occupation class (4 columns; exact rank)
    if kind == 'none':                               # uncoupled control: income does not depend on the occupation output
        return np.zeros((len(r), 0))
    if kind == 'lin':
        return r
    if kind == 'quad':
        quad = [r[:, i] * r[:, j] for i, j in itertools.combinations_with_replacement(range(4), 2)]
        return np.column_stack([r] + quad)
    raise ValueError(kind)


def coarse_bins(d_local, thresholds):
    return np.searchsorted(np.asarray(thresholds, dtype=np.float64), np.asarray(d_local, dtype=np.float64).reshape(-1), side='left')   # bin = #thresholds strictly below (decision: class 1 iff d > 0)


def design(p_occ, kind, c=None, m=1, d=None, stat='bins'):
    """h design: phi(occupation output) [+ one-hot coarse bins c (stat 'bins', m > 1)] [+ continuous income statistic d
    (stat 'cont') [+ d x log-ratio interactions (stat 'cont_int')]]."""
    Z = phi(p_occ, kind)
    if stat == 'bins' and m > 1:
        one = np.eye(m)[np.asarray(c, dtype=np.int64)][:, 1:]
        Z = np.column_stack((Z, one))
    elif stat in ('cont', 'cont_int'):
        d = np.asarray(d, dtype=np.float64).reshape(-1, 1)
        Z = np.column_stack((Z, d) if stat == 'cont' else (Z, d, d * phi(p_occ, 'lin')))
    elif stat not in ('none', 'bins'):
        raise ValueError(stat)
    return Z


def thresholds(d_fit, m, coarse_kind):
    """Label-free bin thresholds from MODEL_FIT log-odds: 'quantile' (m quantiles), 'decision' (m = 2: threshold 0),
    'aligned' (m even: m/2 quantile bins on each side of the decision threshold 0, so bins never straddle the decision)."""
    d_fit = np.asarray(d_fit, dtype=np.float64).reshape(-1)
    if coarse_kind == 'decision':
        return [0.0]
    if coarse_kind == 'quantile':
        return [float(q) for q in np.quantile(d_fit, np.arange(1, m) / m)]
    if coarse_kind == 'aligned':
        if m % 2:
            raise ValueError('aligned bins need even m')
        k = m // 2
        neg, pos = d_fit[d_fit <= 0], d_fit[d_fit > 0]
        return ([float(q) for q in np.quantile(neg, np.arange(1, k) / k)] + [0.0] +
                [float(q) for q in np.quantile(pos, np.arange(1, k) / k)])
    raise ValueError(coarse_kind)


class H:
    """Standardized L2 logistic map; stored as plain coefficients so deployment is numpy only."""

    def __init__(self, Creg=1.0):
        self.Creg = Creg

    def fit(self, Z, y):
        from sklearn.linear_model import LogisticRegression
        self.mu = Z.mean(0); self.sd = Z.std(0); self.sd[self.sd == 0] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = LogisticRegression(C=self.Creg, max_iter=5000, solver='lbfgs', tol=1e-10).fit((Z - self.mu) / self.sd, y)
        self.w = m.coef_.reshape(-1).astype(np.float64); self.b = float(m.intercept_[0])
        return self

    def logit(self, Z):
        return ((Z - self.mu) / self.sd) @ self.w + self.b

    def prob(self, Z):
        return trees.probabilities('income', self.logit(Z))

    def state(self):
        return {'C': self.Creg, 'mu': self.mu.tolist(), 'sd': self.sd.tolist(), 'w': self.w.tolist(), 'b': self.b}


def metrics(y, p, k):
    v = C.task_vectors(np.asarray(y), np.asarray(p, dtype=np.float64), k)
    return {m: float(np.mean(a)) for m, a in v.items()}


# ---------------------------------------------------------------- internal diagnostic (CALIBRATION labels only)
def diag(seed):
    t0 = time.process_time()
    Cal = C.load_role('CALIBRATION', ('features', 'income', 'occupation'))
    Fit = C.load_role('MODEL_FIT', ('features', 'income'))      # MODEL_FIT income labels: for the MODEL_FIT -> CAL variant only
    sa = C.stage_a_outputs(seed)
    y_cal, y_fit = Cal['income'], Fit['income']
    folds = C.household_folds(Cal['unit'], 5, f'AMD_B_calcv_S{seed}')
    ref = {name: metrics(y_cal, sa[f'{name}__CALIBRATION__income'], 2) for name in ('U0', 'Ucal', 'T_TREE')}
    occ_ref = {name: metrics(Cal['occupation'], sa[f'{name}__CALIBRATION__occupation'], 5) for name in ('U0', 'Ucal', 'T_TREE')}
    bases = {}
    for arm in BASES:
        bo, ident = base_outputs(seed, arm, ('MODEL_FIT', 'CALIBRATION'))
        bases[arm] = bo
        ref[f'{arm}_own_income'] = metrics(y_cal, bo['CALIBRATION']['prob']['income'], 2)
        occ_ref[f'{arm}_occupation'] = metrics(Cal['occupation'], bo['CALIBRATION']['prob']['occupation'], 5)
    d_local_fit = bases['LOCAL_TREE']['MODEL_FIT']['raw']['income'].reshape(-1)
    d_local_cal = bases['LOCAL_TREE']['CALIBRATION']['raw']['income'].reshape(-1)
    rows = []

    def cv(Z, tag, Creg=1.0):
        p = np.zeros((len(y_cal), 2))
        for f in range(5):
            tr, te = folds != f, folds == f
            p[te] = H(Creg).fit(Z[tr], y_cal[tr]).prob(Z[te])
        r = metrics(y_cal, p, 2)
        rows.append({'variant': tag, **r, **{f'nll_excess_vs_{k}': r['nll'] - ref[k]['nll'] for k in ('U0', 'Ucal', 'T_TREE')},
                     **{f'brier_excess_vs_{k}': r['brier'] - ref[k]['brier'] for k in ('U0', 'Ucal')},
                     **{f'acc_loss_vs_{k}': ref[k]['accuracy'] - r['accuracy'] for k in ('U0', 'Ucal')}})

    for arm in BASES:
        for kind in ('lin', 'quad'):
            Zc = design(bases[arm]['CALIBRATION']['prob']['occupation'], kind)
            cv(Zc, f'{arm}|{kind}|m1|CAL-CV5')
            Zf = design(bases[arm]['MODEL_FIT']['prob']['occupation'], kind)
            p = H().fit(Zf, y_fit).prob(Zc); r = metrics(y_cal, p, 2)
            rows.append({'variant': f'{arm}|{kind}|m1|FIT->CAL', **r, **{f'nll_excess_vs_{k}': r['nll'] - ref[k]['nll'] for k in ('U0', 'Ucal', 'T_TREE')},
                         **{f'brier_excess_vs_{k}': r['brier'] - ref[k]['brier'] for k in ('U0', 'Ucal')},
                         **{f'acc_loss_vs_{k}': ref[k]['accuracy'] - r['accuracy'] for k in ('U0', 'Ucal')}})
    # coarse income statistic on the JOINT_TREE occupation base (and LOCAL base for reference)
    for arm in ('JOINT_TREE', 'LOCAL_TREE'):
        for kind in ('lin', 'quad'):
            for m, ck in ((2, 'decision'), (2, 'quantile'), (3, 'quantile'), (4, 'quantile'), (6, 'quantile'), (8, 'quantile')):
                th = [0.0] if ck == 'decision' else list(np.quantile(d_local_fit, np.arange(1, m) / m))
                c = coarse_bins(d_local_cal, th)
                cv(design(bases[arm]['CALIBRATION']['prob']['occupation'], kind, c, m), f'{arm}|{kind}|m{m}{ck[0]}|CAL-CV5')
            # coarse statistic alone (no coupling to occupation): ordinary coarsening of LOCAL income, calibrated
            if kind == 'lin':
                for m in (2, 4, 8):
                    th = list(np.quantile(d_local_fit, np.arange(1, m) / m))
                    c = coarse_bins(d_local_cal, th)
                    cv(np.eye(m)[c][:, 1:], f'COARSE_ONLY|m{m}q|CAL-CV5')
    # uncoupled controls on CALIBRATION (no fit)
    res = {'seed': seed, 'role_labels_read': {'CALIBRATION': ['income', 'occupation'], 'MODEL_FIT': ['income (FIT->CAL variant only)']},
           'sex_read': False, 'model_select_read': False, 'references_on_CALIBRATION_income': ref,
           'references_on_CALIBRATION_occupation': occ_ref, 'rows': rows, 'cpu_s': time.process_time() - t0,
           'note': 'CALIBRATION n=2982 (1974 households); NLL differences have SE of order 0.003; indicative only. Ucal temperature was fitted on CALIBRATION.'}
    C.write_json(DIAG_DIR / f'DIAG_S{seed}.json', res)
    print(f"seed {seed} CAL income refs:", {k: round(v['nll'], 4) for k, v in ref.items()})
    print(f"seed {seed} CAL occupation refs:", {k: round(v['nll'], 4) for k, v in occ_ref.items()})
    for r in rows:
        print(f"{r['variant']:<34} nll {r['nll']:.4f}  xs_U0 {r['nll_excess_vs_U0']:+.4f} xs_Ucal {r['nll_excess_vs_Ucal']:+.4f}  "
              f"brier_xs_U0 {r['brier_excess_vs_U0']:+.4f}  acc_loss_U0 {r['acc_loss_vs_U0']:+.4f}")
    return res


# ---------------------------------------------------------------- internal diagnostic 2: an income-specific statistic in h
def diag2(seed):
    """CALIBRATION-only (income labels; no SEX; no MODEL_SELECT): occupation base JOINT_TREE; income statistic = JOINT_TREE or
    LOCAL_TREE income log-odds (continuous, with / without interactions, or binned); household 5-fold cross-fitting."""
    t0 = time.process_time()
    Cal = C.load_role('CALIBRATION', ('features', 'income'))
    sa = C.stage_a_outputs(seed)
    y = Cal['income']
    folds = C.household_folds(Cal['unit'], 5, f'AMD_B_calcv_S{seed}')
    ref = {name: metrics(y, sa[f'{name}__CALIBRATION__income'], 2) for name in ('U0', 'Ucal', 'T_TREE')}
    bo = {arm: base_outputs(seed, arm, ('MODEL_FIT', 'CALIBRATION'))[0] for arm in ('JOINT_TREE', 'LOCAL_TREE')}
    for arm in bo:
        ref[f'{arm}_own_income'] = metrics(y, bo[arm]['CALIBRATION']['prob']['income'], 2)
    p_occ = bo['JOINT_TREE']['CALIBRATION']['prob']['occupation']
    rows = []

    def row(tag, p):
        r = metrics(y, p, 2)
        rows.append({'variant': tag, **r, **{f'nll_excess_vs_{k}': r['nll'] - ref[k]['nll'] for k in ('U0', 'Ucal', 'T_TREE')},
                     **{f'brier_excess_vs_{k}': r['brier'] - ref[k]['brier'] for k in ('U0', 'Ucal')},
                     **{f'acc_loss_vs_{k}': ref[k]['accuracy'] - r['accuracy'] for k in ('U0', 'Ucal')}})

    def cv(Z, tag, Creg=1.0):
        p = np.zeros((len(y), 2))
        for f in range(5):
            tr, te = folds != f, folds == f
            p[te] = H(Creg).fit(Z[tr], y[tr]).prob(Z[te])
        row(tag, p)

    for src in ('JOINT_TREE', 'LOCAL_TREE'):
        d_cal = bo[src]['CALIBRATION']['raw']['income'].reshape(-1); d_fit = bo[src]['MODEL_FIT']['raw']['income'].reshape(-1)
        row(f'{src}_income_raw (no fit)', bo[src]['CALIBRATION']['prob']['income'])
        cv(d_cal[:, None], f'{src}|recal only (h(d))')
        for kind in ('lin', 'quad'):
            for stat in ('cont', 'cont_int'):
                cv(design(p_occ, kind, d=d_cal, stat=stat), f'{src}|occJ {kind}|{stat}')
        for m, ck in ((4, 'aligned'), (6, 'aligned'), (8, 'aligned'), (12, 'aligned'), (16, 'aligned')):
            c = coarse_bins(d_cal, thresholds(d_fit, m, ck))
            cv(design(p_occ, 'lin', c, m), f'{src}|occJ lin|bins m{m} {ck}')
            cv(np.eye(m)[c][:, 1:], f'{src}|bins only m{m} {ck}')
    res = {'seed': seed, 'role_labels_read': {'CALIBRATION': ['income']}, 'sex_read': False, 'model_select_read': False,
           'references_on_CALIBRATION_income': ref, 'rows': rows, 'cpu_s': time.process_time() - t0,
           'note': 'CALIBRATION n=2982; indicative only (NLL difference SE ~0.003).'}
    C.write_json(DIAG_DIR / f'DIAG2_S{seed}.json', res)
    print(f"seed {seed} CAL income refs:", {k: round(v['nll'], 4) for k, v in ref.items()},
          {k: round(v['brier'], 4) for k, v in ref.items()}, {k: round(v['accuracy'], 4) for k, v in ref.items()})
    for r in rows:
        print(f"{r['variant']:<40} nll {r['nll']:.4f}  xs_U0 {r['nll_excess_vs_U0']:+.4f} xs_Ucal {r['nll_excess_vs_Ucal']:+.4f}  "
              f"brier_xs_U0 {r['brier_excess_vs_U0']:+.4f} brier_xs_Ucal {r['brier_excess_vs_Ucal']:+.4f}  acc_loss_U0 {r['acc_loss_vs_U0']:+.4f}")
    return res


# ---------------------------------------------------------------- internal diagnostic 3: pair-aware mixture income statistic
def diag3(seed):
    """CALIBRATION-only (income labels; no SEX; no MODEL_SELECT): occupation base JOINT_TREE; income statistic
    d_mix = (1 - a) d_LOCAL + a d_JOINT (E20 income log-odds), m aligned bins (label-free MODEL_FIT thresholds), coupled
    (phi lin) or uncoupled (phi none); household 5-fold cross-fitting."""
    t0 = time.process_time()
    Cal = C.load_role('CALIBRATION', ('features', 'income'))
    sa = C.stage_a_outputs(seed)
    y = Cal['income']
    folds = C.household_folds(Cal['unit'], 5, f'AMD_B_calcv_S{seed}')
    ref = {name: metrics(y, sa[f'{name}__CALIBRATION__income'], 2) for name in ('U0', 'Ucal', 'T_TREE')}
    bo = {arm: base_outputs(seed, arm, ('MODEL_FIT', 'CALIBRATION'))[0] for arm in ('JOINT_TREE', 'LOCAL_TREE')}
    p_occ = bo['JOINT_TREE']['CALIBRATION']['prob']['occupation']
    rows = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        dm = {r: (1 - a) * bo['LOCAL_TREE'][r]['raw']['income'].reshape(-1) + a * bo['JOINT_TREE'][r]['raw']['income'].reshape(-1)
              for r in ('MODEL_FIT', 'CALIBRATION')}
        for m in (6, 8):
            c = coarse_bins(dm['CALIBRATION'], thresholds(dm['MODEL_FIT'], m, 'aligned'))
            for kind in ('none', 'lin'):
                Z = design(p_occ, kind, c, m)
                p = np.zeros((len(y), 2))
                for f in range(5):
                    tr, te = folds != f, folds == f
                    p[te] = H().fit(Z[tr], y[tr]).prob(Z[te])
                r = metrics(y, p, 2)
                rows.append({'alpha': a, 'm': m, 'phi': kind, **r, 'nll_excess_vs_U0': r['nll'] - ref['U0']['nll'],
                             'nll_excess_vs_Ucal': r['nll'] - ref['Ucal']['nll'], 'brier_excess_vs_U0': r['brier'] - ref['U0']['brier'],
                             'brier_excess_vs_Ucal': r['brier'] - ref['Ucal']['brier'], 'acc_loss_vs_U0': ref['U0']['accuracy'] - r['accuracy']})
    res = {'seed': seed, 'role_labels_read': {'CALIBRATION': ['income']}, 'sex_read': False, 'model_select_read': False,
           'references_on_CALIBRATION_income': ref, 'rows': rows, 'cpu_s': time.process_time() - t0}
    C.write_json(DIAG_DIR / f'DIAG3_S{seed}.json', res)
    for r in rows:
        print(f"a {r['alpha']:.2f} m {r['m']} phi {r['phi']:<4} nll {r['nll']:.4f} xs_U0 {r['nll_excess_vs_U0']:+.4f} xs_Ucal {r['nll_excess_vs_Ucal']:+.4f} "
              f"brier_xs_U0 {r['brier_excess_vs_U0']:+.4f} brier_xs_Ucal {r['brier_excess_vs_Ucal']:+.4f} acc_loss_U0 {r['acc_loss_vs_U0']:+.4f}")
    return res


# ---------------------------------------------------------------- release fit
def stat_logodds(seed, stat_source, alpha, roles, cache):
    """Income log-odds used as the income-specific statistic: one stored arm, or 'MIX' = (1 - alpha) LOCAL_TREE + alpha JOINT_TREE."""
    arms = ('LOCAL_TREE', 'JOINT_TREE') if stat_source == 'MIX' else (stat_source,)
    ident = {}
    for arm in arms:
        if arm not in cache:
            cache[arm] = base_outputs(seed, arm, roles)
        ident[arm] = cache[arm][1]
    if stat_source == 'MIX':
        d = {r: (1 - alpha) * cache['LOCAL_TREE'][0][r]['raw']['income'].reshape(-1) + alpha * cache['JOINT_TREE'][0][r]['raw']['income'].reshape(-1) for r in roles}
    else:
        d = {r: cache[stat_source][0][r]['raw']['income'].reshape(-1) for r in roles}
    return d, ident


def fit(seed, config, base, kind, m, coarse_kind, Creg, version_note, stat='bins', stat_source='LOCAL_TREE', alpha=0.0):
    """Fit h on all of CALIBRATION and save the release for every role. stat: 'none' (m = 1, pure coupling), 'bins' (m coarse
    levels of the stat_source income log-odds), 'cont' / 'cont_int' (continuous stat_source income log-odds)."""
    t0 = time.process_time()
    if stat == 'none':
        m = 1
    bo, ident = base_outputs(seed, base, ROLES)
    dstat, th = None, []
    if stat != 'none':
        dstat, ident_s = stat_logodds(seed, stat_source, alpha, ROLES, {base: (bo, ident)})
        ident = {'base': ident, f'{stat_source}_income_statistic': ident_s}
        if stat == 'bins':
            th = thresholds(dstat['MODEL_FIT'], m, coarse_kind)
    Cal = C.load_role('CALIBRATION', ('features', 'income'))
    if not np.array_equal(Cal['row_id'], bo['CALIBRATION']['row_id']):
        raise AssertionError('CALIBRATION row order differs')

    def Z(role):
        d = None if dstat is None else dstat[role]
        c = coarse_bins(d, th) if stat == 'bins' else None
        return design(bo[role]['prob']['occupation'], kind, c, m, d, stat), c

    Zc, _ = Z('CALIBRATION')
    h = H(Creg).fit(Zc, Cal['income'])
    outs, cells = {}, {}
    for role in ROLES:
        Zr, c = Z(role)
        outs[role] = {'income': h.prob(Zr), 'occupation': bo[role]['prob']['occupation']}
        if c is not None:
            cells[role] = np.bincount(c, minlength=m).tolist()
    cal_fit = metrics(Cal['income'], outs['CALIBRATION']['income'], 2)
    src_name = f'MIX = {1 - alpha} x LOCAL_TREE + {alpha} x JOINT_TREE E20' if stat_source == 'MIX' else f'{stat_source} E20'
    stat_def = {'none': None,
                'bins': f'c = bins of the {src_name} income log-odds (m = {m}, {coarse_kind}); thresholds {json.dumps(th)} (label-free MODEL_FIT log-odds); one-hot into h',
                'cont': f'continuous {src_name} income log-odds d as one column of h',
                'cont_int': f'continuous {src_name} income log-odds d and d x log-ratio interactions'}[stat]
    record = {
        'mechanism': 'Family B CSR: occupation = base output; income = softmax(-s/2, s/2), s = h(phi(occupation output)[, income statistic])',
        'base_occupation': base,
        'base_wire': 'T_TREE routing (frozen CatBoost teacher) + stored E20 leaf corrections (replication Stage B)' if base != 'T_TREE' else 'T_TREE (frozen CatBoost teacher)',
        'phi': kind, 'phi_definition': ('none: income does not depend on the occupation output (UNCOUPLED control)' if kind == 'none' else
                                        'log p_k - log p_4 of the released occupation vector, k = 0..3' + (' plus all degree-2 monomials' if kind == 'quad' else '')),
        'income_statistic': stat, 'income_statistic_source': stat_source, 'mix_alpha': alpha if stat_source == 'MIX' else None,
        'income_statistic_definition': stat_def, 'coarse_m': m if stat == 'bins' else None, 'coarse_kind': coarse_kind if stat == 'bins' else None,
        'thresholds': th,
        'pair_information_bound': ('pair is a deterministic function of the occupation output' if stat == 'none' else
                                   f'pair is a deterministic function of (occupation output, c); excess over the occupation output <= log {m}' if stat == 'bins' else
                                   f'pair is a deterministic function of (occupation output, {src_name} income log-odds)'),
        'h': {'form': 'L2 logistic regression (sklearn lbfgs, tol 1e-10) on the standardized design', **h.state()},
        'h_fit_role': 'CALIBRATION (all 2982 rows; features -> base outputs; income labels; no SEX)',
        'labels_used_at_training': {'CALIBRATION': ['income'],
                                    'MODEL_FIT': 'none read by this code; the stored base corrections (replication Stage B LOCAL_TREE / JOINT_TREE) used MODEL_FIT income, occupation and SEX labels; coarse thresholds use MODEL_FIT features only'},
        'runtime_inputs': 'the admitted 145-column encoding of AGEP, WKHP, COW, SCHL, MAR, POBP; SEX is never an input',
        'identity_check_max_abs_diff': ident, 'calibration_in_sample_income': cal_fit, 'coarse_cell_counts': cells,
        'note': version_note, 'fit_cpu_s': time.process_time() - t0}
    from . import devrun
    d = devrun.save_release(FAMILY, config, seed, outs, record)
    print(json.dumps({'saved': str(d), 'cal_income_in_sample': cal_fit, 'coarse_cells_CAL': cells.get('CALIBRATION'),
                      'coarse_cells_MS': cells.get('MODEL_SELECT'), 'fit_cpu_s': record['fit_cpu_s']}))
    return d


# ---------------------------------------------------------------- deployment package (raw permitted fields -> released vectors)
PKG_SCHEMA = 'AMD1-famB-package-v1'


def _release_config(rec):
    """Release configuration from RELEASE.json (older B1/B2 records predate the explicit keys; defaults reproduce their fit)."""
    stat = rec.get('income_statistic') or 'none'
    src = rec.get('income_statistic_source')
    if src is None and stat != 'none':
        src = 'LOCAL_TREE' if 'LOCAL_TREE E20' in (rec.get('income_statistic_definition') or '') else 'T_TREE'
    return {'base': rec['base_occupation'], 'phi': rec['phi'], 'stat': stat, 'stat_source': src,
            'alpha': float(rec.get('mix_alpha') or 0.0), 'm': int(rec.get('coarse_m') or 1),
            'coarse_kind': rec.get('coarse_kind') or ('aligned' if stat == 'bins' else None)}


def _arms_needed(cfg):
    arms = set()
    if cfg['base'] != 'T_TREE':
        arms.add(cfg['base'])
    if cfg['stat'] != 'none':
        arms |= {'LOCAL_TREE', 'JOINT_TREE'} if cfg['stat_source'] == 'MIX' else ({cfg['stat_source']} - {'T_TREE'})
    return sorted(arms)


def build_package(name, seed, config_id, out_root):
    """Deployment package for one family-B release: frozen encoder (the replication's T_TREE package encoder), admitted teacher
    JSONs, the stored E20 theta files the release uses, h coefficients and bin thresholds (H.npz), and a sha256 map."""
    import shutil
    from . import devrun
    d = Path(out_root) / f'{name}_S{seed}'
    if d.exists():
        raise PermissionError('package exists')
    _, rec = devrun.load_release(FAMILY, config_id, seed, roles=())
    cfg = _release_config(rec)
    th = []
    if cfg['stat'] == 'bins':                                          # label-free recomputation, cross-checked with the record
        dstat, _ = stat_logodds(seed, cfg['stat_source'], cfg['alpha'], ('MODEL_FIT',), {})
        th = thresholds(dstat['MODEL_FIT'], cfg['m'], cfg['coarse_kind'])
        if 'thresholds' in rec and [float(x) for x in rec['thresholds']] != th:
            raise AssertionError('recomputed thresholds differ from the release record')
        if json.dumps(th) not in (rec.get('income_statistic_definition') or rec.get('coarse_definition') or ''):
            raise AssertionError('recomputed thresholds not found in the release record definition')
    d.mkdir(parents=True, mode=0o700)
    enc = json.loads(C.admitted(f'packages/T_TREE_S{seed}/PACKAGE.json').read_text())['encoder']
    meta = {'schema': PKG_SCHEMA, 'kind': 'FAMB', 'release_module': 'amd.fam_b', 'service': name, 'seed': seed, 'encoder': enc,
            'teachers': {}, 'thetas': {}, 'config': cfg, 'h': 'H.npz', 'files': {}, 'campaign': C.CAMPAIGN,
            'release': f'{FAMILY}/{config_id}', 'release_sha256': rec['release_sha256'],
            'outputs': 'income and occupation probability vectors and argmax decisions only'}
    import shutil as _sh
    for t in C.TASKS:
        src = C.admitted(f'stage_a/S{seed}/TREE_TEACHER_{t}.json'); _sh.copy2(src, d / src.name); meta['teachers'][t] = src.name
    for arm in _arms_needed(cfg):
        src = C.admitted(f'stage_b/S{seed}/{arm}__E20.npz'); _sh.copy2(src, d / src.name); meta['thetas'][arm] = src.name
    hs = rec['h']
    with open(d / 'H.npz', 'wb') as f:
        np.savez(f, mu=np.asarray(hs['mu'], dtype=np.float64), sd=np.asarray(hs['sd'], dtype=np.float64),
                 w=np.asarray(hs['w'], dtype=np.float64), b=np.asarray([hs['b']], dtype=np.float64), thresholds=np.asarray(th, dtype=np.float64))
    for f in sorted(p.name for p in d.iterdir()):
        meta['files'][f] = C.sha(d / f)
    (d / 'PACKAGE.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
    for f in d.iterdir():
        f.chmod(0o600)
    return d


def package_release(pkg_dir, rows, requested=('probability', 'decision')):
    """Recompute the released vectors from the package files only (raw permitted rows -> encoder -> routed teachers + stored
    corrections -> occupation output; income = h(phi(occupation output), coarse statistic)). Same arithmetic as `fit`."""
    from acs_replication import deploy as SRC_DEPLOY
    if set(requested) - {'probability', 'decision'}:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - {"probability", "decision"})}')
    pkg_dir = Path(pkg_dir)
    meta = json.loads((pkg_dir / 'PACKAGE.json').read_text())
    if meta.get('schema') != PKG_SCHEMA or meta.get('kind') != 'FAMB':
        raise ValueError('not a family-B package')
    for f, h in meta['files'].items():
        if C.sha(pkg_dir / f) != h:
            raise PermissionError(f'package file hash differs: {f}')
    cfg = meta['config']
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    te = {t: trees.TreeTeacher(json.loads((pkg_dir / meta['teachers'][t]).read_text())) for t in C.TASKS}
    th_by_arm = {}
    for arm, f in meta['thetas'].items():
        with np.load(pkg_dir / f) as z:
            th_by_arm[arm] = {'income': z['theta_income'].copy(), 'occupation': z['theta_occupation'].copy()}
    leaves = {t: te[t].route(X, names) for t in C.TASKS}

    def raw(arm, t):
        return te[t].raw_from_leaves(leaves[t], None if arm == 'T_TREE' else th_by_arm[arm][t])
    p_occ = trees.probabilities('occupation', raw(cfg['base'], 'occupation'))
    with np.load(pkg_dir / meta['h']) as z:
        hz = {k: z[k].copy() for k in z.files}
    d = c = None
    if cfg['stat'] != 'none':
        if cfg['stat_source'] == 'MIX':
            a = cfg['alpha']
            d = (1 - a) * raw('LOCAL_TREE', 'income').reshape(-1) + a * raw('JOINT_TREE', 'income').reshape(-1)
        else:
            d = raw(cfg['stat_source'], 'income').reshape(-1)
        if cfg['stat'] == 'bins':
            c = coarse_bins(d, [float(x) for x in hz['thresholds']])
    Z = design(p_occ, cfg['phi'], c, cfg['m'], d, cfg['stat'])
    p_inc = trees.probabilities('income', ((Z - hz['mu']) / hz['sd']) @ hz['w'] + float(hz['b'][0]))
    out = {'income': p_inc, 'occupation': p_occ}
    return {t: {k: v for k, v in {'probability': out[t], 'decision': out[t].argmax(1).astype(np.int64)}.items() if k in requested} for t in C.TASKS}


def package_selftest(configs, out_root):
    """Label-free parity check on DEV_REPORT raw inputs: package_release must reproduce the saved DEV_REPORT release bitwise."""
    from . import deploy as AD, devrun
    rows = AD.raw_rows(); res = {}
    for config_id in configs:
        for seed in C.SEEDS:
            dpk = build_package(f'B_{config_id}', seed, config_id, out_root)
            ref = devrun.load_release(FAMILY, config_id, seed, roles=('DEV_REPORT',))[0]['DEV_REPORT']
            got = AD.release(dpk, rows)
            res[f'{config_id}_S{seed}'] = {t: bool(np.array_equal(got[t]['probability'], ref[t]) and np.array_equal(got[t]['decision'], ref[t].argmax(1)))
                                           for t in C.TASKS}
    print(json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('diag'); g.add_argument('--seed', type=int, required=True)
    g2 = sub.add_parser('diag2'); g2.add_argument('--seed', type=int, required=True)
    g3 = sub.add_parser('diag3'); g3.add_argument('--seed', type=int, required=True)
    pt = sub.add_parser('pkgtest'); pt.add_argument('--out-root', required=True); pt.add_argument('--config', action='append', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True)
    f.add_argument('--base', choices=BASES, required=True); f.add_argument('--phi', choices=('none', 'lin', 'quad'), default='lin')
    f.add_argument('--m', type=int, default=1); f.add_argument('--coarse-kind', choices=('quantile', 'decision', 'aligned'), default='aligned')
    f.add_argument('--stat', choices=('none', 'bins', 'cont', 'cont_int'), default='none')
    f.add_argument('--stat-source', choices=BASES + ('MIX',), default='LOCAL_TREE'); f.add_argument('--alpha', type=float, default=0.0)
    f.add_argument('--C', type=float, default=1.0); f.add_argument('--note', default='')
    a = ap.parse_args(argv)
    if a.cmd == 'diag':
        diag(a.seed)
    elif a.cmd == 'diag2':
        diag2(a.seed)
    elif a.cmd == 'diag3':
        diag3(a.seed)
    elif a.cmd == 'pkgtest':
        package_selftest(a.config, a.out_root)
    else:
        fit(a.seed, a.config, a.base, a.phi, a.m, a.coarse_kind, a.C, a.note, a.stat, a.stat_source, a.alpha)


if __name__ == '__main__':
    main()
