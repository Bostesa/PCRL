"""Estimators of PROTOCOL.md section 2 on one synthetic sample (labels in {0, 1}; ties score 1/2; a sample lacking a class gives an
UNDEFINED value (NaN, recorded), never 1/2).

  A        population optimal-reader AUC of the known law (drf.finite_law.auc_star on merged observable wires) - in laws.py
  B_indep  sample AUC of the known population-optimal score (true posterior ranking) on an independent test sample
  B_same   the same known score on the study sample (diagnostic: C - B_same >= 0 is the ranking-selection optimism)
  C_same   empirical optimal AUC: cell posteriors and their ranking estimated and evaluated on the same sample
           (drf.solver.auc_star on the sample's per-wire SEX masses, the DRF1 computation)
  C_nested DRF1's nested held-out exact AUC*: the sample split into 5 folds, each fold's OWN empirical optimal AUC, people-weighted
           mean (drf.dr.nested 'heldout_exact_auc_mean' for a fixed release)
  D_nested DRF1's nested held-out LOOKUP: 5-fold cross-fit of nrm.fa.prc.lookup_scores (Beta shrinkage a = 1 toward the training
           rate; unseen wire -> training rate), AUC pooled over all people (nrm.fa.prc.auc)
  D_bank   the declared bank's LOOKUP recipe (scm_bench.readers: exact float64 wire bytes as keys, pseudo-count 2 toward the FIT
           rate, unseen -> FIT rate) fitted on FIT (50%), oriented on VALIDATION (20%), scored once on SCORE (30%)
For D_* the exact population AUC of every fitted ranking under the known law is also returned (the fitted reader's own recovery),
which separates its estimation loss (A - pop) from test-sample variation (sample - pop).

Influence values: for an AUC on people with fixed scores, IF_i = (V1_i - AUC) / n1 (positives) or (V0_i - AUC) / n0 (negatives)
with DeLong placement values (ties 1/2); the linearized SE of a (paired) AUC (difference) is sqrt(G / (G - 1) sum_h (sum_{i in h}
IF_i)^2) over units h (people, or households in the clustered scenario). It treats the scored ranking as FIXED: for C_* that ignores
the ranking's selection on the same sample (the naive practice this diagnostic tests)."""
from __future__ import annotations

import numpy as np

from . import RESULTS  # noqa: F401  (path setup)
from drf import finite_law as FL
from drf import solver as SV
from . import source_prc as PRC
from scm_bench.readers import LOOKUP_PSEUDO, fit_lookup, predict_lookup
from task_conditioned_representation.attacks import orientation, orient
from task_conditioned_representation.inference import auc as INF_AUC

NESTED_A = 1.0                 # drf.dr.LOOKUP_A
BANK_A = float(LOOKUP_PSEUDO)  # 2.0
N_FOLDS = 5
NAN = float('nan')


def auc_if(score, y):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return NAN, None
    u, inv = np.unique(np.asarray(score, dtype=np.float64), return_inverse=True)
    c1 = np.bincount(inv, weights=y, minlength=len(u))
    c0 = np.bincount(inv, minlength=len(u)) - c1
    below0 = np.cumsum(c0) - c0
    above1 = n1 - np.cumsum(c1)
    V1 = (below0 + 0.5 * c0) / n0
    V0 = (above1 + 0.5 * c1) / n1
    a = float(np.dot(c1, V1) / n1)
    IF = np.where(y == 1, (V1[inv] - a) / n1, (V0[inv] - a) / n0)
    return a, IF


def cluster_se(IF, unit):
    if IF is None:
        return NAN
    _, inv = np.unique(unit, return_inverse=True)
    s = np.bincount(inv, weights=IF)
    G = len(s)
    return float(np.sqrt(G / max(G - 1, 1) * np.sum(s ** 2)))


def masses(ids, y, K):
    m1 = np.bincount(ids, weights=y, minlength=K).astype(np.float64)
    return m1, np.bincount(ids, minlength=K).astype(np.float64) - m1


def c_same(ids, y, K):
    """Empirical optimal AUC (same sample). Source value: drf.solver.auc_star; the influence values come from the identical ranking."""
    m1, m0 = masses(ids, y, K)
    if m1.sum() == 0 or m0.sum() == 0:
        return NAN, None
    val = float(SV.auc_star(m1, m0))
    tot = m1 + m0
    r = np.where(tot > 0, m1 / np.where(tot > 0, tot, 1.0), -1.0)
    a, IF = auc_if(r[ids], y)
    if abs(a - val) > 1e-9:
        raise AssertionError(f'C_same: influence AUC {a} differs from drf.solver.auc_star {val}')
    return val, IF


def c_nested(ids, y, folds, K):
    n = len(ids)
    vals, IF = [], np.zeros(n)
    for f in range(N_FOLDS):
        m = folds == f
        v, inf = c_same(ids[m], y[m], K)
        if inf is None:
            return NAN, None
        vals.append((m.sum(), v))
        IF[m] = (m.sum() / n) * inf
    return float(np.average([v for _, v in vals], weights=[k for k, _ in vals])), IF


def d_nested(ids, y, folds, K, m1p, m0p):
    n = len(ids)
    look = np.zeros(n)
    pops, shares, sc_all, unseen = [], [], [], 0
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        look[te] = PRC.lookup_scores(ids[tr], y[tr], ids[te], a=NESTED_A)
        s_all = PRC.lookup_scores(ids[tr], y[tr], np.arange(K), a=NESTED_A)
        pops.append(FL.auc_of_score(s_all, m1p, m0p))
        shares.append(te.mean())
        sc_all.append(s_all)
        seen = np.bincount(ids[tr], minlength=K) > 0
        unseen += int((~seen[ids[te]]).sum())
    pooled = PRC.auc(look, y)
    if not np.isfinite(pooled):
        return {'value': NAN, 'IF': None}
    a, IF = auc_if(look, y)
    if abs(a - pooled) > 1e-9:
        raise AssertionError(f'D_nested: influence AUC {a} differs from nrm.fa.prc.auc {pooled}')
    pop_pooled = FL.auc_of_score(np.concatenate(sc_all), np.concatenate([m1p * s for s in shares]),
                                 np.concatenate([m0p * s for s in shares]))
    return {'value': float(pooled), 'IF': IF, 'pop_fold_mean': float(np.average(pops, weights=shares)),
            'pop_pooled': float(pop_pooled), 'unseen_test_share': unseen / n}


def d_bank(ids, y, parts, K, m1p, m0p, wire_keys=None):
    """Bank LOOKUP; with wire_keys (one bytes key per observable wire) the vectorized table is checked bitwise against
    scm_bench.readers.fit_lookup / predict_lookup on this sample."""
    fit, val, sc = parts <= 4, (parts == 5) | (parts == 6), parts >= 7
    yf = y[fit]
    if yf.sum() == 0 or yf.sum() == len(yf):
        return {'value': NAN, 'IF': None, 'undefined': 'FIT lacks a class'}
    pi_hat = float(yf.mean())
    nw = np.bincount(ids[fit], minlength=K).astype(np.float64)
    n1w = np.bincount(ids[fit], weights=yf, minlength=K)
    s_w = np.where(nw > 0, (n1w + BANK_A * pi_hat) / (nw + BANK_A), pi_hat)
    raw = s_w[ids]
    checked = False
    if wire_keys is not None:
        keys = [wire_keys[i] for i in ids]
        rec = fit_lookup([keys[i] for i in np.flatnonzero(fit)], yf)
        src = predict_lookup(rec, keys)
        if not np.array_equal(src, raw):
            raise AssertionError('D_bank: vectorized LOOKUP differs from scm_bench.readers.predict_lookup')
        checked = True
    yv = y[val]
    if yv.sum() == 0 or yv.sum() == len(yv):
        return {'value': NAN, 'IF': None, 'undefined': 'VALIDATION lacks a class'}
    sign = orientation(yv, raw[val])
    s = orient(raw, sign)
    ys = y[sc]
    if ys.sum() == 0 or ys.sum() == len(ys):
        return {'value': NAN, 'IF': None, 'undefined': 'SCORE lacks a class'}
    v_src = INF_AUC(ys, s[sc])
    a, IF = auc_if(s[sc], ys)
    if abs(a - v_src) > 1e-9:
        raise AssertionError('D_bank: influence AUC differs from task_conditioned_representation.inference.auc')
    pop = FL.auc_of_score(orient(s_w, sign), m1p, m0p)
    return {'value': float(v_src), 'IF': IF, 'pop': float(pop), 'val_auc': float(INF_AUC(yv, s[val])), 'sign': int(sign),
            'unseen_score_share': float(np.mean(nw[ids[sc]] == 0)), 'source_checked': checked}


def b_known(ids, y, post):
    a, IF = auc_if(post[ids], y)
    return a, IF
