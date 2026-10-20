"""PRC: propensity-routed finite cells (Designer A, family FA). PCRL campaign `pcrl_new_release_mechanisms_v1`; EXPLORATORY
DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation.

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women) is
a training label on MODEL_FIT only, never a runtime input or output.

Release operation (one deterministic function of X):
  Z      = ohn.fam_a.compact_po(X)                       (the A1_PO_16 coding, reused by import)
  e(x)   = (qI(x), qO(x), pi(x), qO0(x), qO1(x))         (fitted statistics of x; all from X alone at runtime)
           qI  income posterior        HGB, A1_PO_16 learner settings, binary              (task)
           qO  occupation posterior    A1_PO_16 exactly (ohn.fam_a.make_model)             (task)
           pi  P(SEX = 1 | x)          HGB, same settings, SEX label of MODEL_FIT          (protected configs only)
           qOs P(occ | x, SEX = s)     HGB on [Z, s], evaluated at s = 0 and 1             (protected configs only; guard term)
  (k, l) = argmin over (income cell k, occupation cell l) of
             CE(qI, cI_k) + CE(qO, cO_l)                                                    task cost (cross-entropy to codewords)
           + lam * [ w_pair * D(pi; rP_kl) + w_loc * (D(pi; rI_k) + D(pi; rO_l)) ]           dependence of the RELEASED cells
           + lam * kappa * [ (1-pi)/p0 * |qO0 - cO_l|^2 + pi/p1 * |qO1 - cO_l|^2 ]           per-SEX occupation guard
         where D(pi; r) = pi log(r / rbar) + (1 - pi) log((1 - r) / (1 - rbar)) is the person's expected contribution to the
         log-likelihood gain of the Bayes (LOOKUP) reader of that cell table, and rI, rO, rP are the income, occupation and joint
         (pair) cell SEX rates (constants after fitting).
  output = (dI_k, dO_l): label-refit decoders of the routed cells (MODEL_FIT label counts of the out-of-fold-routed people,
         shrunk toward the codeword with alpha pseudo-counts). Decisions = argmax. Nothing else is transmitted; the cell identity is
         recoverable from the vectors and is exactly what the dependence objective and its checks measure.
lam = 0 gives the equally trained TASK-ONLY TWIN: two independent KL (Bregman) k-means quantizers of the task posteriors; it never
fits pi or qOs and never reads SEX (selftest: bitwise identical with and without SEX).
Codebooks: deterministic KL k-means++ initialisation and Lloyd steps at lam = 0; protected books are warm-started from the twin and
continued along the predeclared strength path (Lloyd: route, then codeword = mean posterior of the cell, then cell SEX rates =
shrunk mean pi of the cell). Routing is trained on out-of-fold (5 household folds) embeddings of MODEL_FIT.

Variants (same code, CONFIGS): PRC_* (strength path), ORD_* (SEX-blind coarsening path: twin at smaller codebooks), HQ_* (post-hoc
grouping / published IBSI objective restricted to h(q): pi, qO0, qO1 replaced by their means within the twin's joint task cells,
so the routing is a function of the task posteriors only), LOC_* (separately designed recipients: w_pair = 0).

Checks (MODEL_FIT only): cross-fitted LOOKUP tables (household folds; Beta-shrunk SEX rates from the other folds score the held-out
fold) for the pair, income and occupation views; plug-in MI; and the NESTED estimator (whole pipeline refitted on outer-training
households, held-out task costs from labels, held-out LOOKUP with tables from the outer-training part), which feeds the predeclared
strength-selection rule `select_path`. MODEL_SELECT labels are never read here.

Commands:  python3 -m nrm.fa.prc selftest      (invented arrays only; no real data, no labels)
Real-data entry points (`fit`, `nested`) run only through the campaign queue after the protocol receipt and the committed card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')                    # one numerical thread (campaign rule); set before numpy / sklearn load

import numpy as np

FAMILY = 'FA'
K_INC, K_OCC = 2, 5
CLIP = 1e-6
HGB = {'lr': 0.025, 'leaves': 16, 'msl': 100, 'l2': 3.0, 'max_iter': 1000}      # ohn.fam_a.CONFIGS['A1_PO_16'] learner settings
CAT_Z = np.array([False, False, True, False, True, False])                      # compact_po: COW, MAR native categories
N_INNER, N_XF, N_OUTER = 5, 5, 5
ROUTE = {'KI': 16, 'KO': 64, 'lam': 0.0, 'w_pair': 1.0, 'w_loc': 1.0, 'kappa': 1.0, 'm_shrink': 20.0, 'alpha_dec': 20.0,
         'init_steps': 50, 'path_steps': 30, 'restrict': None}
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)          # internal grid; released strengths are chosen by select_path (MODEL_FIT only)
ORD_LADDER = ((12, 48), (8, 32), (6, 24), (4, 16), (3, 12), (2, 8))
CAPACITY_LADDER = ((16, 64), (24, 128))
# Predeclared spend budgets over the twin (nested MODEL_FIT held-out estimate + 2 household-clustered SE must stay at or below):
BUDGETS = {
    'S1': {'occ_nll': 0.004, 'inc_nll': 0.002, 'occ_brier_men': 0.0015, 'occ_brier_women': 0.0015, 'occ_acc': 0.003, 'inc_acc': 0.002},
    'S2': {'occ_nll': 0.008, 'inc_nll': 0.004, 'occ_brier_men': 0.0030, 'occ_brier_women': 0.0030, 'occ_acc': 0.006, 'inc_acc': 0.004},
    'S3': {'occ_nll': 0.012, 'inc_nll': 0.006, 'occ_brier_men': 0.0045, 'occ_brier_women': 0.0045, 'occ_acc': 0.008, 'inc_acc': 0.006},
}
CONFIGS = {
    'PRC_TWIN': {'kind': 'task_only_twin', 'route': {}, 'what': 'lam = 0: independent KL k-means quantizers (16 income, 64 occupation '
                 'cells) of the task posteriors, label-refit decoders; never reads SEX'},
    **{f'PRC_{s}': {'kind': 'protected', 'route': {}, 'budget': s, 'what': f'propensity-routed cells, strength chosen by select_path '
                    f'at budget {s} (pair + local dependence, per-SEX guard)'} for s in BUDGETS},
    **{f'ORD_{s}': {'kind': 'sex_blind_control', 'route': {}, 'budget': s, 'what': f'SEX-blind coarsening: the twin at the most '
                    f'coarsened codebook pair of ORD_LADDER whose nested spend meets budget {s}'} for s in BUDGETS},
    'HQ_S2': {'kind': 'post_hoc_grouping_control', 'route': {'restrict': 'twin_cells'}, 'budget': 'S2',
              'what': 'published IBSI / privacy-funnel objective restricted to h(q): routing inputs are functions of the task posteriors'},
    'LOC_S2': {'kind': 'separate_recipients_control', 'route': {'w_pair': 0.0, 'w_loc': 1.0}, 'budget': 'S2',
               'what': 'separately designed recipient releases: each routing sees only its own view table'},
}


# ---------------------------------------------------------------- small utilities
def household_folds(unit, k, tag):
    """Deterministic household-separated fold ids (same construction as ohn.common.household_folds)."""
    groups = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups]
    fold = np.empty(len(groups), dtype=np.int64)
    fold[np.argsort(keys)] = np.arange(len(groups)) % k
    return fold[np.searchsorted(groups, unit)]


def probs(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - p1, p1])
    r = raw - raw.max(1, keepdims=True)
    e = np.exp(r)
    return e / e.sum(1, keepdims=True)


def clipn(P):
    P = np.clip(np.asarray(P, dtype=np.float64), CLIP, 1.0)
    return P / P.sum(1, keepdims=True)


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64)
    g, inv = np.unique(unit, return_inverse=True)
    s = np.bincount(inv, weights=v - v.mean(), minlength=len(g))
    G = len(g)
    return float(np.sqrt(G / max(G - 1, 1) * np.sum(s ** 2)) / len(v))


def auc(score, s):
    from scipy.stats import rankdata
    s = np.asarray(s).astype(bool)
    n1, n0 = int(s.sum()), int((~s).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[s].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def array_sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- statistics e(x): HGB teachers (A1_PO_16 learner settings)
def make_hgb(hp, max_iter, seed, cat):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(hp['lr']), max_iter=int(max_iter),
                                          max_leaf_nodes=int(hp['leaves']), min_samples_leaf=int(hp['msl']),
                                          l2_regularization=float(hp['l2']), max_bins=255, early_stopping=False,
                                          categorical_features=cat, random_state=int(seed))


def fit_hgb(hp, Z, y, K, max_iter, seed, cat):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = make_hgb(hp, max_iter, seed, cat).fit(Z, y)
    if not np.array_equal(m.classes_, np.arange(K)) or m.n_iter_ != int(max_iter):
        raise AssertionError('class support or iteration count differs (support failure: no repair)')
    return m


def predict(m, Z):
    return clipn(probs(m.decision_function(Z)))


def staged_nll(m, Z, y):
    out = np.zeros(m.n_iter_)
    for i, r in enumerate(m.staged_decision_function(Z)):
        p = clipn(probs(r))
        out[i] = -np.log(p[np.arange(len(y)), y]).sum()
    return out


def tstar(hp, Z, y, K, inner, seed, cat):
    """First argmin of the pooled inner-CV NLL over household folds (the A1_PO_16 rule)."""
    cur = np.zeros(int(hp['max_iter']))
    for j in np.unique(inner):
        tr, va = inner != j, inner == j
        cur += staged_nll(fit_hgb(hp, Z[tr], y[tr], K, hp['max_iter'], seed, cat), Z[va], y[va])
    return int(np.argmin(cur)) + 1, cur / len(y)


def sex_aug(Z, s):
    return np.column_stack([Z, np.full(len(Z), float(s))])


TEACHERS = {'inc': ('income', K_INC, False), 'occ': ('occupation', K_OCC, False), 'sex': ('sex', 2, False),
            'occ_s': ('occupation', K_OCC, True)}


def fit_teachers(Z, lab, unit, version, hp, names, T_fixed=None, log=None):
    """Teachers `names` on the training rows: T* (inner CV, unless T_fixed), deployed full refit, and out-of-fold predictions over
    N_XF household folds (fold models at the same T*; T* is not re-selected inside the folds: non-nested, recorded)."""
    inner = household_folds(unit, N_INNER, f'OHN_A_cv_S{version}')
    xf = household_folds(unit, N_XF, f'NRM_FA_xf_S{version}')
    out = {'T': {}, 'curve_min': {}, 'models': {}, 'oof': {}}
    for nm in names:
        field, K, aug = TEACHERS[nm]
        y = lab[field]
        Zt = np.column_stack([Z, lab['sex'].astype(np.float64)]) if aug else Z
        cat = np.append(CAT_Z, False) if aug else CAT_Z
        if T_fixed is not None:
            T = int(T_fixed[nm])
        else:
            T, cur = tstar(hp, Zt, y, K, inner, version, cat)
            out['curve_min'][nm] = float(cur.min())
        out['T'][nm] = T
        out['models'][nm] = fit_hgb(hp, Zt, y, K, T, version, cat)
        if aug:
            oof = {0: np.zeros((len(Z), K)), 1: np.zeros((len(Z), K))}
        else:
            oof = np.zeros((len(Z), K))
        for f in range(N_XF):
            tr, te = xf != f, xf == f
            m = fit_hgb(hp, Zt[tr], y[tr], K, T, version, cat)
            if aug:
                for s in (0, 1):
                    oof[s][te] = predict(m, sex_aug(Z[te], s))
            else:
                oof[te] = predict(m, Zt[te])
        out['oof'][nm] = oof
        if log:
            log(json.dumps({'teacher': nm, 'T': T}))
    return out


def embed_oof(tch):
    o = tch['oof']
    E = {'qI': o['inc'], 'qO': o['occ']}
    if 'sex' in o:
        E['pi'] = np.clip(o['sex'][:, 1], 0.01, 0.99)
        E['qO0'], E['qO1'] = o['occ_s'][0], o['occ_s'][1]
    return E


def embed(models, Z):
    """Runtime statistics from features alone (deployed full refits)."""
    E = {'qI': predict(models['inc'], Z), 'qO': predict(models['occ'], Z)}
    if 'sex' in models:
        E['pi'] = np.clip(predict(models['sex'], Z)[:, 1], 0.01, 0.99)
        E['qO0'] = predict(models['occ_s'], sex_aug(Z, 0))
        E['qO1'] = predict(models['occ_s'], sex_aug(Z, 1))
    return E


# ---------------------------------------------------------------- routing (privacy-funnel Lloyd on e(x))
def _xent(Q, C):
    return -(Q @ np.log(C).T)


def _dep_ab(r, rbar):
    r = np.clip(r, 0.01, 0.99)
    b = np.log((1 - r) / (1 - rbar))
    return np.log(r / rbar) - b, b


def _guard(E, CO, book):
    out = 0.0
    for key, w in (('qO0', (1 - E['pi']) / book['p0']), ('qO1', E['pi'] / book['p1'])):
        Q = E[key]
        sq = (Q ** 2).sum(1)[:, None] - 2 * Q @ CO.T + (CO ** 2).sum(1)[None, :]
        out = out + w[:, None] * sq
    return out


def restrict_embedding(E, book):
    """h(q) restriction (HQ control): pi, qO0, qO1 replaced by their MODEL_FIT means within the twin's joint task cell of x."""
    if book.get('restrict') != 'twin_cells':
        return E
    tw = book['twin']
    k = _xent(E['qI'], tw['CI']).argmin(1)
    l = _xent(E['qO'], tw['CO']).argmin(1)
    j = k * len(tw['CO']) + l
    R = dict(E)
    R['pi'] = book['hq_pi'][j]
    R['qO0'], R['qO1'] = book['hq_qO0'][j], book['hq_qO1'][j]
    return R


def route(E, book, chunk=2048, cur=None):
    """Joint cell (k, l) of every person; lam = 0 separates into the two task-only nearest-codeword rules. With cur = (k, l) it
    also returns each person's linearised cost gain of the proposed cell over the current one (used by the damped Lloyd step)."""
    AI, AO = _xent(E['qI'], book['CI']), _xent(E['qO'], book['CO'])
    lam = float(book.get('tilt', 0.0))              # the RULE's tilt (objective strength is book['lam'])
    if lam == 0.0:
        k, l = AI.argmin(1), AO.argmin(1)
        if cur is None:
            return k, l
        i = np.arange(len(k))
        return k, l, AI[i, cur[0]] - AI[i, k] + AO[i, cur[1]] - AO[i, l]
    R = restrict_embedding(E, book)
    pi, rbar = R['pi'], book['rbar']
    aI, bI = _dep_ab(book['rI'], rbar)
    aO, bO = _dep_ab(book['rO'], rbar)
    aP, bP = _dep_ab(book['rP'], rbar)
    wl, wp = lam * book['w_loc'], lam * book['w_pair']
    AI = AI + wl * (pi[:, None] * aI[None] + bI[None])
    AO = AO + wl * (pi[:, None] * aO[None] + bO[None])
    if book['kappa'] > 0:
        AO = AO + lam * book['kappa'] * _guard(R, book['CO'], book)
    KI, KO = book['CI'].shape[0], book['CO'].shape[0]
    k, l = np.empty(len(pi), dtype=np.int64), np.empty(len(pi), dtype=np.int64)
    gain = np.zeros(len(pi))
    for a in range(0, len(pi), chunk):
        sl = slice(a, a + chunk)
        cost = AI[sl, :, None] + AO[sl, None, :]
        if wp != 0.0:
            cost = cost + wp * (pi[sl, None, None] * aP[None] + bP[None])
        flat = cost.reshape(cost.shape[0], -1)
        f = flat.argmin(1)
        k[sl], l[sl] = f // KO, f % KO
        if cur is not None:
            i = np.arange(flat.shape[0])
            gain[sl] = flat[i, cur[0][sl] * KO + cur[1][sl]] - flat[i, f]
    return (k, l) if cur is None else (k, l, gain)


def _kmeanspp(Q, K, rng):
    n = len(Q)
    C = [Q[rng.integers(n)]]
    d = np.full(n, np.inf)
    for _ in range(1, K):
        c = clipn(np.array(C[-1])[None])[0]
        d = np.minimum(d, np.maximum((Q * (np.log(Q) - np.log(c))).sum(1), 0.0))
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)
        C.append(Q[rng.choice(n, p=p)])
    return clipn(np.array(C))


def _shrunk_rate(sum_pi, n, prior, m):
    return (sum_pi + m * prior) / (n + m)


def update_book(E, book, k, l):
    """Codewords = mean posterior of each occupied cell (Bregman centroid); cell SEX rates = shrunk mean pi (protected books)."""
    KI, KO = book['CI'].shape[0], book['CO'].shape[0]
    for K, cells, Q, key in ((KI, k, E['qI'], 'CI'), (KO, l, E['qO'], 'CO')):
        n = np.bincount(cells, minlength=K)
        S = np.zeros((K, Q.shape[1]))
        np.add.at(S, cells, Q)
        occ = n > 0
        C = book[key].copy()
        C[occ] = S[occ] / n[occ, None]
        book[key] = clipn(C)
    if 'pi' in E and book.get('lam', 0.0) > 0:
        set_tables(E['pi'], book, k, l)
    return book


def set_tables(pi, book, k, l):
    KI, KO = book['CI'].shape[0], book['CO'].shape[0]
    m, rbar = book['m_shrink'], book['rbar']
    nI, nO = np.bincount(k, minlength=KI), np.bincount(l, minlength=KO)
    book['rI'] = _shrunk_rate(np.bincount(k, weights=pi, minlength=KI), nI, rbar, m)
    book['rO'] = _shrunk_rate(np.bincount(l, weights=pi, minlength=KO), nO, rbar, m)
    j = k * KO + l
    nP = np.bincount(j, minlength=KI * KO).reshape(KI, KO)
    sP = np.bincount(j, weights=pi, minlength=KI * KO).reshape(KI, KO)
    lg = lambda r: np.log(r / (1 - r))
    add = 1.0 / (1.0 + np.exp(-(lg(book['rI'])[:, None] + lg(book['rO'])[None, :] - lg(rbar))))
    book['rP'] = _shrunk_rate(sP, nP, add, m)


def objective(E, book, k, l):
    """The exact fitted objective of a hard routing (training embeddings): mean task cross-entropy to the codewords
    + lam * (w_pair * I(S; pair cell) + w_loc * (I(S; income cell) + I(S; occupation cell)))   [plug-in, pi-weighted]
    + lam * kappa * mean per-SEX occupation guard term (true pi, qO0, qO1)."""
    n = len(k)
    J = float(-(E['qI'] * np.log(book['CI'][k])).sum() / n - (E['qO'] * np.log(book['CO'][l])).sum() / n)
    lam = float(book['lam'])
    if lam > 0:
        KO = book['CO'].shape[0]
        pi = E['pi']
        J += lam * (book['w_pair'] * plugin_mi(k * KO + l, pi) + book['w_loc'] * (plugin_mi(k, pi) + plugin_mi(l, pi)))
        if book['kappa'] > 0:
            g = 0.0
            for key, w in (('qO0', (1 - pi) / book['p0']), ('qO1', pi / book['p1'])):
                g = g + w * ((E[key] - book['CO'][l]) ** 2).sum(1)
            J += lam * book['kappa'] * float(np.mean(g))
    return J


TILT_STEPS = 7                       # candidate rule tilts lam, lam/2, ..., lam/64 (damping by the tilt, never by partial moves)


def lloyd(E, book, steps, k=None, l=None):
    """Damped Lloyd, monotone in the exact `objective` and RULE-CONSISTENT: every accepted state is (book, route(E, book)), so the
    deployed function of x reproduces the fitted partition. Each step refreshes codewords (cell means) and tables (shrunk cell
    pi means) from the current partition, then tries the rule tilts lam, lam/2, ..., lam/64 (small tilts move only the people whose
    task cost difference is smallest, i.e. near cell boundaries) and accepts the first whose partition lowers the exact
    objective. Stops when no candidate lowers it. (k, l) must be route(E, book) on entry (recomputed when omitted)."""
    if k is None:
        k, l = route(E, book)
    J = objective(E, book, k, l)
    lam = float(book['lam'])
    tilts = [lam / 2 ** j for j in range(TILT_STEPS)] if lam > 0 else [0.0]
    for _ in range(int(steps)):
        nb = update_book(E, dict(book), k, l)
        acc = False
        for t in tilts:
            nb['tilt'] = float(t)
            k2, l2 = route(E, nb)
            j = objective(E, nb, k2, l2)
            if j < J - 1e-12:
                book, k, l, J, acc = dict(nb), k2, l2, j, True
                break
        if not acc:
            break
    return book, k, l


def twin_book(E, KI, KO, seed, rp):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(KI), int(KO)]))
    book = {'CI': _kmeanspp(E['qI'], KI, rng), 'CO': _kmeanspp(E['qO'], KO, rng), 'lam': 0.0, 'tilt': 0.0, 'restrict': None}
    for key in ('w_pair', 'w_loc', 'kappa', 'm_shrink'):
        book[key] = float(rp[key])
    return lloyd(E, book, rp['init_steps'])


def protected_book(E, twin, k0, l0, lam, rp):
    """Warm start from the twin; tables from the twin's cells; Lloyd continuation at strength lam."""
    book = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in twin.items()}
    book.update({'lam': float(lam), 'w_pair': float(rp['w_pair']), 'w_loc': float(rp['w_loc']), 'kappa': float(rp['kappa']),
                 'm_shrink': float(rp['m_shrink']), 'restrict': rp.get('restrict')})
    book['rbar'] = float(E['pi'].mean())
    book['p0'], book['p1'] = float((1 - E['pi']).mean()), float(E['pi'].mean())
    if rp.get('restrict') == 'twin_cells':
        KO = len(twin['CO'])
        j = k0 * KO + l0
        J = len(twin['CI']) * KO
        n = np.bincount(j, minlength=J).astype(np.float64)
        book['twin'] = {'CI': twin['CI'].copy(), 'CO': twin['CO'].copy()}
        book['hq_pi'] = np.clip(_shrunk_rate(np.bincount(j, weights=E['pi'], minlength=J), n, book['rbar'], 1.0), 0.01, 0.99)
        for key in ('qO0', 'qO1'):
            S = np.zeros((J, K_OCC))
            np.add.at(S, j, E[key])
            mean = E[key].mean(0)
            book[f'hq_{key}'] = clipn((S + mean[None]) / (n[:, None] + 1.0))
    book['tilt'] = 0.0                                                         # start = the twin's own rule and partition
    start = (dict(book), k0, l0)
    k, l = k0, l0
    for lam_t in sorted({g for g in LAMBDA_GRID if g < lam} | {float(lam)}):     # deterministic continuation up the grid
        # the objective strength changes; the rule (codewords, tables, tilt) and its partition do not, so (k, l) stays
        # route(E, book); lloyd refreshes the tables from the current partition before it tries any tilt
        book['lam'] = start[0]['lam'] = float(lam_t)
        if objective(E, start[0], k0, l0) < objective(E, book, k, l):         # never worse than the twin at this strength
            book, k, l = dict(start[0]), k0, l0
        book, k, l = lloyd(E, book, rp['path_steps'], k, l)
    return book, k, l


def decoders(cells, y, C, alpha):
    K, k = C.shape
    cnt = np.bincount(cells * k + y, minlength=K * k).reshape(K, k).astype(np.float64)
    return clipn((cnt + alpha * C) / (cnt.sum(1, keepdims=True) + alpha))


def emit(E, book, dec):
    k, l = route(E, book)
    return {'income': dec['I'][k].copy(), 'occupation': dec['O'][l].copy()}, k, l


# ---------------------------------------------------------------- the whole pipeline (fit on training rows; apply from features)
def route_params(cfg_id, lam=None):
    rp = dict(ROUTE)
    rp.update(CONFIGS[cfg_id]['route'])
    if lam is not None:
        rp['lam'] = float(lam)
    return rp


def fit_pipeline(Z, lab, unit, version, plan, hp=None, T_fixed=None, capacity=(16, 64), log=None):
    """plan: {name: ('twin',) | ('ord', KI, KO) | ('prc', cfg_id, lam)}. Fits on the given (MODEL_FIT) rows only. Teachers pi and
    qOs are fitted only when a protected entry is in the plan (the twin and ORD entries never read SEX)."""
    hp = hp or HGB
    need_sex = any(p[0] == 'prc' for p in plan.values())
    names = ('inc', 'occ', 'sex', 'occ_s') if need_sex else ('inc', 'occ')
    tch = fit_teachers(Z, lab, unit, version, hp, names, T_fixed=T_fixed, log=log)
    E = embed_oof(tch)
    rp0 = route_params('PRC_TWIN')
    KI, KO = capacity
    tb, k0, l0 = twin_book(E, KI, KO, version, rp0)
    state = {'version': int(version), 'T': tch['T'], 'models': tch['models'], 'books': {}, 'dec': {}, 'fit_cells': {},
             'capacity': [int(KI), int(KO)], 'E_fit': E, 'twin_fit': (tb, k0, l0)}
    for name, p in plan.items():
        if p[0] == 'twin':
            book, k, l = tb, k0, l0
        elif p[0] == 'ord':
            book, k, l = twin_book(E, p[1], p[2], version, rp0)
        else:
            book, k, l = protected_book(E, tb, k0, l0, p[2], route_params(p[1], p[2]))
        state['books'][name] = book
        state['dec'][name] = {'I': decoders(k, lab['income'], book['CI'], rp0['alpha_dec']),
                              'O': decoders(l, lab['occupation'], book['CO'], rp0['alpha_dec'])}
        state['fit_cells'][name] = (k, l)
    return state


def apply_pipeline(state, Z, name, E=None):
    """Release of entry `name` from compact features alone (E may be passed to reuse runtime statistics)."""
    E = embed(state['models'], Z) if E is None else E
    rel, k, l = emit(E, state['books'][name], state['dec'][name])
    return rel, k, l


# ---------------------------------------------------------------- dependence checks on cell tables
def lookup_scores(train_cells, train_sex, test_cells, a=1.0):
    """LOOKUP reader: Beta(a * rbar, a * (1 - rbar))-shrunk SEX rate of the cell in the training part."""
    rbar = float(train_sex.mean())
    keys, inv = np.unique(np.concatenate([train_cells, test_cells]), return_inverse=True)
    itr, ite = inv[:len(train_cells)], inv[len(train_cells):]
    n = np.bincount(itr, minlength=len(keys))
    s1 = np.bincount(itr, weights=train_sex.astype(np.float64), minlength=len(keys))
    return ((s1 + a * rbar) / (n + a))[ite]


def lookup_crossfit(cells, sex, unit, tag, k=5):
    """Household cross-fitted LOOKUP reader on fixed cells: out-of-fold AUC and mean log-likelihood gain over the base rate."""
    f = household_folds(unit, k, tag)
    sc = np.zeros(len(cells))
    for j in range(k):
        sc[f == j] = lookup_scores(cells[f != j], sex[f != j], cells[f == j])
    rbar = sex.mean()
    p = np.clip(sc, 1e-4, 1 - 1e-4)
    gain = sex * np.log(p / rbar) + (1 - sex) * np.log((1 - p) / (1 - rbar))
    return {'auc': auc(sc, sex), 'loglik_gain': float(gain.mean()), 'se_gain': cluster_se(gain, unit)}


def plugin_mi(cells, w):
    """Plug-in I(S; cell) in nats from per-person SEX weights w (labels or propensities)."""
    keys, inv = np.unique(cells, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    r = np.clip(np.bincount(inv, weights=w) / n, 1e-9, 1 - 1e-9)
    rb = float(np.mean(w))
    kl = r * np.log(r / rb) + (1 - r) * np.log((1 - r) / (1 - rb))
    return float((n / n.sum() * kl).sum())


def views(k, l, KO):
    return {'pair': k * KO + l, 'income': k, 'occupation': l}


# ---------------------------------------------------------------- nested estimator and the predeclared strength rule
def person_metrics(rel, lab):
    out = {}
    for t, key, K in (('inc', 'income', K_INC), ('occ', 'occupation', K_OCC)):
        P, y = rel[key], lab[key]
        oh = np.eye(K)[y]
        out[f'{t}_nll'] = -np.log(P[np.arange(len(y)), y])
        out[f'{t}_brier'] = ((P - oh) ** 2).sum(1)
        out[f'{t}_acc'] = (P.argmax(1) == y).astype(np.float64)
    return out


def nested(Z, lab, unit, version, plan, hp=None, T_fixed=None, capacity=(16, 64), outer=N_OUTER, log=None):
    """Outer household folds: refit the whole pipeline on the outer-training rows, release the held-out rows from features alone,
    compute held-out task metrics (labels) and held-out LOOKUP scores (tables from the outer-training rows' fitted cells)."""
    of = household_folds(unit, outer, f'NRM_FA_ncv_S{version}')
    res = {name: {'rows': np.zeros(len(Z), dtype=bool)} for name in plan}
    store = {name: {} for name in plan}
    for o in range(outer):
        tr, te = of != o, of == o
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both outer parts')
        st = fit_pipeline(Z[tr], {k: v[tr] for k, v in lab.items()}, unit[tr], version, plan, hp=hp, T_fixed=T_fixed,
                          capacity=capacity, log=log)
        E = embed(st['models'], Z[te])
        for name in plan:
            rel, k, l = apply_pipeline(st, Z[te], name, E=E)
            m = person_metrics(rel, {k2: v[te] for k2, v in lab.items()})
            ktr, ltr = st['fit_cells'][name]
            KO = st['books'][name]['CO'].shape[0]
            vtr, vte = views(ktr, ltr, KO), views(k, l, KO)
            for vw in vtr:
                m[f'look_{vw}'] = lookup_scores(vtr[vw], lab['sex'][tr], vte[vw])
            for key, v in m.items():
                store[name].setdefault(key, np.zeros(len(Z)))[te] = v
            res[name]['rows'] |= te
    out = {}
    for name in plan:
        d = store[name]
        out[name] = {key: float(v.mean()) for key, v in d.items() if not key.startswith('look_')}
        for vw in ('pair', 'income', 'occupation'):
            out[name][f'auc_{vw}'] = auc(d[f'look_{vw}'], lab['sex'])
        sx = lab['sex'].astype(bool)
        out[name]['occ_brier_men'] = float(d['occ_brier'][~sx].mean())
        out[name]['occ_brier_women'] = float(d['occ_brier'][sx].mean())
        out[name]['_per_person'] = d
    return out


def spend(nest, name, ref, lab, unit):
    """Held-out spend of `name` over `ref` (paired, per person) with household-clustered SEs; accuracy as a loss."""
    a, b = nest[name]['_per_person'], nest[ref]['_per_person']
    sx = lab['sex'].astype(bool)
    rows = {'occ_nll': (a['occ_nll'] - b['occ_nll'], None), 'inc_nll': (a['inc_nll'] - b['inc_nll'], None),
            'occ_acc': (b['occ_acc'] - a['occ_acc'], None), 'inc_acc': (b['inc_acc'] - a['inc_acc'], None),
            'occ_brier_men': (a['occ_brier'] - b['occ_brier'], ~sx), 'occ_brier_women': (a['occ_brier'] - b['occ_brier'], sx)}
    out = {}
    for key, (v, sub) in rows.items():
        if sub is None:
            out[key] = (float(v.mean()), cluster_se(v, unit))
        else:
            w = sub / sub.mean()                         # linearised subgroup mean (ratio estimator), clustered
            out[key] = (float(v[sub].mean()), cluster_se(w * (v - v[sub].mean()), unit))
    return out


def select_path(nest, lab, unit, candidates, budgets=BUDGETS, ref='PRC_TWIN', k_se=2.0):
    """Predeclared rule (MODEL_FIT nested estimates only): for each budget, the LAST candidate in the given order (increasing
    strength or coarsening) such that it and every earlier candidate keep estimate + k_se * SE <= budget on every metric.
    Returns {budget: candidate or None} and the audit table."""
    audit, chosen = {}, {}
    for c in candidates:
        audit[c] = spend(nest, c, ref, lab, unit)
    for b, lim in budgets.items():
        pick = None
        for c in candidates:
            if all(audit[c][key][0] + k_se * audit[c][key][1] <= lim[key] for key in lim):
                pick = c
            else:
                break
        chosen[b] = pick
    return chosen, audit


# ---------------------------------------------------------------- invented-data selftest (no real data, no labels)
def _fake_schema():
    names = ['AGEP', 'WKHP'] + [f'COW={i}' for i in range(1, 9)] + ['COW=__OTHER__']
    names += [f'SCHL={i:02d}' for i in (1, 4, 6, 9, 12, 16, 18, 19, 20, 21, 22, 24)] + ['SCHL=__OTHER__']
    names += [f'MAR={i}' for i in range(1, 6)] + ['MAR=__OTHER__']
    names += [f'POBP={i:03d}' for i in (1, 6, 12, 36, 48, 110, 139, 210, 303, 312)] + ['POBP=__OTHER__']
    return tuple(names)


def _fake_population(n, rng):
    names = _fake_schema()
    X = np.zeros((n, len(names)))
    X[:, 0] = rng.normal(size=n)
    hidden = rng.random(n) < 0.5                                       # invented binary trait (plays the protected column)
    X[:, 1] = rng.normal(size=n) + 0.9 * (~hidden)                     # hours depend on it
    for b in ('COW', 'SCHL', 'MAR', 'POBP'):
        idx = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        p = rng.dirichlet(np.ones(len(idx)))
        X[np.arange(n), rng.choice(idx, size=n, p=p)] = 1.0
    from ohn import fam_a as FA
    Z = FA.compact_po(X, names)
    so = (Z[:, 3] - 16) / 4.0
    s_inc = 1.1 * X[:, 1] + 0.6 * so + 0.3 * X[:, 0]
    inc = (rng.random(n) < 1 / (1 + np.exp(-s_inc))).astype(np.int64)
    S = np.column_stack([0.9 * so, -0.5 * X[:, 1] + 0.4 * (Z[:, 2] == 1), 0.3 * X[:, 0] - 0.3 * (Z[:, 4] == 0),
                         1.2 * (~hidden) - 0.8 * so + 0.6 * X[:, 1], 0.8 * (~hidden) - 0.3 * so])
    occ = (rng.random(n)[:, None] > np.cumsum(probs(S), 1)).sum(1).astype(np.int64)
    sex = hidden.astype(np.int64)
    unit = np.repeat(np.arange(n), 1 + (rng.random(n) < 0.4))[:n]      # households of 1-2 people
    return X, names, Z, {'income': inc, 'occupation': occ, 'sex': sex}, unit


def selftest():
    t0 = time.process_time()
    rng = np.random.default_rng(20261020)
    checks, info = {}, {}
    n = 3000
    X, names, Z, lab, unit = _fake_population(n, rng)
    hp = {'lr': 0.1, 'leaves': 8, 'msl': 20, 'l2': 1.0, 'max_iter': 60}
    cap = (6, 16)
    from ohn import fam_a as FA
    checks['coding_deterministic'] = bool(np.array_equal(FA.compact_po(X, names), Z))
    f = household_folds(unit, 5, 'selftest')
    checks['folds_household_separated'] = all(len(np.unique(f[unit == g])) == 1 for g in np.unique(unit)[:500])

    # 1. twin without SEX equals twin with SEX (never reads SEX), deterministic refits, release contract
    lab_nosex = {k: v for k, v in lab.items() if k != 'sex'}
    st_a = fit_pipeline(Z, lab_nosex, unit, 0, {'PRC_TWIN': ('twin',)}, hp=hp, capacity=cap)
    st_b = fit_pipeline(Z, lab, unit, 0, {'PRC_TWIN': ('twin',)}, hp=hp, capacity=cap)
    ra, _, _ = apply_pipeline(st_a, Z, 'PRC_TWIN')
    rb, _, _ = apply_pipeline(st_b, Z, 'PRC_TWIN')
    checks['twin_never_reads_sex'] = bool(array_sha(ra['income'], ra['occupation']) == array_sha(rb['income'], rb['occupation']))
    checks['twin_fits_no_sex_teacher'] = sorted(st_b['models']) == ['inc', 'occ']

    plan = {'PRC_TWIN': ('twin',), 'PRC_A': ('prc', 'PRC_S2', 0.5), 'PRC_B': ('prc', 'PRC_S2', 2.0),
            'HQ': ('prc', 'HQ_S2', 2.0), 'LOC': ('prc', 'LOC_S2', 2.0), 'ORD': ('ord', 3, 8)}
    st = fit_pipeline(Z, lab, unit, 0, plan, hp=hp, capacity=cap)
    st2 = fit_pipeline(Z, lab, unit, 0, plan, hp=hp, capacity=cap)
    E = embed(st['models'], Z)
    rel = {nm: apply_pipeline(st, Z, nm, E=E) for nm in plan}
    rel2 = {nm: apply_pipeline(st2, Z, nm) for nm in plan}
    checks['deterministic_refit_bitwise'] = all(array_sha(rel[nm][0]['income'], rel[nm][0]['occupation'])
                                                == array_sha(rel2[nm][0]['income'], rel2[nm][0]['occupation']) for nm in plan)
    ok = True
    for nm, (r, k, l) in rel.items():
        for key, K in (('income', K_INC), ('occupation', K_OCC)):
            P = r[key]
            ok &= P.shape == (n, K) and P.dtype == np.float64 and bool(np.all(P > 0)) and bool(np.allclose(P.sum(1), 1, atol=1e-12))
        KI, KO = st['books'][nm]['CI'].shape[0], st['books'][nm]['CO'].shape[0]
        ok &= len(np.unique(r['income'], axis=0)) <= KI and len(np.unique(r['occupation'], axis=0)) <= KO
        ok &= bool(np.array_equal(r['income'], st['dec'][nm]['I'][k])) and bool(np.array_equal(r['occupation'], st['dec'][nm]['O'][l]))
    checks['release_contract_finite_cells'] = bool(ok)
    # the released vectors identify the cell exactly (disclosure analysis works on the recoverable identity)
    r, k, l = rel['PRC_B']
    _, inv = np.unique(np.column_stack([r['income'], r['occupation']]), axis=0, return_inverse=True)
    pair_cells = k * st['books']['PRC_B']['CO'].shape[0] + l
    checks['vectors_determine_cells'] = len(np.unique(inv)) == len(np.unique(pair_cells))

    # 2. dependence: plug-in (labels) and household cross-fitted LOOKUP, twin vs protected (direction sanity on invented data)
    dep = {}
    for nm in plan:
        _, k, l = rel[nm]
        KO = st['books'][nm]['CO'].shape[0]
        v = views(k, l, KO)
        dep[nm] = {'mi_pair_labels': plugin_mi(v['pair'], lab['sex'].astype(float)),
                   **{f'xf_{vw}': lookup_crossfit(v[vw], lab['sex'], unit, 'selftest_look')['auc'] for vw in v}}
    info['dependence_labels_invented_data_not_gating'] = dep
    # implementation guarantees (fitted, training embeddings): rule consistency, objective never above the twin's at the same
    # strength, and lower fitted (pi-weighted) pair dependence than the twin for the protected entries
    Ef = st['E_fit']
    tb, k0, l0 = st['twin_fit']
    consistent, obj, fit_mi = True, {}, {}
    for nm in plan:
        b = st['books'][nm]
        kf, lf = st['fit_cells'][nm]
        kr, lr = route(Ef, b)
        consistent &= bool(np.array_equal(kr, kf) and np.array_equal(lr, lf))
        fit_mi[nm] = plugin_mi(kf * b['CO'].shape[0] + lf, Ef['pi'])
        if plan[nm][0] == 'prc':
            bt = dict(b)
            bt.update({'CI': tb['CI'], 'CO': tb['CO'], 'tilt': 0.0})
            set_tables(Ef['pi'], bt, k0, l0)
            obj[nm] = (objective(Ef, b, kf, lf), objective(Ef, bt, k0, l0))
    checks['rule_consistent_fitted_partition'] = consistent
    checks['objective_not_above_twin'] = all(a <= t + 1e-12 for a, t in obj.values())
    checks['protected_lowers_fitted_pair_dependence'] = fit_mi['PRC_B'] < fit_mi['PRC_TWIN'] and fit_mi['PRC_A'] < fit_mi['PRC_TWIN']
    info['fitted_objective_vs_twin'] = obj
    info['fitted_pi_pair_mi'] = fit_mi
    perm = rng.permutation(lab['sex'])
    null = lookup_crossfit(views(*rel['PRC_TWIN'][1:], st['books']['PRC_TWIN']['CO'].shape[0])['pair'], perm, unit, 'selftest_null')
    info['null_auc'] = null['auc']
    checks['lookup_null_near_half'] = abs(null['auc'] - 0.5) < 0.05

    # 3. HQ routing is a function of the task posteriors only; PRC routing may split equal posteriors by propensity
    Eh = {key: v[:200].copy() for key, v in E.items()}
    Eh2 = dict(Eh)
    Eh2['pi'] = np.clip(1 - Eh['pi'], 0.01, 0.99)
    Eh2['qO0'], Eh2['qO1'] = Eh['qO1'], Eh['qO0']
    kh, lh = route(Eh, st['books']['HQ'])
    kh2, lh2 = route(Eh2, st['books']['HQ'])
    kp, lp = route(Eh, st['books']['PRC_B'])
    kp2, lp2 = route(Eh2, st['books']['PRC_B'])
    checks['hq_routing_function_of_q_only'] = bool(np.array_equal(kh, kh2) and np.array_equal(lh, lh2))
    info['prc_reroutes_on_propensity'] = int(((kp != kp2) | (lp != lp2)).sum())
    checks['prc_uses_propensity'] = info['prc_reroutes_on_propensity'] > 0
    # LOC (w_pair = 0): the joint rule separates into recipient-specific rules
    bL = st['books']['LOC']
    checks['loc_has_no_pair_term'] = bL['w_pair'] == 0.0

    # 4. nested estimator (outer household folds) and the predeclared selection rule
    nplan = {'PRC_TWIN': ('twin',), 'P1': ('prc', 'PRC_S2', 0.5), 'P2': ('prc', 'PRC_S2', 2.0), 'P3': ('prc', 'PRC_S2', 8.0)}
    nest = nested(Z, lab, unit, 0, nplan, hp=hp, T_fixed=st['T'], capacity=cap, outer=3)
    info['nested'] = {nm: {key: v for key, v in d.items() if key != '_per_person'} for nm, d in nest.items()}
    checks['nested_finite'] = all(np.isfinite(v) for d in info['nested'].values() for v in d.values())
    chosen, audit = select_path(nest, lab, unit, ['P1', 'P2', 'P3'])
    info['selected'] = chosen
    info['audit'] = {c: {key: [round(a, 5), round(b, 5)] for key, (a, b) in v.items()} for c, v in audit.items()}
    checks['selection_rule_monotone'] = all(
        (chosen[b] is None) or all(audit[c][key][0] + 2 * audit[c][key][1] <= BUDGETS[b][key]
                                   for c in ['P1', 'P2', 'P3'][:['P1', 'P2', 'P3'].index(chosen[b]) + 1] for key in BUDGETS[b])
        for b in BUDGETS)
    info['cpu_s'] = round(time.process_time() - t0, 2)
    passed = all(bool(v) for v in checks.values())
    print(json.dumps({'selftest': 'PASS' if passed else 'FAIL', 'checks': {k: bool(v) for k, v in checks.items()}, 'info': info},
                     indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    return passed


# ---------------------------------------------------------------- real-data entry points (queue only; after the receipt)
def _real(version, mode, operation, purpose, capacity=(16, 64)):
    from nrm import common as NC                                    # coordinator infrastructure (refuses before the receipt)
    from ohn import fam_a as FA
    d = NC.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), operation, purpose)
    Z = FA.compact_po(d['X'], d['feature_names'])
    lab = {'income': d['income'], 'occupation': d['occupation'], 'sex': d['sex']}
    return d, Z, lab


def main(argv=None):
    ap = argparse.ArgumentParser(prog='nrm.fa.prc')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    for c in ('fit', 'nested'):
        p = sub.add_parser(c)
        p.add_argument('--version', type=int, required=True, choices=(0, 1))
        p.add_argument('--operation', required=True)
        p.add_argument('--purpose', required=True)
    a = ap.parse_args(argv)
    from threadpoolctl import threadpool_limits
    if a.cmd == 'selftest':
        with threadpool_limits(limits=1):
            return 0 if selftest() else 1
    raise SystemExit('real-data commands are enabled only after the protocol receipt and the committed mechanism card '
                     '(wave 2; they will call nrm.common.load / nrm.release.save through the queue)')


if __name__ == '__main__':
    sys.exit(main())
