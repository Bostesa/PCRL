"""Designer A mechanism sketch (campaign pcrl_dual_design_release_v1): SEX-STANDARDIZED TASK POSTERIORS (STD).

Self-contained, label-free reference implementation on plain numpy arrays. Nothing here reads a file; `selftest()` uses invented
data only. Real data enter only through the coordinator's queue under a committed card.

Operation (version v; MODEL_FIT only; household folds; Z = ohn.fam_a.compact_po(X), COW and MAR categorical):
  twin       tI(x) = P^(inc | x),  tO(x) = P^(occ | x)                HGB on Z, A1_PO_16 learner settings, T* by inner household CV
  SEX-aware  qI_s(x) = P^(inc | x, s), qO_s(x) = P^(occ | x, s)        HGB on [Z, SEX] (same settings and T* rule); SEX is a
                                                                      TRAINING input only (MODEL_FIT SEX labels)
  weight     w, one fixed scalar shared by both recipients: the value on W_GRID that minimizes the larger of the men's and the
             women's out-of-fold occupation Brier excess of the lam = 1 release over the twin (MODEL_FIT occupation + SEX labels)
  release    P_inc = (1 - lam) tI + lam [(1 - w) qI_0 + w qI_1]
             P_occ = (1 - lam) tO + lam [(1 - w) qO_0 + w qO_1]        decisions = argmax (first class on ties)
  strength   lam = the largest value on LAMBDA_GRID whose out-of-fold spends over the twin meet budget S3 (estimate + 2 household-
             clustered SE, six clauses); None if no value does (nothing is released).
At lam = 1 each released vector is a fitted conditional averaged over SEX with fixed weights: the decomposition term
(pi(x) - w)(q_1 - q_0) of the SEX-blind posterior is absent, and no SEX propensity model is fitted or deployed. lam = 0 is the
equally trained task-only twin (never reads SEX). The teachers use the construction of nrm.fa.prc.fit_teachers (settings, fold
tags, seeds), so the cached TASK.pkl twin and SEXSIDE.pkl 'occ_s' teachers can be reused after a hash check; only the SEX-aware
income teacher is new.

Controls and revisions in the same code:
  release_underfit  UND: the twin's own boosting path stopped at round(frac * T*) iterations (SEX-blind matched-spend control)
  release_comp      COMP: the same SEX-aware components re-weighted by a SEX propensity array pi(x) (structural ablation)
  estimator='ipw'   revision R1: one SEX-blind HGB per task trained with inverse-propensity weights w_s / pi^_s(x) (cross-fitted
                    pi^, clipped to [0.05, 0.95]); same target sum_s w_s P(y | x, s), no counterfactual SEX evaluation; pi^ is a
                    training weight only and is not deployed.
  decisions='twin'  revision R2: decision-preserving release (keep_decisions): the twin's decision class is pooled with every
                    class that exceeds it, so each recipient's argmax equals the twin's.

    python3 mech_sketch.py            runs selftest() (invented data; prints CPU seconds)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')                       # one numerical thread (campaign rule); set before numpy / sklearn load
import hashlib
import time
import warnings

import numpy as np

K_INC, K_OCC = 2, 5
EPS = 1e-6
CAT_Z = np.array([False, False, True, False, True, False])                   # compact_po: COW, MAR native categories
HGB = {'lr': 0.025, 'leaves': 16, 'msl': 100, 'l2': 3.0, 'max_iter': 1000}   # nrm.fa.prc.HGB = A1_PO_16 learner settings
LAMBDA_GRID = (0.5, 0.75, 1.0)
W_GRID = tuple(float(x) for x in np.round(np.arange(0.30, 0.6001, 0.01), 2))
S3 = {'occ_nll': 0.012, 'inc_nll': 0.006, 'occ_brier_men': 0.0045, 'occ_brier_women': 0.0045, 'occ_acc': 0.008,
      'inc_acc': 0.006}                                                      # nrm.fa.prc.BUDGETS['S3'] (spend over the twin)
INNER_TAG, XF_TAG = 'OHN_A_cv_S{v}', 'NRM_FA_xf_S{v}'                        # the nrm.fa.prc.fit_teachers fold tags


# ---------------------------------------------------------------- utilities
def household_folds(unit, k, tag):
    """Deterministic household-separated folds (nrm.fa.prc.household_folds)."""
    groups = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups]
    fold = np.empty(len(groups), dtype=np.int64)
    fold[np.argsort(keys)] = np.arange(len(groups)) % k
    return fold[np.searchsorted(groups, unit)]


def norm(P):
    P = np.clip(np.asarray(P, dtype=np.float64), EPS, 1.0)
    return P / P.sum(1, keepdims=True)


def probs(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - p1, p1])
    e = np.exp(raw - raw.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def brier(P, y):
    E = np.zeros_like(P)
    E[np.arange(len(y)), y] = 1.0
    return ((P - E) ** 2).sum(1)


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64)
    g, inv = np.unique(unit, return_inverse=True)
    s = np.bincount(inv, weights=v - v.mean(), minlength=len(g))
    return float(np.sqrt(len(g) / max(len(g) - 1, 1) * np.sum(s ** 2)) / len(v))


# ---------------------------------------------------------------- HGB teachers (nrm.fa.prc construction)
def _model(hp, T, seed, cat):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(hp['lr']), max_iter=int(T),
                                          max_leaf_nodes=int(hp['leaves']), min_samples_leaf=int(hp['msl']),
                                          l2_regularization=float(hp['l2']), max_bins=255, early_stopping=False,
                                          categorical_features=cat, random_state=int(seed))


def _fit(hp, X, y, K, T, seed, cat, sw=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = _model(hp, T, seed, cat).fit(X, y, sample_weight=sw)
    if not np.array_equal(m.classes_, np.arange(K)):
        raise AssertionError('class support differs in a training set (no repair)')
    return m


def proba(m, X):
    return norm(probs(m.decision_function(X)))


def _x(Z, s):
    """Model input: Z alone (SEX-blind), or [Z, s] with s a column or a constant (SEX-aware evaluation)."""
    Z = np.asarray(Z, dtype=np.float64)
    if s is None:
        return Z, CAT_Z
    return np.column_stack([Z, np.broadcast_to(np.asarray(s, dtype=np.float64), (len(Z),))]), np.append(CAT_Z, False)


def tstar(hp, X, y, K, cat, unit, version, sw=None):
    """First argmin of the pooled inner household 5-fold NLL over boosting iterations (the A1_PO_16 rule)."""
    inner = household_folds(unit, 5, INNER_TAG.format(v=version))
    cur = np.zeros(int(hp['max_iter']))
    for j in range(5):
        tr, va = inner != j, inner == j
        m = _fit(hp, X[tr], y[tr], K, hp['max_iter'], version, cat, None if sw is None else sw[tr])
        wv = np.ones(int(va.sum())) if sw is None else sw[va]
        for i, r in enumerate(m.staged_decision_function(X[va])):
            cur[i] -= np.sum(wv * np.log(norm(probs(r))[np.arange(int(va.sum())), y[va]]))
    return int(np.argmin(cur)) + 1


def teacher(Z, y, K, unit, version, hp, T=None, sex=None, sw=None, oof=True):
    """Full-pool fit plus out-of-fold predictions over 5 household folds at the same T. With `sex`, SEX is an input column and the
    out-of-fold predictions are evaluated at s = 0 and at s = 1."""
    X, cat = _x(Z, sex)
    T = int(T) if T is not None else tstar(hp, X, y, K, cat, unit, version, sw)
    out = {'T': T, 'model': _fit(hp, X, y, K, T, version, cat, sw)}
    if oof:
        xf = household_folds(unit, 5, XF_TAG.format(v=version))
        o = {0: np.zeros((len(y), K)), 1: np.zeros((len(y), K))} if sex is not None else np.zeros((len(y), K))
        for f in range(5):
            tr, te = xf != f, xf == f
            m = _fit(hp, X[tr], y[tr], K, T, version, cat, None if sw is None else sw[tr])
            if sex is None:
                o[te] = proba(m, X[te])
            else:
                for s in (0, 1):
                    o[s][te] = proba(m, _x(Z[te], s)[0])
        out['oof'] = o
    return out


# ---------------------------------------------------------------- fitted weight and strength (MODEL_FIT only)
def guard_balanced_w(tO, qO0, qO1, y_occ, sex, grid=W_GRID):
    """Fixed weight minimizing max(men's, women's) out-of-fold occupation Brier excess of the lam = 1 release over the twin; ties
    go to the value nearest the SEX share. Returns (w, {'men', 'women'} excesses at w)."""
    base, men, wom, share = brier(tO, y_occ), sex == 0, sex == 1, float(np.mean(sex))
    best = None
    for w in grid:
        d = brier(norm((1 - w) * qO0 + w * qO1), y_occ) - base
        key = (round(max(d[men].mean(), d[wom].mean()), 12), abs(w - share))
        if best is None or key < best[0]:
            best = (key, w, float(d[men].mean()), float(d[wom].mean()))
    return float(best[1]), {'men': best[2], 'women': best[3]}


def clauses(P_inc, P_occ, tI, tO, y_inc, y_occ, sex, unit):
    """Six spend clauses over the twin: per-person differences, mean and household-clustered SE."""
    ar = np.arange(len(y_occ))
    d = {'occ_nll': np.log(tO[ar, y_occ]) - np.log(P_occ[ar, y_occ]),
         'inc_nll': np.log(tI[ar, y_inc]) - np.log(P_inc[ar, y_inc]),
         'occ_acc': (tO.argmax(1) == y_occ).astype(float) - (P_occ.argmax(1) == y_occ),
         'inc_acc': (tI.argmax(1) == y_inc).astype(float) - (P_inc.argmax(1) == y_inc)}
    out = {k: (float(v.mean()), cluster_se(v, unit)) for k, v in d.items()}
    db = brier(P_occ, y_occ) - brier(tO, y_occ)
    for nm, g in (('occ_brier_men', sex == 0), ('occ_brier_women', sex == 1)):
        out[nm] = (float(db[g].mean()), cluster_se(db[g], unit[g]))
    return out


def keep_decisions(P, D):
    """R2: pool class D with every class whose probability exceeds P[D] (common value = their mean), then give D +1e-9, so the
    argmax is D and every other coordinate is unchanged."""
    P = np.array(P, dtype=np.float64)
    ar = np.arange(len(P))
    over = P > P[ar, D][:, None]
    over[ar, D] = True
    m = (P * over).sum(1) / over.sum(1)
    P = np.where(over, m[:, None], P)
    P[ar, D] += 1e-9
    return P / P.sum(1, keepdims=True)


def select_strength(table, budget=S3, k_se=2.0):
    """Largest lam whose every clause meets estimate + k_se * SE <= budget; None if none does."""
    ok = [lam for lam in table if all(table[lam][c][0] + k_se * table[lam][c][1] <= budget[c] for c in budget)]
    return max(ok) if ok else None


# ---------------------------------------------------------------- fit and release
def fit(Z, y_inc, y_occ, sex, unit, version, strength, w='balanced', hp=None, T=None, estimator='plugin', grid=LAMBDA_GRID,
        budget=S3, decisions='own', **kw):
    """strength: a float in [0, 1] or 'select' (the MODEL_FIT out-of-fold rule over `grid`). w: 'balanced', 'share' or a float
    (estimator 'ipw' needs 'share' or a float, normally the plug-in path's balanced w). T: optional fixed iteration counts
    {'inc', 'occ', 'inc_s', 'occ_s', 'sex', 'inc_w', 'occ_w'}; missing ones are chosen by inner household CV. decisions: 'own'
    (argmax of the released vector) or 'twin' (revision R2, keep_decisions)."""
    hp, T = (hp or HGB), dict(T or {})
    Z, unit = np.asarray(Z, dtype=np.float64), np.asarray(unit)
    y_inc, y_occ = np.asarray(y_inc, dtype=np.int64), np.asarray(y_occ, dtype=np.int64)
    twin_only = strength != 'select' and float(strength) == 0.0
    need_oof = (not twin_only) and (strength == 'select' or w == 'balanced')
    ti = teacher(Z, y_inc, K_INC, unit, version, hp, T.get('inc'), oof=need_oof)
    to = teacher(Z, y_occ, K_OCC, unit, version, hp, T.get('occ'), oof=need_oof)
    st = {'schema': 'DDR-A-STD-v1', 'version': int(version), 'estimator': estimator, 'twin_inc': ti['model'],
          'twin_occ': to['model'], 'T': {'inc': ti['T'], 'occ': to['T']}, 'lambda': 0.0, 'w': None, 'decisions': decisions}
    if twin_only:
        return st                                                  # the task-only twin: SEX is never read
    sex = np.asarray(sex, dtype=np.int64)
    if estimator == 'plugin':
        si = teacher(Z, y_inc, K_INC, unit, version, hp, T.get('inc_s'), sex=sex, oof=need_oof)
        so = teacher(Z, y_occ, K_OCC, unit, version, hp, T.get('occ_s'), sex=sex, oof=need_oof)
        st.update(std_inc=si['model'], std_occ=so['model'])
        st['T'].update(inc_s=si['T'], occ_s=so['T'])
        if w == 'balanced':
            w, st['guard_oof_at_lam1'] = guard_balanced_w(to['oof'], so['oof'][0], so['oof'][1], y_occ, sex)
        w = float(np.mean(sex)) if w == 'share' else float(w)
        if need_oof:
            rI1 = norm((1 - w) * si['oof'][0] + w * si['oof'][1])
            rO1 = norm((1 - w) * so['oof'][0] + w * so['oof'][1])
    elif estimator == 'ipw':                                           # revision R1
        if w == 'balanced':
            raise ValueError("estimator 'ipw' needs w = 'share' or a float (the plug-in path's balanced w)")
        w = float(np.mean(sex)) if w == 'share' else float(w)
        ps = teacher(Z, sex, 2, unit, version, hp, T.get('sex'), oof=True)          # P(SEX | x), cross-fitted; weights only
        pi = np.clip(ps['oof'][:, 1], 0.05, 0.95)
        om = np.where(sex == 1, w / pi, (1 - w) / (1 - pi))
        om = om / om.mean()
        ii = teacher(Z, y_inc, K_INC, unit, version, hp, T.get('inc_w'), sw=om, oof=need_oof)
        io = teacher(Z, y_occ, K_OCC, unit, version, hp, T.get('occ_w'), sw=om, oof=need_oof)
        st.update(std_inc=ii['model'], std_occ=io['model'], ipw_ess_share=float(om.sum() ** 2 / np.sum(om ** 2) / len(om)))
        st['T'].update(sex=ps['T'], inc_w=ii['T'], occ_w=io['T'])
        if need_oof:
            rI1, rO1 = ii['oof'], io['oof']
    else:
        raise ValueError(f'unknown estimator {estimator}')
    st['w'] = w
    if strength == 'select':
        def oof_release(lam):
            pI, pO = norm((1 - lam) * ti['oof'] + lam * rI1), norm((1 - lam) * to['oof'] + lam * rO1)
            if decisions == 'twin':
                pI, pO = keep_decisions(pI, ti['oof'].argmax(1)), keep_decisions(pO, to['oof'].argmax(1))
            return pI, pO
        table = {lam: clauses(*oof_release(lam), ti['oof'], to['oof'], y_inc, y_occ, sex, unit) for lam in grid}
        st['lambda'] = select_strength(table, budget)
        st['selection'] = {'table': table, 'budget': dict(budget), 'selected': st['lambda']}
    else:
        st['lambda'] = float(strength)
    return st


def _standardized(state, Z):
    if state['estimator'] == 'plugin':
        w = state['w']
        rI = (1 - w) * proba(state['std_inc'], _x(Z, 0)[0]) + w * proba(state['std_inc'], _x(Z, 1)[0])
        rO = (1 - w) * proba(state['std_occ'], _x(Z, 0)[0]) + w * proba(state['std_occ'], _x(Z, 1)[0])
        return rI, rO
    return proba(state['std_inc'], Z), proba(state['std_occ'], Z)


def release(state, Z):
    """Released (P_inc (n, 2), P_occ (n, 5)) from the six fields alone; decisions are their argmax."""
    lam = state['lambda']
    if lam is None:
        raise ValueError('no strength met the predeclared budget: nothing is released')
    Z = np.asarray(Z, dtype=np.float64)
    tI, tO = proba(state['twin_inc'], Z), proba(state['twin_occ'], Z)
    if lam == 0.0:
        return tI, tO
    rI, rO = _standardized(state, Z)
    pI, pO = norm((1 - lam) * tI + lam * rI), norm((1 - lam) * tO + lam * rO)
    if state.get('decisions', 'own') == 'twin':                      # revision R2
        pI, pO = keep_decisions(pI, tI.argmax(1)), keep_decisions(pO, tO.argmax(1))
    return pI, pO


def release_underfit(state, Z, frac_inc, frac_occ):
    """Control UND: the twin's own boosting path stopped at round(frac * T*) iterations (SEX-blind; spend matched on MODEL_FIT)."""
    out = []
    for key, frac in (('twin_inc', frac_inc), ('twin_occ', frac_occ)):
        m = state[key]
        t = max(1, int(round(float(frac) * m.n_iter_)))
        for i, r in enumerate(m.staged_decision_function(np.asarray(Z, dtype=np.float64)), 1):
            if i == t:
                out.append(norm(probs(r)))
                break
    return out[0], out[1]


def release_comp(state, Z, pi):
    """Ablation COMP: the same SEX-aware components weighted by a SEX propensity array pi (the SEX-blind posterior they imply)."""
    pi = np.asarray(pi, dtype=np.float64)[:, None]
    out = []
    for key in ('std_inc', 'std_occ'):
        out.append(norm((1 - pi) * proba(state[key], _x(Z, 0)[0]) + pi * proba(state[key], _x(Z, 1)[0])))
    return out[0], out[1]


# ---------------------------------------------------------------- selftest (invented data only)
def _invent(n, rng):
    """Invented law: SEX first; WKHP-like column = task-relevant proxy, last column = PURE proxy (no task effect). Overlap is
    realistic (true propensity 5-95% range about 0.13-0.85). With a far stronger proxy (range 0.01-0.98) the plug-in HGB kept
    0.89 of the twin's movement under a pure-proxy shift (boosted SEX-aware trees attribute part of the SEX effect to a
    correlated field) and clipped IPW was biased; see PROPOSAL.md, estimator caveat and revision R1."""
    sex = (rng.random(n) < 0.48).astype(np.int64)
    agep = rng.integers(18, 80, n).astype(float)
    wkhp = rng.normal(40.0 - 5.0 * sex, 9.0)
    cow = rng.integers(0, 8, n).astype(float)
    schl = rng.integers(16, 25, n).astype(float)
    mar = rng.integers(0, 5, n).astype(float)
    prox = rng.normal(0.8 * sex, 1.0)
    Z = np.column_stack([agep, wkhp, cow, schl, mar, prox])
    lo = np.log(0.48 / 0.52) + (-(wkhp - 35.0) ** 2 + (wkhp - 40.0) ** 2) / 162.0 + (-(prox - 0.8) ** 2 + prox ** 2) / 2.0
    pi_true = 1.0 / (1.0 + np.exp(-lo))
    L = np.column_stack([0.25 * (schl - 20), -0.5 + 0.03 * (agep - 45), -0.3 + 1.2 * sex + 0.05 * (40 - wkhp),
                         -1.0 - 2.0 * sex + 0.02 * (wkhp - 40), -1.0 - 1.0 * sex + 0.2 * (20 - schl)])
    P = probs(L)
    y_occ = (P.cumsum(1) < rng.random(n)[:, None]).sum(1)
    y_inc = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.06 * (wkhp - 35) + 0.3 * (schl - 20) - 0.7 * sex)))).astype(np.int64)
    return Z, y_inc, np.minimum(y_occ, K_OCC - 1), sex, np.arange(n) // 2, pi_true


def _soft_auc(score, p):
    o = np.argsort(score, kind='mergesort')
    s, a = score[o], p[o]
    b = 1.0 - a
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    A, B, AB = np.add.reduceat(a, start), np.add.reduceat(b, start), np.add.reduceat(a * b, start)
    below = np.r_[0.0, np.cumsum(B)[:-1]]
    return float((np.sum(A * below) + 0.5 * np.sum(A * B - AB)) / (a.sum() * b.sum() - np.sum(a * b)))


def _reader_auc(views, sex, pi_true, half):
    from sklearn.linear_model import LogisticRegression
    F = np.column_stack([np.log(v) for v in views])
    mu, sd = F[half].mean(0), F[half].std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=1000).fit((F[half] - mu) / sd, sex[half])
    return _soft_auc(m.decision_function((F[~half] - mu) / sd), pi_true[~half])


def selftest():
    t0 = time.process_time()
    rng = np.random.default_rng(7)
    Z, yi, yo, sex, unit, pi_true = _invent(4000, rng)
    hp = {'lr': 0.1, 'leaves': 8, 'msl': 20, 'l2': 1.0, 'max_iter': 60}
    Tfix = {'inc': 40, 'inc_s': 40, 'inc_w': 40}         # income fixed for speed; occupation, SEX-aware and SEX models by inner CV
    checks = {}
    # 1. twin (lam = 0): valid vectors, SEX never read (bitwise identical with permuted SEX), no SEX-aware model in the state
    tw = fit(Z, yi, yo, sex, unit, 0, 0.0, hp=hp, T=Tfix)
    tw_perm = fit(Z, yi, yo, rng.permutation(sex), unit, 0, 0.0, hp=hp, T=Tfix)
    tI, tO = release(tw, Z)
    checks['twin_valid'] = tI.shape == (4000, 2) and tO.shape == (4000, 5) and np.allclose(tO.sum(1), 1) and np.isfinite(tO).all()
    checks['twin_never_reads_sex'] = all(np.array_equal(a, b) for a, b in zip(release(tw, Z), release(tw_perm, Z))) \
        and not any(k.startswith('std_') for k in tw)
    # 2. STD with the MODEL_FIT selection rule (lenient budget for the mechanics; the default S3 must return a grid value or None)
    big = {k: 10 * v for k, v in S3.items()}
    st = fit(Z, yi, yo, sex, unit, 0, 'select', hp=hp, T={**Tfix, 'occ': tw['T']['occ']}, budget=big)
    checks['selection_rule'] = st['lambda'] == 1.0 and select_strength(st['selection']['table'], S3) in (None,) + LAMBDA_GRID
    checks['twin_inside_std_equals_twin'] = np.array_equal(proba(st['twin_occ'], Z), tO)
    # 3. identities: lam = 1 is the fixed-weight SEX average; lam = 0.5 is the mixture; deterministic; no propensity model
    P1 = release(st, Z)
    w = st['w']
    direct = norm((1 - w) * proba(st['std_occ'], _x(Z, 0)[0]) + w * proba(st['std_occ'], _x(Z, 1)[0]))
    checks['lam1_is_fixed_weight_average'] = np.allclose(P1[1], direct, atol=1e-12)
    st_half = dict(st, **{'lambda': 0.5})
    checks['lam_path_is_mixture'] = np.allclose(release(st_half, Z)[1], norm(0.5 * tO + 0.5 * P1[1]), atol=1e-12)
    checks['deterministic'] = all(np.array_equal(a, b) for a, b in zip(release(st, Z), release(st, Z)))
    checks['no_propensity_model_deployed'] = not any(k in st for k in ('pi', 'sex_model', 'propensity'))
    # 4. guard-balanced weight is on the grid and no worse (max within-SEX excess) than the grid point nearest the SEX share
    share_w = min(W_GRID, key=lambda g: abs(g - sex.mean()))
    so_oof = teacher(Z, yo, K_OCC, unit, 0, hp, st['T']['occ_s'], sex=sex)['oof']
    to_oof = teacher(Z, yo, K_OCC, unit, 0, hp, tw['T']['occ'])['oof']
    _, g_share = guard_balanced_w(to_oof, so_oof[0], so_oof[1], yo, sex, grid=(share_w,))
    checks['balanced_w_on_grid_and_no_worse'] = w in W_GRID and max(st['guard_oof_at_lam1'].values()) <= max(g_share.values()) + 1e-12
    # 5. removal: the pure-proxy column moves the twin's occupation output, the STD output much less; readers recover less
    Zs = Z.copy()
    Zs[:, 5] += 2.0
    mv = lambda s: float(np.mean(np.abs(np.log(release(s, Zs)[1]) - np.log(release(s, Z)[1]))))
    checks['pure_proxy_dependence_ratio'] = round(mv(st) / mv(tw), 3)
    half = np.zeros(4000, bool)
    half[rng.permutation(4000)[:2000]] = True
    a_tw, a_st = _reader_auc([tO], sex, pi_true, half), _reader_auc([P1[1]], sex, pi_true, half)
    p_tw, p_st = _reader_auc([tI, tO], sex, pi_true, half), _reader_auc(list(P1), sex, pi_true, half)
    checks['reader_auc_occ_twin_std'] = (round(a_tw, 4), round(a_st, 4))
    checks['reader_auc_pair_twin_std'] = (round(p_tw, 4), round(p_st, 4))
    # 6. controls: UND at frac 1 is the twin; COMP with pi = w everywhere equals STD at lam = 1
    u = release_underfit(st, Z, 1.0, 1.0)
    u5 = release_underfit(st, Z, 0.5, 0.5)
    checks['underfit_frac1_is_twin'] = np.allclose(u[1], tO, atol=1e-12) and not np.allclose(u5[1], tO)
    checks['comp_at_constant_pi_is_std'] = np.allclose(release_comp(st, Z, np.full(4000, w))[1], P1[1], atol=1e-12)
    # 7. revision R1 (IPW): the same target, so closer to the plug-in standardized release than the twin is
    ip = fit(Z, yi, yo, sex, unit, 0, 1.0, w=w, hp=hp, T={**Tfix, 'occ': tw['T']['occ']}, estimator='ipw')
    kl = lambda P, Q: float(np.mean((P * np.log(P / Q)).sum(1)))
    checks['ipw_kl_to_std_vs_twin'] = (round(kl(P1[1], release(ip, Z)[1]), 4), round(kl(P1[1], tO), 4))
    checks['ipw_ess_share'] = round(ip['ipw_ess_share'], 3)
    # 8. revision R2: decisions equal the twin's, vectors stay valid, and only rows whose argmax changed are altered
    r2 = release(dict(st, decisions='twin'), Z)
    moved = P1[1].argmax(1) != tO.argmax(1)
    checks['r2_keeps_twin_decisions'] = (np.array_equal(r2[0].argmax(1), tI.argmax(1)) and np.array_equal(r2[1].argmax(1), tO.argmax(1))
                                         and np.allclose(r2[1].sum(1), 1) and np.allclose(r2[1][~moved], P1[1][~moved], atol=1e-8)
                                         and bool(moved.any()))
    ok = (checks['twin_valid'] and checks['twin_never_reads_sex'] and checks['selection_rule'] and checks['twin_inside_std_equals_twin']
          and checks['r2_keeps_twin_decisions']
          and checks['lam1_is_fixed_weight_average'] and checks['lam_path_is_mixture'] and checks['deterministic']
          and checks['no_propensity_model_deployed'] and checks['balanced_w_on_grid_and_no_worse']
          and checks['pure_proxy_dependence_ratio'] < 0.6 and a_st < a_tw - 0.02 and p_st < p_tw - 0.01
          and checks['underfit_frac1_is_twin'] and checks['comp_at_constant_pi_is_std']
          and checks['ipw_kl_to_std_vs_twin'][0] < checks['ipw_kl_to_std_vs_twin'][1])
    checks['w'], checks['T'] = w, st['T']
    for k, v in checks.items():
        print(f'{k}: {v}')
    print('SELFTEST', 'PASS' if ok else 'FAIL', f'CPU_SECONDS {time.process_time() - t0:.2f}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if selftest() else 1)
