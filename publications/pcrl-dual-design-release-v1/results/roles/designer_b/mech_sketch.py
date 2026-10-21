"""Designer B reference sketch (campaign pcrl_dual_design_release_v1): PTC, propensity-tilt cancellation.
Label-free selftest on invented data only:  python3 mech_sketch.py

Operation (deterministic in the six fields x; the task models are FIXED, equally trained and task-only):
  p_I(x), p_O(x)  task-only posteriors (income 2, occupation 5)            MODEL_FIT task labels (the twin's teachers)
  q(x)            P(SEX = 1 | x), SEX-side HGB                               MODEL_FIT SEX labels (card-justified)
  k(x)            = argmax p_O(x), the twin's occupation decision (5 strata; a function of the twin's release)
  release         log r_I = log p_I - alpha (q - qbar)    w_I      + const      (income: one global class-weight vector)
                  log r_O = log p_O - alpha (q - qbar_k)  W_O[k]   + const      (occupation: one class-weight vector per stratum)
                  i.e. exponential tilts r ∝ p * exp(-alpha * amplitude * weights); decisions = argmax of the released vectors.
  weights         closed form on MODEL_FIT OUT-OF-FOLD statistics with the SEX label s (c = centred log-probabilities):
                      w_I     = Cov(c(p_I), s) / Cov(q, s)                    (all of MODEL_FIT)
                      W_O[k]  = Cov(c(p_O), s | k) / Cov(q, s | k),  qbar_k = mean(q | k)
                  so at alpha = 1 the released centred log-probabilities have zero covariance with SEX: income overall,
                  occupation within each decision stratum (exact on MODEL_FIT out-of-fold; approximate on new people). The tower
                  identity Cov(f(x), s) = Cov(f(x), P(s | x)) is why a runtime propensity can cancel a covariance with the unseen SEX.
                  A stratum with fewer than MIN_STRATUM people or Cov(q, s | k) <= 0 falls back to the global occupation vector.
  alpha           STRENGTHS; selected on MODEL_FIT nested household folds only (PROPOSAL.md section 1).
Protocol controls (PROTOCOL.md section 4; two MODEL_FIT spend bracket points each):
  SBC             fit() with sbc_sex(sex) (fixed-seed permutation of the MODEL_FIT SEX column) in place of SEX: propensity model
                  and weights refit by the same rule; strength chosen on MODEL_FIT spend. Named fallback if degenerate or short of
                  the candidate's spend: fallback_target(Z) (keyed Bernoulli(0.5) label per (SCHL, COW) cell).
  SPEND           not in this sketch: SEX-blind KL k-means quantization of the twin's released vectors (nrm.fa.prc twin_book /
                  B1 out-of-fold cell means), codebook sizes bracketing the candidate's MODEL_FIT spend (information-removing).
Further controls / ablations (same code, kw mode / stratify):
  'twin'          alpha = 0: the task-only posteriors unchanged; never fits or reads SEX.
  'shuf'          SAME weights, strata and tilt law; the person's q replaced by q(donor(x)), donor(x) = keyed SHA-256 of the six
                  fields into the MODEL_FIT out-of-fold bank (deterministic; no information about this person's SEX); scale
                  matched on MODEL_FIT nested spend (kw shuf_scale).
  'ztilt'         relabeling ablation: q replaced by q_z = P(SEX | the twin's released log-probabilities) (HGB, cross-fitted),
                  its own weights by the same rule; the release is a function of the twin's release alone (kw scale).
  stratify=False  global occupation tilt (one W_O for all strata): ablation of the stratification.
"""
from __future__ import annotations

import hashlib

import numpy as np

STRENGTHS = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
FLOOR = 1e-6          # probability floor inside the logarithm (as nrm.fa.prc CLIP)
MIN_STRATUM = 200     # smallest occupation-decision stratum with its own weight vector


def softmax(l):
    l = np.asarray(l, np.float64)
    e = np.exp(l - l.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def clog(p):
    """Centred log-probabilities (n, k)."""
    lp = np.log(np.maximum(np.asarray(p, np.float64), FLOOR))
    return lp - lp.mean(1, keepdims=True)


def _weights(c, q, s):
    sc = s - s.mean()
    cqs = float(np.mean((q - q.mean()) * sc))
    return (c * sc[:, None]).mean(0) / cqs, float(q.mean()), cqs


def fit_from_stats(stats, sex=None, sex_expect=None, strength=1.0, stratify=True):
    """Core fit from MODEL_FIT out-of-fold statistics {'p_inc', 'p_occ', 'q'}. `sex` = 0/1 SEX labels (the real fit);
    `sex_expect` = a propensity standing in for the labels (label-free feasibility arithmetic only)."""
    s = np.asarray(sex if sex is not None else sex_expect, np.float64)
    q = np.asarray(stats['q'], np.float64)
    w_inc, qbar, cqs = _weights(clog(stats['p_inc']), q, s)
    if not cqs > 0:
        raise ValueError('propensity not positively associated with SEX on MODEL_FIT')
    w_glob, _, _ = _weights(clog(stats['p_occ']), q, s)
    dec = np.asarray(stats['p_occ']).argmax(1)
    W, qb, fallback = np.tile(w_glob, (5, 1)), np.full(5, qbar), []
    if stratify:
        for k in range(5):
            m = dec == k
            if m.sum() >= MIN_STRATUM:
                wk, qk, ck = _weights(clog(stats['p_occ'][m]), q[m], s[m])
                if ck > 0:
                    W[k], qb[k] = wk, qk
                    continue
            fallback.append(k)
    st = {'w_inc': w_inc, 'qbar': qbar, 'W_occ': W, 'qbar_occ': qb, 'alpha': float(strength), 'stratify': bool(stratify),
          'fallback_strata': fallback, 'stratum_sizes': np.bincount(dec, minlength=5).tolist()}
    r_i, r_o = release_from_stats(dict(st, alpha=1.0), stats)
    sc = s - s.mean()
    st['resid_cov_inc_alpha1'] = float(np.abs((clog(r_i) * sc[:, None]).mean(0)).max())
    st['resid_cov_occ_alpha1'] = float(max(np.abs((clog(r_o[dec == k]) * (s[dec == k] - s[dec == k].mean())[:, None]).mean(0)).max()
                                           for k in range(5) if (dec == k).sum() > 1)) if stratify else \
        float(np.abs((clog(r_o) * sc[:, None]).mean(0)).max())
    return st


def release_from_stats(state, stats, q_override=None, scale=None):
    """Released (P_inc, P_occ) from runtime statistics. `q_override` replaces the person's propensity in the amplitude
    (controls); `scale` overrides alpha. Strata come from the twin's own occupation decision."""
    q = np.asarray(stats['q'] if q_override is None else q_override, np.float64)
    alpha = state['alpha'] if scale is None else float(scale)
    p_i, p_o = np.asarray(stats['p_inc'], np.float64), np.asarray(stats['p_occ'], np.float64)
    if alpha == 0.0:
        return p_i, p_o
    k = p_o.argmax(1)
    r_i = softmax(np.log(np.maximum(p_i, FLOOR)) - alpha * (q - state['qbar'])[:, None] * state['w_inc'][None, :])
    r_o = softmax(np.log(np.maximum(p_o, FLOOR)) - alpha * (q - state['qbar_occ'][k])[:, None] * state['W_occ'][k])
    return r_i, r_o


def sbc_sex(sex, seed=20261021):
    """Protocol SBC input (PROTOCOL.md section 4): a fixed-seed permutation of the MODEL_FIT SEX column. The SBC is fit() with this
    column in place of SEX (propensity model AND weights refit), strength chosen on MODEL_FIT spend (two bracket points)."""
    return np.asarray(sex)[np.random.default_rng(seed).permutation(len(sex))]


def fallback_target(Z, seed=20261021, cols=(3, 2)):
    """Named fallback SBC target, used if the permuted column is degenerate (Cov(q, s) <= 1e-4 on MODEL_FIT out-of-fold, i.e.
    fit() raises) or cannot reach the candidate's MODEL_FIT spend at alpha <= 50: a keyed Bernoulli(0.5) label per
    (SCHL, COW) cell of the compact coding (SEX-independent by construction, learnable from x, same binary type)."""
    Z = np.asarray(Z, np.float64)
    out = np.empty(len(Z), np.int64)
    for i, r in enumerate(Z[:, list(cols)]):
        out[i] = hashlib.sha256(f'DDR_B_SBC_FALLBACK|{seed}|{r[0]:.6g}|{r[1]:.6g}'.encode()).digest()[0] & 1
    return out


def donor_index(Z, key, n_bank):
    """Keyed SHA-256 of each row's six fields -> a MODEL_FIT bank row (deterministic, SEX-free)."""
    Z = np.ascontiguousarray(np.asarray(Z, np.float64))
    return np.array([int.from_bytes(hashlib.sha256(key.encode() + r.tobytes()).digest()[:8], 'big') % n_bank for r in Z])


def donor_q(q, key):
    """Feasibility version of the SHUF propensity: a keyed permutation of the bank's propensities."""
    q = np.asarray(q, np.float64)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
    return q[np.random.default_rng(seed).permutation(len(q))]


# ------------------------------------------------------------------ fit / release on plain arrays
def _hgb(seed, cat):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=16, min_samples_leaf=20, l2_regularization=3.0,
                                          max_iter=120, categorical_features=cat, random_state=seed)


def _folds(unit, k, tag):
    g = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{x}'.encode()).hexdigest() for x in g]
    f = np.empty(len(g), np.int64)
    f[np.argsort(keys)] = np.arange(len(g)) % k
    return f[np.searchsorted(g, unit)]


def _zfeat(p_inc, p_occ):
    return np.column_stack([clog(p_inc), clog(p_occ)])


def _stats(state, Z):
    m = state['models']
    st = {'p_inc': m['inc'].predict_proba(Z), 'p_occ': m['occ'].predict_proba(Z)}
    if state['mode'] in ('ptc', 'shuf'):
        st['q'] = np.clip(m['sex'].predict_proba(Z)[:, 1], 0.01, 0.99)
    elif state['mode'] == 'ztilt':
        st['q'] = np.clip(m['sex_z'].predict_proba(_zfeat(st['p_inc'], st['p_occ']))[:, 1], 0.01, 0.99)
    return st


def fit(Z, y_inc, y_occ, sex, unit, version, strength, **kw):
    """MODEL_FIT fit. kw: mode in {'ptc' (the proposal), 'twin', 'shuf', 'ztilt'}; stratify (default True); cat (categorical
    mask of Z); shuf_scale / scale (spend-matched scale of a control); k (household folds, default 5); models / oof
    (precomputed task and SEX-side teachers and their out-of-fold statistics, e.g. the cached TASK.pkl / SEXSIDE.pkl)."""
    mode, cat, k = kw.get('mode', 'ptc'), kw.get('cat'), int(kw.get('k', 5))
    Z = np.asarray(Z, np.float64)
    lab = {'inc': np.asarray(y_inc), 'occ': np.asarray(y_occ), 'sex': np.asarray(sex)}
    names = ('inc', 'occ') if mode in ('twin', 'ztilt') else ('inc', 'occ', 'sex')
    models = kw.get('models') or {nm: _hgb(int(version), cat).fit(Z, lab[nm]) for nm in names}
    state = {'mode': mode, 'models': dict(models), 'version': int(version), 'alpha': 0.0 if mode == 'twin' else float(strength)}
    if mode == 'twin':                                                   # the twin never fits or reads SEX
        return state
    fold = _folds(np.asarray(unit), k, f'DDR_B_xf_V{int(version)}')
    oof = kw.get('oof')
    if oof is None:
        oof = {'p_inc': np.zeros((len(Z), 2)), 'p_occ': np.zeros((len(Z), 5)), 'q': np.zeros(len(Z))}
        for f in range(k):
            tr, te = fold != f, fold == f
            fm = {nm: _hgb(int(version), cat).fit(Z[tr], lab[nm][tr]) for nm in names}
            oof['p_inc'][te], oof['p_occ'][te] = fm['inc'].predict_proba(Z[te]), fm['occ'].predict_proba(Z[te])
            if 'sex' in fm:
                oof['q'][te] = np.clip(fm['sex'].predict_proba(Z[te])[:, 1], 0.01, 0.99)
    oof = dict(oof)
    if mode == 'ztilt':                                                  # propensity from the twin's released posteriors only
        zf = _zfeat(oof['p_inc'], oof['p_occ'])
        qz = np.zeros(len(Z))
        for f in range(k):
            tr, te = fold != f, fold == f
            qz[te] = _hgb(int(version), None).fit(zf[tr], lab['sex'][tr]).predict_proba(zf[te])[:, 1]
        oof['q'] = np.clip(qz, 0.01, 0.99)
        state['models']['sex_z'] = _hgb(int(version), None).fit(zf, lab['sex'])
    state.update(fit_from_stats(oof, sex=lab['sex'], strength=strength, stratify=kw.get('stratify', True)))
    if mode == 'shuf':
        state['bank_q'] = np.asarray(oof['q'], np.float64).copy()
        state['key'] = f'DDR_B_SHUF|V{int(version)}'
        state['alpha'] = float(kw.get('shuf_scale', strength))
    if mode == 'ztilt':
        state['alpha'] = float(kw.get('scale', strength))
    return state


def release(state, Z):
    Z = np.asarray(Z, np.float64)
    st = _stats(state, Z)
    if state['mode'] == 'twin':
        return st['p_inc'], st['p_occ']
    qo = None
    if state['mode'] == 'shuf':
        qo = state['bank_q'][donor_index(Z, state['key'], len(state['bank_q']))]
    return release_from_stats(state, st, q_override=qo)


# ------------------------------------------------------------------ selftest (invented data only)
def _invented(n, rng):
    """Invented population: SEX shifts hours and the occupation mix; one field predicts SEX only; households in pairs."""
    sex = (rng.random(n) < 0.5).astype(int)
    age = rng.integers(18, 70, n).astype(float)
    hours = np.clip(rng.normal(40 - 6 * sex, 8, n), 1, 99)
    school = rng.integers(0, 6, n).astype(float)
    cow = rng.integers(0, 4, n).astype(float)
    mar = rng.integers(0, 3, n).astype(float)
    proxy = sex * 0.8 + rng.normal(0, 1, n)
    Z = np.column_stack([age, hours, cow, school, mar, proxy])
    y_inc = (0.08 * (hours - 38) + 0.5 * school - 1.5 + rng.logistic(size=n) > 0).astype(int)
    lo = np.column_stack([0.4 * school, 0.6 * sex, 0.9 * sex + 0.2 * mar, 1.2 * (1 - sex) - 0.2 * school,
                          0.8 * (1 - sex) + 0.03 * (hours - 40)]) + rng.gumbel(size=(n, 5))
    return Z, y_inc, lo.argmax(1), sex, np.arange(n) // 2


def selftest():
    import os
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(v, '1')
    rng = np.random.default_rng(7)
    Z, yi, yo, s, u = _invented(4000, rng)
    cat = np.array([False, False, True, False, True, False])
    ok = []
    st = fit(Z, yi, yo, s, u, 0, 1.0, cat=cat)
    # 1. exact cancellation at alpha = 1 on the fitted out-of-fold statistics (income overall, occupation within strata)
    ok.append(('cov_cancelled', bool(st['resid_cov_inc_alpha1'] < 1e-9 and st['resid_cov_occ_alpha1'] < 1e-9)))
    Zn, _, _, sn, _ = _invented(3000, np.random.default_rng(8))
    Pi, Po = release(st, Zn)
    ok.append(('shapes', Pi.shape == (3000, 2) and Po.shape == (3000, 5)))
    ok.append(('simplex', bool(np.allclose(Pi.sum(1), 1) and np.allclose(Po.sum(1), 1) and (Po > 0).all())))
    ok.append(('deterministic', bool(all(np.array_equal(x, y) for x, y in zip((Pi, Po), release(st, Zn))))))
    # 2. the twin never reads SEX
    tw = fit(Z, yi, yo, s, u, 0, 1.0, cat=cat, mode='twin')
    tw2 = fit(Z, yi, yo, rng.permutation(s), u, 0, 1.0, cat=cat, mode='twin')
    ok.append(('twin_sex_free', bool(np.array_equal(release(tw, Zn)[1], release(tw2, Zn)[1]))))
    # 3. new invented people: the mean released log-ratio gap between SEX groups shrinks against the twin (invented law only)
    gap = lambda P: sum(np.abs(clog(x)[sn == 1].mean(0) - clog(x)[sn == 0].mean(0)).sum() for x in P)
    ok.append(('gap_shrinks', bool(gap((Pi, Po)) < 0.5 * gap(release(tw, Zn)))))
    # 4. alpha = 0 releases the task-only posteriors unchanged
    ok.append(('alpha0_identity', bool(np.array_equal(release(dict(st, alpha=0.0), Zn)[1], st['models']['occ'].predict_proba(Zn)))))
    # 5. controls: shuf (keyed donors, same weights), ztilt (propensity from the twin's posteriors only), global ablation
    sh = fit(Z, yi, yo, s, u, 0, 1.0, cat=cat, mode='shuf', shuf_scale=1.0, models=st['models'])
    ok.append(('shuf_same_weights', bool(np.array_equal(sh['W_occ'], st['W_occ']) and not np.array_equal(release(sh, Zn)[1], Po))))
    zt = fit(Z, yi, yo, s, u, 0, 1.0, cat=cat, mode='ztilt', models={'inc': st['models']['inc'], 'occ': st['models']['occ']})
    ok.append(('ztilt_runs', bool(release(zt, Zn)[1].shape == (3000, 5) and zt['resid_cov_occ_alpha1'] < 1e-9)))
    gl = fit(Z, yi, yo, s, u, 0, 1.0, cat=cat, stratify=False, models=st['models'])
    ok.append(('global_ablation', bool(np.allclose(gl['W_occ'], gl['W_occ'][0]) and gl['resid_cov_occ_alpha1'] < 1e-9)))
    # 6. protocol SBC: permuted SEX column (may be degenerate -> documented ValueError), and the named fallback target
    sp = sbc_sex(s)
    try:
        sb = fit(Z, yi, yo, sp, u, 0, 1.0, cat=cat)
        sbc_state = 'fitted'
        ok.append(('sbc_permuted', bool(release(sb, Zn)[1].shape == (3000, 5))))
    except ValueError:
        sbc_state = 'degenerate -> fallback'
        ok.append(('sbc_permuted', True))
    fb = fit(Z, yi, yo, fallback_target(Z), u, 0, 1.0, cat=cat)
    ok.append(('sbc_fallback', bool(release(fb, Zn)[1].shape == (3000, 5) and abs(np.corrcoef(fallback_target(Z), s)[0, 1]) < 0.1)))
    print(f'SBC on the permuted column: {sbc_state}')
    for name, good in ok:
        print(f'{name:20s} {"PASS" if good else "FAIL"}')
    if not all(g for _, g in ok):
        raise SystemExit('selftest FAILED')
    print('selftest PASS (invented data only; no real data, no labels)')


if __name__ == '__main__':
    selftest()
