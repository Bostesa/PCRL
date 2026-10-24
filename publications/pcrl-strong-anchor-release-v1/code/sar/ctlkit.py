"""Shared matched-control constructions of the PCRL strong-anchor release campaign (`pcrl_strong_anchor_release_v1`). Derived
from `ddr.ctlkit` (the dual-design release campaign's adopted constructions, PROTOCOL_AMENDMENT_1 there), with the release-only path
RO added (PROTOCOL section 4). Label-free code: labels arrive as arguments from the family entry points.

1. PSEUDO2, the same-form SEX-independent attribute of the SBC paths (unchanged from ddr.ctlkit):
       A_i = 1[u_i < sigmoid(gamma (beta z(x_i) + g(y_i)) + c)]
   - g(y): the logit of the MODEL_FIT SEX share of each (income, occupation) label cell, PERMUTED across the ten cells;
   - z(x): a keyed random linear score of the standardized compact fields, residualized by OLS on [1, logit pi_S(x)] (the cached
     out-of-fold SEX propensity) and standardized; beta = SD(logit pi_S), so the x-part has SEX's spread;
   - c: matches the SEX share; u_i: a keyed uniform of the row (SHA-256); gamma in (1, 2, 4) by the family's spend-reaching rule;
   - rule (a): among N_SEEDS seeded (permutation, direction) candidates, the one minimizing max over gamma of |corr(A, SEX)| on
     MODEL_FIT; accepted if that maximum <= 0.03 (else flagged SBC_ATTRIBUTE_NOT_INDEPENDENT, still used and reported);
   - profile (b), recorded beside SEX's own: cross-fitted AUC of a propensity model for the attribute, and the R^2 of the logit of
     that propensity on the twin's pair log-probabilities (OLS).
2. DQ, the continuous SEX-blind spend path (SPEND): subtractive-dithered lattice quantization of each recipient's twin log-ratio
   vector in its standardized principal-component coordinates (income: the 1-d log-odds; occupation: the 4 principal components of
   the centred log-probabilities, fitted on the twin's MODEL_FIT out-of-fold outputs). For step delta and a keyed dither d(x)
   uniform on [-delta/2, delta/2) per coordinate (SHA-256 of the six compact fields): s' = delta round((s + d) / delta) - d.
   Information removing, not invertible, continuous outputs. SEX is never read. DQ_GRID_FINE is this campaign's step grid (ratio
   1.1 between neighbours, so that bracket points lie close to a candidate's spend on each side, PROTOCOL 4).
3. RO, the generic release-only path (the dual-design ZTILT construction, here a control for any operation that is release-only):
   q_z(r) = P(SEX = 1 | the twin's released log-probability features r) by HGB (SEX-side settings, 300 iterations), cross-fitted on
   the household folds NRM_FA_xf_S<v> for its out-of-fold values and fitted on all MODEL_FIT people for deployment;
   covariance-cancelling weights (income: one class-weight vector; occupation: one vector per twin decision stratum, with the global
   vector for strata under 200 people or with Cov(q_z, SEX | k) <= 1e-4) fitted on the out-of-fold statistics with the MODEL_FIT SEX
   label; release log r = log p - alpha (q_z - qbar) w + const for both recipients. Out-of-fold path values use nested weights
   (refitted on the other outer household folds, tag SAR_RO_ncv_S<v>). Every SEX-targeted ingredient is a function of the twin's
   released vectors. RO_GRID: alpha on a geometric grid with ratio 1.1 (spend grows about as alpha^2).
"""
from __future__ import annotations
import hashlib

import numpy as np

N_SEEDS = 20
GAMMAS = (1.0, 2.0, 4.0)
CORR_MAX = 0.03
FLOOR = 1e-6
DQ_GRID_FINE = tuple(float(round(0.1 * 1.1 ** j, 6)) for j in range(0, 40))       # 0.1 .. about 4.1
RO_GRID = tuple(float(round(0.05 * 1.1 ** j, 6)) for j in range(0, 42))            # 0.05 .. about 2.7
MIN_STRATUM = 200
COV_MIN = 1e-4
N_OUTER = 5


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def keyed_uniform(keys, tag):
    """Uniform in [0, 1) per key (SHA-256 of tag | key)."""
    return np.array([int(hashlib.sha256(f'{tag}|{k}'.encode()).hexdigest()[:13], 16) / float(16 ** 13) for k in keys])


def row_keys(Z):
    Z = np.ascontiguousarray(np.asarray(Z, np.float64))
    return [r.tobytes().hex() for r in Z]


# ---------------------------------------------------------------- PSEUDO2
def _standardize(Z):
    Z = np.asarray(Z, np.float64)
    sd = Z.std(0)
    sd[sd == 0] = 1.0
    return (Z - Z.mean(0)) / sd


def pseudo2_parts(Z, y_inc, y_occ, sex, pi_oof, seed, version):
    """x-part z (residualized on logit pi, standardized) and label part g for one seeded candidate."""
    rng = np.random.default_rng(np.random.SeedSequence([int(version), 20261021, 77, int(seed)]))
    Zs = _standardize(Z)
    z = Zs @ rng.normal(size=Zs.shape[1])
    lp = _logit(pi_oof)
    Xr = np.column_stack([np.ones(len(lp)), lp])
    coef, *_ = np.linalg.lstsq(Xr, z, rcond=None)
    z = z - Xr @ coef
    z = (z - z.mean()) / (z.std() + 1e-12)
    cell = np.asarray(y_inc) * 5 + np.asarray(y_occ)
    share = np.array([float(sex[cell == c].mean()) if np.any(cell == c) else float(sex.mean()) for c in range(10)])
    g_cells = _logit(np.clip(share, 0.02, 0.98))[rng.permutation(10)]
    return z, g_cells[cell], float(np.std(lp))


def pseudo2_attribute(z, g, beta, gamma, share, u):
    lin = gamma * (beta * z + g)
    lo, hi = -30.0, 30.0
    for _ in range(80):
        c = 0.5 * (lo + hi)
        m = float(np.mean(1.0 / (1.0 + np.exp(-(lin + c)))))
        lo, hi = (c, hi) if m < share else (lo, c)
    p = 1.0 / (1.0 + np.exp(-(lin + 0.5 * (lo + hi))))
    return (u < p).astype(np.int64)


def pseudo2_select(Z, y_inc, y_occ, sex, pi_oof, row_id, version, n_seeds=N_SEEDS, gammas=GAMMAS):
    """Rule (a): the seeded candidate minimizing max over gamma of |corr(A, SEX)|. Returns (make(gamma) -> A, record)."""
    sex = np.asarray(sex)
    share = float(sex.mean())
    u = keyed_uniform([int(r) for r in row_id], f'SAR_PSEUDO2_S{int(version)}')
    rows = []
    for j in range(n_seeds):
        z, g, beta = pseudo2_parts(Z, y_inc, y_occ, sex, pi_oof, j, version)
        cors = {str(gm): float(np.corrcoef(pseudo2_attribute(z, g, beta, gm, share, u), sex)[0, 1]) for gm in gammas}
        rows.append({'seed': j, 'corr_with_sex': cors, 'max_abs_corr': max(abs(x) for x in cors.values())})
    best = min(rows, key=lambda r: (r['max_abs_corr'], r['seed']))
    z, g, beta = pseudo2_parts(Z, y_inc, y_occ, sex, pi_oof, best['seed'], version)
    rec = {'chosen_seed': best['seed'], 'max_abs_corr': best['max_abs_corr'], 'corr_with_sex': best['corr_with_sex'],
           'accepted': best['max_abs_corr'] <= CORR_MAX, 'beta_sd_logit_pi': beta, 'candidates': rows,
           'flag': None if best['max_abs_corr'] <= CORR_MAX else 'SBC_ATTRIBUTE_NOT_INDEPENDENT'}
    return (lambda gm: pseudo2_attribute(z, g, beta, gm, share, u)), rec


def attribute_profile(Z, a, unit, twin_pI, twin_pO, version, cat):
    """Profile (b): cross-fitted AUC of a propensity for the binary a (HGB, 200 iterations) and the R^2 of its logit on the twin's
    pair log-probabilities (OLS)."""
    import warnings
    from sklearn.ensemble import HistGradientBoostingClassifier
    from . import famkit as FK
    f = FK.household_folds(unit, 5, f'SAR_profile_S{int(version)}')
    q = np.zeros(len(a))
    for j in range(5):
        tr, te = f != j, f == j
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=16, min_samples_leaf=100,
                                               categorical_features=cat, random_state=int(version)).fit(Z[tr], a[tr])
        q[te] = m.predict_proba(Z[te])[:, 1]
    return {'propensity_auc': FK.auc(q, a), 'r2_logit_on_twin_pair': r2_on_pair(q, twin_pI, twin_pO)}, np.clip(q, 0.01, 0.99)


def r2_on_pair(q, pI, pO):
    lq = _logit(q)
    F = np.column_stack([np.ones(len(lq)), np.log(np.maximum(pI[:, 1:], FLOOR) / np.maximum(pI[:, :1], FLOOR)),
                         np.log(np.maximum(pO[:, 1:], FLOOR) / np.maximum(pO[:, :1], FLOOR))])
    coef, *_ = np.linalg.lstsq(F, lq, rcond=None)
    return float(1 - np.var(lq - F @ coef) / np.var(lq))


# ---------------------------------------------------------------- DQ: dithered quantization of the twin (SPEND)
def dq_fit(tI, tO):
    """Standardized principal-component basis of the twin's MODEL_FIT out-of-fold outputs (SEX never read)."""
    a = np.log(np.maximum(tI[:, 1], FLOOR) / np.maximum(tI[:, 0], FLOOR))
    lo = np.log(np.maximum(tO, FLOOR))
    c = lo - lo.mean(1, keepdims=True)
    mu = c.mean(0)
    U, S, Vt = np.linalg.svd(c - mu, full_matrices=False)
    V = Vt[:4].T                                            # 5 x 4 (the 5th direction is the constant, removed by centring)
    s = (c - mu) @ V
    return {'a_mu': float(a.mean()), 'a_sd': float(a.std()), 'mu_O': mu, 'V_O': V, 'sd_O': s.std(0)}


def _dither(Z, delta, tag, k):
    keys = row_keys(Z)
    return np.column_stack([(keyed_uniform(keys, f'{tag}|{j}') - 0.5) * delta for j in range(k)])


def dq_release(basis, pI, pO, Z, delta, tag):
    """Subtractive-dithered quantization with step delta (standardized units) of each recipient's coordinates."""
    if delta <= 0:
        return np.asarray(pI, np.float64), np.asarray(pO, np.float64)
    a = np.log(np.maximum(pI[:, 1], FLOOR) / np.maximum(pI[:, 0], FLOOR))
    sa = (a - basis['a_mu']) / basis['a_sd']
    d = _dither(Z, delta, f'{tag}|I', 1)[:, 0]
    sa2 = delta * np.round((sa + d) / delta) - d
    a2 = basis['a_mu'] + basis['a_sd'] * sa2
    p1 = 1.0 / (1.0 + np.exp(-a2))
    rI = np.column_stack([1 - p1, p1])
    lo = np.log(np.maximum(pO, FLOOR))
    c = lo - lo.mean(1, keepdims=True)
    s = (c - basis['mu_O']) @ basis['V_O']
    ss = s / basis['sd_O']
    dO = _dither(Z, delta, f'{tag}|O', ss.shape[1])
    ss2 = delta * np.round((ss + dO) / delta) - dO
    c2 = basis['mu_O'] + (ss2 * basis['sd_O']) @ basis['V_O'].T
    e = np.exp(c2 - c2.max(1, keepdims=True))
    rO = e / e.sum(1, keepdims=True)
    from . import famkit as FK
    return FK.clipn(rI), FK.clipn(rO)


# ---------------------------------------------------------------- RO: the generic release-only path (the dual-design ZTILT law)
def clog(p):
    lp = np.log(np.maximum(np.asarray(p, np.float64), FLOOR))
    return lp - lp.mean(1, keepdims=True)


def softmax(l):
    e = np.exp(l - l.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _w(c, q, s):
    sc = s - s.mean()
    cqs = float(np.mean((q - q.mean()) * sc))
    return (c * sc[:, None]).mean(0) / cqs if cqs > COV_MIN else np.zeros(c.shape[1]), float(q.mean()), cqs


def ro_weights(pI, pO, q, s):
    """Closed-form covariance-cancelling weights on out-of-fold statistics (pI, pO, q) with the binary target s
    (ddr.db.ptc.fit_weights with stratification)."""
    s = np.asarray(s, np.float64)
    w_inc, qbar, cqs = _w(clog(pI), q, s)
    w_glob, _, _ = _w(clog(pO), q, s)
    dec = np.asarray(pO).argmax(1)
    K = np.asarray(pO).shape[1]
    W, qb, fb = np.tile(w_glob, (K, 1)), np.full(K, qbar), []
    for k in range(K):
        m = dec == k
        if m.sum() >= MIN_STRATUM:
            wk, qk, ck = _w(clog(pO[m]), q[m], s[m])
            if ck > COV_MIN:
                W[k], qb[k] = wk, qk
                continue
        fb.append(k)
    return {'w_inc': w_inc, 'qbar': qbar, 'W_occ': W, 'qbar_occ': qb, 'cov_q_s': cqs, 'fallback_strata': fb}


def ro_tilt(pI, pO, q, st, alpha):
    from . import famkit as FK
    pI, pO, q = np.asarray(pI, np.float64), np.asarray(pO, np.float64), np.asarray(q, np.float64)
    k = pO.argmax(1)
    rI = softmax(np.log(np.maximum(pI, FLOOR)) - alpha * (q - st['qbar'])[:, None] * st['w_inc'][None, :])
    rO = softmax(np.log(np.maximum(pO, FLOOR)) - alpha * (q - st['qbar_occ'][k])[:, None] * st['W_occ'][k])
    return FK.clipn(rI), FK.clipn(rO)


def ro_features(pI, pO):
    """The twin's released log-probability features (income log-odds, centred occupation log-probabilities)."""
    return np.column_stack([np.log(np.maximum(pI[:, 1:], FLOOR) / np.maximum(pI[:, :1], FLOOR)), clog(pO)])


def ro_teacher(pI, pO, sex, unit, version):
    """q_z = P(SEX | the twin's released log-probability features): cross-fitted out-of-fold values and the deployable full fit."""
    import warnings
    from sklearn.ensemble import HistGradientBoostingClassifier
    from nrm.fa import prc as P
    F = ro_features(pI, pO)
    mk = lambda: HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(P.HGB['lr']), max_iter=300,
                                                max_leaf_nodes=int(P.HGB['leaves']), min_samples_leaf=int(P.HGB['msl']),
                                                l2_regularization=float(P.HGB['l2']), early_stopping=False, random_state=int(version))
    xf = P.household_folds(unit, P.N_XF, f'NRM_FA_xf_S{version}')
    oof = np.zeros(len(sex))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for f in range(P.N_XF):
            tr, te = xf != f, xf == f
            oof[te] = mk().fit(F[tr], sex[tr]).predict_proba(F[te])[:, 1]
        model = mk().fit(F, sex)
    return model, np.clip(oof, 0.01, 0.99)


def ro_nested(pI, pO, q, s, unit, version, alphas):
    """{alpha: (rI, rO)} on MODEL_FIT: weights fitted on the other outer folds' out-of-fold statistics, applied to the held-out fold."""
    from . import famkit as FK
    of = FK.household_folds(unit, N_OUTER, f'SAR_RO_ncv_S{int(version)}')
    n = len(unit)
    kI, kO = np.asarray(pI).shape[1], np.asarray(pO).shape[1]
    out = {a: (np.zeros((n, kI)), np.zeros((n, kO))) for a in alphas}
    for j in range(N_OUTER):
        tr, te = of != j, of == j
        st = ro_weights(pI[tr], pO[tr], q[tr], s[tr])
        for a in alphas:
            rI, rO = ro_tilt(pI[te], pO[te], q[te], st, a)
            out[a][0][te], out[a][1][te] = rI, rO
    return out


def ro_release(state, pI, pO, alpha):
    """Deployment: the twin's vectors (pI, pO) -> the RO release at strength alpha (q_z from the full-fit model, full-data weights)."""
    q = np.clip(state['q_model'].predict_proba(ro_features(pI, pO))[:, 1], 0.01, 0.99)
    return ro_tilt(pI, pO, q, state['weights'], float(alpha))


def ro_path(pI, pO, sex, unit, version, alphas=RO_GRID):
    """The RO path on MODEL_FIT: (nested out-of-fold releases {alpha: (rI, rO)}, deployable state {'q_model', 'weights'}, record)."""
    model, qz = ro_teacher(pI, pO, sex, unit, version)
    rel = ro_nested(pI, pO, qz, sex, unit, version, alphas)
    st = ro_weights(pI, pO, qz, sex)
    rec = {'cov_qz_sex': st['cov_q_s'], 'fallback_strata': st['fallback_strata'], 'grid': list(alphas)}
    return rel, {'q_model': model, 'weights': st}, rec


def _clip(P):
    from . import famkit as FK
    return FK.clipn(P)


def selftest():
    rng = np.random.default_rng(3)
    n = 3000
    sex = rng.integers(0, 2, n)
    pi = np.clip(0.5 + 0.25 * (sex - 0.5) + rng.normal(0, 0.15, n), 0.02, 0.98)
    Z = np.column_stack([rng.integers(20, 60, n), rng.normal(40, 8, n), rng.integers(0, 8, n), rng.integers(10, 24, n),
                         rng.integers(0, 5, n), rng.integers(1, 60, n)]).astype(float)
    yi, yo = rng.integers(0, 2, n), np.minimum(rng.integers(0, 5, n) + sex, 4)
    make, rec = pseudo2_select(Z, yi, yo, sex, pi, np.arange(n), 0, n_seeds=8)
    A = make(2.0)
    ok = {'pseudo2_binary_share': abs(A.mean() - sex.mean()) < 0.05,
          'pseudo2_low_corr': abs(np.corrcoef(A, sex)[0, 1]) <= max(0.06, rec['max_abs_corr'] + 1e-9),
          'pseudo2_deterministic': np.array_equal(A, make(2.0))}
    tI = rng.dirichlet(np.ones(2), n)
    tO = rng.dirichlet(np.ones(5), n)
    b = dq_fit(tI, tO)
    rI, rO = dq_release(b, tI, tO, Z, 0.5, 'test')
    ok['dq_valid'] = np.allclose(rI.sum(1), 1) and np.allclose(rO.sum(1), 1)
    rI0, rO0 = dq_release(b, tI, tO, Z, 1e-9, 'test')
    ok['dq_small_delta_near_identity'] = float(np.abs(rO0 - tO).max()) < 1e-6
    ok['dq_deterministic'] = np.array_equal(dq_release(b, tI, tO, Z, 0.5, 'test')[1], rO)
    lo = np.log(tO); c = lo - lo.mean(1, keepdims=True); c2 = np.log(rO); c2 = c2 - c2.mean(1, keepdims=True)
    err = ((c2 - b['mu_O']) @ b['V_O'] - (c - b['mu_O']) @ b['V_O']) / b['sd_O']
    ok['dq_error_bounded'] = float(np.abs(err).max()) <= 0.5 + 1e-6
    ok['dq_many_distinct_values'] = np.unique(rO.round(12), axis=0).shape[0] > 0.9 * n
    ok['dq_fine_grid_ratio'] = all(abs(b / x - 1.1) < 1e-3 for x, b in zip(DQ_GRID_FINE, DQ_GRID_FINE[1:]))
    # RO: zero strength is the twin; a positive strength lowers the released covariance with the target
    unit = np.arange(n) // 2
    tO2 = softmax(np.log(tO) + 0.8 * (sex - 0.5)[:, None] * np.array([1.0, -0.5, 0.0, 0.3, -0.8]))
    tI2 = softmax(np.log(tI) + 0.6 * (sex - 0.5)[:, None] * np.array([-1.0, 1.0]))
    st = ro_weights(tI2, tO2, pi, sex)
    r0 = ro_tilt(tI2, tO2, pi, st, 0.0)
    ok['ro_zero_is_twin'] = np.allclose(r0[0], _clip(tI2)) and np.allclose(r0[1], _clip(tO2))
    cov = lambda P: abs(float(np.mean((clog(P)[:, 0] - clog(P)[:, 0].mean()) * (sex - sex.mean()))))
    r1 = ro_tilt(tI2, tO2, pi, st, 1.0)
    ok['ro_reduces_income_covariance'] = cov(r1[0]) < cov(tI2)
    nest = ro_nested(tI2, tO2, pi, sex, unit, 0, (0.0, 0.5))
    ok['ro_nested_shapes'] = nest[0.5][0].shape == (n, 2) and nest[0.5][1].shape == (n, 5)
    ok['ro_grid_ratio'] = all(abs(b / x - 1.1) < 1e-3 for x, b in zip(RO_GRID, RO_GRID[1:]))
    ok = {k: bool(v) for k, v in ok.items()}
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    import json
    r = selftest()
    print(json.dumps(r))
    raise SystemExit(0 if r['all_ok'] else 1)
