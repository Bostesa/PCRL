"""Shared matched-control constructions of the PCRL dual-design release campaign (`pcrl_dual_design_release_v1`; adopted after the
challenger's review, roles/challenger/REVIEW_DESIGNS.md findings 5, 6, 8 and 15; PROTOCOL_AMENDMENT_1.md). Label-free code: labels
arrive as arguments from the family entry points.

1. PSEUDO2, the same-form SEX-independent attribute of the SBC paths (both families):
       A_i = 1[u_i < sigmoid(gamma (beta z(x_i) + g(y_i)) + c)]
   - g(y): the logit of the MODEL_FIT SEX share of each (income, occupation) label cell, PERMUTED across the ten cells;
   - z(x): a keyed random linear score of the standardized compact fields, residualized by OLS on [1, logit pi_S(x)] (the cached
     out-of-fold SEX propensity) and standardized; beta = SD(logit pi_S), so the x-part has SEX's spread;
   - c: matches the SEX share; u_i: a keyed uniform of the row (SHA-256); gamma in (1, 2, 4) by the family's spend-reaching rule;
   - rule (a): among N_SEEDS seeded (permutation, direction) candidates, the one minimizing max over gamma of |corr(A, SEX)| on
     MODEL_FIT; accepted if that maximum <= 0.03 (else flagged SBC_ATTRIBUTE_NOT_INDEPENDENT, still used and reported);
   - profile (b), recorded beside SEX's own: cross-fitted AUC of a propensity model for the attribute, and the R^2 of the logit of
     that propensity on the twin's pair log-probabilities (OLS).
2. DQ, the continuous SEX-blind spend path (SPEND) of both families: subtractive-dithered lattice quantization of each recipient's
   twin log-ratio vector in its standardized principal-component coordinates (income: the 1-d log-odds; occupation: the 4
   principal components of the centred log-probabilities, fitted on the twin's MODEL_FIT out-of-fold outputs). For step delta and a
   keyed dither d(x) uniform on [-delta/2, delta/2) per coordinate (SHA-256 of the six compact fields): s' = delta round((s + d) /
   delta) - d. The released value keeps the decision-relevant structure up to an error uniform on [-delta/2, delta/2) per
   standardized coordinate, independent of the score (information removing, not invertible), with continuous outputs (no finite
   cells, so table readers gain nothing from repeated values). SEX is never read.
"""
from __future__ import annotations
import hashlib

import numpy as np

N_SEEDS = 20
GAMMAS = (1.0, 2.0, 4.0)
CORR_MAX = 0.03
FLOOR = 1e-6


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
    u = keyed_uniform([int(r) for r in row_id], f'DDR_PSEUDO2_S{int(version)}')
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
    f = FK.household_folds(unit, 5, f'DDR_profile_S{int(version)}')
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
    ok = {k: bool(v) for k, v in ok.items()}
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    import json
    r = selftest()
    print(json.dumps(r))
    raise SystemExit(0 if r['all_ok'] else 1)
