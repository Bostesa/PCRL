"""Phase 2: independent re-simulation of the two continuous MMP prototype laws (data-free, synthetic).

Law definitions re-entered from mechanism_design/prototype/mmp.py::ContinuousLaw (read, not imported); own vectorised
sampler, own posterior, own LEACE-type eraser, own readers (sklearn LR / HGB / MLP), own population quadrature.

Population part (exact up to quadrature): grid over h (801 points on [-6, 6]) x all levels of (e, a[, z]), weighted
by the class-conditional laws. Gives Bayes AUCs (twin = AUC*(S;X); MMP = AUC of P(S|h,e,a) with the proxy, equal to
the twin without it), Brier / NLL / accuracy prices of MMP vs eta, within-SEX increases, their exact decomposition
E[(pibar^2 - pi^2)||g||^2 | S=0] and E[((1-pibar)^2 - (1-pi)^2)||g||^2 | S=1], corr(pi, ||g||^2), and the
constant-gap (verifier M5) prediction for comparison.

Fitted part: 3 replications; fit sample 8,001, test sample 50,000. Arms: twin, MMP, the coordinator's comparator
(matched-price partial eraser on each recipient's own centred log-probabilities, fraction t < 1), and a supplementary
non-invertible eraser (t = 1 on the occupation wire only; its price is reported). Readers on each view: LR on
[p, log p], HGB on [p]; MLP on the pair log wire. Invertibility of the t < 1 eraser is checked numerically.
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_PROTO_CONT_V1.json'
COORD = ROOT / 'results' / 'pcrl_mechanism_design_v1' / 'prototype' / 'PROTOTYPE_RESULTS.json'

PF = 0.47
HM, HS = (0.35, -0.40), (0.85, 1.0)
EP = (np.array([.10, .30, .25, .25, .10]), np.array([.07, .26, .25, .29, .13]))
AG = np.arange(25, 65)
_w = np.linspace(1.1, 0.9, 40)
AP = (np.full(40, 1 / 40), _w / _w.sum())
ZP = (np.array([.30, .25, .20, .15, .10]), np.array([.10, .15, .20, .25, .30]))


def loglik(h, e, a, z, s, proxy, use_z=True):
    lp = -0.5 * ((h - HM[s]) / HS[s]) ** 2 - np.log(HS[s]) + np.log(EP[s][e]) + np.log(AP[s][a])
    if proxy and use_z:
        lp = lp + np.log(ZP[s][z])
    return lp


def post(h, e, a, z, proxy, use_z=True):
    lo = np.log(PF / (1 - PF)) + loglik(h, e, a, z, 1, proxy, use_z) - loglik(h, e, a, z, 0, proxy, use_z)
    return 1 / (1 + np.exp(-lo))


def q(h, e, a, s):
    age = (AG[a] - 45) / 10
    p1 = 1 / (1 + np.exp(-(-0.6 + 0.9 * h + 0.55 * (e - 2) + 0.25 * age - 0.45 * s)))
    Z = np.stack([0.70 * (e - 2) + 0.10 * age, -0.45 * h + 0.80 * s - 0.10 * (e - 2), 0.30 * h - 1.10 * s - 0.45 * (e - 2)], -1)
    Z = Z - Z.max(-1, keepdims=True)
    E = np.exp(Z)
    return np.stack([1 - p1, p1], -1), E / E.sum(-1, keepdims=True)


def sample(n, rng, proxy):
    s = (rng.random(n) < PF).astype(int)
    h = np.where(s == 1, rng.normal(HM[1], HS[1], n), rng.normal(HM[0], HS[0], n))
    e = np.where(s == 1, rng.choice(5, n, p=EP[1]), rng.choice(5, n, p=EP[0]))
    a = np.where(s == 1, rng.choice(40, n, p=AP[1]), rng.choice(40, n, p=AP[0]))
    z = np.where(s == 1, rng.choice(5, n, p=ZP[1]), rng.choice(5, n, p=ZP[0])) if proxy else np.zeros(n, int)
    return s, h, e, a, z


def releases(h, e, a, z, proxy):
    pi = post(h, e, a, z, proxy)
    i0, o0 = q(h, e, a, 0)
    i1, o1 = q(h, e, a, 1)
    eta = ((1 - pi)[..., None] * i0 + pi[..., None] * i1, (1 - pi)[..., None] * o0 + pi[..., None] * o1)
    mmp = ((1 - PF) * i0 + PF * i1, (1 - PF) * o0 + PF * o1)
    return pi, eta, mmp, ((i0, i1), (o0, o1))


# ---------------------------------------------------------------- population quadrature
def weighted_auc(score, w1, w0):
    order = np.argsort(score, kind='mergesort')
    sc, a1, a0 = score[order], w1[order] / w1.sum(), w0[order] / w0.sum()
    # group exact ties
    brk = np.r_[True, sc[1:] != sc[:-1]]
    gid = np.cumsum(brk) - 1
    g1 = np.bincount(gid, a1)
    g0 = np.bincount(gid, a0)
    cum0 = np.r_[0.0, np.cumsum(g0)[:-1]]
    return float(np.sum(g1 * (cum0 + 0.5 * g0)))


def population(proxy):
    hg = np.linspace(-6, 6, 801)
    nz = 5 if proxy else 1
    H, Eg, Ag, Zg = np.meshgrid(hg, np.arange(5), np.arange(40), np.arange(nz), indexing='ij')
    H, Eg, Ag, Zg = H.ravel(), Eg.ravel(), Ag.ravel(), Zg.ravel()
    w = {}
    for s in (0, 1):
        lw = loglik(H, Eg, Ag, Zg, s, proxy)
        ws = np.exp(lw - lw.max())
        w[s] = ws / ws.sum()
    pi, eta, mmp, qq = releases(H, Eg, Ag, Zg, proxy)
    px = PF * w[1] + (1 - PF) * w[0]
    res = {'AUC_star_X_twin_bayes': weighted_auc(pi, w[1], w[0])}
    if proxy:
        res['mmp_bayes'] = weighted_auc(np.round(post(H, Eg, Ag, Zg, proxy, use_z=False), 13), w[1], w[0])
    else:
        res['mmp_bayes'] = res['AUC_star_X_twin_bayes']
    res['pibar_check'] = float(np.sum(px * pi))
    v = float(np.sum(px * (pi - PF) ** 2))
    k3 = float(np.sum(px * (pi - PF) ** 3))
    res['var_pi'] = v
    res['kappa3_pi'] = k3
    for t, name in ((0, 'income'), (1, 'occupation')):
        e_, m_ = eta[t], mmp[t]
        q0, q1 = qq[t]
        g2 = np.sum((q1 - q0) ** 2, -1)
        r = {'brier_price': float(np.sum(px * np.sum((m_ - e_) ** 2, -1))),
             'nll_price': float(np.sum(px * np.sum(e_ * (np.log(e_) - np.log(m_)), -1))),
             'accuracy_loss': float(np.sum(px * (e_.max(-1) - np.take_along_axis(e_, m_.argmax(-1)[:, None], -1)[:, 0])))}
        inc = {}
        for s, qs in ((0, q0), (1, q1)):
            d_ = np.sum((m_ - qs) ** 2, -1) - np.sum((e_ - qs) ** 2, -1)
            inc['female' if s else 'male'] = float(np.sum(w[s] * d_))
        r['within_sex_increase'] = inc
        r['decomposition'] = {'male': float(np.sum(w[0] * (PF ** 2 - pi ** 2) * g2)),
                              'female': float(np.sum(w[1] * ((1 - PF) ** 2 - (1 - pi) ** 2) * g2))}
        mg = float(np.sum(px * g2))
        r['E_g2'] = mg
        r['corr_pi_g2'] = float(np.sum(px * (pi - PF) * (g2 - mg)) / np.sqrt(v * np.sum(px * (g2 - mg) ** 2)))
        r['constant_gap_M5_prediction'] = {'male': mg * (v * (3 * PF - 1) + k3) / (1 - PF), 'female': mg * (v * (2 - 3 * PF) - k3) / PF}
        res[name] = r
    return res


# ---------------------------------------------------------------- eraser and readers
def clr(p):
    L = np.log(np.maximum(p, 1e-300))
    return L - L.mean(1, keepdims=True)


def softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


class Eraser:
    """LEACE-type eraser on centred log-probabilities: X' = X - t (X - mu) P^T, P = S^{1/2} a a^T S^{-1/2}, a = unit(S^{-1/2} sigma)."""

    def __init__(self, p, s):
        X = clr(p)
        self.mu = X.mean(0)
        Xc = X - self.mu
        S = Xc.T @ Xc / len(X)
        sig = Xc.T @ (s - s.mean()) / len(X)
        lam, V = np.linalg.eigh(S)
        k = lam > 1e-10 * lam.max()
        V, lam = V[:, k], lam[k]
        Wh = V @ np.diag(lam ** -0.5) @ V.T
        Wi = V @ np.diag(lam ** 0.5) @ V.T
        a = Wh @ sig
        a = a / np.linalg.norm(a)
        self.P = Wi @ np.outer(a, a) @ Wh

    def apply(self, p, t):
        X = clr(p)
        return softmax(X - t * (X - self.mu) @ self.P.T)

    def invert_clr(self, p_erased, t):
        """Inverse of the t < 1 map on the CLR coordinates: X - mu = (X' - mu)(I + t/(1-t) P^T)."""
        X2 = clr(p_erased)
        return self.mu + (X2 - self.mu) @ (np.eye(len(self.mu)) + t / (1 - t) * self.P.T)


def matched_t(er, p, target):
    lo, hi = 0.0, 1.0
    f = lambda t: float(np.mean(np.sum((er.apply(p, t) - p) ** 2, 1)))
    if f(1.0) <= target:
        return 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def feats(rel, view, kind):
    p1, p2 = rel
    blocks = {'income': [p1], 'occupation': [p2], 'pair': [p1, p2]}[view]
    if kind == 'LR':
        return np.concatenate([np.concatenate([b, np.log(np.maximum(b, 1e-300))], 1) for b in blocks], 1)
    if kind == 'HGB':
        return np.concatenate(blocks, 1)
    return np.concatenate([np.log(np.maximum(b, 1e-300)) for b in blocks], 1)


def reader(kind, seed):
    if kind == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    if kind == 'HGB':
        return HistGradientBoostingClassifier(random_state=seed)
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=400, random_state=seed))


def utility(rel, eta, qq, s):
    out = {}
    for t, name in ((0, 'income'), (1, 'occupation')):
        r, e_ = rel[t], eta[t]
        q0, q1 = qq[t]
        inc = {}
        for g, qs, val in (('male', q0, 0), ('female', q1, 1)):
            d_ = np.sum((r - qs) ** 2, 1) - np.sum((e_ - qs) ** 2, 1)
            inc[g] = float(d_[s == val].mean())
        out[name] = {'brier_excess': float(np.mean(np.sum((r - e_) ** 2, 1))),
                     'nll_excess': float(np.mean(np.sum(e_ * (np.log(e_) - np.log(np.maximum(r, 1e-12))), 1))),
                     'accuracy_loss': float(np.mean(e_.max(1) - e_[np.arange(len(e_)), r.argmax(1)])),
                     'within_sex_increase': inc}
    return out


def replication(proxy, r):
    rng = np.random.default_rng(91000 + 17 * r + (1 if proxy else 2))
    data = {}
    for role, n in (('fit', 8001), ('test', 50000)):
        s, h, e, a, z = sample(n, rng, proxy)
        pi, eta, mmp, qq = releases(h, e, a, z, proxy)
        data[role] = dict(s=s, pi=pi, eta=eta, twin_eta=eta, mmp=mmp, qq=qq, x=(h, e, a, z))
    fit, test = data['fit'], data['test']
    ers, ts = [], []
    for t in range(2):
        er = Eraser(fit['eta'][t], fit['s'])
        price = float(np.mean(np.sum((fit['mmp'][t] - fit['eta'][t]) ** 2, 1)))
        ers.append(er)
        ts.append(matched_t(er, fit['eta'][t], price))
    for role in data:
        D = data[role]
        D['erase_matched'] = tuple(ers[t].apply(D['eta'][t], ts[t]) for t in range(2))
        D['erase_full_occ'] = (D['eta'][0], ers[1].apply(D['eta'][1], 1.0))
    # invertibility check of the t < 1 map on the test sample
    inv_err = max(float(np.max(np.abs(ers[t].invert_clr(test['erase_matched'][t], ts[t]) - clr(test['eta'][t])))) for t in range(2))
    out = {'matched_t': ts, 'invertibility_max_abs_clr_error': inv_err, 'arms': {}}
    h, e, a, z = test['x']
    bayes = {'twin_eta': roc_auc_score(test['s'], test['pi']),
             'mmp': roc_auc_score(test['s'], post(h, e, a, z, proxy, use_z=False)) if proxy else roc_auc_score(test['s'], test['pi'])}
    for arm in ('twin_eta', 'mmp', 'erase_matched', 'erase_full_occ'):
        ra = {}
        for view in ('income', 'occupation', 'pair'):
            for kind in ('LR', 'HGB') + (('MLP',) if view == 'pair' else ()):
                m = reader(kind, r)
                m.fit(feats(fit[arm], view, kind), fit['s'])
                ra[f'{view}|{kind}'] = float(roc_auc_score(test['s'], m.predict_proba(feats(test[arm], view, kind))[:, 1]))
        ra['bayes_mc'] = bayes.get(arm, bayes['twin_eta'])
        ra['utility_vs_eta'] = utility(test[arm], test['eta'], test['qq'], test['s']) if arm != 'twin_eta' else None
        out['arms'][arm] = ra
    return out


def summarise(reps):
    arms = reps[0]['arms'].keys()
    out = {'matched_t_mean': np.mean([rp['matched_t'] for rp in reps], 0).tolist(),
           'invertibility_max_abs_clr_error': max(rp['invertibility_max_abs_clr_error'] for rp in reps)}
    for arm in arms:
        keys = [k for k in reps[0]['arms'][arm] if k != 'utility_vs_eta']
        out[arm] = {k: float(np.mean([rp['arms'][arm][k] for rp in reps])) for k in keys}
        if reps[0]['arms'][arm]['utility_vs_eta'] is not None:
            u = {}
            for task in ('income', 'occupation'):
                u[task] = {m: float(np.mean([rp['arms'][arm]['utility_vs_eta'][task][m] for rp in reps])) for m in ('brier_excess', 'nll_excess', 'accuracy_loss')}
                u[task]['within_sex_increase'] = {g: float(np.mean([rp['arms'][arm]['utility_vs_eta'][task]['within_sex_increase'][g] for rp in reps])) for g in ('male', 'female')}
            out[arm]['utility_vs_eta'] = u
    for kind in ('LR', 'HGB', 'MLP'):
        k = f'pair|{kind}'
        out[f'twin_minus_mmp_{k}'] = out['twin_eta'][k] - out['mmp'][k]
        out[f'per_rep_twin_minus_mmp_{k}'] = [rp['arms']['twin_eta'][k] - rp['arms']['mmp'][k] for rp in reps]
        out[f'erase_matched_minus_twin_{k}'] = out['erase_matched'][k] - out['twin_eta'][k]
    for arm in ('erase_matched', 'erase_full_occ'):
        out[f'masking_{arm}'] = {f'{v}|{kd}': out['twin_eta'][f'{v}|{kd}'] - out[arm][f'{v}|{kd}'] for v in ('income', 'occupation', 'pair') for kd in ('LR', 'HGB')}
    return out


def main():
    t0 = time.time()
    res = {}
    for name, proxy in (('L_CONT_PROXY', True), ('L_CONT_NOPROXY', False)):
        pop = population(proxy)
        print(name, 'population', json.dumps(pop), f'{time.time() - t0:.0f}s', flush=True)
        reps = [replication(proxy, r) for r in range(3)]
        summ = summarise(reps)
        print(name, 'summary', json.dumps(summ), f'{time.time() - t0:.0f}s', flush=True)
        res[name] = {'population': pop, 'replications': reps, 'summary': summ}
    if COORD.exists():
        c = json.loads(COORD.read_text())['continuous']
        cmp_ = {}
        for name in res:
            cs = c[name]['summary']
            cmp_[name] = {'coord_twin_pair_slate_best': cs['twin_eta']['pair'], 'coord_mmp_pair_slate_best': cs['mmp']['pair'],
                          'coord_twin_minus_mmp': cs['pair_auc_difference_twin_minus_mmp'],
                          'coord_bayes_twin_mc': cs['twin_eta']['bayes_pair_auc_mc'], 'coord_bayes_mmp_mc': cs['mmp']['bayes_pair_auc_mc'],
                          'coord_local_linear_fractions_mean': np.mean(cs['local_linear_fractions'], 0).tolist(),
                          'coord_within_sex_mmp_occupation': cs['utility_vs_eta_mc']['mmp|task2']['within_sex_brier_increase']}
        res['coordinator_values_for_comparison'] = cmp_
    OUT.write_text(json.dumps(res, indent=1))
    print('WROTE', OUT, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
