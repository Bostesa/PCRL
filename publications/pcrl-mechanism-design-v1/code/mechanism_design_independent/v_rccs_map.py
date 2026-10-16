"""Synthetic test of the LITERAL RCCS map against reader classes inside and outside its criterion (data-free).

Law (continuous, synthetic): w1, w2, c ~ N(0,1) iid; c is SEX-only. logit P(S=1|x) = 0.55 w1 - 0.45 w2 + g c - 0.12.
Income P(Y1=1|x) = sigmoid(-0.3 + 1.0 w1 + 0.6 w2); occupation (3 classes) logits (0, 1.2 w1 - 0.2 w2, -0.8 w1 + 1.0 w2).
The base pair (f1, f2) is a function of w only, so its Bayes recovery is AUC*(S;W) < AUC*(S;X).

RCCS (as specified, with an ORACLE propensity pi_hat = pi, i.e. RCCS's best case): r = logit pi - OLS(logit pi ~ base
wires); psi = (r, r p1, r p2_1..3); z2' = z2 + A psi with column-centred A (10 free parameters); A minimises
Q(A) = R2(S ~ phi1, phi2', q') - R2(S ~ phi1) (OLS on a criterion sample) subject to summed occupation Brier excess
E||f2' - f2||^2 <= B (f2 is the Bayes posterior, so this is the exact Brier excess).
Evaluation on fresh samples: linear and quadratic logistic readers (the criterion class), HGB and MLP readers
(outside it), trained on 8,000 (ATTACK_FIT size) and 100,000 rows; oracle AUC*(S;W) and AUC*(S;X).
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_RCCS_MAP_V1.json'


def sig(x):
    return 1 / (1 + np.exp(-x))


def draw(n, rng, g):
    w1, w2, c = rng.normal(size=(3, n))
    ls = 0.55 * w1 - 0.45 * w2 + g * c - 0.12
    pi = sig(ls)
    s = (rng.random(n) < pi).astype(int)
    p1 = sig(-0.3 + 1.0 * w1 + 0.6 * w2)
    z2 = np.stack([np.zeros(n), 1.2 * w1 - 0.2 * w2, -0.8 * w1 + 1.0 * w2], 1)
    # oracle P(S=1 | w) by Gauss-Hermite over c
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(40)
    pw = (sig((0.55 * w1 - 0.45 * w2 - 0.12)[:, None] + g * gh_x[None, :]) * gh_w[None, :]).sum(1) / gh_w.sum()
    return dict(w1=w1, w2=w2, c=c, pi=pi, s=s, p1=p1, z2=z2, pw=pw)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def wires(p1, p2):
    lp1 = np.log(p1 / (1 - p1))
    phi1 = np.stack([p1, lp1], 1)
    phi2 = np.concatenate([p2, np.log(np.clip(p2, 1e-12, 1))], 1)
    iu = np.triu_indices(3)
    q = np.concatenate([p1[:, None] * p2, (p2[:, :, None] * p2[:, None, :])[:, iu[0], iu[1]]], 1)
    return phi1, phi2, q


def r2(y, X):
    X1 = np.concatenate([np.ones((len(y), 1)), X], 1)
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return 1 - res.var() / y.var()


def build_r(d, coef=None):
    p2 = softmax(d['z2'])
    phi1, phi2, _ = wires(d['p1'], p2)
    X = np.concatenate([np.ones((len(d['p1']), 1)), phi1, phi2], 1)
    lpi = np.log(d['pi'] / (1 - d['pi']))
    if coef is None:
        coef, *_ = np.linalg.lstsq(X, lpi, rcond=None)
    return lpi - X @ coef, coef


def release(d, A, coef):
    r, _ = build_r(d, coef)
    p2 = softmax(d['z2'])
    psi = np.concatenate([r[:, None], r[:, None] * d['p1'][:, None], r[:, None] * p2], 1)   # (n, 5)
    z2s = d['z2'] + psi @ A.T                                                                  # A: (3, 5)
    return softmax(z2s), p2


def unpack(theta):
    A = np.zeros((3, 5))
    A[1:, :] = theta.reshape(2, 5)
    return A - A.mean(0, keepdims=True)


def fit_A(dc, coef, B):
    def parts(theta):
        A = unpack(theta)
        p2s, p2 = release(dc, A, coef)
        phi1, phi2, q = wires(dc['p1'], p2s)
        Q = r2(dc['s'].astype(float), np.concatenate([phi1, phi2, q], 1)) - r2(dc['s'].astype(float), phi1)
        cost = np.mean(np.sum((p2s - p2) ** 2, 1))
        return Q, cost

    def obj(theta):
        Q, cost = parts(theta)
        return Q + 1e4 * max(0.0, cost - B) ** 2 + 1e-6 * np.sum(theta ** 2)
    best = None
    rng = np.random.default_rng(0)
    for start in range(2):
        x0 = np.zeros(10) if start == 0 else rng.normal(scale=0.5, size=10)
        res = minimize(obj, x0, method='Powell', options={'maxfev': 2500, 'xtol': 1e-3, 'ftol': 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    Q, cost = parts(best.x)
    return unpack(best.x), {'Q_final': Q, 'brier_cost': cost, 'obj': best.fun}


def readers(Xtr, ytr, Xte, yte, kind):
    if kind == 'linear_logistic':
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif kind == 'quadratic_logistic':
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    elif kind == 'hgb':
        m = HistGradientBoostingClassifier(random_state=0)
    else:
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=300, random_state=0))
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def features(d, p2, kind):
    phi1, phi2, q = wires(d['p1'], p2)
    if kind == 'quadratic_logistic':
        return np.concatenate([phi1, phi2, q], 1)
    if kind in ('linear_logistic',):
        return np.concatenate([phi1, phi2], 1)
    return np.concatenate([d['p1'][:, None], phi1[:, 1:2], p2, phi2[:, 3:]], 1)   # probs + logs for HGB / MLP


def main():
    t0 = time.time()
    res = {}
    for g in (0.35, 0.6):
        rng = np.random.default_rng(int(g * 100))
        dc = draw(30000, rng, g)        # criterion sample (MODEL_FIT analogue, large: RCCS's best case)
        _, coef = build_r(dc)
        dtr_big = draw(100000, rng, g)
        dte = draw(100000, rng, g)
        out = {'AUC_star_W(base pair Bayes)': float(roc_auc_score(dte['s'], dte['pw'])),
               'AUC_star_X': float(roc_auc_score(dte['s'], dte['pi']))}
        r_te, _ = build_r(dte, coef)
        lpi = np.log(dte['pi'] / (1 - dte['pi']))
        out['V_over_var_logit_pi'] = float(r_te.var() / lpi.var())
        for B in (0.005, 0.009):
            A, info = fit_A(dc, coef, B)
            p2s_te, p2_te = release(dte, A, coef)
            p2s_tr, p2_tr = release(dtr_big, A, coef)
            cost_te = float(np.mean(np.sum((p2s_te - p2_te) ** 2, 1)))
            nll_te = float(np.mean(np.sum(p2_te * (np.log(p2_te) - np.log(p2s_te)), 1)))
            acc_te = float(np.mean(p2_te[np.arange(len(p2_te)), p2_te.argmax(1)] - p2_te[np.arange(len(p2_te)), p2s_te.argmax(1)]))
            row = {'fit': info, 'test_brier_excess': cost_te, 'test_nll_excess': nll_te, 'test_accuracy_loss': acc_te,
                   'readers': {}}
            for kind in ('linear_logistic', 'quadratic_logistic', 'hgb', 'mlp'):
                for ntr in (8000, 100000):
                    if kind == 'mlp' and ntr == 100000:
                        continue
                    sl = slice(0, ntr)
                    sub = {k: (v[sl] if hasattr(v, '__len__') else v) for k, v in dtr_big.items()}
                    base = readers(features(sub, p2_tr[sl], kind), dtr_big['s'][sl], features(dte, p2_te, kind), dte['s'], kind)
                    rcc = readers(features(sub, p2s_tr[sl], kind), dtr_big['s'][sl], features(dte, p2s_te, kind), dte['s'], kind)
                    row['readers'][f'{kind}|n={ntr}'] = {'base_pair_auc': base, 'rccs_pair_auc': rcc, 'rccs_minus_base': rcc - base}
                    print(f'g={g} B={B} {kind} n={ntr}: base {base:.4f} rccs {rcc:.4f} diff {rcc - base:+.4f}', flush=True)
            out[f'B={B}'] = row
        res[f'g={g}'] = out
        print(json.dumps({k: v for k, v in out.items() if not k.startswith('B=')}), f'elapsed {time.time() - t0:.0f}s', flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
