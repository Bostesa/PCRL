"""Post-hoc probe (registered as P7 in PROTOTYPE_VERIFICATION.md before running): MMP with no pure-proxy FIELD but
more fields than release dimensions (synthetic, data-free).

x | S ~ N(S d, I_K), K = 10, P(S=1) = 0.47, so pi(x) = sigmoid(c + d'x) and AUC*(S;X) = Phi(||d|| / sqrt 2).
Within-SEX tasks: income logit b'x + c1 - 0.45 S; occupation logits (0, B1'x + 0.8 S, B2'x - 1.1 S). Every field has
nonzero coefficients in b, B1, B2 (no pure-proxy field). MMP m = (1 - pibar) q^0 + pibar q^1 depends on x only through
U'x (U an orthonormal basis of T = span(b, B1, B2)), so AUC*(S; m) = Phi(||U'd|| / sqrt 2) exactly (it is injective in
U'x). The twin eta also carries d'x through pi. Readers: LR on [p, log p], HGB on [p], fitted on 8,001 and on 100,000
rows; test on 100,000. Three random draws of (d, b, B).
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_PROTO_HD_V1.json'
PF, K = 0.47, 10
ND = NormalDist()


def sig(z):
    return 1 / (1 + np.exp(-z))


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def make(draw):
    rng = np.random.default_rng(5150 + draw)
    d = rng.normal(size=K)
    d *= 0.74 / np.linalg.norm(d)                       # AUC*(S;X) = Phi(0.74/sqrt2) ~ 0.70
    b = rng.choice([-1, 1], K) * rng.uniform(0.15, 0.45, K)
    B = rng.choice([-1, 1], (2, K)) * rng.uniform(0.15, 0.45, (2, K))
    return d, b, B


def tasks(x, s, b, B):
    p1 = sig(-0.3 + x @ b - 0.45 * s)
    Z = np.stack([np.zeros(len(x)), x @ B[0] + 0.8 * s, x @ B[1] - 1.1 * s], 1)
    return np.stack([1 - p1, p1], 1), softmax(Z)


def sample(n, rng, d, b, B):
    s = (rng.random(n) < PF).astype(int)
    x = rng.normal(size=(n, K)) + s[:, None] * d[None, :]
    c = math.log(PF / (1 - PF)) - 0.5 * d @ d
    pi = sig(c + x @ d)
    i0, o0 = tasks(x, 0, b, B)
    i1, o1 = tasks(x, 1, b, B)
    eta = ((1 - pi)[:, None] * i0 + pi[:, None] * i1, (1 - pi)[:, None] * o0 + pi[:, None] * o1)
    mmp = ((1 - PF) * i0 + PF * i1, (1 - PF) * o0 + PF * o1)
    return s, pi, eta, mmp


def feats(rel, kind):
    if kind == 'LR':
        return np.concatenate([np.concatenate([p, np.log(p)], 1) for p in rel], 1)
    return np.concatenate(rel, 1)


def main():
    out = {}
    for draw in range(3):
        d, b, B = make(draw)
        U, _ = np.linalg.qr(np.stack([b, B[0], B[1]], 1))
        rng = np.random.default_rng(777 + draw)
        tr = sample(100000, rng, d, b, B)
        te = sample(100000, rng, d, b, B)
        rec = {'AUC_star_X_exact': ND.cdf(np.linalg.norm(d) / math.sqrt(2)),
               'mmp_bayes_exact': ND.cdf(np.linalg.norm(U.T @ d) / math.sqrt(2)),
               'share_of_d_in_task_span': float(np.linalg.norm(U.T @ d) ** 2 / (d @ d)),
               'AUC_star_X_mc': float(roc_auc_score(te[0], te[1]))}
        for t, name in ((0, 'income'), (1, 'occupation')):
            e_, m_ = te[2][t], te[3][t]
            rec[f'{name}_brier_price'] = float(np.mean(np.sum((m_ - e_) ** 2, 1)))
            rec[f'{name}_nll_price'] = float(np.mean(np.sum(e_ * (np.log(e_) - np.log(m_)), 1)))
            rec[f'{name}_accuracy_loss'] = float(np.mean(e_.max(1) - e_[np.arange(len(e_)), m_.argmax(1)]))
        for arm, idx in (('twin', 2), ('mmp', 3)):
            for kind in ('LR', 'HGB'):
                for n in (8001, 100000):
                    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if kind == 'LR' else HistGradientBoostingClassifier(random_state=draw)
                    m.fit(feats(tr[idx], kind)[:n], tr[0][:n])
                    rec[f'{arm}|{kind}|n={n}'] = float(roc_auc_score(te[0], m.predict_proba(feats(te[idx], kind))[:, 1]))
        for kind in ('LR', 'HGB'):
            for n in (8001, 100000):
                rec[f'twin_minus_mmp|{kind}|n={n}'] = rec[f'twin|{kind}|n={n}'] - rec[f'mmp|{kind}|n={n}']
        out[f'draw{draw}'] = rec
        print(f'draw {draw}:', json.dumps({k: round(v, 4) for k, v in rec.items()}), flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
