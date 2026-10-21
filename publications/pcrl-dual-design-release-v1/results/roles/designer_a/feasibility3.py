"""Designer A, third label-free pass (same cached inputs as feasibility.py; no label file, nothing of MODEL_SELECT).

(a) Person-level SE of the selection clauses under truth T1 (twin correct), for STD at lam 0.5 / 0.75 / 1 and w = .45 / .49:
    accuracy difference per person d = 1[y = a] - 1[y = b] (a twin argmax, b STD argmax), Var = qa + qb - (qa - qb)^2;
    NLL difference per person log t_y - log r_y, Var = sum_y t_y (log t_y / r_y)^2 - KL^2. Household clustering is not
    available label-free (no unit array is cached), so these SEs are person-level (a lower bound on the clustered SE).
(b) Linear R^2 of logit pi_hat on the release's log-probabilities (OLS, all 20,000 rows): the R1 trigger (b) quantity.
(c) Three-decimal roundings of the FEASIBILITY*.json reader AUCs used in PROPOSAL.md.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feasibility as F                                     # noqa: E402


def se_clauses(t, r):
    n = len(t)
    a, b = t.argmax(1), r.argmax(1)
    qa, qb = t[np.arange(n), a], t[np.arange(n), b]
    acc_mean = float(np.mean(qa - qb))
    acc_var = np.where(a != b, qa + qb - (qa - qb) ** 2, 0.0)
    lr = np.log(t / r)
    kl = (t * lr).sum(1)
    nll_var = (t * lr ** 2).sum(1) - kl ** 2
    return {'acc_loss': acc_mean, 'acc_se': float(np.sqrt(acc_var.sum()) / n), 'nll': float(kl.mean()),
            'nll_se': float(np.sqrt(nll_var.sum()) / n)}


def r2(target, P):
    X = np.column_stack([np.ones(len(P)), np.log(P)])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    res = target - X @ beta
    return float(1 - res.var() / target.var())


if __name__ == '__main__':
    t0 = time.process_time()
    out = {}
    for v in (0, 1):
        d = F.load(v)
        pi, qO, q0, q1 = d['pi'], d['qO'], d['q0'], d['q1']
        lp = np.log(pi / (1 - pi))
        comp = F.norm((1 - pi)[:, None] * q0 + pi[:, None] * q1)
        o = {'R2_logit_pi': {'TWIN': r2(lp, qO), 'COMP': r2(lp, comp)}}
        for w in (0.45, 0.49):
            r1 = F.norm((1 - w) * q0 + w * q1)
            o['R2_logit_pi'][f'STD_1.0_w{w}'] = r2(lp, r1)
            for lam in (0.5, 0.75, 1.0):
                o[f'T1_clauses_STD_{lam}_w{w}'] = se_clauses(qO, F.norm((1 - lam) * qO + lam * r1))
        out[f'V{v}'] = o
    rnd = {}
    for name in ('FEASIBILITY.json', 'FEASIBILITY2.json'):
        res = json.loads((F.OUT / name).read_text())
        for run in res['runs']:
            for k, a in run['readers'].items():
                if isinstance(a, dict) and 'occ' in a:
                    rnd[f"{name}|V{run['version']}|{k}"] = {vw: {rk: round(x, 3) for rk, x in a[vw].items()} for vw in ('occ', 'pair')}
    out['rounded_auc'] = rnd
    out['cpu_s'] = round(time.process_time() - t0, 2)
    (F.OUT / 'FEASIBILITY3.json').write_text(json.dumps(out, indent=1))
    for v in ('V0', 'V1'):
        print(v, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in out[v]['R2_logit_pi'].items()}))
        for k, x in out[v].items():
            if k.startswith('T1_'):
                print(' ', k, {kk: round(vv, 4) for kk, vv in x.items()})
    for k, x in rnd.items():
        print(k, x)
    print('CPU_SECONDS', out['cpu_s'])
