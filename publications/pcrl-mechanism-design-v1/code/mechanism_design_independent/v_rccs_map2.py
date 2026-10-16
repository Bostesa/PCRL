"""RCCS vs MIN with the SAME literal map family and budget in the synthetic law of v_rccs_map (falsifier-1 analogue).

RCCS minimises the pair partial criterion R2(S ~ phi1, phi2', q') - R2(S ~ phi1); MIN minimises the occupation-only
criterion R2(S ~ phi2', upper(p2' p2'^T)). Both use z2' = z2 + A psi (10 free parameters), oracle propensity, budget
B on the exact occupation Brier excess. The proposal predicts RCCS pair <= MIN pair - 0.015 with RCCS occupation-only
AUC above MIN's. Readers: linear logistic, HGB, MLP; pair view and occupation-only view.
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from mechanism_design_independent.v_rccs_map import draw, build_r, release, unpack, wires, r2, readers

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_RCCS_MAP2_V1.json'


def fit(dc, coef, B, target):
    s = dc['s'].astype(float)

    def parts(theta):
        A = unpack(theta)
        p2s, p2 = release(dc, A, coef)
        phi1, phi2, q = wires(dc['p1'], p2s)
        if target == 'RCCS':
            Q = r2(s, np.concatenate([phi1, phi2, q], 1)) - r2(s, phi1)
        else:
            iu = np.triu_indices(3)
            Q = r2(s, np.concatenate([phi2, (p2s[:, :, None] * p2s[:, None, :])[:, iu[0], iu[1]]], 1))
        return Q, float(np.mean(np.sum((p2s - p2) ** 2, 1)))

    def obj(theta):
        Q, cost = parts(theta)
        return Q + 1e4 * max(0.0, cost - B) ** 2 + 1e-6 * np.sum(theta ** 2)
    best = None
    rng = np.random.default_rng(1)
    for start in range(2):
        x0 = np.zeros(10) if start == 0 else rng.normal(scale=0.5, size=10)
        res = minimize(obj, x0, method='Powell', options={'maxfev': 2500, 'xtol': 1e-3, 'ftol': 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    Q, cost = parts(best.x)
    return unpack(best.x), {'Q_final': Q, 'brier_cost': cost}


def views(d, p2, view, kind):
    phi1, phi2, _ = wires(d['p1'], p2)
    if view == 'occ':
        return np.concatenate([p2, phi2[:, 3:]], 1)
    if kind == 'linear_logistic':
        return np.concatenate([phi1, phi2], 1)
    return np.concatenate([d['p1'][:, None], phi1[:, 1:2], p2, phi2[:, 3:]], 1)


def main():
    t0 = time.time()
    res = {}
    for g in (0.35, 0.6):
        rng = np.random.default_rng(int(g * 100) + 7)
        dc = draw(30000, rng, g)
        _, coef = build_r(dc)
        dtr = draw(100000, rng, g)
        dte = draw(100000, rng, g)
        B = 0.009
        arms = {}
        for target in ('RCCS', 'MIN'):
            A, info = fit(dc, coef, B, target)
            arms[target] = (A, info)
        out = {'B': B, 'fits': {k: v[1] for k, v in arms.items()}, 'readers': {}}
        rels_tr = {'BASE': release(dtr, np.zeros((3, 5)), coef)[1]}
        rels_te = {'BASE': release(dte, np.zeros((3, 5)), coef)[1]}
        for k, (A, _) in arms.items():
            rels_tr[k] = release(dtr, A, coef)[0]
            rels_te[k] = release(dte, A, coef)[0]
        for view in ('pair', 'occ'):
            for kind, ntr in (('linear_logistic', 8000), ('hgb', 8000), ('hgb', 100000), ('mlp', 8000)):
                sl = slice(0, ntr)
                sub = {kk: (vv[sl] if hasattr(vv, '__len__') else vv) for kk, vv in dtr.items()}
                row = {}
                for arm in ('BASE', 'RCCS', 'MIN'):
                    row[arm] = readers(views(sub, rels_tr[arm][sl], view, kind), dtr['s'][sl],
                                       views(dte, rels_te[arm], view, kind), dte['s'], kind)
                out['readers'][f'{view}|{kind}|n={ntr}'] = row
                print(f'g={g} {view} {kind} n={ntr}: ' + ' '.join(f'{a} {v:.4f}' for a, v in row.items()), flush=True)
        res[f'g={g}'] = out
        print(json.dumps(out['fits']), f'elapsed {time.time() - t0:.0f}s', flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
