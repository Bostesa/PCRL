"""D2 cross-critique check: designer 1's MMP inside designer 2's calibrated linear-Gaussian coalition model (no real data).

MMP subtracts the inferred-SEX term (pi - pibar) * gamma from each release (gamma = within-SEX task gap, female minus
male, projected on the coordinates (f1, g, h)). In the Gaussian model this is the shift F' = F - gamma (pi - pibar):
  c' = c - gamma Var(pi);   Sigma' = Sigma - gamma c' ... (exact second moments below);   Brier price = ||gamma||^2 Var(pi)
(designer 1's exact identity; no recalibration). We report MMP's pair / single AUCs and prices over a grid of gaps, the
RCCS pair at the same occupation price (income frozen), and MMP-on-income followed by RCCS on occupation.
"""
from mechanism_design import no_real_data
no_real_data.install()

import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('g', Path(__file__).resolve().parent / 'proto_gaussian_redundancy.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def mmp_shift(M, gamma):
    """Exact moments after F' = F - gamma (pi - pibar). Returns a model dict usable by G.frontier (residual driver)."""
    Sig, c, varpi = M['Sig'], M['c'], M['varpi']
    gamma = np.asarray(gamma, float)
    cn = c - gamma * varpi
    Sn = Sig - np.outer(gamma, c) - np.outer(c, gamma) + np.outer(gamma, gamma) * varpi
    Wn = float(cn @ np.linalg.solve(Sn, cn))
    return {'Sig': Sn, 'c': cn, 'V': varpi - Wn, 'varpi': varpi, 'W_pair': Wn}


def infos(M):
    Sig, c = M['Sig'], M['c']
    Wp = float(c @ np.linalg.solve(Sig, c))
    Wi = float(c[0] ** 2 / Sig[0, 0])
    Wo = float(c[1:] @ np.linalg.solve(Sig[1:, 1:], c[1:]))
    f = lambda W: G.auc_from_d2(G.W_to_d2(W))
    return {'pair': f(Wp), 'inc': f(Wi), 'occ': f(Wo), 'amplification_(pair-occ)/inc': (Wp - Wo) / Wi if Wi > 0 else None}


def rccs_at(M, cost):
    fr = G.frontier(M, [cost])
    return fr[0]['ALIGN']


def main():
    d2 = {k: G.d2_from_auc(v) for k, v in {'inc': 0.569, 'occ': 0.650}.items()}
    rows = []
    for d2X in (0.40, 0.49, 0.64):
        M = G.build(0.82, 0.6, d2X, d2['inc'], d2['occ'])
        base = infos(M)
        lim_h = M['c'][2] / M['varpi']
        for g1 in (-0.05, -0.10, -0.20):
            for gg in (-0.05, 0.0, 0.05):
                for gh in (0.15, 0.30, 0.45):
                    if gh >= lim_h:          # would remove more than all of h's S-covariance (over-correction)
                        continue
                    gam = np.array([g1, gg, gh])
                    Mm = mmp_shift(M, gam)
                    occ_price = float((gg ** 2 + gh ** 2) * M['varpi'])
                    inc_price_summed = float(2 * g1 ** 2 * M['varpi'])
                    r = rccs_at(M, occ_price)                               # RCCS, income frozen at T_TREE
                    Mi = mmp_shift(M, np.array([g1, 0.0, 0.0]))            # MMP on income only, then RCCS on occupation
                    rc = rccs_at(Mi, occ_price)
                    rows.append({'d2X': d2X, 'gamma': [g1, gg, gh], 'TTREE': base, 'MMP': infos(Mm),
                                 'occ_price_brier': occ_price, 'inc_price_summed_brier': inc_price_summed,
                                 'RCCS_same_occ_price_pair': r['pair_auc'], 'RCCS_same_occ_price_occ': r['occ_auc'],
                                 'MMPinc_then_RCCS_pair': rc['pair_auc']})
    better = sum(1 for r in rows if r['MMP']['pair'] < r['RCCS_same_occ_price_pair'])
    worse_than_base = sum(1 for r in rows if r['MMP']['pair'] > r['TTREE']['pair'])
    print(json.dumps({'n_configs': len(rows), 'MMP_pair_below_RCCS_at_same_occ_price': better,
                      'MMP_pair_above_T_TREE': worse_than_base, 'rows': rows}, indent=1, default=float))


if __name__ == '__main__':
    main()
