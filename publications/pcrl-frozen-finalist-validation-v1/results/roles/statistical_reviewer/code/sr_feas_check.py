"""Statistical reviewer's independent feasibility check (no labels; published endpoints + private replicate draws only).

Different method from ffv/feasibility.py: own condition enumeration, analytic single-clause sample sizes, nonparametric
rescaling of the 2048 replicate deviations, an MVN check via SVD of the centred draws (own seed), orthant limits from the
correlation matrix, and the assurance limit from posterior draws of the true means.
"""
import json
import sys

import numpy as np
from scipy.stats import norm

REPORT, DRAWS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
G = 3955
C1 = 2.3949991
Z80, Z90 = norm.ppf(0.8), norm.ppf(0.9)
BIND = ['UTILITY|income|Ucal|brier_excess|s0', 'UTILITY|income|Ucal|brier_excess|s1',
        'UTILITY|income|U0|accuracy_loss|s1', 'UTILITY|income|Ucal|accuracy_loss|s1']
rep = json.load(open(REPORT))
with np.load(DRAWS) as z:
    keys = [str(k) for k in z['keys']]
    D = z['draws'].copy()
out = {'method': __doc__, 'scorecards': {}}
grid = np.unique(np.round(np.geomspace(G, 1e8, 400)).astype(np.int64))


def own_conditions(eps):
    conds = []
    for k, e in eps.items():
        if e['family'] != 'P':
            continue
        if k.startswith('UTILITY|') or k.startswith('SUBGROUP|'):
            conds.append((k, 'U', e['margin']))
        elif k.startswith('COMPETITIVE|') and ('income_worsening' in k or 'occupation_worsening' in k):
            conds.append((k, 'R', 0.010))
        elif k.startswith('COMPETITIVE|pair_benefit_vs_NEURAL_RAW') or k.startswith('COMPETITIVE|pair_benefit_vs_H_LJ'):
            conds.append((k, 'B', 0.0))
    return conds


def joint_power(theta, se, E, kinds, caps, n, dfn, target=0.0):
    f = np.sqrt(G / n)
    est = theta[None, :] + f * E
    s = se * f
    ok = np.ones(E.shape[0], dtype=bool)
    for j, kd in enumerate(kinds):
        if kd == 'U':
            ok &= (est[:, j] <= caps[j]) if dfn == 'D1' else (est[:, j] + C1 * s[j] <= caps[j])
        elif kd == 'R':
            ok &= est[:, j] + C1 * s[j] <= caps[j]
        else:
            ok &= est[:, j] - C1 * s[j] > (0.020 if dfn == 'D3' else 0.0)
    return ok.mean()


def n_first(theta, se, E, kinds, caps, dfn, target):
    pw = np.array([joint_power(theta, se, E, kinds, caps, n, dfn) for n in grid])
    hit = np.flatnonzero(pw >= target)
    return (int(grid[hit[0]]) if len(hit) else None), pw


for sc in ('ENHANCED', 'ORIGINAL'):
    eps = rep['scorecards'][sc]['endpoints']
    idx = {k.split('::', 1)[1]: i for i, k in enumerate(keys) if k.startswith(sc + '::')}
    conds_all = own_conditions(eps)
    zero_var = [k for k, _, _ in conds_all if float(np.std(D[:, idx[k]], ddof=1)) == 0.0]
    conds = [c for c in conds_all if c[0] not in zero_var]
    ks = [k for k, _, _ in conds]
    kinds = [kd for _, kd, _ in conds]
    caps = np.array([m for _, _, m in conds])
    X = D[:, [idx[k] for k in ks]]
    theta = np.array([eps[k]['point'] for k in ks])
    se_pub = np.array([eps[k]['se'] for k in ks])
    se_draw = X.std(0, ddof=1)
    rec = {'n_conditions': len(conds), 'zero_variance_deterministic': zero_var,
           'max_abs_se_published_minus_draws': float(np.max(np.abs(se_pub - se_draw)))}
    # (a) analytic single-clause sample sizes
    single = {}
    for k in BIND + ['COMPETITIVE|income_worsening_vs_H_LJ|s1', 'COMPETITIVE|pair_benefit_vs_NEURAL_RAW|s1']:
        e = eps[k]; pt, se = e['point'], e['se']
        if k.startswith('UTILITY'):
            slack = e['margin'] - pt
            single[k] = {'slack': slack, 'se': se, 'slack_over_se': slack / se,
                         'D1_point_gate_n80': G * (Z80 * se / slack) ** 2, 'D1_point_gate_n90': G * (Z90 * se / slack) ** 2,
                         'D2_bound_n80': G * ((Z80 + C1) * se / slack) ** 2, 'D2_bound_n90': G * ((Z90 + C1) * se / slack) ** 2,
                         'D1_marginal_power_at_1e5': float(norm.cdf(slack / (se * np.sqrt(G / 1e5)))),
                         'D2_marginal_power_at_1e5': float(norm.cdf(slack / (se * np.sqrt(G / 1e5)) - C1)),
                         'posterior_P_true_slack_positive': float(norm.cdf(slack / se))}
        elif 'worsening' in k:
            slack = 0.010 - pt
            single[k] = {'point': pt, 'se': se, 'slack': slack, 'bound_n80': G * ((Z80 + C1) * se / slack) ** 2 if slack > 0 else None}
        else:
            d = pt - 0.020
            single[k] = {'point': pt, 'se': se, 'D3_margin': d, 'D3_bound_n80': G * ((Z80 + C1) * se / d) ** 2 if d > 0 else 'unreachable at plug-in'}
    rec['analytic_single_clause'] = single
    # (b1) nonparametric rescaling of the replicate deviations (no normality)
    E_np = X - X.mean(0, keepdims=True)
    for dfn in ('D1', 'D2'):
        n80, pw = n_first(theta, se_draw, E_np, kinds, caps, dfn, 0.8)
        n90, _ = n_first(theta, se_draw, E_np, kinds, caps, dfn, 0.9)
        rec[f'nonparametric_S1_{dfn}'] = {'n80': n80, 'n90': n90, 'power_at': {str(int(n)): float(np.interp(n, grid, pw)) for n in (G, 1e5, 1e6, 1e7, 1e8)},
                                          'mc_se_at_0.8': float(np.sqrt(0.16 / len(E_np)))}
    # (b2) MVN via SVD of the centred draws, own seed, 40,000 draws
    U, S, Vt = np.linalg.svd(E_np, full_matrices=False)
    A = (S[:, None] * Vt) / np.sqrt(len(E_np) - 1)          # Sigma = A.T @ A
    rng = np.random.default_rng(20261017)
    E_mvn = rng.standard_normal((40000, A.shape[0])) @ A
    for dfn in ('D1', 'D2'):
        n80, pw = n_first(theta, se_draw, E_mvn, kinds, caps, dfn, 0.8)
        n90, _ = n_first(theta, se_draw, E_mvn, kinds, caps, dfn, 0.9)
        rec[f'mvn_svd_S1_{dfn}'] = {'n80': n80, 'n90': n90, 'power_at': {str(int(n)): float(np.interp(n, grid, pw)) for n in (G, 1e5, 1e6, 1e7, 1e8)}}
    # (c) orthant limits for the binding clauses at their caps (correlation from the draws)
    bj = [ks.index(k) for k in BIND]
    Cb = np.corrcoef(X[:, bj], rowvar=False)
    rec['binding_correlation'] = np.round(Cb, 4).tolist()
    w, V = np.linalg.eigh(Cb); Lb = V * np.sqrt(np.clip(w, 0, None))[None, :]
    Zb = rng.standard_normal((1_000_000, 4)) @ Lb.T
    rec['orthant_limit_at_cap_D1'] = float(np.mean(np.all(Zb <= 0, axis=1)))
    rec['orthant_limit_at_cap_D2'] = float(np.mean(np.all(Zb <= -C1, axis=1)))
    rec['independence_product_D1'] = 0.5 ** 4
    # (d) assurance limit: posterior P that every condition's true value is strictly feasible
    Mu = theta[None, :] + rng.standard_normal((200000, A.shape[0])) @ A
    feas = np.ones(len(Mu), dtype=bool); feas3 = None
    for j, kd in enumerate(kinds):
        feas &= (Mu[:, j] < caps[j]) if kd in ('U', 'R') else (Mu[:, j] > 0.0)
    feas3 = feas.copy()
    for j, kd in enumerate(kinds):
        if kd == 'B':
            feas3 &= Mu[:, j] > 0.020
    rec['assurance_limit_D1_D2'] = float(feas.mean())
    rec['assurance_limit_D3'] = float(feas3.mean())
    marg = {ks[j]: float(np.mean((Mu[:, j] < caps[j]) if kinds[j] in ('U', 'R') else (Mu[:, j] > 0.0))) for j in range(len(ks))}
    rec['posterior_marginal_feasibility_lowest'] = dict(sorted(marg.items(), key=lambda kv: kv[1])[:8])
    out['scorecards'][sc] = rec
# (e) refit components: recompute tau and test the bootstrap-refit shift
tau = {}
shift = []
for s in ('S0', 'S1'):
    v = rep['per_seed'][s]['variation']
    for svc in ('FINALIST', 'H_LJ', 'NEURAL_RAW'):
        a = v[svc]['AUX']['dev_auc_by_refit']
        shift.append({'seed': s, 'release': svc, 'winner': v[svc]['AUX']['winner'], 'full_sample_fit': a[0], 'resample_refits': a[1:],
                      'n_resample_below_full': int(sum(x < a[0] for x in a[1:])), 'mean_shift': float(np.mean(a[1:]) - a[0]),
                      'sd_with_full_fit': float(np.std(a, ddof=1)), 'sd_resamples_only': float(np.std(a[1:], ddof=1))})
    bm = v['benefit_matched_refits']
    for ref in ('NEURAL_RAW', 'H_LJ'):
        comps = [bm[f'{x}|{ref}']['sd'] for x in ('VAR', 'AUX') if bm.get(f'{x}|{ref}', {}).get('sd') is not None]
        aux = bm[f'AUX|{ref}']['benefit_by_refit']
        tau[f'{s}|{ref}'] = {'tau_r_quadrature': float(np.sqrt(np.sum(np.square(comps)))),
                             'aux_benefit_sd_resamples_only': float(np.std(aux[1:], ddof=1)), 'aux_benefit_mean_shift': float(np.mean(aux[1:]) - aux[0])}
out['refit_shift'] = shift
out['refit_tau'] = tau
out['n_resample_refits_below_full_fit'] = [sum(x['n_resample_below_full'] for x in shift), 4 * len(shift)]
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1, default=float)
for sc, r in out['scorecards'].items():
    print(sc, 'conds', r['n_conditions'], 'zero-var', len(r['zero_variance_deterministic']), 'se diff', r['max_abs_se_published_minus_draws'])
    for k, v in r['analytic_single_clause'].items():
        print('  ', k, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for m in ('nonparametric_S1_D1', 'nonparametric_S1_D2', 'mvn_svd_S1_D1', 'mvn_svd_S1_D2'):
        print('  ', m, r[m]['n80'], r[m]['n90'], {k: round(x, 4) for k, x in r[m]['power_at'].items()})
    print('   orthant D1/D2 at cap', r['orthant_limit_at_cap_D1'], r['orthant_limit_at_cap_D2'], 'product', r['independence_product_D1'])
    print('   binding corr', r['binding_correlation'])
    print('   assurance limit D1/D2', r['assurance_limit_D1_D2'], 'D3', r['assurance_limit_D3'])
    print('   lowest posterior feasibility', {k: round(x, 3) for k, x in r['posterior_marginal_feasibility_lowest'].items()})
print('resample refits below full fit:', out['n_resample_refits_below_full_fit'])
for x in shift:
    print('  ', x['seed'], x['release'], x['winner'], 'below', x['n_resample_below_full'], 'shift', round(x['mean_shift'], 4), 'sd_with', round(x['sd_with_full_fit'], 4), 'sd_resamples', round(x['sd_resamples_only'], 4))
for k, v in tau.items():
    print('  tau', k, {kk: round(vv, 5) for kk, vv in v.items()})
