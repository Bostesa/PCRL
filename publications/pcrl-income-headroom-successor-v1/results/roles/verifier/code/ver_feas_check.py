"""Independent verifier, check 9: planning-feasibility numbers (CONFIRMATION_FEASIBILITY.md / _RESULTS.json) recomputed with
different methods. Public-use ACS 2019 Florida records; no label, no population and no sealed record is read here. Inputs: the
published DEV_REPORT endpoint points and SEs (report/DEV_REPORT_RESULTS.json) and the coordinator's private per-replicate
endpoint draws (runs/REPORT/ENDPOINT_DRAWS.npz; derived endpoints only). ihs.feasibility is NOT imported.

Methods (different from the stored MVN eigendecomposition Monte Carlo):
1. analytic normal formula per condition: P(pass at n) = Phi(z), with SE(n) = SE_DEV x sqrt(3955 / n):
   utility / guard point gate z = (cap - theta) / SE(n); upper bound z = (cap - theta) / SE(n) - c1;
   recipient clause z = (0.010 - theta) / SE(n) - c1; pair benefit z = (theta - r) / SE(n) - c1;
   per-condition n80 in closed form; Frechet bound joint <= min marginal;
2. joint power by non-parametric rescaling of the 2,048 household-bootstrap replicates: est_b(n) = theta + (draw_b - mean) x
   sqrt(3955 / n) (no normality, no eigendecomposition);
3. joint power by my own Gaussian Monte Carlo (Cholesky of the replicate covariance + 1e-18 jitter, 40,000 draws, seed 7).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
import numpy as np
from scipy.stats import norm

RES = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'
DRAWS = ROOT / '.private' / 'pcrl_income_headroom_successor_v1' / 'runs' / 'REPORT' / 'ENDPOINT_DRAWS.npz'
OUT = ROOT / '.private' / 'pcrl_income_headroom_successor_v1' / 'verifier'
G_DEV = 3955
C1 = 2.3949990680014097


def conditions(ep):
    keys = sorted(k for k in ep if (k.startswith('UTILITY|S|') or k.startswith('SUBGROUP|S|')
                                    or k.startswith('NW|NEURAL_RAW|') or k.startswith('NW|H_LJ|') or k.startswith('BEN|NEURAL_RAW|') or k.startswith('BEN|H_LJ|')))
    out = []
    for k in keys:
        e = ep[k]
        kind = 'util' if k.startswith(('UTILITY', 'SUBGROUP')) else 'nw' if k.startswith('NW') else 'ben'
        out.append({'key': k, 'kind': kind, 'theta': float(e['point']), 'se': float(e['se']), 'cap': float(e['margin'])})
    return out


def pass_matrix(est, conds, se_n, D):
    """est: (B, J) estimates at n; returns (B,) joint pass."""
    ok = np.ones(est.shape[0], dtype=bool)
    for j, c in enumerate(conds):
        x = est[:, j]
        if c['kind'] == 'util':
            ok &= (x <= c['cap']) if D == 'D1' else (x + C1 * se_n[j] <= c['cap'])
        elif c['kind'] == 'nw':
            ok &= (x + C1 * se_n[j] <= 0.010)
        else:
            ok &= (x - C1 * se_n[j] > 0.0)
            if D == 'D3':
                ok &= (x - C1 * se_n[j] > 0.020)
    return ok


def marg_z(c, theta, se_n, D):
    if c['kind'] == 'util':
        return (c['cap'] - theta) / se_n - (0.0 if D == 'D1' else C1)
    if c['kind'] == 'nw':
        return (0.010 - theta) / se_n - C1
    return (theta - (0.020 if D == 'D3' else 0.0)) / se_n - C1


def main():
    rep = json.loads((RES / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    stored = json.loads((RES / 'CONFIRMATION_FEASIBILITY_RESULTS.json').read_text())
    grid = stored['grid_households']
    z = np.load(DRAWS)
    out = {'scorecards': {}}
    rng_seed = 7
    for sc in ('ENHANCED', 'ORIGINAL'):
        ep = rep['scorecards'][sc]['endpoints']
        conds = conditions(ep)
        degenerate = [c['key'] for c in conds if c['se'] == 0.0]
        conds = [c for c in conds if c['se'] > 0.0]
        D = np.column_stack([z[f"{sc}|{c['key']}"] for c in conds])
        se_pub = np.array([c['se'] for c in conds])
        se_check = {'max_rel_diff_sd_ddof1': float(np.max(np.abs(D.std(0, ddof=1) / se_pub - 1))),
                    'max_rel_diff_sd_ddof0': float(np.max(np.abs(D.std(0, ddof=0) / se_pub - 1)))}
        dev = D - D.mean(0)
        theta0 = np.array([c['theta'] for c in conds])
        util = [j for j, c in enumerate(conds) if c['kind'] == 'util']
        ratio = sorted(((conds[j]['cap'] - theta0[j]) / se_pub[j], conds[j]['key']) for j in util)
        bind_keys = list(stored['scorecards'][sc]['binding'])
        bind = [j for j, c in enumerate(conds) if c['key'] in bind_keys]
        ben = [j for j, c in enumerate(conds) if c['kind'] == 'ben']
        cov = np.cov(D, rowvar=False)
        L = np.linalg.cholesky(cov + 1e-18 * np.eye(len(conds)))
        gauss = np.random.default_rng(rng_seed).standard_normal((40000, len(conds))) @ L.T

        def scen_theta(name):
            th = theta0.copy()
            if name.startswith('S2_benefit_x'):
                th[ben] *= float(name.split('x')[1])
            elif name == 'S3_binding_halfway_to_cap':
                for j in bind:
                    th[j] = th[j] + (conds[j]['cap'] - th[j]) / 2
            elif name == 'S3_binding_at_cap':
                for j in bind:
                    th[j] = conds[j]['cap']
            elif name.startswith('S5_binding_true_slack_'):
                s = float(name.split('_')[-1])
                for j in bind:
                    th[j] = conds[j]['cap'] - s
            elif name.startswith('S7'):
                th[ben] *= 0.5
                for j in util:
                    th[j] = conds[j]['cap'] - (conds[j]['cap'] - th[j]) / 2
            return th

        scen = ['S1_plugin', 'S2_benefit_x0.75', 'S2_benefit_x0.5', 'S3_binding_halfway_to_cap', 'S3_binding_at_cap',
                'S5_binding_true_slack_0.002', 'S5_binding_true_slack_0.001', 'S5_binding_true_slack_0.0005', 'S5_binding_true_slack_0.0',
                'S5_binding_true_slack_-0.001', 'S7_transport_HYPOTHETICAL_benefit_x0.5_utility_margin_x0.5']
        srec = {}
        for name in scen:
            th = scen_theta(name)
            for Dn in ('D1', 'D2', 'D3'):
                def power(n, noise):
                    f = np.sqrt(G_DEV / n)
                    return float(pass_matrix(th + noise * f, conds, se_pub * f, Dn).mean())
                curve_boot = [power(n, dev) for n in grid]
                curve_gauss = [power(n, gauss) for n in grid]
                marg = {conds[j]['key']: [float(norm.cdf(marg_z(conds[j], th[j], se_pub[j] * np.sqrt(G_DEV / n), Dn))) for n in grid] for j in range(len(conds))}
                min_marg = [min(marg[k][i] for k in marg) for i in range(len(grid))]
                prod_marg = [float(np.prod([marg[k][i] for k in marg])) for i in range(len(grid))]

                def n80(curve):
                    for n, p in zip(grid, curve):
                        if p >= 0.8:
                            return n
                    return None
                st = stored['scorecards'][sc]['scenarios'][f'{name}|{Dn}']
                i175 = grid.index(17510)
                srec[f'{name}|{Dn}'] = {
                    'stored_n80': st.get('n_households_80'), 'boot_n80': n80(curve_boot), 'gauss_n80': n80(curve_gauss),
                    'min_marginal_n80_lower_bound': n80(min_marg), 'independence_product_n80': n80(prod_marg),
                    'stored_p17510': st['power_at'].get('17510'), 'boot_p17510': curve_boot[i175], 'gauss_p17510': curve_gauss[i175],
                    'min_marginal_p17510_upper_bound': min_marg[i175],
                    'stored_p3955': st['power_at'].get('3955'), 'boot_p3955': curve_boot[0], 'gauss_p3955': curve_gauss[0],
                    'stored_limit': st.get('limit_n_to_infinity_mc'), 'boot_limit': power(1e14, dev), 'gauss_limit': power(1e14, gauss),
                    'frechet_ok_17510': bool(st['power_at'].get('17510', 0) <= min_marg[i175] + 0.01),
                }
        # analytic closed-form per-condition n80 for the binding clauses and the weakest recipient clause (plug-in)
        analytic = {}
        for j, c in enumerate(conds):
            if c['kind'] == 'util':
                m = c['cap'] - c['theta']
                zz = {'D1_point': norm.ppf(0.8), 'D2_bound': norm.ppf(0.8) + C1}
                analytic[c['key']] = {'margin_se_units': m / c['se'], 'at_17510': m / c['se'] * np.sqrt(17510 / G_DEV),
                                      **{f'n80_{k}': (G_DEV * (v * c['se'] / m) ** 2 if m > 0 else None) for k, v in zz.items()},
                                      'p17510_D1_point': float(norm.cdf(m / (c['se'] * np.sqrt(G_DEV / 17510)))),
                                      'p17510_D2_bound': float(norm.cdf(m / (c['se'] * np.sqrt(G_DEV / 17510)) - C1))}
            else:
                m = (0.010 - c['theta']) if c['kind'] == 'nw' else c['theta']
                analytic[c['key']] = {'margin': m, 'n80': (G_DEV * ((norm.ppf(0.8) + C1) * c['se'] / m) ** 2 if m > 0 else None),
                                      'p17510': float(norm.cdf(m / (c['se'] * np.sqrt(G_DEV / 17510)) - C1)),
                                      'p_at_stored_D1_n80': float(norm.cdf(m / (c['se'] * np.sqrt(G_DEV / stored['scorecards'][sc]['scenarios']['S1_plugin|D1']['n_households_80'])) - C1)) if stored['scorecards'][sc]['scenarios']['S1_plugin|D1']['n_households_80'] else None}
        bind_check = {k: {'stored': stored['scorecards'][sc]['binding'][k],
                          'recomputed_margin_se_units': analytic[k]['margin_se_units'], 'recomputed_at_17510': analytic[k]['at_17510']} for k in bind_keys}
        out['scorecards'][sc] = {'n_conditions_nondegenerate': len(conds), 'degenerate': degenerate, 'se_check': se_check,
                                 'five_smallest_util_margins': ratio[:6], 'binding_check': bind_check, 'scenarios': srec,
                                 'analytic': analytic,
                                 'income_brier_nll_margin_se_units': {c['key']: (c['cap'] - c['theta']) / c['se'] for c in conds
                                                                      if c['key'].startswith('UTILITY|S|income') and ('brier' in c['key'] or 'nll' in c['key'])}}
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OUT / 'FEASIBILITY_CHECK.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    for sc, r in out['scorecards'].items():
        print('==', sc, 'n_cond', r['n_conditions_nondegenerate'], 'degenerate', r['degenerate'], 'se_check', r['se_check'])
        print(' smallest margins', [(round(a, 4), b) for a, b in r['five_smallest_util_margins']])
        for k, v in r['binding_check'].items():
            print(' bind', k, round(v['stored']['margin_se_units'], 6), round(v['recomputed_margin_se_units'], 6), round(v['stored']['margin_se_units_at_17510'], 6), round(v['recomputed_at_17510'], 6))
        for k, v in r['scenarios'].items():
            print(' ', k, 'n80 stored/boot/gauss/minmarg', v['stored_n80'], v['boot_n80'], v['gauss_n80'], v['min_marginal_n80_lower_bound'],
                  '| p17510', v['stored_p17510'], round(v['boot_p17510'], 4), round(v['gauss_p17510'], 4), 'ub', round(v['min_marginal_p17510_upper_bound'], 4),
                  '| p3955', v['stored_p3955'], round(v['boot_p3955'], 4), round(v['gauss_p3955'], 4), '| lim', v['stored_limit'], v['boot_limit'], v['gauss_limit'])
        print(' income brier/nll margins (SE units):', {k.replace('UTILITY|S|income|', ''): round(v, 3) for k, v in r['income_brier_nll_margin_se_units'].items()})
        for k in ('NW|H_LJ|income|s1', 'NW|NEURAL_RAW|income|s1', 'BEN|NEURAL_RAW|s1', 'BEN|H_LJ|s1'):
            if k in r['analytic']:
                print(' analytic', k, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in r['analytic'][k].items()})
        for k in r['binding_check']:
            print(' analytic', k, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in r['analytic'][k].items()})


if __name__ == '__main__':
    main()
