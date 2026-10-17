"""Unit VER_POWER_V1 (check 8: power and sample size). Label-free: uses only the verifier's own saved per-replicate endpoint draws
and points (aggregates from the one logged replay) and the coordinator's published feasibility results (V1 and V2).

Public-use ACS 2019 Florida person records; the endpoints are the frozen scorecards' utility, subgroup-guard, recipient and pair-benefit
clauses. Independent methods (no code from ffv/feasibility*.py):
 A. analytic single-clause sample size n = 3955 x ((z + c) x SE / slack)^2 and analytic boundary limits (orthant probabilities);
 B. nonparametric bootstrap-deviation rescaling: est_b(n) = theta + sqrt(3955 / n) (D_b - mean D), b = 1..2048, D the verifier's own
    household-bootstrap endpoint replicates; joint power = share of b with every required condition holding;
 C. own multivariate-normal Monte Carlo with a different seed (61017) and no factorization: X = Z D_c / sqrt(B - 1), so cov(X) is
    exactly the replicate covariance.
Decisions follow PROTOCOL section 10: D1 = utility/guard point gates, recipient clause U1 <= 0.010, benefit clause L1 > 0; D2 = D1
with every utility/guard condition at U1 <= cap; D3 = D2 with benefit L1 > 0.020; SE(n) = SE_DEV sqrt(3955 / n); c1 = 2.3949991.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import multivariate_normal, norm

import vcommon as V

G_DEV = 3955
C1 = 2.3949991
BIND = ('UTILITY|income|Ucal|brier_excess|s0', 'UTILITY|income|Ucal|brier_excess|s1', 'UTILITY|income|U0|accuracy_loss|s1',
        'UTILITY|income|Ucal|accuracy_loss|s1')
SEED_MC = 61017
N_MC = 20000


def conds_of(eps):
    out = []
    for k, e in eps.items():
        if e['family'] != 'P' or e['se'] == 0.0:
            continue
        if k.startswith(('UTILITY|', 'SUBGROUP|')):
            out.append((k, 'u', e['margin']))
        elif k.startswith('COMPETITIVE|') and '_worsening_' in k:
            out.append((k, 'r', 0.010))
        elif k.startswith('COMPETITIVE|pair_benefit_vs_'):
            out.append((k, 'b', 0.0))
    return out


def holds(est, se_n, kinds, caps, dfn):
    """est: (m, J) estimates; returns (m,) joint indicator and (J,) marginal pass shares."""
    ok = np.empty(est.shape, dtype=bool)
    u, r, b = kinds == 'u', kinds == 'r', kinds == 'b'
    ok[:, u] = (est[:, u] <= caps[u]) if dfn == 'D1' else (est[:, u] + C1 * se_n[u] <= caps[u])
    ok[:, r] = est[:, r] + C1 * se_n[r] <= caps[r]
    ok[:, b] = est[:, b] - C1 * se_n[b] > (0.020 if dfn == 'D3' else 0.0)
    return ok.all(1), ok.mean(0)


def curve_boot(theta, se, Dc, kinds, caps, dfn, grid, tau=None, xi=None):
    pw = []
    for n in grid:
        f = np.sqrt(G_DEV / n)
        est = theta[None, :] + f * Dc
        if tau is not None:
            est = est + tau[None, :] * xi
        pw.append(float(holds(est, se * f, kinds, caps, dfn)[0].mean()))
    return np.array(pw)


def curve_mvn(theta, se, X, kinds, caps, dfn, grid, tau=None, xi=None):
    pw = []
    for n in grid:
        f = np.sqrt(G_DEV / n)
        est = theta[None, :] + f * X
        if tau is not None:
            est = est + tau[None, :] * xi
        pw.append(float(holds(est, se * f, kinds, caps, dfn)[0].mean()))
    return np.array(pw)


def n_at(grid, pw, target):
    i = next((i for i in range(len(grid)) if pw[i] >= target), None)
    return (int(grid[i]), i) if i is not None else (None, None)


def main():
    v2 = V.jload(V.RES / 'CONFIRMATION_FEASIBILITY_RESULTS_V2.json')
    v1 = V.jload(V.RES / 'CONFIRMATION_FEASIBILITY_RESULTS.json')
    mine = V.jload(V.VPRIV / 'dev_results.json')
    with np.load(V.VPRIV / 'dev_endpoint_draws.npz') as z:
        keys = [str(k) for k in z['keys']]
        D_all = z['draws'].copy()
    grid = np.array(v2['grid_households'], dtype=np.int64)
    grid_check = np.unique(np.concatenate(([G_DEV], np.round(np.logspace(np.log10(G_DEV), 8, 100)).astype(np.int64))))
    out = {'unit': 'VER_POWER_V1', 'labels_read': 'none', 'grid_equals_registered_logspace': bool(np.array_equal(grid, grid_check)),
           'grid_step_ratio': float(grid[1] / grid[0]), 'scorecards': {}}
    rng = np.random.default_rng(SEED_MC)
    for SC in ('ENHANCED', 'ORIGINAL'):
        eps = mine['scorecards'][SC]['endpoints']
        cds = conds_of(eps)
        idx = {k.split('::', 1)[1]: i for i, k in enumerate(keys) if k.startswith(SC + '::')}
        D = D_all[:, [idx[k] for k, _, _ in cds]]
        Dc = D - D.mean(0)
        theta = np.array([eps[k]['point'] for k, _, _ in cds])
        se = np.array([eps[k]['se'] for k, _, _ in cds])
        se_chk = float(np.max(np.abs(se - D.std(0, ddof=1))))
        kinds = np.array([kd for _, kd, _ in cds])
        caps = np.array([m for _, _, m in cds])
        names = [k for k, _, _ in cds]
        bi = [names.index(k) for k in BIND]
        r2 = v2['scorecards'][SC]
        res = {'conditions': len(cds), 'by_kind': {k: int((kinds == k).sum()) for k in 'urb'}, 'se_equals_draw_sd_max_diff': se_chk,
               'coordinator_conditions': r2['conditions'], 'coordinator_by_kind': r2['by_kind']}
        # ---------------- A. analytic single-clause n (binding utility clauses, benefits, recipient clauses)
        z80, z90 = norm.ppf(0.8), norm.ppf(0.9)
        single = {}
        for j, (k, kd, m) in enumerate(cds):
            if kd == 'u':
                slack = m - theta[j]
                for dfn, c in (('D1', 0.0), ('D2', C1)):
                    single[f'{k}|{dfn}'] = {'slack': slack, 'se': se[j], 'n80': G_DEV * ((z80 + c) * se[j] / slack) ** 2 if slack > 0 else None,
                                            'n90': G_DEV * ((z90 + c) * se[j] / slack) ** 2 if slack > 0 else None}
            elif kd == 'r':
                slack = m - theta[j]
                single[f'{k}|D1D2'] = {'slack': slack, 'se': se[j], 'n80': G_DEV * ((z80 + C1) * se[j] / slack) ** 2 if slack > 0 else None}
            else:
                for nm, fac in (('S1', 1.0), ('S2x0.75', 0.75)):
                    for dfn, mg in (('D1D2', 0.0), ('D3', 0.020)):
                        slack = fac * theta[j] - mg
                        single[f'{k}|{nm}|{dfn}'] = {'slack': slack, 'se': se[j], 'n80': G_DEV * ((z80 + C1) * se[j] / slack) ** 2 if slack > 0 else None}
        res['A_single_clause'] = single
        res['A_binding_slack_over_se'] = {k: (caps[names.index(k)] - theta[names.index(k)]) / se[names.index(k)] for k in BIND}
        res['A_binding_vs_coordinator'] = {k: {'slack_diff': abs((caps[names.index(k)] - theta[names.index(k)]) - r2['binding_slack'][k]['slack']),
                                               'se_diff': abs(se[names.index(k)] - r2['binding_slack'][k]['se_dev'])} for k in BIND}
        # analytic limits at the cap (correlated boundary clauses; the U0 / Ucal accuracy twins are identical)
        sub = [names.index(k) for k in BIND[:3]]
        R = np.corrcoef(D[:, sub], rowvar=False)
        orth = 1 / 8 + (np.arcsin(R[0, 1]) + np.arcsin(R[0, 2]) + np.arcsin(R[1, 2])) / (4 * np.pi)
        d2cap = float(multivariate_normal(mean=np.zeros(3), cov=R).cdf(np.full(3, -C1)))
        twin_identical = bool(np.array_equal(D[:, names.index(BIND[2])], D[:, names.index(BIND[3])]))
        res['A_limits_at_cap'] = {'D1_orthant_probability': float(orth), 'D2_mvn_cdf_at_minus_c1': d2cap, 'correlations': R.tolist(),
                                  'accuracy_twins_identical_draws': twin_identical,
                                  'coordinator_V2': {'D1': r2['scenarios']['S3_binding_at_cap|D1']['limit_n_to_infinity_mc'],
                                                     'D2': r2['scenarios']['S3_binding_at_cap|D2']['limit_n_to_infinity_mc']},
                                  'coordinator_V1': {'D1': v1['scorecards'][SC]['scenarios']['S3_binding_at_cap|D1']['limit_n_to_infinity'],
                                                     'D2': v1['scorecards'][SC]['scenarios']['S3_binding_at_cap|D2']['limit_n_to_infinity']}}
        # ---------------- scenarios
        scen = {'S1_plugin': theta.copy()}
        for fac, nm in ((0.75, 'S2_benefit_x0.75'), (0.5, 'S2_benefit_x0.5')):
            t = theta.copy(); t[kinds == 'b'] *= fac; scen[nm] = t
        for frac, nm in ((0.5, 'S3_binding_halfway_to_cap'), (1.0, 'S3_binding_at_cap')):
            t = theta.copy()
            for j in bi:
                t[j] = t[j] + frac * (caps[j] - t[j])
            scen[nm] = t
        for sl in (-0.001, 0.0, 0.0005, 0.001, 0.002):
            t = theta.copy()
            for j in bi:
                t[j] = caps[j] - sl
            scen[f'S5_binding_true_slack_{sl}'] = t
        # tau (benefits only): own recomputation from the replay's refit AUCs (VAR, AUX), and |b(s0) - b(s1)| / sqrt 2
        tau_r, tau_h = np.zeros(len(cds)), np.zeros(len(cds))
        for j, (k, kd, _) in enumerate(cds):
            if kd != 'b':
                continue
            s = k[-1]
            ref = k.split('_vs_')[1].split('|')[0]
            comps = [mine['variation'][f'S{s}|{kind}|benefit_vs_{ref}']['sd'] for kind in ('VAR', 'AUX')]
            tau_r[j] = float(np.sqrt(np.sum(np.square(comps))))
            other = k[:-1] + ('1' if k.endswith('0') else '0')
            tau_h[j] = abs(eps[k]['point'] - eps[other]['point']) / np.sqrt(2)
        res['tau_reader_refit'] = {names[j]: tau_r[j] for j in range(len(cds)) if kinds[j] == 'b'}
        res['tau_vs_coordinator_V2_max_abs_diff'] = max(abs(tau_r[j] - r2['tau_reader_refit_rough'][names[j]]) for j in range(len(cds)) if kinds[j] == 'b')
        Zx = rng.standard_normal((len(Dc), len(cds)))           # tau draws paired with the bootstrap rows (method B)
        Xm = np.vstack([(rng.standard_normal((5000, len(Dc))) @ Dc) / np.sqrt(len(Dc) - 1) for _ in range(N_MC // 5000)])
        Zm = rng.standard_normal((N_MC, len(cds)))              # tau draws for method C
        cmp = {}
        todo = [(nm, t, None) for nm, t in scen.items()] + [('S4_plugin_tau_reader_refit_rough', theta, tau_r),
                                                            ('S4_plugin_tau_reader_refit_rough_plus_release_retraining_HYPOTHETICAL', theta, np.sqrt(tau_r ** 2 + tau_h ** 2))]
        for nm, t, tau in todo:
            for dfn in ('D1', 'D2', 'D3'):
                pb = curve_boot(t, se, Dc, kinds, caps, dfn, grid, tau, Zx if tau is not None else None)
                pm = curve_mvn(t, se, Xm, kinds, caps, dfn, grid, tau, Zm if tau is not None else None)
                c = r2['scenarios'][f'{nm}|{dfn}']
                nb, ib = n_at(grid, pb, 0.8)
                nmv, im = n_at(grid, pm, 0.8)
                nc = c['n_households_80']
                ic = int(np.flatnonzero(grid == nc)[0]) if nc else None
                nb9, _ = n_at(grid, pb, 0.9)
                nm9, _ = n_at(grid, pm, 0.9)
                cmp[f'{nm}|{dfn}'] = {'coordinator_V2_n80': nc, 'boot_n80': nb, 'mvn_n80': nmv,
                                      'boot_grid_steps_from_V2': None if (ib is None or ic is None) else ib - ic,
                                      'mvn_grid_steps_from_V2': None if (im is None or ic is None) else im - ic,
                                      'both_none': nc is None and nb is None and nmv is None,
                                      'coordinator_V2_n90': c['n_households_90'], 'boot_n90': nb9, 'mvn_n90': nm9,
                                      'power_at_1e6': {'V2': float(np.interp(1e6, grid, c['curve'])), 'boot': float(np.interp(1e6, grid, pb)), 'mvn': float(np.interp(1e6, grid, pm))},
                                      'power_at_1e8': {'V2': c['curve'][-1], 'boot': float(pb[-1]), 'mvn': float(pm[-1])},
                                      'power_at_3955': {'V2': c['curve'][0], 'boot': float(pb[0]), 'mvn': float(pm[0])},
                                      'max_abs_curve_diff_boot_vs_V2': float(np.max(np.abs(pb - np.array(c['curve'])))),
                                      'max_abs_curve_diff_mvn_vs_V2': float(np.max(np.abs(pm - np.array(c['curve']))))}
        # assurance limit: P(all true margins favourable), true mean ~ N(theta_hat, Sigma_DEV) (own MVN) and the bootstrap proxy
        for dfn in ('D1', 'D2', 'D3'):
            mu_m = theta[None, :] + Xm
            mu_b = theta[None, :] + Dc
            ok_m = np.ones(len(mu_m), dtype=bool)
            ok_b = np.ones(len(mu_b), dtype=bool)
            for j, (k, kd, m) in enumerate(cds):
                if kd in ('u', 'r'):
                    ok_m &= mu_m[:, j] < m
                    ok_b &= mu_b[:, j] < m
                else:
                    g = 0.020 if dfn == 'D3' else 0.0
                    ok_m &= mu_m[:, j] > g
                    ok_b &= mu_b[:, j] > g
            cmp[f'S6_assurance_limit|{dfn}'] = {'mvn': float(ok_m.mean()), 'boot_proxy': float(ok_b.mean()),
                                                 'coordinator_V2': r2['scenarios'][f'S6_assurance|{dfn}']['limit_n_to_infinity_mc']}
        # same-size replication at n = 3955 under the plug-in
        cmp['same_size_replication_plugin'] = {dfn: {'boot': float(holds(theta[None, :] + Dc, se, kinds, caps, dfn)[0].mean()),
                                                     'mvn': float(holds(theta[None, :] + Xm, se, kinds, caps, dfn)[0].mean()),
                                                     'coordinator_V2': r2['same_size_replication_plugin'][dfn]} for dfn in ('D1', 'D2', 'D3')}
        res['comparisons'] = cmp
        out['scorecards'][SC] = res
    gov = {}
    for k in out['scorecards']['ENHANCED']['comparisons']:
        if '|' not in k or k.startswith('S6') or k.startswith('same'):
            continue
        e, o = out['scorecards']['ENHANCED']['comparisons'][k], out['scorecards']['ORIGINAL']['comparisons'][k]
        rk = lambda x: x['boot_n80'] if x['boot_n80'] is not None else float('inf')
        if e['boot_n80'] is None and o['boot_n80'] is None:
            gov[k] = 'ENHANCED' if e['power_at_1e8']['boot'] <= o['power_at_1e8']['boot'] else 'ORIGINAL'
        else:
            gov[k] = 'ENHANCED' if rk(e) >= rk(o) else 'ORIGINAL'
    out['governing_by_boot'] = gov
    out['governing_V2'] = {k: v2['governing_scorecard_by_scenario'][k] for k in gov if k in v2['governing_scorecard_by_scenario']}
    out['governing_disagreements'] = {k: [gov[k], out['governing_V2'].get(k)] for k in gov if gov[k] != out['governing_V2'].get(k)}
    V.jdump(V.VPRIV / 'check8_power.json', out)
    for SC, r in out['scorecards'].items():
        print(SC, r['conditions'], r['by_kind'], 'coord', r['coordinator_conditions'], r['coordinator_by_kind'], 'se chk', r['se_equals_draw_sd_max_diff'], 'tau diff', r['tau_vs_coordinator_V2_max_abs_diff'])
        print('  binding slack/se', {k[8:]: round(v, 4) for k, v in r['A_binding_slack_over_se'].items()})
        print('  single n80', {k: (None if v['n80'] is None else round(v['n80'])) for k, v in r['A_single_clause'].items() if any(b in k for b in BIND) or 'pair_benefit' in k or 'H_LJ|s1|D1D2' in k})
        print('  limits at cap', r['A_limits_at_cap'])
        for k, v in r['comparisons'].items():
            if k.startswith(('S6', 'same')):
                print('  ', k, v)
            else:
                print('   %-75s V2=%s boot=%s mvn=%s steps(b,m)=%s,%s p1e8 V2=%.4f b=%.4f m=%.4f maxdiff b=%.3f m=%.3f' % (
                    k, v['coordinator_V2_n80'], v['boot_n80'], v['mvn_n80'], v['boot_grid_steps_from_V2'], v['mvn_grid_steps_from_V2'],
                    v['power_at_1e8']['V2'], v['power_at_1e8']['boot'], v['power_at_1e8']['mvn'], v['max_abs_curve_diff_boot_vs_V2'], v['max_abs_curve_diff_mvn_vs_V2']))
    print('governing disagreements', out['governing_disagreements'])


if __name__ == '__main__':
    main()
