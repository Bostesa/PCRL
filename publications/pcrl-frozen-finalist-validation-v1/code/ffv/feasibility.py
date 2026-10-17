"""Unit FFV_FEASIBILITY: confirmation-feasibility analysis from existing evidence only (PROTOCOL.md section 10). No label is read;
inputs are the logged evaluation's private per-replicate endpoint draws and its published points and refit spreads.

Estimand: the frozen releases with the frozen fitted readers and frozen winners, applied to n new independent households drawn
from the same population (Florida ACS 2019 development population), unweighted, no transport.

Approximation (registered, reviewed): every endpoint is a household-clustered ratio mean or a two-sample U-statistic (AUC), so
the estimates for n households are approximately multivariate normal with mean theta (a scenario) and covariance
Sigma_DEV x 3955 / n, where Sigma_DEV is the covariance of the 2048 shared household-bootstrap replicates of all endpoints jointly
(both seeds share people; the covariance carries that). Sigma is factored by eigendecomposition (negative eigenvalues clipped).
Decisions use the scorecard rules with SE(n) = SE_DEV x sqrt(3955 / n) and clause critical value c1 = 2.3949991. Joint power is
the Monte Carlo probability (40,000 draws, seed 2026101803, common random numbers across n) that every required condition holds.

A non-shrinking component (refit variation) is a study-level random effect added to AUC-type endpoints only; it is not covered by
conditional intervals, so with it the power is bounded as n grows. The reader-refit component is measured (RNG and auxiliary-sample
refits; a lower bound); the release-retraining component is HYPOTHETICAL (|benefit(s0) - benefit(s1)| / sqrt 2; 1 degree of
freedom, confounded with the reader seed and same-people noise).
"""
from __future__ import annotations
import json

import numpy as np
from scipy.stats import norm

from . import common as C

C1 = 2.3949991
C_B_ONLY = 1.959964
G_DEV = 3955
GRID = np.unique(np.concatenate(([G_DEV], np.round(np.logspace(np.log10(G_DEV), 8, 100)).astype(np.int64))))
N_MC = 40000
MC_SEED = 2026101803
BINDING = ('UTILITY|income|Ucal|brier_excess|s0', 'UTILITY|income|Ucal|brier_excess|s1',
           'UTILITY|income|U0|accuracy_loss|s1', 'UTILITY|income|Ucal|accuracy_loss|s1')
SLACK_GRID = (-0.001, 0.0, 'plug-in', 0.0005, 0.001, 0.002)


def conditions(eps):
    """(key, kind, margin) for the joint success event; zero-variance (IDENTICAL_RELEASE) endpoints are deterministic passes."""
    out = []
    for k, e in eps.items():
        if e['family'] != 'P' or e['se'] == 0.0:
            continue
        if k.startswith(('UTILITY|', 'SUBGROUP|')):
            out.append((k, 'utility', e['margin']))
        elif k.startswith('COMPETITIVE|') and '_worsening_' in k:
            out.append((k, 'recipient', 0.010))
        elif k.startswith('COMPETITIVE|pair_benefit_vs_'):
            out.append((k, 'benefit', 0.0))
    return out


def decide(est, se, conds, definition, c=C1):
    ok = np.ones(len(est), dtype=bool); per = []
    for j, (k, kind, m) in enumerate(conds):
        if kind == 'utility':
            cj = est[:, j] <= m if definition == 'D1' else est[:, j] + c * se[j] <= m
        elif kind == 'recipient':
            cj = est[:, j] + c * se[j] <= m
        else:
            cj = est[:, j] - c * se[j] > (0.020 if definition == 'D3' else 0.0)
        ok &= cj; per.append(cj)
    return ok, per


def limit(theta, conds, definition, tau):
    """Analytic n -> infinity joint power (no sampling noise): with no extra variance, each condition is decided by the sign of its
    true margin (a true value exactly at a point-gate cap gives 1/2, at a bound cap Phi(-c1)); with tau, independent normal limits."""
    p = 1.0
    for j, (k, kind, m) in enumerate(conds):
        t = theta[j]; tj = 0.0 if tau is None else tau[j]
        if kind in ('utility', 'recipient'):
            d = m - t
            if tj > 0:
                p *= norm.cdf(d / tj)
            else:
                p *= 1.0 if d > 0 else (0.5 if (d == 0 and kind == 'utility' and definition == 'D1') else (norm.cdf(-C1) if d == 0 else 0.0))
        else:
            d = t - (0.020 if definition == 'D3' else 0.0)
            p *= norm.cdf(d / tj) if tj > 0 else (1.0 if d > 0 else 0.0)
    return float(p)


def curve(theta, se_dev, L, Z, xi, conds, definition, tau=None, c=C1, mu_draw=None):
    pw, pc = [], []
    ZL = Z @ L.T
    mu = theta[None, :] if mu_draw is None else mu_draw
    for n in GRID:
        f = np.sqrt(G_DEV / n)
        est = mu + f * ZL
        if tau is not None:
            est = est + tau[None, :] * xi
        ok, per = decide(est, se_dev * f, conds, definition, c)
        pw.append(float(ok.mean())); pc.append([float(x.mean()) for x in per])
    return np.array(pw), np.array(pc)


def summarize(pw, pc, conds, lim):
    n80 = next((int(GRID[i]) for i in range(len(GRID)) if pw[i] >= 0.8), None)
    n90 = next((int(GRID[i]) for i in range(len(GRID)) if pw[i] >= 0.9), None)
    def bracket(n):
        if n is None:
            return None
        i = int(np.flatnonzero(GRID == n)[0])
        return [int(GRID[i - 1]) if i > 0 else None, int(GRID[i])]
    at = int(np.flatnonzero(GRID == n80)[0]) if n80 else len(GRID) - 1
    worst = sorted(range(len(conds)), key=lambda j: pc[at][j])[:4]
    status = ('REACHED' if n80 else ('LIMIT_BELOW_TARGET' if lim < 0.8 else 'NOT_REACHED_BY_1E8'))
    return {'n_households_80': n80, 'bracket_80': bracket(n80), 'n_households_90': n90, 'bracket_90': bracket(n90), 'status_80': status,
            'power_at': {str(int(n)): float(np.interp(n, GRID, pw)) for n in (G_DEV, 1e4, 1e5, 1e6, 1e7, 1e8)}, 'limit_n_to_infinity': lim,
            'mc_se_max': float(np.sqrt(0.25 / N_MC)),
            'weakest_conditions_at_n80_or_1e8': [{'condition': conds[j][0], 'marginal_power': pc[at][j]} for j in worst]}


def main():
    rep = json.loads((C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    with np.load(C.RUNS / 'REPORT' / 'ENDPOINT_DRAWS.npz') as z:
        keys = [str(k) for k in z['keys']]; D = z['draws'].copy()
    out = {'schema': 'FFV1-confirmation-feasibility-v1', 'utc': C.now(), 'method': __doc__, 'grid_households': [int(GRID[0]), int(GRID[-1]), len(GRID)],
           'mc_draws': N_MC, 'mc_seed': MC_SEED, 'binding_clauses': list(BINDING), 'scorecards': {}}
    for sc in ('ENHANCED', 'ORIGINAL'):
        eps = rep['scorecards'][sc]['endpoints']
        idx = {k.split('::', 1)[1]: i for i, k in enumerate(keys) if k.startswith(sc + '::')}
        conds = conditions(eps)
        X = D[:, [idx[k] for k, _, _ in conds]]
        Sigma = np.cov(X, rowvar=False); w, V = np.linalg.eigh(Sigma); L = V * np.sqrt(np.clip(w, 0, None))[None, :]
        rng = np.random.default_rng(MC_SEED); Z = rng.standard_normal((N_MC, len(conds))); xi = rng.standard_normal((N_MC, len(conds)))
        Zmu = rng.standard_normal((N_MC, len(conds)))
        theta = np.array([eps[k]['point'] for k, _, _ in conds]); se = np.array([eps[k]['se'] for k, _, _ in conds])
        kinds = np.array([kd for _, kd, _ in conds]); caps = np.array([m for _, _, m in conds])
        # refit components (benefits only): measured reader refit (RNG + auxiliary sample; lower bound) and hypothetical release
        tau_r, tau_h = np.zeros(len(conds)), np.zeros(len(conds))
        for j, (k, kd, m) in enumerate(conds):
            if kd != 'benefit':
                continue
            s = 'S' + k[-1]; ref = k.split('_vs_')[1].split('|')[0]
            bm = rep['per_seed'][s]['variation'].get('benefit_matched_refits', {})
            comps = [bm[f'{kind_}|{ref}']['sd'] for kind_ in ('VAR', 'AUX') if f'{kind_}|{ref}' in bm and bm[f'{kind_}|{ref}']['sd'] is not None]
            tau_r[j] = float(np.sqrt(np.sum(np.square(comps)))) if comps else 0.0
            other = k[:-1] + ('1' if k.endswith('0') else '0')
            tau_h[j] = abs(eps[k]['point'] - eps[other]['point']) / np.sqrt(2)
        scen = {}
        t = theta.copy(); t[kinds == 'benefit'] *= 0.75; scen['S2_benefit_x0.75'] = t
        scen['S1_plugin'] = theta.copy()
        t = theta.copy(); t[kinds == 'benefit'] *= 0.5; scen['S2_benefit_x0.5'] = t
        for frac, name in ((0.5, 'S3_binding_halfway_to_cap'), (1.0, 'S3_binding_at_cap')):
            t = theta.copy()
            for j, (k, _, _) in enumerate(conds):
                if k in BINDING:
                    t[j] = t[j] + frac * (caps[j] - t[j])
            scen[name] = t
        for sl in SLACK_GRID:
            if sl == 'plug-in':
                continue
            t = theta.copy()
            for j, (k, _, _) in enumerate(conds):
                if k in BINDING:
                    t[j] = caps[j] - sl
            scen[f'S5_binding_true_slack_{sl}'] = t
        res = {'conditions': len(conds), 'by_kind': {kd: int((kinds == kd).sum()) for kd in ('utility', 'recipient', 'benefit')},
               'binding_slack': {k: {'point': eps[k]['point'], 'cap': eps[k]['margin'], 'slack': eps[k]['margin'] - eps[k]['point'], 'se_dev': eps[k]['se'],
                                     'slack_over_se': (eps[k]['margin'] - eps[k]['point']) / eps[k]['se']} for k in BINDING if k in eps},
               'benefit_points': {k: float(theta[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'tau_reader_refit_measured_lower_bound': {k: float(tau_r[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'tau_release_retraining_HYPOTHETICAL': {k: float(tau_h[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'scenarios': {}}
        for name, th in scen.items():
            for dfn in ('D1', 'D2', 'D3'):
                pw, pc = curve(th, se, L, Z, xi, conds, dfn)
                res['scenarios'][f'{name}|{dfn}'] = summarize(pw, pc, conds, limit(th, conds, dfn, None))
        for tau_name, tau in (('reader_refit_measured', tau_r), ('release_retraining_HYPOTHETICAL', np.sqrt(tau_r ** 2 + tau_h ** 2))):
            for dfn in ('D1', 'D2', 'D3'):
                pw, pc = curve(theta, se, L, Z, xi, conds, dfn, tau=tau)
                res['scenarios'][f'S4_plugin_tau_{tau_name}|{dfn}'] = summarize(pw, pc, conds, limit(theta, conds, dfn, tau))
        mu_draw = theta[None, :] + Zmu @ L.T                       # assurance: true mean drawn from N(theta_hat, Sigma_DEV)
        for dfn in ('D1', 'D2', 'D3'):
            pw, pc = curve(theta, se, L, Z, xi, conds, dfn, mu_draw=mu_draw)
            res['scenarios'][f'S6_assurance|{dfn}'] = summarize(pw, pc, conds, float('nan'))
        pw, pc = curve(theta, se, L, Z, xi, [c_ for c_ in conds], 'D1', c=C_B_ONLY)
        res['sensitivity_B_only_c1.960_plugin_D1'] = summarize(pw, pc, conds, limit(theta, conds, 'D1', None))
        out['scorecards'][sc] = res
    gov = {}
    for key in out['scorecards']['ENHANCED']['scenarios']:
        e, o = out['scorecards']['ENHANCED']['scenarios'][key], out['scorecards']['ORIGINAL']['scenarios'][key]
        def rank(x):
            return x['n_households_80'] if x['n_households_80'] is not None else float('inf')
        gov[key] = 'ENHANCED' if rank(e) >= rank(o) else 'ORIGINAL'
    out['governing_scorecard_by_scenario'] = gov
    C.write_json_x(C.RESULTS / 'CONFIRMATION_FEASIBILITY_RESULTS.json', out)
    for sc, r in out['scorecards'].items():
        print(sc, json.dumps({k: (v['n_households_80'], v['status_80'], round(v['power_at']['100000'], 3)) for k, v in r['scenarios'].items()}))


if __name__ == '__main__':
    main()
