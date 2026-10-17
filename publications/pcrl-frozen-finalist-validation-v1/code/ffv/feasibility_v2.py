"""Unit FFV_FEASIBILITY_V2: confirmation-feasibility analysis, version 2 (post-lock correction; ffv/feasibility.py stays as locked).

Written after the statistical reviewer's feasibility review (roles/statistical_reviewer/FEASIBILITY_REVIEW.md, items E1-E5). Same
inputs, conditions, scenarios, Monte Carlo draws (seed 2026101803) and decision rules as V1; no label is read. Changes:
- E1: n -> infinity limits are Monte Carlo limits at n = 1e14, which respect the correlation of boundary clauses (V1 multiplied
  per-clause limits as if independent: 0.0625 instead of about 0.265 at the cap);
- E2/E3: the refit term is a ROUGH measured spread, not a lower bound; it is added to the 4 pair-benefit endpoints only, with
  independent effects across benefits (conservative); it comes from the ENHANCED winners and is applied to both scorecards; the
  combined scenario is named for both of its terms and keeps the HYPOTHETICAL label;
- E4: full power curves are saved; point values are labelled as interpolated; when neither scorecard reaches 80% by 1e8, the lower
  power at 1e8 governs;
- E5: assurance limits are computed; the same-size replication probability under the plug-in is reported (descriptive).

Estimand: the frozen releases with the frozen fitted readers and frozen winners, applied to n new independent households drawn
from the same population (Florida ACS 2019 development population), unweighted, no transport. The curves do not apply to designs
that refit readers or re-select winners (that endpoint is a maximum over candidates), transport to another state or year, or use
design weights. The two seeds share people and are not replications.

Approximation (registered, reviewed): estimates for n households ~ MVN(theta, Sigma_DEV x 3955 / n), Sigma_DEV from the 2048 shared
household-bootstrap replicates of all endpoints jointly, eigendecomposed; decisions with SE(n) = SE_DEV x sqrt(3955 / n), c1 = 2.395.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C
from .feasibility import BINDING, C1, C_B_ONLY, G_DEV, GRID, MC_SEED, N_MC, conditions, decide

N_LIMIT = 1e14
SLACK_GRID = (-0.001, 0.0, 0.0005, 0.001, 0.002)


def powers(ns, theta, se_dev, ZL, xi, conds, definition, tau=None, c=C1, mu_draw=None):
    pw, pc = [], []
    mu = theta[None, :] if mu_draw is None else mu_draw
    for n in ns:
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
            'power_at_interpolated': {str(int(n)): float(np.interp(n, GRID, pw)) for n in (G_DEV, 1e4, 1e5, 1e6, 1e7, 1e8)},
            'limit_n_to_infinity_mc': lim, 'curve': [float(x) for x in pw], 'mc_se_max': float(np.sqrt(0.25 / N_MC)),
            'weakest_conditions_at_n80_or_1e8': [{'condition': conds[j][0], 'marginal_power': pc[at][j]} for j in worst]}


def main(report_path=None, draws_path=None, out_path=None):
    report_path = Path(report_path) if report_path else C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json'
    draws_path = Path(draws_path) if draws_path else C.RUNS / 'REPORT' / 'ENDPOINT_DRAWS.npz'
    out_path = Path(out_path) if out_path else C.RESULTS / 'CONFIRMATION_FEASIBILITY_RESULTS_V2.json'
    rep = json.loads(report_path.read_text())
    with np.load(draws_path) as z:
        keys = [str(k) for k in z['keys']]; D = z['draws'].copy()
    out = {'schema': 'FFV1-confirmation-feasibility-v2', 'utc': C.now(), 'method': __doc__, 'grid_households': [int(x) for x in GRID],
           'limit_method': f'Monte Carlo at n = {N_LIMIT:.0e} with the same draws (respects the correlation of boundary clauses)',
           'mc_draws': N_MC, 'mc_seed': MC_SEED, 'binding_clauses': list(BINDING), 'scorecards': {},
           'supersedes': 'CONFIRMATION_FEASIBILITY_RESULTS.json (V1, kept): independence-product limits, NaN assurance limits, "lower bound" wording, curves not saved'}
    for sc in ('ENHANCED', 'ORIGINAL'):
        eps = rep['scorecards'][sc]['endpoints']
        idx = {k.split('::', 1)[1]: i for i, k in enumerate(keys) if k.startswith(sc + '::')}
        conds = conditions(eps)
        X = D[:, [idx[k] for k, _, _ in conds]]
        Sigma = np.cov(X, rowvar=False); w, V = np.linalg.eigh(Sigma); L = V * np.sqrt(np.clip(w, 0, None))[None, :]
        rng = np.random.default_rng(MC_SEED); Z = rng.standard_normal((N_MC, len(conds))); xi = rng.standard_normal((N_MC, len(conds)))
        Zmu = rng.standard_normal((N_MC, len(conds)))
        ZL = Z @ L.T
        theta = np.array([eps[k]['point'] for k, _, _ in conds]); se = np.array([eps[k]['se'] for k, _, _ in conds])
        kinds = np.array([kd for _, kd, _ in conds]); caps = np.array([m for _, _, m in conds])
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
            t = theta.copy()
            for j, (k, _, _) in enumerate(conds):
                if k in BINDING:
                    t[j] = caps[j] - sl
            scen[f'S5_binding_true_slack_{sl}'] = t
        res = {'conditions': len(conds), 'by_kind': {kd: int((kinds == kd).sum()) for kd in ('utility', 'recipient', 'benefit')},
               'binding_slack': {k: {'point': eps[k]['point'], 'cap': eps[k]['margin'], 'slack': eps[k]['margin'] - eps[k]['point'], 'se_dev': eps[k]['se'],
                                     'slack_over_se': (eps[k]['margin'] - eps[k]['point']) / eps[k]['se']} for k in BINDING if k in eps},
               'benefit_points': {k: float(theta[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'tau_reader_refit_rough': {k: float(tau_r[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'tau_reader_refit_rough_note': 'rough measured refit spread of the ENHANCED pair winners (model-seed and household-bootstrap refits, each at most 4 degrees of freedom); the bootstrap refits read systematically lower (24 of 24), so it mixes a resampling shift with variance; direction of the net bias unknown; applied to both scorecards; benefits only',
               'tau_release_retraining_HYPOTHETICAL': {k: float(tau_h[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'scenarios': {}}
        for name, th in scen.items():
            for dfn in ('D1', 'D2', 'D3'):
                pw, pc = powers(GRID, th, se, ZL, xi, conds, dfn)
                lim = float(powers([N_LIMIT], th, se, ZL, xi, conds, dfn)[0][0])
                res['scenarios'][f'{name}|{dfn}'] = summarize(pw, pc, conds, lim)
        for tau_name, tau in (('reader_refit_rough', tau_r), ('reader_refit_rough_plus_release_retraining_HYPOTHETICAL', np.sqrt(tau_r ** 2 + tau_h ** 2))):
            for dfn in ('D1', 'D2', 'D3'):
                pw, pc = powers(GRID, theta, se, ZL, xi, conds, dfn, tau=tau)
                lim = float(powers([N_LIMIT], theta, se, ZL, xi, conds, dfn, tau=tau)[0][0])
                res['scenarios'][f'S4_plugin_tau_{tau_name}|{dfn}'] = summarize(pw, pc, conds, lim)
        mu_draw = theta[None, :] + Zmu @ L.T
        for dfn in ('D1', 'D2', 'D3'):
            pw, pc = powers(GRID, theta, se, ZL, xi, conds, dfn, mu_draw=mu_draw)
            lim = float(powers([N_LIMIT], theta, se, ZL, xi, conds, dfn, mu_draw=mu_draw)[0][0])
            res['scenarios'][f'S6_assurance|{dfn}'] = summarize(pw, pc, conds, lim)
        pw, pc = powers(GRID, theta, se, ZL, xi, conds, 'D1', c=C_B_ONLY)
        res['sensitivity_B_only_c1.960_plugin_D1'] = summarize(pw, pc, conds, float(powers([N_LIMIT], theta, se, ZL, xi, conds, 'D1', c=C_B_ONLY)[0][0]))
        res['same_size_replication_plugin'] = {dfn: float(powers([G_DEV], theta, se, ZL, xi, conds, dfn)[0][0]) for dfn in ('D1', 'D2', 'D3')}
        out['scorecards'][sc] = res
    gov = {}
    for key in out['scorecards']['ENHANCED']['scenarios']:
        e, o = out['scorecards']['ENHANCED']['scenarios'][key], out['scorecards']['ORIGINAL']['scenarios'][key]
        if e['n_households_80'] is None and o['n_households_80'] is None:
            gov[key] = 'ENHANCED' if e['curve'][-1] <= o['curve'][-1] else 'ORIGINAL'
        else:
            rank = lambda x: x['n_households_80'] if x['n_households_80'] is not None else float('inf')
            gov[key] = 'ENHANCED' if rank(e) >= rank(o) else 'ORIGINAL'
    out['governing_scorecard_by_scenario'] = gov
    out['governing_rule'] = 'the larger n80 governs; when neither reaches 80% by 1e8, the lower power at 1e8 governs'
    C.write_json_x(out_path, out)
    for sc, r in out['scorecards'].items():
        print(sc, json.dumps({k: (v['n_households_80'], v['status_80'], round(v['limit_n_to_infinity_mc'], 3)) for k, v in r['scenarios'].items()}))
        print(sc, 'same-size replication', r['same_size_replication_plugin'])


if __name__ == '__main__':
    main(*sys.argv[1:4])
