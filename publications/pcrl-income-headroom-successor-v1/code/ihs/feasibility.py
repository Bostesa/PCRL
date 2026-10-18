"""Unit IHS_FEASIBILITY: planning feasibility for the frozen successor from existing evidence only (brief section 12). No label is
read; inputs are the logged DEV_REPORT evaluation's private per-replicate endpoint draws and its published points, plus the
final readers' refit roster. Method as the validation's V2 (ffv/feasibility_v2.py), adapted to this campaign's endpoint keys.

Estimand: the frozen successor and comparators with the frozen fitted readers and winners, applied to n new independent
households from the same Florida ACS 2019 population, unweighted, no transport (a transport scenario is labelled HYPOTHETICAL).
Approximation: estimates for n households ~ MVN(theta, Sigma_DEV x 3955 / n), Sigma_DEV from the 2048 shared household-bootstrap
replicates of the joint success conditions, eigendecomposed; decisions with SE(n) = SE_DEV x sqrt(3955 / n) and the registered
clause critical value c1. Joint power: Monte Carlo probability (40,000 draws, seed 2026101803, common random numbers) that every
condition holds. The n -> infinity limits are Monte Carlo values at n = 1e14.

Definitions (ENHANCED and ORIGINAL scorecards; the less favourable governs):
- D1, the registered competitive conjunction P: the 28 utility and guard POINT gates, the 8 recipient clause bounds <= 0.010
  and the 4 pair-benefit clause bounds > 0 (claims P / N);
- D2, confirmatory: D1 with every utility and guard UPPER BOUND (point + c1 SE) <= its cap;
- D3: D2 plus every pair-benefit bound > 0.020.
Binding clauses (rule fixed before the DEV_REPORT evaluation): the 4 utility or guard conditions of the successor with the
smallest DEV margin in SE units.
Scenarios: S1 plug-in; S2 benefits x 0.75 and x 0.5; S3 binding clauses halfway to the cap and at the cap; S5 binding clauses at
an assumed true slack (-0.001, 0, 0.0005, 0.001, 0.002); S4 plug-in plus the rough measured reader-refit spread on the benefits;
S6 assurance (flat-prior predictive heuristic); S7 HYPOTHETICAL transport (benefits x 0.5 and every utility margin halved);
same-size replication; power at the ~17,510 untested same-population households (planning only; no population is opened).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C

G_DEV = 3955
GRID = np.unique(np.concatenate(([G_DEV, 17510], np.round(np.logspace(np.log10(G_DEV), 8, 100)).astype(np.int64))))
N_MC, MC_SEED, N_LIMIT = 40000, 2026101803, 1e14
SLACK_GRID = (-0.001, 0.0, 0.0005, 0.001, 0.002)


def conditions(eps):
    out = []
    for k, e in eps.items():
        if e['se'] == 0.0:
            continue
        if k.startswith(('UTILITY|S|', 'SUBGROUP|S|')):
            out.append((k, 'utility', e['margin']))
        elif k.startswith('NW|') and k.split('|')[1] in ('NEURAL_RAW', 'H_LJ'):
            out.append((k, 'recipient', 0.010))
        elif k.startswith('BEN|') and k.split('|')[1] in ('NEURAL_RAW', 'H_LJ'):
            out.append((k, 'benefit', 0.0))
    return out


def decide(est, se, conds, definition, c):
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


def powers(ns, theta, se_dev, ZL, xi, conds, definition, c, tau=None, mu_draw=None):
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
    at = int(np.flatnonzero(GRID == n80)[0]) if n80 else len(GRID) - 1
    worst = sorted(range(len(conds)), key=lambda j: pc[at][j])[:4]
    return {'n_households_80': n80, 'n_households_90': n90,
            'status_80': 'REACHED' if n80 else ('LIMIT_BELOW_TARGET' if lim < 0.8 else 'NOT_REACHED_BY_1E8'),
            'power_at': {str(int(n)): float(pw[int(np.flatnonzero(GRID == n)[0])]) if n in GRID else float(np.interp(n, GRID, pw))
                         for n in (G_DEV, 17510, 100000, 1000000, 10000000, 100000000)},
            'limit_n_to_infinity_mc': lim, 'curve': [float(x) for x in pw],
            'weakest_conditions_at_n80_or_1e8': [{'condition': conds[j][0], 'marginal_power': pc[at][j]} for j in worst]}


def main(out_path=None):
    rep = json.loads((C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    c1 = lock['inference']['critical_values_expected']['c1']
    with np.load(C.RUNS / 'REPORT' / 'ENDPOINT_DRAWS.npz') as z:
        draws = {k: z[k].copy() for k in z.files}
    out = {'schema': 'IHS1-confirmation-feasibility-v1', 'utc': C.now(), 'method': __doc__, 'grid_households': [int(x) for x in GRID],
           'mc_draws': N_MC, 'mc_seed': MC_SEED, 'c1': c1, 'scorecards': {}}
    for sc in ('ENHANCED', 'ORIGINAL'):
        eps = rep['scorecards'][sc]['endpoints']
        conds = conditions(eps)
        X = np.column_stack([draws[f'{sc}|{k}'] for k, _, _ in conds])
        Sigma = np.cov(X, rowvar=False); w, V = np.linalg.eigh(Sigma); L = V * np.sqrt(np.clip(w, 0, None))[None, :]
        rng = np.random.default_rng(MC_SEED)
        Z = rng.standard_normal((N_MC, len(conds))); xi = rng.standard_normal((N_MC, len(conds))); Zmu = rng.standard_normal((N_MC, len(conds)))
        ZL = Z @ L.T
        theta = np.array([eps[k]['point'] for k, _, _ in conds]); se = np.array([eps[k]['se'] for k, _, _ in conds])
        kinds = np.array([kd for _, kd, _ in conds]); caps = np.array([m for _, _, m in conds])
        util = [j for j, (_, kd, _) in enumerate(conds) if kd == 'utility']
        binding = sorted(util, key=lambda j: (caps[j] - theta[j]) / se[j])[:4]
        tau = np.zeros(len(conds))
        for j, (k, kd, _) in enumerate(conds):
            if kd == 'benefit':
                s = int(k[-1]); ref = k.split('|')[1]
                rr = rep['per_seed'][f'S{s}']['refit_roster']
                vals = []
                for svc in ('SUCCESSOR', ref):
                    if svc in rr:
                        vals += [list(rr[svc]['variation'].values()), list(rr[svc]['aux_sample'].values())]
                sds = [float(np.std(v, ddof=1)) for v in vals if len(v) > 1]
                tau[j] = float(np.sqrt(np.sum(np.square(sds)))) if sds else 0.0
        scen = {'S1_plugin': theta.copy()}
        for f_, nm in ((0.75, 'S2_benefit_x0.75'), (0.5, 'S2_benefit_x0.5')):
            t = theta.copy(); t[kinds == 'benefit'] *= f_; scen[nm] = t
        for frac, nm in ((0.5, 'S3_binding_halfway_to_cap'), (1.0, 'S3_binding_at_cap')):
            t = theta.copy()
            for j in binding:
                t[j] = t[j] + frac * (caps[j] - t[j])
            scen[nm] = t
        for sl in SLACK_GRID:
            t = theta.copy()
            for j in binding:
                t[j] = caps[j] - sl
            scen[f'S5_binding_true_slack_{sl}'] = t
        t = theta.copy(); t[kinds == 'benefit'] *= 0.5
        for j in util:
            t[j] = caps[j] - 0.5 * (caps[j] - t[j])
        scen['S7_transport_HYPOTHETICAL_benefit_x0.5_utility_margin_x0.5'] = t
        res = {'conditions': len(conds), 'binding_rule': '4 utility/guard conditions with the smallest DEV margin in SE units',
               'binding': {conds[j][0]: {'point': float(theta[j]), 'cap': float(caps[j]), 'se_dev': float(se[j]),
                                         'margin_se_units': float((caps[j] - theta[j]) / se[j]),
                                         'margin_se_units_at_17510': float((caps[j] - theta[j]) / (se[j] * np.sqrt(G_DEV / 17510)))} for j in binding},
               'benefit_points': {k: float(theta[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'tau_reader_refit_rough': {k: float(tau[j]) for j, (k, kd, _) in enumerate(conds) if kd == 'benefit'},
               'scenarios': {}}
        for nm, th in scen.items():
            for dfn in ('D1', 'D2', 'D3'):
                pw, pc = powers(GRID, th, se, ZL, xi, conds, dfn, c1)
                lim = float(powers([N_LIMIT], th, se, ZL, xi, conds, dfn, c1)[0][0])
                res['scenarios'][f'{nm}|{dfn}'] = summarize(pw, pc, conds, lim)
        for dfn in ('D1', 'D2', 'D3'):
            pw, pc = powers(GRID, theta, se, ZL, xi, conds, dfn, c1, tau=tau)
            res['scenarios'][f'S4_plugin_reader_refit_rough|{dfn}'] = summarize(pw, pc, conds, float(powers([N_LIMIT], theta, se, ZL, xi, conds, dfn, c1, tau=tau)[0][0]))
            mu_draw = theta[None, :] + Zmu @ L.T
            pw, pc = powers(GRID, theta, se, ZL, xi, conds, dfn, c1, mu_draw=mu_draw)
            res['scenarios'][f'S6_assurance|{dfn}'] = summarize(pw, pc, conds, float(powers([N_LIMIT], theta, se, ZL, xi, conds, dfn, c1, mu_draw=mu_draw)[0][0]))
        res['same_size_replication_plugin'] = {dfn: float(powers([G_DEV], theta, se, ZL, xi, conds, dfn, c1)[0][0]) for dfn in ('D1', 'D2', 'D3')}
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
    C.write_json_x(Path(out_path) if out_path else C.RESULTS / 'CONFIRMATION_FEASIBILITY_RESULTS.json', out)
    for sc, r in out['scorecards'].items():
        print(sc, json.dumps({k: (v['n_households_80'], v['status_80'], round(v['power_at']['17510'], 3)) for k, v in r['scenarios'].items()}))


if __name__ == '__main__':
    main(*sys.argv[1:2])
