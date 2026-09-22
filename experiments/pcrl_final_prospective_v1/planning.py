"""Precision planning from already-used 2018 paired household losses (not a stopping gate).

For each primary clause, per-anchor linearized household SEs are computed from the
archived 2018 per-person losses and projected to the fixed 2016 final pool
(15,928 households shared by all anchors). Two dependence cases bracket the
unknown cross-anchor correlation: 'correlated' (SE of the anchor mean = mean of
anchor SEs; conservative) and 'independent' (root-sum-square / 3). Scenarios vary
the true mean and the household variance. None of this changes margins, roles,
weightings or the decision to score the full panel.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .common import ANCHORS, OUT, PRIMARY_ROLES, TASK_ROLE, WEIGHTINGS, atomic_json, now
from .verify_historical import eval_npz

FINAL_HOUSEHOLDS_2016 = 15928
Z = norm.ppf(.975)


def anchor_se(plus, minus, weighted):
    d = plus['loss']-minus['loss']
    w = plus['weights'] if weighted else np.ones(len(d))
    hh, inv = np.unique(plus['households'], return_inverse=True)
    num = np.bincount(inv, weights=w*d); den = np.bincount(inv, weights=w)
    r = num.sum()/den.sum(); n = len(hh)
    resid = num-r*den
    return float(r), float(np.sqrt(n/(n-1)*np.sum(resid**2))/den.sum()), n


def clause_specs(m):
    out = [(f'{m}|task|{w}', TASK_ROLE, w, m, 'J', -.003) for w in WEIGHTINGS]
    out += [(f'{m}|{r}|{w}', r, w, 'J', m, .001) for r in PRIMARY_ROLES for w in WEIGHTINGS]
    return out


def main():
    rows = []
    for m in ('Q', 'D17'):
        for cid, role, w, plus, minus, margin in clause_specs(m):
            per = [anchor_se(eval_npz(a, plus, role), eval_npz(a, minus, role), w == 'PWGTP') for a in ANCHORS]
            est = float(np.mean([p[0] for p in per]))
            n18 = float(np.mean([p[2] for p in per]))
            scale = np.sqrt(n18/FINAL_HOUSEHOLDS_2016)
            se_corr = float(np.mean([p[1] for p in per]))*scale
            se_ind = float(np.sqrt(np.sum([p[1]**2 for p in per]))/3)*scale
            scen = []
            for shrink in (1., .5, 0.):
                for vmult in (1., 1.5, 2.):
                    mu = shrink*est
                    se = se_corr*np.sqrt(vmult)
                    scen.append({'true_mean': mu, 'mean_fraction_of_2018': shrink, 'variance_multiplier': vmult,
                                 'projected_se': se, 'one_sided_halfwidth': Z*se,
                                 'pass_probability': float(norm.cdf((margin-mu)/se-Z))})
            rows.append({'clause': cid, 'margin': margin, 'estimate_2018': est,
                         'anchor_se_2018': [p[1] for p in per], 'households_per_anchor_2018': [p[2] for p in per],
                         'projected_se_correlated': se_corr, 'projected_se_independent': se_ind, 'scenarios': scen})
    record = {'created_utc': now(), 'source': 'archived 2018 test-pool per-person expected losses (already used)',
              'final_households_2016': FINAL_HOUSEHOLDS_2016, 'z_one_sided_975': Z, 'rows': rows,
              'caveats': ['cross-year variance, household dependence, anchor correlation and attacker quality can all change',
                          'a historical half-width is not a lower bound for another comparison',
                          'a one-sided bound can pass with a sufficiently negative estimate even if its half-width exceeds .001',
                          'bitwise-identical predictions have zero paired variance at any sample size',
                          'repeated rows, tokens and anchors do not create independent households',
                          'this table communicates precision and checks implementation; it is not a stopping gate']}
    atomic_json(OUT/'PRECISION_PLANNING.json', record)
    return record


if __name__ == '__main__':
    r = main()
    for row in r['rows']:
        s0 = row['scenarios'][0]
        print(f"{row['clause']:32s} est18={row['estimate_2018']:+.5f} se16c={row['projected_se_correlated']:.5f} "
              f"se16i={row['projected_se_independent']:.5f} P(pass|2018 mean)={s0['pass_probability']:.3f} "
              f"P(pass|half)={row['scenarios'][3]['pass_probability']:.3f}")
