"""CHECK3 (part 2): sampling consequence of the float-split ties in the known score used by B_same / B_indep.

Part 1 found that the study's merged-view float posterior V['post'] = m1 / (m1 + m0) takes MORE distinct values than the exact
posterior of the same law in 18 of 52 views: wires whose posteriors are exactly equal (e.g. REF children of one parent, REF atoms
2k / 2k + 1, RED pair cells (g, z) of one g, every IND wire) get posteriors that differ in the last bits, so b_known ranks them
instead of tying them (ties 1/2 is violated for those pairs). The population AUC A is unaffected (part 1: |A_study - A_exact| <=
1.5e-15) and a tie-broken optimal ranking is still population-optimal, so B stays unbiased for A; but the per-sample value and the
paired comparisons change.

Method: own samples (own sampler, own seeds) of n = 2,000 and 20,000 from the affected study laws. For every view: the study's
b_known (float posterior, its IF) and the exact known-score AUC with exact ties (people grouped by their wire's EXACT posterior;
brute force over all pairs of distinct exact posterior values with integer counts). For the dA = 0 comparisons of those laws:
d_float (study) with the study's linearized paired SE and the R2SE rule (resolved iff |d| > 2 se), against d_exact, which is
exactly 0 because the exact known score of every person is identical in both views (REF levels: q of the coarse atom; RED pair vs
income: S independent of occupation given income; IND / COMP single views: all posteriors equal pi)."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402
from edg import estimators as E       # noqa: E402

T0 = V.utc_now()
VSEED = 202610251805
CASES = (('REF', 0.5, 'skewed', [('L64', 'L16'), ('L256', 'L16'), ('L1024', 'L16')]),
         ('REF', 0.2, 'skewed', [('L64', 'L16'), ('L256', 'L16'), ('L1024', 'L16')]),
         ('RED', 0.5, 'skewed', [('pair', 'income')]),
         ('RED', 0.2, 'skewed', [('pair', 'income')]),
         ('IND', 0.2, 'skewed', [('pair', 'income'), ('pair', 'occupation'), ('occupation', 'income')]),
         ('COMP', 0.5, 'uniform', [('occupation', 'income')]))
REPS = {2000: 100, 20000: 40}
rec = {'check': 3, 'name': 'known_score_float_tie_split_consequence', 'reps': REPS, 'cases': {}}
for ci, (fam, pi, occ, cmps) in enumerate(CASES):
    law = L.build(fam, pi, occ)
    views = sorted({v for c in cmps for v in c})
    lev_of_st = {}
    for name in views:
        Vw = law['views'][name]
        wid, K, m1, m0 = V.study_law_exact_view(law, name)
        own2st = np.full(K, -1)
        own2st[wid] = Vw['inv']
        pe = V.posterior_exact(m1, m0)
        keys_st = [None] * K
        for w in range(K):
            keys_st[own2st[w]] = pe[w]
        uniq = sorted(set(keys_st))
        lev_of_st[name] = (np.array([uniq.index(k) for k in keys_st]), uniq)
    out = {}
    for n, R in REPS.items():
        dev = {v: [] for v in views}
        cm = {f'{x}-{y}': {'d_float': [], 'se_float': [], 'd_exact': []} for x, y in cmps}
        for r in range(R):
            rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(ci, n, r)))
            c, y = V.draw_people(law['p'], law['q'], n, rng)
            bf, ifs, bx = {}, {}, {}
            for name in views:
                Vw = law['views'][name]
                ids = Vw['inv'][c]
                bf[name], ifs[name] = E.b_known(ids, y, Vw['post'])
                lev, uniq = lev_of_st[name]
                pl = lev[ids]
                n1g = np.bincount(pl, weights=y, minlength=len(uniq)).astype(np.int64)
                n0g = np.bincount(pl, minlength=len(uniq)).astype(np.int64) - n1g
                bx[name] = V.sample_auc_by_wire(uniq, n1g, n0g)
                dev[name].append(bf[name] - float(bx[name]))
            for x, yv in cmps:
                k = f'{x}-{yv}'
                cm[k]['d_float'].append(bf[x] - bf[yv])
                cm[k]['se_float'].append(E.cluster_se(ifs[x] - ifs[yv], np.arange(n)))
                cm[k]['d_exact'].append(bx[x] - bx[yv])
        o = {'per_view_B_float_minus_B_exact': {v: {'mean': float(np.mean(d)), 'sd': float(np.std(d, ddof=1)),
                                                    'max_abs': float(np.max(np.abs(d))),
                                                    'share_nonzero': float(np.mean(np.abs(np.array(d)) > 1e-12))}
                                                for v, d in dev.items()}, 'comparisons': {}}
        for k, z in cm.items():
            d, se = np.array(z['d_float']), np.array(z['se_float'])
            o['comparisons'][k] = {'d_exact_all_exactly_zero': all(v == 0 for v in z['d_exact']),
                                   'd_float_share_nonzero': float(np.mean(np.abs(d) > 1e-12)),
                                   'd_float_mean': float(d.mean()), 'd_float_sd': float(d.std(ddof=1)),
                                   'd_float_max_abs': float(np.abs(d).max()),
                                   'R2SE_resolved_share_float': float(np.mean(np.abs(d) - 2 * se > 0)),
                                   'R2SE_resolved_count_float': int(np.sum(np.abs(d) - 2 * se > 0)),
                                   'M20_share_float': float(np.mean(np.abs(d) >= 0.020)), 'R': len(d)}
        out[str(n)] = o
    rec['cases'][f'{fam}|{pi}|{occ}'] = out
    print(fam, pi, occ, {n: {k: (v['d_float_sd'], v['R2SE_resolved_share_float'], v['d_exact_all_exactly_zero'])
                             for k, v in o['comparisons'].items()} for n, o in out.items()})
worst = max(v['R2SE_resolved_share_float'] for c in rec['cases'].values() for o in c.values() for v in o['comparisons'].values())
rec['max_R2SE_resolved_share_float_over_cases'] = worst
rec['all_d_exact_zero'] = all(v['d_exact_all_exactly_zero'] for c in rec['cases'].values() for o in c.values()
                              for v in o['comparisons'].values())
changed = any(v['share_nonzero'] > 0 for c in rec['cases'].values() for o in c.values()
              for v in o['per_view_B_float_minus_B_exact'].values())
rec['status'] = 'FAIL' if changed else 'PASS'
rec['status_reason'] = ('Protocol rule "all AUCs score ties as 1/2" is not met by B_same / B_indep in the affected views: exactly '
                        'tied known posteriors are ranked by float rounding, which changes per-sample B values; the exact-tie '
                        'known score gives d = 0 on every sample for these dA = 0 comparisons, the study implementation gives '
                        'the d_float spread recorded above.' if changed else
                        'The float-split ties never changed a sample value of B on these samples.')
V.write_out('CHECK3b_known_score_ties.json', rec, started_utc=T0)
