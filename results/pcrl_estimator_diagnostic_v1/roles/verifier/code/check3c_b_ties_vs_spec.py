"""CHECK3 (part 3, supplementary coverage): the B known score against the PROTOCOL.md section 3 SPECIFICATION, for every dA = 0
comparison of all 16 laws.

Under the specification the known posterior of every person is the same in both views of each null comparison (IND: every
posterior is pi; REF: the coarse atom's q at every level; RED pair vs income: S independent of occupation given income; COMP
occupation vs income: every single-view posterior is pi), so the exact-tie known score gives B(x) - B(y) = 0 on EVERY sample (checks
1, 4 and 5 verified these relations exactly). This run measures what the study's b_known (float posterior V['post']) gives on own
samples: whether d is identically 0, its SD, the R2SE resolution rate with the study's paired SE, and the M20 rate. Part 2
(check3b) compared with the float law's exact ties; this part also covers the COMP and RED views whose float law deviates from the
specification's ties by ~1e-17 (u, v centred in float; p = w e products rounded)."""
import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402
from edg import estimators as E       # noqa: E402

T0 = V.utc_now()
VSEED = 202610251915
REPS = {2000: 60, 20000: 20}
NULLS = {'IND': [('pair', 'income'), ('pair', 'occupation'), ('occupation', 'income')],
         'REF': [('L64', 'L16'), ('L256', 'L16'), ('L1024', 'L16')], 'RED': [('pair', 'income')], 'COMP': [('occupation', 'income')]}
rec = {'check': 3, 'name': 'B_known_score_vs_specification_ties_all_null_comparisons', 'reps': REPS, 'cells': {}}
spec_distinct = {}
for fi, fam in enumerate(L.FAMILIES):
    for ip, pi in enumerate(L.PIS):
        for io, occ in enumerate(L.OCCS):
            law = L.build(fam, pi, occ)
            key = f'{fam}|{pi}|{occ}'
            spec = {'IND': {'income': 1, 'occupation': 1, 'pair': 1}, 'REF': {k: 8 for k in law['views']},
                    'RED': {'income': 8, 'occupation': 4, 'pair': 8}}.get(fam)
            if fam == 'COMP':
                spec = {'income': 1, 'occupation': 1, 'pair': None}
            spec_distinct[key] = {v: {'spec_distinct_posteriors': spec[v], 'float_V_post_distinct': int(len(np.unique(law['views'][v]['post'])))}
                                  for v in law['views']}
            out = {}
            for n, R in REPS.items():
                acc = {f'{x}-{y}': {'d': [], 'se': []} for x, y in NULLS[fam]}
                for r in range(R):
                    rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(fi, ip, io, n, r)))
                    c, y = V.draw_people(law['p'], law['q'], n, rng)
                    if y.min() == y.max():
                        continue
                    b = {v: E.b_known(law['views'][v]['inv'][c], y, law['views'][v]['post']) for v in {a for p in NULLS[fam] for a in p}}
                    for x, yv in NULLS[fam]:
                        acc[f'{x}-{yv}']['d'].append(b[x][0] - b[yv][0])
                        acc[f'{x}-{yv}']['se'].append(E.cluster_se(b[x][1] - b[yv][1], np.arange(n)))
                o = {}
                for k, z in acc.items():
                    d, se = np.array(z['d']), np.array(z['se'])
                    o[k] = {'R': len(d), 'd_identically_zero': bool(np.all(d == 0)), 'share_d_nonzero': float(np.mean(d != 0)),
                            'd_sd': float(d.std(ddof=1)), 'd_max_abs': float(np.abs(d).max()),
                            'R2SE_resolved': int(np.sum(np.abs(d) - 2 * se > 0)), 'M20': int(np.sum(np.abs(d) >= 0.020))}
                out[str(n)] = o
            rec['cells'][key] = out
rec['spec_vs_float_distinct_posteriors'] = spec_distinct
aff = sorted({f'{c}|{k}' for c, v in rec['cells'].items() for o in v.values() for k, z in o.items() if not z['d_identically_zero']})
rec['null_comparisons_where_B_d_is_not_identically_zero'] = aff
rec['n_null_comparisons_total'] = sum(len(NULLS[k.split('|')[0]]) for k in rec['cells'])
rec['status'] = 'FAIL' if aff else 'PASS'
for c, v in rec['cells'].items():
    for n, o in v.items():
        for k, z in o.items():
            if not z['d_identically_zero']:
                print(c, n, k, 'sd', round(z['d_sd'], 5), 'max', round(z['d_max_abs'], 5), 'R2SE', z['R2SE_resolved'], '/', z['R'], 'M20', z['M20'])
print(rec['status'], len(aff), 'of', rec['n_null_comparisons_total'])
V.write_out('CHECK3c_b_ties_vs_spec.json', rec, started_utc=T0)
