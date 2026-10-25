"""CHECK5: redundant (RED) and complementary (COMP) views behave as specified; the null / non-null classification of every
registered comparison is correct.

Method (own merging, exact masses, brute-force exact AUC):
  RED (4 study laws + 4 own exact laws): A(pair) == A(income) exactly; A(income) > A(occupation); S independent of occupation
      given income, checked as: the exact posterior of every pair wire equals the exact posterior of its income wire; the occupation
      wire's exact posterior equals the coarse-group average posterior.
  COMP (4 study + 4 own): A(income) == A(occupation) == 1/2 (exactly for own laws; for the study's float law the deviation from
      1/2 is reported - its u, v are centred in float, so the exact value of the float law is 1/2 only to ~1e-16) < A(pair).
  Uniform-occupancy own laws have no free permutation, so their exact A values must equal the study's.
  All 16 study laws: dA of every registered comparison from the study's float A values, classified with the analysis's null
  tolerance (|dA| <= 1e-12), against the protocol's stated relation (section 3, Comparisons)."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402

T0 = V.utc_now()
VSEED = 202610251825
rec = {'check': 5, 'name': 'redundant_and_complementary_views', 'study_laws': {}, 'own_laws': {}, 'comparison_classification': {}}
fails = []
PROTOCOL_DA = {'IND': {'pair-income': 0, 'pair-occupation': 0, 'occupation-income': 0},
               'REF': {'L64-L16': 0, 'L256-L16': 0, 'L1024-L16': 0},
               'RED': {'pair-income': 0, 'income-occupation': 1, 'pair-occupation': 1},
               'COMP': {'pair-income': 1, 'pair-occupation': 1, 'occupation-income': 0}}


def red_relations(views_exact, cells_pair_wid, cells_inc_wid, label):
    """views_exact[name] = (m1, m0); conditional-independence check through the internal cells' wire ids."""
    A = {k: V.pop_optimal_auc_exact(*v) for k, v in views_exact.items()}
    pp = V.posterior_exact(*views_exact['pair'])
    pinc = V.posterior_exact(*views_exact['income'])
    ci = all(pp[int(a)] == pinc[int(b)] for a, b in zip(cells_pair_wid, cells_inc_wid))
    r = {'A': A, 'A_pair_equals_A_income_exactly': A['pair'] == A['income'], 'A_income_gt_A_occupation': A['income'] > A['occupation'],
         'pair_posterior_equals_income_posterior_exactly': ci, 'A_income_minus_A_occupation': float(A['income'] - A['occupation'])}
    if not (r['A_pair_equals_A_income_exactly'] and r['A_income_gt_A_occupation'] and ci):
        fails.append(f'{label} RED relations')
    return r


def comp_relations(views_exact, label, exact_half_required):
    A = {k: V.pop_optimal_auc_exact(*v) for k, v in views_exact.items()}
    half = Fr(1, 2)
    r = {'A': A, 'A_income_minus_half': float(A['income'] - half), 'A_occupation_minus_half': float(A['occupation'] - half),
         'A_pair_gt_half': A['pair'] > half}
    if exact_half_required:
        r['single_views_exactly_half'] = A['income'] == half and A['occupation'] == half
        ok = r['single_views_exactly_half'] and r['A_pair_gt_half']
    else:
        ok = abs(r['A_income_minus_half']) <= 1e-12 and abs(r['A_occupation_minus_half']) <= 1e-12 and r['A_pair_gt_half']
    if not ok:
        fails.append(f'{label} COMP relations')
    return r


for pi in L.PIS:
    for occ in L.OCCS:
        for fam in ('RED', 'COMP'):
            law = L.build(fam, pi, occ)
            key = f'{fam}|{pi}|{occ}'
            ve, wid = {}, {}
            for name in ('income', 'occupation', 'pair'):
                w_, K, m1, m0 = V.study_law_exact_view(law, name)
                ve[name], wid[name] = (m1, m0), w_
            if fam == 'RED':
                r = red_relations(ve, wid['pair'], wid['income'], key)
                # occupation wire posterior = average posterior of its coarse group g // 4 (exact on the float law)
                gg = np.repeat(np.arange(16), 16)
                po = V.posterior_exact(*ve['occupation'])
                grp = {}
                for c in range(len(law['p'])):
                    h = int(gg[c] // 4)
                    a, b = grp.get(h, (Fr(0), Fr(0)))
                    grp[h] = (a + V.F(law['p'][c]) * V.F(law['q'][c]), b + V.F(law['p'][c]))
                dev = max(abs(float(po[int(wid['occupation'][c])] - grp[int(gg[c] // 4)][0] / grp[int(gg[c] // 4)][1]))
                          for c in range(len(law['p'])))
                r['occupation_posterior_minus_group_average_max_abs'] = dev
                if dev > 1e-12:
                    fails.append(f'{key} occupation posterior')
            else:
                r = comp_relations(ve, key, exact_half_required=False)
                a, b, u, v = (np.asarray(law['comp'][k]) for k in ('a', 'b', 'u', 'v'))
                r['float_weighted_mean_u'] = float(sum(V.F(x) * V.F(y) for x, y in zip(a, u)))
                r['float_weighted_mean_v'] = float(sum(V.F(x) * V.F(y) for x, y in zip(b, v)))
                r['delta'] = law['comp']['delta']
            r['A_study'] = {k: law['views'][k]['A'] for k in ('income', 'occupation', 'pair')}
            r['A_study_max_abs_minus_exact'] = max(abs(law['views'][k]['A'] - float(r['A'][k])) for k in r['A'])
            if r['A_study_max_abs_minus_exact'] > 1e-12:
                fails.append(f'{key} study A vs exact')
            rec['study_laws'][key] = r

for ip, pi in enumerate((Fr(1, 2), Fr(1, 5))):
    for io, occ in enumerate(('uniform', 'skewed')):
        for fam in ('RED', 'COMP'):
            own = V.own_law(fam, pi, occ, seed=VSEED + 100 * (fam == 'COMP') + 10 * ip + io)
            key = f'OWN-{fam}|{pi}|{occ}'
            ve = {k: (own['views'][k]['m1'], own['views'][k]['m0']) for k in ('income', 'occupation', 'pair')}
            if fam == 'RED':
                r = red_relations(ve, own['views']['pair']['wid'], own['views']['income']['wid'], key)
            else:
                r = comp_relations(ve, key, exact_half_required=True)
            r['supports'] = {k: own['views'][k]['K'] for k in ('income', 'occupation', 'pair')}
            exp = {'RED': {'income': 16, 'occupation': 64, 'pair': 256}, 'COMP': {'income': 16, 'occupation': 15, 'pair': 240}}[fam]
            if r['supports'] != exp:
                fails.append(f'{key} supports')
            if occ == 'uniform':
                st = L.build(fam, float(pi), 'uniform')
                r['own_minus_study_A'] = {k: float(r['A'][k]) - st['views'][k]['A'] for k in ('income', 'occupation', 'pair')}
                if max(abs(x) for x in r['own_minus_study_A'].values()) > 1e-12:
                    fails.append(f'{key} own vs study uniform A')
            rec['own_laws'][key] = r

# classification of every registered comparison (study float A, analysis tolerance 1e-12) vs the protocol's relation
for fam in L.FAMILIES:
    for pi in L.PIS:
        for occ in L.OCCS:
            law = L.build(fam, pi, occ)
            for x, y in law['comparisons']:
                dA = law['views'][x]['A'] - law['views'][y]['A']
                null = abs(dA) <= 1e-12
                exp = PROTOCOL_DA[fam][f'{x}-{y}']
                ok = (null and exp == 0) or ((not null) and exp == 1 and dA > 0)
                rec['comparison_classification'][f'{fam}|{pi}|{occ}|{x}-{y}'] = {'dA_study_float': dA, 'null_by_1e-12': null,
                                                                                'protocol_relation': 'dA = 0' if exp == 0 else 'dA > 0',
                                                                                'ok': ok}
                if not ok:
                    fails.append(f'classification {fam}|{pi}|{occ}|{x}-{y}')
red_A = [float(v['A']['income']) for k, v in rec['study_laws'].items() if k.startswith('RED')]
comp_A = [float(v['A']['pair']) for k, v in rec['study_laws'].items() if k.startswith('COMP')]
rec['RED_A_pair_range'] = [min(red_A), max(red_A)]
rec['COMP_A_pair_range'] = [min(comp_A), max(comp_A)]
rec['protocol_ranges_ok'] = (round(min(red_A), 3) == 0.642 and round(max(red_A), 3) == 0.677 and round(min(comp_A), 3) == 0.563
                             and round(max(comp_A), 3) == 0.604)
if not rec['protocol_ranges_ok']:
    fails.append('protocol ranges')
rec['min_abs_dA_nonnull'] = min(abs(v['dA_study_float']) for v in rec['comparison_classification'].values() if not v['null_by_1e-12'])
rec['max_abs_dA_null'] = max(abs(v['dA_study_float']) for v in rec['comparison_classification'].values() if v['null_by_1e-12'])
rec['failures'] = fails
rec['status'] = 'PASS' if not fails else 'FAIL'
print(rec['status'], fails, rec['RED_A_pair_range'], rec['COMP_A_pair_range'], rec['min_abs_dA_nonnull'], rec['max_abs_dA_null'])
for k, v in list(rec['study_laws'].items()) + list(rec['own_laws'].items()):
    print(k, {a: round(float(b), 6) for a, b in v['A'].items()}, v.get('own_minus_study_A'), v.get('A_income_minus_half'),
          v.get('A_occupation_minus_half'), v.get('float_weighted_mean_u'))
V.write_out('CHECK5_red_comp.json', rec, started_utc=T0)
