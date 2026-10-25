"""CHECK4: irrelevant refinement preserves the population optimum (REF laws).

Method (own merging, exact masses, brute-force exact AUC):
  * 4 study REF laws: A at L16, L64, L256, L1024 computed exactly (own code); they must be IDENTICAL Fractions. The level
    partitions must be nested (every level-l wire lies inside one level-(l-1) wire, 4 children per parent), and every child's
    exact posterior must EQUAL its parent's (S independent of the child given the parent). The study's float A values must agree
    across levels and with the exact value within the protocol's 1e-12; the study's float posteriors V['post'] of children vs
    parents are reported (bitwise-equal share and max deviation).
  * 4 own REF laws built from PROTOCOL.md section 3 in exact rationals (Decimal-60 sigmoid, own permutations): the same exact
    identities. For uniform occupancy the law has no free permutation, so the own A must equal the study's A; for skewed occupancy
    the study's atom permutation is recovered from its Zipf weights and the coarse law rebuilt exactly (own alpha / q) to
    cross-check the study's A value."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402

T0 = V.utc_now()
VSEED = 202610251815
LEVELS = ('L16', 'L64', 'L256', 'L1024')
rec = {'check': 4, 'name': 'irrelevant_refinement', 'study_laws': {}, 'own_laws': {}}
fails = []


def tree_and_posteriors(level_wid, level_m, label):
    """level_wid[name]: own wire id per internal cell; level_m[name]: (m1, m0). Nested-partition and posterior-inheritance checks."""
    out = {}
    for lp, lc in zip(LEVELS[:-1], LEVELS[1:]):
        wp, wc = level_wid[lp], level_wid[lc]
        parent_of = {}
        nested = True
        for c in range(len(wc)):
            if parent_of.setdefault(int(wc[c]), int(wp[c])) != int(wp[c]):
                nested = False
        kids = np.bincount(np.array(list(parent_of.values())), minlength=int(wp.max()) + 1)
        m1p, m0p = level_m[lp]
        m1c, m0c = level_m[lc]
        pp = V.posterior_exact(m1p, m0p)
        pc = V.posterior_exact(m1c, m0c)
        inherit = all(pc[ch] == pp[par] for ch, par in parent_of.items())
        # child masses add up to the parent's (merging is a partition of the same people)
        adds = all(sum(m1c[ch] for ch, par in parent_of.items() if par == P) == m1p[P] for P in range(0, len(m1p), max(1, len(m1p) // 4)))
        out[f'{lc}_in_{lp}'] = {'nested_partition': nested, 'children_per_parent': sorted(set(kids.tolist())),
                                'every_child_posterior_equals_parent_exactly': inherit, 'masses_add_spotcheck': adds}
        if not (nested and set(kids.tolist()) == {4} and inherit and adds):
            fails.append(f'{label} {lc} in {lp}')
    return out


for pi in L.PIS:
    for occ in L.OCCS:
        law = L.build('REF', pi, occ)
        key = f'REF|{pi}|{occ}'
        wids, ms, A = {}, {}, {}
        r = {}
        for name in LEVELS:
            wid, K, m1, m0 = V.study_law_exact_view(law, name)
            wids[name], ms[name] = wid, (m1, m0)
            A[name] = V.pop_optimal_auc_exact(m1, m0)
        r['A_exact'] = {k: v for k, v in A.items()}
        r['A_exact_identical_all_levels'] = len(set(A.values())) == 1
        As = {k: law['views'][k]['A'] for k in LEVELS}
        r['A_study_float'] = As
        r['A_study_spread_across_levels'] = max(As.values()) - min(As.values())
        r['A_study_max_abs_minus_exact'] = max(abs(As[k] - float(A[k])) for k in LEVELS)
        r['tree'] = tree_and_posteriors(wids, ms, key)
        # the study's float posteriors: child vs parent
        fl = {}
        for lp, lc in zip(LEVELS[:-1], LEVELS[1:]):
            ip, ic = law['views'][lp]['inv'], law['views'][lc]['inv']
            pp, pc = np.asarray(law['views'][lp]['post']), np.asarray(law['views'][lc]['post'])
            fl[f'{lc}_vs_{lp}'] = {'share_cells_bitwise_equal': float(np.mean(pp[ip] == pc[ic])),
                                   'max_abs_dev': float(np.abs(pp[ip] - pc[ic]).max())}
        r['study_float_post_child_vs_parent'] = fl
        if not (r['A_exact_identical_all_levels'] and r['A_study_spread_across_levels'] <= 1e-12
                and r['A_study_max_abs_minus_exact'] <= 1e-12):
            fails.append(f'{key} A')
        # exact rebuild of the study's coarse law from the protocol (recovered Zipf permutation, own 60-digit alpha / q)
        w_st = np.asarray(law['ref_struct']['w'])
        H = sum(Fr(1, i) for i in range(1, 17))
        if occ == 'uniform':
            w_ex = [Fr(1, 16)] * 16
        else:
            ks = [int(round(1.0 / (x * float(H)))) for x in w_st]
            w_ex = [Fr(1, k) / H for k in ks]
            r['recovered_zipf_ranks'] = ks
        r['max_abs_w_exact_minus_study'] = max(abs(float(a) - b) for a, b in zip(w_ex, w_st))
        z = [Fr(-1) + Fr(2 * (g // 2), 7) for g in range(16)]
        from decimal import Decimal
        Wd = [V._dec(x) for x in w_ex]
        Zd = [V._dec(x) for x in z]
        Pd = V._dec(Fr(pi).limit_denominator(1000))
        lo, hi = Decimal(-30), Decimal(30)
        for _ in range(230):
            mid = (lo + hi) / 2
            if sum(wi * V._dsig(mid + zi) for wi, zi in zip(Wd, Zd)) < Pd:
                lo = mid
            else:
                hi = mid
        q_ex = [Fr(V._dsig((lo + hi) / 2 + zi)) for zi in Zd]
        A_rebuilt = V.pop_optimal_auc_exact([a * b for a, b in zip(w_ex, q_ex)], [a * (1 - b) for a, b in zip(w_ex, q_ex)])
        r['A_rebuilt_from_protocol'] = A_rebuilt
        r['A_study_minus_rebuilt'] = As['L16'] - float(A_rebuilt)
        r['max_abs_q_study_minus_rebuilt'] = max(abs(float(a) - b) for a, b in zip(q_ex, law['ref_struct']['q_coarse']))
        if abs(r['A_study_minus_rebuilt']) > 1e-12 or r['max_abs_w_exact_minus_study'] > 1e-15:
            fails.append(f'{key} rebuilt')
        rec['study_laws'][key] = r

for ip, pi in enumerate((Fr(1, 2), Fr(1, 5))):
    for io, occ in enumerate(('uniform', 'skewed')):
        own = V.own_law('REF', pi, occ, seed=VSEED + 10 * ip + io)
        key = f'OWN-REF|{pi}|{occ}'
        wids = {k: own['views'][k]['wid'] for k in LEVELS}
        ms = {k: (own['views'][k]['m1'], own['views'][k]['m0']) for k in LEVELS}
        A = {k: V.pop_optimal_auc_exact(*ms[k]) for k in LEVELS}
        r = {'supports': {k: own['views'][k]['K'] for k in LEVELS}, 'A_exact': A,
             'A_exact_identical_all_levels': len(set(A.values())) == 1, 'marginal_error_abs': float(abs(own['marginal_error'])),
             'tree': tree_and_posteriors(wids, ms, key)}
        if occ == 'uniform':
            st = L.build('REF', float(pi), 'uniform')['views']['L16']['A']
            r['study_A_same_uniform_law'] = st
            r['own_minus_study'] = float(A['L16']) - st
            if abs(r['own_minus_study']) > 1e-12:
                fails.append(f'{key} vs study uniform')
        if not r['A_exact_identical_all_levels'] or r['supports'] != {'L16': 16, 'L64': 64, 'L256': 256, 'L1024': 1024}:
            fails.append(f'{key} A')
        rec['own_laws'][key] = r

rng_A = [float(v['A_exact']['L16']) for v in rec['study_laws'].values()]
rec['study_A_range'] = [min(rng_A), max(rng_A)]
rec['protocol_range_0.6708_0.6814_ok'] = round(min(rng_A), 4) == 0.6708 and round(max(rng_A), 4) == 0.6814
if not rec['protocol_range_0.6708_0.6814_ok']:
    fails.append('protocol range')
rec['failures'] = fails
rec['status'] = 'PASS' if not fails else 'FAIL'
print(rec['status'], fails, rec['study_A_range'])
for k, v in rec['study_laws'].items():
    print(k, v['A_exact_identical_all_levels'], v['A_study_spread_across_levels'], v['A_study_max_abs_minus_exact'],
          v['A_study_minus_rebuilt'], v['study_float_post_child_vs_parent'])
for k, v in rec['own_laws'].items():
    print(k, v['A_exact_identical_all_levels'], float(v['A_exact']['L16']), v.get('own_minus_study'),
          {t: (x['nested_partition'], x['every_child_posterior_equals_parent_exactly']) for t, x in v['tree'].items()})
V.write_out('CHECK4_refinement.json', rec, started_utc=T0)
