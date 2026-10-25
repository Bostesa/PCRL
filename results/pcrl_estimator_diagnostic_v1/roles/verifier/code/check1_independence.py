"""CHECK1: independence gives population AUC exactly 1/2 for every view of the IND laws and for any fixed score.

Method (independent of the study's float AUC code):
  * Study IND laws (edg.laws.build arrays): own merging of the emitted vectors, exact masses m1 = p q, m0 = p (1 - q) from the
    exact binary values of the float p, q; population AUC by brute force over all pairs of wire values (Fractions).
    Scores: the optimal (posterior) ranking, random tied integers, random floats, the wire index, the study's float posterior,
    and LOOKUP tables fitted on own synthetic samples (own sampler and seeds; exact Fraction tables, nested a = 1 per fold and
    bank pseudo-count 2). Also: the study's own estimators' fitted-reader population AUCs (d_nested pop_fold_mean / pop_pooled,
    d_bank pop) on the same own samples, and drf.finite_law.auc_of_score of each fixed score on the study's float masses.
  * Own IND laws built from PROTOCOL.md section 3 in exact rationals (pi = 1/2, 1/5; uniform, own-permuted Zipf occupancy):
    the same scores; every value must be EXACTLY 1/2 (Fraction equality).
Why exact 1/2 must hold: under independence P1 = P0 = P, so AUC(f) = sum_{a,b} P(a) P(b) ([f(a) > f(b)] + 1/2 [f(a) = f(b)]),
and swapping (a, b) shows the sum equals 1/2 for every f. P1 == P0 is also checked exactly."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L            # noqa: E402
from edg import estimators as E      # noqa: E402
from drf import finite_law as FL     # noqa: E402

T0 = V.utc_now()
VSEED = 202610251731                 # verifier's own seed root (not a study seed)
HALF = Fr(1, 2)


def lookup_exact(train_w, train_y, K, a):
    """Own exact LOOKUP table: (n1_w + a rbar) / (n_w + a) for seen wires, rbar for unseen (Fractions)."""
    rbar = Fr(int(np.sum(train_y)), len(train_y))
    n = np.bincount(train_w, minlength=K)
    n1 = np.bincount(train_w, weights=train_y, minlength=K).astype(np.int64)
    return [(int(n1[w]) + a * rbar) / (int(n[w]) + a) if n[w] > 0 else rbar for w in range(K)]


def fitted_tables(wid_people, y, K, rng):
    """Own nested (5 folds, a = 1) and bank (FIT parts 0-4, pseudo 2) LOOKUP tables on one own sample."""
    n = len(y)
    folds = rng.permutation(n) % 5
    parts = rng.permutation(n) % 10
    tabs = {f'nested_fold{f}': lookup_exact(wid_people[folds != f], y[folds != f], K, Fr(1)) for f in range(5)}
    fit = parts <= 4
    tabs['bank_fit'] = lookup_exact(wid_people[fit], y[fit], K, Fr(2))
    return tabs, folds, parts


rec = {'check': 1, 'name': 'independence', 'study_laws': {}, 'own_laws': {}}
fails = []

# ------------------------------------------------------------------ study IND laws
for ip, pi in enumerate(L.PIS):
    for io, occ in enumerate(L.OCCS):
        law = L.build('IND', pi, occ)
        key = f'IND|{pi}|{occ}'
        lr = {}
        for iv, (name, Vw) in enumerate(law['views'].items()):
            wid, K, m1, m0 = V.study_law_exact_view(law, name)
            # map own wire id -> study wire id (own merging partition; check 3 verifies it is a bijection)
            own2st = np.full(K, -1)
            own2st[wid] = Vw['inv']
            P1 = [x / sum(m1) for x in m1]
            P0 = [x / sum(m0) for x in m0]
            r = {'K_own_merge': K, 'K_study': Vw['K'], 'P1_equals_P0_exactly': P1 == P0}
            A_ex = V.pop_optimal_auc_exact(m1, m0)
            r['A_exact_on_float_law'] = A_ex
            r['A_study'] = Vw['A']
            r['A_study_minus_half'] = Vw['A'] - 0.5
            rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(1, ip, io, iv)))
            st_scores = {'int_0_3': rng.integers(0, 4, Vw['K']).astype(float), 'uniform_float': rng.random(Vw['K']),
                         'wire_index': np.arange(Vw['K'], dtype=float), 'study_float_posterior': np.asarray(Vw['post'])}
            sc = {}
            for sname, s_st in st_scores.items():
                own_keys = [float(s_st[own2st[w]]) for w in range(K)]
                ex = V.pop_auc_brute(own_keys, m1, m0)
                ex_floatmass = V.pop_auc_brute(list(map(float, s_st)), list(map(float, Vw['m1'])), list(map(float, Vw['m0'])))
                fl = float(FL.auc_of_score(s_st, Vw['m1'], Vw['m0']))
                sc[sname] = {'exact_on_exact_masses': ex, 'exact_equals_half': ex == HALF,
                             'exact_on_study_float_masses_minus_half': float(ex_floatmass - HALF),
                             'drf_auc_of_score_minus_half': fl - 0.5}
                if ex != HALF or abs(fl - 0.5) > 1e-12:
                    fails.append(f'{key} {name} {sname}')
            # fitted scores on own samples (own sampler), own exact tables + the study's estimators' population AUCs
            fit = {}
            for smp in range(2):
                srng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(11, ip, io, iv, smp)))
                c, y = V.draw_people(law['p'], law['q'], 2000, srng)
                tabs, folds, parts = fitted_tables(wid[c], y, K, srng)
                for tname, tab in tabs.items():
                    ex = V.pop_auc_brute(tab, m1, m0)
                    fit[f's{smp}_{tname}_exact'] = ex
                    if ex != HALF:
                        fails.append(f'{key} {name} fitted {tname}')
                ids = Vw['inv'][c]
                dn = E.d_nested(ids, y, folds, Vw['K'], Vw['m1'], Vw['m0'])
                db = E.d_bank(ids, y, parts, Vw['K'], Vw['m1'], Vw['m0'])
                for k_, v_ in (('d_nested_pop_fold_mean', dn['pop_fold_mean']), ('d_nested_pop_pooled', dn['pop_pooled']),
                               ('d_bank_pop', db['pop'])):
                    fit[f's{smp}_{k_}_minus_half'] = v_ - 0.5
                    if abs(v_ - 0.5) > 1e-12:
                        fails.append(f'{key} {name} study {k_}')
            r['fixed_scores'] = sc
            r['fitted_scores'] = fit
            if not (r['P1_equals_P0_exactly'] and A_ex == HALF and abs(Vw['A'] - 0.5) <= 1e-12 and K == Vw['K']):
                fails.append(f'{key} {name} A')
            lr[name] = r
        rec['study_laws'][key] = lr

# ------------------------------------------------------------------ own exact IND laws (PROTOCOL.md section 3)
for ip, pi in enumerate((Fr(1, 2), Fr(1, 5))):
    for io, occ in enumerate(('uniform', 'skewed')):
        own = V.own_law('IND', pi, occ, seed=VSEED + 100 + 10 * ip + io)
        key = f'OWN-IND|{pi}|{occ}'
        lr = {}
        for iv, (name, Vw) in enumerate(own['views'].items()):
            m1, m0, K = Vw['m1'], Vw['m0'], Vw['K']
            P1 = [x / sum(m1) for x in m1]
            P0 = [x / sum(m0) for x in m0]
            r = {'K': K, 'P1_equals_P0_exactly': P1 == P0, 'A_exact': V.pop_optimal_auc_exact(m1, m0)}
            rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(2, ip, io, iv)))
            sc = {}
            for sname, keys in (('int_0_3', [int(x) for x in rng.integers(0, 4, K)]), ('uniform_float', list(rng.random(K))),
                                ('wire_index', list(range(K)))):
                ex = V.pop_auc_brute(keys, m1, m0)
                sc[sname] = ex == HALF
                if ex != HALF:
                    fails.append(f'{key} {name} {sname}')
            pf = [float(x) for x in own['p']]
            qf = [float(x) for x in own['q']]
            for smp in range(2):
                srng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(21, ip, io, iv, smp)))
                c, y = V.draw_people(pf, qf, 2000, srng)
                tabs, _, _ = fitted_tables(Vw['wid'][c], y, K, srng)
                for tname, tab in tabs.items():
                    ex = V.pop_auc_brute(tab, m1, m0)
                    sc[f's{smp}_{tname}'] = ex == HALF
                    if ex != HALF:
                        fails.append(f'{key} {name} fitted {tname}')
            r['scores_exactly_half'] = sc
            if not (r['P1_equals_P0_exactly'] and r['A_exact'] == HALF):
                fails.append(f'{key} {name} A')
            lr[name] = r
        rec['own_laws'][key] = lr
        rec['own_laws'][key]['_supports'] = {k: v['K'] for k, v in own['views'].items()}

rec['failures'] = fails
rec['status'] = 'PASS' if not fails else 'FAIL'
rec['summary'] = ('Every IND view (study laws: exact on the float law values; own laws: exact rationals) has P1 == P0 exactly '
                  'and population AUC exactly 1/2 for the optimal ranking, random tied / untied fixed scores and LOOKUP tables '
                  'fitted on own samples; the study float values (A, drf auc_of_score, d_nested / d_bank fitted-reader pops) are '
                  'within 1e-12 of 1/2.' if not fails else f'{len(fails)} failures')
print(rec['status'], len(fails), fails[:10])
V.write_out('CHECK1_independence.json', rec, started_utc=T0)
