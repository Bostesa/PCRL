"""CHECK2: the known-score sample AUC (edg.estimators.b_known / auc_if) agrees with brute-force enumeration.

Method: small own synthetic samples (own sampler, own seeds) from every study law and view; the study's b_known (known posterior
score V['post'] of the merged wire) and auc_if on several score types (heavily tied integers, untied floats, a constant, the known
posterior) are compared with the exact Fraction AUC obtained by enumerating EVERY (positive, negative) pair of people, ties 1/2.
The influence values returned by auc_if are compared with brute-force DeLong placement values, and the person-unit cluster SE
with sqrt(n / (n - 1) sum IF^2) from those placements. Also: the source AUCs the study cross-checks against (nrm.fa.prc.auc copy
and task_conditioned_representation.inference.auc) on the same scores, and UNDEFINED handling (one class missing)."""
import math
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402
from edg import estimators as E       # noqa: E402
from edg import source_prc as PRC     # noqa: E402
from task_conditioned_representation.inference import auc as INF_AUC   # noqa: E402

T0 = V.utc_now()
VSEED = 202610251740
TOL = 1e-12
rec = {'check': 2, 'name': 'known_score_auc_vs_bruteforce', 'cases': 0, 'max_abs_diff': {}, 'failures': [], 'examples': []}
mx = {'b_known': 0.0, 'auc_if': 0.0, 'prc_auc': 0.0, 'inference_auc': 0.0, 'IF_vs_placements': 0.0, 'cluster_se_rel': 0.0}
n_cases = 0
for fi, fam in enumerate(L.FAMILIES):
    for ip, pi in enumerate(L.PIS):
        for io, occ in enumerate(L.OCCS):
            law = L.build(fam, pi, occ)
            for iv, (name, Vw) in enumerate(law['views'].items()):
                for isz, n in enumerate((20, 75, 300, 1000)):
                    reps = 1 if n == 1000 else 2
                    for smp in range(reps):
                        rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(fi, ip, io, iv, isz, smp)))
                        c, y = V.draw_people(law['p'], law['q'], n, rng)
                        if y.min() == y.max():
                            continue                        # undefined case handled separately below
                        ids = Vw['inv'][c]
                        post = np.asarray(Vw['post'])
                        scores = {'known_posterior': post[ids], 'int_0_2': rng.integers(0, 3, n).astype(float),
                                  'float': rng.random(n), 'constant': np.full(n, 0.7)}
                        bk, bk_if = E.b_known(ids, y, post)
                        ex_b = V.sample_auc_brute(y, score=post[ids])
                        dif = abs(bk - float(ex_b))
                        mx['b_known'] = max(mx['b_known'], dif)
                        if dif > TOL:
                            rec['failures'].append(f'b_known {fam}|{pi}|{occ} {name} n={n}: {bk} vs {float(ex_b)}')
                        for sname, s in scores.items():
                            a, IF = E.auc_if(s, y)
                            ex = V.sample_auc_brute(y, score=s)
                            n_cases += 1
                            for k_, v_ in (('auc_if', a), ('prc_auc', PRC.auc(s, y)), ('inference_auc', INF_AUC(y, s))):
                                d_ = abs(v_ - float(ex))
                                mx[k_] = max(mx[k_], d_)
                                if d_ > TOL:
                                    rec['failures'].append(f'{k_} {fam}|{pi}|{occ} {name} n={n} {sname}: {v_} vs {float(ex)}')
                            if sname == 'constant' and ex != Fr(1, 2):
                                rec['failures'].append('constant score not 1/2')
                            if n <= 300:
                                pl = V.placements_brute(y, s)
                                n1, n0 = int(y.sum()), int(n - y.sum())
                                IFb = np.array([float((pl[i] - ex) / (n1 if y[i] else n0)) for i in range(n)])
                                d_ = float(np.abs(IFb - IF).max())
                                mx['IF_vs_placements'] = max(mx['IF_vs_placements'], d_)
                                if d_ > TOL:
                                    rec['failures'].append(f'IF {fam}|{pi}|{occ} {name} n={n} {sname}: {d_}')
                                se_b = math.sqrt(n / (n - 1) * float(np.sum(IFb ** 2)))
                                se_s = E.cluster_se(IF, np.arange(n))
                                rel = abs(se_s - se_b) / max(se_b, 1e-300) if se_b > 0 else abs(se_s)
                                mx['cluster_se_rel'] = max(mx['cluster_se_rel'], rel)
                                if rel > 1e-9:
                                    rec['failures'].append(f'SE {fam}|{pi}|{occ} {name} n={n} {sname}: {se_s} vs {se_b}')
                        if len(rec['examples']) < 6 and n == 75 and smp == 0:
                            rec['examples'].append({'law': f'{fam}|{pi}|{occ}', 'view': name, 'n': n, 'n1': int(y.sum()),
                                                    'b_known': bk, 'brute_force_exact': ex_b})
# UNDEFINED: one class missing -> study returns NaN (never 1/2); brute force returns None
und = {'auc_if_nan': bool(np.isnan(E.auc_if(np.array([0.1, 0.5, 0.2]), np.array([1, 1, 1]))[0])),
       'b_known_nan': bool(np.isnan(E.b_known(np.array([0, 1]), np.array([0, 0]), np.array([0.3, 0.6]))[0])),
       'bruteforce_none': V.sample_auc_brute(np.array([1, 1, 1]), score=np.array([0.1, 0.5, 0.2])) is None,
       'two_people_same_wire_opposite_class_half': E.b_known(np.array([3, 3]), np.array([1, 0]), np.array([0, 0, 0, 0.4]))[0] == 0.5}
rec['undefined_and_tie'] = und
if not all(und.values()):
    rec['failures'].append(f'undefined/tie handling {und}')
rec['cases'] = n_cases
rec['max_abs_diff'] = mx
rec['status'] = 'PASS' if not rec['failures'] else 'FAIL'
print(rec['status'], n_cases, mx, rec['failures'][:5])
V.write_out('CHECK2_known_score.json', rec, started_utc=T0)
