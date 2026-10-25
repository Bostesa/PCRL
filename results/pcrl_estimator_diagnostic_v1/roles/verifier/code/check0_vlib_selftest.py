"""CHECK0: self-test of the verifier's OWN library (vlib) on hand-computable cases, before it is used to judge the study.
No study code is imported."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

t0 = V.utc_now()
res = {}
# population AUC, hand values (law A / law B of the DRF1 docstring, verified by hand in the verifier's notes):
#   A: m1 = (3, 2), m0 = (2, 3): AUC* = 3/5;  B: m1 = (1, 9), m0 = (0, 10): AUC* = 0.1 * 1 + 0.9 * 1/2 = 11/20
res['lawA_3_5'] = V.pop_optimal_auc_exact([3, 2], [2, 3]) == Fr(3, 5)
res['lawB_11_20'] = V.pop_optimal_auc_exact([1, 9], [0, 10]) == Fr(11, 20)
# XOR-like pure wires: two pure-positive, two pure-negative wires, equal masses -> AUC* = 1
res['pure_wires_1'] = V.pop_optimal_auc_exact([1, 1, 0, 0], [0, 0, 1, 1]) == 1
res['one_wire_half'] = V.pop_optimal_auc_exact([5], [7]) == Fr(1, 2)
# a fixed score that reverses the optimum: AUC = 1 - AUC*
res['reversed_score'] = V.pop_auc_brute([0, 1], [3, 2], [2, 3]) == Fr(2, 5)
# all scores tied -> 1/2 whatever the law
res['all_tied_half'] = V.pop_auc_brute([0, 0, 0], [1, 5, 2], [4, 1, 1]) == Fr(1, 2)
# sample AUC: one positive and one negative with the same score -> 1/2; perfect separation -> 1; reversed -> 0
res['sample_tie_half'] = V.sample_auc_brute([1, 0], score=[0.3, 0.3]) == Fr(1, 2)
res['sample_perfect'] = V.sample_auc_brute([1, 1, 0], score=[2.0, 3.0, 1.0]) == 1
res['sample_reversed'] = V.sample_auc_brute([0, 1], score=[2.0, 1.0]) == 0
res['sample_undefined'] = V.sample_auc_brute([1, 1], score=[1.0, 2.0]) is None
# hand case: y = (1,1,0,0), s = (2,1,1,0): pairs (2>1), (2>0), (1=1 -> 1/2), (1>0) -> 3.5 / 4 = 7/8
res['sample_hand_7_8'] = V.sample_auc_brute([1, 1, 0, 0], score=[2.0, 1.0, 1.0, 0.0]) == Fr(7, 8)
# wire-pair route equals person-pair route on random tied samples
rng = np.random.default_rng(20261025990)
ok = True
for _ in range(200):
    k = int(rng.integers(1, 9))
    w = rng.integers(0, k, int(rng.integers(2, 60)))
    y = rng.integers(0, 2, len(w))
    keys = rng.integers(0, 3, k).astype(float)
    n1w = np.bincount(w, weights=y, minlength=k).astype(int)
    n0w = np.bincount(w, minlength=k) - n1w
    a = V.sample_auc_by_wire(keys, n1w, n0w)
    b = V.sample_auc_brute(y, score=keys[w])
    ok &= (a == b)
res['wire_pair_equals_person_pair_200_random'] = bool(ok)
# placements: mean of V1 over positives equals the AUC; mean of V0 over negatives equals the AUC
y = np.array([1, 0, 1, 0, 1, 1, 0])
s = np.array([0.2, 0.2, 0.9, 0.1, 0.5, 0.5, 0.7])
pl = V.placements_brute(y, s)
auc = V.sample_auc_brute(y, score=s)
res['placements_average_to_auc'] = (sum(p for p, t in zip(pl, y) if t) / 4 == auc) and (sum(p for p, t in zip(pl, y) if not t) / 3 == auc)
# enumeration maximum equals the empirical-posterior ranking AUC on random tiny count tables
ok = True
for _ in range(60):
    k = int(rng.integers(1, 6))
    n1w = [int(x) for x in rng.integers(0, 4, k)]
    n0w = [int(x) for x in rng.integers(0, 4, k)]
    if sum(n1w) == 0 or sum(n0w) == 0:
        continue
    keys = [Fr(a, a + b) if a + b else Fr(-1) for a, b in zip(n1w, n0w)]
    ok &= V.max_auc_by_enumeration(n1w, n0w) == V.sample_auc_by_wire(keys, n1w, n0w)
res['enumeration_max_equals_posterior_ranking'] = bool(ok)
# merging: identical rows merge, -0.0 == +0.0, pair needs both components equal
a = np.array([[0.1, 0.9], [0.1, 0.9], [0.0, 1.0], [-0.0, 1.0]])
b = np.array([[1.0], [2.0], [3.0], [3.0]])
res['merge_single'] = V.merge_own(a)[1] == 2
res['merge_pair'] = V.merge_own((a, b))[1] == 3
# own coarse signal: marginal equals pi to 1e-50; 8 distinct q levels shared by atom pairs (2k, 2k + 1)
w, z, q, alpha, err = V.own_coarse_signal(Fr(1, 5), 'skewed', np.random.default_rng(1))
res['own_coarse_marginal_err_lt_1e-50'] = abs(err) < Fr(1, 10 ** 50)
res['own_coarse_8_levels'] = len(set(q)) == 8 and all(q[2 * k] == q[2 * k + 1] for k in range(8))
res['all_pass'] = bool(all(v is True or v == True for v in res.values()))  # noqa: E712
print(res)
V.write_out('CHECK0_vlib_selftest.json', {'check': 0, 'what': 'verifier own-library self-test on hand-computable cases',
                                          'results': {k: bool(v) for k, v in res.items()}}, started_utc=t0)
