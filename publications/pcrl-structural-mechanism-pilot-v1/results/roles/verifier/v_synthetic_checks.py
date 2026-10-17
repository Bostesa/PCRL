#!/usr/bin/env python3
"""Verifier, independent code: brute-force checks of structural_pilot/partition.py (merge identities, lazy-heap greedy)
and of the frozen lookup reader in structural_pilot/readers.py. Invented data only: no real data, no .npz, no .private.

Independence from the repository's own test (test_partition_synthetic.py): that test computes its NLL and SEX-information
references with partition.kl_rows / kl_bern (the code under test) and its Brier reference through the within-cell-variance
identity itself. Here every reference is computed from first principles in plain Python floats:
  - expected Brier / NLL / accuracy of the pooled release under the plug-in law y ~ teacher posterior, by enumerating y;
  - SEX variance as the between-cell variance of the cell-mean propensity;
  - SEX information as the plug-in mutual information of the joint table P(S = s, C = c) (not the KL_B merge form);
  - the within-SEX guard by summing ||release - e_label||^2 person by person.
The greedy engine is compared with a naive exhaustive greedy that rescores every candidate pair from scratch at every step.

Run from the worktree root:
  <repository>/.venv/bin/python -B results/pcrl_structural_mechanism_pilot_v1/roles/verifier/v_synthetic_checks.py
"""
from __future__ import annotations

import json
import math
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from structural_pilot import partition as P                                        # noqa: E402  (code under test)
from structural_pilot.readers import fit_lookup, lookup_keys, predict_lookup        # noqa: E402  (code under test)

TOL = 1e-12


# ----------------------------------------------------------------------------------------------- brute-force references
def pooled(assign, t):
    sums, cnt = defaultdict(lambda: [0.0] * len(t[0])), Counter()
    for c, row in zip(assign, t):
        s = sums[c]
        for i, v in enumerate(row):
            s[i] += v
        cnt[c] += 1
    return {c: [v / cnt[c] for v in s] for c, s in sums.items()}


def exp_brier(truth, r):
    k = len(truth)
    return sum(truth[y] * sum((r[c] - (1.0 if c == y else 0.0)) ** 2 for c in range(k)) for y in range(k) if truth[y] > 0)


def exp_nll(truth, r):
    return -sum(truth[y] * math.log(r[y]) for y in range(len(truth)) if truth[y] > 0)


def argmax_first(v):
    m = max(v)
    return next(i for i, x in enumerate(v) if x == m)


def metrics(assign, t1, t2, pi, sex, occ):
    """Plug-in metrics of the pooled-posterior release (teacher posteriors as truth; guard against labels)."""
    n_all = len(assign)
    out = {}
    for name, t in (('1', t1), ('2', t2)):
        rel = pooled(assign, t)
        b = nl = a = 0.0
        for c, row in zip(assign, t):
            r = rel[c]
            b += exp_brier(row, r) - exp_brier(row, row)
            nl += exp_nll(row, r) - exp_nll(row, row)
            a += row[argmax_first(row)] - row[argmax_first(r)]
        out['brier' + name], out['nll' + name], out['acc' + name] = b / n_all, nl / n_all, a / n_all
    cnt = Counter(assign)
    s1 = defaultdict(float)
    for c, p in zip(assign, pi):
        s1[c] += p
    pbar = sum(pi) / n_all
    out['sexvar'] = sum(cnt[c] / n_all * (s1[c] / cnt[c] - pbar) ** 2 for c in cnt)
    info = 0.0
    for c in cnt:
        pc = cnt[c] / n_all
        for ps, joint in ((pbar, s1[c] / n_all), (1.0 - pbar, (cnt[c] - s1[c]) / n_all)):
            if joint > 0:
                info += joint * math.log(joint / (ps * pc))
    out['sexinfo'] = info
    rel2 = pooled(assign, t2)
    for s in (0, 1):
        members = [j for j in range(n_all) if sex[j] == s]
        out[f'guard{s}'] = sum(sum((rel2[assign[j]][c] - (1.0 if c == occ[j] else 0.0)) ** 2 for c in range(len(t2[0])))
                               for j in members) / len(members)
    return out


INCREASE = ('brier1', 'brier2', 'nll1', 'nll2', 'acc1', 'acc2', 'guard0', 'guard1')
DECREASE = ('sexvar', 'sexinfo')


def deltas(before, after):
    d = {k: after[k] - before[k] for k in INCREASE}
    d.update({k: before[k] - after[k] for k in DECREASE})
    return d


# ------------------------------------------------------------------------------------------------------- invented data
def invent(seed, n, k_target, edge_cases=True):
    rng = np.random.default_rng(seed)
    t1 = rng.dirichlet([1.5, 1.5], n)
    t2 = rng.dirichlet([0.7] * 5, n)
    pi = rng.beta(1.2, 1.2, n)
    if edge_cases:
        idx = rng.choice(n, n // 8, replace=False)
        t2[idx, rng.integers(0, 5, len(idx))] = 0.0                       # exact zeros in posteriors
        t2[idx] /= t2[idx].sum(1, keepdims=True)
        t2[idx[:3]] = np.eye(5)[[0, 2, 4]]                                 # one-hot posteriors
        t1[idx[3:6]] = [0.5, 0.5]                                          # exact argmax ties
        pi[rng.choice(n, n // 15, replace=False)] = 0.0                    # propensities at the boundary
        pi[rng.choice(n, n // 15, replace=False)] = 1.0
    sex = (rng.random(n) < pi).astype(np.int64)
    sex[:2] = (0, 1)                                                       # both groups present
    occ = np.array([rng.choice(5, p=row / row.sum()) for row in t2], dtype=np.int64)
    w = rng.dirichlet(np.full(k_target, 0.6))
    raw = rng.choice(k_target, n, p=w)
    raw[-3:] = k_target + np.arange(3)                                     # forced singletons
    _, assign = np.unique(raw, return_inverse=True)                        # contiguous labels, every cell non-empty
    return assign.astype(np.int64), int(assign.max() + 1), t1, t2, pi, sex, occ


def to_lists(*arrs):
    return [a.tolist() for a in arrs]


# ------------------------------------------------------------------------------------------------ 1. merge identities
def check_identities():
    worst = Counter()
    n_pairs = 0
    for seed in (11, 12, 13, 14):
        assign, k, t1, t2, pi, sex, occ = invent(seed, 200, 18)
        C = P.Cells(assign, k, t1, t2, pi, {'sex': sex, 'occ': occ})
        L = to_lists(assign, t1, t2, pi, sex, occ)
        base = metrics(*L)
        for a in range(k):
            for b in range(a + 1, k):
                d_eng = {kk: float(v[0]) for kk, v in P.merge_delta(C, a, b).items()}
                merged = [a if x == b else x for x in L[0]]
                d_ref = deltas(base, metrics(merged, *L[1:]))
                for kk in INCREASE + DECREASE:
                    worst[kk] = max(worst[kk], abs(d_eng[kk] - d_ref[kk]))
                n_pairs += 1
    return {'pairs_checked': n_pairs, 'datasets': 4, 'max_abs_error': dict(worst),
            'pass': all(v < TOL for v in worst.values())}


# ----------------------------------------------------------------------------------- 2. greedy engine vs naive greedy
def naive_greedy(L, k0, edges, score_fn, admissible_fn, stop_count=None):
    """Exhaustive greedy: at every step rescore every adjacent pair of live clusters from scratch (brute force), keep the
    admissible ones under the CURRENT spent budget, merge the minimum (score, a, b). Clusters are labelled by their
    smallest initial cell; adjacency is inherited from the initial edges."""
    assign0 = L[0]
    label = list(range(k0))
    spent = defaultdict(float)
    cur = metrics(*L)
    steps, gaps = 0, []
    while True:
        live = set(label)
        if stop_count is not None and len(live) <= stop_count:
            return label, dict(spent), steps, 'stop_count', gaps
        cand = sorted({(min(label[i], label[j]), max(label[i], label[j])) for i, j in edges if label[i] != label[j]})
        scored = []
        for A, B in cand:
            new = [A if x == B else x for x in label]
            after = metrics([new[c] for c in assign0], *L[1:])
            d = deltas(cur, after)
            if admissible_fn(d, spent):
                scored.append(((score_fn(d), A, B), d, new, after))
        if not scored:
            return label, dict(spent), steps, 'exhausted', gaps
        scored.sort(key=lambda z: z[0])
        if len(scored) > 1:
            gaps.append(scored[1][0][0] - scored[0][0][0])
        _, d, label, cur = scored[0]
        for kk, v in d.items():
            spent[kk] += v
        steps += 1


def engine_greedy(assign, k, t1, t2, pi, sex, occ, edges, score_fn, admissible_fn, stop_count=None):
    C = P.Cells(assign, k, t1, t2, pi, {'sex': sex, 'occ': occ})
    return P.greedy(C, edges, score_fn, admissible_fn, stop_count=stop_count)


def sparse_edges(k, rng):
    e = {(i, i + 1) for i in range(k - 1)}
    for i in range(k):
        for j in rng.choice(k, 2, replace=False).tolist():
            if i != j:
                e.add((min(i, j), max(i, j)))
    return sorted(e)


def scar_score(d):
    return d['brier1'] / 0.005 + d['nll1'] / 0.01 + d['brier2'] / 0.005 + d['nll2'] / 0.01 - 40.0 * d['sexinfo']


def check_greedy():
    res = {}
    # (a) all pairs, monotone budgets (non-negative costs), stop at a count
    # (b) sparse graph with inherited edges, no budget
    # (c) tight budgets so the path is exhausted before the stop count
    scenarios = {
        'a_all_pairs_budgeted': (lambda k, rng: [(i, j) for i in range(k) for j in range(i + 1, k)],
                                 lambda d, s: s.get('brier1', 0) + d['brier1'] <= 0.004 and s.get('nll2', 0) + d['nll2'] <= 0.02, 4),
        'b_sparse_graph': (sparse_edges, lambda d, s: True, 3),
        'c_tight_budget_exhausts': (lambda k, rng: [(i, j) for i in range(k) for j in range(i + 1, k)],
                                    lambda d, s: s.get('brier2', 0) + d['brier2'] <= 0.0015, 1),
    }
    for name, (edge_fn, adm, stop) in scenarios.items():
        agree, details = 0, []
        for seed in range(21, 27):
            assign, k, t1, t2, pi, sex, occ = invent(seed, 120, 12)
            rng = np.random.default_rng(seed + 1000)
            edges = edge_fn(k, rng)
            L = to_lists(assign, t1, t2, pi, sex, occ)
            fin, sp_e, st_e, why_e = engine_greedy(assign, k, t1, t2, pi, sex, occ, edges, scar_score, adm, stop)
            lab, sp_n, st_n, why_n, gaps = naive_greedy(L, k, edges, scar_score, adm, stop)
            same_partition = list(map(int, fin)) == lab
            # the engine's accumulated cost must equal the brute-force total change of the final partition
            final_assign = [lab[c] for c in L[0]]
            tot = deltas(metrics(*L), metrics(final_assign, *L[1:]))
            spent_err = max(abs(sp_e.get(kk, 0.0) - tot[kk]) for kk in INCREASE + DECREASE) if st_e else 0.0
            ok = same_partition and st_e == st_n and why_e == why_n and spent_err < 1e-12
            agree += ok
            details.append({'seed': seed, 'cells': k, 'steps': [st_e, st_n], 'stop_reason': [why_e, why_n],
                            'same_partition': same_partition, 'spent_vs_bruteforce_max_err': spent_err,
                            'min_score_gap_naive': min(gaps) if gaps else None})
        res[name] = {'instances': len(details), 'agree': agree, 'pass': agree == len(details), 'detail': details}
    # (d) probe: a budget on a cost that can be NEGATIVE (the within-SEX guard). The engine discards a popped inadmissible
    # pair for good; the exact greedy re-tests admissibility after every merge. Divergence is reported, not failed.
    # Control (d0): the same 30 seeds and score with a budget on a NON-NEGATIVE cost (occupation Brier) must agree 30/30,
    # so any divergence in (d) is attributable to the non-monotone guard budget, not to the reference implementation.
    for tag, key, budget in (('d0_control_nonnegative_cost_budget', 'brier2', 0.002), ('d_probe_negative_cost_budget', 'guard0', 0.002)):
        div, inst, example, neg_steps = 0, 0, None, 0
        for seed in range(41, 71):
            assign, k, t1, t2, pi, sex, occ = invent(seed, 100, 10)
            edges = [(i, j) for i in range(k) for j in range(i + 1, k)]
            L = to_lists(assign, t1, t2, pi, sex, occ)
            adm = (lambda d, s, key=key, budget=budget: s.get(key, 0) + d[key] <= budget)
            score = (lambda d: -d['sexvar'])
            fin, sp_e, st_e, why_e = engine_greedy(assign, k, t1, t2, pi, sex, occ, edges, score, adm, None)
            lab, sp_n, st_n, why_n, _ = naive_greedy(L, k, edges, score, adm, None)
            inst += 1
            if list(map(int, fin)) != lab:
                div += 1
                if example is None:
                    example = {'seed': seed, 'engine_steps': st_e, 'exact_steps': st_n, 'engine_reason': why_e, 'exact_reason': why_n,
                               f'engine_spent_{key}': sp_e.get(key), f'exact_spent_{key}': sp_n.get(key)}
        res[tag] = {'instances': inst, 'diverged_from_exact_greedy': div, 'first_example': example}
    res['d0_control_nonnegative_cost_budget']['pass'] = res['d0_control_nonnegative_cost_budget']['diverged_from_exact_greedy'] == 0
    res['d_probe_negative_cost_budget']['note'] = ('informational: a popped inadmissible pair is discarded for good (partition.py greedy), '
                                                   'which equals the exact greedy only when admissibility can never return, i.e. when every '
                                                   'budgeted cost is non-negative; the within-SEX guard change can be negative')
    return res


# ------------------------------------------------------------------------------------------------------ 3. k-means
def check_kmeans():
    rng = np.random.default_rng(5)
    Q = np.column_stack((rng.dirichlet([2, 2], 600), rng.dirichlet([1] * 5, 600)))
    c1, a1 = P.kmeans(Q, 24, seed=9, iters=30)
    c2, a2 = P.kmeans(Q, 24, seed=9, iters=30)
    exact = np.array([min(range(len(c1)), key=lambda j: (float(((q - c1[j]) ** 2).sum()), j)) for q in Q])
    return {'deterministic_given_seed': bool(np.array_equal(a1, a2) and np.array_equal(c1, c2)),
            'nearest_equals_exact_argmin_share': float(np.mean(P.nearest(Q, c1) == exact)),
            'returned_assignment_equals_nearest_of_returned_centroids': bool(np.array_equal(a1, P.nearest(Q, c1))),
            'note': 'Euclidean only; SCM specified KL-Bregman assignment, which this engine does not implement'}


# -------------------------------------------------------------------------------------------------- 4. lookup reader
def my_key(*rows):
    vals = [float(v) for r in rows for v in r]
    return struct.pack('<%dd' % len(vals), *vals)           # explicit little-endian IEEE-754 binary64


def check_lookup():
    out = {'byteorder': sys.byteorder}
    rng = np.random.default_rng(77)
    m = 30
    inc_tab, occ_tab = rng.dirichlet([1, 1], m), rng.dirichlet([1] * 5, m)
    cell = rng.integers(0, m, 3000)
    rate = rng.uniform(0.1, 0.9, m)
    y = (rng.random(3000) < rate[cell]).astype(np.int64)                  # 1 = female (acs_replication.schema.SEX_CODE)
    inc, occ = inc_tab[cell], occ_tab[cell]
    # (i) key bytes = exact little-endian float64 of the vector(s), pair = income then occupation
    k_m, k_p = lookup_keys(occ), lookup_keys(inc, occ)
    out['marginal_keys_equal_struct_pack'] = all(k == my_key(r) for k, r in zip(k_m, occ))
    out['pair_keys_equal_struct_pack_income_then_occupation'] = all(k == my_key(a, b) for k, a, b in zip(k_p, inc, occ))
    # (ii) smoothing on seen keys: (n_female + 2 pi) / (n + 2), pi = female share of the fitting sample
    rec = fit_lookup(k_p, y)
    pi = float(np.mean(y))
    nf, nn = Counter(), Counter()
    for k, v in zip(k_p, y.tolist()):
        nf[k] += v; nn[k] += 1
    seen = list(nn)
    mine = np.array([(nf[k] + 2 * pi) / (nn[k] + 2) for k in seen])
    got = predict_lookup(rec, seen)
    out['pi_equals_fit_sample_female_share'] = rec['pi'] == pi
    out['seen_key_count'] = [len(rec['table']), len(seen)]
    out['seen_key_score_max_abs_diff'] = float(np.max(np.abs(mine - got)))
    out['seen_key_scores_bitwise_equal'] = bool(np.array_equal(mine, got))
    # (iii) unseen keys -> pi exactly; one ulp away is a different key; -0.0 and +0.0 are different keys
    one_ulp = occ_tab[[3]].copy(); one_ulp[0, 0] = np.nextafter(one_ulp[0, 0], 2.0)
    rec_m = fit_lookup(k_m, y)
    out['unseen_key_scores_pi'] = bool(predict_lookup(rec_m, [my_key(np.full(5, 0.2))])[0] == rec_m['pi'])
    out['one_ulp_change_is_unseen'] = bool(predict_lookup(rec_m, lookup_keys(one_ulp))[0] == rec_m['pi'])
    z_pos, z_neg = np.array([[0.0, 1.0]]), np.array([[-0.0, 1.0]])
    out['signed_zero_gives_distinct_keys'] = lookup_keys(z_pos)[0] != lookup_keys(z_neg)[0]
    # (iv) a value-equal pair of vectors computed differently gives different keys when the last bit differs
    out['0.1+0.2_vs_0.3_distinct_keys'] = lookup_keys(np.array([[0.1 + 0.2, 0.7]]))[0] != lookup_keys(np.array([[0.3, 0.7]]))[0]
    # (v) the decision adds nothing: argmax is a function of the vector, so appending it never splits a key class
    with_dec = {}
    for k, r in zip(k_p, np.column_stack((inc, occ))):
        with_dec.setdefault(k, set()).add((int(np.argmax(r[:2])), int(np.argmax(r[2:]))))
    out['decision_constant_within_key'] = all(len(v) == 1 for v in with_dec.values())
    # (vi) n = 1 key: female -> (1 + 2 pi) / 3, male -> 2 pi / 3
    single = fit_lookup([b'a', b'b'], np.array([1, 0]))
    out['singleton_scores'] = [float(x) for x in predict_lookup(single, [b'a', b'b'])]
    out['singleton_expected'] = [(1 + 2 * 0.5) / 3, (2 * 0.5) / 3]
    out['pass'] = bool(out['marginal_keys_equal_struct_pack'] and out['pair_keys_equal_struct_pack_income_then_occupation']
                       and out['pi_equals_fit_sample_female_share'] and out['seen_key_scores_bitwise_equal']
                       and out['unseen_key_scores_pi'] and out['one_ulp_change_is_unseen'] and out['decision_constant_within_key']
                       and out['singleton_scores'] == out['singleton_expected'] and sys.byteorder == 'little')
    return out


# ------------------------------------------------------------- 5. composition scope (SERVICE_AND_THREAT_MODEL.md section 6)
def auc_finite(pos, neg):
    """Exact AUC of a score for finite laws: pos/neg are lists of (score, probability); ties count 1/2."""
    return sum(p1 * p0 * (1.0 if s1 > s0 else 0.5 if s1 == s0 else 0.0) for s1, p1 in pos for s0, p0 in neg)


def law_moments(law):
    """law: list of (z1, z2, s, prob). Returns Cov(z1, S), Cov(z2, S)."""
    es = sum(p * s for _, _, s, p in law)
    e1 = sum(p * a for a, _, _, p in law)
    e2 = sum(p * b for _, b, _, p in law)
    return [sum(p * a * s for a, _, s, p in law) - e1 * es, sum(p * b * s for _, b, s, p in law) - e2 * es]


def check_composition_examples():
    out = {}
    # (A) exact zero covariance in each release -> zero covariance of the concatenation; nonlinear readers still recover S
    law_a = [(z1, z1 * (2 * s - 1), s, 0.25) for z1 in (-1, 1) for s in (0, 1)]
    cov_a = law_moments(law_a)
    lin = []
    for w1, w2 in ((1, 0), (0, 1), (1, 1), (1, -1), (1, 0.5), (0.3, -2)):
        pos = [(w1 * a + w2 * b, p / 0.5) for a, b, s, p in law_a if s == 1]
        neg = [(w1 * a + w2 * b, p / 0.5) for a, b, s, p in law_a if s == 0]
        lin.append(auc_finite(pos, neg))
    prod_pos = [(a * b, p / 0.5) for a, b, s, p in law_a if s == 1]
    prod_neg = [(a * b, p / 0.5) for a, b, s, p in law_a if s == 0]
    out['A_exact_zero_cov'] = {'cov_z1_s': cov_a[0], 'cov_z2_s': cov_a[1], 'linear_score_aucs': lin,
                               'nonlinear_z1_times_z2_auc': auc_finite(prod_pos, prod_neg)}
    # (B) small nonzero covariances do not compose: Z1 = A + eps S, Z2 = A - eps S
    eps, grid = 0.01, [i / 100 for i in range(-100, 101)]
    pa = 1.0 / len(grid)
    one_pos = [(a + eps, pa) for a in grid]; one_neg = [(a, pa) for a in grid]
    out['B_small_cov_does_not_compose'] = {'eps': eps, 'single_release_linear_auc': auc_finite(one_pos, one_neg),
                                           'pair_linear_reader_z1_minus_z2_auc': auc_finite([(2 * eps, 1.0)], [(0.0, 1.0)])}
    # (C) zero covariance does not make every linear score's AUC 0.5 (LEACE's guarantee is about convex losses)
    neg_c = [(-(-3.0), 0.25), (-(1.0), 0.75)]          # score -X with X | S=0 in {-3 (1/4), 1 (3/4)}, mean 0
    pos_c = [(-(0.0), 1.0)]                             # X | S=1 = 0, mean 0
    out['C_zero_cov_linear_auc'] = {'class_means': [-3 * 0.25 + 1 * 0.75, 0.0], 'auc_of_minus_x': auc_finite(pos_c, neg_c)}
    out['pass'] = bool(cov_a == [0.0, 0.0] and all(abs(x - 0.5) < 1e-15 for x in lin) and out['A_exact_zero_cov']['nonlinear_z1_times_z2_auc'] == 1.0
                       and out['B_small_cov_does_not_compose']['pair_linear_reader_z1_minus_z2_auc'] == 1.0
                       and out['C_zero_cov_linear_auc']['auc_of_minus_x'] == 0.75)
    return out


if __name__ == '__main__':
    report = {'identities': check_identities(), 'greedy': check_greedy(), 'kmeans': check_kmeans(), 'lookup': check_lookup(),
              'composition_examples': check_composition_examples(), 'numpy': np.__version__, 'python': sys.version.split()[0]}
    report['pass'] = bool(report['identities']['pass'] and all(v['pass'] for k, v in report['greedy'].items() if 'pass' in v)
                          and report['kmeans']['deterministic_given_seed'] and report['lookup']['pass']
                          and report['composition_examples']['pass'])
    (HERE / 'V_SYNTHETIC_CHECKS.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({'identities': report['identities'], 'greedy': {k: {kk: v[kk] for kk in v if kk != 'detail'} for k, v in report['greedy'].items()},
                      'kmeans': report['kmeans'], 'lookup': report['lookup'], 'composition_examples': report['composition_examples'],
                      'pass': report['pass']}, indent=1))
