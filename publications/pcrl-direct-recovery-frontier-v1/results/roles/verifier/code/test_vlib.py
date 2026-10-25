"""Invented-data tests of the verifier library (no file, no label, no campaign code): every function of vlib is checked against a
brute-force or exact computation written separately here, or against a known closed-form answer.

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/test_vlib.py      prints JSON; exit 0 iff all pass
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402


def brute_auc_people(c1, c0, score):
    """Exact AUC of a score over explicit people: P(score(S=1 person) > score(S=0 person)) + 1/2 P(tie)."""
    pos = [score[g] for g, c in enumerate(c1) for _ in range(int(c))]
    neg = [score[g] for g, c in enumerate(c0) for _ in range(int(c))]
    tot = Fraction(0)
    for a in pos:
        for b in neg:
            tot += 1 if a > b else (Fraction(1, 2) if a == b else 0)
    return tot / (len(pos) * len(neg))


def t_auc(rng):
    ok = True
    worst = 0.0
    for _ in range(150):
        K = int(rng.integers(1, 7))
        c1 = rng.integers(0, 4, K)
        c0 = rng.integers(0, 4, K)
        if c1.sum() == 0 or c0.sum() == 0:
            continue
        lr = [Fraction(int(a), int(a + b)) if a + b > 0 else Fraction(-1) for a, b in zip(c1, c0)]
        exact = brute_auc_people(c1, c0, lr)
        for got in (V.auc_star_closed(c1, c0), V.auc_star_sorted(c1, c0), float(V.auc_star_exact_counts(c1, c0))):
            worst = max(worst, abs(got - float(exact)))
        ok &= V.auc_star_exact_counts(c1, c0) == exact
        if K <= 4:                                        # optimality: no score of the K groups beats AUC*
            best = max(brute_auc_people(c1, c0, s) for s in itertools.product(range(K), repeat=K))
            ok &= best == exact
    return bool(ok and worst < 1e-12), {'max_abs_error': worst}


def t_known_laws():
    a = V.law_values(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=float))
    # law A: m1 = (3, 2), m0 = (2, 3): AUC* = 3/5, MI = 0.6 ln 1.2 + 0.4 ln 0.8 (rbar = 1/2)
    mi_a = 0.6 * math.log(1.2) + 0.4 * math.log(0.8)
    # law B: m1 = (1, 9), m0 = (0, 10): AUC* = 11/20
    m1, m0 = np.array([1.0, 9.0]), np.array([0.0, 10.0])
    mi_b = 0.05 * math.log(2) + 0.95 * ((9 / 19) * math.log((9 / 19) / 0.5) + (10 / 19) * math.log((10 / 19) / 0.5))
    ok = (abs(a['auc'] - 0.6) < 1e-15 and abs(a['mi'] - mi_a) < 1e-15 and abs(V.auc_star_closed(m1, m0) - 0.55) < 1e-15
          and abs(V.plugin_mi(m1, m0, clip=None) - mi_b) < 1e-15 and abs(V.plugin_mi(m1, m0) - mi_b) < 1e-7
          and abs(V.auc_star_sorted(m1, m0) - 0.55) < 1e-15)
    # one-group law: AUC* = 1/2, MI = 0; swapping classes keeps AUC*; scaling masses keeps AUC*
    one = V.auc_star_closed(np.array([5.0]), np.array([3.0]))
    r = np.random.default_rng(3)
    x1, x0 = r.random(9), r.random(9)
    ok &= abs(one - 0.5) < 1e-15 and abs(V.auc_star_closed(x1, x0) - V.auc_star_closed(x0, x1)) < 1e-14
    ok &= abs(V.auc_star_closed(3 * x1, 0.5 * x0) - V.auc_star_closed(x1, x0)) < 1e-14
    return bool(ok), {'mi_A': a['mi'], 'mi_B': V.plugin_mi(m1, m0, clip=None)}


def t_merge(rng):
    """Two atoms with identical emitted vectors are one observable value; merging never raises AUC*."""
    n = 3000
    atom = rng.integers(0, 6, n)
    T = rng.dirichlet(np.ones(3), 6)
    T[4] = T[1]                                          # planted duplicate emitted vector
    w1 = np.clip(0.2 + 0.1 * atom + 0.05 * rng.random(n), 0, 1)
    vid = V.row_ids(T[atom])
    ok = len(np.unique(vid)) == 5
    merged_atom = np.where(atom == 4, 1, atom)
    a_m = V.law_values(vid, w1)['auc']
    a_h = V.law_values(merged_atom, w1)['auc']
    a_a = V.law_values(atom, w1)['auc']
    ok &= abs(a_m - a_h) < 1e-14 and a_m <= a_a + 1e-15
    # pair view: ids of the pair equal ids of the concatenated vectors
    U = rng.dirichlet(np.ones(2), 4)
    b = rng.integers(0, 4, n)
    pid = V.pair_ids(V.row_ids(T[atom]), V.row_ids(U[b]))
    pid2 = V.row_ids(np.hstack([T[atom], U[b]]))
    ok &= len(np.unique(pid)) == len(np.unique(pid2)) and V.same_partition(pid, pid2)
    # data processing: pair AUC* >= each single view
    ok &= V.law_values(pid, w1)['auc'] >= max(V.law_values(vid, w1)['auc'], V.law_values(V.row_ids(U[b]), w1)['auc']) - 1e-15
    return bool(ok), {'auc_merged': a_m, 'auc_atoms': a_a}


def t_routing(rng):
    Q = rng.dirichlet(np.ones(5) * 0.7, 400)
    C = rng.dirichlet(np.ones(5), 64)
    idx, gap = V.nearest_codeword(Q, C)
    ok = True
    for i in range(len(Q)):
        h = [-sum(Q[i, j] * math.log(C[c, j]) for j in range(5)) for c in range(64)]
        ok &= int(np.argmin(h)) == int(idx[i]) and abs(sorted(h)[1] - sorted(h)[0] - gap[i]) < 1e-12
    # ties: identical codewords -> first index
    C2 = np.vstack([C[:3], C[1:2]])
    idx2, gap2 = V.nearest_codeword(np.vstack([C[1]]), C2)
    ok &= int(idx2[0]) == 1 and gap2[0] == 0.0
    return bool(ok), {}


def t_quantile_bins(rng):
    ok = True
    for x in (rng.random(2001), np.round(rng.random(5000), 2), np.r_[np.full(100, 0.01), rng.random(37), np.full(50, 0.99)]):
        mine = [V.quantile_type7(x, q) for q in (0.25, 0.5, 0.75)]
        ref = np.quantile(x, [0.25, 0.5, 0.75])
        ok &= max(abs(a - b) for a, b in zip(mine, ref)) <= 2 * np.finfo(float).eps
        ok &= np.array_equal(V.pi_bins(x, ref), np.searchsorted(ref, x, side='right'))
    return bool(ok), {}


def t_tables(rng):
    n, M = 500, 9
    atom = rng.integers(0, M - 2, n)                    # atoms M-2, M-1 empty
    Q = rng.dirichlet(np.ones(4), n)
    Q[:5] = np.array([1 - 3e-9, 1e-9, 1e-9, 1e-9])     # tiny posteriors exercise the clip
    atom[:5] = 0
    fb = rng.dirichlet(np.ones(4), M)
    T, cnt = V.b1_tables(atom, Q, M, fb)
    ok = True
    for a in range(M):
        rows = Q[atom == a]
        raw = rows.mean(0) if len(rows) else fb[a]
        cl = np.clip(raw, 1e-6, 1.0)
        ok &= np.allclose(T[a], cl / cl.sum(), atol=1e-15, rtol=0) and cnt[a] == len(rows)
    return bool(ok), {}


def t_costs_spends(rng):
    n = 700
    Vi_c, Vo_c = rng.dirichlet(np.ones(2), n), rng.dirichlet(np.ones(5), n)
    Vi_0, Vo_0 = rng.dirichlet(np.ones(2), n), rng.dirichlet(np.ones(5), n)
    yi, yo, sex = rng.integers(0, 2, n), rng.integers(0, 5, n), rng.integers(0, 2, n)
    sp = V.spends8(Vi_c, Vo_c, Vi_0, Vo_0, yi, yo, sex)

    def direct(P, y):
        nll = [-math.log(P[i, y[i]]) for i in range(len(y))]
        br = [sum((P[i, k] - (1.0 if k == y[i] else 0.0)) ** 2 for k in range(P.shape[1])) for i in range(len(y))]
        acc = [1.0 if int(np.argmax(P[i])) == y[i] else 0.0 for i in range(len(y))]
        return np.array(nll), np.array(br), np.array(acc)
    a = direct(Vi_c, yi); b = direct(Vo_c, yo); c = direct(Vi_0, yi); d = direct(Vo_0, yo)
    want = {'inc_nll': a[0].mean() - c[0].mean(), 'occ_nll': b[0].mean() - d[0].mean(), 'inc_brier': a[1].mean() - c[1].mean(),
            'occ_brier': b[1].mean() - d[1].mean(), 'inc_acc': c[2].mean() - a[2].mean(), 'occ_acc': d[2].mean() - b[2].mean(),
            'occ_brier_men': (b[1] - d[1])[sex == 0].mean(), 'occ_brier_women': (b[1] - d[1])[sex == 1].mean()}
    err = max(abs(sp[k] - want[k]) for k in V.METRICS8)
    # the evaluator's NLL floor: p_y = 0 -> -log(1e-12)
    P = np.array([[1.0, 0.0], [0.5, 0.5]])
    fl = V.person_costs(P, np.array([1, 0]), 1e-12)['nll']
    ok = err < 1e-13 and abs(fl[0] + math.log(1e-12)) < 1e-12 and abs(fl[1] - math.log(2)) < 1e-15
    return bool(ok), {'max_abs_error': err}


def t_se(rng):
    n = 900
    unit = rng.integers(0, 300, n)
    d = rng.normal(size=n)
    groups = {}
    for i, u in enumerate(unit):
        groups.setdefault(int(u), []).append(i)
    G = len(groups)
    s = [sum(d[i] - d.mean() for i in idx) for idx in groups.values()]
    want = math.sqrt(G / (G - 1) * sum(x * x for x in s)) / n
    ok = abs(V.cluster_se(d, unit) - want) < 1e-14
    one = V.cluster_se(d, np.arange(n))
    ok &= abs(one - math.sqrt(n / (n - 1)) * d.std() / math.sqrt(n)) < 1e-14
    mask = rng.random(n) < 0.4
    m, p = d[mask].mean(), mask.mean()
    s2 = [sum(((d[i] - m) / p if mask[i] else 0.0) for i in idx) for idx in groups.values()]
    want2 = math.sqrt(G / (G - 1) * sum(x * x for x in s2)) / n
    ok &= abs(V.subgroup_se(d, mask, unit) - want2) < 1e-14
    return bool(ok), {}


def t_conditions(rng):
    n = 1200
    unit = rng.integers(0, 500, n)
    sex = rng.integers(0, 2, n)
    y = {'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n)}
    U0 = {'income': rng.dirichlet(np.ones(2) * 3, n), 'occupation': rng.dirichlet(np.ones(5) * 3, n)}
    Ucal = {'income': rng.dirichlet(np.ones(2) * 3, n), 'occupation': rng.dirichlet(np.ones(5) * 3, n)}
    refs = {'U0': U0, 'Ucal': Ucal}
    same = V.conditions14(U0, refs, y, sex, unit)
    ok = all(abs(c['value']) == 0.0 for c in same['clauses'] if c['reference'] == 'U0') and all(g['value'] == 0.0 for g in same['guards'])
    rel = {'income': rng.dirichlet(np.ones(2), n), 'occupation': rng.dirichlet(np.ones(5), n)}
    res = V.conditions14(rel, refs, y, sex, unit)
    for c in res['clauses']:
        P, R, yy = rel[c['task']], refs[c['reference']][c['task']], y[c['task']]
        own = [(-math.log(max(P[i, yy[i]], 1e-12)), sum((P[i, k] - (k == yy[i])) ** 2 for k in range(P.shape[1])),
                float(np.argmax(P[i]) == yy[i])) for i in range(n)]
        ref = [(-math.log(max(R[i, yy[i]], 1e-12)), sum((R[i, k] - (k == yy[i])) ** 2 for k in range(R.shape[1])),
                float(np.argmax(R[i]) == yy[i])) for i in range(n)]
        want = {'nll_excess': np.mean([a[0] - b[0] for a, b in zip(own, ref)]),
                'brier_excess': np.mean([a[1] - b[1] for a, b in zip(own, ref)]),
                'accuracy_loss': np.mean([b[2] - a[2] for a, b in zip(own, ref)])}[c['metric']]
        ok &= abs(c['value'] - want) < 1e-13 and c['point_pass'] == (c['value'] <= c['cap'])
    br = lambda P, yy: np.array([sum((P[i, k] - (k == yy[i])) ** 2 for k in range(5)) for i in range(n)])
    d = br(rel['occupation'], y['occupation']) - br(U0['occupation'], y['occupation'])
    ok &= abs(res['guards'][0]['value'] - d[sex == 0].mean()) < 1e-13 and abs(res['guards'][1]['value'] - d[sex == 1].mean()) < 1e-13
    ok &= res['all_14_pass'] == all(c['point_pass'] for c in res['clauses'] + res['guards'])
    ok &= len(res['clauses']) == 12 and len(res['guards']) == 2
    ok &= V.valid_probs(rel['income'], 2) and not V.valid_probs(rel['income'] * 1.1, 2)
    return bool(ok), {'random_release_all_pass': res['all_14_pass']}


def t_path(rng):
    """Every greedy step is the least total-NLL-increase merge over all candidate merges (both recipients), recomputed per person;
    with realized (one-hot) labels and with soft label distributions (expected NLL, revision R1)."""
    n = 1500
    k = rng.integers(0, 16, n)
    l = rng.integers(0, 12, n)                           # 12 of 64 occupation atoms used
    m = rng.integers(0, 4, n)
    ic, oc = V.fine_cells(k, l, m)
    qI = rng.dirichlet(np.ones(2) * 2, n)
    qO = rng.dirichlet(np.ones(5) * 2, n)
    yi = np.array([rng.choice(2, p=p) for p in qI])
    yo = np.array([rng.choice(5, p=p) for p in qO])
    ok, steps = True, []
    for Li, Lo in ((V.onehot(yi, 2), V.onehot(yo, 5)), (rng.dirichlet(np.ones(2), n), rng.dirichlet(np.ones(5), n))):
        path = V.coarsening_path(ic, oc, qI, qO, Li, Lo, max_steps=6, max_spend=10.0)

        def total(Ai, Ao):
            tot = 0.0
            for A, cells, Q, L, M in ((Ai, ic, qI, Li, 16), (Ao, oc, qO, Lo, 64)):
                atom = A[cells]
                T, _ = V.b1_tables(atom, Q, M, np.full((M, Q.shape[1]), 1.0 / Q.shape[1]))
                tot += float(-(L * np.log(T[atom])).sum())
            return tot
        ok &= len(path) == 7
        base = total(*path[0][:2])
        for s in range(1, len(path)):
            Ai, Ao = path[s - 1][0], path[s - 1][1]
            cur = total(Ai, Ao)
            best = None
            for side in ('inc', 'occ'):
                A = Ai if side == 'inc' else Ao
                used = sorted(set(A[ic if side == 'inc' else oc].tolist()))
                for a, b in itertools.combinations(used, 2):
                    B = A.copy(); B[B == b] = a
                    val = total(B, Ao) if side == 'inc' else total(Ai, B)
                    if best is None or val - cur < best[0] - 1e-9:
                        best = (val - cur, side, a, b)
            side, a, b = best[1], best[2], best[3]
            A2i, A2o = Ai.copy(), Ao.copy()
            (A2i if side == 'inc' else A2o)[(A2i if side == 'inc' else A2o) == b] = a
            ok &= np.array_equal(A2i, path[s][0]) and np.array_equal(A2o, path[s][1])
            ok &= abs(path[s][3] - (total(path[s][0], path[s][1]) - base) / n) < 1e-12
        steps.append(len(path) - 1)
    return bool(ok), {'steps': steps}


def t_expected(rng):
    """Expected costs equal the label-enumeration expectation; one-hot distributions reproduce the realized costs; the expected
    spends equal per-person enumeration with the guard distribution of each person's own SEX group."""
    n = 400
    Vc, V0 = rng.dirichlet(np.ones(5), n), rng.dirichlet(np.ones(5), n)
    L = rng.dirichlet(np.ones(5), n)
    e = V.expected_costs(Vc, L)
    ok = True
    for i in range(n):
        nll = sum(L[i, c] * -math.log(Vc[i, c]) for c in range(5))
        br = sum(L[i, c] * sum((Vc[i, k] - (k == c)) ** 2 for k in range(5)) for c in range(5))
        acc = sum(L[i, c] * float(int(np.argmax(Vc[i])) == c) for c in range(5))
        ok &= abs(e['nll'][i] - nll) < 1e-13 and abs(e['brier'][i] - br) < 1e-13 and abs(e['acc'][i] - acc) < 1e-15
    y = rng.integers(0, 5, n)
    r, h = V.person_costs(Vc, y), V.expected_costs(Vc, V.onehot(y, 5))
    ok &= all(np.allclose(r[k], h[k], atol=1e-14, rtol=0) for k in ('nll', 'brier', 'acc'))
    Vi_c, Vi_0 = rng.dirichlet(np.ones(2), n), rng.dirichlet(np.ones(2), n)
    Li = rng.dirichlet(np.ones(2), n)
    sex = rng.integers(0, 2, n)
    G0, G1 = rng.dirichlet(np.ones(5), n), rng.dirichlet(np.ones(5), n)
    Lg = np.where(sex[:, None] == 0, G0, G1)
    sp = V.spends8_expected(Vi_c, Vc, Vi_0, V0, Li, L, Lg, sex)

    def eb(P, D, i):
        return sum(D[i, c] * sum((P[i, k] - (k == c)) ** 2 for k in range(P.shape[1])) for c in range(P.shape[1]))
    men = np.mean([eb(Vc, G0, i) - eb(V0, G0, i) for i in range(n) if sex[i] == 0])
    women = np.mean([eb(Vc, G1, i) - eb(V0, G1, i) for i in range(n) if sex[i] == 1])
    occ_br = np.mean([eb(Vc, L, i) - eb(V0, L, i) for i in range(n)])
    inc_acc = np.mean([Li[i, int(np.argmax(Vi_0[i]))] - Li[i, int(np.argmax(Vi_c[i]))] for i in range(n)])
    ok &= abs(sp['occ_brier_men'] - men) < 1e-13 and abs(sp['occ_brier_women'] - women) < 1e-13
    ok &= abs(sp['occ_brier'] - occ_br) < 1e-13 and abs(sp['inc_acc'] - inc_acc) < 1e-15
    one = V.spends8_expected(Vi_c, Vc, Vi_0, V0, V.onehot(y % 2, 2), V.onehot(y, 5), V.onehot(y, 5), sex)
    real = V.spends8(Vi_c, Vc, Vi_0, V0, y % 2, y, sex)
    ok &= max(abs(one[k] - real[k]) for k in V.METRICS8) < 1e-14
    return bool(ok), {}


def t_aggregate(rng):
    """Aggregate screen against hand computation on invented cells; labels, alias, floor, recipient gate at its boundary."""
    cells = {c: float(rng.normal(0.01, 0.01)) for c in V.REGISTERED}
    ses = {c: float(abs(rng.normal(0.006, 0.001))) for c in V.REGISTERED}
    ag = V.aggregate(cells, ses)
    blocks = {}
    for c in V.REGISTERED:
        blocks.setdefault(c[:4], []).append(cells[c])
    mb = {b: sum(x) / len(x) for b, x in blocks.items()}
    x = sum(mb.values()) / 4
    vm0 = (mb['S0R1'] + mb['S0R2']) / 2
    vm1 = (mb['S1R1'] + mb['S1R2']) / 2
    se_hh = math.sqrt(sum(s * s for s in ses.values()) / 8)
    se_rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / 4)
    ok = (abs(ag['x'] - x) < 1e-15 and abs(ag['version_means'][0] - vm0) < 1e-15 and abs(ag['version_means'][1] - vm1) < 1e-15
          and abs(ag['se_hh'] - se_hh) < 1e-15 and abs(ag['se_rs'] - se_rs) < 1e-15
          and abs(ag['se_agg'] - max(math.sqrt(se_hh ** 2 + se_rs ** 2), 0.001)) < 1e-15)
    lab = V.benefit_label(ag)
    want = ('AGG_POSITIVE' if (x - 2 * ag['se_agg'] > 0 and vm0 > 0 and vm1 > 0) else 'AGG_POSITIVE_POINT_ONLY' if x > 0
            else 'AGG_NOT_POSITIVE')
    ok &= lab == want
    pos = {c: 0.05 for c in V.REGISTERED}
    sm = {c: 0.002 for c in V.REGISTERED}
    ok &= V.benefit_label(V.aggregate(pos, sm)) == 'AGG_POSITIVE' and V.target_label(V.aggregate(pos, sm)) == 'TARGET_AGG_RESOLVED'
    one_neg = dict(pos)
    for c in V.REGISTERED:
        if c.startswith('S1'):
            one_neg[c] = -0.001                          # version 1 mean negative: not AGG_POSITIVE even with a large x? (x = 0.0245)
    ok &= V.benefit_label(V.aggregate(one_neg, sm)) == 'AGG_POSITIVE_POINT_ONLY'
    zero = {c: 0.0 for c in V.REGISTERED}
    ok &= V.benefit_label(V.aggregate(zero, zero)) == 'ALIAS_ZERO' and V.recipient(zero, zero)['gate'] is True
    tiny = V.aggregate({c: 0.0 for c in V.REGISTERED}, {c: 1e-6 for c in V.REGISTERED})
    ok &= tiny['se_agg'] == 0.001 and V.benefit_label(tiny) == 'AGG_NOT_POSITIVE'
    edge = {c: 0.010 for c in V.REGISTERED}
    rr = V.recipient(edge, {c: 0.004 for c in V.REGISTERED})
    ok &= rr['gate'] is True and rr['label'] == 'RECIPIENT_AGG_EDGE'
    over = dict(edge)
    over['S0R1k0'] = 0.0101
    over['S0R1k1'] = 0.0101
    ok &= V.recipient(over, {c: 0.004 for c in V.REGISTERED})['gate'] is False
    ok &= not V.aggregate({c: 0.0 for c in V.REGISTERED[:-1]}, ses)['complete']
    ok &= V.parse_cell('S1R2k0') == (1, 2, 0) and V.parse_cell('S0R10k3') == (0, 10, 3)
    return bool(ok), {'label_random_cells': lab}


def t_weighted_auc_boot(rng):
    n = 300
    y = rng.integers(0, 2, n)
    s = np.round(rng.normal(size=n), 1)                  # ties
    w1 = V.weighted_auc(y, s, np.ones(n))
    pos, neg = s[y == 1], s[y == 0]
    mw = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    ok = abs(w1 - mw) < 1e-12
    w = rng.integers(0, 3, n).astype(float)
    rep_y = np.repeat(y, w.astype(int))
    rep_s = np.repeat(s, w.astype(int))
    pos, neg = rep_s[rep_y == 1], rep_s[rep_y == 0]
    mw2 = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    ok &= abs(V.weighted_auc(y, s, w) - mw2) < 1e-12
    unit = rng.integers(0, 90, n)
    W = V.household_bootstrap_weights(unit, 123, B=50)
    g, inv = np.unique(unit, return_inverse=True)
    ok &= W.shape == (50, n)
    for b in range(50):
        per_h = np.array([W[b][inv == j][0] for j in range(len(g))])
        ok &= all(np.all(W[b][inv == j] == per_h[j]) for j in range(len(g))) and per_h.sum() == len(g)
    W2 = V.household_bootstrap_weights(unit, 123, B=50)
    ok &= np.array_equal(W, W2)
    batch = V.weighted_auc_batch(y, s, W)
    ok &= np.allclose(batch, [V.weighted_auc(y, s, W[b]) for b in range(50)], atol=1e-14, rtol=0)
    ok &= abs(V.auc(y, s) - w1) < 1e-15
    parts = V.household_parts(unit, 10, 'TAG')
    import hashlib
    keys = sorted((hashlib.sha256(f'TAG|{h}'.encode()).hexdigest(), h) for h in g)
    want = {h: i % 10 for i, (_, h) in enumerate(keys)}
    ok &= all(parts[i] == want[unit[i]] for i in range(n))
    return bool(ok), {}


def t_partition():
    ok = V.same_partition(np.array([0, 0, 1, 2]), np.array([5, 5, 3, 9])) and not V.same_partition(np.array([0, 0, 1, 2]),
                                                                                                   np.array([5, 3, 3, 9]))
    ok &= not V.same_partition(np.array([0, 1]), np.array([0, 0]))
    return bool(ok), {}


def t_cmp():
    a = V.cmp_num('x', 1.0, 1.0 + 5e-10)
    b = V.cmp_num('y', 1.0, 1.0 + 2e-9)
    c = V.cmp_num('z', 1.0, None)
    return bool(a['result'] == 'PASS' and b['result'] == 'FAIL' and c['result'] == 'FAIL'), {}


def main():
    t0 = time.process_time()
    rng = np.random.default_rng(20261025)
    tests = {'auc_star_vs_bruteforce_exact_and_optimal': t_auc(rng), 'known_laws_A_B_and_invariances': t_known_laws(),
             'identical_wires_merged': t_merge(rng), 'routing_vs_bruteforce': t_routing(rng),
             'quantile_and_bins': t_quantile_bins(rng), 'b1_tables_clip_fallback': t_tables(rng),
             'costs_and_8_spends': t_costs_spends(rng), 'cluster_and_subgroup_se': t_se(rng),
             'conditions14_vs_loops': t_conditions(rng), 'coarsening_path_greedy_vs_bruteforce': t_path(rng),
             'expected_costs_and_spends_R1': t_expected(rng), 'aggregate_screen_and_recipient_gate': t_aggregate(rng),
             'weighted_auc_bootstrap_and_household_parts': t_weighted_auc_boot(rng),
             'same_partition': t_partition(), 'tolerance_compare': t_cmp()}
    out = {'tests': {k: {'pass': v[0], **v[1]} for k, v in tests.items()}, 'all_pass': all(v[0] for v in tests.values()),
           'cpu_s': round(time.process_time() - t0, 2)}
    print(json.dumps(V.jsonable(out), indent=1))
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
