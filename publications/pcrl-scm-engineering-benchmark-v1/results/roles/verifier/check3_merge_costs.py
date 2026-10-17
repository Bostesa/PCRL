"""Independent verifier, Check 3 (merge cost formulas). Invented data only; no real array is read.

Brute force: every plug-in metric is recomputed person by person (explicit expectations over the classes of the teacher
law, or of the fixed shrunk-label law for the guard) on the release before and after a merge, and the change is compared with
  (a) the PROTOCOL.md section 4 formulas, implemented here from per-cell statistics recomputed from scratch, and
  (b) scm_bench.engine, the object under test (pair_deltas, grouped_income_costs, income_groups, occupation_abs_costs,
      guard_abs, Stats/merge_into and the run_path cost accounting).
The brute-force functions never call scm_bench. Output: check3_merge_costs.json (aggregates only).
"""
from __future__ import annotations

import collections
import json
import sys
import time
from fractions import Fraction
from math import floor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scm_bench import engine as E  # noqa: E402  (object under test only)

KAPPA = 20.0
NG = 32
TOL = 1e-12          # absolute; every metric here is an O(1) per-person average
K1, K2 = 2, 5


# ------------------------------------------------------------------ invented populations
def invent(seed, N, k, tie=False):
    r = np.random.default_rng(seed)
    fine = np.concatenate([np.arange(k), r.integers(0, k, N - k)]); r.shuffle(fine)
    a1 = r.gamma(2.0, 1.0, (k, K1)) + 0.3; a2 = r.gamma(1.5, 1.0, (k, K2)) + 0.2
    T1 = np.empty((N, K1)); T2 = np.empty((N, K2))
    for c in range(k):
        m = np.flatnonzero(fine == c)
        T1[m] = r.dirichlet(a1[c], len(m)); T2[m] = r.dirichlet(a2[c], len(m))
    if tie:   # fine cells 1 and 2 get the same t1 rows in the same order -> exactly equal pooled P(income = 1) and mass
        m1, m2 = np.flatnonzero(fine == 1), np.flatnonzero(fine == 2)
        n = min(len(m1), len(m2)); fine[m1[n:]] = 0; fine[m2[n:]] = 0
        T1[m2[:n]] = T1[m1[:n]]
    pc = r.beta(2.0, 2.0, k)
    pi = np.clip(pc[fine] + 0.1 * r.standard_normal(N), 0.02, 0.98)
    sex = (r.random(N) < pi).astype(np.int64)
    u = r.random(N); occ = np.minimum((u[:, None] > np.cumsum(T2, 1)).sum(1), K2 - 1).astype(np.int64)
    return fine, T1, T2, pi, sex, occ


# ------------------------------------------------------------------ person-level brute force (independent of scm_bench)
def bf_groups(cell, T1):
    """PROTOCOL 4: final cells sorted by pooled P(income=1) ascending, ties by cell index (label = smallest fine index);
    g = min(31, floor(32 (M_before + n_c/2) / N)), exact rational arithmetic."""
    N = len(cell); info = []
    for c in np.unique(cell):
        m = cell == c
        info.append((float(np.mean(T1[m, 1])), int(c), int(m.sum())))
    info.sort(key=lambda x: (x[0], x[1]))
    out, M = {}, 0
    for _, c, n in info:
        out[c] = min(NG - 1, floor(Fraction(32) * (Fraction(M) + Fraction(n, 2)) / N)); M += n
    return out


def bf_release(cell, T1, T2, grouped):
    R1 = np.empty_like(T1); R2 = np.empty_like(T2)
    for c in np.unique(cell):
        m = cell == c
        R2[m] = T2[m].mean(0)
        if not grouped:
            R1[m] = T1[m].mean(0)
    if grouped:
        g = bf_groups(cell, T1); gp = np.array([g[int(c)] for c in cell])
        for gg in np.unique(gp):
            m = gp == gg
            R1[m] = T1[m].mean(0)
    return R1, R2


def exp_metrics(T, R):
    """Plug-in expected Brier, NLL and accuracy of release R when y ~ T (teacher law), averaged over persons."""
    N, K = T.shape
    brier = np.zeros(N); nll = np.zeros(N)
    for kk in range(K):
        e = np.zeros(K); e[kk] = 1.0
        brier += T[:, kk] * ((R - e) ** 2).sum(1)
        nll += np.where(T[:, kk] > 0, -T[:, kk] * np.log(np.where(T[:, kk] > 0, R[:, kk], 1.0)), 0.0)
    acc = T[np.arange(N), R.argmax(1)]
    return float(brier.mean()), float(nll.mean()), float(acc.mean())


def bf_ytilde(fine, T2, sex, occ, k):
    """Fixed shrunk label law: y~_{f,s} = (sum_{i in f,s} e_{y_i} + kappa mu_f) / (n_{f,s} + kappa), inherited per person."""
    Yt = np.zeros_like(T2)
    for f in range(k):
        mf = fine == f
        mu_f = T2[mf].mean(0)
        for s in (0, 1):
            m = mf & (sex == s)
            if not m.any():
                continue
            raw = np.zeros(K2)
            for y in occ[m]:
                raw[y] += 1.0
            Yt[m] = (raw + KAPPA * mu_f) / (m.sum() + KAPPA)
    return Yt


def bf_guard(R2, Yt, sex):
    out = []
    for s in (0, 1):
        m = sex == s; val = np.zeros(int(m.sum()))
        for kk in range(K2):
            e = np.zeros(K2); e[kk] = 1.0
            val += Yt[m, kk] * ((R2[m] - e) ** 2).sum(1)
        out.append(float(val.mean()))
    return out


def hb(p):
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def bf_sex(cell, pi):
    """Between-cell variance of the cell-mean propensity, and plug-in I(T; SEX) = H_B(pi_bar) - E_c H_B(pi_c)."""
    pc = np.empty(len(cell))
    for c in np.unique(cell):
        m = cell == c; pc[m] = pi[m].mean()
    pbar = pi.mean()
    return float(((pc - pbar) ** 2).mean()), float(hb(pbar) - hb(pc).mean())


def bf_task_info(cell, T):
    """Plug-in I(T; Y) under the joint law p(c, k) = sum_{i in c} t_ik / N."""
    N = len(cell)
    P = np.array([T[cell == c].sum(0) for c in np.unique(cell)]) / N
    pc = P.sum(1, keepdims=True); pk = P.sum(0, keepdims=True)
    return float(np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0) / (pc * pk)), 0.0)))


def bf_eval(cell, T1, T2, pi, sex, occ, Yt):
    R1g, R2 = bf_release(cell, T1, T2, grouped=True)
    R1c, _ = bf_release(cell, T1, T2, grouped=False)
    o = {}
    o['g_brier1'], o['g_nll1'], o['g_acc1'] = exp_metrics(T1, R1g)
    o['c_brier1'], o['c_nll1'], o['c_acc1'] = exp_metrics(T1, R1c)
    o['brier2'], o['nll2'], o['acc2'] = exp_metrics(T2, R2)
    o['guard0'], o['guard1'] = bf_guard(R2, Yt, sex)
    o['sexvar'], o['sexinfo'] = bf_sex(cell, pi)
    o['task_info'] = bf_task_info(cell, T1) + bf_task_info(cell, T2)
    o['acc2_labels'] = float((R2.argmax(1) == occ).mean())
    return o, R1g, R2


def bf_delta(bef, aft):
    """Brute-force changes in the engine's sign conventions (costs: after - before; accuracy loss: before - after;
    SEX gain / information removed and task information lost: before - after)."""
    return {'brier1': aft['c_brier1'] - bef['c_brier1'], 'nll1': aft['c_nll1'] - bef['c_nll1'], 'acc1': bef['c_acc1'] - aft['c_acc1'],
            'brier2': aft['brier2'] - bef['brier2'], 'nll2': aft['nll2'] - bef['nll2'], 'acc2': bef['acc2'] - aft['acc2'],
            'guard0': aft['guard0'] - bef['guard0'], 'guard1': aft['guard1'] - bef['guard1'],
            'sexvar': bef['sexvar'] - aft['sexvar'], 'sexinfo': bef['sexinfo'] - aft['sexinfo'],
            'task_info': bef['task_info'] - aft['task_info'],
            'grouped_brier1': aft['g_brier1'] - bef['g_brier1'], 'grouped_nll1': aft['g_nll1'] - bef['g_nll1'],
            'grouped_acc1': bef['g_acc1'] - aft['g_acc1']}


# ------------------------------------------------------------------ PROTOCOL.md section 4 formulas (own implementation)
def cell_stats(cell, T1, T2, pi, sex, Yt):
    st = {}
    for c in np.unique(cell):
        m = cell == c; n = float(m.sum())
        st[int(c)] = {'n': n, 'S1': T1[m].sum(0), 'S2': T2[m].sum(0), 'SP': float(pi[m].sum()),
                      'ns': [float((m & (sex == s)).sum()) for s in (0, 1)],
                      'Y': [Yt[m & (sex == s)].sum(0) for s in (0, 1)]}   # = sum_{f in c} n_fs y~_fs
    return st


def kl(p, q):
    return float(np.sum(np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(q)), 0.0)))


def klb(p, q):
    return kl(np.array([p, 1 - p]), np.array([q, 1 - q]))


def proto_deltas(st, a, b, N, Ns):
    A, B = st[a], st[b]; na, nb = A['n'], B['n']; nm = na + nb; o = {}
    for t in ('1', '2'):
        ma, mb = A['S' + t] / na, B['S' + t] / nb; mm = (A['S' + t] + B['S' + t]) / nm
        o['brier' + t] = na * nb / nm * float(np.sum((ma - mb) ** 2)) / N                       # Ward
        o['nll' + t] = (na * kl(ma, mm) + nb * kl(mb, mm)) / N                                    # KL
        o['acc' + t] = (na * ma.max() + nb * mb.max() - nm * mm.max()) / N                         # n max mu
    ma, mb = A['S2'] / na, B['S2'] / nb; mm = (A['S2'] + B['S2']) / nm
    for s in (0, 1):
        nas, nbs = A['ns'][s], B['ns'][s]; Ya, Yb = A['Y'][s], B['Y'][s]
        o[f'guard{s}'] = ((nas + nbs) * mm @ mm - nas * ma @ ma - nbs * mb @ mb
                          - 2 * (mm @ (Ya + Yb) - ma @ Ya - mb @ Yb)) / Ns[s]
    pa, pb = A['SP'] / na, B['SP'] / nb; pm = (A['SP'] + B['SP']) / nm
    o['sexvar'] = na * nb / nm * (pa - pb) ** 2 / N
    o['sexinfo'] = (na * klb(pa, pm) + nb * klb(pb, pm)) / N
    o['task_info'] = o['nll1'] + o['nll2']
    return o


def proto_grouped(st, N):
    """Grouped income release from per-cell sums: group vector = pooled t1 of the group's members. Returns the
    non-constant parts (-sum_g G|v|^2, -sum_g SG.log v, -sum_g SG[argmax v]) / N; changes equal the exact cost changes."""
    info = sorted(((v['S1'][1] / v['n'], c, v['n']) for c, v in st.items()), key=lambda x: (x[0], x[1]))
    grp, M = {}, 0
    for _, c, n in info:
        grp[c] = min(NG - 1, floor(Fraction(32) * (Fraction(int(M)) + Fraction(int(n), 2)) / N)); M += n
    G = collections.defaultdict(float); SG = collections.defaultdict(lambda: np.zeros(K1))
    for c, v in st.items():
        G[grp[c]] += v['n']; SG[grp[c]] = SG[grp[c]] + v['S1']
    br = nl = ac = 0.0
    for g in G:
        vg = SG[g] / G[g]
        br -= G[g] * float(vg @ vg); nl -= float(np.sum(SG[g] * np.log(vg))); ac -= float(SG[g][vg.argmax()])
    return np.array([br, nl, ac]) / N, grp


# ------------------------------------------------------------------ comparisons
class Acc:
    def __init__(self):
        self.d = collections.defaultdict(lambda: {'n': 0, 'proto_vs_bf': None, 'engine_vs_bf': 0.0, 'engine_vs_proto': None,
                                                  'max_abs_ref': 0.0, 'max_rel_engine_vs_bf': 0.0})

    def add(self, name, bf, pr=None, en=None):
        r = self.d[name]; r['n'] += 1; r['max_abs_ref'] = max(r['max_abs_ref'], abs(bf))
        if pr is not None:
            r['proto_vs_bf'] = max(r['proto_vs_bf'] or 0.0, abs(pr - bf))
        if en is not None:
            r['engine_vs_bf'] = max(r['engine_vs_bf'], abs(en - bf))
            if abs(bf) > 1e-6:
                r['max_rel_engine_vs_bf'] = max(r['max_rel_engine_vs_bf'], abs(en - bf) / abs(bf))
        if pr is not None and en is not None:
            r['engine_vs_proto'] = max(r['engine_vs_proto'] or 0.0, abs(en - pr))

    def out(self):
        res = {}
        for k, r in sorted(self.d.items()):
            worst = max(x for x in (r['proto_vs_bf'], r['engine_vs_bf'], r['engine_vs_proto']) if x is not None)
            res[k] = {**r, 'PASS': bool(worst <= TOL)}
        return res


def run_instance(seed, N, k, steps, pairs_per_step, tie, acc, extra):
    fine, T1, T2, pi, sex, occ = invent(seed, N, k, tie)
    Yt = bf_ytilde(fine, T2, sex, occ, k)
    Ns = [float((sex == s).sum()) for s in (0, 1)]
    t1_self, t2_self = exp_metrics(T1, T1), exp_metrics(T2, T2)
    st = E.Stats(fine, k, T1, T2, pi, sex, occ); alive = st.n > 0
    # fine-cell statistics of the engine vs brute force (fixed law stored as n_fs * y~_fs)
    for s in (0, 1):
        ref = np.zeros((k, K2)); np.add.at(ref, fine[sex == s], Yt[sex == s])
        extra['stats_Y_fixed_law_max_abs'] = max(extra.get('stats_Y_fixed_law_max_abs', 0.0), float(np.abs(st.Y[s] - ref).max()))
        extra['stats_ns_mismatch'] = extra.get('stats_ns_mismatch', 0) + int((st.ns[s] != np.bincount(fine[sex == s], minlength=k)).sum())
    if tie:
        p1 = st.S1[:, 1] / st.n; g_e = E.income_groups(st.n, st.S1, alive); g_b = bf_groups(fine, T1)
        extra.setdefault('tie_tests', []).append({'seed': seed, 'exact_tie_engine_p1': bool(p1[1] == p1[2]), 'equal_mass': bool(st.n[1] == st.n[2]),
                                                  'engine_groups_cells_1_2': [int(g_e[1]), int(g_e[2])], 'bf_groups_cells_1_2': [g_b[1], g_b[2]],
                                                  'tie_straddles_groups (tie-break matters)': bool(g_b[1] != g_b[2])})
    part = np.arange(k); r = np.random.default_rng(seed + 1000)
    for step in range(steps):
        live = np.flatnonzero(alive); cell = part[fine]
        if len(live) < 3:
            break
        bef, R1g, R2 = bf_eval(cell, T1, T2, pi, sex, occ, Yt)
        cs = cell_stats(cell, T1, T2, pi, sex, Yt)
        # absolute engine costs vs brute force (relative to the teacher's own plug-in metrics)
        inc_abs, grp = E.grouped_income_costs(st, alive)
        bf_inc_abs = [bef['g_brier1'] - t1_self[0], bef['g_nll1'] - t1_self[1], t1_self[2] - bef['g_acc1']]
        for nm_, e_, b_ in zip(('abs_grouped_income_brier', 'abs_grouped_income_nll', 'abs_grouped_income_acc'), inc_abs, bf_inc_abs):
            acc.add(nm_, b_, None, float(e_))
        occ_abs = E.occupation_abs_costs(st, alive)
        bf_occ_abs = [bef['brier2'] - t2_self[0], bef['nll2'] - t2_self[1], t2_self[2] - bef['acc2']]
        for nm_, e_, b_ in zip(('abs_occupation_brier', 'abs_occupation_nll', 'abs_occupation_acc'), occ_abs, bf_occ_abs):
            acc.add(nm_, b_, None, float(e_))
        g_abs = E.guard_abs(st, alive)
        acc.add('abs_guard0', bef['guard0'], None, float(g_abs[0])); acc.add('abs_guard1', bef['guard1'], None, float(g_abs[1]))
        # group labels: engine vs brute force vs protocol implementation
        bg = bf_groups(cell, T1); _, pg = proto_grouped(cs, N)
        extra['group_label_checks'] = extra.get('group_label_checks', 0) + len(live)
        extra['group_label_mismatch_engine_vs_bf'] = extra.get('group_label_mismatch_engine_vs_bf', 0) + int(sum(grp[c] != bg[int(c)] for c in live))
        extra['group_label_mismatch_proto_vs_bf'] = extra.get('group_label_mismatch_proto_vs_bf', 0) + int(sum(pg[int(c)] != bg[int(c)] for c in live))
        pgb, _ = proto_grouped(cs, N)
        # candidate pairs: the two largest live cells (multi-merged after a few steps) plus random pairs
        cands = {tuple(sorted(int(x) for x in live[np.argsort(-st.n[live], kind='stable')[:2]]))}
        while len(cands) < pairs_per_step:
            a, b = sorted(int(x) for x in r.choice(live, 2, replace=False)); cands.add((a, b))
        cands = sorted(cands)
        for a, b in cands:
            part2 = part.copy(); part2[part2 == b] = a; cell2 = part2[fine]
            aft, _, _ = bf_eval(cell2, T1, T2, pi, sex, occ, Yt)
            dbf = bf_delta(bef, aft)
            dpr = proto_deltas(cs, a, b, N, Ns)
            den = {kk: float(v[0]) for kk, v in E.pair_deltas(st, np.array([a]), np.array([b])).items()}
            den['task_info'] = den['nll1'] + den['nll2']
            for key in ('brier1', 'nll1', 'acc1', 'brier2', 'nll2', 'acc2', 'guard0', 'guard1', 'sexvar', 'sexinfo', 'task_info'):
                acc.add('delta_' + key, dbf[key], dpr[key], den[key])
            trial = st.copy(); E.merge_into(trial, a, b); al = alive.copy(); al[b] = False
            inc_aft, _ = E.grouped_income_costs(trial, al)
            pga, _ = proto_grouped(cell_stats(cell2, T1, T2, pi, sex, Yt), N)
            for j, key in enumerate(('grouped_brier1', 'grouped_nll1', 'grouped_acc1')):
                acc.add('delta_' + key, dbf[key], float(pga[j] - pgb[j]), float(inc_aft[j] - inc_abs[j]))
            # conventions: grouped vs cell-level income change; negative guard / grouped changes
            for j, key in enumerate(('brier1', 'nll1', 'acc1')):
                extra['n_pairs_income'] = extra.get('n_pairs_income', 0) + (j == 0)
                if abs(dbf['grouped_' + key] - dbf[key]) > 1e-12:
                    extra[f'grouped_ne_cell_{key}'] = extra.get(f'grouped_ne_cell_{key}', 0) + 1
                if dbf['grouped_' + key] < -1e-15:
                    extra[f'grouped_negative_{key}'] = extra.get(f'grouped_negative_{key}', 0) + 1
            for s in (0, 1):
                if dbf[f'guard{s}'] < 0:
                    extra['guard_negative_changes'] = extra.get('guard_negative_changes', 0) + 1
            # contrast 1: label-based occupation accuracy (not the protocol's plug-in metric)
            lab = bef['acc2_labels'] - aft['acc2_labels']
            extra['label_acc_vs_plugin_max_abs'] = max(extra.get('label_acc_vs_plugin_max_abs', 0.0), abs(lab - dbf['acc2']))
            # contrast 2: guard with the shrinkage re-estimated at the merged cell (a non-fixed law), first step only
            if step == 0:
                Yt2 = Yt.copy(); mm = cell2 == a; mu_m = T2[mm].mean(0)
                for s in (0, 1):
                    q = mm & (sex == s)
                    if q.any():
                        raw = np.bincount(occ[q], minlength=K2).astype(float)
                        Yt2[q] = (raw + KAPPA * mu_m) / (q.sum() + KAPPA)
                R2a = bf_release(cell2, T1, T2, grouped=False)[1]
                g_nf = bf_guard(R2a, Yt2, sex)
                for s in (0, 1):
                    extra['guard_nonfixed_law_vs_formula_max_abs'] = max(extra.get('guard_nonfixed_law_vs_formula_max_abs', 0.0),
                                                                         abs((g_nf[s] - bef[f'guard{s}']) - dpr[f'guard{s}']))
        a, b = cands[int(r.integers(len(cands)))]
        E.merge_into(st, a, b); alive[b] = False; part[part == b] = a
        extra['merges_applied'] = extra.get('merges_applied', 0) + 1
        extra['max_merged_cell_size_fine_cells'] = max(extra.get('max_merged_cell_size_fine_cells', 1), int(np.bincount(part).max()))
    extra['pairs_checked'] = acc.d['delta_brier2']['n']


def merged_stats(st, a, b):
    out = {c: v for c, v in st.items() if c not in (a, b)}
    A, B = st[a], st[b]
    out[a] = {'n': A['n'] + B['n'], 'S1': A['S1'] + B['S1'], 'S2': A['S2'] + B['S2'], 'SP': A['SP'] + B['SP'],
              'ns': [A['ns'][s] + B['ns'][s] for s in (0, 1)], 'Y': [A['Y'][s] + B['Y'][s] for s in (0, 1)]}
    return out


def my_greedy(st, kind, N, Ns, beta=None, gamma=None, target=None):
    """Exact greedy of PROTOCOL 5 with the declared conventions (12.9): all live pairs scored each step, priority order with
    ties to the lowest (a, b); acceptance needs every cumulative cost within budget (cell-level occupation and guard, exact
    grouped income); R_k = max(B_k - spent_k, 1e-15); negative per-merge costs clipped to 0 in the priority; +1e-12 in the
    SCM denominator. Same 1e-15 comparison slack as the engine."""
    keys = ('brier1', 'nll1', 'acc1', 'brier2', 'nll2', 'acc2', 'guard0', 'guard1')
    capped = kind != 'IBSI'
    B = ({'brier1': 0.005 * beta, 'nll1': 0.010 * beta, 'acc1': 0.005, 'brier2': 0.005 * beta, 'nll2': 0.010 * beta, 'acc2': 0.005,
          'guard0': 0.005 * beta, 'guard1': 0.005 * beta} if capped else None)
    spent = {x: 0.0 for x in keys}; g0, _ = proto_grouped(st, N); seq = []
    while True:
        live = sorted(st)
        if (target is not None and len(live) <= target) or len(live) < 2:
            break
        cands = []
        R = {x: max(B[x] - spent[x], 1e-15) for x in keys} if capped else None
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                d = proto_deltas(st, a, b, N, Ns)
                if not capped:
                    key = d['nll1'] + d['nll2'] - gamma * d['sexinfo']
                else:
                    norm = sum(max(d[x], 0.0) / R[x] for x in keys)
                    key = -(d['sexvar'] / (norm + 1e-12)) if kind in ('SCM', 'SHUF') else norm
                cands.append((key, a, b, d))
        cands.sort(key=lambda z: (z[0], z[1], z[2]))
        took = None
        for _, a, b, d in cands:
            if not capped:
                took = (a, b, d, None); break
            if any(spent[x] + d[x] > B[x] + 1e-15 for x in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1')):
                continue
            gm, _ = proto_grouped(merged_stats(st, a, b), N); dinc = gm - g0
            if dinc[0] > B['brier1'] + 1e-15 or dinc[1] > B['nll1'] + 1e-15 or dinc[2] > B['acc1'] + 1e-15:
                continue
            took = (a, b, d, dinc); break
        if took is None:
            break
        a, b, d, dinc = took
        if capped:
            for x in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1'):
                spent[x] += d[x]
            spent['brier1'], spent['nll1'], spent['acc1'] = (float(v) for v in dinc)
        st = merged_stats(st, a, b); seq.append((a, b))
    return seq


def run_path_checks(seed, N, k):
    fine, T1, T2, pi, sex, occ = invent(seed, N, k)
    Yt = bf_ytilde(fine, T2, sex, occ, k)
    st0 = E.Stats(fine, k, T1, T2, pi, sex, occ)
    base, _, _ = bf_eval(fine, T1, T2, pi, sex, occ, Yt)
    out = {}
    K = None
    for name, kind, kw in (('SCM_b1.0', 'SCM', {'beta': 1.0}), ('SCM_b0.5', 'SCM', {'beta': 0.5}), ('TWIN_b1.0', 'TWIN', {'beta': 1.0}),
                           ('IBSI_g1.0', 'IBSI', {'gamma': 1.0}), ('SHUF_b1.0', 'SHUF', {'beta': 1.0})):
        kw = dict(kw)
        if kind in ('TWIN', 'IBSI', 'SHUF'):
            kw['target'] = K
        sin = st0
        if kind == 'SHUF':
            sin = st0.copy(); perm = np.random.default_rng(seed + 7).permutation(k); rate = st0.SP / st0.n; sin.SP = st0.n * rate[perm]
        final, grp, st_f, al_f, rec = E.run_path(sin, kind, **kw)
        if name == 'SCM_b1.0':
            K = rec['final_cells']
        # independent greedy replay (own protocol formulas, own sufficient statistics)
        cs0 = cell_stats(fine, T1, T2, pi, sex, Yt)
        if kind == 'SHUF':
            for c in cs0:
                cs0[c]['SP'] = float(sin.SP[c])
        seq = my_greedy(cs0, kind, N, [float((sex == s_).sum()) for s_ in (0, 1)], beta=kw.get('beta'), gamma=kw.get('gamma'),
                        target=kw.get('target'))
        par = np.arange(k)
        for a_, b_ in seq:
            par[b_] = a_

        def root(x):
            while par[x] != x:
                x = par[x]
            return x
        my_final = np.array([root(i) for i in range(k)])
        cell = final[fine]
        aft, R1g, R2 = bf_eval(cell, T1, T2, pi, sex, occ, Yt)
        sp = rec['spent_plugin_vs_C0']
        bf_inc = [aft['g_brier1'] - base['g_brier1'], aft['g_nll1'] - base['g_nll1'], base['g_acc1'] - aft['g_acc1']]
        bf_occ = [aft['brier2'] - base['brier2'], aft['nll2'] - base['nll2'], base['acc2'] - aft['acc2']]
        bf_g = [aft['guard0'] - base['guard0'], aft['guard1'] - base['guard1']]
        err = max(np.max(np.abs(np.array(sp['income_grouped_brier_nll_acc']) - bf_inc)),
                  np.max(np.abs(np.array(sp['occupation_brier_nll_acc']) - bf_occ)),
                  np.max(np.abs(np.array(sp['guard0_guard1']) - bf_g)))
        r = {'final_cells': rec['final_cells'], 'steps': rec['steps'], 'stop_reason': rec['stop_reason'],
             'reported_vs_bruteforce_max_abs': float(err),
             'independent_greedy_same_merge_map': bool(np.array_equal(my_final, final)), 'independent_greedy_steps': len(seq)}
        if kind != 'IBSI':
            # cumulative cell-level spending (sum of accepted per-step deltas) vs brute force of the final release
            cum = [sp['brier2'], sp['nll2'], sp['acc2'], sp['guard0'], sp['guard1'], sp['brier1'], sp['nll1'], sp['acc1']]
            r['cumulative_spent_vs_bruteforce_max_abs'] = float(np.max(np.abs(np.array(cum) - np.array(bf_occ + bf_g + bf_inc))))
            B = E.budgets(kw['beta'])
            bfv = dict(zip(('brier1', 'nll1', 'acc1', 'brier2', 'nll2', 'acc2', 'guard0', 'guard1'), bf_inc + bf_occ + bf_g))
            prot_B = {'brier1': 0.005 * kw['beta'], 'nll1': 0.010 * kw['beta'], 'acc1': 0.005, 'brier2': 0.005 * kw['beta'],
                      'nll2': 0.010 * kw['beta'], 'acc2': 0.005, 'guard0': 0.005 * kw['beta'], 'guard1': 0.005 * kw['beta']}
            r['engine_budgets_equal_protocol'] = bool(all(abs(B[x] - prot_B[x]) < 1e-18 for x in prot_B))
            r['within_budgets_bruteforce'] = bool(all(bfv[x] <= prot_B[x] + 1e-12 for x in prot_B))
            r['budget_use_bruteforce'] = {x: round(bfv[x] / prot_B[x], 4) for x in prot_B}
        tab = E.tables(st_f, al_f, final, grp); rel = E.release(tab, fine)
        r['release_vs_bruteforce_max_abs'] = float(max(np.abs(rel['income'] - R1g).max(), np.abs(rel['occupation'] - R2).max()))
        bg = bf_groups(cell, T1)
        r['group_label_mismatch'] = int(sum(grp[c] != bg[int(c)] for c in np.unique(cell)))
        r['PASS'] = bool(err <= TOL and r['independent_greedy_same_merge_map'] and len(seq) == rec['steps'] and r.get('cumulative_spent_vs_bruteforce_max_abs', 0.0) <= TOL and r['release_vs_bruteforce_max_abs'] <= TOL
                         and r['group_label_mismatch'] == 0 and r.get('within_budgets_bruteforce', True) and r.get('engine_budgets_equal_protocol', True))
        out[name] = r
    return out


def main():
    t0 = time.process_time()
    acc = Acc(); extra = {}
    instances = [  # (seed, N, fine cells k, merge steps, pairs per step, deliberate exact tie in pooled P(income=1))
        (11, 1600, 24, 14, 5, True),     # fewer cells than income groups: every cell its own group; tie straddles groups
        (12, 2400, 60, 30, 5, False),
        (13, 3000, 90, 30, 4, False),
        (14, 1200, 40, 25, 5, True),
    ]
    for inst in instances:
        run_instance(*inst, acc, extra)
    rp = {f'seed{s}_N{N}_k{k}': run_path_checks(s, N, k) for s, N, k in ((21, 2400, 60), (22, 3000, 48))}
    formulas = acc.out()
    groups_ok = extra['group_label_mismatch_engine_vs_bf'] == 0 and extra['group_label_mismatch_proto_vs_bf'] == 0
    stats_ok = extra['stats_Y_fixed_law_max_abs'] <= TOL and extra['stats_ns_mismatch'] == 0
    verdict = {
        'Ward_Brier_occupation (delta_brier2)': formulas['delta_brier2']['PASS'],
        'Ward_Brier_income_cell_level (delta_brier1)': formulas['delta_brier1']['PASS'],
        'KL_NLL_occupation (delta_nll2)': formulas['delta_nll2']['PASS'],
        'KL_NLL_income_cell_level (delta_nll1)': formulas['delta_nll1']['PASS'],
        'accuracy_loss_occupation (delta_acc2)': formulas['delta_acc2']['PASS'],
        'accuracy_loss_income_cell_level (delta_acc1)': formulas['delta_acc1']['PASS'],
        'grouped_income_costs (delta_grouped_*, abs_grouped_*, group labels)': bool(
            all(formulas[x]['PASS'] for x in formulas if 'grouped' in x) and groups_ok),
        'fixed_law_within_SEX_guard (delta_guard0/1, abs_guard0/1, stored Y)': bool(
            all(formulas[x]['PASS'] for x in ('delta_guard0', 'delta_guard1', 'abs_guard0', 'abs_guard1')) and stats_ok),
        'SEX_variance_gain_G (delta_sexvar)': formulas['delta_sexvar']['PASS'],
        'SEX_information_removed (delta_sexinfo)': formulas['delta_sexinfo']['PASS'],
        'IBSI_task_information_lost = dNLL_income + dNLL_occupation, cell level (delta_task_info)': formulas['delta_task_info']['PASS'],
        'absolute_occupation_costs (abs_occupation_*)': bool(all(formulas[x]['PASS'] for x in formulas if x.startswith('abs_occupation'))),
        'run_path_cost_accounting_and_budgets': bool(all(v['PASS'] for d in rp.values() for v in d.values())),
    }
    res = {'schema': 'SCM1-verifier-check3-v1', 'data': 'invented only (no real array read)', 'tolerance_abs': TOL,
           'instances': [dict(zip(('seed', 'N', 'fine_cells', 'merge_steps', 'pairs_per_step', 'exact_tie'), i)) for i in instances],
           'formulas': formulas, 'run_path': rp, 'extra': extra, 'verdict': verdict,
           'CHECK3_PASS': bool(all(verdict.values())), 'cpu_s': round(time.process_time() - t0, 2)}
    (HERE / 'check3_merge_costs.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    print(json.dumps({'verdict': verdict, 'CHECK3_PASS': res['CHECK3_PASS'], 'extra': extra, 'cpu_s': res['cpu_s']}, indent=1, default=float))
    for kname, v in formulas.items():
        f = lambda x: 'n/a' if x is None else f'{x:.2e}'
        print(f"{kname:34s} n={v['n']:4d} pr-bf={f(v['proto_vs_bf'])} en-bf={f(v['engine_vs_bf'])} en-pr={f(v['engine_vs_proto'])} "
              f"|ref|max={v['max_abs_ref']:.2e} rel={v['max_rel_engine_vs_bf']:.2e} {'PASS' if v['PASS'] else 'FAIL'}")
    print(json.dumps(rp, indent=1, default=float))


if __name__ == '__main__':
    main()
