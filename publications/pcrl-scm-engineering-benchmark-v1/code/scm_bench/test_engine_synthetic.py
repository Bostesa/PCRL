"""Synthetic checks of the SCM engine (invented data only): cost identities against person-level brute force, exact greedy
against a naive re-implementation, bit-identical outputs within cells, IBSI first-merge optimality, k-means determinism."""
import itertools
import json

import numpy as np

from scm_bench import engine as E


def invented(n=1500, seed=0):
    r = np.random.default_rng(seed)
    T1 = r.dirichlet([2, 3], n); T2 = r.dirichlet([1, 2, 1, 1, 2], n); pi = r.beta(2, 2, n)
    sex = (r.random(n) < pi).astype(int); occ = np.array([r.choice(5, p=p) for p in T2])
    return T1, T2, pi, sex, occ


def person_costs(T1, T2, sex, occ, fine, final_of_fine, grp_of_final, st0):
    """Person-level brute force of the absolute plug-in costs of a release (income grouped, occupation by final cell)."""
    N = len(T1); cell = final_of_fine[fine]; g = grp_of_final[cell]
    r1 = np.zeros_like(T1); r2 = np.zeros_like(T2)
    for c in np.unique(cell):
        r2[cell == c] = T2[cell == c].mean(0)
    for gg in np.unique(g):
        r1[g == gg] = T1[g == gg].mean(0)
    def costs(T, R):
        return np.array([((T - R) ** 2).sum(1).mean(), (E.xlogy(T, T) - E.xlogy(T, R)).sum(1).mean(),
                         (T.max(1) - T[np.arange(N), R.argmax(1)]).mean()])
    # guard under the fixed fine-cell shrunk label law
    mu_f = st0.S2 / st0.n[:, None]; out_g = []
    for s in (0, 1):
        m = sex == s; ns = st0.ns[s]
        raw = np.zeros_like(st0.S2); np.add.at(raw, fine[m], np.eye(5)[occ[m]])
        ytil = np.where(ns[:, None] > 0, (raw + E.KAPPA * mu_f) / (ns + E.KAPPA)[:, None], 0.0)
        y_i = ytil[fine[m]]
        out_g.append(((r2[m] ** 2).sum(1) - 2 * (r2[m] * y_i).sum(1) + 1).mean())
    return costs(T1, r1), costs(T2, r2), np.array(out_g)


def naive_greedy(st0, kind, beta=None, gamma=None, target=None):
    """Reference: same rules, plain Python loops over pairs, no prefilter."""
    st = st0.copy(); alive = st.n > 0; parent = np.arange(st.k)
    B = E.budgets(beta) if kind != 'IBSI' else None
    inc0, _ = E.grouped_income_costs(st, alive); spent = {k: 0.0 for k in E.BUDGET_KEYS}; seq = []
    while True:
        live = list(np.flatnonzero(alive))
        if target is not None and len(live) <= target or len(live) < 2:
            break
        cands = []
        for a, b in itertools.combinations(live, 2):
            d = {k: float(v[0]) for k, v in E.pair_deltas(st, np.array([a]), np.array([b])).items()}
            if kind == 'IBSI':
                cands.append((d['nll1'] + d['nll2'] - gamma * d['sexinfo'], a, b, d))
            else:
                R = {k: max(B[k] - spent[k], 1e-15) for k in E.BUDGET_KEYS}
                norm = sum(max(d[k], 0.0) / R[k] for k in E.BUDGET_KEYS)
                p = d['sexvar'] / (norm + 1e-12) if kind == 'SCM' else -norm
                cands.append((-p, a, b, d))
        cands.sort(key=lambda x: (x[0], x[1], x[2]))
        took = None
        for _, a, b, d in cands:
            if kind != 'IBSI':
                if any(spent[k] + d[k] > B[k] + 1e-15 for k in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1')):
                    continue
                tr = st.copy(); E.merge_into(tr, a, b); al = alive.copy(); al[b] = False
                inc, _ = E.grouped_income_costs(tr, al); di = inc - inc0
                if di[0] > B['brier1'] + 1e-15 or di[1] > B['nll1'] + 1e-15 or di[2] > B['acc1'] + 1e-15:
                    continue
                took = (a, b, d, di); break
            took = (a, b, d, None); break
        if took is None:
            break
        a, b, d, di = took
        if kind != 'IBSI':
            for k in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1'):
                spent[k] += d[k]
        E.merge_into(st, a, b); alive[b] = False; parent[b] = a; seq.append((a, b))
    return seq


def main():
    res = {}
    T1, T2, pi, sex, occ = invented()
    mu1, mu2, fine, info = E.kl_kmeans(T1, T2, 14, seed=5)
    k = info['occupied_cells']
    st0 = E.Stats(fine, k, T1, T2, pi, sex, occ)
    # k-means: deterministic and nearest-by-KL
    _, _, fine_b, _ = E.kl_kmeans(T1, T2, 14, seed=5)
    res['kmeans_deterministic'] = bool(np.array_equal(fine, fine_b))
    mu1c = st0.S1 / st0.n[:, None]; mu2c = st0.S2 / st0.n[:, None]
    res['kmeans_assignment_is_kl_nearest'] = bool(np.array_equal(E.kl_assign(T1, T2, mu1c, mu2c), fine))
    # absolute costs of C0 vs person-level brute force
    alive = st0.n > 0; inc, grp = E.grouped_income_costs(st0, alive)
    b1, b2, bg = person_costs(T1, T2, sex, occ, fine, np.arange(k), grp, st0)
    res['C0_income_grouped_costs_err'] = float(np.max(np.abs(inc - b1)))
    res['C0_occupation_costs_err'] = float(np.max(np.abs(E.occupation_abs_costs(st0, alive) - b2)))
    res['C0_guard_err'] = float(np.max(np.abs(E.guard_abs(st0, alive) - bg)))
    # pair deltas vs after - before (cell level; guard; SEX)
    worst = 0.0
    for a, b in [(0, 1), (2, 9), (4, 13), (6, 7)]:
        d = E.pair_deltas(st0, np.array([a]), np.array([b]))
        tr = st0.copy(); E.merge_into(tr, a, b); al = alive.copy(); al[b] = False
        occ_d = E.occupation_abs_costs(tr, al) - E.occupation_abs_costs(st0, alive)
        g_d = E.guard_abs(tr, al) - E.guard_abs(st0, alive)
        worst = max(worst, abs(occ_d[0] - d['brier2'][0]), abs(occ_d[1] - d['nll2'][0]), abs(occ_d[2] - d['acc2'][0]),
                    abs(g_d[0] - d['guard0'][0]), abs(g_d[1] - d['guard1'][0]))
    res['pair_delta_identity_err'] = float(worst)
    # exact greedy == naive greedy (same merge sequence) for SCM, TWIN, IBSI
    for kind, kw in (('SCM', {'beta': 1.0}), ('TWIN', {'beta': 1.0, 'target': 6}), ('IBSI', {'gamma': 1.0, 'target': 6})):
        final, grp_f, st_f, al_f, rec = E.run_path(st0, kind, **kw)
        ref = naive_greedy(st0, kind, **kw)
        # replay ref sequence to a final map
        par = np.arange(k)
        for a, b in ref:
            par[b] = a
        def find(x):
            while par[x] != x:
                x = par[x]
            return x
        ref_final = np.array([find(i) for i in range(k)])
        res[f'{kind}_matches_naive_greedy'] = bool(np.array_equal(final, ref_final)) and rec['steps'] == len(ref)
        res[f'{kind}_final_cells'] = rec['final_cells']
        if kind != 'IBSI':
            B = E.budgets(kw['beta']); sp = rec['spent_plugin_vs_C0']
            res[f'{kind}_within_budgets'] = bool(all(sp[key] <= B[key] + 1e-12 for key in E.BUDGET_KEYS))
        # spent vs brute-force recomputation of the final release
        tab = E.tables(st_f, al_f, final, grp_f)
        b1f, b2f, bgf = person_costs(T1, T2, sex, occ, fine, final, grp_f, st0)
        res[f'{kind}_reported_vs_bruteforce_err'] = float(max(np.max(np.abs(np.array(rec['spent_plugin_vs_C0']['income_grouped_brier_nll_acc']) - (b1f - b1))),
                                                             np.max(np.abs(np.array(rec['spent_plugin_vs_C0']['occupation_brier_nll_acc']) - (b2f - b2))),
                                                             np.max(np.abs(np.array(rec['spent_plugin_vs_C0']['guard0_guard1']) - (bgf - bg)))))
        # bit-identical outputs within final cells
        out = E.release(tab, fine); cell = final[fine]
        same = True
        for c in np.unique(cell):
            m = cell == c
            same &= bool((out['occupation'][m] == out['occupation'][m][0]).all() and (out['income'][m] == out['income'][m][0]).all())
        res[f'{kind}_bit_identical_within_cells'] = same
    # IBSI first merge minimises the published score over all pairs
    I, J = np.triu_indices(k, 1); d = E.pair_deltas(st0, I, J); sc = d['nll1'] + d['nll2'] - 1.0 * d['sexinfo']
    final, _, _, _, rec = E.run_path(st0, 'IBSI', gamma=1.0, max_steps=1)
    merged = [i for i in range(k) if final[i] != i]
    res['IBSI_first_merge_is_argmin'] = bool(merged == [int(J[np.argmin(sc)])] and final[merged[0]] == int(I[np.argmin(sc)]))
    errs = [v for kk, v in res.items() if kk.endswith('_err')]
    flags = [v for kk, v in res.items() if isinstance(v, bool)]
    res['all_ok'] = bool(all(e < 1e-12 for e in errs) and all(flags))
    print(json.dumps(res, indent=1))
    assert res['all_ok']


if __name__ == '__main__':
    main()
