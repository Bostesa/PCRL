"""Math reviewer's data-free checks for PROTOCOL.md section 4 (invented numbers only; no real-person array is read).

Run from the worktree root:  <repository>/.venv/bin/python -B results/pcrl_scm_engineering_benchmark_v1/roles/math_reviewer/math_checks.py
Writes MATH_CHECKS.json next to this file.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = {}
L = lambda x: np.log(np.maximum(x, 1e-300))


# ---------- plug-in metrics under the teacher law (truth = t_i), brute force over persons ----------
def bf_metrics(t, q):
    """t: (n, K) teacher probabilities (truth); q: (n, K) released vectors; decision = argmax q (lowest index)."""
    brier = ((q ** 2).sum(1) - 2 * (q * t).sum(1) + 1).sum()          # sum_i E_{y~t_i} ||q_i - e_y||^2
    nll = -(t * L(q)).sum()                                          # sum_i E_{y~t_i} [-log q_{i,y}]
    acc = t[np.arange(len(t)), q.argmax(1)].sum()                    # sum_i P_{t_i}(y = argmax q_i)
    return np.array([brier, nll, acc])


def H(p):
    return -(np.where(p > 0, p * L(p), 0.0)).sum(-1)


def kl(p, q):
    return (np.where(p > 0, p * (L(p) - L(q)), 0.0)).sum(-1)


# ---------- 1a. occupation cell-level identities (Ward / KL / accuracy) ----------
rng = np.random.default_rng(1)
worst = 0.0
for rep in range(300):
    K = 5; n = rng.integers(2, 40); t = rng.dirichlet(np.full(K, rng.choice([0.3, 1.0, 5.0])), size=n)
    if rep % 7 == 0:
        t[0] = np.eye(K)[0]                                           # one-hot row (exact zeros)
    cut = rng.integers(1, n); A, B = t[:cut], t[cut:]; N = 1000.0
    na, nb = len(A), len(B); nm = na + nb; ma, mb = A.mean(0), B.mean(0); mm = t.mean(0)
    before = bf_metrics(A, np.tile(ma, (na, 1))) + bf_metrics(B, np.tile(mb, (nb, 1)))
    after = bf_metrics(t, np.tile(mm, (nm, 1)))
    d_bf = (after - before) / N
    d_cf = np.array([na * nb / nm * ((ma - mb) ** 2).sum(), na * kl(ma, mm) + nb * kl(mb, mm),
                     -(na * ma.max() + nb * mb.max() - nm * mm.max())]) / N   # accuracy as a change (negative = loss)
    worst = max(worst, np.abs(d_bf - d_cf).max())
OUT['occupation_cell_identities_max_abs_error_300_merges'] = worst


# ---------- 1b. grouped income costs from per-cell sums ----------
def grouping(cells_n, cells_s1, n_groups, N):
    """Protocol rule: sort cells by pooled P(income=1) ascending (ties by index), midpoint mass rule."""
    p1 = cells_s1[:, 1] / cells_n
    order = sorted(range(len(cells_n)), key=lambda c: (p1[c], c))
    g = np.empty(len(cells_n), dtype=int); M = 0.0
    for c in order:
        g[c] = min(n_groups - 1, int(np.floor(n_groups * (M + cells_n[c] / 2) / N))); M += cells_n[c]
    return g


def grouped_costs_from_sums(cells_n, cells_s1, g, N):
    """Exact plug-in (Brier, NLL, accuracy) of the group-pooled income release from per-cell sums only."""
    Sg = np.zeros((g.max() + 1, 2)); ng = np.zeros(g.max() + 1)
    np.add.at(Sg, g, cells_s1); np.add.at(ng, g, cells_n)
    use = ng > 0; Sg, ng = Sg[use], ng[use]; nu = Sg / ng[:, None]
    brier = (N - ((Sg ** 2).sum(1) / ng).sum()) / N                  # 1 - sum_g ||S_g||^2 / (n_g N)
    nll = (ng * H(nu)).sum() / N                                     # sum_g n_g H(nu_g) / N
    acc = Sg.max(1).sum() / N                                        # sum_g max_k S_gk / N
    return np.array([brier, nll, acc])


def grouped_costs_bf(t1, assign, g, N):
    gi = g[assign]; q = np.zeros_like(t1)
    for gg in np.unique(gi):
        q[gi == gg] = t1[gi == gg].mean(0)
    return bf_metrics(t1, q) / N


# hand example: 10 people, 4 cells, 2 groups (the protocol's rule with 2 in place of 32)
p = np.array([.10, .20, .30, .40, .50, .60, .70, .80, .90, .70]); t1 = np.stack([1 - p, p], 1)
assign0 = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3]); N = 10.0


def cell_stats(assign, k):
    n = np.bincount(assign, minlength=k).astype(float); s1 = np.zeros((k, 2)); np.add.at(s1, assign, t1)
    return n, s1


hand = {}
n0, s0 = cell_stats(assign0, 4); g0 = grouping(n0, s0, 2, N)
c0 = grouped_costs_from_sums(n0, s0, g0, N); hand['C0'] = {'groups': g0.tolist(), 'brier_nll_acc': c0.round(6).tolist(),
                                                           'bruteforce_equal': bool(np.allclose(c0, grouped_costs_bf(t1, assign0, g0, N), atol=1e-15))}
for name, (a, b) in {'merge_cells_0_1 (same group)': (0, 1), 'merge_cells_1_2 (across groups)': (1, 2)}.items():
    asg = np.where(assign0 == b, a, assign0); keep = sorted(set(asg)); remap = {c: i for i, c in enumerate(keep)}
    asg = np.array([remap[c] for c in asg]); n1, s1_ = cell_stats(asg, len(keep)); g1 = grouping(n1, s1_, 2, N)
    c1 = grouped_costs_from_sums(n1, s1_, g1, N)
    na, nb = n0[a], n0[b]; ma, mb = s0[a] / na, s0[b] / nb; mm = (s0[a] + s0[b]) / (na + nb)
    cell = np.array([na * nb / (na + nb) * ((ma - mb) ** 2).sum(), na * kl(ma, mm) + nb * kl(mb, mm),
                     na * ma.max() + nb * mb.max() - (na + nb) * mm.max()]) / N
    hand[name] = {'groups_after': g1.tolist(), 'grouped_brier_nll_acc': c1.round(6).tolist(),
                  'bruteforce_equal': bool(np.allclose(c1, grouped_costs_bf(t1, asg, g1, N), atol=1e-15)),
                  'exact_grouped_change_vs_C0 (brier, nll, acc_loss)': [round(c1[0] - c0[0], 6), round(c1[1] - c0[1], 6), round(c0[2] - c1[2], 6)],
                  'cell_level_change (brier, nll, acc_loss)': cell.round(6).tolist()}
OUT['grouped_income_hand_example'] = hand

# random check with 32 groups
worst = 0.0
for rep in range(50):
    n = 3000; k = 200; pp = rng.beta(1.2, 3.0, n); tt = np.stack([1 - pp, pp], 1)
    asg = rng.integers(0, k, n); asg = np.unique(asg, return_inverse=True)[1]; k = asg.max() + 1
    nn = np.bincount(asg).astype(float); ss = np.zeros((k, 2)); np.add.at(ss, asg, tt)
    gg = grouping(nn, ss, 32, float(n))
    worst = max(worst, np.abs(grouped_costs_from_sums(nn, ss, gg, float(n)) - grouped_costs_bf(tt, asg, gg, float(n))).max())
OUT['grouped_income_random_32_groups_max_abs_error'] = worst

# exact grouped income spending relative to C0 can be negative (regrouping can improve the grouped release)
neg = {'brier': 0, 'nll': 0, 'acc_loss': 0}; r3 = np.random.default_rng(3)
for rep in range(4000):
    k = r3.integers(6, 14); nn = r3.integers(1, 40, k).astype(float); pp = r3.uniform(0.05, 0.95, k)
    ss = np.stack([nn * (1 - pp), nn * pp], 1); NN = nn.sum()
    c0g = grouped_costs_from_sums(nn, ss, grouping(nn, ss, 4, NN), NN)
    a, b = sorted(r3.choice(k, 2, replace=False))
    n2 = np.delete(nn, b); s2 = np.delete(ss, b, 0); n2[a] += nn[b]; s2[a] += ss[b]
    c1g = grouped_costs_from_sums(n2, s2, grouping(n2, s2, 4, NN), NN)
    for m, d in zip(neg, (c1g[0] - c0g[0], c1g[1] - c0g[1], c0g[2] - c1g[2])):
        neg[m] += int(d < -1e-12)
OUT['grouped_income_negative_spending_counts_of_4000_single_merges_4_groups'] = neg


# ---------- 1c. within-SEX guard with kappa shrinkage: literal re-shrinking vs a fixed fine-cell law ----------
def guard_example(kappa=20.0):
    r = np.random.default_rng(7); K = 3
    sizes = [30, 25, 40]; cells = []
    for c, m in enumerate(sizes):
        sex = (r.random(m) < [0.3, 0.6, 0.5][c]).astype(int)
        mu_true = r.dirichlet(np.ones(K) * 3)
        y = np.array([r.choice(K, p=mu_true) if s == 0 else r.choice(K, p=np.roll(mu_true, 1)) for s in sex])
        t2 = r.dirichlet(mu_true * 20, size=m)
        cells.append({'n': m, 's2': t2.sum(0), 'ns': np.array([(sex == s).sum() for s in (0, 1)], float),
                      'Y': np.stack([np.bincount(y[sex == s], minlength=K) for s in (0, 1)]).astype(float)})
    Ns = sum(c['ns'] for c in cells)

    def mu(c):
        return c['s2'] / c['n']

    def ytil(c, s):                         # protocol: shrink toward the CURRENT cell mean
        return (c['Y'][s] + kappa * mu(c)) / (c['ns'][s] + kappa)

    def merge(a, b):
        return {'n': a['n'] + b['n'], 's2': a['s2'] + b['s2'], 'ns': a['ns'] + b['ns'], 'Y': a['Y'] + b['Y'],
                'T': a['T'] + b['T'], 'fine': a['fine'] + b['fine']}
    for i, c in enumerate(cells):           # fixed fine-cell law: T_{c,s} = sum_{c0 in c} n_{c0,s} ytil_{c0,s}
        c['T'] = np.stack([c['ns'][s] * ytil(c, s) for s in (0, 1)]); c['fine'] = [i]

    def delta_literal(a, b, s):
        m = merge(a, b); mm = mu(m)
        return sum(c['ns'][s] * ((mm ** 2).sum() - (mu(c) ** 2).sum() - 2 * (mm - mu(c)) @ ytil(c, s)) for c in (a, b)) / Ns[s]

    def delta_T(a, b, s):
        m = merge(a, b); mm = mu(m)
        return sum(c['ns'][s] * ((mm ** 2).sum() - (mu(c) ** 2).sum()) - 2 * (mm - mu(c)) @ c['T'][s] for c in (a, b)) / Ns[s]

    def F_fixed(partition, s):              # plug-in within-SEX Brier (minus the constant) under the fixed fine-cell law
        tot = 0.0
        for cell in partition:
            mm = mu(cell)
            for i in cell['fine']:
                c0 = cells[i]; tot += c0['ns'][s] * ((mm ** 2).sum() - 2 * mm @ ytil(c0, s))
        return tot / Ns[s]
    res = {}
    for order in ((0, 1, 2), (1, 2, 0), (0, 2, 1)):
        a, b, c = (cells[i] for i in order)
        m1 = merge(a, b); m2 = merge(m1, c)
        lit = [delta_literal(a, b, s) + delta_literal(m1, c, s) for s in (0, 1)]
        tc = [delta_T(a, b, s) + delta_T(m1, c, s) for s in (0, 1)]
        fx = [F_fixed([m2], s) - F_fixed(cells, s) for s in (0, 1)]
        res[f'merge order {order[0]}+{order[1]}, then +{order[2]}'] = {'literal_cumulative': np.round(lit, 7).tolist(),
                                                                         'T_carry_cumulative': np.round(tc, 7).tolist(),
                                                                         'fixed_law_change': np.round(fx, 7).tolist()}
    # single merge: protocol Delta_s equals brute-force plug-in change with the two cells' own shrunk labels as truth
    a, b = cells[0], cells[1]; m = merge(a, b)
    bf = []
    for s in (0, 1):
        before = sum(c['ns'][s] * ((mu(c) ** 2).sum() - 2 * mu(c) @ ytil(c, s) + 1) for c in (a, b))
        after = sum(c['ns'][s] * ((mu(m) ** 2).sum() - 2 * mu(m) @ ytil(c, s) + 1) for c in (a, b))
        bf.append(abs((after - before) / Ns[s] - delta_literal(a, b, s)))
    res['first_merge_protocol_formula_vs_bruteforce_abs_error'] = max(bf)
    return res


OUT['within_sex_guard_kappa20'] = guard_example(20.0)
OUT['within_sex_guard_kappa0_literal_telescopes'] = guard_example(0.0)


# ---------- 2. KL-Bregman centroid and assignment ----------
worst_c, worst_a = 0.0, 0
for rep in range(200):
    T = rng.dirichlet(np.ones(5) * rng.choice([0.5, 2.0]), size=rng.integers(2, 30)); mbar = T.mean(0)
    obj = lambda m: kl(T, np.tile(m, (len(T), 1))).sum()
    for _ in range(20):                     # random perturbations inside the simplex never improve on the mean
        z = rng.dirichlet(np.ones(5)); e = rng.uniform(1e-4, 0.2)
        worst_c = max(worst_c, obj(mbar) - obj((1 - e) * mbar + e * z))
    C = rng.dirichlet(np.ones(5), size=8); x = rng.dirichlet(np.ones(5))
    worst_a += int(np.argmin([kl(x, c) for c in C]) != np.argmax(x @ L(C).T))
OUT['kl_centroid_mean_never_beaten (max obj(mean)-obj(perturbed), must be <= 0)'] = worst_c
OUT['kl_assignment_argmin_KL_equals_argmax_sum_tlogmu_mismatches_of_200'] = worst_a
OUT['kl_zero_handling'] = {'t=0,mu=0 term with clip': float(0.0 * L(np.array(0.0))),
                           't=0.2,mu=0 term with clip (finite stand-in for +inf)': float(-0.2 * L(np.array(0.0)))}


# ---------- 4. IBSI: plug-in information loss equals the AIB / Prop. 1 JS form ----------
def I_cells(n, pmat):                       # plug-in I(T;Y) for hard cells with conditionals pmat
    pbar = (n[:, None] * pmat).sum(0) / n.sum()
    return (n * kl(pmat, np.tile(pbar, (len(n), 1)))).sum() / n.sum()


n = np.array([12., 30., 7.]); P = rng.dirichlet(np.ones(5), size=3); pi = np.array([.2, .7, .45])
for name, M in (('task', P), ('sex', np.stack([1 - pi, pi], 1))):
    before = I_cells(n, M); nm = n[0] + n[1]; mm = (n[0] * M[0] + n[1] * M[1]) / nm
    after = I_cells(np.array([nm, n[2]]), np.stack([mm, M[2]]))
    js = (n[0] * kl(M[0], mm) + n[1] * kl(M[1], mm)) / n.sum()
    OUT[f'ibsi_{name}_info_lost_minus_js_form'] = float((before - after) - js)


# ---------- 3. priority conventions ----------
with np.errstate(divide='ignore', invalid='ignore'):
    OUT['naive_priority_term_Delta0_R0'] = str(np.float64(0.0) / np.float64(0.0))


# ---------- 7. critical values (df = 3954, the frozen-hybrids DEV_REPORT household count minus 1; illustrative) ----------
df = 3954
OUT['critical_values_df3954'] = {'one_sided_alpha/3': stats.t.ppf(1 - 0.025 / 3, df), 'one_sided_alpha/4': stats.t.ppf(1 - 0.025 / 4, df),
                                 'two_sided_bonferroni_K42': stats.t.ppf(1 - 0.05 / 84, df), 'two_sided_bonferroni_K40': stats.t.ppf(1 - 0.05 / 80, df)}

Path(__file__).with_name('MATH_CHECKS.json').write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps(OUT, indent=1, default=float))
