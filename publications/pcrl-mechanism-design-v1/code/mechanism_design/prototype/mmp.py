"""SEX-mixture marginalisation (MMP): exact finite-law and continuous-law checks (data-free).

Mechanism (designer 1, roles/designer_1/PROPOSAL.md): with q_i^s(x) = P(Y_i | x, S = s), pi(x) = P(S = 1 | x) and
pibar = P(S = 1), the SEX-blind posterior is eta_i = (1 - pi) q_i^0 + pi q_i^1 and MMP releases
m_i = (1 - pibar) q_i^0 + pibar q_i^1 (fixed population weights; no person's SEX is read).

Finite laws (exact, fractions): each law has 6 input cells, S in {0 male, 1 female}, Y1 in {0, 1} (income),
Y2 in {0, 1, 2} (occupation: professional, female-typical, male-typical), Y1 and Y2 conditionally independent given
(X, S). The reference release is the exact SEX-blind posterior eta (a perfect U0). For every pair of set partitions
(Bell(6)^2 = 41,209 pairs) the calibrated release (each block releases its exact posterior) is enumerated, so the
global frontier at a budget is the exact minimum pair Bayes AUC over every calibrated partition release; it weakly
dominates every deterministic release at that budget in Bayes terms (any deterministic release is a partition, and
calibrating it within blocks lowers every proper score without changing the view). Local, greedy-sequential (S12,
S21) and MMP designs are evaluated on the same table.

Continuous laws (approximate): a generative model with conditionally independent features given S gives exact
posteriors; readers are the ACS seven-reader slate (task_conditioned_representation.attacks.fit_reader) fitted on
finite auxiliary samples, so their AUCs are estimates, not population floors.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction as F

import numpy as np

from mechanism_design.prototype.finite_law import FiniteLaw, argmax_first, bayes_auc, mutual_information_bits, set_partitions

CAP = {'accuracy_loss': F(1, 100), 'nll_excess': F(1, 100), 'brier_excess': F(5, 1000)}
CAP_HEADROOM = {'accuracy_loss': F(16, 1000), 'nll_excess': F(20, 1000), 'brier_excess': F(9, 1000)}
BUDGETS = {'cap': CAP, 'cap_plus_headroom': CAP_HEADROOM}


# ----------------------------------------------------------------------------- finite laws
def law_from_components(cells):
    """cells: list of (mass, pi, (P(Y1=1|S=0), P(Y1=1|S=1)), (q2|S=0 3-vector), (q2|S=1 3-vector))."""
    w, joint = [], []
    for m, pi, y1, q20, q21 in cells:
        tab = [[[F(0)] * 3 for _ in range(2)] for _ in range(2)]
        for s, ps, p1, q2 in ((0, 1 - F(pi), F(y1[0]), q20), (1, F(pi), F(y1[1]), q21)):
            for a, pa in ((0, 1 - p1), (1, p1)):
                for b in range(3):
                    tab[s][a][b] = ps * pa * F(q2[b])
        w.append(F(m))
        joint.append(tuple(tuple(tuple(r) for r in tab[s]) for s in range(2)))
    return FiniteLaw(tuple(w), tuple(joint), 2, 3)


def _occ(a, b, c):
    return (F(a), F(b), F(c))


def finite_laws():
    """Laws fixed before the first run (see PREREGISTRATION in run_prototype.py)."""
    L = {}
    # L_NEG (required disclosure): tasks depend on X only through S; X carries S through z.
    cells = []
    for z, pi in enumerate((F(15, 100), F(1, 2), F(85, 100))):
        for wv in (0, 1):
            cells.append((F(1, 6), pi, (F(55, 100), F(30, 100)), _occ('0.40', '0.15', '0.45'), _occ('0.40', '0.50', '0.10')))
    L['L_NEG'] = cells
    # L_PROXY (feasible removal): z is a pure SEX proxy; w is a task feature independent of S; gendered gaps never
    # change the argmax.
    cells = []
    for z, pi in enumerate((F(2, 10), F(1, 2), F(8, 10))):
        for wv in (0, 1):
            y1 = (F(30, 100) + F(35, 100) * wv, F(20, 100) + F(35, 100) * wv)
            q0 = _occ('0.15', '0.25', '0.60') if wv == 0 else _occ('0.60', '0.10', '0.30')
            q1 = _occ('0.15', '0.40', '0.45') if wv == 0 else _occ('0.60', '0.30', '0.10')
            cells.append((F(1, 6), pi, y1, q0, q1))
    L['L_PROXY'] = cells
    # L_PROXY_FLIP (failure mode 2): as L_PROXY but the gendered occupation gap swaps the argmax.
    cells = []
    for z, pi in enumerate((F(2, 10), F(1, 2), F(8, 10))):
        for wv in (0, 1):
            y1 = (F(30, 100) + F(35, 100) * wv, F(20, 100) + F(35, 100) * wv)
            q0 = _occ('0.30', '0.20', '0.50') if wv == 0 else _occ('0.60', '0.15', '0.25')
            q1 = _occ('0.25', '0.55', '0.20') if wv == 0 else _occ('0.55', '0.35', '0.10')
            cells.append((F(1, 6), pi, y1, q0, q1))
    L['L_PROXY_FLIP'] = cells
    # L_MIXED (task-relevant proxy): h ("part-time") moves both P(S) and the tasks within SEX; z is a pure proxy.
    cells = []
    for h in (0, 1):
        for z in (0, 1, 2):
            pi = F(20, 100) + F(30, 100) * h + F(15, 100) * z
            y1 = (F(70, 100) - F(30, 100) * h, F(60, 100) - F(30, 100) * h)
            q0 = _occ('0.45', '0.15', '0.40') if h == 0 else _occ('0.25', '0.40', '0.35')
            q1 = _occ('0.45', '0.30', '0.25') if h == 0 else _occ('0.25', '0.60', '0.15')
            cells.append((F(1, 6), pi, y1, q0, q1))
    L['L_MIXED'] = cells
    # L_NOPROXY (no pure proxies): every SEX-predictive field (h) and the second field (e) affect the tasks within SEX.
    cells = []
    for h in (0, 1, 2):
        for e in (0, 1):
            pi = F(25, 100) + F(25, 100) * h + F(5, 100) * e
            y1 = (F(65, 100) - F(15, 100) * h + F(15, 100) * e, F(55, 100) - F(15, 100) * h + F(15, 100) * e)
            c0 = F(45, 100) + F(10, 100) * e - F(5, 100) * h
            q0 = (c0, F(15, 100) + F(5, 100) * h, 1 - c0 - (F(15, 100) + F(5, 100) * h))
            q1 = (c0, F(35, 100) + F(5, 100) * h, 1 - c0 - (F(35, 100) + F(5, 100) * h))
            cells.append((F(1, 6), pi, y1, q0, q1))
    L['L_NOPROXY'] = cells
    # L_XOR (coalition required disclosure): income needs a, occupation needs b, and SEX follows a XOR b.
    cells = []
    for a_ in (0, 1):
        for b_ in (0, 1):
            pi = F(15, 100) if (a_ ^ b_) == 0 else F(85, 100)
            y1 = (F(30, 100) + F(40, 100) * a_, F(25, 100) + F(40, 100) * a_)
            q0 = _occ('0.60', '0.15', '0.25') if b_ == 0 else _occ('0.20', '0.30', '0.50')
            q1 = _occ('0.55', '0.25', '0.20') if b_ == 0 else _occ('0.15', '0.40', '0.45')
            cells.append((F(1, 4), pi, y1, q0, q1))
    L['L_XOR'] = cells
    return {k: (v, law_from_components(v)) for k, v in L.items()}


def components(cells):
    pis = [F(c[1]) for c in cells]
    w = [F(c[0]) for c in cells]
    pibar = sum(wi * p for wi, p in zip(w, pis))
    q1 = [((1 - F(c[2][0]), F(c[2][0])), (1 - F(c[2][1]), F(c[2][1]))) for c in cells]   # q1[c][s] = vector
    q2 = [(tuple(c[3]), tuple(c[4])) for c in cells]
    return w, pis, pibar, q1, q2


def mix(qs, weight):
    """(1 - weight) q^0 + weight q^1, elementwise."""
    return tuple((1 - weight) * a + weight * b for a, b in zip(qs[0], qs[1]))


def releases(cells):
    w, pis, pibar, q1, q2 = components(cells)
    eta = {1: {c: mix(q1[c], pis[c]) for c in range(len(cells))}, 2: {c: mix(q2[c], pis[c]) for c in range(len(cells))}}
    mmp = {1: {c: mix(q1[c], pibar) for c in range(len(cells))}, 2: {c: mix(q2[c], pibar) for c in range(len(cells))}}
    return eta, mmp, pibar


def exact_metrics(law, task, q):
    """Exact (Fraction) accuracy and summed Brier, float NLL, for release map q (cell -> vector)."""
    acc, brier, nll = F(0), F(0), 0.0
    for c in law.cells:
        t = law.p_y(c, task)
        p = q[c]
        acc += law.w[c] * t[argmax_first(p)]
        brier += law.w[c] * sum(t[y] * sum((p[k] - (1 if k == y else 0)) ** 2 for k in range(len(p))) for y in range(len(t)))
        nll += float(law.w[c]) * sum(float(t[y]) * -math.log(max(float(p[y]), 1e-12)) for y in range(len(t)) if t[y] > 0)
    return acc, brier, nll


def excess(law, task, q, ref):
    a, b, n = exact_metrics(law, task, q)
    ra, rb, rn = exact_metrics(law, task, ref)
    return {'accuracy_loss': ra - a, 'nll_excess': n - rn, 'brier_excess': b - rb}


def within(ex, budget):
    return all(float(ex[k]) <= float(budget[k]) + 1e-15 for k in budget)


def identity_checks(cells, law):
    """Brier identity (exact) and NLL bound E KL(eta || m) <= I(S; X) for both tasks."""
    w, pis, pibar, q1, q2 = components(cells)
    eta, mmp, _ = releases(cells)
    out = {}
    i_sx = sum(float(w[c]) * kl_bern(float(pis[c]), float(pibar)) for c in range(len(cells)))
    for task, qq in ((1, q1), (2, q2)):
        ex = excess(law, task, mmp[task], eta[task])
        rhs = sum(w[c] * (pis[c] - pibar) ** 2 * sum((a - b) ** 2 for a, b in zip(qq[c][1], qq[c][0])) for c in range(len(cells)))
        kl = sum(float(w[c]) * kl_vec([float(v) for v in eta[task][c]], [float(v) for v in mmp[task][c]]) for c in range(len(cells)))
        out[f'task{task}'] = {'brier_excess_exact': str(ex['brier_excess']), 'identity_rhs_exact': str(rhs),
                              'brier_identity_holds_exactly': ex['brier_excess'] == rhs,
                              'nll_excess': ex['nll_excess'], 'expected_kl_eta_m': kl, 'I_S_X_nats': i_sx,
                              'nll_equals_expected_kl': abs(ex['nll_excess'] - kl) < 1e-12, 'nll_bound_holds': kl <= i_sx + 1e-15}
    return out


def kl_bern(p, q):
    return sum(a * math.log(a / b) for a, b in ((p, q), (1 - p, 1 - q)) if a > 0)


def kl_vec(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)


# ----------------------------------------------------------------------------- exhaustive frontier
class PartitionTable:
    """Per-task table over all set partitions of the cells: calibrated release symbol ids and clause excess."""

    def __init__(self, law, task, ref):
        self.law, self.task = law, task
        self.parts = list(set_partitions(list(law.cells)))
        self.sym = []          # per partition: tuple of per-cell symbol ids (equal released vectors share an id)
        self.vec = []          # per partition: dict cell -> exact vector
        self.ex = []
        k = law.k1 if task == 1 else law.k2
        codebook = {}
        for part in self.parts:
            q = {}
            for block in part:
                m = sum(law.w[c] for c in block)
                v = tuple(sum(law.w[c] * law.p_y(c, task)[y] for c in block) / m for y in range(k))
                for c in block:
                    q[c] = v
            ids = tuple(codebook.setdefault(q[c], len(codebook)) for c in law.cells)
            self.sym.append(ids)
            self.vec.append(q)
            self.ex.append(excess(law, task, q, ref))


def float_auc_from_groups(m1, m0):
    """Bayes AUC from per-symbol masses (floats), ties within 1e-12 relative counted one half."""
    m1 = np.asarray(m1, float) / np.sum(m1); m0 = np.asarray(m0, float) / np.sum(m0)
    lr = np.where(m0 > 0, m1 / np.where(m0 > 0, m0, 1), np.inf)
    order = np.argsort(lr, kind='mergesort')
    lr, m1, m0 = lr[order], m1[order], m0[order]
    auc, cum0, i = 0.0, 0.0, 0
    n = len(lr)
    while i < n:
        j = i
        while j + 1 < n and (lr[j + 1] == lr[i] or (np.isfinite(lr[i]) and abs(lr[j + 1] - lr[i]) <= 1e-12 * max(1.0, abs(lr[i])))):
            j += 1
        g1, g0 = m1[i:j + 1].sum(), m0[i:j + 1].sum()
        auc += g1 * cum0 + 0.5 * g1 * g0
        cum0 += g0
        i = j + 1
    return auc


def pair_auc_float(law, ids1, ids2):
    groups = {}
    for c in law.cells:
        key = (ids1[c], ids2[c])
        a = groups.setdefault(key, [0.0, 0.0])
        a[0] += float(law.w[c] * law.p_s(c, 1)); a[1] += float(law.w[c] * law.p_s(c, 0))
    v = list(groups.values())
    return float_auc_from_groups([x[0] for x in v], [x[1] for x in v])


def single_auc_float(law, ids):
    return pair_auc_float(law, ids, tuple(0 for _ in law.cells))


def frontier(law, eta):
    t1, t2 = PartitionTable(law, 1, eta[1]), PartitionTable(law, 2, eta[2])
    out = {}
    single1 = [single_auc_float(law, s) for s in t1.sym]
    single2 = [single_auc_float(law, s) for s in t2.sym]
    for bname, budget in BUDGETS.items():
        f1 = [i for i in range(len(t1.parts)) if within(t1.ex[i], budget)]
        f2 = [j for j in range(len(t2.parts)) if within(t2.ex[j], budget)]
        res = {'n_feasible_income_partitions': len(f1), 'n_feasible_occupation_partitions': len(f2)}
        if not f1 or not f2:
            res['feasible'] = False
            out[bname] = res
            continue
        best, arg = 2.0, None
        for i in f1:
            for j in f2:
                a = pair_auc_float(law, t1.sym[i], t2.sym[j])
                if a < best - 1e-15:
                    best, arg = a, (i, j)
        # local: each recipient minimises its own single-view Bayes AUC (ties: lower Brier excess, then index)
        li = min(f1, key=lambda i: (round(single1[i], 12), float(t1.ex[i]['brier_excess']), i))
        lj = min(f2, key=lambda j: (round(single2[j], 12), float(t2.ex[j]['brier_excess']), j))
        s12 = min(f2, key=lambda j: (round(pair_auc_float(law, t1.sym[li], t2.sym[j]), 12), float(t2.ex[j]['brier_excess']), j))
        s21 = min(f1, key=lambda i: (round(pair_auc_float(law, t1.sym[i], t2.sym[lj]), 12), float(t1.ex[i]['brier_excess']), i))
        gi, gj = arg
        res.update({
            'feasible': True,
            'global_min_pair_auc': float(bayes_auc(law, lambda c: (t1.sym[gi][c], t2.sym[gj][c]))),
            'global_argmin_partitions': [[list(b) for b in t1.parts[gi]], [list(b) for b in t2.parts[gj]]],
            'local_pair_auc': pair_auc_float(law, t1.sym[li], t2.sym[lj]),
            'local_single_aucs': [single1[li], single2[lj]],
            's12_pair_auc': pair_auc_float(law, t1.sym[li], t2.sym[s12]),
            's21_pair_auc': pair_auc_float(law, t1.sym[s21], t2.sym[lj]),
        })
        out[bname] = res
    return out


def level_partition(q, cells):
    groups = {}
    for c in cells:
        groups.setdefault(q[c], []).append(c)
    return [tuple(v) for v in groups.values()]


def evaluate_finite(name, cells, law):
    eta, mmp, pibar = releases(cells)
    out = {'name': name, 'pibar': str(pibar), 'auc_star_X': float(bayes_auc(law, lambda c: c)),
           'I_S_X_bits': mutual_information_bits(law, lambda c: c)}
    view = lambda r: (lambda c: (r[1][c], r[2][c]))
    out['twin_eta'] = {'pair_auc': float(bayes_auc(law, view(eta))), 'income_auc': float(bayes_auc(law, lambda c: eta[1][c])),
                       'occupation_auc': float(bayes_auc(law, lambda c: eta[2][c]))}
    ex = {t: excess(law, t, mmp[t], eta[t]) for t in (1, 2)}
    out['mmp'] = {'pair_auc_exact': str(bayes_auc(law, view(mmp))), 'pair_auc': float(bayes_auc(law, view(mmp))),
                  'income_auc': float(bayes_auc(law, lambda c: mmp[1][c])), 'occupation_auc': float(bayes_auc(law, lambda c: mmp[2][c])),
                  'excess_vs_eta': {f'task{t}': {k: float(v) for k, v in ex[t].items()} for t in (1, 2)},
                  'feasible': {b: all(within(ex[t], bud) for t in (1, 2)) for b, bud in BUDGETS.items()}}
    # MMP's own partition, released calibrated (the cheapest release with exactly MMP's view)
    cal = {}
    for t in (1, 2):
        part = level_partition(mmp[t], law.cells)
        k = law.k1 if t == 1 else law.k2
        q = {}
        for block in part:
            m = sum(law.w[c] for c in block)
            v = tuple(sum(law.w[c] * law.p_y(c, t)[y] for c in block) / m for y in range(k))
            for c in block:
                q[c] = v
        cal[t] = q
    exc = {t: excess(law, t, cal[t], eta[t]) for t in (1, 2)}
    out['mmp_partition_calibrated'] = {'pair_auc': float(bayes_auc(law, view(cal))),
                                       'excess_vs_eta': {f'task{t}': {k: float(v) for k, v in exc[t].items()} for t in (1, 2)},
                                       'feasible': {b: all(within(exc[t], bud) for t in (1, 2)) for b, bud in BUDGETS.items()}}
    out['identities'] = identity_checks(cells, law)
    out['subgroup_brier_increase_mmp_vs_eta'] = subgroup_increase(cells, law, mmp, eta)
    out['w_star'] = w_star(cells)
    out['frontier'] = frontier(law, eta)
    return out


def subgroup_increase(cells, law, mmp, eta):
    """Exact within-SEX summed-Brier increase of MMP over eta, per task and SEX group (S = 1 female)."""
    w, pis, pibar, q1, q2 = components(cells)
    out = {}
    for task, qq in ((1, q1), (2, q2)):
        res = {}
        for s_ in (0, 1):
            ps = [w[c] * (pis[c] if s_ == 1 else 1 - pis[c]) for c in range(len(cells))]
            tot = sum(ps)
            def brier(r, c):
                return sum(qq[c][s_][y] * sum((r[k] - (1 if k == y else 0)) ** 2 for k in range(len(r))) for y in range(len(r)))
            inc = sum(ps[c] * (brier(mmp[task][c], c) - brier(eta[task][c], c)) for c in range(len(cells))) / tot
            res['female' if s_ == 1 else 'male'] = float(inc)
        pop = float(excess(law, task, mmp[task], eta[task])['brier_excess'])
        res['population'] = pop
        res['female_over_population'] = res['female'] / pop if pop > 0 else None
        out[f'task{task}'] = res
    return out


def w_star(cells):
    """Brier-optimal constant mixing weight w* = E[pi ||g||^2] / E[||g||^2] and the Brier price at w* vs at pibar."""
    w, pis, pibar, q1, q2 = components(cells)
    out = {}
    for task, qq in ((1, q1), (2, q2)):
        g2 = [sum((a - b) ** 2 for a, b in zip(qq[c][1], qq[c][0])) for c in range(len(cells))]
        den = sum(w[c] * g2[c] for c in range(len(cells)))
        if den == 0:
            out[f'task{task}'] = {'w_star': None, 'note': 'no within-SEX gap'}
            continue
        ws = sum(w[c] * pis[c] * g2[c] for c in range(len(cells))) / den
        price = lambda wt: sum(w[c] * (pis[c] - wt) ** 2 * g2[c] for c in range(len(cells)))
        out[f'task{task}'] = {'w_star': float(ws), 'pibar': float(pibar), 'brier_price_at_pibar': float(price(pibar)),
                              'brier_price_at_w_star': float(price(ws))}
    return out


# ----------------------------------------------------------------------------- continuous laws (fitted readers)
SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))


def kinds(p):
    p = np.asarray(p, dtype=np.float64); k = p.shape[1]; d = p.argmax(1); one = np.eye(k)[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
            'dec': one, 'conf': p.max(1, keepdims=True)}


class ContinuousLaw:
    """Features conditionally independent given S: H (hours-like, Gaussian), E (education, 5 levels), A (age, uniform
    in 25..64 on a grid, weakly SEX-dependent), and optionally Z (pure proxy, 5 levels). Tasks depend on (H, E, A, S)."""

    def __init__(self, with_proxy, p_female=0.47):
        self.with_proxy = with_proxy
        self.pf = p_female
        self.h_mean = (0.35, -0.40); self.h_sd = (0.85, 1.0)                       # standardised hours, by SEX
        self.e_prob = (np.array([.10, .30, .25, .25, .10]), np.array([.07, .26, .25, .29, .13]))
        self.a_grid = np.arange(25, 65)
        self.a_prob = (np.full(40, 1 / 40), np.linspace(1.1, 0.9, 40) / np.linspace(1.1, 0.9, 40).sum())
        self.z_prob = (np.array([.30, .25, .20, .15, .10]), np.array([.10, .15, .20, .25, .30]))

    def sample(self, n, rng):
        s = (rng.random(n) < self.pf).astype(int)
        h = np.where(s == 1, rng.normal(self.h_mean[1], self.h_sd[1], n), rng.normal(self.h_mean[0], self.h_sd[0], n))
        e = np.array([rng.choice(5, p=self.e_prob[si]) for si in s])
        a = np.array([rng.choice(40, p=self.a_prob[si]) for si in s])
        z = np.array([rng.choice(5, p=self.z_prob[si]) for si in s]) if self.with_proxy else np.zeros(n, int)
        return s, h, e, a, z

    def posterior_s(self, h, e, a, z, use_proxy=True):
        """P(S=1 | features) exactly (conditional independence given S)."""
        def logp(si):
            lp = -0.5 * ((h - self.h_mean[si]) / self.h_sd[si]) ** 2 - np.log(self.h_sd[si])
            lp = lp + np.log(self.e_prob[si][e]) + np.log(self.a_prob[si][a])
            if self.with_proxy and use_proxy:
                lp = lp + np.log(self.z_prob[si][z])
            return lp
        l1 = np.log(self.pf) + logp(1); l0 = np.log(1 - self.pf) + logp(0)
        return 1 / (1 + np.exp(l0 - l1))

    def q(self, h, e, a, s):
        """Within-SEX task posteriors: income P(Y1=1|x,s) (2-vector) and occupation (3-vector)."""
        age = (self.a_grid[a] - 45) / 10
        li = -0.6 + 0.9 * h + 0.55 * (e - 2) + 0.25 * age - 0.45 * s
        p1 = 1 / (1 + np.exp(-li))
        z0 = 0.70 * (e - 2) + 0.10 * age
        z1 = -0.45 * h + 0.80 * s - 0.10 * (e - 2)
        z2 = 0.30 * h - 1.10 * s - 0.45 * (e - 2)
        zz = np.column_stack((z0, z1, z2)); zz = zz - zz.max(1, keepdims=True); ez = np.exp(zz)
        return np.column_stack((1 - p1, p1)), ez / ez.sum(1, keepdims=True)


def continuous_releases(law, h, e, a, z):
    pi = law.posterior_s(h, e, a, z)
    i0, o0 = law.q(h, e, a, 0); i1, o1 = law.q(h, e, a, 1)
    eta1 = (1 - pi)[:, None] * i0 + pi[:, None] * i1; eta2 = (1 - pi)[:, None] * o0 + pi[:, None] * o1
    m1 = (1 - law.pf) * i0 + law.pf * i1; m2 = (1 - law.pf) * o0 + law.pf * o1
    rel = {'twin_eta': (eta1, eta2), 'mmp': (m1, m2)}
    rel['_q'] = ((i0, i1), (o0, o1))
    return rel, pi


class LocalLinearEraser:
    """Critic-style local protection of one recipient's own wire: a LEACE-type rank-one eraser on the centred
    log-probabilities, fitted with SEX on the fit sample, applied with fraction t (t = 1 zeroes the wire's linear
    cross-covariance with SEX). t is chosen so the summed-Brier price over eta matches MMP's price on that task."""

    def __init__(self, p_fit, s_fit):
        L = np.log(np.maximum(p_fit, 1e-300)); L = L - L.mean(1, keepdims=True)
        self.mu = L.mean(0)
        C = np.cov(L, rowvar=False).reshape(L.shape[1], L.shape[1])
        vals, vecs = np.linalg.eigh(C)
        keep = vals > 1e-12 * max(vals.max(), 1e-300)
        self.W = (vecs[:, keep] / np.sqrt(vals[keep])).T            # whitening on the non-degenerate subspace
        self.Winv = vecs[:, keep] * np.sqrt(vals[keep])
        c = ((L - self.mu) * (s_fit - s_fit.mean())[:, None]).mean(0)
        cw = self.W @ c
        self.u = cw / max(np.linalg.norm(cw), 1e-300)

    def apply(self, p, t):
        L = np.log(np.maximum(p, 1e-300)); L = L - L.mean(1, keepdims=True)
        Z = (L - self.mu) @ self.W.T
        Z = Z - t * (Z @ self.u)[:, None] * self.u[None, :]
        L2 = self.mu + Z @ self.Winv.T + (L - self.mu - ((L - self.mu) @ self.W.T) @ self.Winv.T)
        L2 = L2 - L2.max(1, keepdims=True); e = np.exp(L2)
        return e / e.sum(1, keepdims=True)


def matched_fraction(eraser, p_eta, target_price):
    lo, hi = 0.0, 1.0
    if np.mean(np.sum((eraser.apply(p_eta, 1.0) - p_eta) ** 2, 1)) <= target_price:
        return 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.mean(np.sum((eraser.apply(p_eta, mid) - p_eta) ** 2, 1)) < target_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def empirical_auc(score, s):
    from task_conditioned_representation.inference import auc
    return float(auc(s, score))


def fitted_recovery(law, rng_seed, replication, n_aux=8001, n_val=3000, n_test=5943):
    """Fit the 7-reader slate per view on n_aux auxiliary releases, select and orient on n_val, score n_test."""
    from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
    rng = np.random.default_rng(rng_seed)
    data = {}
    for role, n in (('fit', n_aux), ('val', n_val), ('test', n_test)):
        s, h, e, a, z = law.sample(n, rng)
        rel, pi = continuous_releases(law, h, e, a, z)
        data[role] = (s, rel, pi, (h, e, a, z))
    # matched-price local linear masking arm (fitted with SEX on the auxiliary-free fit sample of the defender)
    s_fit, rel_fit = data['fit'][0], data['fit'][1]
    erasers, fractions = [], []
    for t in range(2):
        er = LocalLinearEraser(rel_fit['twin_eta'][t], s_fit)
        price = float(np.mean(np.sum((rel_fit['mmp'][t] - rel_fit['twin_eta'][t]) ** 2, 1)))
        erasers.append(er); fractions.append(matched_fraction(er, rel_fit['twin_eta'][t], price))
    for role in data:
        rel = data[role][1]
        rel['local_linear'] = tuple(erasers[t].apply(rel['twin_eta'][t], fractions[t]) for t in range(2))
    out = {'local_linear_fractions': fractions}
    for arm in ('twin_eta', 'mmp', 'local_linear'):
        views = {}
        for role in data:
            p1, p2 = data[role][1][arm]
            k1, k2 = kinds(p1), kinds(p2)
            views[role] = {'income': k1, 'occupation': k2, 'pair': {k: np.column_stack((k1[k], k2[k])) for k in k1}}
        res = {}
        for view in ('income', 'occupation', 'pair'):
            cands = []
            for name, kind in SLATE:
                rec = fit_reader(views['fit'][view][kind], data['fit'][0], name, MODEL_SEED_BASE + replication)
                sv = predict_reader(rec, views['val'][view][kind]); st = predict_reader(rec, views['test'][view][kind])
                va = empirical_auc(sv, data['val'][0]); sign = 1 if va >= 0.5 else -1
                ta = empirical_auc(st if sign == 1 else 1 - st, data['test'][0])
                cands.append((max(va, 1 - va), name, ta))
            if view == 'pair':
                for v in ('income', 'occupation'):
                    cands += [(c[0], f'IGNORE_other__{v}__{c[1]}', c[2]) for c in res[v]['candidates']]
            best = max(cands, key=lambda c: c[0])
            res[view] = {'winner': best[1], 'test_auc': best[2], 'candidates': cands}
        # Bayes (exact posterior) AUC of S from the view, estimated on the test sample
        s, rel, pi, (h, e, a, z) = data['test']
        if arm in ('twin_eta', 'local_linear'):
            bayes = empirical_auc(pi, s)        # AUC*(S; X); the twin's (and the eraser's) Bayes AUC when injective in X
        else:
            bayes = empirical_auc(law.posterior_s(h, e, a, z, use_proxy=False), s)   # m depends on (H, E, A) only
        out[arm] = {v: {'winner': res[v]['winner'], 'test_auc': res[v]['test_auc']} for v in res}
        out[arm]['bayes_pair_auc_mc'] = bayes
    # utility of MMP relative to the exact SEX-blind posterior (Monte Carlo over the test sample, exact conditionals)
    s, rel, pi, feats = data['test']
    util = {}
    for arm in ('mmp', 'local_linear'):
        for t in (0, 1):
            ke, km = rel['twin_eta'][t], rel[arm][t]
            q0, q1 = rel['_q'][t]
            brier_exc = float(np.mean(np.sum((km - ke) ** 2, 1)))                  # = E||r - eta||^2 (identity)
            nll_exc = float(np.mean(np.sum(ke * (np.log(np.maximum(ke, 1e-12)) - np.log(np.maximum(km, 1e-12))), 1)))
            acc_loss = float(np.mean(ke.max(1) - ke[np.arange(len(ke)), km.argmax(1)]))
            sub = {}
            for g, qs in (('male', q0), ('female', q1)):
                mask = s == (1 if g == 'female' else 0)
                inc = (np.sum(km ** 2, 1) - 2 * np.sum(km * qs, 1)) - (np.sum(ke ** 2, 1) - 2 * np.sum(ke * qs, 1))
                sub[g] = float(inc[mask].mean())
            util[f'{arm}|task{t + 1}'] = {'brier_excess': brier_exc, 'nll_excess': nll_exc, 'accuracy_loss': acc_loss,
                                          'within_sex_brier_increase': sub}
    out['utility_vs_eta_mc'] = util
    return out
