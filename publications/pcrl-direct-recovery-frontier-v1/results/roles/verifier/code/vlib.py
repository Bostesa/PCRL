"""Independent verifier library of the DRF1 campaign (`pcrl_direct_recovery_frontier_v1`; verifier role).

Written by the verifier from the DEFINITIONS in PROTOCOL.md, cards/DR_DRF_v1.md, DESIGN_BRIEF.md and RULES.md, not from the
campaign's implementation: this module imports numpy and the standard library only. It never imports drf.solver, drf.dr,
drf.finite_law, drf.evaluate, drf.rules, drf.devreaders or any other metric / objective / solver / rules / reader code. Data access
(labels, features, saved arrays, frozen teachers) is done by the calling scripts through the loaders the verifier brief allows.

Definitions implemented here (each one restated from the registered documents):
- Routing: nearest codeword by cross-entropy, k = argmin_c -sum_j q_j log C[c, j] (first index on exact ties).
- pi bins: m = number of frozen cuts <= pi (4 bins; cuts = the 0.25 / 0.5 / 0.75 quantiles of the clipped MODEL_FIT out-of-fold
  pi, linear interpolation between order statistics). Fine cells: income 4 k + m, occupation 4 l + m.
- B1 tables: an atom's vector is the mean teacher posterior of the people sent there, clipped at 1e-6 and renormalized; an atom
  with nobody emits its fallback row (the twin codeword), clipped and renormalized the same way.
- Emitted law: identical emitted vectors (bitwise equality of the whole float64 vector; for the pair, of both vectors) are one
  observable value. Masses: m1[w] = sum of w1, m0[w] = sum of (1 - w1) over the people emitting w (w1 = pi: estimated law; w1 =
  true SEX: empirical law).
- AUC* (exact optimal-reader AUC of a finite law): 1/2 + 1/4 sum_{a != b} |P1(a) P0(b) - P1(b) P0(a)| (closed form; checked
  against the tie-grouped likelihood-ratio ranking form sum_g P1_g (P0_below_g + P0_g / 2)).
- Plug-in MI (PRC definition): sum_w p(w) KL(Bern(r_w) || Bern(rbar)), r_w = m1 / (m1 + m0) clipped to [1e-9, 1 - 1e-9], rbar =
  sum m1 / total, nats. The unclipped exact value (0 log 0 = 0) is reported beside.
- Task costs per person: NLL -log p_y (the MODEL_SELECT evaluator floors p_y at 1e-12), summed Brier sum_k (p_k - 1[y = k])^2,
  correctness 1[argmax p = y] (first index on exact ties).
- The 8 MODEL_FIT spends over the identity: mean NLL, mean summed Brier and accuracy loss (identity minus construction) of both
  tasks, and the men's (sex 0) / women's (sex 1) mean summed occupation Brier, construction minus identity.
- Revision R1 (card DR_DRF_R1) replaces each realized label by a label distribution: income costs under the income teacher's
  out-of-fold posterior, occupation costs under the occupation teacher's, the men's / women's occupation Brier under the
  SEX-conditional occupation teacher of each person's own SEX group; the expected costs are E[-log V_y] = -sum_c L_c log V_c,
  E[Brier] = sum_k V_k^2 - 2 sum_c L_c V_c + 1, E[correct] = L[argmax V] (the R1 coarsening path uses the expected NLL too).
- The 14 MODEL_SELECT conditions per version (PROTOCOL section 5): task x {U0, Ucal} x {accuracy loss <= 0.010, NLL excess <= 0.010,
  summed Brier excess <= 0.005} and the men's / women's occupation summed-Brier excess vs U0 <= 0.005 (inclusive caps).
- Household-clustered SE of a mean: sqrt(G / (G - 1) sum_g (sum_{i in g} (d_i - dbar))^2) / n; subgroup mean: linearized ratio.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

import numpy as np

TABLE_CLIP = 1e-6
MI_CLIP = 1e-9
NLL_FLOOR_EVAL = 1e-12
PI_CLIP = (0.01, 0.99)
B_PI = 4
CAPACITY = (16, 64)
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD_CAP = 0.005
METRICS8 = ('inc_nll', 'occ_nll', 'inc_brier', 'occ_brier', 'inc_acc', 'occ_acc', 'occ_brier_men', 'occ_brier_women')
TOL = 1e-9


# ---------------------------------------------------------------- routing, cuts, fine cells
def nearest_codeword(Q, C):
    """(index, gap): argmin over codewords c of the cross-entropy H(q, C_c) = -sum_j q_j log C[c, j], accumulated class by class
    (not by a matrix product), and the gap between the best and the runner-up cross-entropy (near-tie diagnostic)."""
    Q = np.asarray(Q, dtype=np.float64)
    L = np.log(np.asarray(C, dtype=np.float64))
    H = np.zeros((Q.shape[0], L.shape[0]))
    for j in range(Q.shape[1]):
        H -= Q[:, j][:, None] * L[:, j][None, :]
    idx = np.argmin(H, axis=1)
    part = np.partition(H, 1, axis=1)
    return idx.astype(np.int64), part[:, 1] - part[:, 0]


def quantile_type7(x, q):
    """Linear interpolation between order statistics at position (n - 1) q (Hyndman-Fan type 7)."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    h = (len(xs) - 1) * float(q)
    lo = int(math.floor(h))
    hi = min(lo + 1, len(xs) - 1)
    t = h - lo
    return float(xs[lo] + t * (xs[hi] - xs[lo]))


def pi_bins(pi, cuts):
    """Number of cuts <= pi (0..len(cuts))."""
    pi = np.asarray(pi, dtype=np.float64)
    m = np.zeros(len(pi), dtype=np.int64)
    for c in cuts:
        m += (pi >= float(c)).astype(np.int64)
    return m


def fine_cells(k, l, m):
    return np.asarray(k, dtype=np.int64) * B_PI + m, np.asarray(l, dtype=np.int64) * B_PI + m


def identity_maps():
    return np.repeat(np.arange(CAPACITY[0]), B_PI).astype(np.int64), np.repeat(np.arange(CAPACITY[1]), B_PI).astype(np.int64)


# ---------------------------------------------------------------- tables and emitted wires
def renorm_clip(T):
    T = np.clip(np.asarray(T, dtype=np.float64), TABLE_CLIP, 1.0)
    return T / T.sum(axis=1, keepdims=True)


def b1_tables(atom, Q, n_atoms, fallback):
    """B1 vectors of atoms 0..n_atoms-1: clipped mean of Q over the people of each atom; empty atoms emit their fallback row."""
    atom = np.asarray(atom, dtype=np.int64)
    Q = np.asarray(Q, dtype=np.float64)
    n = np.bincount(atom, minlength=n_atoms).astype(np.float64)
    S = np.column_stack([np.bincount(atom, weights=Q[:, j], minlength=n_atoms) for j in range(Q.shape[1])])
    T = np.array(np.asarray(fallback, dtype=np.float64), copy=True)
    nz = n > 0
    T[nz] = S[nz] / n[nz][:, None]
    return renorm_clip(T), n


def row_ids(V):
    """Ids of distinct rows of V under bitwise equality of the whole float64 row (ids in first-appearance-free sorted order)."""
    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64))
    b = V.view(np.dtype((np.void, V.dtype.itemsize * V.shape[1]))).reshape(-1)
    _, inv = np.unique(b, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def pair_ids(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    _, inv = np.unique(a * (int(b.max()) + 1) + b, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def masses(ids, w1):
    ids = np.asarray(ids, dtype=np.int64)
    w1 = np.asarray(w1, dtype=np.float64)
    K = int(ids.max()) + 1
    return np.bincount(ids, weights=w1, minlength=K), np.bincount(ids, weights=1.0 - w1, minlength=K)


# ---------------------------------------------------------------- exact finite-law recovery
def auc_star_closed(m1, m0):
    """1/2 + 1/4 sum_{a != b} |P1(a) P0(b) - P1(b) P0(a)| (diagonal terms vanish); groups without mass are dropped."""
    m1 = np.asarray(m1, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    keep = (m1 + m0) > 0
    p1 = m1[keep] / m1.sum()
    p0 = m0[keep] / m0.sum()
    tot = 0.0
    B = 512
    for i in range(0, len(p1), B):
        blk = np.abs(np.outer(p1[i:i + B], p0) - np.outer(p0[i:i + B], p1))
        tot += float(blk.sum())
    return 0.5 + 0.25 * tot


def auc_star_sorted(m1, m0):
    """Likelihood-ratio ranking with explicit tie groups (equal posterior rates r = m1 / (m1 + m0) under float equality):
    sum_g P1_g (P0 strictly below g + P0_g / 2)."""
    m1 = np.asarray(m1, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    keep = (m1 + m0) > 0
    a, b = m1[keep], m0[keep]
    r = a / (a + b)
    vals, inv = np.unique(r, return_inverse=True)
    P1 = np.bincount(inv, weights=a / m1.sum(), minlength=len(vals))
    P0 = np.bincount(inv, weights=b / m0.sum(), minlength=len(vals))
    below = np.concatenate([[0.0], np.cumsum(P0)[:-1]])
    return float((P1 * (below + 0.5 * P0)).sum())


def auc_star_exact_counts(c1, c0):
    """Exact rational AUC* for integer masses (empirical law): Fractions, LR ranking with exact tie groups."""
    c1 = [int(round(x)) for x in c1]
    c0 = [int(round(x)) for x in c0]
    n1, n0 = sum(c1), sum(c0)
    groups = {}
    for a, b in zip(c1, c0):
        if a + b == 0:
            continue
        key = Fraction(a, a + b)
        g = groups.setdefault(key, [0, 0])
        g[0] += a
        g[1] += b
    out, below = Fraction(0), Fraction(0)
    for key in sorted(groups):
        a, b = groups[key]
        out += Fraction(a, n1) * (below + Fraction(b, 2 * n0))
        below += Fraction(b, n0)
    return out


def plugin_mi(m1, m0, clip=MI_CLIP):
    """sum_w p(w) KL(Bern(r_w) || Bern(rbar)) in nats; r_w clipped to [clip, 1 - clip] (clip=None: exact, 0 log 0 = 0)."""
    m1 = np.asarray(m1, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    tot = m1 + m0
    T = tot.sum()
    rb = m1.sum() / T
    keep = tot > 0
    r = m1[keep] / tot[keep]
    if clip is not None:
        r = np.clip(r, clip, 1.0 - clip)

    def xlog(x, y):
        out = np.zeros_like(x)
        nz = x > 0
        out[nz] = x[nz] * np.log(x[nz] / y)
        return out
    kl = xlog(r, rb) + xlog(1.0 - r, 1.0 - rb)
    return float((tot[keep] / T * kl).sum())


def law_values(ids, w1, exact_counts=False):
    m1, m0 = masses(ids, w1)
    out = {'auc': auc_star_closed(m1, m0), 'auc_sorted': auc_star_sorted(m1, m0), 'mi': plugin_mi(m1, m0),
           'mi_unclipped': plugin_mi(m1, m0, clip=None), 'groups_with_mass': int(((m1 + m0) > 0).sum())}
    if exact_counts:
        out['auc_exact_rational'] = float(auc_star_exact_counts(m1, m0))
    return out


# ---------------------------------------------------------------- task costs and spends
def person_costs(V, y, nll_floor=None):
    V = np.asarray(V, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = np.arange(len(y))
    py = V[n, y]
    if nll_floor is not None:
        py = np.maximum(py, nll_floor)
    sq = (V ** 2).sum(axis=1)
    return {'nll': -np.log(py), 'brier': sq - 2.0 * V[n, y] + 1.0, 'acc': (np.argmax(V, axis=1) == y).astype(np.float64)}


def spends8(Vi_c, Vo_c, Vi_0, Vo_0, y_inc, y_occ, sex):
    """The 8 MODEL_FIT spends of a construction (c) over the identity (0) on the same people (definitions in the module doc)."""
    sex = np.asarray(sex, dtype=np.int64)
    ci, co = person_costs(Vi_c, y_inc), person_costs(Vo_c, y_occ)
    zi, zo = person_costs(Vi_0, y_inc), person_costs(Vo_0, y_occ)
    d_occ_br = co['brier'] - zo['brier']
    return {'inc_nll': float(np.mean(ci['nll'] - zi['nll'])), 'occ_nll': float(np.mean(co['nll'] - zo['nll'])),
            'inc_brier': float(np.mean(ci['brier'] - zi['brier'])), 'occ_brier': float(np.mean(d_occ_br)),
            'inc_acc': float(np.mean(zi['acc'] - ci['acc'])), 'occ_acc': float(np.mean(zo['acc'] - co['acc'])),
            'occ_brier_men': float(np.mean(d_occ_br[sex == 0])), 'occ_brier_women': float(np.mean(d_occ_br[sex == 1]))}


def expected_costs(V, L):
    """Expected per-person costs when the label is drawn from the distribution L (n, K) (revision R1's estimator, card DR_DRF_R1):
    E[-log V_y] = -sum_c L_c log V_c; E[sum_k (V_k - 1[y = k])^2] = sum_k V_k^2 - 2 sum_c L_c V_c + 1; E[1[argmax V = y]] =
    L[argmax V]. A one-hot L gives the realized costs."""
    V = np.asarray(V, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    n = np.arange(len(V))
    return {'nll': -(L * np.log(V)).sum(axis=1), 'brier': (V ** 2).sum(axis=1) - 2.0 * (L * V).sum(axis=1) + 1.0,
            'acc': L[n, np.argmax(V, axis=1)]}


def spends8_expected(Vi_c, Vo_c, Vi_0, Vo_0, L_inc, L_occ, L_guard, sex):
    """The 8 spends with expected costs: income under L_inc (the income teacher's out-of-fold posterior), occupation NLL / Brier /
    accuracy under L_occ (the occupation teacher's), the men's / women's occupation Brier under L_guard (each person's own SEX
    group's SEX-conditional occupation teacher)."""
    sex = np.asarray(sex, dtype=np.int64)
    ci, co = expected_costs(Vi_c, L_inc), expected_costs(Vo_c, L_occ)
    zi, zo = expected_costs(Vi_0, L_inc), expected_costs(Vo_0, L_occ)
    gc, gz = expected_costs(Vo_c, L_guard), expected_costs(Vo_0, L_guard)
    d_g = gc['brier'] - gz['brier']
    return {'inc_nll': float(np.mean(ci['nll'] - zi['nll'])), 'occ_nll': float(np.mean(co['nll'] - zo['nll'])),
            'inc_brier': float(np.mean(ci['brier'] - zi['brier'])), 'occ_brier': float(np.mean(co['brier'] - zo['brier'])),
            'inc_acc': float(np.mean(zi['acc'] - ci['acc'])), 'occ_acc': float(np.mean(zo['acc'] - co['acc'])),
            'occ_brier_men': float(np.mean(d_g[sex == 0])), 'occ_brier_women': float(np.mean(d_g[sex == 1]))}


def onehot(y, k):
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


# ---------------------------------------------------------------- household-clustered SEs
def cluster_se(d, unit):
    d = np.asarray(d, dtype=np.float64)
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    G = int(inv.max()) + 1
    s = np.bincount(inv.reshape(-1), weights=d - d.mean(), minlength=G)
    return float(math.sqrt(G / (G - 1) * float((s ** 2).sum())) / len(d))


def subgroup_se(d, mask, unit):
    d = np.asarray(d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    m = d[mask].mean()
    p = mask.mean()
    z = np.where(mask, (d - m) / p, 0.0)
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    G = int(inv.max()) + 1
    s = np.bincount(inv.reshape(-1), weights=z, minlength=G)
    return float(math.sqrt(G / (G - 1) * float((s ** 2).sum())) / len(d))


# ---------------------------------------------------------------- the 14 MODEL_SELECT conditions of one version
def valid_probs(P, k):
    P = np.asarray(P, dtype=np.float64)
    return bool(P.ndim == 2 and P.shape[1] == k and np.isfinite(P).all() and (P >= 0).all()
                and np.abs(P.sum(axis=1) - 1.0).max() <= 1e-9)


def conditions14(rel, refs, y, sex, unit):
    """rel / refs[name]: {'income': (n, 2), 'occupation': (n, 5)}; y: {'income', 'occupation'} labels. Returns the 12 clauses
    and the 2 guards with values, household-clustered SEs and inclusive point passes."""
    sex = np.asarray(sex, dtype=np.int64)
    own = {t: person_costs(rel[t], y[t], NLL_FLOOR_EVAL) for t in ('income', 'occupation')}
    clauses = []
    for t in ('income', 'occupation'):
        for ref in ('U0', 'Ucal'):
            r = person_costs(refs[ref][t], y[t], NLL_FLOOR_EVAL)
            diffs = {'accuracy_loss': r['acc'] - own[t]['acc'], 'nll_excess': own[t]['nll'] - r['nll'],
                     'brier_excess': own[t]['brier'] - r['brier']}
            for metric in ('accuracy_loss', 'nll_excess', 'brier_excess'):
                d = diffs[metric]
                v = float(d.mean())
                clauses.append({'task': t, 'reference': ref, 'metric': metric, 'value': v, 'cap': CAPS[metric],
                                'se': cluster_se(d, unit), 'point_pass': bool(v <= CAPS[metric])})
    u0 = person_costs(refs['U0']['occupation'], y['occupation'], NLL_FLOOR_EVAL)
    d = own['occupation']['brier'] - u0['brier']
    guards = []
    for g, name in ((0, 'sex0'), (1, 'sex1')):
        mask = sex == g
        v = float(d[mask].mean())
        guards.append({'group': name, 'value': v, 'cap': GUARD_CAP, 'se': subgroup_se(d, mask, unit), 'point_pass': bool(v <= GUARD_CAP)})
    return {'clauses': clauses, 'guards': guards, 'all_14_pass': bool(all(c['point_pass'] for c in clauses + guards))}


def paired_costs(a, b, y, sex, unit):
    """loss(a) - loss(b) per person on the same people (accuracy as a loss 1 - correct; NLL with the evaluator's 1e-12 floor;
    summed Brier), mean and household-clustered SE; the men's / women's subgroup means of the Brier differences of both tasks
    (linearized SE); total NLL = income + occupation NLL difference."""
    sex = np.asarray(sex, dtype=np.int64)
    out, nll = {}, {}
    for t in ('income', 'occupation'):
        ca, cb = person_costs(a[t], y[t], NLL_FLOOR_EVAL), person_costs(b[t], y[t], NLL_FLOOR_EVAL)
        d = {'acc_loss': cb['acc'] - ca['acc'], 'nll': ca['nll'] - cb['nll'], 'brier': ca['brier'] - cb['brier']}
        nll[t] = d['nll']
        for m_ in ('acc_loss', 'nll', 'brier'):
            out[f'{t}|{m_}'] = {'value': float(d[m_].mean()), 'se': cluster_se(d[m_], unit)}
        for g, name in ((0, 'men_brier'), (1, 'women_brier')):
            mask = sex == g
            out[f'{t}|{name}'] = {'value': float(d['brier'][mask].mean()), 'se': subgroup_se(d['brier'], mask, unit)}
    tot = nll['income'] + nll['occupation']
    out['total|nll'] = {'value': float(tot.mean()), 'se': cluster_se(tot, unit)}
    return out


def wire_hash(P):
    """SHA-256 of one recipient's released wire: the float64 probability bytes followed by the int64 argmax bytes."""
    P = np.ascontiguousarray(np.asarray(P, dtype=np.float64))
    h = hashlib.sha256(P.tobytes())
    h.update(np.ascontiguousarray(np.argmax(P, axis=1).astype(np.int64)).tobytes())
    return h.hexdigest()


def distinct_counts(rel):
    i, o = row_ids(rel['income']), row_ids(rel['occupation'])
    return {'income': int(len(np.unique(i))), 'occupation': int(len(np.unique(o))), 'pair': int(len(np.unique(pair_ids(i, o))))}


# ---------------------------------------------------------------- the aggregate paired screen (NRM1 RULES.md 5, my reading)
REGISTERED = tuple(f'S{v}R{r}k{k}' for v in (0, 1) for r in (1, 2) for k in (0, 1))


def parse_cell(c):
    """'S<v>R<r>k<k>' -> (v, r, k)."""
    v = int(c[1])
    r = int(c[c.index('R') + 1:c.index('k')])
    k = int(c[c.index('k') + 1:])
    return v, r, k


def aggregate(cells, ses, use=REGISTERED, se_mc=None):
    """Blocks b = (version, repetition); m_b = mean over the block's reader-seed cells; x = mean of the block means; version mean =
    mean of that version's block means; se_hh = sqrt(sum_c w_c s_c^2) with w_c the cell's weight in x; se_rs = sqrt(sum_b (m_b -
    x)^2 / n_blocks); se_agg = max(sqrt(se_hh^2 + se_rs^2 + se_mc^2), 0.001); exact alias iff every cell value and SE is exactly 0.
    Missing / non-finite registered cells -> complete False."""
    use = [c for c in use]
    ok = all(c in cells and c in ses and math.isfinite(float(cells[c])) and math.isfinite(float(ses[c])) for c in use)
    if not ok:
        return {'complete': False, 'missing': [c for c in use if c not in cells or c not in ses]}
    blocks = {}
    for c in use:
        v, r, k = parse_cell(c)
        blocks.setdefault((v, r), []).append(c)
    mb = {b: float(np.mean([float(cells[c]) for c in cs])) for b, cs in blocks.items()}
    x = float(np.mean(list(mb.values())))
    vers = sorted({b[0] for b in blocks})
    vm = {v: float(np.mean([m for b, m in mb.items() if b[0] == v])) for v in vers}
    w = {}
    for b, cs in blocks.items():
        for c in cs:
            w[c] = 1.0 / (len(blocks) * len(cs))
    se_hh = math.sqrt(sum(w[c] * float(ses[c]) ** 2 for c in use))
    se_rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / len(mb))
    smc = 0.0 if se_mc is None else math.sqrt(sum(w[c] * float(se_mc.get(c, 0.0)) ** 2 for c in use))
    alias = all(float(cells[c]) == 0.0 and float(ses[c]) == 0.0 for c in use)
    se_agg = 0.0 if alias else max(math.sqrt(se_hh ** 2 + se_rs ** 2 + smc ** 2), 0.001)
    return {'complete': True, 'x': x, 'version_means': vm, 'block_means': {f'V{b[0]}R{b[1]}': m for b, m in mb.items()},
            'se_hh': se_hh, 'se_rs': se_rs, 'se_mc': smc, 'se_agg': se_agg, 'alias': alias, 'n_cells': len(use)}


def benefit_label(ag, thr=0.0):
    """Pair benefit: AGG_POSITIVE iff x - 2 se_agg > thr and both version means > thr; POINT_ONLY iff x > thr otherwise;
    NOT_POSITIVE iff x <= thr; ALIAS_ZERO; INCOMPLETE. (thr = 0.020: the target screen with its own label names.)"""
    if not ag.get('complete'):
        return 'INCOMPLETE'
    if ag['alias']:
        return 'ALIAS_ZERO'
    if ag['x'] - 2 * ag['se_agg'] > thr and all(m > thr for m in ag['version_means'].values()):
        return 'AGG_POSITIVE'
    return 'AGG_POSITIVE_POINT_ONLY' if ag['x'] > thr else 'AGG_NOT_POSITIVE'


def target_label(ag, thr=0.020):
    b = benefit_label(ag, thr)
    return {'AGG_POSITIVE': 'TARGET_AGG_RESOLVED', 'AGG_POSITIVE_POINT_ONLY': 'TARGET_AGG_POINT_ONLY'}.get(b, 'TARGET_AGG_NOT_MET')


def recipient(cells, ses, use=REGISTERED, allowance=0.010):
    """Per version v: r_v = mean of v's block means, se_v from v's cells only (the aggregate construction on those cells). Gate
    passes iff r_v <= allowance for both versions (inclusive) or the view is an exact alias. Label (first applicable): ALIAS_ZERO,
    RECIPIENT_AGG_RESOLVED_WORSE, RECIPIENT_AGG_POINT_FAILED, RECIPIENT_AGG_RESOLVED, RECIPIENT_AGG_EDGE."""
    per = {}
    for v in sorted({parse_cell(c)[0] for c in use}):
        per[v] = aggregate(cells, ses, [c for c in use if parse_cell(c)[0] == v])
    if not all(p['complete'] for p in per.values()):
        return {'label': 'INCOMPLETE', 'gate': False, 'per_version': per}
    if all(p['alias'] for p in per.values()):
        return {'label': 'ALIAS_ZERO', 'gate': True, 'per_version': per}
    r = {v: p['x'] for v, p in per.items()}
    s = {v: p['se_agg'] for v, p in per.items()}
    if any(r[v] - 2 * s[v] > allowance for v in r):
        lab = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif any(r[v] > allowance for v in r):
        lab = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(r[v] + 2 * s[v] <= allowance for v in r):
        lab = 'RECIPIENT_AGG_RESOLVED'
    else:
        lab = 'RECIPIENT_AGG_EDGE'
    return {'label': lab, 'gate': bool(all(r[v] <= allowance for v in r)), 'r_v': r, 'se_v': s, 'per_version': per,
            'cells_above_allowance': sorted(c for c in use if float(cells[c]) > allowance)}


def weighted_auc(y, s, w):
    """AUC of fixed scores s with per-person weights w (ties 1/2): sum over positives / negatives of weighted Mann-Whitney."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(s, kind='mergesort')
    s_o, y_o, w_o = s[order], y[order], w[order]
    vals, start = np.unique(s_o, return_index=True)
    wp = np.add.reduceat(np.where(y_o, w_o, 0.0), start)
    wn = np.add.reduceat(np.where(~y_o, w_o, 0.0), start)
    below = np.concatenate([[0.0], np.cumsum(wn)[:-1]])
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def weighted_auc_batch(y, s, W):
    """weighted_auc for each row of the weight matrix W (B, n), with one sort of the fixed scores s."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    order = np.argsort(s, kind='mergesort')
    s_o, y_o = s[order], y[order]
    _, start = np.unique(s_o, return_index=True)
    Wo = W[:, order]
    wp = np.add.reduceat(np.where(y_o[None, :], Wo, 0.0), start, axis=1)
    wn = np.add.reduceat(np.where(~y_o[None, :], Wo, 0.0), start, axis=1)
    below = np.concatenate([np.zeros((W.shape[0], 1)), np.cumsum(wn, axis=1)[:, :-1]], axis=1)
    return (wp * (below + 0.5 * wn)).sum(axis=1) / (wp.sum(axis=1) * wn.sum(axis=1))


def auc(y, s):
    """Unweighted AUC with ties 1/2."""
    return weighted_auc(y, s, np.ones(len(np.asarray(s))))


def household_bootstrap_weights(unit_sc, seed, B=400):
    """B household-bootstrap weight vectors over the SCORE people: households (sorted unique ids) resampled with replacement
    (numpy default_rng(seed), one integers(0, G, G) draw per replicate); a person's weight is the count of its household."""
    g, inv = np.unique(np.asarray(unit_sc), return_inverse=True)
    rng = np.random.default_rng(int(seed))
    W = np.empty((B, len(unit_sc)))
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))
        W[b] = cnt[inv.reshape(-1)]
    return W


def household_parts(unit, k, tag):
    """Household-separated parts: households ordered by SHA-256('<tag>|<household>') and dealt round-robin into k parts."""
    unit = np.asarray(unit)
    groups = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups]
    part = np.empty(len(groups), dtype=np.int64)
    part[np.argsort(keys, kind='stable')] = np.arange(len(groups)) % k
    return part[np.searchsorted(groups, unit)]


# ---------------------------------------------------------------- SEX-blind task-only coarsening path (greedy least-total-NLL merges)
def _atom_nll(n, sq, cy):
    """Total NLL of atoms from aggregates (people count n, summed posterior sq, label counts cy) with clipped mean tables."""
    T = np.where(n[:, None] > 0, sq / np.maximum(n, 1.0)[:, None], 1.0 / sq.shape[1])
    T = renorm_clip(T)
    return -(cy * np.log(T)).sum(axis=1)


def coarsening_path(cells_inc, cells_occ, qI, qO, L_inc, L_occ, max_steps=70, max_spend=0.08):
    """Greedy merges from the identity: at each step the pair of used atoms of one recipient whose merge raises the total
    (income + occupation) NLL least (income examined first; strict improvement needed to prefer occupation; first pair in
    (lower atom, higher atom) lexicographic order on exact ties); the higher atom's cells move to the lower atom. L_inc / L_occ are
    the per-person label distributions entering the NLL: one-hot realized labels (card DR_DRF_v1) or the out-of-fold teacher
    posteriors (expected costs, card DR_DRF_R1). Returns the list of (A_inc, A_occ, step, total-NLL spend in means, the chosen
    increase, the gap to the runner-up increase)."""
    N = len(L_inc)
    A = {'inc': identity_maps()[0].copy(), 'occ': identity_maps()[1].copy()}
    cells = {'inc': np.asarray(cells_inc), 'occ': np.asarray(cells_occ)}
    Q = {'inc': np.asarray(qI, dtype=np.float64), 'occ': np.asarray(qO, dtype=np.float64)}
    Y = {'inc': np.asarray(L_inc, dtype=np.float64), 'occ': np.asarray(L_occ, dtype=np.float64)}
    M = {'inc': CAPACITY[0], 'occ': CAPACITY[1]}

    def agg(side):
        atom = A[side][cells[side]]
        K = Q[side].shape[1]
        n = np.bincount(atom, minlength=M[side]).astype(np.float64)
        sq = np.column_stack([np.bincount(atom, weights=Q[side][:, j], minlength=M[side]) for j in range(K)])
        cy = np.column_stack([np.bincount(atom, weights=Y[side][:, j], minlength=M[side]) for j in range(K)])
        return n, sq, cy

    def total_nll():
        return sum(float(_atom_nll(*agg(s)).sum()) for s in ('inc', 'occ'))

    base = total_nll()
    path = [(A['inc'].copy(), A['occ'].copy(), 0, 0.0, None, None)]
    step = 0
    while step < max_steps:
        cands = []
        for side in ('inc', 'occ'):
            n, sq, cy = agg(side)
            used = np.flatnonzero(n > 0)
            if len(used) < 2:
                continue
            cur = _atom_nll(n, sq, cy)
            ia, ib = np.triu_indices(len(used), 1)
            a, b = used[ia], used[ib]
            inc = _atom_nll(n[a] + n[b], sq[a] + sq[b], cy[a] + cy[b]) - cur[a] - cur[b]
            k = int(np.argmin(inc))
            srt = np.sort(inc)
            cands.append((float(inc[k]), side, int(a[k]), int(b[k]), srt))
        if not cands:
            break
        best = cands[0]
        for c in cands[1:]:
            if c[0] < best[0]:
                best = c
        allv = np.sort(np.concatenate([c[4] for c in cands]))
        gap = float(allv[1] - allv[0]) if len(allv) > 1 else None
        _, side, a, b, _ = best
        A[side][A[side] == b] = a
        step += 1
        spend = (total_nll() - base) / N
        path.append((A['inc'].copy(), A['occ'].copy(), step, spend, best[0], gap))
        if spend > max_spend:
            break
    return path


def same_partition(A, B):
    """True iff two cell -> atom maps induce the same partition of the cells (equal up to relabelling the atoms)."""
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        return False
    pairs = set(zip(A.tolist(), B.tolist()))
    return len(pairs) == len(set(A.tolist())) == len(set(B.tolist()))


# ---------------------------------------------------------------- comparison and record helpers
def sha256_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def cmp_num(name, mine, theirs, tol=TOL):
    ok = (theirs is not None and mine is not None and math.isfinite(float(mine)) and math.isfinite(float(theirs))
          and abs(float(mine) - float(theirs)) <= tol)
    return {'name': name, 'verifier': mine, 'record': theirs,
            'abs_diff': (abs(float(mine) - float(theirs)) if (mine is not None and theirs is not None) else None),
            'tol': tol, 'result': 'PASS' if ok else 'FAIL'}


def cmp_eq(name, mine, theirs):
    return {'name': name, 'verifier': mine, 'record': theirs, 'result': 'PASS' if mine == theirs else 'FAIL'}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)
    return o


def summarize(items):
    fails = [i['name'] for i in items if i['result'] != 'PASS']
    return {'n': len(items), 'n_fail': len(fails), 'failed': fails, 'result': 'PASS' if not fails else 'FAIL'}


def dump(path, obj):
    with open(path, 'x') as f:
        json.dump(jsonable(obj), f, indent=1, sort_keys=True)
