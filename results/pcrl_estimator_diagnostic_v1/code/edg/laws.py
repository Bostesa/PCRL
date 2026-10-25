"""Known finite laws of the EDG1 diagnostic (PROTOCOL.md section 3). Every law is a finite joint law over INTERNAL cells c with
masses p[c] = P(C = c) and posteriors q[c] = P(S = 1 | C = c). Each view (income-like, occupation-like, linked pair, or a refinement
level) is a deterministic function of c: the EMITTED probability vector(s) of c. Identical emitted wires are one observable value:
`drf.finite_law.merge_wires` merges them before any recovery computation, so internal cell ids never act as extra inputs.

Families (fixed before any simulation; parameters below):
  IND   income view i (16 atoms) and occupation view j (64 atoms), independent of each other and of S; pair = (i, j), 1024 values.
  REF   coarse occupation-like view g (16 atoms, moderate signal, 8 posterior levels each shared by two atoms = tied likelihood
        ratios) refined three times (each atom split into 4 children independently of S given the parent): levels L16, L64, L256,
        L1024 of the SAME people. The population optimum is identical at every level (children inherit the parent's posterior).
  RED   income view = g (16 atoms, the REF coarse law); occupation view = (g // 4, z): 4 coarse groups x 16 noise levels z drawn
        independently of (S, g); pair = (g, z), 256 values. S is independent of the occupation view given the income view, so the
        pair adds no population-optimal recovery: AUC*(pair) = AUC*(income) > AUC*(occupation).
  COMP  income i (16), occupation j (16), product occupancy a_i b_j and q(i, j) = pi + delta u_i v_j with sum_i a_i u_i = 0 and
        sum_j b_j v_j = 0: each view alone is uninformative (AUC* = 1/2), the pair is informative. Occupation cells 14 and 15 emit
        the IDENTICAL vector (observable occupation support 15, pair 240): a merging check.
Class frequency pi in {0.5, 0.2}; occupancy 'uniform' or 'skewed' (Zipf weights 1/k over a fixed permutation of the atoms; skewed
child shares 0.55 / 0.25 / 0.13 / 0.07 in a fixed per-parent order)."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.special import ndtr

from . import RESULTS  # noqa: F401  (path setup)
from drf import finite_law as FL

LAW_SEED = 2026102511
FAMILIES = ('IND', 'REF', 'RED', 'COMP')
PIS = (0.5, 0.2)
OCCS = ('uniform', 'skewed')
BETA = 1.0                                  # logit-scale signal of the moderate families (REF, RED)
Z_LEVELS = np.linspace(-1.0, 1.0, 8)        # 8 posterior levels; coarse atoms 2k and 2k + 1 share level k (tied ratios)
CHILD_SKEW = (0.55, 0.25, 0.13, 0.07)
DELTA_SCALE = 0.24                          # COMP: delta = 0.24 sqrt(pi (1 - pi)) (0.12 at pi = 0.5, 0.096 at pi = 0.2)
U_PATTERN = (1.0, -1.0, 0.5, -0.5)
V_PATTERN = (1.0, -1.0, -0.5, 0.5)
HH_SIZES = (1, 2, 3, 4)
HH_PROBS = (0.62, 0.27, 0.08, 0.03)         # mean 1.52 people per sampled household (DRF1 MODEL_FIT: 20,000 / 13,337 = 1.50)
HH_RHO = 0.5                                # Gaussian-copula household correlation of the latent uniforms (outputs)
HH_OPPOSITE = 0.8                           # P(second member has the opposite S of the first)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(t):
    t = np.asarray(t, dtype=np.float64)
    e = np.exp(t - t.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _inc_vec(tau):
    t = _sig(np.asarray(tau, dtype=np.float64))
    return np.column_stack((1.0 - t, t))


def occupancy(k, occ, rng):
    if occ == 'uniform':
        return np.full(k, 1.0 / k)
    w = 1.0 / np.arange(1, k + 1)
    w = w / w.sum()
    return w[rng.permutation(k)]


def child_shares(occ, rng):
    if occ == 'uniform':
        return np.full(4, 0.25)
    return np.asarray(CHILD_SKEW)[rng.permutation(4)]


def solve_alpha(w, z, beta, pi):
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.dot(w, _sig(mid + beta * z))) < pi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _coarse_signal(pi, occ, rng):
    w = occupancy(16, occ, rng)
    z = np.repeat(Z_LEVELS, 2)
    alpha = solve_alpha(w, z, BETA, pi)
    q = _sig(alpha + BETA * z)
    return w, z, q


def build(family, pi, occ):
    """The fixed law of (family, pi, occupancy): internal masses p, posteriors q, per-view emitted vectors, merged observable
    laws and population quantities. Deterministic (seeded by the cell identity)."""
    key = f'{LAW_SEED}|{family}|{pi}|{occ}'
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:15], 16))
    law = {'family': family, 'pi': float(pi), 'occ': occ}
    if family == 'IND':
        a, b = occupancy(16, occ, rng), occupancy(64, occ, rng)
        p = np.outer(a, b).reshape(-1)
        q = np.full(p.shape, float(pi))
        inc = _inc_vec(rng.normal(0.0, 1.0, 16))
        occv = _softmax(rng.normal(0.0, 0.8, (64, 5)))
        ii, jj = np.repeat(np.arange(16), 64), np.tile(np.arange(64), 16)
        vec = {'income': inc[ii], 'occupation': occv[jj]}
        views = {'income': vec['income'], 'occupation': vec['occupation'], 'pair': (vec['income'], vec['occupation'])}
        law['comparisons'] = [('pair', 'income'), ('pair', 'occupation'), ('occupation', 'income')]
    elif family == 'REF':
        w, z, qg = _coarse_signal(pi, occ, rng)
        sh = [np.stack([child_shares(occ, rng) for _ in range(16 * 4 ** l)]) for l in range(3)]
        th16 = np.column_stack((1.2 * (np.log(qg / (1 - qg)) - math.log(pi / (1 - pi))) + rng.normal(0, 0.3, 16),
                                rng.normal(0.0, 0.7, (16, 4))))
        th = [th16]
        for l in range(3):
            th.append(np.repeat(th[-1], 4, axis=0) + rng.normal(0.0, 0.15, (16 * 4 ** (l + 1), 5)))
        # internal cell c = level-1024 index; ancestors by integer division
        c = np.arange(16 * 64)
        g = c // 64
        p = w[g].copy()
        for l in range(3):
            parent = c // (4 ** (3 - l))                       # index at level 16 * 4**l
            child = (c // (4 ** (2 - l))) % 4                  # which child at level l + 1
            p = p * sh[l][parent, child]
        q = qg[g]
        lev = {f'L{16 * 4 ** l}': _softmax(th[l])[c // (4 ** (3 - l))] for l in range(4)}
        views = dict(lev)
        law['comparisons'] = [('L64', 'L16'), ('L256', 'L16'), ('L1024', 'L16')]
        law['ref_struct'] = {'w': w.tolist(), 'q_coarse': qg.tolist(), 'shares': [s.tolist() for s in sh]}
        vec = {'levels': lev}
    elif family == 'RED':
        w, z, qg = _coarse_signal(pi, occ, rng)
        e = occupancy(16, occ, rng)
        tau = 1.2 * (np.log(qg / (1 - qg)) - math.log(pi / (1 - pi))) + rng.normal(0, 0.3, 16)
        inc = _inc_vec(tau)
        th_o = np.column_stack((np.repeat([tau[4 * h:4 * h + 4].mean() for h in range(4)], 16) + rng.normal(0, 0.3, 64),
                                rng.normal(0.0, 0.7, (64, 4))))
        occv = _softmax(th_o)
        gg, zz = np.repeat(np.arange(16), 16), np.tile(np.arange(16), 16)
        p = w[gg] * e[zz]
        q = qg[gg]
        vec = {'income': inc[gg], 'occupation': occv[(gg // 4) * 16 + zz]}
        views = {'income': vec['income'], 'occupation': vec['occupation'], 'pair': (vec['income'], vec['occupation'])}
        law['comparisons'] = [('pair', 'income'), ('income', 'occupation'), ('pair', 'occupation')]
    elif family == 'COMP':
        a, b = occupancy(16, occ, rng), occupancy(16, occ, rng)
        u = np.tile(U_PATTERN, 4); u = u - np.dot(a, u); u = u / np.abs(u).max()
        v = np.tile(V_PATTERN, 4); v = v - np.dot(b, v); v = v / np.abs(v).max()
        delta = DELTA_SCALE * math.sqrt(pi * (1 - pi))
        ii, jj = np.repeat(np.arange(16), 16), np.tile(np.arange(16), 16)
        p = a[ii] * b[jj]
        q = pi + delta * u[ii] * v[jj]
        inc = _inc_vec(1.5 * u + rng.normal(0, 0.3, 16))
        th_o = np.column_stack((1.5 * v + rng.normal(0, 0.3, 16), rng.normal(0.0, 0.7, (16, 4))))
        th_o[15] = th_o[14]                                     # identical emitted vector of occupation cells 14 and 15
        occv = _softmax(th_o)
        vec = {'income': inc[ii], 'occupation': occv[jj]}
        views = {'income': vec['income'], 'occupation': vec['occupation'], 'pair': (vec['income'], vec['occupation'])}
        law['comparisons'] = [('pair', 'income'), ('pair', 'occupation'), ('occupation', 'income')]
        law['comp'] = {'a': a.tolist(), 'b': b.tolist(), 'u': u.tolist(), 'v': v.tolist(), 'delta': delta}
        law['internal_id_auc_pair'] = float(FL.auc_star(p * q, p * (1 - q)))   # WRONG on purpose: unmerged internal ids
    else:
        raise ValueError(family)
    law['p'], law['q'] = p, q
    law['vectors'] = vec
    law['views'] = {}
    for name, wires in views.items():
        uniq, m1, m0, inv = FL.merge_wires(wires, p * q, p * (1 - q), return_inverse=True)
        law['views'][name] = {'inv': inv, 'm1': m1, 'm0': m0, 'K': int(len(m1)), 'wires': uniq,
                              'A': float(FL.auc_star(m1, m0)), 'lr': FL.lr_scores(m1, m0),
                              'post': m1 / (m1 + m0)}
    _check_vectors(law)
    law['checks'] = analytic_checks(law)
    return law


def _check_vectors(law):
    for name, v in law['views'].items():
        comps = v['wires'] if isinstance(v['wires'], tuple) else (v['wires'],)
        for a in comps:
            if not (np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() and np.abs(a.sum(1) - 1).max() <= 1e-12):
                raise AssertionError(f'{law["family"]} {name}: invalid probability vectors')
    if abs(float(np.sum(law['p'])) - 1.0) > 1e-12 or (law['q'] <= 0).any() or (law['q'] >= 1).any():
        raise AssertionError('invalid law')


def analytic_checks(law):
    """Population relations the family is built to satisfy (exact up to float rounding, tolerance 1e-12)."""
    V, fam, out = law['views'], law['family'], {}
    out['marginal_pi_error'] = float(abs(np.dot(law['p'], law['q']) - law['pi']))
    if fam == 'IND':
        out['all_views_half'] = bool(all(abs(v['A'] - 0.5) <= 1e-12 for v in V.values()))
    if fam == 'REF':
        out['levels_equal_A'] = bool(all(abs(V[k]['A'] - V['L16']['A']) <= 1e-12 for k in V))
        # children inherit the parent's posterior: the posterior of every level-L wire equals the posterior of its coarse atom
        g_of = law['views']['L16']['inv']
        ok = True
        for k in V:
            post_c = V[k]['post'][V[k]['inv']]
            ok &= bool(np.abs(post_c - V['L16']['post'][g_of]).max() <= 1e-12)
        out['refinement_irrelevant_given_parent'] = ok
    if fam == 'RED':
        post_pair = V['pair']['post'][V['pair']['inv']]
        post_inc = V['income']['post'][V['income']['inv']]
        out['pair_posterior_equals_income_posterior'] = bool(np.abs(post_pair - post_inc).max() <= 1e-12)
        out['pair_A_equals_income_A'] = bool(abs(V['pair']['A'] - V['income']['A']) <= 1e-12)
        out['occupation_A_below_income_A'] = bool(V['occupation']['A'] < V['income']['A'] - 1e-6)
    if fam == 'COMP':
        out['single_views_half'] = bool(abs(V['income']['A'] - 0.5) <= 1e-12 and abs(V['occupation']['A'] - 0.5) <= 1e-12)
        out['pair_informative'] = bool(V['pair']['A'] > 0.5 + 1e-3)
        out['occupation_support_after_merge'] = V['occupation']['K']
        out['merging_lowers_internal_id_value'] = bool(law['internal_id_auc_pair'] > V['pair']['A'] + 1e-9)
    out['all_pass'] = bool(all(v for k, v in out.items() if isinstance(v, bool)) and out['marginal_pi_error'] <= 1e-12)
    return out


def describe(law):
    d = {'family': law['family'], 'pi': law['pi'], 'occ': law['occ'], 'internal_cells': int(len(law['p'])),
         'views': {k: {'observable_support': v['K'], 'A_population_optimal_auc': v['A'],
                       'min_cell_prob': float((v['m1'] + v['m0']).min()), 'max_cell_prob': float((v['m1'] + v['m0']).max())}
                   for k, v in law['views'].items()},
         'comparisons': law['comparisons'], 'checks': law['checks'],
         'p_sha256': hashlib.sha256(np.ascontiguousarray(law['p']).tobytes()).hexdigest(),
         'q_sha256': hashlib.sha256(np.ascontiguousarray(law['q']).tobytes()).hexdigest()}
    if 'internal_id_auc_pair' in law:
        d['internal_id_auc_pair_unmerged'] = law['internal_id_auc_pair']
    return d


# ---------------------------------------------------------------- sampling
def draw(law, n, rng):
    """n independent people: internal cells and S."""
    c = rng.choice(len(law['p']), size=n, p=law['p'])
    s = (rng.random(n) < law['q'][c]).astype(np.int64)
    return c, s, np.arange(n)


def draw_households(law, n_target, rng):
    """Clustered sensitivity (REF, pi = 0.5 only): explicit synthetic households whose person-level marginal law equals the REF law
    exactly. S: first member Bern(1/2); second member opposite of the first with probability HH_OPPOSITE; further members Bern(1/2).
    Coarse atom g | S by a Gaussian copula with a household latent (rho = HH_RHO); each refinement level's child | parent likewise
    with its own household latent (independent of S). Returns internal cells, S and household ids."""
    if law['family'] != 'REF' or law['pi'] != 0.5:
        raise ValueError('the clustered sensitivity scenario is defined for REF at pi = 0.5 only')
    st = law['ref_struct']
    w, qg = np.asarray(st['w']), np.asarray(st['q_coarse'])
    n_h = int(round(n_target / float(np.dot(HH_SIZES, HH_PROBS))))
    size = rng.choice(HH_SIZES, size=n_h, p=HH_PROBS)
    unit = np.repeat(np.arange(n_h), size)
    n = len(unit)
    first = np.r_[0, np.cumsum(size)[:-1]]
    pos = np.arange(n) - np.repeat(first, size)
    s = (rng.random(n) < 0.5).astype(np.int64)
    sec = np.flatnonzero(pos == 1)
    flip = rng.random(len(sec)) < HH_OPPOSITE
    s[sec] = np.where(flip, 1 - s[sec - 1], s[sec - 1])
    sq, sq1 = math.sqrt(HH_RHO), math.sqrt(1.0 - HH_RHO)

    def u_of():
        return ndtr(sq * rng.normal(size=n_h)[unit] + sq1 * rng.normal(size=n))
    cdf = {1: np.cumsum(w * qg / np.dot(w, qg)), 0: np.cumsum(w * (1 - qg) / np.dot(w, 1 - qg))}
    u = u_of()
    g = np.where(s == 1, np.searchsorted(cdf[1], u, side='right'), np.searchsorted(cdf[0], u, side='right'))
    g = np.minimum(g, 15)
    idx = g
    for l in range(3):
        shares = np.asarray(st['shares'][l])[idx]               # (n, 4)
        u = u_of()
        ch = np.minimum((u[:, None] > np.cumsum(shares, 1)).sum(1), 3)
        idx = idx * 4 + ch
    return idx, s, unit


# ---------------------------------------------------------------- registry
def all_laws():
    return {(f, pi, o): build(f, pi, o) for f in FAMILIES for pi in PIS for o in OCCS}


def write_laws(path):
    laws = all_laws()
    rec = {'schema': 'EDG1-laws-v1', 'law_seed': LAW_SEED, 'beta': BETA, 'z_levels': Z_LEVELS.tolist(),
           'delta_scale': DELTA_SCALE, 'child_skew': CHILD_SKEW, 'household': {'sizes': HH_SIZES, 'probs': HH_PROBS,
                                                                               'rho': HH_RHO, 'opposite': HH_OPPOSITE},
           'laws': {f'{f}|{pi}|{o}': describe(l) for (f, pi, o), l in laws.items()},
           'all_checks_pass': bool(all(l['checks']['all_pass'] for l in laws.values()))}
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    return rec
