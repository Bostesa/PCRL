"""Exact reference for the propensity-routed finite-cell optimizer (role exact_reference, campaign pcrl_finite_solver_adequacy_v1).

Invented finite laws only (../laws.json); no real data, no labels. Every reference quantity (posteriors, task metrics, exact Bayes
AUC, plug-in information, the exact objective, the routing rule's per-state linear costs and their tilt breakpoints, the partition
enumeration) is computed here from the formulas of card FA_PRC_v1 section 1. nrm.fa.prc is imported only to RUN the code under test
(twin_book, protected_path, route, update_book, objective) and compare it with these references.

  python3 fsa_exact.py selftest            invented inputs that are not the declared laws
  python3 fsa_exact.py narrow --out F      STEPs 2-3 (all laws, twin seeds 0 and 1)
  python3 fsa_exact.py enum --out F        STEP 4 (relaxation enumeration)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np

HERE = Path(__file__).resolve().parent
ROLE = HERE.parent
REPO = HERE.parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

QSCALE = 10000          # task posteriors are exact integers in units of 1e-4


# ============================================================ laws
def load_laws(path=None):
    L = json.loads(Path(path or ROLE / 'laws.json').read_text())
    out = {}
    for name, law in L['laws'].items():
        src = law if 'states' in law else L['laws'][law['same_population_law_as']]
        div = 1 if 'states' in law else int(law['row_divisor'])
        out[name] = build_law(name, src['states'], int(law['N']), div, law, L['common'])
    return out, L['common']


def build_law(name, states, N, div, meta, common):
    if any(s['n'] % div for s in states):
        raise ValueError('row divisor does not divide the counts')
    n = np.array([s['n'] // div for s in states], dtype=np.int64)
    if int(n.sum()) != N:
        raise ValueError(f'{name}: counts do not sum to N')
    u = np.array([s['pi_pct'] for s in states], dtype=np.int64)
    arr = lambda key: np.array([s[key] for s in states], dtype=np.int64)
    i0, i1, o0, o1 = arr('inc_S0_pct'), arr('inc_S1_pct'), arr('occ_S0_pct'), arr('occ_S1_pct')
    qI_int = (100 - u)[:, None] * i0 + u[:, None] * i1
    qO_int = (100 - u)[:, None] * o0 + u[:, None] * o1
    law = {'name': name, 'N': N, 'n': n, 'u': u, 'ids': [s['id'] for s in states], 'm': len(states),
           'qI_int': qI_int, 'qO_int': qO_int, 'qO0_int': 100 * o0, 'qO1_int': 100 * o1,
           'w': n / N, 'pi': u / 100.0, 'qI': qI_int / QSCALE, 'qO': qO_int / QSCALE, 'qO0': o0 / 100.0, 'qO1': o1 / 100.0,
           'witness': meta.get('witness'), 'comparator': meta.get('zero_recovery_comparator'),
           'collision_groups': meta.get('collision_groups'), 'kind': meta.get('kind')}
    law['a'] = n * u                       # S = 1 mass per state, units 1/(N * 100)
    law['b'] = n * (100 - u)               # S = 0 mass
    law['p1'] = float(Fraction(int(law['a'].sum()), N * 100))
    law['p0'] = 1.0 - law['p1']
    return law


def code_E(law):
    rep = lambda x: np.repeat(x, law['n'], axis=0)
    return {'qI': rep(law['qI']), 'qO': rep(law['qO']), 'pi': rep(law['pi']), 'qO0': rep(law['qO0']), 'qO1': rep(law['qO1'])}


def to_state(law, rows):
    first = np.concatenate([[0], np.cumsum(law['n'])[:-1]])
    s = np.asarray(rows)[first]
    if not np.array_equal(np.repeat(s, law['n']), rows):
        raise AssertionError('rows of one support state were routed to different cells')
    return s.astype(np.int64)


def to_rows(law, s):
    return np.repeat(np.asarray(s, dtype=np.int64), law['n'])


def canon(k, l):
    """Partition of states as a label-free key (cells relabelled by first occurrence, separately per task)."""
    def rl(g):
        mp = {}
        return tuple(mp.setdefault(int(v), len(mp)) for v in g)
    return rl(k), rl(l)


# ============================================================ exact reference metrics of a release
def group_tables(law, g, q_int):
    """Exact posterior mean of each group (Fractions), keyed by group label."""
    T = {}
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        den = int(law['n'][idx].sum()) * QSCALE
        num = (law['n'][idx, None] * q_int[idx]).sum(0)
        T[int(grp)] = tuple(Fraction(int(v), den) for v in num)
    return T


def exact_argmax(v):
    best = 0
    for j in range(1, len(v)):
        if v[j] > v[best]:
            best = j
    return best


def bayes_auc(law, keys):
    """Exact Bayes AUC of S from the released value of each state (identical values merged; ties count 1/2). Returns a Fraction."""
    a, b = {}, {}
    for x, key in enumerate(keys):
        a[key] = a.get(key, 0) + int(law['a'][x])
        b[key] = b.get(key, 0) + int(law['b'][x])
    vals = list(a)
    A, B = sum(a.values()), sum(b.values())
    num2 = 0
    for i in vals:
        for j in vals:
            lhs, rhs = a[i] * b[j], a[j] * b[i]
            if lhs > rhs:
                num2 += 2 * lhs
            elif lhs == rhs:
                num2 += lhs
    return Fraction(num2, 2 * A * B)


def mi_keys(law, keys):
    """pi-weighted plug-in I(S; key) in nats (with exact pi this is the population mutual information)."""
    a, b = {}, {}
    for x, key in enumerate(keys):
        a[key] = a.get(key, 0) + int(law['a'][x])
        b[key] = b.get(key, 0) + int(law['b'][x])
    A, B = sum(a.values()), sum(b.values())
    rb = A / (A + B)
    out = 0.0
    for key in a:
        t = a[key] + b[key]
        r = a[key] / t
        kl = (r * np.log(r / rb) if r > 0 else 0.0) + ((1 - r) * np.log((1 - r) / (1 - rb)) if r < 1 else 0.0)
        out += t / (A + B) * kl
    return float(out)


def task_block(law, P, q, q_int_exact_tables_dec):
    w = law['w']
    m = law['m']
    nll = float((w * -(q * np.log(P)).sum(1)).sum())
    acc = float((w * q[np.arange(m), q_int_exact_tables_dec]).sum())
    brier = float((w * ((P * P).sum(1) - 2 * (P * q).sum(1) + 1.0)).sum())
    return nll, acc, brier


def release(law, gI, gO, full_ref=None):
    """Release with B1 tables: the income vector of state x is the exact posterior mean of qI over its income group gI[x] (likewise
    occupation). Task metrics against the true posteriors, per-SEX occupation Brier with P(Y_occ | x, S), exact Bayes AUC per view
    (identical vectors merged), and plug-in information on cell ids (surrogate) and on released values."""
    m = law['m']
    gI, gO = np.asarray(gI), np.asarray(gO)
    TI, TO = group_tables(law, gI, law['qI_int']), group_tables(law, gO, law['qO_int'])
    vI = [TI[int(gI[x])] for x in range(m)]
    vO = [TO[int(gO[x])] for x in range(m)]
    PI = np.array([[float(f) for f in v] for v in vI])
    PO = np.array([[float(f) for f in v] for v in vO])
    dI = np.array([exact_argmax(v) for v in vI])
    dO = np.array([exact_argmax(v) for v in vO])
    r = {}
    r['inc_nll'], r['inc_acc'], r['inc_brier'] = task_block(law, PI, law['qI'], dI)
    r['occ_nll'], r['occ_acc'], r['occ_brier'] = task_block(law, PO, law['qO'], dO)
    for s, qs, wt in ((0, law['qO0'], law['b']), (1, law['qO1'], law['a'])):
        ws = wt / wt.sum()
        r[f'occ_brier_S{s}'] = float((ws * ((PO * PO).sum(1) - 2 * (PO * qs).sum(1) + 1.0)).sum())
    keyI, keyO = vI, vO
    keyP = [(vI[x], vO[x]) for x in range(m)]
    r['auc_pair'] = float(bayes_auc(law, keyP))
    r['auc_income'] = float(bayes_auc(law, keyI))
    r['auc_occupation'] = float(bayes_auc(law, keyO))
    KO = int(gO.max()) + 1
    r['mi_cell_pair'] = mi_keys(law, [int(gI[x]) * 1000 + int(gO[x]) for x in range(m)])
    r['mi_cell_income'] = mi_keys(law, [int(v) for v in gI])
    r['mi_cell_occupation'] = mi_keys(law, [int(v) for v in gO])
    r['mi_value_pair'] = mi_keys(law, keyP)
    r['values'] = {'pair': len(set(keyP)), 'income': len(set(keyI)), 'occupation': len(set(keyO))}
    r['cells'] = {'pair': len(set(zip(gI.tolist(), gO.tolist()))), 'income': len(set(gI.tolist())),
                  'occupation': len(set(gO.tolist()))}
    if full_ref is not None:
        r['excess'] = excess(r, full_ref)
    return r


def excess(r, full):
    e = {'inc_acc_loss': full['inc_acc'] - r['inc_acc'], 'occ_acc_loss': full['occ_acc'] - r['occ_acc'],
         'inc_nll_excess': r['inc_nll'] - full['inc_nll'], 'occ_nll_excess': r['occ_nll'] - full['occ_nll'],
         'inc_brier_excess': r['inc_brier'] - full['inc_brier'], 'occ_brier_excess': r['occ_brier'] - full['occ_brier'],
         'occ_brier_excess_S0': r['occ_brier_S0'] - full['occ_brier_S0'],
         'occ_brier_excess_S1': r['occ_brier_S1'] - full['occ_brier_S1']}
    return e


LIMIT_KEYS = {'inc_acc_loss': 'acc_loss_per_task', 'occ_acc_loss': 'acc_loss_per_task', 'inc_nll_excess': 'nll_excess_nats_per_task',
              'occ_nll_excess': 'nll_excess_nats_per_task', 'inc_brier_excess': 'summed_brier_excess_per_task',
              'occ_brier_excess': 'summed_brier_excess_per_task', 'occ_brier_excess_S0': 'occ_summed_brier_excess_men_S0',
              'occ_brier_excess_S1': 'occ_summed_brier_excess_women_S1'}


def feasible(ex, limits):
    fails = [k for k, lk in LIMIT_KEYS.items() if ex[k] > limits[lk] + 1e-12]
    return (not fails), fails


def full_release(law):
    m = law['m']
    return release(law, np.arange(m), np.arange(m))


def decision_release(law, full):
    dI = np.array([exact_argmax(tuple(v)) for v in law['qI_int'].tolist()])
    dO = np.array([exact_argmax(tuple(v)) for v in law['qO_int'].tolist()])
    return release(law, dI, dO, full), dI, dO


# ============================================================ exact objective and the routing rule's linear costs (own code)
def mi_cells_float(law, g):
    return mi_keys(law, [int(v) for v in g])


def ref_objective(law, CI, CO, lam, kappa, w_pair, w_loc, k, l):
    """Card FA_PRC_v1 section 1 objective of a hard partition, exactly over the support states: E[CE(qI, cI_k) + CE(qO, cO_l)]
    + lam [w_pair I(S; pair) + w_loc (I(S; k) + I(S; l))] + lam kappa E[(1 - pi)/p0 |qO0 - cO_l|^2 + pi/p1 |qO1 - cO_l|^2]."""
    w = law['w']
    ce = -(law['qI'] * np.log(CI[k])).sum(1) - (law['qO'] * np.log(CO[l])).sum(1)
    J = float((w * ce).sum())
    if lam > 0:
        J += lam * (w_pair * mi_keys(law, [int(a) * 1000 + int(b) for a, b in zip(k, l)])
                    + w_loc * (mi_cells_float(law, k) + mi_cells_float(law, l)))
        if kappa > 0:
            pi = law['pi']
            g = (1 - pi) / law['p0'] * ((law['qO0'] - CO[l]) ** 2).sum(1) + pi / law['p1'] * ((law['qO1'] - CO[l]) ** 2).sum(1)
            J += lam * kappa * float((w * g).sum())
    return J


def cost_lines(law, nb):
    """Per state x and joint cell c = k * KO + l, the routing rule's cost is A_x(c) + t B_x(c) (card section 1), with the book's
    constants (codewords, shrunk tables rI, rO, rP, rbar, p0, p1). The rate clip at [0.01, 0.99] mirrors the implementation; it
    never binds on these laws (checked)."""
    CI, CO = nb['CI'], nb['CO']
    KI, KO = len(CI), len(CO)
    m = law['m']
    AI = -(law['qI'] @ np.log(CI).T)
    AO = -(law['qO'] @ np.log(CO).T)
    A = AI[:, :, None] + AO[:, None, :]
    pi = law['pi']
    rbar = nb['rbar']
    clip_bind = False

    def D(r):
        nonlocal clip_bind
        rc = np.clip(r, 0.01, 0.99)
        clip_bind |= bool(np.any(rc != r))
        sh = (m,) + (1,) * np.ndim(r)
        p = pi.reshape(sh)
        return p * np.log(rc / rbar)[None] + (1 - p) * np.log((1 - rc) / (1 - rbar))[None]

    DI, DO, DP = D(nb['rI']), D(nb['rO']), D(nb['rP'])
    G = ((1 - pi) / nb['p0'])[:, None] * ((law['qO0'][:, None, :] - CO[None]) ** 2).sum(2) \
        + (pi / nb['p1'])[:, None] * ((law['qO1'][:, None, :] - CO[None]) ** 2).sum(2)
    B = nb['w_loc'] * (DI[:, :, None] + DO[:, None, :]) + nb['w_pair'] * DP + nb['kappa'] * G[:, None, :]
    return A.reshape(m, KI * KO), B.reshape(m, KI * KO), clip_bind


def argmin_at(A, B, t, KO):
    f = (A + t * B).argmin(1)
    return f // KO, f % KO


def tilt_partitions(A, B, T, KO):
    """All partitions of argmin_c A_x(c) + t B_x(c) for t in (0, T]: candidate breakpoints are all pairwise crossings of the
    per-state lines inside (0, T); the partition is constant on each open interval between consecutive candidates. Returns a list of
    (t_lo, t_hi, t_mid, k, l) per interval (consecutive equal partitions kept separate) and the number of intervals too short to
    evaluate safely (< 1e-12 relative), which are skipped and reported."""
    m, C = A.shape
    pts = []
    for x in range(m):
        dA = A[x][None, :] - A[x][:, None]
        dB = B[x][:, None] - B[x][None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = dA / dB
        ok = (dB != 0) & np.isfinite(t) & (t > 0) & (t < T)
        pts.extend(t[ok].tolist())
    edges = np.unique(np.concatenate([[0.0], np.array(pts, dtype=float), [T]]))
    out, short = [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 1e-12 * max(1.0, T):
            short += 1
            continue
        mid = 0.5 * (lo + hi)
        k, l = argmin_at(A, B, mid, KO)
        out.append((float(lo), float(hi), float(mid), k, l))
    return out, short, len(edges) - 2


# ============================================================ instrumented replay of the code under test (calls only its functions)
def replay_path(P, E, law, twin, k0, l0, rp, lams, narrow=None):
    """Mirror of nrm.fa.prc._chain / lloyd that calls the code's own update_book, route and objective in the same order and
    records every proposal step. The caller checks that its states equal protected_path bitwise."""
    Eu, book0 = P._prepare(E, twin, k0, l0, rp)
    targets = sorted({float(x) for x in lams})
    start = (dict(book0), k0, l0)
    book, k, l = dict(book0), k0, l0
    out, trace = {}, []
    for lam_t in sorted({g for g in P.LAMBDA_GRID if g < max(targets)} | set(targets)):
        book['lam'] = start[0]['lam'] = float(lam_t)
        reset = bool(P.objective(Eu, start[0], k0, l0) < P.objective(Eu, book, k, l))
        if reset:
            book, k, l = dict(start[0]), k0, l0
        J = P.objective(Eu, book, k, l)
        lam = float(book['lam'])
        tilts = [lam / 2 ** j for j in range(P.TILT_STEPS)] if lam > 0 else [0.0]
        for step in range(int(rp['path_steps'])):
            nb = P.update_book(Eu, dict(book), k, l)
            props = []
            for t in tilts:
                nb['tilt'] = float(t)
                k2, l2 = P.route(Eu, nb)
                props.append((float(t), k2, l2, float(P.objective(Eu, nb, k2, l2))))
            acc = next((i for i, p in enumerate(props) if p[3] < J - 1e-12), None)
            rec = {'lam': lam, 'step': step, 'reset_to_twin_before_lam': reset if step == 0 else None, 'J_current': J,
                   'proposals': props, 'accepted': acc}
            if narrow is not None:
                rec['narrow'] = narrow(law, Eu, book, nb, k, l, J, props, acc, lam)
            trace.append(rec)
            if acc is None:
                break
            t_acc = props[acc][0]
            nb['tilt'] = t_acc
            book, k, l, J = dict(nb), props[acc][1], props[acc][2], props[acc][3]
        if lam_t in targets:
            out[lam_t] = (dict(book), k.copy(), l.copy())
    return out, trace


def same_state(a, b):
    (b1, k1, l1), (b2, k2, l2) = a, b
    return bool(np.array_equal(k1, k2) and np.array_equal(l1, l2) and b1['tilt'] == b2['tilt'] and np.array_equal(b1['CO'], b2['CO'])
                and np.array_equal(b1['CI'], b2['CI']) and np.array_equal(b1.get('rP'), b2.get('rP')))


# ============================================================ STEP 3: the narrow reference at one proposal step
def narrow_step(P, full, limits, tol):
    def fn(law, E, book, nb, k, l, J, props, acc, lam):
        KO = nb['CO'].shape[0]
        A, Bv, clip_bind = cost_lines(law, nb)
        res = {'clip_binds': clip_bind}
        # current state and the seven proposals: my objective against the code's
        ks, ls = to_state(law, k), to_state(law, l)
        Jm_cur = ref_objective(law, book['CI'], book['CO'], lam, nb['kappa'], nb['w_pair'], nb['w_loc'], ks, ls)
        Jnb_cur_code = float(P.objective(E, nb, k, l))
        Jnb_cur_mine = ref_objective(law, nb['CI'], nb['CO'], lam, nb['kappa'], nb['w_pair'], nb['w_loc'], ks, ls)
        res['current'] = {'J_code_oldbook': J, 'J_mine_oldbook': Jm_cur, 'J_code_under_nb': Jnb_cur_code, 'J_mine_under_nb': Jnb_cur_mine}
        res['refresh_check'] = refresh_check(law, book, nb, ks, ls)
        maxdiff = max(abs(J - Jm_cur), abs(Jnb_cur_code - Jnb_cur_mine))
        route_mismatch = 0
        prop_parts = []
        for (t, k2, l2, jc) in props:
            k2s, l2s = to_state(law, k2), to_state(law, l2)
            km, lm = argmin_at(A, Bv, t, KO)
            route_mismatch += int(not (np.array_equal(km, k2s) and np.array_equal(lm, l2s)))
            jm = ref_objective(law, nb['CI'], nb['CO'], lam, nb['kappa'], nb['w_pair'], nb['w_loc'], k2s, l2s)
            maxdiff = max(maxdiff, abs(jc - jm))
            prop_parts.append({'t': t, 'J_code': jc, 'J_mine': jm, 'k': k2s.tolist(), 'l': l2s.tolist()})
        res['proposals'] = prop_parts
        for tag, T in (('card', lam), ('wide', 4 * lam)):
            ivs, short, ncand = tilt_partitions(A, Bv, T, KO)
            parts = {}
            for (lo, hi, mid, km, lm) in ivs:
                nbt = dict(nb)
                nbt['tilt'] = mid
                kc, lc = P.route(E, nbt)
                kcs, lcs = to_state(law, kc), to_state(law, lc)
                if not (np.array_equal(kcs, km) and np.array_equal(lcs, lm)):
                    route_mismatch += 1
                key = (tuple(kcs.tolist()), tuple(lcs.tolist()))
                if key not in parts:
                    parts[key] = {'t_first': mid, 'intervals': 0, 'iv': []}
                parts[key]['intervals'] += 1
                parts[key]['iv'].append([lo, hi])
            if tag == 'card':                                     # the seven tilts are in (0, lambda]; their partitions belong
                for pp in prop_parts:
                    key = (tuple(pp['k']), tuple(pp['l']))
                    parts.setdefault(key, {'t_first': pp['t'], 'intervals': 0, 'iv': []})
            best = None
            for key, info in parts.items():
                kk, ll = np.array(key[0]), np.array(key[1])
                jc = float(P.objective(E, nb, to_rows(law, kk), to_rows(law, ll)))
                jm = ref_objective(law, nb['CI'], nb['CO'], lam, nb['kappa'], nb['w_pair'], nb['w_loc'], kk, ll)
                maxdiff = max(maxdiff, abs(jc - jm))
                info.update({'J_code': jc, 'J_mine': jm})
                if best is None or jm < best[1]['J_mine'] - tol:
                    best = (key, info)
            improving = [info for info in parts.values() if info['J_mine'] < J - tol]
            res[tag] = {'n_candidates': ncand, 'n_intervals_evaluated': len(ivs), 'n_short_intervals_skipped': short,
                        'n_partitions': len(parts), 'best_J_mine': best[1]['J_mine'], 'best_J_code': best[1]['J_code'],
                        'best_t': best[1]['t_first'], 'best_k': list(best[0][0]), 'best_l': list(best[0][1]),
                        'best_intervals': best[1]['iv'], 'n_improving_partitions': len(improving),
                        'improving_tilt_intervals': sorted(iv for info in improving for iv in info['iv']),
                        'current_partition_in_class': (tuple(ks.tolist()), tuple(ls.tolist())) in parts}
        # coordinator note 02:47Z: the current book's own tilt line t in [0, lambda] (contains the current partition) belongs to the
        # revised solver's declared class, not to the original's; reported separately ('cur_card'), with the closed union
        # 'sv1_union' = nb lines on [0, lambda] (adds t = 0) together with the current book's lines on [0, lambda].
        extra = {}
        for tag2, bk in (('nb_t0', nb), ('cur_card', book)):
            pts = []
            AI0 = -(law['qI'] @ np.log(bk['CI']).T)
            AO0 = -(law['qO'] @ np.log(bk['CO']).T)
            pts.append((0.0, AI0.argmin(1), AO0.argmin(1)))
            if tag2 == 'cur_card' and 'rI' in bk:
                A2, B2, _ = cost_lines(law, bk)
                pts += [(mid, km, lm) for (lo, hi, mid, km, lm) in tilt_partitions(A2, B2, lam, KO)[0]]
            parts2 = {}
            for (tt, km, lm) in pts:
                bt = dict(bk)
                bt['tilt'] = float(tt)
                kc, lc = P.route(E, bt)
                kcs, lcs = to_state(law, kc), to_state(law, lc)
                if not (np.array_equal(kcs, km) and np.array_equal(lcs, lm)):
                    route_mismatch += 1
                key = (tuple(kcs.tolist()), tuple(lcs.tolist()))
                parts2.setdefault(key, {'t_first': float(tt)})
            bst = None
            for key, info in parts2.items():
                kk, ll = np.array(key[0]), np.array(key[1])
                jc = float(P.objective(E, bk, to_rows(law, kk), to_rows(law, ll)))
                jm = ref_objective(law, bk['CI'], bk['CO'], lam, nb['kappa'], nb['w_pair'], nb['w_loc'], kk, ll)
                maxdiff = max(maxdiff, abs(jc - jm))
                info.update({'J_code': jc, 'J_mine': jm})
                if bst is None or jm < bst[1]['J_mine'] - tol:
                    bst = (key, info)
            extra[tag2] = {'n_partitions': len(parts2), 'best_J_mine': bst[1]['J_mine'], 'best_J_code': bst[1]['J_code'],
                           'best_t': bst[1]['t_first'], 'best_k': list(bst[0][0]), 'best_l': list(bst[0][1]),
                           'has_tables': 'rI' in bk,
                           'current_partition_in_class': (tuple(ks.tolist()), tuple(ls.tolist())) in parts2}
        res['cur_card'] = extra['cur_card']
        res['nb_t0'] = extra['nb_t0']
        cands = [('card', res['card']), ('nb_t0', extra['nb_t0']), ('cur_card', extra['cur_card'])]
        src, bu = min(cands, key=lambda z: z[1]['best_J_mine'])
        res['sv1_union'] = {'best_J_mine': bu['best_J_mine'], 'from': src, 'best_k': bu['best_k'], 'best_l': bu['best_l']}
        res['max_abs_objective_diff_code_vs_mine'] = maxdiff
        res['route_mismatches'] = route_mismatch
        # first-improvement choice against the best attainable proposal in the card class
        c = res['card']
        if acc is not None:
            ja = prop_parts[acc]['J_mine']
            res['gap_first_minus_best_card'] = ja - c['best_J_mine']
            res['gap_first_minus_best_wide'] = ja - res['wide']['best_J_mine']
            res['gap_first_minus_best_sv1_union'] = ja - res['sv1_union']['best_J_mine']
            chosen = (np.array(prop_parts[acc]['k']), np.array(prop_parts[acc]['l']))
        else:
            res['missed_improvement_card'] = bool(c['best_J_mine'] < J - tol)
            res['missed_improvement_wide'] = bool(res['wide']['best_J_mine'] < J - tol)
            res['missed_improvement_sv1_union'] = bool(res['sv1_union']['best_J_mine'] < J - tol)
            res['gap_current_minus_best_card'] = J - c['best_J_mine']
            res['gap_current_minus_best_wide'] = J - res['wide']['best_J_mine']
            chosen = (ks, ls)
        relc = release(law, chosen[0], chosen[1], full)
        relb = release(law, np.array(c['best_k']), np.array(c['best_l']), full)
        res['chosen_release'] = slim(relc, limits)
        res['best_card_release'] = slim(relb, limits)
        res['best_minus_chosen'] = {'auc_pair': relb['auc_pair'] - relc['auc_pair'],
                                    'auc_income': relb['auc_income'] - relc['auc_income'],
                                    'auc_occupation': relb['auc_occupation'] - relc['auc_occupation'],
                                    **{k: relb['excess'][k] - relc['excess'][k] for k in relc['excess']}}
        return res
    return fn


def refresh_check(law, book, nb, ks, ls):
    """Independent recomputation of the refreshed book (update_book / set_tables) from card section 1: codewords = cell means of
    the teacher posteriors (unoccupied cells keep the old codeword), clipped at 1e-6 and renormalised; tables = cell pi means
    shrunk with m_shrink pseudo-counts toward rbar, the pair table toward the additive-logit fit of the shrunk local rates."""
    m_sh, rbar = float(nb['m_shrink']), float(book['rbar'])
    d = 0.0
    for key, cells, q_int in (('CI', ks, law['qI_int']), ('CO', ls, law['qO_int'])):
        C = np.array(book[key], dtype=float, copy=True)
        for c, v in group_tables(law, cells, q_int).items():
            C[c] = [float(f) for f in v]
        C = np.clip(C, 1e-6, 1.0)
        C = C / C.sum(1, keepdims=True)
        d = max(d, float(np.abs(C - nb[key]).max()))
    KI, KO = book['CI'].shape[0], book['CO'].shape[0]
    n = law['n'].astype(float)
    spi = n * law['pi']
    rI = (np.bincount(ks, weights=spi, minlength=KI) + m_sh * rbar) / (np.bincount(ks, weights=n, minlength=KI) + m_sh)
    rO = (np.bincount(ls, weights=spi, minlength=KO) + m_sh * rbar) / (np.bincount(ls, weights=n, minlength=KO) + m_sh)
    j = ks * KO + ls
    nP = np.bincount(j, weights=n, minlength=KI * KO).reshape(KI, KO)
    sP = np.bincount(j, weights=spi, minlength=KI * KO).reshape(KI, KO)
    lg = lambda r: np.log(r / (1 - r))
    add = 1.0 / (1.0 + np.exp(-(lg(rI)[:, None] + lg(rO)[None, :] - lg(rbar))))
    rP = (sP + m_sh * add) / (nP + m_sh)
    td = max(float(np.abs(rI - nb['rI']).max()), float(np.abs(rO - nb['rO']).max()), float(np.abs(rP - nb['rP']).max()))
    return {'codeword_max_abs_diff': d, 'table_max_abs_diff': td, 'rbar_minus_exact_p1': rbar - law['p1'],
            'p1_book_minus_exact': float(nb['p1']) - law['p1']}


def slim(r, limits):
    ok, fails = feasible(r['excess'], limits)
    return {'auc_pair': r['auc_pair'], 'auc_income': r['auc_income'], 'auc_occupation': r['auc_occupation'],
            'mi_cell_pair': r['mi_cell_pair'], 'mi_cell_income': r['mi_cell_income'], 'mi_cell_occupation': r['mi_cell_occupation'],
            'mi_value_pair': r['mi_value_pair'], 'excess': r['excess'], 'feasible': ok, 'fails': fails, 'values': r['values'],
            'cells': r['cells']}


# ============================================================ STEPs 2-3 driver
def run_narrow(out_path):
    from nrm.fa import prc as P
    t0 = time.process_time()
    laws, common = load_laws()
    limits = common['limits_vs_full_posterior_release']
    tol = common['tolerances']['objective_tie_abs']
    KI, KO = common['KI'], common['KO']
    lams = list(common['lambda_grid'])
    if tuple(lams) != tuple(P.LAMBDA_GRID):
        raise AssertionError('declared grid differs from the code grid')
    result = {'schema': 'FSA_EXACT_NARROW_v1', 'laws_commit': 'c27a91477', 'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'laws': {}}
    for name, law in laws.items():
        tl = time.process_time()
        E = code_E(law)
        full = full_release(law)
        L = {'posteriors': {law['ids'][x]: {'n': int(law['n'][x]), 'pi': float(law['pi'][x]), 'qI': law['qI'][x].tolist(),
                                            'qO': law['qO'][x].tolist(), 'qO0': law['qO0'][x].tolist(), 'qO1': law['qO1'][x].tolist()}
                            for x in range(law['m'])},
             'p1': law['p1'], 'full_release': {k: full[k] for k in full if k != 'excess'}}
        dec, dI, dO = decision_release(law, full)
        L['decision_only'] = {**slim(dec, limits), 'inc_cell': dI.tolist(), 'occ_cell': dO.tolist()}
        for tag in ('witness', 'comparator'):
            if law[tag]:
                wr = release(law, np.array(law[tag]['inc_cell']), np.array(law[tag]['occ_cell']), full)
                L[tag] = {**slim(wr, limits), 'inc_cell': law[tag]['inc_cell'], 'occ_cell': law[tag]['occ_cell']}
        L['seeds'] = {}
        for seed in common['twin_seeds']:
            S = {}
            rp0 = P.route_params('PRC_TWIN')
            tb, k0, l0 = P.twin_book(E, KI, KO, seed, rp0)
            k0s, l0s = to_state(law, k0), to_state(law, l0)
            tw = release(law, k0s, l0s, full)
            S['twin'] = {**slim(tw, limits), 'inc_cell': k0s.tolist(), 'occ_cell': l0s.tolist(),
                         'J_task_code': float(P.objective(E, tb, k0, l0)),
                         'J_task_mine': ref_objective(law, tb['CI'], tb['CO'], 0.0, 1.0, 1.0, 1.0, k0s, l0s),
                         'max_abs_codeword_minus_B1': codeword_gap(law, tb, k0s, l0s)}
            rp = P.route_params('PRC_S2')
            ref_states = P.protected_path(E, tb, k0, l0, lams, rp)
            rep_states, trace = replay_path(P, E, law, tb, k0, l0, rp, lams, narrow=narrow_step(P, full, limits, tol))
            S['replay_equals_protected_path'] = all(same_state(ref_states[g], rep_states[g]) for g in lams)
            path = {}
            sb = P._protected_start(E, tb, k0, l0, rp)
            for g in lams:
                bk, kk, ll = ref_states[g]
                ks, ls = to_state(law, kk), to_state(law, ll)
                rr = release(law, ks, ls, full)
                path[str(g)] = {**slim(rr, limits), 'inc_cell': ks.tolist(), 'occ_cell': ls.tolist(), 'tilt': float(bk['tilt']),
                                'J_code': float(P.objective(E, bk, kk, ll)),
                                'J_mine': ref_objective(law, bk['CI'], bk['CO'], g, bk['kappa'], bk['w_pair'], bk['w_loc'], ks, ls),
                                'max_abs_codeword_minus_B1': codeword_gap(law, bk, ks, ls),
                                'equals_twin_partition': canon(ks, ls) == canon(k0s, l0s)}
                sb['lam'] = float(g)                                        # card: never worse than the twin at the same lambda
                jt = float(P.objective(E, sb, k0, l0))
                path[str(g)].update({'J_twin_same_lam_code': jt,
                                     'J_twin_same_lam_mine': ref_objective(law, tb['CI'], tb['CO'], g, rp['kappa'], rp['w_pair'],
                                                                           rp['w_loc'], k0s, l0s),
                                     'not_worse_than_twin': bool(path[str(g)]['J_code'] <= jt + 1e-12)})
            S['path'] = path
            S['trace'] = [trace_row(r) for r in trace]
            # HQ (secondary): final states only
            try:
                hq = P.protected_path(E, tb, k0, l0, lams, P.route_params('HQ_S2'))
                S['hq_path'] = {}
                for g in lams:
                    bk, kk, ll = hq[g]
                    ks, ls = to_state(law, kk), to_state(law, ll)
                    rr = release(law, ks, ls, full)
                    S['hq_path'][str(g)] = {**slim(rr, limits), 'inc_cell': ks.tolist(), 'occ_cell': ls.tolist(),
                                            'J_code_objective': float(P.objective(E, bk, kk, ll))}
            except Exception as exc:                                      # reported, never hidden
                S['hq_path'] = {'error': repr(exc)}
            L['seeds'][str(seed)] = S
        L['cpu_s'] = round(time.process_time() - tl, 2)
        result['laws'][name] = L
        print(json.dumps({'law': name, 'cpu_s': L['cpu_s']}), flush=True)
    result['cpu_s'] = round(time.process_time() - t0, 2)
    result['utc_end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    Path(out_path).write_text(json.dumps(result, indent=1, default=jsonable))
    print(json.dumps({'done': out_path, 'cpu_s': result['cpu_s']}))


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Fraction):
        return float(o)
    raise TypeError(type(o))


def codeword_gap(law, book, ks, ls):
    """Largest absolute difference between a search-time codeword and the B1 table (exact mean) of the same occupied cell."""
    g = 0.0
    for C, cells, qi in ((book['CI'], ks, law['qI_int']), (book['CO'], ls, law['qO_int'])):
        T = group_tables(law, cells, qi)
        for c, v in T.items():
            g = max(g, float(np.max(np.abs(C[c] - np.array([float(f) for f in v])))))
    return g


def trace_row(r):
    out = {'lam': r['lam'], 'step': r['step'], 'J_current': r['J_current'], 'accepted': r['accepted'],
           'reset_to_twin_before_lam': r['reset_to_twin_before_lam'],
           'proposal_J_code': [p[3] for p in r['proposals']]}
    if 'narrow' in r:
        out['narrow'] = r['narrow']
    return out


# ============================================================ STEP 4: enumeration (relaxation)
def view_arrays(law, assign, K, q_int, q):
    """Vectorised per-assignment statistics of one task view (assign: (A, m) labels in 0..K-1)."""
    n, a1, b0, w = law['n'], law['a'], law['b'], law['w']
    A, m = assign.shape
    oh = (assign[:, :, None] == np.arange(K)[None, None, :])
    cnt = (oh * n[None, :, None]).sum(1)                                    # (A, K)
    num = np.einsum('amk,mc->akc', oh * n[None, :, None], q_int)            # (A, K, C) exact integers
    with np.errstate(divide='ignore', invalid='ignore'):
        T = num / (cnt[..., None] * float(QSCALE))
    Tx = np.take_along_axis(T, assign[:, :, None], 1)                       # (A, m, C) per-state table
    nll = (w[None] * -(q[None] * np.log(Tx)).sum(2)).sum(1)
    brier_exc = (w[None] * ((Tx - q[None]) ** 2).sum(2)).sum(1)
    numx = np.take_along_axis(num, assign[:, :, None], 1)
    dec = numx.argmax(2)                                                    # exact (integer numerators, first index)
    acc = (w[None] * q[np.arange(m)[None, :], dec]).sum(1)
    sa = (oh * a1[None, :, None]).sum(1)
    sb = (oh * b0[None, :, None]).sum(1)
    # merge identical tables: rep[a, k] = smallest k' with an identical (non-empty) table
    rep = np.tile(np.arange(K), (A, 1))
    for k in range(K):
        for kp in range(k):
            same = (cnt[:, k] > 0) & (cnt[:, kp] > 0) & np.all(num[:, k, :] * cnt[:, kp, None] == num[:, kp, :] * cnt[:, k, None], 1)
            upd = same & (rep[:, k] == k)
            rep[upd, k] = kp
    return {'cnt': cnt, 'Tx': Tx, 'nll': nll, 'brier_exc': brier_exc, 'acc': acc, 'sa': sa, 'sb': sb, 'rep': rep}


def mi_masses(sa, sb):
    sa, sb = sa.astype(float), sb.astype(float)
    tot = sa + sb
    TT = tot.sum(-1, keepdims=True)
    rb = sa.sum(-1, keepdims=True) / TT
    with np.errstate(divide='ignore', invalid='ignore'):
        r = sa / tot
        kl = np.where(r > 0, r * np.log(r / rb), 0.0) + np.where(r < 1, (1 - r) * np.log((1 - r) / (1 - rb)), 0.0)
    return np.where(tot > 0, tot / TT * kl, 0.0).sum(-1)


def auc_masses(sa, sb):
    """Exact Bayes AUC from integer masses per merged value (last axis), vectorised; ties 1/2."""
    lhs = sa[..., :, None] * sb[..., None, :]
    rhs = sa[..., None, :] * sb[..., :, None]
    num2 = (np.where(lhs > rhs, 2 * lhs, 0) + np.where(lhs == rhs, lhs, 0)).sum((-1, -2))
    A, B = sa.sum(-1), sb.sum(-1)
    return num2 / (2.0 * A * B)


def merged_masses(sa, sb, rep, K):
    A = sa.shape[0]
    ma, mb = np.zeros_like(sa), np.zeros_like(sb)
    for k in range(K):
        np.add.at(ma, (np.arange(A), rep[:, k]), sa[:, k])
        np.add.at(mb, (np.arange(A), rep[:, k]), sb[:, k])
    return ma, mb


def subset_codeword_opt(law, lam_kappa):
    """For every subset of states (bitmask) the minimum over the simplex of sum_x w_x [CE(qO_x, c) + lam kappa ((1 - pi)/p0
    |qO0 - c|^2 + pi/p1 |qO1 - c|^2)], by the KKT quadratic per class and bisection on the multiplier (strictly convex)."""
    m = law['m']
    w, pi = law['w'], law['pi']
    al, be = (1 - pi) / law['p0'], pi / law['p1']
    out = np.zeros(2 ** m)
    cw = {}
    for mask in range(1, 2 ** m):
        idx = [x for x in range(m) if mask >> x & 1]
        W = w[idx].sum()
        qbar = (w[idx, None] * law['qO'][idx]).sum(0) / W
        if lam_kappa == 0:
            c = qbar
        else:
            gam = (w[idx] * (al[idx] + be[idx])).sum()
            h = (w[idx, None] * (al[idx, None] * law['qO0'][idx] + be[idx, None] * law['qO1'][idx])).sum(0)
            a2 = 2 * lam_kappa * gam

            def cvec(nu):
                bq = nu - 2 * lam_kappa * h
                return (-bq + np.sqrt(bq * bq + 4 * a2 * W * qbar)) / (2 * a2)
            lo, hi = -1e3, 1e3
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if cvec(mid).sum() > 1:
                    lo = mid
                else:
                    hi = mid
            c = cvec(0.5 * (lo + hi))
            c = c / c.sum()
        val = (w[idx] * (-(law['qO'][idx] * np.log(c)).sum(1)
                         + lam_kappa * (al[idx] * ((law['qO0'][idx] - c) ** 2).sum(1) + be[idx] * ((law['qO1'][idx] - c) ** 2).sum(1)))).sum()
        out[mask] = val
        cw[mask] = c
    return out, cw


def run_enum(out_path, narrow_path=None):
    from nrm.fa import prc as P
    t0 = time.process_time()
    laws, common = load_laws()
    limits = common['limits_vs_full_posterior_release']
    KI, KO, kappa = common['KI'], common['KO'], common['kappa']
    lams = list(common['lambda_grid'])
    narrow = json.loads(Path(narrow_path).read_text()) if narrow_path and Path(narrow_path).is_file() else None
    result = {'schema': 'FSA_EXACT_ENUM_v1', 'laws_commit': 'c27a91477',
              'scope': 'every labelled assignment of the m states to KI x KO cells (KI^m x KO^m); codewords = cell means (declared '
                       'update rule); tables = B1 (cell means). RELAXATION: realizability by the runtime rule is certified only where '
                       'stated (the partition is route(E, its own rule book with tilt t) for some t in (0, 4 lambda], checked with '
                       'the code route) or the solver reached it.', 'laws': {}}
    for name, law in laws.items():
        tl = time.process_time()
        m = law['m']
        full = full_release(law)
        AIa = np.array(list(itertools.product(range(KI), repeat=m)), dtype=np.int64)
        AOa = np.array(list(itertools.product(range(KO), repeat=m)), dtype=np.int64)
        VI = view_arrays(law, AIa, KI, law['qI_int'], law['qI'])
        VO = view_arrays(law, AOa, KO, law['qO_int'], law['qO'])
        TxO = VO['Tx']
        pi = law['pi']
        G = (law['w'][None] * ((1 - pi)[None] / law['p0'] * ((law['qO0'][None] - TxO) ** 2).sum(2)
                               + pi[None] / law['p1'] * ((law['qO1'][None] - TxO) ** 2).sum(2))).sum(1)
        bS = {}
        for s, qs, wt in ((0, law['qO0'], law['b']), (1, law['qO1'], law['a'])):
            ws = wt / wt.sum()
            bS[s] = (ws[None] * (((TxO - qs[None]) ** 2).sum(2) - ((law['qO'] - qs) ** 2).sum(1)[None])).sum(1)
        exI = {'inc_nll_excess': VI['nll'] - full['inc_nll'], 'inc_acc_loss': full['inc_acc'] - VI['acc'],
               'inc_brier_excess': VI['brier_exc']}
        exO = {'occ_nll_excess': VO['nll'] - full['occ_nll'], 'occ_acc_loss': full['occ_acc'] - VO['acc'],
               'occ_brier_excess': VO['brier_exc'], 'occ_brier_excess_S0': bS[0], 'occ_brier_excess_S1': bS[1]}
        feasI = np.ones(len(AIa), bool)
        for kk, v in exI.items():
            feasI &= v <= limits[LIMIT_KEYS[kk]] + 1e-12
        feasO = np.ones(len(AOa), bool)
        for kk, v in exO.items():
            feasO &= v <= limits[LIMIT_KEYS[kk]] + 1e-12
        miI, miO = mi_masses(VI['sa'], VI['sb']), mi_masses(VO['sa'], VO['sb'])
        maI, mbI = merged_masses(VI['sa'], VI['sb'], VI['rep'], KI)
        maO, mbO = merged_masses(VO['sa'], VO['sb'], VO['rep'], KO)
        aucI, aucO = auc_masses(maI, mbI), auc_masses(maO, mbO)
        # codeword-optimised occupation part per assignment and lambda (lower-bound relaxation)
        occ_masks = np.zeros((len(AOa), KO), dtype=np.int64)
        for x in range(m):
            occ_masks[np.arange(len(AOa)), AOa[:, x]] |= (1 << x)
        sub_opt = {g: subset_codeword_opt(law, g * kappa)[0] for g in lams}
        occ_opt = {g: sub_opt[g][occ_masks].sum(1) for g in lams}
        # pair loop over income assignments
        hq_ok_I = hq_ok_O = None
        if law['collision_groups']:
            idx = {sid: x for x, sid in enumerate(law['ids'])}
            grp = [[idx[s] for s in g] for g in law['collision_groups']]
            hq_ok_I = np.all([np.all(AIa[:, g] == AIa[:, [g[0]]], 1) for g in grp], 0)
            hq_ok_O = np.all([np.all(AOa[:, g] == AOa[:, [g[0]]], 1) for g in grp], 0)
        best = {f'obj_{g}': (np.inf, None) for g in lams}
        best.update({f'bound_{g}': (np.inf, None) for g in lams})
        best.update({'auc_feasible': (np.inf, None), 'mi_feasible': (np.inf, None), 'task_opt': (np.inf, None),
                     'auc_feasible_hq': (np.inf, None), 'mi_feasible_hq': (np.inf, None)})
        n_feas = 0
        NA = len(AOa)
        for i in range(len(AIa)):
            jid = AIa[i][None, :] * KO + AOa                                    # unmerged joint cells
            jm = VI['rep'][i][AIa[i]][None, :] * KO + VO['rep'][np.arange(NA)[:, None], AOa]   # merged identities
            sa = np.zeros((NA, KI * KO), dtype=np.int64)
            sb = np.zeros((NA, KI * KO), dtype=np.int64)
            sam = np.zeros((NA, KI * KO), dtype=np.int64)
            sbm = np.zeros((NA, KI * KO), dtype=np.int64)
            for x in range(m):
                sa[np.arange(NA), jid[:, x]] += law['a'][x]
                sb[np.arange(NA), jid[:, x]] += law['b'][x]
                sam[np.arange(NA), jm[:, x]] += law['a'][x]
                sbm[np.arange(NA), jm[:, x]] += law['b'][x]
            miP = mi_masses(sa, sb)
            aucP = auc_masses(sam, sbm)
            task = VI['nll'][i] + VO['nll']
            j0 = int(np.argmin(task))
            if task[j0] < best['task_opt'][0] - 1e-12:
                best['task_opt'] = (float(task[j0]), (i, j0))
            for g in lams:
                J = VI['nll'][i] + VO['nll'] + g * (miP + miI[i] + miO) + g * kappa * G
                j = int(np.argmin(J))
                if J[j] < best[f'obj_{g}'][0] - 1e-12:
                    best[f'obj_{g}'] = (float(J[j]), (i, j))
                Jb = VI['nll'][i] + occ_opt[g] + g * (miP + miI[i] + miO)
                j = int(np.argmin(Jb))
                if Jb[j] < best[f'bound_{g}'][0] - 1e-12:
                    best[f'bound_{g}'] = (float(Jb[j]), (i, j))
            if feasI[i]:
                f = feasO
                n_feas += int(f.sum())
                for tag, val in (('auc_feasible', aucP), ('mi_feasible', miP)):
                    vv = np.where(f, val, np.inf)
                    j = int(np.argmin(vv))
                    if vv[j] < best[tag][0] - 1e-12:
                        best[tag] = (float(vv[j]), (i, j))
                if hq_ok_I is not None and hq_ok_I[i]:
                    f2 = f & hq_ok_O
                    for tag, val in (('auc_feasible_hq', aucP), ('mi_feasible_hq', miP)):
                        vv = np.where(f2, val, np.inf)
                        j = int(np.argmin(vv))
                        if vv[j] < best[tag][0] - 1e-12:
                            best[tag] = (float(vv[j]), (i, j))
        E = code_E(law)
        Lr = {'n_assignments': int(len(AIa) * len(AOa)), 'n_feasible_assignments': n_feas, 'optima': {}}
        for tag, (val, ij) in best.items():
            if ij is None:
                Lr['optima'][tag] = None
                continue
            i, j = ij
            ks, ls = AIa[i], AOa[j]
            rr = release(law, ks, ls, full)
            ent = {'value': val, 'inc_cell': ks.tolist(), 'occ_cell': ls.tolist(), **slim(rr, limits)}
            lam_for = float(tag.split('_')[1]) if tag.startswith(('obj_', 'bound_')) else None
            ent['realizable'] = realizability(P, law, E, ks, ls, lams if lam_for is None else [lam_for])
            if narrow:
                ent['solver_reached'] = solver_reached(narrow, name, ks, ls)
            Lr['optima'][tag] = ent
        Lr['cpu_s'] = round(time.process_time() - tl, 2)
        result['laws'][name] = Lr
        print(json.dumps({'law': name, 'cpu_s': Lr['cpu_s'], 'n': Lr['n_assignments']}), flush=True)
    result['cpu_s'] = round(time.process_time() - t0, 2)
    Path(out_path).write_text(json.dumps(result, indent=1, default=jsonable))
    print(json.dumps({'done': out_path, 'cpu_s': result['cpu_s']}))


def realizability(P, law, E, ks, ls, lams):
    """Certificate that a partition is produced by the declared runtime rule: with its own rule book (codewords = cell means, tables
    = shrunk cell pi means, from the code's update_book) and some tilt t in [0, 4 lambda], route reproduces it exactly."""
    KI, KO = 2, 3
    out = {}
    for g in lams:
        book = {'CI': np.full((KI, 2), 0.5), 'CO': np.full((KO, 5), 0.2), 'lam': float(g), 'w_pair': 1.0, 'w_loc': 1.0, 'kappa': 1.0,
                'm_shrink': 20.0, 'restrict': None, 'rbar': float(E['pi'].mean()), 'p0': float((1 - E['pi']).mean()),
                'p1': float(E['pi'].mean()), 'tilt': 0.0}
        nb = P.update_book(E, book, to_rows(law, ks), to_rows(law, ls))
        A, Bv, _ = cost_lines(law, nb)
        hit = None
        k0, l0 = argmin_at(A, Bv, 0.0, KO)
        if np.array_equal(k0, ks) and np.array_equal(l0, ls):
            hit = 0.0
        if hit is None:
            for (lo, hi, mid, km, lm) in tilt_partitions(A, Bv, 4 * g, KO)[0]:
                if np.array_equal(km, ks) and np.array_equal(lm, ls):
                    hit = mid
                    break
        code_ok = None
        if hit is not None:
            nbt = dict(nb)
            nbt['tilt'] = hit
            kc, lc = P.route(E, nbt) if hit > 0 else (P._xent(E['qI'], nb['CI']).argmin(1), P._xent(E['qO'], nb['CO']).argmin(1))
            code_ok = bool(np.array_equal(to_state(law, kc), ks) and np.array_equal(to_state(law, lc), ls))
        out[str(g)] = {'self_consistent_tilt': hit, 'within_card_range': (hit is not None and hit <= g), 'code_route_reproduces': code_ok}
    return out


def solver_reached(narrow, name, ks, ls):
    key = canon(ks, ls)
    hits = []
    for seed, S in narrow['laws'][name]['seeds'].items():
        for g, st in S['path'].items():
            if canon(st['inc_cell'], st['occ_cell']) == key:
                hits.append(f'seed{seed}@{g}')
        if canon(S['twin']['inc_cell'], S['twin']['occ_cell']) == key:
            hits.append(f'seed{seed}@twin')
    return hits


# ============================================================ selftest (invented inputs that are not the declared laws)
def selftest():
    from nrm.fa import prc as P
    from scipy.stats import rankdata
    t0 = time.process_time()
    rng = np.random.default_rng(7)
    checks = {}
    m = 5
    states = []
    for x in range(m):
        def pct(K):
            v = rng.multinomial(100 - 5 * K, np.ones(K) / K) + 5
            return v.tolist()
        states.append({'id': f't{x}', 'n': int(rng.integers(1, 6)) * 100, 'pi_pct': int(rng.integers(10, 91)),
                       'inc_S0_pct': pct(2), 'inc_S1_pct': pct(2), 'occ_S0_pct': pct(5), 'occ_S1_pct': pct(5)})
    N = sum(s['n'] for s in states)
    law = build_law('selftest', states, N, 1, {}, {})
    E = code_E(law)
    # 1. objective: mine vs code on random books and partitions
    md = 0.0
    for _ in range(30):
        k = rng.integers(0, 2, m)
        l = rng.integers(0, 3, m)
        CI = P.clipn(rng.dirichlet(np.ones(2), 2))
        CO = P.clipn(rng.dirichlet(np.ones(5), 3))
        lam = float(rng.choice([0.0, 0.25, 2.0]))
        book = {'CI': CI, 'CO': CO, 'lam': lam, 'w_pair': 1.0, 'w_loc': 1.0, 'kappa': 1.0, 'p0': float((1 - E['pi']).mean()),
                'p1': float(E['pi'].mean())}
        jc = P.objective(E, book, to_rows(law, k), to_rows(law, l))
        jm = ref_objective(law, CI, CO, lam, 1.0, 1.0, 1.0, k, l)
        md = max(md, abs(jc - jm))
    checks['objective_matches_code_1e-9'] = md < 1e-9
    # 2. exact Bayes AUC vs rank AUC on expanded labelled rows (counts n * pi are integers here)
    ok = True
    for _ in range(20):
        g = rng.integers(0, 3, m)
        keys = [int(v) for v in g]
        auc_ex = float(bayes_auc(law, keys))
        s_rows, sc_rows = [], []
        a, b = {}, {}
        for x in range(m):
            a[keys[x]] = a.get(keys[x], 0) + int(law['a'][x])
            b[keys[x]] = b.get(keys[x], 0) + int(law['b'][x])
        for x in range(m):
            n1 = int(law['a'][x]) // 100
            n0 = int(law['n'][x]) - n1
            r = a[keys[x]] / (a[keys[x]] + b[keys[x]])
            s_rows += [1] * n1 + [0] * n0
            sc_rows += [r] * (n1 + n0)
        s_rows, sc_rows = np.array(s_rows), np.array(sc_rows)
        rk = rankdata(sc_rows)
        n1, n0 = s_rows.sum(), (1 - s_rows).sum()
        auc_rank = (rk[s_rows == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        ok &= abs(auc_ex - auc_rank) < 1e-12
    checks['bayes_auc_matches_rank_auc'] = bool(ok)
    # 3. routing cost lines: my argmin vs the code's route on random protected books
    ok = True
    for _ in range(30):
        k = rng.integers(0, 2, m)
        l = rng.integers(0, 3, m)
        book = {'CI': P.clipn(rng.dirichlet(np.ones(2), 2)), 'CO': P.clipn(rng.dirichlet(np.ones(5), 3)), 'lam': 1.0, 'w_pair': 1.0,
                'w_loc': 1.0, 'kappa': 1.0, 'm_shrink': 20.0, 'restrict': None, 'rbar': float(E['pi'].mean()),
                'p0': float((1 - E['pi']).mean()), 'p1': float(E['pi'].mean()), 'tilt': 0.0}
        nb = P.update_book(E, dict(book), to_rows(law, k), to_rows(law, l))
        rc = refresh_check(law, book, nb, k, l)
        checks['refresh_matches_update_book'] = checks.get('refresh_matches_update_book', True) and (
            rc['codeword_max_abs_diff'] < 1e-12 and rc['table_max_abs_diff'] < 1e-12)
        A, Bv, _ = cost_lines(law, nb)
        for t in rng.uniform(0.01, 5, 5):
            nbt = dict(nb)
            nbt['tilt'] = float(t)
            kc, lc = P.route(E, nbt)
            km, lm = argmin_at(A, Bv, t, 3)
            ok &= np.array_equal(to_state(law, kc), km) and np.array_equal(to_state(law, lc), lm)
        ivs, short, _ = tilt_partitions(A, Bv, 4.0, 3)
        for (lo, hi, mid, km, lm) in ivs:                        # interior points of each interval give the same partition
            for tt in (lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)):
                k2, l2 = argmin_at(A, Bv, tt, 3)
                ok &= np.array_equal(k2, km) and np.array_equal(l2, lm)
    checks['route_matches_cost_lines_and_intervals'] = bool(ok)
    # 4. release metrics: full release has zero excess; NLL excess equals the KL formula
    full = full_release(law)
    r = release(law, np.zeros(m, int), np.zeros(m, int), full)
    qbarI = (law['w'][:, None] * law['qI']).sum(0)
    kl = float((law['w'] * (law['qI'] * np.log(law['qI'] / qbarI)).sum(1)).sum())
    checks['nll_excess_is_kl'] = abs(r['excess']['inc_nll_excess'] - kl) < 1e-12
    checks['one_cell_auc_half'] = r['auc_pair'] == 0.5
    # 5. vectorised enumeration statistics agree with the scalar reference on random assignments
    ok = True
    AIa = rng.integers(0, 2, (20, m))
    AOa = rng.integers(0, 3, (20, m))
    VI = view_arrays(law, AIa, 2, law['qI_int'], law['qI'])
    VO = view_arrays(law, AOa, 3, law['qO_int'], law['qO'])
    maO, mbO = merged_masses(VO['sa'], VO['sb'], VO['rep'], 3)
    aucO = auc_masses(maO, mbO)
    for a in range(20):
        rr = release(law, AIa[a], AOa[a], full)
        ok &= abs(VI['nll'][a] - rr['inc_nll']) < 1e-12 and abs(VO['nll'][a] - rr['occ_nll']) < 1e-12
        ok &= abs(VO['acc'][a] - rr['occ_acc']) < 1e-12 and abs(aucO[a] - rr['auc_occupation']) < 1e-12
    checks['enumeration_matches_scalar'] = bool(ok)
    # 6. codeword optimum: no random simplex point does better
    sub, cw = subset_codeword_opt(law, 2.0)
    ok = True
    al, be = (1 - law['pi']) / law['p0'], law['pi'] / law['p1']
    for mask in (1, 3, 7, 31):
        idx = [x for x in range(m) if mask >> x & 1]
        for c in rng.dirichlet(np.ones(5), 200):
            c = 0.98 * cw[mask] + 0.02 * c
            v = (law['w'][idx] * (-(law['qO'][idx] * np.log(c)).sum(1) + 2.0 * (al[idx] * ((law['qO0'][idx] - c) ** 2).sum(1)
                                                                               + be[idx] * ((law['qO1'][idx] - c) ** 2).sum(1)))).sum()
            ok &= v >= sub[mask] - 1e-12
    checks['codeword_optimum_not_beaten'] = bool(ok)
    print(json.dumps({'selftest': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'max_obj_diff': md,
                      'cpu_s': round(time.process_time() - t0, 2)}, indent=1))
    return all(checks.values())


def run_summary(narrow_path, enum_path, out_path):
    """Compact machine-readable results (RESULTS_V1.json) from the narrow and enumeration outputs; no new numerical experiment."""
    Nr = json.loads(Path(narrow_path).read_text())
    En = json.loads(Path(enum_path).read_text())
    rel = lambda r: {k: r[k] for k in ('auc_pair', 'auc_income', 'auc_occupation', 'mi_cell_pair', 'mi_value_pair', 'feasible', 'fails',
                                       'excess') if k in r}
    out = {'schema': 'FSA_EXACT_RESULTS_v1', 'laws_commit': 'c27a91477', 'sources': {'narrow': narrow_path, 'enum': enum_path},
           'laws': {}, 'checks': {}}
    md = rm = 0.0
    cw = tb = 0.0
    for name, L in Nr['laws'].items():
        R = {'full': {k: L['full_release'][k] for k in ('auc_pair', 'auc_income', 'auc_occupation', 'values')},
             'decision_only': rel(L['decision_only'])}
        for tag in ('witness', 'comparator'):
            if tag in L:
                R[tag] = {**rel(L[tag]), 'inc_cell': L[tag]['inc_cell'], 'occ_cell': L[tag]['occ_cell']}
        Eo = En['laws'][name]['optima']
        R['enum'] = {'n_assignments': En['laws'][name]['n_assignments'],
                     'n_feasible_assignments': En['laws'][name]['n_feasible_assignments'],
                     **{tag: (None if o is None else {'value': o['value'], 'inc_cell': o['inc_cell'], 'occ_cell': o['occ_cell'],
                                                      **rel(o), 'realizable_self_consistent': o['realizable'],
                                                      'solver_reached': o.get('solver_reached')})
                        for tag, o in Eo.items()}}
        R['seeds'] = {}
        for s, S in L['seeds'].items():
            T = {'twin': {**rel(S['twin']), 'inc_cell': S['twin']['inc_cell'], 'occ_cell': S['twin']['occ_cell'],
                          'J_task_abs_diff': abs(S['twin']['J_task_code'] - S['twin']['J_task_mine'])},
                 'replay_equals_protected_path': S['replay_equals_protected_path'], 'path': {}, 'steps': []}
            for g, st in S['path'].items():
                o = Eo.get(f'obj_{float(g)}')
                b = Eo.get(f'bound_{float(g)}')
                T['path'][g] = {**rel(st), 'inc_cell': st['inc_cell'], 'occ_cell': st['occ_cell'], 'tilt': st['tilt'],
                                'J_code': st['J_code'], 'J_abs_diff': abs(st['J_code'] - st['J_mine']),
                                'J_twin_same_lam': st['J_twin_same_lam_code'], 'not_worse_than_twin': st['not_worse_than_twin'],
                                'equals_twin_partition': st['equals_twin_partition'],
                                'max_abs_codeword_minus_B1': st['max_abs_codeword_minus_B1'],
                                'enum_objective_optimum': o['value'], 'gap_to_enum_optimum': st['J_code'] - o['value'],
                                'enum_codeword_bound': b['value'], 'gap_to_bound': st['J_code'] - b['value']}
                md = max(md, abs(st['J_code'] - st['J_mine']))
            for r in S['trace']:
                n = r['narrow']
                md = max(md, n['max_abs_objective_diff_code_vs_mine'])
                rm += n['route_mismatches']
                cw = max(cw, n['refresh_check']['codeword_max_abs_diff'])
                tb = max(tb, n['refresh_check']['table_max_abs_diff'])
                row = {'lam': r['lam'], 'step': r['step'], 'J_current': r['J_current'],
                       'accepted_tilt': None if r['accepted'] is None else n['proposals'][r['accepted']]['t'],
                       'proposal_J': [p['J_mine'] for p in n['proposals']],
                       'card': {k: n['card'][k] for k in ('n_partitions', 'best_J_mine', 'best_k', 'best_l', 'best_intervals',
                                                          'n_improving_partitions', 'improving_tilt_intervals')},
                       'wide_best_J': n['wide']['best_J_mine'], 'cur_card_best_J': n['cur_card']['best_J_mine'],
                       'cur_card_has_tables': n['cur_card']['has_tables'], 'sv1_union_best_J': n['sv1_union']['best_J_mine'],
                       'sv1_union_from': n['sv1_union']['from'],
                       'chosen_release': rel(n['chosen_release']), 'best_card_release': rel(n['best_card_release']),
                       'best_minus_chosen': n['best_minus_chosen']}
                for k in ('gap_first_minus_best_card', 'gap_first_minus_best_wide', 'gap_first_minus_best_sv1_union',
                          'missed_improvement_card', 'missed_improvement_wide', 'missed_improvement_sv1_union',
                          'gap_current_minus_best_card', 'gap_current_minus_best_wide'):
                    if k in n:
                        row[k] = n[k]
                T['steps'].append(row)
            hq = S.get('hq_path', {})
            T['hq_path'] = {g: rel(v) | {'occ_cell': v['occ_cell'], 'inc_cell': v['inc_cell']} for g, v in hq.items()} \
                if 'error' not in hq else hq
            R['seeds'][s] = T
        out['laws'][name] = R
    out['checks'] = {'max_abs_objective_diff_code_vs_mine': md, 'route_mismatches': rm,
                     'refresh_codeword_max_abs_diff': cw, 'refresh_table_max_abs_diff': tb}
    Path(out_path).write_text(json.dumps(out, indent=1, default=jsonable))
    print(json.dumps(out['checks']))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    p = sub.add_parser('summary')
    p.add_argument('--narrow', required=True)
    p.add_argument('--enum', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('narrow')
    p.add_argument('--out', required=True)
    p = sub.add_parser('enum')
    p.add_argument('--out', required=True)
    p.add_argument('--narrow', default=None)
    a = ap.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if a.cmd == 'selftest':
            return 0 if selftest() else 1
        if a.cmd == 'narrow':
            run_narrow(a.out)
        elif a.cmd == 'summary':
            run_summary(a.narrow, a.enum, a.out)
        else:
            run_enum(a.out, a.narrow)
    return 0


if __name__ == '__main__':
    sys.exit(main())
