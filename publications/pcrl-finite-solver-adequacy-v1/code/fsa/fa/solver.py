"""Solver variants for the propensity-routed finite-cell objective of `nrm.fa.prc` (card FA_PRC_v1 section 1; campaign
`pcrl_finite_solver_adequacy_v1`, card cards/FA_FSA_v1.md).

What is NOT changed (imported from nrm.fa.prc and called unchanged): the objective (`objective`: mean task cross-entropy to the
codewords + lam * [w_pair I_pi(pair) + w_loc (I_pi(income) + I_pi(occupation))] + lam * kappa * per-SEX guard), the routing family
(`route`: argmin over joint cells of CE + t * [dependence terms on the shrunk tables + kappa * guard], t <= lam), the codeword and
table refresh (`update_book`), the twin (`twin_book`), the strength continuation and its twin reset (`_chain` logic), the restricted
embeddings of HQ and SHUF (`restrict_embedding`, `_prepare`) and the emitted-table rule (B1 cell means, applied by the caller).

What differs is only the PROPOSAL SET and the SELECTION RULE of one Lloyd step:
- 'original'  nrm.fa.prc.lloyd itself: tilts lam, lam/2, ..., lam/64; the first that lowers the exact objective is accepted.
- 'SV1'       exact line searches: for a fixed book the route cost is A + t * D, linear in the tilt t, so the deployable
              partitions route(E, book with tilt t), t in [0, lam], are piecewise constant in t. For two books per step, the
              REFRESHED book (nrm's update_book of the current partition, the original's only proposal book) and the CURRENT book
              (codewords and tables unchanged; its line contains the current partition at the current tilt), every breakpoint
              is enumerated per person (lower envelope of the lines) and the exact objective is evaluated on every interval
              incrementally. The best interval over both lines is chosen (ties: smallest tilt, refreshed line first), and the
              step is accepted only if nrm.fa.prc.objective on the partition that nrm.fa.prc.route itself returns at the
              interval's midpoint tilt is lower than the current objective by more than 1e-12. Proposal resolution is decoupled
              from lam; zero, the previous tilt and every boundary are candidates. The current-book line was added before any
              real-data or registered-law run, after an invented-data trace showed that every refreshed-book partition can be
              worse than the current state (card FA_FSA_v2).
- 'SV1_first' the SV1 candidate set with the original selection rule (refreshed line first, intervals from the largest tilt
              down; accept the first improving one): the equal-effort old-rule control for the selection component.
Every accepted state is (book, route(E, book)) by construction, so the deployed rule reproduces the fitted partition, and the
accepted exact objective decreases strictly. Neither property is an optimality certificate.

Every lloyd variant returns (book, k, l) like nrm.fa.prc.lloyd and appends one dict per Lloyd step to `trace` when a list is given
(objective before, refreshed-book objective on the unchanged partition, candidates, accepted tilt, objective after, moved people).
"""
from __future__ import annotations
import time

import numpy as np

from nrm.fa import prc as P

ACCEPT_TOL = 1e-12
SOLVERS = ('original', 'original_traced', 'SV1', 'SV1_first')


# ---------------------------------------------------------------- route cost as A + t * D (exactly the terms of nrm.fa.prc.route)
def cost_terms(E, book):
    """Flat joint-cell costs (n x KI*KO): A (task cross-entropy to the codewords) and D (the tilt's coefficient: local and pair
    dependence on the book's shrunk tables, computed on the restricted embedding for HQ / SHUF books, plus kappa x guard)."""
    AI, AO = P._xent(E['qI'], book['CI']), P._xent(E['qO'], book['CO'])
    R = P.restrict_embedding(E, book)
    pi, rbar = R['pi'], book['rbar']
    aI, bI = P._dep_ab(book['rI'], rbar)
    aO, bO = P._dep_ab(book['rO'], rbar)
    aP, bP = P._dep_ab(book['rP'], rbar)
    DI = book['w_loc'] * (pi[:, None] * aI[None] + bI[None])
    DO = book['w_loc'] * (pi[:, None] * aO[None] + bO[None])
    if book['kappa'] > 0:
        DO = DO + book['kappa'] * P._guard(R, book['CO'], book)
    KI, KO = book['CI'].shape[0], book['CO'].shape[0]
    A = (AI[:, :, None] + AO[:, None, :]).reshape(len(pi), KI * KO)
    D = (DI[:, :, None] + DO[:, None, :] + book['w_pair'] * (pi[:, None, None] * aP[None] + bP[None])).reshape(len(pi), KI * KO)
    return A, D


def envelope(A, D, t_max, chunk=2048, max_iter=10000):
    """Per person, the argmin cell of A + t * D on [0, t_max] (lower envelope). Returns (c0, ev_t, ev_i, ev_c): the cells at
    t = 0+ (lexicographic: smallest A, then smallest D, then lowest index) and the events (tilt, person, new cell), sorted by tilt."""
    n, C = A.shape
    c0 = np.empty(n, dtype=np.int64)
    ts, iis, cs = [], [], []
    idx = np.arange(C)
    for a in range(0, n, chunk):
        Ac, Dc = A[a:a + chunk], D[a:a + chunk]
        m = Ac.min(1, keepdims=True)
        tie = Ac <= m + 1e-12 * np.maximum(1.0, np.abs(m))
        cur = np.where(tie, Dc, np.inf).argmin(1)
        c0[a:a + chunk] = cur
        rows = np.arange(len(cur))
        t_cur = np.zeros(len(cur))
        live = np.ones(len(cur), dtype=bool)
        for _ in range(max_iter):
            if not live.any():
                break
            r = rows[live]
            a_cur, d_cur = Ac[r, cur[r]][:, None], Dc[r, cur[r]][:, None]
            down = Dc[r] < d_cur - 1e-15
            with np.errstate(divide='ignore', invalid='ignore'):
                tau = np.where(down, (Ac[r] - a_cur) / (d_cur - Dc[r]), np.inf)
            tau = np.maximum(tau, t_cur[r][:, None])                        # optimality at t_cur: crossings are not earlier
            tmin = tau.min(1)
            hit = tmin <= t_max
            if not hit.any():
                live[r] = False
                break
            near = tau <= (tmin * (1 + 1e-12) + 1e-15)[:, None]
            new = np.where(near, Dc[r], np.inf).argmin(1)                   # after the crossing the smallest slope wins
            rh = r[hit]
            ts.append(tmin[hit]); iis.append(rh + a); cs.append(new[hit])
            cur[rh] = new[hit]
            t_cur[rh] = tmin[hit]
            live[r[~hit]] = False
        else:
            raise RuntimeError('envelope: max_iter reached')
    if ts:
        ev_t, ev_i, ev_c = np.concatenate(ts), np.concatenate(iis), np.concatenate(cs)
        o = np.lexsort((ev_i, ev_t))
        return c0, ev_t[o], ev_i[o], ev_c[o]
    return c0, np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------- the exact objective, incrementally (ranking only)
class IncrementalObjective:
    """nrm.fa.prc.objective for one book, maintained under single-person moves (used to rank the line-search intervals; the
    accepted value is always recomputed with nrm.fa.prc.objective itself)."""

    def __init__(self, E, book, k, l):
        n = len(k)
        self.n, self.KI, self.KO = n, book['CI'].shape[0], book['CO'].shape[0]
        self.lam, self.wp, self.wl, self.kappa = float(book['lam']), float(book['w_pair']), float(book['w_loc']), float(book['kappa'])
        self.ceI = -(E['qI'] @ np.log(book['CI']).T)                       # n x KI
        self.ceO = -(E['qO'] @ np.log(book['CO']).T)                       # n x KO
        self.pi = np.asarray(E['pi'], dtype=np.float64) if 'pi' in E else np.zeros(n)
        self.rb = float(self.pi.mean())
        if self.lam > 0 and self.kappa > 0:
            g = np.zeros((n, self.KO))
            for key, w in (('qO0', (1 - self.pi) / book['p0']), ('qO1', self.pi / book['p1'])):
                Q = E[key]
                g += w[:, None] * (((Q[:, None, :] - book['CO'][None]) ** 2).sum(2))
            self.g = g
        else:
            self.g = np.zeros((n, self.KO))
        self.k, self.l = k.copy(), l.copy()
        j = k * self.KO + l
        J = self.KI * self.KO
        self.nP = np.bincount(j, minlength=J).astype(np.float64)
        self.sP = np.bincount(j, weights=self.pi, minlength=J)
        self.nI = np.bincount(k, minlength=self.KI).astype(np.float64)
        self.sI = np.bincount(k, weights=self.pi, minlength=self.KI)
        self.nO = np.bincount(l, minlength=self.KO).astype(np.float64)
        self.sO = np.bincount(l, weights=self.pi, minlength=self.KO)
        idx = np.arange(n)
        self.S_ce = float(self.ceI[idx, k].sum() + self.ceO[idx, l].sum())
        self.S_g = float(self.g[idx, l].sum())

    def _cell(self, n, s):
        if n <= 0:
            return 0.0
        r = min(max(s / n, 1e-9), 1 - 1e-9)
        rb = self.rb
        return n / self.n * (r * np.log(r / rb) + (1 - r) * np.log((1 - r) / (1 - rb)))

    def _mi_all(self):
        out = []
        for nn, ss in ((self.nP, self.sP), (self.nI, self.sI), (self.nO, self.sO)):
            occ = nn > 0
            r = np.clip(ss[occ] / nn[occ], 1e-9, 1 - 1e-9)
            kl = r * np.log(r / self.rb) + (1 - r) * np.log((1 - r) / (1 - self.rb))
            out.append(float((nn[occ] / self.n * kl).sum()))
        return out

    def value(self):
        J = self.S_ce / self.n
        if self.lam > 0:
            mP, mI, mO = self._mi_all()
            J += self.lam * (self.wp * mP + self.wl * (mI + mO)) + self.lam * self.kappa * self.S_g / self.n
        return J

    def init_mi(self):
        self.mP, self.mI, self.mO = self._mi_all()

    def move(self, i, c_new):
        k0, l0 = self.k[i], self.l[i]
        k1, l1 = divmod(int(c_new), self.KO)
        if k0 == k1 and l0 == l1:
            return
        p = self.pi[i]
        self.S_ce += self.ceI[i, k1] - self.ceI[i, k0] + self.ceO[i, l1] - self.ceO[i, l0]
        self.S_g += self.g[i, l1] - self.g[i, l0]
        for nn, ss, a, b, attr in ((self.nP, self.sP, k0 * self.KO + l0, k1 * self.KO + l1, 'mP'),
                                   (self.nI, self.sI, k0, k1, 'mI'), (self.nO, self.sO, l0, l1, 'mO')):
            if a == b:
                continue
            old = self._cell(nn[a], ss[a]) + self._cell(nn[b], ss[b])
            nn[a] -= 1; ss[a] -= p; nn[b] += 1; ss[b] += p
            setattr(self, attr, getattr(self, attr) + self._cell(nn[a], ss[a]) + self._cell(nn[b], ss[b]) - old)
        self.k[i], self.l[i] = k1, l1

    def value_fast(self):
        J = self.S_ce / self.n
        if self.lam > 0:
            J += self.lam * (self.wp * self.mP + self.wl * (self.mI + self.mO)) + self.lam * self.kappa * self.S_g / self.n
        return J


def line_search(E, nb, t_max):
    """Candidate intervals of the tilt for the fixed refreshed book nb on [0, t_max]: a list of (t_lo, t_hi, objective), one per
    distinct deployable partition along the tilt (objective evaluated incrementally), and the number of person-level events."""
    A, D = cost_terms(E, nb)
    c0, ev_t, ev_i, ev_c = envelope(A, D, t_max)
    KO = nb['CO'].shape[0]
    inc = IncrementalObjective(E, nb, c0 // KO, c0 % KO)
    inc.init_mi()
    cands, lo, pos = [], 0.0, 0
    for b in np.unique(ev_t):
        if b > lo or not cands:
            cands.append((lo, float(b), inc.value_fast()))
        while pos < len(ev_t) and ev_t[pos] == b:
            inc.move(int(ev_i[pos]), int(ev_c[pos]))
            pos += 1
        lo = float(b)
    cands.append((lo, float(t_max), inc.value_fast()))
    return cands, len(ev_t)


def _mid(lo, hi):
    return 0.5 * (lo + hi) if hi > lo else lo


def lloyd_sv1(E, book, steps, k=None, l=None, select='best', trace=None, lines=('refreshed', 'current')):
    """Damped Lloyd with exact tilt line searches (see module docstring). Same signature and return as nrm.fa.prc.lloyd.
    Proposal class per step: the deployable partitions along the tilt line t in [0, lam] of the REFRESHED book (nrm's
    update_book of the current partition, as in the original) and of the CURRENT book (its codewords and tables unchanged; it
    reproduces the current partition at the current tilt). select='best': candidates in increasing incremental objective;
    select='first': in decreasing tilt, refreshed line first (the original order). The first candidate whose exact objective
    (nrm.fa.prc.objective on nrm.fa.prc.route's own partition at the interval midpoint) is below the current objective by more
    than ACCEPT_TOL is accepted; candidates whose incremental value is not below it are skipped ('best' stops there)."""
    if k is None:
        k, l = P.route(E, book)
    J = P.objective(E, book, k, l)
    lam = float(book['lam'])
    if lam <= 0:
        return P.lloyd(E, book, steps, k, l)
    for step in range(int(steps)):
        t0 = time.process_time()
        nb = P.update_book(E, dict(book), k, l)
        J_refresh = P.objective(E, nb, k, l)
        books = {'refreshed': nb}
        if 'current' in lines and all(key in book for key in ('rI', 'rO', 'rP')):
            books['current'] = dict(book)
        cands, n_events = [], {}
        for name in [x for x in lines if x in books]:
            c, ne = line_search(E, books[name], lam)
            cands += [(lo, hi, jv, name) for lo, hi, jv in c]
            n_events[name] = int(ne)
        if select == 'best':
            order = sorted(range(len(cands)), key=lambda c: (cands[c][2], cands[c][0], cands[c][3] != 'refreshed'))
        else:
            order = sorted(range(len(cands)), key=lambda c: (cands[c][3] != 'refreshed', -cands[c][0]))
        rec = {'step': step, 'J_before': J, 'J_refresh_same_partition': J_refresh, 'n_intervals': len(cands),
               'n_events': n_events, 'select': select, 'best_incremental': min(c[2] for c in cands),
               'best_incremental_by_line': {nm: min((c[2] for c in cands if c[3] == nm), default=None) for nm in books}}
        acc, tried = False, 0
        for c in order:
            lo, hi, jv, name = cands[c]
            if jv >= J - ACCEPT_TOL:
                if select == 'best':
                    break
                continue
            t = _mid(lo, hi)
            nb2 = dict(books[name])
            nb2['tilt'] = float(t)
            k2, l2 = P.route(E, nb2)
            j = P.objective(E, nb2, k2, l2)
            tried += 1
            if j < J - ACCEPT_TOL:
                rec.update({'accepted_tilt': float(t), 'accepted_line': name, 'interval': [float(lo), float(hi)], 'J_after': j,
                            'J_interval_incremental': jv, 'moved': int(((k2 != k) | (l2 != l)).sum())})
                book, k, l, J, acc = nb2, k2, l2, j, True
                break
        rec.update({'route_evaluations': tried, 'cpu_s': round(time.process_time() - t0, 3)})
        if not acc:
            rec['accepted_tilt'] = None
        if trace is not None:
            trace.append(rec)
        if not acc:
            break
    return book, k, l


def lloyd_original_traced(E, book, steps, k=None, l=None, trace=None):
    """nrm.fa.prc.lloyd, statement for statement, recording each step (selftest: identical book, k, l)."""
    if k is None:
        k, l = P.route(E, book)
    J = P.objective(E, book, k, l)
    lam = float(book['lam'])
    tilts = [lam / 2 ** j for j in range(P.TILT_STEPS)] if lam > 0 else [0.0]
    for step in range(int(steps)):
        nb = P.update_book(E, dict(book), k, l)
        rec = {'step': step, 'J_before': J, 'J_refresh_same_partition': P.objective(E, nb, k, l), 'tried': []}
        acc = False
        for t in tilts:
            nb['tilt'] = float(t)
            k2, l2 = P.route(E, nb)
            j = P.objective(E, nb, k2, l2)
            rec['tried'].append({'tilt': float(t), 'J': j, 'moved': int(((k2 != k) | (l2 != l)).sum())})
            if j < J - 1e-12:
                rec.update({'accepted_tilt': float(t), 'J_after': j, 'moved': int(((k2 != k) | (l2 != l)).sum())})
                book, k, l, J, acc = dict(nb), k2, l2, j, True
                break
        if trace is not None:
            if not acc:
                rec['accepted_tilt'] = None
            trace.append(rec)
        if not acc:
            break
    return book, k, l


def _lloyd(solver, trace):
    if solver == 'original':
        return lambda E, b, s, k, l: P.lloyd(E, b, s, k, l)
    if solver == 'original_traced':
        return lambda E, b, s, k, l: lloyd_original_traced(E, b, s, k, l, trace=trace)
    if solver == 'SV1':
        return lambda E, b, s, k, l: lloyd_sv1(E, b, s, k, l, select='best', trace=trace)
    if solver == 'SV1_first':
        return lambda E, b, s, k, l: lloyd_sv1(E, b, s, k, l, select='first', trace=trace)
    raise ValueError(solver)


def chain(E, twin, k0, l0, lams, rp, solver='SV1', grid=None, trace=None):
    """nrm.fa.prc.protected_path with a selectable Lloyd step: the same preparation (_prepare: SHUF donor copy), the same
    deterministic continuation over the grid strengths below the largest target, the same twin reset and the same path_steps.
    With solver='original' it calls nrm.fa.prc.protected_path itself. trace (optional dict) collects per-strength step records."""
    grid = tuple(grid if grid is not None else P.LAMBDA_GRID)
    lams = sorted({float(x) for x in lams})
    if solver == 'original' and tuple(P.LAMBDA_GRID) == grid:
        return P.protected_path(E, twin, k0, l0, lams, rp)
    Eu, book0 = P._prepare(E, twin, k0, l0, rp)
    start = (dict(book0), k0, l0)
    book, k, l = dict(book0), k0, l0
    out = {}
    for lam_t in sorted({g for g in grid if g < max(lams)} | set(lams)):
        book['lam'] = start[0]['lam'] = float(lam_t)
        reset = bool(P.objective(Eu, start[0], k0, l0) < P.objective(Eu, book, k, l))
        if reset:
            book, k, l = dict(start[0]), k0, l0
        steps = [] if trace is not None else None
        book, k, l = _lloyd(solver, steps)(Eu, book, rp['path_steps'], k, l)
        if trace is not None:
            trace[lam_t] = {'reset_to_twin': reset, 'steps': steps, 'objective': P.objective(Eu, book, k, l),
                            'tilt': float(book.get('tilt', 0.0))}
        if lam_t in lams:
            out[lam_t] = (dict(book), k.copy(), l.copy())
    return out


# ---------------------------------------------------------------- invented-data selftest (no real data, no labels)
def selftest():
    rng = np.random.default_rng(20261021)
    checks = {}
    n, KI, KO = 900, 3, 4
    lg = rng.normal(size=(n, 2)) * 1.3
    qI = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
    lo = rng.normal(size=(n, 5)) * 1.1
    qO = np.exp(lo) / np.exp(lo).sum(1, keepdims=True)
    pi = 1 / (1 + np.exp(-(1.5 * (qO[:, 0] - qO[:, 1]) + 0.8 * rng.normal(size=n))))
    shift = rng.normal(size=(n, 5)) * 0.3
    qO0 = np.exp(lo + shift) / np.exp(lo + shift).sum(1, keepdims=True)
    qO1 = np.exp(lo - shift) / np.exp(lo - shift).sum(1, keepdims=True)
    E = {'qI': P.clipn(qI), 'qO': P.clipn(qO), 'pi': pi, 'qO0': P.clipn(qO0), 'qO1': P.clipn(qO1)}
    rp = dict(P.ROUTE)
    rp.update({'path_steps': 8, 'init_steps': 20})
    tb, k0, l0 = P.twin_book(E, KI, KO, 0, rp)
    # 1. cost_terms reproduces nrm.fa.prc.route at several tilts (protected start book)
    Eu, b0 = P._prepare(E, tb, k0, l0, rp)
    b0['lam'] = 2.0
    nb = P.update_book(Eu, dict(b0), k0, l0)
    A, D = cost_terms(Eu, nb)
    ok = True
    for t in (0.05, 0.3, 1.0, 2.0):
        nb2 = dict(nb); nb2['tilt'] = t
        k, l = P.route(Eu, nb2)
        c = (A + t * D).argmin(1)
        ok &= bool(np.array_equal(c, k * KO + l))
    checks['cost_terms_equal_route'] = ok
    # 2. the envelope's interval partitions equal route at interval midpoints; incremental objective equals P.objective
    cands, nev = line_search(Eu, nb, 2.0)
    okp, oko = True, True
    c0, ev_t, ev_i, ev_c = envelope(A, D, 2.0)
    for lo_, hi_, jv in cands[:: max(1, len(cands) // 25)]:
        t = _mid(lo_, hi_)
        nb2 = dict(nb); nb2['tilt'] = t
        k, l = P.route(Eu, nb2)
        cur = c0.copy()
        m = ev_t <= lo_ + 0.0
        if lo_ > 0:
            for i, cc in zip(ev_i[m], ev_c[m]):
                cur[i] = cc
        okp &= bool(np.array_equal(cur, k * KO + l)) if t > 0 else True
        oko &= abs(P.objective(Eu, nb2, k, l) - jv) < 1e-9
    checks['interval_partitions_equal_route'] = bool(okp)
    checks['incremental_objective_equals_exact'] = bool(oko)
    checks['events_found'] = nev > 0 and len(cands) > 1
    # 3. the traced original equals nrm.fa.prc.lloyd; chain('original') equals protected_path
    b1 = dict(b0)
    ba, ka, la = P.lloyd(Eu, dict(b1), 8, k0, l0)
    tr = []
    bb, kb, lb = lloyd_original_traced(Eu, dict(b1), 8, k0, l0, trace=tr)
    checks['traced_original_equals_lloyd'] = bool(np.array_equal(ka, kb) and np.array_equal(la, lb)
                                                  and np.array_equal(ba['CO'], bb['CO']) and ba['tilt'] == bb['tilt'])
    pa = P.protected_path(E, tb, k0, l0, [1.0, 2.0], rp)
    pb = chain(E, tb, k0, l0, [1.0, 2.0], rp, solver='original_traced', trace={})
    checks['chain_original_traced_equals_protected_path'] = all(
        np.array_equal(pa[x][1], pb[x][1]) and np.array_equal(pa[x][2], pb[x][2]) for x in (1.0, 2.0))
    # 4. SV1 accepted objectives decrease and every accepted state is rule-consistent (SV1 is not guaranteed to beat the original)
    tr1 = {}
    ps = chain(E, tb, k0, l0, [1.0, 2.0], rp, solver='SV1', trace=tr1)
    mono, cons = True, True
    for lam_t, r in tr1.items():
        for s in r['steps']:
            if s.get('accepted_tilt') is not None:
                mono &= s['J_after'] < s['J_before']
    for x in (1.0, 2.0):
        bk, k, l = ps[x]
        k2, l2 = P.route(E, bk)
        cons &= bool(np.array_equal(k, k2) and np.array_equal(l, l2))
    checks['sv1_monotone'] = mono
    checks['sv1_rule_consistent'] = cons
    jo = {x: P.objective(E, pa[x][0], pa[x][1], pa[x][2]) for x in (1.0, 2.0)}
    js = {x: P.objective(E, ps[x][0], ps[x][1], ps[x][2]) for x in (1.0, 2.0)}
    checks['sv1_objective_values'] = {str(x): [round(jo[x], 6), round(js[x], 6)] for x in (1.0, 2.0)}
    # 5. HQ (restricted) books go through the same machinery
    rph = dict(rp); rph['restrict'] = 'twin_cells'
    ph = chain(E, tb, k0, l0, [2.0], rph, solver='SV1', trace={})
    bk, k, l = ph[2.0]
    k2, l2 = P.route(E, bk)
    checks['sv1_hq_rule_consistent'] = bool(np.array_equal(k, k2) and np.array_equal(l, l2))
    passed = all(v for v in checks.values() if isinstance(v, bool))
    return {'passed': passed, 'checks': checks}


if __name__ == '__main__':
    import json
    import sys
    r = selftest()
    print(json.dumps(r, indent=1, default=str))
    sys.exit(0 if r['passed'] else 1)
