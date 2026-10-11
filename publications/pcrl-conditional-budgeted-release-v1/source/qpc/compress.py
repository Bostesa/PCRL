"""Stage B: class-preserving coarse compression of fine teacher-score partitions at asymmetric caps (m1, m2).

Families: CLASS (m = 1), FINE-TASK, LOCAL, SEQ-12, SEQ-21, JOINT. Provenance: the objective arithmetic, the label
convention, the greedy tie rule and the single-cell exchange are those of dpc/compress.py at source SHA 0a7b05a5
(re-implemented here with cached candidate tables and an exact n log n lookup; dpc is not edited). qpc changes, all
fixed before any real fit (METHOD_CARD section 6):
  * per-recipient caps m1 (income) and m2 (occupation) instead of one m;
  * "at most the cap": after the cap is reached, objective-improving merges (increment < -TOL) are applied in any
    class with >= 2 coarse cells, and every refinement sweep ends with such a merge pass;
  * sequential correction: the first recipient is optimised under the actual F_joint with the other recipient at its
    CLASS-ONLY release (the decision is always disclosed), NOT under D + 1.5 lam I; the old surrogate is fitted as a
    diagnostic only and both are reported (``baseline_correction``);
  * JOINT keeps the unchanged witnesses as candidates, refines the five starts with at most five sweeps, asserts the
    fitted F_joint is no worse than every unchanged witness and reports unresolved local optima.

Objective on the N fitting rows (natural logs):
    D_i = (1/N) sum_c [A_c - S_c . log q_c],  q_c = smooth(S_c / n_c, class_c)
    I_i = I(S; C_i), I_12 = I(S; C_1, C_2): plug-in MI of the exact fitting count tables (0 log 0 = 0), computed as
          (1/N)[sum_cols phi(col) - sum_s n_s log n_s + N log N],  phi(col) = sum_s x_s log x_s - n_col log n_col
    F_task = D1 + D2;  F_local = D1 + D2 + lam (I1 + I2) / 2;  F_joint = D1 + D2 + lam ((I1 + I2) / 2 + I12)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dpc.compress import h_cost, mi_plugin
from qpc import kmeans as KM
from qpc import release as RL

TOL = 1e-12
TIE_TOL = 1e-12
SWEEPS = 5
FAMILIES = ("CLASS", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
WITNESSES = ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
JOINT_START_ORDER = ("JOINT-GREEDY", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
PERM_SEED = 20261006
N_PERM = 100
OLD_RULE_DIAGNOSTIC = True      # SEQ: also fit the dpc stage-1 surrogate as a reported diagnostic (never selected)


# ----------------------------------------------------------------------------------------------- weights
@dataclass(frozen=True)
class Weights:
    """F = wD1 D1 + wD2 D2 + wI1 I1 + wI2 I2 + w12 I12."""
    wD1: float
    wD2: float
    wI1: float
    wI2: float
    w12: float

    def wD(self, r):
        return self.wD1 if r == 1 else self.wD2

    def wI(self, r):
        return self.wI1 if r == 1 else self.wI2


def W_task(r=None):
    if r is None:
        return Weights(1.0, 1.0, 0.0, 0.0, 0.0)
    return Weights(float(r == 1), float(r == 2), 0.0, 0.0, 0.0)


def W_local(lam, r=None):
    if r is None:
        return Weights(1.0, 1.0, lam / 2, lam / 2, 0.0)
    return Weights(float(r == 1), float(r == 2), lam / 2 if r == 1 else 0.0, lam / 2 if r == 2 else 0.0, 0.0)


def W_joint(lam):
    return Weights(1.0, 1.0, lam / 2, lam / 2, lam)


def W_old_seq_stage1(lam, r):
    """dpc stage-one surrogate D_r + 1.5 lam I_r (the other recipient constant). Diagnostic only in qpc."""
    return Weights(float(r == 1), float(r == 2), 1.5 * lam if r == 1 else 0.0, 1.5 * lam if r == 2 else 0.0, 0.0)


def F_values(t, lam):
    out = {"F_task": t["D1"] + t["D2"]}
    if lam is not None:
        out["F_local"] = t["D1"] + t["D2"] + lam * (t["I1"] + t["I2"]) / 2
        out["F_joint"] = t["D1"] + t["D2"] + lam * ((t["I1"] + t["I2"]) / 2 + t["I12"])
    return out


# ----------------------------------------------------------------------------------------------- tables
def fine_table(f1, f2, s, F1, F2):
    idx = (np.asarray(s, dtype=np.int64) * F1 + f1) * F2 + f2
    return np.bincount(idx, minlength=2 * F1 * F2).reshape(2, F1, F2).astype(np.int64)


def check_sex(S_fit, N):
    s = np.asarray(S_fit)
    if s.shape != (N,):
        raise ValueError(f"S_fit must have shape ({N},)")
    if not np.all((s == 0) | (s == 1)):
        raise ValueError("S_fit must be binary {0, 1}")
    return s.astype(np.int64)


def class_labels(fine):
    first = {}
    for f in range(fine.F):
        first.setdefault(int(fine.cell_class[f]), f)
    return np.array([first[int(c)] for c in fine.cell_class], dtype=np.int64)


def labels_from_policy(pol):
    """Label per fine cell = lowest member fine index of its token."""
    first = {}
    for f in range(pol.fine.F):
        first.setdefault(int(pol.cell_token[f]), f)
    return np.array([first[int(t)] for t in pol.cell_token], dtype=np.int64)


# ----------------------------------------------------------------------------------------------- state engine
class State:
    """Both recipients' coarse states (label per fine cell = a member fine index, unique per coarse cell; coarse
    cells never cross predicted classes), per-label statistics and the exact count tables by label."""

    def __init__(self, fine1, fine2, Tfine, lab1, lab2):
        self.fine = {1: fine1, 2: fine2}
        self.Tfine = np.asarray(Tfine, dtype=np.int64)
        self.N = int(self.Tfine.sum())
        self.XL = np.arange(self.N + 1, dtype=np.float64)
        self.XL = self.XL * np.log(np.where(self.XL > 0, self.XL, 1.0))          # n log n, exact per entry
        self.ns = self.Tfine.sum((1, 2))
        self.const_mi = float(-self.XL[self.ns].sum() + self.XL[self.N])
        self.Asum = {r: float(np.sum(self.fine[r].A)) for r in (1, 2)}
        self.Tfine_r = {1: self.Tfine.sum(2), 2: self.Tfine.sum(1)}
        self.lab, self.members, self.n, self.S, self.h, self.cl = {}, {}, {}, {}, {}, {}
        for r, lab in ((1, lab1), (2, lab2)):
            self._init_rec(r, lab)
        self.rebuild_tables()
        self.work = {"greedy_candidates": 0, "greedy_steps": 0, "extra_merge_candidates": 0, "extra_merges": 0,
                     "refine_candidates": 0, "refine_moves": 0}

    # setup ---------------------------------------------------------------------------------------------
    def _init_rec(self, r, lab):
        fine = self.fine[r]
        lab = np.asarray(lab, dtype=np.int64).copy()
        if lab.shape != (fine.F,):
            raise ValueError("labels must cover every fine cell")
        self.lab[r] = lab
        self.n[r] = np.zeros(fine.F, dtype=np.int64)
        self.S[r] = np.zeros((fine.F, fine.K))
        self.h[r] = np.zeros(fine.F)
        mem = {}
        for f in range(fine.F):
            mem.setdefault(int(lab[f]), []).append(f)
        for l, m in mem.items():
            if l not in m:
                raise ValueError("a label must be one of its own members")
            if np.unique(fine.cell_class[m]).size != 1:
                raise ValueError("a coarse cell may not mix predicted classes")
        self.members[r] = mem
        self.cl.setdefault(r, {})
        self.cl[r] = {c: sorted(l for l in mem if fine.cell_class[l] == c) for c in range(fine.K)}
        for l in mem:
            self._recompute(r, l)

    def _recompute(self, r, l):
        fine = self.fine[r]
        S = np.zeros(fine.K)
        n = 0
        for f in self.members[r][l]:            # increasing fine index (members kept sorted)
            S = S + fine.S[f]
            n += int(fine.n[f])
        self.S[r][l], self.n[r][l] = S, n
        self.h[r][l] = h_cost(S[None], np.array([n]), fine.cell_class[l], fine.K)[0]

    def rebuild_tables(self):
        F1, F2 = self.fine[1].F, self.fine[2].F
        A = np.zeros((2, F1, F2), dtype=np.int64)
        np.add.at(A, (slice(None), self.lab[1]), self.Tfine)
        B = np.zeros((2, F1, F2), dtype=np.int64)
        np.add.at(B.transpose(0, 2, 1), (slice(None), self.lab[2]), A.transpose(0, 2, 1))
        self.T12 = B
        self.Tr = {1: B.sum(2), 2: B.sum(1)}

    # helpers -------------------------------------------------------------------------------------------
    def other(self, r):
        return 2 if r == 1 else 1

    def labels(self, r):
        """Canonical grouping: every fine cell labelled by the lowest fine index of its coarse cell (a move can carry
        the label cell itself out of its cell, so the internal label is not exported)."""
        out = np.empty(self.fine[r].F, dtype=np.int64)
        for mem in self.members[r].values():
            out[mem] = min(mem)
        return out

    def alive(self, r):
        return np.array(sorted(self.members[r]), dtype=np.int64)

    def class_labels(self, r, c):
        return np.array(self.cl[r][c], dtype=np.int64)

    def count(self, r, c):
        return len(self.cl[r][c])

    def T12o(self, r):
        return self.T12 if r == 1 else self.T12.transpose(0, 2, 1)

    def phi(self, X):
        """Column terms of plug-in MI over axis 0 (2 sensitive values), exact via the n log n table."""
        return self.XL[X[0]] + self.XL[X[1]] - self.XL[X[0] + X[1]]

    def G(self, r):
        """Fine cells of recipient r aggregated by the other recipient's current labels: (2, F_r, F_other)."""
        o = self.other(r)
        Tf = self.Tfine if r == 1 else self.Tfine.transpose(0, 2, 1)
        out = np.zeros(Tf.shape, dtype=np.int64)
        np.add.at(out.transpose(2, 0, 1), self.lab[o], Tf.transpose(2, 0, 1))
        return out

    # objective -----------------------------------------------------------------------------------------
    def terms(self):
        """Every term recomputed from scratch from the per-label statistics and the label tables."""
        t = {}
        for r in (1, 2):
            labs = self.alive(r)
            t[f"D{r}"] = float((self.Asum[r] + float(np.sum(self.h[r][labs]))) / self.N)
            t[f"I{r}"] = float((self.phi(self.Tr[r]).sum() + self.const_mi) / self.N)
        t["I12"] = float((self.phi(self.T12).sum() + self.const_mi) / self.N)
        return t

    def value(self, W):
        t = self.terms()
        return W.wD1 * t["D1"] + W.wD2 * t["D2"] + W.wI1 * t["I1"] + W.wI2 * t["I2"] + W.w12 * t["I12"]

    # merges --------------------------------------------------------------------------------------------
    def merge_deltas(self, r, c, W):
        """All merges (a < b) of class c on recipient r, lexicographic: (labs, ia, ib, delta, dD, dI, dI12)."""
        fine = self.fine[r]
        labs = self.class_labels(r, c)
        ia, ib = np.triu_indices(labs.size, 1)
        a, b = labs[ia], labs[ib]
        N = self.N
        S, n, h = self.S[r], self.n[r], self.h[r]
        dD = (h_cost(S[a] + S[b], n[a] + n[b], c, fine.K) - h[a] - h[b]) / N
        Tr = self.Tr[r]
        ph = self.phi(Tr[:, labs])
        dI = (self.phi(Tr[:, a] + Tr[:, b]) - ph[ia] - ph[ib]) / N
        delta = W.wD(r) * dD + W.wI(r) * dI
        dI12 = None
        if W.w12 != 0.0:
            X = self.T12o(r)[:, labs][:, :, self.alive(self.other(r))]           # (2, L, Lo)
            rowphi = self.phi(X).sum(-1)
            dI12 = (self.phi(X[:, ia] + X[:, ib]).sum(-1) - rowphi[ia] - rowphi[ib]) / N
            delta = delta + W.w12 * dI12
        return labs, ia, ib, delta, dD, dI, dI12

    def apply_merge(self, r, a, b):
        cc = self.fine[r].cell_class
        assert a < b and cc[a] == cc[b]
        mem = sorted(self.members[r][a] + self.members[r].pop(b))
        self.members[r][a] = mem
        self.cl[r][int(cc[a])].remove(b)
        self.lab[r][mem] = a
        self._recompute(r, a)
        self.n[r][b], self.S[r][b], self.h[r][b] = 0, 0.0, 0.0
        if r == 1:
            self.T12[:, a, :] += self.T12[:, b, :]
            self.T12[:, b, :] = 0
        else:
            self.T12[:, :, a] += self.T12[:, :, b]
            self.T12[:, :, b] = 0
        self.Tr[r][:, a] += self.Tr[r][:, b]
        self.Tr[r][:, b] = 0

    # moves ---------------------------------------------------------------------------------------------
    def move_deltas(self, r, f, W, G):
        """Moves of fine cell f to every other coarse cell of its class: (targets, delta, dD, dI, dI12) or None.
        A move never empties a coarse cell (that is a merge, handled by the merge pass)."""
        a = int(self.lab[r][f])
        mem = self.members[r][a]
        if len(mem) <= 1:
            return None
        fine = self.fine[r]
        c = int(fine.cell_class[f])
        B = np.array([l for l in self.class_labels(r, c) if l != a], dtype=np.int64)
        if B.size == 0:
            return None
        N = self.N
        Sa = np.zeros(fine.K)
        for g in mem:
            if g != f:
                Sa = Sa + fine.S[g]
        na = int(self.n[r][a] - fine.n[f])
        ha = h_cost(Sa[None], np.array([na]), c, fine.K)[0]
        hb = h_cost(self.S[r][B] + fine.S[f], self.n[r][B] + fine.n[f], c, fine.K)
        dD = (ha + hb - self.h[r][a] - self.h[r][B]) / N
        Tr = self.Tr[r]
        tf = self.Tfine_r[r][:, f]
        dI = (self.phi((Tr[:, a] - tf)[:, None]) + self.phi(Tr[:, B] + tf[:, None]) - self.phi(Tr[:, [a]])
              - self.phi(Tr[:, B])) / N
        delta = W.wD(r) * dD + W.wI(r) * dI
        dI12 = None
        if W.w12 != 0.0:
            alive_o = self.alive(self.other(r))
            g = G[:, f, alive_o]
            X = self.T12o(r)
            Xa = X[:, a, alive_o]
            XB = X[:, B][:, :, alive_o]
            dI12 = (self.phi(Xa - g).sum() + self.phi(XB + g[:, None, :]).sum(-1) - self.phi(Xa).sum()
                    - self.phi(XB).sum(-1)) / N
            delta = delta + W.w12 * dI12
        return B, delta, dD, dI, dI12

    def apply_move(self, r, f, b, G):
        a = int(self.lab[r][f])
        cc = self.fine[r].cell_class
        assert a != b and cc[a] == cc[b] and len(self.members[r][a]) > 1
        self.members[r][a] = [g for g in self.members[r][a] if g != f]
        self.members[r][b] = sorted(self.members[r][b] + [f])
        self.lab[r][f] = b
        self._recompute(r, a)
        self._recompute(r, b)
        tf = self.Tfine_r[r][:, f]
        self.Tr[r][:, a] -= tf
        self.Tr[r][:, b] += tf
        g = G[:, f, :]
        if r == 1:
            self.T12[:, a, :] -= g
            self.T12[:, b, :] += g
        else:
            self.T12[:, :, a] -= g
            self.T12[:, :, b] += g


# ----------------------------------------------------------------------------------------------- search
def _merge_record(r, c, labs, ia, ib, delta, dD, dI, dI12, j, kind, tied):
    return {"kind": kind, "recipient": r, "class": c, "a": int(labs[ia[j]]), "b": int(labs[ib[j]]),
            "increment": float(delta[j]), "dD": float(dD[j]), "dI_own": float(dI[j]),
            "dI12": None if dI12 is None else float(dI12[j]), "tied_candidates": int(tied)}


def _merge_loop(st, recips, W, caps, log, improving):
    """improving=False: merge until every class of every recipient in ``recips`` has <= cap coarse cells (positive
    increments allowed). improving=True: apply merges with increment < -TOL in any class with >= 2 cells until none.
    Each step applies the smallest increment; candidates within TIE_TOL are tied and the first in lexicographic
    (recipient, class, a, b) order wins. Candidate tables are cached per (recipient, class) and invalidated when that
    class merges, and for the other recipient whenever the I12 weight is nonzero."""
    cache = {}
    steps = 0
    key_work = "extra_merge_candidates" if improving else "greedy_candidates"
    while True:
        if improving:
            keys = [(r, c) for r in recips for c in range(st.fine[r].K) if st.count(r, c) >= 2]
        else:
            keys = [(r, c) for r in recips for c in range(st.fine[r].K) if st.count(r, c) > caps[r]]
        if not keys:
            return steps
        for k in keys:
            if k not in cache:
                cache[k] = st.merge_deltas(k[0], k[1], W)
                st.work[key_work] += int(cache[k][1].size)
        gmin = min(float(cache[k][3].min()) for k in keys)
        if improving and not gmin < -TOL:
            return steps
        tied = int(sum(int(np.sum(cache[k][3] <= gmin + TIE_TOL)) for k in keys))
        for k in keys:
            labs, ia, ib, delta, dD, dI, dI12 = cache[k]
            ok = delta <= gmin + TIE_TOL
            if improving:
                ok &= delta < -TOL
            hit = np.flatnonzero(ok)
            if hit.size:
                j = int(hit[0])
                r, c = k
                log.append(_merge_record(r, c, labs, ia, ib, delta, dD, dI, dI12, j,
                                         "extra" if improving else "to_cap", tied))
                st.apply_merge(r, int(labs[ia[j]]), int(labs[ib[j]]))
                st.work["extra_merges" if improving else "greedy_steps"] += 1
                steps += 1
                cache.pop(k, None)
                if W.w12 != 0.0:
                    o = st.other(r)
                    for kk in [kk for kk in cache if kk[0] == o]:
                        cache.pop(kk)
                break


def greedy(st, recips, W, caps, log):
    """Agglomerate to the caps, then apply objective-improving extra merges."""
    a = _merge_loop(st, recips, W, caps, log, improving=False)
    b = _merge_loop(st, recips, W, caps, log, improving=True)
    return a, b


def refine(st, recips, W, log, sweeps=SWEEPS):
    """Full sweep = single-fine-cell exchange pass over the recipients in ``recips`` (1 then 2; fine cells in
    increasing index; best target, ties to the lowest coarse label; accept only if delta < -TOL; a move never empties
    a cell) followed by an objective-improving merge pass. At most ``sweeps``; converged = a sweep with no accepted
    move and no merge. Returns (sweeps run, converged)."""
    for sweep in range(1, sweeps + 1):
        moved = 0
        for r in recips:
            G = st.G(r)
            for f in range(st.fine[r].F):
                res = st.move_deltas(r, f, W, G)
                if res is None:
                    continue
                B, delta, dD, dI, dI12 = res
                st.work["refine_candidates"] += int(B.size)
                j = int(np.argmin(delta))
                if delta[j] < -TOL:
                    a = int(st.lab[r][f])
                    st.apply_move(r, f, int(B[j]), G)
                    moved += 1
                    st.work["refine_moves"] += 1
                    log.append({"kind": "move", "sweep": sweep, "recipient": r, "fine_cell": f, "from": a,
                                "to": int(B[j]), "delta": float(delta[j]), "dD": float(dD[j]),
                                "dI_own": float(dI[j]), "dI12": None if dI12 is None else float(dI12[j])})
        mlog = []
        merged = _merge_loop(st, recips, W, None, mlog, improving=True)
        for x in mlog:
            x["sweep"] = sweep
        log.extend(mlog)
        if moved == 0 and merged == 0:
            return sweep, True
    return sweeps, False


def _optimise(st, recips, W, caps, lam, name):
    """Greedy to caps + extra merges, then refinement; records the stage."""
    merges = []
    to_cap, extra = greedy(st, recips, W, caps, merges)
    after_greedy = {**st.terms(), **F_values(st.terms(), lam)}
    g_val = st.value(W)
    moves = []
    sweeps, conv = refine(st, recips, W, moves)
    r_val = st.value(W)
    if r_val > g_val + 1e-12:
        raise AssertionError(f"{name}: refinement increased the stage objective ({g_val} -> {r_val})")
    return {"stage": name, "recipients": list(recips), "weights": W.__dict__, "merges_to_cap": to_cap,
            "extra_merges_greedy": extra, "merges": merges, "moves": moves, "sweeps": sweeps, "converged": conv,
            "positive_increment_merges": int(sum(1 for x in merges if x["increment"] > 0)),
            "after_greedy": after_greedy, "stage_objective_after_greedy": g_val, "stage_objective_after_refine": r_val,
            "after_refine": {**st.terms(), **F_values(st.terms(), lam)}}


# ----------------------------------------------------------------------------------------------- families
def _ident(fine):
    return np.arange(fine.F, dtype=np.int64)


def run_class(fine1, fine2, T):
    st = State(fine1, fine2, T, class_labels(fine1), class_labels(fine2))
    return st, {"stages": []}


def run_independent(fine1, fine2, T, m1, m2, lam, family):
    """FINE-TASK (lam None) and LOCAL: each recipient alone on D_r (+ lam I_r / 2); greedy both, then refine each."""
    st = State(fine1, fine2, T, _ident(fine1), _ident(fine2))
    caps = {1: m1, 2: m2}
    Ws = {r: (W_task(r) if family == "FINE-TASK" else W_local(lam, r)) for r in (1, 2)}
    stages = []
    for r in (1, 2):
        merges = []
        a, b = greedy(st, (r,), Ws[r], caps, merges)
        stages.append({"stage": f"r{r}", "recipients": [r], "weights": Ws[r].__dict__, "merges_to_cap": a,
                       "extra_merges_greedy": b, "merges": merges,
                       "positive_increment_merges": int(sum(1 for x in merges if x["increment"] > 0)),
                       "stage_objective_after_greedy": st.value(Ws[r])})
    after_greedy = {**st.terms(), **F_values(st.terms(), lam)}
    for r, s in zip((1, 2), stages):
        moves = []
        sw, conv = refine(st, (r,), Ws[r], moves)
        v = st.value(Ws[r])
        if v > s["stage_objective_after_greedy"] + 1e-12:
            raise AssertionError("refinement increased the stage objective")
        s.update({"moves": moves, "sweeps": sw, "converged": conv, "stage_objective_after_refine": v})
    return st, {"stages": stages, "after_greedy": after_greedy}


def run_sequential(order, fine1, fine2, T, m1, m2, lam, baseline_diagnostic=True):
    """SEQ-ab: stage 1 optimises recipient a under the actual F_joint with recipient b at its CLASS-ONLY release;
    frozen; stage 2 optimises recipient b under F_joint given the frozen map (b restarts from its fine cells)."""
    a, b = order
    fines = {1: fine1, 2: fine2}
    caps = {1: m1, 2: m2}
    W = W_joint(lam)
    lab = {a: _ident(fines[a]), b: class_labels(fines[b])}
    st1 = State(fine1, fine2, T, lab[1], lab[2])
    s1 = _optimise(st1, (a,), W, caps, lam, f"stage1_r{a}_vs_class_r{b}")
    frozen = st1.labels(a)
    t1 = st1.terms()
    correction = {"counterpart": "CLASS-ONLY", "stage1_F_joint_with_class_counterpart": F_values(t1, lam)["F_joint"],
                  f"D{a}": t1[f"D{a}"], f"I{a}": t1[f"I{a}"], f"D{b}_class": t1[f"D{b}"], f"I{b}_class": t1[f"I{b}"],
                  "I12_with_class_counterpart": t1["I12"], "conditional_I_S_decision_given_code": t1["I12"] - t1[f"I{a}"],
                  "old_surrogate_D_plus_1p5_lam_I_at_this_map": t1[f"D{a}"] + 1.5 * lam * t1[f"I{a}"]}
    if baseline_diagnostic:
        lab_old = {a: _ident(fines[a]), b: class_labels(fines[b])}
        st_old = State(fine1, fine2, T, lab_old[1], lab_old[2])
        Wold = W_old_seq_stage1(lam, a)
        _optimise(st_old, (a,), Wold, caps, lam, "old_stage1_diagnostic")
        to = st_old.terms()
        old_map = st_old.labels(a)
        correction["old_rule_stage1"] = {
            "objective": "D_a + 1.5 lam I_a (dpc; other recipient constant)",
            "same_map_as_corrected": bool(np.array_equal(old_map, frozen)),
            "F_joint_with_class_counterpart": F_values(to, lam)["F_joint"],
            f"D{a}": to[f"D{a}"], f"I{a}": to[f"I{a}"], "I12_with_class_counterpart": to["I12"],
            "corrected_minus_old_F_joint": correction["stage1_F_joint_with_class_counterpart"]
            - F_values(to, lam)["F_joint"]}
    lab2 = {a: frozen, b: _ident(fines[b])}
    st = State(fine1, fine2, T, lab2[1], lab2[2])
    s2 = _optimise(st, (b,), W, caps, lam, f"stage2_r{b}")
    if not np.array_equal(st.labels(a), frozen):
        raise AssertionError("first recipient revised after stage 1")
    return st, {"stages": [s1, s2], "order": list(order), "after_greedy": s2["after_greedy"],
                "baseline_correction": correction}


def _same_labels(x, y):
    return bool(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]))


def run_joint(fine1, fine2, T, m1, m2, lam, witness_labels, witness_source):
    """JOINT-GREEDY (F_joint greedy over both recipients, extra merges, joint refinement) plus joint refinement of
    the four witnesses; candidates = five refined starts + four unchanged witnesses; lowest from-scratch F_joint wins
    (ties: order JOINT-GREEDY, FINE-TASK, LOCAL, SEQ-12, SEQ-21; refined before unchanged)."""
    caps = {1: m1, 2: m2}
    W = W_joint(lam)
    starts, unchanged = {}, {}
    st = State(fine1, fine2, T, _ident(fine1), _ident(fine2))
    rec = _optimise(st, (1, 2), W, caps, lam, "JOINT-GREEDY")
    rec["initial_F_joint"] = rec["after_greedy"]["F_joint"]
    starts["JOINT-GREEDY"] = (st, rec)
    for fam in WITNESSES:
        l1, l2 = witness_labels[fam]
        ws = State(fine1, fine2, T, l1, l2)
        for r in (1, 2):
            for c in range(ws.fine[r].K):
                if ws.count(r, c) > caps[r]:
                    raise ValueError(f"witness {fam} exceeds cap m{r} on class {c}")
        ut = {**ws.terms(), **F_values(ws.terms(), lam)}
        unchanged[fam] = {"terms": ut, "labels": (ws.labels(1), ws.labels(2))}
        moves = []
        sw, conv = refine(ws, (1, 2), W, moves)
        if ws.value(W) > ut["F_joint"] + 1e-12:
            raise AssertionError("joint refinement increased F_joint")
        starts[fam] = (ws, {"stage": f"refine_{fam}", "initial_F_joint": ut["F_joint"], "moves": moves,
                            "sweeps": sw, "converged": conv, "source": witness_source[fam]})
    cand = [(n, "refined", starts[n][0].terms(), (starts[n][0].labels(1), starts[n][0].labels(2)))
            for n in JOINT_START_ORDER]
    cand += [(n, "unchanged", unchanged[n]["terms"], unchanged[n]["labels"]) for n in WITNESSES]
    vals = [F_values(t, lam)["F_joint"] for _, _, t, _ in cand]
    best = min(vals)
    k = vals.index(best)
    win_name, win_kind, _, win_labels = cand[k]
    final = starts[win_name][0] if win_kind == "refined" else State(fine1, fine2, T, *win_labels)
    fj = F_values(final.terms(), lam)["F_joint"]
    dominance = {}
    for n in WITNESSES:
        wv = unchanged[n]["terms"]["F_joint"]
        if not fj <= wv:
            raise AssertionError(f"JOINT final F_joint {fj} exceeds unchanged witness {n} ({wv})")
        dominance[n] = {"witness_F_joint": wv, "final_minus_witness": fj - wv}
    start_rec = {}
    unresolved = []
    work = {k2: 0 for k2 in final.work}
    for n in JOINT_START_ORDER:
        s, r = starts[n]
        for k2 in work:
            work[k2] += s.work[k2]
        fv = F_values(s.terms(), lam)["F_joint"]
        same = _same_labels((s.labels(1), s.labels(2)), win_labels)
        r = dict(r)
        r.update({"refined_F_joint": fv, "gap_to_winner": fv - best, "same_map_as_winner": same,
                  "work": dict(s.work)})
        start_rec[n] = r
        if not same and fv - best > 1e-12:
            unresolved.append({"start": n, "refined_F_joint": fv, "gap_to_winner": fv - best,
                               "converged_within_sweeps": r["converged"]})
    extra = {"starts": start_rec, "winner": {"start": win_name, "kind": win_kind, "F_joint": best},
             "candidates": [{"start": n, "kind": kd, "F_joint": v} for (n, kd, _, _), v in zip(cand, vals)],
             "witness_dominance": dominance, "unresolved_local_optima": unresolved,
             "starts_not_converged": [n for n in JOINT_START_ORDER if not start_rec[n]["converged"]],
             "after_greedy": starts["JOINT-GREEDY"][1]["after_greedy"]}
    final.work = work
    return final, extra


# ----------------------------------------------------------------------------------------------- receipts
def _brute_terms(pair, P1, d1, P2, d2, s):
    """Objective recomputed from the actual released tokens/decoded vectors of the fitting rows (no engine state)."""
    t1, q1, _ = RL.encode(pair.p1, P1, d1)
    t2, q2, _ = RL.encode(pair.p2, P2, d2)
    from dpc.partition import kl_rows
    return {"D1": float(np.mean(kl_rows(P1, q1))), "D2": float(np.mean(kl_rows(P2, q2))), "I1": mi_plugin(s, t1),
            "I2": mi_plugin(s, t2), "I12": mi_plugin(s, t1 * pair.p2.T + t2)}, t1, t2


def perm_null_mi(t1, t2, T2, s, n_perm=N_PERM, seed=PERM_SEED):
    """Permutation-null plug-in MI at the same codes (diagnostic only; never enters an objective or a selection).
    Fixed permutation list: rng = np.random.default_rng(seed); n_perm successive rng.permutation(N) of DEFENSE_FIT S."""
    rng = np.random.default_rng(seed)
    pc = t1 * T2 + t2
    out = {"I1": [], "I2": [], "I12": []}
    for _ in range(n_perm):
        sp = s[rng.permutation(s.shape[0])]
        out["I1"].append(mi_plugin(sp, t1))
        out["I2"].append(mi_plugin(sp, t2))
        out["I12"].append(mi_plugin(sp, pc))
    return {k: {"mean": float(np.mean(v)), "q95": float(np.quantile(v, 0.95)), "max": float(np.max(v))}
            for k, v in out.items()} | {"n_perm": n_perm, "seed": seed}


def sparsity_receipt(t1, t2, pair):
    T1, T2 = pair.p1.T, pair.p2.T
    pc = t1 * T2 + t2
    out = {}
    for name, codes, alpha in (("r1", t1, T1), ("r2", t2, T2), ("pair", pc, T1 * T2)):
        u, cnt = np.unique(codes, return_counts=True)
        p = cnt / cnt.sum()
        out[name] = {"alphabet": int(alpha), "occupied": int(u.size), "unseen_fraction": float(1 - u.size / alpha),
                     "singleton_cells": int(np.sum(cnt == 1)), "singleton_fraction_of_occupied":
                     float(np.mean(cnt == 1)), "lt5_cells": int(np.sum(cnt < 5)),
                     "entropy_nats": float(-(p * np.log(p)).sum()), "counts": [int(x) for x in cnt],
                     "codes": [int(x) for x in u]}
    return out


def fit_policy_pair(family, fine1, fine2, P1, d1, P2, d2, S_fit, m1, m2, lam, witnesses=None, meta=None,
                    baseline_diagnostic=None):
    """Fit one family on the fitting rows. Returns (qpc.PolicyPair, receipts). Witnesses (JOINT only): {family:
    qpc.PolicyPair or its dict} for FINE-TASK, LOCAL, SEQ-12, SEQ-21 on the same fine partitions, caps and lam;
    missing slots are recomputed deterministically (recorded per start as source)."""
    t0, c0 = time.perf_counter(), time.process_time()
    baseline_diagnostic = OLD_RULE_DIAGNOSTIC if baseline_diagnostic is None else bool(baseline_diagnostic)
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family}")
    if family in ("CLASS", "FINE-TASK"):
        if lam is not None:
            raise ValueError(f"{family} is task-only; lam must be None")
    elif lam is None or not np.isfinite(lam) or lam < 0:
        raise ValueError(f"{family} needs a finite lam >= 0")
    if family == "CLASS":
        m1 = m2 = 1
    m1, m2 = int(m1), int(m2)
    if m1 < 1 or m2 < 1:
        raise ValueError("caps must be positive")
    lam = None if lam is None else float(lam)
    P1 = KM.check_probs(P1, fine1.K, "P1")
    P2 = KM.check_probs(P2, fine2.K, "P2")
    d1 = KM.check_decisions(P1, d1, "d1")
    d2 = KM.check_decisions(P2, d2, "d2")
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("P1 and P2 must be aligned fitting rows")
    s = check_sex(S_fit, P1.shape[0])
    f1 = KM._check_deployment(fine1, P1, d1)
    f2 = KM._check_deployment(fine2, P2, d2)
    T = fine_table(f1, f2, s, fine1.F, fine2.F)
    if family == "CLASS":
        st, extra = run_class(fine1, fine2, T)
    elif family in ("FINE-TASK", "LOCAL"):
        st, extra = run_independent(fine1, fine2, T, m1, m2, lam, family)
    elif family in ("SEQ-12", "SEQ-21"):
        st, extra = run_sequential((1, 2) if family == "SEQ-12" else (2, 1), fine1, fine2, T, m1, m2, lam,
                                   baseline_diagnostic)
    else:
        wl, src = {}, {}
        fps = (fine1.fingerprint(), fine2.fingerprint())
        for fam in WITNESSES:
            w = (witnesses or {}).get(fam)
            if w is not None:
                pp = w if isinstance(w, RL.PolicyPair) else RL.PolicyPair.from_dict(w)
                if (pp.p1.fine.fingerprint(), pp.p2.fine.fingerprint()) != fps:
                    raise ValueError(f"witness {fam} was fitted on different fine partitions")
                if pp.family != fam or pp.config.get("m1") != m1 or pp.config.get("m2") != m2 or \
                        pp.config.get("lam") != (None if fam == "FINE-TASK" else lam):
                    raise ValueError(f"witness {fam} has a different family/caps/lam")
                src[fam] = "passed_in"
            else:
                pp, _ = fit_policy_pair(fam, fine1, fine2, P1, d1, P2, d2, s, m1, m2,
                                        None if fam == "FINE-TASK" else lam, meta=meta,
                                        baseline_diagnostic=False)
                src[fam] = "recomputed"
            wl[fam] = (labels_from_policy(pp.p1), labels_from_policy(pp.p2))
        extra_w = set(witnesses or {}) - set(WITNESSES)
        if extra_w:
            raise ValueError(f"unknown witness slots {sorted(extra_w)}")
        st, extra = run_joint(fine1, fine2, T, m1, m2, lam, wl, src)
    meta = dict(meta or {})
    pol1 = RL.make_policy(1, fine1, st.labels(1), family)
    pol2 = RL.make_policy(2, fine2, st.labels(2), family)
    pair = RL.make_pair(pol1, pol2, family, m1, m2, lam, meta)
    terms = st.terms()
    rows, t1, t2 = _brute_terms(pair, P1, d1, P2, d2, s)
    diff = max(abs(terms[k] - rows[k]) for k in ("D1", "D2", "I1", "I2", "I12"))
    if diff > 1e-9:
        raise AssertionError(f"aggregated objective differs from the row-level recomputation by {diff}")
    rec = {"family": family, "m1": m1, "m2": m2, "lam": lam, "eps": KM.EPS, "tol": TOL, "tie_tol": TIE_TOL,
           "sweeps_cap": SWEEPS, "final": {**terms, **F_values(terms, lam if lam is not None else 0.0)},
           "row_level_check": rows, "row_level_max_abs_diff": float(diff),
           "r1": RL.policy_receipt(pol1, t1), "r2": RL.policy_receipt(pol2, t2), "pair_fingerprint": pair.fingerprint(),
           "fine_fingerprints": [fine1.fingerprint(), fine2.fingerprint()], "work": dict(st.work),
           "sparsity_fit": sparsity_receipt(t1, t2, pair), "perm_null_mi_fit": perm_null_mi(t1, t2, pair.p2.T, s),
           "privacy_term": None if lam is None else {
               "lam_weighted_F_local_minus_F_task": lam * (terms["I1"] + terms["I2"]) / 2,
               "lam_weighted_F_joint_minus_F_task": lam * ((terms["I1"] + terms["I2"]) / 2 + terms["I12"]),
               "distortion": terms["D1"] + terms["D2"]}}
    if lam is None:
        rec["final"].pop("F_local")
        rec["final"].pop("F_joint")
    rec.update(extra)
    rec["summary"] = _summary(rec)
    rec["wall_seconds"] = time.perf_counter() - t0
    rec["cpu_seconds"] = time.process_time() - c0
    return pair, rec


def _summary(rec):
    if "starts" in rec:
        st = rec["starts"]
        g = st["JOINT-GREEDY"]
        return {"merges": len(g["merges"]), "positive_increment_merges": g["positive_increment_merges"],
                "accepted_moves": int(sum(sum(1 for x in v["moves"] if x["kind"] == "move") for v in st.values())),
                "refine_merges": int(sum(sum(1 for x in v["moves"] if x["kind"] == "extra") for v in st.values())),
                "sweeps": {k: int(v["sweeps"]) for k, v in st.items()},
                "converged": bool(all(v["converged"] for v in st.values())), "starts": len(st),
                "winner": rec["winner"]["start"] + ":" + rec["winner"]["kind"],
                "unresolved_local_optima": len(rec["unresolved_local_optima"])}
    stages = rec.get("stages", [])
    return {"merges": int(sum(len(s["merges"]) for s in stages)),
            "positive_increment_merges": int(sum(s["positive_increment_merges"] for s in stages)),
            "extra_merges": int(sum(sum(1 for x in s["merges"] if x["kind"] == "extra") for s in stages)
                                + sum(sum(1 for x in s.get("moves", []) if x["kind"] == "extra") for s in stages)),
            "accepted_moves": int(sum(sum(1 for x in s.get("moves", []) if x["kind"] == "move") for s in stages)),
            "sweeps": {s["stage"]: int(s.get("sweeps", 0)) for s in stages},
            "converged": bool(all(s.get("converged", True) for s in stages)), "starts": None, "winner": None}


def config_id(teacher, family, m1, m2, lam=None):
    if family == "CLASS":
        return f"{teacher}|CLASS|i1o1"
    return f"{teacher}|{family}|i{int(m1)}o{int(m2)}" + (f"|l{lam:g}" if family in PRIVACY else "")


# ----------------------------------------------------------------------------------------------- runner unit
def fit_unit(family, fine_dict, T, tr, S_fit, m1, m2, lam, meta, witnesses=None):
    """One Stage B mapping-pair unit. fine_dict = the fine.json content of qpc.partition.fine_unit; T over ALL rows;
    tr = DEFENSE_FIT indices; S_fit = DEFENSE_FIT SEX (binary, aligned with tr). meta: teacher, seed, config,
    teacher_model_sha256, feature_names_sha256. witnesses (JOINT): {family: policy.json dict}. Files: policy.json,
    release.npz (ALL rows)."""
    from qpc.partition import load_fine
    from qpc.stagea import _bind_meta, encode_all, pair_record
    meta = _bind_meta(meta)
    tr = np.asarray(tr)
    fine1, fine2 = load_fine(fine_dict)
    cid = config_id(meta.get("teacher", "U"), family, m1, m2, lam)
    if meta.get("config") not in (None, cid):
        raise ValueError(f"meta config {meta.get('config')} differs from {cid}")
    meta["config"] = cid
    pair, rec = fit_policy_pair(family, fine1, fine2, T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr], S_fit,
                                m1, m2, lam, witnesses, meta)
    RL.check_bound(pair)
    out = encode_all(pair, T)
    rec["pair_record"] = pair_record(pair, out, tr)
    rec["config"] = cid
    KM.json_safe(rec)
    s = json.dumps(pair.to_dict(), allow_nan=False)
    return rec, {"policy.json": lambda p: Path(p).write_text(s), "release.npz": out}


def find_aliases(pairs: dict):
    """{config_id: PolicyPair} -> per pair / recipient: first (sorted) config with an identical fingerprint."""
    out = {"pair": {}, "r1": {}, "r2": {}}
    seen = {"pair": {}, "r1": {}, "r2": {}}
    for cid in sorted(pairs):
        pp = pairs[cid]
        for key, fp in (("pair", pp.fingerprint()), ("r1", pp.p1.fingerprint()), ("r2", pp.p2.fingerprint())):
            seen[key].setdefault(fp, cid)
            out[key][cid] = seen[key][fp]
    return out
