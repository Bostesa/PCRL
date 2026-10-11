"""Class-preserving coarse compression of fine teacher-score partitions: objectives, greedy agglomeration, exchange
refinement and the seven learner families (CLASS-ONLY, FINE-TASK, DIRECT-TASK, LOCAL, SEQ-12, SEQ-21, JOINT).

Objective on fitting rows (natural logs), N fitting rows, coarse cells c of recipient i:
    D_i  = (1/N) sum_c [A_c - S_c . log q_c],  q_c = smooth(S_c / n_c, class_c)   (cells with n_c = 0 contribute 0)
    I_i  = I(S; C_i), I_12 = I(S; C_1, C_2): plug-in mutual information of the exact fitting contingency tables,
           0 log 0 = 0, empty cells contribute 0, no smoothing of any table. Computed as
           (1/N) [ sum_cols phi(col) - sum_s n_s log n_s + N log N ],  phi(col) = sum_s n_sc log n_sc - n_c log n_c.
    F_task  = D1 + D2
    F_local = D1 + D2 + lam (I1 + I2) / 2
    F_joint = D1 + D2 + lam ((I1 + I2) / 2 + I12)
The fine-level table n(s, f1, f2) is built once from the fitting rows (int64); coarse tables are exact sums of it.

Greedy: every step evaluates ALL eligible merges (same recipient, same predicted class, class above the cap m) of
the recipients being optimised, computes the exact objective increment from sufficient statistics and count tables,
and applies the smallest; candidates within TIE_TOL = 1e-12 of the smallest increment are tied and the first in
lexicographic order (recipient, class, cell a, cell b) wins (a coarse cell is labelled by its lowest member fine
index; merged cells keep the lower label). Stops when every class has <= m coarse cells; positive increments are
allowed (recorded).
Refinement: sweeps over the recipients being optimised (recipient 1 then 2), fine cells in increasing index; for each
fine cell whose coarse cell has another member, evaluate moving it to every other coarse cell of its class, take the
best (ties -> lowest coarse label) and accept it only if Delta F < -TOL, TOL = 1e-12; at most SWEEPS = 5 full sweeps;
stop after a sweep without accepted moves. Coarse statistics are re-accumulated from member fine cells (increasing
fine index) after every accepted change, and every reported objective is recomputed from scratch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dpc.partition import (EPS, FinePartition, assign_fine, check_decisions, check_probs, fit_fine, kl_rows,
                           smooth, validate_fine_against_rows)
from dpc.release import Policy, PolicyPair, canonical_tokens, encode

TOL = 1e-12
TIE_TOL = 1e-12
SWEEPS = 5
TASK_FAMILIES = ("FINE-TASK", "DIRECT-TASK")
PRIVACY_FAMILIES = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
WITNESS_FAMILIES = ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
JOINT_START_ORDER = ("JOINT-GREEDY", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")


# ----------------------------------------------------------------------------------------------- arithmetic
def xlogx(x):
    x = np.asarray(x, dtype=np.float64)
    return x * np.log(np.where(x > 0, x, 1.0))


def phi(T):
    """Column term of plug-in MI for a count table T with the sensitive value on axis 0."""
    return xlogx(T).sum(0) - xlogx(T.sum(0))


def mi_from_table(T):
    """Plug-in I(S; C) in nats from an integer table T (2, ...) via the phi decomposition."""
    T = np.asarray(T)
    N = int(T.sum())
    if N == 0:
        return 0.0
    ns = T.reshape(T.shape[0], -1).sum(1)
    return float((phi(T).sum() - xlogx(ns).sum() + xlogx(N)) / N)


def mi_plugin(s, c):
    """Direct plug-in I(S; C) = sum (n/N) log(n N / (n_s n_c)) over nonempty cells, from per-row labels (brute force)."""
    s = np.asarray(s, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    N = s.shape[0]
    _, ci = np.unique(c, return_inverse=True)
    C = int(ci.max()) + 1 if N else 0
    tab = np.bincount(s * C + ci, minlength=2 * C).reshape(2, C).astype(np.float64)
    ns = tab.sum(1, keepdims=True)
    nc = tab.sum(0, keepdims=True)
    m = tab > 0
    return float(np.sum(np.where(m, tab / N * (np.log(np.where(m, tab, 1)) + np.log(N) - np.log(np.where(ns > 0, ns, 1))
                                                 - np.log(np.where(nc > 0, nc, 1))), 0.0)))


def h_cost(S, n, c, K, eps=EPS):
    """-S . log smooth(S/n, c) per cell (0 for n = 0). S (M, K), n (M,), c (M,) or scalar."""
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    n = np.atleast_1d(np.asarray(n))
    c = np.broadcast_to(np.asarray(c, dtype=np.int64), (S.shape[0],))
    out = np.zeros(S.shape[0])
    pos = n > 0
    if pos.any():
        Q = smooth(S[pos] / n[pos, None], c[pos], eps=eps, check=False)
        out[pos] = -(S[pos] * np.log(Q)).sum(1)
    return out


def check_sex(S_fit, N):
    s = np.asarray(S_fit)
    if s.shape != (N,):
        raise ValueError(f"S_fit must have shape ({N},)")
    if not np.all((s == 0) | (s == 1)):
        raise ValueError("S_fit must be binary {0, 1}")
    return s.astype(np.int64)


def fine_table(f1, f2, s, F1, F2):
    idx = (s * F1 + f1) * F2 + f2
    return np.bincount(idx, minlength=2 * F1 * F2).reshape(2, F1, F2).astype(np.int64)


def mi_terms_from_labels(Tfine, lab1, lab2):
    """(I1, I2, I12) for ARBITRARY integer labels of the fine cells (e.g. a constant map), from the fine table."""
    L1, L2 = int(np.max(lab1)) + 1, int(np.max(lab2)) + 1
    T = np.zeros((2, L1, L2), dtype=np.int64)
    np.add.at(T, (slice(None), np.asarray(lab1)[:, None], np.asarray(lab2)[None, :]), Tfine)
    return mi_from_table(T.sum(2)), mi_from_table(T.sum(1)), mi_from_table(T)


@dataclass(frozen=True)
class Weights:
    """F = wD1 D1 + wD2 D2 + wI1 I1 + wI2 I2 + w12 I12 (only differences matter inside a stage)."""
    wD1: float
    wD2: float
    wI1: float
    wI2: float
    w12: float

    def wD(self, r):
        return self.wD1 if r == 1 else self.wD2

    def wI(self, r):
        return self.wI1 if r == 1 else self.wI2


def W_task():
    return Weights(1.0, 1.0, 0.0, 0.0, 0.0)


def W_local(lam, r=None):
    if r is None:
        return Weights(1.0, 1.0, lam / 2, lam / 2, 0.0)
    return Weights(1.0 if r == 1 else 0.0, 1.0 if r == 2 else 0.0, lam / 2 if r == 1 else 0.0,
                   lam / 2 if r == 2 else 0.0, 0.0)


def W_joint(lam):
    return Weights(1.0, 1.0, lam / 2, lam / 2, lam)


def W_seq_stage1(lam, r):
    """F_joint with the other recipient constant: D_r + 1.5 lam I_r (+ constants)."""
    return Weights(1.0 if r == 1 else 0.0, 1.0 if r == 2 else 0.0, 1.5 * lam if r == 1 else 0.0,
                   1.5 * lam if r == 2 else 0.0, 0.0)


def F_values(t, lam):
    out = {"F_task": t["D1"] + t["D2"]}
    if lam is not None:
        out["F_local"] = t["D1"] + t["D2"] + lam * (t["I1"] + t["I2"]) / 2
        out["F_joint"] = t["D1"] + t["D2"] + lam * ((t["I1"] + t["I2"]) / 2 + t["I12"])
    return out


# ----------------------------------------------------------------------------------------------- state engine
class _Rec:
    """One recipient's coarse state: label per fine cell (label = some member fine index; groups never cross classes)."""

    def __init__(self, fine: FinePartition, lab):
        self.fine = fine
        self.K = fine.K
        self.F = fine.F
        self.cls = fine.cell_class
        self.lab = np.asarray(lab, dtype=np.int64).copy()
        self.n = np.zeros(self.F, dtype=np.int64)
        self.S = np.zeros((self.F, self.K))
        self.h = np.zeros(self.F)
        self.members = {}
        for f in range(self.F):
            self.members.setdefault(int(self.lab[f]), []).append(f)
        for l, mem in self.members.items():
            if l not in mem:
                raise ValueError("a label must be one of its own members")
            if len(set(self.cls[mem].tolist())) != 1:
                raise ValueError("a coarse cell may not mix predicted classes")
            self._recompute(l)

    def _recompute(self, l):
        mem = self.members[l]
        S = np.zeros(self.K)
        n = 0
        for f in mem:                       # increasing fine index (members kept sorted)
            S = S + self.fine.S[f]
            n += int(self.fine.n[f])
        self.S[l], self.n[l] = S, n
        self.h[l] = h_cost(S[None], np.array([n]), self.cls[l], self.K)[0]

    def class_labels(self, c):
        return np.array(sorted(l for l in self.members if self.cls[l] == c), dtype=np.int64)

    def coarse_count(self, c):
        return sum(1 for l in self.members if self.cls[l] == c)


class State:
    """Both recipients' coarse states plus the exact pair table by labels."""

    def __init__(self, fine1, fine2, Tfine, lab1, lab2):
        self.R = {1: _Rec(fine1, lab1), 2: _Rec(fine2, lab2)}
        self.Tfine = np.asarray(Tfine, dtype=np.int64)
        self.N = int(self.Tfine.sum())
        self.ns = self.Tfine.sum((1, 2))
        self.const_mi = float(-xlogx(self.ns).sum() + xlogx(self.N))
        self.A = {r: float(np.sum(self.R[r].fine.A)) for r in (1, 2)}
        self.Tfine_r = {1: self.Tfine.sum(2), 2: self.Tfine.sum(1)}
        self.rebuild_tables()
        self.work = {"greedy_candidates": 0, "greedy_steps": 0, "refine_candidates": 0, "refine_moves": 0}

    # tables ------------------------------------------------------------------------------------------
    def _onehot(self, r):
        R = self.R[r]
        M = np.zeros((R.F, R.F))
        M[np.arange(R.F), R.lab] = 1.0
        return M

    def rebuild_tables(self):
        M1, M2 = self._onehot(1), self._onehot(2)
        T = self.Tfine.astype(np.float64)
        T12 = np.einsum("sab,ai,bj->sij", T, M1, M2, optimize=True)
        self.T12 = np.rint(T12).astype(np.int64)
        self.Tr = {1: self.T12.sum(2), 2: self.T12.sum(1)}

    def T12_oriented(self, r):
        return self.T12 if r == 1 else self.T12.transpose(0, 2, 1)

    def G(self, r):
        """Fine rows of recipient r aggregated by the other recipient's current labels: (2, F_r, F_other)."""
        o = 2 if r == 1 else 1
        Tf = self.Tfine if r == 1 else self.Tfine.transpose(0, 2, 1)
        return np.rint(Tf.astype(np.float64) @ self._onehot(o)).astype(np.int64)

    # objective ---------------------------------------------------------------------------------------
    def terms(self):
        t = {}
        for r in (1, 2):
            R = self.R[r]
            labs = sorted(R.members)
            t[f"D{r}"] = float((self.A[r] + float(np.sum(R.h[labs]))) / self.N)
            t[f"I{r}"] = float((phi(self.Tr[r]).sum() + self.const_mi) / self.N)
        t["I12"] = float((phi(self.T12).sum() + self.const_mi) / self.N)
        return t

    def value(self, W: Weights):
        t = self.terms()
        return W.wD1 * t["D1"] + W.wD2 * t["D2"] + W.wI1 * t["I1"] + W.wI2 * t["I2"] + W.w12 * t["I12"]

    def labels(self, r):
        return self.R[r].lab.copy()

    # merges ------------------------------------------------------------------------------------------
    def merge_deltas(self, r, c, W: Weights):
        """All merges (a < b) of class c on recipient r: (labs, ia, ib, delta, dD, dI, dI12) in lexicographic order."""
        R = self.R[r]
        labs = R.class_labels(c)
        L = labs.size
        ia, ib = np.triu_indices(L, 1)
        a, b = labs[ia], labs[ib]
        N = self.N
        dD = (h_cost(R.S[a] + R.S[b], R.n[a] + R.n[b], c, R.K) - R.h[a] - R.h[b]) / N
        Tr = self.Tr[r]
        ph = phi(Tr[:, labs])
        dI = (phi(Tr[:, a] + Tr[:, b]) - ph[ia] - ph[ib]) / N
        delta = W.wD(r) * dD + W.wI(r) * dI
        dI12 = None
        if W.w12 != 0.0:
            o = 2 if r == 1 else 1
            alive_o = np.array(sorted(self.R[o].members), dtype=np.int64)
            X = self.T12_oriented(r)[:, labs][:, :, alive_o]              # (2, L, Lo)
            rowphi = phi(X).sum(-1)                                        # (L,)
            dI12 = (phi(X[:, ia] + X[:, ib]).sum(-1) - rowphi[ia] - rowphi[ib]) / N
            delta = delta + W.w12 * dI12
        self.work["greedy_candidates"] += int(ia.size)
        return labs, ia, ib, delta, dD, dI, dI12

    def apply_merge(self, r, a, b):
        R = self.R[r]
        assert a < b and R.cls[a] == R.cls[b]
        mem = sorted(R.members[a] + R.members.pop(b))
        R.members[a] = mem
        R.lab[mem] = a
        R._recompute(a)
        R.n[b], R.S[b], R.h[b] = 0, 0.0, 0.0
        if r == 1:
            self.T12[:, a, :] += self.T12[:, b, :]
            self.T12[:, b, :] = 0
        else:
            self.T12[:, :, a] += self.T12[:, :, b]
            self.T12[:, :, b] = 0
        self.Tr[r][:, a] += self.Tr[r][:, b]
        self.Tr[r][:, b] = 0

    # moves -------------------------------------------------------------------------------------------
    def move_deltas(self, r, f, W: Weights, G=None):
        """Moves of fine cell f to every other coarse cell of its class: (targets, delta, dD, dI, dI12) or None."""
        R = self.R[r]
        a = int(R.lab[f])
        if len(R.members[a]) <= 1:
            return None
        c = int(R.cls[f])
        B = np.array([l for l in R.class_labels(c) if l != a], dtype=np.int64)
        if B.size == 0:
            return None
        N = self.N
        Sa = np.zeros(R.K)
        for g in R.members[a]:
            if g != f:
                Sa = Sa + R.fine.S[g]
        na = int(R.n[a] - R.fine.n[f])
        ha = h_cost(Sa[None], np.array([na]), c, R.K)[0]
        hb = h_cost(R.S[B] + R.fine.S[f], R.n[B] + R.fine.n[f], c, R.K)
        dD = (ha + hb - R.h[a] - R.h[B]) / N
        Tr = self.Tr[r]
        tf = self.Tfine_r[r][:, f]
        dI = (phi((Tr[:, a] - tf)[:, None]) + phi(Tr[:, B] + tf[:, None]) - phi(Tr[:, [a]]) - phi(Tr[:, B])) / N
        delta = W.wD(r) * dD + W.wI(r) * dI
        dI12 = None
        if W.w12 != 0.0:
            o = 2 if r == 1 else 1
            alive_o = np.array(sorted(self.R[o].members), dtype=np.int64)
            g = G[:, f, alive_o]                                          # (2, Lo)
            X = self.T12_oriented(r)
            Xa = X[:, a, alive_o]
            XB = X[:, B][:, :, alive_o]                                    # (2, nB, Lo)
            dI12 = (phi(Xa - g).sum() + phi(XB + g[:, None, :]).sum(-1) - phi(Xa).sum() - phi(XB).sum(-1)) / N
            delta = delta + W.w12 * dI12
        self.work["refine_candidates"] += int(B.size)
        return B, delta, dD, dI, dI12

    def apply_move(self, r, f, b, G=None):
        R = self.R[r]
        a = int(R.lab[f])
        assert a != b and R.cls[a] == R.cls[b] and len(R.members[a]) > 1
        R.members[a] = [g for g in R.members[a] if g != f]
        R.members[b] = sorted(R.members[b] + [f])
        R.lab[f] = b
        R._recompute(a)
        R._recompute(b)
        tf = self.Tfine_r[r][:, f]
        self.Tr[r][:, a] -= tf
        self.Tr[r][:, b] += tf
        if G is None:
            G = self.G(r)
        g = G[:, f, :]
        if r == 1:
            self.T12[:, a, :] -= g
            self.T12[:, b, :] += g
        else:
            self.T12[:, :, a] -= g
            self.T12[:, :, b] += g


# ----------------------------------------------------------------------------------------------- algorithms
def greedy(state: State, recips, W: Weights, m: int, log: list):
    """Agglomerate until every class of every recipient in ``recips`` has <= m coarse cells."""
    while True:
        evals = []
        for r in recips:
            for c in range(state.R[r].K):
                if state.R[r].coarse_count(c) > m:
                    evals.append((r, c) + state.merge_deltas(r, c, W))
        if not evals:
            return
        gmin = min(float(e[5].min()) for e in evals)
        for r, c, labs, ia, ib, delta, dD, dI, dI12 in evals:      # (recipient, class) order
            hit = np.flatnonzero(delta <= gmin + TIE_TOL)
            if hit.size:
                j = int(hit[0])                                     # lexicographic (a, b) within the class
                a, b = int(labs[ia[j]]), int(labs[ib[j]])
                log.append({"recipient": r, "class": c, "a": a, "b": b, "increment": float(delta[j]),
                            "dD": float(dD[j]), "dI_own": float(dI[j]),
                            "dI12": None if dI12 is None else float(dI12[j]),
                            "tied_candidates": int(np.sum(np.concatenate([e[5] for e in evals]) <= gmin + TIE_TOL))})
                state.apply_merge(r, a, b)
                state.work["greedy_steps"] += 1
                break


def refine(state: State, recips, W: Weights, log: list, sweeps=SWEEPS):
    """Single-cell exchange refinement; returns (sweeps run, converged = a sweep had no accepted move)."""
    done = 0
    for sweep in range(1, sweeps + 1):
        moved = 0
        for r in recips:
            G = state.G(r) if W.w12 != 0.0 else None
            for f in range(state.R[r].F):
                res = state.move_deltas(r, f, W, G)
                if res is None:
                    continue
                B, delta, dD, dI, dI12 = res
                j = int(np.argmin(delta))                            # ties -> lowest coarse label
                if delta[j] < -TOL:
                    a = int(state.R[r].lab[f])
                    b = int(B[j])
                    state.apply_move(r, f, b, G if G is not None else state.G(r))
                    moved += 1
                    state.work["refine_moves"] += 1
                    log.append({"sweep": sweep, "recipient": r, "fine_cell": f, "from": a, "to": b,
                                "delta": float(delta[j]), "dD": float(dD[j]), "dI_own": float(dI[j]),
                                "dI12": None if dI12 is None else float(dI12[j])})
        done = sweep
        if moved == 0:
            return done, True
    return done, False


# ----------------------------------------------------------------------------------------------- receipts
def _policy_from_state(state, r, family, meta):
    fine = state.R[r].fine
    return Policy(recipient=r, fine=fine, cell_token=canonical_tokens(fine, state.labels(r)), family=family,
                  meta=dict(meta))


def _labels_from_policy(pol: Policy):
    """Label per fine cell = lowest member fine index of its token."""
    first = {}
    for f in range(pol.fine.F):
        first.setdefault(int(pol.cell_token[f]), f)
    return np.array([first[int(t)] for t in pol.cell_token], dtype=np.int64)


def evaluate_rows(pair: PolicyPair, P1, d1, P2, d2, S_fit, lam=None):
    """Brute-force objective from the actual released tokens/prototypes of the given rows (no engine state)."""
    t1, q1, _ = encode(pair.p1, P1, d1)
    t2, q2, _ = encode(pair.p2, P2, d2)
    s = check_sex(S_fit, t1.shape[0])
    t = {"D1": float(np.mean(kl_rows(check_probs(P1), q1))), "D2": float(np.mean(kl_rows(check_probs(P2), q2))),
         "I1": mi_plugin(s, t1), "I2": mi_plugin(s, t2), "I12": mi_plugin(s, t1 * pair.p2.T + t2)}
    t.update(F_values(t, lam))
    return t


def _policy_receipt(pol: Policy):
    return {"tokens": pol.T, "tokens_per_class": pol.tokens_per_class(),
            "effective_states_per_class": pol.effective_states_per_class(),
            "effective_states": int(sum(pol.effective_states_per_class())),
            "fallback_tokens": [int(t) for t in np.flatnonzero(pol.token_fallback)],
            "max_abs_unsmoothed_vs_smoothed": pol.max_abs_unsmoothed_vs_smoothed(),
            "fingerprint": pol.fingerprint(), "assignment_partition_fingerprint": pol.fine.fingerprint()}


def _finish(state, pair_family, m, lam, meta, P1, d1, P2, d2, S_fit, extra, t0, c0):
    p1 = _policy_from_state(state, 1, pair_family, meta)
    p2 = _policy_from_state(state, 2, pair_family, meta)
    pair = PolicyPair(p1, p2, pair_family, dict(meta))
    terms = state.terms()
    rows = evaluate_rows(pair, P1, d1, P2, d2, S_fit, lam)
    diff = max(abs(terms[k] - rows[k]) for k in ("D1", "D2", "I1", "I2", "I12"))
    if diff > 1e-9:
        raise AssertionError(f"aggregated objective differs from row-level recomputation by {diff}")
    rec = {"family": pair_family, "m": m, "lam": lam, "eps": EPS, "tol": TOL, "tie_tol": TIE_TOL, "sweeps_cap": SWEEPS,
           "final": {**terms, **F_values(terms, lam)}, "row_level_check": rows,
           "row_level_max_abs_diff": float(diff), "r1": _policy_receipt(p1), "r2": _policy_receipt(p2),
           "pair_fingerprint": pair.fingerprint(), "work": dict(state.work)}
    rec.update(extra)
    rec.setdefault("after_greedy", None)
    rec["summary"] = _summary(rec)
    rec["wall_seconds"] = time.perf_counter() - t0
    rec["cpu_seconds"] = time.process_time() - c0
    return pair, rec


def _summary(rec):
    """Uniform optimisation summary: merges / positive-increment merges / accepted moves / sweeps / convergence."""
    if "stages" in rec:
        st = rec["stages"]
        return {"merges": int(sum(len(s["merges"]) for s in st)),
                "positive_increment_merges": int(sum(s["positive_increment_merges"] for s in st)),
                "accepted_moves": int(sum(len(s["moves"]) for s in st)),
                "sweeps": {s["stage"]: int(s["sweeps"]) for s in st},
                "converged": bool(all(s["converged"] for s in st)), "starts": None, "winner": None}
    if "starts" in rec:
        st = rec["starts"]
        g = st["JOINT-GREEDY"]
        return {"merges": len(g["merges"]), "positive_increment_merges": g["positive_increment_merges"],
                "accepted_moves": int(sum(len(v["refine_moves"]) for v in st.values())),
                "sweeps": {k: int(v["sweeps"]) for k, v in st.items()},
                "converged": bool(all(v["converged"] for v in st.values())), "starts": len(st),
                "winner": rec["winner"]["start"] + ":" + rec["winner"]["kind"]}
    return {"merges": 0, "positive_increment_merges": 0, "accepted_moves": 0, "sweeps": {}, "converged": True,
            "starts": None, "winner": None}


# ----------------------------------------------------------------------------------------------- families
def _prep(fine1, fine2, P1, d1, P2, d2, S_fit):
    P1 = check_probs(P1, fine1.K, "P1")
    P2 = check_probs(P2, fine2.K, "P2")
    d1 = check_decisions(P1, d1, "d1")
    d2 = check_decisions(P2, d2, "d2")
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("P1 and P2 must be aligned fitting rows")
    s = check_sex(S_fit, P1.shape[0])
    f1 = validate_fine_against_rows(fine1, P1, d1)
    f2 = validate_fine_against_rows(fine2, P2, d2)
    T = fine_table(f1, f2, s, fine1.F, fine2.F)
    return P1, d1, P2, d2, s, T


def _fine_labels(fine):
    return np.arange(fine.F, dtype=np.int64)


def _class_labels(fine):
    first = {}
    for f in range(fine.F):
        first.setdefault(int(fine.cell_class[f]), f)
    return np.array([first[int(c)] for c in fine.cell_class], dtype=np.int64)


def _check_m(m):
    if int(m) != m or m < 1:
        raise ValueError("m must be a positive integer")
    return int(m)


def fit_class_only(fine1, fine2, P1, d1, P2, d2, S_fit, lam=None, meta=None):
    """m = 1: one token per predicted class (fallback classes keep their fallback token)."""
    t0, c0 = time.perf_counter(), time.process_time()
    P1, d1, P2, d2, s, T = _prep(fine1, fine2, P1, d1, P2, d2, S_fit)
    st = State(fine1, fine2, T, _class_labels(fine1), _class_labels(fine2))
    meta = {"family": "CLASS-ONLY", "m": 1, "lam": None, **(meta or {})}
    return _finish(st, "CLASS-ONLY", 1, lam, meta, P1, d1, P2, d2, S_fit, {}, t0, c0)


def _stage_record(name, st, W, lam, merges, moves, sweeps, conv, before):
    after_greedy = before
    return {"stage": name, "weights": W.__dict__, "merges": merges, "moves": moves, "sweeps": sweeps,
            "converged": conv, "after_greedy": after_greedy, "after_refine": {**st.terms(), **F_values(st.terms(), lam)}}


def _run_recipients(st, plan, m, lam):
    """plan: list of (stage name, recipients tuple, Weights). Greedy then refinement per stage."""
    stages = []
    for name, recips, W in plan:
        merges, moves = [], []
        greedy(st, recips, W, m, merges)
        g_terms = {**st.terms(), **F_values(st.terms(), lam)}
        g_val = st.value(W)
        sweeps, conv = refine(st, recips, W, moves)
        r_val = st.value(W)
        if r_val > g_val + 1e-12:
            raise AssertionError(f"refinement increased the stage objective ({g_val} -> {r_val})")
        rec = _stage_record(name, st, W, lam, merges, moves, sweeps, conv, g_terms)
        rec.update({"stage_objective_after_greedy": g_val, "stage_objective_after_refine": r_val,
                    "positive_increment_merges": int(sum(1 for x in merges if x["increment"] > 0))})
        stages.append(rec)
    return stages


def _run_independent(st, Ws, m, lam):
    """Separable objectives (FINE-TASK, LOCAL): greedy on recipient 1 then 2, snapshot, then refinement of recipient 1
    then 2 (each its own <= SWEEPS sweeps). For a separable objective this equals optimising each alone."""
    stages = {}
    for r in (1, 2):
        merges = []
        greedy(st, (r,), Ws[r], m, merges)
        stages[r] = {"stage": f"r{r}", "weights": Ws[r].__dict__, "merges": merges,
                     "positive_increment_merges": int(sum(1 for x in merges if x["increment"] > 0)),
                     "stage_objective_after_greedy": st.value(Ws[r])}
    after_greedy = {**st.terms(), **F_values(st.terms(), lam)}
    for r in (1, 2):
        moves = []
        sweeps, conv = refine(st, (r,), Ws[r], moves)
        v = st.value(Ws[r])
        if v > stages[r]["stage_objective_after_greedy"] + 1e-12:
            raise AssertionError("refinement increased the stage objective")
        stages[r].update({"moves": moves, "sweeps": sweeps, "converged": conv, "stage_objective_after_refine": v})
    return [stages[1], stages[2]], after_greedy


def fit_fine_task(fine1, fine2, P1, d1, P2, d2, S_fit, m, meta=None):
    t0, c0 = time.perf_counter(), time.process_time()
    m = _check_m(m)
    P1, d1, P2, d2, s, T = _prep(fine1, fine2, P1, d1, P2, d2, S_fit)
    st = State(fine1, fine2, T, _fine_labels(fine1), _fine_labels(fine2))
    stages, ag = _run_independent(st, {1: Weights(1, 0, 0, 0, 0), 2: Weights(0, 1, 0, 0, 0)}, m, None)
    meta = {"family": "FINE-TASK", "m": m, "lam": None, **(meta or {})}
    return _finish(st, "FINE-TASK", m, None, meta, P1, d1, P2, d2, S_fit, {"stages": stages, "after_greedy": ag},
                   t0, c0)


def fit_local(fine1, fine2, P1, d1, P2, d2, S_fit, m, lam, meta=None):
    t0, c0 = time.perf_counter(), time.process_time()
    m = _check_m(m)
    lam = float(lam)
    P1, d1, P2, d2, s, T = _prep(fine1, fine2, P1, d1, P2, d2, S_fit)
    st = State(fine1, fine2, T, _fine_labels(fine1), _fine_labels(fine2))
    stages, ag = _run_independent(st, {1: W_local(lam, 1), 2: W_local(lam, 2)}, m, lam)
    meta = {"family": "LOCAL", "m": m, "lam": lam, **(meta or {})}
    return _finish(st, "LOCAL", m, lam, meta, P1, d1, P2, d2, S_fit, {"stages": stages, "after_greedy": ag}, t0, c0)


def fit_sequential(order, fine1, fine2, P1, d1, P2, d2, S_fit, m, lam, meta=None):
    """order = (1, 2) for SEQ-12 or (2, 1) for SEQ-21. Stage 1: first recipient alone on D_r + 1.5 lam I_r (F_joint
    with the other recipient constant); frozen. Stage 2: second recipient under the full F_joint given the frozen map.
    Only the current recipient is refined; the first is never revised."""
    t0, c0 = time.perf_counter(), time.process_time()
    m = _check_m(m)
    lam = float(lam)
    a, b = order
    fam = f"SEQ-{a}{b}"
    P1, d1, P2, d2, s, T = _prep(fine1, fine2, P1, d1, P2, d2, S_fit)
    st = State(fine1, fine2, T, _fine_labels(fine1), _fine_labels(fine2))
    stages = _run_recipients(st, [("stage1_r%d" % a, (a,), W_seq_stage1(lam, a))], m, lam)
    frozen = st.labels(a)
    # stage-one coefficient identity: F_joint(C_a, constant C_b) - D_b(const) == D_a + 1.5 lam I_a
    t = st.terms()
    const = np.zeros(st.R[b].F, dtype=np.int64)
    lab1, lab2 = (frozen, const) if a == 1 else (const, frozen)
    I1c, I2c, I12c = mi_terms_from_labels(T, lab1, lab2)
    Ia_c = I1c if a == 1 else I2c
    Ib_c = I2c if a == 1 else I1c
    fj_const_minus_db = t[f"D{a}"] + lam * ((Ia_c + Ib_c) / 2 + I12c)
    stage1 = t[f"D{a}"] + 1.5 * lam * t[f"I{a}"]
    ident = abs(fj_const_minus_db - stage1)
    if ident > 1e-12 or abs(Ib_c) > 1e-12 or abs(I12c - t[f"I{a}"]) > 1e-12:
        raise AssertionError(f"stage-one coefficient identity failed ({ident})")
    stages += _run_recipients(st, [("stage2_r%d" % b, (b,), W_joint(lam))], m, lam)
    if not np.array_equal(st.labels(a), frozen):
        raise AssertionError("first recipient revised after stage 1")
    meta = {"family": fam, "m": m, "lam": lam, **(meta or {})}
    extra = {"stages": stages, "after_greedy": stages[-1]["after_greedy"], "order": list(order),
             "stage1_coefficient": 1.5 * lam,
             "stage1_identity": {"F_joint_with_constant_minus_D_const": fj_const_minus_db, "stage1_objective": stage1,
                                 "abs_diff": ident, "I12_with_constant": I12c, "I_own": t[f"I{a}"]}}
    return _finish(st, fam, m, lam, meta, P1, d1, P2, d2, S_fit, extra, t0, c0)


def fit_direct_task(P1, d1, P2, d2, S_fit, m, meta=None, rounds=None):
    """Row-level KL k-means with m prototypes per predicted class on the original teacher vectors (same init, round,
    tie and empty-cell rules as the fine partition; k = min(m, distinct, rows)); identity map. No sensitive labels
    enter the fit; S_fit is used only for the receipt objective."""
    from dpc.partition import ROUNDS
    t0, c0 = time.perf_counter(), time.process_time()
    m = _check_m(m)
    P1 = check_probs(P1, name="P1")
    P2 = check_probs(P2, name="P2")
    g1 = fit_fine(P1, d1, P1.shape[1], max_cells=m, rounds=rounds or ROUNDS, tag="DIRECT|r1")
    g2 = fit_fine(P2, d2, P2.shape[1], max_cells=m, rounds=rounds or ROUNDS, tag="DIRECT|r2")
    P1, d1, P2, d2, s, T = _prep(g1, g2, P1, d1, P2, d2, S_fit)
    st = State(g1, g2, T, _fine_labels(g1), _fine_labels(g2))
    meta = {"family": "DIRECT-TASK", "m": m, "lam": None, **(meta or {})}
    extra = {"direct_partitions": {"r1": g1.receipt, "r2": g2.receipt}}
    return _finish(st, "DIRECT-TASK", m, None, meta, P1, d1, P2, d2, S_fit, extra, t0, c0)


def fit_joint(fine1, fine2, P1, d1, P2, d2, S_fit, m, lam, witnesses=None, meta=None):
    """Greedy F_joint agglomeration over both recipients + joint refinement; also joint refinement from the FINE-TASK,
    LOCAL, SEQ-12 and SEQ-21 solutions (passed in as PolicyPairs on the same fine partitions, or recomputed). The
    lowest final F_joint wins among the refined starts and the unchanged witnesses (exact comparison of from-scratch
    objectives; ties -> fixed order JOINT-GREEDY, FINE-TASK, LOCAL, SEQ-12, SEQ-21, refined before unchanged)."""
    t0, c0 = time.perf_counter(), time.process_time()
    m = _check_m(m)
    lam = float(lam)
    P1, d1, P2, d2, s, T = _prep(fine1, fine2, P1, d1, P2, d2, S_fit)
    W = W_joint(lam)
    wit = dict(witnesses or {})
    extra_keys = set(wit) - set(WITNESS_FAMILIES)
    if extra_keys:
        raise ValueError(f"unknown witness slots {sorted(extra_keys)}")
    fps = (fine1.fingerprint(), fine2.fingerprint())
    witness_source = {}
    for fam in WITNESS_FAMILIES:
        if fam in wit:
            pp = wit[fam]
            if (pp.p1.fine.fingerprint(), pp.p2.fine.fingerprint()) != fps:
                raise ValueError(f"witness {fam} was fitted on different fine partitions")
            if pp.family != fam or pp.config.get("m") != m or (fam != "FINE-TASK" and pp.config.get("lam") != lam):
                raise ValueError(f"witness {fam} has a different family/m/lam")
            witness_source[fam] = "passed_in"
        else:
            if fam == "FINE-TASK":
                pp, _ = fit_fine_task(fine1, fine2, P1, d1, P2, d2, S_fit, m)
            elif fam == "LOCAL":
                pp, _ = fit_local(fine1, fine2, P1, d1, P2, d2, S_fit, m, lam)
            else:
                pp, _ = fit_sequential((1, 2) if fam == "SEQ-12" else (2, 1), fine1, fine2, P1, d1, P2, d2, S_fit,
                                       m, lam)
            wit[fam] = pp
            witness_source[fam] = "recomputed"
    starts = {}
    work_total = {"greedy_candidates": 0, "greedy_steps": 0, "refine_candidates": 0, "refine_moves": 0}
    # 1. greedy joint start
    st = State(fine1, fine2, T, _fine_labels(fine1), _fine_labels(fine2))
    merges = []
    greedy(st, (1, 2), W, m, merges)
    g_terms = {**st.terms(), **F_values(st.terms(), lam)}
    moves = []
    sw, conv = refine(st, (1, 2), W, moves)
    starts["JOINT-GREEDY"] = {"state": st, "initial_F_joint": g_terms["F_joint"], "after_greedy": g_terms,
                              "merges": merges, "moves": moves, "sweeps": sw, "converged": conv}
    # 2. witnesses
    witness_initial = {}
    for fam in WITNESS_FAMILIES:
        pp = wit[fam]
        sw_state = State(fine1, fine2, T, _labels_from_policy(pp.p1), _labels_from_policy(pp.p2))
        for r, pol in ((1, pp.p1), (2, pp.p2)):
            for c in range(pol.K):
                if sw_state.R[r].coarse_count(c) > m:
                    raise ValueError(f"witness {fam} exceeds the cap m on recipient {r} class {c}")
        init_terms = {**sw_state.terms(), **F_values(sw_state.terms(), lam)}
        witness_initial[fam] = {"terms": init_terms, "labels": (sw_state.labels(1), sw_state.labels(2)),
                                "fingerprint": pp.fingerprint()}
        mv = []
        sw, conv = refine(sw_state, (1, 2), W, mv)
        starts[fam] = {"state": sw_state, "initial_F_joint": init_terms["F_joint"], "moves": mv, "sweeps": sw,
                       "converged": conv}
    for v in starts.values():
        for k in work_total:
            work_total[k] += v["state"].work[k]
    # 3. choose
    cand = []
    for name in JOINT_START_ORDER:
        cand.append((name, "refined", starts[name]["state"].terms()))
    for name in WITNESS_FAMILIES:
        cand.append((name, "unchanged", witness_initial[name]["terms"]))
    vals = [F_values(t, lam)["F_joint"] for _, _, t in cand]
    best = min(vals)
    k = vals.index(best)
    win_name, win_kind, _ = cand[k]
    if win_kind == "refined":
        final_state = starts[win_name]["state"]
    else:
        l1, l2 = witness_initial[win_name]["labels"]
        final_state = State(fine1, fine2, T, l1, l2)
    final_state.work = work_total
    fj_final = F_values(final_state.terms(), lam)["F_joint"]
    for name in WITNESS_FAMILIES:
        wv = witness_initial[name]["terms"]["F_joint"]
        if not fj_final <= wv:
            raise AssertionError(f"JOINT final F_joint {fj_final} exceeds witness {name} {wv}")
    meta = {"family": "JOINT", "m": m, "lam": lam, **(meta or {})}
    start_rec = {}
    for name in JOINT_START_ORDER:
        v = starts[name]
        start_rec[name] = {"initial_F_joint": v["initial_F_joint"],
                           "refined_F_joint": F_values(v["state"].terms(), lam)["F_joint"],
                           "refine_moves": v["moves"], "sweeps": v["sweeps"], "converged": v["converged"],
                           "work": dict(v["state"].work)}
        if name == "JOINT-GREEDY":
            start_rec[name]["merges"] = v["merges"]
            start_rec[name]["after_greedy"] = v["after_greedy"]
            start_rec[name]["positive_increment_merges"] = int(sum(1 for x in v["merges"] if x["increment"] > 0))
        else:
            start_rec[name]["source"] = witness_source[name]
            start_rec[name]["witness_fingerprint"] = witness_initial[name]["fingerprint"]
    extra = {"starts": start_rec, "winner": {"start": win_name, "kind": win_kind, "F_joint": best},
             "candidates": [{"start": n, "kind": kd, "F_joint": v} for (n, kd, _), v in zip(cand, vals)],
             "witness_dominance": {n: {"witness_F_joint": witness_initial[n]["terms"]["F_joint"],
                                       "final_minus_witness": fj_final - witness_initial[n]["terms"]["F_joint"]}
                                   for n in WITNESS_FAMILIES},
             "after_greedy": starts["JOINT-GREEDY"]["after_greedy"]}
    pair, rec = _finish(final_state, "JOINT", m, lam, meta, P1, d1, P2, d2, S_fit, extra, t0, c0)
    return pair, rec, wit


def check_joint_dominance(joint: PolicyPair, witnesses: dict, P1, d1, P2, d2, S_fit, lam):
    """Independent row-level check: list of witnesses whose F_joint is strictly below the JOINT policy's."""
    fj = evaluate_rows(joint, P1, d1, P2, d2, S_fit, lam)["F_joint"]
    bad = []
    for name, w in witnesses.items():
        fw = evaluate_rows(w, P1, d1, P2, d2, S_fit, lam)["F_joint"]
        if fw < fj - 1e-12:
            bad.append({"witness": name, "witness_F_joint": fw, "joint_F_joint": fj})
    return bad


# ----------------------------------------------------------------------------------------------- entry points
def fit_policy_pair(family, fine1, fine2, P1, d1, P2, d2, S_fit, m, lam, witnesses=None, meta=None):
    """Single entry point. family in CLASS-ONLY, FINE-TASK, DIRECT-TASK, LOCAL, SEQ-12, SEQ-21, JOINT. Task-only
    families (and CLASS-ONLY) require lam None. ``witnesses`` (JOINT only): {"FINE-TASK", "LOCAL", "SEQ-12",
    "SEQ-21"} -> PolicyPair fitted on the same fine partitions, m and lam; missing slots are recomputed (recorded as
    starts[*]["source"]). ``meta`` (e.g. teacher, seed, teacher_model_sha256, feature_names_sha256) is copied into
    pair.config and each Policy.meta. Returns (PolicyPair, receipts)."""
    if family in ("CLASS-ONLY", "FINE-TASK", "DIRECT-TASK"):
        if lam is not None:
            raise ValueError(f"{family} is task-only; lam must be None")
    elif family in PRIVACY_FAMILIES:
        if lam is None or not np.isfinite(lam) or lam < 0:
            raise ValueError(f"{family} needs a finite lam >= 0")
    else:
        raise ValueError(f"unknown family {family}")
    if family == "CLASS-ONLY":
        if m not in (1, None):
            raise ValueError("CLASS-ONLY is m = 1")
        pair, rec = fit_class_only(fine1, fine2, P1, d1, P2, d2, S_fit, meta=meta)
    elif family == "FINE-TASK":
        pair, rec = fit_fine_task(fine1, fine2, P1, d1, P2, d2, S_fit, m, meta)
    elif family == "DIRECT-TASK":
        pair, rec = fit_direct_task(P1, d1, P2, d2, S_fit, m, meta)
        rec["fine_partitions_unused_by_design"] = [fine1.fingerprint(), fine2.fingerprint()]
    elif family == "LOCAL":
        pair, rec = fit_local(fine1, fine2, P1, d1, P2, d2, S_fit, m, lam, meta)
    elif family in ("SEQ-12", "SEQ-21"):
        pair, rec = fit_sequential((1, 2) if family == "SEQ-12" else (2, 1), fine1, fine2, P1, d1, P2, d2, S_fit, m,
                                   lam, meta)
    else:
        pair, rec, _ = fit_joint(fine1, fine2, P1, d1, P2, d2, S_fit, m, lam, witnesses, meta)
    pair.config = {"family": family, "m": 1 if family == "CLASS-ONLY" else int(m), "lam": lam, **(meta or {})}
    return pair, rec


def config_id(family, m, lam):
    if family == "CLASS-ONLY":
        return "CLASS-ONLY__m1"
    if lam is None:
        return f"{family}__m{m}"
    return f"{family}__m{m}__lam{lam:g}"


def fit_bank(fine1, fine2, P1, d1, P2, d2, S_fit, ms=(2, 4, 8), lams=(0.1, 1.0, 10.0), class_only=True,
             meta=None):
    """One teacher/seed bank in dependency order (witnesses reused by JOINT). Returns {config_id: (pair, receipts)}."""
    out = {}
    meta = dict(meta or {})
    if class_only:
        out[config_id("CLASS-ONLY", 1, None)] = fit_policy_pair("CLASS-ONLY", fine1, fine2, P1, d1, P2, d2, S_fit,
                                                                1, None, meta=meta)
    for m in ms:
        for fam in TASK_FAMILIES:
            out[config_id(fam, m, None)] = fit_policy_pair(fam, fine1, fine2, P1, d1, P2, d2, S_fit, m, None,
                                                           meta=meta)
        for lam in lams:
            for fam in ("LOCAL", "SEQ-12", "SEQ-21"):
                out[config_id(fam, m, lam)] = fit_policy_pair(fam, fine1, fine2, P1, d1, P2, d2, S_fit, m, lam,
                                                              meta=meta)
            wit = {"FINE-TASK": out[config_id("FINE-TASK", m, None)][0],
                   **{f: out[config_id(f, m, lam)][0] for f in ("LOCAL", "SEQ-12", "SEQ-21")}}
            out[config_id("JOINT", m, lam)] = fit_policy_pair("JOINT", fine1, fine2, P1, d1, P2, d2, S_fit, m, lam,
                                                              witnesses=wit, meta=meta)
    return out


def find_aliases(pairs: dict):
    """{config_id: PolicyPair} -> {config_id: canonical config_id} where the canonical ID is the first (sorted) config
    with an identical pair fingerprint (identical assignment partitions, maps and prototypes on both recipients).
    Also returns per-recipient alias maps."""
    out = {"pair": {}, "r1": {}, "r2": {}}
    seen = {"pair": {}, "r1": {}, "r2": {}}
    for cid in sorted(pairs):
        pp = pairs[cid]
        for key, fp in (("pair", pp.fingerprint()), ("r1", pp.p1.fingerprint()), ("r2", pp.p2.fingerprint())):
            seen[key].setdefault(fp, cid)
            out[key][cid] = seen[key][fp]
    return out
