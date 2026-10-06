"""KL/Bregman k-means within one teacher-predicted class: three registered starts, a registered convergence rule,
best-coherent-iterate retention and deterministic start selection (qpc, confidence-capacity study).

Provenance: the arithmetic (smoothing, masked KL, sufficient statistics, deployment) is imported unchanged from the
pinned dpc package (dpc.partition at source SHA 0a7b05a5); the "source" start is the dpc deterministic quantile
initialisation, and ``rule="dpc"`` reproduces dpc.partition.kmeans_class / fit_fine bit for bit (tested).

Registered rules (see results/pcrl_confidence_capacity_v1/METHOD_CARD.md, section 3):

* Rows: the fitting rows of ONE predicted class c, Pc (n, K) float64. k = min(m, #distinct vectors, #rows).
* KL(p || q) = dpc.partition.kl_matrix (masked, increasing-k summation); q = smooth(centroid, c), eps = 1e-12.
* Starts (fixed order, also the tie order): "source", "kpp:20261006", "kpp:20261007".
    - "source": distinct vectors (np.unique axis 0, ascending lexicographic) stably re-sorted by p_c DESCENDING,
      order statistics at positions floor((2j+1) U / (2k)), j = 0..k-1 (dpc rule).
    - "kpp:<seed>": KL k-means++ over the distinct vectors U (ascending lexicographic order) with multiplicities w
      (np.unique return_counts). Generator: rng = np.random.default_rng([seed, K, c]) (numpy SeedSequence of the
      three integers; PCG64). Exactly one rng.random() draw per centre. Draw rule for weights v (length |U|):
      cum = np.cumsum(v); t = rng.random() * cum[-1]; i = np.searchsorted(cum, t, side="right"); if i == |U| then
      i = the last index with v > 0. Centre 1: v = w (multiplicity-weighted, i.e. uniform over rows). Centre j > 1:
      v_u = w_u * min_{chosen} KL(u || smooth(chosen, c)) with already-chosen distinct vectors set to exactly 0;
      if every unchosen v is 0, v = w on the unchosen vectors (recorded as "degenerate_draws"). Initial centroid =
      the chosen distinct vector (unsmoothed), in selection order.
    - KL roundoff guard (initialisation probabilities ONLY): values in [-1e-12, 0) are clipped to 0 and counted
      ("kl_clipped"); any value < -1e-12 raises. Assignments and objectives never clip: they use the raw masked KL
      and the exact sufficient-statistic objective, so the guard cannot change the optimised objective.
* Iteration (rule "qpc"): pass r assigns every row to argmin_j KL(p || Q_j) (ties -> lowest j), giving the coherent
  iterate (Q, a): deploying Q reproduces a on these rows and the decoded prototype of cell j is smooth(S_j / n_j, c).
  Its objective is J(a) = A_c + sum_{j: n_j > 0} h_j, A_c = sum over rows of sum_k p_k log p_k (computed once),
  h_j = -S_j . log smooth(S_j / n_j, c) (= the class total of KL(p || decoded prototype)). Stop rules, in order:
    1. assignment fixed point: r > 1 and a_r == a_{r-1}           -> converged, reason "assignment_fixed_point";
    2. relative tolerance: rel_r = |J_{r-1} - J_r| / max(|J_{r-1}|, 1e-300) < RTOL = 1e-9 for PATIENCE = 3
       successive passes (r-2, r-1, r)                            -> one final update + assignment pass, then
       converged, reason "relative_tolerance";
    3. cap: after pass r == rounds (200)                           -> one final update + assignment pass; converged
       only if that final assignment equals pass r's ("assignment_fixed_point"), else reason "cap", converged False.
  Otherwise update centroid_j = S_j / n_j (arithmetic mean = KL Bregman centroid); an EMPTY cell keeps its previous
  centroid (dpc rule; every empty cell per pass is reported). The returned iterate is the best coherent iterate:
  lowest J over every assignment pass, later passes win exact ties (so a fixed point returns the centroid that equals
  its prototype).
* rule "dpc" (A1 reproduction only): dpc.partition.kmeans_class semantics -- same passes, stop rule 1 only, cap =
  rounds (20) with the final assignment, and the LAST iterate is returned.
* Cells empty in the returned assignment are removed (order kept). Start selection per class: the lowest J wins; a
  later start replaces the incumbent only if J_new < J_inc - 1e-12 * max(|J_inc|, 1) (ties -> earlier start).
  No task label or SEX enters any step; only the teacher probabilities of the class's fitting rows.
* Absent class (no fitting row): one reserved fallback cell, n = 0, mean uniform 1/K, centroid smooth(uniform, c).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from dpc import partition as DPT
from dpc.compress import h_cost

EPS = DPT.EPS
ROUNDS = 200
RTOL = 1e-9
PATIENCE = 3
KL_CLIP = 1e-12
SELECT_TOL = 1e-12
STARTS = ("source", "kpp:20261006", "kpp:20261007")
SPARSE_N = DPT.SPARSE_N
check_probs = DPT.check_probs
check_decisions = DPT.check_decisions
smooth = DPT.smooth
kl_matrix = DPT.kl_matrix
cell_stats = DPT.cell_stats
assign_fine = DPT.assign_fine


# ----------------------------------------------------------------------------------------------- partition record
@dataclass
class FinePartition(DPT.FinePartition):
    """dpc.partition.FinePartition with qpc serialisation (kind "qpc.FinePartition"); all dpc arithmetic applies."""

    def to_dict(self):
        z = super().to_dict()
        z["kind"] = "qpc.FinePartition"
        return z

    @classmethod
    def from_dict(cls, z):
        if z.get("kind") != "qpc.FinePartition":
            raise ValueError(f"not a qpc.FinePartition record (kind {z.get('kind')!r})")
        K = int(z["K"])
        f = cls(K=K, cell_class=np.asarray(z["cell_class"], dtype=np.int64),
                centroid=np.asarray(z["centroid"], dtype=np.float64).reshape(-1, K),
                mean=np.asarray(z["mean"], dtype=np.float64).reshape(-1, K),
                n=np.asarray(z["n"], dtype=np.int64), S=np.asarray(z["S"], dtype=np.float64).reshape(-1, K),
                A=np.asarray(z["A"], dtype=np.float64), fallback=np.asarray(z["fallback"], dtype=bool),
                eps=float(z["eps"]), receipt=z.get("receipt", {}))
        if f.eps != EPS:
            raise ValueError("eps differs from the fixed 1e-12 rule")
        f.validate()
        if "fingerprint" in z and z["fingerprint"] != f.fingerprint():
            raise ValueError("fine partition fingerprint mismatch after load")
        return f

    @classmethod
    def from_dpc(cls, f: DPT.FinePartition):
        return cls(K=f.K, cell_class=f.cell_class, centroid=f.centroid, mean=f.mean, n=f.n, S=f.S, A=f.A,
                   fallback=f.fallback, eps=f.eps, receipt=dict(f.receipt))


# ----------------------------------------------------------------------------------------------- starts
def parse_start(start):
    if start == "source":
        return "source", None
    if isinstance(start, str) and start.startswith("kpp:"):
        return "kpp", int(start[4:])
    raise ValueError(f"unknown start {start!r}")


def guarded_kl(P, Q):
    """KL matrix for initialisation probabilities: [-1e-12, 0) -> 0 (counted); < -1e-12 raises."""
    M = kl_matrix(P, Q)
    if np.any(M < -KL_CLIP):
        raise ValueError(f"KL below -{KL_CLIP} ({float(M.min())}): not roundoff")
    neg = M < 0
    return np.where(neg, 0.0, M), int(neg.sum())


def source_init(Pc, c, k):
    """dpc deterministic quantile initialisation: (centroids (k, K), #distinct, positions)."""
    U = np.unique(Pc, axis=0)
    U = U[np.argsort(-U[:, c], kind="stable")]
    nU = U.shape[0]
    pos = [((2 * j + 1) * nU) // (2 * k) for j in range(k)]
    return U[pos].copy(), nU, pos


def _draw(rng, v):
    cum = np.cumsum(v)
    t = rng.random() * cum[-1]
    i = int(np.searchsorted(cum, t, side="right"))
    if i >= v.shape[0]:
        i = int(np.flatnonzero(v > 0)[-1])
    return i


def kpp_init(Pc, c, k, seed, K):
    """KL k-means++ over distinct vectors with multiplicities (rule in the module docstring)."""
    U, w = np.unique(Pc, axis=0, return_counts=True)
    w = w.astype(np.float64)
    nU = U.shape[0]
    rng = np.random.default_rng([int(seed), int(K), int(c)])
    chosen = [_draw(rng, w)]
    dmin = np.full(nU, np.inf)
    clipped, degenerate, kl_evals = 0, 0, 0
    for _ in range(1, k):
        dk, nc = guarded_kl(U, smooth(U[chosen[-1]], c)[None])
        clipped += nc
        kl_evals += nU
        dmin = np.minimum(dmin, dk[:, 0])
        v = w * dmin
        v[chosen] = 0.0
        if not v.sum() > 0:
            v = w.copy()
            v[chosen] = 0.0
            degenerate += 1
        chosen.append(_draw(rng, v))
    return U[chosen].copy(), nU, {"seed": int(seed), "rng": f"default_rng([{int(seed)}, {int(K)}, {int(c)}])",
                                  "chosen_distinct_index": [int(i) for i in chosen],
                                  "chosen_multiplicity": [int(w[i]) for i in chosen], "kl_clipped": clipped,
                                  "degenerate_draws": degenerate, "kl_evals": kl_evals}


# ----------------------------------------------------------------------------------------------- one class, one start
def class_objective(Pc_negent_total, S, n, c, K):
    """J = A_c + sum_j h_j over nonempty cells (class total of KL(p || decoded prototype))."""
    return float(Pc_negent_total + float(np.sum(h_cost(S, n, c, K))))


def kmeans_class(Pc, c, m, start="source", rounds=ROUNDS, rule="qpc", rtol=RTOL, patience=PATIENCE,
                 init_override=None):
    """Returns (Q (k, K) smoothed assignment centroids, a (n,) assignment, receipt). Cells not yet pruned.
    ``init_override`` (tests only; never used by a registered start) supplies explicit initial centroids."""
    if rule not in ("qpc", "dpc"):
        raise ValueError("rule must be 'qpc' or 'dpc'")
    Pc = np.asarray(Pc, dtype=np.float64)
    n_rows, K = Pc.shape
    nU = np.unique(Pc, axis=0).shape[0]
    if init_override is not None:
        C = np.array(init_override, dtype=np.float64)
        k = C.shape[0]
        init = {"explicit": True, "kl_evals": 0, "kl_clipped": 0}
        kind = "explicit"
    else:
        kind, seed = parse_start(start)
        k = int(min(m, nU, n_rows))
    if kind == "explicit":
        pass
    elif kind == "source":
        C, nU, pos = source_init(Pc, c, k)
        init = {"positions": [int(x) for x in pos], "kl_evals": 0, "kl_clipped": 0}
    else:
        C, nU, init = kpp_init(Pc, c, k, seed, K)
    negent = float(np.sum(DPT.neg_entropy_rows(Pc)))
    cls_k = np.full(k, c)
    Q = smooth(C, cls_k)
    traj, changed, empty = [], [], []
    work = {"assign_passes": 0, "updates": 0, "kl_evals": int(init.get("kl_evals", 0))}
    best = None
    prev_a, prev_J = None, None
    streak, reason, converged, rounds_used = 0, "cap", False, 0

    def assign(Qm):
        work["assign_passes"] += 1
        work["kl_evals"] += n_rows * Qm.shape[0]
        return kl_matrix(Pc, Qm).argmin(1)

    def record(Qm, a):
        nonlocal best
        nn, S, _ = cell_stats(Pc, a, k)
        J = class_objective(negent, S, nn, c, K)
        traj.append(J)
        empty.append(int(np.sum(nn == 0)))
        if rule == "qpc" and (best is None or J <= best[2]):
            best = (Qm.copy(), a.copy(), J, len(traj))
        return nn, S, J

    def update(nn, S):
        nz = nn > 0
        C[nz] = S[nz] / nn[nz, None]             # empty cell keeps its previous centroid
        work["updates"] += 1
        return smooth(C, cls_k)

    for r in range(1, rounds + 1):
        a = assign(Q)
        rounds_used = r
        nn, S, J = record(Q, a)
        changed.append(int(n_rows if prev_a is None else np.sum(a != prev_a)))
        if prev_a is not None and np.array_equal(a, prev_a):
            converged, reason = True, "assignment_fixed_point"
            last = (Q, a)
            break
        if rule == "qpc" and prev_J is not None:
            rel = abs(prev_J - J) / max(abs(prev_J), 1e-300)
            streak = streak + 1 if rel < rtol else 0
            if streak >= patience:
                Q = update(nn, S)
                a2 = assign(Q)
                record(Q, a2)
                changed.append(int(np.sum(a2 != a)))
                converged, reason = True, "relative_tolerance"
                last = (Q, a2)
                break
        Q = update(nn, S)
        prev_a, prev_J = a, J
    else:
        a2 = assign(Q)                          # final deployment assignment with the last centroids
        record(Q, a2)
        changed.append(int(np.sum(a2 != prev_a)))
        converged = bool(np.array_equal(a2, prev_a))
        reason = "assignment_fixed_point" if converged else "cap"
        last = (Q, a2)
    if rule == "dpc":
        Qout, aout, best_pass = last[0], last[1], len(traj)
        Jout = traj[-1]
    else:
        Qout, aout, Jout, best_pass = best
    nn = np.bincount(aout, minlength=k)
    rec = {"start": start, "rule": rule, "class": int(c), "rows": int(n_rows), "distinct_vectors": int(nU),
           "initial_cells": k, "init": init, "rounds_cap": int(rounds), "rounds_used": int(rounds_used),
           "converged": bool(converged), "stop_reason": reason, "rtol": rtol if rule == "qpc" else None,
           "patience": patience if rule == "qpc" else None, "objective_trajectory": traj,
           "changed_per_pass": changed, "empty_cells_per_pass": empty,
           "empty_cell_events": int(sum(1 for e in empty if e > 0)), "returned_pass": int(best_pass),
           "objective": float(Jout), "mean_kl": float(Jout / n_rows), "final_cell_counts": [int(v) for v in nn],
           "empty_in_returned": int(np.sum(nn == 0)), "work": work}
    return Qout, aout, rec


# ----------------------------------------------------------------------------------------------- per recipient
def _fallback_cells(K, c):
    u = np.full(K, 1.0 / K)
    return {"cls": np.array([c]), "cen": smooth(u, c)[None], "mean": u[None], "n": np.array([0]),
            "S": np.zeros((1, K)), "A": np.zeros(1), "fb": np.array([True])}


def _cells(Pc, c, Q, a):
    n, S, A = cell_stats(Pc, a, Q.shape[0])
    keep = np.flatnonzero(n > 0)
    return {"cls": np.full(keep.size, c), "cen": Q[keep], "mean": S[keep] / n[keep, None], "n": n[keep],
            "S": S[keep], "A": A[keep], "fb": np.zeros(keep.size, dtype=bool)}, int(Q.shape[0] - keep.size)


def _assemble(K, blocks):
    cat = lambda key: np.concatenate([b[key] for b in blocks])  # noqa: E731
    return FinePartition(K=K, cell_class=cat("cls").astype(np.int64), centroid=cat("cen"), mean=cat("mean"),
                         n=cat("n").astype(np.int64), S=cat("S"), A=cat("A"), fallback=cat("fb"))


def _check_deployment(fine, P, d):
    fine.validate()
    cell = assign_fine(P, d, fine)
    n, S, A = cell_stats(P, cell, fine.F)
    if not (np.array_equal(n, fine.n) and np.array_equal(S, fine.S) and np.array_equal(A, fine.A)):
        raise AssertionError("training statistics differ from the deployed assignment on the training rows")
    return cell


def select_start(recs):
    """Index of the winning start (lowest objective; replace only if lower by > 1e-12 * max(|J|, 1))."""
    w = 0
    for i in range(1, len(recs)):
        Jb = recs[w]["objective"]
        if recs[i]["objective"] < Jb - SELECT_TOL * max(abs(Jb), 1.0):
            w = i
    return w


@dataclass
class RecipientFit:
    """Winner partition (per-class winning start), the per-start partitions and JSON-safe receipts."""
    partition: FinePartition
    by_start: dict
    receipt: dict = field(default_factory=dict)


def fit_recipient(P, d, K, m, *, starts=STARTS, rounds=ROUNDS, rule="qpc", tag=""):
    """Direct per-class KL k-means of fitting rows with every start; per-class winner. rule "dpc" (one start only)
    reproduces dpc.partition.fit_fine(P, d, K, max_cells=m, rounds=rounds)."""
    import time
    t0, c0 = time.perf_counter(), time.process_time()
    P = check_probs(P, K)
    d = check_decisions(P, d)
    if int(m) != m or m < 1 or rounds < 1:
        raise ValueError("m and rounds must be positive integers")
    starts = tuple(starts)
    if rule == "dpc" and starts != ("source",):
        raise ValueError("rule 'dpc' is the single source start")
    blocks = {s: [] for s in starts}
    win_blocks, per_class = [], []
    for c in range(K):
        rows = np.flatnonzero(d == c)
        if rows.size == 0:
            fbk = _fallback_cells(K, c)
            for s in starts:
                blocks[s].append(fbk)
            win_blocks.append(fbk)
            per_class.append({"class": c, "rows": 0, "fallback": True, "winner": None, "cells": 1,
                              "effective_cells": 0, "starts": []})
            continue
        Pc = P[rows]
        recs, cells = [], []
        for s in starts:
            Q, a, rec = kmeans_class(Pc, c, m, s, rounds, rule)
            cb, removed = _cells(Pc, c, Q, a)
            rec["removed_empty_cells"] = removed
            rec["cells"] = int(cb["n"].size)
            rec["cell_counts"] = [int(v) for v in cb["n"]]
            rec["sparse_cells"] = [int(j) for j, v in enumerate(cb["n"]) if v < SPARSE_N]
            recs.append(rec)
            cells.append(cb)
            blocks[s].append(cb)
        w = select_start(recs)
        win_blocks.append(cells[w])
        per_class.append({"class": c, "rows": int(rows.size), "fallback": False, "winner": starts[w],
                          "winner_objective": recs[w]["objective"], "cells": recs[w]["cells"],
                          "effective_cells": recs[w]["cells"], "starts": recs})
    fine = _assemble(K, win_blocks)
    _check_deployment(fine, P, d)
    by_start = {}
    for s in starts:
        fs = _assemble(K, blocks[s])
        _check_deployment(fs, P, d)
        by_start[s] = fs
    work = {"assign_passes": 0, "updates": 0, "kl_evals": 0}
    for pc in per_class:
        for r in pc["starts"]:
            for kk in work:
                work[kk] += r["work"][kk]
    rec = {"tag": tag, "K": int(K), "m": int(m), "rows": int(P.shape[0]), "rounds_cap": int(rounds), "rule": rule,
           "starts": list(starts), "rtol": RTOL, "patience": PATIENCE, "eps": EPS, "F": fine.F,
           "per_class": per_class, "effective_cells_per_class": [int(pc["effective_cells"]) for pc in per_class],
           "fallback_classes": [int(pc["class"]) for pc in per_class if pc["fallback"]],
           "winners": [pc["winner"] for pc in per_class],
           "objective_total": float(sum(pc.get("winner_objective", 0.0) for pc in per_class)),
           "mean_kl_fit": float(sum(pc.get("winner_objective", 0.0) for pc in per_class) / max(P.shape[0], 1)),
           "all_converged": bool(all(r["converged"] for pc in per_class for r in pc["starts"])),
           "winners_converged": bool(all(next(r for r in pc["starts"] if r["start"] == pc["winner"])["converged"]
                                         for pc in per_class if not pc["fallback"])),
           "fingerprints": {"winner": fine.fingerprint(), **{s: by_start[s].fingerprint() for s in starts}},
           "work": work, "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}
    fine.receipt = {"tag": tag, "K": int(K), "m": int(m), "rule": rule, "rounds_cap": int(rounds),
                    "winners": rec["winners"], "F": fine.F}
    for s in starts:
        by_start[s].receipt = {"tag": tag, "K": int(K), "m": int(m), "rule": rule, "rounds_cap": int(rounds),
                               "winners": [None if pc["fallback"] else s for pc in per_class], "F": by_start[s].F}
    return RecipientFit(fine, by_start, rec)


def json_safe(obj):
    """Strict JSON check (no NaN/inf); returns the canonical serialisation's sha256."""
    s = json.dumps(obj, sort_keys=True, allow_nan=False)
    return hashlib.sha256(s.encode()).hexdigest()
