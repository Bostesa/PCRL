"""Teacher-score fine partitions (task-only KL/Bregman k-means within each teacher-predicted class).

Fixed rules (recorded in results/pcrl_decision_preserving_compression_v1/METHOD_CARD.md):

* Inputs are teacher probability vectors P (n, K) float64 and the teacher decision d = argmax P with numpy's
  first-index tie rule. ``d`` is validated against ``argmax P`` on every call: any other label array (true task
  label, SEX, per-row loss bins, ...) is refused, so no partition can be keyed by a label.
* smooth(mean, c) = (mean + eps*1 + eps*e_c) / (1 + (K+1)*eps), eps = 1e-12, float64; checked finite, >= 0,
  |sum - 1| <= 1e-12 and argmax strictly c.
* KL(p || q) = sum_k p_k (log p_k - log q_k) over p_k > 0 (0 log 0 = 0), summed in increasing k; q is smoothed.
* Per-cell sufficient statistics: n, S = sum of member p (float64 bincount, row order), A = sum over members of
  sum_k p_k log p_k (0 log 0 = 0). Sum over members of KL(p || q) = A - S . log q.
* k-means per predicted class c: n_cells = min(max_cells, #distinct vectors, #rows). Initial centroids: the distinct
  vectors (np.unique rows, ascending lexicographic) stably re-sorted by p_c DESCENDING (ties keep lexicographic
  order), then the order statistics at positions floor((2j+1) U / (2 n_cells)), j = 0..n_cells-1 (block midpoints).
  Round r: assign every row of the class to argmin_j KL(p || smooth(centroid_j, c)) (ties -> lowest j); if r > 1
  and the assignment equals round r-1's, stop (converged; rounds_used = r); else update centroid_j = arithmetic
  mean of its members (empty cell keeps its previous centroid). At most ``rounds`` = 20 rounds; if round 20 is
  reached without convergence a final assignment with the last centroids is made and converged=False is recorded.
  The stored assignment centroid is the smoothed centroid used in the last assignment, and the stored statistics
  (n, S, A, unsmoothed mean S/n) are those of the rows that this deployment rule maps to each cell, so fitting-row
  statistics always equal deployment-on-fitting-rows. Cells left empty by the final assignment are removed
  (order kept), so deployment can never route a row to a cell without fitting rows other than a class fallback.
* A predicted class with zero fitting rows gets one reserved FALLBACK cell with n = 0, unsmoothed mean uniform 1/K
  and centroid smooth(uniform, c) (argmax c via eps).
* Deployment (assign_fine): the row's predicted class selects that class's cells; nearest cell by KL to the stored
  smoothed centroid, ties -> lowest index. A class with only a fallback cell maps to it. Deployment never updates
  anything.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12
SUM_TOL = 1e-12          # prototype normalisation check
INPUT_SUM_TOL = 1e-9     # teacher probability rows accepted as inputs
MAX_CELLS = 16
ROUNDS = 20
SPARSE_N = 5


# ----------------------------------------------------------------------------------------------- validation
def check_probs(P, K=None, name="P"):
    """Teacher probability matrix: 2-D float64, finite, nonnegative, rows summing to 1 within INPUT_SUM_TOL."""
    P = np.asarray(P)
    if P.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, K); got shape {P.shape}")
    if not np.issubdtype(P.dtype, np.floating):
        raise ValueError(f"{name} must be a floating probability matrix; got dtype {P.dtype}")
    P = P.astype(np.float64, copy=False)
    if K is not None and P.shape[1] != K:
        raise ValueError(f"{name} has {P.shape[1]} columns; expected K={K}")
    if P.shape[1] < 2:
        raise ValueError("K must be >= 2")
    if not np.all(np.isfinite(P)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(P < 0):
        raise ValueError(f"{name} contains negative probabilities")
    if P.shape[0] and np.max(np.abs(P.sum(1) - 1.0)) > INPUT_SUM_TOL:
        raise ValueError(f"{name} rows do not sum to 1 within {INPUT_SUM_TOL}")
    return P + 0.0       # copy; also maps any -0.0 to +0.0


def teacher_decisions(P):
    """d = argmax p with numpy's first-index tie rule."""
    return np.asarray(P).argmax(1).astype(np.int64)


def check_decisions(P, d, name="d"):
    """The decision array must be exactly the teacher argmax (first-index ties). This is the gate that refuses any
    label-keyed partition: a true-label, SEX or loss-bin array differs from argmax P and is rejected."""
    t = teacher_decisions(P)
    if d is None:
        return t
    d = np.asarray(d)
    if d.shape != t.shape:
        raise ValueError(f"{name} has shape {d.shape}; expected {t.shape}")
    if not np.issubdtype(d.dtype, np.integer):
        if not np.all(d == np.round(d)):
            raise ValueError(f"{name} must be integer class indices")
    d = d.astype(np.int64)
    bad = int(np.sum(d != t))
    if bad:
        raise ValueError(f"{name} differs from the teacher decision argmax(P) on {bad} rows: only the teacher's own "
                         f"predicted class may key a partition (true labels, SEX or other labels are refused)")
    return d


# ----------------------------------------------------------------------------------------------- smoothing / KL
def smooth(mean_p, c, eps=EPS, check=True):
    """(mean + eps*1 + eps*e_c) / (1 + (K+1) eps). mean_p (K,) or (M, K); c scalar or (M,)."""
    m = np.asarray(mean_p, dtype=np.float64)
    one = m.ndim == 1
    M = np.atleast_2d(m)
    K = M.shape[1]
    c = np.broadcast_to(np.asarray(c, dtype=np.int64), (M.shape[0],))
    E = np.zeros_like(M)
    E[np.arange(M.shape[0]), c] = eps
    Q = (M + eps * np.ones_like(M) + E) / (1.0 + (K + 1) * eps)
    if check:
        check_prototypes(Q, c)
    return Q[0] if one else Q


def check_prototypes(Q, c):
    """Finite, nonnegative, sum 1 within SUM_TOL, argmax strictly c (Q[c] > Q[k] for every k != c)."""
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    c = np.broadcast_to(np.asarray(c, dtype=np.int64), (Q.shape[0],))
    if not np.all(np.isfinite(Q)):
        raise ValueError("prototype not finite")
    if np.any(Q < 0):
        raise ValueError("prototype negative")
    if Q.shape[0] and np.max(np.abs(Q.sum(1) - 1.0)) > SUM_TOL:
        raise ValueError(f"prototype does not sum to 1 within {SUM_TOL}")
    qc = Q[np.arange(Q.shape[0]), c]
    other = Q.copy()
    other[np.arange(Q.shape[0]), c] = -np.inf
    if Q.shape[0] and not np.all(qc > other.max(1)):
        raise ValueError("prototype argmax is not strictly its class")
    return True


def neg_entropy_rows(P):
    """sum_k p_k log p_k per row with 0 log 0 = 0 (summed in increasing k)."""
    P = np.asarray(P, dtype=np.float64)
    out = np.zeros(P.shape[0])
    for k in range(P.shape[1]):
        p = P[:, k]
        out = out + np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return out


def kl_matrix(P, Q):
    """KL(p_r || q_j) for every row r and prototype j: sum over p_k > 0 of p_k (log p_k - log q_jk), increasing k.
    Q must be strictly positive (smoothed)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    if np.any(Q <= 0):
        raise ValueError("KL reference must be strictly positive (smoothed)")
    LQ = np.log(Q)
    out = np.zeros((P.shape[0], Q.shape[0]))
    for k in range(P.shape[1]):
        p = P[:, k:k + 1]
        lp = np.log(np.where(p > 0, p, 1.0))
        out = out + np.where(p > 0, p * (lp - LQ[None, :, k]), 0.0)
    return out


def kl_rows(P, Q):
    """KL(p_r || q_r) row-wise (Q aligned with P)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    out = np.zeros(P.shape[0])
    for k in range(P.shape[1]):
        p = P[:, k]
        lp = np.log(np.where(p > 0, p, 1.0))
        out = out + np.where(p > 0, p * (lp - np.log(Q[:, k])), 0.0)
    return out


def cell_stats(P, cell, F):
    """n (int64), S (float64 bincount sums, row order) and A (sum of neg-entropies) per cell id in [0, F)."""
    P = np.asarray(P, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.int64)
    n = np.bincount(cell, minlength=F).astype(np.int64)
    S = np.stack([np.bincount(cell, weights=P[:, k], minlength=F) for k in range(P.shape[1])], 1)
    A = np.bincount(cell, weights=neg_entropy_rows(P), minlength=F)
    return n, S.astype(np.float64), A.astype(np.float64)


# ----------------------------------------------------------------------------------------------- k-means
def _init_centroids(Pc, c, k):
    U = np.unique(Pc, axis=0)                               # ascending lexicographic
    order = np.argsort(-U[:, c], kind="stable")            # p_c descending, lexicographic tie-break
    U = U[order]
    nU = U.shape[0]
    pos = [((2 * j + 1) * nU) // (2 * k) for j in range(k)]
    return U[pos].copy(), nU, pos


def kmeans_class(Pc, c, max_cells=MAX_CELLS, rounds=ROUNDS):
    """KL/Bregman k-means for the rows of one predicted class c. Returns (assignment centroids smoothed, cell id per
    row after the final deployment assignment, receipt). Cells are not yet pruned."""
    n_rows = Pc.shape[0]
    nU = np.unique(Pc, axis=0).shape[0]
    k = int(min(max_cells, nU, n_rows))
    C, nU, pos = _init_centroids(Pc, c, k)
    Q = smooth(C, np.full(k, c))
    prev = None
    converged = False
    rounds_used = 0
    history = []
    for r in range(1, rounds + 1):
        a = kl_matrix(Pc, Q).argmin(1)
        rounds_used = r
        if prev is not None and np.array_equal(a, prev):
            converged = True
            break
        changed = int(n_rows if prev is None else np.sum(a != prev))
        history.append(changed)
        n, S, _ = cell_stats(Pc, a, k)
        for j in range(k):
            if n[j] > 0:
                C[j] = S[j] / n[j]
            # empty cell: keeps its previous centroid
        Q = smooth(C, np.full(k, c))
        prev = a
    if not converged:
        a = kl_matrix(Pc, Q).argmin(1)                    # final deployment assignment with the last centroids
        converged = bool(prev is not None and np.array_equal(a, prev))
    rec = {"class": int(c), "rows": int(n_rows), "distinct_vectors": int(nU), "initial_cells": k,
           "init_positions": [int(x) for x in pos], "rounds_used": int(rounds_used), "converged": bool(converged),
           "changed_per_round": history}
    return Q, a, rec


@dataclass
class FinePartition:
    """Fine cells enumerated in (predicted class, within-class index) order.

    cell_class (F,), centroid (F, K) smoothed assignment centroids, mean (F, K) unsmoothed member means (uniform for a
    fallback), n (F,) int64, S (F, K), A (F,), fallback (F,) bool. ``receipt`` is JSON-safe."""
    K: int
    cell_class: np.ndarray
    centroid: np.ndarray
    mean: np.ndarray
    n: np.ndarray
    S: np.ndarray
    A: np.ndarray
    fallback: np.ndarray
    eps: float = EPS
    receipt: dict = field(default_factory=dict)

    @property
    def F(self):
        return int(self.cell_class.shape[0])

    def cells_of(self, c):
        return np.flatnonzero(self.cell_class == c)

    def validate(self):
        K, F = self.K, self.F
        if self.centroid.shape != (F, K) or self.mean.shape != (F, K) or self.S.shape != (F, K):
            raise ValueError("fine partition array shapes inconsistent")
        if np.any(np.diff(self.cell_class) < 0):
            raise ValueError("fine cells must be enumerated in class order")
        for c in range(K):
            idx = self.cells_of(c)
            if idx.size == 0:
                raise ValueError(f"class {c} has no fine cell (a fallback is required)")
            fb = self.fallback[idx]
            if fb.any() and (idx.size != 1 or self.n[idx[0]] != 0):
                raise ValueError(f"class {c}: a fallback cell must be the only cell of its class and have n = 0")
            if (~fb).any() and np.any(self.n[idx[~fb]] <= 0):
                raise ValueError(f"class {c}: a non-fallback fine cell has no fitting rows")
        check_prototypes(self.centroid, self.cell_class)
        return True

    def fingerprint(self):
        h = hashlib.sha256()
        for a in (np.int64(self.K), self.cell_class.astype("<i8"), self.centroid.astype("<f8"),
                  self.fallback.astype(np.uint8), self.n.astype("<i8"), self.S.astype("<f8"), self.A.astype("<f8")):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def to_dict(self):
        return {"kind": "dpc.FinePartition", "K": int(self.K), "eps": float(self.eps),
                "cell_class": self.cell_class.tolist(), "centroid": self.centroid.tolist(),
                "mean": self.mean.tolist(), "n": self.n.tolist(), "S": self.S.tolist(), "A": self.A.tolist(),
                "fallback": [bool(x) for x in self.fallback], "receipt": self.receipt,
                "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, z):
        if z.get("kind") != "dpc.FinePartition":
            raise ValueError("not a dpc.FinePartition record")
        K = int(z["K"])
        f = cls(K=K, cell_class=np.asarray(z["cell_class"], dtype=np.int64),
                centroid=np.asarray(z["centroid"], dtype=np.float64).reshape(-1, K),
                mean=np.asarray(z["mean"], dtype=np.float64).reshape(-1, K),
                n=np.asarray(z["n"], dtype=np.int64), S=np.asarray(z["S"], dtype=np.float64).reshape(-1, K),
                A=np.asarray(z["A"], dtype=np.float64), fallback=np.asarray(z["fallback"], dtype=bool),
                eps=float(z["eps"]), receipt=z.get("receipt", {}))
        if float(z["eps"]) != EPS:
            raise ValueError("eps differs from the fixed 1e-12 rule")
        f.validate()
        if "fingerprint" in z and z["fingerprint"] != f.fingerprint():
            raise ValueError("fine partition fingerprint mismatch after load")
        return f


def to_dict(fine: FinePartition):
    return fine.to_dict()


def from_dict(z):
    return FinePartition.from_dict(z)


def assign_fine(P, d, fine: FinePartition):
    """Deployment: nearest fine cell (KL to the smoothed centroid) within the row's predicted class; ties -> lowest
    index; a class with only a fallback cell maps to it. Returns global fine-cell ids (int64). Updates nothing."""
    P = check_probs(P, fine.K)
    d = check_decisions(P, d)
    out = np.full(P.shape[0], -1, dtype=np.int64)
    for c in range(fine.K):
        rows = np.flatnonzero(d == c)
        if rows.size == 0:
            continue
        idx = fine.cells_of(c)
        if idx.size == 1:
            out[rows] = idx[0]
        else:
            out[rows] = idx[kl_matrix(P[rows], fine.centroid[idx]).argmin(1)]
    assert np.all(out >= 0)
    return out


def fit_fine(P, d, K, *, max_cells=MAX_CELLS, rounds=ROUNDS, tag=""):
    """Task-only fine partition of fitting rows (see module docstring). No label other than the teacher decision is
    accepted. Returns a FinePartition whose statistics equal assign_fine(P, d) on these rows."""
    P = check_probs(P, K)
    d = check_decisions(P, d)
    if max_cells < 1 or rounds < 1:
        raise ValueError("max_cells and rounds must be >= 1")
    cls_, cen, mean, n_, S_, A_, fb = [], [], [], [], [], [], []
    per_class = []
    for c in range(K):
        rows = np.flatnonzero(d == c)
        if rows.size == 0:
            u = np.full(K, 1.0 / K)
            cls_.append(np.array([c]))
            cen.append(smooth(u, c)[None])
            mean.append(u[None])
            n_.append(np.array([0]))
            S_.append(np.zeros((1, K)))
            A_.append(np.zeros(1))
            fb.append(np.array([True]))
            per_class.append({"class": c, "rows": 0, "fallback": True, "effective_cells": 0, "cells": 1,
                              "sparse_cells": [], "removed_empty_cells": 0})
            continue
        Pc = P[rows]
        Q, a, rec = kmeans_class(Pc, c, max_cells, rounds)
        n, S, A = cell_stats(Pc, a, Q.shape[0])
        keep = np.flatnonzero(n > 0)
        cls_.append(np.full(keep.size, c))
        cen.append(Q[keep])
        mean.append(S[keep] / n[keep, None])
        n_.append(n[keep])
        S_.append(S[keep])
        A_.append(A[keep])
        fb.append(np.zeros(keep.size, dtype=bool))
        rec.update({"fallback": False, "cells": int(keep.size), "effective_cells": int(keep.size),
                    "removed_empty_cells": int(Q.shape[0] - keep.size),
                    "sparse_cells": [int(j) for j, v in enumerate(n[keep]) if v < SPARSE_N],
                    "cell_counts": [int(v) for v in n[keep]]})
        per_class.append(rec)
    fine = FinePartition(K=K, cell_class=np.concatenate(cls_).astype(np.int64), centroid=np.concatenate(cen),
                         mean=np.concatenate(mean), n=np.concatenate(n_).astype(np.int64), S=np.concatenate(S_),
                         A=np.concatenate(A_), fallback=np.concatenate(fb))
    fine.validate()
    # consistency: deployment on the fitting rows reproduces the stored statistics exactly
    cell = assign_fine(P, d, fine)
    n, S, A = cell_stats(P, cell, fine.F)
    if not (np.array_equal(n, fine.n) and np.array_equal(S, fine.S) and np.array_equal(A, fine.A)):
        raise AssertionError("fine statistics differ from deployment on the fitting rows")
    fine.receipt = {"tag": tag, "K": K, "rows": int(P.shape[0]), "max_cells": int(max_cells), "rounds": int(rounds),
                    "eps": EPS, "F": fine.F, "per_class": per_class,
                    "effective_cells_per_class": [int(r["effective_cells"]) for r in per_class],
                    "fallback_classes": [int(r["class"]) for r in per_class if r["fallback"]],
                    "sparse_cells_total": int(sum(len(r["sparse_cells"]) for r in per_class)),
                    "max_abs_mean_vs_centroid": float(np.max(np.abs(fine.mean - fine.centroid)))}
    return fine


def validate_fine_against_rows(fine: FinePartition, P, d):
    """Refuse a fine partition whose stored statistics are not exactly the deployment statistics of these fitting
    rows (e.g. a planted row-label grouping that is not a nearest-centroid partition of the teacher scores)."""
    fine.validate()
    cell = assign_fine(P, d, fine)
    n, S, A = cell_stats(check_probs(P, fine.K), cell, fine.F)
    if not (np.array_equal(n, fine.n) and np.array_equal(S, fine.S) and np.array_equal(A, fine.A)):
        raise ValueError("fine partition statistics are not those of nearest-centroid deployment on the fitting "
                         "rows; partitions keyed by anything other than the teacher scores are refused")
    return cell


def fit_fine_pair(P1, d1, P2, d2, *, max_cells=MAX_CELLS, rounds=ROUNDS, tag=""):
    """Fine partitions for recipient 1 (income, K=2) and recipient 2 (occupation, K=6) on aligned fitting rows."""
    P1 = check_probs(P1, name="P1")
    P2 = check_probs(P2, name="P2")
    if P1.shape[0] != P2.shape[0]:
        raise ValueError("P1 and P2 must be aligned fitting rows")
    import time
    t0, c0 = time.perf_counter(), time.process_time()
    f1 = fit_fine(P1, d1, P1.shape[1], max_cells=max_cells, rounds=rounds, tag=f"{tag}|r1")
    f2 = fit_fine(P2, d2, P2.shape[1], max_cells=max_cells, rounds=rounds, tag=f"{tag}|r2")
    rec = {"r1": f1.receipt, "r2": f2.receipt, "fingerprints": [f1.fingerprint(), f2.fingerprint()],
           "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}
    return f1, f2, rec


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh)
