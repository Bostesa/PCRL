"""Stage D feasibility geometry F1-F5 (ccm; role C; PROTOCOL section 5; MATH_REVIEW.md B1, B2, R4.7, R5.5, R7.4, R8.4).
Pure functions of reference vectors (Ucal) and decisions; no data loading, no labels, no SEX. SYNTHETIC tests only
(tests/pcrl_confidence_constrained_mechanism_v1/test_geometry.py). Contract constants come from ccm.guard.CONFIG;
contract = "G" (PRIMARY) or "G_exp" (secondary diagnostic, never feeds the go rule). Bins never mix predicted classes;
every metric is reported per predicted class and in total. A representative q ACCEPTS a row iff ccm.guard.accepts
(tightened construction targets AND the canonical predicate) unless stated otherwise; SERVING held-out rows uses the
canonical predicate only.

F1 certified greedy cover. Rows in the fixed label-free cover order: decreasing max probability, then row index. Each
  row joins the FIRST existing bin of its class (creation order) for which adding it keeps the bin certifiably
  admissible, else opens a new bin. For each candidate bin, in order:
    a. NECESSARY prefilters, vectorised over the class's bins, each with a permissive ENGINEERING['prefilter_tol'] so
       they never prune a certifiable bin. G: nll_necessary on the running coordinatewise max; per label y the minimum
       of q.q - 2 q_y over {q >= L, sum q = 1} (at q = L + s e_y) must not exceed b + c_y; class: L_dec + s > L_k.
       G_exp: coordinatewise range <= 2 sqrt(b); then (in the walk) pairwise ||p - p_j||_2 <= 2 sqrt(b) and
       JS(p, p_j) <= d for every member j, and the bin-level ccm.guard.exp_bounds (gjs <= d, var <= b).
    b. SUFFICIENT check: the bin's current representative q* accepts the new row -> join, q* unchanged (it accepts every
       member).
    c. otherwise a certification attempt on members + row: the cheap start point (G: M / sum M; G_exp: member mean),
       then SLSQP (ccm.guard.g_try / exp_try), each checked against every member -> join with the new q*.
  At most ENGINEERING_GEOMETRY['f1_max_attempts_per_point'] certification attempts (c) per row; after that only
  sufficient checks (b) are used for the row (cap hits are counted; the cover stays certified, it is an upper bound).
  A row that cannot be certified even as a singleton is UNCOVERABLE.
  Reports: bins (members, witness representative), bin sizes, size distribution, compression ratio bins / rows.
F2 packing lower bound. Same order, per class: a row enters the packing iff it is closed-form pairwise infeasible with
  every row already in it. G: sum_k max(p_k, p'_k) > exp(d)(1 + 1e-12) (PROTOCOL; conservative B2.3). G_exp:
  ||p - p'||_2^2 > 4 b + m or JS(p, p') > d + m (m = 1e-9; max(KL(p||q), KL(p'||q)) >= their mean >= JS). Every packed
  row needs its own bin.
F3 capacity coverage (the go-rule F3):
  * K = 2 under G: the EXACT maximum capacity coverage (F3x) by dynamic programming (R4.7, R5.5). Each row's admissible
    set of s = q_1 is an interval [lo, hi] (NLL, both Brier labels, class; tightened construction targets), a bin is
    admissible iff max lo <= min hi (Helly in 1-D), the intervals are monotone in t = p_1, so an optimal code uses
    disjoint windows of rows sorted by t:  f[c][j] = max(f[c][j-1], f[c-1][l(j)-1] + j - l(j) + 1), with l(j) the
    leftmost feasible window start (two pointers). Every chosen window is then CERTIFIED by its midpoint
    representative q = (1 - s, s) checked against ALL members with the canonical predicate. F3 = the certified DP
    coverage; its representatives are the ones F4 uses. F3x_untightened (true d, b; diagnostic) is also reported.
  * otherwise (K = 6, and every G_exp unit): the greedy top-C of the disjoint F1 bins per class (largest first, ties by
    creation order) — a CONSTRUCTIVE LOWER BOUND on the best coverage. Canonical representatives of the selected bins
    = ccm.guard.bin_representative(members) (history independent; if not CERTIFIED the F1 witness is kept, flagged).
  F3_greedy (the greedy top-C) is always reported as a diagnostic.
F3u certified coverage upper bound (PROTOCOL section 5): N(r) = rows r' of r's class with sum_k max(p_k, p'_k) <=
  exp(d)(1 + 1e-12) (r included); F3u = (sum over classes of the cap largest N(r)) / rows, capped at 1. Under G_exp the
  neighbourhood is the G_exp closed form (the negation of the F2 G_exp test), never the G/NLL one (R7.4); it is a valid
  G_exp upper bound.
  Diagnostic (not registered): packing_coverage_bound = 1 - sum_c max(0, |F2 pack_c| - cap) / rows, also a valid upper
  bound (reported under F2).
F4 held-out fallback. REPRESENTATIVE CHOICE RULE (R8.4; deterministic, p-only): a held-out input is served by the FIRST
  representative, in the registered cover order, of its decision class that satisfies the CANONICAL predicate for it;
  otherwise it falls back to ccm.guard.fallback_release (Ucal itself, or the eta-nudge where the top is tied).
  Registered cover order of representatives = increasing cover-order rank of the bin's earliest member (= F1
  creation order).
  Reports the fallback rate, the released vectors (every one re-checked with the canonical predicate) and blocking
  categories per fallback row: NO_TOKEN_FOR_CLASS, NLL_BLOCKS (no representative passes NLL / KL, some passes Brier /
  sq), BRIER_BLOCKS (the converse), BOTH_BLOCK (none passes either), JOINT_BLOCK (each passes for some representative,
  never both), CLASS_OR_SIMPLEX (NLL and Brier pass together but class/simplex fails).
F5 decision-only (confidence-INELIGIBLE: no released vector) and CLASS eligibility (each whole class one admissible
  bin).
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from ccm import guard as GD

ENGINEERING_GEOMETRY = {
    "f1_max_attempts_per_point": 32,
    "neighbourhood_chunk": 512,
}


def _dk(d, b):
    return (float(GD.CONFIG["d"]) if d is None else float(d)), (float(GD.CONFIG["b"]) if b is None else float(b))


def _prep(P, dec):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("P must be (n, K)")
    dec = GD.decisions(P) if dec is None else np.asarray(dec, dtype=np.intp)
    if dec.shape != (P.shape[0],):
        raise ValueError("dec must be (n,)")
    return P, dec


def cover_order(P):
    """Fixed label-free order: decreasing max probability, then row index."""
    P = np.asarray(P, dtype=np.float64)
    return np.lexsort((np.arange(P.shape[0]), -P.max(axis=1)))


def cover_rank(P):
    """rank[r] = position of row r in the cover order."""
    o = cover_order(P)
    rank = np.empty(o.size, dtype=np.int64)
    rank[o] = np.arange(o.size)
    return rank


def size_distribution(sizes):
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        return {"n_bins": 0}
    edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 1 << 62]
    hist = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = int(np.sum((sizes >= lo) & (sizes < hi)))
        if k:
            hist[f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"] = k
    return {"n_bins": int(sizes.size), "max": int(sizes.max()), "mean": float(sizes.mean()),
            "median": float(np.median(sizes)), "singletons": int(np.sum(sizes == 1)), "hist": hist}


class _ClassBins:
    """Running bins of one predicted class (arrays grow geometrically)."""

    def __init__(self, K, contract):
        self.K, self.contract = K, contract
        self.n = 0
        self.A = np.empty((16, K))          # G: running max M              G_exp: running max
        self.B = np.empty((16, K))          # G: running c = min(p.p - 2p)  G_exp: running min
        self.Q = np.empty((16, K))          # current witness representative
        self.members = []
        self.gid = []

    def _grow(self):
        if self.n == self.A.shape[0]:
            for nm in ("A", "B", "Q"):
                old = getattr(self, nm)
                new = np.empty((2 * old.shape[0], self.K))
                new[: self.n] = old[: self.n]
                setattr(self, nm, new)

    def add(self, r, p, q, cp, gid):
        self._grow()
        j = self.n
        self.A[j] = p
        self.B[j] = cp if self.contract == "G" else p
        self.Q[j] = q
        self.members.append([r])
        self.gid.append(gid)
        self.n += 1

    def join(self, j, r, p, q, cp):
        self.A[j] = np.maximum(self.A[j], p)
        self.B[j] = np.minimum(self.B[j], cp if self.contract == "G" else p)
        if q is not None:
            self.Q[j] = q
        self.members[j].append(r)


def f1_cover(P, dec=None, contract="G", d=None, b=None, max_attempts_per_point=None):
    """F1 certified greedy cover (module docstring). Returns a dict with 'bins' (creation order; class, members, witness
    q, last method), 'bin_of_row' (-1 = uncoverable), 'uncoverable', 'order', 'summary' (JSON-able)."""
    d, b = _dk(d, b)
    P, dec = _prep(P, dec)
    if contract not in GD.CONTRACTS:
        raise ValueError(f"unknown contract {contract!r}")
    n, K = P.shape
    cap_att = ENGINEERING_GEOMETRY["f1_max_attempts_per_point"] if max_attempts_per_point is None \
        else int(max_attempts_per_point)
    tol = GD.ENGINEERING["prefilter_tol"]
    ed, emd, nb_ = math.exp(d), np.exp(-d), GD.nll_bound(d)
    two_sqrt_b = 2.0 * math.sqrt(b)
    t0 = time.process_time()
    order = cover_order(P)
    states = {}
    bin_list = []
    bin_of_row = np.full(n, -1, dtype=np.int64)
    method_of_bin = []
    uncoverable = []
    st_cnt = {"sufficient_accepts": 0, "attempts": 0, "attempt_accepts": 0, "cap_hits": 0,
              "accept_start": 0, "accept_slsqp": 0, "open_start": 0, "open_slsqp": 0}
    for r in order.tolist():
        p = P[r]
        c = int(dec[r])
        S = states.get(c)
        if S is None:
            S = states[c] = _ClassBins(K, contract)
        cp = np.sum(p * p) - 2.0 * p if contract == "G" else None
        joined = False
        nb = S.n
        if nb:
            if contract == "G":
                newM = np.maximum(S.A[:nb], p)
                idx = np.flatnonzero(np.sum(newM, axis=1) <= ed + tol)
                if idx.size:
                    L = emd * newM[idx]
                    sl = 1.0 - np.sum(L, axis=1)
                    newc = np.minimum(S.B[idx], cp)
                    val = np.sum(L * L, axis=1)[:, None] + 2.0 * sl[:, None] * L + (sl * sl)[:, None] \
                        - 2.0 * (L + sl[:, None])
                    keep = np.all(val <= newc + b + tol, axis=1)
                    if K > 1:
                        oth = np.delete(L, c, axis=1)
                        keep &= np.all((L[:, c] + sl)[:, None] > oth - tol, axis=1)
                    idx = idx[keep]
            else:
                rng_ = np.maximum(S.A[:nb], p) - np.minimum(S.B[:nb], p)
                idx = np.flatnonzero(np.all(rng_ <= two_sqrt_b + tol, axis=1))
            if idx.size:
                suff = GD.accepts(S.Q[idx], p, contract, d, b, c)
                attempts = 0
                for jj, j in enumerate(idx.tolist()):
                    if suff[jj]:
                        S.join(j, r, p, None, cp)
                        st_cnt["sufficient_accepts"] += 1
                        joined = True
                        break
                    if attempts >= cap_att:
                        if not suff[jj:].any():
                            break
                        if attempts == cap_att:
                            st_cnt["cap_hits"] += 1
                            attempts += 1
                        continue
                    mem = S.members[j]
                    Pm = np.vstack([P[mem], p[None, :]])
                    if contract == "G_exp":
                        Pmem = P[mem]
                        if not np.all(np.sum((Pmem - p) ** 2, axis=1) <= 4.0 * b + tol):
                            continue
                        if not np.all(GD.js_div(Pmem, p[None, :]) <= d + tol):
                            continue
                        v = GD.exp_bounds(Pm)
                        if v["gjs"] > d + tol or v["var"] > b + tol:
                            continue
                        attempts += 1
                        st_cnt["attempts"] += 1
                        q, meth, _, _ = GD.exp_try(Pm, c, d, b, order=("start", "slsqp"))
                    else:
                        newMj = np.maximum(S.A[j], p)
                        if not np.sum(newMj) <= nb_:
                            continue
                        attempts += 1
                        st_cnt["attempts"] += 1
                        q, meth, _, _ = GD.g_try(Pm, c, d, b, order=("start", "slsqp"),
                                                 stats=(newMj, emd * newMj, np.minimum(S.B[j], cp)))
                    if q is not None:
                        S.join(j, r, p, q, cp)
                        st_cnt["attempt_accepts"] += 1
                        st_cnt["accept_" + meth] += 1
                        method_of_bin[S.gid[j]] = meth
                        joined = True
                        break
        if joined:
            continue
        if contract == "G":
            q, meth, _, _ = GD.g_try(p[None, :], c, d, b, order=("start", "slsqp"))
        else:
            q, meth, _, _ = GD.exp_try(p[None, :], c, d, b, order=("start", "slsqp"))
        if q is None:
            uncoverable.append(r)
            continue
        st_cnt["open_" + meth] += 1
        gid = len(bin_list)
        S.add(r, p, q, cp, gid)
        bin_list.append((c, S.n - 1))
        method_of_bin.append(meth)
    bins = []
    for gid, (c, j) in enumerate(bin_list):
        S = states[c]
        mem = np.asarray(S.members[j], dtype=np.int64)
        bin_of_row[mem] = gid
        bins.append({"id": gid, "class": c, "members": mem, "size": int(mem.size), "q": S.Q[j].copy(),
                     "method": method_of_bin[gid]})
    cpu = time.process_time() - t0
    sizes = np.array([bb["size"] for bb in bins], dtype=np.int64)
    per_class = {}
    for c in sorted(set(dec.tolist())):
        sc = np.array([bb["size"] for bb in bins if bb["class"] == c], dtype=np.int64)
        nr = int(np.sum(dec == c))
        per_class[int(c)] = {"n_rows": nr, "n_bins": int(sc.size), "compression_ratio": (sc.size / nr) if nr else None,
                             "sizes": size_distribution(sc)}
    summary = {"metric": "F1_cover", "contract": contract, "d": d, "b": b, "n_rows": int(n), "n_bins": int(len(bins)),
               "compression_ratio": len(bins) / n if n else None, "n_uncoverable": len(uncoverable),
               "sizes": size_distribution(sizes), "per_class": per_class, "stats": st_cnt,
               "max_attempts_per_point": cap_att, "cpu_s": cpu}
    return {"bins": bins, "bin_of_row": bin_of_row, "uncoverable": np.asarray(uncoverable, dtype=np.int64),
            "order": order, "summary": summary, "contract": contract, "d": d, "b": b, "dec": dec}


def pairwise_infeasible(p, Q, contract="G", d=None, b=None):
    """Closed-form, conservative pairwise infeasibility of p with each row of Q (True = never in one bin)."""
    d, b = _dk(d, b)
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    if contract == "G":
        return np.sum(np.maximum(Q, p), axis=1) > GD.nll_bound(d)
    if contract == "G_exp":
        m = GD.ENGINEERING["exp_closed_form_abs_margin"]
        diff = Q - p
        return (np.sum(diff * diff, axis=1) > 4.0 * b + m) | (GD.js_div(Q, p[None, :]) > d + m)
    raise ValueError(f"unknown contract {contract!r}")


def f2_packing(P, dec=None, contract="G", d=None, b=None):
    """F2 greedy packing (pairwise infeasible rows, cover order, per class). Lower bound on the number of bins."""
    d, b = _dk(d, b)
    P, dec = _prep(P, dec)
    t0 = time.process_time()
    order = cover_order(P)
    packs = {}
    for c in sorted(set(dec.tolist())):
        rows = order[dec[order] == c]
        buf = np.empty((max(16, min(len(rows), 1024)), P.shape[1]))
        m = 0
        chosen = []
        for r in rows.tolist():
            p = P[r]
            if m == 0 or np.all(pairwise_infeasible(p, buf[:m], contract, d, b)):
                if m == buf.shape[0]:
                    nbuf = np.empty((2 * m, P.shape[1]))
                    nbuf[:m] = buf
                    buf = nbuf
                buf[m] = p
                m += 1
                chosen.append(r)
        packs[int(c)] = np.asarray(chosen, dtype=np.int64)
    per_class = {c: int(v.size) for c, v in packs.items()}
    summary = {"metric": "F2_packing", "contract": contract, "d": d, "b": b, "n_rows": int(P.shape[0]),
               "lower_bound_bins": int(sum(per_class.values())), "per_class": per_class,
               "cpu_s": time.process_time() - t0}
    return {"packing": packs, "summary": summary}


def packing_coverage_bound(f2, cap, n):
    """F3u_packing (registered before FEASIBILITY_LOCK; PROTOCOL section 5): every packed row needs its own bin and a
    class has at most `cap` bins, so at least |pack_c| - cap packed rows of class c stay uncovered:
    coverage <= 1 - sum_c max(0, |pack_c| - cap) / n. A valid upper bound on ANY admissible code, like the neighbourhood
    F3u; the registered bound is F3u_registered = min(F3u, F3u_packing)."""
    lost = sum(max(0, int(v) - int(cap)) for v in f2["summary"]["per_class"].values())
    return 1.0 - lost / n if n else None


def _min_bound(*vals):
    v = [float(x) for x in vals if x is not None]
    return min(v) if v else None


def _order_reps(reps, rep_cls, first_rank):
    o = np.lexsort((np.arange(len(first_rank)), np.asarray(first_rank)))
    return np.asarray(reps, dtype=np.float64)[o], np.asarray(rep_cls, dtype=np.intp)[o], o


def f3_capacity_coverage(P, cover, cap, served=True):
    """F3_greedy: at most `cap` F1 bins per class (largest first, ties by creation order); canonical representatives,
    returned in the registered cover order (creation order). A constructive lower bound on the best coverage."""
    P = np.asarray(P, dtype=np.float64)
    contract, d, b, dec = cover["contract"], cover["d"], cover["b"], cover["dec"]
    n = P.shape[0]
    t0 = time.process_time()
    sel = []
    per_class = {}
    for c in sorted(set(dec.tolist())):
        cb = [bb for bb in cover["bins"] if bb["class"] == c]
        cb.sort(key=lambda bb: (-bb["size"], bb["id"]))
        chosen = cb[: int(cap)]
        nr = int(np.sum(dec == c))
        cov = int(sum(bb["size"] for bb in chosen))
        per_class[int(c)] = {"n_rows": nr, "n_bins_available": len(cb), "n_selected": len(chosen),
                             "covered_rows": cov, "coverage": cov / nr if nr else None}
        sel.extend(chosen)
    sel.sort(key=lambda bb: bb["id"])                       # registered cover order
    reps, rep_cls, rep_ids, canon = [], [], [], []
    for bb in sel:
        q, cert = GD.bin_representative(P[bb["members"]], dec=bb["class"], contract=contract, d=d, b=b)
        canon.append(q is not None)
        reps.append(bb["q"] if q is None else q)
        rep_cls.append(bb["class"])
        rep_ids.append(bb["id"])
    reps = np.asarray(reps, dtype=np.float64).reshape(-1, P.shape[1])
    rep_cls = np.asarray(rep_cls, dtype=np.intp)
    covered = int(sum(bb["size"] for bb in sel))
    summary = {"metric": "F3_greedy", "wording": "constructive lower bound (greedy top-C of the disjoint F1 bins)",
               "contract": contract, "d": d, "b": b, "cap_per_class": int(cap), "n_rows": int(n),
               "covered_rows": covered, "coverage": covered / n if n else None, "n_selected": len(sel),
               "n_canonical_not_certified": int(len(canon) - sum(canon)), "per_class": per_class}
    if served:
        sv = served_mask(P, dec, reps, rep_cls, contract, d, b)
        summary["served_fraction_diagnostic"] = float(sv.mean()) if n else None
    summary["cpu_s"] = time.process_time() - t0
    return {"selected_ids": np.asarray(rep_ids, dtype=np.int64), "reps": reps, "rep_classes": rep_cls,
            "canonical": np.asarray(canon, dtype=bool), "summary": summary}


# --------------------------------------------------------------------------------------------- income exact (K = 2)

def income_intervals(P, dec, d=None, b=None, tightened=True):
    """K = 2: per row the admissible interval [lo, hi] of s = q_1 for q = (1 - s, s) (empty when lo > hi):
        NLL     s >= E p_1 f,  1 - s >= E p_0 f                      (f = 1 + nll_rel_margin if tightened, else 1)
        Brier   y = 0: 2 s^2 - 1 <= b' + c_0;  y = 1: 2 (1 - s)^2 - 1 <= b' + c_1,  c_y = p.p - 2 p_y
                (b' = b - brier_margin if tightened, else b)
        Class   dec = 1: 2 s - 1 >= m;  dec = 0: 1 - 2 s >= m      (m = class_margin if tightened, else 0)."""
    d, b = _dk(d, b)
    P = np.asarray(P, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.intp)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("income_intervals needs K = 2")
    E = np.exp(-d)
    eng = GD.ENGINEERING
    f = 1.0 + eng["nll_rel_margin"] if tightened else 1.0
    bb = b - eng["brier_margin"] if tightened else b
    m = eng["class_margin"] if tightened else 0.0
    pp = np.sum(P * P, axis=1)
    c0, c1 = pp - 2.0 * P[:, 0], pp - 2.0 * P[:, 1]
    r0, r1 = (1.0 + bb + c0) / 2.0, (1.0 + bb + c1) / 2.0
    with np.errstate(invalid="ignore"):
        hi_b = np.where(r0 >= 0, np.sqrt(np.maximum(r0, 0.0)), -np.inf)
        lo_b = np.where(r1 >= 0, 1.0 - np.sqrt(np.maximum(r1, 0.0)), np.inf)
    lo = np.maximum.reduce([E * P[:, 1] * f, lo_b, np.where(dec == 1, (1.0 + m) / 2.0, 0.0)])
    hi = np.minimum.reduce([1.0 - E * P[:, 0] * f, hi_b, np.where(dec == 0, (1.0 - m) / 2.0, 1.0)])
    return lo, hi


def _windows_dp(lo, hi, cap):
    """Rows already sorted by t. Exact max coverage by at most `cap` disjoint feasible windows. Returns (value, windows
    [(i, j) inclusive]) ; l(j) by two pointers with monotone deques (window max lo <= window min hi)."""
    n = len(lo)
    left = np.empty(n, dtype=np.int64)
    dq_lo, dq_hi = deque(), deque()
    i = 0
    for j in range(n):
        while dq_lo and lo[dq_lo[-1]] <= lo[j]:
            dq_lo.pop()
        dq_lo.append(j)
        while dq_hi and hi[dq_hi[-1]] >= hi[j]:
            dq_hi.pop()
        dq_hi.append(j)
        while i <= j and not (lo[dq_lo[0]] <= hi[dq_hi[0]]):
            i += 1
            while dq_lo and dq_lo[0] < i:
                dq_lo.popleft()
            while dq_hi and dq_hi[0] < i:
                dq_hi.popleft()
        left[j] = i                                # i > j: no feasible window ends at j
    C = int(cap)
    f = np.zeros((C + 1, n + 1), dtype=np.int64)
    for c in range(1, C + 1):
        for j in range(1, n + 1):
            best = f[c, j - 1]
            l = left[j - 1]
            if l <= j - 1:
                take = f[c - 1, l] + (j - l)
                if take > best:
                    best = take
            f[c, j] = best
    wins = []
    c, j = C, n
    while c > 0 and j > 0:
        if f[c, j] == f[c, j - 1]:
            j -= 1
            continue
        l = int(left[j - 1])
        wins.append((l, j - 1))
        c -= 1
        j = l
    wins.reverse()
    return int(f[C, n]), wins


def f3x_income(P, cap, dec=None, d=None, b=None):
    """F3 for K = 2 under G: exact DP coverage (tightened targets), every chosen window certified by its midpoint
    representative with the canonical predicate; representatives in the registered cover order."""
    d, b = _dk(d, b)
    P, dec = _prep(P, dec)
    t0 = time.process_time()
    n = P.shape[0]
    rank = cover_rank(P)
    lo, hi = income_intervals(P, dec, d, b, tightened=True)
    lo_u, hi_u = income_intervals(P, dec, d, b, tightened=False)
    reps, rep_cls, first_rank, sizes = [], [], [], []
    per_class = {}
    tot_x = tot_u = tot_cert = 0
    n_bins = n_cert = 0
    for c in sorted(set(dec.tolist())):
        rows = np.flatnonzero(dec == c)
        rows = rows[np.lexsort((rows, P[rows, 1]))]          # sorted by t = p_1, ties by row index
        val, wins = _windows_dp(lo[rows], hi[rows], cap)
        val_u, _ = _windows_dp(lo_u[rows], hi_u[rows], cap)
        cov_c = cert_c = 0
        for (i, j) in wins:
            mem = rows[i: j + 1]
            s = 0.5 * (float(np.max(lo[mem])) + float(np.min(hi[mem])))
            q = np.array([1.0 - s, s])
            ok = bool(np.all(GD.check_release(q[None, :], P[mem], d, b, c)["ok"]))
            n_bins += 1
            if ok:
                n_cert += 1
                cert_c += mem.size
                reps.append(q)
                rep_cls.append(c)
                first_rank.append(int(rank[mem].min()))
                sizes.append(int(mem.size))
            cov_c += mem.size
        nr = rows.size
        per_class[int(c)] = {"n_rows": int(nr), "F3x_rows": val, "F3x_untightened_rows": val_u, "n_windows": len(wins),
                             "certified_rows": cert_c, "coverage": cert_c / nr if nr else None}
        tot_x += val
        tot_u += val_u
        tot_cert += cert_c
    reps_o, cls_o, o = _order_reps(np.asarray(reps).reshape(-1, 2), rep_cls, first_rank)
    summary = {"metric": "F3_capacity_coverage", "method": "income_exact_dp", "contract": "G", "d": d, "b": b,
               "cap_per_class": int(cap), "n_rows": int(n), "F3x": tot_x / n if n else None,
               "F3x_untightened": tot_u / n if n else None, "covered_rows": int(tot_cert),
               "coverage": tot_cert / n if n else None, "n_bins": n_bins, "n_bins_certified": n_cert,
               "bin_sizes": size_distribution(np.asarray(sizes, dtype=np.int64)), "per_class": per_class,
               "cpu_s": time.process_time() - t0}
    return {"reps": reps_o, "rep_classes": cls_o, "first_rank": np.asarray(first_rank, dtype=np.int64)[o],
            "summary": summary}


# ----------------------------------------------------------------------------------------------------------- F3u

def neighbourhood_counts(P, dec=None, contract="G", d=None, b=None, chunk=None):
    """N(r) = rows r' of r's class (r included) NOT closed-form pairwise infeasible with r (G: sum_k max(p_k, p'_k) <=
    exp(d)(1 + 1e-12), summed in k order; G_exp: the negation of the G_exp test). Exact, chunked O(n_c^2 K)."""
    d, b = _dk(d, b)
    P, dec = _prep(P, dec)
    chunk = ENGINEERING_GEOMETRY["neighbourhood_chunk"] if chunk is None else int(chunk)
    N = np.zeros(P.shape[0], dtype=np.int64)
    bound = GD.nll_bound(d)
    m = GD.ENGINEERING["exp_closed_form_abs_margin"]
    for c in sorted(set(dec.tolist())):
        rows = np.flatnonzero(dec == c)
        Pc = P[rows]
        for s in range(0, rows.size, chunk):
            A = Pc[s: s + chunk]
            if contract == "G":
                acc = np.zeros((A.shape[0], Pc.shape[0]))
                for k in range(P.shape[1]):
                    acc += np.maximum(A[:, k, None], Pc[None, :, k])
                ok = acc <= bound
            elif contract == "G_exp":
                l2 = np.zeros((A.shape[0], Pc.shape[0]))
                for k in range(P.shape[1]):
                    df = Pc[None, :, k] - A[:, k, None]
                    l2 += df * df
                ok = ~(l2 > 4.0 * b + m)
                ii, jj = np.nonzero(ok)
                if ii.size:
                    ok[ii, jj] = ~(GD.js_div(A[ii], Pc[jj]) > d + m)
            else:
                raise ValueError(f"unknown contract {contract!r}")
            N[rows[s: s + chunk]] = ok.sum(axis=1)
    return N


def f3u_coverage_bound(P, cap, dec=None, contract="G", d=None, b=None):
    """F3u (PROTOCOL section 5): upper bound on the coverage ANY admissible code with at most `cap` bins per class can
    reach. F3u = (sum over classes of the sum of the `cap` largest N(r)) / rows, capped at 1 (registered). Per class,
    the same sum capped at the class's rows is reported as a diagnostic."""
    d, b = _dk(d, b)
    P, dec = _prep(P, dec)
    t0 = time.process_time()
    N = neighbourhood_counts(P, dec, contract, d, b)
    total = 0
    per_class = {}
    for c in sorted(set(dec.tolist())):
        Nc = np.sort(N[dec == c])[::-1]
        top = int(Nc[: int(cap)].sum())
        total += top
        nr = int(Nc.size)
        per_class[int(c)] = {"n_rows": nr, "sum_top_cap_N": top, "class_bound_diagnostic": min(top, nr) / nr,
                             "max_N": int(Nc[0]) if nr else 0}
    n = P.shape[0]
    summary = {"metric": "F3u_coverage_bound", "contract": contract,
               "neighbourhood": "G closed form (NLL)" if contract == "G" else "G_exp closed form (L2 and JS)",
               "d": d, "b": b, "cap_per_class": int(cap), "n_rows": int(n), "sum_top_cap_N": int(total),
               "F3u": min(1.0, total / n) if n else None, "per_class": per_class, "cpu_s": time.process_time() - t0}
    return {"N": N, "summary": summary}


# ------------------------------------------------------------------------------------------------------------ F4

def _check_matrix(P, dec_c, Qc, contract, d, b):
    return GD.check(Qc[None, :, :], P[:, None, :], contract, d, b, dec_c)


def served_mask(P, dec, reps, rep_classes, contract="G", d=None, b=None):
    """Rows of P for which SOME representative of the row's decision class satisfies the canonical predicate."""
    d, b = _dk(d, b)
    P, dec = _prep(P, dec)
    reps = np.asarray(reps, dtype=np.float64).reshape(-1, P.shape[1])
    rep_classes = np.asarray(rep_classes, dtype=np.intp)
    out = np.zeros(P.shape[0], dtype=bool)
    for c in sorted(set(dec.tolist())):
        rows = np.flatnonzero(dec == c)
        Qc = reps[rep_classes == c]
        if Qc.shape[0] == 0:
            continue
        for s in range(0, rows.size, 4096):
            rr = rows[s: s + 4096]
            out[rr] = _check_matrix(P[rr], c, Qc, contract, d, b)["ok"].any(axis=1)
    return out


def f4_heldout_fallback(P_new, reps, rep_classes, dec_new=None, contract="G", d=None, b=None):
    """F4 (module docstring): FIRST representative in the registered order (the order of `reps`) of the row's class that
    satisfies the canonical predicate, else ccm.guard.fallback_release. Returns fallback flags, served representative
    index (-1 = fallback), the released vectors, blocking categories and the summary."""
    d, b = _dk(d, b)
    P_new, dec_new = _prep(P_new, dec_new)
    reps = np.asarray(reps, dtype=np.float64).reshape(-1, P_new.shape[1])
    rep_classes = np.asarray(rep_classes, dtype=np.intp)
    t0 = time.process_time()
    n = P_new.shape[0]
    a, bb_ = ("nll", "brier") if contract == "G" else ("kl", "sq")
    served_by = np.full(n, -1, dtype=np.int64)
    cat = np.array([""] * n, dtype=object)
    for c in sorted(set(dec_new.tolist())):
        rows = np.flatnonzero(dec_new == c)
        ridx = np.flatnonzero(rep_classes == c)
        if ridx.size == 0:
            cat[rows] = "NO_TOKEN_FOR_CLASS"
            continue
        Qc = reps[ridx]
        for s in range(0, rows.size, 4096):
            rr = rows[s: s + 4096]
            ch = _check_matrix(P_new[rr], c, Qc, contract, d, b)
            ok = ch["ok"]
            anyok = ok.any(axis=1)
            served_by[rr[anyok]] = ridx[np.argmax(ok[anyok], axis=1)]
            fb = ~anyok
            any_a = ch[a].any(axis=1)
            any_b = ch[bb_].any(axis=1)
            any_ab = (ch[a] & ch[bb_]).any(axis=1)
            k = np.where(~any_a & any_b, "NLL_BLOCKS",
                         np.where(any_a & ~any_b, "BRIER_BLOCKS",
                                  np.where(~any_a & ~any_b, "BOTH_BLOCK",
                                           np.where(~any_ab, "JOINT_BLOCK", "CLASS_OR_SIMPLEX"))))
            cat[rr[fb]] = k[fb]
    fallback = served_by < 0
    release = np.empty_like(P_new)
    release[~fallback] = reps[served_by[~fallback]]
    fb_info = {"n": 0, "n_tied": 0, "n_fail_G": 0}
    if fallback.any():
        release[fallback], fb_info = GD.fallback_release(P_new[fallback], dec_new[fallback], d=d, b=b)
    rel_ok = GD.check(release, P_new, contract, d, b, dec_new)["ok"]
    cats = {}
    for k in ("NO_TOKEN_FOR_CLASS", "NLL_BLOCKS", "BRIER_BLOCKS", "BOTH_BLOCK", "JOINT_BLOCK", "CLASS_OR_SIMPLEX"):
        cnt = int(np.sum(cat[fallback] == k))
        if cnt:
            cats[k] = cnt
    per_class = {}
    for c in sorted(set(dec_new.tolist())):
        msk = dec_new == c
        per_class[int(c)] = {"n": int(msk.sum()), "fallback": int(fallback[msk].sum()),
                             "rate": float(fallback[msk].mean())}
    summary = {"metric": "F4_heldout_fallback", "contract": contract, "d": d, "b": b, "n": int(n),
               "choice_rule": "first representative in registered cover order satisfying the canonical predicate",
               "n_reps": int(reps.shape[0]), "fallback": int(fallback.sum()),
               "fallback_rate": float(fallback.mean()) if n else None, "blocking": cats, "per_class": per_class,
               "fallback_release": {"n_tied": fb_info["n_tied"], "eta": GD.CONFIG["fallback_eta"]},
               "n_released_failing_contract": int(np.sum(~rel_ok)), "cpu_s": time.process_time() - t0}
    return {"fallback": fallback, "served_by": served_by, "release": release, "category": cat, "summary": summary}


# ------------------------------------------------------------------------------------------------------------ F5

def f5_class_eligibility(P, dec=None, contract="G", d=None, b=None):
    """F5 CLASS: each predicted class as ONE bin; eligible iff every class is CERTIFIED (bin_representative)."""
    d, b = _dk(d, b)
    P, dec = _prep(P, dec)
    per = {}
    reps = {}
    for c in sorted(set(dec.tolist())):
        q, cert = GD.bin_representative(P[dec == c], dec=c, contract=contract, d=d, b=b)
        per[int(c)] = {k: v for k, v in cert.items() if k not in ("solver",)}
        if q is not None:
            reps[int(c)] = q
    eligible = all(v["status"] == "CERTIFIED" for v in per.values())
    return {"summary": {"metric": "F5_CLASS", "contract": contract, "eligible": bool(eligible), "per_class": per},
            "reps": reps}


def decision_only_record(contract="G"):
    """F5 premise: the decision-only release carries no confidence vector -> confidence-INELIGIBLE under G / G_exp."""
    return {"metric": "F5_decision_only", "contract": contract, "eligible": False,
            "reason": "no released vector q exists; the contract requires q_k >= exp(-d) p_k etc. for every input"}


def run_geometry(P_fit, P_held, cap, dec_fit=None, dec_held=None, contract="G", d=None, b=None):
    """F1-F5 for one (seed, recipient, contract). JSON-able summaries (keys starting with '_' hold private arrays).
    go_inputs: F3_coverage (K = 2 under G: certified exact DP; otherwise greedy constructive lower bound),
    F4_fallback_rate, F3u, F3x (K = 2 under G only, else None)."""
    d, b = _dk(d, b)
    P_fit, dec_fit = _prep(P_fit, dec_fit)
    P_held, dec_held = _prep(P_held, dec_held)
    K = P_fit.shape[1]
    t0 = time.process_time()
    f1 = f1_cover(P_fit, dec_fit, contract, d, b)
    f2 = f2_packing(P_fit, dec_fit, contract, d, b)
    f3g = f3_capacity_coverage(P_fit, f1, cap)
    f3x = f3x_income(P_fit, cap, dec_fit, d, b) if (K == 2 and contract == "G") else None
    f3 = f3x if f3x is not None else f3g
    f3u = f3u_coverage_bound(P_fit, cap, dec_fit, contract, d, b)
    f4 = f4_heldout_fallback(P_held, f3["reps"], f3["rep_classes"], dec_held, contract, d, b)
    f5 = f5_class_eligibility(P_fit, dec_fit, contract, d, b)
    f3s = dict(f3["summary"])
    if f3x is None:
        f3s["metric"] = "F3_capacity_coverage"
    f2s = dict(f2["summary"])
    f2s["packing_coverage_bound_diagnostic"] = packing_coverage_bound(f2, cap, P_fit.shape[0])
    return {"contract": contract, "K": int(K), "cap_per_class": int(cap),
            "n_tied_fit": int(GD.tied_top(P_fit).sum()), "n_tied_held": int(GD.tied_top(P_held).sum()),
            "F1": f1["summary"], "F2": f2s, "F3": f3s, "F3_greedy": f3g["summary"], "F3u": f3u["summary"],
            "F4": f4["summary"], "F5": f5["summary"], "F5_decision_only": decision_only_record(contract),
            "go_inputs": {"F3_coverage": f3s["coverage"], "F4_fallback_rate": f4["summary"]["fallback_rate"],
                          "F3u": f3u["summary"]["F3u"],
                          "F3u_packing": f2s["packing_coverage_bound_diagnostic"],
                          "F3u_registered": _min_bound(f3u["summary"]["F3u"], f2s["packing_coverage_bound_diagnostic"]),
                          "F3x": None if f3x is None else f3x["summary"]["F3x"]},
            "cpu_s": time.process_time() - t0,
            "_f1": f1, "_f3": f3, "_f3_greedy": f3g, "_f3u": f3u, "_f4": f4}
