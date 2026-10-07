"""Learned-decoder mapper for the lcr study (role C): constrained and weighted coarse assignment search with the D1
decoder (prompt sections 7 and 8). Every frozen rule is a module constant below and in SEARCH_RULES.json.

Source of the search engine: qpc/compress.py (pinned at 7f3ec67, imported for canonicalisation helpers only, never
edited). Its sufficient-statistic tables, canonical labels, exact n log n plug-in MI and the vectorised single-cell
move deltas are re-implemented here for the new objective; the qpc greedy merge-to-cap phase is NOT used (every
proposal here moves a WHOLE fine cell to an existing same-class token; tokens can empty, never be created).

Fitting quantities on the N fitting rows (OSF_DEFENSE_FIT), recipient i in {1, 2}, token t with fitting count n_t,
label counts y_t, teacher sums s_t (canonical accumulation over member fine cells in increasing index = qpc
token_tables = lcr.decoder.token_stats, bitwise), SEX counts:
    q_t   = lcr.decoder.solve_batch(y_t, s_t, n_t, d_t)  (the ACTUAL released, eps-smoothed, class-dominant vector)
    L_i   = (1/N) sum_t sum_k y_t[k] * -log clip(q_t[k], 1e-12, 1)          (true-label log loss, nats)
    B_i   = (1/N) sum_t sum_k y_t[k] * sum_j (q_t[j] - [j = k])^2           (source multiclass Brier)
    I_i   = I(SEX; token_i), I12 = I(SEX; (token_1, token_2))  plug-in MI of the exact integer count tables
    T     = L_1 + L_2 + 0.5 (B_1 + B_2)            Phi = I12 + 0.5 (I1 + I2)
State values are "exact": each is a sum of the sorted NONZERO per-token (per-cell) contributions, so it is a function of
the partition only (not of the search path or slot labels). Candidate deltas are vectorised difference formulas; an
accepted move is re-checked on the exact state values and reverted if the exact check disagrees.

Arms (objective; enforced constraints; search):
    C-TASK          T, per recipient separately; none (budgets recorded); no SEX enters the search
    W-LOCAL(lam)    T + lam (I1 + I2)/2, per recipient; none            (weighted control of K-LOCAL)
    W-SEQ-ab(lam)   T + lam Phi, sequential a then b; none              (weighted control of K-SEQ-ab)
    W-JOINT(lam)    T + lam Phi, joint single moves; none               (weighted control of K-JOINT-SINGLE)
    K-LOCAL         I_i per recipient; own budgets + own local cap
    K-SEQ-ab        Phi; stage 1: recipient a vs the partner's CLASS-ONLY view, ONLY a's constraints (the temporary
                    partner is never required feasible and never replaced by a constant); a frozen; stage 2: b under
                    Phi with frozen a, b's constraints; the FINAL release must satisfy both recipients' constraints
    K-JOINT-SINGLE  Phi; both budgets + both local caps; joint single moves
    K-JOINT-PAIR    K-JOINT-SINGLE + one atomic paired step after each single sweep
Budgets (one profile): L_i <= L_i(U) + 0.005, B_i <= B_i(U) + 0.003 (U = continuous teacher on the same rows);
local caps I_i <= I_i(C-TASK) of the same seed (refs, exact bits); decision preservation and state caps 8/64 per
predicted class are structural (tokens never cross classes; tokens are never created; B's decoder certifies argmax).

    fit_unit(arm, fine_dict, T, tr, Y_fit, S_fit, lam=None, starts=None, witnesses=None, refs=None, meta=None)
        -> (record, {"policy.json": dict, "decoder.json": dict, "release.npz": dict of arrays})
    refs_from_ctask(ctask_record) -> refs for the K- arms
    python -m lcr.mapper rules [--out SEARCH_RULES.json]     print / write the frozen rules
    python -m lcr.mapper timing [--out TIMING.json] [--seeds 0]   SYNTHETIC full-bank timing (under lcr.sema)
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from dpc.compress import mi_plugin
from dpc.utility import per_row
from lcr import run as RN
from qpc import compress as QC
from qpc import kmeans as KM
from qpc import partition as PT
from qpc import release as RL

SCHEMA = "lcr-mapper-v1"
ARMS = ("C-TASK", "W-LOCAL", "W-SEQ-12", "W-SEQ-21", "W-JOINT", "K-LOCAL", "K-SEQ-12", "K-SEQ-21", "K-JOINT-SINGLE",
        "K-JOINT-PAIR")
# ------------------------------------------------------------------ frozen rules (mirrored in SEARCH_RULES.json)
KAPPA = 32.0
EPS = 1e-12
LOSS_CLIP = 1e-12
CAPS = (8, 64)                     # states per teacher-predicted class (income, occupation); upper bounds
BUDGET = {"ll": 0.005, "brier": 0.003}
TASK_BRIER_WEIGHT = 0.5            # T = L1 + L2 + 0.5 (B1 + B2)
PHI_LOCAL_WEIGHT = 0.5             # Phi = I12 + 0.5 (I1 + I2)
TOL = 1e-12                        # strict improvement: candidate delta < -TOL and exact new objective < exact old
TIE_TOL = 1e-12                    # accept-choice ties: delta within TIE_TOL of the best -> lowest canonical target
BUDGET_MARGIN = 1e-10              # search feasibility: L_i <= limit - 1e-10, B_i <= limit - 1e-10 (exact state values)
CAP_MARGIN = 0.0                   # search feasibility: I_i <= I_i(C-TASK) - 0 (same exact MI function, same bits)
DEPLOYED_TOL = 0.0                 # final deployed (row-level) budgets and caps: value <= limit, no tolerance
SWEEPS = 5
N_NEAREST = 4                      # per fine cell: 4 nearest targets by KL(cell mean || target released q)
N_BEST = 4                         # per fine cell: 4 targets with the best exact objective change
PAIR_KEEP_PHI = 4                  # paired step: per recipient, 4 retained by exact one-sided Phi change ...
PAIR_KEEP_TASK = 4                 # ... and 4 more by exact task-loss change (L_i + 0.5 B_i)
PAIR_STEP = "after_each_single_sweep"
PAIR_ACCEPT_PER_STEP = 1
EVAL_CEILING = 8_000_000           # TOTAL proposal evaluations per unit (every arm; equal for SINGLE, PAIR, both SEQ)
PARITY_ATOL = 1e-10                # |state term - deployed row-level term| for L, B, I, I12
MI_PARITY_ATOL = 1e-12             # exact table MI vs dpc mi_plugin on the deployed tokens
SPEC = {
    "C-TASK": {"kind": "local", "w": (1.0, TASK_BRIER_WEIGHT, 0.0, 0.0), "constrained": False, "sex": False},
    "W-LOCAL": {"kind": "local", "w": "lam_local", "constrained": False, "sex": True},
    "W-SEQ-12": {"kind": "seq", "order": (1, 2), "w": "lam_phi", "constrained": False, "sex": True},
    "W-SEQ-21": {"kind": "seq", "order": (2, 1), "w": "lam_phi", "constrained": False, "sex": True},
    "W-JOINT": {"kind": "joint", "w": "lam_phi", "constrained": False, "pair": False, "sex": True},
    "K-LOCAL": {"kind": "local", "w": (0.0, 0.0, 1.0, 0.0), "constrained": True, "sex": True},
    "K-SEQ-12": {"kind": "seq", "order": (1, 2), "w": (0.0, 0.0, PHI_LOCAL_WEIGHT, 1.0), "constrained": True,
                 "sex": True},
    "K-SEQ-21": {"kind": "seq", "order": (2, 1), "w": (0.0, 0.0, PHI_LOCAL_WEIGHT, 1.0), "constrained": True,
                 "sex": True},
    "K-JOINT-SINGLE": {"kind": "joint", "w": (0.0, 0.0, PHI_LOCAL_WEIGHT, 1.0), "constrained": True, "pair": False,
                       "sex": True},
    "K-JOINT-PAIR": {"kind": "joint", "w": (0.0, 0.0, PHI_LOCAL_WEIGHT, 1.0), "constrained": True, "pair": True,
                     "sex": True}}


def weights(arm, lam):
    w = SPEC[arm]["w"]
    if w == "lam_local":
        return (1.0, TASK_BRIER_WEIGHT, lam / 2.0, 0.0)
    if w == "lam_phi":
        return (1.0, TASK_BRIER_WEIGHT, lam * PHI_LOCAL_WEIGHT, lam)
    return w


def objective(t, w):
    """wL (L1 + L2) + wB (B1 + B2) + wI (I1 + I2) + w12 I12 (zero-weight terms are not read)."""
    v = 0.0
    if w[0]:
        v += w[0] * (t["L1"] + t["L2"])
    if w[1]:
        v += w[1] * (t["B1"] + t["B2"])
    if w[2]:
        v += w[2] * (t["I1"] + t["I2"])
    if w[3]:
        v += w[3] * t["I12"]
    return float(v)


def config_id(arm, lam=None):
    if arm == "C-TASK":
        return RN.ctask_id()
    if arm.startswith("W-"):
        return RN.weighted_id(arm[2:], lam)
    return RN.constrained_id(arm[2:])


def registered_starts(arm, lam=None):
    """The registered start (local/seq) or witness (joint) config IDs, in their fixed tie order."""
    src_priv = [RN.d0_id(f, l) for l in RN.LAMS for f in RN.PRIVACY]
    if arm == "C-TASK":
        return [RN.d0_id("FINE-TASK")]
    if arm in ("K-LOCAL", "W-LOCAL"):
        return [RN.ctask_id()]
    if arm[2:] in ("SEQ-12", "SEQ-21"):
        return [RN.ctask_id()] + [RN.d0_id(arm[2:], l) for l in RN.LAMS]
    if arm in ("K-JOINT-SINGLE", "K-JOINT-PAIR"):
        return [RN.ctask_id(), RN.constrained_id("LOCAL"), RN.constrained_id("SEQ-12"),
                RN.constrained_id("SEQ-21"), RN.d0_id("FINE-TASK")] + src_priv
    if arm == "W-JOINT":
        return [RN.ctask_id(), RN.weighted_id("LOCAL", lam), RN.weighted_id("SEQ-12", lam),
                RN.weighted_id("SEQ-21", lam), RN.d0_id("FINE-TASK")] + src_priv
    raise ValueError(arm)


def required_starts(arm, lam=None):
    """Slots that must be present even in fixture mode (source maps may be a subset there)."""
    reg = registered_starts(arm, lam)
    if arm == "C-TASK":
        return reg
    if SPEC[arm]["kind"] == "joint":
        return reg[:4]
    return reg[:1]


def _dec():
    from lcr import decoder
    return decoder


# ------------------------------------------------------------------ helpers
def _ssum(x):
    """Order-independent sum of the NONZERO entries (sorted, then numpy pairwise): a function of the multiset."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[x != 0.0]
    return float(np.sort(x).sum()) if x.size else 0.0


def _pair_labels(z, fines):
    pp = z if isinstance(z, RL.PolicyPair) else RL.PolicyPair.from_dict(z)
    fps = (fines[1].fingerprint(), fines[2].fingerprint())
    if (pp.p1.fine.fingerprint(), pp.p2.fine.fingerprint()) != fps:
        raise ValueError("start/witness map was fitted on different fine partitions")
    return {1: QC.labels_from_policy(pp.p1), 2: QC.labels_from_policy(pp.p2)}, pp.fingerprint()


# ------------------------------------------------------------------ fixed problem (fitting rows)
class Problem:
    """Fixed per-unit data: fine partitions, deployed fine cells of the fitting rows, per-cell sufficient statistics
    (n, label counts, canonical teacher sums = fine.S), SEX tables (optional), U losses, budgets and caps."""

    def __init__(self, fine1, fine2, P, dd, Y, s, caps, budget, dec):
        self.fine = {1: fine1, 2: fine2}
        self.dec = dec
        self.N = int(P[1].shape[0])
        self.K = {r: int(self.fine[r].K) for r in (1, 2)}
        self.cell, self.ncell, self.Ycell, self.Spad, self.Ypad, self.npad = {}, {}, {}, {}, {}, {}
        self.crange, self.mean = {}, {}
        for r in (1, 2):
            fine = self.fine[r]
            cell = KM.assign_fine(P[r], dd[r], fine)
            n = np.bincount(cell, minlength=fine.F).astype(np.int64)
            if not np.array_equal(n, fine.n):
                raise ValueError(f"recipient {r}: deployed fine-cell counts on the fitting rows differ from the fine "
                                 "partition's n (wrong rows or partition)")
            self.cell[r] = cell
            self.ncell[r] = n
            Yc = dec.cell_label_counts(fine, P[r], dd[r], Y[r]).astype(np.float64)
            self.Ycell[r] = Yc
            K = fine.K
            self.Spad[r] = np.vstack([np.asarray(fine.S, dtype=np.float64), np.zeros((1, K))])
            self.Ypad[r] = np.vstack([Yc, np.zeros((1, K))])
            self.npad[r] = np.concatenate([n, [0]]).astype(np.int64)
            self.crange[r] = {c: (int(fine.cells_of(c)[0]), int(fine.cells_of(c)[-1]) + 1) for c in range(K)}
            self.mean[r] = np.asarray(fine.mean, dtype=np.float64)
        self.y = {r: np.asarray(Y[r], dtype=np.int64) for r in (1, 2)}
        self.P, self.dd = P, dd
        self.caps = {1: int(caps[0]), 2: int(caps[1])}
        self.budget = dict(budget)
        self.LU, self.BU = {}, {}
        for r in (1, 2):
            pr = per_row(P[r], self.y[r], self.K[r])
            self.LU[r], self.BU[r] = float(pr["ll"].mean()), float(pr["br"].mean())
        self.limL = {r: self.LU[r] + self.budget["ll"] for r in (1, 2)}
        self.limB = {r: self.BU[r] + self.budget["brier"] for r in (1, 2)}
        self.capI = {1: None, 2: None}
        self.s = None
        XL = np.arange(self.N + 1, dtype=np.float64)
        self.XL = XL * np.log(np.where(XL > 0, XL, 1.0))
        if s is not None:
            self.set_sex(s)

    def set_sex(self, s):
        s = np.asarray(s, dtype=np.int64)
        self.s = s
        F1, F2 = self.fine[1].F, self.fine[2].F
        idx = (s * F1 + self.cell[1]) * F2 + self.cell[2]
        self.Tfine = np.bincount(idx, minlength=2 * F1 * F2).reshape(2, F1, F2).astype(np.int64)
        self.tf = {1: self.Tfine.sum(2), 2: self.Tfine.sum(1)}
        ns = np.bincount(s, minlength=2)
        self.const_mi = float(-self.XL[ns].sum() + self.XL[self.N])

    def phi(self, X):
        return self.XL[X[0]] + self.XL[X[1]] - self.XL[X[0] + X[1]]


# ------------------------------------------------------------------ search state
class State:
    """Both recipients' coarse maps. Slots = the canonical token IDs of the starting map (contiguous per class);
    slots may empty, never appear. Per slot: members (sorted fine cells), n, Y, S (canonical fold), released q, token
    loss totals (ll, br), canonical index (lowest member). Exact integer SEX tables Tr (2, slots) and T12."""

    def __init__(self, pb, labels, sex=True, counters=None):
        self.pb = pb
        self.sex = bool(sex and pb.s is not None)
        self.ct = counters if counters is not None else new_counters()
        self.lab, self.members, self.cls, self.srange = {}, {}, {}, {}
        self.tn, self.tY, self.tS, self.tQ, self.tll, self.tbr = {}, {}, {}, {}, {}, {}
        self.minm, self.Mpad, self.mlen, self.memo, self.ver = {}, {}, {}, {}, {}
        self._vc = 0
        for r in (1, 2):
            fine = pb.fine[r]
            tok = RL.canonical_tokens(fine, np.asarray(labels[r]))
            S = int(tok.max()) + 1
            mem = [[] for _ in range(S)]
            for f in range(fine.F):
                mem[int(tok[f])].append(f)
            cls = np.array([int(fine.cell_class[m[0]]) for m in mem], dtype=np.int64)
            for j, m in enumerate(mem):
                if np.unique(fine.cell_class[m]).size != 1:
                    raise ValueError(f"recipient {r}: a token mixes predicted classes")
            for c in range(fine.K):
                if int(np.sum(cls == c)) > pb.caps[r]:
                    raise ValueError(f"recipient {r}: class {c} has {int(np.sum(cls == c))} tokens > cap {pb.caps[r]}")
            self.lab[r] = tok.astype(np.int64)
            self.members[r] = mem
            self.cls[r] = cls
            self.srange[r] = {c: (int(np.flatnonzero(cls == c)[0]), int(np.flatnonzero(cls == c)[-1]) + 1)
                              for c in range(fine.K)}
            self.mlen[r] = np.array([len(m) for m in mem], dtype=np.int64)
            W = int(self.mlen[r].max()) + 1
            self.Mpad[r] = np.full((S, max(W, 2)), fine.F, dtype=np.int64)
            for j, m in enumerate(mem):
                self.Mpad[r][j, :len(m)] = m
            self.minm[r] = np.array([m[0] for m in mem], dtype=np.int64)
            self.ver[r] = np.arange(S, dtype=np.int64) + self._vc
            self._vc += S
            K = fine.K
            self.memo[r] = {}
            for c in range(K):
                c0, c1 = pb.crange[r][c]
                s0, s1 = self.srange[r][c]
                Fc, Sc = c1 - c0, s1 - s0
                self.memo[r][c] = {"ver": np.full((Fc, Sc), -1, dtype=np.int64),
                                   "n": np.full((Fc, Sc), -1, dtype=np.int64), "Y": np.zeros((Fc, Sc, K)),
                                   "S": np.zeros((Fc, Sc, K)), "Q": np.zeros((Fc, Sc, K)), "ll": np.zeros((Fc, Sc)),
                                   "br": np.zeros((Fc, Sc))}
            idx = self._pad_rows(r, np.arange(S))
            n, Y, Ssum = self._fold(r, idx)
            Q, ll, br = self._solve(r, Y, Ssum, n, cls)
            self.tn[r], self.tY[r], self.tS[r], self.tQ[r], self.tll[r], self.tbr[r] = n, Y, Ssum, Q, ll, br
        if self.sex:
            self.Tr = {}
            for r in (1, 2):
                S = len(self.members[r])
                X = np.zeros((2, S), dtype=np.int64)
                for s_ in (0, 1):
                    X[s_] = np.bincount(self.lab[r], weights=pb.tf[r][s_], minlength=S).astype(np.int64)
                self.Tr[r] = X
            S1, S2 = len(self.members[1]), len(self.members[2])
            A = np.zeros((2, S1, pb.fine[2].F), dtype=np.int64)
            np.add.at(A, (slice(None), self.lab[1]), pb.Tfine)
            B = np.zeros((2, S1, S2), dtype=np.int64)
            np.add.at(B.transpose(0, 2, 1), (slice(None), self.lab[2]), A.transpose(0, 2, 1))
            self.T12 = B

    # -------------------------------------------------------------- sufficient statistics and solves
    def _pad_rows(self, r, slots):
        L = int(self.mlen[r][slots].max()) if len(slots) else 1
        return self.Mpad[r][slots, :max(L, 1)]

    def _fold(self, r, idx):
        """Canonical sums over member cells (sorted, PAD rows are zeros): bitwise lcr.decoder.token_stats."""
        pb = self.pb
        n = pb.npad[r][idx].sum(1)
        Y = pb.Ypad[r][idx[:, 0]]
        S = pb.Spad[r][idx[:, 0]]
        for t in range(1, idx.shape[1]):
            Y = Y + pb.Ypad[r][idx[:, t]]
            S = S + pb.Spad[r][idx[:, t]]
        return n.astype(np.int64), Y, S

    def _solve(self, r, Y, S, n, d):
        """Released q and token loss totals; n = 0 rows get zeros (empty / fallback: no fitting rows, no loss)."""
        M, K = Y.shape
        Q = np.zeros((M, K))
        ll = np.zeros(M)
        br = np.zeros(M)
        sup = np.flatnonzero(n > 0)
        if sup.size:
            _, Qs, _, _ = self.pb.dec.solve_batch(Y[sup], S[sup], n[sup], d[sup])
            Q[sup] = Qs
            l2, b2 = self.pb.dec.token_losses(Y[sup], Qs)
            ll[sup], br[sup] = l2, b2
            self.ct["solves"] += int(sup.size)
            self.ct["solve_calls"] += 1
        return Q, ll, br

    def _toggle_idx(self, r, f, slots):
        """Member index rows of (slot XOR {f}) for each slot, sorted, padded with F (zero row)."""
        F = self.pb.fine[r].F
        L = int(self.mlen[r][slots].max()) if len(slots) else 1
        base = self.Mpad[r][slots, :L]
        X = np.concatenate([base, np.full((len(slots), 1), f, dtype=np.int64)], 1)
        inside = self.lab[r][f] == np.asarray(slots)
        if inside.any():
            row = np.flatnonzero(inside)
            X[row] = np.where(X[row] == f, F, X[row])
        X.sort(1)
        return X

    def _fold_pairs(self, r, cells, slots):
        """Canonical statistics of (slot XOR {cell}) for aligned arrays of cells and slots."""
        F = self.pb.fine[r].F
        L = int(self.mlen[r][slots].max()) if slots.size else 1
        X = np.empty((slots.size, L + 1), dtype=np.int64)
        X[:, :L] = self.Mpad[r][slots, :L]
        X[:, L] = cells
        inside = self.lab[r][cells] == slots
        if inside.any():
            sub = X[inside]
            sub[sub == cells[inside][:, None]] = F
            X[inside] = sub
        X.sort(1)
        return self._fold(r, X)

    def ensure(self, r, c, cells, slots, verify=False):
        """Memo entries for every (cell in cells) x (slot in slots) of class c. An entry = the D1 solve of the canonical
        statistics of (slot XOR {cell}); it is filled from the canonical fold and indexed by the slot's member-set
        version id (a fresh id on every membership change; undo restores the old id). verify=True (every applied move,
        and the tests) re-folds the requested entries and requires BITWISE equality of (n, Y, S) with the stored
        entry, so a solve is never reused for different label counts or teacher sums.
        Returns (n, Y, S, Q, ll, br) with shape (len(cells), len(slots), ...)."""
        pb = self.pb
        K = pb.K[r]
        c0 = pb.crange[r][c][0]
        s0 = self.srange[r][c][0]
        cells = np.asarray(cells, dtype=np.int64)
        slots = np.asarray(slots, dtype=np.int64)
        nc, ns = cells.size, slots.size
        m = self.memo[r][c]
        if nc == 0 or ns == 0:
            z = np.zeros((nc, ns))
            return z.astype(np.int64), np.zeros((nc, ns, K)), np.zeros((nc, ns, K)), np.zeros((nc, ns, K)), z, z
        cc = np.repeat(cells, ns)
        jj_s = np.tile(slots, nc)
        ii = cc - c0
        jj = jj_s - s0
        hit = m["ver"][ii, jj] == self.ver[r][jj_s]
        miss = np.flatnonzero(~hit)
        self.ct["memo_hits"] += int(hit.sum())
        self.ct["memo_misses"] += int(miss.size)
        if miss.size:
            n, Y, S = self._fold_pairs(r, cc[miss], jj_s[miss])
            d = np.full(miss.size, c, dtype=np.int64)
            Q, ll, br = self._solve(r, Y, S, n, d)
            mi, mj = ii[miss], jj[miss]
            m["n"][mi, mj] = n
            m["Y"][mi, mj] = Y
            m["S"][mi, mj] = S
            m["Q"][mi, mj] = Q
            m["ll"][mi, mj] = ll
            m["br"][mi, mj] = br
            m["ver"][mi, mj] = self.ver[r][jj_s[miss]]
        if verify:
            n2, Y2, S2 = self._fold_pairs(r, cc, jj_s)
            ok = (m["n"][ii, jj] == n2) & np.all(m["Y"][ii, jj].view(np.uint64) == Y2.view(np.uint64), 1) & \
                np.all(m["S"][ii, jj].view(np.uint64) == S2.view(np.uint64), 1)
            self.ct["memo_verified"] += int(ok.size)
            if not ok.all():
                raise AssertionError("memo entry statistics differ from the canonical fold (stale cache key)")
        sh = (nc, ns)
        return (m["n"][ii, jj].reshape(sh), m["Y"][ii, jj].reshape(sh + (K,)), m["S"][ii, jj].reshape(sh + (K,)),
                m["Q"][ii, jj].reshape(sh + (K,)), m["ll"][ii, jj].reshape(sh), m["br"][ii, jj].reshape(sh))

    def prefill(self, r, classes=None):
        """One batched solve for every (cell, alive slot) candidate of recipient r (the memo makes later lookups
        exact hits until a slot changes)."""
        pb = self.pb
        for c in (range(pb.K[r]) if classes is None else classes):
            c0, c1 = pb.crange[r][c]
            alive = self.alive_slots(r, c)
            if (c1 - c0) < 2 or alive.size == 0:
                continue
            self._ensure_batched(r, c, np.arange(c0, c1), alive)

    def refresh(self, r, slots):
        """After a move: recompute the memo entries of the changed (alive) slots for every cell of their class in
        one batch (the other entries are still bitwise valid)."""
        slots = [int(j) for j in slots if self.mlen[r][j] > 0]
        if not slots:
            return
        c = int(self.cls[r][slots[0]])
        c0, c1 = self.pb.crange[r][c]
        if c1 - c0 >= 2:
            self._ensure_batched(r, c, np.arange(c0, c1), np.asarray(slots, dtype=np.int64))

    def _ensure_batched(self, r, c, cells, slots):
        # chunk to bound memory (cells x slots x width)
        L = int(self.mlen[r][slots].max()) + 1
        step = max(1, int(4_000_000 // max(1, slots.size * L * self.pb.K[r])))
        for a in range(0, cells.size, step):
            self.ensure(r, c, cells[a:a + step], slots)

    # -------------------------------------------------------------- views
    def alive_slots(self, r, c):
        s0, s1 = self.srange[r][c]
        idx = np.arange(s0, s1)
        return idx[self.mlen[r][idx] > 0]

    def other(self, r):
        return 2 if r == 1 else 1

    def labels(self, r):
        return self.lab[r].copy()

    def canonical_labels(self, r):
        out = np.empty(self.pb.fine[r].F, dtype=np.int64)
        for m in self.members[r]:
            if m:
                out[m] = m[0]
        return out

    def token_counts(self, r):
        return [int(self.alive_slots(r, c).size) for c in range(self.pb.K[r])]

    def T12o(self, r):
        return self.T12 if r == 1 else self.T12.transpose(0, 2, 1)

    def grow(self, r, f):
        """(2, slots of the other recipient): fine cell f of recipient r aggregated by the other's current slots."""
        o = self.other(r)
        Tf = self.pb.Tfine[:, f, :] if r == 1 else self.pb.Tfine[:, :, f]
        So = len(self.members[o])
        g = np.zeros((2, So), dtype=np.int64)
        for s_ in (0, 1):
            g[s_] = np.bincount(self.lab[o], weights=Tf[s_], minlength=So).astype(np.int64)
        return g

    # -------------------------------------------------------------- exact terms
    def terms(self):
        pb = self.pb
        t = {}
        for r in (1, 2):
            al = self.tn[r] > 0
            t[f"L{r}"] = _ssum(self.tll[r][al]) / pb.N
            t[f"B{r}"] = _ssum(self.tbr[r][al]) / pb.N
        if self.sex:
            for r in (1, 2):
                t[f"I{r}"] = (_ssum(pb.phi(self.Tr[r])) + pb.const_mi) / pb.N
            t["I12"] = (_ssum(pb.phi(self.T12)) + pb.const_mi) / pb.N
            t["Phi"] = t["I12"] + PHI_LOCAL_WEIGHT * (t["I1"] + t["I2"])
        t["T"] = t["L1"] + t["L2"] + TASK_BRIER_WEIGHT * (t["B1"] + t["B2"])
        return t

    # -------------------------------------------------------------- moves
    def apply(self, r, f, b):
        """Move fine cell f (recipient r) to slot b of its class. Stats of both affected slots come from the memo
        (verified bitwise against a fresh canonical fold). Returns an undo record."""
        pb = self.pb
        a = int(self.lab[r][f])
        c = int(self.cls[r][a])
        assert a != b and int(self.cls[r][b]) == c and self.mlen[r][b] > 0
        und = {"r": r, "f": f, "a": a, "b": b, "rows": {}}
        for j in (a, b):
            und["rows"][j] = (self.tn[r][j], self.tY[r][j].copy(), self.tS[r][j].copy(), self.tQ[r][j].copy(),
                              self.tll[r][j], self.tbr[r][j], self.minm[r][j], list(self.members[r][j]),
                              self.Mpad[r][j].copy(), self.mlen[r][j], self.ver[r][j])
        n, Y, S, Q, ll, br = self.ensure(r, c, [f], [a, b], verify=True)
        self.members[r][a] = [g for g in self.members[r][a] if g != f]
        self.members[r][b] = sorted(self.members[r][b] + [f])
        self.lab[r][f] = b
        need = max(len(self.members[r][a]), len(self.members[r][b])) + 1
        if need > self.Mpad[r].shape[1]:
            self.Mpad[r] = np.hstack([self.Mpad[r], np.full((self.Mpad[r].shape[0], need - self.Mpad[r].shape[1]),
                                                            pb.fine[r].F, dtype=np.int64)])
        for jj, j in enumerate((a, b)):
            m = self.members[r][j]
            self.ver[r][j] = self._vc
            self._vc += 1
            self.Mpad[r][j] = pb.fine[r].F
            self.Mpad[r][j, :len(m)] = m
            self.mlen[r][j] = len(m)
            self.minm[r][j] = m[0] if m else pb.fine[r].F
            self.tn[r][j], self.tY[r][j], self.tS[r][j] = n[0, jj], Y[0, jj], S[0, jj]
            self.tQ[r][j], self.tll[r][j], self.tbr[r][j] = Q[0, jj], ll[0, jj], br[0, jj]
        # bitwise check of the new stats against the canonical fold of the new member sets
        n2, Y2, S2 = self._fold(r, self._pad_rows(r, np.array([a, b])))
        if not (np.array_equal(n2, self.tn[r][[a, b]]) and np.array_equal(Y2.view(np.uint64),
                                                                            self.tY[r][[a, b]].view(np.uint64))
                and np.array_equal(S2.view(np.uint64), self.tS[r][[a, b]].view(np.uint64))):
            raise AssertionError("memo statistics differ from the canonical fold after a move")
        if self.sex:
            tf = pb.tf[r][:, f]
            self.Tr[r][:, a] -= tf
            self.Tr[r][:, b] += tf
            g = self.grow(r, f)
            if r == 1:
                self.T12[:, a, :] -= g
                self.T12[:, b, :] += g
            else:
                self.T12[:, :, a] -= g
                self.T12[:, :, b] += g
        return und

    def undo(self, und):
        r, f, a, b = und["r"], und["f"], und["a"], und["b"]
        pb = self.pb
        if self.sex:
            g = self.grow(r, f)          # the other recipient's labels are unchanged since apply
            tf = pb.tf[r][:, f]
            self.Tr[r][:, a] += tf
            self.Tr[r][:, b] -= tf
            if r == 1:
                self.T12[:, a, :] += g
                self.T12[:, b, :] -= g
            else:
                self.T12[:, :, a] += g
                self.T12[:, :, b] -= g
        self.lab[r][f] = a
        for j, v in und["rows"].items():
            (self.tn[r][j], self.tY[r][j], self.tS[r][j], self.tQ[r][j], self.tll[r][j], self.tbr[r][j],
             self.minm[r][j], self.members[r][j], row, self.mlen[r][j], self.ver[r][j]) = v
            self.Mpad[r][j] = self.pb.fine[r].F
            self.Mpad[r][j, :row.size] = row


def new_counters():
    return {"evals": 0, "solves": 0, "solve_calls": 0, "memo_hits": 0, "memo_misses": 0, "memo_verified": 0,
            "accepted": 0,
            "rejected_by_budget": {"L": 0, "B": 0, "I": 0}, "rejected_on_exact_recheck": 0, "pair_evals": 0,
            "pair_pool_screened": 0, "pair_accepted": 0, "pair_rejected_infeasible": 0}


# ------------------------------------------------------------------ feasibility
def constraint_values(pb, t, r):
    out = {"L": t[f"L{r}"], "L_limit": pb.limL[r], "B": t[f"B{r}"], "B_limit": pb.limB[r]}
    if pb.capI[r] is not None:
        out.update({"I": t[f"I{r}"], "I_limit": pb.capI[r]})
    out["L_slack"] = out["L_limit"] - out["L"]
    out["B_slack"] = out["B_limit"] - out["B"]
    if "I" in out:
        out["I_slack"] = out["I_limit"] - out["I"]
    return out


def feasible(pb, t, recips):
    """Search feasibility on exact state values (margins BUDGET_MARGIN on L/B, CAP_MARGIN on I)."""
    for r in recips:
        if t[f"L{r}"] > pb.limL[r] - BUDGET_MARGIN or t[f"B{r}"] > pb.limB[r] - BUDGET_MARGIN:
            return False
        if pb.capI[r] is not None and t[f"I{r}"] > pb.capI[r] - CAP_MARGIN:
            return False
    return True


# ------------------------------------------------------------------ one fine cell
def _cell(st, r, f, w, cur, cons, need_feas_all=False):
    """Candidates of fine cell f: targets (alive same-class slots, canonical order), exact deltas, the frozen
    neighbourhood (4 nearest + 4 best, deduplicated) and the chosen move (or None)."""
    pb = st.pb
    a = int(st.lab[r][f])
    c = int(st.cls[r][a])
    B = st.alive_slots(r, c)
    B = B[B != a]
    if B.size == 0:
        return None
    B = B[np.argsort(st.minm[r][B], kind="stable")]
    canon = st.minm[r][B]
    N = pb.N
    slots = np.concatenate([[a], B])
    n, Y, S, Q, ll, br = st.ensure(r, c, [f], slots)
    dL = (ll[0, 0] + ll[0, 1:] - st.tll[r][a] - st.tll[r][B]) / N
    dB = (br[0, 0] + br[0, 1:] - st.tbr[r][a] - st.tbr[r][B]) / N
    delta = w[0] * dL + w[1] * dB
    dI = dI12 = None
    if st.sex and (w[2] or w[3] or (cons and pb.capI[r] is not None) or need_feas_all):
        tf = pb.tf[r][:, f]
        Tr = st.Tr[r]
        dI = (pb.phi((Tr[:, a] - tf)[:, None]) + pb.phi(Tr[:, B] + tf[:, None]) - pb.phi(Tr[:, [a]])
              - pb.phi(Tr[:, B])) / N
        if w[2]:
            delta = delta + w[2] * dI
        if w[3] or need_feas_all:
            g = st.grow(r, f)
            X = st.T12o(r)
            Xa = X[:, a, :]
            XB = X[:, B, :]
            dI12 = (pb.phi(Xa - g).sum() + pb.phi(XB + g[:, None, :]).sum(-1) - pb.phi(Xa).sum()
                    - pb.phi(XB).sum(-1)) / N
            if w[3]:
                delta = delta + w[3] * dI12
    st.ct["evals"] += int(B.size)
    # neighbourhood
    m = pb.mean[r][f]
    pos = m > 0
    kl = (m[pos][None, :] * (np.log(m[pos])[None, :] - np.log(st.tQ[r][B][:, pos]))).sum(1)
    near = np.lexsort((canon, kl))[:N_NEAREST]
    best = np.lexsort((canon, delta))[:N_BEST]
    hood = np.unique(np.concatenate([near, best]))
    out = {"a": a, "c": c, "B": B, "canon": canon, "delta": delta, "dL": dL, "dB": dB, "dI": dI, "dI12": dI12,
           "hood": hood, "move": None}
    improving = delta[hood] < -TOL
    if r in cons:
        okL = cur[f"L{r}"] + dL[hood] <= pb.limL[r] - BUDGET_MARGIN
        okB = cur[f"B{r}"] + dB[hood] <= pb.limB[r] - BUDGET_MARGIN
        okI = np.ones(hood.size, dtype=bool) if pb.capI[r] is None else \
            cur[f"I{r}"] + dI[hood] <= pb.capI[r] - CAP_MARGIN
        feas = okL & okB & okI
        rej = improving & ~feas
        st.ct["rejected_by_budget"]["L"] += int(np.sum(rej & ~okL))
        st.ct["rejected_by_budget"]["B"] += int(np.sum(rej & ~okB))
        st.ct["rejected_by_budget"]["I"] += int(np.sum(rej & ~okI))
        out["own_feasible"] = feas
    else:
        feas = np.ones(hood.size, dtype=bool)
        out["own_feasible"] = feas
    ok = improving & feas
    if ok.any():
        h = hood[ok]
        dmin = float(delta[h].min())
        tied = h[delta[h] <= dmin + TIE_TOL]
        j = int(tied[np.argmin(canon[tied])])
        out["move"] = (int(f), int(B[j]), float(delta[j]))
    return out


def _accept(st, r, f, b, w, cons, cur, cur_obj):
    """Apply, re-check exactly (strict objective decrease, constraints of every enforced recipient) or revert."""
    und = st.apply(r, f, b)
    t = st.terms()
    v = objective(t, w)
    if v < cur_obj and feasible(st.pb, t, cons):
        st.refresh(r, [und["a"], b])                          # memo entries of the two changed slots
        st.ct["accepted"] += 1
        return True, t, v
    st.undo(und)
    st.ct["rejected_on_exact_recheck"] += 1
    return False, cur, cur_obj


def _pair_step(st, w, cons, cur, cur_obj, budget_left):
    """Atomic paired step: per recipient retain 8 proposals that satisfy that recipient's own budgets and local cap
    (4 by exact one-sided Phi change, then 4 more by exact task-loss change L_i + 0.5 B_i, canonical ties
    (value, fine cell, target canonical index)); evaluate the 8 x 8 Cartesian product atomically (both moves applied,
    joint table and both decoders recomputed, every constraint re-checked on exact values); accept the best feasible
    strict Phi improvement (ties: first in (i, j) order)."""
    pb = st.pb
    props = {}
    for r in (1, 2):
        pool = []
        st.prefill(r)
        for f in range(pb.fine[r].F):
            if pb.ncell[r][f] == 0:
                continue
            res = _cell(st, r, f, w, cur, cons, need_feas_all=True)
            if res is None:
                continue
            st.ct["pair_pool_screened"] += int(res["B"].size)
            for k, hk in enumerate(res["hood"]):
                if not res["own_feasible"][k]:
                    continue
                dphi = float(res["dI12"][hk] + PHI_LOCAL_WEIGHT * res["dI"][hk])
                dtask = float(res["dL"][hk] + TASK_BRIER_WEIGHT * res["dB"][hk])
                pool.append((dphi, dtask, int(f), int(res["canon"][hk]), int(res["B"][hk])))
        by_phi = sorted(pool, key=lambda x: (x[0], x[2], x[3]))[:PAIR_KEEP_PHI]
        chosen = {(x[2], x[4]) for x in by_phi}
        rest = [x for x in pool if (x[2], x[4]) not in chosen]
        by_task = sorted(rest, key=lambda x: (x[1], x[2], x[3]))[:PAIR_KEEP_TASK]
        props[r] = by_phi + by_task
    best = None
    for i, m1 in enumerate(props[1]):
        for j, m2 in enumerate(props[2]):
            if st.ct["pair_evals"] >= budget_left:
                break
            st.ct["pair_evals"] += 1
            st.ct["evals"] += 1
            u1 = st.apply(1, m1[2], m1[4])
            u2 = st.apply(2, m2[2], m2[4])
            t = st.terms()
            v = objective(t, w)
            ok = feasible(pb, t, cons)
            st.undo(u2)
            st.undo(u1)
            if not ok:
                st.ct["pair_rejected_infeasible"] += 1
                continue
            if v < cur_obj - TOL and (best is None or v < best[0]):
                best = (v, i, j, m1, m2)
    info = {"proposals": {str(r): [{"fine_cell": x[2], "target_canon": x[3], "dPhi_one_sided": x[0],
                                     "dTask": x[1]} for x in props[r]] for r in (1, 2)}, "accepted": None}
    if best is None:
        return False, cur, cur_obj, info
    v, i, j, m1, m2 = best
    u1 = st.apply(1, m1[2], m1[4])
    u2 = st.apply(2, m2[2], m2[4])
    t = st.terms()
    vv = objective(t, w)
    if not (vv < cur_obj - TOL and feasible(pb, t, cons)):
        st.undo(u2)
        st.undo(u1)
        st.ct["rejected_on_exact_recheck"] += 1
        return False, cur, cur_obj, info
    st.refresh(1, [u1["a"], m1[4]])
    st.refresh(2, [u2["a"], m2[4]])
    st.ct["pair_accepted"] += 1
    info["accepted"] = {"i": i, "j": j, "r1": {"fine_cell": m1[2], "target_canon": m1[3]},
                        "r2": {"fine_cell": m2[2], "target_canon": m2[3]}, "objective_after": vv}
    return True, t, vv, info


def refine(st, recips, w, cons, ceiling, pair=False):
    """At most SWEEPS sweeps; one deterministic ordering (recipients as given, fine cells increasing); per cell the
    best strictly improving feasible move of its neighbourhood is applied at once; each sweep restarts from the
    coherent accepted state (memo refreshed); PAIR: one paired step after each single sweep. Stops at a no-change
    sweep, the sweep cap or the proposal-evaluation ceiling."""
    pb = st.pb
    e0 = st.ct["evals"]
    cur = st.terms()
    cur_obj = objective(cur, w)
    start_obj = cur_obj
    log, pairs = [], []
    stop, sweeps = "sweep_cap", 0
    for sweep in range(1, SWEEPS + 1):
        sweeps = sweep
        changed = 0
        hit_ceiling = False
        for r in recips:
            st.prefill(r)
            for f in range(pb.fine[r].F):
                if pb.ncell[r][f] == 0:
                    continue
                if st.ct["evals"] - e0 >= ceiling:
                    hit_ceiling = True
                    break
                res = _cell(st, r, f, w, cur, cons)
                if res is None or res["move"] is None:
                    continue
                _, b, dlt = res["move"]
                ok, cur, cur_obj = _accept(st, r, f, b, w, cons, cur, cur_obj)
                if ok:
                    changed += 1
                    log.append([sweep, r, int(f), int(res["a"]), int(b), dlt])
            if hit_ceiling:
                break
        if pair and not hit_ceiling:
            ok, cur, cur_obj, info = _pair_step(st, w, cons, cur, cur_obj, budget_left=st.ct["pair_evals"]
                                                + max(0, ceiling - (st.ct["evals"] - e0)))
            info["sweep"] = sweep
            pairs.append(info)
            if ok:
                changed += 1
        if hit_ceiling:
            stop = "eval_ceiling"
            break
        if changed == 0:
            stop = "no_change_sweep"
            break
    if cur_obj > start_obj:
        raise AssertionError("refinement increased the objective")
    return {"sweeps": sweeps, "stop": stop, "moves": log, "pair_steps": pairs, "objective_start": start_obj,
            "objective_end": cur_obj, "evals": st.ct["evals"] - e0}


# ------------------------------------------------------------------ arm drivers
def _start_record(name, st, w, pb, cons_all):
    t = st.terms()
    return {"name": name, "terms": t, "objective": objective(t, w),
            "constraints": {str(r): constraint_values(pb, t, r) for r in (1, 2)},
            "feasible_all": bool(feasible(pb, t, cons_all)) if cons_all else None,
            "token_counts": {str(r): st.token_counts(r) for r in (1, 2)}}


def _class_only(fine):
    return QC.class_labels(fine)


def _run_local(pb, arm, w, starts, cons_flag, E):
    out, cands = [], []
    for name, labs in starts:
        ct = new_counters()
        c0 = time.process_time()
        st = State(pb, labs, sex=SPEC[arm]["sex"], counters=ct)
        rec = {"start": _start_record(name, st, w, pb, (1, 2) if cons_flag else ()), "stages": []}
        ok_all = True
        for r in (1, 2):
            t = st.terms()
            cons = (r,) if cons_flag else ()
            if cons_flag and not feasible(pb, t, cons):
                rec["stages"].append({"recipient": r, "status": "INFEASIBLE_START", "refined": False})
                ok_all = False
                continue
            s = refine(st, (r,), w, cons, ceiling=E // (2 * len(starts)))
            s.update({"recipient": r, "status": "REFINED"})
            rec["stages"].append(s)
        rec["final"] = _start_record(name, st, w, pb, (1, 2) if cons_flag else ())
        rec["eligible"] = bool(ok_all and (not cons_flag or rec["final"]["feasible_all"]))
        rec["work"] = ct
        rec["cpu_s"] = time.process_time() - c0
        out.append(rec)
        cands.append((name, "refined", st, rec))
    return out, cands


def _run_seq(pb, arm, w, starts, cons_flag, E):
    a, b = SPEC[arm]["order"]
    out, cands = [], []
    E_stage = E // (2 * len(starts))
    for name, labs in starts:
        ct = new_counters()
        c0 = time.process_time()
        lab1 = {a: labs[a], b: _class_only(pb.fine[b])}
        st1 = State(pb, lab1, counters=ct)
        rec = {"name": name, "stage1": {"recipient": a, "partner": "CLASS-ONLY",
                                         "partner_constraints_enforced": False,
                                         "start": _start_record(name, st1, w, pb, (a,) if cons_flag else ())}}
        cons1 = (a,) if cons_flag else ()
        if cons_flag and not feasible(pb, st1.terms(), cons1):
            rec["stage1"]["status"] = "INFEASIBLE_START"
            rec["eligible"] = False
            rec["work"] = ct
            rec["cpu_s"] = time.process_time() - c0
            out.append(rec)
            continue
        s1 = refine(st1, (a,), w, cons1, ceiling=E_stage)
        s1["status"] = "REFINED"
        rec["stage1"].update(s1)
        t1 = st1.terms()
        rec["stage1"]["end"] = {"terms": t1, "partner_class_only_constraints": constraint_values(pb, t1, b)}
        frozen = st1.canonical_labels(a)
        lab2 = {a: frozen, b: labs[b]}
        st2 = State(pb, lab2, counters=ct)
        if not np.array_equal(st2.canonical_labels(a), frozen):
            raise AssertionError("frozen first map changed")
        rec["stage2"] = {"recipient": b, "frozen": a, "start": _start_record(name, st2, w, pb,
                                                                              (b,) if cons_flag else ())}
        cons2 = (b,) if cons_flag else ()
        if cons_flag and not feasible(pb, st2.terms(), cons2):
            rec["stage2"]["status"] = "INFEASIBLE_START"
            rec["eligible"] = False
        else:
            s2 = refine(st2, (b,), w, cons2, ceiling=E_stage)
            s2["status"] = "REFINED"
            rec["stage2"].update(s2)
            if not np.array_equal(st2.canonical_labels(a), frozen):
                raise AssertionError("first recipient revised after stage 1")
            rec["final"] = _start_record(name, st2, w, pb, (1, 2) if cons_flag else ())
            rec["eligible"] = bool(not cons_flag or rec["final"]["feasible_all"])
            cands.append((name, "refined", st2, rec))
        rec["work"] = ct
        rec["cpu_s"] = time.process_time() - c0
        out.append(rec)
    return out, cands


def _run_joint(pb, arm, w, starts, cons_flag, E):
    out, cands = [], []
    E_start = E // len(starts)
    cons = (1, 2) if cons_flag else ()
    for name, labs in starts:
        ct = new_counters()
        c0 = time.process_time()
        st = State(pb, labs, counters=ct)
        rec = {"name": name, "unchanged": _start_record(name, st, w, pb, cons)}
        if cons_flag and not rec["unchanged"]["feasible_all"]:
            rec.update({"status": "EXCLUDED_INFEASIBLE_WITNESS", "eligible_unchanged": False, "eligible": False,
                        "work": ct, "cpu_s": time.process_time() - c0})
            out.append(rec)
            continue
        ut = st.terms()
        snap = {r: st.canonical_labels(r) for r in (1, 2)}
        s = refine(st, (1, 2), w, cons, ceiling=E_start, pair=SPEC[arm]["pair"])
        rec.update(s)
        rec["final"] = _start_record(name, st, w, pb, cons)
        rec["status"] = "REFINED"
        rec["eligible_unchanged"] = True
        rec["eligible"] = bool(not cons_flag or rec["final"]["feasible_all"])
        rec["work"] = ct
        rec["cpu_s"] = time.process_time() - c0
        out.append(rec)
        cands.append((name, "refined", st, rec))
        cands.append((name, "unchanged", (snap, ut), rec))
    return out, cands


def _winner(cands, w, cons_flag, pb):
    """Lowest exact objective among eligible candidates; ties: registered start order, refined before unchanged."""
    best = None
    for k, (name, kind, obj, rec) in enumerate(cands):
        if kind == "refined":
            t = obj.terms()
            elig = rec.get("eligible", False)
        else:
            t = obj[1]
            elig = rec.get("eligible_unchanged", False)
        if cons_flag and not (elig and feasible(pb, t, (1, 2))):
            continue
        v = objective(t, w)
        if best is None or v < best[0]:
            best = (v, k)
    return best


# ------------------------------------------------------------------ deployed (from-scratch) recomputation
def deployed_terms(pb, pair, release, tr):
    """L, B (dpc per_row on the released rows), I, I12 (dpc mi_plugin on the released tokens) on the fitting rows."""
    out = {}
    toks = {}
    for i in (1, 2):
        tok = np.asarray(release[f"tok{i}"])[tr]
        q = np.asarray(release[f"q{i}"])[tr]
        pr = per_row(q, pb.y[i], pb.K[i])
        out[f"L{i}"] = float(pr["ll"].mean())
        out[f"B{i}"] = float(pr["br"].mean())
        if not np.array_equal(np.asarray(release[f"hard{i}"])[tr], pb.dd[i]):
            raise AssertionError("decision preservation violated on the fitting rows")
        toks[i] = tok
    if pb.s is not None:
        out["I1"] = mi_plugin(pb.s, toks[1])
        out["I2"] = mi_plugin(pb.s, toks[2])
        out["I12"] = mi_plugin(pb.s, toks[1] * pair.p2.T + toks[2])
        out["Phi"] = out["I12"] + PHI_LOCAL_WEIGHT * (out["I1"] + out["I2"])
    out["T"] = out["L1"] + out["L2"] + TASK_BRIER_WEIGHT * (out["B1"] + out["B2"])
    return out, toks


def d0_terms(st):
    """Descriptive: L, B of the same map with the D0 mean-teacher decoder (smooth(S_t / n_t, class))."""
    pb = st.pb
    t = {}
    for r in (1, 2):
        al = np.flatnonzero(st.tn[r] > 0)
        Q0 = KM.smooth(st.tS[r][al] / st.tn[r][al, None], st.cls[r][al])
        ll, br = pb.dec.token_losses(st.tY[r][al], Q0)
        t[f"L{r}"], t[f"B{r}"] = _ssum(ll) / pb.N, _ssum(br) / pb.N
    t["T"] = t["L1"] + t["L2"] + TASK_BRIER_WEIGHT * (t["B1"] + t["B2"])
    return t


# ------------------------------------------------------------------ inputs
def _yfit(Y_fit):
    if isinstance(Y_fit, dict):
        def g(i):
            for k in (i, str(i), {1: "income", 2: "occupation"}[i]):
                if k in Y_fit:
                    return np.asarray(Y_fit[k], dtype=np.int64)
            raise KeyError(f"Y_fit lacks recipient {i}")
        return {1: g(1), 2: g(2)}
    Y = np.asarray(Y_fit)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Y_fit must be {1: y1, 2: y2} or an (n, 2) array")
    return {1: Y[:, 0].astype(np.int64), 2: Y[:, 1].astype(np.int64)}


def refs_from_ctask(ctask_record):
    """refs for the K- arms from the C-TASK unit's record: exact fitting MI caps + cross-check values."""
    ft = ctask_record["final_state_terms"]
    return {"I_ctask": {"1": float(ft["I1"]), "2": float(ft["I2"])},
            "ctask_pair_fingerprint": ctask_record["pair_fingerprint"],
            "L_U": {"1": ctask_record["budgets"]["L_U"]["1"], "2": ctask_record["budgets"]["L_U"]["2"]},
            "B_U": {"1": ctask_record["budgets"]["B_U"]["1"], "2": ctask_record["budgets"]["B_U"]["2"]}}


def rules():
    """Every frozen rule (SEARCH_RULES.json body; the lead locks it)."""
    return {
        "schema": SCHEMA, "owner": "role C (mapper)", "module": "lcr/mapper.py",
        "arms": list(ARMS),
        "config_ids": {a: (config_id(a, 0.01) if a.startswith("W-") else config_id(a)) for a in ARMS},
        "decoder": {"name": "D1 (lcr.decoder.solve_batch)", "kappa": KAPPA, "eps": EPS, "loss_clip": LOSS_CLIP,
                    "token_statistics": "n_t, label counts y_t (exact), teacher sums s_t accumulated from zeros over "
                                        "member fine cells in increasing fine index (= qpc token_tables = "
                                        "lcr.decoder.token_stats, bitwise); fallback/empty tokens (n_t = 0) carry no "
                                        "loss and the pinned D0 vector at release"},
        "fitting_quantities": {
            "rows": "OSF_DEFENSE_FIT (tr), deployed fine-cell routing of the frozen teacher",
            "L_i": "(1/N) sum_t sum_k y_t[k] * -log clip(q_t[k], 1e-12, 1) (lcr.decoder.token_losses)",
            "B_i": "(1/N) sum_t sum_k y_t[k] * sum_j (q_t[j] - [j=k])^2 (source multiclass Brier)",
            "I_i": "plug-in I(SEX; full token of recipient i), exact integer tables, n log n lookup, natural log",
            "I12": "plug-in I(SEX; (token_1, token_2))",
            "T": "L_1 + L_2 + 0.5 (B_1 + B_2)", "Phi": "I12 + 0.5 (I1 + I2)",
            "exact_state_value": "sum of the sorted nonzero per-token (per-cell) contributions (path/label independent)",
            "incremental": "candidate deltas by vectorised difference formulas from per-token sufficient statistics "
                           "(n_t, y_t, s_t, SEX counts) and the pair count table; accepted moves re-checked on exact "
                           "state values and reverted if the exact check fails"},
        "budgets": {"profile": "one profile, all seeds", "L": "L_i <= L_i(U) + 0.005", "B": "B_i <= B_i(U) + 0.003",
                    "U": "continuous U teacher probabilities on the same fitting rows, dpc per_row conventions",
                    "local_cap": "I_i <= I_i(C-TASK) of the same seed (exact record bits, refs['I_ctask'])",
                    "decision_preservation": "structural: tokens never cross predicted classes; decoder certifies "
                                             "strict argmax = class; release_arrays_d1 re-checks every row",
                    "state_caps": {"income": CAPS[0], "occupation": CAPS[1], "per": "teacher-predicted class",
                                   "rule": "upper bounds; tokens are never created; empty tokens are removed"},
                    "search_margins": {"L_B": BUDGET_MARGIN, "I_cap": CAP_MARGIN},
                    "deployed_tolerance": DEPLOYED_TOL,
                    "budget_values": BUDGET},
        "objectives": {
            "C-TASK": "T (per recipient L_i + 0.5 B_i); no SEX enters the search; budgets recorded, not enforced",
            "W-LOCAL": "T + lam (I1 + I2)/2 (per recipient L_i + 0.5 B_i + lam I_i / 2); no hard budget",
            "W-SEQ-12/21": "T + lam Phi; sequential; no hard budget",
            "W-JOINT": "T + lam Phi; joint single moves; no hard budget",
            "K-LOCAL": "min I_i per recipient s.t. its own budgets and local cap",
            "K-SEQ-12/21": "Phi; stage 1 only the first recipient's budgets + cap (CLASS-ONLY partner never required "
                           "feasible, never a constant); stage 2 the second's; FINAL must satisfy both",
            "K-JOINT-SINGLE": "Phi s.t. both budgets + both local caps; single-recipient moves",
            "K-JOINT-PAIR": "K-JOINT-SINGLE + atomic paired step"},
        "lam_grid": list(RN.LAMS),
        "neighbourhood": {
            "proposal": "move ONE WHOLE fine cell to an existing alive token of its predicted class (tokens never "
                        "created; a move may empty its source token, which is then removed)",
            "targets_per_cell": f"union of the {N_NEAREST} nearest targets by KL(fine-cell mean teacher prob || "
                                f"target's current released q) and the {N_BEST} targets with the best exact objective "
                                "change (C-TASK: task objective, no SEX), deduplicated",
            "ties": "lexicographic (value, target canonical index = lowest member fine index)",
            "exact_objective_change_evaluated_for": "every alive same-class target (counted as proposal evaluations)",
            "accept": f"best strictly improving (delta < -{TOL}) feasible target of the neighbourhood; ties within "
                      f"{TIE_TOL} -> lowest canonical target; applied at once; exact re-check (new exact objective "
                      "< old, constraints of every enforced recipient on exact values) else reverted",
            "ordering": "one deterministic ordering per start: recipients 1 then 2 (local: each alone; seq: one per "
                        "stage), fine cells in increasing index; cells with no fitting rows or in single-token "
                        "classes have no proposal",
            "sweeps": SWEEPS,
            "sweep_restart": "each sweep restarts from the coherent accepted state (memo refreshed)",
            "stop": "a sweep with no accepted single or paired move ('no_change_sweep'), the sweep cap "
                    "('sweep_cap') or the unit's proposal-evaluation share ('eval_ceiling')",
            "feasibility_screen": "constrained arms: neighbourhood candidates are screened on the moved "
                                  "recipient's own budgets and local cap (exact current value + delta, margins "
                                  "above); improving-but-infeasible candidates are counted as rejected_by_budget"},
        "pair_step": {"arm": "K-JOINT-PAIR only", "order": PAIR_STEP, "accepted_per_step": PAIR_ACCEPT_PER_STEP,
                      "pool": "every neighbourhood candidate of every fine cell of the recipient at the post-sweep "
                              "state that satisfies that recipient's own budgets and local cap (non-improving "
                              "included)",
                      "retain": f"{PAIR_KEEP_PHI} by exact one-sided Phi change, then {PAIR_KEEP_TASK} more (not "
                                "already retained) by exact task-loss change L_i + 0.5 B_i; ties (value, fine cell, "
                                "target canonical index)",
                      "evaluate": "Cartesian product (8 x 8) atomically: both moves applied, joint table and both "
                                  "decoders recomputed, every constraint re-checked on exact values",
                      "accept": f"best feasible strict Phi improvement (< -{TOL}); ties first in (i, j) order"},
        "starts": {a: (registered_starts(a, 0.01) if a.startswith("W-") else registered_starts(a)) for a in ARMS},
        "starts_notes": {
            "lam_placeholder": "W-JOINT witnesses are at the SAME lambda as the unit (l0.01 shown)",
            "maps": "each start/witness is a policy.json map; only its cell->token map is used; D1 recomputed",
            "local": "C-TASK starts from source FINE-TASK; K-LOCAL / W-LOCAL from C-TASK",
            "seq": "C-TASK + the six source SEQ-ab maps; per start: stage 1 from the start's first-recipient map "
                   "with the partner at CLASS-ONLY, stage 2 from the start's second-recipient map with the frozen "
                   "first map",
            "joint": "every witness that satisfies all constraints (K- arms; W-JOINT: all) is a candidate UNCHANGED "
                     "and is refined; an infeasible witness is recorded EXCLUDED_INFEASIBLE_WITNESS and is never "
                     "eligible by its name",
            "infeasible_start": "local/seq: a start infeasible for the stage's enforced constraints is not refined "
                                "(INFEASIBLE_START, ineligible); no restoration phase",
            "winner": "lowest exact objective among eligible candidates; ties: registered order, refined before "
                      "unchanged; none eligible -> status INFEASIBLE (descriptive release of the first candidate)"},
        "equal_work": {"EVAL_CEILING_per_unit": EVAL_CEILING,
                       "definition": "proposal evaluation = one exact objective change of one (cell, target) "
                                     "candidate, or one atomic pair; pair-pool screening counts",
                       "allocation": "joint: ceiling / #starts per start; seq: ceiling / (2 #starts) per stage per "
                                     "start; local: ceiling / (2 #starts) per recipient",
                       "same_for": "every arm (K-JOINT-SINGLE = K-JOINT-PAIR = K-SEQ-12 = K-SEQ-21 = 8e6 total)",
                       "recorded": "actual evaluations, solves, memo hits/misses and CPU per start and per unit"},
        "cache": {"rule": "memo of decoder solves per (recipient, class, fine cell, slot); an entry is reused only if "
                          "its stored (n_t, y_t, s_t) are BITWISE equal to the freshly folded statistics (exact "
                          "sufficient-statistic key, -0.0 impossible for sums of nonnegative values); never reused "
                          "across different label counts or teacher sums",
                  "batch": "lcr.decoder.solve_batch (rows independent and bitwise equal to solve_token)"},
        "receipts": ["every accepted move; rejected_by_budget counts (L, B, I); rejected_on_exact_recheck",
                     "sweeps and stop reasons per stage/start", "objective and constraint values of every start, "
                     "of every refined candidate and of the winner", "coherent final recomputation from the "
                     "deployed release (row-level L, B; mi_plugin I, I12) with state-vs-deployed parity",
                     "budget slack, local-cap slack, actual token counts per class", "proposal counts, solves, "
                     "memo hits, CPU per start and unit"],
        "tolerances": {"TOL": TOL, "TIE_TOL": TIE_TOL, "BUDGET_MARGIN": BUDGET_MARGIN, "CAP_MARGIN": CAP_MARGIN,
                       "PARITY_ATOL": PARITY_ATOL, "MI_PARITY_ATOL": MI_PARITY_ATOL, "DEPLOYED_TOL": DEPLOYED_TOL},
        "randomness": "none",
        "not_claimed": ["global optimality (greedy local search, <= 5 sweeps)", "a population or SEX-AUC bound",
                        "convexity of the discrete program (only the fixed-token decoder is convex)",
                        "the official Taylor sequential solver"]}


def code_hashes():
    """sha256 of the study code a fit runs (mapper, decoder, run helpers) and the qpc/dpc closure it imports."""
    from pathlib import Path
    wt = Path(__file__).resolve().parents[1]
    fs = ("lcr/mapper.py", "lcr/decoder.py", "lcr/run.py", "qpc/compress.py", "qpc/release.py", "qpc/kmeans.py",
          "qpc/partition.py", "dpc/compress.py", "dpc/release.py", "dpc/partition.py", "dpc/utility.py")
    return {f: hashlib.sha256((wt / f).read_bytes()).hexdigest() for f in fs}


def rules_sha256():
    return hashlib.sha256(json.dumps(rules(), sort_keys=True, allow_nan=False).encode()).hexdigest()


# ------------------------------------------------------------------ the unit
def fit_unit(arm, fine_dict, T, tr, Y_fit, S_fit, lam=None, starts=None, witnesses=None, refs=None, meta=None):
    """One new mapping-pair unit. See the module docstring. Returns (record, files)."""
    t0, c0 = time.perf_counter(), time.process_time()
    if arm not in ARMS:
        raise ValueError(f"REFUSED: unknown arm {arm!r}")
    meta = dict(meta or {})
    refs = dict(refs or {})
    fixture = bool(meta.get("fixture", False))
    spec = SPEC[arm]
    if arm.startswith("W-"):
        if lam is None or not np.isfinite(lam) or float(lam) <= 0:
            raise ValueError(f"REFUSED: {arm} needs a finite lam > 0")
        lam = float(lam)
        if lam not in RN.LAMS and not fixture:
            raise ValueError(f"REFUSED: lam {lam} is not on the registered grid {RN.LAMS}")
    elif lam is not None:
        raise ValueError(f"REFUSED: {arm} takes no lam")
    cid = config_id(arm, lam)
    if meta.get("config") not in (None, cid):
        raise ValueError(f"meta config {meta.get('config')} differs from {cid}")
    for k in RL.BINDING_KEYS:
        if not (isinstance(meta.get(k), str) and len(meta[k]) == 64):
            raise ValueError(f"meta must carry {k} (sha256 hex) to bind the policy for deployment")
    caps = tuple(int(x) for x in refs.get("caps", CAPS))
    budget = {k: float(v) for k, v in refs.get("budget", BUDGET).items()}
    if not fixture and (caps != CAPS or budget != BUDGET):
        raise ValueError("REFUSED: caps/budget differ from the registered profile (fixture mode only)")
    # starts / witnesses
    kind = spec["kind"]
    given = witnesses if kind == "joint" else starts
    other = starts if kind == "joint" else witnesses
    if other:
        raise ValueError(f"REFUSED: {arm} takes {'witnesses' if kind == 'joint' else 'starts'} only")
    if not isinstance(given, dict) or not given:
        raise ValueError(f"REFUSED: {arm} needs its registered {'witnesses' if kind == 'joint' else 'starts'}")
    reg = registered_starts(arm, lam)
    extra = sorted(set(given) - set(reg))
    if extra:
        raise ValueError(f"REFUSED: unregistered start/witness keys {extra}")
    need = reg if not fixture else required_starts(arm, lam)
    miss = [k for k in need if k not in given]
    if miss:
        raise ValueError(f"REFUSED: missing start/witness keys {miss}")
    dec = _dec()
    fine1, fine2 = PT.load_fine(fine_dict)
    fines = {1: fine1, 2: fine2}
    tr = np.asarray(tr, dtype=np.int64)
    P = {1: KM.check_probs(np.asarray(T["p1"])[tr], fine1.K, "P1"), 2: KM.check_probs(np.asarray(T["p2"])[tr],
                                                                                         fine2.K, "P2")}
    dd = {1: KM.check_decisions(P[1], np.asarray(T["d1"])[tr], "d1"),
          2: KM.check_decisions(P[2], np.asarray(T["d2"])[tr], "d2")}
    Y = _yfit(Y_fit)
    s = QC.check_sex(S_fit, tr.size)
    for i in (1, 2):
        if Y[i].shape != (tr.size,):
            raise ValueError("Y_fit is not aligned with tr")
    pb = Problem(fine1, fine2, P, dd, Y, None if not spec["sex"] else s, caps, budget, dec)
    order = [k for k in reg if k in given]
    start_maps, start_fps = [], {}
    for k in order:
        labs, fp = _pair_labels(given[k], fines)
        start_maps.append((k, labs))
        start_fps[k] = fp
    # local caps (constrained arms)
    ref_rec = {}
    if spec["constrained"]:
        ic = refs.get("I_ctask")
        if not isinstance(ic, dict):
            raise ValueError("REFUSED: K- arms need refs['I_ctask'] (refs_from_ctask of the same seed's C-TASK)")
        pb.capI = {1: float(ic.get("1", ic.get(1))), 2: float(ic.get("2", ic.get(2)))}
        ck = RN.ctask_id()
        st_ct = State(pb, dict(start_maps)[ck], counters=new_counters())
        tct = st_ct.terms()
        if (tct["I1"], tct["I2"]) != (pb.capI[1], pb.capI[2]):
            raise ValueError(f"REFUSED: refs I_ctask {pb.capI} differ from the passed C-TASK map's exact MI "
                             f"({tct['I1']}, {tct['I2']})")
        if refs.get("ctask_pair_fingerprint") not in (None, start_fps[ck]):
            raise ValueError("REFUSED: refs C-TASK fingerprint differs from the passed C-TASK map")
        ref_rec = {"I_ctask": {"1": pb.capI[1], "2": pb.capI[2]}, "ctask_map_bitwise_equal": True,
                   "ctask_pair_fingerprint": start_fps[ck]}
    for key, mine in (("L_U", pb.LU), ("B_U", pb.BU)):
        if key in refs:
            v = refs[key]
            for i in (1, 2):
                if abs(float(v.get(str(i), v.get(i))) - mine[i]) > 1e-12:
                    raise ValueError(f"REFUSED: refs {key} differs from the U losses recomputed on these rows")
    w = weights(arm, lam)
    E = EVAL_CEILING
    runner = {"local": _run_local, "seq": _run_seq, "joint": _run_joint}[kind]
    start_recs, cands = runner(pb, arm, w, start_maps, spec["constrained"], E)
    best = _winner(cands, w, spec["constrained"], pb)
    status = "FEASIBLE" if best is not None or not spec["constrained"] else "INFEASIBLE"
    if best is None:
        if not cands:
            # no refined candidate at all: descriptive release of the first registered start map, unchanged
            name, labs = start_maps[0]
            win_name, win_kind = name, "unchanged_descriptive"
            final_labels = labs
        else:
            name, kind_, obj, _ = cands[0]
            win_name, win_kind = name, kind_ + "_descriptive"
            final_labels = ({r: obj.canonical_labels(r) for r in (1, 2)} if kind_ == "refined" else obj[0])
    else:
        name, kind_, obj, _ = cands[best[1]]
        win_name, win_kind = name, kind_
        final_labels = ({r: obj.canonical_labels(r) for r in (1, 2)} if kind_ == "refined" else obj[0])
    # final state (from scratch) with SEX for the record
    pb_rec = pb
    if not spec["sex"]:
        pb_rec = Problem(fine1, fine2, P, dd, Y, s, caps, budget, dec)
    if spec["constrained"]:
        pb_rec.capI = pb.capI
    fct = new_counters()
    fst = State(pb_rec, final_labels, counters=fct)
    ft = fst.terms()
    # policy pair, decoders, release (lcr.decoder helpers)
    fam = arm
    m1, m2 = caps
    pol1 = RL.make_policy(1, fine1, fst.canonical_labels(1), fam)
    pol2 = RL.make_policy(2, fine2, fst.canonical_labels(2), fam)
    pair = RL.make_pair(pol1, pol2, fam, m1, m2, lam, {**meta, "config": cid})
    RL.check_bound(pair)
    tok_fit = {}
    for i, pol in ((1, pol1), (2, pol2)):
        tok_fit[i], _, _ = RL.encode(pol, P[i], dd[i])
    dec1 = dec.decode_policy(pol1, tok_fit[1], P[1], Y[1], config=cid)
    dec2 = dec.decode_policy(pol2, tok_fit[2], P[2], Y[2], config=cid)
    body = dec.decoder_pair_dict(cid, pair, dec1, dec2)
    release = dec.release_arrays_d1(pair, dec1, dec2, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    dep, _ = deployed_terms(pb_rec, pair, release, tr)
    # parity: state (search arithmetic) vs deployed rows
    par = {}
    for k in ("L1", "L2", "B1", "B2", "I1", "I2", "I12"):
        par[k] = abs(ft[k] - dep[k])
    q_eq = {}
    for i, dtab in ((1, dec1), (2, dec2)):
        canon_slot = {}
        for j, m in enumerate(fst.members[i]):
            if m:
                canon_slot[int(pol1.cell_token[m[0]] if i == 1 else pol2.cell_token[m[0]])] = j
        diffs = [float(np.max(np.abs(dtab.q[t] - fst.tQ[i][j]))) for t, j in canon_slot.items() if fst.tn[i][j] > 0]
        bit = all(np.array_equal(dtab.q[t], fst.tQ[i][j]) for t, j in canon_slot.items() if fst.tn[i][j] > 0)
        q_eq[str(i)] = {"bitwise_equal": bool(bit), "max_abs_diff": max(diffs) if diffs else 0.0}
    parity_ok = all(v <= PARITY_ATOL for k, v in par.items() if k[0] in "LB") and \
        all(v <= MI_PARITY_ATOL for k, v in par.items() if k[0] == "I")
    if not parity_ok:
        raise AssertionError(f"state vs deployed parity failed: {par}")
    dep_cons = {}
    for i in (1, 2):
        dc = {"L": dep[f"L{i}"], "L_limit": pb.limL[i], "L_slack": pb.limL[i] - dep[f"L{i}"],
              "B": dep[f"B{i}"], "B_limit": pb.limB[i], "B_slack": pb.limB[i] - dep[f"B{i}"]}
        dc["budgets_ok"] = bool(dc["L_slack"] >= -DEPLOYED_TOL and dc["B_slack"] >= -DEPLOYED_TOL)
        if spec["constrained"]:
            dc.update({"I": dep[f"I{i}"], "I_limit": pb.capI[i], "I_slack_table_exact": pb.capI[i] - ft[f"I{i}"],
                       "I_slack_mi_plugin": pb.capI[i] - dep[f"I{i}"]})
            dc["local_cap_ok"] = bool(dc["I_slack_table_exact"] >= -DEPLOYED_TOL)
        dep_cons[str(i)] = dc
    deployed_feasible = all(v["budgets_ok"] and v.get("local_cap_ok", True) for v in dep_cons.values())
    if spec["constrained"] and status == "FEASIBLE" and not deployed_feasible:
        raise AssertionError("winner feasible in search but not on the deployed release")
    work_tot = {k: 0 for k in ("evals", "solves", "solve_calls", "memo_hits", "memo_misses", "memo_verified", "accepted",
                               "rejected_on_exact_recheck", "pair_evals", "pair_pool_screened", "pair_accepted",
                               "pair_rejected_infeasible")}
    rej_tot = {"L": 0, "B": 0, "I": 0}
    cpu_starts = 0.0
    for r_ in start_recs:
        for k in work_tot:
            work_tot[k] += int(r_["work"][k])
        for k in rej_tot:
            rej_tot[k] += int(r_["work"]["rejected_by_budget"][k])
        cpu_starts += float(r_["cpu_s"])
    rec = {"schema": SCHEMA, "arm": arm, "config": cid, "lam": lam, "fixture_mode": fixture, "status": status,
           "objective_weights": {"wL": w[0], "wB": w[1], "wI": w[2], "w12": w[3]},
           "sex_used_in_search": bool(spec["sex"]),
           "budgets": {"L_U": {"1": pb.LU[1], "2": pb.LU[2]}, "B_U": {"1": pb.BU[1], "2": pb.BU[2]},
                       "L_limit": {"1": pb.limL[1], "2": pb.limL[2]}, "B_limit": {"1": pb.limB[1], "2": pb.limB[2]},
                       "allowance": budget, "enforced": bool(spec["constrained"]), "caps": list(caps)},
           "refs": ref_rec, "starts_given": order, "start_fingerprints": start_fps,
           "starts": start_recs,
           "winner": {"start": win_name, "kind": win_kind, "objective": objective(ft, w)},
           "final_state_terms": {**ft, "objective": objective(ft, w)},
           "final_state_constraints": {str(r): constraint_values(pb_rec, ft, r) for r in (1, 2)},
           "deployed": {"terms": dep, "constraints": dep_cons, "feasible": bool(deployed_feasible),
                        "parity_abs_diff": par, "parity_ok": bool(parity_ok), "q_vs_state": q_eq},
           "token_counts": {str(r): fst.token_counts(r) for r in (1, 2)},
           "tokens_total": int(sum(fst.token_counts(1)) + sum(fst.token_counts(2))),
           "pair_fingerprint": pair.fingerprint(), "decoder_sha256": body["decoder_sha256"],
           "decoder_stats_hash": {"1": dec1.stats_hash(), "2": dec2.stats_hash()},
           "certificates": {"1": dec.certificate_summary(dec1), "2": dec.certificate_summary(dec2)},
           "work": {**work_tot, "rejected_by_budget": rej_tot, "eval_ceiling": E,
                    "cpu_s_starts": cpu_starts},
           "rules_sha256": rules_sha256(), "code_sha256": code_hashes()}
    if arm == "C-TASK":
        st0 = State(pb_rec, start_maps[0][1], counters=new_counters())
        t0d1 = st0.terms()
        rec["ctask_report"] = {"d1_initialisation": {"start": start_maps[0][0], "terms": t0d1},
                               "refined": ft, "d0_same_maps": {"start": d0_terms(st0), "refined": d0_terms(fst)},
                               "d0_external_references": refs.get("D0")}
    rec["wall_s_mapper"] = time.perf_counter() - t0
    rec["cpu_s_mapper"] = time.process_time() - c0
    files = {"policy.json": pair.to_dict(), "decoder.json": body, "release.npz": release}
    return rec, files



# ------------------------------------------------------------------ synthetic timing (TIMING.json["fitting"])
def synthetic_labels(T, tr, seed):
    """SYNTHETIC true labels on the fitting rows from a deliberately miscalibrated law of the synthetic teacher
    (income: logit scaled by 1.25; occupation: log-probabilities tempered by 0.8). No real label is read."""
    rng = np.random.default_rng([int(seed), 11])
    p1 = np.clip(np.asarray(T["p1"])[tr][:, 1], 1e-9, 1 - 1e-9)
    z = 1.25 * np.log(p1 / (1 - p1))
    y1 = (rng.random(tr.size) < 1 / (1 + np.exp(-z))).astype(np.int64)
    L = 0.8 * np.log(np.clip(np.asarray(T["p2"])[tr], 1e-12, 1.0))
    E = np.exp(L - L.max(1, keepdims=True))
    Pt = E / E.sum(1, keepdims=True)
    u = rng.random(tr.size)[:, None]
    y2 = (u > np.cumsum(Pt, 1)).sum(1).clip(0, Pt.shape[1] - 1).astype(np.int64)
    return {1: y1, 2: y2}


def _synthetic_seed(seed):
    from cbp import fit as CF
    T, tr, S_fit = CF.synthetic_teacher(seed=seed)
    Y = synthetic_labels(T, tr, seed)
    P1, d1, P2, d2 = T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr]
    c0 = time.process_time()
    f1, f2, _ = PT.fit_fine_pair(P1, d1, P2, d2)
    fine = {"fine1": f1.to_dict(), "fine2": f2.to_dict()}
    src = {}
    pp, _ = QC.fit_policy_pair("FINE-TASK", f1, f2, P1, d1, P2, d2, S_fit, 8, 64, None, baseline_diagnostic=False)
    src[RN.d0_id("FINE-TASK")] = pp
    for lam in RN.LAMS:
        wit = {"FINE-TASK": src[RN.d0_id("FINE-TASK")]}
        for fam in RN.PRIVACY:
            w = wit if fam == "JOINT" else None
            pp, _ = QC.fit_policy_pair(fam, f1, f2, P1, d1, P2, d2, S_fit, 8, 64, lam, witnesses=w,
                                       baseline_diagnostic=False)
            src[RN.d0_id(fam, lam)] = pp
            if fam != "JOINT":
                wit[fam] = pp
    return T, tr, S_fit, Y, fine, {k: v.to_dict() for k, v in src.items()}, time.process_time() - c0


def _d1_decode(cid, pz, T, tr, Y, meta):
    dec = _dec()
    pair = RL.PolicyPair.from_dict(pz)
    P = {1: T["p1"][tr], 2: T["p2"][tr]}
    tabs = []
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok, _, _ = RL.encode(pol, P[i], np.asarray(T[f"d{i}"])[tr])
        tabs.append(dec.decode_policy(pol, tok, P[i], Y[i], config=cid + "|D1"))
    dec.release_arrays_d1(pair, tabs[0], tabs[1], T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    return tabs


def timing(seeds=(0,), out_path=None, units="all"):
    """Full registered new-fit bank (30 units per seed: C-TASK, 5 constrained, 24 weighted) plus the 26 D1 fixed-map
    decodes per seed, on SYNTHETIC real-shaped data. Writes TIMING.json["fitting"] atomically (other keys kept)."""
    import platform
    import resource
    import sys
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta0 = {"teacher": "U", "teacher_model_sha256": "a" * 64, "feature_names_sha256": "b" * 64}
    _dec()                                            # import the decoder now, so the hashed code is the code run
    code0 = code_hashes()
    per_seed = []
    for k in seeds:
        T, tr, S_fit, Y, fine, src, setup_cpu = _synthetic_seed(k)
        done = {}
        rows = {}

        def run(arm, lam=None, starts=None, witnesses=None, refs=None):
            cid = config_id(arm, lam)
            c0, w0 = time.process_time(), time.perf_counter()
            rec, files = fit_unit(arm, fine, T, tr, Y, S_fit, lam=lam, starts=starts, witnesses=witnesses,
                                  refs=refs, meta={**meta0, "seed": k, "config": cid})
            cpu, wall = time.process_time() - c0, time.perf_counter() - w0
            done[cid] = (rec, files["policy.json"])
            wk = rec["work"]
            rows[cid] = {"arm": arm, "lam": lam, "cpu_s": cpu, "wall_s": wall, "status": rec["status"],
                         "winner": rec["winner"]["start"] + ":" + rec["winner"]["kind"],
                         "evals": wk["evals"], "solves": wk["solves"], "solve_calls": wk["solve_calls"],
                         "memo_hits": wk["memo_hits"], "accepted": wk["accepted"], "pair_evals": wk["pair_evals"],
                         "pair_accepted": wk["pair_accepted"], "rejected_by_budget": wk["rejected_by_budget"],
                         "tokens_total": rec["tokens_total"], "deployed_feasible": rec["deployed"]["feasible"],
                         "parity_max": max(rec["deployed"]["parity_abs_diff"].values()),
                         "stops": sorted({x.get("stop") for s in rec["starts"] for x in
                                          ([s] + s.get("stages", []) + [s.get("stage1", {}), s.get("stage2", {})])
                                          if isinstance(x, dict) and x.get("stop")})}
            sys.stderr.write(f"[timing s{k}] {cid}: {cpu:.1f} CPU-s evals {wk['evals']} status {rec['status']}\n")
            return rec

        def P_(cid):
            return done[cid][1]
        ct = run("C-TASK", starts={RN.d0_id("FINE-TASK"): src[RN.d0_id("FINE-TASK")]})
        refs = refs_from_ctask(ct)
        seq_src = {a: {RN.d0_id(a, l): src[RN.d0_id(a, l)] for l in RN.LAMS} for a in ("SEQ-12", "SEQ-21")}
        src_all = {RN.d0_id("FINE-TASK"): src[RN.d0_id("FINE-TASK")],
                   **{RN.d0_id(f, l): src[RN.d0_id(f, l)] for l in RN.LAMS for f in RN.PRIVACY}}
        if units in ("all", "constrained"):
            run("K-LOCAL", starts={RN.ctask_id(): P_(RN.ctask_id())}, refs=refs)
            for a in ("SEQ-12", "SEQ-21"):
                run(f"K-{a}", starts={RN.ctask_id(): P_(RN.ctask_id()), **seq_src[a]}, refs=refs)
            wit = {RN.ctask_id(): P_(RN.ctask_id()), RN.constrained_id("LOCAL"): P_(RN.constrained_id("LOCAL")),
                   RN.constrained_id("SEQ-12"): P_(RN.constrained_id("SEQ-12")),
                   RN.constrained_id("SEQ-21"): P_(RN.constrained_id("SEQ-21")), **src_all}
            for a in ("JOINT-SINGLE", "JOINT-PAIR"):
                run(f"K-{a}", witnesses=wit, refs=refs)
        if units in ("all", "weighted"):
            for lam in RN.LAMS:
                run("W-LOCAL", lam, starts={RN.ctask_id(): P_(RN.ctask_id())})
                for a in ("SEQ-12", "SEQ-21"):
                    run(f"W-{a}", lam, starts={RN.ctask_id(): P_(RN.ctask_id()), **seq_src[a]})
                wit = {RN.ctask_id(): P_(RN.ctask_id()), **{RN.weighted_id(f, lam): P_(RN.weighted_id(f, lam))
                                                             for f in ("LOCAL", "SEQ-12", "SEQ-21")}, **src_all}
                run("W-JOINT", lam, witnesses=wit)
        dec_rows = {}
        d1_maps = [RN.d0_id("FINE-TASK"), RN.d0_id("FINE-TASK")] + [RN.d0_id(f, l) for l in RN.LAMS
                                                                     for f in RN.PRIVACY]
        for j, cid in enumerate(d1_maps):
            c0 = time.process_time()
            _d1_decode(cid, src[cid], T, tr, Y, meta0)
            dec_rows[f"{j:02d}:{cid}" + ("(DIRECT-TASK proxy)" if j == 0 else "")] = time.process_time() - c0
        per_seed.append({"seed": k, "setup_cpu_s": setup_cpu, "units": rows, "d1_fixed_map_decodes": dec_rows})
    fit_cpu = [sum(v["cpu_s"] for v in s["units"].values()) for s in per_seed]
    dec_cpu = [sum(s["d1_fixed_map_decodes"].values()) for s in per_seed]
    by_arm = {}
    for s in per_seed:
        for v in s["units"].values():
            by_arm.setdefault(v["arm"], []).append(v["cpu_s"])
    n_seeds = len(per_seed)
    mean_fit = float(np.mean(fit_cpu)) if fit_cpu else 0.0
    mean_dec = float(np.mean(dec_cpu)) if dec_cpu else 0.0
    proj = 3 * (mean_fit + mean_dec)
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec = {"owner": "role C (mapper)", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "started_at": t_start,
           "data": "SYNTHETIC ONLY: cbp.fit.synthetic_teacher (39,170 rows; a fixed 15,434 fitting rows; income K=2 "
                   "with 2 predicted classes; occupation K=6 with 5 predicted classes, class 5 never predicted; "
                   "binary S correlated with both scores) + lcr.mapper.synthetic_labels (miscalibrated law); fine "
                   "partitions qpc.partition 32/128 per class; source D0 maps fitted with plain qpc (setup, not "
                   "counted). No Adult row, label or SEX was read.",
           "environment": {"threads": "OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, one process",
                           "semaphore": "lcr.sema (label C:fit-timing)", "python": platform.python_version(),
                           "numpy": np.__version__, "machine": platform.machine()},
           "measure": "process CPU seconds per unit (time.process_time), including the decoder.json/release.npz "
                      "build and the deployed recomputation",
           "code_sha256_at_start": code0, "code_unchanged_during_run": bool(code_hashes() == code0),
           "seeds_measured": list(seeds), "units_per_seed": len(per_seed[0]["units"]) if per_seed else 0,
           "d1_fixed_map_decodes_per_seed": 26,
           "measured": {"fit_cpu_s_per_seed": fit_cpu, "d1_decode_cpu_s_per_seed": dec_cpu,
                        "cpu_s_by_arm": {a: {"units": len(v), "total": float(sum(v)), "mean": float(np.mean(v)),
                                             "max": float(max(v))} for a, v in by_arm.items()},
                        "setup_cpu_s_per_seed_not_counted": [s["setup_cpu_s"] for s in per_seed],
                        "peak_rss_bytes": int(maxrss if sys.platform == "darwin" else maxrss * 1024)},
           "per_seed": per_seed,
           "projection": {"bank_3_seeds_cpu_s": proj, "bank_3_seeds_cpu_h": proj / 3600,
                          "conservative_x2_cpu_h": 2 * proj / 3600,
                          "basis": f"mean of {n_seeds} measured synthetic seed(s) x 3; qpc/cbp real/synthetic ratio "
                                   "about 1.0 for the comparable bank",
                          "budget_line": "fitting should be well under 3 CPU-h of the 20 CPU-h study"},
           "recommendation": None}
    rec["recommendation"] = {
        "bank": "FULL (90 new units + 78 D1 fixed-map decodes); no reduction",
        "sharding": "2 shards under lcr.sema: lcr.run fit_chains (constrained chains first, then weighted); "
                    "ctask stage first (3 units), d1 stage independent",
        "within_budget": bool(2 * proj / 3600 < 3.0)}
    rec = RN._finite(rec)
    if out_path is not None:
        from pathlib import Path
        p = Path(out_path)
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur["fitting"] = rec
        tmp = p.with_name(p.name + ".tmp_fitting")
        tmp.write_text(json.dumps(cur, indent=1, allow_nan=False) + "\n")
        tmp.replace(p)
    return rec


# ------------------------------------------------------------------ CLI
def main(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m lcr.mapper")
    ap.add_argument("cmd", choices=("rules", "timing"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--units", default="all")
    a = ap.parse_args(argv)
    if a.cmd == "rules":
        z = rules()
        z["rules_sha256"] = rules_sha256()
        s = json.dumps(z, indent=1, allow_nan=False) + "\n"
        if a.out:
            open(a.out, "w").write(s)
        else:
            sys.stdout.write(s)
        return 0
    import os
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1 (run under lcr.sema)")
    r = timing(tuple(int(x) for x in a.seeds.split(",")), a.out, units=a.units)
    sys.stdout.write(json.dumps({k: r[k] for k in ("measured", "projection", "recommendation")}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
