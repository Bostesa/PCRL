"""[lra port of lcr/mapper.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Learned-decoder mapper for the lra study (role C): constrained and weighted coarse assignment search with the D1
decoder (prompt sections 7 and 8). Every frozen rule is a module constant below and in SEARCH_RULES.json.

Source of the search engine: qpc/compress.py (pinned at 7f3ec67, imported for canonicalisation helpers only, never
edited). Its sufficient-statistic tables, canonical labels, exact n log n plug-in MI and the vectorised single-cell
move deltas are re-implemented here for the new objective; the qpc greedy merge-to-cap phase is NOT used (every
proposal here moves a WHOLE fine cell to an existing same-class token; tokens can empty, never be created).

Fitting quantities on the N fitting rows (OSF_DEFENSE_FIT), recipient i in {1, 2}, token t with fitting count n_t,
label counts y_t, teacher sums s_t (canonical accumulation over member fine cells in increasing index = qpc
token_tables = lra.decoder.token_stats, bitwise), SEX counts:
    q_t   = lra.decoder.solve_batch(y_t, s_t, n_t, d_t)  (the ACTUAL released, eps-smoothed, class-dominant vector)
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
        -> (record, {"policy.json": dict, "decoder.json": dict, "release.npz": dict of arrays, "trace.json": dict})
        files also holds "trace.json" (persisted starts, accepted moves, statistics hashes, termination receipts)
    refs_from_ctask(ctask_record) -> refs for the K- arms
    replay_unit(record, fine_dict, T, tr, Y_fit, S_fit, refs=None, *, trace, files=None, per_state=True, starts=None)
        -> report: independent from-scratch replay of a unit's trace (correctness checks 6, 7, 8)
    python -m lra.mapper rules [--out SEARCH_RULES.json]     print / write the frozen rules
    python -m lra.mapper timing [--out TIMING.json] [--seeds 0]   SYNTHETIC full-bank timing (under lra.sema)
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from dpc.compress import mi_plugin
from dpc.utility import per_row
from lra import run as RN
from qpc import compress as QC
from qpc import kmeans as KM
from qpc import partition as PT
from qpc import release as RL

SCHEMA = "lra-mapper-v1"
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
# ------------------------------------------------------------------ persisted trace and replay (prompt sec. 8, check 8)
TRACE_SCHEMA = "lra-mapper-trace-v1"
REPLAY_TERM_ATOL = 1e-12           # |trace state term - from-scratch term| (L1 L2 B1 B2 I1 I2 I12 Phi T), every state
REPLAY_DELTA_ATOL = 1e-12          # |vectorised per-move delta - from-scratch difference| per term (dL dB dI dI12)
#                                    objective delta: REPLAY_DELTA_ATOL * (1 + sum |w|)
HASH_RULE = {"stats_sha256": "sha256(n_t as <i8 || y_t as <f8 (K) || s_t as <f8 (K)) of one token",
             "q_sha256": "sha256(q_t as <f8 (K)), the released D1 vector; null for a token with n_t = 0 (no solve)",
             "state_stats_sha256": "sha256 over recipients 1, 2 and their nonempty tokens in increasing canonical "
                                   "label of (canonical label <i8 || n_t <i8 || y_t <f8 || s_t <f8)",
             "state_q_sha256": "sha256 over the same tokens with n_t > 0 of (canonical label <i8 || q_t <f8)",
             "trace_sha256": "sha256 of json.dumps(trace, sort_keys=True, separators=(',', ':'), allow_nan=False)"}
MOVE_FIELDS = {"move": ["step", "sweep", "type (single|pair)", "parts", "terms_after", "objective_after"],
               "part": ["r", "f", "from_canon", "to_canon", "from_slot", "to_slot", "stats_sha256 [from, to] after",
                        "q_sha256 [from, to] after", "single: delta dL dB dI dI12 (vectorised)",
                        "pair: dL dB dI dI12 dPhi_one_sided dTask (one-sided, vectorised, on the pre-pair state)"]}
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
    src_fine = [RN.d0_id("FINE-TASK"), RN.d0_id("CLASS")]     # CLASS: a feasible-only witness (lead, 2026-10-07)
    if arm in ("K-JOINT-SINGLE", "K-JOINT-PAIR"):
        return [RN.ctask_id(), RN.constrained_id("LOCAL"), RN.constrained_id("SEQ-12"),
                RN.constrained_id("SEQ-21")] + src_fine + src_priv
    if arm == "W-JOINT":
        return [RN.ctask_id(), RN.weighted_id("LOCAL", lam), RN.weighted_id("SEQ-12", lam),
                RN.weighted_id("SEQ-21", lam)] + src_fine + src_priv
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
    from lra import decoder
    return decoder


# ------------------------------------------------------------------ helpers
def _ssum(x):
    """Order-independent sum of the NONZERO entries (sorted, then numpy pairwise): a function of the multiset."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[x != 0.0]
    return float(np.sort(x).sum()) if x.size else 0.0


def _sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def stats_sha(n, Y, S):
    """HASH_RULE stats_sha256 of one token's sufficient statistics."""
    return _sha(np.asarray(n, dtype="<i8").reshape(1), np.asarray(Y, dtype="<f8"), np.asarray(S, dtype="<f8"))


def q_sha(q):
    return _sha(np.asarray(q, dtype="<f8"))


def json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def canon_labels(lab):
    """Canonical labels of a grouping: each cell's label = the lowest fine index of its group."""
    lab = np.asarray(lab, dtype=np.int64)
    _, inv = np.unique(lab, return_inverse=True)
    mins = np.full(int(inv.max()) + 1, lab.size, dtype=np.int64)
    np.minimum.at(mins, inv, np.arange(lab.size, dtype=np.int64))
    return mins[inv]


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
        """Canonical sums over member cells (sorted, PAD rows are zeros): bitwise lra.decoder.token_stats."""
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

    # -------------------------------------------------------------- trace helpers (HASH_RULE)
    def labels_json(self):
        return {str(r): self.canonical_labels(r).tolist() for r in (1, 2)}

    def slot_hashes(self, r, j):
        """(stats_sha256, q_sha256) of slot j as the search holds it (memo-sourced after a move); None if emptied /
        unsolved."""
        if self.mlen[r][j] == 0:
            return None, None
        hs = stats_sha(self.tn[r][j], self.tY[r][j], self.tS[r][j])
        return hs, (q_sha(self.tQ[r][j]) if self.tn[r][j] > 0 else None)

    def state_hashes(self):
        hs, hq = hashlib.sha256(), hashlib.sha256()
        for r in (1, 2):
            al = np.flatnonzero(self.mlen[r] > 0)
            al = al[np.argsort(self.minm[r][al], kind="stable")]
            for j in al:
                c = np.asarray(self.minm[r][j], dtype="<i8").reshape(1)
                hs.update(c.tobytes() + np.asarray(self.tn[r][j], dtype="<i8").reshape(1).tobytes()
                          + np.asarray(self.tY[r][j], dtype="<f8").tobytes()
                          + np.asarray(self.tS[r][j], dtype="<f8").tobytes())
                if self.tn[r][j] > 0:
                    hq.update(c.tobytes() + np.asarray(self.tQ[r][j], dtype="<f8").tobytes())
        return hs.hexdigest(), hq.hexdigest()

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
        out["move"] = (int(f), int(B[j]), float(delta[j]), j)
    return out


_TRACE_CPU = {"s": 0.0}            # CPU spent building the persisted trace (per unit; reset by fit_unit)


def _opt(x, j):
    return None if x is None else float(x[j])


def _trace_part(st, r, f, a, b, from_canon, to_canon, extra):
    """One move part of the persisted trace; hashes are of the two affected slots AFTER the move (the values the
    search holds, i.e. the memo-served statistics and cached solves)."""
    c0 = time.process_time()
    ha, qa = st.slot_hashes(r, a)
    hb, qb = st.slot_hashes(r, b)
    out = {"r": int(r), "f": int(f), "from_canon": int(from_canon), "to_canon": int(to_canon), "from_slot": int(a),
           "to_slot": int(b), **extra, "stats_sha256": [ha, hb], "q_sha256": [qa, qb]}
    _TRACE_CPU["s"] += time.process_time() - c0
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
    strict Phi improvement (ties: first in (i, j) order).
    budget_left = the proposal evaluations still allowed to this start/stage (its ceiling share minus what it used).
    The pool screening and the pair evaluations BOTH count and BOTH stop at it (F R-4: the screening used to run
    unchecked, with the pair budget taken from the pre-screening remainder, so one step could overshoot a share by one
    full screening pass); a step that reaches it sets info["hit_ceiling"] and the stage stops at 'eval_ceiling'."""
    pb = st.pb
    lim = st.ct["evals"] + max(0, int(budget_left))
    props = {}
    for r in (1, 2):
        pool = []
        st.prefill(r)
        for f in range(pb.fine[r].F):
            if pb.ncell[r][f] == 0:
                continue
            if st.ct["evals"] >= lim:
                return False, cur, cur_obj, {"proposals": None, "accepted": None, "hit_ceiling": True,
                                             "aborted_in": "pool_screening"}
            res = _cell(st, r, f, w, cur, cons, need_feas_all=True)
            if res is None:
                continue
            st.ct["pair_pool_screened"] += int(res["B"].size)
            for k, hk in enumerate(res["hood"]):
                if not res["own_feasible"][k]:
                    continue
                dphi = float(res["dI12"][hk] + PHI_LOCAL_WEIGHT * res["dI"][hk])
                dtask = float(res["dL"][hk] + TASK_BRIER_WEIGHT * res["dB"][hk])
                pool.append((dphi, dtask, int(f), int(res["canon"][hk]), int(res["B"][hk]),
                             {"dL": float(res["dL"][hk]), "dB": float(res["dB"][hk]), "dI": float(res["dI"][hk]),
                              "dI12": float(res["dI12"][hk]), "dPhi_one_sided": dphi, "dTask": dtask}))
        by_phi = sorted(pool, key=lambda x: (x[0], x[2], x[3]))[:PAIR_KEEP_PHI]
        chosen = {(x[2], x[4]) for x in by_phi}
        rest = [x for x in pool if (x[2], x[4]) not in chosen]
        by_task = sorted(rest, key=lambda x: (x[1], x[2], x[3]))[:PAIR_KEEP_TASK]
        props[r] = by_phi + by_task
    best = None
    hit = False
    for i, m1 in enumerate(props[1]):
        for j, m2 in enumerate(props[2]):
            if st.ct["evals"] >= lim:
                hit = True
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
        if hit:
            break
    info = {"proposals": {str(r): [{"fine_cell": x[2], "target_canon": x[3], "dPhi_one_sided": x[0],
                                     "dTask": x[1]} for x in props[r]] for r in (1, 2)}, "accepted": None,
            "hit_ceiling": hit}
    if best is None:
        return False, cur, cur_obj, info
    v, i, j, m1, m2 = best
    pre = {r: (int(st.lab[r][m[2]]), int(st.minm[r][st.lab[r][m[2]]])) for r, m in ((1, m1), (2, m2))}
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
    info["trace_move"] = {"type": "pair",
                          "parts": [_trace_part(st, r, m[2], pre[r][0], m[4], pre[r][1], m[3], m[5])
                                    for r, m in ((1, m1), (2, m2))],
                          "pair_index": [i, j], "terms_after": dict(t), "objective_after": vv}
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
    log, pairs, tmoves = [], [], []
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
                _, b, dlt, j = res["move"]
                a, fc, tc = int(res["a"]), int(st.minm[r][res["a"]]), int(res["canon"][j])
                ok, cur, cur_obj = _accept(st, r, f, b, w, cons, cur, cur_obj)
                if ok:
                    changed += 1
                    log.append([sweep, r, int(f), a, int(b), dlt])
                    ext = {"delta": dlt, "dL": float(res["dL"][j]), "dB": float(res["dB"][j]),
                           "dI": _opt(res["dI"], j), "dI12": _opt(res["dI12"], j)}
                    tmoves.append({"step": len(tmoves), "sweep": sweep, "type": "single",
                                   "parts": [_trace_part(st, r, f, a, b, fc, tc, ext)], "terms_after": dict(cur),
                                   "objective_after": cur_obj})
            if hit_ceiling:
                break
        if pair and not hit_ceiling:
            ok, cur, cur_obj, info = _pair_step(st, w, cons, cur, cur_obj,
                                                budget_left=max(0, ceiling - (st.ct["evals"] - e0)))
            info["sweep"] = sweep
            tm = info.pop("trace_move", None)
            pairs.append(info)
            if ok:
                changed += 1
                tmoves.append({"step": len(tmoves), "sweep": sweep, **tm})
            if info.get("hit_ceiling"):
                hit_ceiling = True
        if hit_ceiling:
            stop = "eval_ceiling"
            break
        if changed == 0:
            stop = "no_change_sweep"
            break
    if cur_obj > start_obj:
        raise AssertionError("refinement increased the objective")
    return {"sweeps": sweeps, "stop": stop, "moves": log, "pair_steps": pairs, "objective_start": start_obj,
            "objective_end": cur_obj, "evals": st.ct["evals"] - e0, "ceiling_share": int(ceiling),
            "trace_moves": tmoves}


def _termination(s):
    """Termination receipt of one refined stage (trace)."""
    mv = s["trace_moves"]
    return {"stop": s["stop"], "sweeps": s["sweeps"], "evals": int(s["evals"]), "ceiling_share": s["ceiling_share"],
            "accepted": len(mv), "pairs_accepted": sum(1 for m in mv if m["type"] == "pair"),
            "pair_steps": len(s["pair_steps"]), "objective_start": s["objective_start"],
            "objective_end": s["objective_end"]}


def _trace_stage(stage, st, recips, enforced, status="REFINED", **extra):
    """A stage of the persisted trace; call BEFORE refining (start labels/terms/hashes), then _close_stage."""
    c0 = time.process_time()
    hs, hq = st.state_hashes()
    out = {"stage": stage, "recipients": list(recips), "enforced": [int(r) for r in enforced],
           "start_labels": st.labels_json(), "start_terms": st.terms(), "start_stats_sha256": hs,
           "start_q_sha256": hq, "status": status, **extra, "moves": [], "termination": None}
    _TRACE_CPU["s"] += time.process_time() - c0
    return out


def _close_stage(tstage, s):
    """Move the refine() trace moves into the trace stage (they are NOT kept in the record) + receipt."""
    tstage["moves"] = s.pop("trace_moves")
    tstage["termination"] = _termination({**s, "trace_moves": tstage["moves"]})
    return tstage


# ------------------------------------------------------------------ arm drivers
def _start_record(name, st, w, pb, cons_all):
    t = st.terms()
    return {"name": name, "terms": t, "objective": objective(t, w),
            "constraints": {str(r): constraint_values(pb, t, r) for r in (1, 2)},
            "feasible_all": bool(feasible(pb, t, cons_all)) if cons_all else None,
            "token_counts": {str(r): st.token_counts(r) for r in (1, 2)}}


def _class_only(fine):
    return QC.class_labels(fine)


def _lab_json(labs):
    return {str(r): canon_labels(labs[r]).tolist() for r in (1, 2)}


def _run_local(pb, arm, w, starts, cons_flag, E):
    out, cands, tr_ = [], [], []
    for pos, (name, labs) in enumerate(starts):
        ct = new_counters()
        c0 = time.process_time()
        st = State(pb, labs, sex=SPEC[arm]["sex"], counters=ct)
        rec = {"start": _start_record(name, st, w, pb, (1, 2) if cons_flag else ()), "stages": []}
        ts = {"name": name, "position": pos, "labels": st.labels_json(), "stages": []}
        ok_all = True
        for r in (1, 2):
            t = st.terms()
            cons = (r,) if cons_flag else ()
            tstage = _trace_stage(f"r{r}", st, (r,), cons)
            if cons_flag and not feasible(pb, t, cons):
                rec["stages"].append({"recipient": r, "status": "INFEASIBLE_START", "refined": False})
                tstage["status"] = "INFEASIBLE_START"
                ts["stages"].append(tstage)
                ok_all = False
                continue
            s = refine(st, (r,), w, cons, ceiling=E // (2 * len(starts)))
            ts["stages"].append(_close_stage(tstage, s))
            s.update({"recipient": r, "status": "REFINED"})
            rec["stages"].append(s)
        rec["final"] = _start_record(name, st, w, pb, (1, 2) if cons_flag else ())
        rec["eligible"] = bool(ok_all and (not cons_flag or rec["final"]["feasible_all"]))
        rec["work"] = ct
        rec["cpu_s"] = time.process_time() - c0
        ts.update({"final_labels": st.labels_json(), "eligible": rec["eligible"]})
        out.append(rec)
        tr_.append(ts)
        cands.append((name, "refined", st, rec))
    return out, cands, tr_


def _run_seq(pb, arm, w, starts, cons_flag, E):
    a, b = SPEC[arm]["order"]
    out, cands, tr_ = [], [], []
    E_stage = E // (2 * len(starts))
    for pos, (name, labs) in enumerate(starts):
        ct = new_counters()
        c0 = time.process_time()
        part = _class_only(pb.fine[b])
        lab1 = {a: labs[a], b: part}
        st1 = State(pb, lab1, counters=ct)
        rec = {"name": name, "stage1": {"recipient": a, "partner": "CLASS-ONLY",
                                         "partner_constraints_enforced": False,
                                         "start": _start_record(name, st1, w, pb, (a,) if cons_flag else ())}}
        ts = {"name": name, "position": pos, "labels": _lab_json(labs), "stages": []}
        cons1 = (a,) if cons_flag else ()
        t1s = st1.terms()
        pc = constraint_values(pb, t1s, b)
        t1 = _trace_stage("seq1", st1, (a,), cons1, partner={
            "recipient": int(b), "view": "CLASS-ONLY", "labels": canon_labels(part).tolist(),
            "constraints_enforced": False, "constraint_values_at_start": pc,
            "violates_own_budgets_at_start": bool(pc["L_slack"] < 0 or pc["B_slack"] < 0)})
        ts["stages"].append(t1)
        if cons_flag and not feasible(pb, t1s, cons1):
            rec["stage1"]["status"] = "INFEASIBLE_START"
            t1["status"] = "INFEASIBLE_START"
            rec["eligible"] = False
            rec["work"] = ct
            rec["cpu_s"] = time.process_time() - c0
            ts["eligible"] = False
            out.append(rec)
            tr_.append(ts)
            continue
        s1 = refine(st1, (a,), w, cons1, ceiling=E_stage)
        _close_stage(t1, s1)
        s1["status"] = "REFINED"
        rec["stage1"].update(s1)
        t1e = st1.terms()
        rec["stage1"]["end"] = {"terms": t1e, "partner_class_only_constraints": constraint_values(pb, t1e, b)}
        t1["partner"]["constraint_values_at_end"] = constraint_values(pb, t1e, b)
        frozen = st1.canonical_labels(a)
        lab2 = {a: frozen, b: labs[b]}
        st2 = State(pb, lab2, counters=ct)
        if not np.array_equal(st2.canonical_labels(a), frozen):
            raise AssertionError("frozen first map changed")
        rec["stage2"] = {"recipient": b, "frozen": a, "start": _start_record(name, st2, w, pb,
                                                                              (b,) if cons_flag else ())}
        cons2 = (b,) if cons_flag else ()
        t2 = _trace_stage("seq2", st2, (b,), cons2, frozen={"recipient": int(a), "labels": frozen.tolist()})
        ts["stages"].append(t2)
        if cons_flag and not feasible(pb, st2.terms(), cons2):
            rec["stage2"]["status"] = "INFEASIBLE_START"
            t2["status"] = "INFEASIBLE_START"
            rec["eligible"] = False
        else:
            s2 = refine(st2, (b,), w, cons2, ceiling=E_stage)
            _close_stage(t2, s2)
            s2["status"] = "REFINED"
            rec["stage2"].update(s2)
            if not np.array_equal(st2.canonical_labels(a), frozen):
                raise AssertionError("first recipient revised after stage 1")
            rec["final"] = _start_record(name, st2, w, pb, (1, 2) if cons_flag else ())
            rec["eligible"] = bool(not cons_flag or rec["final"]["feasible_all"])
            ts["final_labels"] = st2.labels_json()
            cands.append((name, "refined", st2, rec))
        ts["eligible"] = rec["eligible"]
        rec["work"] = ct
        rec["cpu_s"] = time.process_time() - c0
        out.append(rec)
        tr_.append(ts)
    return out, cands, tr_


def _run_joint(pb, arm, w, starts, cons_flag, E):
    out, cands, tr_ = [], [], []
    E_start = E // len(starts)
    cons = (1, 2) if cons_flag else ()
    for pos, (name, labs) in enumerate(starts):
        ct = new_counters()
        c0 = time.process_time()
        st = State(pb, labs, counters=ct)
        rec = {"name": name, "unchanged": _start_record(name, st, w, pb, cons)}
        ts = {"name": name, "position": pos, "labels": st.labels_json(), "stages": [],
              "joint_witness": {"terms": rec["unchanged"]["terms"], "feasible_all": rec["unchanged"]["feasible_all"]}}
        if cons_flag and not rec["unchanged"]["feasible_all"]:
            rec.update({"status": "EXCLUDED_INFEASIBLE_WITNESS", "eligible_unchanged": False, "eligible": False,
                        "work": ct, "cpu_s": time.process_time() - c0})
            ts["joint_witness"].update({"status": "EXCLUDED_INFEASIBLE_WITNESS", "eligible_unchanged": False})
            ts["eligible"] = False
            out.append(rec)
            tr_.append(ts)
            continue
        ut = st.terms()
        snap = {r: st.canonical_labels(r) for r in (1, 2)}
        tstage = _trace_stage("joint", st, (1, 2), cons)
        s = refine(st, (1, 2), w, cons, ceiling=E_start, pair=SPEC[arm]["pair"])
        ts["stages"].append(_close_stage(tstage, s))
        rec.update(s)
        rec["final"] = _start_record(name, st, w, pb, cons)
        rec["status"] = "REFINED"
        rec["eligible_unchanged"] = True
        rec["eligible"] = bool(not cons_flag or rec["final"]["feasible_all"])
        rec["work"] = ct
        rec["cpu_s"] = time.process_time() - c0
        ts["joint_witness"].update({"status": "REFINED", "eligible_unchanged": True})
        ts.update({"final_labels": st.labels_json(), "eligible": rec["eligible"]})
        out.append(rec)
        tr_.append(ts)
        cands.append((name, "refined", st, rec))
        cands.append((name, "unchanged", (snap, ut), rec))
    return out, cands, tr_


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


# ------------------------------------------------------------------ replay of a persisted trace (checks 6, 7, 8)
REPLAY_CODES = ("TERMS_MISMATCH", "DELTA_MISMATCH", "STATS_HASH_MISMATCH", "CACHED_SOLVE_MISMATCH",
                "NOT_STRICT_IMPROVEMENT", "ENFORCED_CONSTRAINT_VIOLATED", "PAIR_INFEASIBLE", "PARTNER_NOT_CLASS_ONLY",
                "PARTNER_REQUIRED_FEASIBLE", "FROZEN_MAP_CHANGED", "FINAL_INFEASIBLE", "FINAL_STATE_MISMATCH",
                "RELEASE_MISMATCH", "WINNER_MISMATCH", "INFEASIBLE_WITNESS_ELIGIBLE", "MOVE_INVALID", "START_MISMATCH",
                "TERMINATION_INVALID", "TRACE_INCOMPLETE", "REFERENCE_MISMATCH", "TRACE_HASH_MISMATCH")
REPLAY_GATE = {"check6": ("ENFORCED_CONSTRAINT_VIOLATED", "PAIR_INFEASIBLE", "FINAL_INFEASIBLE"),
               "check7": ("PARTNER_NOT_CLASS_ONLY", "PARTNER_REQUIRED_FEASIBLE", "FROZEN_MAP_CHANGED",
                          "FINAL_INFEASIBLE"),
               "check8": ("TERMS_MISMATCH", "DELTA_MISMATCH", "STATS_HASH_MISMATCH", "CACHED_SOLVE_MISMATCH",
                          "START_MISMATCH", "TRACE_INCOMPLETE", "TRACE_HASH_MISMATCH")}
_TERM_KEYS = ("L1", "L2", "B1", "B2", "I1", "I2", "I12", "Phi", "T")


class _Defect(Exception):
    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class _Scratch:
    """From-scratch evaluator of a labelled partition on the fitting rows. It does not use State, its slots, its memo
    or its incrementally maintained tables:
    - token statistics are folded from zeros over the member cells in increasing fine index (= lra.decoder.token_stats);
    - q is a FRESH lra.decoder.solve_batch of those statistics; losses come from lra.decoder.token_losses;
    - L_i and B_i are the sorted-nonzero sums of the per-token totals divided by N;
    - MI comes from the fine-cell SEX table built once from the fitting rows, aggregated by the labels.
    Fresh solves are memoised by the exact member set, which is a pure function of the partition and the fixed data."""

    def __init__(self, fines, cell, Ycell, s, dec):
        self.fine, self.cell, self.Ycell, self.dec = fines, cell, Ycell, dec
        self.N = int(cell[1].size)
        self.memo = {1: {}, 2: {}}
        self.solves = 0
        XL = np.arange(self.N + 1, dtype=np.float64)
        self.XL = XL * np.log(np.where(XL > 0, XL, 1.0))
        F1, F2 = fines[1].F, fines[2].F
        s = np.asarray(s, dtype=np.int64)
        self.Tf = np.bincount((s * F1 + cell[1]) * F2 + cell[2], minlength=2 * F1 * F2).reshape(2, F1, F2)
        self.tf = {1: self.Tf.sum(2), 2: self.Tf.sum(1)}
        ns = np.bincount(s, minlength=2)
        self.const = float(-(self.XL[ns[0]] + self.XL[ns[1]]) + self.XL[self.N])

    def _phi(self, X):
        return self.XL[X[0]] + self.XL[X[1]] - self.XL[X[0] + X[1]]

    def groups(self, r, lab):
        order = np.argsort(lab, kind="stable")
        sl = lab[order]
        cut = np.flatnonzero(np.diff(sl)) + 1
        a = np.concatenate([[0], cut])
        b = np.concatenate([cut, [lab.size]])
        return [(int(sl[i]), order[i:j]) for i, j in zip(a, b)]

    def evaluate(self, lab):
        """(terms, tokens): tokens[r] = {canonical label: entry with n, Y, S, d, q, ll, br, stats_sha, q_sha, members}."""
        out, toks = {}, {}
        N = self.N
        for r in (1, 2):
            fine = self.fine[r]
            K = fine.K
            ents, miss = {}, []
            for c, m in self.groups(r, lab[r]):
                key = m.tobytes()
                e = self.memo[r].get(key)
                if e is None:
                    cls = np.asarray(fine.cell_class)[m]
                    if np.any(cls != cls[0]):
                        raise _Defect("MOVE_INVALID", f"recipient {r}: a token mixes predicted classes")
                    n = int(np.asarray(fine.n)[m].sum())
                    Y = np.zeros(K)
                    S = np.zeros(K)
                    for f in m:
                        Y = Y + self.Ycell[r][f]
                        S = S + fine.S[f]
                    e = {"n": n, "Y": Y, "S": S, "d": int(cls[0]), "q": None, "ll": 0.0, "br": 0.0,
                         "stats_sha": stats_sha(n, Y, S), "q_sha": None, "members": m}
                    self.memo[r][key] = e
                    if n > 0:
                        miss.append(e)
                ents[c] = e
            if miss:
                Ym = np.array([e["Y"] for e in miss])
                Sm = np.array([e["S"] for e in miss])
                _, Q, _, _ = self.dec.solve_batch(Ym, Sm, np.array([e["n"] for e in miss], dtype=np.int64),
                                                  np.array([e["d"] for e in miss], dtype=np.int64))
                ll, br = self.dec.token_losses(Ym, Q)
                for k, e in enumerate(miss):
                    e["q"], e["ll"], e["br"], e["q_sha"] = Q[k].copy(), float(ll[k]), float(br[k]), q_sha(Q[k])
                self.solves += len(miss)
            sup = [e for e in ents.values() if e["n"] > 0]
            out[f"L{r}"] = _ssum([e["ll"] for e in sup]) / N
            out[f"B{r}"] = _ssum([e["br"] for e in sup]) / N
            toks[r] = ents
        for r in (1, 2):
            F = self.fine[r].F
            X = np.vstack([np.bincount(lab[r], weights=self.tf[r][s_], minlength=F) for s_ in (0, 1)]).astype(np.int64)
            out[f"I{r}"] = (_ssum(self._phi(X)) + self.const) / N
        F1, F2 = self.fine[1].F, self.fine[2].F
        idx = (lab[1][:, None] * F2 + lab[2][None, :]).ravel()
        X = np.vstack([np.bincount(idx, weights=self.Tf[s_].ravel(), minlength=F1 * F2)
                       for s_ in (0, 1)]).astype(np.int64)
        out["I12"] = (_ssum(self._phi(X)) + self.const) / N
        out["Phi"] = out["I12"] + PHI_LOCAL_WEIGHT * (out["I1"] + out["I2"])
        out["T"] = out["L1"] + out["L2"] + TASK_BRIER_WEIGHT * (out["B1"] + out["B2"])
        return out, toks

    def state_hashes(self, toks):
        hs, hq = hashlib.sha256(), hashlib.sha256()
        for r in (1, 2):
            for c in sorted(toks[r]):
                e = toks[r][c]
                cb = np.asarray(c, dtype="<i8").reshape(1).tobytes()
                hs.update(cb + np.asarray(e["n"], dtype="<i8").reshape(1).tobytes()
                          + np.asarray(e["Y"], dtype="<f8").tobytes() + np.asarray(e["S"], dtype="<f8").tobytes())
                if e["n"] > 0:
                    hq.update(cb + np.asarray(e["q"], dtype="<f8").tobytes())
        return hs.hexdigest(), hq.hexdigest()


def replay_unit(record, fine_dict, T, tr, Y_fit, S_fit, refs=None, *, trace, files=None, per_state=True,
                starts=None):
    """Independent replay of one mapper unit from its persisted trace (prompt sec. 8; gate checks 6, 7 and 8).

    The inputs are those given to fit_unit (fine_dict; T over ALL rows; tr; Y_fit, S_fit aligned with tr) plus the unit's
    record and its trace.json. refs is optional: the refs passed to fit_unit, cross-checked against the record. files is
    optional: {"policy.json", "release.npz"}, which enables the deployed-release checks. starts is optional: the
    start/witness policy dicts, whose labels are checked against the persisted start labels.

    It rebuilds every start from its persisted canonical labels and applies the accepted moves IN ORDER; paired moves are
    applied atomically. At every accepted state it:
    - checks the recorded (incremental) terms and the vectorised deltas against a from-scratch rebuild (_Scratch), within
      REPLAY_TERM_ATOL / REPLAY_DELTA_ATOL;
    - checks the sufficient-statistic hashes and the cached solves bitwise against fresh folds and fresh solves;
    - checks the stage's enforced constraints, and for eligible constrained candidates both recipients' constraints;
    - verifies the sequential CLASS-ONLY partner, the frozen map, the termination receipts, the witness exclusion, the
      winner, the record's final_state_terms and, with files, the released policy and vectors.

    It never raises on a defect: every defect becomes a violation {code, start, stage, step, detail} with a code in
    REPLAY_CODES. It raises ValueError only on malformed problem inputs."""
    from qpc.kmeans import assign_fine
    dec = _dec()
    V = []
    per = []
    st_ = {"max_abs_diff": 0.0, "by_term": {k: 0.0 for k in _TERM_KEYS}, "max_delta": 0.0, "bitwise_states": 0,
           "n_states": 0, "single": 0, "pair": 0, "solves_checked": 0, "partner_infeasible_states": 0}

    def bad(code, start=None, stage=None, step=None, detail=""):
        assert code in REPLAY_CODES, code
        V.append({"code": code, "start": start, "stage": stage, "step": step, "detail": str(detail)[:500]})

    def report():
        codes = {v["code"] for v in V}
        checks = {
            "trace_hash": "TRACE_HASH_MISMATCH" not in codes, "complete": "TRACE_INCOMPLETE" not in codes,
            "references": "REFERENCE_MISMATCH" not in codes, "starts": "START_MISMATCH" not in codes,
            "incremental_terms": not codes & {"TERMS_MISMATCH", "DELTA_MISMATCH"},
            "stats_hashes": "STATS_HASH_MISMATCH" not in codes, "cached_solves": "CACHED_SOLVE_MISMATCH" not in codes,
            "strict_improvement": "NOT_STRICT_IMPROVEMENT" not in codes, "moves_valid": "MOVE_INVALID" not in codes,
            "enforced_constraints": "ENFORCED_CONSTRAINT_VIOLATED" not in codes,
            "paired_moves_feasible": "PAIR_INFEASIBLE" not in codes,
            "partner_not_required": not codes & {"PARTNER_NOT_CLASS_ONLY", "PARTNER_REQUIRED_FEASIBLE"},
            "frozen_map": "FROZEN_MAP_CHANGED" not in codes, "termination": "TERMINATION_INVALID" not in codes,
            "witnesses": "INFEASIBLE_WITNESS_ELIGIBLE" not in codes, "winner": "WINNER_MISMATCH" not in codes,
            "final_state": "FINAL_STATE_MISMATCH" not in codes, "final_feasible": "FINAL_INFEASIBLE" not in codes,
            "release": (None if files is None else "RELEASE_MISMATCH" not in codes)}
        for g, cs in REPLAY_GATE.items():
            checks[g] = not codes & set(cs)
        return {"ok": not V, "arm": record.get("arm"), "config": record.get("config"),
                "max_abs_diff": st_["max_abs_diff"], "max_abs_diff_by_term": st_["by_term"],
                "max_delta_abs_diff": st_["max_delta"], "n_states": st_["n_states"],
                "bitwise_equal_states": st_["bitwise_states"],
                "n_moves": {"single": st_["single"], "pair": st_["pair"]}, "n_solves_checked": st_["solves_checked"],
                "fresh_solves": SC.solves if SC is not None else 0,
                "partner_infeasible_states": st_["partner_infeasible_states"],
                "tolerances": {"REPLAY_TERM_ATOL": REPLAY_TERM_ATOL, "REPLAY_DELTA_ATOL": REPLAY_DELTA_ATOL,
                               "BUDGET_MARGIN": BUDGET_MARGIN, "CAP_MARGIN": CAP_MARGIN, "DEPLOYED_TOL": DEPLOYED_TOL},
                "checks": checks, "per_state": per if per_state else None, "violations": V}

    SC = None
    # ---------------------------------------------------------------- problem inputs (malformed -> ValueError)
    arm = record.get("arm") if isinstance(record, dict) else None
    if arm not in ARMS:
        raise ValueError("replay_unit: record has no registered arm")
    spec = SPEC[arm]
    kind = spec["kind"]
    cons_flag = bool(spec["constrained"])
    fine1, fine2 = PT.load_fine(fine_dict)
    fines = {1: fine1, 2: fine2}
    tr = np.asarray(tr, dtype=np.int64)
    P = {i: np.asarray(T[f"p{i}"], dtype=np.float64)[tr] for i in (1, 2)}
    dd = {i: np.asarray(T[f"d{i}"]).astype(np.int64)[tr] for i in (1, 2)}
    Y = _yfit(Y_fit)
    s = np.asarray(S_fit, dtype=np.int64)
    if s.shape != tr.shape or any(Y[i].shape != tr.shape for i in (1, 2)):
        raise ValueError("replay_unit: Y_fit / S_fit not aligned with tr")
    cell = {}
    for r in (1, 2):
        cell[r] = assign_fine(P[r], dd[r], fines[r])
        if not np.array_equal(np.bincount(cell[r], minlength=fines[r].F), np.asarray(fines[r].n)):
            raise ValueError(f"replay_unit: recipient {r} fine-cell counts on tr differ from the partition")
    Ycell = {r: dec.cell_label_counts(fines[r], P[r], dd[r], Y[r]).astype(np.float64) for r in (1, 2)}
    ncell = {r: np.bincount(cell[r], minlength=fines[r].F) for r in (1, 2)}
    SC = _Scratch(fines, cell, Ycell, s, dec)
    N = SC.N
    # ---------------------------------------------------------------- trace integrity and references
    if not isinstance(trace, dict) or trace.get("schema") != TRACE_SCHEMA or not isinstance(trace.get("starts"), list):
        bad("TRACE_INCOMPLETE", detail="trace missing or not of schema " + TRACE_SCHEMA)
        return report()
    try:
        if record.get("trace_sha256") != json_sha(RN._finite(trace)):
            bad("TRACE_HASH_MISMATCH", detail="record trace_sha256 differs from the trace")
    except (TypeError, ValueError) as e:
        bad("TRACE_INCOMPLETE", detail=f"trace not JSON-clean: {e}")
        return report()
    for k in ("arm", "config", "lam", "kind", "constrained"):
        want = {"kind": kind, "constrained": cons_flag}.get(k, record.get(k))
        if trace.get(k) != want:
            bad("TRACE_INCOMPLETE", detail=f"trace {k}={trace.get(k)!r} != {want!r}")
    fixture = bool(record.get("fixture_mode"))
    lam = record.get("lam")
    w = weights(arm, lam)
    ow = record.get("objective_weights", {})
    if [ow.get(k) for k in ("wL", "wB", "wI", "w12")] != list(w) or trace.get("objective_weights") != list(w):
        bad("REFERENCE_MISMATCH", detail=f"objective weights differ from weights({arm}, {lam})")
    if record.get("rules_sha256") != trace.get("rules_sha256") or record.get("rules_sha256") != rules_sha256():
        bad("REFERENCE_MISMATCH", detail="rules_sha256 differs between record, trace and the current rules")
    bud = record.get("budgets", {})
    caps = tuple(int(x) for x in bud.get("caps", ()))
    allow = {k: float(v) for k, v in bud.get("allowance", {}).items()}
    if not fixture and (caps != CAPS or allow != BUDGET):
        bad("REFERENCE_MISMATCH", detail=f"caps/budget {caps}/{allow} differ from the registered profile")
    if len(caps) != 2 or set(allow) != {"ll", "brier"}:
        bad("TRACE_INCOMPLETE", detail="record budgets lack caps/allowance")
        return report()
    if trace.get("caps") != list(caps) or trace.get("budget_allowance") != allow:
        bad("REFERENCE_MISMATCH", detail="trace caps/allowance differ from the record")
    if refs is not None:
        rc = tuple(int(x) for x in refs.get("caps", CAPS))
        rb = {k: float(v) for k, v in refs.get("budget", BUDGET).items()}
        if rc != caps or rb != allow:
            bad("REFERENCE_MISMATCH", detail="refs caps/budget differ from the record")
    capd = {1: caps[0], 2: caps[1]}
    LU, BU, limL, limB = {}, {}, {}, {}
    for r in (1, 2):
        pr = per_row(P[r], Y[r], fines[r].K)
        LU[r], BU[r] = float(pr["ll"].mean()), float(pr["br"].mean())
        for key, mine in (("L_U", LU[r]), ("B_U", BU[r])):
            v = bud.get(key, {}).get(str(r))
            if v is None or abs(float(v) - mine) > 1e-12:
                bad("REFERENCE_MISMATCH", detail=f"record {key}[{r}]={v} differs from the recomputed {mine}")
        limL[r], limB[r] = LU[r] + allow["ll"], BU[r] + allow["brier"]
    capI = {1: None, 2: None}
    if cons_flag:
        ic = (record.get("refs") or {}).get("I_ctask")
        if not isinstance(ic, dict):
            bad("REFERENCE_MISMATCH", detail="constrained record lacks refs.I_ctask")
            return report()
        capI = {1: float(ic["1"]), 2: float(ic["2"])}
        if refs is not None and refs.get("I_ctask") is not None:
            ri = refs["I_ctask"]
            if (float(ri.get("1", ri.get(1))), float(ri.get("2", ri.get(2)))) != (capI[1], capI[2]):
                bad("REFERENCE_MISMATCH", detail="refs I_ctask differ from the record (bitwise)")
    E = int((record.get("work") or {}).get("eval_ceiling", -1))
    if E != EVAL_CEILING or trace.get("eval_ceiling") != E:
        bad("REFERENCE_MISMATCH", detail=f"eval ceiling {E} / trace {trace.get('eval_ceiling')} != {EVAL_CEILING}")
    tstarts = trace["starts"]
    names = [z.get("name") for z in tstarts]
    if names != list(record.get("starts_given", [])) or len(tstarts) != len(record.get("starts", [])):
        bad("START_MISMATCH", detail=f"trace starts {names} differ from record starts_given")
        return report()
    reg = registered_starts(arm, lam)
    if names != [k for k in reg if k in names] or (not fixture and names != reg) or \
            any(k not in names for k in required_starts(arm, lam)):
        bad("START_MISMATCH", detail=f"starts {names} are not the registered set/order {reg}")
    nst = len(tstarts)
    share = E // nst if kind == "joint" else E // (2 * nst)
    wsum = 1.0 + sum(abs(x) for x in w)

    def feas(t, r, margin=True):
        mL = BUDGET_MARGIN if margin else DEPLOYED_TOL
        mI = CAP_MARGIN if margin else DEPLOYED_TOL
        if t[f"L{r}"] > limL[r] - mL or t[f"B{r}"] > limB[r] - mL:
            return False
        if cons_flag and capI[r] is not None and t[f"I{r}"] > capI[r] - mI:
            return False
        return True

    def cmp_terms(rec_terms, t, where):
        """Recorded terms vs from-scratch; returns max abs diff over the shared keys."""
        mx, bit = 0.0, True
        for k in _TERM_KEYS:
            if k in rec_terms and rec_terms[k] is not None:
                d = abs(float(rec_terms[k]) - t[k])
                bit &= float(rec_terms[k]) == t[k]
                st_["by_term"][k] = max(st_["by_term"][k], d)
                mx = max(mx, d)
        st_["max_abs_diff"] = max(st_["max_abs_diff"], mx)
        if mx > REPLAY_TERM_ATOL:
            bad("TERMS_MISMATCH", *where, detail=f"max |recorded - from-scratch| = {mx:.3e}")
        return mx, bit

    def check_labels(lab_json, where, code="START_MISMATCH"):
        lab = {}
        for r in (1, 2):
            x = np.asarray(lab_json[str(r)], dtype=np.int64)
            if x.shape != (fines[r].F,) or not np.array_equal(canon_labels(x), x):
                raise _Defect(code, f"recipient {r} labels are not canonical labels over all F cells")
            cls = np.asarray(fines[r].cell_class)
            if np.any(cls[x] != cls):
                raise _Defect(code, f"recipient {r}: a token mixes predicted classes")
            for c in range(fines[r].K):
                nt = np.unique(x[cls == c]).size
                if nt > capd[r]:
                    raise _Defect(code, f"recipient {r} class {c}: {nt} tokens > cap {capd[r]}")
            lab[r] = x
        return lab

    def apply_part(lab, part):
        r, f, fc, tc = int(part["r"]), int(part["f"]), int(part["from_canon"]), int(part["to_canon"])
        x = lab[r]
        F = x.size
        cls = np.asarray(fines[r].cell_class)
        if not (0 <= f < F and 0 <= tc < F) or x[f] != fc or x[tc] != tc or tc == fc or cls[tc] != cls[f]:
            raise _Defect("MOVE_INVALID", f"recipient {r}: move of cell {f} from {fc} to {tc} is not a move of a "
                                          f"whole fine cell to an existing same-class token")
        if ncell[r][f] == 0:
            raise _Defect("MOVE_INVALID", f"recipient {r}: cell {f} has no fitting rows (no proposal)")
        y = x.copy()
        y[f] = tc
        out = dict(lab)
        out[r] = canon_labels(y)
        return out

    def hashes_after(part, toks_old, lab_new, toks_new, where):
        r, f, fc = int(part["r"]), int(part["f"]), int(part["from_canon"])
        rest = [g for g in toks_old[r][fc]["members"] if g != f]
        efrom = toks_new[r][int(lab_new[r][rest[0]])] if rest else None
        eto = toks_new[r][int(lab_new[r][f])]
        hs = part.get("stats_sha256") or [None, None]
        hq = part.get("q_sha256") or [None, None]
        for e, h, q, nm in ((efrom, hs[0], hq[0], "from"), (eto, hs[1], hq[1], "to")):
            want_s = None if e is None else e["stats_sha"]
            want_q = None if e is None else e["q_sha"]
            if h != want_s:
                bad("STATS_HASH_MISMATCH", *where, detail=f"recipient {r} {nm}-token statistics hash differs from the "
                                                          "fresh canonical fold (stale cache key)")
            if q != want_q:
                bad("CACHED_SOLVE_MISMATCH", *where, detail=f"recipient {r} {nm}-token cached q differs bitwise from "
                                                            "a fresh solve of its statistics")
            if e is not None and e["n"] > 0:
                st_["solves_checked"] += 1

    def add_state(where, kind_, enforced, t, mx, bit, both_ok):
        st_["n_states"] += 1
        st_["bitwise_states"] += int(bit)
        if per_state:
            per.append({"start": where[0], "stage": where[1], "step": where[2], "kind": kind_,
                        "enforced": list(enforced), "feasible_enforced": all(feas(t, r) for r in enforced),
                        "feasible_both": both_ok, "max_abs_diff": mx})

    def replay_stage(ts, tg, lab, recips, enforced, name, frozen=None, partner=None):
        """Replays one stage from lab; returns (end labels, end terms, end toks, status, states_both_ok)."""
        sname = tg.get("stage")
        rec_enf = [int(x) for x in tg.get("enforced", [])]
        if rec_enf != list(enforced):
            code = ("PARTNER_REQUIRED_FEASIBLE" if partner is not None and partner in rec_enf
                    else "ENFORCED_CONSTRAINT_VIOLATED")
            bad(code, name, sname, -1, f"trace enforced {rec_enf} != registered {list(enforced)}")
        if [int(x) for x in tg.get("recipients", [])] != list(recips):
            bad("TRACE_INCOMPLETE", name, sname, -1, "stage recipients differ")
        # stage start labels (derived) vs the persisted stage start labels
        tl = tg.get("start_labels") or {}
        for r in (1, 2):
            if not np.array_equal(np.asarray(tl.get(str(r), []), dtype=np.int64), lab[r]):
                code = ("FROZEN_MAP_CHANGED" if frozen == r else "PARTNER_NOT_CLASS_ONLY" if partner == r
                        else "START_MISMATCH")
                bad(code, name, sname, -1, f"recipient {r} stage start labels differ from the derived start")
        t, toks = SC.evaluate(lab)
        where = (name, sname, -1)
        mx, bit = cmp_terms(tg.get("start_terms") or {}, t, where)
        hs, hq = SC.state_hashes(toks)
        if tg.get("start_stats_sha256") != hs:
            bad("STATS_HASH_MISMATCH", *where, detail="stage start statistics hash differs from the fresh fold")
        if tg.get("start_q_sha256") != hq:
            bad("CACHED_SOLVE_MISMATCH", *where, detail="stage start q hash differs from fresh solves")
        both = all(feas(t, r) for r in (1, 2))
        add_state(where, "start", enforced, t, mx, bit, both)
        own_ok = all(feas(t, r) for r in enforced)
        status = tg.get("status")
        if cons_flag and not own_ok:
            if status != "INFEASIBLE_START":
                bad("ENFORCED_CONSTRAINT_VIOLATED", *where, detail="a stage whose start violates its enforced "
                                                                   "constraints was refined")
            return lab, t, toks, "INFEASIBLE_START", both
        if status == "INFEASIBLE_START":
            if partner is not None and not feas(t, partner):
                bad("PARTNER_REQUIRED_FEASIBLE", *where, detail="stage 1 start satisfies its own constraints but was "
                                                                "refused: the CLASS-ONLY partner was required feasible")
            else:
                bad("TERMINATION_INVALID", *where, detail="feasible start recorded INFEASIBLE_START")
            return lab, t, toks, "INFEASIBLE_START", both
        if status != "REFINED":
            bad("TRACE_INCOMPLETE", *where, detail=f"stage status {status!r}")
        obj = objective(t, w)
        obj_start = obj
        states_both = both
        moves = tg.get("moves") or []
        last_sweep = 0
        for k, mv in enumerate(moves):
            where = (name, sname, k)
            parts = mv.get("parts") or []
            typ = mv.get("type")
            if mv.get("step") != k:
                bad("TERMINATION_INVALID", *where, detail="move steps not consecutive")
            sw = int(mv.get("sweep", 0))
            if sw < last_sweep or sw < 1:
                bad("TERMINATION_INVALID", *where, detail="sweep numbers not nondecreasing from 1")
            last_sweep = max(last_sweep, sw)
            if typ == "single" and len(parts) == 1:
                st_["single"] += 1
            elif typ == "pair" and len(parts) == 2 and [int(p["r"]) for p in parts] == [1, 2]:
                st_["pair"] += 1
                if not (arm == "K-JOINT-PAIR"):
                    raise _Defect("MOVE_INVALID", f"paired move in arm {arm}")
            else:
                raise _Defect("MOVE_INVALID", f"malformed move entry {typ!r} with {len(parts)} parts")
            for p in parts:
                r = int(p["r"])
                if r not in recips:
                    code = ("FROZEN_MAP_CHANGED" if frozen == r else "PARTNER_NOT_CLASS_ONLY" if partner == r
                            else "MOVE_INVALID")
                    raise _Defect(code, f"move of recipient {r} in a stage of {list(recips)}")
            if typ == "pair":                  # one-sided deltas against the pre-pair state (not accepted states)
                for p in parts:
                    r = int(p["r"])
                    t1, _ = SC.evaluate(apply_part(lab, p))
                    for key, val in (("dL", t1[f"L{r}"] - t[f"L{r}"]), ("dB", t1[f"B{r}"] - t[f"B{r}"]),
                                     ("dI", t1[f"I{r}"] - t[f"I{r}"]), ("dI12", t1["I12"] - t["I12"])):
                        if p.get(key) is not None:
                            d = abs(float(p[key]) - val)
                            st_["max_delta"] = max(st_["max_delta"], d)
                            if d > REPLAY_DELTA_ATOL:
                                bad("DELTA_MISMATCH", *where, detail=f"pair part r{r} one-sided {key} off by {d:.3e}")
                    dphi = (t1["I12"] - t["I12"]) + PHI_LOCAL_WEIGHT * (t1[f"I{r}"] - t[f"I{r}"])
                    dtask = (t1[f"L{r}"] - t[f"L{r}"]) + TASK_BRIER_WEIGHT * (t1[f"B{r}"] - t[f"B{r}"])
                    for key, val in (("dPhi_one_sided", dphi), ("dTask", dtask)):
                        if p.get(key) is not None and abs(float(p[key]) - val) > 2 * REPLAY_DELTA_ATOL:
                            bad("DELTA_MISMATCH", *where, detail=f"pair part r{r} {key} off by "
                                                                 f"{abs(float(p[key]) - val):.3e}")
            new = lab
            for p in parts:
                new = apply_part(new, p)
            t2, toks2 = SC.evaluate(new)
            obj2 = objective(t2, w)
            if typ == "single":
                p = parts[0]
                r = int(p["r"])
                for key, val in (("dL", t2[f"L{r}"] - t[f"L{r}"]), ("dB", t2[f"B{r}"] - t[f"B{r}"]),
                                 ("dI", t2[f"I{r}"] - t[f"I{r}"]), ("dI12", t2["I12"] - t["I12"])):
                    if p.get(key) is not None:
                        d = abs(float(p[key]) - val)
                        st_["max_delta"] = max(st_["max_delta"], d)
                        if d > REPLAY_DELTA_ATOL:
                            bad("DELTA_MISMATCH", *where, detail=f"vectorised {key} off by {d:.3e}")
                dlt = p.get("delta")
                if dlt is None or abs(float(dlt) - (obj2 - obj)) > REPLAY_DELTA_ATOL * wsum:
                    bad("DELTA_MISMATCH", *where, detail=f"vectorised objective delta {dlt} vs {obj2 - obj}")
                if dlt is None or not float(dlt) < -TOL or not obj2 < obj:
                    bad("NOT_STRICT_IMPROVEMENT", *where, detail=f"objective {obj} -> {obj2} (delta {dlt})")
            elif not obj2 < obj - TOL:
                bad("NOT_STRICT_IMPROVEMENT", *where, detail=f"paired objective {obj} -> {obj2}")
            for p in parts:
                hashes_after(p, toks, new, toks2, where)
            mx, bit = cmp_terms(mv.get("terms_after") or {}, t2, where)
            oa = mv.get("objective_after")
            if oa is None or abs(float(oa) - obj2) > REPLAY_TERM_ATOL * wsum:
                bad("TERMS_MISMATCH", *where, detail=f"objective_after {oa} vs from-scratch {obj2}")
            ok_enf = all(feas(t2, r) for r in enforced)
            if not ok_enf:
                bad("PAIR_INFEASIBLE" if typ == "pair" else "ENFORCED_CONSTRAINT_VIOLATED", *where,
                    detail="accepted state violates an enforced budget or local cap (from-scratch values)")
            both = all(feas(t2, r) for r in (1, 2))
            states_both &= both
            if partner is not None and not feas(t2, partner):
                st_["partner_infeasible_states"] += 1        # informational: never required (check 7)
            add_state(where, typ, enforced, t2, mx, bit, both)
            lab, t, toks, obj = new, t2, toks2, obj2
        # termination receipt
        rc = tg.get("termination") or {}
        where = (name, sname, None)
        msgs = []
        if rc.get("stop") not in ("no_change_sweep", "sweep_cap", "eval_ceiling"):
            msgs.append(f"stop {rc.get('stop')!r}")
        sweeps = rc.get("sweeps")
        if not isinstance(sweeps, int) or not 1 <= sweeps <= SWEEPS or last_sweep > (sweeps or 0):
            msgs.append(f"sweeps {sweeps} (last move sweep {last_sweep})")
        if rc.get("accepted") != len(moves) or rc.get("pairs_accepted") != sum(m.get("type") == "pair" for m in moves):
            msgs.append("accepted counts differ from the moves")
        if rc.get("ceiling_share") != share:
            msgs.append(f"ceiling_share {rc.get('ceiling_share')} != registered {share}")
        ev = rc.get("evals", -1)
        over = max(capd[r] for r in recips) - 1
        if not (0 <= ev <= share + over):
            msgs.append(f"evals {ev} exceed the share {share} + per-cell granularity {over}")
        if rc.get("stop") == "eval_ceiling" and ev < share:
            msgs.append("eval_ceiling stop below the share")
        if rc.get("stop") == "no_change_sweep" and any(m.get("sweep") == sweeps for m in moves):
            msgs.append("no_change_sweep but the last sweep accepted a move")
        if rc.get("stop") == "sweep_cap" and (sweeps != SWEEPS or not any(m.get("sweep") == SWEEPS for m in moves)):
            msgs.append("sweep_cap without a changing final sweep")
        for key, val in (("objective_start", obj_start), ("objective_end", obj)):
            if rc.get(key) is None or abs(float(rc[key]) - val) > REPLAY_TERM_ATOL * wsum:
                msgs.append(f"{key} {rc.get(key)} vs from-scratch {val}")
        if msgs:
            bad("TERMINATION_INVALID", *where, detail="; ".join(msgs))
        return lab, t, toks, "REFINED", states_both

    def rec_log(pos, sname):
        z = record["starts"][pos]
        if kind == "local":
            g = [x for x in z.get("stages", []) if x.get("recipient") == int(sname[1])]
            return g[0] if g else None
        if kind == "seq":
            return z.get("stage1" if sname == "seq1" else "stage2")
        return z

    def cross_record(pos, tg):
        """The trace stage vs the record's compact move log and receipt (the trace must be the record's search)."""
        z = rec_log(pos, tg.get("stage"))
        if z is None or z.get("status") != "REFINED":
            return
        singles = [[m["sweep"], m["parts"][0]["r"], m["parts"][0]["f"], m["parts"][0]["from_slot"],
                    m["parts"][0]["to_slot"], m["parts"][0]["delta"]] for m in tg.get("moves", [])
                   if m.get("type") == "single"]
        logm = [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in z.get("moves", [])]
        rc = tg.get("termination") or {}
        if singles != logm or any(z.get(k) != rc.get(k) for k in ("stop", "sweeps", "evals")):
            bad("TRACE_INCOMPLETE", names[pos], tg.get("stage"), None,
                "trace moves/receipt differ from the record's move log and stop/sweeps/evals")

    # ---------------------------------------------------------------- per start
    cands = []                     # (position, name, kind, labels, terms, eligible) in the mapper's order
    for pos, ts in enumerate(tstarts):
        name = ts.get("name")
        try:
            lab0 = check_labels(ts["labels"], None)
            if starts is not None and name in starts:
                pp = RL.PolicyPair.from_dict(starts[name])
                for r, pol in ((1, pp.p1), (2, pp.p2)):
                    if not np.array_equal(QC.labels_from_policy(pol), lab0[r]):
                        bad("START_MISMATCH", name, None, None, f"recipient {r} labels differ from the start map")
            t0, toks0 = SC.evaluate(lab0)
            if cons_flag and name == RN.ctask_id() and (t0["I1"], t0["I2"]) != (capI[1], capI[2]):
                bad("REFERENCE_MISMATCH", name, None, None, "I_ctask differs from the exact MI of the C-TASK start")
            stages = ts.get("stages") or []
            if kind == "local":
                lab, ok_all, both_all = lab0, True, True
                if [g.get("stage") for g in stages] != ["r1", "r2"]:
                    raise _Defect("TRACE_INCOMPLETE", "local start needs stages r1, r2")
                for r, tg in zip((1, 2), stages):
                    lab, t, toks, stt, sb = replay_stage(ts, tg, lab, (r,), (r,) if cons_flag else (), name)
                    ok_all &= stt == "REFINED"
                    both_all &= sb
                    if stt == "REFINED":
                        cross_record(pos, tg)
                elig = bool(ok_all and (not cons_flag or all(feas(t, r) for r in (1, 2))))
                if cons_flag and elig and not both_all:
                    bad("ENFORCED_CONSTRAINT_VIOLATED", name, None, None,
                        "an eligible candidate passed through a state violating the other recipient's constraints")
                cands.append((pos, name, "refined", lab, t, elig))
                fin = lab
            elif kind == "seq":
                a, b = spec["order"]
                if not stages or stages[0].get("stage") != "seq1":
                    raise _Defect("TRACE_INCOMPLETE", "seq start needs stage seq1")
                g1 = stages[0]
                cl = canon_labels(QC.class_labels(fines[b]))
                pt = g1.get("partner") or {}
                if (pt.get("view") != "CLASS-ONLY" or pt.get("recipient") != b or pt.get("constraints_enforced")
                        is not False or not np.array_equal(np.asarray(pt.get("labels", []), dtype=np.int64), cl)):
                    bad("PARTNER_NOT_CLASS_ONLY", name, "seq1", -1,
                        "stage-1 partner is not the recorded CLASS-ONLY view with constraints_enforced false")
                lab1 = {a: lab0[a], b: cl}
                lab, t, toks, stt, _ = replay_stage(ts, g1, lab1, (a,), (a,) if cons_flag else (), name,
                                                     partner=b)
                fin = None
                elig = False
                if stt == "REFINED":
                    cross_record(pos, g1)
                    if len(stages) != 2 or stages[1].get("stage") != "seq2":
                        raise _Defect("TRACE_INCOMPLETE", "refined seq1 without a seq2 stage")
                    g2 = stages[1]
                    fz = g2.get("frozen") or {}
                    if fz.get("recipient") != a or not np.array_equal(np.asarray(fz.get("labels", []),
                                                                                 dtype=np.int64), lab[a]):
                        bad("FROZEN_MAP_CHANGED", name, "seq2", -1, "frozen map differs from the stage-1 end map")
                    lab2 = {a: lab[a], b: lab0[b]}
                    end_a = lab[a]
                    lab, t, toks, stt2, sb = replay_stage(ts, g2, lab2, (b,), (b,) if cons_flag else (), name,
                                                          frozen=a)
                    if not np.array_equal(lab[a], end_a):
                        bad("FROZEN_MAP_CHANGED", name, "seq2", None, "first recipient revised in stage 2")
                    if stt2 == "REFINED":
                        cross_record(pos, g2)
                        elig = bool(not cons_flag or all(feas(t, r) for r in (1, 2)))
                        if cons_flag and elig and not sb:
                            bad("ENFORCED_CONSTRAINT_VIOLATED", name, "seq2", None,
                                "an eligible candidate passed through a stage-2 state violating a constraint")
                        cands.append((pos, name, "refined", lab, t, elig))
                        fin = lab
                elif len(stages) != 1:
                    raise _Defect("TRACE_INCOMPLETE", "seq2 present after an INFEASIBLE_START seq1")
            else:
                jw = ts.get("joint_witness") or {}
                cmp_terms(jw.get("terms") or {}, t0, (name, "witness", -1))
                wfeas = (not cons_flag) or all(feas(t0, r) for r in (1, 2))
                if bool(jw.get("feasible_all")) != bool(wfeas) and cons_flag:
                    bad("INFEASIBLE_WITNESS_ELIGIBLE", name, "witness", -1, "witness feasibility recorded wrongly")
                fin = None
                elig = False
                if not wfeas:
                    if jw.get("status") != "EXCLUDED_INFEASIBLE_WITNESS" or stages or ts.get("eligible") or \
                            jw.get("eligible_unchanged"):
                        bad("INFEASIBLE_WITNESS_ELIGIBLE", name, "witness", -1,
                            "an infeasible witness was refined or made eligible")
                else:
                    if jw.get("status") != "REFINED" or len(stages) != 1 or stages[0].get("stage") != "joint":
                        raise _Defect("TRACE_INCOMPLETE", "feasible witness without one refined joint stage")
                    lab, t, toks, stt, sb = replay_stage(ts, stages[0], lab0, (1, 2), (1, 2) if cons_flag else (),
                                                         name)
                    cross_record(pos, stages[0])
                    elig = bool(not cons_flag or all(feas(t, r) for r in (1, 2)))
                    cands.append((pos, name, "refined", lab, t, elig))
                    cands.append((pos, name, "unchanged", lab0, t0, True))
                    fin = lab
            if bool(ts.get("eligible")) != bool(elig):
                bad("WINNER_MISMATCH", name, None, None, f"trace eligible {ts.get('eligible')} vs replay {elig}")
            if fin is not None:
                fl = ts.get("final_labels") or {}
                if any(not np.array_equal(np.asarray(fl.get(str(r), []), dtype=np.int64), fin[r]) for r in (1, 2)):
                    bad("FINAL_STATE_MISMATCH", name, None, None, "final_labels differ from the replayed end state")
            elif ts.get("final_labels") is not None:
                bad("TRACE_INCOMPLETE", name, None, None, "final_labels without a refined candidate")
        except _Defect as e:
            bad(e.code, name, None, None, e.detail)
        except (KeyError, TypeError, IndexError, ValueError) as e:
            bad("TRACE_INCOMPLETE", name, None, None, f"{type(e).__name__}: {e}")
    # ---------------------------------------------------------------- winner, final state, status
    best = None
    for k, (pos, name, kd, lab, t, elig) in enumerate(cands):
        if cons_flag and not (elig and all(feas(t, r) for r in (1, 2))):
            continue
        v = objective(t, w)
        if best is None or v < best[0]:
            best = (v, k)
    status = "FEASIBLE" if best is not None or not cons_flag else "INFEASIBLE"
    if best is None:
        if not cands:
            win = (0, names[0], "unchanged_descriptive", None)
        else:
            pos, name, kd, lab, t, _ = cands[0]
            win = (pos, name, kd + "_descriptive", lab)
    else:
        pos, name, kd, lab, t, _ = cands[best[1]]
        win = (pos, name, kd, lab)
    tw = trace.get("winner") or {}
    rw = record.get("winner") or {}
    if (tw.get("start"), tw.get("kind"), tw.get("position")) != (win[1], win[2], win[0]) or \
            (rw.get("start"), rw.get("kind")) != (win[1], win[2]) or record.get("status") != status or \
            tw.get("status") != status:
        bad("WINNER_MISMATCH", detail=f"replayed winner {win[1]}:{win[2]} ({status}) vs trace "
                                      f"{tw.get('start')}:{tw.get('kind')} / record {rw.get('start')}:{rw.get('kind')} "
                                      f"({record.get('status')})")
    try:
        wl = check_labels(tw.get("labels") or {}, None, code="FINAL_STATE_MISMATCH")
    except (_Defect, KeyError, TypeError, ValueError) as e:
        bad("FINAL_STATE_MISMATCH", detail=f"trace winner labels unusable: {e}")
        return report()
    if win[3] is not None and any(not np.array_equal(wl[r], win[3][r]) for r in (1, 2)):
        bad("FINAL_STATE_MISMATCH", detail="trace winner labels differ from the replayed winner's state")
    if win[3] is None:
        try:
            l0 = check_labels(tstarts[0]["labels"], None)
            if any(not np.array_equal(wl[r], l0[r]) for r in (1, 2)):
                bad("FINAL_STATE_MISMATCH", detail="descriptive winner differs from the first start")
        except (_Defect, KeyError) as e:
            bad("FINAL_STATE_MISMATCH", detail=str(e))
    tf_, toksf = SC.evaluate(wl)
    fst = record.get("final_state_terms") or {}
    cmp = {k: (fst.get(k), tf_[k]) for k in _TERM_KEYS}
    cmp["objective"] = (fst.get("objective"), objective(tf_, w))
    off = {k: (None if a is None else abs(float(a) - b)) for k, (a, b) in cmp.items()}
    if any(v is None or v > REPLAY_TERM_ATOL * (wsum if k == "objective" else 1.0) for k, v in off.items()):
        bad("FINAL_STATE_MISMATCH", detail=f"record final_state_terms vs from-scratch winner: {off}")
    if cons_flag and status == "FEASIBLE":
        if not all(feas(tf_, r) for r in (1, 2)) or not all(feas(tf_, r, margin=False) for r in (1, 2)) or \
                not (record.get("deployed") or {}).get("feasible"):
            bad("FINAL_INFEASIBLE", detail="FEASIBLE winner violates a fitting budget or local cap")
    # ---------------------------------------------------------------- deployed release (optional)
    if files is not None:
        try:
            pp = RL.PolicyPair.from_dict(files["policy.json"])
            if pp.fingerprint() != record.get("pair_fingerprint"):
                bad("RELEASE_MISMATCH", detail="policy fingerprint differs from the record")
            rel = files["release.npz"]
            for r, pol in ((1, pp.p1), (2, pp.p2)):
                if not np.array_equal(QC.labels_from_policy(pol), wl[r]):
                    bad("RELEASE_MISMATCH", detail=f"recipient {r} released policy labels differ from the winner")
                    continue
                tok = np.asarray(rel[f"tok{r}"])[tr]
                ct_ = np.asarray(pol.cell_token)
                if not np.array_equal(tok, ct_[cell[r]]):
                    bad("RELEASE_MISMATCH", detail=f"recipient {r} released tokens differ from the policy routing")
                if not np.array_equal(np.asarray(rel[f"hard{r}"])[tr], dd[r]):
                    bad("RELEASE_MISMATCH", detail=f"recipient {r} released decisions differ from the teacher's")
                q = np.asarray(rel[f"q{r}"], dtype=np.float64)[tr]
                Qrow = np.zeros_like(q)
                lab_row = wl[r][cell[r]]
                for c, e in toksf[r].items():
                    if e["n"] > 0:
                        Qrow[lab_row == c] = e["q"]
                if not np.array_equal(q.view(np.uint64), Qrow.view(np.uint64)):
                    bad("RELEASE_MISMATCH", detail=f"recipient {r} released q differs bitwise from fresh solves")
                pr = per_row(q, Y[r], fines[r].K)
                dl, db = float(pr["ll"].mean()), float(pr["br"].mean())
                if abs(dl - tf_[f"L{r}"]) > PARITY_ATOL or abs(db - tf_[f"B{r}"]) > PARITY_ATOL:
                    bad("RELEASE_MISMATCH", detail=f"recipient {r} row-level L/B differ from the state values")
                if cons_flag and status == "FEASIBLE" and (dl > limL[r] + DEPLOYED_TOL or db > limB[r] + DEPLOYED_TOL):
                    bad("FINAL_INFEASIBLE", detail=f"recipient {r} deployed row-level budgets violated")
                if abs(mi_plugin(s, tok) - tf_[f"I{r}"]) > MI_PARITY_ATOL:
                    bad("RELEASE_MISMATCH", detail=f"recipient {r} mi_plugin differs from the table MI")
        except (KeyError, TypeError, ValueError) as e:
            bad("RELEASE_MISMATCH", detail=f"{type(e).__name__}: {e}")
    return report()


def replay_trace(problem_inputs, record, trace, files=None, **kw):
    """Alias: problem_inputs = {"fine_dict", "T", "tr", "Y_fit", "S_fit"[, "refs"]} (fit_unit's inputs)."""
    z = problem_inputs
    return replay_unit(record, z["fine_dict"], z["T"], z["tr"], z["Y_fit"], z["S_fit"], z.get("refs"), trace=trace,
                       files=files, **kw)


def rules():
    """Every frozen rule (SEARCH_RULES.json body; the lead locks it)."""
    return {
        "schema": SCHEMA, "owner": "role C (mapper)", "module": "lra/mapper.py",
        "arms": list(ARMS),
        "config_ids": {a: (config_id(a, 0.01) if a.startswith("W-") else config_id(a)) for a in ARMS},
        "decoder": {"name": "D1 (lra.decoder.solve_batch)", "kappa": KAPPA, "eps": EPS, "loss_clip": LOSS_CLIP,
                    "token_statistics": "n_t, label counts y_t (exact), teacher sums s_t accumulated from zeros over "
                                        "member fine cells in increasing fine index (= qpc token_tables = "
                                        "lra.decoder.token_stats, bitwise); fallback/empty tokens (n_t = 0) carry no "
                                        "loss and the pinned D0 vector at release"},
        "fitting_quantities": {
            "rows": "OSF_DEFENSE_FIT (tr), deployed fine-cell routing of the frozen teacher",
            "L_i": "(1/N) sum_t sum_k y_t[k] * -log clip(q_t[k], 1e-12, 1) (lra.decoder.token_losses)",
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
                      "accept": f"best feasible strict Phi improvement (< -{TOL}); ties first in (i, j) order",
                      "ceiling": "the pool screening (every (cell, target) exact change) and every atomic pair "
                                 "evaluation count toward the start's share and both stop at it (checked before each "
                                 "screened cell and before each pair); a step that reaches it stops the stage at "
                                 "'eval_ceiling' (F R-4; before, the screening ran unchecked)"},
        "starts": {a: (registered_starts(a, 0.01) if a.startswith("W-") else registered_starts(a)) for a in ARMS},
        "starts_notes": {
            "lam_placeholder": "W-JOINT witnesses are at the SAME lambda as the unit (l0.01 shown)",
            "maps": "each start/witness is a policy.json map; only its cell->token map is used; D1 recomputed",
            "not_starts": "U|DIRECT-TASK|i8o64 is a Stage A identity map on its own assignment cells, not a "
                          "cell->token map of the fine partitions, so it cannot be a start or witness",
            "class_witness": "U|CLASS|i1o1 (one token per predicted class) is a joint witness after FINE-TASK "
                             "(prompt sec. 8 'feasible source mappings'); like every witness it is eligible only if "
                             "it satisfies all constraints unchanged (K- arms), else EXCLUDED_INFEASIBLE_WITNESS",
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
                       "granularity": "the share is checked before each fine cell (and, in the paired step, before "
                                      "each screened cell and each pair), so a stage's evaluations are <= share + "
                                      "(cap - 1) of its recipients",
                       "recorded": "actual evaluations, ceiling share and stop reason per stage/start (record and "
                                   "trace termination receipts); solves, memo hits/misses and CPU per start and unit"},
        "weighted_controls": {
            "rule": "the D-bank W- arms use the same D1 decoder, the same start/witness structure (W-LOCAL from C-TASK; "
                    "W-SEQ from C-TASK + the six source SEQ maps; W-JOINT witnesses C-TASK, W-LOCAL/W-SEQ-12/W-SEQ-21 "
                    "at the same lambda, FINE-TASK and the 24 source maps), the same caps, the same single-move "
                    "neighbourhood construction (4 nearest + 4 best, ties, ordering, sweeps) and the same ceiling "
                    "as the corresponding K- arm; the '4 best' component and acceptance rank by the arm's own exact "
                    "objective (T + lam Phi, or T + lam (I1 + I2)/2 for W-LOCAL); no hard fitting budget",
            "corresponds": {"W-LOCAL": "K-LOCAL", "W-SEQ-12": "K-SEQ-12", "W-SEQ-21": "K-SEQ-21",
                            "W-JOINT": "K-JOINT-SINGLE"}},
        "trace": {
            "file": "trace.json (one per unit, every arm; JSON-clean; NOT inside record.json); record.trace_sha256 = "
                    "sha256 of its canonical JSON, record.trace_schema, record.trace_counts",
            "schema": TRACE_SCHEMA,
            "per_start": "name, registered position, canonical start labels per recipient (label = lowest member "
                         "fine index of the cell's token, over all F cells), stages, final_labels (refined candidates "
                         "only), eligible; joint: joint_witness (unchanged terms, feasible_all, status)",
            "per_stage": "stage (r1|r2|seq1|seq2|joint), recipients, enforced recipients, start_labels, start_terms, "
                         "start state statistics/q hashes, status (REFINED|INFEASIBLE_START), the ordered accepted "
                         "moves, the termination receipt (stop, sweeps, evals, ceiling_share, accepted, "
                         "pairs_accepted, pair_steps, objective_start, objective_end); seq1: partner {recipient, "
                         "view CLASS-ONLY, labels, constraints_enforced false, its constraint values}; seq2: frozen "
                         "{recipient, labels}",
            "per_move": MOVE_FIELDS,
            "hashes": HASH_RULE,
            "winner": "start, position, kind, status, canonical labels (= the released policy), objective"},
        "replay": {
            "function": "replay_unit(record, fine_dict, T, tr, Y_fit, S_fit, refs=None, *, trace, files=None, "
                        "per_state=True, starts=None) -> {ok, max_abs_diff, max_abs_diff_by_term, "
                        "max_delta_abs_diff, n_states, bitwise_equal_states, n_moves, n_solves_checked, checks, "
                        "per_state, violations}",
            "from_scratch": "per state, every token's statistics are folded from zeros over its member cells in "
                            "increasing fine index, solved FRESH by lra.decoder.solve_batch (memoised only by the "
                            "exact member set), losses by token_losses; MI from the fine-cell SEX table of the rows",
            "checks": ["persisted start labels canonical, class-pure, within caps; equal to the start maps if given",
                       "record/trace references: U losses, caps, allowance, I_ctask (bitwise = exact MI of the "
                       "C-TASK start), eval ceiling, rules hash, trace hash, objective weights",
                       "moves applied IN ORDER (paired moves atomically): whole fine cell with fitting rows to an "
                       "existing same-class token",
                       f"recorded state terms vs from-scratch <= {REPLAY_TERM_ATOL}; vectorised deltas (and pair "
                       f"one-sided deltas) <= {REPLAY_DELTA_ATOL} (objective delta x (1 + sum |w|))",
                       "statistics hashes and cached q of both affected tokens BITWISE equal to fresh folds / "
                       "fresh solves; stage start state hashes likewise",
                       "strict improvement of every accepted move (single: delta < -TOL and exact decrease; "
                       "pair: < -TOL)",
                       "constrained arms: every accepted state meets the stage's enforced budgets and local caps "
                       "(search margins); a paired state meets both recipients'; eligible candidates meet both "
                       "recipients' constraints at every state outside seq stage 1",
                       "sequential: stage-1 partner = CLASS-ONLY labels, enforced = [first recipient] only, a start "
                       "that meets its own constraints is refined (else PARTNER_REQUIRED_FEASIBLE); stage 2 starts "
                       "from the frozen stage-1 map, which never moves; the final release meets both budgets (and "
                       "both caps on K- arms)",
                       "joint: an infeasible witness is EXCLUDED, never refined or eligible",
                       "termination receipts consistent with the moves, the registered share and the record",
                       "winner = the registered rule recomputed from scratch; record final_state_terms; with files: "
                       "released policy labels, tokens, decisions and q (bitwise fresh solves), row-level parity"],
            "codes": list(REPLAY_CODES), "gate": {k: list(v) for k, v in REPLAY_GATE.items()},
            "tolerances": {"REPLAY_TERM_ATOL": REPLAY_TERM_ATOL, "REPLAY_DELTA_ATOL": REPLAY_DELTA_ATOL,
                           "solves_and_statistics": "bitwise (sha256)"},
            "never_raises_on_a_defect": True},
        "cache": {"rule": "memo of decoder solves per (recipient, class, fine cell, slot); an entry is reused only if "
                          "its stored (n_t, y_t, s_t) are BITWISE equal to the freshly folded statistics (exact "
                          "sufficient-statistic key, -0.0 impossible for sums of nonnegative values); never reused "
                          "across different label counts or teacher sums",
                  "batch": "lra.decoder.solve_batch (rows independent and bitwise equal to solve_token)"},
        "receipts": ["the persisted trace (trace.json) of every unit: starts, accepted moves with per-move "
                     "incremental terms and sufficient-statistic hashes, termination receipts, sequential partner "
                     "and frozen maps, winner",
                     "every accepted move; rejected_by_budget counts (L, B, I); rejected_on_exact_recheck",
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
    fs = ("lra/mapper.py", "lra/decoder.py", "lra/run.py", "qpc/compress.py", "qpc/release.py", "qpc/kmeans.py",
          "qpc/partition.py", "dpc/compress.py", "dpc/release.py", "dpc/partition.py", "dpc/utility.py")
    return {f: hashlib.sha256((wt / f).read_bytes()).hexdigest() for f in fs}


def rules_sha256():
    return hashlib.sha256(json.dumps(rules(), sort_keys=True, allow_nan=False).encode()).hexdigest()


# ------------------------------------------------------------------ the unit
def fit_unit(arm, fine_dict, T, tr, Y_fit, S_fit, lam=None, starts=None, witnesses=None, refs=None, meta=None):
    """One new mapping-pair unit. See the module docstring. Returns (record, files)."""
    t0, c0 = time.perf_counter(), time.process_time()
    _TRACE_CPU["s"] = 0.0
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
    start_recs, cands, tstarts = runner(pb, arm, w, start_maps, spec["constrained"], E)
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
    # policy pair, decoders, release (lra.decoder helpers)
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
    # persisted trace (prompt sec. 8): coherent starts, accepted moves, sufficient-statistic hashes, termination
    # receipts, sequential partner/frozen maps and the winner; replayed by replay_unit (checks 6-8)
    ct0 = time.process_time()
    pos = {n_: k_ for k_, (n_, _) in enumerate(start_maps)}
    trace = RN._finite({
        "schema": TRACE_SCHEMA, "arm": arm, "config": cid, "lam": lam, "kind": kind, "fixture_mode": fixture,
        "sex_used_in_search": bool(spec["sex"]), "constrained": bool(spec["constrained"]),
        "objective_weights": [float(x) for x in w], "eval_ceiling": int(E), "caps": list(caps),
        "budget_allowance": budget, "rules_sha256": rec["rules_sha256"], "hash_rule": HASH_RULE,
        "move_fields": MOVE_FIELDS, "starts": tstarts,
        "winner": {"start": win_name, "position": pos[win_name], "kind": win_kind, "status": status,
                   "labels": {str(r): canon_labels(final_labels[r]).tolist() for r in (1, 2)},
                   "objective": objective(ft, w)}})
    rec["trace_schema"] = TRACE_SCHEMA
    rec["trace_sha256"] = json_sha(trace)
    rec["trace_counts"] = {"starts": len(tstarts),
                           "moves": sum(len(g["moves"]) for z in tstarts for g in z["stages"]),
                           "pairs": sum(1 for z in tstarts for g in z["stages"] for m in g["moves"]
                                        if m["type"] == "pair")}
    rec["trace_cpu_s"] = _TRACE_CPU["s"] + (time.process_time() - ct0)
    rec["wall_s_mapper"] = time.perf_counter() - t0
    rec["cpu_s_mapper"] = time.process_time() - c0
    files = {"policy.json": pair.to_dict(), "decoder.json": body, "release.npz": release, "trace.json": trace}
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
    pp, _ = QC.fit_policy_pair("CLASS", f1, f2, P1, d1, P2, d2, S_fit, 1, 1, None, baseline_diagnostic=False)
    src[RN.d0_id("CLASS")] = pp
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
    """Full registered new-fit bank (30 units per seed: C-TASK, 5 constrained, 24 weighted) plus the 27 D1 fixed-map
    decodes per seed (DIRECT-TASK proxy, FINE-TASK, CLASS, 24 privacy maps), on SYNTHETIC real-shaped data, including
    the trace build and an independent replay_unit of every unit (timed separately). Writes TIMING.json["fitting"]
    atomically under the flock of TIMING.json.lock (other keys kept)."""
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
            import zlib
            tb = json.dumps(files["trace.json"], allow_nan=False).encode()
            rb = json.dumps(RN._finite(rec), allow_nan=False).encode()
            c1 = time.process_time()
            rep = replay_unit(rec, fine, T, tr, Y, S_fit, refs, trace=json.loads(tb), files=files, per_state=False)
            rcpu = time.process_time() - c1
            if not rep["ok"]:
                sys.stderr.write(f"[timing s{k}] REPLAY FAILED {cid}: {rep['violations'][:3]}\n")
            rows[cid] = {"arm": arm, "lam": lam, "cpu_s": cpu, "wall_s": wall, "status": rec["status"],
                         "trace_cpu_s": rec["trace_cpu_s"], "record_bytes": len(rb), "trace_bytes": len(tb),
                         "trace_bytes_zlib6": len(zlib.compress(tb, 6)), "trace_moves": rec["trace_counts"]["moves"],
                         "trace_pairs": rec["trace_counts"]["pairs"], "replay_cpu_s": rcpu,
                         "replay_ok": rep["ok"], "replay_violations_first": rep["violations"][:3],
                         "replay_states": rep["n_states"],
                         "replay_max_abs_diff": rep["max_abs_diff"], "replay_bitwise_states": rep["bitwise_equal_states"],
                         "ceiling_shares_and_stops": sorted({(x.get("ceiling_share"), x.get("stop")) for s in
                                                             rec["starts"] for x in ([s] + s.get("stages", []) +
                                                                                     [s.get("stage1", {}),
                                                                                      s.get("stage2", {})])
                                                             if isinstance(x, dict) and x.get("stop")}),
                         "winner": rec["winner"]["start"] + ":" + rec["winner"]["kind"],
                         "evals": wk["evals"], "solves": wk["solves"], "solve_calls": wk["solve_calls"],
                         "memo_hits": wk["memo_hits"], "accepted": wk["accepted"], "pair_evals": wk["pair_evals"],
                         "pair_accepted": wk["pair_accepted"], "rejected_by_budget": wk["rejected_by_budget"],
                         "tokens_total": rec["tokens_total"], "deployed_feasible": rec["deployed"]["feasible"],
                         "parity_max": max(rec["deployed"]["parity_abs_diff"].values()),
                         "stops": sorted({x.get("stop") for s in rec["starts"] for x in
                                          ([s] + s.get("stages", []) + [s.get("stage1", {}), s.get("stage2", {})])
                                          if isinstance(x, dict) and x.get("stop")})}
            sys.stderr.write(f"[timing s{k}] {cid}: {cpu:.1f} CPU-s (trace {rec['trace_cpu_s']:.2f}) evals "
                             f"{wk['evals']} status {rec['status']} trace {len(tb) / 1e6:.2f} MB replay "
                             f"{rcpu:.1f} CPU-s\n")
            return rec

        def P_(cid):
            return done[cid][1]
        ct = run("C-TASK", starts={RN.d0_id("FINE-TASK"): src[RN.d0_id("FINE-TASK")]})
        refs = refs_from_ctask(ct)
        seq_src = {a: {RN.d0_id(a, l): src[RN.d0_id(a, l)] for l in RN.LAMS} for a in ("SEQ-12", "SEQ-21")}
        src_all = {RN.d0_id("FINE-TASK"): src[RN.d0_id("FINE-TASK")], RN.d0_id("CLASS"): src[RN.d0_id("CLASS")],
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
        d1_maps = [RN.d0_id("FINE-TASK"), RN.d0_id("FINE-TASK"), RN.d0_id("CLASS")] + \
            [RN.d0_id(f, l) for l in RN.LAMS for f in RN.PRIVACY]
        for j, cid in enumerate(d1_maps):
            c0 = time.process_time()
            _d1_decode(cid, src[cid], T, tr, Y, meta0)
            dec_rows[f"{j:02d}:{cid}" + ("(DIRECT-TASK proxy)" if j == 0 else "")] = time.process_time() - c0
        per_seed.append({"seed": k, "setup_cpu_s": setup_cpu, "units": rows, "d1_fixed_map_decodes": dec_rows})
    agg = {}
    for key in ("trace_cpu_s", "record_bytes", "trace_bytes", "trace_bytes_zlib6", "replay_cpu_s"):
        agg[key] = [float(sum(v[key] for v in s["units"].values())) for s in per_seed]
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
                   "binary S correlated with both scores) + lra.mapper.synthetic_labels (miscalibrated law); fine "
                   "partitions qpc.partition 32/128 per class; source D0 maps fitted with plain qpc (setup, not "
                   "counted). No Adult row, label or SEX was read.",
           "environment": {"threads": "OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, one process",
                           "semaphore": "lra.sema (label C:fit-timing)", "python": platform.python_version(),
                           "numpy": np.__version__, "machine": platform.machine()},
           "measure": "process CPU seconds per unit (time.process_time), including the decoder.json/release.npz "
                      "build and the deployed recomputation",
           "code_sha256_at_start": code0, "code_unchanged_during_run": bool(code_hashes() == code0),
           "replay_ok_all_units": bool(all(v["replay_ok"] for s in per_seed for v in s["units"].values())),
           "seeds_measured": list(seeds), "units_per_seed": len(per_seed[0]["units"]) if per_seed else 0,
           "d1_fixed_map_decodes_per_seed": 27,
           "measured": {"fit_cpu_s_per_seed": fit_cpu, "d1_decode_cpu_s_per_seed": dec_cpu,
                        "cpu_s_by_arm": {a: {"units": len(v), "total": float(sum(v)), "mean": float(np.mean(v)),
                                             "max": float(max(v))} for a, v in by_arm.items()},
                        "setup_cpu_s_per_seed_not_counted": [s["setup_cpu_s"] for s in per_seed],
                        "trace_cpu_s_per_seed_included_in_fit": agg["trace_cpu_s"],
                        "trace_cpu_share_of_fit": [a / b if b else 0.0 for a, b in zip(agg["trace_cpu_s"], fit_cpu)],
                        "record_bytes_per_seed": agg["record_bytes"], "trace_bytes_per_seed": agg["trace_bytes"],
                        "trace_bytes_zlib6_per_seed": agg["trace_bytes_zlib6"],
                        "replay_cpu_s_per_seed_not_in_fit": agg["replay_cpu_s"],
                        "by_arm_trace_and_replay": {a: {"trace_cpu_s": float(sum(v["trace_cpu_s"] for s in per_seed
                                                                                 for v in s["units"].values()
                                                                                 if v["arm"] == a)),
                                                        "trace_bytes_max": int(max(v["trace_bytes"] for s in per_seed
                                                                                   for v in s["units"].values()
                                                                                   if v["arm"] == a)),
                                                        "record_bytes_max": int(max(v["record_bytes"] for s in per_seed
                                                                                    for v in s["units"].values()
                                                                                    if v["arm"] == a)),
                                                        "replay_cpu_s": float(sum(v["replay_cpu_s"] for s in per_seed
                                                                                  for v in s["units"].values()
                                                                                  if v["arm"] == a))}
                                                    for a in by_arm},
                        "peak_rss_bytes": int(maxrss if sys.platform == "darwin" else maxrss * 1024)},
           "per_seed": per_seed,
           "projection": {"bank_3_seeds_cpu_s": proj, "bank_3_seeds_cpu_h": proj / 3600,
                          "conservative_x2_cpu_h": 2 * proj / 3600,
                          "units": "3 C-TASK + 72 weighted + 15 constrained = 90 new mapping-pair fits, plus 81 D1 "
                                   "fixed-map decodes (27 per seed incl. CLASS|D1)",
                          "trace_bytes_3_seeds": 3 * float(np.mean(agg["trace_bytes"])) if per_seed else 0.0,
                          "record_bytes_3_seeds": 3 * float(np.mean(agg["record_bytes"])) if per_seed else 0.0,
                          "replay_all_units_3_seeds_cpu_h": 3 * float(np.mean(agg["replay_cpu_s"])) / 3600
                          if per_seed else 0.0,
                          "basis": f"mean of {n_seeds} measured synthetic seed(s) x 3; qpc/cbp real/synthetic ratio "
                                   "about 1.0 for the comparable bank",
                          "budget_line": "fitting should be well under 3 CPU-h of the 20 CPU-h study"},
           "recommendation": None}
    rec["recommendation"] = {
        "bank": "FULL (90 new units + 81 D1 fixed-map decodes incl. CLASS|D1); no reduction",
        "sharding": "2 shards under lra.sema: lra.run fit_chains (constrained chains first, then weighted); "
                    "ctask stage first (3 units), d1 stage independent",
        "within_budget": bool(2 * proj / 3600 < 3.0)}
    rec = RN._finite(rec)
    if out_path is not None:
        import fcntl
        import os
        from pathlib import Path
        p = Path(out_path)
        lk = p.with_name(p.name + ".lock")
        with open(lk, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                cur = json.loads(p.read_text()) if p.exists() else {}
                cur["fitting"] = rec                          # role C owns only this key; the others are kept
                tmp = p.with_name(p.name + ".tmp_fitting")
                tmp.write_text(json.dumps(cur, indent=1, allow_nan=False) + "\n")
                tmp.replace(p)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        try:
            os.remove(lk)
        except FileNotFoundError:
            pass
    return rec


# ------------------------------------------------------------------ CLI
def main(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m lra.mapper")
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
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1 (run under lra.sema)")
    r = timing(tuple(int(x) for x in a.seeds.split(",")), a.out, units=a.units)
    sys.stdout.write(json.dumps({k: r[k] for k in ("measured", "projection", "recommendation")}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
