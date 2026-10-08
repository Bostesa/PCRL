"""Exact finite-law oracles (ccm; role C; PROTOCOL section 6; MATH_REVIEW.md R9). SYNTHETIC tests only
(tests/pcrl_confidence_constrained_mechanism_v1/test_oracle.py). Contract constants and capacities come from
ccm.guard.CONFIG; certificates use ccm.guard.bin_representative (construction targets + canonical predicate).

LAW: n inputs x (n <= ORACLE_ENGINEERING['max_inputs']) with P(x) (sums to 1), P(S=1|x) in [0, 1], reference vectors
p1(x) (K1 classes, recipient 1 = income) and p2(x) (K2, recipient 2 = occupation), decisions dec1(x), dec2(x) (default
first-index argmax). Capacity per predicted class: recipient 1 -> CONFIG income (8), recipient 2 -> CONFIG occupation
(64), whatever K is (the toy laws use K2 = 3 for recipient 2).

PARTITION DOMAIN (TOY_LAWS conventions): a release partitions the DISTINCT reference values of a recipient; inputs with
equal vectors always share a token. Values are indexed by first appearance in law order. A deterministic release is a
set partition of the values into blocks of ONE decision class, every block admissible (bin_representative CERTIFIED),
at most cap blocks per class. The recipient's view is the TOKEN (block identity); its representative is a function of
it, so MI(S; token) >= MI(S; q). Fixed label-free order of partitions: the restricted growth string (RGS) over values,
compared lexicographically.

MEASURES (exact plug-in, float64): MI(S; T) in nats = sum_{t,s} P(t,s) [log P(t,s) - log P(t) - log P(s)] (0 log 0 = 0)
and Bayes accuracy BA = sum_t max_s P(t,s), for t1, t2 and the coalition (t1, t2).

ARMS (minimisation: values within ORACLE_ENGINEERING['tie_tol'] = 1e-12 of the minimum are tied; ties -> fewest tokens
-> lexicographic RGS; the number of tied optima is reported as n_tied_optima):
  decision_only            tokens = decisions; confidence-INELIGIBLE (no released vector); a leakage floor (R9.2).
  identity_release         reference: each distinct value its own token (release p itself); eligible iff every value
                           satisfies G for itself (strict top).
  class                    one block per predicted class; eligible iff every class block is CERTIFIED.
  task_only                per recipient: fewest admissible tokens, ties by lexicographic RGS (SEX-blind).
  local                    per recipient: min own MI(S; t_i).
  seq12_nonadaptive        t1 = local of recipient 1; t2 = argmin over recipient 2's admissible partitions of the
                           coalition MI(S; (t1, t2)). NON-ADAPTIVE: chosen given the first PARTITION, not conditioned on
                           realised tokens (not Taylor et al.'s adaptive algorithm).
  seq21_nonadaptive        the same with the recipients swapped.
  joint                    min coalition MI(S; (t1, t2)) over admissible pairs; ties: total tokens, then (RGS1, RGS2).
  stochastic_local         per recipient an LP (scipy.optimize.linprog, HiGHS) minimising own Bayes accuracy over
                           channels W(t|v) on values whose every supported output satisfies G for v. Outputs are the
                           certified representatives of the MAXIMAL admissible value blocks, allowed for v iff v is in
                           the block: WLOG by R9.5 (relabel any output to a maximal set containing what it serves, then
                           merge), so the finite LP is EXACT for the uncapped problem.
                           CAPACITY: the toy laws have at most 6 inputs and capacity never binds; the LP is solved
                           only if every predicted class has at most cap maximal admissible blocks (then every
                           channel the LP can return respects the capacity and the uncapped optimum IS the capped
                           optimum). Otherwise the arm REFUSES
                           (REFUSED_CAPACITY_COULD_BIND) instead of solving a relaxed LP.
                           Receipts: the LP optimum; an independent re-evaluation of the returned channel (entries below
                           lp_support_tol zeroed, rows renormalised; BA and MI recomputed from W); every supported
                           (v, output) re-checked with the canonical predicate; a cross-check LP over the specified
                           candidate list (representatives of ALL admissible blocks containing v, plus p(v) itself when
                           G holds for it) whose optimum must agree; the coalition of the two channels (independent
                           given x).
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

from ccm import guard as GD

ORACLE_ENGINEERING = {
    "tie_tol": 1e-12,          # nats / accuracy units: numerical tie between candidate optima
    "lp_support_tol": 1e-12,   # channel entries below this are treated as zero in the re-evaluation
    "lp_recheck_tol": 1e-9,    # |re-evaluated BA - LP optimum| and |cross-check optimum - optimum| allowed
    "max_inputs": 8,
}

_ALIASES = {
    "P_x": ("P_float", "P_x", "p_x", "Px", "P", "prob"),
    "P_s1": ("sex1_float", "P_s1", "p_s1", "P_S1_given_x", "sex1"),
    "p1": ("p1_float", "p1", "P1"),
    "p2": ("p2_float", "p2", "P2"),
    "dec1": ("dec1", "d1"),
    "dec2": ("dec2", "d2"),
    "names": ("id", "names", "ids", "name"),
}


def _num(v):
    """Float or (nested) list of floats from floats or exact rational strings like '1/4'."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_num(u) for u in v]
    if isinstance(v, str):
        return float(Fraction(v))
    return float(v)


def _pick(m, key, required=True):
    for a in _ALIASES[key]:
        if a in m:
            return m[a]
    if required:
        raise KeyError(f"law mapping lacks {key!r} (accepted keys: {_ALIASES[key]})")
    return None


def law_from_mapping(m):
    """A law from a columnar mapping or from {'inputs': [record, ...]} (TOY_LAWS.json records: id, P_float, sex1_float,
    p1_float, p2_float; exact rational strings are accepted where floats are absent)."""
    if isinstance(m.get("inputs"), (list, tuple)) and m["inputs"] and isinstance(m["inputs"][0], dict):
        recs = m["inputs"]
        cols = {key: [_num(_pick(r, key)) for r in recs] for key in ("P_x", "P_s1", "p1", "p2")}
        for key in ("dec1", "dec2", "names"):
            v = [_pick(r, key, required=False) for r in recs]
            cols[key] = None if any(x is None for x in v) else v
    else:
        cols = {key: _num(_pick(m, key)) for key in ("P_x", "P_s1", "p1", "p2")}
        for key in ("dec1", "dec2", "names"):
            cols[key] = _pick(m, key, required=False)
    return validate_law(**cols)


def validate_law(P_x, P_s1, p1, p2, dec1=None, dec2=None, names=None):
    Px = np.asarray(P_x, dtype=np.float64)
    Ps1 = np.asarray(P_s1, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    n = Px.shape[0]
    if Px.ndim != 1 or Ps1.shape != (n,) or p1.ndim != 2 or p2.ndim != 2 or p1.shape[0] != n or p2.shape[0] != n:
        raise ValueError("law shapes: P_x (n,), P_s1 (n,), p1 (n, K1), p2 (n, K2)")
    if n == 0 or n > ORACLE_ENGINEERING["max_inputs"]:
        raise ValueError(f"law must have 1..{ORACLE_ENGINEERING['max_inputs']} inputs (got {n})")
    if np.any(Px < 0) or abs(Px.sum() - 1.0) > 1e-9 or np.any((Ps1 < 0) | (Ps1 > 1)):
        raise ValueError("P_x must be a distribution and P_s1 in [0, 1]")
    for p in (p1, p2):
        if np.any(p < 0) or np.any(np.abs(p.sum(axis=1) - 1.0) > 1e-9) or not np.all(np.isfinite(p)):
            raise ValueError("reference vectors must lie on the simplex")
    dec1 = GD.decisions(p1) if dec1 is None else np.asarray(dec1, dtype=np.intp)
    dec2 = GD.decisions(p2) if dec2 is None else np.asarray(dec2, dtype=np.intp)
    names = [str(i) for i in range(n)] if names is None else [str(v) for v in names]
    Pxs = np.stack([Px * (1.0 - Ps1), Px * Ps1], axis=1)      # P(x, s), s = 0, 1
    return {"n": n, "P_x": Px, "P_s1": Ps1, "p": {1: p1, 2: p2}, "dec": {1: dec1, 2: dec2}, "names": names,
            "Pxs": Pxs}


# ------------------------------------------------------------------------------------------------------ measures

def mi_ba_from_joint(J):
    """J (..., T, 2) = P(t, s). Returns (MI nats, Bayes accuracy) over the leading axes."""
    J = np.asarray(J, dtype=np.float64)
    Pt = J.sum(axis=-1, keepdims=True)
    Ps = J.sum(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(J > 0.0, J * (np.log(np.where(J > 0, J, 1.0)) - np.log(np.where(Pt > 0, Pt, 1.0))
                                      - np.log(np.where(Ps > 0, Ps, 1.0))), 0.0)
    return term.sum(axis=(-2, -1)), J.max(axis=-1).sum(axis=-1)


def joint_tokens(T, Pxs, n_cells):
    """T (A, n) token ids in [0, n_cells) -> J (A, n_cells, 2)."""
    T = np.atleast_2d(np.asarray(T, dtype=np.int64))
    A, n = T.shape
    idx = (T + n_cells * np.arange(A)[:, None]).ravel()
    J = np.empty((A, n_cells, 2))
    for s in (0, 1):
        J[:, :, s] = np.bincount(idx, weights=np.tile(Pxs[:, s], A), minlength=A * n_cells).reshape(A, n_cells)
    return J


def measures_tokens(t, Pxs):
    t = np.asarray(t, dtype=np.int64)
    mi, ba = mi_ba_from_joint(joint_tokens(t[None, :], Pxs, int(t.max()) + 1))
    return {"mi": float(mi[0]), "bayes_acc": float(ba[0])}


def measures_pair(t1, t2, Pxs):
    t1 = np.asarray(t1, dtype=np.int64)
    t2 = np.asarray(t2, dtype=np.int64)
    m2 = int(t2.max()) + 1
    return {"t1": measures_tokens(t1, Pxs), "t2": measures_tokens(t2, Pxs),
            "t12": measures_tokens(t1 * m2 + t2, Pxs)}


def channel_measures(W, Pxs):
    """W (n, T) channel rows (per input) -> MI and BA of S from the output."""
    J = np.einsum("xs,xt->ts", Pxs, W)
    mi, ba = mi_ba_from_joint(J)
    return {"mi": float(mi), "bayes_acc": float(ba)}


# ------------------------------------------------------------------------------------------ admissible partitions

def set_partitions_rgs(m):
    """All restricted growth strings of length m in lexicographic order (Bell(m) of them)."""
    if m == 0:
        yield ()
        return
    a = [0] * m

    def rec(i, mx):
        if i == m:
            yield tuple(a)
            return
        for v in range(mx + 2):
            a[i] = v
            yield from rec(i + 1, max(mx, v))
    yield from rec(1, 0)


def _rgs(labels):
    seen = {}
    out = []
    for v in labels:
        if v not in seen:
            seen[v] = len(seen)
        out.append(seen[v])
    return tuple(out)


class Recipient:
    """Admissible value partitions of one recipient (cached block certificates)."""

    def __init__(self, P, dec, cap, contract="G", d=None, b=None, names=None):
        P = np.asarray(P, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.intp)
        self.n_inputs, self.K = P.shape
        self.names = names or [str(i) for i in range(self.n_inputs)]
        keys, vid = {}, np.empty(self.n_inputs, dtype=np.int64)
        for x in range(self.n_inputs):
            key = tuple(P[x].tolist())
            if key not in keys:
                keys[key] = len(keys)
            vid[x] = keys[key]
        self.vid = vid
        self.m = len(keys)
        self.V = np.array([list(k) for k in keys], dtype=np.float64).reshape(self.m, self.K)
        self.vdec = np.empty(self.m, dtype=np.intp)
        for v in range(self.m):
            ds = set(dec[vid == v].tolist())
            if len(ds) != 1:
                raise ValueError("inputs with equal reference vectors must share the decision")
            self.vdec[v] = ds.pop()
        self.cap = int(cap)
        self.contract = contract
        self.d, self.b = GD._dk(d, b)
        self._blocks = {}
        self._enumerate()

    def inputs_of(self, vmask):
        return [x for x in range(self.n_inputs) if vmask >> int(self.vid[x]) & 1]

    def block(self, mask):
        """Certificate of a value block (bitmask over values; one decision class)."""
        if mask not in self._blocks:
            vals = [v for v in range(self.m) if mask >> v & 1]
            cls = {int(self.vdec[v]) for v in vals}
            if len(cls) != 1:
                raise ValueError("blocks must be within one decision class")
            q, cert = GD.bin_representative(self.V[vals], dec=cls.pop(), contract=self.contract, d=self.d, b=self.b)
            self._blocks[mask] = (q, cert, vals)
        return self._blocks[mask]

    def admissible(self, mask):
        return self.block(mask)[0] is not None

    def class_masks(self, c):
        vals = [v for v in range(self.m) if self.vdec[v] == c]
        return vals, [sum(1 << vals[i] for i in range(len(vals)) if s >> i & 1) for s in range(1, 1 << len(vals))]

    def _enumerate(self):
        classes = sorted(set(self.vdec.tolist()))
        per = []
        for c in classes:
            items = [v for v in range(self.m) if self.vdec[v] == c]
            opts = []
            for rgs in set_partitions_rgs(len(items)):
                nb = max(rgs) + 1
                if nb > self.cap:
                    continue
                masks = [0] * nb
                for it, lab in zip(items, rgs):
                    masks[lab] |= 1 << it
                if all(self.admissible(mk) for mk in masks):
                    opts.append(masks)
            per.append(opts)
        parts = []
        for combo in itertools.product(*per):
            masks = [mk for blocks in combo for mk in blocks]
            lab = np.empty(self.m, dtype=np.int64)
            for j, mk in enumerate(masks):
                for v in range(self.m):
                    if mk >> v & 1:
                        lab[v] = j
            parts.append((_rgs(lab.tolist()), len(masks)))
        parts.sort()
        self.rgs = np.array([p[0] for p in parts], dtype=np.int64).reshape(len(parts), self.m)
        self.n_tokens = np.array([p[1] for p in parts], dtype=np.int64)
        self.n_partitions = len(parts)
        self.tokens = self.rgs[:, self.vid] if parts else np.zeros((0, self.n_inputs), dtype=np.int64)

    def blocks_of(self, a):
        lab = self.rgs[a]
        return [sum(1 << v for v in range(self.m) if lab[v] == t) for t in range(int(lab.max()) + 1)]

    def describe(self, a):
        """Blocks of partition a as lists of input ids (token order)."""
        return [[self.names[x] for x in self.inputs_of(mk)] for mk in self.blocks_of(a)]

    def receipts(self, a):
        out = []
        for t, mk in enumerate(self.blocks_of(a)):
            q, cert, vals = self.block(mk)
            out.append({"token": t, "values": vals, "inputs": [self.names[x] for x in self.inputs_of(mk)],
                        "class": int(cert["dec"]), "q": q.tolist(), "status": cert["status"],
                        "method": cert.get("method"), "worst": cert.get("worst")})
        return out

    def tokens_per_class(self, a):
        lab = self.rgs[a]
        out = {}
        for t in range(int(lab.max()) + 1):
            c = int(self.vdec[int(np.flatnonzero(lab == t)[0])])
            out[c] = out.get(c, 0) + 1
        return out


def _argmin_tied(values, tokens, tol):
    """(index, n_tied): minimum value; ties (within tol of the minimum) -> fewest tokens -> lowest index (lex order)."""
    values = np.asarray(values, dtype=np.float64)
    m = np.min(values)
    cand = np.flatnonzero(values <= m + tol)
    tk = np.asarray(tokens)[cand]
    return int(cand[np.flatnonzero(tk == tk.min())[0]]), int(cand.size)


# --------------------------------------------------------------------------------------------------- stochastic LP

def _lp_min_ba(Pvs, outputs_allowed, T):
    """min sum_t max_s sum_v P(v,s) W(t|v) over channels supported on the allowed (v, t) pairs."""
    m = Pvs.shape[0]
    pairs = [(v, t) for v in range(m) for t in outputs_allowed[v]]
    nw = len(pairs)
    c = np.concatenate([np.zeros(nw), np.ones(T)])
    A_ub = np.zeros((2 * T, nw + T))
    A_eq = np.zeros((m, nw + T))
    for j, (v, t) in enumerate(pairs):
        A_ub[2 * t, j] = Pvs[v, 0]
        A_ub[2 * t + 1, j] = Pvs[v, 1]
        A_eq[v, j] = 1.0
    for t in range(T):
        A_ub[2 * t, nw + t] = -1.0
        A_ub[2 * t + 1, nw + t] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(2 * T), A_eq=A_eq, b_eq=np.ones(m),
                  bounds=[(0, None)] * (nw + T), method="highs")
    if res.status != 0:
        return None, res
    W = np.zeros((m, T))
    for j, (v, t) in enumerate(pairs):
        W[v, t] = res.x[j]
    return W, res


def maximal_admissible_blocks(R):
    """Per class, the admissible value blocks not strictly contained in another admissible block."""
    out = {}
    for c in sorted(set(R.vdec.tolist())):
        _, masks = R.class_masks(c)
        adm = [mk for mk in masks if R.admissible(mk)]
        out[c] = [mk for mk in adm if not any(o != mk and (o & mk) == mk for o in adm)]
    return out


def stochastic_local_lp(R, Pxs):
    """Exact stochastic local arm for one recipient (module docstring). Returns a JSON-able dict plus 'W' (values x
    outputs) and 'outputs'."""
    tol = ORACLE_ENGINEERING["lp_support_tol"]
    Pvs = np.zeros((R.m, 2))
    np.add.at(Pvs, R.vid, Pxs)
    maxi = maximal_admissible_blocks(R)
    per_class_max = {c: len(v) for c, v in maxi.items()}
    if any(len(v) > R.cap for v in maxi.values()):
        return {"status": "REFUSED_CAPACITY_COULD_BIND", "maximal_blocks_per_class": per_class_max, "cap": R.cap}
    outs = [mk for c in sorted(maxi) for mk in maxi[c]]
    Q = np.array([R.block(mk)[0] for mk in outs]).reshape(len(outs), R.K)
    allowed = [[t for t, mk in enumerate(outs) if mk >> v & 1] for v in range(R.m)]
    if any(not a for a in allowed):
        return {"status": "NO_ADMISSIBLE_OUTPUT", "values_without_output": [v for v in range(R.m) if not allowed[v]]}
    Wraw, res = _lp_min_ba(Pvs, allowed, len(outs))
    if Wraw is None:
        return {"status": "LP_FAILED", "message": str(res.message)}
    W = np.where(Wraw >= tol, Wraw, 0.0)
    W = W / W.sum(axis=1, keepdims=True)
    re = channel_measures(W, Pvs)
    sup_v, sup_t = np.nonzero(W > 0)
    chk = GD.check(Q[sup_t], R.V[sup_v], R.contract, R.d, R.b, R.vdec[sup_v])
    gap = abs(re["bayes_acc"] - float(res.fun))
    # cross-check: the specified candidate list (all admissible blocks containing v, plus p(v) when G holds for it)
    cand_q, index, cand = [], {}, [set() for _ in range(R.m)]

    def add(q):
        key = tuple(np.asarray(q, dtype=np.float64).tolist())
        if key not in index:
            index[key] = len(cand_q)
            cand_q.append(key)
        return index[key]

    for c in sorted(maxi):
        _, masks = R.class_masks(c)
        for mk in masks:
            q = R.block(mk)[0]
            if q is not None:
                t = add(q)
                for v in range(R.m):
                    if mk >> v & 1:
                        cand[v].add(t)
    for v in range(R.m):
        if GD.check(R.V[v], R.V[v], R.contract, R.d, R.b, int(R.vdec[v]))["ok"]:
            cand[v].add(add(R.V[v]))
    Wx, resx = _lp_min_ba(Pvs, [sorted(s) for s in cand], len(cand_q))
    cross = None if Wx is None else float(resx.fun)
    per_class = {}
    for t in sorted(set(sup_t.tolist())):
        cl = int(R.block(outs[t])[1]["dec"])
        per_class[cl] = per_class.get(cl, 0) + 1
    ok_all = bool(np.all(chk["ok"]))
    return {"status": "OPTIMAL", "lp_optimum_bayes_acc": float(res.fun), "reevaluated": re, "recheck_gap": gap,
            "all_supported_outputs_satisfy_contract": ok_all, "worst_supported": chk["worst"],
            "min_raw_entry": float(Wraw.min()), "cross_check_optimum": cross,
            "cross_check_gap": None if cross is None else abs(cross - float(res.fun)),
            "recheck_pass": bool(gap <= ORACLE_ENGINEERING["lp_recheck_tol"] and ok_all and cross is not None
                                 and abs(cross - float(res.fun)) <= ORACLE_ENGINEERING["lp_recheck_tol"]),
            "maximal_blocks_per_class": per_class_max, "n_outputs": len(outs),
            "supported_outputs_per_class": per_class,
            "within_capacity": bool(all(v <= R.cap for v in per_class.values())),
            "outputs_values": [[int(v) for v in range(R.m) if mk >> v & 1] for mk in outs],
            "W": W, "outputs": Q}


# ------------------------------------------------------------------------------------------------------- the oracle

def run_oracle(law, contract="G", d=None, b=None, cap1=None, cap2=None):
    """All arms on one law (a dict from validate_law / law_from_mapping, or a raw mapping)."""
    if "Pxs" not in law:
        law = law_from_mapping(law)
    d, b = GD._dk(d, b)
    Pxs = law["Pxs"]
    n = law["n"]
    tol = ORACLE_ENGINEERING["tie_tol"]
    caps = {1: GD.capacity_for("income") if cap1 is None else int(cap1),
            2: GD.capacity_for("occupation") if cap2 is None else int(cap2)}
    R = {i: Recipient(law["p"][i], law["dec"][i], caps[i], contract, d, b, law["names"]) for i in (1, 2)}
    rep = {"contract": contract, "d": d, "b": b, "n_inputs": n, "names": law["names"], "capacity": caps,
           "n_values": {i: R[i].m for i in (1, 2)},
           "n_admissible_partitions": {i: R[i].n_partitions for i in (1, 2)},
           "prior": {"P_S1": float(Pxs[:, 1].sum()), "bayes_acc_prior": float(Pxs.sum(axis=0).max())}, "arms": {}}
    arms = rep["arms"]

    def det_arm(name, a1, a2, extra=None):
        arm = {"eligible": True, "rgs": {1: R[1].rgs[a1].tolist(), 2: R[2].rgs[a2].tolist()},
               "partitions": {1: R[1].describe(a1), 2: R[2].describe(a2)},
               "tokens": {1: int(R[1].n_tokens[a1]), 2: int(R[2].n_tokens[a2])},
               "tokens_per_class": {1: R[1].tokens_per_class(a1), 2: R[2].tokens_per_class(a2)},
               "measures": measures_pair(R[1].tokens[a1], R[2].tokens[a2], Pxs),
               "receipts": {1: R[1].receipts(a1), 2: R[2].receipts(a2)}}
        if extra:
            arm.update(extra)
        arms[name] = arm

    arms["decision_only"] = {"eligible": False, "reason": "confidence-ineligible: no released vector (leakage floor)",
                             "measures": measures_pair(law["dec"][1], law["dec"][2], Pxs)}
    idok = {i: bool(np.all(GD.check(R[i].V, R[i].V, contract, d, b, R[i].vdec)["ok"])) for i in (1, 2)}
    arms["identity_release"] = {"eligible": bool(idok[1] and idok[2]), "eligible_per_recipient": idok,
                                "tokens": {i: R[i].m for i in (1, 2)},
                                "measures": measures_pair(R[1].vid, R[2].vid, Pxs)}
    cls = {}
    for i in (1, 2):
        per = {}
        for c in sorted(set(R[i].vdec.tolist())):
            mask = sum(1 << v for v in range(R[i].m) if R[i].vdec[v] == c)
            q, cert, vals = R[i].block(mask)
            per[int(c)] = {"values": vals, "status": cert["status"], "q": None if q is None else q.tolist(),
                           "worst": cert.get("worst")}
        cls[i] = per
    arms["class"] = {"eligible": all(v["status"] == "CERTIFIED" for i in (1, 2) for v in cls[i].values()),
                     "per_recipient": cls, "measures": measures_pair(law["dec"][1], law["dec"][2], Pxs)}
    if R[1].n_partitions == 0 or R[2].n_partitions == 0:
        rep["status"] = "NO_ADMISSIBLE_PARTITION"
        return rep
    rep["status"] = "OK"
    loc = {i: mi_ba_from_joint(joint_tokens(R[i].tokens, Pxs, n))[0] for i in (1, 2)}
    ta = {i: int(np.flatnonzero(R[i].n_tokens == R[i].n_tokens.min())[0]) for i in (1, 2)}
    det_arm("task_only", ta[1], ta[2])
    la, lt = {}, {}
    for i in (1, 2):
        la[i], lt[i] = _argmin_tied(loc[i], R[i].n_tokens, tol)
    det_arm("local", la[1], la[2], {"n_tied_optima": {1: lt[1], 2: lt[2]}})

    def coalition_given(i_fixed, a_fixed):
        j = 2 if i_fixed == 1 else 1
        tf = R[i_fixed].tokens[a_fixed]
        Tj = R[j].tokens
        cells = tf[None, :] * n + Tj if i_fixed == 1 else Tj * n + tf[None, :]
        return mi_ba_from_joint(joint_tokens(cells, Pxs, n * n))[0]

    a2, t2n = _argmin_tied(coalition_given(1, la[1]), R[2].n_tokens, tol)
    det_arm("seq12_nonadaptive", la[1], a2, {"design": "NON-ADAPTIVE: t2 chosen given the t1 PARTITION",
                                              "n_tied_optima": t2n})
    a1, t1n = _argmin_tied(coalition_given(2, la[2]), R[1].n_tokens, tol)
    det_arm("seq21_nonadaptive", a1, la[2], {"design": "NON-ADAPTIVE: t1 chosen given the t2 PARTITION",
                                              "n_tied_optima": t1n})
    A1, A2 = R[1].n_partitions, R[2].n_partitions
    M = np.empty((A1, A2))
    for a in range(A1):
        M[a] = coalition_given(1, a)
    cand = np.argwhere(M <= M.min() + tol)
    n_tied = int(cand.shape[0])
    tot = R[1].n_tokens[cand[:, 0]] + R[2].n_tokens[cand[:, 1]]
    cand = cand[tot == tot.min()]
    ja1, ja2 = (int(v) for v in cand[0])          # argwhere is row-major = lexicographic (RGS1, RGS2)
    det_arm("joint", ja1, ja2, {"n_pairs": int(A1 * A2), "n_tied_optima": n_tied})
    lp = {i: stochastic_local_lp(R[i], Pxs) for i in (1, 2)}
    arm = {"per_recipient": {i: {k: v for k, v in lp[i].items() if k not in ("W", "outputs")} for i in (1, 2)},
           "eligible": all(lp[i]["status"] == "OPTIMAL" for i in (1, 2))}
    if arm["eligible"]:
        W1 = lp[1]["W"][R[1].vid]
        W2 = lp[2]["W"][R[2].vid]
        Wc = np.einsum("xa,xb->xab", W1, W2).reshape(n, -1)
        arm["measures"] = {"t1": lp[1]["reevaluated"], "t2": lp[2]["reevaluated"], "t12": channel_measures(Wc, Pxs)}
        arm["channels"] = {i: {"W_values_by_output": lp[i]["W"].tolist(), "outputs": lp[i]["outputs"].tolist()}
                           for i in (1, 2)}
    arms["stochastic_local"] = arm
    return rep


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    return o


def run_all(laws, contract="G"):
    """Every law of a TOY_LAWS.json document (or a {name: law} mapping) -> JSON-able dict {name: oracle report}.
    Capacities: recipient 1 = income (8), recipient 2 = occupation (64) per class."""
    L = laws["laws"] if isinstance(laws, dict) and "laws" in laws else laws
    out = {}
    for name, law in L.items():
        out[name] = to_jsonable(run_oracle(law_from_mapping(law), contract=contract))
    return {"contract": contract, "constants": to_jsonable(GD.CONFIG), "oracle_engineering": ORACLE_ENGINEERING,
            "laws": out}
