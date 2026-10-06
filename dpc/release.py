"""Coarse class-preserving release policies: encoding, recipient views and exact serialization.

A Policy for one recipient = an assignment partition (``fine``: the FinePartition the deployment rule routes a row to)
plus a map ``cell_token`` from assignment cells to integer token IDs. Tokens never cross predicted classes. Token IDs
are enumerated in (class, coarse index) order, coarse cells of a class ordered by their lowest member fine-cell index;
a class without fitting rows has exactly one fallback token. Decoded prototype of a token =
smooth(S_t / n_t, class_t) with S_t, n_t the sums of its member fine cells (sequential accumulation in increasing
fine-cell index); a fallback token decodes to smooth(uniform, class). Decision of a token = its class, which is
checked to be the strict argmax of its prototype.

The release for a person is (token id, decoded probability vector, decision). encode() checks pointwise that the
released decision equals argmax(p) for every row and raises otherwise.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from dpc.partition import (EPS, FinePartition, assign_fine, check_decisions, check_probs, check_prototypes, smooth)

FAMILIES = ("CLASS-ONLY", "FINE-TASK", "DIRECT-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT")


def token_tables(fine: FinePartition, cell_token):
    """(token_class, token_n, token_S, token_fallback, token_proto) from the assignment partition and the map."""
    cell_token = np.asarray(cell_token, dtype=np.int64)
    T = int(cell_token.max()) + 1
    K = fine.K
    tc = np.full(T, -1, dtype=np.int64)
    tn = np.zeros(T, dtype=np.int64)
    tS = np.zeros((T, K))
    tfb = np.zeros(T, dtype=bool)
    for f in range(fine.F):                       # sequential accumulation in increasing fine index
        t = cell_token[f]
        c = int(fine.cell_class[f])
        if tc[t] == -1:
            tc[t] = c
        elif tc[t] != c:
            raise ValueError(f"token {t} mixes predicted classes {tc[t]} and {c}; tokens must stay within a class")
        tn[t] += fine.n[f]
        tS[t] = tS[t] + fine.S[f]
        tfb[t] = tfb[t] or bool(fine.fallback[f])
    if np.any(tc < 0):
        raise ValueError("every token ID in 0..T-1 must be used by at least one cell")
    proto = np.zeros((T, K))
    for t in range(T):
        if tn[t] > 0:
            proto[t] = smooth(tS[t] / tn[t], tc[t])
        else:
            proto[t] = smooth(np.full(K, 1.0 / K), tc[t])
    return tc, tn, tS, tfb, proto


def canonical_tokens(fine: FinePartition, labels):
    """Canonical token IDs for a grouping of fine cells (``labels`` any per-cell group key): tokens numbered by first
    appearance scanning fine cells in increasing index, i.e. (class, lowest member index) order."""
    labels = np.asarray(labels)
    seen = {}
    out = np.empty(fine.F, dtype=np.int64)
    for f in range(fine.F):
        key = labels[f].item() if hasattr(labels[f], "item") else labels[f]
        if key not in seen:
            seen[key] = len(seen)
        out[f] = seen[key]
    return out


@dataclass
class Policy:
    recipient: int
    fine: FinePartition
    cell_token: np.ndarray
    family: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cell_token = np.asarray(self.cell_token, dtype=np.int64)
        if self.cell_token.shape != (self.fine.F,):
            raise ValueError("cell_token must map every assignment cell")
        self.fine.validate()
        tc, tn, tS, tfb, proto = token_tables(self.fine, self.cell_token)
        # token IDs must be canonical: classes non-decreasing in token order and first-appearance numbering
        if not np.array_equal(canonical_tokens(self.fine, self.cell_token), self.cell_token):
            raise ValueError("token IDs are not in canonical (class, lowest member cell) order; use canonical_tokens")
        if np.any(tfb & (tn > 0)):
            raise ValueError("a fallback token cannot carry fitting rows")
        for c in range(self.fine.K):
            toks = np.flatnonzero(tc == c)
            if toks.size == 0:
                raise ValueError(f"class {c} has no token")
        check_prototypes(proto, tc)
        self.token_class, self.token_n, self.token_S, self.token_fallback, self.token_proto = tc, tn, tS, tfb, proto

    @property
    def K(self):
        return self.fine.K

    @property
    def T(self):
        return int(self.token_class.shape[0])

    def tokens_per_class(self):
        return [int(np.sum(self.token_class == c)) for c in range(self.K)]

    def effective_states_per_class(self):
        """Tokens with at least one fitting row, per predicted class."""
        return [int(np.sum((self.token_class == c) & (self.token_n > 0))) for c in range(self.K)]

    def max_abs_unsmoothed_vs_smoothed(self):
        used = self.token_n > 0
        if not used.any():
            return 0.0
        return float(np.max(np.abs(self.token_S[used] / self.token_n[used, None] - self.token_proto[used])))

    def fingerprint(self):
        """Hash of everything that determines deployment and decoding: assignment centroids/classes, map, prototypes."""
        h = hashlib.sha256()
        for a in (np.int64(self.K), self.fine.cell_class.astype("<i8"), self.fine.centroid.astype("<f8"),
                  self.cell_token.astype("<i8"), self.token_class.astype("<i8"), self.token_proto.astype("<f8")):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def to_dict(self):
        return {"kind": "dpc.Policy", "recipient": int(self.recipient), "family": self.family,
                "fine": self.fine.to_dict(), "cell_token": self.cell_token.tolist(),
                "token_class": self.token_class.tolist(), "token_proto": self.token_proto.tolist(),
                "token_n": self.token_n.tolist(), "meta": self.meta, "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, z):
        if z.get("kind") != "dpc.Policy":
            raise ValueError("not a dpc.Policy record")
        p = cls(recipient=int(z["recipient"]), fine=FinePartition.from_dict(z["fine"]),
                cell_token=np.asarray(z["cell_token"], dtype=np.int64), family=z.get("family", ""),
                meta=z.get("meta", {}))
        if "token_proto" in z and not np.array_equal(np.asarray(z["token_proto"], dtype=np.float64), p.token_proto):
            raise ValueError("stored prototypes differ from those recomputed from the cell sums")
        if "fingerprint" in z and z["fingerprint"] != p.fingerprint():
            raise ValueError("policy fingerprint mismatch after load")
        return p


@dataclass
class PolicyPair:
    p1: Policy
    p2: Policy
    family: str = ""
    config: dict = field(default_factory=dict)

    def fingerprint(self):
        return hashlib.sha256((self.p1.fingerprint() + self.p2.fingerprint()).encode()).hexdigest()

    # tuple-like access: pair[0] is recipient 1, pair[1] recipient 2
    def __getitem__(self, i):
        return (self.p1, self.p2)[i]

    def __iter__(self):
        return iter((self.p1, self.p2))

    def __len__(self):
        return 2

    def to_dict(self):
        return {"kind": "dpc.PolicyPair", "family": self.family, "config": self.config, "p1": self.p1.to_dict(),
                "p2": self.p2.to_dict(), "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, z):
        if z.get("kind") != "dpc.PolicyPair":
            raise ValueError("not a dpc.PolicyPair record")
        pp = cls(Policy.from_dict(z["p1"]), Policy.from_dict(z["p2"]), z.get("family", ""), z.get("config", {}))
        if "fingerprint" in z and z["fingerprint"] != pp.fingerprint():
            raise ValueError("policy pair fingerprint mismatch after load")
        return pp


# ----------------------------------------------------------------------------------------------- serialization
def alphabet_size(policy: Policy):
    """Full token alphabet of one recipient (including fallback tokens)."""
    return policy.T


def fingerprint(obj):
    """Canonical fingerprint of a Policy or PolicyPair (assignment centroids, map, prototypes)."""
    return obj.fingerprint()


def policy_pair_to_dict(pair: PolicyPair):
    return pair.to_dict()


def policy_pair_from_dict(z):
    return PolicyPair.from_dict(z)


def save_policy(obj, path):
    """JSON (floats via repr: exact float64 round trip). Works for Policy and PolicyPair."""
    with open(path, "w") as fh:
        json.dump(obj.to_dict(), fh)


def load_policy(path):
    with open(path) as fh:
        z = json.load(fh)
    return PolicyPair.from_dict(z) if z.get("kind") == "dpc.PolicyPair" else Policy.from_dict(z)


def policy_to_npz_arrays(pol: Policy, prefix=""):
    f = pol.fine
    a = {"K": np.int64(f.K), "recipient": np.int64(pol.recipient), "eps": np.float64(f.eps),
         "cell_class": f.cell_class, "centroid": f.centroid, "mean": f.mean, "n": f.n, "S": f.S, "A": f.A,
         "fallback": f.fallback, "cell_token": pol.cell_token, "token_proto": pol.token_proto,
         "family": np.array(pol.family), "meta_json": np.array(json.dumps(pol.meta)),
         "fine_receipt_json": np.array(json.dumps(f.receipt)), "fingerprint": np.array(pol.fingerprint())}
    return {prefix + k: v for k, v in a.items()}


def save_policy_npz(pol: Policy, path):
    np.savez(path, **policy_to_npz_arrays(pol))


def policy_from_npz_arrays(z, prefix=""):
    g = lambda k: z[prefix + k]  # noqa: E731
    K = int(g("K"))
    fine = FinePartition(K=K, cell_class=np.asarray(g("cell_class"), dtype=np.int64),
                         centroid=np.asarray(g("centroid"), dtype=np.float64), mean=np.asarray(g("mean")),
                         n=np.asarray(g("n"), dtype=np.int64), S=np.asarray(g("S"), dtype=np.float64),
                         A=np.asarray(g("A"), dtype=np.float64), fallback=np.asarray(g("fallback"), dtype=bool),
                         eps=float(g("eps")), receipt=json.loads(str(g("fine_receipt_json"))))
    if fine.eps != EPS:
        raise ValueError("eps differs from the fixed 1e-12 rule")
    p = Policy(recipient=int(g("recipient")), fine=fine, cell_token=np.asarray(g("cell_token")),
               family=str(g("family")), meta=json.loads(str(g("meta_json"))))
    if not np.array_equal(np.asarray(g("token_proto")), p.token_proto) or str(g("fingerprint")) != p.fingerprint():
        raise ValueError("npz policy failed its prototype/fingerprint check")
    return p


def load_policy_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return policy_from_npz_arrays(z)


# ----------------------------------------------------------------------------------------------- encoding / views
def encode(policy: Policy, P, d=None):
    """(tokens int64, decoded probabilities float64 (n, K), decisions int64). Raises if any released decision differs
    from argmax(P) (pointwise class preservation)."""
    P = check_probs(P, policy.K)
    d = check_decisions(P, d)
    cell = assign_fine(P, d, policy.fine)
    tok = policy.cell_token[cell]
    probs = policy.token_proto[tok]
    dec = policy.token_class[tok]
    if not np.array_equal(dec, d):
        raise AssertionError("class preservation violated")
    if not np.array_equal(probs.argmax(1), d):
        raise AssertionError("decoded prototype argmax differs from the decision")
    return tok.astype(np.int64), probs.astype(np.float64), dec.astype(np.int64)


def onehot(idx, width):
    out = np.zeros((idx.shape[0], width), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def recipient_view(policy: Policy, P, d=None):
    tok, probs, dec = encode(policy, P, d)
    return {"tokens": tok, "token_onehot": onehot(tok, policy.T), "probs": probs,
            "decision": dec, "decision_onehot": onehot(dec, policy.K), "alphabet": policy.T}


def release_views(policy1: Policy, policy2: Policy, P1, d1, P2, d2):
    """Per-recipient views (one-hot token over the FULL alphabet incl. fallbacks, decoded probabilities, decision
    one-hot) and the aligned pair. ``pair_index`` = t1 * T2 + t2 is the exact token tuple as one integer."""
    v1 = recipient_view(policy1, P1, d1)
    v2 = recipient_view(policy2, P2, d2)
    if v1["tokens"].shape[0] != v2["tokens"].shape[0]:
        raise ValueError("recipient inputs must be aligned rows")
    pair = {"tokens": np.stack([v1["tokens"], v2["tokens"]], 1),
            "pair_index": v1["tokens"] * policy2.T + v2["tokens"],
            "pair_alphabet": policy1.T * policy2.T,
            "token_onehot": np.hstack([v1["token_onehot"], v2["token_onehot"]]),
            "probs": np.hstack([v1["probs"], v2["probs"]]),
            "decision_onehot": np.hstack([v1["decision_onehot"], v2["decision_onehot"]])}
    return {"r1": v1, "r2": v2, "pair": pair}


def renumbered_view(policy: Policy, perm, P, d=None):
    """Token one-hot view after an invertible renumbering t -> perm[t] (used to test that views are invariant up to a
    column permutation; canonical Policy objects always use canonical IDs)."""
    perm = np.asarray(perm, dtype=np.int64)
    if sorted(perm.tolist()) != list(range(policy.T)):
        raise ValueError("perm must be a permutation of the token alphabet")
    tok, probs, dec = encode(policy, P, d)
    return onehot(perm[tok], policy.T), probs, dec
