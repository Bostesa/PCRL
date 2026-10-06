"""Class-preserving categorical release policies for the confidence-capacity study (qpc).

Provenance: adapted from dpc/release.py at source SHA 0a7b05a5 (imported, not edited). The arithmetic (token tables,
canonical token IDs, smoothed decoded prototypes, encode, content fingerprint) is dpc's; qpc adds its own record kinds
("qpc.FinePartition", "qpc.Policy", "qpc.PolicyPair", so old dpc outputs and new qpc outputs cannot be confused),
asymmetric per-recipient caps (m1, m2) in the pair config, mandatory metadata binding (teacher model sha256 and
feature-schema sha256), the dpc-format release arrays and explicit release invariants.

A Policy for recipient i = an assignment partition ``fine`` (routing centroids: the deployment rule sends a row of
predicted class c to the KL-nearest smoothed centroid among class c's cells, ties to the lowest index) plus a map
``cell_token`` from assignment cells to canonical token IDs. Stage A (DIRECT-TASK) policies use the identity map on a
direct k-means partition; Stage B policies map fine cells to coarse tokens. Tokens never cross predicted classes.
Token IDs are canonical: numbered by first appearance scanning cells in increasing index, i.e. (class, lowest member
cell) order. Decoded prototype of token t = smooth(S_t / n_t, class_t) = (mean + eps*1 + eps*e_c)/(1 + (K+1) eps),
eps = 1e-12, with S_t, n_t accumulated over member cells in increasing index; a fallback token decodes to
smooth(uniform, class). Decision of a token = its class (strict argmax of its prototype, checked).

Release per row: (token, decoded probability vector, decision). Two tokens with identical prototypes are distinct
disclosures (the full token identity is released and audited).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from dpc import release as DRL
from dpc.partition import FinePartition as _DpcFine
from qpc.kmeans import EPS, FinePartition, check_decisions, check_probs, smooth

canonical_tokens = DRL.canonical_tokens
token_tables = DRL.token_tables
onehot = DRL.onehot
PROTO_SUM_TOL = 1e-12
SCHEMA = "qpc-release-v1"


def _as_qpc_fine(f):
    if isinstance(f, FinePartition):
        return f
    if isinstance(f, _DpcFine):
        return FinePartition.from_dpc(f)
    raise TypeError("fine must be a FinePartition")


@dataclass
class Policy(DRL.Policy):
    """dpc.release.Policy with qpc kinds; fingerprint is dpc's content hash (centroids, map, prototypes)."""

    def __post_init__(self):
        self.fine = _as_qpc_fine(self.fine)
        super().__post_init__()

    def to_dict(self):
        z = super().to_dict()
        z["kind"] = "qpc.Policy"
        z["schema"] = SCHEMA
        return z

    @classmethod
    def from_dict(cls, z):
        if z.get("kind") != "qpc.Policy":
            raise ValueError(f"not a qpc.Policy record (kind {z.get('kind')!r}); dpc policies are not qpc releases")
        p = cls(recipient=int(z["recipient"]), fine=FinePartition.from_dict(z["fine"]),
                cell_token=np.asarray(z["cell_token"], dtype=np.int64), family=z.get("family", ""),
                meta=json.loads(json.dumps(z.get("meta", {}))))
        if not np.array_equal(np.asarray(z["token_proto"], dtype=np.float64), p.token_proto):
            raise ValueError("stored prototypes differ from those recomputed from the cell sums")
        if z.get("fingerprint") != p.fingerprint():
            raise ValueError("policy fingerprint mismatch after load")
        return p

    def caps_ok(self, m):
        """Every predicted class carries at most m tokens (a fallback class carries exactly one)."""
        return all(t <= max(int(m), 1) for t in self.tokens_per_class())


@dataclass
class PolicyPair(DRL.PolicyPair):
    def to_dict(self):
        z = super().to_dict()
        z["kind"] = "qpc.PolicyPair"
        z["schema"] = SCHEMA
        return z

    @classmethod
    def from_dict(cls, z):
        if z.get("kind") != "qpc.PolicyPair":
            raise ValueError(f"not a qpc.PolicyPair record (kind {z.get('kind')!r})")
        pp = cls(Policy.from_dict(z["p1"]), Policy.from_dict(z["p2"]), z.get("family", ""),
                 json.loads(json.dumps(z.get("config", {}))))
        if z.get("fingerprint") != pp.fingerprint():
            raise ValueError("policy pair fingerprint mismatch after load")
        return pp


def make_policy(recipient, fine, labels=None, family="", meta=None):
    """Policy from an assignment partition and any per-cell grouping key (None = identity map)."""
    fine = _as_qpc_fine(fine)
    lab = np.arange(fine.F) if labels is None else np.asarray(labels)
    return Policy(recipient=int(recipient), fine=fine, cell_token=canonical_tokens(fine, lab), family=family,
                  meta=dict(meta or {}))


def make_pair(p1, p2, family, m1, m2, lam=None, meta=None):
    if p1.recipient != 1 or p2.recipient != 2:
        raise ValueError("pair must be (recipient 1, recipient 2)")
    cfg = {"family": family, "m1": int(m1), "m2": int(m2), "lam": None if lam is None else float(lam),
           **(meta or {})}
    for p in (p1, p2):
        p.meta = {**p.meta, **cfg}
        p.family = family
    if not p1.caps_ok(m1) or not p2.caps_ok(m2):
        raise ValueError(f"policy exceeds its per-recipient caps ({m1}, {m2})")
    return PolicyPair(p1, p2, family, cfg)


# ----------------------------------------------------------------------------------------------- binding
BINDING_KEYS = ("teacher_model_sha256", "feature_names_sha256")


def binding(pair: PolicyPair):
    return {k: pair.config.get(k) for k in BINDING_KEYS}


def check_bound(pair: PolicyPair):
    b = binding(pair)
    miss = [k for k, v in b.items() if not (isinstance(v, str) and len(v) == 64)]
    if miss:
        raise ValueError(f"policy pair is not bound: missing {miss}")
    for p in pair:
        for k in BINDING_KEYS:
            if p.meta.get(k) != b[k]:
                raise ValueError(f"recipient {p.recipient} meta {k} differs from the pair binding")
    return b


# ----------------------------------------------------------------------------------------------- serialisation
def fingerprint(obj):
    return obj.fingerprint()


def alphabet_size(policy: Policy):
    """Full token alphabet (including fallback tokens)."""
    return policy.T


def to_dict(obj):
    return obj.to_dict()


def from_dict(z):
    k = z.get("kind")
    if k == "qpc.PolicyPair":
        return PolicyPair.from_dict(z)
    if k == "qpc.Policy":
        return Policy.from_dict(z)
    raise ValueError(f"not a qpc policy record (kind {k!r})")


def save_policy(obj, path):
    """JSON; float repr gives an exact float64 round trip."""
    with open(path, "w") as fh:
        json.dump(obj.to_dict(), fh, allow_nan=False)


def load_policy(path):
    with open(path) as fh:
        return from_dict(json.load(fh))


# ----------------------------------------------------------------------------------------------- encoding
def check_release_rows(policy: Policy, P, tok, q, dec):
    """Release invariants on the encoded rows: decision preservation, normalisation, strict argmax, and decoded
    vector = smooth(token mean) by the registered formula (recomputed here independently of token_tables)."""
    d = np.asarray(P).argmax(1)
    if not np.array_equal(dec, d):
        raise AssertionError("decision preservation violated")
    if q.shape[0] and np.max(np.abs(q.sum(1) - 1.0)) > PROTO_SUM_TOL:
        raise AssertionError("decoded vector not normalised")
    qc = q[np.arange(q.shape[0]), dec]
    other = q.copy()
    other[np.arange(q.shape[0]), dec] = -np.inf
    if q.shape[0] and not np.all(qc > other.max(1)):
        raise AssertionError("decoded vector argmax is not strictly the decision")
    K = policy.K
    used = np.unique(tok)
    for t in used:
        c = int(policy.token_class[t])
        if policy.token_n[t] > 0:
            mean = policy.token_S[t] / policy.token_n[t]
        else:
            mean = np.full(K, 1.0 / K)
        e = np.zeros(K)
        e[c] = EPS
        ref = (mean + EPS * np.ones(K) + e) / (1.0 + (K + 1) * EPS)
        if not np.array_equal(ref, policy.token_proto[t]):
            raise AssertionError(f"token {t}: decoded vector differs from the registered smoothing")
    return True


def encode(policy: Policy, P, d=None):
    """(tokens int64, decoded probabilities float64 (n, K), decisions int64) with every release invariant checked."""
    P = check_probs(P, policy.K)
    d = check_decisions(P, d)
    tok, q, dec = DRL.encode(policy, P, d)
    check_release_rows(policy, P, tok, q, dec)
    return tok, q, dec


def release_arrays(pair: PolicyPair, row_id, P1, d1, P2, d2):
    """dpc-format release over the given rows: row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2."""
    out = {"row_id": np.asarray(row_id)}
    for i, (pol, P, d) in enumerate(((pair.p1, P1, d1), (pair.p2, P2, d2)), 1):
        tok, q, dec = encode(pol, P, d)
        if tok.shape[0] != out["row_id"].shape[0]:
            raise ValueError("row_id is not aligned with the encoded rows")
        out.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": dec, f"alpha{i}": np.int64(alphabet_size(pol))})
    return out


def release_views(pair: PolicyPair, P1, d1, P2, d2):
    """Recipient views (token one-hot over the FULL alphabet incl. fallbacks, decoded probs, decision) and the pair
    (pair_index = t1 * T2 + t2)."""
    for pol, P, d in ((pair.p1, P1, d1), (pair.p2, P2, d2)):
        encode(pol, P, d)
    return DRL.release_views(pair.p1, pair.p2, P1, d1, P2, d2)


def token_parity(tokA, qA, tokB, qB):
    """Compare two releases of the same rows: exact token IDs, a token bijection (canonical relabelling) and decoded
    vectors (bitwise and max-abs difference)."""
    tokA, tokB = np.asarray(tokA, dtype=np.int64), np.asarray(tokB, dtype=np.int64)
    out = {"rows": int(tokA.shape[0]), "tokens_bitwise_equal": bool(np.array_equal(tokA, tokB))}
    pairs = np.unique(np.stack([tokA, tokB], 1), axis=0) if tokA.size else np.zeros((0, 2), dtype=np.int64)
    a_ok = np.unique(pairs[:, 0]).size == pairs.shape[0]
    b_ok = np.unique(pairs[:, 1]).size == pairs.shape[0]
    out["token_bijection"] = bool(a_ok and b_ok)
    out["bijection"] = {int(a): int(b) for a, b in pairs} if (a_ok and b_ok) else None
    qA, qB = np.asarray(qA, dtype=np.float64), np.asarray(qB, dtype=np.float64)
    out["q_bitwise_equal"] = bool(qA.shape == qB.shape and np.array_equal(qA, qB))
    out["q_max_abs_diff"] = float(np.max(np.abs(qA - qB))) if qA.shape == qB.shape and qA.size else None
    out["ok"] = bool(out["token_bijection"] and out["q_max_abs_diff"] is not None and out["q_max_abs_diff"] <= 1e-15)
    return out


def policy_receipt(pol: Policy, fit_tokens=None):
    r = {"alphabet": pol.T, "tokens_per_class": pol.tokens_per_class(),
         "effective_states_per_class": pol.effective_states_per_class(),
         "effective_states": int(sum(pol.effective_states_per_class())),
         "token_n": [int(x) for x in pol.token_n], "fallback_tokens": [int(t) for t in np.flatnonzero(pol.token_fallback)],
         "max_abs_unsmoothed_vs_smoothed": pol.max_abs_unsmoothed_vs_smoothed(),
         "fingerprint": pol.fingerprint(), "assignment_partition_fingerprint": pol.fine.fingerprint()}
    if fit_tokens is not None:
        u, cnt = np.unique(fit_tokens, return_counts=True)
        r["states_emitted_fit"] = int(u.size)
        r["fit_counts"] = {int(a): int(b) for a, b in zip(u, cnt)}
        p = cnt / cnt.sum()
        r["emitted_entropy_nats_fit"] = float(-(p * np.log(p)).sum())
        r["singleton_tokens_fit"] = int(np.sum(cnt == 1))
    return r


def sha_arrays(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
