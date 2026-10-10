"""Write-once, hash-bound bank-channel checkpoint package without labels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from zipfile import ZipFile

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import _bank_hash
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.source_bridge import action_map_sha256
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    FactorGate, ProductPolicy, LocalPolicy, SequentialPolicy,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.deterministic_policy import (
    DeterministicJointPolicy,
)
from .r2_laws import IdentityPolicy, BoundaryJointPolicy, BoundaryFactorPolicy


SCHEMA = "BAR1-selected-policy-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
COMMON = {"bank1_probability", "bank1_decision", "bank1_old_to_global",
          "bank2_probability", "bank2_decision", "bank2_old_to_global",
          "actions1", "actions2", "gate_w1", "gate_b1", "gate_w2", "gate_b2"}
FACTOR = {f"factor{r}_{key}" for r in (1, 2) for key in ("w1", "b1", "w2", "b2")}
SEQ = {f"{prefix}_{key}" for prefix in ("first", "second")
       for key in ("w1", "b1", "w2", "b2")} | {"second_embedding"}
KINDS = {"B0", "T", "D", "P", "L", "J", "S12", "S21"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _private(path):
    path = Path(path)
    if (path.is_symlink() or not path.is_dir() or
            stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise PermissionError("policy package directory must be owner-only")
    return path


def _padded(actions):
    out = np.full((len(actions), 4), -1, dtype=np.int64)
    for i, row in enumerate(actions):
        out[i, :len(row)] = row
    return out


def _unpad(array):
    if array.dtype != np.int64 or array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("action map has wrong type or shape")
    rows = []
    for row in array:
        used = row[row >= 0]
        if (not 1 <= len(used) <= 4 or len(set(used.tolist())) != len(used) or
                not np.array_equal(row, np.r_[used, np.full(4-len(used), -1)])):
            raise ValueError("action map has noncanonical padding")
        rows.append(tuple(int(v) for v in used))
    return tuple(rows)


def _base(policy):
    if isinstance(policy, BoundaryFactorPolicy):
        return policy.base
    return policy


def _kind(policy):
    if isinstance(policy, IdentityPolicy):
        return "identity"
    if isinstance(policy, BoundaryJointPolicy):
        return "boundary_joint"
    if isinstance(policy, BoundaryFactorPolicy):
        return "boundary_factor"
    if isinstance(policy, DeterministicJointPolicy):
        return "deterministic"
    if isinstance(policy, SequentialPolicy):
        return "sequential"
    raise TypeError("unregistered selected bank policy")


def _arrays(policy):
    base = _base(policy)
    arrays = {}
    for r, bank, actions in ((1, base.bank1, base.actions1),
                             (2, base.bank2, base.actions2)):
        arrays[f"bank{r}_probability"] = bank.probability
        arrays[f"bank{r}_decision"] = bank.decision
        arrays[f"bank{r}_old_to_global"] = np.asarray(bank.old_to_global, dtype=np.int64)
        arrays[f"actions{r}"] = _padded(actions)
    for name in ("w1", "b1", "w2", "b2"):
        arrays[f"gate_{name}"] = getattr(base.gate, name)
    if isinstance(base, (ProductPolicy, LocalPolicy)):
        for r, factor in ((1, base.factor1), (2, base.factor2)):
            for name in ("w1", "b1", "w2", "b2"):
                arrays[f"factor{r}_{name}"] = getattr(factor, name)
    elif isinstance(base, SequentialPolicy):
        for prefix, factor in (("first", base.first), ("second", base.second)):
            for name in ("w1", "b1", "w2", "b2"):
                arrays[f"{prefix}_{name}"] = getattr(factor, name)
        arrays["second_embedding"] = base.second.embedding
    return arrays


def save_policy(directory, policy, *, arm_id, source_seed,
                source_binding_sha256, fit_receipt_sha256,
                selected_start, selected_update):
    root = _private(directory)
    kind = _kind(policy)
    base = _base(policy)
    if (arm_id not in KINDS or source_seed not in (0, 1) or
            not all(type(x) is str and HEX64.fullmatch(x) for x in
                    (source_binding_sha256, fit_receipt_sha256)) or
            type(selected_start) is not str or
            type(selected_update) is not int or selected_update < 0):
        raise ValueError("policy source/selection binding is incomplete")
    if arm_id == "B0" and kind != "identity":
        raise ValueError("arm and identity law disagree")
    if (arm_id in ("J", "T") and kind not in ("boundary_joint", "identity") or
            arm_id in ("P", "L") and kind not in ("boundary_factor", "identity") or
            arm_id == "D" and kind not in ("deterministic", "identity") or
            arm_id in ("S12", "S21") and kind not in ("sequential", "identity")):
        raise ValueError("arm and fitted law differ")
    arr = _arrays(policy)
    expected = COMMON | (FACTOR if isinstance(base, (ProductPolicy, LocalPolicy)) else set()) | \
               (SEQ if isinstance(base, SequentialPolicy) else set())
    if set(arr) != expected or any(not np.isfinite(v).all() for v in arr.values()):
        raise ValueError("policy arrays differ from its registered architecture")
    package = root / "policy.npz"
    receipt_path = root / "POLICY_RECEIPT.json"
    if package.exists() or receipt_path.exists():
        raise FileExistsError("selected policy is write-once")
    with os.fdopen(os.open(package, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as out:
        np.savez_compressed(out, **arr)
        out.flush(); os.fsync(out.fileno())
    receipt = {
        "schema": SCHEMA, "arm_id": arm_id, "source_seed": source_seed,
        "source_binding_sha256": source_binding_sha256,
        "fit_receipt_sha256": fit_receipt_sha256,
        "selected_start": selected_start, "selected_update": selected_update,
        "policy_kind": kind, "base_law_kind": base.law_kind,
        "eta": policy.eta if isinstance(policy, BoundaryJointPolicy) else None,
        "eta1": policy.eta1 if isinstance(policy, BoundaryFactorPolicy) else None,
        "eta2": policy.eta2 if isinstance(policy, BoundaryFactorPolicy) else None,
        "order": base.order if isinstance(base, SequentialPolicy) else None,
        "first_checkpoint_sha256": (base.first_checkpoint_sha256
                                    if isinstance(base, SequentialPolicy) else None),
        "bank_sha256_by_recipient": {"r1": _bank_hash(base.bank1),
                                     "r2": _bank_hash(base.bank2)},
        "action_sha256_by_recipient": {"r1": action_map_sha256(base.actions1),
                                       "r2": action_map_sha256(base.actions2)},
        "policy_file_sha256": _sha(package),
    }
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(receipt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as out:
        out.write(payload)
        out.flush(); os.fsync(out.fileno())
    return {**receipt, "receipt_sha256": hashlib.sha256(payload).hexdigest()}


def load_policy(directory, *, receipt_sha256):
    root = _private(directory)
    receipt_path, package = root / "POLICY_RECEIPT.json", root / "policy.npz"
    if type(receipt_sha256) is not str or not HEX64.fullmatch(receipt_sha256) or \
            _sha(receipt_path) != receipt_sha256:
        raise ValueError("selected policy receipt hash differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("arm_id") not in KINDS or \
            _sha(package) != receipt.get("policy_file_sha256"):
        raise ValueError("selected policy package hash differs")
    kind = receipt["policy_kind"]
    expected = COMMON | (FACTOR if receipt["base_law_kind"] in
                         (ProductPolicy.law_kind, LocalPolicy.law_kind) else set()) | \
               (SEQ if receipt["base_law_kind"] == SequentialPolicy.law_kind else set())
    with ZipFile(package) as archive:
        if {x.filename for x in archive.infolist()} != {f"{x}.npy" for x in expected}:
            raise ValueError("policy array members differ")
    with np.load(package, allow_pickle=False) as archive:
        arr = {name: np.array(archive[name], copy=True) for name in expected}
    banks, actions = [], []
    for r in (1, 2):
        p, d, old = (arr[f"bank{r}_{key}"] for key in
                     ("probability", "decision", "old_to_global"))
        if (p.dtype != np.float64 or p.ndim != 2 or d.dtype != np.int64 or
                d.shape != (len(p),) or old.dtype != np.int64 or old.ndim != 1 or
                not np.isfinite(p).all() or np.any(p < 0) or
                not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12) or
                not np.array_equal(np.argmax(p, axis=1), d) or
                np.any((old < 0) | (old >= len(p)))):
            raise ValueError("frozen bank schema differs")
        bank = FixedBank(p, d, tuple(int(x) for x in old))
        action = _unpad(arr[f"actions{r}"])
        if (_bank_hash(bank) != receipt["bank_sha256_by_recipient"][f"r{r}"] or
                action_map_sha256(action) != receipt["action_sha256_by_recipient"][f"r{r}"]):
            raise ValueError("frozen bank or action hash differs")
        banks.append(bank); actions.append(action)
    gate = GateWeights(*(arr[f"gate_{key}"] for key in ("w1", "b1", "w2", "b2")))
    if receipt["base_law_kind"] == SequentialPolicy.law_kind:
        def factor(prefix, embedding=None):
            return FactorGate(*(arr[f"{prefix}_{key}"] for key in
                                ("w1", "b1", "w2", "b2")), embedding)
        base = SequentialPolicy(*banks, *actions, gate, factor("first"),
                                factor("second", arr["second_embedding"]),
                                order=receipt["order"],
                                first_checkpoint_sha256=receipt["first_checkpoint_sha256"])
    elif receipt["base_law_kind"] in (ProductPolicy.law_kind, LocalPolicy.law_kind):
        def factor(r):
            return FactorGate(*(arr[f"factor{r}_{key}"] for key in
                                ("w1", "b1", "w2", "b2")))
        cls = ProductPolicy if receipt["base_law_kind"] == ProductPolicy.law_kind else LocalPolicy
        base = cls(*banks, *actions, gate, factor(1), factor(2))
    elif receipt["base_law_kind"] == DeterministicJointPolicy.law_kind:
        base = DeterministicJointPolicy(*banks, *actions, gate)
    else:
        base = JointPolicy(*banks, *actions, gate)
    if kind == "identity":
        policy = IdentityPolicy(base)
    elif kind == "boundary_joint":
        policy = BoundaryJointPolicy(base, receipt["eta"])
    elif kind == "boundary_factor":
        policy = BoundaryFactorPolicy(base, receipt["eta1"], receipt["eta2"])
    elif kind in ("deterministic", "sequential"):
        policy = base
    else:
        raise ValueError("unregistered policy kind")
    if _kind(policy) != kind:
        raise ValueError("loaded policy kind differs")
    return policy, receipt
