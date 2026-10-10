"""Write-once selected matched-control policies for release and attack replay.

Per-person FIT draws and labels are never part of a policy package. A caller
must supply the expected receipt SHA from the trusted fit/selection registry.
"""

import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from bank import FixedBank
from channel import GateWeights
from policy_package import (_private_directory, _sha, _padded, _unpad,
                            load_policy_package)
from release import _bank_hash, _mechanism_hash
from source_bridge import action_map_sha256

from deterministic_policy import DeterministicJointPolicy
from factor_controls import FactorGate, ProductPolicy, LocalPolicy, SequentialPolicy


SCHEMA = "SCV1-selected-control-policy-v1"
COMMON = {"bank1_probability", "bank1_decision", "bank1_old_to_global",
          "bank2_probability", "bank2_decision", "bank2_old_to_global",
          "actions1", "actions2", "gate_w1", "gate_b1", "gate_w2", "gate_b2"}
FACTOR = ("w1", "b1", "w2", "b2")
HEX = set("0123456789abcdef")


def _hex64(value):
    return type(value) is str and len(value) == 64 and set(value) <= HEX


def _factor_arrays(arrays, prefix, factor):
    for name in FACTOR:
        arrays[f"{prefix}_{name}"] = getattr(factor, name)
    if factor.embedding is not None:
        arrays[f"{prefix}_embedding"] = factor.embedding


def _arrays(policy):
    arrays = {}
    for recipient, bank, actions in ((1, policy.bank1, policy.actions1),
                                     (2, policy.bank2, policy.actions2)):
        arrays[f"bank{recipient}_probability"] = bank.probability
        arrays[f"bank{recipient}_decision"] = bank.decision
        arrays[f"bank{recipient}_old_to_global"] = np.asarray(bank.old_to_global,
                                                               dtype=np.int64)
        arrays[f"actions{recipient}"] = _padded(actions)
    for name in FACTOR:
        arrays[f"gate_{name}"] = getattr(policy.gate, name)
    if type(policy) in (ProductPolicy, LocalPolicy):
        _factor_arrays(arrays, "factor1", policy.factor1)
        _factor_arrays(arrays, "factor2", policy.factor2)
    elif type(policy) is SequentialPolicy:
        _factor_arrays(arrays, "first", policy.first)
        _factor_arrays(arrays, "second", policy.second)
    return arrays


def _expected_keys(kind):
    if kind == DeterministicJointPolicy.law_kind:
        return COMMON
    if kind in (ProductPolicy.law_kind, LocalPolicy.law_kind):
        return COMMON | {f"{prefix}_{key}" for prefix in ("factor1", "factor2")
                         for key in FACTOR}
    if kind == SequentialPolicy.law_kind:
        return (COMMON | {f"{prefix}_{key}" for prefix in ("first", "second")
                          for key in FACTOR} | {"second_embedding"})
    raise ValueError("unregistered control law kind")


def save_control_policy_package(directory, policy, *, arm_id, release_id,
                                source_seed, source_binding_sha256,
                                row_unit_pair_sha256, bank_sha256, action_sha256,
                                fit_unit_id, start, update, fit_receipt_sha256):
    """Persist exactly the selected public mechanism under owner-only custody."""
    root = _private_directory(Path(directory))
    if (type(policy) not in (DeterministicJointPolicy, ProductPolicy,
                             LocalPolicy, SequentialPolicy) or
            type(arm_id) is not str or not arm_id or
            type(release_id) is not str or not release_id or
            type(source_seed) is not int or source_seed not in (0, 1) or
            type(fit_unit_id) is not str or not fit_unit_id or
            start not in ("identity_biased", "uniform", "uniform_logit") or
            type(update) is not int or update < 0 or
            len(bank_sha256) != 2 or len(action_sha256) != 2 or
            not all(_hex64(value) for value in
                    (source_binding_sha256, row_unit_pair_sha256,
                     fit_receipt_sha256, *bank_sha256, *action_sha256))):
        raise ValueError("selected control identity is incomplete")
    if ((_bank_hash(policy.bank1), _bank_hash(policy.bank2)) != tuple(bank_sha256) or
            (action_map_sha256(policy.actions1),
             action_map_sha256(policy.actions2)) != tuple(action_sha256)):
        raise ValueError("control bank/actions differ from admitted hashes")
    arrays = _arrays(policy)
    if set(arrays) != _expected_keys(policy.law_kind):
        raise ValueError("control parameter set differs from law kind")
    package = root / "policy.npz"
    receipt_path = root / "POLICY_RECEIPT.json"
    if package.exists() or receipt_path.exists():
        raise FileExistsError("selected control package is write-once")
    fd = os.open(package, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    receipt = {
        "schema": SCHEMA, "status": "SELECTED_FIT_POLICY",
        "arm_id": arm_id, "release_id": release_id, "source_seed": source_seed,
        "source_binding_sha256": source_binding_sha256,
        "row_unit_pair_sha256": row_unit_pair_sha256,
        "bank_sha256_by_recipient": {"r1": bank_sha256[0], "r2": bank_sha256[1]},
        "action_sha256_by_recipient": {"r1": action_sha256[0], "r2": action_sha256[1]},
        "law_kind": policy.law_kind,
        "mechanism_descriptor": policy.mechanism_descriptor(),
        "mechanism_sha256": _mechanism_hash(policy),
        "fit_unit_id": fit_unit_id, "selected_start": start,
        "selected_update": update, "fit_receipt_sha256": fit_receipt_sha256,
        "policy_file_sha256": _sha(package),
    }
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(receipt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return {**receipt, "receipt_sha256": hashlib.sha256(payload).hexdigest()}


def _restore_bank(arrays, receipt, recipient):
    p = arrays[f"bank{recipient}_probability"]
    d = arrays[f"bank{recipient}_decision"]
    old = arrays[f"bank{recipient}_old_to_global"]
    if (p.dtype != np.float64 or p.ndim != 2 or min(p.shape) < 1 or
            d.dtype != np.int64 or d.shape != (len(p),) or
            old.dtype != np.int64 or old.ndim != 1 or
            np.any(old < 0) or np.any(old >= len(p)) or
            not np.isfinite(p).all() or np.any(p < 0) or
            not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0) or
            not np.array_equal(np.argmax(p, axis=1), d)):
        raise ValueError("selected control public bank differs from fixed schema")
    bank = FixedBank(p, d, tuple(int(i) for i in old))
    actions = _unpad(arrays[f"actions{recipient}"])
    if (_bank_hash(bank) != receipt["bank_sha256_by_recipient"][f"r{recipient}"] or
            action_map_sha256(actions) !=
            receipt["action_sha256_by_recipient"][f"r{recipient}"]):
        raise ValueError("control bank/action hash differs")
    return bank, actions


def _restore_factor(arrays, prefix, *, embedding=False):
    values = [arrays[f"{prefix}_{name}"] for name in ("w1", "b1", "w2", "b2")]
    if embedding:
        values.append(arrays[f"{prefix}_embedding"])
    return FactorGate(*values)


def _load_control(directory, *, receipt_sha256):
    root = _private_directory(Path(directory))
    path, package = root / "POLICY_RECEIPT.json", root / "policy.npz"
    if not _hex64(receipt_sha256) or _sha(path) != receipt_sha256:
        raise ValueError("selected control receipt SHA differs")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != SCHEMA or
            receipt.get("status") != "SELECTED_FIT_POLICY" or
            _sha(package) != receipt.get("policy_file_sha256")):
        raise ValueError("selected control package identity differs")
    kind = receipt.get("law_kind")
    expected = _expected_keys(kind)
    with ZipFile(package) as archive:
        if {item.filename for item in archive.infolist()} != {f"{key}.npy" for key in expected}:
            raise ValueError("selected control NPZ members differ")
    with np.load(package, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in expected}
    bank1, actions1 = _restore_bank(arrays, receipt, 1)
    bank2, actions2 = _restore_bank(arrays, receipt, 2)
    gate = GateWeights(*(arrays[f"gate_{name}"] for name in
                         ("w1", "b1", "w2", "b2")))
    args = (bank1, bank2, actions1, actions2, gate)
    if kind == DeterministicJointPolicy.law_kind:
        policy = DeterministicJointPolicy(*args)
    elif kind == ProductPolicy.law_kind:
        policy = ProductPolicy(*args, _restore_factor(arrays, "factor1"),
                               _restore_factor(arrays, "factor2"))
    elif kind == LocalPolicy.law_kind:
        policy = LocalPolicy(*args, _restore_factor(arrays, "factor1"),
                             _restore_factor(arrays, "factor2"))
    else:
        descriptor = receipt.get("mechanism_descriptor")
        if type(descriptor) is not dict:
            raise ValueError("sequential policy descriptor missing")
        policy = SequentialPolicy(*args, _restore_factor(arrays, "first"),
                                  _restore_factor(arrays, "second", embedding=True),
                                  order=descriptor.get("order"),
                                  first_checkpoint_sha256=descriptor.get(
                                      "first_checkpoint_sha256"))
    if (policy.mechanism_descriptor() != receipt.get("mechanism_descriptor") or
            _mechanism_hash(policy) != receipt.get("mechanism_sha256")):
        raise ValueError("selected control law differs from receipt")
    return policy, receipt


def load_selected_policy_package(directory, *, receipt_sha256):
    """Dispatch either R12 stochastic or R13 matched-control package exactly."""
    root = _private_directory(Path(directory))
    receipt_path = root / "POLICY_RECEIPT.json"
    if not _hex64(receipt_sha256) or _sha(receipt_path) != receipt_sha256:
        raise ValueError("selected policy receipt SHA differs")
    schema = json.loads(receipt_path.read_text()).get("schema")
    if schema == "SCV1-selected-stochastic-policy-v1":
        return load_policy_package(root, receipt_sha256=receipt_sha256)
    if schema == SCHEMA:
        return _load_control(root, receipt_sha256=receipt_sha256)
    raise ValueError("unregistered selected policy package schema")
