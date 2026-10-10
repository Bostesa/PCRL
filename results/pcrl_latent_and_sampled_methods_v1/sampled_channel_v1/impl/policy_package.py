"""Write-once, hash-bound package for a selected stochastic score policy.

This module never loads Adult roles. It persists the public bank and mechanism
needed for exact sampled release and defense-aware attack replay.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from zipfile import ZipFile

import numpy as np

if __package__:
    from .bank import FixedBank
    from .channel import GateWeights, JointPolicy
    from .release import _bank_hash, _mechanism_hash
    from .source_bridge import action_map_sha256
else:
    from bank import FixedBank
    from channel import GateWeights, JointPolicy
    from release import _bank_hash, _mechanism_hash
    from source_bridge import action_map_sha256


SCHEMA = "SCV1-selected-stochastic-policy-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ARRAY_KEYS = frozenset({"bank1_probability", "bank1_decision", "bank1_old_to_global",
                        "bank2_probability", "bank2_decision", "bank2_old_to_global",
                        "actions1", "actions2", "w1", "b1", "w2", "b2"})


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _private_directory(path: Path) -> Path:
    root = path.resolve(strict=True)
    if (path.is_symlink() or not root.is_dir() or
            stat.S_IMODE(root.stat().st_mode) & 0o077):
        raise PermissionError("policy package needs an owner-only real directory")
    return root


def _padded(actions) -> np.ndarray:
    out = np.full((len(actions), 4), -1, dtype=np.int64)
    for index, row in enumerate(actions):
        out[index, :len(row)] = row
    return out


def _unpad(value: np.ndarray):
    if value.dtype != np.int64 or value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("action package must be int64 fine-cell by four")
    rows = []
    for row in value:
        used = row[row >= 0]
        if (len(used) < 1 or len(set(used.tolist())) != len(used) or
                not np.array_equal(row, np.r_[used, np.full(4 - len(used), -1)])):
            raise ValueError("action package has noncanonical padding")
        rows.append(tuple(int(i) for i in used))
    return tuple(rows)


def save_policy_package(directory, policy: JointPolicy, *, arm_id: str,
                        release_id: str,
                        source_seed: int, source_binding_sha256: str,
                        row_unit_pair_sha256: str, bank_sha256: tuple[str, str],
                        action_sha256: tuple[str, str], fit_unit_id: str,
                        start: str, update: int, fit_receipt_sha256: str) -> dict:
    """Persist a selected policy only; caller owns fit/admission evidence."""
    root = _private_directory(Path(directory))
    if (type(policy) is not JointPolicy or policy.law_kind != "stochastic_joint_v1" or
            type(arm_id) is not str or not arm_id or
            type(release_id) is not str or not release_id or
            type(source_seed) is not int or source_seed not in (0, 1) or
            type(fit_unit_id) is not str or not fit_unit_id or
            start not in ("identity_biased", "uniform_logit") or
            type(update) is not int or update < 0 or
            len(bank_sha256) != 2 or len(action_sha256) != 2 or
            any(type(value) is not str or HEX64.fullmatch(value) is None
                for value in (source_binding_sha256, row_unit_pair_sha256,
                              fit_receipt_sha256, *bank_sha256, *action_sha256))):
        raise ValueError("selected policy identity is incomplete")
    actual_bank = (_bank_hash(policy.bank1), _bank_hash(policy.bank2))
    actual_action = (action_map_sha256(policy.actions1),
                     action_map_sha256(policy.actions2))
    if actual_bank != tuple(bank_sha256) or actual_action != tuple(action_sha256):
        raise ValueError("policy bank/action bytes differ from admitted identity")
    package = root / "policy.npz"
    receipt_path = root / "POLICY_RECEIPT.json"
    if package.exists() or receipt_path.exists():
        raise FileExistsError("selected policy package is write-once")
    arrays = {}
    for recipient, bank, actions in ((1, policy.bank1, policy.actions1),
                                     (2, policy.bank2, policy.actions2)):
        arrays[f"bank{recipient}_probability"] = bank.probability
        arrays[f"bank{recipient}_decision"] = bank.decision
        arrays[f"bank{recipient}_old_to_global"] = np.array(bank.old_to_global,
                                                               dtype=np.int64)
        arrays[f"actions{recipient}"] = _padded(actions)
    arrays.update({name: getattr(policy.gate, name)
                   for name in ("w1", "b1", "w2", "b2")})
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
        "mechanism_sha256": _mechanism_hash(policy),
        "law_kind": policy.law_kind, "fit_unit_id": fit_unit_id,
        "selected_start": start, "selected_update": update,
        "fit_receipt_sha256": fit_receipt_sha256,
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


def load_policy_package(directory, *, receipt_sha256: str) -> tuple[JointPolicy, dict]:
    """Restore exact public law after checking sealed receipt and package bytes."""
    root = _private_directory(Path(directory))
    receipt_path, package = root / "POLICY_RECEIPT.json", root / "policy.npz"
    if (type(receipt_sha256) is not str or HEX64.fullmatch(receipt_sha256) is None or
            _sha(receipt_path) != receipt_sha256):
        raise ValueError("selected policy receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != SCHEMA or
            receipt.get("status") != "SELECTED_FIT_POLICY" or
            receipt.get("law_kind") != "stochastic_joint_v1" or
            type(receipt.get("arm_id")) is not str or not receipt["arm_id"] or
            _sha(package) != receipt.get("policy_file_sha256")):
        raise ValueError("selected policy receipt or package bytes differ")
    with ZipFile(package) as archive:
        if {item.filename for item in archive.infolist()} != {f"{key}.npy" for key in ARRAY_KEYS}:
            raise ValueError("selected policy NPZ members differ")
    with np.load(package, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in ARRAY_KEYS}
    banks, actions = [], []
    for r in (1, 2):
        probability = arrays[f"bank{r}_probability"]
        decision = arrays[f"bank{r}_decision"]
        old = arrays[f"bank{r}_old_to_global"]
        if (probability.dtype != np.float64 or decision.dtype != np.int64 or
                old.dtype != np.int64 or probability.ndim != 2 or
                decision.shape != (len(probability),) or old.ndim != 1 or
                np.any((old < 0) | (old >= len(probability))) or
                not np.isfinite(probability).all() or np.any(probability < 0) or
                not np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0) or
                not np.array_equal(np.argmax(probability, axis=1), decision)):
            raise ValueError("selected policy bank schema differs")
        bank = FixedBank(probability, decision, tuple(int(i) for i in old))
        rows = _unpad(arrays[f"actions{r}"])
        if (_bank_hash(bank) != receipt["bank_sha256_by_recipient"][f"r{r}"] or
                action_map_sha256(rows) != receipt["action_sha256_by_recipient"][f"r{r}"]):
            raise ValueError("selected policy bank/action binding differs")
        banks.append(bank)
        actions.append(rows)
    policy = JointPolicy(*banks, *actions,
                         GateWeights(*(arrays[name] for name in ("w1", "b1", "w2", "b2"))))
    if _mechanism_hash(policy) != receipt["mechanism_sha256"]:
        raise ValueError("selected policy mechanism binding differs")
    return policy, receipt
