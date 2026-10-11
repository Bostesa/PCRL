"""Write exact deterministic score-reference wires without label access.

The caller must obtain its role arrays through a separately admitted source
worker. This pure writer keeps only actual public probabilities, decisions and
finite-bank IDs in its private role package; it never fabricates continuous IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array


SCHEMA = "SCV1-private-reference-release-v1"
WIRE_SCHEMA = "SCV1-reference-score-release-v1"
SOURCE_IDENTITY_KEYS = frozenset({"role_source_file_sha256",
                                  "reference_file_sha256",
                                  "bank_receipt_sha256",
                                  "source_audit_sha256"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def reconstruct_reference_wires(probability, decision, code, *, recipient: int,
                                interface_kind: str, release_id: str,
                                source_binding_sha256: str,
                                bank_sha256: str | None,
                                bank_size: int | None) -> tuple[dict, ...]:
    """Return exactly the recipient's public fields in sealed row order."""
    p, d, c = np.asarray(probability), np.asarray(decision), np.asarray(code)
    width = {1: 2, 2: 6}.get(recipient)
    if (width is None or interface_kind not in ("continuous", "finite") or
            type(release_id) is not str or not release_id or
            type(source_binding_sha256) is not str or
            HEX64.fullmatch(source_binding_sha256) is None or
            p.dtype != np.float64 or p.ndim != 2 or p.shape[1] != width or
            d.dtype != np.int64 or d.shape != (len(p),) or
            c.dtype != np.int64 or c.shape != (len(p),) or
            not len(p) or not np.isfinite(p).all() or np.any(p < 0) or
            np.max(np.abs(p.sum(axis=1) - 1)) > 1e-8 or
            not np.array_equal(np.argmax(p, axis=1), d)):
        raise ValueError("reference probability/decision wire differs")
    if interface_kind == "continuous":
        if (bank_sha256 is not None or bank_size is not None or
                not np.array_equal(c, np.full(len(c), -1, np.int64))):
            raise ValueError("continuous reference cannot publish a code")
        bits = None
    else:
        if (type(bank_sha256) is not str or HEX64.fullmatch(bank_sha256) is None or
                type(bank_size) is not int or bank_size < 1 or
                np.any((c < 0) | (c >= bank_size))):
            raise ValueError("finite reference code/bank identity differs")
        bits = (bank_size - 1).bit_length()
    metadata = {"model_id": release_id,
                "source_binding_sha256": source_binding_sha256,
                "bank_sha256": bank_sha256,
                "code_length_bits": bits}
    wires = []
    for index in range(len(p)):
        wire = {"schema": WIRE_SCHEMA, "recipient": recipient,
                "probability": p[index].tolist(), "decision": int(d[index]),
                "release_id": release_id, "public_metadata": dict(metadata)}
        if interface_kind == "finite":
            wire["code_id"] = int(c[index])
        wires.append(wire)
    return tuple(wires)


def materialize_reference_role(*, row_id, unit, prob1, prob2, decision1,
                               decision2, code1, code2, arm_id: str, seed: int,
                               role: str, release_id: str,
                               source_binding_sha256: str, source_identity: dict,
                               interface_kind: str,
                               bank_sha256_by_recipient: dict,
                               expected_row_unit_sha256: str,
                               output_dir, bank1=None, bank2=None) -> dict:
    """Persist a label-free ATT/INNER deterministic release, write once."""
    supplied = Path(output_dir)
    root = supplied.resolve(strict=False)
    if (any(part.is_symlink() for part in (supplied, *supplied.parents)) or
            ".private" not in root.parts or root.exists() or
            not root.parent.is_dir() or root.parent.is_symlink() or
            root.parent.stat().st_uid != os.getuid() or
            stat.S_IMODE(root.parent.stat().st_mode) & 0o077):
        raise PermissionError("reference output must be a new owner-only private directory")
    ids, groups = np.asarray(row_id), np.asarray(unit)
    if (ids.dtype != np.int64 or ids.ndim != 1 or not len(ids) or
            groups.dtype != np.int64 or groups.shape != ids.shape or
            np.any(np.diff(ids) <= 0) or
            type(expected_row_unit_sha256) is not str or
            sha_array(np.column_stack((ids, groups))) != expected_row_unit_sha256):
        raise ValueError("reference ordered people/groups differ")
    if (type(arm_id) is not str or not arm_id or seed not in (0, 1) or
            role not in ("ATTACK_FIT_NEW", "INNER_SELECTION") or
            type(source_identity) is not dict or
            source_identity.keys() != SOURCE_IDENTITY_KEYS or
            type(source_identity["reference_file_sha256"]) is not str or
            type(source_identity["source_audit_sha256"]) is not str or
            type(source_identity["role_source_file_sha256"]) is not str or
            any(type(value) is not str or HEX64.fullmatch(value) is None
                for value in source_identity.values() if value is not None) or
            bank_sha256_by_recipient.keys() != {"r1", "r2"}):
        raise ValueError("reference source identity is incomplete")
    if interface_kind == "continuous":
        if (bank1 is not None or bank2 is not None or
                source_identity["bank_receipt_sha256"] is not None):
            raise ValueError("continuous reference cannot claim a bank")
        banks = (None, None)
        sizes = (None, None)
    elif interface_kind == "finite":
        if source_identity["bank_receipt_sha256"] is None:
            raise ValueError("finite reference bank receipt is missing")
        banks = (np.asarray(bank1), np.asarray(bank2))
        sizes = tuple(len(bank) for bank in banks)
        for bank, width, key in zip(banks, (2, 6), ("r1", "r2")):
            if (bank.dtype != np.float64 or bank.ndim != 2 or
                    bank.shape[1] != width or not len(bank) or
                    not np.isfinite(bank).all() or np.any(bank < 0) or
                    np.max(np.abs(bank.sum(axis=1) - 1)) > 1e-8 or
                    sha_array(bank) != bank_sha256_by_recipient[key]):
                raise ValueError("reference full public bank differs")
    else:
        raise ValueError("reference interface kind differs")
    wires = {}
    for recipient, probability, decision, code, bank, size in (
            (1, prob1, decision1, code1, banks[0], sizes[0]),
            (2, prob2, decision2, code2, banks[1], sizes[1])):
        q, d, c = np.asarray(probability), np.asarray(decision), np.asarray(code)
        wires[f"r{recipient}"] = reconstruct_reference_wires(
            q, d, c, recipient=recipient, interface_kind=interface_kind,
            release_id=release_id, source_binding_sha256=source_binding_sha256,
            bank_sha256=bank_sha256_by_recipient[f"r{recipient}"],
            bank_size=size)
        if bank is not None and not np.array_equal(
                np.ascontiguousarray(q).view(np.uint8),
                np.ascontiguousarray(bank[c]).view(np.uint8)):
            raise ValueError("finite code does not emit its full-bank probability")
        first_by_unit = {}
        for index, group in enumerate(groups):
            emitted = (np.ascontiguousarray(q[index]).tobytes(),
                       int(d[index]), int(c[index]))
            first = first_by_unit.setdefault(int(group), emitted)
            if first != emitted:
                raise ValueError("deterministic reference differs within duplicate unit")
    root.mkdir(mode=0o700)
    codes = root / "reference_release.npz"
    fd = os.open(codes, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        np.savez_compressed(output, row_id=ids, unit=groups,
                            prob_r1=prob1, prob_r2=prob2,
                            decision_r1=decision1, decision_r2=decision2,
                            code_r1=code1, code_r2=code2)
        output.flush()
        os.fsync(output.fileno())
    wire_sha = {}
    for recipient in (1, 2):
        digest = hashlib.sha256()
        for wire in wires[f"r{recipient}"]:
            digest.update(_json(wire) + b"\n")
        wire_sha[f"r{recipient}"] = digest.hexdigest()
    receipt = {"schema": SCHEMA, "status": "FROZEN_DETERMINISTIC",
               "arm_id": arm_id, "source_seed": seed, "role": role,
               "release_id": release_id, "rows": int(len(ids)),
               "groups": int(len(np.unique(groups))),
               "row_unit_pair_sha256": expected_row_unit_sha256,
               "source_binding_sha256": source_binding_sha256,
               "source_identity": source_identity,
               "interface_kind": interface_kind,
               "bank_sha256_by_recipient": bank_sha256_by_recipient,
               "codes_file_sha256": _sha_file(codes),
               "wire_sha256_by_recipient": wire_sha}
    payload = _json(receipt) + b"\n"
    target = root / "REFERENCE_RELEASE_RECEIPT.json"
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return receipt
