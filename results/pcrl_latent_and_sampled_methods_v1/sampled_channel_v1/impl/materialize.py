"""Write-once private code arrays for one cached joint realization.

The trusted public bank reconstructs complete seven-field recipient wires.
No raw scores, source cells, labels, person IDs or uniforms enter those wires.
This module is pure custody; an R03/R00 gate must run before real role reads.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

if __package__:
    from .release import CachedJointRelease, WIRE_SCHEMA
    from .source_bridge import sha_array
else:
    from release import CachedJointRelease, WIRE_SCHEMA
    from source_bridge import sha_array


SCHEMA = "SCV1-private-realization-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_IDENTITY_KEYS = frozenset({"tstar_bank_admission_receipt_sha256",
                                  "tstar_role_tokens_file_sha256",
                                  "reference_file_sha256"})


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _write_once(path: Path, payload: bytes) -> str:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return hashlib.sha256(payload).hexdigest()


def reconstruct_wire_from_codes(service: CachedJointRelease, code_ids,
                                *, recipient: int) -> tuple[dict, ...]:
    """Reconstruct actual transmitted fields from sealed bank and code IDs."""
    if not isinstance(service, CachedJointRelease) or recipient not in (1, 2):
        raise ValueError("trusted paired service and recipient are required")
    ids = np.asarray(code_ids)
    bank = service.policy.bank1 if recipient == 1 else service.policy.bank2
    if (ids.dtype != np.int64 or ids.ndim != 1 or
            np.any((ids < 0) | (ids >= len(bank.probability)))):
        raise ValueError("saved public code IDs differ from sealed bank")
    metadata = {
        "model_id": service.release_id,
        "source_binding_sha256": service.source_binding_sha256,
        "canonical_bank_sha256": service.bank_sha256[recipient - 1],
        "mechanism_sha256": service.mechanism_sha256,
        "code_length_bits": (len(bank.probability) - 1).bit_length(),
        "fallback": False,
        "sampling_contract": "cached_one_joint_draw_v1",
    }
    return tuple({"schema": WIRE_SCHEMA, "recipient": recipient,
                  "code_id": int(code),
                  "probability": bank.probability[code].tolist(),
                  "decision": int(bank.decision[code]),
                  "release_id": service.release_id,
                  "public_metadata": dict(metadata)} for code in ids)


def materialize_role(service: CachedJointRelease, *, row_id, unit,
                     score1, score2, cell1, cell2, output_dir,
                     expected_row_unit_sha256: str,
                     source_identity: dict[str, str]) -> dict:
    """Persist one actual cached joint draw in exact admitted role order."""
    if not isinstance(service, CachedJointRelease):
        raise TypeError("materialization requires one sealed paired service")
    root = Path(output_dir)
    if (not root.is_dir() or root.is_symlink() or
            stat.S_IMODE(root.stat().st_mode) & 0o077 or
            service.cache_path.parent.resolve() != root.resolve()):
        raise PermissionError("one private owner-only directory must own the cache")
    codes_path = root / "release_codes.npz"
    receipt_path = root / "RELEASE_RECEIPT.json"
    if codes_path.exists() or receipt_path.exists():
        raise FileExistsError("a release realization is write-once")
    if (type(source_identity) is not dict or
            source_identity.keys() != SOURCE_IDENTITY_KEYS or
            any(type(value) is not str or HEX64.fullmatch(value) is None
                for value in source_identity.values())):
        raise ValueError("source role/reference identities are incomplete")
    ids, groups = np.asarray(row_id), np.asarray(unit)
    a, b = np.asarray(score1), np.asarray(score2)
    x, y = np.asarray(cell1), np.asarray(cell2)
    n = len(ids)
    if (n < 1 or ids.dtype != np.int64 or groups.dtype != np.int64 or
            x.dtype != np.int64 or y.dtype != np.int64 or
            a.dtype != np.float64 or b.dtype != np.float64 or
            groups.shape != (n,) or x.shape != (n,) or y.shape != (n,) or
            a.shape != (n, service.policy.bank1.probability.shape[1]) or
            b.shape != (n, service.policy.bank2.probability.shape[1]) or
            not np.all(np.diff(ids) > 0) or
            sha_array(np.column_stack((ids, groups))) != expected_row_unit_sha256):
        raise ValueError("sampled role row/source identity differs from sealed input")
    c1, c2 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for index in range(n):
        request = dict(unit=int(groups[index]), score1=a[index], score2=b[index],
                       cell1=int(x[index]), cell2=int(y[index]))
        c1[index] = service.recipient(recipient=1, **request)["code_id"]
        c2[index] = service.recipient(recipient=2, **request)["code_id"]
    fd = os.open(codes_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        np.savez_compressed(output, row_id=ids, unit=groups,
                            code_r1=c1, code_r2=c2)
        output.flush()
        os.fsync(output.fileno())
    wire_hashes = {}
    for recipient, codes in ((1, c1), (2, c2)):
        digest = hashlib.sha256()
        for wire in reconstruct_wire_from_codes(service, codes, recipient=recipient):
            digest.update(_json(wire) + b"\n")
        wire_hashes[f"r{recipient}"] = digest.hexdigest()
    receipt = {
        "schema": SCHEMA, "status": "FROZEN_ONE_DRAW",
        "release_id": service.release_id,
        "source_seed": service.source_seed, "role": service.role,
        "realization": service.realization, "rows": n,
        "groups": int(len(np.unique(groups))),
        "row_unit_pair_sha256": expected_row_unit_sha256,
        "source_binding_sha256": service.source_binding_sha256,
        "private_stream_root_sha256": hashlib.sha256(service.private_root).hexdigest(),
        "source_identity": source_identity,
        "canonical_bank_sha256_by_recipient": {
            "r1": service.bank_sha256[0], "r2": service.bank_sha256[1]},
        "mechanism_sha256": service.mechanism_sha256,
        "cache_file_sha256": sha_file(service.cache_path),
        "codes_file_sha256": sha_file(codes_path),
        "wire_sha256_by_recipient": wire_hashes,
    }
    _write_once(receipt_path, _json(receipt) + b"\n")
    return receipt
