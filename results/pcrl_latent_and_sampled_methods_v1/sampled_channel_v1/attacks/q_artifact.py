"""Write-once public-prototype Q table custody for the sampled attack slate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .q_aware import PrototypeFeatureTable, table_content_sha256


SCHEMA = "SCV1-q-prototype-table-artifact-v1"
ARRAYS = ("r1", "r2", "pair", "r1_mass", "r2_mass", "pair_mass")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _private_root(path: str | Path) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise PermissionError("Q table directory cannot be a symlink")
    root = supplied.resolve(strict=True)
    if (".private" not in root.parts or not root.is_dir() or
            stat.S_IMODE(root.stat().st_mode) & 0o077):
        raise PermissionError("Q table requires an owner-only private directory")
    return root


def save_q_table_artifact(
    table: PrototypeFeatureTable, directory: str | Path, *,
    policy_receipt_sha256: str,
) -> dict:
    """Persist the fixed postfit table and source policy link without labels."""
    root = _private_root(directory)
    if type(policy_receipt_sha256) is not str or HEX64.fullmatch(policy_receipt_sha256) is None:
        raise ValueError("Q table requires an exact selected policy receipt SHA")
    content_sha = table_content_sha256(table)
    if content_sha != table.table_sha256:
        raise ValueError("public Q table content differs from its identity")
    data_path = root / "q_table.npz"
    receipt_path = root / "Q_TABLE_RECEIPT.json"
    if data_path.exists() or receipt_path.exists():
        raise FileExistsError("public Q table custody is write-once")
    identity_bytes = json.dumps(table.identity, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode()
    fd = os.open(data_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        np.savez_compressed(
            output, **{name: getattr(table, name) for name in ARRAYS},
            identity_utf8=np.frombuffer(identity_bytes, dtype=np.uint8),
            prototype_count=np.asarray([table.prototype_count], dtype=np.int64))
        output.flush()
        os.fsync(output.fileno())
    receipt = {"schema": SCHEMA, "status": "FROZEN_PUBLIC_PROTOTYPE",
               "policy_receipt_sha256": policy_receipt_sha256,
               "table_content_sha256": content_sha,
               "public_centroids_sha256": table.identity["public_centroids_sha256"],
               "q_table_file_sha256": _sha(data_path)}
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(receipt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return {**receipt, "receipt_sha256": hashlib.sha256(payload).hexdigest()}


def load_q_table_artifact(
    directory: str | Path, *, expected_receipt_sha256: str,
    expected_content_sha256: str, expected_policy_receipt_sha256: str,
) -> PrototypeFeatureTable:
    """Read only the exact table sealed by a postfit source/queue receipt."""
    root = _private_root(directory)
    data_path, receipt_path = root / "q_table.npz", root / "Q_TABLE_RECEIPT.json"
    for path in (data_path, receipt_path):
        if (path.is_symlink() or not path.is_file() or
                stat.S_IMODE(path.stat().st_mode) & 0o077):
            raise PermissionError("Q table artifact is not a private regular file")
    if (type(expected_receipt_sha256) is not str or
            HEX64.fullmatch(expected_receipt_sha256) is None or
            _sha(receipt_path) != expected_receipt_sha256):
        raise ValueError("Q table receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    if (type(receipt) is not dict or receipt.keys() != {
            "schema", "status", "policy_receipt_sha256", "table_content_sha256",
            "public_centroids_sha256", "q_table_file_sha256"} or
            receipt["schema"] != SCHEMA or
            receipt["status"] != "FROZEN_PUBLIC_PROTOTYPE" or
            receipt["policy_receipt_sha256"] != expected_policy_receipt_sha256 or
            receipt["table_content_sha256"] != expected_content_sha256 or
            _sha(data_path) != receipt["q_table_file_sha256"]):
        raise ValueError("Q table source policy, content or file SHA differs")
    with ZipFile(data_path) as archive:
        members = archive.infolist()
        if (len(members) != len(ARRAYS) + 2 or
                {item.filename for item in members} !=
                {f"{name}.npy" for name in (*ARRAYS, "identity_utf8", "prototype_count")}):
            raise ValueError("Q table NPZ member set differs")
    with np.load(data_path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in ARRAYS}
        identity_bytes = np.array(archive["identity_utf8"], copy=True)
        count = np.array(archive["prototype_count"], copy=True)
    if (identity_bytes.dtype != np.uint8 or identity_bytes.ndim != 1 or
            count.dtype != np.int64 or count.shape != (1,) or count[0] < 1):
        raise ValueError("Q table identity/count array schema differs")
    identity = json.loads(identity_bytes.tobytes().decode("utf-8"))
    if identity.get("public_centroids_sha256") != receipt["public_centroids_sha256"]:
        raise ValueError("Q table public centroid source differs")
    table = PrototypeFeatureTable(**values, prototype_count=int(count[0]),
                                  identity=identity,
                                  table_sha256=receipt["table_content_sha256"])
    if table_content_sha256(table) != expected_content_sha256:
        raise ValueError("Q table reconstructed content SHA differs")
    return table
