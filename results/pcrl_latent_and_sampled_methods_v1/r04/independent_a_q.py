"""Independent public Q table custody and actual-code feature attachment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .independent_a_source import sha_file


ARRAYS = ("r1", "r2", "pair", "r1_mass", "r2_mass", "pair_mass")
ZERO_RULE = "all_zero_features_with_zero_support_v1"


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def load_independent_q_table(directory: Path, receipt_sha256: str,
                             expected_policy_receipt_sha256: str) -> tuple[dict, dict]:
    root = Path(directory).resolve(strict=True)
    receipt_path, archive_path = root / "Q_TABLE_RECEIPT.json", root / "q_table.npz"
    if sha_file(receipt_path) != receipt_sha256:
        raise ValueError("A public Q table receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "SCV1-q-prototype-table-artifact-v1"
            or receipt.get("status") != "FROZEN_PUBLIC_PROTOTYPE"
            or receipt.get("policy_receipt_sha256") != expected_policy_receipt_sha256
            or sha_file(archive_path) != receipt.get("q_table_file_sha256")):
        raise ValueError("A Q table policy or frozen file differs")
    with ZipFile(archive_path) as archive:
        if {member.filename for member in archive.infolist()} != {
                name + ".npy" for name in (*ARRAYS, "identity_utf8", "prototype_count")}:
            raise ValueError("A Q table NPZ members differ")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in ARRAYS}
        identity_bytes = np.array(archive["identity_utf8"], copy=True)
        count = np.array(archive["prototype_count"], copy=True)
    if (identity_bytes.dtype != np.uint8 or identity_bytes.ndim != 1
            or count.dtype != np.int64 or count.shape != (1,)):
        raise ValueError("A Q table identity/count arrays differ")
    identity = json.loads(identity_bytes.tobytes().decode("utf-8"))
    if (identity.get("schema") != "SCV1-q-uniform-prototype-v1"
            or identity.get("zero_support_rule") != ZERO_RULE
            or identity.get("prototype_count") != int(count[0])
            or identity.get("public_centroids_sha256") !=
               receipt["public_centroids_sha256"]):
        raise ValueError("A Q public prototype identity differs")
    digest = hashlib.sha256(_json(identity) + b"\n")
    for name in ARRAYS:
        array = arrays[name]
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("A Q table array not finite float64")
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(_json(list(canonical.shape)) + b"\n")
        digest.update(canonical.tobytes())
    if digest.hexdigest() != receipt["table_content_sha256"]:
        raise ValueError("A Q table content SHA differs")
    for view in ("r1", "r2", "pair"):
        feature, mass = arrays[view], arrays[view + "_mass"]
        if (feature.shape[:-1] != mass.shape or feature.shape[-1] != 19
                or np.any(mass < 0) or
                np.any(feature[mass == 0] != 0)):
            raise ValueError("A Q zero-support row or shape differs")
    return arrays, identity


def actual_q_features(table: dict, *, view: str, code1: np.ndarray,
                      code2: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    if view not in ("r1", "r2", "pair"):
        raise ValueError("A Q view unregistered")
    first = np.asarray(code1)
    second = None if code2 is None else np.asarray(code2)
    features, mass = table[view], table[view + "_mass"]
    if (first.dtype != np.int64 or first.ndim != 1 or
            np.any((first < 0) | (first >= mass.shape[0]))):
        raise ValueError("A actual first emitted code invalid for Q")
    if view == "pair":
        if (second is None or second.dtype != np.int64 or second.shape != first.shape
                or np.any((second < 0) | (second >= mass.shape[1]))):
            raise ValueError("A actual second emitted code invalid for pair Q")
        attached, support = features[first, second], mass[first, second]
    else:
        if second is not None:
            raise ValueError("A local Q view received a second code")
        attached, support = features[first], mass[first]
    if np.any(attached[support == 0] != 0):
        raise ValueError("A unsupported emitted code lacks all-zero Q row")
    return np.ascontiguousarray(attached), int(np.count_nonzero(support == 0))
