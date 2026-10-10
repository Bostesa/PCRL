"""Independent HMAC, sticky cache and complete sampled A wire replay."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import sqlite3

import numpy as np

from .independent_a_source import sha_array, sha_file
from .independent_a_law import IndependentAPolicy, _bank_hash


STREAM = "SCV1-uniform-v1"
CACHE = "SCV1-joint-cache-v1"
WIRE = "SCV1-score-release-v1"


def _json(value, *, ascii_only: bool = True) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=ascii_only, allow_nan=False).encode("utf-8")


def _uniform(root: bytes, seed: int, role: str, draw: int, unit: int) -> float:
    if len(root) != 32 or seed not in (0, 1) or draw not in (0, 1, 2) or unit < 0:
        raise ValueError("A HMAC stream identity differs")
    message = json.dumps([STREAM, seed, role, draw, unit], separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    value = int.from_bytes(hmac.new(root, message, hashlib.sha256).digest()[:8], "big")
    return min((value + 0.5) / (1 << 64), np.nextafter(1.0, 0.0))


def _cache_key(root: bytes, receipt: dict, unit: int) -> str:
    payload = _json([CACHE, receipt["release_id"], receipt["source_seed"],
                     receipt["role"], receipt["realization"], unit], ascii_only=False)
    return hmac.new(root, payload, hashlib.sha256).hexdigest()


def _input_digest(score1: np.ndarray, score2: np.ndarray,
                  cell1: int, cell2: int, binding_sha: str) -> str:
    digest = hashlib.sha256(b"SCV1-input-v1\0")
    digest.update(np.ascontiguousarray(score1, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(score2, dtype="<f8").tobytes())
    digest.update(int(cell1).to_bytes(8, "little", signed=True))
    digest.update(int(cell2).to_bytes(8, "little", signed=True))
    digest.update(bytes.fromhex(binding_sha))
    return digest.hexdigest()


def _wire_hash(policy: IndependentAPolicy, receipt: dict,
               recipient: int, codes: np.ndarray) -> str:
    bank = policy.arrays[f"bank{recipient}_probability"]
    decision = policy.arrays[f"bank{recipient}_decision"]
    metadata = {"model_id": receipt["release_id"],
                "source_binding_sha256": receipt["source_binding_sha256"],
                "canonical_bank_sha256": receipt["canonical_bank_sha256_by_recipient"][f"r{recipient}"],
                "mechanism_sha256": receipt["mechanism_sha256"],
                "code_length_bits": (len(bank) - 1).bit_length(),
                "fallback": False, "sampling_contract": "cached_one_joint_draw_v1"}
    digest = hashlib.sha256()
    for code in codes:
        c = int(code)
        if not 0 <= c < len(bank):
            raise ValueError("A actual code outside canonical public bank")
        wire = {"schema": WIRE, "recipient": recipient, "code_id": c,
                "probability": bank[c].tolist(), "decision": int(decision[c]),
                "release_id": receipt["release_id"], "public_metadata": metadata}
        digest.update(_json(wire) + b"\n")
    return digest.hexdigest()


def _selected_root(private_path: Path, draw: int,
                   public_path: Path | None, expected_public_sha256: str | None) -> bytes:
    if public_path is None:
        # Only the data-free fixture uses an isolated raw 32-byte root.
        return Path(private_path).read_bytes()
    if expected_public_sha256 is None or sha_file(public_path) != expected_public_sha256:
        raise ValueError("A stream public root manifest hash differs")
    manifest = json.loads(Path(public_path).read_text())
    if (manifest.get("schema") != "scv1-public-stream-root-hashes-v1"
            or manifest.get("private_file_sha256") != sha_file(private_path)
            or manifest.get("private_file_mode") != "0600"
            or manifest.get("release_hmac_message") !=
               "compact ASCII JSON [SCV1-uniform-v1,source_seed,role,realization,unit]"
            or manifest.get("seq_fit_hmac_domain") != "SCV1-seq-fit-first-v1"):
        raise ValueError("A private HMAC root file differs from public commitment")
    private = json.loads(Path(private_path).read_text())
    if (private.get("schema") != "scv1-private-stream-roots-v1"
            or set(private.get("release_realization_roots_hex", {})) != {"0", "1", "2"}):
        raise ValueError("A private HMAC root bundle schema differs")
    roots = {}
    for name, hex_value in private["release_realization_roots_hex"].items():
        if (not isinstance(hex_value, str) or len(hex_value) != 64
                or any(c not in "0123456789abcdef" for c in hex_value)):
            raise ValueError("A HMAC root is not lowercase 32-byte hex")
        roots[name] = bytes.fromhex(hex_value)
    if ({name: hashlib.sha256(root).hexdigest() for name, root in roots.items()}
            != manifest["release_realization_root_sha256"]):
        raise ValueError("A HMAC roots differ from public hashes")
    seq_hex = private.get("seq_fit_root_hex")
    if (not isinstance(seq_hex, str) or len(seq_hex) != 64
            or any(c not in "0123456789abcdef" for c in seq_hex)
            or hashlib.sha256(bytes.fromhex(seq_hex)).hexdigest() !=
               manifest.get("seq_fit_root_sha256")
            or len(set(roots.values()) | {bytes.fromhex(seq_hex)}) != 4):
        raise ValueError("A fourth private stream commitment differs")
    return roots[str(draw)]


def replay_sampled_release(policy: IndependentAPolicy, *, directory: Path,
                           expected_receipt_sha256: str, root_path: Path,
                           public_stream_manifest_path: Path | None = None,
                           expected_stream_manifest_sha256: str | None = None,
                           row_id: np.ndarray, unit: np.ndarray,
                           score1: np.ndarray, score2: np.ndarray,
                           cell1: np.ndarray, cell2: np.ndarray,
                           expected_source_identity: dict) -> dict:
    """Recreate every emitted code and its cache/wire custody from source rows."""
    root_dir = Path(directory).resolve(strict=True)
    receipt_path = root_dir / "RELEASE_RECEIPT.json"
    cache_path = root_dir / "cache.sqlite3"
    code_path = root_dir / "release_codes.npz"
    if sha_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("A role realization receipt bytes differ")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "SCV1-private-realization-v1"
            or receipt.get("status") != "FROZEN_ONE_DRAW"
            or receipt["source_identity"] != expected_source_identity
            or receipt["mechanism_sha256"] != policy.receipt["mechanism_sha256"]
            or receipt["canonical_bank_sha256_by_recipient"] !=
               policy.receipt["bank_sha256_by_recipient"]
            or receipt["source_binding_sha256"] != policy.receipt["source_binding_sha256"]
            or receipt["source_seed"] != policy.receipt["source_seed"]
            or receipt["release_id"] != policy.receipt["release_id"]
            or sha_file(cache_path) != receipt["cache_file_sha256"]
            or sha_file(code_path) != receipt["codes_file_sha256"]):
        raise ValueError("A sampled release/policy binding differs")
    private_root = _selected_root(root_path, receipt["realization"],
                                  public_stream_manifest_path,
                                  expected_stream_manifest_sha256)
    if len(private_root) != 32 or hashlib.sha256(private_root).hexdigest() != receipt["private_stream_root_sha256"]:
        raise ValueError("A private HMAC root differs from precommitment")
    n = len(row_id)
    ids = np.asarray(row_id)
    groups = np.asarray(unit)
    x1, x2 = np.asarray(cell1), np.asarray(cell2)
    s1, s2 = np.asarray(score1), np.asarray(score2)
    if (ids.dtype != np.int64 or groups.dtype != np.int64
            or x1.dtype != np.int64 or x2.dtype != np.int64
            or s1.dtype != np.float64 or s2.dtype != np.float64
            or ids.shape != (n,) or groups.shape != (n,)
            or x1.shape != (n,) or x2.shape != (n,)
            or s1.shape != (n, 2) or s2.shape != (n, 6)
            or not np.all(np.diff(ids) > 0)
            or receipt["rows"] != n or receipt["groups"] != len(np.unique(groups))
            or receipt["row_unit_pair_sha256"] != sha_array(np.column_stack((ids, groups)))):
        raise ValueError("A sampled source rows or role order differ")
    with np.load(code_path, allow_pickle=False) as archive:
        if set(archive.files) != {"row_id", "unit", "code_r1", "code_r2"}:
            raise ValueError("A realized code NPZ members differ")
        saved = {name: np.array(archive[name], copy=True) for name in archive.files}
    if (not np.array_equal(saved["row_id"], ids)
            or not np.array_equal(saved["unit"], groups)
            or any(saved[key].dtype != np.int64 or saved[key].shape != (n,)
                   for key in ("code_r1", "code_r2"))):
        raise ValueError("A realized code/person order differs")
    uri = cache_path.as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        configuration = connection.execute("SELECT value FROM config WHERE id=1").fetchone()
        cached = connection.execute("SELECT key,input_digest,z1,z2 FROM releases").fetchall()
    finally:
        connection.close()
    bank_hashes = (_bank_hash(policy.arrays["bank1_probability"], policy.arrays["bank1_decision"]),
                   _bank_hash(policy.arrays["bank2_probability"], policy.arrays["bank2_decision"]))
    expected_config = {"schema": CACHE, "release_id": receipt["release_id"],
                       "source_seed": receipt["source_seed"], "role": receipt["role"],
                       "realization": receipt["realization"],
                       "source_binding_sha256": receipt["source_binding_sha256"],
                       "bank_sha256": list(bank_hashes),
                       "mechanism_sha256": receipt["mechanism_sha256"],
                       "private_root_sha256": receipt["private_stream_root_sha256"],
                       "offline": True}
    if configuration is None or json.loads(configuration[0]) != expected_config:
        raise ValueError("A sticky cache source/policy configuration differs")
    cache = {key: (digest, int(a), int(b)) for key, digest, a, b in cached}
    if len(cache) != len(cached) or len(cache) != receipt["groups"]:
        raise ValueError("A sticky cache does not have one entry per group")
    seen = {}
    for position in range(n):
        person = int(groups[position])
        input_digest = _input_digest(s1[position], s2[position], int(x1[position]),
                                     int(x2[position]), receipt["source_binding_sha256"])
        key = _cache_key(private_root, receipt, person)
        if key not in cache or cache[key][0] != input_digest:
            raise ValueError("A sticky cache key/input digest differs")
        uniform = _uniform(private_root, receipt["source_seed"], receipt["role"],
                           receipt["realization"], person)
        actual = policy.sample(s1[position], s2[position], int(x1[position]),
                               int(x2[position]), uniform)
        if (actual != cache[key][1:] or actual !=
                (int(saved["code_r1"][position]), int(saved["code_r2"][position]))):
            raise ValueError("A independently sampled pair differs from saved/cache pair")
        if person in seen and seen[person] != (input_digest, actual):
            raise ValueError("A duplicate unit changed its cached public output")
        seen[person] = (input_digest, actual)
    wire_hashes = {f"r{r}": _wire_hash(policy, receipt, r, saved[f"code_r{r}"])
                   for r in (1, 2)}
    if wire_hashes != receipt["wire_sha256_by_recipient"]:
        raise ValueError("A complete sampled recipient wire SHA differs")
    return {"rows": n, "groups": len(seen), "release_receipt_sha256": expected_receipt_sha256,
            "code_file_sha256": receipt["codes_file_sha256"],
            "cache_file_sha256": receipt["cache_file_sha256"],
            "wire_sha256_by_recipient": wire_hashes,
            "ordered_codes_exact": True, "hmac_cache_exact": True}
