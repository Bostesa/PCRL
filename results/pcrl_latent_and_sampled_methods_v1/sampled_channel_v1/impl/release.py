"""One cached joint draw and recipient-local seven-field sampled score wire.

The provider owns the stable private person key and both source scores. This
module never puts either on the public wire. Production draws from the OS
CSPRNG; offline development realizations use a private HMAC root for replay.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat

import numpy as np

if __package__:
    from .channel import JointPolicy
else:
    from channel import JointPolicy


WIRE_SCHEMA = "SCV1-score-release-v1"
CACHE_SCHEMA = "SCV1-joint-cache-v1"
STREAM_SCHEMA = "SCV1-uniform-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TWO64 = 1 << 64


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _secret(root) -> bytes:
    if not isinstance(root, bytes) or len(root) != 32:
        raise ValueError("private HMAC/cache root must be 256 bits")
    return root


def _unit(value):
    if type(value) in (int, np.int64) and value >= 0:
        return int(value)
    if type(value) is str and value:
        return value
    raise ValueError("provider must supply a stable private person/unit key")


def _uniform_from_uint64(value: int) -> float:
    # Float64 rounds some extreme 64-bit fractions to 1.0; stay in [0,1).
    return min((value + 0.5) / TWO64, np.nextafter(1.0, 0.0))


def offline_uniform(private_root: bytes, *, source_seed: int, role: str,
                    realization: int, unit) -> float:
    """One HMAC-SHA256 uniform for a complete cached pair, domain separated."""
    root = _secret(private_root)
    if (type(source_seed) is not int or source_seed < 0 or
            type(realization) is not int or realization < 0 or
            type(role) is not str or not role):
        raise ValueError("offline stream identity is incomplete")
    message = json.dumps([STREAM_SCHEMA, source_seed, role, realization,
                          _unit(unit)], separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    digest = hmac.new(root, message, hashlib.sha256).digest()
    return _uniform_from_uint64(int.from_bytes(digest[:8], "big"))


def draw_pair(triples, uniform: float) -> tuple[int, int]:
    """Right-sided inverse CDF over sorted actual global output-pair IDs."""
    rows = tuple(triples)
    if (not rows or not np.isfinite(uniform) or uniform < 0 or uniform >= 1 or
            any(type(i) not in (int, np.int64) or
                type(j) not in (int, np.int64) or i < 0 or j < 0 or
                not np.isfinite(p) or p < 0 for i, j, p in rows) or
            tuple((i, j) for i, j, _ in rows) !=
            tuple(sorted(set((i, j) for i, j, _ in rows))) or
            not np.isclose(sum(p for _, _, p in rows), 1, atol=1e-12, rtol=0)):
        raise ValueError("sampled law needs sorted unique public pairs and unit mass")
    mass = np.array([p for _, _, p in rows], dtype=np.float64)
    index = int(np.searchsorted(np.cumsum(mass), uniform, side="right"))
    if index >= len(rows):
        index = int(np.flatnonzero(mass > 0)[-1])
    return int(rows[index][0]), int(rows[index][1])


def _bank_hash(bank) -> str:
    h = hashlib.sha256(b"SCV1-canonical-bank-v1\n")
    probability = np.asarray(bank.probability, dtype="<f8")
    decision = np.asarray(bank.decision, dtype="<i8")
    h.update(_json(list(probability.shape)).encode() + b"\n")
    h.update(probability.tobytes())
    h.update(decision.tobytes())
    return h.hexdigest()


def _mechanism_hash(policy: JointPolicy) -> str:
    if type(policy.law_kind) is not str or not policy.law_kind:
        raise ValueError("public policy law kind is missing")
    descriptor = policy.mechanism_descriptor()
    arrays = policy.mechanism_arrays()
    if (type(descriptor) is not dict or descriptor.get("law_kind") != policy.law_kind or
            type(arrays) is not tuple or len(arrays) < 4):
        raise ValueError("public policy descriptor or parameter list differs")
    h = hashlib.sha256(b"SCV1-joint-mechanism-v1\n")
    h.update(_json({"policy": descriptor, "parameter_count": len(arrays),
                    "bank1": _bank_hash(policy.bank1),
                    "bank2": _bank_hash(policy.bank2),
                    "actions1": policy.actions1,
                    "actions2": policy.actions2}).encode() + b"\n")
    for value in arrays:
        source = np.asarray(value)
        if source.dtype != np.dtype("float64") or not np.isfinite(source).all():
            raise ValueError("public policy parameter must be finite float64")
        array = np.ascontiguousarray(source, dtype="<f8")
        h.update(_json(list(array.shape)).encode() + b"\n")
        h.update(array.tobytes())
    return h.hexdigest()


class CachedJointRelease:
    """Durable one-draw paired service, with one public view per recipient."""

    def __init__(self, policy: JointPolicy, *, cache_path, release_id: str,
                 source_seed: int, role: str, realization: int,
                 source_binding_sha256: str, private_root: bytes,
                 offline: bool = False):
        if not isinstance(policy, JointPolicy):
            raise TypeError("service requires a frozen joint policy")
        if (type(release_id) is not str or not release_id or
                type(source_seed) is not int or source_seed < 0 or
                type(role) is not str or not role or
                type(realization) is not int or realization < 0 or
                not isinstance(source_binding_sha256, str) or
                HEX64.fullmatch(source_binding_sha256) is None or
                type(offline) is not bool):
            raise ValueError("service version/source identity is incomplete")
        self.policy = policy
        self.cache_path = Path(cache_path)
        self.release_id = release_id
        self.source_seed = source_seed
        self.role = role
        self.realization = realization
        self.source_binding_sha256 = source_binding_sha256
        self.private_root = _secret(private_root)
        self.offline = offline
        self.bank_sha256 = (_bank_hash(policy.bank1), _bank_hash(policy.bank2))
        self.mechanism_sha256 = _mechanism_hash(policy)
        self._config = _json({
            "schema": CACHE_SCHEMA, "release_id": release_id,
            "source_seed": source_seed, "role": role,
            "realization": realization, "source_binding_sha256": source_binding_sha256,
            "bank_sha256": self.bank_sha256,
            "mechanism_sha256": self.mechanism_sha256,
            "private_root_sha256": hashlib.sha256(self.private_root).hexdigest(),
            "offline": offline,
        })
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.cache_path, timeout=5)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        parent = self.cache_path.parent
        if (not parent.is_dir() or parent.is_symlink() or
                stat.S_IMODE(parent.stat().st_mode) & 0o077):
            raise ValueError("private cache parent must be a private real directory")
        if self.cache_path.is_symlink():
            raise ValueError("sample cache path must not be a symlink")
        try:
            fd = os.open(self.cache_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
        if self.cache_path.is_symlink() or not self.cache_path.is_file():
            raise ValueError("sample cache must be a regular file")
        if stat.S_IMODE(self.cache_path.stat().st_mode) & 0o077:
            raise PermissionError("sample cache has nonprivate permissions")
        with self._transaction() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS releases (key TEXT PRIMARY KEY, input_digest TEXT NOT NULL, z1 INTEGER NOT NULL, z2 INTEGER NOT NULL)")
            row = connection.execute("SELECT value FROM config WHERE id=1").fetchone()
            if row is None:
                connection.execute("INSERT INTO config(id,value) VALUES (1,?)", (self._config,))
            elif row[0] != self._config:
                raise ValueError("cached service source, policy or private root differs")

    def _cache_key(self, unit) -> str:
        message = _json([CACHE_SCHEMA, self.release_id, self.source_seed,
                         self.role, self.realization, _unit(unit)]).encode("utf-8")
        return hmac.new(self.private_root, message, hashlib.sha256).hexdigest()

    def _input_digest(self, score1, score2, cell1, cell2) -> str:
        # Validate via the policy before bytes become a cache identity.
        self.policy.sparse_law(score1, score2, cell1, cell2)
        h = hashlib.sha256(b"SCV1-input-v1\0")
        for value in (score1, score2):
            array = np.ascontiguousarray(value, dtype="<f8")
            h.update(array.tobytes())
        for cell in (cell1, cell2):
            h.update(int(cell).to_bytes(8, "little", signed=True))
        h.update(bytes.fromhex(self.source_binding_sha256))
        return h.hexdigest()

    def _cached_pair(self, *, unit, score1, score2, cell1, cell2) -> tuple[int, int]:
        key = self._cache_key(unit)
        digest = self._input_digest(score1, score2, cell1, cell2)
        with self._transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT input_digest,z1,z2 FROM releases WHERE key=?",
                                     (key,)).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise ValueError("stable person key was reused with different source input")
                pair = (int(row[1]), int(row[2]))
                law = self.policy.sparse_law(score1, score2, cell1, cell2)
                if not any((i, j) == pair and mass > 0 for i, j, mass in law):
                    raise ValueError("cached public pair is outside the sealed positive-support law")
                return pair
            law = self.policy.sparse_law(score1, score2, cell1, cell2)
            if self.offline:
                uniform = offline_uniform(self.private_root, source_seed=self.source_seed,
                                          role=self.role, realization=self.realization,
                                          unit=unit)
            else:
                uniform = _uniform_from_uint64(secrets.randbits(64))
            z1, z2 = self.policy.sample_public_pair(score1, score2, cell1, cell2,
                                                    uniform)
            if not any((i, j) == (z1, z2) and mass > 0
                       for i, j, mass in law):
                raise ValueError("policy sampler output is outside its public law")
            connection.execute("INSERT INTO releases(key,input_digest,z1,z2) VALUES (?,?,?,?)",
                               (key, digest, z1, z2))
            return z1, z2

    def recipient(self, *, recipient: int, unit, score1, score2, cell1, cell2) -> dict:
        if type(recipient) is not int or recipient not in (1, 2):
            raise ValueError("recipient must be 1 or 2")
        z1, z2 = self._cached_pair(unit=unit, score1=score1, score2=score2,
                                   cell1=cell1, cell2=cell2)
        bank = self.policy.bank1 if recipient == 1 else self.policy.bank2
        code = z1 if recipient == 1 else z2
        metadata = {
            "model_id": self.release_id,
            "source_binding_sha256": self.source_binding_sha256,
            "canonical_bank_sha256": self.bank_sha256[recipient - 1],
            "mechanism_sha256": self.mechanism_sha256,
            "code_length_bits": (len(bank.probability) - 1).bit_length(),
            "fallback": False,
            "sampling_contract": "cached_one_joint_draw_v1",
        }
        return {"schema": WIRE_SCHEMA, "recipient": recipient,
                "code_id": code, "probability": bank.probability[code].tolist(),
                "decision": int(bank.decision[code]), "release_id": self.release_id,
                "public_metadata": metadata}
