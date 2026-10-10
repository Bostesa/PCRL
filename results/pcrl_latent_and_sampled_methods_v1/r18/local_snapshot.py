"""Write-once local byte custody with fresh uncached hash reads.

This module never decompresses a role archive. A separate R03 unit owns any
numerical replay of the restored files.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-z0-9_.-]+\Z")
FIXED_A_NAMES = frozenset({
    "original_lock.json", "a_producer_amendment.json", "repair_manifest.json",
    "portfolio_manifest.json", "r01_source_audit.json",
    "bank_admission_receipt.json", "bank_r1.npz", "bank_r2.npz",
    "tokens_att.npz", "tokens_inner.npz", "references_att.npz",
    "references_inner.npz", "att_release.npz", "att_release_receipt.json",
    "inner_release.npz", "inner_release_receipt.json",
    "att_queue_receipt.json", "inner_queue_receipt.json",
})


def _plain_path(path: Path) -> None:
    candidate = path.absolute()
    if any(piece.is_symlink() for piece in (candidate, *candidate.parents)):
        raise PermissionError("snapshot path has a symlink ancestor")


def _owner_dir(path: Path) -> None:
    _plain_path(path)
    info = path.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or
            stat.S_IMODE(info.st_mode) & 0o077):
        raise PermissionError("snapshot parent must be owner-only")


def _source_file(path: Path) -> None:
    _plain_path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError("snapshot source must be an owned plain file")
    if ".private" in path.parts and stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("snapshot private source must be owner-only")


def _uncached_read(path: Path):
    if not hasattr(fcntl, "F_NOCACHE"):
        raise OSError("uncached file reads are unavailable on this host")
    _source_file(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk
    finally:
        os.close(fd)


def uncached_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in _uncached_read(path):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_exact(source: Path, target: Path, expected_sha: str) -> int:
    _owner_dir(target.parent)
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            for chunk in _uncached_read(source):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(fd)
    digest, size = uncached_sha256(target)
    if digest != expected_sha or stat.S_IMODE(target.stat().st_mode) != 0o600:
        raise ValueError("snapshot copy hash or permission differs")
    _fsync_directory(target.parent)
    return size


def _write_once_json(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(fd)
    digest, size = uncached_sha256(path)
    if digest != hashlib.sha256(payload).hexdigest() or size != len(payload):
        raise ValueError("snapshot JSON hash differs after write")
    _fsync_directory(path.parent)
    return digest


def copy_and_restore(entries: dict, version_dir: Path, *,
                     required_names: set[str] | frozenset[str] = FIXED_A_NAMES) -> dict:
    """Hash preflight all source bytes, then create one snapshot and fresh restore."""
    if type(entries) is not dict or set(entries) != set(required_names) or not entries:
        raise ValueError("snapshot inventory names differ")
    version = Path(version_dir).absolute()
    _owner_dir(version.parent)
    _plain_path(version)
    if version.exists():
        raise FileExistsError("snapshot version already exists")
    sources = {}
    members = {}
    for name in sorted(entries):
        row = entries[name]
        if (NAME.fullmatch(name) is None or type(row) is not dict or
                row.keys() != {"source_path", "sha256"} or
                type(row["source_path"]) is not str or
                type(row["sha256"]) is not str or
                HEX64.fullmatch(row["sha256"]) is None):
            raise ValueError("snapshot inventory record differs")
        source = Path(row["source_path"]).absolute()
        digest, size = uncached_sha256(source)
        if digest != row["sha256"]:
            raise ValueError("snapshot source hash differs")
        sources[name] = source
        members[name] = {"sha256": digest, "bytes": size}
    version.mkdir(mode=0o700)
    snapshot = version / "snapshot"
    restore = version / "restore"
    snapshot.mkdir(mode=0o700)
    restore.mkdir(mode=0o700)
    for name in sorted(entries):
        size = _copy_exact(sources[name], snapshot / name, members[name]["sha256"])
        if size != members[name]["bytes"]:
            raise ValueError("snapshot byte count differs")
    inventory_digest = hashlib.sha256(json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    manifest = {"schema": "pcrl-r18-private-a-fixed-snapshot-v1",
                "status": "SNAPSHOT_HASH_READ_PASS", "version": version.name,
                "member_count": len(members), "members": members,
                "source_inventory_sha256": inventory_digest,
                "uncached_hash_read": True}
    manifest_sha = _write_once_json(snapshot / "MANIFEST.json", manifest)
    for name in sorted(entries):
        size = _copy_exact(snapshot / name, restore / name, members[name]["sha256"])
        if size != members[name]["bytes"]:
            raise ValueError("restored byte count differs")
    _copy_exact(snapshot / "MANIFEST.json", restore / "MANIFEST.json", manifest_sha)
    proof = {"schema": "pcrl-r18-private-a-fixed-restore-proof-v1",
             "status": "LOCAL_UNCACHED_SNAPSHOT_AND_RESTORE_PASS",
             "version": version.name, "member_count": len(members),
             "members": members, "snapshot_manifest_sha256": manifest_sha,
             "source_inventory_sha256": inventory_digest,
             "uncached_hash_read": True}
    _write_once_json(version / "RESTORE_PROOF.json", proof)
    _fsync_directory(snapshot)
    _fsync_directory(restore)
    _fsync_directory(version)
    _fsync_directory(version.parent)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--version-dir", type=Path, required=True)
    args = parser.parse_args()
    _owner_dir(args.inventory.parent)
    if stat.S_IMODE(args.inventory.stat().st_mode) != 0o600:
        raise PermissionError("snapshot inventory must be owner-only")
    entries = json.loads(args.inventory.read_text())
    proof = copy_and_restore(entries, args.version_dir)
    result = {"status": proof["status"], "version": proof["version"],
              "member_count": proof["member_count"],
              "snapshot_manifest_sha256": proof["snapshot_manifest_sha256"],
              "restore_proof_sha256": uncached_sha256(
                  args.version_dir / "RESTORE_PROOF.json")[0]}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
