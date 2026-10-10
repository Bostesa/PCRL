"""Write-once, hash-checked local custody of the selected R1 release.

The private plan is reviewed before snapshot. This module only copies bytes;
it never deserializes models, draws, arrays, or person-level releases.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile


def sha256_uncached(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        fcntl.fcntl(source.fileno(), fcntl.F_NOCACHE, 1)
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _new_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.mkdir(mode=0o700, parents=False)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise PermissionError("custody directory must be owner-only")


def _copy(source: Path, target: Path, expected: str, size: int) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("custody source must be a regular file")
    if source.stat().st_size != size or sha256_uncached(source) != expected:
        raise ValueError("custody source bytes differ from plan")
    missing = []
    cursor = target.parent
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    if any(stat.S_IMODE(directory.stat().st_mode) & 0o077 for directory in missing):
        raise PermissionError("custody parent directory is not owner-only")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb", buffering=0) as src, os.fdopen(fd, "wb", buffering=0) as dst:
            fcntl.fcntl(src.fileno(), fcntl.F_NOCACHE, 1)
            fcntl.fcntl(dst.fileno(), fcntl.F_NOCACHE, 1)
            shutil.copyfileobj(src, dst, 1024 * 1024)
            os.fsync(dst.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if target.stat().st_size != size or sha256_uncached(target) != expected:
        raise ValueError("custody copy fails uncached hash verification")


def _write_json(target: Path, record: dict) -> None:
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(record, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def _checked_plan(path: Path, *, verify_originals: bool) -> dict:
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("private custody plan must be owner-only")
    plan = json.loads(path.read_text())
    if plan.get("schema") != "pcrl-bank-r1-custody-plan-v1":
        raise ValueError("custody plan schema differs")
    names: set[str] = set()
    for entry in plan["entries"]:
        name = entry["archive_name"]
        if (not isinstance(name, str) or name in names or
                Path(name).is_absolute() or ".." in Path(name).parts or
                not name.startswith(("data/", "reader/", "code/"))):
            raise ValueError("custody path is unsafe or duplicated")
        names.add(name)
        if verify_originals and sha256_uncached(Path(entry["source_path"])) != entry["sha256"]:
            raise ValueError("custody source changed after plan")
    if len(plan["entries"]) != 55:
        raise ValueError("50 deployment, 4 reader, and 1 source-code entries required")
    return plan


def snapshot(plan_path: Path, target: Path) -> dict:
    plan = _checked_plan(plan_path, verify_originals=True)
    _new_dir(target)
    _copy(plan_path, target / "PRIVATE_PLAN.json", sha256_uncached(plan_path),
          plan_path.stat().st_size)
    for entry in plan["entries"]:
        _copy(Path(entry["source_path"]), target / entry["archive_name"],
              entry["sha256"], entry["bytes"])
    receipt = {"schema": "pcrl-bank-r1-local-snapshot-v1", "status": "HASH_VERIFIED",
               "plan_sha256": sha256_uncached(plan_path), "entries": len(plan["entries"]),
               "source_hashes": {e["archive_name"]: e["sha256"] for e in plan["entries"]},
               "verification": "F_NOCACHE source and copied-file SHA-256 reads"}
    _write_json(target / "SNAPSHOT_RECEIPT.json", receipt)
    return receipt


def restore(snapshot_dir: Path, target: Path) -> dict:
    plan_path = snapshot_dir / "PRIVATE_PLAN.json"
    plan = _checked_plan(plan_path, verify_originals=False)
    snap_receipt_path = snapshot_dir / "SNAPSHOT_RECEIPT.json"
    snap_receipt = json.loads(snap_receipt_path.read_text())
    if (snap_receipt.get("status") != "HASH_VERIFIED" or
            snap_receipt.get("plan_sha256") != sha256_uncached(plan_path)):
        raise ValueError("snapshot receipt or plan changed")
    _new_dir(target)
    for entry in plan["entries"]:
        archived = snapshot_dir / entry["archive_name"]
        _copy(archived, target / entry["archive_name"],
              entry["sha256"], entry["bytes"])
    code_root = target / "code_root"
    _new_dir(code_root)
    with tarfile.open(target / "code/runtime_source.tar", "r:") as archive:
        members = archive.getmembers()
        for item in members:
            path = Path(item.name)
            if (path.is_absolute() or ".." in path.parts or
                    not (item.isfile() or item.isdir())):
                raise ValueError("runtime source archive contains unsafe member")
        archive.extractall(code_root, members=members, filter="data")
    for item in code_root.rglob("*"):
        if item.is_dir():
            item.chmod(0o700)
        elif item.is_file():
            item.chmod(0o600)
        else:
            raise ValueError("runtime source contains nonregular member")
    receipt = {"schema": "pcrl-bank-r1-local-restore-v1", "status": "HASH_VERIFIED",
               "snapshot_receipt_sha256": sha256_uncached(snap_receipt_path),
               "plan_sha256": sha256_uncached(plan_path), "entries": len(plan["entries"]),
               "runtime_archive_sha256": sha256_uncached(target / "code/runtime_source.tar"),
               "verification": "F_NOCACHE snapshot and restored-file SHA-256 reads"}
    _write_json(target / "RESTORE_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=("snapshot", "restore"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    result = (snapshot(args.source, args.target) if args.operation == "snapshot"
              else restore(args.source, args.target))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
