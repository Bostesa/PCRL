"""Archive one immutable fitted unit, read it back, and verify every file.

Run on the tagged Linux task host using its bucket-scoped instance role. This
module never removes the original unit. Public reports cite the private
manifest's hash rather than person-level paths or model contents.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path("/opt/pcrl/work/results/pcrl_adaptive_release_v1/private")
BUCKET = "pcrl-ux-archive-ed9d21fd"
PREFIX = "pcrl_adaptive_release_v1/units"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(unit: Path) -> dict[str, str]:
    if not unit.is_dir() or unit.is_symlink():
        raise ValueError("unit must be a real directory")
    files = sorted(unit.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("symlinks are not archived as evidence")
    result = {path.relative_to(unit).as_posix(): digest(path)
              for path in files if path.is_file()}
    receipt = unit / "COMPLETE.json"
    if not receipt.is_file():
        raise ValueError("unit has no immutable COMPLETE receipt")
    declared = json.loads(receipt.read_text()).get("artifacts")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("completion receipt lacks artifact hashes")
    for relative, expected in declared.items():
        if result.get(relative) != expected:
            raise ValueError(f"completion artifact mismatch: {relative}")
    return result


def aws(*args: str) -> dict:
    command = ["aws", "--region", "us-east-1", *args, "--output", "json"]
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(process.stdout) if process.stdout.strip() else {}


def archive(name: str) -> dict:
    if not name or "/" in name or name in (".", ".."):
        raise ValueError("one unit basename required")
    unit = ROOT / name
    files = inventory(unit)
    manifest_dir = ROOT.parent / "archive_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{name}.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get("files") != files:
            raise ValueError("existing archive manifest differs from current unit")
        return prior
    key = f"{PREFIX}/{name}.tar.gz"
    try:
        aws("s3api", "head-object", "--bucket", BUCKET, "--key", key)
    except subprocess.CalledProcessError:
        pass
    else:
        raise FileExistsError("remote archive key exists without local manifest")
    with tempfile.TemporaryDirectory(prefix="pcrl-archive-", dir="/opt/pcrl") as scratch:
        temp = Path(scratch)
        tar_path = temp / f"{name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as stream:
            stream.add(unit, arcname=name, recursive=True)
        tar_hash = digest(tar_path)
        put = aws("s3api", "put-object", "--bucket", BUCKET, "--key", key,
                  "--body", str(tar_path), "--server-side-encryption", "AES256")
        version = put.get("VersionId")
        if not version:
            raise ValueError("versioned private archive required")
        head = aws("s3api", "head-object", "--bucket", BUCKET, "--key", key,
                   "--version-id", version)
        if (head.get("ContentLength") != tar_path.stat().st_size or
                head.get("ServerSideEncryption") != "AES256"):
            raise ValueError("archive size or encryption mismatch")
        restored = temp / "readback.tar.gz"
        aws("s3api", "get-object", "--bucket", BUCKET, "--key", key,
            "--version-id", version, str(restored))
        if digest(restored) != tar_hash:
            raise ValueError("archive readback SHA mismatch")
        extract = temp / "restore"
        extract.mkdir()
        with tarfile.open(restored, "r:gz") as stream:
            names = stream.getnames()
            if any(Path(member).is_absolute() or ".." in Path(member).parts or
                   not (member == name or member.startswith(name + "/"))
                   for member in names):
                raise ValueError("unsafe archive member")
            stream.extractall(extract, filter="data")
        if inventory(extract / name) != files:
            raise ValueError("restored file inventory mismatch")
        record = {
            "schema": "pcrl-adaptive-unit-archive-v1",
            "utc": datetime.now(timezone.utc).isoformat(),
            "unit": name, "bucket": BUCKET, "key": key, "version_id": version,
            "bytes": tar_path.stat().st_size, "archive_sha256": tar_hash,
            "files": files, "file_count": len(files),
            "readback_sha256_verified": True,
            "restored_inventory_verified": True,
            "original_retained": True,
        }
    manifest_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    value = archive(args.name)
    print(json.dumps({key: value[key] for key in
                      ("unit", "version_id", "bytes", "archive_sha256",
                       "file_count", "restored_inventory_verified")}, sort_keys=True))
