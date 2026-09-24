"""Archive completed (and failed-attempt) units: tar + SHA-256 + read-back verify.

Adapted from `AR/archive_unit.py` with this study's constants. Runs on the
task host with its bucket-scoped instance role (no profile). A completed unit
is one with a runner receipt `<units>/_receipts/<id>.json`; the unit directory
must match the receipt's `outputs_sha256` exactly (every file, nothing extra)
before upload, and the receipt itself is uploaded next to the tarball. Each tarball is uploaded SSE-AES256 to a versioned key,
downloaded again by version id, SHA-compared, extracted to a scratch directory
and re-inventoried. Originals are never removed. `sweep` archives everything
not yet archived (completed units, failed attempts, runner status) and is
idempotent; an existing manifest that disagrees with the unit is an error.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

STUDY = "pcrl_shared_context_release_v1"
BUCKET = "pcrl-ux-archive-ed9d21fd"
PREFIX = f"{STUDY}/units"
HOST_RESULTS = Path("/opt/pcrl/work/results") / STUDY
UNITS = HOST_RESULTS / "private" / "units"
MANIFESTS = HOST_RESULTS / "private" / "archive_manifests"
SCRATCH = "/opt/pcrl"
SCHEMA = "pcrl-sc-unit-archive-v1"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def inventory(unit: Path, *, receipt: Path | None = None) -> dict[str, str]:
    if not unit.is_dir() or unit.is_symlink():
        raise ValueError("unit must be a real directory")
    files = sorted(unit.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("symlinks are not archived as evidence")
    result = {path.relative_to(unit).as_posix(): digest(path) for path in files
              if path.is_file() and "__pycache__" not in path.parts}
    if receipt is None:
        return result
    declared = json.loads(Path(receipt).read_text()).get("outputs_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("runner receipt lacks output hashes")
    if declared != result:
        raise ValueError("unit files differ from the runner receipt")
    return result


def aws(*args: str) -> dict:
    process = subprocess.run(["aws", "--region", "us-east-1", *args, "--output", "json"],
                             check=True, capture_output=True, text=True)
    return json.loads(process.stdout) if process.stdout.strip() else {}


def _safe_members(names: list[str], top: str) -> bool:
    return all(not Path(m).is_absolute() and ".." not in Path(m).parts and
               (m == top or m.startswith(top + "/")) for m in names)


def archive(source: Path, key_name: str, *, manifests: Path = MANIFESTS,
            receipt: Path | None = None, upload: bool = True,
            scratch: str = SCRATCH) -> dict:
    """Archive one directory as `<PREFIX>/<key_name>.tar.gz`; verify the read-back."""
    if not key_name or ".." in key_name.split("/") or key_name.startswith("/"):
        raise ValueError("safe archive name required")
    files = inventory(source, receipt=receipt)
    manifest_path = manifests / f"{key_name.replace('/', '__')}.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get("files") != files:
            raise ValueError(f"existing archive manifest differs from current {key_name}")
        return prior
    key = f"{PREFIX}/{key_name}.tar.gz"
    top = source.name
    with tempfile.TemporaryDirectory(prefix="pcrl-sc-archive-", dir=scratch) as temp_name:
        temp = Path(temp_name)
        tar_path = temp / f"{top}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as stream:
            stream.add(source, arcname=top, recursive=True)
        tar_hash = digest(tar_path)
        record = {"schema": SCHEMA, "utc": datetime.now(timezone.utc).isoformat(),
                  "unit": key_name, "bucket": BUCKET, "key": key,
                  "bytes": tar_path.stat().st_size, "archive_sha256": tar_hash,
                  "files": files, "file_count": len(files), "original_retained": True,
                  "runner_receipt_sha256": digest(receipt) if receipt else None}
        restored = tar_path
        if upload:
            try:
                aws("s3api", "head-object", "--bucket", BUCKET, "--key", key)
            except subprocess.CalledProcessError:
                pass
            else:
                raise FileExistsError(f"remote archive key exists without local manifest: {key}")
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
            record["version_id"] = version
        if digest(restored) != tar_hash:
            raise ValueError("archive readback SHA mismatch")
        extract = temp / "restore"
        extract.mkdir()
        with tarfile.open(restored, "r:gz") as stream:
            if not _safe_members(stream.getnames(), top):
                raise ValueError("unsafe archive member")
            stream.extractall(extract, filter="data")
        if inventory(extract / top) != files:
            raise ValueError("restored file inventory mismatch")
        record.update(readback_sha256_verified=upload, restored_inventory_verified=True,
                      uploaded=upload)
    manifests.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    if upload:
        aws("s3api", "put-object", "--bucket", BUCKET,
            "--key", f"{STUDY}/archive_manifests/{manifest_path.name}",
            "--body", str(manifest_path), "--server-side-encryption", "AES256")
        if receipt is not None:
            aws("s3api", "put-object", "--bucket", BUCKET,
                "--key", f"{PREFIX}/{key_name}.RECEIPT.json",
                "--body", str(receipt), "--server-side-encryption", "AES256")
    return record


def sweep(units: Path = UNITS, *, manifests: Path = MANIFESTS, upload: bool = True,
          scratch: str = SCRATCH) -> dict:
    """Archive every completed unit and every failed attempt not yet archived."""
    done, skipped, errors = [], [], {}
    if not units.is_dir():
        return {"archived": done, "skipped": skipped, "errors": errors}
    receipts = units / "_receipts"
    for unit in sorted(p for p in units.iterdir()
                       if p.is_dir() and p.name[0] not in "._"):
        receipt = receipts / f"{unit.name}.json"
        if not receipt.is_file():
            skipped.append(unit.name)
            continue
        try:
            archive(unit, unit.name, manifests=manifests, receipt=receipt, upload=upload,
                    scratch=scratch)
            done.append(unit.name)
        except Exception as error:  # noqa: BLE001 - reported, sweep continues
            errors[unit.name] = str(error)
    attempts = units / ".attempts"
    if attempts.is_dir():
        for attempt in sorted(attempts.glob("*/attempt-*")):
            finished = ((attempt / "FAILED.json").is_file() or
                        (receipts / f"{attempt.parent.name}.json").is_file())
            if not finished:
                continue
            name = f"attempts/{attempt.parent.name}/{attempt.name}"
            try:
                archive(attempt, name, manifests=manifests, upload=upload, scratch=scratch)
                done.append(name)
            except Exception as error:  # noqa: BLE001
                errors[name] = str(error)
    status = units / "STATUS.json"
    if upload and status.is_file():
        aws("s3api", "put-object", "--bucket", BUCKET, "--key", f"{STUDY}/control/STATUS.json",
            "--body", str(status), "--server-side-encryption", "AES256")
    return {"archived": done, "skipped_incomplete": skipped, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("unit")
    one.add_argument("name")
    one.add_argument("--units", default=str(UNITS))
    one.add_argument("--local-only", action="store_true", help="dry run: tar/verify without S3")
    many = sub.add_parser("sweep")
    many.add_argument("--units", default=str(UNITS))
    many.add_argument("--manifests", default=str(MANIFESTS))
    many.add_argument("--local-only", action="store_true")
    many.add_argument("--scratch", default=SCRATCH)
    args = parser.parse_args()
    if args.action == "unit":
        value = archive(Path(args.units) / args.name, args.name,
                        receipt=Path(args.units) / "_receipts" / f"{args.name}.json",
                        upload=not args.local_only)
        print(json.dumps({k: value.get(k) for k in ("unit", "version_id", "bytes", "archive_sha256",
                                                     "file_count", "restored_inventory_verified")},
                         sort_keys=True))
    else:
        result = sweep(Path(args.units), manifests=Path(args.manifests),
                       upload=not args.local_only, scratch=args.scratch)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1 if result["errors"] else 0)
