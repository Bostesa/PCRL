"""Stage only hash-pinned, label-stripped 2018 inputs for the task host.

The local manifest is private. It contains original private paths and S3
versions; public reports must cite its hash, not copy the manifest contents.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experiments.pcrl_task_aligned_cuts_v1 import data

BUCKET = "pcrl-ux-archive-ed9d21fd"
PREFIX = "pcrl_adaptive_release_v1/inputs"
PROFILE = "vein"
REGION = "us-east-1"
HOST_ROOT = Path("/opt/pcrl/work")


def _aws(*args: str) -> dict:
    process = subprocess.run(
        ["aws", "--profile", PROFILE, "--region", REGION, *args,
         "--output", "json"], check=True, capture_output=True, text=True)
    return json.loads(process.stdout) if process.stdout.strip() else {}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(index_path: str | Path, manifest_path: str | Path) -> dict:
    manifest_file = Path(manifest_path)
    if manifest_file.exists():
        raise FileExistsError("private input-stage manifest is immutable")
    value = data.index(index_path)
    index_file = Path(index_path).resolve()
    sanitized = data._sanitized_root()  # existing verified, label-stripped receipt
    if not (sanitized / "SANITIZATION.json").is_file():
        raise FileNotFoundError("verified sanitized 2018 receipt required")
    records: list[tuple[Path, Path]] = [(index_file, HOST_ROOT / data.INDEX_RELATIVE)]
    for name in ("SANITIZATION.json", "anchor_0.joblib", "anchor_1.joblib", "anchor_2.joblib"):
        source = (sanitized / name).resolve()
        remote = HOST_ROOT / data.SANITIZED_RELATIVE / name
        records.append((source, remote))
    for anchor in (0, 1, 2):
        for kind, name, filename in (("encoder", None, None),
                                     ("map", "Q", "Q.npz"),
                                     ("map", "D17", "Q.npz"),
                                     ("map", "D33", "Q.npz")):
            historical = data.member_record(value, anchor, kind, name, filename)
            source = data.verified_member(historical)
            records.append((source, Path(historical["path"])))
    files = []
    for ordinal, (source, destination) in enumerate(records):
        expected = _digest(source)
        key = f"{PREFIX}/{ordinal:02d}-{source.name}"
        _aws("s3", "cp", str(source), f"s3://{BUCKET}/{key}",
             "--sse", "AES256", "--only-show-errors")
        head = _aws("s3api", "head-object", "--bucket", BUCKET, "--key", key)
        if (head.get("ContentLength") != source.stat().st_size or
                head.get("ServerSideEncryption") != "AES256"):
            raise ValueError("staged private object size/encryption mismatch")
        files.append({"source": str(source), "destination": str(destination),
                      "s3_key": key, "s3_version_id": head.get("VersionId"),
                      "sha256": expected, "bytes": source.stat().st_size})
    manifest = {"schema": "pcrl-adaptive-input-stage-v1", "bucket": BUCKET,
                "source_index_sha256": _digest(index_file), "files": files}
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"files": len(files), "total_bytes": sum(f["bytes"] for f in files),
            "manifest_sha256": _digest(manifest_file)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--manifest", required=True)
    arguments = parser.parse_args()
    print(json.dumps(stage(arguments.index, arguments.manifest), sort_keys=True))
