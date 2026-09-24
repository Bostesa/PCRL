"""Reuse the predecessor's staged, label-stripped 2018 inputs for this study.

The adaptive-release study (AR) staged 17 hash-pinned objects under
`s3://pcrl-ux-archive-ed9d21fd/pcrl_adaptive_release_v1/inputs/NN-<name>`
(pinned index, SANITIZATION receipt, three sanitized anchors, and per anchor
the encoder plus the Q/D17/D33 maps). Its private manifest records each
object's version, byte size, SHA-256 and host destination. This module

* `plan`   (Mac): reads AR's private manifest, keeps only those 17 objects
  (never `original_prepared_*`, which still carry outer labels) plus, with
  `--include-j`, AR's label-free J external index (pinned below), lists the S3 versions, checks size + SSE, and with
  `--verify-sha` streams every object through SHA-256 (temporary file of at
  most one object in the scratch directory, deleted immediately). Writes this
  study's private write-once manifest.
* `upload-manifest` (Mac): puts that manifest under
  `pcrl_shared_context_release_v1/control/INPUT_STAGE.json` (SSE, versioned)
  and prints the version id + SHA the user data pins.
* `restore` (host): fetches every object by version id, verifies SHA-256 and
  size before moving it into place, and writes a restore receipt.
No object is re-uploaded; the predecessor's versions are the inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STUDY = "pcrl_shared_context_release_v1"
BUCKET = "pcrl-ux-archive-ed9d21fd"
REGION = "us-east-1"
PROFILE = "vein"
SOURCE_PREFIX = "pcrl_adaptive_release_v1/inputs/"
CONTROL_KEY = f"{STUDY}/control/INPUT_STAGE.json"
AR_MANIFEST = Path("/Users/nathansamson/PCRL/.worktrees/pcrl-adaptive-release-v1/"
                  "results/pcrl_adaptive_release_v1/private/INPUT_STAGE.json")
MANIFEST = ROOT / "results" / STUDY / "agents" / "infra" / "private" / "INPUT_STAGE.json"  # git-ignored (infra/)
SCHEMA = "pcrl-sc-input-stage-v1"
ALLOWED_KEY = re.compile(r"^pcrl_adaptive_release_v1/inputs/(0[0-9]|1[0-6])-"
                         r"(REUSABLE_INPUTS_PINNED\.json|SANITIZATION\.json|anchor_[012]\.joblib|"
                         r"encoder\.joblib|Q\.npz)$")
FORBIDDEN = ("original_prepared",)
# J continuity reference: J16 itself is already inside the sanitized anchors
# (pool["J"]); AR's external_audit additionally requires its private external
# index (metadata only: paths, hashes, pool shapes; no person rows or labels),
# pinned (J_INDEX) to the exact version/SHA AR passed as --external-index-sha256.
HOST_WORK = "/opt/pcrl/work"
HOST_J_DIR = f"{HOST_WORK}/results/{STUDY}/private/j_inputs"
J_INDEX = {"s3_key": "pcrl_adaptive_release_v1/inputs/EXTERNAL_RELEASE_INPUTS.private.json",
           "s3_version_id": "GELw62dAVhylHvcpI83mTbTeoX3.cU3j",
           "sha256": "461b06f0bba10416e4a6bec6bc0c69a3821ae2ef9733bc45030ffd0dfea4ad8a",
           "bytes": 32251,
           "destination": f"{HOST_J_DIR}/EXTERNAL_RELEASE_INPUTS.private.json"}


def _aws(*args: str, profile: str | None = PROFILE) -> dict:
    command = ["aws", *(["--profile", profile] if profile else []), "--region", REGION,
               *args, "--output", "json"]
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(process.stdout) if process.stdout.strip() else {}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_records(ar_manifest: dict) -> list[dict]:
    """The 17 label-stripped AR objects (+ optional pinned J index); refuse anything else."""
    if ar_manifest.get("schema") != "pcrl-adaptive-input-stage-v1" or ar_manifest.get("bucket") != BUCKET:
        raise ValueError("unexpected predecessor input manifest")
    files = [entry for entry in ar_manifest["files"] if entry["s3_key"] != J_INDEX["s3_key"]]
    extra = [entry for entry in ar_manifest["files"] if entry["s3_key"] == J_INDEX["s3_key"]]
    if len(extra) > 1 or any({k: e.get(k) for k in J_INDEX} != J_INDEX for e in extra):
        raise PermissionError("J external index entry differs from its registered pin")
    records = []
    for entry in files:
        key = entry["s3_key"]
        if any(word in key for word in FORBIDDEN) or not ALLOWED_KEY.match(key):
            raise PermissionError(f"refusing non-allowlisted input object {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) or not entry.get("s3_version_id"):
            raise ValueError(f"{key}: missing SHA-256 or version pin")
        records.append({"s3_key": key, "s3_version_id": entry["s3_version_id"],
                        "sha256": entry["sha256"], "bytes": int(entry["bytes"]),
                        "destination": entry["destination"]})
    if len(records) != 17 or len({r["s3_key"] for r in records}) != 17:
        raise ValueError("expected exactly the 17 predecessor input objects")
    for record in records:
        destination = record["destination"]
        if record["s3_key"].endswith(("anchor_0.joblib", "anchor_1.joblib", "anchor_2.joblib",
                                      "SANITIZATION.json")) and \
                "/private/sanitized_2018_v2/" not in destination:
            raise PermissionError("prepared anchors must restore only as sanitized copies")
    return records + [dict(J_INDEX) for _ in extra]


def plan(ar_manifest_path: Path = AR_MANIFEST, manifest_path: Path = MANIFEST, *,
         verify_sha: bool = False, scratch: str | None = None, include_j: bool = False) -> dict:
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError("study input-stage manifest is write-once")
    ar_value = json.loads(Path(ar_manifest_path).read_text())
    if include_j:
        ar_value = {**ar_value, "files": [*ar_value["files"], dict(J_INDEX)]}
    records = select_records(ar_value)
    listing = _aws("s3api", "list-object-versions", "--bucket", BUCKET, "--prefix", SOURCE_PREFIX)
    versions = {(v["Key"], v["VersionId"]): v for v in listing.get("Versions", [])}
    for record in records:
        found = versions.get((record["s3_key"], record["s3_version_id"]))
        if found is None or int(found["Size"]) != record["bytes"]:
            raise ValueError(f"{record['s3_key']}: pinned version absent or size differs")
        head = _aws("s3api", "head-object", "--bucket", BUCKET, "--key", record["s3_key"],
                    "--version-id", record["s3_version_id"])
        if head.get("ServerSideEncryption") != "AES256":
            raise ValueError(f"{record['s3_key']}: object is not SSE-encrypted")
        record["s3_latest"] = bool(found.get("IsLatest"))
        record["sha256_verified"] = False
        if verify_sha:
            with tempfile.TemporaryDirectory(dir=scratch) as temp:
                target = Path(temp) / "object"
                _aws("s3api", "get-object", "--bucket", BUCKET, "--key", record["s3_key"],
                     "--version-id", record["s3_version_id"], str(target))
                actual = _digest(target)
                target.unlink()
            if actual != record["sha256"]:
                raise ValueError(f"{record['s3_key']}: SHA-256 differs from predecessor pin")
            record["sha256_verified"] = True
    value = {"schema": SCHEMA, "study": STUDY, "bucket": BUCKET,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "reused_from": {"manifest_sha256": _digest(Path(ar_manifest_path)),
                             "schema": ar_value["schema"],
                             "source_index_sha256": ar_value["source_index_sha256"]},
             "excluded_objects": "original_prepared_* (outer labels) never staged",
             "j_external_index_included": include_j,
             "all_sha256_verified": all(r["sha256_verified"] for r in records),
             "total_bytes": sum(r["bytes"] for r in records), "files": records}
    manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, manifest_path)
    return {"files": len(records), "total_bytes": value["total_bytes"],
            "all_sha256_verified": value["all_sha256_verified"],
            "manifest_sha256": _digest(manifest_path)}


def upload_manifest(manifest_path: Path = MANIFEST) -> dict:
    value = json.loads(Path(manifest_path).read_text())
    if value.get("schema") != SCHEMA or not value.get("all_sha256_verified"):
        raise ValueError("upload only a fully SHA-verified study manifest")
    put = _aws("s3api", "put-object", "--bucket", BUCKET, "--key", CONTROL_KEY,
               "--body", str(manifest_path), "--server-side-encryption", "AES256")
    return {"key": CONTROL_KEY, "version_id": put["VersionId"],
            "sha256": _digest(Path(manifest_path))}


def restore(manifest_path: str | Path, *, receipt_path: str | Path | None = None,
            profile: str | None = None) -> dict:
    """Host side: fetch pinned versions, verify SHA-256 and size, then move into place."""
    value = json.loads(Path(manifest_path).read_text())
    if value.get("schema") != SCHEMA or value.get("bucket") != BUCKET:
        raise ValueError("unknown input-stage manifest")
    records = select_records({"schema": "pcrl-adaptive-input-stage-v1", "bucket": BUCKET,
                              "files": [{**r} for r in value["files"]]})
    restored = []
    for record in records:
        destination = Path(record["destination"])
        if destination.is_file() and _digest(destination) == record["sha256"]:
            restored.append({**record, "action": "already_present_verified"})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        _aws("s3api", "get-object", "--bucket", BUCKET, "--key", record["s3_key"],
             "--version-id", record["s3_version_id"], str(temporary), profile=profile)
        if temporary.stat().st_size != record["bytes"] or _digest(temporary) != record["sha256"]:
            temporary.unlink()
            raise ValueError(f"{record['s3_key']}: restored bytes differ from pinned SHA-256")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        restored.append({**record, "action": "restored_verified"})
    receipt = {"schema": "pcrl-sc-input-restore-v1",
               "utc": datetime.now(timezone.utc).isoformat(),
               "manifest_sha256": _digest(Path(manifest_path)),
               "files": len(restored), "all_verified": True,
               "outer_labels_staged": False, "records": restored}
    if receipt_path:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {key: receipt[key] for key in ("files", "all_verified", "manifest_sha256")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    item = sub.add_parser("plan")
    item.add_argument("--ar-manifest", default=str(AR_MANIFEST))
    item.add_argument("--manifest", default=str(MANIFEST))
    item.add_argument("--verify-sha", action="store_true")
    item.add_argument("--scratch")
    item.add_argument("--include-j", action="store_true")
    item = sub.add_parser("upload-manifest")
    item.add_argument("--manifest", default=str(MANIFEST))
    item = sub.add_parser("restore")
    item.add_argument("--manifest", required=True)
    item.add_argument("--receipt", default="/opt/pcrl/logs/INPUT_RESTORE.json")
    args = parser.parse_args(argv)
    if args.action == "plan":
        result = plan(Path(args.ar_manifest), Path(args.manifest), verify_sha=args.verify_sha,
                      scratch=args.scratch, include_j=args.include_j)
    elif args.action == "upload-manifest":
        result = upload_manifest(Path(args.manifest))
    else:
        result = restore(args.manifest, receipt_path=args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
