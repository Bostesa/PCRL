"""Repair one omitted S3 cache-root field in the sealed A bank spec set.

This reads only spec, plan, lock, seal, and source-file metadata. It does not
read role arrays, enqueue work, or run the bank adapter. The v1 set is retained.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from build_a_score_bank_specs import build_specs, PLAN_SHA256, ROOT, CAMPAIGN, SHARED
from compute_queue import Queue
from safe_data_guard import sha_file


V1_INDEX_SHA256 = "e69510454b63e5c91a2fd1ec9c0db310e036a252379d6a06395043d703cb65ca"
V1_ROOT = SHARED / ".private" / "a_score_bank_specs_v1"
V2_ROOT = SHARED / ".private" / "a_score_bank_specs_v2"
QUEUE_ROOT = SHARED / ".private" / "compute"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("spec metadata is not a JSON object")
    return value


def _repair_one(original: dict, cache_root: Path) -> dict:
    """Produce an exact one-key extension, refusing any preexisting cache key."""
    if not isinstance(original, dict) or "cache_root" in original:
        raise ValueError("v1 spec does not have the expected missing key")
    repaired = dict(original)
    repaired["cache_root"] = str(cache_root)
    if set(repaired) != set(original) | {"cache_root"} or any(
        repaired[key] != value for key, value in original.items()
    ):
        raise AssertionError("spec changed beyond the cache-root addition")
    return repaired


def build_repaired_specs(lock_sha256: str, seal_sha256: str) -> tuple[list[dict], dict]:
    """Validate the immutable v1 set and derive ten one-field extensions."""
    index_path = V1_ROOT / "SPEC_INDEX.json"
    if sha_file(index_path) != V1_INDEX_SHA256:
        raise PermissionError("v1 spec index differs from the preserved set")
    index = _json(index_path)
    if (index.get("schema") != "pcrl-a-score-bank-s3-spec-index-v1" or
            index.get("status") != "SPECS_BUILT_NOT_ENQUEUED" or
            index.get("plan_sha256") != PLAN_SHA256 or
            index.get("campaign_lock_sha256") != lock_sha256 or
            index.get("local_seal_receipt_sha256") != seal_sha256):
        raise PermissionError("v1 spec index identities differ")
    files = index.get("unit_specs_sha256")
    if not isinstance(files, dict) or len(files) != 10:
        raise ValueError("v1 spec index does not contain exactly ten units")
    if {p.name for p in V1_ROOT.iterdir()} != set(files) | {"SPEC_INDEX.json"}:
        raise PermissionError("v1 spec directory has unexpected files")
    seal_path = CAMPAIGN / "LOCAL_SEAL_RECEIPT.json"
    if (sha_file(seal_path) != seal_sha256 or
            not Queue._local_seal_ok({"local_seal_path": str(seal_path),
                                      "local_seal_sha256": seal_sha256})):
        raise PermissionError("committed A preprocessing seal differs")
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    if sha_file(lock_path) != lock_sha256:
        raise PermissionError("effective A preprocessing lock differs")
    lock = _json(lock_path)
    cache_rel = lock.get("a_score_bank_cache_root")
    if not isinstance(cache_rel, str) or Path(cache_rel).is_absolute():
        raise PermissionError("lock has no relative private cache root")
    cache = (ROOT / cache_rel).resolve(strict=True)
    if not cache.is_dir():
        raise PermissionError("private cache root is absent")
    expected = build_specs(SHARED / "A_SCORE_BANK_PREPROCESS_SPEC_DRAFT.json",
                           lock_path, lock_sha256, ROOT, QUEUE_ROOT)
    expected_by_name = {f"{spec['unit_id']}.json": spec for spec in expected}
    if set(files) != set(expected_by_name):
        raise PermissionError("v1 unit IDs differ from the sealed plan")
    repaired = []
    for name in sorted(files):
        original_path = V1_ROOT / name
        if sha_file(original_path) != files[name]:
            raise PermissionError("a preserved v1 unit spec differs")
        original = _json(original_path)
        if original != expected_by_name[name]:
            raise PermissionError("a v1 unit spec differs from the sealed plan")
        repaired.append(_repair_one(original, cache))
    return repaired, index


def _write_json_exclusive(path: Path, value: dict) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def write_repaired_specs(lock_sha256: str, seal_sha256: str,
                         expected_source_sha256: str) -> dict:
    if sha_file(Path(__file__)) != expected_source_sha256:
        raise PermissionError("reviewed repair source differs")
    specs, _ = build_repaired_specs(lock_sha256, seal_sha256)
    if V2_ROOT.exists():
        raise FileExistsError("v2 spec set is write-once")
    V2_ROOT.mkdir(parents=True, mode=0o700)
    files = {}
    for spec in specs:
        name = f"{spec['unit_id']}.json"
        files[name] = _write_json_exclusive(V2_ROOT / name, spec)
    new_index = {
        "schema": "pcrl-a-score-bank-s3-spec-index-v2",
        "status": "SPECS_BUILT_NOT_ENQUEUED",
        "plan_sha256": PLAN_SHA256,
        "campaign_lock_sha256": lock_sha256,
        "local_seal_receipt_sha256": seal_sha256,
        "supersedes_spec_index_sha256": V1_INDEX_SHA256,
        "repair_source_sha256": expected_source_sha256,
        "unit_specs_sha256": files,
    }
    index_sha = _write_json_exclusive(V2_ROOT / "SPEC_INDEX.json", new_index)
    return {"units": len(specs), "spec_index_sha256": index_sha}


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-lock-sha256", required=True)
    parser.add_argument("--local-seal-sha256", required=True)
    parser.add_argument("--repair-source-sha256", required=True)
    args = parser.parse_args()
    try:
        result = write_repaired_specs(args.campaign_lock_sha256,
                                      args.local_seal_sha256,
                                      args.repair_source_sha256)
    except BaseException as exc:
        print("A_SCORE_BANK_SPEC_REPAIR_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))
