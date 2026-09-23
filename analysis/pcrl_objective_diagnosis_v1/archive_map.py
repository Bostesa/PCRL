"""Pin local diagnosis inputs to verified private archive members."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .run import digest, write_json


def _groups(records: list[dict], target: int) -> list[list[dict]]:
    groups, current, size = [], [], 0
    for record in records:
        if current and size+record["bytes"] > target:
            groups.append(current); current = []; size = 0
        current.append(record); size += record["bytes"]
    if current:
        groups.append(current)
    return groups


def _members(manifest_path: Path, index_path: Path, target: int, chunk_key: str) -> dict:
    index = json.loads(index_path.read_text())
    if digest(manifest_path) != index["manifest_sha256"]:
        raise AssertionError("private archive manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    groups = _groups(manifest["files"], target)
    chunks = index[chunk_key]
    if len(groups) != len(chunks):
        raise AssertionError("archive chunk registration differs")
    out = {}
    for number, group in enumerate(groups):
        if (chunks[number]["files"] != len(group)
                or chunks[number]["uncompressed_bytes"] != sum(r["bytes"] for r in group)):
            raise AssertionError(f"archive chunk {number} grouping differs")
        for record in group:
            out[record["path"]] = {"member_sha256": record["sha256"],
                "member_bytes": record["bytes"], "archive_bucket": index.get("bucket", index.get("archive_bucket")),
                "archive_key": chunks[number]["key"], "archive_version_id": chunks[number]["version_id"],
                "archive_part_sha256": chunks[number]["sha256"]}
    return out


def augment(reusable_path: Path, prospective_manifest: Path, historical_manifest: Path,
            prospective_index: Path, historical_index: Path, repo: Path) -> None:
    data = json.loads(reusable_path.read_text())
    prospective = _members(prospective_manifest, prospective_index, 900_000_000, "parts")
    historical = _members(historical_manifest, historical_index, 1_000_000_000, "chunks")
    for anchor, entry in data["anchors"].items():
        for key in ("prepared", "encoder"):
            field = entry[key]
            relative = field["path"].split("/results/pcrl_final_prospective_v1/private/restore/", 1)[1]
            archive_path = "results/pcrl_final_prospective_v1/private/restore/"+relative
            record = prospective[archive_path]
            if field["sha256"] != record["member_sha256"]:
                raise AssertionError(f"archived {key} hash differs, anchor {anchor}")
            field["archive"] = record
            field["archive"]["member_path"] = archive_path
        for name, files in entry["maps"].items():
            for file, field in files.items():
                relative = field["path"].split("/results/pcrl_final_prospective_v1/private/restore/", 1)[1]
                archive_path = "results/pcrl_final_prospective_v1/private/restore/"+relative
                record = prospective[archive_path]
                if field["sha256"] != record["member_sha256"]:
                    raise AssertionError(f"archived map hash differs: {anchor}/{name}/{file}")
                field["archive"] = record
                field["archive"]["member_path"] = archive_path
        fine = entry["historical_fineC_tables"]
        archive_path = f"results/pcrl_task_directed_release_v1/private/run/anchor_{anchor}/branches/fineC/tables.joblib"
        record = historical[archive_path]
        if fine["sha256"] != record["member_sha256"]:
            raise AssertionError(f"archived fineC hash differs, anchor {anchor}")
        fine["archive"] = record
        fine["archive"]["member_path"] = archive_path
    data["source_files"] = {}
    for relative in ("experiments/acs_transfer_data.py", "experiments/acs_transfer_heads.py",
                     "experiments/pcrl_task_directed_release_v1/encoding.py",
                     "experiments/pcrl_task_directed_release_v1/mechanisms.py",
                     "experiments/pcrl_task_directed_release_v1/finite.py",
                     "experiments/pcrl_task_directed_release_v1/audits.py",
                     "experiments/pcrl_task_directed_release_v1/robustness.py"):
        local = repo/relative
        frozen_bytes = subprocess.run(["git", "show", f"{data['source_commit']}:{relative}"],
                                      cwd=repo, check=True, capture_output=True).stdout
        frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
        if digest(local) != frozen_hash:
            raise AssertionError(f"historical executable source changed: {relative}")
        data["source_files"][relative] = frozen_hash
    environment = repo/"results/pcrl_task_directed_release_v1/ENVIRONMENT.json"
    data["compatible_historical_environment"] = {"path": str(environment), "sha256": digest(environment),
                                                    "record": json.loads(environment.read_text())}
    data["private_archive_manifests"] = {
        "prospective": {"key": json.loads(prospective_index.read_text())["manifest_object"]["key"],
                         "version_id": json.loads(prospective_index.read_text())["manifest_object"]["version_id"],
                         "sha256": digest(prospective_manifest)},
        "task_directed": {"key": json.loads(historical_index.read_text())["manifest_key"],
                          "version_id": json.loads(historical_index.read_text())["manifest_version_id"],
                          "sha256": digest(historical_manifest)},
    }
    data["restore_instructions"] = (
        "Use AWS_PROFILE=vein and s3api get-object with each pinned bucket/key/version_id. "
        "Stream the part through zstd -dc and tar -xOf - <member_path> to a private destination; "
        "verify member_sha256 before trusted loading. Retrieve and verify the pinned private manifests "
        "first. Do not commit restored person-level prepared.joblib or household metadata. "
        "The 2018 Linux x86 environment record and frozen source hashes are pinned above; "
        "Mac float32 re-encoding is not a substitute for archived encodings.")
    write_json(reusable_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("reusable", "prospective-manifest", "historical-manifest",
                 "prospective-index", "historical-index", "repo"):
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args()
    augment(a.reusable, a.prospective_manifest, a.historical_manifest,
            a.prospective_index, a.historical_index, a.repo)
