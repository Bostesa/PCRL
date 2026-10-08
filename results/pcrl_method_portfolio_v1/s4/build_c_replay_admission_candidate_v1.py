"""Build a hash-only, non-effective S4 C R1 replay admission candidate.

This reads S3 queue metadata, private JSON output bytes only for their hashes,
and public lock/spec files. It never opens role NPZs or candidate score arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SCHEMA_PATH = HERE / "S4_C_REPLAY_UNIT_SCHEMA_DRAFT.json"
QUEUE_ROOT = CAMPAIGN / "shared/.private/compute"
OUT = CAMPAIGN / "S4_C_REPLAY_ADMISSION_CANDIDATE.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def completed_parent(queue: dict, unit_id: str) -> dict:
    state = queue.get("units", {}).get(unit_id, {})
    spec = state.get("spec", {})
    receipt_path = QUEUE_ROOT / "units" / unit_id / "RECEIPT.json"
    receipt = read(receipt_path)
    outputs = spec.get("outputs", [])
    if (state.get("status") != "done" or receipt.get("status") != "done" or
            state.get("returncode") != 0 or receipt.get("returncode") != 0):
        raise ValueError(f"{unit_id} is not completed")
    if (len(outputs) != 1 or receipt.get("unit_id") != unit_id or
            state.get("output_sha256") != receipt.get("output_sha256")):
        raise ValueError(f"{unit_id} output map differs")
    output = Path(outputs[0])
    digest = receipt["output_sha256"].get(str(output))
    if not output.is_file() or not isinstance(digest, str) or sha(output) != digest:
        raise ValueError(f"{unit_id} output bytes differ")
    return {"unit_id": unit_id, "queue_receipt_sha256": sha(receipt_path),
            "output_sha256": digest}


def build() -> dict:
    schema = read(SCHEMA_PATH)
    if schema.get("schema") != "pcrl-s4-c-replay-unit-schema-v1" or len(
            schema.get("planned_units", {})) != 17:
        raise ValueError("S4 C prospective unit schema differs")
    for path, digest in schema["source_sha256_by_path"].items():
        if sha(HERE.parents[2] / path) != digest:
            raise ValueError(f"S4 C source changed: {path}")
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    fit_seal_path = CAMPAIGN / "LOCAL_SEAL_RECEIPT.json"
    fit_review_path = CAMPAIGN / "S4_PREFIT_SEAL_REVIEW.json"
    manifest_path = CAMPAIGN / "DATA_ACCESS_MANIFEST.json"
    if (sha(lock_path) != schema["fit_lock_sha256"] or
            sha(fit_seal_path) != schema["historical_fit_seal_sha256"] or
            sha(manifest_path) != schema["data_manifest_sha256"]):
        raise ValueError("S4 C historical fit/data source changed")
    fit_seal = read(fit_seal_path)
    if (fit_seal.get("status") != "PASS" or
            fit_seal.get("lock_sha256") != schema["fit_lock_sha256"] or
            fit_seal.get("review_sha256") != sha(fit_review_path)):
        raise ValueError("S4 C historical FIT review/seal differs")
    queue = read(QUEUE_ROOT / "QUEUE.json")
    units = {}
    parents_seen = set()
    for unit_id, item in schema["planned_units"].items():
        entry = dict(item)
        if entry["stage"] == "independent_c_r1_arm_replay":
            entry["parents"] = {label: completed_parent(queue, parent)
                                for label, parent in item["parents"].items()}
            parents_seen.update(item["parents"].values())
        else:
            entry["parents"] = {"c_finalizer": completed_parent(
                queue, item["parents"]["c_finalizer"]),
                "arm_reports": item["parents"]["arm_reports"]}
            parents_seen.add(item["parents"]["c_finalizer"])
        units[unit_id] = entry
    if len(parents_seen) != 135:
        raise ValueError("S4 C full R1 135-unit parent union differs")
    return {
        "schema": "pcrl-s4-c-replay-admission-v1",
        "status": "DRAFT_NO_ROLE_ACCESS",
        "adult_role_access": False,
        "scope": "S4_C_R1_INDEPENDENT_ARM_AND_AGGREGATE_REPLAY",
        "lock_sha256": schema["fit_lock_sha256"],
        "lock_commit_sha": fit_seal["lock_commit_sha"],
        "fit_seal_relative_path": fit_seal_path.name,
        "fit_seal_sha256": sha(fit_seal_path),
        "fit_review_relative_path": fit_review_path.name,
        "fit_review_sha256": sha(fit_review_path),
        "data_manifest_sha256": sha(manifest_path),
        "unit_schema_sha256": sha(SCHEMA_PATH),
        "source_sha256_by_path": schema["source_sha256_by_path"],
        "queue_source_sha256": "PENDING_SEPARATE_S3_ROUTE",
        "verifier_synthetic_v12_index_sha256": sha(HERE / "S4_VERIFIER_QUEUE_INDEX_V12.json"),
        "unit_specs": units,
        "unit_count": 17,
        "parent_c_unit_count": 135,
        "cpu_reserve_s": 3900,
        "post_arm_registry_schema": "pcrl-s4-c-r1-arm-result-hash-registry-v1",
        "post_arm_registry_builder_sha256": sha(HERE / "build_c_arm_result_registry_v1.py"),
        "scientific_scope": "Exploratory development only; all eight C arms and both paired seeds, frozen controls and 2048 paired group intervals; no old assessment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.resolve() != OUT.resolve():
        raise PermissionError("S4 C admission candidate output path differs")
    result = build()
    payload = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    with OUT.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"path": str(OUT), "sha256": sha(OUT),
                      "unit_count": result["unit_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
