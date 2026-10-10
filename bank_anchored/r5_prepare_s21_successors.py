"""Build versioned queue specs after the S21 conditional-null amendment.

This builder reads only receipts and hashes; it never opens attack labels or
private frozen predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "results/pcrl_bank_anchored_release_v1"
R5 = CAMP / "r5"
SOURCE = (
    ("R5_RECOVERY_INTERVALS_K1575_V1", "R5_RECOVERY_INTERVALS_K1575_V2",
     "bank_anchored.r5_recovery_intervals_v2",
     "bank_anchored/r5_recovery_intervals_v2.py"),
    ("R5_REPLAY_PUBLIC_SELECTION", "R5_REPLAY_PUBLIC_SELECTION_V2",
     "bank_anchored.r5_replay_selection_v2",
     "bank_anchored/r5_replay_selection_v2.py"),
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("successor source symlink refused")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise PermissionError("successor source missing")
    return {"path": str(path), "sha256": sha(path)}


def write_once(path: Path, payload: dict, mode: int) -> str:
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.read_bytes() != data:
            raise FileExistsError("successor artifact already differs")
        return sha(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(data).hexdigest()


def build(amendment_path: Path) -> dict:
    plan_path = R5 / "S21_CONDITIONAL_NULL_CORRECTION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    state = json.loads((CAMP / "r0/.private/compute/state.json").read_text())
    current = [item["sha256"] for item in state["amendments"]
               if "verification" in item["stages"]][-1]
    if (plan.get("schema") != "BAR1-r5-s21-conditional-null-correction-plan-v1" or
            amendment.get("schema") != "pcrl-bank-dated-amendment-v1" or
            current != sha(amendment_path) or
            not set(plan["new_unit_ids"]).issubset(set(amendment["unit_ids"]))):
        raise PermissionError("exact registered successor amendment missing")
    python = "/Users/nathansamson/PCRL/.venv/bin/python"
    if not os.access(python, os.X_OK):
        raise PermissionError("exact venv entrypoint missing")
    records = []
    for old_id, new_id, module, worker in SOURCE:
        old_spec_path = R5 / ".private" / f"{old_id}_QUEUE_SPEC.json"
        old_spec = json.loads(old_spec_path.read_text())
        failure_path = CAMP / "r0/.private/compute/units" / old_id / "RECEIPT.json"
        failure = json.loads(failure_path.read_text())
        if (old_spec["unit_id"] != old_id or failure["unit_id"] != old_id or
                failure["status"] != "failed" or
                plan["failed_predecessors"][old_id]["queue_receipt_sha256"] !=
                sha(failure_path) or
                plan["sources"]["recovery_v2_worker_sha256" if
                                "RECOVERY" in new_id else
                                "selection_v2_worker_sha256"] != sha(ROOT / worker)):
            raise PermissionError("failed predecessor or V2 worker source changed")
        for item in old_spec["sources"]:
            if pin(Path(item["path"])) != item:
                raise PermissionError("inherited source hash changed")
        spec = dict(old_spec)
        spec["unit_id"] = new_id
        spec["amendment_sha256"] = current
        spec["argv"] = list(old_spec["argv"])
        spec["argv"][0] = python
        spec["argv"][2] = module
        for position, value in enumerate(spec["argv"]):
            if value == "--output":
                spec["argv"][position + 1] = str(R5 / f"{new_id}.json" if
                                                  "RECOVERY" in new_id else
                                                  R5 / f"{new_id}_RECEIPT.json")
            if value == "--receipt":
                spec["argv"][position + 1] = str(R5 / f"{new_id}_RECEIPT.json")
        spec["outputs"] = ([str(R5 / f"{new_id}.json"),
                            str(R5 / f"{new_id}_RECEIPT.json")] if
                           "RECOVERY" in new_id else
                           [str(R5 / f"{new_id}_RECEIPT.json")])
        for output in spec["outputs"]:
            if Path(output).exists():
                raise FileExistsError("successor output already exists")
        additions = [ROOT / worker, Path(__file__), plan_path, amendment_path,
                     failure_path, old_spec_path,
                     ROOT / "tests/test_bank_r5_recovery_v2.py" if
                     "RECOVERY" in new_id else
                     ROOT / "tests/test_bank_r5_selection_v2.py"]
        pins = {item["path"]: item for item in old_spec["sources"]}
        for path in additions:
            item = pin(path)
            pins[item["path"]] = item
        spec["sources"] = [pins[key] for key in sorted(pins)]
        spec_path = R5 / ".private" / f"{new_id}_QUEUE_SPEC.json"
        digest = write_once(spec_path, spec, 0o600)
        records.append({"unit_id": new_id, "predecessor": old_id,
                        "queue_spec_path": str(spec_path),
                        "queue_spec_sha256": digest,
                        "source_count": len(spec["sources"]),
                        "cpu_reserve_s": spec["cpu_reserve_s"],
                        "max_wall_s": spec["max_wall_s"],
                        "max_rss_gib": spec["max_rss_gib"]})
    index = {"schema": "BAR1-r5-s21-successor-queue-index-v1",
             "status": "SPECS_READY_NO_NEW_ROLE_READ",
             "amendment_sha256": current,
             "correction_plan_sha256": sha(plan_path), "units": records}
    write_once(R5 / "S21_SUCCESSOR_QUEUE_INDEX.json", index, 0o644)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--amendment", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.amendment), sort_keys=True))


if __name__ == "__main__":
    main()
