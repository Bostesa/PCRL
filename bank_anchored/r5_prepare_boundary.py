"""Build seven write-once FIT numeric-boundary diagnostic queue successors.

The builder checks only receipts and source bytes. It does not open law,
task-label, SEX or per-person prediction archives. R0 must register a dated
verification amendment before calling this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "results/pcrl_bank_anchored_release_v1"
R5 = CAMP / "r5"
QUEUE = CAMP / "r0/.private/compute/units"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("diagnostic source symlink refused")
    absolute = path.resolve(strict=True)
    if not absolute.is_file():
        raise PermissionError("diagnostic source is not regular")
    return {"path": str(absolute), "sha256": sha(absolute)}


def _write_once(path: Path, obj: dict, mode: int) -> str:
    raw = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.read_bytes() != raw:
            raise FileExistsError("diagnostic successor bytes differ")
        return sha(path)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def successor(old: dict, item: dict, worker_pin: dict, extra: list[dict],
              amendment_sha: str, python_entry: Path) -> dict:
    if (old.get("unit_id") != item["source_failed_unit_id"] or
            old.get("stage") != "verification" or
            old.get("max_wall_s") != 120 or
            old.get("cpu_reserve_s") != 120 or
            old.get("max_rss_gib") != 1.0 or
            len(old.get("outputs", ())) != 1 or
            len(old.get("argv", ())) != 13 or
            old["argv"][1:3] != ["-m", "bank_anchored.r5_replay_utility"]):
        raise PermissionError("failed H replay recipe differs")
    original_sources = old["sources"]
    original_worker = [source for source in original_sources
                       if source["path"] == worker_pin["path"]]
    if len(original_worker) != 1:
        raise PermissionError("failed H worker source is absent or ambiguous")
    for source in original_sources:
        if source != original_worker[0] and pin(Path(source["path"])) != source:
            raise PermissionError("failed H immutable source changed")
    source_map = {source["path"]: source for source in original_sources
                  if source != original_worker[0]}
    for source in [worker_pin, *extra]:
        if source["path"] in source_map and source_map[source["path"]] != source:
            raise PermissionError("diagnostic source path has conflicting bytes")
        source_map[source["path"]] = source
    new = dict(old)
    new["unit_id"] = item["successor_unit_id"]
    new["argv"] = [str(python_entry), *old["argv"][1:]]
    new["argv"][-1] = str(R5 / f"{item['successor_unit_id']}_RECEIPT.json")
    new["outputs"] = [new["argv"][-1]]
    new["sources"] = list(source_map.values())
    new["amendment_sha256"] = amendment_sha
    return new


def build(plan_path: Path, amendment_path: Path, python_entry: Path) -> dict:
    plan_pin = pin(plan_path)
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "BAR1-r5-numeric-boundary-prefit-plan-v1" or
            len(plan.get("successors", ())) != 7 or
            plan.get("new_r5_utility_worker_sha256") !=
            sha(ROOT / "bank_anchored/r5_replay_utility.py")):
        raise PermissionError("diagnostic prefit plan/source differs")
    amendment_pin = pin(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    if (amendment.get("schema") != "pcrl-bank-dated-amendment-v1" or
            "verification" not in amendment.get("affected_stages", ()) or
            amendment.get("source_lock_sha256") != plan["source_lock_sha256"]):
        raise PermissionError("dated verification amendment differs")
    entry = python_entry.absolute()
    if not entry.is_file() or not os.access(entry, os.X_OK):
        raise PermissionError("exact scientific interpreter missing")
    smoke = subprocess.run([str(entry), "-c",
                            "import numpy; import bank_anchored.r5_replay_utility"],
                           cwd=ROOT, capture_output=True, timeout=20, check=False)
    if smoke.returncode:
        raise RuntimeError("exact scientific interpreter cannot import diagnostic")
    private = R5 / ".private"
    if not private.is_dir() or private.stat().st_mode & 0o077:
        raise PermissionError("private diagnostic spec parent is not owner-only")
    worker = pin(ROOT / "bank_anchored/r5_replay_utility.py")
    extra = [pin(path) for path in (
        plan_path, amendment_path, Path(__file__),
        ROOT / "tests/test_bank_r5_utility_boundary.py")]
    indexes = []
    for item in plan["successors"]:
        old_id, new_id = item["source_failed_unit_id"], item["successor_unit_id"]
        old_path = private / f"{old_id}_QUEUE_SPEC.json"
        old_pin = pin(old_path)
        if old_pin["sha256"] != item["source_h_spec_sha256"]:
            raise PermissionError("failed H source spec changed")
        old = json.loads(old_path.read_text())
        queue_receipt_path = QUEUE / old_id / "RECEIPT.json"
        queue_pin = pin(queue_receipt_path)
        queue_receipt = json.loads(queue_receipt_path.read_text())
        if (queue_pin["sha256"] != item["failed_queue_receipt_sha256"] or
                queue_receipt.get("status") != "failed" or
                queue_receipt.get("unit_id") != old_id or
                queue_receipt.get("spec_sha256") != canonical_sha(old)):
            raise PermissionError("failed H queue custody differs")
        old_index_path = R5 / f"{old_id}_SPEC_INDEX.json"
        old_index = json.loads(old_index_path.read_text())
        if (old_index["queue_spec_sha256"] != old_pin["sha256"] or
                old_index["r3_receipt_sha256"] != item["source_r3_receipt_sha256"] or
                old_index["r3_csv_sha256"] != item["source_r3_csv_sha256"] or
                old_index["law_sha256"] != item["source_law_sha256"]):
            raise PermissionError("failed H R3/law source identity differs")
        sources = [*extra, old_pin, queue_pin, pin(old_index_path)]
        spec = successor(old, item, worker, sources,
                         amendment_pin["sha256"], entry)
        spec_path = private / f"{new_id}_QUEUE_SPEC.json"
        spec_sha = _write_once(spec_path, spec, 0o600)
        public = {"schema": "BAR1-r5-boundary-diagnostic-queue-index-v1",
                  "unit_id": new_id, "source_failed_unit_id": old_id,
                  "source_r3_unit_id": old_index["source_r3_unit_id"],
                  "source_law_sha256": old_index["law_sha256"],
                  "failed_queue_receipt_sha256": queue_pin["sha256"],
                  "queue_spec_sha256": spec_sha,
                  "verification_amendment_sha256": amendment_pin["sha256"],
                  "source_count": len(spec["sources"])}
        _write_once(R5 / f"{new_id}_SPEC_INDEX.json", public, 0o644)
        indexes.append(public)
    return {"schema": "BAR1-r5-boundary-diagnostic-spec-receipt-v1",
            "status": "READY_FOR_QUEUE", "source_plan_sha256": plan_pin["sha256"],
            "amendment_sha256": amendment_pin["sha256"],
            "unit_count": len(indexes), "units": indexes}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.amendment, args.python), sort_keys=True))


if __name__ == "__main__":
    main()
