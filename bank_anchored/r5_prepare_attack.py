"""Prepare independent attack AUC/view replay specs from completed H1 receipts.

Only receipt and source bytes are read here. R5 role labels and predictions open
later inside each admitted verification queue unit.
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
R4 = CAMP / "r4"
R5 = CAMP / "r5"
QSTATE = CAMP / "r0/.private/compute/state.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("R5 attack replay source symlink refused")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise PermissionError("R5 attack replay source is not a regular file")
    return {"path": str(resolved), "sha256": sha(resolved)}


def write_once(path: Path, payload: dict, mode: int) -> str:
    raw = (json.dumps(payload, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.read_bytes() != raw:
            raise FileExistsError("R5 attack replay source/output bytes differ")
        return sha(path)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode), "wb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(raw).hexdigest()


def build(unit_id: str, amendment_path: Path, python_entry: Path) -> dict:
    worker_index_path = R4 / "ATTACK_UNIT_INDEX_H1.json"
    queue_index_path = R4 / "ATTACK_QUEUE_INDEX_H1.json"
    worker_index_pin, queue_index_pin = pin(worker_index_path), pin(queue_index_path)
    worker_index = json.loads(worker_index_path.read_text())
    queue_index = json.loads(queue_index_path.read_text())
    if (worker_index.get("schema") != "pcrl-bank-r4-attack-unit-index-v3-h1" or
            queue_index.get("schema") != "pcrl-bank-r4-queue-unit-index-v3-h1" or
            queue_index.get("attack_unit_index_sha256") != worker_index_pin["sha256"] or
            worker_index.get("unit_count") != 38 or
            queue_index.get("unit_count") != 38):
        raise PermissionError("H1 attack index/queue family differs")
    workers = {item["unit_id"]: item for item in worker_index["units"]}
    queues = {item["unit_id"]: item for item in queue_index["units"]}
    if set(workers) != set(queues) or unit_id not in workers:
        raise PermissionError("H1 unit is absent or duplicated")
    state = json.loads(QSTATE.read_text())
    if state["units"].get(unit_id, {}).get("status") != "done":
        raise PermissionError("H1 attack fit has not completed under shared queue")
    worker = workers[unit_id]
    queued = queues[unit_id]
    worker_path = R4 / ".private/units" / f"{unit_id}.json"
    queue_path = R4 / ".private/queue_units" / f"{unit_id}.json"
    receipt_path = R4 / f"{unit_id}_RECEIPT.json"
    worker_pin, queue_pin, receipt_pin = map(pin,
                                             (worker_path, queue_path, receipt_path))
    if (worker_pin["sha256"] != worker["spec_sha256"] or
            worker_pin["sha256"] != queued["attack_spec_sha256"] or
            queue_pin["sha256"] != queued["queue_spec_sha256"]):
        raise PermissionError("H1 indexed unit spec hashes differ")
    source = json.loads(worker_path.read_text())
    queue = json.loads(queue_path.read_text())
    public = json.loads(receipt_path.read_text())
    if (source.get("unit_id") != unit_id or queue.get("unit_id") != unit_id or
            queue.get("stage") != "attacks" or
            queue.get("amendment_sha256") != queue_index["stage_lock_sha256"] or
            public.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
            public.get("unit_id") != unit_id or
            public.get("unit_spec_sha256") != worker_pin["sha256"] or
            public.get("campaign_seal_sha256") != source.get("campaign_seal_sha256") or
            (public.get("arm_id"), public.get("source_seed"),
             public.get("realization")) !=
            (worker["arm_id"], worker["source_seed"], worker["realization"])):
        raise PermissionError("completed H1 public receipt or queue identity differs")
    for item in queue["sources"]:
        if pin(Path(item["path"])) != item:
            raise PermissionError("H1 queue source changed before R5 replay")
    amendment_pin = pin(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    latest = [item["sha256"] for item in state["amendments"]
              if "verification" in item["stages"]][-1]
    if (amendment.get("schema") != "pcrl-bank-dated-amendment-v1" or
            "verification" not in amendment.get("affected_stages", ()) or
            amendment.get("source_lock_sha256") != source["campaign_seal_sha256"] or
            latest != amendment_pin["sha256"]):
        raise PermissionError("latest R5 verification amendment differs")
    entry = python_entry.absolute()
    if not entry.is_file() or not os.access(entry, os.X_OK):
        raise PermissionError("exact numerical Python entrypoint missing")
    smoke = subprocess.run(
        [str(entry), "-c", "import numpy, scipy, sklearn; import bank_anchored.r5_replay_attack"],
        cwd=ROOT, capture_output=True, text=True, timeout=20, check=False)
    if smoke.returncode:
        raise RuntimeError("exact numerical Python cannot import R5 attack replay")
    private = R5 / ".private"
    if not private.is_dir() or private.stat().st_mode & 0o077:
        raise PermissionError("R5 replay spec parent is not owner-only")
    rid = f"R5_REPLAY_{unit_id}"
    output = R5 / f"{rid}_RECEIPT.json"
    files = [*queue["sources"], worker_pin, queue_pin, receipt_pin,
             worker_index_pin, queue_index_pin, amendment_pin]
    files.extend(pin(path) for path in (
        Path(__file__), ROOT / "bank_anchored/r5_replay_attack.py",
        ROOT / "bank_anchored/r5_view_replay.py",
        ROOT / "bank_anchored/r5_pstar_view.py",
        ROOT / "bank_anchored/r5_independent.py",
        CAMP / "r0/compute_queue.py"))
    dedup = {}
    for item in files:
        if item["path"] in dedup and dedup[item["path"]] != item:
            raise PermissionError("R5 replay source path has conflicting bytes")
        dedup[item["path"]] = item
    spec = {"unit_id": rid, "stage": "verification",
            "amendment_sha256": amendment_pin["sha256"],
            "argv": [str(entry), "-m", "bank_anchored.r5_replay_attack",
                     "--r4-spec", str(worker_path.resolve()),
                     "--r4-spec-sha256", worker_pin["sha256"],
                     "--r4-receipt", str(receipt_path.resolve()),
                     "--r4-receipt-sha256", receipt_pin["sha256"],
                     "--output", str(output)],
            "cwd": str(ROOT), "env": {}, "deps": [unit_id],
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 2.0,
            "projected_output_bytes": 100000,
            "outputs": [str(output)], "sources": list(dedup.values())}
    spec_sha = write_once(private / f"{rid}_QUEUE_SPEC.json", spec, 0o600)
    index = {"schema": "BAR1-r5-attack-replay-unit-index-v1", "unit_id": rid,
             "source_r4_unit_id": unit_id, "arm_id": source["arm_id"],
             "source_seed": source["source_seed"],
             "realization": source["realization"],
             "r4_receipt_sha256": receipt_pin["sha256"],
             "r4_worker_spec_sha256": worker_pin["sha256"],
             "r4_queue_spec_sha256": queue_pin["sha256"],
             "queue_spec_sha256": spec_sha,
             "verification_amendment_sha256": amendment_pin["sha256"],
             "source_count": len(dedup)}
    write_once(R5 / f"{rid}_SPEC_INDEX.json", index, 0o644)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--r4-unit-id", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.r4_unit_id, args.amendment, args.python),
                     sort_keys=True))


if __name__ == "__main__":
    main()
