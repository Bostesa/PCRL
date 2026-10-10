"""Plan H1 attack successors without creating a worker spec or opening data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


SCHEMA = "pcrl-bank-r4-attack-h1-source-plan-v1"


def build_plan(attack: Mapping, queue: Mapping, lock: Mapping, *,
               attack_sha: str, queue_sha: str, lock_sha: str) -> dict:
    if (attack.get("schema") != "pcrl-bank-r4-attack-unit-index-v2" or
            queue.get("schema") != "pcrl-bank-r4-queue-unit-index-v2" or
            lock.get("schema") != "pcrl-bank-attack-stage-lock-v1" or
            attack.get("unit_count") != len(attack.get("units", [])) or
            attack.get("planned_fits") != 30 * attack["unit_count"] or
            queue.get("unit_count") != attack["unit_count"] or
            queue.get("attack_unit_index_sha256") != attack_sha or
            queue.get("stage_lock_sha256") != lock_sha or
            lock.get("campaign_seal_sha256") !=
            attack.get("campaign_seal_sha256") or
            lock.get("units") != attack.get("stage_lock_units")):
        raise PermissionError("old attack index/queue/stage lock differ")
    old_queue = {item["unit_id"]: item for item in queue["units"]}
    if len(old_queue) != len(queue["units"]):
        raise PermissionError("old queue index repeats a unit")
    rows = []
    for item in attack["units"]:
        uid = item["unit_id"]
        q = old_queue.get(uid)
        lock_row = attack["stage_lock_units"].get(uid)
        if (q is None or q.get("attack_spec_sha256") != item.get("spec_sha256") or
                item.get("planned_fits") != 30 or
                lock_row != {"spec_sha256": item["spec_sha256"],
                             "arm_id": item["arm_id"],
                             "source_seed": item["source_seed"],
                             "realization": item["realization"],
                             "point_eligible": True}):
            raise PermissionError("old queued unit differs from frozen science")
        seed, draw = item["source_seed"], item["realization"]
        rows.append({"predecessor_unit_id": uid,
                     "successor_unit_id": uid + "_H1",
                     "arm_id": item["arm_id"], "source_seed": seed,
                     "realization": draw,
                     "attacker_seed": 100000 + 1000 * seed + draw,
                     "predecessor_worker_spec_sha256": item["spec_sha256"],
                     "predecessor_queue_spec_sha256": q["queue_spec_sha256"],
                     "planned_fits": 30})
    if (len({r["predecessor_unit_id"] for r in rows}) != len(rows) or
            len({r["successor_unit_id"] for r in rows}) != len(rows)):
        raise PermissionError("old or successor unit IDs repeat")
    return {"schema": SCHEMA, "status": "PLAN_ONLY_NO_H1_WORKER_SPEC",
            "campaign_seal_sha256": attack["campaign_seal_sha256"],
            "predecessor_attack_unit_index_sha256": attack_sha,
            "predecessor_attack_queue_index_sha256": queue_sha,
            "predecessor_stage_lock_sha256": lock_sha,
            "successor_suffix": "H1", "unit_count": len(rows),
            "planned_fits": 30 * len(rows), "units": rows}


def _read(path: str, sha: str) -> dict:
    p = Path(path)
    if (p.is_symlink() or not p.resolve(strict=True).is_file() or
            hashlib.sha256(p.read_bytes()).hexdigest() != sha):
        raise PermissionError("predecessor source bytes differ")
    return json.loads(p.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("old-index", "old-index-sha256", "old-queue",
                 "old-queue-sha256", "old-lock", "old-lock-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    attack = _read(args.old_index, args.old_index_sha256)
    queue = _read(args.old_queue, args.old_queue_sha256)
    lock = _read(args.old_lock, args.old_lock_sha256)
    plan = build_plan(attack, queue, lock,
                      attack_sha=args.old_index_sha256,
                      queue_sha=args.old_queue_sha256,
                      lock_sha=args.old_lock_sha256)
    if plan["unit_count"] != 38 or plan["planned_fits"] != 1140:
        raise PermissionError("the H1 correction must cover all 38 old units")
    out = Path(args.output)
    if out.parent.is_symlink() or out.exists() or ".private" in out.parts:
        raise PermissionError("H1 source plan must be a new public file")
    blob = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"plan_sha256": hashlib.sha256(blob).hexdigest(),
                      "unit_count": plan["unit_count"],
                      "status": plan["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
