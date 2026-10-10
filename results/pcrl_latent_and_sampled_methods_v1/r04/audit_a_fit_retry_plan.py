"""Independently compare an inert A FIT retry graph with its sealed predecessor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def contains_exact(obj: object, needle: str) -> bool:
    if isinstance(obj, dict):
        return any(contains_exact(k, needle) or contains_exact(v, needle)
                   for k, v in obj.items())
    if isinstance(obj, list):
        return any(contains_exact(v, needle) for v in obj)
    return obj == needle


def audit(campaign: Path) -> dict:
    amendment_path = campaign / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_PATH_CORRECTION.json"
    proposal_path = campaign / "sampled_channel_v1/design/A_FIT_RETRY_UNIT_PLAN_PROPOSAL_001.json"
    failure_path = campaign / "r03/A_FIRST_FIT_PREFIT_FAILURE_RECEIPT.json"
    old_amendment = json.loads(amendment_path.read_text())
    proposal = json.loads(proposal_path.read_text())
    failure = json.loads(failure_path.read_text())
    old_id, new_id = proposal["failed_unit_id"], proposal["retry_unit_id"]
    old, new = old_amendment["unit_plan"], proposal["unit_plan"]
    if (proposal["schema"] != "pcrl-sampled-channel-a-fit-retry-unit-plan-proposal-v1"
            or proposal["status"] != "INERT_PREFIT_ENGINEERING_PROPOSAL_NOT_EFFECTIVE"
            or proposal["predecessor_effective_amendment_sha256"] != sha256(amendment_path)
            or proposal["prior_failed_public_receipt_sha256"] != sha256(failure_path)
            or failure["status"] != "FAILED_BEFORE_ROLE_ACCESS"
            or failure["outputs_written"] != 0
            or failure["unit_id"] != old_id
            or failure["amendment_sha256"] != sha256(amendment_path)
            or old_id == new_id or old_id not in old or old_id in new or new_id in old
            or set(new) != (set(old) - {old_id}) | {new_id}):
        raise ValueError("retry provenance, failed state, or immutable unit IDs differ")
    expected = json.loads(json.dumps(old).replace(f'"{old_id}"', f'"{new_id}"'))
    if new != expected or contains_exact(new, old_id):
        raise ValueError("retry plan has more than the exact immutable FIT ID substitution")
    actual_changed = []
    for key, before in old.items():
        after = new[new_id if key == old_id else key]
        changed = {field for field in set(before) | set(after)
                   if before.get(field) != after.get(field)}
        if changed:
            actual_changed.append(new_id if key == old_id else key)
            if changed != ({"output_id"} if key == old_id
                           else {"deps", "fit_unit_id"}):
                raise ValueError(f"unexpected retry fields: {key} {changed}")
    if (len(actual_changed) != 11
            or sorted(actual_changed) != sorted(proposal["changed_unit_ids"])
            or len(old) != len(new) or len(new) != proposal["unit_count"]):
        raise ValueError("retry affected unit count or list differs")
    remaining, visited = set(new), set()
    while remaining:
        ready = {unit for unit in remaining if set(new[unit].get("deps", ())) <= visited}
        if not ready:
            raise ValueError("retry graph has a missing dependency or cycle")
        remaining -= ready
        visited |= ready
    cpu = sum(unit["cpu_seconds"] for unit in new.values())
    wall = sum(unit["wall_seconds"] for unit in new.values())
    if (cpu != wall or cpu != proposal["cpu_wall_cap_sum_seconds"]
            or cpu != sum(unit["cpu_seconds"] for unit in old.values())
            or wall != sum(unit["wall_seconds"] for unit in old.values())
            or canonical_sha(new) != proposal["unit_plan_canonical_json_sha256"]):
        raise ValueError("retry caps or canonical plan hash differ")
    return {
        "schema": "pcrl-r04-a-fit-retry-plan-independent-audit-v1",
        "status": "INERT_RETRY_PLAN_ID_ONLY_PASS",
        "scope": "inert unit plan only; successor executable sources and full seal require separate audit",
        "predecessor_amendment_sha256": sha256(amendment_path),
        "proposal_sha256": sha256(proposal_path),
        "failed_public_receipt_sha256": sha256(failure_path),
        "failed_unit_status": failure["status"],
        "failed_unit_outputs_written": failure["outputs_written"],
        "old_fit_unit_id": old_id,
        "retry_fit_unit_id": new_id,
        "changed_unit_ids": sorted(actual_changed),
        "old_plan_canonical_sha256": canonical_sha(old),
        "retry_plan_canonical_sha256": canonical_sha(new),
        "units": len(new),
        "kind_counts": dict(Counter(unit["kind"] for unit in new.values())),
        "cpu_cap_sum_seconds": cpu,
        "wall_cap_sum_seconds": wall,
        "topological_closure": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    if args.public_receipt.exists():
        raise FileExistsError(args.public_receipt)
    args.public_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
