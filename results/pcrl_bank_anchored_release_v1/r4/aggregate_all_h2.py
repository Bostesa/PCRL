"""Hash-pin the 36 completed H1 attacks and two P* H2 technical retries.

Every numerical AUC remains a development score on reused INNER rows.
The two failed P* H1 units stay explicit and contribute no result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from .aggregate_h1 import ROOT, _write_once, read_pinned, rows_for_receipt, sha


H2_RECEIPT_SCHEMA = "pcrl-bank-r4-pstar-fresh-attack-h2-v1"
H1_PSTAR_SCHEMA = "pcrl-bank-r4-pstar-fresh-attack-v1"


def _index(path: Path, expected: str, schema: str, count: int) -> dict:
    data = read_pinned(path, expected)
    if data.get("schema") != schema or data.get("unit_count") != count or \
            len(data.get("units", [])) != count:
        raise PermissionError("mixed H1/H2 attack index differs")
    return data


def aggregate(*, h1_index_sha: str, h1_queue_sha: str, h1_lock_sha: str,
              h2_plan_sha: str, h2_index_sha: str, h2_queue_sha: str,
              h2_lock_sha: str) -> dict:
    h1 = _index(ROOT / "ATTACK_UNIT_INDEX_H1.json", h1_index_sha,
                "pcrl-bank-r4-attack-unit-index-v3-h1", 38)
    h1q = _index(ROOT / "ATTACK_QUEUE_INDEX_H1.json", h1_queue_sha,
                 "pcrl-bank-r4-queue-unit-index-v3-h1", 38)
    h1lock = read_pinned(ROOT.parent / "r0" / "ATTACK_STAGE_LOCK_H1.json",
                         h1_lock_sha)
    h2plan = _index(ROOT / "PSTAR_H2_SOURCE_PLAN.json", h2_plan_sha,
                    "pcrl-bank-r4-pstar-h2-source-plan-v1", 2)
    h2 = _index(ROOT / "PSTAR_H2_UNIT_INDEX.json", h2_index_sha,
                "pcrl-bank-r4-pstar-h2-unit-index-v1", 2)
    h2q = _index(ROOT / "PSTAR_H2_QUEUE_INDEX.json", h2_queue_sha,
                 "pcrl-bank-r4-pstar-h2-queue-index-v1", 2)
    h2lock = read_pinned(ROOT.parent / "r0" / "ATTACK_STAGE_LOCK_H2.json",
                         h2_lock_sha)
    if (h1q.get("attack_unit_index_sha256") != h1_index_sha or
            h1q.get("stage_lock_sha256") != h1_lock_sha or
            h1lock.get("units") != h1.get("stage_lock_units") or
            h2plan.get("h1_attack_unit_index_sha256") != h1_index_sha or
            h2plan.get("h1_attack_queue_index_sha256") != h1_queue_sha or
            h2plan.get("h1_stage_lock_sha256") != h1_lock_sha or
            h2.get("h2_source_plan_sha256") != h2_plan_sha or
            h2q.get("attack_unit_index_sha256") != h2_index_sha or
            h2q.get("stage_lock_sha256") != h2_lock_sha or
            h2lock.get("units") != h2.get("stage_lock_units") or
            h1.get("campaign_seal_sha256") != h2.get("campaign_seal_sha256")):
        raise PermissionError("mixed H1/H2 source/lock linkage differs")
    failed = {row["predecessor_unit_id"] for row in h2plan["units"]}
    replacement = {row["successor_unit_id"] for row in h2plan["units"]}
    if (failed != {f"R4_ATTACK_P_STAR_S{seed}_R0_H1" for seed in (0, 1)} or
            replacement != {f"R4_ATTACK_P_STAR_S{seed}_R0_H2" for seed in (0, 1)} or
            {row["unit_id"] for row in h2["units"]} != replacement or
            {row["unit_id"] for row in h1["units"] if row["unit_id"] not in failed} !=
            {row["unit_id"] for row in h1["units"] if row["arm_id"] != "P_STAR"}):
        raise PermissionError("only the two failed P* units may be H2 successors")
    all_rows, units = [], []
    for index, lock_sha, source_kind in ((h1, h1_lock_sha, "H1"),
                                         (h2, h2_lock_sha, "H2_PSTAR")):
        for entry in index["units"]:
            uid = entry["unit_id"]
            if uid in failed:
                continue
            path = ROOT / f"{uid}_RECEIPT.json"
            if not path.is_file():
                raise FileNotFoundError(f"planned completed attack receipt absent: {uid}")
            digest = sha(path)
            receipt = read_pinned(path, digest)
            if source_kind == "H2_PSTAR":
                if receipt.get("schema") != H2_RECEIPT_SCHEMA:
                    raise PermissionError("P* H2 corrected receipt schema differs")
                # Reuse the numerical-field checker; this local copy changes
                # only its expected schema label after exact H2 admission.
                common = receipt | {"schema": H1_PSTAR_SCHEMA}
            else:
                common = receipt
            rows, unit = rows_for_receipt(common, entry, lock_sha, digest)
            for row in rows:
                row["source_kind"] = source_kind
            unit["source_kind"] = source_kind
            all_rows.extend(rows)
            units.append(unit)
    if (len(units) != 38 or len(all_rows) != 114 or
            len({u["unit_id"] for u in units}) != 38 or
            sum(u["source_kind"] == "H2_PSTAR" for u in units) != 2):
        raise AssertionError("mixed H1/H2 aggregate is not exactly 38 three-view units")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(all_rows)
    csv_sha = _write_once(ROOT / "ATTACK_AUC_ALL_H2.csv",
                          stream.getvalue().encode())
    summary = {
        "schema": "pcrl-bank-r4-mixed-h1-h2-public-attack-aggregate-v1",
        "status": "COMPLETE_DEVELOPMENT_RECEIPTS_ONLY",
        "source_sha256": {"h1_index": h1_index_sha, "h1_queue": h1_queue_sha,
                          "h1_lock": h1_lock_sha, "h2_plan": h2_plan_sha,
                          "h2_index": h2_index_sha, "h2_queue": h2_queue_sha,
                          "h2_lock": h2_lock_sha},
        "attack_auc_csv_sha256": csv_sha,
        "unit_count": 38, "view_rows": 114,
        "completed_h1_units": 36, "completed_h2_pstar_units": 2,
        "failed_h1_pstar_unit_ids": sorted(failed),
        "all_control_gates_pass": all(row["all_control_gates_pass"]
                                      for row in all_rows),
        "units": units,
        "scope": "Exploratory ATT-fitted, INNER-scored recovery. H2 retries were selected after H1 technical failures; no independent confirmation.",
    }
    blob = (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    summary_sha = _write_once(ROOT / "ATTACK_AGGREGATE_ALL_H2.json", blob)
    return {"status": summary["status"], "unit_count": 38,
            "view_rows": 114, "csv_sha256": csv_sha,
            "aggregate_sha256": summary_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("h1-index-sha", "h1-queue-sha", "h1-lock-sha",
                 "h2-plan-sha", "h2-index-sha", "h2-queue-sha", "h2-lock-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = aggregate(h1_index_sha=args.h1_index_sha,
                       h1_queue_sha=args.h1_queue_sha,
                       h1_lock_sha=args.h1_lock_sha,
                       h2_plan_sha=args.h2_plan_sha,
                       h2_index_sha=args.h2_index_sha,
                       h2_queue_sha=args.h2_queue_sha,
                       h2_lock_sha=args.h2_lock_sha)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
