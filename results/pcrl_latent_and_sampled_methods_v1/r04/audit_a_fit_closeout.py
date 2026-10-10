"""Hash-only independent linkage of the 16 frozen A FIT queue outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .independent_a_source import sha_file


def run(campaign: Path) -> dict:
    producer_path = campaign / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json"
    closeout_path = campaign / "r03/A_PRODUCER_FIT_CLOSEOUT.json"
    state_path = campaign / "r03/.private/compute/state.json"
    producer = json.loads(producer_path.read_text())
    closeout = json.loads(closeout_path.read_text())
    units = json.loads(state_path.read_text())["units"]
    planned = {uid: entry for uid, entry in producer["unit_plan"].items()
               if entry.get("kind") == "fit"}
    rows = closeout["rows"]
    if (len(planned) != 16 or len(rows) != 16
            or closeout.get("effective_a_amendment_sha256") != sha_file(producer_path)
            or closeout.get("fit_plan_count") != 16
            or closeout.get("fit_done_rc0_count") != 14
            or closeout.get("fit_failed_rc1_count") != 2
            or closeout.get("selected_policy_count") != 0
            or set(row["unit_id"] for row in rows) != set(planned)):
        raise ValueError("A frozen FIT closeout or 16-unit registry differs")
    counts = {"complete_null": 0, "seq21_stage1_stop": 0}
    checked = []
    for row in rows:
        uid = row["unit_id"]
        plan = planned[uid]
        unit = units[uid]
        queue_path = campaign / "r03/.private/compute/units" / uid / "RECEIPT.json"
        queue = json.loads(queue_path.read_text())
        fit_path = campaign / "r03/.private/a_units" / uid / "FIT_RECEIPT.json"
        fit = json.loads(fit_path.read_text())
        if (row["arm_id"] != plan["arm_id"] or row["seed"] != plan["seed"]
                or fit["arm_id"] != row["arm_id"] or fit["source_seed"] != row["seed"]
                or fit["unit_id"] != uid
                or fit["a_amendment_sha256"] != sha_file(producer_path)
                or row["queue_receipt_sha256"] != sha_file(queue_path)
                or row["private_fit_receipt_sha256"] != sha_file(fit_path)
                or queue["unit_id"] != uid
                or queue["spec_sha256"] != unit["spec_sha256"]
                or queue["status"] != row["queue_status"] != unit["status"]
                or queue["returncode"] != row["returncode"] != unit["returncode"]
                or fit["status"] != row["scientific_status"]
                or row["selected_policy_present"] is not False
                or any((fit_path.parent / name).exists() for name in
                       ("policy.npz", "POLICY_RECEIPT.json"))):
            raise ValueError(f"A FIT queue, private receipt or selection differs: {uid}")
        if row["returncode"] == 0:
            if (fit["status"] != "COMPLETE_NO_FEASIBLE_POLICY"
                    or unit["output_sha256"].get(str(fit_path.resolve())) != sha_file(fit_path)
                    or (fit.get("selected", "absent") is not None if
                        fit["schema"] == "SCV1-private-stochastic-fit-v1" else
                        fit.get("fit", {}).get("complete") is not True or
                        fit.get("fit", {}).get("success") is not False)):
                raise ValueError(f"A completed no-selection FIT differs: {uid}")
            counts["complete_null"] += 1
        elif row["arm_id"] == "A_SEQ21" and row["returncode"] == 1:
            stages = fit["fit"]
            starts = stages["stage1_starts"]
            if (fit["status"] != "INCOMPLETE_RESOURCE" or unit["output_sha256"]
                    or queue.get("timed_out") is not False
                    or queue.get("resource_limit") is not None
                    or stages.get("complete") is not False
                    or stages.get("stage1_complete") is not True
                    or len(starts) != 2 or len(stages["stage2_starts"]) != 0
                    or any(start["completed"] is not True or start["updates"] != 150
                           or len(start["checkpoints"]) != 7 or
                           any(check["feasible"] is not False for check in start["checkpoints"])
                           for start in starts)
                    or (fit_path.parent / "SEQ_FIRST_FIT_DRAW.json").exists()):
                raise ValueError(f"A SEQ21 stage-one stop differs: {uid}")
            counts["seq21_stage1_stop"] += 1
        else:
            raise ValueError(f"unexpected A FIT queue outcome: {uid}")
        checked.append({"unit_id": uid, "arm_id": row["arm_id"], "seed": row["seed"],
                        "queue_status": queue["status"], "returncode": queue["returncode"],
                        "fit_status": fit["status"],
                        "queue_receipt_sha256": sha_file(queue_path),
                        "fit_receipt_sha256": sha_file(fit_path)})
    if counts != {"complete_null": 14, "seq21_stage1_stop": 2}:
        raise ValueError("A 16 FIT outcome counts differ")
    return {"schema": "pcrl-r04-a-fit-closeout-linkage-v1",
            "status": "ALL_16_A_FIT_STATUS_LINKAGE_PASS",
            "scope": "hash-only queue/FIT aggregate status; no role arrays or checkpoint weights",
            "producer_amendment_sha256": sha_file(producer_path),
            "r03_closeout_sha256": sha_file(closeout_path),
            "planned_fit_count": 16, "selected_policy_count": 0,
            "complete_null_count": 14, "stage1_stop_failed_count": 2,
            "stage1_stop_is_full_no_feasible_fit": False,
            "global_infeasibility_claim": False,
            "rows": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args()
    output = args.public_receipt
    if output.exists():
        raise FileExistsError(output)
    result = run(args.campaign.resolve(strict=True))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
