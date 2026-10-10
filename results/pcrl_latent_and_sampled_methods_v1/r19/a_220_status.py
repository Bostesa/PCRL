#!/usr/bin/env python3
"""Propose one sanitized status row for each sealed A producer unit.

The shared R00 ledger remains authoritative. This read-only exporter joins the
220-unit public plan to R03's verified queue receipts and the first-fit public
checkpoint. A completed process with no selected policy does not trigger its
release, Q or attack children.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from queue_unit_status import DEFAULT_STATE, SHA256, UNIT_ID, propose_rows


BASE = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = BASE / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json"
DEFAULT_CHECKPOINT = BASE / "r03" / "A_FIRST_REAL_FIT_CHECKPOINT.json"
DEFAULT_CLOSEOUT = BASE / "r03" / "A_PRODUCER_FIT_CLOSEOUT.json"
AMENDMENT_SHA256 = "526f509f8077f2ee1c412f6d290df939966a6b81f1ae5a87e9639d4bd317271f"
CHECKPOINT_SHA256 = "58ec6fe13369587cf3bbaee94a336ebb162679d56b28970da6bb1a5bed10bb30"
CLOSEOUT_SHA256 = "beb24a699c696adf0c0f7b483c7a1c9e074510393907e05a710d1fac892a9d39"
PARENT = "A-FIT-A_JOINT_STOCH-S0-R1"
FIELDS = ("unit_id", "kind", "arm_or_reference_id", "seed", "fit_unit_id", "status",
          "scientific_status", "charged_cpu_s", "wall_s",
          "sampled_tree_rss_peak_bytes", "queue_receipt_sha256", "fit_receipt_sha256")
NULL_CHILD_KINDS = frozenset(("release", "q_preprocess", "attack"))


def build_rows(plan: dict, queued: dict, checkpoint: dict, closeout: dict, *,
               expected_fit_count: int = 16, expected_null_children: int = 116,
               expected_blocked_children: int = 20) -> list[dict]:
    """Join sealed plan, verified queue rows and all reported fit outcomes."""
    if (checkpoint.get("unit_id") != PARENT or
            checkpoint.get("scientific_status") != "COMPLETE_NO_FEASIBLE_POLICY" or
            checkpoint.get("selected_policy_present") is not False or
            checkpoint.get("feasible_checkpoint_count") != 0 or
            checkpoint.get("checkpoint_count") != 26 or
            checkpoint.get("start_count") != 2):
        raise ValueError("unverified null-policy checkpoint")
    for key in ("private_queue_receipt_sha256", "private_fit_receipt_sha256"):
        if not isinstance(checkpoint.get(key), str) or not SHA256.fullmatch(checkpoint[key]):
            raise ValueError("invalid checkpoint hash")
    parent = queued.get(PARENT)
    if (not isinstance(parent, dict) or parent.get("status") != "done" or
            parent.get("stage") != "a" or
            parent.get("receipt_sha256") != checkpoint["private_queue_receipt_sha256"]):
        raise ValueError("parent queue receipt mismatch")
    if PARENT not in plan or plan[PARENT].get("kind") != "fit":
        raise ValueError("parent absent from sealed plan")
    fit_ids = {uid for uid, spec in plan.items() if spec.get("kind") == "fit"}
    reports = closeout.get("rows")
    if (not isinstance(reports, list) or len(reports) != expected_fit_count or
            len(fit_ids) != expected_fit_count or
            closeout.get("fit_plan_count") != expected_fit_count or
            closeout.get("selected_policy_count") != 0):
        raise ValueError("incomplete fit closeout")
    fit_reports = {record.get("unit_id"): record for record in reports}
    if set(fit_reports) != fit_ids or len(fit_reports) != len(reports):
        raise ValueError("fit closeout IDs mismatch")
    counts = {"done": 0, "failed": 0}
    for uid, record in fit_reports.items():
        q = queued.get(uid)
        if (not isinstance(q, dict) or q.get("stage") != "a" or
                q.get("status") != record.get("queue_status") or
                q.get("receipt_sha256") != record.get("queue_receipt_sha256") or
                record.get("selected_policy_present") is not False or
                not isinstance(record.get("private_fit_receipt_sha256"), str) or
                not SHA256.fullmatch(record["private_fit_receipt_sha256"])):
            raise ValueError("fit closeout receipt mismatch")
        status = record["queue_status"]
        if (status == "done" and record.get("returncode") == 0 and
                record.get("scientific_status") == "COMPLETE_NO_FEASIBLE_POLICY"):
            counts["done"] += 1
        elif (status == "failed" and record.get("returncode") == 1 and
              record.get("scientific_status") == "INCOMPLETE_RESOURCE" and
              plan[uid].get("arm_id") == "A_SEQ21"):
            counts["failed"] += 1
        else:
            raise ValueError("fit closeout status mismatch")
    if (counts["done"] != closeout.get("fit_done_rc0_count") or
            counts["failed"] != closeout.get("fit_failed_rc1_count") or
            (expected_fit_count == 16 and counts != {"done": 14, "failed": 2})):
        raise ValueError("fit closeout count mismatch")
    if (fit_reports[PARENT]["private_fit_receipt_sha256"] !=
            checkpoint["private_fit_receipt_sha256"]):
        raise ValueError("first fit receipt mismatch")

    null_children = {uid for uid, spec in plan.items()
                     if spec.get("fit_unit_id") in fit_reports and
                     fit_reports[spec["fit_unit_id"]]["queue_status"] == "done" and
                     spec.get("kind") in NULL_CHILD_KINDS}
    blocked_children = {uid for uid, spec in plan.items()
                        if spec.get("fit_unit_id") in fit_reports and
                        fit_reports[spec["fit_unit_id"]]["queue_status"] == "failed" and
                        spec.get("kind") in NULL_CHILD_KINDS}
    if (len(null_children) != expected_null_children or
            len(blocked_children) != expected_blocked_children):
        raise ValueError("selected-parent descendant count mismatch")
    if any(uid in queued for uid in null_children | blocked_children):
        raise ValueError("null-parent descendant was queued")

    rows = []
    for uid, spec in sorted(plan.items()):
        identity = spec.get("arm_id", spec.get("reference_id"))
        if (not isinstance(uid, str) or not UNIT_ID.fullmatch(uid) or ".." in uid or
                not isinstance(spec, dict) or not isinstance(spec.get("kind"), str) or
                not isinstance(identity, str) or not UNIT_ID.fullmatch(identity) or
                isinstance(spec.get("seed"), bool) or spec.get("seed") not in (0, 1)):
            raise ValueError("invalid planned unit identity")
        fit_id = spec.get("fit_unit_id", "")
        if fit_id and (not isinstance(fit_id, str) or not UNIT_ID.fullmatch(fit_id)):
            raise ValueError("invalid planned fit identity")
        row = dict.fromkeys(FIELDS, "")
        row.update(unit_id=uid, kind=spec["kind"], arm_or_reference_id=identity,
                   seed=str(spec["seed"]), fit_unit_id=fit_id)
        q = queued.get(uid)
        if uid in null_children:
            row["status"] = "NOT_TRIGGERED_PARENT_NO_FEASIBLE_POLICY"
        elif uid in blocked_children:
            row["status"] = "BLOCKED_PARENT_TECHNICAL_INCOMPLETE"
        elif q is None:
            row["status"] = "PLANNED_NOT_ENQUEUED"
        else:
            if q.get("stage") != "a" or q.get("unit_id") != uid:
                raise ValueError("queue identity mismatch")
            row["status"] = q["status"]
            for key in ("charged_cpu_s", "wall_s", "sampled_tree_rss_peak_bytes"):
                row[key] = q.get(key, "")
            row["queue_receipt_sha256"] = q.get("receipt_sha256", "")
        if uid in fit_reports:
            row["scientific_status"] = fit_reports[uid]["scientific_status"]
            row["fit_receipt_sha256"] = fit_reports[uid]["private_fit_receipt_sha256"]
        rows.append(row)
    return rows


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only A 220-unit status proposal")
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    try:
        amendment_bytes = args.amendment.read_bytes()
        if hashlib.sha256(amendment_bytes).hexdigest() != AMENDMENT_SHA256:
            raise ValueError("A amendment hash mismatch")
        amendment = json.loads(amendment_bytes)
        plan = amendment["unit_plan"]
        if amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or len(plan) != 220:
            raise ValueError("invalid A plan")
        checkpoint_bytes = args.checkpoint.read_bytes()
        if hashlib.sha256(checkpoint_bytes).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("A checkpoint hash mismatch")
        checkpoint = json.loads(checkpoint_bytes)
        closeout_bytes = args.closeout.read_bytes()
        if hashlib.sha256(closeout_bytes).hexdigest() != CLOSEOUT_SHA256:
            raise ValueError("A closeout hash mismatch")
        closeout = json.loads(closeout_bytes)
        state = json.loads(args.state.read_bytes())
        for record in closeout["rows"]:
            if record["queue_status"] == "done":
                fit_output_hashes = state["units"][record["unit_id"]].get("output_sha256", {})
                if record["private_fit_receipt_sha256"] not in fit_output_hashes.values():
                    raise ValueError("fit receipt not bound to queue output")
        queued = {row["unit_id"]: row for row in propose_rows(args.state)
                  if row["record_kind"] == "unit" and row["stage"] == "a"}
        rows = build_rows(plan, queued, checkpoint, closeout)
        if len(rows) != 220:
            raise ValueError("incomplete A rows")
        output = csv_text(rows)
    except Exception:
        print("A plan export rejected: invalid or incomplete evidence", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
