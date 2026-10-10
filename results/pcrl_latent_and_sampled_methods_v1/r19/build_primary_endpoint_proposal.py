#!/usr/bin/env python3
"""Build a D-only instance of the proposed shared endpoint table.

The table is a tagged union: absolute task rows, absolute recovery rows, and
raw signed simultaneous contrast rows. A is deliberately absent until its
stage results exist; its superseding seven-reference 6,885 IDs fit the schema.
"""

import csv
import hashlib
import io
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
R02 = HERE.parent / "r02"
INPUTS = {
    "task": (HERE / "D_PRIMARY_TASK_ENDPOINTS_PROPOSAL.csv",
             "41029367fe51f0eb1672cd4d49a51446417d14be6b3886f08f4896d06f912b7e"),
    "recovery": (HERE / "D_PRIMARY_RECOVERY_ENDPOINTS_PROPOSAL.csv",
                 "27d4be956660a273116a029cd78bd1ffe9b45b36bfcb379a76fa8813a184c5b4"),
    "clauses": (HERE / "D_K567_CLAUSES_PROPOSAL.csv",
                "3501b7f1fa268bad28bb55f87cd9bf92e773f602b3147a4866e0ee54a4c35877"),
}
D_ORDER_SHA = "32bc44199da6035c0afa226dbda78d0728ce51f2921a51a0ef2fe5cd6e9573b1"
ARM = {"T": "D_T", "D-K": "D_K", "L": "D_L", "S12": "D_S12",
       "S21": "D_S21", "C": "D_C", "F": "D_F"}
FIELDS = (
    "schema_version", "record_id", "record_kind", "family", "service_contract",
    "comparison_scope", "arm_id", "other_id", "summary_id", "seed_id", "draw_id",
    "task", "reference_id", "view", "metric", "contrast_kind", "contrast_convention",
    "accuracy", "nll", "brier", "accuracy_delta", "nll_excess", "brier_excess",
    "sex_auc", "reader_id", "point_estimate", "interval_lower", "interval_upper",
    "interval_status", "task_reference_point_status", "all_task_point_status",
    "source_point_receipt_sha256", "source_interval_receipt_sha256",
)
SCHEMA = "pcrl-primary-endpoints-unified-proposal-v1"
OUTPUT = HERE / "PRIMARY_ENDPOINTS_D_UNIFIED_PROPOSAL.csv"


def input_rows(which):
    path, expected = INPUTS[which]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{which} input hash changed")
    return list(csv.DictReader(io.StringIO(raw.decode())))


def base(record_id, record_kind, arm_id=""):
    row = dict.fromkeys(FIELDS, "")
    row.update(schema_version=SCHEMA, record_id=record_id, record_kind=record_kind,
               family="D", arm_id=arm_id)
    return row


def service(arm_id):
    return "D_CONTINUOUS_LATENT_16_EACH" if arm_id == "D_F" else "D_HARD_CODE_10_BIT"


def point_status(value):
    if value == "true":
        return "PASS_POINT"
    if value == "false":
        return "FAIL_POINT"
    raise ValueError("invalid point status")


def main():
    tasks = input_rows("task")
    recoveries = input_rows("recovery")
    clauses = input_rows("clauses")
    if (len(tasks), len(recoveries), len(clauses)) != (56, 42, 567):
        raise ValueError("D endpoint row count changed")
    order_path = R02 / "D_FAMILY_ORDER.json"
    if hashlib.sha256(order_path.read_bytes()).hexdigest() != D_ORDER_SHA:
        raise ValueError("D family order hash changed")
    ordered_ids = json.loads(order_path.read_text())["ordered_ids"]
    if [c["name"] for c in clauses] != ordered_ids:
        raise ValueError("D clause order differs from sealed order")

    rows = []
    for source in tasks:
        arm_id = ARM[source["arm"]]
        seed = source["seed"]
        if seed not in ("0", "1"):
            raise ValueError("unexpected D seed")
        row = base(f"D:absolute_task:{arm_id}:s{seed}:{source['task']}:{source['reference']}",
                   "absolute_task", arm_id)
        row.update(service_contract=service(arm_id), summary_id=f"s{seed}", seed_id=seed,
                   task=source["task"], reference_id=source["reference"],
                   accuracy=source["accuracy"], nll=source["nll"], brier=source["brier"],
                   accuracy_delta=str(-float(source["accuracy_drop"])),
                   nll_excess=source["nll_excess"], brier_excess=source["brier_excess"],
                   interval_status=source["simultaneous_interval_status"],
                   task_reference_point_status=point_status(source["task_reference_point_pass"]),
                   all_task_point_status=point_status(source["all_task_reference_point_pass"]),
                   source_point_receipt_sha256=source["source_receipt_sha256"],
                   source_interval_receipt_sha256=source["source_interval_receipt_sha256"])
        rows.append(row)

    for source in recoveries:
        arm_id = ARM[source["arm"]]
        seed = source["seed"]
        if seed not in ("0", "1") or source["view"] not in ("r1", "r2", "pair"):
            raise ValueError("unexpected D recovery identity")
        row = base(f"D:absolute_recovery:{arm_id}:s{seed}:{source['view']}",
                   "absolute_recovery", arm_id)
        row.update(service_contract=service(arm_id), summary_id=f"s{seed}", seed_id=seed,
                   view=source["view"], metric="sex_auc", sex_auc=source["sex_auc"],
                   reader_id=source["reader_id"], interval_status=source["simultaneous_interval_status"],
                   all_task_point_status=point_status(source["all_task_reference_point_pass"]),
                   source_point_receipt_sha256=source["source_receipt_sha256"],
                   source_interval_receipt_sha256=source["source_interval_receipt_sha256"])
        rows.append(row)

    for source in clauses:
        name = source["name"]
        parts = name.split(":")
        summary, kind = parts[:2]
        if summary not in ("s0", "s1", "mean") or kind not in ("utility", "arm_pair", "u_anchor"):
            raise ValueError("unexpected D clause")
        arm_id = parts[2]
        if arm_id not in ARM.values():
            raise ValueError("unexpected D arm")
        row = base(f"D:signed_contrast:{name}", "signed_contrast", arm_id)
        row.update(summary_id=summary, seed_id=(summary[1] if summary in ("s0", "s1") else ""),
                   contrast_kind=kind, metric=parts[-1], point_estimate=source["point"],
                   interval_lower=source["lower"], interval_upper=source["upper"],
                   interval_status=(source["utility_interval_status"] if kind == "utility" else
                                    "RAW_SIGNED_INTERVAL_UNINTERPRETED"),
                   source_interval_receipt_sha256=source["source_interval_receipt_sha256"])
        if kind == "utility":
            if len(parts) != 6:
                raise ValueError("invalid utility clause")
            row.update(service_contract=service(arm_id), task=parts[3], reference_id=parts[4],
                       other_id=parts[4], contrast_convention="candidate_minus_reference")
        else:
            if len(parts) != 6:
                raise ValueError("invalid recovery clause")
            other_id, view = parts[3:5]
            row.update(other_id=other_id, view=view, contrast_convention="first_minus_second")
            if kind == "u_anchor":
                row.update(service_contract="D_ARM_VS_U_REFERENCE", comparison_scope="ARM_VS_U_ANCHOR")
            else:
                row.update(service_contract=("D_CROSS_CAPACITY_DIAGNOSTIC" if "D_F" in (arm_id, other_id)
                                             else "D_HARD_CODE_10_BIT"),
                           comparison_scope=("HARD_VS_CONTINUOUS" if "D_F" in (arm_id, other_id)
                                             else "HARD_VS_HARD"))
        rows.append(row)

    if len(rows) != 665 or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate or missing D records")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    OUTPUT.write_text(stream.getvalue())


if __name__ == "__main__":
    main()
