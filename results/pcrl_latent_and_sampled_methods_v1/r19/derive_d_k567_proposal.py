#!/usr/bin/env python3
"""Derive R19 public D interval proposals from R04's independent receipt.

Reads public aggregate CSV/JSON only. Writes only the three R19 proposal CSVs.
The signed attack contrasts remain raw; their direction is not interpreted as
a universal candidate-versus-control test.
"""

import csv
import hashlib
import io
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RECEIPT = HERE.parent / "r04" / "D_567_INDEPENDENT_INTERVALS.json"
RECEIPT_SHA = "3c8ad07fbedcc62044c538451f2ffd4292f0b95088120fc9c26206c02c7d5efa"
ARM_CODE = {"T": "D_T", "D-K": "D_K", "L": "D_L", "S12": "D_S12",
            "S21": "D_S21", "C": "D_C", "F": "D_F"}
TASK_FILE = HERE / "D_PRIMARY_TASK_ENDPOINTS_PROPOSAL.csv"
RECOVERY_FILE = HERE / "D_PRIMARY_RECOVERY_ENDPOINTS_PROPOSAL.csv"
CLAUSE_FILE = HERE / "D_K567_CLAUSES_PROPOSAL.csv"


def read_csv(path):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames), list(reader)


def write_csv(path, fields, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def utility_status(clause, metric):
    if metric == "accuracy":
        return ("PASS" if clause["lower"] >= -0.010 else
                "VIOLATION" if clause["upper"] < -0.010 else "UNRESOLVED")
    cap = 0.010 if metric == "nll" else 0.005
    return ("PASS" if clause["upper"] <= cap else
            "VIOLATION" if clause["lower"] > cap else "UNRESOLVED")


def main():
    raw = RECEIPT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("R04 interval receipt hash changed")
    receipt = json.loads(raw)
    if (receipt.get("status") != "FIXED_567_INTERVALS_COMPUTED" or
            receipt.get("clause_count") != 567 or len(receipt.get("clauses", [])) != 567 or
            receipt.get("point_replay", {}).get("s0", {}).get("status") != "POINT_PARITY_PASS" or
            receipt.get("point_replay", {}).get("s1", {}).get("status") != "POINT_PARITY_PASS"):
        raise ValueError("R04 interval receipt has an unexpected scope")
    clauses = {c["name"]: c for c in receipt["clauses"]}
    if len(clauses) != 567:
        raise ValueError("duplicate clause names")

    task_fields, task_rows = read_csv(TASK_FILE)
    if len(task_rows) != 56:
        raise ValueError("unexpected task row count")
    new_task_fields = [f for f in task_fields if f not in
                       ("source_interval_receipt_sha256", "accuracy_delta_lower",
                        "accuracy_delta_upper", "accuracy_delta_interval_status",
                        "nll_excess_lower", "nll_excess_upper", "nll_excess_interval_status",
                        "brier_excess_lower", "brier_excess_upper", "brier_excess_interval_status")]
    new_task_fields += ["accuracy_delta_lower", "accuracy_delta_upper", "accuracy_delta_interval_status",
                        "nll_excess_lower", "nll_excess_upper", "nll_excess_interval_status",
                        "brier_excess_lower", "brier_excess_upper", "brier_excess_interval_status",
                        "source_interval_receipt_sha256"]
    for row in task_rows:
        arm = ARM_CODE[row["arm"]]
        seed = int(row["seed"])
        statuses = []
        for metric, stem, observed in (("accuracy", "accuracy_delta", -float(row["accuracy_drop"])),
                                       ("nll", "nll_excess", float(row["nll_excess"])),
                                       ("brier", "brier_excess", float(row["brier_excess"]))):
            name = f"s{seed}:utility:{arm}:{row['task']}:{row['reference']}:{metric}"
            clause = clauses[name]
            if not math.isclose(observed, clause["point"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("task point and independent interval disagree")
            status = utility_status(clause, metric)
            row[f"{stem}_lower"] = str(clause["lower"])
            row[f"{stem}_upper"] = str(clause["upper"])
            row[f"{stem}_interval_status"] = status
            statuses.append(status)
        row["simultaneous_interval_status"] = ("VIOLATION" if "VIOLATION" in statuses else
                                               "PASS" if all(s == "PASS" for s in statuses) else
                                               "UNRESOLVED")
        row["source_interval_receipt_sha256"] = RECEIPT_SHA

    recovery_fields, recovery_rows = read_csv(RECOVERY_FILE)
    if len(recovery_rows) != 42:
        raise ValueError("unexpected recovery row count")
    new_recovery_fields = [f for f in recovery_fields if f != "source_interval_receipt_sha256"]
    new_recovery_fields += ["source_interval_receipt_sha256"]
    for row in recovery_rows:
        if row["arm"] not in ARM_CODE or row["view"] not in ("r1", "r2", "pair"):
            raise ValueError("unexpected recovery row identity")
        row["simultaneous_interval_status"] = "NO_ABSOLUTE_AUC_CI_SIGNED_K567_COMPUTED"
        row["source_interval_receipt_sha256"] = RECEIPT_SHA

    clause_fields = ["name", "aggregate", "kind", "point", "lower", "upper", "standard_error",
                     "numerically_constant", "utility_boundary", "utility_interval_status",
                     "source_interval_receipt_sha256"]
    clause_rows = []
    for clause in receipt["clauses"]:
        name = clause["name"]
        aggregate, kind, *_ = name.split(":")
        if aggregate not in ("s0", "s1", "mean") or kind not in ("utility", "arm_pair", "u_anchor"):
            raise ValueError("unexpected clause identity")
        metric = name.rsplit(":", 1)[1]
        if kind == "utility":
            if metric not in ("accuracy", "nll", "brier"):
                raise ValueError("unexpected utility metric")
            boundary = "lower>=-0.010" if metric == "accuracy" else (
                "upper<=0.010" if metric == "nll" else "upper<=0.005")
            status = utility_status(clause, metric)
        else:
            boundary = "SIGNED_RAW_CONTRAST"
            status = "NOT_APPLICABLE_WITHOUT_DIRECTION"
        clause_rows.append(dict(name=name, aggregate=aggregate, kind=kind,
                                point=clause["point"], lower=clause["lower"], upper=clause["upper"],
                                standard_error=clause["se"], numerically_constant=clause["numerically_constant"],
                                utility_boundary=boundary, utility_interval_status=status,
                                source_interval_receipt_sha256=RECEIPT_SHA))

    write_csv(TASK_FILE, new_task_fields, task_rows)
    write_csv(RECOVERY_FILE, new_recovery_fields, recovery_rows)
    write_csv(CLAUSE_FILE, clause_fields, clause_rows)


if __name__ == "__main__":
    main()
