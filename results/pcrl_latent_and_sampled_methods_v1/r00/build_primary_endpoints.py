"""Integrate independently checked public D and fixed-A endpoint aggregates.

This is a public-table join. It does not read private people, fit models, choose
attacks or recompute the statistical intervals.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
D_SOURCE = ROOT / "r19/PRIMARY_ENDPOINTS_D_UNIFIED_PROPOSAL.csv"
A_POINT = ROOT / "r02/A_FIXED_POINT_ROWS_R2.csv"
A_INTERVAL = ROOT / "r02/A_K6885_FIXED_INFERENCE_R2.json"
A_RECEIPT = ROOT / "r02/A_FIXED_ENDPOINT_RECEIPT_R2.json"
A_REPLAY = ROOT / "r04/A_FIXED_ENDPOINT_INDEPENDENT_REPLAY.json"
OUT = ROOT / "PRIMARY_ENDPOINTS.csv"
EXPECTED = {
    D_SOURCE: "4ec0adda1dc9aa83df64093e51cfdbcda842730820e32d1ed94da1820d68cd12",
    A_POINT: "1941d5614de9a97f55a22b3c83cbc0a01eced155525c9832659821c8e6762daa",
    A_INTERVAL: "8147ac65865ea6fd98c677a84da4edeef4fb6df6e18c72cdc0e2ff45706ef784",
    A_RECEIPT: "98917c98b807b7961174378086008b2fd8542be9822ed495c9e9e922cb1bf102",
    A_REPLAY: "68074743feafe85417f6828d925fd32843358fbf0e1861e008236d260de2238c",
}
TASKS = ("income", "occupation")
METRICS = ("accuracy", "nll", "brier")
VIEWS = ("r1", "r2", "pair")
REFS = ("U0", "Ucal")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blank(fields: list[str]) -> dict[str, str]:
    return dict.fromkeys(fields, "")


def build() -> tuple[list[str], list[dict[str, str]]]:
    for path, expected in EXPECTED.items():
        if digest(path) != expected:
            raise ValueError(f"public source hash changed: {path.name}")
    receipt = json.loads(A_RECEIPT.read_text())
    replay = json.loads(A_REPLAY.read_text())
    interval = json.loads(A_INTERVAL.read_text())
    if (receipt["status"] != "INCOMPLETE_FITTED_ARM_SOURCES" or
            replay["status"] != "FIXED_1161_OF_6885_ENDPOINT_REPLAY_PASS" or
            replay["endpoint_receipt_sha256"] != EXPECTED[A_RECEIPT] or
            receipt["point_csv_sha256"] != EXPECTED[A_POINT] or
            receipt["interval_json_sha256"] != EXPECTED[A_INTERVAL] or
            (interval["available_clause_count"], interval["missing_clause_count"],
             len(interval["clauses"])) != (1161, 5724, 6885)):
        raise ValueError("fixed A result or independent replay differs")
    with D_SOURCE.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        d_rows = list(reader)
    if len(d_rows) != 665 or len({r["record_id"] for r in d_rows}) != 665:
        raise ValueError("D primary rows differ")
    with A_POINT.open(newline="") as stream:
        points = list(csv.DictReader(stream))
    if len(points) != 324:
        raise ValueError("fixed A point inventory differs")
    values: dict[tuple[int, str, str, str], float] = {}
    for row in points:
        if row["draw"] not in ("0", "mean"):
            raise ValueError("fixed A draw differs")
        if row["draw"] != "0":
            continue
        key = (int(row["seed"]), row["arm"], row["scope"], row["metric"])
        if key in values:
            raise ValueError("duplicate fixed A point")
        values[key] = float(row["point"])
    arms = sorted({key[1] for key in values})
    if len(arms) != 9 or len(values) != 162:
        raise ValueError("fixed A seed/arm/metric point coverage differs")
    a_rows = []
    for seed in (0, 1):
        for arm in arms:
            all_pass = all(
                values[seed, arm, task, "accuracy"] - values[seed, ref, task, "accuracy"] >= -0.010
                and values[seed, arm, task, "nll"] - values[seed, ref, task, "nll"] <= 0.010
                and values[seed, arm, task, "brier"] - values[seed, ref, task, "brier"] <= 0.005
                for task in TASKS for ref in REFS)
            for task in TASKS:
                actual = {metric: values[seed, arm, task, metric] for metric in METRICS}
                for ref in REFS:
                    delta = {metric: actual[metric] - values[seed, ref, task, metric]
                             for metric in METRICS}
                    point_pass = (delta["accuracy"] >= -0.010 and
                                  delta["nll"] <= 0.010 and delta["brier"] <= 0.005)
                    row = blank(fields)
                    row.update(schema_version="pcrl-primary-endpoints-v1",
                               record_id=f"A:absolute_task:{arm}:s{seed}:{task}:{ref}",
                               record_kind="absolute_task", family="A",
                               service_contract="SCORE_ONLY_FIXED_REFERENCE_OR_DIAGNOSTIC",
                               comparison_scope="A_FIXED_ONLY", arm_id=arm,
                               summary_id=f"s{seed}", seed_id=str(seed), draw_id="0",
                               task=task, reference_id=ref,
                               accuracy=actual["accuracy"], nll=actual["nll"],
                               brier=actual["brier"], accuracy_delta=delta["accuracy"],
                               nll_excess=delta["nll"], brier_excess=delta["brier"],
                               task_reference_point_status="PASS_POINT" if point_pass else "FAIL_POINT",
                               all_task_point_status="PASS_POINT" if all_pass else "FAIL_POINT",
                               interval_status="ABSOLUTE_CI_NOT_COMPUTED",
                               source_point_receipt_sha256=EXPECTED[A_RECEIPT],
                               source_interval_receipt_sha256=EXPECTED[A_INTERVAL])
                    a_rows.append(row)
            for view in VIEWS:
                row = blank(fields)
                row.update(schema_version="pcrl-primary-endpoints-v1",
                           record_id=f"A:absolute_recovery:{arm}:s{seed}:{view}",
                           record_kind="absolute_recovery", family="A",
                           service_contract="SCORE_ONLY_FIXED_REFERENCE_OR_DIAGNOSTIC",
                           comparison_scope="A_FIXED_ONLY", arm_id=arm,
                           summary_id=f"s{seed}", seed_id=str(seed), draw_id="0",
                           view=view, metric="sex_auc",
                           sex_auc=values[seed, arm, view, "auc_envelope"],
                           interval_status="ABSOLUTE_CI_NOT_COMPUTED",
                           all_task_point_status="PASS_POINT" if all_pass else "FAIL_POINT",
                           source_point_receipt_sha256=EXPECTED[A_RECEIPT],
                           source_interval_receipt_sha256=EXPECTED[A_INTERVAL])
                a_rows.append(row)
    for clause in interval["clauses"]:
        parts = clause["name"].split(":")
        if len(parts) != 6:
            raise ValueError("fixed A clause name differs")
        summary, kind, first, fourth, fifth, sixth = parts
        row = blank(fields)
        row.update(schema_version="pcrl-primary-endpoints-v1",
                   record_id=f"A:clause:{clause['name']}", record_kind="signed_clause",
                   family="A", service_contract="SCORE_ONLY_K6885_REGISTERED_FAMILY",
                   comparison_scope="A_K6885_FIXED_ONLY_INCOMPLETE",
                   summary_id=summary, arm_id=first, contrast_kind=kind,
                   contrast_convention="first-minus-second",
                   interval_status=clause["status"],
                   source_point_receipt_sha256=EXPECTED[A_RECEIPT],
                   source_interval_receipt_sha256=EXPECTED[A_INTERVAL])
        if kind == "utility":
            row.update(task=fourth, reference_id=fifth, metric=sixth)
        elif kind in ("arm_pair", "ref_pair"):
            row.update(other_id=fourth, view=fifth, metric=sixth)
        else:
            raise ValueError("fixed A clause kind differs")
        for source, target in (("point_margin", "point_estimate"),
                               ("lower", "interval_lower"), ("upper", "interval_upper")):
            if clause[source] is not None:
                row[target] = clause[source]
        a_rows.append(row)
    if len(a_rows) != 72 + 54 + 6885:
        raise ValueError("fixed A public endpoint row count differs")
    rows = d_rows + a_rows
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate primary endpoint ID")
    return fields, rows


def main() -> None:
    fields, rows = build()
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} sha256={digest(OUT)}")


if __name__ == "__main__":
    main()
