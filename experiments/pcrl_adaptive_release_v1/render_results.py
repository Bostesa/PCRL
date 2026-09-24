"""Render only pinned aggregate results from the closed 2018 assessment.

This module reads no person records, model weights, or private contributions.
It checks the registered signs and interval decisions before writing public
tables. Outer aggregate reports are optional independent crosschecks.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping

from . import audit, evaluate, inference, inference_run

ANCHORS = (0, 1, 2)
WEIGHTINGS = ("U", "PWGTP")
CSV_FIELDS = (
    "family", "id", "candidate", "arm", "comparator", "comparator_names",
    "role", "weighting", "clause", "orientation", "threshold", "estimate",
    "lower", "upper", "bootstrap_se", "anchor_estimates",
    "point_screen_passed", "interval_passed", "demonstrated_adverse",
    "decision", "benefit_over_H", "benefit_over_H_lower", "benefit_over_H_upper",
)


def _json_pinned(path: str | Path, sha256: str) -> dict:
    source = evaluate._pin(path, sha256)
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")
    value = json.loads(source.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("pinned JSON input must be an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _close(left: float, right: float, *, atol: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def _checked_rows(report: Mapping, manifest: Mapping, family: str) -> list[dict]:
    expected = manifest["endpoints"]
    rows = report.get("rows")
    if report.get("family_size") != len(expected) or not isinstance(rows, list):
        raise ValueError(f"{family} endpoint count differs from frozen family")
    if [row.get("id") for row in rows] != [item["id"] for item in expected]:
        raise ValueError(f"{family} endpoint IDs/order differ from frozen family")
    for row, endpoint in zip(rows, expected, strict=True):
        if any(row.get(key) != value for key, value in endpoint.items()):
            raise ValueError(f"{family} endpoint metadata, signs, or threshold differ")
        threshold = _number(endpoint["threshold"], "threshold")
        estimate = _number(row.get("estimate"), "estimate")
        lower = _number(row.get("lower"), "lower")
        upper = _number(row.get("upper"), "upper")
        _number(row.get("bootstrap_se"), "bootstrap_se")
        anchors = row.get("anchor_estimates")
        if not isinstance(anchors, list) or len(anchors) != 3:
            raise ValueError("each endpoint needs exactly three anchor estimates")
        anchor_values = [_number(x, "anchor estimate") for x in anchors]
        # A percentile bootstrap interval can exclude its original-sample
        # estimate when the resampling distribution is biased or asymmetric.
        if lower > upper:
            raise ValueError("invalid endpoint uncertainty interval")
        if not _close(estimate, sum(anchor_values) / 3):
            raise ValueError("reported estimate differs from equal three-anchor aggregation")
        expected_flags = {
            "point_screen_passed": estimate <= threshold,
            "passed_upper_bound": upper <= threshold,
            "demonstrated_adverse": lower > threshold,
        }
        if any(row.get(key) is not value for key, value in expected_flags.items()):
            raise ValueError("point, bound, or adverse pass flag contradicts interval")
        if family == "capability":
            if (not _close(_number(row.get("point_benefit_over_H"), "H benefit"), -estimate)
                    or not isinstance(row.get("benefit_over_H_interval"), list)
                    or len(row["benefit_over_H_interval"]) != 2
                    or not _close(_number(row["benefit_over_H_interval"][0], "H benefit lower"), -upper)
                    or not _close(_number(row["benefit_over_H_interval"][1], "H benefit upper"), -lower)
                    or row.get("point_capability_screen_passed") is not expected_flags["point_screen_passed"]
                    or row.get("interval_supports_registered_threshold") is not expected_flags["passed_upper_bound"]):
                raise ValueError("H-capability benefit axis or interval flag differs")
    return rows


def _verify_primary_decisions(inference_report: Mapping, rows: list[dict]) -> dict:
    candidates = sorted({row["candidate"] for row in rows})
    expected = {}
    for name in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == name]
        passed = sum(row["passed_upper_bound"] for row in candidate_rows)
        expected[name] = {
            "all_primary_clauses_passed": bool(candidate_rows) and passed == len(candidate_rows),
            "passed": passed, "total": len(candidate_rows),
        }
    if inference_report.get("decisions") != expected:
        raise ValueError("primary conjunction differs from interval-bound decisions")
    return expected


def _outer_score(report: Mapping, canonical: str, role: str,
                 weighting: str) -> float:
    if role not in audit.ROLES or weighting not in WEIGHTINGS:
        raise ValueError("unknown locked role or weighting")
    releases = report["releases"]
    if canonical == "H":
        scores = [_number(item["roles"][role]["H"][weighting], "H score")
                  for item in releases.values()]
        if not scores or any(not _close(scores[0], value) for value in scores[1:]):
            raise ValueError("shared H aggregate score differs across releases")
        return scores[0]
    return _number(releases[canonical]["roles"][role]["candidate"][weighting],
                   "outer candidate score")


def _check_outer_reports(outer: Mapping[int, tuple[str | Path, str]] | None,
                         lock: Mapping, lock_sha256: str,
                         rows: list[dict]) -> str:
    if outer is None:
        return "NOT_SUPPLIED"
    if set(outer) != set(ANCHORS):
        raise ValueError("all three pinned outer aggregate reports are required")
    reports = {}
    for anchor in ANCHORS:
        path, pin = outer[anchor]
        report = _json_pinned(path, pin)
        locked = lock["anchors"][str(anchor)]
        if (report.get("schema") != "pcrl-adaptive-outer-audit-v1"
                or report.get("anchor") != anchor
                or report.get("assessment_year") != 2018
                or report.get("development_only") is not True
                or report.get("no_outer_fit_or_selection") is not True
                or report.get("selection_lock_sha256") != lock_sha256
                or set(report.get("releases", {})) != set(locked["releases"])):
            raise ValueError("outer aggregate report differs from locked anchor/panel")
        for canonical, source in locked["releases"].items():
            item = report["releases"][canonical]
            if item.get("source") != source or set(item.get("roles", {})) != set(audit.ROLES):
                raise ValueError("outer release source or role roster differs")
        reports[anchor] = report
    for row in rows:
        for anchor in ANCHORS:
            mapping = lock["anchors"][str(anchor)]["logical_to_canonical"]
            plus = "H" if row["plus"] == "H" else mapping[row["plus"]]
            minus = "H" if row["minus"] == "H" else mapping[row["minus"]]
            report = reports[anchor]
            contrast = (_outer_score(report, plus, row["role"], row["weighting"])
                        - _outer_score(report, minus, row["role"], row["weighting"]))
            if not _close(contrast, float(row["anchor_estimates"][anchor])):
                raise ValueError("outer aggregate contrast differs from inference anchor estimate")
    return "PASS_ALL_THREE_ANCHORS"


def _csv_record(row: Mapping, family: str) -> dict:
    status = ("PASS" if row["passed_upper_bound"] else
              "ADVERSE" if row["demonstrated_adverse"] else "UNRESOLVED")
    record = {key: row.get(key, "") for key in CSV_FIELDS}
    record.update(family=family, interval_passed=row["passed_upper_bound"], decision=status,
                  arm=row.get("arm", ""))
    for key in ("comparator_names", "anchor_estimates"):
        if key in row:
            record[key] = json.dumps(row[key], separators=(",", ":"), allow_nan=False)
    if family == "capability":
        record.update(benefit_over_H=row["point_benefit_over_H"],
                      benefit_over_H_lower=row["benefit_over_H_interval"][0],
                      benefit_over_H_upper=row["benefit_over_H_interval"][1])
    return record


def _write_new(path: str | Path, content: str) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"preserve existing aggregate result: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(content)
    os.replace(temporary, target)


def render_results(selection_lock_path: str | Path, lock_sha256: str,
                   inner_selection_path: str | Path, inner_selection_sha256: str,
                   inference_path: str | Path, inference_sha256: str,
                   outer_reports: Mapping[int, tuple[str | Path, str]] | None,
                   output_csv: str | Path, output_table: str | Path,
                   output_plot: str | Path) -> dict:
    """Validate pinned aggregate evidence and write immutable public summaries."""
    outputs = [Path(path).resolve() for path in (output_csv, output_table, output_plot)]
    if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
        raise FileExistsError("aggregate output paths must be distinct and unused")
    lock = _json_pinned(selection_lock_path, lock_sha256)
    inner = _json_pinned(inner_selection_path, inner_selection_sha256)
    report = _json_pinned(inference_path, inference_sha256)
    inference_run.validate_locked_resolution(lock)
    family = inference.family_manifest(lock["slots"], alias_of=lock.get("alias_of", {}))
    capability_manifest = inference_run.expected_capability_manifest(lock["slots"])
    if (lock.get("family_manifest") != family or report.get("family_manifest") != family
            or lock.get("capability_manifest") != capability_manifest
            or report.get("capability_manifest") != capability_manifest):
        raise ValueError("locked primary or separate capability family differs")
    if (inner.get("schema") != "pcrl-adaptive-inner-selection-v1"
            or inner.get("source_role") != "inner_selection"
            or inner.get("outer_pool_opened") is not False
            or inner.get("candidate_choices") != lock.get("selection_source", {}).get("candidate_choices")
            or inner.get("family_choices") != lock.get("selection_source", {}).get("family_choices")):
        raise ValueError("inner selection differs from committed selection lock")
    if (report.get("schema") != "pcrl-adaptive-outer-inference-v1"
            or report.get("assessment_year") != 2018
            or report.get("development_only") is not True
            or report.get("selection_lock_sha256") != lock_sha256):
        raise ValueError("inference is not the locked 2018 development assessment")
    resolution = inference_run.validate_locked_resolution(lock)
    if report.get("anchor_resolution") != {str(a): resolution[a] for a in ANCHORS}:
        raise ValueError("inference anchor aliases differ from lock")
    primary_rows = _checked_rows(report, family, "primary")
    decisions = _verify_primary_decisions(report, primary_rows)
    capability = report.get("capability")
    if not isinstance(capability, dict) or capability.get("primary_clause_rescue") is not False:
        raise ValueError("capability may not rescue a primary clause")
    capability_rows = _checked_rows(capability, capability_manifest, "capability")
    crosscheck = _check_outer_reports(outer_reports, lock, lock_sha256,
                                     [*primary_rows, *capability_rows])
    all_records = [_csv_record(row, "primary") for row in primary_rows]
    all_records += [_csv_record(row, "capability") for row in capability_rows]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(all_records)
    table = {
        "schema": "pcrl-adaptive-aggregate-endpoint-table-v1",
        "scope": "2018 development; aggregate-only; selected models conditional on development",
        "input_sha256": {"selection_lock": lock_sha256,
                         "inner_selection": inner_selection_sha256,
                         "inference": inference_sha256,
                         "outer_aggregate_reports": {str(a): outer_reports[a][1]
                                                     for a in ANCHORS} if outer_reports else {}},
        "outer_aggregate_crosscheck": crosscheck,
        "primary": decisions,
        "primary_family_size": len(primary_rows),
        "capability_family_size": len(capability_rows),
        "capability_is_separate": True,
        "rows": [{"family": record["family"], "id": record["id"],
                  "candidate": record["candidate"], "comparator": record["comparator"],
                  "role": record["role"], "weighting": record["weighting"],
                  "threshold": record["threshold"], "estimate": record["estimate"],
                  "lower": record["lower"], "upper": record["upper"],
                  "decision": record["decision"]} for record in all_records],
    }
    def point(row: Mapping, family_name: str) -> dict:
        if family_name == "capability":
            return {"family": "capability", "id": row["id"],
                    "axis": "CE_H_minus_CE_candidate",
                    "estimate": row["point_benefit_over_H"],
                    "lower": row["benefit_over_H_interval"][0],
                    "upper": row["benefit_over_H_interval"][1],
                    "threshold": -row["threshold"],
                    "primary_clause_rescue": False}
        return {"family": "primary", "id": row["id"],
                "axis": row["orientation"], "estimate": row["estimate"],
                "lower": row["lower"], "upper": row["upper"],
                "threshold": row["threshold"],
                "interval_passed": row["passed_upper_bound"]}
    plot = {"schema": "pcrl-adaptive-aggregate-plot-data-v1",
            "primary": [point(row, "primary") for row in primary_rows],
            "capability": [point(row, "capability") for row in capability_rows]}
    _write_new(output_csv, stream.getvalue())
    _write_new(output_table, json.dumps(table, sort_keys=True, indent=2, allow_nan=False) + "\n")
    _write_new(output_plot, json.dumps(plot, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"primary_all_passed": all(value["all_primary_clauses_passed"]
                                         for value in decisions.values()),
            "primary_endpoint_count": len(primary_rows),
            "capability_endpoint_count": len(capability_rows),
            "outer_aggregate_crosscheck": crosscheck}


def _anchor_args(values: list[str], label: str) -> dict[int, str]:
    result = {}
    for item in values:
        anchor_text, sep, value = item.partition("=")
        if not sep or anchor_text not in {"0", "1", "2"} or not value:
            raise ValueError(f"{label} must be ANCHOR=VALUE for anchor 0, 1, or 2")
        anchor = int(anchor_text)
        if anchor in result:
            raise ValueError(f"duplicate {label} anchor")
        result[anchor] = value
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--inner-selection", required=True)
    parser.add_argument("--inner-selection-sha256", required=True)
    parser.add_argument("--inference", required=True)
    parser.add_argument("--inference-sha256", required=True)
    parser.add_argument("--outer-report", action="append", default=[])
    parser.add_argument("--outer-sha256", action="append", default=[])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-plot", required=True)
    args = parser.parse_args(argv)
    reports = _anchor_args(args.outer_report, "outer report")
    pins = _anchor_args(args.outer_sha256, "outer SHA")
    if set(reports) != set(pins) or (reports and set(reports) != set(ANCHORS)):
        parser.error("provide paths and SHA-256 pins for all three outer reports, or none")
    summary = render_results(
        args.selection_lock, args.lock_sha256,
        args.inner_selection, args.inner_selection_sha256,
        args.inference, args.inference_sha256,
        {a: (reports[a], pins[a]) for a in ANCHORS} if reports else None,
        args.output_csv, args.output_table, args.output_plot)
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
