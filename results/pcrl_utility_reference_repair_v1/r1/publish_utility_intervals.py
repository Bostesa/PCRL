"""Publish aggregate utility bounds only after the sealed R3 numerical replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


FIELDS = ("seed", "task", "reference", "endpoint", "point", "bootstrap_sd",
          "lower", "upper", "cap", "status")
NUMERIC = ("point", "bootstrap_sd", "lower", "upper", "cap")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_queue(study: Path, generation: int, uid: str,
                 output: Path) -> tuple[dict, str]:
    path = study / f".private/queue_g{generation}/QUEUE.json"
    queue = json.loads(path.read_text())
    unit = queue.get("units", {}).get(uid, {})
    if (set(queue.get("units", {})) != {uid}
            or unit.get("status") != "DONE"
            or unit.get("output_sha256") != {str(output): sha(output)}):
        raise PermissionError(f"q{generation} output not completed and frozen")
    return queue, sha(path)


def publish(study: Path) -> dict:
    study = study.resolve(strict=True)
    private = study / ".private"
    r1_path = private / "r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json"
    r3_path = private / "r3/UTILITY_INTERVAL_REPLAY.json"
    q14, q14_sha = _check_queue(study, 14, "R1_UTILITY_INTERVALS", r1_path)
    q15, q15_sha = _check_queue(study, 15, "R3_UTILITY_INTERVAL_REPLAY", r3_path)
    if q15.get("carry_forward") != {
            "archived_queue_relative": "queue_g14",
            "archived_queue_sha256": q14_sha,
            "charged_cpu_s": q14["used_cpu_s"], "unit_count": 1}:
        raise PermissionError("q15 did not inherit the exact q14 budget")
    r1 = json.loads(r1_path.read_text())
    r3 = json.loads(r3_path.read_text())
    independent_path = study / "r3/UTILITY_INTERVAL_NUMERICAL_VERIFICATION.json"
    independent = json.loads(independent_path.read_text())
    gate = study / "FULL_INPUT_GATE.json"
    q14_lock = private / "stage_seals/utility_intervals/CAMPAIGN_LOCK.json"
    q15_lock = private / "stage_seals/utility_interval_replay/CAMPAIGN_LOCK.json"
    if (r1.get("schema") != "pcrl-utility-simultaneous-intervals-v1"
            or r3.get("schema") != "pcrl-utility-r3-independent-interval-replay-v1"
            or r3.get("status") != "REPLAYED"
            or r1.get("primary_F_gate_sha256") != sha(gate)
            or r3.get("full_input_gate_sha256") != sha(gate)
            or r3.get("r1_interval_output_sha256") != sha(r1_path)
            or r1.get("sampled_group_index_sha256") !=
            r3.get("sampled_group_index_sha256")
            or r1.get("rows") != 2235 or r3.get("rows") != 2235
            or r1.get("groups") != 2234 or r3.get("groups") != 2234
            or r1.get("family_size") != 24 or r3.get("family_size") != 24
            or r1.get("replicates") != 2048 or r3.get("replicates") != 2048
            or r1.get("status_counts") != r3.get("status_counts")
            or r1.get("family_status") != r3.get("family_status")
            or r3.get("all_clause_statuses_match") is not True
            or r3.get("max_abs_interval_value_delta_against_R1", math.inf) > 1e-10
            or independent.get("schema") !=
            "pcrl-utility-r3-interval-numerical-verification-v1"
            or independent.get("status") != "PASS_INDEPENDENT_NUMERICAL_REPLAY"
            or independent.get("r1_interval_output_sha256") != sha(r1_path)
            or independent.get("independent_replay_output_sha256") != sha(r3_path)
            or independent.get("independent_replay_queue_sha256") != q15_sha
            or independent.get("status_counts") != r1.get("status_counts")
            or independent.get("family_status") != r1.get("family_status")
            or json.loads(q15_lock.read_text()).get("r1_interval_output_sha256")
            != sha(r1_path)
            or json.loads(q15_lock.read_text()).get("prior_interval_queue_state_sha256")
            != q14_sha
            or json.loads(q14_lock.read_text()).get("full_input_gate_sha256")
            != sha(gate)):
        raise PermissionError("R1/R3 interval replay or archived science lineage differs")
    if len(r1.get("clauses", [])) != 24 or len(r3.get("clauses", [])) != 24:
        raise PermissionError("reported 24-clause family incomplete")
    max_delta = 0.0
    for a, b in zip(r1["clauses"], r3["clauses"]):
        if (set(a) != set(FIELDS) or set(b) != set(FIELDS)
                or any(a[key] != b[key]
                       for key in ("seed", "task", "reference", "endpoint", "status"))):
            raise PermissionError("R1/R3 clause order or status differs")
        for key in NUMERIC:
            max_delta = max(max_delta, abs(float(a[key]) - float(b[key])))
    if max_delta > 1e-10:
        raise PermissionError("R1/R3 numeric bounds differ")
    csv_path = study / "FULL_INPUT_UTILITY_INTERVALS.csv"
    summary_path = study / "r1/UTILITY_INTERVAL_VERIFICATION.json"
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError("public interval publication is write-once")
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in r1["clauses"])
    summary = {
        "schema": "pcrl-utility-public-interval-verification-v1",
        "status": "INDEPENDENT_NUMERICAL_REPLAY_PASSED",
        "scope": "conditional post-selection development inference on reused Adult INNER_SELECTION people",
        "full_input_gate_sha256": sha(gate),
        "r1_interval_output_sha256": sha(r1_path),
        "r3_interval_replay_sha256": sha(r3_path),
        "r3_public_numerical_verification_sha256": sha(independent_path),
        "q14_queue_state_sha256": q14_sha,
        "q15_queue_state_sha256": q15_sha,
        "q14_science_lock_sha256": sha(q14_lock),
        "q15_science_lock_sha256": sha(q15_lock),
        "sampled_group_index_sha256": r1["sampled_group_index_sha256"],
        "rows": 2235, "paired_groups": 2234, "replicates": 2048,
        "family_size": 24, "family_status": r1["family_status"],
        "status_counts": r1["status_counts"],
        "r3_reported_max_abs_interval_value_delta_against_R1":
            r3["max_abs_interval_value_delta_against_R1"],
        "publisher_max_abs_clause_value_delta": max_delta,
        "max_abs_point_delta_against_F_gate":
            r3["max_abs_point_delta_against_F_gate"],
        "bounds": "separate simultaneous one-sided 95% descriptive bounds",
        "interpretation_limit": "selected F and U0/Ucal fit held fixed; no post-selection or population confirmation",
    }
    with summary_path.open("x") as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return {"csv_sha256": sha(csv_path), "summary_sha256": sha(summary_path),
            "family_status": r1["family_status"], "status_counts": r1["status_counts"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study), sort_keys=True))
