"""Compare an independent INNER replay with the public task-only gate table.

This reads aggregate JSON and CSV only. It does not import the gate worker or
candidate implementation, and it never opens person-level data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CAP = {"accuracy_loss": 0.010, "nll_excess": 0.010, "brier_excess": 0.005}
VALUE = {"accuracy_loss": "accuracy", "nll_excess": "nll", "brier_excess": "brier"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(replay: dict, gate_rows: list[dict], bits: int) -> dict:
    if replay["status"] != "PASS" or len(replay["cases"]) != 2:
        raise ValueError("independent replay incomplete")
    if len(gate_rows) != 24:
        raise ValueError("gate must contain exactly 24 clauses")
    seen = set()
    checked = []
    maximum_gap = 0.0
    for row in gate_rows:
        if int(row["capacity_bits"]) != bits:
            raise ValueError("wrong capacity in gate table")
        seed = int(row["seed"])
        task, reference, metric = row["task"], row["reference"], row["metric"]
        key = (seed, task, reference, metric)
        if (seed not in (0, 1) or task not in ("income", "occupation")
                or reference not in ("U0", "Ucal") or metric not in CAP or key in seen):
            raise ValueError("duplicate or unexpected gate clause")
        seen.add(key)
        name = VALUE[metric]
        candidate = replay["cases"][f"T0_B{bits}__s{seed}"]["metrics"][task][name]
        baseline = replay["reference"][str(seed)][reference][task][name]
        cost = baseline - candidate if metric == "accuracy_loss" else candidate - baseline
        expected_status = "PASS" if cost <= CAP[metric] else "FAIL"
        comparison = {"candidate_value": candidate, "reference_value": baseline,
                      "cost": cost, "cap": CAP[metric]}
        for column, value in comparison.items():
            maximum_gap = max(maximum_gap, abs(value - float(row[column])))
        if expected_status != row["point_status"]:
            raise ValueError(f"status disagreement: {key}")
        checked.append({"seed": seed, "task": task, "reference": reference,
                        "metric": metric, "cost": cost, "cap": CAP[metric],
                        "point_status": expected_status})
    if maximum_gap > 1e-12:
        raise ValueError(f"numeric gap {maximum_gap} exceeds 1e-12")
    return {"status": "PASS", "capacity_bits": bits, "clauses_checked": len(checked),
            "pass_count": sum(row["point_status"] == "PASS" for row in checked),
            "fail_count": sum(row["point_status"] == "FAIL" for row in checked),
            "maximum_absolute_gate_gap": maximum_gap,
            "failed_clauses": [row for row in checked if row["point_status"] == "FAIL"],
            "scope": "reused INNER development point gate; no interval or privacy verification"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--bits", type=int, choices=(10, 16), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.gate.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = compare(json.loads(args.replay.read_text()), rows, args.bits)
    result.update(schema="pcrl-r5-independent-t0-gate-comparison-v1",
                  replay_sha256=sha(args.replay), gate_csv_sha256=sha(args.gate))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
