"""Compare independently replayed INNER aggregates to the R2 old-loss table.

Inputs are public aggregates only. No R2 endpoint implementation is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(independent_path: Path, reported_path: Path,
        expected_independent_sha: str, expected_reported_sha: str,
        output: Path) -> dict:
    if (sha(independent_path) != expected_independent_sha
            or sha(reported_path) != expected_reported_sha):
        raise PermissionError("aggregate input byte pin differs")
    independent = json.loads(independent_path.read_text())
    if independent.get("schema") != "pcrl-r5-independent-inner-replay-v1":
        raise ValueError("independent replay schema differs")
    with reported_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    checks = []
    for row in rows:
        seed = int(row["seed"])
        task = row["task"]
        arm = row["first"]
        kind = row["record_type"]
        metric = row["metric"]
        if kind == "absolute" and metric in (
            "accuracy", "nll", "brier", "constant_accuracy",
            "accuracy_gain_over_constant",
        ):
            if arm in ("U0", "Ucal"):
                source = independent["reference"][str(seed)][arm][task]
            else:
                source = independent["cases"][f"{arm}__s{seed}"]["metrics"][task]
            actual = float(source[metric])
        elif kind == "class_count":
            source = (independent["reference"][str(seed)][arm][task] if arm in ("U0", "Ucal")
                      else independent["cases"][f"{arm}__s{seed}"]["metrics"][task])
            actual = int(source["class_counts"][int(metric.split("_")[1])])
        elif kind == "recall":
            source = (independent["reference"][str(seed)][arm][task] if arm in ("U0", "Ucal")
                      else independent["cases"][f"{arm}__s{seed}"]["metrics"][task])
            actual = source["per_class_recall"][int(metric.split("_")[1])]
            if actual is None and row["value"] == "":
                continue
        elif kind == "state" and arm not in ("U0", "Ucal"):
            source = independent["cases"][f"{arm}__s{seed}"]["states"][task]
            alias = {"global_capacity_states": "global_capacity_states",
                     "global_bank_states": "global_table_states",
                     "occupied_states": "occupied_states",
                     "observed_entropy_bits": "entropy_bits",
                     "minimum_capacity_bits_for_bank": "minimum_capacity_bits",
                     "transmitted_continuous_dimensions": "transmitted_continuous_dimensions"}
            if metric not in alias or alias[metric] not in source:
                continue
            actual = float(source[alias[metric]])
        elif kind == "difference" and metric in ("accuracy", "nll", "brier"):
            first = (independent["reference"][str(seed)][arm][task] if arm in ("U0", "Ucal")
                     else independent["cases"][f"{arm}__s{seed}"]["metrics"][task])
            second = independent["cases"][f"{row['second']}__s{seed}"]["metrics"][task]
            actual = float(second[metric] - first[metric])
        else:
            continue
        expected = float(row["value"])
        checks.append({"seed": seed, "task": task, "type": kind, "arm": arm,
                       "second": row["second"], "metric": metric,
                       "absolute_gap": abs(actual - expected)})
    maximum = max((item["absolute_gap"] for item in checks), default=float("inf"))
    difference_count = sum(item["type"] == "difference" for item in checks)
    if not checks or difference_count != 36 or maximum > 1e-10:
        raise AssertionError("reported old-loss aggregate differs from independent replay")
    receipt = {"schema": "pcrl-r5-old-loss-aggregate-comparison-v1",
               "status": "PASS", "independent_sha256": expected_independent_sha,
               "reported_sha256": expected_reported_sha,
               "compared_values": len(checks), "difference_values": difference_count,
               "maximum_absolute_gap": maximum,
               "scope": "public aggregate comparison on reused INNER; no person-level input"}
    with output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for flag in ("independent", "reported", "independent-sha", "reported-sha", "output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.independent), Path(args.reported),
                         args.independent_sha, args.reported_sha, Path(args.output)),
                     sort_keys=True))


if __name__ == "__main__":
    main()
