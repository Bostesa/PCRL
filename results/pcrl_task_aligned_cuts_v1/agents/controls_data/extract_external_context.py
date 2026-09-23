"""Copy only aggregate historical 2018 external-baseline results, with source pins.

This reads committed public CSV blobs. It never loads archived person-level arrays.
The output describes a different, repeatedly used development audit and is not a
same-slate comparison with this study's token mechanisms.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path


SOURCE_COMMIT = "7f961d5c7f6f0562efcb25a27a77bb5221c279a7"
SOURCE_ROOT = "results/pcrl_invariant_baselines_v1"
CONDITIONS = ("J", "leace_A0", "splince_A0", "optnet16_L1", "optnet16_L2", "optnet16_C1")
ENDPOINTS = (
    "utility/same_residence",
    "recovery/A/SEX",
    "recovery/AB/SEX",
    "recovery/A/RAC1P",
    "recovery/AB/RAC1P",
)


def source_csv(name: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    path = f"{SOURCE_ROOT}/{name}"
    blob = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"])
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
    return rows, {
        "commit": SOURCE_COMMIT,
        "path": path,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    paired, paired_pin = source_csv("PAIRED_INTERVALS.csv")
    criteria, criteria_pin = source_csv("CRITERIA_SUMMARY.csv")
    per_seed, per_seed_pin = source_csv("PER_SEED.csv")
    contrasts = [
        row for row in paired
        if row["left"] in CONDITIONS[1:] and row["right"] == "J"
        and row["endpoint"] in ENDPOINTS
        and row["weight"] in ("unweighted", "person_weighted")
    ]
    expected = {(arm, weight, endpoint)
                for arm in CONDITIONS[1:]
                for weight in ("unweighted", "person_weighted")
                for endpoint in ENDPOINTS}
    actual = {(row["left"], row["weight"], row["endpoint"])
              for row in contrasts}
    if actual != expected or len(contrasts) != len(expected):
        raise ValueError(f"historical contrast family mismatch: {len(contrasts)} rows")
    gains = [
        row for row in criteria
        if row["condition"] in CONDITIONS
        and row["weight"] in ("unweighted", "person_weighted")
    ]
    if len(gains) != len(CONDITIONS) * 2:
        raise ValueError("historical task-gain rows incomplete")
    observed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in per_seed:
        if (row["condition"] not in CONDITIONS or row["split"] != "test"
                or row["budget"] != "360" or row["scope"] != "kernel_expanded_catchup"):
            continue
        if row["kind"] == "utility_loss" and row["endpoint"] == "same_residence":
            endpoint = "utility/same_residence"
        elif row["kind"] in ("additional_recovery", "absolute_recovery") and row["endpoint"] in (
                "A/SEX", "AB/SEX", "A/RAC1P", "AB/RAC1P"):
            endpoint = f"{row['kind']}/{row['endpoint']}"
        else:
            continue
        observed[row["condition"], row["weight"], endpoint].append(float(row["value"]))
    absolute = []
    for condition in CONDITIONS:
        for weight in ("unweighted", "person_weighted"):
            for endpoint in ("utility/same_residence", *(
                    f"{kind}/{role}" for kind in ("additional_recovery", "absolute_recovery")
                    for role in ("A/SEX", "AB/SEX", "A/RAC1P", "AB/RAC1P"))):
                values = observed[condition, weight, endpoint]
                if len(values) != 3:
                    raise ValueError(f"incomplete per-seed primary-scope aggregate: {condition}/{weight}/{endpoint}")
                absolute.append({"condition": condition, "weight": weight, "endpoint": endpoint,
                                 "mean_of_three_seeds": sum(values) / 3,
                                 "seed_min": min(values), "seed_max": max(values),
                                 "seed_values": json.dumps(values),
                                 "scope": "test/kernel_expanded_catchup/budget360"})
    absolute_lookup = {(x["condition"], x["weight"], x["endpoint"]):
                       x["mean_of_three_seeds"] for x in absolute}
    for row in contrasts:
        endpoint = row["endpoint"]
        if endpoint.startswith("recovery/"):
            endpoint = endpoint.replace("recovery/", "absolute_recovery/", 1)
        recomputed = (absolute_lookup[row["left"], row["weight"], endpoint]
                      - absolute_lookup["J", row["weight"], endpoint])
        if abs(recomputed - float(row["estimate"])) > 1e-9:
            raise ValueError(f"historical aggregate contrast does not replay: {row['left']}/{row['weight']}/{endpoint}")
    meta = {
        "schema": "pcrl-historical-external-context-v1",
        "scope": "previous 2018 development study; repeatedly used data; different audit slate, access, and split",
        "paired_source": paired_pin,
        "criteria_source": criteria_pin,
        "per_seed_source": per_seed_pin,
        "sign": "left-minus-J task log loss or recovery; negative favors left",
        "access": "J and external maps output continuous A aux16 beside H_A4 (A width20, AB width22); current matched finite releases use 17 visible token labels and supervised residence T0",
        "contrast_rows": len(contrasts),
        "task_gain_rows": len(gains),
        "absolute_rows": len(absolute),
        "absolute_replay_max_tolerance": 1e-9,
        "absolute_aggregation": "arithmetic mean over three overlapping-seed aggregate test-pool scores; not three independent populations",
        "non_comparability": "context only; do not combine with current same-slate inner-pilot or outer-assessment inference",
    }
    for name, rows in (("EXTERNAL_2018_VS_J.csv", contrasts),
                       ("EXTERNAL_2018_TASK_GAIN.csv", gains),
                       ("EXTERNAL_2018_ABSOLUTE.csv", absolute)):
        fields = list(rows[0])
        with (out_dir / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "EXTERNAL_2018_CONTEXT.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contrast_rows": len(contrasts), "task_gain_rows": len(gains),
                      "absolute_rows": len(absolute),
                      "paired_sha256": paired_pin["sha256"],
                      "criteria_sha256": criteria_pin["sha256"],
                      "per_seed_sha256": per_seed_pin["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
