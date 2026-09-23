"""Read only the completed 2016 aggregate decision, never 2016 people."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import digest, write_json


def replay(source: Path, output: Path) -> dict:
    data = json.loads(source.read_text())
    primary = {}
    for candidate in ("Q", "D17"):
        record = data["primary"][candidate]
        task = [row for row in record["clauses"] if row["clause"] == "task"]
        privacy = [row for row in record["clauses"] if row["clause"] == "privacy"]
        if len(task) != 2 or len(privacy) != 8:
            raise AssertionError("registered primary clause count changed")
        primary[candidate] = {"decision": record["decision"],
                              "task_passed": sum(row["passed"] for row in task),
                              "task_total": len(task), "privacy_passed": sum(row["passed"] for row in privacy),
                              "privacy_total": len(privacy),
                              "task": [{key: row[key] for key in ("weighting", "estimate", "threshold", "upper_bound_97_5", "passed")}
                                       for row in task]}
    secondary = {}
    for comparator in ("D17", "D33"):
        rows = [row for row in data["secondary"]["rows"] if row["comparator"] == comparator]
        task = [row for row in rows if row["role"].startswith("utility:")]
        sensitive = [row for row in rows if row["role"].startswith("attack:")]
        if len(task) != 2 or len(sensitive) != 8:
            raise AssertionError("secondary comparator count changed")
        secondary[comparator] = {
            "task": [{key: row[key] for key in ("weighting", "estimate", "lower", "upper", "excludes_zero")}
                     for row in task],
            "sensitive_positive_excludes_zero": sum(row["lower"] > 0 for row in sensitive),
            "sensitive_endpoints": len(sensitive),
        }
    result = {"source": str(source), "source_sha256": digest(source),
              "registered_2016_aggregate_only": True, "primary": primary, "secondary": secondary,
              "interpretation": "post hoc diagnostic context; original inference unchanged"}
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.source, args.output)
