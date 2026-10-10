"""Copy sealed aggregate endpoints into one public CSV; never read person arrays."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(utility: Path, recovery: Path, output: Path) -> None:
    docs = [(utility, "utility", 192), (recovery, "recovery", 756)]
    rows = []
    seen = set()
    for path, kind, count in docs:
        doc = json.loads(path.read_text())
        if doc["K"] != 948 or len(doc["slots"]) != count:
            raise ValueError("Fixed simultaneous endpoint inventory differs")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for key, value in sorted(doc["slots"].items()):
            parts = key.split(":")
            if len(parts) != 6 or parts[1] != kind or key in seen:
                raise ValueError("Endpoint identity differs or repeats")
            seen.add(key)
            row = {"slot_id": key, "kind": kind, "summary": parts[0],
                   "arm": parts[2], "comparison": parts[4] if kind == "utility" else parts[3],
                   "task": parts[3] if kind == "utility" else "",
                   "view": "" if kind == "utility" else parts[4], "metric": parts[5],
                   "K": 948, "source": str(path.relative_to(ROOT)), "source_sha256": digest}
            row.update({field: value.get(field) for field in ("point", "se", "lower", "upper", "status", "df")})
            rows.append(row)
    if len(seen) != 948:
        raise ValueError("All 948 prospective slots must remain")
    columns = ["slot_id", "kind", "summary", "arm", "comparison", "task", "view", "metric",
               "point", "se", "lower", "upper", "status", "K", "df", "source", "source_sha256"]
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


if __name__ == "__main__":
    export(ROOT / "r3/R3_UTILITY_INTERVALS_V1.json",
           ROOT / "r3/R3_RECOVERY_INFERENCE_V1.json", ROOT / "PRIMARY_ENDPOINTS.csv")
