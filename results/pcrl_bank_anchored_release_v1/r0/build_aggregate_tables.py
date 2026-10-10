"""Combine immutable public R3 scalar rows for the bank release closeout."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "r3"
ARMS = ("B0", "T", "D", "P", "L", "J", "S12", "S21")
BASE_FIELDS = (
    "role", "arm", "seed", "task", "reference", "metric", "candidate_value",
    "reference_value", "margin", "cap", "primary_gate", "point_status",
    "diagnostic_point_status",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(role: str, arm: str, seed: int) -> Path:
    suffix = "-R1" if arm == "B0" else "_G1"
    return R3 / f"R3_{arm}_S{seed}_{role}{suffix}.csv"


def assemble(role: str, output: str) -> None:
    rows = []
    for arm in ARMS:
        for seed in (0, 1):
            path = source(role, arm, seed)
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != BASE_FIELDS:
                    raise ValueError(f"unexpected scalar schema: {path.name}")
                source_rows = list(reader)
            if len(source_rows) != 18:
                raise ValueError(f"expected 18 aggregate scalar rows: {path.name}")
            for row in source_rows:
                if row["arm"] != arm or int(row["seed"]) != seed:
                    raise ValueError(f"scalar arm/seed mismatch: {path.name}")
                rows.append({**row, "source_csv_sha256": digest(path)})
    destination = ROOT / output
    if destination.exists():
        raise FileExistsError(f"write-once aggregate exists: {destination.name}")
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*BASE_FIELDS, "source_csv_sha256"))
        writer.writeheader()
        writer.writerows(rows)
    print(destination.name, len(rows), digest(destination))


if __name__ == "__main__":
    assemble("FIT", "FIT_PROXY_AND_OLD_REFERENCE_DIAGNOSTICS.csv")
    assemble("INNER", "ORIGINAL_UTILITY_CAPS.csv")
