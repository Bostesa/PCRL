"""Write the fixed-arm point nomination table from public R3 gate receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .r3_eligibility import point_nomination_rows
from .r3_family import ARMS


FIELDS = ("arm_id", "planned_role", "utility_gate", "nomination",
          "fresh_attack_eligible", "gate_receipt_sha256")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_table(*, new_gates: list[Path], historical_gates: list[Path],
                incomplete: set[str], output: Path) -> dict:
    groups = []
    hashes = {}
    for paths in (new_gates, historical_gates):
        docs = {}
        for path in paths:
            if path.is_symlink():
                raise PermissionError("nomination gate symlink refused")
            path = path.resolve(strict=True)
            item = json.loads(path.read_text())
            arm = item.get("arm_id")
            if arm in docs or arm not in ARMS:
                raise PermissionError("duplicate or unknown nomination gate")
            docs[arm] = item
            hashes[arm] = _sha(path)
        groups.append(docs)
    rows = point_nomination_rows(groups[0], incomplete, groups[1])
    for row in rows:
        row["gate_receipt_sha256"] = hashes.get(row["arm_id"], "")
    if output.is_symlink() or output.exists() or not output.parent.is_dir():
        raise PermissionError("nomination table must be write-once")
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644),
                   "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    return {"schema": "pcrl-bank-r3-point-nomination-table-v1",
            "table_sha256": _sha(output), "rows": len(rows),
            "source_gate_sha256_by_arm": hashes,
            "scope": "point eligibility only; J remains the locked nominee if eligible"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--new-gate", type=Path, action="append", default=[])
    parser.add_argument("--historical-gate", type=Path, action="append", default=[])
    parser.add_argument("--incomplete", choices=ARMS, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_table(new_gates=args.new_gate,
                                 historical_gates=args.historical_gate,
                                 incomplete=set(args.incomplete),
                                 output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
