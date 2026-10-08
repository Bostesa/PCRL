"""Atomically preserve the first admitted export before same-source byte replay.

Only file bytes and names are inspected. No role array or mixed source opens.
This is a one-shot custodian operation, not a scientific fit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


PRIVATE = Path(__file__).resolve().parent / ".private"
SOURCE = PRIVATE / "role_export_v1"
FIRST = PRIVATE / "role_export_v1_first_run"
RECEIPT = PRIVATE / "first_export_preservation.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root: Path) -> list[dict]:
    return [{"relative_path": str(p.relative_to(root)), "sha256": digest(p)}
            for p in sorted(root.rglob("*")) if p.is_file()]


def main() -> None:
    if not SOURCE.is_dir() or FIRST.exists() or RECEIPT.exists():
        raise RuntimeError("preservation precondition failed")
    before = inventory(SOURCE)
    if len(before) != 26 or {x["relative_path"] for x in before}.issuperset({"EXPORT_RECEIPT.json"}) is False:
        raise RuntimeError("first export inventory is incomplete")
    os.rename(SOURCE, FIRST)
    after = inventory(FIRST)
    if before != after:
        raise RuntimeError("preserved export file bytes differ")
    report = {"schema": "pcrl-first-export-preservation-v1", "status": "PASS",
              "method": "atomic same-filesystem rename", "file_count": len(after), "files": after}
    fd = os.open(RECEIPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"status": "PASS", "files": len(after), "receipt_sha256": digest(RECEIPT)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print("PRESERVE_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
