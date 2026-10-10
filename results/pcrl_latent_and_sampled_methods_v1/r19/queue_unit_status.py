#!/usr/bin/env python3
"""Propose public UNIT_STATUS rows from R03 state without exporting private paths.

This is a read-only integration helper. R00 decides whether and when to copy
its output into the shared campaign UNIT_STATUS file.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys


DEFAULT_STATE = Path(__file__).resolve().parents[1] / "r03" / ".private" / "compute" / "state.json"
FIELDS = ("record_kind", "record_id", "unit_id", "charge_id", "stage", "status",
          "direct_child_cpu_s", "charged_cpu_s",
          "wall_s", "sampled_tree_rss_peak_bytes", "receipt_sha256")
STAGES = frozenset(("admission", "d_first", "d_followup", "verification", "a", "optional"))
STATUSES = frozenset(("queued", "held", "running", "done", "failed", "incomplete", "superseded"))
FINAL_STATUSES = frozenset(("done", "failed", "incomplete"))
UNIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NUMERIC = ("direct_child_cpu_s", "charged_cpu_s", "wall_s", "sampled_tree_rss_peak_bytes")


def _number(entry: dict, key: str) -> str:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid metric")
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid metric")
    if key == "sampled_tree_rss_peak_bytes" and not isinstance(value, int):
        raise ValueError("invalid metric")
    return str(value)


def propose_rows(state_path: Path) -> list[dict[str, str]]:
    """Return whitelisted fields for units, supersessions and charge receipts."""
    state = json.loads(state_path.read_text())
    if state.get("schema") != "pcrl-latent-sampled-queue-v1" or not isinstance(state.get("units"), dict):
        raise ValueError("invalid queue state")
    root = state_path.parent.resolve()
    rows = []
    expected_unit_receipts = set()
    for uid, entry in sorted(state["units"].items()):
        if not isinstance(uid, str) or not UNIT_ID.fullmatch(uid) or ".." in uid:
            raise ValueError("invalid unit ID")
        if not isinstance(entry, dict) or not isinstance(entry.get("spec"), dict):
            raise ValueError("invalid unit")
        stage = entry["spec"].get("stage")
        status = entry.get("status")
        if stage not in STAGES or status not in STATUSES:
            raise ValueError("invalid unit")
        if entry["spec"].get("unit_id", uid) != uid or entry.get("stage", stage) != stage:
            raise ValueError("unit identity mismatch")
        row = dict.fromkeys(FIELDS, "")
        row.update(record_kind="unit", record_id=uid, unit_id=uid, stage=stage, status=status)
        if status in FINAL_STATUSES:
            receipt_path = state_path.parent / "units" / uid / "RECEIPT.json"
            expected_unit_receipts.add(receipt_path.resolve())
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
            if any(receipt.get(key) != value for key, value in (("unit_id", uid), ("stage", stage),
                                                                 ("status", status))):
                raise ValueError("receipt identity mismatch")
            for key in NUMERIC:
                value = _number(entry, key)
                if receipt.get(key) != entry[key]:
                    raise ValueError("receipt metric mismatch")
                row[key] = value
            row["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
        elif status == "superseded":
            value = entry.get("supersession_receipt_sha256")
            if not isinstance(value, str) or not SHA256.fullmatch(value):
                raise ValueError("invalid supersession receipt")
            receipt_path = root / "units" / uid / "SUPERSESSION.json"
            expected_unit_receipts.add(receipt_path.resolve())
            if Path(entry.get("supersession_receipt_path", "")).resolve() != receipt_path.resolve():
                raise ValueError("supersession receipt path mismatch")
            receipt_bytes = receipt_path.read_bytes()
            if hashlib.sha256(receipt_bytes).hexdigest() != value:
                raise ValueError("supersession receipt hash mismatch")
            receipt = json.loads(receipt_bytes)
            if (receipt.get("schema") != "pcrl-queue-unstarted-supersession-v1" or
                    receipt.get("old_unit_id") != uid or
                    receipt.get("new_unit_id") != entry.get("superseded_by") or
                    receipt.get("old_child_started") is not False):
                raise ValueError("supersession receipt identity mismatch")
            cpu = receipt.get("old_cpu_charge_s")
            if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu < 0:
                raise ValueError("invalid supersession charge")
            row["charged_cpu_s"] = str(cpu)
            row["receipt_sha256"] = value
        rows.append(row)

    unit_root = root / "units"
    actual_unit_receipts = ({path.resolve() for pattern in ("*/RECEIPT.json", "*/SUPERSESSION.json")
                             for path in unit_root.glob(pattern)} if unit_root.exists() else set())
    if actual_unit_receipts != expected_unit_receipts:
        raise ValueError("unaccounted unit receipt")

    charges = state.get("charges", {})
    if not isinstance(charges, dict):
        raise ValueError("invalid charges")
    for cid, entry in sorted(charges.items()):
        if (not isinstance(cid, str) or not UNIT_ID.fullmatch(cid) or ".." in cid or
                not isinstance(entry, dict) or entry.get("charge_id") != cid or
                entry.get("stage") not in STAGES):
            raise ValueError("invalid charge")
        cpu = entry.get("cpu_s")
        if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0:
            raise ValueError("invalid charge CPU")
        digest = entry.get("receipt_sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("invalid charge receipt")
        receipt_path = Path(entry.get("receipt_path", "")).resolve()
        receipt_bytes = receipt_path.read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != digest:
            raise ValueError("charge receipt hash mismatch")
        row = dict.fromkeys(FIELDS, "")
        row.update(record_kind="charge", record_id=cid, charge_id=cid,
                   stage=entry["stage"], status="charged", charged_cpu_s=str(cpu),
                   receipt_sha256=digest)
        rows.append(row)
    return rows


def csv_text(rows: list[dict[str, str]]) -> str:
    """Serialize the complete validated snapshot, never a partial stream."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only public R03 UNIT_STATUS proposal")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    try:
        output = csv_text(propose_rows(args.state))
    except Exception:
        print("queue export rejected: invalid or incomplete queue state", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
