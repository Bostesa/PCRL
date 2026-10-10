"""Admit historical score controls from independent original INNER point audits.

These controls were never fitted against the new B0-relative FIT proxy. Their
eligibility receipt intentionally says that proxy is not applicable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def original_inner_pass(arm: str, audit: dict) -> bool:
    if arm == "W_Q":
        if audit.get("schema") != "pcrl-r5-bank-old-point-audit-v1" or audit.get("status") != "PASS":
            raise PermissionError("historical bank audit identity differs")
        summary = audit.get("summary", {}).get("W_Q", {})
        rows = [row for row in audit.get("rows", []) if row.get("arm") == "W_Q"]
        expected = {(seed, task, ref) for seed in (0, 1)
                    for task in ("income", "occupation") for ref in ("U0", "Ucal")}
        if ({(r.get("seed"), r.get("task"), r.get("reference")) for r in rows} != expected
                or len(rows) != 8 or summary.get("clauses_total") != 24
                or summary.get("cells_total") != 8):
            raise PermissionError("W_Q two-seed original INNER inventory differs")
        count = sum(row.get("point_clauses", {}).get(metric) is True
                    for row in rows for metric in ("accuracy", "nll", "brier"))
        if count != summary.get("clauses_pass"):
            raise PermissionError("W_Q independent clause count differs")
        return count == 24
    if arm == "P_STAR":
        if (audit.get("schema") != "pcrl-bank-r5-pstar-common-inner-replay-v1"
                or audit.get("role") != "INNER_SELECTION"
                or audit.get("status") != "PASS_UTILITY_POINT_ONLY"):
            raise PermissionError("P* independent original INNER replay differs")
        cells = audit.get("cells", [])
        expected = {(seed, task, ref) for seed in (0, 1)
                    for task in ("income", "occupation") for ref in ("U0", "Ucal")}
        if ({(r.get("seed"), r.get("task"), r.get("reference")) for r in cells} != expected
                or len(cells) != 8):
            raise PermissionError("P* two-seed original INNER inventory differs")
        return (audit.get("all_24_point_clauses_pass") is True
                and all(row.get("point_pass") is True for row in cells))
    raise PermissionError("historical score control not registered")


def make_receipt(arm: str, audit_path: Path, output: Path,
                 *, source_replay_path: Path | None = None) -> dict:
    audit_path = audit_path.resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    passed = original_inner_pass(arm, audit)
    source = {"independent_inner_audit_sha256": _sha(audit_path)}
    if arm == "W_Q":
        if source_replay_path is None:
            raise PermissionError("W_Q underlying independent source replay missing")
        source_replay_path = source_replay_path.resolve(strict=True)
        if audit.get("source_replay_sha256") != _sha(source_replay_path):
            raise PermissionError("W_Q audit/source replay SHA differs")
        source["underlying_independent_replay_sha256"] = _sha(source_replay_path)
    elif source_replay_path is not None:
        raise PermissionError("P* receipt has no extra source replay input")
    receipt = {"schema": "pcrl-bank-r3-historical-original-inner-eligibility-v1",
               "arm_id": arm,
               "all_two_seed_task_reference_point_caps_pass": passed,
               "original_inner_point_clauses_passed": 24 if passed else None,
               "original_inner_point_clauses_total": 24,
               "fit_bank_proxy_applied": False,
               "fit_bank_proxy_pass": None,
               "eligible_reference_scope": "historical original U0/Ucal INNER point clauses only",
               "independent_source_hashes": source,
               "scope": "historical development control; no new FIT proxy, interval pass, privacy benefit or confirmation"}
    if not passed:
        raise PermissionError("historical control does not meet original INNER point gate")
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise PermissionError("historical gate output must be write-once")
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    with os.fdopen(os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return {**receipt, "receipt_sha256": _sha(output)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=("W_Q", "P_STAR"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(make_receipt(args.arm, args.audit, args.output,
                                  source_replay_path=args.source_replay), sort_keys=True))


if __name__ == "__main__":
    main()
