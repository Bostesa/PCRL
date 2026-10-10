"""Corrected, write-once INNER-only attack admission from immutable R3 units.

The original combined FIT/INNER gates remain on disk as historical error
records. This successor changes no task costs, thresholds or source rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .r3_eligibility import _read_unit, summarize_arm
from .r3_family import ARMS


NEW_ARMS = ("B0", "T", "D", "P", "L", "J", "S12", "S21")
SOURCE_KEYS = ("s0_fit", "s0_inner", "s1_fit", "s1_inner")
NOMINATION_FIELDS = ("arm_id", "planned_role", "attack_utility_gate",
                     "fit_bank_proxy_pass", "nomination",
                     "fresh_attack_eligible", "gate_receipt_sha256")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_new(path: Path, payload: bytes) -> str:
    if (path.is_symlink() or path.exists() or path.parent.name != "r3"
            or not path.parent.is_dir()):
        raise PermissionError("R3 successor must be a new r3 file")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(path)


def correct_attack_gate(old: dict, old_sha256: str) -> dict:
    """Use only original INNER 24/24 as the sealed attack admission condition."""
    fit = old.get("fit_clauses", {})
    inner = old.get("inner_clauses", {})
    if (old.get("schema") != "pcrl-bank-r3-arm-point-eligibility-v1"
            or old.get("arm_id") not in NEW_ARMS
            or not isinstance(old.get("source_seal_sha256"), str)
            or len(old["source_seal_sha256"]) != 64
            or not isinstance(old_sha256, str) or len(old_sha256) != 64
            or fit.get("total") != 12 or inner.get("total") != 24
            or type(fit.get("passed")) is not int or not 0 <= fit["passed"] <= 12
            or type(inner.get("passed")) is not int or not 0 <= inner["passed"] <= 24
            or old.get("fit_bank_proxy_pass") is not (fit["passed"] == 12)
            or old.get("inner_original_caps_pass") is not (inner["passed"] == 24)
            or old.get("all_two_seed_task_reference_point_caps_pass") is not
               (fit["passed"] == 12 and inner["passed"] == 24)
            or set(old.get("source_receipts", {})) != set(SOURCE_KEYS)):
        raise PermissionError("prior combined gate identity or counts differ")
    return {
        "schema": "pcrl-bank-r3-inner-attack-eligibility-v2",
        "arm_id": old["arm_id"],
        "source_seal_sha256": old["source_seal_sha256"],
        "fit_clauses": fit,
        "inner_clauses": inner,
        "fit_bank_proxy_pass": fit["passed"] == 12,
        "inner_original_caps_pass": inner["passed"] == 24,
        "all_two_seed_task_reference_point_caps_pass": inner["passed"] == 24,
        "gate_semantics": "attack admission uses original INNER 24/24 only; FIT B0 proxy is separate",
        "scope": "development point attack eligibility, not interval pass or privacy benefit",
        "source_receipts": old["source_receipts"],
        "supersedes_combined_gate_sha256": old_sha256,
    }


def write_attack_gate(old_path: Path, output: Path) -> dict:
    """Recheck old gate against four immutable aggregate units before writing V2."""
    if old_path.is_symlink():
        raise PermissionError("old gate symlink refused")
    old_path = old_path.resolve(strict=True)
    old = json.loads(old_path.read_text())
    gate = correct_attack_gate(old, _sha(old_path))
    docs = {}
    for seed in (0, 1):
        for role, suffix in (("OSF_DEFENSE_FIT", "fit"),
                             ("INNER_SELECTION", "inner")):
            key = f"s{seed}_{suffix}"
            unit_id = old["source_receipts"][key]["unit_id"]
            if not isinstance(unit_id, str) or "/" in unit_id or ".." in unit_id:
                raise PermissionError("source unit path differs")
            receipt, rows, identity = _read_unit(old_path.parent / f"{unit_id}_RECEIPT.json")
            if identity != old["source_receipts"][key]:
                raise PermissionError("old gate source bytes differ")
            docs[(seed, role)] = receipt, rows
    fresh = summarize_arm(gate["arm_id"], docs)
    for key in ("source_seal_sha256", "fit_clauses", "inner_clauses",
                "fit_bank_proxy_pass", "inner_original_caps_pass",
                "all_two_seed_task_reference_point_caps_pass"):
        if fresh[key] != old[key]:
            raise PermissionError("old gate and current aggregate units differ")
    payload = (json.dumps(gate, sort_keys=True, indent=2) + "\n").encode()
    sha = _write_new(output, payload)
    return {**gate, "receipt_sha256": sha}


def attack_nomination_rows(gates: dict[str, dict],
                           historical_gates: dict[str, dict] | None = None) -> list[dict]:
    """Keep all fixed arms while J nomination follows INNER eligibility only."""
    historical_gates = {} if historical_gates is None else dict(historical_gates)
    if set(gates) - set(NEW_ARMS) or set(historical_gates) - {"W_Q", "P_STAR"}:
        raise PermissionError("attack nomination arm differs")
    out = []
    for arm in ARMS:
        fit_status = ""
        gate = gates.get(arm)
        if gate is not None:
            if (gate.get("schema") != "pcrl-bank-r3-inner-attack-eligibility-v2"
                    or gate.get("arm_id") != arm
                    or gate.get("inner_clauses", {}).get("total") != 24
                    or gate.get("all_two_seed_task_reference_point_caps_pass") is not
                       (gate["inner_clauses"].get("passed") == 24)
                    or gate.get("inner_original_caps_pass") is not
                       gate["all_two_seed_task_reference_point_caps_pass"]):
                raise PermissionError("effective attack gate differs")
            status = "PASS" if gate["all_two_seed_task_reference_point_caps_pass"] else "FAIL"
            fit_status = str(gate["fit_bank_proxy_pass"])
        elif arm in historical_gates:
            h = historical_gates[arm]
            if (h.get("schema") != "pcrl-bank-r3-historical-original-inner-eligibility-v1"
                    or h.get("arm_id") != arm or h.get("fit_bank_proxy_applied") is not False
                    or h.get("original_inner_point_clauses_total") != 24
                    or h.get("all_two_seed_task_reference_point_caps_pass") is not
                       (h.get("original_inner_point_clauses_passed") == 24)):
                raise PermissionError("historical INNER gate differs")
            status = "HISTORICAL_INNER_PASS" if h["all_two_seed_task_reference_point_caps_pass"] else "HISTORICAL_INNER_FAIL"
        else:
            status = "NOT_RUN" if arm in NEW_ARMS else "HISTORICAL_SEPARATE"
        out.append({"arm_id": arm,
                    "planned_role": ("fixed_bank_baseline" if arm == "B0" else
                                     "locked_joint_nominee" if arm == "J" else
                                     "historical_compatible_control" if arm in ("W_Q", "P_STAR") else
                                     "matched_control"),
                    "attack_utility_gate": status,
                    "fit_bank_proxy_pass": fit_status,
                    "nomination": ("J_NOMINEE" if arm == "J" and status == "PASS" else
                                   "NOT_ELIGIBLE" if arm == "J" and status == "FAIL" else
                                   "PENDING" if arm == "J" else "CONTROL"),
                    "fresh_attack_eligible": status in ("PASS", "HISTORICAL_INNER_PASS")})
    return out


def write_attack_nomination(gate_paths: list[Path], historical_paths: list[Path],
                            output: Path) -> dict:
    groups, hashes = [], {}
    for paths in (gate_paths, historical_paths):
        docs = {}
        for path in paths:
            if path.is_symlink():
                raise PermissionError("nomination gate symlink refused")
            path = path.resolve(strict=True)
            gate = json.loads(path.read_text())
            arm = gate.get("arm_id")
            if arm in docs or arm not in ARMS:
                raise PermissionError("duplicate or unknown attack gate")
            docs[arm] = gate
            hashes[arm] = _sha(path)
        groups.append(docs)
    rows = attack_nomination_rows(groups[0], groups[1])
    for row in rows:
        row["gate_receipt_sha256"] = hashes.get(row["arm_id"], "")
    if output.is_symlink() or output.exists() or output.parent.name != "r3":
        raise PermissionError("attack nomination must be new R3 file")
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644),
                   "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=NOMINATION_FIELDS)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    return {"schema": "pcrl-bank-r3-inner-attack-nomination-v2",
            "table_sha256": _sha(output), "rows": len(rows),
            "source_gate_sha256_by_arm": hashes,
            "scope": "INNER point eligibility only; FIT proxy reported separately"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="operation", required=True)
    gate = sub.add_parser("gate", allow_abbrev=False)
    gate.add_argument("--old-gate", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    nomination = sub.add_parser("nomination", allow_abbrev=False)
    nomination.add_argument("--new-gate", type=Path, action="append", default=[])
    nomination.add_argument("--historical-gate", type=Path, action="append", default=[])
    nomination.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "gate":
        result = write_attack_gate(args.old_gate, args.output)
    else:
        result = write_attack_nomination(args.new_gate, args.historical_gate, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
