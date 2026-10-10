"""Public two-seed point eligibility from R3's queued aggregate utility units.

This assembles recorded FIT and INNER clauses. It never opens private vectors
or reads labels. The gate is exploratory and does not assert interval PASS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .r3_family import ARMS
from .r3_worker import CSV_FIELDS


ROLES = ("OSF_DEFENSE_FIT", "INNER_SELECTION")
TASKS = ("income", "occupation")
METRICS = ("accuracy", "nll", "brier")
REFERENCES = {"OSF_DEFENSE_FIT": ("B0",),
              "INNER_SELECTION": ("U0", "Ucal")}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize_arm(arm: str, docs: dict) -> dict:
    """Require exact four-unit coverage and tally every primary clause."""
    if arm not in ("B0", "T", "D", "P", "L", "J", "S12", "S21") or set(docs) != {(seed, role)
                                        for seed in (0, 1) for role in ROLES}:
        raise PermissionError("arm or four-unit coverage differs")
    seal = None
    fit_pass = inner_pass = 0
    for seed in (0, 1):
        for role in ROLES:
            receipt, rows = docs[(seed, role)]
            count = 6 if role == "OSF_DEFENSE_FIT" else 12
            if (receipt.get("schema") != "pcrl-bank-r3-utility-receipt-v1"
                    or receipt.get("arm") != arm
                    or receipt.get("seed") != seed
                    or receipt.get("role") != role
                    or receipt.get("stage") != ("witness" if arm == "B0" else "protection")
                    or receipt.get("primary_clause_count") != count):
                raise PermissionError("point unit receipt identity differs")
            if seal is None:
                seal = receipt.get("source_seal_sha256")
            if receipt.get("source_seal_sha256") != seal or not isinstance(seal, str):
                raise PermissionError("point units do not share campaign seal")
            expected = {(task, reference, metric)
                        for task in TASKS for reference in REFERENCES[role]
                        for metric in METRICS}
            seen = {}
            for row in rows:
                if row.get("primary_gate") not in ("True", "true", "1", True):
                    continue
                key = (row.get("task"), row.get("reference"), row.get("metric"))
                if (row.get("arm") != arm or str(row.get("seed")) != str(seed)
                        or row.get("role") != role or key in seen
                        or row.get("point_status") not in ("PASS", "FAIL")):
                    raise PermissionError("primary clause identity or status differs")
                seen[key] = row["point_status"]
            if set(seen) != expected:
                raise PermissionError("primary clause inventory incomplete")
            passed = sum(status == "PASS" for status in seen.values())
            if (receipt.get("primary_clause_pass_count") != passed
                    or receipt.get("primary_point_pass") is not (passed == count)):
                raise PermissionError("point receipt and CSV disagree")
            if role == "OSF_DEFENSE_FIT":
                fit_pass += passed
            else:
                inner_pass += passed
    fit_ok = fit_pass == 12
    inner_ok = inner_pass == 24
    return {"schema": "pcrl-bank-r3-arm-point-eligibility-v1", "arm_id": arm,
            "source_seal_sha256": seal,
            "fit_clauses": {"passed": fit_pass, "total": 12},
            "inner_clauses": {"passed": inner_pass, "total": 24},
            "fit_bank_proxy_pass": fit_ok,
            "inner_original_caps_pass": inner_ok,
            "all_two_seed_task_reference_point_caps_pass": fit_ok and inner_ok,
            "gate_semantics": "conjunction of 12 FIT B0-proxy and 24 INNER U0/Ucal point clauses",
            "scope": "development point eligibility, not interval pass or privacy benefit"}


def point_nomination_rows(gates: dict[str, dict],
                          incomplete: set[str] | None = None,
                          historical_gates: dict[str, dict] | None = None) -> list[dict]:
    """Keep the locked J nomination distinct from controls and missing arms."""
    incomplete = set() if incomplete is None else set(incomplete)
    historical_gates = {} if historical_gates is None else dict(historical_gates)
    if (set(gates) - set(ARMS) or incomplete - set(ARMS)
            or set(gates) & incomplete or incomplete & {"B0", "W_Q", "P_STAR"}
            or set(historical_gates) - {"W_Q", "P_STAR"}
            or set(historical_gates) & set(gates)):
        raise PermissionError("unregistered nomination arm")
    out = []
    for arm in ARMS:
        gate = gates.get(arm)
        historical = historical_gates.get(arm)
        if historical is not None:
            if (historical.get("schema") != "pcrl-bank-r3-historical-original-inner-eligibility-v1"
                    or historical.get("arm_id") != arm
                    or historical.get("fit_bank_proxy_applied") is not False
                    or historical.get("original_inner_point_clauses_total") != 24
                    or historical.get("all_two_seed_task_reference_point_caps_pass") is not
                       (historical.get("original_inner_point_clauses_passed") == 24)):
                raise PermissionError("historical original-INNER gate identity differs")
            status = ("HISTORICAL_INNER_PASS" if
                      historical["all_two_seed_task_reference_point_caps_pass"]
                      else "HISTORICAL_INNER_FAIL")
        elif gate is None:
            status = ("INCOMPLETE_SEARCH" if arm in incomplete else
                      "HISTORICAL_SEPARATE" if arm in ("W_Q", "P_STAR") else
                      "NOT_RUN")
        else:
            if (gate.get("schema") != "pcrl-bank-r3-arm-point-eligibility-v1"
                    or gate.get("arm_id") != arm
                    or gate.get("all_two_seed_task_reference_point_caps_pass") is not
                       (gate.get("fit_bank_proxy_pass") is True and
                        gate.get("inner_original_caps_pass") is True)):
                raise PermissionError("nomination gate identity differs")
            status = "PASS" if gate["all_two_seed_task_reference_point_caps_pass"] else "FAIL"
        out.append({"arm_id": arm,
                    "planned_role": ("fixed_bank_baseline" if arm == "B0" else
                                     "locked_joint_nominee" if arm == "J" else
                                     "historical_compatible_control" if arm in ("W_Q", "P_STAR") else
                                     "matched_control"),
                    "utility_gate": status,
                    "nomination": ("J_NOMINEE" if arm == "J" and status == "PASS" else
                                   "NOT_ELIGIBLE" if arm == "J" and status == "FAIL" else
                                   "INCOMPLETE_SEARCH" if arm == "J" and status == "INCOMPLETE_SEARCH" else
                                   "PENDING" if arm == "J" else "CONTROL"),
                    "fresh_attack_eligible": status in ("PASS", "HISTORICAL_INNER_PASS")})
    return out


def _read_unit(receipt_path: Path) -> tuple[dict, list[dict], dict]:
    receipt_path = receipt_path.resolve(strict=True)
    if receipt_path.parent.name != "r3" or not receipt_path.name.endswith("_RECEIPT.json"):
        raise PermissionError("R3 aggregate receipt path differs")
    csv_path = receipt_path.with_name(receipt_path.name.removesuffix("_RECEIPT.json") + ".csv")
    receipt = json.loads(receipt_path.read_text())
    if _sha(csv_path) != receipt.get("output_csv_sha256"):
        raise PermissionError("aggregate CSV differs from queued receipt")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise PermissionError("R3 aggregate CSV schema differs")
        rows = list(reader)
    if len(rows) != 18:
        raise PermissionError("R3 aggregate row count differs")
    identity = {"receipt_sha256": _sha(receipt_path), "csv_sha256": _sha(csv_path),
                "unit_id": receipt.get("unit_id")}
    return receipt, rows, identity


def write_gate(arm: str, input_paths: dict, output: Path) -> dict:
    if set(input_paths) != {(seed, role) for seed in (0, 1) for role in ROLES}:
        raise PermissionError("gate input path inventory differs")
    docs = {}
    sources = {}
    for key, path in input_paths.items():
        receipt, rows, identity = _read_unit(path)
        docs[key] = receipt, rows
        seed, role = key
        sources[f"s{seed}_{'fit' if role == 'OSF_DEFENSE_FIT' else 'inner'}"] = identity
    result = summarize_arm(arm, docs)
    result["source_receipts"] = sources
    output = output.resolve(strict=False)
    if (output.parent.name != "r3" or output.exists() or output.is_symlink()
            or not output.parent.is_dir()):
        raise PermissionError("new point gate output path differs")
    payload = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return {**result, "receipt_sha256": _sha(output)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, required=True)
    for seed in (0, 1):
        for suffix in ("fit", "inner"):
            parser.add_argument(f"--s{seed}-{suffix}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {(seed, role): getattr(args, f"s{seed}_{suffix}")
             for seed in (0, 1)
             for role, suffix in (("OSF_DEFENSE_FIT", "fit"),
                                  ("INNER_SELECTION", "inner"))}
    print(json.dumps(write_gate(args.arm, paths, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
