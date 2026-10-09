"""Fail-closed task-gate admission before any conditional role access.

This module reads a public R0 gate receipt and the four declared private F
package *bytes*. It has no data-loader import and cannot open role arrays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


SEEDS = ("0", "1")
TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


@dataclass(frozen=True)
class GateAdmission:
    selected_recipe: str
    gate_sha256: str
    source_science_lock_sha256: str
    common_row_unit_pair_sha256: str
    f_package_paths: tuple[tuple[str, str, Path, str], ...]


def admit_conditional_gate(
    gate_path: str | Path,
    *,
    expected_gate_sha256: str,
    expected_source_lock_sha256: str,
    expected_common_row_unit_pair_sha256: str,
    private_root: str | Path,
) -> GateAdmission:
    """Require the selected F to pass every unchanged U0/Ucal point clause."""
    if not all(_full_sha(item) for item in (
            expected_gate_sha256, expected_source_lock_sha256,
            expected_common_row_unit_pair_sha256)):
        raise PermissionError("expected gate or role identity SHA is malformed")
    path = Path(gate_path).resolve(strict=True)
    actual_sha = _sha256(path)
    if actual_sha != expected_gate_sha256:
        raise PermissionError("full-input gate SHA differs from conditional seal")
    gate = json.loads(path.read_text())
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") not in ("A", "B")
            or gate.get("common_role") != "INNER_SELECTION"
            or gate.get("common_row_unit_pair_sha256") !=
            expected_common_row_unit_pair_sha256
            or gate.get("source_science_lock_sha256") !=
            expected_source_lock_sha256):
        raise PermissionError("full-input gate status, recipe, role or source lock differs")
    eligibility = gate.get("per_seed_task")
    if (not isinstance(eligibility, dict)
            or set(eligibility) != set(SEEDS)):
        raise PermissionError("full-input point eligibility seed coverage differs")
    for seed in SEEDS:
        tasks = eligibility[seed]
        if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
            raise PermissionError("full-input point eligibility task coverage differs")
        for task in TASKS:
            references = tasks[task]
            if (not isinstance(references, dict)
                    or set(references) != set(REFERENCES)
                    or any(references[reference] is not True
                           for reference in REFERENCES)):
                raise PermissionError("full-input point eligibility is incomplete or false")
    package_records = gate.get("f_packages")
    if (not isinstance(package_records, dict)
            or set(package_records) != set(SEEDS)):
        raise PermissionError("F package seed coverage differs")
    root = Path(private_root).resolve(strict=True)
    if root.name != ".private":
        raise PermissionError("F package root must be study private custody")
    packages = []
    for seed in SEEDS:
        tasks = package_records[seed]
        if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
            raise PermissionError("F package task coverage differs")
        for task in TASKS:
            record = tasks[task]
            if (not isinstance(record, dict)
                    or set(record) != {"relative_path", "sha256"}
                    or not _full_sha(record["sha256"])):
                raise PermissionError("F package identity is incomplete")
            relative = Path(record["relative_path"])
            if (relative.is_absolute() or ".." in relative.parts
                    or not relative.parts):
                raise PermissionError("F package path escapes private root")
            package = (root / relative).resolve(strict=True)
            if not package.is_relative_to(root):
                raise PermissionError("F package path escapes private root")
            if _sha256(package) != record["sha256"]:
                raise PermissionError("F package bytes differ from full-input gate")
            packages.append((seed, task, package, record["sha256"]))
    return GateAdmission(
        selected_recipe=gate["selected_recipe"],
        gate_sha256=actual_sha,
        source_science_lock_sha256=expected_source_lock_sha256,
        common_row_unit_pair_sha256=expected_common_row_unit_pair_sha256,
        f_package_paths=tuple(packages),
    )

