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


def _private_file(root: Path, relative_value: object, expected_sha: object,
                  label: str) -> Path:
    if not isinstance(relative_value, str) or not _full_sha(expected_sha):
        raise PermissionError(f"{label} identity is incomplete")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PermissionError(f"{label} path escapes private root")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError(f"{label} path escapes private root")
    if _sha256(path) != expected_sha:
        raise PermissionError(f"{label} bytes differ from receipt")
    return path


def _check_f_manifest(package: Path, root: Path, *, seed: int, task: str,
                      recipe: str) -> tuple[str, str, str, str]:
    content = json.loads(package.read_text())
    if (content.get("schema") != "pcrl-utility-full-F-task-package-v1"
            or content.get("seed") != seed or content.get("task") != task
            or content.get("selected_recipe") != recipe
            or not all(_full_sha(content.get(key)) for key in (
                "selection_lock_sha256", "feature_schema_sha256",
                "fit_row_unit_sha256", "calibration_row_unit_sha256"))):
        raise PermissionError("F package task, recipe or training identity differs")
    model = content.get("model")
    if not isinstance(model, dict):
        raise PermissionError("F model component identity is incomplete")
    if recipe == "A":
        _private_file(root, model.get("relative_path"), model.get("sha256"),
                      "F model component")
    else:
        relative = model.get("relative_directory")
        hashes = model.get("sha256")
        if (not isinstance(relative, str) or not isinstance(hashes, dict)
                or set(hashes) != {"encoder.pt", "head_0.joblib", "head_1.joblib"}):
            raise PermissionError("F model component bundle is incomplete")
        for name, digest in hashes.items():
            _private_file(root, str(Path(relative) / name), digest,
                          "F model component")
    calibrator = content.get("calibrator")
    if not isinstance(calibrator, dict):
        raise PermissionError("F calibrator component identity is incomplete")
    _private_file(root, calibrator.get("relative_path"),
                  calibrator.get("sha256"), "F calibrator component")
    return tuple(content[key] for key in (
        "selection_lock_sha256", "feature_schema_sha256",
        "fit_row_unit_sha256", "calibration_row_unit_sha256",
    ))


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
    receipt_records = gate.get("f_receipts")
    if (not isinstance(receipt_records, dict)
            or set(receipt_records) != set(SEEDS)):
        raise PermissionError("F package receipt seed coverage differs")
    root = Path(private_root).resolve(strict=True)
    if root.name != ".private":
        raise PermissionError("F package root must be study private custody")
    packages = []
    package_identities = set()
    for seed in SEEDS:
        receipt_record = receipt_records[seed]
        if (not isinstance(receipt_record, dict)
                or set(receipt_record) != {"relative_path", "sha256"}):
            raise PermissionError("F package receipt identity is incomplete")
        receipt_path = _private_file(
            root, receipt_record["relative_path"], receipt_record["sha256"],
            "F package receipt")
        expected_receipt = root / "r1" / "full" / f"seed_{seed}" / "F_PACKAGE_RECEIPT.json"
        if receipt_path != expected_receipt:
            raise PermissionError("F package receipt path differs from R1 custody")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "pcrl-utility-full-F-package-v1"
                or receipt.get("seed") != int(seed)
                or receipt.get("selected_recipe") != gate["selected_recipe"]
                or not all(_full_sha(receipt.get(key)) for key in (
                    "selection_lock_sha256", "fit_row_unit_sha256",
                    "calibration_row_unit_sha256"))
                or not isinstance(receipt.get("packages"), dict)
                or set(receipt.get("task_packages", {})) != set(TASKS)):
            raise PermissionError("F package receipt content differs")
        tasks = package_records[seed]
        if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
            raise PermissionError("F package task coverage differs")
        for task in TASKS:
            record = tasks[task]
            if (not isinstance(record, dict)
                    or set(record) != {"relative_path", "sha256"}
                    or not _full_sha(record["sha256"])):
                raise PermissionError("F package identity is incomplete")
            if record != receipt["task_packages"][task]:
                raise PermissionError("F package receipt/task identity differs")
            package = _private_file(
                root, record["relative_path"], record["sha256"], "F package")
            identity = _check_f_manifest(
                package, root, seed=int(seed), task=task,
                recipe=gate["selected_recipe"])
            content = json.loads(package.read_text())
            model_key = task if gate["selected_recipe"] == "A" else "B_source_bundle"
            if (receipt["selection_lock_sha256"] != identity[0]
                    or receipt["fit_row_unit_sha256"] != identity[2]
                    or receipt["calibration_row_unit_sha256"] != identity[3]
                    or receipt["packages"].get(model_key) != content["model"]
                    or receipt["packages"].get(f"calibrator_{task}") !=
                    content["calibrator"]):
                raise PermissionError("F package receipt/component identity differs")
            package_identities.add(identity)
            packages.append((seed, task, package, record["sha256"]))
    if len(package_identities) != 1:
        raise PermissionError("F packages disagree on source, selection or roles")
    return GateAdmission(
        selected_recipe=gate["selected_recipe"],
        gate_sha256=actual_sha,
        source_science_lock_sha256=expected_source_lock_sha256,
        common_row_unit_pair_sha256=expected_common_row_unit_pair_sha256,
        f_package_paths=tuple(packages),
    )
