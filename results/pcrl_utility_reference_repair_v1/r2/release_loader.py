"""Restore the exact masked task package into a complete selected-input release.

The public runtime sees only selected per-person values, a frozen task head,
the one held-out global calibrator, and public package metadata. No task or
protected label is a runtime input.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file
from results.pcrl_utility_reference_repair_v1.r1.source_task import (
    predict_source, restore_source,
)
from results.pcrl_utility_reference_repair_v1.r2.mask_adapter import (
    FixedFillSelectedHead,
)


REPO = Path(__file__).resolve().parents[3]
for source_dir in (
    REPO / "results/pcrl_method_portfolio_v1/teams/c",
    REPO / "results/pcrl_method_portfolio_v1/shared",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
from training import restore_head  # noqa: E402
from prototype import AdultGroupedSchema, PurposeRelease  # noqa: E402
from results.pcrl_method_portfolio_v1.shared import common_eval  # noqa: E402


def _full_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _private_file(root: Path, record: object, label: str) -> Path:
    if (not isinstance(record, dict)
            or not isinstance(record.get("relative_path"), str)
            or not _full_sha(record.get("sha256"))):
        raise PermissionError(f"{label} identity is incomplete")
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError(f"{label} path escapes private root")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or sha_file(path) != record["sha256"]:
        raise PermissionError(f"{label} bytes or path differ")
    return path


class _SourcePurposeHead:
    def __init__(self, model, heads, index: int):
        self.model = model
        self.heads = heads
        self.index = index

    def predict_proba(self, full_x83):
        return predict_source(self.model, self.heads, full_x83)[self.index]


class _GlobalCalibrator:
    def __init__(self, record: dict):
        self.record = record

    def transform(self, raw):
        return common_eval.apply_global_temperature(raw, self.record)


@dataclass(frozen=True)
class BoundPurposeRelease:
    """Add the bank's exact public ID and package SHA to the inherited wire."""

    release: PurposeRelease
    mask_id: str
    mask_sha256: str
    task_package_sha256: str

    @property
    def purpose(self):
        return self.release.purpose

    @property
    def selected_atoms(self):
        return self.release.selected_atoms

    @property
    def schema(self):
        return self.release.schema

    @property
    def head_id(self):
        return self.release.head_id

    @property
    def calibrator_id(self):
        return self.release.calibrator_id

    def view(self, x83, feature_names):
        view = self.release.view(x83, feature_names)
        view["mask_id"] = self.mask_id
        view["mask_sha256"] = self.mask_sha256
        view["task_package_sha256"] = self.task_package_sha256
        return view


def restore_mask_task_release(
    private_root: str | Path,
    fit_receipt_record: dict,
    mask: object,
    task: str,
    seed: int,
    recipe: str,
    *,
    gate_sha256: str,
    conditional_lock_sha256: str,
) -> BoundPurposeRelease:
    """Verify a private fit receipt, task package, head and CAL before use."""
    if (task not in ("income", "occupation_group")
            or recipe not in ("A", "B") or seed not in (0, 1, 2)
            or not _full_sha(gate_sha256)
            or not _full_sha(conditional_lock_sha256)):
        raise ValueError("masked task release identity is invalid")
    root = Path(private_root).resolve(strict=True)
    if root.name != ".private":
        raise PermissionError("masked release requires private study custody")
    receipt_path = _private_file(root, fit_receipt_record, "mask fit receipt")
    expected = root / "masks" / mask.mask_id / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
    if receipt_path != expected:
        raise PermissionError("mask fit receipt path differs")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "pcrl-utility-mask-fit-receipt-v1"
            or receipt.get("mask_id") != mask.mask_id
            or receipt.get("mask_sha256") != mask.mask_sha256
            or receipt.get("seed") != seed
            or receipt.get("selected_recipe") != recipe
            or receipt.get("gate_sha256") != gate_sha256
            or receipt.get("conditional_lock_sha256") != conditional_lock_sha256
            or receipt.get("selected_atoms") != list(mask.atoms)
            or receipt.get("source_indices") != list(mask.column_indices)
            or set(receipt.get("task_packages", {})) !=
            {"income", "occupation_group"}):
        raise PermissionError("masked fit receipt identity differs")
    package_path = _private_file(
        root, receipt["task_packages"][task], "masked task package")
    package = json.loads(package_path.read_text())
    if (package.get("schema") != "pcrl-utility-masked-task-package-v1"
            or package.get("seed") != seed or package.get("task") != task
            or package.get("selected_recipe") != recipe
            or package.get("mask_id") != mask.mask_id
            or package.get("mask_sha256") != mask.mask_sha256
            or package.get("selected_atoms") != list(mask.atoms)
            or package.get("source_indices") != list(mask.column_indices)
            or package.get("source_schema_sha256") != mask.source_schema_sha256
            or package.get("gate_sha256") != gate_sha256
            or package.get("conditional_lock_sha256") != conditional_lock_sha256
            or package.get("fit_row_unit_sha256") !=
            receipt.get("fit_row_unit_sha256")
            or package.get("calibration_row_unit_sha256") !=
            receipt.get("calibration_row_unit_sha256")
            or package.get("calibrator") != receipt.get("calibrators", {}).get(task)
            or package.get("fill_constants") != receipt.get("fills", {}).get("final")):
        raise PermissionError("masked task package and fit receipt differ")
    model_record = package.get("model")
    if recipe == "A":
        if model_record != receipt.get("model_stages", {}).get("final", {}).get(task):
            raise PermissionError("A checkpoint differs from fit receipt")
        model_path = _private_file(root, model_record, "A masked checkpoint")
        head = restore_head(model_path, expected_sha256=model_record["sha256"],
                            private_root=root)
        if (head.model_id != package.get("model_id")
                or head.purpose != task or tuple(head.selected_atoms) != mask.atoms):
            raise PermissionError("A checkpoint metadata differs from task package")
        head_id = f"{head.model_id}:SHA256:{model_record['sha256']}"
    else:
        if model_record != receipt.get("model_stages", {}).get("final"):
            raise PermissionError("B source bundle differs from fit receipt")
        if (not isinstance(model_record, dict)
                or not isinstance(model_record.get("relative_directory"), str)
                or set(model_record.get("sha256", {})) !=
                {"encoder.pt", "head_0.joblib", "head_1.joblib"}):
            raise PermissionError("B source bundle identity is incomplete")
        bundle = (root / model_record["relative_directory"]).resolve(strict=True)
        if not bundle.is_relative_to(root):
            raise PermissionError("B source bundle escapes private root")
        model, heads = restore_source(bundle, seed, model_record["sha256"])
        fill_path = _private_file(root, package["fill_constants"],
                                  "B fixed fill")
        fill = np.load(fill_path, allow_pickle=False)
        joint_sha = hashlib.sha256(json.dumps(
            model_record["sha256"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        head = FixedFillSelectedHead(
            purpose=task, selected_atoms=mask.atoms,
            source_indices=mask.column_indices,
            full_input_head=_SourcePurposeHead(model, heads,
                                                0 if task == "income" else 1),
            base_model_id=package["model_id"],
            base_model_sha256=joint_sha,
            fill_constants=fill,
            source_schema_sha256=mask.source_schema_sha256,
        )
        head_id = head.model_id
    calibrator_path = _private_file(root, package["calibrator"],
                                    "masked CAL calibrator")
    calibrator = common_eval.load_calibrator(
        calibrator_path, package["calibrator"]["sha256"])
    expected_label = "income" if task == "income" else "occupation"
    if (calibrator.get("task") != expected_label
            or calibrator.get("arm_id") != package["model_id"]
            or calibrator.get("model_seed") != seed):
        raise PermissionError("masked calibrator does not bind task head")
    release = PurposeRelease(
        schema=AdultGroupedSchema(), selected_atoms=mask.atoms,
        head=head, purpose=task, head_id=head_id,
        calibrator_id=f"{package['model_id']}:SHA256:{package['calibrator']['sha256']}",
        calibrator=_GlobalCalibrator(calibrator),
    )
    return BoundPurposeRelease(
        release=release, mask_id=mask.mask_id,
        mask_sha256=mask.mask_sha256,
        task_package_sha256=receipt["task_packages"][task]["sha256"],
    )
