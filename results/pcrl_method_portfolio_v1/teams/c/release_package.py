"""Private, hash-bound Team C deployment package.

S2 supplies the fixed H-GLOBAL-TEMP load/apply functions. This module never
opens role data or embeds rows, labels, or predictions in the package. The
transmitted view is still constructed solely by ``PurposeRelease``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from prototype import AdultGroupedSchema, PurposeRelease
from training import FittedTaskHead, restore_head


PACKAGE_SCHEMA = "team-c-private-release-package-v1"
CALIBRATOR_SCHEMA = "pcrl-portfolio-global-temp-v1"
SOURCE_KEYS = frozenset((
    "fit_features", "fit_label", "calibration_features", "calibration_label",
))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: str) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _checked_sources(source_role_hashes: Mapping[str, str]) -> dict[str, str]:
    sources = dict(source_role_hashes)
    if set(sources) != SOURCE_KEYS or not all(map(_valid_sha, sources.values())):
        raise ValueError("four full source role hashes are required")
    return {key: sources[key] for key in sorted(SOURCE_KEYS)}


def _inside_root(path: Path, root: Path, *, must_exist: bool) -> Path:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError("private package component escapes private root")
    return resolved.relative_to(root)


def _calibrator_identity(record: Mapping[str, object], head: FittedTaskHead) -> None:
    task = "occupation" if head.purpose == "occupation_group" else head.purpose
    if (record.get("schema") != CALIBRATOR_SCHEMA
            or record.get("task") != task
            or record.get("arm_id") != head.model_id
            or record.get("model_seed") != head.receipt.get("seed")
            or record.get("fit_role") != "CALIBRATION_HELDOUT"
            or record.get("draw_id") != "deterministic"
            or not _valid_sha(record.get("fit_row_unit_pair_sha256"))):
        raise ValueError("calibrator record does not bind this head and held-out role")


@dataclass(frozen=True)
class _BoundCalibrator:
    record: Mapping[str, object]
    apply: Callable[[np.ndarray, dict], np.ndarray]

    def transform(self, raw_probabilities: np.ndarray) -> np.ndarray:
        return self.apply(raw_probabilities, dict(self.record))


def save_release_package(
    head: FittedTaskHead,
    head_checkpoint_path: str | Path,
    head_sha256: str,
    calibrator_record: Mapping[str, object],
    calibrator_path: str | Path,
    calibrator_sha256: str,
    *,
    package_path: str | Path,
    private_root: str | Path,
    campaign_lock_sha256: str,
    data_manifest_sha256: str,
    source_role_hashes: Mapping[str, str],
) -> str:
    """Write once and return the SHA of the private deployment manifest.

    The head checkpoint and S2 calibrator must already exist under the same
    private root. Their bytes are checked before the package is written.
    """
    root = Path(private_root)
    head_path = Path(head_checkpoint_path)
    cal_path = Path(calibrator_path)
    package_path = Path(package_path)
    if not all(map(_valid_sha, (
        head_sha256, calibrator_sha256, campaign_lock_sha256,
        data_manifest_sha256,
    ))):
        raise ValueError("release package requires full SHA-256 bindings")
    sources = _checked_sources(source_role_hashes)
    head_relative = _inside_root(head_path, root, must_exist=True)
    cal_relative = _inside_root(cal_path, root, must_exist=True)
    _inside_root(package_path, root, must_exist=False)
    if _file_sha256(head_path) != head_sha256:
        raise ValueError("head checkpoint SHA-256 mismatch")
    saved_head = restore_head(
        head_path, expected_sha256=head_sha256, private_root=root,
    )
    if (saved_head.purpose != head.purpose
            or saved_head.model_id != head.model_id
            or saved_head.selected_atoms != head.selected_atoms
            or saved_head.selected_columns != head.selected_columns
            or json.dumps(saved_head.receipt, sort_keys=True) != json.dumps(
                head.receipt, sort_keys=True
            )):
        raise ValueError("saved head identity differs from the scored head")
    scored_state = head.model.state_dict()
    saved_state = saved_head.model.state_dict()
    if (tuple(scored_state) != tuple(saved_state)
            or any(not np.array_equal(
                scored_state[key].detach().cpu().numpy(),
                saved_state[key].detach().cpu().numpy(),
            ) for key in scored_state)):
        raise ValueError("saved head weights differ from the scored head")
    if _file_sha256(cal_path) != calibrator_sha256:
        raise ValueError("calibrator file hash mismatch")
    with cal_path.open() as stream:
        stored_record = json.load(stream)
    if stored_record != dict(calibrator_record):
        raise ValueError("calibrator record differs from the saved file")
    _calibrator_identity(stored_record, head)
    schema = AdultGroupedSchema()
    selected_atoms = schema.normalize_atoms(head.selected_atoms)
    if (head.purpose not in ("income", "occupation_group")
            or head.selected_columns != len(schema.column_indices(selected_atoms))
            or not head.model_id):
        raise ValueError("head identity or selection is invalid")
    schema_sha = hashlib.sha256("\n".join(schema.feature_names).encode()).hexdigest()
    if head.receipt.get("feature_schema_sha256") != schema_sha:
        raise ValueError("head schema differs from pinned Adult features")
    package = {
        "schema": PACKAGE_SCHEMA,
        "purpose": head.purpose,
        "arm_id": head.model_id,
        "seed": head.receipt["seed"],
        "selected_atoms": selected_atoms,
        "feature_schema_sha256": schema_sha,
        "head_path": head_relative.as_posix(),
        "head_sha256": head_sha256,
        "calibrator_path": cal_relative.as_posix(),
        "calibrator_sha256": calibrator_sha256,
        "calibrator_fit_row_unit_pair_sha256": stored_record["fit_row_unit_pair_sha256"],
        "campaign_lock_sha256": campaign_lock_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "source_role_hashes": sources,
    }
    payload = (json.dumps(package, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    package_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{package_path.name}.", dir=package_path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, package_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def restore_release_package(
    package_path: str | Path,
    *,
    expected_sha256: str,
    private_root: str | Path,
    expected_campaign_lock_sha256: str,
    expected_data_manifest_sha256: str,
    expected_source_role_hashes: Mapping[str, str],
    load_calibrator: Callable[[Path, str], dict],
    apply_global_temperature: Callable[[np.ndarray, dict], np.ndarray],
) -> PurposeRelease:
    """Verify every private binding before constructing a deployable release."""
    root = Path(private_root)
    package_path = Path(package_path)
    _inside_root(package_path, root, must_exist=True)
    if not _valid_sha(expected_sha256) or _file_sha256(package_path) != expected_sha256:
        raise ValueError("release package SHA-256 mismatch")
    with package_path.open() as stream:
        package = json.load(stream)
    if package.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("unknown release package schema")
    if (package.get("campaign_lock_sha256") != expected_campaign_lock_sha256
            or package.get("data_manifest_sha256") != expected_data_manifest_sha256
            or package.get("source_role_hashes") != _checked_sources(expected_source_role_hashes)):
        raise ValueError("campaign, data manifest, or source role hashes changed")
    head_path = root / package["head_path"]
    cal_path = root / package["calibrator_path"]
    _inside_root(head_path, root, must_exist=True)
    _inside_root(cal_path, root, must_exist=True)
    head = restore_head(
        head_path, expected_sha256=package["head_sha256"], private_root=root,
    )
    record = load_calibrator(cal_path, package["calibrator_sha256"])
    if _file_sha256(cal_path) != package["calibrator_sha256"]:
        raise ValueError("calibrator file hash mismatch")
    _calibrator_identity(record, head)
    if (head.purpose != package["purpose"] or head.model_id != package["arm_id"]
            or head.receipt.get("seed") != package["seed"]
            or list(head.selected_atoms) != package["selected_atoms"]
            or head.receipt.get("feature_schema_sha256") != package["feature_schema_sha256"]
            or record["fit_row_unit_pair_sha256"] != package["calibrator_fit_row_unit_pair_sha256"]):
        raise ValueError("head/calibrator package identity mismatch")
    return PurposeRelease(
        AdultGroupedSchema(), head.selected_atoms, head, head.purpose,
        f"{head.model_id}:{package['head_sha256'][:16]}",
        f"H-GLOBAL-TEMP:{package['calibrator_sha256'][:16]}",
        _BoundCalibrator(record, apply_global_temperature),
    )
