"""Independent restoration of C's hash-bound public task releases.

This reads saved package, checkpoint and calibrator files as data. No Team C
training, package or shared evaluator implementation is imported.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from replay_c_values_v1 import (atom_indices, head_probabilities,
                                load_head_checkpoint, temperature_apply)


PURPOSES = ("income", "occupation_group")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_file(root: Path, relative: str, digest: str) -> Path:
    if (not isinstance(relative, str) or Path(relative).is_absolute() or
            not isinstance(digest, str) or len(digest) != 64 or
            any(ch not in "0123456789abcdef" for ch in digest)):
        raise ValueError("C private artifact identity is incomplete")
    root = root.resolve(strict=True)
    if root.name != ".private":
        raise ValueError("C private artifact root differs")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or _sha(path) != digest:
        raise ValueError("C private artifact path or SHA differs")
    return path


def load_packages(
    fit_handoff: Mapping[str, object], private_root: str | Path,
    feature_names: Sequence[str], lock_sha256: str, manifest_sha256: str,
    expected_role_sha256: Mapping[str, str],
) -> dict[str, dict]:
    """Restore the two public model packages from a frozen fit handoff."""
    if (fit_handoff.get("schema") != "team-c-fit-handoff-v1" or
            fit_handoff.get("campaign_lock_sha256") != lock_sha256 or
            fit_handoff.get("data_manifest_sha256") != manifest_sha256 or
            fit_handoff.get("fit_cal_role_file_hashes") != dict(expected_role_sha256)):
        raise ValueError("C fit handoff or admitted role source differs")
    units = fit_handoff.get("arm_receipt", {}).get("package_units")
    if (not isinstance(units, list) or len(units) != 2 or
            [unit.get("purpose") for unit in units] != list(PURPOSES) or
            len(fit_handoff.get("head_ids", ())) != 2 or
            len(fit_handoff.get("calibrator_ids", ())) != 2):
        raise ValueError("C fit handoff lacks both ordered public packages")
    atoms = atom_indices(feature_names)
    root = Path(private_root)
    result = {}
    for index, unit in enumerate(units):
        purpose = unit["purpose"]
        package_path = _private_file(root, unit["package_relative_path"],
                                     unit["package_sha256"])
        package = json.loads(package_path.read_text())
        selected = tuple(package.get("selected_atoms", ()))
        if (package.get("schema") != "team-c-private-release-package-v1" or
                package.get("purpose") != purpose or
                f"{package.get('arm_id')}:{package.get('head_sha256', '')[:16]}" !=
                fit_handoff["head_ids"][index] or
                package.get("seed") != fit_handoff.get("seed") or
                package.get("campaign_lock_sha256") != lock_sha256 or
                package.get("data_manifest_sha256") != manifest_sha256 or
                package.get("source_role_hashes") != unit.get("source_role_hashes") or
                selected != tuple(fit_handoff.get("selected_atoms", {}).get(purpose, ())) or
                selected != tuple(atom for atom in atoms if atom in selected) or
                package.get("feature_schema_sha256") != hashlib.sha256(
                    "\n".join(feature_names).encode()).hexdigest()):
            raise ValueError("C public package identity or mask differs")
        sources = package["source_role_hashes"]
        task = "occupation" if purpose == "occupation_group" else "income"
        required = {
            "fit_features": expected_role_sha256["OSF_DEFENSE_FIT:features"],
            "fit_label": expected_role_sha256[f"OSF_DEFENSE_FIT:{task}"],
            "calibration_features": expected_role_sha256["CALIBRATION_HELDOUT:features"],
            "calibration_label": expected_role_sha256[f"CALIBRATION_HELDOUT:{task}"],
        }
        if sources != required:
            raise ValueError("C package source role hashes differ")
        head_path = _private_file(root, package["head_path"],
                                  package["head_sha256"])
        head = load_head_checkpoint(head_path, package["head_sha256"], feature_names)
        meta = head["metadata"]
        if (meta.get("purpose") != purpose or meta.get("model_id") != package["arm_id"] or
                meta.get("receipt", {}).get("seed") != package["seed"] or
                head["selected_atoms"] != selected):
            raise ValueError("C public head metadata differs from package")
        cal_path = _private_file(root, package["calibrator_path"],
                                 package["calibrator_sha256"])
        record = json.loads(cal_path.read_text())
        if (record.get("schema") != "pcrl-portfolio-global-temp-v1" or
                record.get("task") != task or
                record.get("arm_id") != package["arm_id"] or
                record.get("model_seed") != package["seed"] or
                record.get("fit_role") != "CALIBRATION_HELDOUT" or
                record.get("draw_id") != "deterministic" or
                record.get("fit_row_unit_pair_sha256") !=
                package.get("calibrator_fit_row_unit_pair_sha256") or
                record.get("argmax_preserved") is not True or
                f"H-GLOBAL-TEMP:{package['calibrator_sha256'][:16]}" !=
                fit_handoff["calibrator_ids"][index]):
            raise ValueError("C public calibrator identity differs")
        result[purpose] = {"package": package, "head": head,
                           "calibrator": record,
                           "package_sha256": unit["package_sha256"]}
    return result


def predict_release(package: Mapping[str, object], role_x: np.ndarray,
                    feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Forward one independently restored head and global temperature map."""
    atoms = atom_indices(feature_names)
    selected = tuple(package["head"]["selected_atoms"])
    columns = sorted(index for atom in selected for index in atoms[atom])
    x = np.asarray(role_x)
    if x.ndim != 2 or x.shape[1] != 83 or not np.isfinite(x).all():
        raise ValueError("C role feature matrix differs")
    raw = head_probabilities(package["head"], x[:, columns])
    calibrated = temperature_apply(raw, package["calibrator"])
    return raw, calibrated
