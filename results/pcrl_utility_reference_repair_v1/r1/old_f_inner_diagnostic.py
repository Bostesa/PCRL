"""Separate same-role diagnostic for the prior focused validation-stage F.

This restores only the prior hash-bound full-input head/calibrator packages.
It never opens the predecessor's mixed-role release, assessment, or label data.
The contrast is observational because the old F used fewer FIT labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import array_sha
from results.pcrl_utility_reference_repair_v1.r1.fit_worker import (
    _metrics, _role, _write_once_json, _write_once_npz,
)
from results.pcrl_utility_reference_repair_v1.r1.role_store import RoleOnlyStore, sha_file


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
C_DIR = REPO / "results/pcrl_method_portfolio_v1/teams/c"
SHARED_DIR = REPO / "results/pcrl_method_portfolio_v1/shared"
for directory in (C_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from release_package import restore_release_package  # noqa: E402
from results.pcrl_method_portfolio_v1.shared import common_eval  # noqa: E402


TASKS = ("income", "occupation_group")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def _inside(root: Path, relative: str, basename: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise PermissionError("old F input path must be relative")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.name != basename:
        raise PermissionError("old F input path escapes exact admitted component")
    return path


def validate_old_f_inputs(old_private_root: Path, inputs: dict,
                          seed: int) -> dict[str, dict]:
    """Hash and validate only old FIT receipt, F package, head, CAL, receipt."""
    old = old_private_root.resolve(strict=True)
    seed_record = inputs["seed"][str(seed)]
    if set(seed_record["tasks"]) != set(TASKS):
        raise PermissionError("old F diagnostic needs both purpose packages")
    for task in TASKS:
        e = seed_record["tasks"][task]
        if e.get("stage") != "validation" or e.get("mask_id") != "F_FULL83":
            raise PermissionError("old F diagnostic admits validation F_FULL83 only")
    fit_path = _inside(old, seed_record["fit_receipt_relative_path"], "FIT_RECEIPT.json")
    if sha_file(fit_path) != seed_record["fit_receipt_sha256"]:
        raise PermissionError("old FIT receipt differs from pinned diagnostic input")
    fit = json.loads(fit_path.read_text())
    if fit.get("seed") != seed or fit.get("status") != "NO_FEASIBLE_J":
        raise PermissionError("old FIT receipt identity differs")
    result = {}
    path_fields = (("relative_package_path", "package_sha256", "release_package.json"),
                   ("relative_head_pool_receipt_path", "head_pool_receipt_sha256",
                    "HEAD_POOL_RECEIPT.json"),
                   ("relative_head_path", "head_sha256", "head.npz"),
                   ("relative_calibrator_path", "calibrator_sha256", "calibrator.json"))
    for task in TASKS:
        entry = seed_record["tasks"][task]
        inherited = fit["validation_head_packages"][task]["F_FULL83"]
        for key, inherited_key in (("relative_package_path", "relative_package_path"),
                                   ("package_sha256", "package_sha256"),
                                   ("relative_head_pool_receipt_path",
                                    "head_pool_receipt_relative_path"),
                                   ("head_pool_receipt_sha256", "head_pool_receipt_sha256"),
                                   ("head_sha256", "head_sha256"),
                                   ("calibrator_sha256", "calibrator_sha256")):
            if entry[key] != inherited[inherited_key]:
                raise PermissionError("old F entry differs from frozen FIT receipt")
        if inherited.get("stage") != "validation" or inherited.get("mask_id") != "F_FULL83":
            raise PermissionError("old FIT receipt lacks validation F_FULL83")
        paths = {}
        for key, sha_key, basename in path_fields:
            path = _inside(old, entry[key], basename)
            if sha_file(path) != entry[sha_key]:
                raise PermissionError("old F component SHA differs")
            paths[basename] = path
        package = json.loads(paths["release_package.json"].read_text())
        receipt = json.loads(paths["HEAD_POOL_RECEIPT.json"].read_text())
        identity = receipt["cache_identity"]
        if (package.get("schema") != "team-c-private-release-package-v1"
                or package.get("purpose") != task or package.get("seed") != seed
                or package.get("head_path") != entry["relative_head_path"]
                or package.get("head_sha256") != entry["head_sha256"]
                or package.get("calibrator_path") != entry["relative_calibrator_path"]
                or package.get("calibrator_sha256") != entry["calibrator_sha256"]
                or package.get("campaign_lock_sha256") != fit["campaign_lock_sha256"]
                or package.get("data_manifest_sha256") != fit["data_manifest_sha256"]
                or identity.get("mask_id") != "F_FULL83"
                or identity.get("purpose") != task or identity.get("seed") != seed
                or identity.get("campaign_lock_sha256") != fit["campaign_lock_sha256"]
                or identity.get("data_manifest_sha256") != fit["data_manifest_sha256"]
                or identity.get("role_file_sha256") != package.get("source_role_hashes")
                or identity.get("selected_columns") != list(range(83))
                or receipt.get("fit_rows") != (11574 if seed == 0 else 11575)
                or receipt.get("calibration_rows") != 2000):
            raise PermissionError("old F package/receipt identity differs")
        result[task] = {
            "package_path": paths["release_package.json"],
            "package_sha256": entry["package_sha256"],
            "campaign_lock_sha256": fit["campaign_lock_sha256"],
            "data_manifest_sha256": fit["data_manifest_sha256"],
            "source_role_hashes": package["source_role_hashes"],
            "fit_rows": receipt["fit_rows"],
            "artifact_hashes": {sha_key: entry[sha_key] for _, sha_key, _ in path_fields},
        }
    return result


def _completed_primary_gate(gate_path: Path, expected_sha256: str) -> dict:
    if sha_file(gate_path) != expected_sha256:
        raise PermissionError("primary F gate differs from diagnostic lock")
    gate = json.loads(gate_path.read_text())
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") not in ("PASS_FULL_INPUT_POINT_GATE",
                                          "FAIL_FULL_INPUT_POINT_GATE")):
        raise PermissionError("primary F gate must finish before old F diagnostic")
    return gate


def run(store: RoleOnlyStore, seed: int, old_root: Path, inputs: dict,
        new_root: Path, output: Path, primary_gate_sha256: str) -> dict:
    old_packages = validate_old_f_inputs(old_root, inputs, seed)
    restored = {}
    for task in TASKS:
        e = old_packages[task]
        restored[task] = restore_release_package(
            e["package_path"], expected_sha256=e["package_sha256"],
            private_root=old_root,
            expected_campaign_lock_sha256=e["campaign_lock_sha256"],
            expected_data_manifest_sha256=e["data_manifest_sha256"],
            expected_source_role_hashes=e["source_role_hashes"],
            load_calibrator=common_eval.load_calibrator,
            apply_global_temperature=common_eval.apply_global_temperature,
        )
    feature, yi, yo, refs = _role(store, "INNER_SELECTION", references_seed=seed)
    names = tuple(str(name) for name in feature["feature_names"])
    prob = tuple(restored[task].view(feature["X"], names)["probabilities"]
                 for task in TASKS)
    arrays = {"row_id": feature["row_id"], "unit": feature["unit"],
              "income": yi, "occupation": yo,
              "old_F_income": prob[0], "old_F_occupation_group": prob[1]}
    arrays.update({key: refs[key] for key in ("U0_1", "U0_2", "Ucal_1", "Ucal_2")})
    pred_path = output.parent / "OLD_F_INNER_PREDICTIONS.npz"
    pred_sha = _write_once_npz(pred_path, **arrays)
    metric = {"old_validation_F": _metrics(prob, yi, yo),
              "U0": _metrics((refs["U0_1"], refs["U0_2"]), yi, yo),
              "Ucal": _metrics((refs["Ucal_1"], refs["Ucal_2"]), yi, yo)}
    margins = {task: {reference: {
        "accuracy_drop": metric[reference][task]["accuracy"]
                         - metric["old_validation_F"][task]["accuracy"],
        "nll_excess": metric["old_validation_F"][task]["nll"]
                      - metric[reference][task]["nll"],
        "brier_excess": metric["old_validation_F"][task]["brier"]
                        - metric[reference][task]["brier"],
    } for reference in ("U0", "Ucal")} for task in TASKS}
    receipt = {
        "schema": "pcrl-utility-old-validation-f-inner-diagnostic-v1",
        "seed": seed, "comparison_type": "historical_validation_F_same_INNER_role",
        "causal_attribution": "NOT_SUPPORTED",
        "old_fit_rows": {task: old_packages[task]["fit_rows"] for task in TASKS},
        "old_artifact_hashes": {task: old_packages[task]["artifact_hashes"] for task in TASKS},
        "primary_F_gate_sha256": primary_gate_sha256,
        "inner_row_unit_sha256": array_sha(np.column_stack((feature["row_id"], feature["unit"]))),
        "predictions_relative_path": str(pred_path.relative_to(new_root)),
        "predictions_sha256": pred_sha,
        "metrics": metric, "margins": margins,
    }
    _write_once_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--old-private-root", type=Path, required=True)
    parser.add_argument("--old-inputs", type=Path, required=True)
    parser.add_argument("--full-input-gate", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_running_unit(args.unit_id, args.queue_root, args.lock,
                        "r1_old_f_diagnostic", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("old F diagnostic requires one numerical thread")
    root = args.private_root.resolve(strict=True)
    expected = root / "r1" / "old_f_inner" / f"seed_{args.seed}" / "DIAGNOSTIC_RECEIPT.json"
    if args.output.resolve() != expected:
        raise PermissionError("old F diagnostic output escapes exact private path")
    lock = json.loads(args.lock.read_text())
    if (sha_file(args.manifest) != lock.get("source_map_sha256")
            or sha_file(args.old_inputs) != lock.get("old_f_diagnostic_inputs_sha256")
            or sha_file(args.full_input_gate) != lock.get("full_input_gate_sha256")):
        raise PermissionError("old F diagnostic role, custody or gate SHA differs")
    _completed_primary_gate(args.full_input_gate, lock["full_input_gate_sha256"])
    inputs = json.loads(args.old_inputs.read_text())
    if (inputs.get("schema") != "pcrl-utility-old-f-diagnostic-inputs-v1"
            or inputs.get("prior_study_tip") !=
            "9e6bfea62e82297d024fd9588a5f4bfb30e0a9de"):
        raise PermissionError("old F diagnostic source pin differs")
    for relative, expected_sha in inputs["source_sha256_by_repo_path"].items():
        if sha_file(REPO / relative) != expected_sha:
            raise PermissionError("old F deployment source SHA differs")
    store = RoleOnlyStore(args.export_root, args.manifest)
    run(store, args.seed, args.old_private_root.resolve(strict=True), inputs,
        root, expected, lock["full_input_gate_sha256"])


if __name__ == "__main__":
    main()
