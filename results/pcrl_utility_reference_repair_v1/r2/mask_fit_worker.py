"""One sealed mask×seed task fit, with private write-once deployment custody.

This entry point verifies the queue, conditional lock, full-input gate and all
four F package components before it opens any role-only task array. It never
has an API for SEX or old assessment data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import array_sha
from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_utility_reference_repair_v1.r1.source_task import (
    predict_source, save_source,
)
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.mask_inputs import (
    build_mask_input_plan, mask_calibration_matrix,
)
from results.pcrl_utility_reference_repair_v1.r2.mask_train import fit_mask_task_only


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
for source_dir in (
    REPO / "results/pcrl_method_portfolio_v1/teams/c",
    REPO / "results/pcrl_method_portfolio_v1/shared",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
from training import save_head  # noqa: E402
from results.pcrl_method_portfolio_v1.shared import common_eval  # noqa: E402


TASKS = (("income", "income", 0), ("occupation_group", "occupation", 1))
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"


def _write_once_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _write_once_json(path: Path, record: dict) -> str:
    return _write_once_bytes(path, (json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())


def _write_once_npy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha_file(path)


def _relative(path: Path, private_root: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(private_root):
        raise PermissionError("masked package output escaped private root")
    return str(resolved.relative_to(private_root))


def expected_output_paths(recipe: str, private_root: Path, mask_id: str,
                          seed: int) -> tuple[Path, ...]:
    """Exact queue output inventory for the prospectively selected recipe."""
    if recipe not in ("A", "B") or seed not in (0, 1, 2):
        raise ValueError("mask fit recipe or seed is not prospectively admitted")
    root = private_root / "masks" / mask_id / f"seed{seed}"
    names = ["fill_inner.npy", "fill_outer.npy", "fill_final.npy"]
    if recipe == "A":
        names += [f"{stage}_{task}_head.npz"
                  for stage in ("outer", "final")
                  for task, _, _ in TASKS]
    else:
        names += ["inner_encoder.pt"]
        names += [f"{stage}_source_bundle/{name}"
                  for stage in ("outer", "final")
                  for name in ("encoder.pt", "head_0.joblib", "head_1.joblib")]
    names += [f"{task}_calibrator.json" for task, _, _ in TASKS]
    names += [f"FINAL_{task}_PACKAGE.json" for task, _, _ in TASKS]
    names += ["OUTER_TASK_METRICS.json", "MASK_FIT_RECEIPT.json"]
    return tuple(root / name for name in names)


def _role(store: RoleOnlyStore, role: str):
    feature = store.read(role, "features")
    income = store.read(role, "income")
    occupation = store.read(role, "occupation")
    assert_same_people(feature, income, occupation)
    return feature, income["income"], occupation["occupation"]


def fit_mask_unit(store: RoleOnlyStore, mask, seed: int, recipe: str,
                  private_root: Path, output: Path, *, gate_sha256: str,
                  lock_sha256: str, source_lock_sha256: str) -> dict:
    """Run only after the CLI has established the immutable conditional gate."""
    folder = output.parent
    fit_feature, yi, yo = _role(store, "OSF_DEFENSE_FIT")
    names = tuple(str(name) for name in fit_feature["feature_names"])
    plan = build_mask_input_plan(
        fit_feature["X"], yi, yo, fit_feature["unit"], mask, seed=seed)
    core = fit_mask_task_only(
        recipe, plan, yi, yo, fit_feature["unit"], seed=seed,
        feature_names=names)
    fill_records = {}
    for stage in ("inner", "outer", "final"):
        path = folder / f"fill_{stage}.npy"
        digest = _write_once_npy(path, plan.fill_constants[stage])
        fill_records[stage] = {
            "relative_path": _relative(path, private_root),
            "sha256": digest,
        }
    model_records = {}
    if recipe == "A":
        for stage, models in (("outer", core.outer_models),
                              ("final", core.final_models)):
            model_records[stage] = {}
            for task, _, index in TASKS:
                path = folder / f"{stage}_{task}_head.npz"
                digest = save_head(models[index], path, private_root=private_root)
                model_records[stage][task] = {
                    "relative_path": _relative(path, private_root),
                    "sha256": digest, "training_receipt": models[index].receipt,
                    "model_id": models[index].model_id,
                }
    else:
        inner_path = folder / "inner_encoder.pt"
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{inner_path.name}.", dir=inner_path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                torch.save(core.inner_encoder.state_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, inner_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        model_records["inner_encoder"] = {
            "relative_path": _relative(inner_path, private_root),
            "sha256": sha_file(inner_path),
        }
        for stage, (model, heads) in (("outer", core.outer_models),
                                      ("final", core.final_models)):
            bundle = folder / f"{stage}_source_bundle"
            hashes = save_source(model, heads, bundle)
            model_records[stage] = {
                "relative_directory": str(bundle.relative_to(private_root)),
                "sha256": hashes, "C_by_task": core.selected_C,
            }
    outer_record = {
        "schema": "pcrl-utility-masked-outer-task-metrics-v1",
        "mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256,
        "seed": seed, "selected_recipe": recipe,
        "role": "OSF_DEFENSE_FIT:outer_validation",
        "probability_convention": "raw_uncalibrated",
        "metrics": core.outer_raw_metrics,
        "outer_split": core.outer_split_receipt,
        "inner_split": core.inner_split_receipt,
        "selected_C": core.selected_C,
        "C_search": core.C_search,
        "fit_row_unit_sha256": array_sha(np.column_stack((
            fit_feature["row_id"], fit_feature["unit"]))),
    }
    outer_path = folder / "OUTER_TASK_METRICS.json"
    outer_sha = _write_once_json(outer_path, outer_record)

    # The held-out calibration role is first opened after all final model
    # states and their FIT-only outer metrics have been written.
    cal_feature, cal_yi, cal_yo = _role(store, "CALIBRATION_HELDOUT")
    if tuple(str(name) for name in cal_feature["feature_names"]) != names:
        raise PermissionError("FIT/CAL predictor order changed")
    cal_x = mask_calibration_matrix(cal_feature["X"], plan)
    if recipe == "A":
        selected = cal_x[:, plan.source_indices]
        cal_raw = tuple(head.predict_proba(selected) for head in core.final_models)
    else:
        model, heads = core.final_models
        cal_raw = predict_source(model, heads, cal_x)
    calibration_records = {}
    task_packages = {}
    for task, label, index in TASKS:
        arm_id = f"UTILITY-REPAIR:{recipe}:s{seed}:{task}:final:{mask.mask_id}"
        calibration = common_eval.fit_global_temperature(
            cal_raw[index], (cal_yi, cal_yo)[index], task=label,
            arm_id=arm_id, model_seed=seed, row_id=cal_feature["row_id"],
            unit=cal_feature["unit"])
        cal_path = folder / f"{task}_calibrator.json"
        cal_sha = common_eval.save_calibrator(calibration, cal_path)
        calibration_records[task] = {
            "relative_path": _relative(cal_path, private_root),
            "sha256": cal_sha,
        }
        task_path = folder / f"FINAL_{task}_PACKAGE.json"
        package = {
            "schema": "pcrl-utility-masked-task-package-v1",
            "seed": seed, "task": task, "selected_recipe": recipe,
            "mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256,
            "selected_atoms": list(mask.atoms),
            "source_indices": list(mask.column_indices),
            "source_schema_sha256": store.manifest["source_schema_sha256"],
            "model": (model_records["final"][task] if recipe == "A" else
                      model_records["final"]),
            "model_id": arm_id,
            "fill_constants": fill_records["final"],
            "calibrator": calibration_records[task],
            "gate_sha256": gate_sha256,
            "conditional_lock_sha256": lock_sha256,
            "source_science_lock_sha256": source_lock_sha256,
            "fit_row_unit_sha256": outer_record["fit_row_unit_sha256"],
            "calibration_row_unit_sha256": array_sha(np.column_stack((
                cal_feature["row_id"], cal_feature["unit"]))),
        }
        task_packages[task] = {
            "relative_path": str(task_path.relative_to(private_root)),
            "sha256": _write_once_json(task_path, package),
        }
    receipt = {
        "schema": "pcrl-utility-mask-fit-receipt-v1",
        "mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256,
        "seed": seed, "selected_recipe": recipe,
        "selected_atoms": list(mask.atoms),
        "source_indices": list(mask.column_indices),
        "gate_sha256": gate_sha256,
        "conditional_lock_sha256": lock_sha256,
        "source_science_lock_sha256": source_lock_sha256,
        "fit_row_unit_sha256": outer_record["fit_row_unit_sha256"],
        "calibration_row_unit_sha256": array_sha(np.column_stack((
            cal_feature["row_id"], cal_feature["unit"]))),
        "fills": fill_records,
        "model_stages": model_records,
        "outer_task_metrics": {"relative_path": _relative(outer_path, private_root),
                               "sha256": outer_sha},
        "calibrators": calibration_records,
        "task_packages": task_packages,
    }
    _write_once_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-id", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r2_mask_fit", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 mask worker requires one numerical thread per library")
    private_root = args.private_root.resolve(strict=True)
    if private_root != STUDY / ".private":
        raise PermissionError("R2 masked model custody is outside this study")
    lock_sha = sha_file(args.lock)
    lock = json.loads(args.lock.read_text())
    if (lock.get("source_map_sha256") != sha_file(args.manifest)
            or lock.get("mask_bank_sha256") != sha_file(MASK_BANK)):
        raise PermissionError("conditional role manifest or mask bank differs")
    expected = private_root / "masks" / args.mask_id / f"seed{args.seed}" / "MASK_FIT_RECEIPT.json"
    if args.output.resolve() != expected:
        raise PermissionError("R2 mask receipt path differs")
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock["common_row_unit_pair_sha256"],
        private_root=private_root)
    if lock.get("selected_recipe") != admitted.selected_recipe:
        raise PermissionError("conditional recipe differs from sealed F gate")
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    allowed_masks = lock.get("admitted_mask_ids")
    if (not isinstance(allowed_masks, list)
            or len(allowed_masks) != len(set(allowed_masks))
            or args.mask_id not in allowed_masks
            or any(mask_id not in {item.mask_id for item in bank.masks}
                   for mask_id in allowed_masks)):
        raise PermissionError("mask is outside the sealed conditional branch")
    mask = bank.mask(args.mask_id)
    exact_outputs = {str(path) for path in expected_output_paths(
        admitted.selected_recipe, private_root, args.mask_id, args.seed)}
    if set(spec["outputs"]) != exact_outputs:
        raise PermissionError("R2 queue output inventory differs from selected recipe")
    store = RoleOnlyStore(args.export_root, args.manifest)
    fit_mask_unit(store, mask, args.seed, admitted.selected_recipe,
                  private_root, expected, gate_sha256=admitted.gate_sha256,
                  lock_sha256=lock_sha,
                  source_lock_sha256=admitted.source_science_lock_sha256)


if __name__ == "__main__":
    main()
