"""Sealed, SEX-free common-role utility gate for one fitted mask×seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import array_sha
from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.release_loader import (
    restore_mask_task_release,
)
from results.pcrl_method_portfolio_v1.shared import common_eval


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
TASKS = (("income", "income", 1), ("occupation_group", "occupation", 2))


def _write_once_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
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


def _write_once_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha_file(path)


def _role(store: RoleOnlyStore, seed: int):
    feature = store.read("INNER_SELECTION", "features")
    income = store.read("INNER_SELECTION", "income")
    occupation = store.read("INNER_SELECTION", "occupation")
    references = store.read("INNER_SELECTION", f"references_s{seed}")
    assert_same_people(feature, income, occupation, references)
    return feature, income["income"], occupation["occupation"], references


def fit_source_lock_for_mask(lock: dict, mask_id: str,
                             current_lock_sha256: str) -> str:
    """Resolve the fit-stage source lock without a self-hash in the lock."""
    literal = lock.get("mask_fit_source_lock_sha256")
    mapping = lock.get("mask_fit_source_lock_sha256_by_id")
    if literal is not None and mapping is not None:
        raise PermissionError("mask fit lock source has two conflicting forms")
    if mapping is not None:
        if not isinstance(mapping, dict) or mask_id not in mapping:
            raise PermissionError("mask fit source lock mapping is incomplete")
        value = mapping[mask_id]
    else:
        value = literal
    source = current_lock_sha256 if value == "CURRENT_LOCK" else value
    if (not isinstance(source, str) or len(source) != 64
            or any(char not in "0123456789abcdef" for char in source)):
        raise PermissionError("mask fit source lock SHA is malformed")
    return source


def _f_predictions(path: Path, expected_sha256: str, seed: int,
                   common_row_unit_sha256: str) -> dict[str, np.ndarray]:
    if sha_file(path) != expected_sha256:
        raise PermissionError("R1 gate predictions bytes differ")
    fields = {"row_id", "unit", "income", "occupation",
              "F_income", "F_occupation_group", "U0_1", "U0_2",
              "Ucal_1", "Ucal_2"}
    with ZipFile(path) as archive:
        if {item.filename for item in archive.infolist()} != {
                name + ".npy" for name in fields}:
            raise PermissionError("R1 gate prediction members differ")
    with np.load(path, allow_pickle=False) as z:
        records = {key: np.array(z[key], copy=True) for key in fields}
    if array_sha(np.column_stack((records["row_id"], records["unit"]))) != common_row_unit_sha256:
        raise PermissionError("R1 gate prediction people differ")
    for task, label, index in TASKS:
        p = records[f"F_{task}"]
        if (p.dtype != np.float64 or p.shape != (len(records["row_id"]),
                                                 2 if index == 1 else 6)):
            raise PermissionError("R1 gate F probability shape differs")
    return records


def gate_mask_unit(store: RoleOnlyStore, mask, seed: int, recipe: str,
                   private_root: Path, output: Path, *, fit_receipt_sha256: str,
                   f_gate_receipt_sha256: str, gate_sha256: str,
                   lock_sha256: str, fit_source_lock_sha256: str,
                   common_row_unit_sha256: str) -> dict:
    fit_path = private_root / "masks" / mask.mask_id / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
    fit_record = {"relative_path": str(fit_path.relative_to(private_root)),
                  "sha256": fit_receipt_sha256}
    releases = {
        task: restore_mask_task_release(
            private_root, fit_record, mask, task, seed, recipe,
            gate_sha256=gate_sha256,
            conditional_lock_sha256=fit_source_lock_sha256)
        for task, _, _ in TASKS
    }
    f_receipt_path = private_root / "r1" / "gate" / f"seed_{seed}" / "GATE_RECEIPT.json"
    if sha_file(f_receipt_path) != f_gate_receipt_sha256:
        raise PermissionError("R1 common-role gate receipt bytes differ")
    f_receipt = json.loads(f_receipt_path.read_text())
    if (f_receipt.get("schema") != "pcrl-utility-full-input-gate-result-v1"
            or f_receipt.get("seed") != seed
            or f_receipt.get("selected_recipe") != recipe
            or f_receipt.get("inner_row_unit_sha256") != common_row_unit_sha256):
        raise PermissionError("R1 common-role gate identity differs")
    f_prediction_path = private_root / f_receipt["predictions_relative_path"]
    if f_prediction_path != f_receipt_path.parent / "GATE_PREDICTIONS.npz":
        raise PermissionError("R1 gate prediction path differs")
    frozen = _f_predictions(
        f_prediction_path, f_receipt["predictions_sha256"], seed,
        common_row_unit_sha256)
    feature, yi, yo, references = _role(store, seed)
    for key, actual in (("row_id", feature["row_id"]),
                        ("unit", feature["unit"]), ("income", yi),
                        ("occupation", yo),
                        ("U0_1", references["U0_1"]),
                        ("U0_2", references["U0_2"]),
                        ("Ucal_1", references["Ucal_1"]),
                        ("Ucal_2", references["Ucal_2"])):
        if not np.array_equal(frozen[key], actual):
            raise PermissionError("R1 gate/reference role alignment differs: " + key)
    names = tuple(str(value) for value in feature["feature_names"])
    views = {task: releases[task].view(feature["X"], names)
             for task, _, _ in TASKS}
    arrays = {"row_id": feature["row_id"], "unit": feature["unit"],
              "income": yi, "occupation": yo}
    metrics = {"mask": {}, "F": {}, "U0": {}, "Ucal": {}}
    for task, label, index in TASKS:
        y = yi if task == "income" else yo
        arrays[f"MASK_{task}"] = views[task]["probabilities"]
        arrays[f"F_{task}"] = frozen[f"F_{task}"]
        for reference in ("U0", "Ucal"):
            arrays[f"{reference}_{index}"] = references[f"{reference}_{index}"]
        metrics["mask"][task] = common_eval.task_metrics(
            arrays[f"MASK_{task}"], y, task=label)
        metrics["F"][task] = common_eval.task_metrics(
            arrays[f"F_{task}"], y, task=label)
        for reference in ("U0", "Ucal"):
            metrics[reference][task] = common_eval.task_metrics(
                arrays[f"{reference}_{index}"], y, task=label)
    point = {}
    for task, _, _ in TASKS:
        point[task] = {"references": {}}
        for reference in ("U0", "Ucal", "F"):
            mask_metrics = metrics["mask"][task]
            baseline = metrics[reference][task]
            delta = {
                "accuracy_loss": baseline["accuracy"] - mask_metrics["accuracy"],
                "nll_excess": mask_metrics["nll"] - baseline["nll"],
                "brier_excess": mask_metrics["brier"] - baseline["brier"],
            }
            delta["point_eligible"] = bool(
                delta["accuracy_loss"] <= 0.010
                and delta["nll_excess"] <= 0.010
                and delta["brier_excess"] <= 0.005)
            point[task]["references"][reference] = delta
    pred_path = output.parent / "MASK_GATE_PREDICTIONS.npz"
    prediction_sha = _write_once_npz(pred_path, arrays)
    receipt = {
        "schema": "pcrl-utility-masked-common-task-gate-v1",
        "seed": seed, "mask_id": mask.mask_id,
        "mask_sha256": mask.mask_sha256,
        "selected_recipe": recipe,
        "full_input_gate_sha256": gate_sha256,
        "conditional_lock_sha256": lock_sha256,
        "mask_fit_source_lock_sha256": fit_source_lock_sha256,
        "mask_fit_receipt_sha256": fit_receipt_sha256,
        "R1_gate_receipt_sha256": f_gate_receipt_sha256,
        "common_role": "INNER_SELECTION",
        "common_row_unit_pair_sha256": common_row_unit_sha256,
        "prediction_relative_path": str(pred_path.relative_to(private_root)),
        "prediction_sha256": prediction_sha,
        "metrics": metrics,
        "point_eligibility": point,
        "all_task_reference_points_eligible": all(
            point[task]["references"][reference]["point_eligible"]
            for task, _, _ in TASKS for reference in ("U0", "Ucal", "F")),
        "development_only": True,
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
                               "r2_mask_gate", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 gate requires one numerical thread per library")
    private_root = args.private_root.resolve(strict=True)
    if private_root != STUDY / ".private":
        raise PermissionError("R2 gate custody is outside this study")
    lock_sha = sha_file(args.lock)
    lock = json.loads(args.lock.read_text())
    if (lock.get("source_map_sha256") != sha_file(args.manifest)
            or lock.get("mask_bank_sha256") != sha_file(MASK_BANK)):
        raise PermissionError("R2 gate manifest or bank differs")
    expected = private_root / "mask_gate" / args.mask_id / f"seed{args.seed}" / "MASK_GATE_RECEIPT.json"
    if args.output.resolve() != expected:
        raise PermissionError("R2 gate output path differs")
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock["common_row_unit_pair_sha256"],
        private_root=private_root)
    if lock.get("selected_recipe") != admitted.selected_recipe:
        raise PermissionError("R2 gate recipe differs from full-input gate")
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    if args.mask_id not in lock.get("admitted_mask_ids", []):
        raise PermissionError("R2 gate mask is outside sealed branch")
    fit_source_lock_sha = fit_source_lock_for_mask(
        lock, args.mask_id, lock_sha)
    mask = bank.mask(args.mask_id)
    pred = expected.parent / "MASK_GATE_PREDICTIONS.npz"
    if set(spec["outputs"]) != {str(expected), str(pred)}:
        raise PermissionError("R2 gate output inventory differs")
    fit_path = private_root / "masks" / args.mask_id / f"seed{args.seed}" / "MASK_FIT_RECEIPT.json"
    f_gate_path = private_root / "r1" / "gate" / f"seed_{args.seed}" / "GATE_RECEIPT.json"
    fit_sha = spec.get("inputs", {}).get(str(fit_path))
    f_gate_sha = spec.get("inputs", {}).get(str(f_gate_path))
    if (fit_sha != sha_file(fit_path) or f_gate_sha != sha_file(f_gate_path)):
        raise PermissionError("R2 gate dependency receipts differ from queue")
    store = RoleOnlyStore(args.export_root, args.manifest)
    gate_mask_unit(
        store, mask, args.seed, admitted.selected_recipe,
        private_root, expected, fit_receipt_sha256=fit_sha,
        f_gate_receipt_sha256=f_gate_sha,
        gate_sha256=admitted.gate_sha256, lock_sha256=lock_sha,
        fit_source_lock_sha256=fit_source_lock_sha,
        common_row_unit_sha256=admitted.common_row_unit_pair_sha256)


if __name__ == "__main__":
    main()
