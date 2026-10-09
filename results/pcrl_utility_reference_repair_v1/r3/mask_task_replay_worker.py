"""Queued independent replay of selected-input task gates for fixed masks.

This restores A head weights and global temperatures without importing R2
deployment, task scoring, selection, or gate code. Private rows remain private.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r3.independent_task import task_metrics
from results.pcrl_utility_reference_repair_v1.r3.source_reference import (
    apply_inverse_temperature, file_sha,
)
from results.pcrl_utility_reference_repair_v1.r3.task_gate_replay_worker import (
    _require, _role,
)
from results.pcrl_utility_reference_repair_v1.r3.task_replay import (
    _read_bound_npz, _resolve_private, array_sha,
)


THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
TASKS = (("income", 2, "income", 1),
         ("occupation_group", 6, "occupation", 2))
MASK_FIELDS = {"row_id", "unit", "income", "occupation", "MASK_income",
               "MASK_occupation_group", "F_income", "F_occupation_group",
               "U0_1", "U0_2", "Ucal_1", "Ucal_2"}
F_FIELDS = {"row_id", "unit", "income", "occupation", "F_income",
            "F_occupation_group", "U0_1", "U0_2", "Ucal_1", "Ucal_2"}
TOLERANCE = 2e-7


def _json(spec: dict, path: Path) -> dict:
    _require(spec, path)
    return json.loads(path.read_text())


def _network(width: int, classes: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(width, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, classes),
    )


def _a_forward(spec: dict, root: Path, model_entry: dict, package: dict,
               mask: dict, task: str, classes: int,
               selected: np.ndarray) -> np.ndarray:
    path = _resolve_private(root, model_entry["relative_path"])
    if _require(spec, path) != model_entry["sha256"]:
        raise PermissionError("masked head differs from package")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["meta"].item())
        if (meta.get("schema") != "team-c-head-checkpoint-v1"
                or meta.get("purpose") != task
                or meta.get("model_id") != package["model_id"]
                or meta.get("selected_atoms") != mask["atoms"]
                or meta.get("selected_columns") != len(mask["column_indices"])
                or meta.get("receipt") != model_entry["training_receipt"]
                or meta.get("spec", {}).get("hidden_width") != 64
                or len(meta.get("state_keys", [])) != 6):
            raise PermissionError("masked head metadata differs")
        model = _network(len(mask["column_indices"]), classes)
        state = {key: torch.from_numpy(z[f"state_{i}"].copy())
                 for i, key in enumerate(meta["state_keys"])}
        model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(1)
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(selected,
                                                            dtype=np.float32)))
        return torch.softmax(logits.double(), dim=1).numpy()


def _point(candidate: dict, baseline: dict) -> dict:
    exact = {
        "accuracy_loss": (Decimal(str(baseline["accuracy"]))
                          - Decimal(str(candidate["accuracy"]))),
        "nll_excess": (Decimal(str(candidate["nll"]))
                       - Decimal(str(baseline["nll"]))),
        "brier_excess": (Decimal(str(candidate["brier"]))
                         - Decimal(str(baseline["brier"]))),
    }
    cost = {key: float(value) for key, value in exact.items()}
    return {**cost, "point_eligible": bool(
        exact["accuracy_loss"] <= Decimal("0.010")
        and exact["nll_excess"] <= Decimal("0.010")
        and exact["brier_excess"] <= Decimal("0.005"))}


def replay(spec: dict, root: Path, export_root: Path, manifest: dict,
           gate_path: Path, bank_path: Path, lock: dict) -> dict:
    gate = _json(spec, gate_path)
    bank = _json(spec, bank_path)
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("common_role") != "INNER_SELECTION"
            or lock.get("full_input_gate_sha256") != file_sha(gate_path)
            or lock.get("mask_bank_sha256") != file_sha(bank_path)
            or bank.get("schema") != "pcrl-focused-mask-bank-v1"
            or gate.get("selected_recipe") != "A"):
        raise PermissionError("full F gate, selected recipe or bank differs")
    masks = {record["mask_id"]: record for record in bank["masks"]}
    admitted = set(lock["admitted_mask_ids"])
    if (not admitted or not admitted <= set(masks)
            or len(admitted) != len(lock["admitted_mask_ids"])):
        raise PermissionError("masked replay bank coverage differs")
    inner = _role(spec, export_root, manifest, "INNER_SELECTION",
                  ("features", "income", "occupation",
                   "references_s0", "references_s1"))
    common_sha = array_sha(np.column_stack((inner["row_id"], inner["unit"])))
    if common_sha != gate["common_row_unit_pair_sha256"]:
        raise PermissionError("masked replay common-role people differ")
    results = {}
    all_point = True
    max_restore_difference = 0.0
    max_metric_difference = 0.0
    for seed in (0, 1):
        f_dir = root / "r1/gate" / f"seed_{seed}"
        f_receipt = _json(spec, f_dir / "GATE_RECEIPT.json")
        if _require(spec, f_dir / "GATE_RECEIPT.json") != gate[
                "r1_gate_receipt_sha256_by_seed"][str(seed)]:
            raise PermissionError("R1 F gate receipt differs")
        f_path = f_dir / "GATE_PREDICTIONS.npz"
        if _require(spec, f_path) != gate["r1_gate_prediction_sha256_by_seed"][str(seed)]:
            raise PermissionError("R1 F gate vectors differ")
        frozen_f = _read_bound_npz(f_path, f_receipt["predictions_sha256"], F_FIELDS)
        results[str(seed)] = {}
        for mask_id in sorted(admitted):
            mask = masks[mask_id]
            fit_dir = root / "masks" / mask_id / f"seed{seed}"
            pred_dir = root / "mask_gate" / mask_id / f"seed{seed}"
            fit_path = fit_dir / "MASK_FIT_RECEIPT.json"
            mask_gate_path = pred_dir / "MASK_GATE_RECEIPT.json"
            fit = _json(spec, fit_path)
            receipt = _json(spec, mask_gate_path)
            if (fit.get("schema") != "pcrl-utility-mask-fit-receipt-v1"
                    or fit.get("seed") != seed or fit.get("mask_id") != mask_id
                    or fit.get("mask_sha256") != mask["mask_sha256"]
                    or fit.get("selected_recipe") != "A"
                    or fit.get("gate_sha256") != file_sha(gate_path)
                    or receipt.get("schema") !=
                    "pcrl-utility-masked-common-task-gate-v1"
                    or receipt.get("seed") != seed
                    or receipt.get("mask_id") != mask_id
                    or receipt.get("mask_sha256") != mask["mask_sha256"]
                    or receipt.get("full_input_gate_sha256") != file_sha(gate_path)
                    or receipt.get("conditional_lock_sha256") !=
                    lock.get("prior_lead_gate_lock_sha256")
                    or receipt.get("mask_fit_receipt_sha256") != file_sha(fit_path)
                    or receipt.get("R1_gate_receipt_sha256") !=
                    file_sha(f_dir / "GATE_RECEIPT.json")
                    or receipt.get("mask_fit_source_lock_sha256") !=
                    fit.get("conditional_lock_sha256")
                    or receipt.get("common_row_unit_pair_sha256") != common_sha):
                raise PermissionError("masked fit/gate receipt identity differs")
            pred_path = pred_dir / "MASK_GATE_PREDICTIONS.npz"
            if (receipt.get("prediction_relative_path") !=
                    str(pred_path.relative_to(root))
                    or _require(spec, pred_path) != receipt["prediction_sha256"]):
                raise PermissionError("masked gate prediction SHA differs")
            arrays = _read_bound_npz(pred_path, receipt["prediction_sha256"],
                                     MASK_FIELDS)
            for key in ("row_id", "unit", "income", "occupation"):
                if (not np.array_equal(arrays[key], inner[key])
                        or not np.array_equal(arrays[key], frozen_f[key])):
                    raise PermissionError("mask prediction people/labels differ")
            for task, _, _, index in TASKS:
                if not np.array_equal(arrays[f"F_{task}"], frozen_f[f"F_{task}"]):
                    raise PermissionError("mask gate F component differs")
                for reference in ("U0", "Ucal"):
                    key = f"{reference}_{index}"
                    if (not np.array_equal(arrays[key], frozen_f[key])
                            or not np.array_equal(arrays[key],
                                                  inner[f"s{seed}_{key}"])):
                        raise PermissionError("mask gate reference component differs")
            task_results = {}
            for task, classes, label_key, index in TASKS:
                entry = fit["task_packages"][task]
                package_path = _resolve_private(root, entry["relative_path"])
                package = _json(spec, package_path)
                if (_require(spec, package_path) != entry["sha256"]
                        or package.get("schema") !=
                        "pcrl-utility-masked-task-package-v1"
                        or package.get("seed") != seed
                        or package.get("task") != task
                        or package.get("mask_id") != mask_id
                        or package.get("mask_sha256") != mask["mask_sha256"]
                        or package.get("selected_atoms") != mask["atoms"]
                        or package.get("source_indices") != mask["column_indices"]
                        or package.get("source_schema_sha256") !=
                        bank["source_schema_sha256"]
                        or package.get("gate_sha256") != file_sha(gate_path)
                        or package.get("conditional_lock_sha256") !=
                        fit["conditional_lock_sha256"]
                        or package.get("fit_row_unit_sha256") !=
                        fit["fit_row_unit_sha256"]
                        or package.get("calibration_row_unit_sha256") !=
                        fit["calibration_row_unit_sha256"]
                        or package.get("model") !=
                        fit["model_stages"]["final"][task]
                        or package.get("calibrator") != fit["calibrators"][task]
                        or package.get("fill_constants") != fit["fills"]["final"]):
                    raise PermissionError("masked task package differs")
                fill_entry = package["fill_constants"]
                fill_path = _resolve_private(root, fill_entry["relative_path"])
                if _require(spec, fill_path) != fill_entry["sha256"]:
                    raise PermissionError("masked public fill differs")
                with fill_path.open("rb") as stream:
                    fill = np.load(stream, allow_pickle=False)
                if fill.dtype != np.float32 or fill.shape != (83,):
                    raise PermissionError("masked public fill schema differs")
                selected = np.ascontiguousarray(inner["X"][:, mask["column_indices"]])
                raw = _a_forward(spec, root, package["model"], package,
                                 mask, task, classes, selected)
                cal_entry = package["calibrator"]
                cal_path = _resolve_private(root, cal_entry["relative_path"])
                cal = _json(spec, cal_path)
                if (_require(spec, cal_path) != cal_entry["sha256"]
                        or cal.get("schema") != "pcrl-portfolio-global-temp-v1"
                        or cal.get("task") != label_key
                        or cal.get("arm_id") != package["model_id"]
                        or cal.get("model_seed") != seed
                        or cal.get("fit_role") != "CALIBRATION_HELDOUT"):
                    raise PermissionError("masked CAL identity differs")
                restored = apply_inverse_temperature(raw, float(cal["alpha"]))
                saved = arrays[f"MASK_{task}"]
                difference = float(np.max(np.abs(restored - saved)))
                max_restore_difference = max(max_restore_difference, difference)
                if difference > TOLERANCE:
                    raise ValueError("masked probability differs from restored package")
                candidate = task_metrics(saved, inner[label_key])
                references = {
                    name: task_metrics(
                        arrays[f"F_{task}" if name == "F" else f"{name}_{index}"],
                        inner[label_key])
                    for name in ("U0", "Ucal", "F")}
                points = {name: _point(candidate, baseline)
                          for name, baseline in references.items()}
                for arm, own in (("mask", candidate), *references.items()):
                    reported = receipt["metrics"][arm][task]
                    for metric in ("accuracy", "nll", "brier"):
                        delta = abs(own[metric] - reported[metric])
                        max_metric_difference = max(max_metric_difference, delta)
                        if delta > 1e-12:
                            raise ValueError("mask gate metric differs")
                    if (own["class_counts"] != reported["class_counts"]
                            or own["recall"] != reported["per_class_recall"]):
                        raise ValueError("mask gate class metric differs")
                for reference, own in points.items():
                    reported = receipt["point_eligibility"][task]["references"][reference]
                    if (own["point_eligible"] is not reported["point_eligible"]
                            or any(abs(own[key] - reported[key]) > 1e-12 for key in
                                   ("accuracy_loss", "nll_excess", "brier_excess"))):
                        raise ValueError("mask gate point result differs")
                    all_point &= own["point_eligible"]
                task_results[task] = {"candidate": candidate,
                                      "reference": references,
                                      "point": points,
                                      "restore_max_abs_difference": difference}
            mask_point = all(arm["point"][name]["point_eligible"]
                             for arm in task_results.values()
                             for name in ("U0", "Ucal", "F"))
            if mask_point is not receipt["all_task_reference_points_eligible"]:
                raise ValueError("mask aggregate point flag differs")
            results[str(seed)][mask_id] = {
                "point_eligible": mask_point, "task": task_results}
    return {
        "schema": "pcrl-utility-r3-masked-task-replay-v1",
        "status": "REPLAYED",
        "full_input_gate_sha256": file_sha(gate_path),
        "mask_bank_sha256": file_sha(bank_path),
        "masks": sorted(admitted), "paired_seeds": [0, 1],
        "all_mask_seed_task_reference_points_eligible": bool(all_point),
        "max_abs_restored_probability_difference": max_restore_difference,
        "max_abs_metric_difference": max_metric_difference,
        "by_seed_mask": results,
        "scope": "reused Adult development roles; task point gate only; no SEX result",
    }


def _write_once(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--full-input-gate", type=Path, required=True)
    parser.add_argument("--mask-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r3_mask_task_replay", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("independent masked task replay requires one thread")
    root = args.private_root.resolve(strict=True)
    if root != Path(__file__).resolve().parents[1] / ".private":
        raise PermissionError("independent masked task replay custody differs")
    _require(spec, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    lock = json.loads(args.lock.read_text())
    result = replay(spec, root, args.export_root.resolve(strict=True), manifest,
                    args.full_input_gate, args.mask_bank, lock)
    _write_once(args.output, result)


if __name__ == "__main__":
    main()
