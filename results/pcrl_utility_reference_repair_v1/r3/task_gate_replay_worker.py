"""Queued independent numerical replay of A/B selection and the F task gate.

This worker opens only admitted FIT, CAL and INNER role-only files and the
private output packages named in its sealed unit spec. It imports no runner
scoring, candidate selection or role-loading implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r3.source_reference import (
    file_sha, fit_inverse_temperature,
)
from results.pcrl_utility_reference_repair_v1.r3.task_replay import (
    TASKS, _f_raw, _read_bound_npz, _resolve_private, array_sha,
    gate_metrics, restore_f_predictions, select_recipe,
)


THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
TOLERANCE = 2e-7


def _require(spec: dict, path: Path) -> str:
    resolved = path.resolve(strict=True)
    expected = spec["inputs"].get(str(resolved))
    if not isinstance(expected, str) or file_sha(resolved) != expected:
        raise PermissionError("replay input absent or differs from sealed unit")
    return expected


def _private_json(spec: dict, path: Path) -> dict:
    _require(spec, path)
    return json.loads(path.read_text())


def _private_npz(spec: dict, path: Path, fields: set[str]) -> dict[str, np.ndarray]:
    return _read_bound_npz(path, _require(spec, path), fields)


def _role(spec: dict, export_root: Path, manifest: dict,
          role: str, kinds: tuple[str, ...]) -> dict:
    result = {}
    identity = None
    for kind in kinds:
        key = f"{role}:{kind}"
        path = export_root / role / f"{kind}.npz"
        if _require(spec, path) != manifest["role_file_sha256"][key]:
            raise PermissionError("role input differs from admitted manifest")
        with np.load(path, allow_pickle=False) as archive:
            expected = ({"row_id", "unit", "X", "feature_names"} if kind == "features"
                        else {"row_id", "unit", kind} if kind in ("income", "occupation")
                        else {"row_id", "unit", "U0_1", "U0_2", "Ucal_1",
                              "Ucal_2", "d_1", "d_2"})
            if set(archive.files) != expected:
                raise PermissionError("role-only member set differs")
            arrays = {name: archive[name] for name in expected}
        current = (arrays.pop("row_id"), arrays.pop("unit"))
        if identity is None:
            identity = current
        elif (not np.array_equal(identity[0], current[0])
              or not np.array_equal(identity[1], current[1])):
            raise PermissionError("role-only files have different people/order")
        if kind.startswith("references_s"):
            result.update({f"{kind.split('_')[-1]}_{name}": value
                           for name, value in arrays.items()})
        else:
            result.update(arrays)
    row, unit = identity
    if (row.dtype != np.int64 or unit.dtype != np.int64
            or row.shape != (manifest["roles"][role]["rows"],)
            or unit.shape != row.shape or len(np.unique(row)) != len(row)
            or len(np.unique(unit)) != manifest["roles"][role]["groups"]):
        raise PermissionError("role-only identity differs")
    result["row_id"], result["unit"] = row, unit
    if "X" in result:
        x = result["X"]
        names = result.pop("feature_names")
        import hashlib
        schema = hashlib.sha256("\n".join(map(str, names)).encode()).hexdigest()
        if (x.dtype != np.float32 or x.shape != (len(row), 83)
                or not np.isfinite(x).all()
                or schema != manifest["source_schema_sha256"]):
            raise PermissionError("role-only feature schema differs")
    for kind, width in (("income", 2), ("occupation", 6)):
        if kind in result:
            y = result[kind]
            if (y.dtype != np.int64 or y.shape != row.shape
                    or np.any(y < 0) or np.any(y >= width)):
                raise PermissionError("role-only task label schema differs")
    return result


def _own_split(unit: np.ndarray, yi: np.ndarray, yo: np.ndarray,
               seed: int, fraction: float, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent reconstruction of the predeclared group-random split."""
    groups = np.unique(unit)
    selected_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * fraction)))
    selection = groups[np.random.default_rng(seed + offset).permutation(len(groups))
                       [:selected_count]]
    in_fit = ~np.isin(unit, selection)
    for y, width in ((yi, 2), (yo, 6)):
        for cls in range(width):
            if np.any(y[in_fit] == cls):
                continue
            eligible = np.unique(unit[(~in_fit) & (y == cls)])
            if len(eligible) == 0:
                raise ValueError("split cannot restore task class support")
            moved = min((int(value) for value in eligible),
                        key=lambda value: (int(np.count_nonzero(unit == value)), value))
            in_fit[unit == moved] = True
    fit, validation = np.flatnonzero(in_fit), np.flatnonzero(~in_fit)
    if len(validation) == 0 or len(np.intersect1d(unit[fit], unit[validation])):
        raise ValueError("split lacks validation or group separation")
    return fit, validation


def _check_model_inputs(spec: dict, private_root: Path, receipt: dict) -> None:
    entries = receipt["packages"]
    if receipt["selected_recipe"] == "A":
        for task, _, _ in TASKS:
            entry = entries[task]
            if _require(spec, _resolve_private(private_root, entry["relative_path"])) != entry["sha256"]:
                raise PermissionError("A package weight hash differs")
    else:
        entry = entries["B_source_bundle"]
        directory = _resolve_private(private_root, entry["relative_directory"])
        for name, expected in entry["sha256"].items():
            if _require(spec, directory / name) != expected:
                raise PermissionError("B package weight hash differs")
    for task, _, _ in TASKS:
        key = f"calibrator_{task}"
        if key in entries:
            entry = entries[key]
            if _require(spec, _resolve_private(private_root, entry["relative_path"])) != entry["sha256"]:
                raise PermissionError("F calibrator hash differs")


def _close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return bool(abs(float(a) - float(b)) <= tolerance)


def replay(spec: dict, private_root: Path, export_root: Path,
           manifest: dict, selection_path: Path, f_lock_path: Path) -> dict:
    fit = _role(spec, export_root, manifest, "OSF_DEFENSE_FIT",
                ("features", "income", "occupation"))
    cal = _role(spec, export_root, manifest, "CALIBRATION_HELDOUT",
                ("features", "income", "occupation"))
    inner = _role(spec, export_root, manifest, "INNER_SELECTION",
                  ("features", "income", "occupation",
                   "references_s0", "references_s1"))
    selection = _private_json(spec, selection_path)
    f_lock = _private_json(spec, f_lock_path)
    if (selection.get("schema") != "pcrl-utility-recipe-selection-v1"
            or selection.get("status") != "FROZEN"
            or selection.get("rule") != "lowest_sum_outer_fit_validation_nll_A_tie"
            or f_lock.get("schema") != "pcrl-utility-f-artifact-lock-v1"
            or f_lock.get("status") != "FROZEN"
            or f_lock.get("recipe_selection_sha256") != file_sha(selection_path)):
        raise PermissionError("recipe or F artifact lock differs")
    screen_arrays = {}
    screen_receipts = {}
    screen_comparison = {}
    for seed in (0, 1):
        screen_dir = private_root / "r1" / "screen" / f"seed_{seed}"
        receipt_path = screen_dir / "SCREEN_RECEIPT.json"
        receipt = _private_json(spec, receipt_path)
        screen_receipts[seed] = receipt
        if (file_sha(receipt_path) !=
                selection["screen_output_sha256_by_seed"][str(seed)]):
            raise PermissionError("screen receipt differs from frozen selection")
        arrays = _private_npz(spec, screen_dir / "SCREEN_PREDICTIONS.npz",
                              {"row_id", "unit", "income", "occupation",
                               "A_income", "A_occupation_group",
                               "B_income", "B_occupation_group"})
        if file_sha(screen_dir / "SCREEN_PREDICTIONS.npz") != receipt["predictions_sha256"]:
            raise PermissionError("screen arrays differ from receipt")
        outer_fit, outer_val = _own_split(fit["unit"], fit["income"],
                                          fit["occupation"], seed, 0.25, 26000)
        if (receipt["outer_split"]["fit_row_index_sha256"] !=
                array_sha(outer_fit.astype(np.int64))
                or receipt["outer_split"]["validation_row_index_sha256"] !=
                array_sha(outer_val.astype(np.int64))):
            raise PermissionError("screen outer split differs from independent replay")
        nested_fit, nested_val = _own_split(
            fit["unit"][outer_fit], fit["income"][outer_fit],
            fit["occupation"][outer_fit], seed, 0.15, 17001)
        if (receipt["B_inner_split"]["fit_row_index_sha256"] !=
                array_sha(nested_fit.astype(np.int64))
                or receipt["B_inner_split"]["validation_row_index_sha256"] !=
                array_sha(nested_val.astype(np.int64))):
            raise PermissionError("B C-selection split differs from independent replay")
        if (not np.array_equal(arrays["row_id"], fit["row_id"][outer_val])
                or not np.array_equal(arrays["unit"], fit["unit"][outer_val])
                or not np.array_equal(arrays["income"], fit["income"][outer_val])
                or not np.array_equal(arrays["occupation"], fit["occupation"][outer_val])):
            raise PermissionError("screen people/labels differ from independent FIT split")
        for recipe in ("A", "B"):
            package = receipt["packages"][recipe]
            temporary_receipt = {
                "selected_recipe": recipe, "seed": seed,
                "packages": package if recipe == "A" else {"B_source_bundle": package},
            }
            _check_model_inputs(spec, private_root, temporary_receipt)
            restored = _f_raw(temporary_receipt, private_root, fit["X"][outer_val])
            deltas = [float(np.max(np.abs(restored[i] - arrays[f"{recipe}_{task}"])))
                      for i, (task, _, _) in enumerate(TASKS)]
            if any(value > TOLERANCE for value in deltas):
                raise ValueError("screen probabilities differ from saved package")
            screen_comparison[f"{seed}:{recipe}"] = {
                "max_abs_difference": max(deltas),
                "bitwise_equal": all(np.array_equal(restored[i], arrays[f"{recipe}_{task}"])
                                     for i, (task, _, _) in enumerate(TASKS)),
            }
        screen_arrays[seed] = arrays
    selected = select_recipe(screen_arrays)
    if selected["selected_recipe"] != selection["selected_recipe"]:
        raise ValueError("selected recipe differs from independent FIT-only ranking")
    usefulness = {}
    for seed in (0, 1):
        receipt = screen_receipts[seed]
        for recipe in ("A", "B"):
            for task, _, _ in TASKS:
                if not _close(selected["per_seed"][seed][recipe][task]["nll"],
                              receipt["metrics"][recipe][task]["nll"]):
                    raise ValueError("screen runner task NLL differs")
        if selection["selected_recipe"] == "B":
            chosen = {task: receipt["packages"]["B"]["C_by_task"][task]["selected_C"]
                      for task, _, _ in TASKS}
            if chosen != selection["selected_C_by_seed_task"][str(seed)]:
                raise ValueError("frozen B C differs from screened FIT C")
        outer_fit, outer_val = _own_split(fit["unit"], fit["income"],
                                          fit["occupation"], seed, 0.25, 26000)
        usefulness[str(seed)] = {}
        for task, _, label_key in TASKS:
            y = fit[label_key]
            fit_prior = np.bincount(y[outer_fit], minlength=2 if task == "income" else 6)
            constant_class = int(np.argmax(fit_prior))
            constant_accuracy = float(np.mean(y[outer_val] == constant_class))
            chosen_accuracy = selected["per_seed"][seed][selection["selected_recipe"]][task]["accuracy"]
            usefulness[str(seed)][task] = {
                "fit_constant_accuracy": constant_accuracy,
                "selected_accuracy": chosen_accuracy,
                "gain": chosen_accuracy - constant_accuracy,
                "useful_at_0_010": bool(chosen_accuracy - constant_accuracy >= 0.010),
            }
    gate_out = {}
    all_eligible = True
    for seed in (0, 1):
        full_dir = private_root / "r1" / "full" / f"seed_{seed}"
        receipt_path = full_dir / "F_PACKAGE_RECEIPT.json"
        receipt = _private_json(spec, receipt_path)
        if (file_sha(receipt_path) != f_lock["full_refit_output_sha256_by_seed"][str(seed)]
                or receipt["selected_recipe"] != selection["selected_recipe"]
                or receipt["selection_lock_sha256"] != file_sha(selection_path)
                or receipt["fit_rows"] != len(fit["row_id"])
                or receipt["calibration_rows"] != len(cal["row_id"])
                or receipt["fit_row_unit_sha256"] !=
                array_sha(np.column_stack((fit["row_id"], fit["unit"])))
                or receipt["calibration_row_unit_sha256"] !=
                array_sha(np.column_stack((cal["row_id"], cal["unit"])))):
            raise PermissionError("F full-refit package/role identity differs")
        _check_model_inputs(spec, private_root, receipt)
        for task, _, _ in TASKS:
            entry = receipt["task_packages"][task]
            package_path = _resolve_private(private_root, entry["relative_path"])
            if (_require(spec, package_path) != entry["sha256"]
                    or entry["sha256"] !=
                    f_lock["f_package_sha256_by_seed_task"][str(seed)][task]):
                raise PermissionError("F task package differs from artifact lock")
            package = json.loads(package_path.read_text())
            if (package["seed"] != seed or package["task"] != task
                    or package["selected_recipe"] != selection["selected_recipe"]
                    or package["selection_lock_sha256"] != file_sha(selection_path)
                    or package["fit_row_unit_sha256"] != receipt["fit_row_unit_sha256"]
                    or package["calibration_row_unit_sha256"] !=
                    receipt["calibration_row_unit_sha256"]
                    or package["model"] != (receipt["packages"][task]
                                            if selection["selected_recipe"] == "A"
                                            else receipt["packages"]["B_source_bundle"])
                    or package["calibrator"] != receipt["packages"][f"calibrator_{task}"]):
                raise PermissionError("F task package content differs")
        raw_cal = _f_raw(receipt, private_root, cal["X"])
        alpha_differences = {}
        for i, (task, _, label_key) in enumerate(TASKS):
            entry = receipt["packages"][f"calibrator_{task}"]
            cal_rec = _private_json(spec, _resolve_private(private_root, entry["relative_path"]))
            own_alpha = fit_inverse_temperature(raw_cal[i], cal[label_key])
            alpha_differences[task] = abs(own_alpha - float(cal_rec["alpha"]))
            if alpha_differences[task] > 1e-8:
                raise ValueError("independent CAL fit differs from saved calibrator")
        gate_dir = private_root / "r1" / "gate" / f"seed_{seed}"
        gate_receipt_path = gate_dir / "GATE_RECEIPT.json"
        gate_receipt = _private_json(spec, gate_receipt_path)
        if (gate_receipt["F_artifact_lock_sha256"] != file_sha(f_lock_path)
                or gate_receipt["F_package_receipt_sha256"] != file_sha(receipt_path)
                or gate_receipt["inner_row_unit_sha256"] !=
                array_sha(np.column_stack((inner["row_id"], inner["unit"])))):
            raise PermissionError("gate result differs from frozen F artifact")
        arrays = _private_npz(spec, gate_dir / "GATE_PREDICTIONS.npz",
                              {"row_id", "unit", "income", "occupation",
                               "F_income", "F_occupation_group", "U0_1",
                               "U0_2", "Ucal_1", "Ucal_2"})
        if (file_sha(gate_dir / "GATE_PREDICTIONS.npz") != gate_receipt["predictions_sha256"]
                or not np.array_equal(arrays["row_id"], inner["row_id"])
                or not np.array_equal(arrays["unit"], inner["unit"])
                or not np.array_equal(arrays["income"], inner["income"])
                or not np.array_equal(arrays["occupation"], inner["occupation"])):
            raise PermissionError("gate arrays differ from admitted INNER people/labels")
        for i, (task, _, _) in enumerate(TASKS, 1):
            for kind in ("U0", "Ucal"):
                if not np.array_equal(arrays[f"{kind}_{i}"],
                                      inner[f"s{seed}_{kind}_{i}"]):
                    raise PermissionError("gate reference differs from admitted role")
        restored = restore_f_predictions(receipt, private_root, inner["X"])
        f_deltas = {task: float(np.max(np.abs(restored[i] - arrays[f"F_{task}"])))
                    for i, (task, _, _) in enumerate(TASKS)}
        if any(value > TOLERANCE for value in f_deltas.values()):
            raise ValueError("F gate vector differs from restored package")
        scored = gate_metrics(arrays)
        if scored["point_feasible"] != gate_receipt["point_feasible"]:
            raise ValueError("F point gate differs from runner receipt")
        for arm in ("F", "U0", "Ucal"):
            for task, _, _ in TASKS:
                for metric in ("accuracy", "nll", "brier"):
                    if not _close(scored["metrics"][arm][task][metric],
                                  gate_receipt["metrics"][arm][task][metric]):
                        raise ValueError("task metric differs from runner receipt")
        all_eligible &= scored["point_feasible"]
        gate_out[str(seed)] = {
            "point_feasible": scored["point_feasible"],
            "metrics": scored["metrics"], "margins": scored["margins"],
            "F_restore_max_abs_difference": f_deltas,
            "independent_cal_alpha_abs_difference": alpha_differences,
        }
    useful_all = all(cell["useful_at_0_010"]
                     for tasks in usefulness.values() for cell in tasks.values())
    return {"schema": "pcrl-utility-r3-task-gate-replay-v1",
            "selected_recipe": selected["selected_recipe"],
            "raw_fit_validation_total_nll": selected["total_raw_nll"],
            "selected_recipe_fit_validation_usefulness": usefulness,
            "selected_recipe_useful_all_seed_task": useful_all,
            "screen_package_comparison": screen_comparison,
            "gate_by_seed": gate_out,
            "point_eligible_all_seed_task_reference_clauses": all_eligible,
            "conditional_admission_eligible": bool(all_eligible and useful_all),
            "source_selection_sha256": file_sha(selection_path),
            "F_artifact_lock_sha256": file_sha(f_lock_path),
            "scope": "reused Adult development roles; point gate only; no privacy result"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--recipe-selection", type=Path, required=True)
    parser.add_argument("--f-artifact-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r3_task_gate_replay", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("independent replay requires one numerical thread")
    _require(spec, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    result = replay(spec, args.private_root.resolve(strict=True),
                    args.export_root.resolve(strict=True), manifest,
                    args.recipe_selection, args.f_artifact_lock)
    if args.output.exists():
        raise FileExistsError("independent task replay is write-once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
