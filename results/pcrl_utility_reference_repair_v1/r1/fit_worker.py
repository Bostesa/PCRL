"""Sealed R1 full-input recipe screen, refit, and common-role utility gate.

Every stage enters through R0's live queue admission before a role archive is
opened. Only admitted role-only task arrays are available; SEX is never read.
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

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import (
    array_sha, inner_split, outer_split,
)
from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_utility_reference_repair_v1.r1.source_task import (
    fit_source_heads, predict_source, restore_source, save_source,
    select_source_head_C, train_source_encoder,
)


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
C_DIR = REPO / "results/pcrl_method_portfolio_v1/teams/c"
SHARED_DIR = REPO / "results/pcrl_method_portfolio_v1/shared"
for directory in (C_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from prototype import AdultGroupedSchema  # noqa: E402
from training import HeadSpec, fit_task_head, restore_head, save_head  # noqa: E402
from results.pcrl_method_portfolio_v1.shared import common_eval  # noqa: E402


TASKS = (("income", "income", 2), ("occupation_group", "occupation", 6))
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def _write_once_json(path: Path, payload: dict) -> str:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _write_once_npz(path: Path, **arrays) -> str:
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


def _role(store: RoleOnlyStore, role: str, *, references_seed: int | None = None):
    feature = store.read(role, "features")
    income = store.read(role, "income")
    occupation = store.read(role, "occupation")
    records = [feature, income, occupation]
    reference = None
    if references_seed is not None:
        reference = store.read(role, f"references_s{references_seed}")
        records.append(reference)
    assert_same_people(*records)
    return feature, income["income"], occupation["occupation"], reference


def _head_id(recipe: str, seed: int, task: str, stage: str) -> str:
    return f"UTILITY-REPAIR:{recipe}:s{seed}:{task}:{stage}:F_FULL83"


def _calibrate(raw: tuple[np.ndarray, np.ndarray], cal,
               seed: int, recipe: str, stage: str) -> tuple[tuple[np.ndarray, ...], dict]:
    feature, yi, yo, _ = cal
    labels = (yi, yo)
    probabilities = []
    records = {}
    for i, (task, label, _) in enumerate(TASKS):
        record = common_eval.fit_global_temperature(
            raw[i], labels[i], task=label,
            arm_id=_head_id(recipe, seed, task, stage), model_seed=seed,
            row_id=feature["row_id"], unit=feature["unit"],
        )
        records[task] = record
        probabilities.append(common_eval.apply_global_temperature(raw[i], record))
    return tuple(probabilities), records


def _a_fit(x: np.ndarray, yi: np.ndarray, yo: np.ndarray,
           unit: np.ndarray, names: tuple[str, ...], seed: int, stage: str):
    schema = AdultGroupedSchema()
    schema.validate_feature_names(names)
    all_atoms = tuple(schema.atoms)
    heads = []
    for task, _, _ in TASKS:
        labels = yi if task == "income" else yo
        heads.append(fit_task_head(
            x, labels, unit, names, all_atoms,
            purpose=task, model_id=_head_id("A", seed, task, stage),
            seed=seed, spec=HeadSpec(),
        ))
    return tuple(heads)


def _a_predict(heads, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return tuple(head.predict_proba(x) for head in heads)


def _b_screen_fit(x: np.ndarray, yi: np.ndarray, yo: np.ndarray,
                  unit: np.ndarray, seed: int):
    inner_fit, inner_val, inner_receipt = inner_split(unit, yi, yo, seed)
    selection_model = train_source_encoder(x[inner_fit], yi[inner_fit],
                                           yo[inner_fit], seed)
    selected = select_source_head_C(
        selection_model, x[inner_fit], yi[inner_fit], yo[inner_fit],
        x[inner_val], yi[inner_val], yo[inner_val],
    )
    c_values = {task: selected[task]["selected_C"] for task, _, _ in TASKS}
    model = train_source_encoder(x, yi, yo, seed)
    heads = fit_source_heads(model, x, yi, yo, c_values)
    return model, heads, selected, inner_receipt


def _metrics(prob: tuple[np.ndarray, ...], yi: np.ndarray, yo: np.ndarray):
    return {task: common_eval.task_metrics(prob[i], y, task=label)
            for i, ((task, label, _), y) in enumerate(zip(TASKS, (yi, yo)))}


def screen(store: RoleOnlyStore, seed: int, root: Path, output: Path) -> dict:
    fit = _role(store, "OSF_DEFENSE_FIT")
    f, yi, yo, _ = fit
    names = tuple(str(name) for name in f["feature_names"])
    AdultGroupedSchema().validate_feature_names(names)
    outer_fit, outer_val, outer_receipt = outer_split(f["unit"], yi, yo, seed)
    a_heads = _a_fit(f["X"][outer_fit], yi[outer_fit], yo[outer_fit],
                     f["unit"][outer_fit], names, seed, "outer_screen")
    b_model, b_heads, b_choice, inner_receipt = _b_screen_fit(
        f["X"][outer_fit], yi[outer_fit], yo[outer_fit],
        f["unit"][outer_fit], seed,
    )
    raw = {
        "A": _a_predict(a_heads, f["X"][outer_val]),
        "B": predict_source(b_model, b_heads, f["X"][outer_val]),
    }
    screen_root = output.parent
    packages = {}
    for recipe in ("A", "B"):
        if recipe == "A":
            entries = {}
            for i, (task, _, _) in enumerate(TASKS):
                path = screen_root / f"A_{task}_head.npz"
                digest = save_head(a_heads[i], path, private_root=root)
                entries[task] = {"relative_path": str(path.relative_to(root)),
                                 "sha256": digest,
                                 "training_receipt": a_heads[i].receipt}
            packages[recipe] = entries
        else:
            bundle = screen_root / "B_source_bundle"
            hashes = save_source(b_model, b_heads, bundle)
            packages[recipe] = {"relative_directory": str(bundle.relative_to(root)),
                                "sha256": hashes, "C_by_task": b_choice,
                                "inner_split": inner_receipt}
    pred_path = screen_root / "SCREEN_PREDICTIONS.npz"
    arrays = {"row_id": f["row_id"][outer_val], "unit": f["unit"][outer_val],
              "income": yi[outer_val], "occupation": yo[outer_val]}
    for recipe in ("A", "B"):
        for i, (task, _, _) in enumerate(TASKS):
            arrays[f"{recipe}_{task}"] = raw[recipe][i]
    pred_sha = _write_once_npz(pred_path, **arrays)
    receipt = {
        "schema": "pcrl-utility-recipe-screen-v1", "seed": seed,
        "outer_split": outer_receipt, "B_inner_split": inner_receipt,
        "predictions_relative_path": str(pred_path.relative_to(root)),
        "predictions_sha256": pred_sha,
        "packages": packages,
        "metrics": {recipe: _metrics(raw[recipe], yi[outer_val], yo[outer_val])
                    for recipe in ("A", "B")},
        "task_only_selection_score": {
            recipe: float(sum(_metrics(raw[recipe], yi[outer_val], yo[outer_val])[task]["nll"]
                              for task, _, _ in TASKS)) for recipe in ("A", "B")},
        "fit_role_row_unit_sha256": array_sha(np.column_stack((f["row_id"], f["unit"]))),
        "calibration_role_access": "NOT_OPENED_IN_RECIPE_SELECTION",
    }
    _write_once_json(output, receipt)
    return receipt


def _selection(path: Path, seed: int) -> dict:
    choice = json.loads(path.read_text())
    if (choice.get("schema") != "pcrl-utility-recipe-selection-v1"
            or choice.get("status") != "FROZEN"
            or choice.get("selected_recipe") not in ("A", "B")
            or set(choice.get("screen_output_sha256_by_seed", {})) != {"0", "1"}
            or choice.get("rule") != "lowest_sum_outer_fit_validation_nll_A_tie"):
        raise PermissionError("R0 selected recipe lock is incomplete")
    return choice


def refit(store: RoleOnlyStore, seed: int, root: Path,
          output: Path, selection_path: Path) -> dict:
    choice = _selection(selection_path, seed)
    screen_path = root / "r1" / "screen" / f"seed_{seed}" / "SCREEN_RECEIPT.json"
    if sha_file(screen_path) != choice["screen_output_sha256_by_seed"][str(seed)]:
        raise PermissionError("recipe choice differs from frozen screen receipt")
    screen_receipt = json.loads(screen_path.read_text())
    recipe = choice["selected_recipe"]
    fit = _role(store, "OSF_DEFENSE_FIT")
    cal = _role(store, "CALIBRATION_HELDOUT")
    f, yi, yo, _ = fit
    cf, _, _, _ = cal
    names = tuple(str(name) for name in f["feature_names"])
    if names != tuple(str(name) for name in cf["feature_names"]):
        raise PermissionError("FIT/CAL feature order differs")
    packages = {}
    if recipe == "A":
        heads = _a_fit(f["X"], yi, yo, f["unit"], names, seed, "full_fit")
        predict = lambda x: _a_predict(heads, x)
        for i, (task, _, _) in enumerate(TASKS):
            path = output.parent / f"A_{task}_head.npz"
            digest = save_head(heads[i], path, private_root=root)
            packages[task] = {"relative_path": str(path.relative_to(root)),
                              "sha256": digest,
                              "training_receipt": heads[i].receipt}
    else:
        frozen_c = {task: screen_receipt["packages"]["B"]["C_by_task"][task]["selected_C"]
                    for task, _, _ in TASKS}
        if frozen_c != choice.get("selected_C_by_seed_task", {}).get(str(seed)):
            raise PermissionError("B numeric C differs from R0 recipe selection lock")
        model = train_source_encoder(f["X"], yi, yo, seed)
        heads = fit_source_heads(model, f["X"], yi, yo, frozen_c)
        bundle = output.parent / "B_source_bundle"
        hashes = save_source(model, heads, bundle)
        packages["B_source_bundle"] = {"relative_directory": str(bundle.relative_to(root)),
                                       "sha256": hashes, "C_by_task": frozen_c}
        predict = lambda x: predict_source(model, heads, x)
    cal_raw = predict(cf["X"])
    _, calibrators = _calibrate(cal_raw, cal, seed, recipe, "full_fit")
    for task, _, _ in TASKS:
        path = output.parent / f"{task}_calibrator.json"
        digest = common_eval.save_calibrator(calibrators[task], path)
        packages[f"calibrator_{task}"] = {"relative_path": str(path.relative_to(root)),
                                          "sha256": digest}
    task_packages = {}
    for task, _, _ in TASKS:
        package_path = output.parent / f"F_{task}_PACKAGE.json"
        content = {
            "schema": "pcrl-utility-full-F-task-package-v1",
            "seed": seed, "task": task, "selected_recipe": recipe,
            "selection_lock_sha256": sha_file(selection_path),
            "model": (packages[task] if recipe == "A" else
                      packages["B_source_bundle"]),
            "calibrator": packages[f"calibrator_{task}"],
            "feature_schema_sha256": store.manifest["source_schema_sha256"],
            "fit_row_unit_sha256": array_sha(np.column_stack((f["row_id"], f["unit"]))),
            "calibration_row_unit_sha256": array_sha(np.column_stack((cf["row_id"], cf["unit"]))),
        }
        task_packages[task] = {
            "relative_path": str(package_path.relative_to(root)),
            "sha256": _write_once_json(package_path, content),
        }
    receipt = {
        "schema": "pcrl-utility-full-F-package-v1", "seed": seed,
        "selected_recipe": recipe,
        "selection_lock_sha256": sha_file(selection_path),
        "screen_receipt_sha256": sha_file(screen_path),
        "fit_rows": len(yi), "calibration_rows": len(cf["X"]),
        "fit_row_unit_sha256": array_sha(np.column_stack((f["row_id"], f["unit"]))),
        "calibration_row_unit_sha256": array_sha(np.column_stack((cf["row_id"], cf["unit"]))),
        "packages": packages,
        "task_packages": task_packages,
    }
    _write_once_json(output, receipt)
    return receipt


def _restore_predict(receipt: dict, root: Path):
    recipe = receipt["selected_recipe"]
    seed = receipt["seed"]
    if recipe == "A":
        heads = tuple(restore_head(
            root / receipt["packages"][task]["relative_path"],
            expected_sha256=receipt["packages"][task]["sha256"],
            private_root=root) for task, _, _ in TASKS)
        predict = lambda x: _a_predict(heads, x)
    else:
        entry = receipt["packages"]["B_source_bundle"]
        model, heads = restore_source(root / entry["relative_directory"],
                                      seed, entry["sha256"])
        predict = lambda x: predict_source(model, heads, x)
    calibrators = {
        task: common_eval.load_calibrator(
            root / receipt["packages"][f"calibrator_{task}"]["relative_path"],
            receipt["packages"][f"calibrator_{task}"]["sha256"])
        for task, _, _ in TASKS}
    def calibrated(x):
        raw = predict(x)
        return tuple(common_eval.apply_global_temperature(raw[i], calibrators[task])
                     for i, (task, _, _) in enumerate(TASKS))
    return calibrated


def gate(store: RoleOnlyStore, seed: int, root: Path,
         output: Path, selection_path: Path, f_lock_path: Path) -> dict:
    choice = _selection(selection_path, seed)
    f_lock = json.loads(f_lock_path.read_text())
    if (f_lock.get("schema") != "pcrl-utility-f-artifact-lock-v1"
            or f_lock.get("status") != "FROZEN"
            or f_lock.get("recipe_selection_sha256") != sha_file(selection_path)
            or set(f_lock.get("full_refit_output_sha256_by_seed", {})) != {"0", "1"}
            or set(f_lock.get("f_package_sha256_by_seed_task", {})) != {"0", "1"}):
        raise PermissionError("both F packages must be frozen before INNER labels")
    for bound_seed in (0, 1):
        bound_receipt_path = root / "r1" / "full" / f"seed_{bound_seed}" / "F_PACKAGE_RECEIPT.json"
        if sha_file(bound_receipt_path) != f_lock["full_refit_output_sha256_by_seed"][str(bound_seed)]:
            raise PermissionError("frozen F package receipt differs from artifact lock")
        bound_receipt = json.loads(bound_receipt_path.read_text())
        bound_packages = f_lock["f_package_sha256_by_seed_task"][str(bound_seed)]
        if set(bound_packages) != {"income", "occupation_group"}:
            raise PermissionError("F artifact lock lacks one task package")
        for task in bound_packages:
            entry = bound_receipt["task_packages"][task]
            if (entry["sha256"] != bound_packages[task]
                    or sha_file(root / entry["relative_path"]) != bound_packages[task]):
                raise PermissionError("frozen F task package differs from artifact lock")
    receipt_path = root / "r1" / "full" / f"seed_{seed}" / "F_PACKAGE_RECEIPT.json"
    f_receipt = json.loads(receipt_path.read_text())
    if (f_receipt["selected_recipe"] != choice["selected_recipe"]
            or f_receipt["selection_lock_sha256"] != sha_file(selection_path)):
        raise PermissionError("frozen F package/selection identity differs")
    predict = _restore_predict(f_receipt, root)
    inner = _role(store, "INNER_SELECTION", references_seed=seed)
    feature, yi, yo, refs = inner
    p = predict(feature["X"])
    arrays = {"row_id": feature["row_id"], "unit": feature["unit"],
              "income": yi, "occupation": yo,
              "F_income": p[0], "F_occupation_group": p[1]}
    arrays.update({key: refs[key] for key in ("U0_1", "U0_2", "Ucal_1", "Ucal_2")})
    pred_path = output.parent / "GATE_PREDICTIONS.npz"
    pred_sha = _write_once_npz(pred_path, **arrays)
    metrics = {"F": _metrics(p, yi, yo),
               "U0": _metrics((refs["U0_1"], refs["U0_2"]), yi, yo),
               "Ucal": _metrics((refs["Ucal_1"], refs["Ucal_2"]), yi, yo)}
    margins = {}
    for task, _, _ in TASKS:
        margins[task] = {}
        for reference in ("U0", "Ucal"):
            candidate = metrics["F"][task]
            baseline = metrics[reference][task]
            acc_drop = baseline["accuracy"] - candidate["accuracy"]
            nll_excess = candidate["nll"] - baseline["nll"]
            brier_excess = candidate["brier"] - baseline["brier"]
            margins[task][reference] = {
                "accuracy_drop": acc_drop, "nll_excess": nll_excess,
                "brier_excess": brier_excess,
                "point_feasible": bool(acc_drop <= 0.010 and
                                       nll_excess <= 0.010 and brier_excess <= 0.005),
            }
    receipt = {
        "schema": "pcrl-utility-full-input-gate-result-v1", "seed": seed,
        "selected_recipe": choice["selected_recipe"],
        "F_artifact_lock_sha256": sha_file(f_lock_path),
        "F_package_receipt_sha256": sha_file(receipt_path),
        "predictions_relative_path": str(pred_path.relative_to(root)),
        "predictions_sha256": pred_sha,
        "inner_row_unit_sha256": array_sha(np.column_stack((feature["row_id"], feature["unit"]))),
        "metrics": metrics, "margins": margins,
        "point_feasible": all(margins[task][ref]["point_feasible"]
                              for task, _, _ in TASKS for ref in ("U0", "Ucal")),
    }
    _write_once_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "refit", "gate"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe-selection", type=Path)
    parser.add_argument("--f-artifact-lock", type=Path)
    args = parser.parse_args()
    assert_running_unit(args.unit_id, args.queue_root, args.lock,
                        f"r1_{args.stage}", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R1 worker requires one numerical thread per library")
    root = args.private_root.resolve(strict=True)
    expected = root / "r1" / ("screen" if args.stage == "screen" else
                              "full" if args.stage == "refit" else "gate") / f"seed_{args.seed}" / {
        "screen": "SCREEN_RECEIPT.json", "refit": "F_PACKAGE_RECEIPT.json",
        "gate": "GATE_RECEIPT.json"}[args.stage]
    if args.output.resolve() != expected:
        raise PermissionError("R1 output escapes its exact stage/seed path")
    if sha_file(args.manifest) != json.loads(args.lock.read_text())["source_map_sha256"]:
        raise PermissionError("R1 role manifest differs from campaign lock")
    store = RoleOnlyStore(args.export_root, args.manifest)
    if args.stage == "screen":
        screen(store, args.seed, root, expected)
    elif args.stage == "refit":
        if args.recipe_selection is None:
            raise PermissionError("R0 recipe selection lock required")
        refit(store, args.seed, root, expected, args.recipe_selection)
    else:
        if args.recipe_selection is None or args.f_artifact_lock is None:
            raise PermissionError("R0 recipe and F artifact locks required")
        gate(store, args.seed, root, expected, args.recipe_selection,
             args.f_artifact_lock)


if __name__ == "__main__":
    main()
