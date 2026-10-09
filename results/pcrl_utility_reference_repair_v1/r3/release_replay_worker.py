"""Independent feature-only replay of every selected conditional release.

Restores frozen heads and calibration through R3 code, then rebuilds each
recipient/coalition wire and complete attack matrix from admitted raw features.
No task or SEX label is read, and no R2 deployment/view code is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r3.mask_task_replay_worker import _a_forward
from results.pcrl_utility_reference_repair_v1.r3.source_reference import (
    apply_inverse_temperature, file_sha,
)
from results.pcrl_utility_reference_repair_v1.r3.task_gate_replay_worker import (
    _require, _role,
)
from results.pcrl_utility_reference_repair_v1.r3.task_replay import (
    _resolve_private, array_sha,
)


STUDY = Path(__file__).resolve().parents[1]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
TASKS = ("income", "occupation_group")
VIEWS = TASKS + ("coalition",)
VIEW_FIELDS = ("row_id", "unit", "x", "x_defense_aware", "x_raw_only")
LOCAL_NUMERIC = ("selected_values", "probabilities", "decisions")
COALITION_NUMERIC = ("selected_values", "income_probabilities",
                     "occupation_group_probabilities", "income_decisions",
                     "occupation_group_decisions")
LOCAL_CONSTANTS = ("feature_names", "selected_groups", "mask_id",
                   "mask_sha256", "task_package_sha256", "head_id",
                   "calibrator_id", "flags")
COALITION_CONSTANTS = ("feature_names", "selected_groups", "income_groups",
                       "occupation_group_groups", "income_mask_id",
                       "occupation_group_mask_id", "income_mask_sha256",
                       "occupation_group_mask_sha256",
                       "income_task_package_sha256",
                       "occupation_group_task_package_sha256",
                       "income_head_id", "occupation_group_head_id",
                       "income_calibrator_id", "occupation_group_calibrator_id",
                       "flags")
TOLERANCE = 2e-7


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise PermissionError(message)


def _json(spec: dict, path: Path) -> dict:
    _require(spec, path)
    value = json.loads(path.read_text())
    _check(isinstance(value, dict), "sealed JSON is not an object")
    return value


def _role_names(spec: dict, export: Path, role: str) -> tuple[str, ...]:
    path = export / role / "features.npz"
    _require(spec, path)
    with np.load(path, allow_pickle=False) as z:
        return tuple(str(name) for name in z["feature_names"])


def _mask_task(spec: dict, root: Path, bank_mask: dict, seed: int,
               task: str, gate_sha: str) -> dict:
    mask_id = bank_mask["mask_id"]
    fit_path = root / "masks" / mask_id / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
    fit = _json(spec, fit_path)
    _check(fit["schema"] == "pcrl-utility-mask-fit-receipt-v1" and
           fit["mask_id"] == mask_id and fit["mask_sha256"] == bank_mask["mask_sha256"] and
           fit["seed"] == seed and fit["selected_recipe"] == "A" and
           fit["gate_sha256"] == gate_sha,
           "selected mask fit identity differs")
    entry = fit["task_packages"][task]
    package_path = _resolve_private(root, entry["relative_path"])
    _check(_require(spec, package_path) == entry["sha256"],
           "selected task package SHA differs")
    package = json.loads(package_path.read_text())
    _check(package["schema"] == "pcrl-utility-masked-task-package-v1" and
           package["mask_id"] == mask_id and package["mask_sha256"] == bank_mask["mask_sha256"] and
           package["task"] == task and package["seed"] == seed and
           package["selected_recipe"] == "A" and package["gate_sha256"] == gate_sha and
           package["source_indices"] == bank_mask["column_indices"] and
           package["selected_atoms"] == bank_mask["atoms"] and
           package["conditional_lock_sha256"] == fit["conditional_lock_sha256"] and
           package["model"] == fit["model_stages"]["final"][task] and
           package["calibrator"] == fit["calibrators"][task] and
           package["fill_constants"] == fit["fills"]["final"],
           "selected task package metadata differs")
    fill_entry = package["fill_constants"]
    fill_path = _resolve_private(root, fill_entry["relative_path"])
    _check(_require(spec, fill_path) == fill_entry["sha256"],
           "selected task public fill SHA differs")
    with fill_path.open("rb") as stream:
        fill = np.load(stream, allow_pickle=False)
    _check(fill.dtype == np.float32 and fill.shape == (83,),
           "selected task public fill schema differs")
    cal_entry = package["calibrator"]
    cal_path = _resolve_private(root, cal_entry["relative_path"])
    _check(_require(spec, cal_path) == cal_entry["sha256"],
           "selected task calibrator SHA differs")
    cal = json.loads(cal_path.read_text())
    _check(cal["schema"] == "pcrl-portfolio-global-temp-v1" and
           cal["task"] == ("income" if task == "income" else "occupation") and
           cal["arm_id"] == package["model_id"] and cal["model_seed"] == seed and
           cal["fit_role"] == "CALIBRATION_HELDOUT", "selected task CAL identity differs")
    return {"package": package, "package_sha256": entry["sha256"],
            "model_sha256": package["model"]["sha256"],
            "calibrator_sha256": cal_entry["sha256"], "alpha": float(cal["alpha"])}


def _blocks(selected: np.ndarray, probability: np.ndarray,
            decision: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.ascontiguousarray(selected, dtype=np.float64)
    return raw, np.concatenate((raw, probability,
                                np.eye(classes, dtype=np.float64)[decision]), axis=1)


def _schema_hash(numeric: tuple[str, ...], constants: tuple[str, ...],
                 constant_values: dict, blocks: tuple[np.ndarray, ...]) -> str:
    spans = []
    start = 0
    for name, block in zip(numeric, blocks):
        spans.append((name, start, start + block.shape[1]))
        start += block.shape[1]
    metadata = tuple((name, json.dumps(constant_values[name], sort_keys=True,
                                       separators=(",", ":")))
                     for name in constants)
    content = {"components": numeric, "wire_components": numeric + constants,
               "component_columns": tuple(spans), "wire_constants": metadata}
    return hashlib.sha256(json.dumps(content, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _view_hashes(x: np.ndarray, aware: np.ndarray, raw: np.ndarray,
                 schema_hash: str) -> dict:
    return {"x": array_sha(np.asarray(x, dtype=np.float64)),
            "x_defense_aware": array_sha(np.asarray(aware, dtype=np.float64)),
            "cell_key": None, "x_metadata": None, "x_code_bits": None,
            "x_raw_only": array_sha(np.asarray(raw, dtype=np.float64)),
            "wire_schema": schema_hash}


def _compare_float(saved: np.ndarray, expected: np.ndarray,
                   label: str) -> float:
    _check(saved.shape == expected.shape and np.isfinite(saved).all() and
           np.isfinite(expected).all(), label + " shape or values differ")
    delta = float(np.max(np.abs(saved - expected))) if saved.size else 0.0
    _check(delta <= TOLERANCE, label + " differs from independent replay")
    return delta


def _verify_role(spec: dict, root: Path, source: dict, names: tuple[str, ...],
                 bank_masks: dict, pair: dict, seed: int, role: str,
                 release: dict, task_models: dict) -> dict:
    entry = release["roles"][role]
    archive_path = root / "releases" / f"seed{seed}" / pair["pair_id"] / f"{role}.npz"
    _check(_require(spec, archive_path) == entry["npz_sha256"] and
           entry["source_feature_sha256"] ==
           spec["role_inputs"][f"{role}:features"]["sha256"],
           "release role source or archive differs")
    with np.load(archive_path, allow_pickle=False) as archive:
        _check(set(archive.files) == set(entry["array_sha256"]),
               "release array member set differs")
        saved = {name: archive[name] for name in archive.files}
    for name, value in saved.items():
        _check(array_sha(value) == entry["array_sha256"][name],
               "release array digest differs: " + name)
    expected_fields = ({f"{view}__view__{field}" for view in VIEWS
                        for field in VIEW_FIELDS} |
                       {f"{task}__wire__{field}" for task in TASKS
                        for field in LOCAL_NUMERIC} |
                       {f"coalition__wire__{field}" for field in COALITION_NUMERIC})
    _check(set(saved) == expected_fields and
           np.array_equal(saved["income__view__row_id"], source["row_id"]) and
           np.array_equal(saved["coalition__view__unit"], source["unit"]),
           "release complete-view inventory or people differ")
    maxima = {"probability": 0.0, "complete_view": 0.0}
    expected = {}
    for task, classes, mask_id in (("income", 2, pair["income_mask_id"]),
                                   ("occupation_group", 6, pair["occupation_mask_id"])):
        mask = bank_masks[mask_id]
        selected = np.ascontiguousarray(source["X"][:, mask["column_indices"]])
        model = task_models[task]
        raw = _a_forward(spec, root, model["package"]["model"], model["package"],
                         mask, task, classes, selected)
        probability = apply_inverse_temperature(raw, model["alpha"])
        decision = probability.argmax(axis=1)
        prefix = task + "__wire__"
        _check(np.array_equal(saved[prefix + "selected_values"], selected),
               task + " selected values differ")
        maxima["probability"] = max(maxima["probability"], _compare_float(
            saved[prefix + "probabilities"], probability, task + " probability"))
        _check(np.array_equal(saved[prefix + "decisions"], decision),
               task + " decisions differ")
        constant = entry["wire_constants"][task]
        _check(set(constant) == set(LOCAL_CONSTANTS) and
               constant["feature_names"] == [names[i] for i in mask["column_indices"]] and
               constant["selected_groups"] == mask["atoms"] and
               constant["mask_id"] == mask_id and
               constant["mask_sha256"] == mask["mask_sha256"] and
               constant["task_package_sha256"] == model["package_sha256"] and
               constant["head_id"] ==
               model["package"]["model_id"] + ":SHA256:" + model["model_sha256"] and
               constant["calibrator_id"] ==
               model["package"]["model_id"] + ":SHA256:" + model["calibrator_sha256"] and
               constant["flags"] == [],
               task + " public wire constants differ")
        selected64, numeric = _blocks(selected, saved[prefix + "probabilities"],
                                      saved[prefix + "decisions"], classes)
        aware = np.concatenate((numeric, raw), axis=1)
        blocks = (selected64, saved[prefix + "probabilities"],
                  np.eye(classes, dtype=np.float64)[saved[prefix + "decisions"]])
        schema_hash = _schema_hash(LOCAL_NUMERIC, LOCAL_CONSTANTS, constant, blocks)
        expected[task] = {"raw": raw, "probability": saved[prefix + "probabilities"],
                          "decision": decision, "selected": selected64,
                          "x": numeric, "aware": aware,
                          "schema_hash": schema_hash}
    union = np.ascontiguousarray(source["X"][:, pair["union_column_indices"]])
    _check(np.array_equal(saved["coalition__wire__selected_values"], union) and
           np.array_equal(saved["coalition__wire__income_probabilities"],
                          expected["income"]["probability"]) and
           np.array_equal(saved["coalition__wire__occupation_group_probabilities"],
                          expected["occupation_group"]["probability"]) and
           np.array_equal(saved["coalition__wire__income_decisions"],
                          expected["income"]["decision"]) and
           np.array_equal(saved["coalition__wire__occupation_group_decisions"],
                          expected["occupation_group"]["decision"]),
           "coalition wire differs from both recipients")
    constant = entry["wire_constants"]["coalition"]
    _check(set(constant) == set(COALITION_CONSTANTS) and
           constant["feature_names"] == [names[i] for i in pair["union_column_indices"]] and
           constant["selected_groups"] == pair["union_atoms"] and
           constant["income_groups"] == bank_masks[pair["income_mask_id"]]["atoms"] and
           constant["occupation_group_groups"] ==
           bank_masks[pair["occupation_mask_id"]]["atoms"] and
           constant["income_mask_id"] == pair["income_mask_id"] and
           constant["occupation_group_mask_id"] == pair["occupation_mask_id"] and
           constant["income_mask_sha256"] == bank_masks[pair["income_mask_id"]]["mask_sha256"] and
           constant["occupation_group_mask_sha256"] ==
           bank_masks[pair["occupation_mask_id"]]["mask_sha256"] and
           constant["income_task_package_sha256"] ==
           task_models["income"]["package_sha256"] and
           constant["occupation_group_task_package_sha256"] ==
           task_models["occupation_group"]["package_sha256"] and
           constant["income_head_id"] == entry["wire_constants"]["income"]["head_id"] and
           constant["occupation_group_head_id"] ==
           entry["wire_constants"]["occupation_group"]["head_id"] and
           constant["income_calibrator_id"] ==
           entry["wire_constants"]["income"]["calibrator_id"] and
           constant["occupation_group_calibrator_id"] ==
           entry["wire_constants"]["occupation_group"]["calibrator_id"] and
           constant["flags"] == [],
           "coalition public wire constants differ")
    union64 = np.ascontiguousarray(union, dtype=np.float64)
    i, o = expected["income"], expected["occupation_group"]
    x = np.concatenate((union64, i["probability"], o["probability"],
                        np.eye(2, dtype=np.float64)[i["decision"]],
                        np.eye(6, dtype=np.float64)[o["decision"]]), axis=1)
    aware = np.concatenate((x, i["raw"], o["raw"]), axis=1)
    expected["coalition"] = {"selected": union64, "x": x, "aware": aware,
                             "schema_hash": _schema_hash(
                                 COALITION_NUMERIC, COALITION_CONSTANTS,
                                 constant, (union64, i["probability"], o["probability"],
                                            np.eye(2, dtype=np.float64)[i["decision"]],
                                            np.eye(6, dtype=np.float64)[o["decision"]]))}
    _check(entry["union_source_columns"] == pair["union_column_indices"] and
           entry["ordered_row_unit_sha256"] ==
           array_sha(np.column_stack((source["row_id"], source["unit"]))),
           "coalition raw union columns or people differ")
    for view in VIEWS:
        detail = expected[view]
        for field, source_value in (("row_id", source["row_id"]),
                                    ("unit", source["unit"])):
            _check(np.array_equal(saved[f"{view}__view__{field}"], source_value),
                   view + " complete-view people differ")
        for field, value in (("x_raw_only", detail["selected"]),
                             ("x", detail["x"]),
                             ("x_defense_aware", detail["aware"])):
            maxima["complete_view"] = max(maxima["complete_view"],
                                           _compare_float(saved[f"{view}__view__{field}"],
                                                          value, view + " " + field))
        hashes = _view_hashes(saved[f"{view}__view__x"],
                              saved[f"{view}__view__x_defense_aware"],
                              saved[f"{view}__view__x_raw_only"],
                              detail["schema_hash"])
        _check(hashes == entry["view_content_sha256"][view],
               view + " complete-view hash/schema differs")
    return maxima


def replay(spec: dict, root: Path, export: Path, manifest_path: Path,
           bank_path: Path, gate_path: Path, audit_path: Path,
           lock: dict) -> dict:
    manifest = _json(spec, manifest_path)
    bank = _json(spec, bank_path)
    gate = _json(spec, gate_path)
    audit = _json(spec, audit_path)
    _check(lock.get("source_map_sha256") == file_sha(manifest_path) and
           lock.get("mask_bank_sha256") == file_sha(bank_path) and
           lock.get("full_input_gate_sha256") == file_sha(gate_path) and
           lock.get("pair_selection_audit_sha256") == file_sha(audit_path) and
           gate.get("status") == "PASS_FULL_INPUT_POINT_GATE" and
           gate.get("selected_recipe") == "A" and
           audit.get("status") == "PASS_AGGREGATE_SELECTION_REPLAY" and
           bank.get("schema") == "pcrl-focused-mask-bank-v1",
           "release replay source lock differs")
    selected = {str(seed): sorted({value for value in audit["by_seed"][str(seed)][
        "selected_rule_pair_ids"].values() if value is not None}) for seed in (0, 1)}
    _check(lock.get("selected_pairs_by_seed") == selected,
           "release replay selected pair set differs")
    pair_by_id = {row["pair_id"]: row for row in bank["pairs"]}
    mask_by_id = {row["mask_id"]: row for row in bank["masks"]}
    roles = {role: _role(spec, export, manifest, role, ("features",))
             for role in ROLES}
    names = {role: _role_names(spec, export, role) for role in ROLES}
    _check(names[ROLES[0]] == names[ROLES[1]],
           "release replay role feature names differ")
    results = {}
    max_probability = 0.0
    max_view = 0.0
    for seed in (0, 1):
        results[str(seed)] = {}
        for pair_id in selected[str(seed)]:
            pair = pair_by_id[pair_id]
            pair = dict(pair, income_mask_id=pair_id.split("__")[0],
                        occupation_mask_id=pair_id.split("__")[1])
            task_models = {
                task: _mask_task(spec, root, mask_by_id[mask_id], seed, task,
                                 file_sha(gate_path))
                for task, mask_id in (("income", pair["income_mask_id"]),
                                      ("occupation_group", pair["occupation_mask_id"]))
            }
            path = root / "releases" / f"seed{seed}" / pair_id / "RELEASE_RECEIPT.json"
            release = _json(spec, path)
            selection = _json(spec, root / "pair_select" / f"seed{seed}" /
                              "PAIR_SELECT_RECEIPT.json")
            _check(release["schema"] == "pcrl-utility-conditional-complete-release-v1" and
                   release["status"] == "FROZEN" and release["seed"] == seed and
                   release["pair_id"] == pair_id and
                   release["bank_sha256"] == selection["bank_sha256"] and
                   release["lock_sha256"] == lock["prior_release_lock_sha256"] and
                   release["gate_sha256"] == file_sha(gate_path) and
                   release["manifest_sha256"] == file_sha(manifest_path) and
                   release["task_package_sha256_by_purpose"] == {
                       task: task_models[task]["package_sha256"] for task in TASKS} and
                   release["public_mask_sha256_by_purpose"] == {
                       task: mask_by_id[pair["income_mask_id" if task == "income"
                                             else "occupation_mask_id"]]["mask_sha256"]
                       for task in TASKS} and
                   set(release["roles"]) == set(ROLES),
                   "release pair receipt identity differs")
            results[str(seed)][pair_id] = {}
            for role in ROLES:
                maxima = _verify_role(spec, root, roles[role], names[role],
                                      mask_by_id, pair, seed, role, release,
                                      task_models)
                max_probability = max(max_probability, maxima["probability"])
                max_view = max(max_view, maxima["complete_view"])
                results[str(seed)][pair_id][role] = maxima
    return {"schema": "pcrl-utility-r3-conditional-release-replay-v1",
            "status": "PASS_RELEASE_REPLAY", "development_only": True,
            "pair_selection_audit_sha256": file_sha(audit_path),
            "full_input_gate_sha256": file_sha(gate_path),
            "source_release_lock_sha256": lock["prior_release_lock_sha256"],
            "selected_pairs_by_seed": selected,
            "role_access": ["ATTACK_FIT_NEW:features", "INNER_SELECTION:features"],
            "max_abs_probability_difference": max_probability,
            "max_abs_complete_view_difference": max_view,
            "by_seed_pair_role": results,
            "scope": "reused Adult development feature roles only; no SEX or task labels"}


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
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
    for name in ("unit-id", "queue-root", "lock", "manifest", "export-root",
                 "private-root", "full-input-gate", "mask-bank",
                 "pair-selection-audit", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    path = Path(args.output)
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r3_release_replay", path)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R3 release replay requires one numerical thread")
    root = Path(args.private_root).resolve(strict=True)
    output = root / "r3/RELEASE_REPLAY.json"
    if (root != STUDY / ".private" or path.resolve() != output
            or spec["outputs"] != [str(output)] or
            set(spec.get("role_inputs", {})) !=
            {"ATTACK_FIT_NEW:features", "INNER_SELECTION:features"}):
        raise PermissionError("R3 release replay output or role contract differs")
    lock = json.loads(Path(args.lock).read_text())
    result = replay(spec, root, Path(args.export_root), Path(args.manifest),
                    Path(args.mask_bank), Path(args.full_input_gate),
                    Path(args.pair_selection_audit), lock)
    _write_once(output, result)


if __name__ == "__main__":
    main()
