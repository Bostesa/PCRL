"""Independent same-INNER task replay of the optional historical F diagnostic.

This computes endpoints from separately sealed old and new F predictions. It
does not import either producer's scoring, selection, or deployment code.
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
from results.pcrl_utility_reference_repair_v1.r3.independent_task import task_metrics
from results.pcrl_utility_reference_repair_v1.r3.source_reference import file_sha
from results.pcrl_utility_reference_repair_v1.r3.task_replay import _read_bound_npz, array_sha


THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
OLD_FIELDS = {"row_id", "unit", "income", "occupation", "old_F_income",
              "old_F_occupation_group", "U0_1", "U0_2", "Ucal_1", "Ucal_2"}
NEW_FIELDS = {"row_id", "unit", "income", "occupation", "F_income",
              "F_occupation_group", "U0_1", "U0_2", "Ucal_1", "Ucal_2"}
TASKS = (("income", "income"), ("occupation_group", "occupation"))


def _require(spec: dict, path: Path) -> str:
    resolved = path.resolve(strict=True)
    digest = spec["inputs"].get(str(resolved))
    if digest is None or file_sha(resolved) != digest:
        raise PermissionError("diagnostic replay input differs from sealed queue")
    return digest


def _read_json(spec: dict, path: Path) -> dict:
    _require(spec, path)
    return json.loads(path.read_text())


def _role_labels(spec: dict, export_root: Path, manifest: dict) -> dict:
    out = {}
    identity = None
    for field in ("income", "occupation"):
        path = export_root / "INNER_SELECTION" / f"{field}.npz"
        if _require(spec, path) != manifest["role_file_sha256"][f"INNER_SELECTION:{field}"]:
            raise PermissionError("INNER task role differs from manifest")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"row_id", "unit", field}:
                raise PermissionError("INNER task role members differ")
            row, unit, label = (np.array(z[name], copy=True)
                                for name in ("row_id", "unit", field))
        if identity is None:
            identity = (row, unit)
        elif not (np.array_equal(row, identity[0]) and
                  np.array_equal(unit, identity[1])):
            raise PermissionError("INNER task role people differ")
        width = 2 if field == "income" else 6
        if (label.dtype != np.int64 or label.shape != row.shape
                or np.any(label < 0) or np.any(label >= width)):
            raise PermissionError("INNER task labels differ")
        out[field] = label
    row, unit = identity
    meta = manifest["roles"]["INNER_SELECTION"]
    if (row.dtype != np.int64 or unit.dtype != np.int64
            or row.shape != (meta["rows"],) or unit.shape != row.shape
            or len(np.unique(row)) != len(row)
            or len(np.unique(unit)) != meta["groups"]
            or array_sha(np.column_stack((row, unit))) !=
            meta["row_unit_pair_sha256"]):
        raise PermissionError("INNER task role identity differs")
    out["row_id"], out["unit"] = row, unit
    return out


def _compare(old: dict[str, np.ndarray], new: dict[str, np.ndarray],
             role: dict[str, np.ndarray]) -> dict:
    for key in ("row_id", "unit", "income", "occupation", "U0_1",
                "U0_2", "Ucal_1", "Ucal_2"):
        if not np.array_equal(old[key], new[key]):
            raise PermissionError("historical and selected F people or references differ")
    for key in ("row_id", "unit", "income", "occupation"):
        if not np.array_equal(old[key], role[key]):
            raise PermissionError("diagnostic prediction does not match INNER role")
    metrics = {"old_validation_F": {}, "new_all_FIT_F": {}}
    delta_old_minus_new = {}
    for task, label_key in TASKS:
        a = task_metrics(old[f"old_F_{task}"], role[label_key])
        b = task_metrics(new[f"F_{task}"], role[label_key])
        metrics["old_validation_F"][task] = a
        metrics["new_all_FIT_F"][task] = b
        delta_old_minus_new[task] = {
            key: a[key] - b[key] for key in ("accuracy", "nll", "brier")}
    return {"metrics": metrics, "old_minus_new": delta_old_minus_new}


def replay(spec: dict, private: Path, export_root: Path, manifest: dict,
           gate_path: Path) -> dict:
    gate = _read_json(spec, gate_path)
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") not in ("PASS_FULL_INPUT_POINT_GATE",
                                          "FAIL_FULL_INPUT_POINT_GATE")
            or gate.get("common_role") != "INNER_SELECTION"):
        raise PermissionError("completed primary full-input gate required")
    role = _role_labels(spec, export_root, manifest)
    if array_sha(np.column_stack((role["row_id"], role["unit"]))) != gate[
            "common_row_unit_pair_sha256"]:
        raise PermissionError("primary gate role differs")
    by_seed = {}
    for seed in (0, 1):
        old_dir = private / "r1" / "old_f_inner" / f"seed_{seed}"
        new_dir = private / "r1" / "gate" / f"seed_{seed}"
        old_receipt = _read_json(spec, old_dir / "DIAGNOSTIC_RECEIPT.json")
        new_receipt = _read_json(spec, new_dir / "GATE_RECEIPT.json")
        if (old_receipt.get("schema") !=
                "pcrl-utility-old-validation-f-inner-diagnostic-v1"
                or old_receipt.get("seed") != seed
                or old_receipt.get("primary_F_gate_sha256") != file_sha(gate_path)
                or old_receipt.get("inner_row_unit_sha256") !=
                gate["common_row_unit_pair_sha256"]
                or old_receipt.get("causal_attribution") != "NOT_SUPPORTED"
                or old_receipt.get("old_fit_rows") != {
                    "income": 11574 if seed == 0 else 11575,
                    "occupation_group": 11574 if seed == 0 else 11575}
                or _require(spec, new_dir / "GATE_RECEIPT.json") !=
                gate["r1_gate_receipt_sha256_by_seed"][str(seed)]
                or new_receipt.get("predictions_sha256") !=
                gate["r1_gate_prediction_sha256_by_seed"][str(seed)]):
            raise PermissionError("diagnostic or primary receipt identity differs")
        old_pred = old_dir / "OLD_F_INNER_PREDICTIONS.npz"
        new_pred = new_dir / "GATE_PREDICTIONS.npz"
        if (old_receipt.get("predictions_relative_path") !=
                str(old_pred.relative_to(private))
                or new_receipt.get("predictions_relative_path") !=
                str(new_pred.relative_to(private))
                or _require(spec, old_pred) != old_receipt["predictions_sha256"]
                or _require(spec, new_pred) != new_receipt["predictions_sha256"]):
            raise PermissionError("diagnostic prediction bytes differ")
        old = _read_bound_npz(old_pred, old_receipt["predictions_sha256"], OLD_FIELDS)
        new = _read_bound_npz(new_pred, new_receipt["predictions_sha256"], NEW_FIELDS)
        result = _compare(old, new, role)
        result["reference_metrics"] = {}
        for reference in ("U0", "Ucal"):
            result["reference_metrics"][reference] = {}
            for index, (task, label_key) in enumerate(TASKS, 1):
                result["reference_metrics"][reference][task] = task_metrics(
                    old[f"{reference}_{index}"], role[label_key])
        max_metric_difference = 0.0
        for arm in ("old_validation_F", "U0", "Ucal"):
            for task, _ in TASKS:
                scored = (result["metrics"][arm][task] if arm == "old_validation_F"
                          else result["reference_metrics"][arm][task])
                reported = old_receipt["metrics"][arm][task]
                for key in ("accuracy", "nll", "brier"):
                    difference = abs(scored[key] - reported[key])
                    max_metric_difference = max(max_metric_difference, difference)
                    if difference > 1e-12:
                        raise ValueError("historical diagnostic metric differs")
                if (scored["class_counts"] != reported["class_counts"]
                        or scored["recall"] != reported["per_class_recall"]):
                    raise ValueError("historical diagnostic class result differs")
        for task, _ in TASKS:
            for reference in ("U0", "Ucal"):
                old_metrics = result["metrics"]["old_validation_F"][task]
                ref_metrics = result["reference_metrics"][reference][task]
                reported = old_receipt["margins"][task][reference]
                for key, actual in (
                    ("accuracy_drop", ref_metrics["accuracy"] - old_metrics["accuracy"]),
                    ("nll_excess", old_metrics["nll"] - ref_metrics["nll"]),
                    ("brier_excess", old_metrics["brier"] - ref_metrics["brier"]),
                ):
                    if abs(reported[key] - actual) > 1e-12:
                        raise ValueError("historical diagnostic margin differs")
        result["max_abs_old_receipt_metric_difference"] = max_metric_difference
        by_seed[str(seed)] = result
    return {
        "schema": "pcrl-utility-r3-old-f-inner-replay-v1",
        "status": "OBSERVATIONAL_DIAGNOSTIC_REPLAYED",
        "primary_F_gate_sha256": file_sha(gate_path),
        "common_role": "INNER_SELECTION",
        "by_seed": by_seed,
        "interpretation": "Same permitted INNER people, unequal prior vs new FIT training exposures; no causal attribution or privacy result",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r3_old_f_replay", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("independent diagnostic requires one thread")
    private = args.private_root.resolve(strict=True)
    if private != Path(__file__).resolve().parents[1] / ".private":
        raise PermissionError("independent diagnostic private root differs")
    _require(spec, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    result = replay(spec, private, args.export_root.resolve(strict=True),
                    manifest, args.full_input_gate)
    _write_once(args.output, result)


if __name__ == "__main__":
    main()
