"""Sealed paired-group simultaneous utility intervals on the selected F gate.

Only previously frozen INNER task predictions and role-only task labels enter.
The interval is conditional descriptive development inference, not confirmation.
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
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import array_sha
from results.pcrl_utility_reference_repair_v1.r1.role_store import RoleOnlyStore, sha_file


STUDY = Path(__file__).resolve().parents[1]
TASKS = (("income", "income", 2, 1),
         ("occupation_group", "occupation", 6, 2))
REFERENCES = ("U0", "Ucal")
ENDPOINTS = ("accuracy_drop", "nll_excess", "brier_excess")
CAPS = {"accuracy_drop": 0.010, "nll_excess": 0.010,
        "brier_excess": 0.005}
REPLICATES = 2048
RNG_SEED = 20261009
BATCH = 64
CLIP = 1e-12
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
NPZ_MEMBERS = {"row_id", "unit", "income", "occupation", "F_income",
               "F_occupation_group", "U0_1", "U0_2", "Ucal_1", "Ucal_2"}


def bootstrap_group_means(costs: np.ndarray, unit: np.ndarray, *,
                          draws: int = REPLICATES, seed: int = RNG_SEED,
                          batch_size: int = BATCH) -> tuple[np.ndarray, str, int]:
    """Resample whole groups, preserving row weighting and the full cost vector."""
    x = np.asarray(costs, dtype=np.float64)
    groups = np.asarray(unit)
    if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0
            or not np.isfinite(x).all() or groups.shape != (len(x),)
            or groups.dtype != np.int64 or draws < 2 or batch_size < 1):
        raise ValueError("bootstrap requires finite aligned people/group costs")
    unique, inverse = np.unique(groups, return_inverse=True)
    count = len(unique)
    if count > np.iinfo(np.int32).max:
        raise ValueError("group count exceeds frozen int32 draw contract")
    group_sum = np.zeros((count, x.shape[1]), dtype=np.float64)
    np.add.at(group_sum, inverse, x)
    group_sizes = np.bincount(inverse, minlength=count).astype(np.float64)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    boot = np.empty((draws, x.shape[1]), dtype=np.float64)
    offset = 0
    while offset < draws:
        take = min(batch_size, draws - offset)
        indices = rng.integers(0, count, size=(take, count), dtype=np.int32)
        digest.update(np.ascontiguousarray(indices).tobytes())
        for i, sampled in enumerate(indices):
            multiplicity = np.bincount(sampled, minlength=count)
            boot[offset + i] = (multiplicity @ group_sum) / (multiplicity @ group_sizes)
        offset += take
    return boot, digest.hexdigest(), count


def simultaneous_cost_bounds(point: np.ndarray, bootstrap: np.ndarray,
                             caps: np.ndarray) -> dict:
    """Separate simultaneous one-sided basic-bootstrap upper/lower bounds."""
    value = np.asarray(point, dtype=np.float64)
    boot = np.asarray(bootstrap, dtype=np.float64)
    limit = np.asarray(caps, dtype=np.float64)
    if (value.ndim != 1 or limit.shape != value.shape
            or boot.ndim != 2 or boot.shape[1] != len(value)
            or len(boot) < 2 or not np.isfinite(value).all()
            or not np.isfinite(boot).all() or not np.isfinite(limit).all()):
        raise ValueError("simultaneous bounds need finite matching cost family")
    sd = np.std(boot, axis=0, ddof=1)
    nonzero = sd > 0
    if np.any(~nonzero) and np.max(np.abs(boot[:, ~nonzero] - value[~nonzero])) > 1e-12:
        raise ValueError("zero-variance cost differs from its point estimate")
    if np.any(nonzero):
        upper_pivot = np.max((value[nonzero] - boot[:, nonzero]) / sd[nonzero], axis=1)
        lower_pivot = np.max((boot[:, nonzero] - value[nonzero]) / sd[nonzero], axis=1)
        critical_upper = max(0.0, float(np.quantile(upper_pivot, 0.95,
                                                    method="higher")))
        critical_lower = max(0.0, float(np.quantile(lower_pivot, 0.95,
                                                    method="higher")))
    else:
        critical_upper = critical_lower = 0.0
    upper = value + critical_upper * sd
    lower = value - critical_lower * sd
    rows = []
    for point_j, sd_j, low_j, up_j, cap_j in zip(value, sd, lower, upper, limit):
        status = ("PASS" if up_j <= cap_j else
                  "VIOLATION" if low_j > cap_j else "UNRESOLVED")
        rows.append({"point": float(point_j), "bootstrap_sd": float(sd_j),
                     "lower": float(low_j), "upper": float(up_j),
                     "cap": float(cap_j), "status": status})
    return {"rows": rows, "upper_critical": critical_upper,
            "lower_critical": critical_lower,
            "nonzero_sd_costs": int(np.count_nonzero(nonzero))}


def _probabilities(value: np.ndarray, rows: int, classes: int) -> np.ndarray:
    p = np.asarray(value, dtype=np.float64)
    if (p.shape != (rows, classes) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or np.max(np.abs(p.sum(axis=1) - 1.0)) > 1e-8):
        raise PermissionError("stored task probability geometry differs")
    return p


def _costs(candidate: np.ndarray, reference: np.ndarray,
           labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = len(labels)
    classes = candidate.shape[1]
    f = _probabilities(candidate, rows, classes)
    u = _probabilities(reference, rows, classes)
    y = np.asarray(labels)
    if (y.dtype != np.int64 or y.shape != (rows,)
            or np.any(y < 0) or np.any(y >= classes)):
        raise PermissionError("INNER task labels differ from frozen class order")
    index = np.arange(rows)
    accuracy = (u.argmax(axis=1) == y).astype(np.float64) - (
        f.argmax(axis=1) == y).astype(np.float64)
    nll = -np.log(np.maximum(f[index, y], CLIP)) + np.log(
        np.maximum(u[index, y], CLIP))
    onehot = np.zeros_like(f)
    onehot[index, y] = 1.0
    brier = np.sum((f - onehot) ** 2, axis=1) - np.sum((u - onehot) ** 2, axis=1)
    return accuracy, nll, brier


def _read_prediction(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha_file(path) != expected_sha256:
        raise PermissionError("frozen gate predictions differ")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != NPZ_MEMBERS:
            raise PermissionError("frozen gate prediction members differ")
        return {name: np.array(archive[name], copy=True) for name in NPZ_MEMBERS}


def _write_once(path: Path, report: dict) -> None:
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
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


def run(store: RoleOnlyStore, private: Path, gate: dict,
        gate_sha256: str, output: Path) -> dict:
    yi = store.read("INNER_SELECTION", "income")
    yo = store.read("INNER_SELECTION", "occupation")
    if (not np.array_equal(yi["row_id"], yo["row_id"])
            or not np.array_equal(yi["unit"], yo["unit"])):
        raise PermissionError("INNER task role rows/groups differ")
    row = yi["row_id"]
    unit = yi["unit"]
    if (len(row) != 2235 or len(np.unique(unit)) != 2234
            or array_sha(np.column_stack((row, unit))) !=
            gate["common_row_unit_pair_sha256"]):
        raise PermissionError("INNER role people differ from primary F gate")
    costs = []
    order = []
    input_hashes = {}
    max_point_difference = 0.0
    for seed in (0, 1):
        folder = private / "r1" / "gate" / f"seed_{seed}"
        receipt_path = folder / "GATE_RECEIPT.json"
        expected_receipt = gate["r1_gate_receipt_sha256_by_seed"][str(seed)]
        if sha_file(receipt_path) != expected_receipt:
            raise PermissionError("primary F gate receipt differs")
        receipt = json.loads(receipt_path.read_text())
        pred_path = folder / "GATE_PREDICTIONS.npz"
        expected_pred = gate["r1_gate_prediction_sha256_by_seed"][str(seed)]
        if (receipt["predictions_sha256"] != expected_pred
                or receipt["predictions_relative_path"] != str(pred_path.relative_to(private))):
            raise PermissionError("primary F gate prediction identity differs")
        saved = _read_prediction(pred_path, expected_pred)
        input_hashes[str(seed)] = {"gate_receipt_sha256": expected_receipt,
                                   "gate_predictions_sha256": expected_pred}
        for name, value in (("row_id", row), ("unit", unit),
                            ("income", yi["income"]), ("occupation", yo["occupation"])):
            if not np.array_equal(saved[name], value):
                raise PermissionError("primary F gate person or task labels differ")
        for task, label_name, classes, suffix in TASKS:
            f = _probabilities(saved[f"F_{task}"], len(row), classes)
            labels = yi["income"] if label_name == "income" else yo["occupation"]
            for reference in REFERENCES:
                u = _probabilities(saved[f"{reference}_{suffix}"], len(row), classes)
                values = _costs(f, u, labels)
                for endpoint, vector in zip(ENDPOINTS, values):
                    point = float(np.mean(vector))
                    prior = float(receipt["margins"][task][reference][endpoint])
                    max_point_difference = max(max_point_difference, abs(point - prior))
                    if abs(point - prior) > 1e-12:
                        raise PermissionError("interval point cost differs from primary gate")
                    costs.append(vector)
                    order.append({"seed": seed, "task": task,
                                  "reference": reference, "endpoint": endpoint})
    matrix = np.column_stack(costs)
    if matrix.shape != (2235, 24):
        raise PermissionError("24-clause complete utility family differs")
    point = matrix.mean(axis=0)
    boot, draw_sha, group_count = bootstrap_group_means(matrix, unit)
    caps = np.asarray([CAPS[item["endpoint"]] for item in order], dtype=np.float64)
    bounds = simultaneous_cost_bounds(point, boot, caps)
    clauses = [{**identity, **result} for identity, result in zip(order, bounds["rows"])]
    counts = {status: sum(item["status"] == status for item in clauses)
              for status in ("PASS", "VIOLATION", "UNRESOLVED")}
    family = ("ALL_24_PASS" if counts["PASS"] == 24 else
              "HAS_VIOLATION" if counts["VIOLATION"] else "PRECISION_UNRESOLVED")
    report = {
        "schema": "pcrl-utility-simultaneous-intervals-v1",
        "scope": "conditional post-selection development inference on reused Adult INNER people",
        "selected_recipe": gate["selected_recipe"],
        "primary_F_gate_sha256": gate_sha256,
        "INNER_row_unit_sha256": gate["common_row_unit_pair_sha256"],
        "rows": len(row), "groups": group_count,
        "replicates": REPLICATES, "rng_seed": RNG_SEED,
        "rng": "numpy.default_rng.PCG64", "draw_batch_size": BATCH,
        "sampled_group_index_sha256": draw_sha,
        "numpy_version": np.__version__,
        "clipping": CLIP, "quantile": 0.95, "quantile_method": "higher",
        "family_size": len(clauses), "bounds_type": "separate_simultaneous_one_sided",
        "upper_critical": bounds["upper_critical"],
        "lower_critical": bounds["lower_critical"],
        "nonzero_sd_costs": bounds["nonzero_sd_costs"],
        "max_abs_point_replay_difference": max_point_difference,
        "input_sha256_by_seed": input_hashes,
        "clauses": clauses, "status_counts": counts, "family_status": family,
        "interpretation_limit": "not post-selection confirmation; fitted models and historical INNER exposure held fixed",
    }
    _write_once(output, report)
    return report


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
    assert_running_unit(args.unit_id, args.queue_root, args.lock,
                        "r1_utility_intervals", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("utility interval worker requires one numerical thread")
    private = args.private_root.resolve(strict=True)
    expected = private / "r1" / "intervals" / "SIMULTANEOUS_UTILITY_INTERVALS.json"
    if args.output.resolve() != expected:
        raise PermissionError("interval output escapes exact private path")
    lock = json.loads(args.lock.read_text())
    if (sha_file(args.manifest) != lock.get("source_map_sha256")
            or sha_file(args.full_input_gate) != lock.get("full_input_gate_sha256")):
        raise PermissionError("interval role or F gate differs from science lock")
    gate = json.loads(args.full_input_gate.read_text())
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("common_role") != "INNER_SELECTION"
            or gate.get("selected_recipe") != "A"):
        raise PermissionError("completed selected F point gate required")
    store = RoleOnlyStore(args.export_root, args.manifest)
    run(store, private, gate, lock["full_input_gate_sha256"], expected)


if __name__ == "__main__":
    main()
