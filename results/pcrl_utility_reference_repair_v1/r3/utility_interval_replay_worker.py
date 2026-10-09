"""Independent sealed replay of the full-input simultaneous utility intervals.

This uses only R3 scoring arithmetic. It never imports R1 scoring, interval,
selection, or role-loading code. The queue binds all person-level inputs.
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
from results.pcrl_utility_reference_repair_v1.r3.source_reference import file_sha
from results.pcrl_utility_reference_repair_v1.r3.task_gate_replay_worker import _require, _role
from results.pcrl_utility_reference_repair_v1.r3.task_replay import array_sha, _read_bound_npz


THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
MEMBERS = {"row_id", "unit", "income", "occupation", "F_income",
           "F_occupation_group", "U0_1", "U0_2", "Ucal_1", "Ucal_2"}
TASKS = (("income", "income", 2, 1), ("occupation_group", "occupation", 6, 2))
REFERENCES = ("U0", "Ucal")
ENDPOINTS = ("accuracy_drop", "nll_excess", "brier_excess")
CAPS = {"accuracy_drop": .010, "nll_excess": .010, "brier_excess": .005}


def group_draw_means(costs: np.ndarray, groups: np.ndarray, *, draws: int = 2048,
                     seed: int = 20261009, batch_size: int = 64) -> tuple[np.ndarray, str, int]:
    """Draw sorted group IDs, then sum their people and 24 costs per draw."""
    x = np.asarray(costs, dtype=np.float64)
    unit = np.asarray(groups)
    if (x.ndim != 2 or not x.size or not np.isfinite(x).all()
            or unit.dtype != np.int64 or unit.shape != (len(x),)
            or draws < 2 or batch_size < 1):
        raise ValueError("invalid paired group cost matrix")
    identifiers, inverse = np.unique(unit, return_inverse=True)
    group_sum = np.zeros((len(identifiers), x.shape[1]), dtype=np.float64)
    np.add.at(group_sum, inverse, x)
    group_size = np.bincount(inverse, minlength=len(identifiers)).astype(np.int64)
    random = np.random.default_rng(seed)
    digest = hashlib.sha256()
    estimates = np.empty((draws, x.shape[1]), dtype=np.float64)
    for start in range(0, draws, batch_size):
        take = min(batch_size, draws - start)
        selected = random.integers(0, len(identifiers),
                                   size=(take, len(identifiers)), dtype=np.int32)
        digest.update(np.ascontiguousarray(selected).tobytes())
        estimates[start:start + take] = (
            group_sum[selected].sum(axis=1) / group_size[selected].sum(axis=1)[:, None])
    return estimates, digest.hexdigest(), len(identifiers)


def one_sided_bounds(point: np.ndarray, bootstrap: np.ndarray,
                     caps: np.ndarray) -> dict:
    value = np.asarray(point, dtype=np.float64)
    sampled = np.asarray(bootstrap, dtype=np.float64)
    cap = np.asarray(caps, dtype=np.float64)
    if (value.ndim != 1 or sampled.ndim != 2 or sampled.shape[1] != len(value)
            or len(sampled) < 2 or cap.shape != value.shape
            or not np.isfinite(value).all() or not np.isfinite(sampled).all()
            or not np.isfinite(cap).all()):
        raise ValueError("invalid simultaneous cost family")
    sd = sampled.std(axis=0, ddof=1)
    variable = sd > 0
    if np.any(~variable) and np.max(np.abs(sampled[:, ~variable] - value[~variable])) > 1e-12:
        raise ValueError("constant draw cost differs from its estimate")
    if variable.any():
        upper_max = ((value[variable] - sampled[:, variable]) / sd[variable]).max(axis=1)
        lower_max = ((sampled[:, variable] - value[variable]) / sd[variable]).max(axis=1)
        upper_critical = max(0.0, float(np.quantile(upper_max, .95, method="higher")))
        lower_critical = max(0.0, float(np.quantile(lower_max, .95, method="higher")))
    else:
        upper_critical = lower_critical = 0.0
    lo = value - lower_critical * sd
    hi = value + upper_critical * sd
    rows = []
    for j in range(len(value)):
        status = ("PASS" if hi[j] <= cap[j] else
                  "VIOLATION" if lo[j] > cap[j] else "UNRESOLVED")
        rows.append({"point": float(value[j]), "bootstrap_sd": float(sd[j]),
                     "lower": float(lo[j]), "upper": float(hi[j]),
                     "cap": float(cap[j]), "status": status})
    return {"rows": rows, "upper_critical": upper_critical,
            "lower_critical": lower_critical,
            "nonzero_sd_costs": int(variable.sum())}


def _prob(value: np.ndarray, n: int, classes: int) -> np.ndarray:
    p = np.asarray(value, dtype=np.float64)
    if (p.shape != (n, classes) or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
            or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-8):
        raise PermissionError("stored task probabilities differ")
    return p


def _oriented_costs(f: np.ndarray, u: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, ...]:
    n, classes = f.shape
    candidate = _prob(f, n, classes)
    reference = _prob(u, n, classes)
    y = np.asarray(label)
    if (y.dtype != np.int64 or y.shape != (n,)
            or np.any((y < 0) | (y >= classes))):
        raise PermissionError("task label geometry differs")
    indexes = np.arange(n)
    accuracy = (reference.argmax(axis=1) == y).astype(float) - (
        candidate.argmax(axis=1) == y).astype(float)
    nll = np.log(np.maximum(reference[indexes, y], 1e-12)) - np.log(
        np.maximum(candidate[indexes, y], 1e-12))
    target = np.eye(classes)[y]
    brier = np.square(candidate - target).sum(axis=1) - np.square(
        reference - target).sum(axis=1)
    return accuracy, nll, brier


def replay(spec: dict, root: Path, export_root: Path, manifest: dict,
           gate_path: Path, r1_path: Path, lock: dict) -> dict:
    _require(spec, gate_path)
    _require(spec, r1_path)
    gate = json.loads(gate_path.read_text())
    original = json.loads(r1_path.read_text())
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("common_role") != "INNER_SELECTION"
            or gate.get("selected_recipe") != "A"
            or lock.get("full_input_gate_sha256") != file_sha(gate_path)
            or original.get("schema") != "pcrl-utility-simultaneous-intervals-v1"
            or original.get("primary_F_gate_sha256") != file_sha(gate_path)
            or original.get("family_size") != 24
            or original.get("replicates") != 2048
            or original.get("rng_seed") != 20261009
            or original.get("draw_batch_size") != 64
            or original.get("quantile_method") != "higher"):
        raise PermissionError("R1 interval or selected F gate identity differs")
    inner = _role(spec, export_root, manifest, "INNER_SELECTION", ("income", "occupation"))
    rows, groups = inner["row_id"], inner["unit"]
    if (len(rows) != 2235 or len(np.unique(groups)) != 2234
            or array_sha(np.column_stack((rows, groups))) !=
            gate["common_row_unit_pair_sha256"]):
        raise PermissionError("INNER people differ from selected F gate")
    order = []
    vectors = []
    max_gate_point_delta = 0.0
    for seed in (0, 1):
        directory = root / "r1/gate" / f"seed_{seed}"
        receipt_path = directory / "GATE_RECEIPT.json"
        pred_path = directory / "GATE_PREDICTIONS.npz"
        if (_require(spec, receipt_path) != gate["r1_gate_receipt_sha256_by_seed"][str(seed)]
                or _require(spec, pred_path) !=
                gate["r1_gate_prediction_sha256_by_seed"][str(seed)]):
            raise PermissionError("R1 gate receipt or vector SHA differs")
        receipt = json.loads(receipt_path.read_text())
        saved = _read_bound_npz(pred_path, file_sha(pred_path), MEMBERS)
        for name in ("row_id", "unit", "income", "occupation"):
            if not np.array_equal(saved[name], inner[name]):
                raise PermissionError("saved gate person/task identity differs")
        for task, label_name, classes, suffix in TASKS:
            f = _prob(saved[f"F_{task}"], len(rows), classes)
            for reference in REFERENCES:
                u = _prob(saved[f"{reference}_{suffix}"], len(rows), classes)
                for endpoint, vector in zip(ENDPOINTS,
                                            _oriented_costs(f, u, inner[label_name])):
                    point = float(vector.mean())
                    previous = float(receipt["margins"][task][reference][endpoint])
                    max_gate_point_delta = max(max_gate_point_delta, abs(point - previous))
                    if abs(point - previous) > 1e-12:
                        raise PermissionError("independent point cost differs from F gate")
                    order.append({"seed": seed, "task": task,
                                  "reference": reference, "endpoint": endpoint})
                    vectors.append(vector)
    matrix = np.column_stack(vectors)
    if matrix.shape != (2235, 24):
        raise PermissionError("complete interval cost family differs")
    point = matrix.mean(axis=0)
    boot, draw_hash, count = group_draw_means(matrix, groups)
    cap = np.array([CAPS[identity["endpoint"]] for identity in order])
    calculated = one_sided_bounds(point, boot, cap)
    if (draw_hash != original["sampled_group_index_sha256"]
            or count != original["groups"]
            or len(original["clauses"]) != 24
            or calculated["nonzero_sd_costs"] != original["nonzero_sd_costs"]):
        raise PermissionError("R1 bootstrap draw/family identity differs")
    max_interval_delta = 0.0
    all_statuses_match = True
    for identity, own, prior in zip(order, calculated["rows"], original["clauses"]):
        if any(prior[key] != identity[key] for key in identity):
            raise PermissionError("R1 clause order differs")
        for key in ("point", "bootstrap_sd", "lower", "upper", "cap"):
            max_interval_delta = max(max_interval_delta, abs(own[key] - prior[key]))
        all_statuses_match &= own["status"] == prior["status"]
    for key in ("upper_critical", "lower_critical"):
        max_interval_delta = max(max_interval_delta,
                                 abs(calculated[key] - original[key]))
    if max_interval_delta > 1e-10 or not all_statuses_match:
        raise PermissionError("independent interval values or statuses differ")
    counts = {status: sum(row["status"] == status for row in calculated["rows"])
              for status in ("PASS", "VIOLATION", "UNRESOLVED")}
    family = ("ALL_24_PASS" if counts["PASS"] == 24 else
              "HAS_VIOLATION" if counts["VIOLATION"] else "PRECISION_UNRESOLVED")
    if counts != original["status_counts"] or family != original["family_status"]:
        raise PermissionError("R1 family classification differs")
    return {
        "schema": "pcrl-utility-r3-independent-interval-replay-v1",
        "status": "REPLAYED", "scope": "reused Adult development INNER task role only",
        "full_input_gate_sha256": file_sha(gate_path),
        "r1_interval_output_sha256": file_sha(r1_path),
        "rows": len(rows), "groups": count, "family_size": len(order),
        "replicates": 2048, "sampled_group_index_sha256": draw_hash,
        "max_abs_point_delta_against_F_gate": max_gate_point_delta,
        "max_abs_interval_value_delta_against_R1": max_interval_delta,
        "all_clause_statuses_match": all_statuses_match,
        "status_counts": counts, "family_status": family,
        "upper_critical": calculated["upper_critical"],
        "lower_critical": calculated["lower_critical"],
        "clauses": [{**identity, **row} for identity, row in zip(order, calculated["rows"])],
        "interpretation_limit": "conditional descriptive intervals, not post-selection confirmation",
    }


def _write_once(path: Path, result: dict) -> None:
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
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
    parser.add_argument("--r1-interval-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r3_utility_interval_replay", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("independent interval replay needs one numerical thread")
    root = args.private_root.resolve(strict=True)
    if (root != Path(__file__).resolve().parents[1] / ".private"
            or args.output.resolve() != root / "r3/UTILITY_INTERVAL_REPLAY.json"):
        raise PermissionError("independent interval replay private custody differs")
    if (_require(spec, args.manifest) !=
            json.loads(args.lock.read_text()).get("source_map_sha256")):
        raise PermissionError("role-only manifest differs from lock")
    manifest = json.loads(args.manifest.read_text())
    result = replay(spec, root, args.export_root.resolve(strict=True), manifest,
                    args.full_input_gate, args.r1_interval_output,
                    json.loads(args.lock.read_text()))
    _write_once(args.output, result)


if __name__ == "__main__":
    main()
