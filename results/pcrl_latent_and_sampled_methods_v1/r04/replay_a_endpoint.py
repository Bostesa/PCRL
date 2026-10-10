"""R03-only independent replay of all saved A point cells and K=6,885 bounds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit_a_family_s1 import independent_ids
from .independent_a_intervals import (NAMES, STOCHASTIC, _Release,
                                      independent_a_intervals)
from .independent_a_source import bound_json, bound_npz, sha_array, sha_file
from .replay_a_prefit import queue_guard


VIEWS = ("r1", "r2", "pair")
TASKS = ("income", "occupation")
METRICS = ("accuracy", "nll", "brier")
FINITE_READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                  "CELL_A1", "CODE_BIT_LR_C1", "PROB_LR_C1",
                  "PROB_HGB_15x200", "Q_POST_LR_C1")
CONTINUOUS_READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                      "PROB_LR_C1", "PROB_HGB_15x200")
TOLERANCE = 1e-8


def _truth(manifest_path: Path, manifest_sha: str) -> dict:
    manifest = bound_json(manifest_path, manifest_sha)
    truth = {}
    for field in (*TASKS, "sex"):
        key = f"INNER_SELECTION:{field}"
        path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
        record = bound_npz(path, manifest["role_file_sha256"][key],
                           {"row_id", "unit", field})
        if truth and (not np.array_equal(record["row_id"], truth["row_id"])
                      or not np.array_equal(record["unit"], truth["unit"])):
            raise ValueError("A endpoint INNER task/SEX ordered people differ")
        truth.setdefault("row_id", record["row_id"])
        truth.setdefault("unit", record["unit"])
        truth[field] = record[field]
    if (len(truth["row_id"]) != 2235 or len(np.unique(truth["unit"])) != 2234
            or not np.isin(truth["sex"], (0, 1)).all()):
        raise ValueError("A endpoint INNER source membership differs")
    return truth


def _reader_ids(arm: str, view: str) -> tuple[str, ...]:
    core = CONTINUOUS_READERS if arm in ("U0", "Ucal") else FINITE_READERS
    return core if view != "pair" else (
        *core, *(f"IGNORE_R2__{name}" for name in core),
        *(f"IGNORE_R1__{name}" for name in core))


def _cell(path: Path, record: dict, truth: dict, arm: str) -> dict:
    if (".private" not in path.parts or sha_file(path) != record["sha256"]
            or sorted(record["npz_keys"]) != record["npz_keys"]):
        raise ValueError("A normalized private cell custody differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(record["npz_keys"]):
            raise ValueError("A normalized cell member set differs")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    row, group = arrays["row_id"], arrays["unit"]
    if not np.array_equal(row, truth["row_id"]) or not np.array_equal(group, truth["unit"]):
        raise ValueError("A normalized cell ordered INNER people differ")
    if (row.dtype != np.int64 or group.dtype != np.int64 or row.ndim != 1
            or group.shape != row.shape):
        raise ValueError("A normalized cell row/group representation differs")
    for recipient, classes in ((1, 2), (2, 6)):
        q = arrays[f"prob_{recipient}"]
        if (q.dtype != np.float64 or q.shape != (len(row), classes)
                or not np.isfinite(q).all() or np.any((q < 0) | (q > 1))
                or not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-8)):
            raise ValueError("A normalized sampled task probability invalid")
    result = {key: arrays[key] for key in ("row_id", "unit", "prob_1", "prob_2")}
    readers = {}
    for view in VIEWS:
        expected = _reader_ids(arm, view)
        if tuple(record["reader_ids"][view]) != expected:
            raise ValueError("A normalized fixed reader slate/aliases differ")
        readers[view] = {}
        for reader in expected:
            key = f"reader__{view}__{reader}"
            values = arrays.pop(key)
            if (values.dtype != np.float64 or values.shape != row.shape
                    or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
                raise ValueError("A normalized fixed reader vector invalid")
            readers[view][reader] = values
    if set(arrays) != {"row_id", "unit", "prob_1", "prob_2"}:
        raise ValueError("A normalized cell has unregistered arrays")
    result["reader_predictions"] = readers
    return result


def _point_table(records: dict) -> dict[tuple[str, ...], tuple[float, float | None]]:
    table = {}
    for seed in (0, 1):
        truth = records[seed]["truth"]
        n = len(truth["row_id"])
        weight = np.ones(n, np.float64)
        for arm in NAMES:
            values = []
            for draw, raw in sorted(records[seed]["releases"][arm].items()):
                score = _Release(raw, truth, truth["sex"]).score(weight, float(n))
                values.append(score)
                for task in TASKS:
                    for metric in METRICS:
                        table[(str(seed), arm, str(draw), task, metric)] = (
                            float(score["task"][task][metric]), None)
                for view in VIEWS:
                    table[(str(seed), arm, str(draw), view, "auc_envelope")] = (
                        float(score["auc"][view]), None)
            for scope, metric in ((*[(t, m) for t in TASKS for m in METRICS],
                                   *[(v, "auc_envelope") for v in VIEWS])):
                seq = [v["task"][scope][metric] if metric != "auc_envelope"
                       else v["auc"][scope] for v in values]
                table[(str(seed), arm, "mean", scope, metric)] = (
                    float(np.mean(seq)), float(np.std(seq, ddof=1)) if len(seq) > 1 else 0.0)
    return table


def _compare_csv(path: Path, expected: dict) -> float:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["seed", "arm", "draw", "scope", "metric", "point", "draw_sd"]:
            raise ValueError("A endpoint point CSV columns differ")
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError("A endpoint point CSV count differs")
    largest, seen = 0.0, set()
    for row in rows:
        key = tuple(row[name] for name in ("seed", "arm", "draw", "scope", "metric"))
        if key not in expected or key in seen:
            raise ValueError("A endpoint point CSV identity/duplicate differs")
        seen.add(key)
        actual, sd = expected[key]
        difference = abs(float(row["point"]) - actual)
        if sd is None:
            if row["draw_sd"] not in ("", "None"):
                raise ValueError("A endpoint fixed-draw SD unexpectedly present")
        else:
            difference = max(difference, abs(float(row["draw_sd"]) - sd))
        largest = max(largest, difference)
    if largest > TOLERANCE:
        raise ValueError("A endpoint point CSV differs from actual sampled outputs")
    return largest


def run(args: argparse.Namespace) -> dict:
    lock = Path(args.original_lock).resolve(strict=True)
    queue_guard(lock, args.original_lock_sha256)
    amendment = bound_json(Path(args.a_amendment), args.a_amendment_sha256)
    analysis = bound_json(Path(args.analysis_amendment), args.analysis_amendment_sha256)
    if (amendment.get("status") != "LOCALLY_SEALED_A_STAGE"
            or analysis.get("status") != "LOCALLY_SEALED_A_ENDPOINT"
            or analysis.get("producer_amendment_sha256") != args.a_amendment_sha256):
        raise PermissionError("A endpoint replay requires effective producer and analysis seals")
    campaign = Path(args.campaign).resolve(strict=True)
    manifest_path = campaign / "r02/A_FAMILY_ORDER_S1.json"
    if sha_file(manifest_path) != amendment["interval_family"]["manifest_sha256"]:
        raise ValueError("A K6885 ordered family differs")
    public_receipt = bound_json(Path(args.endpoint_receipt), args.endpoint_receipt_sha256)
    if (public_receipt.get("status") != "COMPLETE" or
            public_receipt.get("family") != "A_10_ARMS_7_REFS_9_SUMMARIES_S1" or
            public_receipt.get("clauses") != 6885 or
            public_receipt.get("attack_instances") != 58 or
            public_receipt.get("bootstrap_replicates") != 2048 or
            public_receipt.get("source_manifest_sha256") != sha_file(manifest_path)):
        raise ValueError("A public endpoint receipt differs")
    root = Path(args.endpoint_private_root).resolve(strict=True)
    binding = bound_json(root / "A_SOURCE_BINDING.json",
                         public_receipt["private_binding_sha256"])
    if (binding.get("status") != "ALL_58_COMPLETE_VIEWS_VERIFIED" or
            binding.get("a_amendment_sha256") != args.a_amendment_sha256 or
            binding.get("analysis_amendment_sha256") != args.analysis_amendment_sha256 or
            binding.get("role_manifest_sha256") != args.role_manifest_sha256 or
            len(binding.get("cells", [])) != 58):
        raise ValueError("A private normalized source binding differs")
    truth = _truth(Path(args.role_manifest).resolve(strict=True),
                   args.role_manifest_sha256)
    if sha_array(np.column_stack((truth["row_id"], truth["unit"]))) != \
            binding["ordered_inner_row_group_sha256"]:
        raise ValueError("A private normalized source ordered group hash differs")
    records = {seed: {"truth": truth, "releases": {}} for seed in (0, 1)}
    source_sha = []
    seen = set()
    for entry in binding["cells"]:
        arm, seed, draw = entry["arm"], entry["seed"], entry["draw"]
        key = (arm, seed, draw)
        wanted = {0, 1, 2} if arm in STOCHASTIC else {0}
        if arm not in NAMES or seed not in (0, 1) or draw not in wanted or key in seen:
            raise ValueError("A private normalized source cell identity differs")
        seen.add(key)
        raw = entry["normalized_private_cell"]
        expected_path = root / "cells" / f"{arm}_s{seed}_d{draw}.npz"
        if Path(raw["private_file"]).resolve(strict=True) != expected_path:
            raise ValueError("A private normalized cell path differs")
        records[seed]["releases"].setdefault(arm, {})[draw] = _cell(
            expected_path, raw, truth, arm)
        source_sha.append(raw["sha256"])
        attack_root = Path(entry["private_attack_root"]).resolve(strict=True)
        if (sha_file(attack_root / "ATTACK_ATTEMPT_INDEX.json") !=
                entry["attack_index_sha256"] or
                sha_file(attack_root / "common/FROZEN_COMPLETE.json") !=
                entry["reader_freeze_sha256"]):
            raise ValueError("A normalized source attack index/freeze changed")
        for view in VIEWS:
            if (sha_file(attack_root / "common" / view / "readers/READERS_RECEIPT.json") !=
                    entry["reader_receipt_sha256"][view] or
                    sha_file(attack_root / "common" / view / "controls/CUSTODY_RECEIPT.json") !=
                    entry["v2_control_receipt_sha256"][view]):
                raise ValueError("A normalized source reader/V2 receipt changed")
    expected = {(arm, seed, draw) for arm in NAMES for seed in (0, 1)
                for draw in ((0, 1, 2) if arm in STOCHASTIC else (0,))}
    if seen != expected:
        raise ValueError("A private normalized source lacks a sealed attack cell")
    public_json = Path(args.endpoint_json).resolve(strict=True)
    public_csv = Path(args.endpoint_csv).resolve(strict=True)
    if (sha_file(public_json) != public_receipt["interval_json_sha256"] or
            sha_file(public_csv) != public_receipt["point_csv_sha256"]):
        raise ValueError("A public endpoint output bytes changed")
    point_table = _point_table(records)
    if public_receipt.get("point_rows") != len(point_table):
        raise ValueError("A endpoint public point row count differs")
    point_difference = _compare_csv(public_csv, point_table)
    independent = independent_a_intervals(records)
    published = json.loads(public_json.read_text())
    if (independent["clause_count"] != 6885 or
            published.get("clauses") is None or
            len(published["clauses"]) != 6885 or
            tuple(clause["name"] for clause in published["clauses"]) != independent_ids() or
            independent["missing_clause_count"] != published["missing_clause_count"] or
            independent["missing_clause_count"] != 0 or
            independent["critical_family"] != published["critical_family"]):
        raise ValueError("A complete K6885 endpoint family differs")
    max_difference = {name: 0.0 for name in ("point_margin", "se", "lower", "upper")}
    for own, other in zip(independent["clauses"], published["clauses"]):
        if (own["name"] != other["name"] or own["status"] != other["status"] or
                own["numerical_se_status"] != other["numerical_se_status"]):
            raise ValueError("A endpoint clause identity/status differs")
        for name in max_difference:
            difference = abs(own[name] - other[name])
            max_difference[name] = max(max_difference[name], difference)
            if difference > TOLERANCE:
                raise ValueError(f"A endpoint {name} differs at {own['name']}")
    critical_difference = abs(independent["critical_95"] - published["critical_95"])
    if critical_difference > TOLERANCE:
        raise ValueError("A endpoint simultaneous critical differs")
    return {"schema": "pcrl-r04-a-endpoint-independent-replay-v1",
            "status": "A_58_CELL_6885_INTERVAL_REPLAY_PASS",
            "a_amendment_sha256": args.a_amendment_sha256,
            "analysis_amendment_sha256": args.analysis_amendment_sha256,
            "endpoint_receipt_sha256": args.endpoint_receipt_sha256,
            "private_binding_sha256": public_receipt["private_binding_sha256"],
            "normalized_cell_count": len(source_sha),
            "clause_count": 6885, "point_csv_rows": public_receipt["point_rows"],
            "maximum_point_csv_difference": point_difference,
            "maximum_clause_differences": max_difference,
            "critical_95_difference": critical_difference,
            "group_bootstrap_replicates": 2048,
            "scope": "actual saved INNER q/readers and fixed two-seed three-draw family"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "original-lock-sha256", "a-amendment",
                 "a-amendment-sha256", "analysis-amendment",
                 "analysis-amendment-sha256", "campaign", "endpoint-private-root",
                 "endpoint-receipt", "endpoint-receipt-sha256", "endpoint-json",
                 "endpoint-csv", "role-manifest", "role-manifest-sha256",
                 "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("independent A endpoint replay receipt exists")
    result = run(args)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
