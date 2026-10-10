"""Independent paired-group replay of the fixed 360 utility interval slots.

This uses only frozen per-person cost vectors. It imports no R3 interval,
endpoint, candidate or selection code, and runs under the shared queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.stats import t as student_t


ARMS = ("B0", "T", "D", "P", "L", "J", "S12", "S21", "W_Q", "P_STAR")
TASKS = ("income", "occupation")
REFERENCES = ("U0", "Ucal")
METRICS = ("accuracy", "nll", "brier")
SOURCE_NAMES = ("candidate", "B0", "U0", "Ucal")
FIELDS = {"row_id", "unit"} | {
    f"{source}_{task}_{metric}"
    for source in SOURCE_NAMES for task in TASKS for metric in METRICS}
NULL = {"point": None, "se": None, "lower": None, "upper": None,
        "status": "NOT_ESTIMABLE_DEPENDENCY"}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _status(metric: str, lower: float, upper: float) -> str:
    if metric == "accuracy":
        return "PASS" if lower >= -.010 else "VIOLATION" if upper < -.010 else "UNRESOLVED"
    cap = .010 if metric == "nll" else .005
    return "PASS" if upper <= cap else "VIOLATION" if lower > cap else "UNRESOLVED"


def independent_utility_intervals(records: dict[tuple[str, int], dict], *,
                                  expected_groups: int = 2234,
                                  bootstrap_draws: int = 2048) -> dict[str, dict]:
    names = [f"{summary}:utility:{arm}:{task}:{reference}:{metric}"
             for summary in ("s0", "s1", "mean")
             for arm in ARMS for task in TASKS
             for reference in REFERENCES for metric in METRICS]
    result = {name: dict(NULL) for name in names}
    if not records:
        return result
    if (expected_groups < 2 or bootstrap_draws < 2 or
            any(arm not in ARMS or seed not in (0, 1) for arm, seed in records)):
        raise ValueError("independent utility family or resample size differs")
    first = next(iter(records.values()))
    if set(first) != FIELDS:
        raise ValueError("private utility vector member inventory differs")
    row = np.asarray(first["row_id"])
    unit = np.asarray(first["unit"])
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1 or
            unit.shape != row.shape or len(row) == 0 or
            not np.all(np.diff(row) > 0) or
            len(np.unique(unit)) != expected_groups):
        raise ValueError("paired people/groups differ")
    for item in records.values():
        if (set(item) != FIELDS or not np.array_equal(item["row_id"], row)
                or not np.array_equal(item["unit"], unit)):
            raise ValueError("cross-arm paired people/groups differ")
        for field in FIELDS - {"row_id", "unit"}:
            arr = np.asarray(item[field])
            if arr.dtype != np.float64 or arr.shape != row.shape or not np.isfinite(arr).all():
                raise ValueError("per-person task cost vector differs")
    groups, inverse = np.unique(unit, return_inverse=True)
    group_count = np.bincount(inverse).astype(np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(20261010))
    sampled = rng.integers(0, len(groups), size=(bootstrap_draws, len(groups)), dtype=np.int64)
    weights = np.empty((bootstrap_draws, len(groups)), dtype=np.int64)
    for index, draw in enumerate(sampled):
        weights[index] = np.bincount(draw, minlength=len(groups))
    denominator = weights @ group_count
    critical = float(student_t.ppf(1 - .05 / (2 * 1575), expected_groups - 1))
    seedwise = {}
    for (arm, seed), item in records.items():
        for task in TASKS:
            for reference in REFERENCES:
                for metric in METRICS:
                    delta = (np.asarray(item[f"candidate_{task}_{metric}"])
                             - np.asarray(item[f"{reference}_{task}_{metric}"]))
                    grouped = np.bincount(inverse, weights=delta, minlength=len(groups))
                    replicate = (weights @ grouped) / denominator
                    point = float(np.mean(delta))
                    se = (0.0 if np.max(np.abs(replicate - point)) <= 1e-12
                          else float(np.std(replicate, ddof=1)))
                    lower, upper = point - critical * se, point + critical * se
                    base = f"utility:{arm}:{task}:{reference}:{metric}"
                    result[f"s{seed}:{base}"] = {
                        "point": point, "se": se, "lower": lower, "upper": upper,
                        "status": _status(metric, lower, upper)}
                    seedwise.setdefault(base, {})[seed] = (point, replicate)
    for base, paired in seedwise.items():
        if set(paired) != {0, 1}:
            continue
        point = (paired[0][0] + paired[1][0]) / 2
        replicate = (paired[0][1] + paired[1][1]) / 2
        se = (0.0 if np.max(np.abs(replicate - point)) <= 1e-12
              else float(np.std(replicate, ddof=1)))
        lower, upper = point - critical * se, point + critical * se
        metric = base.rsplit(":", 1)[-1]
        result[f"mean:{base}"] = {
            "point": point, "se": se, "lower": lower, "upper": upper,
            "status": _status(metric, lower, upper)}
    return result


def _archive(path: Path) -> dict[str, np.ndarray]:
    with ZipFile(path) as source:
        names = {item.filename for item in source.infolist()}
        if len(names) != len(source.infolist()) or names != {key + ".npy" for key in FIELDS}:
            raise PermissionError("private R3 utility cost member inventory differs")
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in FIELDS}


def replay(config_path: Path, config_sha: str, result_path: Path,
           result_sha: str, receipt_path: Path, receipt_sha: str) -> dict:
    for path, digest in ((config_path, config_sha), (result_path, result_sha),
                         (receipt_path, receipt_sha)):
        if sha(path) != digest:
            raise PermissionError("R3 interval config/result/receipt source changed")
    config = json.loads(config_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    source = json.loads(result_path.read_text())
    if (config.get("schema") != "pcrl-bank-r3-interval-job-v1" or
            receipt.get("schema") != "pcrl-bank-r3-utility-interval-receipt-v1" or
            source.get("schema") != "pcrl-bank-r3-utility-intervals-v1" or
            receipt.get("unit_id") != config.get("unit_id") or
            receipt.get("result_sha256") != result_sha or
            receipt.get("source_seal_sha256") != config.get("seal", {}).get("sha256") or
            source.get("source_seal_sha256") != receipt.get("source_seal_sha256") or
            source.get("fixed_family_K") != 1575 or
            source.get("utility_slots") != 360 or
            source.get("group_count") != 2234 or
            source.get("bootstrap_draws") != 2048 or
            source.get("bootstrap_generator") != "PCG64DXSM" or
            source.get("bootstrap_seed") != 20261010):
        raise PermissionError("R3 fixed-family interval identity differs")
    records = {}
    for entry in config["entries"]:
        if entry["arm"] not in ARMS or entry["seed"] not in (0, 1):
            raise PermissionError("interval arm/seed outside registered family")
        vector = Path(entry["vector"]["path"])
        receipt_source = Path(entry["receipt"]["path"])
        if (".private" not in vector.parts or
                sha(vector) != entry["vector"]["sha256"] or
                sha(receipt_source) != entry["receipt"]["sha256"]):
            raise PermissionError("R3 private vector/role receipt changed")
        utility = json.loads(receipt_source.read_text())
        if (utility.get("role") != "INNER_SELECTION" or
                utility.get("arm") != entry["arm"] or
                utility.get("seed") != entry["seed"] or
                utility.get("private_vector_sha256") != entry["vector"]["sha256"]):
            raise PermissionError("INNER utility vector source differs")
        key = entry["arm"], entry["seed"]
        if key in records:
            raise PermissionError("duplicate arm/seed interval vector")
        records[key] = _archive(vector)
    replayed = independent_utility_intervals(records)
    if set(replayed) != set(source["intervals"]):
        raise ValueError("fixed 360 utility interval slots differ")
    max_gap = 0.0
    counts = {}
    for name, item in replayed.items():
        reported = source["intervals"][name]
        if reported["status"] != item["status"]:
            raise ValueError("independent utility interval status differs")
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        for field in ("point", "se", "lower", "upper"):
            if item[field] is None:
                if reported[field] is not None:
                    raise ValueError("missing utility interval slot was estimated")
            else:
                gap = abs(float(reported[field]) - item[field])
                max_gap = max(max_gap, gap)
                if gap > 1e-10:
                    raise ValueError("independent utility interval value differs")
    if counts != source["status_counts"] or receipt["status_counts"] != counts:
        raise ValueError("fixed family utility status counts differ")
    return {"schema": "BAR1-r5-independent-utility-interval-replay-v1",
            "status": "PASS", "r3_receipt_sha256": receipt_sha,
            "r3_result_sha256": result_sha, "source_entries": len(records),
            "verified_slots": len(replayed), "status_counts": counts,
            "max_absolute_gap": max_gap,
            "scope": "conditional development paired-group intervals only"}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("config", "config-sha256", "results", "results-sha256",
                 "r3-receipt", "r3-receipt-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
    active = assert_active_from_env(stage="verification")
    result = replay(Path(args.config), args.config_sha256,
                    Path(args.results), args.results_sha256,
                    Path(args.r3_receipt), args.r3_receipt_sha256)
    result["unit_id"] = active["unit_id"]
    output = Path(args.output).resolve(strict=False)
    expected = Path(__file__).resolve().parents[1] / "results/pcrl_bank_anchored_release_v1/r5"
    if output.parent != expected or output.exists():
        raise PermissionError("independent interval receipt output path differs")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
