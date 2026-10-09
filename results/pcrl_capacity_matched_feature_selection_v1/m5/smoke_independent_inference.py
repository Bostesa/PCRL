"""Role-sized synthetic timing for M5's independent 207-clause bootstrap.

All values are generated in memory from a fixed seed. No role store, prior
artifact, or Adult label is imported or opened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from results.pcrl_capacity_matched_feature_selection_v1.m5.independent_inference import (
    LOCAL_IDS, RULES, infer_from_saved_values,
)


def _fixture() -> tuple[dict, dict]:
    rng = np.random.default_rng(73012)
    n = 2235
    truth = {
        "truth__row_id": np.arange(n, dtype=np.int64),
        "truth__unit": np.r_[np.arange(n - 1, dtype=np.int64), n - 2],
        "truth__income": rng.integers(0, 2, n, dtype=np.int64),
        "truth__occupation": rng.integers(0, 6, n, dtype=np.int64),
        "truth__sex": np.arange(n, dtype=np.int64) % 2,
    }
    refs = {}
    for task, width in (("income", 2), ("occupation_group", 6)):
        for ref in ("U0", "Ucal", "F"):
            raw = rng.random((n, width), dtype=np.float64) + .1
            refs[f"reference__{task}__{ref}"] = raw / raw.sum(axis=1, keepdims=True)
    all_values, plans = {}, {}
    for seed in (0, 1):
        all_values[seed] = {}
        plans[seed] = {"schema": "pcrl-focused-m3-control-plan-v1",
                       "rules": {}}
        for ordinal, rule in enumerate(RULES):
            pair_id = f"SYN_PAIR_{ordinal}"
            plans[seed]["rules"][rule] = {"status": "SELECTED", "pair_id": pair_id}
            arrays = {**truth, **refs}
            for task, width in (("income", 2), ("occupation_group", 6)):
                raw = rng.random((n, width), dtype=np.float64) + .1
                arrays[f"candidate__{task}"] = raw / raw.sum(axis=1, keepdims=True)
            for view in ("income", "occupation_group", "coalition"):
                readers = LOCAL_IDS if view != "coalition" else LOCAL_IDS + tuple(
                    alias + reader for alias in ("IGNORE_R1__", "IGNORE_R2__")
                    for reader in LOCAL_IDS)
                for reader in readers:
                    arrays[f"reader__{view}__{reader}"] = rng.random(n, dtype=np.float64)
            all_values[seed][pair_id] = arrays
    return all_values, plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    values, plans = _fixture()
    result = infer_from_saved_values(values, plans, bytes(range(32)))
    if (result["status"] != "DEVELOPMENT_SIMULTANEOUS_INTERVALS"
            or result["registered_clause_count"] != 207
            or result["active_clause_count"] != 207):
        raise RuntimeError("M5 role-sized synthetic interval family is incomplete")
    payload = {
        "schema": "pcrl-focused-m5-inference-smoke-v1", "status": "PASS",
        "rows": 2235, "groups": 2234, "seeds": 2, "rules": 10,
        "clauses": 207, "replicates": 2048,
        "elapsed_monotonic_seconds": time.monotonic() - start,
        "critical_95": result["critical_95"],
        "clause_status_counts": {name: sum(row["status"] == name
                                      for row in result["clauses"])
                                 for name in ("PASS", "VIOLATION", "UNRESOLVED")},
        "result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with target.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o600)
    print(json.dumps({"status": "PASS", "output_sha256": hashlib.sha256(
        target.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
